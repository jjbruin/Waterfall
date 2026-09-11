"""Import a partner's monthly budget spreadsheet into `isbs_budget_is_supplements`.

WHY THIS EXISTS. MRI is the source for `isbs_budget_is`, but a budget is not loaded into
MRI until it has been analysed and approved. The valuation team does that analysis HERE —
questioning the partner's line items against the appraiser's starting points — so between
receiving a budget and approving it, THE APP HOLDS THE ONLY COPY. The supplement table is
where it lives, and `data_service._append_isbs_supplements` folds it into `isbs_raw` as
`vSource = 'Budget IS'`, which is what the valuation comparison already reads.

Where MRI later carries the same (vcode, dtEntry, vSource, vAccount), THE APP'S ROW WINS —
see the precedence block in data_service. The app is where the work was done.

THE SHAPE OF THE PROBLEM, and why the rules below are what they are:

  * A partner's spreadsheet uses THEIR line names, not our chart of accounts, so every
    line needs assigning. Our COA has 169 accounts, which is unusable as a flat list —
    `account_choices()` narrows it to the accounts that deal actually used in the last
    12 months (23 for Asbury Commons, 52 for Berger Pittsburgh), with descriptions.

  * MRI stores revenue NEGATIVE and expenses POSITIVE. A partner's file almost always
    does the opposite, and getting it wrong silently inverts NOI. This is NOT blocked:
    each line carries its own flip decision, pre-set from how that account behaved in the
    deal's own prior-year actuals, and the analyst overrides it.

  * Spreadsheets contain subtotal rows, and skipping them is CORRECT. So "an unmapped
    line with a value" is not an error. Instead `reconcile()` shows total revenue, total
    expense and NOI as the spreadsheet states them against what the selected lines
    actually produce, and the analyst decides whether the difference is explained.
"""
from __future__ import annotations

import io
import logging
import re
from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import text

logger = logging.getLogger(__name__)

SUPPLEMENT_TABLE = "isbs_budget_is_supplements"

#: Flag a budgeted account outside this band of the same account's prior-year actual.
#: Wide enough for a real change in plan, tight enough to catch a units error — a budget
#: entered in thousands lands at 0.001x and is impossible to miss.
MAGNITUDE_LOW, MAGNITUDE_HIGH = 0.5, 2.0

#: Accounts that are legitimate in a budget but change what the comparison MEANS, so
#: their presence is surfaced rather than assumed. Mirrors config.py.
_BELOW_THE_LINE = {"5190", "7030", "7060", "7050"}


# ──────────────────────────────────────────────────────────────────────────────
# What the analyst may map a line to
# ──────────────────────────────────────────────────────────────────────────────

def account_choices(vcode: str, isbs_raw: pd.DataFrame,
                    as_of: Optional[date] = None) -> List[Dict[str, Any]]:
    """Accounts this deal actually used in the last 12 months of actuals.

    The full COA is 169 accounts and choosing from it is the reason line coding is slow
    and wrong. A deal's own recent history is a far better list: it is short, it is
    already the vocabulary the analyst thinks in, and it carries the SIGN the account
    behaves with, which is what pre-sets each line's flip decision.

    Returned newest-magnitude first, because the lines that matter are the big ones.
    `months` is included so a 12-month staple reads differently from a 2-month one-off.
    An empty list is a real answer — a deal with no actuals has nothing to suggest — and
    the caller should fall back to the full COA rather than blocking.
    """
    if isbs_raw is None or isbs_raw.empty:
        return []
    df = isbs_raw
    want = str(vcode).strip().lower()
    sub = df[(df["vcode"].astype(str).str.strip().str.lower() == want)
             & (df["vSource"] == "Interim IS")].copy()
    if sub.empty:
        return []

    sub["_dt"] = pd.to_datetime(sub["dtEntry"], format="mixed", errors="coerce")
    sub = sub.dropna(subset=["_dt"])
    if sub.empty:
        return []
    end = pd.Timestamp(as_of) if as_of else sub["_dt"].max()
    sub = sub[(sub["_dt"] > end - pd.DateOffset(months=12)) & (sub["_dt"] <= end)]
    if sub.empty:
        return []

    sub["_acct"] = sub["vAccount"].astype(str).str.strip()
    sub["_amt"] = pd.to_numeric(sub["mAmount"], errors="coerce").fillna(0.0)
    desc_col = "vDescription" if "vDescription" in sub.columns else None

    out: List[Dict[str, Any]] = []
    for acct, g in sub.groupby("_acct"):
        total = float(g["_amt"].sum())
        if total == 0:
            continue
        out.append({
            "account": acct,
            "description": (str(g[desc_col].dropna().iloc[0])
                            if desc_col and g[desc_col].notna().any() else ""),
            "months": int(g["_dt"].dt.to_period("M").nunique()),
            "prior_total": total,
            # The sign MRI stores this account with, for this deal. Revenue is negative
            # in MRI; a partner file that shows revenue positive needs flipping, and this
            # is what decides the default rather than a global assumption about 4xxx.
            "mri_sign": -1 if total < 0 else 1,
            "below_the_line": acct in _BELOW_THE_LINE,
        })
    out.sort(key=lambda r: abs(r["prior_total"]), reverse=True)
    return out


def category_choices(vcode: str, isbs_raw: pd.DataFrame,
                     as_of: Optional[date] = None) -> List[Dict[str, Any]]:
    """The CATEGORIES the budget comparison actually displays, each with its accounts.

    A line is mapped to a category first — `Rental Income`, `Vacancy`, `Real Estate
    Taxes` — because those are the ~20 rows `get_budget_review` renders from
    `config.IS_ACCOUNTS`, and they are the vocabulary the analyst is already reading on
    screen. Mapping to a bare account number asks them to translate in their head from a
    169-item list into a row they cannot see.

    An ACCOUNT is still required within the category, because the supplement stores
    `vAccount` and every downstream consumer — NOI, FAD, DSCR, the waterfall — reads
    individual accounts, not categories. It is defaulted to the account THIS DEAL used
    most in the last 12 months, so the common case is one click and the precision is
    kept.

    Categories the deal has actually used come first, annotated with last year's figure,
    so `Rental Income — 12mo, 5.3M` sits above one it has never touched. A deal with no
    actuals still gets the full list, just unranked: an empty suggestion list is a reason
    to show everything, not to block.
    """
    import config

    used = {c["account"]: c for c in account_choices(vcode, isbs_raw, as_of)}
    out: List[Dict[str, Any]] = []
    for section, cats in config.IS_ACCOUNTS.items():
        for cat, accts in cats.items():
            rows = []
            for a in accts:
                u = used.get(a)
                rows.append({
                    "account": a,
                    "description": u["description"] if u else "",
                    "prior_total": u["prior_total"] if u else 0.0,
                    "months": u["months"] if u else 0,
                    # Sign this account behaves with FOR THIS DEAL, which pre-sets the
                    # line's flip. Falls back to the section's convention when the deal
                    # has no history for it — revenue negative, expense positive.
                    "mri_sign": u["mri_sign"] if u else (-1 if section == "REVENUES" else 1),
                    "used": bool(u),
                })
            rows.sort(key=lambda r: abs(r["prior_total"]), reverse=True)
            cat_total = sum(r["prior_total"] for r in rows)
            out.append({
                "section": section,
                "category": cat,
                "accounts": rows,
                # The deal's own most-used account in this category; falls back to the
                # first account config lists, so a category is never un-defaultable.
                "default_account": (rows[0]["account"] if rows and rows[0]["used"]
                                    else (accts[0] if accts else None)),
                "prior_total": cat_total,
                "months": max((r["months"] for r in rows), default=0),
                "used_by_deal": any(r["used"] for r in rows),
                "below_the_line": section in ("DEBT_SERVICE", "OTHER_BTL"),
            })
    # Used categories first, then by size — the lines that matter are the big ones.
    out.sort(key=lambda c: (not c["used_by_deal"], -abs(c["prior_total"])))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Reading the spreadsheet
# ──────────────────────────────────────────────────────────────────────────────

_TOTAL_ROW = re.compile(
    r"^\s*(total\s+)?(revenue|income|expenses?|operating expenses?|"
    r"net operating income|noi|egi|effective gross)\b", re.I)


def parse_budget_workbook(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Every labelled row x every month column, plus any totals the sheet states.

    Reuses `cashflow_parser._detect_horizontal_dates`, which already solves the hard
    part — finding which row holds the dates and which column holds the labels — and is
    the same detector the Argus and partner-cashflow imports rely on.

    Unlike those, NO row is classified here. A budget's lines are the partner's own
    vocabulary and the analyst assigns each one; guessing would produce exactly the COA
    mapping problem this import exists to make visible. Rows that LOOK like totals are
    flagged `looks_like_total` so the UI can leave them unmapped by default, but they
    are still returned — a sheet whose "Total Revenue" is the only revenue line is a real
    thing, and refusing to show it would be worse than showing it unticked.
    """
    import openpyxl
    from cashflow_parser import _detect_horizontal_dates

    det = _detect_horizontal_dates(file_bytes)
    if not det:
        raise ValueError(
            f"Could not find a row of month columns in '{filename}'. A budget is "
            f"expected with months across the top and line items down the side, and "
            f"the detector needs at least four month columns to be confident.")

    date_row, label_col = det["date_row_idx"], det["label_col_idx"]

    # `dates` is a FLAT list aligned to the row's cells FROM COLUMN 1 — the detector
    # skips cell 0, which is the row's own label ("For the Months"). So entry i of that
    # list describes sheet column i+1, and non-date cells are None. Reading it as
    # (col, date) pairs silently mis-aligns every amount by one column.
    periods: List[Dict[str, Any]] = []
    for i, dt in enumerate(det["dates"]):
        if dt is None:
            continue
        ts = pd.Timestamp(dt)
        periods.append({"col": i + 1,
                        "period": (ts + pd.offsets.MonthEnd(0)).date().isoformat()})
    if not periods:
        raise ValueError(f"No month columns resolved in '{filename}'.")

    # The detector returns only the rows ABOVE the date row, so the body is read here.
    wb = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True, read_only=True)
    ws = wb[det.get("sheet_name") or wb.sheetnames[0]]
    body = [list(r) for r in ws.iter_rows(values_only=True)]
    wb.close()

    lines: List[Dict[str, Any]] = []
    for r in range(date_row + 1, len(body)):
        row_vals = body[r]
        label = row_vals[label_col] if label_col < len(row_vals) else None
        if label is None or not str(label).strip():
            continue
        label = str(label).strip()
        amounts: Dict[str, float] = {}
        for p in periods:
            if p["col"] >= len(row_vals):
                continue
            v = _to_number(row_vals[p["col"]])
            if v is not None:
                amounts[p["period"]] = v
        if not amounts:
            continue
        lines.append({
            "row": r,
            "label": label,
            "amounts": amounts,
            "total": round(sum(amounts.values()), 2),
            "months": len(amounts),
            "looks_like_total": bool(_TOTAL_ROW.match(label)),
        })

    return {
        "filename": filename,
        "sheet": det.get("sheet_name"),
        "periods": [p["period"] for p in periods],
        "lines": lines,
        "stated_totals": _stated_totals(lines),
    }


def _to_number(v) -> Optional[float]:
    """Spreadsheet cell to float. Handles $, commas and (parenthetical negatives)."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()").replace("$", "").replace(",", "").strip()
    try:
        n = float(s)
    except ValueError:
        return None
    return -n if neg else n


def _stated_totals(lines: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Revenue / expense / NOI totals AS THE SPREADSHEET STATES THEM, if it states any.

    These are what the reconciliation compares against. A sheet with no stated totals is
    normal — the comparison then shows only the computed side.
    """
    def find(pattern: str) -> Optional[float]:
        rx = re.compile(pattern, re.I)
        hits = [l for l in lines if rx.search(l["label"])]
        return hits[-1]["total"] if hits else None   # the last match is the grand total

    return {
        "revenue": find(r"^\s*total\s+(revenue|income)"),
        "expense": find(r"^\s*total\s+(operating\s+)?expenses?"),
        "noi": find(r"net operating income|^\s*noi\b"),
    }
