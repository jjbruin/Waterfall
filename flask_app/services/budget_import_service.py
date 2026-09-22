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

#: Categories whose account list in `config.IS_ACCOUNTS` is not what a BUDGET should map
#: to. Only one so far: `DEBT_SERVICE.Principal` is deliberately `[]` there, because for
#: ACTUALS principal is derived from the balance-sheet balance change rather than read
#: from an account — config says so in its own comment: "Computed separately: BS balance
#: change for Actuals, 7060 for Budget". A budget has no balance sheet, so 7060 IS the
#: budget's representation of principal, and without this the screen would offer a
#: Principal category with no accounts in it: impossible to complete, and whatever the
#: analyst picked would be refused as not-in-category.
#:
#: Overriding here rather than in `config.IS_ACCOUNTS` on purpose — adding 7060 there
#: would change what the ACTUALS column displays for every deal, to fix an import screen.
_CATEGORY_ACCOUNTS_FOR_BUDGET = {"Principal": ["7060"]}


def category_accounts() -> Dict[str, List[str]]:
    """{category: [accounts]} as the IMPORT should see it — config, plus the overrides
    above. One definition, so the dropdown and the not-in-category check cannot drift."""
    import config
    out: Dict[str, List[str]] = {}
    for cats in config.IS_ACCOUNTS.values():
        for cat, accts in cats.items():
            out[cat] = list(_CATEGORY_ACCOUNTS_FOR_BUDGET.get(cat, accts))
    return out


def category_for_account(account) -> Optional[str]:
    """The ONE category an account belongs to, or None if we do not carry it.

    Jack, Sep 22 2026: "Auto-map should take the account number off the upload, go
    to our global mapping, and match it. Account 4090 comes in ... the app looks up
    to the global mapping, says 4090 is CAM, so it maps to CAM. Done." And: "We want
    one source of mapping, and that's the account number. The category dropdown
    should come out entirely and just display whatever the account dictates."

    It CAN be one source: measured against `category_accounts()`, all 80 accounts
    belong to exactly one category, so a category chosen separately could only ever
    agree with the account or contradict it. Contradicting it was blocking, which is
    the "two separate steps and they fight each other" he describes -- the analyst
    picked the account, the category stayed on something else, and submit refused.
    """
    if account in (None, ""):
        return None
    acct = str(account).strip()
    for cat, accts in category_accounts().items():
        if acct in {str(a).strip() for a in accts}:
            return cat
    return None


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


# The statement each section belongs to, and where the subtotals fall. IS_ACCOUNTS is
# already in statement order; this names what that order MEANS, which is the part asset
# management said was missing: "seeing it in statement order would tell us which
# accounts are income, opex, below the line expense, capex".
_COA_SECTIONS = [
    ("REVENUES", "Revenue", "Income — what the property bills"),
    ("EXPENSES", "Operating expenses", "Deducted from revenue to reach NOI"),
    ("DEBT_SERVICE", "Debt service", "Below NOI — interest and principal"),
    ("OTHER_BTL", "Other below the line", "Outside NOI by policy"),
]
_COA_SUBTOTAL_AFTER = {"EXPENSES": "Net operating income (NOI)"}


def chart_of_accounts(vcode: Optional[str] = None,
                      isbs_raw: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Peaceable's chart of accounts, laid out the way a statement reads.

    Answers "which accounts are income, opex, below the line, capex" without anyone
    having to open config.py. When a deal is named, each account also carries whether
    that deal has used it recently, so a mapper can tell a live account from a dormant
    one.
    """
    import config

    used = {}
    if vcode is not None and isbs_raw is not None:
        try:
            used = {c["account"]: c for c in account_choices(vcode, isbs_raw)}
        except Exception as e:                       # a deal with no history is fine
            logger.info("No account history for %s: %s", vcode, e)

    overrides = category_accounts()
    sections = []
    seen = set()
    for key, title, note in _COA_SECTIONS:
        cats = []
        for cat, config_accts in config.IS_ACCOUNTS.get(key, {}).items():
            accts = overrides.get(cat, config_accts)
            rows = []
            for a in accts:
                a = str(a)
                seen.add(a)
                u = used.get(a)
                rows.append({"account": a,
                             "description": (u or {}).get("description", ""),
                             "used_by_deal": bool(u),
                             "prior_total": (u or {}).get("prior_total")})
            cats.append({"category": cat, "accounts": rows})
        sections.append({"key": key, "title": title, "note": note,
                         "categories": cats,
                         "subtotal_after": _COA_SUBTOTAL_AFTER.get(key)})

    # CapEx is not in IS_ACCOUNTS because it is not an income-statement line, but it is
    # exactly one of the buckets asset management asked to be able to see.
    capex = [{"account": str(a), "description": (used.get(str(a)) or {}).get("description", ""),
              "used_by_deal": str(a) in used, "prior_total": (used.get(str(a)) or {}).get("prior_total")}
             for a in sorted(config.CAPEX_ACCTS)]
    if capex:
        sections.append({"key": "CAPEX", "title": "Capital expenditure",
                         "note": "Not an income statement line; funded from reserves",
                         "categories": [{"category": "Capital Expenditure", "accounts": capex}],
                         "subtotal_after": None})
        seen.update(a["account"] for a in capex)

    return {"vcode": vcode, "sections": sections,
            "account_count": len(seen),
            "ranked_for_deal": bool(used)}


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
    overrides = category_accounts()
    out: List[Dict[str, Any]] = []
    for section, cats in config.IS_ACCOUNTS.items():
        for cat, config_accts in cats.items():
            accts = overrides.get(cat, config_accts)
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

#: A row that is a SUBTOTAL of other rows on the same sheet. Importing one double-counts
#: everything under it, silently — so anything that announces itself as a total is
#: flagged, whatever it is a total OF. The earlier version required the total to be of a
#: fixed vocabulary (revenue / income / expenses / NOI / EGI), which let
#: "Total Capital Expenditures" through: the Argus keyword rules then matched
#: "capital expenditure" and pre-filled it to 7050 alongside the real Tenant
#: Improvements line, and only the duplicate-account check downstream stopped it.
#:
#: Over-flagging is cheap and under-flagging is not. A flagged row is still RETURNED and
#: still mappable — the flag only means "do not pre-fill this, and leave it unticked" —
#: so a sheet whose sole revenue line is literally called "Total Revenue" still works,
#: it just needs one click.
_TOTAL_ROW = re.compile(
    r"^\s*(?:"
    r"(?:sub)?total\b"                                    # Total …, Subtotal …
    r"|(?:revenue|income|expenses?|operating expenses?"
    r"|net operating income|noi|egi|effective gross)\b"
    r")", re.I)


# An account number the SHEET states. Reading one is not guessing: 4090 in the
# partner's own "Account Number" column, or on the end of "CAM Reimb - 4090", is our
# code, written down. That is categorically different from inferring an account from
# the words "CAM Reimb", which is the guess this import refuses to make.
_ACCT_HEADER_RE = re.compile(
    r'(?i)\b(acct|account|gl)\b.*\b(number|num|no|code|#)\b'
    r'|\b(acct|account|gl)\s*#')
# 3+ digits, so a year, a month number or a floor does not read as an account.
_ACCT_IN_LABEL_RE = re.compile(r'[-–—:]\s*(\d{3,6})\s*$')
#: "4010 - Rental Income" -- the account LEADS the description. This is how our own
#: chart of accounts writes a line and how every roll-up an analyst builds by hand
#: comes out, and it was not matched at all: only a TRAILING account was. Jack's
#: final file was 19 lines every one of which named its account in plain sight, and
#: the app read none of them.
_ACCT_LEADS_LABEL_RE = re.compile(r'^\s*(\d{3,6})\s*[-–—:]')
_ACCT_VALUE_RE = re.compile(r'^\s*(\d{3,6})(\.0+)?\s*$')


def _find_account_column(body: List[list], date_row: int, label_col: int,
                        period_cols: Optional[set] = None) -> Optional[int]:
    """The column holding our account numbers, if the sheet has one.

    Looked for by HEADER first ("Account Number", "PSC Acct Number", "GL Code"), then
    confirmed by content, so a column of years or unit numbers cannot pass as accounts.

    A MONTH COLUMN IS NEVER A CANDIDATE, whatever its header says. An account number
    and a monthly amount are both 3-6 digits -- a January figure of 370,871 reads as an
    account number on its own -- so the only reliable separator is that the month
    columns are already known, and they are excluded rather than argued with.
    """
    header_rows = body[max(0, date_row - 3):date_row + 1]
    candidates = []
    width = max((len(r) for r in body[:date_row + 6]), default=0)
    for col in range(width):
        if col == label_col or (period_cols and col in period_cols):
            continue
        header = ' '.join(str(r[col]) for r in header_rows
                          if col < len(r) and r[col] is not None)
        if header and _ACCT_HEADER_RE.search(header):
            candidates.append(col)
    # Confirm on the body: most populated cells must look like an account number.
    for col in candidates:
        vals = [r[col] for r in body[date_row + 1:] if col < len(r) and r[col] is not None]
        vals = [v for v in vals if str(v).strip()]
        if not vals:
            continue
        hits = sum(1 for v in vals if _ACCT_VALUE_RE.match(str(v).strip()))
        if hits >= max(1, int(0.6 * len(vals))):
            return col
    return None


def _looks_like_account_column(cells: List[str]) -> bool:
    """Mostly numbers that are ACCOUNTS WE CARRY -- not merely 3-6 digit numbers.

    Shape alone is not enough and assuming it was got this wrong immediately: a
    roll-up column of annual totals (1200, 240, 120) matches "3-6 digits" perfectly,
    so it was read as the account column and every line came back with an account
    number of 1200. `_find_account_column` already says this in its own docstring --
    "an account number and a monthly amount are both 3-6 digits" -- and answers it
    with a header match, which a block boundary does not have.

    So the test is membership of our chart of accounts. A budget may state the odd
    account we do not carry (Evergreen's file has 7076 and 5019), hence 80% rather
    than all; an amount column will essentially never clear it.
    """
    if not cells:
        return False
    known = {str(a).strip() for accts in category_accounts().values() for a in accts}
    hits = 0
    for v in cells:
        t = str(v).strip()
        if t.endswith(".0"):
            t = t[:-2]
        if re.fullmatch(r"\d{3,6}", t) and t in known:
            hits += 1
    return hits / len(cells) >= 0.8


def _rebase_to_amount_block(body: List[list], date_row: int, label_col: int,
                            period_cols: set, cells_of) -> Optional[int]:
    """The label column must belong to the SAME block as the amounts.

    Jack's v5 puts TWO INDEPENDENT TABLES side by side: a 19-row roll-up in columns
    A-B, and the 50-row detail it was rolled up FROM in columns D-G, with the months
    beside the detail. The detector finds the leftmost labels, so every line read its
    name from the roll-up and its figures from the detail -- "5051 - Water" carrying
    Property Management's 366,157.78. Nothing about that looks wrong on screen: the
    labels are real, the amounts are real, and they belong to different lines.

    A second block ANNOUNCES ITSELF with a second account column: a run of 3-6 digit
    numbers between the detected labels and the first month. Without one there is no
    evidence of a block boundary and the label column is left exactly where it was --
    a sheet whose labels simply have a sub-description beside them must not be
    re-based onto the sub-description.

    Returns (label column, account column) to use instead, or (None, None) to keep
    the detected ones. The account column comes back with it because finding the
    block boundary IS finding the account column -- it was identified by its content
    here, and handing it over beats discarding it and asking the header-based finder
    to locate it again, which needs a header the sheet may not have.
    """
    if not period_cols:
        return None, None
    first_period = min(period_cols)
    if label_col >= first_period - 1:
        return None, None

    def _mostly_text(cells, ref_count):
        if not cells or len(cells) < 0.5 * ref_count:
            return False
        numeric = 0
        for v in cells:
            try:
                float(str(v).replace(',', ''))
                numeric += 1
            except ValueError:
                pass
        return numeric / len(cells) <= 0.5

    ref = len(cells_of(label_col)) or 1
    boundary = next((c for c in range(label_col + 1, first_period)
                     if c not in period_cols
                     and _looks_like_account_column(cells_of(c))), None)
    if boundary is None:
        return None, None

    # The labels for these amounts are the nearest text column to their LEFT, on the
    # far side of that account column.
    for col in range(first_period - 1, boundary, -1):
        if col in period_cols:
            continue
        if _mostly_text(cells_of(col), ref):
            return col, boundary
    return None, None


def _resolve_label_and_account(body: List[list], date_row: int, label_col: int,
                               period_cols: set) -> tuple:
    """(label column, account column, whether the label was shifted).

    The detector finds ONE label column. A budget often has two: the account number
    and the line's name, side by side. When the detected one is essentially all
    numbers it is the ACCOUNT, and the description is the next column to its right
    that carries text on the same rows.

    Returns the label column unchanged when there is nothing to shift to, so a
    sheet with a single text label column behaves exactly as before.
    """
    def _cells(col):
        out = []
        for r in body[date_row + 1:]:
            if col < len(r) and r[col] is not None and str(r[col]).strip():
                out.append(str(r[col]).strip())
        return out

    rebased, rebased_acct = _rebase_to_amount_block(
        body, date_row, label_col, period_cols, _cells)
    if rebased is not None:
        # A header still wins if the sheet has one -- it is the stronger evidence, and
        # `_find_account_column` confirms it on the content anyway.
        return (rebased,
                _find_account_column(body, date_row, rebased, period_cols)
                or rebased_acct,
                True)

    label_cells = _cells(label_col)
    if not label_cells:
        return label_col, _find_account_column(body, date_row, label_col,
                                               period_cols), False

    def _numeric(v):
        try:
            float(str(v).replace(',', ''))
            return True
        except ValueError:
            return False

    numeric_share = sum(1 for v in label_cells if _numeric(v)) / len(label_cells)
    if numeric_share < 0.9:
        # An ordinary text label column: nothing to shift.
        return label_col, _find_account_column(body, date_row, label_col,
                                               period_cols), False

    # It is a number column. Find the description beside it -- the nearest column
    # to the right, outside the month columns, that is mostly TEXT and populated on
    # a comparable number of rows. Comparable matters: a sparse note column three
    # columns over is not this budget's line names.
    width = max((len(r) for r in body[:date_row + 6]), default=0)
    for col in range(label_col + 1, width):
        if col in period_cols:
            continue
        cells = _cells(col)
        if not cells or len(cells) < 0.5 * len(label_cells):
            continue
        if sum(1 for v in cells if _numeric(v)) / len(cells) > 0.5:
            continue          # another number column, not a description
        return col, label_col, True

    # Numbers with no description anywhere: keep the original behaviour rather
    # than inventing a label.
    return label_col, _find_account_column(body, date_row, label_col,
                                           period_cols), False


def _account_from(row_vals: list, acct_col: Optional[int], label: str) -> Optional[str]:
    """This row's stated account number.

    THE LABEL'S OWN ACCOUNT WINS over a separate column, and that ordering is the
    fix for a silent mis-pairing. A worksheet often carries two independent blocks
    side by side -- Jack's v5 has his rolled-up summary in columns A-B and the
    partner's detail in D-H -- and an account column found in the RIGHT block was
    being read onto the LEFT block's labels, row by row. "Property Management
    Fees" came back as account 4090 (Estimated CAM) carrying the water-reimbursement
    figures, with nothing on screen saying the two had been joined. A label that
    names its own account cannot be mispaired with anything.
    """
    for pat in (_ACCT_LEADS_LABEL_RE, _ACCT_IN_LABEL_RE):
        m = pat.search(label or '')
        if m:
            return m.group(1)
    if acct_col is not None and acct_col < len(row_vals):
        v = row_vals[acct_col]
        if v is not None and _ACCT_VALUE_RE.match(str(v).strip()):
            return _ACCT_VALUE_RE.match(str(v).strip()).group(1)
    return None


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

    # THE LABEL COLUMN MAY BE THE ACCOUNT COLUMN, with the description beside it.
    # The raw budget arrives that way: column A holds the account number, column B
    # the partner's line name. The detector picks A as the label, so every line
    # came through named "4010" or "1" with the description never read at all --
    # Jack: "all I got was a list of account numbers with no descriptions", which
    # is what made him build a helper column joining the two by hand.
    #
    # Detected on TYPE, not on whether the values look like accounts: column A here
    # is 195 placeholder `1`s and 59 real accounts, so "most of them look like
    # accounts" is false. A label column is text; a column that is essentially all
    # numbers is not a label, whatever it holds.
    label_col, acct_col, desc_shift = _resolve_label_and_account(
        body, date_row, label_col, {p["col"] for p in periods})

    lines: List[Dict[str, Any]] = []
    for r in range(date_row + 1, len(body)):
        row_vals = body[r]
        label = row_vals[label_col] if label_col < len(row_vals) else None
        if label is None or not str(label).strip():
            continue
        label = str(label).strip()
        stated_account = _account_from(row_vals, acct_col, label)
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
            # What the SHEET says this account is, if it says anything. The screen
            # pre-fills from this and marks it as coming from the file.
            "stated_account": stated_account,
        })

    return {
        "filename": filename,
        "sheet": det.get("sheet_name"),
        "periods": [p["period"] for p in periods],
        "lines": lines,
        "stated_totals": _stated_totals(lines),
        "account_column": acct_col,
        "label_column": label_col,
        # Said out loud so the confirmation panel can report it: the analyst should
        # be able to see that the app read the description column rather than
        # wonder why the names look different from the file.
        "description_column_used": desc_shift,
        "stated_account_count": sum(1 for l in lines if l.get("stated_account")),
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
