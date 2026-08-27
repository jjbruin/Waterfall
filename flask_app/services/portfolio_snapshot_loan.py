"""Portfolio Snapshot — Subtab 4 (Loan) data assembly (Step 4a).

Backend only: no route, no blueprint, no UI. Nothing imports this module.

Reads, never writes:
  * Step 1  ``portfolio_snapshot_service.resolve_investor_deals`` — deals by fund
  * Step 3a ``portfolio_snapshot_operating.is_dev_deal`` — the single definition
            of "development", shared so the debt path and the Debt Yield
            suppression can never disagree
  * One Pager ``cap_stack`` (ISBS debt) and ``property_performance`` (YTD DSCR,
            YTD NOI), plus ``_child_vcodes_for_parent`` for portfolio parents
  * Step 2  ``portfolio_snapshot_persistence`` — the per-deal loan comment
  * ``valuations`` and ``loans`` frames, injected

All Loan-subtab metrics are property-level and are **never** scaled by
ownership. Only the four TIAA columns on Subtab 2 get scaled.

Definitions as verified against live on 2026-08-24:

Debt
    Operating deals: the One Pager's ISBS-derived balance
    (``cap_stack['debt']``, ISBS Interim BS as of quarter end). This is the
    fresh source that reproduced the PDF.
    Development deals: ``mOrigLoanAmt`` summed over the deal's loans — the full
    committed facility. A dev deal's ISBS balance is unreliable: JB Fair Park
    carries a single 2022-12-31 "SENIOR FINANCING" row of 66,363,992 that the
    app keeps alive because an active MRI loan exists, while the balance sheet
    balances without it (assets ~10.9M vs equity ~10.8M) and accounts
    5190/7030/7060 have no rows at all, so nothing has ever been serviced.

LTV
    Debt / most recent ``mIncomeCapConcludedValue`` by ``dtValuation``.
    Every valuation in the table is a 12/31 and there are no 2026 rows, so this
    currently always lands on 2025-12-31 — matching the PDF's "LTV ('25 Vals)".
    Above ``LTV_REVIEW_CEILING`` the number is withheld and flagged rather than
    printed: JB Fair Park computes 457.7% purely from the stale debt above.

Debt Yield
    (single-quarter Interim IS NOI x 4) / Debt — a quarterly run-rate
    annualisation, computed correctly for **any** quarter.

    The source Excel intends single-period NOI x 4, but its SUMIFS on
    ``Interim IS`` at the quarter-end date actually returns the **YTD cumulative**
    balance (verified live 2026-08-24: Camp Creek YTD@26Q1 1,770,634 equals
    Jan+Feb+Mar exactly, and YTD@26Q2 3,342,478 equals Jan..Jun). At Q1 those
    coincide, so x4 is right; at Q2/Q3/Q4 it over-counts by 2x / 3x / 4x. This
    implementation takes the true single-quarter periodic NOI instead, so it
    reproduces the PDF at Q1 and stays correct afterwards.

    Note the two annualisation bases are NOT interchangeable beyond Q1:
        single-quarter x 4   -> latest-quarter run-rate  (implemented here)
        YTD / months * 12    -> average-to-date annualised
    Both equal ytd x 4 at Q1. ``debt_yield_ytd_annualised`` carries the second
    form alongside for comparison.

    Rendered as "Dev" for development deals — see DEV_DISPLAY.

Rate / Maturity
    Computed here, never read as a literal string. The deal's own loans are used
    when it has any; a portfolio parent with none falls back to its children's.
    One loan, or several that share rate + maturity + interest type, renders the
    actual terms; two or more with differing terms renders "Various".
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import pandas as pd

from flask_app.services.portfolio_snapshot_debt import (
    BASIS_COMMITTED, BASIS_UNAVAILABLE, committed_facility, deal_loan_rows,
    resolve_debt,
)

log = logging.getLogger(__name__)

#: Above this, an LTV is treated as evidence of stale data rather than a real
#: number. 150% is well clear of a genuinely over-levered asset while catching
#: the stale-debt cases (JB Fair Park 457.7%, Trolley Square 366.4%).
LTV_REVIEW_CEILING = 1.50

VARIOUS = "Various"

#: Development deals have no stabilised operations, so the three ratio columns
#: render this literal instead of a number — matching the PDF. Debt, Rate and
#: Maturity still display normally for them.
#:
#: One precedence rule sits ahead of this: a dev deal with NO debt basis at all
#: renders n/a, not "Dev" — see ``dev_no_data`` in ``assemble_loan``.
DEV_DISPLAY = "Dev"
DEV_RATIO_COLUMNS = ("ltv", "ytd_dscr", "debt_yield")

# ══════════════════════════════════════════════════════════════════════════
# TEMPORARY HARDCODED EXCEPTION — REMOVE WHEN THE REAL RULE LANDS
# ══════════════════════════════════════════════════════════════════════════
#: Development deals that nonetheless render a REAL LTV instead of "Dev".
#:
#: Creator instruction, 26Q1: Jefferson Waters Creek received a true
#: income-based valuation on entering lease-up, so its LTV is meaningful even
#: though the deal is still classified as development. DSCR and Debt Yield stay
#: "Dev" for it — only the LTV column is exempted.
#:
#: Verified: computed 57.51% against the PDF's 57.4% (+0.11pp). That tie also
#: settles the numerator — the dev debt basis (mOrigLoanAmt 51,667,000, the
#: committed facility) is correct here. The ISBS drawn balance (48,416,160)
#: would give 53.89% and would NOT match, so the exception un-suppresses the
#: LTV computation without changing which debt figure feeds it.
#:
#: THIS IS A ONE-QUARTER STOPGAP, keyed by vcode, and is deliberately the only
#: hardcoded deal reference in this module. The rule it stands in for is:
#:
#:      a development deal shows a real LTV when its
#:      `mIncomeCapConcludedValue` reflects a true income-based valuation
#:      (rather than a cost-basis or as-stabilised placeholder)
#:
#: Implementing that properly needs a way to tell a genuine income-based
#: valuation from a placeholder in the `valuations` table — most likely a
#: valuation-method or basis column that is not currently extracted. Once that
#: exists, delete this constant and `_ltv_exception()` and drive the exemption
#: off the data. Anything added here in the meantime is technical debt: it will
#: silently keep overriding the real rule after it ships.
WATERS_CREEK_LTV_EXCEPTION = {"P0000078"}       # Jefferson Waters Creek


def _ltv_exception(vcode: str) -> bool:
    """True for a dev deal that still renders a real LTV. TEMPORARY —
    see WATERS_CREEK_LTV_EXCEPTION."""
    return str(vcode or "").strip().upper() in WATERS_CREEK_LTV_EXCEPTION
# ══════════════════════════════════════════════════════════════════════════


def _num(v):
    if v is None:
        return None
    try:
        f = float(str(v).replace(",", "").replace("$", ""))
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f


# ── Subtotals ─────────────────────────────────────────────────────────────
#
# REVERSES this subtab's "ratios are not summed" position. The reference PDF
# page 4 carries a total row per fund and a portfolio total, each with Debt, LTV,
# YTD DSCR and Debt Yield, so they are needed; computed HERE, not in the
# component, so a freeze captures them.
#
# THE AGGREGATION, derived from the PDF's own five fund rows rather than assumed:
#
#   Debt   SUMMED over EVERY deal, development included. Ties all five funds to
#          the cent — 270.5 / 268.4 / 366.5 / 279.2 / 201.5.
#
#   Ratios DEBT-WEIGHTED over the deals that CARRY A VALUE. Not "non-dev": the
#          distinction matters because Jefferson Waters Creek is a dev deal with
#          a real LTV (see WATERS_CREEK_LTV_EXCEPTION), and TGA 2022's published
#          60.4% only reproduces if its 57.5% is included — excluding it gives
#          61.6%. Carrying-a-value is also the rule that needs no dev concept at
#          all, so it cannot drift from the "Dev" display.
#
#          LTV ties all five funds exactly: 67.4 / 60.4 / 59.4 / 62.1 / 69.1.
#          Simple averaging misses every one of them (63.9 / 57.1 / 58.5 /
#          61.6 / 69.1), which settles weighted-vs-simple decisively.
#          DSCR ties TGA23 and TGA25 and is one rounding step out on TGA22
#          (1.68 against 1.6) and TGA24 (1.57 against 1.5); simple averaging is
#          0.3 out on TGA24, so weighted is still the better fit.
#
#   The PDF's footnote — "Summary level performance metrics (LTV, DSCR, and Debt
#   Yield) exclude the development deals" — describes the effect rather than the
#   mechanism: a dev deal renders "Dev" and so carries no value to weight, which
#   excludes it automatically. Waters Creek is the case that shows the footnote's
#   wording is looser than the arithmetic.
#
# TWO PUBLISHED FIGURES DO NOT REPRODUCE under any rule tested; recorded in
# KNOWN_LOAN_SUBTOTAL_DIFFS rather than fitted to.
_RATIO_KEYS = ("ltv", "ytd_dscr", "debt_yield")

#: Fund total cells on PDF page 4 that no consistent rule reproduces.
#: label -> (metric, ours, published, why it is being left alone)
KNOWN_LOAN_SUBTOTAL_DIFFS = {
    ("Total Individual Investments", "ytd_dscr"): (
        "PDF prints n/a although three members carry a DSCR "
        "(Nottingham 1.1x, Evergreen 2.9x, Ascent 2.1x, weighting to 2.08x). "
        "No denominator produces n/a from those."),
    ("Total Individual Investments", "debt_yield"): (
        "PDF 4.1% against 8.71% debt-weighted over the deals with a value. "
        "Weighting over ALL the fund's debt gives 3.76%, nearer but still not "
        "4.1%, and that same denominator breaks TGA 2022 (4.58% against a "
        "published 9.6%)."),
    ("Total PSC TGA 2025 LLC", "debt_yield"): (
        "PDF 4.9% against 13.10%, which is simply Burton's own figure since it "
        "is the only member carrying one. 4.9% IS Burton weighted over the "
        "fund's whole debt (13.1 x 75.3 / 201.5) — but that denominator gives "
        "4.58% on TGA 2022 where 9.6% is published, so the two funds disagree "
        "about the rule and neither can be adopted without breaking the other."),
}


def _debt_weighted(rows: list, key: str):
    """Debt-weighted mean of one ratio over the rows carrying a value."""
    num = den = 0.0
    for r in rows:
        v, d = r.get(key), r.get("debt")
        if v is None or not d:
            continue
        num += v * d
        den += d
    return (num / den) if den else None


def loan_subtotal(rows: list, label: str) -> dict:
    """One total row over ``rows`` — a fund's deals, or every deal."""
    debts = [r.get("debt") for r in rows if r.get("debt") is not None]
    out = {
        "label": label,
        "deal_count": len(rows),
        "dev_count": sum(1 for r in rows if r.get("is_dev")),
        "debt": sum(debts) if debts else None,
        "debt_basis": "sum over every deal, development included",
        "ratio_basis": "debt-weighted over the deals carrying a value",
    }
    for k in _RATIO_KEYS:
        out[k] = _debt_weighted(rows, k)
        out[f"{k}_n"] = sum(1 for r in rows
                            if r.get(k) is not None and r.get("debt"))
    return out


def _s(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return str(v).strip()


def months_elapsed(quarter: str) -> int:
    """Months of the calendar year covered by a YTD figure at this quarter end."""
    try:
        return int(str(quarter).split("Q")[1]) * 3
    except (IndexError, ValueError):
        return 12


# ── Loans ─────────────────────────────────────────────────────────────────

def _deal_loans(loans: pd.DataFrame, vcode: str,
                child_vcodes: Optional[list] = None) -> pd.DataFrame:
    """The loan rows that belong to a deal, falling back to its children.

    Delegates to portfolio_snapshot_debt so the Financial subtab, which needs
    the same rows to size a committed facility, cannot end up selecting them by
    a slightly different rule. Kept as a local name because the terms code below
    also uses it.
    """
    return deal_loan_rows(loans, vcode, child_vcodes)


def _loan_terms(rows: pd.DataFrame) -> dict:
    """Rate / maturity / type for a deal, or "Various" when they differ.

    'Various' is derived by comparing the rows, never read from the data.
    """
    out = {"rate": None, "maturity": None, "interest_type": None,
           "loan_count": 0, "various": False, "rate_display": None,
           "maturity_display": None}
    if rows is None or rows.empty:
        return out

    def maturity_of(r):
        for c in ("dtMaturity", "dtEvent"):
            if c in r.index:
                t = pd.to_datetime(r.get(c), errors="coerce")
                if pd.notna(t):
                    return t.date()
        return None

    def rate_of(r):
        """(numeric rate or None, display string).

        A variable loan carries no nRate — its pricing is vIndex + vSpread, the
        same split the One Pager's _format_loan_str uses. Reading only nRate
        left every floating-rate deal with a blank Rate (Jefferson Addison
        Heights, Eastchase, Stephens, Waters Creek and one of Brainerd's two).
        """
        rate = _num(r.get("nRate"))
        if rate is not None and rate >= 1:
            rate = rate / 100.0
        # ONE decimal on a fixed rate, and a spread in BASIS POINTS — the units
        # the reference document prints ("3.9% fixed", "SOFR + 350"). Two
        # decimals put "5.29%" and "SOFR + 3.50%" in that column, and the
        # spread-as-percent form was the last 2-decimal figure left in the
        # printed report.
        #
        # NOTE this rounds a real hundredth away on screen too — 5.29% now reads
        # 5.3%. That is the published precision; revert these two f-strings if
        # the extra digit is wanted back.
        if rate is not None:
            return rate, f"{rate * 100:.1f}%"
        idx = _s(r.get("vIndex"))
        spr = _num(r.get("vSpread"))
        if idx and spr is not None:
            spr_bps = (spr * 10000) if spr < 1 else (spr * 100)
            return None, f"{idx} + {spr_bps:.0f}"
        if idx:
            return None, idx
        return None, None

    terms = []
    for _, r in rows.iterrows():
        rate, rate_disp = rate_of(r)
        terms.append((rate, maturity_of(r), _s(r.get("vIntType")).lower(),
                      rate_disp))

    out["loan_count"] = len(terms)
    distinct = {t for t in terms}
    if len(distinct) == 1:
        rate, mat, itype, rate_disp = terms[0]
        out.update(rate=rate, maturity=mat, interest_type=itype or None,
                   rate_display=rate_disp,
                   maturity_display=(None if mat is None
                                     else f"{mat.month}/{mat.day}/{mat.year}"))
        return out

    # More than one distinct term set -> Various on whichever field differs.
    # Compared on the rate *descriptor*, so a fixed 4.25% and a floating
    # SOFR+2.00% register as different even though both lack a shared nRate.
    rate_descs = {t[3] for t in terms}
    mats = {t[1] for t in terms}
    types = {t[2] for t in terms}
    out["various"] = True
    if len(rate_descs) == 1:
        out["rate"] = terms[0][0]
        out["rate_display"] = terms[0][3]
    else:
        out["rate_display"] = VARIOUS
    if len(mats) == 1 and terms[0][1] is not None:
        m = terms[0][1]
        out["maturity"] = m
        out["maturity_display"] = f"{m.month}/{m.day}/{m.year}"
    else:
        out["maturity_display"] = VARIOUS
    out["interest_type"] = terms[0][2] if len(types) == 1 else VARIOUS
    return out


# ── Valuations ────────────────────────────────────────────────────────────

def _latest_valuation(valuations: pd.DataFrame, vcode: str) -> dict:
    """Most recent mIncomeCapConcludedValue by dtValuation for a deal."""
    out = {"value": None, "as_of": None}
    if valuations is None or valuations.empty:
        return out
    df = valuations.copy()
    vc_col = next((c for c in df.columns if c.lower() == "vcode"), None)
    val_col = next((c for c in df.columns
                    if c.lower() == "mincomecapconcludedvalue"), None)
    dt_col = next((c for c in df.columns if c.lower() == "dtvaluation"), None)
    if not (vc_col and val_col):
        return out
    df["_vc"] = df[vc_col].astype(str).str.strip().str.upper()
    sub = df[df["_vc"] == str(vcode).strip().upper()].copy()
    if sub.empty:
        return out
    sub["_v"] = sub[val_col].map(_num)
    if dt_col:
        sub["_dt"] = pd.to_datetime(sub[dt_col], errors="coerce")
        sub = sub.dropna(subset=["_dt"]).sort_values("_dt", ascending=False)
    sub = sub[sub["_v"].notna() & (sub["_v"] > 0)]
    if sub.empty:
        return out
    row = sub.iloc[0]
    out["value"] = float(row["_v"])
    out["as_of"] = row["_dt"].date() if dt_col and pd.notna(row.get("_dt")) else None
    return out


def _default_comment_loader(investor_code: str, quarter: str) -> dict:
    try:
        from flask_app.services.portfolio_snapshot_persistence import get_elements
        rows = get_elements("comment", investor_code, quarter,
                            scope="deal", field="loan")
        return {r.get("scope_key"): r.get("comment_text") for r in rows}
    except Exception as exc:
        log.debug("loan comments unavailable: %s", exc)
        return {}


# ── Assembly ──────────────────────────────────────────────────────────────

def assemble_loan(investor_code: str, quarter: str, *,
                  resolved: dict,
                  one_pager_provider: Callable[[str, str], dict],
                  loans: pd.DataFrame,
                  valuations: pd.DataFrame,
                  inv: Optional[pd.DataFrame] = None,
                  quarterly_noi_provider: Optional[Callable] = None,
                  comment_loader: Optional[Callable] = None,
                  ltv_ceiling: float = LTV_REVIEW_CEILING) -> dict:
    """Build the Loan subtab for one investor and quarter."""
    from flask_app.services.portfolio_snapshot_operating import (
        is_dev_deal, resolve_strategy)
    from flask_app.services.portfolio_snapshot_service import (
        group_total_label, PORTFOLIO_TOTAL_LABEL,
    )

    loader = comment_loader or _default_comment_loader
    comments = loader(investor_code, quarter) or {}
    n_months = months_elapsed(quarter)

    diag = {"deals": 0, "dev": 0, "debt_from_isbs": 0, "debt_from_orig": 0,
            "ltv_ok": 0, "ltv_no_valuation": 0, "ltv_flagged_review": 0,
            "ltv_dev": 0, "ltv_dev_exception": 0, "dev_no_data": 0,
            "dscr_dev": 0,
            "dy_ok": 0, "dy_dev_suppressed": 0, "dy_no_ytd": 0,
            "various_terms": 0, "comments_attached": 0, "provider_errors": 0}

    child_cache: dict = {}

    def children_of(vcode: str) -> list:
        if inv is None:
            return []
        if vcode not in child_cache:
            try:
                from one_pager import _child_vcodes_for_parent  # read-only reuse
                child_cache[vcode] = _child_vcodes_for_parent(vcode, inv) or []
            except Exception:
                child_cache[vcode] = []
        return child_cache[vcode]

    def build_row(vcode: str, name: str, strategy: str,
                  extra_flags: Optional[list] = None) -> dict:
        flags = list(extra_flags or [])
        dev = is_dev_deal(strategy)
        if dev:
            diag["dev"] += 1
        # TEMPORARY: dev deal exempted from the "Dev" LTV suppression only.
        # DSCR and Debt Yield below stay keyed on `dev` alone, so they still
        # render "Dev" for it. See WATERS_CREEK_LTV_EXCEPTION.
        ltv_exempt = dev and _ltv_exception(vcode)
        if ltv_exempt:
            diag["ltv_dev_exception"] += 1
            flags.append("TEMPORARY exception — real LTV shown despite dev "
                         "classification (income-based valuation at lease-up)")

        try:
            payload = one_pager_provider(vcode, quarter) or {}
        except Exception as exc:
            diag["provider_errors"] += 1
            flags.append(f"One Pager unavailable: {str(exc)[:80]}")
            payload = {}
        cap = payload.get("cap_stack") or {}
        perf = payload.get("property_performance") or {}

        # ---- loans, terms ----
        kids = children_of(vcode)
        lrows = _deal_loans(loans, vcode, kids)
        terms = _loan_terms(lrows)
        inherited = bool(len(lrows)) and lrows["_vc"].iloc[0] != str(vcode).upper()
        if inherited:
            flags.append(f"loan terms inherited from {len(lrows)} child loan(s)")
        if terms["various"]:
            diag["various_terms"] += 1
        if terms["loan_count"] == 0:
            flags.append("no loan record")

        orig_total = committed_facility(lrows)

        # ---- debt ----
        # The basis choice lives in portfolio_snapshot_debt.resolve_debt, which
        # the Financial subtab calls too — the two used to disagree about the
        # same deal's Debt (JB Fair Park 66.36 vs 77.37).
        # Pre-override ISBS balance — see portfolio_snapshot_debt.resolve_debt.
        isbs_debt = _num(cap.get("debt_isbs", cap.get("debt")))
        debt, debt_basis = resolve_debt(cap, dev, orig_total)
        if debt_basis == BASIS_COMMITTED:
            diag["debt_from_orig"] += 1
        elif debt_basis == BASIS_UNAVAILABLE:
            flags.append("dev deal with no mOrigLoanAmt")
        elif debt:
            diag["debt_from_isbs"] += 1
        else:
            flags.append("no ISBS debt balance")

        # ---- empty loan block: n/a takes precedence over the "Dev" label ----
        # A dev deal with no debt basis at all — no ISBS balance, no
        # mOrigLoanAmt, no loan record — has nothing to show, so the three
        # ratio columns read n/a rather than claiming a status the data does
        # not support. Gated here, AHEAD of the dev checks below.
        # Pegasus Life Storage at 26Q1 is the case: the PDF shows n/a for its
        # LTV, DSCR and Debt Yield, while the label alone would render "Dev".
        #
        # Deliberately keyed on `debt is None` — the "loan block is empty"
        # signal — and NOT on per-column ratio availability. Per-column gating
        # over-flips: no dev deal carries a real YTD DSCR (raw ytd_actual is
        # None on 9 of the 10 at 26Q1), so DSCR would fall to n/a almost
        # everywhere and JB Fair Park would stop reading "Dev" where the PDF
        # shows it. Scoped to dev deals so a non-dev deal never loses its real
        # DSCR, which is NOI / debt service and does not depend on the balance.
        dev_no_data = dev and debt is None
        if dev_no_data:
            diag["dev_no_data"] += 1

        # ---- LTV ----
        # Dev deals render "Dev": the ratio is meaningless pre-stabilisation and
        # their debt basis is a committed facility, not a drawn balance. The
        # >ceiling guard therefore only ever fires for NON-dev deals now, as a
        # backstop against stale data (its original catch, JB Fair Park at 457%,
        # is a dev deal and is handled here instead).
        val = _latest_valuation(valuations, vcode)
        ltv = ltv_flag = None
        if dev and not ltv_exempt:
            diag["ltv_dev"] += 1
        elif debt and val["value"]:
            raw = debt / val["value"]
            if raw > ltv_ceiling:
                ltv_flag = (f"review — LTV {raw * 100:.0f}% exceeds "
                            f"{ltv_ceiling * 100:.0f}%, likely stale debt")
                flags.append(ltv_flag)
                diag["ltv_flagged_review"] += 1
            else:
                ltv = raw
                diag["ltv_ok"] += 1
        elif not val["value"]:
            flags.append("no valuation — LTV unavailable")
            diag["ltv_no_valuation"] += 1

        # ---- YTD DSCR (One Pager, read-only) ----
        dscr_ytd = None if dev else _num((perf.get("dscr") or {}).get("ytd_actual"))
        if dev:
            diag["dscr_dev"] += 1

        # ---- Debt Yield: single-quarter NOI x 4 (see module docstring) ----
        ytd_noi = _num((perf.get("noi") or {}).get("ytd_actual"))
        q_noi = None if quarterly_noi_provider is None else _num(
            quarterly_noi_provider(vcode, quarter))
        annualised = dy = dy_ytd = None
        if dev:
            flags.append(
                "no debt basis — LTV / YTD DSCR / Debt Yield shown as n/a"
                if dev_no_data
                else "ratios shown as 'Dev' — development deal")
            diag["dy_dev_suppressed"] += 1
        elif not debt:
            flags.append("Debt Yield n/a — no debt balance")
        elif q_noi is None:
            flags.append("Debt Yield n/a — no complete quarter of actual NOI")
            diag["dy_no_ytd"] += 1
        else:
            annualised = q_noi * 4
            dy = annualised / debt
            diag["dy_ok"] += 1
            if ytd_noi:
                # comparison basis only; equals `dy` at Q1
                dy_ytd = (ytd_noi / n_months * 12) / debt

        comment = comments.get(vcode)
        if comment:
            diag["comments_attached"] += 1
        diag["deals"] += 1

        return {
            "vcode": vcode, "name": name, "strategy": strategy, "is_dev": dev,
            "debt": debt, "debt_basis": debt_basis,
            "isbs_debt": isbs_debt, "orig_loan_amt": orig_total,
            "valuation": val["value"], "valuation_as_of": val["as_of"],
            "ltv": ltv, "ltv_review_flag": ltv_flag,
            # n/a (None) wins over "Dev" when the loan block is empty —
            # see dev_no_data above.
            "ltv_display": (None if dev_no_data else
                            DEV_DISPLAY if (dev and not ltv_exempt) else ltv),
            "ltv_dev_exception": ltv_exempt,
            "dev_no_data": dev_no_data,
            "ytd_dscr": dscr_ytd,
            "ytd_dscr_display": (None if dev_no_data else
                                 DEV_DISPLAY if dev else dscr_ytd),
            "ytd_noi": ytd_noi, "quarter_noi": q_noi,
            "annualised_noi": annualised,
            "months_elapsed": n_months, "debt_yield": dy,
            "debt_yield_display": (None if dev_no_data else
                                   DEV_DISPLAY if dev else dy),
            "debt_yield_ytd_annualised": dy_ytd,
            "debt_yield_basis": "single-quarter Interim IS NOI x 4 / debt",
            "loan_count": terms["loan_count"],
            "rate": terms["rate"], "rate_display": terms["rate_display"],
            "maturity": terms["maturity"],
            "maturity_display": terms["maturity_display"],
            "interest_type": terms["interest_type"],
            "terms_various": terms["various"],
            "loans_inherited_from_children": inherited,
            "loan_comment": comment,
            "flags": flags,
        }

    groups: dict[str, list] = {}
    for group, items in (resolved.get("groups") or {}).items():
        groups[group] = [build_row(e["vcode"], e["name"], resolve_strategy(e)[0])
                         for e in items]

    flagged_rows = []
    for f in (resolved.get("flagged") or []):
        row = build_row(f["vcode"], f["name"], resolve_strategy(f)[0],
                        extra_flags=[f"ownership {f.get('reason', 'unavailable')}"])
        row["ownership_flagged"] = True
        flagged_rows.append(row)

    return {
        "investor_code": resolved.get("investor_code", investor_code),
        "investor_name": resolved.get("investor_name", investor_code),
        "quarter": quarter, "subtab": "loan", "scaled": False,
        "months_elapsed": n_months,
        "ltv_review_ceiling": ltv_ceiling,
        "groups": groups,
        "ownership_flagged": flagged_rows,
        # Alongside `groups`, not nested in it, so every existing consumer keeps
        # reading {name: [rows]} unchanged.
        "group_labels": {g: group_total_label(g) for g in groups},
        "subtotals": {g: loan_subtotal(rows, group_total_label(g))
                      for g, rows in groups.items()},
        "total": loan_subtotal(
            [r for rows in groups.values() for r in rows] + flagged_rows,
            PORTFOLIO_TOTAL_LABEL),
        "diagnostics": diag,
    }


# ── Self-test ─────────────────────────────────────────────────────────────

# PDF 26Q1 loan page, as supplied
_PDF = {
    "P0000019": ("Giant 7", 70.9, None),
    "P0000075": ("Camp Creek", 58.0, 13.5),
    "P0000030": ("Nottingham Village", 75.0, None),
    "P0000079": ("Post Commons", None, 10.7),
}


def _selftest():                                    # pragma: no cover
    import json
    import os
    import sys
    import tempfile
    import sqlalchemy

    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    for p in (root, os.path.join(root, "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import live_api as api
    from flask_app.services.portfolio_snapshot_service import resolve_investor_deals
    from flask_app.services import portfolio_snapshot_persistence as P

    INV, Q = "TGAM", "2026-Q1"
    ti = api.token_info()
    print(f"LIVE token={ti['username']} ({ti['hours_left']}h)  "
          f"build={api.get('/api/data/version').get('version')}  "
          f"actuals_through={api.get('/api/data/config').get('actuals_through')}")

    inv = pd.DataFrame(api.get("/api/data/deals/all").get("deals") or [])
    loans = pd.DataFrame(api.get("/api/data/tables/loans/rows",
                                 params={"page": 1, "page_size": 500}
                                 ).get("rows") or [])
    vals = pd.DataFrame(api.get("/api/data/tables/valuations/rows",
                                params={"page": 1, "page_size": 500}
                                ).get("rows") or [])
    print(f"loans {len(loans)} rows, valuations {len(vals)} rows")

    seen, frontier, rows = set(), [INV], []

    def fetch(col, v):
        d = api.get("/api/data/tables/relationships/rows",
                    params={"page": 1, "page_size": 500, f"filter__{col}": v})
        return [r for r in (d.get("rows") or [])
                if str(r.get(col) or "").strip().upper() == v.upper()]

    while frontier:
        node = frontier.pop().upper()
        if node in seen:
            continue
        seen.add(node)
        kids = fetch("InvestorID", node)
        rows.extend(kids)
        for r in kids:
            c = str(r.get("InvestmentID") or "").strip().upper()
            if c:
                rows.extend(fetch("InvestmentID", c))
                if c not in seen:
                    frontier.append(c)
    rel = pd.DataFrame(rows).drop_duplicates()
    resolved = resolve_investor_deals(INV, Q, rel, inv)
    print(f"Step 1: {resolved['diagnostics']['deal_count']} deals, "
          f"{len(resolved['flagged'])} ownership-flagged\n")

    tmp = os.path.join(tempfile.mkdtemp(prefix="ps_step4a_"), "t.db")
    eng = sqlalchemy.create_engine(f"sqlite:///{tmp}")
    P._engine = lambda: eng                          # type: ignore[assignment]
    P._is_postgres = lambda: False                   # type: ignore[assignment]
    P.save_comment(INV, Q, "deal", "loan", "Fixed through 2032; no extension used.",
                   scope_key="P0000019", updated_by="selftest")
    P.save_comment(INV, Q, "deal", "loan", "SOFR cap purchased at 4.0%.",
                   scope_key="P0000075", updated_by="selftest")

    cache: dict = {}

    def provider(vc, q):
        if (vc, q) not in cache:
            cache[(vc, q)] = api.get(f"/api/financials/{vc}/one-pager",
                                     params={"quarter": q})
        return cache[(vc, q)]

    qcache: dict = {}

    def q_noi(vc, q):
        """True single-quarter periodic NOI for the report quarter.

        aggregate_periodic drops any quarter without all three months, so a
        partially reported quarter yields None and the caller flags it rather
        than annualising a stub.
        """
        if (vc, q) in qcache:
            return qcache[(vc, q)]
        yr, qn = int(q.split("-Q")[0]), int(q.split("Q")[1])
        qend = pd.Timestamp(year=yr, month=qn * 3, day=1) + pd.offsets.MonthEnd(0)
        ch = api.get(f"/api/financials/{vc}/performance-chart",
                     params={"freq": "Quarterly", "periods": 12,
                             "period_end": str(qend.date())})
        val = None
        for lbl, a in zip(ch.get("periods") or [], ch.get("actual_noi") or []):
            if lbl == f"Q{qn} {yr}" and a is not None:
                val = a
                break
        qcache[(vc, q)] = val
        return val

    out = assemble_loan(INV, Q, resolved=resolved, one_pager_provider=provider,
                        loans=loans, valuations=vals, inv=inv,
                        quarterly_noi_provider=q_noi,
                        comment_loader=lambda i, q: _default_comment_loader(i, q))

    flat = {r["vcode"]: r for rs in out["groups"].values() for r in rs}
    for r in out["ownership_flagged"]:
        flat[r["vcode"]] = r

    checks = []

    def chk(label, cond):
        checks.append((label, bool(cond)))
        print(f"    [{'PASS' if cond else 'FAIL'}] {label}")

    print("=" * 112)
    print(f"COMPUTED vs 26Q1 PDF — LTV and Debt Yield")
    print(f"{'deal':<26}{'metric':<16}{'computed':>11}{'PDF':>8}{'delta':>9}   verdict")
    print("-" * 112)
    for vc, (label, p_ltv, p_dy) in _PDF.items():
        r = flat.get(vc)
        if not r:
            chk(f"{label} present", False)
            continue
        first = True
        if p_ltv is not None:
            c = (r["ltv"] or 0) * 100
            d = c - p_ltv
            ok = abs(d) <= 1.0
            print(f"{label:<26}{'LTV %':<16}{c:>11.2f}{p_ltv:>8.1f}{d:>+9.2f}"
                  f"   {'ok' if ok else 'MISMATCH'}")
            checks.append((f"{label} LTV within 1pt", ok))
            first = False
        if p_dy is not None:
            c = (r["debt_yield"] or 0) * 100
            d = c - p_dy
            ok = abs(d) <= 0.5
            print(f"{'' if not first else label:<26}{'Debt Yield %':<16}"
                  f"{c:>11.2f}{p_dy:>8.1f}{d:>+9.2f}   {'ok' if ok else 'MISMATCH'}")
            checks.append((f"{label} Debt Yield within 0.5pt", ok))

    print("\n" + "=" * 112)
    print("LTV — all deals (PDF column is \"LTV ('25 Vals)\")")
    print(f"{'vcode':<9}{'deal':<30}{'debt':>14}{'basis':<12}{'valuation':>14}"
          f"{'as of':>12}{'LTV':>9}")
    print("-" * 112)
    for g, rs in out["groups"].items():
        for r in rs:
            basis = "orig" if r["is_dev"] else "isbs"
            print(f"{r['vcode']:<9}{r['name'][:29]:<30}"
                  f"{(r['debt'] or 0):>14,.0f}{basis:<12}"
                  f"{(r['valuation'] or 0):>14,.0f}"
                  f"{str(r['valuation_as_of'] or '-'):>12}"
                  f"{(f'{r['ltv']*100:.1f}%' if r['ltv'] else 'n/a'):>9}")

    print("\n" + "=" * 112)
    print("RATE / MATURITY — Various logic")
    for vc in ("P0000067", "P0000075", "P0000109", "P0000003", "P0000019"):
        r = flat.get(vc)
        if not r:
            continue
        print(f"  {vc} {r['name'][:32]:<34} loans={r['loan_count']:<3}"
              f"rate={str(r['rate_display']):<10}maturity={str(r['maturity_display']):<12}"
              f"various={r['terms_various']}"
              f"{'  (inherited from children)' if r['loans_inherited_from_children'] else ''}")

    print("\n" + "=" * 112)
    print('DEV DEALS — three ratio columns show "Dev"; Debt/Rate/Maturity stay numeric')
    print(f"{'vcode':<9}{'deal':<30}{'LTV':>7}{'DSCR':>7}{'DebtYld':>9}"
          f"{'Debt':>14}{'Rate':>17}{'Maturity':>12}")
    print("-" * 112)
    for r in sorted((x for x in flat.values() if x["is_dev"]),
                    key=lambda x: x["name"]):
        debt_s = f"{r['debt']:,.0f}" if r["debt"] else "n/a"
        print(f"{r['vcode']:<9}{r['name'][:29]:<30}"
              f"{str(r['ltv_display']):>7}{str(r['ytd_dscr_display']):>7}"
              f"{str(r['debt_yield_display']):>9}{debt_s:>14}"
              f"{str(r['rate_display'] or 'n/a'):>17}"
              f"{str(r['maturity_display'] or 'n/a'):>12}")

    print("\n" + "=" * 112)
    print("STRUCTURE CHECKS")
    chk("groups present", len(out["groups"]) >= 5)
    chk("no scaling on loan metrics", out["scaled"] is False)
    # Property, not identity — see the note in portfolio_snapshot_service.
    chk("ownership-flagged deals still appear with deal-level figures",
        len(out["ownership_flagged"]) == len(resolved["flagged"]))
    chk("Giant 7 loan comment attached",
        (flat.get("P0000019") or {}).get("loan_comment", "").startswith("Fixed"))
    chk("Camp Creek loan comment attached",
        (flat.get("P0000075") or {}).get("loan_comment") ==
        "SOFR cap purchased at 4.0%.")
    chk("deal with no comment yields None",
        (flat.get("P0000030") or {}).get("loan_comment") is None)
    jb = flat.get("P0000021") or {}
    # Superseded: JB Fair Park is a dev deal, so its LTV is "Dev" and the >150%
    # guard no longer fires for it. The guard is now a backstop for non-dev
    # stale data only. Asserted below as "shows 'Dev', not the 457% flag".
    chk("JB Fair Park LTV withheld (no garbage number)",
        jb.get("ltv") is None)
    chk("JB Fair Park debt uses mOrigLoanAmt (dev path)",
        jb.get("debt") == jb.get("orig_loan_amt") and jb.get("debt") is not None)
    chk("JB Fair Park debt differs from the stale ISBS figure",
        jb.get("isbs_debt") not in (None, jb.get("debt")))
    chk("every dev deal has Debt Yield n/a",
        all(r["debt_yield"] is None for r in flat.values() if r["is_dev"]))
    chk("Redevelopment is NOT dev (Crowne Plaza absent anyway)",
        all(r["strategy"].strip().lower() != "redevelopment" or not r["is_dev"]
            for r in flat.values()))
    chk("Brainerd renders Various", (flat.get("P0000067") or {}).get("terms_various") is True)
    chk("all dev deals render 'Dev' for LTV / YTD DSCR / Debt Yield "
        "(except the temporary LTV exception and the empty-loan-block case)",
        all(r["ltv_display"] == DEV_DISPLAY
            and r["ytd_dscr_display"] == DEV_DISPLAY
            and r["debt_yield_display"] == DEV_DISPLAY
            for r in flat.values()
            if r["is_dev"] and not r["ltv_dev_exception"]
            and not r["dev_no_data"]) and
        any(r["is_dev"] for r in flat.values()))

    # ---- empty loan block: n/a takes precedence over "Dev" ----
    print("\n" + "=" * 108)
    print('EMPTY LOAN BLOCK — dev deal with no debt basis reads n/a, not "Dev"')
    print("=" * 108)
    print(f"  {'vcode':<10}{'deal':<30}{'debt':>14}{'LTV':>9}{'DSCR':>8}"
          f"{'DebtYld':>9}   dev_no_data")
    print("  " + "-" * 106)
    for r in sorted((x for x in flat.values() if x["is_dev"]),
                    key=lambda x: x["name"]):
        def d(v):
            if v is None:
                return "n/a"
            return v if isinstance(v, str) else f"{v:.4f}"
        print(f"  {r['vcode']:<10}{r['name'][:29]:<30}"
              f"{(r['debt'] if r['debt'] else 0):>14,.0f}"
              f"{d(r['ltv_display'])[:8]:>9}{d(r['ytd_dscr_display'])[:7]:>8}"
              f"{d(r['debt_yield_display'])[:8]:>9}   "
              f"{r['dev_no_data']}")

    peg = flat.get("P0000066") or {}
    chk("Pegasus has no debt basis (empty loan block)",
        peg.get("debt") is None and peg.get("orig_loan_amt") is None
        and peg.get("loan_count") == 0)
    chk("Pegasus is flagged dev_no_data", peg.get("dev_no_data") is True)
    chk("Pegasus LTV reads n/a, not 'Dev' (matches PDF)",
        peg.get("ltv_display") is None)
    chk("Pegasus YTD DSCR reads n/a, not 'Dev' (matches PDF)",
        peg.get("ytd_dscr_display") is None)
    chk("Pegasus Debt Yield reads n/a, not 'Dev' (matches PDF)",
        peg.get("debt_yield_display") is None)
    chk("Pegasus carries the n/a explanation flag",
        any("no debt basis" in f for f in peg.get("flags") or []))

    no_data = sorted(r["vcode"] for r in flat.values() if r["dev_no_data"])
    chk("EXACTLY ONE deal changes — only Pegasus is dev_no_data",
        no_data == ["P0000066"])
    chk("the other 9 dev deals keep 'Dev' on DSCR and Debt Yield",
        all(r["ytd_dscr_display"] == DEV_DISPLAY
            and r["debt_yield_display"] == DEV_DISPLAY
            for r in flat.values()
            if r["is_dev"] and r["vcode"] != "P0000066"))
    chk("the other 9 dev deals keep 'Dev' on LTV (bar Waters Creek)",
        all(r["ltv_display"] == DEV_DISPLAY for r in flat.values()
            if r["is_dev"] and r["vcode"] not in ("P0000066", "P0000078")))
    chk("gate is scoped to dev deals — no non-dev deal is dev_no_data",
        all(not r["dev_no_data"] for r in flat.values() if not r["is_dev"]))
    chk("no non-dev deal lost its real DSCR to the gate",
        all(r["ytd_dscr_display"] == r["ytd_dscr"]
            for r in flat.values() if not r["is_dev"]))

    # ---- TEMPORARY Waters Creek LTV exception ----
    print("\n" + "=" * 108)
    print('TEMPORARY LTV EXCEPTION — Jefferson Waters Creek: real LTV, '
          '"Dev" for DSCR / Debt Yield')
    print("=" * 108)
    wc = flat.get("P0000078") or {}
    print(f"    deal   : P0000078 {wc.get('name', '?')}")
    print(f"    strategy={wc.get('strategy')!r}  is_dev={wc.get('is_dev')}  "
          f"ltv_dev_exception={wc.get('ltv_dev_exception')}")
    print(f"    LTV display        : {wc.get('ltv_display')!r}"
          f"   (numeric {wc.get('ltv')})")
    print(f"    YTD DSCR display   : {wc.get('ytd_dscr_display')!r}")
    print(f"    Debt Yield display : {wc.get('debt_yield_display')!r}")
    print(f"    debt {wc.get('debt')!r} ({wc.get('debt_basis')})")
    print(f"    ISBS debt {wc.get('isbs_debt')!r}   "
          f"mOrigLoanAmt {wc.get('orig_loan_amt')!r}")
    print(f"    valuation {wc.get('valuation')!r} as of {wc.get('valuation_as_of')!r}")
    if wc.get("valuation") and wc.get("isbs_debt"):
        print(f"    LTV on ISBS drawn balance would be "
              f"{wc['isbs_debt'] / wc['valuation']:.2%} "
              f"(shown value uses the committed facility)")

    chk("Waters Creek is classified as a dev deal", wc.get("is_dev") is True)
    chk("Waters Creek is the LTV exception",
        wc.get("ltv_dev_exception") is True)
    chk("Waters Creek shows a REAL numeric LTV, not 'Dev'",
        isinstance(wc.get("ltv_display"), (int, float))
        and wc.get("ltv_display") is not None
        and wc.get("ltv_display") == wc.get("ltv"))
    chk("Waters Creek still shows 'Dev' for YTD DSCR",
        wc.get("ytd_dscr_display") == DEV_DISPLAY)
    chk("Waters Creek still shows 'Dev' for Debt Yield",
        wc.get("debt_yield_display") == DEV_DISPLAY)
    chk("Waters Creek is the ONLY LTV exception",
        [r["vcode"] for r in flat.values() if r["ltv_dev_exception"]]
        == ["P0000078"])
    # Excludes Waters Creek (the temporary LTV exception) and any deal caught
    # by the empty-loan-block gate, which reads n/a rather than "Dev".
    chk("every OTHER dev deal shows 'Dev' for LTV",
        all(r["ltv_display"] == DEV_DISPLAY for r in flat.values()
            if r["is_dev"] and r["vcode"] != "P0000078"
            and not r["dev_no_data"]))
    chk("no non-dev deal is ever an LTV exception",
        all(not r["ltv_dev_exception"] for r in flat.values()
            if not r["is_dev"]))
    chk("the exception constant names exactly one deal",
        WATERS_CREEK_LTV_EXCEPTION == {"P0000078"})
    chk("dev deals still show numeric Debt",
        all(r["debt"] is not None for r in flat.values()
            if r["is_dev"] and r["orig_loan_amt"]))
    chk("dev deals still show Rate/Maturity where a loan exists",
        all(r["rate_display"] is not None for r in flat.values()
            if r["is_dev"] and r["loan_count"] > 0))
    chk("JB Fair Park shows 'Dev', not the 457% flag",
        (flat.get("P0000021") or {}).get("ltv_display") == DEV_DISPLAY
        and (flat.get("P0000021") or {}).get("ltv_review_flag") is None)
    chk("LTV guard no longer fires for any dev deal (backstop only)",
        all(r["ltv_review_flag"] is None for r in flat.values() if r["is_dev"]))
    chk("non-dev deals keep numeric LTV display",
        all(r["ltv_display"] is None or isinstance(r["ltv_display"], float)
            for r in flat.values() if not r["is_dev"]))
    chk("Debt Yield basis is single-quarter x 4",
        all(r["debt_yield_basis"].startswith("single-quarter")
            for r in flat.values()))
    chk("at Q1 single-quarter x4 equals YTD-annualised",
        all(r["debt_yield_ytd_annualised"] is None
            or abs(r["debt_yield"] - r["debt_yield_ytd_annualised"]) < 1e-9
            for r in flat.values() if r["debt_yield"] is not None))
    chk("dev deals use the committed facility, operating use ISBS",
        all((r["debt_basis"].startswith("mOrigLoanAmt") if r["is_dev"]
             else r["debt_basis"].startswith("ISBS")) for r in flat.values()
            if r["debt"] is not None))

    d = out["diagnostics"]
    print(f"\n  diagnostics: {d}")
    print("\n" + "=" * 112)
    print("ASSEMBLED STRUCTURE — two deals verbatim")
    for vc in ("P0000075", "P0000021"):
        print(f"\n{vc}:")
        print(json.dumps(flat.get(vc), indent=2, default=str)[:1300])

    print(f"\n  {sum(1 for _, c in checks if c)}/{len(checks)} checks passed")
    return 0


if __name__ == "__main__":                          # pragma: no cover
    raise SystemExit(_selftest())
