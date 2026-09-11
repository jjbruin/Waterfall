"""Modeled debt service for the valuation section's budget comparison.

WHY. The comparison is Estimate | Budget | Valuation. Its Interest and Principal rows
come from three different places, and two of them are unreliable for the job:

  * The appraiser's Argus download is UNLEVERED — Argus projects property-level cash
    flow before debt — so accounts 5190 and 7060 are simply absent and the Valuation
    column showed 0 interest, 0 principal and a blank DSCR. A DSCR you cannot see is a
    DSCR nobody checks.

  * A partner's budget may or may not carry debt service, and when it does it is their
    amortization assumption, not ours.

Meanwhile the app already models this deal's debt properly, from the loan terms, in the
module that produces Deal Analysis and the waterfall. So the schedule is BUILT here from
those same loans and laid into the right months — the same thing `compute.py` already
does for the AM forecast, where it strips whatever debt service the source carried and
replaces it with the modeled schedule (compute.py, "Replace forecast debt service with
modeled"). This is that rule, applied to the valuation comparison.

WHAT IT DOES NOT TOUCH. The Estimate column. That column means "actuals so far, plus
budget for the rest of the year", and its interest is interest that was actually paid.
Replacing a reported figure with a modeled one would be a worse number, not a better one.

ACCOUNTS. Interest lands on 5190 and principal on 7060, because those are what the
comparison reads (`config.IS_ACCOUNTS['DEBT_SERVICE']['Interest'] == ['5190']`, and
`_get_budget_principal` reads 7060). Note the AM forecast writes interest to 7030
instead — `list(INTEREST_ACCTS)[0]` happens to yield 7030 from that set — so the two
differ on purpose until that is settled deliberately rather than as a side effect here.

BALLOONS ARE EXCLUDED, matching compute.py: a balloon is repaid from sale proceeds, not
from operating cash, and including it would wreck DSCR in the maturity year.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

#: What the budget comparison reads. See the module docstring on 7030 vs 5190.
INTEREST_ACCOUNT = "5190"
PRINCIPAL_ACCOUNT = "7060"


def _deal_vcodes(vcode: str, data: dict) -> List[str]:
    """The deal plus any child properties.

    Loans aggregate UP from properties to the parent deal (see consolidation.py), so a
    portfolio deal whose debt sits on its children would otherwise model zero debt and
    report a blank DSCR while carrying a mortgage.
    """
    out = [str(vcode)]
    try:
        from consolidation import get_property_vcodes_for_deal
        deals = data.get("inv")
        if deals is not None and not deals.empty:
            out += [str(v) for v in get_property_vcodes_for_deal(str(vcode), deals)]
    except Exception as e:                                        # noqa: BLE001
        logger.warning("Debt service: child property lookup failed for %s (%s)", vcode, e)
    return list(dict.fromkeys(out))


def monthly_schedule(vcode: str, start: date, end: date, data: dict) -> Dict[str, Any]:
    """Modeled interest and principal by month, from this deal's actual loan terms.

    Returns the schedule plus the notes a reader needs to judge it — an empty schedule
    is reported as "no loans", never as zero debt service.
    """
    import loans as loans_mod

    notes: List[str] = []
    ml = data.get("mri_loans_raw")
    if ml is None or ml.empty:
        return _empty("No loan data is loaded.", start, end)

    col = next((c for c in ml.columns if c.lower() == "vcode"), None)
    if col is None:
        return _empty("The loan table has no vCode column.", start, end)

    codes = {c.strip().lower() for c in _deal_vcodes(vcode, data)}
    sub = ml[ml[col].astype(str).str.strip().str.lower().isin(codes)]
    if sub.empty:
        return _empty(f"{vcode} has no loans in MRI, so there is no debt service to model.",
                      start, end)

    # Paid-off loans are already filtered at the data layer (data_service.load_all), so
    # anything still here is live. Not re-filtered: a second, divergent copy of that rule
    # is how the two stop agreeing.
    loan_objs = loans_mod.build_loans_from_mri_loans(sub)
    if not loan_objs:
        return _empty(f"{vcode} has loan rows but none could be modeled "
                      f"(missing amount, rate or maturity).", start, end)

    frames = []
    for lo in loan_objs:
        try:
            f = loans_mod.amortize_monthly_schedule(lo, start, end)
            if f is not None and not f.empty:
                frames.append(f)
        except Exception as e:                                    # noqa: BLE001
            notes.append(f"Loan {getattr(lo, 'loan_id', '?')} could not be amortized ({e}).")
    if not frames:
        return _empty(f"{vcode}'s loans are outside {start:%b %Y}–{end:%b %Y} "
                      f"(matured, or not yet originated).", start, end)

    sched = pd.concat(frames, ignore_index=True)

    # A balloon is repaid from sale proceeds, not operating cash — same rule and the same
    # test as compute.py, so DSCR does not collapse in the maturity year.
    balloon_keys = set()
    balloon_total = 0.0
    for loan_id, grp in sched.groupby("LoanID"):
        g = grp.sort_values("event_date")
        last = g.iloc[-1]
        if (last["ending_balance"] < 1.0 and last["principal"] > 0
                and len(g) > 1 and g.iloc[-2]["ending_balance"] > 0):
            balloon_keys.add((loan_id, last["event_date"]))
            balloon_total += float(last["principal"])

    sched["is_balloon"] = [
        (r.LoanID, r.event_date) in balloon_keys for r in sched.itertuples()
    ]
    sched["principal_op"] = sched.apply(
        lambda r: 0.0 if r["is_balloon"] else float(r["principal"]), axis=1)

    if balloon_total:
        notes.append(
            f"A balloon of {balloon_total:,.0f} matures in this window and is EXCLUDED — "
            f"it is repaid from sale proceeds, not operating cash.")

    variable = [lo for lo in loan_objs if getattr(lo, "is_variable", lambda: False)()]
    if variable:
        notes.append(
            f"{len(variable)} variable-rate loan(s) are modeled interest-only for their "
            f"full term (loans.py). Immaterial over one year; material over an "
            f"appraiser's ten.")

    children = [c for c in _deal_vcodes(vcode, data) if c.lower() != str(vcode).lower()]
    if children:
        notes.append(f"Includes debt on {len(children)} child property(ies).")

    by_month = (sched.groupby("event_date", as_index=False)[["interest", "principal_op"]]
                .sum().rename(columns={"principal_op": "principal"}))
    by_month = by_month.sort_values("event_date")

    return {
        "rows": [{"period": pd.Timestamp(r["event_date"]).strftime("%Y-%m-%d"),
                  "interest": round(float(r["interest"]), 2),
                  "principal": round(float(r["principal"]), 2)}
                 for _, r in by_month.iterrows()],
        "interest": round(float(by_month["interest"].sum()), 2),
        "principal": round(float(by_month["principal"].sum()), 2),
        "loan_count": len(loan_objs),
        "balloon_excluded": round(balloon_total, 2),
        "notes": notes,
        "available": True,
    }


def _empty(reason: str, start: date, end: date) -> Dict[str, Any]:
    """No modeled debt service, and WHY — never a bare zero.

    A 0 that means "we could not model this" is indistinguishable from a deal with no
    debt, and the second is a real and different thing.
    """
    return {"rows": [], "interest": None, "principal": None, "loan_count": 0,
            "balloon_excluded": 0.0, "notes": [reason], "available": False}


def for_year(vcode: str, year: int, data: dict) -> Dict[str, Any]:
    """Calendar-year totals — what the comparison's Interest and Principal rows want."""
    return monthly_schedule(vcode, date(year, 1, 1), date(year, 12, 31), data)
