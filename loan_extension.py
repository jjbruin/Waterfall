"""Does the loan clear its extension test, and if not, by how much?

THE QUESTION. Jefferson Waters Creek matures 2026-12-05 with two twelve-month
options, and the deal sells 2027-04-30. Taking one option covers the sale, but
the option is CONDITIONED: the lender tests coverage at the extension date. Jim,
Sep 17 2026: "1.10 is the extension test." So the question a deal team actually
asks is not "may we extend" but "what would we have to pay down to be allowed
to".

THE MATH IS NOT NEW AND IS NOT REWRITTEN HERE. `planned_loans` already sizes a
loan against LTV, DSCR and debt yield and reports which one binds; those
primitives -- `twelve_month_noi_after_date`, `projected_cap_rate_at_date`,
`solve_principal_from_annual_ds` -- are used directly. What differs is the
ORCHESTRATION: sizing a new loan asks "how much can we borrow", an extension
test asks "is the balance we ALREADY have small enough", and the answer is a
paydown rather than a loan amount. Jim, Sep 15 2026, on a different screen:
"why are you trying to recreate a calculation engine that we have already built
and vetted?" -- so nothing below re-derives a constraint.

SEEDED FROM MRI, THEN NEGOTIABLE. Jim: "consider the scenario where we are
negotiating potential changes to the existing covenants. We would like to model
beginning with the existing covenant and then making changes from there." Every
test value starts at what MRI carries and can be overridden; each one reports
which it is, so a printed answer always says whether it rests on the lender's
number or on a proposal.

THE RATE IS DERIVED FROM THE SCHEDULE, because `nRate` is null on this loan --
the modelled rate is assembled from index and spread elsewhere. Interest over
balance, annualised, off the months the model actually charged.

NOTHING HERE CHANGES THE FORECAST. It reports a required paydown; it does not
apply one, and it does not extend the loan. Both remain decisions.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

#: Which MRI column seeds each test. `nReqDSR` is the EXTENSION coverage test
#: (Jim, Sep 17 2026); `nRequiredDCR` is the ongoing covenant and is NOT used
#: here. Naming them in one place so the screen and the solve cannot drift.
SEED_FIELDS = {
    "min_dscr": ("nreqdsr", "Extension DSCR (nReqDSR)"),
    "max_ltv": ("nltv", "Max LTV (nLTV)"),
    "min_debt_yield": ("nrequireddy", "Min debt yield (nRequiredDY)"),
}


def _num(v) -> Optional[float]:
    try:
        f = float(v)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def derive_rate(loan_sched, loan_id, balance) -> Optional[float]:
    """The loan's annual rate, from what the model actually charged.

    ``nRate`` is null on Waters Creek, so the stored field answers nothing. The
    schedule's trailing interest over the outstanding balance, annualised, is
    the rate the forecast itself used -- which is the rate the coverage test
    should be run at.
    """
    if loan_sched is None or getattr(loan_sched, "empty", True) or not balance:
        return None
    df = loan_sched
    if "LoanID" in df.columns and loan_id is not None:
        df = df[df["LoanID"].astype(str) == str(loan_id)]
    if df.empty or "interest" not in df.columns:
        return None
    df = df.sort_values("event_date") if "event_date" in df.columns else df
    vals = [v for v in (_num(x) for x in df["interest"].tail(3)) if v]
    if not vals:
        return None
    monthly = sum(vals) / len(vals)
    r = (monthly * 12.0) / float(balance)
    return round(r, 6) if r > 0 else None


def seed_tests(covenants: dict) -> dict:
    """The lender's own numbers, as the starting point for a negotiation.

    A value MRI does not carry is None, never 0 -- a zero DSCR test would read
    as "no coverage required" and pass everything.
    """
    out = {}
    for key, (col, label) in SEED_FIELDS.items():
        raw = covenants.get({"nreqdsr": "req_dsr", "nltv": "ltv",
                             "nrequireddy": "required_debt_yield"}[col])
        v = _num(raw)
        # MRI stores LTV and debt yield as decimals here (0.55), but a user
        # typing "55" means the same thing. Normalised on the way in only for
        # the ratios, never for DSCR, where 1.10 and 110 are not the same.
        if v is not None and key in ("max_ltv", "min_debt_yield") and v > 1.5:
            v = v / 100.0
        out[key] = {"value": v, "source": "MRI" if v is not None else None,
                    "label": label, "field": col}
    return out


def test(loan, fc_deal_full, mri_val, vcode, loan_sched,
         overrides: dict = None) -> dict:
    """Run the extension test for one loan from `loan_maturity.detect`.

    Returns the constraint that binds and the paydown required to clear it, or
    a reason it could not be computed. Never a zero standing in for unknown.
    """
    overrides = overrides or {}
    cov = loan.get("covenants") or {}
    tests = seed_tests(cov)
    for k, v in overrides.items():
        if k not in tests:
            continue
        nv = _num(v)
        if nv is not None and k in ("max_ltv", "min_debt_yield") and nv > 1.5:
            nv = nv / 100.0
        tests[k] = {**tests[k], "value": nv,
                    "source": "proposed" if nv is not None else None}

    from planned_loans import (projected_cap_rate_at_date,
                               twelve_month_noi_after_date)

    try:
        test_date = date.fromisoformat(loan["maturity"])
    except (KeyError, ValueError, TypeError):
        return {"available": False,
                "reason": "The loan has no readable maturity date."}

    balance = _num(loan.get("balance_outstanding"))
    if not balance:
        return {"available": False,
                "reason": "No outstanding balance at maturity to test."}

    # FORWARD 12 MONTHS FROM THE EXTENSION DATE, which is the period the lender
    # is being asked to lend into -- not the trailing year and not the year of
    # the sale.
    noi = None
    try:
        noi = twelve_month_noi_after_date(fc_deal_full, test_date)
    except Exception as e:
        logger.warning("extension test NOI failed for %s: %s", vcode, e)
    noi = _num(noi)
    if not noi or noi <= 0:
        return {"available": False, "test_date": test_date.isoformat(),
                "reason": ("The forecast carries no positive NOI for the twelve "
                           "months after %s, so no coverage test can be run."
                           % test_date.isoformat())}

    rate = derive_rate(loan_sched, loan.get("loan_id"), balance)

    cap_rate, value = None, None
    try:
        cr = _num(projected_cap_rate_at_date(mri_val, str(vcode), test_date))
        if cr and cr > 0:
            cap_rate, value = cr, noi / cr
    except Exception as e:
        logger.warning("extension test cap rate failed for %s: %s", vcode, e)

    return _solve(loan, tests, noi, rate, cap_rate, value, balance, test_date)


def _solve(loan, tests, noi, rate, cap_rate, value, balance, test_date) -> dict:
    """The constraint arithmetic, shared by the baseline and every what-if.

    Kept separate so a proposed covenant is re-solved from the SAME inputs the
    baseline used -- the same NOI, the same derived rate, the same cap rate. If
    the what-if recomputed NOI from the forecast it could land on a slightly
    different figure, and the analyst would be comparing two answers that differ
    for a reason nobody intended.
    """
    constraints = []

    # DSCR. Interest-only is the case here, and it is also the conservative
    # reading: an amortising loan's debt service is higher, so solving it as IO
    # would overstate what the balance can carry. Amortisation is used when the
    # schedule shows it.
    d = tests["min_dscr"]["value"]
    if d and rate:
        max_ds = noi / d
        max_loan = max_ds / rate
        constraints.append({
            "key": "dscr", "label": "Extension DSCR",
            "test": d, "source": tests["min_dscr"]["source"],
            "max_balance": max_loan, "detail":
                "NOI %s / DSCR %.2f = %s of debt service, at %.2f%% supports %s"
                % (_m(noi), d, _m(max_ds), rate * 100, _m(max_loan))})
    elif d and not rate:
        constraints.append({
            "key": "dscr", "label": "Extension DSCR", "test": d,
            "source": tests["min_dscr"]["source"], "max_balance": None,
            "detail": "Cannot test: the loan's rate could not be derived."})

    # LTV
    lt = tests["max_ltv"]["value"]
    if lt and value:
        constraints.append({
            "key": "ltv", "label": "Max LTV",
            "test": lt, "source": tests["max_ltv"]["source"],
            "max_balance": value * lt, "detail":
                "NOI %s / cap %.2f%% = value %s, at %.0f%% LTV supports %s"
                % (_m(noi), cap_rate * 100, _m(value), lt * 100,
                   _m(value * lt))})
    elif lt and not value:
        constraints.append({
            "key": "ltv", "label": "Max LTV", "test": lt,
            "source": tests["max_ltv"]["source"], "max_balance": None,
            "detail": "Cannot test: no cap rate for this deal at %s."
                      % test_date.isoformat()})

    # Debt yield
    dy = tests["min_debt_yield"]["value"]
    if dy:
        constraints.append({
            "key": "debt_yield", "label": "Min debt yield",
            "test": dy, "source": tests["min_debt_yield"]["source"],
            "max_balance": noi / dy, "detail":
                "NOI %s / %.2f%% supports %s" % (_m(noi), dy * 100, _m(noi / dy))})

    testable = [c for c in constraints if c["max_balance"] is not None]
    if not testable:
        return {"available": False, "test_date": test_date.isoformat(),
                "noi": noi, "rate": rate, "balance": balance,
                "constraints": constraints, "tests": tests,
                "reason": ("No extension test can be evaluated. "
                           + " ".join(c["detail"] for c in constraints
                                      if "Cannot test" in c["detail"]))}

    # THE BINDING ONE IS THE SMALLEST, because every test must be satisfied at
    # once -- the same rule `size_prospective_loan` applies to a new loan.
    binding = min(testable, key=lambda c: c["max_balance"])
    supportable = binding["max_balance"]
    paydown = max(0.0, balance - supportable)

    return {
        "available": True,
        "test_date": test_date.isoformat(),
        "noi": noi,
        "rate": rate,
        "cap_rate": cap_rate,
        "value": value,
        "balance": balance,
        "tests": tests,
        "constraints": constraints,
        "binding": binding["key"],
        "max_supportable_balance": supportable,
        "required_paydown": paydown,
        "passes": paydown <= 0.0,
        "headline": _headline(loan, binding, balance, supportable, paydown,
                              test_date),
    }


def _headline(loan, binding, balance, supportable, paydown, test_date) -> str:
    ext = loan.get("extension") or {}
    src = " (proposed)" if binding.get("source") == "proposed" else ""
    if paydown <= 0:
        return ("At %s the balance of %s clears the binding test, %s of %.2f%s "
                "— it supports %s. No paydown required on these assumptions."
                % (test_date.isoformat(), _m(balance), binding["label"],
                   binding["test"], src, _m(supportable)))
    n = ext.get("options_needed_to_reach_sale")
    tail = ""
    if n:
        tail = (" Exercising %d of the %s option%s would carry the maturity to %s."
                % (n, ext.get("raw"), "" if n == 1 else "s",
                   ext.get("maturity_if_exercised")))
    return ("To clear %s of %.2f%s at %s, the balance has to come down from %s "
            "to %s — a paydown of %s.%s"
            % (binding["label"], binding["test"], src, test_date.isoformat(),
               _m(balance), _m(supportable), _m(paydown), tail))


def _m(v) -> str:
    try:
        return "${:,.0f}".format(float(v))
    except (TypeError, ValueError):
        return "-"


def resolve(prior: dict, overrides: dict = None, loan: dict = None) -> dict:
    """Re-run the tests on a PRIOR result with proposed covenant values.

    Takes the baseline's own NOI, rate, cap rate and balance rather than
    recomputing them, so the only thing that moves between two answers is the
    covenant being negotiated. Needs no forecast and no database read, which is
    also why the what-if endpoint does not have to cache a DataFrame per deal.
    """
    if not prior or not prior.get("available"):
        return prior or {"available": False,
                         "reason": "No baseline extension test to vary."}
    overrides = overrides or {}
    tests = {k: dict(v) for k, v in (prior.get("tests") or {}).items()}
    for k, v in overrides.items():
        if k not in tests:
            continue
        nv = _num(v)
        if nv is not None and k in ("max_ltv", "min_debt_yield") and nv > 1.5:
            nv = nv / 100.0
        tests[k] = {**tests[k], "value": nv,
                    "source": "proposed" if nv is not None else None}
    try:
        test_date = date.fromisoformat(prior["test_date"])
    except (KeyError, ValueError, TypeError):
        return {"available": False, "reason": "The baseline has no test date."}
    return _solve(loan or {}, tests, prior.get("noi"), prior.get("rate"),
                  prior.get("cap_rate"), prior.get("value"),
                  prior.get("balance"), test_date)
