"""A loan that matures before the deal sells, and what is missing because of it.

THE DEFECT THIS EXISTS TO MAKE VISIBLE. The amortization schedule ends at
maturity. If the deal sells later, every month between the two carries no debt
service at all -- so the forecast distributes cash that would really have gone
to the lender, and the balance simply stops being anywhere. Nothing said so.

Jefferson Waters Creek, measured Sep 16 2026 against a manually entered sale
date of 2027-04-30: the schedule ends 2026-11-30, the month before the
2026-12-05 maturity, with $51,667,000 outstanding and interest-only payments of
about $318,673 a month. December through April is five months and roughly
$1.59M of interest that the model never charges. Jim: "the forecast simply
stopped paying debt service at loan maturity without prompting us to address the
loan extension and any conditions or loan covenants that may be required."

WHAT THIS DOES NOT DO, ON PURPOSE: it does not extend the loan. Exercising an
extension is a business decision with conditions attached -- Waters Creek's
options are conditioned on covenant tests -- and a model that quietly assumed
the extension was taken would replace one invented answer with another. The only
honest output while the decision is open is the size of the hole and the facts
needed to close it.

WHAT IT REPORTS, per loan that matures early:
  * the balance left outstanding at maturity;
  * the months between maturity and the sale, and an ESTIMATE of the interest
    not charged over them;
  * the extension options MRI carries (``ExtensionOptions``, e.g. "2x12"),
    parsed into count and months, and HOW MANY of them it would take to reach
    the sale date -- often fewer than all of them;
  * the covenant fields on the loan, unchanged and unjudged, so the analyst sees
    what the extension is conditioned on.

THE INTEREST FIGURE IS AN ESTIMATE AND IS LABELLED ONE. It is taken from the
schedule's own trailing months rather than from ``nRate``, which is null on this
very loan -- the modelled rate is assembled elsewhere from index and spread.
Reading the schedule uses the number the model actually charged; reading nRate
would have produced nothing at all here.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

#: "2x12", "1X24", "2 x 12", "(2) 12-month" -- MRI's own spellings.
_EXT_PATTERNS = (
    re.compile(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*$"),
    re.compile(r"^\s*\((\d+)\)\s*(\d+)\s*-?\s*month", re.I),
    re.compile(r"^\s*(\d+)\s*(?:options?|ext\w*)\s*(?:of|at|x)?\s*(\d+)\s*-?\s*month", re.I),
)


def parse_extension_options(raw) -> dict:
    """``"2x12"`` -> two options of twelve months each.

    An unrecognised value is reported as unparsed rather than guessed at: the
    number of extensions available decides whether a maturity can reach the sale
    date at all, and inventing it would answer the question this module exists
    to ask.
    """
    s = ("" if raw is None else str(raw)).strip()
    if not s or s.upper() in ("NA", "N/A", "NAN", "NONE", "NAT", "NULL", "0"):
        return {"raw": s, "count": 0, "months": 0, "parsed": True,
                "total_months": 0}
    for pat in _EXT_PATTERNS:
        m = pat.match(s)
        if m:
            c, mo = int(m.group(1)), int(m.group(2))
            return {"raw": s, "count": c, "months": mo, "parsed": True,
                    "total_months": c * mo}
    return {"raw": s, "count": None, "months": None, "parsed": False,
            "total_months": None}


def _as_date(v) -> Optional[date]:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    try:
        d = pd.to_datetime(v, errors="coerce")
    except Exception:
        return None
    if d is None or (isinstance(d, float) and pd.isna(d)) or pd.isna(d):
        return None
    return d.date() if hasattr(d, "date") else None


def _add_months(d: date, months: int) -> date:
    y, m = divmod((d.year * 12 + d.month - 1) + months, 12)
    m += 1
    day = d.day
    while day > 1:
        try:
            return date(y, m, day)
        except ValueError:
            day -= 1
    return date(y, m, 1)


def _months_between(a: date, b: date) -> float:
    return round(((b - a).days) / 30.4375, 1)


def _num(v):
    try:
        f = float(v)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def _maturity_of(rl: dict):
    """The loan's maturity date, which is NOT in ``dtMaturity``.

    MRI carries the date in ``dtEvent`` on the row whose ``vDateType`` is
    "Maturity". Measured on production Sep 16 2026: ``dtMaturity`` is empty on
    all 91 loan rows, while ``dtEvent`` is populated on all 91 and on all 83 of
    the Maturity rows. Jefferson Waters Creek's is 2026-12-05 there, the date
    the deal team quotes.

    Reading ``dtMaturity`` therefore found nothing for every loan in the
    portfolio and fell back to the schedule's last period -- which is close
    enough to look right, and wrong enough to compute an extension from the
    wrong base date. ``dtMaturity`` is still read second in case it is ever
    populated; the fallback to the schedule stays last.
    """
    vdt = str(rl.get("vdatetype") or "").strip().lower()
    if vdt == "maturity":
        d = _as_date(rl.get("dtevent"))
        if d:
            return d
    return _as_date(rl.get("dtmaturity")) or _as_date(rl.get("dtevent"))


def detect(loan_sched, loans_raw, vcode: str, sale_date) -> dict:
    """Loans on this deal that mature before it sells.

    ``loan_sched`` is the modelled amortization (the engine's own output);
    ``loans_raw`` is the MRI loans table, for maturities, extension options and
    covenants. Returns ``{"has_gap": False, ...}`` when there is nothing to say.
    """
    empty = {"has_gap": False, "loans": [], "vcode": vcode,
             "sale_date": None, "totals": {}}
    sale = _as_date(sale_date)
    if sale is None:
        return empty
    empty["sale_date"] = sale.isoformat()

    sched = loan_sched
    if sched is None or (hasattr(sched, "empty") and sched.empty):
        return empty
    if not isinstance(sched, pd.DataFrame):
        try:
            sched = pd.DataFrame(sched)
        except Exception:
            return empty
    if sched.empty or "event_date" not in sched.columns:
        return empty

    s = sched.copy()
    s["_d"] = pd.to_datetime(s["event_date"], errors="coerce")
    s = s.dropna(subset=["_d"])
    if s.empty:
        return empty

    lr = loans_raw.copy() if loans_raw is not None and not getattr(
        loans_raw, "empty", True) else pd.DataFrame()
    lm = {str(c).lower(): c for c in lr.columns}
    if not lr.empty and lm.get("vcode"):
        lr = lr[lr[lm["vcode"]].astype(str).str.strip().str.upper()
                == str(vcode).strip().upper()]

    id_col = "LoanID" if "LoanID" in s.columns else (
        "loan_id" if "loan_id" in s.columns else None)
    groups = s.groupby(id_col) if id_col else [(None, s)]

    out: List[dict] = []
    for loan_id, g in groups:
        g = g.sort_values("_d")
        last = g.iloc[-1]
        last_date = last["_d"].date()
        balance = _num(last.get("ending_balance"))

        # THE SCHEDULE ENDING BEFORE THE SALE IS THE SIGNAL, not the stored
        # maturity date: the schedule is what the forecast actually charged, and
        # it is the thing that stops. A maturity field that disagrees with it is
        # reported alongside rather than used instead.
        if last_date >= sale:
            continue
        # A loan that amortised to zero on schedule is repaid, not a gap.
        if balance is not None and abs(balance) < 1.0:
            continue

        row = {}
        if not lr.empty and lm.get("loanid") is not None and loan_id is not None:
            m = lr[lr[lm["loanid"]].astype(str).str.strip()
                   == str(loan_id).strip()]
            if not m.empty:
                row = m.iloc[0].to_dict()
        if not row and not lr.empty:
            row = lr.iloc[0].to_dict()
        rl = {str(k).lower(): v for k, v in row.items()}

        maturity = _maturity_of(rl) or last_date
        ext = parse_extension_options(rl.get("extensionoptions"))

        # How many of the options it would take to reach the sale -- usually
        # fewer than all of them, which is the practical question.
        needed, reach = None, None
        if ext.get("parsed") and (ext.get("count") or 0) > 0:
            for n in range(1, int(ext["count"]) + 1):
                cand = _add_months(maturity, n * int(ext["months"]))
                if cand >= sale:
                    needed, reach = n, cand
                    break
            if needed is None:
                reach = _add_months(maturity, int(ext["total_months"]))

        # Interest the model did not charge, estimated from what it DID charge
        # in the months before maturity. `nRate` is null on this very loan.
        tail = g.tail(3)
        monthly = None
        if "interest" in g.columns and len(tail):
            vals = [v for v in (_num(x) for x in tail["interest"]) if v]
            if vals:
                monthly = sum(vals) / len(vals)
        months_gap = _months_between(last_date, sale)
        est_interest = round(monthly * months_gap, 2) if monthly else None

        out.append({
            "loan_id": None if loan_id is None else str(loan_id),
            "schedule_ends": last_date.isoformat(),
            "maturity": maturity.isoformat(),
            # Reported, never silently preferred -- see above.
            "maturity_disagrees_with_schedule": maturity.isoformat() != last_date.isoformat(),
            "balance_outstanding": balance,
            "months_unmodelled": months_gap,
            "monthly_interest_estimate": round(monthly, 2) if monthly else None,
            "interest_unmodelled_estimate": est_interest,
            "extension": {
                **ext,
                "options_needed_to_reach_sale": needed,
                "maturity_if_exercised": reach.isoformat() if reach else None,
                "reaches_sale_date": bool(needed),
            },
            # Unchanged and unjudged. Which field is the EXTENSION test and
            # which is the ongoing one is a question for the deal team, not an
            # assumption to bake in here.
            "covenants": {
                "required_dcr": _num(rl.get("nrequireddcr")),
                "req_dsr": _num(rl.get("nreqdsr")),
                "ltv": _num(rl.get("nltv")),
                "required_ltv": _num(rl.get("nrequiredltv")),
                "debt_yield": _num(rl.get("ndy")),
                "required_debt_yield": _num(rl.get("nrequireddy")),
            },
        })

    if not out:
        return empty

    tot_int = sum(l["interest_unmodelled_estimate"] or 0 for l in out)
    tot_bal = sum(l["balance_outstanding"] or 0 for l in out)
    return {
        "has_gap": True,
        "vcode": vcode,
        "sale_date": sale.isoformat(),
        "loans": out,
        "totals": {
            "loan_count": len(out),
            "balance_outstanding": tot_bal,
            "interest_unmodelled_estimate": round(tot_int, 2) if tot_int else None,
            "max_months_unmodelled": max(l["months_unmodelled"] for l in out),
        },
        "headline": _headline(out, tot_bal, tot_int),
    }


def _headline(loans: List[dict], balance: float, interest: float) -> str:
    n = len(loans)
    worst = max(loans, key=lambda l: l["months_unmodelled"])
    parts = [
        "%d loan%s matures before this deal sells." % (n, "" if n == 1 else "s"),
        "The forecast charges no debt service for %.1f months and leaves "
        "%s outstanding." % (worst["months_unmodelled"],
                             _money(balance)),
    ]
    if interest:
        parts.append("That is roughly %s of interest the model does not charge, "
                     "so distributable cash is overstated by about that much."
                     % _money(interest))
    ext = worst.get("extension") or {}
    if ext.get("reaches_sale_date"):
        parts.append("MRI records %s; %d of them would carry the maturity to %s."
                     % (ext.get("raw"), ext["options_needed_to_reach_sale"],
                        ext["maturity_if_exercised"]))
    elif ext.get("parsed") and (ext.get("count") or 0) > 0:
        parts.append("MRI records %s, which even fully exercised only reaches %s."
                     % (ext.get("raw"), ext.get("maturity_if_exercised")))
    elif ext.get("raw") and not ext.get("parsed"):
        parts.append("MRI records extension options as %r, which could not be "
                     "read." % ext.get("raw"))
    else:
        parts.append("MRI records no extension options, so the loan has to be "
                     "repaid or refinanced at maturity.")
    parts.append("Nothing here assumes an extension is exercised.")
    return " ".join(parts)


def _money(v) -> str:
    try:
        return "${:,.0f}".format(float(v))
    except (TypeError, ValueError):
        return "-"
