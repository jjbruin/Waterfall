"""Committed preferred equity, from MRI's IA_Commitment (the ``commitments``
table), as of a quarter.

ONE ENGINE. ``resolve_committed_pref`` is the only place the rule lives.
``one_pager.get_capitalization_stack``, ``one_pager.get_pe_performance`` and
``investment_metrics.capitalization_sources`` all call it, so the three cannot
disagree about the same fact. Before this they disagreed on twelve deals:
the One Pager summed accounting ``Typename='Commitment'`` rows while
Investment Metrics already read IA_Commitment first.

THE AS-OF RULE IS A DATE RANGE, NOT A QUARTER-END SNAPSHOT. A commitment is
revised by ENDING one row and opening the next the following day; measured on
the 80 deal-level chains in MRI, 30 pairs are contiguous, 0 have a gap and 1
overlaps. Only ONE ended row in the whole deal-level population lands on a
quarter end, and there is no ``09-30`` EndDate anywhere in the table. So the
row that applies on ``Q`` is the one in force on ``Q``:

    StartDate <= Q  AND  (EndDate IS NULL OR EndDate >= Q)

Measured against the sent 26Q2 TIAA report this reproduces Burton
(26,597,500), JB Fair Park (14,300,000) and Nottingham (9,135,000) — the three
figures that had been typed into that PDF by hand.

NON-OP ONLY. There is no tranche column on this table; the pref side is
identified by the investor NOT being an operating partner, the same test
``one_pager`` already applies to split ``pref_equity`` from ``partner_equity``.
In practice the pref investor is a ``PPI*`` vehicle, and 74 of the 77 deals
carrying rows have exactly one such chain.

WHAT HAPPENS WHEN NOTHING IS IN EFFECT MATTERS MORE THAN THE RULE.
``queries/MRI_Commitments.sql`` still filters ``EndDate IS NULL``, so ENDED
rows never reach the app. Until that changes, a deal whose current row starts
after the quarter (JB Fair Park's begins 2026-07-30) has NO row in effect at
26Q2 — and falling through to funded pref would have moved its printed figure
by -22,850,000 to a number matching neither today, the report, nor the answer
the query change will give. So the fallback KEEPS THE ACCOUNTING FIGURE: this
module is a no-op for such a deal until the data arrives, and corrects itself
automatically when it does. Charlene's call, 2026-10-01.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional, Tuple

import pandas as pd

#: Rows that are corrections rather than commitments: a row opened and closed
#: on the same day for nothing. Eleven exist, all 0.00 or 0.01. Summing them
#: makes a chain look populated when it carries no pledge.
TOMBSTONE_MAX_ABS = 0.01

#: A chain whose every row is MRI back-filling a commitment record for an
#: uploaded transaction is not a pledge register. Apple - Bales Drive is the
#: live case: its only two rows are auto-generated artifacts totalling 170,179
#: against an accounting commitment of 4,172,975. Reading them would publish a
#: 96% fall driven entirely by back-fill, so the accounting figure is kept and
#: the basis says the question is open.
AUTO_NOTE = "auto generated"


def norm_id(value: Any) -> str:
    """Strip and upper-case an MRI identifier; nullish spellings become ''."""
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower() in ("nan", "none", "nat", "<na>"):
        return ""
    return s.upper()


def is_op(investor_id: Any) -> bool:
    """Operating-partner side. The pref side is everything else."""
    return norm_id(investor_id).startswith("OP")


def _as_date(value: Any) -> Optional[date]:
    # NULLS ARE GUARDED BEFORE ANY isinstance TEST, and the order is the whole
    # point. `pd.NaT` IS an instance of `datetime` (`isinstance(pd.NaT,
    # datetime)` is True), so an isinstance branch placed first returns
    # `NaT.date()`, which is NaT, and the caller's `end >= as_of` then raises
    # "Cannot compare NaT with datetime.date object".
    #
    # This was unreachable until 2026-10-01. While MRI_Commitments.sql filtered
    # `EndDate IS NULL` the column was entirely null, so pandas typed it object
    # or float, NaN is NOT a datetime, and the pd.to_datetime path below
    # returned None correctly. The moment ended rows loaded the column became
    # datetime64 and every OPEN row's EndDate arrived as NaT. It took the One
    # Pager down on every 26Q3 deal. A fixture using Python `None` for an open
    # row cannot catch it -- the test has to pass pd.NaT.
    if value is None:
        return None
    try:
        if pd.isna(value):          # pd.NaT, NaN, pd.NA
            return None
    except (TypeError, ValueError):
        pass                        # not a scalar pandas understands; carry on
    if value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        ts = pd.to_datetime(value, errors="coerce")
    except Exception:
        return None
    if ts is None or pd.isna(ts):
        return None
    try:
        return ts.date()
    except Exception:
        return None


def _amount(value: Any) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if pd.isna(f) else f


def deal_commitment_rows(commitments: Optional[pd.DataFrame],
                         investment_id: str) -> list:
    """The deal's NON-OP commitment rows, tombstones dropped.

    Returns dicts rather than a frame: callers walk chains, and the frames this
    is handed are sometimes raw MRI (``EntityID`` space-padded) and sometimes
    pre-normalised.
    """
    if commitments is None or getattr(commitments, "empty", True):
        return []
    if "EntityID" not in commitments.columns:
        return []
    # A deal may map to several InvestmentIDs; all of them are this deal.
    if isinstance(investment_id, (list, tuple, set)):
        wanted = {norm_id(x) for x in investment_id if norm_id(x)}
    else:
        wanted = {norm_id(investment_id)} - {""}
    if not wanted:
        return []
    out = []
    for r in commitments.to_dict("records"):
        if norm_id(r.get("EntityID")) not in wanted:
            continue
        inv = norm_id(r.get("InvestorID"))
        if is_op(inv):
            continue
        sd, ed = _as_date(r.get("StartDate")), _as_date(r.get("EndDate"))
        amt = _amount(r.get("Amount"))
        if sd is not None and ed is not None and sd == ed \
                and abs(amt) <= TOMBSTONE_MAX_ABS:
            continue                       # same-day correction, not a pledge
        out.append({
            "entity": norm_id(r.get("EntityID")),
            "investor": inv, "amount": amt, "start": sd, "end": ed,
            "uid": r.get("CommitmentUID"),
            "auto": AUTO_NOTE in str(r.get("TransactionNote") or "").lower(),
        })
    return out


def row_in_effect(rows: list, as_of: date) -> Optional[dict]:
    """The row in force on ``as_of`` for one chain, or None.

    On the measured data a chain yields at most one; where revisions overlap
    (one pair does) the LATEST start wins, which is the revision.
    """
    live = [r for r in rows
            if r["start"] is not None and r["start"] <= as_of
            and (r["end"] is None or r["end"] >= as_of)]
    if not live:
        return None
    live.sort(key=lambda r: (r["start"], str(r["uid"])))
    return live[-1]


def resolve_committed_pref(
    commitments: Optional[pd.DataFrame],
    investment_id: str,
    as_of: Any,
    accounting_committed: Optional[float] = None,
    sold_and_dropped: bool = False,
) -> Tuple[Optional[float], str]:
    """Committed pref for one deal as of ``as_of``, with the basis.

    ``accounting_committed`` is today's figure (the sum of the deal's
    ``Typename='Commitment'`` accounting rows). It is used ONLY when the
    commitments table has nothing to say — see the module docstring.

    ``sold_and_dropped`` is for a sold deal that is NOT kept on the report: it
    carries no commitment after the sale. A KEPT sold deal passes False and is
    resolved at the quarter its stack is read, which the caller supplies as
    ``as_of`` — the commitment in effect on the last held quarter.

    Returns ``(amount_or_None, basis)``. NEVER 0 for "no pledge on file": a
    zero is indistinguishable from a real zero to every consumer, and the
    Snapshot would print "$0.0M committed" for a deal carrying real pref.
    """
    if sold_and_dropped:
        return None, "sold — no commitment carried after disposal"

    q = _as_date(as_of)
    rows = deal_commitment_rows(commitments, investment_id)
    if not rows:
        if accounting_committed:
            return (float(accounting_committed),
                    "no commitments row; accounting figure kept")
        return None, "no commitment row"

    if all(r["auto"] for r in rows):
        # Back-fill artifacts only — not a pledge register for this deal.
        if accounting_committed:
            return (float(accounting_committed), "pending accounting")
        return None, "pending accounting"

    if q is None:
        return None, "no quarter supplied"

    # One chain per (entity, investor): a deal with several InvestmentIDs can
    # carry the same investor on more than one, and they are separate pledges.
    chains: dict = {}
    for r in rows:
        chains.setdefault((r["entity"], r["investor"]), []).append(r)

    total, parts = 0.0, []
    for key in sorted(chains):
        pick = row_in_effect(chains[key], q)
        if pick is None:
            continue
        total += pick["amount"]
        parts.append(f"{key[1]} uid {pick['uid']} "
                     f"({pick['start']}..{pick['end'] or 'open'})")

    if not parts:
        # Every chain's current row begins after the quarter, which today means
        # the row that WAS in effect is an ended one we do not load.
        if accounting_committed:
            return (float(accounting_committed),
                    "no commitments row in effect; accounting figure kept")
        return None, "no commitments row in effect"

    return total, "commitments in effect " + "; ".join(parts)
