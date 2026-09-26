"""What an analyst supplies to the Budget Review beyond the two spreadsheets.

Asset management, Sep 25 2026, three asks that are all inputs rather than
calculations:

* Override a 2026 Estimate cell, visibly -- "if the 2026 estimate is substantially
  off from reality, we don't have to show a number that we know is not going to be
  achieved and therefore don't have a large variance when comparing 2026 to 2027."
* Take the Budget column's debt service from underwriting instead of the model.
* Budgeted occupancy, read off a row the team adds to the budget import.

None of these is a second engine. An override REPLACES a figure the engine
computed and keeps the computed one beside it; the debt-service basis chooses
between two existing sources; occupancy is a stored input. The arithmetic that
turns them into totals, NOI and DSCR stays in `valuation_service.get_budget_review`.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import text

logger = logging.getLogger(__name__)

DEBT_BASES = ("modeled", "underwriting")


def _now():
    return datetime.utcnow()


def _ensure(engine):
    from flask_app.services.valuation_service import ensure_valuation_tables
    ensure_valuation_tables(engine)


# ──────────────────────────────────────────────────────────────────────────────
# Estimate overrides
# ──────────────────────────────────────────────────────────────────────────────

def overridable_rows() -> List[str]:
    """The Estimate rows an analyst may override: LINE ITEMS ONLY.

    Totals, NOI, Total Debt Service and DSCR are computed from the line items, so
    overriding one would leave a column that no longer adds up with nothing saying
    which figure to believe. Override the line; the totals follow.
    """
    import config
    rows: List[str] = []
    for section in ("REVENUES", "EXPENSES", "OTHER_BTL"):
        rows.extend(config.IS_ACCOUNTS.get(section, {}).keys())
    rows.extend(["Interest Expense", "Principal Payments"])
    return rows


def get_overrides(engine, record_id: int) -> Dict[str, Dict[str, Any]]:
    _ensure(engine)
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT row_label, amount, computed_amount, note, updated_by, updated_at
            FROM valuation_estimate_overrides WHERE record_id = :r
        """), {"r": record_id}).fetchall()
    return {r[0]: {"amount": float(r[1]),
                   "computed_amount": None if r[2] is None else float(r[2]),
                   "note": r[3], "updated_by": r[4],
                   "updated_at": str(r[5]) if r[5] is not None else None}
            for r in rows}


def set_override(engine, record_id: int, row_label: str, amount: Optional[float],
                 computed_amount: Optional[float], note: Optional[str],
                 username: str) -> Dict[str, Any]:
    """Store (or, with amount None, clear) an override for one Estimate row.

    Refused on an approved record, and on a row that is not a line item.
    """
    from flask_app.services.valuation_service import _require_not_approved
    _ensure(engine)
    _require_not_approved(engine, record_id)
    if row_label not in overridable_rows():
        raise ValueError(f"'{row_label}' cannot be overridden — only line items can; "
                         f"totals, NOI and DSCR are computed from them.")
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM valuation_estimate_overrides "
                          "WHERE record_id = :r AND row_label = :l"),
                     {"r": record_id, "l": row_label})
        if amount is None:
            return {"status": "cleared", "row": row_label}
        conn.execute(text("""
            INSERT INTO valuation_estimate_overrides
                (record_id, row_label, amount, computed_amount, note, updated_by, updated_at)
            VALUES (:r, :l, :a, :c, :n, :u, :t)
        """), {"r": record_id, "l": row_label, "a": float(amount),
               "c": None if computed_amount is None else float(computed_amount),
               "n": (note or "").strip() or None, "u": username, "t": _now()})
    return {"status": "saved", "row": row_label, "amount": float(amount)}


# ──────────────────────────────────────────────────────────────────────────────
# Debt-service basis
# ──────────────────────────────────────────────────────────────────────────────

def get_debt_basis(engine, record_id: int) -> str:
    _ensure(engine)
    with engine.connect() as conn:
        row = conn.execute(text("SELECT debt_service_basis FROM valuation_records "
                                "WHERE id = :i"), {"i": record_id}).fetchone()
    basis = (row[0] if row else None) or "modeled"
    return basis if basis in DEBT_BASES else "modeled"


def set_debt_basis(engine, record_id: int, basis: str, username: str) -> Dict[str, Any]:
    from flask_app.services.valuation_service import _require_not_approved
    if basis not in DEBT_BASES:
        raise ValueError(f"Unknown debt service basis '{basis}'. Expected one of {DEBT_BASES}.")
    _ensure(engine)
    _require_not_approved(engine, record_id)
    with engine.begin() as conn:
        conn.execute(text("UPDATE valuation_records SET debt_service_basis = :b WHERE id = :i"),
                     {"b": basis, "i": record_id})
    logger.info("Valuation record %s: debt service basis -> %s (%s)", record_id, basis, username)
    return {"status": "saved", "basis": basis}


# ──────────────────────────────────────────────────────────────────────────────
# Budgeted occupancy
# ──────────────────────────────────────────────────────────────────────────────

_OCC_LABEL = re.compile(r"occupan|\bocc\.?\s*%|\bocc\b", re.I)


def is_occupancy_label(label) -> bool:
    return bool(label) and bool(_OCC_LABEL.search(str(label)))


def to_pct(v) -> Optional[float]:
    """0.95, 95, '95%', '95.0 %' -> 95.0. None for anything that is not a number."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        n = float(v)
        if n != n:        # NaN
            return None
        pct_marked = False
    else:
        s = str(v).strip()
        if not s:
            return None
        pct_marked = s.endswith("%")
        try:
            n = float(s.rstrip("%").replace(",", "").strip())
        except ValueError:
            return None
    if not pct_marked and abs(n) <= 1.5:
        n *= 100.0        # stored as a fraction, which is how Excel holds a % cell
    return n


def normalise_occupancy(by_period: Dict[str, float]) -> Dict[str, Any]:
    """Validate the monthly percentages read off the sheet.

    Anything outside 0-100 is refused rather than clipped: a clipped 950% reads as a
    plausible 100% and nobody would ever see that the row was mis-scaled.
    """
    good = {p: round(v, 4) for p, v in by_period.items() if 0 <= v <= 100}
    bad = sorted(p for p, v in by_period.items() if not 0 <= v <= 100)
    return {"by_period": good, "rejected_periods": bad}


def save_occupancy(engine, vcode: str, by_period: Dict[str, float], source_file: str,
                   username: str) -> int:
    """Replace this deal's budgeted occupancy for the months in THIS file."""
    _ensure(engine)
    with engine.begin() as conn:
        for period, pct in by_period.items():
            conn.execute(text("DELETE FROM valuation_budget_occupancy "
                              "WHERE vcode = :v AND period = :p"), {"v": vcode, "p": period})
            conn.execute(text("""
                INSERT INTO valuation_budget_occupancy
                    (vcode, period, occupancy_pct, source_file, updated_by, updated_at)
                VALUES (:v, :p, :o, :f, :u, :t)
            """), {"v": vcode, "p": period, "o": float(pct), "f": source_file,
                   "u": username, "t": _now()})
    return len(by_period)


def budget_occupancy_quarters(engine, vcode: str, year: int) -> List[Dict[str, Any]]:
    """Quarterly average of the stored monthly budget, labelled like the MRI history
    (`2027-Q1`). A quarter with no months is omitted, never shown as 0%."""
    _ensure(engine)
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT period, occupancy_pct FROM valuation_budget_occupancy "
                                 "WHERE vcode = :v"), {"v": vcode}).fetchall()
    buckets: Dict[str, List[float]] = {}
    for period, pct in rows:
        p = str(period)[:10]
        try:
            y, m = int(p[:4]), int(p[5:7])
        except ValueError:
            continue
        if y != year:
            continue
        buckets.setdefault(f"{y}-Q{(m - 1) // 3 + 1}", []).append(float(pct))
    return [{"quarter": q, "occupancy": sum(v) / len(v), "months": len(v), "budgeted": True}
            for q, v in sorted(buckets.items())]
