"""The monthly cell phone reimbursement, paid automatically (Jim, Oct 7 2026).

"Every employee is entitled to a fixed monthly reimbursement for the use of their
personal cell phones for business. The current rate is $50/month per employee which may
be updated over time. The CFO should have control of the reimbursement rate." Decided
the same day: every employee in Expenses is on by default and the CFO can switch one off
or set start / end months; it starts with October 2026; and a cell phone BILL claimed on
a report from then on is declined, with a note that it is paid automatically.

  * THE RATE is dated, like the mileage rate, and set by the CFO only. The rate for a
    month is the latest one effective on or before the 1st of that month.
  * EACH PAYROLL BATCH PAYS every month owed, once. `er_phone_paid` is the ledger --
    (employee, month) is its key -- so a month is never paid twice, a month skipped by
    one batch is picked up by the next, and voiding a batch releases its months.
  * A month with no rate in force is NOT paid and is said, never paid at $0.
  * THE FILTER (`is_cell_phone_bill`) declines a line naming a wireless carrier or a
    cell phone service, dated in or after the first month -- home or office internet on
    the same Telephone & Internet category is still claimable.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Dict, List, Optional

from sqlalchemy import text

from flask_app.services import expense_service as ex

#: The first month paid automatically (Jim, Oct 7 2026). Claims before it stand.
FIRST_MONTH = "2026-10"
#: Telephone & Internet, at PSC Manager -- the account cell phone lines were coded to.
PHONE_ACCOUNT = "MR53000015"
#: Jim's figure on Oct 7 2026, written once as the first rate so the program does not
#: start unpaid; every change after it is the CFO's, dated, on the Expenses setup tab.
INITIAL_RATE = 50.0
#: Who may set the rate and switch employees on or off.
RATE_ROLES = ("cfo",)

_DONE: set = set()


def ensure_tables(engine) -> None:
    ex.ensure_tables(engine)
    key = str(getattr(engine, "url", "")) or id(engine)
    if key in _DONE:
        return
    with engine.begin() as c:
        c.execute(text("""CREATE TABLE IF NOT EXISTS er_phone_rates (
            effective_date TEXT PRIMARY KEY,
            rate           DOUBLE PRECISION NOT NULL,
            set_by         TEXT,
            set_at         TEXT)"""))
        c.execute(text("""CREATE TABLE IF NOT EXISTS er_phone_eligibility (
            user_id     INTEGER PRIMARY KEY,
            eligible    BOOLEAN NOT NULL DEFAULT TRUE,
            start_month TEXT,
            end_month   TEXT,
            updated_by  TEXT,
            updated_at  TEXT)"""))
        c.execute(text("""CREATE TABLE IF NOT EXISTS er_phone_paid (
            user_id  INTEGER NOT NULL,
            month    TEXT NOT NULL,
            batch_id TEXT NOT NULL,
            amount   DOUBLE PRECISION NOT NULL,
            PRIMARY KEY (user_id, month))"""))
        if not c.execute(text("SELECT COUNT(*) FROM er_phone_rates")).scalar():
            c.execute(text("INSERT INTO er_phone_rates (effective_date, rate, set_by, set_at) "
                           "VALUES (:d, :r, :by, :at)"),
                      {"d": FIRST_MONTH + "-01", "r": INITIAL_RATE,
                       "by": "initial rate (Jim, Oct 7 2026)", "at": ex._now()})
    _DONE.add(key)


# ------------------------------------------------------------------ months

def _month(v, label: str) -> Optional[str]:
    s = (str(v or "")).strip()[:7]
    if not s:
        return None
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", s):
        raise ValueError("%s must be a month (YYYY-MM), not %r." % (label, v))
    return s


def _months(first: str, last: str) -> List[str]:
    y, m = int(first[:4]), int(first[5:7])
    out = []
    while (y, m) <= (int(last[:4]), int(last[5:7])):
        out.append("%04d-%02d" % (y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def month_label(month: str) -> str:
    return date(int(month[:4]), int(month[5:7]), 1).strftime("%B %Y")


# ------------------------------------------------------------------ the rate

def rates(engine) -> List[dict]:
    ensure_tables(engine)
    with engine.connect() as c:
        return [dict(r) for r in c.execute(text(
            "SELECT * FROM er_phone_rates ORDER BY effective_date DESC")).mappings().all()]


def rate_for_month(engine, month: str, _rates=None) -> Optional[dict]:
    first = month + "-01"
    for r in (_rates if _rates is not None else rates(engine)):
        if r["effective_date"] <= first:
            return r
    return None


def set_rate(engine, effective_month, rate, by: str) -> List[dict]:
    """A new monthly rate, from the first of a month. Money, so rounded to the cent."""
    ensure_tables(engine)
    month = _month(effective_month, "The month it takes effect")
    if not month:
        raise ValueError("Say which month the rate takes effect.")
    try:
        r = round(float(str(rate).replace("$", "").replace(",", "").strip()), 2)
    except ValueError:
        raise ValueError("Rate %r is not a number." % (rate,))
    if not (0 < r <= 500):
        raise ValueError("A cell phone reimbursement of $%s a month is not plausible." % r)
    with engine.connect() as c:
        paid = c.execute(text("SELECT COUNT(*) FROM er_phone_paid WHERE month >= :m"),
                         {"m": month}).scalar()
    if paid:
        raise ValueError("%d reimbursement(s) for %s or later are already in a batch at the old "
                         "rate. Choose a later month, or void those batches first."
                         % (paid, month_label(month)))
    with engine.begin() as c:
        c.execute(text("DELETE FROM er_phone_rates WHERE effective_date = :d"), {"d": month + "-01"})
        c.execute(text("INSERT INTO er_phone_rates (effective_date, rate, set_by, set_at) "
                       "VALUES (:d, :r, :by, :at)"),
                  {"d": month + "-01", "r": r, "by": by, "at": ex._now()})
    return rates(engine)


# ------------------------------------------------------------------ who

def eligibility(engine) -> List[dict]:
    """Every employee in Expenses, with whether and when they are paid. No row = on, from
    the first month."""
    ensure_tables(engine)
    with engine.connect() as c:
        rows = {int(r["user_id"]): dict(r) for r in c.execute(text(
            "SELECT * FROM er_phone_eligibility")).mappings().all()}
    from flask_app.auth.sections import SUPERUSER
    out = []
    # EMPLOYEES = set up in Expenses with a name on their reports (`er_employees`), not
    # every user: the admin account and board-only advisors have logins, not phones.
    for e in ex.employees(engine):
        if not (e.get("full_name") or "").strip() or e.get("username") == SUPERUSER:
            continue
        r = rows.get(int(e["user_id"])) or {}
        out.append({"user_id": e["user_id"], "full_name": e.get("full_name") or e.get("username"),
                    "eligible": bool(r.get("eligible", True)),
                    "start_month": r.get("start_month") or FIRST_MONTH,
                    "end_month": r.get("end_month"),
                    "updated_by": r.get("updated_by"), "updated_at": r.get("updated_at")})
    return out


def set_eligibility(engine, user_id: int, body: dict, by: str) -> List[dict]:
    ensure_tables(engine)
    if int(user_id) not in {int(e["user_id"]) for e in eligibility(engine)}:
        raise LookupError("User %s is not an employee set up in Expenses." % user_id)
    start = _month(body.get("start_month"), "Start month") or FIRST_MONTH
    end = _month(body.get("end_month"), "End month")
    if start < FIRST_MONTH:
        raise ValueError("The automatic reimbursement starts with %s." % month_label(FIRST_MONTH))
    if end and end < start:
        raise ValueError("The end month is before the start month.")
    with engine.begin() as c:
        c.execute(text("DELETE FROM er_phone_eligibility WHERE user_id = :u"), {"u": int(user_id)})
        c.execute(text("INSERT INTO er_phone_eligibility (user_id, eligible, start_month, end_month, "
                       "updated_by, updated_at) VALUES (:u, :e, :s, :en, :by, :at)"),
                  {"u": int(user_id), "e": bool(body.get("eligible", True)), "s": start, "en": end,
                   "by": by, "at": ex._now()})
    return eligibility(engine)


# ------------------------------------------------------------------ what is owed

def due(engine, through_month: str) -> Dict[str, list]:
    """Every (employee, month) owed through ``through_month`` and not yet in a batch.

    Returns {"items": [...], "problems": [...]}; a month with no rate in force is a
    problem, not a $0 payment."""
    ensure_tables(engine)
    with engine.connect() as c:
        paid = {(int(r[0]), r[1]) for r in c.execute(text("SELECT user_id, month FROM er_phone_paid"))}
    rs = rates(engine)
    items, problems = [], []
    for e in eligibility(engine):
        if not e["eligible"]:
            continue
        last = min(through_month, e["end_month"]) if e["end_month"] else through_month
        if e["start_month"] > last:
            continue
        for m in _months(e["start_month"], last):
            if (int(e["user_id"]), m) in paid:
                continue
            r = rate_for_month(engine, m, rs)
            if not r:
                problems.append("No cell phone rate is in force for %s, so %s is not paid for it."
                                % (month_label(m), e["full_name"]))
                continue
            items.append({"user_id": int(e["user_id"]), "employee": e["full_name"], "month": m,
                          "amount": float(r["rate"])})
    items.sort(key=lambda x: (x["employee"] or "", x["month"]))
    return {"items": items, "problems": problems}


def record_paid(conn, items: List[dict], batch_id: str) -> None:
    """Inside the batch's own transaction: a month someone else just paid fails the key."""
    for it in items:
        conn.execute(text("INSERT INTO er_phone_paid (user_id, month, batch_id, amount) "
                          "VALUES (:u, :m, :b, :a)"),
                     {"u": it["user_id"], "m": it["month"], "b": batch_id, "a": it["amount"]})


def release(conn, batch_id: str) -> int:
    return conn.execute(text("DELETE FROM er_phone_paid WHERE batch_id = :b"), {"b": batch_id}).rowcount


# ------------------------------------------------------------------ the filter

_CARRIER = re.compile(r"\b(verizon|at\s*&\s*t|att\s+wireless|t[\s-]?mobile|sprint|cricket|"
                      r"metro\s*pcs|boost\s+mobile|mint\s+mobile|visible|us\s+cellular|google\s+fi|"
                      r"tracfone|straight\s+talk|xfinity\s+mobile|spectrum\s+mobile)\b", re.I)
_PHONE = re.compile(r"\b(cell(ular)?(\s+phone)?|mobile\s+(phone|service|plan|bill)|"
                    r"wireless\s+(service|plan|bill|phone)|phone\s+(bill|plan|service)|iphone|smartphone)\b", re.I)
_INTERNET = re.compile(r"\b(fios|internet|broadband|wi-?fi|router|modem|home\s+office\s+internet)\b", re.I)


def is_cell_phone_bill(vendor, comment) -> bool:
    """A cell phone BILL, by the carrier or the words -- not a phone charger, not internet.

    A carrier alone counts unless the line is plainly internet ("Verizon Fios"); the
    words count on their own ("monthly cell phone"). Accessories and devices bought for
    the business are not bills: "charger", "case" and "repair" are left alone.
    """
    s = " ".join(str(x or "") for x in (vendor, comment))
    if re.search(r"\b(charger|case|cable|screen protector|repair|adapter|headset)\b", s, re.I):
        return False
    if _PHONE.search(s):
        return True
    return bool(_CARRIER.search(s)) and not _INTERNET.search(s)


def declined(line_date, vendor, comment, engine=None) -> Optional[str]:
    """The note a declined line carries, or None if the line may be claimed."""
    if not line_date or str(line_date)[:7] < FIRST_MONTH:
        return None
    if not is_cell_phone_bill(vendor, comment):
        return None
    r = rate_for_month(engine, str(line_date)[:7]) if engine is not None else None
    amount = (" ($%.2f a month)" % float(r["rate"])) if r else ""
    return ("is a cell phone bill -- cell phones are reimbursed automatically%s from %s, so "
            "it is not claimed on a report. Remove this line." % (amount, month_label(FIRST_MONTH)))
