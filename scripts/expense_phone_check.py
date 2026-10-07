"""Guardrail: the monthly cell phone reimbursement (Jim, Oct 7 2026).

"Every employee is entitled to a fixed monthly reimbursement for the use of their
personal cell phones ... $50/month ... The CFO should have control of the reimbursement
rate. ... an automatic monthly batch on the accounting side ... paired with a filter
that declines expensing cell phone bills." Decided: everyone in Expenses by default,
the CFO can exclude; from October 2026; cell phone bills only (internet still claimed).

  1. THE RATE is the CFO's: $50 from October 2026 to start; a later rate applies from
     its month; only the cfo role may set it (an accountant and an analyst are refused,
     the CFO is not); an implausible rate is refused; a rate for a month already paid
     is refused.
  2. WHO: employees set up in Expenses with a name, never the admin account; the CFO can
     switch one off or bound the months.
  3. EVERY MONTH OWED, ONCE: a payroll batch pays each month owed through its payroll
     month; a second batch pays nothing twice; a month a batch skipped is picked up by
     the next; voiding a batch releases its months. No report is needed.
  4. THE FILTER declines a cell phone bill dated October 2026 or later -- at save, and on
     the report check for a line read off a receipt -- with the note; a September bill,
     internet, and a phone charger are not declined.

Usage: python scripts/expense_phone_check.py [--inject=double|nofilter|everyone]
"""
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
INJECT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--inject=")), "")
_passed, _failed = [], []


def chk(label, cond, detail=None):
    (_passed if cond else _failed).append(label)
    print(("   ok   " if cond else "   FAIL ") + label + ("" if cond or detail is None else "   [%s]" % (detail,)))


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="expense_phone_")
    os.environ.pop("DATABASE_URL", None)
    os.environ["DB_PATH"] = os.path.join(tmp, "check.db")
    import jwt
    import pandas as pd
    from sqlalchemy import text
    from flask_app import create_app
    from flask_app.auth.models import create_user, list_users
    from flask_app.db import get_engine
    from flask_app.services import data_service, expense_phone as ph, expense_service as ex
    from database import PROTECTED_TABLES

    if INJECT == "double":
        ph.record_paid = lambda conn, items, batch_id: None
    if INJECT == "nofilter":
        ph.declined = lambda *a, **k: None
    if INJECT == "everyone":
        _orig_el = ph.eligibility
        ph.eligibility = lambda engine: [dict(e, eligible=True) for e in _orig_el(engine)]

    app = create_app()
    app.config["DATABASE_URL"] = None
    client = app.test_client()
    data_service.get_data = lambda *a, **k: {"inv": pd.DataFrame(columns=["vcode", "Investment_Name", "InvestmentID"])}
    people = {"admin": "admin", "cfo1": "cfo", "acct": "accountant", "ana": "analyst",
              "emp": "analyst", "emp2": "analyst", "nobody": "analyst"}
    with app.app_context():
        eng = get_engine()
        for u, role in people.items():
            create_user(u, "pw-" + u, role=role)
        with eng.begin() as c:
            c.execute(text('CREATE TABLE IF NOT EXISTS gl_accounts ("ACCTNUM" TEXT, "ACCTNAME" TEXT, "TYPE" TEXT)'))
            c.execute(text("INSERT INTO gl_accounts VALUES ('MR53000015', 'Other Expense: Telephone & Internet', 'I')"))
        ids = {u["username"]: u["id"] for u in list_users()}

    def H(name):
        return {"Authorization": "Bearer " + jwt.encode(
            {"sub": str(ids[name]), "username": name, "role": people[name],
             "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
            app.config["JWT_SECRET"], algorithm="HS256")}

    def call(method, path, who, body=None):
        r = client.open(path, method=method, json=body, headers=H(who))
        return r.status_code, (r.get_json(silent=True) or {})

    # Names on reports: emp, emp2 and admin; "nobody" has a login and no name.
    for u, n in (("emp", "Fred Kurz"), ("emp2", "Elaine Johnson"), ("admin", "Admin"), ("cfo1", "Joseph Stewart")):
        call("PUT", "/api/expenses/employees/%d" % ids[u], "admin", {"full_name": n, "approver_user_id": None})

    print("1. The rate is the CFO's")
    chk("the three tables are protected",
        all(t in PROTECTED_TABLES for t in ("er_phone_rates", "er_phone_eligibility", "er_phone_paid")))
    st, ph_view = call("GET", "/api/expenses/phone", "cfo1")
    chk("it starts at $50 a month from October 2026",
        st == 200 and [(r["effective_date"], r["rate"]) for r in ph_view["rates"]] == [("2026-10-01", 50.0)],
        ph_view.get("rates"))
    chk("the CFO may edit; the screen is told so", ph_view.get("can_edit") is True)
    chk("an accountant reads it but is told it is not theirs",
        call("GET", "/api/expenses/phone", "acct")[1].get("can_edit") is False)
    chk("an accountant cannot set the rate",
        call("PUT", "/api/expenses/phone/rate", "acct", {"effective_month": "2026-12", "rate": 60})[0] == 403)
    chk("an analyst cannot read it", call("GET", "/api/expenses/phone", "ana")[0] == 403)
    chk("an implausible rate is refused",
        call("PUT", "/api/expenses/phone/rate", "cfo1", {"effective_month": "2026-12", "rate": 5000})[0] == 400)
    st, r = call("PUT", "/api/expenses/phone/rate", "cfo1", {"effective_month": "2026-12", "rate": "55"})
    chk("the CFO sets $55 from December", st == 200 and r["rates"][0]["rate"] == 55.0, r)

    print("\n2. Who is paid")
    names = [e["full_name"] for e in call("GET", "/api/expenses/phone", "cfo1")[1]["employees"]]
    chk("employees with a name on reports, never the admin account or a nameless login",
        sorted(names) == ["Elaine Johnson", "Fred Kurz", "Joseph Stewart"], names)
    chk("the CFO switches Elaine off",
        call("PUT", "/api/expenses/phone/employees/%d" % ids["emp2"], "cfo1",
             {"eligible": False, "start_month": "2026-10"})[0] == 200)
    chk("a start before October 2026 is refused",
        call("PUT", "/api/expenses/phone/employees/%d" % ids["emp"], "cfo1",
             {"eligible": True, "start_month": "2026-08"})[0] == 400)
    chk("an accountant cannot change who is paid",
        call("PUT", "/api/expenses/phone/employees/%d" % ids["emp"], "acct", {"eligible": False})[0] == 403)
    with app.app_context():
        d = ph.due(get_engine(), "2026-12")
    got = sorted((i["employee"], i["month"], i["amount"]) for i in d["items"])
    chk("through December: Fred and Joseph, Oct-Nov at $50 and Dec at $55; not Elaine",
        got == sorted([(n, m, a) for n in ("Fred Kurz", "Joseph Stewart")
                       for m, a in (("2026-10", 50.0), ("2026-11", 50.0), ("2026-12", 55.0))]), got)

    print("\n3. Every month owed, once")
    body = {"report_ids": [], "payroll_date": "2026-10-30", "credit_suffix": "End of Month", "commit": True}
    st, b1 = call("POST", "/api/expense-coding/batches", "acct", body)
    lines = b1.get("lines") or []
    chk("an October batch with no reports pays October's cell phones",
        st == 200 and sorted((l["acctnum"], l["amount"]) for l in lines if l["acctnum"] == "MR53000015")
        == [("MR53000015", 50.0), ("MR53000015", 50.0)], (st, b1.get("errors"), lines))
    chk("...credited to payroll", [l["amount"] for l in lines if l["acctnum"] == "MR20000001"] == [-100.0])
    chk("...described ER, initials, Cell Phone Reimbursement, the month",
        "ER FK Cell Phone Reimbursement October 2026" in [l["descrpn"] for l in lines], [l["descrpn"] for l in lines])
    st, b2 = call("POST", "/api/expense-coding/batches", "acct", {**body, "payroll_date": "2026-10-31"})
    chk("a second October batch pays nobody twice",
        st == 400 or not [l for l in (b2.get("lines") or []) if l["acctnum"] == "MR53000015"], (st, b2.get("lines")))
    st, pv = call("POST", "/api/expense-coding/batches", "acct", {**body, "payroll_date": "2026-12-15", "commit": False})
    chk("December's batch picks up November AND December, at their own rates",
        sorted(l["amount"] for l in pv["lines"] if l["acctnum"] == "MR53000015") == [50.0, 50.0, 55.0, 55.0],
        [l["amount"] for l in pv["lines"] if l["acctnum"] == "MR53000015"])
    chk("a rate for a month already paid is refused",
        call("PUT", "/api/expenses/phone/rate", "cfo1", {"effective_month": "2026-10", "rate": 60})[0] == 400)
    call("POST", "/api/expense-coding/batches/%s/void" % b1["batch_id"], "acct")
    with app.app_context():
        again = ph.due(get_engine(), "2026-10")["items"]
    chk("voiding the October batch makes October owed again", len(again) == 2, again)

    print("\n4. The filter")
    for vendor, comment, want in (
            ("Verizon", "Monthly wireless service, 1 line", True),
            ("T-Mobile", "Essentials plan", True),
            ("", "cell phone bill - October", True),
            ("AT&T", "monthly bill", True),
            ("Verizon", "Fios home internet", False),
            ("Comcast", "home office internet", False),
            ("Best Buy", "iPhone charger", False),
            ("Staples", "printer paper", False)):
        chk(f"{vendor or '-'} / {comment}: {'declined' if want else 'allowed'}",
            ph.is_cell_phone_bill(vendor, comment) == want)
    st, rep = call("POST", "/api/expenses/reports", "emp", {"period_start": "2026-09-01", "period_end": "2026-10-31"})
    base = {"category_account": "MR53000015", "purpose": "Other", "deal_code": "OPERATIONS",
            "receipt": "N", "no_receipt_reason": "check", "amount": "103.46", "vendor": "Verizon",
            "comment": "Monthly wireless service"}
    st, r = call("POST", "/api/expenses/reports/%d/lines" % rep["id"], "emp", {**base, "line_date": "2026-10-07"})
    chk("an October cell phone bill is refused at save, with the note",
        st == 400 and "reimbursed automatically" in (r.get("error") or ""), (st, r))
    st, r = call("POST", "/api/expenses/reports/%d/lines" % rep["id"], "emp", {**base, "line_date": "2026-09-07"})
    chk("a September one is still claimed", st == 200, (st, r.get("error")))
    st, r = call("POST", "/api/expenses/reports/%d/lines" % rep["id"], "emp",
                 {**base, "line_date": "2026-10-07", "vendor": "Verizon", "comment": "Fios internet"})
    chk("October internet is still claimed", st == 200, (st, r.get("error")))
    with app.app_context():
        with get_engine().begin() as c:     # as a line read off a receipt arrives: never "saved"
            c.execute(text("INSERT INTO er_lines (report_id, sort_order, line_date, category_account, vendor, "
                           "comment, amount, receipt, updated_at) VALUES (:r, 99, '2026-10-12', 'MR53000015', "
                           "'T-Mobile', 'Monthly wireless service', 50, 'N', 'x')"), {"r": rep["id"]})
    st, r = call("GET", "/api/expenses/reports/%d" % rep["id"], "emp")
    chk("a cell phone line read off a receipt blocks submission, with the note",
        any("reimbursed automatically" in e for e in r["check"]["errors"]), r["check"]["errors"])

    print("\n%d passed, %d failed" % (len(_passed), len(_failed)))
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
