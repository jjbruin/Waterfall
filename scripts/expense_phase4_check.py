"""Guardrail: expense reports, phase 4 -- recurring lines, duplicates, accounting's return.

Each rule is asserted in BOTH directions, because each has an obvious way to
be satisfied vacuously:
  * "recurring lines are copied" is satisfied by copying everything -- so an
    unmarked line must NOT come across, and pressing it twice adds nothing;
  * "duplicates are flagged" is satisfied by flagging everything -- so $50 of
    phone on the same day from two different vendors, and a colleague's DRAFT,
    must NOT be flagged;
  * "accounting can return a report" is satisfied by letting anyone -- so an
    analyst, a batched report, and a return with no reason must all be refused,
    and the returned report must need its approver again.
"""
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_passed, _failed = [], []


def chk(label, cond, detail=""):
    (_passed if cond else _failed).append(label)
    print("   %s %s%s" % ("ok  " if cond else "FAIL", label,
                          ("   [%s]" % (detail,)) if detail and not cond else ""))


def main():
    tmp = tempfile.mkdtemp(prefix="expense_p4_")
    os.environ.pop("DATABASE_URL", None)
    os.environ["DB_PATH"] = os.path.join(tmp, "check.db")

    import jwt
    import pandas as pd
    from sqlalchemy import text
    from flask_app import create_app
    from flask_app.auth.models import create_user, list_users
    from flask_app.db import get_engine
    from flask_app.services import data_service

    app = create_app()
    app.config["DATABASE_URL"] = None
    client = app.test_client()
    data_service.get_data = lambda *a, **k: {"inv": pd.DataFrame([
        {"vcode": "P0000001", "Investment_Name": "Apple Self Storage", "Portfolio_Name": "",
         "Sale_Status": "", "Sale_Date": None, "Lifecycle": "Stable"}])}
    people = {"admin": "admin", "emp": "analyst", "peer": "analyst", "mgr": "analyst",
              "acct": "accountant", "ana": "analyst"}
    with app.app_context():
        for u, role in people.items():
            create_user(u, "pw-" + u, role=role)
        with get_engine().begin() as c:
            c.execute(text('CREATE TABLE IF NOT EXISTS gl_accounts ("ACCTNUM" TEXT, "ACCTNAME" TEXT, "TYPE" TEXT)'))
            c.execute(text("INSERT INTO gl_accounts VALUES ('MR53000015', 'Other Expense: Telephone & Internet', 'I')"))
            c.execute(text("INSERT INTO gl_accounts VALUES ('MR53000011', 'Other Expense: Travel', 'I')"))
        ids = {u["username"]: u["id"] for u in list_users()}

    def H(name):
        return {"Authorization": "Bearer " + jwt.encode(
            {"sub": str(ids[name]), "username": name, "role": people[name],
             "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
            app.config["JWT_SECRET"], algorithm="HS256")}

    def call(method, path, who, body=None):
        r = client.open("/api/expenses" + path, method=method, json=body, headers=H(who))
        return r.status_code, (r.get_json(silent=True) or {})

    for u in ("emp", "peer"):
        call("PUT", "/employees/%d" % ids[u], "admin", {"full_name": u.title(), "approver_user_id": ids["mgr"]})

    def report(who, start, end):
        return call("POST", "/reports", who, {"period_start": start, "period_end": end})[1]["id"]

    PHONE = {"line_date": "2026-08-01", "category_account": "MR53000015", "purpose": "Company Related",
             "deal_code": "OPERATIONS", "vendor": "Verizon", "comment": "Cell Phone Reimbursement",
             "amount": "50", "receipt": "N", "no_receipt_reason": "monthly allowance"}
    TAXI = {"line_date": "2026-08-12", "category_account": "MR53000011", "purpose": "Property Visit - Existing",
            "deal_code": "P0000001", "vendor": "Yellow Cab Co.", "comment": "Taxi to site", "amount": "42.10",
            "receipt": "N", "no_receipt_reason": "driver had no printer"}

    print("1. Recurring lines come forward; nothing else does")
    aug = report("emp", "2026-08-01", "2026-08-31")
    call("POST", "/reports/%d/lines" % aug, "emp", {**PHONE, "recurring": True})
    call("POST", "/reports/%d/lines" % aug, "emp", TAXI)
    sep = report("emp", "2026-09-01", "2026-09-30")
    st, b = call("POST", "/reports/%d/copy-recurring" % sep, "emp")
    lines = b.get("lines", [])
    chk("the recurring line is copied", st == 200 and len(lines) == 1
        and lines[0]["comment"] == "Cell Phone Reimbursement" and lines[0]["amount"] == 50, b.get("error"))
    chk("an unmarked line is NOT", not [x for x in lines if x["comment"] == "Taxi to site"])
    chk("the copy is dated at the new period's start", lines and lines[0]["line_date"] == "2026-09-01")
    chk("the copy carries no receipt -- this month's is a different file",
        lines and lines[0]["receipt_id"] is None and lines[0]["receipt"] is None)
    chk("the copy stays recurring, so it comes forward again", lines and bool(lines[0]["recurring"]))
    st, b = call("POST", "/reports/%d/copy-recurring" % sep, "emp")
    chk("pressing it twice adds nothing", len(b.get("lines", [])) == 1 and b["copied"]["added"] == 0)
    first = report("peer", "2026-09-01", "2026-09-30")
    chk("with no earlier recurring line it says so",
        call("POST", "/reports/%d/copy-recurring" % first, "peer")[0] == 400)
    chk("another employee cannot copy into this report",
        call("POST", "/reports/%d/copy-recurring" % sep, "peer")[0] == 404)

    print("\n2. Duplicates: same vendor, date and amount on another report")
    dup = report("emp", "2026-08-15", "2026-08-31")
    st, b = call("POST", "/reports/%d/lines" % dup, "emp", TAXI)
    chk("the same taxi on the employee's other report is flagged, naming it",
        any("your report #%d" % aug in w for w in b["check"]["warnings"]), b["check"]["warnings"])
    chk("...as a warning, not an error", not b["check"]["errors"], b["check"]["errors"])
    st, b = call("POST", "/reports/%d/lines" % dup, "emp", {**TAXI, "vendor": "Yellow  Cab, Co"})
    chk("vendor spelling and punctuation do not hide it",
        sum("duplicate" in w for w in b["check"]["warnings"]) == 2, b["check"]["warnings"])
    st, b = call("POST", "/reports/%d/lines" % dup, "emp", {**TAXI, "amount": "42.11"})
    chk("a different amount is not flagged", sum("duplicate" in w for w in b["check"]["warnings"]) == 2)
    pr = report("peer", "2026-08-01", "2026-08-31")
    st, b = call("POST", "/reports/%d/lines" % pr, "peer", PHONE)
    chk("a colleague's line matching only an employee's DRAFT is not flagged (a draft is private)",
        not any("duplicate" in w for w in b["check"]["warnings"]), b["check"]["warnings"])
    call("POST", "/reports/%d/submit" % aug, "emp")
    st, b = call("GET", "/reports/%d" % pr, "peer")
    chk("once that report is submitted, the colleague's line is flagged -- without naming it",
        any("another employee's report" in w for w in b["check"]["warnings"])
        and not any("#%d" % aug in w for w in b["check"]["warnings"]), b["check"]["warnings"])
    st, b = call("POST", "/reports/%d/lines" % pr, "peer", {**PHONE, "vendor": "AT&T"})
    chk("$50 on the same day to a DIFFERENT vendor is not a duplicate",
        sum("duplicate" in w for w in b["check"]["warnings"]) == 1, b["check"]["warnings"])

    print("\n3. Accounting returns an approved report, with a reason")
    st, _ = call("POST", "/reports/%d/decide" % aug, "mgr", {"action": "approve"})
    chk("an analyst cannot", call("POST", "/reports/%d/accounting-return" % aug, "ana",
                                  {"note": "x"})[0] == 403)
    chk("accounting must say why", call("POST", "/reports/%d/accounting-return" % aug, "acct",
                                        {"note": " "})[0] == 400)
    st, b = call("GET", "/reports/%d" % aug, "acct")
    chk("accounting is offered the return", b["permissions"]["accounting_return"] is True)
    st, b = call("POST", "/reports/%d/accounting-return" % aug, "acct",
                 {"note": "Split the taxi between deals"})
    chk("accounting returns it", st == 200 and b.get("returned") == aug, b)
    st, b = call("GET", "/reports/%d" % aug, "emp")
    chk("it is the employee's again, with accounting's note in its history",
        b["status"] == "returned" and b["permissions"]["edit"]
        and any(e["note"] == "Split the taxi between deals" for e in b["events"]))
    st, b = call("POST", "/reports/%d/submit" % aug, "emp")
    chk("resubmitted, it needs its approver again -- an approved figure cannot be changed unseen",
        b.get("status") == "submitted" and b.get("route_kind") == "approver", b.get("error"))
    call("POST", "/reports/%d/decide" % aug, "mgr", {"action": "approve"})

    print("\n4. ...but not once it is in a batch")
    from flask_app.services import expense_coding as ec
    with app.app_context():
        with get_engine().begin() as c:
            ec.ensure_tables(get_engine())
            c.execute(text("UPDATE er_reports SET status = 'batched', batch_id = 'ER-TEST' WHERE id = :i"),
                      {"i": aug})
    chk("a batched report cannot be returned -- voiding the batch is how",
        call("POST", "/reports/%d/accounting-return" % aug, "acct", {"note": "late"})[0] == 403)
    st, b = call("GET", "/reports/%d" % aug, "acct")
    chk("...and the return is not offered", st == 200 and b["permissions"]["accounting_return"] is False)

    print("\n%d passed, %d failed" % (len(_passed), len(_failed)))
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
