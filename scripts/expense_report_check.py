"""Guardrail: expense reports, phase 1 -- who approves, who sees, what is refused.

Jim, Oct 2 2026: an employee's report goes to their approver; "CFO can approve
the approvers"; "CEO or President can approve any expense report in the event
an approver is out"; and the section access process stays as built -- the Admin
sets who opens the section, so every rule here is about RECORDS inside it.

Driven through the real Flask app against a fresh SQLite database, with tokens
minted the way `accounting_access_check` does, so no password is typed.

EVERY NARROWING IS ASSERTED IN BOTH DIRECTIONS. "Another employee cannot read
the draft" is satisfied by a screen nobody can read, so the owner reading it is
asserted beside it; "the CEO may approve in place of the approver" beside "the
CEO may not approve their own".
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
                          ("   [%s]" % detail) if detail and not cond else ""))


def main():
    tmp = tempfile.mkdtemp(prefix="expense_check_")
    os.environ.pop("DATABASE_URL", None)
    os.environ["DB_PATH"] = os.path.join(tmp, "check.db")

    import jwt
    from sqlalchemy import text
    from flask_app import create_app
    from flask_app.auth import sections as S
    from flask_app.auth.models import create_user, list_users
    from flask_app.db import get_engine
    from database import PROTECTED_TABLES

    app = create_app()
    app.config["DATABASE_URL"] = None
    client = app.test_client()

    people = {"admin": "admin", "emp": "analyst", "mgr": "analyst", "other": "analyst",
              "cfo": "cfo", "acct": "accountant", "ceo": "analyst", "pres": "analyst"}
    with app.app_context():
        eng = get_engine()
        for u, role in people.items():
            create_user(u, "pw-" + u, role=role)
        ids = {u["username"]: u["id"] for u in list_users()}
        from flask_app.services import review_service
        review_service._ensure_tables()
        with eng.begin() as c:
            c.execute(text('CREATE TABLE IF NOT EXISTS gl_accounts ("ACCTNUM" TEXT, "ACCTNAME" TEXT, "TYPE" TEXT)'))
            c.execute(text("DELETE FROM gl_accounts"))
            for a, n, t in (("MR53000000", "Other Expenses", "M"),
                            ("MR53000004", "Other Expense: Meals & Entertainment", "I"),
                            ("MR53000011", "Other Expense: Travel", "I"),
                            ("MR11000012", "Deal Cost Receivable", "B")):
                c.execute(text('INSERT INTO gl_accounts VALUES (:a, :n, :t)'), {"a": a, "n": n, "t": t})
            c.execute(text('CREATE TABLE IF NOT EXISTS deals (vcode TEXT, "Investment_Name" TEXT, '
                           '"Portfolio_Name" TEXT, "Lifecycle" TEXT)'))
            for row in (("P1", "Apple Self Storage", None, "Stable"),
                        ("P2", "Sold Plaza", None, "Sold"),
                        ("P3", "Child Parcel", "Parent Deal", "Income"),
                        ("P4", "Gallery of New Hampshire", None, "Income")):
                c.execute(text("INSERT INTO deals (vcode, \"Investment_Name\", \"Portfolio_Name\", "
                               "\"Lifecycle\") VALUES (:a, :b, :c, :d)"),
                          dict(zip("abcd", row)))
        from sqlalchemy import inspect
        if not inspect(eng).has_table("prospect_deals"):
            with eng.begin() as c:
                c.execute(text("CREATE TABLE prospect_deals (vcode TEXT, deal_name TEXT, "
                               "stage TEXT, onboarded_vcode TEXT)"))
        with eng.begin() as c:
            for vc, n, st in (("N1", "Windsor Square", "due_diligence"), ("N2", "Dead Deal", "passed")):
                c.execute(text("INSERT INTO prospect_deals (vcode, deal_name, stage) VALUES (:v, :n, :s)"),
                          {"v": vc, "n": n, "s": st})
            c.execute(text("INSERT INTO review_roles (user_id, review_role) VALUES (:u, 'ceo')"),
                      {"u": ids["ceo"]})
            c.execute(text("INSERT INTO review_roles (user_id, review_role) VALUES (:u, 'president')"),
                      {"u": ids["pres"]})

    def H(name):
        return {"Authorization": "Bearer " + jwt.encode(
            {"sub": str(ids[name]), "username": name, "role": people[name],
             "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
            app.config["JWT_SECRET"], algorithm="HS256")}

    def call(method, path, who, body=None):
        r = client.open("/api/expenses" + path, method=method, json=body, headers=H(who))
        return r.status_code, (r.get_json(silent=True) or {})

    def new_report(who, start="2026-09-01", end="2026-09-30"):
        st, b = call("POST", "/reports", who, {"period_start": start, "period_end": end})
        assert st == 200, b
        return b["id"]

    GOOD = {"line_date": "2026-09-10", "category_account": "MR53000011",
            "purpose": "Property Visit - Existing", "deal_code": "P1", "vendor": "American",
            "comment": "Apple Site Visit - Airfare", "amount": "712.98", "receipt": "Y"}

    def add(rid, who="emp", **kw):
        return call("POST", "/reports/%d/lines" % rid, who, {**GOOD, **kw})

    print("\n1. Registry: a section, a protected and restricted table family")
    chk("Expenses is a section", "expenses" in S.SECTION_KEYS)
    chk("/api/expenses belongs to the Expenses section",
        S.sections_for_api("/api/expenses/reports") == ("expenses",))
    tabs = ("er_employees", "er_reports", "er_lines", "er_line_splits", "er_events",
            "er_mileage_rates")
    chk("all six er_ tables are protected", all(t in PROTECTED_TABLES for t in tabs))
    chk("er_ tables are restricted to Accounting on raw-table paths",
        all(S.table_section(t) == "accounting" for t in tabs))

    print("\n2. The form's lists")
    st, o = call("GET", "/options", "emp")
    accts = [c["account"] for c in o["categories"]]
    codes = [d["code"] for d in o["deals"]]
    chk("categories are MR5 income-statement accounts", accts == ["MR53000004", "MR53000011"], accts)
    chk("a roll-up header (TYPE M) is not a category", "MR53000000" not in accts)
    chk("Operations is offered first", codes[:1] == ["OPERATIONS"])
    chk("an owned deal is offered", "P1" in codes and "P4" in codes)
    chk("a sold deal and a child property are not", "P2" not in codes and "P3" not in codes)
    chk("a live pipeline deal is offered, a passed one is not", "N1" in codes and "N2" not in codes)
    chk("the purposes are accounting's six", len(o["purposes"]) == 6)

    print("\n3. Setup: the admin sets approvers; nobody approves themselves")
    rid = new_report("emp")
    add(rid)
    st, b = call("POST", "/reports/%d/submit" % rid, "emp")
    chk("submit with no approver is refused, saying why", st == 400 and "approver" in b.get("error", ""), b)
    for who in ("emp", "acct", "cfo"):
        st, _ = call("PUT", "/employees/%d" % ids["emp"], who, {"approver_user_id": ids["mgr"]})
        chk("%s cannot set an approver" % who, st == 403, st)
    st, _ = call("PUT", "/employees/%d" % ids["emp"], "admin",
                 {"full_name": "Erin Employee", "approver_user_id": ids["emp"]})
    chk("an employee cannot be their own approver", st == 400)
    st, b = call("PUT", "/employees/%d" % ids["emp"], "admin",
                 {"full_name": "Erin Employee", "approver_user_id": ids["mgr"]})
    chk("the admin sets an approver", st == 200 and b["route"]["approver_user_id"] == ids["mgr"], b)
    call("PUT", "/employees/%d" % ids["mgr"], "admin", {"full_name": "Max Manager"})
    call("PUT", "/employees/%d" % ids["acct"], "admin", {"full_name": "Ann Acct", "approver_user_id": ids["cfo"]})
    call("PUT", "/employees/%d" % ids["other"], "admin", {"approver_user_id": ids["mgr"]})
    st, emps = call("GET", "/employees", "acct")
    route = {e["username"]: e["route"] for e in emps.get("employees", [])}
    chk("accounting may read the employee list", st == 200)
    chk("an analyst may not", call("GET", "/employees", "emp")[0] == 403)
    chk("an employee's report goes to their approver", route["emp"]["kind"] == "approver")
    chk("an approver's own report goes to the CFO", route["mgr"]["kind"] == "cfo", route["mgr"])
    chk("the CFO's own report goes to the CEO or President",
        route["cfo"]["kind"] == "ceo_president", route["cfo"])

    print("\n4. Lines: mileage computed, splits foot, a missing receipt explains itself")
    st, b = add(rid, miles="85", amount="999")
    chk("mileage with no rate in force is refused", st == 400 and "rate" in b.get("error", ""), b)
    chk("an analyst cannot set the rate",
        call("PUT", "/mileage-rates", "emp", {"effective_date": "2026-01-01", "rate": "0.725"})[0] == 403)
    st, _ = call("PUT", "/mileage-rates", "acct", {"effective_date": "2026-01-01", "rate": "0.725"})
    chk("accounting sets the rate", st == 200)
    st, b = add(rid, miles="85", amount="999", comment="Mileage to airport 85 miles")
    ml = [x for x in b["lines"] if x.get("miles")][0]
    chk("the amount is miles x rate, rounded once, the typed amount ignored",
        ml["amount"] == round(85 * 0.725, 2) and ml["mileage_rate"] == 0.725, ml["amount"])
    st, b = add(rid, splits=[{"deal_code": "P1", "amount": "50"}, {"deal_code": "P4", "amount": "40"}],
                amount="100")
    chk("a split that does not foot is saved as a draft", st == 200)
    chk("...and named as what blocks submit", any("split 90.00" in e for e in b["check"]["errors"]),
        b["check"]["errors"])
    bad_line = [x for x in b["lines"] if x["splits"]][0]["id"]
    st, b = call("PUT", "/reports/%d/lines/%d" % (rid, bad_line), "emp",
                 {**GOOD, "amount": "100", "splits": [{"deal_code": "P1", "amount": "60"},
                                                      {"deal_code": "P4", "amount": "40"}]})
    split_line = [x for x in b["lines"] if x["id"] == bad_line][0]
    chk("a split that foots is clean", not b["check"]["by_line"][str(bad_line)]["errors"]
        if str(bad_line) in b["check"]["by_line"] else not b["check"]["by_line"][bad_line]["errors"])
    chk("a split line carries no deal of its own", split_line["deal_code"] is None)
    st, b = add(rid, receipt="N")
    chk("no receipt and no reason blocks submit",
        any("no receipt and no reason" in e for e in b["check"]["errors"]))
    nr = b["lines"][-1]["id"]
    call("PUT", "/reports/%d/lines/%d" % (rid, nr), "emp", {**GOOD, "receipt": "N",
                                                           "no_receipt_reason": "Lost"})
    st, b = add(rid, category_account="MR53000000")
    chk("a roll-up header named as a category blocks submit",
        any("not an MR5 expense account" in e for e in b["check"]["errors"]))
    call("DELETE", "/reports/%d/lines/%d" % (rid, b["lines"][-1]["id"]), "emp")

    print("\n5. A draft is its owner's alone")
    chk("the owner reads it", call("GET", "/reports/%d" % rid, "emp")[0] == 200)
    for who in ("mgr", "other", "acct", "ceo", "cfo", "admin"):
        chk("%s gets 404 on another's draft" % who, call("GET", "/reports/%d" % rid, who)[0] == 404)
    chk("another employee cannot add a line to it", add(rid, who="other")[0] == 404)
    chk("the manager's list does not show it",
        rid not in [r["id"] for r in call("GET", "/reports?scope=all", "mgr")[1]["reports"]])

    print("\n6. Submitted: the approver and the backups read it; nobody else does")
    st, b = call("POST", "/reports/%d/submit" % rid, "emp")
    chk("a complete report submits", st == 200 and b["status"] == "submitted", b.get("error"))
    chk("the manager reads it and may decide",
        call("GET", "/reports/%d" % rid, "mgr")[1]["permissions"]["decide_as"] == "approver")
    chk("it is on the manager's to-approve list",
        rid in [r["id"] for r in call("GET", "/reports?scope=to_approve", "mgr")[1]["reports"]])
    chk("the CEO may decide it", call("GET", "/reports/%d" % rid, "ceo")[1]["permissions"]["decide_as"] == "ceo")
    for who in ("other", "acct", "cfo"):
        chk("%s gets 404 on a submitted report not theirs to decide" % who,
            call("GET", "/reports/%d" % rid, who)[0] == 404)
    chk("the owner cannot change it while submitted", add(rid)[0] == 403)
    chk("the owner cannot decide it", call("POST", "/reports/%d/decide" % rid, "emp",
                                           {"action": "approve"})[0] == 403)
    st, b = call("POST", "/reports/%d/recall" % rid, "emp")
    chk("the owner may recall it", st == 200 and b["status"] == "draft")
    chk("a recalled report cannot be deleted -- it has a history",
        call("DELETE", "/reports/%d" % rid, "emp")[0] == 403)
    call("POST", "/reports/%d/submit" % rid, "emp")

    print("\n7. Return needs a note; approval locks it and hands it to accounting")
    chk("return with no note is refused",
        call("POST", "/reports/%d/decide" % rid, "mgr", {"action": "return"})[0] == 400)
    st, b = call("POST", "/reports/%d/decide" % rid, "mgr", {"action": "return", "note": "Split the hotel"})
    chk("returned with a note", st == 200 and b["status"] == "returned")
    chk("the owner may change a returned report", add(rid)[0] == 200)
    call("POST", "/reports/%d/submit" % rid, "emp")
    st, b = call("POST", "/reports/%d/decide" % rid, "mgr", {"action": "approve"})
    chk("the approver approves", st == 200 and b["status"] == "approved"
        and b["decided_basis"] == "approver", b.get("decided_basis"))
    chk("accounting now reads it", call("GET", "/reports/%d" % rid, "acct")[0] == 200)
    chk("...and finds it in its list",
        rid in [r["id"] for r in call("GET", "/reports?scope=all", "acct")[1]["reports"]])
    chk("the owner cannot change an approved report", add(rid)[0] == 403)
    chk("it cannot be decided twice",
        call("POST", "/reports/%d/decide" % rid, "mgr", {"action": "return", "note": "x"})[0] == 403)
    chk("another employee still gets 404", call("GET", "/reports/%d" % rid, "other")[0] == 404)

    print("\n8. The backups: the CEO or President in place of an absent approver")
    r2 = new_report("emp", "2026-10-01", "2026-10-31")
    add(r2)
    call("POST", "/reports/%d/submit" % r2, "emp")
    st, b = call("POST", "/reports/%d/decide" % r2, "pres", {"action": "approve"})
    chk("the President approves another's report",
        st == 200 and b["decided_basis"] == "President in place of Max Manager", b.get("decided_basis"))

    print("\n9. An approver's report goes to the CFO; the CFO's to the CEO or President")
    r3 = new_report("mgr")
    add(r3, who="mgr")
    st, b = call("POST", "/reports/%d/submit" % r3, "mgr")
    chk("the manager's report is routed to the CFO", b.get("route_kind") == "cfo", b.get("error"))
    chk("the manager cannot approve their own",
        call("POST", "/reports/%d/decide" % r3, "mgr", {"action": "approve"})[0] in (403, 404))
    chk("the employee they approve cannot either", call("GET", "/reports/%d" % r3, "emp")[0] == 404)
    st, b = call("POST", "/reports/%d/decide" % r3, "cfo", {"action": "approve"})
    chk("the CFO approves it", st == 200 and b["decided_basis"] == "CFO", b.get("decided_basis"))
    r4 = new_report("cfo")
    add(r4, who="cfo")
    st, b = call("POST", "/reports/%d/submit" % r4, "cfo")
    chk("the CFO's report is routed to the CEO or President", b.get("route_kind") == "ceo_president")
    chk("the CFO cannot approve their own",
        call("POST", "/reports/%d/decide" % r4, "cfo", {"action": "approve"})[0] in (403, 404))
    st, b = call("POST", "/reports/%d/decide" % r4, "ceo", {"action": "approve"})
    chk("the CEO approves it, not 'in place of' anyone",
        st == 200 and b["decided_basis"] == "CEO", b.get("decided_basis"))

    print("\n10. The CEO's own report is not the CEO's to approve")
    call("PUT", "/employees/%d" % ids["ceo"], "admin", {"approver_user_id": ids["mgr"]})
    r5 = new_report("ceo")
    add(r5, who="ceo")
    call("POST", "/reports/%d/submit" % r5, "ceo")
    chk("the CEO cannot approve their own, review role or not",
        call("POST", "/reports/%d/decide" % r5, "ceo", {"action": "approve"})[0] == 403)
    chk("the President can", call("POST", "/reports/%d/decide" % r5, "pres",
                                  {"action": "approve"})[0] == 200)

    print("\n11. A draft never submitted may be deleted")
    r6 = new_report("emp", "2026-11-01", "2026-11-30")
    add(r6)
    chk("another employee cannot delete it", call("DELETE", "/reports/%d" % r6, "other")[0] == 404)
    chk("the owner can", call("DELETE", "/reports/%d" % r6, "emp")[0] == 200)
    chk("and it is gone", call("GET", "/reports/%d" % r6, "emp")[0] == 404)

    print("\n%d passed, %d failed" % (len(_passed), len(_failed)))
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
