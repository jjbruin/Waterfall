"""Guardrail: expense coding and the payroll batch (phase 3).

THE ACCEPTANCE TEST IS ACCOUNTING'S OWN FILE. `2026-09-24 Payroll ERs Entries -
JE Upload.csv` (119 lines, PSCMAN / PPI2 / PSC3 / TGA6) is read back into the
app as approved lines carrying the coding accounting gave them, batched, and
the batch compared with the file LINE FOR LINE. A format check written from a
description proves only that the code agrees with itself (the treasury v492
lesson). The file is not in the repo -- it names employees and what they spent
-- so that section SKIPS, saying so, where it is absent; set ER_ACCEPTED_CSV to
point at it.

Expected differences, each asserted rather than tolerated:
  * PPI2's cab charges are debited 159.66 and credited 159.96 in the file, so it
    is out of balance by 0.30 -- the app refuses the file as written, and its own
    batch balances;
  * PPI2's Canadian amounts were converted at slightly different rates line to
    line (1.41341-1.41350); one rate per batch reproduces them to a cent or two.

Also: the ownership walk through the REAL `build_chain` on fixture commitments
shaped like Gallery (PPI25 a pass-through to PSCKOC 70 / PSC3 30, an operating
partner outside) and Apple (PPI2 keeps its own intercompany account), access,
locking, the race on batching, and void.
"""
import csv
import io
import os
import re
import sys
import tempfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ACCEPTED = os.environ.get("ER_ACCEPTED_CSV") or str(
    Path.home() / "OneDrive - peaceablestreet.com" / "Documents" /
    "2026-09-24 Payroll ERs Entries - JE Upload.csv")

_passed, _failed = [], []
#: --inject=nogate (batch unreviewed coding) | noacct (approval emails skip accounting)
#:          | selfemail (the submitter is emailed about their own submission)
#:          | silentnobody (nobody to email is not recorded)
INJECT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--inject=")), "")


def chk(label, cond, detail=""):
    (_passed if cond else _failed).append(label)
    print("   %s %s%s" % ("ok  " if cond else "FAIL", label,
                          ("   [%s]" % (detail,)) if detail and not cond else ""))


def skip(label, why):
    print("   skip %s   (%s)" % (label, why))


def main():
    tmp = tempfile.mkdtemp(prefix="expense_coding_")
    os.environ.pop("DATABASE_URL", None)
    os.environ["DB_PATH"] = os.path.join(tmp, "check.db")

    import jwt
    import pandas as pd
    from sqlalchemy import text
    from flask_app import create_app
    from flask_app.auth import sections as S
    from flask_app.auth.models import create_user, list_users
    from flask_app.db import get_engine
    from flask_app.services import data_service, expense_coding as ec
    from database import PROTECTED_TABLES

    app = create_app()
    app.config["DATABASE_URL"] = None
    client = app.test_client()
    inv = pd.DataFrame([
        {"vcode": "P0000040", "Investment_Name": "The Gallery", "InvestmentID": "THEGAL"},
        {"vcode": "P0000003", "Investment_Name": "Apple Self Storage", "InvestmentID": "APPLE"},
        {"vcode": "P0000037", "Investment_Name": "Pontchartrain Landing", "InvestmentID": "PONTCH"},
    ]).assign(Portfolio_Name="", Sale_Status="", Sale_Date=None, Lifecycle="Income")
    data_service.get_data = lambda *a, **k: {"inv": inv}

    people = {"admin": "admin", "acct": "accountant", "ana": "analyst", "mgr": "analyst",
              "emp": "analyst", "cfo1": "cfo", "amgr": "accounting_manager"}
    # EVERY EMAIL IS CAPTURED, never sent, and sent inline so the history it writes can be
    # read straight back. The manager has no address, so "not sent" is exercised too.
    from flask_app.auth import email_utils
    from flask_app.services import expense_notify as en
    sent = []
    email_utils.send_email_result = lambda to, subject, html: (
        sent.append({"to": to, "subject": subject, "html": html}) or {"ok": True})
    en.SYNC = True
    from flask_app.services import expense_coding as _ec
    if INJECT == "nogate":
        _orig_rs = _ec.review_state
        import inspect as _insp
        def _rs(rep):
            caller = _insp.stack()[1].function
            return "reviewed" if caller == "build_batch" else _orig_rs(rep)
        _ec.review_state = _rs
    if INJECT == "noacct":
        en.accounting = lambda *a, **k: []
    if INJECT == "selfemail":
        _orig_rv = en.reviewers
        en.reviewers = lambda engine, exclude_username=None, **k: _orig_rv(engine, None, **k)
    with app.app_context():
        eng = get_engine()
        for u, role in people.items():
            create_user(u, "pw-" + u, role=role,
                        email=None if u == "mgr" else u + "@example.invalid")
        with eng.begin() as c:
            c.execute(text('CREATE TABLE IF NOT EXISTS gl_accounts ("ACCTNUM" TEXT, "ACCTNAME" TEXT, "TYPE" TEXT)'))
            for a, n in (("MR53000004", "Other Expense: Meals & Entertainment"),
                         ("MR53000005", "Other Expense: Meetings & Conferences"),
                         ("MR53000010", "Other Expense: Office Expense"),
                         ("MR53000011", "Other Expense: Travel"),
                         ("MR53000015", "Other Expense: Telephone & Internet"),
                         ("MR53000016", "Other Expense: Dues & Subscriptions")):
                c.execute(text("INSERT INTO gl_accounts VALUES (:a, :n, 'I')"), {"a": a, "n": n})
            c.execute(text("DROP TABLE IF EXISTS deals"))
            c.execute(text('CREATE TABLE deals (vcode TEXT, "InvestmentID" TEXT, "Investment_Name" TEXT)'))
            for _, r in inv.iterrows():
                c.execute(text("INSERT INTO deals VALUES (:v, :i, :n)"),
                          {"v": r["vcode"], "i": r["InvestmentID"], "n": r["Investment_Name"]})
            # The intercompany population, as the GL shows it.
            c.execute(text('CREATE TABLE IF NOT EXISTS gl_detail ("ENTITYID" TEXT, "ACCTNUM" TEXT, '
                           '"RLTDENTITY" TEXT, "BASIS" TEXT, "PERIOD" TEXT, "DESCRPN" TEXT, "AMT" REAL)'))
            for e in ("PSCKOC", "PSC3", "PPI2", "TGA6", "KOCPARENT"):
                c.execute(text("INSERT INTO gl_detail VALUES (:e, 'MR15000002', '', 'A', '202608', 'x', -1)"),
                          {"e": e})
            c.execute(text("INSERT INTO gl_detail VALUES ('PSCMAN', 'MR15000001', 'TGA6', 'B', '202608', 'x', 1)"))
            # Commitments: EntityID is invested IN, InvestorID is the owner.
            c.execute(text("DROP TABLE IF EXISTS commitments"))
            c.execute(text('CREATE TABLE commitments ("CommitmentUID" TEXT, "EntityID" TEXT, '
                           '"InvestorID" TEXT, "Amount" REAL, "CapitalPercent" REAL, '
                           '"StartDate" TEXT, "EndDate" TEXT)'))
            for i, (ent, own, amt) in enumerate((
                    ("THEGAL", "PPI25", 9_000_000), ("THEGAL", "OPPARTNER", 1_000_000),
                    ("PPI25", "PSCKOC", 7_000_000), ("PPI25", "PSC3", 3_000_000),
                    ("APPLE", "PPI2", 5_000_000),
                    ("PONTCH", "INVF7", 4_000_000), ("INVF7", "PSC3", 4_000_000),
                    # Above PSCKOC sits an owner that ALSO keeps an intercompany
                    # account: the walk must stop at PSCKOC, not climb to it.
                    ("PSCKOC", "KOCPARENT", 1_000_000))):
                c.execute(text("INSERT INTO commitments VALUES (:u, :e, :o, :a, NULL, '2020-01-01', NULL)"),
                          {"u": "C%d" % i, "e": ent, "o": own, "a": amt})
        ids = {u["username"]: u["id"] for u in list_users()}

    def H(name):
        return {"Authorization": "Bearer " + jwt.encode(
            {"sub": str(ids[name]), "username": name, "role": people[name],
             "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
            app.config["JWT_SECRET"], algorithm="HS256")}

    def call(method, path, who, body=None):
        r = client.open(path, method=method, json=body, headers=H(who))
        return r.status_code, (r.get_json(silent=True) or {}), r

    print("1. Registry and access")
    chk("/api/expense-coding belongs to Accounting",
        S.sections_for_api("/api/expense-coding/lines") == ("accounting",))
    chk("the four phase-3 tables are protected",
        all(t in PROTECTED_TABLES for t in ("er_coding", "er_entity_currency", "er_recurring",
                                            "er_batches")))
    chk("an analyst cannot even READ the coding grid (it is everyone's expenses)",
        call("GET", "/api/expense-coding/lines", "ana")[0] == 403)
    chk("accounting can", call("GET", "/api/expense-coding/lines", "acct")[0] == 200)

    # THE COLUMN RACE (found Oct 7 2026): the coding screen loads lines and settings at
    # once; both found the review columns missing and the second ALTER failed "duplicate
    # column" -- a 500 and an empty grid. Reproduced: the first read of er_reports' columns
    # inside expense_coding.ensure_tables is made STALE, as if another request added them
    # in between.
    import sqlalchemy
    from flask_app.services import expense_coding as ec_mod
    with app.app_context():
        eng2 = get_engine()
        ec_mod.ensure_tables(eng2)                    # the columns now exist
        ec_mod._DONE.clear()
        real_inspect = sqlalchemy.inspect
        seen = {"n": 0}

        class _Stale:
            def __init__(self, e):
                self._i = real_inspect(e)

            def get_columns(self, table):
                import inspect as _pyinspect
                cols = self._i.get_columns(table)
                caller = _pyinspect.stack()[1]
                if table == "er_reports" and caller.function == "ensure_tables" \
                        and caller.filename.endswith("expense_coding.py"):
                    seen["n"] += 1
                    if seen["n"] == 1:
                        return [c for c in cols if not str(c["name"]).startswith("coding_")]
                return cols

            def __getattr__(self, a):
                return getattr(self._i, a)
        sqlalchemy.inspect = _Stale
        try:
            ec_mod.ensure_tables(eng2)
            raced = None
        except Exception as e:
            raced = e
        finally:
            sqlalchemy.inspect = real_inspect
        chk("a request racing another to add the review columns does not fail",
            raced is None and seen["n"] >= 1, raced)

    print("\n2. Who owns the expense: the real build_chain, walked to the intercompany entity")
    with app.app_context():
        g = ec.propose_interco(get_engine(), "P0000040")
        a = {x["entity"]: x for x in g["allocations"]}
        chk("Gallery: PPI25 is a pass-through, so PSCKOC 70% and PSC3 30%",
            set(a) == {"PSCKOC", "PSC3"} and abs(a["PSCKOC"]["pct"] - 70) < 1e-6
            and abs(a["PSC3"]["pct"] - 30) < 1e-6, g)
        chk("...re-based over PSC's side: the operating partner's 10% is not ours to bear",
            "re-based" in (g.get("basis") or ""), g.get("basis"))
        chk("...RLTDENTITY at the entity is the entity below it (PPI25)",
            a.get("PSCKOC", {}).get("rltd") == "PPI25")
        g = ec.propose_interco(get_engine(), "P0000003")
        chk("Apple: PPI2 keeps its own intercompany account, so the walk stops there",
            [(x["entity"], round(x["pct"], 6)) for x in g["allocations"]] == [("PPI2", 100.0)], g)
        g = ec.propose_interco(get_engine(), "P0000037")
        chk("Pontchartrain: INVF7 keeps none, so PSC3, with RLTDENTITY INVF7 (as in the Sep 24 file)",
            [(x["entity"], x["rltd"]) for x in g["allocations"]] == [("PSC3", "INVF7")], g)

    # ---- a report through to a batch -------------------------------------
    call("PUT", "/api/expenses/employees/%d" % ids["emp"], "admin",
         {"full_name": "Fred Kurz", "approver_user_id": ids["mgr"]})
    st, rep, _ = call("POST", "/api/expenses/reports", "emp",
                      {"period_start": "2026-09-01", "period_end": "2026-09-30"})
    rid = rep["id"]
    base = {"line_date": "2026-08-17", "purpose": "Property Visit - Existing", "receipt": "N",
            "no_receipt_reason": "fixture"}
    for body in (
        {**base, "category_account": "MR53000015", "deal_code": "OPERATIONS",
         "comment": "Cell Phone Reimbursement - August", "amount": "75.81"},
        {**base, "category_account": "MR53000011", "deal_code": "PIPELINE",
         "deal_name": "Market Poplar", "comment": "Site Visit - Airfare", "amount": "1219.80"},
        {**base, "category_account": "MR53000011", "deal_code": "P0000040",
         "comment": "Site Visit - Hotel", "amount": "100.01"},
        {**base, "category_account": "MR53000011", "deal_code": "P0000037",
         "comment": "Pontchartrain Site Visit - Dinner at Arnauds", "amount": "228.03"},
        {**base, "category_account": "MR53000011", "deal_code": "P0000003",
         "comment": "Apple Site Visit - Airfare", "amount": "712.98"}):
        call("POST", "/api/expenses/reports/%d/lines" % rid, "emp", body)
    sent.clear()
    call("POST", "/api/expenses/reports/%d/submit" % rid, "emp")

    print("\n2b. The workflow emails (the CFO's ask, Oct 7 2026)")
    hist = lambda: call("GET", "/api/expenses/reports/%d" % rid, "emp")[1]["events"]
    chk("submitting emails no one when the approver has no address -- and SAYS so in the history",
        not sent and any(e["action"] == "email not sent" and "no email address" in (e["note"] or "")
                         for e in hist()), [(e["action"], e["basis"]) for e in hist()])

    print("\n3. Only an approved report reaches accounting")
    st, b, _ = call("GET", "/api/expense-coding/lines", "acct")
    chk("a submitted report is not on the grid", not b["rows"])
    sent.clear()
    call("POST", "/api/expenses/reports/%d/decide" % rid, "mgr", {"action": "approve"})
    to = sorted(m["to"] for m in sent)
    chk("approval emails accounting -- accountant, accounting manager, CFO -- and the employee",
        to == sorted(["acct@example.invalid", "amgr@example.invalid", "cfo1@example.invalid",
                      "emp@example.invalid"]), to)
    chk("...not the analyst or the admin role", not any(t.startswith(("ana@", "admin@")) for t in to))
    chk("...accounting's links to Expense Coding, the employee's to their report",
        all("/expense-coding" in m["html"] for m in sent if m["to"] != "emp@example.invalid")
        and any("/expenses?report=%d" % rid in m["html"] for m in sent if m["to"] == "emp@example.invalid"))
    chk("...and each send is in the report's history",
        sum(1 for e in hist() if e["action"] == "emailed") == 4)
    st, b, _ = call("GET", "/api/expense-coding/lines", "acct")
    rows = {r["comment"]: r for r in b["rows"]}
    chk("once approved, every line is on the grid", len(b["rows"]) == 5, len(b["rows"]))
    chk("operations books to the category's account",
        rows["Cell Phone Reimbursement - August"]["booking"] == "expense")
    chk("a pipeline deal books to Deal Cost Receivable",
        rows["Site Visit - Airfare"]["booking"] == "deal_cost")
    chk("an owned deal books intercompany",
        rows["Site Visit - Hotel"]["booking"] == "interco")
    # MRI's rule (Jim, Oct 2 2026): 80 characters, letters, digits and spaces only;
    # the employee as initials.
    chk("the description is ER, initials, deal, comment -- Operations left out, no punctuation",
        rows["Cell Phone Reimbursement - August"]["description"]
        == "ER FK Cell Phone Reimbursement August"
        and rows["Site Visit - Airfare"]["description"]
        == "ER FK Market Poplar Site Visit Airfare", rows["Site Visit - Airfare"]["description"])
    chk("an intercompany description is prefixed IC",
        rows["Site Visit - Hotel"]["description"].startswith("IC ER FK The Gallery"),
        rows["Site Visit - Hotel"]["description"])
    chk("a comment's words the deal already says are not repeated",
        rows["Pontchartrain Site Visit - Dinner at Arnauds"]["description"]
        == "IC ER FK Pontchartrain Landing Site Visit Dinner at Arnauds",
        rows["Pontchartrain Site Visit - Dinner at Arnauds"]["description"])
    from flask_app.services.treasury_upload import mri_description
    long_ = mri_description("ER", "FK", "Some Deal", "Conference Registration Reimbursement for the "
                            "International Association annual meeting in a far city", keep=2)
    chk("over 80, common long words are abbreviated before any are dropped",
        len(long_) <= 80 and "Conf" in long_ and "Reimb" in long_, long_)
    chk("under 80, words are kept whole",
        mri_description("ER", "FK", "", "Conference Registration", keep=2) == "ER FK Conference Registration")
    chk("the suffix keeps its room when the rest is trimmed",
        len(mri_description("x" * 10, "y " * 60, suffix="USD 712.98")) <= 80
        and mri_description("x" * 10, "y " * 60, suffix="USD 712.98").endswith("USD 712 98"))

    print("\n4. Accounting's corrections stick and the proposal stays beside them")
    din = rows["Pontchartrain Site Visit - Dinner at Arnauds"]
    st, _, _ = call("PUT", "/api/expense-coding/lines/%d/0" % din["line_id"], "acct",
                    {"expense_account": "MR53000004",
                     "description": "Interco - ER - Fred Kurz - Pontchartrain Site Visit - Dinner at Arnaud's!"})
    chk("an analyst cannot code", call("PUT", "/api/expense-coding/lines/%d/0" % din["line_id"],
                                       "ana", {"expense_account": "MR53000004"})[0] == 403)
    st, b, _ = call("GET", "/api/expense-coding/lines", "acct")
    d2 = [r for r in b["rows"] if r["line_id"] == din["line_id"]][0]
    chk("accounting's typed description is held to MRI's rule as it is saved",
        d2["description"] == "Interco ER Fred Kurz Pontchartrain Site Visit Dinner at Arnaud s",
        d2["description"])
    chk("the recode to Meals is applied, and the employee's Travel is still shown",
        d2["expense_account"] == "MR53000004" and d2["employee_account"] == "MR53000011"
        and d2["expense_account_changed"])
    chk("shares that do not total 100% are refused",
        call("PUT", "/api/expense-coding/lines/%d/0" % din["line_id"], "acct",
             {"interco": [{"entity": "PSC3", "pct": 60}]})[0] == 400)

    print("\n4b. The coding review: accounting submits, the CFO reviews, only reviewed is batched")
    rv = {"report_ids": [rid], "payroll_date": "2026-09-24", "credit_suffix": "End of Month",
          "fx": {"PPI2": 1.41344}}
    st, pv, _ = call("POST", "/api/expense-coding/batches", "acct", rv)
    chk("before review the batch is REFUSED, saying why",
        any("has not been submitted for review" in e for e in pv["errors"]), pv["errors"])
    chk("an analyst cannot submit coding",
        call("POST", "/api/expense-coding/review/submit", "ana", {"report_ids": [rid]})[0] == 403)
    sent.clear()
    st, out, _ = call("POST", "/api/expense-coding/review/submit", "acct", {"report_ids": [rid]})
    chk("the accountant submits it for review", st == 200 and out["submitted"] == [rid], out)
    chk("...which emails the reviewers, CFO and accounting manager, and not the submitter",
        sorted(m["to"] for m in sent) == ["amgr@example.invalid", "cfo1@example.invalid"],
        [m["to"] for m in sent])
    state = lambda: {r["review_state"] for r in call("GET", "/api/expense-coding/lines", "acct")[1]["rows"]}
    chk("...and the grid says it is waiting", state() == {"submitted"}, state())
    chk("the accountant cannot review", call("POST", "/api/expense-coding/review/decide", "acct",
                                             {"report_ids": [rid], "action": "approve"})[0] == 403)
    chk("a return needs a reason", call("POST", "/api/expense-coding/review/decide", "cfo1",
                                        {"report_ids": [rid], "action": "return"})[0] == 400)
    sent.clear()
    st, _, _ = call("POST", "/api/expense-coding/review/decide", "cfo1",
                    {"report_ids": [rid], "action": "return", "note": "Dinner is Meals, not Travel"})
    chk("the CFO returns it with a reason; it is not submitted any more", st == 200 and state() == {""})
    chk("...and the accountant who submitted it is emailed, with the reason",
        [m["to"] for m in sent] == ["acct@example.invalid"]
        and "Dinner is Meals" in sent[0]["html"], [m["to"] for m in sent])
    # The CFO is in accounting too, and may submit; she is then not emailed about her
    # own submission -- only the other reviewer is.
    sent.clear()
    call("POST", "/api/expense-coding/review/submit", "cfo1", {"report_ids": [rid]})
    chk("a reviewer who submits is not emailed about her own submission",
        [m["to"] for m in sent] == ["amgr@example.invalid"], [m["to"] for m in sent])
    st, _, _ = call("POST", "/api/expense-coding/review/decide", "cfo1",
                    {"report_ids": [rid], "action": "approve"})
    chk("resubmitted and approved: reviewed", st == 200 and state() == {"reviewed"}, state())
    st, sv, _ = call("PUT", "/api/expense-coding/lines/%d/0" % din["line_id"], "acct",
                     {"expense_account": "MR53000004",
                      "description": "Interco - ER - Fred Kurz - Pontchartrain Site Visit - Dinner at Arnaud's!"})
    chk("coding changed AFTER review goes back for review", sv.get("back_for_review") is True
        and state() == {"submitted"}, (sv, state()))
    st, pv, _ = call("POST", "/api/expense-coding/batches", "acct", rv)
    chk("...and the batch refuses it again until it is re-reviewed",
        any("is waiting for review" in e for e in pv["errors"]), pv["errors"])
    call("POST", "/api/expense-coding/review/decide", "amgr", {"report_ids": [rid], "action": "approve"})
    chk("the accounting manager may review too", state() == {"reviewed"}, state())
    # NOBODY TO EMAIL is said, not silent: with no reviewer on file, submitting coding
    # leaves "email not sent -- no reviewer to email" in the history.
    if INJECT == "silentnobody":
        en._nobody = lambda *a, **k: None
    real_rv = en.reviewers
    en.reviewers = lambda *a, **k: []
    try:
        with app.app_context():
            en.coding_event(get_engine(), [rid], "coding_submitted", "http://check/",
                            {"id": ids["acct"], "username": "acct"})
    finally:
        en.reviewers = real_rv
    chk("with no reviewer to email, the history SAYS no email was sent",
        any(e["action"] == "email not sent" and e["basis"] == "no reviewer to email" for e in hist()),
        [(e["action"], e["basis"]) for e in hist()][-3:])

    print("\n5. The batch")
    call("PUT", "/api/expense-coding/currency", "acct", {"entity_id": "PPI2", "currency": "CAD"})
    body = {"report_ids": [rid], "payroll_date": "2026-09-24", "credit_suffix": "End of Month"}
    st, pv, _ = call("POST", "/api/expense-coding/batches", "acct", body)
    chk("a CAD entity with no rate is refused, naming it",
        any("PPI2 books in CAD" in e for e in pv["errors"]), pv["errors"])
    body["fx"] = {"PPI2": 1.41344}
    st, pv, _ = call("POST", "/api/expense-coding/batches", "acct", body)
    chk("with the rate the preview is clean", st == 200 and not pv["errors"], pv["errors"])
    L = pv["lines"]
    by = lambda e, a: [x for x in L if x["entityid"] == e and x["acctnum"] == a]
    chk("one payroll credit at PSC Manager for the total",
        [x["amount"] for x in by("PSCMAN", "MR20000001")] == [-round(75.81 + 1219.80 + 100.01 + 228.03 + 712.98, 2)])
    chk("its description names the period and the suffix",
        by("PSCMAN", "MR20000001")[0]["descrpn"] == "ER Trinet Payroll 202609 End of Month",
        by("PSCMAN", "MR20000001")[0]["descrpn"])
    gal = sorted((x["rltdentity"], x["amount"]) for x in by("PSCMAN", "MR15000001")
                 if "The Gallery" in x["descrpn"])
    chk("Gallery's 100.01 splits to the cent: PSC3 30.00, PSCKOC 70.01",
        gal == [("PSC3", 30.0), ("PSCKOC", 70.01)], gal)
    chk("PSC3 books the dinner as Meals, RLTDENTITY INVF7, against Due To PSC Manager",
        [(x["acctnum"], x["amount"], x["rltdentity"]) for x in L if x["entityid"] == "PSC3"
         and "Arnaud" in x["descrpn"]] == [("MR53000004", 228.03, "INVF7"),
                                            ("MR15000002", -228.03, "")])
    ppi2 = [x for x in L if x["entityid"] == "PPI2"]
    chk("PPI2 books in CAD, with the USD in the description",
        [x["amount"] for x in ppi2] == [1007.75, -1007.75]
        and ppi2[0]["descrpn"].endswith("USD 712 98"), ppi2)
    import re as _re
    bad = [x["descrpn"] for x in L if len(x["descrpn"]) > 80 or _re.search(r"[^A-Za-z0-9 ]", x["descrpn"])]
    chk("EVERY line of the batch is 80 characters or fewer, letters, digits and spaces only",
        L and not bad, bad[:3])
    chk("PSC Manager's side of PPI2 stays in USD",
        [x["amount"] for x in by("PSCMAN", "MR15000001") if x["rltdentity"] == "PPI2"] == [712.98])
    for e in ("PSCMAN", "PSC3", "PSCKOC", "PPI2"):
        chk("%s balances" % e, abs(sum(x["amount"] for x in L if x["entityid"] == e)) < 0.005)

    print("\n6. Generate, lock, race, void")
    body["commit"] = True
    st, gen, _ = call("POST", "/api/expense-coding/batches", "acct", body)
    chk("the batch is generated with its file", st == 200 and gen.get("csv", "").startswith("EntityID,"))
    st, rep2, _ = call("GET", "/api/expenses/reports/%d" % rid, "emp")
    chk("the employee sees it batched", rep2["status"] == "batched")
    chk("a batched report cannot be coded",
        call("PUT", "/api/expense-coding/lines/%d/0" % din["line_id"], "acct", {"description": "x"})[0] == 403)
    st, again, _ = call("POST", "/api/expense-coding/batches", "acct", body)
    chk("the same reports cannot be batched twice", st == 400)
    st, _, raw = call("GET", "/api/expense-coding/batches/%s/csv" % gen["batch_id"], "acct")
    chk("the file downloads", st == 200 and raw.data.decode().startswith("EntityID,AcctNum,Amount"))
    st, bl, _ = call("GET", "/api/expense-coding/batches", "acct")
    chk("the batch reads not yet in the GL", bl["batches"][0]["status"] == "generated, not yet in the GL")
    with app.app_context():
        with get_engine().begin() as c:
            c.execute(text("INSERT INTO gl_detail VALUES ('PSCMAN', 'MR20000001', '', 'B', '202609', "
                           ":d, :a)"), {"d": gen["credit_description"], "a": -gen["total"]})
    st, bl, _ = call("GET", "/api/expense-coding/batches", "acct")
    chk("once MRI shows the payroll credit, posted", bl["batches"][0]["status"] == "posted")
    st, v, _ = call("POST", "/api/expense-coding/batches/%s/void" % gen["batch_id"], "acct")
    chk("void releases the report to approved, coding kept",
        call("GET", "/api/expenses/reports/%d" % rid, "emp")[1]["status"] == "approved"
        and [r for r in call("GET", "/api/expense-coding/lines", "acct")[1]["rows"]
             if r["line_id"] == din["line_id"]][0]["expense_account"] == "MR53000004")

    print("\n7. A recurring reimbursement (FK - Benefits) rides in a batch when ticked")
    st, recs, _ = call("POST", "/api/expense-coding/recurring", "acct",
                       {"user_id": ids["emp"], "description": "Benefits Reimbursement",
                        "account": "MR51000005", "amount": "3960.60"})
    rec_id = recs[0]["id"] if isinstance(recs, list) and recs else None
    body.pop("commit")
    body["recurring_ids"] = [rec_id]
    st, pv, _ = call("POST", "/api/expense-coding/batches", "acct", body)
    chk("it adds its line and the credit grows by it",
        [x["amount"] for x in pv["lines"] if x["acctnum"] == "MR51000005"] == [3960.60]
        and by("PSCMAN", "MR20000001") is not None
        and [x["amount"] for x in pv["lines"] if x["acctnum"] == "MR20000001"]
        == [-round(75.81 + 1219.80 + 100.01 + 228.03 + 712.98 + 3960.60, 2)])
    body["recurring_ids"] = []
    st, pv, _ = call("POST", "/api/expense-coding/batches", "acct", body)
    chk("unticked, it does not", not [x for x in pv["lines"] if x["acctnum"] == "MR51000005"])

    print("\n8. ACCEPTANCE: accounting's Sep 24 upload, rebuilt")
    if not os.path.exists(ACCEPTED):
        skip("the Sep 24 rebuild", "%s is not here; set ER_ACCEPTED_CSV" % ACCEPTED)
    else:
        acceptance(call, app, ids, people, H, client)

    print("\n%d passed, %d failed" % (len(_passed), len(_failed)))
    return 1 if _failed else 0


def acceptance(call, app, ids, people, H, client):
    from flask_app.services import treasury_upload as tu
    raw = open(ACCEPTED, encoding="latin-1").read()
    rows = list(csv.DictReader(io.StringIO(raw)))
    gl = [{"entityid": r["EntityID"], "acctnum": r["AcctNum"], "amount": r["Amount"],
           "descrpn": r["Descrpn"], "rltdentity": r["RLTDENTITY"], "period": r["Period"],
           "basis": r["Basis"], "entrdate": r["ENTRDATE"]} for r in rows]
    v = tu.validate_gl(gl)
    chk("the file as written is REFUSED: PPI2 is out by 0.30",
        any("Out of balance by entity" in e and "PPI2" in e and "0.30" in e for e in v["errors"]),
        v["errors"])

    # Every PSCMAN debit is one coded line; its intercompany pair says the
    # entity's expense account. Read accounting's coding OUT of the file.
    pscman = [r for r in rows if r["EntityID"] == "PSCMAN" and r["AcctNum"] != "MR20000001"]
    pairs = {}
    for r in rows:
        if r["EntityID"] != "PSCMAN" and r["AcctNum"] != "MR15000002":
            pairs.setdefault((r["EntityID"], r["Descrpn"].split(" (")[0]), r)
    names = []
    for r in pscman:
        m = re.match(r"(?:Interco - )?ER - ([^-]+?) - ", r["Descrpn"])
        names.append(m.group(1).strip() if m else "?")
    from flask_app.auth.models import create_user, list_users
    from flask_app.db import get_engine
    with app.app_context():
        for n in sorted(set(names)):
            un = "acc_" + re.sub(r"\W", "", n).lower()
            if un not in ids:
                create_user(un, "pw", role="analyst")
        ids.update({u["username"]: u["id"] for u in list_users()})
    for n in sorted(set(names)):
        un = "acc_" + re.sub(r"\W", "", n).lower()
        people[un] = "analyst"
        call("PUT", "/api/expenses/employees/%d" % ids[un], "admin",
             {"full_name": n, "approver_user_id": ids["mgr"]})
    reports, line_meta = {}, []
    for r, n in zip(pscman, names):
        un = "acc_" + re.sub(r"\W", "", n).lower()
        if un not in reports:
            reports[un] = call("POST", "/api/expenses/reports", un,
                               {"period_start": "2026-09-01", "period_end": "2026-09-30"})[1]["id"]
        acct = r["AcctNum"]
        kind = ("PIPELINE" if acct == "MR11000012" else
                "P0000003" if acct == "MR15000001" else "OPERATIONS")
        cat = acct if acct.startswith("MR53") else "MR53000011"
        st, rep, _ = call("POST", "/api/expenses/reports/%d/lines" % reports[un], un, {
            "line_date": "2026-09-01", "category_account": cat, "purpose": "Other",
            "deal_code": kind, "deal_name": "x", "comment": "x", "amount": r["Amount"],
            "receipt": "N", "no_receipt_reason": "acceptance"})
        line_meta.append((rep["lines"][-1]["id"], r))
    for un, rid in reports.items():
        call("POST", "/api/expenses/reports/%d/submit" % rid, un)
        call("POST", "/api/expenses/reports/%d/decide" % rid, "mgr", {"action": "approve"})
    call("PUT", "/api/expense-coding/currency", "acct", {"entity_id": "PPI2", "currency": "CAD"})
    for lid, r in line_meta:
        body = {"description": r["Descrpn"]}
        if r["AcctNum"] == "MR15000001":
            ent = r["RLTDENTITY"]
            pr = pairs.get((ent, r["Descrpn"].split(" (")[0]))
            body.update({"booking": "interco",
                         "expense_account": pr["AcctNum"] if pr else "MR53000011",
                         "interco": [{"entity": ent, "pct": 100,
                                      "rltd": (pr or {}).get("RLTDENTITY", "")}]})
        elif r["AcctNum"] == "MR11000012":
            body["booking"] = "deal_cost"
        else:
            body.update({"booking": "expense", "expense_account": r["AcctNum"]})
        call("PUT", "/api/expense-coding/lines/%d/0" % lid, "acct", body)
    call("POST", "/api/expense-coding/review/submit", "acct", {"report_ids": list(reports.values())})
    call("POST", "/api/expense-coding/review/decide", "cfo1",
         {"report_ids": list(reports.values()), "action": "approve"})
    st, pv, _ = call("POST", "/api/expense-coding/batches", "acct", {
        "report_ids": list(reports.values()), "payroll_date": "2026-09-24",
        "credit_suffix": "End of Month", "fx": {"PPI2": 1.41344}})
    chk("the app's batch of the same lines is clean", not pv["errors"], pv["errors"][:3])
    ours = Counter((x["entityid"], x["acctnum"], "%.2f" % x["amount"], x["descrpn"].strip(),
                    x["rltdentity"]) for x in pv["lines"])
    # Their descriptions carry " - " and up to 90 characters; MRI's rule (Jim, Oct 2
    # 2026) is 80 and no punctuation, so the app writes THEIR text cleaned to it.
    from flask_app.services.treasury_upload import mri_description as _clean
    theirs = Counter((r["EntityID"], r["AcctNum"], "%.2f" % float(r["Amount"]),
                      _clean(r["Descrpn"]), r["RLTDENTITY"]) for r in rows)
    same = sum((ours & theirs).values())
    only_t = theirs - ours
    only_o = ours - theirs
    print("      %d of %d accepted lines reproduced exactly" % (same, len(rows)))
    for k in sorted(only_t):
        print("      file only: %s" % (k,))
    for k in sorted(only_o):
        print("      app only:  %s" % (k,))
    # The payroll CREDIT is the app's own text, abbreviated to MRI's rule; theirs
    # read "Expense Reimbursement Activity - Trinet Payroll - 202609 End of Month".
    credit = lambda k: k[0] == "PSCMAN" and k[1] == "MR20000001"
    chk("the payroll credit is the app's abbreviated text, for the same amount",
        [k for k in only_o if credit(k)] == [("PSCMAN", "MR20000001", "-19036.08",
                                             "ER Trinet Payroll 202609 End of Month", "")],
        [k for k in only_o if credit(k)])
    non_ppi2_t = [k for k in only_t if k[0] != "PPI2" and not credit(k)]
    non_ppi2_o = [k for k in only_o if k[0] != "PPI2" and not credit(k)]
    chk("every PSCMAN, PSC3 and TGA6 line is reproduced exactly (amount, account, "
        "description, related entity)", not non_ppi2_t and not non_ppi2_o,
        (non_ppi2_t[:3], non_ppi2_o[:3]))
    ppi2_t = sorted(float(k[2]) for k in only_t.elements() if k[0] == "PPI2")
    ppi2_o = sorted(float(k[2]) for k in only_o.elements() if k[0] == "PPI2")
    chk("PPI2 differs only by the CAD rounding and the 0.30 typo, each within 0.31",
        len(ppi2_t) == len(ppi2_o) and all(abs(a - b) <= 0.31 for a, b in zip(ppi2_t, ppi2_o)),
        (ppi2_t, ppi2_o))
    st, _, raw = call("POST", "/api/expense-coding/batches", "acct", {
        "report_ids": list(reports.values()), "payroll_date": "2026-09-24",
        "credit_suffix": "End of Month", "fx": {"PPI2": 1.41344}, "commit": True})
    head = raw.get_json()["csv"].splitlines()[0]
    chk("the file carries MRI's own header row",
        head == raw_header(ACCEPTED), (head, raw_header(ACCEPTED)))
    # The CFO, Oct 7 2026: "journal entry credits need to be negative amounts in the
    # file ... most likely only for Accounts Payable - MR20000001 and MR15000002 Due
    # To/From PSC Manager". Asserted on the file itself, both directions: every line on
    # those two accounts is negative, and the expense debits are positive.
    import csv as _csv, io as _io
    body = list(_csv.reader(_io.StringIO(raw.get_json()["csv"])))[1:]
    cr = [float(r[2]) for r in body if r[1] in ("MR20000001", "MR15000002")]
    dr = [float(r[2]) for r in body if r[1].startswith("MR53")]
    chk("credits to MR20000001 / MR15000002 are NEGATIVE in the file",
        cr and all(a < 0 for a in cr), cr[:6])
    chk("...and the expense debits are positive", dr and all(a > 0 for a in dr), dr[:6])
    chk("...and the file balances to the cent", abs(sum(float(r[2]) for r in body)) < 0.005)

    # The coding screen (the CFO's other three asks, Oct 7 2026).
    v = open(os.path.join(ROOT, "vue_app", "src", "views", "ExpenseCodingView.vue"),
             encoding="utf-8").read()
    chk("the Account column shows the account NAME, the number on hover",
        "{{ acctName(rowAccount(r)) }}" in v)
    chk("the expense account is a real dropdown (name -- number), not a typed box",
        '<select v-model="draft.expense_account"' in v and 'list="ec-accounts"' not in v
        and "{{ c.name }} — {{ c.account }}" in v)
    chk("...which keeps a current account that is not a category, rather than changing it",
        "(not an expense category)" in v)
    chk("the payroll grid totals all reports AND the selected ones",
        "gridTotals.all.total" in v and "gridTotals.sel.total" in v
        and "reportsOnGrid.value.filter(r => pick.value[r.id])" in v)


def raw_header(path):
    return open(path, encoding="latin-1").readline().strip()


if __name__ == "__main__":
    sys.exit(main())
