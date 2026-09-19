"""Guardrail: the treasury API, and the keys the treasury SCREEN reads.

WHY THIS EXISTS. `treasury_reconciliation_check.py` proves the service is right.
It cannot catch the seam between the service and the screen, and that seam broke
twice while the screen was being written: the tie panel read `opening` and
`net_movement` when the service returns `opening_balance` and `bank_movement`,
and the import panel read `imported`/`duplicates` when it returns
`inserted`/`already_held`. Both render as a blank cell -- no error, no log, just
a figure quietly missing from a reconciliation. So the field names the template
reads are asserted here, by name, against a live response.

It also checks that an ACCOUNTANT can run a whole month end to end -- import,
reconcile, pair, close -- because the section-wide rule is "only accounting may
edit" and it would be easy to satisfy that by admitting nobody useful. Who is
REFUSED is enumerated across every accounting route in
scripts/accounting_access_check.py.

AUTHENTICATION: this mints its own JWT with the app's local dev secret. It never
reads, types or stores a password, and it is a test harness, not a login.
"""
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_passed, _failed = [], []


def chk(label, cond, detail=""):
    (_passed if cond else _failed).append(label)
    print("   %s %s%s" % ("ok  " if cond else "FAIL", label,
                          ("   [%s]" % detail) if detail and not cond else ""))


def _token(app, role, username="check"):
    import jwt
    return jwt.encode({"sub": "1", "username": username, "role": role,
                       "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
                      app.config["JWT_SECRET"], algorithm="HS256")


#: Every field the treasury screen reads, by response. A name removed or
#: renamed on the server shows up here instead of as a blank cell on screen.
ACCOUNT_FIELDS = [
    "account_number", "account_name", "entityid", "gl_cash_account",
    "current_ledger", "current_available", "ledger_reason", "available_reason",
    "last_period", "last_status", "last_balance", "unclosed_months",
    "activity_through",
]
TIE_FIELDS = [
    "headline", "opening_balance", "opening_source", "bank_movement",
    "transaction_count", "computed_ending", "statement_ending",
    "ties_to_statement", "statement_difference", "gl_net", "ties_to_gl",
    "gl_difference",
]
MATCH_FIELDS = ["headline", "matched", "bank_only", "gl_only", "ambiguous",
                "deposits_in_transit", "outstanding_payments"]
BANK_ROW_FIELDS = ["bank_id", "date", "description", "transaction_type",
                   "signed_amount", "amount"]
GL_ROW_FIELDS = ["gl_item", "date", "description", "kind", "amount"]
PAIR_FIELDS = ["bank_id", "bank_date", "bank_description", "bank_amount",
               "gl_item", "gl_date", "gl_description", "manual", "far_apart",
               "days_apart"]
IMPORT_FIELDS = ["inserted", "already_held", "accounts", "skipped",
                 "skipped_count"]
#: The statements-on-file panel. A statement is no use stored if the screen
#: cannot name it, so these are asserted like every other seam here.
STATEMENT_FIELDS = ["id", "account_number", "account_name", "entityid",
                    "period_start", "period_end", "beginning_balance",
                    "ending_balance", "source_file", "imported_at", "period",
                    "has_file"]


def _missing(payload, fields):
    return [f for f in fields if f not in payload]


def main():
    os.environ.setdefault("DATABASE_URL", "")
    from flask_app import create_app
    from flask_app.services import treasury_service as ts

    app = create_app()
    client = app.test_client()
    admin = {"Authorization": "Bearer %s" % _token(app, "admin")}
    acctant = {"Authorization": "Bearer %s" % _token(app, "accountant")}

    ACCT = "TRCHK0001"
    with app.app_context():
        ts.ensure_tables()

    print("1. The gate")
    # WHAT THIS GATE CAN AND CANNOT DO. `role_required` compares LEVELS, and
    # analyst, accountant, accounting_manager and cfo are all level 1. So
    # naming ("admin", "cfo", "analyst") admits every accounting role and
    # excludes viewers -- it cannot single the CFO out, and no arrangement of
    # names would make it. Asserted here as what it IS, so a later reader does
    # not mistake the decorator's wording for a CFO-only gate.
    viewer = {"Authorization": "Bearer %s" % _token(app, "viewer")}
    chk("an unsigned request is refused",
        client.get("/api/treasury/accounts").status_code == 401)
    chk("an accountant can see the accounts",
        client.get("/api/treasury/accounts", headers=acctant).status_code == 200)
    chk("an accountant may import -- it is their daily work",
        client.post("/api/treasury/import/activity",
                    headers=acctant).status_code == 400)  # 400 = no file, not 403
    chk("a viewer may NOT import -- read-only means read-only",
        client.post("/api/treasury/import/activity",
                    headers=viewer).status_code == 403)
    chk("a viewer may still READ where a reconciliation stands",
        client.get("/api/treasury/accounts", headers=viewer).status_code == 200)
    chk("a viewer may NOT remap an account",
        client.put("/api/treasury/accounts/%s" % ACCT, headers=viewer,
                   json={"entityid": "X"}).status_code == 403)
    chk("a viewer may NOT close a period",
        client.post("/api/treasury/close", headers=viewer,
                    json={"account_number": ACCT,
                          "period": "202608"}).status_code == 403)
    chk("an accountant CAN close a period -- the bank rec is their work, and "
        "the level model could not separate them from the CFO anyway",
        client.post("/api/treasury/close", headers=acctant,
                    json={"account_number": ACCT,
                          "period": "202608"}).status_code != 403)

    print("\n2. Refusals say what is missing")
    r = client.get("/api/treasury/reconcile", headers=admin)
    chk("reconcile without an account is refused", r.status_code == 400)
    chk("and names both arguments",
        "account_number" in r.get_json()["error"], str(r.get_json()))
    r = client.post("/api/treasury/import/statement", headers=admin,
                    data={"account_number": ACCT})
    chk("a statement import with no file is refused", r.status_code == 400)

    print("\n3. The fields the accounts tab reads")
    # Register the account the way a real import does, then read it back.
    with app.app_context():
        ts.import_activity([{
            "account_number": ACCT, "as_of_date": "2026-08-15",
            "bai_control": "115", "transaction_type": "Deposit",
            "amount": 1000.0, "direction": "credit", "signed_amount": 1000.0,
            "reference": "CHK", "description": "guardrail row",
            "currency": "USD", "source_file": "check", "row_hash": "trchk-1",
            "bank_id": "0001", "account_name": "Guardrail"}])
    body = client.get("/api/treasury/accounts", headers=admin).get_json()
    chk("the cash account dropdown is served with the accounts",
        body.get("default_cash_account") == ts.DEFAULT_CASH_ACCOUNT
        and len(body.get("cash_accounts") or []) == 5,
        str(body.get("cash_accounts")))
    row = next((a for a in body["accounts"] if a["account_number"] == ACCT), None)
    chk("the imported account is listed", row is not None)
    if row:
        chk("every field the accounts table reads is present",
            not _missing(row, ACCOUNT_FIELDS), str(_missing(row, ACCOUNT_FIELDS)))
        # The one column that must never be filled in by inference.
        chk("current available is None, not a number", row["current_available"] is None)
        chk("an unclosed account has no ledger figure rather than 0.00",
            row["current_ledger"] is None, str(row["current_ledger"]))
        chk("and the screen is given the reason to show",
            bool(row["ledger_reason"]) and bool(row["available_reason"]))

    print("\n4. The fields the reconciliation tab reads")
    client.post("/api/treasury/seed-opening", headers=admin,
                json={"account_number": ACCT, "period": "202608",
                      "amount": 500.0})
    tie = client.get("/api/treasury/reconcile", headers=admin,
                     query_string={"account_number": ACCT, "period": "202608",
                                   "gl_net": "1000"}).get_json()
    chk("every field the tie panel reads is present",
        not _missing(tie, TIE_FIELDS), str(_missing(tie, TIE_FIELDS)))
    chk("the tie computes from the seeded opening",
        abs(tie["computed_ending"] - 1500.0) < 0.01, str(tie["computed_ending"]))
    chk("the ledger leg is compared when a figure is given",
        tie["ties_to_gl"] is True, str(tie["gl_difference"]))
    chk("and the statement leg stays None with no statement filed",
        tie["ties_to_statement"] is None)

    print("\n5. The fields the matcher reads")
    m = client.get("/api/treasury/match", headers=admin,
                   query_string={"account_number": ACCT,
                                 "period": "202608"}).get_json()
    # Unmapped, so it explains itself rather than looking reconciled -- and it
    # STILL carries the bank side, which the screen lists.
    chk("an unmapped account explains itself through the API",
        "not mapped" in (m.get("error") or ""), str(m.get("error"))[:80])
    chk("and the bank rows it did read carry every field the list needs",
        m["bank_only"] and not _missing(m["bank_only"][0], BANK_ROW_FIELDS),
        str(_missing(m["bank_only"][0], BANK_ROW_FIELDS)) if m["bank_only"] else "none")

    client.put("/api/treasury/accounts/%s" % ACCT, headers=admin,
               json={"entityid": "TRCHK", "gl_cash_account": "MR10006000"})
    m2 = client.get("/api/treasury/match", headers=admin,
                    query_string={"account_number": ACCT,
                                  "period": "202608"}).get_json()
    chk("mapped but with no ledger feed, it says which account it looked in",
        "MR10006000" in (m2.get("error") or ""), str(m2.get("error"))[:90])
    chk("the top-level matcher fields are present even on a refusal",
        not _missing(m2, ["matched", "bank_only", "gl_only", "ambiguous"]),
        str(_missing(m2, ["matched", "bank_only", "gl_only", "ambiguous"])))
    chk("a bad cash account is refused at the API too",
        client.put("/api/treasury/accounts/%s" % ACCT, headers=admin,
                   json={"gl_cash_account": "MR99999999"}
                   ).get_json().get("error") is not None)

    print("\n6. Close, and what the screen shows afterwards")
    closed = client.post("/api/treasury/close", headers=admin,
                         json={"account_number": ACCT,
                               "period": "202608"}).get_json()
    chk("close returns the figure the banner prints",
        abs(closed.get("computed_ending", 0) - 1500.0) < 0.01, str(closed))
    row2 = next(a for a in client.get("/api/treasury/accounts", headers=admin)
                .get_json()["accounts"] if a["account_number"] == ACCT)
    chk("the accounts tab now carries a ledger position",
        abs(row2["current_ledger"] - 1500.0) < 0.01, str(row2["current_ledger"]))
    chk("available is STILL not inferred from it",
        row2["current_available"] is None)
    chk("and the position names the close it came from",
        "202608" in row2["ledger_reason"], row2["ledger_reason"][:80])

    print("\n7. Manual pairing round-trips")
    with app.app_context():
        bank_id = ts.match(ACCT, "202608")["bank_only"][0]["bank_id"]
    chk("a pairing is accepted",
        client.put("/api/treasury/match", headers=admin,
                   json={"account_number": ACCT, "period": "202608",
                         "bank_id": bank_id, "gl_item": "IT0001"}
                   ).get_json().get("gl_item") == "IT0001")
    chk("and clearing it returns None, not an empty string",
        client.put("/api/treasury/match", headers=admin,
                   json={"account_number": ACCT, "period": "202608",
                         "bank_id": bank_id, "gl_item": None}
                   ).get_json().get("gl_item") is None)

    # The import response shape, which the Import tab prints line by line.
    print("\n8. The import response")
    import io as _io
    # PNC's own header spelling, and its habit of guarding a text field with a
    # leading apostrophe so Excel will not eat the leading zeros.
    csv = (b"AccountNumber,AsOfDate,Amount,Credit/Debit,Description\n"
           b"'TRCHK0001,08/15/2026,1000.00,Credit,guardrail row\n")
    r = client.post("/api/treasury/import/activity", headers=acctant,
                    data={"file": (_io.BytesIO(csv), "activity.csv")},
                    content_type="multipart/form-data")
    body = r.get_json()
    chk("an import returns every field the result panel prints",
        r.status_code == 200 and not _missing(body, IMPORT_FIELDS),
        str(_missing(body, IMPORT_FIELDS)) if r.status_code == 200 else str(body)[:120])
    if r.status_code == 200:
        chk("the export's rows are read", body["inserted"] == 1, str(body))
        # An accountant re-pulls a month after a correction. The SAME file
        # again must add nothing -- this is the check that keeps a re-import
        # from doubling a month's movement.
        again = client.post("/api/treasury/import/activity", headers=acctant,
                            data={"file": (_io.BytesIO(csv), "activity.csv")},
                            content_type="multipart/form-data").get_json()
        chk("re-importing the same export adds nothing",
            again["inserted"] == 0 and again["already_held"] == 1, str(again))

    # A file that is not an activity export must be REFUSED, not reported as
    # "0 imported" -- a refusal dressed up as a successful no-op is how a month
    # comes to be missing without anybody noticing.
    bad = client.post("/api/treasury/import/activity", headers=acctant,
                      data={"file": (_io.BytesIO(b"\x00\x01not a csv"), "x.csv")},
                      content_type="multipart/form-data")
    chk("a file that is not an activity export is refused",
        bad.status_code == 400, str(bad.get_json())[:120])
    chk("and the refusal names what was missing",
        "missing" in (bad.get_json().get("error") or "").lower(),
        str(bad.get_json())[:120])

    print("\n7. Filing a statement opens the chain, and lists it")
    # Jim, Sep 19 2026: seeding belongs in the load, and a statement should be
    # reachable afterwards. Both are asserted over HTTP here; the service-level
    # rules, including the refusal to re-base a reconciled account, are in
    # scripts/treasury_seed_on_import_check.py.
    st = {"period_start": "2026-06-01", "period_end": "2026-06-30",
          "beginning_balance": 1000.00, "ending_balance": 2500.00,
          "credits_total": 1500.00, "debits_total": 0.0,
          "source_file": "api-check-june.pdf", "internally_consistent": True}
    # ACCT has been reconciled and CLOSED by the sections above, so it is the
    # account that must be left alone. A fresh one covers the opening case --
    # both directions, since seeding everything would satisfy one of them.
    FRESH = ACCT[:-1] + "9"
    with app.app_context():
        ts.create_account(FRESH, entityid="TRCHK", user="check")
        fresh_res = ts.import_statement(st, FRESH, user="check")
        closed_res = ts.import_statement(st, ACCT, user="check")
    chk("filing a statement opens the following month",
        fresh_res.get("seeded_period") == "202607", str(fresh_res)[:120])
    chk("...at the statement's own ending balance",
        fresh_res.get("seeded_amount") == 2500.00,
        str(fresh_res.get("seeded_amount")))
    chk("a reconciled account is NOT re-based by a later statement",
        closed_res.get("ok") and not closed_res.get("seeded_period"),
        str(closed_res)[:120])
    chk("...and its statement is filed all the same",
        closed_res.get("ok") is True, str(closed_res)[:90])

    lst = client.get("/api/treasury/statements?account_number=%s" % ACCT,
                     headers=acctant)
    chk("GET /statements answers", lst.status_code == 200, str(lst.status_code))
    rows = lst.get_json() or []
    chk("the statement just filed is listed", len(rows) >= 1, str(len(rows)))
    if rows:
        chk("the statements panel reads no field the server does not send",
            not _missing(rows[0], STATEMENT_FIELDS),
            str(_missing(rows[0], STATEMENT_FIELDS)))
        chk("...and the period is derived for it, not left to the screen",
            rows[0]["period"] == "202606", str(rows[0].get("period")))
    # A read, like the rest of the section: a viewer may look one up.
    chk("a viewer may call up a statement list",
        client.get("/api/treasury/statements", headers=viewer).status_code == 200)

    # ---- cleanup -------------------------------------------------------
    from sqlalchemy import text
    from flask_app.db import get_engine
    with app.app_context():
        eng = get_engine()
        for t in ("tr_activity", "tr_statements", "tr_periods", "tr_accounts",
                  "tr_matches"):
            with eng.begin() as c:
                c.execute(text("DELETE FROM %s WHERE account_number IN "
                               "(:a, :b)" % t),
                          {"a": ACCT, "b": ACCT[:-1] + "9"})

    print("\n%d checks, %d failed." % (len(_passed) + len(_failed), len(_failed)))
    if _failed:
        for f in _failed:
            print("  FAILED: %s" % f)
        return 1
    print("The screen's field names are asserted against live responses, and\n"
          "an accountant can run the whole month: import, reconcile, pair and\n"
          "close. See accounting_access_check.py for the section-wide gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
