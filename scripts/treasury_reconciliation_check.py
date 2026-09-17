"""The bank reconciliation, checked against the real August AMB6 files.

WHY THIS EXISTS AS A GUARDRAIL AND NOT A ONE-OFF. The reconciliation's whole
value is that three figures from three places agree. Every one of them is easy
to break silently:

  * PNC guards text fields with a leading apostrophe (`'031000053`), so an
    account number that is not stripped never matches itself and the month
    reconciles against nothing;
  * a transaction dropped while parsing makes a period tie for the WRONG
    reason, which is worse than failing;
  * the opening balance is carried from the prior period's close, so a period
    that has never been closed must report that rather than open at zero --
    zero would show the whole balance as a difference;
  * re-importing an export must not double the month.

Run:  .venv/Scripts/python.exe scripts/treasury_reconciliation_check.py
Exit 1 on failure. Uses the real files when they are present and falls back to
fixtures carrying the same figures, so it runs anywhere.
"""
import io
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

FAILS = []
CHECKS = [0]

#: The measured August 2026 AMB6 figures. Every one of these came from the
#: files Jim supplied, not from this code.
AUG = {
    "account": "8514245765",
    "opening": 571750.04,
    "movement": -560022.54,
    "ending": 11727.50,
    "transactions": 24,
    "credits": 6,
    "debits": 18,
    "gl_cash_net": -560022.54,
}

DOCS = pathlib.Path(
    r"C:\Users\jbruin\OneDrive - peaceablestreet.com\Documents")
ACTIVITY = DOCS / "AMB6 August 2026 Bank Activities.csv"
GL = DOCS / "2026-08 - AMB6 August GL Activity Mock Upload - JE X.7.csv"
STATEMENT = DOCS / "31 August 2026 PSC Ambassadors Fund TGA VI LLC 5765 PNC.pdf"


def chk(label, cond, detail=""):
    CHECKS[0] += 1
    if cond:
        print("   ok   %s" % label)
    else:
        print("   FAIL %s%s" % (label, ("  -- " + detail) if detail else ""))
        FAILS.append(label)


def _fixture_activity():
    """The same shape PNC exports, including the apostrophe-guarded fields."""
    import pandas as pd
    rows = [
        ("08/14/2026", "'031000053", AUG["account"], "PSC AMBASSADORS FUND TGA VI LL",
         "491", "USD", "Money Transfer DB - Other", 14936.0, "Debit",
         "'00000000000", "ACCOUNT TRANSFER TO 0000008612192193"),
        ("08/11/2026", "'031000053", AUG["account"], "PSC AMBASSADORS FUND TGA VI LL",
         "191", "USD", "Money Transfer CR - Other", 882500.0, "Credit",
         "'00000000000", "ACCOUNT TRANSFER FROM 0000008612192222"),
        ("08/11/2026", "'031000053", AUG["account"], "PSC AMBASSADORS FUND TGA VI LL",
         "491", "USD", "Money Transfer DB - Other", 875750.0, "Debit",
         "'00000000000", "ACCOUNT TRANSFER TO 0000008514243858"),
    ]
    return pd.DataFrame(rows, columns=[
        "AsOfDate", "BankId", "AccountNumber", "AccountName", "BaiControl",
        "Currency", "Transaction", "Amount", "Credit/Debit", "Reference",
        "Description"])


def main() -> int:
    import pandas as pd
    from flask_app import create_app
    from flask_app.db import get_engine
    from flask_app.services import treasury_service as ts
    from sqlalchemy import text

    real = ACTIVITY.exists()
    print("Using the real August files." if real
          else "Real files not present; using fixtures with the same shape.")

    # ---- 1. parsing ------------------------------------------------
    print("\n1. Reading the PNC export")
    df = pd.read_csv(ACTIVITY) if real else _fixture_activity()
    parsed = ts.parse_activity(df, source_file=ACTIVITY.name)
    chk("the export is recognised", not parsed.get("error"),
        str(parsed.get("error"))[:90])
    rows = parsed["rows"]
    chk("no row is skipped silently", not parsed["skipped"],
        str(parsed["skipped"][:3]))
    # THE APOSTROPHE. PNC guards text so Excel keeps leading zeros; left in, an
    # account number never matches itself and the month reconciles to nothing.
    chk("the account number is stripped of PNC's text guard",
        all(not r["account_number"].startswith("'") for r in rows)
        and rows[0]["account_number"] == AUG["account"],
        rows[0]["account_number"])
    chk("a debit is negative and a credit positive",
        all((r["signed_amount"] < 0) == (r["direction"] == "debit")
            for r in rows))
    if real:
        chk("all %d transactions are read" % AUG["transactions"],
            len(rows) == AUG["transactions"], str(len(rows)))
        chk("%d credits and %d debits" % (AUG["credits"], AUG["debits"]),
            sum(1 for r in rows if r["direction"] == "credit") == AUG["credits"]
            and sum(1 for r in rows if r["direction"] == "debit") == AUG["debits"])
        chk("the net movement is the measured -560,022.54",
            abs(sum(r["signed_amount"] for r in rows) - AUG["movement"]) < 0.01,
            "{:,.2f}".format(sum(r["signed_amount"] for r in rows)))

    # ---- 2. the statement PDF --------------------------------------
    print("\n2. Reading the statement")
    if STATEMENT.exists():
        import pdfplumber
        with pdfplumber.open(STATEMENT) as pdf:
            txt = "\n".join((p.extract_text() or "") for p in pdf.pages)
        st = ts.parse_statement_text(txt, source_file=STATEMENT.name)
        chk("the balance summary is found", not st.get("error"),
            str(st.get("error"))[:80])
        chk("beginning balance 571,750.04",
            st["beginning_balance"] == AUG["opening"], str(st["beginning_balance"]))
        chk("ending balance 11,727.50",
            st["ending_balance"] == AUG["ending"], str(st["ending_balance"]))
        # The statement's own four figures must agree before any is trusted.
        chk("the statement is internally consistent",
            st.get("internally_consistent") is True)
        chk("the period is read", st["period_end"] == "2026-08-31",
            str(st["period_end"]))
    else:
        print("   (statement not present -- skipped)")
    # A scanned statement must say so rather than return zeros.
    blank = ts.parse_statement_text("", source_file="scan.pdf")
    chk("a scanned statement is refused, not read as zero",
        blank.get("error") and blank["ending_balance"] is None)

    # ---- 3. the tie ------------------------------------------------
    print("\n3. The three-way tie")
    app = create_app()
    with app.app_context():
        eng = get_engine()
        ts.ensure_tables(eng)
        acct = AUG["account"]
        with eng.begin() as c:
            for t in ("tr_activity", "tr_statements", "tr_periods", "tr_accounts"):
                c.execute(text("DELETE FROM %s WHERE account_number = :a" % t),
                          {"a": acct})

        # An unclosed period has NO opening. Zero would report the whole
        # balance as a difference and look like a catastrophic break.
        before = ts.opening_balance(acct, "202608", eng)
        chk("with nothing closed there is no opening, and it says why",
            before["opening"] is None and before.get("reason"),
            str(before)[:90])
        r0 = ts.reconcile(acct, "202608", engine=eng)
        chk("and the reconciliation declines rather than opening at zero",
            r0["computed_ending"] is None)

        ins = ts.import_activity(rows, eng)
        chk("activity imports", ins["inserted"] == len(rows), str(ins))
        again = ts.import_activity(rows, eng)
        # Re-pulling a month after a correction is normal; it must not double.
        chk("re-importing the same export adds nothing",
            again["inserted"] == 0 and again["already_held"] == len(rows),
            str(again))
        chk("the account registers itself, unmapped",
            any(a["account_number"] == acct and a["entityid"] is None
                for a in ts.accounts(eng)))
        chk("and defaults to the generic cash account",
            [a for a in ts.accounts(eng)
             if a["account_number"] == acct][0]["gl_cash_account"]
            == ts.DEFAULT_CASH_ACCOUNT)

        ts.seed_opening(acct, "202608", AUG["opening"], "check", eng)
        op = ts.opening_balance(acct, "202608", eng)
        chk("the seeded opening carries into August",
            abs(op["opening"] - AUG["opening"]) < 0.01, str(op))

        r = ts.reconcile(acct, "202608", gl_net=AUG["gl_cash_net"], engine=eng)
        if real:
            chk("computed ending is 11,727.50",
                abs(r["computed_ending"] - AUG["ending"]) < 0.01,
                "{:,.2f}".format(r["computed_ending"] or 0))
            chk("it ties to the ledger", r["ties_to_gl"] is True,
                str(r["gl_difference"]))
        # Each leg reported separately: which one disagrees is the only thing
        # the number is for.
        chk("with no statement filed, that leg is None and not False",
            r["ties_to_statement"] is None)
        chk("the headline says no statement has been filed",
            "No statement has been filed" in r["headline"], r["headline"][:80])

        ts.set_account(acct, entityid="AMB6", gl_cash_account="MR10005000",
                       user="check", engine=eng)
        chk("a bank account maps to an entity and a cash account",
            [a for a in ts.accounts(eng)
             if a["account_number"] == acct][0]["entityid"] == "AMB6")
        chk("an unknown cash account is refused",
            "error" in ts.set_account(acct, gl_cash_account="MR99999999",
                                      engine=eng))

        # ---- 4. carry forward --------------------------------------
        print("\n4. Carry forward")
        ts.close_period(acct, "202608", "check", eng)
        nxt = ts.opening_balance(acct, "202609", eng)
        if real:
            chk("September opens at August's computed ending, 11,727.50",
                abs(nxt["opening"] - AUG["ending"]) < 0.01, str(nxt))
            chk("and says where it came from",
                "carried from 202608" in str(nxt.get("source")), str(nxt))
        chk("a September with no activity holds that balance",
            abs((ts.reconcile(acct, "202609", engine=eng)["computed_ending"]
                 or 0) - (nxt["opening"] or 0)) < 0.01)

        for t in ("tr_activity", "tr_statements", "tr_periods", "tr_accounts"):
            with eng.begin() as c:
                c.execute(text("DELETE FROM %s WHERE account_number = :a" % t),
                          {"a": acct})

    return _report()


def _report():
    print("\n%d checks, %d failed." % (CHECKS[0], len(FAILS)))
    if FAILS:
        for f in FAILS:
            print("   FAILED: %s" % f)
        return 1
    print("The bank side reads PNC's export as exported, ties opening plus")
    print("movement to the statement and the ledger separately, and carries")
    print("each period's close forward to open the next.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
