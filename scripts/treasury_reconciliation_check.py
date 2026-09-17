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
    # A ZERO-BALANCE ACCOUNT PRINTS `.00`, NOT `0.00`. Measured against the 64
    # real June 2026 statements: this one detail refused 46 of them, including
    # statements carrying real amounts, because a single `.00` anywhere in the
    # four-figure row broke the whole line. 14 filed before the fix, 45 after.
    z = ts.parse_statement_text(
        "Account Number: XX-XXXX-4178\n"
        "For the period 06/16/2026 to 06/30/2026\n"
        "Balance Summary\n"
        "Beginning Deposits and Checks and Ending\n"
        "balance other credits other debits balance\n"
        ".00 .00 .00 .00\n", source_file="zero.pdf")
    chk("a zero-balance statement reads as 0.00, not as a refusal",
        not z.get("error") and z["ending_balance"] == 0.0, str(z)[:110])
    chk("and it still foots", z.get("internally_consistent") is True)
    # The commoner case: real money with a zero in one column.
    mixed = ts.parse_statement_text(
        "Account Number: XX-XXXX-9221\n"
        "balance other credits other debits balance\n"
        "30,832.24 .00 11,712.76 19,119.48\n", source_file="mixed.pdf")
    chk("a row mixing real amounts with .00 reads all four",
        mixed["beginning_balance"] == 30832.24
        and mixed["ending_balance"] == 19119.48, str(mixed)[:110])

    # PNC writes an overdraft with a TRAILING minus. Read as positive it would
    # be a wrong balance rather than a refusal, which is the worse failure.
    chk("a trailing minus is read as negative", ts._money("1,234.56-") == -1234.56)
    chk("and an ordinary figure is unaffected",
        ts._money("571,750.04") == 571750.04)
    chk("`.00` parses as zero", ts._money(".00") == 0.0)

    # THE FOUR FIGURES COME FROM UNDER THE SUMMARY HEADER. Loosening the number
    # pattern made more lines matchable, so "the first four money figures in
    # the document" stopped being safe -- a totals block further up could win.
    anchored = ts.parse_statement_text(
        "Some Total 1.00 2.00 3.00 4.00\n"
        "balance other credits other debits balance\n"
        "100.00 50.00 25.00 125.00\n", source_file="anchor.pdf")
    chk("an earlier four-figure row does not win over the summary",
        anchored["beginning_balance"] == 100.0
        and anchored["ending_balance"] == 125.0, str(anchored)[:110])

    # A statement from a DIFFERENT BANK is refused rather than half-read. One
    # of the 64 June files is a Wells Fargo statement.
    wf = ts.parse_statement_text(
        "Account Number: 5825-5092\nWells Fargo\n", source_file="wf.pdf")
    chk("a non-PNC statement is refused, not half-read",
        bool(wf.get("error")) and wf["ending_balance"] is None)

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

        _match_section(ts, eng, rows, real)
        _position_section(ts, eng, acct, real)
        _seed_section(ts, eng, acct, real)

        for t in ("tr_activity", "tr_statements", "tr_periods", "tr_accounts",
                  "tr_matches"):
            with eng.begin() as c:
                c.execute(text("DELETE FROM %s WHERE account_number = :a" % t),
                          {"a": acct})

    return _report()


# ---- 5. matching bank items to the ledger --------------------------
#
# Uses the GL upload's own MR10005000 lines as the ledger side, because that is
# exactly what `gl_detail` holds once the entry is posted.
def _seed_gl(eng, entityid, acct, period):
    """Put the GL upload's cash lines into gl_detail as the ledger side."""
    import pandas as pd
    from sqlalchemy import text as _t
    if not GL.exists():
        return 0
    gl = pd.read_csv(GL).dropna(how="all")
    gl = gl[gl["AcctNum"] == acct]
    with eng.begin() as c:
        c.execute(_t('DELETE FROM gl_detail WHERE "ENTITYID" = :e'),
                  {"e": entityid})
        for i, r in gl.reset_index(drop=True).iterrows():
            d = str(r["ENTRDATE"])
            c.execute(_t(
                'INSERT INTO gl_detail ("ENTITYID","PERIOD","ENTRDATE",'
                '"ACCTNAME","ACCTNUM","BASIS","BALFOR","ITEM","REF","DESCRPN",'
                '"SEGMENTID","RLTDENTITY","RLTDENTITY_NAME","AMT") VALUES '
                '(:e,:p,:d,:an,:a,:b,:bf,:i,:rf,:ds,NULL,NULL,NULL,:amt)'),
                {"e": entityid, "p": period, "d": d, "an": "Cash",
                 "a": acct, "b": "B", "bf": "N", "i": "IT%04d" % i,
                 "rf": "JE", "ds": str(r["Descrpn"])[:200],
                 "amt": float(r["Amount"])})
    return len(gl)


def _match_section(ts, eng, rows, real):
    from sqlalchemy import text as _t
    acct_no = AUG["account"]
    print("\n5. Matching bank items to the ledger")
    if not real:
        print("   (real files not present -- skipped)")
        return
    n = _seed_gl(eng, "AMB6", "MR10005000", "202608")
    chk("the ledger's cash lines are available", n == 24, str(n))
    ts.import_activity(rows, eng)
    ts.set_account(acct_no, entityid="AMB6", gl_cash_account="MR10005000",
                   user="check", engine=eng)

    m = ts.match(acct_no, "202608", eng)
    chk("no error", not m.get("error"), str(m.get("error"))[:90])
    # PAIRED, NOT SET-COMPARED: 285.92 appears seven times in August. A set
    # comparison would call all seven matched the moment one was.
    chk("all 24 bank items pair with 24 ledger entries",
        len(m["matched"]) == 24, str(len(m["matched"])))
    chk("nothing is left on either side",
        not m["bank_only"] and not m["gl_only"],
        "bank_only=%d gl_only=%d" % (len(m["bank_only"]), len(m["gl_only"])))
    chk("the totals agree", m["ties"] is True, str(m.get("difference")))
    chk("repeated amounts are consumed, not reused",
        len({p["gl_item"] for p in m["matched"]}) == 24)
    chk("and the repeats were noticed as ambiguous", m["ambiguous"] > 0,
        str(m["ambiguous"]))

    # ---- deposits in transit, which is September's real case ----
    # Drop two bank credits: the ledger has them, the bank has not seen them.
    with eng.begin() as c:
        ids = [r[0] for r in c.execute(_t(
            "SELECT id FROM tr_activity WHERE account_number = :a "
            "  AND direction = 'credit' ORDER BY amount DESC LIMIT 2"),
            {"a": acct_no}).fetchall()]
        for i in ids:
            c.execute(_t("DELETE FROM tr_activity WHERE id = :i"), {"i": i})
    m2 = ts.match(acct_no, "202608", eng)
    chk("ledger entries the bank has not seen are left over",
        len(m2["gl_only"]) == 2, str(len(m2["gl_only"])))
    chk("and are named deposits in transit",
        all(g["kind"] == "deposit in transit" for g in m2["gl_only"]),
        str([g["kind"] for g in m2["gl_only"]]))
    chk("the difference equals what is outstanding",
        abs(m2["difference"] + m2["deposits_in_transit"]) < 0.01,
        "%s vs %s" % (m2["difference"], m2["deposits_in_transit"]))
    chk("the headline says so in an accountant's words",
        "deposits in transit" in m2["headline"], m2["headline"][:110])

    # ---- a manual pairing outranks the matcher ----
    m3 = ts.match(acct_no, "202608", eng)
    if m3["bank_only"] and m3["gl_only"]:
        b = m3["bank_only"][0]["bank_id"]
        g = m3["gl_only"][0]["gl_item"]
        ts.set_match(acct_no, "202608", b, g, "check", eng)
        m4 = ts.match(acct_no, "202608", eng)
        chk("a hand-made pairing is honoured even across amounts",
            any(p["bank_id"] == b and p["gl_item"] == g and p["manual"]
                for p in m4["matched"]))
        ts.set_match(acct_no, "202608", b, None, "check", eng)
        m5 = ts.match(acct_no, "202608", eng)
        chk("and can be removed again",
            not any(p["bank_id"] == b and p["manual"] for p in m5["matched"]))

    # An unmapped account cannot be matched, and says why rather than
    # returning an empty result that looks reconciled.
    ts.set_account(acct_no, entityid="", engine=eng)
    m6 = ts.match(acct_no, "202608", eng)
    chk("an unmapped account explains itself",
        bool(m6.get("error")) and "not mapped" in m6["error"], str(m6.get("error"))[:70])
    ts.set_account(acct_no, entityid="AMB6", engine=eng)
    with eng.begin() as c:
        c.execute(_t('DELETE FROM gl_detail WHERE "ENTITYID" = :e'), {"e": "AMB6"})


# ---- 6. the position shown on the accounts tab ---------------------
#
# Jim asked this tab for "the list of accounts, current ledger and current
# available". Ledger we can carry; available exists only at the bank. The
# checks below exist to stop a later change from quietly filling that column
# with the ledger figure, which is the one number a treasurer would act on.
def _position_section(ts, eng, acct, real):
    print("\n6. The accounts tab position")
    a = [x for x in ts.accounts(eng) if x["account_number"] == acct][0]
    chk("available is not known, and is None rather than a number",
        a["current_available"] is None)
    chk("and it says why", "only at the bank" in a["available_reason"])
    chk("ledger is not the available figure dressed up",
        a["current_ledger"] != a["current_available"]
        or a["current_ledger"] is None)
    if real:
        # August is closed at this point; September holds no activity.
        chk("the ledger carries the last close forward",
            abs(a["current_ledger"] - AUG["ending"]) < 0.01,
            str(a["current_ledger"]))
        chk("and names the period it was carried from",
            "202608" in a["ledger_reason"], a["ledger_reason"][:90])

    # An account with activity but nothing closed has no ledger position, and
    # must not report 0.00 -- a real zero balance and an unknown one are not
    # the same fact.
    p = ts._position(None, None, [("2026-08", -560022.54, 24, "2026-08-29")])
    chk("no close means no ledger figure, not zero", p["current_ledger"] is None)
    chk("and the reason names the missing close",
        "no period has been closed" in p["ledger_reason"].lower(),
        p["ledger_reason"][:80])
    chk("the months awaiting a close are listed",
        p["unclosed_months"] == ["2026-08"], str(p["unclosed_months"]))

    # A close followed by later activity moves the position.
    p2 = ts._position("202608", 11727.50,
                      [("2026-08", -560022.54, 24, "2026-08-29"),
                       ("2026-09", 510000.00, 4, "2026-09-03")])
    chk("activity after the close is added, and earlier activity is not",
        abs(p2["current_ledger"] - 521727.50) < 0.01, str(p2["current_ledger"]))
    chk("the position says which date it runs through",
        p2["activity_through"] == "2026-09-03", str(p2["activity_through"]))


# ---- 7. starting the chain from a statement ------------------------
#
# Jim asked whether to type fifty opening balances or read them off the June
# statements. The statement is the better source, but ONLY for the first
# period: taking one later would re-base the chain and hide a break, which is
# the exact thing `opening_balance` refuses to do. Both halves are checked.
def _seed_section(ts, eng, acct, real):
    from sqlalchemy import text as _t
    print("\n7. Starting the chain from a statement")

    ts.set_account(acct, entityid="AMB6", gl_cash_account="MR10005000",
                   user="check", engine=eng)

    # No statement on file for the prior month -> refused, and it says which
    # month to file.
    with eng.begin() as c:
        c.execute(_t("DELETE FROM tr_statements WHERE account_number = :a"),
                  {"a": acct})
        c.execute(_t("DELETE FROM tr_periods WHERE account_number = :a"),
                  {"a": acct})
    r = ts.seed_from_statement(acct, "202608", "check", eng)
    chk("with no statement on file, seeding is refused",
        bool(r.get("error")) and "202607" in r["error"], str(r)[:100])

    # File a July statement, then seed August from it.
    ts.import_statement({"beginning_balance": 100.0, "ending_balance": 571750.04,
                         "credits_total": 0.0, "debits_total": 0.0,
                         "period_start": "2026-07-01", "period_end": "2026-07-31",
                         "source_file": "july.pdf", "internally_consistent": True},
                        acct, eng)
    r = ts.seed_from_statement(acct, "202608", "check", eng)
    chk("with one on file, August opens at the July statement's ENDING balance",
        not r.get("error") and abs(r["amount"] - 571750.04) < 0.01, str(r)[:110])
    chk("and the note names the file it came from, not 'by hand'",
        "july.pdf" in (r.get("source") or ""), str(r.get("source"))[:90])
    op = ts.opening_balance(acct, "202608", eng)
    chk("the ordinary opening lookup now finds it",
        abs((op.get("opening") or 0) - 571750.04) < 0.01, str(op)[:100])

    # THE HALF THAT MATTERS: once a period is genuinely reconciled, seeding is
    # refused. Re-basing on a statement would paper over a broken chain.
    with eng.begin() as c:
        c.execute(_t("UPDATE tr_periods SET status = 'closed' "
                     " WHERE account_number = :a"), {"a": acct})
    r2 = ts.seed_from_statement(acct, "202609", "check", eng)
    chk("once a period is RECONCILED, seeding is refused",
        bool(r2.get("error")) and "carried forward" in r2["error"],
        str(r2)[:110])
    chk("and the refusal says why, not just no",
        "hide any break" in (r2.get("error") or ""), str(r2)[:110])

    # THE MASK IS NOT ALWAYS A TAIL, and reading it as one was wrong on five of
    # Jim's six June statements that carried real balances. `790-XXXXX55` hides
    # the MIDDLE digits: taking "the last four visible digits" gives 790 + 55 ->
    # `79055` -> `9055`, an account that does not exist, while the real account
    # 7900021255 was registered all along.
    chk("a middle-masked number becomes a whole-number pattern",
        ts._mask_pattern("790-XXXXX55").pattern == r"^790\d{5}55$",
        ts._mask_pattern("790-XXXXX55").pattern)
    chk("and it matches the real account",
        bool(ts._mask_pattern("790-XXXXX55").match("7900021255")))
    chk("but not one that merely ends in 55",
        not ts._mask_pattern("790-XXXXX55").match("8612199055"))
    chk("a front-masked number still works",
        bool(ts._mask_pattern("XX-XXXX-5765").match("8514245765")))
    chk("and is anchored, so a longer number does not match",
        not ts._mask_pattern("XX-XXXX-5765").match("18514245765"))
    chk("a mask with no digits at all is refused",
        ts._mask_pattern("XX-XXXX-") is None)

    # Registering an account by hand, for one PNC will not serve activity for.
    ts.create_account("9999000111", entityid="ZZTEST",
                      gl_cash_account="MR10005000", user="check", engine=eng)
    chk("an account can be registered by hand",
        any(a["account_number"] == "9999000111" for a in ts.accounts(eng)))
    chk("registering it twice is refused rather than duplicated",
        "already registered" in str(ts.create_account(
            "9999000111", user="check", engine=eng).get("error")))
    # THE NUMBER IS NEVER INFERRED. A masked statement number is not an account
    # number, and accepting one would create an account that real activity can
    # never join.
    chk("a masked number is refused as an account number",
        bool(ts.create_account("XX-XXXX-7891", user="check",
                               engine=eng).get("error")))
    chk("an unknown cash account is refused here too",
        bool(ts.create_account("9999000222", gl_cash_account="MR99999999",
                               user="check", engine=eng).get("error")))
    chk("a hand-registered account defaults to the generic cash account",
        [a for a in ts.accounts(eng)
         if a["account_number"] == "9999000111"][0]["gl_cash_account"]
        == ts.DEFAULT_CASH_ACCOUNT)
    with eng.begin() as c:
        c.execute(_t("DELETE FROM tr_accounts WHERE account_number "
                     "IN ('9999000111','9999000222')"))

    # Routing a masked statement number to an account.
    m = ts.match_account_by_suffix("XX-XXXX-" + acct[-4:], eng)
    chk("a masked statement number routes to its account",
        m.get("account_number") == acct, str(m)[:90])
    chk("an unknown suffix is refused with the digits it looked for",
        "9999" in str(ts.match_account_by_suffix("XX-XXXX-9999", eng)), "")
    chk("too few digits is refused rather than guessed",
        bool(ts.match_account_by_suffix("XX-X", eng).get("error")))

    # AMBIGUITY IS REFUSED, NOT RESOLVED. All fifty of today's accounts have
    # unique last-four, but that is a fact about today's accounts.
    # A GENUINE COLLISION now needs the same LENGTH as well as the same visible
    # digits, because the pattern is anchored to the whole number. That is the
    # point -- it is why 8612199055 no longer collides with 790-XXXXX55.
    alt = "9999" + acct[-6:]
    ts.set_account(alt, entityid="X", engine=eng)
    with eng.begin() as c:
        c.execute(_t("INSERT INTO tr_accounts (account_number, gl_cash_account,"
                     " active) VALUES (:a, :g, 1)"),
                  {"a": alt, "g": "MR10005000"})
    m2 = ts.match_account_by_suffix("XX-XXXX-" + acct[-4:], eng)
    chk("two accounts sharing a suffix is REFUSED, not resolved by picking one",
        bool(m2.get("error")) and "More than one" in m2["error"], str(m2)[:100])
    with eng.begin() as c:
        c.execute(_t("DELETE FROM tr_accounts WHERE account_number = :a"),
                  {"a": alt})


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
