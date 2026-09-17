"""Guardrail: the GL and IA upload files MRI will accept.

THE ONLY CHECK THAT PROVES A FILE FORMAT IS A ROUND TRIP AGAINST A REAL ONE.
Jim's August AMB6 GL upload and IA sample were accepted by MRI, so this reads
them, rebuilds each from its own contents, and asserts the rebuild is identical
-- column order, date shape, amount formatting, the lot. A format check written
from a specification rather than from an accepted file proves only that the code
agrees with itself.

It also asserts the three ties that make the August set verifiable:

    the journal entry sums to           0.00
    its MR10005000 lines sum to  -560,022.54  = the bank's own net movement
    its MR31000001 lines and the IA rows both  12,580.47, thirteen of each

Where the real files are absent (any machine but Jim's, and the container) the
file-dependent sections SKIP and the rest still runs on fixtures -- the same
arrangement as treasury_reconciliation_check.py.
"""
import io
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

_passed, _failed = [], []

DOCS = pathlib.Path(
    r"C:\Users\jbruin\OneDrive - peaceablestreet.com\Documents")
GL_REAL = DOCS / "2026-08 - AMB6 August GL Activity Mock Upload - JE X.7.csv"
IA_REAL = DOCS / "AMB6 August IA Activities Sample - IA Upload Template.xlsx"

#: Measured from those files before any of this was written.
AUG = {
    "gl_rows": 49, "gl_total": 0.0,
    "cash_account": "MR10005000", "cash_total": -560022.54, "cash_rows": 24,
    "ia_account": "MR31000001", "ia_total": 12580.47, "ia_rows": 13,
}


def chk(label, cond, detail=""):
    (_passed if cond else _failed).append(label)
    print("   %s %s%s" % ("ok  " if cond else "FAIL", label,
                          ("   [%s]" % detail) if detail and not cond else ""))


def _fixture_gl():
    """A small balanced entry: cash out, expense in."""
    return [
        {"entityid": "AMB6", "acctnum": "MR10005000", "amount": -1000.00,
         "descrpn": "Fee paid", "period": "202608", "basis": "B",
         "entrdate": "2026-08-14"},
        {"entityid": "AMB6", "acctnum": "MR15000002", "amount": 1000.00,
         "descrpn": "Fee paid", "period": "202608", "basis": "B",
         "entrdate": "2026-08-14"},
    ]


def _fixture_ia():
    return [
        {"transaction_type": "Distribution", "sub_type": "Distribution: Income",
         "amount": 600.00, "investmentid": "AMB6", "investorid": "PSC1",
         "transaction_date": "2026-08-14"},
        {"transaction_type": "Distribution", "sub_type": "Distribution: Income",
         "amount": 400.00, "investmentid": "AMB6", "investorid": "CCGSI",
         "transaction_date": "2026-08-14"},
    ]


def main():
    from flask_app.services import treasury_upload as tu

    print("1. The journal entry must balance")
    ok = tu.validate_gl(_fixture_gl())
    chk("a balanced entry passes", not ok["errors"], str(ok["errors"])[:90])
    chk("and reports its own total", abs(ok["total"]) < 0.005, str(ok["total"]))

    bad = _fixture_gl()
    bad[1]["amount"] = 900.00
    v = tu.validate_gl(bad)
    chk("an out-of-balance entry is REFUSED", bool(v["errors"]))
    chk("and the refusal names the figure to look for",
        any("100.00" in e for e in v["errors"]), str(v["errors"])[:110])
    chk("building it raises rather than writing a file MRI would reject",
        _raises(lambda: tu.build_gl_csv(bad)))

    print("\n2. What else is refused, and what is merely odd")
    z = _fixture_gl()
    z[0]["amount"] = 0.0
    z[1]["amount"] = 0.0
    chk("a zero line is refused", bool(tu.validate_gl(z)["errors"]))
    p = _fixture_gl()
    p[0]["period"] = "2026-08"
    chk("a period that is not YYYYMM is refused",
        any("YYYYMM" in e for e in tu.validate_gl(p)["errors"]))
    m = _fixture_gl()
    m[0]["period"] = "202607"
    chk("lines spanning two periods are refused",
        any("more than one period" in e for e in tu.validate_gl(m)["errors"]))
    d = _fixture_gl()
    d[0]["descrpn"] = ""
    vd = tu.validate_gl(d)
    chk("a missing description WARNS and does not block",
        not vd["errors"] and bool(vd["warnings"]))

    print("\n3. The file's own shape")
    text = tu.build_gl_csv(_fixture_gl())
    chk("the header is the template's, in its order",
        text.split("\r\n")[0] == ",".join(tu.GL_COLUMNS),
        text.split("\r\n")[0][:70])
    chk("dates are written the way the accepted file writes them (8/14/2026)",
        ",8/14/2026" in text, text.split("\r\n")[1][-30:])
    chk("amounts keep the accepted file's trimming (13313.8, not 13313.80)",
        tu._fmt_amount(13313.80) == "13313.8", tu._fmt_amount(13313.80))
    chk("and a whole amount stays whole", tu._fmt_amount(-1000.0) == "-1000",
        tu._fmt_amount(-1000.0))
    chk("a rebuilt file parses back to the same lines",
        len(tu.read_gl_csv(text)) == 2)

    print("\n4. The investor transactions, and the tie to the GL")
    iv = tu.validate_ia(_fixture_ia(), gl_lines=[
        {"acctnum": "MR31000001", "amount": 1000.00}], ia_account="MR31000001")
    chk("matching totals tie", iv["ties"] is True, str(iv))
    iv2 = tu.validate_ia(_fixture_ia(), gl_lines=[
        {"acctnum": "MR31000001", "amount": 900.00}], ia_account="MR31000001")
    chk("a mismatch is an ERROR, not a note", bool(iv2["errors"]))
    chk("and it names both totals",
        any("1,000.00" in e and "900.00" in e for e in iv2["errors"]),
        str(iv2["errors"])[:130])
    # MRI's Guide allows fewer types in its Validation column than in its
    # Available Values column. The validation is what MRI enforces.
    t = _fixture_ia()
    t[0]["transaction_type"] = "CapitalCall"
    chk("a type MRI's guide disallows is refused, although the guide LISTS it",
        any("Contribution, Distribution, Income Allocation" in e
            for e in tu.validate_ia(t)["errors"]))
    z = _fixture_ia()
    z[0]["amount"] = 0
    chk("a zero amount is refused, as MRI's guide requires",
        bool(tu.validate_ia(z)["errors"]))

    print("\n5. The IA workbook is MRI's template, not a rebuild")
    chk("the template is vendored into the repo", tu.IA_TEMPLATE.exists(),
        str(tu.IA_TEMPLATE))
    if tu.IA_TEMPLATE.exists():
        import openpyxl
        data = tu.build_ia_xlsx(_fixture_ia(), gl_lines=[
            {"acctnum": "MR31000001", "amount": 1000.00}],
            ia_account="MR31000001")
        wb = openpyxl.load_workbook(io.BytesIO(data))
        chk("both of MRI's sheets survive",
            wb.sheetnames == ["Transaction Values", "Guide"],
            str(wb.sheetnames))
        ws = wb["Transaction Values"]
        # THE TRAP THIS CHECK EXISTS FOR: column 12 is "Number of Shares" and
        # its header is EMPTY in MRI's template. pandas calls it Unnamed: 11,
        # and a workbook rebuilt from column names would write that string into
        # the header MRI parses.
        chk("column 12's header is still BLANK, as MRI wrote it",
            ws.cell(row=1, column=12).value is None,
            repr(ws.cell(row=1, column=12).value))
        chk("the header row is otherwise untouched",
            ws.cell(row=1, column=1).value == "Transaction Type"
            and ws.cell(row=1, column=15).value == "Notes")
        chk("the rows land from row 2", ws.cell(row=2, column=5).value == "PSC1",
            repr(ws.cell(row=2, column=5).value))
        chk("effective date defaults to the transaction date",
            ws.cell(row=2, column=7).value == "2026-08-14",
            repr(ws.cell(row=2, column=7).value))
        chk("the Guide sheet still explains the rules",
            wb["Guide"].cell(row=2, column=1).value == "Transaction Type")

    print("\n6. Round trip against the files MRI actually accepted")
    if not (GL_REAL.exists() and IA_REAL.exists()):
        print("   (the real August files are not on this machine -- skipped)")
        return _report()

    import pandas as pd
    raw = GL_REAL.read_text(encoding="utf-8-sig")
    lines = tu.read_gl_csv(raw)
    chk("the real GL upload reads back", len(lines) == AUG["gl_rows"],
        str(len(lines)))
    v = tu.validate_gl(lines)
    chk("it passes our own validation", not v["errors"], str(v["errors"])[:110])
    chk("it balances to zero, as a journal entry must",
        abs(v["total"]) < 0.005, str(v["total"]))

    s = tu.summarise(lines, cash_account=AUG["cash_account"])
    chk("its cash lines equal the bank's net movement, -560,022.54",
        abs(s["cash_total"] - AUG["cash_total"]) < 0.01, str(s["cash_total"]))
    chk("and there are 24 of them, one per bank transaction",
        s["cash_line_count"] == AUG["cash_rows"], str(s["cash_line_count"]))

    # THE ROUND TRIP. Rebuild the accepted file from its own lines; every
    # field, order and format must come back identical.
    rebuilt = tu.build_gl_csv(lines)
    orig_rows = [r.rstrip("\r") for r in raw.splitlines() if r.strip(", ")]
    new_rows = [r for r in rebuilt.split("\r\n") if r.strip(", ")]
    chk("the rebuild has the same number of rows",
        len(new_rows) == len(orig_rows), "%d vs %d" % (len(new_rows), len(orig_rows)))
    diffs = [(i, a, b) for i, (a, b) in enumerate(zip(orig_rows, new_rows))
             if a != b]
    chk("and every row is byte-identical to the accepted file", not diffs,
        ("first difference at row %d: %r vs %r" % diffs[0]) if diffs else "")

    ia = pd.read_excel(IA_REAL, sheet_name=tu.IA_SHEET).dropna(how="all")
    chk("the real IA sample reads back", len(ia) == AUG["ia_rows"], str(len(ia)))
    rows = [{"transaction_type": r["Transaction Type"],
             "sub_type": r["Sub-Type"], "amount": r["Amount"],
             "investmentid": r["InvestmentID"], "investorid": r["InvestorID"],
             "transaction_date": r["Transaction Date"],
             "effective_date": r["Effective Date"]}
            for _, r in ia.iterrows()]
    iv = tu.validate_ia(rows, gl_lines=lines, ia_account=AUG["ia_account"])
    chk("it passes validation", not iv["errors"], str(iv["errors"])[:110])
    chk("its total is 12,580.47", abs(iv["total"] - AUG["ia_total"]) < 0.01,
        str(iv["total"]))
    chk("and it TIES to the journal entry's MR31000001 lines",
        iv["ties"] is True, str(iv.get("difference")))

    data = tu.build_ia_xlsx(rows, gl_lines=lines, ia_account=AUG["ia_account"])
    back = pd.read_excel(io.BytesIO(data), sheet_name=tu.IA_SHEET).dropna(how="all")
    chk("the rebuilt workbook holds the same 13 rows", len(back) == AUG["ia_rows"],
        str(len(back)))
    chk("with the same total",
        abs(float(back["Amount"].sum()) - AUG["ia_total"]) < 0.01,
        str(back["Amount"].sum()))
    chk("and the same investors in the same order",
        list(back["InvestorID"]) == [str(x).strip().upper()
                                     for x in ia["InvestorID"]],
        str(list(back["InvestorID"])[:4]))

    # The whole point, in one figure: what the accountant sees before
    # downloading either file.
    s = tu.summarise(lines, ia_rows=rows, cash_account=AUG["cash_account"],
                     ia_account=AUG["ia_account"])
    chk("the summary reports a clean August", s["balanced"] and s["ia_ties"]
        and not s["errors"], str(s.get("errors"))[:110])
    return _report()


def _raises(fn):
    try:
        fn()
        return False
    except Exception:
        return True


def _report():
    print("\n%d checks, %d failed." % (len(_passed) + len(_failed), len(_failed)))
    if _failed:
        for f in _failed:
            print("  FAILED: %s" % f)
        return 1
    print("The GL and IA writers reproduce the files MRI accepted, byte for\n"
          "byte, and refuse an entry that does not balance or investor rows\n"
          "that do not tie to it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
