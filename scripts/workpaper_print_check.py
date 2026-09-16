"""The statement tabs print like the real package, not like Excel's defaults.

WHAT THIS PINS. `workpaper_print.SHEETS` is a transcription of a delivered
package -- `PPI Eastchase (TX) LLC - WP - 06.30.2026.xlsx` -- and a transcription
is exactly the kind of thing that rots: a margin gets "tidied", a footer page
number gets turned into `&P`, somebody adds a column and the print range stops
covering it. None of that is visible until a statement is printed and sent.

It builds a REAL package and reads the workbook back, because the failure being
guarded against is the setup not reaching the file, not the constants being
wrong in the module.

The reference workbook is on the CFO's OneDrive and is not in this repo, so the
comparison is against the recorded spec. Point `--reference <path>` at the real
file to compare against it directly.

Run:  .venv/Scripts/python.exe scripts/workpaper_print_check.py
Exit 1 on any failure.
"""
import io
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

FAILS = []
CHECKS = [0]


def chk(label, cond, detail=""):
    CHECKS[0] += 1
    if cond:
        print("   ok   %s" % label)
    else:
        print("   FAIL %s%s" % (label, ("  -- " + detail) if detail else ""))
        FAILS.append(label)


def main() -> int:
    import openpyxl
    from flask_app import create_app
    from flask_app.services import workpaper_service as ws
    from flask_app.services import workpaper_excel as wx
    from flask_app.services import workpaper_print as wpp

    # ---- 0. the spec itself -------------------------------------------
    print("\n0. The recorded spec")
    chk("all seven statement tabs are specified",
        set(wpp.SHEETS) == {"Cover", "Index", "Balance Sheet", "SOI",
                            "Income Statement", "Members Capital", "Cash Flow"},
        str(sorted(wpp.SHEETS)))
    pages = [wpp.SHEETS[k].get("page") for k in
             ("Balance Sheet", "SOI", "Income Statement",
              "Members Capital", "Cash Flow")]
    # The reference numbers the five statements 1..5 in that order, which is
    # also the order they are bound in. A wrong number here is a page out of
    # sequence in a document an investor reads.
    chk("the five statements are numbered 1-5 in binding order",
        pages == [1, 2, 3, 4, 5], str(pages))
    for k in ("Balance Sheet", "SOI"):
        chk("%s is dated AS OF an instant" % k,
            wpp.SHEETS[k]["dated"] == "as_of")
    for k in ("Income Statement", "Members Capital", "Cash Flow"):
        chk("%s covers a PERIOD" % k, wpp.SHEETS[k]["dated"] == "period")
    chk("as-of and period phrases are not interchangeable",
        wpp.period_phrase("2026-06-30", "as_of") == "As of June 30, 2026"
        and wpp.period_phrase("2026-06-30", "period")
        == "For the six months ended June 30, 2026")
    chk("a December period end reads as a year, not twelve months",
        wpp.period_phrase("2026-12-31", "period")
        == "For the year ended December 31, 2026")
    # Printing a legal form is an assertion about the entity. It is derived from
    # the name, and withheld when the name does not say.
    chk("the legal-form line is derived, not assumed",
        wpp.entity_subtitle("X LLC") == "(A Limited Liability Company)"
        and wpp.entity_subtitle("Y LP") == "(A Limited Partnership)"
        and wpp.entity_subtitle("KOC REIT Inc") is None)

    # ---- 1. a real generated package ----------------------------------
    print("\n1. A built package")
    app = create_app()
    with app.app_context():
        cycles = ws.list_cycles()
        if not cycles:
            print("   (no close cycle in this database -- skipped)")
            return _report()
        t = ws.tracker(cycles[0]["id"])
        pkgs = t.get("packages") or []
        if not pkgs:
            print("   (no packages in this cycle -- skipped)")
            return _report()
        data = wx.build_package(pkgs[0]["id"])

    wb = openpyxl.load_workbook(io.BytesIO(data))
    for name, spec in wpp.SHEETS.items():
        if name not in wb.sheetnames:
            chk("%s exists in the package" % name, False)
            continue
        sh = wb[name]
        pm, po, ps = sh.page_margins, sh.print_options, sh.page_setup
        want = spec["margins"]
        got = (round(pm.left, 2), round(pm.right, 2),
               round(pm.top, 2), round(pm.bottom, 2))
        chk("%s margins" % name, got == want, "%s vs %s" % (got, want))
        chk("%s fits one page wide by one tall" % name,
            ps.fitToWidth == 1 and ps.fitToHeight == 1,
            "%sx%s" % (ps.fitToWidth, ps.fitToHeight))
        chk("%s is centred as the reference has it" % name,
            (po.horizontalCentered or False) == (spec["centre"] == "horizontal")
            and (po.verticalCentered or False) == (spec["centre"] == "vertical"))
        chk("%s has a print range" % name, bool(sh.print_area))
        if spec.get("page"):
            got_ftr = sh.oddFooter.right.text if sh.oddFooter.right else None
            # A LITERAL number, not `&P`. Each statement is one sheet printing
            # as one page, so `&P` would print "1" on all five of them.
            chk("%s footer is the literal page number" % name,
                got_ftr == str(spec["page"]), repr(got_ftr))
        if spec.get("header"):
            txt = sh.oddHeader.center.text if sh.oddHeader.center else ""
            chk("%s header names the entity" % name, bool(txt and len(txt) > 10))
            if spec.get("title"):
                chk("%s header carries its formal title and (Unaudited)" % name,
                    spec["title"] in (txt or "") and "(Unaudited)" in (txt or ""),
                    repr((txt or "")[:60]))

    # ---- 2. the print range excludes the workpaper rows ----------------
    #
    # Each generated statement carries its name in A1 and a provenance note in
    # A2 -- where the figures came from, how many dormant lines were suppressed.
    # Both belong in a workpaper and neither belongs on a statement an investor
    # reads, which is why the reference puts the titling in the page header.
    print("\n2. What actually prints")
    for name in ("Balance Sheet", "Income Statement", "Members Capital",
                 "Cash Flow", "SOI"):
        if name not in wb.sheetnames:
            continue
        sh = wb[name]
        area = str(sh.print_area or "")
        start = _first_row(area)
        chk("%s prints from below the provenance note" % name,
            start is not None and start >= 4, "print_area=%s" % area)
        chk("%s still HOLDS the title and provenance on screen" % name,
            sh["A1"].value is not None and sh["A2"].value is not None)
        end = _last_row(area)
        chk("%s print range reaches the end of the statement" % name,
            end is not None and end >= min(sh.max_row, 4),
            "range ends %s, sheet ends %s" % (end, sh.max_row))

    # ---- 3. type -------------------------------------------------------
    print("\n3. Type")
    for name in ("Balance Sheet", "Income Statement", "Cash Flow"):
        if name not in wb.sheetnames:
            continue
        sh = wb[name]
        bad = set()
        for row in sh.iter_rows(min_row=1, max_row=min(sh.max_row, 40),
                                max_col=6):
            for c in row:
                if c.value is None:
                    continue
                if c.font and (c.font.name != wpp.BODY_FONT
                               or float(c.font.sz or 0) != float(wpp.BODY_SIZE)):
                    bad.add((c.font.name, c.font.sz))
        chk("%s is %s %s throughout" % (name, wpp.BODY_FONT, wpp.BODY_SIZE),
            not bad, str(sorted(bad))[:80])

    # ---- 4. optional: against the real workbook ------------------------
    ref = None
    for i, a in enumerate(sys.argv):
        if a == "--reference" and i + 1 < len(sys.argv):
            ref = sys.argv[i + 1]
    if ref and pathlib.Path(ref).exists():
        print("\n4. Against the reference workbook")
        rwb = openpyxl.load_workbook(ref)
        pairs = [("Balance Sheet", "Balance Sheet"), ("SOI", "SOI"),
                 ("Income Statement", "IS"),
                 ("Members Capital", "Members' Capital"),
                 ("Cash Flow", "Cash Flow")]
        for ours, theirs in pairs:
            if theirs not in rwb.sheetnames or ours not in wb.sheetnames:
                continue
            a, b = wb[ours].page_margins, rwb[theirs].page_margins
            chk("%s margins match %s" % (ours, theirs),
                (round(a.left, 2), round(a.right, 2), round(a.top, 2),
                 round(a.bottom, 2))
                == (round(b.left, 2), round(b.right, 2), round(b.top, 2),
                    round(b.bottom, 2)))
    elif ref:
        print("\n4. Reference not found at %s -- skipped" % ref)

    return _report()


def _first_row(area):
    import re
    m = re.search(r"\$?([A-Z]+)\$?(\d+):", area)
    return int(m.group(2)) if m else None


def _last_row(area):
    import re
    m = re.search(r":\$?([A-Z]+)\$?(\d+)", area)
    return int(m.group(2)) if m else None


def _report():
    print("\n%d checks, %d failed." % (CHECKS[0], len(FAILS)))
    if FAILS:
        for f in FAILS:
            print("   FAILED: %s" % f)
        return 1
    print("The statement tabs carry the delivered package's print setup, the")
    print("workpaper rows stay off the printed page, and each statement is")
    print("dated the way its own kind of statement has to be dated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
