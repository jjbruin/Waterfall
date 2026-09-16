"""Guardrail: multi-loan terms, one page per subtab, fund-group separators.

Covers Prompt C of the Sep 2 2026 work order:

  1. the Loan tab lists each facility's real Rate and Maturity instead of
     collapsing to "Various", largest first and in the SAME order across both
     columns, and the deals it does that for are genuinely multi-loan rather
     than leftovers of the Loan_Date fan-out
  2. Financial, Operating and Loan each print to exactly ONE page with no row
     dropped
  3. horizontal separators break up the fund groups on those three subtabs and
     NOT on Summary

Parts 1 and 3 are offline — the real ``_loan_terms`` against injected rows, and
the print stylesheet read as source. Part 2 needs a rendered PDF; produce one
with scripts/snapshot_print_check.mjs (``WF_UPSTREAM=http://127.0.0.1:5000``
prints against a local backend) and pass its path.

    python scripts/snapshot_print_formatting_check.py terms
    python scripts/snapshot_print_formatting_check.py fanout          # needs the DB
    python scripts/snapshot_print_formatting_check.py separators
    python scripts/snapshot_print_formatting_check.py pages <file.pdf> [payload_dir]
    python scripts/snapshot_print_formatting_check.py fonts <file.pdf>
    python scripts/snapshot_print_formatting_check.py all   <file.pdf> [payload_dir]
"""
from __future__ import annotations

import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PRINT_VIEW = os.path.join(ROOT, "vue_app", "src", "views",
                          "PortfolioSnapshotPrintView.vue")

CHECKS: list = []


def chk(label, cond, detail=""):
    CHECKS.append(bool(cond))
    print("  [{}] {}".format("PASS" if cond else "FAIL", label)
          + ("\n           " + detail if detail else ""))


# ── 1. Per-loan terms ─────────────────────────────────────────────────────
#
# The four deals that rendered "Various" at 26Q1, with their real rows.
MULTI_LOAN = {
    "Ascent on Steamboat": [
        {"LoanID": 288, "mOrigLoanAmt": 25350000.0, "nRate": 0.0368,
         "vIntType": "Fixed", "dtEvent": "7/1/2026"},
        {"LoanID": 289, "mOrigLoanAmt": 7372000.0, "nRate": 0.0559,
         "vIntType": "Fixed", "dtEvent": "7/1/2026"},
    ],
    "Brainerd Place Apartments": [
        {"LoanID": 291, "mOrigLoanAmt": 31000000.0, "nRate": 0.0425,
         "vIntType": "Fixed", "dtEvent": "6/30/2026"},
        {"LoanID": 311, "mOrigLoanAmt": 33400000.0, "vIndex": "SOFR",
         "vSpread": 0.065, "vIntType": "Variable", "dtEvent": "12/1/2026"},
    ],
    "Mount Prospect Plaza": [
        {"LoanID": 293, "mOrigLoanAmt": 16600000.0, "nRate": 0.0535,
         "vIntType": "Fixed", "dtEvent": "8/18/2027"},
        {"LoanID": 316, "mOrigLoanAmt": 6000000.0, "vIndex": "SOFR",
         "vSpread": 0.025, "vIntType": "Variable", "dtEvent": "8/18/2027"},
    ],
    "Poplar Prairie": [
        {"LoanID": 301, "mOrigLoanAmt": 22900000.0, "nRate": 0.0746,
         "vIntType": "Fixed", "dtEvent": "9/28/2027"},
        {"LoanID": 302, "mOrigLoanAmt": 6530000.0, "vIndex": "SOFR",
         "vSpread": 0.0285, "vIntType": "Variable", "dtEvent": "9/28/2027"},
    ],
}

EXPECTED = {
    # largest facility first, and a field whose values are all the same stays
    # a single value rather than being repeated once per loan
    "Ascent on Steamboat": ("3.7% fixed | 5.6% fixed", "7/1/2026"),
    "Brainerd Place Apartments": ("SOFR + 650 | 4.2% fixed",
                                  "12/1/2026 | 6/30/2026"),
    "Mount Prospect Plaza": ("5.3% fixed | SOFR + 250", "8/18/2027"),
    "Poplar Prairie": ("7.5% fixed | SOFR + 285", "9/28/2027"),
}


def check_terms() -> None:
    import pandas as pd
    from flask_app.services.portfolio_snapshot_loan import _loan_terms, VARIOUS

    print("\n1. Multi-loan deals list their loans")
    for name, rows in MULTI_LOAN.items():
        out = _loan_terms(pd.DataFrame(rows))
        want_rate, want_mat = EXPECTED[name]
        chk(f"{name}: Rate", out["rate_display"] == want_rate,
            f"got {out['rate_display']!r}   want {want_rate!r}")
        chk(f"{name}: Maturity", out["maturity_display"] == want_mat,
            f"got {out['maturity_display']!r}   want {want_mat!r}")
        chk(f"{name}: neither cell says {VARIOUS!r}",
            VARIOUS not in (out["rate_display"], out["maturity_display"]))
        # Positional pairing only means anything if both columns are piped.
        # Where only one is, the single value applies to every loan listed.
        r_n = len((out["rate_display"] or "").split(" | "))
        m_n = len((out["maturity_display"] or "").split(" | "))
        chk(f"{name}: rate and maturity pair up",
            r_n == m_n or 1 in (r_n, m_n),
            f"{r_n} rate(s) against {m_n} maturity(ies)")
        chk(f"{name}: the breakdown is published for audit",
            len(out.get("terms_list") or []) == 2,
            str(out.get("terms_list")))

    print("\n1b. A single-loan deal is untouched")
    one = _loan_terms(pd.DataFrame([MULTI_LOAN["Poplar Prairie"][0]]))
    chk("one loan renders its own terms, unpiped",
        one["rate_display"] == "7.5% fixed" and "|" not in one["rate_display"],
        f"got {one['rate_display']!r} / {one['maturity_display']!r}")
    chk("and is not flagged multi", one["various"] is False)

    print("\n1c. Two loans on IDENTICAL terms collapse to one line")
    same = MULTI_LOAN["Ascent on Steamboat"][0]
    dup = _loan_terms(pd.DataFrame([same, dict(same, LoanID=999)]))
    chk("identical facilities are not printed twice",
        "|" not in (dup["rate_display"] or ""),
        f"got {dup['rate_display']!r}")


# ── 2. Genuine multi-loan, not the Loan_Date fan-out ──────────────────────

def check_fanout() -> None:
    """Every deal that pipes its terms has 2+ DISTINCT LoanIDs in the data.

    The fan-out bug (fixed at data_service._collapse_loan_date_events, 6bbfbf6)
    repeated ONE facility once per Loan_Date event, which would look identical
    on the page. Distinct LoanIDs is what separates the two.
    """
    print("\n2. The piped deals are genuinely multi-loan")
    try:
        import sqlalchemy as sa
        import pandas as pd
        eng = sa.create_engine("sqlite:///" + os.path.join(ROOT, "waterfall.db"))
        df = pd.read_sql("SELECT vCode, LoanID FROM loans", eng)
    except Exception as exc:
        chk("local loans table readable", False, str(exc)[:120])
        return
    df["vCode"] = df["vCode"].astype(str).str.upper()
    for vcode in ("P0000065", "P0000067", "P0000069", "P0000082"):
        sub = df[df["vCode"] == vcode]
        chk(f"{vcode}: {len(sub)} row(s), {sub['LoanID'].nunique()} distinct LoanID",
            len(sub) >= 2 and sub["LoanID"].nunique() == len(sub),
            "one row per facility — no repeated LoanID, so not a fan-out")


# ── 3. Fund-group separators ──────────────────────────────────────────────

def check_separators() -> None:
    print("\n3. Fund-group separators on the three data subtabs")
    src = io.open(PRINT_VIEW, encoding="utf-8").read()
    chk("a rule above every fund subtotal",
        ":deep(table.grid tr.subtotal td)" in src)
    chk("a heavier rule above the portfolio total",
        ":deep(table.grid tfoot tr:first-child td)" in src)
    chk("the inter-group spacer is preserved on paper",
        ":deep(table.grid tr.spacer td)" in src)
    chk("light VERTICAL rules at the column boundaries",
        "border-right: 0.5px solid" in src
        and ":deep(table.grid th:last-child)" in src,
        "hairline between columns, none on the last")
    chk("a slightly stronger rule at the zone boundaries",
        "border-left: 0.5px solid #c9ced6" in src,
        "before the TIAA Investment block and before the manual columns")
    chk("scoped to table.grid, which Summary does not render",
        "table.grid" in src and ".summary .card" in src,
        "Summary is narrative and charts; it has no table.grid")


# ── 4. One page per subtab, nothing dropped ───────────────────────────────

def check_pages(pdf_path: str, payload_dir: str = "") -> None:
    print("\n4. Financial, Operating and Loan each print to ONE page")
    try:
        pages = _pdf_text(pdf_path)
    except ImportError:
        chk("a PDF reader is available", False,
            "pip install pymupdf   (or pdfplumber)")
        return

    chk("the document is 4 pages — Summary + one per subtab",
        len(pages) == 4, f"got {len(pages)}")
    probes = {"Financial": "TIAA Investment", "Operating": "ECON OCC",
              "Loan": "DEBT YIELD"}
    for tab, probe in probes.items():
        hits = [i + 1 for i, t in enumerate(pages) if probe in t]
        chk(f"{tab} occupies exactly one page", len(hits) == 1,
            f"found on page(s) {hits}")

    check_margins(pdf_path)

    if not payload_dir:
        print("      (pass a payload dir to also check no row was clipped)")
        return
    for tab, page_idx in (("financial", 1), ("operating", 2), ("loan", 3)):
        fn = os.path.join(payload_dir, tab + ".json")
        if not os.path.exists(fn):
            continue
        d = json.load(io.open(fn, encoding="utf-8"))
        d = d.get(tab, d)
        rows = []
        for b in (d.get("groups") or {}).values():
            rows += (b["deals"] if isinstance(b, dict) else b)
        rows += d.get("ownership_flagged") or []
        txt = pages[page_idx]
        missing = [r["name"] for r in rows
                   if r["name"].split("(")[0].strip()[:18] not in txt]
        chk(f"{tab}: all {len(rows)} deals are on the page",
            not missing, "missing: " + ", ".join(missing[:5]))


#: The .print-page padding box, in inches, on the landscape sheet: an 11.00in
#: page with 0.5in of side padding. Anything drawn outside it is past the margin.
PRINT_BOX = (0.50, 10.50)

#: The Financial band spans FOUR columns — % of Pref, Invested, Un-funded, Total
#: Commitment — not the deal-level cap stack to its left. See the `span-tiaa`
#: cell in SnapshotFinancial.vue.
TIAA_BAND_COLUMNS = 4


def _rule_segments(pdf_path: str, page_index: int) -> list:
    """``[(y0, x0, x1, fill)]`` for the thin rules drawn on one page, in inches.

    Borders are vector strokes and thin filled rectangles, not text, so nothing
    in the text layer can answer whether a table fits or where a rule runs. Fat
    rectangles — page and cell backgrounds — are filtered out by the dimension
    test, which is what keeps the full-page background from reading as a rule
    spanning 0.00 to 11.00in.
    """
    try:
        import fitz
    except ImportError:
        fitz = None
    if fitz is not None:
        page = fitz.open(pdf_path)[page_index]
        out = []
        for d in page.get_drawings():
            for it in d["items"]:
                if it[0] == "re":
                    r = it[1]
                    thin = (r.height <= 1.6 and r.width > 6) or \
                           (r.width <= 1.6 and r.height > 6)
                    if thin:
                        out.append((r.y0 / 72, r.x0 / 72, r.x1 / 72,
                                    d.get("fill")))
                elif it[0] == "l":
                    p1, p2 = it[1], it[2]
                    out.append((min(p1.y, p2.y) / 72, min(p1.x, p2.x) / 72,
                                max(p1.x, p2.x) / 72, d.get("color")))
        return out
    # pdfplumber backend. The repo's .venv carries this and not PyMuPDF, so a
    # fitz-only helper made every check here report "needs pymupdf" — five FAILs
    # that say nothing about the document. Same mistake as _pdf_prose's.
    try:
        import pdfplumber
    except ImportError:
        return []
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_index]
        out = []
        for r in list(page.rects) + list(page.lines):
            w = abs(r["x1"] - r["x0"])
            h = abs(r["bottom"] - r["top"])
            if not ((h <= 1.6 and w > 6) or (w <= 1.6 and h > 6)):
                continue
            col = r.get("non_stroking_color") or r.get("stroking_color")
            if isinstance(col, (int, float)):
                col = (col, col, col)
            out.append((r["top"] / 72, r["x0"] / 72, r["x1"] / 72, col))
        return out


def check_margins(pdf_path: str) -> None:
    """Every table fits INSIDE the page margin, and the band rule marks its band.

    Vertical fit had a check (``check_pages``); horizontal fit had none, and
    that is what let the Financial table run past the right margin unnoticed.
    `table.grid` sets ``white-space: nowrap``, so ``width: 100%`` is a floor and
    not a ceiling: add a column, or a longer header, and the table simply grows
    out of the page. Measured on live 26Q2 before the fix it was 10.224in inside
    a 10.000in box, and the Net ROE column's right border was off the sheet.

    The band rule is here rather than with the border checks because it is the
    same failure in a different direction: a rule that runs the full width of
    the row says nothing about which columns are TIAA's.
    """
    print("\n4b. Tables fit the page WIDTH, and the TIAA band rule marks it")
    lo, hi = PRINT_BOX
    for idx, label in ((1, "Financial"), (2, "Operating"), (3, "Loan")):
        segs = _rule_segments(pdf_path, idx)
        if not segs:
            chk(f"{label}: rules readable (needs pymupdf)", False)
            continue
        x0 = min(s[1] for s in segs)
        x1 = max(s[2] for s in segs)
        chk(f"{label}: table is inside the {lo}–{hi}in printable box",
            x0 >= lo - 0.01 and x1 <= hi + 0.01,
            f"drawn {x0:.3f}..{x1:.3f}in (width {x1 - x0:.3f}in)")

    # The band row's bottom rule must cover the band and NOTHING else. Before
    # the fix it ran 0.510..10.729in — the full table — because
    # `.spanrow th { border-bottom: none }` lost on specificity to
    # `table.grid th` and every empty cell drew the ordinary header underline.
    segs = _rule_segments(pdf_path, 1)
    # Horizontal only. The column separators are vertical rules ~0.007in wide
    # whose top edge falls in the same y range, and counting them made this read
    # 14 segments on a row that has at most a handful.
    band = [s for s in segs if 0.80 < s[0] < 0.88 and (s[2] - s[1]) > 0.05]
    dark = [s for s in band if s[3] and s[3][0] < 0.6]
    light = [s for s in band if not (s[3] and s[3][0] < 0.6)]
    chk("Financial: the TIAA band carries its own rule",
        bool(dark), f"{len(band)} segments on the band row, none of them dark")
    chk(f"Financial: the band rule spans its {TIAA_BAND_COLUMNS} columns",
        len(dark) == TIAA_BAND_COLUMNS,
        f"{len(dark)} dark segment(s): "
        + ", ".join(f"{d[1]:.2f}..{d[2]:.2f}in" for d in dark))
    # `not light` is trivially true when nothing was read at all, which is how
    # this passed while the three checks above it were failing for want of a
    # PDF reader. It only means anything once the band has been found.
    chk("Financial: no rule under the REST of the band row — it would read as "
        "a full-width line, not a group marker",
        bool(band) and not light,
        ("no band row found — the probe, not the document" if not band else
         "light segments at " + ", ".join(f"{s[1]:.2f}..{s[2]:.2f}in"
                                          for s in light[:4])))


def _pdf_text(pdf_path: str) -> list:
    """The text of each page, through whichever PDF reader is installed.

    Same reason as ``_pdf_chars``: this check is the one that answers the
    question the work order asks — does each subtab fit on one page — and it
    was skipping on any machine without pdfplumber, which includes this repo's
    own environment.
    """
    try:
        import fitz
    except ImportError:
        fitz = None
    if fitz is not None:
        return [page.get_text() for page in fitz.open(pdf_path)]
    import pdfplumber                                   # noqa: F401
    with pdfplumber.open(pdf_path) as pdf:
        return [(p.extract_text() or "") for p in pdf.pages]


def _pdf_chars(pdf_path: str) -> list:
    """Per page, ``[(char, size, fontname)]`` in reading order.

    Reads through PyMuPDF or pdfplumber, whichever is installed. This used to
    require pdfplumber alone and simply reported itself unavailable without it,
    which is a check that does not run — and a check that does not run is worth
    less than no check, because the suite still prints a line for it.
    """
    try:
        import fitz
    except ImportError:
        fitz = None
    if fitz is not None:
        out = []
        for page in fitz.open(pdf_path):
            chars = []
            for b in page.get_text("dict")["blocks"]:
                if b["type"] != 0:
                    continue
                for line in b["lines"]:
                    for s in line["spans"]:
                        size = round(s["size"], 1)
                        font = s["font"].split("+")[-1]
                        chars.extend((c, size, font) for c in s["text"])
            out.append(chars)
        return out
    import pdfplumber                                   # noqa: F401
    with pdfplumber.open(pdf_path) as pdf:
        return [[(c["text"], round(c["size"], 1),
                  c["fontname"].split("+")[-1]) for c in pg.chars]
                for pg in pdf.pages]


def _pdf_prose(pdf_path: str, page_index: int, header: str) -> list:
    """``[(text, size, font)]`` for the spans in the COMMENT COLUMN of one page.

    Located by the column's own header rather than by a phrase from the data:
    the comment wording changes every quarter, so a hardcoded probe
    ("Occupancy held at") stops being on the page and the check SKIPS —
    printing a line and asserting nothing. Both comment probes were skipping on
    live 26Q2 when this was rewritten.

    Anchoring on the header also keeps the page's other prose out. Selecting
    "any long run of words" instead pulled in the subtitle above the table
    (8.25pt) and the Loan tab's italic excluding-development footnote below it,
    and reported three different comment sizes on a page that has one.
    """
    spans = _page_spans(pdf_path, page_index)
    if not spans:
        return []
    head = next((s for s in spans
                 if header.lower() in s["text"].strip().lower()), None)
    if head is None:
        return []
    x0, y_head = head["x0"] - 2, head["y1"]
    out = []
    for s in spans:
        t = s["text"].strip()
        if s["x0"] < x0 or s["y0"] < y_head:
            continue                      # left of the column, or above it
        if "italic" in s["font"].lower():
            continue                      # the tfoot footnote, not a comment
        if len(t) >= 20 and t.count(" ") >= 2:
            out.append((t, round(s["size"], 1), s["font"].split("+")[-1]))
    return out


def _page_spans(pdf_path: str, page_index: int) -> list:
    """``[{text, size, font, x0, y0, y1}]`` for one page, from either reader.

    Both backends are implemented because the two machines that run this do not
    agree: the repo's ``.venv`` carries pdfplumber and not PyMuPDF, and a bare
    system interpreter here carried PyMuPDF and not pdfplumber. A helper that
    knew only one of them made the comment checks report "no prose found" —
    a FAIL that says the document is wrong when it is the probe that is missing.
    """
    try:
        import fitz
    except ImportError:
        fitz = None
    if fitz is not None:
        page = fitz.open(pdf_path)[page_index]
        return [{"text": s["text"], "size": s["size"],
                 "font": s["font"].split("+")[-1],
                 "x0": s["bbox"][0], "y0": s["bbox"][1], "y1": s["bbox"][3]}
                for b in page.get_text("dict")["blocks"] if b["type"] == 0
                for line in b["lines"] for s in line["spans"]
                if s["text"].strip()]
    try:
        import pdfplumber
    except ImportError:
        return []
    # pdfplumber gives characters, not spans. Group them into runs sharing a
    # baseline, a size and a font — which is what a span is.
    with pdfplumber.open(pdf_path) as pdf:
        chars = pdf.pages[page_index].chars
        runs, cur = [], None
        for c in chars:
            key = (round(c["top"], 1), round(c["size"], 1), c["fontname"])
            if cur is None or cur["key"] != key or c["x0"] - cur["x1"] > 3:
                cur = {"key": key, "text": c["text"], "size": c["size"],
                       "font": c["fontname"].split("+")[-1],
                       "x0": c["x0"], "x1": c["x1"],
                       "y0": c["top"], "y1": c["bottom"]}
                runs.append(cur)
            else:
                cur["text"] += c["text"]
                cur["x1"] = c["x1"]
        return [r for r in runs if r["text"].strip()]


def check_fonts(pdf_path: str) -> None:
    """Manual-input cells print at the table's size; comments one step below it.

    Measured off the PDF, not asserted off the CSS: a `<textarea>`, an `<input>`
    and a `<span class="cmt-text">` each reach their type size by a different
    route, and only the rendered document proves where each one landed.

    THE TWO CATEGORIES ARE NOT THE SAME, and this check used to require both to
    equal the table:

      * a MANUAL FIGURE is a figure. It sits in a column of figures and must be
        indistinguishable from the computed ones beside it — exactly equal.
      * a COMMENT is prose about a row, and prints at 7px against the table's
        8px, deliberately. It is what sets the Operating page's height: on live
        26Q2 the comment column wrapped 50 of 55 rows onto a second line and the
        page ran 1.13in over a single sheet. Shrinking the prose bought the page
        back; shrinking the figures would not have been enough.

    The regressions this was written for are still caught, because both were
    LARGER than the table, not smaller: form controls do not inherit font-size
    from their container (the UA gives them ~13.3px), and `.cmt-text` hardcodes
    12px. A comment is required to be smaller than the figures but within one
    step of them, so 12px fails and so would an illegible 4px.
    """
    print("\n5. Manual figures print at the table's size, comments just below")
    try:
        pages = _pdf_chars(pdf_path)
    except ImportError:
        chk("a PDF reader is available", False,
            "pip install pymupdf   (or pdfplumber)")
        return

    def sizes_for(chars, phrase):
        txt = "".join(c[0] for c in chars)
        i = txt.find(phrase)
        if i < 0:
            return None
        return sorted({(c[1], c[2]) for c in chars[i:i + len(phrase)]})

    #: A comment prints at 7px where the table is 8px. Bounded rather than
    #: pinned to 0.875 exactly, so a future half-step does not fail the suite
    #: while still refusing 12px above and anything unreadable below.
    COMMENT_RATIO = (0.80, 0.95)

    # Anchor: an ordinary deal-name cell on each page is the size the figures
    # must match and the comments are measured against.
    for idx, label, figures, comments in (
        (1, "Financial", ("5.87M", "4.4%"), ()),
        # Operating and Loan carry the comment column. The Financial page is
        # NOT probed for prose: its footnote block is prose at the same 7px by
        # design, and it is not a comment.
        (2, "Operating", (), "Operating comment"),
        (3, "Loan", (), "Loan comment"),
    ):
        if idx >= len(pages):
            chk(f"{label}: page {idx} exists", False,
                f"the document has {len(pages)} pages")
            continue
        anchor = sizes_for(pages[idx], "Evergreen Plaza")
        if not anchor:
            chk(f"{label}: anchor cell found", False,
                "no 'Evergreen Plaza' row on this page")
            continue
        for probe in figures:
            got = sizes_for(pages[idx], probe)
            if got is None:
                print(f"      (skipped {label} {probe!r} — not on the page)")
                continue
            chk(f"{label}: manual figure {probe!r} matches an ordinary cell",
                got == anchor, f"{got} against anchor {anchor}")
        if not comments:
            continue
        prose = _pdf_prose(pdf_path, idx, comments)
        a_size = anchor[0][0]
        # Vacuity guard. Every quarter has comments on these two pages; none
        # found means the detector broke, not that the page is clean. `chk`
        # here returns None, so it cannot gate the rest — the condition is
        # tested separately.
        found = len(prose) >= 3
        chk(f"{label}: comment prose found on the page ({len(prose)} runs)",
            found, "" if found else "no prose runs detected — that is the "
                                    "probe failing, not the document")
        if not found:
            continue
        sizes = sorted({s for _, s, _ in prose})
        ratios = [s / a_size for s in sizes] if a_size else [0]
        chk(f"{label}: every comment prints at ONE size",
            len(sizes) == 1, f"sizes found: {sizes}")
        chk(f"{label}: comments print below the figures, within one step",
            all(COMMENT_RATIO[0] <= r <= COMMENT_RATIO[1] for r in ratios),
            f"{sizes}pt against {a_size}pt anchor "
            f"(ratios {[round(r, 3) for r in ratios]}, want {COMMENT_RATIO})")
        chk(f"{label}: comments are in the table's own typeface",
            {f for _, _, f in prose} == {f for _, f in anchor},
            f"{sorted({f for _, _, f in prose})} against "
            f"{sorted({f for _, f in anchor})}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    pdf = sys.argv[2] if len(sys.argv) > 2 else ""
    pdir = sys.argv[3] if len(sys.argv) > 3 else ""
    if cmd == "terms":
        check_terms()
    elif cmd == "fanout":
        check_fanout()
    elif cmd == "separators":
        check_separators()
    elif cmd == "pages" and pdf:
        check_pages(pdf, pdir)
    elif cmd == "fonts" and pdf:
        check_fonts(pdf)
    elif cmd == "all" and pdf:
        check_terms()
        check_fanout()
        check_separators()
        check_pages(pdf, pdir)
        check_fonts(pdf)
    else:
        print(__doc__)
        raise SystemExit(2)
    print("\n  {}/{} checks passed".format(sum(CHECKS), len(CHECKS)))
    raise SystemExit(0 if all(CHECKS) else 1)
