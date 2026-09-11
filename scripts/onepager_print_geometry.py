"""Guardrail: measure what actually landed on the printed One Pager sheet.

Reads the PDFs written by scripts/onepager_print_sweep.mjs and reports, per deal:

  pages        how many sheets the deal printed on. The whole point of the
               out-of-flow chart is that this stays at 1 for every deal.
  text_bottom  y of the lowest piece of DOM text, in points from the top of the
               page. The ECharts canvas rasterises, so every string pdfplumber
               can extract is real page text and none of it is chart labels.
  chart_top    y of the top of the chart image, same origin.
  chart_h      the chart image's height in points. This is the number the
               "is it legible" question is really about.
  gap          chart_top - text_bottom. POSITIVE is dead space between the
               narrative and the chart — the blank band this change exists to
               fill. NEGATIVE is overlap, which is expected on the longest deal
               and is legible because .bp-section paints above the chart.

Why measure the PDF rather than the DOM: print CSS only exists on the print
path, and `@media print` rules (the fixed sheet height, the absolute chart, the
z-index pair) are not observable from a screen render. The PDF is the artifact.

Usage
  .venv/Scripts/python.exe scripts/onepager_print_geometry.py \
      --tag after [--compare before] [--deals scripts/onepager_print_population.txt]
"""
from __future__ import annotations

import argparse
import os
import sys

import pdfplumber

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PDF_DIR = os.path.join(ROOT, "vue_app", ".chartcheck")

# Letter portrait with App.vue's `@page { margin: 0 }`; .one-pager-page adds
# 0.4in top and bottom, so the sheet box is 792 - 57.6 = 734.4pt starting at
# y = 28.8pt. Kept here so a geometry number can be read against the box it
# lives in rather than against the raw paper.
PAGE_H = 792.0
PAD_TOP = 28.8
SHEET_H = 734.4


# Body text is set to 10.7px = 8.00pt exactly, the legibility floor 62161a9
# established and the one number on this page that must never move. Chrome's
# print-to-pdf silently scales the WHOLE document down when content overflows
# the paper width, and nothing else in the PDF announces that it happened — so
# the rendered height of a body glyph is what reveals it. Cap-height of the
# digits/letters in these table rows measures 8.0pt at 100%; anything smaller
# means the page was shrunk and every other number in this report is in scaled
# points rather than CSS points.
BODY_PT_AT_100 = 8.02

# The chart's own title starts this far below the top edge of its box, measured
# off the rendered raster. Text that reaches into that strip overlaps the chart
# BOX but not anything drawn in it, and prints clean. Only overlap deeper than
# this collides with chart ink, and that is the number worth failing on — bare
# geometric overlap over-reports by counting a 1pt intrusion into white space.
CHART_TOP_WHITESPACE_PT = 4.08


def measure(path: str) -> dict:
    """Page count plus the text and chart geometry of page one."""
    with pdfplumber.open(path) as pdf:
        pages = len(pdf.pages)
        p0 = pdf.pages[0]

        # Lowest DOM text on page one. `.print-date` and the title sit at the
        # top; the narrative is what reaches lowest.
        words = p0.extract_words() or []
        text_bottom = max((w["bottom"] for w in words), default=None)

        # Page scale, read off the body text. Take the MODE of glyph heights
        # rather than any single word: the page mixes 16px title, 11px section
        # headers and 10.7px body, and the body is by far the most common.
        heights = [round(w["bottom"] - w["top"], 2) for w in words]
        body_pt = max(set(heights), key=heights.count) if heights else None
        scale = (body_pt / BODY_PT_AT_100) if body_pt else None

        # The chart canvas is the page's one large raster. Guard against
        # incidental small images (none today) by taking the tallest.
        imgs = [im for im in (p0.images or []) if (im["y1"] - im["y0"]) > 20]
        chart = max(imgs, key=lambda im: im["y1"] - im["y0"]) if imgs else None
        chart_w = (chart["x1"] - chart["x0"]) if chart is not None else None
        # The printable column: .one-pager-page's 0.5in side padding inside a
        # 612pt sheet. A canvas wider than this is what triggers the shrink, so
        # it is reported next to the width rather than left to be worked out.
        column_w = p0.width - 72.0
        if chart is not None:
            # pdfplumber image y0/y1 are from the page BOTTOM; convert to a
            # top-down origin so every number in this report shares one axis.
            chart_top = p0.height - chart["y1"]
            chart_bot = p0.height - chart["y0"]
            chart_h = chart_bot - chart_top
        else:
            chart_top = chart_bot = chart_h = None

        # .chart-section's border-top: a full-width hairline sitting a few
        # points ABOVE the canvas. It is the topmost thing the chart draws, so
        # it — not the image edge, and not the chart title inside the image —
        # is what the narrative must clear. Measuring the ceiling against the
        # title instead put this rule straight through Burton's last line at
        # 195px while every image-based check still read "no collision".
        rule_top = None
        if chart_top is not None:
            rules = [r for r in (p0.rects or [])
                     if (r["x1"] - r["x0"]) > (p0.width - 100)
                     and (r["bottom"] - r["top"]) < 2
                     and chart_top - 12 < r["top"] < chart_top]
            if rules:
                rule_top = min(r["top"] for r in rules)

        # Every page's text, for the "nothing clipped" comparison.
        full_text = "\n".join((pg.extract_text() or "") for pg in pdf.pages)

    # Compare CONTENT, not the raw character count. A page rendered at a
    # different scale wraps its lines in different places, so the newline count
    # changes while not one character is lost — comparing len(text) reports
    # that as text disappearing. See _covers for why ALL whitespace goes.
    normalised = "".join(full_text.split())

    return {
        "pages": pages,
        "text_bottom": text_bottom,
        "chart_top": chart_top,
        "chart_bottom": chart_bot,
        "chart_h": chart_h,
        "chart_w": chart_w,
        "column_w": column_w,
        "rule_top": rule_top,
        "body_pt": body_pt,
        "scale": scale,
        "gap": (chart_top - text_bottom)
        if (chart_top is not None and text_bottom is not None) else None,
        "clear": ((rule_top if rule_top is not None else chart_top) - text_bottom)
        if (chart_top is not None and text_bottom is not None) else None,
        "text": full_text,
        "norm": normalised,
        "nchars": len(normalised),
    }


def _covers(after: str, before: str) -> bool:
    """Did the AFTER page print every character the BEFORE page did?

    Compares the whitespace-free character MULTISET. Three things move when a
    page is re-rendered at a different scale, none of which loses any text, and
    each of which defeats a stricter comparison:

      * line wrapping moves, so a hyphenated word splits in a new place —
        "five-year" on one page is "five-" + "year" on the other, which a
        word-level check reports as two words lost and one gained;
      * at 0.859 adjacent table cells sat close enough that pdfplumber ran them
        into one token, "$21.0MCoupon:", which separates properly at 100%;
      * a value that re-wraps onto its own line moves in pdfplumber's reading
        ORDER, so the character stream is permuted — "16%" reads back as "%16",
        and a loan's "(+1X24)" extension term lands elsewhere in the stream.
        Every such page came back with an IDENTICAL character count, which is
        what identifies it as reordering rather than loss.

    The multiset is invariant to all three and still fails on a genuinely
    dropped character, which is the question being asked: was any TEXT lost.
    """
    from collections import Counter
    return not (Counter(before) - Counter(after))


def pdf_path(vcode: str, quarter: str, tag: str) -> str:
    return os.path.join(PDF_DIR, f"onepager_{vcode}_{quarter}_{tag}.pdf")


def fmt(v, nd=1):
    return "-" if v is None else f"{v:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--compare", default=None,
                    help="a second tag to diff against (e.g. the before run)")
    ap.add_argument("--deals",
                    default=os.path.join(HERE, "onepager_print_population.txt"))
    ap.add_argument("--only", default=None, help="comma-separated vcodes")
    # WHICH PRINT LAYOUT IS IN FORCE. These are not styling preferences, they
    # are two different contracts, and the same measurement is a pass under one
    # and a failure under the other:
    #   unclipped — the chart is out of flow and the canvas is capped at the
    #     printable column, so the page must render at 100% and no text may be
    #     dropped. Overflow shows as narrative overlapping the chart.
    #   original  — the sheet is a fixed page with `overflow: hidden` and the
    #     canvas is wider than the column, so the page IS scaled (~0.96-0.99)
    #     and anything past the bottom IS clipped, by design. Asserting 100%
    #     scale here would report the intended layout as broken.
    # Text loss under `original` is expected and is not silently tolerated —
    # it is measured per deal by scripts/onepager_overflow_report.py, which is
    # the check that matters for that layout.
    ap.add_argument("--layout", choices=("unclipped", "original"),
                    default="unclipped",
                    help="which print contract to assert against")
    # Overlap between the narrative and the chart is a DESIGN DECISION, not a
    # defect, and which deals may overlap depends on the chart height in force:
    #   * at 180px only Poplar Prairie reaches the chart, so naming it kept the
    #     check green while a SECOND deal reaching the chart failed loudly;
    #   * at 300px — the chart's original size, restored deliberately — 26 of
    #     the 61 deals reach it, and that was accepted in exchange for never
    #     deleting narrative to make room. Pass `any` to say so at the call
    #     site, so the policy is visible in the command rather than buried.
    # What is NOT negotiable, and is still asserted unconditionally below:
    # every character prints, every deal is one page, and the page is at 100%.
    # Overlap hides part of a CHART; the alternative deleted investor TEXT.
    ap.add_argument("--allow-rule-hits", default="P0000082",
                    help="vcodes permitted to overlap the chart rule, "
                         "or 'any' when overlap is accepted by design")
    args = ap.parse_args()

    deals = []
    with open(args.deals, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            deals.append((parts[0], parts[1] if len(parts) > 1 else "2026-Q2"))
    if args.only:
        keep = set(args.only.split(","))
        deals = [d for d in deals if d[0] in keep]

    rows = []
    missing = []
    for vcode, quarter in deals:
        p = pdf_path(vcode, quarter, args.tag)
        if not os.path.exists(p):
            missing.append(vcode)
            continue
        m = measure(p)
        m["vcode"] = vcode
        m["quarter"] = quarter
        if args.compare:
            q = pdf_path(vcode, quarter, args.compare)
            m["cmp"] = measure(q) if os.path.exists(q) else None
        rows.append(m)

    if missing:
        print(f"MISSING {len(missing)} PDFs for tag '{args.tag}': "
              f"{', '.join(missing[:10])}"
              + (" ..." if len(missing) > 10 else ""))

    hdr = f"{'vcode':<11}{'pg':>3}{'scale':>7}{'body':>6}{'ch_w':>7}" \
          f"{'text_bot':>10}{'chart_top':>11}{'chart_h':>9}{'gap':>8}{'clear':>8}"
    if args.compare:
        hdr += f"{'  | was:':>8}{'scale':>7}{'ch_h':>7}{'gap':>8}{'pg':>4}{'chars':>7}"
    print(f"\n{hdr}")
    print("-" * len(hdr))

    for m in sorted(rows, key=lambda r: (r["gap"] if r["gap"] is not None else 1e9)):
        line = (f"{m['vcode']:<11}{m['pages']:>3}"
                f"{(('%.3f' % m['scale']) if m['scale'] else '-'):>7}"
                f"{fmt(m['body_pt'], 2):>6}{fmt(m['chart_w'], 0):>7}"
                f"{fmt(m['text_bottom']):>10}"
                f"{fmt(m['chart_top']):>11}{fmt(m['chart_h']):>9}"
                f"{fmt(m['gap']):>8}{fmt(m['clear']):>8}")
        if args.compare:
            c = m.get("cmp")
            if c is None:
                line += f"{'  | (no before)':>8}"
            else:
                dchars = m["nchars"] - c["nchars"]
                line += (f"  | {'':>4}{(('%.3f' % c['scale']) if c['scale'] else '-'):>7}"
                         f"{fmt(c['chart_h'], 0):>7}{fmt(c['gap']):>8}"
                         f"{c['pages']:>4}{dchars:>+7}")
        print(line)

    # ---- assertions ------------------------------------------------------
    print()
    n = len(rows)
    one_page = [m for m in rows if m["pages"] == 1]
    has_chart = [m for m in rows if m["chart_h"] is not None]
    overlap = [m for m in rows if m["gap"] is not None and m["gap"] < 0]
    # The two checks that the 800x600 harness could not see. A page that is
    # scaled is a page whose body text is below the 8pt floor, and the cause is
    # always a canvas wider than the printable column.
    unscaled = [m for m in rows if m["scale"] and m["scale"] > 0.995]
    fits = [m for m in rows
            if m["chart_w"] is not None and m["chart_w"] <= m["column_w"] + 1]
    print(f"  page at 100% scale    {len(unscaled):>3} / {n}")
    print(f"  canvas fits column    {len(fits):>3} / {n}"
          f"   (column {rows[0]['column_w']:.0f}pt)" if rows else "")
    print(f"  exactly one page      {len(one_page):>3} / {n}")
    print(f"  chart present         {len(has_chart):>3} / {n}")
    ov_desc = ", ".join("%s by %.0fpt" % (m["vcode"], -m["gap"]) for m in overlap)
    print(f"  text overlaps box     {len(overlap):>3} / {n}"
          + (f" — {ov_desc}" if overlap else ""))
    allow_any = args.allow_rule_hits.strip().lower() == "any"
    allowed = set() if allow_any else set(
        x for x in args.allow_rule_hits.split(",") if x)
    hit = [m for m in rows if m["clear"] is not None and m["clear"] < 0]
    unexpected = [] if allow_any else [
        m for m in hit if m["vcode"] not in allowed]
    hit_desc = ", ".join("%s by %.1fpt" % (m["vcode"], -m["clear"]) for m in hit)
    print(f"  text hits chart rule  {len(hit):>3} / {n}"
          + (f" — {hit_desc}" if hit else "")
          + (("   [all accepted by design]" if allow_any
               else f"   [{len(allowed & {m['vcode'] for m in hit})} accepted]")
             if hit else ""))
    if unexpected:
        print("    UNEXPECTED: "
              + ", ".join(m["vcode"] for m in unexpected))
    clears = [(m["clear"], m["vcode"]) for m in rows
              if m["clear"] is not None and m["vcode"] not in allowed]
    if clears:
        c, v = min(clears)
        print(f"  tightest clearance    {c:>6.1f}pt  ({v})")
    if has_chart:
        hs = sorted(m["chart_h"] for m in has_chart)
        print(f"  chart height (pt)     min {hs[0]:.1f}  max {hs[-1]:.1f}")
    if args.compare:
        both = [m for m in rows if m.get("cmp")]
        lost = [m for m in both if not _covers(m["norm"], m["cmp"]["norm"])]
        regressed = [m for m in both
                     if m["pages"] > m["cmp"]["pages"]]
        print(f"  text lost vs before   {len(lost):>3} / {len(both)}"
              + (f" — {', '.join(m['vcode'] for m in lost)}" if lost else ""))
        print(f"  page count regressed  {len(regressed):>3} / {len(both)}"
              + (f" — {', '.join(m['vcode'] for m in regressed)}"
                 if regressed else ""))

    # Always required, under either contract.
    bad = (len(one_page) != n) or (len(has_chart) != n) or bool(missing)
    if args.layout == "unclipped":
        bad = bad or (len(unscaled) != n) or (len(fits) != n) or bool(unexpected)
    else:
        print("\n  layout=original: page scale and canvas width are REPORTED,"
              "\n  not asserted — both are intended here. Run"
              "\n  scripts/onepager_overflow_report.py for the text-loss check.")
    if args.compare:
        both = [m for m in rows if m.get("cmp")]
        bad = bad or any(not _covers(m["norm"], m["cmp"]["norm"]) for m in both)
    print("\n" + ("FAIL" if bad else "PASS"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
