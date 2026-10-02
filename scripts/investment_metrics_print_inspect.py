"""Guardrail: the printed Investment Metrics sheet matches the reference's geometry.

Run `scripts/investment_metrics_print_check.mjs` first to produce the PDF; this
measures it.

WHAT IT CHECKS AND WHY EACH ONE EXISTS
--------------------------------------
Every check below failed at least once while the sheet was being built, which
is the only reason any of them is here.

* **Two pages, letter LANDSCAPE.** The first render came out as one portrait
  page — the router guard had redirected to the sign-in form and the harness
  printed that. A PDF was produced, so nothing looked wrong.
* **Horizontal fit.** `white-space: nowrap` makes `width: 100%` a floor rather
  than a ceiling, so one extra column or one longer heading pushes the table
  off the sheet with no error anywhere (the failure `check_margins` was written
  for on the Portfolio Snapshot). Twenty-one columns is a lot of floor.
* **Vertical fit, and NO THIRD PAGE.** Current on sheet one, Sold on sheet two.
  A trailing blank sheet is the classic `page-break-after: always` mistake.
* **Twenty separate column rules, not one long line.** The reference stops each
  rule 1.08pt short of the next, and that gap is what makes the columns read as
  columns. Drawn as cell borders they butt together into a single stroke across
  the table; the first render did exactly that.
* **Continuous vertical rules.** Drawn as cell borders they came out as a run
  of dashes — present on white rows, painted over on shaded ones.
* **The shading is there, and on the right rows.** The reference bands
  alternate rows #D9D9D9 starting with the SECOND deal. A phase error is
  invisible unless it is measured.
* **Garamond at the reference's sizes**, including the fact that the footnotes
  (4.68pt) are LARGER than the table (3.6pt), which reads as a mistake and is
  not one.

KNOWN AND ACCEPTED: every rule prints thicker than the reference's. Chrome
cannot draw a line thinner than one CSS pixel (0.75pt) when printing, and the
reference's rules are 0.12pt and 0.24pt. There is no CSS for a hairline, so the
check asserts the rules EXIST and are positioned, not that they are 0.12pt.
"""
import sys
from collections import defaultdict

try:
    import pdfplumber
except ImportError:                                   # pragma: no cover
    print("pdfplumber is required: pip install pdfplumber")
    sys.exit(2)

PASS = FAIL = SKIP = 0
FAILURES = []

# Measured from `PSC Investment Metrics 06.30.26 OW.pdf`, in points.
PAGE_W, PAGE_H = 792.0, 612.0
FRAME_LEFT, FRAME_RIGHT = 18.0, 734.16
COL_RULE_COUNT = {0: 20, 1: 21}       # per page
FIRST_COL_RULE_X = 86.16
LAST_COL_RULE_X = {0: 707.28, 1: 730.44}
BAND_GREY = 0.851                      # #D9D9D9
BODY_PT, NOTE_PT = 3.6, 4.68
# Chrome snaps to 0.75pt (one CSS pixel), and small vertical drift accumulates
# over fifty 4.68pt rows, so positions are checked to a visible tolerance
# rather than to the point.
TOL_X, TOL_Y = 2.5, 4.0


def chk(label, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [ok]   {label}")
    else:
        FAIL += 1
        FAILURES.append(label)
        print(f"  [FAIL] {label}" + (f" — {detail}" if detail else ""))


def skip(label, why):
    global SKIP
    SKIP += 1
    print(f"  [skip] {label} — {why}")


def grey(nsc):
    """Mean channel value, or None. NEVER write `grey(x) or 1`: black is 0.0,
    which is falsy, and every rule would silently vanish from the results."""
    if nsc is None:
        return None
    if isinstance(nsc, (int, float)):
        return float(nsc)
    if isinstance(nsc, (list, tuple)) and nsc:
        return float(sum(nsc) / len(nsc))
    return None


def dark_rects(pg):
    out = []
    for r in pg.rects:
        g = grey(r.get("non_stroking_color"))
        if g is not None and g < 0.3:
            out.append(r)
    return out


def horizontal_runs(pg):
    """Thin dark horizontals, merged where they touch, grouped by y."""
    by_y = defaultdict(list)
    for r in dark_rects(pg):
        if r["height"] < 1.2 and r["width"] > 3:
            by_y[round(r["top"], 1)].append((r["x0"], r["x1"]))
    runs = {}
    for y, segs in by_y.items():
        segs.sort()
        merged = []
        for a, b in segs:
            if merged and a <= merged[-1][1] + 0.5:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        runs[y] = merged
    return dict(sorted(runs.items()))


def verticals(pg):
    return sorted(
        (r for r in dark_rects(pg) if r["width"] < 1.5 and r["height"] > 20),
        key=lambda r: r["x0"],
    )


def bands(pg):
    out = defaultdict(list)
    for r in pg.rects:
        g = grey(r.get("non_stroking_color"))
        if g is None or not (0.6 < g < 0.95):
            continue
        if not (2.0 < r["height"] < 8.0):
            continue
        out[round(r["top"], 1)].append(r)
    return dict(sorted(out.items()))


def rule_is_unbroken(path, page_no, rule, scale=4.0):
    """Is the rule actually VISIBLE down its whole length, once rendered?

    THE OBJECT MODEL IS NOT ENOUGH HERE, and that is the point of this check.
    A vertical rule drawn as one tall rect reads as one tall rect in the PDF's
    object list no matter what is drawn on top of it — so the shaded rows can
    paint straight over it and every structural check still passes while the
    printed page shows a column of dashes. It happened exactly that way: the
    rects measured continuous and the rasterized sheet showed the rule missing
    on every grey row.

    So this rasterizes and samples the pixels. Returns (fraction_dark, sampled)
    or None when no rasterizer is installed.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError:                               # pragma: no cover
        return None
    doc = pdfium.PdfDocument(path)
    img = doc[page_no].render(scale=scale).to_pil().convert("L")
    x = int(round((rule["x0"] + rule["x1"]) / 2 * scale))
    y0 = int(round(rule["top"] * scale)) + 2
    y1 = int(round(rule["bottom"] * scale)) - 2
    w, h = img.size
    if not (0 <= x < w) or y1 <= y0:
        return None
    dark = 0
    sampled = 0
    for y in range(y0, min(y1, h)):
        sampled += 1
        # The rule may land between two device pixels; take the darkest of the
        # three around its centre rather than insisting on one column.
        v = min(img.getpixel((xx, y)) for xx in (x - 1, x, x + 1) if 0 <= xx < w)
        if v < 160:
            dark += 1
    return (dark / sampled if sampled else 0.0), sampled


def text_rows(pg, small=True):
    rows = defaultdict(list)
    for w in pg.extract_words(extra_attrs=["fontname", "size"]):
        if small and w["size"] > 4.0:
            continue
        if not small and w["size"] <= 4.0:
            continue
        rows[round(w["top"], 1)].append(w)
    return dict(sorted(rows.items()))


def main(path):
    print(f"Investment Metrics — printed sheet, measured against the reference\n{path}\n")
    with pdfplumber.open(path) as pdf:
        pages = pdf.pages

        print("1. The sheet")
        chk("exactly two pages (Current, then Sold) — no trailing blank",
            len(pages) == 2, f"got {len(pages)}")
        if len(pages) != 2:
            return report()
        for i, pg in enumerate(pages):
            chk(f"page {i + 1} is letter LANDSCAPE ({PAGE_W:.0f}x{PAGE_H:.0f}pt)",
                abs(pg.width - PAGE_W) < 1 and abs(pg.height - PAGE_H) < 1,
                f"got {pg.width:.0f}x{pg.height:.0f}")

        for i, pg in enumerate(pages):
            name = ("Current", "Sold")[i]
            print(f"\n2.{i + 1} {name} sheet — fit")
            xs = [c["x0"] for c in pg.chars] + [c["x1"] for c in pg.chars]
            ys = [c["top"] for c in pg.chars] + [c["bottom"] for c in pg.chars]
            if not xs:
                chk(f"{name}: page carries text", False, "no characters at all")
                continue
            # HORIZONTAL FIT. The one that has no natural error.
            chk(f"{name}: nothing is drawn past the right margin",
                max(xs) <= FRAME_RIGHT + 1.0,
                f"rightmost glyph at {max(xs):.2f}pt, margin {FRAME_RIGHT}")
            chk(f"{name}: nothing is drawn left of the frame",
                min(xs) >= FRAME_LEFT - 1.0, f"leftmost glyph at {min(xs):.2f}pt")
            chk(f"{name}: everything fits the sheet vertically",
                max(ys) <= PAGE_H - 4, f"lowest glyph at {max(ys):.2f}pt")

            runs = horizontal_runs(pg)
            print(f"\n2.{i + 1}b {name} sheet — rules")
            # The column rules are the widest group of short runs.
            best_y, best = None, []
            for y, m in runs.items():
                if len(m) > len(best):
                    best_y, best = y, m
            chk(f"{name}: {COL_RULE_COUNT[i]} SEPARATE column rules, not one line",
                len(best) == COL_RULE_COUNT[i],
                f"got {len(best)} run(s) at y={best_y}")
            if best:
                chk(f"{name}: the column rules start at the Asset Class boundary",
                    abs(best[0][0] - FIRST_COL_RULE_X) < TOL_X,
                    f"first rule starts {best[0][0]:.2f}, reference "
                    f"{FIRST_COL_RULE_X}")
                chk(f"{name}: the column rules end at the last column",
                    abs(best[-1][1] - LAST_COL_RULE_X[i]) < TOL_X,
                    f"last rule ends {best[-1][1]:.2f}, reference "
                    f"{LAST_COL_RULE_X[i]}")
            grouped = [y for y, m in runs.items() if len(m) == 3]
            chk(f"{name}: three capitalization pair rules",
                len(grouped) >= 1, "no row of exactly three rules found")

            v = verticals(pg)
            chk(f"{name}: four continuous vertical rules",
                len(v) == 4, f"got {len(v)} — fragments mean a rule with holes")
            if v:
                shortest = min(r["height"] for r in v)
                chk(f"{name}: every vertical rule spans the table",
                    shortest > 100,
                    f"shortest is {shortest:.1f}pt — it is drawn per row")
                # And the rule must SURVIVE the shaded rows, which only a
                # rasterized sample can tell.
                worst, worst_at = 1.0, None
                missing_rasterizer = False
                for r in v:
                    got = rule_is_unbroken(path, i, r)
                    if got is None:
                        missing_rasterizer = True
                        break
                    frac, _ = got
                    if frac < worst:
                        worst, worst_at = frac, r["x0"]
                if missing_rasterizer:
                    skip(f"{name}: vertical rules are not painted over",
                         "pypdfium2 not installed")
                else:
                    # The detail is built defensively: `worst_at` is None when
                    # nothing failed, and an f-string that formats it eagerly
                    # raises TypeError on the PASSING path — which kills the
                    # run and takes every later check with it.
                    detail = ("" if worst_at is None else
                              f"rule at x={worst_at:.1f} is only {worst:.0%} "
                              "visible — the shaded rows are covering it")
                    chk(f"{name}: no vertical rule is painted over by the shading",
                        worst > 0.95, detail)

            print(f"\n2.{i + 1}c {name} sheet — shading and type")
            b = bands(pg)
            chk(f"{name}: alternate rows are shaded", len(b) > 5,
                f"{len(b)} shaded rows")
            if b:
                first = next(iter(b.values()))
                g = grey(first[0].get("non_stroking_color"))
                chk(f"{name}: the shade is the reference's #D9D9D9",
                    g is not None and abs(g - BAND_GREY) < 0.02,
                    f"grey {g}")
                # The shading must start on the SECOND deal, not the first.
                data = [t for t in text_rows(pg) if t > min(b)]
                chk(f"{name}: shading starts below the first deal row",
                    min(b) > min(text_rows(pg)), "phase looks inverted")
                _ = data

            fonts = {(w["fontname"].split("+")[-1], round(w["size"], 2))
                     for row in text_rows(pg).values() for w in row}
            fams = {f for f, _ in fonts}
            chk(f"{name}: the table is set in Garamond",
                any("Garamond" in f for f in fams), f"families {sorted(fams)}")
            chk(f"{name}: the table is set at {BODY_PT}pt",
                all(abs(s - BODY_PT) < 0.2 for _, s in fonts),
                f"sizes {sorted({s for _, s in fonts})}")
            notes = {round(w["size"], 2)
                     for row in text_rows(pg, small=False).values() for w in row}
            chk(f"{name}: the footnotes are {NOTE_PT}pt — LARGER than the table",
                notes and all(abs(s - NOTE_PT) < 0.2 for s in notes),
                f"sizes {sorted(notes)}")

        print("\n3. Content the reference requires")
        p0 = pages[0].extract_text() or ""
        p1 = pages[1].extract_text() or ""
        chk("page 1 is the Current portfolio",
            "PSC Investment Summary - Current Portfolio" in p0)
        chk("page 2 is the Sold portfolio",
            "PSC Investment Summary - Sold Portfolio" in p1)
        chk("the units note prints on both sheets",
            "($ in USD millions)" in p0 and "($ in USD millions)" in p1)
        chk("the as-of date prints on the Current sheet only",
            any(t.endswith(("-26", "-25", "-27")) or "-Jun-" in t
                for t in p0.split("\n")[:3]))
        chk("both sheets carry the Total / Average row",
            "Total / Average" in p0 and "Total / Average" in p1)
        chk("the Grand Total follows the Sold table", "Grand Total" in p1)
        chk("the grouped capitalization heading prints",
            "Underwritten Capitalization at Stabilization" in p0)
        chk("the disclaimer prints under both tables",
            "Projected and realized returns are" in p0
            and "Projected and realized returns are" in p1)
        chk("the Sold sheet carries the Realized Final IRR column",
            "Realized" in p1 and "Final" in p1)
    return report()


def report():
    print(f"\n{PASS} passed, {FAIL} failed, {SKIP} skipped")
    if FAILURES:
        print("failed:")
        for f in FAILURES:
            print(f"  - {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("usage: investment_metrics_print_inspect.py <rendered.pdf>")
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
