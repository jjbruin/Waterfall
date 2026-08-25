"""Inspect the printed Portfolio Snapshot PDF: pagination, headers, content.

Pairs with ``scripts/snapshot_print_check.mjs``, which produces the PDF by
driving the real browser print path. This asserts what the document actually
contains, which is the part that cannot be read off the CSS:

  * exactly 4 pages, and no trailing blank
  * the client line and "Balances as of" subtitle on EVERY page
  * the big centred title on pages 1-2 only, as published
  * page 1 carries the charts (vector marks, not a rasterised canvas)
  * pages 2-4 carry their subtab's own signature content
  * no table is clipped — the rightmost column header must be present

Also renders each page to PNG so the layout can be looked at.

Usage
    python scripts/snapshot_print_inspect.py vue_app/.chartcheck/snapshot_TGAM_2026-Q1.pdf
"""
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"no such file: {path}", file=sys.stderr)
        return 2

    import pdfplumber
    import pypdfium2 as pdfium

    checks = []

    def ck(label, ok, note=""):
        checks.append((label, ok, note))

    out_dir = os.path.join(os.path.dirname(path), "pages")
    os.makedirs(out_dir, exist_ok=True)

    with pdfplumber.open(path) as pdf:
        pages = [(p.extract_text() or "") for p in pdf.pages]
        geom = [(len(p.chars), len(p.rects), len(p.curves), len(p.lines),
                 len(p.images)) for p in pdf.pages]

    print(f"{len(pages)} page(s) in {os.path.basename(path)}\n")
    ck("exactly 4 pages", len(pages) == 4, f"got {len(pages)}")

    CLIENT = "TIAA"
    SUB = "Current Portfolio Update (Balances as of"
    TITLE = "PORTFOLIO SNAPSHOT"

    #: page index -> (label, strings that must appear)
    EXPECT = {
        0: ("page 1 — summary + charts",
            ["Portfolio Exposure", "Asset Allocation", "Deal Type Allocation",
             "Currently Funded", "Total Commitment", "Multifamily", "Retail",
             "Self-Storage", "Office", "Value-Add", "Income",
             "New Construction"]),
        1: ("page 2 — Financial",
            ["Property", "Debt", "Total Pref", "Ptr. Equity", "Total Cap",
             "TIAA Investment", "% of Pref", "Invested", "Un-funded",
             "Total Commitment", "ITD Distributions", "Net ROE",
             "Total Individual Investments", "Total PSC TGA 2022 LLC",
             "Total PSC TGA 2025 LLC", "Portfolio Totals",
             "Excluding Development Deals", "City West", "Footnotes"]),
        2: ("page 3 — Operating",
            ["Econ Occ", "NOI At Close", "NOI U/W YE", "NOI Projected YE",
             "Expected Growth", "Actual Growth", "n/a",
             # fund labels + subtotals, PDF page 3
             "Total Individual Investments", "Total PSC TGA 2022 LLC",
             "Total PSC TGA 2023 LLC", "Total PSC TGA 2024 LLC",
             "Total PSC TGA 2025 LLC", "Portfolio Totals"]),
        3: ("page 4 — Loan",
            ["Debt", "LTV", "DSCR", "Debt Yield", "Rate", "Maturity",
             # fund labels + subtotals, PDF page 4
             "Total Individual Investments", "Total PSC TGA 2022 LLC",
             "Total PSC TGA 2023 LLC", "Total PSC TGA 2024 LLC",
             "Total PSC TGA 2025 LLC", "Portfolio Totals"]),
    }

    for i, text in enumerate(pages):
        chars, rects, curves, lines, images = geom[i]
        label = EXPECT.get(i, (f"page {i + 1}", []))[0]
        print("=" * 92)
        print(f"{label}   chars={chars} rects={rects} curves={curves} "
              f"lines={lines} images={images}")
        print("=" * 92)

        ck(f"page {i + 1} is not blank", chars > 50, f"{chars} chars")
        ck(f"page {i + 1} carries the client line", CLIENT in text)
        ck(f"page {i + 1} carries the as-of subtitle", SUB in text)

        want_title = i in (0, 1)
        has_title = TITLE in text
        ck(f"page {i + 1} {'has' if want_title else 'has no'} the centred title",
           has_title == want_title)

        # Case-insensitive: table headers are uppercased by CSS, so the DOM says
        # "Econ Occ" and the paper says "ECON OCC".
        low = text.lower()
        missing = [s for s in EXPECT.get(i, (None, []))[1]
                   if s.lower() not in low]
        if missing:
            print(f"  MISSING: {missing}")
        ck(f"page {i + 1} content present", not missing,
           f"missing {missing[:6]}" if missing else "")

        first = [ln for ln in text.split("\n") if ln.strip()][:4]
        for ln in first:
            print(f"    | {ln[:110]}")

    # Page 1's charts must be VECTOR marks, not one rasterised canvas: the
    # print route asks for the SVG renderer precisely so the type inside the
    # bars stays sharp on paper.
    c1 = geom[0]
    ck("page 1 charts are vector, not a raster image",
       c1[4] == 0 and (c1[1] + c1[2] + c1[3]) > 20,
       f"images={c1[4]} shapes={c1[1] + c1[2] + c1[3]}")

    # Render for eyeballing.
    doc = pdfium.PdfDocument(path)
    print("\n" + "=" * 92)
    for i in range(len(doc)):
        p = os.path.join(out_dir, f"page{i + 1}.png")
        doc[i].render(scale=2.0).to_pil().save(p)
        print(f"  rendered {p}")

    print("\n" + "=" * 92)
    failed = [(l, n) for l, ok, n in checks if not ok]
    print(f"{len(checks) - len(failed)}/{len(checks)} checks passed")
    for label, note in failed:
        print(f"    [FAIL] {label}{'  — ' + note if note else ''}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
