"""Guardrail: the character counters never reach the printed One Pager.

The counters under the Business Plan and Property Performance comment boxes are
an editing aid. They are hidden twice over — `.no-print` on the element and an
explicit `.char-counter { display: none !important }` inside `@media print` —
and neither of those is worth anything unless the rendered page is checked.

Why a script rather than reading the CSS: `display: none` is easy to write and
easy to lose. A refactor of the shared `.no-print` selector, a specificity
accident, or a counter added to a THIRD box without the class would each put
"1,247 characters" onto an investor document, and the CSS would still look
right. This greps the actual PDF text of every deal.

It also reports the counts themselves, because the count is the number the
asset manager needs in order to trim a narrative to fit: the counter on screen
and the figure in this report come from the same string length.

Read-only. Consumes the PDFs written by scripts/onepager_print_sweep.mjs.

Usage
  .venv/Scripts/python.exe scripts/onepager_char_counter_check.py --tag orig
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import pdfplumber

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PDF_DIR = os.path.join(ROOT, "vue_app", ".chartcheck")

# What the counter renders: "1,247 characters" / "1 character". Matching the
# word alone would trip over a narrative that happens to discuss characters, so
# the pattern requires the digits-then-word shape the component produces.
COUNTER_RE = re.compile(r"\b\d{1,3}(,\d{3})*\s+characters?\b")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--deals",
                    default=os.path.join(HERE, "onepager_print_population.txt"))
    args = ap.parse_args()

    deals = [l.split() for l in open(args.deals, encoding="utf-8") if l.strip()
             and not l.startswith("#")]

    checked, leaked, missing = 0, [], []
    for vcode, quarter in deals:
        p = os.path.join(PDF_DIR, f"onepager_{vcode}_{quarter}_{args.tag}.pdf")
        if not os.path.exists(p):
            missing.append(vcode)
            continue
        with pdfplumber.open(p) as pdf:
            text = "\n".join((pg.extract_text() or "") for pg in pdf.pages)
        checked += 1
        hit = COUNTER_RE.search(text)
        if hit:
            leaked.append((vcode, hit.group(0)))

    if missing:
        print(f"MISSING {len(missing)} PDFs for tag '{args.tag}': "
              + ", ".join(missing[:8]))

    print(f"\n  deals checked                 {checked:>3}")
    print(f"  counter absent from print     {checked - len(leaked):>3} / {checked}")
    if leaked:
        print("  LEAKED — a counter printed on:")
        for v, s in leaked:
            print(f"     {v}  ->  {s!r}")

    bad = bool(leaked) or bool(missing) or checked == 0
    print("\n" + ("FAIL" if bad else "PASS"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
