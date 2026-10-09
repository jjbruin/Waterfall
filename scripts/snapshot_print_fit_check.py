"""Guardrail: the Portfolio Snapshot measures each page as it will print.

THE DEFECT. The printed Snapshot is one sheet per subtab, but its tables were
sized by hand for the row count they were tuned on. TIAA 26Q2 carries 37 deals
and the Financial page ran ~0.3in long, so its footnote block printed on a
fifth sheet of its own. PortfolioSnapshotPrintView now fits each page to its
sheet at print time (fitPagesToSheet).

WHY THE FIT IS FRAGILE, AND WHAT THIS PINS. Chrome fixes the page count from
the layout at `beforeprint`, while that layout is still the SCREEN one -- a zoom
applied later (from a matchMedia('print') listener) prints onto the stale count
and leaves blank sheets. So the page has to lay ITSELF out as paper to measure:
every print rule the document depends on is written under
`@container style(--paper: 1)`, which is on in print (App.vue: html) and while
`.paper` is set during the measurement. One ordinary `@media print` rule in a
subtab would apply on paper but not to the measurement, and the fit would
silently be computed against the wrong page -- which is exactly how the first
attempt over-shrank TIAA's Financial page to the floor.

  1. no `@media print {` block in the subtab components the document mounts
  2. App.vue turns `--paper` on for all printing; the view turns it on for the
     measurement and gives the measured page its 11in sheet width
  3. the view fits in `beforeprint` -- NOT from a matchMedia('print') listener
  4. the sheet height is landscape letter (8.5in) and the zoom has a floor
  5. the Summary page is `flows` and is never fitted: a long narrative runs to
     a second sheet rather than being shrunk (Charlene, Oct 8 2026)

Rendered evidence is `snapshot_print_formatting_check.py pages <pdf>`, which
fails on TIAA 26Q2 before this fix (5 pages) and passes after (4).

NON-VACUOUS: `--self-test` runs the same checks against the pre-fix sources at
PRE_FIX (git show) and requires them to FAIL.

Usage
  .venv/Scripts/python.exe scripts/snapshot_print_fit_check.py
  .venv/Scripts/python.exe scripts/snapshot_print_fit_check.py --self-test
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRE_FIX = "9590d7a"
VIEW = "vue_app/src/views/PortfolioSnapshotPrintView.vue"
APP = "vue_app/src/App.vue"
SUBTABS = [f"vue_app/src/components/snapshot/{c}.vue"
           for c in ("SnapshotFinancial", "SnapshotOperating", "SnapshotLoan")]


def _strip(css: str) -> str:
    return re.sub(r"/\*.*?\*/", "", css, flags=re.S)


def _styles(src: str) -> str:
    return _strip("".join(re.findall(r"<style[^>]*>(.*?)</style>", src, re.S)))


def check(read) -> list[str]:
    fails: list[str] = []
    for path in SUBTABS:
        n = len(re.findall(r"@media\s+print\s*\{", _styles(read(path))))
        if n:
            fails.append(f"{os.path.basename(path)}: {n} `@media print` block(s) -- "
                         "use `@container style(--paper: 1)` or the fit measures "
                         "a different page from the one that prints")
        if "@container style(--paper: 1)" not in _styles(read(path)):
            fails.append(f"{os.path.basename(path)}: no `--paper` print block")

    app = _styles(read(APP))
    m = re.search(r"@media\s+print\s*\{(.*)", app, re.S)
    if not (m and re.search(r"\bhtml\s*\{[^}]*--paper\s*:\s*1", m.group(1))):
        fails.append("App.vue does not set `html { --paper: 1 }` under @media print")

    view = read(VIEW)
    vcss = _styles(view)
    if not re.search(r"\.print-doc\.paper[^{]*\{[^}]*--paper\s*:\s*1", vcss):
        fails.append("the view never turns `--paper` on for the measurement (.print-doc.paper)")
    if not re.search(r"\.print-doc\.paper\s+\.print-page\s*\{[^}]*width\s*:\s*11in", vcss):
        fails.append("the measured page is not given its 11in sheet width")
    if "@container style(--paper: 1)" not in vcss:
        fails.append("the view's own print rules are not under the `--paper` switch")

    # code only: the view's comments explain why matchMedia is NOT used
    script = re.sub(r"//[^\n]*|/\*.*?\*/", "", view.split("</script>", 1)[0], flags=re.S)
    if not re.search(r"addEventListener\(\s*'beforeprint'\s*,\s*fitPagesToSheet", script):
        fails.append("fitPagesToSheet is not run from `beforeprint`")
    if re.search(r"matchMedia\(\s*'print'\s*\)", script):
        fails.append("a matchMedia('print') listener is used -- Chrome has already "
                     "fixed the page count by then; blank sheets follow")
    sheet = re.search(r"SHEET_PX\s*=\s*([\d.]+)\s*\*\s*96", script)
    if not sheet or float(sheet.group(1)) != 8.5:
        fails.append("SHEET_PX is not landscape letter height (8.5 * 96)")
    # the Summary is prose and may run to a second sheet; only tables are fitted
    tpl = view.split("<style", 1)[0]
    first = re.search(r'<section class="(print-page[^"]*)"', tpl)
    if not first or "flows" not in first.group(1).split():
        fails.append("the Summary page (first .print-page) is not marked `flows`")
    if not re.search(r"FITTED\s*=\s*'\.print-page:not\(\.flows\)'", script)             or "querySelectorAll<HTMLElement>(FITTED)" not in script:
        fails.append("fitPagesToSheet does not skip the `flows` (Summary) page")
    floor = re.search(r"MIN_ZOOM\s*=\s*([\d.]+)", script)
    if not floor or not (0.75 <= float(floor.group(1)) < 1):
        fails.append("MIN_ZOOM missing or outside [0.75, 1)")
    return fails


def _disk(path: str) -> str:
    return open(os.path.join(ROOT, path), encoding="utf-8").read()


def _git(path: str) -> str:
    return subprocess.run(["git", "show", f"{PRE_FIX}:{path}"], cwd=ROOT,
                          capture_output=True, text=True, encoding="utf-8",
                          check=True).stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    now = check(_disk)
    for f in now:
        print(f"FAIL  {f}")
    print("snapshot print fit: " + ("OK" if not now else f"{len(now)} failure(s)"))
    if not a.self_test:
        return 1 if now else 0
    old = check(_git)
    print(f"pre-fix {PRE_FIX}: {'FAILS as required' if old else 'PASSES -- check is vacuous'}"
          f" ({len(old)} failures)")
    for f in old[:6]:
        print(f"    {f}")
    return 0 if (not now and old) else 1


if __name__ == "__main__":
    sys.exit(main())
