"""Guardrail: the One Pager does not print a computed-looking zero where the
input for the calculation is missing.

Two cells, one principle — a `0` that means "no data" is indistinguishable from
a real zero to every reader of the page.

  1. P.E. Exposure on Value.  `pe_exposure_on_value` is only assigned when
     `current_valuation > 0`, so its DEFAULT is what a deal with no valuation
     prints. As `0.0` that rendered "0.0%" beside a Valuation cell reading an em
     dash: an exposure computed against a valuation the same row says does not
     exist. The default must be None, which `fmtPct` renders as the same dash.

  2. Economic Occ. variance.  `toFixed` keeps the sign of a value that rounds
     away to zero, so a variance too small to show at one decimal still printed
     its minus sign — "-0.0%", which reads as a shortfall against budget that
     the report is not claiming (Mount Prospect Plaza, 26Q2: actual 95.6000 vs
     budget 95.6439). Only the exact "-0.0" string may be rewritten; anything
     that rounds to a real figure must be left alone.

The Vue formatter is EXECUTED, not reimplemented, by lifting it out of the
component and running it under node — so this cannot drift from the code it
checks. Requires node on PATH; skips that half with a clear message if absent.

Usage
    python scripts/onepager_missing_vs_zero_check.py
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
VUE = ROOT / "vue_app" / "src" / "views" / "OnePagerView.vue"
PY = ROOT / "one_pager.py"

_checks = []


def chk(label, ok):
    _checks.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def read(p):
    # The working tree is CRLF on Windows and every pattern below anchors on
    # "\n"; without this the matchers silently find nothing and the guardrail
    # reports a defect that is not there.
    return p.read_text(encoding="utf-8").replace("\r\n", "\n")


print(__doc__.strip().split("\n")[0])
print()
print("1. pe_exposure_on_value defaults to None, not 0.0  (one_pager.py)")
src = read(PY)

m = re.search(r"^\s*'pe_exposure_on_value':\s*(.+?),\s*$", src, re.M)
chk("the default is declared exactly once", m is not None
    and len(re.findall(r"'pe_exposure_on_value':", src)) == 1)
if m:
    chk(f"the default is None (found: {m.group(1).strip()})",
        m.group(1).strip() == "None")

# The assignment must stay behind the valuation guard, or the default never
# survives to be printed and the fix above is inert.
assign = re.search(
    r"if cap\['current_valuation'\] > 0:\s*\n\s*"
    r"cap\['pe_exposure_on_value'\]\s*=", src)
chk("it is still only assigned when current_valuation > 0", assign is not None)

print()
print("2. occupancy variance never prints '-0.0%'  (OnePagerView.vue)")
vsrc = read(VUE)

chk("the Economic Occ. row routes through fmtOccVariance, not a raw toFixed",
    "variance: fmtOccVariance(" in vsrc
    and "ytd_actual - p.economic_occ.ytd_budget).toFixed" not in vsrc)

fn = re.search(
    r"function fmtOccVariance\(actual: [^)]*\): string \{\n(.*?)\n\}",
    vsrc, re.S)
chk("fmtOccVariance found in the component", fn is not None)

if fn and shutil.which("node"):
    body = re.sub(r":\s*number \| null \| undefined", "", fn.group(1))
    cases = [
        # actual, budget, expected  — the live case first
        [95.60000000000001, 95.6439394, "0.0%"],   # Mount Prospect Plaza 26Q2
        [95.6, 95.6, "0.0%"],                      # exactly equal
        [95.6, 95.65, "-0.1%"],                    # rounds to a real figure
        [95.6, 95.5, "0.1%"],                      # positive, untouched
        [90.0, 92.3, "-2.3%"],                     # Dorsett Ridge, untouched
        [None, 95.6, ""],                          # missing input -> empty
        [95.6, None, ""],
    ]
    js = (f"function fmtOccVariance(actual, budget) {{\n{body}\n}}\n"
          f"const cases = {json.dumps(cases)};\n"
          "console.log(JSON.stringify(cases.map("
          "c => fmtOccVariance(c[0], c[1]))));")
    got = json.loads(subprocess.run(
        ["node", "-e", js], capture_output=True, text=True,
        check=True).stdout)
    for (a, b, want), g in zip(cases, got):
        chk(f"fmtOccVariance({a}, {b}) = {g!r}, want {want!r}", g == want)
    chk("no case produced '-0.0%'", "-0.0%" not in got)
elif fn:
    print("  [SKIP] node not on PATH — formatter not executed")

print()
print("3. the print timestamp is gone  (it was ours, never the browser's)")
chk("no printTimestamp ref remains", "printTimestamp" not in vsrc)
chk("no .print-date element remains",
    'class="print-date"' not in vsrc)
chk("the document.title blanking is KEPT — that is the real browser-header "
    "suppression and must not be removed with it",
    "document.title = ' '" in vsrc)

print()
ok = sum(_checks)
print(f"{ok}/{len(_checks)} checks passed")
sys.exit(0 if ok == len(_checks) else 1)
