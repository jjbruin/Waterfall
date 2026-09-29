"""Guardrail: no screen or population file pins a quarter as a literal.

WHY THIS EXISTS. Two views seeded their quarter box with the literal
'2026-Q2', and two print-sweep scripts carried the same literal as a fallback.
A pinned quarter is right for exactly one quarter and silently wrong from the
day the next one ends — Review Tracking opens on a finished quarter, the One
Pager batch builds a stale document, and the print sweep keeps rendering the
old period while the before/after comparison still looks clean. Nothing on
screen or in the output says so, which is what makes it worth a guardrail
rather than a code review.

IT ASSERTS BOTH DIRECTIONS. "No literal appears" is satisfied by deleting the
quarter handling altogether, so each file must ALSO still resolve a quarter —
the views from the server, the scripts from `--quarter`.

It reads source, so it SKIPS with a reason where the sources are absent (the
container image ships no `vue_app/`), rather than failing for the wrong reason.

Usage
    .venv/Scripts/python.exe scripts/quarter_hardcode_check.py
"""
from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

PASS = FAIL = SKIP = 0


def chk(label, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  OK   {label}")
    else:
        FAIL += 1
        print(f"  FAIL {label}" + (f"  -> {detail}" if detail else ""))


def skip(label, why):
    global SKIP
    SKIP += 1
    print(f"  SKIP {label} — {why}")


#: A four-digit year and a quarter, anywhere. Deliberately NOT anchored and
#: deliberately not limited to quoted strings: the point is that the token does
#: not appear at all, in code, in a comment, in a placeholder or in data.
LITERAL = re.compile(r"\b(19|20)\d{2}-Q[1-4]\b")

#: Files that must carry no literal quarter, and what each must still do
#: instead. The second element is asserted separately so that deleting the
#: feature cannot pass this check.
#
#: For the views this is the CALL, not the name: the import alone satisfied
#: `"defaultQuarter" in src` even with the call deleted, so the paired direction
#: was vacuous and an injection that removed the resolution passed 24/24.
GUARDED = [
    ("vue_app/src/views/OnePagerView.vue",         "defaultQuarter()"),
    ("vue_app/src/views/ReviewTrackingView.vue",   "defaultQuarter()"),
    ("scripts/onepager_print_population.txt",      None),
    ("scripts/onepager_print_sweep.mjs",           "--quarter"),
    ("scripts/onepager_print_geometry.py",         "--quarter"),
    ("scripts/onepager_char_counter_check.py",     "--quarter"),
    ("scripts/onepager_overflow_report.py",        "--quarter"),
]

print("A. no literal quarter in the guarded views, scripts and data files")
for rel, must_have in GUARDED:
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        skip(rel, "not present in this tree")
        continue
    src = open(path, encoding="utf-8", errors="replace").read()
    hits = sorted({m.group(0) for m in LITERAL.finditer(src)})
    chk(f"{rel} pins no quarter", not hits, f"found {hits}")
    if must_have:
        # THE PAIRED DIRECTION: the literal being absent must not mean the
        # quarter handling was simply deleted.
        chk(f"...and still resolves one ({must_have})", must_have in src,
            f"{must_have!r} not found in {rel}")

print("\nB. the server is what resolves a screen's quarter")
helper = os.path.join(ROOT, "vue_app", "src", "api", "quarters.ts")
if not os.path.exists(helper):
    skip("the quarters helper", "vue_app/ is not in this tree")
else:
    src = open(helper, encoding="utf-8").read()
    chk("the helper pins no quarter either",
        not LITERAL.search(src),
        str(sorted({m.group(0) for m in LITERAL.finditer(src)})))
    chk("it asks the server rather than computing a quarter locally",
        "/api/portfolio-snapshot/quarters" in src)
    chk("it returns '' when the server cannot say, never a guess",
        "default: ''" in src or "default || ''" in src or "|| ''" in src)
    chk("a failed lookup is not cached as the answer", "cached = null" in src)

print("\nC. the population file lists deals, not periods")
pop = os.path.join(ROOT, "scripts", "onepager_print_population.txt")
if not os.path.exists(pop):
    skip("the population file", "not present in this tree")
else:
    lines = [l.strip() for l in open(pop, encoding="utf-8")
             if l.strip() and not l.strip().startswith("#")]
    chk("it still names the population", len(lines) > 0, str(len(lines)))
    chk("every entry is a bare vcode",
        all(len(l.split()) == 1 for l in lines),
        str([l for l in lines if len(l.split()) != 1][:3]))

print("\nD. the shared reader refuses rather than guessing")
sys.path.insert(0, HERE)
try:
    from onepager_population import read_population
except Exception as exc:                                    # noqa: BLE001
    skip("the shared reader", f"could not import: {exc}")
else:
    import io
    quiet = io.StringIO()
    got = read_population(pop, None, stream=quiet)
    chk("a bare population with no --quarter is REFUSED", got is None)
    chk("...and says how to fix it", "--quarter" in quiet.getvalue(),
        quiet.getvalue()[:80])
    got2 = read_population(pop, "2030-Q1", stream=quiet)
    chk("...and is read when the quarter is supplied",
        got2 is not None and len(got2) > 0)
    chk("...applying it to every row",
        got2 and all(q == "2030-Q1" for _, q in got2))
    bad = read_population(pop, "not-a-quarter", stream=quiet)
    chk("a malformed --quarter is refused", bad is None)

print(f"\n{'=' * 60}\n{PASS} passed, {FAIL} failed, {SKIP} skipped")
sys.exit(1 if FAIL else 0)
