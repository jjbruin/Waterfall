"""Guardrail: comparing a sent PDF against what the app computes. NO FREEZING.

These rules used to be enforced through the "Freeze as sent (from PDFs)" action.
That action is gone — the freeze is for 26Q3 onward and always freezes from LIVE
data — but the comparison it was built on is worth keeping: reading a sent PDF
and asking where the app disagrees with it is a REPORT, and a useful one.

So the rules move here, against `portfolio_snapshot_pdf_compare`, and are
asserted WITHOUT a freeze anywhere in the file. That is the point: if this
module could still write into the frozen store, the removal would not be real.

What it pins:

  1. AN OVERLAY OVERWRITES ONLY THE CELLS IT NAMES, keeps the computed value
     beside each one, and counts what it applied. Asserted in both directions —
     a cell the overlay does not name must come through untouched, or "the PDF
     wins" is indistinguishable from "the PDF replaces everything".
  2. A PRINTED-UNITS CELL keeps the PDF's own text and leaves the number alone.
  3. CELLS THAT WOULD NOT LAND are predicted, not discovered afterwards.
  4. THE MODULE CANNOT FREEZE. No import of the freeze engine at module level,
     and nothing in it writes to the frozen store.

Usage
    .venv/Scripts/python.exe scripts/pdf_overlay_compare_check.py
"""
from __future__ import annotations

import copy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

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


import flask_app.services.portfolio_snapshot_pdf_compare as C      # noqa: E402

print("A. an overlay overwrites only the cells it names")
payload = {"subtabs": {"financial": {"groups": {"G": {"deals": [
    {"vcode": "D1", "name": "Deal One", "total_pref": 9_100_000}]}}}}}
one_pagers = {"D1": {"vcode": "D1",
                     "cap_stack": {"debt": 12.0, "untouched_cap": 333.0},
                     "property_performance": {"noi": 5.0}}}
overlay = {"D1": {"cap_stack.debt": {"published": 9.0, "page": 4},
                  "property_performance.noi": {"published": 7.7, "page": 4}}}
pay = copy.deepcopy(payload)
applied = C._apply_overlay(pay, one_pagers, overlay)
chk("it reports how many cells it applied", applied == 2, str(applied))
chk("a named cell takes the PDF's value",
    one_pagers["D1"]["cap_stack"]["debt"] == 9.0,
    str(one_pagers["D1"]["cap_stack"]["debt"]))
chk("...and the second one too",
    one_pagers["D1"]["property_performance"]["noi"] == 7.7)
# THE PAIRED DIRECTION. Without this, an overlay that flattened everything to
# the PDF would pass every check above.
chk("a cell the overlay does NOT name is untouched",
    one_pagers["D1"]["cap_stack"]["untouched_cap"] == 333.0,
    str(one_pagers["D1"]["cap_stack"]["untouched_cap"]))
# `_apply_overlay` records what it overwrote into the PAYLOAD, so the drift
# between the PDF and the app stays measurable after the fact.
_rec = pay.get("published_overrides") or []
chk("the computed value is kept beside the published one",
    any(str(r).find("12.0") >= 0 and "computed_at_freeze" in str(r)
        for r in (_rec if isinstance(_rec, list) else [_rec])),
    str(_rec)[:200])

print("\nB. a cell naming a field that does not exist is REPORTED, not dropped")
miss = C.dry_run_unapplied(
    {"payload": copy.deepcopy(payload),
     "one_pagers": {"D1": {"cap_stack": {"debt": 1.0}}}},
    {"D1": {"cap_stack.no_such_field": {"published": 1.0, "page": 8}}})
chk("an unplaceable cell is predicted before anything is written",
    isinstance(miss, list) and len(miss) >= 1, str(miss)[:120])
landed = C.dry_run_unapplied(
    {"payload": copy.deepcopy(payload),
     "one_pagers": {"D1": {"cap_stack": {"debt": 1.0}}}},
    {"D1": {"cap_stack.debt": {"published": 9.0, "page": 4}}})
chk("...and a cell that WOULD land is not reported", landed == [], str(landed))

print("\nC. the live comparison describes what would change")
cmp_out = C.compare_overlay_to_live(
    {"D1": {"cap_stack": {"debt": 12.0}}},
    {"D1": {"cap_stack.debt": {"published": 9.0, "page": 4}}})
chk("it counts the cells compared", (cmp_out or {}).get("cells_total", 0) >= 1,
    str(cmp_out)[:140])
chk("...and how many differ from live",
    (cmp_out or {}).get("differs_total", 0) >= 1, str(cmp_out)[:140])
same = C.compare_overlay_to_live(
    {"D1": {"cap_stack": {"debt": 9.0}}},
    {"D1": {"cap_stack.debt": {"published": 9.0, "page": 4}}})
chk("a matching cell is NOT counted as a difference",
    (same or {}).get("differs_total", 1) == 0, str(same)[:140])

print("\nD. the comparison module cannot freeze")
src = open(C.__file__, encoding="utf-8").read()
chk("no module-level import of the freeze engine",
    not [l for l in src.splitlines()
         if l.startswith(("from flask_app", "import flask_app"))])
for bad in ("freeze_part", "_write_frozen", "INSERT INTO", "UPDATE ", "DELETE FROM"):
    chk(f"it never {bad.strip().lower()}s", bad not in src, f"found {bad!r}")
chk("the freeze engine no longer carries the overlay code",
    "_apply_overlay" not in open(
        os.path.join(ROOT, "flask_app", "services",
                     "portfolio_snapshot_freeze.py"), encoding="utf-8").read())
import inspect                                                      # noqa: E402
from flask_app.services import portfolio_snapshot_freeze as F       # noqa: E402
chk("...and freeze_part takes no overlay argument",
    "overlay" not in inspect.signature(F.freeze_part).parameters)

print("\nE. the overlay BUILDER is still here, for comparison")
builder = os.path.join(ROOT, "scripts", "build_26q2_overlay.py")
chk("scripts/build_26q2_overlay.py is kept", os.path.exists(builder))
if os.path.exists(builder):
    bsrc = open(builder, encoding="utf-8").read()
    chk("...and it does not post to any freeze endpoint",
        "freeze-overlay" not in bsrc and "freeze-all" not in bsrc)

print(f"\n{'=' * 60}\n{PASS} passed, {FAIL} failed, {SKIP} skipped")
sys.exit(1 if FAIL else 0)
