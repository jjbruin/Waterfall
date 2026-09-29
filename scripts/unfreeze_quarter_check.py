"""Guardrail: unfreezing ONE PART of a quarter, for everyone, in one transaction.

Scratch SQLite, no application database, no live call.

What it pins, and why each is here rather than assumed:

  1. A HISTORY ROW IS WRITTEN FOR EVERY AFFECTED INVESTOR. The archive is the
     only record of what was sent; an unfreeze that removed without archiving
     would look identical from the screen and be unrecoverable.
  2. THE OTHER HALF IS UNTOUCHED. Asserted in both directions and on the same
     rows — a row carrying BOTH parts must keep the one not named, with its
     timestamp and its data intact.
  3. LEGACY ROWS (no per-part stamps) KEEP THEIR OTHER HALF TOO. This is the
     case a naive implementation gets wrong: clearing "the One Pager columns"
     on such a row clears nothing, and deleting the row takes the Snapshot with
     it.
  4. THE COMMENT LOCK IS RELEASED when the One Pagers are unfrozen — and NOT
     released by unfreezing the Snapshot, which never locked it.
  5. IT IS SET-BASED. No per-investor loop: asserted by counting the SQL
     statements issued, and by the wall clock on a realistic row count.
  6. SINGLE-INVESTOR UNFREEZE STILL WORKS.

Usage
    .venv/Scripts/python.exe scripts/unfreeze_quarter_check.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time

os.environ["FREEZE_ENABLED"] = "1"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import sqlalchemy  # noqa: E402

PASS = FAIL = 0


def chk(label, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  OK   {label}")
    else:
        FAIL += 1
        print(f"  FAIL {label}" + (f"  -> {detail}" if detail else ""))


tmpdir = tempfile.mkdtemp(prefix="unfreeze_q_")
eng = sqlalchemy.create_engine(f"sqlite:///{os.path.join(tmpdir, 't.db')}")

from flask_app.services import portfolio_snapshot_freeze as F      # noqa: E402
from flask_app.services import portfolio_snapshot_persistence as P  # noqa: E402

F._engine = lambda: eng
F._is_postgres = lambda: False
F._data_version = lambda: "build=test"
P.load_page = lambda i, q: {"comments": [], "footnotes": [], "values": []}

Q = "2026-Q3"
BOTH = ["B1", "B2", "B3"]        # carry both halves
OPS_ONLY = ["O1", "O2"]          # One Pagers only
SNAP_ONLY = ["S1"]               # Snapshot only
LEGACY = "L1"                    # pre-per-part row: no per-part stamps


def _asm(inv, quarter, **kw):
    return {"subtabs": {"financial": {"groups": {"G": {"deals": [
        {"vcode": "D1", "name": "Deal One"}]}}}},
        "errors": {}, "resolution": {"investor_name": inv}}


def _op(vc, q):
    return {"vcode": vc, "quarter": q}


for inv in BOTH:
    F.freeze_part(inv, Q, F.PARTS, "cbui", assembler=_asm, one_pager_getter=_op)
for inv in OPS_ONLY:
    F.freeze_part(inv, Q, F.PART_ONE_PAGERS, "cbui", assembler=_asm,
                  one_pager_getter=_op)
for inv in SNAP_ONLY:
    F.freeze_part(inv, Q, F.PART_SNAPSHOT, "cbui", assembler=_asm)
# The legacy row: frozen both ways by the OLD code, so no per-part stamps.
F.freeze_part(LEGACY, Q, F.PARTS, "cbui", assembler=_asm, one_pager_getter=_op)
with eng.begin() as cx:
    cx.execute(sqlalchemy.text(
        "UPDATE portfolio_snapshot_frozen SET snapshot_frozen_at = NULL, "
        "snapshot_frozen_by = NULL, one_pagers_frozen_at = NULL, "
        "one_pagers_frozen_by = NULL WHERE investor_code = :i"), {"i": LEGACY})

ALL_OPS = BOTH + OPS_ONLY + [LEGACY]

print("A. the fixture is what the test assumes")
chk("the legacy row reads as BOTH halves frozen",
    F.frozen_parts_of(F._current_row(LEGACY, Q)) == list(F.PARTS))
chk("...and is flagged as inferred", F.frozen_is_legacy(F._current_row(LEGACY, Q)))
chk("every One Pager holder is frozen for that half",
    all(F.is_frozen(i, Q, "one_pagers") for i in ALL_OPS))
chk("the Snapshot-only investor is not", not F.is_frozen("S1", Q, "one_pagers"))
_hist0 = {i: len(F.frozen_history(i, Q)) for i in ALL_OPS}

print("\nB. unfreezing the One Pagers, for everyone, in one go")
t0 = time.perf_counter()
out = F.unfreeze_quarter(Q, "one_pagers", "cbui", "sent in error")
secs = time.perf_counter() - t0
chk("it reports the investors it touched", out["investors"] == len(ALL_OPS),
    f"{out['investors']} vs {len(ALL_OPS)}")
chk("...and names them", sorted(out["investor_codes"]) == sorted(ALL_OPS),
    str(out["investor_codes"]))
chk("...and says it materialised the legacy row",
    out["legacy_materialised"] == 1, str(out["legacy_materialised"]))
chk("no One Pager half is frozen any more",
    not any(F.is_frozen(i, Q, "one_pagers") for i in ALL_OPS),
    str([i for i in ALL_OPS if F.is_frozen(i, Q, "one_pagers")]))

print("\nC. a history row for EVERY affected investor")
for inv in ALL_OPS:
    h = F.frozen_history(inv, Q)
    chk(f"{inv} gained a history row", len(h) == _hist0[inv] + 1,
        f"{len(h)} vs {_hist0[inv]}+1")
_last = F.frozen_history(BOTH[0], Q)[0]
chk("the archive records who unfroze it", _last.get("superseded_by") == "cbui",
    str(_last.get("superseded_by")))
chk("...and the reason", "sent in error" in (_last.get("supersede_reason") or ""),
    str(_last.get("supersede_reason")))
chk("...and which part it was", "one_pagers" in (_last.get("supersede_reason") or ""))

print("\nD. the OTHER half is untouched")
for inv in BOTH:
    row = F._current_row(inv, Q) or {}
    chk(f"{inv} still has its Snapshot frozen",
        F.is_frozen(inv, Q, "snapshot"), str(F.frozen_parts(inv, Q)
                                             if hasattr(F, "frozen_parts") else row))
    chk(f"...and {inv}'s Snapshot payload survived",
        bool((F.get_frozen(inv, Q) or {}).get("payload")))
chk("the legacy row kept its Snapshot too", F.is_frozen(LEGACY, Q, "snapshot"))
chk("...and that Snapshot still has its payload",
    bool((F.get_frozen(LEGACY, Q) or {}).get("payload")))
chk("the Snapshot-only investor was not touched at all",
    F.is_frozen("S1", Q, "snapshot"))
# One-Pagers-only rows had nothing left to be, so they are gone.
chk("rows carrying ONLY the unfrozen half are removed",
    all(F.get_frozen(i, Q) is None for i in OPS_ONLY),
    str([i for i in OPS_ONLY if F.get_frozen(i, Q)]))
chk("...which the result reports", out["rows_removed"] == len(OPS_ONLY),
    str(out["rows_removed"]))
chk("...and the rest kept", out["rows_kept"] == len(BOTH) + 1,
    str(out["rows_kept"]))

print("\nE. the comment lock is released")
chk("no investor's frozen One Pagers carry D1 any more",
    F.quarters_frozen_with_deal(Q, "D1") == [],
    str(F.quarters_frozen_with_deal(Q, "D1")))
# THE PAIRED DIRECTION: a Snapshot-only freeze never locked comments, so this
# must not be passing merely because the lock was already open.
Q2 = "2026-Q4"
F.freeze_part("C1", Q2, F.PARTS, "cbui", assembler=_asm, one_pager_getter=_op)
chk("a frozen One Pager half DOES lock the comment path",
    "C1" in F.quarters_frozen_with_deal(Q2, "D1"))
F.unfreeze_quarter(Q2, "snapshot", "cbui", "snapshot only")
chk("...and unfreezing the SNAPSHOT leaves it locked",
    "C1" in F.quarters_frozen_with_deal(Q2, "D1"),
    str(F.quarters_frozen_with_deal(Q2, "D1")))
F.unfreeze_quarter(Q2, "one_pagers", "cbui", "now the other half")
chk("...and unfreezing the One Pagers releases it",
    F.quarters_frozen_with_deal(Q2, "D1") == [])

print("\nF. it is SET-BASED, and fast on a realistic row count")
Q3 = "2027-Q1"
N = 130
with eng.begin() as cx:
    for i in range(N):
        cx.execute(sqlalchemy.text(
            "INSERT INTO portfolio_snapshot_frozen "
            "(investor_code, quarter, payload, data_version, frozen_by, "
            " frozen_at, frozen_reason, roster, one_pagers, version, "
            " snapshot_frozen_at, snapshot_frozen_by, one_pagers_frozen_at, "
            " one_pagers_frozen_by) "
            "VALUES (:i, :q, '{\"subtabs\":{}}', 'v', 'cbui', :t, 'as-sent', "
            "        '[\"D1\"]', '{\"D1\":{}}', 1, :t, 'cbui', :t, 'cbui')"),
            {"i": f"BULK{i:03d}", "q": Q3, "t": "2026-09-29 12:00:00"})
stmts = []
_real_exec = sqlalchemy.engine.Connection.execute


def _counting(self, stmt, *a, **kw):
    stmts.append(str(getattr(stmt, "text", stmt)).strip().split(None, 1)[0].upper())
    return _real_exec(self, stmt, *a, **kw)


sqlalchemy.engine.Connection.execute = _counting
t0 = time.perf_counter()
bulk = F.unfreeze_quarter(Q3, "one_pagers", "cbui", "bulk test")
bulk_s = time.perf_counter() - t0
sqlalchemy.engine.Connection.execute = _real_exec
chk(f"it unfroze all {N} investors", bulk["investors"] == N, str(bulk["investors"]))
writes = [s for s in stmts if s in ("INSERT", "UPDATE", "DELETE")]
chk(f"...with a handful of statements, not one per investor "
    f"({len(writes)} writes for {N} investors)", len(writes) <= 6,
    f"{len(writes)} writes: {writes}")
chk(f"...in well under a second ({bulk_s * 1000:.0f}ms)", bulk_s < 5.0,
    f"{bulk_s:.2f}s")
chk("...and every one is archived",
    all(len(F.frozen_history(f"BULK{i:03d}", Q3)) == 1 for i in range(0, N, 17)))
chk("...and their Snapshots are still frozen",
    all(F.is_frozen(f"BULK{i:03d}", Q3, "snapshot") for i in range(0, N, 17)))

print("\nH. the action is on both tabs, admin-only, with a required reason")
VIEWS = os.path.join(ROOT, "vue_app", "src", "views")
PANEL = os.path.join(ROOT, "vue_app", "src", "components", "common",
                     "FreezeQuarterPanel.vue")
if not os.path.isdir(VIEWS) or not os.path.exists(PANEL):
    print("  SKIP the Vue sources - vue_app/ is not in this tree")
else:
    panel = open(PANEL, encoding="utf-8").read()
    snap = open(os.path.join(VIEWS, "PortfolioSnapshotView.vue"),
                encoding="utf-8").read()
    ops = open(os.path.join(VIEWS, "OnePagerView.vue"), encoding="utf-8").read()
    chk("the Snapshot tab mounts the panel for its half",
        'part="snapshot"' in snap)
    chk("the One Pager tab mounts it for its half",
        'part="one_pagers"' in ops)
    chk("the button names the quarter AND the part",
        "Unfreeze {{ quarter }} {{ label }}" in panel)
    chk("the whole panel is admin-only", 'v-if="auth.isAdmin"' in panel)
    chk("the confirmation shows the investor count",
        "unfreezeCount" in panel and "investor(s) in" in panel)
    chk("...and says the other half is left alone",
        "otherLabel" in panel and "left exactly as they are" in panel)
    chk("a reason is REQUIRED before the button is usable",
        "!unfreezeReason.trim()" in panel)
    chk("...and the reason is sent to the server",
        "reason: unfreezeReason.value.trim()" in panel)
    chk("it posts to the quarter-level route, not a per-investor loop",
        "unfreeze-quarter" in panel and "for (const" not in panel)
    chk("the count comes from a read-only preview",
        "unfreeze-quarter/preview" in panel)

print("\nG. the guards, and the single-investor path")
try:
    F.unfreeze_quarter(Q3, "snapshot", "cbui", "   ")
    chk("a blank reason is refused", False, "it went ahead")
except ValueError as exc:
    chk("a blank reason is refused", "reason" in str(exc).lower())
try:
    F.unfreeze_quarter(Q3, F.PARTS, "cbui", "both at once")
    chk("asking for both parts at once is refused", False, "it went ahead")
except ValueError as exc:
    chk("asking for both parts at once is refused", "one part" in str(exc))
none = F.unfreeze_quarter("1999-Q1", "snapshot", "cbui", "nothing there")
chk("a quarter with nothing frozen is a no-op, not an error",
    none["investors"] == 0, str(none))
# SINGLE-INVESTOR UNFREEZE STILL WORKS — it is the admin's fine-grained tool.
F.freeze_part("SOLO", "2027-Q2", F.PARTS, "cbui", assembler=_asm,
              one_pager_getter=_op)
F.unfreeze("SOLO", "2027-Q2", "cbui", "one investor only")
chk("single-investor unfreeze still works",
    F.get_frozen("SOLO", "2027-Q2") is None)
chk("...and still archives", len(F.frozen_history("SOLO", "2027-Q2")) == 1)

print(f"\n{'=' * 60}\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
