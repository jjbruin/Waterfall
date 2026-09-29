"""Guardrail: freezing a QUARTER, per half, for every investor.

Runs entirely against a scratch SQLite with stubbed assembly — NO application
database and NO live call, so it is safe anywhere and cannot freeze a real
quarter. The screen half reads the Vue sources and SKIPS with a reason when they
are absent, so this still runs inside the container image (which ships no
`vue_app/`).

What it pins, and why each is here rather than assumed:

  1. THE SNAPSHOT BUTTON FREEZES ONLY SNAPSHOTS, and the One Pager button only
     One Pagers. Asserted in BOTH directions: "the Snapshot is frozen" alone is
     satisfied by a batch that freezes everything, which is the exact failure
     the two-button split exists to prevent.
  2. IT COVERS EVERY INVESTOR, and the per-investor `investors` slice the screen
     posts in chunks reaches the same place. One investor passing proves
     nothing — a bug collapsing the list to its first element satisfies it.
  3. AN ALREADY-FROZEN INVESTOR IS SKIPPED **AND UNTOUCHED**. Reporting
     "skipped" is not enough on its own: a batch that silently re-froze while
     printing "skipped" would pass that, and the whole point is that a
     published-PDF freeze is never overwritten. The STORED ROW is compared
     before and after, byte for byte on the fields that would move.
  4. A FAILING INVESTOR IS REPORTED AND THE REST STILL FREEZE — driven over
     HTTP, not by calling the service in a loop, because the endpoint's own
     isolation is the thing the button depends on.
  5. NON-ADMINS ARE REFUSED, and admins are not. A rule checked only in the
     refusing direction is satisfied by locking everyone out.
  6. THE QUARTER STATUS COUNTS none / partly / all against the SAME population
     the batch iterates, and never reports more frozen than there are investors.
  7. EACH BUTTON EXISTS IN EXACTLY ONE PLACE, and both go through the one
     shared panel.

Usage
    .venv/Scripts/python.exe scripts/freeze_quarter_ui_check.py
"""
from __future__ import annotations

import os
import sys
import tempfile

# FREEZING IS OFF BY DEFAULT (FREEZE_ENABLED, see flask_app/services/freeze_gate.py)
# and this suite exists to exercise freezing, so it switches it ON for itself,
# explicitly and in-process. Set BEFORE flask_app is imported: the Config class
# reads the environment at import time.
#
# It does NOT weaken the gate — scripts/freeze_disabled_check.py owns the OFF
# direction and asserts every entry point refuses. Two suites, one for each
# state, rather than one suite that silently depends on whichever state it
# happens to run in.
import os as _os
_os.environ["FREEZE_ENABLED"] = "1"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import sqlalchemy  # noqa: E402

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


# ── scratch engine, wired in before anything touches a real database ────────
tmpdir = tempfile.mkdtemp(prefix="freeze_quarter_")
eng = sqlalchemy.create_engine(f"sqlite:///{os.path.join(tmpdir, 't.db')}")

from flask_app.services import portfolio_snapshot_freeze as F      # noqa: E402
from flask_app.services import portfolio_snapshot_persistence as P  # noqa: E402

F._engine = lambda: eng
F._is_postgres = lambda: False
F._data_version = lambda: "build=test;actuals_through=2026-07-31"

Q = "2026-Q2"
BROKEN = "BADINV"          # the investor whose One Pagers cannot be built


def _stub_report(investor, quarter):
    return {
        "subtabs": {"financial": {"groups": {"G1": {"deals": [
            {"vcode": "D1", "name": "Deal One"},
            {"vcode": "D2", "name": "Deal Two"},
        ]}}}},
        "errors": {},
        "resolution": {"investor_name": investor, "quarter": quarter},
    }


def _stub_op(vcode, quarter):
    # One investor's One Pagers blow up, so section D has a real failure to
    # isolate rather than a simulated one.
    if _CURRENT.get("investor") == BROKEN:
        raise RuntimeError("one pager build blew up")
    return {"vcode": vcode, "quarter": quarter}


_CURRENT: dict = {}
_real_freeze_part = F.freeze_part


def _tracking_freeze_part(investor_code, quarter, part, frozen_by, **kw):
    """Records which investor is in flight, so the stub can fail just one."""
    _CURRENT["investor"] = investor_code
    try:
        return _real_freeze_part(investor_code, quarter, part, frozen_by, **kw)
    finally:
        _CURRENT.pop("investor", None)


F.assemble_full_report = _stub_report
F._default_one_pager_getter = lambda: _stub_op
F.freeze_part = _tracking_freeze_part
P.load_page = lambda i, q: {"comments": [], "footnotes": [], "values": []}

# ── the app, with a real token (the decorators are applied at import time) ──
from flask import Flask                                            # noqa: E402
from flask_app.config import Config                                # noqa: E402
import flask_app.api.portfolio_snapshot as S                       # noqa: E402
import jwt as _jwt                                                 # noqa: E402
from datetime import datetime as _dt, timedelta as _td             # noqa: E402

app = Flask(__name__)
app.config.from_object(Config)
app.config["TESTING"] = True
app.register_blueprint(S.portfolio_snapshot_bp,
                       url_prefix="/api/portfolio-snapshot")


def _tok(role, username="cbui"):
    return {"Authorization": "Bearer " + _jwt.encode(
        {"sub": "1", "username": username, "role": role,
         "exp": _dt.utcnow() + _td(hours=1)},
        Config.JWT_SECRET, algorithm="HS256")}


ADMIN = _tok("admin")
ANALYST = _tok("analyst", "jim")
VIEWER = _tok("viewer", "guest")

# The population the batch iterates. Patched because the real one reads the
# application database; the ENDPOINT still calls it, so the wiring is unchanged.
INVESTORS = ["AAA", "BBB", "CCC", BROKEN, "EEE"]
S._investor_list = lambda data=None: [{"code": c, "name": c} for c in INVESTORS]

cli = app.test_client()
SNAP = "/api/portfolio-snapshot/freeze-all/snapshots"
OPS = "/api/portfolio-snapshot/freeze-all/one-pagers"
STATUS = "/api/portfolio-snapshot/quarter-status"


def _row(inv, quarter=Q):
    return F._current_row(inv, quarter) or {}


# ── A. the Snapshot batch freezes ONLY the Snapshot half ───────────────────
print("A. the Snapshot button freezes only Snapshots, for every investor")
r = cli.post(SNAP, json={"quarter": Q}, headers=ADMIN)
body = r.get_json() or {}
chk("the batch answers", r.status_code in (200, 207), f"{r.status_code} {str(body)[:120]}")
chk("every investor was attempted", body.get("investors") == len(INVESTORS),
    str(body.get("investors")))
chk("all of them froze", body.get("frozen") == len(INVESTORS), str(body.get("frozen")))
chk("...and the store agrees, for EVERY investor",
    all(F.is_frozen(c, Q, F.PART_SNAPSHOT) for c in INVESTORS))
# THE PAIRED DIRECTION — without this, a batch freezing both halves passes.
chk("the One Pagers are NOT frozen by the Snapshot button",
    not any(F.is_frozen(c, Q, F.PART_ONE_PAGERS) for c in INVESTORS),
    str([c for c in INVESTORS if F.is_frozen(c, Q, F.PART_ONE_PAGERS)]))
chk("...and the stored row really holds no One Pagers",
    all(not (_row(c).get("one_pagers_frozen_at")) for c in INVESTORS))

# ── B. an already-frozen investor is skipped AND UNTOUCHED ─────────────────
print("\nB. an already-frozen investor is skipped and left exactly as it was")
before = {c: (_row(c).get("version"), str(_row(c).get("snapshot_frozen_at")))
          for c in INVESTORS}
r2 = cli.post(SNAP, json={"quarter": Q}, headers=ADMIN)
b2 = r2.get_json() or {}
chk("every investor is reported skipped", b2.get("skipped") == len(INVESTORS),
    str(b2.get("skipped")))
chk("nothing froze on the second run", b2.get("frozen") == 0, str(b2.get("frozen")))
after = {c: (_row(c).get("version"), str(_row(c).get("snapshot_frozen_at")))
         for c in INVESTORS}
# THE CHECK THAT MATTERS: "skipped" in the response is a CLAIM. This is the
# evidence — a silent re-freeze would bump version and move the timestamp.
chk("the stored rows did not move (version and timestamp identical)",
    before == after, f"{before} != {after}")
chk("the skip says why, rather than reporting a bare count",
    all((row.get("reason") or "") for row in (b2.get("results") or [])
        if row.get("skipped")))

# ── C. the One Pager batch freezes ONLY the One Pager half ─────────────────
print("\nC. the One Pager button freezes only One Pagers")
r3 = cli.post(OPS, json={"quarter": Q}, headers=ADMIN)
b3 = r3.get_json() or {}
good = [c for c in INVESTORS if c != BROKEN]
chk("the good investors froze", b3.get("frozen") == len(good), str(b3.get("frozen")))
chk("...and the store agrees",
    all(F.is_frozen(c, Q, F.PART_ONE_PAGERS) for c in good))
chk("the Snapshot half is untouched by it — still frozen from section A",
    all(F.is_frozen(c, Q, F.PART_SNAPSHOT) for c in INVESTORS))
chk("...and its timestamps did not move",
    {c: str(_row(c).get("snapshot_frozen_at")) for c in INVESTORS}
    == {c: before[c][1] for c in INVESTORS})

# ── D. one investor failing does not stop the rest ─────────────────────────
print("\nD. a failing investor is reported and the rest still freeze")
rows = {r["investor"]: r for r in (b3.get("results") or [])}
chk("the failing investor is reported as not frozen",
    rows.get(BROKEN, {}).get("frozen") is False, str(rows.get(BROKEN)))
chk("...with an error naming the cause",
    "blew up" in str(rows.get(BROKEN, {}).get("error", "")),
    str(rows.get(BROKEN, {}).get("error"))[:120])
chk("the investor BEFORE it froze", rows.get("CCC", {}).get("frozen") is True)
chk("the investor AFTER it froze", rows.get("EEE", {}).get("frozen") is True)
chk("its One Pagers are genuinely not frozen",
    not F.is_frozen(BROKEN, Q, F.PART_ONE_PAGERS))
chk("the response reports a partial run (207), not a blanket success",
    r3.status_code == 207, str(r3.status_code))

# ── E. the chunked call the screen makes reaches the same place ────────────
print("\nE. the investors slice the screen posts in chunks is honoured")
QC = "2026-Q1"
rc = cli.post(SNAP, json={"quarter": QC, "investors": ["AAA", "CCC"]},
              headers=ADMIN)
bc = rc.get_json() or {}
chk("only the named investors are touched", bc.get("investors") == 2,
    str(bc.get("investors")))
chk("...and they froze", F.is_frozen("AAA", QC, F.PART_SNAPSHOT)
    and F.is_frozen("CCC", QC, F.PART_SNAPSHOT))
# Pinned in BOTH directions: a bug ignoring the slice would freeze everyone.
chk("an investor NOT in the slice is untouched",
    not F.is_frozen("BBB", QC, F.PART_SNAPSHOT))
chk("...and two slices together cover the quarter",
    (cli.post(SNAP, json={"quarter": QC, "investors": ["BBB", BROKEN, "EEE"]},
              headers=ADMIN).get_json() or {}).get("frozen") == 3)

# ── F. who may press it ────────────────────────────────────────────────────
print("\nF. the batch is admin-only, and admins are not locked out")
for label, hdr in (("analyst", ANALYST), ("viewer", VIEWER)):
    for name, url in (("Snapshots", SNAP), ("One Pagers", OPS)):
        got = cli.post(url, json={"quarter": "2027-Q1"}, headers=hdr).status_code
        chk(f"{label} is refused on {name}", got == 403, f"got {got}")
chk("...and nothing was frozen by those attempts",
    not F.is_frozen("AAA", "2027-Q1"))
# THE PAIRED DIRECTION. Refusing everybody would satisfy every check above.
chk("an admin is still admitted",
    cli.post(SNAP, json={"quarter": "2027-Q1"},
             headers=ADMIN).status_code in (200, 207))
chk("...and that really froze something", F.is_frozen("AAA", "2027-Q1",
                                                      F.PART_SNAPSHOT))
# The STATUS read is open to any signed-in user — it is what draws the state
# line, and a viewer who cannot read it sees a blank where "frozen" should be.
chk("a viewer may still READ the quarter state",
    cli.get(f"{STATUS}?quarter={Q}", headers=VIEWER).status_code == 200)
chk("an anonymous caller may not", cli.get(f"{STATUS}?quarter={Q}").status_code == 401)

# ── G. the quarter status: none / partly / all ─────────────────────────────
print("\nG. the quarter status counts against the population the batch uses")
st = (cli.get(f"{STATUS}?quarter={Q}", headers=ADMIN).get_json() or {})
chk("it reports the investor population", st.get("investors") == len(INVESTORS),
    str(st.get("investors")))
snap = (st.get("parts") or {}).get("snapshot") or {}
ops = (st.get("parts") or {}).get("one_pagers") or {}
chk("the Snapshot half reads 'all'", snap.get("state") == "all", str(snap))
chk("...with every investor counted", snap.get("frozen") == len(INVESTORS))
chk("the One Pager half reads 'partly'", ops.get("state") == "partly", str(ops))
chk("...counting the ones that really froze", ops.get("frozen") == len(good),
    str(ops.get("frozen")))
chk("...and saying how many are left", ops.get("remaining") == 1,
    str(ops.get("remaining")))
fresh = (cli.get(f"{STATUS}?quarter=2025-Q3", headers=ADMIN).get_json() or {})
chk("an untouched quarter reads 'none'",
    ((fresh.get("parts") or {}).get("snapshot") or {}).get("state") == "none")
chk("...and counts zero, not null",
    ((fresh.get("parts") or {}).get("snapshot") or {}).get("frozen") == 0)
chk("a missing quarter is refused",
    cli.get(STATUS, headers=ADMIN).status_code == 400)

# A frozen investor no longer in the population must not inflate the count —
# it would render as "6 of 5" and read as a bug in the screen.
F.freeze_part("GONE", Q, F.PART_SNAPSHOT, "cbui",
              assembler=_stub_report,
              elements_loader=lambda i, q: {"comments": [], "footnotes": [],
                                            "values": []})
st2 = (cli.get(f"{STATUS}?quarter={Q}", headers=ADMIN).get_json() or {})
snap2 = (st2.get("parts") or {}).get("snapshot") or {}
chk("an investor off the current list does not inflate the count",
    snap2.get("frozen") <= snap2.get("total"), str(snap2))
chk("...and is excluded, not silently counted",
    "GONE" not in (snap2.get("investors") or []))

# ── H. the overlay note ────────────────────────────────────────────────────
print("\nH. the published-overlay note reports, and never blocks")
ov = st.get("overlay") or {}
chk("the note is present on every status read", "pending" in ov, str(ov))
chk("it states its basis rather than asserting bare truth", bool(ov.get("basis")))
_seen = F.quarter_part_state(Q)
chk("no overlay was applied in this run, so none is claimed",
    _seen.get("overlay_investors") == [], str(_seen.get("overlay_investors")))
chk("with no overlay file on the server, nothing is pending",
    ov.get("pending") is False or S._overlay_file_for(Q) is not None, str(ov))
chk("a malformed quarter yields no overlay file rather than raising",
    S._overlay_file_for("nonsense") is None)
chk("the overlay filename follows the built convention",
    S._overlay_file_for.__doc__ and "overlay_26q2" in S._overlay_file_for.__doc__)

# ── I. the screens: one button each, through one shared panel ──────────────
print("\nI. each button exists in exactly one place")
VIEWS = os.path.join(ROOT, "vue_app", "src", "views")
PANEL = os.path.join(ROOT, "vue_app", "src", "components", "common",
                     "FreezeQuarterPanel.vue")
if not os.path.isdir(VIEWS) or not os.path.exists(PANEL):
    skip("the Vue sources", "vue_app/ is not in this tree (container image)")
else:
    snap_src = open(os.path.join(VIEWS, "PortfolioSnapshotView.vue"),
                    encoding="utf-8").read()
    op_src = open(os.path.join(VIEWS, "OnePagerView.vue"), encoding="utf-8").read()
    panel = open(PANEL, encoding="utf-8").read()

    chk("the Snapshot tab mounts the panel for the snapshot half",
        'part="snapshot"' in snap_src)
    chk("the One Pager tab mounts it for the one_pagers half",
        'part="one_pagers"' in op_src)
    chk("the Snapshot tab does NOT carry the One Pager button",
        'part="one_pagers"' not in snap_src)
    chk("the One Pager tab does NOT carry the Snapshot button",
        'part="snapshot"' not in op_src)
    chk("both import the SAME panel, so there is one implementation",
        "FreezeQuarterPanel" in snap_src and "FreezeQuarterPanel" in op_src)
    chk("the old per-half buttons are gone from the Snapshot tab",
        "Freeze all One Pagers" not in snap_src
        and "Freeze all Snapshots" not in snap_src)
    chk("the old batch handler is gone with them",
        "doFreezeAll" not in snap_src and "freezeAllPart" not in snap_src)
    chk("the published-overlay freeze is still its own action",
        "showOverlayPanel" in snap_src and "freeze-overlay" in snap_src)
    chk("the button names the quarter", "Freeze {{ quarter }}" in panel)
    chk("...and says it covers all investors", "all investors" in panel)
    chk("the panel is admin-gated on the screen too", "auth.isAdmin" in panel)
    chk("it shows the three states", all(
        s in panel for s in ("Not frozen", "Partly frozen", "frozen — all")))
    chk("it reports progress while it runs", "progress" in panel and "pct" in panel)
    chk("it warns when the published PDFs are still to be applied",
        "Apply the published PDFs first" in panel)
    chk("it reports per investor, not just a count",
        "result.results" in panel and "r.investor" in panel)
    chk("it posts in slices, so progress is real rather than animated",
        "CHUNK" in panel and "investors: slice" in panel)

# ── J. nothing that was removed is still referenced ────────────────────────
print("\nJ. the removals left nothing dangling")
chk("the unused frozen_parts() wrapper is gone",
    not hasattr(F, "frozen_parts"))
chk("...and frozen_parts_of, which callers DO use, is still there",
    callable(getattr(F, "frozen_parts_of", None)))
chk("freeze_part is untouched and still the one core",
    callable(getattr(F, "freeze_part", None)))
chk("the bulk read reuses it rather than re-deriving the rule",
    "frozen_parts_of" in (F.quarter_part_state.__doc__ or "")
    or "frozen_parts_of" in open(
        os.path.join(ROOT, "flask_app", "services",
                     "portfolio_snapshot_freeze.py"), encoding="utf-8"
    ).read().split("def quarter_part_state")[1][:2000])

print(f"\n{'=' * 60}\n{PASS} passed, {FAIL} failed, {SKIP} skipped")
sys.exit(1 if FAIL else 0)
