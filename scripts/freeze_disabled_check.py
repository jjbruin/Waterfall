"""Guardrail: with FREEZE_ENABLED off, nothing freezes — and everything else works.

WHY IT IS OFF. On Sep 29 2026 the all-investors batch ran against production as
a SINGLE request over ~145 investors and the app was unavailable for about 35
minutes. The rows were correct and were cleanly unfrozen afterwards; the shape
of the request was the problem. Freezing stays off until it runs as a background
job.

What this pins, and why each is here:

  1. EVERY freeze entry point refuses: both batch buttons, the published-overlay
     freeze, re-freeze, the Portfolio Snapshot approval chain, and the One Pager
     approval snapshot. Checked at the ENDPOINT (a readable 503) and at the CORE
     (`freeze_part` raises), because the endpoint list is a moving target and
     the core is what makes a future entry point safe by default.
  2. IT DEFAULTS OFF. An unset, empty, '0' or misspelt flag means off. A flag
     that defaults on protects nobody.
  3. APPROVALS STILL COMPLETE. The workflow advances and the status changes;
     only the freeze is skipped. This is the direction that would be missed —
     "nothing freezes" is trivially satisfied by breaking approvals altogether.
  4. UNFREEZE STILL WORKS, for admins. A mistake made before the switch has to
     remain correctable.
  5. WITH THE FLAG ON, freezing works again — otherwise every check above is
     satisfied by a permanently broken freeze.

Runs against a scratch SQLite with stubbed assembly: no application database,
no live call.

Usage
    .venv/Scripts/python.exe scripts/freeze_disabled_check.py
"""
from __future__ import annotations

import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Set BEFORE the app is built, so nothing can read a stale value at import time.
os.environ.pop("FREEZE_ENABLED", None)

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


tmpdir = tempfile.mkdtemp(prefix="freeze_off_")
eng = sqlalchemy.create_engine(f"sqlite:///{os.path.join(tmpdir, 't.db')}")

from flask_app.services import portfolio_snapshot_freeze as F      # noqa: E402
from flask_app.services import portfolio_snapshot_persistence as P  # noqa: E402
from flask_app.services import freeze_gate as G                    # noqa: E402

F._engine = lambda: eng
F._is_postgres = lambda: False
F._data_version = lambda: "build=test"

Q = "2026-Q2"


def _stub_report(investor, quarter):
    return {"subtabs": {"financial": {"groups": {"G1": {"deals": [
        {"vcode": "D1", "name": "Deal One"}]}}}},
        "errors": {}, "resolution": {}}


F.assemble_full_report = _stub_report
F._default_one_pager_getter = lambda: (lambda vc, q: {"vcode": vc})
P.load_page = lambda i, q: {"comments": [], "footnotes": [], "values": []}

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
ADMIN = {"Authorization": "Bearer " + _jwt.encode(
    {"sub": "1", "username": "cbui", "role": "admin",
     "exp": _dt.utcnow() + _td(hours=1)}, Config.JWT_SECRET, algorithm="HS256")}

S._investor_list = lambda data=None: [{"code": c, "name": c}
                                      for c in ("AAA", "BBB")]
cli = app.test_client()
SNAP = "/api/portfolio-snapshot/freeze-all/snapshots"
OPS = "/api/portfolio-snapshot/freeze-all/one-pagers"

print("A. the flag defaults OFF, and fails closed on anything unclear")
chk("unset means off", not G.freeze_enabled())
for val, want in (("", False), ("0", False), ("false", False), ("no", False),
                  ("maybe", False), ("TRUE", True), ("1", True),
                  ("yes", True), ("on", True)):
    os.environ["FREEZE_ENABLED"] = val
    with app.test_request_context():
        app.config["FREEZE_ENABLED"] = Config.__dict__.get("FREEZE_ENABLED")
    got = G._truthy(val)
    chk(f"{val!r} -> {'on' if want else 'off'}", got is want, str(got))
os.environ.pop("FREEZE_ENABLED", None)
app.config["FREEZE_ENABLED"] = False

print("\nB. every freeze entry point refuses at the ENDPOINT")
with app.app_context():
    pass
for name, url, body in (
        ("freeze all Snapshots", SNAP, {"quarter": Q}),
        ("freeze all One Pagers", OPS, {"quarter": Q}),
        ("freeze from published PDFs", "/api/portfolio-snapshot/freeze-overlay",
         {"investor": "AAA", "quarter": Q, "overlay": {}}),
        ("re-freeze", "/api/portfolio-snapshot/refreeze",
         {"investor": "AAA", "quarter": Q, "reason": "x"})):
    r = cli.post(url, json=body, headers=ADMIN)
    b = r.get_json() or {}
    chk(f"{name} is refused", r.status_code == 503, f"got {r.status_code}")
    chk("...saying freezing is disabled",
        "disabled" in str(b.get("error", "")).lower(), str(b)[:90])
    chk("...and flagged so the screen can tell it from an outage",
        b.get("freeze_disabled") is True)

print("\nC. ...and at the CORE, so a new entry point is safe by default")
with app.app_context():
    for label, fn in (
            ("freeze_part", lambda: F.freeze_part("AAA", Q, F.PART_SNAPSHOT, "cbui")),
            ("freeze() (approval chain)", lambda: F.freeze("AAA", Q, "cbui"))):
        try:
            fn()
            chk(f"{label} refuses", False, "it returned instead of raising")
        except G.FreezeDisabled:
            chk(f"{label} refuses with FreezeDisabled", True)
        except Exception as exc:                              # noqa: BLE001
            chk(f"{label} refuses with FreezeDisabled", False,
                f"{type(exc).__name__}: {exc}")

# REFREEZE NEEDS SOMETHING FROZEN FIRST, or it refuses for the wrong reason
# ("not frozen") and the check passes without ever reaching the gate. Freeze
# with the flag ON, switch it off, and only then try.
os.environ["FREEZE_ENABLED"] = "1"
app.config["FREEZE_ENABLED"] = True
with app.app_context():
    F.freeze_part("BBB", Q, F.PARTS, "cbui")
_before = F.get_frozen("BBB", Q)
os.environ.pop("FREEZE_ENABLED", None)
app.config["FREEZE_ENABLED"] = False
with app.app_context():
    try:
        F.refreeze("BBB", Q, "cbui", "a correction")
        chk("refreeze() refuses", False, "it returned instead of raising")
    except G.FreezeDisabled:
        chk("refreeze() refuses with FreezeDisabled — reaching the GATE, "
            "not the 'not frozen' guard", True)
    except Exception as exc:                                  # noqa: BLE001
        chk("refreeze() refuses with FreezeDisabled", False,
            f"{type(exc).__name__}: {exc}")
_after = F.get_frozen("BBB", Q)
chk("...and the frozen copy it would have replaced is untouched",
    (_before or {}).get("version") == (_after or {}).get("version")
    and (_before or {}).get("frozen_at") == (_after or {}).get("frozen_at"))
with app.app_context():
    F.unfreeze("BBB", Q, "cbui", "tidying the fixture")

chk("NOTHING was written by the refused calls", F.get_frozen("AAA", Q) is None)
chk("...for either investor", F.get_frozen("BBB", Q) is None)

print("\nD. the One Pager approval snapshot is gated too")
import flask_app.services.review_service as RS                     # noqa: E402
from flask_app.services import data_service as _ds                 # noqa: E402

# OBSERVE THE CALL, DO NOT INFER IT FROM THE RETURN. `_save_snapshot` wraps its
# whole body in try/except and swallows failures, so "it returned without
# raising" is true whether or not it ran — the first version of this check
# asserted exactly that and passed with the gate deleted. `get_data()` is the
# first thing the body does, so whether it was reached IS the answer.
_reached = {"n": 0}
_real_get_data = _ds.get_data


def _recording_get_data(*a, **kw):
    _reached["n"] += 1
    raise RuntimeError("the body was entered")


_ds.get_data = _recording_get_data
try:
    with app.app_context():
        RS._save_snapshot("P0000001", Q, "cbui")
        _raised_d = None
except Exception as exc:                                      # noqa: BLE001
    _raised_d = exc
finally:
    _ds.get_data = _real_get_data

chk("the snapshot body is never entered with freezing off", _reached["n"] == 0,
    f"get_data() called {_reached['n']} time(s)")
chk("...and it does not raise into the approval", _raised_d is None,
    f"{type(_raised_d).__name__}: {_raised_d}" if _raised_d else "")

# THE PAIRED DIRECTION: with the flag ON it MUST enter the body, or the check
# above is satisfied by a function that never does anything.
os.environ["FREEZE_ENABLED"] = "1"
app.config["FREEZE_ENABLED"] = True
_ds.get_data = _recording_get_data
try:
    with app.app_context():
        RS._save_snapshot("P0000001", Q, "cbui")
finally:
    _ds.get_data = _real_get_data
    os.environ.pop("FREEZE_ENABLED", None)
    app.config["FREEZE_ENABLED"] = False
chk("...and with freezing ON it IS entered", _reached["n"] == 1,
    f"get_data() called {_reached['n']} time(s)")

print("\nE. an approval still COMPLETES — only the freeze is skipped")
# THE PAIRED DIRECTION, and the one that would be missed: "nothing freezes" is
# trivially satisfied by breaking approvals altogether. Driven through the REAL
# `approve`, at the REAL final transition (step 4 -> 'approved'), which is the
# only step that calls freeze().
_state = {"status": "pending_ceo", "step": 4, "approver": None}


def _fake_set_status(inv, q, status, step, approver=None):
    _state.update(status=status, step=step, approver=approver)


P._ensure_tables = lambda: None
P._set_status = _fake_set_status
P.document_status = lambda inv, q: {"status": _state["status"],
                                    "current_step": _state["step"]}
P._user_roles = lambda uid, roles=None: set(roles or [])

with app.app_context():
    try:
        P.approve("AAA", Q, 1, "cbui", roles=["ceo"])
        _raised = None
    except Exception as exc:                                  # noqa: BLE001
        _raised = exc

chk("the approval does not raise when the freeze is disabled",
    _raised is None, f"{type(_raised).__name__}: {_raised}" if _raised else "")
chk("...the workflow really advanced to 'approved'",
    _state["status"] == "approved", str(_state))
chk("...and it recorded who approved", _state["approver"] == "cbui",
    str(_state["approver"]))
chk("...but NOTHING was frozen by it", F.get_frozen("AAA", Q) is None)

print("\nF. unfreeze still works for admins")
os.environ["FREEZE_ENABLED"] = "1"
app.config["FREEZE_ENABLED"] = True
with app.app_context():
    F.freeze_part("AAA", Q, F.PARTS, "cbui")
chk("with the flag ON a freeze succeeds again",
    F.is_frozen("AAA", Q, F.PART_SNAPSHOT))
os.environ.pop("FREEZE_ENABLED", None)
app.config["FREEZE_ENABLED"] = False
with app.app_context():
    F.unfreeze("AAA", Q, "cbui", "frozen by mistake")
chk("unfreeze works with freezing DISABLED", F.get_frozen("AAA", Q) is None)
chk("...and it is archived, not lost", len(F.frozen_history("AAA", Q)) >= 1)
r = cli.post("/api/portfolio-snapshot/unfreeze",
             json={"investor": "AAA", "quarter": Q, "reason": "x"},
             headers=ADMIN)
chk("the unfreeze endpoint is not gated by the flag",
    r.status_code != 503, f"got {r.status_code}")

print("\nG. the switch is visible to the screen")
st = cli.get(f"/api/portfolio-snapshot/quarter-status?quarter={Q}",
             headers=ADMIN).get_json() or {}
chk("quarter-status carries freeze_enabled", st.get("freeze_enabled") is False,
    str(st.get("freeze_enabled")))
chk("...with a reason to show", bool(st.get("freeze_disabled_reason")))
PANEL = os.path.join(ROOT, "vue_app", "src", "components", "common",
                     "FreezeQuarterPanel.vue")
if os.path.exists(PANEL):
    src = open(PANEL, encoding="utf-8").read()
    chk("the panel disables the button on it", "!freezeEnabled" in src)
    chk("...and SAYS why rather than only greying out", "fqp-off" in src)
    chk("...treating a failed read as disabled",
        "freeze_enabled === true" in src)
else:
    print("  SKIP the panel — vue_app/ is not in this tree")

print("\nH. a freeze is not an approval — approved_at stays NULL")
# `approved_at` used to be a column DEFAULT, so EVERY row carried one, including
# an as-sent freeze that deliberately leaves `approved_by` NULL. That is a
# timestamp nobody produced, in the column that records a decision — and it also
# made the legacy fallback in section I fire on every unstamped row.
os.environ["FREEZE_ENABLED"] = "1"
app.config["FREEZE_ENABLED"] = True
with app.app_context():
    F.freeze_part("ASSENT", Q, F.PARTS, "cbui")
_raw = F._current_row("ASSENT", Q) or {}
chk("an as-sent freeze leaves approved_by NULL", _raw.get("approved_by") is None,
    str(_raw.get("approved_by")))
chk("...and approved_at NULL, not a default timestamp",
    _raw.get("approved_at") is None, str(_raw.get("approved_at")))

# THE PAIRED DIRECTION. "approved_at is never set" would be satisfied by a
# column that can no longer be written at all, which would lose the approval.
with app.app_context():
    F.freeze("APPROVED", Q, "ceo-user")
_ap = F._current_row("APPROVED", Q) or {}
chk("an APPROVAL-chain freeze does set approved_by",
    _ap.get("approved_by") == "ceo-user", str(_ap.get("approved_by")))
chk("...and sets approved_at with it", _ap.get("approved_at") is not None)

# And a later part-freeze must not erase the moment the approval happened.
with app.app_context():
    F.freeze_part("APPROVED", Q, F.PART_ONE_PAGERS, "cbui")
_ap2 = F._current_row("APPROVED", Q) or {}
chk("a later part-freeze carries the approval forward",
    _ap2.get("approved_by") == "ceo-user"
    and _ap2.get("approved_at") is not None, str(_ap2.get("approved_at")))

print("\nI. a legacy row is reported as legacy, not asserted as both halves")
# A row written BEFORE the per-part columns existed: frozen_at set, neither part
# stamped. It really did freeze both halves — the old code froze the whole row —
# so reporting both is right; claiming it was MEASURED is not.
with F._engine().begin() as _cx:
    _cx.execute(sqlalchemy.text(
        "UPDATE portfolio_snapshot_frozen "
        "SET snapshot_frozen_at = NULL, one_pagers_frozen_at = NULL "
        "WHERE investor_code = 'ASSENT'"))
_legacy = F._current_row("ASSENT", Q) or {}
chk("a legacy row still reports BOTH halves",
    F.frozen_parts_of(_legacy) == list(F.PARTS), str(F.frozen_parts_of(_legacy)))
chk("...and is FLAGGED as inferred", F.frozen_is_legacy(_legacy) is True)
chk("...and get_frozen carries the flag",
    (F.get_frozen("ASSENT", Q) or {}).get("frozen_parts_legacy") is True)

# A row with real per-part stamps is NOT legacy — otherwise the flag means
# nothing, being true of everything.
chk("a properly stamped row is NOT legacy",
    F.frozen_is_legacy(F._current_row("APPROVED", Q)) is False)
chk("...and get_frozen says so",
    (F.get_frozen("APPROVED", Q) or {}).get("frozen_parts_legacy") is False)

# THE NARROWING ITSELF: approved_at alone must no longer imply frozen.
_approved_only = {"approved_at": "2026-09-28 21:22:03", "frozen_at": None,
                  "snapshot_frozen_at": None, "one_pagers_frozen_at": None}
chk("approved_at ALONE no longer reads as frozen",
    F.frozen_parts_of(_approved_only) == [], str(F.frozen_parts_of(_approved_only)))
chk("...and is not called legacy either", F.frozen_is_legacy(_approved_only) is False)
chk("frozen_at alone still does read as frozen",
    F.frozen_parts_of({"frozen_at": "2026-09-28 21:22:03"}) == list(F.PARTS))

_qs = F.quarter_part_state(Q)
chk("quarter_part_state names the legacy investors",
    "ASSENT" in (_qs.get("legacy_investors") or []), str(_qs.get("legacy_investors")))
chk("...and does not name a properly stamped one",
    "APPROVED" not in (_qs.get("legacy_investors") or []))

os.environ.pop("FREEZE_ENABLED", None)
app.config["FREEZE_ENABLED"] = False

print("\nJ. the DDL no longer defaults approved_at")
_src = open(os.path.join(ROOT, "flask_app", "services",
                         "portfolio_snapshot_freeze.py"), encoding="utf-8").read()
chk("CREATE TABLE declares approved_at with no default",
    "approved_at TIMESTAMP DEFAULT" not in _src)
chk("an existing table is migrated, not only new ones",
    "_drop_approved_at_default" in _src)
chk("...and the INSERT NAMES approved_at, which is what closes it on SQLite too",
    "approved_by, approved_at" in _src)

print("\nK. the schema work is off the read path, and the ALTER is guarded")
# THE DEFECT THIS PINS. The first version ran an unconditional
# `ALTER TABLE ... DROP DEFAULT` inside `_ensure_table`, which is reached from
# `_current_row`, `quarter_part_state` and `quarters_frozen_with_deal` — the
# last of which the One Pager comment lock calls on EVERY save. On PostgreSQL
# that takes an ACCESS EXCLUSIVE lock, so every read serialised behind a
# catalog lock for a migration with work to do exactly once.
_ddl = []
_real_connect = eng.connect
_real_begin = eng.begin

import sqlalchemy.engine as _sae                                   # noqa: E402


class _Spy:
    """Wraps a connection and records every DDL statement executed on it."""

    def __init__(self, inner):
        self._inner = inner

    def execute(self, stmt, *a, **kw):
        sql = str(getattr(stmt, "text", stmt))
        head = sql.strip().split(None, 1)[0].upper() if sql.strip() else ""
        if head in ("CREATE", "ALTER", "DROP"):
            _ddl.append(" ".join(sql.split())[:70])
        return self._inner.execute(stmt, *a, **kw)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __enter__(self):
        # `engine.begin()` returns a CONTEXT MANAGER, not a connection, so the
        # thing to wrap is what __enter__ yields — wrapping the manager itself
        # gives an object with no .execute, which is how this first failed.
        return _Spy(self._inner.__enter__())

    def __exit__(self, *a):
        return self._inner.__exit__(*a)


eng.connect = lambda *a, **kw: _Spy(_real_connect(*a, **kw))
eng.begin = lambda *a, **kw: _Spy(_real_begin(*a, **kw))

# The schema is already built by this point in the run, so these are the
# steady state — a warm process serving ordinary traffic.
_ddl.clear()
F._current_row("APPROVED", Q)
chk("_current_row issues no DDL", not _ddl, str(_ddl[:2]))
_ddl.clear()
F.quarter_part_state(Q)
chk("quarter_part_state issues no DDL", not _ddl, str(_ddl[:2]))
_ddl.clear()
F.quarters_frozen_with_deal(Q, "D1")
chk("quarters_frozen_with_deal issues no DDL — the comment-lock path",
    not _ddl, str(_ddl[:2]))
_ddl.clear()
F.get_frozen("APPROVED", Q)
chk("get_frozen issues no DDL", not _ddl, str(_ddl[:2]))

# THE ALTER ITSELF: at most once, and never with no default present. On SQLite
# `_is_postgres()` is False so it returns before touching anything — assert the
# guard exists in BOTH forms rather than only the one this engine exercises.
_ddl.clear()
for _ in range(5):
    F._drop_approved_at_default()
chk("the migration issues no ALTER when there is nothing to drop",
    not [d for d in _ddl if d.upper().startswith("ALTER")], str(_ddl[:2]))
_src_fz = open(os.path.join(ROOT, "flask_app", "services",
                            "portfolio_snapshot_freeze.py"), encoding="utf-8").read()
chk("...because it asks information_schema first",
    "information_schema.columns" in _src_fz
    and "if not have_default:" in _src_fz)
chk("...and returns before the ALTER when there is none",
    _src_fz.index("if not have_default:")
    < _src_fz.index("ALTER COLUMN approved_at DROP DEFAULT"))

# ONCE PER PROCESS: a second ensure does nothing; forcing re-runs it.
_ddl.clear()
F._ensure_table()
chk("a warm _ensure_table issues no DDL at all", not _ddl, str(_ddl[:2]))
_ddl.clear()
F.ensure_schema(force=True)
chk("...and force=True really does re-run it", bool(_ddl), "no DDL seen")
chk("...creating the tables it is responsible for",
    any(d.upper().startswith("CREATE") for d in _ddl), str(_ddl[:2]))

eng.connect = _real_connect
eng.begin = _real_begin
chk("startup calls it, so the first request does not pay for it",
    "ensure_schema()" in open(os.path.join(ROOT, "flask_app", "__init__.py"),
                              encoding="utf-8").read())

print(f"\n{'=' * 60}\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
