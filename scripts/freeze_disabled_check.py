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

print(f"\n{'=' * 60}\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
