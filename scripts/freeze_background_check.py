"""Guardrail: the sustainable freeze — each deal built once, in the background.

Runs against a scratch SQLite with stubbed assembly: no application database,
no live call, and it cannot freeze a real quarter.

What it pins, and why each is here rather than assumed:

  1. EACH DEAL IS BUILT ONCE PER BATCH. Counted at the builder, not inferred
     from a timing. Asserted in BOTH directions — the reuse count must also be
     right, or "built once" is satisfied by a batch that builds nothing.
  2. THE RESULT IS IDENTICAL to the per-investor build. The dedup is a
     performance change; if the stored bytes move, it is not. Compared payload
     by payload against a run with the cache disabled.
  3. SKIP-IF-FROZEN STILL HOLDS, and the stored row does not move.
  4. THE APP STAYS RESPONSIVE. A request issued WHILE a job runs completes
     quickly. Without this the whole change is pointless: the synchronous
     version was correct too.
  5. A SECOND JOB IS REFUSED, from the database rather than from memory.
  6. AN INTERRUPTED JOB IS RECORDED as interrupted, and what was frozen before
     the interruption is still frozen.

Usage
    .venv/Scripts/python.exe scripts/freeze_background_check.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time

os.environ["FREEZE_ENABLED"] = "1"      # this suite exercises freezing

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


tmpdir = tempfile.mkdtemp(prefix="freeze_bg_")
eng = sqlalchemy.create_engine(
    f"sqlite:///{os.path.join(tmpdir, 't.db')}",
    connect_args={"check_same_thread": False})

from flask_app.services import portfolio_snapshot_freeze as F      # noqa: E402
from flask_app.services import portfolio_snapshot_persistence as P  # noqa: E402
from flask_app.services import freeze_batch as FB                  # noqa: E402
from flask_app.services import data_service as DS                  # noqa: E402

F._engine = lambda: eng
F._is_postgres = lambda: False
F._data_version = lambda: "build=test"
FB._engine = lambda: eng
FB._is_postgres = lambda: False
DS.get_data = lambda *a, **kw: {}
P.load_page = lambda i, q: {"comments": [], "footnotes": [], "values": []}

Q = "2026-Q3"

#: Who holds what. Deliberately overlapping: D1 is held by everyone, so a batch
#: that builds per investor builds it five times and a batch that dedups builds
#: it once. A population where nobody shares a deal could not tell them apart.
HOLDINGS = {
    "INV1": ["D1", "D2"],
    "INV2": ["D1", "D2", "D3"],
    "INV3": ["D1", "D3"],
    "INV4": ["D1", "D4"],
    "INV5": ["D1", "D2", "D4"],
}
CODES = list(HOLDINGS)
DISTINCT = sorted({v for vs in HOLDINGS.values() for v in vs})
PAIRS = sum(len(v) for v in HOLDINGS.values())

BUILDS = {"n": 0, "per_deal": {}}


def _stub_report(investor, quarter, **kw):
    return {"subtabs": {"financial": {"groups": {"G": {"deals": [
        {"vcode": v, "name": v} for v in HOLDINGS[investor]]}}}},
        "errors": {}, "resolution": {"investor_name": investor}}


def _stub_builder(vcode, quarter):
    # SLOW ON PURPOSE, so section D has something to be responsive during and
    # so a per-investor run is distinguishable from a deduped one.
    BUILDS["n"] += 1
    BUILDS["per_deal"][vcode] = BUILDS["per_deal"].get(vcode, 0) + 1
    time.sleep(0.05)
    return {"vcode": vcode, "quarter": quarter, "value": 123.45}


F.assemble_full_report = _stub_report
F._default_one_pager_getter = lambda: _stub_builder

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
S._investor_list = lambda data=None: [{"code": c, "name": c} for c in CODES]
cli = app.test_client()
OPS = "/api/portfolio-snapshot/freeze-all/one-pagers"


def wait_for(job_id, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        j = FB.get_job(job_id) or {}
        if j.get("status") and j["status"] != "running":
            return j
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never finished")


# ── A. each deal is built once per batch ───────────────────────────────────
print("A. each distinct deal is built ONCE for the whole batch")
BUILDS["n"] = 0
BUILDS["per_deal"] = {}
r = cli.post(OPS, json={"quarter": Q}, headers=ADMIN)
chk("the button returns at once with a job id", r.status_code == 202,
    f"{r.status_code} {str(r.get_json())[:100]}")
job = wait_for((r.get_json() or {})["id"])
chk("the job finished", job["status"] == "done", str(job)[:120])
chk("every investor was frozen", job["frozen"] == len(CODES), str(job["frozen"]))
chk(f"the builder ran once per DISTINCT deal ({len(DISTINCT)}), not per holding "
    f"({PAIRS})", BUILDS["n"] == len(DISTINCT),
    f"{BUILDS['n']} builds, expected {len(DISTINCT)}")
chk("...and no deal was built twice",
    all(v == 1 for v in BUILDS["per_deal"].values()), str(BUILDS["per_deal"]))
# THE PAIRED DIRECTION: "built once" is also satisfied by building nothing.
chk("...while every holding was still served",
    (job["stats"] or {}).get("deal_reuses") == PAIRS - len(DISTINCT),
    str(job.get("stats")))
chk("the job reports the saving", (job["stats"] or {}).get("deal_builds")
    == len(DISTINCT), str(job.get("stats")))

# ── B. identical to the per-investor build ─────────────────────────────────
print("\nB. the deduped result is IDENTICAL to the per-investor build")
import json                                                        # noqa: E402


def _content(inv, quarter):
    """The stored One Pagers with the QUARTER stripped out.

    The two runs necessarily use different quarters — the second has to go
    somewhere the first has not already frozen — and the payload legitimately
    carries its own quarter, so that one field differs by construction. Every
    other byte must match, and that is what is compared. Stripping more than
    this would be hiding the answer.
    """
    blob = F.get_frozen(inv, quarter)["one_pagers"]
    return json.dumps({vc: {k: v for k, v in (p or {}).items() if k != "quarter"}
                       for vc, p in (blob or {}).items()}, sort_keys=True)


deduped = {c: _content(c, Q) for c in CODES}
# Re-run with the cache DISABLED, into a different quarter, and compare.
Q2 = "2026-Q4"
_real_getter = FB.FreezeBatch.one_pager_getter


def _no_cache(self, vcode, quarter):
    return self._full_op_builder(vcode, quarter) if self._full_op_builder else None


def _no_cache_init(self, vcode, quarter):
    from flask_app.services.portfolio_snapshot_freeze import (
        _default_one_pager_getter)
    if self._full_op_builder is None:
        self._full_op_builder = _default_one_pager_getter()
    return self._full_op_builder(vcode, quarter)


FB.FreezeBatch.one_pager_getter = _no_cache_init
BUILDS["n"] = 0
r2 = cli.post(OPS, json={"quarter": Q2}, headers=ADMIN)
job2 = wait_for((r2.get_json() or {})["id"])
FB.FreezeBatch.one_pager_getter = _real_getter
per_investor = {c: _content(c, Q2) for c in CODES}
chk("the uncached run really did build per holding", BUILDS["n"] == PAIRS,
    f"{BUILDS['n']} builds, expected {PAIRS}")
chk("every investor's stored One Pagers are byte-identical either way",
    deduped == per_investor,
    str([c for c in CODES if deduped[c] != per_investor[c]]))
chk("...and that is a real comparison, not two empties",
    all(len(v) > 20 for v in deduped.values()))

# ── C. skip-if-frozen still holds ──────────────────────────────────────────
print("\nC. an already-frozen investor is skipped and left untouched")
before = {c: (F._current_row(c, Q).get("version"),
              str(F._current_row(c, Q).get("one_pagers_frozen_at")))
          for c in CODES}
BUILDS["n"] = 0
r3 = cli.post(OPS, json={"quarter": Q}, headers=ADMIN)
job3 = wait_for((r3.get_json() or {})["id"])
chk("all reported skipped", job3["skipped"] == len(CODES), str(job3["skipped"]))
chk("nothing froze", job3["frozen"] == 0, str(job3["frozen"]))
after = {c: (F._current_row(c, Q).get("version"),
             str(F._current_row(c, Q).get("one_pagers_frozen_at")))
         for c in CODES}
chk("the stored rows did not move", before == after, f"{before} != {after}")
chk("...and nothing was even built for them", BUILDS["n"] == 0, str(BUILDS["n"]))

# ── D. the app stays responsive while a job runs ───────────────────────────
print("\nD. a request issued WHILE a job runs completes quickly")
Q3 = "2027-Q1"
t0 = time.time()
r4 = cli.post(OPS, json={"quarter": Q3}, headers=ADMIN)
post_ms = (time.time() - t0) * 1000
chk("the POST itself returns immediately (< 500ms)", post_ms < 500,
    f"{post_ms:.0f}ms")
jid = (r4.get_json() or {})["id"]
# While it runs, ask for something else through the same app.
lat = []
while (FB.get_job(jid) or {}).get("status") == "running" and len(lat) < 12:
    t1 = time.time()
    rr = cli.get("/api/portfolio-snapshot/freeze-job/active", headers=ADMIN)
    lat.append((time.time() - t1) * 1000)
    chk_ok = rr.status_code == 200
    if not chk_ok:
        break
    time.sleep(0.01)
job4 = wait_for(jid)
chk("concurrent requests were served while the job ran", len(lat) > 0,
    "the job finished before any concurrent request could be made")
worst = max(lat) if lat else 0
chk(f"...and none was slow (worst {worst:.0f}ms < 1000ms)", worst < 1000,
    f"{[round(x) for x in lat]}")
chk("the job still completed", job4["status"] == "done", str(job4)[:100])

# ── E. only one job at a time ──────────────────────────────────────────────
print("\nE. a second job is refused while one is running")
Q4 = "2027-Q2"
r5 = cli.post(OPS, json={"quarter": Q4}, headers=ADMIN)
chk("the first is accepted", r5.status_code == 202, str(r5.status_code))
r6 = cli.post(OPS, json={"quarter": Q4}, headers=ADMIN)
b6 = r6.get_json() or {}
chk("the second is refused with 409", r6.status_code == 409, str(r6.status_code))
chk("...naming the job already running", "already running" in str(b6.get("error", "")),
    str(b6)[:120])
chk("...and flagged so the screen can say so", b6.get("job_running") is True)
job5 = wait_for((r5.get_json() or {})["id"])
chk("the first job was unaffected", job5["status"] == "done", str(job5)[:90])
# THE PAIRED DIRECTION: once it is finished, a new job IS accepted.
r7 = cli.post(OPS, json={"quarter": "2027-Q3"}, headers=ADMIN)
chk("a new job is accepted once the first finished", r7.status_code == 202,
    str(r7.status_code))
wait_for((r7.get_json() or {})["id"])

# ── F. an interrupted job is recorded ──────────────────────────────────────
print("\nF. a job left running by a dead worker is marked interrupted")
with eng.begin() as cx:
    cx.execute(sqlalchemy.text(
        f"INSERT INTO {FB._TABLE} (quarter, part, status, worker_id, total, "
        f"done, frozen, skipped, failed) "
        f"VALUES ('2025-Q1','one_pagers','running','a-dead-worker',10,4,4,0,0)"))
    stale_id = cx.execute(sqlalchemy.text(
        f"SELECT MAX(id) FROM {FB._TABLE}")).scalar()
n = FB.reap_stale()
stale = FB.get_job(stale_id)
chk("the stale job is reaped", n >= 1, str(n))
chk("...and marked interrupted, not done or failed",
    stale["status"] == "interrupted", str(stale["status"]))
chk("...saying the worker restarted", "restart" in (stale.get("message") or ""),
    str(stale.get("message"))[:90])
chk("...keeping the progress it had reached", stale["done"] == 4, str(stale["done"]))
chk("...and what it froze is still frozen", F.is_frozen(CODES[0], Q, "one_pagers"))
# A job belonging to THIS worker is not touched — otherwise a reap during a run
# would kill the live job.
with eng.begin() as cx:
    cx.execute(sqlalchemy.text(
        f"INSERT INTO {FB._TABLE} (quarter, part, status, worker_id, total) "
        f"VALUES ('2025-Q2','one_pagers','running',:w,3)"), {"w": FB.WORKER_ID})
    mine = cx.execute(sqlalchemy.text(f"SELECT MAX(id) FROM {FB._TABLE}")).scalar()
FB.reap_stale()
chk("a job belonging to THIS worker is left alone",
    (FB.get_job(mine) or {})["status"] == "running",
    str((FB.get_job(mine) or {}).get("status")))

print(f"\n{'=' * 60}\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
