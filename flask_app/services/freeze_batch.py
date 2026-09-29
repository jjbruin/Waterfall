"""Freezing a quarter for every investor: each deal built once, in the background.

TWO PROBLEMS, ONE MODULE.

**BUILD EACH DEAL ONCE.** A quarter's freeze covers ~130 investors holding a few
hundred distinct deals between them, and the same deal is held by many of them.
`assemble_full_report` created a FRESH provider pair per report, and the
providers memoise on ``(vcode, quarter)`` — no investor in the key — so the work
was already investor-independent and was simply being thrown away after each
report. A batch holds ONE provider pair and one full-One-Pager cache for the
whole run, so a deal held by forty investors is computed once and reused
thirty-nine times. Per-investor work — the roster, the ownership resolution,
the investor-share columns — is still done per investor, because it genuinely
differs.

**DO NOT BLOCK THE REQUEST WORKER.** The batch used to run inside the POST that
started it. On Sep 29 2026 that took the app down for about 35 minutes: one
gunicorn worker, one request, everything else queued behind it.

WHY A THREAD AND NOT A CONTAINER APPS JOB. The expensive thing is not the
freeze, it is ``data_service.get_data()`` — the whole set of DataFrames the
computation reads. In-process that cache is already warm; a separate Container
Apps Job would start cold and load it again, for every run, on a 2 GB container
where that load is the memory high-water mark. A job would also need its own
image tag, revision and env wiring, and would still have to report progress
through the database, which is where it lives anyway. So: a daemon-free
``threading.Thread`` in the app process, with all state in the database.

WHAT A THREAD DOES AND DOES NOT BUY, stated honestly. Requests keep being
served: the freeze spends most of its time in pandas and SQLAlchemy, which
release the GIL, so the worker answers other requests throughout. It is not
free — CPU-bound stretches contend for the GIL and add latency. It is the
difference between "slower while a freeze runs" and "down while a freeze runs".

ONE JOB AT A TIME, and it is enforced in the DATABASE, not only in memory: a
second click is refused by looking for a running row, so the rule survives a
process that has forgotten its own in-memory lock.

A RESTART MARKS THE JOB INTERRUPTED, NOT RUNNING FOREVER. Each row records the
``worker_id`` of the process that started it; at startup, rows still marked
running under a DIFFERENT worker are marked ``interrupted``. Investors frozen
before the restart stay frozen — that is the point of per-investor commits.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import threading
import time
import uuid
from typing import Optional

from sqlalchemy import text

log = logging.getLogger(__name__)

_TABLE = "portfolio_snapshot_freeze_jobs"

#: Identifies THIS process. A row left "running" under another id is a job whose
#: worker died; see `reap_stale`.
WORKER_ID = uuid.uuid4().hex

STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_INTERRUPTED = "interrupted"

_lock = threading.Lock()
_schema_ready: set = set()


def _engine():
    from flask_app.db import get_engine
    return get_engine()


def _is_postgres() -> bool:
    try:
        return _engine().dialect.name == "postgresql"
    except Exception:
        return False


def ensure_job_table() -> None:
    """Create the job table. Once per process per engine — it is on a read path."""
    key = str(_engine().url)
    if key in _schema_ready:
        return
    pk = ("SERIAL PRIMARY KEY" if _is_postgres()
          else "INTEGER PRIMARY KEY AUTOINCREMENT")
    with _engine().begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {_TABLE} (
                id {pk},
                quarter TEXT NOT NULL,
                part TEXT NOT NULL,
                status TEXT NOT NULL,
                started_by TEXT,
                started_at TIMESTAMP,
                finished_at TIMESTAMP,
                worker_id TEXT,
                total INTEGER DEFAULT 0,
                done INTEGER DEFAULT 0,
                frozen INTEGER DEFAULT 0,
                skipped INTEGER DEFAULT 0,
                failed INTEGER DEFAULT 0,
                message TEXT,
                results TEXT,
                stats TEXT,
                -- Last time the job wrote progress. `reap_stale` uses it to
                -- tell a job whose worker DIED from one that is simply running
                -- in another worker: a dead job stops beating, a live one does
                -- not. Without it, a worker restarting under a multi-worker
                -- setup would reap its sibling's live job.
                heartbeat TIMESTAMP
            )
        """))
    # ADDED AFTER THE TABLE FIRST SHIPPED. `CREATE TABLE IF NOT EXISTS` never
    # touches a table that already exists, so a database that created this table
    # before `heartbeat` was added would raise "no such column" on the FIRST
    # progress write -- the thread dies, and the job sits at 0 of N looking
    # live. Found exactly that way by freeze_quarter_ui_check.
    for col, typ in (("heartbeat", "TIMESTAMP"),):
        try:
            with _engine().begin() as conn:
                conn.execute(text(
                    f"ALTER TABLE {_TABLE} ADD COLUMN {col} {typ}"))
        except Exception:
            pass          # already there; the CREATE above covers a new table
    _schema_ready.add(key)


def _now():
    import datetime as dt
    return dt.datetime.utcnow()


# ── the batch: one provider pair, one deal cache ───────────────────────────

class FreezeBatch:
    """Shared per-run computation. Build each deal once; reuse for every holder.

    HANDS OUT COPIES. The provider cache returns the SAME object to every
    caller, which was safe while a pair lived for one report and is not safe
    across investors: anything that mutated a One Pager would silently corrupt
    every other investor holding that deal. `one_pager_getter` therefore
    deep-copies on the way out. It is a few KB per deal and it removes a whole
    class of cross-investor bug — the wrong trade to optimise.
    """

    def __init__(self, quarter: str, data: Optional[dict] = None):
        from flask_app.services import data_service
        self.quarter = quarter
        self.data = data if data is not None else data_service.get_data()
        # BUILT ON FIRST USE, not here. Constructing a provider reads the shared
        # frames, and a One-Pagers-only freeze never touches the Snapshot pair —
        # so eager construction did work no caller had asked for. Each is still
        # built AT MOST ONCE and then shared by every investor in the run, which
        # is the dedup.
        self._snapshot_op = None
        self._snapshot_noi = None
        self._full_op_builder = None
        self._full_op_cache: dict = {}
        self.stats = {"deal_builds": 0, "deal_reuses": 0, "investors": 0}

    def _snapshot_providers(self):
        from flask_app.services.portfolio_snapshot_freeze import (
            _one_pager_provider, _quarterly_noi_provider)
        if self._snapshot_op is None:
            self._snapshot_op = _one_pager_provider(self.data)
            self._snapshot_noi = _quarterly_noi_provider(self.data)
        return self._snapshot_op, self._snapshot_noi

    def assembler(self, investor: str, quarter: str) -> dict:
        """The per-investor report, with the batch's shared providers.

        THE PROVIDERS ARE PASSED AS THUNKS, so the pair is built on the first
        deal actually asked for rather than on the first report started. An
        assembler that needs no provider — a caller that supplies its own
        assembly — then costs nothing, and construction still happens at most
        once for the whole run.
        """
        from flask_app.services.portfolio_snapshot_freeze import (
            assemble_full_report)

        def _op(vcode, q):
            return self._snapshot_providers()[0](vcode, q)

        def _noi(vcode, q):
            return self._snapshot_providers()[1](vcode, q)

        return assemble_full_report(
            investor, quarter, data=self.data,
            one_pager_provider=_op, quarterly_noi_provider=_noi)

    def one_pager_getter(self, vcode: str, quarter: str):
        from flask_app.services.portfolio_snapshot_freeze import (
            _default_one_pager_getter)
        if self._full_op_builder is None:
            self._full_op_builder = _default_one_pager_getter()
        key = (vcode, quarter)
        if key not in self._full_op_cache:
            self._full_op_cache[key] = self._full_op_builder(vcode, quarter)
            self.stats["deal_builds"] += 1
        else:
            self.stats["deal_reuses"] += 1
        return copy.deepcopy(self._full_op_cache[key])


# ── job rows ───────────────────────────────────────────────────────────────

def active_job() -> Optional[dict]:
    """The running job, or None. Read from the DATABASE, not from memory."""
    ensure_job_table()
    with _engine().connect() as conn:
        row = conn.execute(text(
            f"SELECT * FROM {_TABLE} WHERE status = :s "
            f"ORDER BY id DESC LIMIT 1"), {"s": STATUS_RUNNING}).mappings().fetchone()
    return _row(row) if row else None


def get_job(job_id: int) -> Optional[dict]:
    ensure_job_table()
    with _engine().connect() as conn:
        row = conn.execute(text(f"SELECT * FROM {_TABLE} WHERE id = :i"),
                           {"i": job_id}).mappings().fetchone()
    return _row(row) if row else None


def _row(row) -> dict:
    d = dict(row)
    for k in ("results", "stats"):
        try:
            d[k] = json.loads(d.get(k) or "null")
        except Exception:
            d[k] = None
    for k in ("started_at", "finished_at"):
        if d.get(k) is not None:
            d[k] = str(d[k])
    return d


def _configured_workers() -> int:
    try:
        return max(1, int(os.environ.get("GUNICORN_WORKERS") or 1))
    except Exception:
        return 1


def reap_stale(grace_seconds: Optional[float] = None) -> int:
    """Mark jobs left running by a DEAD process as interrupted. Returns how many.

    Called at startup. A job row still marked running under a worker that is not
    this one cannot be running *here* — but under a MULTI-WORKER setup it may be
    running perfectly well in a sibling, and reaping it would mark a live job
    interrupted. So the test is the HEARTBEAT, which `_progress` writes after
    every investor: a dead job stops beating, a live one does not.

    With ONE worker configured — which is what production runs — there is no
    sibling, so a foreign running row is provably dead and is reaped at once.
    With more than one, a grace window is required, and until it expires a new
    freeze is refused because the old one still looks live. That is the honest
    trade: 5 minutes of caution beats killing a running job.

    Investors frozen before the restart stay frozen; the freeze commits per
    investor.
    """
    if grace_seconds is None:
        grace_seconds = 0.0 if _configured_workers() <= 1 else 300.0
    ensure_job_table()
    import datetime as dt
    cutoff = _now() - dt.timedelta(seconds=grace_seconds)
    with _engine().begin() as conn:
        res = conn.execute(text(
            f"UPDATE {_TABLE} SET status = :new, finished_at = :now, "
            f"message = :msg "
            f"WHERE status = :run AND (worker_id IS NULL OR worker_id <> :wid) "
            f"  AND (heartbeat IS NULL OR heartbeat < :cutoff)"),
            {"new": STATUS_INTERRUPTED, "now": _now(), "run": STATUS_RUNNING,
             "wid": WORKER_ID, "cutoff": cutoff,
             "msg": "the worker restarted while this job was running; "
                    "investors frozen before the restart are still frozen"})
        n = res.rowcount or 0
    if n:
        log.warning("Marked %s freeze job(s) interrupted after a restart", n)
    return n


# â”€â”€ running one â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def start_job(app, quarter: str, part: str, codes: list, started_by: str):
    """Create the job row and the thread. Returns (job dict, error message).

    REFUSES A SECOND JOB, from the database. Two freezes at once would race on
    the same rows and neither progress count would describe anything.
    """
    ensure_job_table()
    with _lock:
        running = active_job()
        if running:
            return None, (
                f"A freeze is already running (job {running['id']}, "
                f"{running['quarter']} {running['part']}, "
                f"{running.get('done') or 0} of {running.get('total') or 0} "
                f"investors). Wait for it to finish.")
        with _engine().begin() as conn:
            conn.execute(text(
                f"INSERT INTO {_TABLE} (quarter, part, status, started_by, "
                f"started_at, worker_id, total, done, frozen, skipped, failed) "
                f"VALUES (:q, :p, :s, :by, :at, :wid, :tot, 0, 0, 0, 0)"),
                {"q": quarter, "p": part, "s": STATUS_RUNNING, "by": started_by,
                 "at": _now(), "wid": WORKER_ID, "tot": len(codes)})
            job_id = conn.execute(text(
                f"SELECT MAX(id) FROM {_TABLE}")).scalar()

    t = threading.Thread(target=_run, args=(app, job_id, quarter, part, codes,
                                            started_by),
                         name=f"freeze-{job_id}", daemon=True)
    t.start()
    return get_job(job_id), None


def _progress(job_id: int, **fields):
    # EVERY progress write is also a heartbeat. `reap_stale` uses it to tell a
    # job whose worker died from one running in a sibling worker.
    fields.setdefault("heartbeat", _now())
    sets = ", ".join(f"{k} = :{k}" for k in fields)
    with _engine().begin() as conn:
        conn.execute(text(f"UPDATE {_TABLE} SET {sets} WHERE id = :i"),
                     {**fields, "i": job_id})


def _run(app, job_id: int, quarter: str, part: str, codes: list, who: str):
    """The batch itself. Per investor: freeze, record, move on.

    PER-INVESTOR ISOLATION IS UNCHANGED from the synchronous version â€” one
    investor failing must not abort the rest, and each result row carries either
    a receipt or an error. What changed is only WHERE this runs.

    Progress is written after EVERY investor, not at the end, because a progress
    bar that only moves when the work is done is not a progress bar â€” and
    because a restart mid-run must leave a truthful count behind.
    """
    from flask_app.services import portfolio_snapshot_freeze as FZ
    results = []
    frozen = skipped = failed = 0
    with app.app_context():
        try:
            batch = FreezeBatch(quarter)
        except Exception as exc:                              # noqa: BLE001
            log.exception("freeze job %s could not load data", job_id)
            _progress(job_id, status=STATUS_FAILED, finished_at=_now(),
                      message=f"could not load data: {exc}")
            return
        for i, code in enumerate(codes, 1):
            row = {"investor": code, "part": part}
            try:
                if FZ.is_frozen(code, quarter, part):
                    row.update(skipped=True, frozen=False,
                               reason="already frozen for this part â€” Re-freeze "
                                      "it if it genuinely has to change")
                    skipped += 1
                else:
                    from flask_app.serializers import safe_json
                    receipt = FZ.freeze_part(
                        code, quarter, part, who,
                        assembler=batch.assembler,
                        one_pager_getter=batch.one_pager_getter)
                    row["receipt"] = safe_json(receipt)
                    row["frozen"] = True
                    errs = (row["receipt"] or {}).get("one_pager_errors")
                    if errs:
                        row["frozen"] = False
                        row["error"] = (f"{len(errs)} One Pager(s) could not be "
                                        f"built: " + "; ".join(errs[:3]))
                        failed += 1
                    else:
                        frozen += 1
            except Exception as exc:                          # noqa: BLE001
                log.exception("freeze job %s failed for %s", job_id, code)
                row.update(frozen=False,
                           error=f"{code} {quarter} is still live: {exc}")
                failed += 1
            results.append(row)
            batch.stats["investors"] = i
            _progress(job_id, done=i, frozen=frozen, skipped=skipped,
                      failed=failed, stats=json.dumps(batch.stats))
        _progress(job_id, status=STATUS_DONE, finished_at=_now(),
                  results=json.dumps(results), stats=json.dumps(batch.stats),
                  message=(f"{frozen} frozen, {skipped} already frozen, "
                           f"{failed} failed, of {len(codes)}"))
        log.info("freeze job %s finished: %s deal builds, %s reuses",
                 job_id, batch.stats["deal_builds"], batch.stats["deal_reuses"])
