"""Accounting workpaper packages — the quarterly close, in the app.

WHAT THIS REPLACES. Accounting produces one Excel workbook per reportable
entity per quarter (the PPI Eastchase 06.30.2026 package is the worked
example: 37 tabs). Today every data tab is a Spreadsheet Server query or an
Excel PivotTable over one, and the financial statements are driven by an
account-to-FS-line tagging column maintained inside each workbook. This module
produces the same package from the app's own MRI tables, and tracks who has to
do what by when.

THE PACKAGE DECOMPOSES INTO THREE KINDS OF TAB, and only the first is free:

  QUERY   Trial Balance, GL Detail, Investor Detail, Investment Detail,
          IA Rollforward, Commitment Rollforward, and the dozen pivots over
          them (Cash, Intercompany, Accruals, Expenses, Contributions, ...).
          Sourced from gl_detail / ia_transactions / commitments.
  DERIVED Cover, Index, Balance Sheet, Schedule of Investments, Income
          Statement, Members' Capital, Cash Flow -- the trial balance folded
          through the FS mapping (see wp_fs_map).
  EXHIBIT Valuation Support, Tax Pricing, Cap Call Support, YTD Cash Support,
          Capital Rec, Org Chart. These are NOT derivable: they are the
          accountant's own evidence, uploaded and placed in the workbook.

WHO IS IN SCOPE. The population is not a judgement call: an entity needs a
package when accounting tags it ENTGRPID='REP' in MRI's ENTITYGRPD, which the
app loads into entity_groups. Two entities carry it today (AMB6, PPIECH).
sync_packages() creates a package per REP entity per cycle and never deletes
one, so an entity dropped from REP mid-quarter keeps the work already done.

STATUS IS A CHAIN, NOT A FLAG. not_started -> in_progress -> submitted ->
manager_approved -> cfo_approved, with a return-to-preparer from either review
step that REQUIRES a note. Every transition is written to wp_events; the
tracker reads state, the audit trail explains it.

DEADLINES BELONG TO THE CYCLE, PROGRESS TO THE PACKAGE. The CFO sets one due
date per step for the whole close (wp_cycle_steps); each package records its
own completion of those steps (wp_package_steps). That is what lets the
tracker say "12 of 19 steps done, 2 overdue" without a due date being copied
onto every row and drifting.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from flask_app.db import get_engine

logger = logging.getLogger(__name__)

# ── The closing checklist ────────────────────────────────────────────────
# Read off the example package's own tabs, in the order an accountant works
# them: source data first, then the reconciliations each tab supports, then
# the statements, then the sign-offs. The CFO will revise these -- the labels
# and order live here so that revision is one edit, not a migration.
STEP_TEMPLATE: List[Dict[str, Any]] = [
    {"key": "tb_load",          "label": "Trial balance loaded and tied",      "owner": "accountant"},
    {"key": "gl_review",        "label": "GL detail reviewed",                 "owner": "accountant"},
    {"key": "cash_rec",         "label": "Cash reconciled to bank",            "owner": "accountant"},
    {"key": "intercompany",     "label": "Intercompany and receivables",       "owner": "accountant"},
    {"key": "accruals",         "label": "Accrued expenses and management fee","owner": "accountant"},
    {"key": "investments",      "label": "Investment rollforward and valuation","owner": "accountant"},
    {"key": "capital_activity", "label": "Capital activity and commitments",   "owner": "accountant"},
    {"key": "exhibits",         "label": "Supporting exhibits attached",       "owner": "accountant"},
    {"key": "fs_draft",         "label": "Financial statements drafted",       "owner": "accountant"},
    {"key": "preparer_signoff", "label": "Preparer sign-off",                  "owner": "accountant"},
    {"key": "manager_review",   "label": "Accounting manager review",          "owner": "manager"},
    {"key": "cfo_approval",     "label": "CFO approval",                       "owner": "cfo"},
]
STEP_KEYS = [s["key"] for s in STEP_TEMPLATE]

# ── Exhibit slots ────────────────────────────────────────────────────────
# Named places in the workbook where an accountant's own evidence belongs,
# taken from the example package's non-derivable tabs. A slot is a promise
# about WHERE a file lands in the download, which is why they are declared
# rather than free-form: "upload anything, we will put it somewhere" is how a
# package stops being presentable to an auditor.
EXHIBIT_SLOTS: List[Dict[str, str]] = [
    {"key": "valuation_support", "label": "Valuation Support"},
    {"key": "cash_support",      "label": "YTD Cash Support (bank reports)"},
    {"key": "capital_rec",       "label": "Capital Reconciliation"},
    {"key": "cap_call_support",  "label": "Capital Call Support"},
    {"key": "tax_pricing",       "label": "Tax Pricing"},
    {"key": "fee_allocation",    "label": "Investment Cafe Fee Allocation"},
    {"key": "org_chart",         "label": "Org Chart"},
    {"key": "other",             "label": "Other supporting material"},
]
EXHIBIT_SLOT_KEYS = [s["key"] for s in EXHIBIT_SLOTS]

# ── Workflow ─────────────────────────────────────────────────────────────
STATE_ORDER = ["not_started", "in_progress", "submitted", "manager_approved", "cfo_approved"]
STATE_LABELS = {
    "not_started":      "Not started",
    "in_progress":      "In preparation",
    "submitted":        "Submitted for review",
    "manager_approved": "Manager approved",
    "cfo_approved":     "CFO approved",
    "returned":         "Returned to preparer",
}

_DDL = [
    """
    CREATE TABLE IF NOT EXISTS wp_cycles (
        id            {pk},
        period_label  TEXT NOT NULL,
        period_end    TEXT NOT NULL,
        status        TEXT DEFAULT 'open',
        created_by    TEXT,
        created_at    TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS wp_cycle_steps (
        id          {pk},
        cycle_id    INTEGER NOT NULL,
        step_key    TEXT NOT NULL,
        due_date    TEXT,
        updated_by  TEXT,
        updated_at  TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS wp_packages (
        id           {pk},
        cycle_id     INTEGER NOT NULL,
        entityid     TEXT NOT NULL,
        entity_name  TEXT,
        state        TEXT DEFAULT 'not_started',
        preparer     TEXT,
        reviewer     TEXT,
        notes        TEXT,
        created_at   TEXT,
        updated_at   TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS wp_package_steps (
        id            {pk},
        package_id    INTEGER NOT NULL,
        step_key      TEXT NOT NULL,
        done          INTEGER DEFAULT 0,
        completed_by  TEXT,
        completed_at  TEXT,
        note          TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS wp_exhibits (
        id            {pk},
        package_id    INTEGER NOT NULL,
        slot_key      TEXT NOT NULL,
        filename      TEXT NOT NULL,
        content_type  TEXT,
        size_bytes    INTEGER,
        caption       TEXT,
        uploaded_by   TEXT,
        uploaded_at   TEXT,
        content       {blob}
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS wp_events (
        id          {pk},
        package_id  INTEGER NOT NULL,
        action      TEXT NOT NULL,
        from_state  TEXT,
        to_state    TEXT,
        actor       TEXT,
        note        TEXT,
        created_at  TEXT
    )
    """,
    # Account -> financial-statement line. In the example package this is a
    # column typed into the Trial Balance tab of every workbook, 192 accounts
    # deep; here it is maintained once and reused by every entity and period.
    """
    CREATE TABLE IF NOT EXISTS wp_fs_map (
        id          {pk},
        acctnum     TEXT NOT NULL,
        statement   TEXT,
        fs_line     TEXT,
        cf_category TEXT,
        sort_order  INTEGER DEFAULT 0,
        updated_by  TEXT,
        updated_at  TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS wp_roles (
        id        {pk},
        username  TEXT NOT NULL,
        wp_role   TEXT NOT NULL
    )
    """,
]


def ensure_tables(engine=None):
    """Create the workpaper tables if missing. Idempotent, both dialects."""
    engine = engine or get_engine()
    is_pg = engine.dialect.name == "postgresql"
    pk = "SERIAL PRIMARY KEY" if is_pg else "INTEGER PRIMARY KEY AUTOINCREMENT"
    blob = "BYTEA" if is_pg else "BLOB"
    with engine.begin() as conn:
        for ddl in _DDL:
            conn.execute(text(ddl.format(pk=pk, blob=blob)))

        # CREATE TABLE IF NOT EXISTS never alters an existing table, so a
        # column added after a database was first built has to be added here
        # or it is simply missing on every database but a fresh one.
        for table, column, coltype in (("wp_fs_map", "cf_category", "TEXT"),):
            try:
                conn.execute(text(
                    "SELECT %s FROM %s LIMIT 1" % (column, table)))
            except Exception:
                logger.info("adding %s.%s", table, column)
                conn.execute(text(
                    "ALTER TABLE %s ADD COLUMN %s %s" % (table, column, coltype)))


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


# ── Roles ────────────────────────────────────────────────────────────────

def get_roles(engine=None) -> Dict[str, List[str]]:
    engine = engine or get_engine()
    ensure_tables(engine)
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT username, wp_role FROM wp_roles")).fetchall()
    out: Dict[str, List[str]] = {}
    for u, r in rows:
        out.setdefault(r, []).append(u)
    return out


def user_can(username: str, role: str, app_role: str = "", engine=None) -> bool:
    """Is this user allowed to act as `role`?

    An admin can act as any role. That is deliberate for a mock nobody has
    been assigned in yet -- an approval chain that cannot be exercised cannot
    be reviewed by the CFO either. Assign real people in wp_roles and the
    chain tightens on its own.
    """
    if app_role == "admin":
        return True
    return username in get_roles(engine).get(role, [])


# ── Cycles and deadlines ─────────────────────────────────────────────────

def list_cycles(engine=None) -> List[dict]:
    engine = engine or get_engine()
    ensure_tables(engine)
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT id, period_label, period_end, status, created_by, created_at "
            "FROM wp_cycles ORDER BY period_end DESC, id DESC")).mappings().all()
    return [dict(r) for r in rows]


def create_cycle(period_label: str, period_end: str, username: str, engine=None) -> dict:
    engine = engine or get_engine()
    ensure_tables(engine)
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO wp_cycles (period_label, period_end, status, created_by, created_at) "
            "VALUES (:l, :e, 'open', :u, :n)"),
            {"l": period_label, "e": period_end, "u": username, "n": _now()})
        cid = conn.execute(text(
            "SELECT id FROM wp_cycles WHERE period_label = :l AND period_end = :e "
            "ORDER BY id DESC"), {"l": period_label, "e": period_end}).scalar()
        # Every step exists for the cycle from the start, with no due date.
        # A missing row and "no deadline set" would otherwise be the same
        # thing, and the CFO could not tell which steps they had not got to.
        for s in STEP_TEMPLATE:
            conn.execute(text(
                "INSERT INTO wp_cycle_steps (cycle_id, step_key, due_date) "
                "VALUES (:c, :k, NULL)"), {"c": cid, "k": s["key"]})
    sync_packages(cid, engine=engine)
    return {"id": cid, "period_label": period_label, "period_end": period_end}


def validate_due_date(cycle_id: int, step_key: str, due_date: Optional[str],
                      engine=None) -> List[str]:
    """Check a proposed deadline. Raises on impossible; returns warnings.

    REJECT WHAT CANNOT BE TRUE, WARN WHAT IS MERELY ODD. A date the CFO
    cannot have meant is worse than no date: it renders overdue in red on
    every package from the moment it is typed, and the grid stops meaning
    anything. But a CFO who wants an unusual deadline is allowed to have one
    -- the app is not the authority on how long a close takes.

    Impossible, so refused:
      * not a date at all
      * before the period it closes had ended -- the case that prompted this,
        a deadline of 2020-01-01 sitting on a period ended 2026-06-30
        (Sep 14 2026)

    Odd, so reported and saved anyway:
      * more than a year after the period end
      * out of sequence against the deadlines already set on other steps
    """
    if not due_date:
        return []                      # clearing a deadline is always allowed

    try:
        due = date.fromisoformat(str(due_date).strip())
    except ValueError:
        raise ValueError(f"'{due_date}' is not a date (expected YYYY-MM-DD)")

    engine = engine or get_engine()
    with engine.connect() as conn:
        end_raw = conn.execute(text(
            "SELECT period_end FROM wp_cycles WHERE id = :c"),
            {"c": cycle_id}).scalar()
    if end_raw is None:
        raise ValueError(f"No close cycle {cycle_id}")

    try:
        period_end = date.fromisoformat(str(end_raw)[:10])
    except ValueError:
        # A cycle whose own period_end is unreadable cannot anchor anything.
        # Do not invent an anchor; let the deadline through unchecked.
        return []

    if due < period_end:
        raise ValueError(
            f"{due.isoformat()} is before the period it closes "
            f"({period_end.isoformat()}). A close step cannot be due before "
            f"the period has ended.")

    warnings: List[str] = []
    if (due - period_end).days > 365:
        warnings.append(
            f"{due.isoformat()} is {(due - period_end).days} days after the "
            f"period end — check this is not a typo.")

    order = {s["key"]: i for i, s in enumerate(STEP_TEMPLATE)}
    for other in cycle_steps(cycle_id, engine=engine):
        if other["key"] == step_key or not other.get("due_date"):
            continue
        try:
            other_due = date.fromisoformat(str(other["due_date"])[:10])
        except ValueError:
            continue
        earlier_step = order[step_key] < order[other["key"]]
        if earlier_step and due > other_due:
            warnings.append(
                f"Due after '{other['label']}' ({other_due.isoformat()}), "
                f"which comes later in the close.")
        elif not earlier_step and due < other_due:
            warnings.append(
                f"Due before '{other['label']}' ({other_due.isoformat()}), "
                f"which comes earlier in the close.")
    return warnings


def set_step_due_date(cycle_id: int, step_key: str, due_date: Optional[str],
                      username: str, engine=None) -> dict:
    if step_key not in STEP_KEYS:
        raise ValueError(f"Unknown step '{step_key}'")
    engine = engine or get_engine()
    warnings = validate_due_date(cycle_id, step_key, due_date, engine=engine)
    with engine.begin() as conn:
        updated = conn.execute(text(
            "UPDATE wp_cycle_steps SET due_date = :d, updated_by = :u, updated_at = :n "
            "WHERE cycle_id = :c AND step_key = :k"),
            {"d": due_date or None, "u": username, "n": _now(),
             "c": cycle_id, "k": step_key}).rowcount
        if not updated:
            conn.execute(text(
                "INSERT INTO wp_cycle_steps (cycle_id, step_key, due_date, updated_by, updated_at) "
                "VALUES (:c, :k, :d, :u, :n)"),
                {"c": cycle_id, "k": step_key, "d": due_date or None,
                 "u": username, "n": _now()})
    return {"status": "ok", "warnings": warnings}


def cycle_steps(cycle_id: int, engine=None) -> List[dict]:
    engine = engine or get_engine()
    ensure_tables(engine)
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT step_key, due_date FROM wp_cycle_steps WHERE cycle_id = :c"),
            {"c": cycle_id}).fetchall()
    due = {k: d for k, d in rows}
    return [{**s, "due_date": due.get(s["key"])} for s in STEP_TEMPLATE]


# ── Packages ─────────────────────────────────────────────────────────────

def rep_entities(engine=None) -> List[dict]:
    """The entities accounting has tagged REP, with names where known."""
    engine = engine or get_engine()
    with engine.connect() as conn:
        try:
            rows = conn.execute(text(
                "SELECT g.\"ENTITYID\" AS entityid, e.\"NAME\" AS name "
                "FROM entity_groups g "
                "LEFT JOIN entities e ON e.\"ENTITYID\" = g.\"ENTITYID\" "
                "WHERE UPPER(TRIM(g.\"ENTGRPID\")) = 'REP' "
                "ORDER BY g.\"ENTITYID\"")).mappings().all()
        except Exception:
            logger.warning("entity_groups/entities not loaded yet", exc_info=True)
            return []
    return [dict(r) for r in rows]


def sync_packages(cycle_id: int, engine=None) -> dict:
    """Create a package for each REP entity that has none in this cycle.

    Never deletes. An entity dropped from REP mid-quarter keeps whatever work
    has already been done on it -- losing a half-finished package because a
    tag changed in MRI is not a trade anyone would make.
    """
    engine = engine or get_engine()
    ensure_tables(engine)
    entities = rep_entities(engine)
    created = 0
    with engine.begin() as conn:
        have = {r[0] for r in conn.execute(text(
            "SELECT entityid FROM wp_packages WHERE cycle_id = :c"), {"c": cycle_id}).fetchall()}
        for ent in entities:
            if ent["entityid"] in have:
                continue
            conn.execute(text(
                "INSERT INTO wp_packages (cycle_id, entityid, entity_name, state, created_at, updated_at) "
                "VALUES (:c, :e, :n, 'not_started', :t, :t)"),
                {"c": cycle_id, "e": ent["entityid"], "n": ent.get("name"), "t": _now()})
            created += 1
    return {"created": created, "rep_entities": len(entities)}


def _package_row(conn, package_id: int) -> Optional[dict]:
    row = conn.execute(text(
        "SELECT p.*, c.period_label, c.period_end FROM wp_packages p "
        "JOIN wp_cycles c ON c.id = p.cycle_id WHERE p.id = :i"),
        {"i": package_id}).mappings().first()
    return dict(row) if row else None


def tracker(cycle_id: int, engine=None) -> dict:
    """Everything the tracker grid needs in one call.

    One query per table rather than per package: the grid is entities x steps
    and a per-row round trip would make it quadratic for no reason.
    """
    engine = engine or get_engine()
    ensure_tables(engine)
    steps = cycle_steps(cycle_id, engine)
    with engine.connect() as conn:
        packages = [dict(r) for r in conn.execute(text(
            "SELECT * FROM wp_packages WHERE cycle_id = :c ORDER BY entityid"),
            {"c": cycle_id}).mappings().all()]
        ids = [p["id"] for p in packages]
        done: Dict[int, Dict[str, dict]] = {i: {} for i in ids}
        exhibits: Dict[int, int] = {i: 0 for i in ids}
        if ids:
            marks = conn.execute(text(
                "SELECT package_id, step_key, done, completed_by, completed_at "
                "FROM wp_package_steps WHERE package_id IN :ids"
            ).bindparams(__import__("sqlalchemy").bindparam("ids", expanding=True)),
                {"ids": ids}).mappings().all()
            for m in marks:
                done[m["package_id"]][m["step_key"]] = dict(m)
            for pid, n in conn.execute(text(
                "SELECT package_id, COUNT(*) FROM wp_exhibits WHERE package_id IN :ids "
                "GROUP BY package_id"
            ).bindparams(__import__("sqlalchemy").bindparam("ids", expanding=True)),
                    {"ids": ids}).fetchall():
                exhibits[pid] = n

    today = date.today().isoformat()
    for p in packages:
        marks = done.get(p["id"], {})
        p["steps"] = []
        overdue = 0
        complete = 0
        for s in steps:
            m = marks.get(s["key"], {})
            is_done = bool(m.get("done"))
            # Overdue is only meaningful for work not yet done. A step
            # finished after its deadline is late history, not a live alarm.
            late = bool(s["due_date"] and not is_done and s["due_date"] < today)
            if is_done:
                complete += 1
            if late:
                overdue += 1
            p["steps"].append({
                "key": s["key"], "label": s["label"], "owner": s["owner"],
                "due_date": s["due_date"], "done": is_done,
                "completed_by": m.get("completed_by"),
                "completed_at": m.get("completed_at"), "overdue": late,
            })
        p["steps_complete"] = complete
        p["steps_total"] = len(steps)
        p["overdue_count"] = overdue
        p["exhibit_count"] = exhibits.get(p["id"], 0)
        p["state_label"] = STATE_LABELS.get(p["state"], p["state"])
    return {"cycle_id": cycle_id, "steps": steps, "packages": packages}


def set_step_done(package_id: int, step_key: str, done: bool, username: str,
                  note: str = "", engine=None) -> dict:
    if step_key not in STEP_KEYS:
        raise ValueError(f"Unknown step '{step_key}'")
    engine = engine or get_engine()
    with engine.begin() as conn:
        n = conn.execute(text(
            "UPDATE wp_package_steps SET done = :d, completed_by = :u, completed_at = :t, note = :note "
            "WHERE package_id = :p AND step_key = :k"),
            {"d": 1 if done else 0, "u": username if done else None,
             "t": _now() if done else None, "note": note or None,
             "p": package_id, "k": step_key}).rowcount
        if not n:
            conn.execute(text(
                "INSERT INTO wp_package_steps (package_id, step_key, done, completed_by, completed_at, note) "
                "VALUES (:p, :k, :d, :u, :t, :note)"),
                {"p": package_id, "k": step_key, "d": 1 if done else 0,
                 "u": username if done else None, "t": _now() if done else None,
                 "note": note or None})
        # Ticking the first step is what takes a package off "not started";
        # nobody should have to remember to flip a status as well.
        conn.execute(text(
            "UPDATE wp_packages SET state = 'in_progress', updated_at = :t "
            "WHERE id = :p AND state IN ('not_started', 'returned')"),
            {"p": package_id, "t": _now()})
    return {"status": "ok"}


def _log(conn, package_id, action, frm, to, actor, note):
    conn.execute(text(
        "INSERT INTO wp_events (package_id, action, from_state, to_state, actor, note, created_at) "
        "VALUES (:p, :a, :f, :t, :u, :n, :c)"),
        {"p": package_id, "a": action, "f": frm, "t": to, "u": actor,
         "n": note or None, "c": _now()})


def transition(package_id: int, action: str, username: str, app_role: str,
               note: str = "", engine=None) -> dict:
    """submit | approve_manager | approve_cfo | return_to_preparer | reopen."""
    engine = engine or get_engine()
    with engine.begin() as conn:
        pkg = _package_row(conn, package_id)
        if not pkg:
            raise ValueError("Package not found")
        state = pkg["state"]

        if action == "submit":
            if state not in ("not_started", "in_progress", "returned"):
                raise ValueError(f"Cannot submit from '{state}'")
            new = "submitted"
        elif action == "approve_manager":
            if state != "submitted":
                raise ValueError("Only a submitted package can be approved by the manager")
            if not user_can(username, "manager", app_role, engine):
                raise PermissionError("Not an accounting manager")
            new = "manager_approved"
        elif action == "approve_cfo":
            if state != "manager_approved":
                raise ValueError("The accounting manager must approve first")
            if not user_can(username, "cfo", app_role, engine):
                raise PermissionError("Not the CFO")
            new = "cfo_approved"
        elif action == "return_to_preparer":
            if state not in ("submitted", "manager_approved"):
                raise ValueError(f"Nothing to return from '{state}'")
            # A return without a reason is the preparer's problem to guess.
            if not (note or "").strip():
                raise ValueError("A note is required when returning a package")
            new = "returned"
        elif action == "reopen":
            if state != "cfo_approved":
                raise ValueError("Only an approved package is reopened")
            if app_role != "admin" and not user_can(username, "cfo", app_role, engine):
                raise PermissionError("Only the CFO reopens an approved package")
            new = "in_progress"
        else:
            raise ValueError(f"Unknown action '{action}'")

        conn.execute(text("UPDATE wp_packages SET state = :s, updated_at = :t WHERE id = :i"),
                     {"s": new, "t": _now(), "i": package_id})
        _log(conn, package_id, action, state, new, username, note)
    return {"status": "ok", "state": new, "state_label": STATE_LABELS.get(new, new)}


def package_detail(package_id: int, engine=None) -> dict:
    engine = engine or get_engine()
    ensure_tables(engine)
    with engine.connect() as conn:
        pkg = _package_row(conn, package_id)
        if not pkg:
            raise ValueError("Package not found")
        marks = {m["step_key"]: dict(m) for m in conn.execute(text(
            "SELECT * FROM wp_package_steps WHERE package_id = :p"),
            {"p": package_id}).mappings().all()}
        exhibits = [dict(r) for r in conn.execute(text(
            "SELECT id, slot_key, filename, content_type, size_bytes, caption, "
            "uploaded_by, uploaded_at FROM wp_exhibits WHERE package_id = :p "
            "ORDER BY slot_key, id"), {"p": package_id}).mappings().all()]
        events = [dict(r) for r in conn.execute(text(
            "SELECT action, from_state, to_state, actor, note, created_at "
            "FROM wp_events WHERE package_id = :p ORDER BY id DESC"),
            {"p": package_id}).mappings().all()]

    steps = cycle_steps(pkg["cycle_id"], engine)
    today = date.today().isoformat()
    out_steps = []
    for s in steps:
        m = marks.get(s["key"], {})
        is_done = bool(m.get("done"))
        out_steps.append({
            **s, "done": is_done, "note": m.get("note"),
            "completed_by": m.get("completed_by"), "completed_at": m.get("completed_at"),
            "overdue": bool(s["due_date"] and not is_done and s["due_date"] < today),
        })
    pkg["state_label"] = STATE_LABELS.get(pkg["state"], pkg["state"])
    return {"package": pkg, "steps": out_steps, "exhibits": exhibits,
            "events": events, "slots": EXHIBIT_SLOTS}


# ── Exhibits ─────────────────────────────────────────────────────────────

def add_exhibit(package_id: int, slot_key: str, filename: str, content: bytes,
                content_type: str, caption: str, username: str, engine=None) -> dict:
    if slot_key not in EXHIBIT_SLOT_KEYS:
        raise ValueError(f"Unknown exhibit slot '{slot_key}'")
    engine = engine or get_engine()
    ensure_tables(engine)
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO wp_exhibits (package_id, slot_key, filename, content_type, "
            "size_bytes, caption, uploaded_by, uploaded_at, content) "
            "VALUES (:p, :s, :f, :c, :z, :cap, :u, :t, :b)"),
            {"p": package_id, "s": slot_key, "f": filename, "c": content_type,
             "z": len(content), "cap": caption or None, "u": username,
             "t": _now(), "b": content})
        _log(conn, package_id, "exhibit_added", None, None, username,
             f"{slot_key}: {filename}")
    return {"status": "ok"}


def get_exhibit(exhibit_id: int, engine=None) -> Optional[dict]:
    engine = engine or get_engine()
    with engine.connect() as conn:
        row = conn.execute(text("SELECT * FROM wp_exhibits WHERE id = :i"),
                           {"i": exhibit_id}).mappings().first()
    return dict(row) if row else None


def delete_exhibit(exhibit_id: int, username: str, engine=None) -> dict:
    engine = engine or get_engine()
    with engine.begin() as conn:
        row = conn.execute(text("SELECT package_id, filename FROM wp_exhibits WHERE id = :i"),
                           {"i": exhibit_id}).first()
        conn.execute(text("DELETE FROM wp_exhibits WHERE id = :i"), {"i": exhibit_id})
        if row:
            _log(conn, row[0], "exhibit_removed", None, None, username, row[1])
    return {"status": "ok"}


def package_exhibits(package_id: int, engine=None) -> List[dict]:
    engine = engine or get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT * FROM wp_exhibits WHERE package_id = :p ORDER BY slot_key, id"),
            {"p": package_id}).mappings().all()
    return [dict(r) for r in rows]


# ── Account -> financial statement mapping ───────────────────────────────

def get_fs_map(engine=None) -> List[dict]:
    engine = engine or get_engine()
    ensure_tables(engine)
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT acctnum, statement, fs_line, cf_category, sort_order FROM wp_fs_map "
            "ORDER BY statement, sort_order, acctnum")).mappings().all()
    return [dict(r) for r in rows]


def set_fs_map(entries: List[dict], username: str, engine=None) -> dict:
    """Replace the mapping wholesale. It is one small edited table, not a feed."""
    engine = engine or get_engine()
    ensure_tables(engine)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM wp_fs_map"))
        for i, e in enumerate(entries):
            acct = str(e.get("acctnum") or "").strip()
            if not acct:
                continue
            conn.execute(text(
                "INSERT INTO wp_fs_map (acctnum, statement, fs_line, cf_category, "
                "sort_order, updated_by, updated_at) "
                "VALUES (:a, :s, :l, :c, :o, :u, :t)"),
                {"a": acct, "s": (e.get("statement") or "").strip() or None,
                 "l": (e.get("fs_line") or "").strip() or None,
                 "c": (e.get("cf_category") or "").strip() or None,
                 "o": e.get("sort_order", i), "u": username, "t": _now()})
    return {"status": "ok", "rows": len(entries)}
