"""The close TRACKER: who owes what, when it was due, and who signed it off.

WHAT THIS IS, AND WHAT IT IS NOT. ``workpaper_service`` owns a PACKAGE -- one
entity's workbook, its steps, its exhibits, its state machine. This module owns
the CFO's view ACROSS entities: the grid he keeps today in
``2Q26 - PSC Reporting Checklist & Calendar.xlsx``, tab "Reporting Calendar".
Fifty-five rows, one per reporting entity, and for each a target date, a
preparer's initials, and a set of sign-offs per deliverable.

They are deliberately separate tables. The workbench's ``wp_package_steps`` is a
checklist of things to DO inside one workbook; a tracker cell is a record of a
REVIEW HAVING HAPPENED, by a named person, on a named date. Folding the second
into the first would have made every sign-off a tick box and lost the "who",
which is the column the CFO actually reads.

THE FOUR DELIVERABLES, in the order the team signs them off (Jim, Sep 16 2026):
workpapers, then the financial statements, then the capital account statements,
and finally posting to Investment Cafe, the investor portal.

STAGES ARE PER DELIVERABLE, NOT A FIXED THREE. The first three each run
Prepared -> Manager -> CFO, the three-column group the spreadsheet repeats.
Investment Cafe does NOT: its columns are FS Posted, Reviewed, CAS Posted,
Reviewed, Released -- five, because the financial statements and the capital
account statements are posted and checked separately before the quarter is
released to investors. Forcing that into three columns would have discarded two
of them, so the stage list is data rather than a shape.

SEQUENCE IS REPORTED, NEVER ENFORCED. A capital account statement signed before
its workpapers is odd, not impossible -- a correction gets re-signed out of
order routinely. ``out_of_sequence`` names it and the entry still saves.
Refusing it would only move the record back into somebody's email, which is the
thing this screen exists to replace.

ORDER IS THE CFO'S TO SET. ``sort_order`` is his, typed on the tracker, and the
grid sorts by it -- his explicit request, so preparation can be grouped by
related entities and due dates rather than by whatever order MRI returns. Note
the 2Q26 sheet has that column EMPTY while the 4Q23 sheet has it filled 1..n:
the ordering has been redone by hand each quarter and lost in between.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import text

from flask_app.db import get_engine

logger = logging.getLogger(__name__)

#: The deliverables, in sign-off order, each with its own stages. ``owner``
#: labels the column with who is EXPECTED to enter it; it is not enforced,
#: because the person who actually signed is recorded on the cell.
DELIVERABLES = (
    {
        "key": "workpapers",
        "label": "Workpapers",
        "note": "Including FiHi - the spreadsheet's own wording.",
        "stages": (
            {"key": "prepared", "label": "Prepared", "owner": "Preparer"},
            {"key": "review_1", "label": "Reviewed", "owner": "Acctg Mgr"},
            {"key": "review_2", "label": "Reviewed", "owner": "CFO"},
        ),
    },
    {
        "key": "financial_statements",
        "label": "Financial Statements",
        "note": "",
        "stages": (
            {"key": "prepared", "label": "Prepared", "owner": "Preparer"},
            {"key": "review_1", "label": "Reviewed", "owner": "Acctg Mgr"},
            {"key": "review_2", "label": "Reviewed", "owner": "CFO"},
        ),
    },
    {
        "key": "capital_accounts",
        "label": "Capital Account Statements",
        "note": "",
        "stages": (
            {"key": "prepared", "label": "Prepared", "owner": "Preparer"},
            {"key": "review_1", "label": "Reviewed", "owner": "Acctg Mgr"},
            {"key": "review_2", "label": "Reviewed", "owner": "CFO"},
        ),
    },
    {
        "key": "investment_cafe",
        "label": "Investment Cafe",
        "note": ("The investor portal. Posting is still done IN Cafe and this "
                 "records that it happened. Handing the statements over "
                 "directly would need an integration with that system, which "
                 "does not exist yet."),
        "stages": (
            {"key": "fs_posted", "label": "FS Posted", "owner": "Preparer"},
            {"key": "fs_reviewed", "label": "Reviewed", "owner": "Acctg Mgr"},
            {"key": "cas_posted", "label": "CAS Posted", "owner": "Preparer"},
            {"key": "cas_reviewed", "label": "Reviewed", "owner": "Acctg Mgr"},
            {"key": "released", "label": "Released", "owner": "CFO"},
        ),
    },
)

DELIVERABLE_KEYS = tuple(d["key"] for d in DELIVERABLES)
STAGES_BY_DELIVERABLE = {
    d["key"]: tuple(s["key"] for s in d["stages"]) for d in DELIVERABLES}

#: Every cell of one row, flattened, in sign-off order. The sequence check reads
#: this: a cell is "out of sequence" when anything before it is still unsigned.
_ORDERED_CELLS = tuple(
    (d["key"], s["key"]) for d in DELIVERABLES for s in d["stages"])


_DDL = [
    # One row per (package, deliverable, stage). ABSENT means not signed -- never
    # a row carrying a null date, so "not signed" and "signed, date unknown"
    # cannot be confused by any reader.
    """
    CREATE TABLE IF NOT EXISTS wp_tracker_cell (
        id           {pk},
        package_id   INTEGER NOT NULL,
        deliverable  TEXT NOT NULL,
        stage        TEXT NOT NULL,
        signed_by    TEXT,
        signed_on    TEXT,
        note         TEXT,
        updated_by   TEXT,
        updated_at   TEXT
    )
    """,
    # The CFO's target date per (package, deliverable). Separate from the cell
    # table because a target has no signer; storing it as a cell with a null
    # `signed_by` would make every "is this signed?" test wrong.
    """
    CREATE TABLE IF NOT EXISTS wp_tracker_target (
        id           {pk},
        package_id   INTEGER NOT NULL,
        deliverable  TEXT NOT NULL,
        target_date  TEXT,
        updated_by   TEXT,
        updated_at   TEXT
    )
    """,
]


#: Set once the DDL has run in this process. Every read and every write calls
#: `ensure_tracker_tables`, and without this the two ALTER TABLE statements
#: below are attempted — and fail — on every single request. On PostgreSQL that
#: is two round trips and two server-side errors per call, which is noise in the
#: log and latency on the screen. Per process, not global state that matters:
#: a new worker re-runs it, which is exactly when it should.
_DDL_DONE = set()


def ensure_tracker_tables(engine=None) -> None:
    """Create the tracker tables, and the columns the tracker adds to packages."""
    engine = engine or get_engine()
    key = str(getattr(engine, "url", "")) or id(engine)
    if key in _DDL_DONE:
        return
    is_pg = engine.dialect.name == "postgresql"
    pk = "SERIAL PRIMARY KEY" if is_pg else "INTEGER PRIMARY KEY AUTOINCREMENT"
    with engine.begin() as conn:
        for ddl in _DDL:
            conn.execute(text(ddl.format(pk=pk)))
    # CREATE TABLE IF NOT EXISTS never alters an EXISTING table, so a column
    # added later has to be added explicitly or it is missing on every database
    # but a fresh one. Same trap `workpaper_service.ensure_tables` documents.
    for table, column, coltype in (
            ("wp_packages", "sort_order", "INTEGER"),
            ("wp_packages", "property_name", "TEXT")):
        try:
            with engine.begin() as c2:
                c2.execute(text(
                    "ALTER TABLE {} ADD COLUMN {} {}".format(
                        table, column, coltype)))
        except Exception:
            pass        # already present
    _DDL_DONE.add(key)


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def norm_date(v) -> Optional[str]:
    """A date as ISO ``yyyy-mm-dd``, or None. Never a partial and never a guess.

    The spreadsheet's own cells are free text -- ``KH 7/7/2026``,
    ``RE - 07/07/2026``, ``JS 7/7/26`` -- so anything imported from it arrives
    in three formats at once. An unparseable value returns None rather than
    today, because a sign-off dated by accident is worse than one left blank.
    """
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in ("none", "nan", "nat", "null"):
        return None
    if isinstance(v, datetime):
        return v.date().isoformat()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s).date().isoformat()
    except ValueError:
        return None


def describe() -> List[dict]:
    """The deliverable/stage shape, for the screen to render its own columns."""
    return [
        {
            "key": d["key"],
            "label": d["label"],
            "note": d["note"],
            "stages": [dict(s) for s in d["stages"]],
        }
        for d in DELIVERABLES
    ]


# ---------------------------------------------------------------- the grid

def _cells_for_cycle(conn, cycle_id: int) -> Dict[int, Dict[str, dict]]:
    rows = conn.execute(text(
        "SELECT c.package_id, c.deliverable, c.stage, c.signed_by, "
        "       c.signed_on, c.note "
        "  FROM wp_tracker_cell c "
        "  JOIN wp_packages p ON p.id = c.package_id "
        " WHERE p.cycle_id = :cid"), {"cid": cycle_id}).fetchall()
    out: Dict[int, Dict[str, dict]] = {}
    for r in rows:
        out.setdefault(int(r[0]), {})["%s:%s" % (r[1], r[2])] = {
            "signed_by": r[3], "signed_on": r[4], "note": r[5]}
    return out


def _targets_for_cycle(conn, cycle_id: int) -> Dict[int, Dict[str, str]]:
    rows = conn.execute(text(
        "SELECT t.package_id, t.deliverable, t.target_date "
        "  FROM wp_tracker_target t "
        "  JOIN wp_packages p ON p.id = t.package_id "
        " WHERE p.cycle_id = :cid"), {"cid": cycle_id}).fetchall()
    out: Dict[int, Dict[str, str]] = {}
    for r in rows:
        out.setdefault(int(r[0]), {})[str(r[1])] = r[2]
    return out


def grid(cycle_id: int, engine=None) -> dict:
    """The whole tracker for one close cycle, ordered as the CFO set it.

    Rows carry every cell, signed or not, so the screen renders a fixed grid and
    a missing sign-off is a visible blank rather than a missing column.
    """
    engine = engine or get_engine()
    ensure_tracker_tables(engine)
    today = datetime.utcnow().date().isoformat()

    with engine.connect() as conn:
        cyc = conn.execute(text(
            "SELECT id, period_label, period_end, status "
            "  FROM wp_cycles WHERE id = :cid"), {"cid": cycle_id}).fetchone()
        if not cyc:
            return {"error": "No close cycle %s." % cycle_id}

        pkgs = conn.execute(text(
            "SELECT id, entityid, entity_name, state, preparer, reviewer, "
            "       sort_order, property_name "
            "  FROM wp_packages WHERE cycle_id = :cid"),
            {"cid": cycle_id}).fetchall()
        cells = _cells_for_cycle(conn, cycle_id)
        targets = _targets_for_cycle(conn, cycle_id)

    rows = []
    for p in pkgs:
        pid = int(p[0])
        pc, pt = cells.get(pid, {}), targets.get(pid, {})

        # Walk the flattened cell order once, marking the first unsigned cell.
        # Everything signed AFTER an unsigned one is out of sequence.
        seen_unsigned = False
        out_of_sequence = []
        deliverables = []
        for d in DELIVERABLES:
            stages = []
            for s in d["stages"]:
                k = "%s:%s" % (d["key"], s["key"])
                cell = pc.get(k) or {}
                signed = bool(cell.get("signed_on") or cell.get("signed_by"))
                if signed and seen_unsigned:
                    out_of_sequence.append("%s / %s" % (d["label"], s["label"]))
                if not signed:
                    seen_unsigned = True
                stages.append({
                    "key": s["key"], "label": s["label"], "owner": s["owner"],
                    "signed": signed,
                    "signed_by": cell.get("signed_by"),
                    "signed_on": cell.get("signed_on"),
                    "note": cell.get("note"),
                })
            target = pt.get(d["key"])
            done = all(x["signed"] for x in stages)
            deliverables.append({
                "key": d["key"], "label": d["label"],
                "target_date": target,
                # Late only when there IS a target and it has passed with work
                # still outstanding. No target means no expectation, which is a
                # different thing from being on time.
                "overdue": bool(target and not done and target < today),
                "complete": done,
                "stages": stages,
            })

        signed_total = sum(1 for d in deliverables for s in d["stages"] if s["signed"])
        rows.append({
            "package_id": pid,
            "entityid": p[1],
            "entity_name": p[2],
            "state": p[3],
            "preparer": p[4],
            "reviewer": p[5],
            "sort_order": p[6],
            "property_name": p[7],
            "deliverables": deliverables,
            "signed_count": signed_total,
            "cell_count": len(_ORDERED_CELLS),
            "out_of_sequence": out_of_sequence,
        })

    # THE CFO'S ORDER FIRST, and unordered rows last rather than first: a row he
    # has not placed yet sorts to the bottom where it is obvious, instead of
    # displacing the sequence he did set.
    rows.sort(key=lambda r: (
        r["sort_order"] is None,
        r["sort_order"] if r["sort_order"] is not None else 0,
        (r["entity_name"] or r["entityid"] or "").lower()))

    unordered = [r["entityid"] for r in rows if r["sort_order"] is None]
    dupes = _duplicate_orders(rows)
    return {
        "cycle": {"id": cyc[0], "period_label": cyc[1],
                  "period_end": cyc[2], "status": cyc[3]},
        "deliverables": describe(),
        "rows": rows,
        "diagnostics": {
            "row_count": len(rows),
            "unordered_count": len(unordered),
            "unordered": unordered[:40],
            # Two rows sharing an order number is not an error -- the sort is
            # stable on name beneath it -- but it means the CFO's intent is
            # ambiguous for those rows, so it is said out loud.
            "duplicate_orders": dupes,
            "out_of_sequence_rows": [
                {"entityid": r["entityid"], "cells": r["out_of_sequence"]}
                for r in rows if r["out_of_sequence"]],
        },
    }


def _duplicate_orders(rows: List[dict]) -> List[dict]:
    seen: Dict[int, List[str]] = {}
    for r in rows:
        if r["sort_order"] is not None:
            seen.setdefault(int(r["sort_order"]), []).append(r["entityid"])
    return [{"order": k, "entities": v}
            for k, v in sorted(seen.items()) if len(v) > 1]


# ---------------------------------------------------------------- writes

def set_order(package_id: int, order, user: str = "", engine=None) -> dict:
    """Set (or clear) the CFO's display order for one row."""
    engine = engine or get_engine()
    ensure_tracker_tables(engine)
    val = None
    if order is not None and str(order).strip() != "":
        try:
            val = int(str(order).strip())
        except ValueError:
            return {"error": "Order must be a whole number, or blank to clear."}
        if val < 0:
            return {"error": "Order cannot be negative."}
    with engine.begin() as conn:
        conn.execute(text(
            "UPDATE wp_packages SET sort_order = :o, updated_at = :t "
            " WHERE id = :pid"),
            {"o": val, "t": _now(), "pid": package_id})
    return {"ok": True, "package_id": package_id, "sort_order": val}


def renumber(cycle_id: int, user: str = "", engine=None) -> dict:
    """Rewrite the order as 1..n in the order currently shown.

    Offered because hand-typed numbers drift into 10, 20, 25, 27, 27, 28 and
    inserting a row between two of them stops being possible. It preserves the
    sequence on screen exactly; it only makes the numbers contiguous.
    """
    engine = engine or get_engine()
    g = grid(cycle_id, engine)
    if g.get("error"):
        return g
    with engine.begin() as conn:
        for i, r in enumerate(g["rows"], start=1):
            conn.execute(text(
                "UPDATE wp_packages SET sort_order = :o WHERE id = :pid"),
                {"o": i, "pid": r["package_id"]})
    return {"ok": True, "renumbered": len(g["rows"])}


def set_target(package_id: int, deliverable: str, target_date,
               user: str = "", engine=None) -> dict:
    """Set or clear the CFO's target date for one deliverable on one row."""
    if deliverable not in DELIVERABLE_KEYS:
        return {"error": "Unknown deliverable %r." % deliverable}
    engine = engine or get_engine()
    ensure_tracker_tables(engine)
    raw = (str(target_date).strip() if target_date is not None else "")
    iso = norm_date(target_date) if raw else None
    if raw and iso is None:
        return {"error": "Could not read %r as a date." % raw}
    with engine.begin() as conn:
        existing = conn.execute(text(
            "SELECT id FROM wp_tracker_target "
            " WHERE package_id = :pid AND deliverable = :d"),
            {"pid": package_id, "d": deliverable}).fetchone()
        if existing:
            conn.execute(text(
                "UPDATE wp_tracker_target SET target_date = :v, "
                "       updated_by = :u, updated_at = :t WHERE id = :id"),
                {"v": iso, "u": user, "t": _now(), "id": existing[0]})
        else:
            conn.execute(text(
                "INSERT INTO wp_tracker_target "
                "  (package_id, deliverable, target_date, updated_by, updated_at) "
                "VALUES (:pid, :d, :v, :u, :t)"),
                {"pid": package_id, "d": deliverable, "v": iso,
                 "u": user, "t": _now()})
    return {"ok": True, "target_date": iso}


def set_signoff(package_id: int, deliverable: str, stage: str,
                signed_by=None, signed_on=None, note=None,
                user: str = "", engine=None) -> dict:
    """Record, amend or clear one sign-off.

    Clearing means BOTH ``signed_by`` and ``signed_on`` empty; the row is
    deleted rather than left with nulls, so absence keeps meaning "not signed".
    """
    if deliverable not in DELIVERABLE_KEYS:
        return {"error": "Unknown deliverable %r." % deliverable}
    if stage not in STAGES_BY_DELIVERABLE[deliverable]:
        return {"error": "%r has no stage %r." % (deliverable, stage)}

    engine = engine or get_engine()
    ensure_tracker_tables(engine)
    who = (str(signed_by).strip() if signed_by is not None else "")
    raw = (str(signed_on).strip() if signed_on is not None else "")
    when = norm_date(signed_on) if raw else None
    if raw and when is None:
        return {"error": "Could not read %r as a date." % raw}

    clearing = not who and not when
    with engine.begin() as conn:
        existing = conn.execute(text(
            "SELECT id FROM wp_tracker_cell "
            " WHERE package_id = :pid AND deliverable = :d AND stage = :s"),
            {"pid": package_id, "d": deliverable, "s": stage}).fetchone()
        if clearing:
            if existing:
                conn.execute(text("DELETE FROM wp_tracker_cell WHERE id = :id"),
                             {"id": existing[0]})
            return {"ok": True, "cleared": True}
        if existing:
            conn.execute(text(
                "UPDATE wp_tracker_cell SET signed_by = :by, signed_on = :on, "
                "       note = :n, updated_by = :u, updated_at = :t "
                " WHERE id = :id"),
                {"by": who or None, "on": when, "n": note,
                 "u": user, "t": _now(), "id": existing[0]})
        else:
            conn.execute(text(
                "INSERT INTO wp_tracker_cell "
                "  (package_id, deliverable, stage, signed_by, signed_on, "
                "   note, updated_by, updated_at) "
                "VALUES (:pid, :d, :s, :by, :on, :n, :u, :t)"),
                {"pid": package_id, "d": deliverable, "s": stage,
                 "by": who or None, "on": when, "n": note,
                 "u": user, "t": _now()})
    return {"ok": True, "signed_by": who or None, "signed_on": when}


def set_preparer(package_id: int, preparer, user: str = "", engine=None) -> dict:
    """The preparer's INITIALS, as the spreadsheet carries them (KH, NL, RE)."""
    engine = engine or get_engine()
    ensure_tracker_tables(engine)
    val = (str(preparer).strip() if preparer is not None else "") or None
    with engine.begin() as conn:
        conn.execute(text(
            "UPDATE wp_packages SET preparer = :p, updated_at = :t "
            " WHERE id = :pid"),
            {"p": val, "t": _now(), "pid": package_id})
    return {"ok": True, "preparer": val}


def set_property(package_id: int, name, user: str = "", engine=None) -> dict:
    """The property or portfolio this entity reports on, for grouping."""
    engine = engine or get_engine()
    ensure_tracker_tables(engine)
    val = (str(name).strip() if name is not None else "") or None
    with engine.begin() as conn:
        conn.execute(text(
            "UPDATE wp_packages SET property_name = :n WHERE id = :pid"),
            {"n": val, "pid": package_id})
    return {"ok": True, "property_name": val}


def carry_forward(from_cycle_id: int, to_cycle_id: int,
                  user: str = "", engine=None) -> dict:
    """Copy order, preparer and property from one cycle to the next.

    NOT the sign-offs or the target dates. Those are facts about a quarter and
    copying them would assert that work was done when it has not been. What
    carries is the ARRANGEMENT -- the CFO's ordering and his preparer
    assignments -- which is exactly the part being retyped every quarter today.
    """
    engine = engine or get_engine()
    ensure_tracker_tables(engine)
    with engine.connect() as conn:
        src = conn.execute(text(
            "SELECT entityid, sort_order, preparer, property_name "
            "  FROM wp_packages WHERE cycle_id = :cid"),
            {"cid": from_cycle_id}).fetchall()
        dst = conn.execute(text(
            "SELECT id, entityid FROM wp_packages WHERE cycle_id = :cid"),
            {"cid": to_cycle_id}).fetchall()
    by_entity = {str(r[0]).strip().upper(): r for r in src}
    applied, missing = 0, []
    with engine.begin() as conn:
        for pid, eid in dst:
            r = by_entity.get(str(eid).strip().upper())
            if not r:
                missing.append(eid)
                continue
            conn.execute(text(
                "UPDATE wp_packages SET sort_order = :o, preparer = :p, "
                "       property_name = :n WHERE id = :pid"),
                {"o": r[1], "p": r[2], "n": r[3], "pid": pid})
            applied += 1
    return {"ok": True, "applied": applied,
            # An entity in the new cycle that the old one did not have is new
            # to the population, and the CFO has to place it himself.
            "not_in_source": sorted(missing)}
