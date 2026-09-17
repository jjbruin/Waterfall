"""The close TRACKER: who owes what, when it was due, and who signed it off.

WHAT THIS IS, AND WHAT IT IS NOT. ``workpaper_service`` owns a PACKAGE -- one
entity's workbook, its steps, its exhibits, its state machine. This module owns
the CFO's view ACROSS entities: the grid he keeps today in
``2Q26 - PSC Reporting Checklist & Calendar.xlsx``, tab "Reporting Calendar" --
a target date, a preparer's initials, and a set of sign-offs per deliverable.

THE POPULATION IS THE `REP` TAG, NOT THAT SPREADSHEET. Jim, Sep 16 2026: "the
population should be driven by the REP tag, not the spreadsheet I provided."
MRI tags 58 entities REP and the 2Q26 sheet carries 55 rows; four tagged
entities are absent from it (INVCW, NOTTNV, PEGASU, TGACW) and one of its rows,
"Various / PSC Investee Fund LLC - All", is a roll-up rather than an entity.
Those are not discrepancies to reconcile -- the tag decides, so an entity tagged
REP appears whether or not anybody remembered to add a row for it, which is the
failure the spreadsheet has no way to catch. `workpaper_service.sync_packages`
is the one place the population is read.

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

import pandas as pd
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
    # NAMED FOR THE ACT, NOT THE VENDOR. This was keyed `investment_cafe` --
    # a portal's product name welded into the schema, the API and every stored
    # row. Investment Cafe is the CHANNEL we deliver through today; the
    # deliverable is getting finished statements to investors. Renamed before
    # anything was deployed, because the key is stored on every tracker cell
    # and changing it later is a migration rather than an edit.
    {
        "key": "investor_delivery",
        "label": "Investor Delivery",
        "channel": "Investment Cafe",
        "note": ("Posting happens IN Investment Cafe and this records that it "
                 "did. The goal is for the app to produce the finished PDF "
                 "and deliver it -- one entity, a batch, or a link handed to "
                 "the portal. See DELIVERY_GOAL below."),
        "stages": (
            {"key": "fs_posted", "label": "FS Posted", "owner": "Preparer"},
            {"key": "fs_reviewed", "label": "Reviewed", "owner": "Acctg Mgr"},
            {"key": "cas_posted", "label": "CAS Posted", "owner": "Preparer"},
            {"key": "cas_reviewed", "label": "Reviewed", "owner": "Acctg Mgr"},
            {"key": "released", "label": "Released", "owner": "CFO"},
        ),
    },
)

#: WHERE THIS IS GOING, so that what gets built next does not have to undo what
#: is here. Jim, Sep 16 2026: "our goal is to deliver finished and
#: professionally formatted financial statements by pdf either individually, in
#: batches, or by direct link to the portal."
#:
#: BORROW THE ONE PAGER'S BATCH PDF. It already does two of those three modes,
#: and the split is the part worth copying (Jim, Sep 16 2026: "we have a pdf
#: reporting function within the one pager that handles individual and
#: batches"):
#:
#:   * SERVER assembles MANY entities in one request.
#:     `POST /api/financials/one-pager/batch` takes a list of vcodes and returns
#:     `pages: [{vcode, data, chart, error?}]`. Note the per-entity `error`: one
#:     deal that fails to compute does not take the batch down with it, which at
#:     58 entities is the difference between a usable run and an all-or-nothing
#:     one.
#:   * CLIENT renders them stacked, one sheet per entity with
#:     `page-break-after: always` between them under `@media print`, and a single
#:     `window.print()` produces the whole batch as one PDF.
#:
#: So "individually" and "in batches" are already solved problems here, and the
#: statements have the same shape: `statement_service.build()` is per entity and
#: period, exactly like `get_one_pager_data`.
#:
#: THE ONE MODE NEITHER PATTERN COVERS is the portal link, because that needs a
#: file the SERVER produced and can address, and both paths end at a browser
#: print dialog. That is the only piece the integration actually adds -- and it
#: is a reason to keep the formatting in a print VIEW rather than in a Python
#: PDF library, since a headless browser can later drive the same view
#: server-side. One renderer, three modes; two of them working today.
#:
#: AND "PROFESSIONALLY FORMATTED" HAS A TEST. `scripts/snapshot_print_formatting_
#: check.py` reads the produced PDF's drawing operators to assert margins and
#: page counts -- borders are vector, so the text layer cannot answer either
#: question. At 58 entities that check is what keeps a table from growing off
#: the sheet unnoticed, which is exactly what happened to the snapshot's
#: Financial tab on Sep 16 2026.
#:
#: WHAT THAT MEANS FOR THIS MODULE, and why it is shaped as it is:
#:   * the deliverable is keyed `investor_delivery`, not for a portal, so a
#:     second channel does not need a second deliverable;
#:   * its stages already separate the FINANCIAL STATEMENTS from the CAPITAL
#:     ACCOUNT STATEMENTS, which is what a batch would render per entity;
#:   * `released` is the single point at which a quarter is investor-visible,
#:     which is the event a portal hand-off would hang from.
#: Nothing here renders a PDF yet, deliberately -- the integration is a future
#: project and this only keeps the road to it open.
DELIVERY_GOAL = (
    "Finished, professionally formatted statements delivered as PDF: one "
    "entity at a time, a batch across entities, or a link handed to the "
    "investor portal. Individual and batch follow the One Pager's pattern -- a "
    "server-side batch endpoint plus a print view with page breaks. Only the "
    "portal link needs machinery that does not exist yet.")

DELIVERABLE_KEYS = tuple(d["key"] for d in DELIVERABLES)
STAGES_BY_DELIVERABLE = {
    d["key"]: tuple(s["key"] for s in d["stages"]) for d in DELIVERABLES}

#: Every cell of one row, flattened, in sign-off order. The sequence check reads
#: this: a cell is "out of sequence" when anything before it is still unsigned.
_ORDERED_CELLS = tuple(
    (d["key"], s["key"]) for d in DELIVERABLES for s in d["stages"])


_DDL = [
    # WHO MAY BE ASSIGNED AS PREPARER OR REVIEWER.
    #
    # Not derived from `users`: measured Sep 17 2026, every account is `analyst`
    # or `admin` and NONE carries an accounting role, while the close is
    # actually prepared by KH, NL and RE -- initials matching no account at all.
    # A dropdown built from `users` would have been empty of the people who do
    # the work, which is worse than the free-text box it replaces.
    #
    # So the list is maintained here and OPTIONALLY linked to an account by
    # `username`. When those people get accounts the link closes the gap without
    # the assignments having to be redone.
    """
    CREATE TABLE IF NOT EXISTS wp_preparers (
        id        {pk},
        initials  TEXT NOT NULL,
        name      TEXT,
        wp_role   TEXT,
        username  TEXT,
        active    INTEGER DEFAULT 1
    )
    """,
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
            ("wp_packages", "property_name", "TEXT"),
            # HOW the property was arrived at, so an inferred name is never
            # mistaken for a recorded one. NULL means a person typed it.
            ("wp_packages", "property_basis", "TEXT")):
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
            "       sort_order, property_name, property_basis "
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
            # NULL when a person typed it; a sentence when the app inferred it.
            "property_basis": p[8],
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
        # THE BASIS IS CLEARED when a person types the value. It describes how
        # the app INFERRED a name; leaving it attached to something the CFO
        # typed would credit his decision to a walk of the commitments table.
        conn.execute(text(
            "UPDATE wp_packages SET property_name = :n, property_basis = NULL "
            " WHERE id = :pid"),
            {"n": val, "pid": package_id})
    return {"ok": True, "property_name": val, "property_basis": None}


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
            "SELECT entityid, sort_order, preparer, property_name, "
            "       property_basis FROM wp_packages WHERE cycle_id = :cid"),
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
                "       property_name = :n, property_basis = :b "
                " WHERE id = :pid"),
                {"o": r[1], "p": r[2], "n": r[3], "b": r[4], "pid": pid})
            applied += 1
    return {"ok": True, "applied": applied,
            # An entity in the new cycle that the old one did not have is new
            # to the population, and the CFO has to place it himself.
            "not_in_source": sorted(missing)}


# ---------------------------------------------------------------- people

#: The roles a close assignment can carry. `cfo` is here because the CFO signs
#: the second review and is assignable like anyone else.
PREPARER_ROLES = ("accountant", "accounting_manager", "cfo")


def _initials_from(username: str, email: str = "") -> str:
    """``jstewart`` -> ``JS``.

    A guess, and only ever a SEED for a list the CFO edits -- never a value
    written onto a package. Two letters from a username is right often enough to
    save typing and wrong often enough that it must not be authoritative.
    """
    u = (username or "").strip()
    if not u:
        u = (email or "").split("@")[0]
    if not u:
        return ""
    if "." in u:
        a, b = u.split(".", 1)
        return (a[:1] + b[:1]).upper()
    return (u[:2]).upper() if len(u) > 1 else u[:1].upper()


def list_preparers(engine=None, include_inactive: bool = False) -> List[dict]:
    """Everyone assignable to a close, from three sources that have to agree.

    THE USER LIST IS THE PRIMARY SOURCE. Jim is adding the accountants and the
    accounting manager to it (Sep 17 2026); anyone whose role is in
    :data:`PREPARER_ROLES` appears here automatically, so the dropdown fills as
    soon as those accounts exist and nobody has to maintain a second list.

    `wp_preparers` remains for two things a user list cannot do: someone who
    prepares but has no account yet, and CORRECTING the initials guessed from a
    username -- a row linked by `username` overrides the guess without changing
    the account.

    And anyone ALREADY assigned on a package is included whatever their origin.
    Initials typed before this list existed must not vanish from the dropdown,
    or the CFO opens the screen to find his own assignments unselectable.

    COLLISIONS ARE REPORTED, NOT MERGED. Two people whose usernames give the
    same two letters are two people; silently folding them together would put
    one person's initials against the other's work. Each entry carries `clash`
    so the screen can show the name beside the initials.
    """
    engine = engine or get_engine()
    ensure_tracker_tables(engine)

    by_initials: Dict[str, dict] = {}
    order: List[str] = []

    def put(entry):
        key = str(entry["initials"]).strip().upper()
        if not key:
            return
        if key in by_initials:
            prior = by_initials[key]
            # An explicit wp_preparers row outranks a guess from a username.
            if entry["source"] == "list" and prior["source"] == "user":
                entry["clash"] = prior.get("clash") or []
                by_initials[key] = entry
            else:
                prior.setdefault("clash", []).append(
                    entry.get("name") or entry.get("username") or key)
            return
        by_initials[key] = entry
        order.append(key)

    # 1. accounts carrying an accounting role
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT username, email, role FROM users")).fetchall()
        for uname, email, role in rows:
            if str(role or "").strip().lower() not in PREPARER_ROLES:
                continue
            put({"id": None, "initials": _initials_from(uname, email),
                 "name": uname, "username": uname,
                 "wp_role": str(role).strip().lower(), "active": True,
                 "source": "user"})
    except Exception as e:                                  # pragma: no cover
        logger.warning("list_preparers could not read users: %s", e)

    # 2. the maintained list, which may correct or extend the above
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT id, initials, name, wp_role, username, active "
            "  FROM wp_preparers")).fetchall()
    for r in rows:
        if not (include_inactive or r[5]):
            continue
        put({"id": r[0], "initials": r[1], "name": r[2], "wp_role": r[3],
             "username": r[4], "active": bool(r[5]), "source": "list"})

    # 3. anyone already assigned
    with engine.connect() as conn:
        used = {str(r[0]).strip().upper()
                for r in conn.execute(text(
                    "SELECT DISTINCT preparer FROM wp_packages "
                    " WHERE preparer IS NOT NULL AND preparer <> ''")).fetchall()
                if r[0]}
    for ini in sorted(used - set(by_initials)):
        put({"id": None, "initials": ini, "name": None, "wp_role": None,
             "username": None, "active": True, "source": "in use"})

    out = [by_initials[k] for k in order]
    out.sort(key=lambda p: (p["initials"] or "").upper())
    return out


def add_preparer(initials, name=None, wp_role=None, username=None,
                 engine=None) -> dict:
    ini = (str(initials or "").strip().upper())[:6]
    if not ini:
        return {"error": "Initials are required."}
    if wp_role and wp_role not in PREPARER_ROLES:
        return {"error": "Unknown role %r." % wp_role}
    engine = engine or get_engine()
    ensure_tracker_tables(engine)
    with engine.begin() as conn:
        dup = conn.execute(text(
            "SELECT id FROM wp_preparers WHERE UPPER(initials) = :i"),
            {"i": ini}).fetchone()
        if dup:
            conn.execute(text(
                "UPDATE wp_preparers SET name = :n, wp_role = :r, "
                "       username = :u, active = 1 WHERE id = :id"),
                {"n": name, "r": wp_role, "u": username, "id": dup[0]})
            return {"ok": True, "initials": ini, "updated": True}
        conn.execute(text(
            "INSERT INTO wp_preparers (initials, name, wp_role, username, active) "
            "VALUES (:i, :n, :r, :u, 1)"),
            {"i": ini, "n": name, "r": wp_role, "u": username})
    return {"ok": True, "initials": ini}


def remove_preparer(initials, engine=None) -> dict:
    """Deactivates rather than deletes: a name already on a signed-off package
    has to keep resolving."""
    engine = engine or get_engine()
    ensure_tracker_tables(engine)
    with engine.begin() as conn:
        conn.execute(text(
            "UPDATE wp_preparers SET active = 0 WHERE UPPER(initials) = :i"),
            {"i": str(initials or "").strip().upper()})
    return {"ok": True}


# ---------------------------------------------------------------- property

def derive_properties(engine=None, max_hops: int = 2) -> Dict[str, dict]:
    """Entity -> the deal it reports on, walking ``commitments`` up to N levels.

    Jim, Sep 17 2026: "import the associated deal name in the Property column"
    and then, once the first hop was measured, "build hop 2 with the basis
    visible."

    WHY MORE THAN ONE HOP. Most reporting entities do not commit into a deal;
    they commit into another entity that does. Measured on production, one hop
    leaves 35 of the 58 blank and two hops leaves 18 -- and six of the seven new
    names match the CFO's own spreadsheet exactly (INVBPS -> Brainerd Place,
    INVGAT and PIG2PA -> The Gathering, PPIPIT -> Berger Pittsburgh, TGAAS ->
    Ascent, TGANOT -> Nottingham Village).

    WHY THE BASIS IS RETURNED WITH IT, AND WHY THAT MATTERS MORE AT TWO HOPS
    THAN AT ONE. The seventh new name is wrong: TGA6 is a fund, and at two
    levels exactly one deal happens to be reachable, so it comes back as
    "Presidential Arms JV" where the CFO's sheet says "Various". The walk cannot
    tell a single-purpose chain from a fund that happens to have one reachable
    holding. So every derived value says how it was reached, the screen shows
    it, and the field stays editable -- the answer is a starting point that
    declares itself, not a fact.

    Three outcomes, unchanged in spirit from one hop:
      * exactly one deal reachable  -> that deal's name
      * several                     -> "Various", the CFO's own spelling
      * none                        -> absent, left for him to type
    """
    engine = engine or get_engine()
    out: Dict[str, dict] = {}
    try:
        with engine.connect() as conn:
            com = pd.read_sql(text("SELECT * FROM commitments"), conn)
            dl = pd.read_sql(text("SELECT * FROM deals"), conn)
    except Exception as e:
        logger.warning("derive_properties failed: %s", e)
        return out
    if com.empty or dl.empty:
        return out
    cm = {str(c).lower(): c for c in com.columns}
    dm = {str(c).lower(): c for c in dl.columns}
    if not (cm.get("investorid") and cm.get("entityid")
            and dm.get("investmentid") and dm.get("investment_name")):
        return out
    if cm.get("enddate"):
        com = com[com[cm["enddate"]].isna()]
    com = com.assign(
        _inv=com[cm["investorid"]].astype(str).str.strip().str.upper(),
        _ent=com[cm["entityid"]].astype(str).str.strip().str.upper())
    names = dl.set_index(
        dl[dm["investmentid"]].astype(str).str.strip().str.upper()
    )[dm["investment_name"]].to_dict()
    holdings = com.groupby("_inv")["_ent"].apply(list).to_dict()

    def reach(start: str):
        """Deals reachable from `start`, and the fewest hops that found one.

        Breadth first, so `depth` is the SHALLOWEST level at which this entity
        touches a deal -- which is what the basis should report. A cycle in the
        data would otherwise walk forever, so visited nodes are never re-queued.
        """
        seen = {start}
        found: Dict[str, int] = {}
        frontier = [(start, 0)]
        while frontier:
            node, depth = frontier.pop(0)
            if depth >= max_hops:
                continue
            for ent in holdings.get(node, []):
                if names.get(ent):
                    found.setdefault(ent, depth + 1)
                elif ent not in seen:
                    seen.add(ent)
                    frontier.append((ent, depth + 1))
        return found

    for inv in holdings:
        found = reach(inv)
        if not found:
            continue
        hops = min(found.values())
        via = "" if hops <= 1 else ", %d levels down" % hops
        if len(found) == 1:
            deal = next(iter(found))
            out[inv] = {"property_name": str(names[deal]).strip(),
                        "hops": hops,
                        "basis": "the one deal it holds" + via}
        else:
            out[inv] = {"property_name": "Various",
                        "hops": hops,
                        "basis": "%d deals%s" % (len(found), via)}
    return out


def apply_derived_properties(cycle_id: int, overwrite: bool = False,
                             engine=None) -> dict:
    """Fill the Property column from the deals, leaving typed values alone.

    ``overwrite`` is False by default because a value the CFO typed is a
    decision and a derived one is a guess; the guess never wins.
    """
    engine = engine or get_engine()
    ensure_tracker_tables(engine)
    derived = derive_properties(engine)
    filled, skipped, unresolved = 0, 0, []
    with engine.begin() as conn:
        rows = conn.execute(text(
            "SELECT id, entityid, property_name FROM wp_packages "
            " WHERE cycle_id = :c"), {"c": cycle_id}).fetchall()
        for pid, eid, existing in rows:
            d = derived.get(str(eid).strip().upper())
            if not d:
                unresolved.append(eid)
                continue
            if existing and str(existing).strip() and not overwrite:
                skipped += 1
                continue
            conn.execute(text(
                "UPDATE wp_packages SET property_name = :n, property_basis = :b "
                " WHERE id = :id"),
                {"n": d["property_name"], "b": d.get("basis"), "id": pid})
            filled += 1
    return {"ok": True, "filled": filled, "kept_existing": skipped,
            "unresolved": sorted(unresolved),
            "unresolved_count": len(unresolved)}
