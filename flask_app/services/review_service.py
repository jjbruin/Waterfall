"""Review workflow service for One Pager approval pipeline."""

import json
import logging
from sqlalchemy import text
from flask_app.db import get_engine, is_postgres

logger = logging.getLogger(__name__)


# Sequential review steps
REVIEW_STEPS = [
    {'step': 0, 'role': 'asset_manager', 'status': 'draft',           'label': 'Draft'},
    {'step': 1, 'role': 'head_am',       'status': 'pending_head_am', 'label': 'Head of AM'},
    {'step': 2, 'role': 'president',     'status': 'pending_president', 'label': 'President'},
    {'step': 3, 'role': 'cco',           'status': 'pending_cco',     'label': 'CCO'},
    {'step': 4, 'role': 'ceo',           'status': 'pending_ceo',     'label': 'CEO'},
    {'step': 5, 'role': None,            'status': 'approved',        'label': 'Approved'},
]

REVIEW_ROLE_NAMES = [s['role'] for s in REVIEW_STEPS if s['role']]


def _ensure_tables():
    """Create review tables if they don't exist."""
    pk = "SERIAL PRIMARY KEY" if is_postgres() else "INTEGER PRIMARY KEY AUTOINCREMENT"
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS review_roles (
                id {pk},
                user_id INTEGER NOT NULL,
                review_role TEXT NOT NULL,
                UNIQUE(user_id, review_role),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS review_submissions (
                id {pk},
                vcode TEXT NOT NULL,
                quarter TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'draft',
                current_step INTEGER NOT NULL DEFAULT 0,
                submitted_by INTEGER,
                returned_to_step INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(vcode, quarter)
            )
        """))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS review_notes (
                id {pk},
                vcode TEXT NOT NULL,
                quarter TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                username TEXT NOT NULL,
                review_role TEXT,
                action TEXT NOT NULL,
                note_text TEXT,
                addressed INTEGER NOT NULL DEFAULT 0,
                addressed_by TEXT,
                addressed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
    # Migration: add addressed columns if missing (existing tables)
    with engine.begin() as conn:
        if is_postgres():
            row = conn.execute(text("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'review_notes' AND column_name = 'addressed'
            """)).fetchone()
            needs_migration = row is None
        else:
            cols = conn.execute(text("PRAGMA table_info(review_notes)")).fetchall()
            needs_migration = not any(c[1] == 'addressed' for c in cols)

        if needs_migration:
            conn.execute(text(
                "ALTER TABLE review_notes ADD COLUMN addressed INTEGER NOT NULL DEFAULT 0"
            ))
            conn.execute(text(
                "ALTER TABLE review_notes ADD COLUMN addressed_by TEXT"
            ))
            conn.execute(text(
                "ALTER TABLE review_notes ADD COLUMN addressed_at TIMESTAMP"
            ))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS one_pager_snapshots (
                id {pk},
                vcode TEXT NOT NULL,
                quarter TEXT NOT NULL,
                snapshot_data TEXT NOT NULL,
                approved_by TEXT NOT NULL,
                approved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(vcode, quarter)
            )
        """))


def _step_for(step_num: int) -> dict:
    """Get step definition by step number."""
    for s in REVIEW_STEPS:
        if s['step'] == step_num:
            return s
    return REVIEW_STEPS[0]


def _step_for_status(status: str) -> dict:
    """Get step definition by status string."""
    for s in REVIEW_STEPS:
        if s['status'] == status:
            return s
    return REVIEW_STEPS[0]


# ── Role management ─────────────────────────────────────────

def get_user_review_roles(user_id: int) -> list[str]:
    """Get review roles for a user."""
    _ensure_tables()
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT review_role FROM review_roles WHERE user_id = :uid"),
            {"uid": user_id},
        ).fetchall()
    return [r[0] for r in rows]


def list_review_role_assignments() -> list[dict]:
    """List all review role assignments with user info."""
    _ensure_tables()
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT rr.id, rr.user_id, u.username, rr.review_role
            FROM review_roles rr
            JOIN users u ON u.id = rr.user_id
            ORDER BY rr.review_role, u.username
        """)).mappings().fetchall()
    return [dict(r) for r in rows]


def assign_review_role(user_id: int, review_role: str) -> dict | None:
    """Assign a review role to a user. Returns the assignment or None if duplicate."""
    _ensure_tables()
    if review_role not in REVIEW_ROLE_NAMES:
        raise ValueError(f"Invalid review role: {review_role}")
    engine = get_engine()
    try:
        with engine.begin() as conn:
            conn.execute(
                text("INSERT INTO review_roles (user_id, review_role) VALUES (:uid, :role)"),
                {"uid": user_id, "role": review_role},
            )
            row = conn.execute(
                text("SELECT id, user_id, review_role FROM review_roles WHERE user_id = :uid AND review_role = :role"),
                {"uid": user_id, "role": review_role},
            ).mappings().fetchone()
        return dict(row) if row else None
    except Exception:
        return None


def remove_review_role(role_id: int) -> bool:
    """Remove a review role assignment."""
    engine = get_engine()
    with engine.begin() as conn:
        result = conn.execute(
            text("DELETE FROM review_roles WHERE id = :id"), {"id": role_id}
        )
        return result.rowcount > 0


# ── Submission management ────────────────────────────────────

def get_submission(vcode: str, quarter: str) -> dict:
    """Get or create a submission record. Returns submission + notes + permissions context."""
    _ensure_tables()
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM review_submissions WHERE vcode = :v AND quarter = :q"),
            {"v": vcode, "q": quarter},
        ).mappings().fetchone()

    if row:
        sub = dict(row)
    else:
        # Virtual draft (not persisted until first action)
        sub = {
            "id": None,
            "vcode": vcode,
            "quarter": quarter,
            "status": "draft",
            "current_step": 0,
            "submitted_by": None,
            "returned_to_step": None,
            "created_at": None,
            "updated_at": None,
        }

    step_info = _step_for(sub["current_step"])
    sub["current_step_label"] = step_info["label"]
    sub["current_step_role"] = step_info["role"]

    # Get notes
    sub["notes"] = _get_notes(vcode, quarter)

    return sub


def _get_notes(vcode: str, quarter: str) -> list[dict]:
    """Get all review notes for a submission, newest first."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT id, user_id, username, review_role, action, note_text,
                       addressed, addressed_by, addressed_at, created_at
                FROM review_notes
                WHERE vcode = :v AND quarter = :q
                ORDER BY created_at DESC
            """),
            {"v": vcode, "q": quarter},
        ).mappings().fetchall()
    return [dict(r) for r in rows]


def _ensure_submission(vcode: str, quarter: str, user_id: int) -> dict:
    """Get existing submission or create one."""
    engine = get_engine()
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT * FROM review_submissions WHERE vcode = :v AND quarter = :q"),
            {"v": vcode, "q": quarter},
        ).mappings().fetchone()
        if row:
            return dict(row)
        conn.execute(
            text("""
                INSERT INTO review_submissions (vcode, quarter, status, current_step, submitted_by)
                VALUES (:v, :q, 'draft', 0, :uid)
            """),
            {"v": vcode, "q": quarter, "uid": user_id},
        )
        row = conn.execute(
            text("SELECT * FROM review_submissions WHERE vcode = :v AND quarter = :q"),
            {"v": vcode, "q": quarter},
        ).mappings().fetchone()
        return dict(row)


def _add_note(vcode: str, quarter: str, user_id: int, username: str,
              review_role: str | None, action: str, note_text: str | None):
    """Insert a review note."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO review_notes (vcode, quarter, user_id, username, review_role, action, note_text)
                VALUES (:v, :q, :uid, :uname, :role, :action, :note)
            """),
            {"v": vcode, "q": quarter, "uid": user_id, "uname": username,
             "role": review_role, "action": action, "note": note_text},
        )


def _update_submission(vcode: str, quarter: str, status: str, current_step: int,
                       returned_to_step: int | None = None):
    """Update submission status and step."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text("""
                UPDATE review_submissions
                SET status = :s, current_step = :step, returned_to_step = :ret,
                    updated_at = CURRENT_TIMESTAMP
                WHERE vcode = :v AND quarter = :q
            """),
            {"s": status, "step": current_step, "ret": returned_to_step,
             "v": vcode, "q": quarter},
        )


# ── Workflow actions ─────────────────────────────────────────

def submit_for_review(vcode: str, quarter: str, user_id: int, username: str,
                      note_text: str | None = None) -> dict:
    """Submit a draft/returned document for review. Moves to step 1 (pending_head_am)."""
    sub = _ensure_submission(vcode, quarter, user_id)
    if sub["status"] not in ("draft", "returned"):
        raise ValueError(f"Cannot submit: current status is '{sub['status']}'")

    user_roles = get_user_review_roles(user_id)
    if "asset_manager" not in user_roles:
        raise PermissionError("Only asset managers can submit for review")

    next_step = REVIEW_STEPS[1]
    _update_submission(vcode, quarter, next_step["status"], next_step["step"])
    _add_note(vcode, quarter, user_id, username, "asset_manager", "submit", note_text)
    return get_submission(vcode, quarter)


def approve(vcode: str, quarter: str, user_id: int, username: str,
            note_text: str | None = None) -> dict:
    """Approve at current step and advance to next."""
    sub = _ensure_submission(vcode, quarter, user_id)
    current_step_num = sub["current_step"]

    if current_step_num < 1 or current_step_num > 4:
        raise ValueError(f"Nothing to approve at step {current_step_num}")

    current_step_def = _step_for(current_step_num)
    user_roles = get_user_review_roles(user_id)
    if current_step_def["role"] not in user_roles:
        raise PermissionError(
            f"You need the '{current_step_def['role']}' role to approve at this step"
        )

    next_step = REVIEW_STEPS[current_step_num + 1]
    _update_submission(vcode, quarter, next_step["status"], next_step["step"])
    _add_note(vcode, quarter, user_id, username, current_step_def["role"], "approve", note_text)

    # When CEO approves (step 4 → step 5 approved), save snapshot
    if next_step["status"] == "approved":
        _save_snapshot(vcode, quarter, username)

    return get_submission(vcode, quarter)


def return_to_draft(vcode: str, quarter: str, user_id: int, username: str,
                    note_text: str) -> dict:
    """Return document to draft status. Note is required."""
    if not note_text or not note_text.strip():
        raise ValueError("A note is required when returning a document")

    sub = _ensure_submission(vcode, quarter, user_id)
    current_step_num = sub["current_step"]

    if current_step_num < 1:
        raise ValueError("Document is already in draft")

    current_step_def = _step_for(current_step_num)
    user_roles = get_user_review_roles(user_id)
    if current_step_def["role"] not in user_roles:
        raise PermissionError(
            f"You need the '{current_step_def['role']}' role to return at this step"
        )

    _update_submission(vcode, quarter, "returned", 0, returned_to_step=current_step_num)
    _add_note(vcode, quarter, user_id, username, current_step_def["role"], "return", note_text)
    return get_submission(vcode, quarter)


def add_note(vcode: str, quarter: str, user_id: int, username: str,
             note_text: str) -> dict:
    """Add a discussion note (any participant)."""
    if not note_text or not note_text.strip():
        raise ValueError("Note text is required")

    _ensure_submission(vcode, quarter, user_id)
    user_roles = get_user_review_roles(user_id)
    role = user_roles[0] if user_roles else None
    _add_note(vcode, quarter, user_id, username, role, "note", note_text)
    return get_submission(vcode, quarter)


def acknowledge_note(note_id: int, user_id: int, username: str) -> dict:
    """Mark a reviewer's note as addressed. Only asset managers can do this."""
    user_roles = get_user_review_roles(user_id)
    if "asset_manager" not in user_roles:
        raise PermissionError("Only asset managers can mark notes as addressed")

    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT id, vcode, quarter, review_role, action, addressed FROM review_notes WHERE id = :id"),
            {"id": note_id},
        ).mappings().fetchone()

    if not row:
        raise ValueError("Note not found")

    # Only reviewer notes (return/approve/note from non-asset_manager roles) can be addressed
    if row["review_role"] == "asset_manager" and row["action"] not in ("return",):
        raise ValueError("Cannot acknowledge your own notes")

    with engine.begin() as conn:
        conn.execute(
            text("""
                UPDATE review_notes
                SET addressed = :val, addressed_by = :by, addressed_at = CURRENT_TIMESTAMP
                WHERE id = :id
            """),
            {"val": 1 if not row["addressed"] else 0,
             "by": username if not row["addressed"] else None,
             "id": note_id},
        )

    return get_submission(row["vcode"], row["quarter"])


def is_editable(vcode: str, quarter: str) -> bool:
    """Check if comments can be edited (locked only after final approval)."""
    _ensure_tables()
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT status FROM review_submissions WHERE vcode = :v AND quarter = :q"),
            {"v": vcode, "q": quarter},
        ).fetchone()
    if row is None:
        return True  # No submission yet = draft
    return row[0] != "approved"


# ── Snapshots ────────────────────────────────────────────────

def _save_snapshot(vcode: str, quarter: str, approved_by: str):
    """Capture the current One Pager computed data as a frozen snapshot."""
    try:
        from flask_app.services import data_service
        from flask_app.services.financials_service import get_one_pager_data, get_one_pager_chart
        from flask_app.serializers import safe_json

        data = data_service.get_data()
        op_data = get_one_pager_data(
            vcode, quarter, data["inv"], data["isbs_raw"],
            data["mri_loans_raw"], data["mri_val"],
            data["wf"], data["acct"],
            occupancy_raw=data["occupancy_raw"],
            budget_econ_occ=data.get("budget_econ_occ"),
            deal_terms=data.get("deal_terms_raw"),
            at_close_noi=data.get("at_close_noi_raw"),
            event_dates=data.get("event_dates_raw"),
            full_data=data,
            relationships=data.get("relationships_raw"),
        )
        chart_data = get_one_pager_chart(
            vcode, data["isbs_raw"], data["occupancy_raw"], quarter=quarter,
            inv=data["inv"])

        snapshot = safe_json({"data": op_data, "chart": chart_data})
        snapshot_json = json.dumps(snapshot)

        engine = get_engine()
        with engine.begin() as conn:
            # Upsert: delete existing then insert (cross-DB compatible)
            conn.execute(
                text("DELETE FROM one_pager_snapshots WHERE vcode = :v AND quarter = :q"),
                {"v": vcode, "q": quarter},
            )
            conn.execute(
                text("""
                    INSERT INTO one_pager_snapshots (vcode, quarter, snapshot_data, approved_by)
                    VALUES (:v, :q, :data, :by)
                """),
                {"v": vcode, "q": quarter, "data": snapshot_json, "by": approved_by},
            )
        logger.info("Saved One Pager snapshot for %s %s", vcode, quarter)
    except Exception:
        logger.exception("Failed to save One Pager snapshot for %s %s", vcode, quarter)


def get_snapshot(vcode: str, quarter: str) -> dict | None:
    """Retrieve a frozen One Pager snapshot. Returns None if not found."""
    _ensure_tables()
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("""
                SELECT snapshot_data, approved_by, approved_at
                FROM one_pager_snapshots
                WHERE vcode = :v AND quarter = :q
            """),
            {"v": vcode, "q": quarter},
        ).mappings().fetchone()
    if not row:
        return None
    return {
        "snapshot": json.loads(row["snapshot_data"]),
        "approved_by": row["approved_by"],
        "approved_at": row["approved_at"].isoformat() if hasattr(row["approved_at"], "isoformat") else str(row["approved_at"]),
    }


# ── Tracking ─────────────────────────────────────────────────

def get_tracking_data(quarter_filter: str | None = None,
                      status_filter: str | None = None,
                      investor_filter: str | None = None) -> list[dict]:
    """Get production tracking data for all deals.

    LEFT JOINs deals with submissions so unsubmitted deals appear as 'Draft'.
    Optional investor_filter restricts to deals where the given InvestorID
    is an upstream investor (investor → PPI → deal via relationships).
    """
    _ensure_tables()
    engine = get_engine()

    # Build query — double-quote mixed-case columns for PostgreSQL compatibility
    sql = """
        SELECT
            d.vcode,
            d."Investment_Name" as deal_name,
            COALESCE(d."Sale_Status", '') as sale_status,
            COALESCE(rs.quarter, :default_quarter) as quarter,
            COALESCE(rs.status, 'draft') as status,
            COALESCE(rs.current_step, 0) as current_step,
            rs.updated_at,
            rs.submitted_by,
            COALESCE(nc.addressable_count, 0) as addressable_count,
            COALESCE(nc.addressed_count, 0) as addressed_count
        FROM deals d
        LEFT JOIN review_submissions rs ON rs.vcode = d.vcode
        LEFT JOIN (
            SELECT vcode, quarter,
                   COUNT(*) as addressable_count,
                   SUM(CASE WHEN addressed = 1 THEN 1 ELSE 0 END) as addressed_count
            FROM review_notes
            WHERE action IN ('return', 'note')
              AND (review_role IS NOT NULL AND review_role != 'asset_manager')
            GROUP BY vcode, quarter
        ) nc ON nc.vcode = d.vcode AND nc.quarter = rs.quarter
    """
    params: dict = {"default_quarter": quarter_filter or ""}

    conditions = []
    if quarter_filter:
        conditions.append("(rs.quarter = :qf OR rs.quarter IS NULL)")
        params["qf"] = quarter_filter
    if status_filter:
        if status_filter == "draft":
            conditions.append("(rs.status = 'draft' OR rs.status IS NULL)")
        else:
            conditions.append("rs.status = :sf")
            params["sf"] = status_filter
    if investor_filter:
        # Recursive traversal through active relationships only (EndDate NULL = current)
        conditions.append("""
            TRIM(d."InvestmentID") IN (
                WITH RECURSIVE reachable AS (
                    SELECT TRIM(r."InvestmentID") as investment_id
                    FROM relationships r
                    WHERE TRIM(r."InvestorID") = :inv
                      AND (COALESCE(CAST(r."EndDate" AS TEXT), '') = '')
                    UNION
                    SELECT TRIM(r."InvestmentID")
                    FROM relationships r
                    JOIN reachable rc ON TRIM(r."InvestorID") = rc.investment_id
                    WHERE COALESCE(CAST(r."EndDate" AS TEXT), '') = ''
                )
                SELECT investment_id FROM reachable
            )
        """)
        params["inv"] = investor_filter

    # Exclude child properties (but keep parent portfolio deals and sold deals)
    conditions.append("""d.vcode NOT IN (
        SELECT d2.vcode FROM deals d2
        JOIN relationships r ON TRIM(r."InvestmentID") = TRIM(d2."InvestmentID")
        JOIN deals d3 ON TRIM(d3."InvestmentID") = TRIM(r."InvestorID")
        WHERE d2."Portfolio_Name" IS NOT NULL AND d2."Portfolio_Name" != ''
          AND d2."Portfolio_Name" = d3."Portfolio_Name"
          AND d2.vcode != d3.vcode
    )""")

    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += """ ORDER BY d."Investment_Name" """

    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).mappings().fetchall()

    results = []
    for r in rows:
        row_dict = dict(r)
        step_info = _step_for(row_dict["current_step"])
        row_dict["step_label"] = step_info["label"]
        results.append(row_dict)
    return results


def get_investor_list() -> list[str]:
    """Get distinct upstream investor IDs that invest into active deals.

    Recursively traces ownership chains from deals upward through any number
    of intermediate entities (e.g. deal ← PPI ← KOCTRS ← PSCKOC).
    Excludes OP (operating partner) and PPI entities from the investor list.
    """
    engine = get_engine()
    sql = """
        WITH RECURSIVE upstream AS (
            SELECT TRIM(r."InvestorID") as investor_id,
                   TRIM(r."InvestmentID") as investment_id
            FROM relationships r
            JOIN deals d ON TRIM(d."InvestmentID") = TRIM(r."InvestmentID")
            WHERE (COALESCE(CAST(r."EndDate" AS TEXT), '') = '')
              AND d.vcode NOT IN (
                  SELECT d2.vcode FROM deals d2
                  JOIN relationships r2 ON TRIM(r2."InvestmentID") = TRIM(d2."InvestmentID")
                  JOIN deals d3 ON TRIM(d3."InvestmentID") = TRIM(r2."InvestorID")
                  WHERE d2."Portfolio_Name" IS NOT NULL AND d2."Portfolio_Name" != ''
                    AND d2."Portfolio_Name" = d3."Portfolio_Name"
                    AND d2.vcode != d3.vcode
              )
            UNION
            SELECT TRIM(r."InvestorID"), TRIM(r."InvestmentID")
            FROM relationships r
            JOIN upstream u ON TRIM(r."InvestmentID") = u.investor_id
            WHERE COALESCE(CAST(r."EndDate" AS TEXT), '') = ''
        )
        SELECT DISTINCT investor_id
        FROM upstream
        WHERE investor_id NOT LIKE 'OP%'
          AND investor_id NOT LIKE 'PPI%'
        ORDER BY investor_id
    """
    with engine.connect() as conn:
        rows = conn.execute(text(sql)).fetchall()
    return [r[0] for r in rows]
