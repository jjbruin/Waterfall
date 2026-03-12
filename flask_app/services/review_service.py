"""Review workflow service for One Pager approval pipeline."""

from sqlalchemy import text
from flask_app.db import get_engine


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
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS review_roles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                review_role TEXT NOT NULL,
                UNIQUE(user_id, review_role),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS review_submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
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
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS review_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vcode TEXT NOT NULL,
                quarter TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                username TEXT NOT NULL,
                review_role TEXT,
                action TEXT NOT NULL,
                note_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
                SELECT id, user_id, username, review_role, action, note_text, created_at
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


def is_editable(vcode: str, quarter: str) -> bool:
    """Check if comments can be edited (only in draft or returned status)."""
    _ensure_tables()
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT status FROM review_submissions WHERE vcode = :v AND quarter = :q"),
            {"v": vcode, "q": quarter},
        ).fetchone()
    if row is None:
        return True  # No submission yet = draft
    return row[0] in ("draft", "returned")


# ── Tracking ─────────────────────────────────────────────────

def get_tracking_data(quarter_filter: str | None = None,
                      status_filter: str | None = None) -> list[dict]:
    """Get production tracking data for all deals.

    LEFT JOINs deals with submissions so unsubmitted deals appear as 'Draft'.
    """
    _ensure_tables()
    engine = get_engine()

    # Build query
    sql = """
        SELECT
            d.vcode,
            d.Investment_Name as deal_name,
            COALESCE(rs.quarter, :default_quarter) as quarter,
            COALESCE(rs.status, 'draft') as status,
            COALESCE(rs.current_step, 0) as current_step,
            rs.updated_at,
            rs.submitted_by
        FROM deals d
        LEFT JOIN review_submissions rs ON rs.vcode = d.vcode
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

    # Exclude sold deals and child properties
    conditions.append("COALESCE(d.Sale_Status, '') != 'SOLD'")
    conditions.append("COALESCE(d.Portfolio_Name, '') = ''")

    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY d.Investment_Name"

    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).mappings().fetchall()

    results = []
    for r in rows:
        row_dict = dict(r)
        step_info = _step_for(row_dict["current_step"])
        row_dict["step_label"] = step_info["label"]
        results.append(row_dict)
    return results
