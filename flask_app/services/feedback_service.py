"""Feedback & request tracking service.

Provides CRUD for user-submitted requests (errors, improvements, reports,
analysis) and email communication with reply tracking.

Database tables:
  - user_requests: Main request records
  - user_request_messages: Thread of messages (user submissions, admin
    emails, user email replies)
"""

import json
import logging
import secrets
from datetime import datetime, timezone

from sqlalchemy import text
from flask_app.db import get_engine

log = logging.getLogger(__name__)

REQUEST_TYPES = ["error", "improvement", "report", "analysis"]
REQUEST_STATUSES = ["open", "in_progress", "resolved", "closed"]
PRIORITY_LEVELS = ["low", "medium", "high"]


# ── Table creation ──────────────────────────────────────────

def _ensure_feedback_tables():
    """Create feedback tables if they don't exist."""
    engine = get_engine()
    dialect = engine.dialect.name

    if dialect == "postgresql":
        pk = "SERIAL PRIMARY KEY"
    else:
        pk = "INTEGER PRIMARY KEY AUTOINCREMENT"

    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS user_requests (
                id              {pk},
                user_id         INTEGER NOT NULL,
                username        TEXT NOT NULL,
                request_type    TEXT NOT NULL,
                title           TEXT NOT NULL,
                description     TEXT NOT NULL,
                priority        TEXT NOT NULL DEFAULT 'medium',
                status          TEXT NOT NULL DEFAULT 'open',
                page_context    TEXT,
                deal_context    TEXT,
                reply_token     TEXT UNIQUE,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved_at     TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS user_request_messages (
                id              {pk},
                request_id      INTEGER NOT NULL,
                sender_type     TEXT NOT NULL,
                sender_name     TEXT NOT NULL,
                message         TEXT NOT NULL,
                sent_via        TEXT NOT NULL DEFAULT 'app',
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (request_id) REFERENCES user_requests(id)
            )
        """))


# ── CRUD ────────────────────────────────────────────────────

def create_request(user_id: int, username: str, request_type: str,
                   title: str, description: str, priority: str = "medium",
                   page_context: str = None, deal_context: str = None) -> dict:
    """Create a new user request. Returns the created record."""
    _ensure_feedback_tables()
    if request_type not in REQUEST_TYPES:
        raise ValueError(f"Invalid type: {request_type}")
    if priority not in PRIORITY_LEVELS:
        raise ValueError(f"Invalid priority: {priority}")

    reply_token = secrets.token_urlsafe(32)
    engine = get_engine()

    with engine.begin() as conn:
        result = conn.execute(text("""
            INSERT INTO user_requests
                (user_id, username, request_type, title, description,
                 priority, page_context, deal_context, reply_token)
            VALUES
                (:uid, :uname, :rtype, :title, :desc,
                 :priority, :page, :deal, :token)
        """), {
            "uid": user_id, "uname": username, "rtype": request_type,
            "title": title, "desc": description, "priority": priority,
            "page": page_context, "deal": deal_context, "token": reply_token,
        })

        # Get the ID of the inserted row
        if engine.dialect.name == "postgresql":
            # PostgreSQL SERIAL — query max id (safe within transaction)
            row = conn.execute(text(
                "SELECT id FROM user_requests WHERE reply_token = :token"
            ), {"token": reply_token}).fetchone()
            req_id = row[0]
        else:
            req_id = result.lastrowid

        # Add the initial description as the first message
        conn.execute(text("""
            INSERT INTO user_request_messages
                (request_id, sender_type, sender_name, message, sent_via)
            VALUES (:rid, 'user', :uname, :msg, 'app')
        """), {"rid": req_id, "uname": username, "msg": description})

    return get_request(req_id)


def get_request(request_id: int) -> dict:
    """Get a single request with its messages."""
    _ensure_feedback_tables()
    engine = get_engine()

    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT * FROM user_requests WHERE id = :id"
        ), {"id": request_id}).fetchone()
        if not row:
            raise ValueError(f"Request {request_id} not found")

        cols = row._mapping
        req = {k: _serialize(v) for k, v in cols.items()}

        msgs = conn.execute(text(
            "SELECT * FROM user_request_messages WHERE request_id = :rid ORDER BY created_at"
        ), {"rid": request_id}).fetchall()
        req["messages"] = [
            {k: _serialize(v) for k, v in m._mapping.items()}
            for m in msgs
        ]

    return req


def list_requests(user_id: int = None, status: str = None,
                  request_type: str = None) -> list:
    """List requests, optionally filtered."""
    _ensure_feedback_tables()
    engine = get_engine()

    query = "SELECT * FROM user_requests WHERE 1=1"
    params = {}
    if user_id is not None:
        query += " AND user_id = :uid"
        params["uid"] = user_id
    if status:
        query += " AND status = :status"
        params["status"] = status
    if request_type:
        query += " AND request_type = :rtype"
        params["rtype"] = request_type
    query += " ORDER BY created_at DESC"

    with engine.connect() as conn:
        rows = conn.execute(text(query), params).fetchall()
        results = []
        for row in rows:
            r = {k: _serialize(v) for k, v in row._mapping.items()}
            # Count messages
            cnt = conn.execute(text(
                "SELECT COUNT(*) FROM user_request_messages WHERE request_id = :rid"
            ), {"rid": r["id"]}).fetchone()
            r["message_count"] = cnt[0]
            results.append(r)

    return results


def update_request_status(request_id: int, status: str,
                          admin_name: str = None) -> dict:
    """Update request status. Admin only."""
    if status not in REQUEST_STATUSES:
        raise ValueError(f"Invalid status: {status}")

    engine = get_engine()
    now = datetime.now(timezone.utc).isoformat()
    resolved = now if status in ("resolved", "closed") else None

    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE user_requests
            SET status = :status, updated_at = :now, resolved_at = :resolved
            WHERE id = :id
        """), {"status": status, "now": now, "resolved": resolved, "id": request_id})

        if admin_name:
            conn.execute(text("""
                INSERT INTO user_request_messages
                    (request_id, sender_type, sender_name, message, sent_via)
                VALUES (:rid, 'system', :name, :msg, 'app')
            """), {
                "rid": request_id, "name": admin_name,
                "msg": f"Status changed to {status}",
            })

    return get_request(request_id)


def add_message(request_id: int, sender_type: str, sender_name: str,
                message: str, sent_via: str = "app") -> dict:
    """Add a message to a request thread."""
    engine = get_engine()

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO user_request_messages
                (request_id, sender_type, sender_name, message, sent_via)
            VALUES (:rid, :stype, :sname, :msg, :via)
        """), {
            "rid": request_id, "stype": sender_type,
            "sname": sender_name, "msg": message, "via": sent_via,
        })
        conn.execute(text(
            "UPDATE user_requests SET updated_at = :now WHERE id = :id"
        ), {"now": datetime.now(timezone.utc).isoformat(), "id": request_id})

    return get_request(request_id)


# ── Email ───────────────────────────────────────────────────

def send_request_email(request_id: int, admin_name: str,
                       message: str, subject: str = None) -> bool:
    """Send an email to the user who submitted a request.

    The email includes a reply link back to the app.
    Also records the outbound message in the thread.
    """
    from flask import current_app
    from flask_app.auth.email_utils import send_email

    req = get_request(request_id)
    engine = get_engine()

    # Look up user email
    with engine.connect() as conn:
        user = conn.execute(text(
            "SELECT email, username FROM users WHERE id = :uid"
        ), {"uid": req["user_id"]}).fetchone()
        if not user or not user[0]:
            log.warning("No email for user %s — message not sent", req["username"])
            return False

    user_email = user[0]
    user_name = user[1]
    app_url = current_app.config.get("APP_URL", "")
    reply_token = req.get("reply_token", "")
    reply_link = f"{app_url}/feedback?reply={reply_token}"

    if not subject:
        type_label = req["request_type"].replace("_", " ").title()
        subject = f"RE: [{type_label}] {req['title']}"

    html = f"""
    <div style="font-family: Arial, sans-serif; max-width: 560px; margin: 0 auto;">
        <h2 style="color: #1d4e7e; font-size: 16px;">
            Waterfall XIRR — Request Update
        </h2>
        <p>Hi <strong>{user_name}</strong>,</p>
        <p>Regarding your request: <strong>{req['title']}</strong></p>
        <div style="background: #f5f7fa; border-left: 3px solid #1d4e7e;
                    padding: 12px 16px; margin: 16px 0; border-radius: 4px;">
            {message.replace(chr(10), '<br>')}
        </div>
        <p style="margin: 20px 0;">
            <a href="{reply_link}"
               style="background: #1d4e7e; color: white; padding: 10px 20px;
                      text-decoration: none; border-radius: 6px;
                      display: inline-block; font-size: 14px;">
                View &amp; Reply
            </a>
        </p>
        <p style="font-size: 12px; color: #888;">
            Request #{req['id']} | Status: {req['status'].replace('_', ' ').title()}
        </p>
    </div>
    """

    sent = send_email(user_email, subject, html)

    # Record the outbound message regardless
    add_message(request_id, "admin", admin_name, message, "email")

    return sent


def get_request_by_token(reply_token: str) -> dict:
    """Look up a request by its reply token."""
    _ensure_feedback_tables()
    engine = get_engine()

    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT id FROM user_requests WHERE reply_token = :token"
        ), {"token": reply_token}).fetchone()
        if not row:
            raise ValueError("Invalid reply token")

    return get_request(row[0])


def handle_inbound_email(reply_token: str, from_email: str,
                         body_text: str) -> dict:
    """Process an inbound email reply (from SendGrid Inbound Parse or similar).

    Matches the reply token to a request and adds the reply as a message.
    """
    req = get_request_by_token(reply_token)

    # Verify the sender matches the request owner
    engine = get_engine()
    with engine.connect() as conn:
        user = conn.execute(text(
            "SELECT email FROM users WHERE id = :uid"
        ), {"uid": req["user_id"]}).fetchone()

    sender_name = req["username"]
    if user and user[0] and user[0].lower() != from_email.lower():
        log.warning("Inbound email from %s doesn't match user %s (%s)",
                     from_email, sender_name, user[0])
        # Still accept it but log the mismatch

    return add_message(req["id"], "user", sender_name, body_text, "email")


# ── Export for design sessions ──────────────────────────────

def export_all_requests() -> list:
    """Export all requests with full message threads.

    Designed for consumption during Claude design sessions — provides
    complete context of user feedback, errors, and feature requests.
    """
    _ensure_feedback_tables()
    engine = get_engine()

    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT * FROM user_requests ORDER BY created_at DESC"
        )).fetchall()

        results = []
        for row in rows:
            r = {k: _serialize(v) for k, v in row._mapping.items()}
            msgs = conn.execute(text(
                "SELECT * FROM user_request_messages "
                "WHERE request_id = :rid ORDER BY created_at"
            ), {"rid": r["id"]}).fetchall()
            r["messages"] = [
                {k: _serialize(v) for k, v in m._mapping.items()}
                for m in msgs
            ]
            results.append(r)

    return results


# ── Helpers ─────────────────────────────────────────────────

def _serialize(val):
    """Convert datetime objects to ISO strings for JSON."""
    if isinstance(val, datetime):
        return val.isoformat()
    return val
