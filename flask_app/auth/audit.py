"""The access log: who granted what, and who read or changed sensitive detail.

Board plan, Oct 5 2026: "every read, export and change of compensation detail is
logged -- who, when, what -- and the log is visible to the admin account", and
every grant of a section or permission is logged. Append-only from the app: there
is no update or delete path here. The table is NEVER exposed through Data
Explorer, the export or the assistant (``sections.RESTRICTED_TABLES``); the admin
username reads it on Board > Access.
"""
from __future__ import annotations

import json
import logging

from sqlalchemy import text

logger = logging.getLogger(__name__)

_READY = set()


def _engine():
    from flask_app.db import get_engine
    return get_engine()


def ensure(engine=None):
    engine = engine or _engine()
    if id(engine) in _READY:
        return engine
    pk = ("id SERIAL PRIMARY KEY" if engine.dialect.name == "postgresql"
          else "id INTEGER PRIMARY KEY AUTOINCREMENT")
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS access_audit (
                {pk},
                at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                actor TEXT NOT NULL,
                action TEXT NOT NULL,
                target_user_id INTEGER,
                detail TEXT
            )
        """))
    _READY.add(id(engine))
    return engine


def log(actor: str, action: str, target_user_id: int | None = None,
        detail: dict | None = None, engine=None) -> None:
    """Append one row. A failure to log is RAISED, not swallowed: an action whose
    trail cannot be written must not look as though it was recorded."""
    engine = ensure(engine)
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO access_audit (actor, action, target_user_id, detail) "
            "VALUES (:a, :act, :t, :d)"),
            {"a": actor or "?", "act": action,
             "t": int(target_user_id) if target_user_id is not None else None,
             "d": json.dumps(detail, default=str, sort_keys=True) if detail else None})


def recent(limit: int = 500, engine=None) -> list[dict]:
    engine = ensure(engine)
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT a.id, a.at, a.actor, a.action, a.target_user_id, u.username AS target, "
            "a.detail FROM access_audit a LEFT JOIN users u ON u.id = a.target_user_id "
            "ORDER BY a.id DESC LIMIT :n"), {"n": int(limit)}).mappings().fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["at"] = str(d["at"]) if d["at"] is not None else None
        try:
            d["detail"] = json.loads(d["detail"]) if d["detail"] else None
        except ValueError:
            pass
        out.append(d)
    return out
