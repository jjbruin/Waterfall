"""Permissions INSIDE a section, granted by username (Board plan, Oct 5 2026).

A section says where a user may GO; these say what they may DO there. Like the
Board section itself they are OPT-IN: no row means no permission, the admin
USERNAME holds all of them, and the admin ROLE holds none -- Jim: "the admin
role, which developers hold, grants neither". Only the admin username grants
them, and every change is written to the access log.

A permission REQUIRES ITS SECTION. Removing someone's Board access, or letting
an advisor's end date pass, takes every Board permission with it -- a grant that
outlived the section would be a key to a door that is no longer there, and the
next person to re-open the door would find it already unlocked.

Some permissions IMPLY others (editing salary planning implies viewing it; the
package builder is a Board editor). The implication is applied on READ, so
revoking the stronger one never leaves the weaker one behind as an orphan row.
"""
from __future__ import annotations

from sqlalchemy import text

from flask_app.auth.sections import SUPERUSER, denied_sections

PERMISSIONS = (
    {"key": "board_edit", "label": "Board editor", "section": "board",
     "implies": (),
     "describe": "Narrative, as-of dates and non-compensation assumptions"},
    {"key": "board_build", "label": "Package builder", "section": "board",
     "implies": ("board_edit",),
     "describe": "Creates each meeting and (Phase 6) builds and freezes its package"},
    {"key": "comp_view", "label": "Salary planning: view", "section": "board",
     "implies": (),
     "describe": "Position-level salary planning; everyone else sees company totals"},
    {"key": "comp_edit", "label": "Salary planning: edit", "section": "board",
     "implies": ("comp_view",),
     "describe": "Changes the payroll plan"},
)
PERMISSION_KEYS = tuple(p["key"] for p in PERMISSIONS)
_BY_KEY = {p["key"]: p for p in PERMISSIONS}

_READY = set()


def _engine():
    from flask_app.db import get_engine
    return get_engine()


def ensure(engine=None):
    engine = engine or _engine()
    if id(engine) in _READY:
        return engine
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS user_permissions (
                user_id INTEGER NOT NULL,
                permission TEXT NOT NULL,
                granted_by TEXT,
                granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, permission)
            )
        """))
    _READY.add(id(engine))
    return engine


def stored(user_id: int, engine=None) -> set:
    """The rows actually stored for a user -- before implication or the section."""
    engine = ensure(engine)
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT permission FROM user_permissions WHERE user_id = :u"),
            {"u": int(user_id)}).fetchall()
    return {r[0] for r in rows if r[0] in _BY_KEY}


def effective(user: dict | None, engine=None) -> set:
    """What ``user`` may do: their grants, plus what those imply, minus any
    whose section they cannot open. The admin username: everything."""
    if not user:
        return set()
    if user.get("username") == SUPERUSER:
        return set(PERMISSION_KEYS)
    if user.get("id") is None:
        return set()
    held = stored(user["id"], engine)
    for k in list(held):
        held |= set(_BY_KEY[k]["implies"])
    denied = denied_sections(int(user["id"]), engine)
    return {k for k in held if _BY_KEY[k]["section"] not in denied}


def has(user: dict | None, key: str, engine=None) -> bool:
    if key not in _BY_KEY:
        raise KeyError("Unknown permission %r" % key)
    return key in effective(user, engine)


def set_permissions(user_id: int, perms: dict, actor: str, engine=None) -> set:
    """Apply {permission: bool} for a user. Unknown keys raise. Logged per change."""
    unknown = [k for k in perms if k not in _BY_KEY]
    if unknown:
        raise ValueError("Unknown permission(s): %s" % ", ".join(unknown))
    engine = ensure(engine)
    before = stored(user_id, engine)
    with engine.begin() as conn:
        for k, on in perms.items():
            conn.execute(text(
                "DELETE FROM user_permissions WHERE user_id = :u AND permission = :p"),
                {"u": int(user_id), "p": k})
            if on:
                conn.execute(text(
                    "INSERT INTO user_permissions (user_id, permission, granted_by) "
                    "VALUES (:u, :p, :by)"), {"u": int(user_id), "p": k, "by": actor})
    after = stored(user_id, engine)
    from flask_app.auth import audit
    for k in sorted(set(perms)):
        if (k in before) != (k in after):
            audit.log(actor, "permission_granted" if k in after else "permission_removed",
                      target_user_id=user_id, detail={"permission": k}, engine=engine)
    return after


def all_stored(engine=None) -> dict:
    """{user_id: [permission keys]} -- the rows, for the Access screen."""
    engine = ensure(engine)
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT user_id, permission FROM user_permissions")).fetchall()
    out = {}
    for uid, p in rows:
        if p in _BY_KEY:
            out.setdefault(int(uid), []).append(p)
    return out


def forget_user(user_id: int, engine=None):
    """Drop a deleted user's grants, so a recycled id does not inherit them."""
    engine = ensure(engine)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM user_permissions WHERE user_id = :u"),
                     {"u": int(user_id)})
