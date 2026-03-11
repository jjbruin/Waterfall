"""User model backed by users table (SQLite or PostgreSQL)."""

import hashlib
import os
from sqlalchemy import text
from flask_app.db import get_engine


def _ensure_users_table():
    """Create users table if it doesn't exist."""
    engine = get_engine()
    dialect = engine.dialect.name

    if dialect == "postgresql":
        ddl = """
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'viewer',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
    else:
        ddl = """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'viewer',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """

    with engine.begin() as conn:
        conn.execute(text(ddl))


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()


def create_user(username: str, password: str, role: str = "viewer") -> dict | None:
    """Create a new user. Returns user dict or None if username taken."""
    _ensure_users_table()
    salt = os.urandom(16).hex()
    pw_hash = _hash_password(password, salt)
    engine = get_engine()
    try:
        with engine.begin() as conn:
            conn.execute(
                text("INSERT INTO users (username, password_hash, salt, role) VALUES (:u, :h, :s, :r)"),
                {"u": username, "h": pw_hash, "s": salt, "r": role},
            )
        return {"username": username, "role": role}
    except Exception:
        return None


def authenticate(username: str, password: str) -> dict | None:
    """Verify credentials. Returns user dict or None."""
    _ensure_users_table()
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM users WHERE username = :u"), {"u": username}
        ).mappings().fetchone()
    if row is None:
        return None
    pw_hash = _hash_password(password, dict(row)["salt"])
    if pw_hash != dict(row)["password_hash"]:
        return None
    return {"id": row["id"], "username": row["username"], "role": row["role"]}


def get_user_by_id(user_id: int) -> dict | None:
    """Look up user by ID."""
    _ensure_users_table()
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM users WHERE id = :id"), {"id": user_id}
        ).mappings().fetchone()
    if row is None:
        return None
    return {"id": row["id"], "username": row["username"], "role": row["role"]}


def list_users() -> list[dict]:
    """List all users (id, username, role, created_at)."""
    _ensure_users_table()
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT id, username, role, created_at FROM users ORDER BY id")
        ).mappings().fetchall()
    return [
        {"id": r["id"], "username": r["username"], "role": r["role"],
         "created_at": r["created_at"]}
        for r in rows
    ]


def update_user_role(user_id: int, role: str) -> bool:
    """Update a user's role. Returns True on success."""
    engine = get_engine()
    with engine.begin() as conn:
        result = conn.execute(
            text("UPDATE users SET role = :r WHERE id = :id"),
            {"r": role, "id": user_id},
        )
        return result.rowcount > 0


def delete_user(user_id: int) -> bool:
    """Delete a user. Returns True on success."""
    engine = get_engine()
    with engine.begin() as conn:
        result = conn.execute(
            text("DELETE FROM users WHERE id = :id"), {"id": user_id}
        )
        return result.rowcount > 0


def change_password(user_id: int, new_password: str) -> bool:
    """Change a user's password. Returns True on success."""
    salt = os.urandom(16).hex()
    pw_hash = _hash_password(new_password, salt)
    engine = get_engine()
    with engine.begin() as conn:
        result = conn.execute(
            text("UPDATE users SET password_hash = :h, salt = :s WHERE id = :id"),
            {"h": pw_hash, "s": salt, "id": user_id},
        )
        return result.rowcount > 0


def ensure_default_admin():
    """Create default admin user if no users exist."""
    _ensure_users_table()
    engine = get_engine()
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM users")).scalar()
    if count == 0:
        create_user("admin", "admin", role="admin")
