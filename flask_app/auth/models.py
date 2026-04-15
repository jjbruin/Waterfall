"""User model backed by users table (SQLite or PostgreSQL)."""

import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone
from sqlalchemy import text
from flask_app.db import get_engine


def _ensure_users_table():
    """Create users table if it doesn't exist."""
    engine = get_engine()
    dialect = engine.dialect.name

    if dialect == "postgresql":
        pk = "SERIAL PRIMARY KEY"
    else:
        pk = "INTEGER PRIMARY KEY AUTOINCREMENT"

    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS users (
                id {pk},
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'viewer',
                email TEXT,
                must_change_password BOOLEAN NOT NULL DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS password_reset_tokens (
                id {pk},
                user_id INTEGER NOT NULL,
                token TEXT UNIQUE NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                used BOOLEAN NOT NULL DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """))

    # Migrate existing users table: add email and must_change_password if missing
    _migrate_users_columns(engine)


def _migrate_users_columns(engine):
    """Add email and must_change_password columns if they don't exist."""
    with engine.connect() as conn:
        # Check existing columns
        dialect = engine.dialect.name
        if dialect == "postgresql":
            cols = conn.execute(text(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'users'"
            )).fetchall()
            col_names = {r[0] for r in cols}
        else:
            cols = conn.execute(text("PRAGMA table_info(users)")).fetchall()
            col_names = {r[1] for r in cols}

    with engine.begin() as conn:
        if "email" not in col_names:
            conn.execute(text("ALTER TABLE users ADD COLUMN email TEXT"))
        if "must_change_password" not in col_names:
            conn.execute(text(
                "ALTER TABLE users ADD COLUMN must_change_password BOOLEAN NOT NULL DEFAULT FALSE"
            ))


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()


def create_user(username: str, password: str, role: str = "viewer",
                email: str | None = None, must_change_password: bool = False) -> dict | None:
    """Create a new user. Returns user dict or None if username taken."""
    _ensure_users_table()
    salt = os.urandom(16).hex()
    pw_hash = _hash_password(password, salt)
    engine = get_engine()
    try:
        with engine.begin() as conn:
            conn.execute(
                text("""INSERT INTO users (username, password_hash, salt, role, email, must_change_password)
                        VALUES (:u, :h, :s, :r, :e, :m)"""),
                {"u": username, "h": pw_hash, "s": salt, "r": role,
                 "e": email, "m": must_change_password},
            )
        return {"username": username, "role": role, "email": email}
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
    d = dict(row)
    return {
        "id": d["id"],
        "username": d["username"],
        "role": d["role"],
        "must_change_password": bool(d.get("must_change_password", False)),
    }


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
    d = dict(row)
    return {
        "id": d["id"],
        "username": d["username"],
        "role": d["role"],
        "must_change_password": bool(d.get("must_change_password", False)),
    }


def get_user_by_email(email: str) -> dict | None:
    """Look up user by email."""
    _ensure_users_table()
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM users WHERE email = :e"), {"e": email}
        ).mappings().fetchone()
    if row is None:
        return None
    return {"id": row["id"], "username": row["username"], "role": row["role"],
            "email": row["email"]}


def list_users() -> list[dict]:
    """List all users (id, username, role, email, created_at)."""
    _ensure_users_table()
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT id, username, role, email, must_change_password, created_at FROM users ORDER BY id")
        ).mappings().fetchall()
    return [
        {"id": r["id"], "username": r["username"], "role": r["role"],
         "email": r.get("email"), "must_change_password": bool(r.get("must_change_password", False)),
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


def update_user_email(user_id: int, email: str) -> bool:
    """Update a user's email. Returns True on success."""
    engine = get_engine()
    with engine.begin() as conn:
        result = conn.execute(
            text("UPDATE users SET email = :e WHERE id = :id"),
            {"e": email, "id": user_id},
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


def change_password(user_id: int, new_password: str, clear_must_change: bool = True) -> bool:
    """Change a user's password. Returns True on success."""
    salt = os.urandom(16).hex()
    pw_hash = _hash_password(new_password, salt)
    engine = get_engine()
    with engine.begin() as conn:
        if clear_must_change:
            result = conn.execute(
                text("UPDATE users SET password_hash = :h, salt = :s, must_change_password = FALSE WHERE id = :id"),
                {"h": pw_hash, "s": salt, "id": user_id},
            )
        else:
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


# ── Password reset tokens ────────────────────────────────────────────

def create_reset_token(user_id: int) -> str:
    """Create a password reset token valid for 1 hour. Returns the token string."""
    token = secrets.token_urlsafe(32)
    expires = datetime.now(timezone.utc) + timedelta(hours=1)
    engine = get_engine()
    with engine.begin() as conn:
        # Invalidate any existing unused tokens for this user
        conn.execute(
            text("UPDATE password_reset_tokens SET used = TRUE WHERE user_id = :uid AND used = FALSE"),
            {"uid": user_id},
        )
        conn.execute(
            text("""INSERT INTO password_reset_tokens (user_id, token, expires_at)
                    VALUES (:uid, :tok, :exp)"""),
            {"uid": user_id, "tok": token, "exp": expires},
        )
    return token


def validate_reset_token(token: str) -> dict | None:
    """Validate a reset token. Returns user dict if valid, None otherwise."""
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("""SELECT prt.user_id, prt.expires_at, u.username, u.email
                    FROM password_reset_tokens prt
                    JOIN users u ON u.id = prt.user_id
                    WHERE prt.token = :tok AND prt.used = FALSE"""),
            {"tok": token},
        ).mappings().fetchone()
    if row is None:
        return None
    expires = row["expires_at"]
    if isinstance(expires, str):
        expires = datetime.fromisoformat(expires)
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    if datetime.now(timezone.utc) > expires:
        return None
    return {"user_id": row["user_id"], "username": row["username"], "email": row["email"]}


def consume_reset_token(token: str) -> bool:
    """Mark a reset token as used."""
    engine = get_engine()
    with engine.begin() as conn:
        result = conn.execute(
            text("UPDATE password_reset_tokens SET used = TRUE WHERE token = :tok"),
            {"tok": token},
        )
        return result.rowcount > 0
