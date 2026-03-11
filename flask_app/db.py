"""Database engine management for Flask app.

Supports SQLite (default) and PostgreSQL via DATABASE_URL.
Uses SQLAlchemy for connection pooling and dialect abstraction.
"""

from sqlalchemy import create_engine, text, event
from sqlalchemy.engine import Engine
from flask import current_app, g


_engine: Engine | None = None


def get_engine() -> Engine:
    """Get or create the SQLAlchemy engine from app config.

    Reads DATABASE_URL first (PostgreSQL), falls back to DB_PATH (SQLite).
    """
    global _engine
    if _engine is not None:
        return _engine

    database_url = current_app.config.get("DATABASE_URL")
    if database_url:
        _engine = create_engine(
            database_url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
    else:
        db_path = current_app.config["DB_PATH"]
        _engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
        )
        # Enable WAL mode for better concurrent reads
        @event.listens_for(_engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

    return _engine


def get_connection():
    """Get a connection from the engine, stored in Flask g for request lifecycle."""
    if "db_conn" not in g:
        g.db_conn = get_engine().connect()
    return g.db_conn


def close_connection(exception=None):
    """Close the connection at end of request."""
    conn = g.pop("db_conn", None)
    if conn is not None:
        conn.close()


def reset_engine():
    """Dispose engine and clear cache. Call after config changes."""
    global _engine
    if _engine is not None:
        _engine.dispose()
        _engine = None


def init_app(app):
    """Register teardown handler with Flask app."""
    app.teardown_appcontext(close_connection)


def is_postgres() -> bool:
    """Check if we're using PostgreSQL."""
    url = str(get_engine().url)
    return url.startswith("postgresql")
