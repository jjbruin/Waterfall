"""Migrate data from SQLite to PostgreSQL.

Usage:
    python -m flask_app.migrate_to_postgres

Reads DB_PATH (SQLite source) and DATABASE_URL (PostgreSQL target) from
environment or .env file. Copies all tables from SQLite into PostgreSQL.
"""

import os
import sys
import sqlite3
import pandas as pd
from sqlalchemy import create_engine, text

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def migrate(sqlite_path: str, pg_url: str):
    """Copy all tables from SQLite to PostgreSQL."""
    print(f"Source: {sqlite_path}")
    print(f"Target: {pg_url.split('@')[0].rsplit(':', 1)[0]}@***")  # hide password

    sqlite_conn = sqlite3.connect(sqlite_path)
    pg_engine = create_engine(pg_url)

    # Get all table names from SQLite
    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",
        sqlite_conn,
    )["name"].tolist()

    print(f"\nFound {len(tables)} tables to migrate:")

    for table in sorted(tables):
        try:
            df = pd.read_sql(f"SELECT * FROM [{table}]", sqlite_conn)
            # Use 'replace' to create/overwrite tables in PostgreSQL
            df.to_sql(table, pg_engine, if_exists="replace", index=False)
            print(f"  {table}: {len(df)} rows")
        except Exception as e:
            print(f"  {table}: ERROR - {e}")

    sqlite_conn.close()
    pg_engine.dispose()
    print("\nMigration complete.")


if __name__ == "__main__":
    # Try loading .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    sqlite_path = os.environ.get("DB_PATH", "waterfall.db")
    pg_url = os.environ.get("DATABASE_URL")

    if not pg_url:
        print("ERROR: DATABASE_URL environment variable not set.")
        print("Set it to your PostgreSQL connection string:")
        print("  export DATABASE_URL=postgresql://user:pass@host:5432/waterfall_xirr")
        sys.exit(1)

    if not os.path.exists(sqlite_path):
        print(f"ERROR: SQLite database not found: {sqlite_path}")
        sys.exit(1)

    migrate(sqlite_path, pg_url)
