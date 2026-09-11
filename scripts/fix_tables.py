"""Fix tables that failed migration due to type mismatches or % in data."""
import os
import sqlite3
import psycopg2
import time
import io
import csv

SQLITE_PATH = os.environ.get(
    "SQLITE_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "waterfall.db"),
)

# NEVER hardcode this. The literal that used to sit here was the live `wfadmin`
# password; it reached a PUBLIC repo on 2026-04-10 and stayed readable for five months
# (rotated Sep 11 2026). Read it from the environment, the way flask_app/config.py and
# scripts/event_dates_exit_probe.py already do.
PG_URL = os.environ.get("DATABASE_URL")
if not PG_URL:
    raise SystemExit(
        "DATABASE_URL is not set. Export it first, e.g.\n"
        "  export DATABASE_URL='postgresql://USER:PASS@psql-waterfall-dev."
        "postgres.database.azure.com:5432/waterfall_xirr?sslmode=require'"
    )
if PG_URL.startswith("postgres://"):
    PG_URL = PG_URL.replace("postgres://", "postgresql://", 1)

sq = sqlite3.connect(SQLITE_PATH)
pg = psycopg2.connect(PG_URL)

FIX_TABLES = ["occupancy", "tenants", "prospective_loans"]

for table in FIX_TABLES:
    start = time.time()
    cur = pg.cursor()

    cols = sq.execute(f"PRAGMA table_info([{table}])").fetchall()
    col_names = [c[1] for c in cols]

    # Drop and recreate as all TEXT
    cur.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE')
    col_defs = ", ".join([f'"{name}" TEXT' for name in col_names])
    cur.execute(f'CREATE TABLE "{table}" ({col_defs})')
    pg.commit()

    rows = sq.execute(f"SELECT * FROM [{table}]").fetchall()
    if not rows:
        print(f"  {table}: 0 rows")
        cur.close()
        continue

    # Use COPY with StringIO for fast, safe bulk insert (no % issues)
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter='\t', lineterminator='\n')
    for row in rows:
        writer.writerow(['' if v is None else str(v).replace('\t', ' ').replace('\n', ' ').replace('\r', ' ') for v in row])

    buf.seek(0)
    quoted_cols = ", ".join([f'"{c}"' for c in col_names])
    cur.copy_expert(f"""COPY "{table}" ({quoted_cols}) FROM STDIN WITH (FORMAT text, NULL '')""", buf)
    pg.commit()

    elapsed = time.time() - start
    print(f"  {table}: {len(rows)} rows ({elapsed:.1f}s)")
    cur.close()

# Verify counts
cur = pg.cursor()
for table in FIX_TABLES:
    cur.execute(f'SELECT COUNT(*) FROM "{table}"')
    print(f"  {table} verified: {cur.fetchone()[0]} rows")
cur.close()

pg.close()
sq.close()
print("Fix complete!")
