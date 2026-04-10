"""Fix tables that failed migration due to type mismatches or % in data."""
import sqlite3
import psycopg2
import time
import io
import csv

SQLITE_PATH = "C:/Users/jbruin/Documents/GitHub/waterfall-xirr/waterfall.db"
PG_URL = "postgresql://wfadmin:Wf3d9097e0365c445456dcc52e!@psql-waterfall-dev.postgres.database.azure.com:5432/waterfall_xirr?sslmode=require"

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
