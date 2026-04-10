"""Migrate SQLite data to PostgreSQL."""
import sqlite3
import psycopg2
import psycopg2.extras
import time

SQLITE_PATH = "C:/Users/jbruin/Documents/GitHub/waterfall-xirr/waterfall.db"
PG_URL = "postgresql://wfadmin:Wf3d9097e0365c445456dcc52e!@psql-waterfall-dev.postgres.database.azure.com:5432/waterfall_xirr?sslmode=require"

SKIP_TABLES = {"sqlite_sequence", "calculation_cache"}

sq = sqlite3.connect(SQLITE_PATH)
sq.row_factory = sqlite3.Row
pg = psycopg2.connect(PG_URL)
pg.autocommit = False

tables = [r[0] for r in sq.execute(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
).fetchall() if r[0] not in SKIP_TABLES]

print(f"Migrating {len(tables)} tables...")

for table in tables:
    start = time.time()
    cols_info = sq.execute(f"PRAGMA table_info([{table}])").fetchall()
    col_names = [c["name"] for c in cols_info]
    row_count = sq.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]

    def pg_type(sqlite_type, is_pk):
        t = (sqlite_type or "").upper()
        if is_pk and t == "INTEGER":
            return "SERIAL PRIMARY KEY"
        if "INT" in t:
            return "BIGINT"
        if "REAL" in t or "FLOAT" in t or "DOUBLE" in t or "NUMERIC" in t:
            return "DOUBLE PRECISION"
        if "BOOL" in t:
            return "BOOLEAN"
        if "BLOB" in t:
            return "BYTEA"
        return "TEXT"

    col_defs = []
    for c in cols_info:
        ptype = pg_type(c["type"], c["pk"] == 1)
        nullable = "" if c["notnull"] == 0 or "PRIMARY KEY" in ptype else " NOT NULL"
        col_defs.append(f'"{c["name"]}" {ptype}{nullable}')

    create_sql = f'CREATE TABLE IF NOT EXISTS "{table}" ({", ".join(col_defs)})'
    cur = pg.cursor()

    if table == "users":
        print(f"  {table}: SKIPPED (already exists)")
        cur.close()
        continue

    cur.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE')
    cur.execute(create_sql)

    if row_count == 0:
        pg.commit()
        print(f"  {table}: 0 rows ({time.time()-start:.1f}s)")
        cur.close()
        continue

    rows = sq.execute(f"SELECT * FROM [{table}]").fetchall()
    # Use explicit column selection to avoid mismatches
    actual_cols = len(rows[0]) if rows else len(col_names)
    col_names = col_names[:actual_cols]
    placeholders = ", ".join(["%s"] * actual_cols)
    quoted_cols = ", ".join([f'"{c}"' for c in col_names])
    insert_sql = f'INSERT INTO "{table}" ({quoted_cols}) VALUES ({placeholders})'

    batch_size = 5000
    for i in range(0, len(rows), batch_size):
        batch = [tuple(row)[:actual_cols] for row in rows[i:i+batch_size]]
        try:
            psycopg2.extras.execute_batch(cur, insert_sql, batch, page_size=1000)
        except Exception as e:
            pg.rollback()
            # Fall back to row-by-row insert
            print(f"    Batch failed ({e}), trying row-by-row...")
            cur2 = pg.cursor()
            cur2.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE')
            cur2.execute(create_sql)
            ok = 0
            for row in rows:
                try:
                    cur2.execute(insert_sql, tuple(row)[:actual_cols])
                    ok += 1
                except Exception:
                    pg.rollback()
                    cur2 = pg.cursor()
                    cur2.execute(f'SAVEPOINT sp')
                    continue
            pg.commit()
            print(f"    Inserted {ok}/{len(rows)} rows")
            cur2.close()
            cur.close()
            elapsed = time.time() - start
            print(f"  {table}: {ok} rows ({elapsed:.1f}s)")
            continue

    pg.commit()
    elapsed = time.time() - start
    print(f"  {table}: {row_count} rows ({elapsed:.1f}s)")
    cur.close()

# Reset sequences for SERIAL columns
cur = pg.cursor()
for table in tables:
    if table == "users":
        continue
    cols_info = sq.execute(f"PRAGMA table_info([{table}])").fetchall()
    for c in cols_info:
        if c["pk"] == 1 and "INT" in (c["type"] or "").upper():
            col_name = c["name"]
            try:
                cur.execute(f'SELECT MAX("{col_name}") FROM "{table}"')
                max_val = cur.fetchone()[0]
                if max_val:
                    seq = f"{table}_{col_name}_seq"
                    cur.execute(f"SELECT setval('{seq}', {max_val})")
            except Exception:
                pg.rollback()
pg.commit()
cur.close()

sq.close()
pg.close()
print("\nMigration complete!")
