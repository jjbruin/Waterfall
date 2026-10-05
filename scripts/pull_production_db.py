"""Copy production's PostgreSQL data into the local SQLite database.

Jim, Oct 5 2026: "The production database was updated this morning. Can you copy
over the data from the production data to the local database?"

THE APP'S OWN EXPORT CANNOT DO THIS. ``database.export_all_tables_to_zip`` reads
a SQLite file, so on Azure the Database Tools "Export Database" button exports
the container's own (empty) SQLite file and none of production's PostgreSQL.

Usage, from the repo root, with the local Flask server STOPPED (Windows will not
replace a file another process has open):

    set DATABASE_URL=postgresql://...      (cmd)   /  $env:DATABASE_URL="..." (PowerShell)
    .venv\\Scripts\\python scripts\\pull_production_db.py [--dest waterfall.db] [--no-files]

What it does, and why each part is there:

  * THE CREDENTIAL IS YOURS. It is read from DATABASE_URL and never printed,
    logged or written anywhere.
  * THE LOCAL SCHEMA IS KEPT. It works on a COPY of the current local file and
    replaces each table's ROWS, not its definition. Recreating tables from
    production would lose SQLite's INTEGER PRIMARY KEY AUTOINCREMENT, and the
    next row the app inserts locally would get a NULL id. A column production
    has and the local table lacks is ADDED; a table that does not exist locally
    is created from the data.
  * YOUR LOGINS ARE KEPT. ``users``, ``password_reset_tokens`` and
    ``user_section_access`` are not copied, so ``admin`` / ``admin`` still works
    and production's password hashes never land on a laptop.
  * NOTHING IS LOST. The old file is kept as ``<dest>.bak-<timestamp>``, the new
    one is built beside it and only swapped in once every table has copied and
    its row count matches production's.
  * ``--no-files`` leaves stored documents (lease PDFs, statement PDFs, receipt
    images -- any binary column) empty, which keeps the local file small.
"""
import argparse
import os
import shutil
import sqlite3
import sys
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, inspect, text

#: Never copied: the local accounts stay the local accounts.
SKIP_TABLES = {"users", "password_reset_tokens", "user_section_access"}
CHUNK = 50_000


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _cell(v):
    """A value sqlite3 can bind: numpy scalars to Python, timestamps to ISO text."""
    if isinstance(v, (bytes, bytearray, memoryview)):
        return bytes(v)
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(v, "item") and not isinstance(v, (str, bytes)) and not hasattr(v, "isoformat"):
        v = v.item()
    return _date_text(v) if hasattr(v, "isoformat") else v


def _date_text(v):
    """A date the way this SQLite database has always held one.

    SQLITE COMPARES DATES AS TEXT. The first version wrote ``isoformat()`` --
    ``2026-06-30T00:00:00`` -- and since "T" sorts after " ", every row dated
    6/30 fell OUTSIDE a "through 6/30 23:59:59" filter: 879 IA rows on the first
    real pull (Oct 5 2026), silently, on every local screen that cuts by date.
    So: a midnight with no time zone is a plain date (``2026-06-30``, what the
    local tables held before), and anything else keeps its time after a SPACE.
    """
    if hasattr(v, "hour"):
        if getattr(v, "tzinfo", None) is None and (v.hour, v.minute, v.second,
                                                   getattr(v, "microsecond", 0)) == (0, 0, 0, 0):
            return v.date().isoformat() if hasattr(v, "date") else v.isoformat()[:10]
        return v.isoformat(sep=" ")
    return v.isoformat()


_GLOB_T = "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T"


def repair_dates(path: str) -> int:
    """Rewrite ``YYYY-MM-DDT...`` text left by the first version, in place.

    Exactly what ``_date_text`` writes now: ``T00:00:00`` becomes the plain
    date, any other ``T`` becomes a space. Only text matching the ISO shape is
    touched.
    """
    db = sqlite3.connect(path)
    changed = 0
    tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    for t in tables:
        for col in [r[1] for r in db.execute(f"PRAGMA table_info({_q(t)})")]:
            c = _q(col)
            cur = db.execute(f"UPDATE {_q(t)} SET {c} = substr({c}, 1, 10) "
                             f"WHERE typeof({c}) = 'text' AND {c} GLOB '{_GLOB_T}00:00:00'")
            changed += cur.rowcount
            cur = db.execute(f"UPDATE {_q(t)} SET {c} = substr({c}, 1, 10) || ' ' || substr({c}, 12) "
                             f"WHERE typeof({c}) = 'text' AND {c} GLOB '{_GLOB_T}*'")
            changed += cur.rowcount
    db.commit()
    db.close()
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", default="waterfall.db")
    ap.add_argument("--no-files", action="store_true",
                    help="leave binary columns (stored PDFs, images) empty")
    ap.add_argument("--repair-dates", action="store_true",
                    help="only fix 'YYYY-MM-DDT..' dates written by the first version, in place")
    args = ap.parse_args()

    if args.repair_dates:
        n = repair_dates(args.dest)
        print(f"Rewrote {n:,} date values in {args.dest}.")
        return 0

    url = os.environ.get("DATABASE_URL", "").strip()
    # PULL_SOURCE_URL is for this script's own test only (a SQLite stand-in).
    url = os.environ.get("PULL_SOURCE_URL", "").strip() or url
    if not (url.startswith("postgres") or os.environ.get("PULL_SOURCE_URL")):
        print("DATABASE_URL is not set to a PostgreSQL URL in this terminal. Set it first "
              "(see the top of this file); it is read, never printed.")
        return 2
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if not os.path.exists(args.dest):
        print(f"{args.dest} does not exist; run from the repo root or pass --dest.")
        return 2

    pg = create_engine(url, connect_args={"connect_timeout": 20} if url.startswith("postgres") else {})
    try:
        with pg.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as e:
        # The message can carry the host but not the password; print the type
        # and the first line only.
        print(f"Could not connect to production: {type(e).__name__}: "
              f"{str(e).splitlines()[0][:200]}")
        print("If it timed out, this machine's IP may need adding to the Azure "
              "PostgreSQL firewall (psql-waterfall-dev > Networking).")
        return 2

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    work = f"{args.dest}.pulling"
    backup = f"{args.dest}.bak-{stamp}"
    shutil.copy2(args.dest, work)
    lite = sqlite3.connect(work)

    insp = inspect(pg)
    tables = sorted(t for t in insp.get_table_names() if t not in SKIP_TABLES)
    local = {r[0] for r in lite.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    print(f"{len(tables)} production tables to copy into a copy of {args.dest} "
          f"(skipping {', '.join(sorted(SKIP_TABLES))}).")

    report, failed = [], []
    for t in tables:
        try:
            cols_pg = insp.get_columns(t)
            binary = {c["name"] for c in cols_pg
                      if "BYTEA" in str(c["type"]).upper() or "LARGEBINARY" in type(c["type"]).__name__.upper()}
            with pg.connect() as c:
                n_src = c.execute(text(f"SELECT COUNT(*) FROM {_q(t)}")).scalar()
            if t in local:
                have = {r[1] for r in lite.execute(f"PRAGMA table_info({_q(t)})")}
                for c in cols_pg:
                    if c["name"] not in have:
                        lite.execute(f"ALTER TABLE {_q(t)} ADD COLUMN {_q(c['name'])}")
                lite.execute(f"DELETE FROM {_q(t)}")
            created = t not in local
            n_dst = 0
            sel = ", ".join("NULL AS " + _q(c["name"]) if (args.no_files and c["name"] in binary)
                            else _q(c["name"]) for c in cols_pg)
            with pg.connect().execution_options(stream_results=True) as c:
                for chunk in pd.read_sql(text(f"SELECT {sel} FROM {_q(t)}"), c, chunksize=CHUNK):
                    for b in binary & set(chunk.columns):
                        chunk[b] = chunk[b].map(lambda v: bytes(v) if isinstance(v, memoryview) else v)
                    if created and n_dst == 0:
                        chunk.to_sql(t, lite, if_exists="replace", index=False)
                    else:
                        cols = list(chunk.columns)
                        lite.executemany(
                            f"INSERT INTO {_q(t)} ({', '.join(_q(x) for x in cols)}) "
                            f"VALUES ({', '.join('?' for _ in cols)})",
                            [tuple(_cell(v) for v in row)
                             for row in chunk.itertuples(index=False, name=None)])
                    n_dst += len(chunk)
            if created and n_src == 0:
                cols_sql = ", ".join(_q(c["name"]) for c in cols_pg)
                lite.execute(f"CREATE TABLE IF NOT EXISTS {_q(t)} ({cols_sql})")
            lite.commit()
            ok = n_dst == n_src
            report.append((t, n_src, n_dst, "new" if created else "", ok))
            print(f"  {'ok ' if ok else 'MISMATCH'} {t}: {n_dst:,} of {n_src:,}" + (" (new table)" if created else ""))
            if not ok:
                failed.append(t)
        except Exception as e:
            lite.rollback()
            failed.append(t)
            print(f"  FAILED {t}: {type(e).__name__}: {str(e).splitlines()[0][:200]}")

    lite.close()
    if failed:
        print(f"\n{len(failed)} table(s) did not copy cleanly: {', '.join(failed)}.")
        print(f"Your local database is UNCHANGED. The partial copy is at {work}.")
        return 1
    os.replace(args.dest, backup)
    os.replace(work, args.dest)
    total = sum(r[2] for r in report)
    print(f"\nDone: {len(report)} tables, {total:,} rows, every count matching production.")
    print(f"Previous local database kept as {backup}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
