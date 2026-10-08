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
  * ONLY THE LAST TWO BACKUPS ARE KEPT (Charlene, Oct 8 2026). Each backup is a
    full copy (~250 MB, ~1.5 GB with stored PDFs) and nothing ever removed them.
    After a SUCCESSFUL swap the oldest ``<dest>.bak-*`` files beyond
    ``--keep-backups`` (default 2, newest first by modified time) are deleted and
    named on screen. A failed or partial run deletes nothing, and
    ``--keep-all-backups`` turns pruning off.
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
SKIP_TABLES = {"users", "password_reset_tokens", "user_section_access",
               # Who holds what, and the trail of it -- not laptop material.
               "user_permissions", "access_audit"}

#: Compensation and payroll planning (Board plan, Oct 5 2026): left on the
#: server unless asked for by name with --include-compensation, which only a
#: salary-planning holder should ever pass.
SKIP_PREFIXES = ("comp_",)
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


DEFAULT_KEEP_BACKUPS = 2

#: SQLite side files that can sit next to a database; never treated as backups.
_SIDE_FILES = ("-wal", "-shm", "-journal")


def list_backups(dest: str) -> list:
    """``<dest>.bak-*`` files beside ``dest``, newest first (by modified time)."""
    folder = os.path.dirname(os.path.abspath(dest))
    prefix = os.path.basename(dest) + ".bak-"
    found = [os.path.join(folder, n) for n in os.listdir(folder)
             if n.startswith(prefix) and not n.endswith(_SIDE_FILES)
             and os.path.isfile(os.path.join(folder, n))]
    return sorted(found, key=os.path.getmtime, reverse=True)


def prune_backups(dest: str, keep: int) -> tuple:
    """Delete all but the ``keep`` newest backups of ``dest``.

    Returns ``(kept, removed, failed)`` as path lists. ``keep`` below 1 is
    refused: pruning must never be able to delete the backup the run just made.
    A file that cannot be deleted (open elsewhere) is reported, not fatal.
    """
    if keep < 1:
        raise ValueError("keep must be at least 1")
    backups = list_backups(dest)
    kept, removed, failed = backups[:keep], [], []
    for path in backups[keep:]:
        try:
            os.remove(path)
            removed.append(path)
        except OSError:
            failed.append(path)
    return kept, removed, failed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", default="waterfall.db")
    ap.add_argument("--no-files", action="store_true",
                    help="leave binary columns (stored PDFs, images) empty")
    ap.add_argument("--repair-dates", action="store_true",
                    help="only fix 'YYYY-MM-DDT..' dates written by the first version, in place")
    ap.add_argument("--include-compensation", action="store_true",
                    help="also copy comp_* tables (salary-planning holders only)")
    ap.add_argument("--keep-backups", type=int, default=DEFAULT_KEEP_BACKUPS,
                    help="after a successful pull, keep only this many newest "
                         f"<dest>.bak-* files (default {DEFAULT_KEEP_BACKUPS}, minimum 1)")
    ap.add_argument("--keep-all-backups", action="store_true",
                    help="do not delete any old backups")
    args = ap.parse_args()
    if args.keep_backups < 1:
        print("--keep-backups must be at least 1 (use --keep-all-backups to keep every one).")
        return 2

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
    include_comp = args.include_compensation
    tables = sorted(t for t in insp.get_table_names() if t not in SKIP_TABLES
                    and (include_comp or not t.lower().startswith(SKIP_PREFIXES)))
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

    # Only now, with the new copy in place and its backup written.
    if args.keep_all_backups:
        print("Old backups left as they are (--keep-all-backups).")
    else:
        kept, removed, stuck = prune_backups(args.dest, args.keep_backups)
        for p in removed:
            print(f"Removed old backup {os.path.basename(p)}.")
        for p in stuck:
            print(f"Could not remove old backup {os.path.basename(p)} (in use?); left in place.")
        print(f"Keeping the {len(kept)} most recent backup(s): "
              f"{', '.join(os.path.basename(p) for p in kept)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
