"""Guardrail -- the production pull keeps only the last two local backups.

scripts/pull_production_db.py renames the old waterfall.db to
waterfall.db.bak-<timestamp> on every successful pull, and nothing ever removed
those (~250 MB each, ~1.5 GB with stored PDFs). From Oct 8 2026 a successful
pull keeps the --keep-backups newest (default 2) and deletes the rest.

Runs entirely in a temp folder -- no production, no real database:

    .venv/Scripts/python.exe scripts/pull_production_db_backups_check.py

  A  prune_backups keeps the N newest by modified time, deletes the rest, and
     never touches the database itself, the .pulling work file or -wal/-shm
  B  BOTH DIRECTIONS end to end (a SQLite file stands in for production through
     PULL_SOURCE_URL): a successful pull leaves exactly 2 backups, the newest
     being the one it just made; a FAILED pull deletes nothing
  C  --keep-all-backups deletes nothing; --keep-backups 0 is refused
  D  re-injection: a prune that keeps everything fails A (not vacuous)
"""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import pull_production_db as P                                    # noqa: E402

FAILS: list = []
SCRIPT = os.path.join(ROOT, "scripts", "pull_production_db.py")


def chk(label, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + label + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        FAILS.append(label)


def touch(path, age_s):
    with open(path, "wb") as f:
        f.write(b"x")
    t = time.time() - age_s
    os.utime(path, (t, t))


def make_db(path, rows, not_null=False):
    db = sqlite3.connect(path)
    db.execute(f"CREATE TABLE deals (vcode TEXT {'NOT NULL' if not_null else ''})")
    db.executemany("INSERT INTO deals VALUES (?)", [(r,) for r in rows])
    db.commit()
    db.close()


def q(path, sql):
    """One query, connection closed -- Windows will not delete an open file."""
    db = sqlite3.connect(path)
    try:
        return db.execute(sql).fetchall()
    finally:
        db.close()


def backups(folder):
    return sorted(n for n in os.listdir(folder) if ".bak-" in n)


def run(folder, src, *extra):
    env = {**os.environ, "PULL_SOURCE_URL": "sqlite:///" + src.replace("\\", "/")}
    env.pop("DATABASE_URL", None)
    r = subprocess.run([sys.executable, SCRIPT, "--dest", os.path.join(folder, "waterfall.db"), *extra],
                       env=env, capture_output=True, text=True, cwd=folder)
    return r.returncode, r.stdout + r.stderr


def scenario_files(folder):
    dest = os.path.join(folder, "waterfall.db")
    touch(dest, 0)
    for name, age in (("waterfall.db.bak-20261001-000000", 400), ("waterfall.db.bak-20261003-000000", 300),
                      ("waterfall.db.bak-20261005-original", 200), ("waterfall.db.bak-20261006-000000", 100)):
        touch(os.path.join(folder, name), age)
    for side in ("waterfall.db.pulling", "waterfall.db.bak-20261006-000000-wal", "other.txt"):
        touch(os.path.join(folder, side), 500)
    return dest


def main():
    # ---- A: the prune itself --------------------------------------------
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        dest = scenario_files(d)
        kept, removed, failed = P.prune_backups(dest, 2)
        names = lambda ps: sorted(os.path.basename(p) for p in ps)
        chk("A: keeps the 2 newest backups",
            names(kept) == ["waterfall.db.bak-20261005-original", "waterfall.db.bak-20261006-000000"],
            names(kept))
        chk("A: deletes the 2 oldest",
            names(removed) == ["waterfall.db.bak-20261001-000000", "waterfall.db.bak-20261003-000000"]
            and not failed, names(removed))
        left = sorted(os.listdir(d))
        chk("A: never touches the database, the .pulling file, -wal or other files",
            all(f in left for f in ("waterfall.db", "waterfall.db.pulling",
                                    "waterfall.db.bak-20261006-000000-wal", "other.txt")), left)

    # ---- B: end to end, both directions ---------------------------------
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        dest = os.path.join(d, "waterfall.db")
        src = os.path.join(d, "prod.db")
        make_db(dest, ["OLD"])
        make_db(src, ["P1", "P2"])
        for name, age in (("waterfall.db.bak-20261001-000000", 300),
                          ("waterfall.db.bak-20261003-000000", 200),
                          ("waterfall.db.bak-20261005-000000", 100)):
            touch(os.path.join(d, name), age)
        code, out = run(d, src)
        b = backups(d)
        rows = q(dest, "SELECT COUNT(*) FROM deals")[0][0]
        chk("B: a successful pull swaps in the new copy", code == 0 and rows == 2, out[-300:])
        chk("B: ...and leaves exactly 2 backups, the one it just made plus the newest old one",
            len(b) == 2 and "waterfall.db.bak-20261005-000000" in b
            and any(x not in ("waterfall.db.bak-20261005-000000",) for x in b), b)
        chk("B: ...and says which it removed",
            "Removed old backup waterfall.db.bak-20261001-000000" in out
            and "Removed old backup waterfall.db.bak-20261003-000000" in out, out[-400:])

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        dest = os.path.join(d, "waterfall.db")
        src = os.path.join(d, "prod.db")
        make_db(dest, ["OLD"], not_null=True)        # local NOT NULL ...
        make_db(src, [None])                          # ... production NULL -> the copy fails
        for name, age in (("waterfall.db.bak-20261001-000000", 300),
                          ("waterfall.db.bak-20261003-000000", 200),
                          ("waterfall.db.bak-20261005-000000", 100)):
            touch(os.path.join(d, name), age)
        code, out = run(d, src)
        chk("B: a FAILED pull deletes no backup and leaves the database as it was",
            code == 1 and len(backups(d)) == 3
            and q(dest, "SELECT vcode FROM deals") == [("OLD",)],
            (code, backups(d)))

    # ---- C: the switches -------------------------------------------------
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        dest = os.path.join(d, "waterfall.db")
        src = os.path.join(d, "prod.db")
        make_db(dest, ["OLD"])
        make_db(src, ["P1"])
        for name, age in (("waterfall.db.bak-20261001-000000", 300),
                          ("waterfall.db.bak-20261003-000000", 200)):
            touch(os.path.join(d, name), age)
        code, out = run(d, src, "--keep-all-backups")
        chk("C: --keep-all-backups deletes nothing", code == 0 and len(backups(d)) == 3, backups(d))
        code, out = run(d, src, "--keep-backups", "0")
        chk("C: --keep-backups 0 is refused", code == 2 and len(backups(d)) == 3, (code, out[-200:]))

    # ---- D: re-injection -------------------------------------------------
    real = P.prune_backups
    P.prune_backups = lambda dest, keep: (P.list_backups(dest), [], [])
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            dest = scenario_files(d)
            _, removed, _ = P.prune_backups(dest, 2)
            chk("D: a prune that keeps everything would fail A (check is not vacuous)",
                len(removed) != 2)
    finally:
        P.prune_backups = real

    print(f"\n{len(FAILS)} failing")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
