"""Guardrail: the frozen store is protected from a CSV replace, and still writable.

`portfolio_snapshot_frozen` and `portfolio_snapshot_frozen_history` hold the ONLY
record of what an investor was actually sent. A recomputation cannot reproduce
them — that is the whole point of freezing — so a CSV import that dropped and
rebuilt them would not be a refresh, it would be the loss of the record.

BOTH DIRECTIONS, AND THE SECOND IS THE ONE THAT MATTERS. Protection without an
app write path is a LOCKOUT, not a safeguard: `isbs_uw_supplements` was
protected once, had no app writer, and its 56 rows simply froze. So this asserts
that the tables are protected AND that freezing, re-freezing, unfreezing and the
quarter-level unfreeze all still write through.

  1. Both tables are in `PROTECTED_TABLES`.
  2. EVERY CSV import entry point refuses them, by name, returning `protected`
     rather than silently succeeding. Driven through the real functions, not by
     reading the set — a check that only reads the set cannot tell whether
     anything consults it.
  3. The rows SURVIVE a refusal. "Refused" and "refused after truncating" look
     the same from the return value.
  4. The app can still freeze, re-freeze, unfreeze and quarter-unfreeze.

Usage
    .venv/Scripts/python.exe scripts/frozen_store_protected_check.py
"""
from __future__ import annotations

import os
import sys
import tempfile

os.environ["FREEZE_ENABLED"] = "1"      # this suite exercises freezing

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd      # noqa: E402
import sqlalchemy        # noqa: E402

PASS = FAIL = 0


def chk(label, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  OK   {label}")
    else:
        FAIL += 1
        print(f"  FAIL {label}" + (f"  -> {detail}" if detail else ""))


tmpdir = tempfile.mkdtemp(prefix="frozen_protected_")
eng = sqlalchemy.create_engine(f"sqlite:///{os.path.join(tmpdir, 't.db')}")

import database                                                     # noqa: E402
from flask_app.services import portfolio_snapshot_freeze as F       # noqa: E402
from flask_app.services import portfolio_snapshot_persistence as P  # noqa: E402

database.set_engine(eng)
F._engine = lambda: eng
F._is_postgres = lambda: False
F._data_version = lambda: "build=test"
P.load_page = lambda i, q: {"comments": [], "footnotes": [], "values": []}

TABLES = ("portfolio_snapshot_frozen", "portfolio_snapshot_frozen_history")
Q = "2026-Q3"


def _asm(inv, quarter, **kw):
    return {"subtabs": {"financial": {"groups": {"G": {"deals": [
        {"vcode": "D1", "name": "Deal One"}]}}}},
        "errors": {}, "resolution": {"investor_name": inv}}


def _op(vc, q):
    return {"vcode": vc, "quarter": q}


def _can_freeze(inv):
    """True when the app can still write a freeze. Never raises."""
    try:
        F.freeze_part(inv, "2099-Q1", F.PART_SNAPSHOT, "probe", assembler=_asm)
        return True
    except Exception:
        return False


def _count(t):
    try:
        with eng.connect() as c:
            return c.execute(sqlalchemy.text(f"SELECT COUNT(*) FROM {t}")).scalar()
    except Exception:
        return None


print("A. both tables are protected")
for t in TABLES:
    chk(f"{t} is in PROTECTED_TABLES", t in database.PROTECTED_TABLES)
# A NEIGHBOUR THAT IS NOT PROTECTED, so "everything is protected" cannot pass.
chk("...and an ordinary MRI table is NOT protected (the set still discriminates)",
    "gl_detail" not in database.PROTECTED_TABLES)

print("\nB. the app can still write — protection is not a lockout")
F.freeze_part("INV1", Q, F.PARTS, "cbui", assembler=_asm, one_pager_getter=_op)
F.freeze_part("INV2", Q, F.PART_ONE_PAGERS, "cbui", assembler=_asm,
              one_pager_getter=_op)
chk("a freeze writes", F.is_frozen("INV1", Q, "snapshot"))
chk("...for both investors", F.is_frozen("INV2", Q, "one_pagers"))
before_rows = _count(TABLES[0])
chk("rows really are in the protected table", (before_rows or 0) >= 2,
    str(before_rows))
F.refreeze("INV1", Q, "cbui", "a correction", assembler=_asm,
           one_pager_getter=_op)
chk("a re-freeze writes", (F.get_frozen("INV1", Q) or {}).get("version") == 2,
    str((F.get_frozen("INV1", Q) or {}).get("version")))
chk("...and archives to the protected history table", (_count(TABLES[1]) or 0) >= 1,
    str(_count(TABLES[1])))

print("\nC. no CSV can reach them, and if one ever could it is refused")
# THEY ARE NOT CSV TARGETS TODAY. `TABLE_DEFINITIONS` maps a filename to each
# importable table, and neither of these is in it — so every import path refuses
# them as "Unknown table" BEFORE the protection check is reached. Worth stating
# plainly rather than letting the green tick imply more: today the protection is
# belt and braces, exactly as `wp_fs_map`'s was when it was added at v506.
chk("neither is a CSV import target today",
    not any(t in database.TABLE_DEFINITIONS for t in TABLES),
    str([t for t in TABLES if t in database.TABLE_DEFINITIONS]))

df = pd.DataFrame([{"investor_code": "WIPED", "quarter": "1999-Q1",
                    "payload": "{}"}])
folder = os.path.join(tmpdir, "csvs")
os.makedirs(folder, exist_ok=True)
for t in TABLES:
    df.to_csv(os.path.join(folder, f"{t}.csv"), index=False)

for t in TABLES:
    n_before = _count(t)
    r1 = database.import_csv_dataframe(t, df)
    chk(f"import_csv_dataframe does not import into {t}",
        r1.get("status") in ("protected", "error"), str(r1)[:110])
    r3 = database.import_single_csv(folder, t)
    chk(f"import_single_csv does not import into {t}",
        (r3.get(t) or {}).get("status") in ("protected", "error"), str(r3)[:110])
    # "Refused" and "refused after truncating" return the same thing; only the
    # count can tell them apart.
    chk(f"...and {t}'s rows are untouched", _count(t) == n_before,
        f"{_count(t)} vs {n_before}")
    chk(f"...and no 'WIPED' row got in", 0 == (
        eng.connect().execute(sqlalchemy.text(
            f"SELECT COUNT(*) FROM {t} WHERE investor_code = 'WIPED'")).scalar()))

# THE PROTECTION BRANCH ITSELF. The unknown-table check fires first, so today
# the `PROTECTED_TABLES` branch would never run — and a rule that is never
# reached is not a rule. This simulates the future in which somebody DOES map a
# CSV to the frozen store: mapped, the import must come back `protected`.
_saved = dict(database.TABLE_DEFINITIONS)
try:
    for t in TABLES:
        database.TABLE_DEFINITIONS[t] = {"csv": f"{t}.csv"}
    for t in TABLES:
        n_before = _count(t)
        r = database.import_csv_dataframe(t, df)
        chk(f"MAPPED to a CSV, {t} comes back 'protected'",
            r.get("status") == "protected", str(r)[:110])
        r3 = database.import_single_csv(folder, t)
        chk("...and the single-CSV path too",
            (r3.get(t) or {}).get("status") == "protected", str(r3)[:110])
        bulk = database.import_csvs_to_database(folder)
        chk("...and the bulk folder import",
            (bulk.get(t) or {}).get("status") == "protected",
            str(bulk.get(t))[:110])
        chk(f"...with {t}'s rows still there", _count(t) == n_before,
            f"{_count(t)} vs {n_before}")
        # PROVE THE SIMULATION IS REAL. Without the protection the same call
        # would have imported, so the refusal is the protection and not the
        # mapping. Remove it from the set and watch it go through.
        database.PROTECTED_TABLES.discard(t)
        r_open = database.import_csv_dataframe(t, df)
        chk("...and UNPROTECTED the very same call is NOT refused",
            r_open.get("status") != "protected", str(r_open)[:110])
        database.PROTECTED_TABLES.add(t)
finally:
    database.TABLE_DEFINITIONS.clear()
    database.TABLE_DEFINITIONS.update(_saved)
    for t in TABLES:
        database.PROTECTED_TABLES.add(t)

# WHAT THE UNPROTECTED CALL ACTUALLY DID, and it is worse than losing rows.
# `to_sql(if_exists="replace")` replaces the SCHEMA as well: the table came back
# with the CSV's three columns, so every frozen row was gone AND the freeze
# engine could no longer write at all — the next `freeze_part` fails with
# "no column named approved_by". One stray CSV would take out the record of what
# was sent and the ability to record anything further. That is what protection
# is for, stated as an assertion rather than a worry.
chk("the unprotected import really did wipe the frozen rows",
    not F.is_frozen("INV1", Q, "snapshot"))
_cols = {c["name"] for c in sqlalchemy.inspect(eng).get_columns(TABLES[0])}
chk("...and replaced the SCHEMA, not just the rows",
    "approved_by" not in _cols, str(sorted(_cols)))
chk("...leaving the freeze engine unable to write", not _can_freeze("WRECKED"))

# AND IT DOES NOT HEAL ITSELF. `ensure_schema` is `CREATE TABLE IF NOT EXISTS`
# plus the `_ADDED_COLUMNS` back-fill; the wrecked table EXISTS, so the CREATE is
# a no-op, and `approved_by` is a base column that the back-fill list does not
# carry. So the app stays unable to freeze until somebody drops the table by
# hand. That is the real cost of one stray CSV, and it is why these two tables
# belong in PROTECTED_TABLES rather than merely being absent from
# TABLE_DEFINITIONS.
F.ensure_schema(force=True)
_cols2 = {c["name"] for c in sqlalchemy.inspect(eng).get_columns(TABLES[0])}
chk("ensure_schema does NOT repair it — the damage needs a hand",
    "approved_by" not in _cols2, str(sorted(_cols2)))

# Repaired here the only way it can be — BOTH tables, because the loop above
# unprotected each in turn and the history table was replaced too — so section D
# has something to test.
with eng.begin() as cx:
    for t in TABLES:
        cx.execute(sqlalchemy.text(f"DROP TABLE IF EXISTS {t}"))
F.ensure_schema(force=True)
_cols3 = {c["name"] for c in sqlalchemy.inspect(eng).get_columns(TABLES[0])}
_hcols = {c["name"] for c in sqlalchemy.inspect(eng).get_columns(TABLES[1])}
chk("...and after a DROP the schema comes back correct",
    "approved_by" in _cols3 and "one_pagers_frozen_at" in _cols3)
chk("...for the history table too", "data_version" in _hcols)
F.freeze_part("INV1", Q, F.PARTS, "cbui", assembler=_asm, one_pager_getter=_op)
F.freeze_part("INV2", Q, F.PART_ONE_PAGERS, "cbui", assembler=_asm,
              one_pager_getter=_op)
chk("...and the app freezes again", F.is_frozen("INV1", Q, "snapshot")
    and F.is_frozen("INV2", Q, "one_pagers"))

print("\nD. and the app can still UNFREEZE — both ways")
F.unfreeze("INV1", Q, "cbui", "single investor")
chk("single-investor unfreeze still works", F.get_frozen("INV1", Q) is None)
out = F.unfreeze_quarter(Q, "one_pagers", "cbui", "quarter level")
chk("quarter-level unfreeze still works", out["investors"] >= 1, str(out))
chk("...and INV2 is no longer frozen for that half",
    not F.is_frozen("INV2", Q, "one_pagers"))
# The history table was dropped and rebuilt above, so the count here is only
# what the two unfreezes in THIS section archived — not the whole run.
chk("...with each unfreeze archiving as it went",
    (_count(TABLES[1]) or 0) >= 2, str(_count(TABLES[1])))
# THE PAIRED DIRECTION for the whole file: a fresh freeze must still be possible
# AFTER all the refusals, or protection has quietly broken the write path.
F.freeze_part("INV3", Q, F.PARTS, "cbui", assembler=_asm, one_pager_getter=_op)
chk("a NEW freeze still works after every refusal",
    F.is_frozen("INV3", Q, "snapshot") and F.is_frozen("INV3", Q, "one_pagers"))

print(f"\n{'=' * 60}\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
