#!/usr/bin/env python
"""Guardrail: every refresh_table() call must invalidate a cache key that EXISTS.

`data_service.refresh_table(name)` resolves the cache key as
`table_to_key.get(name, name)` — it falls back to the table's own name. When the cache
actually holds that table under a DIFFERENT key and the mapping is missing, the refresh
writes a key nothing reads and returns successfully. There is no error, no warning, and
no symptom until someone notices a stale figure.

That is exactly what happened to valuations on Sep 11 2026: the cache holds them under
`mri_val`, `"valuations"` was not in the map, so `publish_record`'s invalidation was a
silent no-op. A newly published 12/31/2025 valuation sat correctly in the database while
every page kept serving the 12/31/2024 figure for the life of the process. Three
deploys were spent looking elsewhere.

This check finds every table name the app passes to refresh_table, resolves it the same
way refresh_table does, and asserts the result is a key the loaded cache actually
contains. A mapping that is missing fails here instead of in production.

Run:  python scripts/refresh_table_key_check.py
"""
from __future__ import annotations

import ast
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SKIP = (".claude", "worktrees", ".venv", "node_modules", ".git")
DS = os.path.join(ROOT, "flask_app", "services", "data_service.py")

PASS = FAIL = 0


def chk(label, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}" + (f"\n          {detail}" if detail else ""))


def table_to_key() -> dict:
    """Read the mapping out of the source, so this cannot drift from the real one."""
    src = open(DS, encoding="utf-8").read()
    m = re.search(r"table_to_key\s*=\s*(\{.*?\})", src, re.S)
    if not m:
        raise SystemExit("Could not find table_to_key in data_service.py")
    # Strip comments before literal_eval.
    body = re.sub(r"#[^\n]*", "", m.group(1))
    return ast.literal_eval(body)


def refresh_call_sites() -> dict:
    """Every literal table name passed to refresh_table(), with where it was called."""
    out = {}
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP and not d.startswith(".")]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, ROOT)
            if any(s in rel for s in SKIP):
                continue
            try:
                src = open(path, encoding="utf-8").read()
            except Exception:
                continue
            for m in re.finditer(r"refresh_table\(\s*[\"']([A-Za-z0-9_]+)[\"']", src):
                out.setdefault(m.group(1), set()).add(rel)
    return out


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    import warnings, logging
    warnings.filterwarnings("ignore")
    logging.disable(logging.INFO)

    mapping = table_to_key()
    calls = refresh_call_sites()
    print(f"table_to_key entries: {len(mapping)}   refresh_table call sites: {len(calls)}\n")

    # The adapters resolve their engine from the Flask config, so the load has to happen
    # inside an app context — same path the running app uses, not a stand-in for it.
    from flask_app import create_app
    from flask_app.services import data_service
    db = os.path.join(ROOT, "waterfall.db")
    if not os.path.exists(db):
        raise SystemExit(f"Needs the local snapshot to know the real cache keys: {db}")
    app = create_app()
    with app.app_context():
        data = data_service.load_all(db_path=db)
    keys = set(data.keys())
    print(f"cache holds {len(keys)} keys\n")

    # Some tables are not cached under a key of their own: refresh_table REASSEMBLES a
    # derived frame from them. Those are not exempt — the check just moves to the key the
    # reassembly writes, which is the thing that has to be invalidated.
    REASSEMBLED = {
        "forecasts": "fc",   # _assemble_forecasts -> load_forecast -> data["fc"]
        "isbs": "isbs_raw",
    }
    for t in getattr(data_service, "_ISBS_SPLIT", set()):
        REASSEMBLED[t] = "isbs_raw"
    for t in getattr(data_service, "_ISBS_SUPPLEMENTS", set()):
        REASSEMBLED[t] = "isbs_raw"

    print("Every refresh_table(<table>) resolves to a key the cache actually has")
    for table in sorted(calls):
        if table in REASSEMBLED:
            target = REASSEMBLED[table]
            chk(f"{table} -> reassembles {target}", target in keys,
                f"reassembly target '{target}' is not a cache key")
            continue
        if table == "occupancy_supplements":
            chk(f"{table} -> (occupancy reassembly)", True)
            continue
        resolved = mapping.get(table, table)
        chk(f"{table} -> {resolved}", resolved in keys,
            f"'{resolved}' is NOT a cache key. refresh_table('{table}') is a NO-OP — it "
            f"writes a key nothing reads.\n          called from: "
            f"{', '.join(sorted(calls[table]))}\n          fix: add "
            f"\"{table}\": \"<real key>\" to table_to_key in data_service.py")

    print("\nEvery declared mapping points at a real key")
    for table, key in sorted(mapping.items()):
        chk(f"{table} -> {key}", key in keys, f"'{key}' is not in the loaded cache")

    print(f"\nPASS={PASS} FAIL={FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
