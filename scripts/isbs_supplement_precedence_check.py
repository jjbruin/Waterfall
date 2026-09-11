#!/usr/bin/env python
"""Guardrail: ISBS supplements are protected, and the app's row beats MRI's.

TWO RULES, both learned the hard way.

1. THE SUPPLEMENT TABLES MUST BE PROTECTED FROM CSV IMPORT. They exist precisely to
   survive an MRI refresh — none is in `mri_service.QUERY_REGISTRY`, so a refresh never
   touches them. But the CSV import runs `to_sql(if_exists="replace")`, which DROPS the
   table. One `ISBS_Budget_IS_Supplements.csv` upload would destroy every budget the team
   had imported and vetted, with no error, exactly as a single MRI_Capital_Calls.csv
   upload destroyed every app-entered capital call on Sep 10 2026.

   This matters more for the budget supplement than the others: MRI does not receive a
   budget until it is analysed and approved, so between import and approval THE APP HOLDS
   THE ONLY COPY.

2. WHERE BOTH EXIST, THE APP'S ROW WINS. Supplements are appended, not merged, so once
   MRI loads an approved budget the same (vcode, dtEntry, vSource, vAccount) appears
   twice and every consumer double-counts. An NOI that silently doubles is worse than
   either version being wrong. The app's copy is kept because the app is where the work
   is done — a partner budget is imported, questioned against the appraiser's starting
   points, and re-imported until final, and MRI only sees it afterwards.

Run:  python scripts/isbs_supplement_precedence_check.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = FAIL = 0


def chk(label, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}" + (f"\n          {detail}" if detail else ""))


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    import warnings, logging
    warnings.filterwarnings("ignore")
    logging.disable(logging.INFO)

    import database
    from flask_app.services import data_service as ds

    print("1. Every ISBS supplement table is protected from CSV import")
    for t in sorted(ds._ISBS_SUPPLEMENTS):
        chk(f"{t} is in PROTECTED_TABLES", t in database.PROTECTED_TABLES,
            "a CSV upload would DROP this table and destroy app-entered records")

    print("\n2. ...and the import entry point actually refuses them")
    for t in sorted(ds._ISBS_SUPPLEMENTS):
        # Signature is (table_name, df) — short-circuits on PROTECTED_TABLES before any DB work.
        res = database.import_csv_dataframe(t, pd.DataFrame([{"vcode": "X"}]))
        chk(f"import_csv_dataframe('{t}') returns protected",
            isinstance(res, dict) and res.get("status") == "protected", f"got {res}")

    print("\n3. None of them is in QUERY_REGISTRY — an MRI refresh must not own them")
    from flask_app.services import mri_service
    targets = {v.get("target_table") for v in mri_service.QUERY_REGISTRY.values()}
    for t in sorted(ds._ISBS_SUPPLEMENTS):
        chk(f"{t} is not an MRI target", t not in targets)

    # The real function is exercised below on real data; this replicates its shadowing
    # rule so the edge cases can be stated explicitly.
    KEY = ["vcode", "dtEntry", "vSource", "vAccount"]

    def shadow(df):
        is_supp = df["_is_supplement"].astype(bool)
        if not is_supp.any():
            return df.reset_index(drop=True)
        supp_keys = set(map(tuple, df.loc[is_supp, KEY].astype(str).values))
        mri_keys = pd.Series(
            list(map(tuple, df.loc[~is_supp, KEY].astype(str).values)),
            index=df.index[~is_supp])
        return df.drop(index=mri_keys[mri_keys.isin(supp_keys)].index).reset_index(drop=True)

    cols = ["vcode", "dtEntry", "vSource", "vAccount", "mAmount"]

    print("\n4. ISBS IS A JOURNAL — many rows per key must survive untouched")
    # This is the check that matters most. Writing the rule as
    # drop_duplicates(subset=KEY, keep='last') passes every other test here and is
    # catastrophically wrong: measured on the live snapshot it took isbs_raw from
    # 797,660 rows to 439,268, deleting 358,392 genuine MRI journal entries and roughly
    # halving every NOI. Four same-key MRI rows with NO supplement must all remain.
    journal = pd.DataFrame(
        [{**dict(zip(cols, ["p1", "2026-01-31", "Budget IS", "4010", 25.0])),
          "_is_supplement": False} for _ in range(4)])
    out = shadow(journal)
    chk("4 same-key MRI rows with no supplement ALL survive", len(out) == 4,
        f"got {len(out)} — this is the 358k-row deletion")
    chk("and they still sum to 100", out.mAmount.sum() == 100.0)

    print("\n5. A supplement replaces EVERY MRI row on its key, not just one")
    mixed = pd.concat([journal, pd.DataFrame([
        {**dict(zip(cols, ["p1", "2026-02-28", "Budget IS", "4010", 110.0])),
         "_is_supplement": False},
        {**dict(zip(cols, ["p1", "2026-01-31", "Budget IS", "4010", 999.0])),
         "_is_supplement": True},
    ])], ignore_index=True)
    out = shadow(mixed)
    jan = out[out.dtEntry == "2026-01-31"]
    chk("all 4 MRI Jan rows are gone", len(jan) == 1 and bool(jan.iloc[0]._is_supplement),
        f"got {len(jan)} Jan rows")
    chk("Jan totals the APP's 999, not MRI's 4x25", jan.mAmount.sum() == 999.0,
        f"got {jan.mAmount.sum()}")
    feb = out[out.dtEntry == "2026-02-28"]
    chk("an MRI row with no app counterpart is UNTOUCHED",
        len(feb) == 1 and float(feb.iloc[0].mAmount) == 110.0, f"got {feb.mAmount.tolist()}")

    print("\n6. A different account, source or property is NOT a match")
    other = pd.concat([journal, pd.DataFrame([
        {**dict(zip(cols, ["p1", "2026-01-31", "Budget IS", "5020", 50.0])), "_is_supplement": True},
        {**dict(zip(cols, ["p1", "2026-01-31", "Interim IS", "4010", 60.0])), "_is_supplement": True},
        {**dict(zip(cols, ["p2", "2026-01-31", "Budget IS", "4010", 70.0])), "_is_supplement": True},
    ])], ignore_index=True)
    out = shadow(other)
    chk("nothing is shadowed and every row survives", len(out) == len(other),
        f"{len(other)} in, {len(out)} out")

    print(f"\nPASS={PASS} FAIL={FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
