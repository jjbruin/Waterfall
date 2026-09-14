"""An MRI refresh must not destroy isbs_interim_is_historical.

The table is normally EMPTY and that is fine: every Interim IS row MRI returns
goes into isbs_interim_is (the full history, 2018 onward -- the refresh applies
no date filter, despite the names). Only split_isbs_table(), the one-time
migration off the legacy monolithic isbs table, ever populates this one.

It still has to be protected, because _ISBS_SPLIT in data_service concatenates
it into isbs_raw -- rows placed here reach every ISBS consumer, and nothing
else would put them back.

Until Sep 14 2026 the ISBS import unconditionally did:

    DROP TABLE IF EXISTS isbs_interim_is_historical
    CREATE TABLE isbs_interim_is_historical (...)      -- empty
    import_results[...] = {"rows": 0, "status": "ok"}

Every successful refresh wiped the table and reported an import that never
happened. No error, no warning, a green tick. It was empty in production when
this was found, so nothing was actually lost -- but the same code would have
destroyed whatever the migration had put there, and said it went fine.

Run:  python scripts/isbs_historical_preserve_check.py
Exits non-zero if a refresh would destroy or alter the table.
"""
import sys

import pandas as pd
import sqlalchemy as sa

from flask_app import create_app
from flask_app.services import mri_service

TABLE = "isbs_interim_is_historical"
SOURCES = ("Interim IS", "Interim BS", "Budget IS", "Projected IS", "Valuation IS")


def _seed(engine):
    """Two rows that only the CSV could have provided (pre-2025 dates)."""
    pd.DataFrame([
        {"vcode": "P0000001", "dtEntry": "2019-12-31", "vSource": "Interim IS",
         "vAccount": "4010", "mAmount": "-1234.56", "vInput": "A", "statement_id": "x"},
        {"vcode": "P0000002", "dtEntry": "2020-06-30", "vSource": "Interim IS",
         "vAccount": "5090", "mAmount": "789.01", "vInput": "A", "statement_id": "y"},
    ]).to_sql(TABLE, engine, index=False, if_exists="replace")


def _stub_mri():
    """One row per live vSource, so the split branch does real work."""
    rows = [{"vcode": "P0000009", "dtEntry": "2026-06-30", "vSource": s,
             "vAccount": "4010", "mAmount": "1.0", "vInput": "A",
             "statement_id": "z"} for s in SOURCES]
    mri_service.run_query = lambda name, save_csv=True: {
        "_dataframe": pd.DataFrame(rows), "elapsed_seconds": 0.0}


def main() -> int:
    app = create_app()
    failures = []
    with app.app_context():
        from flask_app.db import get_engine
        engine = get_engine()
        _seed(engine)
        _stub_mri()

        before = pd.read_sql(f"select * from {TABLE}", engine)
        result = mri_service.import_query_to_database("ISBS_Download")
        after = pd.read_sql(f"select * from {TABLE}", engine)

        reported = result["tables"].get(TABLE, {})
        print(f"history rows before : {len(before)}")
        print(f"history rows after  : {len(after)}")
        print(f"reported            : {reported}")

        if len(after) != len(before):
            failures.append(f"row count changed {len(before)} -> {len(after)}")
        if set(after.get("vcode", [])) != set(before.get("vcode", [])):
            failures.append("contents changed")
        if reported.get("status") == "ok" and reported.get("rows") == 0 and len(before):
            failures.append('reported {"rows": 0, "status": "ok"} over existing rows '
                            "-- the old wipe, announced as a successful import")

        # A database that has never loaded the CSV still needs the schema,
        # or the loaders fail on a missing table rather than an empty one.
        with engine.begin() as conn:
            conn.execute(sa.text(f"DROP TABLE IF EXISTS {TABLE}"))
        mri_service.import_query_to_database("ISBS_Download")
        try:
            fresh = pd.read_sql(f"select * from {TABLE}", engine)
            print(f"fresh database      : table created, {len(fresh)} rows")
        except Exception as e:
            failures.append(f"fresh database left no table: {e}")

    print()
    if failures:
        print("FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS - an ISBS refresh leaves isbs_interim_is_historical alone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
