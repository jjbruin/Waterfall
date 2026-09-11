#!/usr/bin/env python
"""READ-ONLY: find accounting rows whose EffectiveDate will not parse to a date.

The Aug 2026 One Pager audit found 101 such rows worth $20.18M, carrying Excel serials
(43402, 43448, ...) instead of dates. This confirms whether that is STILL true, on
whichever database you point it at, before anyone tries to fix it.

    # local SQLite snapshot
    python scripts/effectivedate_serial_check.py

    # live Azure PostgreSQL
    DATABASE_URL='postgresql://USER:PASS@psql-waterfall-dev.postgres.database.azure.com:5432/waterfall_xirr?sslmode=require' \
        python scripts/effectivedate_serial_check.py

Issues SELECTs only; writes nothing.

WHY THIS MATTERS MORE THAN IT LOOKS. `loaders.normalize_accounting_feed` parses with
`errors="coerce"`, so an unparseable date becomes NaT rather than an error. Every
consumer that filters on a date then drops the row SILENTLY. Nothing warns, and the
money simply is not there.

The shape of the defect CHANGED after the audit, which is why it needs re-measuring
rather than re-reading. In Aug the rows were INCLUDED in Total Cap (which had no date
filter) and DROPPED from PE Performance (which did) — an inconsistency between two
figures on one page. `9086f16` (v198) added the date filter to the cap stack
(`one_pager.py:878`), so both paths now drop them. That is consistent, but it means the
money is invisible EVERYWHERE rather than inconsistently counted. Woodlands Square's
entire pref equity contribution was the largest single row.
"""
from __future__ import annotations

import datetime as _dt
import os
import sys

import pandas as pd
from sqlalchemy import create_engine, text

EXCEL_EPOCH = _dt.date(1899, 12, 30)  # Excel's day 0, accounting for its 1900 leap bug


def get_engine():
    url = os.environ.get("DATABASE_URL")
    if url:
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        print(f"source: PostgreSQL ({url.split('@')[-1].split('?')[0]})")
        return create_engine(url)
    db = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "waterfall.db")
    print(f"source: sqlite ({db})")
    return create_engine(f"sqlite:///{db}")


def as_excel_serial(v):
    """Return the date an Excel serial would represent, or None if it is not one."""
    try:
        n = float(str(v).strip())
    except (TypeError, ValueError):
        return None
    # Excel serials for plausible business dates: 1990-01-01 is 32874, 2100 is ~73050.
    if not (20000 <= n <= 80000) or n != int(n):
        return None
    return EXCEL_EPOCH + _dt.timedelta(days=int(n))


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    eng = get_engine()
    with eng.connect() as conn:
        df = pd.read_sql(text(
            "SELECT \"InvestmentID\", \"InvestorID\", \"EffectiveDate\", \"MajorType\", "
            "\"Typename\", \"Amt\" FROM accounting"
        ), conn)

    print(f"accounting rows: {len(df):,}")

    # Exactly the app's own parse, from loaders.normalize_accounting_feed.
    parsed = pd.to_datetime(df["EffectiveDate"], errors="coerce")
    bad = df[parsed.isna()].copy()

    if bad.empty:
        print("\nCLEAN — every EffectiveDate parses. Nothing is being silently dropped.")
        print("If this ran against local SQLite, re-run it against Azure before closing "
              "the item: the audit finding was measured on live data.")
        return 0

    bad["Amt_num"] = pd.to_numeric(bad["Amt"], errors="coerce").fillna(0.0)
    bad["as_date"] = bad["EffectiveDate"].map(as_excel_serial)

    total = bad["Amt_num"].abs().sum()
    serials = bad["as_date"].notna().sum()
    print(f"\nUNPARSEABLE: {len(bad):,} rows, ${total:,.2f} absolute")
    print(f"  of which look like Excel serials: {serials:,}"
          f"   (other unparseable: {len(bad) - serials:,})")

    print("\nBy Typename:")
    g = (bad.groupby("Typename")
            .agg(rows=("Amt_num", "size"), amount=("Amt_num", lambda s: s.abs().sum()))
            .sort_values("amount", ascending=False))
    for name, r in g.iterrows():
        print(f"  {str(name)[:44]:<44} {int(r['rows']):>5} rows  ${r['amount']:>16,.2f}")

    print("\nLargest single rows (these are what vanish from a page):")
    for _, r in bad.reindex(bad["Amt_num"].abs().sort_values(ascending=False).index).head(10).iterrows():
        d = r["as_date"].isoformat() if r["as_date"] else "not a serial"
        print(f"  {str(r['InvestmentID'])[:12]:<12} {str(r['InvestorID'])[:12]:<12} "
              f"raw={str(r['EffectiveDate'])[:12]:<12} -> {d:<12} ${r['Amt_num']:>15,.2f}"
              f"  {str(r['Typename'])[:28]}")

    print("\nAffected InvestmentIDs:", ", ".join(sorted(set(bad["InvestmentID"].astype(str)))[:20]))
    print("\nNOTE: `accounting` IS in mri_service.QUERY_REGISTRY, so 'Refresh All Data from "
          "MRI'\n      overwrites this table. A fix applied to the database DOES NOT "
          "SURVIVE a refresh.\n      The durable fix is upstream, in MRI or in the export "
          "that feeds it.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
