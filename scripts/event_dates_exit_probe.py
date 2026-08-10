"""READ-ONLY probe: Current Anticipated Exit coverage in the event_dates table.

Answers the three questions behind the One Pager "Current Anticipated Exit" field:
  1. how many deals have a matching row,
  2. a sample of the returned dtEvent dates,
  3. which deals have MORE than one matching row (duplicate handling).

Match = vEventType 'Disposition' AND vEvent 'Closing' AND vDateType 'Projected',
by vCode. Value = dtEvent. Issues a single SELECT; writes nothing.

    # local (needs DATABASE_URL, or a waterfall.db holding the table)
    python scripts/event_dates_exit_probe.py

    # against Azure PostgreSQL
    DATABASE_URL='postgresql://USER:PASS@psql-waterfall-dev.postgres.database.azure.com/DBNAME?sslmode=require' \
        python scripts/event_dates_exit_probe.py

NOTE: event_dates is populated by importing MRI_Event_Dates.csv. The upstream
Anticipated_Exit column on the deals table comes from Prop_Info_Core.sql run
against MRI directly, so it is NOT evidence that this table is populated here.
"""
import os
import sys

import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FILTERS = {"veventtype": "Disposition", "vevent": "Closing", "vdatetype": "Projected"}


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


def main():
    eng = get_engine()

    try:
        df = pd.read_sql(text("SELECT * FROM event_dates"), eng)
    except Exception as e:
        print(f"\nFAIL: cannot read event_dates -- {type(e).__name__}: {e}")
        print("The table may not exist in this database yet (it is imported from "
              "MRI_Event_Dates.csv). Nothing to report until it is populated.")
        sys.exit(2)

    print(f"event_dates rows: {len(df):,}   columns: {list(df.columns)}")
    if df.empty:
        print("\nTable exists but is EMPTY -- every deal would show N/A.")
        sys.exit(2)

    cols = {str(c).strip().lower(): c for c in df.columns}
    missing = [k for k in list(FILTERS) + ["vcode", "dtevent"] if k not in cols]
    if missing:
        print(f"\nFAIL: expected column(s) not present: {missing}")
        sys.exit(2)

    # What values actually exist, so a filter typo cannot masquerade as "no data"
    for k in FILTERS:
        vals = df[cols[k]].astype(str).str.strip().value_counts().head(8)
        print(f"\n  distinct {cols[k]} (top 8): {dict(vals)}")

    mask = pd.Series(True, index=df.index)
    for k, want in FILTERS.items():
        mask &= df[cols[k]].astype(str).str.strip().str.lower() == want.lower()
    hits = df.loc[mask, [cols["vcode"], cols["dtevent"]]].copy()
    hits.columns = ["vcode", "dtevent"]
    hits["dtevent"] = pd.to_datetime(hits["dtevent"], errors="coerce")

    print("\n" + "=" * 78)
    print(f"(1) matching rows: {len(hits):,}   distinct deals: "
          f"{hits['vcode'].astype(str).str.strip().str.lower().nunique():,}")
    unparsed = int(hits["dtevent"].isna().sum())
    if unparsed:
        print(f"    !! {unparsed} matching row(s) have an unparseable dtEvent -> N/A")

    print("\n(2) sample of returned dtEvent dates (latest per deal, first 15):")
    key = hits["vcode"].astype(str).str.strip().str.lower()
    latest = hits.dropna(subset=["dtevent"]).groupby(key)["dtevent"].max().sort_values()
    for vc, dt in latest.head(15).items():
        print(f"      {vc:<12} {dt.date()}   -> renders {dt.month}/{dt.day}/{dt.year}")
    if len(latest) > 15:
        print(f"      ... {len(latest) - 15} more")
    if len(latest):
        print(f"    range: {latest.min().date()} .. {latest.max().date()}")

    dupes = key.value_counts()
    dupes = dupes[dupes > 1]
    print(f"\n(3) deals with MORE than one matching row: {len(dupes)}")
    for vc, n in dupes.head(20).items():
        ds = sorted(d.date() for d in hits.loc[key == vc, "dtevent"].dropna())
        print(f"      {vc:<12} {n} rows: {[str(d) for d in ds]}  -> latest {ds[-1] if ds else 'N/A'}")
    if len(dupes) > 20:
        print(f"      ... {len(dupes) - 20} more")
    if len(dupes):
        print("    -> the field takes MAX(dtEvent), matching Prop_Info_Core.sql upstream.")

    # Coverage against the deals the One Pager can be run for
    try:
        deals = pd.read_sql(text("SELECT * FROM deals"), eng)
        dcols = {str(c).strip().lower(): c for c in deals.columns}
        if "vcode" in dcols:
            dk = deals[dcols["vcode"]].astype(str).str.strip().str.lower()
            sold = pd.Series(False, index=deals.index)
            if "sale_status" in dcols:
                sold = deals[dcols["sale_status"]].astype(str).str.upper().eq("SOLD")
            active = set(dk[~sold])
            have = set(latest.index)
            print(f"\n    coverage: {len(active & have)} of {len(active)} active deals have a "
                  f"date; {len(active - have)} would show N/A")
    except Exception:
        pass

    print("=" * 78)


if __name__ == "__main__":
    main()
