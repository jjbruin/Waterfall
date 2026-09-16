"""Find entities whose relationship ownership percentages do not close to 100%.

WHY THIS MATTERS, and it is not cosmetic. `run_upstream_waterfall_period`'s
passthrough branch pays each investor `cash_available * OwnershipPct` and never
checks that the percentages close:

    investor_cash = cash_available * ownership_pct        # waterfall.py ~2028

So an entity whose percentages sum to 123.18% distributes $123,180 out of
$100,000 in. Cash is conjured. Jim, Sep 16 2026, on Ascent on Steamboat: "How
can we allocate $123,179 when the cash flow distribution was only $100,000."
Proven synthetically: percentages of 83.18 + 40.00 turn $100,000 into
$123,180.00.

Under 100% is the same defect facing the other way -- cash silently vanishes and
the beneficiaries are short.

FOUR SERVICES share this engine: ownership_service (Upstream Analysis),
portfolio_analysis_service, ppi_upstream_service and psckoc_service. Two of them
are investor-facing. A chain that does not close is therefore not only an
Upstream Analysis problem.

This reads `relationships`, which is the feed the passthrough uses. Note that
`commitments` is the source the ownership tree prefers where the two disagree --
see ownership_chain_service -- so an entity flagged here may well be correct in
commitments and wrong only in this feed.

Run:  .venv/Scripts/python.exe scripts/ownership_pct_closure_check.py
Exit 0 always: this reports on DATA, it does not gate a commit.
"""
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

TOL = 0.01   # percentage points


def main() -> int:
    # READ THE ONE TABLE, not the whole data layer. `data_service.get_data()`
    # assembles 937,650 ISBS rows among much else; inside the 2GB production
    # container that is enough to swallow the run before this prints a line
    # (measured Sep 16 2026). A diagnostic that cannot survive the environment
    # it diagnoses is not a diagnostic.
    # The engine is built from DATABASE_URL directly rather than through
    # flask_app.db.get_engine(), which requires an application context. Running
    # inside the container this reads the SAME secret the app reads, so the
    # credential is never handled here or seen anywhere outside the container.
    import os
    from sqlalchemy import create_engine, text

    url = os.environ.get("DATABASE_URL") or "sqlite:///waterfall.db"
    where = "PostgreSQL" if url.startswith(("postgres", "postgresql")) else "SQLite"
    print(f"Reading `relationships` from {where}.")
    try:
        with create_engine(url).connect() as conn:
            rel = pd.read_sql(text("SELECT * FROM relationships"), conn)
    except Exception as e:
        print(f"Could not read `relationships`: {str(e)[:200]}")
        return 0

    if rel is None or rel.empty:
        print("No relationship data in this database — nothing to check.")
        return 0

    df = rel.copy()
    lower = {str(c).lower(): c for c in df.columns}
    inv_c, pct_c = lower.get("investmentid"), lower.get("ownershippct")
    end_c = lower.get("enddate")
    if not inv_c or not pct_c:
        print("relationships has no InvestmentID/OwnershipPct columns.")
        return 0
    if end_c:
        df = df[df[end_c].isna()]

    df["_e"] = df[inv_c].astype(str).str.strip().str.upper()
    df["_p"] = pd.to_numeric(df[pct_c], errors="coerce").fillna(0.0)
    # Stored either as 0-1 or 0-100; normalise to percentage points.
    df["_p"] = df["_p"].apply(lambda v: v * 100.0 if 0 < abs(v) <= 1.0 else v)

    g = df.groupby("_e")["_p"].agg(["sum", "count"]).reset_index()
    bad = g[(g["sum"] - 100.0).abs() > TOL].sort_values(
        by="sum", key=lambda s: (s - 100.0).abs(), ascending=False)

    print(f"Entities in `relationships`: {len(g)}")
    print(f"Percentages do NOT close to 100%: {len(bad)}")
    if bad.empty:
        print("\nEvery entity closes. The passthrough conserves cash on this data.")
        return 0

    print()
    print("  %-14s %10s %8s   %s" % ("entity", "sum %", "rows", "effect on $100,000"))
    print("  " + "-" * 62)
    for _, r in bad.head(40).iterrows():
        out = 100_000.0 * r["sum"] / 100.0
        print("  %-14s %9.2f%% %8d   $%s  (%s%s)" % (
            r["_e"], r["sum"], int(r["count"]), "{:,.0f}".format(out),
            "+" if out > 100_000 else "", "{:,.0f}".format(out - 100_000.0)))
    if len(bad) > 40:
        print(f"  … and {len(bad) - 40} more")

    print()
    print("Each of these conjures or destroys cash in any upstream trace that")
    print("passes through it — Upstream Analysis, Portfolio Analysis, PPI")
    print("upstream and PSCKOC alike. Fixing the feed fixes all four.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
