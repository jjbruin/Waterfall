"""Guardrail for the DSCR payoff/refi principal fix.

Runs the REAL committed get_property_performance() on each side of the change and
diffs every deal x every DSCR column, so the blast radius is measured rather than
argued about.  Nothing is written back to the app or the database.

The fix only touches _get_bs_principal_change(), which feeds exactly two columns —
YTD Actual and Projected YE.  YTD Budget (5190+7060 from Budget IS), U/W YE
(account 7010 from Projected IS) and At Close (the at_close_noi table / earliest
Projected IS December) never consult it, so `report` asserts they are untouched.

Usage
-----
    # before: run from a worktree checked out at main
    python scripts/dscr_payoff_fix_check.py capture before.csv

    # after: run from the working tree carrying the fix
    python scripts/dscr_payoff_fix_check.py capture after.csv

    python scripts/dscr_payoff_fix_check.py report before.csv after.csv

By default both sides sweep the live-equivalent quarter (the most recent completed
quarter present in the snapshot).  Pass --quarters 2026-Q1,2025-Q4,... to widen the
sweep — users can pick any quarter from the One Pager dropdown, so a single-quarter
diff understates the blast radius.

The ISBS snapshot is the ~103MB CSV export; parsing it takes minutes, so the
needed columns are cached as a pickle under the system temp dir.  Point
WF_CSV_DIR at a different snapshot to re-run elsewhere.
"""
import argparse
import os
import sys
import tempfile

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

CSV_DIR = os.environ.get(
    "WF_CSV_DIR",
    r"C:\Users\cbui\OneDrive - peaceablestreet.com\Peaceable Street Capital"
    r" - Documents\Asset Mgmt\8. Asset_Mgmt_System\csv_data",
)
CACHE_DIR = os.path.join(tempfile.gettempdir(), "waterfall_xirr_guardrails")
ISBS_CACHE = os.path.join(CACHE_DIR, "isbs_dscr_cols.pkl")

DSCR_COLS = [
    ("dscr_ytd_actual", "YTD Actual"),
    ("dscr_ytd_budget", "YTD Budget"),
    ("dscr_actual_ye", "Projected YE"),
    ("dscr_uw_ye", "U/W YE"),
    ("dscr_at_close", "At Close"),
]
# Columns the fix must not be able to reach.
UNTOUCHABLE = ["dscr_ytd_budget", "dscr_uw_ye", "dscr_at_close"]


def _load_isbs():
    """ISBS with only the columns the DSCR path reads, normalized as at load time."""
    from flask_app.services.data_service import _normalize_isbs

    if os.path.exists(ISBS_CACHE):
        raw = pd.read_pickle(ISBS_CACHE)
    else:
        src = os.path.join(CSV_DIR, "ISBS_Download.csv")
        cols = ["vcode", "dtEntry", "vSource", "vAccount", "mAmount"]
        parts = [
            chunk
            for chunk in pd.read_csv(
                src, chunksize=100_000, dtype=object, low_memory=False, usecols=cols
            )
        ]
        raw = pd.concat(parts, ignore_index=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        raw.to_pickle(ISBS_CACHE)
    return _normalize_isbs(raw)


def _names():
    imap = pd.read_csv(os.path.join(CSV_DIR, "investment_map.csv"), dtype=object)
    imap.columns = [c.strip() for c in imap.columns]
    by_vcode = {
        str(r["vcode"]).strip().lower(): str(r["Investment_Name"])
        for _, r in imap.iterrows()
    }
    # Legacy vcodes live only in MRI_Loans; without this they report as '?'.
    loans = pd.read_csv(os.path.join(CSV_DIR, "MRI_Loans.csv"), dtype=object)
    for _, r in loans.iterrows():
        key = str(r.get("vCode", "")).strip().lower()
        if key and key not in by_vcode:
            by_vcode[key] = str(r.get("vPropertyName", ""))
    return by_vcode


def capture(out_path, quarters=None):
    from one_pager import (
        get_available_quarters,
        get_property_performance,
        most_recent_completed_quarter,
    )

    isbs = _load_isbs()
    loans_all = pd.read_csv(os.path.join(CSV_DIR, "MRI_Loans.csv"), dtype=object)
    names = _names()

    if quarters:
        qtrs = quarters
    else:
        qtrs = [most_recent_completed_quarter(get_available_quarters(isbs))]
    print(f"quarters: {', '.join(qtrs)}")

    rows = []
    for qtr in qtrs:
        for vcode in sorted(isbs["vcode"].dropna().unique()):
            row = {
                "vcode": vcode,
                "name": names.get(vcode, "?"),
                "quarter": qtr,
                "error": None,
            }
            try:
                perf = get_property_performance(
                    vcode, qtr, isbs, None, None, mri_loans_all_df=loans_all
                )
            except Exception as exc:  # a crash is a regression too — record it
                row["error"] = str(exc)
                rows.append(row)
                continue
            dscr = perf.get("dscr", {})
            for key, _ in DSCR_COLS:
                row[key] = dscr.get(key.replace("dscr_", ""))
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(
        f"wrote {out_path}: {len(df)} rows, {df['vcode'].nunique()} deals, "
        f"{int(df['error'].notna().sum())} errors"
    )


def _same(x, y):
    if pd.isna(x) and pd.isna(y):
        return True
    if pd.isna(x) or pd.isna(y):
        return False
    return abs(x - y) <= 1e-9 * max(1.0, abs(x))


def report(before_path, after_path):
    before = pd.read_csv(before_path)
    after = pd.read_csv(after_path)
    key = ["vcode", "quarter"]
    merged = before.merge(after, on=key, suffixes=("_b", "_a"))
    if not (len(merged) == len(before) == len(after)):
        print(f"FAIL: row sets differ (before {len(before)}, after {len(after)}, "
              f"joined {len(merged)})")
        return 1

    failures = []
    print(f"compared {len(merged)} deal-quarters across "
          f"{merged['vcode'].nunique()} deals, "
          f"{merged['quarter'].nunique()} quarter(s)\n")

    print("per-column change counts")
    changed_rows = set()
    for col, label in DSCR_COLS:
        hits = [
            (r["vcode"], r["quarter"])
            for _, r in merged.iterrows()
            if not _same(r[col + "_b"], r[col + "_a"])
        ]
        changed_rows.update(hits)
        flag = ""
        if col in UNTOUCHABLE and hits:
            flag = "   <-- FAIL: fix must not reach this column"
            failures.append(f"{label} changed on {len(hits)} deal-quarter(s)")
        print(f"   {label:<14} {len(hits):>4} / {len(merged)}{flag}")

    unchanged = len(merged) - len(changed_rows)
    print(f"\n   deal-quarters changed   : {len(changed_rows)}")
    print(f"   deal-quarters unchanged : {unchanged} "
          f"({100 * unchanged / len(merged):.1f}%)")

    if changed_rows:
        print("\nchanged deal-quarters")
        for _, r in merged.sort_values(key).iterrows():
            hits = [c for c, _ in DSCR_COLS if not _same(r[c + "_b"], r[c + "_a"])]
            if not hits:
                continue
            print(f"\n  {str(r['vcode']).upper()}  {r['name_b']}  [{r['quarter']}]")
            for col, label in DSCR_COLS:
                fb = "—" if pd.isna(r[col + "_b"]) else f"{r[col + '_b']:.4f}"
                fa = "—" if pd.isna(r[col + "_a"]) else f"{r[col + '_a']:.4f}"
                if col in hits:
                    print(f"     {label:<14} {fb:>9} -> {fa:>9}   CHANGED")
                else:
                    print(f"     {label:<14} {fb:>9}     unchanged")

    # A DSCR that got worse means a clean month was misread as an event.
    worse = [
        (r["vcode"], r["quarter"], col)
        for _, r in merged.iterrows()
        for col in ("dscr_ytd_actual", "dscr_actual_ye")
        if not _same(r[col + "_b"], r[col + "_a"])
        and pd.notna(r[col + "_b"])
        and pd.notna(r[col + "_a"])
        and r[col + "_a"] < r[col + "_b"]
    ]
    if worse:
        failures.append(f"{len(worse)} deal-quarter(s) moved DOWN")
        print("\nFAIL: DSCR moved down (the fix can only remove overstated principal)")
        for vcode, qtr, col in worse:
            print(f"   {vcode} {qtr} {col}")

    errs = merged[merged["error_a"].notna()]
    if len(errs):
        failures.append(f"{len(errs)} deal-quarter(s) raised")
        print(f"\nFAIL: {len(errs)} deal-quarter(s) raised after the fix")

    print("\n" + ("FAILURES: " + "; ".join(failures) if failures else "PASS"))
    return 1 if failures else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("out")
    cap.add_argument("--quarters", default=None,
                     help="comma-separated, e.g. 2026-Q1,2025-Q4")
    rep = sub.add_parser("report")
    rep.add_argument("before")
    rep.add_argument("after")
    args = ap.parse_args()

    if args.cmd == "capture":
        qtrs = [q.strip() for q in args.quarters.split(",")] if args.quarters else None
        capture(args.out, qtrs)
        return 0
    return report(args.before, args.after)


if __name__ == "__main__":
    sys.exit(main())
