"""Guardrail: One Pager NOI/Occupancy chart window — BEFORE vs AFTER.

Runs the *real* committed get_one_pager_chart on each side, no replication:

    git checkout main   && python scripts/onepager_chart_window_check.py capture before
    git checkout <feat> && python scripts/onepager_chart_window_check.py capture after
    python scripts/onepager_chart_window_check.py report

BEFORE (main)  = cap_to_last_actual=True, 12 quarters, window derived from the
                 union of periods that have data, quarter-blind.
AFTER (branch) = 10 consecutive calendar quarters ending at the selected report
                 quarter; quarters the deal predates read 0, an in-progress
                 quarter stays null.

The filtered ISBS slice is cached to the scratchpad so both runs are fast.

DATA SOURCE: local ISBS_Download.csv snapshot (Apr 15), NOT live Azure. This
snapshot's actuals stop at 25Q4, so 26Q1 is partial and 26Q2 is absent here —
on live data those are complete and the zero-vs-gap split shifts forward.
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask_app.services.financials_service import get_one_pager_chart

try:  # only present once the Date-Closed rule lands
    from flask_app.services.financials_service import _resolve_date_closed
except ImportError:  # older trees (before / mid captures)
    _resolve_date_closed = lambda inv, vcode: None  # noqa: E731

CSV_DIR = (r"C:\Users\cbui\OneDrive - peaceablestreet.com\Peaceable Street Capital - Documents"
           r"\Asset Mgmt\8. Asset_Mgmt_System\csv_data")
ISBS_CSV = os.path.join(CSV_DIR, "ISBS_Download.csv")
OCC_CSV = os.path.join(CSV_DIR, "MRI_Occupancy_Download.csv")
ACCT_CSV = os.path.join(CSV_DIR, "accounting_feed.csv")
MAP_CSV = os.path.join(CSV_DIR, "investment_map.csv")

SCRATCH = (r"C:\Users\cbui\AppData\Local\Temp\1\claude"
           r"\C--Users-cbui-Documents-waterfall-xirr"
           r"\fea2fef0-f2bb-45a6-a127-5d60708f31f0\scratchpad")
CACHE = os.path.join(SCRATCH, "isbs_slice.pkl")

# vcode -> label. p0000109/113 first report 25Q3 (mid-window acquisition).
DEALS = [
    ("p0000082", "long history (full window)"),
    ("p0000084", "long history (full window)"),
    ("p0000099", "first actual 25Q2"),
    ("p0000109", "Burton — first actual 25Q3  <-- mid-window acquisition"),
    ("p0000113", "first actual 25Q3  <-- mid-window acquisition"),
    ("p0000085", "first actual 25Q4  <-- very late start"),
]
QUARTERS = ["2026-Q1", "2026-Q2"]
N = 10


def normalize_isbs(df):
    """Mirror of data_service._normalize_isbs (kept local to avoid Flask ctx)."""
    df.columns = [c.strip() for c in df.columns]
    df["vcode"] = df["vcode"].astype(str).str.strip().str.lower()
    df["vSource"] = df["vSource"].astype(str).str.strip()
    df["vAccount"] = (df["vAccount"].astype(str).str.strip()
                      .str.replace(r"\.0$", "", regex=True))
    df["mAmount"] = pd.to_numeric(df["mAmount"], errors="coerce").fillna(0.0)
    df["dtEntry_parsed"] = pd.to_datetime(df["dtEntry"], errors="coerce")
    return df


def load_inv():
    """investment_map with Acquisition_Date enriched from the accounting feed —
    mirrors data_service._enrich_acquisition_dates, which is what live `inv`
    carries."""
    inv = pd.read_csv(MAP_CSV, dtype=str)
    inv.columns = [c.strip() for c in inv.columns]
    acct = pd.read_csv(ACCT_CSV, dtype=str, usecols=["InvestmentID", "EffectiveDate"])
    acct["_dt"] = pd.to_datetime(acct["EffectiveDate"], errors="coerce")
    earliest = acct.dropna(subset=["_dt"]).groupby(
        acct["InvestmentID"].astype(str).str.strip())["_dt"].min()
    inv["Acquisition_Date"] = (
        inv["InvestmentID"].astype(str).str.strip()
        .map(earliest).fillna(pd.to_datetime(inv.get("Acquisition_Date"), errors="coerce"))
    )
    return inv


def load_data():
    if os.path.exists(CACHE):
        isbs = pd.read_pickle(CACHE)
    else:
        keep = {v for v, _ in DEALS}
        parts = []
        for ch in pd.read_csv(ISBS_CSV, dtype=str, chunksize=200_000):
            ch.columns = [c.strip() for c in ch.columns]
            ch = ch[ch["vcode"].astype(str).str.strip().str.lower().isin(keep)]
            if not ch.empty:
                parts.append(ch)
        isbs = normalize_isbs(pd.concat(parts, ignore_index=True))
        isbs.to_pickle(CACHE)
    return isbs, pd.read_csv(OCC_CSV, dtype=str)


def capture(side):
    """Run the committed get_one_pager_chart on this tree and dump to JSON.

    before = origin/main (quarter-blind, cap_to_last_actual)
    mid    = calendar window, zero-fill driven purely by absence of data
    after  = same, plus pre-close quarters forced to 0 from Date Closed
    """
    isbs, occ = load_data()
    inv = load_inv() if side == "after" else None
    out = {}
    for vcode, _ in DEALS:
        if side == "before":
            # main's signature is quarter-blind — one window per deal.
            entry = {"(data-derived)": get_one_pager_chart(vcode, isbs, occ)}
        elif side == "mid":
            entry = {q: get_one_pager_chart(vcode, isbs, occ, quarter=q)
                     for q in QUARTERS}
        else:
            entry = {q: get_one_pager_chart(vcode, isbs, occ, quarter=q, inv=inv)
                     for q in QUARTERS}
            dc = _resolve_date_closed(inv, vcode)
            entry["_date_closed"] = None if dc is None else str(pd.Timestamp(dc).date())
        out[vcode] = entry
    path = os.path.join(SCRATCH, f"chart_{side}.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"wrote {path}")


def fmt(c, i):
    def m(v):
        return " gap  " if v is None else f"{v / 1e6:6.2f}"

    def p(v):
        return " -- " if v is None else f"{v:4.1f}"
    return (f"{c['periods'][i]:<9} ACT {m(c['actual_noi'][i])}  "
            f"UW {m(c['uw_noi'][i])}  OCC {p(c['occupancy'][i])}")


def _pre_close(label, dc):
    """Is this quarter label's period end strictly before Date Closed?"""
    if not dc:
        return False
    qn, yr = label.split()[0][1:], label.split()[1]
    end = pd.Timestamp(year=int(yr), month=int(qn) * 3, day=1) + pd.offsets.MonthEnd(0)
    return end < pd.Timestamp(dc)


def report():
    with open(os.path.join(SCRATCH, "chart_before.json")) as f:
        before = json.load(f)
    with open(os.path.join(SCRATCH, "chart_after.json")) as f:
        after = json.load(f)
    mid_path = os.path.join(SCRATCH, "chart_mid.json")
    mid = json.load(open(mid_path)) if os.path.exists(mid_path) else None

    failures = []
    for vcode, note in DEALS:
        print("=" * 78)
        print(f"{vcode}   {note}")
        print("=" * 78)

        b = before[vcode]["(data-derived)"]
        print(f"-- BEFORE (main, quarter-blind, {len(b['periods'])} quarters) --")
        for i in range(len(b["periods"])):
            print("   " + fmt(b, i))
        if not b["periods"]:
            print("   (empty)")

        dc = after[vcode].get("_date_closed")
        for q in QUARTERS:
            a = after[vcode][q]
            zeros = sum(1 for i in range(len(a["periods"]))
                        if a["actual_noi"][i] == 0 and a["uw_noi"][i] == 0)
            gaps = sum(1 for i in range(len(a["periods"]))
                       if a["actual_noi"][i] is None or a["uw_noi"][i] is None)
            print(f"-- AFTER  quarter={q}  (Date Closed {dc or 'unknown'}; "
                  f"{len(a['periods'])} quarters, {zeros} zero-filled, "
                  f"{gaps} with a gap) --")
            for i in range(len(a["periods"])):
                tag = "   PRE-CLOSE" if _pre_close(a["periods"][i], dc) else ""
                print("   " + fmt(a, i) + tag)

            # invariants
            yr, qn = q.split("-Q")
            if len(a["periods"]) != N:
                failures.append(f"{vcode} {q}: {len(a['periods'])} quarters, expected {N}")
            if a["periods"][-1] != f"Q{qn} {yr}":
                failures.append(f"{vcode} {q}: ends {a['periods'][-1]}, expected Q{qn} {yr}")
            if len({len(a[k]) for k in ("periods", "actual_noi", "uw_noi", "occupancy")}) != 1:
                failures.append(f"{vcode} {q}: series lengths differ")

            # every pre-close quarter must be a hard zero on all three series
            for i, lbl in enumerate(a["periods"]):
                if not _pre_close(lbl, dc):
                    continue
                vals = (a["actual_noi"][i], a["uw_noi"][i], a["occupancy"][i])
                if any(v != 0 for v in vals):
                    failures.append(f"{vcode} {q}: pre-close {lbl} not zeroed -> {vals}")

            # no value may change on a post-close quarter both sides reported
            bmap = dict(zip(b["periods"], b["actual_noi"]))
            for i, lbl in enumerate(a["periods"]):
                if _pre_close(lbl, dc):
                    continue  # intentionally rewritten
                bv, av = bmap.get(lbl), a["actual_noi"][i]
                if bv is not None and av is not None and abs(bv - av) > 0.01:
                    failures.append(f"{vcode} {q}: {lbl} ACT drifted {bv:,.0f} -> {av:,.0f}")

            # post-close quarters must be untouched by the Date-Closed rule
            if mid:
                m = mid[vcode][q]
                for i, lbl in enumerate(a["periods"]):
                    if _pre_close(lbl, dc):
                        continue
                    for k in ("actual_noi", "uw_noi", "occupancy"):
                        if m[k][i] != a[k][i]:
                            failures.append(
                                f"{vcode} {q}: post-close {lbl} {k} regressed "
                                f"{m[k][i]} -> {a[k][i]}")
        print()

    if mid:
        print("#" * 78)
        print("# PRE-CLOSE ENFORCEMENT — data-driven zero (mid) vs Date Closed (after)")
        print("#" * 78)
        changed = 0
        for vcode, _ in DEALS:
            dc = after[vcode].get("_date_closed")
            rows = []
            for q in QUARTERS:
                m, a = mid[vcode][q], after[vcode][q]
                for i, lbl in enumerate(a["periods"]):
                    if not _pre_close(lbl, dc):
                        continue
                    for k, nm in (("actual_noi", "ACT"), ("uw_noi", "UW"),
                                  ("occupancy", "OCC")):
                        if m[k][i] != a[k][i]:
                            rows.append(f"     {q}  {lbl:<9} {nm:<4} "
                                        f"{m[k][i]} -> {a[k][i]}")
            print(f"  {vcode}  Date Closed {dc or 'unknown'}")
            if rows:
                changed += len(rows)
                for r in rows:
                    print(r)
            else:
                print("     (no pre-close value changed — already zero)")
        print(f"\n  {changed} stray pre-close value(s) now forced to zero.\n")

    print("=" * 78)
    if failures:
        print(f"FAIL ({len(failures)})")
        for f_ in failures:
            print("  - " + f_)
        sys.exit(1)
    print("PASS — every window is exactly 10 quarters, ends at the selected "
          "quarter, series aligned, every pre-close quarter is a hard zero on "
          "all three series, and no post-close quarter changed.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "capture":
        capture(sys.argv[2])
    else:
        report()
