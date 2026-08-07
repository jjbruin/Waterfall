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

CSV_DIR = (r"C:\Users\cbui\OneDrive - peaceablestreet.com\Peaceable Street Capital - Documents"
           r"\Asset Mgmt\8. Asset_Mgmt_System\csv_data")
ISBS_CSV = os.path.join(CSV_DIR, "ISBS_Download.csv")
OCC_CSV = os.path.join(CSV_DIR, "MRI_Occupancy_Download.csv")

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
    """Run the committed get_one_pager_chart and dump results to JSON."""
    isbs, occ = load_data()
    out = {}
    for vcode, _ in DEALS:
        if side == "before":
            # main's signature is quarter-blind — one window per deal.
            out[vcode] = {"(data-derived)": get_one_pager_chart(vcode, isbs, occ)}
        else:
            out[vcode] = {q: get_one_pager_chart(vcode, isbs, occ, quarter=q)
                          for q in QUARTERS}
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


def report():
    with open(os.path.join(SCRATCH, "chart_before.json")) as f:
        before = json.load(f)
    with open(os.path.join(SCRATCH, "chart_after.json")) as f:
        after = json.load(f)

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

        for q in QUARTERS:
            a = after[vcode][q]
            zeros = sum(1 for i in range(len(a["periods"]))
                        if a["actual_noi"][i] == 0 and a["uw_noi"][i] == 0)
            gaps = sum(1 for i in range(len(a["periods"]))
                       if a["actual_noi"][i] is None or a["uw_noi"][i] is None)
            print(f"-- AFTER  quarter={q}  ({len(a['periods'])} quarters, "
                  f"{zeros} zero-filled, {gaps} with a gap) --")
            for i in range(len(a["periods"])):
                print("   " + fmt(a, i))

            # invariants
            yr, qn = q.split("-Q")
            if len(a["periods"]) != N:
                failures.append(f"{vcode} {q}: {len(a['periods'])} quarters, expected {N}")
            if a["periods"][-1] != f"Q{qn} {yr}":
                failures.append(f"{vcode} {q}: ends {a['periods'][-1]}, expected Q{qn} {yr}")
            if len({len(a[k]) for k in ("periods", "actual_noi", "uw_noi", "occupancy")}) != 1:
                failures.append(f"{vcode} {q}: series lengths differ")

            # no value may change on a quarter both sides reported
            bmap = dict(zip(b["periods"], b["actual_noi"]))
            for i, lbl in enumerate(a["periods"]):
                bv, av = bmap.get(lbl), a["actual_noi"][i]
                if bv is not None and av is not None and abs(bv - av) > 0.01:
                    failures.append(f"{vcode} {q}: {lbl} ACT drifted {bv:,.0f} -> {av:,.0f}")
        print()

    print("=" * 78)
    if failures:
        print(f"FAIL ({len(failures)})")
        for f_ in failures:
            print("  - " + f_)
        sys.exit(1)
    print("PASS — every window is exactly 10 quarters, ends at the selected "
          "quarter, series aligned, and no reported value drifted.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "capture":
        capture(sys.argv[2])
    else:
        report()
