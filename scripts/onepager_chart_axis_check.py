"""Guardrail: One Pager NOI/Occupancy chart — right-axis (NOI) scaling.

The left axis is pinned 0-100 for occupancy, so a bar's top sits at occ/100 of
the plot height. The right axis had no min/max, so ECharts auto-fit it to the
NOI data: the peak NOI point always landed near the top of the plot area and the
auto *min* usually sat above zero, lifting the whole line off the baseline. The
lines therefore rendered above the bar tops, reading as "NOI > occupancy" even
though the units are unrelated.

This script replicates buildChartOption()'s axis math (BEFORE = ECharts nice
auto-scale, AFTER = the committed formula) against the real chart payloads from
get_one_pager_chart, and reports, per quarter, where each NOI point lands as a
fraction of plot height versus that quarter's bar top.

    python scripts/onepager_chart_axis_check.py

PASS requires, for every deal/quarter:
  * no NOI point above its own quarter's bar top      (the reported bug)
  * nothing clipped: axis max >= max NOI, min <= min NOI
  * not flattened: the peak NOI point still uses >= 33% of the right axis

Worst compression in the sample is Old Kinderhook (p0000031) at 37%: its
occupancy feed is 0.0 for nine of ten quarters with a single 38.2% reading, and
it has negative-NOI quarters, so the axis floor goes to -0.50 while the peak
(1.15) is held to 45% of the plot height. The lines still span ~36% of the plot,
so the shape reads fine — that is the honest cost of the constraint, not a bug.

DATA SOURCE: local ISBS_Download.csv snapshot (Apr 15), NOT live Azure.
"""
import math
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask_app.services.financials_service import get_one_pager_chart

CSV_DIR = (r"C:\Users\cbui\OneDrive - peaceablestreet.com\Peaceable Street Capital - Documents"
           r"\Asset Mgmt\8. Asset_Mgmt_System\csv_data")
ISBS_CSV = os.path.join(CSV_DIR, "ISBS_Download.csv")
OCC_CSV = os.path.join(CSV_DIR, "MRI_Occupancy_Download.csv")
ACCT_CSV = os.path.join(CSV_DIR, "accounting_feed.csv")
MAP_CSV = os.path.join(CSV_DIR, "investment_map.csv")

SCRATCH = (r"C:\Users\cbui\AppData\Local\Temp\1\claude"
           r"\C--Users-cbui-Documents-waterfall-xirr"
           r"\b08b6806-6e02-42a2-b86c-80a95b30b00b\scratchpad")
CACHE = os.path.join(SCRATCH, "isbs_axis_slice.pkl")

# Deliberately spread across NOI magnitudes, from single-tenant retail (~$0.1M/qtr)
# to multi-property portfolios (~$3M+/qtr).
DEALS = [
    ("p0000109", "Burton Portfolio           (mid-window acq, 7 zero qtrs)"),
    ("p0000113", "Burton - Westwood Plaza    (child, inherits parent date)"),
    ("p0000111", "Burton - Foley Square      (child)"),
    ("p0000101", "Town Fair Tire - Avon      (tiny single-tenant NOI)"),
    ("p0000107", "Town Fair Tire Portfolio   (parent, aggregated)"),
    ("p0000033", "OREI Portfolio             (large multifamily)"),
    ("p0000036", "PMAT Midwest Portfolio     (large retail)"),
    ("p0000031", "Old Kinderhook Resort      (volatile occupancy)"),
    ("p0000082", "Poplar Prairie             (long history)"),
    ("p0000099", "ReNew Glenmoore            (closed 2025-02)"),
    ("p0000085", "Jefferson Eastchase        (late first actual)"),
]
QUARTERS = ["2026-Q1", "2026-Q2"]


def normalize_isbs(df):
    df.columns = [c.strip() for c in df.columns]
    df["vcode"] = df["vcode"].astype(str).str.strip().str.lower()
    df["vSource"] = df["vSource"].astype(str).str.strip()
    df["vAccount"] = (df["vAccount"].astype(str).str.strip()
                      .str.replace(r"\.0$", "", regex=True))
    df["mAmount"] = pd.to_numeric(df["mAmount"], errors="coerce").fillna(0.0)
    df["dtEntry_parsed"] = pd.to_datetime(df["dtEntry"], errors="coerce")
    return df


def load_inv():
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


# --------------------------------------------------------------------------
# axis math
# --------------------------------------------------------------------------
NICE_STEPS = [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]


def nice_ceil(v):
    """Mirror of niceCeil() in OnePagerView.vue."""
    if not v > 0:
        return 0.0
    mag = 10 ** math.floor(math.log10(v / 5))
    for s in NICE_STEPS:
        if s * mag * 5 >= v:
            return s * mag * 5
    return math.ceil(v)


def after_axis(uw, act, occ):
    """Mirror of the committed right-axis block in buildChartOption()."""
    noi = [v for v in list(uw) + list(act) if v is not None]
    bars = [v for v in occ if v is not None and v > 0]
    max_noi = max(noi) if noi else 0.0
    min_noi = min(noi) if noi else 0.0
    headroom = min(0.95, max(0.45, (min(bars) if bars else 95) / 100))
    lo = -nice_ceil(-min_noi) if min_noi < 0 else 0.0
    top = max(max_noi, 0.0)
    hi = nice_ceil(max(lo + (top - lo) / headroom, 0.05))
    return lo, hi, headroom


def before_axis(uw, act):
    """Approximation of ECharts' nice auto-scale (splitNumber 5, no min/max)."""
    noi = [v for v in list(uw) + list(act) if v is not None]
    if not noi:
        return 0.0, 1.0
    lo, hi = min(noi), max(noi)
    if hi == lo:
        return (0.0, hi * 2) if hi else (0.0, 1.0)
    span = hi - lo
    raw = span / 5
    mag = 10 ** math.floor(math.log10(raw))
    step = next((s * mag for s in [1, 2, 5, 10] if s * mag >= raw), 10 * mag)
    return math.floor(lo / step) * step, math.ceil(hi / step) * step


def main():
    isbs, occ_raw = load_data()
    inv = load_inv()
    failures = []
    print("=" * 100)
    print("Right-axis (NOI) scaling - fraction of PLOT HEIGHT used, per quarter")
    print("  bar = occ/100 (left axis 0-100).  line = (noi - min) / (max - min).")
    print("  FAIL = a NOI point rendering ABOVE that quarter's bar top.")
    print("=" * 100)

    for vcode, label in DEALS:
        for q in QUARTERS:
            cr = get_one_pager_chart(vcode, isbs, occ_raw, quarter=q, inv=inv)
            # buildChartOption() scales to $M and rounds to 2dp before plotting.
            uw = [None if v is None else round(v / 1e6, 2) for v in cr["uw_noi"]]
            act = [None if v is None else round(v / 1e6, 2) for v in cr["actual_noi"]]
            occ = [None if v is None else round(v, 1) for v in cr["occupancy"]]

            b_lo, b_hi = before_axis(uw, act)
            a_lo, a_hi, headroom = after_axis(uw, act, occ)
            noi = [v for v in uw + act if v is not None]

            print(f"\n{vcode}  {label}   report quarter {q}")
            print(f"  BEFORE axis (auto, approx): [{b_lo:.2f}, {b_hi:.2f}]"
                  f"    AFTER axis: [{a_lo:.2f}, {a_hi:.2f}]  headroom={headroom:.2f}")

            def frac(v, lo, hi):
                return None if v is None or hi == lo else (v - lo) / (hi - lo)

            worst_before = worst_after = -1.0
            peak_after = 0.0
            for i, per in enumerate(cr["periods"]):
                bar = None if occ[i] is None else occ[i] / 100
                fb = [frac(uw[i], b_lo, b_hi), frac(act[i], b_lo, b_hi)]
                fa = [frac(uw[i], a_lo, a_hi), frac(act[i], a_lo, a_hi)]
                peak_after = max([peak_after] + [f for f in fa if f is not None])
                over_b = over_a = ""
                if bar is not None and bar > 0:
                    for f in [f for f in fb if f is not None]:
                        worst_before = max(worst_before, f - bar)
                    for f in [f for f in fa if f is not None]:
                        worst_after = max(worst_after, f - bar)
                    if any(f > bar + 1e-9 for f in fb if f is not None):
                        over_b = " <-BEFORE line above bar"
                    if any(f > bar + 1e-9 for f in fa if f is not None):
                        over_a = "  ***AFTER STILL ABOVE BAR***"

                def s(v):
                    return " gap " if v is None else f"{v:5.2f}"

                def sf(v):
                    return "  -- " if v is None else f"{v * 100:4.0f}%"

                print(f"    {per:<9} occ {'  -- ' if occ[i] is None else f'{occ[i]:5.1f}'}"
                      f" (bar {'  -- ' if bar is None else f'{bar * 100:4.0f}%'})"
                      f"  UW {s(uw[i])} ACT {s(act[i])}"
                      f"  | before {sf(fb[0])}/{sf(fb[1])}"
                      f"  after {sf(fa[0])}/{sf(fa[1])}{over_b}{over_a}")

            clipped = bool(noi) and (a_hi < max(noi) - 1e-9 or a_lo > min(noi) + 1e-9)

            def ov(v):
                # -1.0 is the sentinel for "window has no bars to compare against"
                return " n/a" if v < -0.999 else f"{v * 100:+.0f}%"

            print(f"    -> worst overshoot before: {ov(worst_before)} of plot height"
                  f" | after: {ov(worst_after)}"
                  f" | peak NOI uses {peak_after * 100:.0f}% of right axis"
                  f" | clipped: {clipped}")
            if worst_after > 1e-9:
                failures.append(f"{vcode} {q}: NOI above bar by {worst_after * 100:.0f}%")
            if clipped:
                failures.append(f"{vcode} {q}: CLIPPED")
            if noi and max(noi) > 0 and peak_after < 0.33:
                failures.append(f"{vcode} {q}: FLATTENED (peak uses {peak_after * 100:.0f}%)")

    print("\n" + "=" * 100)
    if failures:
        print("FAIL")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("PASS - every NOI point sits at or below its quarter's bar top;"
          " nothing clipped; peak NOI still uses >=33% of the right axis.")


if __name__ == "__main__":
    main()
