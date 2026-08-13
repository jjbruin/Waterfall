"""Guardrail: the One Pager's no-quarter default must match the dropdown label.

THE BUG. OnePagerView.vue's first load deliberately sends no `quarter` param and
only afterwards labels the dropdown with getMostRecentCompletedQuarter(). The
server used to fill that gap with `available[0]` — the NEWEST quarter with
actuals anywhere in the portfolio. While the newest quarter was also the most
recent completed one the two agreed and nothing showed. The moment 26Q3 actuals
landed they diverged: a page labelled 2026-Q2 rendered 2026-Q3 figures. YTD
Budget is a cumulative Jan-to-date sum, so it absorbed a whole extra quarter and
overstated by ~51%, while YTD Actual (anchored to each deal's last reported
month) barely moved — which is why the budget column looked wrong first.

THE FIX. financials_service.get_one_pager_data() defaults to
one_pager.most_recent_completed_quarter(available), the server-side port of the
Vue rule.

WHAT THIS CHECKS.
  1. Rule parity — the helper matches an independent port of the Vue function
     across a matrix of dates and quarter lists, including quarter boundaries,
     lists with holes, and lists with nothing completed.
  2. Real data — with an in-progress quarter present, the served default lands
     on the labelled quarter and its YTD Budget window is Jan..labelled-quarter
     rather than bleeding a quarter further.
  3. Wiring — the production call site actually uses the helper, so reverting it
     to available[0] fails here rather than silently in the browser.

    python scripts/onepager_quarter_default_check.py

DATA SOURCE: local ISBS_Download.csv snapshot (Apr 15 2026), NOT live Azure.
That snapshot's actuals stop at 2026-03-31 and so contain no in-progress
quarter — the condition that triggers the bug. Section 2 therefore injects
synthetic actuals for ONE donor deal to reproduce Jim's 26Q3 load. The injected
rows are clones of that deal's latest monthly snapshot: they move which periods
EXIST, which is all the default rule reads. They are not meant to be realistic
amounts, and no assertion depends on their values.
"""
import inspect
import math
import os
import sys
import tempfile
from datetime import date

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask_app.services.data_service import _normalize_isbs
from flask_app.services import financials_service
from one_pager import (
    IS_ACCOUNTS,
    get_available_quarters,
    get_property_performance,
    most_recent_completed_quarter,
    quarter_to_date_range,
)

CSV_DIR = (r"C:\Users\cbui\OneDrive - peaceablestreet.com\Peaceable Street Capital - Documents"
           r"\Asset Mgmt\8. Asset_Mgmt_System\csv_data")
ISBS_CSV = os.path.join(CSV_DIR, "ISBS_Download.csv")

SCRATCH = os.path.join(tempfile.gettempdir(), "waterfall_xirr_guardrails")
os.makedirs(SCRATCH, exist_ok=True)
CACHE = os.path.join(SCRATCH, "isbs_quarter_default_slice.pkl")

failures = []


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        failures.append(label)


# ---------------------------------------------------------------- section 1
def vue_most_recent_completed(quarters, today):
    """Independent port of getMostRecentCompletedQuarter() in OnePagerView.vue.

    Deliberately transcribed from the JS rather than sharing code with the
    helper, so a change to one side shows up as a parity failure.
    """
    cur_year = today.year
    cur_quarter = math.ceil(today.month / 3)

    def parse(q):
        y, qs = q.split('-')
        return int(y), int(qs.replace('Q', ''))

    completed = [q for q in quarters
                 if parse(q)[0] < cur_year
                 or (parse(q)[0] == cur_year and parse(q)[1] < cur_quarter)]
    if not completed:
        return quarters[-1]
    best = completed[0]
    for q in completed[1:]:
        by, bq = parse(q)
        ay, aq = parse(best)
        if by > ay or (by == ay and bq > aq):
            best = q
    return best


FULL_Q3 = ["2026-Q3", "2026-Q2", "2026-Q1", "2025-Q4", "2025-Q3"]
FULL_Q2 = ["2026-Q2", "2026-Q1", "2025-Q4", "2025-Q3"]
HOLED = ["2026-Q3", "2026-Q1", "2025-Q4"]          # Q2 never reported
ONLY_OPEN = ["2026-Q3"]                             # nothing completed yet

CASES = [
    ("Q3 actuals present, mid-Q3", FULL_Q3, date(2026, 8, 13), "2026-Q2"),
    ("no Q3 yet, mid-Q3", FULL_Q2, date(2026, 8, 13), "2026-Q2"),
    ("first day of Q3", FULL_Q3, date(2026, 7, 1), "2026-Q2"),
    ("last day of Q2 (Q2 not done)", FULL_Q3, date(2026, 6, 30), "2026-Q1"),
    ("last day of Q4", FULL_Q3, date(2026, 12, 31), "2026-Q3"),
    ("new year, Jan 1", FULL_Q3, date(2027, 1, 1), "2026-Q3"),
    ("list with a hole at Q2", HOLED, date(2026, 8, 13), "2026-Q1"),
    ("nothing completed -> oldest", ONLY_OPEN, date(2026, 8, 13), "2026-Q3"),
]

print("=" * 78)
print("1. RULE PARITY  (helper vs independent port of the Vue function)")
print("=" * 78)
for label, quarters, today, expected in CASES:
    got = most_recent_completed_quarter(quarters, today)
    vue = vue_most_recent_completed(quarters, today)
    check(f"{label:<30} {today}  -> {got}",
          got == expected and got == vue,
          "" if got == expected and got == vue else f"expected {expected}, vue said {vue}")

check("empty list returns None", most_recent_completed_quarter([]) is None)
check("malformed entries ignored",
      most_recent_completed_quarter(["2026-Q3", "garbage", "2026-Q1"], date(2026, 8, 13))
      == "2026-Q1")

# ---------------------------------------------------------------- section 3
print()
print("=" * 78)
print("3. WIRING  (production call site uses the helper)")
print("=" * 78)
src = inspect.getsource(financials_service.get_one_pager_data)
check("get_one_pager_data calls most_recent_completed_quarter",
      "most_recent_completed_quarter(available)" in src)
check("available[0] is no longer the primary default",
      "quarter_str = available[0]" not in src)

# ---------------------------------------------------------------- section 2
print()
print("=" * 78)
print("2. REAL DATA  (served quarter and its budget window)")
print("=" * 78)

if not os.path.exists(CACHE) and not os.path.exists(ISBS_CSV):
    print(f"  SKIP - no local ISBS snapshot at {ISBS_CSV}")
else:
    if os.path.exists(CACHE):
        df = pd.read_pickle(CACHE)
    else:
        print("  slicing ISBS_Download.csv (a few minutes, cached afterwards) ...")
        head = pd.read_csv(ISBS_CSV, nrows=5)
        cols = [c for c in head.columns
                if c.strip().lower() in ("vcode", "vsource", "dtentry", "vaccount", "mamount")]
        df = pd.concat(pd.read_csv(ISBS_CSV, usecols=cols, chunksize=200_000, dtype=str),
                       ignore_index=True)
        df.to_pickle(CACHE)
    df = _normalize_isbs(df)

    TODAY = date(2026, 8, 13)

    # Reproduce Jim's 26Q3 load: give ONE deal in-progress-quarter actuals.
    act = df[df["vSource"] == "Interim IS"]
    donor = act.groupby("vcode")["dtEntry_parsed"].max().idxmax()
    donor_anchor = act[act["vcode"] == donor]["dtEntry_parsed"].max()
    src_rows = act[(act["vcode"] == donor) & (act["dtEntry_parsed"] == donor_anchor)]
    injected = []
    for d in ["2026-04-30", "2026-05-31", "2026-06-30", "2026-07-31"]:
        r = src_rows.copy()
        r["dtEntry_parsed"] = pd.Timestamp(d)
        r["dtEntry"] = d
        injected.append(r)
    df_q3 = pd.concat([df] + injected, ignore_index=True)
    print(f"  donor deal {donor}: injected actuals through 2026-07-31 "
          f"(in-progress 2026-Q3)\n")

    available = get_available_quarters(df_q3)
    label = vue_most_recent_completed(available, TODAY)      # what the dropdown says
    new_default = most_recent_completed_quarter(available, TODAY)   # what we now serve
    old_default = available[0]                                # what we used to serve

    print(f"  dropdown label (Vue) ......... {label}")
    print(f"  served default (fixed) ....... {new_default}")
    print(f"  served default (old bug) ..... {old_default}")
    check("served default equals dropdown label", new_default == label)
    check("old default did diverge (bug reproduced)", old_default != label,
          f"{old_default} != {label}")

    def budget_months(vcode, quarter):
        """Periods the YTD Budget sums, replicating one_pager.py:956-960.

        Verified against the real function's ytd_budget by the caller.
        """
        _, qe = quarter_to_date_range(quarter)
        bud = df_q3[(df_q3["vcode"] == vcode) & (df_q3["vSource"] == "Budget IS")]
        jan1 = pd.Timestamp(f"{quarter.split('-')[0]}-01-01") - pd.DateOffset(days=1)
        sel = bud[(bud["dtEntry_parsed"] > jan1) & (bud["dtEntry_parsed"] <= pd.Timestamp(qe))]
        rev = exp = 0.0
        for _, accts in IS_ACCOUNTS["REVENUES"].items():
            rev += -sel[sel["vAccount"].isin(accts)]["mAmount"].sum()
        for _, accts in IS_ACCOUNTS["EXPENSES"].items():
            exp += sel[sel["vAccount"].isin(accts)]["mAmount"].sum()
        exp += sel[sel["vAccount"].isin(IS_ACCOUNTS["TAX_ABATEMENT"])]["mAmount"].sum()
        months = [pd.Timestamp(x).strftime("%b") for x in sorted(sel["dtEntry_parsed"].unique())]
        return months, rev - exp

    d26 = df_q3[df_q3["dtEntry_parsed"].dt.year == 2026]
    cand = (d26[d26["vSource"] == "Budget IS"].groupby("vcode").size()
            .sort_values(ascending=False).index[:5])
    expected_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]

    print()
    for v in cand:
        perf_new = get_property_performance(v, new_default, df_q3, pd.DataFrame())
        perf_old = get_property_performance(v, old_default, df_q3, pd.DataFrame())
        months, noi = budget_months(v, new_default)
        months_old, _ = budget_months(v, old_default)

        print(f"  {v}")
        print(f"    served budget months .... {months}")
        print(f"    (old bug summed) ........ {months_old}")
        b_new = perf_new["noi"]["ytd_budget"]
        b_old = perf_old["noi"]["ytd_budget"]
        delta = (b_old / b_new - 1) * 100 if b_new else float("nan")
        print(f"    NOI ytd_budget .......... {b_new:,.0f}   (old served {b_old:,.0f}, "
              f"+{delta:.1f}%)")
        check(f"{v}: replication matches real ytd_budget", abs(noi - b_new) < 1.0)
        check(f"{v}: budget window is Jan-Jun for {new_default}", months == expected_months,
              "" if months == expected_months else f"got {months}")

print()
print("=" * 78)
if failures:
    print(f"FAILED ({len(failures)}):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("ALL CHECKS PASSED")
