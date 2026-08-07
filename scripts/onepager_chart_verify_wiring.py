"""Read-only verification for feat/onepager-chart-window. Changes nothing.

PART A — quarter selection wiring. Calls the real /one-pager/chart view body
inside a Flask request context with ?quarter=... on the URL, so the proof runs
through request.args -> get_one_pager_chart -> _quarter_window rather than
around them. Shows the window end moving with the requested quarter.

PART B — what actually drives a zero. Prints every window quarter against the
deal's Date Closed (One Pager's date_closed = inv.Acquisition_Date, which
data_service derives from min(EffectiveDate) in the accounting feed) and flags
any pre-close quarter that is NOT zero.

DATA SOURCE: local CSV snapshots (ISBS Apr 15), NOT live Azure.
"""
import json
import os
import sys

import pandas as pd
from flask import Flask

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask_app.api import financials as fin_api
from flask_app.services.financials_service import _quarter_window, _resolve_date_closed

CSV_DIR = (r"C:\Users\cbui\OneDrive - peaceablestreet.com\Peaceable Street Capital - Documents"
           r"\Asset Mgmt\8. Asset_Mgmt_System\csv_data")
OCC_CSV = os.path.join(CSV_DIR, "MRI_Occupancy_Download.csv")
ACCT_CSV = os.path.join(CSV_DIR, "accounting_feed.csv")
MAP_CSV = os.path.join(CSV_DIR, "investment_map.csv")

SCRATCH = (r"C:\Users\cbui\AppData\Local\Temp\1\claude"
           r"\C--Users-cbui-Documents-waterfall-xirr"
           r"\fea2fef0-f2bb-45a6-a127-5d60708f31f0\scratchpad")
CACHE = os.path.join(SCRATCH, "isbs_slice.pkl")

DEALS = ["p0000082", "p0000084", "p0000099", "p0000109", "p0000113", "p0000085"]
QUARTERS = ["2026-Q1", "2026-Q2"]


def load_inv():
    """investment_map with Acquisition_Date enriched from the accounting feed —
    mirrors data_service._enrich_acquisition_dates."""
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


def quarter_of(ts):
    return f"Q{(ts.month - 1) // 3 + 1} {ts.year}"


def make_app(isbs, occ, inv):
    """Bare app + the real view function, with only the data source stubbed."""
    app = Flask(__name__)
    fin_api._get_data = lambda: {"isbs_raw": isbs, "occupancy_raw": occ, "inv": inv}
    return app


def call_endpoint(app, vcode, quarter):
    """Invoke the real /one-pager/chart view body with ?quarter=... on the URL."""
    url = f"/api/financials/{vcode}/one-pager/chart"
    if quarter is not None:
        url += f"?quarter={quarter}"
    with app.test_request_context(url):
        # __wrapped__ skips only @login_required; the view body is the real one.
        resp = fin_api.one_pager_chart.__wrapped__(vcode)
    return json.loads(resp.get_data(as_text=True))


def main():
    isbs = pd.read_pickle(CACHE)
    occ = pd.read_csv(OCC_CSV, dtype=str)
    inv = load_inv()
    app = make_app(isbs, occ, inv)
    closed = {vc: _resolve_date_closed(inv, vc) for vc in DEALS}

    print("#" * 78)
    print("# PART A — does the requested quarter drive the window end?")
    print("#" * 78)
    print("Through the real view: request.args['quarter'] -> get_one_pager_chart\n")
    print(f"{'vcode':<10} {'?quarter=':<12} {'n':>3}  {'first':<9} {'last':<9}  ends at requested?")
    print("-" * 78)
    ok = True
    for vcode in DEALS:
        for q in QUARTERS:
            c = call_endpoint(app, vcode, q)
            want = quarter_of(_quarter_window(q, 1)[0])
            hit = c["periods"][-1] == want
            ok &= hit
            print(f"{vcode:<10} {q:<12} {len(c['periods']):>3}  "
                  f"{c['periods'][0]:<9} {c['periods'][-1]:<9}  "
                  f"{'YES' if hit else 'NO  <-- FAIL'}")
        # control: no quarter param at all -> falls back to the data-derived window
        c = call_endpoint(app, vcode, None)
        last = c["periods"][-1] if c["periods"] else "(empty)"
        print(f"{vcode:<10} {'(omitted)':<12} {len(c['periods']):>3}  "
              f"{(c['periods'][0] if c['periods'] else '-'):<9} {last:<9}  "
              f"fallback: data-derived")
        print()

    print(f"PART A: {'PASS — the requested quarter sets the end.' if ok else 'FAIL'}\n")

    print("#" * 78)
    print("# PART B — what makes a quarter show 0?")
    print("#" * 78)
    for vcode in DEALS:
        dc = closed.get(vcode)
        dc_q = quarter_of(dc) if dc is not None else "unknown"
        print("=" * 78)
        print(f"{vcode}   Date Closed = "
              f"{dc.date() if dc is not None else '?'}  ({dc_q})")
        print("=" * 78)
        c = call_endpoint(app, vcode, "2026-Q2")
        window = _quarter_window("2026-Q2", 10)
        violations = []
        for i, dt in enumerate(window):
            pre = dc is not None and dt < pd.Timestamp(dc)
            a, u, o = c["actual_noi"][i], c["uw_noi"][i], c["occupancy"][i]

            def s(v):
                return " null " if v is None else f"{v / 1e6:6.2f}"
            nonzero = pre and any(v not in (0, None) for v in (a, u, o))
            if nonzero:
                violations.append((c["periods"][i], a, u, o))
            print(f"   {c['periods'][i]:<9} ACT {s(a)}  UW {s(u)}  "
                  f"OCC {'null' if o is None else f'{o:5.1f}'}   "
                  f"{'PRE-CLOSE' if pre else ''}"
                  f"{'  <-- NOT ZERO' if nonzero else ''}")
        if violations:
            print(f"   >>> {len(violations)} pre-close quarter(s) carry non-zero data")
        else:
            print("   >>> all pre-close quarters are zero")
        print()


if __name__ == "__main__":
    main()
