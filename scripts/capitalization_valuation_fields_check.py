#!/usr/bin/env python
"""Guardrail: a blank valuation column does not become a zero in the cap stack.

`get_deal_capitalization()` reads three fields off the valuations table —
`mIncomeCapConcludedValue`, `fCapRate`, `nCostSaleRate`. It used to take all three off
the newest row and fall back to 0.0 on a blank, so one partial row zeroed the other two.

THIS IS NOT COSMETIC. The Dashboard's weighted-average cap rate is
`sum(cap_rate x valuation) / sum(valuation)`. A deal whose newest row carries a valuation
but no cap rate puts its FULL valuation into the denominator contributing nothing to the
numerator. Measured on live data, one such row on a $33.9M deal moves the portfolio KPI
by -7.4 bps — a reported figure, moved by a data gap, silently.

Each field now falls back independently to the most recent row that carries it. Fields may
therefore come from different valuation dates; that is deliberate and the trade is stated
in the code — a stale-but-real cap rate beats a fabricated zero.

Run:  python scripts/capitalization_valuation_fields_check.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compute import get_deal_capitalization  # noqa: E402

PASS = FAIL = 0


def chk(label, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}" + (f"\n          {detail}" if detail else ""))


VCODE = "P0000107"
COLS = ["vCode", "vPropertyName", "dtValuation", "vMethod", "mAnnualNOI", "fCapRate",
        "nTermCapRate", "nDiscountRateForEquityInterest", "mIncomeCapConcludedValue",
        "mDebtValue", "mEquityValue", "mMezzanineValue", "nCostSaleRate"]


def val_frame(rows):
    """rows = [(date, concluded_value, cap_rate, cost_of_sale), ...] — None means blank."""
    recs = []
    for dt, v, cr, cos in rows:
        r = {c: None for c in COLS}
        r.update({"vCode": VCODE, "vPropertyName": "Town Fair Tire Portfolio",
                  "dtValuation": dt, "mIncomeCapConcludedValue": v,
                  "fCapRate": cr, "nCostSaleRate": cos})
        recs.append(r)
    return pd.DataFrame(recs, columns=COLS)


def cap_for(val_df):
    e = pd.DataFrame()
    # `inv` must carry InvestmentID/vcode or the function bails before the valuation
    # block — and it bails through a try/except that returns a partial dict, so a bad
    # fixture reads as a failing assertion rather than an error. Keep this minimal row.
    inv = pd.DataFrame([{"vcode": VCODE, "InvestmentID": "TFT",
                         "Investment_Name": "Town Fair Tire Portfolio"}])
    # get_deal_capitalization(acct, inv, wf, mri_val, mri_loans, deal_vcode, ...)
    c = get_deal_capitalization(e, inv, e, val_df, e, VCODE, isbs_raw=None)
    # The three keys under test must exist; a missing one means the block never ran.
    for k in ("current_valuation", "cap_rate", "cost_of_sale"):
        c.setdefault(k, "<never set — valuation block did not run>")
    return c


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("1. The Town Fair case: a newer row with ONLY a concluded value")
    df = val_frame([
        ("12/31/2025 0:00", 33_910_000.0, None, None),      # the partial row
        ("12/31/2024 0:00", 30_750_000.0, 0.0738, 0.04),    # the complete one behind it
    ])
    c = cap_for(df)
    chk("the new valuation is taken", c["current_valuation"] == 33_910_000.0,
        f"got {c['current_valuation']!r}")
    chk("cap_rate falls back to 7.38% instead of becoming 0",
        abs(c["cap_rate"] - 0.0738) < 1e-12, f"got {c['cap_rate']!r}")
    chk("cost_of_sale falls back to 4% instead of becoming 0",
        abs(c["cost_of_sale"] - 0.04) < 1e-12, f"got {c['cost_of_sale']!r}")

    print("\n2. A complete newest row still wins outright")
    df = val_frame([
        ("12/31/2025 0:00", 33_910_000.0, 0.0700, 0.03),
        ("12/31/2024 0:00", 30_750_000.0, 0.0738, 0.04),
    ])
    c = cap_for(df)
    chk("every field comes from the newest row",
        c["current_valuation"] == 33_910_000.0
        and abs(c["cap_rate"] - 0.0700) < 1e-12
        and abs(c["cost_of_sale"] - 0.03) < 1e-12,
        f"got {c['current_valuation']!r} / {c['cap_rate']!r} / {c['cost_of_sale']!r}")

    print("\n3. A literal 0 is the blank arriving as a number, for value and rate")
    df = val_frame([
        ("12/31/2025 0:00", 0.0, 0.0, 0.0),
        ("12/31/2024 0:00", 30_750_000.0, 0.0738, 0.04),
    ])
    c = cap_for(df)
    chk("a 0 valuation does not mask the real one",
        c["current_valuation"] == 30_750_000.0, f"got {c['current_valuation']!r}")
    chk("a 0 cap rate does not mask the real one",
        abs(c["cap_rate"] - 0.0738) < 1e-12, f"got {c['cap_rate']!r}")
    chk("a 0 cost-of-sale IS taken — it is expressible, unlike the other two",
        c["cost_of_sale"] == 0.0, f"got {c['cost_of_sale']!r}")

    print("\n4. Nothing usable anywhere leaves the defaults and does not raise")
    df = val_frame([("12/31/2025 0:00", None, None, None)])
    c = cap_for(df)
    chk("no crash, no invented figures",
        c["current_valuation"] == 0.0 and c["cap_rate"] == 0.0, f"got {c!r}"[:120])

    print("\n5. Comma-formatted source values still parse (MRI ships these)")
    df = val_frame([("12/31/2025 0:00", "33,910,000", None, None),
                    ("12/31/2024 0:00", "30,750,000", "0.0738", "0.04")])
    c = cap_for(df)
    chk("commas stripped, fallback still works",
        c["current_valuation"] == 33_910_000.0 and abs(c["cap_rate"] - 0.0738) < 1e-12,
        f"got {c['current_valuation']!r} / {c['cap_rate']!r}")

    print(f"\nPASS={PASS} FAIL={FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
