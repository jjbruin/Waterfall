#!/usr/bin/env python
"""Guardrail: the One Pager picks the right valuation row.

Two rules, both proven by CALLING the real committed `get_capitalization_stack`, not by
reading the source.

  1. A BLANK COLUMN MUST NOT DISCARD A GOOD VALUATION. The lookup sorts by date and used
     to take `iloc[0]` unconditionally, then read `mIncomeCapConcludedValue` off it. A
     newer row with that column empty produced 0.0 — the page showed NO valuation — while
     a complete older row sat directly behind it. This is the live shape of the sentinel
     problem CLAUDE.md warns about: a 0 meaning "no data" is indistinguishable from a real
     zero, and `pe_exposure_on_value` then silently does not compute either.

  2. THE VALUATION IS AS OF THE REPORT QUARTER. Debt is fetched with `as_of_date=q_end`
     and the equity block filters `EffectiveDate <= q_end`, but the valuation ignored the
     quarter and always took the newest row on file. Once a 12/31/2025 valuation is
     published, re-opening 25Q2 showed that FUTURE valuation against 25Q2's debt and
     equity — P.E. Exposure on Value became a ratio between two different dates.

Synthetic valuation frames are used so both conditions exist deterministically; every
other input is the real loaded data. Run:  python scripts/onepager_valuation_selection_check.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from one_pager import get_capitalization_stack  # noqa: E402

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
    """rows = [(date_str, concluded_value_or_None), ...]"""
    recs = []
    for dt, v in rows:
        r = {c: None for c in COLS}
        r["vCode"] = VCODE
        r["vPropertyName"] = "Town Fair Tire Portfolio"
        r["dtValuation"] = dt
        r["mIncomeCapConcludedValue"] = v
        recs.append(r)
    return pd.DataFrame(recs, columns=COLS)


def cap_for(val_df, quarter=None):
    empty = pd.DataFrame()
    return get_capitalization_stack(
        VCODE, mri_loans=empty, mri_val=val_df, waterfalls=empty,
        acct=empty, inv_map=empty, isbs_raw=None, quarter_str=quarter,
    )


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("1. A blank on the newest row must not discard a good older valuation")
    # Newest row (2025) has NO concluded value; the 2024 row does. Pre-fix this returned
    # 0.0 and the page showed nothing.
    df = val_frame([("12/31/2025 0:00", None), ("12/31/2024 0:00", 30_750_000.0)])
    c = cap_for(df)
    chk("falls back to the most recent row that carries a value",
        c["current_valuation"] == 30_750_000.0,
        f"got {c['current_valuation']!r}, expected 30750000.0")

    # And the complete newer row must still win when it has one.
    df = val_frame([("12/31/2025 0:00", 33_910_000.0), ("12/31/2024 0:00", 30_750_000.0)])
    c = cap_for(df)
    chk("a complete newer row still wins",
        c["current_valuation"] == 33_910_000.0,
        f"got {c['current_valuation']!r}, expected 33910000.0")

    # A zero is not a value — it is the same "no data" sentinel by another route.
    df = val_frame([("12/31/2025 0:00", 0.0), ("12/31/2024 0:00", 30_750_000.0)])
    c = cap_for(df)
    chk("a newest row of 0 does not mask a real older valuation",
        c["current_valuation"] == 30_750_000.0,
        f"got {c['current_valuation']!r}")

    # Nothing usable anywhere: still no crash, and no invented number.
    df = val_frame([("12/31/2025 0:00", None)])
    c = cap_for(df)
    chk("no usable row anywhere leaves the default, and does not raise",
        c["current_valuation"] in (0.0, None), f"got {c['current_valuation']!r}")

    print()
    print("2. The valuation is as of the report quarter")
    df = val_frame([("12/31/2025 0:00", 33_910_000.0), ("12/31/2024 0:00", 30_750_000.0)])

    c = cap_for(df, quarter="2026-Q1")
    chk("26Q1 sees the 12/31/2025 valuation",
        c["current_valuation"] == 33_910_000.0, f"got {c['current_valuation']!r}")

    c = cap_for(df, quarter="2025-Q2")
    chk("25Q2 does NOT see a valuation dated after the quarter it reports",
        c["current_valuation"] == 30_750_000.0,
        f"got {c['current_valuation']!r}, expected the 12/31/2024 figure 30750000.0")

    c = cap_for(df, quarter="2025-Q4")
    chk("25Q4 sees the 12/31/2025 valuation (boundary is inclusive)",
        c["current_valuation"] == 33_910_000.0, f"got {c['current_valuation']!r}")

    c = cap_for(df, quarter="2024-Q1")
    chk("a quarter before every valuation shows none, not the newest one",
        c["current_valuation"] in (0.0, None), f"got {c['current_valuation']!r}")

    c = cap_for(df)  # no quarter — other callers must be unaffected
    chk("with no quarter the behaviour is unchanged (newest wins)",
        c["current_valuation"] == 33_910_000.0, f"got {c['current_valuation']!r}")

    print()
    print("3. valuation_year follows the row actually used")
    df = val_frame([("12/31/2025 0:00", 33_910_000.0), ("12/31/2024 0:00", 30_750_000.0)])
    c = cap_for(df, quarter="2025-Q2")
    chk("25Q2 labels the year of the row it displayed, not the newest on file",
        c.get("valuation_year") == "2024", f"got {c.get('valuation_year')!r}")

    print(f"\nPASS={PASS} FAIL={FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
