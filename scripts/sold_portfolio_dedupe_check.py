"""Guardrail: Reports > Sold Portfolio counts one investment ONCE.

Background
----------
A sold investment can sit on two SOLD vcodes that share an InvestmentID. Donald
Lynch is MCCORD on P0000049 (the property, Property_Count 1) and on its twin
P0000073 (Property_Count 0). `compute_all_sold_returns` matches accounting by
InvestmentID, so both rows pulled the same 117 accounting rows (every date and
amount identical), listed the investment twice, and "Portfolio Total" counted
its $2.45M of contributions and $5.18M of distributions twice: IRR 18.2540%
against 18.2254% with it once, which is the reference workbook's Sold Total.

`get_sold_deals` now keeps one row per InvestmentID, preferring the row that
carries the property.

Asserted in BOTH directions: the duplicate is dropped, and nothing else is -
rows with no InvestmentID are never collapsed into each other, and distinct
investments are never merged.

Synthetic frames only: no database, no network.

    python scripts/sold_portfolio_dedupe_check.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loaders import normalize_accounting_feed  # noqa: E402
from flask_app.services import sold_service as V  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

failures: list = []


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  - ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def deal(vcode, iid, name, status="SOLD", pc="1", port=None, sale="9/4/2026"):
    return dict(vcode=vcode, InvestmentID=iid, Investment_Name=name, Portfolio_Name=port if port is not None else name,
                Sale_Status=status, Property_Count=pc, Sale_Date=sale, Acquisition_Date="6/30/2021")


# the twin pair FIRST in the table, the property-bearing row SECOND: the choice must not depend on order
INV = pd.DataFrame([
    deal("P0000073", "MCX", "McX Centre", pc="0"),
    deal("P0000049", "MCX", "McX Centre", pc="1"),
    deal("PALPHA", "ALPHA", "Alpha Plaza"),
    deal("PNOID1", None, "No Id One"),
    deal("PNOID2", None, "No Id Two"),
    deal("PLIVE", "LIVE", "Live Deal", status=None, sale=None),
    deal("PKID", "KID", "Alpha Kid", port="Alpha Plaza"),          # a child of a sold parent (existing rule)
])

print(__doc__.strip().split("\n")[0])
print()
print("1. the twin pair becomes ONE row, and the property-bearing row is the one kept")
s = V.get_sold_deals(INV)
mcx = s[s["InvestmentID"] == "MCX"]
check("one row for MCX", len(mcx) == 1, f"got {len(mcx)}")
check("it is the row with the property (P0000049), whichever came first", list(mcx["vcode"]) == ["P0000049"], f"got {list(mcx['vcode'])}")

print("\n2. nothing else is dropped (both directions)")
check("a distinct sold investment is kept", "PALPHA" in set(s["vcode"]))
check("two rows with NO InvestmentID are NOT collapsed into each other", {"PNOID1", "PNOID2"} <= set(s["vcode"]),
      f"got {sorted(s['vcode'])}")
check("a live deal stays out, a child of a sold parent stays out (existing rules intact)",
      "PLIVE" not in set(s["vcode"]) and "PKID" not in set(s["vcode"]))
check("the original order is kept", list(s["vcode"]) == ["P0000049", "PALPHA", "PNOID1", "PNOID2"], f"got {list(s['vcode'])}")

print("\n3. the report's Portfolio Total counts the investment once")
rows = [
    ("MCX", "PPI4", "2021-06-30", 1018, -2_500_000, "Contribution", "Contribution: Investments"),
    ("MCX", "PPI4", "2026-09-04", 1016, 5_200_000, "Distribution", "Distribution: Return of Capital"),
    ("MCX", "OPMCX", "2021-06-30", 1018, -1_000_000, "Contribution", "Contribution: Investments"),
    ("ALPHA", "PPI1", "2019-03-01", 1018, -1_000_000, "Contribution", "Contribution: Investments"),
    ("ALPHA", "PPI1", "2022-03-01", 1016, 1_800_000, "Distribution", "Distribution: Return of Capital"),
]
acct = normalize_accounting_feed(pd.DataFrame(rows, columns=["InvestmentID", "InvestorID", "EffectiveDate", "SubtypeUID", "Amt", "MajorType", "Typename"]).assign(Capital="Y", Partner="x"))
V.clear_sold_cache()
res = V.compute_all_sold_returns(s[s["InvestmentID"].isin(["MCX", "ALPHA"])], acct, INV)
tot = res[res["_is_deal_total"]].iloc[0]
check("Portfolio Total contributions = MCX once + ALPHA = 3,500,000", abs(tot["Total Contributions"] - 3_500_000) < 1, f"got {tot['Total Contributions']:,.0f}")
check("Portfolio Total distributions = 5,200,000 + 1,800,000", abs(tot["Total Distributions"] - 7_000_000) < 1, f"got {tot['Total Distributions']:,.0f}")
check("the report has one MCX row", int((res["Investment Name"] == "McX Centre").sum()) == 1)

print()
if failures:
    print(f"FAILED: {len(failures)} check(s)")
    sys.exit(1)
print("All checks passed.")
