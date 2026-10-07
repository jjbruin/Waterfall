"""Guardrail: U/W ROE to Date does not depend on the accounting feed, and it
never prints a computed-looking zero where there is no underwritten capital.

Background
----------
`get_pe_performance` computed the U/W ROE from ISBS Projected IS (accounts 7071
and 7073) only — "No actual accounting data used" — but the block sat inside
`if capital_events:`, itself inside `if not deal_acct.empty:`, itself inside
`if acct is not None ...`. A deal with underwritten rows and no ACTUAL capital
events (a twin deal whose accounting is filed under the other InvestmentID, a
child with no accounting of its own) silently lost a number that never needed
them. Moving it out exposes the second defect it was hiding: with no
underwritten contribution `calculate_roe_detailed` answers 0.0 (capital base 0),
and Investment Metrics reads the components as "the engine ran", so a deal with
no UW contribution printed a real-looking 0.0%. The function now leaves the
components unset in that case (-> a dash), and the scalar keeps its initial 0.0,
which the One Pager already prints as a dash.

Assert in BOTH directions: a real figure must still appear (gate gone), a real
ZERO (contribution, no distributions) must still be a zero, and the no-capital
cases must NOT produce a figure.

Synthetic frames only: no database, no network.

    python scripts/uw_roe_ungated_check.py
"""
from __future__ import annotations

import os
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loaders import normalize_accounting_feed, load_waterfalls  # noqa: E402
from one_pager import get_pe_performance  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

VCODE = "P0009999"
IID = "TESTIID"
Q = "2026-Q2"
failures: list = []


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  - ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def isbs(contrib=1_000_000.0, dists=(("2025-03-31", 50_000.0), ("2025-06-30", 50_000.0)), roc=None, vcode=VCODE):
    rows = []
    if contrib is not None:
        rows.append(("7073", "2024-01-31", contrib))
    if roc is not None:
        rows.append(("7073", "2024-12-31", -roc))
    for d, a in dists:
        rows.append(("7071", d, a))
    return pd.DataFrame([{
        "vcode": vcode.lower(), "vSource": "Projected IS", "vAccount": acc,
        "dtEntry_parsed": pd.Timestamp(d), "mAmount": amt, "_is_supplement": True,
    } for acc, d, amt in rows])


DEALS = pd.DataFrame([{"vcode": VCODE, "InvestmentID": IID, "Investment_Name": "UW Gate Test",
                       "Portfolio_Name": None, "Property_Count": "1"}])
WF = load_waterfalls(pd.DataFrame([{
    "vcode": VCODE, "vmisc": "CF_WF", "iOrder": 1, "PropCode": "PPI99", "vState": "Pref",
    "FXRate": 1.0, "nPercent": 0.10, "mAmount": None, "vtranstype": None, "vAmtType": None,
    "vNotes": "Pref", "dteffective": "2024-01-01"}]))


def acct_frame(rows):
    return normalize_accounting_feed(pd.DataFrame(
        [{"InvestmentID": IID, "InvestorID": b, "EffectiveDate": c, "MajorType": d, "Typename": e,
          "Amt": f, "TypeID": g, "Capital": "N", "Partner": ""} for b, c, d, e, f, g in rows]))


# the three accounting states the old nesting depended on
ACCT_NONE = None                                                    # no feed at all
ACCT_OP_ONLY = acct_frame([("OPTEST", "2024-01-01", "Contribution", "Contribution: Investments", -500_000.0, 1018)])
ACCT_REAL = acct_frame([
    ("PPI99", "2024-01-01", "Contribution", "Contribution: Investments", -1_000_000.0, 1018),
    ("PPI99", "2025-06-30", "Distribution", "Distribution: Preferred Return", 50_000.0, 1019),
    ("OPTEST", "2024-01-01", "Contribution", "Contribution: Investments", -500_000.0, 1018)])


def run(acct, frame):
    return get_pe_performance(VCODE, Q, acct, WF, DEALS, isbs_raw=frame)


print(__doc__.strip().split("\n")[0])
print()
print("1. a real U/W ROE appears whatever the accounting looks like (the gate is gone)")
ref = run(ACCT_REAL, isbs())
check("reference: with real accounting the figure is computed",
      bool(ref["uw_roe_components"]) and ref["uw_roe_to_date"] > 0, f"{ref['uw_roe_to_date']:.5f}")
check("reference: actual ROE components exist (gate open)", bool(ref["roe_components"]))
for name, acct in (("no accounting feed", ACCT_NONE), ("accounting with no PE capital events", ACCT_OP_ONLY)):
    pe = run(acct, isbs())
    check(f"{name}: the U/W figure is computed", bool(pe["uw_roe_components"]) and pe["uw_roe_to_date"] > 0)
    check(f"{name}: and equals the one with real accounting",
          abs(pe["uw_roe_to_date"] - ref["uw_roe_to_date"]) < 1e-12,
          f"{pe['uw_roe_to_date']:.5f} vs {ref['uw_roe_to_date']:.5f}")
    check(f"{name}: the ACTUAL ROE is still not invented", not pe["roe_components"] and pe["roe_to_date"] == 0.0)

print("\n2. a real ZERO is still a zero (underwritten capital, no distributions)")
for name, acct in (("no accounting feed", ACCT_NONE), ("real accounting", ACCT_REAL)):
    pe = run(acct, isbs(dists=()))
    check(f"{name}: components present", bool(pe["uw_roe_components"]))
    check(f"{name}: figure is 0.0", pe["uw_roe_to_date"] == 0.0)

print("\n3. no underwritten capital -> nothing computed (dash), never a computed-looking 0.0")
cases = {
    "distributions only, no 7073": isbs(contrib=None),
    "return of capital only (no contribution)": isbs(contrib=None, roc=250_000.0),
    "no Projected IS rows for the deal": isbs(vcode="P0000001"),
}
for cname, frame in cases.items():
    for aname, acct in (("no feed", ACCT_NONE), ("real accounting", ACCT_REAL)):
        pe = run(acct, frame)
        check(f"{cname} / {aname}: components unset", pe["uw_roe_components"] is None)
        check(f"{cname} / {aname}: scalar keeps the initial 0.0 (prints as a dash)", pe["uw_roe_to_date"] == 0.0)
pe = run(ACCT_REAL, None)
check("no ISBS frame at all: components unset", pe["uw_roe_components"] is None)

print("\n4. structure: the call is not nested under any accounting gate")
src = Path(__file__).resolve().parent.parent.joinpath("one_pager.py").read_text(encoding="utf-8").replace("\r\n", "\n")
m = re.search(r"\n( *)_compute_uw_roe\(pe, isbs_raw, vcode, quarter_end\)", src)
check("get_pe_performance calls _compute_uw_roe", bool(m))
check("the call sits at function level (4 spaces), outside every if/try", bool(m) and len(m.group(1)) == 4)
check("the old nested U/W block is gone", "_uw_detail = calculate_roe_detailed" not in src)

print()
if failures:
    print(f"FAILED: {len(failures)} check(s)")
    sys.exit(1)
print("All checks passed.")
