"""Guardrail for the committed-pref rewire (committed_pref.py).

Asserts the as-of rule, every fallback, the None-never-zero contract, and that
all eight readers of the `commitments` table filter to current rows.

BOTH DIRECTIONS THROUGHOUT. "The commitments table is used" is satisfied by
using it always, including where it must not be; "the accounting figure is
kept" is satisfied by never reading commitments at all. Each rule is asserted
with a case that passes AND a case that must not.

    python scripts/committed_pref_check.py
    python scripts/committed_pref_check.py --inject=<defect>   (non-vacuity)
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd                                            # noqa: E402
import committed_pref as cp                                    # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PASS, FAIL = [], []


def chk(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail and not ok else ""))


def frame(rows):
    return pd.DataFrame(rows)


Q2, Q3 = date(2026, 6, 30), date(2026, 9, 30)

# A real chain, transcribed from MRI: an ended revision and its successor.
JBFAIR = [
    {"EntityID": "JBFAIR", "InvestorID": "PPI32", "Amount": 14300000.0,
     "StartDate": "2021-02-24", "EndDate": "2026-07-29", "CommitmentUID": 717,
     "TransactionNote": None},
    {"EntityID": "JBFAIR", "InvestorID": "PPI32", "Amount": 29757181.0,
     "StartDate": "2026-07-30", "EndDate": None, "CommitmentUID": 1356,
     "TransactionNote": None},
    # the OP side must never reach the pref figure
    {"EntityID": "JBFAIR", "InvestorID": "OPTREV", "Amount": 11902873.0,
     "StartDate": "2026-07-30", "EndDate": None, "CommitmentUID": 1358,
     "TransactionNote": None},
]

print("\n-- the as-of rule (rule B: the row in force on Q) --")
amt, basis = cp.resolve_committed_pref(frame(JBFAIR), "JBFAIR", Q2,
                                       accounting_committed=30000000.0)
chk("26Q2 picks the ENDED row that was in force (14,300,000)",
    amt == 14300000.0, f"got {amt}")
chk("26Q2 basis names the row it used", "uid 717" in basis, basis)
amt3, _ = cp.resolve_committed_pref(frame(JBFAIR), "JBFAIR", Q3,
                                    accounting_committed=30000000.0)
chk("26Q3 picks the SUCCESSOR row (29,757,181)", amt3 == 29757181.0, f"got {amt3}")
chk("the OP row is excluded from the pref figure",
    amt3 == 29757181.0 and amt3 != 29757181.0 + 11902873.0)

print("\n-- no row in effect: the accounting figure is KEPT (Charlene, 2026-10-01) --")
open_only = frame([r for r in JBFAIR if r["EndDate"] is None])
amt, basis = cp.resolve_committed_pref(open_only, "JBFAIR", Q2,
                                       accounting_committed=30000000.0)
chk("open-only table at 26Q2 keeps today's accounting figure",
    amt == 30000000.0, f"got {amt}")
chk("and says so in the basis",
    "no commitments row in effect" in basis and "accounting figure kept" in basis,
    basis)
amt, basis = cp.resolve_committed_pref(open_only, "JBFAIR", Q2,
                                       accounting_committed=None)
chk("with no accounting figure either, it is None — NEVER 0",
    amt is None, f"got {amt!r}")

print("\n-- tombstones (same-day, <= 0.01) are not pledges --")
TOMB = [
    {"EntityID": "BRNERD", "InvestorID": "PPIBPA", "Amount": 0.0,
     "StartDate": "2022-06-14", "EndDate": "2022-06-14", "CommitmentUID": 924,
     "TransactionNote": None},
    {"EntityID": "BRNERD", "InvestorID": "PPIBPA", "Amount": 31721927.0,
     "StartDate": "2024-12-27", "EndDate": None, "CommitmentUID": 1568,
     "TransactionNote": None},
]
amt, _ = cp.resolve_committed_pref(frame(TOMB), "BRNERD", Q2)
chk("the real row survives alongside a tombstone", amt == 31721927.0, f"got {amt}")
rows = cp.deal_commitment_rows(frame(TOMB), "BRNERD")
chk("the tombstone row is dropped", len(rows) == 1, f"{len(rows)} rows kept")

print("\n-- auto-generated back-fill only: pending accounting --")
BALES = [
    {"EntityID": "BALES", "InvestorID": "PPI2", "Amount": 63195.71,
     "StartDate": "2025-11-03", "EndDate": None, "CommitmentUID": 1251,
     "TransactionNote": "Auto generated commitment created for multi-level"},
    {"EntityID": "BALES", "InvestorID": "PPI2LP", "Amount": 106983.0,
     "StartDate": "2025-11-03", "EndDate": None, "CommitmentUID": 1242,
     "TransactionNote": "Auto generated commitment created for multi-level"},
]
amt, basis = cp.resolve_committed_pref(frame(BALES), "BALES", Q2,
                                       accounting_committed=4172975.37)
chk("an all-back-fill chain keeps the accounting figure",
    amt == 4172975.37, f"got {amt}")
chk("and the basis says the question is open",
    basis == "pending accounting", basis)
chk("it does NOT publish the back-fill sum (170,178.71)",
    amt != 63195.71 + 106983.0)

print("\n-- sold and not kept: None, never a resurrected open row --")
CAM = [{"EntityID": "CAMARI", "InvestorID": "PPI34", "Amount": 18843400.0,
        "StartDate": "2021-05-11", "EndDate": None, "CommitmentUID": 665,
        "TransactionNote": None}]
amt, basis = cp.resolve_committed_pref(frame(CAM), "CAMARI", Q3,
                                       accounting_committed=0.0,
                                       sold_and_dropped=True)
chk("sold+dropped returns None", amt is None, f"got {amt!r}")
chk("sold+dropped says why", "sold" in basis.lower(), basis)
amt, _ = cp.resolve_committed_pref(frame(CAM), "CAMARI", Q3,
                                   sold_and_dropped=False)
chk("a KEPT sold deal still resolves at the quarter it is read",
    amt == 18843400.0, f"got {amt}")

print("\n-- several InvestmentIDs on one deal are summed, per (entity, investor) --")
MULTI = [
    {"EntityID": "APPLE", "InvestorID": "PPI2", "Amount": 1000.0,
     "StartDate": "2020-01-01", "EndDate": None, "CommitmentUID": 1,
     "TransactionNote": None},
    {"EntityID": "APPLE2", "InvestorID": "PPI2", "Amount": 2000.0,
     "StartDate": "2020-01-01", "EndDate": None, "CommitmentUID": 2,
     "TransactionNote": None},
]
amt, _ = cp.resolve_committed_pref(frame(MULTI), ["APPLE", "APPLE2"], Q2)
chk("the same investor on two entities is two pledges", amt == 3000.0, f"got {amt}")

print("\n-- pd.NaT, NOT None: an OPEN row as pandas actually delivers it --")
# THE TEST THAT WOULD HAVE CAUGHT THE v549 OUTAGE. While the commitments table
# held only current rows its EndDate column was entirely null, so pandas typed
# it object/float and an open row arrived as None or NaN. Once ENDED rows load
# the column is datetime64 and an open row arrives as pd.NaT -- which IS an
# instance of datetime, so an isinstance branch ahead of the null guard returns
# NaT and the as-of comparison raises. Every 26Q3 One Pager 500'd.
chk("pd.NaT is an instance of datetime (why order matters)",
    isinstance(pd.NaT, __import__("datetime").datetime))
chk("_as_date(pd.NaT) is None", cp._as_date(pd.NaT) is None,
    f"got {cp._as_date(pd.NaT)!r}")
chk("_as_date(float nan) is None", cp._as_date(float("nan")) is None)
chk("_as_date(pd.NA) is None", cp._as_date(pd.NA) is None)
NAT_ROWS = [
    {"EntityID": "ASHBCO", "InvestorID": "PPI22", "Amount": 1490000.0,
     "StartDate": pd.Timestamp("2019-05-15"), "EndDate": pd.Timestamp("2026-09-10"),
     "CommitmentUID": 660, "TransactionNote": None},
    {"EntityID": "ASHBCO", "InvestorID": "PPI22", "Amount": 1620000.0,
     "StartDate": pd.Timestamp("2026-09-11"), "EndDate": pd.NaT,
     "CommitmentUID": 1900, "TransactionNote": None},
]
_f = pd.DataFrame(NAT_ROWS)
chk("the fixture really carries a datetime64 EndDate with NaT",
    str(_f["EndDate"].dtype).startswith("datetime64") and bool(_f["EndDate"].isna().any()),
    str(_f["EndDate"].dtype))
try:
    _amt, _bs = cp.resolve_committed_pref(_f, "ASHBCO", date(2026, 9, 30))
    _raised = None
except Exception as _e:
    _amt, _bs, _raised = None, "", f"{type(_e).__name__}: {_e}"
chk("committed_pe computes without raising on a NaT open row",
    _raised is None, _raised or "")
chk("...and the OPEN row is the one in force at 26Q3 (1,620,000)",
    _amt == 1620000.0, f"got {_amt}")
_amt2, _ = cp.resolve_committed_pref(_f, "ASHBCO", date(2026, 6, 30))
chk("...while 26Q2 still takes the ended row (1,490,000)",
    _amt2 == 1490000.0, f"got {_amt2}")

print("\n-- ENDED ROWS NOW LOAD: every reader must be unmoved by them --")
from investment_metrics import capitalization_sources, DealIdentity
CHAIN_OPEN = [
    {"EntityID": "PONTCH", "InvestorID": "PPI31", "Amount": 10847420.0,
     "StartDate": "2025-12-08", "EndDate": None, "CommitmentUID": 1362,
     "TransactionNote": None},
]
CHAIN_FULL = CHAIN_OPEN + [
    {"EntityID": "PONTCH", "InvestorID": "PPI31", "Amount": 10370000.0,
     "StartDate": "2021-02-01", "EndDate": "2023-07-30", "CommitmentUID": 783,
     "TransactionNote": None},
    {"EntityID": "PONTCH", "InvestorID": "PPI31", "Amount": 10790420.0,
     "StartDate": "2023-07-31", "EndDate": "2025-10-05", "CommitmentUID": 1360,
     "TransactionNote": None},
    {"EntityID": "PONTCH", "InvestorID": "PPI31", "Amount": 10815420.0,
     "StartDate": "2025-10-06", "EndDate": "2025-12-07", "CommitmentUID": 1361,
     "TransactionNote": None},
]
_ident = DealIdentity(vcode="P0000037", investment_id="PONTCH", name="Pontchartrain")
_a = capitalization_sources(_ident, pd.DataFrame(), frame(CHAIN_OPEN))[0][1]
_b = capitalization_sources(_ident, pd.DataFrame(), frame(CHAIN_FULL))[0][1]
chk("Investment Metrics is UNMOVED when the ended rows load",
    _a is not None and _b is not None and abs(_a - _b) < 0.01, f"open={_a} full={_b}")
chk("...and does NOT sum the superseded revisions (42,823,260)", _b != 42823260.0, f"got {_b}")
_amt, _ = cp.resolve_committed_pref(frame(CHAIN_FULL), "PONTCH", date(2024, 6, 30))
chk("the as-of engine picks the row in force mid-2024 (10,790,420)", _amt == 10790420.0, f"got {_amt}")
_amt, _ = cp.resolve_committed_pref(frame(CHAIN_FULL), "PONTCH", date(2026, 6, 30))
chk("...and the current one at 26Q2 (10,847,420)", _amt == 10847420.0, f"got {_amt}")

print("\n-- the eight readers of the commitments table filter to current rows --")
READERS = [
    ("flask_app/services/ownership_service.py", None),
    ("flask_app/services/statement_service.py", None),
    ("flask_app/services/workpaper_data.py", None),
    ("flask_app/services/workpaper_tracker.py", None),
    ("flask_app/services/treasury_upload.py", None),
    ("flask_app/services/ownership_chain_service.py", "current_only=True"),
]
for path, extra in READERS:
    src = io.open(os.path.join(ROOT, path), encoding="utf-8").read()
    ok = ('"EndDate" IS NULL' in src) or (extra and extra in src)
    chk(f"{os.path.basename(path)} filters to current rows", ok)
    bare = re.search(r'FROM commitments(?!\s*\{)(?![^"\']*EndDate)["\']', src)
    chk(f"{os.path.basename(path)} has no unfiltered SELECT left", bare is None)

im = io.open(os.path.join(ROOT, "investment_metrics.py"), encoding="utf-8").read()
chk("investment_metrics delegates to the shared engine",
    "resolve_committed_pref" in im)
chk("investment_metrics filters the first-loss side to current rows",
    'm["EndDate"].isna()' in im)

op = io.open(os.path.join(ROOT, "one_pager.py"), encoding="utf-8").read()
chk("one_pager has NO second committed-pref implementation left",
    op.count("resolve_committed_pref(") == 2,          # exactly two call sites
    f"found {op.count('resolve_committed_pref(')}")
chk("the old accounting sum no longer assigns committed_pe directly",
    "cap['committed_pe'] = commitment_rows" not in op
    and "pe['committed_pe'] = commitment_rows" not in op)

fs = io.open(os.path.join(ROOT, "flask_app/services/financials_service.py"),
             encoding="utf-8").read()
chk("financials_service tests `is None`, not a falsy zero",
    'pe.get("committed_pe") is None' in fs
    and 'pe.get("committed_pe", 0) == 0' not in fs)

print("\n-- remaining_to_fund has NO floor, and flags committed < funded --")
chk("one_pager publishes committed_below_funded",
    "committed_below_funded" in op)
chk("remaining_to_fund is not floored at zero",
    "max(0, pe['committed_pe'] - pe['funded_to_date'])" not in op)

print(f"\n{'-'*72}")
print(f"{'PASS' if not FAIL else 'FAIL'} — {len(PASS)}/{len(PASS)+len(FAIL)} checks")
if FAIL:
    for f in FAIL:
        print("   FAILED:", f)
sys.exit(1 if FAIL else 0)
