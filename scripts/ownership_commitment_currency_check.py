"""Guardrail: the ownership tree uses the CURRENT commitment, not the sum.

WHY THIS EXISTS. Commitments are an amendment history. When a commitment
changes, the old row is closed with an EndDate and a new row opens with a later
StartDate. The current commitment is therefore the amount on the most recent
StartDate that has no EndDate -- ONE ROW.

The first version of ownership_chain_service summed every row that had not yet
ended. That is wrong in both directions simultaneously: the amended investor is
counted once per amendment and inflated, and because every share is that
investor's amount over the level total, every OTHER owner at the level is
understated by the same distortion. Nothing looks broken, because the level
still sums to 100%.

The local database cannot produce this shape -- it holds three commitment rows,
all open, one per pair -- so the scenario is constructed here. That is the whole
point: the defect Jim found on production data was invisible to every check
that ran against the local fixture.

Run:  .venv/Scripts/python.exe scripts/ownership_commitment_currency_check.py
Fails against the commit before the fix.
"""
import sys
import pathlib

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from flask_app.services.ownership_chain_service import _Source, _owners_of  # noqa: E402

FAIL = []


def check(cond, msg):
    if not cond:
        FAIL.append(msg)


def source_from(rows):
    """A _Source carrying only these commitments, with no database."""
    src = _Source.__new__(_Source)
    com = pd.DataFrame(rows)
    for c in ("EntityID", "InvestorID"):
        com[c] = com[c].astype(str).str.strip().str.upper()
    com["Amount"] = pd.to_numeric(com["Amount"], errors="coerce").fillna(0.0)
    src.com, src.superseded_rows = _Source._current_only(com)
    src.names, src.deal_by_investment = {}, {}
    src.wf_codes, src.wf_step_counts = set(), {}
    # Every attribute _owners_of reads. A fixture that constructs _Source by
    # hand goes stale the moment the real __init__ grows a field -- which it
    # just did, and the guardrail failed with AttributeError rather than
    # testing anything.
    src.balances = {}
    src.load_errors = []
    src.raw_commitment_rows = len(com)
    src.commitment_columns = [str(c) for c in com.columns]
    return src


def by_id(owners):
    return {o["entity_id"]: o for o in owners}


# ── 1. An amended commitment counts ONCE, at its latest amount ───────────
# ALPHA committed 10M, then amended to 25M. The 10M row is closed. Summing
# would give 35M and hand ALPHA 77.8% of a 45M level instead of its true 50%.
src = source_from([
    {"EntityID": "DEAL1", "InvestorID": "ALPHA", "Amount": 10_000_000,
     "StartDate": "2023-01-01", "EndDate": "2024-06-30", "CapitalPercent": 0},
    {"EntityID": "DEAL1", "InvestorID": "ALPHA", "Amount": 25_000_000,
     "StartDate": "2024-07-01", "EndDate": None, "CapitalPercent": 0},
    {"EntityID": "DEAL1", "InvestorID": "BETA", "Amount": 25_000_000,
     "StartDate": "2023-01-01", "EndDate": None, "CapitalPercent": 0},
])
o = by_id(_owners_of(src, "DEAL1"))
check("ALPHA" in o, "ALPHA vanished from the level entirely")
if "ALPHA" in o:
    check(o["ALPHA"]["committed"] == 25_000_000,
          "ALPHA committed is %s, expected 25,000,000 -- the superseded 10,000,000 "
          "row is being counted" % f"{o['ALPHA']['committed']:,.0f}")
    check(abs(o["ALPHA"]["pct"] - 50.0) < 0.01,
          "ALPHA holds %.2f%%, expected 50.00%%" % o["ALPHA"]["pct"])
    check(o["ALPHA"]["since"] == "2024-07-01",
          "ALPHA effective date is %r, expected 2024-07-01" % o["ALPHA"]["since"])
if "BETA" in o:
    check(abs(o["BETA"]["pct"] - 50.0) < 0.01,
          "BETA holds %.2f%%, expected 50.00%% -- an inflated ALPHA understates "
          "every other owner at the level" % o["BETA"]["pct"])

# ── 1b. THE SHAPE THAT ACTUALLY BROKE IT: several rows, NONE ended ───────
# The scenario above does not reproduce the production defect, because closing
# the old row with a past EndDate already removed it. What Jim described is
# rows that are ALL still open -- MRI does not reliably close the superseded
# one -- and then only the latest StartDate counts. The summing version reads
# 45M for OMEGA here and hands it 81.8% of a level it actually holds 50% of.
src = source_from([
    {"EntityID": "DEAL1B", "InvestorID": "OMEGA", "Amount": 15_000_000,
     "StartDate": "2023-01-01", "EndDate": None, "CapitalPercent": 0},
    {"EntityID": "DEAL1B", "InvestorID": "OMEGA", "Amount": 30_000_000,
     "StartDate": "2024-07-01", "EndDate": None, "CapitalPercent": 0},
    {"EntityID": "DEAL1B", "InvestorID": "SIGMA", "Amount": 30_000_000,
     "StartDate": "2023-01-01", "EndDate": None, "CapitalPercent": 0},
])
o = by_id(_owners_of(src, "DEAL1B"))
check(o.get("OMEGA", {}).get("committed") == 30_000_000,
      "OMEGA committed is %s, expected 30,000,000 -- an earlier row that was "
      "never closed is still being added in"
      % f"{o.get('OMEGA', {}).get('committed', 0):,.0f}")
check("OMEGA" in o and abs(o["OMEGA"]["pct"] - 50.0) < 0.01,
      "OMEGA holds %.2f%%, expected 50.00%%" % o.get("OMEGA", {}).get("pct", -1))
check("SIGMA" in o and abs(o["SIGMA"]["pct"] - 50.0) < 0.01,
      "SIGMA holds %.2f%%, expected 50.00%% -- inflating one owner understates "
      "every other owner at the level" % o.get("SIGMA", {}).get("pct", -1))
check(o.get("OMEGA", {}).get("since") == "2024-07-01",
      "OMEGA effective date is %r, expected 2024-07-01"
      % o.get("OMEGA", {}).get("since"))

# ── 2. TWO amendments, so summing is off by more than one row ────────────
src = source_from([
    {"EntityID": "DEAL2", "InvestorID": "GAMMA", "Amount": 5_000_000,
     "StartDate": "2022-01-01", "EndDate": "2023-01-01", "CapitalPercent": 0},
    {"EntityID": "DEAL2", "InvestorID": "GAMMA", "Amount": 8_000_000,
     "StartDate": "2023-01-02", "EndDate": "2024-01-01", "CapitalPercent": 0},
    {"EntityID": "DEAL2", "InvestorID": "GAMMA", "Amount": 12_000_000,
     "StartDate": "2024-01-02", "EndDate": None, "CapitalPercent": 0},
])
o = by_id(_owners_of(src, "DEAL2"))
check(o.get("GAMMA", {}).get("committed") == 12_000_000,
      "GAMMA committed is %s, expected 12,000,000"
      % f"{o.get('GAMMA', {}).get('committed', 0):,.0f}")

# ── 3. A FUTURE EndDate is still an ending date ──────────────────────────
# "no ending date" is the rule, not "has not ended yet". The earlier code
# admitted a future EndDate and would pick the wrong row here.
src = source_from([
    {"EntityID": "DEAL3", "InvestorID": "DELTA", "Amount": 9_000_000,
     "StartDate": "2026-01-01", "EndDate": "2099-12-31", "CapitalPercent": 0},
    {"EntityID": "DEAL3", "InvestorID": "DELTA", "Amount": 4_000_000,
     "StartDate": "2025-01-01", "EndDate": None, "CapitalPercent": 0},
])
o = by_id(_owners_of(src, "DEAL3"))
check(o.get("DELTA", {}).get("committed") == 4_000_000,
      "DELTA committed is %s, expected 4,000,000 -- a row with a future EndDate "
      "is being treated as open"
      % f"{o.get('DELTA', {}).get('committed', 0):,.0f}")

# ── 4. Co-equal commitments on the SAME latest date are summed ───────────
src = source_from([
    {"EntityID": "DEAL4", "InvestorID": "EPS", "Amount": 3_000_000,
     "StartDate": "2025-05-01", "EndDate": None, "CapitalPercent": 0},
    {"EntityID": "DEAL4", "InvestorID": "EPS", "Amount": 2_000_000,
     "StartDate": "2025-05-01", "EndDate": None, "CapitalPercent": 0},
])
o = by_id(_owners_of(src, "DEAL4"))
check(o.get("EPS", {}).get("committed") == 5_000_000,
      "EPS committed is %s, expected 5,000,000 -- two genuine commitments "
      "starting the same day are one position"
      % f"{o.get('EPS', {}).get('committed', 0):,.0f}")

# ── 5. An entity whose every commitment has ended has no owners ──────────
src = source_from([
    {"EntityID": "DEAL5", "InvestorID": "ZETA", "Amount": 7_000_000,
     "StartDate": "2020-01-01", "EndDate": "2021-01-01", "CapitalPercent": 0},
])
check(_owners_of(src, "DEAL5") == [],
      "a fully closed commitment history still reports an owner")

# ── 6. The shares still total 100% after all of this ─────────────────────
src = source_from([
    {"EntityID": "DEAL6", "InvestorID": "A", "Amount": 1_000_000,
     "StartDate": "2020-01-01", "EndDate": "2021-01-01", "CapitalPercent": 0},
    {"EntityID": "DEAL6", "InvestorID": "A", "Amount": 30_000_000,
     "StartDate": "2021-01-02", "EndDate": None, "CapitalPercent": 0},
    {"EntityID": "DEAL6", "InvestorID": "B", "Amount": 10_000_000,
     "StartDate": "2021-01-02", "EndDate": None, "CapitalPercent": 0},
])
owners = _owners_of(src, "DEAL6")
total = sum(x["pct"] for x in owners)
check(abs(total - 100.0) < 0.001, "shares total %.4f%%, expected 100%%" % total)

# ── 6b. EVERY FLAVOUR OF NULL COUNTS AS "no end date" ────────────────────
# THE ONE THAT REACHED PRODUCTION. The open test used to render the cell with
# astype(str) and match the result against ("", "none", "nan", "nat", "null").
# pd.NA renders as "<NA>" and is not in that list -- and pd.NA is what
# PostgreSQL produces where SQLite produces None. On Azure every one of 601
# commitments failed the test, the table emptied, and the screen reported "no
# commitments" for every deal in the portfolio while every local test passed.
#
# So this asserts the property, not the spelling: each null flavour, on its
# own, must leave the row standing.
for label, null in (("None", None), ("NaT", pd.NaT),
                    ("float nan", float("nan")), ("pd.NA", pd.NA)):
    src = source_from([
        {"EntityID": "DEALN", "InvestorID": "ALPHA", "Amount": 1_000_000,
         "StartDate": "2024-01-01", "EndDate": null, "CapitalPercent": 0},
    ])
    owners = _owners_of(src, "DEALN")
    check(len(owners) == 1,
          f"an EndDate of {label} dropped the row — it must read as 'no end "
          f"date'. This is the defect that emptied the production table.")

# A genuine end date still ends it, whichever flavour of null sits beside it.
src = source_from([
    {"EntityID": "DEALN2", "InvestorID": "A", "Amount": 1_000_000,
     "StartDate": "2020-01-01", "EndDate": "2021-01-01", "CapitalPercent": 0},
    {"EntityID": "DEALN2", "InvestorID": "B", "Amount": 4_000_000,
     "StartDate": "2020-01-01", "EndDate": pd.NA, "CapitalPercent": 0},
])
o = by_id(_owners_of(src, "DEALN2"))
check("A" not in o, "a dated EndDate no longer ends the commitment")
check("B" in o and abs(o["B"]["pct"] - 100.0) < 0.01,
      "the open commitment did not survive beside a closed one")

# ── 6c. Timezone-aware StartDates, as PostgreSQL returns them ────────────
# timestamptz arrives tz-aware; SQLite strings arrive naive. A frame carrying
# either must still pick the latest row rather than raise or match nothing.
src = source_from([
    {"EntityID": "DEALTZ", "InvestorID": "A", "Amount": 1_000_000,
     "StartDate": "2023-01-01T00:00:00+00:00", "EndDate": None, "CapitalPercent": 0},
    {"EntityID": "DEALTZ", "InvestorID": "A", "Amount": 7_000_000,
     "StartDate": "2024-06-01T00:00:00+00:00", "EndDate": None, "CapitalPercent": 0},
])
o = by_id(_owners_of(src, "DEALTZ"))
check(o.get("A", {}).get("committed") == 7_000_000,
      "tz-aware StartDates did not resolve to the latest row (got %s)"
      % f"{o.get('A', {}).get('committed', 0):,.0f}")

# ── 7. Column names are matched WITHOUT REGARD TO CASE ───────────────────
# PostgreSQL folds unquoted identifiers to lower case; SQLite preserves them.
# The same table is `EntityID` locally and `entityid` on Azure, and matching a
# single spelling meant the screen either reported "no commitments" or raised,
# depending on which column happened to miss. Nothing local can catch that,
# which is why it reached production.
from flask_app.services.ownership_chain_service import _Source  # noqa: E402

for spelling in ("EntityID", "entityid", "ENTITYID"):
    df = pd.DataFrame([
        {"CommitmentUID": 832, spelling: "30BEAR",
         "InvestorID" if spelling == "EntityID" else "investorid": "PPI27",
         "Amount" if spelling == "EntityID" else "amount": 3_000_000,
         "CapitalPercent": 100, "StartDate": "2020-08-07T00:00:00",
         "EndDate": None},
    ])
    out = _Source._canonicalise(df, ("EntityID", "InvestorID", "Amount",
                                     "CapitalPercent", "StartDate", "EndDate"))
    check("EntityID" in out.columns,
          f"_canonicalise left {spelling!r} unresolved; PostgreSQL's lower-cased "
          f"columns would not be found")
    check("InvestorID" in out.columns and "Amount" in out.columns,
          f"_canonicalise resolved the entity column for {spelling!r} but not "
          f"its siblings")

# It must not invent a column that genuinely is not there.
bare = _Source._canonicalise(pd.DataFrame([{"nothing": 1}]),
                             ("EntityID", "Amount"))
check("EntityID" not in bare.columns,
      "_canonicalise invented an EntityID column out of nothing")

# ── 8. The fixture carries everything _owners_of actually reads ──────────
# This exists because the fixture DID go stale: `balances` was added to
# _Source.__init__ and the guardrail died with AttributeError instead of
# checking anything. A hand-built stand-in for a real object needs a test that
# it is still a faithful stand-in.
_probe = source_from([
    {"EntityID": "DEALF", "InvestorID": "A", "Amount": 1_000_000,
     "StartDate": "2024-01-01", "EndDate": None, "CapitalPercent": 0},
])
try:
    got = _owners_of(_probe, "DEALF")
    check(len(got) == 1 and "balance" in got[0],
          "_owners_of no longer returns a balance, or the fixture cannot reach it")
except AttributeError as e:
    FAIL.append(f"the test fixture is missing an attribute _Source now has: {e}")

if FAIL:
    print("FAIL")
    for m in FAIL:
        print("  -", m)
    sys.exit(1)
print("OK - the current commitment is the latest open StartDate, not the sum; "
      "future EndDates are ended; same-day commitments combine; shares total 100%")
