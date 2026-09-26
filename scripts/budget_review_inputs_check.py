#!/usr/bin/env python
"""Guardrail: the Budget Review's analyst inputs (asset management, Sep 25 2026).

  1. THE THIRD COLUMN CAN BE UNDERWRITING. Same engine, different source -- and UW
     debt service is ONE figure (7010, P&I), so Interest and Principal are blank
     there, never split by a guess, and DSCR runs on the 7010 total.
  2. THE BUDGET COLUMN'S DEBT SERVICE CAN COME FROM UW. When UW has no figure for
     the year, the choice is REPORTED and the column is not blanked.
  3. AN ESTIMATE LINE CAN BE OVERRIDDEN, and the totals follow it. The computed
     figure is kept beside it; clearing restores it; a total cannot be overridden;
     an approved record refuses.
  4. BUDGETED OCCUPANCY IS READ OFF THE BUDGET, above or below the month header,
     and kept OUT of the mappable lines. A dollar line labelled "Occupancy Tax" is
     NOT taken for it. Out-of-range figures are refused, not clipped.
  5. THE UW DEBT SERVICE HELPER is One Pager's own logic, extracted: partial-year
     detection gives the same months it always did.

Pure fixtures, a temporary SQLite database, no network.
Run:  python scripts/budget_review_inputs_check.py
"""
import io
import os
import sys
import tempfile

import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  ok   " if cond else "  FAIL ") + name + (("  -- " + str(detail)) if detail and not cond else ""))


from flask_app.services import budget_import_service as B            # noqa: E402
from flask_app.services import valuation_budget_inputs as I          # noqa: E402
from flask_app.services import valuation_service as VS               # noqa: E402
from one_pager import uw_debt_service_for_year                       # noqa: E402

RENT = B.category_for_account("4010")
REPAIRS = B.category_for_account("5060")


def _me(y, m):
    return pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)


def isbs_rows(vcode, uw_7010_months=12):
    """Actuals through Jun 2026 (YTD cumulative), budget Jul 2026-Dec 2027 (monthly),
    UW 2027 (YTD cumulative) with 7010 growing for `uw_7010_months` months."""
    rows = []
    v = vcode.lower()
    rows += [dict(vcode=v, dtEntry_parsed=_me(2026, 6), vSource="Interim IS", vAccount="4010", mAmount=-600.0),
             dict(vcode=v, dtEntry_parsed=_me(2026, 6), vSource="Interim IS", vAccount="5060", mAmount=200.0)]
    for m in range(7, 13):
        rows += [dict(vcode=v, dtEntry_parsed=_me(2026, m), vSource="Budget IS", vAccount="4010", mAmount=-100.0),
                 dict(vcode=v, dtEntry_parsed=_me(2026, m), vSource="Budget IS", vAccount="5060", mAmount=30.0)]
    for m in range(1, 13):
        rows += [dict(vcode=v, dtEntry_parsed=_me(2027, m), vSource="Budget IS", vAccount="4010", mAmount=-110.0),
                 dict(vcode=v, dtEntry_parsed=_me(2027, m), vSource="Budget IS", vAccount="5060", mAmount=35.0),
                 dict(vcode=v, dtEntry_parsed=_me(2027, m), vSource="Budget IS", vAccount="5190", mAmount=10.0)]
        cum = m / 12
        rows += [dict(vcode=v, dtEntry_parsed=_me(2027, m), vSource="Projected IS", vAccount="4010", mAmount=-1300.0 * cum),
                 dict(vcode=v, dtEntry_parsed=_me(2027, m), vSource="Projected IS", vAccount="5060", mAmount=400.0 * cum),
                 dict(vcode=v, dtEntry_parsed=_me(2027, m), vSource="Projected IS", vAccount="7010",
                      mAmount=-50.0 * min(m, uw_7010_months))]
    return rows


tmp = os.path.join(tempfile.mkdtemp(), "wf.db")
eng = create_engine("sqlite:///" + tmp)
VS.ensure_valuation_tables(eng)
with eng.begin() as c:
    c.execute(text("INSERT INTO valuation_cycles (id, year, as_of_date) VALUES (1, 2026, '2026-12-31')"))
    for rid, vc in ((1, "P0000901"), (2, "P0000902"), (3, "P0000903")):
        c.execute(text("INSERT INTO valuation_records (id, cycle_id, vcode, status) VALUES (:i, 1, :v, 'draft')"),
                  {"i": rid, "v": vc})

isbs = pd.DataFrame(isbs_rows("P0000901") + isbs_rows("P0000902", uw_7010_months=3)
                    + [r for r in isbs_rows("P0000903") if r["vSource"] != "Projected IS"])
data = {"isbs_raw": isbs, "occupancy_raw": None}


def row(rev, label):
    return next(r for r in rev["rows"] if r["account"] == label)


print("\n1. The third column can be underwriting")
val = VS.get_budget_review(eng, 1, data)
uw = VS.get_budget_review(eng, 1, data, compare="underwriting")
check("default is Valuation Yr 1", val["compare"]["label"] == "Valuation Yr 1", val["compare"])
check("underwriting is labelled for the budget year", uw["compare"]["label"] == "UW 2027", uw["compare"])
check("UW revenue is UW's December cumulative (1,300)",
      abs(row(uw, RENT)["valuation"] - 1300) < 0.01, row(uw, RENT))
check("UW NOI = 1,300 - 400", abs(row(uw, "Net Operating Income")["valuation"] - 900) < 0.01,
      row(uw, "Net Operating Income"))
check("UW Total Debt Service is the 7010 total (600)",
      abs(row(uw, "Total Debt Service")["valuation"] - 600) < 0.01, row(uw, "Total Debt Service"))
check("UW Interest and Principal are BLANK, not split",
      row(uw, "Interest Expense")["valuation"] is None and row(uw, "Principal Payments")["valuation"] is None,
      (row(uw, "Interest Expense"), row(uw, "Principal Payments")))
check("UW DSCR runs on the 7010 total (900 / 600 = 1.50)",
      abs(row(uw, "DSCR")["valuation"] - 1.5) < 1e-9, row(uw, "DSCR"))
check("the Estimate and Budget columns do not move when the third column does",
      all(a["estimate"] == b["estimate"] and a["budget"] == b["budget"]
          for a, b in zip(val["rows"], uw["rows"])))
nouw = VS.get_budget_review(eng, 3, data, compare="underwriting")
check("no UW year: reported unavailable, with a reason", not nouw["compare"]["available"]
      and "does not run through" in (nouw["compare"]["note"] or ""), nouw["compare"])
try:
    VS.get_budget_review(eng, 1, data, compare="bogus")
    check("an unknown comparison is refused", False)
except ValueError:
    check("an unknown comparison is refused", True)

print("\n2. The Budget column's debt service can come from UW")
before = row(val, "Total Debt Service")["budget"]
check("before: Budget debt service is the file's 5190 (12 x 10 = 120)", abs(before - 120) < 0.01, before)
I.set_debt_basis(eng, 1, "underwriting", "guardrail")
after = VS.get_budget_review(eng, 1, data)
check("after: Budget Total Debt Service is UW's 7010 (600)",
      abs(row(after, "Total Debt Service")["budget"] - 600) < 0.01, row(after, "Total Debt Service"))
check("...Budget Interest/Principal blank, since UW does not split them",
      row(after, "Interest Expense")["budget"] is None, row(after, "Interest Expense"))
check("...and the payload says the basis was applied",
      after["debt_service"]["budget_basis"] == "underwriting" and after["debt_service"]["budget_basis_applied"])
check("...Budget DSCR recomputes on it",
      abs(row(after, "DSCR")["budget"] - row(after, "Net Operating Income")["budget"] / 600) < 1e-9)
I.set_debt_basis(eng, 3, "underwriting", "guardrail")
none = VS.get_budget_review(eng, 3, data)
check("UW chosen but absent: NOT applied, column NOT blanked, and it SAYS so",
      not none["debt_service"]["budget_basis_applied"]
      and abs(row(none, "Total Debt Service")["budget"] - 120) < 0.01
      and any("carries no 2027" in n for n in none["debt_service"]["basis_notes"]),
      none["debt_service"])
I.set_debt_basis(eng, 2, "underwriting", "guardrail")
part = VS.get_budget_review(eng, 2, data)
check("partial-year UW debt service is reported (3 months), not annualised into the budget",
      abs(row(part, "Total Debt Service")["budget"] - 150) < 0.01
      and any("covers 3 months" in n for n in part["debt_service"]["basis_notes"]),
      part["debt_service"]["basis_notes"])
try:
    I.set_debt_basis(eng, 1, "whatever", "guardrail")
    check("an unknown basis is refused", False)
except ValueError:
    check("an unknown basis is refused", True)
I.set_debt_basis(eng, 1, "modeled", "guardrail")

print("\n3. An Estimate line can be overridden, and the totals follow")
base = VS.get_budget_review(eng, 1, data)
rent0 = row(base, RENT)["estimate"]
noi0 = row(base, "Net Operating Income")["estimate"]
check("computed Estimate rent = 600 YTD + 6 x 100 budget", abs(rent0 - 1200) < 0.01, rent0)
I.set_override(eng, 1, RENT, 1000.0, rent0, "Lost the anchor in Q3", "guardrail")
o = VS.get_budget_review(eng, 1, data)
r = row(o, RENT)
check("the overridden line shows the analyst's figure", r["estimate"] == 1000.0, r)
check("...flagged, with the computed figure kept beside it",
      r.get("estimate_overridden") and abs(r["estimate_computed"] - 1200) < 0.01, r)
check("...and the note and who", r.get("override_note") == "Lost the anchor in Q3" and r.get("override_by") == "guardrail", r)
check("Total Revenues follows the override and says it includes one",
      abs(row(o, "Total Revenues")["estimate"] - 1000) < 0.01 and row(o, "Total Revenues")["estimate_includes_override"])
check("NOI moves by exactly the override (-200)",
      abs(row(o, "Net Operating Income")["estimate"] - (noi0 - 200)) < 0.01)
check("the variance to Budget recomputes from the override",
      abs(r["var_est_bud"] - (1000 - r["budget"])) < 0.01, r)
check("Budget and third columns are untouched by an Estimate override",
      all(a["budget"] == b["budget"] and a["valuation"] == b["valuation"]
          for a, b in zip(base["rows"], o["rows"])))
I.set_override(eng, 1, "Interest Expense", 55.0, 0.0, None, "guardrail")
o2 = VS.get_budget_review(eng, 1, data)
check("a debt-service line can be overridden, and Total Debt Service follows",
      abs(row(o2, "Total Debt Service")["estimate"] - (55 + row(o2, "Principal Payments")["estimate"])) < 0.01)
I.set_override(eng, 1, "Interest Expense", None, None, None, "guardrail")
I.set_override(eng, 1, RENT, None, None, None, "guardrail")
cleared = VS.get_budget_review(eng, 1, data)
check("clearing restores the computed figure", row(cleared, RENT)["estimate"] == rent0
      and not row(cleared, RENT).get("estimate_overridden"))
for label in ("Net Operating Income", "Total Revenues", "DSCR", "Total Debt Service"):
    try:
        I.set_override(eng, 1, label, 1.0, None, None, "guardrail")
        check(f"'{label}' cannot be overridden (computed from lines)", False)
    except ValueError:
        check(f"'{label}' cannot be overridden (computed from lines)", True)
with eng.begin() as c:
    c.execute(text("UPDATE valuation_records SET status = 'approved' WHERE id = 3"))
for label, fn in (("an override", lambda: I.set_override(eng, 3, RENT, 1.0, None, None, "g")),
                  ("a basis change", lambda: I.set_debt_basis(eng, 3, "modeled", "g"))):
    try:
        fn()
        check(f"an APPROVED record refuses {label}", False)
    except ValueError:
        check(f"an APPROVED record refuses {label}", True)

print("\n4. Budgeted occupancy is read off the budget, and kept out of the lines")
MONTHS = [f"2027-{m:02d}-01" for m in range(1, 13)]


def xlsx(rows):
    buf = io.BytesIO()
    pd.DataFrame(rows).to_excel(buf, index=False, header=False)
    return buf.getvalue()


above = xlsx([["Budgeted Occupancy"] + [0.95] * 3 + [0.97] * 9,
              [None] + MONTHS,
              ["4010 - Rental Income"] + [100.0] * 12,
              ["5060 - Repairs"] + [7.0] * 12])
p = B.parse_budget_workbook(above, "above.xlsx")
occ = p.get("occupancy") or {}
check("an occupancy row ABOVE the month header is found", occ.get("label") == "Budgeted Occupancy", occ)
check("...fractions read as percentages (0.95 -> 95)",
      occ.get("by_period", {}).get("2027-01-31") == 95.0 and occ["by_period"].get("2027-12-31") == 97.0, occ)
check("...and it is not a mappable line", all("ccupan" not in l["label"] for l in p["lines"])
      and len(p["lines"]) == 2, [l["label"] for l in p["lines"]])

# A % cell arrives from Excel as a NUMBER (0.93), which the line reader would happily
# take as an amount -- so this is the case where keeping it out of the lines matters.
# (A first version wrote "93%" as text; the line reader cannot parse that, so the row
# never became a line whatever the code did and the check passed vacuously.)
below = xlsx([[None] + MONTHS,
              ["Occupancy %"] + [0.93] * 12,
              ["4010 - Rental Income"] + [100.0] * 12])
p2 = B.parse_budget_workbook(below, "below.xlsx")
check("an occupancy row BELOW the header (0.93 as Excel holds a % cell) reads 93",
      (p2.get("occupancy") or {}).get("by_period", {}).get("2027-06-30") == 93.0, p2.get("occupancy"))
check("...and is NOT a mappable line", len(p2["lines"]) == 1, [l["label"] for l in p2["lines"]])
check("'93%' written as text reads 93, '0.95' as 95, and text is not a number",
      I.to_pct("93%") == 93.0 and I.to_pct(0.95) == 95.0 and I.to_pct("n/a") is None)

tax = xlsx([[None] + MONTHS,
            ["Occupancy Tax"] + [1250.0] * 12,
            ["4010 - Rental Income"] + [100.0] * 12])
p3 = B.parse_budget_workbook(tax, "tax.xlsx")
check("a DOLLAR line labelled 'Occupancy Tax' is NOT taken for occupancy",
      not p3.get("occupancy") and any(l["label"] == "Occupancy Tax" for l in p3["lines"]),
      (p3.get("occupancy"), [l["label"] for l in p3["lines"]]))

annual = xlsx([["2027 Budgeted Occupancy", 0.92],
               [None] + MONTHS,
               ["4010 - Rental Income"] + [100.0] * 12])
p4 = B.parse_budget_workbook(annual, "annual.xlsx")
o4 = p4.get("occupancy") or {}
check("ONE annual figure applies to every month, and says so",
      o4.get("basis") == "annual" and len(o4.get("by_period", {})) == 12
      and set(o4["by_period"].values()) == {92.0}, o4)

bad = I.normalise_occupancy({"2027-01-31": 95.0, "2027-02-28": 950.0})
check("an out-of-range month is REFUSED, not clipped to 100",
      "2027-02-28" not in bad["by_period"] and bad["rejected_periods"] == ["2027-02-28"], bad)

I.save_occupancy(eng, "P0000901", occ["by_period"], "above.xlsx", "guardrail")
q = I.budget_occupancy_quarters(eng, "P0000901", 2027)
check("stored monthly, averaged to quarters labelled like the MRI history",
      [x["quarter"] for x in q] == ["2027-Q1", "2027-Q2", "2027-Q3", "2027-Q4"]
      and q[0]["occupancy"] == 95.0 and q[1]["occupancy"] == 97.0, q)
rev = VS.get_budget_review(eng, 1, data)
bq = [x for x in rev["occupancy_trend"] if x.get("budgeted")]
check("the review's occupancy trend carries the budgeted quarters, flagged",
      len(bq) == 4 and all(x["budgeted"] for x in bq), rev["occupancy_trend"])
I.save_occupancy(eng, "P0000901", {"2027-01-31": 90.0}, "v2.xlsx", "guardrail")
q2 = I.budget_occupancy_quarters(eng, "P0000901", 2027)
check("a re-import REPLACES the months it carries and leaves the others",
      abs(q2[0]["occupancy"] - (90 + 95 + 95) / 3) < 1e-9 and q2[1]["occupancy"] == 97.0, q2)

print("\n5. The UW debt service helper is One Pager's logic, unchanged")
uwp = isbs[(isbs["vcode"] == "p0000902") & (isbs["vSource"] == "Projected IS")]
h = uw_debt_service_for_year(uwp, 2027)
check("partial year: 150 over 3 active months", abs(h["amount"] - 150) < 0.01 and h["months_active"] == 3, h)
uwf = isbs[(isbs["vcode"] == "p0000901") & (isbs["vSource"] == "Projected IS")]
hf = uw_debt_service_for_year(uwf, 2027)
check("full year: 600 over 12 months", abs(hf["amount"] - 600) < 0.01 and hf["months_active"] == 12, hf)
check("no UW data: zero, never an exception", uw_debt_service_for_year(pd.DataFrame(), 2027)["amount"] == 0.0)

from database import PROTECTED_TABLES                                  # noqa: E402
check("both new tables are protected from a CSV replace",
      {"valuation_estimate_overrides", "valuation_budget_occupancy"} <= PROTECTED_TABLES)

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
if FAIL:
    print("FAILED:")
    for f in FAIL:
        print("   " + f)
sys.exit(1 if FAIL else 0)
