#!/usr/bin/env python
"""Guardrail: asset management's three import rules (Jack Day, Oct 7 2026).

  1. ONLY A ROW WITH A 4-DIGIT ACCOUNT IS IMPORTED. "if a row has a 4-digit account
     number, bring it in and map it by that account." His own example is the fixture:
     5060 Repairs $20 | Internet Marketing $30 (no account) | 5040 Mgmt Fee $40 |
     5060 Total Repairs $90 -- three come in, the TOTAL included; Internet Marketing does
     not. Both directions: an accounted row is pre-filled, an unaccounted one is refused
     even when a payload maps it; 3- and 5-digit numbers are not accounts.
  2. THE BUDGET COLUMN'S DEBT SERVICE IS THE BUDGET'S OWN by default (5190 / 7060). A
     budget with neither is BLANK and says so -- never $0 -- and "modeled" stays choosable.
  3. LOSS TO LEASE (4042) NETS AGAINST RENTAL INCOME on the Budget Review, in every column,
     and is not Vacancy there. Total revenue does not move. `config.IS_ACCOUNTS` -- which
     Financials and the One Pager read -- still has 4042 under Vacancy.

Pure fixtures, a temporary SQLite database, no network.
Run:  python scripts/am_import_rules_check.py [--inject=anydigits|skiptotals|noblock|debtmodel|vacancy|global|noprefill|forget]
"""
import io
import os
import re
import sys
import tempfile
from datetime import date

import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INJECT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--inject=")), "")

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  ok   " if cond else "  FAIL ") + name + (("  -- " + str(detail)) if detail and not cond else ""))


import logging                                                       # noqa: E402
logging.disable(logging.WARNING)
import config                                                        # noqa: E402
from flask_app.services import budget_import_service as B            # noqa: E402
from flask_app.services import budget_import_validate as V           # noqa: E402
from flask_app.services import line_mapping_service as LM            # noqa: E402
from flask_app.services import valuation_budget_inputs as I          # noqa: E402
from flask_app.services import valuation_service as VS               # noqa: E402

# ── the injected defects ───────────────────────────────────────────────────────
if INJECT == "anydigits":      # any 3-6 digit number counts as an account
    B._IMPORT_ACCOUNT_RE = re.compile(r'^\d{3,6}$')
if INJECT == "skiptotals":     # a total that carries an account is left out again
    _wfa = B.with_file_accounts
    def _skip(parsed):
        _wfa(parsed)
        for l in parsed.get("lines") or []:
            if l.get("looks_like_total"):
                l["file_account"] = None
        return parsed
    B.with_file_accounts = _skip
if INJECT == "noblock":        # a row with no account is let through
    _fa = B.file_account
    B.file_account = lambda l: _fa(l) or "5050"
if INJECT == "debtmodel":      # the default basis goes back to the model
    I.DEFAULT_DEBT_BASIS = "modeled"
if INJECT == "vacancy":        # 4042 stays in Vacancy on the review
    B._REVIEW_MOVES = {}
if INJECT == "global":         # moved in config itself, so Financials and One Pager move too
    config.IS_ACCOUNTS["REVENUES"]["Vacancy"].remove("4042")
    config.IS_ACCOUNTS["REVENUES"]["Rental Income"].append("4042")

MONTHS = [f"{m}/{[31,28,31,30,31,30,31,31,30,31,30,31][m-1]}/2027" for m in range(1, 13)]


def xlsx(rows):
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    for r in rows:
        ws.append(r)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _me(y, m):
    return pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)


# ── fixture: two records; P0000951's 2027 budget carries 5190, P0000952's does not ──
def isbs_rows(vcode, with_debt):
    v, rows = vcode.lower(), []
    # Estimate: actuals through Jun 2026 (YTD cumulative), budget Jul-Dec 2026.
    rows += [dict(vcode=v, dtEntry_parsed=_me(2026, 6), vSource="Interim IS", vAccount="4010", mAmount=-600.0),
             dict(vcode=v, dtEntry_parsed=_me(2026, 6), vSource="Interim IS", vAccount="4042", mAmount=60.0),
             dict(vcode=v, dtEntry_parsed=_me(2026, 6), vSource="Interim IS", vAccount="4030", mAmount=30.0)]
    for m in range(7, 13):
        rows += [dict(vcode=v, dtEntry_parsed=_me(2026, m), vSource="Budget IS", vAccount="4010", mAmount=-100.0)]
    for m in range(1, 13):
        rows += [dict(vcode=v, dtEntry_parsed=_me(2027, m), vSource="Budget IS", vAccount="4010", mAmount=-110.0),
                 dict(vcode=v, dtEntry_parsed=_me(2027, m), vSource="Budget IS", vAccount="4042", mAmount=5.0),
                 dict(vcode=v, dtEntry_parsed=_me(2027, m), vSource="Budget IS", vAccount="5060", mAmount=35.0)]
        if with_debt:
            rows += [dict(vcode=v, dtEntry_parsed=_me(2027, m), vSource="Budget IS", vAccount="5190", mAmount=10.0),
                     dict(vcode=v, dtEntry_parsed=_me(2027, m), vSource="Budget IS", vAccount="7060", mAmount=4.0)]
    return rows


tmp = os.path.join(tempfile.mkdtemp(), "wf.db")
eng = create_engine("sqlite:///" + tmp)
VS.ensure_valuation_tables(eng)
with eng.begin() as c:
    c.execute(text("INSERT INTO valuation_cycles (id, year, as_of_date) VALUES (1, 2026, '2026-12-31')"))
    for rid, vc in ((1, "P0000951"), (2, "P0000952")):
        c.execute(text("INSERT INTO valuation_records (id, cycle_id, vcode, status) VALUES (:i, 1, :v, 'draft')"),
                  {"i": rid, "v": vc})
isbs = pd.DataFrame(isbs_rows("P0000951", True) + isbs_rows("P0000952", False))
isbs["dtEntry"] = isbs["dtEntry_parsed"].dt.strftime("%m/%d/%Y")   # the raw column, as MRI sends it
isbs["vDescription"] = ""
data = {"isbs_raw": isbs, "occupancy_raw": None}


def row(rev, label):
    return next(r for r in rev["rows"] if r["account"] == label)


# ═════════════════════════════════════════════════════════════════════════════
print("\n1. Only a row with a 4-digit account is imported -- Jack's own example")
sheet = xlsx([["Account", "Line Item"] + MONTHS,
              [5060, "Repairs & Maintenance"] + [20 / 12] * 12,
              [None, "Internet Marketing"] + [30 / 12] * 12,
              [5040, "Management Fee"] + [40 / 12] * 12,
              [5060, "Total Repairs & Maintenance"] + [90 / 12] * 12])
pa = LM.parse(eng, 1, "budget", sheet, "jack.xlsx", data)
by = {l["label"]: l for l in pa["lines"]}
sug = {l["label"]: pa["suggested"].get(str(l["row"])) for l in pa["lines"]}
check("all four rows are read", set(by) == {"Repairs & Maintenance", "Internet Marketing",
                                            "Management Fee", "Total Repairs & Maintenance"}, list(by))
check("Repairs & Maintenance is pre-filled 5060", (sug["Repairs & Maintenance"] or {}).get("account") == "5060",
      sug["Repairs & Maintenance"])
check("Management Fee is pre-filled 5040", (sug["Management Fee"] or {}).get("account") == "5040",
      sug["Management Fee"])
check("the TOTAL with an account is pre-filled too (5060) -- no guessing it is a subtotal",
      (sug["Total Repairs & Maintenance"] or {}).get("account") == "5060", sug["Total Repairs & Maintenance"])
check("Internet Marketing (no account) is NOT pre-filled", sug["Internet Marketing"] is None,
      sug["Internet Marketing"])
check("...and is marked as carrying no file account", by["Internet Marketing"].get("file_account") is None,
      by["Internet Marketing"].get("file_account"))
mapping = {k: dict(v) for k, v in pa["suggested"].items()}
ck = LM.check(eng, 1, "budget", pa, mapping, data)
check("the three accounted rows import", ck["can_import"] and ck["mapped_count"] == 3, ck["blocking"])
check("...and the screen is told 1 row has no account", ck["no_account_count"] == 1, ck["no_account_count"])
bad = dict(mapping)
bad[str(by["Internet Marketing"]["row"])] = {"account": "5010", "flip": False}
cb = LM.check(eng, 1, "budget", pa, bad, data)
check("a payload that maps Internet Marketing anyway is REFUSED, by name",
      not cb["can_import"] and any(b["code"] == "no_file_account" and "Internet Marketing" in b["message"]
                                   for b in cb["blocking"]), cb["blocking"])
lo = dict(mapping)
del lo[str(by["Total Repairs & Maintenance"]["row"])]
cl = LM.check(eng, 1, "budget", pa, lo, data)
check("clearing the total (to avoid a double count) is allowed, and LISTED as left out",
      cl["can_import"] and [x["account"] for x in cl["left_out"]] == ["5060"], cl["left_out"])
check("3 digits is not an account", B.file_account({"stated_account": "506", "label": "x"}) is None)
check("5 digits is not an account", B.file_account({"stated_account": "50600", "label": "x"}) is None)
check("an account leading the label counts ('4010 - Rental Income')",
      B.file_account({"stated_account": None, "label": "4010 - Rental Income"}) == "4010")
check("a label with no number has none", B.file_account({"stated_account": None, "label": "Rental Income"}) is None)

# ═════════════════════════════════════════════════════════════════════════════
print("\n2. The Budget column's debt service is the budget's own, by default")
r1 = VS.get_budget_review(eng, 1, data)
d1 = r1["debt_service"]
check("the default basis is 'budget'", d1["budget_basis"] == "budget", d1["budget_basis"])
check("Budget interest is the budget's 5190 (12 x 10 = 120)",
      abs((row(r1, "Interest Expense")["budget"] or 0) - 120) < 0.01, row(r1, "Interest Expense"))
check("Budget principal is the budget's 7060 (12 x 4 = 48)",
      abs((row(r1, "Principal Payments")["budget"] or 0) - 48) < 0.01, row(r1, "Principal Payments"))
check("...and the payload says the column holds the budget's figures", d1["budget_from"] == "budget",
      d1["budget_from"])
r2 = VS.get_budget_review(eng, 2, data)
d2 = r2["debt_service"]
check("a budget with no 5190/7060 shows BLANK debt service, not $0",
      row(r2, "Interest Expense")["budget"] is None and row(r2, "Total Debt Service")["budget"] is None,
      (row(r2, "Interest Expense"), row(r2, "Total Debt Service")))
check("...a blank Budget DSCR, not 0.00", row(r2, "DSCR")["budget"] is None, row(r2, "DSCR"))
check("...and SAYS so", any("carries no debt service" in n for n in d2["basis_notes"]), d2["basis_notes"])
I.set_debt_basis(eng, 1, "modeled", "guardrail")
m1 = VS.get_budget_review(eng, 1, data)
check("'modeled' is still choosable; with no loans to model it falls back to the budget's figure and says so",
      m1["debt_service"]["budget_basis"] == "modeled" and m1["debt_service"]["budget_from"] == "budget"
      and any("no loans are modeled" in n for n in m1["debt_service"]["basis_notes"]),
      m1["debt_service"])
I.set_debt_basis(eng, 1, "budget", "guardrail")
check("the bases on offer", I.DEBT_BASES == ("budget", "modeled", "underwriting"), I.DEBT_BASES)

# ═════════════════════════════════════════════════════════════════════════════
print("\n3. Loss to lease (4042) nets against Rental Income on the review, in every column")
RENT, VAC = "Rental Income", "Vacancy"
check("the import shows 4042 under Rental Income", B.category_for_account("4042") == RENT,
      B.category_for_account("4042"))
check("config still has 4042 under Vacancy (Financials, One Pager)",
      "4042" in config.IS_ACCOUNTS["REVENUES"][VAC]
      and "4042" not in config.IS_ACCOUNTS["REVENUES"][RENT])
# Estimate: 600 rent YTD + 600 budget remainder, less 60 loss to lease = 1,140; vacancy is 4030's 30.
check("Estimate Rental Income is net of loss to lease (1,200 - 60 = 1,140)",
      abs(row(r1, RENT)["estimate"] - 1140) < 0.01, row(r1, RENT))
check("Estimate Vacancy no longer carries it (4030's 30 only)",
      abs(row(r1, VAC)["estimate"] - (-30)) < 0.01, row(r1, VAC))
check("Budget Rental Income is net (1,320 - 60 = 1,260)", abs(row(r1, RENT)["budget"] - 1260) < 0.01,
      row(r1, RENT))
check("Budget Vacancy is zero -- the budget's only contra-revenue was 4042",
      abs(row(r1, VAC)["budget"]) < 0.01, row(r1, VAC))
check("Total Revenues does not move: Estimate 1,200 - 60 - 30 = 1,110",
      abs(row(r1, "Total Revenues")["estimate"] - 1110) < 0.01, row(r1, "Total Revenues"))

# The Valuation column: an Argus line coded 4042 reduces rental income.
from flask_app.services import argus_service                          # noqa: E402
fc = pd.DataFrame([{"vcode": "P0000951", "event_date": date(2027, m, 28), "vAccount": acct, "mAmount_norm": amt}
                   for m in range(1, 13)
                   for acct, amt in (("4010", 1000.0), ("4042", -50.0))])
orig = argus_service.get_forecast_df_by_id
argus_service.get_forecast_df_by_id = lambda *a, **k: fc
try:
    with eng.begin() as c:
        c.execute(text("UPDATE valuation_records SET argus_import_id = 1 WHERE id = 1"))
    rv = VS.get_budget_review(eng, 1, data)
finally:
    argus_service.get_forecast_df_by_id = orig
check("Valuation Rental Income is net of the Argus loss to lease (12,000 - 600 = 11,400)",
      abs(row(rv, RENT)["valuation"] - 11400) < 0.01, row(rv, RENT))
check("...and Valuation Vacancy is untouched by it", abs(row(rv, VAC)["valuation"]) < 0.01, row(rv, VAC))
# What the Argus import WRITES for a 4042 line read as -600 for the year: the account's
# own sign rule makes it a debit, which is what nets it against rent.
written = V.imported_amounts({"amounts": {"2027-01-31": -600.0}}, {"account": "4042"}, "argus")
check("an Argus 4042 line is written as a debit (reduces revenue), whichever sign the file shows",
      written.get("2027-01-31") == 600.0 and
      V.imported_amounts({"amounts": {"2027-01-31": 600.0}}, {"account": "4042"}, "argus").get("2027-01-31") == 600.0,
      written)

# ═════════════════════════════════════════════════════════════════════════════
print("\n4. The screen: a saved mapping is read by the rule, and a leave-out is kept")
# Found on Camp Creek's saved budget mapping: it imported interest through unnumbered
# "CIBC" rows and left "5190 Total Interest" out. Under the rule the CIBC rows drop, so
# re-applying the mapping as saved would have imported NO interest.
pn = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "vue_app", "src", "components", "common", "LineMappingPanel.vue"),
          encoding="utf-8").read()
if INJECT == "noprefill":      # a saved draft is applied as saved
    pn = pn.replace("      prefillFromFile()\n", "")
if INJECT == "forget":         # clearing an accounted row is not recorded
    pn = pn.replace("if (line?.file_account) mapping.value = { ...mapping.value, [key]: { left_out: true } }", "")
check("loading a saved draft fills its undecided accounted rows from the file",
      "      prefillFromFile()\n" in pn and "async function loadDraft" in pn
      and pn.index("      prefillFromFile()\n") > pn.index("async function loadDraft"))
check("...skipping a row the analyst left out, and any account we do not carry",
      "next[key]?.left_out) continue" in pn and "if (!a) continue" in pn)
check("...and says how many it filled", "filledOnLoad" in pn and "now imported under it" in pn)
check("clearing a row that carries an account RECORDS the leave-out",
      "if (line?.file_account) mapping.value = { ...mapping.value, [key]: { left_out: true } }" in pn)
check("a row with no account offers no account to pick",
      '<span v-if="!line.file_account" class="lm-note-inline">no account in the file</span>' in pn)

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
