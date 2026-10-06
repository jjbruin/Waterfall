#!/usr/bin/env python
"""Guardrail: the appraiser's Argus cash flow is loaded ONCE and mapped like the budget.

Asset management, Sep 28 2026, three asks:

  1. ONE UPLOAD. The Argus file was uploaded on Assumptions & Documents (read by
     `argus_parser`, accounts guessed from keywords) and AGAIN on Budget Review to see
     its mapping (read by the budget parser), and the mapping was written back BY LABEL
     onto the first import. Now the Budget Review panel's reading IS the cash flow.
  2. MAP BY THE ACCOUNT NUMBER IN THE FILE, as the budget does -- no keyword guesses --
     and read that account wherever it sits beside the description.
  3. A LINE READ AS A SUBTOTAL CAN BE OVERTURNED ("Total Recoveries" was a real line).

Plus the defect found building it: the "Partnership costs" proposal tick box never left
the browser, so from v502 it wrote nothing. It is now a real line when ticked.

Pure fixtures, a temporary SQLite database, no network.
Run:  python scripts/argus_single_load_check.py
"""
import io
import os
import sys
import tempfile

import pandas as pd
from sqlalchemy import create_engine, text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PASS, FAIL, SKIP = [], [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  ok   " if cond else "  FAIL ") + name + (("  -- " + str(detail)) if detail and not cond else ""))


def skip(name, why):
    SKIP.append(name)
    print(f"  skip {name} -- {why}")


import logging                                                          # noqa: E402
logging.disable(logging.WARNING)
from flask_app.services import budget_import_service as B              # noqa: E402
from flask_app.services import line_mapping_service as LM              # noqa: E402
from flask_app.services import valuation_service as VS                 # noqa: E402

MONTHS = [f"2027-{m:02d}-01" for m in range(1, 13)]
LINES = [  # label, account stated in the file, monthly amount
    ("Potential Base Rent", "4012", 1000.0),     # keywords would say 4010; the FILE says 4012
    ("Total Recoveries", "4090", 200.0),         # a real line the app reads as a subtotal
    ("Real Estate Taxes", "5090", 300.0),
    ("Utilities", "5050", 80.0),
    ("Total Operating Expenses", None, 380.0),   # a real subtotal
    ("Net Operating Income", None, 820.0),       # a real subtotal
]


def xlsx(rows):
    buf = io.BytesIO()
    pd.DataFrame(rows).to_excel(buf, index=False, header=False)
    return buf.getvalue()


def desc_then_acct():
    return xlsx([[None, None] + MONTHS] + [[d, a] + [v] * 12 for d, a, v in LINES])


print("\n1. The account is read wherever it sits beside the description")
shapes = {
    "description | account": desc_then_acct(),
    "description | 'Account' header": xlsx([["Description", "Account"] + MONTHS]
                                           + [[d, a] + [v] * 12 for d, a, v in LINES]),
    "account | description": xlsx([[None, None] + MONTHS] + [[a, d] + [v] * 12 for d, a, v in LINES]),
    "description | account as a number": xlsx([[None, None] + MONTHS] + [
        [d, int(a) if a else None] + [v] * 12 for d, a, v in LINES]),
}
for name, wb in shapes.items():
    p = B.parse_budget_workbook(wb, "f.xlsx")
    got = {l["label"]: l.get("stated_account") for l in p["lines"]}
    check(f"{name}: every stated account is read",
          all(got.get(d) == a for d, a, _ in LINES if a), got)
rollup = xlsx([[None, None] + MONTHS] + [[d, str(int(v * 12))] + [v] * 12 for d, a, v in LINES])
p = B.parse_budget_workbook(rollup, "rollup.xlsx")
check("a column of annual TOTALS beside the labels is NOT read as accounts",
      all(l.get("stated_account") is None for l in p["lines"]),
      [(l["label"], l.get("stated_account")) for l in p["lines"]])

# ── a temporary database with one valuation record ─────────────────────────
tmp = os.path.join(tempfile.mkdtemp(), "wf.db")
eng = create_engine("sqlite:///" + tmp)
VS.ensure_valuation_tables(eng)
with eng.begin() as c:
    c.execute(text("""CREATE TABLE argus_imports (id INTEGER PRIMARY KEY AUTOINCREMENT,
        vcode TEXT NOT NULL, import_label TEXT, import_type TEXT, original_filename TEXT,
        file_hash TEXT, is_active INTEGER DEFAULT 1, imported_by TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"""))
    c.execute(text("""CREATE TABLE argus_cashflows (id INTEGER PRIMARY KEY AUTOINCREMENT,
        import_id INTEGER NOT NULL, vcode TEXT NOT NULL, period_date TEXT, line_item TEXT,
        coa_account INTEGER, amount REAL, amount_norm REAL, category TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"""))
    c.execute(text("INSERT INTO valuation_cycles (id, year, as_of_date) VALUES (1, 2026, '2026-12-31')"))
    c.execute(text("INSERT INTO valuation_cycles (id, year, as_of_date) VALUES (2, 2025, '2025-12-31')"))
    c.execute(text("INSERT INTO valuation_records (id, cycle_id, vcode, status) VALUES (1, 1, 'P0000901', 'draft')"))
data = {"isbs_raw": pd.DataFrame(), "occupancy_raw": None}

print("\n2. Argus is mapped like the budget: the file's account, no keyword guesses")
pa = LM.parse(eng, 1, "argus", desc_then_acct(), "argus.xlsx", data)
by_label = {l["label"]: l for l in pa["lines"]}
sug = {by_label_l["label"]: pa["suggested"].get(str(by_label_l["row"]))
       for by_label_l in pa["lines"]}
check("no suggestion is a keyword guess",
      not any((m or {}).get("from_keywords") for m in pa["suggested"].values()), pa["suggested"])
check("'Potential Base Rent' takes the FILE's 4012, not the keyword rule's 4010",
      (sug["Potential Base Rent"] or {}).get("account") == "4012", sug["Potential Base Rent"])
check("...and is marked as read from the file",
      (sug["Potential Base Rent"] or {}).get("from_file") is True, sug["Potential Base Rent"])
check("Utilities -> 5050 from the file, category from our chart",
      (sug["Utilities"] or {}).get("account") == "5050"
      and (sug["Utilities"] or {}).get("category") == B.category_for_account("5050"), sug["Utilities"])
nokw = LM.parse(eng, 1, "argus", xlsx([[None] + MONTHS, ["Potential Base Rent"] + [1000] * 12]),
                "nokw.xlsx", data)
check("a file stating NO account gets NO pre-fill, not a keyword guess",
      nokw["suggested_count"] == 0, nokw["suggested"])

print("\n3. A line read as a subtotal is not final")
tr = by_label["Total Recoveries"]
check("'Total Recoveries' is read as a subtotal", tr["looks_like_total"] is True)
check("...so it is NOT pre-filled", sug["Total Recoveries"] is None, sug["Total Recoveries"])
check("...but it CARRIES the account it states, so 'not a subtotal' can pre-fill it",
      tr.get("stated_account") == "4090", tr)
mapping = {k: dict(v) for k, v in pa["suggested"].items()}
mapping[str(tr["row"])] = {"account": "4090", "not_subtotal": True}
ck = LM.check(eng, 1, "argus", pa, mapping, data)
check("a subtotal the analyst maps is accepted by the gate", ck["can_import"], ck["blocking"])

print("\n4. One upload: applying the mapping IS the Valuation cash flow")
before = VS.get_budget_review(eng, 1, data)
check("before: no Valuation column", not before["compare"]["available"], before["compare"])
res = LM.commit(eng, 1, "argus", pa, mapping, "guardrail", data)
with eng.connect() as c:
    linked = c.execute(text("SELECT argus_import_id FROM valuation_records WHERE id = 1")).scalar()
    n_imports = c.execute(text("SELECT COUNT(*) FROM argus_imports")).scalar()
    rows = c.execute(text("SELECT coa_account, SUM(amount_norm), COUNT(*) FROM argus_cashflows "
                          "WHERE import_id = :i GROUP BY coa_account"), {"i": linked}).fetchall()
by_acct = {int(r[0]): (float(r[1]), int(r[2])) for r in rows}
check("with no import on the record, applying CREATES one and links it",
      linked and n_imports == 1 and res["import_id"] == linked, (linked, n_imports, res))
check("exactly the mapped lines are written, 12 months each (4 lines x 12 = 48)",
      sum(v[1] for v in by_acct.values()) == 48, by_acct)
check("revenue is positive and expense negative -- signs from the ACCOUNT",
      by_acct[4012][0] == 12000 and by_acct[5090][0] == -3600, by_acct)
check("the overturned subtotal is written under its account (4090)",
      4090 in by_acct and by_acct[4090][0] == 2400, by_acct)
after = VS.get_budget_review(eng, 1, data)
rent_row = next(r for r in after["rows"] if r["account"] == B.category_for_account("4012"))
check("the Valuation Yr 1 column shows it straight away (12 x 1,000)",
      after["compare"]["available"] and abs(rent_row["valuation"] - 12000) < 0.01, rent_row)

print("\n5. Applying again REPLACES, it does not stack or orphan")
mapping2 = {k: v for k, v in mapping.items() if v.get("account") != "5050"}
LM.commit(eng, 1, "argus", pa, mapping2, "guardrail", data)
with eng.connect() as c:
    linked2 = c.execute(text("SELECT argus_import_id FROM valuation_records WHERE id = 1")).scalar()
    n_imports2 = c.execute(text("SELECT COUNT(*) FROM argus_imports")).scalar()
    n_rows2 = c.execute(text("SELECT COUNT(*) FROM argus_cashflows WHERE import_id = :i"),
                        {"i": linked2}).scalar()
    n_all = c.execute(text("SELECT COUNT(*) FROM argus_cashflows")).scalar()
check("same import, replaced in place", linked2 == linked and n_imports2 == 1, (linked2, n_imports2))
check("the un-mapped line is gone and nothing stacked (3 lines x 12 = 36)",
      n_rows2 == 36 and n_all == 36, (n_rows2, n_all))

print("\n6. An import another record links is never rewritten")
with eng.begin() as c:
    c.execute(text("INSERT INTO valuation_records (id, cycle_id, vcode, status, argus_import_id) "
                   "VALUES (2, 2, 'P0000901', 'draft', :i)"), {"i": linked})
LM.commit(eng, 1, "argus", pa, mapping, "guardrail", data)
with eng.connect() as c:
    r1 = c.execute(text("SELECT argus_import_id FROM valuation_records WHERE id = 1")).scalar()
    r2 = c.execute(text("SELECT argus_import_id FROM valuation_records WHERE id = 2")).scalar()
    shared_rows = c.execute(text("SELECT COUNT(*) FROM argus_cashflows WHERE import_id = :i"),
                            {"i": r2}).scalar()
check("record 1 gets a NEW import", r1 != r2 and r2 == linked, (r1, r2))
check("the other cycle's import is untouched (still 36 rows)", shared_rows == 36, shared_rows)

print("\n7. The Partnership costs proposal is written when ticked -- and only then")
offered = pa["proposed_lines"]
check("the partnership line is offered for Argus", any(p["account"] == "5130" for p in offered), offered)
ticked = {**pa, "accepted_proposals": [{"account": "5130", "amount": 20000.0,
                                        "label": "Partnership costs (house default)"}]}
ck = LM.check(eng, 1, "argus", ticked, mapping, data)
check("the check SEES a ticked proposal (it never did)",
      ck["mapped_count"] == len([m for m in mapping.values() if m.get("account")]) + 1,
      ck["mapped_count"])
LM.commit(eng, 1, "argus", ticked, mapping, "guardrail", data)
d = LM.get_draft(eng, 1, "argus")
check("the draft keeps the TICK on the parsed file, and no synthetic line in the mapping",
      d and [p["account"] for p in d["parsed"].get("accepted_proposals") or []] == ["5130"]
      and not any(int(k) < 0 for k in d["mapping"]),
      d and (d["parsed"].get("accepted_proposals"), sorted(d["mapping"])))
with eng.connect() as c:
    imp = c.execute(text("SELECT argus_import_id FROM valuation_records WHERE id = 1")).scalar()
    p5130 = c.execute(text("SELECT SUM(amount_norm), COUNT(*) FROM argus_cashflows "
                           "WHERE import_id = :i AND coa_account = 5130"), {"i": imp}).fetchone()
check("ticked: 5130 is written, 20,000 spread over the file's 12 months, as a cost",
      p5130[1] == 12 and abs(float(p5130[0]) + 20000) < 0.01, tuple(p5130))
LM.commit(eng, 1, "argus", pa, mapping, "guardrail", data)
with eng.connect() as c:
    none5130 = c.execute(text("SELECT COUNT(*) FROM argus_cashflows WHERE import_id = :i "
                              "AND coa_account = 5130"), {"i": imp}).scalar()
check("unticked: nothing at 5130", none5130 == 0, none5130)
rogue = {**pa, "accepted_proposals": [{"account": "4010", "amount": 999999.0}]}
p2, m2 = LM.with_accepted_proposals(rogue, mapping, "argus")
check("an account the screen never offered is NOT added under a proposal's name",
      len(p2["lines"]) == len(pa["lines"]), len(p2["lines"]))
p3, _ = LM.with_accepted_proposals(ticked, mapping, "budget")
check("the budget source ignores proposals (they are offered on Argus only)",
      len(p3["lines"]) == len(pa["lines"]))

print("\n8. An approved valuation refuses")
with eng.begin() as c:
    c.execute(text("UPDATE valuation_records SET status = 'approved' WHERE id = 1"))
try:
    LM.commit(eng, 1, "argus", pa, mapping, "guardrail", data)
    check("applying to an approved record is refused", False)
except ValueError:
    check("applying to an approved record is refused", True)

print("\n9. There is ONE way in")
check("valuation_service.import_argus is gone", not hasattr(VS, "import_argus"))
try:
    from flask_app import create_app
    app = create_app()
    rules = {str(r) for r in app.url_map.iter_rules()}
    check("POST /records/<id>/argus is gone",
          "/api/valuations/records/<int:record_id>/argus" not in rules)
    check("...and the mapping commit that replaces it is registered",
          "/api/valuations/records/<int:record_id>/mapping/commit" in rules)
except Exception as e:                                                      # noqa: BLE001
    skip("route registration", f"app would not start here ({e})")

print("\n10. The screens say so")
view = os.path.join(ROOT, "vue_app", "src", "views", "ValuationsView.vue")
panel = os.path.join(ROOT, "vue_app", "src", "components", "common", "LineMappingPanel.vue")
if not (os.path.exists(view) and os.path.exists(panel)):
    skip("screen checks", "Vue source is not shipped in the image")
else:
    v = open(view, encoding="utf-8").read()
    pn = open(panel, encoding="utf-8").read()
    check("Assumptions & Documents has no Argus upload any more",
          "onArgusUpload" not in v and "Import Argus Export" not in v)
    check("...and says where the cash flow is loaded now", "openArgusLoader" in v)
    check("the tab is 'Load Valuation Cash Flow'",
          "Load Valuation Cash Flow</button>" in v and "Review Argus Coding" not in v)
    check("the panel shows no keyword-guess tag", "from_keywords" not in pn)
    check("the subtotal reading can be overturned",
          "setNotSubtotal" in pn and "not a subtotal" in pn and "not_subtotal" in pn)
    # Jack, Oct 6 2026: "Add the flip sign checkbox. The budget upload has it and the
    # Argus load doesn't." It writes `reverse` for Argus, never `flip`.
    check("the flip box is offered for Argus too, writing its own key",
          "v-if=\"source !== 'argus'\" class=\"ctr\"" not in pn
          and "source === 'argus' ? 'reverse' : 'flip'" in pn
          and "[flipKey]" in pn)
    check("a ticked proposal rides on the parsed file", "accepted_proposals" in pn)

print("\n%d passed, %d failed, %d skipped" % (len(PASS), len(FAIL), len(SKIP)))
if FAIL:
    print("FAILED:")
    for f in FAIL:
        print("   " + f)
sys.exit(1 if FAIL else 0)
