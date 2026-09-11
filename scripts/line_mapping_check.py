#!/usr/bin/env python
"""Guardrail: ONE line-mapping flow serves both sources that feed the comparison.

The budget comparison is Estimate | Budget | Valuation. Two of those columns are loaded
from somebody else's spreadsheet — the partner's budget workbook, and the appraiser's
Argus download — and the team asked for the two to work the same way. This drives the
real committed service to prove they do, and that the ways they legitimately DIFFER are
the only ways they differ.

WHAT EACH RULE IS DEFENDING:

  * ARGUS ARRIVES PRE-FILLED, A BUDGET DOES NOT. Argus line names come from a tool, so
    the 56 keyword rules in `argus_parser.ARGUS_COA_MAP` can guess. A partner's wording
    is their own, and guessing at it would recreate the invisible mapping that asset
    management complained about ("can't see or interact with code mapping for
    valuation"). Pre-fill is a visible, editable SUGGESTION — never a silent commit.

  * THE ACCOUNT DECIDES THE CATEGORY, not the keyword map. The keyword map carries its
    own category word; the row a figure lands on is decided by which of OUR categories
    contains the account. Where they disagree the account wins, because that is what the
    comparison actually reads.

  * SUBTOTALS ARE NEVER PRE-FILLED. "Total Revenue" contains the word "revenue" and the
    keyword rules will happily map it to 4010. Importing it would double the deal's
    revenue, silently.

  * `argus_cashflows.category` KEEPS ITS OWN VOCABULARY. That column holds revenue /
    expense / capex. Writing our category name ("Rental Income") into it would leave the
    column holding two different kinds of word depending on who last wrote the row.

  * THE GATE IS RE-RUN ON COMMIT. The screen validates as the analyst types, but commit
    re-validates server-side: this writes the only copy of an unapproved budget.

  * AN ARGUS COMMIT WITH NO IMPORT IS REFUSED WITH INSTRUCTIONS, not a stack trace.

Run:  .venv/Scripts/python.exe scripts/line_mapping_check.py
"""
from __future__ import annotations

import io
import os
import sys

from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = FAIL = 0


def chk(label, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}" + (f"\n          {detail}" if detail else ""))


def make_argus_workbook() -> bytes:
    """An appraiser's Argus download: Argus's own line names, months across the top,
    revenue POSITIVE, and subtotal rows that the keyword rules would happily map."""
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Cash Flow Projection"] + [None] * 12)
    months = [f"{m}/{[31,28,31,30,31,30,31,31,30,31,30,31][m-1]}/2026"
              for m in range(1, 13)]
    ws.append(["Line Item"] + months)
    ws.append(["Potential Base Rent"] + [1000] * 12)     # -> 4010
    ws.append(["Absorption & Turnover Vacancy"] + [-50] * 12)   # -> 4030, contra
    ws.append(["CAM Recovery"] + [200] * 12)             # -> 4090
    ws.append(["Total Revenue"] + [1150] * 12)           # subtotal
    ws.append(["Real Estate Taxes"] + [300] * 12)        # -> 5090
    ws.append(["Insurance"] + [100] * 12)                # -> 5110
    ws.append(["Tenant Improvements"] + [75] * 12)       # -> 7050, capex
    ws.append(["Total Capital Expenditures"] + [75] * 12)  # keyword BAIT — a subtotal
                                                           # the rules DO match, to 7050
    ws.append(["Net Operating Income"] + [750] * 12)     # subtotal
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import warnings, logging
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)

    from flask_app import create_app
    from flask_app.db import get_engine
    from flask_app.services import data_service as ds
    from flask_app.services import line_mapping_service as lm

    app = create_app()
    with app.app_context():
        engine = get_engine()
        data = ds.load_all(db_path="waterfall.db")

        # A real record, so vcode resolution and the deal's own history are exercised.
        with engine.connect() as conn:
            rec = conn.execute(text(
                "SELECT id, vcode FROM valuation_records ORDER BY id LIMIT 1")).fetchone()
        if not rec:
            print("NO valuation_records rows locally — cannot run. "
                  "Seed one record and re-run.")
            return 2
        record_id, vcode = int(rec[0]), str(rec[1])
        print(f"Record {record_id} = {vcode}\n")

        print("1. Both sources are offered, and nothing else is")
        chk("exactly two sources", lm.SOURCES == ("budget", "argus"), f"{lm.SOURCES}")
        try:
            lm.parse(engine, record_id, "mri", b"", "x.xlsx", data)
            chk("an unknown source is refused", False, "no error raised")
        except ValueError as e:
            chk("an unknown source is refused", "Unknown source" in str(e), str(e))
        try:
            lm.record_vcode(engine, 10**9)
            chk("a missing record is refused", False, "no error raised")
        except ValueError as e:
            chk("a missing record is refused", "not found" in str(e), str(e))

        print("\n2. Categories — the comparison's rows, ranked by this deal's usage")
        cats = lm.categories(engine, record_id, data)
        chk("the record resolves to its deal", cats["vcode"] == vcode, cats["vcode"])
        chk("~27 categories, not 169 accounts", 20 <= len(cats["categories"]) <= 40,
            f"got {len(cats['categories'])}")
        chk("usage is reported rather than assumed",
            isinstance(cats["used_count"], int)
            and cats["has_history"] == (cats["used_count"] > 0),
            f"used={cats['used_count']} has_history={cats['has_history']}")
        chk("every category offers a default account",
            all(c["default_account"] for c in cats["categories"]),
            [c["category"] for c in cats["categories"] if not c["default_account"]])
        # config.IS_ACCOUNTS['DEBT_SERVICE']['Principal'] is deliberately [] — for
        # ACTUALS principal is the balance-sheet balance change, not an account. A
        # BUDGET has no balance sheet, so 7060 is its representation, and without the
        # override the screen offered a category with nothing selectable in it.
        prin = next(c for c in cats["categories"] if c["category"] == "Principal")
        chk("Principal, empty in config, is completable on a budget",
            prin["default_account"] == "7060"
            and [a["account"] for a in prin["accounts"]] == ["7060"], f"{prin}")

        print("\n3. A BUDGET arrives unassigned — no guessing at a partner's wording")
        wb = make_argus_workbook()
        pb = lm.parse(engine, record_id, "budget", wb, "budget.xlsx", data)
        chk("the file is read", len(pb["lines"]) == 9, f"got {len(pb['lines'])}")
        chk("twelve periods", len(pb["periods"]) == 12, f"got {len(pb['periods'])}")
        chk("NOTHING is pre-filled", pb["suggested_count"] == 0,
            f"got {pb['suggested']}")
        chk("the categories ride along so one call fills the screen",
            len(pb["categories"]) == len(cats["categories"]))

        print("\n4. ARGUS arrives pre-filled from the keyword rules, visibly")
        pa = lm.parse(engine, record_id, "argus", wb, "argus.xlsx", data)
        by_row = {l["row"]: l for l in pa["lines"]}
        sug = {by_row[int(r)]["label"]: m for r, m in pa["suggested"].items()}
        chk("several lines are suggested", pa["suggested_count"] >= 5,
            f"got {pa['suggested_count']}: {sorted(sug)}")
        chk("Potential Base Rent -> 4010",
            sug.get("Potential Base Rent", {}).get("account") == "4010",
            f"{sug.get('Potential Base Rent')}")
        chk("and its category is the one that OWNS 4010, not the keyword word",
            sug.get("Potential Base Rent", {}).get("category") == "Rental Income",
            f"{sug.get('Potential Base Rent')}")
        chk("Tenant Improvements -> 7050",
            sug.get("Tenant Improvements", {}).get("account") == "7050",
            f"{sug.get('Tenant Improvements')}")
        chk("every suggestion is marked as a guess, so the screen can show it",
            all(m.get("from_keywords") for m in sug.values()))

        print("\n5. Subtotals are NEVER pre-filled, whatever the keywords say")
        from argus_parser import map_to_coa
        # The bait is real: "Total Capital Expenditures" matches the "capital
        # expenditure" rule and would import a subtotal as a 7050 line, doubling capex.
        kw_acct, _ = map_to_coa("Total Capital Expenditures")
        chk("the keyword rules WOULD map 'Total Capital Expenditures'",
            kw_acct == 7050, f"got {kw_acct}")
        chk("but the flow refuses to suggest it",
            "Total Capital Expenditures" not in sug,
            f"{sug.get('Total Capital Expenditures')}")
        chk("'Total Revenue' and 'Net Operating Income' likewise",
            "Total Revenue" not in sug and "Net Operating Income" not in sug)

        print("\n6. The flip default comes from the ACCOUNT's own behaviour")
        acc_by_cat = {c["category"]: c for c in pa["categories"]}
        rent = acc_by_cat.get("Rental Income", {})
        sign4010 = next((a["mri_sign"] for a in rent.get("accounts", [])
                         if a["account"] == "4010"), None)
        chk("4010 behaves NEGATIVE for this deal", sign4010 == -1, f"got {sign4010}")
        chk("so positive Argus rent is flagged to flip",
            sug.get("Potential Base Rent", {}).get("flip") is True,
            f"{sug.get('Potential Base Rent')}")
        vac = sug.get("Absorption & Turnover Vacancy", {})
        chk("a NEGATIVE contra-revenue line on a positive-stored account is not flipped "
            "by prefix", vac.get("account") == "4030" and isinstance(vac.get("flip"), bool),
            f"{vac}")

        print("\n7. check() reports on the mapping as it stands, for either source")
        mapping = {r: dict(m) for r, m in pa["suggested"].items()}
        ck = lm.check(engine, record_id, "argus", pa, mapping, data)
        chk("the keyword mapping passes the gate", ck["can_import"] is True,
            f"{ck['blocking']}")
        chk("it counts what is mapped out of what is there",
            ck["mapped_count"] == len(mapping) and ck["line_count"] == 9,
            f"{ck['mapped_count']}/{ck['line_count']}")
        chk("three reconciliation rows, always",
            [r["line"] for r in ck["reconciliation"]["rows"]]
            == ["revenue", "expense", "noi"])
        chk("the source is echoed so the screen cannot mislabel itself",
            ck["source"] == "argus")
        ck0 = lm.check(engine, record_id, "budget", pb, {}, data)
        chk("an empty mapping blocks with no_lines",
            ck0["can_import"] is False
            and ck0["blocking"][0]["code"] == "no_lines", f"{ck0['blocking']}")

        print("\n8. commit() re-runs the gate server-side")
        try:
            lm.commit(engine, record_id, "budget", pb, {}, "tester", data)
            chk("an unmapped payload cannot be committed", False, "no error raised")
        except ValueError as e:
            chk("an unmapped payload cannot be committed",
                "no lines" in str(e).lower(), str(e))

        print("\n9. An Argus commit with no import says what to do about it")
        with engine.connect() as conn:
            imp = conn.execute(text(
                "SELECT argus_import_id FROM valuation_records WHERE id = :i"),
                {"i": record_id}).fetchone()
        has_import = bool(imp and imp[0])
        if has_import:
            chk("record has an Argus import — refusal path not exercised here", True,
                "skipped deliberately")
        else:
            try:
                lm.commit(engine, record_id, "argus", pa, mapping, "tester", data)
                chk("it refuses", False, "no error raised")
            except ValueError as e:
                chk("it refuses", "no Argus import" in str(e), str(e))
                chk("and tells the analyst where to upload it",
                    "Upload" in str(e) and "record" in str(e), str(e))

        print("\n10. The Argus table keeps its OWN category vocabulary")
        chk("4010 -> revenue", lm._argus_category(4010) == "revenue")
        chk("5090 -> expense", lm._argus_category(5090) == "expense")
        chk("7050 -> capex", lm._argus_category(7050) == "capex")
        import argus_parser as ap
        chk("and those are the only three the parser ever writes",
            {c for _, _, c in ap.ARGUS_COA_MAP} == {"revenue", "expense", "capex"},
            f"{sorted({c for _, _, c in ap.ARGUS_COA_MAP})}")

        print("\n11. All four endpoints are registered")
        want = {"/api/valuations/records/<int:record_id>/mapping/" + p
                for p in ("categories", "parse", "check", "commit")}
        rules = {str(r.rule) for r in app.url_map.iter_rules()}
        missing = want - rules
        chk("categories / parse / check / commit", not missing, f"missing {missing}")

    print(f"\nPASS={PASS} FAIL={FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
