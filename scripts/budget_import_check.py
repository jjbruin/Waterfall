#!/usr/bin/env python
"""Guardrail: the partner-budget import reads, checks and writes correctly.

Drives the real committed functions against a synthetic workbook built here, plus the
real ISBS snapshot for the account list — so the rules are exercised, not described.

WHAT EACH RULE IS DEFENDING, because several of them replaced a worse version:

  * The ACCOUNT LIST is the deal's own last 12 months, not the 169-account COA. Choosing
    from 169 is why line coding is slow and wrong; Asbury Commons uses 23.

  * SIGN IS PER ACCOUNT, not per 4xxx/5xxx prefix. Berger's 4030 Residential Vacancy and
    4042 Loss to Lease are 4xxx accounts stored POSITIVE (contra-revenue), and 5220 Other
    (Income) Expense is 5xxx stored NEGATIVE. A blanket prefix rule gets all three wrong,
    so the flip default comes from how the account actually behaved for that deal.

  * UNMAPPED LINES DO NOT BLOCK. Spreadsheets carry subtotal rows and skipping them is
    correct. `reconcile()` shows stated-vs-computed revenue, expense and NOI and lets the
    analyst judge — a difference is information, never an error.

  * COMMIT REPLACES, it does not append. A budget is re-imported until final; appending
    would stack every revision. Replacement is scoped to (vcode, the periods in THIS
    file) so a second file for other months does not erase the first.

Run:  python scripts/budget_import_check.py
"""
from __future__ import annotations

import io
import os
import sys
import tempfile

import pandas as pd
from sqlalchemy import create_engine, text

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


def make_workbook() -> bytes:
    """A partner budget in the shape they actually arrive in: months across the top,
    the partner's own line names down the side, revenue shown POSITIVE (their
    convention, the opposite of MRI's), and subtotal rows mixed in."""
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["2026 Operating Budget"] + [None] * 12)
    months = [f"{m}/{[31,28,31,30,31,30,31,31,30,31,30,31][m-1]}/2026" for m in range(1, 13)]
    ws.append(["Line Item"] + months)
    ws.append(["Base Rental Revenue"] + [1000] * 12)
    ws.append(["CAM Recoveries"] + [200] * 12)
    ws.append(["Total Revenue"] + [1200] * 12)          # subtotal — skipped
    ws.append(["Real Estate Taxes"] + [300] * 12)
    ws.append(["Insurance"] + [100] * 12)
    ws.append(["Total Operating Expenses"] + [400] * 12)  # subtotal — skipped
    ws.append(["Net Operating Income"] + [800] * 12)      # subtotal — skipped
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

    from flask_app.services import budget_import_service as svc
    from flask_app.services import budget_import_validate as val

    print("1. Reading the workbook")
    parsed = svc.parse_budget_workbook(make_workbook(), "budget.xlsx")
    labels = [l["label"] for l in parsed["lines"]]
    chk("twelve month columns detected", len(parsed["periods"]) == 12,
        f"got {parsed['periods']}")
    chk("month-ends, not raw dates", parsed["periods"][0] == "2026-01-31",
        f"got {parsed['periods'][0]}")
    chk("every labelled row returned, subtotals included",
        "Base Rental Revenue" in labels and "Total Revenue" in labels, f"got {labels}")
    tot = next(l for l in parsed["lines"] if l["label"] == "Total Revenue")
    chk("subtotal rows are FLAGGED, not dropped", tot["looks_like_total"] is True)
    base = next(l for l in parsed["lines"] if l["label"] == "Base Rental Revenue")
    chk("a line totals its months", base["total"] == 12000, f"got {base['total']}")
    chk("stated totals are picked up",
        parsed["stated_totals"]["revenue"] == 14400
        and parsed["stated_totals"]["noi"] == 9600,
        f"got {parsed['stated_totals']}")

    rows = {l["label"]: l["row"] for l in parsed["lines"]}

    print("\n2. Reconciliation — subtotals skipped, and it still ties")
    mapping = {
        str(rows["Base Rental Revenue"]): {"account": "4010", "flip": True},
        str(rows["CAM Recoveries"]):      {"account": "4090", "flip": True},
        str(rows["Real Estate Taxes"]):   {"account": "5090", "flip": False},
        str(rows["Insurance"]):           {"account": "5110", "flip": False},
    }
    rec = val.reconcile(parsed, mapping)
    got = {r["line"]: r for r in rec["rows"]}
    chk("revenue computed as a positive magnitude", got["revenue"]["computed"] == 14400,
        f"got {got['revenue']['computed']}")
    chk("expense computed", got["expense"]["computed"] == 4800,
        f"got {got['expense']['computed']}")
    chk("NOI computed", got["noi"]["computed"] == 9600, f"got {got['noi']['computed']}")
    chk("ties to the spreadsheet with all four lines mapped",
        all(r["difference"] == 0 for r in rec["rows"]),
        f"got {[(r['line'], r['difference']) for r in rec['rows']]}")

    print("\n3. A MISSED line shows up as a difference, and does not block")
    partial = {k: v for k, v in mapping.items() if k != str(rows["Insurance"])}
    rec2 = val.reconcile(parsed, partial)
    d = {r["line"]: r["difference"] for r in rec2["rows"]}
    chk("expense is 1,200 short", d["expense"] == -1200, f"got {d['expense']}")
    chk("NOI is 1,200 over", d["noi"] == 1200, f"got {d['noi']}")

    print("\n4. Validation — blocking is reserved for genuinely wrong")
    isbs = pd.DataFrame(columns=["vcode", "dtEntry", "vSource", "vAccount",
                                 "mAmount", "vDescription"])
    v = val.validate(parsed, mapping, "P0000004", isbs)
    chk("a clean mapping can import", v["can_import"] is True, f"{v['blocking']}")
    v0 = val.validate(parsed, {}, "P0000004", isbs)
    chk("mapping nothing BLOCKS", v0["can_import"] is False
        and v0["blocking"][0]["code"] == "no_lines")
    dupe = dict(mapping)
    dupe[str(rows["CAM Recoveries"])] = {"account": "4010", "flip": True}
    vd = val.validate(parsed, dupe, "P0000004", isbs)
    chk("the same account twice in a month BLOCKS",
        any(b["code"] == "duplicate_account_month" for b in vd["blocking"]),
        f"{vd['blocking']}")
    chk("an unmapped subtotal row does NOT block", v["can_import"] is True)

    print("\n5. Warnings against the deal's real history")
    from flask_app import create_app
    from flask_app.services import data_service as ds
    app = create_app()
    with app.app_context():
        data = ds.load_all(db_path="waterfall.db")
    real = data["isbs_raw"]

    ch = svc.account_choices("P0000004", real)
    chk("the account list is the deal's own recent accounts, not the whole COA",
        0 < len(ch) < 60, f"got {len(ch)}")
    chk("each choice carries a description", all(c["description"] for c in ch[:5]))
    r4010 = next(c for c in ch if c["account"] == "4010")
    chk("4010 Rental Income is stored NEGATIVE for this deal",
        r4010["mri_sign"] == -1, f"got {r4010['mri_sign']}")

    vr = val.validate(parsed, mapping, "P0000004", real)
    codes = {w["code"] for w in vr["warnings"]}
    chk("a tiny budget vs a real year is flagged on magnitude", "magnitude" in codes,
        f"got {sorted(codes)}")
    chk("accounts used last year but not budgeted are named",
        "account_not_budgeted" in codes, f"got {sorted(codes)}")
    chk("and none of that blocks the import", vr["can_import"] is True)

    print("\n6. Category choices — the rows the comparison actually shows")
    cats = svc.category_choices("P0000004", real)
    chk("~27 categories, not 169 accounts", 20 <= len(cats) <= 40, f"got {len(cats)}")
    chk("categories the deal uses come first", cats[0]["used_by_deal"] is True)
    rent = next(c for c in cats if c["category"] == "Rental Income")
    chk("Rental Income defaults to 4010, the account this deal used most",
        rent["default_account"] == "4010", f"got {rent['default_account']}")
    chk("its sign default is NEGATIVE, from the deal's own history",
        next(a for a in rent["accounts"] if a["account"] == "4010")["mri_sign"] == -1)
    unused = next(c for c in cats if not c["used_by_deal"])
    chk("a category the deal has never used is still offered, with a default",
        unused["default_account"] is not None, f"{unused['category']}")

    print("\n7. A category and account that disagree BLOCK")
    bad = dict(mapping)
    bad[str(rows["Base Rental Revenue"])] = {"category": "Rental Income",
                                             "account": "5090", "flip": True}
    vb = val.validate(parsed, bad, "P0000004", real)
    chk("5090 under Rental Income is refused",
        any(b["code"] == "account_not_in_category" for b in vb["blocking"]),
        f"{vb['blocking']}")
    ok = dict(mapping)
    ok[str(rows["Base Rental Revenue"])] = {"category": "Rental Income",
                                            "account": "4012", "flip": True}
    vo = val.validate(parsed, ok, "P0000004", real)
    chk("a sibling account WITHIN the category is fine",
        not any(b["code"] == "account_not_in_category" for b in vo["blocking"]))

    print("\n8. A wrong flip is caught by sign, not by prefix")
    wrong = dict(mapping)
    wrong[str(rows["Base Rental Revenue"])] = {"account": "4010", "flip": False}
    vw = val.validate(parsed, wrong, "P0000004", real)
    chk("revenue left positive is flagged opposite-sign",
        any(w["code"] == "sign_opposite_prior" for w in vw["warnings"]),
        f"{sorted({w['code'] for w in vw['warnings']})}")

    print("\n9. Commit REPLACES the same months and leaves other months alone")
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    eng = create_engine(f"sqlite:///{path}")
    res = val.commit(eng, "P0000004", parsed, mapping, "tester")
    chk("rows written = 4 lines x 12 months", res["rows_written"] == 48,
        f"got {res['rows_written']}")
    chk("nothing replaced on a first import", res["rows_replaced"] == 0)
    with eng.connect() as c:
        n = c.execute(text(f'SELECT COUNT(*) FROM {svc.SUPPLEMENT_TABLE}')).scalar()
        signed = c.execute(text(
            f'SELECT "mAmount" FROM {svc.SUPPLEMENT_TABLE} '
            f'WHERE "vAccount" = :a AND "dtEntry" = :d'),
            {"a": "4010", "d": "2026-01-31"}).scalar()
    chk("48 rows in the supplement", n == 48, f"got {n}")
    chk("the flip was applied — 4010 stored NEGATIVE", signed == -1000, f"got {signed}")

    res2 = val.commit(eng, "P0000004", parsed, mapping, "tester")
    with eng.connect() as c:
        n2 = c.execute(text(f'SELECT COUNT(*) FROM {svc.SUPPLEMENT_TABLE}')).scalar()
    chk("a re-import replaces rather than stacks", n2 == 48, f"got {n2}")
    chk("and reports what it replaced", res2["rows_replaced"] == 48,
        f"got {res2['rows_replaced']}")

    # A second file covering DIFFERENT months must not erase the first. 2027 is used
    # deliberately: the workbook above covers Jan-Dec 2026, so any month inside that
    # range is SUPPOSED to be replaced — an earlier version of this check used April and
    # read the correct replacement (48 - 4 + 1 = 45) as a failure.
    other = dict(parsed)
    other["lines"] = [dict(l, amounts={"2027-01-31": 500}, total=500, months=1)
                      for l in parsed["lines"] if l["label"] == "Base Rental Revenue"]
    val.commit(eng, "P0000004", other,
               {str(other["lines"][0]["row"]): {"account": "4010", "flip": True}},
               "tester")
    with eng.connect() as c:
        n3 = c.execute(text(f'SELECT COUNT(*) FROM {svc.SUPPLEMENT_TABLE}')).scalar()
    chk("a second file for months OUTSIDE the range ADDS, it does not erase", n3 == 49, f"got {n3}")

    try:
        os.unlink(path)
    except Exception:
        pass

    print(f"\nPASS={PASS} FAIL={FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
