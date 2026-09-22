"""Guardrail: the budget import reads the file it was given, and writes to the table
it actually has.

Jack (asset management), Sep 22 2026, after rebuilding the same Evergreen Plaza budget
EIGHT times to get it through this screen:

    "It only ingests column A ... That's why I built the helper column joining the
     account number and description. We shouldn't have to do that."
    "Auto-map should take the account number off the upload, go to our global mapping,
     and match it. Account 4090 comes in, the app looks up to the global mapping, says
     4090 is CAM, so it maps to CAM. Done."
    "Right now it's two separate steps and they fight each other."
    "Fifteen of their repair lines all belong in one account on our side. The app
     treats every line after the first as a conflict and won't let us submit."
    And then, on the version that finally mapped: the Budget column stayed empty, with
    `column "vcode" does not exist ... Perhaps you meant "isbs_budget_is_supplements.vCode"`.

FIVE RULES, each asserted in BOTH directions, because each has an opposite failure that
would satisfy a one-sided check:

  1. COLUMN NAMES ARE READ FROM THE TABLE. Quoting is not enough -- `"vcode"` is
     case-SENSITIVE on PostgreSQL and case-INSENSITIVE on SQLite, so a hardcoded
     spelling passes every local test and fails every production import. It is not
     enough to assert the resolver works on a `vCode` table: it must also leave a
     `vcode` table alone, or "always use vCode" would pass.

  2. THE LABEL IS THE LINE'S NAME, NOT ITS ACCOUNT NUMBER. And the opposite: a sheet
     whose label column is genuinely text must not be shifted off it.

  3. THE LABEL AND THE AMOUNTS COME FROM THE SAME BLOCK. A sheet carrying two tables
     side by side reads its names from one and its figures from the other, and NOTHING
     about the result looks wrong. And the opposite: a sheet with a sub-description
     beside its labels must NOT be re-based onto the sub-description, or the fix costs
     more files than it saves.

  4. THE ACCOUNT DECIDES THE CATEGORY. Both ways: a category disagreeing with the
     account is corrected rather than blocked, and an account we do not carry at all
     IS blocked -- a check that only ever corrects would accept anything.

  5. MANY LINES MAY SHARE AN ACCOUNT. They combine, and the combined figure must be
     right; merely not-blocking would be satisfied by dropping every line after the
     first, which is worse than the refusal it replaced.

Run:  python scripts/budget_import_mapping_check.py
"""
import io
import os
import sys
import tempfile

import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask_app.services import budget_import_service as B          # noqa: E402
from flask_app.services import budget_import_validate as V         # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  ok   " if cond else "  FAIL ") + name + (("  -- " + str(detail)) if detail and not cond else ""))


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures: the three real shapes, built here rather than shipped
# ──────────────────────────────────────────────────────────────────────────────
MONTHS = [f"2027-{m:02d}-01" for m in range(1, 13)]


def _xlsx(rows):
    buf = io.BytesIO()
    pd.DataFrame(rows).to_excel(buf, index=False, header=False)
    return buf.getvalue()


def shape_account_and_description():
    """Jack's original: account in A, the line's name in B, months from E."""
    head = [None, None, None, None] + MONTHS
    out = [head]
    for acct, label, amt in (("4010", "Base Rent", 100.0),
                             ("4090", "CAM Reimb", 10.0),
                             ("4090", "Water Reimb", 5.0),
                             ("5060", "General Repairs", 7.0),
                             ("5060", "Electrical Repairs", 3.0)):
        out.append([acct, label, None, None] + [amt] * 12)
    return _xlsx(out)


def shape_account_leads_label():
    """Jack's v8: one column, "4010 - Rental Income", months from B."""
    out = [[None] + MONTHS]
    for label, amt in (("4010 - Rental Income", 100.0),
                       ("5060 - Repairs & Maintenance", 7.0)):
        out.append([label] + [amt] * 12)
    return _xlsx(out)


def shape_two_blocks():
    """Jack's v5: a roll-up in A-B, the detail it came FROM in D-F, months from G.

    The two blocks do NOT line up row for row -- that is the whole danger. Row 2's
    roll-up says 5051 Water; row 2 of the DETAIL is 4090 CAM. Reading the label from
    the left and the amounts from the right produces "5051 - Water" carrying CAM's
    figures, and every part of that is a real value from a real cell.
    """
    out = [[None, None, None, None, None, None] + MONTHS]
    rollup = [("4010 - Rental Income", 1200.0), ("5051 - Water", 240.0),
              ("5060 - Repairs & Maintenance", 120.0)]
    detail = [("4010", "Rental Income", "Base Rent", 100.0),
              ("4090", "Estimated CAM", "CAM Reimb", 60.0),
              ("5060", "Repairs & Maintenance", "HVAC", 5.0),
              ("5060", "Repairs & Maintenance", "Locks/Keys", 5.0),
              ("5051", "Water", "Water and Sewer", 20.0)]
    for i in range(max(len(rollup), len(detail))):
        left = list(rollup[i]) if i < len(rollup) else [None, None]
        if i < len(detail):
            acct, ourdesc, theirlabel, amt = detail[i]
            right = [acct, ourdesc, theirlabel] + [amt] * 12
        else:
            right = [None, None, None] + [None] * 12
        out.append(left + [None] + right)
    return _xlsx(out)


def shape_label_with_subdescription():
    """The NEGATIVE case for rule 3: labels in A, a sub-description in B, months from
    C, and NO account column between. Re-basing here would read every line's name off
    the sub-description."""
    out = [[None, None] + MONTHS]
    for label, sub, amt in (("Base Rent", "all suites", 100.0),
                            ("CAM Reimb", "net of vacancy", 10.0)):
        out.append([label, sub] + [amt] * 12)
    return _xlsx(out)


print("\n1. The label is the line's NAME, and comes from the amounts' own block")
p = B.parse_budget_workbook(shape_account_and_description(), "a.xlsx")
labels = [l["label"] for l in p["lines"]]
accts = [l.get("stated_account") for l in p["lines"]]
check("account in its own column: the label is the description, not the number",
      "Base Rent" in labels and "4010" not in [str(x) for x in labels], labels)
check("account in its own column: the account is still read",
      accts[:2] == ["4010", "4090"], accts)

p = B.parse_budget_workbook(shape_account_leads_label(), "b.xlsx")
check("account LEADING the label is read ('4010 - Rental Income')",
      [l.get("stated_account") for l in p["lines"]] == ["4010", "5060"],
      [l.get("stated_account") for l in p["lines"]])

p = B.parse_budget_workbook(shape_two_blocks(), "c.xlsx")
pairs = [(l.get("stated_account"), l["label"], round(float(l["total"]), 2))
         for l in p["lines"]]
check("two blocks: every line's account, name and figures come from ONE row",
      pairs == [("4010", "Base Rent", 1200.0), ("4090", "CAM Reimb", 720.0),
                ("5060", "HVAC", 60.0), ("5060", "Locks/Keys", 60.0),
                ("5051", "Water and Sewer", 240.0)], pairs)
check("two blocks: the roll-up column is NOT what got imported",
      not any(str(lbl).startswith("5051 - Water") for _, lbl, _ in pairs), pairs)

p = B.parse_budget_workbook(shape_label_with_subdescription(), "d.xlsx")
check("a sub-description beside the labels does NOT re-base the label column",
      [l["label"] for l in p["lines"]] == ["Base Rent", "CAM Reimb"],
      [l["label"] for l in p["lines"]])

print("\n2. The account decides the category")
check("4090 -> CAM, exactly as asked", B.category_for_account("4090") == "CAM",
      B.category_for_account("4090"))
check("an account we do not carry has no category",
      B.category_for_account("999999") is None)
cat_acct = B.category_accounts()
dupes = {}
for cat, accts in cat_acct.items():
    for a in accts:
        dupes.setdefault(str(a).strip(), []).append(cat)
multi = {a: c for a, c in dupes.items() if len(c) > 1}
check("no account belongs to two categories, so one source IS possible",
      not multi, multi)

parsed = B.parse_budget_workbook(shape_account_and_description(), "a.xlsx")
rows = {str(l["row"]): l for l in parsed["lines"]}


def row_for(label):
    return next(k for k, l in rows.items() if l["label"] == label)


# A category that DISAGREES with its account: corrected, not refused.
mapping = {row_for("Base Rent"): {"category": "Repairs & Maintenance",
                                  "account": "4010", "flip": False}}
out = V.validate(parsed, mapping, "P0000018", None)
check("a category disagreeing with its account is CORRECTED to the account's",
      mapping[row_for("Base Rent")]["category"] == "Rental Income",
      mapping[row_for("Base Rent")]["category"])
check("...and does not block the import",
      out["can_import"] and not any(b.get("code") == "account_not_in_category"
                                    for b in out["blocking"]), out["blocking"])

# An account on no category at all: that one IS blocked.
mapping = {row_for("Base Rent"): {"category": None, "account": "999999", "flip": False}}
out = V.validate(parsed, mapping, "P0000018", None)
check("an account on NO category blocks, naming the line",
      any(b.get("code") == "account_not_in_any_category" for b in out["blocking"]),
      out["blocking"])

print("\n3. Many lines may share one account")
mapping = {}
for lbl, acct in (("CAM Reimb", "4090"), ("Water Reimb", "4090"),
                  ("General Repairs", "5060"), ("Electrical Repairs", "5060")):
    mapping[row_for(lbl)] = {"category": B.category_for_account(acct),
                             "account": acct, "flip": False}
out = V.validate(parsed, mapping, "P0000018", None)
check("lines sharing an account do not block",
      out["can_import"] and not any(b.get("code") == "duplicate_account_month"
                                    for b in out["blocking"]), out["blocking"])
combined = [w for w in out["warnings"] if w.get("code") == "lines_combined"]
check("...and the combining is REPORTED, not silent", len(combined) == 2,
      [w.get("message") for w in combined])
msg4090 = next((w["message"] for w in combined if "4090" in w["message"]), "")
check("...naming both lines and their combined total (10+5 over 12 months = 180)",
      "CAM Reimb" in msg4090 and "Water Reimb" in msg4090 and "180" in msg4090,
      msg4090)

print("\n4. The column names are read from the table, whatever it calls them")
for spelling in ("vCode", "vcode"):
    tmp = os.path.join(tempfile.mkdtemp(), "wf.db")
    eng = create_engine("sqlite:///" + tmp)
    with eng.begin() as c:
        c.execute(text(
            f'CREATE TABLE {V.SUPPLEMENT_TABLE} ("{spelling}" TEXT, "dtEntry" TEXT, '
            f'"vSource" TEXT, "vAccount" TEXT, "mAmount" REAL, "vInput" TEXT)'))
    cols = V._supplement_columns(eng)
    check(f"a table spelled '{spelling}' resolves to '{spelling}'",
          cols["vcode"] == spelling, cols)

    m = {row_for("CAM Reimb"): {"category": "CAM", "account": "4090", "flip": False},
         row_for("Water Reimb"): {"category": "CAM", "account": "4090", "flip": False}}
    res = V.commit(eng, "P0000018", parsed, m, "guardrail")
    with eng.connect() as c:
        n = c.execute(text(f'SELECT COUNT(*) FROM {V.SUPPLEMENT_TABLE}')).scalar()
        total = c.execute(
            text(f'SELECT SUM("mAmount") FROM {V.SUPPLEMENT_TABLE}')).scalar()
    check(f"'{spelling}': the write lands (2 lines x 12 months = 24 rows)", n == 24, n)
    check(f"'{spelling}': both lines are written, not deduplicated to one",
          abs(float(total) - 180.0) < 0.01, total)
    check(f"'{spelling}': commit reports what it wrote", res.get("rows_written") == 24, res)

    # Re-importing REPLACES the same months rather than stacking a second revision.
    V.commit(eng, "P0000018", parsed, m, "guardrail")
    with eng.connect() as c:
        n2 = c.execute(text(f'SELECT COUNT(*) FROM {V.SUPPLEMENT_TABLE}')).scalar()
    check(f"'{spelling}': a re-import replaces, it does not stack", n2 == 24, n2)

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
if FAIL:
    print("FAILED:")
    for f in FAIL:
        print("   " + f)
sys.exit(1 if FAIL else 0)
