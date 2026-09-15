"""A drafted statement must READ like a statement.

Two rules, both about presentation and neither about arithmetic:

  ORDER     Lines run in statement order -- most liquid first on the balance
            sheet, revenue before expenses on the income statement -- not in
            whatever order the account mapping happened to be written. Before
            this, PPIECH's assets came out with "Due from Manager" above
            "Cash and cash equivalents" because that is the order the accounts
            were tagged in.

  DORMANT   A line with no balance AND no movement is marked, so the printed
            statement and the screen can leave it off. PPIECH showed four
            0.00 lines; a page of them reads as a trial balance, not a
            balance sheet. A zero line that HAD movement is NOT dormant --
            hiding it would make the statement disagree with the trial
            balance behind it, and "this went out and came back" is a fact
            about the period.

Both fail against the code as it stood on Sep 14 2026: there was no rank
table and no dormant flag.
"""
import sys

sys.path.insert(0, ".")

from flask_app.services import fs_line_seed as seed          # noqa: E402
from flask_app.services import statement_service as ss       # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"  {detail}" if detail else ""))
    if not cond:
        failures.append(name)


print("Every caption accounting uses has a rank")
unranked = sorted(l for l in seed.LINE_SECTION if l not in seed.LINE_ORDER)
check("all 56 statement captions are ranked", not unranked, f"unranked: {unranked}")

print("Balance sheet order is most-liquid-first")
bs_sections = ss.BALANCE_SHEET_SECTIONS
scrambled = [
    {"section": "Members' Capital", "fs_line": "Capital contributions"},
    {"section": "Assets", "fs_line": "Due from manager"},
    {"section": "Liabilities", "fs_line": "Accrued expenses"},
    {"section": "Assets", "fs_line": "Cash and cash equivalents"},
    {"section": "Assets", "fs_line": "Prepaid expenses"},
    {"section": "Assets", "fs_line": "Investment in real estate"},
]
ordered = [l["fs_line"] for l in
           sorted(scrambled, key=lambda l: ss.line_sort_key(bs_sections, l))]
check("cash leads the assets", ordered[0] == "Cash and cash equivalents", ordered[0])
check("assets, then liabilities, then capital",
      ordered.index("Accrued expenses") > ordered.index("Investment in real estate")
      and ordered.index("Capital contributions") > ordered.index("Accrued expenses"),
      " -> ".join(ordered))
# Liquidity order within assets: a due-from is a receivable and sits above
# prepaids; a prepaid is consumed within the year and sits above a long-term
# investment.
check("receivable, then prepaid, then investment",
      ordered.index("Due from manager") < ordered.index("Prepaid expenses")
      < ordered.index("Investment in real estate"), " -> ".join(ordered))

print("Income statement runs income before expenses")
inc = [{"section": "Expenses", "fs_line": "Professional fees"},
       {"section": "Income", "fs_line": "Investment income"}]
inc_ordered = [l["fs_line"] for l in
               sorted(inc, key=lambda l: ss.line_sort_key(ss.INCOME_SECTIONS, l))]
check("income first", inc_ordered[0] == "Investment income", " -> ".join(inc_ordered))

print("An unranked caption sorts last rather than first")
mixed = [{"section": "Assets", "fs_line": "Zzz brand new caption"},
         {"section": "Assets", "fs_line": "Cash and cash equivalents"}]
last = [l["fs_line"] for l in
        sorted(mixed, key=lambda l: ss.line_sort_key(bs_sections, l))][-1]
check("new caption does not displace ranked ones", last == "Zzz brand new caption", last)

print("Dormant means no balance and no movement")
check("zero balance, zero movement -> dormant",
      ss.is_dormant({"closing": 0.0, "ytd": 0.0}))
check("zero balance WITH movement -> kept",
      not ss.is_dormant({"closing": 0.0, "ytd": -125_000.0}))
check("balance with no movement -> kept",
      not ss.is_dormant({"closing": 2_614_646.68, "ytd": 0.0}))
check("rounding noise below half a cent -> dormant",
      ss.is_dormant({"closing": 0.004, "ytd": -0.004}))

print("Consumers suppress dormant lines but say how many")
excel = open("flask_app/services/workpaper_excel.py", encoding="utf-8").read()
vue = open("vue_app/src/views/WorkpapersView.vue", encoding="utf-8").read()
check("workbook filters dormant", 'not l.get("dormant")' in excel)
check("workbook states the count", "are not shown" in excel)
check("screen filters dormant", "!x.dormant" in vue)
check("screen states the count", "dormant_count" in vue)

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("All presentation checks passed.")
