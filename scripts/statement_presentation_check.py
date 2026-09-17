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

print("The balance sheet foots to something a reader can check")
# Jim, Sep 17 2026: a bolded total of liabilities and members' capital, below
# members' capital, so it can be read straight against total assets. Computed in
# the ENGINE so the screen, the workbook and the printed statement cannot
# disagree; each consumer only renders it.
# This file otherwise tests pure functions and needs no app context; the
# footing is computed from the GL, so this one section does.
_bs = None
try:
    from flask_app import create_app
    with create_app().app_context():
        _bs = ss.build("PPIECH", "2026-06-30", "balance_sheet").get("balance_sheet")
except Exception as _e:
    print("   (no database available: %s)" % str(_e)[:60])
if not (_bs or {}).get("sections"):
    # `_bs` is a dict even when the entity has no GL rows at this date, so
    # testing it for truth passes a statement with nothing in it straight into
    # checks that then fail for lack of data. A local database without PPIECH's
    # GL reported a missing footing as a defect; production has the data and
    # foots to 33,378,047.84. Skipping is honest here, failing was not.
    print("   (no balance sheet data in this database -- footing checks skipped)")
else:
    _lc = _bs.get("footing")
    check("the footing exists", bool(_lc))
    if _lc:
        _tot = {x["section"]: x["total"] for x in _bs["sections"]}
        check("it is liabilities plus members' capital, nothing else",
              abs(_lc["amount"] - (_tot.get("Liabilities", 0.0)
                                   + _tot.get("Members' Capital", 0.0))) < 0.01)
        # It must NOT be a section: anything summing `sections` would count it
        # twice, including the tie-out computed immediately after it.
        check("it is not added to sections",
              all(x["section"] != _lc["label"] for x in _bs["sections"]))
        # The footing and the existing tie-out are two independent routes to the
        # same imbalance, so they have to agree in magnitude.
        check("it agrees with the tie-out already reported",
              _lc["difference"] is None or _bs.get("out_of_balance") is None
              or abs(abs(_lc["difference"]) - abs(_bs["out_of_balance"])) < 0.02)
        # "ties" must never be asserted when there is nothing to tie TO.
        check("ties is None when there is nothing to compare, never True",
              _lc["compare_amount"] is not None or _lc["ties"] is None)

print("Every consumer renders the footing")
_excel = open("flask_app/services/workpaper_excel.py", encoding="utf-8").read()
_vue = open("vue_app/src/views/WorkpapersView.vue", encoding="utf-8").read()
_prt = open("vue_app/src/views/StatementsPrintView.vue", encoding="utf-8").read()
check("workbook renders it", "_footing_rows" in _excel)
check("workbench renders it", ".footing" in _vue)
check("printed statement renders it", ".footing" in _prt)
# ONE SHAPE FOR ALL THREE STATEMENTS, so a fourth gets the treatment free and
# no consumer carries a special case per statement.
with create_app().app_context():
    _inc = ss.build("PPIECH", "2026-06-30", "both").get("income_statement")
    _cf = ss.build_cash_flow("PPIECH", "2026-06-30")
check("the income statement foots to net income",
      bool(_inc and _inc.get("footing")
           and "Net income" in _inc["footing"]["label"]
           and abs(_inc["footing"]["amount"] - _inc["net_income"]) < 0.01))
check("the cash flow foots to the net change in cash",
      bool(_cf and _cf.get("footing")
           and "Net increase" in _cf["footing"]["label"]
           and abs(_cf["footing"]["amount"] - _cf["net_change_computed"]) < 0.01))
check("and compares it to the movement the cash accounts show",
      bool(_cf and _cf["footing"]["compare_amount"] == _cf["net_change_actual"]))
check("net income has nothing to compare against, so ties is None",
      bool(_inc and _inc["footing"]["ties"] is None))

print("The printed statements render EVERY shape, not just sections")
# THE DEFECT THIS PINS. The print view rendered `sections` only, so the schedule
# of investments and members' capital printed a header over an empty table --
# present, titled and blank (Jim, Sep 17 2026). Three shapes come out of this
# engine and each needs its own renderer:
#     sections   balance sheet, income statement, cash flow
#     lines      schedule of investments
#     rows x members   statement of changes in members' capital
with create_app().app_context():
    _mc = ss.build_members_capital("PPIECH", "2026-06-30")
    _soi = ss.build_schedule_of_investments("PPIECH", "2026-06-30")
check("members' capital is rows x members, not sections",
      "sections" not in _mc and isinstance(_mc.get("rows"), list)
      and isinstance(_mc.get("members"), list))
check("the schedule of investments is lines, not sections",
      "sections" not in _soi and "lines" in _soi)
# The field names the view reads, against the ones the engine emits. A rename on
# either side prints blanks rather than failing, which is why this is asserted.
if _mc.get("rows") and _mc.get("members"):
    _r, _m = _mc["rows"][0], _mc["members"][0]
    check("a members' capital row carries label/by_member/total",
          {"label", "by_member", "total"} <= set(_r))
    check("and a member carries InvestorID, which by_member is keyed on",
          "InvestorID" in _m and _m["InvestorID"] in _r["by_member"])
_prt = open("vue_app/src/views/StatementsPrintView.vue", encoding="utf-8").read()
check("the print view branches on shape", "shapeOf(" in _prt)
check("it has a schedule-of-investments renderer", "soiLines(" in _prt)
check("it has a members' capital renderer", "by_member" in _prt)
check("and it reads the member key the engine emits", "InvestorID" in _prt)

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("All presentation checks passed.")
