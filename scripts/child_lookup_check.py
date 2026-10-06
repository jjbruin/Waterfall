"""Guardrail: ONE child-property lookup for the whole app (Oct 6 2026).

There were five copies of "which properties belong to this portfolio deal", and they
disagreed. `consolidation.build_property_map` (and `get_property_vcodes_for_deal`, and
a private copy in `compute.prepare_cap_lookups`) matched children on the parent's
Investment_Name only; `one_pager._child_vcodes_for_parent` and
`valuation_service._child_parent_map` also matched the parent's own Portfolio_Name and
required the parent to be one. Measured on 134 deals they disagreed on two:

  Burton (P0000109)   "Burton Retail Portfolio", buildings labelled "Burton Portfolio":
                      the Dashboard listed the buildings as deals of their own and the
                      capitalization engine put Burton's debt at 0 (cap stack: 75.3M)
  P0000073            took the OTHER "Donald Lynch" (P0000049) as its child -- each of
                      the two counted as the other's child, so the Dashboard hid both

And `ownership_service` called the lookup with its arguments REVERSED; the error was
swallowed, so every portfolio deal was seeded without its children's accounting.

What must stay true, in both directions:
  1. A parent is Property_Count >= 1; a child is Property_Count 0 whose Portfolio_Name
     is the parent's Investment_Name OR its own Portfolio_Name. Burton-shaped data finds
     its children; a non-parent sharing a name finds none; nobody is their own child;
     a sibling is never another sibling's child.
  2. Every copy answers alike: one_pager and valuation_service delegate; the Dashboard's
     capitalization lookups ARE build_property_map.
  3. Reversed arguments raise instead of failing silently, and ownership_service passes
     them in order.
  4. identify_sub_portfolio_deals is DELIBERATELY not this rule (it picks whose FORECASTS
     Deal Analysis sums; Burton's forecast is on the parent, and switching it blanked the
     projection 60 rows -> 0). Asserted so a future "unify" is a decision, not an accident.

Usage: python scripts/child_lookup_check.py [--inject=namerule|nogate|reversed]
  namerule -- children matched on the parent's Investment_Name only (the old rule)
  nogate   -- any deal may be a parent (no Property_Count gate)
  reversed -- ownership_service's reversed call restored
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

INJECT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--inject=")), "")
_passed, _failed = [], []


def chk(label, cond, detail=""):
    (_passed if cond else _failed).append(label)
    print("   %s %s%s" % ("ok  " if cond else "FAIL", label,
                          ("   [%s]" % (detail,)) if detail and not cond else ""))


DEALS = pd.DataFrame([
    # vcode, InvestmentID, Investment_Name, Portfolio_Name, Property_Count
    ("P0000007", "BERGER", "Berger Pittsburgh Portfolio", "Berger Pittsburgh Portfolio", 4),
    ("P0000041", "BEARUN", "Bear Run", "Berger Pittsburgh Portfolio", 0),
    ("P0000042", "HERITA", "Heritage Hills", "Berger Pittsburgh Portfolio", 0),
    ("P0000109", "BURTON", "Burton Retail Portfolio", "Burton Portfolio", 3),
    ("P0000111", "BURT-1", "Burton - Foley Square", "Burton Portfolio", 0),
    ("P0000112", "BURT-2", "Burton - Jubilee Square", "Burton Portfolio", 0),
    ("P0000049", "MCCORD", "Donald Lynch", "Donald Lynch", 1),
    ("P0000073", "MCCORD", "Donald Lynch", "Donald Lynch", 0),
    ("P0000010", "WESTBK", "Centre at Westbank", "", 0),
], columns=["vcode", "InvestmentID", "Investment_Name", "Portfolio_Name", "Property_Count"])


def main():
    import consolidation as C
    if INJECT == "namerule":
        def _old(deals):
            df = deals.copy()
            m = {}
            for _, r in df.iterrows():
                kids = [c for c in df[df.Portfolio_Name == r.Investment_Name].vcode if c != r.vcode]
                if kids:
                    m[r.vcode] = kids
            return m
        C.build_property_map = _old
    if INJECT == "nogate":
        _real = C.build_property_map
        C.build_property_map = lambda deals: _real(deals.drop(columns=["Property_Count"]))

    import one_pager
    from flask_app.services import valuation_service
    import compute

    print("1. The rule")
    pm = C.build_property_map(DEALS)
    chk("a parent named differently from its children's label finds them (Burton)",
        sorted(pm.get("P0000109", [])) == ["P0000111", "P0000112"], pm.get("P0000109"))
    chk("the usual case still works (Berger)",
        sorted(pm.get("P0000007", [])) == ["P0000041", "P0000042"], pm.get("P0000007"))
    chk("a deal that is not a portfolio has no children, whoever shares its name (P0000073)",
        "P0000073" not in pm, pm.get("P0000073"))
    chk("...while the real parent of the pair keeps its child (P0000049 -> P0000073)",
        pm.get("P0000049") == ["P0000073"], pm.get("P0000049"))
    chk("nobody is their own child", all(p not in kids for p, kids in pm.items()))
    chk("a child property is never a parent", not ({"P0000041", "P0000111"} & set(pm)))
    chk("a standalone deal is absent", "P0000010" not in pm)

    print("2. One answer everywhere")
    chk("get_property_vcodes_for_deal is build_property_map",
        all(sorted(C.get_property_vcodes_for_deal(v, DEALS)) == sorted(pm.get(v, []))
            for v in DEALS.vcode))
    chk("one_pager._child_vcodes_for_parent agrees on every deal",
        all(sorted(one_pager._child_vcodes_for_parent(v, DEALS)) == sorted(pm.get(v, []))
            for v in DEALS.vcode))
    inv_map = {c: p for p, kids in pm.items() for c in kids}
    chk("valuation_service._child_parent_map is its inverse",
        valuation_service._child_parent_map(DEALS) == inv_map,
        valuation_service._child_parent_map(DEALS))
    chk("the Dashboard's capitalization lookups use it (compute.prepare_cap_lookups)",
        compute.prepare_cap_lookups(pd.DataFrame(), DEALS, None, None)["prop_map"] == pm)

    print("3. Called the right way round")
    try:
        C.get_property_vcodes_for_deal(DEALS, "P0000109")
        raised = False
    except TypeError:
        raised = True
    chk("reversed arguments raise TypeError, not a silent failure", raised)
    own = (ROOT / "flask_app/services/ownership_service.py").read_text(encoding="utf-8")
    if INJECT == "reversed":
        own = own.replace("get_property_vcodes_for_deal(str(entity_id), inv)",
                          "get_property_vcodes_for_deal(inv, str(entity_id))")
    # The first argument must be the deal id, never the deals frame.
    calls = re.findall(r"get_property_vcodes_for_deal\(\s*([A-Za-z_][\w.]*)", own)
    chk("ownership_service passes (deal_vcode, deals)",
        calls and all(c == "str" for c in calls), calls)

    print("4. Forecast consolidation is deliberately separate")
    sub = C.identify_sub_portfolio_deals(None, DEALS)
    chk("Burton is NOT a forecast sub-portfolio (its forecast is on the parent)",
        "BURTON" not in sub, sub)
    chk("Berger still is", sorted(sub.get("BERGER", [])) == ["BEARUN", "HERITA"], sub)

    print("\n%d passed, %d failed" % (len(_passed), len(_failed)))
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
