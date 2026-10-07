"""Guardrail: the ownership screen seeds a portfolio WITH its child properties.

`ownership_service.run_upstream_analysis` called
`consolidation.get_property_vcodes_for_deal(inv, entity_id)` -- arguments reversed. It
raised, the surrounding `except` logged a warning and carried on with no children, so
every portfolio deal was seeded from the parent's accounting alone (found Oct 6 2026,
fixed Oct 7). Measured on local data: only P0000049 moved (seed capital 0 -> 59,100);
the other portfolios' children carry no separate accounting today.

The parent/child rule itself is the February 2026 one in consolidation.py and is NOT
asserted here -- only that this caller uses it correctly.

Asserted in both directions:
  1. Every call passes the deal id first and the deals frame second.
  2. Called that way, the lookup returns a portfolio's children...
  3. ...and called the old way it fails -- so a reversed call can never again pass
     unnoticed as "no children".

Usage: python scripts/ownership_child_call_check.py [--inject=reversed]
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


def main():
    from consolidation import get_property_vcodes_for_deal
    src = (ROOT / "flask_app/services/ownership_service.py").read_text(encoding="utf-8")
    if INJECT == "reversed":
        src = src.replace("get_property_vcodes_for_deal(str(entity_id), inv)",
                          "get_property_vcodes_for_deal(inv, str(entity_id))")

    print("1. The call")
    firsts = re.findall(r"get_property_vcodes_for_deal\(\s*([A-Za-z_][\w.]*)", src)
    chk("ownership_service calls the lookup", bool(firsts))
    chk("...with the deal id first, the deals frame second",
        firsts and all(f == "str" for f in firsts), firsts)

    print("2. The lookup, called that way")
    deals = pd.DataFrame([
        ("P0000033", "OREI Portfolio", ""),
        ("P0000061", "Whitney Manor Apartments", "OREI Portfolio"),
        ("P0000062", "Westchase Apartments", "OREI Portfolio"),
        ("P0000010", "Centre at Westbank", ""),
    ], columns=["vcode", "Investment_Name", "Portfolio_Name"])
    chk("returns a portfolio's children",
        sorted(get_property_vcodes_for_deal("P0000033", deals)) == ["P0000061", "P0000062"])
    chk("and none for a standalone deal", get_property_vcodes_for_deal("P0000010", deals) == [])

    print("3. The old way fails rather than answering 'no children'")
    try:
        out = get_property_vcodes_for_deal(deals, "P0000033")
        raised = False
    except Exception:                                   # noqa: BLE001
        raised, out = True, None
    chk("reversed arguments raise", raised, out)

    print("\n%d passed, %d failed" % (len(_passed), len(_failed)))
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
