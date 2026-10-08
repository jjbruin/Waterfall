"""Guardrail: a floating loan's rate cap is READ from MRI, never guessed.

Run: .venv\\Scripts\\python scripts\\loan_caps_check.py

`loan_caps.cap_terms` owns the cap on a floating loan (Board p. 28): the strike
from `vIntRatereset`, the expiry -- and the strike where the field is empty --
from asset management's free text `vHedgedStrat`, and max rate = strike + spread.
Every text below is one MRI holds (production, Oct 8 2026). Checked both ways:
what it must read, and what it must REFUSE to read.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flask_app.services.loan_caps import cap_terms  # noqa: E402

PASSED, FAILED = 0, []


def chk(label, cond, detail=""):
    global PASSED
    if cond:
        PASSED += 1
        print("   ok  ", label)
    else:
        FAILED.append(label)
        print("   FAIL", label, detail)


def near(a, b):
    return a is not None and b is not None and abs(a - b) < 1e-9


print("\n1. What MRI holds, read")
b = cap_terms(".05", "Yes", "5.00%, exp: 6/27", "0.04")
chk("Belleville: 5.00% cap + 4.00% = 9.00%, through 6/27 (the January deck's line)",
    near(b["strike"], .05) and near(b["max_rate"], .09) and b["expiry"] == "6/27" and b["capped"] is True, b)
m = cap_terms(".02", "Yes", "2.00% (eff. 5.70%), exp: 10/26", ".037")
chk("Middle Island: the (eff. x%) form, 2.00% + 3.70%, through 10/26",
    near(m["strike"], .02) and near(m["max_rate"], .057) and m["expiry"] == "10/26", m)
t = cap_terms(None, "Yes", "4.00%, exp: 8/28", "0.0225")
chk("Trolley Square: no vIntRatereset, the strike is read from the text",
    near(t["strike"], .04) and t["source"] == "vHedgedStrat" and near(t["max_rate"], .0625), t)
p = cap_terms(".06", "Yes", "6.00% (eff. 8.85%), full term", "0.0285", "9/27")
chk("Poplar Prairie: 'full term' runs to the loan's maturity", p["expiry"] == "9/27" and near(p["max_rate"], .0885), p)
chk("a strike written 5 (percent) and 0.05 (decimal) is the same strike",
    near(cap_terms("5", None, None, "4")["strike"], .05) and near(cap_terms("5", None, None, "4")["max_rate"], .09))
chk("a strike in vIntRatereset with no text is a cap, with no expiry claimed",
    cap_terms(".01", None, None, "0.04")["capped"] is True and cap_terms(".01", None, None, "0.04")["expiry"] is None)

print("\n2. What it must refuse to read")
mp = cap_terms(None, "Yes", "5.00% for $5.3M, 5yr $825k", ".025")
chk("Mount Prospect's partial cap is NOT read as a 5.00% strike", mp["strike"] is None and mp["max_rate"] is None, mp)
chk("...it says it cannot read it (capped None, readable False) and keeps the words",
    mp["capped"] is None and mp["readable"] is False and "for $5.3M" in (mp["text"] or ""))
h = cap_terms(None, "Yes", None, ".04")
chk("a hedge flagged Yes with no terms is unknown, not uncapped", h["capped"] is None and h["problem"])
n = cap_terms(None, None, None, "0.03")
chk("no strike, no hedge, no text: uncapped (False), and no max rate", n["capped"] is False and n["max_rate"] is None)
d = cap_terms(".03", "Yes", "2.50%, exp: 1/27", ".03")
chk("field and text disagree: the field is used AND the disagreement is reported",
    near(d["strike"], .03) and d["problem"] and "2.50%" in d["problem"], d)
chk("no spread: the strike is known, the max rate is not (None, not the strike)",
    cap_terms(".05", "Yes", "5.00%, exp: 6/27", None)["max_rate"] is None)

print("\n%d passed, %d failed" % (PASSED, len(FAILED)))
for f in FAILED:
    print("  -", f)
sys.exit(1 if FAILED else 0)
