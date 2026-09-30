"""Guardrail: the two One Pager participation cells agree, and a 1.0 is 100%.

The One Pager prints the PE participation TWICE — Capitalization
(`cap_stack.pe_participation`) and PE Performance (`pe_performance
.participation`) — and until 2026-09-30 they resolved it differently:

    Capitalization   waterfall base, MRI deal terms OVERRIDE
    PE Performance   waterfall FIRST, deal terms only as a fallback

so on a deal where the two sources disagree the same term printed two numbers.
Both now take MRI deal terms first and fall back to the waterfall only where
deal terms say nothing.

The second defect was the percent normalisation. A share is stated either as a
fraction (0.475) or as a percentage (47.5), and the reader decided with
`v if v < 1 else v / 100`. That sends an exactly-1.0 share — ALL of it — down
the percentage branch, 1.0 / 100 = 0.01, and the page prints "1%" for a
hundred percent. `one_pager.normalize_share` uses `<= 1` and is now the single
definition, called from all four read sites.

WHY THIS FILE RATHER THAN THE EXISTING TWO. `onepager_pe_terms_fallback_check`
and `onepager_participation_zero_check` both `import live_api`, which is NOT IN
THE REPOSITORY — they cannot run on any branch, main included. This one is
fixture-driven and runs anywhere; the fixtures are the REAL waterfall FXRate
and deal_terms.pe_split_capital values read off production on 2026-09-30.

Usage
    python scripts/onepager_participation_precedence_check.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import one_pager as OP                                           # noqa: E402
from flask_app.services.financials_service import (              # noqa: E402
    _enrich_cap_stack_from_deal_terms)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_checks = []


def chk(label, ok, detail=""):
    _checks.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  -- {detail}" if not ok and detail else ""))


#: Measured on production 2026-09-30 at 26Q2 — every deal carrying BOTH a
#: waterfall Share row and a deal_terms participation where the two disagree,
#: plus the shape each one exercises.
#:   vcode, name, waterfall FXRate, deal_terms.pe_split_capital, expected share
LIVE = [
    ("P0000008", "5-15 Broad St",             0.333, 0.33,  0.33),
    ("P0000028", "Merle Hay",                 0.7,   0.3,   0.3),
    ("P0000033", "OREI Portfolio",            0.75,  0.475, 0.475),
    ("P0000061", "Whitney Manor Apartments",  0.75,  0.475, 0.475),
    ("P0000062", "Westchase Apartments",      0.75,  0.475, 0.475),
    ("P0000073", "Donald Lynch",              0.2,   0.3,   0.3),
    # The two OPJPI deals: the waterfall reports the OPERATING PARTNER's whole
    # 1.0 share because it takes the first vState='Share' row with no PropCode
    # filter; deal_terms states 0 affirmatively and wins.
    ("P0000077", "Jefferson Addison Heights", 1.0,   0.0,   0.0),
    ("P0000085", "Jefferson Eastchase",       1.0,   0.0,   0.0),
]


def _dt(vcode, split):
    return pd.DataFrame([{"vcode": vcode, "pe_coupon": None,
                          "pe_split_capital": split}])


def both_cells(vcode, wf_value, dt_value):
    """(capitalization, pe_performance) participation for one deal.

    Drives the SHIPPING resolvers: `_enrich_cap_stack_from_deal_terms` for the
    Capitalization cell and `_pe_terms_fallback` for the PE block, each seeded
    with the waterfall figure exactly as its own reader would have left it.
    """
    seed = None if wf_value is None else OP.normalize_share(wf_value)
    cap = {"pe_participation": seed, "pe_coupon": 0.0}
    _enrich_cap_stack_from_deal_terms(cap, _dt(vcode, dt_value), vcode)
    pe = {"participation": seed, "coupon": 0.0}
    OP._pe_terms_fallback(pe, _dt(vcode, dt_value), vcode)
    return cap["pe_participation"], pe["participation"]


print(__doc__.strip().split("\n")[0])

print("\n1. normalize_share — one definition, and 1.0 is 100%")
n = OP.normalize_share
chk("a 1.0 share is 100%, not 1% (the render bug)", n(1.0) == 1.0,
    f"got {n(1.0)}")
chk("an integer 1 is 100% too", n(1) == 1)
chk("a fraction below 1 is left alone", n(0.475) == 0.475 and n(0.0) == 0.0)
chk("0.999 stays a fraction, it is not a percentage", n(0.999) == 0.999)
chk("a percentage above 1 is divided", n(47.5) == 0.475 and n(50) == 0.5)
chk("100 is 100%", n(100) == 1.0)
# BOTH directions: `<= 1` must not swallow the percentage branch entirely.
chk("1.5 is still read as a percentage (1.5%)", abs(n(1.5) - 0.015) < 1e-12)

print("\n2. the two cells agree, on every live deal where the sources differ")
print(f"    {'vcode':<10}{'deal':<27}{'wf':>7}{'terms':>8}"
      f"{'CAP':>9}{'PE':>9}   agree")
for vc, nm, wf, dt, want in LIVE:
    cap_v, pe_v = both_cells(vc, wf, dt)
    agree = cap_v == pe_v
    print(f"    {vc:<10}{nm[:26]:<27}{wf:>7}{dt:>8}"
          f"{cap_v if cap_v is None else round(cap_v, 4):>9}"
          f"{pe_v if pe_v is None else round(pe_v, 4):>9}   {agree}")
    chk(f"{vc} {nm[:22]}: both cells show the deal-terms value {want}",
        cap_v == pe_v == want, f"cap={cap_v} pe={pe_v}")

print("\n3. the waterfall is still the fallback — both directions")
# No deal_terms row at all: the seeded waterfall value must survive in BOTH.
cap_v, pe_v = both_cells("P0000999", 0.42, None)
chk("with NO deal_terms value both cells keep the waterfall's 0.42",
    cap_v == pe_v == 0.42, f"cap={cap_v} pe={pe_v}")
# A deal_terms row that exists but carries nothing is not a value either.
cap_v, pe_v = both_cells("P0000998", 0.42, float("nan"))
chk("a NaN deal_terms value is not a value — the waterfall stands",
    cap_v == pe_v == 0.42, f"cap={cap_v} pe={pe_v}")
# And a negative is refused rather than shown.
cap_v, pe_v = both_cells("P0000997", 0.42, -1.0)
chk("a negative deal_terms value is refused, waterfall stands",
    cap_v == pe_v == 0.42, f"cap={cap_v} pe={pe_v}")
# Neither source: stays None so the page prints N/A, never 0.0%.
cap_v, pe_v = both_cells("P0000996", None, None)
chk("with NEITHER source both cells stay None (page prints N/A)",
    cap_v is None and pe_v is None, f"cap={cap_v} pe={pe_v}")

print("\n4. a waterfall 1.0 with no deal terms renders 100%, not 1%")
cap_v, pe_v = both_cells("P0000995", 1.0, None)
chk("both cells show 1.0 (=100%), the latent case the <= 1 fix covers",
    cap_v == pe_v == 1.0, f"cap={cap_v} pe={pe_v}")

print("\n5. the COUPON keeps waterfall primacy — this change is participation only")
dt = pd.DataFrame([{"vcode": "P0000994", "pe_coupon": 9.0,
                    "pe_split_capital": None}])
pe = {"participation": None, "coupon": 0.08}
OP._pe_terms_fallback(pe, dt, "P0000994")
chk("a coupon the waterfall supplied is NOT overwritten by deal terms",
    pe["coupon"] == 0.08, f"got {pe['coupon']}")
pe = {"participation": None, "coupon": 0.0}
OP._pe_terms_fallback(pe, dt, "P0000994")
chk("but an absent coupon is still filled from deal terms (9% -> 0.09)",
    abs(pe["coupon"] - 0.09) < 1e-12, f"got {pe['coupon']}")

print()
ok = sum(_checks)
print(f"{ok}/{len(_checks)} checks passed")
sys.exit(0 if ok == len(_checks) else 1)
