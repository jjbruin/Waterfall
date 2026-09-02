"""Guardrail: Financial subtab — funding row removed, ITD summed, Net ROE manual.

Covers the four changes in the Sep 2 2026 work order (Prompt A):

  1. the "Total Current Funding" row is gone, and Portfolio Totals is untouched
  2. ITD Distributions renders with its unit ("$15.33M") and is SUMMED onto
     every aggregate row
  3. Net ROE is manual at EVERY level and renders with "%" — never derived
  4. the excluding-development row sums ITD over non-development deals only and
     takes its own typed Net ROE

Runs the REAL committed ``assemble_financial`` on both sides of the change. The
fixture is injected — ``one_pager_provider``, ``manual_loader`` and
``footnote_loader`` are already parameters — so this needs no database, no
network and no live token, unlike the module self-tests, which import the
uncommitted ``live_api`` helper.

    python scripts/snapshot_itd_roe_check.py capture before.json
    python scripts/snapshot_itd_roe_check.py report before.json after.json

For a genuine before/after, run ``capture`` from a worktree at the pre-change
commit for the "before" file and from the working tree for the "after".
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Fixture ───────────────────────────────────────────────────────────────
#
# Real vcodes, so the excluding-development population (EXCLUDING_DEV_VCODES)
# and the per-cell n/a map (PDF_NA_CELLS) are exercised exactly as they are in
# production. The amounts are invented and deliberately round.
#
#   P0000019  Giant 7                    operating, ITD entered
#   PCITWES   City West                  Net ROE is n/a per the PDF
#   P0000030  Nottingham Village         operating, ITD entered
#   P0000067  Brainerd Place Apartments  DEVELOPMENT — out of the ex-dev row
#   P0000075  stand-in                   operating, ITD NOT entered
DEALS = {
    "IND": [
        {"vcode": "P0000019", "name": "Giant 7", "investment_strategy": "Core"},
        {"vcode": "PCITWES", "name": "City West", "investment_strategy": "Core"},
    ],
    "TGA22": [
        {"vcode": "P0000030", "name": "Nottingham Village",
         "investment_strategy": "Value Add"},
        {"vcode": "P0000067", "name": "Brainerd Place Apartments",
         "investment_strategy": "Development"},
        {"vcode": "P0000075", "name": "Stand-in Deal",
         "investment_strategy": "Core"},
    ],
}

CAP = {
    "P0000019": dict(debt=30e6, pref_equity=21e6, partner_equity=9e6,
                     committed_pe=21e6),
    "PCITWES": dict(debt=0.0, pref_equity=8e6, partner_equity=4e6,
                    committed_pe=8e6),
    "P0000030": dict(debt=20e6, pref_equity=9.1e6, partner_equity=5e6,
                     committed_pe=12.058427e6),
    "P0000067": dict(debt=40e6, pref_equity=23.6e6, partner_equity=10e6,
                     committed_pe=23.6e6),
    "P0000075": dict(debt=15e6, pref_equity=6e6, partner_equity=3e6,
                     committed_pe=6e6),
}

# Typed manual entries. Note the AGGREGATE keys: a fund subtotal, the portfolio
# total and the excluding-development row each carry their own Net ROE, stored
# through the same table and the same endpoint as a deal cell.
# ITD IS STORED IN MILLIONS, the unit the column displays — these are the real
# 26Q1 figures (Giant 7 5.87, JB Fair Park 1.17). v410 wrote this fixture in
# DOLLARS and passed, which is exactly why every live cell then rendered
# "$0.00M": the fixture agreed with the bug instead of catching it.
MANUAL = {
    "P0000019": {"itd": 5.87, "net_roe": 4.4},
    "PCITWES": {"itd": 2.10},                          # net_roe is n/a here
    "P0000030": {"itd": 1.25, "net_roe": 7.9},
    "P0000067": {"itd": 0.90, "net_roe": -2.1},        # development
    # P0000075 deliberately has no ITD, so the column is partial.
    "__GROUP__:IND": {"net_roe": 5.1},
    "__GROUP__:TGA22": {"net_roe": 3.2},
    "__TOTAL__": {"net_roe": 4.8},
    "__EXCLUDING_DEV__": {"net_roe": 6.3},
}

RESOLVED = {"investor_code": "TGAM", "investor_name": "TIAA",
            "groups": DEALS, "flagged": []}

KEEP = ("label", "deal_count", "total_pref", "total_cap", "total_commitment",
        "itd", "itd_display", "itd_deal_count", "itd_source",
        "net_roe", "net_roe_display", "net_roe_vcode", "manual_entered",
        "excluded_count")


def _provider(vcode, quarter):
    c = CAP.get(vcode) or {}
    total = (c.get("debt", 0) + c.get("pref_equity", 0)
             + c.get("partner_equity", 0))
    return {"cap_stack": {
        **c,
        "debt_isbs": c.get("debt"),
        "total_cap": total,
        "total_cap_isbs": total,
        "pe_exposure_on_cap": None,
        "debt_pct": None, "pref_equity_pct": None, "partner_equity_pct": None,
        "investor_pct_of_pref": 0.9,
    }}


def _slim(row):
    return None if not row else {k: row.get(k) for k in KEEP if k in row}


def capture(outfile: str) -> int:
    from flask_app.services.portfolio_snapshot_financial import assemble_financial
    payload = assemble_financial(
        "TGAM", "2026-Q1",
        resolved=RESOLVED,
        one_pager_provider=_provider,
        manual_loader=lambda i, q: MANUAL,
        footnote_loader=lambda i, q: [],
    )
    slim = {
        "has_funding_row": "total_current_funding" in payload,
        "funding_row_label": (payload.get("total_current_funding")
                              or {}).get("label"),
        "total": _slim(payload.get("total")),
        "excluding_dev": _slim(payload.get("total_excluding_dev")),
        "groups": {g: _slim(b.get("subtotal"))
                   for g, b in (payload.get("groups") or {}).items()},
        "deals": {r["vcode"]: {k: r.get(k) for k in
                               ("itd", "itd_display", "net_roe",
                                "net_roe_display")}
                  for b in (payload.get("groups") or {}).values()
                  for r in b["deals"]},
    }
    with open(outfile, "w", encoding="utf-8") as fh:
        json.dump(slim, fh, indent=2, default=str)
    print("captured -> " + outfile)
    return 0


def report(before_f: str, after_f: str) -> int:
    with open(before_f, encoding="utf-8") as fh:
        b = json.load(fh)
    with open(after_f, encoding="utf-8") as fh:
        a = json.load(fh)

    checks = []

    def chk(label, cond, detail=""):
        checks.append(bool(cond))
        print("  [{}] {}".format("PASS" if cond else "FAIL", label)
              + ("\n           " + detail if detail else ""))

    print("\n1. Total Current Funding row removed")
    # An invariant of the AFTER side. Against a pre-removal baseline the detail
    # line shows the row that went; against a later one it says "already gone",
    # and either way the assertion is the same.
    chk("the row is absent", not a["has_funding_row"],
        "baseline: " + (repr(b.get("funding_row_label"))
                        if b["has_funding_row"] else "already removed"))

    print("\n2. Portfolio Totals intact (the row above it must not move)")
    for col in ("deal_count", "total_pref", "total_cap", "total_commitment"):
        bv, av = b["total"].get(col), a["total"].get(col)
        chk("Portfolio Totals " + col + " unchanged", bv == av,
            "{} -> {}".format(bv, av))

    print("\n3. ITD renders with its unit, per deal")
    for vc, want in (("P0000019", "$5.87M"), ("P0000030", "$1.25M"),
                     ("P0000067", "$0.90M")):
        got = a["deals"][vc]["itd_display"]
        chk("{} ITD displays {}".format(vc, want), got == want,
            "got " + repr(got) + "   before: "
            + repr(b["deals"][vc]["itd_display"]))
    chk("a deal with no ITD entered still reads 'pending entry'",
        a["deals"]["P0000075"]["itd_display"] == "pending entry",
        "got " + repr(a["deals"]["P0000075"]["itd_display"]))

    print("\n4. ITD summed onto every aggregate row")
    #   IND   = 5.87 + 2.10                     = 7.97
    #   TGA22 = 1.25 + 0.90 (P0000075 has none) = 2.15
    #   TOTAL                                   = 10.12
    for key, want in (("IND", 7.97), ("TGA22", 2.15)):
        got = a["groups"][key].get("itd")
        chk("{} subtotal ITD = ${:,.2f}M".format(key, want),
            got is not None and abs(got - want) < 1e-6,
            "got {}   before: {}".format(got, b["groups"][key].get("itd")))
    chk("Portfolio Totals ITD = $10.12M",
        a["total"].get("itd") is not None
        and abs(a["total"]["itd"] - 10.12) < 1e-6,
        "got {}   before: {}".format(a["total"].get("itd"),
                                     b["total"].get("itd")))
    # Stated as an invariant of the AFTER side, not as a delta: this script is
    # run against whatever baseline is to hand, and against v410 (which already
    # sums) a "was absent before" assertion fails for the wrong reason.
    chk("every aggregate row carries a sum",
        all(x.get("itd") is not None for x in
            [a["total"], a["groups"]["IND"], a["excluding_dev"]]),
        "before: total={}  IND={}".format(b["total"].get("itd"),
                                          b["groups"]["IND"].get("itd")))
    chk("a partial column reports how many deals fed it",
        a["groups"]["TGA22"].get("itd_deal_count") == 2
        and a["groups"]["TGA22"].get("deal_count") == 3,
        str(a["groups"]["TGA22"].get("itd_source")))
    chk("Portfolio Totals ITD displays with its unit",
        a["total"].get("itd_display") == "$10.12M",
        "got " + repr(a["total"].get("itd_display")))

    # ── The v410 regression, asserted directly ────────────────────────────
    #
    # ITD is stored in the unit its column displays. A formatter that divides
    # by 1e6 turns every real figure into "$0.00M" — which is what shipped, and
    # what the live page showed for every deal. This is the check that would
    # have caught it, had the fixture not been written in dollars too.
    print("\n4b. No value renders as $0.00M (the v410 regression)")
    zeroed = [(vc, r["itd"], r["itd_display"])
              for vc, r in a["deals"].items()
              if r["itd"] not in (None, 0) and r["itd_display"] == "$0.00M"]
    chk("no non-zero ITD renders as $0.00M", not zeroed, str(zeroed))
    # Stated about the AFTER side. Against the v410 baseline this showed the
    # bug; against any later baseline the bug is already gone, and a "BEFORE it
    # was broken" assertion then fails for the wrong reason.
    chk("every entered ITD renders its real figure",
        all(r["itd_display"] != "$0.00M"
            for r in a["deals"].values() if r["itd"] not in (None, 0)),
        "baseline showed " + ("the $0.00M bug"
                              if all(r["itd_display"] == "$0.00M"
                                     for r in b["deals"].values()
                                     if r["itd"] not in (None, 0))
                              else "correct figures already"))
    chk("the 26Q1 reference figures render exactly",
        a["deals"]["P0000019"]["itd_display"] == "$5.87M"
        and a["groups"]["IND"]["itd_display"] == "$7.97M",
        "Giant 7 and its fund subtotal")

    print("\n5. Net ROE manual at every level, displayed with %")
    chk("per-deal Net ROE displays 4.4%",
        a["deals"]["P0000019"]["net_roe_display"] == "4.4%",
        "got " + repr(a["deals"]["P0000019"]["net_roe_display"])
        + "   before: " + repr(b["deals"]["P0000019"]["net_roe_display"]))
    chk("a negative Net ROE keeps its sign",
        a["deals"]["P0000067"]["net_roe_display"] == "-2.1%",
        "got " + repr(a["deals"]["P0000067"]["net_roe_display"]))
    chk("the PDF's n/a cell is still n/a, not 'pending entry'",
        a["deals"]["PCITWES"]["net_roe_display"] == "n/a",
        "got " + repr(a["deals"]["PCITWES"]["net_roe_display"]))
    for key, want in (("IND", "5.1%"), ("TGA22", "3.2%")):
        got = a["groups"][key].get("net_roe_display")
        chk("{} subtotal Net ROE is the typed {}".format(key, want),
            got == want,
            "got " + repr(got) + "   before: "
            + repr(b["groups"][key].get("net_roe_display")))
    chk("Portfolio Totals Net ROE is the typed 4.8%",
        a["total"].get("net_roe_display") == "4.8%",
        "got " + repr(a["total"].get("net_roe_display")))
    chk("every aggregate row exposes the key the UI saves against",
        a["groups"]["IND"].get("net_roe_vcode") == "__GROUP__:IND"
        and a["total"].get("net_roe_vcode") == "__TOTAL__"
        and a["excluding_dev"].get("net_roe_vcode") == "__EXCLUDING_DEV__")
    # Every Net ROE above is a stored entry. None of them is the sum or the
    # average of the deals beneath it, and this proves it: the deal-level
    # figures average 3.4%, and no aggregate reads that.
    chk("no aggregate Net ROE equals the average of its deals",
        abs(a["groups"]["IND"]["net_roe"] - (4.4)) > 1e-9
        and abs(a["total"]["net_roe"] - ((4.4 + 7.9 - 2.1) / 3)) > 1e-9)

    print("\n6. Excluding Development Deals")
    ex = a["excluding_dev"]
    #   non-development ITD = 5.87 + 2.10 + 1.25 = 9.22  (P0000067 removed)
    chk("ITD = non-development deals only ($9.22M)",
        ex.get("itd") is not None and abs(ex["itd"] - 9.22) < 1e-6,
        "got {}   before: {}".format(ex.get("itd"),
                                     b["excluding_dev"].get("itd")))
    chk("Brainerd's 0.90 is the difference from Portfolio Totals",
        abs((a["total"]["itd"] or 0) - (ex["itd"] or 0) - 0.90) < 1e-6)
    chk("Net ROE is the typed 6.3%, not a sum",
        ex.get("net_roe_display") == "6.3%",
        "got " + repr(ex.get("net_roe_display")) + "   before: "
        + repr(b["excluding_dev"].get("net_roe_display")))
    chk("exactly one development deal was removed",
        ex.get("excluded_count") == 1)

    passed = sum(checks)
    print("\n  {}/{} checks passed".format(passed, len(checks)))
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "capture":
        raise SystemExit(capture(sys.argv[2]))
    if len(sys.argv) >= 4 and sys.argv[1] == "report":
        raise SystemExit(report(sys.argv[2], sys.argv[3]))
    print(__doc__)
    raise SystemExit(2)
