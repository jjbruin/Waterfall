"""Guardrail: every Financial-subtab row FOOTS — Debt + Total Pref + Ptr Equity
= Total Cap, against the figures the row actually PRINTS.

THE DEFECT this was written for. Total Cap was re-footed onto `debt_isbs` (the
ISBS/current balance) while the Debt COLUMN prints `resolve_debt`'s choice,
which is the COMMITTED facility on a development deal (PDF footnote 6). On the
four dev deals where those two differ the printed row visibly did not add up,
by exactly `debt - debt_isbs`:

    JB Fair Park            77.4 + 30.0 + 3.9 = 111.3  vs  100.2 shown
    Jefferson Eastchase     53.9 + 29.4 + 14.7 = 98.0  vs   83.9
    Jefferson Addison Hts   44.0 + 24.8 + 17.5 = 86.3  vs   79.3
    Jefferson Waters Creek  51.7 + 23.0 + 14.3 = 89.0  vs   87.3

Runs the REAL committed ``assemble_financial`` against live data through the
same injected dependencies as ``snapshot_financial_pdf_check`` (whose ``build``
is imported rather than copied, so the two harnesses cannot wire the page
differently).

Read-only: GET only, via ``scripts/live_api.py`` (needs WF_TOKEN).

Usage
    set WF_TOKEN=<jwt>
    python scripts/snapshot_total_cap_foot_check.py                  # 2026-Q2
    python scripts/snapshot_total_cap_foot_check.py --quarter 2026-Q1
    python scripts/snapshot_total_cap_foot_check.py --json out.json  # for before/after
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import live_api as api                                          # noqa: E402
from snapshot_financial_pdf_check import build                  # noqa: E402
from flask_app.services.portfolio_snapshot_financial import (   # noqa: E402
    NA_LABEL,
)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

#: The four rows the report author reported as not footing. Named so a
#: regression on any of them is called out by name rather than buried in a
#: count, and so the fix cannot be declared good while one of them still fails.
REPORTED = {
    "P0000021": "JB Fair Park",
    "P0000085": "Jefferson Eastchase",
    "P0000077": "Jefferson Addison Heights",
    "P0000078": "Jefferson Waters Creek",
}

#: Non-dev rows carried as controls: they were footing before the change and
#: must still foot after it. Any operating deal would do; these are picked to
#: span funds and to include the two deals whose Debt is n/a.
CONTROLS = {
    "P0000019": "Giant 7",
    "P0000018": "Evergreen Plaza",
    "P0000066": "Pegasus Life Storage",      # Debt n/a, real 0.0
    "PCITWES": "City West",                  # Debt n/a, foreclosed
}

#: Dollars. The columns print to one decimal in $M, so anything under a dollar
#: is float noise, not a footing failure.
TOL = 1.0


def M(v):
    return "     —" if v is None else f"{v / 1e6:>6,.1f}"


def foot_check(r):
    """Does this row add up against WHAT IT PRINTS?

    The printed Debt is ``debt_display``: the raw ``debt`` on an ordinary row,
    the literal "n/a" where the cell does not apply. An n/a Debt contributes
    ZERO to the footing test, which is the same treatment ``build_row`` gives
    it when it re-foots Total Cap — the whole point of the n/a is that the
    figure stays off the page, so a row reading "n/a | 3.6 | 2.4 | 6.0" foots.
    """
    printed = r.get("debt_display")
    debt_na = printed == NA_LABEL
    legs = [0.0 if debt_na else r.get("debt"),
            r.get("total_pref"), r.get("ptr_equity")]
    tc = r.get("total_cap")
    if tc is None or any(v is None for v in legs):
        # Not a failure — an absent leg is reported as such. A row with an
        # unknown component cannot be asked to add up, and fabricating a zero
        # to make it testable is exactly what the assembly refuses to do.
        return None, sum(v for v in legs if v is not None), None
    total = sum(legs)
    return abs(total - tc) <= TOL, total, total - tc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--investor", default="TGAM")
    ap.add_argument("--quarter", default="2026-Q2")
    ap.add_argument("--json", help="write the per-row figures here")
    args = ap.parse_args()

    ti = api.token_info()
    print(f"LIVE  token={ti['username']} ({ti['hours_left']}h left)  "
          f"build={api.get('/api/data/version').get('version')}")
    print(f"real assemble_financial for {args.investor} {args.quarter}\n")

    _, out = build(args.investor, args.quarter)
    rows = [r for g in (out.get("groups") or {}).values()
            for r in (g.get("deals") or [])]

    # ── the diagnosis: what does Total Cap actually contain? ──────────────
    print("PER-ROW FOOTING  (all figures $M)")
    print(f"{'deal':<28} {'dev':>3} {'Debt(col)':>10} {'in TotCap':>10} "
          f"{'committed':>10} {'Pref':>7} {'PtrEq':>7} {'TotCap':>8} "
          f"{'sum legs':>9} {'gap':>7} {'gap==col-isbs':>14}")
    print("-" * 128)

    failures, absent, dump = [], [], []
    for r in sorted(rows, key=lambda x: (not x["is_dev"], x["name"])):
        ok, total, gap = foot_check(r)
        debt, isbs = r.get("debt"), r.get("debt_isbs")
        # The hypothesis under test: the row misses by the difference between
        # the Debt basis the column prints and the one Total Cap was built on.
        basis_gap = (debt - isbs) if (debt is not None
                                      and isbs is not None) else None
        explained = ("—" if (gap is None or basis_gap is None)
                     else ("yes" if abs(gap - basis_gap) <= TOL else "NO"))
        mark = "" if ok else ("  ??" if ok is None else "  << FAILS")
        # The Debt cell prints what the ROW prints — the literal "n/a" where
        # the column is suppressed, so the table reads as the page does.
        debt_cell = (f"{NA_LABEL:>6}" if r.get("debt_display") == NA_LABEL
                     else M(debt))
        print(f"{r['name'][:28]:<28} {'Y' if r['is_dev'] else '':>3} "
              f"{debt_cell}{M(isbs)}{M(r.get('debt_orig'))}"
              f"{M(r.get('total_pref'))}{M(r.get('ptr_equity'))}"
              f"{M(r.get('total_cap'))}{M(total)}{M(gap)}"
              f"{explained:>14}{mark}")

        dump.append({k: r.get(k) for k in (
            "vcode", "name", "is_dev", "debt", "debt_display", "debt_isbs",
            "debt_orig", "debt_basis", "total_pref", "ptr_equity",
            "total_cap", "total_cap_funded_basis")}
            | {"sum_legs": total, "gap": gap, "foots": ok})
        if ok is False:
            failures.append(r)
        elif ok is None:
            absent.append(r)

    # ── scoring ───────────────────────────────────────────────────────────
    print(f"\n{'-' * 128}")
    testable = [d for d in dump if d["foots"] is not None]
    passed = [d for d in testable if d["foots"]]
    print(f"FOOTING   {len(passed)}/{len(testable)} rows foot"
          f"  (tolerance ${TOL:,.0f})")
    if absent:
        print(f"          {len(absent)} row(s) not testable (a leg is None): "
              + ", ".join(r["name"] for r in absent))

    print("\nTHE FOUR REPORTED ROWS")
    for vc, name in REPORTED.items():
        d = next((x for x in dump if x["vcode"] == vc), None)
        if not d:
            print(f"  {name:<28} NOT ON THE PAGE this quarter")
            continue
        state = ("FOOTS" if d["foots"] else
                 "not testable" if d["foots"] is None else "FAILS")
        print(f"  {name:<28} {state:<13} "
              f"Debt {M(d['debt'])} + Pref {M(d['total_pref'])} + PtrEq "
              f"{M(d['ptr_equity'])} = {M(d['sum_legs'])} vs TotCap "
              f"{M(d['total_cap'])}  (gap {M(d['gap'])})")

    print("\nNON-DEV CONTROLS")
    for vc, name in CONTROLS.items():
        d = next((x for x in dump if x["vcode"] == vc), None)
        if not d:
            print(f"  {name:<28} NOT ON THE PAGE this quarter")
            continue
        state = ("FOOTS" if d["foots"] else
                 "not testable" if d["foots"] is None else "FAILS")
        print(f"  {name:<28} {state:<13} "
              f"Debt {M(d['debt'])} + Pref {M(d['total_pref'])} + PtrEq "
              f"{M(d['ptr_equity'])} = {M(d['sum_legs'])} vs TotCap "
              f"{M(d['total_cap'])}  (gap {M(d['gap'])})")

    # Subtotals have to foot too, or the fix just moves the disagreement one
    # row down: the column totals are sums of the per-row fields, so Total Cap
    # can only tie if every leg beneath it did.
    print("\nSUBTOTALS AND TOTALS")
    agg = [(g["label"], g["subtotal"]) for g in (out.get("groups") or {}).values()]
    for extra in ("total", "excluding_dev"):
        s = out.get(extra)
        if isinstance(s, dict):
            agg.append((s.get("label") or extra, s))
    sub_fail = 0
    for label, s in agg:
        legs = [s.get("debt"), s.get("total_pref"), s.get("ptr_equity")]
        tc = s.get("total_cap")
        if tc is None or any(v is None for v in legs):
            print(f"  {label[:44]:<44} not testable")
            continue
        tot = sum(legs)
        ok = abs(tot - tc) <= TOL * len(rows)
        sub_fail += 0 if ok else 1
        print(f"  {label[:44]:<44} {M(tot)} vs {M(tc)}  gap {M(tot - tc)}"
              f"{'' if ok else '   << FAILS'}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(dump, fh, indent=1)
        print(f"\nwrote {args.json}")

    bad = len(testable) - len(passed)
    print(f"\n{'PASS' if not (bad or sub_fail) else 'FAIL'} — "
          f"{bad} row(s) and {sub_fail} aggregate(s) do not foot")
    return 1 if (bad or sub_fail) else 0


if __name__ == "__main__":
    sys.exit(main())
