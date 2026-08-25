"""Guardrail — Operating subtotals aggregate only what the page shows.

Runs the REAL committed ``assemble_operating`` against LIVE data and checks the
three fixes of 2026-08-25 together, because they interact:

  1. A cell reading "n/a" contributes nothing to the row below it. This is the
     one that moves numbers: Portfolio U/W YE NOI printed 123.9M over visible
     cells that summed to about 104M, because ten suppressed development rows
     kept feeding the total.
  2. An unpopulated At Close NOI is None (em dash), not the One Pager's default
     0 printed as a measured "0.0".
  3. A deal owned for less than one quarter reads n/a in every metric column.

The load-bearing assertion is SUBTOTAL == SUM OF VISIBLE CELLS, checked on every
fund total and the portfolio total from the rows themselves rather than against
a transcribed constant. That is the property the page has to have; the PDF
comparison below is corroboration, and it carries the vintage caveats already
documented for Projected YE.

One Pager payloads are cached to disk (--refresh to re-pull) because there are
35 of them and they dominate the runtime.

    python scripts/snapshot_operating_subtotal_check.py [--refresh]
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

QUARTER = "2026-Q1"
INV = "TGAM"
CACHE = os.path.join(os.environ.get("CLAUDE_JOB_DIR", os.path.expanduser("~")),
                     "tmp", "op_payloads_26q1.json")
#: The pre-fix bundle, captured from live before any of this landed.
BEFORE = os.path.join(os.environ.get("CLAUDE_JOB_DIR", os.path.expanduser("~")),
                      "tmp", "live26q1.json")

#: PDF page 3, portfolio total row. At Close is EXCLUDED from the pass/fail set:
#: the published 62.7 is 22.7 below the sum of its own five printed fund
#: subtotals (85.4), a spreadsheet-range error in the source document that was
#: diagnosed and accepted earlier. Reproducing it would be reproducing a bug.
PDF_PORTFOLIO = {"uw_ye": 104.6, "projected_ye": 95.7}
PDF_AT_CLOSE_PRINTED = 62.7
PDF_AT_CLOSE_SUBTOTAL_SUM = 85.4

#: Deals the reference PDF withholds entirely under "recent acquisition".
EXPECT_INSUFFICIENT = {"P0000116": "Plaza Del Mar", "P0000118": "Hanestowne"}
#: Dev deals whose NOI is un-suppressed and so must STILL feed the subtotals.
EXPECT_FEEDING = {"P0000078": "Jefferson Waters Creek",
                  "P0000066": "Pegasus Life Storage"}
#: No at_close_noi row and no December Projected IS to scan — every column
#: unpopulated, so every cell must be an em dash and none may print "0.0".
EXPECT_UNPOPULATED = {"PCITWES": "City West"}

_NOI = ("at_close", "uw_ye", "projected_ye")


def _fetch_payloads(vcodes, refresh):
    import live_api as api
    cache = {}
    if os.path.exists(CACHE) and not refresh:
        cache = json.load(open(CACHE, encoding="utf-8"))
    missing = [v for v in vcodes if v not in cache]
    for i, vc in enumerate(missing, 1):
        print(f"    fetching One Pager {i}/{len(missing)}  {vc}", flush=True)
        try:
            cache[vc] = api.get(f"/api/financials/{vc}/one-pager",
                                params={"quarter": QUARTER})
        except Exception as exc:
            print(f"      ! {exc}")
            cache[vc] = {}
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump(cache, open(CACHE, "w", encoding="utf-8"), default=str)
    return cache


def _resolve():
    """Step 1 population, via narrow per-entity relationship pulls."""
    import live_api as api
    import pandas as pd
    from flask_app.services.portfolio_snapshot_service import resolve_investor_deals

    inv = pd.DataFrame(api.get("/api/data/deals/all").get("deals") or [])
    seen, frontier, rows = set(), [INV], []

    def fetch(col, val):
        d = api.get("/api/data/tables/relationships/rows",
                    params={"page": 1, "page_size": 500, f"filter__{col}": val})
        return [r for r in (d.get("rows") or [])
                if str(r.get(col) or "").strip().upper() == val.upper()]

    while frontier:
        node = frontier.pop().upper()
        if node in seen:
            continue
        seen.add(node)
        kids = fetch("InvestorID", node)
        rows.extend(kids)
        for r in kids:
            child = str(r.get("InvestmentID") or "").strip().upper()
            if child:
                rows.extend(fetch("InvestmentID", child))
                if child not in seen:
                    frontier.append(child)
    return resolve_investor_deals(INV, QUARTER, pd.DataFrame(rows).drop_duplicates(),
                                  inv)


def main():
    refresh = "--refresh" in sys.argv
    import live_api as api
    from flask_app.services.portfolio_snapshot_operating import (
        assemble_operating, operating_subtotal, NA_LABEL,
        INSUFFICIENT_HISTORY_MONTHS, _visible_noi,
    )

    ti = api.token_info()
    print(f"LIVE token={ti['username']} ({ti['hours_left']}h left)  "
          f"build={api.get('/api/data/version').get('version')}")
    print(f"insufficient-history threshold: {INSUFFICIENT_HISTORY_MONTHS} months\n")

    resolved = _resolve()
    vcodes = [e["vcode"] for items in resolved["groups"].values() for e in items]
    vcodes += [f["vcode"] for f in resolved.get("flagged") or []]
    print(f"Step 1: {len(vcodes)} deals")
    payloads = _fetch_payloads(vcodes, refresh)

    out = assemble_operating(
        INV, QUARTER, resolved=resolved,
        one_pager_provider=lambda vc, q: payloads.get(vc) or {},
        comment_loader=lambda i, q: {})

    groups = out["groups"]
    flagged = out.get("ownership_flagged") or []
    flat = {r["vcode"]: r for rows in groups.values() for r in rows}
    flat.update({r["vcode"]: r for r in flagged})
    checks = []

    def check(label, ok):
        checks.append((label, bool(ok)))
        print(f"    [{'PASS' if ok else 'FAIL'}] {label}")

    # ── 1. THE LOAD-BEARING PROPERTY ──────────────────────────────────────
    print("=" * 100)
    print("SUBTOTAL == SUM OF VISIBLE CELLS   (every fund total and the portfolio total)")
    print("=" * 100)
    blocks = [(g, rows, out["subtotals"][g]) for g, rows in groups.items()]
    blocks.append(("PORTFOLIO",
                   [r for rows in groups.values() for r in rows] + flagged,
                   out["total"]))
    print(f"{'group':<34}{'column':<14}{'subtotal':>12}{'visible sum':>13}"
          f"{'rows fed':>10}{'of':>4}")
    print("-" * 100)
    for gname, rows, sub in blocks:
        for k in _NOI:
            vis = [_visible_noi(r, k) for r in rows]
            vis = [v for v in vis if v is not None]
            expect = sum(vis) if vis else None
            got = (sub.get("noi") or {}).get(k)
            ok = (expect is None and got is None) or (
                expect is not None and got is not None
                and abs(got - expect) < 0.01)
            print(f"{gname[:33]:<34}{k:<14}"
                  f"{('None' if got is None else f'{got/1e6:.2f}M'):>12}"
                  f"{('None' if expect is None else f'{expect/1e6:.2f}M'):>13}"
                  f"{len(vis):>10}{len(rows):>4}"
                  f"{'' if ok else '   <<< MISMATCH'}")
            checks.append((f"{gname} {k} == visible sum", ok))
        print()
    print(f"    {sum(1 for _, o in checks if o)}/{len(checks)} column identities hold\n")

    # ── 2. PORTFOLIO TOTAL vs THE PDF ─────────────────────────────────────
    print("=" * 100)
    print("PORTFOLIO TOTAL vs PDF page 3")
    print("=" * 100)
    before = None
    if os.path.exists(BEFORE):
        b = json.load(open(BEFORE, encoding="utf-8"))
        before = (b.get("subtabs") or {}).get("operating") or {}
    tot = out["total"]["noi"]
    for k, pdf in PDF_PORTFOLIO.items():
        now = (tot.get(k) or 0) / 1e6
        was = (((before or {}).get("total") or {}).get("noi") or {}).get(k)
        was = (was or 0) / 1e6 if was is not None else None
        line = (f"    {k:<14} before {('n/a' if was is None else f'{was:8.1f}M')}"
                f"   ->  after {now:8.1f}M   PDF {pdf:6.1f}M"
                f"   delta {now - pdf:+.1f}M")
        print(line)
        check(f"portfolio {k} within 1.5M of PDF", abs(now - pdf) <= 1.5)

    ac = (tot.get("at_close") or 0) / 1e6
    print(f"\n    at_close       after {ac:8.1f}M   PDF printed "
          f"{PDF_AT_CLOSE_PRINTED:.1f}M, but the PDF's own five fund subtotals "
          f"sum to {PDF_AT_CLOSE_SUBTOTAL_SUM:.1f}M")
    print("                   -> compared to the subtotal sum, not the printed "
          "total (known source-document range error)")
    check("portfolio at_close within 1.5M of the PDF's own subtotal sum",
          abs(ac - PDF_AT_CLOSE_SUBTOTAL_SUM) <= 1.5)

    # ── 3. THE THREE FIXES, PER DEAL ──────────────────────────────────────
    print("\n" + "=" * 100)
    print("PER-DEAL BEHAVIOUR")
    print("=" * 100)

    print("\n  Fix 3 — insufficient operating history (every metric n/a):")
    got_insuf = {vc: r for vc, r in flat.items() if r.get("insufficient_history")}
    for vc, name in EXPECT_INSUFFICIENT.items():
        r = flat.get(vc)
        if not r:
            check(f"{name} present", False)
            continue
        cells = ([r.get("econ_occ_display")]
                 + list((r.get("noi_display") or {}).values())
                 + [r.get("expected_growth_display"),
                    r.get("actual_growth_display")])
        print(f"      {name:<22} owned {r.get('months_owned')} mo   "
              f"cells: {cells}")
        check(f"{name} flagged insufficient_history",
              r.get("insufficient_history") is True)
        check(f"{name} shows n/a in all 6 metric cells",
              all(c == NA_LABEL for c in cells))
    check(f"exactly {len(EXPECT_INSUFFICIENT)} deals flagged insufficient "
          f"(got {sorted(got_insuf)})",
          set(got_insuf) == set(EXPECT_INSUFFICIENT))

    print("\n  Fix 2 — unpopulated NOI is an em dash, never a measured 0.0:")
    for vc, name in EXPECT_UNPOPULATED.items():
        r = flat.get(vc)
        if not r:
            check(f"{name} present", False)
            continue
        print(f"      {name:<22} raw noi {r['noi']}   display "
              f"{r.get('noi_display')}")
        check(f"{name} raw NOI is None in every column",
              all(v is None for v in (r["noi"] or {}).values()))
        check(f"{name} displays no zero", not any(
            v == 0 for v in (r.get("noi_display") or {}).values()))
    zeros = [r["name"] for r in flat.values()
             if any(v == 0 for v in (r["noi"] or {}).values())]
    check(f"no deal carries a literal 0.0 NOI anywhere (got {zeros})", not zeros)

    print("\n  Regression — un-suppressed dev NOI must STILL feed the subtotals:")
    for vc, name in EXPECT_FEEDING.items():
        r = flat.get(vc)
        if not r:
            check(f"{name} present", False)
            continue
        fed = [k for k in _NOI if _visible_noi(r, k) is not None]
        print(f"      {name:<22} is_dev={r['is_dev']}  exception="
              f"{r.get('dev_display_exception')}  feeds {fed}")
        check(f"{name} still contributes NOI", bool(fed))

    print("\n  Regression — dev suppression intact, raw metrics untouched:")
    devs = [r for r in flat.values() if r["is_dev"]]
    check(f"all {len(devs)} dev rows show n/a in every non-exempted column",
          all(v == NA_LABEL for r in devs
              for c in r["dev_suppressed_columns"]
              for v in ([r.get("econ_occ_display")] if c == "econ_occ"
                        else list((r.get("noi_display") or {}).values())
                        if c == "noi" else [r.get(f"{c}_display")])))
    check("raw metrics never carry the n/a literal (freeze-safety)",
          all(not isinstance(v, str)
              for r in flat.values()
              for v in (list((r["noi"] or {}).values())
                        + list((r["econ_occ"] or {}).values())
                        + [r["expected_growth"], r["actual_growth"]])
              if v is not None))

    # ── 4. BEFORE/AFTER on raw values ─────────────────────────────────────
    if before:
        print("\n" + "=" * 100)
        print("BEFORE/AFTER — which raw NOI values moved, and why")
        print("=" * 100)
        bflat = {r["vcode"]: r for rows in (before.get("groups") or {}).values()
                 for r in rows}
        bflat.update({r["vcode"]: r
                      for r in before.get("ownership_flagged") or []})
        moved = []
        for vc, r in flat.items():
            bn = (bflat.get(vc) or {}).get("noi") or {}
            for k in _NOI:
                a, b2 = (r["noi"] or {}).get(k), bn.get(k)
                if a != b2:
                    moved.append((r["name"], k, b2, a))
        for name, k, b2, a in moved:
            print(f"    {name[:30]:<32}{k:<14}{str(b2):>10}  ->  {str(a)}")
        # Only 0.0 -> None is legitimate. Any other movement means a metric was
        # changed, which this work was explicitly not allowed to do.
        illegit = [m for m in moved if not (m[2] == 0 and m[3] is None)]
        check(f"every raw NOI change is 0.0 -> None ({len(moved)} changes, "
              f"{len(illegit)} illegitimate)", not illegit)

    passed = sum(1 for _, ok in checks if ok)
    print("\n" + "=" * 100)
    print(f"  {passed}/{len(checks)} checks passed")
    print(f"  diagnostics: {out['diagnostics']}")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
