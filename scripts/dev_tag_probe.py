"""Read-only probe for the combined Portfolio Snapshot dev-tag change.

Answers, against LIVE:
  1. exactly which deals carry Lifecycle / Investment_Strategy "new construction"
  2. which deals are genuine development after the change
  3. the loan cells those dev deals show today, so "force Dev" has a baseline
  4. Pegasus's raw loan block, to justify the debt-free display

No writes, no asserts — evidence only.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd                                              # noqa: E402

from scripts import live_api                                     # noqa: E402

QUARTER = os.environ.get("WF_CHECK_QUARTER", "2026-Q1")


def page(table, params=None, size=500, cap=60):
    out, seen, n = [], set(), 1
    while n <= cap:
        d = live_api.get(f"/api/data/tables/{table}/rows",
                         params={"page": n, "page_size": size, **(params or {})})
        rows = d.get("rows") or []
        if not rows:
            break
        new = 0
        for r in rows:
            k = tuple(sorted((str(a), str(b)) for a, b in r.items()))
            if k not in seen:
                seen.add(k)
                out.append(r)
                new += 1
        if new == 0 or len(rows) < size:
            break
        n += 1
    return pd.DataFrame(out)


def main():
    ti = live_api.token_info()
    print(f"LIVE token={ti['username']} ({ti['hours_left']}h left)")
    inv = pd.DataFrame(live_api.get("/api/data/deals/all").get("deals") or [])
    print(f"deals: {len(inv)} rows; columns include "
          f"{[c for c in inv.columns if 'ifecycle' in c or 'trateg' in c]}")

    # ---- 1. who carries which classification value -----------------------
    print("\n" + "=" * 96)
    print("1. CLASSIFICATION VALUES ACROSS THE WHOLE FEED")
    print("=" * 96)
    for col in ("Investment_Strategy", "Lifecycle"):
        if col not in inv.columns:
            print(f"  {col}: COLUMN ABSENT")
            continue
        vc = inv[col].fillna("").astype(str).str.strip().str.lower()
        print(f"\n  {col} — {(vc != '').sum()}/{len(inv)} populated")
        for val, n in vc.value_counts().items():
            tag = "  <-- in DEV_STRATEGIES today" if val in (
                "development", "new construction") else ""
            print(f"    {val!r:<26} {n:>4}{tag}")
        for val in ("new construction", "development"):
            hit = inv[vc == val]
            if len(hit):
                print(f"    deals with {col}={val!r}:")
                for _, r in hit.iterrows():
                    print(f"       {r.get('vcode'):<12}{str(r.get('Investment_Name'))[:38]:<40}"
                          f"Year_Built={r.get('Year_Built')!r}")

    # ---- 2/3. dev deals and their loan cells ----------------------------
    from flask_app.services.portfolio_snapshot_service import resolve_investor_deals
    from flask_app.services.portfolio_snapshot_operating import resolve_strategy
    from flask_app.services.portfolio_snapshot_loan import assemble_loan
    from config import DEV_STRATEGIES, is_dev_deal

    print("\n" + "=" * 96)
    print(f"2. DEV CLASSIFICATION under the CURRENT working tree "
          f"DEV_STRATEGIES={sorted(DEV_STRATEGIES)}")
    print("=" * 96)

    rel = page("relationships")
    loans = page("loans")
    vals = page("valuations")

    def one_pager(vc, q):
        return live_api.get(f"/api/financials/{vc}/one-pager",
                            params={"quarter": q}) or {}

    def q_noi(vc, q):
        pp = (one_pager(vc, q) or {}).get("property_performance") or {}
        return (pp.get("noi") or {}).get("ytd_actual")

    seen_rows = {}
    for code in ("TGAM", "TIAA", "PSC3", "WRI"):
        try:
            resolved = resolve_investor_deals(code, QUARTER, rel, inv)
        except Exception as exc:
            print(f"  {code}: resolve failed {str(exc)[:70]}")
            continue
        try:
            ln = assemble_loan(code, QUARTER, resolved=resolved,
                               one_pager_provider=one_pager, loans=loans,
                               valuations=vals, inv=inv,
                               quarterly_noi_provider=q_noi)
        except Exception as exc:
            print(f"  {code}: assemble_loan failed {type(exc).__name__}: {exc}")
            continue
        for rows in (ln.get("groups") or {}).values():
            for r in rows:
                seen_rows.setdefault(r["vcode"], (code, r))
        for r in ln.get("ownership_flagged") or []:
            seen_rows.setdefault(r["vcode"], (code, r))
        print(f"  {code}: {len(ln.get('groups') or {})} groups, "
              f"diag dev={ln['diagnostics'].get('dev')} "
              f"dev_no_data={ln['diagnostics'].get('dev_no_data')}")

    print(f"\n  {len(seen_rows)} distinct deals seen across those reports")

    def d(v):
        if v is None:
            return "—"
        if isinstance(v, str):
            return v
        return f"{v:,.4f}" if abs(v) < 100 else f"{v:,.0f}"

    print("\n" + "=" * 96)
    print("3. LOAN CELLS — every DEV deal (what 'force Dev' must produce)")
    print("=" * 96)
    print(f"  {'vcode':<11}{'name':<30}{'loans':>6}{'debt':>13}"
          f"{'LTV disp':>11}{'DSCR disp':>11}{'DY disp':>11}{'no_data':>8}")
    devs = [(vc, r) for vc, (_, r) in seen_rows.items() if r.get("is_dev")]
    for vc, r in sorted(devs):
        print(f"  {vc:<11}{str(r.get('name'))[:28]:<30}{r.get('loan_count'):>6}"
              f"{d(r.get('debt')):>13}{d(r.get('ltv_display')):>11}"
              f"{d(r.get('ytd_dscr_display')):>11}"
              f"{d(r.get('debt_yield_display')):>11}"
              f"{str(r.get('dev_no_data')):>8}")
    print(f"\n  dev deals: {len(devs)}")
    nd = [vc for vc, r in devs if r.get("dev_no_data")]
    print(f"  dev deals currently dev_no_data (would flip n/a -> 'Dev'): {nd}")
    strings = [(vc, r.get("rate_display"), r.get("maturity_display"))
               for vc, r in sorted(devs)]
    print("\n  Rate / Maturity for dev deals (must stay REAL):")
    for vc, rt, mt in strings:
        print(f"    {vc:<11}rate={str(rt):<16}maturity={mt}")

    # ---- 4. the debt-free candidates ------------------------------------
    print("\n" + "=" * 96)
    print("4. DEBT-FREE CANDIDATES — non-dev, no loan record, debt 0/None")
    print("=" * 96)
    print(f"  {'vcode':<11}{'name':<30}{'is_dev':>7}{'loans':>6}{'debt':>13}"
          f"{'isbs':>13}{'orig':>13}{'DSCR':>9}")
    cands = []
    for vc, (_, r) in sorted(seen_rows.items()):
        debt = r.get("debt")
        if r.get("loan_count") == 0 and (debt is None or abs(debt) < 1e-9):
            cands.append(vc)
            print(f"  {vc:<11}{str(r.get('name'))[:28]:<30}"
                  f"{str(r.get('is_dev')):>7}{r.get('loan_count'):>6}"
                  f"{d(debt):>13}{d(r.get('isbs_debt')):>13}"
                  f"{d(r.get('orig_loan_amt')):>13}"
                  f"{d(r.get('ytd_dscr')):>9}")
    print(f"\n  candidates: {cands}")
    print("  ^ if this list is exactly ['P0000066'] a data-driven debt-free rule "
          "is safe;\n    if it is longer, the rule must stay keyed by vcode.")

    peg = (seen_rows.get("P0000066") or (None, {}))[1]
    print("\n  PEGASUS raw loan block:")
    for k in ("is_dev", "strategy", "loan_count", "debt", "debt_basis",
              "isbs_debt", "orig_loan_amt", "valuation", "ltv", "ltv_display",
              "ytd_dscr", "ytd_dscr_display", "debt_yield",
              "debt_yield_display", "rate_display", "maturity_display",
              "dev_no_data"):
        print(f"    {k:<22}{peg.get(k)!r}")
    print(f"    flags: {peg.get('flags')}")


if __name__ == "__main__":
    main()
