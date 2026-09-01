"""SAFETY CHECK (read-only): who would the "no 2015-12-31 row -> 0" rule zero?

Proposed rule: a deal with NO 2015-12-31 row in its Projected IS
(``isbs_projected_is``, vSource='Projected IS') gets At-Close NOI = 0. Deals
that DO have one are untouched.

This script does not implement the rule. It measures the population it would
hit, against live, and prints the full before/after so the blast radius is
seen before anything changes.

Two sources, both live:
  * the set of vcodes with a 2015-12-31 Projected IS row
  * each deal's CURRENT At-Close NOI, from the real One Pager endpoint — the
    same number the Snapshot Operating subtab and the One Pager print

NOTE on the rows endpoint: ``filter__dtEntry`` is CASE-SENSITIVE on the column
name and silently returns the WHOLE table on a mismatch (108,963 rows instead
of 626). The filtered total is asserted below rather than trusted.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import live_api                                    # noqa: E402

QUARTER = os.environ.get("WF_CHECK_QUARTER", "2026-Q1")


def page_all(table, params=None, page_size=500, cap=400):
    """Every row of a filtered query, de-duplicated.

    De-duplication is not paranoia: the endpoint's OFFSET paging can repeat
    rows across pages, which would inflate any count taken from it.
    """
    out, seen, page = [], set(), 1
    while page <= cap:
        d = live_api.get(f"/api/data/tables/{table}/rows",
                         params={"page": page, "page_size": page_size,
                                 **(params or {})})
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
        if new == 0 or len(rows) < page_size:
            break
        page += 1
    return out, d.get("total")


def col(row, name):
    for k in row:
        if k.lower() == name.lower():
            return row[k]
    return None


def main():
    print("SAFETY CHECK — 'no 2015-12-31 Projected IS row -> At-Close NOI = 0'")
    print(f"quarter {QUARTER}   token {live_api.token_info()['username']}")
    print("=" * 100)

    # ── 1. who has a 2015-12-31 Projected IS row ──────────────────────────
    rows, total = page_all("isbs_projected_is",
                           {"filter__dtEntry": "2015-12-31"})
    unfiltered = live_api.get("/api/data/tables/isbs_projected_is/rows",
                              params={"page": 1, "page_size": 1}).get("total")
    print(f"\nisbs_projected_is: {unfiltered:,} rows total, "
          f"{total:,} on 2015-12-31 ({len(rows):,} fetched, de-duplicated)")
    assert total and total < unfiltered, (
        "the date filter did not apply — the column name case is wrong and the "
        "endpoint returned the whole table")

    has2015 = {str(col(r, "vcode") or "").strip().upper() for r in rows}
    has2015.discard("")
    print(f"deals WITH a 2015-12-31 Projected IS row: {len(has2015)}")

    # ── 2. the deal population, and which have Projected IS data at all ───
    deals, _ = page_all("deals", {"sort": "vcode", "order": "asc"})
    dmap = {str(col(r, "vcode") or "").strip().upper(): r for r in deals}
    print(f"deals table: {len(dmap)} deals")

    # ── 3. current At-Close NOI, from the real One Pager ──────────────────
    print(f"\nreading current At-Close NOI from the One Pager for "
          f"{len(dmap)} deals (live, read-only)…")
    cur = {}
    errs = {}
    for i, vc in enumerate(sorted(dmap), 1):
        try:
            d = live_api.get(f"/api/financials/{vc}/one-pager",
                             params={"quarter": QUARTER}) or {}
            pp = d.get("property_performance") or {}
            cur[vc] = {
                "noi": (pp.get("noi") or {}).get("at_close"),
                "rev": (pp.get("revenue") or {}).get("at_close"),
                "exp": (pp.get("expenses") or {}).get("at_close"),
                "dscr": (pp.get("dscr") or {}).get("at_close"),
            }
        except Exception as exc:
            errs[vc] = str(exc)[:70]
        if i % 25 == 0:
            print(f"   … {i}/{len(dmap)}")
    print(f"   read {len(cur)}, {len(errs)} errors")

    # ── 4. the before/after ───────────────────────────────────────────────
    def name(vc):
        return str(col(dmap[vc], "Investment_Name") or "")[:32]

    def life(vc):
        return str(col(dmap[vc], "Lifecycle") or "")[:16]

    def yb(vc):
        return str(col(dmap[vc], "Year_Built") or "")[:12]

    def nz(v):
        return v is not None and abs(float(v)) > 0.005

    would_zero = [vc for vc in sorted(cur) if vc not in has2015]
    unchanged = [vc for vc in sorted(cur) if vc in has2015]

    print("\n" + "=" * 100)
    print(f"A. WOULD BE SET TO 0 — {len(would_zero)} deals with NO 2015-12-31 row")
    print("=" * 100)
    hdr = (f"{'vcode':<10}{'name':<34}{'Lifecycle':<17}{'Year_Built':<13}"
           f"{'At-Close NOI (before)':>22}  -> after")
    print(hdr)
    changing, already_zero = [], []
    for vc in would_zero:
        v = cur[vc]["noi"]
        (changing if nz(v) else already_zero).append(vc)
    for label, pool in (("  -- currently NON-ZERO: these are the real changes --", changing),
                        ("  -- currently zero / null: no visible change --", already_zero)):
        print(label)
        for vc in pool:
            v = cur[vc]["noi"]
            shown = "None" if v is None else f"{float(v):,.0f}"
            print(f"{vc:<10}{name(vc):<34}{life(vc):<17}{yb(vc):<13}"
                  f"{shown:>22}  ->      0")

    print("\n" + "=" * 100)
    print(f"B. UNCHANGED — {len(unchanged)} deals WITH a 2015-12-31 row")
    print("=" * 100)
    nonzero_kept = [vc for vc in unchanged if nz(cur[vc]["noi"])]
    zero_kept = [vc for vc in unchanged if not nz(cur[vc]["noi"])]
    print(f"  {len(nonzero_kept)} keep a non-zero At-Close NOI, "
          f"{len(zero_kept)} are already zero/null")
    print(f"{'vcode':<10}{'name':<34}{'Lifecycle':<17}{'Year_Built':<13}"
          f"{'At-Close NOI (kept)':>22}")
    for vc in nonzero_kept:
        print(f"{vc:<10}{name(vc):<34}{life(vc):<17}{yb(vc):<13}"
              f"{float(cur[vc]['noi']):>22,.0f}")

    # ── 5. the two the user named ─────────────────────────────────────────
    print("\n" + "=" * 100)
    print("C. THE TWO NAMED TARGETS")
    print("=" * 100)
    for vc, want in (("P0000100", -59333), ("P0000077", -115440)):
        if vc not in cur:
            print(f"  {vc}: NOT READ")
            continue
        v = cur[vc]["noi"]
        print(f"  {vc} {name(vc):<34} At-Close NOI = "
              f"{('None' if v is None else f'{float(v):,.0f}'):>14}   "
              f"has 2015 row: {vc in has2015}   "
              f"-> {'ZEROED by the rule' if vc not in has2015 else 'NOT CAUGHT'}"
              f"   (expected ~{want:,})")

    # ── 6. the risk: non-dev deals that would lose a real figure ──────────
    print("\n" + "=" * 100)
    print("D. RISK — deals losing a NON-ZERO At-Close that are NOT development")
    print("=" * 100)
    from config import is_dev_deal
    from flask_app.services.portfolio_snapshot_operating import resolve_strategy
    risky = []
    for vc in changing:
        strat, src = resolve_strategy({
            "investment_strategy": col(dmap[vc], "Investment_Strategy"),
            "strategy": col(dmap[vc], "Lifecycle")})
        if not is_dev_deal(strat):
            risky.append((vc, strat, cur[vc]["noi"]))
    if not risky:
        print("  none — every deal losing a real figure is a development deal")
    for vc, strat, v in risky:
        print(f"  {vc:<10}{name(vc):<34}strategy={strat!r:<20}"
              f"At-Close NOI {float(v):>16,.0f}  <-- would lose a real value")

    # ── 7. THE NARROWED RULE — dev AND no 2015 row ───────────────────────
    print("\n" + "=" * 100)
    print("E. NARROWED RULE AS SHIPPED — development AND no 2015-12-31 row")
    print("=" * 100)
    dev_of = {}
    for vc in cur:
        s, _ = resolve_strategy({
            "investment_strategy": col(dmap[vc], "Investment_Strategy"),
            "strategy": col(dmap[vc], "Lifecycle")})
        dev_of[vc] = (is_dev_deal(s), s)

    narrowed = [vc for vc in would_zero if dev_of[vc][0]]
    spared = [vc for vc in changing if not dev_of[vc][0]]
    narrowed_changing = [vc for vc in narrowed if nz(cur[vc]["noi"])]

    print(f"  zeroed by the BROAD rule (no 2015 row)      {len(would_zero)}"
          f"   ({len(changing)} visibly change)")
    print(f"  zeroed by the NARROWED rule (+ development) {len(narrowed)}"
          f"   ({len(narrowed_changing)} visibly change)")
    print(f"  SPARED by the narrowing                     {len(spared)}\n")

    print("  -- ZEROED (development, no 2015 row, currently non-zero) --")
    print(f"  {'vcode':<10}{'name':<34}{'strategy':<18}"
          f"{'At-Close NOI':>16}  -> after")
    for vc in narrowed_changing:
        print(f"  {vc:<10}{name(vc):<34}{dev_of[vc][1][:16]:<18}"
              f"{float(cur[vc]['noi']):>16,.0f}  ->      0")
    print("\n  -- ZEROED but already 0/null (no visible change) --")
    for vc in [v for v in narrowed if not nz(cur[v]["noi"])]:
        print(f"  {vc:<10}{name(vc):<34}{dev_of[vc][1][:16]:<18}"
              f"{'0':>16}  ->      0")

    print("\n  -- SPARED by the narrowing: keep their real values --")
    print(f"  {'vcode':<10}{'name':<34}{'strategy':<18}{'At-Close NOI kept':>18}")
    for vc in spared:
        print(f"  {vc:<10}{name(vc):<34}{dev_of[vc][1][:16]:<18}"
              f"{float(cur[vc]['noi']):>18,.0f}")

    non_dev_zeroed = [vc for vc in narrowed if not dev_of[vc][0]]
    print(f"\n  non-development deals zeroed by the narrowed rule: "
          f"{len(non_dev_zeroed)}  {non_dev_zeroed if non_dev_zeroed else '(none)'}")

    # Pegasus under the pending classification fix
    import config
    keep = set(config.DEV_STRATEGIES)
    try:
        config.DEV_STRATEGIES = {"development"}
        after_fix = [vc for vc in would_zero
                     if is_dev_deal(resolve_strategy({
                         "investment_strategy": col(dmap[vc], "Investment_Strategy"),
                         "strategy": col(dmap[vc], "Lifecycle")})[0])]
        drops = sorted(set(narrowed) - set(after_fix))
        print(f"\n  once 'new construction' leaves DEV_STRATEGIES, these drop "
              f"out of the gate: {drops}")
        for vc in drops:
            v = cur[vc]["noi"]
            print(f"    {vc:<10}{name(vc):<34}would keep "
                  f"{('None' if v is None else f'{float(v):,.0f}')}")
    finally:
        config.DEV_STRATEGIES = keep

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"  deals read                          {len(cur)}")
    print(f"  have a 2015-12-31 row (unchanged)   {len(unchanged)}")
    print(f"  no 2015-12-31 row (-> 0)            {len(would_zero)}")
    print(f"     of which visibly change          {len(changing)}")
    print(f"     of which already 0/null          {len(already_zero)}")
    print(f"  NON-dev deals losing a real value   {len(risky)}")
    if errs:
        print(f"  one-pager read errors               {len(errs)}: "
              f"{sorted(errs)[:6]}")


if __name__ == "__main__":
    main()
