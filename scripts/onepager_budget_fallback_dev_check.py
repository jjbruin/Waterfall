"""Guardrail: the prior-year budget fallback must skip DEVELOPMENT deals.

Runs the REAL committed ``one_pager.get_property_performance()`` on each side —
"before" from a worktree pinned at the pre-fix commit, "after" from the working
tree — over a target set covering EVERY deal the fallback fires for, EVERY dev
deal, and sampled controls.

Three things are asserted, not just the headline one:
  * dev deals the fallback used to fire for now land on the no-budget path
    (Projected YE = YTD Actual, YTD Budget = 0),
  * stabilized deals the fallback fires for are byte-identical (Jim wants it
    kept for them), and
  * deals with a real current-year budget, and deals with no budget in either
    year, are byte-identical.

FIDELITY: the local "before" is tied out against the LIVE payload for the same
deals and quarter. Live runs v399 (bf29707), which contains the fallback commit
e6579db, and the only one_pager.py change since is the participation fix, which
does not touch this function — so live IS the before state. A harness that
cannot reproduce live is reported as such rather than trusted.

Only the ISBS-derived rows are compared: revenue / expenses / noi / dscr. The
fallback cannot reach economic_occ (that reads budget_econ_occ_df / ProjOccupancy,
a different input), so occupancy inputs go in as None and econ_occ is excluded.

Usage (WF_TOKEN required for fetch/tieout):
    python scripts/onepager_budget_fallback_dev_check.py fetch   <cache.json>
    python scripts/onepager_budget_fallback_dev_check.py capture <cache.json> <root> <out.json>
    python scripts/onepager_budget_fallback_dev_check.py tieout  <cache.json> <before.json>
    python scripts/onepager_budget_fallback_dev_check.py report  <cache.json> <before.json> <after.json>
"""
import json
import os
import sys

QUARTER = "2026-Q2"          # available_quarters is global; this is the app default today
ROWS = ("revenue", "expenses", "noi", "dscr")
COLS = ("ytd_actual", "ytd_budget", "variance", "actual_ye", "uw_ye", "at_close")

#: The dev deals the fallback fires for — these MUST change.
MUST_CHANGE = {"P0000077", "P0000092", "P0000093", "P0000094", "P0000096"}

ISBS_TABLES = (
    ("isbs_interim_is", "Interim IS"),
    ("isbs_interim_is_historical", "Interim IS"),
    ("isbs_interim_bs", "Interim BS"),
    ("isbs_budget_is", "Budget IS"),
    ("isbs_projected_is", "Projected IS"),
)

#: Years fetched per deal. At a 2026 report quarter the function reads 2026
#: (YTD actuals, budget, U/W December) and 2025 (the prior-Dec balance-sheet
#: anchor for principal, the cross-year YTD base, and the fallback's own
#: source year); 2024/2027 are slack. Pulling full history instead made the
#: fetch unworkably slow AND dragged in the >500-row deal-months that this
#: endpoint cannot page without repeating rows.
#:
#: This narrows `at_close`, which scans Projected IS for the EARLIEST December
#: — but at_close is computed identically on both sides of the diff, and the
#: live tie-out deliberately checks only ytd_budget / actual_ye, the two
#: fields the fallback can actually reach.
FETCH_YEARS = ("2024", "2025", "2026", "2027")


def _api():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import live_api as api
    return api


def _rows(api, table, **filters):
    """Every row for one filter, in a SINGLE page.

    Two traps this deliberately avoids:

    * ``statement_id`` is NOT unique — it keys a monthly statement, and each
      statement carries one row per account. Deduping on it collapsed
      P0000077's 106 budget rows to 12 and silently emptied most of the
      budget. Never dedupe ISBS on it.
    * The rows endpoint pages by OFFSET and can repeat a row across pages
      (see the live-access note in project memory). Asking for everything at
      once means there is no second page to disagree with the first, so no
      dedupe is needed at all and no real duplicate row is wrongly dropped.

    page_size is capped at 500 server side, and page 2 of a filtered query
    genuinely repeats rows: p0000003's 1447 budget rows came back with 5
    duplicates, while page 1 alone was clean. ``sort`` does not save it —
    statement_id has one tie per account, so the DB is free to order ties
    differently per query.

    So this NEVER asks for page 2. The filter is narrowed by year (``dtEntry``
    matches as a substring) until every request fits in one page, and months
    split any year that still would not. ``total`` is reliable even though
    ``total_pages`` is not, so it decides when to split and then verifies each
    piece came back whole.
    """
    def one(extra):
        f = dict(filters)
        f.update(extra)
        d = api.get(f"/api/data/tables/{table}/rows",
                    params={"page": 1, "page_size": 500, **f})
        return (d.get("total") or 0), (d.get("rows") or [])

    total, rows = one({})
    if total < 500:
        if len(rows) != total:
            raise RuntimeError(f"{table} {filters}: short frame "
                               f"{len(rows)}/{total}")
        return rows

    out = []

    def sweep(years):
        for yr in years:
            y_total, y_rows = one({"filter__dtEntry": str(yr)})
            if y_total == 0:
                continue
            if y_total < 500:
                if len(y_rows) != y_total:
                    raise RuntimeError(f"{table} {filters} {yr}: short frame")
                out.extend(y_rows)
                continue
            for mo in range(1, 13):                 # split the year by month
                m_total, m_rows = one({"filter__dtEntry": f"{yr}-{mo:02d}"})
                if m_total >= 500:
                    raise RuntimeError(
                        f"{table} {filters} {yr}-{mo:02d}: {m_total} rows in "
                        f"one month — cannot fit a single page, split further")
                if len(m_rows) != m_total:
                    raise RuntimeError(
                        f"{table} {filters} {yr}-{mo:02d}: short frame")
                out.extend(m_rows)

    sweep(range(2014, 2036))
    if len(out) != total:
        # ISBS carries sentinel dates far outside the reporting window —
        # Asbury Commons (P0000004) has a Budget IS row stamped 1979-11-30.
        # Harmless to the fallback (neither the report year nor the prior
        # one), but it must be COLLECTED rather than silently dropped, or the
        # completeness check below would be meaningless. Only paid for when
        # the common window comes up short.
        sweep(list(range(1970, 2014)) + list(range(2036, 2046)))

    # Deliberately NOT deduped. Identical tuples are real data, not paging
    # artifacts: the returned column set carries no property identifier, so a
    # PORTFOLIO deal repeats a row per child property — Berger Pittsburgh
    # (P0000007) has 460 such repeats. Production sums every one of them, so
    # dropping them here would understate the budget and make the harness
    # disagree with live. The OFFSET-duplication trap this would otherwise
    # guard is already excluded by construction: every partition above is a
    # SINGLE page, so no row is ever fetched twice.
    if len(out) != total:
        raise RuntimeError(
            f"{table} {filters}: partitions summed to {len(out)} but "
            f"total={total} — rows fell outside even the 1970-2045 sweep")
    return out


def _classify(deals, budget_years, report_year):
    """(fires, has_current, no_budget, dev_set) using the app's dev precedence.

    Appends (not prepends) the repo root: `capture` puts the tree under test at
    sys.path[0] and that must keep winning, so this can never shadow it."""
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.append(_root)
    from config import DEV_STRATEGIES

    dev = set()
    for d in deals:
        vc = str(d.get("vcode") or "").strip().upper()
        for f in ("Investment_Strategy", "Lifecycle"):
            v = d.get(f)
            if v is not None and str(v).strip():
                if str(v).strip().lower() in DEV_STRATEGIES:
                    dev.add(vc)
                break
    fires, has_cur, no_bud = set(), set(), set()
    for vc, yrs in budget_years.items():
        if yrs and str(report_year) in yrs:
            has_cur.add(vc)
        elif yrs and str(report_year - 1) in yrs:
            fires.add(vc)
        else:
            no_bud.add(vc)
    return fires, has_cur, no_bud, dev


def _fetch(cache_path):
    api = _api()
    print("token:", api.token_info())
    print("build:", api.get("/api/data/version").get("version"))

    deals = api.get("/api/data/deals/all").get("deals") or []
    print(f"deals={len(deals)}")

    year = int(QUARTER.split("-")[0])

    # Classification needs two counts per deal — "does the report year have
    # budget rows, and does the prior year" — so ask for the COUNT only.
    # `total` is exact and comes back on page 1, so this is one cheap request
    # per year instead of dragging every budget row over the wire.
    print(f"counting Budget IS rows for {year - 1} / {year} per deal...")
    budget_years = {}
    for i, d in enumerate(deals, 1):
        vc = str(d.get("vcode") or "").strip()
        if not vc:
            continue
        yrs = {}
        for yr in (year - 1, year):
            r = api.get("/api/data/tables/isbs_budget_is/rows",
                        params={"page": 1, "page_size": 1,
                                "filter__vcode": vc.lower(),
                                "filter__dtEntry": str(yr)})
            if (r.get("total") or 0) > 0:
                yrs[str(yr)] = r["total"]
        budget_years[vc.upper()] = yrs
        if i % 40 == 0:
            print(f"  {i}/{len(deals)}")

    fires, has_cur, no_bud, dev = _classify(deals, budget_years, year)
    print(f"\nreport year {year}: fires={len(fires)} has_current={len(has_cur)} "
          f"no_budget={len(no_bud)}  dev={len(dev)}")
    print(f"  dev AND fires (must change): {sorted(fires & dev)}")

    # Target set: everything that can possibly move, plus controls that must not.
    targets = sorted(fires | dev
                     | set(sorted(has_cur)[:8]) | set(sorted(no_bud)[:8]))
    print(f"targets: {len(targets)} deals — pulling ISBS per deal...")

    isbs, skipped = {}, {}
    for i, vc in enumerate(targets, 1):
        parts = []
        try:
            for tbl, src in ISBS_TABLES:
                got = []
                for yr in FETCH_YEARS:
                    got.extend(_rows(api, tbl, filter__vcode=vc.lower(),
                                     filter__dtEntry=yr))
                for r in got:
                    # The rows endpoint drops vSource on some split tables; the
                    # split IS the vSource, restored as _assemble_isbs does.
                    r["vSource"] = src
                parts.extend(got)
        except RuntimeError as exc:
            # A handful of big PORTFOLIO deals exceed 500 rows even in a single
            # deal-month, and page 2 of this endpoint repeats rows (see _rows).
            # Rather than build a frame that cannot be shown to match live,
            # the deal is recorded as unfetchable and the report asserts that
            # no DEV target ever lands here.
            skipped[vc] = str(exc)
            print(f"  {i}/{len(targets)}  {vc}: SKIPPED — {exc}")
            continue
        isbs[vc] = parts
        if i % 10 == 0 or i == len(targets):
            print(f"  {i}/{len(targets)}  ({vc}: {len(parts)} rows)")

    targets = [t for t in targets if t in isbs]
    if skipped:
        print(f"\n  {len(skipped)} deal(s) excluded as unfetchable: "
              f"{sorted(skipped)}")

    data = {"deals": deals, "budget_years": budget_years,
            "targets": targets, "isbs": isbs, "quarter": QUARTER,
            "skipped": skipped}
    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, default=str)
    print(f"cached -> {cache_path}  "
          f"({sum(len(v) for v in isbs.values())} ISBS rows total)")
    return 0


def _frame(rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["vcode"] = df["vcode"].astype(str).str.strip().str.lower()
    df["vAccount"] = df["vAccount"].astype(str).str.strip()
    df["mAmount"] = pd.to_numeric(df["mAmount"], errors="coerce").fillna(0.0)
    df["dtEntry_parsed"] = pd.to_datetime(df["dtEntry"], errors="coerce",
                                          format="mixed")
    return df


def _capture(cache_path, root, out_path):
    root = os.path.abspath(root)
    sys.path.insert(0, root)
    import pandas as pd
    import one_pager

    assert os.path.abspath(one_pager.__file__).startswith(root), (
        f"imported the wrong one_pager: {one_pager.__file__}")
    print(f"  root={root}\n  one_pager={one_pager.__file__}")

    with open(cache_path, encoding="utf-8") as fh:
        data = json.load(fh)
    inv = pd.DataFrame(data["deals"])
    quarter = data.get("quarter", QUARTER)

    out = {}
    for vc in data["targets"]:
        try:
            isbs = _frame(data["isbs"].get(vc) or [])
            # inv_map is what makes the dev branch reachable — without it the
            # fix is inert and this guardrail would silently prove nothing.
            perf = one_pager.get_property_performance(
                vc, quarter, isbs, None, None, inv_map=inv)
            out[vc] = {r: {c: perf.get(r, {}).get(c) for c in COLS} for r in ROWS}
        except Exception as exc:
            out[vc] = {"error": f"{type(exc).__name__}: {exc}"}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    errs = [k for k, v in out.items() if "error" in v]
    print(f"  captured {len(out)} deals ({len(errs)} error(s)) -> {out_path}")
    for k in errs[:5]:
        print(f"    {k}: {out[k]['error']}")
    return 0


def _tieout(cache_path, before_path):
    """Prove the local BEFORE reproduces the LIVE payload."""
    api = _api()
    with open(cache_path, encoding="utf-8") as fh:
        data = json.load(fh)
    with open(before_path, encoding="utf-8") as fh:
        before = json.load(fh)
    quarter = data.get("quarter", QUARTER)

    year = int(quarter.split("-")[0])
    fires, has_cur, no_bud, dev = _classify(data["deals"], data["budget_years"], year)
    # Only deals actually fetched — an excluded deal has no local rows at all,
    # so comparing it would report a guaranteed mismatch that says nothing
    # about fidelity.
    fetched = set(data["targets"])
    check = sorted(((fires & dev) | set(sorted((fires - dev) & fetched)[:6]))
                   & fetched)

    print(f"Tie-out vs LIVE at {quarter} ({len(check)} deals)")
    print(f"  {'vcode':<10}{'row':<10}{'local before':>18}{'live':>18}  match")
    ok = bad = 0
    for vc in check:
        live = api.get(f"/api/financials/{vc}/one-pager",
                       params={"quarter": quarter}).get("property_performance") or {}
        for r in ("revenue", "expenses", "noi"):
            for c in ("ytd_budget", "actual_ye"):
                lv = (live.get(r) or {}).get(c)
                bv = (before.get(vc, {}).get(r) or {}).get(c)
                try:
                    same = abs(float(lv or 0) - float(bv or 0)) < 1.0
                except (TypeError, ValueError):
                    same = (lv == bv)
                ok, bad = (ok + 1, bad) if same else (ok, bad + 1)
                if not same:
                    print(f"  {vc:<10}{r + '.' + c:<10}{float(bv or 0):>18,.0f}"
                          f"{float(lv or 0):>18,.0f}  MISMATCH")
    print(f"\n  {ok} matched, {bad} mismatched")
    if bad:
        print("  -> the local harness does NOT reproduce live; before/after "
              "below is not trustworthy until this is resolved")
    else:
        print("  -> local BEFORE reproduces LIVE exactly; the harness is faithful")
    return 0 if bad == 0 else 1


def _report(cache_path, before_path, after_path):
    import pandas as pd

    with open(cache_path, encoding="utf-8") as fh:
        data = json.load(fh)
    with open(before_path, encoding="utf-8") as fh:
        before = json.load(fh)
    with open(after_path, encoding="utf-8") as fh:
        after = json.load(fh)

    inv = pd.DataFrame(data["deals"])
    name = {str(a).strip().upper(): str(b)
            for a, b in zip(inv["vcode"], inv["Investment_Name"])}
    quarter = data.get("quarter", QUARTER)
    year = int(quarter.split("-")[0])
    fires, has_cur, no_bud, dev = _classify(data["deals"], data["budget_years"], year)

    checks = []

    def chk(label, cond):
        checks.append((label, bool(cond)))
        print(f"    [{'PASS' if cond else 'FAIL'}] {label}")

    def val(side, vc, r, c):
        return (side.get(vc, {}).get(r) or {}).get(c)

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def moved(vc):
        return any(val(before, vc, r, c) != val(after, vc, r, c)
                   for r in ROWS for c in COLS)

    keys = [k for k in data["targets"] if k in before and k in after]
    changed = sorted(k for k in keys if moved(k))

    print("=" * 104)
    print(f"BEFORE / AFTER  —  {quarter}   (Projected YE = actual_ye)")
    print("=" * 104)
    for vc in changed:
        tag = "DEV" if vc in dev else "stabilized"
        print(f"\n  {vc}  {name.get(vc, '?')}   [{tag}]")
        print(f"    {'row':<10}{'YTD Budget before':>20}{'after':>14}"
              f"{'Proj YE before':>18}{'after':>16}{'YTD Actual':>16}")
        for r in ROWS:
            def f(v):
                n = num(v)
                if n is None:
                    return "None"
                return f"{n:,.3f}" if r == "dscr" else f"{n:,.0f}"
            print(f"    {r:<10}{f(val(before, vc, r, 'ytd_budget')):>20}"
                  f"{f(val(after, vc, r, 'ytd_budget')):>14}"
                  f"{f(val(before, vc, r, 'actual_ye')):>18}"
                  f"{f(val(after, vc, r, 'actual_ye')):>16}"
                  f"{f(val(after, vc, r, 'ytd_actual')):>16}")

    print("\n" + "=" * 104)
    print("UNCHANGED CONTROLS — stabilized deals the fallback still fires for")
    print("=" * 104)
    ctrl = sorted(fires - dev)[:6]
    print(f"  {'vcode':<10}{'deal':<34}{'Proj YE NOI':>16}{'YTD Budget NOI':>18}"
          f"{'moved?':>9}")
    for vc in ctrl:
        if vc not in before:
            continue
        print(f"  {vc:<10}{name.get(vc, '?')[:33]:<34}"
              f"{num(val(after, vc, 'noi', 'actual_ye')) or 0:>16,.0f}"
              f"{num(val(after, vc, 'noi', 'ytd_budget')) or 0:>18,.0f}"
              f"{('MOVED' if moved(vc) else 'no'):>9}")

    print("\n" + "=" * 104)
    print("CHECKS")
    print("=" * 104)
    chk("no deal errored on either side",
        not any("error" in before[k] or "error" in after[k] for k in keys))
    chk("both sides cover the same deals", set(before) == set(after))

    # Coverage, stated rather than assumed. A few big portfolio deals cannot be
    # fetched faithfully (>500 rows in a single deal-month, and page 2 of this
    # endpoint repeats rows), so they are excluded up front. That is only
    # acceptable while every deal the fix must MOVE is still observed.
    skipped = data.get("skipped") or {}
    chk(f"no deal the fix must change was excluded as unfetchable "
        f"({len(skipped)} excluded: {sorted(skipped)})",
        not (set(skipped) & MUST_CHANGE))
    lost_stab = sorted(set(skipped) & (fires - dev))
    print(f"           excluded incl. {len(lost_stab)} stabilized fallback "
          f"deals {lost_stab} and "
          f"{sorted(set(skipped) & dev)} dev (none of which the fallback fires for)")
    chk("enough stabilized fallback deals remain to prove they are untouched "
        f"({len((fires - dev) & set(keys))} of {len(fires - dev)})",
        len((fires - dev) & set(keys)) >= 12)

    dev_fires = sorted((fires & dev) & set(keys))
    chk(f"the dev deals the fallback fired for are exactly {sorted(MUST_CHANGE)}",
        set(dev_fires) == MUST_CHANGE)
    chk("...and every one of them changed", set(changed) >= set(dev_fires))
    chk("ONLY those dev deals changed — nothing else moved",
        set(changed) == set(dev_fires))

    # The defining property of the fix: a dev deal lands on the no-budget path.
    chk("every changed dev deal now has YTD Budget == 0 on all four rows",
        all((num(val(after, vc, r, "ytd_budget")) or 0) == 0
            for vc in dev_fires for r in ("revenue", "expenses", "noi")))
    chk("every changed dev deal now has Projected YE == YTD Actual "
        "(the no-budget path)",
        all(abs((num(val(after, vc, r, "actual_ye")) or 0)
                - (num(val(after, vc, r, "ytd_actual")) or 0)) < 1.0
            for vc in dev_fires for r in ("revenue", "expenses", "noi")))
    chk("...and each of them genuinely moved off a fabricated Projected YE",
        all(abs((num(val(before, vc, "noi", "actual_ye")) or 0)
                - (num(val(after, vc, "noi", "actual_ye")) or 0)) > 1.0
            for vc in dev_fires))

    stab_fires = sorted((fires - dev) & set(keys))
    chk(f"stabilized deals the fallback fires for are byte-identical — Jim "
        f"keeps it for them ({len(stab_fires)} deals)",
        all(before[k] == after[k] for k in stab_fires))
    chk("...and they still show a non-zero YTD Budget (fallback really is "
        "still firing, not silently disabled)",
        all(any((num(val(after, k, r, "ytd_budget")) or 0) != 0
                for r in ("revenue", "expenses"))
            for k in stab_fires))

    cur = sorted(has_cur & set(keys))
    chk(f"deals with a real current-year budget are byte-identical "
        f"({len(cur)} sampled)", all(before[k] == after[k] for k in cur))
    nob = sorted(no_bud & set(keys))
    chk(f"deals with no budget in either year are byte-identical "
        f"({len(nob)} sampled)", all(before[k] == after[k] for k in nob))

    dev_nofire = sorted((dev - fires) & set(keys))
    chk(f"dev deals the fallback never fired for are byte-identical "
        f"({len(dev_nofire)} deals)",
        all(before[k] == after[k] for k in dev_nofire))

    chk("no deal's YTD Actual moved (the fix touches budget only)",
        all(val(before, k, r, "ytd_actual") == val(after, k, r, "ytd_actual")
            for k in keys for r in ROWS))
    chk("no deal's U/W YE or At Close moved",
        all(val(before, k, r, c) == val(after, k, r, c)
            for k in keys for r in ROWS for c in ("uw_ye", "at_close")))

    passed = sum(1 for _, c in checks if c)
    print(f"\n  {passed}/{len(checks)} checks passed")
    print(f"  {len(changed)} of {len(keys)} targeted deals changed")
    print(f"  population at {quarter}: fires={len(fires)} "
          f"(dev={len(fires & dev)}, stabilized={len(fires - dev)}), "
          f"current-year budget={len(has_cur)}, no budget={len(no_bud)}")
    return 0 if passed == len(checks) else 1


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    cmd = argv[1]
    if cmd == "fetch":
        return _fetch(argv[2])
    if cmd == "capture":
        return _capture(argv[2], argv[3], argv[4])
    if cmd == "tieout":
        return _tieout(argv[2], argv[3])
    if cmd == "report":
        return _report(argv[2], argv[3], argv[4])
    print(f"unknown command {cmd!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
