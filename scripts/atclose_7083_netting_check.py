"""Guardrail for netting account 7083 into the At Close expense line.

Two independent checks, because the fix has two call sites:

  SQL PATH (primary).  ``Prop_Info_AtClose.sql`` populates the ``at_close_noi``
  table, which ``one_pager.get_property_performance()`` reads verbatim
  (``perf['expenses']['at_close'] = at_close_expenses``). The bucket expression
  is pure arithmetic over the Projected IS rows at each deal's at_close_date,
  so it is simulated here EXACTLY, old expression vs new, over ALL 80 deals.
  That is the portfolio-wide regression: the only deal whose expenses or NOI
  may move is P0000001.

  FALLBACK PATH.  Runs the REAL committed ``get_property_performance()`` with
  ``at_close_noi_df=None`` — "before" out of a worktree pinned at origin/main,
  "after" out of the working tree — and checks it lands on the same number the
  SQL now produces, plus that no OTHER column moved.

Live data is fetched once and cached so both sides see identical inputs.

Usage (WF_TOKEN must be set):
    python scripts/atclose_7083_netting_check.py fetch   <cache.json>
    python scripts/atclose_7083_netting_check.py sqldiff <cache.json>
    python scripts/atclose_7083_netting_check.py capture <cache.json> <root> <out.json>
    python scripts/atclose_7083_netting_check.py report  <cache.json> <before> <after>
"""
import json
import os
import sys

QUARTER = "2026-Q1"
BEARFOOT = "P0000001"
#: Deals run through the real function on both sides. Bearfoot plus the two
#: look-alikes the fix must NOT move, plus two ordinary operating controls.
FALLBACK_DEALS = ["P0000001", "P0000100", "P0000077", "P0000075", "P0000019"]
SPLIT = {"isbs_projected_is": "Projected IS", "isbs_interim_is": "Interim IS",
         "isbs_interim_bs": "Interim BS", "isbs_budget_is": "Budget IS"}


def _api():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import live_api
    return live_api


def _rows(api, table, vcode):
    out, p = [], 1
    while True:
        d = api.get(f"/api/data/tables/{table}/rows",
                    params={"page": p, "page_size": 500, "filter__vcode": vcode})
        out += d.get("rows") or []
        if p >= d.get("total_pages", 1):
            break
        p += 1
    seen, u = set(), []
    for r in out:                      # OFFSET paging can repeat rows
        k = tuple(sorted((a, str(b)) for a, b in r.items()))
        if k not in seen:
            seen.add(k)
            u.append(r)
    return [r for r in u
            if str(r.get("vcode", "")).strip().lower() == vcode.lower()]


def _fetch(cache_path):
    import pandas as pd
    api = _api()
    ti = api.token_info()
    print(f"LIVE {ti['username']} {ti['hours_left']}h  "
          f"build={api.get('/api/data/version').get('version')}")

    ac = api.get("/api/data/tables/at_close_noi/rows",
                 params={"page": 1, "page_size": 500}).get("rows") or []
    inv = api.get("/api/data/deals/all").get("deals") or []
    print(f"  at_close_noi rows: {len(ac)}")

    at_close_rows = {}
    for i, r in enumerate(ac, 1):
        vc = str(r["vcode"]).strip()
        acd = pd.Timestamp(r["at_close_date"])
        try:
            rs = _rows(api, "isbs_projected_is", vc)
        except Exception as exc:
            print(f"    {vc}: ERR {exc}")
            continue
        keep = []
        for x in rs:
            dt = pd.to_datetime(x.get("dtEntry"), errors="coerce")
            if pd.notna(dt) and dt == acd:
                keep.append({"vAccount": str(x.get("vAccount") or "").strip()
                                          .replace(".0", ""),
                             "vInput": x.get("vInput"),
                             "mAmount": x.get("mAmount")})
        at_close_rows[vc.upper()] = keep
        if i % 20 == 0:
            print(f"    ...{i}/{len(ac)}", flush=True)

    full = {}
    for vc in FALLBACK_DEALS:
        frames = []
        for tbl, src in SPLIT.items():
            try:
                rs = _rows(api, tbl, vc)
            except Exception:
                rs = []
            for x in rs:
                x["vSource"] = src
            frames += rs
        full[vc.upper()] = frames
        print(f"  full ISBS {vc}: {len(frames)} rows")

    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump({"at_close_noi": ac, "deals": inv,
                   "at_close_rows": at_close_rows, "full_isbs": full}, fh,
                  default=str)
    print(f"  cached -> {cache_path}")
    return 0


def _bucket(rows, net_7083):
    """The SQL's at_close_revenue / at_close_expenses / at_close_noi."""
    rev = exp = 0.0
    for r in rows:
        a = str(r["vAccount"])
        try:
            amt = float(str(r["mAmount"]).replace(",", ""))
        except (TypeError, ValueError):
            continue
        if a.startswith("4"):
            rev += amt
        elif a.startswith("5"):
            exp += amt
        elif net_7083 and a == "7083":
            exp += amt
    return rev, exp, rev + exp


def _sqldiff(cache_path):
    with open(cache_path, encoding="utf-8") as fh:
        c = json.load(fh)
    inv = {str(d.get("vcode", "")).upper(): d.get("Investment_Name")
           for d in c["deals"]}
    moved, checked = [], 0
    for vc, rows in sorted(c["at_close_rows"].items()):
        checked += 1
        b_rev, b_exp, b_noi = _bucket(rows, net_7083=False)
        a_rev, a_exp, a_noi = _bucket(rows, net_7083=True)
        if abs(a_exp - b_exp) >= 0.005 or abs(a_noi - b_noi) >= 0.005 \
                or abs(a_rev - b_rev) >= 0.005:
            moved.append((vc, inv.get(vc, "?"), b_rev, b_exp, b_noi,
                          a_rev, a_exp, a_noi))
    print("=" * 104)
    print(f"SQL BUCKET SIMULATION — old vs new, all {checked} at_close_noi deals")
    print("=" * 104)
    print(f"  {'vcode':<10}{'deal':<30}{'expenses':>28}{'NOI':>28}")
    for vc, nm, br, be, bn, ar, ae, an in moved:
        print(f"  {vc:<10}{str(nm)[:29]:<30}"
              f"{f'{be:,.2f} -> {ae:,.2f}':>28}{f'{bn:,.2f} -> {an:,.2f}':>28}")
        print(f"  {'':<10}revenue {br:,.2f} -> {ar:,.2f} (unchanged: "
              f"{abs(ar - br) < 0.005})")
    print(f"\n  deals moved: {len(moved)} of {checked}")
    ok = [m[0] for m in moved] == [BEARFOOT]
    print(f"  [{'PASS' if ok else 'FAIL'}] the ONLY deal that moves is {BEARFOOT}")
    if not ok:
        print("  !! STOP — unexpected deals moved, do not proceed")
    return 0 if ok else 1


def _capture(cache_path, root, out_path):
    root = os.path.abspath(root)
    sys.path.insert(0, root)
    import pandas as pd
    import one_pager
    assert os.path.abspath(one_pager.__file__).startswith(root), \
        f"wrong one_pager: {one_pager.__file__}"
    has = hasattr(one_pager, "AT_CLOSE_RESERVE_RELEASE_ACCTS")
    print(f"  root={root}\n  one_pager={one_pager.__file__}\n"
          f"  AT_CLOSE_RESERVE_RELEASE_ACCTS present: {has}")

    sys.path.insert(0, root)
    from flask_app.services.data_service import _normalize_isbs

    with open(cache_path, encoding="utf-8") as fh:
        c = json.load(fh)
    inv = pd.DataFrame(c["deals"])
    out = {}
    for vc, rows in c["full_isbs"].items():
        if not rows:
            continue
        isbs = _normalize_isbs(pd.DataFrame(rows))
        # at_close_noi_df deliberately omitted -> exercises the FALLBACK path
        perf = one_pager.get_property_performance(
            vc, QUARTER, isbs, pd.DataFrame(), None, inv_map=inv)
        out[vc] = {m: {k: perf[m].get(k) for k in
                       ("ytd_actual", "ytd_budget", "at_close",
                        "actual_ye", "uw_ye")}
                   for m in ("revenue", "expenses", "noi", "dscr")}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(f"  captured {len(out)} deals -> {out_path}")
    return 0


def _report(cache_path, before_path, after_path):
    with open(cache_path, encoding="utf-8") as fh:
        c = json.load(fh)
    with open(before_path, encoding="utf-8") as fh:
        before = json.load(fh)
    with open(after_path, encoding="utf-8") as fh:
        after = json.load(fh)

    checks = []

    def chk(label, cond):
        checks.append(bool(cond))
        print(f"    [{'PASS' if cond else 'FAIL'}] {label}")

    def g(d, vc, m, k):
        return (d.get(vc, {}).get(m, {}) or {}).get(k)

    def f(v):
        return "None" if v is None else f"{float(v):,.2f}"

    print("=" * 104)
    print("FALLBACK PATH — real get_property_performance(), at_close_noi_df=None")
    print("=" * 104)
    for vc in sorted(before):
        print(f"\n  {vc}")
        print(f"    {'metric':<10}{'column':<14}{'before':>18}{'after':>18}   moved")
        for m in ("revenue", "expenses", "noi", "dscr"):
            for k in ("at_close", "ytd_actual", "ytd_budget",
                      "actual_ye", "uw_ye"):
                b, a = g(before, vc, m, k), g(after, vc, m, k)
                mv = (b is None) != (a is None) or (
                    b is not None and abs(float(b) - float(a)) >= 0.005)
                if not mv and k != "at_close":
                    continue
                print(f"    {m:<10}{k:<14}{f(b):>18}{f(a):>18}   "
                      f"{'<== MOVED' if mv else ''}")

    # 1. Bearfoot at close goes to zero on the fallback
    chk("fallback: Bearfoot At Close expenses 60,100 -> 0",
        abs(float(g(before, BEARFOOT, 'expenses', 'at_close') or 0) - 60100) < 0.01
        and abs(float(g(after, BEARFOOT, 'expenses', 'at_close') or 0)) < 0.005)
    chk("fallback: Bearfoot At Close NOI -60,100 -> 0",
        abs(float(g(before, BEARFOOT, 'noi', 'at_close') or 0) + 60100) < 0.01
        and abs(float(g(after, BEARFOOT, 'noi', 'at_close') or 0)) < 0.005)
    chk("fallback: Bearfoot At Close revenue unchanged at 0",
        abs(float(g(after, BEARFOOT, 'revenue', 'at_close') or 0)) < 0.005)
    chk("fallback: Bearfoot At Close DSCR stays None",
        g(before, BEARFOOT, 'dscr', 'at_close') is None
        and g(after, BEARFOOT, 'dscr', 'at_close') is None)

    # 2. no OTHER column on Bearfoot moved
    other = [(m, k) for m in ("revenue", "expenses", "noi", "dscr")
             for k in ("ytd_actual", "ytd_budget", "actual_ye", "uw_ye")
             if (g(before, BEARFOOT, m, k) is None)
             != (g(after, BEARFOOT, m, k) is None)
             or (g(before, BEARFOOT, m, k) is not None
                 and abs(float(g(before, BEARFOOT, m, k))
                         - float(g(after, BEARFOOT, m, k))) >= 0.005)]
    chk(f"Bearfoot: no non-At-Close column moved ({len(other)} moved)", not other)
    for m, k in other:
        print(f"           {m}.{k}: {g(before, BEARFOOT, m, k)} -> "
              f"{g(after, BEARFOOT, m, k)}")

    # 3. no other deal moved at all
    others_moved = [(vc, m, k) for vc in before if vc != BEARFOOT
                    for m in ("revenue", "expenses", "noi", "dscr")
                    for k in ("at_close", "ytd_actual", "ytd_budget",
                              "actual_ye", "uw_ye")
                    if (g(before, vc, m, k) is None) != (g(after, vc, m, k) is None)
                    or (g(before, vc, m, k) is not None
                        and abs(float(g(before, vc, m, k))
                                - float(g(after, vc, m, k))) >= 0.005)]
    chk(f"fallback: no deal other than Bearfoot moved ({len(others_moved)})",
        not others_moved)
    for vc, m, k in others_moved:
        print(f"           {vc} {m}.{k}: {g(before, vc, m, k)} -> "
              f"{g(after, vc, m, k)}")

    # 4. primary (SQL) and fallback agree for Bearfoot
    sql_rev, sql_exp, sql_noi = _bucket(c["at_close_rows"][BEARFOOT], True)
    fb_exp = float(g(after, BEARFOOT, 'expenses', 'at_close') or 0)
    fb_noi = float(g(after, BEARFOOT, 'noi', 'at_close') or 0)
    print(f"\n  SQL (new)  expenses={sql_exp:,.2f}  noi(app sign)={-sql_noi:,.2f}")
    print(f"  fallback   expenses={fb_exp:,.2f}  noi={fb_noi:,.2f}")
    chk("primary SQL and fallback agree on Bearfoot At Close expenses",
        abs(sql_exp - fb_exp) < 0.005)
    chk("primary SQL and fallback agree on Bearfoot At Close NOI",
        abs((-sql_noi) - fb_noi) < 0.005)

    print(f"\n  {sum(checks)}/{len(checks)} checks passed")
    return 0 if all(checks) else 1


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    cmd = argv[1]
    if cmd == "fetch":
        return _fetch(argv[2])
    if cmd == "sqldiff":
        return _sqldiff(argv[2])
    if cmd == "capture":
        return _capture(argv[2], argv[3], argv[4])
    if cmd == "report":
        return _report(argv[2], argv[3], argv[4])
    print(f"unknown command {cmd!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
