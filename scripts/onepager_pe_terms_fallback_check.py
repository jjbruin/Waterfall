"""Guardrail for the PE Performance coupon / participation deal_terms fallback.

Runs the REAL committed ``one_pager.get_pe_performance()`` on each side — the
"before" out of a worktree pinned at ``origin/main``, the "after" out of the
working tree — over EVERY deal, and diffs the two coupon / participation values
per vcode.

The pass condition is deliberately two-sided:
  * every deal that HAS a waterfall Pref/Share row is byte-identical, and
  * exactly the deals with deal_terms and no waterfall row gain a value.

Only the coupon / participation block is exercised, so ``acct`` and ``isbs_raw``
go in as None — the funded / ROE / accrual fields those drive are untouched by
this change and fetching the accounting feed over REST is not viable anyway.

Live data is fetched ONCE and cached, so both sides see identical inputs and a
data refresh mid-run cannot masquerade as a code difference.

Usage (WF_TOKEN must be set):
    python scripts/onepager_pe_terms_fallback_check.py fetch   <cache.json>
    python scripts/onepager_pe_terms_fallback_check.py capture <cache.json> <root> <out.json>
    python scripts/onepager_pe_terms_fallback_check.py report  <cache.json> <before.json> <after.json>
"""
import json
import os
import sys

QUARTER = "2026-Q1"

#: The three live deals with PE capital, complete deal_terms and zero waterfall
#: rows. These MUST change — they are the reason the fix exists. They are not
#: the whole changed set: 17 more dormant/sold deals share the same shape, and
#: three deals DO have a waterfall but encode their pref as vState='Default'
#: rather than 'Pref', so the per-field fallback supplies their coupon while
#: their real Share participation is left alone. See PREF_ABSENT_WITH_WATERFALL.
MUST_CHANGE = {"P0000116", "P0000003", "P0000114"}

#: Has a waterfall, has a Share row, has NO vState='Pref' row. The coupon here
#: comes from deal_terms; the participation stays the waterfall's.
PREF_ABSENT_WITH_WATERFALL = {"P0000033", "P0000061", "P0000062"}


def _fetch(cache_path):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    import live_api as api

    ti = api.token_info()
    print(f"LIVE user={ti['username']} {ti['hours_left']}h  "
          f"build={api.get('/api/data/version').get('version')}")

    def rows(table):
        out, page = [], 1
        while True:
            d = api.get(f"/api/data/tables/{table}/rows",
                        params={"page": page, "page_size": 500})
            out += d.get("rows") or []
            if page >= d.get("total_pages", 1):
                break
            page += 1
        # The rows endpoint pages by OFFSET and can repeat a row; drop exact dupes.
        seen, uniq = set(), []
        for r in out:
            k = json.dumps(r, sort_keys=True, default=str)
            if k not in seen:
                seen.add(k)
                uniq.append(r)
        if len(uniq) != len(out):
            print(f"  {table}: dropped {len(out) - len(uniq)} duplicate row(s)")
        return uniq

    data = {
        "deals": api.get("/api/data/deals/all").get("deals") or [],
        "waterfalls": rows("waterfalls"),
        "deal_terms": rows("deal_terms"),
    }
    print(f"  deals={len(data['deals'])} waterfalls={len(data['waterfalls'])} "
          f"deal_terms={len(data['deal_terms'])}")
    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, default=str)
    print(f"  cached -> {cache_path}")
    return 0


def _capture(cache_path, root, out_path):
    """Import one_pager from `root` and run it over every deal."""
    root = os.path.abspath(root)
    sys.path.insert(0, root)
    import pandas as pd
    import one_pager

    assert os.path.abspath(one_pager.__file__).startswith(root), (
        f"imported the wrong one_pager: {one_pager.__file__}")
    supports = "deal_terms" in one_pager.get_pe_performance.__code__.co_varnames
    print(f"  root={root}")
    print(f"  one_pager={one_pager.__file__}")
    print(f"  get_pe_performance accepts deal_terms: {supports}")

    with open(cache_path, encoding="utf-8") as fh:
        data = json.load(fh)
    inv = pd.DataFrame(data["deals"])
    wf = pd.DataFrame(data["waterfalls"])
    dt = pd.DataFrame(data["deal_terms"])

    vc_col = next(c for c in inv.columns if c.lower() == "vcode")
    out = {}
    for vcode in sorted({str(v).strip() for v in inv[vc_col] if str(v).strip()}):
        kwargs = {"isbs_raw": None}
        if supports:
            kwargs["deal_terms"] = dt
        try:
            pe = one_pager.get_pe_performance(
                vcode, QUARTER, None, wf, inv, **kwargs)
            out[vcode.upper()] = {"coupon": pe.get("coupon"),
                                  "participation": pe.get("participation")}
        except Exception as exc:                        # never hide a crash
            out[vcode.upper()] = {"error": f"{type(exc).__name__}: {exc}"}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    errs = [v for v in out.values() if "error" in v]
    print(f"  captured {len(out)} deals ({len(errs)} error(s)) -> {out_path}")
    return 0


def _report(cache_path, before_path, after_path):
    import pandas as pd

    with open(cache_path, encoding="utf-8") as fh:
        data = json.load(fh)
    with open(before_path, encoding="utf-8") as fh:
        before = json.load(fh)
    with open(after_path, encoding="utf-8") as fh:
        after = json.load(fh)

    inv = pd.DataFrame(data["deals"])
    vc_col = next(c for c in inv.columns if c.lower() == "vcode")
    nm_col = next(c for c in inv.columns if c.lower() == "investment_name")
    name = {str(a).strip().upper(): str(b)
            for a, b in zip(inv[vc_col], inv[nm_col])}
    wf_vcodes = {str(r.get("vcode") or "").strip().upper()
                 for r in data["waterfalls"]}
    dt_rows = {str(r.get("vcode") or "").strip().upper(): r
               for r in data["deal_terms"]}

    checks = []

    def chk(label, cond):
        checks.append((label, bool(cond)))
        print(f"    [{'PASS' if cond else 'FAIL'}] {label}")

    def fmt(v):
        return "N/A" if not v else f"{v * 100:.2f}%"

    changed = sorted(k for k in before
                     if before[k] != after.get(k))

    print("=" * 100)
    print("CHANGED DEALS — PE Performance Coupon / Participation")
    print("=" * 100)
    print(f"  {'vcode':<10}{'deal':<32}{'coupon':>18}{'participation':>22}")
    print("  " + "-" * 96)
    for k in changed:
        b, a = before[k], after[k]
        print(f"  {k:<10}{name.get(k, '?')[:31]:<32}"
              f"{fmt(b.get('coupon')) + ' -> ' + fmt(a.get('coupon')):>18}"
              f"{fmt(b.get('participation')) + ' -> ' + fmt(a.get('participation')):>22}")
        dt = dt_rows.get(k, {})
        print(f"  {'':<10}deal_terms: pe_coupon={dt.get('pe_coupon')!r} "
              f"pe_split_capital={dt.get('pe_split_capital')!r}  "
              f"waterfall rows: {'YES' if k in wf_vcodes else 'NONE'}")

    print("\n" + "=" * 100)
    print("CHECKS")
    print("=" * 100)
    chk("no deal errored on either side",
        not any("error" in v for v in list(before.values()) + list(after.values())))
    chk("both sides cover the same deals", set(before) == set(after))

    # THE safety property, and the one that means "waterfall stays primary":
    # a field the waterfall already supplied is never touched. Asserted per
    # FIELD rather than per deal, because a deal can legitimately have a Share
    # row and no Pref row.
    overwritten = [(k, f) for k in before for f in ("coupon", "participation")
                   if before[k].get(f) and before[k][f] != after[k].get(f)]
    chk("no field the waterfall supplied was overwritten "
        f"({len(overwritten)} violations)", not overwritten)
    for k, f in overwritten:
        print(f"           {k} {f}: {before[k][f]} -> {after[k].get(f)}")

    chk("every changed field was 0/None before",
        all(not before[k].get(f) for k in changed
            for f in ("coupon", "participation")
            if before[k].get(f) != after[k].get(f)))

    chk("every changed field equals its deal_terms value",
        all(abs((after[k][f] or 0) - float(dt_rows.get(k, {}).get(col) or 0))
            < 1e-12
            for k in changed
            for f, col in (("coupon", "pe_coupon"),
                           ("participation", "pe_split_capital"))
            if before[k].get(f) != after[k].get(f)))

    chk(f"the three target deals all changed {sorted(MUST_CHANGE)}",
        MUST_CHANGE <= set(changed))

    burton = "P0000109"
    chk(f"Burton {burton} unchanged at coupon "
        f"{fmt(before.get(burton, {}).get('coupon'))} / participation "
        f"{fmt(before.get(burton, {}).get('participation'))}",
        before.get(burton) == after.get(burton)
        and before.get(burton, {}).get("coupon"))

    full_wf = [k for k in before if before[k].get("coupon")
               and before[k].get("participation")]
    chk("every deal whose waterfall gave BOTH values is byte-identical "
        f"({len(full_wf)} deals)",
        all(before[k] == after[k] for k in full_wf))

    chk("no deal LOST a value",
        all(not (before[k].get(f) and not after[k].get(f))
            for k in before for f in ("coupon", "participation")))

    pref_absent = sorted(k for k in changed if k in wf_vcodes)
    chk("the only waterfall-backed deals that changed are the known "
        f"vState='Default' pref deals {sorted(PREF_ABSENT_WITH_WATERFALL)}",
        set(pref_absent) == PREF_ABSENT_WITH_WATERFALL)
    chk("...and they changed COUPON ONLY, keeping the waterfall participation",
        all(before[k]["participation"] == after[k]["participation"]
            and before[k]["participation"] for k in pref_absent))

    no_wf_no_dt = sorted(k for k in before
                         if k not in wf_vcodes and k not in dt_rows)
    chk(f"deals with neither waterfall nor deal_terms still read N/A "
        f"({len(no_wf_no_dt)} deals)",
        all(not after[k]["coupon"] and not after[k]["participation"]
            for k in no_wf_no_dt))

    passed = sum(1 for _, c in checks if c)
    print(f"\n  {passed}/{len(checks)} checks passed")
    print(f"  {len(changed)} of {len(before)} deals changed")
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
    if cmd == "report":
        return _report(argv[2], argv[3], argv[4])
    print(f"unknown command {cmd!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
