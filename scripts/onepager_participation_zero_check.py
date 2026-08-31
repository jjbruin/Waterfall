"""Guardrail: 'absent' (N/A) vs 'present zero' (0.0%) on One Pager Participation.

Runs the REAL committed producers on each side — a worktree pinned at
``origin/main`` for "before", the working tree for "after" — over EVERY deal,
for BOTH places the page prints a participation:

  * Capitalization   cap_stack.pe_participation
      one_pager.get_capitalization_stack()  +
      financials_service._enrich_cap_stack_from_deal_terms()   (an OVERRIDE)
  * PE Performance   pe_performance.participation
      one_pager.get_pe_performance()                           (a FALLBACK)

Both must land on the same number or the page prints two answers for one term.

The decisive check is not the raw value but what the USER SEES: each side is
rendered through ITS OWN OnePagerView rule — truthiness before
(``x ? fmtPct(x) : 'N/A'``), null-check after (``x != null ? ...``) — so the
default flip from 0.0 to None registers as "no visible change" on the 68 deals
that legitimately have no participation term, and the 2 target deals register
as 1.00% -> 0.0%.

``acct`` / ``isbs_raw`` / ``mri_loans`` / ``mri_val`` go in as None: only the
coupon / participation block is under test, and the funded / ROE / accrual
fields those drive are untouched by this change.  The whole cap and pe dicts
are still captured and diffed, so a stray perturbation of any OTHER field
fails the run.

Live data is fetched ONCE and cached, so both sides see identical inputs.

Usage (WF_TOKEN must be set for `fetch`):
    python scripts/onepager_participation_zero_check.py fetch   <cache.json>
    python scripts/onepager_participation_zero_check.py capture <cache.json> <root> <out.json>
    python scripts/onepager_participation_zero_check.py report  <cache.json> <before.json> <after.json>
"""
import json
import os
import sys

QUARTER = "2026-Q1"

#: The deals this fix exists for.  Both carry two vState='Share' rows owned by
#: OPJPI (the operating partner) at FXRate 1.0 and deal_terms.pe_split_capital
#: == 0.0.  They MUST move 1.00% -> 0.0% in BOTH blocks.
MUST_CHANGE = {"P0000077", "P0000085"}

#: Real participation terms that must not move, in either block.
SPOT_CHECK = {
    "P0000109": 0.55,   # Burton Retail Portfolio
    "P0000066": 0.50,   # Pegasus Life Storage
    "P0000001": 0.30,   # 30 Bearfoot
    "P0000006": 0.50,   # Belleville Self Storage
    "P0000029": 0.50,   # Middle Island
    "P0000033": 0.75,   # OREI Portfolio  (waterfall wins over deal_terms 0.475)
    "P0000028": 0.70,   # Merle Hay       (waterfall wins over deal_terms 0.30)
}


def _norm(v):
    """JSON-safe, comparison-stable."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    try:
        import numpy as np
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
    except Exception:
        pass
    if isinstance(v, (int, float, str)):
        return v
    return str(v)


def _fetch(cache_path):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
        # The rows endpoint pages by OFFSET and can repeat a row; drop dupes.
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
    root = os.path.abspath(root)
    sys.path.insert(0, root)
    import pandas as pd
    import one_pager
    from flask_app.services.financials_service import (
        _enrich_cap_stack_from_deal_terms as enrich)

    # Never let a stale sys.path silently test the wrong tree.
    assert os.path.abspath(one_pager.__file__).startswith(root), (
        f"imported the wrong one_pager: {one_pager.__file__}")
    import flask_app.services.financials_service as fs
    assert os.path.abspath(fs.__file__).startswith(root), (
        f"imported the wrong financials_service: {fs.__file__}")
    print(f"  root={root}")
    print(f"  one_pager          = {one_pager.__file__}")
    print(f"  financials_service = {fs.__file__}")

    with open(cache_path, encoding="utf-8") as fh:
        data = json.load(fh)
    inv = pd.DataFrame(data["deals"])
    wf = pd.DataFrame(data["waterfalls"])
    dt = pd.DataFrame(data["deal_terms"])

    vc_col = next(c for c in inv.columns if c.lower() == "vcode")
    out = {}
    for vcode in sorted({str(v).strip() for v in inv[vc_col] if str(v).strip()}):
        rec = {}
        try:
            cap = one_pager.get_capitalization_stack(
                vcode, None, None, wf, None, inv)
            # Mirrors get_one_pager_data(): the override runs when deal_terms
            # has rows at all, not when this deal has one.
            if dt is not None and not dt.empty:
                enrich(cap, dt, vcode)
            rec["cap"] = {k: _norm(v) for k, v in cap.items()}
        except Exception as exc:
            rec["cap_error"] = f"{type(exc).__name__}: {exc}"
        try:
            pe = one_pager.get_pe_performance(
                vcode, QUARTER, None, wf, inv, isbs_raw=None, deal_terms=dt)
            rec["pe"] = {k: _norm(v) for k, v in pe.items()}
        except Exception as exc:
            rec["pe_error"] = f"{type(exc).__name__}: {exc}"
        out[vcode.upper()] = rec

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    errs = [k for k, v in out.items() if "cap_error" in v or "pe_error" in v]
    print(f"  captured {len(out)} deals ({len(errs)} error(s)) -> {out_path}")
    for k in errs[:5]:
        print(f"    {k}: {out[k].get('cap_error') or out[k].get('pe_error')}")
    return 0


def _render(val, null_check):
    """Reproduce OnePagerView exactly.

    before: {{ x ? fmtPct(x) : 'N/A' }}          -> truthiness
    after:  {{ x != null ? fmtPct(x) : 'N/A' }}  -> null check
    fmtPct: val == null || isNaN -> '—'; pct = val > 1 ? val : val*100; toFixed(1)
    """
    ok = (val is not None) if null_check else bool(val)
    if not ok:
        return "N/A"
    pct = val if val > 1 else val * 100
    return f"{pct:.1f}%"


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
    name = {str(a).strip().upper(): str(b) for a, b in zip(inv[vc_col], inv[nm_col])}
    dt_rows = {str(r.get("vcode") or "").strip().upper(): r
               for r in data["deal_terms"]}
    wf_share = set()
    for r in data["waterfalls"]:
        if str(r.get("vState") or "").strip().lower() == "share":
            wf_share.add(str(r.get("vcode") or "").strip().upper())

    checks = []

    def chk(label, cond):
        checks.append((label, bool(cond)))
        print(f"    [{'PASS' if cond else 'FAIL'}] {label}")

    def capv(side, k):
        return (side.get(k, {}).get("cap") or {}).get("pe_participation")

    def pev(side, k):
        return (side.get(k, {}).get("pe") or {}).get("participation")

    keys = sorted(before)

    # ---- what the user actually sees ---------------------------------
    vis = {}
    for k in keys:
        vis[k] = {
            "cap_b": _render(capv(before, k), False),
            "cap_a": _render(capv(after, k), True),
            "pe_b": _render(pev(before, k), False),
            "pe_a": _render(pev(after, k), True),
        }
    moved = [k for k in keys
             if vis[k]["cap_b"] != vis[k]["cap_a"] or vis[k]["pe_b"] != vis[k]["pe_a"]]

    print("=" * 104)
    print("WHAT CHANGED ON THE PAGE  (rendered through each side's own Vue rule)")
    print("=" * 104)
    print(f"  {'vcode':<10}{'deal':<32}{'Capitalization':>26}{'PE Performance':>26}")
    print("  " + "-" * 100)
    for k in moved:
        v = vis[k]
        print(f"  {k:<10}{name.get(k, '?')[:31]:<32}"
              f"{v['cap_b'] + '  ->  ' + v['cap_a']:>26}"
              f"{v['pe_b'] + '  ->  ' + v['pe_a']:>26}")
        d = dt_rows.get(k, {})
        print(f"  {'':<10}deal_terms.pe_split_capital={d.get('pe_split_capital')!r}"
              f"   pe_coupon={d.get('pe_coupon')!r}"
              f"   waterfall Share row: {'YES' if k in wf_share else 'NONE'}")
    if not moved:
        print("  (nothing)")

    print("\n" + "=" * 104)
    print("SPOT CHECK — deals with a real participation term")
    print("=" * 104)
    print(f"  {'vcode':<10}{'deal':<32}{'expect':>9}{'cap before':>13}"
          f"{'cap after':>12}{'pe before':>12}{'pe after':>11}")
    for k, exp in sorted(SPOT_CHECK.items()):
        v = vis[k]
        print(f"  {k:<10}{name.get(k, '?')[:31]:<32}{exp * 100:>8.1f}%"
              f"{v['cap_b']:>13}{v['cap_a']:>12}{v['pe_b']:>12}{v['pe_a']:>11}")

    print("\n" + "=" * 104)
    print("CHECKS")
    print("=" * 104)
    chk("no deal errored on either side",
        not any("cap_error" in v or "pe_error" in v
                for v in list(before.values()) + list(after.values())))
    chk("both sides cover the same deals", set(before) == set(after))

    # -- the two targets ------------------------------------------------
    chk(f"exactly the 2 target deals moved on the page {sorted(MUST_CHANGE)}",
        set(moved) == MUST_CHANGE)
    chk("...and both moved 1.0% -> 0.0% in the Capitalization block",
        all(vis[k]["cap_b"] == "1.0%" and vis[k]["cap_a"] == "0.0%"
            for k in MUST_CHANGE))
    chk("...and both moved 1.0% -> 0.0% in the PE Performance block",
        all(vis[k]["pe_b"] == "1.0%" and vis[k]["pe_a"] == "0.0%"
            for k in MUST_CHANGE))
    chk("...and the two blocks now AGREE for them (one term, one number)",
        all(capv(after, k) == pev(after, k) == 0.0 for k in MUST_CHANGE))

    # -- healthy deals ---------------------------------------------------
    chk(f"every spot-checked real participation is unmoved and correct "
        f"({len(SPOT_CHECK)} deals)",
        all(capv(after, k) == capv(before, k)
            and pev(after, k) == pev(before, k)
            and abs((pev(after, k) or -1) - exp) < 1e-12
            for k, exp in SPOT_CHECK.items()))

    nonzero = [k for k in keys if pev(before, k) or capv(before, k)]
    chk("no deal with a NON-ZERO participation before changed its value "
        f"({len(nonzero)} deals)",
        all(capv(after, k) == capv(before, k) and pev(after, k) == pev(before, k)
            for k in nonzero if k not in MUST_CHANGE))

    # -- THE distinction -------------------------------------------------
    absent = [k for k in keys if capv(after, k) is None and pev(after, k) is None]
    chk(f"absent deals render N/A in BOTH blocks, never 0.0% "
        f"({len(absent)} deals)",
        all(vis[k]["cap_a"] == "N/A" and vis[k]["pe_a"] == "N/A" for k in absent))
    chk("...and every one of them was N/A before too (no visible change)",
        all(vis[k]["cap_b"] == "N/A" and vis[k]["pe_b"] == "N/A" for k in absent))

    present_zero = [k for k in keys
                    if capv(after, k) == 0.0 or pev(after, k) == 0.0]
    chk(f"the ONLY deals holding a present zero are the 2 targets "
        f"(found {sorted(present_zero)})",
        set(present_zero) == MUST_CHANGE)
    chk("present zero renders 0.0%, absent renders N/A — the distinction is "
        "visible and total",
        all(_render(capv(after, k), True) == "0.0%" for k in present_zero)
        and all(_render(capv(after, k), True) == "N/A" for k in absent)
        and len(present_zero) + len(absent) + len(nonzero) >= len(keys))

    # every deal lands in exactly one of the three states
    states = {}
    for k in keys:
        c, p = capv(after, k), pev(after, k)
        states[k] = ("absent" if c is None and p is None else
                     "zero" if c == 0.0 or p == 0.0 else "value")
    chk(f"every deal is exactly one of absent/zero/value "
        f"({sum(1 for s in states.values() if s == 'absent')} absent, "
        f"{sum(1 for s in states.values() if s == 'zero')} zero, "
        f"{sum(1 for s in states.values() if s == 'value')} value = {len(keys)})",
        len(states) == len(keys))

    # -- a present zero must come from deal_terms, never invented ---------
    def dt_split(k):
        """None when absent.  NOT `or -1` — 0.0 is falsy and that idiom is the
        very bug under test; it silently turned the real zero into a sentinel."""
        v = dt_rows.get(k, {}).get("pe_split_capital")
        try:
            return None if v is None or str(v).strip() == "" else float(v)
        except (TypeError, ValueError):
            return None

    chk("every present zero is backed by deal_terms.pe_split_capital == 0",
        all(dt_split(k) == 0.0 for k in present_zero))
    chk("no deal_terms row anywhere else carries a zero split "
        "(so the present-zero set cannot grow silently)",
        {k for k in dt_rows if dt_split(k) == 0.0} == MUST_CHANGE)

    # -- nothing else moved ----------------------------------------------
    other_cap = [(k, f) for k in keys
                 for f in (before[k].get("cap") or {})
                 if f != "pe_participation"
                 and (before[k].get("cap") or {}).get(f)
                 != (after[k].get("cap") or {}).get(f)]
    chk(f"no OTHER cap_stack field changed on any deal "
        f"({len(other_cap)} violations)", not other_cap)
    for k, f in other_cap[:10]:
        print(f"           {k} cap.{f}: {before[k]['cap'].get(f)!r} -> "
              f"{after[k]['cap'].get(f)!r}")

    other_pe = [(k, f) for k in keys
                for f in (before[k].get("pe") or {})
                if f != "participation"
                and (before[k].get("pe") or {}).get(f)
                != (after[k].get("pe") or {}).get(f)]
    chk(f"no OTHER pe_performance field changed on any deal — coupon included "
        f"({len(other_pe)} violations)", not other_pe)
    for k, f in other_pe[:10]:
        print(f"           {k} pe.{f}: {before[k]['pe'].get(f)!r} -> "
              f"{after[k]['pe'].get(f)!r}")

    chk("pe_coupon specifically is byte-identical on every deal",
        all((before[k].get("pe") or {}).get("coupon")
            == (after[k].get("pe") or {}).get("coupon") for k in keys))
    chk("cap pe_coupon specifically is byte-identical on every deal",
        all((before[k].get("cap") or {}).get("pe_coupon")
            == (after[k].get("cap") or {}).get("pe_coupon") for k in keys))

    passed = sum(1 for _, c in checks if c)
    print(f"\n  {passed}/{len(checks)} checks passed")
    print(f"  {len(moved)} of {len(before)} deals changed on the page")
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
