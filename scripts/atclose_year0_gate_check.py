"""Guardrail: At-Close requires a Year-0 (2015-12-31) Projected IS row.

Runs the REAL committed ``one_pager.get_property_performance`` on each side of
the change — ``capture before`` from a worktree at main, ``capture after`` from
the working tree — then ``report`` diffs them. Same shape as
``onepager_chart_window_check.py``: nothing here re-implements the rule, so a
passing report is a statement about the shipped function.

Inputs are LIVE and read-only: each deal's own Projected IS rows plus its
``at_close_noi`` row, so BOTH At-Close paths are exercised (the pre-computed
table for deals that have a row, the earliest-December fallback for the rest).
The local SQLite snapshot is stale and would not reproduce the figures.

Usage:
    python scripts/atclose_year0_gate_check.py fixture            # live -> json
    python scripts/atclose_year0_gate_check.py capture before     # in worktree
    python scripts/atclose_year0_gate_check.py capture after      # working tree
    python scripts/atclose_year0_gate_check.py report
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd                                             # noqa: E402

OUT = os.environ.get(
    "WF_GATE_DIR",
    os.path.join(os.environ.get("TEMP", "/tmp"), "atclose_year0_gate"))
QUARTER = os.environ.get("WF_CHECK_QUARTER", "2026-Q1")

#: Every deal the narrowed rule has to get right, in three groups.
#:
#: EXPECT_ZERO  development + no 2015-12-31 row -> At-Close goes to 0
#: EXPECT_KEEP  no 2015-12-31 row but NOT development -> must keep its value.
#:              This group is the point of the narrowing; before it, all nine
#:              of these were being zeroed.
#: EXPECT_SAME  has the 2015-12-31 row -> untouched either way, and ties to the
#:              reference PDF.
EXPECT_ZERO = [
    "P0000100",   # Green Valley Ranch & Telluride   -59,333   NAMED, Development
    "P0000077",   # Jefferson Addison Heights       -115,440   NAMED, Development
    "P0000090",   # Brainerd - Bldg B1               116,543   Development child
    "P0000092",   # Brainerd - Bldg E              1,731,776   Development child
    "P0000096",   # Brainerd - Bldg D                160,511   Development child
    "P0000098",   # Brainerd - Bldg F(2)           1,039,591   Development child
    "P0000021",   # JB Fair Park                          -0   Development, already 0
    "P0000110",   # Trolley Square                        -0   Development, already 0
]

#: Caught TODAY only because Lifecycle "New Construction" is in DEV_STRATEGIES.
#: Verified separately, in both classification states — see check 5.
PEGASUS = "P0000066"                     # 624,689, on 9 Snapshot reports

EXPECT_KEEP = [
    "P0000014",   # Crowne Plaza                      20,441   Redevelopment
    "P0000038",   # Quakertown Shopping Center     1,015,688   Sold
    "P0000073",   # Donald Lynch                      52,439   Value-Add
    "P0000101",   # Town Fair Tire - Avon            736,747   Stable, child of 107
    "P0000102",   # Town Fair Tire - Milford         348,624   Stable, child of 107
    "P0000103",   # Town Fair Tire - Norwalk         250,095   Stable, child of 107
    "P0000104",   # Town Fair Tire - Orange          188,496   Stable, child of 107
    "P0000105",   # Town Fair Tire - Wallingford     310,684   Stable, child of 107
    "P0000106",   # Town Fair Tire - Warwick         456,684   Stable, child of 107
]

EXPECT_SAME = [
    "P0000107",   # Town Fair Tire Portfolio       2,305,024   the PARENT
    "P0000109",   # Burton Retail Portfolio        8,978,768
    "P0000006",   # Belleville Self Storage           70,716
    "P0000030",   # Nottingham Village             2,169,441
    "P0000075",   # Camp Creek                     6,407,215
    "P0000116",   # Plaza Del Mar                  2,163,438
    "P0000019",   # Giant 7                        8,985,445
    "P0000099",   # ReNew Glenmoore                2,995,260
    "P0000001",   # 30 Bearfoot                           -0   7083 netting
    "P0000068",   # The Point at Plymouth Meeting  4,294,439
]

DEALS = EXPECT_ZERO + [PEGASUS] + EXPECT_KEEP + EXPECT_SAME


# ── fixture: pull the real inputs once, from live ─────────────────────────

def build_fixture():
    from scripts import live_api

    def page(table, params, size=500, cap=40):
        out, page_n = [], 1
        while page_n <= cap:
            d = live_api.get(f"/api/data/tables/{table}/rows",
                             params={"page": page_n, "page_size": size, **params})
            rows = d.get("rows") or []
            out += rows
            if len(rows) < size:
                break
            page_n += 1
        # de-duplicate: OFFSET paging can repeat rows
        seen, uniq = set(), []
        for r in out:
            k = tuple(sorted((str(a), str(b)) for a, b in r.items()))
            if k not in seen:
                seen.add(k)
                uniq.append(r)
        return uniq

    os.makedirs(OUT, exist_ok=True)
    acn = page("at_close_noi", {})
    # The deals table IS inv_map — the gate's dev classification reads
    # Investment_Strategy / Lifecycle out of it, so it must be real, not stubbed.
    deals = page("deals", {"sort": "vcode", "order": "asc"})
    fix = {"quarter": QUARTER, "at_close_noi": acn, "deals": deals, "isbs": {}}
    for i, vc in enumerate(DEALS, 1):
        lc = vc.lower()
        rows = []
        for src in ("isbs_projected_is", "isbs_interim_is", "isbs_budget_is"):
            try:
                rows += page(src, {"filter__vcode": lc})
            except Exception as exc:
                print(f"  {vc} {src}: {str(exc)[:60]}")
        fix["isbs"][vc] = rows
        print(f"  [{i}/{len(DEALS)}] {vc}: {len(rows)} isbs rows")
    path = os.path.join(OUT, "fixture.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(fix, fh)
    print(f"\nfixture -> {path}")


# ── capture: run the real function on this checkout ───────────────────────

def _frame(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c in ("mAmount", "mAmount_norm"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["dtEntry_parsed"] = pd.to_datetime(df.get("dtEntry"), errors="coerce")
    if "vcode" in df.columns:
        df["vcode"] = df["vcode"].astype(str).str.strip().str.lower()
    if "vAccount" in df.columns:
        df["vAccount"] = df["vAccount"].astype(str).str.strip()
    return df


def capture(label, dev_strategies=None):
    """Run the real function over the fixture.

    ``dev_strategies`` overrides ``config.DEV_STRATEGIES`` for the run. Used to
    answer the Pegasus question without editing shipped code: it is caught today
    only because "new construction" is in that set, and must drop out when the
    separate classification fix removes it. ``_is_dev_deal`` imports
    ``config.is_dev_deal``, which reads the module-level set at call time, so
    rebinding it here is exactly what that fix will do.
    """
    if dev_strategies is not None:
        import config
        config.DEV_STRATEGIES = set(dev_strategies)

    from one_pager import get_property_performance

    with open(os.path.join(OUT, "fixture.json"), encoding="utf-8") as fh:
        fix = json.load(fh)
    acn = pd.DataFrame(fix["at_close_noi"])
    for c in ("at_close_revenue", "at_close_expenses", "at_close_noi",
              "at_close_interest", "at_close_principal"):
        if c in acn.columns:
            acn[c] = pd.to_numeric(acn[c], errors="coerce")

    inv_map = pd.DataFrame(fix.get("deals") or [])
    if not inv_map.empty and "vcode" in inv_map.columns:
        inv_map["vcode"] = inv_map["vcode"].astype(str).str.strip()

    got = {}
    for vc in DEALS:
        isbs = _frame(fix["isbs"].get(vc) or [])
        try:
            perf = get_property_performance(
                vc, fix["quarter"], isbs, None,
                at_close_noi_df=acn, inv_map=inv_map) or {}
        except Exception as exc:
            got[vc] = {"error": f"{type(exc).__name__}: {exc}"[:160]}
            continue
        got[vc] = {
            "revenue": (perf.get("revenue") or {}).get("at_close"),
            "expenses": (perf.get("expenses") or {}).get("at_close"),
            "noi": (perf.get("noi") or {}).get("at_close"),
            "dscr": (perf.get("dscr") or {}).get("at_close"),
            "zeroed_flag": perf.get("at_close_zeroed_no_year0", False),
            # every other column must be untouched by this change
            "noi_uw_ye": (perf.get("noi") or {}).get("uw_ye"),
            "noi_actual_ye": (perf.get("noi") or {}).get("actual_ye"),
            "noi_ytd_actual": (perf.get("noi") or {}).get("ytd_actual"),
            "occ_at_close": (perf.get("economic_occ") or {}).get("at_close"),
            "occ_uw_ye": (perf.get("economic_occ") or {}).get("uw_ye"),
            # the gate's two inputs, recorded so a mis-read of either is
            # visible in the report rather than inferred from the outcome
            "has_year0": _year0(isbs),
            "is_dev": _dev(vc, inv_map),
        }
    path = os.path.join(OUT, f"{label}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(got, fh, indent=1, default=str)
    print(f"captured {len(got)} deals -> {path}")


def _dev(vcode, inv_map):
    """Independent read of the dev classification — resolve_strategy's own
    field precedence, spelled out here rather than calling one_pager's helper,
    so the report records what the DATA says and not what the gate decided."""
    from config import is_dev_deal
    if inv_map is None or inv_map.empty:
        return None
    m = inv_map[inv_map["vcode"].astype(str).str.strip().str.upper()
                == str(vcode).strip().upper()]
    if m.empty:
        return None
    row = m.iloc[0]
    for field in ("Investment_Strategy", "Lifecycle"):
        if field in row.index and pd.notna(row[field]):
            v = str(row[field]).strip()
            if v:
                return is_dev_deal(v)
    return False


def _year0(isbs):
    """Independent read of the gate's input — deliberately NOT the shipped
    helper, so a bug in that helper cannot hide behind agreement with itself."""
    if isbs is None or isbs.empty or "vSource" not in isbs.columns:
        return False
    p = isbs[isbs["vSource"] == "Projected IS"]
    if p.empty:
        return False
    d = pd.to_datetime(p["dtEntry_parsed"], errors="coerce").dropna()
    return bool((d.dt.normalize() == pd.Timestamp("2015-12-31")).any())


# ── report ────────────────────────────────────────────────────────────────

def report():
    with open(os.path.join(OUT, "before.json"), encoding="utf-8") as fh:
        b = json.load(fh)
    with open(os.path.join(OUT, "after.json"), encoding="utf-8") as fh:
        a = json.load(fh)

    checks = []

    def chk(label, ok, detail=""):
        checks.append((label, bool(ok)))
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}"
              + (f"\n         {detail}" if detail and not ok else ""))

    def f(v):
        if v is None:
            return "None"
        try:
            return f"{float(v):,.0f}"
        except (TypeError, ValueError):
            return str(v)

    print("BEFORE / AFTER — At-Close, real committed function both sides")
    print("=" * 108)

    def block(title, pool):
        print(f"\n{title}")
        print(f"{'vcode':<10}{'dev':<6}{'2015':<7}"
              f"{'NOI before':>14}{'NOI after':>13}   "
              f"{'Rev before':>13}{'Rev after':>12}   "
              f"{'Exp before':>13}{'Exp after':>12}")
        for vc in pool:
            rb, ra = b.get(vc, {}), a.get(vc, {})
            if "error" in rb or "error" in ra:
                print(f"{vc:<10}ERROR before={rb.get('error')} "
                      f"after={ra.get('error')}")
                continue
            print(f"{vc:<10}{('YES' if rb.get('is_dev') else 'no'):<6}"
                  f"{('YES' if rb.get('has_year0') else 'no'):<7}"
                  f"{f(rb.get('noi')):>14}{f(ra.get('noi')):>13}   "
                  f"{f(rb.get('revenue')):>13}{f(ra.get('revenue')):>12}   "
                  f"{f(rb.get('expenses')):>13}{f(ra.get('expenses')):>12}")

    block("A. EXPECT ZERO — development, no 2015-12-31 row", EXPECT_ZERO)
    block("B. EXPECT KEEP — no 2015-12-31 row but NOT development "
          "(these were being zeroed before the narrowing)", EXPECT_KEEP)
    block("C. EXPECT SAME — has the 2015-12-31 row", EXPECT_SAME)
    block("D. PEGASUS — dev only via Lifecycle 'New Construction'", [PEGASUS])

    print("\n" + "=" * 108)
    print("CHECKS")
    print("=" * 108)

    for vc in DEALS:
        if "error" in b.get(vc, {}) or "error" in a.get(vc, {}):
            chk(f"{vc} computed without error", False,
                f"before={b.get(vc, {}).get('error')} "
                f"after={a.get(vc, {}).get('error')}")

    # 1. the two named targets
    for vc, was in (("P0000100", -59333), ("P0000077", -115440)):
        rb, ra = b.get(vc, {}), a.get(vc, {})
        chk(f"{vc}: before ~{was:,}",
            rb.get("noi") is not None and abs(float(rb["noi"]) - was) < 2,
            f"before={f(rb.get('noi'))}")
        chk(f"{vc}: At-Close NOI is now exactly 0",
            ra.get("noi") == 0, f"after={ra.get('noi')!r}")

    # 2. the zero group
    chk("every EXPECT_ZERO deal is classified development",
        all(b[vc].get("is_dev") is True for vc in EXPECT_ZERO),
        str({vc: b[vc].get("is_dev") for vc in EXPECT_ZERO
             if b[vc].get("is_dev") is not True}))
    chk("every EXPECT_ZERO deal has At-Close NOI == 0",
        all(a[vc].get("noi") == 0 for vc in EXPECT_ZERO),
        str({vc: a[vc].get("noi") for vc in EXPECT_ZERO
             if a[vc].get("noi") != 0}))
    chk("every zeroed deal carries the audit flag",
        all(a[vc].get("zeroed_flag") is True for vc in EXPECT_ZERO))
    chk("whole column zeroed — rev/exp 0 and DSCR None, so rev - exp == noi",
        all(a[vc].get("revenue") == 0 and a[vc].get("expenses") == 0
            and a[vc].get("dscr") is None for vc in EXPECT_ZERO),
        str({vc: (a[vc].get('revenue'), a[vc].get('expenses'),
                  a[vc].get('dscr')) for vc in EXPECT_ZERO
             if a[vc].get('revenue') != 0 or a[vc].get('expenses') != 0}))

    # 3. THE NARROWING — non-dev deals without the row keep their values
    chk("no EXPECT_KEEP deal is classified development",
        all(b[vc].get("is_dev") is False for vc in EXPECT_KEEP),
        str({vc: b[vc].get("is_dev") for vc in EXPECT_KEEP
             if b[vc].get("is_dev") is not False}))
    chk("every EXPECT_KEEP deal lacks the 2015-12-31 row "
        "(so only the dev condition spares it)",
        all(b[vc].get("has_year0") is False for vc in EXPECT_KEEP))
    moved_keep = {vc: (b[vc], a[vc]) for vc in EXPECT_KEEP if b[vc] != a[vc]}
    chk(f"all {len(EXPECT_KEEP)} non-development deals are UNCHANGED",
        not moved_keep,
        "; ".join(f"{vc}: {f(x.get('noi'))} -> {f(y.get('noi'))}"
                  for vc, (x, y) in moved_keep.items()))
    chk("no non-development deal carries the zeroed flag",
        all(not a[vc].get("zeroed_flag") for vc in EXPECT_KEEP))

    # 4. deals WITH the row are byte-identical
    moved_same = {vc: (b[vc], a[vc]) for vc in EXPECT_SAME if b[vc] != a[vc]}
    chk(f"all {len(EXPECT_SAME)} deals WITH a 2015-12-31 row are UNCHANGED",
        not moved_same,
        "; ".join(f"{vc}: {f(x.get('noi'))} -> {f(y.get('noi'))}"
                  for vc, (x, y) in moved_same.items()))

    # 5. no other column moved on ANY deal — the change is At-Close only
    other = ("noi_uw_ye", "noi_actual_ye", "noi_ytd_actual",
             "occ_at_close", "occ_uw_ye")
    drift = {vc: [k for k in other if b[vc].get(k) != a[vc].get(k)]
             for vc in DEALS if vc in b and vc in a}
    drift = {vc: ks for vc, ks in drift.items() if ks}
    chk("no OTHER column moved on any deal (U/W YE, Projected YE, YTD, Econ Occ)",
        not drift, str(drift))

    # 6. the parent/child pair the narrowing exists to protect
    chk("Town Fair Tire PARENT (P0000107) keeps its 2,305,024",
        a.get("P0000107", {}).get("noi") is not None
        and abs(float(a["P0000107"]["noi"]) - 2305024) < 2,
        f"after={f(a.get('P0000107', {}).get('noi'))}")
    kids = [vc for vc in EXPECT_KEEP if vc.startswith("P00001")
            and vc in ("P0000101", "P0000102", "P0000103",
                       "P0000104", "P0000105", "P0000106")]
    kid_sum = sum(float(a[vc]["noi"] or 0) for vc in kids)
    chk(f"all six Town Fair Tire children keep their values "
        f"(sum {kid_sum:,.0f} vs parent 2,305,024)",
        abs(kid_sum - 2291330) < 5, f"sum={kid_sum:,.0f}")

    # 7. Pegasus, in both classification states
    print("\n  -- Pegasus, under each dev classification --")
    peg_now = a.get(PEGASUS, {})
    chk("Pegasus is dev TODAY (Lifecycle 'New Construction' in DEV_STRATEGIES)",
        b.get(PEGASUS, {}).get("is_dev") is True)
    chk("Pegasus IS zeroed today — expected, pending the classification fix",
        peg_now.get("noi") == 0, f"after={f(peg_now.get('noi'))}")
    fixed_path = os.path.join(OUT, "after_devfix.json")
    if os.path.exists(fixed_path):
        with open(fixed_path, encoding="utf-8") as fh:
            af = json.load(fh)
        pf = af.get(PEGASUS, {})
        chk("with 'new construction' removed, Pegasus is no longer dev",
            pf.get("is_dev") is False, f"is_dev={pf.get('is_dev')}")
        chk("with 'new construction' removed, Pegasus KEEPS 624,689",
            pf.get("noi") is not None and abs(float(pf["noi"]) - 624689) < 2,
            f"noi={f(pf.get('noi'))}")
        chk("removing it changes nothing else in the zero group",
            all(af[vc].get("noi") == 0 for vc in EXPECT_ZERO),
            str({vc: af[vc].get("noi") for vc in EXPECT_ZERO
                 if af[vc].get("noi") != 0}))
        chk("removing it changes nothing in the keep group",
            all(af[vc] == a[vc] for vc in EXPECT_KEEP))
    else:
        chk("after_devfix.json captured (run: capture after_devfix)", False,
            f"missing {fixed_path}")

    print("\n" + "=" * 108)
    zeroed = [vc for vc in DEALS if a.get(vc, {}).get("zeroed_flag")]
    print(f"zeroed: {len(zeroed)} -> {zeroed}")
    bad = [c for c, ok in checks if not ok]
    print(f"{len(checks) - len(bad)}/{len(checks)} checks passed")
    for c in bad:
        print(f"  FAILED: {c}")
    return 1 if bad else 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "fixture":
        build_fixture()
    elif cmd == "capture":
        label = sys.argv[2]
        # "after_devfix" simulates the pending dev-classification fix by
        # dropping "new construction" from DEV_STRATEGIES for this run only.
        capture(label,
                dev_strategies={"development"}
                if label == "after_devfix" else None)
    else:
        raise SystemExit(report())
