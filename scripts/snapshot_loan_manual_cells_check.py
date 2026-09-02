"""Guardrail — the Loan subtab's TYPED ratio cells (LTV / YTD DSCR / Debt Yield).

Proves, on LIVE data, that making six recent acquisitions' three ratio columns
typeable did exactly that and nothing else:

  1. each of the six carries the figure it was pre-filled with, formatted in
     the unit its column displays ("69.0%", "1.9x"), and is marked typeable;
  2. a stored entry beats the seed, and a CLEARED cell stays cleared instead of
     springing back to the seed;
  3. the raw computed ``ltv`` / ``ytd_dscr`` / ``debt_yield`` are untouched on
     every row, so the fund subtotals and the portfolio total do not move;
  4. every deal that is NOT one of the six is byte-identical to before;
  5. the debt-free and development literals still outrank a typed cell;
  6. persistence accepts the three new field names, round-trips them, and still
     rejects an unknown one.

BEFORE and AFTER come from the SAME live inputs and the same committed
function — the only difference is ``MANUAL_RATIO_SEEDS``, emptied for the
before-side run. The providers memoise, so the second run costs no extra HTTP.

Read-only against live: GETs only, and the persistence checks run against a
throwaway SQLite file.

    WF_TOKEN=<jwt> python scripts/snapshot_loan_manual_cells_check.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import pandas as pd
import sqlalchemy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import live_api as api                                    # noqa: E402
from flask_app.services import portfolio_snapshot_loan as L   # noqa: E402
from flask_app.services import portfolio_snapshot_persistence as P  # noqa: E402

INV, Q = "TGAM", "2026-Q2"

#: What the work order asked for, in the unit each column displays.
EXPECTED = {
    "P0000109": {"ltv": (69.0, "69.0%")},
    "P0000116": {"ltv": (64.2, "64.2%")},
    "P0000117": {"ltv": (69.7, "69.7%"), "ytd_dscr": (1.9, "1.9x"),
                 "debt_yield": (12.1, "12.1%")},
    "P0000118": {"ltv": (75.7, "75.7%"), "ytd_dscr": (1.5, "1.5x"),
                 "debt_yield": (8.9, "8.9%")},
    # 5.93 keeps its second decimal — see _manual_fmt.
    "P0000119": {"ltv": (70.6, "70.6%"), "ytd_dscr": (1.1, "1.1x"),
                 "debt_yield": (5.93, "5.93%")},
    "P0000120": {"ltv": (74.0, "74.0%"), "ytd_dscr": (1.5, "1.5x"),
                 "debt_yield": (9.7, "9.7%")},
}

CHECKS: list = []


def chk(label, cond):
    CHECKS.append((label, bool(cond)))
    print(f"    [{'PASS' if cond else 'FAIL'}] {label}")


def flatten(out: dict) -> dict:
    flat = {r["vcode"]: r for rs in out["groups"].values() for r in rs}
    for r in out.get("ownership_flagged") or []:
        flat[r["vcode"]] = r
    return flat


def build(manual: dict, seeds: dict, providers: dict) -> dict:
    """One assemble_loan run with a given seed table and stored-value map."""
    saved = L.MANUAL_RATIO_SEEDS
    L.MANUAL_RATIO_SEEDS = seeds
    try:
        return L.assemble_loan(
            INV, Q,
            resolved=providers["resolved"],
            one_pager_provider=providers["one_pager"],
            loans=providers["loans"], valuations=providers["valuations"],
            inv=providers["inv"],
            quarterly_noi_provider=providers["q_noi"],
            comment_loader=lambda i, q: {},
            manual_loader=lambda i, q: manual,
        )
    finally:
        L.MANUAL_RATIO_SEEDS = saved


def main() -> int:
    ti = api.token_info()
    print(f"LIVE token={ti['username']} ({ti['hours_left']}h)  "
          f"build={api.get('/api/data/version').get('version')}")
    print(f"scope: {INV} {Q}\n")

    # ---- live inputs -----------------------------------------------------
    # The resolved population comes from the live Step 1 endpoint, whose payload
    # is already the shape assemble_loan expects — no local ownership walk.
    resolved = api.get("/api/portfolio-snapshot/deals",
                       params={"investor": INV, "quarter": Q})
    inv = pd.DataFrame(api.get("/api/data/deals/all").get("deals") or [])
    loans = pd.DataFrame(api.get("/api/data/tables/loans/rows",
                                 params={"page": 1, "page_size": 500}
                                 ).get("rows") or [])
    vals = pd.DataFrame(api.get("/api/data/tables/valuations/rows",
                                params={"page": 1, "page_size": 500}
                                ).get("rows") or [])
    n_deals = sum(len(v) for v in resolved["groups"].values())
    print(f"Step 1: {n_deals} deals; loans {len(loans)}, valuations {len(vals)}\n")

    op_cache: dict = {}

    def one_pager(vc, q):
        if (vc, q) not in op_cache:
            op_cache[(vc, q)] = api.get(f"/api/financials/{vc}/one-pager",
                                        params={"quarter": q})
        return op_cache[(vc, q)]

    q_cache: dict = {}

    def q_noi(vc, q):
        if (vc, q) in q_cache:
            return q_cache[(vc, q)]
        yr, qn = int(q.split("-Q")[0]), int(q.split("Q")[1])
        qend = pd.Timestamp(year=yr, month=qn * 3, day=1) + pd.offsets.MonthEnd(0)
        ch = api.get(f"/api/financials/{vc}/performance-chart",
                     params={"freq": "Quarterly", "periods": 12,
                             "period_end": str(qend.date())})
        val = None
        for lbl, a in zip(ch.get("periods") or [], ch.get("actual_noi") or []):
            if lbl == f"Q{qn} {yr}" and a is not None:
                val = a
                break
        q_cache[(vc, q)] = val
        return val

    providers = {"resolved": resolved, "one_pager": one_pager, "loans": loans,
                 "valuations": vals, "inv": inv, "q_noi": q_noi}

    SEEDS = dict(L.MANUAL_RATIO_SEEDS)

    print("Building AFTER (seeds live) — this pulls one One Pager per deal…")
    after = build({}, SEEDS, providers)
    print("Building BEFORE (seed table emptied, same cached inputs)…\n")
    before = build({}, {}, providers)

    fa, fb = flatten(after), flatten(before)

    # ---- 1. the six cells, before -> after -------------------------------
    print("=" * 100)
    print("BEFORE -> AFTER, the six deals")
    print(f"{'vcode':<10}{'deal':<30}{'column':<12}{'before':>14}{'after':>12}"
          f"   source")
    print("-" * 100)
    for vc, fields in EXPECTED.items():
        rb, ra = fb.get(vc) or {}, fa.get(vc) or {}
        if not ra:
            chk(f"{vc} present on the page", False)
            continue
        for f, (want_val, want_disp) in fields.items():
            b = rb.get(f"{f}_display")
            a = ra.get(f"{f}_display")
            def s(v):
                if v is None:
                    return "—"
                return v if isinstance(v, str) else (
                    f"{v * 100:.1f}%" if f != "ytd_dscr" else f"{v:.1f}x")
            print(f"{vc:<10}{str(ra.get('name'))[:29]:<30}{f:<12}"
                  f"{s(b):>14}{s(a):>12}   {ra.get(f'{f}_source')}")
            chk(f"{vc} {f} is typeable", ra.get(f"{f}_is_manual") is True)
            chk(f"{vc} {f} holds {want_val}",
                ra.get(f"{f}_manual") == want_val)
            chk(f"{vc} {f} displays {want_disp!r}",
                ra.get(f"{f}_display") == want_disp)
            chk(f"{vc} {f} reports the pre-filled source",
                ra.get(f"{f}_source") == L.MANUAL_SOURCE_SEED
                and ra.get(f"{f}_entered") is False)
            chk(f"{vc} {f} keeps the computed figure as *_computed",
                ra.get(f"{f}_computed") == rb.get(f))

    # The two deals that keep their computed DSCR / Debt Yield.
    for vc in ("P0000109", "P0000116"):
        ra = fa.get(vc) or {}
        chk(f"{vc} keeps a COMPUTED DSCR and Debt Yield (LTV only is typed)",
            not ra.get("ytd_dscr_is_manual")
            and not ra.get("debt_yield_is_manual")
            and ra.get("ytd_dscr") is not None
            and ra.get("debt_yield") is not None)
    # The one cell that replaces a real computed number, stated rather than hidden.
    p119 = fa.get("P0000119") or {}
    print(f"\n  P0000119 DSCR: typed {p119.get('ytd_dscr_display')} over a "
          f"computed {p119.get('ytd_dscr_computed')!r}")
    chk("P0000119's typed DSCR replaces a computed figure, which survives "
        "as *_computed",
        p119.get("ytd_dscr_computed") is not None
        and p119.get("ytd_dscr_display") == "1.1x")

    # ---- 2. raw fields and aggregates do not move ------------------------
    print("\n" + "=" * 100)
    print("NOTHING ELSE MOVED")
    print("=" * 100)
    for f in L.MANUAL_RATIO_FIELDS:
        chk(f"raw {f} identical on EVERY deal, typed ones included",
            all(fa[v].get(f) == fb[v].get(f) for v in fb))
    chk("fund subtotals identical",
        after["subtotals"] == before["subtotals"])
    chk("portfolio total identical", after["total"] == before["total"])
    for k, v in after["subtotals"].items():
        ltv = v.get("ltv")
        print(f"    subtotal {k:<26} LTV "
              f"{('%.1f%%' % (ltv * 100)) if ltv else '—':>8}  "
              f"(weighting {v.get('ltv_n')} deals)")

    # Every non-typed deal must be byte-identical; a typed deal may differ ONLY
    # in the keys the overlay owns.
    #
    # Two sets, and the difference matters: `*_display` EXISTS on every row
    # already, so it belongs in what a typed row is allowed to change but not
    # in what only a typed row may carry. Conflating them asserted that no
    # other deal has an `ltv_display`, which every one of them does.
    overlay_added = {f"{f}{suf}" for f in L.MANUAL_RATIO_FIELDS
                     for suf in ("_manual", "_is_manual", "_source",
                                 "_entered", "_computed")}
    overlay_keys = overlay_added | {f"{f}_display"
                                    for f in L.MANUAL_RATIO_FIELDS}
    untouched, drifted = [], []
    for vc in fb:
        if vc in EXPECTED:
            continue
        diff = [k for k in set(fa[vc]) | set(fb[vc])
                if fa[vc].get(k) != fb[vc].get(k)]
        (drifted if diff else untouched).append((vc, diff))
    print(f"    {len(untouched)} non-typed deals unchanged, "
          f"{len(drifted)} drifted")
    for vc, diff in drifted:
        print(f"      {vc}: {diff}")
    chk("every deal outside the six is byte-identical", not drifted)
    chk("no deal outside the six gained a typed-cell key",
        not any(k in overlay_added for vc in fb if vc not in EXPECTED
                for k in fa[vc]))
    for vc in EXPECTED:
        diff = {k for k in set(fa[vc]) | set(fb[vc])
                if fa[vc].get(k) != fb[vc].get(k)}
        chk(f"{vc} differs ONLY in overlay keys (and its flags note)",
            diff <= overlay_keys | {"flags"})

    # ---- 3. stored entry vs seed vs cleared ------------------------------
    print("\n" + "=" * 100)
    print("STORED ENTRY BEATS THE SEED; A CLEARED CELL STAYS CLEARED")
    print("=" * 100)
    typed_in = build({"P0000117": {"ltv": 71.25, "ytd_dscr": None}},
                     SEEDS, providers)
    t = flatten(typed_in)["P0000117"]
    print(f"    P0000117 after an entry: ltv={t['ltv_display']!r} "
          f"(source {t['ltv_source']!r}), dscr={t['ytd_dscr_display']!r}, "
          f"dy={t['debt_yield_display']!r} (source {t['debt_yield_source']!r})")
    # "71.25%", not "71.2%": the second decimal is real, so _manual_fmt keeps
    # it — the same rule that keeps Presidential Arms' 5.93%.
    chk("an entered LTV replaces the seed and reports it was entered",
        t["ltv_manual"] == 71.25 and t["ltv_display"] == "71.25%"
        and t["ltv_source"] == L.MANUAL_SOURCE_ENTERED
        and t["ltv_entered"] is True)
    chk("a CLEARED DSCR renders an em dash, NOT the seed",
        t["ytd_dscr_manual"] is None and t["ytd_dscr_display"] is None
        and t["ytd_dscr_entered"] is True)
    chk("an untouched cell on the same deal still shows its seed",
        t["debt_yield_manual"] == 12.1
        and t["debt_yield_source"] == L.MANUAL_SOURCE_SEED)
    chk("a stored value for a deal with no seeds is ignored by this subtab",
        "ltv_is_manual" not in (flatten(
            build({"P0000030": {"ltv": 55.0}}, SEEDS, providers)
        )["P0000030"]))

    # ---- 4. the literals still outrank a typed cell ----------------------
    print("\n" + "=" * 100)
    print("DEBT-FREE AND DEV LITERALS STILL WIN")
    print("=" * 100)
    peg, jb = "P0000066", "P0000021"          # Pegasus (debt free), JB (dev)
    forced = flatten(build({}, {**SEEDS,
                            peg: {"ltv": 50.0},
                            jb: {"debt_yield": 9.0}}, providers))
    print(f"    Pegasus ltv_display={forced[peg]['ltv_display']!r}, "
          f"JB Fair Park debt_yield_display="
          f"{forced[jb]['debt_yield_display']!r}")
    chk("a seed on the debt-free deal still reads 'N/A'",
        forced[peg]["ltv_display"] == L.NA_DISPLAY)
    chk("a seed on a development deal still reads 'Dev'",
        forced[jb]["debt_yield_display"] == L.DEV_DISPLAY)
    chk("neither of the six is dev or debt-free, so precedence is not "
        "silently doing work here",
        all(not (fa[v].get("is_dev") or fa[v].get("debt_free"))
            for v in EXPECTED))

    # ---- 5. formatting rule ---------------------------------------------
    print("\n" + "=" * 100)
    print("UNITS AND FORMATTING")
    print("=" * 100)
    chk("ltv formats percentage points", L.format_manual_ratio("ltv", 69.0) == "69.0%")
    chk("debt_yield formats percentage points, keeping a real second decimal",
        L.format_manual_ratio("debt_yield", 5.93) == "5.93%"
        and L.format_manual_ratio("debt_yield", 8.9) == "8.9%")
    chk("ytd_dscr formats a multiple",
        L.format_manual_ratio("ytd_dscr", 1.9) == "1.9x")
    chk("None formats as no value (not a 'pending' literal)",
        L.format_manual_ratio("ltv", None) is None)
    chk("manual_ratio_fields is empty for a deal with no seeds",
        L.manual_ratio_fields("P0000030") == ())
    chk("manual_ratio_fields is case- and whitespace-insensitive",
        L.manual_ratio_fields(" p0000117 ")
        == ("ltv", "ytd_dscr", "debt_yield"))

    # ---- 6. persistence accepts the three fields ------------------------
    print("\n" + "=" * 100)
    print("PERSISTENCE (throwaway SQLite — no application database touched)")
    print("=" * 100)
    tmp = os.path.join(tempfile.mkdtemp(prefix="ps_loan_manual_"), "t.db")
    eng = sqlalchemy.create_engine(f"sqlite:///{tmp}")
    P._engine = lambda: eng                       # type: ignore[assignment]
    P._is_postgres = lambda: False                # type: ignore[assignment]
    for f, v in (("ltv", 69.0), ("ytd_dscr", 1.9), ("debt_yield", 12.1)):
        P.save_value(INV, Q, "P0000117", f, v, updated_by="guardrail")
    got = {r["field"]: r["value"] for r in
           P.get_elements("value", INV, Q, deal_vcode="P0000117")}
    print(f"    stored: {got}")
    chk("all three round-trip as numbers",
        got == {"ltv": 69.0, "ytd_dscr": 1.9, "debt_yield": 12.1})
    P.save_value(INV, Q, "P0000117", "ltv", None, updated_by="guardrail")
    cleared = P.get_elements("value", INV, Q, deal_vcode="P0000117",
                             field="ltv")
    chk("clearing a cell stores a NULL rather than deleting the row",
        len(cleared) == 1 and cleared[0]["value"] is None)
    chk("the loader reads the three fields back into {vcode: {field: value}}",
        L._default_manual_loader(INV, Q).get("P0000117", {}).get("ytd_dscr")
        == 1.9)
    chk("the loader keeps a Net ROE entry OUT of the ratio map",
        (P.save_value(INV, Q, "P0000117", "net_roe", 4.4) or True)
        and "net_roe" not in L._default_manual_loader(INV, Q)["P0000117"])
    try:
        P.save_value(INV, Q, "P0000117", "bogus", 1.0)
        chk("an unknown field is still rejected", False)
    except ValueError:
        chk("an unknown field is still rejected", True)

    ok = sum(1 for _, c in CHECKS if c)
    print(f"\n  {ok}/{len(CHECKS)} checks passed — "
          f"{'ALL PASS' if ok == len(CHECKS) else 'FAILURES PRESENT'}")
    return 0 if ok == len(CHECKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
