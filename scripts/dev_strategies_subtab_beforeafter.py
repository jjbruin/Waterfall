"""Before/after for the DEV_STRATEGIES change, on the REAL subtab assemblies.

Runs the committed ``assemble_operating`` / ``assemble_loan`` /
``assemble_financial`` and the committed ``get_property_performance`` on each
side of the change — ``capture before`` from a worktree at main, ``capture
after`` from the working tree — then ``report`` diffs them.

Live inputs, read-only. The One Pager provider is the LIVE endpoint, so both
sides see identical data and the only variable is the code.

Usage:
    python scripts/dev_strategies_subtab_beforeafter.py capture before
    python scripts/dev_strategies_subtab_beforeafter.py capture after
    python scripts/dev_strategies_subtab_beforeafter.py report
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import live_api                                     # noqa: E402

OUT = os.environ.get(
    "WF_NC_DIR",
    os.path.join(os.environ.get("TEMP", "/tmp"), "dev_nc_beforeafter"))
QUARTER = os.environ.get("WF_CHECK_QUARTER", "2026-Q1")

PEGASUS, BELLEVILLE = "P0000066", "P0000006"

#: Investors whose reports carry the two deals, plus one that carries neither
#: (a control: nothing on it may move).
INVESTORS = ["TGAM", "PSC3", "WRI"]


def _one_pager(vcode, quarter):
    """Live One Pager payload — identical on both sides, so only code varies."""
    return live_api.get(f"/api/financials/{vcode}/one-pager",
                        params={"quarter": quarter}) or {}


def _quarter_noi(vcode, quarter):
    """That quarter's periodic NOI, for the Debt Yield column.

    The app builds this from the ISBS cache (``portfolio_snapshot_freeze.
    _quarterly_noi_provider``). Here it is read off the One Pager's YTD NOI,
    which is EXACT at Q1 and only at Q1: year-to-date through March is the
    quarter. Verified against live — Belleville reports quarter_noi 244,741 and
    ytd_noi 244,741 for 2026-Q1.

    Without a provider ``assemble_loan`` leaves Debt Yield at None for every
    deal, which silently hid the very cell this change is meant to restore.
    """
    if not str(quarter).endswith("Q1"):
        raise RuntimeError(
            f"_quarter_noi is only exact at Q1; got {quarter!r}")
    pp = (_one_pager(vcode, quarter) or {}).get("property_performance") or {}
    return (pp.get("noi") or {}).get("ytd_actual")


def capture(label):
    import pandas as pd
    from flask_app.services.portfolio_snapshot_service import (
        resolve_investor_deals)
    from flask_app.services.portfolio_snapshot_operating import (
        assemble_operating)
    from flask_app.services.portfolio_snapshot_loan import assemble_loan
    from flask_app.services.portfolio_snapshot_financial import (
        assemble_financial)

    os.makedirs(OUT, exist_ok=True)

    def page(table, params=None, size=500, cap=60):
        out, seen, n = [], set(), 1
        while n <= cap:
            d = live_api.get(f"/api/data/tables/{table}/rows",
                             params={"page": n, "page_size": size,
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
            if new == 0 or len(rows) < size:
                break
            n += 1
        return pd.DataFrame(out)

    inv = page("deals", {"sort": "vcode", "order": "asc"})
    rel = page("relationships")
    loans = page("loans")
    vals = page("valuations")

    got = {"quarter": QUARTER, "investors": {}, "one_pager": {}}

    # The One Pager itself, for the two deals — the At-Close question.
    for vc in (PEGASUS, BELLEVILLE):
        pp = (_one_pager(vc, QUARTER) or {}).get("property_performance") or {}
        got["one_pager"][vc] = {
            k: {c: (pp.get(k) or {}).get(c)
                for c in ("at_close", "ytd_actual", "actual_ye", "uw_ye")}
            for k in ("revenue", "expenses", "noi", "economic_occ", "dscr")}

    for code in INVESTORS:
        try:
            resolved = resolve_investor_deals(code, QUARTER, rel, inv)
        except Exception as exc:
            got["investors"][code] = {"error": str(exc)[:120]}
            continue
        blk = {}
        try:
            op = assemble_operating(code, QUARTER, resolved=resolved,
                                    one_pager_provider=_one_pager)
            blk["operating"] = _rows(op)
            blk["operating_diag"] = op.get("diagnostics")
            blk["operating_total"] = op.get("total")
        except Exception as exc:
            blk["operating_error"] = f"{type(exc).__name__}: {exc}"[:160]
        try:
            ln = assemble_loan(code, QUARTER, resolved=resolved,
                               one_pager_provider=_one_pager,
                               loans=loans, valuations=vals, inv=inv,
                               quarterly_noi_provider=_quarter_noi)
            blk["loan"] = _rows(ln)
            blk["loan_diag"] = ln.get("diagnostics")
            blk["loan_total"] = ln.get("total")
            blk["loan_subtotals"] = ln.get("subtotals")
        except Exception as exc:
            blk["loan_error"] = f"{type(exc).__name__}: {exc}"[:160]
        try:
            fin = assemble_financial(code, QUARTER, resolved=resolved,
                                     one_pager_provider=_one_pager)
            blk["financial"] = _rows(fin)
            blk["financial_diag"] = fin.get("diagnostics")
            blk["excluding_dev"] = fin.get("total_excluding_dev")
        except Exception as exc:
            blk["financial_error"] = f"{type(exc).__name__}: {exc}"[:160]
        got["investors"][code] = blk

    path = os.path.join(OUT, f"{label}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(got, fh, indent=1, default=str)
    print(f"captured -> {path}")


def _rows(sub):
    out = {}

    def walk(o):
        if isinstance(o, dict):
            if o.get("vcode") and ("is_dev" in o or "name" in o):
                out[o["vcode"]] = o
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(sub.get("groups"))
    walk(sub.get("ownership_flagged"))
    return out


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
            return "—"
        if isinstance(v, str):
            return v
        try:
            return f"{float(v):,.0f}"
        except (TypeError, ValueError):
            return str(v)

    def row(side, inv_code, sub, vc):
        return ((side.get("investors", {}).get(inv_code, {}) or {})
                .get(sub, {}) or {}).get(vc, {})

    def find(side, sub, vc):
        for code in INVESTORS:
            r = row(side, code, sub, vc)
            if r:
                return code, r
        return None, {}

    print("BEFORE / AFTER — real subtab assemblies, live data both sides")
    print("=" * 100)

    # ── Belleville: the win ───────────────────────────────────────────────
    print("\nBELLEVILLE SELF STORAGE (P0000006) — OPERATING")
    print("-" * 100)
    ci, ob = find(b, "operating", BELLEVILLE)
    _, oa = find(a, "operating", BELLEVILLE)
    print(f"  (from investor report {ci})")
    print(f"  {'field':<26}{'before':>26}{'after':>26}")
    print(f"  {'is_dev':<26}{str(ob.get('is_dev')):>26}{str(oa.get('is_dev')):>26}")
    print(f"  {'econ_occ_display':<26}{f(ob.get('econ_occ_display')):>26}"
          f"{f(oa.get('econ_occ_display')):>26}")
    for k in ("at_close", "projected_ye", "uw_ye"):
        print(f"  {'noi_display.' + k:<26}"
              f"{f((ob.get('noi_display') or {}).get(k)):>26}"
              f"{f((oa.get('noi_display') or {}).get(k)):>26}")
    print(f"  {'expected_growth_display':<26}{f(ob.get('expected_growth_display')):>26}"
          f"{f(oa.get('expected_growth_display')):>26}")
    print(f"  {'actual_growth_display':<26}{f(ob.get('actual_growth_display')):>26}"
          f"{f(oa.get('actual_growth_display')):>26}")

    chk("Belleville was dev, is now operating",
        ob.get("is_dev") is True and oa.get("is_dev") is False)
    chk("Belleville Econ Occ was 'n/a', now a real 93.2%",
        ob.get("econ_occ_display") == "n/a"
        and isinstance(oa.get("econ_occ_display"), (int, float))
        and abs(float(oa["econ_occ_display"]) - 93.218) < 0.01,
        f"after={oa.get('econ_occ_display')!r}")
    nb, na = ob.get("noi_display") or {}, oa.get("noi_display") or {}
    chk("Belleville NOI was 'n/a' in all three columns",
        all(nb.get(k) == "n/a" for k in ("at_close", "projected_ye", "uw_ye")),
        str(nb))
    chk("Belleville NOI now real: 70,716 / 1,141,974 / 1,135,963",
        abs(float(na.get("at_close") or 0) - 70716.21) < 1
        and abs(float(na.get("projected_ye") or 0) - 1141973.54) < 1
        and abs(float(na.get("uw_ye") or 0) - 1135963.01) < 1, str(na))
    chk("Belleville is no longer listed as dev-suppressed",
        not (oa.get("dev_suppressed_columns") or []))

    print("\nBELLEVILLE SELF STORAGE (P0000006) — LOAN")
    print("-" * 100)
    ci, lb = find(b, "loan", BELLEVILLE)
    _, la = find(a, "loan", BELLEVILLE)
    print(f"  {'field':<26}{'before':>26}{'after':>26}")
    for k in ("is_dev", "debt", "debt_basis", "valuation",
              "ltv_display", "ytd_dscr_display", "debt_yield_display"):
        print(f"  {k:<26}{f(lb.get(k)):>26}{f(la.get(k)):>26}")

    chk("Belleville LTV was the literal 'Dev', is now a real ~82.6%",
        lb.get("ltv_display") == "Dev"
        and isinstance(la.get("ltv_display"), (int, float))
        and abs(float(la["ltv_display"]) - 0.8257) < 0.002,
        f"after={la.get('ltv_display')!r}")
    chk("Belleville YTD DSCR was 'Dev', is now a real ~0.708",
        lb.get("ytd_dscr_display") == "Dev"
        and isinstance(la.get("ytd_dscr_display"), (int, float))
        and abs(float(la["ytd_dscr_display"]) - 0.7079) < 0.002,
        f"after={la.get('ytd_dscr_display')!r}")
    chk("Belleville Debt Yield was 'Dev', is now a real ~5.44%",
        lb.get("debt_yield_display") == "Dev"
        and isinstance(la.get("debt_yield_display"), (int, float))
        and abs(float(la["debt_yield_display"]) - 0.05439) < 0.0005,
        f"after={la.get('debt_yield_display')!r}")
    chk("Belleville Debt is unchanged in VALUE (18.0M both sides) — only the "
        "basis label moves from committed facility to ISBS",
        abs(float(lb.get("debt") or 0) - float(la.get("debt") or 0)) < 1,
        f"{lb.get('debt')} -> {la.get('debt')}")

    # ── Pegasus: Debt must stay a dash, At-Close must return ──────────────
    print("\nPEGASUS LIFE STORAGE (P0000066) — FINANCIAL Debt")
    print("-" * 100)
    ci, fb = find(b, "financial", PEGASUS)
    _, fa = find(a, "financial", PEGASUS)
    print(f"  {'field':<26}{'before':>26}{'after':>26}")
    for k in ("is_dev", "debt", "debt_isbs", "debt_basis",
              "debt_display", "pdf_na_cells", "net_roe_display"):
        print(f"  {k:<26}{f(fb.get(k)):>26}{f(fa.get(k)):>26}")

    chk("Pegasus debt_display was None (em dash) BEFORE — a by-product of "
        "the dev branch in resolve_debt",
        fb.get("debt_display") is None)
    # THE REQUIREMENT is "not $0.0". PDF_NA_CELLS renders the literal "n/a",
    # not an em dash: disp() returns a string verbatim and only None becomes
    # DASH. That makes Pegasus read exactly like City West, the app's only
    # other debt-free blanked deal, which is the consistent answer.
    chk("Pegasus debt_display is NOT the number 0.0 — the regression is averted",
        not isinstance(fa.get("debt_display"), (int, float)),
        f"after={fa.get('debt_display')!r} — regressed to $0.0")
    chk("Pegasus debt_display is the n/a literal, same as City West",
        fa.get("debt_display") == "n/a", f"after={fa.get('debt_display')!r}")
    chk("the raw debt did change to 0.0 (so the dash is now deliberate, "
        "not a by-product of the dev branch)",
        fa.get("debt") == 0.0, f"raw debt={fa.get('debt')!r}")
    chk("Pegasus pdf_na_cells now names debt",
        "debt" in (fa.get("pdf_na_cells") or []),
        str(fa.get("pdf_na_cells")))
    chk("Pegasus Net ROE still prompts as 'pending entry', not n/a",
        fa.get("net_roe_display") == "pending entry",
        f"after={fa.get('net_roe_display')!r}")

    # ── Pegasus LOAN: debt free, N/A everywhere, and NOT "Dev" ────────────
    print("\nPEGASUS LIFE STORAGE (P0000066) — LOAN (debt-free display)")
    print("-" * 100)
    ci, plb = find(b, "loan", PEGASUS)
    _, pla = find(a, "loan", PEGASUS)
    print(f"  {'field':<26}{'before':>26}{'after':>26}")
    for k in ("is_dev", "debt_free", "debt", "debt_display", "loan_count",
              "rate_display", "maturity_display", "ltv_display",
              "ytd_dscr_display", "debt_yield_display", "dev_no_data"):
        print(f"  {k:<26}{f(plb.get(k)):>26}{f(pla.get(k)):>26}")

    chk("Pegasus is flagged debt_free", pla.get("debt_free") is True)
    chk("Pegasus Loan Debt renders a DASH (debt_display is None) — not $0.0",
        "debt_display" in pla and pla.get("debt_display") is None,
        f"debt_display={pla.get('debt_display')!r}")
    chk("Pegasus raw debt is still the real 0.0, so subtotals are untouched",
        pla.get("debt") == 0.0, f"raw debt={pla.get('debt')!r}")
    for col in ("ltv_display", "ytd_dscr_display", "debt_yield_display",
                "rate_display", "maturity_display"):
        chk(f"Pegasus {col} is the literal 'N/A'",
            pla.get(col) == "N/A", f"{col}={pla.get(col)!r}")
    chk("Pegasus shows N/A and NEVER 'Dev' on any loan column",
        not any(pla.get(k) == "Dev" for k in
                ("ltv_display", "ytd_dscr_display", "debt_yield_display",
                 "rate_display", "maturity_display")))
    chk("Pegasus is not classified dev, so the N/A is the debt-free rule and "
        "not a dev gate", pla.get("is_dev") is False)

    print("\nPEGASUS LIFE STORAGE (P0000066) — AT-CLOSE (One Pager, live)")
    print("-" * 100)
    print(f"  {'field':<26}{'before':>26}{'after':>26}")
    for blk in ("revenue", "expenses", "noi"):
        print(f"  {blk + '.at_close':<26}"
              f"{f((b['one_pager'][PEGASUS].get(blk) or {}).get('at_close')):>26}"
              f"{f((a['one_pager'][PEGASUS].get(blk) or {}).get('at_close')):>26}")
    print("  (the One Pager block above is the LIVE deployed build on both "
          "sides — identical by construction. The local gate is asserted "
          "directly below, and the Operating subtab rows are the real\n   "
          "local computation.)")

    # The At-Close gate is local code; assert it directly.
    from one_pager import (AT_CLOSE_YEAR0_DEV_ONLY,
                           _at_close_force_suppressed)
    chk("the At-Close Year-0 gate is still dev-only", AT_CLOSE_YEAR0_DEV_ONLY)
    chk("Pegasus is named in AT_CLOSE_FORCE_SUPPRESS, so un-tagging it cannot "
        "un-zero its At-Close column", _at_close_force_suppressed(PEGASUS))
    chk("the force list is scoped — Belleville is NOT in it, so it keeps its "
        "real At-Close", not _at_close_force_suppressed(BELLEVILLE))
    _, prow = find(a, "operating", PEGASUS)
    chk("Pegasus is operating on the Operating subtab (the dev tag is gone)",
        prow.get("is_dev") is False, f"is_dev={prow.get('is_dev')}")

    # ── Item 2: genuine dev deals are FORCED to "Dev" on the three ────────
    print("\n" + "=" * 100)
    print("GENUINE DEV DEALS — LTV / DSCR / Debt Yield forced to 'Dev'; "
          "Rate / Maturity / Debt real")
    print("=" * 100)
    print(f"  {'vcode':<11}{'name':<27}{'LTV':>9}{'DSCR':>7}{'DY':>7}"
          f"{'Rate':>16}{'Maturity':>12}{'Debt':>14}")
    dev_seen = {}
    for code in INVESTORS:
        for vc, r in ((a["investors"].get(code, {}) or {})
                      .get("loan", {}) or {}).items():
            if r.get("is_dev"):
                dev_seen[vc] = r
    for vc, r in sorted(dev_seen.items()):
        print(f"  {vc:<11}{str(r.get('name'))[:25]:<27}"
              f"{f(r.get('ltv_display')):>9}{f(r.get('ytd_dscr_display')):>7}"
              f"{f(r.get('debt_yield_display')):>7}"
              f"{str(r.get('rate_display')):>16}"
              f"{str(r.get('maturity_display')):>12}"
              f"{f(r.get('debt_display')):>14}")

    chk(f"at least one dev deal is on the reports ({len(dev_seen)} seen)",
        bool(dev_seen))
    # WATERS_CREEK_LTV_EXCEPTION is a deliberate creator instruction that keeps
    # a real LTV on one dev deal; it is required for the TGA22 fund LTV
    # subtotal to reproduce the published 60.4%. Excluded by name, not by a
    # blanket allowance, so a second exemption appearing would fail here.
    WC = "P0000078"
    chk("every dev deal shows 'Dev' for YTD DSCR",
        all(r.get("ytd_dscr_display") == "Dev" for r in dev_seen.values()),
        str({v: r.get("ytd_dscr_display") for v, r in dev_seen.items()
             if r.get("ytd_dscr_display") != "Dev"}))
    chk("every dev deal shows 'Dev' for Debt Yield",
        all(r.get("debt_yield_display") == "Dev" for r in dev_seen.values()),
        str({v: r.get("debt_yield_display") for v, r in dev_seen.items()
             if r.get("debt_yield_display") != "Dev"}))
    chk("every dev deal shows 'Dev' for LTV, bar the Waters Creek exception",
        all(r.get("ltv_display") == "Dev"
            for v, r in dev_seen.items() if v != WC),
        str({v: r.get("ltv_display") for v, r in dev_seen.items()
             if v != WC and r.get("ltv_display") != "Dev"}))
    chk("Waters Creek is still the ONLY dev deal with a numeric LTV",
        [v for v, r in dev_seen.items()
         if isinstance(r.get("ltv_display"), (int, float))] in ([], [WC]),
        str([v for v, r in dev_seen.items()
             if isinstance(r.get("ltv_display"), (int, float))]))
    chk("no dev deal lost its Rate or Maturity to a 'Dev' or 'N/A' literal",
        all(r.get("rate_display") not in ("Dev", "N/A")
            and r.get("maturity_display") not in ("Dev", "N/A")
            for r in dev_seen.values()),
        str({v: (r.get("rate_display"), r.get("maturity_display"))
             for v, r in dev_seen.items()
             if r.get("rate_display") in ("Dev", "N/A")
             or r.get("maturity_display") in ("Dev", "N/A")}))
    chk("no dev deal lost its Debt to a literal — all still numeric",
        all(isinstance(r.get("debt_display"), (int, float))
            for r in dev_seen.values()),
        str({v: r.get("debt_display") for v, r in dev_seen.items()
             if not isinstance(r.get("debt_display"), (int, float))}))
    chk("no dev deal is suppressed to a bare dash on the three ratio columns "
        "(the old dev_no_data escape is gone)",
        all(r.get(k) is not None for r in dev_seen.values()
            for k in ("ltv_display", "ytd_dscr_display",
                      "debt_yield_display")))

    # ── no NON-dev deal picked up a literal ───────────────────────────────
    # Only the SUPPRESSION literals count. Rate and Maturity are formatted
    # strings on every deal ("5.4% fixed", "10/15/2034") — treating any string
    # as a suppression flagged all 31 operating deals and said nothing.
    SUPPRESSION = {"Dev", "N/A", "n/a"}
    print("\n  NON-DEV deals carrying a suppression literal "
          f"{sorted(SUPPRESSION)} in any loan column (only Pegasus may):")
    leaked_lit = {}
    for code in INVESTORS:
        for vc, r in ((a["investors"].get(code, {}) or {})
                      .get("loan", {}) or {}).items():
            if r.get("is_dev"):
                continue
            lits = {k: r.get(k) for k in
                    ("ltv_display", "ytd_dscr_display", "debt_yield_display",
                     "rate_display", "maturity_display", "debt_display")
                    if r.get(k) in SUPPRESSION}
            if lits:
                leaked_lit[vc] = lits
                print(f"    {vc:<11}{str(r.get('name'))[:26]:<28}{lits}")
    chk("only Pegasus carries a suppression literal among the non-dev deals",
        set(leaked_lit) <= {PEGASUS},
        f"also: {sorted(set(leaked_lit) - {PEGASUS})}")
    # The three ratio columns are numeric-or-literal, so a string there on a
    # non-dev, non-debt-free deal would be a leak of any kind, not just of the
    # known words.
    ratio_str = {}
    for code in INVESTORS:
        for vc, r in ((a["investors"].get(code, {}) or {})
                      .get("loan", {}) or {}).items():
            if r.get("is_dev") or r.get("debt_free"):
                continue
            bad = {k: r.get(k) for k in ("ltv_display", "ytd_dscr_display",
                                         "debt_yield_display")
                   if isinstance(r.get(k), str)}
            if bad:
                ratio_str[vc] = bad
    chk("no ordinary operating deal has a STRING in LTV / DSCR / Debt Yield — "
        "they stay numeric or an em dash", not ratio_str, str(ratio_str))
    print("\n  NON-DEV deals whose Debt renders a dash (only Pegasus may):")
    dashed = {}
    for code in INVESTORS:
        for vc, r in ((a["investors"].get(code, {}) or {})
                      .get("loan", {}) or {}).items():
            if not r.get("is_dev") and ("debt_display" in r
                                        and r.get("debt_display") is None):
                dashed[vc] = r.get("debt")
                print(f"    {vc:<11}{str(r.get('name'))[:26]:<28}"
                      f"raw debt={r.get('debt')!r}")
    chk("only Pegasus renders a dashed Debt among the non-dev deals",
        set(dashed) <= {PEGASUS}, f"also: {sorted(set(dashed) - {PEGASUS})}")

    # ── Pegasus Operating / Summary unchanged ─────────────────────────────
    print("\nPEGASUS — OPERATING (was carried by DEV_DISPLAY_EXCEPTIONS)")
    print("-" * 100)
    ci, pb = find(b, "operating", PEGASUS)
    _, pa = find(a, "operating", PEGASUS)
    print(f"  {'field':<26}{'before':>26}{'after':>26}")
    print(f"  {'is_dev':<26}{str(pb.get('is_dev')):>26}{str(pa.get('is_dev')):>26}")
    print(f"  {'econ_occ_display':<26}{f(pb.get('econ_occ_display')):>26}"
          f"{f(pa.get('econ_occ_display')):>26}")
    for k in ("at_close", "projected_ye", "uw_ye"):
        print(f"  {'noi_display.' + k:<26}"
              f"{f((pb.get('noi_display') or {}).get(k)):>26}"
              f"{f((pa.get('noi_display') or {}).get(k)):>26}")
    print(f"  {'expected_growth_display':<26}{f(pb.get('expected_growth_display')):>26}"
          f"{f(pa.get('expected_growth_display')):>26}")

    chk("Pegasus Econ Occ unchanged (was via the exception, now ordinary)",
        pb.get("econ_occ_display") == pa.get("econ_occ_display"),
        f"{pb.get('econ_occ_display')} -> {pa.get('econ_occ_display')}")
    chk("Pegasus NOI unchanged in all three columns",
        (pb.get("noi_display") or {}) == (pa.get("noi_display") or {}),
        f"{pb.get('noi_display')} -> {pa.get('noi_display')}")

    # ── nothing else moved ────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("BLAST RADIUS — every other deal on every subtab")
    print("=" * 100)
    # NEW KEYS are excluded from the comparison and asserted separately. The
    # change ADDS `debt_display` and `debt_free` to every loan row, so a raw
    # dict comparison would report all 42 deals as moved and drown the real
    # signal. Compared on the keys the two sides share; the new keys are then
    # checked on their own terms (above, and in the two "only Pegasus" checks).
    NEW_KEYS = {"debt_display", "debt_free"}
    moved = []
    new_key_report = {}
    for code in INVESTORS:
        for sub in ("operating", "loan", "financial"):
            rb = (b["investors"].get(code, {}) or {}).get(sub, {}) or {}
            ra = (a["investors"].get(code, {}) or {}).get(sub, {}) or {}
            chk(f"{code}/{sub}: same deal population",
                set(rb) == set(ra), f"{set(rb) ^ set(ra)}")
            for vc in sorted(set(rb) & set(ra)):
                shared = (set(rb[vc]) & set(ra[vc])) - NEW_KEYS
                if {k: rb[vc][k] for k in shared} != {
                        k: ra[vc][k] for k in shared}:
                    moved.append((code, sub, vc))
                    if vc not in (PEGASUS, BELLEVILLE):
                        diffs = {k: (rb[vc][k], ra[vc][k]) for k in shared
                                 if rb[vc][k] != ra[vc][k]}
                        new_key_report[(sub, vc)] = diffs
    others = sorted({(s, v) for _, s, v in moved
                     if v not in (PEGASUS, BELLEVILLE)})
    chk("ONLY Pegasus and Belleville moved on any subtab, on any report "
        "(comparing the keys both sides share)",
        not others, f"also moved: {others}\n         {new_key_report}")

    print(f"\n  rows that moved: "
          f"{sorted({(s, v) for _, s, v in moved})}")

    # The new keys, on every deal — a value, not just an absence.
    only_new = []
    for code in INVESTORS:
        ra = (a["investors"].get(code, {}) or {}).get("loan", {}) or {}
        for vc, r in ra.items():
            if not NEW_KEYS <= set(r):
                only_new.append(vc)
    chk("every loan row carries the new debt_display / debt_free keys",
        not only_new, f"missing on: {sorted(set(only_new))}")

    # dev counts and the excluding-dev row
    print("\n" + "=" * 100)
    print("DIAGNOSTICS — dev counts, loan subtotals, excluding-dev row")
    print("=" * 100)
    for code in INVESTORS:
        for sub, key in (("operating", "operating_diag"),
                         ("loan", "loan_diag"),
                         ("financial", "financial_diag")):
            db = (b["investors"].get(code, {}) or {}).get(key) or {}
            da = (a["investors"].get(code, {}) or {}).get(key) or {}
            if db.get("dev") != da.get("dev"):
                print(f"  {code}/{sub}: dev {db.get('dev')} -> {da.get('dev')}")
        eb = (b["investors"].get(code, {}) or {}).get("excluding_dev") or {}
        ea = (a["investors"].get(code, {}) or {}).get("excluding_dev") or {}
        chk(f"{code}: excluding-dev population unchanged "
            f"({eb.get('excluded_count')} deals)",
            eb.get("excluded_vcodes") == ea.get("excluded_vcodes"),
            f"{eb.get('excluded_vcodes')} -> {ea.get('excluded_vcodes')}")
        sb = (b["investors"].get(code, {}) or {}).get("loan_subtotals") or {}
        sa = (a["investors"].get(code, {}) or {}).get("loan_subtotals") or {}
        # A subtotal may move only in the group that actually holds one of the
        # two deals, and only in fields that legitimately depend on the
        # classification. Anything else would mean the change leaked.
        loan_rows = (a["investors"][code].get("loan") or {})
        host = {g for g in set(sb) | set(sa)}
        changed = sorted(k for k in host if sb.get(k) != sa.get(k))
        ALLOWED = {"dev_count", "debt", "ltv", "ltv_n", "ytd_dscr",
                   "ytd_dscr_n", "debt_yield", "debt_yield_n", "debt_basis",
                   "ratio_basis", "deal_count"}
        leaked = {}
        for g in changed:
            fb_, fa_ = sb.get(g) or {}, sa.get(g) or {}
            bad = sorted(k for k in set(fb_) | set(fa_)
                         if fb_.get(k) != fa_.get(k) and k not in ALLOWED)
            if bad:
                leaked[g] = bad
        chk(f"{code}: loan subtotals move only in classification-dependent "
            f"fields (groups touched: {changed or 'none'})",
            not leaked and (not changed
                            or any(v in loan_rows
                                   for v in (PEGASUS, BELLEVILLE))),
            f"leaked={leaked}")

    # ── loan subtotals: Debt sums everything, ratios exclude dev ──────────
    #
    # The mechanism is `loan_subtotal`/`_debt_weighted`, which weight over the
    # rows carrying a RAW ratio value. A dev deal's raw ltv/ytd_dscr/
    # debt_yield are all None (the "Dev" literal lives only on the *_display
    # twins), so it is excluded arithmetically rather than by a dev test — the
    # PDF's footnote describes the effect, not the mechanism. Asserted on the
    # counts, so a dev deal leaking into a ratio denominator would show up
    # even if the resulting average happened to look plausible.
    print("\n" + "=" * 100)
    print("LOAN SUBTOTALS — Debt sums every deal; ratios exclude the dev deals")
    print("=" * 100)
    for code in INVESTORS:
        rows_by_vc = (a["investors"].get(code, {}) or {}).get("loan", {}) or {}
        subs = (a["investors"].get(code, {}) or {}).get("loan_subtotals") or {}
        if not subs:
            continue
        print(f"\n  {code}")
        print(f"    {'group':<34}{'deals':>6}{'dev':>5}{'debt':>15}"
              f"{'ltv_n':>7}{'dscr_n':>7}{'dy_n':>6}")
        for g, s in sorted(subs.items()):
            print(f"    {g[:32]:<34}{s.get('deal_count'):>6}"
                  f"{s.get('dev_count'):>5}{f(s.get('debt')):>15}"
                  f"{s.get('ltv_n'):>7}{s.get('ytd_dscr_n'):>7}"
                  f"{s.get('debt_yield_n'):>6}")
        # Every dev deal must be absent from all three ratio denominators.
        dev_vcs = {v for v, r in rows_by_vc.items() if r.get("is_dev")}
        leaks = {}
        for v in dev_vcs:
            r = rows_by_vc[v]
            carried = [k for k in ("ltv", "ytd_dscr", "debt_yield")
                       if r.get(k) is not None and r.get("debt")]
            if carried:
                leaks[v] = carried
        # Waters Creek legitimately carries a raw LTV (its exception), and the
        # published TGA22 60.4% only reproduces WITH it — so it is the one
        # allowed contributor, and only to LTV. Removed by exact match, so it
        # contributing to DSCR or Debt Yield would still be reported.
        if leaks.get("P0000078") == ["ltv"]:
            del leaks["P0000078"]
        chk(f"{code}: no dev deal contributes to a ratio subtotal "
            f"(bar Waters Creek's LTV exception)", not leaks, str(leaks))
        # Debt, by contrast, must include them.
        tot = (a["investors"].get(code, {}) or {}).get("loan_total") or {}
        all_debt = [r.get("debt") for r in rows_by_vc.values()
                    if r.get("debt") is not None]
        chk(f"{code}: portfolio-total Debt sums EVERY deal including dev "
            f"({len(all_debt)} deals)",
            tot.get("debt") is None
            or abs(float(tot["debt"]) - sum(all_debt)) < 1,
            f"total={tot.get('debt')} vs sum={sum(all_debt)}")
        chk(f"{code}: the debt-free deal contributes nothing to the Debt total",
            all(r.get("debt") in (0, 0.0, None)
                for v, r in rows_by_vc.items() if r.get("debt_free")))

    print("\n" + "=" * 100)
    bad = [c for c, ok in checks if not ok]
    print(f"{len(checks) - len(bad)}/{len(checks)} passed")
    for c in bad:
        print(f"  FAILED: {c}")
    return 1 if bad else 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "capture":
        capture(sys.argv[2])
    else:
        raise SystemExit(report())
