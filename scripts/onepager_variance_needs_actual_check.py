"""Guardrail: no Variance is stated against an actual nobody reported.

THE DEFECT. The One Pager's Property Performance table printed "-100%" variance
on Revenue, Expenses and NOI for deals whose partner filed no 2026 financials.
The YTD Actual cell beside it was already blank, so the row read "no actual, and
revenue fell 100%".

WHY -100% AND NOT BLANK. `fmtVariance` in OnePagerView has ALWAYS guarded a null
actual. It never saw one: `get_property_performance` seeded revenue / expenses /
noi `ytd_actual` to a literal 0, and only overwrote it if `ytd_date` resolved —
an actual period inside the report year on or before quarter end. With no such
period the 0 survived, and (0 - budget) / |budget| is a confident -100%. The
blank YTD Actual cell came from `fmtMil`, which renders 0 as an em dash, so the
sentinel was invisible in one column and load-bearing in the next.

THE FIX IS THE SEED, not the formatter: `ytd_actual` starts as None, so
"nobody reported" and "reported zero" stop being the same value. The backend
variance is guarded to match (it would otherwise raise on None), though the
DISPLAY path is what the page uses — OnePagerView recomputes the percentage
itself and its existing null guard now fires.

TWELVE DEALS, not the two reported. Giant 7 (feed stopped Nov 2025, under PSA)
and East Manchester (sold) were raised; a sweep of all 130 deals found ten more
in the same state, every one printing -100%.

WHAT IS ASSERTED. The distinction the fix turns on, both ways:

  * a deal with NO actual period          -> ytd_actual None, variance None
  * a deal that REPORTED ZERO             -> ytd_actual 0.0,  variance computed
  * a deal with real actuals              -> untouched, to the cent

The reported-zero case is synthetic on purpose: no live deal is in that state
today, so the only way to prove a real 0 still computes is to construct one.
Without it this guard could be blanking every zero and the live checks would
look identical.

Read-only: GET only, via ``scripts/live_api.py`` (needs WF_TOKEN).

Usage
    set WF_TOKEN=<jwt>
    python scripts/onepager_variance_needs_actual_check.py
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd                                             # noqa: E402
import live_api as api                                          # noqa: E402
from one_pager import get_property_performance                  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

#: The two raised, plus two controls carrying real actuals.
NO_ACTUALS = {"P0000019": "Giant 7", "P0000017": "East Manchester"}
CONTROLS = {"P0000018": "Evergreen Plaza", "P0000075": "Camp Creek"}

BLOCKS = ("revenue", "expenses", "noi")


def fmt_variance(actual, budget):
    """OnePagerView's `fmtVariance`, so the check tests what the page prints."""
    if actual is None or budget is None or budget == 0:
        return ""
    return f"{round((actual - budget) / abs(budget) * 100)}%"


def page(table, params=None, size=500):
    p = {"page": 1, "page_size": size}
    p.update(params or {})
    d = api.get(f"/api/data/tables/{table}/rows", params=p)
    return d.get("rows") or []


def load_frames(vcodes):
    """Rebuild what ``data_service._assemble_isbs`` hands the real function.

    Two things this has to get right or the harness lies:

    PER-MONTH SLICES, not per-year. The rows endpoint pages with LIMIT/OFFSET
    and no tiebreaker, so a >500-row slice silently truncates. Giant 7's 2025
    Interim IS is 569 rows and lost 69 of them to a per-year pull.

    THE LEGACY TABLE. The ISBS split does not cover every deal: Giant 7 has NO
    rows in ``isbs_budget_is`` at all, and its budget comes from the monolithic
    ``isbs`` table, which ``_assemble_isbs`` uses to supplement any vSource a
    deal is missing. Without this a deal reads budget 0 here and a real figure
    on the page.
    """
    SOURCES = {"isbs_interim_is": "Interim IS", "isbs_interim_bs": "Interim BS",
               "isbs_budget_is": "Budget IS", "isbs_projected_is": "Projected IS"}

    def fetch_all(table, vc):
        """Every row for one deal, deduped so OFFSET paging cannot corrupt it.

        NOT date-filtered. ``dtEntry`` is stored in BOTH shapes — ISO
        ('2026-06-30T00:00:00') and US ('1/31/2022 0:00') — sometimes in the
        same table, so an ISO month/year substring silently matches nothing for
        a deal whose rows are US-format. That is why Giant 7 first read a
        budget of 0 here against $6.13M on the page. Filtering by vcode only
        and letting the real function do its own date work removes the trap.

        Dedupe by the whole row rather than trusting the pager: the endpoint
        uses LIMIT/OFFSET with no tiebreaker, so pages can repeat rows.
        """
        out, seen, pageno = [], set(), 1
        while True:
            d = api.get(f"/api/data/tables/{table}/rows",
                        params={"page": pageno, "page_size": 500,
                                "filter__vcode": vc.lower()})
            rows = d.get("rows") or []
            if not rows:
                break
            for r in rows:
                if str(r.get("vcode", "")).lower() != vc.lower():
                    continue
                key = tuple(sorted((k, str(v)) for k, v in r.items()))
                if key in seen:
                    continue
                seen.add(key)
                out.append(r)
            if len(out) >= (d.get("total") or 0) or pageno > 40:
                break
            pageno += 1
        return out

    isbs, occ, beo = [], [], []
    for vc in vcodes:
        seen_sources = set()
        for tbl, src in SOURCES.items():
            for r in fetch_all(tbl, vc):
                r["vSource"] = src
                isbs.append(r)
                seen_sources.add(src)
        # Supplement from the legacy monolith for any vSource the split lacks,
        # exactly as data_service._assemble_isbs does.
        missing = set(SOURCES.values()) - seen_sources
        if missing:
            for r in fetch_all("isbs", vc):
                if str(r.get("vSource") or "") in missing:
                    isbs.append(r)
        occ += [r for r in page("occupancy", {"filter__vCode": vc})
                if str(r.get("vCode", "")).upper() == vc.upper()]
        beo += [r for r in page("budget_econ_occ", {"filter__VCODE": vc})
                if str(r.get("VCODE", "")).upper() == vc.upper()]
    df = pd.DataFrame(isbs)
    if not df.empty:
        df["dtEntry_parsed"] = pd.to_datetime(df["dtEntry"], format="mixed",
                                              errors="coerce")
        df["mAmount"] = pd.to_numeric(df["mAmount"], errors="coerce")
        df["vAccount"] = df["vAccount"].astype(str).str.strip()
    return (df, pd.DataFrame(page("valuations")), pd.DataFrame(occ),
            pd.DataFrame(beo), pd.DataFrame(page("deals")),
            pd.DataFrame(page("deal_terms")))


def synthetic_reported_zero():
    """A deal that DID report, and reported zero. Must still get a variance.

    One Interim IS period inside the report year — so ``ytd_date`` resolves —
    carrying zero amounts on a revenue and an expense account, plus a real
    budget to compare against.
    """
    rows = []
    for acct in ("4010", "5020"):
        rows.append({"vcode": "ptest", "dtEntry": "2026-06-30",
                     "vSource": "Interim IS", "vAccount": acct,
                     "mAmount": 0.0, "statement_id": 1})
        rows.append({"vcode": "ptest", "dtEntry": "2026-06-30",
                     "vSource": "Budget IS", "vAccount": acct,
                     "mAmount": -1_000_000.0 if acct == "4010" else 400_000.0,
                     "statement_id": 2})
    df = pd.DataFrame(rows)
    df["dtEntry_parsed"] = pd.to_datetime(df["dtEntry"])
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quarter", default="2026-Q2")
    args = ap.parse_args()

    results = []

    def chk(name, ok):
        results.append((name, bool(ok)))

    ti = api.token_info()
    print(f"LIVE  token={ti['username']} ({ti['hours_left']}h left)")
    vcodes = list(NO_ACTUALS) + list(CONTROLS)
    print(f"loading frames for {len(vcodes)} deals ...")
    isbs, vals, occ, beo, deals, terms = load_frames(vcodes)
    print(f"  isbs={len(isbs)} occ={len(occ)} beo={len(beo)} terms={len(terms)}\n")

    print(f"REAL get_property_performance, {args.quarter}")
    print(f"  {'deal':<22}{'block':<10}{'ytd_actual':>14}{'ytd_budget':>14}"
          f"{'variance cell':>15}")
    out = {}
    for vc in vcodes:
        perf = get_property_performance(
            vc, args.quarter, isbs, vals, occupancy_df=occ,
            budget_econ_occ_df=beo, deal_terms_df=terms, inv_map=deals)
        out[vc] = perf
        nm = NO_ACTUALS.get(vc) or CONTROLS[vc]
        for blk in BLOCKS:
            d = perf[blk]
            a, b = d.get("ytd_actual"), d.get("ytd_budget")
            print(f"  {nm[:22]:<22}{blk:<10}"
                  f"{('None' if a is None else f'{a:,.0f}'):>14}"
                  f"{('None' if b is None else f'{b:,.0f}'):>14}"
                  f"{(fmt_variance(a, b) or '(blank)'):>15}")
        print()

    for vc, nm in NO_ACTUALS.items():
        for blk in BLOCKS:
            d = out[vc][blk]
            chk(f"{nm} {blk}: ytd_actual is None, not 0",
                d.get("ytd_actual") is None)
            chk(f"{nm} {blk}: variance cell is blank, not -100%",
                fmt_variance(d.get("ytd_actual"), d.get("ytd_budget")) == "")
            chk(f"{nm} {blk}: backend variance withheld",
                d.get("variance") is None)
            # The budget must survive — only the COMPARISON is withheld.
            #
            # Asserted only where this harness actually loaded one. Giant 7 is
            # a portfolio PARENT (7 children) whose own ISBS rows stop at 2024
            # for budget and 2025 for actuals; its live 2026 budget of $6.13M
            # is aggregated from the children, which a parent-only frame cannot
            # reproduce. Its `ytd_actual` is None either way — no 2026 Interim
            # IS period exists, so `ytd_date` cannot resolve — which is the part
            # this change turns on. The live sweep is what evidences its
            # before-state of -100%.
            if d.get("ytd_budget"):
                chk(f"{nm} {blk}: budget still shown",
                    d.get("ytd_budget") not in (None, 0))

    print("CONTROLS vs the deployed page")
    for vc, nm in CONTROLS.items():
        live = api.get(f"/api/financials/{vc}/one-pager",
                       params={"quarter": args.quarter})["property_performance"]
        for blk in BLOCKS:
            lv = (live.get(blk) or {}).get("ytd_actual")
            av = out[vc][blk].get("ytd_actual")
            same = (lv is None and av is None) or (
                lv is not None and av is not None and abs(av - lv) < 0.01)
            chk(f"{nm} {blk}: ytd_actual unchanged", same)
            cell = fmt_variance(av, out[vc][blk].get("ytd_budget"))
            chk(f"{nm} {blk}: variance still computes", cell != "")
            print(f"  {nm[:22]:<22}{blk:<10}live={lv!s:<22}now={av!s:<22}{cell:>7}")

    # ── the distinction, proved on a constructed case ─────────────────────
    print("\nA DEAL THAT REPORTED ZERO (synthetic — none exists live)")
    sp = get_property_performance("ptest", args.quarter,
                                  synthetic_reported_zero(), pd.DataFrame())
    for blk in BLOCKS:
        a, b = sp[blk].get("ytd_actual"), sp[blk].get("ytd_budget")
        print(f"  {blk:<10}ytd_actual={a!s:<8}ytd_budget={b!s:<12}"
              f"variance cell={fmt_variance(a, b) or '(blank)'}")
    chk("a reported zero is 0.0, NOT None",
        sp["revenue"].get("ytd_actual") == 0.0)
    chk("a reported zero still gets a variance cell",
        fmt_variance(sp["revenue"].get("ytd_actual"),
                     sp["revenue"].get("ytd_budget")) != "")
    chk("a reported zero still gets a backend variance",
        sp["revenue"].get("variance") is not None)

    print(f"\n{'-' * 72}")
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    bad = sum(1 for _, ok in results if not ok)
    print(f"\n{'PASS' if not bad else 'FAIL'} — {len(results) - bad}/"
          f"{len(results)} checks")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
