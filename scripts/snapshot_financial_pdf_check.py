"""Guardrail: Financial subtab vs the TIAA reference PDF, page 2.

Runs the REAL committed ``assemble_financial`` (and the real
``resolve_investor_deals`` behind it) against live data, injecting the live REST
endpoints as its dependencies, then checks the assembled page against the
page-2 transcription in ``_PDF``.

Two kinds of check, kept apart on purpose:

  STRUCTURE — grouping, membership, labels, the excluding-development row, the
  n/a cells. All of this is what this change controls, so every one of these
  must pass.

  VALUES — the dollar figures. These are NOT all expected to match: the PDF is a
  31-Mar-2026 vintage and live data has moved, and two columns cannot match at
  all today (see the notes on UNFUNDED_STRUCTURAL and MANUAL_UNENTERED). Values
  are reported with their deltas and scored separately, so a structural pass is
  never hidden by a data difference and vice versa.

Read-only: GET only, via ``scripts/live_api.py`` (needs WF_TOKEN).

Usage
    set WF_TOKEN=<jwt>
    python scripts/snapshot_financial_pdf_check.py
    python scripts/snapshot_financial_pdf_check.py --quarter 2026-Q2   # structure only
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
from flask_app.services.portfolio_snapshot_service import (      # noqa: E402
    resolve_investor_deals, group_total_label, PORTFOLIO_TOTAL_LABEL,
    GROUP_OVERRIDES, KEEP_DESPITE_SOLD,
)
from flask_app.services.portfolio_snapshot_financial import (    # noqa: E402
    assemble_financial, EXCLUDING_DEV_VCODES, EXCLUDING_DEV_COLUMNS,
    PDF_NA_CELLS, NA_LABEL, PENDING, _SUM_COLS,
)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ── PDF page 2, transcribed ($M as printed; "n/a" is the PDF's literal) ───
# vcode -> (name, debt, total_pref, ptr_equity, total_cap, pct_of_pref,
#           invested, unfunded, total_commitment)
# None = the PDF prints a bare accounting dash.
_PDF = {
    "P0000019": ("Giant 7",            95.1,  21.0, 15.0, 131.1, 0.57, 11.5, 0.5, 11.9),
    "P0000017": ("East Manchester",      9.6,   3.6,  2.4,  15.6, 0.76,  2.7, None, 2.7),
    "P0000021": ("JB Fair Park",       48.98,  14.3,  3.9,  67.1, 0.85,  6.1, 6.1, 12.2),
    "P0000030": ("Nottingham Village",  38.9,   9.1,  6.3,  54.3, 0.41,  3.8, None, 3.8),
    "P0000018": ("Evergreen Plaza",     45.4,  16.4,  8.1,  69.8, 0.73, 12.0, None, 12.0),
    "PCITWES":  ("City West",          NA_LABEL, 5.9, 14.2, 20.2, 0.84,  5.0, None, 5.0),
    "P0000065": ("Ascent on Steamboat", 32.5,  21.7,  7.6,  61.8, 0.69, 15.0, None, 15.0),
    "P0000066": ("Pegasus Life Storage", None,  8.1,  2.6,  10.7, 0.91,  7.4, None, 7.4),
}

_PDF_GROUP_TOTALS = {
    # group key -> (label, debt, total_pref, ptr_equity, total_cap,
    #               pct_of_pref, invested, unfunded, total_commitment)
    "Individual Investments": ("Total Individual Investments",
                               270.5, 100.1, 60.1, 430.6, 0.70, 63.4, 6.5, 69.9),
    "TGA22": ("Total PSC TGA 2022 LLC", 268.4, 133.0, 57.5, 458.8, 0.82, 97.5, 12.0, 109.5),
    "TGA23": ("Total PSC TGA 2023 LLC", 366.5, 169.5, 93.3, 629.3, 0.67, 112.7, None, 112.7),
    "TGA24": ("Total PSC TGA 2024 LLC", 279.2, 119.3, 59.1, 457.5, 0.77, 89.6, 2.2, 91.8),
    "TGA25": ("Total PSC TGA 2025 LLC", 201.5,  73.6, 38.4, 282.6, 0.90, 46.0, 20.2, 66.2),
}

_PDF_PORTFOLIO = (1386.0, 595.3, 308.4, 2258.9, 0.68, 404.2, 40.9, 445.1)
_PDF_EXDEV_COMMITMENT = 299.3
_PDF_INDIVIDUAL_MEMBERS = set(_PDF.keys())

COLS = ("debt", "total_pref", "ptr_equity", "total_cap", "pct_of_pref",
        "invested", "unfunded", "total_commitment")

#: Un-funded is structurally 0 for every deal while COMMITMENT_BASIS ==
#: "funded": that basis defines Total Commitment as pct x Total Pref, which is
#: the Invested formula, so Un-funded = Commitment - Invested is 0 by
#: construction. The PDF has 40.9 at portfolio level. Not a bug in this change
#: and not fixable by it — switching to "committed_pe" is a separate decision
#: with its own reconciliation (see the module docstring's committed_gap note).
UNFUNDED_STRUCTURAL = True

#: Net ROE and ITD Distributions are manual-entry, formula TBD
#: (get_net_roe / get_itd), and nothing is entered on live. The PDF's 4.4% /
#: 41.3 therefore cannot be reproduced from data. Copying the published figures
#: into the app would present them as computed, so they are left pending.
MANUAL_UNENTERED = True


def build(investor, quarter):
    """Run the real resolve + assemble over live data."""
    inv = pd.DataFrame(api.get("/api/data/deals/all").get("deals") or [])

    def fetch(col, val):
        d = api.get("/api/data/tables/relationships/rows",
                    params={"page": 1, "page_size": 500, f"filter__{col}": val})
        return [r for r in (d.get("rows") or [])
                if str(r.get(col) or "").strip().upper() == val.upper()]

    seen, frontier, rows = set(), [investor], []
    while frontier:
        node = frontier.pop().upper()
        if node in seen:
            continue
        seen.add(node)
        kids = fetch("InvestorID", node)
        rows.extend(kids)
        for r in kids:
            child = str(r.get("InvestmentID") or "").strip().upper()
            if child:
                rows.extend(fetch("InvestmentID", child))
                if child not in seen:
                    frontier.append(child)
    rel = pd.DataFrame(rows).drop_duplicates()
    resolved = resolve_investor_deals(investor, quarter, rel, inv)

    cache = {}

    def one_pager(vcode, q):
        if (vcode, q) not in cache:
            cache[(vcode, q)] = api.get(f"/api/financials/{vcode}/one-pager",
                                        params={"quarter": q})
        return cache[(vcode, q)]

    out = assemble_financial(investor, quarter, resolved=resolved,
                             one_pager_provider=one_pager,
                             manual_loader=lambda i, q: {},
                             footnote_loader=lambda i, q: [])
    return resolved, out


def M(v):
    return "—" if v is None else f"{v / 1e6:,.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--investor", default="TGAM")
    ap.add_argument("--quarter", default="2026-Q1")
    args = ap.parse_args()

    ti = api.token_info()
    print(f"LIVE  token={ti['username']} ({ti['hours_left']}h left)  "
          f"build={api.get('/api/data/version').get('version')}  "
          f"actuals_through={api.get('/api/data/config').get('actuals_through')}")
    print(f"real assemble_financial for {args.investor} {args.quarter}\n")

    resolved, out = build(args.investor, args.quarter)
    flat = {r["vcode"]: r for b in out["groups"].values() for r in b["deals"]}
    for r in out["ownership_flagged"]:
        flat[r["vcode"]] = r

    struct, values = [], []
    pdf_scope = (args.investor.upper() == "TGAM" and args.quarter == "2026-Q1")

    # ── STRUCTURE ────────────────────────────────────────────────────────
    print("=" * 104)
    print("STRUCTURE")
    print("=" * 104)

    gnames = list(out["groups"])
    struct.append(("exactly the PDF's 5 groups, Individual first",
                   gnames == ["Individual Investments", "TGA22", "TGA23",
                              "TGA24", "TGA25"]))
    struct.append(("no TGAM2 group", "TGAM2" not in gnames))
    for g, (label, *_) in _PDF_GROUP_TOTALS.items():
        blk = out["groups"].get(g) or {}
        struct.append((f"{g} total labelled '{label}'",
                       (blk.get("subtotal") or {}).get("label") == label))
    struct.append((f"portfolio total labelled '{PORTFOLIO_TOTAL_LABEL}'",
                   out["total"]["label"] == PORTFOLIO_TOTAL_LABEL))

    if pdf_scope:
        indiv = {r["vcode"] for r in out["groups"]["Individual Investments"]["deals"]}
        struct.append(("Individual Investments holds exactly the PDF's 8 deals",
                       indiv == _PDF_INDIVIDUAL_MEMBERS))
        print(f"  Individual Investments: {len(indiv)} deals")
        for vc in sorted(indiv):
            mark = []
            if vc in GROUP_OVERRIDES:
                mark.append("regrouped")
            if vc in KEEP_DESPITE_SOLD:
                mark.append("kept-despite-sold")
            if vc in PDF_NA_CELLS:
                mark.append("n/a cells")
            print(f"      {vc:<9}{flat[vc]['name'][:34]:<36}"
                  f"{'  '.join(mark)}")
        missing = _PDF_INDIVIDUAL_MEMBERS - indiv
        extra = indiv - _PDF_INDIVIDUAL_MEMBERS
        if missing:
            print(f"      MISSING vs PDF: {sorted(missing)}")
        if extra:
            print(f"      EXTRA vs PDF:   {sorted(extra)}")

    seen = {}
    for g, b in out["groups"].items():
        for r in b["deals"]:
            seen.setdefault(r["vcode"], []).append(g)
    dups = {k: v for k, v in seen.items() if len(v) > 1}
    struct.append(("no deal counted in two groups", not dups))
    print(f"  duplicate vcodes across groups: {dups or 'none'}")

    cw = flat.get("PCITWES") or {}
    struct.append(("City West present", bool(cw)))
    struct.append(("City West Debt reads n/a", cw.get("debt_display") == NA_LABEL))
    struct.append(("City West Net ROE reads n/a",
                   cw.get("net_roe_display") == NA_LABEL))
    struct.append(("City West raw Debt untouched (not a string)",
                   not isinstance(cw.get("debt"), str)))
    print(f"  City West: debt_display={cw.get('debt_display')!r} "
          f"raw debt={cw.get('debt')!r} net_roe_display={cw.get('net_roe_display')!r}")

    ex = out.get("total_excluding_dev") or {}
    struct.append(("excluding-development row present", bool(ex)))
    struct.append(("excluding-dev removes exactly EXCLUDING_DEV_VCODES",
                   set(ex.get("excluded_vcodes") or [])
                   == (EXCLUDING_DEV_VCODES & set(flat))))
    struct.append(("excluding-dev populates only the PDF's 3 columns",
                   all(ex.get(c) is None for c in _SUM_COLS
                       if c not in EXCLUDING_DEV_COLUMNS)))
    struct.append(("excluding-dev never fabricates ITD / Net ROE",
                   ex.get("itd") is None and ex.get("net_roe") is None
                   and ex.get("itd_display") == PENDING))
    struct.append(("excluding-dev commitment below the portfolio total",
                   (ex.get("total_commitment") or 0)
                   < (out["total"]["total_commitment"] or 0)))
    print(f"  excluding-dev: removed {ex.get('excluded_count')} deals, "
          f"{ex.get('deal_count')} remain")
    for n in (ex.get("excluded_names") or []):
        print(f"      - {n}")

    struct.append(("is_dev populated (Lifecycle proxy, not the empty raw field)",
                   sum(1 for r in flat.values() if r["is_dev"]) > 0))
    struct.append(("no raw metric carries a display literal",
                   all(not isinstance(r.get(c), str)
                       for r in flat.values() for c in COLS)))
    print(f"  is_dev count: {sum(1 for r in flat.values() if r['is_dev'])}  "
          f"(Operating/Loan see the same set)")

    for label, ok in struct:
        print(f"    [{'PASS' if ok else 'FAIL'}] {label}")

    # ── VALUES ───────────────────────────────────────────────────────────
    if pdf_scope:
        print("\n" + "=" * 104)
        print("VALUES vs PDF page 2   (tol: $0.35M, 2pp)")
        print("=" * 104)
        hdr = (f"{'row':<30}{'column':<18}{'live':>10}{'PDF':>10}"
               f"{'delta':>10}   verdict")
        print(hdr)
        print("-" * 104)

        def cmp_row(name, row, want, skip_unfunded=True):
            for col, exp in zip(COLS, want):
                got = row.get(col)
                if exp == NA_LABEL:
                    ok = row.get(f"{col}_display") == NA_LABEL
                    print(f"{name[:29]:<30}{col:<18}"
                          f"{str(row.get(f'{col}_display')):>10}{'n/a':>10}"
                          f"{'':>10}   {'ok' if ok else 'MISMATCH'}")
                    values.append((f"{name} {col}", ok, False))
                    continue
                if col == "unfunded" and UNFUNDED_STRUCTURAL:
                    print(f"{name[:29]:<30}{col:<18}{M(got):>10}"
                          f"{('—' if exp is None else exp):>10}{'':>10}"
                          f"   SKIP (commitment basis)")
                    continue
                if exp is None:
                    ok = got is None or abs(got) < 50_000
                    print(f"{name[:29]:<30}{col:<18}{M(got):>10}{'—':>10}"
                          f"{'':>10}   {'ok' if ok else 'MISMATCH'}")
                    values.append((f"{name} {col}", ok, True))
                    continue
                if got is None:
                    print(f"{name[:29]:<30}{col:<18}{'—':>10}{exp:>10}"
                          f"{'':>10}   MISSING")
                    values.append((f"{name} {col}", False, True))
                    continue
                if col == "pct_of_pref":
                    d, ok = got * 100 - exp * 100, abs(got - exp) <= 0.02
                    print(f"{name[:29]:<30}{col:<18}{got*100:>9.0f}%"
                          f"{exp*100:>9.0f}%{d:>+10.1f}   "
                          f"{'ok' if ok else 'MISMATCH'}")
                else:
                    d, ok = got / 1e6 - exp, abs(got / 1e6 - exp) <= 0.35
                    print(f"{name[:29]:<30}{col:<18}{M(got):>10}{exp:>10}"
                          f"{d:>+10.1f}   {'ok' if ok else 'MISMATCH'}")
                values.append((f"{name} {col}", ok, True))

        for vc, (nm, *want) in _PDF.items():
            if vc in flat:
                cmp_row(nm, flat[vc], want)
            else:
                print(f"{nm[:29]:<30}{'NOT IN SET':<18}")
                values.append((f"{nm} present", False, True))
        print()
        for g, (label, *want) in _PDF_GROUP_TOTALS.items():
            blk = out["groups"].get(g)
            if blk:
                cmp_row(label, blk["subtotal"], want)
                print()
        cmp_row(PORTFOLIO_TOTAL_LABEL, out["total"], _PDF_PORTFOLIO)
        exc = (ex.get("total_commitment") or 0) / 1e6
        ok = abs(exc - _PDF_EXDEV_COMMITMENT) <= 0.35
        print(f"{'Excluding Development':<30}{'total_commitment':<18}"
              f"{exc:>10,.1f}{_PDF_EXDEV_COMMITMENT:>10}"
              f"{exc - _PDF_EXDEV_COMMITMENT:>+10.1f}   "
              f"{'ok' if ok else 'MISMATCH'}")
        values.append(("Excluding-dev total_commitment", ok, True))

        if MANUAL_UNENTERED:
            print(f"\n  SKIP  Portfolio ITD (PDF 41.3) and Net ROE (PDF 4.4%), "
                  f"and the excluding-dev 41.3 / 5.8%:")
            print(f"        both columns are manual entry with no formula and "
                  f"nothing entered on live.")

    # ── SUMMARY ──────────────────────────────────────────────────────────
    print("\n" + "=" * 104)
    s_fail = [l for l, ok in struct if not ok]
    print(f"STRUCTURE  {len(struct) - len(s_fail)}/{len(struct)} passed")
    for l in s_fail:
        print(f"    [FAIL] {l}")
    if pdf_scope:
        v_scored = [(l, ok) for l, ok, scored in values if scored]
        v_fail = [l for l, ok in v_scored if not ok]
        print(f"VALUES     {len(v_scored) - len(v_fail)}/{len(v_scored)} within "
              f"tolerance  ({len(v_fail)} data differences — PDF is a 31-Mar "
              f"vintage; see the header notes)")
        for l in v_fail[:20]:
            print(f"    [diff] {l}")
        if len(v_fail) > 20:
            print(f"    ... and {len(v_fail) - 20} more")
    print(f"\n  diagnostics: {out['diagnostics']}")
    # Only STRUCTURE gates the exit code. Value drift is data, not this change.
    return 1 if s_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
