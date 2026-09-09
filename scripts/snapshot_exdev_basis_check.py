"""Guardrail: pages 6 and 8 use ONE definition of "development deal".

THE DEFECT this was written for. The Financial subtab's "Excluding Development
Deals" row filtered a hardcoded vcode list (``EXCLUDING_DEV_VCODES``, eight
deals transcribed from the reference PDF) while the Loan subtab's
"Excluding development deals (N)" row filters each row's own ``is_dev``. The
list omitted JB FAIR PARK, which ``is_dev_deal`` calls development, so the two
pages of one report disagreed about a single deal:

    page 6   hardcoded list   28 deals   371.4M commitment / 47.394M ITD
    page 8   row `is_dev`     27 deals   (JB Fair Park removed)

The per-row flags never disagreed — both subtabs resolve them through
``resolve_strategy`` + ``is_dev_deal``. Only the two exclusion POPULATIONS did.

WHAT IS ASSERTED, and why it is stated this way. Every check is a PROPERTY of
the page rather than a comparison against a transcribed list of names:

  * the two subtabs' per-row ``is_dev`` flags agree on every deal;
  * page 6's excluded set is exactly the development deals — all of them, and
    nothing else;
  * page 6 keeps no development deal, and page 8 keeps no development deal;
  * the two pages remove the SAME deals and report the SAME kept count.

A list-based assertion is what let the original defect through: it can only
prove the code matches the list, never that the list matches reality. These
checks keep holding as the portfolio changes.

JB Fair Park is additionally named, because a regression on the reported deal
must be called out by name rather than buried in a count.

Read-only: GET only, via ``scripts/live_api.py`` (needs WF_TOKEN). Page 6 is
built by the REAL committed ``assemble_financial`` with the REAL stored manual
values, so the ITD figures are the ones the page shows — a stubbed manual
loader would report ITD None and prove nothing. Page 8's flags come from the
live ``/bundle``, i.e. the Loan subtab exactly as the app serves it.

Usage
    set WF_TOKEN=<jwt>
    python scripts/snapshot_exdev_basis_check.py
    python scripts/snapshot_exdev_basis_check.py --quarter 2026-Q1
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
    resolve_investor_deals,
)
from flask_app.services.portfolio_snapshot_financial import (    # noqa: E402
    assemble_financial,
)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

JB = "P0000021"          # JB Fair Park — the deal the two pages disagreed on
PEGASUS = "P0000066"     # the hardcoded list's other named keep; must stay non-dev


def live_manual(investor, quarter):
    """{vcode: {field: value}} from the live values table — the real entries.

    Mirrors ``portfolio_snapshot_financial._load_manual``'s shape so the real
    assembly reads live ITD / Net ROE exactly as the deployed app does.
    """
    page = api.get("/api/portfolio-snapshot/elements",
                   params={"investor": investor, "quarter": quarter})
    out = {}
    for r in (page.get("values") or []):
        out.setdefault(r.get("deal_vcode"), {})[r.get("field")] = r.get("value")
    return out


def build_financial(investor, quarter):
    """Page 6 from the real committed assembly, with real manual values."""
    from snapshot_financial_pdf_check import build as _build
    import snapshot_financial_pdf_check as H

    # `build` stubs the manual loader (it is a PDF-fidelity harness and the PDF
    # has no live entries). Patch the real one in for the duration.
    real = H.assemble_financial

    def with_manual(inv_code, q, **kw):
        kw["manual_loader"] = live_manual
        return real(inv_code, q, **kw)

    H.assemble_financial = with_manual
    try:
        return _build(investor, quarter)[1]
    finally:
        H.assemble_financial = real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--investor", default="TGAM")
    ap.add_argument("--quarter", default="2026-Q2")
    args = ap.parse_args()

    ti = api.token_info()
    print(f"LIVE  token={ti['username']} ({ti['hours_left']}h left)  "
          f"build={api.get('/api/data/version').get('version')}")
    print(f"page 6 = real assemble_financial (+ real manual values); "
          f"page 8 = live /bundle\n")

    fin = build_financial(args.investor, args.quarter)
    frows = {r["vcode"]: r for g in (fin.get("groups") or {}).values()
             for r in (g.get("deals") or [])}

    bundle = api.get("/api/portfolio-snapshot/bundle",
                     params={"investor": args.investor, "quarter": args.quarter})
    loan = (bundle.get("subtabs") or {}).get("loan") or {}
    lrows = {r["vcode"]: r for g in (loan.get("groups") or {}).values()
             for r in g}

    ex = fin.get("total_excluding_dev") or {}
    p6_removed = set(ex.get("excluded_vcodes") or [])
    p6_kept = set(frows) - p6_removed
    # Page 8's row is `rows.filter(r => !r.is_dev)` in SnapshotLoan.vue, so its
    # population is derived the same way here rather than re-implemented.
    p8_removed = {vc for vc, r in lrows.items() if r.get("is_dev")}
    p8_kept = set(lrows) - p8_removed
    dev = {vc for vc, r in frows.items() if r.get("is_dev")}

    print(f"POPULATION   page 6 rows={len(frows)}   page 8 rows={len(lrows)}")
    print(f"  development per `is_dev`      : {len(dev)}")
    print(f"  page 6 excluding-dev removes  : {len(p6_removed)}  "
          f"-> keeps {len(p6_kept)}")
    print(f"  page 8 excluding-dev removes  : {len(p8_removed)}  "
          f"-> keeps {len(p8_kept)}")

    print(f"\nPAGE 6 EXCLUDING-DEV TOTALS")
    tc, itd = ex.get("total_commitment"), ex.get("itd")
    print(f"  deal_count       {ex.get('deal_count')}")
    print(f"  Total Commitment {'—' if tc is None else f'${tc / 1e6:,.1f}M'}")
    print(f"  ITD              {'—' if itd is None else f'${itd:,.3f}M'}")

    jb = frows.get(JB)
    if jb:
        print(f"\nJB FAIR PARK ({JB})")
        print(f"  is_dev on page 6 = {jb.get('is_dev')}   "
              f"is_dev on page 8 = {(lrows.get(JB) or {}).get('is_dev')}")
        print(f"  removed from page 6 excluding-dev = {JB in p6_removed}")
        print(f"  removed from page 8 excluding-dev = {JB in p8_removed}")
        print(f"  its commitment ${(jb.get('total_commitment') or 0) / 1e6:,.2f}M"
              f"   its ITD {jb.get('itd')}")

    # ── checks ────────────────────────────────────────────────────────────
    results = []

    def chk(name, ok):
        results.append((name, bool(ok)))

    disagree = {vc for vc in set(frows) & set(lrows)
                if bool(frows[vc].get("is_dev")) != bool(lrows[vc].get("is_dev"))}
    chk("pages 6 and 8 agree on is_dev for every deal", not disagree)
    chk("page 6 removes exactly the development deals", p6_removed == dev)
    chk("page 6 keeps no development deal",
        not any(frows[vc].get("is_dev") for vc in p6_kept))
    chk("page 8 keeps no development deal",
        not any(lrows[vc].get("is_dev") for vc in p8_kept))
    chk("both pages remove the SAME deals", p6_removed == p8_removed)
    chk("both pages report the same kept count", len(p6_kept) == len(p8_kept))
    chk("JB Fair Park is classified development", bool(dev and JB in dev))
    chk("JB Fair Park is removed from page 6", JB in p6_removed)
    chk("JB Fair Park is removed from page 8", JB in p8_removed)
    # The hardcoded list's other named keep. It must NOT become development as
    # a side effect: it was reclassified at source when "new construction" came
    # out of DEV_STRATEGIES, and the PDF prints a dash for its debt.
    chk("Pegasus Life Storage is still NOT development",
        PEGASUS not in frows or not frows[PEGASUS].get("is_dev"))
    chk("excluding-dev commitment is below the portfolio total",
        (tc or 0) < ((fin.get("total") or {}).get("total_commitment") or 0))
    # ITD is a sum over the kept deals, so it must equal that sum exactly —
    # this is what catches a population change that misses the ITD leg.
    itd_expected = [frows[vc].get("itd") for vc in p6_kept
                    if frows[vc].get("itd") is not None]
    chk("excluding-dev ITD = sum over the kept deals",
        (itd is None and not itd_expected)
        or (itd is not None and abs(itd - sum(itd_expected)) < 1e-6))
    commit_expected = [frows[vc].get("total_commitment") for vc in p6_kept
                       if frows[vc].get("total_commitment") is not None]
    chk("excluding-dev commitment = sum over the kept deals",
        (tc is None and not commit_expected)
        or (tc is not None and abs(tc - sum(commit_expected)) < 1))

    print(f"\n{'-' * 72}")
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if disagree:
        print(f"\n  is_dev disagreements: "
              + ", ".join(f"{vc} ({frows[vc]['name']})" for vc in sorted(disagree)))
    if p6_removed != p8_removed:
        print(f"\n  only page 6 removes: "
              f"{sorted(p6_removed - p8_removed)}")
        print(f"  only page 8 removes: "
              f"{sorted(p8_removed - p6_removed)}")

    bad = sum(1 for _, ok in results if not ok)
    print(f"\n{'PASS' if not bad else 'FAIL'} — {len(results) - bad}/"
          f"{len(results)} checks")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
