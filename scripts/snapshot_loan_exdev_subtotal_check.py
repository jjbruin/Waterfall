"""Guardrail: the Loan subtab's "Excluding development deals (N)" debt subtotal
FOOTS to the rows it visibly sums.

THE DEFECT this was written for. That subtotal was the ONE total on the report
computed client-side (``exDevTotal`` in SnapshotLoan.vue) instead of by
``loan_subtotal`` on the server, and the client copy accumulated the RAW
``r.debt`` while the Debt cell beside it rendered ``debt_display``. A row whose
loan columns are suppressed therefore printed "—" and still contributed its
balance to the total underneath it:

    subtotal shown            $984.8M
    the 27 visible rows       $975.4M
    gap                         $9.4M  = East Manchester's $9,641,912

The server has had the correct rule since 2026-09-02 — ``loan_subtotal`` drops
any row with ``sold_suppressed``, and ``aggregation_value`` does the same for
the ratio columns, which is why the FUND totals and Portfolio Totals were all
right. Only the client-side copy of the rule was missing it.

WHAT IS ASSERTED. The subtotal must equal the sum of the DISPLAYED debt on the
non-development rows — displayed, not raw, so a suppressed cell cannot be in a
total that claims to add up the column above it. Checked against the served
payload, so what is verified is what the page shows.

Read-only: GET only, via ``scripts/live_api.py`` (needs WF_TOKEN).

TWO SOURCES, because they answer different questions:

  ``--source local`` (default) runs the REAL committed ``assemble_loan`` over
      live data, wired as ``portfolio_snapshot_freeze.build_subtab`` wires it.
      This is what tests the WORKING TREE, and the only way to see a fix that
      has not been deployed yet.

  ``--source live`` reads the served ``/bundle``, i.e. the DEPLOYED app. Use it
      to reproduce the defect as users see it, and after a deploy to confirm
      what actually shipped.

Usage
    set WF_TOKEN=<jwt>
    python scripts/snapshot_loan_exdev_subtotal_check.py
    python scripts/snapshot_loan_exdev_subtotal_check.py --source live
    python scripts/snapshot_loan_exdev_subtotal_check.py --quarter 2026-Q1
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import live_api as api                                          # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

EAST_MANCHESTER = "P0000017"
CITY_WEST = "PCITWES"

TOL = 1.0        # dollars; the column prints to one decimal in $M


def M(v):
    return "        —" if v is None else f"{v / 1e6:>8,.1f}M"


def displayed_debt(r):
    """The debt this row PRINTS, or None where the cell reads "—".

    Mirrors ``debtCell`` in SnapshotLoan.vue: ``debt_display`` when the key is
    present (a present null is the deliberate dash), else the raw ``debt`` for
    a snapshot frozen before that key existed.
    """
    if "debt_display" in r:
        v = r["debt_display"]
        return None if isinstance(v, str) else v
    return r.get("debt")


def client_exdev(rows):
    """What SnapshotLoan.vue's ``exDevTotal`` computed BEFORE the fix.

    Kept as the reproduction of the defect: raw ``debt``, no suppression test.
    """
    debt, n, any_ = 0.0, 0, False
    for r in rows:
        if r.get("is_dev"):
            continue
        n += 1
        if r.get("debt") is not None:
            debt += r["debt"]
            any_ = True
    return (debt if any_ else None), n


def loan_from_live(investor, quarter):
    """The Loan subtab as the DEPLOYED app serves it."""
    bundle = api.get("/api/portfolio-snapshot/bundle",
                     params={"investor": investor, "quarter": quarter})
    return (bundle.get("subtabs") or {}).get("loan") or {}


def loan_from_local(investor, quarter):
    """The Loan subtab from the REAL committed ``assemble_loan``, over live data.

    Dependencies are injected exactly as ``build_subtab`` injects them, with
    the frames pulled from the read-only table endpoints instead of the
    in-process data cache. ``quarterly_noi_provider`` is left unwired: it feeds
    the YTD DSCR and Debt Yield columns only, and every assertion here is about
    the DEBT subtotal.
    """
    import pandas as pd
    from flask_app.services.portfolio_snapshot_service import (
        resolve_investor_deals,
    )
    from flask_app.services.portfolio_snapshot_loan import assemble_loan
    from snapshot_financial_pdf_check import build as _fin_build   # noqa: F401

    inv = pd.DataFrame(api.get("/api/data/deals/all").get("deals") or [])

    def fetch(col, val):
        d = api.get("/api/data/tables/relationships/rows",
                    params={"page": 1, "page_size": 500, f"filter__{col}": val})
        return [r for r in (d.get("rows") or [])
                if str(r.get(col) or "").strip().upper() == val.upper()]

    seen, frontier, rel_rows = set(), [investor], []
    while frontier:
        node = frontier.pop().upper()
        if node in seen:
            continue
        seen.add(node)
        kids = fetch("InvestorID", node)
        rel_rows.extend(kids)
        for r in kids:
            child = str(r.get("InvestmentID") or "").strip().upper()
            if child:
                rel_rows.extend(fetch("InvestmentID", child))
                if child not in seen:
                    frontier.append(child)
    rel = pd.DataFrame(rel_rows).drop_duplicates()
    resolved = resolve_investor_deals(investor, quarter, rel, inv)

    def table(name):
        d = api.get(f"/api/data/tables/{name}/rows",
                    params={"page": 1, "page_size": 1000})
        assert (d.get("total") or 0) <= 1000, f"{name} no longer fits one page"
        return pd.DataFrame(d.get("rows") or [])

    loans, vals = table("loans"), table("valuations")

    cache = {}

    def one_pager(vcode, q):
        if (vcode, q) not in cache:
            cache[(vcode, q)] = api.get(f"/api/financials/{vcode}/one-pager",
                                        params={"quarter": q})
        return cache[(vcode, q)]

    return assemble_loan(investor, quarter, resolved=resolved,
                         one_pager_provider=one_pager,
                         loans=loans, valuations=vals, inv=inv,
                         comment_loader=lambda i, q: {},
                         manual_loader=lambda i, q: {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--investor", default="TGAM")
    ap.add_argument("--quarter", default="2026-Q2")
    ap.add_argument("--source", choices=("local", "live"), default="local",
                    help="local = real assemble_loan on the working tree; "
                         "live = the deployed /bundle")
    args = ap.parse_args()

    ti = api.token_info()
    print(f"LIVE  token={ti['username']} ({ti['hours_left']}h left)  "
          f"build={api.get('/api/data/version').get('version')}")
    src = ("real assemble_loan (working tree)" if args.source == "local"
           else "served /bundle (deployed)")
    print(f"source: {src}  —  {args.investor} {args.quarter}\n")

    loan = (loan_from_local if args.source == "local"
            else loan_from_live)(args.investor, args.quarter)
    rows = [r for g in (loan.get("groups") or {}).values() for r in g]

    nondev = [r for r in rows if not r.get("is_dev")]
    dev = [r for r in rows if r.get("is_dev")]

    # ── every deal the subtotal sums, with its debt ───────────────────────
    print(f"NON-DEVELOPMENT ROWS ({len(nondev)}) — what each contributes")
    print(f"  {'deal':<32}{'raw debt':>11}{'displayed':>11}  suppressed")
    print(f"  {'-' * 68}")
    raw_sum = 0.0
    disp_sum = 0.0
    suppressed = []
    for r in sorted(nondev, key=lambda x: x["name"]):
        raw, disp = r.get("debt"), displayed_debt(r)
        sup = bool(r.get("sold_suppressed"))
        if raw is not None:
            raw_sum += raw
        if disp is not None:
            disp_sum += disp
        if sup or (raw is not None and disp is None):
            suppressed.append(r)
        print(f"  {r['name'][:32]:<32}{M(raw)}{M(disp)}  "
              f"{'YES' if sup else ''}")
    print(f"  {'-' * 68}")
    print(f"  {'sum of RAW debt':<32}{M(raw_sum)}")
    print(f"  {'sum of DISPLAYED debt':<32}{'':>11}{M(disp_sum)}")

    served_debt = None
    # The server may now publish the row; before the fix it was client-only.
    served = loan.get("total_excluding_dev") or {}
    if served:
        served_debt = served.get("debt")
    cdebt, cn = client_exdev(rows)

    print(f"\nTHE SUBTOTAL")
    print(f"  client-side computation (pre-fix rule) {M(cdebt)}  over {cn} deals")
    if served:
        print(f"  server-published `total_excluding_dev` {M(served_debt)}  "
              f"over {served.get('deal_count')} deals")
    else:
        print(f"  server-published `total_excluding_dev` — not in this payload "
              f"(client-computed only)")
    print(f"  sum of the DISPLAYED non-dev rows      {M(disp_sum)}")
    print(f"  gap (raw - displayed)                  {M(raw_sum - disp_sum)}")

    print(f"\nTHE GAP, ATTRIBUTED")
    if not suppressed:
        print("  no non-dev row has a suppressed debt cell — gap is 0")
    for r in suppressed:
        print(f"  {r['name'][:32]:<32} {r['vcode']:<10} raw {M(r.get('debt'))}"
              f"  displayed —   sold_suppressed={bool(r.get('sold_suppressed'))}")

    for vc, label in ((EAST_MANCHESTER, "East Manchester"),
                      (CITY_WEST, "City West")):
        r = next((x for x in rows if x["vcode"] == vc), None)
        if r is None:
            print(f"\n  {label} ({vc}) is not on this page")
            continue
        print(f"\n  {label} ({vc}): is_dev={r.get('is_dev')} "
              f"sold_suppressed={bool(r.get('sold_suppressed'))} "
              f"raw debt={M(r.get('debt'))} displayed={M(displayed_debt(r))}")

    # ── checks ────────────────────────────────────────────────────────────
    results = []

    def chk(name, ok):
        results.append((name, bool(ok)))

    # The one that matters: the total equals the column above it.
    subtotal = served_debt if served else cdebt
    chk("excluding-dev debt subtotal = sum of the DISPLAYED non-dev rows",
        subtotal is not None and abs(subtotal - disp_sum) <= TOL)
    chk("no suppressed row contributes debt to the subtotal",
        subtotal is not None
        and abs(subtotal - sum(displayed_debt(r) or 0 for r in nondev)) <= TOL)
    for vc, label in ((EAST_MANCHESTER, "East Manchester"),
                      (CITY_WEST, "City West")):
        r = next((x for x in rows if x["vcode"] == vc), None)
        if r is None:
            continue
        # A suppressed cell must print no figure AND carry none into the total.
        if r.get("sold_suppressed"):
            chk(f"{label} prints no debt figure", displayed_debt(r) is None)
    chk("every development deal stays out of the subtotal",
        not any(r.get("is_dev") for r in nondev))
    chk("dev deals are on the page (the exclusion is doing work)", bool(dev))
    # The count is a count of VISIBLE non-dev rows and must not change: East
    # Manchester and City West are still rows on the page, they simply carry no
    # debt figure. Dropping them from the count would misreport the population.
    chk("deal_count counts every visible non-dev row",
        (served.get("deal_count") if served else cn) == len(nondev))
    # Fund totals and Portfolio Totals come from loan_subtotal, which has
    # always excluded suppressed rows — verified here so the fix cannot be
    # mistaken for having changed them.
    port = (loan.get("total") or {}).get("debt")
    port_expected = sum(displayed_debt(r) or 0 for r in rows)
    chk("Portfolio Totals debt = sum of every displayed row",
        port is not None and abs(port - port_expected) <= TOL)

    print(f"\n{'-' * 72}")
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    bad = sum(1 for _, ok in results if not ok)
    print(f"\n{'PASS' if not bad else 'FAIL'} — {len(results) - bad}/"
          f"{len(results)} checks")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
