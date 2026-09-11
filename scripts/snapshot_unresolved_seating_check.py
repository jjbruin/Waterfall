"""Guardrail: an ownership-unresolved deal is reported in its own fund block.

A deal whose look-through cannot be resolved used to be lifted out of its fund
and printed under a separate "Ownership % unavailable" heading at the foot of
the Financial table — so two ordinary PSC TGA 2024 LLC members, 45th & Main and
Town Fair Tire Portfolio, appeared to be outside the fund while their Zone A
dollars still counted toward Portfolio Totals. Adding up the TGA 2024 block by
hand gave a different answer from the page's own total.

HOW IT COMPARES. The two payloads are built by the SAME local code on the SAME
data, differing only in whether the service published a ``derived_group`` for
the unresolved deals — which is precisely the switch this change added, since
the assembly falls back to the old segregated list for a deal it cannot place.
That isolates the seating from everything else.

An earlier version of this script diffed against the LIVE payload instead, and
that is the wrong experiment: the live server injects a ``committed_debt_provider``
built from the loans frame, a harness has none, and every development deal's
Debt then reads n/a locally for reasons that have nothing to do with seating.
The live payload is still fetched, but only for checks a missing provider cannot
disturb — that the same deals are on the page, and that each seated row's own
figures are unchanged.

It asserts:

  * every unresolved deal moves from ``ownership_flagged`` into the group its
    own ownership chain derives;
  * its four Zone B cells are still withheld — this change moves a row, it does
    not invent a percentage;
  * its Zone A figures are unchanged, cell for cell;
  * Portfolio Totals do not move at all, because seating shifts a row between
    two populations that were both already inside the total;
  * the block subtotals move by exactly the seated rows and by nothing else.

Read-only: GETs against live, no writes anywhere.

Usage
  set WF_TOKEN=...
  .venv/Scripts/python.exe scripts/snapshot_unresolved_seating_check.py \
      [--investor TGAM] [--quarter 2026-Q2]
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import live_api as api                                        # noqa: E402
from flask_app.services.portfolio_snapshot_financial import (  # noqa: E402
    assemble_financial,
)
from flask_app.services.portfolio_snapshot_service import (    # noqa: E402
    resolve_investor_deals,
)

ZONE_A = ("debt", "total_pref", "ptr_equity", "total_cap")
ZONE_B = ("pct_of_pref", "invested", "total_commitment", "unfunded")
TOTAL_COLS = ("debt", "total_pref", "ptr_equity", "total_cap",
              "invested", "total_commitment", "unfunded")


def m(v):
    return "—" if v is None else f"{v:,.0f}"


def relationships_for(investor: str) -> pd.DataFrame:
    """The ownership edges reachable from one investor.

    Walked node by node with exact-match post-filtering: ``filter__`` on the
    rows endpoint is a case-insensitive CONTAINS, so ``TGAM`` alone would also
    drag in TGAM2 and TGAM3.
    """
    seen, frontier, rows = set(), [investor.upper()], []

    def fetch(col, val):
        d = api.get("/api/data/tables/relationships/rows",
                    params={"page": 1, "page_size": 500, f"filter__{col}": val})
        return [r for r in (d.get("rows") or [])
                if str(r.get(col) or "").strip().upper() == val.upper()]

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
    return pd.DataFrame(rows).drop_duplicates()


def flatten(payload):
    """{vcode: row}, and {vcode: group} for whatever is seated in a block."""
    flat, where = {}, {}
    for g, blk in (payload.get("groups") or {}).items():
        for r in blk.get("deals") or []:
            flat[r["vcode"]] = r
            where[r["vcode"]] = g
    for r in payload.get("ownership_flagged") or []:
        flat[r["vcode"]] = r
    return flat, where


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--investor", default="TGAM")
    ap.add_argument("--quarter", default="2026-Q2")
    args = ap.parse_args()
    inv_code, q = args.investor.upper(), args.quarter

    ti = api.token_info()
    print(f"live token={ti['username']} ({ti['hours_left']}h left)")
    print(f"investor={inv_code} quarter={q}\n")

    live = api.get("/api/portfolio-snapshot/financial",
                   {"investor": inv_code, "quarter": q})

    inv = pd.DataFrame(api.get("/api/data/deals/all").get("deals") or [])
    rel = relationships_for(inv_code)
    resolved = resolve_investor_deals(inv_code, q, rel, inv)

    cache: dict = {}

    def provider(vc, quarter):
        if (vc, quarter) not in cache:
            cache[(vc, quarter)] = api.get(f"/api/financials/{vc}/one-pager",
                                           params={"quarter": quarter})
        return cache[(vc, quarter)]

    def build(res):
        return assemble_financial(inv_code, q, resolved=res,
                                  one_pager_provider=provider,
                                  manual_loader=lambda i, qq: {},
                                  footnote_loader=lambda i, qq: [])

    # BEFORE = the same code with the seating switch off. Dropping
    # `derived_group` sends each unresolved deal down the documented fallback
    # path, which is the behaviour this change replaces.
    import copy
    res_before = copy.deepcopy(resolved)
    for f in res_before.get("flagged") or []:
        f.pop("derived_group", None)

    before = build(res_before)
    after = build(resolved)

    b_flat, b_where = flatten(before)
    a_flat, a_where = flatten(after)
    live_flat, live_where = flatten(live)
    checks: list = []

    def chk(label, cond):
        checks.append((label, bool(cond)))
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")

    # Which deals are we talking about? Identified by the payload, never named
    # in this script — a vcode list here would go stale the moment a chain is
    # repaired or another one breaks.
    unresolved = sorted(r["vcode"] for r in a_flat.values()
                        if r.get("ownership_unresolved"))
    derived = {f["vcode"]: f.get("derived_group")
               for f in (resolved.get("flagged") or [])}

    print("OWNERSHIP-UNRESOLVED DEALS")
    if not unresolved:
        print("  none this quarter — the seating checks below are vacuous\n")
    for vc in unresolved:
        r = a_flat[vc]
        was = "ownership_flagged (segregated)" if vc in {
            x["vcode"] for x in (before.get("ownership_flagged") or [])
        } else f"group {b_where.get(vc)}"
        live_was = "ownership_flagged (segregated)" if vc in {
            x["vcode"] for x in (live.get("ownership_flagged") or [])
        } else f"group {live_where.get(vc)}"
        print(f"  {vc}  {r['name']}")
        print(f"     on live: {live_was}")
        print(f"     before : {was}")
        print(f"     after  : group {a_where.get(vc)}   "
              f"(derived from first hop {derived.get(vc)!r})")
        print(f"     Zone A : " + "  ".join(
            f"{c}={m(r.get(c))}" for c in ZONE_A))
        print(f"     Zone B : " + "  ".join(
            f"{c}={r.get(c)!r}" for c in ZONE_B))
    print()

    print("SEATING")
    chk("nothing is left segregated after the change",
        (after.get("ownership_flagged") or []) == [])
    chk("every unresolved deal now sits in a group",
        all(vc in a_where for vc in unresolved))
    chk("each sits in the group its own chain derives",
        all(a_where.get(vc) == g for vc, g in derived.items() if g))
    chk("every unresolved deal still withholds all four Zone B cells",
        all(a_flat[vc].get(c) is None for vc in unresolved for c in ZONE_B))
    chk("blocks stay in name order",
        all([r["name"].lower() for r in blk["deals"]]
            == sorted(r["name"].lower() for r in blk["deals"])
            for blk in (after.get("groups") or {}).values()))

    print("\nAGAINST LIVE (only what a missing debt provider cannot disturb)")
    chk("live and local report the same population",
        set(a_flat) == set(live_flat))
    chk("live segregates exactly the deals this change seats",
        {x["vcode"] for x in (live.get("ownership_flagged") or [])}
        == set(unresolved))
    chk("each seated row's equity ties to live, cell for cell",
        all(abs((live_flat[vc].get(c) or 0) - (a_flat[vc].get(c) or 0)) < 1
            for vc in unresolved for c in ("total_pref", "ptr_equity")))
    chk("live withholds the same four cells this change still withholds",
        all(live_flat[vc].get(c) is None
            for vc in unresolved for c in ZONE_B))

    print("\nNOTHING ELSE MOVED")
    chk("the same set of deals is on the page",
        set(a_flat) == set(b_flat))
    moved = [(vc, c, b_flat[vc].get(c), a_flat[vc].get(c))
             for vc in sorted(set(a_flat) & set(b_flat))
             for c in ZONE_A + ZONE_B
             if (b_flat[vc].get(c) is None) != (a_flat[vc].get(c) is None)
             or (b_flat[vc].get(c) is not None
                 and abs((b_flat[vc].get(c) or 0)
                         - (a_flat[vc].get(c) or 0)) > 1)]
    chk("no deal-level figure changed on any row", not moved)
    for vc, c, bv, av in moved[:20]:
        print(f"       {vc} {c}: {m(bv)} -> {m(av)}")

    t_b, t_a = before.get("total") or {}, after.get("total") or {}
    chk("Portfolio Totals deal_count unchanged",
        t_b.get("deal_count") == t_a.get("deal_count"))
    tmoved = [(c, t_b.get(c), t_a.get(c)) for c in TOTAL_COLS
              if abs((t_b.get(c) or 0) - (t_a.get(c) or 0)) > 1]
    chk("no Portfolio Totals figure moved", not tmoved)
    for c, bv, av in tmoved:
        print(f"       total {c}: {m(bv)} -> {m(av)}")

    print("\nSUBTOTALS — the intended movement, block by block")
    seated_in = {}
    for vc in unresolved:
        seated_in.setdefault(a_where.get(vc), []).append(vc)
    ok_sub = True
    for g, blk in (after.get("groups") or {}).items():
        sb = ((before.get("groups") or {}).get(g) or {}).get("subtotal") or {}
        sa = blk.get("subtotal") or {}
        seats = seated_in.get(g, [])
        for c in ("total_pref", "ptr_equity", "total_cap"):
            delta = (sa.get(c) or 0) - (sb.get(c) or 0)
            expect = sum(a_flat[vc].get(c) or 0 for vc in seats)
            if abs(delta - expect) > 1:
                ok_sub = False
                print(f"  {g} {c}: moved {m(delta)}, expected {m(expect)}")
        if seats:
            print(f"  {g}: +{len(seats)} deal(s) {seats}")
            for c in ("total_pref", "ptr_equity", "total_cap"):
                print(f"     {c:<12} {m(sb.get(c))} -> {m(sa.get(c))}")
            print(f"     {'deal_count':<12} {sb.get('deal_count')} -> "
                  f"{sa.get('deal_count')}")
            print(f"     {'pct_of_pref':<12} "
                  f"{(sb.get('pct_of_pref') or 0) * 100:.2f}% -> "
                  f"{(sa.get('pct_of_pref') or 0) * 100:.2f}%   "
                  f"(Total Commitment / Total Pref — the denominator gains the "
                  f"seated deal's pref, the numerator has no commitment to add)")
    chk("every subtotal moved by exactly the deals seated into it", ok_sub)

    bad = [lbl for lbl, ok in checks if not ok]
    print(f"\n{'PASS' if not bad else 'FAIL'} — {len(checks) - len(bad)}/"
          f"{len(checks)} checks")
    for lbl in bad:
        print(f"   FAILED: {lbl}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
