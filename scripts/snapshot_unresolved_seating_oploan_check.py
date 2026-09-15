"""Guardrail: unresolved-ownership seating on the OPERATING and LOAN subtabs.

The Financial subtab stopped segregating these deals in efb377d, which said in
terms: "the Summary, Operating and Loan subtabs are untouched and still
segregate these two. Extending the other three is a separate call." This is that
call for two of the three.

WHY IT MATTERED MORE HERE, NOT LESS. Both subtabs report `scaled: False` —
every operating metric is property-level and every loan metric is facility- or
property-level, so not one cell on either row depends on the ownership
percentage. A deal was being lifted out of its fund block over a number the page
never applies.

AND IT BROKE THE FOOTING. A flagged row was excluded from `groups`, and so from
every subtotal, while still counted in `total`. Measured live at 26Q2 before the
fix:

    Operating   deal_count   subtotals 34   total 35
    Loan        deal_count   subtotals 34   total 35
    Loan        debt         subtotals 1,322,930,063.62   total 1,368,324,063.62
                             -> Evergreen Plaza's $45,394,000 inside the grand
                                total, under no subtotal, with no row to explain

HOW IT COMPARES. Both payloads are built by the SAME local code on the SAME
data, differing only in whether the service published `derived_group` for the
unresolved deals — the switch the assembly falls back on for a deal it cannot
place. That isolates seating from everything else. Following the lesson recorded
in snapshot_unresolved_seating_check.py, this does NOT diff against the live
payload: a harness has no committed_debt_provider and no loans frame, so metric
values differ locally for reasons that have nothing to do with seating. Live is
used only for the population check, which a missing provider cannot disturb.

Read-only: GETs against live, no writes anywhere.

Usage
  WF_TOKEN=... .venv/Scripts/python.exe \
      scripts/snapshot_unresolved_seating_oploan_check.py [--investor TGAM] [--quarter 2026-Q2]
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

import live_api as api  # noqa: E402
from flask_app.services.portfolio_snapshot_service import (  # noqa: E402
    resolve_investor_deals,
)
from flask_app.services.portfolio_snapshot_operating import assemble_operating  # noqa: E402
from flask_app.services.portfolio_snapshot_loan import assemble_loan  # noqa: E402

#: Deals that are on main but not yet in the deployed image, so the local build
#: legitimately carries them and live does not. Empty this as they deploy —
#: a stale entry here would mask a deal that genuinely went missing.
#: `ad55705` (Sep 15 2026) added both to KEEP_DESPITE_SOLD.
KNOWN_UNDEPLOYED: dict = {
    "PCAMARI": "Camarillo Village — kept-despite-sold, merged ad55705",
    "POUTLOO": "Outlook Nine Mile — kept-despite-sold, merged ad55705",
}

CHECKS: list = []


def chk(label: str, ok, note: str = "") -> bool:
    CHECKS.append((bool(ok), label, note))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  — {note}" if note else ""))
    return bool(ok)


def relationships_for(investor: str) -> pd.DataFrame:
    """Walk the ownership graph outward, one narrow request per entity."""
    seen, frontier, rows = set(), [investor], []

    def fetch(col, val):
        d = api.get("/api/data/tables/relationships/rows",
                    params={"page": 1, "page_size": 500, f"filter__{col}": val})
        # The rows endpoint returns the WHOLE table on a filter miss — re-filter.
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
            if not child:
                continue
            rows.extend(fetch("InvestmentID", child))
            if child not in seen:
                frontier.append(child)
    return pd.DataFrame(rows).drop_duplicates()


def flatten(payload: dict) -> tuple:
    """{vcode: row}, {vcode: group} over groups + any segregated list."""
    flat, where = {}, {}
    for g, rows in (payload.get("groups") or {}).items():
        for r in (rows or []):
            flat[r["vcode"]] = r
            where[r["vcode"]] = g
    for r in (payload.get("ownership_flagged") or []):
        flat[r["vcode"]] = r
        where[r["vcode"]] = "__SEGREGATED__"
    return flat, where


def run(subtab: str, build, resolved: dict, live: dict) -> None:
    print(f"\n{'=' * 70}\n{subtab.upper()}\n{'=' * 70}")

    res_before = copy.deepcopy(resolved)
    for f in res_before.get("flagged") or []:
        f.pop("derived_group", None)

    before, after = build(res_before), build(resolved)
    b_flat, b_where = flatten(before)
    a_flat, a_where = flatten(after)

    unresolved = {f["vcode"]: f for f in (resolved.get("flagged") or [])}
    print(f"  unresolved deals: {[(v, f['name']) for v, f in unresolved.items()]}")
    for vc, f in unresolved.items():
        print(f"    {f['name']} ({vc}): derived_group={f.get('derived_group')!r}")
        print(f"      BEFORE block: {b_where.get(vc)!r}   AFTER block: {a_where.get(vc)!r}")

    print("\n  1. Seating")
    for vc, f in unresolved.items():
        g = f.get("derived_group")
        if not g:
            chk(f"{f['name']}: no derivable group -> stays segregated",
                a_where.get(vc) == "__SEGREGATED__")
            continue
        chk(f"{f['name']}: was segregated before", b_where.get(vc) == "__SEGREGATED__")
        chk(f"{f['name']}: now seated in {g}", a_where.get(vc) == g)
        chk(f"{f['name']}: carries the reason as a flag",
            any("ownership" in str(x).lower() for x in (a_flat[vc].get("flags") or [])),
            str(a_flat[vc].get("flags"))[:90])
        chk(f"{f['name']}: marked ownership_unresolved, NOT ownership_flagged",
            a_flat[vc].get("ownership_unresolved") is True
            and not a_flat[vc].get("ownership_flagged"))

    print("\n  2. The segregated list is empty now")
    chk("nothing left in ownership_flagged",
        (after.get("ownership_flagged") or []) == [],
        f"{len(after.get('ownership_flagged') or [])} remain")

    print("\n  3. Nothing else moved")
    chk("same population before and after", set(b_flat) == set(a_flat),
        f"{len(b_flat)} vs {len(a_flat)}")
    moved = {vc for vc in b_flat
             if b_where.get(vc) != a_where.get(vc) and vc not in unresolved}
    chk("no deal other than the unresolved ones changed block", not moved, str(moved))
    chk("every seated row's own metrics are unchanged by seating",
        all(b_flat[vc] == a_flat[vc] or
            {k: v for k, v in b_flat[vc].items() if k not in
             ("flags", "ownership_flagged", "ownership_unresolved")}
            == {k: v for k, v in a_flat[vc].items() if k not in
                ("flags", "ownership_flagged", "ownership_unresolved")}
            for vc in unresolved if vc in b_flat and vc in a_flat))

    print("\n  4. Footing: the subtotals now reach the total")
    subs_b = before.get("subtotals") or {}
    subs_a = after.get("subtotals") or {}
    tot_b, tot_a = before.get("total") or {}, after.get("total") or {}
    for field in ("deal_count", "debt"):
        if field not in tot_a:
            continue
        sb = sum((subs_b.get(g) or {}).get(field) or 0 for g in subs_b)
        sa = sum((subs_a.get(g) or {}).get(field) or 0 for g in subs_a)
        tb, ta = tot_b.get(field) or 0, tot_a.get(field) or 0
        print(f"    {field:<12} BEFORE subtotals={sb:>18,.2f} total={tb:>18,.2f}"
              f"  gap={tb - sb:>14,.2f}")
        print(f"    {field:<12} AFTER  subtotals={sa:>18,.2f} total={ta:>18,.2f}"
              f"  gap={ta - sa:>14,.2f}")
        chk(f"{field}: subtotals == total after seating", abs(ta - sa) < 0.01)
        chk(f"{field}: Portfolio Totals did NOT move", abs(ta - tb) < 0.01,
            "seating shifts a row between two populations already inside it")

    print("\n  5. Against live (population only — a harness has no providers)")
    live_flat, _ = flatten(live)
    extra = set(a_flat) - set(live_flat)
    missing = set(live_flat) - set(a_flat)
    # CONTAINMENT, not equality. Local runs main; live runs whatever image is
    # deployed, so main commits awaiting a deploy legitimately show up locally
    # and not live. Seating must never LOSE a deal live has — that is the real
    # assertion — and any extra is named so it cannot hide a mistake.
    chk("no deal that live shows is missing locally", not missing, str(missing))
    if extra:
        print(f"    local-only (on main, not yet deployed): "
              f"{sorted((a_flat[v].get('name'), v) for v in extra)}")
    chk("every local-only deal is explained by an undeployed main commit",
        all(v in KNOWN_UNDEPLOYED for v in extra),
        str(sorted(extra - set(KNOWN_UNDEPLOYED))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--investor", default="TGAM")
    ap.add_argument("--quarter", default="2026-Q2")
    a = ap.parse_args()
    inv_code, q = a.investor, a.quarter
    print(f"investor={inv_code} quarter={q}")

    inv = pd.DataFrame(api.get("/api/data/deals/all").get("deals") or [])
    rel = relationships_for(inv_code)
    resolved = resolve_investor_deals(inv_code, q, rel, inv)

    cache: dict = {}

    def provider(vc, quarter):
        if (vc, quarter) not in cache:
            cache[(vc, quarter)] = api.get(f"/api/financials/{vc}/one-pager",
                                           params={"quarter": quarter})
        return cache[(vc, quarter)]

    run("operating",
        lambda res: assemble_operating(inv_code, q, resolved=res,
                                       one_pager_provider=provider),
        resolved,
        api.get("/api/portfolio-snapshot/operating",
                {"investor": inv_code, "quarter": q}))

    # Empty loans/valuations: the A/B gives BOTH sides the same frames, so the
    # only difference remains the seating switch. Metric VALUES are not compared
    # against live for exactly this reason — see the module note.
    run("loan",
        lambda res: assemble_loan(inv_code, q, resolved=res,
                                  one_pager_provider=provider,
                                  loans=pd.DataFrame(),
                                  valuations=pd.DataFrame(), inv=inv),
        resolved,
        api.get("/api/portfolio-snapshot/loan",
                {"investor": inv_code, "quarter": q}))

    bad = [c for c in CHECKS if not c[0]]
    print(f"\n{len(CHECKS) - len(bad)}/{len(CHECKS)} checks pass")
    for _, label, note in bad:
        print(f"  FAILED: {label}" + (f"  — {note}" if note else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except api.TokenExpired as exc:
        print(f"token rejected: {exc}")
        sys.exit(2)
