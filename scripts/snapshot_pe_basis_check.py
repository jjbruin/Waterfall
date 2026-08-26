"""Guardrail: PE-basis look-through, so PE-only dollars are scaled by a PE-only %.

THE BUG THIS LOCKS DOWN. ``cap_stack.pref_equity`` and ``cap_stack.committed_pe``
are built from non-OP investors only — ``one_pager.get_capitalization_stack``
routes any investor whose id starts with ``OP`` into ``partner_equity`` instead.
The snapshot used to scale those PE-only dollars by ``lookthrough_pct``, which
normalises its final hop against *every* owner of the deal entity, operating
partner included. Any deal whose OP carries a real ownership percentage therefore
had that stake subtracted twice.

Pegasus Life Storage was the only such deal at 26Q1/26Q2: OPPEGA holds 7.37% of
PEGASU, so funded came out at 32,334,654.75 x 0.83367 = 26,956,431.63 instead of
32,334,654.75 x 0.90 = 29,101,189.28 — low by 2,144,757.65.

WHY THE CHECKS ARE WRITTEN ON THE BASIS, NOT ON PEGASUS. ICPEGA also carries
OPPEGA at 7.37% and is not in the TGAM population today; naming Pegasus would let
the same defect arrive unnoticed with the next deal. So the invariants below are
universal (every deal, every quarter) and Pegasus is only a *named regression
value* in check 4 — the arithmetic checks would fail for any newly OP-diluted deal
whether or not anyone remembered to add it here.

Read-only against live: GET only, Bearer token from WF_TOKEN. The assembly runs
LOCALLY on the working tree, so a change here is visible before it deploys — the
REST endpoints run the deployed build and cannot show local work.

Usage
    WF_TOKEN=<jwt> python scripts/snapshot_pe_basis_check.py [investor]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd                                               # noqa: E402
import live_api as api                                            # noqa: E402
from flask_app.services.portfolio_snapshot_service import (        # noqa: E402
    resolve_investor_deals, _is_op,
)
from flask_app.services.portfolio_snapshot_summary import (        # noqa: E402
    assemble_summary,
)
from flask_app.services.portfolio_snapshot_financial import (      # noqa: E402
    assemble_financial,
)

INVESTOR = (sys.argv[1] if len(sys.argv) > 1 else "TGAM").upper()
QUARTERS = ("2026-Q1", "2026-Q2")
EPS = 0.01

#: 26Q1 page 1, transcribed. Self-Storage is the bucket the bug moved.
PDF_Q1_SELF_STORAGE = 34_172_689
#: 26Q2 has no published page. Expected = the Q1 figure plus Citizen Storage
#: Swartz Creek (2,300,000 x 0.90), the one Self-Storage deal added in Q2.
EXPECTED_Q2_SELF_STORAGE = 36_242_689
#: Pegasus regression values — deal-level pref_equity is unchanged by the fix;
#: only the percentage applied to it moves.
PEGASUS = {"vcode": "P0000066", "pref_equity": 32_334_654.75,
           "pct_deal_level": 0.83367, "pct_pe": 0.90,
           "funded_before": 26_956_431.63, "funded_after": 29_101_189.28}

checks: list[tuple[str, bool]] = []


def chk(label, ok):
    checks.append((label, bool(ok)))
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}")
    return ok


def _relationships(investor):
    """Narrow per-entity pulls, so OFFSET paging is never used — paging
    /api/data/tables/<t>/rows duplicates rows (see live-azure-readonly-access)."""
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
    return pd.DataFrame(rows).drop_duplicates()


def _owner_totals(rel, iid):
    """(total pct, non-OP pct) over the OPEN ownership rows of ``iid``."""
    cols = {c.lower(): c for c in rel.columns}
    c_inv, c_own = cols.get("investmentid"), cols.get("investorid")
    c_pct, c_end = cols.get("ownershippct"), cols.get("enddate")
    total = pe = 0.0
    for r in rel.itertuples(index=False):
        d = r._asdict()
        if str(d.get(c_inv) or "").strip().upper() != iid.upper():
            continue
        if c_end:
            end = str(d.get(c_end) or "").strip()
            if end not in ("", "None", "NaT", "nan", "null"):
                continue
        p = pd.to_numeric(d.get(c_pct), errors="coerce")
        p = 0.0 if pd.isna(p) else float(p)
        total += p
        if not _is_op(str(d.get(c_own) or "")):
            pe += p
    return total, pe


def main():
    ti = api.token_info()
    print(f"LIVE token={ti['username']} ({ti['hours_left']}h left)  "
          f"deployed build={api.get('/api/data/version').get('version')}")
    print(f"investor={INVESTOR}  quarters={', '.join(QUARTERS)}")
    print("assembly: LOCAL working tree; inputs: live (GET only)\n")

    inv = pd.DataFrame(api.get("/api/data/deals/all").get("deals") or [])
    rel = _relationships(INVESTOR)
    op_cache: dict[str, tuple] = {}

    cache: dict = {}

    def one_pager(vcode, q):
        if (vcode, q) not in cache:
            cache[(vcode, q)] = api.get(f"/api/financials/{vcode}/one-pager",
                                        params={"quarter": q})
        return cache[(vcode, q)]

    for q in QUARTERS:
        print("=" * 100)
        print(f"{q}")
        print("=" * 100)
        resolved = resolve_investor_deals(INVESTOR, q, rel, inv)
        local = assemble_summary(INVESTOR, q, resolved=resolved,
                                 one_pager_provider=one_pager,
                                 comment_loader=lambda i, _q: {},
                                 editable_loader=lambda i, _q: True)
        deployed = api.get("/api/portfolio-snapshot/summary",
                           params={"investor": INVESTOR, "quarter": q})

        L = {d["vcode"]: d for d in local["deals"]}
        aa = local["asset_allocation"]
        alloc = {d["vcode"]: d for b in aa["buckets"] for d in b["deals"]}

        # ---- 1. basis invariant: dollars = PE dollars x PE percentage ----
        print("\n1. every allocated deal: funded == pref_equity x pct_pe")
        bad = []
        for vc, a in alloc.items():
            row = L[vc]
            want = (row["funded_deal_level"] or 0) * row["lookthrough_pct_pe"]
            if abs((a["funded"] or 0) - want) > EPS:
                bad.append((vc, a["funded"], want))
        chk(f"{len(alloc)} deals scale by the PE basis", not bad)
        for vc, got, want in bad[:5]:
            print(f"        {vc}: funded {got:,.2f} != {want:,.2f}")

        # ---- 2. pct_pe is the deal-level pct re-normalised over non-OP owners
        #         pct = SUM acc_i*(p_i/T), pct_pe = SUM acc_i*(p_i/T_pe)
        #         => pct_pe == pct * T/T_pe, exactly, for every deal.
        print("\n2. pct_pe == pct x (all owners / non-OP owners), independently")
        bad2, diluted = [], []
        for vc in alloc:
            row, ent = L[vc], None
            for g in (resolved.get("groups") or {}).values():
                for e in g:
                    if e["vcode"] == vc:
                        ent = e
            if ent is None:
                continue
            iid = ent["iid"]
            if iid not in op_cache:
                op_cache[iid] = _owner_totals(rel, iid)
            total, pe = op_cache[iid]
            if pe <= 0 or total <= 0:
                continue
            want = row["lookthrough_pct"] * (total / pe)
            if abs(row["lookthrough_pct_pe"] - want) > 1e-9:
                bad2.append((vc, row["lookthrough_pct_pe"], want))
            if abs(total - pe) > 1e-9:
                diluted.append((vc, ent["name"], total - pe,
                                row["lookthrough_pct"],
                                row["lookthrough_pct_pe"]))
        chk("PE percentage matches an independent recomputation", not bad2)
        for vc, got, want in bad2[:5]:
            print(f"        {vc}: pct_pe {got!r} != {want!r}")

        print(f"\n   OP-diluted deals (OP holds a real %): {len(diluted)}")
        for vc, nm, opp, p, ppe in diluted:
            print(f"      {vc} {nm[:34]:<35} OP {opp:.2f}%  "
                  f"pct {p:.6f} -> pct_pe {ppe:.6f}")
        chk("every OP-diluted deal now uses a HIGHER percentage",
            all(ppe > p for _, _, _, p, ppe in diluted))

        # ---- 3. only OP-diluted deals may differ from the deployed build ----
        print("\n3. diff vs the deployed build")
        dep = {d["vcode"]: d
               for b in deployed["asset_allocation"]["buckets"]
               for d in b["deals"]}
        dil = {vc for vc, *_ in diluted}
        moved, wrongly_moved = [], []
        for vc, a in alloc.items():
            if vc not in dep:
                continue
            d0 = dep[vc]["funded"] or 0
            d1 = a["funded"] or 0
            if abs(d1 - d0) > EPS:
                moved.append((vc, L[vc]["name"], d0, d1))
                if vc not in dil:
                    wrongly_moved.append(vc)
        for vc, nm, d0, d1 in moved:
            print(f"      {vc} {nm[:32]:<33}{d0:>16,.2f} -> {d1:>16,.2f}"
                  f"   {d1 - d0:>+15,.2f}")
        chk(f"only OP-diluted deals moved ({len(moved)} moved, "
            f"{len(alloc) - len(moved)} unchanged)", not wrongly_moved)
        for vc in wrongly_moved:
            print(f"        {vc} moved but is not OP-diluted")

        # ---- 4. bucket totals ----
        print("\n4. bucket totals")
        b_local = {b["label"]: b for b in aa["buckets"]}
        b_dep = {b["label"]: b
                 for b in deployed["asset_allocation"]["buckets"]}
        for lab in sorted(set(b_local) | set(b_dep)):
            bl = b_local.get(lab, {}).get("funded") or 0
            bd = b_dep.get(lab, {}).get("funded") or 0
            print(f"      {lab:<14}{bd:>16,.2f} -> {bl:>16,.2f}"
                  f"   {bl - bd:>+15,.2f}")
        for lab in ("Multifamily", "Retail", "Office"):
            if lab in b_local and lab in b_dep:
                chk(f"{lab} funded unchanged",
                    abs((b_local[lab]["funded"] or 0)
                        - (b_dep[lab]["funded"] or 0)) < EPS)
        ss = b_local["Self-Storage"]["funded"]
        target = (PDF_Q1_SELF_STORAGE if q == "2026-Q1"
                  else EXPECTED_Q2_SELF_STORAGE)
        chk(f"Self-Storage {ss:,.2f} ties {target:,} (within $1)",
            abs(ss - target) < 1.0)
        chk("every bucket's funded shares still sum to 100%",
            all(abs(sum((b["funded"] or 0) for b in aa["buckets"])
                    - (aa["total_funded"] or 0)) < EPS for _ in (0,)))

        # ---- 5. Pegasus named regression ----
        if PEGASUS["vcode"] in L:
            print("\n5. Pegasus named regression values")
            row = L[PEGASUS["vcode"]]
            chk("pref_equity unchanged by the fix",
                abs((row["funded_deal_level"] or 0)
                    - PEGASUS["pref_equity"]) < EPS)
            chk(f"deal-level pct still {PEGASUS['pct_deal_level']}",
                abs(row["lookthrough_pct"]
                    - PEGASUS["pct_deal_level"]) < 1e-9)
            chk(f"PE pct is {PEGASUS['pct_pe']}",
                abs(row["lookthrough_pct_pe"] - PEGASUS["pct_pe"]) < 1e-7)
            chk(f"funded {PEGASUS['funded_before']:,.2f} -> "
                f"{PEGASUS['funded_after']:,.2f}",
                abs((alloc[PEGASUS["vcode"]]["funded"] or 0)
                    - PEGASUS["funded_after"]) < EPS)

        # ---- 6. City West still out; population intact ----
        print("\n6. exclusions and population")
        kept = [d for d in local["deals"] if d["kept_despite_sold"]]
        chk("a KEEP_DESPITE_SOLD deal is reported", len(kept) >= 1)
        chk("no KEEP_DESPITE_SOLD deal is in any bucket",
            all(k["vcode"] not in alloc for k in kept))
        chk("allocation population matches the deployed build",
            set(alloc) == set(dep))

        # ---- 7. Summary <-> Financial identity (the docstring's promise) ----
        print("\n7. Summary funded == Financial invested")
        fin = assemble_financial(
            INVESTOR, q, resolved=resolved, one_pager_provider=one_pager,
            committed_debt_provider=lambda vc: None,
            manual_loader=lambda i, _q: {}, footnote_loader=lambda i, _q: [])
        fin_rows = [r for g in fin["groups"].values() for r in g["deals"]]
        fin_inv = sum(r["invested"] for r in fin_rows
                      if r.get("invested") is not None
                      and not r.get("kept_despite_sold"))
        print(f"      summary total_funded {aa['total_funded']:>18,.2f}")
        print(f"      financial invested   {fin_inv:>18,.2f}")
        chk("the two subtabs agree",
            abs(fin_inv - (aa["total_funded"] or 0)) < 1.0)
        peg_fin = [r for r in fin_rows if r["vcode"] == PEGASUS["vcode"]]
        if peg_fin:
            chk("Financial's % of Pref for Pegasus is the PE basis",
                abs(peg_fin[0]["pct_of_pref"] - PEGASUS["pct_pe"]) < 1e-7)
        print()

    print("=" * 100)
    passed = sum(1 for _, ok in checks if ok)
    print(f"{passed}/{len(checks)} checks passed")
    for label, ok in checks:
        if not ok:
            print(f"  FAILED: {label}")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
