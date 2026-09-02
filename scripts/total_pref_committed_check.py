"""Guardrail: Total Pref on the committed basis, and the two funded fallbacks.

Read-only against live; runs the REAL committed assemble_financial. Verifies
every requirement of the change and every thing it must NOT move.

Modes
    capture-before   run from a worktree at origin/main, write before.json
    check            run on the working tree; compare, verify, verdict

    WF_TOKEN=<jwt> python scripts/total_pref_committed_check.py capture-before
    WF_TOKEN=<jwt> python scripts/total_pref_committed_check.py check

`check` alone still works: it produces "before" by calling the same committed
module with commitment_basis="funded", which reproduces the pre-switch
total_pref / total_cap exactly. The worktree capture exists to PROVE that claim
against shipped main rather than assert it.
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd                                               # noqa: E402
import live_api as api                                            # noqa: E402
from snapshot_pdf_26q1_reconcile import PDF_FIN                    # noqa: E402

INVESTOR, QUARTER = "TGAM", "2026-Q1"
TOL_M = 0.051
BEFORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                      "outputs", "total_pref_committed_before.json")

PDF_TOTALS = dict(total_pref=595.3, total_cap=2258.9, invested=404.2,
                  total_commitment=445.1, ptr_equity=308.4, debt=1386.0)
#: PDF page 2 printed subtotal "% of Pref", by group.
PDF_SUB_PCT = {"Total Individual Investments": 70, "Total PSC TGA 2022 LLC": 82,
               "Total PSC TGA 2023 LLC": 67, "Total PSC TGA 2024 LLC": 77,
               "Total PSC TGA 2025 LLC": 90}
#: The three commitment-VALUE disputes waiting on Alay. Expected to differ.
DISPUTED = {"P0000030": "Nottingham", "P0000109": "Burton",
            "P0000021": "JB Fair Park"}
FALLBACK_VCODES = {"P0000017", "PCITWES"}

pd.set_option("display.width", 240)


def build(basis=None):
    """Assemble Financial + Summary from live inputs on the current tree."""
    from flask_app.services.portfolio_snapshot_service import (
        resolve_investor_deals)
    from flask_app.services.portfolio_snapshot_financial import (
        assemble_financial, COMMITMENT_BASIS)
    from flask_app.services.portfolio_snapshot_summary import assemble_summary
    from flask_app.services.portfolio_snapshot_debt import (
        committed_facility, deal_loan_rows)

    inv = pd.DataFrame(api.get("/api/data/deals/all").get("deals") or [])

    def fetch(col, val):
        d = api.get("/api/data/tables/relationships/rows",
                    params={"page": 1, "page_size": 500, f"filter__{col}": val})
        return [r for r in (d.get("rows") or [])
                if str(r.get(col) or "").strip().upper() == val.upper()]

    seen, frontier, rel_rows = set(), [INVESTOR], []
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

    d = api.get("/api/data/tables/loans/rows",
                params={"page": 1, "page_size": 500})
    loans = pd.DataFrame(d.get("rows") or [])

    cache = {}

    def provider(vc, q):
        if (vc, q) not in cache:
            cache[(vc, q)] = api.get(f"/api/financials/{vc}/one-pager",
                                     params={"quarter": q})
        return cache[(vc, q)]

    resolved = resolve_investor_deals(INVESTOR, QUARTER, rel, inv)
    kw = dict(resolved=resolved, one_pager_provider=provider,
              committed_debt_provider=lambda vc: committed_facility(
                  deal_loan_rows(loans, vc)),
              manual_loader=lambda i, q: {}, footnote_loader=lambda i, q: [])
    fin = (assemble_financial(INVESTOR, QUARTER, commitment_basis=basis, **kw)
           if basis else assemble_financial(INVESTOR, QUARTER, **kw))
    summ = assemble_summary(INVESTOR, QUARTER, resolved=resolved,
                            one_pager_provider=provider,
                            comment_loader=lambda i, q: {},
                            editable_loader=lambda i, q: True)
    return fin, summ, COMMITMENT_BASIS, cache


def flatten(fin):
    out = {}
    for b in fin["groups"].values():
        for r in b["deals"]:
            out[r["vcode"]] = r
    for r in fin.get("ownership_flagged") or []:
        out[r["vcode"]] = r
    return out


FIELDS = ("total_pref", "total_cap", "invested", "ptr_equity", "debt",
          "total_commitment", "unfunded", "pct_of_pref", "funded_pref",
          "committed_pref", "pref_basis",
          # Per-ROW, not per-vcode: KEEP_DESPITE_SOLD membership says a deal
          # WOULD be kept if the sold gate fired, and the gate only fires in
          # the quarters after the sale. East Manchester is in that set and is
          # kept at 26Q2, but at 26Q1 it was still held and is an ordinary row
          # on both pages. The like-for-like population check below has to read
          # the flag the assembly actually set for THIS quarter.
          "kept_despite_sold")


def snap(fin, summ):
    flat = flatten(fin)
    return {
        "deals": {vc: {f: r.get(f) for f in FIELDS} for vc, r in flat.items()},
        "total": {f: fin["total"].get(f) for f in FIELDS},
        "subtotals": {g: {"label": b["subtotal"]["label"],
                          **{f: b["subtotal"].get(f) for f in FIELDS}}
                      for g, b in fin["groups"].items()},
        "summary_total_committed": (summ.get("asset_allocation") or {})
        .get("total_committed"),
        "summary_total_funded": (summ.get("asset_allocation") or {})
        .get("total_funded"),
    }


def main():
    mode = (sys.argv[1] if len(sys.argv) > 1 else "check").lower()
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    ti = api.token_info()
    print(f"LIVE token={ti['username']} ({ti['hours_left']}h)  "
          f"build={api.get('/api/data/version').get('version')}  "
          f"tree HEAD={head}  mode={mode}\n")

    if mode == "capture-before":
        fin, summ, basis, _ = build()
        os.makedirs(os.path.dirname(BEFORE), exist_ok=True)
        with open(BEFORE, "w") as fh:
            json.dump({"head": head, "basis": basis, **snap(fin, summ)}, fh)
        print(f"wrote {os.path.normpath(BEFORE)}  (basis={basis!r})")
        return 0

    fin_a, summ_a, basis, cache = build()
    after = snap(fin_a, summ_a)
    fin_b, summ_b, _, _ = build(basis="funded")
    before_local = snap(fin_b, summ_b)

    print(f"shipped COMMITMENT_BASIS = {basis!r}")
    if basis != "committed_pe":
        print("  FAIL: the switch is not in place")
        return 1

    checks = []

    def chk(label, cond, detail=""):
        checks.append((label, bool(cond)))
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}"
              + (f"   {detail}" if detail else ""))

    # ── worktree cross-check ─────────────────────────────────────────────
    print("=" * 118)
    print("0. Does commitment_basis='funded' reproduce shipped main?")
    print("=" * 118)
    if os.path.exists(BEFORE):
        with open(BEFORE) as fh:
            wt = json.load(fh)
        moved = []
        for vc, b in wt["deals"].items():
            l = before_local["deals"].get(vc) or {}
            for f in ("total_pref", "total_cap", "invested", "ptr_equity",
                      "debt", "total_commitment", "unfunded"):
                bv, lv = b.get(f), l.get(f)
                if bv is None and lv is None:
                    continue
                if bv is None or lv is None or abs(bv - lv) > 1:
                    moved.append((vc, f, bv, lv))
        chk(f"worktree main ({wt['head']}, basis={wt['basis']!r}) == local "
            f"basis='funded' on all 7 columns",
            not moved, f"{len(moved)} divergences" if moved else "")
        for vc, f, bv, lv in moved[:10]:
            print(f"        {vc} {f}: main={bv} local={lv}")
    else:
        print("  SKIPPED — no before.json; run capture-before in a worktree at "
              "origin/main to prove the 'before' baseline")

    # ── the fallbacks ────────────────────────────────────────────────────
    print("\n" + "=" * 118)
    print("1. The two funded fallbacks")
    print("=" * 118)
    for vc, want in (("P0000017", 3_600_000.0), ("PCITWES", 5_925_000.0)):
        r = after["deals"].get(vc) or {}
        cap = (cache[(vc, QUARTER)].get("cap_stack") or {})
        print(f"  {vc:<9}{'':2}total_pref={r.get('total_pref'):>14,.2f}  "
              f"funded_pref={r.get('funded_pref'):>14,.2f}  "
              f"committed_pref={r.get('committed_pref'):>14,.2f}  "
              f"basis={r.get('pref_basis')!r}")
        print(f"           cap_stack.committed_pe={cap.get('committed_pe'):>14,.2f}"
              f"  committed_pe_basis={cap.get('committed_pe_basis')!r}")
        chk(f"{vc} Total Pref = {want:,.0f} (funded), not 0",
            r.get("total_pref") is not None
            and abs(r["total_pref"] - want) < 1)
        chk(f"{vc} pref_basis says it fell back",
            r.get("pref_basis") == "funded (no commitment row)")
    fb = {vc for vc, r in after["deals"].items()
          if r.get("pref_basis") == "funded (no commitment row)"}
    chk("the fallback fires on EXACTLY those two deals", fb == FALLBACK_VCODES,
        f"fired on {sorted(fb)}")

    # ── must-not-move columns ────────────────────────────────────────────
    print("\n" + "=" * 118)
    print("2. Columns that must NOT move")
    print("=" * 118)
    for f in ("invested", "ptr_equity", "debt", "pct_of_pref"):
        moved = [(vc, before_local["deals"][vc].get(f), r.get(f))
                 for vc, r in after["deals"].items()
                 if not (before_local["deals"].get(vc, {}).get(f) is None
                         and r.get(f) is None)
                 and (before_local["deals"].get(vc, {}).get(f) is None
                      or r.get(f) is None
                      or abs(before_local["deals"][vc][f] - r[f]) > 1e-9)]
        chk(f"{f} unchanged on all {len(after['deals'])} deals", not moved,
            f"moved: {moved[:4]}" if moved else "")
    chk("Ptr Equity is still cap_stack.partner_equity (funded), deal by deal",
        all(abs(r["ptr_equity"]
                - (cache[(vc, QUARTER)].get("cap_stack") or {})
                .get("partner_equity", 0)) < 1
            for vc, r in after["deals"].items()
            if r.get("ptr_equity") is not None))
    chk("subtotal '% of Pref' unchanged (anchored to funded_pref)",
        all(abs((before_local["subtotals"][g].get("pct_of_pref") or 0)
                - (s.get("pct_of_pref") or 0)) < 1e-9
            for g, s in after["subtotals"].items()))

    # ── identities ───────────────────────────────────────────────────────
    print("\n" + "=" * 118)
    print("3. Identities")
    print("=" * 118)
    chk("Total Pref == committed_pref wherever a commitment row exists",
        all(abs(r["total_pref"] - r["committed_pref"]) < 1
            for r in after["deals"].values()
            if r.get("committed_pref") and r.get("total_pref") is not None))
    chk("Invested == % of Pref x funded_pref",
        all(abs(r["invested"] - r["pct_of_pref"] * r["funded_pref"]) < 1e-6
            for r in after["deals"].values()
            if None not in (r.get("invested"), r.get("pct_of_pref"),
                            r.get("funded_pref"))))
    chk("Total Commitment == % of Pref x Total Pref",
        all(abs(r["total_commitment"] - r["pct_of_pref"] * r["total_pref"]) < 1e-6
            for r in after["deals"].values()
            if None not in (r.get("total_commitment"), r.get("pct_of_pref"),
                            r.get("total_pref"))))
    chk("Un-funded == Total Commitment - Invested",
        all(abs(r["unfunded"] - (r["total_commitment"] - r["invested"])) < 1e-6
            for r in after["deals"].values()
            if None not in (r.get("unfunded"), r.get("total_commitment"),
                            r.get("invested"))))

    # ── PDF ties ─────────────────────────────────────────────────────────
    print("\n" + "=" * 118)
    print("4. Total Pref vs the baseline PDF, before -> after ($M)")
    print("=" * 118)
    print(f"  {'vcode':<9}{'deal':<30}{'before':>9}{'after':>9}{'PDF':>7}"
          f"{'B':>3}{'A':>3}   note")
    tb = ta = 0
    for vc, tup in PDF_FIN.items():
        r, b = after["deals"].get(vc) or {}, before_local["deals"].get(vc) or {}
        pv = tup[3]
        bv = (b.get("total_pref") or 0) / 1e6
        av = (r.get("total_pref") or 0) / 1e6
        okb, oka = abs(bv - pv) <= TOL_M, abs(av - pv) <= TOL_M
        tb += okb
        ta += oka
        note = []
        if vc in DISPUTED:
            note.append("DISPUTED (Alay)")
        if vc in FALLBACK_VCODES:
            note.append("FALLBACK")
        if abs(av - bv) > TOL_M:
            note.append("moved")
        print(f"  {vc:<9}{tup[0][:29]:<30}{bv:>9,.2f}{av:>9,.2f}{pv:>7,.1f}"
              f"{('Y' if okb else '.'):>3}{('Y' if oka else '.'):>3}   "
              f"{' '.join(note)}")
    print(f"\n  Total Pref ties the PDF: before {tb}/33  ->  after {ta}/33")
    chk("Total Pref PDF ties improved", ta > tb, f"{tb} -> {ta}")
    chk("the 5 discriminating deals all tie on committed",
        all(abs(((after["deals"].get(vc) or {}).get("total_pref") or 0) / 1e6
                - PDF_FIN[vc][3]) <= TOL_M
            for vc in ("P0000019", "P0000088", "P0000067", "P0000110",
                       "P0000114")))
    chk("Pegasus stays the whole tranche 32,334,655 (TGA22 + PPILFS)",
        abs(((after["deals"].get("P0000066") or {}).get("total_pref") or 0)
            - 32_334_655) < 1)
    chk("the 3 disputed deals still differ from the PDF (expected)",
        all(abs(((after["deals"].get(vc) or {}).get("total_pref") or 0) / 1e6
                - PDF_FIN[vc][3]) > TOL_M for vc in DISPUTED))

    # ── Total Cap ────────────────────────────────────────────────────────
    print("\n" + "=" * 118)
    print("5. Total Cap re-foot, before -> after ($M)")
    print("=" * 118)
    print(f"  {'vcode':<9}{'deal':<30}{'before':>10}{'after':>10}{'PDF':>9}"
          f"{'B':>3}{'A':>3}")
    cb = ca = 0
    for vc, tup in PDF_FIN.items():
        r, b = after["deals"].get(vc) or {}, before_local["deals"].get(vc) or {}
        pv = tup[5]
        bv, av = (b.get("total_cap") or 0) / 1e6, (r.get("total_cap") or 0) / 1e6
        okb, oka = abs(bv - pv) <= TOL_M, abs(av - pv) <= TOL_M
        cb += okb
        ca += oka
        print(f"  {vc:<9}{tup[0][:29]:<30}{bv:>10,.2f}{av:>10,.2f}{pv:>9,.1f}"
              f"{('Y' if okb else '.'):>3}{('Y' if oka else '.'):>3}")
    print(f"\n  Total Cap ties the PDF: before {cb}/33  ->  after {ca}/33")
    chk("Total Cap did not get worse", ca >= cb, f"{cb} -> {ca}")
    chk("Total Cap == debt_isbs + Total Pref + Ptr Equity, every deal",
        all(abs(r["total_cap"] - ((cache[(vc, QUARTER)].get("cap_stack") or {})
                                  .get("debt_isbs", 0)
                                  + r["total_pref"] + r["ptr_equity"])) < 1
            for vc, r in after["deals"].items()
            if None not in (r.get("total_cap"), r.get("total_pref"),
                            r.get("ptr_equity"))))

    # ── inter-page agreement ─────────────────────────────────────────────
    print("\n" + "=" * 118)
    print("6. Page totals and inter-page agreement ($M)")
    print("=" * 118)
    for lbl, key in (("Debt", "debt"), ("Total Pref", "total_pref"),
                     ("Ptr Equity", "ptr_equity"), ("Total Cap", "total_cap"),
                     ("Invested", "invested"),
                     ("Total Commitment", "total_commitment"),
                     ("Un-funded", "unfunded")):
        b = (before_local["total"].get(key) or 0) / 1e6
        a = (after["total"].get(key) or 0) / 1e6
        print(f"  {lbl:<20}before {b:>10,.2f}   after {a:>10,.2f}   "
              f"delta {a - b:>+10,.2f}   PDF {PDF_TOTALS.get(key, 0):>8,.1f}")
    fin_c = (after["total"].get("total_commitment") or 0) / 1e6
    sum_c = (after["summary_total_committed"] or 0) / 1e6
    sum_c_b = (before_local["summary_total_committed"] or 0) / 1e6
    fin_c_b = (before_local["total"].get("total_commitment") or 0) / 1e6
    print(f"\n  page 2 Total Commitment   before {fin_c_b:>10,.2f}   "
          f"after {fin_c:>10,.2f}")
    print(f"  page 1 total_committed    before {sum_c_b:>10,.2f}   "
          f"after {sum_c:>10,.2f}")
    print(f"  INTER-PAGE GAP            before {sum_c_b - fin_c_b:>+10,.2f}   "
          f"after {sum_c - fin_c:>+10,.2f}")
    # Like-for-like population. Page 1's asset allocation deliberately drops a
    # kept-despite-sold deal — portfolio_snapshot_service, kept_vcodes — while
    # page 2 keeps its row. So the two totals cannot be equal outright; the
    # invariant is that they agree once that population difference is removed.
    #
    # Read off the ROW's flag, not KEEP_DESPITE_SOLD membership. The set names
    # every deal that WOULD be kept once sold; the gate only fires in the
    # quarters after the sale. East Manchester joined the set on 2026-09-02 and
    # is kept from 26Q2, but at 26Q1 — the quarter this script runs — it was
    # still held and is an ordinary row on BOTH pages. Subtracting it on
    # membership removed 2.72M page 1 had never excluded.
    kept = {vc: r for vc, r in after["deals"].items()
            if r.get("kept_despite_sold")}
    kept_contrib = sum((r.get("total_commitment") or 0) / 1e6
                       for r in kept.values())
    print(f"  page 2 less kept-despite-sold ({sorted(kept)}, "
          f"{kept_contrib:,.2f}M) = {fin_c - kept_contrib:,.2f}")
    chk("page 1 and page 2 agree on committed, like-for-like population",
        abs(sum_c - (fin_c - kept_contrib)) < 0.01,
        f"gap {sum_c - (fin_c - kept_contrib):+,.4f}M "
        f"(raw {sum_c - fin_c:+,.2f}M; was {sum_c_b - fin_c_b:+,.2f}M)")
    chk("the inter-page gap shrank by an order of magnitude",
        abs(sum_c - fin_c) < abs(sum_c_b - fin_c_b) / 10,
        f"{sum_c_b - fin_c_b:+,.2f}M -> {sum_c - fin_c:+,.2f}M")
    tp = (after["total"].get("total_pref") or 0) / 1e6
    disputed_delta = sum(
        ((after["deals"][vc].get("total_pref") or 0) / 1e6 - PDF_FIN[vc][3])
        for vc in DISPUTED)
    print(f"\n  portfolio Total Pref {tp:,.2f} less the 3 disputed deltas "
          f"({disputed_delta:+,.2f}) = {tp - disputed_delta:,.2f}   PDF 595.3")
    chk("portfolio Total Pref foots to the PDF once the disputes are removed",
        abs(tp - disputed_delta - 595.3) <= 0.1,
        f"{tp - disputed_delta:,.3f} vs 595.3")

    # ── subtotal % of Pref ───────────────────────────────────────────────
    print("\n" + "=" * 118)
    print("7. Subtotal '% of Pref' vs the PDF (must be unchanged)")
    print("=" * 118)
    print(f"  {'group':<34}{'before':>9}{'after':>9}{'PDF':>7}")
    for g, s in after["subtotals"].items():
        lbl = s["label"]
        bp = (before_local["subtotals"][g].get("pct_of_pref") or 0) * 100
        ap = (s.get("pct_of_pref") or 0) * 100
        print(f"  {lbl[:33]:<34}{bp:>9.1f}{ap:>9.1f}"
              f"{PDF_SUB_PCT.get(lbl, 0):>7}")

    npass = sum(1 for _, ok in checks if ok)
    print("\n" + "=" * 118)
    print(f"VERDICT: {npass}/{len(checks)} checks pass")
    for lbl, ok in checks:
        if not ok:
            print(f"  FAILED: {lbl}")
    return 0 if npass == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
