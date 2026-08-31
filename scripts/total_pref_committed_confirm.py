"""READ-ONLY confirm before switching Total Pref to the committed basis.

Runs on current main (ce0cc37) against live inputs. Changes nothing. Answers the
questions the implementation depends on:

  1. per deal: funded pref, committed_pe, and whether the funded FALLBACK is
     needed (committed_pe == 0 with real funded pref)
  2. Total Cap: which re-footing formula ties the PDF best, measured, not assumed
       A  shipped            cap_stack.total_cap_isbs (debt_isbs + funded + ptr)
       B  debt_isbs + committed + ptr
       C  resolved debt (footnote 6) + committed + ptr
  3. Invested must be pct x FUNDED — confirm it is separable from total_pref
  4. the two page totals today, and what they become on the committed basis
     (the $74M inter-page disagreement)
  5. subtotal "% of Pref" = Invested / Total Pref against the PDF's printed
     subtotal percentages

Usage
    WF_TOKEN=<jwt> python scripts/total_pref_committed_confirm.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd                                               # noqa: E402
import live_api as api                                            # noqa: E402
from snapshot_pdf_26q1_reconcile import PDF_FIN                    # noqa: E402
from flask_app.services.portfolio_snapshot_service import (        # noqa: E402
    resolve_investor_deals,
)
from flask_app.services.portfolio_snapshot_debt import (           # noqa: E402
    BASIS_ISBS, committed_facility, deal_loan_rows, resolve_debt,
)
from flask_app.services.portfolio_snapshot_operating import (      # noqa: E402
    is_dev_deal, resolve_strategy,
)

INVESTOR, QUARTER = "TGAM", "2026-Q1"
TOL = 0.051
PDF_TOTAL_PREF, PDF_TOTAL_CAP = 595.3, 2258.9
PDF_TOTAL_COMMITMENT, PDF_TOTAL_INVESTED = 445.1, 404.2
#: PDF page 2 printed subtotal "% of Pref", by group total row.
PDF_SUBTOTAL_PCT = {"Individual": 70, "TGA2022": 82, "TGA2023": 67,
                    "TGA2024": 77, "TGA2025": 90}

pd.set_option("display.width", 240)


def main():
    ti = api.token_info()
    print(f"LIVE token={ti['username']} ({ti['hours_left']}h)  "
          f"build={api.get('/api/data/version').get('version')}  "
          f"local HEAD={os.popen('git rev-parse --short HEAD').read().strip()}\n")

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
    if (d.get("total") or 0) > 500:
        print(f"  WARNING loans total={d['total']} > 500 — frame is short")

    resolved = resolve_investor_deals(INVESTOR, QUARTER, rel, inv)
    entries = {}
    for items in (resolved.get("groups") or {}).values():
        for e in items:
            entries[str(e["vcode"]).upper()] = e

    rows = []
    for vc, tup in PDF_FIN.items():
        name, pdf_pref, pdf_ptr, pdf_cap = tup[0], tup[3], tup[4], tup[5]
        e = entries.get(vc.upper())
        strat, _ = (resolve_strategy(e) if e else ("", ""))
        dev = is_dev_deal(strat) if e else None
        pct = (e or {}).get("lookthrough_pct_pe")
        p = api.get(f"/api/financials/{vc}/one-pager",
                    params={"quarter": QUARTER}) or {}
        cap = p.get("cap_stack") or {}
        pe = p.get("pe_performance") or {}
        funded = cap.get("pref_equity") or 0.0
        committed_raw = cap.get("committed_pe") or 0.0
        pe_committed = pe.get("committed_pe") or 0.0
        ptr = cap.get("partner_equity") or 0.0
        debt_isbs = cap.get("debt_isbs", cap.get("debt")) or 0.0
        try:
            debt_res, basis = resolve_debt(
                cap, bool(dev), committed_facility(deal_loan_rows(loans, vc)))
        except Exception:
            debt_res, basis = debt_isbs, BASIS_ISBS
        # the fallback under test
        needs_fb = (committed_raw == 0 and funded > 0)
        committed_fb = funded if needs_fb else committed_raw
        rows.append(dict(
            vcode=vc, name=name, dev=dev, group=tup[1], pct=pct,
            funded=funded, committed_raw=committed_raw,
            committed_fb=committed_fb, pe_committed=pe_committed,
            needs_fb=needs_fb, ptr=ptr,
            debt_isbs=debt_isbs, debt_res=debt_res or 0.0, debt_basis=basis,
            cap_isbs=cap.get("total_cap_isbs", cap.get("total_cap")) or 0.0,
            pdf_pref=pdf_pref, pdf_ptr=pdf_ptr, pdf_cap=pdf_cap))
    df = pd.DataFrame(rows)

    # ── 1. the fallback ──────────────────────────────────────────────────
    print("=" * 118)
    print("1. Which deals need the funded fallback?")
    print("=" * 118)
    fb = df[df.needs_fb]
    print(f"  deals with committed_pe == 0 but funded pref > 0: {len(fb)}")
    print(fb[["vcode", "name", "funded", "committed_raw", "pe_committed",
              "pdf_pref"]].to_string(index=False,
                                     float_format=lambda v: f"{v:,.2f}"))
    print("\n  does pe_performance.committed_pe already carry the fallback?")
    for _, r in fb.iterrows():
        ok = abs(r.pe_committed - r.funded) < 1
        print(f"    {r.vcode:<9}{r['name'][:26]:<27}pe_committed="
              f"{r.pe_committed:>14,.2f}  funded={r.funded:>14,.2f}   "
              f"{'YES' if ok else 'NO  <-- pe fallback did NOT fire'}")
    print("  -> a fallback inside get_capitalization_stack is required; "
          "pe_performance's cannot be relied on.")

    # ── 2. Total Cap variants ────────────────────────────────────────────
    df["capA"] = df.cap_isbs
    df["capB"] = df.debt_isbs + df.committed_fb + df.ptr
    df["capC"] = df.debt_res + df.committed_fb + df.ptr
    print("\n" + "=" * 118)
    print("2. Total Cap: which formula ties the PDF? (ties within +-0.051M)")
    print("=" * 118)
    for k, lbl in (("capA", "A shipped: debt_isbs + FUNDED + ptr (total_cap_isbs)"),
                   ("capB", "B debt_isbs + COMMITTED + ptr"),
                   ("capC", "C resolved debt (fn6) + COMMITTED + ptr")):
        ties = ((df[k] / 1e6 - df.pdf_cap).abs() <= TOL).sum()
        print(f"  {lbl:<52} ties {ties:>2}/33   "
              f"portfolio {df[k].sum() / 1e6:>8,.1f} vs PDF {PDF_TOTAL_CAP}")

    # ── 3. Total Pref ────────────────────────────────────────────────────
    print("\n" + "=" * 118)
    print("3. Total Pref: before (funded) vs after (committed + fallback)")
    print("=" * 118)
    tf = ((df.funded / 1e6 - df.pdf_pref).abs() <= TOL).sum()
    tc = ((df.committed_fb / 1e6 - df.pdf_pref).abs() <= TOL).sum()
    print(f"  funded ties {tf}/33   portfolio {df.funded.sum() / 1e6:,.2f}")
    print(f"  committed+fallback ties {tc}/33   "
          f"portfolio {df.committed_fb.sum() / 1e6:,.2f}   PDF {PDF_TOTAL_PREF}")

    # ── 4. the two page totals ───────────────────────────────────────────
    print("\n" + "=" * 118)
    print("4. Inter-page agreement (both columns are pct-scaled)")
    print("=" * 118)
    sc = df[df.pct.notna()]
    inv_scaled = (sc.pct * sc.funded).sum()
    fin_now = inv_scaled                      # COMMITMENT_BASIS='funded' today
    fin_after = (sc.pct * sc.committed_fb).sum()
    summ_now = (sc.pct * sc.committed_raw).sum()
    summ_after = fin_after
    print(f"  Invested (pct x funded)                 {inv_scaled / 1e6:>10,.2f}"
          f"   PDF {PDF_TOTAL_INVESTED}")
    print(f"  Financial Total Commitment  BEFORE      {fin_now / 1e6:>10,.2f}")
    print(f"  Summary  total_committed    BEFORE      {summ_now / 1e6:>10,.2f}")
    print(f"    inter-page gap BEFORE                 "
          f"{(summ_now - fin_now) / 1e6:>10,.2f}")
    print(f"  Financial Total Commitment  AFTER       {fin_after / 1e6:>10,.2f}")
    print(f"  Summary  total_committed    AFTER       {summ_after / 1e6:>10,.2f}")
    print(f"    inter-page gap AFTER                  "
          f"{(summ_after - fin_after) / 1e6:>10,.2f}   PDF "
          f"{PDF_TOTAL_COMMITMENT}")
    print("  (Summary moves too, by the fallback: "
          f"{(summ_after - summ_now) / 1e6:+,.2f}M)")

    # ── 5. subtotal % of Pref ────────────────────────────────────────────
    print("\n" + "=" * 118)
    print("5. Subtotal '% of Pref' = Invested / Total Pref vs the PDF")
    print("=" * 118)
    gmap = {"Individual": "Individual", "Individual+TGA2022": "Individual",
            "TGA2022": "TGA2022", "TGA2023": "TGA2023",
            "TGA2024": "TGA2024", "TGA2025": "TGA2025"}
    df["g"] = df.group.map(gmap)
    print(f"  {'group':<14}{'PDF %':>8}{'funded %':>11}{'committed %':>13}")
    for g, sub in df.groupby("g"):
        s = sub[sub.pct.notna()]
        i = (s.pct * s.funded).sum()
        pf = i / sub.funded.sum() * 100 if sub.funded.sum() else None
        pc = i / sub.committed_fb.sum() * 100 if sub.committed_fb.sum() else None
        print(f"  {g:<14}{PDF_SUBTOTAL_PCT.get(g, 0):>8}"
              f"{pf:>11.1f}{pc:>13.1f}")
    s = df[df.pct.notna()]
    i = (s.pct * s.funded).sum()
    print(f"  {'PORTFOLIO':<14}{68:>8}"
          f"{i / df.funded.sum() * 100:>11.1f}"
          f"{i / df.committed_fb.sum() * 100:>13.1f}")

    # ── 6. per-deal before/after ─────────────────────────────────────────
    print("\n" + "=" * 118)
    print("6. Per deal: Total Pref and Total Cap, before -> after ($M)")
    print("=" * 118)
    print(f"  {'vcode':<9}{'deal':<30}{'pref B':>9}{'pref A':>9}{'PDF':>7}"
          f"{'ok':>4}{'cap B':>10}{'cap A':>10}{'PDF':>8}{'ok':>4}  note")
    for _, r in df.iterrows():
        pb, pa = r.funded / 1e6, r.committed_fb / 1e6
        cb, ca = r.capA / 1e6, r.capB / 1e6
        note = []
        if r.needs_fb:
            note.append("FALLBACK")
        if abs(pa - pb) > TOL:
            note.append("pref moves")
        print(f"  {r.vcode:<9}{r['name'][:29]:<30}{pb:>9,.2f}{pa:>9,.2f}"
              f"{r.pdf_pref:>7,.1f}"
              f"{('Y' if abs(pa - r.pdf_pref) <= TOL else '.'):>4}"
              f"{cb:>10,.2f}{ca:>10,.2f}{r.pdf_cap:>8,.1f}"
              f"{('Y' if abs(ca - r.pdf_cap) <= TOL else '.'):>4}  "
              f"{' '.join(note)}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "outputs", "total_pref_committed_confirm.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nwrote {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
