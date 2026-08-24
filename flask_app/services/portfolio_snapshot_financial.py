"""Portfolio Snapshot — Subtab 2 (Financial) data assembly (Step 5a).

Backend only: no route, no blueprint, no UI. Nothing imports this module.

Reads, never writes:
  * Step 1  ``resolve_investor_deals`` — deals by fund, and the look-through %
  * Step 3a ``is_dev_deal`` — carried per row as information only; this subtab
            has no "Excluding Development Deals" total (that is Loan-tab only)
  * One Pager ``get_capitalization_stack`` (via the injected provider) — every
            Zone A column, so they are byte-consistent with the One Pager
  * Step 2  ``portfolio_snapshot_persistence`` — manual values + footnotes

Three column zones:

Zone A — deal-level capitalisation, NOT scaled
    Debt, Total Pref, Ptr Equity, Total Cap and the cap-stack percentages, taken
    straight from the One Pager payload.

Zone B — the four "TIAA Investment" columns, the ONLY scaled columns
    % of Pref        = Step 1's multi-hop look-through (Nottingham 41.2124%)
    Invested         = % of Pref x funded pref        (cap_stack.pref_equity)
    Total Commitment = % of Pref x commitment basis   (see COMMITMENT_BASIS)
    Un-funded        = Total Commitment - Invested

    COMMITMENT_BASIS is a deliberate switch because the two candidates disagree
    and only one reproduces the PDF. Verified on Nottingham at 26Q1:
        funded pref            9,135,000  -> Commitment 3,764,751, Un-funded 0
        committed_pe          12,058,426  -> Commitment 4,969,564, Un-funded 1,204,814
    The PDF shows Total Commitment ~3.8M and Un-funded blank, i.e. it scales
    *funded*. Note 9,135,000 + 2,923,426.79 (a contribution dated 2026-06-01)
    equals the 12,058,426.79 commitment row to the cent, so the committed figure
    is real — the PDF simply is not using it. Both are always computed and
    returned; this constant only decides which fills ``total_commitment``.

Zone C — manual entry, never derived (formula TBD)
    Net ROE and ITD Distributions are per-deal editable boxes. Analysts type
    them from the Acct Excel; nothing here computes them.

    Storage is Step 2's ``portfolio_snapshot_values``, keyed
    (investor_code, quarter, deal_vcode, field) with field in
    {"net_roe", "itd"}. Writes go through the Step 2 approval pipeline —
    ``save_value`` honours ``is_editable``, so an approved page rejects edits,
    and a saved value resets to the page's current review status. Absent ->
    ``pending entry``, never a fabricated number and never zero.

    *** MANUAL FOR NOW — FORMULA TBD ***
    The assembly reads these through exactly two accessors, ``get_net_roe()``
    and ``get_itd()``. When the methods are settled (Net ROE: net of fund-level
    expenses weighted by dollars invested and time; ITD: footnote-1 fee
    allocation) the computation drops into the body of those two functions and
    nothing in the assembly changes.

Totals: per-fund subtotals and a portfolio total over **all** deals. There is
deliberately no "Excluding Development Deals" subtotal on this subtab — that
total belongs to the Loan subtab only.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

log = logging.getLogger(__name__)

#: Which figure the Total Commitment column scales. "funded" reproduces the
#: reference PDF; "committed_pe" uses the accounting commitment rows, which are
#: the authoritative commitment but do not match the published page.
COMMITMENT_BASIS = "funded"          # "funded" | "committed_pe"

PENDING = "pending entry"

MANUAL_FIELDS = ("net_roe", "itd")

#: Zone A columns summed for subtotals. % columns are recomputed from the sums
#: rather than added, since averaging percentages is meaningless.
_SUM_COLS = ("debt", "total_pref", "ptr_equity", "total_cap",
             "committed_pref", "invested", "total_commitment", "unfunded")


def _num(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f


def _load_manual(investor_code: str, quarter: str) -> dict:
    """{vcode: {field: value}} from Step 2 persistence (read-only)."""
    out: dict = {}
    try:
        from flask_app.services.portfolio_snapshot_persistence import get_elements
        for r in get_elements("value", investor_code, quarter):
            out.setdefault(r.get("deal_vcode"), {})[r.get("field")] = r.get("value")
    except Exception as exc:
        log.debug("manual values unavailable: %s", exc)
    return out


# ── Zone C accessors — the ONLY read path for Net ROE and ITD ─────────────
#
# The assembly never touches the values table directly. Swapping either column
# from manual entry to a computed formula means replacing the body of one
# function here; assemble_financial does not change.

def _manual_value(field: str, deal_vcode: str, investor_code: str,
                  quarter: str, manual: Optional[dict] = None):
    """Stored manual entry for one (deal, field), or None if not entered.

    ``manual`` is an optional prefetched {vcode: {field: value}} map so a page
    of 35 deals costs one query instead of 70. Without it the accessor stands
    alone and loads for itself.
    """
    if manual is None:
        manual = _load_manual(investor_code, quarter)
    return _num((manual.get(deal_vcode) or {}).get(field))


def get_net_roe(deal_vcode: str, investor_code: str, quarter: str,
                manual: Optional[dict] = None):
    """Net ROE for one deal.

    MANUAL ENTRY FOR NOW — FORMULA TBD. Returns the analyst-entered value from
    portfolio_snapshot_values (field 'net_roe'), or None when nothing has been
    entered; the caller renders that as "pending entry".

    To automate: compute here and return the number. The intended method is net
    of fund-level expenses, weighted by dollars invested and time — which needs
    an expense-allocation source that does not exist in the app yet. Until then
    a typed figure is the only honest option, and returning None rather than 0
    keeps an un-entered cell visibly empty.
    """
    return _manual_value("net_roe", deal_vcode, investor_code, quarter, manual)


def get_itd(deal_vcode: str, investor_code: str, quarter: str,
            manual: Optional[dict] = None):
    """ITD Distributions for one deal.

    MANUAL ENTRY FOR NOW — FORMULA TBD. Same contract as get_net_roe.

    To automate: the raw inception-to-date distribution total is already
    derivable from the accounting feed (and the One Pager's PE block exposes
    return_of_capital), but the reported figure carries the footnote-1 fee
    allocation, which is the part with no data source. Compute here once that
    allocation is defined.
    """
    return _manual_value("itd", deal_vcode, investor_code, quarter, manual)


def _load_footnotes(investor_code: str, quarter: str) -> list:
    try:
        from flask_app.services.portfolio_snapshot_persistence import get_elements
        return get_elements("footnote", investor_code, quarter)
    except Exception as exc:
        log.debug("footnotes unavailable: %s", exc)
        return []


def _subtotal(rows: list, label: str) -> dict:
    """Sum the dollar columns; recompute ratios from the sums."""
    out = {"label": label, "deal_count": len(rows)}
    for c in _SUM_COLS:
        vals = [r[c] for r in rows if r.get(c) is not None]
        out[c] = sum(vals) if vals else None
    tp, tc = out.get("total_pref"), out.get("total_cap")
    out["pct_of_pref"] = ((out["invested"] / tp)
                          if (out.get("invested") is not None and tp) else None)
    out["debt_pct"] = (out["debt"] / tc) if (out.get("debt") is not None and tc) else None
    out["pref_pct"] = (tp / tc) if (tp is not None and tc) else None
    out["ptr_pct"] = ((out["ptr_equity"] / tc)
                      if (out.get("ptr_equity") is not None and tc) else None)
    # Manual columns are never summed — a subtotal of partly-entered figures
    # would read as complete. Counted instead.
    out["manual_entered"] = {
        f: sum(1 for r in rows if r.get(f) not in (None, PENDING))
        for f in MANUAL_FIELDS}
    return out


def assemble_financial(investor_code: str, quarter: str, *,
                       resolved: dict,
                       one_pager_provider: Callable[[str, str], dict],
                       manual_loader: Optional[Callable] = None,
                       footnote_loader: Optional[Callable] = None,
                       commitment_basis: str = COMMITMENT_BASIS) -> dict:
    """Build the Financial subtab for one investor and quarter."""
    from flask_app.services.portfolio_snapshot_operating import is_dev_deal

    manual = (manual_loader or _load_manual)(investor_code, quarter) or {}
    footnotes = (footnote_loader or _load_footnotes)(investor_code, quarter) or []

    diag = {"deals": 0, "dev": 0, "provider_errors": 0,
            "pct_unavailable": 0, "commitment_missing": 0,
            "manual_pending": 0, "manual_entered": 0}

    def build_row(entry: dict, extra_flags: Optional[list] = None) -> dict:
        vcode = entry["vcode"]
        flags = list(extra_flags or [])
        strat = entry.get("investment_strategy", "")
        dev = is_dev_deal(strat)
        if dev:
            diag["dev"] += 1

        try:
            payload = one_pager_provider(vcode, quarter) or {}
        except Exception as exc:
            diag["provider_errors"] += 1
            flags.append(f"One Pager unavailable: {str(exc)[:80]}")
            payload = {}
        cap = payload.get("cap_stack") or {}

        # ---- Zone A: deal-level, unscaled ----
        debt = _num(cap.get("debt"))
        total_pref = _num(cap.get("pref_equity"))
        ptr_equity = _num(cap.get("partner_equity"))
        total_cap = _num(cap.get("total_cap"))
        committed_pref = _num(cap.get("committed_pe"))

        # ---- Zone B: the four scaled columns ----
        pct = entry.get("lookthrough_pct")
        if pct is None:
            diag["pct_unavailable"] += 1
            flags.append("% of Pref unavailable — ownership chain unresolved")

        invested = (pct * total_pref) if (pct is not None
                                          and total_pref is not None) else None
        basis_val = (total_pref if commitment_basis == "funded"
                     else committed_pref)
        if commitment_basis == "committed_pe" and not committed_pref:
            diag["commitment_missing"] += 1
            flags.append("no commitment row — Total Commitment unavailable")
        total_commitment = (pct * basis_val) if (pct is not None
                                                and basis_val is not None) else None
        unfunded = ((total_commitment - invested)
                    if (total_commitment is not None and invested is not None)
                    else None)
        # Both bases carried so the choice is auditable and reversible.
        commitment_funded = (pct * total_pref) if (pct is not None
                                                  and total_pref is not None) else None
        commitment_committed = (pct * committed_pref) if (
            pct is not None and committed_pref is not None) else None

        # ---- Zone C: manual entry, read only through the two accessors ----
        row_manual = {}
        for f, accessor in (("net_roe", get_net_roe), ("itd", get_itd)):
            v = accessor(vcode, investor_code, quarter, manual)
            row_manual[f] = v
            row_manual[f + "_display"] = PENDING if v is None else v
            row_manual[f + "_source"] = "manual entry (formula TBD)"
            if v is None:
                diag["manual_pending"] += 1
            else:
                diag["manual_entered"] += 1

        diag["deals"] += 1
        return {
            "vcode": vcode, "name": entry["name"],
            "investment_strategy": strat, "is_dev": dev,
            # Zone A
            "debt": debt, "total_pref": total_pref, "ptr_equity": ptr_equity,
            "total_cap": total_cap, "committed_pref": committed_pref,
            "debt_pct": _num(cap.get("debt_pct")),
            "pref_pct": _num(cap.get("pref_equity_pct")),
            "ptr_pct": _num(cap.get("partner_equity_pct")),
            "pe_exposure_on_cap": _num(cap.get("pe_exposure_on_cap")),
            # Zone B (scaled)
            "pct_of_pref": pct,
            "invested": invested,
            "total_commitment": total_commitment,
            "unfunded": unfunded,
            "commitment_basis": commitment_basis,
            "total_commitment_if_funded": commitment_funded,
            "total_commitment_if_committed": commitment_committed,
            # Zone C (manual)
            **row_manual,
            "flags": flags,
        }

    groups: dict[str, dict] = {}
    all_rows: list = []
    for group, items in (resolved.get("groups") or {}).items():
        rows = [build_row(e) for e in items]
        groups[group] = {"deals": rows, "subtotal": _subtotal(rows, group)}
        all_rows.extend(rows)

    flagged_rows = []
    for f in (resolved.get("flagged") or []):
        row = build_row(f, extra_flags=[f"ownership {f.get('reason','unavailable')}"])
        row["ownership_flagged"] = True
        flagged_rows.append(row)

    # Portfolio total over ALL deals, including the ownership-flagged ones:
    # their Zone A figures are deal-level and ownership-independent, so they
    # belong in the total; only their Zone B dollars stay None.
    #
    # No "Excluding Development Deals" total here — that subtotal lives on the
    # Loan subtab only (creator decision 2026-08-24). `is_dev` is still carried
    # per row as information, but nothing on this subtab acts on it.
    total_rows = all_rows + flagged_rows
    total = _subtotal(total_rows, "Portfolio Total")

    return {
        "investor_code": resolved.get("investor_code", investor_code),
        "investor_name": resolved.get("investor_name", investor_code),
        "quarter": quarter, "subtab": "financial",
        "scaled_columns": ["pct_of_pref", "invested", "total_commitment",
                           "unfunded"],
        "commitment_basis": commitment_basis,
        "groups": groups,
        "ownership_flagged": flagged_rows,
        "total": total,
        "footnotes": footnotes,
        "diagnostics": diag,
    }


# ── Self-test ─────────────────────────────────────────────────────────────

# PDF 26Q1, Financial page — the values supplied for Nottingham
_PDF = {
    "P0000030": {"name": "Nottingham Village", "pct_of_pref": 41.0,
                 "invested_m": 3.8, "commitment_m": 3.8, "unfunded_m": 0.0,
                 "total_pref_m": 9.1},
}


def _selftest():                                    # pragma: no cover
    import json
    import os
    import sys
    import tempfile
    import sqlalchemy
    import pandas as pd

    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    for p in (root, os.path.join(root, "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import live_api as api
    from flask_app.services.portfolio_snapshot_service import resolve_investor_deals
    from flask_app.services import portfolio_snapshot_persistence as P

    INV, Q = "TGAM", "2026-Q1"
    ti = api.token_info()
    print(f"LIVE token={ti['username']} ({ti['hours_left']}h)  "
          f"build={api.get('/api/data/version').get('version')}  "
          f"actuals_through={api.get('/api/data/config').get('actuals_through')}")

    inv = pd.DataFrame(api.get("/api/data/deals/all").get("deals") or [])
    seen, frontier, rows = set(), [INV], []

    def fetch(col, v):
        d = api.get("/api/data/tables/relationships/rows",
                    params={"page": 1, "page_size": 500, f"filter__{col}": v})
        return [r for r in (d.get("rows") or [])
                if str(r.get(col) or "").strip().upper() == v.upper()]

    while frontier:
        node = frontier.pop().upper()
        if node in seen:
            continue
        seen.add(node)
        kids = fetch("InvestorID", node)
        rows.extend(kids)
        for r in kids:
            c = str(r.get("InvestmentID") or "").strip().upper()
            if c:
                rows.extend(fetch("InvestmentID", c))
                if c not in seen:
                    frontier.append(c)
    rel = pd.DataFrame(rows).drop_duplicates()
    resolved = resolve_investor_deals(INV, Q, rel, inv)
    print(f"Step 1: {resolved['diagnostics']['deal_count']} deals, "
          f"{len(resolved['flagged'])} ownership-flagged\n")

    # Step 2 on a scratch db: two footnotes, no manual values (so Zone C is
    # exercised in its 'pending entry' state, which is the point of the check).
    tmp = os.path.join(tempfile.mkdtemp(prefix="ps_step5a_"), "t.db")
    eng = sqlalchemy.create_engine(f"sqlite:///{tmp}")
    P._engine = lambda: eng                          # type: ignore[assignment]
    P._is_postgres = lambda: False                   # type: ignore[assignment]
    P.add_footnote(INV, Q, "itd_distributions", "Net of the footnote-1 fee "
                   "allocation.", updated_by="selftest")
    P.add_footnote(INV, Q, "net_roe", "Net of fund-level expenses, weighted by "
                   "dollars invested and time.", updated_by="selftest")

    cache: dict = {}

    def provider(vc, q):
        if (vc, q) not in cache:
            cache[(vc, q)] = api.get(f"/api/financials/{vc}/one-pager",
                                     params={"quarter": q})
        return cache[(vc, q)]

    out = assemble_financial(INV, Q, resolved=resolved,
                             one_pager_provider=provider,
                             manual_loader=lambda i, q: _load_manual(i, q),
                             footnote_loader=lambda i, q: _load_footnotes(i, q))

    flat = {r["vcode"]: r for g in out["groups"].values() for r in g["deals"]}
    for r in out["ownership_flagged"]:
        flat[r["vcode"]] = r

    checks = []

    def chk(label, cond):
        checks.append((label, bool(cond)))
        print(f"    [{'PASS' if cond else 'FAIL'}] {label}")

    print("=" * 118)
    print("ZONE B — the four scaled TIAA columns (Nottingham vs PDF)")
    print(f"{'metric':<26}{'computed':>16}{'PDF':>10}{'delta':>12}   verdict")
    print("-" * 118)
    r = flat.get("P0000030") or {}
    p = _PDF["P0000030"]
    tests = [
        ("% of Pref", (r.get("pct_of_pref") or 0) * 100, p["pct_of_pref"], 0.5),
        ("Invested ($M)", (r.get("invested") or 0) / 1e6, p["invested_m"], 0.06),
        ("Total Commitment ($M)", (r.get("total_commitment") or 0) / 1e6,
         p["commitment_m"], 0.06),
        ("Un-funded ($M)", (r.get("unfunded") or 0) / 1e6, p["unfunded_m"], 0.06),
        ("Total Pref ($M)", (r.get("total_pref") or 0) / 1e6,
         p["total_pref_m"], 0.06),
    ]
    for metric, comp, pdf, tol in tests:
        d = comp - pdf
        ok = abs(d) <= tol
        print(f"{metric:<26}{comp:>16.4f}{pdf:>10.2f}{d:>+12.4f}   "
              f"{'ok' if ok else 'MISMATCH'}")
        checks.append((f"Nottingham {metric}", ok))

    print(f"\n  commitment basis in use: {out['commitment_basis']!r}")
    print(f"  Nottingham both bases:  funded -> "
          f"{(r.get('total_commitment_if_funded') or 0)/1e6:.4f}M   "
          f"committed_pe -> "
          f"{(r.get('total_commitment_if_committed') or 0)/1e6:.4f}M   "
          f"(committed_pref {(r.get('committed_pref') or 0)/1e6:.4f}M)")

    print("\n" + "=" * 118)
    print("ZONE A + ZONE B — all deals by fund")
    hdr = (f"{'vcode':<9}{'deal':<27}{'dev':<4}{'Debt':>13}{'TotPref':>12}"
           f"{'PtrEq':>12}{'TotalCap':>13}{'%Pref':>8}{'Invested':>12}"
           f"{'Commit':>12}{'Unfund':>10}")
    print(hdr)
    print("-" * 118)
    for g, blk in out["groups"].items():
        print(f"  -- {g}")
        for r_ in blk["deals"]:
            print(f"{r_['vcode']:<9}{r_['name'][:26]:<27}"
                  f"{'Y' if r_['is_dev'] else '':<4}"
                  f"{(r_['debt'] or 0):>13,.0f}{(r_['total_pref'] or 0):>12,.0f}"
                  f"{(r_['ptr_equity'] or 0):>12,.0f}{(r_['total_cap'] or 0):>13,.0f}"
                  f"{((r_['pct_of_pref'] or 0)*100):>7.2f}%"
                  f"{(r_['invested'] or 0):>12,.0f}"
                  f"{(r_['total_commitment'] or 0):>12,.0f}"
                  f"{(r_['unfunded'] or 0):>10,.0f}")
        s = blk["subtotal"]
        print(f"{'':9}{'SUBTOTAL ' + g[:17]:<27}{'':<4}"
              f"{(s['debt'] or 0):>13,.0f}{(s['total_pref'] or 0):>12,.0f}"
              f"{(s['ptr_equity'] or 0):>12,.0f}{(s['total_cap'] or 0):>13,.0f}"
              f"{((s['pct_of_pref'] or 0)*100):>7.2f}%"
              f"{(s['invested'] or 0):>12,.0f}{(s['total_commitment'] or 0):>12,.0f}"
              f"{(s['unfunded'] or 0):>10,.0f}\n")
    for r_ in out["ownership_flagged"]:
        print(f"{r_['vcode']:<9}{r_['name'][:26]:<27}{'':<4}"
              f"{(r_['debt'] or 0):>13,.0f}{(r_['total_pref'] or 0):>12,.0f}"
              f"{(r_['ptr_equity'] or 0):>12,.0f}{(r_['total_cap'] or 0):>13,.0f}"
              f"{'n/a':>8}{'n/a':>12}{'n/a':>12}{'n/a':>10}   <- ownership flagged")

    t = out["total"]
    print(f"\nPORTFOLIO TOTAL — {t['deal_count']} deals, all included "
          f"(no ex-dev total on this subtab)")
    for lbl, key in (("Debt", "debt"), ("Total Pref", "total_pref"),
                     ("Ptr Equity", "ptr_equity"), ("Total Cap", "total_cap"),
                     ("Invested", "invested"),
                     ("Total Commitment", "total_commitment"),
                     ("Un-funded", "unfunded")):
        print(f"      {lbl:<18}{(t[key] or 0):>18,.0f}")
    print(f"      {'manual entered':<18}{str(t['manual_entered']):>18}")

    print("\n" + "=" * 118)
    print("ZONE C — manual columns must read 'pending entry', not 0")
    for vc in ("P0000030", "P0000075", "P0000019"):
        rr = flat.get(vc) or {}
        print(f"  {vc} {rr.get('name','')[:26]:<28}"
              f"net_roe={rr.get('net_roe_display')!r:<18}"
              f"itd={rr.get('itd_display')!r}")

    print("\n" + "=" * 118)
    print("FOOTNOTES carried on the structure")
    for f_ in out["footnotes"]:
        print(f"  ({f_['number']}) anchor={f_['anchor']!r}  {f_['text'][:58]}")

    print("\n" + "=" * 118)
    print("STRUCTURE CHECKS")
    chk("groups present with subtotals",
        all("subtotal" in b and "deals" in b for b in out["groups"].values()))
    chk("only the 4 TIAA columns are declared scaled",
        out["scaled_columns"] == ["pct_of_pref", "invested",
                                  "total_commitment", "unfunded"])
    chk("Zone A comes from the One Pager cap stack (Nottingham pref 9,135,000)",
        abs((flat.get("P0000030") or {}).get("total_pref", 0) - 9135000) < 1)
    chk("Total Cap = Debt + Total Pref + Ptr Equity (Nottingham)",
        abs((r.get("total_cap") or 0)
            - ((r.get("debt") or 0) + (r.get("total_pref") or 0)
               + (r.get("ptr_equity") or 0))) < 1)
    chk("Un-funded = Commitment - Invested for every deal",
        all(x["unfunded"] is None
            or abs(x["unfunded"] - (x["total_commitment"] - x["invested"])) < 1e-6
            for x in flat.values()
            if x["total_commitment"] is not None and x["invested"] is not None))
    chk("Net ROE pending for all deals (none entered)",
        all(x["net_roe_display"] == PENDING for x in flat.values()))
    chk("ITD pending for all deals (none entered)",
        all(x["itd_display"] == PENDING for x in flat.values()))
    chk("manual columns are None underneath, not 0",
        all(x["net_roe"] is None and x["itd"] is None for x in flat.values()))
    chk("45th & Main present, Zone B withheld",
        any(x["vcode"] == "P0000089" and x["pct_of_pref"] is None
            for x in out["ownership_flagged"]))
    chk("2 footnotes carried, numbered 1..2",
        [f_["number"] for f_ in out["footnotes"]] == [1, 2])
    chk("subtotals sum their deals (Individual Investments invested)",
        abs((out["groups"].get("Individual Investments", {})
             .get("subtotal", {}).get("invested") or 0)
            - sum(x["invested"] or 0 for x in
                  out["groups"].get("Individual Investments", {}).get("deals", []))) < 1)
    chk("no Excluding-Development total on this subtab",
        "total_excluding_dev" not in out)
    n_all = sum(len(b["deals"]) for b in out["groups"].values())         + len(out["ownership_flagged"])
    chk("portfolio total covers ALL deals (incl. ownership-flagged)",
        out["total"]["deal_count"] == n_all)
    chk("total Debt equals the sum of every deal's Debt",
        abs((out["total"]["debt"] or 0)
            - sum(x["debt"] or 0 for x in flat.values())) < 1)
    chk("Zone C read through the accessors (source tagged)",
        all(x.get("net_roe_source") == "manual entry (formula TBD)"
            and x.get("itd_source") == "manual entry (formula TBD)"
            for x in flat.values()))
    chk("get_net_roe/get_itd return None when nothing is entered",
        get_net_roe("P0000030", INV, Q) is None
        and get_itd("P0000030", INV, Q) is None)

    d = out["diagnostics"]
    print(f"\n  diagnostics: {d}")

    print("\n" + "=" * 118)
    print("ASSEMBLED STRUCTURE — one deal per fund")
    seen_g = set()
    for g, blk in out["groups"].items():
        if not blk["deals"] or g in seen_g:
            continue
        seen_g.add(g)
        print(f"\n{g} — {blk['deals'][0]['vcode']}:")
        print(json.dumps(blk["deals"][0], indent=2, default=str)[:900])
        if len(seen_g) >= 2:
            break

    print("\n" + "=" * 118)
    print("ZONE C round-trip — enter a value through the pipeline, read it back")
    P.save_value(INV, Q, "P0000030", "net_roe", 0.0912, updated_by="selftest")
    P.save_value(INV, Q, "P0000030", "itd", 1250000.0, updated_by="selftest")
    chk("get_net_roe returns the entered value",
        abs((get_net_roe("P0000030", INV, Q) or 0) - 0.0912) < 1e-12)
    chk("get_itd returns the entered value",
        abs((get_itd("P0000030", INV, Q) or 0) - 1250000.0) < 1e-6)
    out2 = assemble_financial(INV, Q, resolved=resolved,
                              one_pager_provider=provider)
    f2 = {r["vcode"]: r for g in out2["groups"].values() for r in g["deals"]}
    n2 = f2.get("P0000030") or {}
    print(f"  Nottingham after entry: net_roe={n2.get('net_roe_display')}  "
          f"itd={n2.get('itd_display')}")
    print(f"  Camp Creek (untouched): net_roe="
          f"{(f2.get('P0000075') or {}).get('net_roe_display')!r}")
    chk("assembly surfaces the entered value, not 'pending entry'",
        n2.get("net_roe_display") == 0.0912
        and n2.get("itd_display") == 1250000.0)
    chk("other deals still pending",
        (f2.get("P0000075") or {}).get("net_roe_display") == PENDING)
    chk("total counts entered values without summing them",
        out2["total"]["manual_entered"]["net_roe"] == 1)

    print("\n  approval gate on manual entry:")
    P.submit_for_review(INV, Q, 1, "cbui", roles=["asset_manager"])
    for role in ("head_am", "president", "cco", "ceo"):
        P.approve(INV, Q, 9, "approver_" + role, roles=[role])
    try:
        P.save_value(INV, Q, "P0000030", "net_roe", 0.5)
        chk("approved page refuses a manual-value edit", False)
    except P.NotEditable:
        chk("approved page refuses a manual-value edit", True)
    chk("the approved value is still readable",
        abs((get_net_roe("P0000030", INV, Q) or 0) - 0.0912) < 1e-12)

    print(f"\n  {sum(1 for _, c in checks if c)}/{len(checks)} checks passed")
    return 0


if __name__ == "__main__":                          # pragma: no cover
    raise SystemExit(_selftest())
