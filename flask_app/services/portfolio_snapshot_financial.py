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
    Debt, Total Pref, Ptr Equity, Total Cap and the cap-stack percentages.

    TOTAL PREF IS THE COMMITTED PREF TRANCHE (``cap_stack.committed_pe``), not
    funded. Established against the baseline PDF at 26Q1 three independent ways:

      * The PDF's own arithmetic. ``Invested = pct x FUNDED`` holds 33/33, and
        ``Total Commitment = pct x (the Total Pref column)`` holds 30/33 — the
        three misses being rounding on the printed one-decimal figure (Mount
        Prospect, Town Fair) and Brainerd's printed 74% disagreeing with the
        live 63.142%. So ``Un-funded = pct x (Total Pref - funded)``, i.e. the
        Total Pref column is the committed basis by construction.
      * The six deals where the PDF itself prints a non-zero Un-funded — its own
        data declaring committed > funded — span dev and non-dev alike, and
        Total Pref matches ``committed_pe`` on five of six (Giant 7 21.00,
        Brainerd 31.72, Seasons 35.45, Trolley 6.75, Stephens 22.73). There is
        no dev/non-dev split.
      * The portfolio total. Funded sums to 546.88 (-48.4 vs the PDF's 595.3);
        committed to 641.64 (+46.3); committed less the three disputed
        commitment VALUES gives 595.35 against 595.3.

    Per-deal ties go 27/33 funded -> 30/33 committed. The three that still
    differ — Nottingham, Burton, JB Fair Park — are commitment-VALUE disputes
    (the PDF blanks their Un-funded, i.e. its source thinks them fully funded),
    not a basis question. They are the IA_Contribution-vs-IA_Commitment item
    waiting on Alay; do not bend the basis to them.

    PTR EQUITY STAYS FUNDED. The OP side has commitment rows on 31 of 33 deals,
    but switching to them scores WORSE — 28/33 funded against 27/33 committed —
    and the data is not trustworthy: Brainerd's larger OP "Commitment" row
    (18,777,867.84) equals its cumulative funded contributions to the cent, so
    summing both OP rows double-counts by 10.6M. Unlike the pref rows, the OP
    rows carry no 1% acquisition fee, so the test that corroborated the pref
    tranches cannot be run. Left alone deliberately.

    Total Cap is RE-FOOTED as debt_isbs + Total Pref + Ptr Equity so the four
    printed columns add up. The debt leg stays the ISBS/current basis, NOT the
    footnote-6 dev rebase — see the total_cap block in build_row.

Zone B — the four "TIAA Investment" columns, the ONLY scaled columns
    % of Pref        = Step 1's multi-hop look-through (Nottingham 41.2124%)
    Invested         = % of Pref x funded pref        (cap_stack.pref_equity)
    Total Commitment = % of Pref x commitment basis   (see COMMITMENT_BASIS)
    Un-funded        = Total Commitment - Invested

    INVESTED STAYS ON FUNDED. It ties the PDF 33/33 and is the one column that
    must not follow the Total Pref switch — it is TIAA's actually-contributed
    slice, not its pledge. Both commitment bases are always computed and
    returned (``total_commitment_if_funded`` / ``_if_committed``) so the choice
    stays auditable and reversible.

    Switching Total Commitment onto the committed basis also closes the
    inter-page disagreement: page 1 (portfolio_snapshot_summary) has always
    scaled ``committed_pe``, while this page shipped ``COMMITMENT_BASIS="funded"``,
    so the same figure read 409.23 here and 478.28 there at 26Q1 — a 69.05M gap.
    Both now read 485.99. The residual against the PDF's 445.1 is the same three
    disputed commitment values.

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

Totals: per-fund subtotals and a portfolio total over **all** deals, labelled
with the reference PDF's wording via ``portfolio_snapshot_service`` (so the
three table subtabs cannot drift apart), plus an "Excluding Development Deals"
row under the portfolio total — see ``EXCLUDING_DEV_VCODES``.

Display-only suppression, as on the Operating subtab: the ``*_display`` twins
carry ``NA_LABEL`` where the PDF prints n/a, and the raw fields are left exactly
as computed. Nothing here changes a metric value.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

log = logging.getLogger(__name__)

#: Which figure the Total Commitment column scales. "committed_pe" is the
#: accounting commitment rows — the authoritative commitment, the basis the
#: reference PDF's Total Pref column is on, and what page 1 has always scaled.
#: "funded" is retained so the pre-2026-08-31 numbers can be reproduced.
COMMITMENT_BASIS = "committed_pe"    # "funded" | "committed_pe"

PENDING = "pending entry"

#: What a cell reads where the reference PDF prints "n/a". Distinct from
#: PENDING (nothing typed yet) and from None (no data) — see PDF_NA_CELLS.
NA_LABEL = "n/a"

MANUAL_FIELDS = ("net_roe", "itd")

# ══════════════════════════════════════════════════════════════════════════
# TEMPORARY HARDCODE — the PDF's "Excluding Development Deals" population
# ══════════════════════════════════════════════════════════════════════════
#: The deals the PDF's excluding-development row removes, keyed by vcode.
#:
#: This is NOT ``is_dev``, and the difference is the whole reason it is a
#: hardcode. Our classification (Lifecycle proxy, via ``resolve_strategy``)
#: marks TEN deals development at 26Q1. The PDF's row removes EIGHT: it keeps
#: JB Fair Park and Pegasus Life Storage in the subtotal, both of which produce
#: income (ITD 1.17 / ROE 4.8% and ITD 0.91 / ROE 2.8% on page 2) and so are not
#: "development" for this purpose.
#:
#: Derived from the PDF, not guessed: Portfolio Total Commitment 445.1 less the
#: excluding-dev 299.3 leaves 145.8 to explain, and these eight deals'
#: Total Commitment sums to exactly 145.8 (23.6 + 20.7 + 22.3 + 18.0 + 16.7 +
#: 18.0 + 6.1 + 20.4). Nine subsets of the candidate pool hit 145.8 arithmetically,
#: so the fit alone is not proof; this is the only one that is also coherent —
#: it is exactly the deals whose page-3 comments read "Construction in progress"
#: / "Pref equity funding has started", and exactly the negative-Net-ROE rows
#: minus the two "Recent acquisition, not enough operating history" deals
#: (Hanestowne Village, Plaza Del Mar), which are not development deals.
#:
#: CONFIRM WITH THE AUTHOR before trusting the row. The rule it stands in for
#: needs a development/stabilisation state in the data that does not exist —
#: once it does, delete this and filter on it.
EXCLUDING_DEV_VCODES: frozenset = frozenset({
    "P0000067",     # Brainerd Place Apartments
    "P0000078",     # Jefferson Waters Creek
    "P0000077",     # Jefferson Addison Heights
    "P0000085",     # Jefferson Eastchase
    "P0000089",     # 45th & Main
    "P0000100",     # Green Valley Ranch & Telluride
    "P0000110",     # Trolley Square
    "P0000114",     # Jefferson Stephens
})

#: Which columns the excluding-dev row actually populates. The PDF leaves every
#: other cell on that row blank, so the assembly emits None for them rather than
#: a number nobody asked for.
EXCLUDING_DEV_COLUMNS = ("total_commitment", "itd", "net_roe")
# ══════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════
# TEMPORARY HARDCODE — per-cell n/a, reference-PDF fidelity
# ══════════════════════════════════════════════════════════════════════════
#: vcode -> the columns the PDF prints as "n/a" for that deal.
#:
#: City West was lost to foreclosure (see KEEP_DESPITE_SOLD in
#: portfolio_snapshot_service). The PDF keeps the row but blanks Debt — the loan
#: went with the asset — and Net ROE under footnote (2) "City west is excluded
#: from ROE calculations". Its live Debt computes to 0.0, which would render as
#: "$0.0" and read as a real debt-free position rather than as not-applicable.
#:
#: Display only: the raw `debt` stays 0.0 and the raw `net_roe` stays whatever
#: the accessor returned, so no value moves and the guardrails still see them.
PDF_NA_CELLS: dict[str, frozenset] = {
    "PCITWES": frozenset({"debt", "net_roe"}),      # City West
}
# ══════════════════════════════════════════════════════════════════════════


def _na_cells(vcode: str) -> frozenset:
    return PDF_NA_CELLS.get(str(vcode or "").strip().upper(), frozenset())

#: Zone A columns summed for subtotals. % columns are recomputed from the sums
#: rather than added, since averaging percentages is meaningless.
#: ``funded_pref`` is summed but never printed — it is the denominator the
#: subtotal "% of Pref" needs; see _subtotal.
_SUM_COLS = ("debt", "total_pref", "ptr_equity", "total_cap",
             "funded_pref", "committed_pref", "invested", "total_commitment",
             "unfunded")


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
    # "% of Pref" on a subtotal row is the dollar-weighted average of the
    # per-deal look-through percentages above it, which is
    # sum(pct x funded) / sum(funded) = invested / funded_pref. The denominator
    # is FUNDED, deliberately, even though Total Pref is now committed:
    #   * it keeps the cell a weighted average of the column it sits under;
    #   * dividing by committed would move a displayed cell as a side effect of
    #     the Total Pref basis switch — TGA 2025 would read 46% against the
    #     PDF's 90%, purely because Burton's disputed commitment (54.23M vs the
    #     PDF's 26.6M) inflates that group's denominator.
    # On funded this cell is unchanged from before the switch, and TGA 2025
    # reproduces the PDF's 90% exactly.
    fp = out.get("funded_pref")
    out["pct_of_pref"] = ((out["invested"] / fp)
                          if (out.get("invested") is not None and fp) else None)
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
                       committed_debt_provider: Optional[Callable] = None,
                       manual_loader: Optional[Callable] = None,
                       footnote_loader: Optional[Callable] = None,
                       commitment_basis: str = COMMITMENT_BASIS) -> dict:
    """Build the Financial subtab for one investor and quarter."""
    # resolve_strategy, not entry["investment_strategy"] directly: that raw field
    # is 0/110 populated, so reading it made `is_dev` False on EVERY row of this
    # subtab while Operating and Loan saw ten dev deals. resolve_strategy is the
    # documented single source (Investment_Strategy, falling back to the
    # Lifecycle proxy) precisely so the three subtabs cannot disagree.
    from flask_app.services.portfolio_snapshot_operating import (
        is_dev_deal, resolve_strategy,
    )
    from flask_app.services.portfolio_snapshot_service import (
        group_total_label, PORTFOLIO_TOTAL_LABEL, resolve_committed_pref,
    )
    from flask_app.services.portfolio_snapshot_debt import (
        BASIS_ISBS, resolve_debt,
    )

    # committed_debt_provider(vcode) -> committed facility, or None.
    #
    # Injected rather than read here, same shape as one_pager_provider: this
    # module takes DataFrames from nobody and has no access to the loans frame.
    # build_subtab in portfolio_snapshot_freeze wires it from data["mri_loans_raw"].
    # Absent, every deal falls back to the ISBS basis — which is the pre-2026-08-25
    # behaviour, so an un-wired caller degrades to the old numbers rather than
    # blanking the column. The self-test relies on that.
    committed_of = committed_debt_provider or (lambda vcode: None)

    manual = (manual_loader or _load_manual)(investor_code, quarter) or {}
    footnotes = (footnote_loader or _load_footnotes)(investor_code, quarter) or []

    diag = {"deals": 0, "dev": 0, "provider_errors": 0,
            "pct_unavailable": 0, "commitment_missing": 0,
            "manual_pending": 0, "manual_entered": 0,
            "pdf_na_cells": 0, "excluding_dev_deals": 0,
            "debt_rebased_dev": 0}

    def build_row(entry: dict, extra_flags: Optional[list] = None) -> dict:
        vcode = entry["vcode"]
        flags = list(extra_flags or [])
        strat, strat_source = resolve_strategy(entry)
        dev = is_dev_deal(strat)
        if dev:
            diag["dev"] += 1
        na = _na_cells(vcode)
        if na:
            diag["pdf_na_cells"] += 1
            flags.append("TEMPORARY — " + ", ".join(sorted(na))
                         + " shown as n/a to match the reference PDF "
                           "(see PDF_NA_CELLS)")
        if entry.get("kept_despite_sold"):
            flags.append("kept on the report despite being sold/foreclosed "
                         "before quarter end")

        try:
            payload = one_pager_provider(vcode, quarter) or {}
        except Exception as exc:
            diag["provider_errors"] += 1
            flags.append(f"One Pager unavailable: {str(exc)[:80]}")
            payload = {}
        cap = payload.get("cap_stack") or {}

        # ---- Zone A: deal-level, unscaled ----
        #
        # Debt goes through the shared resolver, so this subtab and the Loan
        # subtab print the same figure for the same deal. Before this, Financial
        # took cap_stack['debt'] unconditionally while Loan used the committed
        # facility for dev deals, and JB Fair Park read 66.36 here against 77.37
        # there. Only DEV deals move; all 23 operating deals keep the ISBS
        # balance, which already ties to the PDF.
        # The pre-override ISBS balance — see portfolio_snapshot_debt.resolve_debt.
        # Diagnostic only; the printed figure comes from resolve_debt below.
        debt_isbs = _num(cap.get("debt_isbs", cap.get("debt")))
        debt_orig = None
        try:
            debt_orig = committed_of(vcode)
        except Exception as exc:                       # provider must not break a row
            flags.append(f"committed facility unavailable: {str(exc)[:60]}")
        debt, debt_basis = resolve_debt(cap, dev, debt_orig)
        if dev and debt_basis != BASIS_ISBS:
            diag["debt_rebased_dev"] += 1
            if (debt_isbs is not None and debt is not None
                    and abs(debt - debt_isbs) > 1):
                flags.append(
                    f"Debt on the committed facility ({debt / 1e6:,.2f}M), not "
                    f"the ISBS balance ({debt_isbs / 1e6:,.2f}M) — dev deal, "
                    f"PDF footnote (6)")
        # total_cap is NOT recomputed from `debt` — see resolve_debt's docstring.
        # A rebased dev deal therefore does not foot exactly, as on the PDF.
        funded_pref = _num(cap.get("pref_equity"))
        committed_pref = _num(cap.get("committed_pe"))
        ptr_equity = _num(cap.get("partner_equity"))

        # Total Pref = the COMMITTED pref tranche. See the module docstring for
        # the three lines of evidence. `committed_pe` already carries a funded
        # fallback for the two deals with no commitment row (East Manchester,
        # City West) — one_pager.get_capitalization_stack, so page 1 gets it too.
        # The `or funded_pref` here is belt-and-braces for a payload built before
        # that fallback existed (a frozen snapshot, a stale worker); it can only
        # raise a zero, never lower a real pledge.
        if commitment_basis == "committed_pe":
            total_pref, pref_basis = resolve_committed_pref(cap)
            if pref_basis == "funded (no commitment row)":
                flags.append(
                    "no commitment row — Total Pref falls back to funded pref "
                    f"({(funded_pref or 0) / 1e6:,.2f}M)")
        else:
            total_pref, pref_basis = funded_pref, "funded"

        # Total Cap is RE-FOOTED so Debt + Total Pref + Ptr Equity add up to it
        # on screen. Without this it would stay `total_cap_isbs`, which is built
        # from FUNDED pref, and the four printed columns would visibly not sum.
        #
        # The debt leg stays `debt_isbs` — the ISBS/current basis — NOT the
        # footnote-6 dev rebase in `debt` above. The One Pager rebases a dev
        # deal's own Total Cap onto hard costs and this column must not follow
        # it; a rebased dev deal therefore still does not foot exactly, as on
        # the PDF. Measured at 26Q1: this formula ties the PDF's Total Cap on
        # 21/33 deals, the same as the shipped `total_cap_isbs` (21/33). Using
        # the resolved footnote-6 debt instead would tie 22/33, but that moves
        # the Debt basis of Total Cap, which is out of scope here.
        # Explicit None test, not truthiness: a genuinely zero cap stack is data,
        # and must not silently fall through to the funded-basis total.
        if None in (debt_isbs, total_pref, ptr_equity):
            total_cap = _num(cap.get("total_cap_isbs", cap.get("total_cap")))
        else:
            total_cap = debt_isbs + total_pref + ptr_equity
        # The pre-switch value, so the change stays auditable on the payload.
        total_cap_funded_basis = _num(
            cap.get("total_cap_isbs", cap.get("total_cap")))

        # ---- Zone B: the four scaled columns ----
        #
        # THE PE BASIS, not the deal-level look-through. Every dollar scaled
        # here comes from `pref_equity` or `committed_pe`, which
        # `one_pager.get_capitalization_stack` builds from non-OP investors
        # only — an operating partner's capital is routed to `partner_equity`.
        # Scaling by the whole-deal `lookthrough_pct` would remove the OP stake
        # a second time (Pegasus: $2,144,757.65 low). `lookthrough_pct_pe`
        # re-normalises the final hop against the non-OP owners and is equal to
        # `lookthrough_pct` wherever the OP sits at 0%, which is 34 of 35 TGAM
        # deals at 26Q2. Page 1's `funded` reads the same field, so the two
        # subtabs still cannot drift — see portfolio_snapshot_summary's
        # docstring and the Invested identity in its self-test.
        pct = entry.get("lookthrough_pct_pe")
        pct_deal_level = entry.get("lookthrough_pct")
        if pct is None:
            diag["pct_unavailable"] += 1
            flags.append("% of Pref unavailable — ownership chain unresolved")
        if (pct is not None and pct_deal_level is not None
                and abs(pct - pct_deal_level) > 1e-12):
            diag["op_diluted"] = diag.get("op_diluted", 0) + 1
            flags.append(
                f"operating partner holds a real ownership %: % of Pref is the "
                f"PE basis {pct * 100:.4f}%, not the deal-level look-through "
                f"{pct_deal_level * 100:.4f}%")

        # Invested scales FUNDED, always — TIAA's actually-contributed slice.
        # It reads `funded_pref` explicitly rather than `total_pref` so it cannot
        # follow the Total Pref basis switch: it ties the PDF 33/33 on funded and
        # that identity is what proves the Total Pref column is committed.
        invested = (pct * funded_pref) if (pct is not None
                                           and funded_pref is not None) else None
        # Total Commitment scales the same figure Total Pref prints, so
        # `Total Commitment = pct x Total Pref` — the PDF identity — holds, and
        # `Un-funded = pct x (Total Pref - funded)` falls out of it.
        basis_val = (funded_pref if commitment_basis == "funded"
                     else total_pref)
        if commitment_basis == "committed_pe" and not committed_pref:
            # Not a missing figure any more — the funded fallback covers it —
            # but still counted so the two no-commitment-row deals stay visible.
            diag["commitment_missing"] += 1
        total_commitment = (pct * basis_val) if (pct is not None
                                                and basis_val is not None) else None
        unfunded = ((total_commitment - invested)
                    if (total_commitment is not None and invested is not None)
                    else None)
        # Both bases carried so the choice is auditable and reversible.
        commitment_funded = (pct * funded_pref) if (pct is not None
                                                   and funded_pref is not None) else None
        commitment_committed = (pct * committed_pref) if (
            pct is not None and committed_pref is not None) else None

        # ---- Zone C: manual entry, read only through the two accessors ----
        row_manual = {}
        for f, accessor in (("net_roe", get_net_roe), ("itd", get_itd)):
            v = accessor(vcode, investor_code, quarter, manual)
            row_manual[f] = v
            # n/a outranks "pending entry": the PDF states this cell does not
            # apply, so prompting an analyst to fill it would be wrong.
            row_manual[f + "_display"] = (
                NA_LABEL if f in na else (PENDING if v is None else v))
            row_manual[f + "_source"] = "manual entry (formula TBD)"
            if f in na:
                pass                       # not pending — it does not apply
            elif v is None:
                diag["manual_pending"] += 1
            else:
                diag["manual_entered"] += 1

        diag["deals"] += 1
        return {
            "vcode": vcode, "name": entry["name"],
            "investment_strategy": strat, "strategy_source": strat_source,
            "is_dev": dev,
            "kept_despite_sold": bool(entry.get("kept_despite_sold")),
            "pdf_na_cells": sorted(na),
            # Zone A — raw values, unsuppressed
            "debt": debt,
            # Both candidates carried so the discarded one stays auditable and
            # the basis is never a guess from the number's size.
            "debt_basis": debt_basis,
            "debt_isbs": debt_isbs,
            "debt_orig": debt_orig,
            "total_pref": total_pref, "ptr_equity": ptr_equity,
            "total_cap": total_cap, "committed_pref": committed_pref,
            # Both pref bases carried, same principle as the two debt bases:
            # `funded_pref` is what Invested scales and what the subtotal
            # "% of Pref" divides by, and `pref_basis` says which source
            # `total_pref` actually came from ("funded (no commitment row)" on
            # East Manchester and City West).
            "funded_pref": funded_pref,
            "pref_basis": pref_basis,
            "total_cap_funded_basis": total_cap_funded_basis,
            "debt_pct": _num(cap.get("debt_pct")),
            "pref_pct": _num(cap.get("pref_equity_pct")),
            "ptr_pct": _num(cap.get("partner_equity_pct")),
            "pe_exposure_on_cap": _num(cap.get("pe_exposure_on_cap")),
            # Zone B (scaled)
            "pct_of_pref": pct,
            "invested": invested,
            "total_commitment": total_commitment,
            "unfunded": unfunded,
            # Display twin for the one Zone A column the PDF ever blanks.
            "debt_display": NA_LABEL if "debt" in na else debt,
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
        # The PDF labels a group only on its total row ("Total PSC TGA 2022
        # LLC"), so the mapped label lives on the subtotal. `group` is kept
        # alongside as the stable key the UI iterates and persistence uses.
        groups[group] = {"deals": rows, "group": group,
                         "label": group_total_label(group),
                         "subtotal": _subtotal(rows, group_total_label(group))}
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
    total_rows = all_rows + flagged_rows
    total = _subtotal(total_rows, PORTFOLIO_TOTAL_LABEL)

    # ---- "Excluding Development Deals", the row under Portfolio Totals ----
    #
    # Restored 2026-08-25: it IS on the reference PDF (page 2), directly beneath
    # Portfolio Totals, populating three columns and leaving the rest blank. The
    # earlier decision to keep it off this subtab is reversed; the self-test
    # assertion that it was absent is replaced by checks that it is present and
    # correctly scoped.
    #
    # Population is EXCLUDING_DEV_VCODES, not `is_dev` — see that constant.
    kept = [r for r in total_rows if r["vcode"] not in EXCLUDING_DEV_VCODES]
    removed = [r for r in total_rows if r["vcode"] in EXCLUDING_DEV_VCODES]
    diag["excluding_dev_deals"] = len(removed)
    ex_full = _subtotal(kept, "Excluding Development Deals")
    # Blank every column the PDF leaves blank on this row. Emitting the full
    # subtotal would put eight more numbers on screen that the published page
    # does not show, and a reader would reasonably take them as sourced.
    total_excluding_dev = {
        "label": "Excluding Development Deals",
        "deal_count": ex_full["deal_count"],
        "excluded_count": len(removed),
        "excluded_vcodes": sorted(r["vcode"] for r in removed),
        "excluded_names": sorted(r["name"] for r in removed),
        "populated_columns": list(EXCLUDING_DEV_COLUMNS),
        "manual_entered": ex_full["manual_entered"],
        **{c: (ex_full.get(c) if c in EXCLUDING_DEV_COLUMNS else None)
           for c in _SUM_COLS},
        # Net ROE over the reduced population, and ITD. Both are manual
        # (formula TBD) with no portfolio-level entry, so they are None and
        # render as "pending entry" — never the PDF's figure copied across,
        # which would present a published number as a computed one.
        "net_roe": None,
        "itd": None,
        "net_roe_display": PENDING,
        "itd_display": PENDING,
        "basis": ("Total Commitment recomputed over the "
                  f"{ex_full['deal_count']} non-development deals; "
                  "ITD and Net ROE are manual (formula TBD) and carry no "
                  "portfolio-level entry"),
    }

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
        "total_excluding_dev": total_excluding_dev,
        "footnotes": footnotes,
        "diagnostics": diag,
    }


# ── Self-test ─────────────────────────────────────────────────────────────

# PDF 26Q1, Financial page.
#
# Giant 7 is the primary Zone B case: it is one of the six deals where the PDF
# itself prints a non-zero Un-funded, so it discriminates funded from committed,
# and on the committed basis all five figures tie the published page.
#
# Nottingham is kept as the documented DISPUTE. Its commitment row (12,058,427)
# is real — 9,135,000 funded plus a 2,923,426.79 contribution dated 2026-06-01
# sums to it to the cent — but the PDF prints 9.1 and blanks Un-funded, i.e. its
# source treats the deal as fully funded. That is the IA_Contribution vs
# IA_Commitment question with Alay, not a basis error, so the self-test asserts
# what the code SHOULD produce and reports the PDF delta rather than failing.
_PDF = {
    "P0000019": {"name": "Giant 7", "pct_of_pref": 57.0,
                 "invested_m": 11.5, "commitment_m": 11.9, "unfunded_m": 0.5,
                 "total_pref_m": 21.0, "disputed": False},
    "P0000030": {"name": "Nottingham Village", "pct_of_pref": 41.0,
                 "invested_m": 3.8, "commitment_m": 3.8, "unfunded_m": 0.0,
                 "total_pref_m": 9.1, "disputed": True,
                 # what the committed basis correctly produces
                 "expect_total_pref_m": 12.058427,
                 "expect_commitment_m": 4.969564,
                 "expect_unfunded_m": 1.204814},
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
    print("ZONE B — the four scaled TIAA columns vs PDF")
    for vc, p in _PDF.items():
        r_ = flat.get(vc) or {}
        tag = "  (DISPUTED commitment value — see _PDF)" if p["disputed"] else ""
        print(f"\n  {vc} {p['name']}{tag}")
        print(f"  {'metric':<24}{'computed':>16}{'expected':>12}"
              f"{'delta':>12}{'PDF':>8}   verdict")
        print("  " + "-" * 114)
        tests = [
            ("% of Pref", (r_.get("pct_of_pref") or 0) * 100,
             p["pct_of_pref"], 0.5),
            ("Invested ($M)", (r_.get("invested") or 0) / 1e6,
             p["invested_m"], 0.06),
            ("Total Commitment ($M)", (r_.get("total_commitment") or 0) / 1e6,
             p.get("expect_commitment_m", p["commitment_m"]), 0.06),
            ("Un-funded ($M)", (r_.get("unfunded") or 0) / 1e6,
             p.get("expect_unfunded_m", p["unfunded_m"]), 0.06),
            ("Total Pref ($M)", (r_.get("total_pref") or 0) / 1e6,
             p.get("expect_total_pref_m", p["total_pref_m"]), 0.06),
        ]
        pdf_col = {"% of Pref": p["pct_of_pref"],
                   "Invested ($M)": p["invested_m"],
                   "Total Commitment ($M)": p["commitment_m"],
                   "Un-funded ($M)": p["unfunded_m"],
                   "Total Pref ($M)": p["total_pref_m"]}
        for metric, comp, exp, tol in tests:
            d = comp - exp
            ok = abs(d) <= tol
            print(f"  {metric:<24}{comp:>16.4f}{exp:>12.4f}{d:>+12.4f}"
                  f"{pdf_col[metric]:>8.2f}   {'ok' if ok else 'MISMATCH'}")
            checks.append((f"{p['name']} {metric}", ok))

    r = flat.get("P0000030") or {}
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
    chk("Zone A comes from the One Pager cap stack "
        "(Nottingham funded pref 9,135,000)",
        abs((flat.get("P0000030") or {}).get("funded_pref", 0) - 9135000) < 1)
    chk("Total Pref is the COMMITTED tranche "
        "(Nottingham committed 12,058,427, not funded 9,135,000)",
        abs((flat.get("P0000030") or {}).get("total_pref", 0) - 12058427) < 1)

    # ---- the committed-basis invariants (2026-08-31) ----
    chk("Total Pref == committed_pref wherever a commitment row exists",
        all(abs(x["total_pref"] - x["committed_pref"]) < 1
            for x in flat.values()
            if x.get("committed_pref") and x.get("total_pref") is not None))
    fb_rows = [x for x in flat.values()
               if x.get("pref_basis") == "funded (no commitment row)"]
    chk("the funded fallback fires on exactly the 2 no-commitment-row deals",
        {x["vcode"] for x in fb_rows} == {"P0000017", "PCITWES"})
    chk("fallback deals print funded pref, never 0",
        all(x["total_pref"] == x["funded_pref"] and x["total_pref"] > 0
            for x in fb_rows))
    chk("East Manchester Total Pref = 3,600,000 via the fallback",
        abs((flat.get("P0000017") or {}).get("total_pref", 0) - 3_600_000) < 1)
    chk("City West Total Pref = 5,925,000 via the fallback",
        abs((flat.get("PCITWES") or {}).get("total_pref", 0) - 5_925_000) < 1)
    chk("Invested scales FUNDED, not Total Pref, on every scaled deal",
        all(abs(x["invested"] - x["pct_of_pref"] * x["funded_pref"]) < 1e-6
            for x in flat.values()
            if x.get("invested") is not None
            and x.get("pct_of_pref") is not None
            and x.get("funded_pref") is not None))
    chk("Total Commitment = % of Pref x Total Pref (the PDF identity)",
        all(abs(x["total_commitment"] - x["pct_of_pref"] * x["total_pref"]) < 1e-6
            for x in flat.values()
            if x.get("total_commitment") is not None
            and x.get("pct_of_pref") is not None
            and x.get("total_pref") is not None))
    chk("Total Cap re-foots to debt_isbs + Total Pref + Ptr Equity",
        all(abs(x["total_cap"] - (x["debt_isbs"] + x["total_pref"]
                                  + x["ptr_equity"])) < 1
            for x in flat.values()
            if None not in (x.get("total_cap"), x.get("debt_isbs"),
                            x.get("total_pref"), x.get("ptr_equity"))))
    chk("Ptr Equity is untouched — still cap_stack.partner_equity (funded)",
        all(abs(x["ptr_equity"]
                - (cache[(x["vcode"], Q)].get("cap_stack") or {})
                .get("partner_equity", 0)) < 1
            for x in flat.values() if x.get("ptr_equity") is not None))
    chk("subtotal '% of Pref' divides FUNDED pref, so it did not move",
        all(b["subtotal"]["pct_of_pref"] is None
            or abs(b["subtotal"]["pct_of_pref"]
                   - b["subtotal"]["invested"] / b["subtotal"]["funded_pref"]) < 1e-9
            for b in out["groups"].values()))
    chk("Un-funded = Commitment - Invested for every deal",
        all(x["unfunded"] is None
            or abs(x["unfunded"] - (x["total_commitment"] - x["invested"])) < 1e-6
            for x in flat.values()
            if x["total_commitment"] is not None and x["invested"] is not None))
    # PDF_NA_CELLS deals are excepted: n/a outranks "pending entry".
    chk("Net ROE pending for all deals (none entered)",
        all(x["net_roe_display"] == PENDING for x in flat.values()
            if "net_roe" not in x["pdf_na_cells"]))
    chk("ITD pending for all deals (none entered)",
        all(x["itd_display"] == PENDING for x in flat.values()
            if "itd" not in x["pdf_na_cells"]))
    chk("manual columns are None underneath, not 0",
        all(x["net_roe"] is None and x["itd"] is None for x in flat.values()))
    # Property, not identity — see the note in portfolio_snapshot_service.
    chk("every ownership-flagged deal withholds Zone B",
        all(x["pct_of_pref"] is None and x["invested"] is None
            for x in out["ownership_flagged"]))
    chk("2 footnotes carried, numbered 1..2",
        [f_["number"] for f_ in out["footnotes"]] == [1, 2])
    chk("subtotals sum their deals (Individual Investments invested)",
        abs((out["groups"].get("Individual Investments", {})
             .get("subtotal", {}).get("invested") or 0)
            - sum(x["invested"] or 0 for x in
                  out["groups"].get("Individual Investments", {}).get("deals", []))) < 1)

    # ---- restored 2026-08-25: the row IS on the PDF (page 2) ----
    ex = out.get("total_excluding_dev") or {}
    chk("Excluding-Development total present",
        bool(ex) and ex.get("label") == "Excluding Development Deals")
    chk("excluding-dev removes exactly the EXCLUDING_DEV_VCODES present",
        set(ex.get("excluded_vcodes") or [])
        == (EXCLUDING_DEV_VCODES & set(flat)))
    chk("excluding-dev deal_count = all deals minus those removed",
        ex.get("deal_count") == len(flat) - len(ex.get("excluded_vcodes") or []))
    chk("excluding-dev Total Commitment = sum over the kept deals",
        abs((ex.get("total_commitment") or 0)
            - sum(x["total_commitment"] or 0 for vc, x in flat.items()
                  if vc not in EXCLUDING_DEV_VCODES)) < 1)
    chk("excluding-dev Total Commitment < the portfolio total",
        (ex.get("total_commitment") or 0) < (out["total"]["total_commitment"] or 0))
    chk("excluding-dev blanks every column the PDF leaves blank",
        all(ex.get(c) is None for c in _SUM_COLS
            if c not in EXCLUDING_DEV_COLUMNS))
    chk("excluding-dev never fabricates ITD / Net ROE",
        ex.get("itd") is None and ex.get("net_roe") is None)
    chk("group totals carry the PDF's labels",
        [b["subtotal"]["label"] for b in out["groups"].values()][:2]
        == ["Total Individual Investments", "Total PSC TGA 2022 LLC"])
    chk("portfolio total is labelled 'Portfolio Totals'",
        out["total"]["label"] == "Portfolio Totals")
    chk("is_dev is populated via the Lifecycle proxy, not the empty raw field",
        sum(1 for x in flat.values() if x["is_dev"]) > 0)
    chk("City West kept, with Debt and Net ROE as n/a",
        (flat.get("PCITWES") or {}).get("debt_display") == NA_LABEL
        and (flat.get("PCITWES") or {}).get("net_roe_display") == NA_LABEL)
    chk("City West raw Debt untouched underneath the n/a",
        not isinstance((flat.get("PCITWES") or {}).get("debt"), str))
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
