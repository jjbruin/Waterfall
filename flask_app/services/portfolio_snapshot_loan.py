"""Portfolio Snapshot — Subtab 4 (Loan) data assembly (Step 4a).

Backend only: no route, no blueprint, no UI. Nothing imports this module.

Reads, never writes:
  * Step 1  ``portfolio_snapshot_service.resolve_investor_deals`` — deals by fund
  * Step 3a ``portfolio_snapshot_operating.is_dev_deal`` — the single definition
            of "development", shared so the debt path and the Debt Yield
            suppression can never disagree
  * One Pager ``cap_stack`` (ISBS debt) and ``property_performance`` (YTD DSCR,
            YTD NOI), plus ``_child_vcodes_for_parent`` for portfolio parents
  * Step 2  ``portfolio_snapshot_persistence`` — the per-deal loan comment
  * ``valuations`` and ``loans`` frames, injected

All Loan-subtab metrics are property-level and are **never** scaled by
ownership. Only the four TIAA columns on Subtab 2 get scaled.

Definitions as verified against live on 2026-08-24:

Debt
    Operating deals: the One Pager's ISBS-derived balance
    (``cap_stack['debt']``, ISBS Interim BS as of quarter end). This is the
    fresh source that reproduced the PDF.
    Development deals: ``mOrigLoanAmt`` summed over the deal's loans — the full
    committed facility. A dev deal's ISBS balance is unreliable: JB Fair Park
    carries a single 2022-12-31 "SENIOR FINANCING" row of 66,363,992 that the
    app keeps alive because an active MRI loan exists, while the balance sheet
    balances without it (assets ~10.9M vs equity ~10.8M) and accounts
    5190/7030/7060 have no rows at all, so nothing has ever been serviced.

LTV
    Debt / most recent ``mIncomeCapConcludedValue`` by ``dtValuation``.
    Every valuation in the table is a 12/31 and there are no 2026 rows, so this
    currently always lands on 2025-12-31 — matching the PDF's "LTV ('25 Vals)".
    Above ``LTV_REVIEW_CEILING`` the number is withheld and flagged rather than
    printed: JB Fair Park computes 457.7% purely from the stale debt above.

Debt Yield
    (single-quarter Interim IS NOI x 4) / Debt — a quarterly run-rate
    annualisation, computed correctly for **any** quarter.

    The source Excel intends single-period NOI x 4, but its SUMIFS on
    ``Interim IS`` at the quarter-end date actually returns the **YTD cumulative**
    balance (verified live 2026-08-24: Camp Creek YTD@26Q1 1,770,634 equals
    Jan+Feb+Mar exactly, and YTD@26Q2 3,342,478 equals Jan..Jun). At Q1 those
    coincide, so x4 is right; at Q2/Q3/Q4 it over-counts by 2x / 3x / 4x. This
    implementation takes the true single-quarter periodic NOI instead, so it
    reproduces the PDF at Q1 and stays correct afterwards.

    Note the two annualisation bases are NOT interchangeable beyond Q1:
        single-quarter x 4   -> latest-quarter run-rate  (implemented here)
        YTD / months * 12    -> average-to-date annualised
    Both equal ytd x 4 at Q1. ``debt_yield_ytd_annualised`` carries the second
    form alongside for comparison.

    Rendered as "Dev" for development deals — see DEV_DISPLAY — and as "N/A"
    for a debt-free deal, where there is no denominator at all.

Rate / Maturity
    Computed here, never read as a literal string. The deal's own loans are used
    when it has any; a portfolio parent with none falls back to its children's.
    One loan, or several that share rate + maturity + interest type, renders the
    actual terms; two or more with differing terms renders "Various".

    REAL for development deals — the "Dev" literal is confined to the three
    ratio columns, because a construction facility has a rate and a maturity
    even before the asset stabilises.

DISPLAY PRECEDENCE (2026-09-01), highest first. Every rule below touches the
``*_display`` twins only; the raw fields are always the computed figures, so
the subtotals and the guardrails see the truth and a frozen payload re-renders
under whatever the rule is at read time.

  1. Debt free (DEBT_FREE_DEALS) — Debt an em dash, and LTV / YTD DSCR /
     Debt Yield / Rate / Maturity the literal "N/A". Says "this asset carries
     no debt", which is neither "no data" nor "Dev".
  2. Development (config.DEV_STRATEGIES) — "Dev" on LTV, YTD DSCR and Debt
     Yield, for EVERY development deal with no exemption of any kind. Debt,
     Rate and Maturity stay real.
  3. A TYPED cell (MANUAL_RATIO_SEEDS, 2026-09-02) — the six recent
     acquisitions whose ratios cannot be computed from the data on record
     carry a typeable figure instead, pre-filled and editable, on the same
     footing as the Financial subtab's Net ROE and ITD. The computed twin
     survives as ``*_computed``.

     THE SUBTOTALS WEIGHT THE TYPED FIGURE (changed 2026-09-02, same day):
     they aggregate what the row DISPLAYS, so a fund total can be re-derived
     from the rows printed above it. See ``aggregation_value`` — including why
     the unit conversion there is load-bearing, and why a cleared cell falls
     out of the total instead of reverting to its computed twin.
  4. Otherwise the computed value, or an em dash where it could not be
     computed.
"""

from __future__ import annotations

import logging
from datetime import date as _date
from typing import Callable, Optional

import pandas as pd

from flask_app.services.portfolio_snapshot_debt import (
    BASIS_COMMITTED, BASIS_UNAVAILABLE, committed_facility, deal_loan_rows,
    resolve_debt,
)

log = logging.getLogger(__name__)

#: Above this, an LTV is treated as evidence of stale data rather than a real
#: number. 150% is well clear of a genuinely over-levered asset while catching
#: the stale-debt cases (JB Fair Park 457.7%, Trolley Square 366.4%).
LTV_REVIEW_CEILING = 1.50

#: Retired as a displayed value Sep 2 2026 — a multi-loan deal now lists its
#: loans (see _loan_terms). Kept because the self-test and the guardrails name
#: it when asserting that nothing renders it any more.
VARIOUS = "Various"

#: What separates one loan's terms from the next in a single cell. Spaces
#: around the bar so the column can wrap between loans instead of forcing the
#: table wider.
TERM_SEP = " | "

#: vIntType values that mean the loan is priced off an index rather than a
#: single all-in rate. Only 'Variable' occurs on live today (19 of 90 rows);
#: the synonyms are here so a relabelling upstream does not silently drop a
#: loan back to showing its number. See rate_of() for why these take
#: precedence over nRate.
FLOATING_INT_TYPES = {"variable", "floating", "adjustable"}

#: Development deals have no stabilised operations, so the three ratio columns
#: render this literal instead of a number — matching the PDF. Debt, Rate and
#: Maturity still display normally for them.
#:
#: UNCONDITIONAL as of 2026-09-01. It previously yielded to a ``dev_no_data``
#: test — a dev deal with no debt basis at all rendered n/a instead of "Dev" —
#: which existed for exactly one deal, Pegasus Life Storage, and only because
#: that deal was miscoded development. With "new construction" out of
#: config.DEV_STRATEGIES, Pegasus is operating and no dev deal is left in that
#: state (verified live at 26Q1: all 9 carry a debt basis, ``dev_no_data`` is
#: empty). The test is gone rather than left dormant: a dev deal's three ratio
#: columns are unreportable because the asset is pre-stabilisation, which is
#: true whether or not a facility is on record, so letting a missing balance
#: turn "Dev" into a bare dash lost information. ``dev_no_data`` survives on
#: the payload as a diagnostic only — see ``assemble_loan``.
#:
#: NO EXEMPTIONS as of 2026-09-01. The Waters Creek LTV exception was retired
#: on instruction — see the retirement note below — so this literal now applies
#: to every development deal in all three columns.
DEV_DISPLAY = "Dev"
DEV_RATIO_COLUMNS = ("ltv", "ytd_dscr", "debt_yield")

# ══════════════════════════════════════════════════════════════════════════
# TEMPORARY HARDCODED EXCEPTION 1 of 2 — DEBT-FREE DEALS
# Remove when the real rule lands. See also exception 2 (Waters Creek LTV).
# ══════════════════════════════════════════════════════════════════════════
#: Deals held with NO DEBT, whose loan row reports that fact rather than
#: computing ratios from a zero balance.
#:
#: Rendered as: Debt an em dash, and LTV / YTD DSCR / Debt Yield / Rate /
#: Maturity the literal ``NA_DISPLAY``. This is a POSITIVE statement — "this
#: asset carries no debt, so these columns do not apply" — which is why it is a
#: literal and not the bare dash that means "no data". It is also emphatically
#: NOT "Dev": these are operating assets, and conflating the two would undo the
#: classification fix that made this entry necessary.
#:
#: P0000066 Pegasus Life Storage is the case: ISBS balance a real 0.0, no loan
#: record, no mOrigLoanAmt, and the reference PDF prints n/a across its ratio
#: columns. It used to reach that display by accident — misclassified
#: development, so ``resolve_debt`` took the dev branch, found no committed
#: facility and returned (None, unavailable), which the old ``dev_no_data``
#: test turned into dashes. Correcting the classification puts it on the ISBS
#: branch, where the balance is a real 0.0 and Debt would print "$0.0" — a
#: measured zero where the page means not-applicable.
#:
#: WHY THIS IS KEYED BY VCODE AND NOT DERIVED. The obvious data rule —
#: non-dev, no loan record, debt 0-or-None — is not specific enough: measured
#: live at 26Q1 it also catches PCITWES City West (identical fingerprint: 0
#: loans, ISBS 0.0, no facility), a foreclosed deal whose blank columns mean
#: "this deal is gone", not "this deal is unlevered". The two need different
#: words on the page and the data cannot currently tell them apart. The rule
#: this stands in for is:
#:
#:      a deal shows N/A across its debt columns when it is held
#:      UNLEVERED BY DESIGN, as distinct from having no debt on record
#:
#: which needs a capital-structure intent field, or a lifecycle value that
#: separates "unlevered" from "disposed", neither of which is extracted today.
#: Anything added here in the meantime is technical debt.
NA_DISPLAY = "N/A"
DEBT_FREE_DEALS = {"P0000066"}                  # Pegasus Life Storage


def _debt_free(vcode: str) -> bool:
    """True for a deal held with no debt, whose debt columns read N/A.
    TEMPORARY — see DEBT_FREE_DEALS."""
    return str(vcode or "").strip().upper() in DEBT_FREE_DEALS
# ══════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════
# TEMPORARY HARDCODED EXCEPTION 2 of 2 — MANUAL RATIO CELLS
# Remove the seeds when the fallback formulas land in the data.
# ══════════════════════════════════════════════════════════════════════════
#: Deals whose LTV / YTD DSCR / Debt Yield are TYPED, not computed, with the
#: figure each cell starts life holding.
#:
#: WHY THESE SIX AND WHY NOW. All six are recent acquisitions, and the three
#: ratios cannot be computed for them from what the source systems hold today:
#:
#:   LTV        every one of the six lacks a valuation dated on or before the
#:              report's year-end, so `_latest_valuation` returns nothing and
#:              the computed cell is empty (the 2026 acquisitions have no
#:              12/31 valuation yet; Burton and Plaza Del Mar have none at all).
#:   YTD DSCR   needs a One Pager `dscr.ytd_actual`, which needs a full YTD
#:              Interim IS plus a balance-sheet principal movement.
#:   Debt Yield needs a COMPLETE quarter of actual NOI; `aggregate_periodic`
#:              drops a quarter missing any of its three months.
#:
#: Verified on live 26Q2 before this existed: P0000117/118/120 had all three
#: empty, P0000119 had DSCR only, and Burton/Plaza Del Mar had DSCR and Debt
#: Yield but no LTV. Hence the per-deal field lists below — Burton and Plaza
#: Del Mar take LTV alone and KEEP their computed DSCR and Debt Yield, and
#: P0000119's typed 1.1x deliberately replaces a computed 3.8x on the page.
#:
#: THIS IS A HARDCODE AND IT IS A SYMPTOM REPAIR, in the sense CLAUDE.md means:
#: vcodes in a constant, standing in for data that is not there. It is
#: deliberately shaped so removing it is one deletion — take a deal out of this
#: dict and its cells go back to whatever the engine computes, with no other
#: code change — and so nothing it touches can leak:
#:
#:   * the raw computed `ltv` / `ytd_dscr` / `debt_yield` are LEFT ALONE, so the
#:     fund subtotals, the portfolio total, every guardrail and any frozen
#:     payload keep reading the computed truth. A typed cell contributes to no
#:     aggregate, exactly as the Financial subtab's Net ROE and ITD do not.
#:   * `*_computed` carries the figure the engine produced, beside the typed
#:     one, so a cell can always be read back against its own arithmetic.
#:   * a deal absent from this dict is byte-identical to before.
#:
#: SEEDS ARE A DEFAULT, NOT A VALUE. The seed shows until somebody types over
#: it; after that the stored entry wins, including a deliberately CLEARED cell
#: (a stored NULL renders the em dash and does NOT spring back to the seed).
#: See `resolve_manual_ratio`.
#:
#: Units follow the house rule for every manual cell — stored in the unit the
#: column DISPLAYS, nothing converted in or out (see `format_manual` in
#: portfolio_snapshot_financial and `fmtPctPts` in the Vue formatters):
#:
#:      ltv          PERCENTAGE POINTS   69.0  -> "69.0%"
#:      debt_yield   PERCENTAGE POINTS   12.1  -> "12.1%"
#:      ytd_dscr     COVERAGE MULTIPLE    1.9  -> "1.9x"
#:
#: which is why the seeds below are 69.0 and not 0.69. The computed twins are
#: decimals for the two percentages, so they are NOT interchangeable with these
#: and must never be summed together — the reason a typed cell stays out of the
#: aggregates rather than being written into the raw field.
MANUAL_RATIO_FIELDS = ("ltv", "ytd_dscr", "debt_yield")

MANUAL_RATIO_SEEDS: dict[str, dict] = {
    "P0000109": {"ltv": 69.0},                                    # Burton Retail Portfolio
    "P0000116": {"ltv": 64.2},                                    # Plaza Del Mar
    "P0000117": {"ltv": 69.7, "ytd_dscr": 1.9, "debt_yield": 12.1},   # Fairview Heights
    "P0000118": {"ltv": 75.7, "ytd_dscr": 1.5, "debt_yield": 8.9},    # Hanestowne Waterstone
    "P0000119": {"ltv": 70.6, "ytd_dscr": 1.1, "debt_yield": 5.93},   # Presidential Arms
    "P0000120": {"ltv": 74.0, "ytd_dscr": 1.5, "debt_yield": 9.7},    # Citizen Storage
}

#: The two source strings a typed cell reports, so the page can always say
#: whether a figure is somebody's entry or the value it was pre-filled with.
MANUAL_SOURCE_ENTERED = "manual entry"
MANUAL_SOURCE_SEED = "manual entry (pre-filled — editable)"


def manual_ratio_fields(vcode: str) -> tuple:
    """The ratio columns that are TYPED for one deal — empty for every other."""
    return tuple(f for f in MANUAL_RATIO_FIELDS
                 if f in MANUAL_RATIO_SEEDS.get(
                     str(vcode or "").strip().upper(), {}))


def format_manual_ratio(field: str, value) -> Optional[str]:
    """How a typed ratio reads, unit included — the ONE place that rule lives.

    Mirrors ``portfolio_snapshot_financial.format_manual``: the unit belongs to
    the value, so the same cell reads "69.0%" / "1.9x" on screen, in print, and
    in the string ``PUT /value`` hands back after a save.

    None returns None, not a "pending" literal: unlike Net ROE and ITD — which
    are pending until an analyst supplies them — every cell here starts with a
    seed, so the only way to reach None is to clear one deliberately. That is a
    dash, the same as any other absent figure.

    No magnitude heuristic, for the reason given on the seeds: 0.9 is a
    legitimate 0.9% and a legitimate 0.9x.

    ONE decimal, the precision the reference document prints and the precision
    the computed cells beside these use (``fmtPct``/``fmtX`` in the Vue
    formatters both default to 1) — EXCEPT where a second decimal is carrying
    information, which is a property of the number and not a guess about it:
    5.93 renders "5.93%" because rounding it to 5.9% would silently drop a
    digit somebody typed, while 69.0 stays "69.0%" rather than becoming "69%".
    """
    if value is None:
        return None
    if field == "ytd_dscr":
        return f"{value:{_manual_fmt(value)}}x"
    if field in ("ltv", "debt_yield"):
        return f"{value:{_manual_fmt(value)}}%"
    return str(value)


def _manual_fmt(value: float) -> str:
    """``.1f``, or ``.2f`` when the hundredths digit is not just rounding."""
    return ".2f" if round(value, 1) != round(value, 2) else ".1f"


def resolve_manual_ratio(field: str, vcode: str, stored: Optional[dict]):
    """``(value, source, entered)`` for one typed cell.

    ``stored`` is this deal's ``{field: value}`` map out of Step 2 persistence.
    MEMBERSHIP is what decides, not truthiness: a field present with a NULL
    value is a cell somebody emptied on purpose, and restoring the seed under
    it would overwrite an edit with a default.
    """
    vc = str(vcode or "").strip().upper()
    if field not in MANUAL_RATIO_SEEDS.get(vc, {}):
        return None, None, False
    if stored is not None and field in stored:
        return stored[field], MANUAL_SOURCE_ENTERED, True
    return MANUAL_RATIO_SEEDS[vc][field], MANUAL_SOURCE_SEED, False


def _default_manual_loader(investor_code: str, quarter: str) -> dict:
    """{vcode: {field: value}} of typed ratios from Step 2 (read-only).

    Same shape and same read as ``portfolio_snapshot_financial._load_manual``;
    filtered to this subtab's fields so a Net ROE entry can never land in a
    ratio cell. Failure degrades to "nothing stored", which shows the seeds —
    never a blank page.
    """
    out: dict = {}
    try:
        from flask_app.services.portfolio_snapshot_persistence import get_elements
        for r in get_elements("value", investor_code, quarter):
            field = r.get("field")
            if field in MANUAL_RATIO_FIELDS:
                out.setdefault(
                    str(r.get("deal_vcode") or "").strip().upper(),
                    {})[field] = r.get("value")
    except Exception as exc:
        log.debug("manual ratio values unavailable: %s", exc)
    return out
# ══════════════════════════════════════════════════════════════════════════

# ── RETIRED 2026-09-01: the Waters Creek LTV exception ────────────────────
#
# `WATERS_CREEK_LTV_EXCEPTION` / `_ltv_exception()` used to un-suppress the LTV
# column for P0000078 Jefferson Waters Creek, the one development deal that had
# received a true income-based valuation on entering lease-up. Its computed
# 57.51% tied the PDF's 57.4% (+0.11pp), and that tie also confirmed the
# numerator: the dev debt basis (mOrigLoanAmt 51,667,000, the committed
# facility) is right, where the ISBS drawn balance (48,416,160) would have
# given 53.89% and missed.
#
# REMOVED ON INSTRUCTION: every genuine development deal now shows "Dev" for
# LTV, YTD DSCR and Debt Yield, with no per-deal exemption. The rule is now
# uniform and there is no hardcoded deal reference left in the dev path.
#
# TWO CONSEQUENCES, both deliberate and neither hidden:
#
#   * Waters Creek's LTV cell goes from 57.5% to "Dev".
#   * The TGA 2022 fund LTV subtotal no longer reproduces the published 60.4%.
#     That figure only came out with Waters Creek's 57.5% weighted in; without
#     it the debt-weighted mean over the remaining four members is 61.6%. This
#     is recorded in KNOWN_LOAN_SUBTOTAL_DIFFS rather than worked around,
#     because the alternative — keeping a value out of the display but inside
#     the subtotal — would make the total unauditable from the rows above it.
#
# The rule this stood in for is unchanged and still unimplemented: a
# development deal could show a real LTV when its `mIncomeCapConcludedValue`
# reflects a genuine income-based valuation rather than a cost-basis or
# as-stabilised placeholder. That needs a valuation-method column the
# `valuations` table does not currently expose. If it ever lands, it belongs in
# the data, not here.


def _num(v):
    if v is None:
        return None
    try:
        f = float(str(v).replace(",", "").replace("$", ""))
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f


# ── Subtotals ─────────────────────────────────────────────────────────────
#
# REVERSES this subtab's "ratios are not summed" position. The reference PDF
# page 4 carries a total row per fund and a portfolio total, each with Debt, LTV,
# YTD DSCR and Debt Yield, so they are needed; computed HERE, not in the
# component, so a freeze captures them.
#
# THE AGGREGATION, derived from the PDF's own five fund rows rather than assumed:
#
#   Debt   SUMMED over EVERY deal, development included. Ties all five funds to
#          the cent — 270.5 / 268.4 / 366.5 / 279.2 / 201.5.
#
#   Ratios DEBT-WEIGHTED over the deals that CARRY A DISPLAYED VALUE — see
#          ``aggregation_value``. Carrying-a-value needs no dev concept at all,
#          so the subtotal cannot drift from the "Dev" display: whatever the
#          display rule suppresses is, by the same act, out of the weighting.
#
#          "Displayed" rather than "computed" since 2026-09-02, when the three
#          ratio columns became typeable on six recent acquisitions. A typed
#          entry is weighted like any other displayed figure, converted into
#          the computed unit first. The invariant is the point: every total on
#          this page can be re-derived from the rows printed above it, which is
#          the same standard the retired Waters Creek exception was held to
#          (see KNOWN_LOAN_SUBTOTAL_DIFFS, TGA 2022 LTV).
#
#          LTV tied all five funds exactly — 67.4 / 60.4 / 59.4 / 62.1 / 69.1 —
#          while the Waters Creek LTV exception was live. Simple averaging
#          missed every one (63.9 / 57.1 / 58.5 / 61.6 / 69.1), which settles
#          weighted-vs-simple decisively and is unaffected by the retirement.
#          DSCR ties TGA23 and TGA25 and is one rounding step out on TGA22
#          (1.68 against 1.6) and TGA24 (1.57 against 1.5); simple averaging is
#          0.3 out on TGA24, so weighted is still the better fit.
#
#   The PDF's footnote — "Summary level performance metrics (LTV, DSCR, and Debt
#   Yield) exclude the development deals" — describes the effect rather than the
#   mechanism: a dev deal renders "Dev" and so carries no value to weight, which
#   excludes it automatically. Since the Waters Creek exception was retired on
#   2026-09-01 the footnote and the arithmetic finally say the same thing — at
#   the cost of the TGA 2022 LTV tie, below.
#
# THREE PUBLISHED FIGURES DO NOT REPRODUCE; recorded in
# KNOWN_LOAN_SUBTOTAL_DIFFS rather than fitted to.
_RATIO_KEYS = ("ltv", "ytd_dscr", "debt_yield")

#: Fund total cells on PDF page 4 that no consistent rule reproduces.
#: label -> (metric, ours, published, why it is being left alone)
KNOWN_LOAN_SUBTOTAL_DIFFS = {
    ("Total PSC TGA 2022 LLC", "ltv"): (
        "PDF 60.4% against 61.6%. This one is a KNOWN COST, not a mystery: the "
        "published figure reproduced exactly while Jefferson Waters Creek's "
        "real 57.5% LTV was weighted in, and that exception was retired on "
        "2026-09-01 so that every development deal shows 'Dev' with no "
        "per-deal carve-out. Waters Creek's debt (51,667,000 of the fund's "
        "243.3M) leaving the denominator moves the mean from 60.4% to 61.6%. "
        "Restoring the tie means restoring the exception — the two cannot both "
        "hold. Deliberately NOT fixed by weighting a value the rows above no "
        "longer display, which would make the total unauditable from them."),
    ("Total Individual Investments", "ytd_dscr"): (
        "PDF prints n/a although three members carry a DSCR "
        "(Nottingham 1.1x, Evergreen 2.9x, Ascent 2.1x, weighting to 2.08x). "
        "No denominator produces n/a from those."),
    ("Total Individual Investments", "debt_yield"): (
        "PDF 4.1% against 8.71% debt-weighted over the deals with a value. "
        "Weighting over ALL the fund's debt gives 3.76%, nearer but still not "
        "4.1%, and that same denominator breaks TGA 2022 (4.58% against a "
        "published 9.6%)."),
    ("Total PSC TGA 2025 LLC", "debt_yield"): (
        "PDF 4.9% against 13.10%, which is simply Burton's own figure since it "
        "is the only member carrying one. 4.9% IS Burton weighted over the "
        "fund's whole debt (13.1 x 75.3 / 201.5) — but that denominator gives "
        "4.58% on TGA 2022 where 9.6% is published, so the two funds disagree "
        "about the rule and neither can be adopted without breaking the other. "
        "NOTE the 13.10% figure is this harness's, computed from the PDF's own "
        "member rows; on LIVE data the same total now also weights Hanestowne's "
        "typed 8.9% (26Q1: 12.3%, 26Q2: 11.4% with Citizen Storage as well) — "
        "see aggregation_value. The published-vs-ours question is unchanged by "
        "that; the denominator is still the disagreement."),
}


#: The ratio columns a TYPED entry stores in PERCENTAGE POINTS, where the
#: computed twin is a decimal. Nothing else needs converting — a typed DSCR and
#: a computed DSCR are both plain multiples.
_MANUAL_PCT_KEYS = ("ltv", "debt_yield")


def aggregation_value(row: dict, key: str):
    """What a subtotal weights for one deal and one ratio column.

    THE RULE: a subtotal aggregates the figure the row DISPLAYS — the typed
    entry where there is one, the computed figure otherwise — so a fund total
    can always be re-derived from the rows printed above it. Before 2026-09-02
    this read the computed field alone, which left the TGA 2025 and TGA 6 LTV
    totals blank while their members displayed typed LTVs, and left TGA 6's
    DSCR reading 3.81x out of Presidential Arms' computed figure on a page
    whose Presidential Arms row says 1.1x.

    THE UNIT CONVERSION IS THE WHOLE DIFFICULTY, and is why this is not the
    one-line change it looks like. A typed cell stores the unit its column
    DISPLAYS (69.0 for 69%, per format_manual_ratio), while the computed twin
    is a decimal (0.6202…). Measured on live 26Q2: weighting Burton's typed
    69.0 beside Seasons at Bel Air's computed 0.6203 without converting gives
    those two deals a mean LTV of 3,201%. Typed percentages are therefore
    divided by 100 to enter the computed unit; DSCR passes through.

    THREE CASES CARRY NO VALUE and are skipped, so they neither move a total
    nor silently put a figure in one that the page does not show:

      * a computed cell that could not be computed (None) — unchanged;
      * a CLEARED typed cell (stored NULL, renders an em dash). The computed
        figure is deliberately NOT used as a fallback here: the analyst emptied
        the cell, and weighting a number the row no longer prints is exactly
        what made these totals unauditable;
      * a cell suppressed to a literal — "Dev" for a development deal, "N/A"
        for a debt-free one. Development stays out of the summary metrics, as
        PDF page 4's footnote states, and it stays out by the same mechanism as
        before: no displayed value, no weight. Seeding a dev deal could not
        sneak one in.
    """
    if not row.get(f"{key}_is_manual"):
        return row.get(key)
    if row.get(f"{key}_display") in (DEV_DISPLAY, NA_DISPLAY):
        return None
    v = row.get(f"{key}_manual")
    if v is None:
        return None
    return (v / 100.0) if key in _MANUAL_PCT_KEYS else v


def _debt_weighted(rows: list, key: str):
    """Debt-weighted mean of one ratio over the rows carrying a value.

    "Carrying a value" is ``aggregation_value``, i.e. what the row displays.
    """
    num = den = 0.0
    for r in rows:
        v, d = aggregation_value(r, key), r.get("debt")
        if v is None or not d:
            continue
        num += v * d
        den += d
    return (num / den) if den else None


def loan_subtotal(rows: list, label: str) -> dict:
    """One total row over ``rows`` — a fund's deals, or every deal."""
    debts = [r.get("debt") for r in rows if r.get("debt") is not None]
    out = {
        "label": label,
        "deal_count": len(rows),
        "dev_count": sum(1 for r in rows if r.get("is_dev")),
        "debt": sum(debts) if debts else None,
        "debt_basis": "sum over every deal, development included",
        "ratio_basis": ("debt-weighted over the deals carrying a DISPLAYED "
                        "value — a typed entry where there is one, the "
                        "computed figure otherwise"),
    }
    for k in _RATIO_KEYS:
        out[k] = _debt_weighted(rows, k)
        # `_n` counts what was actually weighted, so it moves with the rule
        # rather than describing the retired one; `_typed_n` says how much of
        # that came from a typed cell, which is what makes a moved total
        # explainable without reading the code.
        out[f"{k}_n"] = sum(1 for r in rows
                            if aggregation_value(r, k) is not None
                            and r.get("debt"))
        out[f"{k}_typed_n"] = sum(1 for r in rows
                                  if r.get(f"{k}_is_manual")
                                  and aggregation_value(r, k) is not None
                                  and r.get("debt"))
    return out


def _s(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return str(v).strip()


def months_elapsed(quarter: str) -> int:
    """Months of the calendar year covered by a YTD figure at this quarter end."""
    try:
        return int(str(quarter).split("Q")[1]) * 3
    except (IndexError, ValueError):
        return 12


# ── Loans ─────────────────────────────────────────────────────────────────

def _deal_loans(loans: pd.DataFrame, vcode: str,
                child_vcodes: Optional[list] = None) -> pd.DataFrame:
    """The loan rows that belong to a deal, falling back to its children.

    Delegates to portfolio_snapshot_debt so the Financial subtab, which needs
    the same rows to size a committed facility, cannot end up selecting them by
    a slightly different rule. Kept as a local name because the terms code below
    also uses it.
    """
    return deal_loan_rows(loans, vcode, child_vcodes)


def _loan_terms(rows: pd.DataFrame) -> dict:
    """Rate / maturity / type for a deal, or "Various" when they differ.

    'Various' is derived by comparing the rows, never read from the data.
    """
    out = {"rate": None, "maturity": None, "interest_type": None,
           "loan_count": 0, "various": False, "rate_display": None,
           "maturity_display": None}
    if rows is None or rows.empty:
        return out

    def maturity_of(r):
        for c in ("dtMaturity", "dtEvent"):
            if c in r.index:
                t = pd.to_datetime(r.get(c), errors="coerce")
                if pd.notna(t):
                    return t.date()
        return None

    def rate_of(r):
        """(numeric rate or None, display string).

        A variable loan carries no nRate — its pricing is vIndex + vSpread, the
        same split the One Pager's _format_loan_str uses. Reading only nRate
        left every floating-rate deal with a blank Rate (Jefferson Addison
        Heights, Eastchase, Stephens, Waters Creek and one of Brainerd's two).
        """
        # A floating loan is priced by its index and spread, so that form wins
        # over any all-in nRate sitting beside it. nRate used to be checked
        # first, which meant a floating loan carrying both showed the number
        # and its index was discarded even though it was right there in the
        # row -- Plaza Del Mar read "7.7%" against a PDF that says "SOFR + 400".
        #
        # BOTH index and spread are required. A floating loan missing either
        # falls through to the number rather than rendering a partial "SOFR",
        # and a loan whose vIntType is not floating is never touched -- Trolley
        # Square is 'Fixed' with a stray vIndex and no spread, and must keep
        # reading "6.3% fixed" as the PDF has it.
        itype = _s(r.get("vIntType")).lower()
        idx_f, spr_f = _s(r.get("vIndex")), _num(r.get("vSpread"))
        if itype in FLOATING_INT_TYPES and idx_f and spr_f is not None:
            bps = (spr_f * 10000) if spr_f < 1 else (spr_f * 100)
            return None, f"{idx_f} + {bps:.0f}"

        rate = _num(r.get("nRate"))
        if rate is not None and rate >= 1:
            rate = rate / 100.0
        # ONE decimal on a fixed rate, and a spread in BASIS POINTS — the units
        # the reference document prints ("3.9% fixed", "SOFR + 350"). Two
        # decimals put "5.29%" and "SOFR + 3.50%" in that column, and the
        # spread-as-percent form was the last 2-decimal figure left in the
        # printed report.
        #
        # NOTE this rounds a real hundredth away on screen too — 5.29% now reads
        # 5.3%. That is the published precision; revert these two f-strings if
        # the extra digit is wanted back.
        if rate is not None:
            # "3.9% fixed", the reference document's form. Whitelisted on
            # 'fixed' rather than appending whatever vIntType holds: the column
            # carries 'Non-Interest Bearing' on Post Commons, which would print
            # "6.3% non-interest bearing" where the PDF reads "6.3% fixed", and
            # 'Variable' on a loan that reached this branch only because it has
            # an all-in rate -- the PDF prints those as INDEX + spread, never as
            # "7.0% variable". Anything else keeps the bare number, which is
            # what the six rows with no vIntType at all already render.
            if _s(r.get("vIntType")).lower() == "fixed":
                return rate, f"{rate * 100:.1f}% fixed"
            return rate, f"{rate * 100:.1f}%"
        idx = _s(r.get("vIndex"))
        spr = _num(r.get("vSpread"))
        if idx and spr is not None:
            spr_bps = (spr * 10000) if spr < 1 else (spr * 100)
            return None, f"{idx} + {spr_bps:.0f}"
        if idx:
            return None, idx
        return None, None

    def fmt_mat(m):
        return None if m is None else f"{m.month}/{m.day}/{m.year}"

    terms = []
    for _, r in rows.iterrows():
        rate, rate_disp = rate_of(r)
        terms.append({"rate": rate, "maturity": maturity_of(r),
                      "interest_type": _s(r.get("vIntType")).lower(),
                      "rate_display": rate_disp,
                      "amount": _num(r.get("mOrigLoanAmt"))})

    out["loan_count"] = len(terms)

    # Two loans on identical terms are one line, not the same value printed
    # twice. Keyed on what the reader sees.
    seen, uniq = set(), []
    for t in terms:
        k = (t["rate_display"], t["maturity"], t["interest_type"])
        if k not in seen:
            seen.add(k)
            uniq.append(t)

    if len(uniq) == 1:
        t = uniq[0]
        out.update(rate=t["rate"], maturity=t["maturity"],
                   interest_type=t["interest_type"] or None,
                   rate_display=t["rate_display"],
                   maturity_display=fmt_mat(t["maturity"]))
        return out

    # ── More than one loan: LIST THEM, do not collapse to "Various" ────────
    #
    # "Various" told the reader a deal had several loans and then withheld all
    # of them. Each loan's actual terms are printed instead, joined by " | ",
    # LARGEST FACILITY FIRST so Rate and Maturity are in the same order and
    # position n of one pairs with position n of the other.
    #
    # A field whose values are all the SAME stays a single value rather than
    # being repeated once per loan: three of the four multi-loan deals at 26Q1
    # share one maturity across both facilities, and "8/18/2027 | 8/18/2027"
    # doubles the column width to say what one date already says. The pairing
    # still reads correctly — a single value applies to every loan listed.
    #
    # ``various`` stays True: it means "this deal has more than one facility",
    # which is still the case and is what the callers test.
    uniq.sort(key=lambda t: (-(t["amount"] or 0.0),
                             t["maturity"] or _date.max,
                             t["rate_display"] or ""))
    out["various"] = True
    out["terms_list"] = [
        {"rate_display": t["rate_display"],
         "maturity_display": fmt_mat(t["maturity"]),
         "amount": t["amount"], "interest_type": t["interest_type"] or None}
        for t in uniq]

    rate_descs = [t["rate_display"] for t in uniq]
    mats = [t["maturity"] for t in uniq]
    types = [t["interest_type"] for t in uniq]

    if len(set(rate_descs)) == 1:
        out["rate"] = uniq[0]["rate"]
        out["rate_display"] = rate_descs[0]
    else:
        # A loan with no readable pricing prints an em dash in its slot rather
        # than closing up, so the positions still line up with Maturity.
        out["rate_display"] = TERM_SEP.join(d or "—" for d in rate_descs)

    if len(set(mats)) == 1:
        out["maturity"] = mats[0]
        out["maturity_display"] = fmt_mat(mats[0])
    else:
        out["maturity_display"] = TERM_SEP.join(
            fmt_mat(m) or "—" for m in mats)

    out["interest_type"] = (types[0] if len(set(types)) == 1
                            else TERM_SEP.join(t or "—" for t in types))
    return out


# ── Valuations ────────────────────────────────────────────────────────────

def valuation_year_end(quarter: Optional[str]) -> Optional[pd.Timestamp]:
    """The fiscal year-end a report of this quarter values as-of.

    THE RULE: the most recent December 31 that is on or before the quarter's
    own end date.  Equivalently, the prior completed year-end for Q1-Q3, and
    the quarter's own date for Q4:

        2026-Q1 (ends 3/31/26)  -> 2025-12-31
        2026-Q2 (ends 6/30/26)  -> 2025-12-31
        2026-Q3 (ends 9/30/26)  -> 2025-12-31
        2026-Q4 (ends 12/31/26) -> 2026-12-31
        2027-Q1 (ends 3/31/27)  -> 2026-12-31

    Returns None for an unparseable quarter, which the caller treats as
    "no guard" so a malformed input can never blank out a whole page.
    """
    try:
        year = int(str(quarter).split("-Q")[0])
        qn = int(str(quarter).split("Q")[1])
    except (AttributeError, IndexError, ValueError):
        return None
    if not 1 <= qn <= 4:
        return None
    return pd.Timestamp(year=year if qn == 4 else year - 1, month=12, day=31)


def _latest_valuation(valuations: pd.DataFrame, vcode: str,
                      quarter: Optional[str] = None) -> dict:
    """Most recent mIncomeCapConcludedValue by dtValuation for a deal.

    YEAR-END GUARD — DO NOT REMOVE.  With ``quarter`` supplied, candidates are
    first restricted to ``dtValuation <= valuation_year_end(quarter)``, so a
    report can never value a deal off a PARTIAL current-year valuation.

    This became live-relevant in Aug 2026: MRI_VAL went download-only and the
    app became the system of record for valuations, so
    ``valuation_nav_service.publish`` now writes rows with
    ``dtValuation = the record's as_of_date`` -- which can be ANY date, not
    just a 12/31.  Without the guard, a valuation published as-of 2026-06-30
    would immediately outrank the 2025-12-31 row on a 26Q1 report and silently
    restate a already-published quarter's LTV. Today the table is 12/31-only,
    so this fires on nothing; it is preventative.

    NOT a 12/31-of-2025 hardcode and NOT a "12/31 only" filter: the boundary
    moves with the report (see valuation_year_end), and a genuine 2026-12-31
    valuation is selected as soon as reporting reaches 26Q4. Any real year-end
    still qualifies.

    A deal whose ONLY row is after the boundary yields no valuation and the
    caller renders the existing "no valuation" state. That is deliberate --
    blank is honest where a partial-year figure would be wrong.

    Without ``quarter`` the behaviour is exactly as before (no guard), so the
    self-test and any other caller are unaffected.
    """
    out = {"value": None, "as_of": None}
    if valuations is None or valuations.empty:
        return out
    df = valuations.copy()
    vc_col = next((c for c in df.columns if c.lower() == "vcode"), None)
    val_col = next((c for c in df.columns
                    if c.lower() == "mincomecapconcludedvalue"), None)
    dt_col = next((c for c in df.columns if c.lower() == "dtvaluation"), None)
    if not (vc_col and val_col):
        return out
    df["_vc"] = df[vc_col].astype(str).str.strip().str.upper()
    sub = df[df["_vc"] == str(vcode).strip().upper()].copy()
    if sub.empty:
        return out
    sub["_v"] = sub[val_col].map(_num)
    if dt_col:
        # format="mixed" IS LOAD-BEARING, not tidying. Plain to_datetime infers
        # ONE format from the first element and coerces every row that does not
        # match to NaT, which dropna then deletes. Today every row is MRI's
        # "YYYY-MM-DDT00:00:00" so a single format happens to fit -- but
        # valuation_nav_service.publish writes as_of.strftime("%Y-%m-%d")
        # ("2026-06-30"), and the legacy MRI export carries "12/31/2025 0:00".
        # Either of those lands as NaT under the inferred format and the row
        # vanishes, so an app-published valuation would be INVISIBLE to LTV
        # rather than merely mis-ranked. Matches one_pager.py, which already
        # parses this same column with format="mixed".
        sub["_dt"] = pd.to_datetime(sub[dt_col], format="mixed",
                                    errors="coerce")
        sub = sub.dropna(subset=["_dt"]).sort_values("_dt", ascending=False)
        # The guard. Applied BEFORE the pick, and only when both a boundary
        # and a parsed date column exist -- an undated frame keeps its old
        # behaviour rather than losing every row.
        cutoff = valuation_year_end(quarter)
        if cutoff is not None:
            sub = sub[sub["_dt"] <= cutoff]
    sub = sub[sub["_v"].notna() & (sub["_v"] > 0)]
    if sub.empty:
        return out
    row = sub.iloc[0]
    out["value"] = float(row["_v"])
    out["as_of"] = row["_dt"].date() if dt_col and pd.notna(row.get("_dt")) else None
    return out


def _default_comment_loader(investor_code: str, quarter: str) -> dict:
    try:
        from flask_app.services.portfolio_snapshot_persistence import get_elements
        rows = get_elements("comment", investor_code, quarter,
                            scope="deal", field="loan")
        return {r.get("scope_key"): r.get("comment_text") for r in rows}
    except Exception as exc:
        log.debug("loan comments unavailable: %s", exc)
        return {}


# ── Assembly ──────────────────────────────────────────────────────────────

def assemble_loan(investor_code: str, quarter: str, *,
                  resolved: dict,
                  one_pager_provider: Callable[[str, str], dict],
                  loans: pd.DataFrame,
                  valuations: pd.DataFrame,
                  inv: Optional[pd.DataFrame] = None,
                  quarterly_noi_provider: Optional[Callable] = None,
                  comment_loader: Optional[Callable] = None,
                  manual_loader: Optional[Callable] = None,
                  ltv_ceiling: float = LTV_REVIEW_CEILING) -> dict:
    """Build the Loan subtab for one investor and quarter."""
    from flask_app.services.portfolio_snapshot_operating import (
        is_dev_deal, resolve_strategy)
    from flask_app.services.portfolio_snapshot_service import (
        group_total_label, PORTFOLIO_TOTAL_LABEL,
    )

    loader = comment_loader or _default_comment_loader
    comments = loader(investor_code, quarter) or {}
    # Typed ratio cells, one query for the page — see MANUAL_RATIO_SEEDS.
    manual = (manual_loader or _default_manual_loader)(
        investor_code, quarter) or {}
    n_months = months_elapsed(quarter)

    diag = {"deals": 0, "dev": 0, "debt_from_isbs": 0, "debt_from_orig": 0,
            "ltv_ok": 0, "ltv_no_valuation": 0, "ltv_flagged_review": 0,
            "ltv_dev": 0, "ltv_dev_exception": 0, "dev_no_data": 0,
            "debt_free": 0, "dscr_dev": 0,
            "dy_ok": 0, "dy_dev_suppressed": 0, "dy_no_ytd": 0,
            "various_terms": 0, "comments_attached": 0, "provider_errors": 0,
            "manual_deals": 0, "manual_cells": 0,
            "manual_entered": 0, "manual_prefilled": 0, "manual_cleared": 0}

    child_cache: dict = {}

    def children_of(vcode: str) -> list:
        if inv is None:
            return []
        if vcode not in child_cache:
            try:
                from one_pager import _child_vcodes_for_parent  # read-only reuse
                child_cache[vcode] = _child_vcodes_for_parent(vcode, inv) or []
            except Exception:
                child_cache[vcode] = []
        return child_cache[vcode]

    def build_row(vcode: str, name: str, strategy: str,
                  extra_flags: Optional[list] = None) -> dict:
        flags = list(extra_flags or [])
        dev = is_dev_deal(strategy)
        if dev:
            diag["dev"] += 1

        try:
            payload = one_pager_provider(vcode, quarter) or {}
        except Exception as exc:
            diag["provider_errors"] += 1
            flags.append(f"One Pager unavailable: {str(exc)[:80]}")
            payload = {}
        cap = payload.get("cap_stack") or {}
        perf = payload.get("property_performance") or {}

        # ---- loans, terms ----
        kids = children_of(vcode)
        lrows = _deal_loans(loans, vcode, kids)
        terms = _loan_terms(lrows)
        inherited = bool(len(lrows)) and lrows["_vc"].iloc[0] != str(vcode).upper()
        if inherited:
            flags.append(f"loan terms inherited from {len(lrows)} child loan(s)")
        if terms["various"]:
            diag["various_terms"] += 1
        if terms["loan_count"] == 0:
            flags.append("no loan record")

        orig_total = committed_facility(lrows)

        # ---- debt ----
        # The basis choice lives in portfolio_snapshot_debt.resolve_debt, which
        # the Financial subtab calls too — the two used to disagree about the
        # same deal's Debt (JB Fair Park 66.36 vs 77.37).
        # Pre-override ISBS balance — see portfolio_snapshot_debt.resolve_debt.
        isbs_debt = _num(cap.get("debt_isbs", cap.get("debt")))
        debt, debt_basis = resolve_debt(cap, dev, orig_total)
        if debt_basis == BASIS_COMMITTED:
            diag["debt_from_orig"] += 1
        elif debt_basis == BASIS_UNAVAILABLE:
            flags.append("dev deal with no mOrigLoanAmt")
        elif debt:
            diag["debt_from_isbs"] += 1
        else:
            flags.append("no ISBS debt balance")

        # ---- diagnostic only: a dev deal with an empty loan block ----------
        # No ISBS balance, no mOrigLoanAmt, no loan record. This USED TO
        # suppress the three ratio columns to a bare dash ahead of the "Dev"
        # label; it no longer does — see DEV_DISPLAY for why that test was
        # removed and how Pegasus Life Storage, its only subject, is handled
        # now. Kept on the payload because frozen snapshots carry the key and
        # the guardrails audit it, and because a dev deal reaching this state
        # is still worth surfacing.
        dev_no_data = dev and debt is None
        if dev_no_data:
            diag["dev_no_data"] += 1
            flags.append("dev deal with an empty loan block — ratios still "
                         "read 'Dev' (the asset is pre-stabilisation)")

        # ---- debt free: N/A across the debt columns, ahead of everything ---
        # A positive statement that the asset carries no debt, so no ratio
        # applies — distinct from the em dash that means "no data", and
        # distinct from "Dev". Checked here so it wins over both. See
        # DEBT_FREE_DEALS.
        debt_free = _debt_free(vcode)
        if debt_free:
            diag["debt_free"] += 1
            flags.append("held with no debt — Debt shown as a dash, and LTV / "
                         "YTD DSCR / Debt Yield / Rate / Maturity as "
                         f"{NA_DISPLAY!r}")

        # ---- LTV ----
        # Dev deals render "Dev": the ratio is meaningless pre-stabilisation and
        # their debt basis is a committed facility, not a drawn balance. The
        # >ceiling guard therefore only ever fires for NON-dev deals now, as a
        # backstop against stale data (its original catch, JB Fair Park at 457%,
        # is a dev deal and is handled here instead).
        # `quarter` scopes the valuation to the report's own year-end, so a
        # mid-year published valuation cannot restate an earlier quarter —
        # see the year-end guard in _latest_valuation.
        val = _latest_valuation(valuations, vcode, quarter)
        ltv = ltv_flag = None
        if dev:
            diag["ltv_dev"] += 1
        elif debt and val["value"]:
            raw = debt / val["value"]
            if raw > ltv_ceiling:
                ltv_flag = (f"review — LTV {raw * 100:.0f}% exceeds "
                            f"{ltv_ceiling * 100:.0f}%, likely stale debt")
                flags.append(ltv_flag)
                diag["ltv_flagged_review"] += 1
            else:
                ltv = raw
                diag["ltv_ok"] += 1
        elif not val["value"]:
            flags.append("no valuation — LTV unavailable")
            diag["ltv_no_valuation"] += 1

        # ---- YTD DSCR (One Pager, read-only) ----
        dscr_ytd = None if dev else _num((perf.get("dscr") or {}).get("ytd_actual"))
        if dev:
            diag["dscr_dev"] += 1

        # ---- Debt Yield: single-quarter NOI x 4 (see module docstring) ----
        ytd_noi = _num((perf.get("noi") or {}).get("ytd_actual"))
        q_noi = None if quarterly_noi_provider is None else _num(
            quarterly_noi_provider(vcode, quarter))
        annualised = dy = dy_ytd = None
        if dev:
            flags.append("ratios shown as 'Dev' — development deal")
            diag["dy_dev_suppressed"] += 1
        elif debt_free:
            pass                        # already flagged, and N/A not n/a
        elif not debt:
            flags.append("Debt Yield n/a — no debt balance")
        elif q_noi is None:
            flags.append("Debt Yield n/a — no complete quarter of actual NOI")
            diag["dy_no_ytd"] += 1
        else:
            annualised = q_noi * 4
            dy = annualised / debt
            diag["dy_ok"] += 1
            if ytd_noi:
                # comparison basis only; equals `dy` at Q1
                dy_ytd = (ytd_noi / n_months * 12) / debt

        comment = comments.get(vcode)
        if comment:
            diag["comments_attached"] += 1
        diag["deals"] += 1

        # ---- displays ------------------------------------------------------
        # Every raw field below is left EXACTLY as computed; only the *_display
        # twins carry a suppression. That keeps the subtotals, the guardrails
        # and a frozen payload reading the true figures, and lets the display
        # rule change later without moving a number. A typed ratio cell obeys
        # the same rule — see the overlay under this dict.
        #
        # Precedence, highest first:
        #   1. debt free  -> N/A on five columns, dash on Debt
        #   2. dev        -> "Dev" on the three ratio columns, UNCONDITIONALLY
        #   3. a TYPED cell (MANUAL_RATIO_SEEDS) -> the entry, or its seed
        #   4. the computed value
        row = {
            "vcode": vcode, "name": name, "strategy": strategy, "is_dev": dev,
            "debt": debt, "debt_basis": debt_basis,
            # Dash for a debt-free deal, so a real 0.0 balance does not print
            # "$0.0". The raw `debt` stays 0.0 and still feeds the subtotals,
            # where it contributes nothing either way.
            "debt_display": None if debt_free else debt,
            "debt_free": debt_free,
            "isbs_debt": isbs_debt, "orig_loan_amt": orig_total,
            "valuation": val["value"], "valuation_as_of": val["as_of"],
            "ltv": ltv, "ltv_review_flag": ltv_flag,
            "ltv_display": (NA_DISPLAY if debt_free else
                            DEV_DISPLAY if dev else ltv),
            # Always False since the Waters Creek exception was retired. The
            # KEY survives because SnapshotLoan.vue reads it for a tooltip and
            # a star, and snapshots frozen while the exception was live carry
            # it as True — dropping the field would make those old payloads and
            # the component disagree. See the retirement note near the top.
            "ltv_dev_exception": False,
            "dev_no_data": dev_no_data,
            "ytd_dscr": dscr_ytd,
            "ytd_dscr_display": (NA_DISPLAY if debt_free else
                                 DEV_DISPLAY if dev else dscr_ytd),
            "ytd_noi": ytd_noi, "quarter_noi": q_noi,
            "annualised_noi": annualised,
            "months_elapsed": n_months, "debt_yield": dy,
            "debt_yield_display": (NA_DISPLAY if debt_free else
                                   DEV_DISPLAY if dev else dy),
            "debt_yield_ytd_annualised": dy_ytd,
            "debt_yield_basis": "single-quarter Interim IS NOI x 4 / debt",
            "loan_count": terms["loan_count"],
            "rate": terms["rate"],
            # Rate and Maturity stay REAL for a dev deal — only the three ratio
            # columns carry "Dev". A debt-free deal has no facility to describe,
            # so these two read N/A with the rest.
            "rate_display": (NA_DISPLAY if debt_free
                             else terms["rate_display"]),
            "maturity": terms["maturity"],
            "maturity_display": (NA_DISPLAY if debt_free
                                 else terms["maturity_display"]),
            "interest_type": terms["interest_type"],
            "terms_various": terms["various"],
            "loans_inherited_from_children": inherited,
            "loan_comment": comment,
            "flags": flags,
        }

        # ---- typed ratio cells (the six recent acquisitions) ---------------
        #
        # Applied AFTER the dict so it is visibly an overlay on a settled row
        # and not a fourth branch inside each metric. Three invariants, all
        # asserted in scripts/snapshot_loan_manual_cells_check.py:
        #
        #   * a deal with no seeds is untouched — no key added, nothing moved;
        #   * the raw `ltv` / `ytd_dscr` / `debt_yield` are never overwritten,
        #     so subtotals, totals and guardrails still see the computation;
        #   * the debt-free and dev literals still outrank a typed cell, so
        #     adding a seed to such a deal could not silently un-suppress it.
        typed = manual_ratio_fields(vcode)
        if typed:
            diag["manual_deals"] += 1
            stored = manual.get(str(vcode).strip().upper())
            literal = (NA_DISPLAY if debt_free else
                       DEV_DISPLAY if dev else None)
            for f in typed:
                value, source, entered = resolve_manual_ratio(f, vcode, stored)
                diag["manual_cells"] += 1
                diag["manual_entered" if entered else "manual_prefilled"] += 1
                if entered and value is None:
                    diag["manual_cleared"] += 1
                row[f"{f}_computed"] = row.get(f)
                row[f"{f}_manual"] = value
                row[f"{f}_is_manual"] = True
                row[f"{f}_source"] = source
                row[f"{f}_entered"] = entered
                # The literal wins where there is one; today none of the six is
                # dev or debt free, so this resolves to the typed figure.
                row[f"{f}_display"] = (literal if literal is not None
                                       else format_manual_ratio(f, value))
            flags.append(
                "LTV / YTD DSCR / Debt Yield typed on this deal where marked "
                "(pre-filled, editable) — the computed figures are kept as "
                "*_computed and still feed the subtotals")
        return row

    groups: dict[str, list] = {}
    for group, items in (resolved.get("groups") or {}).items():
        groups[group] = [build_row(e["vcode"], e["name"], resolve_strategy(e)[0])
                         for e in items]

    flagged_rows = []
    for f in (resolved.get("flagged") or []):
        row = build_row(f["vcode"], f["name"], resolve_strategy(f)[0],
                        extra_flags=[f"ownership {f.get('reason', 'unavailable')}"])
        row["ownership_flagged"] = True
        flagged_rows.append(row)

    return {
        "investor_code": resolved.get("investor_code", investor_code),
        "investor_name": resolved.get("investor_name", investor_code),
        "quarter": quarter, "subtab": "loan", "scaled": False,
        "months_elapsed": n_months,
        "ltv_review_ceiling": ltv_ceiling,
        "groups": groups,
        "ownership_flagged": flagged_rows,
        # Alongside `groups`, not nested in it, so every existing consumer keeps
        # reading {name: [rows]} unchanged.
        "group_labels": {g: group_total_label(g) for g in groups},
        "subtotals": {g: loan_subtotal(rows, group_total_label(g))
                      for g, rows in groups.items()},
        "total": loan_subtotal(
            [r for rows in groups.values() for r in rows] + flagged_rows,
            PORTFOLIO_TOTAL_LABEL),
        "diagnostics": diag,
    }


# ── Self-test ─────────────────────────────────────────────────────────────

# PDF 26Q1 loan page, as supplied
_PDF = {
    "P0000019": ("Giant 7", 70.9, None),
    "P0000075": ("Camp Creek", 58.0, 13.5),
    "P0000030": ("Nottingham Village", 75.0, None),
    "P0000079": ("Post Commons", None, 10.7),
}


def _selftest():                                    # pragma: no cover
    import json
    import os
    import sys
    import tempfile
    import sqlalchemy

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
    loans = pd.DataFrame(api.get("/api/data/tables/loans/rows",
                                 params={"page": 1, "page_size": 500}
                                 ).get("rows") or [])
    vals = pd.DataFrame(api.get("/api/data/tables/valuations/rows",
                                params={"page": 1, "page_size": 500}
                                ).get("rows") or [])
    print(f"loans {len(loans)} rows, valuations {len(vals)} rows")

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

    tmp = os.path.join(tempfile.mkdtemp(prefix="ps_step4a_"), "t.db")
    eng = sqlalchemy.create_engine(f"sqlite:///{tmp}")
    P._engine = lambda: eng                          # type: ignore[assignment]
    P._is_postgres = lambda: False                   # type: ignore[assignment]
    P.save_comment(INV, Q, "deal", "loan", "Fixed through 2032; no extension used.",
                   scope_key="P0000019", updated_by="selftest")
    P.save_comment(INV, Q, "deal", "loan", "SOFR cap purchased at 4.0%.",
                   scope_key="P0000075", updated_by="selftest")

    cache: dict = {}

    def provider(vc, q):
        if (vc, q) not in cache:
            cache[(vc, q)] = api.get(f"/api/financials/{vc}/one-pager",
                                     params={"quarter": q})
        return cache[(vc, q)]

    qcache: dict = {}

    def q_noi(vc, q):
        """True single-quarter periodic NOI for the report quarter.

        aggregate_periodic drops any quarter without all three months, so a
        partially reported quarter yields None and the caller flags it rather
        than annualising a stub.
        """
        if (vc, q) in qcache:
            return qcache[(vc, q)]
        yr, qn = int(q.split("-Q")[0]), int(q.split("Q")[1])
        qend = pd.Timestamp(year=yr, month=qn * 3, day=1) + pd.offsets.MonthEnd(0)
        ch = api.get(f"/api/financials/{vc}/performance-chart",
                     params={"freq": "Quarterly", "periods": 12,
                             "period_end": str(qend.date())})
        val = None
        for lbl, a in zip(ch.get("periods") or [], ch.get("actual_noi") or []):
            if lbl == f"Q{qn} {yr}" and a is not None:
                val = a
                break
        qcache[(vc, q)] = val
        return val

    out = assemble_loan(INV, Q, resolved=resolved, one_pager_provider=provider,
                        loans=loans, valuations=vals, inv=inv,
                        quarterly_noi_provider=q_noi,
                        comment_loader=lambda i, q: _default_comment_loader(i, q))

    flat = {r["vcode"]: r for rs in out["groups"].values() for r in rs}
    for r in out["ownership_flagged"]:
        flat[r["vcode"]] = r

    checks = []

    def chk(label, cond):
        checks.append((label, bool(cond)))
        print(f"    [{'PASS' if cond else 'FAIL'}] {label}")

    print("=" * 112)
    print(f"COMPUTED vs 26Q1 PDF — LTV and Debt Yield")
    print(f"{'deal':<26}{'metric':<16}{'computed':>11}{'PDF':>8}{'delta':>9}   verdict")
    print("-" * 112)
    for vc, (label, p_ltv, p_dy) in _PDF.items():
        r = flat.get(vc)
        if not r:
            chk(f"{label} present", False)
            continue
        first = True
        if p_ltv is not None:
            c = (r["ltv"] or 0) * 100
            d = c - p_ltv
            ok = abs(d) <= 1.0
            print(f"{label:<26}{'LTV %':<16}{c:>11.2f}{p_ltv:>8.1f}{d:>+9.2f}"
                  f"   {'ok' if ok else 'MISMATCH'}")
            checks.append((f"{label} LTV within 1pt", ok))
            first = False
        if p_dy is not None:
            c = (r["debt_yield"] or 0) * 100
            d = c - p_dy
            ok = abs(d) <= 0.5
            print(f"{'' if not first else label:<26}{'Debt Yield %':<16}"
                  f"{c:>11.2f}{p_dy:>8.1f}{d:>+9.2f}   {'ok' if ok else 'MISMATCH'}")
            checks.append((f"{label} Debt Yield within 0.5pt", ok))

    print("\n" + "=" * 112)
    print("LTV — all deals (PDF column is \"LTV ('25 Vals)\")")
    print(f"{'vcode':<9}{'deal':<30}{'debt':>14}{'basis':<12}{'valuation':>14}"
          f"{'as of':>12}{'LTV':>9}")
    print("-" * 112)
    for g, rs in out["groups"].items():
        for r in rs:
            basis = "orig" if r["is_dev"] else "isbs"
            print(f"{r['vcode']:<9}{r['name'][:29]:<30}"
                  f"{(r['debt'] or 0):>14,.0f}{basis:<12}"
                  f"{(r['valuation'] or 0):>14,.0f}"
                  f"{str(r['valuation_as_of'] or '-'):>12}"
                  f"{(f'{r['ltv']*100:.1f}%' if r['ltv'] else 'n/a'):>9}")

    print("\n" + "=" * 112)
    print("RATE / MATURITY — Various logic")
    for vc in ("P0000067", "P0000075", "P0000109", "P0000003", "P0000019"):
        r = flat.get(vc)
        if not r:
            continue
        print(f"  {vc} {r['name'][:32]:<34} loans={r['loan_count']:<3}"
              f"rate={str(r['rate_display']):<10}maturity={str(r['maturity_display']):<12}"
              f"various={r['terms_various']}"
              f"{'  (inherited from children)' if r['loans_inherited_from_children'] else ''}")

    print("\n" + "=" * 112)
    print('DEV DEALS — three ratio columns show "Dev"; Debt/Rate/Maturity stay numeric')
    print(f"{'vcode':<9}{'deal':<30}{'LTV':>7}{'DSCR':>7}{'DebtYld':>9}"
          f"{'Debt':>14}{'Rate':>17}{'Maturity':>12}")
    print("-" * 112)
    for r in sorted((x for x in flat.values() if x["is_dev"]),
                    key=lambda x: x["name"]):
        debt_s = f"{r['debt']:,.0f}" if r["debt"] else "n/a"
        print(f"{r['vcode']:<9}{r['name'][:29]:<30}"
              f"{str(r['ltv_display']):>7}{str(r['ytd_dscr_display']):>7}"
              f"{str(r['debt_yield_display']):>9}{debt_s:>14}"
              f"{str(r['rate_display'] or 'n/a'):>17}"
              f"{str(r['maturity_display'] or 'n/a'):>12}")

    print("\n" + "=" * 112)
    print("STRUCTURE CHECKS")
    chk("groups present", len(out["groups"]) >= 5)
    chk("no scaling on loan metrics", out["scaled"] is False)
    # Property, not identity — see the note in portfolio_snapshot_service.
    chk("ownership-flagged deals still appear with deal-level figures",
        len(out["ownership_flagged"]) == len(resolved["flagged"]))
    chk("Giant 7 loan comment attached",
        (flat.get("P0000019") or {}).get("loan_comment", "").startswith("Fixed"))
    chk("Camp Creek loan comment attached",
        (flat.get("P0000075") or {}).get("loan_comment") ==
        "SOFR cap purchased at 4.0%.")
    chk("deal with no comment yields None",
        (flat.get("P0000030") or {}).get("loan_comment") is None)
    jb = flat.get("P0000021") or {}
    # Superseded: JB Fair Park is a dev deal, so its LTV is "Dev" and the >150%
    # guard no longer fires for it. The guard is now a backstop for non-dev
    # stale data only. Asserted below as "shows 'Dev', not the 457% flag".
    chk("JB Fair Park LTV withheld (no garbage number)",
        jb.get("ltv") is None)
    chk("JB Fair Park debt uses mOrigLoanAmt (dev path)",
        jb.get("debt") == jb.get("orig_loan_amt") and jb.get("debt") is not None)
    chk("JB Fair Park debt differs from the stale ISBS figure",
        jb.get("isbs_debt") not in (None, jb.get("debt")))
    chk("every dev deal has Debt Yield n/a",
        all(r["debt_yield"] is None for r in flat.values() if r["is_dev"]))
    chk("Redevelopment is NOT dev (Crowne Plaza absent anyway)",
        all(r["strategy"].strip().lower() != "redevelopment" or not r["is_dev"]
            for r in flat.values()))
    chk("Brainerd renders Various", (flat.get("P0000067") or {}).get("terms_various") is True)
    chk("all dev deals render 'Dev' for LTV / YTD DSCR / Debt Yield, with no "
        "exemption of any kind",
        all(r["ltv_display"] == DEV_DISPLAY
            and r["ytd_dscr_display"] == DEV_DISPLAY
            and r["debt_yield_display"] == DEV_DISPLAY
            for r in flat.values() if r["is_dev"]) and
        any(r["is_dev"] for r in flat.values()))
    chk("dev deals keep REAL Rate / Maturity / Debt — the 'Dev' literal is "
        "confined to the three ratio columns",
        all(r["rate_display"] not in (DEV_DISPLAY, NA_DISPLAY)
            and r["maturity_display"] not in (DEV_DISPLAY, NA_DISPLAY)
            and not isinstance(r["debt_display"], str)
            for r in flat.values() if r["is_dev"]))

    # ---- dev suppression is now unconditional ----
    print("\n" + "=" * 108)
    print('DEV RATIO COLUMNS — "Dev" is forced, with or without a debt basis')
    print("=" * 108)
    print(f"  {'vcode':<10}{'deal':<30}{'debt':>14}{'LTV':>9}{'DSCR':>8}"
          f"{'DebtYld':>9}   dev_no_data")
    print("  " + "-" * 106)
    for r in sorted((x for x in flat.values() if x["is_dev"]),
                    key=lambda x: x["name"]):
        def d(v):
            if v is None:
                return "n/a"
            return v if isinstance(v, str) else f"{v:.4f}"
        print(f"  {r['vcode']:<10}{r['name'][:29]:<30}"
              f"{(r['debt'] if r['debt'] else 0):>14,.0f}"
              f"{d(r['ltv_display'])[:8]:>9}{d(r['ytd_dscr_display'])[:7]:>8}"
              f"{d(r['debt_yield_display'])[:8]:>9}   "
              f"{r['dev_no_data']}")

    # Pegasus Life Storage was the dev_no_data case until 2026-09-01, when
    # "new construction" left config.DEV_STRATEGIES and it became the operating
    # deal it always was. It is held DEBT FREE (ISBS 0.0, no loan record, no
    # mOrigLoanAmt), so it now takes the DEBT_FREE_DEALS path: Debt an em dash,
    # and five columns the literal "N/A". Asserted against BOTH of the other
    # two routes to a blank cell — the dev gate and a bare dash — so they can
    # never be confused for each other.
    peg = flat.get("P0000066") or {}
    print(f"    Pegasus: is_dev={peg.get('is_dev')} "
          f"debt_free={peg.get('debt_free')} debt={peg.get('debt')!r} "
          f"debt_display={peg.get('debt_display')!r} "
          f"loans={peg.get('loan_count')} ltv={peg.get('ltv_display')!r} "
          f"dscr={peg.get('ytd_dscr_display')!r} "
          f"dy={peg.get('debt_yield_display')!r} "
          f"rate={peg.get('rate_display')!r} "
          f"maturity={peg.get('maturity_display')!r} "
          f"dev_no_data={peg.get('dev_no_data')}")
    chk("Pegasus is no longer classified development",
        peg.get("is_dev") is False)
    chk("Pegasus has no debt on record (empty loan block)",
        peg.get("debt") in (None, 0) and peg.get("orig_loan_amt") is None
        and peg.get("loan_count") == 0)
    chk("Pegasus is flagged debt_free", peg.get("debt_free") is True)
    chk("Pegasus is NOT flagged dev_no_data — that diagnostic is dev-only",
        not peg.get("dev_no_data"))
    chk("Pegasus Debt renders an em dash, NOT its real 0.0 balance",
        peg.get("debt_display") is None and peg.get("debt") == 0)
    chk("Pegasus reads 'N/A' on all five debt columns",
        all(peg.get(k) == NA_DISPLAY for k in
            ("ltv_display", "ytd_dscr_display", "debt_yield_display",
             "rate_display", "maturity_display")))
    chk("Pegasus never reads 'Dev' — it is an operating, unlevered asset",
        not any(peg.get(k) == DEV_DISPLAY for k in
                ("ltv_display", "ytd_dscr_display", "debt_yield_display",
                 "rate_display", "maturity_display")))
    chk("Pegasus is the ONLY debt-free deal — City West has the same data "
        "fingerprint and must NOT be swept in (see DEBT_FREE_DEALS)",
        sorted(r["vcode"] for r in flat.values() if r.get("debt_free"))
        in ([], ["P0000066"]))
    cw = flat.get("PCITWES") or {}
    if cw:
        chk("City West keeps its em dashes and gains no N/A literal",
            not cw.get("debt_free")
            and not any(cw.get(k) == NA_DISPLAY for k in
                        ("ltv_display", "ytd_dscr_display",
                         "debt_yield_display", "rate_display",
                         "maturity_display")))

    no_data = sorted(r["vcode"] for r in flat.values() if r["dev_no_data"])
    print(f"    dev_no_data deals: {no_data}")
    chk("no deal is dev_no_data now that Pegasus is operating",
        no_data == [])
    chk("every dev deal keeps 'Dev' on DSCR and Debt Yield",
        all(r["ytd_dscr_display"] == DEV_DISPLAY
            and r["debt_yield_display"] == DEV_DISPLAY
            for r in flat.values() if r["is_dev"]))
    chk("every dev deal keeps 'Dev' on LTV — no deal excluded",
        all(r["ltv_display"] == DEV_DISPLAY for r in flat.values()
            if r["is_dev"]))
    chk("gate is scoped to dev deals — no non-dev deal is dev_no_data",
        all(not r["dev_no_data"] for r in flat.values() if not r["is_dev"]))
    # The debt-free deal is excluded BY NAME rather than by allowing any
    # mismatch: its DSCR is deliberately the N/A literal, and every other
    # operating deal must still pass its real number straight through.
    #
    # A TYPED cell is excluded the same way — by the row's own
    # `ytd_dscr_is_manual` flag, not by vcode — because its display is an
    # entered figure, which is neither the computed number nor a suppression of
    # it. The invariant this protects is unchanged for every other deal, and
    # the typed cells have their own before/after guardrail
    # (scripts/snapshot_loan_manual_cells_check.py) asserting that the raw
    # `ytd_dscr` beneath them never moved.
    chk("no ordinary operating deal lost its real DSCR to a suppression",
        all(r["ytd_dscr_display"] == r["ytd_dscr"]
            for r in flat.values()
            if not r["is_dev"] and not r.get("debt_free")
            and not r.get("ytd_dscr_is_manual")))
    chk("a typed DSCR leaves the computed figure in the raw field",
        all(r["ytd_dscr"] == r.get("ytd_dscr_computed")
            for r in flat.values() if r.get("ytd_dscr_is_manual")))
    chk("every operating deal's Debt display IS its raw debt, bar the "
        "debt-free one",
        all(r["debt_display"] == r["debt"] for r in flat.values()
            if not r.get("debt_free")))

    # ---- Waters Creek: the retired LTV exception ----
    print("\n" + "=" * 108)
    print('RETIRED LTV EXCEPTION — Jefferson Waters Creek now shows "Dev" '
          'for LTV like every other dev deal')
    print("=" * 108)
    wc = flat.get("P0000078") or {}
    print(f"    deal   : P0000078 {wc.get('name', '?')}")
    print(f"    strategy={wc.get('strategy')!r}  is_dev={wc.get('is_dev')}  "
          f"ltv_dev_exception={wc.get('ltv_dev_exception')}")
    print(f"    LTV display        : {wc.get('ltv_display')!r}"
          f"   (raw numeric {wc.get('ltv')})")
    print(f"    YTD DSCR display   : {wc.get('ytd_dscr_display')!r}")
    print(f"    Debt Yield display : {wc.get('debt_yield_display')!r}")
    print(f"    debt {wc.get('debt')!r} ({wc.get('debt_basis')})")
    print(f"    ISBS debt {wc.get('isbs_debt')!r}   "
          f"mOrigLoanAmt {wc.get('orig_loan_amt')!r}")
    print(f"    valuation {wc.get('valuation')!r} as of {wc.get('valuation_as_of')!r}")
    if wc.get("valuation") and wc.get("debt"):
        print(f"    the LTV it USED to display was "
              f"{wc['debt'] / wc['valuation']:.2%} "
              f"(committed facility / valuation) — now suppressed")

    chk("Waters Creek is classified as a dev deal", wc.get("is_dev") is True)
    chk("Waters Creek shows 'Dev' for LTV — the exception is retired",
        wc.get("ltv_display") == DEV_DISPLAY)
    chk("Waters Creek shows 'Dev' for YTD DSCR",
        wc.get("ytd_dscr_display") == DEV_DISPLAY)
    chk("Waters Creek shows 'Dev' for Debt Yield",
        wc.get("debt_yield_display") == DEV_DISPLAY)
    # The raw LTV is not computed for a dev deal at all (the `if dev` branch
    # short-circuits ahead of the arithmetic), so there is no suppressed number
    # sitting on the payload waiting to be weighted into a subtotal. This is
    # what makes the TGA22 LTV difference honest rather than hidden — see
    # KNOWN_LOAN_SUBTOTAL_DIFFS.
    chk("Waters Creek carries NO raw LTV either, so it cannot feed a subtotal",
        wc.get("ltv") is None)
    chk("NO deal is an LTV exception any more",
        not any(r["ltv_dev_exception"] for r in flat.values()))
    chk("EVERY dev deal shows 'Dev' for LTV, with no carve-out",
        all(r["ltv_display"] == DEV_DISPLAY for r in flat.values()
            if r["is_dev"]))
    chk("no dev deal carries a raw ratio value at all, so the three ratio "
        "subtotals exclude development entirely",
        all(r["ltv"] is None and r["ytd_dscr"] is None
            and r["debt_yield"] is None
            for r in flat.values() if r["is_dev"]))
    chk("the retired constant is gone from the module",
        not hasattr(sys.modules[__name__], "WATERS_CREEK_LTV_EXCEPTION"))
    chk("dev deals still show numeric Debt",
        all(r["debt"] is not None for r in flat.values()
            if r["is_dev"] and r["orig_loan_amt"]))
    chk("dev deals still show Rate/Maturity where a loan exists",
        all(r["rate_display"] is not None for r in flat.values()
            if r["is_dev"] and r["loan_count"] > 0))
    chk("JB Fair Park shows 'Dev', not the 457% flag",
        (flat.get("P0000021") or {}).get("ltv_display") == DEV_DISPLAY
        and (flat.get("P0000021") or {}).get("ltv_review_flag") is None)
    chk("LTV guard no longer fires for any dev deal (backstop only)",
        all(r["ltv_review_flag"] is None for r in flat.values() if r["is_dev"]))
    # Excludes the debt-free deal by name: its LTV is deliberately the N/A
    # literal, and every OTHER operating deal must stay numeric-or-em-dash —
    # bar a TYPED cell, whose display is the entered figure with its unit
    # ("69.0%"), excluded by the row's own flag rather than by vcode.
    chk("ordinary operating deals keep a numeric LTV display",
        all(r["ltv_display"] is None or isinstance(r["ltv_display"], float)
            for r in flat.values()
            if not r["is_dev"] and not r.get("debt_free")
            and not r.get("ltv_is_manual")))
    chk("a typed LTV leaves the computed figure in the raw field",
        all(r["ltv"] == r.get("ltv_computed")
            for r in flat.values() if r.get("ltv_is_manual")))
    chk("Debt Yield basis is single-quarter x 4",
        all(r["debt_yield_basis"].startswith("single-quarter")
            for r in flat.values()))
    chk("at Q1 single-quarter x4 equals YTD-annualised",
        all(r["debt_yield_ytd_annualised"] is None
            or abs(r["debt_yield"] - r["debt_yield_ytd_annualised"]) < 1e-9
            for r in flat.values() if r["debt_yield"] is not None))
    chk("dev deals use the committed facility, operating use ISBS",
        all((r["debt_basis"].startswith("mOrigLoanAmt") if r["is_dev"]
             else r["debt_basis"].startswith("ISBS")) for r in flat.values()
            if r["debt"] is not None))

    d = out["diagnostics"]
    print(f"\n  diagnostics: {d}")
    print("\n" + "=" * 112)
    print("ASSEMBLED STRUCTURE — two deals verbatim")
    for vc in ("P0000075", "P0000021"):
        print(f"\n{vc}:")
        print(json.dumps(flat.get(vc), indent=2, default=str)[:1300])

    print(f"\n  {sum(1 for _, c in checks if c)}/{len(checks)} checks passed")
    return 0


if __name__ == "__main__":                          # pragma: no cover
    raise SystemExit(_selftest())
