"""Portfolio Snapshot — Subtab 3 (Operating) data assembly (Step 3a).

Backend only: no route, no blueprint, no UI. Nothing imports this module.

Reads, never writes:
  * Step 1  ``portfolio_snapshot_service.resolve_investor_deals`` — deals by fund
  * One Pager ``property_performance`` — Econ Occ and the three NOI points
  * Step 2  ``portfolio_snapshot_persistence`` — the per-deal operating comment

The One Pager payload arrives through an injected ``one_pager_provider`` rather
than being fetched here. In-app that wraps ``financials_service.get_one_pager_data``
(deliberately **without** ``full_data``, so no waterfall is run per deal — 30+
waterfalls per page load is the performance trap flagged in the build spec); the
self-test wraps the REST endpoint. Same dependency-injection shape as Step 1.

Operating metrics are property-level and are **never** scaled by ownership. Only
the four TIAA columns on Subtab 2 get scaled.

DEVELOPMENT DEALS — every metric column reads "n/a" (``NA_LABEL``), because
operating history does not exist before stabilisation and a growth ratio taken
off a near-zero At Close NOI is noise, not information. The comments column is
kept: construction and lease-up status is what a dev row reports. Suppression
lands on the ``*_display`` fields only, so no computed value moves; see
``DEV_SUPPRESSED_COLUMNS`` and the TEMPORARY ``DEV_DISPLAY_EXCEPTIONS``.

INSUFFICIENT OPERATING HISTORY — a deal owned for less than one full quarter
also reads n/a in every metric column, for the same reason a dev deal does and
by the same mechanism. See ``INSUFFICIENT_HISTORY_MONTHS``.

The UI must render the ``*_display`` twins, never the raw fields — a metric
formatted straight from ``noi`` or ``expected_growth`` silently opts out of the
suppression rule. That is the same contract ``format.ts`` states from the other
side: the UI does not recompute or second-guess a backend display decision.

SUBTOTALS AGGREGATE WHAT IS VISIBLE. Every total on this page sums the
``*_display`` twins, so a cell reading n/a contributes nothing to the row below
it. This is not a stylistic choice — it is the only way the page can be read.
The alternative shipped first and was wrong: subtotals summed the raw fields,
so ten development rows showed n/a while their real NOI went on feeding the
total underneath, and Portfolio U/W YE printed 123.9M over visible cells that
summed to about 104M. A total that disagrees with the column above it is the
first thing a reader checks and the last thing they trust. The reference PDF
excludes what it withholds, and 104.6M is what it prints.

Because the totals read the display twins rather than re-deriving the rules,
every suppression — dev, the dev exceptions, insufficient history, and whatever
lands next — is honoured by construction. There is no second copy of the rule
here to fall out of step with ``shown()``.

UNITS — the two percentage fields on this subtab are on DIFFERENT scales, and
that is deliberate rather than an oversight:

  * ``econ_occ`` / ``econ_occ_display`` are **percentage points** (92.23 = 92.23%).
    They are copied verbatim out of the One Pager ``property_performance``
    payload, where ``one_pager.get_property_performance`` scales every branch of
    ``economic_occ`` to 0-100 (``uw_ye`` x100, ``at_close`` x100-if-ratio,
    ``ytd_actual`` = avg ``Occ%`` points - bad-debt points, ``actual_ye`` a
    weighted average of points). Rewriting them to ratios here would silently
    diverge from the One Pager this mirrors, and would break every already-frozen
    payload, which stores this dict as-is and renders through the same component.
  * ``expected_growth`` / ``actual_growth`` are **decimal ratios** (0.053 = 5.3%),
    like ``ltv`` and ``debt_yield`` on the Loan subtab.

So the UI must format them with different formatters — ``fmtPctPts`` and
``fmtPct`` respectively. Feeding occupancy through the ratio formatter is what
rendered Flats at Dorsett Ridge as "9223.0%".

Growth definitions (per the build spec):
    Expected Growth = (U/W YE NOI      - At Close NOI) / At Close NOI
    Actual Growth   = (Projected YE NOI - At Close NOI) / At Close NOI
where Projected YE is the One Pager's ``actual_ye`` (YTD actual + remainder-of-
year budget). Note that this makes Actual Growth a *moving* figure for a past
quarter: it shifts as actuals land, so the same historical quarter recomputed
later will not match a PDF produced earlier. Flagged, not hidden.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Callable, Optional

# The dev-deal definition moved to config.py so one_pager.py (core, no flask on
# its import path) can share it. Re-exported here under the same names, so
# portfolio_snapshot_financial and portfolio_snapshot_loan import unchanged.
from config import DEV_STRATEGIES, is_dev_deal  # noqa: F401

log = logging.getLogger(__name__)

#: Values of **deals.Investment_Strategy** that mark a development deal.
#:
#: Creator decision 2026-08-24: dev detection reads Investment_Strategy ONLY --
#: no fallback to Lifecycle, no "has numbers" hybrid. One field drives the "Dev"
#: display (LTV / YTD DSCR / Debt Yield), the mOrigLoanAmt debt path, and the
#: "Excluding Development Deals" subtotal, so those three can never disagree.
#:
#: MEASUREMENT NOTE (live build 09fe220ae0da, 2026-08-24): deals.Investment_Strategy
#: is 0/110 populated. No MRI query selects it and it is absent from
#: mri_service.MRI_COLUMNS, so nothing populates it on refresh. Until it is fed,
#: this rule classifies every deal as operating. Lifecycle (97/110) carries the
#: values that look like strategies -- Development 24, Value-Add 31, Income 22,
#: Stable 16, New Construction 2, Redevelopment 1, Lease up 1 -- but is
#: deliberately NOT consulted here.
#:
#: The set itself now lives in config.DEV_STRATEGIES and is imported at the top
#: of this module, so one_pager.py's cap-stack debt basis classifies a deal the
#: same way this page does. The commentary above still applies to it.

#: What a suppressed metric renders as. The PDF prints the literal "n/a" in
#: every metric cell of a development row, so the backend hands that string
#: through and the UI prints it verbatim.
#:
#: Deliberately a LITERAL and not None: None already means "no data" and renders
#: as an em dash. "Suppressed because the deal is pre-stabilisation" and "we
#: looked and there is nothing there" are different statements, and collapsing
#: them would make Hanestowne Waterstone (no operating history at all)
#: indistinguishable from Brainerd Place (a real 6.6M U/W NOI that the report
#: deliberately withholds). The `is_dev` tag beside the deal name is what tells
#: the reader WHY the cell reads n/a.
#:
#: NOTE the Loan subtab spells its equivalent differently -- there, "Dev" is the
#: suppression literal and n/a (None -> em dash) is the no-data case. That is
#: the PDF's own convention on its Loan page, so the two subtabs are correctly
#: inconsistent with each other and each consistent with its page.
NA_LABEL = "n/a"

#: Superseded 2026-08-25. Econ Occ used to render "Dev" for a development deal
#: while NOI and the two growth columns showed real numbers -- which is how
#: Green Valley Ranch came to display an Expected Growth of -2761.9% (see
#: DEV_SUPPRESSED_COLUMNS). The PDF suppresses every metric on a dev row, so
#: NA_LABEL replaced this. Kept only so the rename is traceable; nothing reads
#: it and it should go once that is no longer useful.
DEV_LABEL = "Dev"

#: The metric columns a development classification suppresses. Everything the
#: PDF shows as n/a on a dev row, which is every metric on the subtab.
#:
#: WHY THE GROWTH COLUMNS MATTER MOST. Growth is (later - At Close) / At Close,
#: and a development deal's At Close NOI is the reading least likely to be real.
#: Eight of TIAA's ten dev deals at 26Q1 carry At Close of exactly +/-0.0, so
#: `_growth` returns None for them and the old display happened to look right.
#: The two that do not are the whole problem:
#:
#:   Green Valley Ranch (P0000100)  At Close -59,333  ->  Expected Growth -2761.9%
#:   Pegasus Life Storage (P0000066) At Close 624,689 ->  Expected Growth +169.0%
#:
#: Neither is a data error to chase -- a near-zero base makes the ratio
#: meaningless, and no correction to the numerator fixes that. The column simply
#: does not apply before stabilisation, which is what the PDF says by printing
#: n/a. So growth is suppressed by the dev rule for every dev deal INCLUDING
#: both exceptions below, and that is what makes Pegasus match the PDF.
#:
#: The comments column is NOT here and must never be: construction and lease-up
#: status is precisely what a dev row is on the page to report.
DEV_SUPPRESSED_COLUMNS = ("econ_occ", "noi", "expected_growth", "actual_growth")

#: Months of ownership below which a deal has no operating history to report and
#: every metric column reads n/a — the same suppression the dev rule applies, for
#: an unrelated reason, so a brand-new acquisition is not read as a real figure.
#:
#: One full quarter. A deal bought inside the reporting quarter has not been
#: owned for a period the page can describe: there is no full month of
#: occupancy, and the NOI columns are underwriting, not performance.
#:
#: WHY OWNERSHIP AGE AND NOT THE DATA ITSELF. Two candidate tests were measured
#: against live 26Q1 and both fail:
#:
#:   "no occupancy history"  also catches Giant 7 and East Manchester, whose
#:       occupancy feed simply stopped in Nov 2025 (both under PSA to sell). They
#:       carry real At Close and U/W readings the PDF prints. Missing recent data
#:       on a nine-year-old asset is not the same statement as a new one.
#:
#:   "econ_occ U/W == 100.0 exactly"  is a magic constant standing in for
#:       "the underwriting carries no vacancy line". A fully-leased single-tenant
#:       asset can legitimately underwrite to 100% — Town Fair Tire is already at
#:       98.7% — so this suppresses real readings on exactly the asset class
#:       where 100% is plausible. Plaza Del Mar's 100.0% IS degenerate (zero
#:       vacancy against 3.9M of revenue), but ownership age says so without
#:       having to guess from the value.
#:
#: Ownership age is also what the reference PDF says in its own comment on these
#: rows: "recent acquisition, not enough operating history". At 26Q1 the margin
#: is wide enough that the threshold is not a tuned parameter — Hanestowne
#: Waterstone is 0.4 months owned and Plaza Del Mar 0.5, and the next-youngest
#: deal in the portfolio is Jefferson Stephens at 5.4. Anything from ~1 to ~5
#: selects the same two deals.
INSUFFICIENT_HISTORY_MONTHS = 3.0

#: One Pager ``property_performance`` column -> our NOI key. The One Pager calls
#: Projected YE "actual_ye"; kept as a map so the three read paths that need the
#: translation cannot disagree.
_OP_NOI_SOURCE = {"at_close": "at_close", "uw_ye": "uw_ye",
                  "projected_ye": "actual_ye"}

#: The ``property_performance`` blocks that must ALL be zero for a column to
#: count as unpopulated. See ``_payload_unpopulated``.
_POPULATION_BLOCKS = ("revenue", "expenses", "noi")


# ══════════════════════════════════════════════════════════════════════════
# TEMPORARY HARDCODED EXCEPTIONS — REMOVE WHEN THE REAL RULE LANDS
# ══════════════════════════════════════════════════════════════════════════
#: Development deals that still render REAL numbers in some columns, keyed by
#: vcode -> the set of DEV_SUPPRESSED_COLUMNS exempted for that deal.
#:
#: Same shape of debt, and the same reasoning, as WATERS_CREEK_LTV_EXCEPTION in
#: portfolio_snapshot_loan.py: the classification is right, but one or two
#: columns are meaningful anyway, and nothing in the data currently says which.
#: Per-COLUMN rather than per-deal because both exceptions here are partial --
#: a blanket "treat as operating" would put back the garbage growth figures that
#: are the reason this rule exists.
#:
#: Verified cell-by-cell against the 26Q1 PDF, page 3 (Operating):
#:
#:   P0000078 Jefferson Waters Creek
#:     PDF row:  n/a | - | n/a | 2.9 | n/a | $2.3 | n/a | n/a
#:     Econ Occ is n/a in all three columns even though live carries a real U/W
#:     YE reading of 74.8% -- so occupancy is ACTIVELY suppressed here, not
#:     merely absent. NOI is shown: computed U/W YE 2.931M and Projected YE
#:     2.337M tie to the PDF's 2.9 and 2.3. Growth stays suppressed (its At
#:     Close is 0, so it would read n/a regardless). Deal is ~56% leased and
#:     mid-lease-up, which is why the NOI projections are real while the
#:     occupancy series is not yet meaningful.
#:
#: RETIRED 2026-09-01 — P0000066 Pegasus Life Storage.
#:
#:     It carried frozenset({"econ_occ", "noi"}) to force real occupancy and NOI
#:     through, because the PDF treats it as operating (48.0% | - | 88.3% | 1.7 |
#:     90.3% | $1.1 | n/a | n/a) while we classified it development. That
#:     classification was the bug, not the display: Lifecycle "New Construction"
#:     described the BUILDING's vintage, and the deal is a finished 2020 asset
#:     bought in 2022 with no construction-draw record. "new construction" has
#:     now been removed from config.DEV_STRATEGIES, so the deal is operating,
#:     nothing suppresses it, and this entry became unreachable — `_dev_exempt`
#:     is only ever consulted for a dev deal. Deleted rather than left inert so
#:     it cannot mislead the next reader into thinking a suppression is in play.
#:
#:     Its At-Close column and both growth columns still read as a dash / n/a,
#:     and still correctly, but by a DIFFERENT and now explicit route. The
#:     At-Close Year-0 gate in one_pager keys on DEV_STRATEGIES, so un-tagging
#:     the deal would have un-zeroed that column to 624,689 — a figure that has
#:     never been published. `one_pager.AT_CLOSE_FORCE_SUPPRESS` now names the
#:     deal outright, so the column stays zeroed, `_payload_unpopulated` still
#:     sees all three rows at zero and renders the dash, and `_growth` still
#:     returns None on a zero base. The PDF's own comment -- "U/W YE references
#:     reforecasted U/W, not initial" -- remains the editorial reason, and no
#:     data field carries it.
#:
#: The rule the survivor stands in for is roughly "a dev deal shows the columns
#: its stabilisation stage supports", which needs a lease-up/stabilisation state
#: in the data that does not currently exist. The honest route that would retire
#: it is an editorial per-cell suppression control on the page. Until then this
#: dict is technical debt and will silently keep overriding whatever real rule
#: ships.
DEV_DISPLAY_EXCEPTIONS = {
    "P0000078": frozenset({"noi"}),                 # Jefferson Waters Creek
}


def _dev_exempt(vcode: str, column: str) -> bool:
    """True when a dev deal still shows a real value in ``column``. TEMPORARY —
    see DEV_DISPLAY_EXCEPTIONS."""
    return column in DEV_DISPLAY_EXCEPTIONS.get(
        str(vcode or "").strip().upper(), frozenset())
# ══════════════════════════════════════════════════════════════════════════


def display_values(row: dict, column: str) -> list:
    """Every value the UI renders for one metric column of one row.

    A list because ``noi`` is three columns on the page under one suppression
    key. Exists so a caller auditing the display rule reads the same fields the
    UI does, instead of re-deriving which ``*_display`` name goes with which
    column and drifting out of step with it.
    """
    if column == "noi":
        return list((row.get("noi_display") or {}).values())
    return [row.get(f"{column}_display")]

#: TEMPORARY (2026-08-24): consult Lifecycle when Investment_Strategy is empty.
#:
#: This relaxes the "Investment_Strategy ONLY" decision recorded above, for one
#: reason: that field is 0/110 populated, so the rule classifies every deal as
#: operating and the whole "Dev" display — including the Waters Creek LTV
#: exception in the Loan subtab — is dead code that cannot be tested against the
#: PDF. With the fallback on, 10 of TIAA's 35 deals classify as development
#: (9 Lifecycle=Development + Pegasus Life Storage, Lifecycle=New Construction),
#: which is the population the PDF renders as "Dev".
#:
#: Investment_Strategy still WINS wherever it is populated — this only fills the
#: gap. Set to False to restore the strict Investment_Strategy-only rule; that is
#: the correct end state once MRI feeds the field (which needs a
#: Prop_Info_Core.sql + mri_service.MRI_COLUMNS change).
DEV_STRATEGY_ALLOW_LIFECYCLE_FALLBACK = True


def resolve_strategy(entry: dict) -> tuple[str, str]:
    """(strategy value, which field it came from) for one Step 1 deal entry.

    The single place the strategy field is chosen, so the "Dev" display, the
    mOrigLoanAmt debt path and the "Excluding Development Deals" subtotal cannot
    disagree about what a development deal is. The source is returned alongside
    so a proxy-derived classification is never mistaken for the real field.
    """
    pure = str(entry.get("investment_strategy") or "").strip()
    if pure:
        return pure, "Investment_Strategy"
    if DEV_STRATEGY_ALLOW_LIFECYCLE_FALLBACK:
        fallback = str(entry.get("strategy") or "").strip()
        if fallback:
            return fallback, "Lifecycle (proxy)"
    return "", "unavailable"


def _num(v):
    """Float or None. Zero is preserved — it is data, not absence."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f


def _growth(later, base):
    """(later - base) / base, or None when it cannot be computed.

    Returns None rather than 0.0 when the base is missing or zero, so a deal
    with no At Close NOI is flagged instead of silently reading as flat.
    """
    if later is None or base is None or base == 0:
        return None
    return (later - base) / base


def _payload_unpopulated(pp: dict, column: str) -> bool:
    """True when the One Pager returned its untouched default for ``column``.

    ``one_pager.get_property_performance`` seeds revenue, expenses and NOI to a
    literal ``0`` and only overwrites them if it finds data — an ``at_close_noi``
    row, or a December Projected IS date to scan. When it finds neither, all
    three stay at zero and the column is indistinguishable from a property that
    genuinely earned nothing.

    That is the City West bug: a foreclosed deal with no ``at_close_noi`` row
    printed "0.0" in the At Close column, which reads as a measured zero rather
    than as an absence. Real zero NOI is possible — a deal can break even — but
    zero revenue AND zero expenses AND zero NOI together is not a property, it
    is the default. Requiring all three is what keeps a genuine break-even
    reading (nonzero revenue, matching expenses) out of this branch.

    Read here rather than fixed at source deliberately: those defaults are One
    Pager public API, and the One Pager view, Property Financials and the
    dashboard all do arithmetic on them. Turning them into None upstream is the
    right end state and a much wider blast radius than this page.
    """
    if not pp:
        return True
    for block in _POPULATION_BLOCKS:
        v = (pp.get(block) or {}).get(column)
        if v is None:
            continue
        try:
            if float(v) != 0.0:
                return False
        except (TypeError, ValueError):
            return False                # non-numeric: not the numeric default
    return True


def _parse_date(v) -> Optional[date]:
    """Lenient date parse, or None. Accepts ISO and the US formats MRI emits."""
    s = str(v or "").strip()
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        pass
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s.split(" ")[0], fmt).date()
        except ValueError:
            continue
    return None


def months_owned(payload: dict, quarter_end: Optional[date]) -> Optional[float]:
    """Months from the deal's closing date to ``quarter_end``, or None.

    Reads ``date_closed`` straight out of the One Pager payload this module
    already fetches — the same field the One Pager prints as "Date Closed",
    which ``data_service._enrich_acquisition_dates`` derives from the earliest
    accounting activity. No new query, and no second opinion about when a deal
    closed.

    Both spellings of the block are accepted: ``one_pager`` builds
    ``general_information``, and ``financials_service.get_one_pager_data``
    republishes it as ``general``. This module's provider is injected and wraps
    one or the other depending on caller, so reading a single key would work in
    app and silently return None under the REST-backed guardrails — which is
    exactly how this rule first measured zero deals.
    """
    if not quarter_end:
        return None
    p = payload or {}
    block = p.get("general") or p.get("general_information") or {}
    closed = _parse_date(block.get("date_closed"))
    if not closed:
        return None
    return (quarter_end - closed).days / 30.44


# ── Subtotals ─────────────────────────────────────────────────────────────
#
# REVERSES the "no subtotals" decision this module shipped with (see the note in
# the docstring). The reference PDF page 3 carries a total row per fund and a
# portfolio total, so the subtab needs them; they are computed HERE rather than
# in the component so a freeze captures them.
#
# THE AGGREGATION, derived from the PDF's own Individual Investments row rather
# than assumed — that row reads 90.8% | 22.6 | 94.6% | 26.0 | 95.1% | 24.4 |
# 15.0% | 7.9% over its eight member deals:
#
#   NOI       SUMMED. 22.6 / 26.1 / 24.4 against the published 22.6 / 26.0 /
#             24.4 — exact on two columns and one rounding step on the third.
#             A missing or "-" reading counts as zero, which is what the PDF's
#             own dashes do — and so does a cell a suppression rule withholds,
#             which is the whole subtotal contract in the module docstring. The
#             eight member deals of that Individual Investments row are all
#             operating, which is why it validated the arithmetic cleanly while
#             the portfolio row (ten suppressed dev rows) did not.
#
#   Growth    RECOMPUTED FROM THE SUMS, not averaged from the member growths:
#             (26.0 - 22.6) / 22.6 = 15.04% and (24.4 - 22.6) / 22.6 = 7.96%
#             against the published 15.0% and 7.9%. Averaging the deal-level
#             growths gives neither. Confirmed on all five funds and the
#             portfolio row (66.7% and 52.6% both reproduce to 0.2pp).
#
#   Econ Occ  NOI-WEIGHTED average over the deals that carry a reading, weighted
#             by that same column's NOI. Simple averaging is decisively wrong —
#             At Close comes out 82.7% against a published 90.8%. Weighted lands
#             within 0.2-1.5pp on all three columns, the residual being that the
#             published inputs are already rounded to one decimal.
#
#             Revenue would be the theoretically better weight for an occupancy
#             figure, and the backend has it; NOI is used because NOI is what
#             the page shows, so it is the only weight this can be validated
#             against. Revisit if a fund's mix ever makes the two diverge.
_NOI_KEYS = ("at_close", "uw_ye", "projected_ye")


def _visible(v):
    """The number a display cell shows, or None when it shows no number.

    The single place a suppressed cell is turned into "contributes nothing".
    ``*_display`` fields are polymorphic by design — a float when the metric
    computed, the literal NA_LABEL when a rule withholds it, None when there is
    no data — and every total on this page is built through this function, so it
    can only ever sum what a reader can see.
    """
    if v is None or isinstance(v, str):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f


def _visible_noi(row: dict, key: str):
    return _visible((row.get("noi_display") or {}).get(key))


def _weighted(rows: list, occ_key: str, noi_key: str):
    """NOI-weighted mean of one occupancy column, or None.

    Only rows carrying BOTH a reading and a non-zero weight contribute: a deal
    with no NOI cannot pull an income-weighted average, and a deal with no
    reading has nothing to contribute.

    A row whose occupancy cell is suppressed contributes to no occupancy
    column, and the weight is the VISIBLE NOI — so a deal reading n/a across the
    row is absent from both sides of the ratio rather than silently steering an
    average the reader cannot see it in.
    """
    num = den = 0.0
    for r in rows:
        if isinstance(r.get("econ_occ_display"), str):
            continue                        # occupancy withheld on this row
        o = (r.get("econ_occ") or {}).get(occ_key)
        w = _visible_noi(r, noi_key)
        if o is None or not w:
            continue
        num += o * w
        den += w
    return (num / den) if den else None


def operating_subtotal(rows: list, label: str) -> dict:
    """One total row over ``rows`` — a fund's deals, or every deal.

    Sums the DISPLAY twins, not the raw metrics: see the subtotal contract in
    the module docstring. ``noi_contributors`` reports how many rows actually
    fed each column, which is what makes the exclusion auditable rather than
    something a reader has to take on trust.
    """
    noi, contributors = {}, {}
    for k in _NOI_KEYS:
        vals = [_visible_noi(r, k) for r in rows]
        vals = [v for v in vals if v is not None]
        noi[k] = sum(vals) if vals else None
        contributors[k] = len(vals)

    occ = {"at_close": _weighted(rows, "at_close", "at_close"),
           "uw_ye": _weighted(rows, "uw_ye", "uw_ye"),
           "projected_ye": _weighted(rows, "projected_ye", "projected_ye")}

    return {
        "label": label,
        "deal_count": len(rows),
        # Counted, but NOT contributing: a dev row's metrics read n/a, so its
        # NOI is excluded from the sums above. The count stays because the row
        # is on the page and the reader can see it.
        "dev_count": sum(1 for r in rows if r.get("is_dev")),
        "suppressed_count": sum(
            1 for r in rows
            if all(_visible_noi(r, k) is None for k in _NOI_KEYS)),
        "noi": noi,
        "noi_contributors": contributors,
        "econ_occ": occ,
        # From the sums, as the PDF does.
        "expected_growth": _growth(noi["uw_ye"], noi["at_close"]),
        "actual_growth": _growth(noi["projected_ye"], noi["at_close"]),
        "econ_occ_basis": ("NOI-weighted mean over deals whose occupancy cell "
                           "is visible"),
        "noi_basis": "sum of the visible cells; a suppressed cell contributes 0",
    }


def _default_comment_loader(investor_code: str, quarter: str) -> dict:
    """vcode -> operating comment, from Step 2 persistence (read-only)."""
    try:
        from flask_app.services.portfolio_snapshot_persistence import get_elements
        rows = get_elements("comment", investor_code, quarter,
                            scope="deal", field="operating")
        return {r.get("scope_key"): r.get("comment_text") for r in rows}
    except Exception as exc:            # persistence not initialised yet
        log.debug("operating comments unavailable: %s", exc)
        return {}


def assemble_operating(investor_code: str, quarter: str, *,
                       resolved: dict,
                       one_pager_provider: Callable[[str, str], dict],
                       comment_loader: Optional[Callable] = None) -> dict:
    """Build the Operating subtab for one investor and quarter.

    ``resolved`` is the output of Step 1's ``resolve_investor_deals``.
    ``one_pager_provider(vcode, quarter)`` returns a One Pager payload.
    """
    from flask_app.services.portfolio_snapshot_service import (
        group_total_label, PORTFOLIO_TOTAL_LABEL,
    )

    loader = comment_loader or _default_comment_loader
    comments = loader(investor_code, quarter) or {}

    # Quarter end for the insufficient-history test. Via the One Pager's public
    # helper rather than the service's private _quarter_end, so this module does
    # not depend on a private name for a value it can derive itself. None just
    # disables the rule — a quarter string this cannot parse should not blank a
    # page.
    try:
        from one_pager import quarter_to_date_range      # read-only reuse
        _, q_end = quarter_to_date_range(quarter)
    except Exception as exc:
        log.debug("quarter end unavailable for %r, insufficient-history rule "
                  "disabled: %s", quarter, exc)
        q_end = None

    groups: dict[str, list] = {}
    diag = {"deals": 0, "with_noi": 0, "dev": 0, "missing_at_close": 0,
            "provider_errors": 0, "comments_attached": 0,
            "dev_suppressed": 0, "dev_exceptions": 0,
            # NOI columns where the One Pager returned its untouched default
            # (revenue, expenses and NOI all zero) and we emit None instead.
            "unpopulated_noi": 0,
            "insufficient_history": 0}

    def build_row(vcode: str, name: str, strategy: str,
                  extra_flags: Optional[list] = None) -> dict:
        flags = list(extra_flags or [])
        dev = is_dev_deal(strategy)
        occ = {"at_close": None, "uw_ye": None, "projected_ye": None,
               "ytd_actual": None}
        noi = {"at_close": None, "uw_ye": None, "projected_ye": None}

        try:
            payload = one_pager_provider(vcode, quarter) or {}
        except Exception as exc:
            diag["provider_errors"] += 1
            flags.append(f"One Pager unavailable: {str(exc)[:90]}")
            payload = {}

        pp = payload.get("property_performance") or {}
        if not pp:
            flags.append("no property_performance for this quarter")
        else:
            n, o = pp.get("noi") or {}, pp.get("economic_occ") or {}
            # None, not the One Pager's default 0, where nothing was populated:
            # "no At Close NOI row for this deal" must not print as a measured
            # zero. See _payload_unpopulated.
            noi = {}
            for key, src in _OP_NOI_SOURCE.items():
                if _payload_unpopulated(pp, src):
                    noi[key] = None
                    diag["unpopulated_noi"] += 1
                else:
                    noi[key] = _num(n.get(src))
            occ = {"at_close": _num(o.get("at_close")),
                   "uw_ye": _num(o.get("uw_ye")),
                   "projected_ye": _num(o.get("actual_ye")),
                   "ytd_actual": _num(o.get("ytd_actual"))}

        # ---- insufficient operating history ----------------------------------
        # A deal bought inside the reporting quarter has nothing to report yet.
        # Independent of the dev rule and applied to the same four columns; see
        # INSUFFICIENT_HISTORY_MONTHS for why this reads ownership age rather
        # than trying to infer "too new" from the readings themselves.
        # The guard fails OPEN — no closing date means no suppression — so a
        # payload that does not carry one silently exempts the deal. That is
        # backwards for a rule whose purpose is catching deals too new to
        # report, and it is how the rule managed to fire on NOTHING until the
        # lean provider started supplying the date (Sep 3 2026): `months_owned`
        # returned None for all 30 deals on the page.
        #
        # Left failing open on purpose — suppressing on unknown would blank a
        # deal whose payload lacks the field for an unrelated reason — but the
        # condition is now REPORTED, so it shows up as a flag rather than as a
        # rule that quietly does nothing.
        mo = months_owned(payload, q_end)
        if mo is None:
            diag["no_closing_date"] = diag.get("no_closing_date", 0) + 1
            flags.append(
                "no closing date on the payload — the insufficient-history "
                "rule could not be applied to this deal")
        insufficient = mo is not None and mo < INSUFFICIENT_HISTORY_MONTHS
        if insufficient:
            diag["insufficient_history"] += 1
            flags.append(
                f"insufficient operating history — owned {mo:.1f} month(s) at "
                f"quarter end (< {INSUFFICIENT_HISTORY_MONTHS:g}); every metric "
                f"shown as n/a")

        exp_g = _growth(noi["uw_ye"], noi["at_close"])
        act_g = _growth(noi["projected_ye"], noi["at_close"])

        if noi["at_close"] in (None, 0):
            diag["missing_at_close"] += 1
            flags.append("growth unavailable: no At Close NOI")
        for label, key in (("U/W YE NOI", "uw_ye"),
                           ("Projected YE NOI", "projected_ye")):
            if noi[key] is None:
                flags.append(f"{label} missing")
        if any(v is not None for v in noi.values()):
            diag["with_noi"] += 1
        if dev:
            diag["dev"] += 1

        comment = comments.get(vcode)
        if comment:
            diag["comments_attached"] += 1

        # ---- dev suppression: every metric reads n/a before stabilisation ----
        # Applied to the *_display fields ONLY. The raw `econ_occ`, `noi`,
        # `expected_growth` and `actual_growth` below are left exactly as
        # computed, so this is a display rule that moves no value: a frozen
        # payload keeps its real numbers and re-renders under whatever the rule
        # is at read time, and the guardrails still audit the true figures.
        exempted = sorted(DEV_DISPLAY_EXCEPTIONS.get(
            str(vcode or "").strip().upper(), frozenset()))
        if dev:
            diag["dev_suppressed"] += 1
            suppressed = [c for c in DEV_SUPPRESSED_COLUMNS
                          if not _dev_exempt(vcode, c)]
            flags.append("development deal — "
                         + ", ".join(suppressed) + " shown as n/a")
            if exempted:
                diag["dev_exceptions"] += 1
                flags.append("TEMPORARY exception — real "
                             + ", ".join(exempted)
                             + " shown despite dev classification "
                               "(see DEV_DISPLAY_EXCEPTIONS)")

        def shown(column: str, value):
            """``value``, or NA_LABEL when a rule suppresses ``column``.

            One gate for all four metric columns, so occupancy, NOI and the two
            growth figures can never disagree about whether a deal is being
            suppressed — the disagreement that produced "Dev" occupancy sitting
            next to a -2761.9% growth figure on the same row. It is also the
            only place suppression is decided, which is what lets
            ``operating_subtotal`` honour every rule just by reading the twins.

            Insufficient history is tested FIRST, so it outranks a dev
            exception: a deal owned for two weeks has nothing to show in a
            column regardless of which ones its stabilisation stage would
            normally support.
            """
            if insufficient:
                return NA_LABEL
            if dev and not _dev_exempt(vcode, column):
                return NA_LABEL
            return value

        diag["deals"] += 1
        return {
            "vcode": vcode, "name": name,
            "strategy": strategy,
            "is_dev": dev,
            # Which columns the dev rule withheld, and which a TEMPORARY
            # exception let through. Surfaced so the report itself can mark the
            # hardcode rather than it living only in this file.
            "dev_suppressed_columns": (
                [c for c in DEV_SUPPRESSED_COLUMNS if not _dev_exempt(vcode, c)]
                if dev else []),
            "dev_display_exception": exempted if dev else [],
            # Why this row reads n/a when it is not a development deal, and the
            # number the rule turned on — surfaced so the page can explain
            # itself rather than the reason living only in this file.
            "insufficient_history": insufficient,
            "months_owned": (round(mo, 1) if mo is not None else None),
            # Fall back Projected YE -> YTD actual -> At Close: several deals
            # carry only some of the three (Giant 7 has no Projected YE
            # occupancy at 26Q1), and blanking the cell when a reading does
            # exist would be wrong.
            "econ_occ_display": shown("econ_occ", next(
                (v for v in (occ["projected_ye"], occ["ytd_actual"],
                             occ["at_close"]) if v is not None), None)),
            "econ_occ_basis": (
                "insufficient operating history" if insufficient
                else "dev" if (dev and not _dev_exempt(vcode, "econ_occ"))
                else next(
                    (k for k in ("projected_ye", "ytd_actual", "at_close")
                     if occ[k] is not None), None)),
            # Raw values, UNSUPPRESSED — see the note above `shown`.
            "econ_occ": occ,
            "noi": noi,
            "expected_growth": exp_g,
            "actual_growth": act_g,
            # Per-column display twins. The PDF's three NOI columns and two
            # growth columns each suppress independently, which is why NOI is a
            # dict here rather than one flag for the group: Waters Creek shows
            # NOI while its occupancy is withheld, on the same row.
            "noi_display": {k: shown("noi", v) for k, v in noi.items()},
            "expected_growth_display": shown("expected_growth", exp_g),
            "actual_growth_display": shown("actual_growth", act_g),
            "operating_comment": comment,
            "flags": flags,
        }

    for group, items in (resolved.get("groups") or {}).items():
        rows = [build_row(e["vcode"], e["name"], resolve_strategy(e)[0])
                for e in items]
        groups[group] = rows

    # Deals Step 1 flagged (broken ownership chain, e.g. 45th & Main) still
    # belong on the Operating page: the operating metrics are property-level and
    # do not depend on ownership at all. They carry their Step 1 flag forward.
    flagged_rows = []
    for f in (resolved.get("flagged") or []):
        row = build_row(f["vcode"], f["name"], resolve_strategy(f)[0],
                        extra_flags=[f"ownership {f.get('reason', 'unavailable')}"])
        row["ownership_flagged"] = True
        flagged_rows.append(row)

    return {
        "investor_code": resolved.get("investor_code", investor_code),
        "investor_name": resolved.get("investor_name", investor_code),
        "quarter": quarter,
        "subtab": "operating",
        "scaled": False,        # property-level metrics are never scaled
        "groups": groups,
        "ownership_flagged": flagged_rows,
        # Subtotals ride ALONGSIDE `groups` rather than nesting inside it, so
        # every existing consumer — the component, the guardrails, a frozen
        # payload — keeps reading `groups` as {name: [rows]} unchanged.
        "group_labels": {g: group_total_label(g) for g in groups},
        "subtotals": {g: operating_subtotal(rows, group_total_label(g))
                      for g, rows in groups.items()},
        "total": operating_subtotal(
            [r for rows in groups.values() for r in rows] + flagged_rows,
            PORTFOLIO_TOTAL_LABEL),
        "diagnostics": diag,
    }


# ── Self-test ─────────────────────────────────────────────────────────────

_PDF_26Q1 = {
    # vcode: (label, econ_occ_lo, econ_occ_hi, at_close, uw_ye, proj_ye,
    #         expected_growth_pct, actual_growth_pct)
    "P0000019": ("Giant 7", 97.8, 98.3, 8.8, 9.3, 9.4, 5.3, 6.4),
    "P0000075": ("Camp Creek", None, None, 6.9, 6.8, 6.4, 9.0, 6.8),
    "P0000030": ("Nottingham Village", None, None, 3.6, 3.2, 2.1, 71.7, 53.2),
    "P0000068": ("Point at Plymouth Meeting", None, None, None, None, None,
                 55.1, 11.9),
}


def _selftest():                                    # pragma: no cover
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

    QUARTER = "2026-Q1"
    INV = "TGAM"

    ti = api.token_info()
    print(f"LIVE token={ti['username']} ({ti['hours_left']}h left)  "
          f"build={api.get('/api/data/version').get('version')}  "
          f"actuals_through={api.get('/api/data/config').get('actuals_through')}")

    # ---- Step 1 foundation, via narrow per-entity relationship pulls ----
    inv = pd.DataFrame(api.get("/api/data/deals/all").get("deals") or [])
    seen, frontier, rows = set(), [INV], []

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
    rel = pd.DataFrame(rows).drop_duplicates()
    resolved = resolve_investor_deals(INV, QUARTER, rel, inv)
    print(f"Step 1: {resolved['diagnostics']['deal_count']} deals in "
          f"{resolved['diagnostics']['group_count']} groups, "
          f"{len(resolved['flagged'])} ownership-flagged\n")

    # ---- Step 2 persistence on a scratch db, to prove comments attach ----
    tmp = os.path.join(tempfile.mkdtemp(prefix="ps_step3a_"), "t.db")
    eng = sqlalchemy.create_engine(f"sqlite:///{tmp}")
    P._engine = lambda: eng                          # type: ignore[assignment]
    P._is_postgres = lambda: False                   # type: ignore[assignment]
    P.save_comment(INV, QUARTER, "deal", "operating",
                   "Leasing ahead of plan; ECRI driving in-place rents.",
                   scope_key="P0000019", updated_by="selftest")
    P.save_comment(INV, QUARTER, "deal", "operating",
                   "Anchor renewal signed.", scope_key="P0000075",
                   updated_by="selftest")

    cache: dict = {}

    def provider(vcode, quarter):
        key = (vcode, quarter)
        if key not in cache:
            cache[key] = api.get(f"/api/financials/{vcode}/one-pager",
                                 params={"quarter": quarter})
        return cache[key]

    out = assemble_operating(INV, QUARTER, resolved=resolved,
                             one_pager_provider=provider,
                             comment_loader=lambda i, q: _default_comment_loader(i, q))

    flat = {r["vcode"]: r for rows_ in out["groups"].values() for r in rows_}
    for r in out["ownership_flagged"]:
        flat[r["vcode"]] = r

    print("=" * 108)
    print(f"COMPUTED vs 26Q1 PDF   ({out['investor_name']} {out['quarter']})")
    print("=" * 108)
    hdr = (f"{'deal':<26}{'metric':<20}{'computed':>12}{'PDF':>10}"
           f"{'delta':>10}   verdict")
    print(hdr)
    print("-" * 108)
    checks = []
    for vc, (label, olo, ohi, p_ac, p_uw, p_pj, p_eg, p_ag) in _PDF_26Q1.items():
        r = flat.get(vc)
        if not r:
            print(f"{label:<26}{'NOT IN SET':<20}")
            checks.append((f"{label} present", False))
            continue
        rowsout = [
            ("NOI At Close ($M)", (r["noi"]["at_close"] or 0) / 1e6, p_ac, 0.35),
            ("NOI U/W YE ($M)", (r["noi"]["uw_ye"] or 0) / 1e6, p_uw, 0.35),
            ("NOI Proj YE ($M)", (r["noi"]["projected_ye"] or 0) / 1e6, p_pj, 0.35),
            ("Expected Growth %", (r["expected_growth"] or 0) * 100, p_eg, 0.4),
            ("Actual Growth %", (r["actual_growth"] or 0) * 100, p_ag, 0.4),
        ]
        first = True
        for metric, comp, pdf, tol in rowsout:
            if pdf is None:
                continue
            delta = comp - pdf
            ok = abs(delta) <= tol
            verdict = "ok" if ok else ("ROUNDING?" if abs(delta) <= 1.0
                                       else "MISMATCH")
            print(f"{(label if first else ''):<26}{metric:<20}"
                  f"{comp:>12.3f}{pdf:>10.2f}{delta:>+10.3f}   {verdict}")
            first = False
            checks.append((f"{label} {metric}", ok))
        if olo is not None:
            occ = r["econ_occ"]["projected_ye"] or r["econ_occ"]["at_close"] or 0
            ok = olo <= occ <= ohi
            print(f"{'':<26}{'Econ Occ %':<20}{occ:>12.2f}"
                  f"{f'{olo}-{ohi}':>10}{'':>10}   {'ok' if ok else 'MISMATCH'}")
            checks.append((f"{label} Econ Occ in band", ok))
        print()

    print("=" * 108)
    print("STRUCTURE CHECKS")
    struct = [
        ("groups present and non-empty",
         len(out["groups"]) >= 5 and all(out["groups"].values())),
        ("Individual Investments group exists",
         "Individual Investments" in out["groups"]),
        ("no scaling applied to operating metrics", out["scaled"] is False),
        ("45th & Main still appears (ownership-flagged)",
         any(r["vcode"] == "P0000089" for r in out["ownership_flagged"])),
        ("45th & Main carries its ownership flag",
         any("ownership" in f.lower()
             for r in out["ownership_flagged"] if r["vcode"] == "P0000089"
             for f in r["flags"])),
        ("Giant 7 comment attached",
         (flat.get("P0000019") or {}).get("operating_comment", "").startswith(
             "Leasing ahead")),
        ("Camp Creek comment attached",
         (flat.get("P0000075") or {}).get("operating_comment") ==
         "Anchor renewal signed."),
        ("deal with no comment yields None, not ''",
         (flat.get("P0000030") or {}).get("operating_comment") is None),
        # Every suppressed column on every dev row, exceptions honoured. Reads
        # the *_display twins, which is what the UI renders.
        ("dev deals show n/a in every non-exempted metric column",
         all(v == NA_LABEL
             for r in flat.values() if r["is_dev"]
             for c in r["dev_suppressed_columns"]
             for v in display_values(r, c))),
        ("dev exceptions still show a real value, not n/a",
         all(v != NA_LABEL
             for r in flat.values() if r["is_dev"]
             for c in r["dev_display_exception"]
             for v in display_values(r, c))),
        ("dev deals keep their comment column",
         all("operating_comment" in r for r in flat.values() if r["is_dev"])),
        # The suppression is display-only: the raw figures must survive
        # untouched so a freeze keeps real numbers and the audit can see them.
        ("raw metrics never carry the n/a literal",
         all(not isinstance(v, str)
             for r in flat.values()
             for v in (list((r["noi"] or {}).values())
                       + list((r["econ_occ"] or {}).values())
                       + [r["expected_growth"], r["actual_growth"]])
             if v is not None)),
        ("missing data is None, never 0-faked",
         all(r["expected_growth"] is None or isinstance(r["expected_growth"], float)
             for r in flat.values())),
    ]
    for label, ok in struct:
        print(f"    [{'PASS' if ok else 'FAIL'}] {label}")
        checks.append((label, ok))

    d = out["diagnostics"]
    print(f"\n  diagnostics: {d}")
    devs = [r["name"] for r in flat.values() if r["is_dev"]]
    print(f"  dev deals ({len(devs)}): {sorted(devs)[:8]}")
    nog = [r["name"] for r in flat.values() if r["expected_growth"] is None]
    print(f"  no-growth (flagged) ({len(nog)}): {sorted(nog)[:8]}")

    print("\n" + "=" * 108)
    print("ASSEMBLED STRUCTURE — two deals verbatim")
    import json
    for vc in ("P0000019", "P0000030"):
        print(f"\n{vc}:")
        print(json.dumps(flat.get(vc), indent=2, default=str)[:1100])

    passed = sum(1 for _, c in checks if c)
    print(f"\n  {passed}/{len(checks)} checks passed")
    return 0


if __name__ == "__main__":                          # pragma: no cover
    raise SystemExit(_selftest())
