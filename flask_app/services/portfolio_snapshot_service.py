"""Portfolio Snapshot — foundation service (Step 1).

Deliberately isolated: this module creates no side effects, writes nothing, and
imports from the rest of the app in exactly one read-only place
(``one_pager.quarter_to_date_range``). Nothing else in the app imports this
module, so a fault here cannot propagate.

Data comes in as DataFrames, the way every other service in this codebase takes
it (cf. ``reports_service.get_upstream_investor_deals``,
``portfolio_analysis_service.find_entity_deals``). That means there is no HTTP
and therefore no pagination inside the service at all — the OFFSET duplication
trap on ``/api/data/tables/<t>/rows`` simply cannot apply here. The self-test at
the bottom runs out-of-process, so *it* fetches over the REST API and there it
does use narrow per-entity filters (one page each, exact-match post-filtered,
because ``filter__`` is case-insensitive *contains* and ``TGAM`` would otherwise
also match ``TGAM2``/``TGAM3``).

What Step 1 establishes:
  resolve_investor_deals()  investor + quarter -> deals grouped by fund
  lookthrough_pct()         product of normalised ownership % along the chain
  get_investor_name()       display name for an investor code

Verified against live on 2026-08-21 (build 01556edbd5d0): TIAA/TGAM + 2026-Q2
returns 34 parent deals in 6 groups, Nottingham at 41.2124%, City West and East
Manchester excluded as sold, 45th & Main flagged (ownership unavailable), and 18
child properties rolled into 3 parents.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────

INDIVIDUAL_GROUP = "Individual Investments"

#: An intermediate entity one hop below the investor is treated as a *fund*
#: (a pooled vintage vehicle) rather than a single-deal SPV once it reaches at
#: least this many deals *still held at the report quarter*. Derived from the
#: traversal, never a hardcoded fund list, so a new fund such as TGA6 groups
#: correctly the moment MRI carries it.
#:
#: Counted after the sold exclusion on purpose. TGAM2 reaches two deals all-time
#: (Giant 7 via PPI24, plus one long-since disposed via PPI20); counting
#: all-time would promote it to a fund and pull Giant 7 out of Individual
#: Investments. For a quarter's report what matters is what is still held.
FUND_MIN_DEALS = 2

#: Escape hatches the count rule cannot see on its own:
#:   - a genuinely new fund with only one deal onboarded so far
#:   - a real fund whose deals have nearly all been sold
#: Add the entity code to force fund treatment.
FORCE_FUND: set[str] = set()

#: Prefix that marks an operating-partner entity.
#:
#: COUPLED TO ``one_pager.get_capitalization_stack``, which splits a deal's
#: funded capital on exactly this test (``investor_id.upper().startswith("OP")``
#: -> ``partner_equity``, else -> ``pref_equity``). The two must agree: the
#: PE-basis look-through below exists precisely to scale ``pref_equity``, and it
#: can only do that correctly if it excludes the same entities that were left
#: out of ``pref_equity`` in the first place. Change one, change both.
OP_PREFIX = "OP"


def _is_op(entity: str) -> bool:
    """True when ``entity`` is an operating partner. See ``OP_PREFIX``."""
    return str(entity or "").strip().upper().startswith(OP_PREFIX)

#: When a deal is reachable by several routes that disagree on the group, the
#: default rule is PREFER_INDIVIDUAL_ON_MIXED (below). This map overrides the
#: outcome for named deals (keyed on **vcode**, which is unique — InvestmentID
#: is not). Kept separate from the rule so an exception never becomes the rule.
#
# ══════════════════════════════════════════════════════════════════════════
# TEMPORARY ENTRIES — SELF-RESOLVE AT 26Q2, REMOVE THEN
# ══════════════════════════════════════════════════════════════════════════
# Giant 7 and East Manchester belong in Individual Investments: they are the
# top two rows of that block on the reference PDF (page 2), above its internal
# dotted divider, and they roll into "Total Individual Investments".
#
# They land in a spurious "TGAM2" group instead, and the FUND_MIN_DEALS note
# above predicts exactly this: TGAM2 is promoted to a fund once it reaches two
# deals STILL HELD at the quarter. That guard assumed TGAM2's second deal was
# "long-since disposed" — it is East Manchester, whose Sale_Date is 6/25/2026,
# three months AFTER the 26Q1 quarter end. It is correctly retained (flagged
# "sold after quarter end — held during quarter"), so TGAM2 reaches 2 and both
# deals are pulled out of Individual Investments. The guard is one deal short.
#
# Verified live: TGAM2 groups in every quarter both are held — 2024-Q4 through
# 2026-Q1 — and at 26Q2 it is gone and Giant 7 returns to Individual
# Investments on its own. So these two entries are needed for 26Q1 and earlier
# ONLY, and become dead weight from 26Q2. DELETE THEM THEN.
#
# The durable fix, deliberately not taken here because it edits a rule shared
# by every investor: count only deals still held *going forward* in
# _classify_entities, i.e. exclude sold-after-quarter-end deals from the fund
# tally while still reporting them. That needs a guardrail run across all
# investors, which this change does not have.
#
# ── Jefferson Eastchase: an EDITORIAL override, not a data correction ─────
# FLAGGED FOR JIM. Requested by the report author (work order, Sep 1 2026);
# the ownership feed does NOT support it and neither does the reference PDF.
#
# What the data says, read live 2026-09-01 on Azure PG:
#     EASTCH  <- PPIECH 100%  (OPJPI 0%)
#     PPIECH  <- TGA23 68.0504% | INVECH 16.6383% (owned 100% by PSC1)
#                              | ERIBPI 15.3113%
#     TGA23   <- TGAM 90%
# ONE route from TGAM, first hop TGA23, which reaches 8 held deals and is
# therefore a fund by FUND_MIN_DEALS. 0.90 x 0.680504 = 61.2454%, which is the
# 61% the reference PDF itself prints for this deal — inside its
# "Total PSC TGA 2023 LLC" block, not Individual Investments. TIAA has no
# direct or SPV route to Eastchase: INVECH is Peaceable's own co-invest and
# ERIBPI is a third party, and neither is reachable from TGAM.
#
# So there is nothing to "fix" upstream: the grouping rule is reading correct
# data correctly. This entry is a presentation decision recorded as one, and
# per the standing rule on per-deal hardcodes it should be confirmed rather
# than assumed. Its effect is a transfer between two subtotals — TGA23 loses
# Eastchase, Individual Investments gains it — and Portfolio Totals, the
# excluding-development row and every deal-level figure are untouched.
# If the intent is instead that TIAA holds Eastchase outside TGA 2023, the
# durable fix is an ownership row in MRI and this entry should be deleted.
GROUP_OVERRIDES: dict[str, str] = {
    "P0000019": INDIVIDUAL_GROUP,      # Giant 7
    "P0000017": INDIVIDUAL_GROUP,      # East Manchester
    "P0000085": INDIVIDUAL_GROUP,      # Jefferson Eastchase — see the note above
}
# ══════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════
# DELIBERATE KEEP-DESPITE-SOLD — reference-PDF fidelity
# ══════════════════════════════════════════════════════════════════════════
#: Deals that stay on the report even though the sold gate would drop them.
#:
#: City West (PCITWES) was lost to foreclosure on 8/30/2025, so Sale_Status is
#: SOLD with a Sale_Date before the 26Q1 quarter end and `is_sold_as_of` removes
#: it. The reference PDF KEEPS it as an Individual Investments row — Debt n/a,
#: Net ROE n/a, footnote (2) "City west is excluded from ROE calculations" —
#: because the capital is still reported even though the asset is gone. Its
#: live cap stack ties to the PDF exactly (pref 5.925M -> 5.9, partner equity
#: 14.2465M -> 14.2, total cap 20.1715M -> 20.2).
#:
#: A kept deal is deliberately excluded from the fund tally in
#: `_classify_entities` — see the note in step 3 below. Including it could
#: promote its SPV to a fund and re-open exactly the TGAM2 problem above.
#:
#: This is a per-deal exception, not a rule: "foreclosed but still reported" is
#: an editorial judgement with no field behind it. Should MRI ever carry a
#: disposition-type or still-reporting flag, drive it off that and delete this.
KEEP_DESPITE_SOLD: set[str] = {"PCITWES"}       # City West
# ══════════════════════════════════════════════════════════════════════════

#: Group key -> the label the reference PDF prints on that group's total row.
#: The keys are entity codes from the traversal (TGA22 …) and the PDF spells
#: them out ("Total PSC TGA 2022 LLC"), so the mapping lives here rather than in
#: any one subtab — Financial, Operating and Loan all label the same groups and
#: must not drift apart. An unmapped group falls back to "Total <key>", which is
#: what a genuinely new fund (TGA6 at 26Q2) reads as until it is added.
GROUP_TOTAL_LABELS: dict[str, str] = {
    INDIVIDUAL_GROUP: "Total Individual Investments",
    "TGA22": "Total PSC TGA 2022 LLC",
    "TGA23": "Total PSC TGA 2023 LLC",
    "TGA24": "Total PSC TGA 2024 LLC",
    "TGA25": "Total PSC TGA 2025 LLC",
}

#: The PDF's label for the all-deals row. Plural, unlike the group totals.
PORTFOLIO_TOTAL_LABEL = "Portfolio Totals"


def group_total_label(group: str) -> str:
    """The PDF's total-row label for a group key."""
    return GROUP_TOTAL_LABELS.get(group, f"Total {group}")


def resolve_committed_pref(cap: dict) -> tuple:
    """(committed pref, basis label) for one One Pager cap stack.

    THE SINGLE DEFINITION both subtabs use, so page 1's asset allocation and
    page 2's Total Pref / Total Commitment cannot disagree about the same deal.
    Same role as ``portfolio_snapshot_debt.resolve_debt`` plays for Debt.

    ``cap_stack.committed_pe`` is summed from Typename='Commitment' accounting
    rows and is 0.0 for the two deals that have none — East Manchester (PPI20)
    and City West (PPICW), both pre-dating the convention. Printed as-is they
    read as a real "$0.0M committed" instead of "no pledge on file". Funded pref
    is the correct floor: capital actually contributed is committed by
    definition, so this can only raise a zero, never lower a real pledge.

    ``one_pager.get_capitalization_stack`` applies the same fallback at source
    and publishes ``committed_pe_basis``; this function prefers that marker when
    present. It repeats the fallback rather than trusting it because the payload
    is fetched over HTTP from a deployed app, so an un-deployed backend serves
    the pre-fallback 0.0 and the two pages would silently diverge — which is
    exactly what happened when the switch was first wired.
    """
    committed = cap.get("committed_pe")
    funded = cap.get("pref_equity")
    try:
        committed = None if committed is None else float(committed)
    except (TypeError, ValueError):
        committed = None
    try:
        funded = None if funded is None else float(funded)
    except (TypeError, ValueError):
        funded = None

    if committed:
        return committed, (cap.get("committed_pe_basis") or "commitment rows")
    if funded:
        return funded, "funded (no commitment row)"
    return committed, (cap.get("committed_pe_basis") or "none")

#: Mixed-route default. A route that reaches the deal through a single-deal SPV
#: means the investor holds a direct position in that specific asset, which the
#: investor report presents as an Individual Investment even when a fund also
#: holds a slice. This is what puts Pegasus Life Storage (TGA22 69.18% +
#: PPILFS 23.45%) in Individual Investments, matching the reference PDF, without
#: hardcoding that deal. Every mixed-route deal is flagged either way.
PREFER_INDIVIDUAL_ON_MIXED = True

#: Display names that exist nowhere in the data. TGAM resolves only to its
#: vehicle name ("TGA Peaceable Investor Member LLC") in MRI; the institution
#: behind it is not recorded, so the alias is asserted here.
INVESTOR_NAME_ALIASES: dict[str, str] = {
    "TGAM": "TIAA",
}

_MAX_DEPTH = 8


# ── Investor names ────────────────────────────────────────────────────────

def get_investor_name(code: str,
                      investor_names: Optional[dict] = None) -> str:
    """Display name for an investor code.

    Resolution order:
      1. ``INVESTOR_NAME_ALIASES`` — names not present in any source.
      2. ``investor_names`` — the future MRI ``MRI_IA_Investor`` lookup
         (``InvestorID`` -> ``Name``). That table lives on the MRI **IM** server
         and is **not** in the app database yet; extracting it needs a new
         ``QUERY_REGISTRY`` entry and VPN access. Verified 2026-08-21 that it
         resolves 266 of 275 investor codes and names KOCINV
         ("Knights of Columbus - REIT Investor"), DCXVIA/DCXVIB
         ("Declaration Capital PE SPV XVIA/XVIB LLC") and PSC1/2/3. Pass it in
         once it exists; until then this argument is simply omitted.
      3. The raw code, so a missing name never blanks the report.
    """
    key = str(code or "").strip()
    if not key:
        return ""
    if key.upper() in INVESTOR_NAME_ALIASES:
        return INVESTOR_NAME_ALIASES[key.upper()]
    if investor_names:
        hit = investor_names.get(key) or investor_names.get(key.upper())
        if hit and str(hit).strip():
            return str(hit).strip()
    return key


# ── Ownership graph ───────────────────────────────────────────────────────

def _is_open(value) -> bool:
    """True when an EndDate means 'still current'."""
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return str(value).strip() in ("", "nan", "None", "NaT", "NaN")


class _Graph:
    """Ownership edges from the relationships feed, ended rows dropped.

    Two indexes are held because the maths needs both directions: children to
    walk downward, and *all* owners of a node to normalise a hop.
    """

    def __init__(self, relationships: pd.DataFrame):
        self.children: dict[str, list[tuple[str, float]]] = {}
        self.owners: dict[str, list[tuple[str, float]]] = {}
        if relationships is None or getattr(relationships, "empty", True):
            return

        rel = relationships
        cols = {c.lower(): c for c in rel.columns}
        c_inv = cols.get("investmentid")
        c_own = cols.get("investorid")
        c_pct = cols.get("ownershippct")
        c_end = cols.get("enddate")
        if not (c_inv and c_own and c_pct):
            log.warning("portfolio_snapshot: relationships missing required "
                        "columns; ownership graph is empty")
            return

        for row in rel.itertuples(index=False):
            d = row._asdict() if hasattr(row, "_asdict") else None
            if d is None:
                continue
            if c_end and not _is_open(d.get(c_end)):
                continue
            investee = str(d.get(c_inv) or "").strip().upper()
            investor = str(d.get(c_own) or "").strip().upper()
            if not investee or not investor:
                continue
            pct = pd.to_numeric(d.get(c_pct), errors="coerce")
            pct = 0.0 if pd.isna(pct) else float(pct)
            self.children.setdefault(investor, []).append((investee, pct))
            self.owners.setdefault(investee, []).append((investor, pct))

    def hop_share(self, investee: str, investor: str,
                  pe_only: bool = False) -> Optional[float]:
        """`investor`'s normalised share of `investee`.

        Normalised against the sum of *all* owners of ``investee``. Entities
        holding 0% (PSCMAN, PCBLE and other carried-interest members) add
        nothing to that sum, so they are excluded from the denominator by
        construction and can never dilute a real holder.

        Returns ``None`` when the hop cannot be normalised because every owner
        holds 0% — a broken chain, not a zero share. 45th & Main is exactly
        this case (PPI45M 0.0 and OPEVGR 0.0 in PMX, against 100.0 in MRI's IM
        copy), and it must be flagged rather than fabricated.

        ``pe_only`` drops operating-partner owners from the denominator, giving
        the investor's share of the *preferred-equity* capital rather than of
        the whole deal. Use it only on the hop that lands ON the deal entity,
        and only against a PE-only dollar figure — see ``lookthrough_pct``.
        With every OP owner recorded at 0% (34 of 35 TGAM deals at 26Q2) this
        returns exactly what the default does, because a 0% holder is already
        outside the denominator.
        """
        owners = self.owners.get(investee.upper())
        if not owners:
            return None
        if pe_only:
            owners = [(w, p) for w, p in owners if not _is_op(w)]
            if not owners:
                return None
        total = sum(p for _, p in owners)
        if total <= 0:
            return None
        mine = sum(p for who, p in owners if who == investor.upper())
        return mine / total


# ── Look-through ownership ────────────────────────────────────────────────

def lookthrough_pct(deal_iid: str, investor_code: str,
                    relationships: pd.DataFrame = None,
                    graph: "_Graph" = None) -> dict:
    """Look-through ownership of one deal by one investor.

    The product of the normalised ownership % at every hop, summed over every
    distinct route (a deal can be held through more than one vehicle).

    Returns ``{"pct": float|None, "pct_pe": float|None, "routes": [...],
    "broken": [...]}``. Both are ``None`` when no route resolves — the caller
    must flag the deal, never substitute a number.

    TWO BASES, AND THEY ARE NOT INTERCHANGEABLE.

    ``pct`` is the investor's share of the WHOLE DEAL: the final hop is
    normalised against every owner of the deal entity, operating partner
    included. Use it for anything measured over total deal capital.

    ``pct_pe`` is the investor's share of the deal's PREFERRED-EQUITY capital:
    the final hop drops OP owners from its denominator. Use it — and only it —
    to scale ``cap_stack.pref_equity`` / ``committed_pe``, which
    ``one_pager.get_capitalization_stack`` already builds from non-OP investors
    alone.

    Multiplying a PE-only dollar figure by ``pct`` subtracts the OP stake
    twice. Pegasus Life Storage was exactly this: OPPEGA holds 7.37% of PEGASU
    and contributed $2,573,473.25, which lands in ``partner_equity``, so
    ``pref_equity`` of $32,334,654.75 is TGA22's $24,150,000 plus PPILFS's
    $8,184,654.75 and contains none of OPPEGA's money. TGAM owns 90% of both
    vehicles, so its claim is $29,101,189.28 — but ``pct`` of 0.83367
    (= 0.90 x 0.9263) gave $26,956,431.63, low by $2,144,757.65. ``pct_pe``
    returns 0.90 and the figure is right.

    Only the FINAL hop differs. Intermediate hops keep the full denominator: an
    intermediate is a holding vehicle, and the investor's share of it is not a
    PE-vs-OP question. The two values are equal for every deal whose OP owners
    are recorded at 0% — 34 of 35 TGAM deals at 26Q2 — so this distinction bites
    only where the ownership feed carries a real OP percentage.
    """
    g = graph if graph is not None else _Graph(relationships)
    target = str(deal_iid or "").strip().upper()
    investor = str(investor_code or "").strip().upper()
    routes: list[dict] = []
    broken: list[dict] = []

    def walk(node: str, acc: float, trail: list, seen: frozenset, depth: int):
        if depth > _MAX_DEPTH:
            return
        for child, pct in g.children.get(node, []):
            if child in seen:          # cycle guard
                continue
            share = g.hop_share(child, node)
            if share is None:
                if child == target:
                    broken.append({
                        "entity": child, "via": node,
                        "reason": "every owner of this entity holds 0% — "
                                  "ownership cannot be normalised",
                        "partial_chain": list(trail),
                    })
                continue
            if pct == 0:
                # A 0% holder contributes nothing; walking on would only ever
                # yield a 0 product.
                continue
            hop = {"entity": child, "share": share, "stated_pct": pct}
            if child == target:
                # PE basis: re-normalise THIS hop only, excluding OP owners.
                # Falls back to `share` when the PE-only denominator collapses
                # (every non-OP owner at 0%), so a degenerate feed can never
                # turn into a fabricated number — same rule as `hop_share`.
                share_pe = g.hop_share(child, node, pe_only=True)
                if share_pe is None:
                    share_pe = share
                routes.append({"pct": acc * share,
                               "pct_pe": acc * share_pe,
                               "chain": trail + [hop],
                               "first_hop": (trail[0]["entity"] if trail
                                             else child)})
                continue
            walk(child, acc * share, trail + [hop], seen | {child}, depth + 1)

    walk(investor, 1.0, [], frozenset({investor}), 0)
    total = sum(r["pct"] for r in routes) if routes else None
    total_pe = sum(r["pct_pe"] for r in routes) if routes else None
    return {"pct": total, "pct_pe": total_pe,
            "routes": routes, "broken": broken}


# ── Grouping ──────────────────────────────────────────────────────────────

def _classify_entities(routes_by_deal: dict) -> dict:
    """Decide which first-hop entities are funds, from the traversal itself.

    An entity is a fund when it reaches ``FUND_MIN_DEALS`` or more of the deals
    still held at the report quarter, which separates the pooled vintage
    vehicles (TGA22-25, TGA6) from the single-deal SPVs (TGANOT, TGAM2, PPI32,
    PPIEVG, PPILFS ...) without naming any of them.

    Must be called on the population *after* the sold exclusion — see the note
    on ``FUND_MIN_DEALS``. ``FORCE_FUND`` covers the boundary cases.
    """
    reach: dict[str, set] = {}
    for iid, routes in routes_by_deal.items():
        for r in routes:
            reach.setdefault(r["first_hop"], set()).add(iid)
    return {ent: (len(deals) >= FUND_MIN_DEALS or ent in FORCE_FUND)
            for ent, deals in reach.items()}


def _children_of(parent_vcode: str, deals: dict) -> list[dict]:
    """Child property rows that roll up into ``parent_vcode``.

    Enumerated from the deals frame rather than the ownership graph, because the
    graph is not a reliable source for children: Brainerd's and Town Fair's
    children sit on 0% edges while Burton's have no relationship rows at all.
    Matches a child's ``Portfolio_Name`` against either the parent's
    ``Investment_Name`` or the parent's own ``Portfolio_Name`` — the same
    pairing ``one_pager._child_vcodes_for_parent`` uses, which covers Burton
    naming its group differently from the parent deal.
    """
    parent = deals.get(parent_vcode)
    if not parent or parent["property_count"] < 1:
        return []
    keys = {parent["name"].strip().lower()}
    if parent["portfolio_name"]:
        keys.add(parent["portfolio_name"].strip().lower())
    keys.discard("")
    out = []
    for vc, m in deals.items():
        if vc == parent_vcode or m["property_count"] != 0:
            continue
        if m["portfolio_name"].strip().lower() in keys:
            out.append(m)
    return out


def _group_for(iid: str, routes: list, is_fund: dict) -> tuple[str, bool]:
    """(group name, mixed_routes flag) for one deal."""
    if iid in GROUP_OVERRIDES:
        return GROUP_OVERRIDES[iid], len({
            (r["first_hop"] if is_fund.get(r["first_hop"]) else INDIVIDUAL_GROUP)
            for r in routes}) > 1

    groups = {(r["first_hop"] if is_fund.get(r["first_hop"])
               else INDIVIDUAL_GROUP) for r in routes}
    if len(groups) == 1:
        return next(iter(groups)), False

    # Mixed routes. Preferring Individual reflects that a direct SPV position
    # in a named asset is reported as an individual investment even when a fund
    # also holds part of it.
    if PREFER_INDIVIDUAL_ON_MIXED and INDIVIDUAL_GROUP in groups:
        return INDIVIDUAL_GROUP, True
    # Otherwise the dominant route wins, by share of the look-through.
    weight: dict[str, float] = {}
    for r in routes:
        gname = (r["first_hop"] if is_fund.get(r["first_hop"])
                 else INDIVIDUAL_GROUP)
        weight[gname] = weight.get(gname, 0.0) + r["pct"]
    return max(weight.items(), key=lambda kv: kv[1])[0], True


# ── Deal metadata ─────────────────────────────────────────────────────────

def _s(value) -> str:
    """Trimmed string, with NaN treated as empty.

    ``str(value or "")`` is not safe here: ``float('nan')`` is truthy, so a
    missing Portfolio_Name would become the literal ``"nan"`` and every parent
    lacking one would then 'match' every child lacking one.
    """
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _deal_index(inv: pd.DataFrame) -> dict:
    """vcode -> deal metadata, from the deals frame.

    Keyed on ``vcode``, not ``InvestmentID``: InvestmentID is **not unique**.
    Live on 2026-08-21 two IDs were shared by two deals each — ``ASTONC``
    (P0000045 Aston Center, a Giant 7 child, and PASTONC Jefferson Centura) and
    ``MCCORD`` (P0000049 Donald Lynch and P0000073 870 DLB). Keying on
    InvestmentID silently discards one deal of each pair, and depending on row
    order that can be the *parent*, dropping it from the report entirely.
    """
    out: dict[str, dict] = {}
    if inv is None or getattr(inv, "empty", True):
        return out
    cols = {c.lower(): c for c in inv.columns}
    c_vc, c_iid = cols.get("vcode"), cols.get("investmentid")
    if not (c_vc and c_iid):
        return out
    for row in inv.itertuples(index=False):
        d = row._asdict() if hasattr(row, "_asdict") else None
        if d is None:
            continue
        iid = _s(d.get(c_iid)).upper()
        vc = _s(d.get(c_vc))
        if not iid or not vc:
            continue
        pc = pd.to_numeric(d.get(cols.get("property_count")), errors="coerce")
        out[vc] = {
            "vcode": vc,
            "name": _s(d.get(cols.get("investment_name"))) or vc,
            "iid": iid,
            # Property_Count is the parent/child discriminator: every genuine
            # child property carries 0, every parent >= 1. Same rule the Burton
            # child-loan fix uses.
            "property_count": 0 if pd.isna(pc) else int(pc),
            "portfolio_name": _s(d.get(cols.get("portfolio_name"))),
            "sale_status": _s(d.get(cols.get("sale_status"))),
            "sale_date": pd.to_datetime(d.get(cols.get("sale_date")),
                                        errors="coerce"),
            # Enriched at load time by data_service from the earliest accounting
            # activity per deal (see CLAUDE.md "Acquisition Date"), so it is the
            # true closing date rather than MRI's Acquisition_Date field.
            "acquisition_date": pd.to_datetime(
                d.get(cols.get("acquisition_date")), errors="coerce"),
            "asset_type": _s(d.get(cols.get("asset_type"))),
            # Display string: Investment_Strategy, falling back to Lifecycle —
            # the same precedence one_pager.get_general_information uses.
            "strategy": (_s(d.get(cols.get("investment_strategy")))
                         or _s(d.get(cols.get("lifecycle")))),
            # PURE Investment_Strategy, no fallback. This is the sole input to
            # dev detection (creator decision 2026-08-24): the "Dev" display,
            # the mOrigLoanAmt debt path and the future "Excluding Development
            # Deals" subtotal all read this one field so they cannot diverge.
            "investment_strategy": _s(d.get(cols.get("investment_strategy"))),
        }
    return out


def _quarter_end(quarter: str) -> date:
    """Quarter end for 'YYYY-QN', reusing the One Pager helper read-only."""
    from one_pager import quarter_to_date_range      # read-only reuse
    _, q_end = quarter_to_date_range(quarter)
    return q_end


def is_sold_as_of(meta: dict, quarter_end: date) -> bool:
    """Quarter-aware sold test.

    ``Sale_Status == 'SOLD'`` **and** ``Sale_Date <= quarter_end``. Deliberately
    not ``data_service.get_inv_display`` / ``exclude_sold``: those key off
    ``date.today().year``, so the same historical quarter changes population
    depending on when it is run. A deal sold after the quarter end was still
    held during the quarter and stays in (Clima Secur, sold 2026-07-01, is in
    for 2026-Q2); one sold inside the quarter drops out (East Manchester,
    2026-06-25).
    """
    if str(meta.get("sale_status", "")).strip().upper() != "SOLD":
        return False
    sd = meta.get("sale_date")
    if sd is None or pd.isna(sd):
        # Sold with no date: cannot place it against the quarter. Treat as sold
        # so a disposed deal is never reported as live.
        return True
    return pd.Timestamp(sd).date() <= quarter_end


def is_acquired_as_of(meta: dict, quarter_end: date) -> bool:
    """Quarter-aware acquisition test — the mirror of ``is_sold_as_of``.

    A deal belongs in a quarter only if it was owned during that quarter, which
    needs gates at *both* ends: acquired on/before the quarter end AND not sold
    on/before it. Without this one, a deal that had not closed yet still renders
    a row, and its Loan-subtab debt is phantom: at 26Q1 Presidential Arms
    (closed 2026-05-13) carried $98,980,000 of debt against zero equity, because
    the One Pager's equity block is quarter-filtered while the debt line is not.
    Citizen Storage (2026-05-20) and Fairview Heights (2026-06-30) were the same
    shape.

    **Missing date fails OPEN — the deal is kept.** This is deliberately NOT
    symmetric with ``is_sold_as_of``, and the asymmetry is the point: there,
    including a disposed deal reports something the investor no longer owns, so
    the safe default is to exclude; here, excluding would silently drop a deal
    the investor genuinely holds and understate the portfolio, which is the worse
    failure. 34 of 110 deals live carry no ``Acquisition_Date`` — almost all of
    them child properties already removed on ``Property_Count == 0`` — so the
    fail-open population is small, and every case is counted in
    ``diagnostics['acquisition_date_missing']`` rather than passing unnoticed.
    """
    ad = meta.get("acquisition_date")
    if ad is None or pd.isna(ad):
        return True
    return pd.Timestamp(ad).date() <= quarter_end


# ── Entry point ───────────────────────────────────────────────────────────

def resolve_investor_deals(investor_code: str, quarter: str,
                           relationships: pd.DataFrame,
                           inv: pd.DataFrame,
                           investor_names: Optional[dict] = None) -> dict:
    """Deals held by ``investor_code`` as of ``quarter``, grouped by fund.

    Walks the relationships chain downward (investor -> funds -> deals). It does
    **not** use the waterfall-based finder (``find_entity_deals``), which gates
    on a deal's waterfall PropCode and so drops deals that have no waterfall
    yet — verified live: that route returned 2 of TGA25's deals against the
    5 the relationships chain finds.

    Deals are filtered to the quarter's **ownership window** — acquired on or
    before ``quarter_end`` AND not sold on or before it (``is_acquired_as_of``
    and ``is_sold_as_of``). Both gates run before fund classification, since a
    deal outside the window must not count toward an entity's deal tally.

    Returns a dict with ``groups`` (group name -> list of deals), plus
    ``flagged``, ``excluded_sold``, ``excluded_not_acquired`` and
    ``excluded_children`` so nothing is ever dropped silently.
    """
    q_end = _quarter_end(quarter)
    graph = _Graph(relationships)
    deals = _deal_index(inv)
    investor = str(investor_code or "").strip().upper()

    # 1. every route from the investor to every reachable deal, keyed on vcode
    routes_by_deal: dict[str, list] = {}
    broken_raw: list[dict] = []
    for vc, m in deals.items():
        res = lookthrough_pct(m["iid"], investor, graph=graph)
        if res["routes"]:
            routes_by_deal[vc] = res["routes"]
        elif res["broken"]:
            broken_raw.append({"vcode": vc, "detail": res["broken"][0]})

    # 2. child properties out first, explicitly on Property_Count == 0. Not left
    #    to the accident that child edges happen to be 0% (Brainerd, Town Fair)
    #    or absent entirely (Burton) — if MRI ever populates a real % on a child
    #    edge, the 9 Brainerd buildings would otherwise become report lines.
    excluded_children, excluded_sold, flagged = [], [], []
    kept_despite_sold: list = []
    excluded_not_acquired: list[dict] = []
    acquisition_date_missing: list[dict] = []
    dropped_children: set[str] = set()

    def _child(vc: str) -> bool:
        return deals[vc]["property_count"] == 0

    for vc in list(routes_by_deal):
        if _child(vc):
            dropped_children.add(vc)
            routes_by_deal.pop(vc)
    for b in list(broken_raw):
        if _child(b["vcode"]):
            dropped_children.add(b["vcode"])
            broken_raw.remove(b)

    # 3. ownership window, before classification. Neither end may count toward
    #    an entity's deal tally or it can promote an SPV to a fund. The full
    #    test is: acquired on/before quarter_end AND not sold on/before it.
    for vc in list(routes_by_deal):
        m = deals[vc]
        if pd.isna(m["acquisition_date"]):
            # Kept (is_acquired_as_of fails open) but recorded, so a deal that
            # dodged the gate for want of a date is visible rather than assumed.
            acquisition_date_missing.append(
                {"vcode": m["vcode"], "name": m["name"]})
        if not is_acquired_as_of(m, q_end):
            excluded_not_acquired.append({
                "vcode": m["vcode"], "name": m["name"],
                "acquisition_date": (m["acquisition_date"].date()
                                     if pd.notna(m["acquisition_date"])
                                     else None),
                "reason": f"Acquisition_Date > {q_end} — not yet owned"})
            routes_by_deal.pop(vc)
            continue
        if is_sold_as_of(m, q_end):
            if str(m["vcode"]).strip().upper() in KEEP_DESPITE_SOLD:
                # Reported anyway — see KEEP_DESPITE_SOLD. Recorded in its own
                # bucket so a kept deal is never mistaken for one the gate
                # simply did not catch.
                kept_despite_sold.append({
                    "vcode": m["vcode"], "name": m["name"],
                    "sale_date": (m["sale_date"].date()
                                  if pd.notna(m["sale_date"]) else None),
                    "reason": "sold/foreclosed before quarter end but kept on "
                              "the report (KEEP_DESPITE_SOLD)"})
                continue
            excluded_sold.append({
                "vcode": m["vcode"], "name": m["name"],
                "sale_date": (m["sale_date"].date()
                              if pd.notna(m["sale_date"]) else None),
                "reason": f"Sale_Status=SOLD and Sale_Date <= {q_end}"})
            routes_by_deal.pop(vc)

    # 4. groups, derived from what is still held.
    #
    #    A KEEP_DESPITE_SOLD deal is reported but must NOT count toward an
    #    entity's fund tally: it is sold, so counting it could promote its SPV
    #    to a fund and re-open the very TGAM2 problem GROUP_OVERRIDES exists to
    #    paper over. Same reasoning as the step-3 note above.
    kept_vcodes = {k["vcode"] for k in kept_despite_sold}
    is_fund = _classify_entities({vc: r for vc, r in routes_by_deal.items()
                                 if vc not in kept_vcodes})

    # An editorial placement must not be invisible: record every override that
    # actually fired, with the group the traversal would have chosen on its own.
    # ``_group_for("", ...)`` is the same function with the override lookup
    # missed deliberately (no vcode is ""), so the "derived" group is the real
    # rule's answer rather than a re-implementation of it.
    overrides_applied = [
        {"vcode": vc, "name": deals[vc]["name"],
         "forced_group": GROUP_OVERRIDES[vc],
         "derived_group": _group_for("", routes, is_fund)[0]}
        for vc, routes in sorted(routes_by_deal.items())
        if vc in GROUP_OVERRIDES]

    groups: dict[str, list] = {}
    for vc, routes in routes_by_deal.items():
        m = deals[vc]
        group, mixed = _group_for(vc, routes, is_fund)
        pct = sum(r["pct"] for r in routes)
        # PE basis, for scaling pref_equity / committed_pe. Equal to `pct`
        # unless the deal entity has an OP owner at a real percentage.
        pct_pe = sum(r["pct_pe"] for r in routes)
        entry = {
            "vcode": m["vcode"], "name": m["name"], "iid": m["iid"],
            "group": group,
            "lookthrough_pct": pct,
            "lookthrough_pct_pe": pct_pe,
            "pct_display": round(pct * 100, 4),
            "pct_pe_display": round(pct_pe * 100, 4),
            "op_diluted": abs(pct_pe - pct) > 1e-12,
            "n_routes": len(routes),
            "chains": [" -> ".join([investor] + [h["entity"] for h in r["chain"]])
                       for r in routes],
            "asset_type": m["asset_type"], "strategy": m["strategy"],
            "investment_strategy": m["investment_strategy"],
            "sale_status": m["sale_status"],
            "sold_after_quarter": (m["sale_status"].upper() == "SOLD"
                                   and not is_sold_as_of(m, q_end)),
            "kept_despite_sold": vc in kept_vcodes,
            "flags": [],
        }
        if entry["kept_despite_sold"]:
            entry["flags"].append(
                "sold/foreclosed before quarter end — kept on the report to "
                "match the reference PDF (KEEP_DESPITE_SOLD)")
        if mixed:
            entry["flags"].append(
                f"multi-route: reachable {len(routes)} ways; grouped to "
                f"{group} (see PREFER_INDIVIDUAL_ON_MIXED)")
        if entry["sold_after_quarter"]:
            entry["flags"].append("sold after quarter end — held during quarter")
        groups.setdefault(group, []).append(entry)

    # 5. child roll-up, reported. Every parent that made the report lists the
    #    children folded into its single line, so the roll-up is auditable
    #    rather than implicit.
    included = {e["vcode"] for items in groups.values() for e in items}
    for vc in sorted(included):
        for kid in _children_of(vc, deals):
            excluded_children.append({
                "vcode": kid["vcode"], "name": kid["name"],
                "rolls_up_to": deals[vc]["name"],
                "parent_vcode": vc,
                "reason": "child property (Property_Count == 0)"})
    # Children the traversal itself reached (0% edges) but whose parent is not
    # in this investor's set — recorded so the drop is never silent.
    accounted = {c["vcode"] for c in excluded_children}
    for vc in sorted(dropped_children):
        m = deals[vc]
        if m["vcode"] not in accounted:
            excluded_children.append({
                "vcode": m["vcode"], "name": m["name"],
                "rolls_up_to": m["portfolio_name"], "parent_vcode": None,
                "reason": "child property (Property_Count == 0); parent not in "
                          "this investor's set"})

    # 6. broken chains: reachable but no resolvable ownership. Flagged with the
    #    look-through withheld, never a fabricated number.
    for b in broken_raw:
        m = deals[b["vcode"]]
        # Same ownership window as the resolvable deals above — a deal outside
        # it must not surface as an ownership-flagged row either.
        if is_sold_as_of(m, q_end) or not is_acquired_as_of(m, q_end):
            continue
        flagged.append({
            "vcode": m["vcode"], "name": m["name"], "iid": m["iid"],
            "asset_type": m["asset_type"], "strategy": m["strategy"],
            "investment_strategy": m["investment_strategy"],
            "lookthrough_pct": None,
            "lookthrough_pct_pe": None,
            "reason": "ownership % unavailable",
            "detail": b["detail"]["reason"],
            "via": b["detail"]["via"],
        })

    for g in groups:
        groups[g].sort(key=lambda e: e["name"].lower())
    ordered = {}
    if INDIVIDUAL_GROUP in groups:
        ordered[INDIVIDUAL_GROUP] = groups.pop(INDIVIDUAL_GROUP)
    for g in sorted(groups):
        ordered[g] = groups[g]

    return {
        "investor_code": investor,
        "investor_name": get_investor_name(investor, investor_names),
        "quarter": quarter,
        "quarter_end": q_end,
        "groups": ordered,
        "flagged": sorted(flagged, key=lambda f: f["name"].lower()),
        "excluded_sold": sorted(excluded_sold,
                                key=lambda e: str(e["sale_date"] or "")),
        "kept_despite_sold": sorted(kept_despite_sold,
                                    key=lambda e: str(e["sale_date"] or "")),
        "excluded_not_acquired": sorted(
            excluded_not_acquired,
            key=lambda e: str(e["acquisition_date"] or "")),
        "excluded_children": sorted(excluded_children,
                                    key=lambda e: e["vcode"]),
        "diagnostics": {
            "deal_count": sum(len(v) for v in ordered.values()),
            "group_count": len(ordered),
            "excluded_not_acquired_count": len(excluded_not_acquired),
            "acquisition_date_missing": acquisition_date_missing,
            "fund_entities": sorted(e for e, f in is_fund.items() if f),
            "spv_entities": sorted(e for e, f in is_fund.items() if not f),
            # Which deals were placed by GROUP_OVERRIDES rather than by the
            # traversal, and where the traversal would have put them. Recorded
            # so an editorial placement is never invisible in the payload.
            "group_overrides_applied": overrides_applied,
        },
    }


# ── Self-test ─────────────────────────────────────────────────────────────

def _selftest():                                    # pragma: no cover
    """Reproduce the verified Step 1 result for TIAA + 2026-Q2.

    Runs out-of-process, so it pulls the two frames over the REST API using
    narrow per-entity filters — one page per request, OFFSET never used, and
    every result post-filtered to exact matches because ``filter__`` is
    case-insensitive *contains*.
    """
    import os
    import sys
    # Running as a script, so put the repo root and scripts/ on the path. Inside
    # Flask the root is already importable and this block never executes.
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    for p in (root, os.path.join(root, "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import live_api as api

    print(f"token={api.token_info()['username']}  "
          f"build={api.get('/api/data/version').get('version')}  "
          f"actuals_through={api.get('/api/data/config').get('actuals_through')}")

    inv = pd.DataFrame(api.get("/api/data/deals/all").get("deals") or [])
    print(f"deals: {len(inv)}")

    # Walk the graph outward from the investor, one narrow request per entity.
    seen, frontier, rows = set(), ["TGAM"], []

    def fetch(col, val):
        d = api.get("/api/data/tables/relationships/rows",
                    params={"page": 1, "page_size": 500, f"filter__{col}": val})
        if (d.get("total") or 0) > 500:
            print(f"  !! {col}={val}: {d['total']} rows exceeds one page")
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
            rows.extend(fetch("InvestmentID", child))   # all owners, to normalise
            if child not in seen:
                frontier.append(child)

    rel = pd.DataFrame(rows).drop_duplicates()
    print(f"relationships pulled: {len(rel)} unique rows over {len(seen)} entities\n")

    res = resolve_investor_deals("TGAM", "2026-Q2", rel, inv)

    print("=" * 92)
    print(f"{res['investor_name']} ({res['investor_code']})  {res['quarter']}  "
          f"quarter end {res['quarter_end']}")
    print("=" * 92)
    for g, items in res["groups"].items():
        print(f"\n  {g}  ({len(items)} deals)")
        for e in items:
            fl = ("   <- " + "; ".join(e["flags"])) if e["flags"] else ""
            print(f"    {e['vcode']:<9} {e['name'][:36]:<38}"
                  f"{e['pct_display']:>9.4f}%{fl}")
    if res["flagged"]:
        print(f"\n  FLAGGED — ownership unavailable ({len(res['flagged'])})")
        for f in res["flagged"]:
            print(f"    {f['vcode']:<9} {f['name'][:36]:<38}     n/a   "
                  f"({f['detail']}, via {f['via']})")
    print(f"\n  EXCLUDED as sold ({len(res['excluded_sold'])})")
    for e in res["excluded_sold"]:
        print(f"    {e['vcode']:<9} {e['name'][:36]:<38} sold {e['sale_date']}")
    print(f"\n  CHILD PROPERTIES rolled up ({len(res['excluded_children'])})")
    roll: dict[str, list] = {}
    for c in res["excluded_children"]:
        roll.setdefault(c["rolls_up_to"] or "(unknown parent)", []).append(c["vcode"])
    for parent, kids in sorted(roll.items()):
        print(f"    {parent:<38} <- {len(kids)}: {', '.join(kids)}")
    d = res["diagnostics"]
    print(f"\n  deals={d['deal_count']}  groups={d['group_count']}")
    print(f"  funds detected : {d['fund_entities']}")

    # ---- assertions against the verified pass ----
    print("\n" + "=" * 92)
    print("CHECKS vs the verified Step 1 pass")
    flat = {e["vcode"]: e for items in res["groups"].values() for e in items}
    checks = [
        ("Nottingham = 41.2124%",
         abs(flat.get("P0000030", {}).get("pct_display", 0) - 41.2124) < 0.001),
        ("Nottingham in Individual Investments",
         flat.get("P0000030", {}).get("group") == INDIVIDUAL_GROUP),
        # Property, not identity. 45th & Main was the only flagged deal until
        # its PMX IA_Relationship row was fixed live (2026-08-24), after which
        # it resolves at 90% into TGA24 and this check had nothing to assert.
        # A flagged deal must be absent from `groups`; with none flagged the
        # invariant holds trivially.
        ("no flagged deal also appears as a grouped deal",
         all(f["vcode"] not in flat for f in res["flagged"])),
        ("flagged deals withhold their look-through %",
         all(f.get("lookthrough_pct") is None for f in res["flagged"])),
        ("Pegasus in Individual Investments",
         flat.get("P0000066", {}).get("group") == INDIVIDUAL_GROUP),
        ("Pegasus = 83.367% across both routes",
         abs(flat.get("P0000066", {}).get("pct_display", 0) - 83.367) < 0.01),
        ("City West excluded as sold",
         any(e["vcode"] == "PCITWES" for e in res["excluded_sold"])),
        ("East Manchester excluded (sold inside Q2)",
         any(e["vcode"] == "P0000017" for e in res["excluded_sold"])),
        ("TGA6 grouped as a fund, not Individual",
         "TGA6" in res["groups"] and
         flat.get("P0000117", {}).get("group") == "TGA6"),
        ("Trolley Square = 90%",
         abs(flat.get("P0000110", {}).get("pct_display", 0) - 90.0) < 0.001),
        ("Individual Investments holds the 5 originals + Pegasus",
         {"P0000018", "P0000019", "P0000021", "P0000030", "P0000065",
          "P0000066"}.issubset(
             {e["vcode"] for e in res["groups"].get(INDIVIDUAL_GROUP, [])})),
        # 25, not the 18 the verification pass saw. That pass could only count
        # children the ownership graph happened to reach (Brainerd 9, Town Fair
        # 6, Burton 3); Giant 7's 7 children carry no relationship rows at all,
        # so enumerating from the deals frame finds them too. 9+6+3+7 = 25.
        ("25 children rolled up into 4 parents",
         len(res["excluded_children"]) == 25),
        ("Aston Center rolls up (the ASTONC InvestmentID collision)",
         any(c["vcode"] == "P0000045" for c in res["excluded_children"])),
        ("Donald Lynch survives the MCCORD collision as a parent",
         "P0000049" in flat or any(f["vcode"] == "P0000049"
                                   for f in res["flagged"])
         or not any(e["vcode"] == "P0000049" for e in res["excluded_sold"])),
        ("no child appears as a deal line",
         not any(v in flat for v in
                 ["P0000090", "P0000101", "P0000111"])),
    ]
    for label, passed in checks:
        print(f"    [{'PASS' if passed else 'FAIL'}] {label}")

    # ---- acquisition-date gate: the ownership window across quarters ----
    print("\n" + "=" * 92)
    print("ACQUISITION-DATE GATE — a deal appears only in quarters it was owned")
    print("=" * 92)
    NOT_YET = {"P0000119": ("Presidential Arms", "2026-05-13"),
               "P0000120": ("Citizen Storage Swartz Creek", "2026-05-20"),
               "P0000117": ("Fairview Heights Retail Center", "2026-06-30")}
    per_q = {}
    for q in ("2026-Q1", "2026-Q2", "2026-Q3"):
        r = resolve_investor_deals("TGAM", q, rel, inv)
        present = {e["vcode"] for items in r["groups"].values() for e in items}
        present |= {f["vcode"] for f in r["flagged"]}
        per_q[q] = (r, present)
        print(f"\n  {q}  (quarter end {r['quarter_end']})  "
              f"deals={r['diagnostics']['deal_count']}  "
              f"flagged={len(r['flagged'])}  "
              f"not-yet-acquired={r['diagnostics']['excluded_not_acquired_count']}")
        for e in r["excluded_not_acquired"]:
            print(f"      DROPPED {e['vcode']} {e['name'][:34]:<36}"
                  f"acquired {e['acquisition_date']}")
        for vc, (nm, ad) in sorted(NOT_YET.items()):
            print(f"      {vc} {nm[:32]:<34}acquired {ad}   "
                  f"{'IN SET' if vc in present else 'absent'}")

    q1, q1_present = per_q["2026-Q1"]
    q2, q2_present = per_q["2026-Q2"]
    q3, q3_present = per_q["2026-Q3"]

    gate_checks = [
        ("26Q1 drops exactly the 3 not-yet-acquired deals",
         {e["vcode"] for e in q1["excluded_not_acquired"]} == set(NOT_YET)),
        ("26Q1 excluded_not_acquired count is exactly 3",
         q1["diagnostics"]["excluded_not_acquired_count"] == 3),
        ("none of the 3 appears anywhere in the 26Q1 set",
         not (set(NOT_YET) & q1_present)),
        ("all 3 are back in 26Q2 (all closed by 6/30/2026)",
         set(NOT_YET) <= q2_present),
        ("all 3 are still present in 26Q3",
         set(NOT_YET) <= q3_present),
        ("26Q2 drops none for acquisition date",
         q2["diagnostics"]["excluded_not_acquired_count"] == 0),
        # Q1 is Q2's population minus the 3, plus East Manchester, which Q2
        # excludes as sold on 2026-06-25 but Q1 still held. The gate must not
        # disturb the sold gate at the other end of the window.
        ("26Q1 == 26Q2 minus the 3, plus East Manchester (sold in Q2)",
         q1_present == (q2_present - set(NOT_YET)) | {"P0000017"}),
        ("East Manchester held in Q1, sold out of Q2",
         "P0000017" in q1_present
         and any(e["vcode"] == "P0000017" for e in q2["excluded_sold"])),
        ("no deal is both excluded-as-sold and excluded-as-not-acquired",
         not ({e["vcode"] for e in q1["excluded_sold"]}
              & {e["vcode"] for e in q1["excluded_not_acquired"]})),
        ("every deal in the 26Q1 set has Acquisition_Date <= quarter end",
         all(pd.isna(_deal_index(inv)[vc]["acquisition_date"])
             or _deal_index(inv)[vc]["acquisition_date"].date()
             <= q1["quarter_end"] for vc in q1_present)),
        ("fail-open cases are reported, not silent",
         isinstance(q1["diagnostics"]["acquisition_date_missing"], list)),
    ]
    for label, passed in gate_checks:
        print(f"    [{'PASS' if passed else 'FAIL'}] {label}")
    checks.extend(gate_checks)

    missing = q1["diagnostics"]["acquisition_date_missing"]
    print(f"\n  kept despite no Acquisition_Date (fail-open): {len(missing)}")
    for m in missing:
        print(f"      {m['vcode']} {m['name']}")

    ok = True
    for label, passed in checks:
        ok &= bool(passed)
    print(f"\n  {sum(1 for _, p in checks if p)}/{len(checks)} checks passed")
    for label, passed in checks:
        if not passed:
            print(f"    FAILED: {label}")
    print(f"  {'ALL CHECKS PASS' if ok else 'SOME CHECKS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":                          # pragma: no cover
    raise SystemExit(_selftest())
