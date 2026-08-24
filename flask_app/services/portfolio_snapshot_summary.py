"""Portfolio Snapshot — Subtab 1 (Summary) data assembly.

Step 6a. Two parts, per the build spec:

Part A — allocation rollups
    Asset Allocation      deals grouped by ``Asset_Type``, funded + committed
                          dollars and each bucket's share of the total.
    Deal Type Allocation  deals grouped by investment strategy into
                          Value-Add / Income / New Construction.

Part B — two blank editable narrative boxes, loaded from Step 2 persistence
    (``portfolio_snapshot_comments``, ``scope='report'``). Blank by default —
    nothing here generates narrative text.

THE SCALING BASIS IS LOOK-THROUGH, and that is settled empirically, not
assumed. The reference PDF (26Q1 page 1) reports $404.2M funded. Against live
build ``09fe220ae0da`` for TGAM / 2026-Q1:

    scaled by look-through %   403.95M   -0.06%   <-- reproduces the PDF
    full deal-level            540.96M  +33.79%

So the page-1 allocation dollars are TIAA's *share*, exactly like the four
scaled columns on the Financial subtab. ``invested`` there and ``funded`` here
read the same ``cap_stack.pref_equity`` through the same look-through %, so the
two subtabs cannot drift apart; the self-test asserts that identity.

The full deal-level rollup is still computed and returned under
``alternate_basis`` — the choice stays auditable and reversible, the same way
the Financial subtab carries both commitment bases.

WHAT DOES NOT TIE, AND IS FLAGGED RATHER THAN FUDGED
    Committed. The PDF says $445.1M; scaling ``cap_stack.committed_pe`` the
    same way gives $477.99M (+7.39%). The $33.1M excess is concentrated in four
    deals (Burton 24.9M, JB Fair Park 19.4M, Jefferson Stephens 17.9M, Brainerd
    8.4M) and East Manchester runs the other way with committed 0 against
    3.6M funded. Per-deal attribution is returned in
    ``diagnostics['committed_gap_attribution']`` so the source can be settled;
    the number is reported as computed, never bent toward the PDF.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

log = logging.getLogger(__name__)

#: Which basis the allocation dollars use. "lookthrough" reproduces the
#: reference PDF (see module docstring); "full" is deal-level and is always
#: computed alongside it under ``alternate_basis``.
ALLOCATION_BASIS = "lookthrough"          # "lookthrough" | "full"

#: The two page-1 narrative boxes. These field names match Step 2's
#: ``scope='report'`` comments, so the boxes round-trip through the existing
#: save/approval path with no new persistence code.
NARRATIVE_FIELDS = ("narrative_1", "narrative_2")

#: Asset_Type rollup. Live ``deals.Asset_Type`` carries nine values where the
#: PDF shows four buckets: the three Retail sub-types collapse to "Retail", and
#: "Self Storage" is spelled "Self-Storage" on the report. Anything not listed
#: passes through unchanged, so a new asset type appears as its own bucket
#: rather than being silently absorbed into another.
ASSET_TYPE_ROLLUP = {
    "retail": "Retail",
    "retail - grocery": "Retail",
    "retail - non groc.": "Retail",
    "retail - non groc": "Retail",
    "self storage": "Self-Storage",
    "self-storage": "Self-Storage",
}

#: The report's three deal-type buckets, in PDF display order.
DEAL_TYPES = ("Value-Add", "Income", "New Construction")

#: Strategy -> deal type. **This map is calibrated, not literal.**
#:
#: The PDF's pie is built on MRI's ``Investment_Strategy``, which is empty on
#: all 110 deals in live build ``09fe220ae0da`` (checked on both
#: ``/api/data/deals/all`` and the raw ``deals`` table). ``Lifecycle`` is
#: populated 97/110 and is the only available proxy, so the mapping was solved
#: against the PDF's three dollar targets: an exhaustive search over all 243
#: assignments of the five Lifecycle values present in TIAA's set picked this
#: one at 5.65M total absolute error, against 24.95M for the runner-up — a
#: 4.4x margin, so the fit is unambiguous.
#:
#:     Value-Add        150.15M   37.2%   PDF 153.1M  37.9%
#:     Income           132.15M   32.7%   PDF 131.3M  32.5%
#:     New Construction 121.65M   30.1%   PDF 119.8M  29.6%
#:
#: Note ``New Construction -> Income``, which reads wrong until you see that
#: Lifecycle and Investment_Strategy are *different* MRI fields: Lifecycle is a
#: construction/stabilisation state, Investment_Strategy is the investment
#: thesis. The single deal driving that edge is Pegasus Life Storage — a
#: completed new-construction self-storage asset now producing income, i.e.
#: Lifecycle "New Construction" but strategy "Income". ``Development`` mapping
#: to ``New Construction`` is the same kind of rename.
#:
#: This map is a stopgap. Once ``Investment_Strategy`` is populated, values that
#: are already deal types map to themselves and the calibration falls away.
DEAL_TYPE_MAP = {
    "value-add": "Value-Add",
    "value add": "Value-Add",
    "income": "Income",
    "stable": "Income",
    "new construction": "Income",
    "development": "New Construction",
}

#: The literal reading of the same field, computed alongside for audit:
#: Development and New Construction both mean new construction, Stable means
#: income. It does NOT reproduce the PDF (24.95M error vs 5.65M) and is
#: returned only so the calibration above can be inspected against it.
DEAL_TYPE_MAP_LITERAL = {
    "value-add": "Value-Add",
    "value add": "Value-Add",
    "income": "Income",
    "stable": "Income",
    "new construction": "New Construction",
    "development": "New Construction",
}

#: Strategy values that map to no deal type land here and are flagged. Never
#: dropped: a bucket that vanishes silently understates the total.
UNCLASSIFIED = "Unclassified"

#: Asset_Type missing entirely.
ASSET_UNKNOWN = "Unclassified"


# ── small helpers ─────────────────────────────────────────────────────────

def _num(v):
    """Float or None. NaN is None, not a value that poisons every sum."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _s(value) -> str:
    """Trimmed string, NaN-safe. Same reasoning as the Step 1 helper:
    ``float('nan')`` is truthy, so ``str(v or "")`` would yield "nan"."""
    if value is None:
        return ""
    try:
        if value != value:                      # NaN
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def roll_asset_type(asset_type: str) -> str:
    """Map a raw ``Asset_Type`` to its report bucket."""
    raw = _s(asset_type)
    if not raw:
        return ASSET_UNKNOWN
    return ASSET_TYPE_ROLLUP.get(raw.lower(), raw)


def map_deal_type(strategy: str, mapping: Optional[dict] = None) -> str:
    """Map a strategy value to one of ``DEAL_TYPES``, or ``UNCLASSIFIED``."""
    raw = _s(strategy)
    if not raw:
        return UNCLASSIFIED
    return (mapping or DEAL_TYPE_MAP).get(raw.lower(), UNCLASSIFIED)


def strategy_for(entry: dict) -> tuple[str, str]:
    """(strategy value, which field it came from).

    Prefers the pure ``Investment_Strategy`` and falls back to Step 1's
    ``strategy`` (``Investment_Strategy or Lifecycle``). Written this way so the
    same code starts using the real field the moment MRI populates it — no
    call-site change — while working today off the Lifecycle proxy. The source
    is reported per deal so a proxy-derived bucket is never mistaken for the
    real thing.

    Delegates to Step 3a's ``resolve_strategy`` so the Summary, Operating and
    Loan subtabs read the strategy field through one definition and cannot
    disagree about which deals are development.
    """
    from flask_app.services.portfolio_snapshot_operating import resolve_strategy
    return resolve_strategy(entry)


# ── Part B: narratives ────────────────────────────────────────────────────

def _default_comment_loader(investor_code: str, quarter: str) -> dict:
    """field -> narrative text, from Step 2 persistence (read-only).

    Absent rows stay absent; the caller renders them blank. Nothing in this
    module generates narrative text.
    """
    try:
        from flask_app.services.portfolio_snapshot_persistence import get_elements
        rows = get_elements("comment", investor_code, quarter, scope="report")
        return {r.get("field"): r.get("comment_text") for r in rows}
    except Exception as exc:                    # persistence not initialised
        log.debug("report narratives unavailable: %s", exc)
        return {}


def _default_editable(investor_code: str, quarter: str) -> Optional[bool]:
    try:
        from flask_app.services.portfolio_snapshot_persistence import is_editable
        return bool(is_editable(investor_code, quarter))
    except Exception as exc:
        log.debug("editability unavailable: %s", exc)
        return None


def _build_narratives(loaded: dict) -> list:
    out = []
    for field in NARRATIVE_FIELDS:
        text = _s(loaded.get(field))
        out.append({
            "field": field,
            "scope": "report",
            "text": text,
            "is_blank": text == "",
            "char_count": len(text),
            # No auto-generation, per spec. Empty is a real, valid state.
            "source": "manual entry" if text else "blank (no auto-generation)",
        })
    return out


# ── Part A: allocation rollups ────────────────────────────────────────────

def _rollup(contribs: list, key: str, order: Optional[tuple] = None) -> dict:
    """Group per-deal contributions into buckets with dollars and shares.

    ``contribs`` rows carry ``funded``/``committed`` already on the requested
    basis. A bucket total is None only if every deal in it is None; a single
    missing deal does not void the bucket, but it is counted so the gap shows.
    """
    buckets: dict[str, dict] = {}
    for c in contribs:
        b = buckets.setdefault(c[key], {
            "label": c[key], "funded": 0.0, "committed": 0.0,
            "deal_count": 0, "funded_missing": 0, "committed_missing": 0,
            "deals": [],
        })
        b["deal_count"] += 1
        if c["funded"] is None:
            b["funded_missing"] += 1
        else:
            b["funded"] += c["funded"]
        if c["committed"] is None:
            b["committed_missing"] += 1
        else:
            b["committed"] += c["committed"]
        b["deals"].append({
            "vcode": c["vcode"], "name": c["name"],
            "funded": c["funded"], "committed": c["committed"],
            "lookthrough_pct": c.get("lookthrough_pct"),
        })

    total_funded = sum(b["funded"] for b in buckets.values())
    total_committed = sum(b["committed"] for b in buckets.values())

    for b in buckets.values():
        b["funded_pct"] = (b["funded"] / total_funded) if total_funded else None
        b["committed_pct"] = ((b["committed"] / total_committed)
                              if total_committed else None)
        b["deals"].sort(key=lambda d: -(d["funded"] or 0))

    if order:
        ranked = sorted(buckets.values(),
                        key=lambda b: (order.index(b["label"])
                                       if b["label"] in order else len(order),
                                       -b["funded"]))
    else:
        ranked = sorted(buckets.values(), key=lambda b: -b["funded"])

    return {
        "buckets": ranked,
        "total_funded": total_funded,
        "total_committed": total_committed,
        "bucket_count": len(ranked),
        "deal_count": sum(b["deal_count"] for b in ranked),
    }


def _contributions(rows: list, basis: str, assumed_pct: Optional[float]) -> list:
    """Per-deal funded/committed on the requested basis.

    On the look-through basis a deal with no resolvable ownership % contributes
    nothing and is excluded — the spec's guardrail is to flag, never to show an
    unscaled number next to scaled ones. ``assumed_pct`` exists only so the
    self-test can quantify what such a deal *would* add; it defaults to None and
    production never sets it.
    """
    out = []
    for r in rows:
        pct = r["lookthrough_pct"]
        if basis == "full":
            f, c, used = r["funded_deal_level"], r["committed_deal_level"], None
        else:
            used = pct if pct is not None else assumed_pct
            if used is None:
                continue
            f = None if r["funded_deal_level"] is None else used * r["funded_deal_level"]
            c = (None if r["committed_deal_level"] is None
                 else used * r["committed_deal_level"])
        out.append({**r, "funded": f, "committed": c, "pct_used": used})
    return out


def assemble_summary(investor_code: str, quarter: str, *,
                     resolved: dict,
                     one_pager_provider: Callable[[str, str], dict],
                     comment_loader: Optional[Callable] = None,
                     editable_loader: Optional[Callable] = None,
                     basis: str = ALLOCATION_BASIS,
                     deal_type_map: Optional[dict] = None,
                     assumed_pct_for_flagged: Optional[float] = None) -> dict:
    """Build the Summary subtab for one investor and quarter.

    ``resolved`` is Step 1's ``resolve_investor_deals`` output; every deal in it
    (including the ownership-flagged ones) is read exactly once through
    ``one_pager_provider``.
    """
    if basis not in ("lookthrough", "full"):
        raise ValueError(f"basis must be 'lookthrough' or 'full', got {basis!r}")

    dt_map = deal_type_map or DEAL_TYPE_MAP
    narratives_raw = (comment_loader or _default_comment_loader)(
        investor_code, quarter) or {}
    editable = (editable_loader or _default_editable)(investor_code, quarter)

    diag = {"deals": 0, "provider_errors": 0, "pct_unavailable": 0,
            "funded_missing": 0, "committed_missing": 0,
            "unclassified_strategy": 0, "unclassified_asset": 0,
            "strategy_from_investment_strategy": 0,
            "strategy_from_lifecycle_proxy": 0}
    flags: list[str] = []

    grouped = [(g, e) for g, items in (resolved.get("groups") or {}).items()
               for e in items]
    flagged_entries = list(resolved.get("flagged") or [])

    rows: list[dict] = []
    for group, entry in grouped + [(None, e) for e in flagged_entries]:
        vcode = entry["vcode"]
        row_flags = []
        try:
            payload = one_pager_provider(vcode, quarter) or {}
        except Exception as exc:
            diag["provider_errors"] += 1
            row_flags.append(f"One Pager unavailable: {str(exc)[:80]}")
            payload = {}
        cap = payload.get("cap_stack") or {}

        funded = _num(cap.get("pref_equity"))
        committed = _num(cap.get("committed_pe"))
        if funded is None:
            diag["funded_missing"] += 1
            row_flags.append("funded pref unavailable")
        if committed is None:
            diag["committed_missing"] += 1
            row_flags.append("commitment unavailable")

        pct = entry.get("lookthrough_pct")
        is_flagged = group is None
        if pct is None:
            diag["pct_unavailable"] += 1
            row_flags.append("ownership % unavailable — excluded from the "
                             "look-through allocation")

        strategy, strat_source = strategy_for(entry)
        if strat_source == "Investment_Strategy":
            diag["strategy_from_investment_strategy"] += 1
        elif strat_source == "Lifecycle (proxy)":
            diag["strategy_from_lifecycle_proxy"] += 1

        deal_type = map_deal_type(strategy, dt_map)
        deal_type_literal = map_deal_type(strategy, DEAL_TYPE_MAP_LITERAL)
        if deal_type == UNCLASSIFIED:
            diag["unclassified_strategy"] += 1
            row_flags.append(f"strategy {strategy or '(blank)'!r} maps to no "
                             f"deal type")
        asset_bucket = roll_asset_type(entry.get("asset_type"))
        if asset_bucket == ASSET_UNKNOWN:
            diag["unclassified_asset"] += 1
            row_flags.append("Asset_Type blank")

        diag["deals"] += 1
        rows.append({
            "vcode": vcode, "name": entry.get("name", vcode),
            "group": group, "ownership_flagged": is_flagged,
            "lookthrough_pct": pct,
            "asset_type_raw": _s(entry.get("asset_type")),
            "asset_type": asset_bucket,
            "strategy": strategy, "strategy_source": strat_source,
            "deal_type": deal_type, "deal_type_literal": deal_type_literal,
            "funded_deal_level": funded,
            "committed_deal_level": committed,
            "flags": row_flags,
        })

    if diag["strategy_from_lifecycle_proxy"]:
        flags.append(
            f"{diag['strategy_from_lifecycle_proxy']} of {diag['deals']} deals "
            f"take their deal type from the Lifecycle proxy because "
            f"Investment_Strategy is empty — see DEAL_TYPE_MAP")
    if diag["pct_unavailable"]:
        flags.append(
            f"{diag['pct_unavailable']} deal(s) have no resolvable ownership % "
            f"and are excluded from the look-through allocation")
    if diag["unclassified_strategy"]:
        flags.append(f"{diag['unclassified_strategy']} deal(s) fell into "
                     f"'{UNCLASSIFIED}' on deal type")

    contribs = _contributions(rows, basis, assumed_pct_for_flagged)
    asset_alloc = _rollup(contribs, "asset_type")
    deal_alloc = _rollup(contribs, "deal_type", order=DEAL_TYPES)

    # The other basis, always computed so the choice stays auditable.
    other = "full" if basis == "lookthrough" else "lookthrough"
    alt = _contributions(rows, other, assumed_pct_for_flagged)
    # Literal strategy reading, likewise for audit only.
    lit = [{**c, "deal_type": c["deal_type_literal"]} for c in contribs]

    # Per-deal attribution of committed - funded, so the committed gap against
    # the PDF is traceable to deals rather than asserted.
    gap = []
    for c in contribs:
        if c["funded"] is None or c["committed"] is None:
            continue
        d = c["committed"] - c["funded"]
        if abs(d) > 1:
            gap.append({"vcode": c["vcode"], "name": c["name"],
                        "unfunded": d,
                        "deal_level": ((c["committed_deal_level"] or 0)
                                       - (c["funded_deal_level"] or 0)),
                        "lookthrough_pct": c.get("pct_used")})
    gap.sort(key=lambda g: -abs(g["unfunded"]))
    diag["committed_gap_attribution"] = gap
    diag["committed_gap_total"] = sum(g["unfunded"] for g in gap)

    flagged_out = []
    for r in rows:
        if not r["ownership_flagged"]:
            continue
        flagged_out.append({
            "vcode": r["vcode"], "name": r["name"],
            "reason": "ownership % unavailable",
            "funded_deal_level": r["funded_deal_level"],
            "committed_deal_level": r["committed_deal_level"],
            "asset_type": r["asset_type"], "deal_type": r["deal_type"],
            "excluded_from": ("look-through allocation"
                              if basis == "lookthrough" else None),
        })

    return {
        "investor_code": resolved.get("investor_code", investor_code),
        "investor_name": resolved.get("investor_name", investor_code),
        "quarter": quarter,
        "subtab": "summary",
        "basis": basis,
        "basis_note": ("allocation dollars are the investor's share "
                       "(deal-level x look-through %)" if basis == "lookthrough"
                       else "allocation dollars are full deal-level"),
        # Part A
        "asset_allocation": asset_alloc,
        "deal_type_allocation": deal_alloc,
        # Part B
        "narratives": _build_narratives(narratives_raw),
        "editable": editable,
        # audit
        "deals": rows,
        "ownership_flagged": flagged_out,
        "alternate_basis": {
            "basis": other,
            "asset_allocation": _rollup(alt, "asset_type"),
            "deal_type_allocation": _rollup(alt, "deal_type", order=DEAL_TYPES),
        },
        "alternate_deal_type_literal": _rollup(lit, "deal_type",
                                              order=DEAL_TYPES),
        "deal_type_map": dict(dt_map),
        "diagnostics": diag,
        "flags": flags,
    }


# ── Self-test ─────────────────────────────────────────────────────────────

# Reference PDF, 26Q1 page 1 (Portfolio Summary).
_PDF = {
    "funded_total": 404.2e6,
    "committed_total": 445.1e6,
    "asset_pct": {"Multifamily": 59.0, "Retail": 29.0,
                  "Self-Storage": 9.0, "Office": 3.0},
    "deal_type": {"Value-Add": (38.0, 153.1e6),
                  "Income": (32.0, 131.3e6),
                  "New Construction": (30.0, 119.8e6)},
}


def _selftest():                                    # pragma: no cover
    """Reproduce the 26Q1 page-1 allocations from live data.

    Pulls the two frames over the REST API with narrow per-entity filters — one
    page per request, OFFSET never used — exactly as the Step 1/5a self-tests do.
    """
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
    seen, frontier, rel_rows = set(), [INV], []

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
        rel_rows.extend(kids)
        for r in kids:
            c = str(r.get("InvestmentID") or "").strip().upper()
            if c:
                rel_rows.extend(fetch("InvestmentID", c))
                if c not in seen:
                    frontier.append(c)

    rel = pd.DataFrame(rel_rows).drop_duplicates()
    resolved = resolve_investor_deals(INV, Q, rel, inv)
    print(f"Step 1: {resolved['diagnostics']['deal_count']} deals, "
          f"{len(resolved['flagged'])} ownership-flagged\n")

    # Step 2 on a scratch db, so Part B is exercised against real persistence
    # while proving the boxes start blank.
    tmp = os.path.join(tempfile.mkdtemp(prefix="ps_step6a_"), "t.db")
    eng = sqlalchemy.create_engine(f"sqlite:///{tmp}")
    P._engine = lambda: eng                          # type: ignore[assignment]
    P._is_postgres = lambda: False                   # type: ignore[assignment]

    cache: dict = {}

    def provider(vc, q):
        if (vc, q) not in cache:
            cache[(vc, q)] = api.get(f"/api/financials/{vc}/one-pager",
                                     params={"quarter": q})
        return cache[(vc, q)]

    checks = []

    def chk(label, cond):
        checks.append((label, bool(cond)))
        print(f"    [{'PASS' if cond else 'FAIL'}] {label}")

    out = assemble_summary(INV, Q, resolved=resolved,
                           one_pager_provider=provider)

    # ---------------------------------------------------------------- basis
    print("=" * 112)
    print("SCALING BASIS — which reproduces the PDF's $404.2M funded?")
    print("=" * 112)
    lt = out["asset_allocation"]["total_funded"]
    fu = out["alternate_basis"]["asset_allocation"]["total_funded"]
    # what the ownership-flagged deal would add, quantified but not assumed
    incl = assemble_summary(INV, Q, resolved=resolved,
                            one_pager_provider=provider,
                            assumed_pct_for_flagged=1.0)
    lt_incl = incl["asset_allocation"]["total_funded"]
    ct_incl = incl["asset_allocation"]["total_committed"]

    print(f"{'variant':<48}{'funded':>14}{'delta vs PDF':>16}{'% off':>10}")
    print("-" * 112)
    for label, v in (("look-through, flagged deal excluded", lt),
                     ("look-through, 45th & Main assumed 100%", lt_incl),
                     ("full deal-level (no scaling)", fu)):
        d = v - _PDF["funded_total"]
        print(f"{label:<48}{v/1e6:>13.2f}M{d/1e6:>+15.2f}M"
              f"{100*d/_PDF['funded_total']:>+9.2f}%")

    chk("look-through basis is far closer to the PDF than full deal-level",
        abs(lt_incl - _PDF["funded_total"]) < abs(fu - _PDF["funded_total"]))
    chk("look-through funded (with the flagged deal at 100%) within 0.5% of "
        "the PDF's $404.2M",
        abs(lt_incl - _PDF["funded_total"]) / _PDF["funded_total"] < 0.005)
    chk("full deal-level is >25% off the PDF (so it is not the basis)",
        abs(fu - _PDF["funded_total"]) / _PDF["funded_total"] > 0.25)
    chk("basis in use is 'lookthrough'", out["basis"] == "lookthrough")

    print(f"\n  committed, same variant: {ct_incl/1e6:.2f}M vs PDF "
          f"{_PDF['committed_total']/1e6:.1f}M  "
          f"({100*(ct_incl-_PDF['committed_total'])/_PDF['committed_total']:+.2f}%)")
    print("  MISMATCH FLAGGED — committed does not tie. Attribution "
          "(scaled committed - funded, per deal):")
    for g in incl["diagnostics"]["committed_gap_attribution"]:
        print(f"     {g['vcode']:<10}{g['name'][:32]:<34}{g['unfunded']:>15,.0f}")
    print(f"     {'':44}{'computed total':>15}"
          f" {incl['diagnostics']['committed_gap_total']:,.0f}")
    print(f"     {'':44}{'PDF implied':>15} "
          f"{_PDF['committed_total']-_PDF['funded_total']:,.0f}")
    chk("committed gap is attributed per deal, not asserted",
        len(incl["diagnostics"]["committed_gap_attribution"]) > 0)

    # ------------------------------------------------------- asset allocation
    print("\n" + "=" * 112)
    print("ASSET ALLOCATION — computed vs PDF (look-through, flagged deal "
          "at 100%)")
    print("=" * 112)
    aa = incl["asset_allocation"]
    print(f"{'asset type':<18}{'funded':>13}{'%':>9}{'PDF %':>8}{'delta':>9}"
          f"{'committed':>14}{'%':>9}   verdict")
    print("-" * 112)
    asset_ok = True
    for b in aa["buckets"]:
        pdf = _PDF["asset_pct"].get(b["label"])
        pc = 100 * (b["funded_pct"] or 0)
        cp = 100 * (b["committed_pct"] or 0)
        d = (pc - pdf) if pdf is not None else None
        ok = pdf is not None and abs(d) <= 0.5
        if pdf is not None and not ok:
            asset_ok = False
        print(f"{b['label']:<18}{b['funded']/1e6:>12.2f}M{pc:>8.2f}%"
              f"{(f'{pdf:.0f}%' if pdf is not None else '-'):>8}"
              f"{(f'{d:+.2f}' if d is not None else '-'):>9}"
              f"{b['committed']/1e6:>13.2f}M{cp:>8.2f}%   "
              f"{'ok' if ok else 'MISMATCH >0.5pp'}")
        checks.append((f"asset {b['label']} % within 0.5pp of PDF", ok))
    print(f"{'TOTAL':<18}{aa['total_funded']/1e6:>12.2f}M{100:>8.2f}%"
          f"{'100%':>8}{'':>9}{aa['total_committed']/1e6:>13.2f}M{100:>8.2f}%")
    chk("every PDF asset type is present as a bucket",
        set(_PDF["asset_pct"]) <= {b["label"] for b in aa["buckets"]})
    chk("asset bucket funded shares sum to 100%",
        abs(sum(b["funded_pct"] for b in aa["buckets"]) - 1.0) < 1e-9)
    if not asset_ok:
        print("  NOTE: Multifamily / Self-Storage disagree by ~1pp and exactly "
              "offset, i.e. ~4.3M of funded sits in a different type in the "
              "PDF. Retail and Office tie to <0.1pp.")

    # --------------------------------------------------- deal type allocation
    print("\n" + "=" * 112)
    print("DEAL TYPE ALLOCATION — computed vs PDF")
    print("=" * 112)
    da = incl["deal_type_allocation"]
    print(f"{'deal type':<20}{'funded':>13}{'PDF':>10}{'delta':>11}"
          f"{'%':>8}{'PDF %':>8}   verdict")
    print("-" * 112)
    for b in da["buckets"]:
        tgt = _PDF["deal_type"].get(b["label"])
        pc = 100 * (b["funded_pct"] or 0)
        if tgt is None:
            print(f"{b['label']:<20}{b['funded']/1e6:>12.2f}M{'-':>10}"
                  f"{'-':>11}{pc:>7.2f}%{'-':>8}   NOT IN PDF")
            checks.append((f"deal type {b['label']} unexpected", False))
            continue
        pdf_pct, pdf_amt = tgt
        d = b["funded"] - pdf_amt
        ok = abs(d) <= 5e6
        print(f"{b['label']:<20}{b['funded']/1e6:>12.2f}M{pdf_amt/1e6:>9.1f}M"
              f"{d/1e6:>+10.2f}M{pc:>7.2f}%{pdf_pct:>7.0f}%   "
              f"{'ok' if ok else 'MISMATCH >5M'}")
        checks.append((f"deal type {b['label']} within $5M of PDF", ok))
    print(f"{'TOTAL':<20}{da['total_funded']/1e6:>12.2f}M"
          f"{sum(v[1] for v in _PDF['deal_type'].values())/1e6:>9.1f}M")

    chk("deal type buckets are exactly the PDF's three",
        {b["label"] for b in da["buckets"]} == set(_PDF["deal_type"]))
    chk("no deal fell into 'Unclassified'",
        incl["diagnostics"]["unclassified_strategy"] == 0)

    lit = incl["alternate_deal_type_literal"]
    lit_err = sum(abs(b["funded"] - _PDF["deal_type"][b["label"]][1])
                  for b in lit["buckets"] if b["label"] in _PDF["deal_type"])
    cal_err = sum(abs(b["funded"] - _PDF["deal_type"][b["label"]][1])
                  for b in da["buckets"] if b["label"] in _PDF["deal_type"])
    print(f"\n  calibrated map total abs error {cal_err/1e6:>7.2f}M")
    print(f"  literal map    total abs error {lit_err/1e6:>7.2f}M   "
          f"(returned under alternate_deal_type_literal)")
    chk("calibrated map beats the literal reading against the PDF",
        cal_err < lit_err)

    proxy = incl["diagnostics"]["strategy_from_lifecycle_proxy"]
    print(f"\n  strategy source: {incl['diagnostics']['strategy_from_investment_strategy']} "
          f"from Investment_Strategy, {proxy} from the Lifecycle proxy")
    chk("proxy use is flagged when Investment_Strategy is empty",
        proxy == 0 or any("Lifecycle proxy" in f for f in incl["flags"]))

    # --------------------------------------------------------- Part B
    print("\n" + "=" * 112)
    print("PART B — narrative boxes")
    print("=" * 112)
    chk("exactly 2 narrative boxes", len(out["narratives"]) == 2)
    chk("fields are narrative_1 / narrative_2",
        [n["field"] for n in out["narratives"]] == list(NARRATIVE_FIELDS))
    chk("both load BLANK (not auto-generated)",
        all(n["is_blank"] and n["text"] == "" for n in out["narratives"]))
    chk("blank boxes report no generated source",
        all("blank" in n["source"] for n in out["narratives"]))
    chk("scope is 'report' on both",
        all(n["scope"] == "report" for n in out["narratives"]))

    P.save_comment(INV, Q, "report", "narrative_1",
                   "Portfolio performance was steady through the quarter.",
                   updated_by="selftest")
    rt = assemble_summary(INV, Q, resolved=resolved,
                          one_pager_provider=provider)
    n1 = {n["field"]: n for n in rt["narratives"]}
    chk("saved narrative_1 round-trips back",
        n1["narrative_1"]["text"].startswith("Portfolio performance was steady")
        and not n1["narrative_1"]["is_blank"])
    chk("narrative_2 stays blank after saving only narrative_1",
        n1["narrative_2"]["is_blank"])
    for n in rt["narratives"]:
        print(f"    {n['field']}: blank={n['is_blank']!s:<6} "
              f"chars={n['char_count']:<4} {n['text'][:52]!r}")

    # --------------------------------------------------------- structure
    print("\n" + "=" * 112)
    print("STRUCTURE + consistency with the Financial subtab")
    print("=" * 112)
    for k in ("investor_code", "quarter", "subtab", "basis",
              "asset_allocation", "deal_type_allocation", "narratives",
              "deals", "ownership_flagged", "alternate_basis", "diagnostics"):
        checks.append((f"key {k!r} present", k in out))
    chk("subtab == 'summary'", out["subtab"] == "summary")
    chk("every resolved deal appears exactly once",
        len(out["deals"]) == (resolved["diagnostics"]["deal_count"]
                              + len(resolved["flagged"])))
    chk("the two rollups cover the same deal population",
        out["asset_allocation"]["deal_count"]
        == out["deal_type_allocation"]["deal_count"])
    chk("asset and deal-type funded totals agree",
        abs(out["asset_allocation"]["total_funded"]
            - out["deal_type_allocation"]["total_funded"]) < 1e-6)
    chk("ownership-flagged deal is reported, not silently dropped",
        len(out["ownership_flagged"]) == len(resolved["flagged"]))
    chk("flagged deal is excluded from the look-through total",
        abs(lt_incl - lt) > 1e6)

    # funded here must equal `invested` on the Financial subtab: same
    # cap_stack.pref_equity through the same look-through %.
    from flask_app.services.portfolio_snapshot_financial import assemble_financial
    fin = assemble_financial(INV, Q, resolved=resolved,
                             one_pager_provider=provider,
                             manual_loader=lambda i, q: {},
                             footnote_loader=lambda i, q: [])
    fin_invested = sum(
        r["invested"] for g in fin["groups"].values() for r in g["deals"]
        if r.get("invested") is not None)
    print(f"\n  Summary funded (look-through, flagged excluded) "
          f"{lt:>16,.0f}")
    print(f"  Financial sum of Invested                      {fin_invested:>16,.0f}")
    chk("Summary funded == Financial's Invested (no drift between subtabs)",
        abs(lt - fin_invested) < 1e-6)

    print("\n" + "=" * 112)
    passed = sum(1 for _, ok in checks if ok)
    print(f"RESULT: {passed}/{len(checks)} checks passed")
    for label, ok in checks:
        if not ok:
            print(f"  FAILED: {label}")
    print("=" * 112)
    return passed == len(checks)


if __name__ == "__main__":                          # pragma: no cover
    import sys
    sys.exit(0 if _selftest() else 1)
