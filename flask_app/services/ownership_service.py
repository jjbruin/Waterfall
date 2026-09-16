"""Ownership service — tree building, visualization, upstream analysis.

Wraps ownership_tree.py functionality (no Streamlit dependency).
"""

import pandas as pd
from typing import Optional

from ownership_tree import (
    build_ownership_tree, load_relationships, visualize_ownership_tree,
    identify_waterfall_requirements, get_ultimate_investors,
    consolidate_ultimate_investors, OwnershipNode,
)


def get_ownership_tree(relationships_raw: pd.DataFrame,
                       root_entity: Optional[str] = None) -> dict:
    """Build ownership tree and return serializable structure.

    Returns dict with nodes (list of node dicts) and summary stats.
    """
    if relationships_raw is None or relationships_raw.empty:
        return {"nodes": [], "entity_count": 0, "relationship_count": 0}

    relationships = load_relationships(relationships_raw)
    nodes = build_ownership_tree(relationships)

    node_list = []
    for eid, node in nodes.items():
        node_list.append({
            "entity_id": node.entity_id,
            "name": node.name,
            "investors": [
                {"investor_id": iid, "ownership_pct": pct}
                for iid, pct in node.investors
            ],
            "investments": node.investments,
            "is_passthrough": node.is_passthrough,
            "needs_waterfall": node.needs_waterfall,
            "level": node.level,
        })

    return {
        "nodes": node_list,
        "entity_count": len(nodes),
        "relationship_count": len(relationships),
    }


def get_entity_tree_text(relationships_raw: pd.DataFrame, entity_id: str,
                         max_depth: int = 20) -> dict:
    """Get ownership tree visualization for a specific entity.

    Returns dict with tree_text (ASCII), ultimate_investors, and entity info.
    """
    if relationships_raw is None or relationships_raw.empty:
        return {"tree_text": "", "ultimate_investors": [], "entity_info": None}

    relationships = load_relationships(relationships_raw)
    nodes = build_ownership_tree(relationships)

    result = {"tree_text": "", "ultimate_investors": [], "entity_info": None}

    if entity_id in nodes:
        node = nodes[entity_id]
        result["entity_info"] = {
            "entity_id": node.entity_id,
            "name": node.name,
            "investor_count": len(node.investors),
            "is_passthrough": node.is_passthrough,
            "needs_waterfall": node.needs_waterfall,
        }
        result["tree_text"] = visualize_ownership_tree(entity_id, nodes, max_depth=max_depth)

        # Ultimate investors
        ultimate = get_ultimate_investors(entity_id, nodes, normalize=True)
        ultimate = consolidate_ultimate_investors(ultimate)
        result["ultimate_investors"] = [
            {
                "investor_id": iid,
                "ownership_pct": pct,
                "name": nodes[iid].name if iid in nodes else "",
            }
            for iid, pct in ultimate
        ]

    return result


def get_entity_investors(relationships_raw: pd.DataFrame, entity_id: str) -> list[dict]:
    """Get direct investors for a specific entity."""
    if relationships_raw is None or relationships_raw.empty:
        return []

    relationships = load_relationships(relationships_raw)
    nodes = build_ownership_tree(relationships)

    if entity_id not in nodes:
        return []

    node = nodes[entity_id]
    return [
        {
            "investor_id": iid,
            "ownership_pct": pct,
            "name": nodes[iid].name if iid in nodes else "",
        }
        for iid, pct in node.investors
    ]


def get_waterfall_requirements(relationships_raw: pd.DataFrame,
                               inv: pd.DataFrame = None) -> list[dict]:
    """Identify entities that need waterfall definitions.

    Returns list of requirement dicts.
    """
    if relationships_raw is None or relationships_raw.empty:
        return []

    relationships = load_relationships(relationships_raw)
    nodes = build_ownership_tree(relationships)

    deal_entities = None
    if inv is not None and not inv.empty:
        deal_entities = set(inv["vcode"].astype(str).str.strip())

    requirements = identify_waterfall_requirements(nodes, deal_entities)

    return [
        {
            "entity_id": req.entity_id,
            "entity_name": req.entity_name,
            "num_investors": req.num_investors,
            "investor_ids": req.investor_ids,
            "deal_vcode": req.deal_vcode,
        }
        for req in requirements
    ]


def _wf_label(wf_type: str) -> str:
    """The waterfall's name as a person says it."""
    return {"CF_WF": "Cash Flow", "Cap_WF": "Capital",
            "Promote_WF": "Promote"}.get(wf_type, wf_type)


def crossed_fund_investors(min_assets: int = 2) -> dict:
    """Entities that invest into more than one thing, and what.

    Used to qualify an upstream result. A distribution traced to a beneficiary
    that sits above a MULTI-ASSET fund is an estimate of this property's
    contribution, not a payable amount: the fund's actual distribution depends
    on every investment crossed inside it, and a loss elsewhere can consume a
    gain here before any of it reaches the owner.

    Read from `commitments` rather than `relationships`, because a commitment is
    the record of money actually promised into a vehicle -- see
    ownership_chain_service for why that source is preferred where the two
    disagree.
    """
    out = {}
    try:
        from flask_app.db import get_engine
        from sqlalchemy import text as _text
        with get_engine().connect() as conn:
            com = pd.read_sql(_text("SELECT * FROM commitments"), conn)
    except Exception:
        logger.warning("crossed_fund_investors: commitments not loaded", exc_info=True)
        return out
    if com.empty:
        return out

    lower = {str(c).lower(): c for c in com.columns}
    ent, inv_c = lower.get("entityid"), lower.get("investorid")
    end = lower.get("enddate")
    if not ent or not inv_c:
        return out
    if end:
        com = com[com[end].isna()]
    com = com.assign(
        _e=com[ent].astype(str).str.strip().str.upper(),
        _i=com[inv_c].astype(str).str.strip().str.upper())
    for investor, grp in com.groupby("_i"):
        held = sorted({e for e in grp["_e"] if e})
        if len(held) >= min_assets:
            out[investor] = held
    return out


def qualify_beneficiaries(beneficiary_totals: dict, upstream_rows: list,
                          crossed: dict, deal_entity: str,
                          distribution_amount: float):
    """Mark the beneficiaries whose figure is an estimate, and word the note.

    A beneficiary reached THROUGH a multi-asset fund has not been told what it
    will receive. The fund distributes on its own performance across every
    investment crossed inside it, so this property's contribution can be offset
    by another's before any cash arrives. That belongs on the row, not in a
    caption somebody scrolls past.

    Extracted from ``run_upstream_analysis`` so it can be tested without the
    waterfall machinery: the local database has three commitment rows and no
    fund holding two assets, so this branch could not otherwise be exercised at
    all before it shipped.
    """
    deal_up = str(deal_entity).strip().upper()

    # Every entity seen on any path that ENDS at a given beneficiary.
    paths_by_beneficiary = {}
    for r in upstream_rows:
        chain = [x.strip().upper()
                 for x in str(r.get("Path", "")).split("->") if x.strip()]
        if chain:
            paths_by_beneficiary.setdefault(chain[-1], set()).update(chain)

    out = []
    for bid, amt in sorted(beneficiary_totals.items(), key=lambda x: -x[1]):
        seen = paths_by_beneficiary.get(str(bid).strip().upper(), set())
        # The deal itself is not a fund holding several assets, whatever else
        # it happens to invest in; excluding it stops every row being flagged.
        funds = sorted({e for e in seen if e in crossed and e != deal_up})
        out.append({
            "entity_id": bid,
            "amount": amt,
            "pct_of_total": amt / distribution_amount if distribution_amount > 0 else 0,
            "is_estimate": bool(funds),
            "crossed_funds": [{"entity_id": f, "asset_count": len(crossed[f])}
                              for f in funds],
        })

    note = None
    if any(b["is_estimate"] for b in out):
        named = sorted({f["entity_id"] for b in out for f in b["crossed_funds"]})
        note = (
            "Marked rows are reached through "
            + ", ".join(named)
            + (", which invest" if len(named) > 1 else ", which invests")
            + " in more than one asset. Those amounts are an ESTIMATE of this "
              "property's contribution to the fund. The actual distribution "
              "depends on the performance of all crossed investments in the "
              "fund, and a result elsewhere can offset this one before any cash "
              "reaches the owner.")
    return out, note


def run_upstream_analysis(entity_id: str, distribution_amount: float,
                          relationships_raw: pd.DataFrame, wf: pd.DataFrame,
                          inv: pd.DataFrame,
                          wf_type: str = "CF_WF") -> dict:
    """Run upstream waterfall analysis for an entity.

    Runs the entity's waterfall with the given distribution amount, then traces
    cash flows upstream through the ownership chain to the beneficial owners.

    ``wf_type`` MUST BE THE CALLER'S CHOICE, NOT A DEFAULT NOBODY SEES. It was
    hardcoded to "CF_WF" at both levels, so a sale or refinancing -- which runs
    the Capital waterfall and REDUCES capital outstanding -- was silently
    modelled as an operating distribution, which does not. The two produce
    different splits from the same dollar, and nothing on screen said which had
    been used. The literal is `Cap_WF`, matching the `vmisc` values in the
    table; `run_waterfall` compares it exactly and keys `is_cap_wf` off it.
    """
    from waterfall import run_waterfall, run_recursive_upstream_waterfalls
    from loaders import load_waterfalls
    from datetime import date
    import numpy as np

    if relationships_raw is None or relationships_raw.empty:
        return {"error": "No relationship data available"}

    relationships = load_relationships(relationships_raw)
    wf_steps = load_waterfalls(wf)

    # Run deal-level CF waterfall
    test_wf = wf.copy()
    test_wf["PropCode"] = test_wf["PropCode"].fillna("").astype(str).str.strip()
    test_wf["vcode"] = test_wf["vcode"].fillna("").astype(str).str.strip()

    # Ensure nPercent_dec
    if "nPercent_dec" not in test_wf.columns:
        p = pd.to_numeric(test_wf["nPercent"], errors="coerce").fillna(0.0)
        test_wf["nPercent_dec"] = np.where(p > 1.0, p / 100.0, p)

    test_cash = pd.DataFrame([{
        "event_date": date(2025, 12, 31),
        "cash_available": distribution_amount,
    }])

    try:
        alloc, states = run_waterfall(
            wf_steps=test_wf,
            vcode=entity_id,
            wf_name=wf_type,
            period_cash=test_cash,
            initial_states={},
        )
    except Exception as e:
        return {"error": f"{_wf_label(wf_type)} waterfall failed: {e}"}

    if alloc.empty:
        return {"error": f"The {_wf_label(wf_type)} waterfall for {entity_id} "
                          f"produced no allocations. Is one set up for it?"}

    # Run upstream waterfalls
    try:
        from waterfall import build_amfee_exclusions
        _excl = {}  # No accounting data available in test analysis
        upstream_alloc, entity_states, beneficiary_totals = \
            run_recursive_upstream_waterfalls(
                deal_allocations=alloc,
                wf_steps=wf_steps,
                relationships=relationships,
                wf_type=wf_type,
                amfee_exclusions=_excl,
            )
    except Exception as e:
        return {"error": f"Upstream waterfall failed: {e}"}

    # Build serializable results
    deal_alloc_rows = []
    for _, row in alloc.iterrows():
        deal_alloc_rows.append({
            "PropCode": str(row.get("PropCode", "")),
            "vState": str(row.get("vState", "")),
            "Allocated": float(row.get("Allocated", 0)),
        })

    upstream_rows = []
    if not upstream_alloc.empty:
        for _, row in upstream_alloc.iterrows():
            upstream_rows.append({
                "Entity": str(row.get("Entity", "")),
                "PropCode": str(row.get("PropCode", "")),
                "vState": str(row.get("vState", "")),
                "Allocated": float(row.get("Allocated", 0)),
                "Level": int(row.get("Level", 0)) if "Level" in row.index else 0,
                "Path": str(row.get("Path", "")),
            })

    # ── Terminal beneficiaries, and whether their figure is an estimate ──
    #
    # A beneficiary reached THROUGH a multi-asset fund has not been told what it
    # will receive. The fund distributes on its own performance across every
    # investment crossed inside it, so this property's contribution can be
    # offset by another's before anything reaches the owner. The number is a
    # contribution estimate and must say so on the row, not in a caption
    # somebody scrolls past.
    beneficiaries, footnote = qualify_beneficiaries(
        beneficiary_totals, upstream_rows, crossed_fund_investors(),
        entity_id, distribution_amount)

    return {
        "success": True,
        "wf_type": wf_type,
        "wf_label": _wf_label(wf_type),
        "estimate_footnote": footnote,
        "distribution_amount": distribution_amount,
        "deal_allocations": deal_alloc_rows,
        "upstream_allocations": upstream_rows,
        "beneficiaries": beneficiaries,
        "total_allocated": sum(b["amount"] for b in beneficiaries),
    }
