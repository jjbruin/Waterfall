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


def run_upstream_analysis(entity_id: str, distribution_amount: float,
                          relationships_raw: pd.DataFrame, wf: pd.DataFrame,
                          inv: pd.DataFrame) -> dict:
    """Run upstream waterfall analysis for an entity.

    Runs the entity's CF waterfall with the given distribution amount,
    then traces cash flows upstream through the ownership chain.

    Returns dict with allocation results, terminal beneficiaries, etc.
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
            wf_name="CF_WF",
            period_cash=test_cash,
            initial_states={},
        )
    except Exception as e:
        return {"error": f"Deal waterfall failed: {e}"}

    if alloc.empty:
        return {"error": "Deal waterfall produced no allocations"}

    # Run upstream waterfalls
    try:
        from waterfall import build_amfee_exclusions
        _excl = {}  # No accounting data available in test analysis
        upstream_alloc, entity_states, beneficiary_totals = \
            run_recursive_upstream_waterfalls(
                deal_allocations=alloc,
                wf_steps=wf_steps,
                relationships=relationships,
                wf_type="CF_WF",
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

    # Terminal beneficiaries
    beneficiaries = []
    if beneficiary_totals:
        for bid, amt in sorted(beneficiary_totals.items(), key=lambda x: -x[1]):
            beneficiaries.append({
                "entity_id": bid,
                "amount": amt,
                "pct_of_total": amt / distribution_amount if distribution_amount > 0 else 0,
            })

    return {
        "success": True,
        "distribution_amount": distribution_amount,
        "deal_allocations": deal_alloc_rows,
        "upstream_allocations": upstream_rows,
        "beneficiaries": beneficiaries,
        "total_allocated": sum(b["amount"] for b in beneficiaries),
    }
