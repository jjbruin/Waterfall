"""
ownership_tree.py
Process multi-tiered ownership relationships from MRI_IA_Relationship.csv

Handles:
- Multi-level ownership chains
- Pass-through entities (100% owned by single parent)
- Ultimate beneficial owners
- Waterfall requirement identification
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
import pandas as pd
from datetime import date


@dataclass
class OwnershipNode:
    """Represents an entity in the ownership tree"""
    entity_id: str
    name: str
    investors: List[Tuple[str, float]] = field(default_factory=list)  # [(InvestorID, ownership_pct)]
    investments: List[str] = field(default_factory=list)  # InvestmentIDs this entity invests in
    is_passthrough: bool = False  # True if 100% owned by single parent
    needs_waterfall: bool = False  # True if multiple investors
    level: int = 0  # Depth in tree (0 = ultimate investors)


@dataclass
class WaterfallRequirement:
    """Describes a required waterfall"""
    entity_id: str
    entity_name: str
    num_investors: int
    investor_ids: List[str]
    investor_names: List[str]
    ownership_pcts: List[float]
    deal_vcode: Optional[str] = None  # If this is a deal-level entity
    ultimate_investors: List[Tuple[str, float]] = field(default_factory=list)  # [(InvestorID, effective_pct)]


def load_relationships(df: pd.DataFrame) -> pd.DataFrame:
    """
    Load and normalize MRI_IA_Relationship.csv
    
    Expected columns: InvestmentID, InvestorID, OwnershipPct, Name, StartDate
    """
    rel = df.copy()
    rel.columns = [str(c).strip() for c in rel.columns]
    
    required = {"InvestmentID", "InvestorID", "OwnershipPct"}
    missing = [c for c in required if c not in rel.columns]
    if missing:
        raise ValueError(f"MRI_IA_Relationship.csv missing columns: {missing}")
    
    rel["InvestmentID"] = rel["InvestmentID"].astype(str).str.strip()
    rel["InvestorID"] = rel["InvestorID"].astype(str).str.strip()
    rel["OwnershipPct"] = pd.to_numeric(rel["OwnershipPct"], errors="coerce").fillna(0.0)
    
    # Normalize percentages
    rel["OwnershipPct"] = rel["OwnershipPct"].apply(
        lambda x: x / 100.0 if x > 1.0 else x
    )
    
    if "Name" in rel.columns:
        rel["Name"] = rel["Name"].fillna("").astype(str).str.strip()
    else:
        rel["Name"] = ""
    
    if "StartDate" in rel.columns:
        rel["StartDate"] = pd.to_datetime(rel["StartDate"], errors="coerce").dt.date
    
    return rel


def build_ownership_tree(relationships: pd.DataFrame) -> Dict[str, OwnershipNode]:
    """
    Build complete ownership tree from relationships
    
    Returns:
        Dict[entity_id] -> OwnershipNode
    """
    nodes = {}
    
    # First pass: create all nodes
    all_entities = set(relationships["InvestmentID"].unique()) | set(relationships["InvestorID"].unique())
    
    for entity_id in all_entities:
        # Get entity name (from InvestmentID rows where this entity appears)
        name_rows = relationships[relationships["InvestmentID"] == entity_id]
        name = name_rows["Name"].iloc[0] if not name_rows.empty and "Name" in name_rows.columns else ""
        
        nodes[entity_id] = OwnershipNode(
            entity_id=entity_id,
            name=name
        )
    
    # Second pass: build relationships
    for _, row in relationships.iterrows():
        investment_id = row["InvestmentID"]
        investor_id = row["InvestorID"]
        ownership_pct = row["OwnershipPct"]
        
        # Investment has this investor
        nodes[investment_id].investors.append((investor_id, ownership_pct))
        
        # Investor invests in this
        nodes[investor_id].investments.append(investment_id)
    
    # Third pass: identify passthroughs and waterfall needs
    for entity_id, node in nodes.items():
        if len(node.investors) == 1 and abs(node.investors[0][1] - 1.0) < 0.0001:
            node.is_passthrough = True
        elif len(node.investors) > 1:
            node.needs_waterfall = True
    
    return nodes


def get_ultimate_investors(
    entity_id: str,
    nodes: Dict[str, OwnershipNode],
    visited: Optional[Set[str]] = None,
    current_ownership: float = 1.0,
    normalize: bool = True
) -> List[Tuple[str, float]]:
    """
    Recursively find ultimate investors and their effective ownership percentages

    Args:
        entity_id: Starting entity
        nodes: Complete ownership tree
        visited: Entities already processed (prevent circular references)
        current_ownership: Cumulative ownership percentage
        normalize: If True, normalize ownership percentages at each level to sum to 100%

    Returns:
        List of (ultimate_investor_id, effective_ownership_pct)
    """
    if visited is None:
        visited = set()

    if entity_id in visited:
        return []  # Circular reference

    visited.add(entity_id)

    node = nodes.get(entity_id)
    if not node:
        return [(entity_id, current_ownership)]  # Leaf node (ultimate investor)

    if not node.investors:
        return [(entity_id, current_ownership)]  # No investors = ultimate investor

    ultimate = []

    # Calculate sum of ownership percentages at this level
    total_ownership = sum(pct for _, pct in node.investors)

    for investor_id, ownership_pct in node.investors:
        # Normalize ownership if percentages don't sum to 100%
        if normalize and total_ownership > 0 and abs(total_ownership - 1.0) > 0.0001:
            normalized_pct = ownership_pct / total_ownership
        else:
            normalized_pct = ownership_pct

        effective_pct = current_ownership * normalized_pct

        # Recurse up the chain
        upstream = get_ultimate_investors(investor_id, nodes, visited.copy(), effective_pct, normalize)
        ultimate.extend(upstream)

    return ultimate


def consolidate_ultimate_investors(ultimate_list: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """
    Consolidate duplicate ultimate investors (sum their percentages)
    
    Example: If same investor appears via multiple paths, sum the percentages
    """
    consolidated = {}
    
    for investor_id, pct in ultimate_list:
        if investor_id in consolidated:
            consolidated[investor_id] += pct
        else:
            consolidated[investor_id] = pct
    
    return [(k, v) for k, v in consolidated.items()]


def identify_waterfall_requirements(
    nodes: Dict[str, OwnershipNode],
    deal_entities: Optional[Set[str]] = None
) -> List[WaterfallRequirement]:
    """
    Identify all entities that require waterfalls
    
    Args:
        nodes: Complete ownership tree
        deal_entities: Set of entity IDs that are actual deals (have PropCodes)
    
    Returns:
        List of WaterfallRequirement objects
    """
    requirements = []
    
    if deal_entities is None:
        deal_entities = set()
    
    for entity_id, node in nodes.items():
        if not node.needs_waterfall:
            continue
        
        # Get investor details
        investor_ids = [inv_id for inv_id, _ in node.investors]
        investor_names = [nodes[inv_id].name if inv_id in nodes else "" for inv_id in investor_ids]
        ownership_pcts = [pct for _, pct in node.investors]
        
        # Get ultimate beneficial owners
        ultimate = get_ultimate_investors(entity_id, nodes)
        ultimate = consolidate_ultimate_investors(ultimate)
        
        # Sort by ownership percentage
        ultimate = sorted(ultimate, key=lambda x: x[1], reverse=True)
        
        req = WaterfallRequirement(
            entity_id=entity_id,
            entity_name=node.name,
            num_investors=len(node.investors),
            investor_ids=investor_ids,
            investor_names=investor_names,
            ownership_pcts=ownership_pcts,
            deal_vcode=entity_id if entity_id in deal_entities else None,
            ultimate_investors=ultimate
        )
        
        requirements.append(req)
    
    return requirements


def trace_ownership_path(
    from_entity: str,
    to_entity: str,
    nodes: Dict[str, OwnershipNode]
) -> List[Tuple[str, str, float]]:
    """
    Trace ownership path from one entity to another
    
    Returns:
        List of (from_id, to_id, ownership_pct) tuples representing the path
    """
    def find_path(current: str, target: str, visited: Set[str], path: List) -> Optional[List]:
        if current == target:
            return path
        
        if current in visited:
            return None
        
        visited.add(current)
        
        node = nodes.get(current)
        if not node:
            return None
        
        for investor_id, ownership_pct in node.investors:
            new_path = path + [(current, investor_id, ownership_pct)]
            result = find_path(investor_id, target, visited.copy(), new_path)
            if result is not None:
                return result
        
        return None
    
    return find_path(from_entity, to_entity, set(), []) or []


def calculate_effective_ownership(path: List[Tuple[str, str, float]]) -> float:
    """Calculate effective ownership percentage along a path"""
    if not path:
        return 0.0
    
    effective = 1.0
    for _, _, pct in path:
        effective *= pct
    
    return effective


def get_passthrough_chain(entity_id: str, nodes: Dict[str, OwnershipNode]) -> List[str]:
    """
    Get chain of passthrough entities above this entity
    
    Returns:
        List of entity IDs from current up to first non-passthrough parent
    """
    chain = [entity_id]
    current = entity_id
    
    while True:
        node = nodes.get(current)
        if not node or not node.is_passthrough or not node.investors:
            break
        
        parent_id = node.investors[0][0]  # Single 100% owner
        chain.append(parent_id)
        current = parent_id
    
    return chain


def create_waterfall_summary_df(requirements: List[WaterfallRequirement]) -> pd.DataFrame:
    """
    Create summary DataFrame of waterfall requirements
    
    Returns:
        DataFrame with columns: EntityID, EntityName, NumInvestors, IsDeal, Investors, UltimateInvestors
    """
    rows = []
    
    for req in requirements:
        # Format investor list
        investor_list = "; ".join([
            f"{inv_id} ({pct*100:.2f}%)" 
            for inv_id, pct in zip(req.investor_ids, req.ownership_pcts)
        ])
        
        # Format ultimate investors (top 5)
        ultimate_list = "; ".join([
            f"{inv_id} ({pct*100:.2f}%)" 
            for inv_id, pct in req.ultimate_investors[:5]
        ])
        if len(req.ultimate_investors) > 5:
            ultimate_list += f" ... and {len(req.ultimate_investors) - 5} more"
        
        rows.append({
            "EntityID": req.entity_id,
            "EntityName": req.entity_name,
            "NumInvestors": req.num_investors,
            "IsDeal": "Yes" if req.deal_vcode else "No",
            "DirectInvestors": investor_list,
            "UltimateInvestors": ultimate_list,
        })
    
    return pd.DataFrame(rows)


def visualize_ownership_tree(
    entity_id: str,
    nodes: Dict[str, OwnershipNode],
    max_depth: int = 20,
    current_depth: int = 0,
    visited: Optional[Set[str]] = None
) -> str:
    """
    Create text visualization of ownership tree showing all branches

    Args:
        entity_id: Root entity
        nodes: Complete ownership tree
        max_depth: Maximum depth to traverse (default 20 to show all levels)
        current_depth: Current depth in tree
        visited: Set of visited entity IDs to prevent circular references

    Returns:
        Multi-line string representation
    """
    if visited is None:
        visited = set()

    if current_depth > max_depth:
        return "  " * current_depth + "└─ ... (max depth reached)\n"

    if entity_id in visited:
        return "  " * current_depth + f"└─ {entity_id} (circular reference)\n"

    visited.add(entity_id)

    node = nodes.get(entity_id)
    if not node:
        return "  " * current_depth + f"└─ {entity_id}: [ULTIMATE OWNER]\n"

    prefix = "  " * current_depth + ("└─ " if current_depth > 0 else "")

    passthrough = " [PASSTHROUGH]" if node.is_passthrough else ""
    waterfall = " [NEEDS WATERFALL]" if node.needs_waterfall else ""
    ultimate = " [ULTIMATE OWNER]" if not node.investors else ""

    result = f"{prefix}{entity_id}: {node.name}{passthrough}{waterfall}{ultimate}\n"

    # Sort investors by ownership percentage (highest first)
    sorted_investors = sorted(node.investors, key=lambda x: x[1], reverse=True)

    for i, (investor_id, ownership_pct) in enumerate(sorted_investors):
        is_last = (i == len(sorted_investors) - 1)
        branch_char = "└─" if is_last else "├─"
        result += "  " * (current_depth + 1) + f"{branch_char} {ownership_pct*100:.2f}% owned by:\n"
        result += visualize_ownership_tree(investor_id, nodes, max_depth, current_depth + 2, visited.copy())

    return result
