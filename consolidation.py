"""
consolidation.py
Property to Deal consolidation logic for sub-portfolio deals

Handles cases where:
- Multiple properties roll up to a single deal
- Loans aggregate from both deal and property levels
- Forecasts: For sub-portfolios, ONLY property-level forecasts are used
  (summed across properties). Parent-level forecasts are ignored.
"""

import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import date
from utils import normalize_columns


def identify_sub_portfolio_deals(
    relationships: pd.DataFrame,
    deals: pd.DataFrame
) -> Dict[str, List[str]]:
    """
    Identify deals that have sub-properties using Portfolio_Name in deals table.

    A deal is a parent if its Investment_Name matches the Portfolio_Name of
    other deal records (properties).

    Handles edge cases:
    - Self-referencing: Some portfolios have the parent deal's own Portfolio_Name
      equal to its Investment_Name (parent excluded from property list).
    - All matching is exact (user to ensure data consistency).

    Args:
        relationships: MRI_IA_Relationship data (kept for backward compat, ignored)
        deals: investment_map data with Portfolio_Name column

    Returns:
        Dict mapping deal InvestmentID -> list of property InvestmentIDs
    """
    deals_df = deals.copy()
    normalize_columns(deals_df)

    # Ensure required columns exist
    for col in ['InvestmentID', 'Investment_Name', 'Portfolio_Name', 'vcode']:
        if col not in deals_df.columns:
            return {}

    # NOT THE APP-WIDE CHILD LOOKUP, deliberately (Oct 6 2026). `build_property_map` is
    # that, and it now finds Burton's three buildings. This answers a different question
    # -- whose FORECASTS Deal Analysis sums -- and the sub-portfolio branch below sums
    # the properties' forecasts and ignores the parent's. Burton's forecast lives on the
    # PARENT (60 rows, none on the buildings), so treating it as a sub-portfolio here
    # blanked its projection: measured, 60 rows -> 0. Left on the old name-only rule
    # until the forecast source is decided per deal (open_items 8.5).
    deals_df['InvestmentID'] = deals_df['InvestmentID'].astype(str).str.strip()
    deals_df['Investment_Name'] = deals_df['Investment_Name'].fillna('').astype(str).str.strip()
    deals_df['Portfolio_Name'] = deals_df['Portfolio_Name'].fillna('').astype(str).str.strip()

    # Find all unique non-empty Portfolio_Names
    portfolio_names = deals_df[deals_df['Portfolio_Name'] != '']['Portfolio_Name'].unique()

    sub_portfolios = {}

    for pf_name in portfolio_names:
        # Find parent deal: Investment_Name matches this Portfolio_Name
        parent = deals_df[deals_df['Investment_Name'] == pf_name]

        if parent.empty:
            continue  # No parent found for this portfolio group

        parent_inv_id = str(parent.iloc[0]['InvestmentID'])

        # Get all properties in this portfolio, excluding the parent itself
        properties = deals_df[
            (deals_df['Portfolio_Name'] == pf_name) &
            (deals_df['InvestmentID'] != parent_inv_id)
        ]

        if not properties.empty:
            sub_portfolios[parent_inv_id] = properties['InvestmentID'].tolist()

    return sub_portfolios


def get_deal_vcode(investment_id: str, deals: pd.DataFrame) -> Optional[str]:
    """Get vcode for an InvestmentID"""
    deals_df = deals.copy()
    normalize_columns(deals_df)

    match = deals_df[deals_df['InvestmentID'].astype(str) == str(investment_id)]
    if not match.empty:
        return str(match.iloc[0]['vcode'])
    return None


def build_property_map(deals: pd.DataFrame) -> Dict[str, List[str]]:
    """THE child-property lookup: parent deal vcode -> its child property vcodes.

    ONE RULE, used app-wide (Oct 6 2026). There were three copies and they disagreed:
    this function matched children on the parent's Investment_Name only, while
    `one_pager._child_vcodes_for_parent` and `valuation_service._child_parent_map` also
    matched the parent's own Portfolio_Name and required a parent to be one. Measured on
    all 134 deals they disagreed on two:

      Burton (P0000109)   named "Burton Retail Portfolio", its three buildings labelled
                          "Burton Portfolio" -- missed here, so the Dashboard listed
                          them as deals of their own and the capitalization engine put
                          Burton's debt at 0
      P0000073            "Donald Lynch" -- here it took the OTHER "Donald Lynch"
                          (P0000049, Portfolio_Name "Donald Lynch") as its child,
                          though it is not a portfolio (Property_Count 0)

    The rule:
      * a PARENT has Property_Count >= 1 (where the column exists; without it, any deal
        whose name other deals carry as their Portfolio_Name, as before);
      * a CHILD carries Property_Count 0 and a Portfolio_Name equal to the parent's
        Investment_Name or the parent's own Portfolio_Name;
      * a deal is never its own child.

    Returns {parent: [children]}; standalone deals are absent (not empty-list).
    """
    if deals is None or deals.empty:
        return {}

    df = deals.copy()
    normalize_columns(df)
    if 'vcode' not in df.columns and 'vCode' in df.columns:
        df = df.rename(columns={'vCode': 'vcode'})
    for col in ['vcode', 'Investment_Name', 'Portfolio_Name']:
        if col not in df.columns:
            return {}

    df['vcode'] = df['vcode'].astype(str).str.strip()
    df['Investment_Name'] = df['Investment_Name'].fillna('').astype(str).str.strip()
    df['Portfolio_Name'] = df['Portfolio_Name'].fillna('').astype(str).str.strip()
    has_count = 'Property_Count' in df.columns
    pc = (pd.to_numeric(df['Property_Count'], errors='coerce').fillna(0)
          if has_count else pd.Series(0, index=df.index))

    by_label: Dict[str, List[str]] = {}
    for vc, pn, n in zip(df['vcode'], df['Portfolio_Name'], pc):
        if pn and (not has_count or n < 1):
            by_label.setdefault(pn, []).append(vc)

    prop_map: Dict[str, List[str]] = {}
    for vc, nm, pn, n in zip(df['vcode'], df['Investment_Name'], df['Portfolio_Name'], pc):
        if has_count and n < 1:
            continue
        labels = [nm] if not has_count else [nm, pn]
        children: List[str] = []
        for label in labels:
            for c in by_label.get(label, []) if label else []:
                if c != vc and c not in children:
                    children.append(c)
        if children:
            prop_map[vc] = children

    return prop_map


def get_property_vcodes_for_deal(deal_vcode: str, deals: pd.DataFrame) -> List[str]:
    """A deal's child property vcodes (not the deal itself); [] for a standalone deal.

    The single rule is `build_property_map`'s; prefer that in loops.
    """
    if not isinstance(deals, pd.DataFrame):
        # The arguments have been passed reversed before (ownership_service), and the
        # error that raised was swallowed by the caller. Fail loudly instead.
        raise TypeError("get_property_vcodes_for_deal(deal_vcode, deals): "
                        f"deals must be a DataFrame, got {type(deals).__name__}")
    return list(build_property_map(deals).get(str(deal_vcode).strip(), []))


def get_parent_deal_for_property(property_vcode: str, deals: pd.DataFrame) -> Optional[dict]:
    """
    Check if a deal is a child property and return parent deal info.

    Args:
        property_vcode: The vcode to check (e.g., 'P0000061')
        deals: investment_map DataFrame

    Returns:
        Dict with parent info if this is a child property:
        {'vcode': 'P0000033', 'Investment_Name': 'OREI Portfolio', 'InvestmentID': 'OREIMF'}
        Returns None if this is not a child property (standalone or parent deal)
    """
    deals_df = deals.copy()
    normalize_columns(deals_df)
    deals_df['vcode'] = deals_df['vcode'].astype(str).str.strip()
    deals_df['Portfolio_Name'] = deals_df['Portfolio_Name'].fillna('').astype(str).str.strip()
    deals_df['Investment_Name'] = deals_df['Investment_Name'].fillna('').astype(str).str.strip()

    # Find the property's row
    prop_row = deals_df[deals_df['vcode'] == str(property_vcode)]
    if prop_row.empty:
        return None

    portfolio_name = prop_row.iloc[0]['Portfolio_Name']
    if not portfolio_name:
        return None  # No Portfolio_Name, so not a child property

    # Find parent deal where Investment_Name = this property's Portfolio_Name
    parent = deals_df[
        (deals_df['Investment_Name'] == portfolio_name) &
        (deals_df['vcode'] != str(property_vcode))  # exclude self
    ]

    if parent.empty:
        return None  # No parent found

    parent_row = parent.iloc[0]
    return {
        'vcode': str(parent_row['vcode']),
        'Investment_Name': str(parent_row['Investment_Name']),
        'InvestmentID': str(parent_row.get('InvestmentID', ''))
    }


def consolidate_property_forecasts(
    deal_vcode: str,
    property_vcodes: List[str],
    forecasts: pd.DataFrame
) -> pd.DataFrame:
    """
    Consolidate property-level forecasts to deal level.

    For sub-portfolio deals (property_vcodes not empty):
    - ONLY use property-level forecasts (summed across properties)
    - Parent-level forecasts are IGNORED even if they exist
    - Returns empty if no property forecasts exist

    For standalone deals (property_vcodes empty):
    - Use deal-level forecasts

    Args:
        deal_vcode: The deal's vcode
        property_vcodes: List of property vcodes that belong to this deal
        forecasts: Full forecasts DataFrame

    Returns:
        Consolidated forecast DataFrame with deal_vcode as the vcode
    """
    fc = forecasts.copy()

    # Normalize vcode column
    if 'Vcode' in fc.columns and 'vcode' not in fc.columns:
        fc = fc.rename(columns={'Vcode': 'vcode'})
    fc['vcode'] = fc['vcode'].astype(str)

    # Sub-portfolio: ONLY use property forecasts, never parent level
    if property_vcodes:
        property_fc = fc[fc['vcode'].isin(property_vcodes)]

        if property_fc.empty:
            # No property forecasts - return empty (don't fall back to parent)
            return pd.DataFrame()

        # Aggregate property forecasts by date and account
        date_col = 'Date' if 'Date' in fc.columns else 'event_date'

        agg_fc = property_fc.groupby([date_col, 'vAccount'], as_index=False).agg({
            'mAmount': 'sum',
            'vSource': 'first',  # Keep first source
            'Pro_Yr': 'first'    # Keep first Pro_Yr
        })

        # Set the vcode to the deal
        agg_fc['vcode'] = deal_vcode

        return agg_fc

    # Standalone deal: use deal-level forecasts
    deal_fc = fc[fc['vcode'] == deal_vcode]
    if not deal_fc.empty:
        return deal_fc.copy()

    # No forecasts available
    return pd.DataFrame()


def get_deal_loans(
    deal_vcode: str,
    property_vcodes: List[str],
    loans: pd.DataFrame
) -> pd.DataFrame:
    """
    Get all loans for a deal (including any at property level).

    Per user: "All loans will be included in our MRI_Loans table and should
    roll to the deal level."

    Args:
        deal_vcode: The deal's vcode
        property_vcodes: List of property vcodes
        loans: MRI_Loans DataFrame

    Returns:
        All loans for this deal and its properties
    """
    if loans.empty:
        return pd.DataFrame()

    ln = loans.copy()
    normalize_columns(ln)

    # Normalize vCode column
    if 'vcode' in ln.columns and 'vCode' not in ln.columns:
        ln = ln.rename(columns={'vcode': 'vCode'})
    ln['vCode'] = ln['vCode'].astype(str)

    # Get loans at deal level and property level
    all_vcodes = [deal_vcode] + property_vcodes
    deal_loans = ln[ln['vCode'].isin(all_vcodes)]

    return deal_loans


def build_consolidated_forecast(
    deal_investment_id: str,
    deals: pd.DataFrame,
    relationships: pd.DataFrame,
    forecasts: pd.DataFrame,
    loans: pd.DataFrame,
    debug: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Build consolidated forecast for a deal, handling sub-portfolios.

    Args:
        deal_investment_id: The deal's InvestmentID (e.g., 'BRNERD', 'TFTP')
        deals: investment_map data
        relationships: MRI_IA_Relationship data
        forecasts: forecast_feed data
        loans: MRI_Loans data
        debug: If True, return debug info

    Returns:
        Tuple of:
        - consolidated_forecast: DataFrame ready for waterfall
        - deal_loans: All loans for this deal
        - debug_info: Dict with consolidation details
    """
    debug_info = {
        'deal_investment_id': deal_investment_id,
        'is_sub_portfolio': False,
        'property_count': 0,
        'forecast_source': None,
        'loan_count': 0
    }

    # Get deal vcode
    deal_vcode = get_deal_vcode(deal_investment_id, deals)
    if not deal_vcode:
        debug_info['error'] = f'No vcode found for {deal_investment_id}'
        return pd.DataFrame(), pd.DataFrame(), debug_info

    debug_info['deal_vcode'] = deal_vcode

    # Check for sub-properties
    sub_portfolios = identify_sub_portfolio_deals(relationships, deals)

    property_vcodes = []
    if deal_investment_id in sub_portfolios:
        debug_info['is_sub_portfolio'] = True
        property_inv_ids = sub_portfolios[deal_investment_id]
        debug_info['property_investment_ids'] = property_inv_ids

        # Get vcodes for properties
        for prop_inv_id in property_inv_ids:
            prop_vcode = get_deal_vcode(prop_inv_id, deals)
            if prop_vcode:
                property_vcodes.append(prop_vcode)

        debug_info['property_vcodes'] = property_vcodes
        debug_info['property_count'] = len(property_vcodes)

    # Consolidate forecasts
    # For sub-portfolios: ONLY use property forecasts (summed), ignore parent level
    # For standalone deals: use deal-level forecasts
    consolidated_fc = consolidate_property_forecasts(
        deal_vcode=deal_vcode,
        property_vcodes=property_vcodes,
        forecasts=forecasts
    )

    # Set forecast source for debug info
    if property_vcodes:
        # Sub-portfolio: always properties_aggregated (or none if empty)
        debug_info['forecast_source'] = 'properties_aggregated' if len(consolidated_fc) > 0 else 'none'
    else:
        # Standalone deal
        debug_info['forecast_source'] = 'deal_level' if len(consolidated_fc) > 0 else 'none'

    # Get loans
    deal_loans = get_deal_loans(deal_vcode, property_vcodes, loans)
    debug_info['loan_count'] = len(deal_loans)

    return consolidated_fc, deal_loans, debug_info


def get_sub_portfolio_summary(
    deals: pd.DataFrame,
    relationships: pd.DataFrame
) -> pd.DataFrame:
    """
    Get a summary of all sub-portfolio deals for display.

    Returns:
        DataFrame with deal info and property counts
    """
    sub_portfolios = identify_sub_portfolio_deals(relationships, deals)

    if not sub_portfolios:
        return pd.DataFrame()

    deals_df = deals.copy()
    normalize_columns(deals_df)

    rows = []
    for deal_inv_id, property_inv_ids in sub_portfolios.items():
        deal_match = deals_df[deals_df['InvestmentID'].astype(str) == deal_inv_id]
        if not deal_match.empty:
            deal_row = deal_match.iloc[0]
            rows.append({
                'vcode': deal_row.get('vcode', ''),
                'InvestmentID': deal_inv_id,
                'Investment_Name': deal_row.get('Investment_Name', ''),
                'Property_Count': len(property_inv_ids),
                'Properties': ', '.join(property_inv_ids)
            })

    return pd.DataFrame(rows)
