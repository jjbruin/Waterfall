"""
portfolio.py
Fund and investor-level aggregation

Layer 3: Fund aggregation (PPI proceeds across deals)
Layer 4: Investor waterfalls (LP/GP splits within funds)
"""

from datetime import date
from typing import Dict, Tuple, Optional
import pandas as pd

from models import InvestorState
from waterfall import run_waterfall_period, compound_if_year_end
from metrics import investor_metrics


def aggregate_ppi_proceeds_to_funds(
    deal_waterfalls: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    fund_deals: pd.DataFrame,
    wf_type: str = "Cap_WF"
) -> Dict[str, pd.DataFrame]:
    """
    Aggregate PPI proceeds from deal-level waterfalls up to fund level
    
    Args:
        deal_waterfalls: Dict[vcode] -> (allocations_df, investor_summary_df)
        fund_deals: Mapping of deals to funds
        wf_type: "Cap_WF" or "CF_WF"
    
    Returns:
        Dict[FundID] -> monthly_proceeds_df with columns [event_date, cash_available]
    """
    fund_proceeds = {}
    
    for _, row in fund_deals.iterrows():
        fund_id = row["FundID"]
        vcode = row["vcode"]
        ppi_code = row["PPI_PropCode"]
        ownership = row["Ownership_Pct"]
        
        if vcode not in deal_waterfalls:
            continue
        
        alloc_df, _ = deal_waterfalls[vcode]
        
        if alloc_df.empty:
            continue
        
        # Filter to PPI entity proceeds
        ppi_allocs = alloc_df[alloc_df["PropCode"] == ppi_code].copy()
        
        if ppi_allocs.empty:
            continue
        
        # Apply ownership percentage
        ppi_allocs["fund_share"] = ppi_allocs["Allocated"] * ownership
        
        # Aggregate by month
        monthly = ppi_allocs.groupby("event_date", as_index=False)["fund_share"].sum()
        monthly = monthly.rename(columns={"fund_share": "cash_available"})
        
        # Accumulate to fund
        if fund_id not in fund_proceeds:
            fund_proceeds[fund_id] = monthly
        else:
            # Merge with existing
            existing = fund_proceeds[fund_id]
            combined = pd.merge(
                existing, monthly, 
                on="event_date", 
                how="outer",
                suffixes=("", "_new")
            )
            combined["cash_available"] = (
                combined["cash_available"].fillna(0) + 
                combined.get("cash_available_new", pd.Series(dtype=float)).fillna(0)
            )
            fund_proceeds[fund_id] = combined[["event_date", "cash_available"]]
    
    return fund_proceeds


def seed_investor_states_from_accounting(
    investor_acct: pd.DataFrame,
    fund_id: str
) -> Dict[str, InvestorState]:
    """
    Seed LP/GP states from historical accounting
    
    Similar to deal-level seeding but for fund investors
    """
    acct = investor_acct[investor_acct["FundID"] == fund_id].copy()
    acct = acct.sort_values(["EffectiveDate", "InvestorID"])
    
    states = {}
    
    for _, r in acct.iterrows():
        inv_id = str(r["InvestorID"])
        d = r["EffectiveDate"]
        amt = float(r["Amount"])
        
        if pd.isna(d):
            continue
        
        if inv_id not in states:
            states[inv_id] = InvestorState(propcode=inv_id)
            if r["is_contribution"]:
                states[inv_id].last_accrual_date = d
        
        stt = states[inv_id]
        
        if r["is_contribution"]:
            cf = amt if amt < 0 else -abs(amt)
            stt.cashflows.append((d, cf))
            stt.capital_outstanding += abs(cf)
            
            if stt.last_accrual_date is None:
                stt.last_accrual_date = d
        
        elif r["is_distribution"]:
            cf = amt if amt > 0 else abs(amt)
            stt.cashflows.append((d, cf))
            
            # Fund distributions typically return capital
            capital_return = min(cf, max(0.0, stt.capital_outstanding))
            stt.capital_outstanding -= capital_return
    
    return states


def run_investor_waterfall(
    investor_wf_steps: pd.DataFrame,
    fund_id: str,
    period_cash: pd.DataFrame,
    initial_states: Optional[Dict[str, InvestorState]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run investor-level waterfall (LP/GP splits)
    
    Args:
        investor_wf_steps: Investor waterfall definitions
        fund_id: Fund identifier
        period_cash: Monthly cash available
        initial_states: Pre-seeded investor states
    
    Returns:
        (allocations_df, investor_summary_df)
    """
    steps = investor_wf_steps[investor_wf_steps["FundID"] == fund_id].copy()
    steps = steps.sort_values("iOrder")
    
    if steps.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    istates = initial_states if initial_states is not None else {}
    pref_rates = {}
    
    all_rows = []
    for _, r in period_cash.sort_values("event_date").iterrows():
        d = r["event_date"]
        cash = float(r["cash_available"])
        
        # Rename PropCode to InvestorID for investor waterfalls
        steps_renamed = steps.rename(columns={"InvestorID": "PropCode"})
        
        _rem, rows = run_waterfall_period(
            steps_renamed, istates, d, cash, pref_rates,
            is_capital_waterfall=True  # Investor waterfalls are typically capital
        )
        
        # Rename back
        for row in rows:
            row["InvestorID"] = row["PropCode"]
        
        all_rows.extend(rows)
    
    alloc_df = pd.DataFrame(all_rows)
    
    # Investor summary
    inv_rows = []
    for inv_id, stt in istates.items():
        unrealized = stt.capital_outstanding + stt.pref_unpaid_compounded + stt.pref_accrued_current_year
        
        as_of = period_cash['event_date'].max()
        metrics = investor_metrics(stt, as_of_date=as_of, unrealized_nav=unrealized)
        
        inv_rows.append({
            "InvestorID": inv_id,
            "FundID": fund_id,
            "CapitalOutstanding": stt.capital_outstanding,
            "UnpaidPrefCompounded": stt.pref_unpaid_compounded,
            "AccruedPrefCurrentYear": stt.pref_accrued_current_year,
            "UnrealizedNAV": unrealized,
            "IRR": metrics['IRR'],
            "ROE": metrics['ROE'],
            "MOIC": metrics['MOIC'],
            "TotalContributions": metrics['TotalContributions'],
            "TotalDistributions": metrics['TotalDistributions'],
        })
    
    inv_sum = pd.DataFrame(inv_rows)
    if not inv_sum.empty:
        inv_sum = inv_sum.sort_values("InvestorID")
    
    return alloc_df, inv_sum


def aggregate_investor_across_funds(
    investor_summaries: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Roll up a single investor's performance across all funds
    
    Args:
        investor_summaries: Dict[FundID] -> investor_summary_df
    
    Returns:
        DataFrame with one row per investor showing total metrics
    """
    all_summaries = []
    for fund_id, summary in investor_summaries.items():
        summary = summary.copy()
        summary["FundID"] = fund_id
        all_summaries.append(summary)
    
    if not all_summaries:
        return pd.DataFrame()
    
    combined = pd.concat(all_summaries, ignore_index=True)
    
    # Group by investor
    agg = combined.groupby("InvestorID").agg({
        "CapitalOutstanding": "sum",
        "UnpaidPrefCompounded": "sum",
        "AccruedPrefCurrentYear": "sum",
        "UnrealizedNAV": "sum",
        "TotalContributions": "sum",
        "TotalDistributions": "sum",
    }).reset_index()
    
    # Recalculate MOIC at portfolio level
    agg["MOIC"] = (agg["TotalDistributions"] + agg["UnrealizedNAV"]) / agg["TotalContributions"]
    
    # Note: Portfolio IRR requires combining all cashflows across funds
    # This requires access to InvestorState objects, not just summaries
    agg["IRR"] = None
    agg["ROE"] = None
    
    return agg
