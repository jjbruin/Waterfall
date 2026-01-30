"""
cash_management.py
Manages cash reserves, capital expenditures, capital calls, and distribution logic
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime


def load_beginning_cash_balance(isbs_df: pd.DataFrame, deal_vcode: str, forecast_start_date) -> float:
    """
    Extract beginning cash balance from ISBS_Download.csv
    
    Args:
        isbs_df: ISBS_Download dataframe
        deal_vcode: Deal identifier
        forecast_start_date: Start date of forecast (to find most recent prior balance)
    
    Returns:
        Beginning cash balance (sum of all cash accounts)
    """
    if isbs_df is None or isbs_df.empty:
        return 0.0
    
    # Normalize column names
    isbs = isbs_df.copy()
    isbs.columns = [str(c).strip() for c in isbs.columns]
    
    # Check if required columns exist
    required_cols = ['vcode', 'dtEntry', 'vSource', 'vAccount', 'mAmount']
    missing = [col for col in required_cols if col not in isbs.columns]
    if missing:
        print(f"Warning: Missing columns in ISBS data: {missing}")
        return 0.0
    
    # Filter for the deal (case-insensitive)
    isbs['vcode'] = isbs['vcode'].astype(str).str.strip().str.lower()
    deal_vcode_lower = str(deal_vcode).strip().lower()
    isbs = isbs[isbs['vcode'] == deal_vcode_lower]
    
    if isbs.empty:
        print(f"Warning: No ISBS data found for deal {deal_vcode}")
        return 0.0
    
    # Filter for Interim BS (balance sheet) - exact match with stripped whitespace
    isbs['vSource'] = isbs['vSource'].astype(str).str.strip()
    isbs = isbs[isbs['vSource'] == 'Interim BS']
    
    if isbs.empty:
        print(f"Warning: No 'Interim BS' records found for deal {deal_vcode}")
        return 0.0
    
    # Convert date column (handle Excel serial dates)
    if 'dtEntry' in isbs.columns:
        # Try to parse as Excel serial date first
        try:
            isbs['dtEntry_parsed'] = pd.to_datetime(isbs['dtEntry'], unit='D', origin='1899-12-30', errors='coerce')
        except:
            isbs['dtEntry_parsed'] = pd.to_datetime(isbs['dtEntry'], errors='coerce')
        
        # If that didn't work, try standard datetime parsing
        null_dates = isbs['dtEntry_parsed'].isna()
        if null_dates.any():
            isbs.loc[null_dates, 'dtEntry_parsed'] = pd.to_datetime(isbs.loc[null_dates, 'dtEntry'], errors='coerce')
        
        # Convert forecast_start_date to datetime
        if not isinstance(forecast_start_date, pd.Timestamp):
            forecast_start_date = pd.to_datetime(forecast_start_date)
        
        # Find most recent entry before forecast start
        isbs = isbs[isbs['dtEntry_parsed'] < forecast_start_date]
        
        if isbs.empty:
            print(f"Warning: No ISBS records before forecast start date {forecast_start_date}")
            return 0.0
        
        # Get the most recent date
        most_recent_date = isbs['dtEntry_parsed'].max()
        isbs = isbs[isbs['dtEntry_parsed'] == most_recent_date]
        print(f"Using ISBS data from: {most_recent_date.strftime('%Y-%m-%d')}")
    
    # Filter for cash accounts
    cash_accounts = ['1012', '1010', '1100', '1120', '1130', '1140', '1145']
    isbs['vAccount'] = isbs['vAccount'].astype(str).str.strip()
    isbs = isbs[isbs['vAccount'].isin(cash_accounts)]
    
    if isbs.empty:
        print(f"Warning: No cash accounts found in ISBS data for deal {deal_vcode}")
        return 0.0
    
    # Sum the amounts
    if 'mAmount' in isbs.columns:
        total_cash = pd.to_numeric(isbs['mAmount'], errors='coerce').sum()
        cash_balance = float(total_cash) if not pd.isna(total_cash) else 0.0
        print(f"Beginning cash balance for {deal_vcode}: ${cash_balance:,.2f}")
        print(f"  From {len(isbs)} cash account records")
        return cash_balance
    
    return 0.0


def build_cash_flow_schedule_from_fad(
    fad_monthly: pd.DataFrame,
    capital_calls: List[Dict],
    beginning_cash: float,
    deal_vcode: str
) -> pd.DataFrame:
    """
    Build period-by-period cash flow schedule from pre-calculated FAD
    
    Args:
        fad_monthly: DataFrame with 'event_date' and 'fad' columns (from cashflows_monthly_fad())
        capital_calls: List of capital call dicts with call_date and amount
        beginning_cash: Starting cash balance
        deal_vcode: Deal identifier
    
    Returns:
        DataFrame with cash schedule by period
    """
    # Create schedule from FAD data
    schedule = fad_monthly.copy()
    
    # Ensure we have required columns
    if 'event_date' not in schedule.columns or 'fad' not in schedule.columns:
        raise ValueError(f"fad_monthly must have 'event_date' and 'fad' columns. Got: {schedule.columns.tolist()}")
    
    # Initialize tracking columns
    schedule['beginning_cash'] = 0.0
    schedule['capital_call'] = 0.0
    schedule['capex_paid'] = 0.0
    schedule['operating_cf'] = schedule['fad'].fillna(0)  # FAD is our operating cash flow
    schedule['deficit_covered'] = 0.0
    schedule['distributable'] = 0.0
    schedule['distributed'] = 0.0  # Will be filled after waterfall
    schedule['ending_cash'] = 0.0
    
    # Sort by date
    schedule = schedule.sort_values('event_date').reset_index(drop=True)
    
    # Add capital calls to schedule
    if capital_calls:
        for call in capital_calls:
            call_date = call.get('call_date')
            amount = call.get('amount', 0)
            
            if call_date and amount:
                # Find matching period or closest future period
                mask = schedule['event_date'] >= call_date
                if mask.any():
                    idx = schedule[mask].index[0]
                    schedule.loc[idx, 'capital_call'] += amount
    
    # Process each period
    cash_balance = beginning_cash
    
    for idx in schedule.index:
        # Start of period
        schedule.loc[idx, 'beginning_cash'] = cash_balance
        
        # Add capital calls
        capital_call = schedule.loc[idx, 'capital_call']
        cash_balance += capital_call
        
        # For now, assume no CapEx since we don't have it in FAD
        # FAD already accounts for CapEx if it's in your calculation
        schedule.loc[idx, 'capex_paid'] = 0
        
        # Handle operating cash flow (FAD)
        operating_cf = schedule.loc[idx, 'operating_cf']
        
        if operating_cf < 0:
            # Negative FAD - cover from cash if available
            deficit = abs(operating_cf)
            deficit_covered = min(deficit, max(0, cash_balance))
            schedule.loc[idx, 'deficit_covered'] = deficit_covered
            cash_balance -= deficit_covered
            schedule.loc[idx, 'distributable'] = 0.0
            
        else:
            # Positive FAD
            if cash_balance < 0:
                # Use FAD to cover cash deficit first
                refill_amount = min(operating_cf, abs(cash_balance))
                cash_balance += refill_amount
                schedule.loc[idx, 'distributable'] = max(0, operating_cf - refill_amount)
            else:
                # All FAD is distributable
                schedule.loc[idx, 'distributable'] = operating_cf
        
        # End of period (before distributions)
        schedule.loc[idx, 'ending_cash'] = cash_balance
    
    return schedule


def build_cash_flow_schedule(
    forecast_df: pd.DataFrame,
    capital_calls: List[Dict],
    beginning_cash: float,
    deal_vcode: str
) -> pd.DataFrame:
    """
    Build period-by-period cash flow schedule with:
    - Cash balance tracking
    - CapEx funding from reserves
    - Operating cash flow
    - Distributable amounts
    
    Args:
        forecast_df: Forecast with event_date, capex, and cash flow columns
        capital_calls: List of capital call dicts with call_date and amount
        beginning_cash: Starting cash balance
        deal_vcode: Deal identifier
    
    Returns:
        DataFrame with cash schedule by period
    """
    # Create a copy of forecast
    schedule = forecast_df.copy()
    
    # Ensure we have required columns
    if 'event_date' not in schedule.columns:
        raise ValueError("forecast_df must have 'event_date' column")
    
    # Identify cash flow column (try multiple variations)
    cf_col = None
    for col in ['fad', 'FAD', 'cash_flow', 'operating_cf']:
        if col in schedule.columns:
            cf_col = col
            break
    
    if cf_col is None:
        raise ValueError("Could not find cash flow column in forecast")
    
    # Identify CapEx column
    capex_col = None
    for col in ['capex', 'CapEx', 'capital_expenditure', 'cap_ex']:
        if col in schedule.columns:
            capex_col = col
            break
    
    # Initialize tracking columns
    schedule['beginning_cash'] = 0.0
    schedule['capital_call'] = 0.0
    schedule['capex_paid'] = 0.0
    schedule['operating_cf'] = schedule[cf_col].fillna(0)
    schedule['deficit_covered'] = 0.0
    schedule['distributable'] = 0.0
    schedule['distributed'] = 0.0  # Will be filled after waterfall
    schedule['ending_cash'] = 0.0
    
    # Sort by date
    schedule = schedule.sort_values('event_date').reset_index(drop=True)
    
    # Add capital calls to schedule
    if capital_calls:
        for call in capital_calls:
            call_date = call.get('call_date')
            amount = call.get('amount', 0)
            
            if call_date and amount:
                # Find matching period or closest future period
                mask = schedule['event_date'] >= call_date
                if mask.any():
                    idx = schedule[mask].index[0]
                    schedule.loc[idx, 'capital_call'] += amount
    
    # Process each period
    cash_balance = beginning_cash
    
    for idx in schedule.index:
        # Start of period
        schedule.loc[idx, 'beginning_cash'] = cash_balance
        
        # Add capital calls
        capital_call = schedule.loc[idx, 'capital_call']
        cash_balance += capital_call
        
        # Pay CapEx from cash
        capex = 0
        if capex_col and capex_col in schedule.columns:
            capex = abs(schedule.loc[idx, capex_col]) if pd.notna(schedule.loc[idx, capex_col]) else 0
        
        capex_paid = min(capex, max(0, cash_balance))
        schedule.loc[idx, 'capex_paid'] = capex_paid
        cash_balance -= capex_paid
        
        # Handle operating cash flow
        operating_cf = schedule.loc[idx, 'operating_cf']
        
        if operating_cf < 0:
            # Negative operating CF - cover from cash if available
            deficit = abs(operating_cf)
            deficit_covered = min(deficit, max(0, cash_balance))
            schedule.loc[idx, 'deficit_covered'] = deficit_covered
            cash_balance -= deficit_covered
            schedule.loc[idx, 'distributable'] = 0.0
            
        else:
            # Positive operating CF
            if cash_balance < 0:
                # Use operating CF to cover cash deficit first
                refill_amount = min(operating_cf, abs(cash_balance))
                cash_balance += refill_amount
                schedule.loc[idx, 'distributable'] = max(0, operating_cf - refill_amount)
            else:
                # All operating CF is distributable
                schedule.loc[idx, 'distributable'] = operating_cf
        
        # End of period (before distributions)
        schedule.loc[idx, 'ending_cash'] = cash_balance
        
        # Note: 'distributed' column will be filled after running waterfall
        # For now, assume distributions don't affect cash (they leave the entity)
        # The waterfall will tell us how much was actually distributed
    
    return schedule


def apply_distributions_to_cash_schedule(
    cash_schedule: pd.DataFrame,
    distributions: pd.DataFrame
) -> pd.DataFrame:
    """
    Update cash schedule with actual distributions from waterfall results
    
    Args:
        cash_schedule: Cash flow schedule from build_cash_flow_schedule
        distributions: Waterfall results with event_date and distributed amounts
    
    Returns:
        Updated cash schedule with distributions applied
    """
    schedule = cash_schedule.copy()
    
    if distributions is None or distributions.empty:
        return schedule
    
    # Merge distributions into schedule
    if 'event_date' in distributions.columns:
        # Sum distributions by date
        dist_by_date = distributions.groupby('event_date').agg({
            'amount': 'sum'  # Adjust column name as needed
        }).reset_index()
        
        # Update schedule
        for idx in schedule.index:
            event_date = schedule.loc[idx, 'event_date']
            match = dist_by_date[dist_by_date['event_date'] == event_date]
            
            if not match.empty:
                distributed = match['amount'].iloc[0]
                schedule.loc[idx, 'distributed'] = distributed
                # Distributions reduce ending cash
                schedule.loc[idx, 'ending_cash'] -= distributed
    
    return schedule


def get_sale_period_total_cash(
    cash_schedule: pd.DataFrame,
    sale_date
) -> Tuple[float, float]:
    """
    Get total cash available at sale (ending cash reserves to distribute)
    
    At sale, all accumulated cash reserves should be distributed along with
    the final period's FAD. This returns the ending_cash which represents
    the cash reserves that have accumulated (beginning cash + capital calls
    - deficits covered).
    
    Args:
        cash_schedule: Cash flow schedule
        sale_date: Date of sale
    
    Returns:
        Tuple of (remaining_cash_balance, total_available_for_distribution)
    """
    if cash_schedule.empty:
        return 0.0, 0.0
    
    # Convert sale_date to match schedule date type
    schedule = cash_schedule.copy()
    schedule['event_date'] = pd.to_datetime(schedule['event_date']).dt.date
    
    if hasattr(sale_date, 'date'):
        sale_date = sale_date.date()
    
    # Find the sale period or last period before/at sale
    schedule_up_to_sale = schedule[schedule['event_date'] <= sale_date]
    
    if schedule_up_to_sale.empty:
        return 0.0, 0.0
    
    # Get ending cash from the last period up to (and including) sale date
    # This represents accumulated reserves to be distributed at sale
    last_idx = schedule_up_to_sale.index[-1]
    ending_cash = float(schedule_up_to_sale.loc[last_idx, 'ending_cash'])
    
    remaining_cash = max(0.0, ending_cash)  # Only positive cash adds to distribution
    
    return remaining_cash, remaining_cash


def summarize_cash_usage(cash_schedule: pd.DataFrame) -> Dict:
    """
    Create summary statistics of cash usage over the forecast period
    
    Returns:
        Dictionary with summary metrics
    """
    if cash_schedule.empty:
        return {}
    
    summary = {
        'beginning_cash': cash_schedule['beginning_cash'].iloc[0],
        'total_capital_calls': cash_schedule['capital_call'].sum(),
        'total_capex_paid': cash_schedule['capex_paid'].sum(),
        'total_deficits_covered': cash_schedule['deficit_covered'].sum(),
        'total_distributable': cash_schedule['distributable'].sum(),
        'total_distributed': cash_schedule['distributed'].sum(),
        'ending_cash': cash_schedule['ending_cash'].iloc[-1],
        'min_cash_balance': cash_schedule['ending_cash'].min(),
        'max_cash_balance': cash_schedule['ending_cash'].max(),
        'periods_with_deficit': (cash_schedule['operating_cf'] < 0).sum(),
        'periods_with_distributions': (cash_schedule['distributable'] > 0).sum()
    }
    
    return summary
