"""
capital_calls.py
Functions for loading and processing capital calls data
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict

from config import typename_to_pool


def load_capital_calls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Load and normalize capital calls data from CSV
    
    Expected columns:
    - investor_id or investor: Investor identifier
    - call_date or date: Date of capital call
    - amount: Amount being called
    - deal_name or vcode: Deal identifier (optional)
    """
    # Make a copy to avoid modifying original
    cc = df.copy()
    
    # Normalize column names
    cc.columns = [str(c).strip().lower() for c in cc.columns]
    
    # Handle different column name variations
    if 'investor' in cc.columns and 'investor_id' not in cc.columns:
        cc = cc.rename(columns={'investor': 'investor_id'})
    if 'propcode' in cc.columns and 'investor_id' not in cc.columns:
        cc = cc.rename(columns={'propcode': 'investor_id'})

    if 'calldate' in cc.columns and 'call_date' not in cc.columns:
        cc = cc.rename(columns={'calldate': 'call_date'})
    if 'date' in cc.columns and 'call_date' not in cc.columns:
        cc = cc.rename(columns={'date': 'call_date'})

    if 'vcode' in cc.columns and 'deal_name' not in cc.columns:
        cc = cc.rename(columns={'vcode': 'deal_name'})

    # Normalize typename column (for pool routing)
    if 'typename' not in cc.columns:
        cc['typename'] = 'Contribution: Investments'

    # Convert date to datetime
    if 'call_date' in cc.columns:
        cc['call_date'] = pd.to_datetime(cc['call_date'])
    
    # Convert amount to numeric
    if 'amount' in cc.columns:
        cc['amount'] = pd.to_numeric(cc['amount'], errors='coerce')
    
    # Sort by date
    if 'call_date' in cc.columns:
        cc = cc.sort_values('call_date')
    
    return cc


def build_capital_call_schedule(cc_df: pd.DataFrame, deal_vcode: str = None) -> List[Dict]:
    """
    Build a list of capital call events from the dataframe
    
    Returns:
        List of dicts with keys: investor_id, call_date, amount, deal_name
    """
    if cc_df is None or cc_df.empty:
        return []
    
    # Filter by deal if specified
    if deal_vcode and 'deal_name' in cc_df.columns:
        cc_df = cc_df[cc_df['deal_name'] == deal_vcode].copy()
    
    # Build schedule
    schedule = []
    for _, row in cc_df.iterrows():
        call = {
            'investor_id': row.get('investor_id', 'Unknown'),
            'call_date': row.get('call_date'),
            'amount': row.get('amount', 0),
            'deal_name': row.get('deal_name', deal_vcode),
            'typename': row.get('typename', 'Contribution: Investments'),
        }
        schedule.append(call)

    return schedule


def apply_capital_calls_to_states(capital_calls: List[Dict], investor_states: Dict) -> Dict:
    """
    Apply capital calls to investor states by routing to the correct
    capital pool and recording contribution cashflows.

    Pool routing uses the 'typename' field in each call dict via
    typename_to_pool().  Falls back to 'initial' when typename is
    absent or unrecognised.

    Args:
        capital_calls: List of capital call dicts
            (investor_id, call_date, amount, typename)
        investor_states: Dict of PropCode -> InvestorState

    Returns:
        Updated investor_states dict
    """
    if not capital_calls:
        return investor_states

    for call in capital_calls:
        inv_id = call.get('investor_id')
        amount = call.get('amount', 0)
        call_date = call.get('call_date')

        if not inv_id or not amount:
            continue

        if inv_id in investor_states:
            stt = investor_states[inv_id]
            # Route to correct pool
            pool_name = typename_to_pool(call.get('typename', ''))
            pool = stt.get_pool(pool_name)
            pool.capital_outstanding += abs(amount)

            # Operating capital: update cumulative cap
            if pool_name == "operating":
                if pool.cumulative_cap is None:
                    pool.cumulative_cap = 0.0
                pool.cumulative_cap += abs(amount)

            # Record as contribution cashflow (negative) for XIRR
            if call_date is not None:
                d = call_date.date() if hasattr(call_date, 'date') else call_date
                stt.cashflows.append((d, -abs(amount)))

    return investor_states


def integrate_capital_calls_with_forecast(forecast_df: pd.DataFrame, capital_calls: List[Dict]) -> pd.DataFrame:
    """
    Integrate capital calls into the forecast as cash inflows
    
    Args:
        forecast_df: Forecast dataframe with date/period columns
        capital_calls: List of capital call dicts
    
    Returns:
        Updated forecast dataframe
    """
    if not capital_calls or forecast_df is None or forecast_df.empty:
        return forecast_df
    
    fc = forecast_df.copy()
    
    # Try to find date or period column
    date_col = None
    if 'date' in fc.columns:
        date_col = 'date'
    elif 'period' in fc.columns:
        date_col = 'period'
    elif 'year' in fc.columns:
        date_col = 'year'
    
    if not date_col:
        # Can't integrate without a time dimension
        return fc
    
    # Ensure date column is datetime
    if fc[date_col].dtype != 'datetime64[ns]':
        fc[date_col] = pd.to_datetime(fc[date_col], errors='coerce')
    
    # Add capital calls column if it doesn't exist
    if 'capital_calls' not in fc.columns:
        fc['capital_calls'] = 0.0
    
    # Map each call to the appropriate period
    for call in capital_calls:
        call_date = call.get('call_date')
        amount = call.get('amount', 0)
        
        if call_date and amount:
            # Find the matching period
            mask = fc[date_col] == call_date
            if mask.any():
                fc.loc[mask, 'capital_calls'] += amount
            else:
                # Find closest period
                fc['date_diff'] = abs((fc[date_col] - call_date).dt.days)
                closest_idx = fc['date_diff'].idxmin()
                fc.loc[closest_idx, 'capital_calls'] += amount
                fc = fc.drop('date_diff', axis=1)
    
    return fc


def capital_calls_summary_table(capital_calls: List[Dict]) -> pd.DataFrame:
    """
    Create a summary table of capital calls
    
    Returns:
        DataFrame with columns: call_date, investor_id, amount, deal_name
    """
    if not capital_calls:
        return pd.DataFrame(columns=['call_date', 'investor_id', 'amount', 'deal_name'])
    
    df = pd.DataFrame(capital_calls)
    
    # Format the date
    if 'call_date' in df.columns:
        df['call_date'] = pd.to_datetime(df['call_date']).dt.strftime('%Y-%m-%d')
    
    # Format the amount
    if 'amount' in df.columns:
        df['amount'] = df['amount'].apply(lambda x: f"${x:,.0f}")
    
    # Select and order columns
    cols = ['call_date', 'investor_id', 'amount', 'deal_name']
    available_cols = [c for c in cols if c in df.columns]
    
    return df[available_cols]


def capital_calls_by_investor(capital_calls: List[Dict]) -> pd.DataFrame:
    """
    Summarize capital calls by investor
    
    Returns:
        DataFrame with columns: investor_id, total_calls, num_calls, avg_call, first_call, last_call
    """
    if not capital_calls:
        return pd.DataFrame(columns=['investor_id', 'total_calls', 'num_calls'])
    
    df = pd.DataFrame(capital_calls)
    
    # Group by investor
    summary = df.groupby('investor_id').agg({
        'amount': ['sum', 'count', 'mean'],
        'call_date': ['min', 'max']
    }).reset_index()
    
    # Flatten column names
    summary.columns = ['investor_id', 'total_calls', 'num_calls', 'avg_call', 'first_call', 'last_call']
    
    # Format amounts
    summary['total_calls'] = summary['total_calls'].apply(lambda x: f"${x:,.0f}")
    summary['avg_call'] = summary['avg_call'].apply(lambda x: f"${x:,.0f}")
    
    # Format dates
    summary['first_call'] = pd.to_datetime(summary['first_call']).dt.strftime('%Y-%m-%d')
    summary['last_call'] = pd.to_datetime(summary['last_call']).dt.strftime('%Y-%m-%d')
    
    return summary


def validate_capital_calls(cc_df: pd.DataFrame) -> List[str]:
    """
    Validate capital calls data and return list of issues
    
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    if cc_df is None or cc_df.empty:
        errors.append("Capital calls dataframe is empty")
        return errors
    
    # Check required columns
    required = ['investor_id', 'call_date', 'amount']
    for col in required:
        if col not in cc_df.columns:
            errors.append(f"Missing required column: {col}")
    
    # Check for null values
    if 'investor_id' in cc_df.columns and cc_df['investor_id'].isnull().any():
        errors.append("Found null values in investor_id column")
    
    if 'amount' in cc_df.columns:
        if cc_df['amount'].isnull().any():
            errors.append("Found null values in amount column")
        if (cc_df['amount'] < 0).any():
            errors.append("Found negative amounts")
    
    if 'call_date' in cc_df.columns and cc_df['call_date'].isnull().any():
        errors.append("Found null values in call_date column")
    
    return errors
