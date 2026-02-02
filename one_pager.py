"""
one_pager.py
Core data retrieval and calculation functions for One Pager Investor Report

Provides functions to extract and calculate:
- General information from investment_map
- Capitalization stack from MRI_Loans, MRI_VAL, waterfalls, commitments
- Property performance from ISBS_Download
- PE performance from accounting_feed, commitments, waterfalls
- Chart data for NOI/Occupancy trends
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from database import execute_query, get_db_connection


# ============================================================
# QUARTER UTILITIES
# ============================================================

def get_quarter_from_date(d: date) -> str:
    """Convert date to quarter string (e.g., '2025-Q4')"""
    quarter = (d.month - 1) // 3 + 1
    return f"{d.year}-Q{quarter}"


def quarter_to_date_range(quarter_str: str) -> Tuple[date, date]:
    """
    Convert quarter string to (start_date, end_date)

    Args:
        quarter_str: Quarter in format 'YYYY-QN' (e.g., '2025-Q4')

    Returns:
        Tuple of (quarter_start_date, quarter_end_date)
    """
    year = int(quarter_str.split('-')[0])
    q = int(quarter_str.split('Q')[1])

    start_month = (q - 1) * 3 + 1
    end_month = q * 3

    start_date = date(year, start_month, 1)

    # Get last day of quarter
    if end_month == 12:
        end_date = date(year, 12, 31)
    else:
        end_date = date(year, end_month + 1, 1) - timedelta(days=1)

    return start_date, end_date


def get_year_start(quarter_str: str) -> date:
    """Get January 1st of the year for a quarter"""
    year = int(quarter_str.split('-')[0])
    return date(year, 1, 1)


def get_available_quarters(isbs_df: pd.DataFrame) -> List[str]:
    """
    Get list of available quarters from ISBS actual data

    Args:
        isbs_df: ISBS DataFrame with dtEntry and vSource columns

    Returns:
        Sorted list of quarter strings (most recent first)
    """
    if isbs_df is None or isbs_df.empty:
        return []

    df = isbs_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Filter for actual data only
    if 'vSource' in df.columns:
        df = df[df['vSource'].astype(str).str.strip() == 'Interim IS']

    if 'dtEntry' not in df.columns:
        return []

    # Parse dates
    try:
        df['dtEntry_parsed'] = pd.to_datetime(df['dtEntry'], unit='D', origin='1899-12-30', errors='coerce')
    except:
        df['dtEntry_parsed'] = pd.to_datetime(df['dtEntry'], errors='coerce')

    # Get unique quarters
    quarters = set()
    for dt in df['dtEntry_parsed'].dropna():
        quarters.add(get_quarter_from_date(dt.date()))

    return sorted(list(quarters), reverse=True)


def get_trailing_quarters(quarter_str: str, count: int = 10) -> List[str]:
    """
    Get list of trailing quarters including the specified quarter

    Args:
        quarter_str: Starting quarter (e.g., '2025-Q4')
        count: Number of quarters to return

    Returns:
        List of quarter strings from oldest to newest
    """
    year = int(quarter_str.split('-')[0])
    q = int(quarter_str.split('Q')[1])

    quarters = []
    for _ in range(count):
        quarters.append(f"{year}-Q{q}")
        q -= 1
        if q == 0:
            q = 4
            year -= 1

    return list(reversed(quarters))


# ============================================================
# GENERAL INFORMATION
# ============================================================

def get_general_information(inv_map: pd.DataFrame, vcode: str) -> Dict[str, Any]:
    """
    Get general deal information from investment_map

    Args:
        inv_map: Investment map DataFrame
        vcode: Deal vcode

    Returns:
        Dictionary with general info fields
    """
    info = {
        'partner': '',
        'asset_type': '',
        'location': '',
        'investment_strategy': '',
        'units': 0,
        'sqft': 0,
        'date_closed': None,
        'year_built': '',
        'anticipated_exit': None,
        'investment_name': '',
    }

    if inv_map is None or inv_map.empty:
        return info

    df = inv_map.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Normalize vcode column
    if 'vcode' not in df.columns and 'vCode' in df.columns:
        df = df.rename(columns={'vCode': 'vcode'})

    df['vcode'] = df['vcode'].astype(str).str.strip()
    deal_row = df[df['vcode'] == str(vcode).strip()]

    if deal_row.empty:
        return info

    row = deal_row.iloc[0]

    # Map columns (handle various possible column names)
    col_mappings = {
        'partner': ['Operating_Partner', 'Partner', 'vPartner', 'partner'],
        'asset_type': ['Asset_Type', 'AssetType', 'vAssetType', 'asset_type'],
        'location': ['City', 'Location', 'vCity', 'location'],
        'investment_strategy': ['Investment_Strategy', 'Lifecycle', 'InvestmentStrategy', 'Strategy', 'vStrategy'],
        'units': ['Total_Units', 'Units', 'iUnits', 'units'],
        'sqft': ['Size_Sqf', 'SF', 'SquareFeet', 'SqFt', 'sqft', 'mSF'],
        'date_closed': ['Acquisition_Date', 'DateClosed', 'Date_Closed', 'dtClosed', 'ClosingDate'],
        'year_built': ['Year_Built', 'YearBuilt', 'iYearBuilt'],
        'anticipated_exit': ['Sale_Date', 'AnticipatedExit', 'Anticipated_Exit', 'dtExit', 'ExitDate'],
        'investment_name': ['Investment_Name', 'InvestmentName', 'vName'],
    }

    for key, possible_cols in col_mappings.items():
        for col in possible_cols:
            if col in row.index and pd.notna(row[col]):
                val = row[col]
                if key in ['units', 'sqft']:
                    info[key] = int(float(val)) if pd.notna(val) else 0
                elif key in ['date_closed', 'anticipated_exit']:
                    try:
                        info[key] = pd.to_datetime(val).date()
                    except:
                        info[key] = None
                else:
                    info[key] = str(val).strip()
                break

    return info


# ============================================================
# CAPITALIZATION / EXPOSURE / DEAL TERMS
# ============================================================

def get_capitalization_stack(
    vcode: str,
    mri_loans: pd.DataFrame,
    mri_val: pd.DataFrame,
    waterfalls: pd.DataFrame,
    commitments: pd.DataFrame,
    acct: pd.DataFrame,
    inv_map: pd.DataFrame
) -> Dict[str, Any]:
    """
    Get capitalization stack and deal terms

    Args:
        vcode: Deal vcode
        mri_loans: Loans DataFrame
        mri_val: Valuations DataFrame
        waterfalls: Waterfalls DataFrame
        commitments: Commitments DataFrame
        acct: Accounting feed DataFrame
        inv_map: Investment map DataFrame

    Returns:
        Dictionary with cap stack data
    """
    cap = {
        'purchase_price': 0.0,
        'pe_coupon': 0.0,  # From waterfalls nPercent
        'pe_participation': 0.0,  # From waterfalls FXRate where vState='Share'
        'loan_maturity': None,
        'loan_rate': 0.0,
        'loan_type': '',  # Fixed/Variable
        'rate_cap': None,
        'debt': 0.0,
        'debt_pct': 0.0,
        'pref_equity': 0.0,
        'pref_equity_pct': 0.0,
        'partner_equity': 0.0,
        'partner_equity_pct': 0.0,
        'total_cap': 0.0,
        'current_valuation': 0.0,
        'pe_exposure_on_cap': 0.0,
        'pe_exposure_on_value': 0.0,
        'committed_pe': 0.0,
    }

    vcode_str = str(vcode).strip()

    # Get debt from MRI_Loans
    if mri_loans is not None and not mri_loans.empty:
        loans = mri_loans.copy()
        loans.columns = [str(c).strip() for c in loans.columns]
        if 'vCode' not in loans.columns and 'vcode' in loans.columns:
            loans = loans.rename(columns={'vcode': 'vCode'})
        if 'vCode' in loans.columns:
            loans['vCode'] = loans['vCode'].astype(str).str.strip()
            deal_loans = loans[loans['vCode'] == vcode_str]

            if not deal_loans.empty:
                if 'mOrigLoanAmt' in deal_loans.columns:
                    cap['debt'] = pd.to_numeric(deal_loans['mOrigLoanAmt'], errors='coerce').fillna(0).sum()

                # Get loan terms from most recent/primary loan
                loan_row = deal_loans.iloc[0]

                if 'dtEvent' in deal_loans.columns or 'dtMaturity' in deal_loans.columns:
                    mat_col = 'dtMaturity' if 'dtMaturity' in deal_loans.columns else 'dtEvent'
                    try:
                        cap['loan_maturity'] = pd.to_datetime(loan_row[mat_col]).date()
                    except:
                        pass

                if 'nRate' in deal_loans.columns:
                    rate = pd.to_numeric(loan_row['nRate'], errors='coerce')
                    if pd.notna(rate):
                        cap['loan_rate'] = rate if rate < 1 else rate / 100

                if 'vIntType' in deal_loans.columns:
                    cap['loan_type'] = str(loan_row['vIntType']).strip()

    # Get valuation from MRI_VAL
    if mri_val is not None and not mri_val.empty:
        val = mri_val.copy()
        val.columns = [str(c).strip() for c in val.columns]
        if 'vcode' not in val.columns and 'vCode' in val.columns:
            val = val.rename(columns={'vCode': 'vcode'})
        if 'vcode' in val.columns:
            val['vcode'] = val['vcode'].astype(str).str.strip()
            deal_val = val[val['vcode'] == vcode_str]

            if not deal_val.empty:
                val_row = deal_val.iloc[-1]  # Most recent

                if 'mIncomeCapConcludedValue' in deal_val.columns:
                    v = pd.to_numeric(val_row['mIncomeCapConcludedValue'], errors='coerce')
                    cap['current_valuation'] = float(v) if pd.notna(v) else 0.0

                # Purchase price might be in a separate column or use initial valuation
                if 'mPurchasePrice' in deal_val.columns:
                    pp = pd.to_numeric(val_row['mPurchasePrice'], errors='coerce')
                    cap['purchase_price'] = float(pp) if pd.notna(pp) else 0.0

    # Get PE coupon and participation from waterfalls
    if waterfalls is not None and not waterfalls.empty:
        wf = waterfalls.copy()
        wf.columns = [str(c).strip() for c in wf.columns]
        if 'vcode' not in wf.columns and 'vCode' in wf.columns:
            wf = wf.rename(columns={'vCode': 'vcode'})
        if 'vcode' in wf.columns:
            wf['vcode'] = wf['vcode'].astype(str).str.strip()
            deal_wf = wf[wf['vcode'] == vcode_str]

            if not deal_wf.empty:
                # Get coupon from nPercent (first pref return row)
                pref_rows = deal_wf[deal_wf['vState'].astype(str).str.strip().str.lower() == 'pref']
                if not pref_rows.empty:
                    coupon = pd.to_numeric(pref_rows.iloc[0]['nPercent'], errors='coerce')
                    if pd.notna(coupon):
                        cap['pe_coupon'] = coupon if coupon < 1 else coupon / 100

                # Get participation from FXRate where vState='Share'
                share_rows = deal_wf[deal_wf['vState'].astype(str).str.strip().str.lower() == 'share']
                if not share_rows.empty:
                    part = pd.to_numeric(share_rows.iloc[0]['FXRate'], errors='coerce')
                    if pd.notna(part):
                        cap['pe_participation'] = part if part < 1 else part / 100

    # Get committed PE from commitments
    if commitments is not None and not commitments.empty:
        comm = commitments.copy()
        comm.columns = [str(c).strip() for c in comm.columns]
        # Filter by vcode or EntityID
        if 'vcode' in comm.columns:
            comm['vcode'] = comm['vcode'].astype(str).str.strip()
            deal_comm = comm[comm['vcode'] == vcode_str]
            if not deal_comm.empty and 'CommittedAmount' in deal_comm.columns:
                cap['committed_pe'] = pd.to_numeric(deal_comm['CommittedAmount'], errors='coerce').fillna(0).sum()

    # Get equity from accounting feed
    if acct is not None and not acct.empty and inv_map is not None:
        from loaders import build_investmentid_to_vcode

        try:
            inv_to_vcode = build_investmentid_to_vcode(inv_map)
            deal_investment_ids = [iid for iid, vc in inv_to_vcode.items() if str(vc) == vcode_str]

            acct_norm = acct.copy()
            acct_norm.columns = [str(c).strip() for c in acct_norm.columns]
            acct_norm["InvestmentID"] = acct_norm["InvestmentID"].astype(str).str.strip()
            deal_acct = acct_norm[acct_norm["InvestmentID"].isin(deal_investment_ids)].copy()

            if not deal_acct.empty:
                deal_acct["MajorType"] = deal_acct["MajorType"].fillna("").astype(str).str.strip()
                deal_acct["Amt"] = pd.to_numeric(deal_acct["Amt"], errors="coerce").fillna(0.0)

                if "TypeName" not in deal_acct.columns and "Typename" in deal_acct.columns:
                    deal_acct["TypeName"] = deal_acct["Typename"]
                elif "TypeName" not in deal_acct.columns:
                    deal_acct["TypeName"] = ""
                deal_acct["TypeName"] = deal_acct["TypeName"].fillna("").astype(str).str.strip()
                deal_acct["InvestorID"] = deal_acct["InvestorID"].astype(str).str.strip()

                investor_balances = {}
                for _, row in deal_acct.iterrows():
                    investor_id = row["InvestorID"]
                    major_type = row["MajorType"].lower()
                    type_name = row["TypeName"].lower()
                    amt = float(row["Amt"])

                    if investor_id not in investor_balances:
                        investor_balances[investor_id] = 0.0

                    if "contrib" in major_type:
                        investor_balances[investor_id] += abs(amt)
                    if "distri" in major_type and "return of capital" in type_name:
                        investor_balances[investor_id] -= abs(amt)

                for investor_id, balance in investor_balances.items():
                    if investor_id.upper().startswith("OP"):
                        cap['partner_equity'] += max(0, balance)
                    else:
                        cap['pref_equity'] += max(0, balance)
        except Exception as e:
            pass

    # Calculate totals and percentages
    cap['total_cap'] = cap['debt'] + cap['pref_equity'] + cap['partner_equity']

    if cap['total_cap'] > 0:
        cap['debt_pct'] = cap['debt'] / cap['total_cap']
        cap['pref_equity_pct'] = cap['pref_equity'] / cap['total_cap']
        cap['partner_equity_pct'] = cap['partner_equity'] / cap['total_cap']
        cap['pe_exposure_on_cap'] = (cap['debt'] + cap['pref_equity']) / cap['total_cap']

    if cap['current_valuation'] > 0:
        cap['pe_exposure_on_value'] = (cap['debt'] + cap['pref_equity']) / cap['current_valuation']

    return cap


# ============================================================
# PROPERTY PERFORMANCE
# ============================================================

# Income Statement account classifications (matching app.py)
IS_ACCOUNTS = {
    'REVENUES': {
        'Rental Income': ['4010', '4012'],
        'Commercial': ['4020', '4041'],
        'Abated Apartments': ['4045'],
        'Vacancy': ['4040', '4043', '4030', '4042'],
        'RUBS': ['4070'],
        'RET': ['4091'],
        'INS': ['4092'],
        'CAM': ['4090', '4097', '4093', '4094', '4096', '4095'],
        'Other Income': ['4063', '4060', '4061', '4062', '4080', '4065'],
    },
    'EXPENSES': {
        'Real Estate Taxes': ['5090'],
        'Property & Liability Insurance': ['5110', '5114'],
        'Salary & Benefits': ['5018', '5010', '5016', '5012', '5014'],
        'Utilities': ['5051', '5053', '5050', '5052', '5054', '5055'],
        'Repairs & Maintenance': ['5060', '5067', '5063', '5069', '5061', '5064', '5065', '5068', '5070', '5066'],
        'Administrative': ['5020', '5022', '5021', '5023', '5025', '5026', '5080'],
        'Marketing & Advertising': ['5045'],
        'Legal & Professional': ['5087', '5085'],
        'Management Fee': ['5040'],
        'Other Expenses': ['5096', '5095', '5091', '5100'],
    },
    'DEBT_SERVICE': {
        'Interest': ['5190'],
        'Principal': ['2145', '2150', '2152', '2154', '2156'],
    },
}


def get_property_performance(
    vcode: str,
    quarter_str: str,
    isbs_df: pd.DataFrame,
    mri_val: pd.DataFrame,
    occupancy_df: pd.DataFrame = None
) -> Dict[str, Any]:
    """
    Get property performance metrics for a quarter

    Args:
        vcode: Deal vcode
        quarter_str: Quarter string (e.g., '2025-Q4')
        isbs_df: ISBS DataFrame with income statement data
        mri_val: Valuations DataFrame for At Close data
        occupancy_df: Occupancy DataFrame

    Returns:
        Dictionary with performance metrics
    """
    perf = {
        'economic_occ': {'ytd_actual': None, 'ytd_budget': None, 'variance': None, 'at_close': None, 'actual_ye': None, 'uw_ye': None},
        'revenue': {'ytd_actual': 0, 'ytd_budget': 0, 'variance': 0, 'at_close': 0, 'actual_ye': 0, 'uw_ye': 0},
        'expenses': {'ytd_actual': 0, 'ytd_budget': 0, 'variance': 0, 'at_close': 0, 'actual_ye': 0, 'uw_ye': 0},
        'noi': {'ytd_actual': 0, 'ytd_budget': 0, 'variance': 0, 'at_close': 0, 'actual_ye': 0, 'uw_ye': 0},
        'dscr': {'ytd_actual': None, 'ytd_budget': None, 'variance': None, 'at_close': None, 'actual_ye': None, 'uw_ye': None},
    }

    if isbs_df is None or isbs_df.empty:
        return perf

    vcode_str = str(vcode).strip().lower()
    _, quarter_end = quarter_to_date_range(quarter_str)
    year_start = get_year_start(quarter_str)

    # Prepare ISBS data
    isbs = isbs_df.copy()
    isbs.columns = [str(c).strip() for c in isbs.columns]

    if 'vcode' in isbs.columns:
        isbs['vcode'] = isbs['vcode'].astype(str).str.strip().str.lower()
        isbs = isbs[isbs['vcode'] == vcode_str]

    if isbs.empty:
        return perf

    # Parse dates
    if 'dtEntry' in isbs.columns:
        try:
            isbs['dtEntry_parsed'] = pd.to_datetime(isbs['dtEntry'], unit='D', origin='1899-12-30', errors='coerce')
        except:
            isbs['dtEntry_parsed'] = pd.to_datetime(isbs['dtEntry'], errors='coerce')

    # Normalize columns
    if 'vSource' in isbs.columns:
        isbs['vSource'] = isbs['vSource'].astype(str).str.strip()
    if 'vAccount' in isbs.columns:
        isbs['vAccount'] = isbs['vAccount'].astype(str).str.strip()
    if 'mAmount' in isbs.columns:
        isbs['mAmount'] = pd.to_numeric(isbs['mAmount'], errors='coerce').fillna(0)

    # Helper to calculate amounts from ISBS data
    def calc_amounts(data_df, as_of_date=None, sum_range=None):
        """Calculate revenue, expense, NOI from ISBS data"""
        if data_df.empty:
            return 0, 0, 0, 0  # revenue, expenses, noi, debt_service

        if as_of_date is not None:
            # Cumulative as of date
            period_data = data_df[data_df['dtEntry_parsed'] == as_of_date]
        elif sum_range is not None:
            # Sum over date range
            start, end = sum_range
            period_data = data_df[(data_df['dtEntry_parsed'] > start) & (data_df['dtEntry_parsed'] <= end)]
        else:
            period_data = data_df

        if period_data.empty:
            return 0, 0, 0, 0

        revenue = 0
        expenses = 0
        debt_service = 0

        # Revenue accounts are stored as negative (credit convention) - negate to get positive
        for category, acct_list in IS_ACCOUNTS['REVENUES'].items():
            revenue += -period_data[period_data['vAccount'].isin(acct_list)]['mAmount'].sum()

        # Expense accounts are stored as positive (debit convention)
        for category, acct_list in IS_ACCOUNTS['EXPENSES'].items():
            expenses += period_data[period_data['vAccount'].isin(acct_list)]['mAmount'].sum()

        for category, acct_list in IS_ACCOUNTS['DEBT_SERVICE'].items():
            debt_service += period_data[period_data['vAccount'].isin(acct_list)]['mAmount'].sum()

        noi = revenue - expenses
        return revenue, expenses, noi, debt_service

    # Get actual data (Interim IS)
    actual_data = isbs[isbs['vSource'] == 'Interim IS']
    budget_data = isbs[isbs['vSource'] == 'Budget IS']
    uw_data = isbs[isbs['vSource'] == 'Projected IS']

    # Find the as-of date for YTD actual (last date in quarter or closest available)
    if not actual_data.empty:
        actual_periods = sorted(actual_data['dtEntry_parsed'].dropna().unique())
        # Find period closest to or before quarter end
        ytd_date = None
        for p in reversed(actual_periods):
            if pd.Timestamp(p).date() <= quarter_end:
                ytd_date = pd.Timestamp(p)
                break

        if ytd_date:
            rev, exp, noi, ds = calc_amounts(actual_data, as_of_date=ytd_date)
            perf['revenue']['ytd_actual'] = rev
            perf['expenses']['ytd_actual'] = exp
            perf['noi']['ytd_actual'] = noi
            if ds > 0:
                perf['dscr']['ytd_actual'] = noi / ds

    # Get YTD budget
    if not budget_data.empty:
        # Sum budget from year start to quarter end
        jan1 = pd.Timestamp(f"{quarter_str.split('-')[0]}-01-01") - pd.DateOffset(days=1)
        qtr_end = pd.Timestamp(quarter_end)
        rev, exp, noi, ds = calc_amounts(budget_data, sum_range=(jan1, qtr_end))
        perf['revenue']['ytd_budget'] = rev
        perf['expenses']['ytd_budget'] = exp
        perf['noi']['ytd_budget'] = noi
        if ds > 0:
            perf['dscr']['ytd_budget'] = noi / ds

    # Get U/W YE (full year projected)
    if not uw_data.empty:
        year = int(quarter_str.split('-')[0])
        jan1 = pd.Timestamp(f"{year}-01-01") - pd.DateOffset(days=1)
        dec31 = pd.Timestamp(f"{year}-12-31")
        rev, exp, noi, ds = calc_amounts(uw_data, sum_range=(jan1, dec31))
        perf['revenue']['uw_ye'] = rev
        perf['expenses']['uw_ye'] = exp
        perf['noi']['uw_ye'] = noi
        if ds > 0:
            perf['dscr']['uw_ye'] = noi / ds

    # Calculate variances
    for metric in ['revenue', 'expenses', 'noi']:
        perf[metric]['variance'] = perf[metric]['ytd_actual'] - perf[metric]['ytd_budget']

    if perf['dscr']['ytd_actual'] is not None and perf['dscr']['ytd_budget'] is not None:
        perf['dscr']['variance'] = perf['dscr']['ytd_actual'] - perf['dscr']['ytd_budget']

    # Get occupancy from occupancy_df if available
    if occupancy_df is not None and not occupancy_df.empty:
        occ = occupancy_df.copy()
        occ.columns = [str(c).strip() for c in occ.columns]
        if 'vCode' in occ.columns or 'vcode' in occ.columns:
            vcode_col = 'vCode' if 'vCode' in occ.columns else 'vcode'
            occ[vcode_col] = occ[vcode_col].astype(str).str.strip().str.lower()
            occ = occ[occ[vcode_col] == vcode_str]

            if not occ.empty and 'Qtr' in occ.columns:
                occ['Qtr'] = occ['Qtr'].astype(str).str.strip()
                qtr_occ = occ[occ['Qtr'] == quarter_str]
                if not qtr_occ.empty and 'OccupancyPercent' in qtr_occ.columns:
                    perf['economic_occ']['ytd_actual'] = pd.to_numeric(qtr_occ.iloc[0]['OccupancyPercent'], errors='coerce')

    return perf


# ============================================================
# PREFERRED EQUITY PERFORMANCE
# ============================================================

def get_pe_performance(
    vcode: str,
    quarter_str: str,
    acct: pd.DataFrame,
    commitments: pd.DataFrame,
    waterfalls: pd.DataFrame,
    inv_map: pd.DataFrame
) -> Dict[str, Any]:
    """
    Get Preferred Equity performance metrics

    Args:
        vcode: Deal vcode
        quarter_str: Quarter string
        acct: Accounting feed DataFrame
        commitments: Commitments DataFrame
        waterfalls: Waterfalls DataFrame
        inv_map: Investment map DataFrame

    Returns:
        Dictionary with PE performance metrics
    """
    pe = {
        'committed_pe': 0.0,
        'remaining_to_fund': 0.0,
        'coupon': 0.0,
        'participation': 0.0,
        'funded_to_date': 0.0,
        'return_of_capital': 0.0,
        'roe_to_date': 0.0,
        'uw_roe_to_date': 0.0,
        'current_pe_balance': 0.0,
        'accrued_balance': 0.0,
    }

    vcode_str = str(vcode).strip()
    _, quarter_end = quarter_to_date_range(quarter_str)

    # Get coupon and participation from waterfalls
    if waterfalls is not None and not waterfalls.empty:
        wf = waterfalls.copy()
        wf.columns = [str(c).strip() for c in wf.columns]
        if 'vcode' not in wf.columns and 'vCode' in wf.columns:
            wf = wf.rename(columns={'vCode': 'vcode'})
        if 'vcode' in wf.columns:
            wf['vcode'] = wf['vcode'].astype(str).str.strip()
            deal_wf = wf[wf['vcode'] == vcode_str]

            if not deal_wf.empty:
                pref_rows = deal_wf[deal_wf['vState'].astype(str).str.strip().str.lower() == 'pref']
                if not pref_rows.empty:
                    coupon = pd.to_numeric(pref_rows.iloc[0]['nPercent'], errors='coerce')
                    if pd.notna(coupon):
                        pe['coupon'] = coupon if coupon < 1 else coupon / 100

                share_rows = deal_wf[deal_wf['vState'].astype(str).str.strip().str.lower() == 'share']
                if not share_rows.empty:
                    part = pd.to_numeric(share_rows.iloc[0]['FXRate'], errors='coerce')
                    if pd.notna(part):
                        pe['participation'] = part if part < 1 else part / 100

    # Get committed PE from commitments
    if commitments is not None and not commitments.empty:
        comm = commitments.copy()
        comm.columns = [str(c).strip() for c in comm.columns]
        if 'vcode' in comm.columns:
            comm['vcode'] = comm['vcode'].astype(str).str.strip()
            deal_comm = comm[comm['vcode'] == vcode_str]
            if not deal_comm.empty and 'CommittedAmount' in deal_comm.columns:
                pe['committed_pe'] = pd.to_numeric(deal_comm['CommittedAmount'], errors='coerce').fillna(0).sum()

    # Get funded and ROC from accounting feed
    if acct is not None and not acct.empty and inv_map is not None:
        from loaders import build_investmentid_to_vcode

        try:
            inv_to_vcode = build_investmentid_to_vcode(inv_map)
            deal_investment_ids = [iid for iid, vc in inv_to_vcode.items() if str(vc) == vcode_str]

            acct_norm = acct.copy()
            acct_norm.columns = [str(c).strip() for c in acct_norm.columns]
            acct_norm["InvestmentID"] = acct_norm["InvestmentID"].astype(str).str.strip()
            acct_norm["EffectiveDate"] = pd.to_datetime(acct_norm["EffectiveDate"], errors='coerce')

            # Filter to deal and up to quarter end
            deal_acct = acct_norm[
                (acct_norm["InvestmentID"].isin(deal_investment_ids)) &
                (acct_norm["EffectiveDate"].dt.date <= quarter_end)
            ].copy()

            if not deal_acct.empty:
                deal_acct["MajorType"] = deal_acct["MajorType"].fillna("").astype(str).str.strip()
                deal_acct["Amt"] = pd.to_numeric(deal_acct["Amt"], errors="coerce").fillna(0.0)

                if "TypeName" not in deal_acct.columns and "Typename" in deal_acct.columns:
                    deal_acct["TypeName"] = deal_acct["Typename"]
                elif "TypeName" not in deal_acct.columns:
                    deal_acct["TypeName"] = ""
                deal_acct["TypeName"] = deal_acct["TypeName"].fillna("").astype(str).str.strip()
                deal_acct["InvestorID"] = deal_acct["InvestorID"].astype(str).str.strip()

                # Sum contributions (funded) and ROC for non-OP investors (PE investors)
                for _, row in deal_acct.iterrows():
                    investor_id = row["InvestorID"]
                    if investor_id.upper().startswith("OP"):
                        continue  # Skip operating partners

                    major_type = row["MajorType"].lower()
                    type_name = row["TypeName"].lower()
                    amt = float(row["Amt"])

                    if "contrib" in major_type:
                        pe['funded_to_date'] += abs(amt)
                    if "distri" in major_type and "return of capital" in type_name:
                        pe['return_of_capital'] += abs(amt)
        except Exception as e:
            pass

    # Calculate derived metrics
    pe['current_pe_balance'] = pe['funded_to_date'] - pe['return_of_capital']
    pe['remaining_to_fund'] = max(0, pe['committed_pe'] - pe['funded_to_date'])

    return pe


# ============================================================
# CHART DATA
# ============================================================

def get_noi_chart_data(
    vcode: str,
    quarter_str: str,
    isbs_df: pd.DataFrame,
    occupancy_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Get NOI and Occupancy data for trailing quarters chart

    Args:
        vcode: Deal vcode
        quarter_str: Current quarter string
        isbs_df: ISBS DataFrame
        occupancy_df: Occupancy DataFrame

    Returns:
        DataFrame with columns: Quarter, Occupancy, NOI_Actual, NOI_UW
    """
    quarters = get_trailing_quarters(quarter_str, 10)
    vcode_str = str(vcode).strip().lower()

    chart_data = []

    for qtr in quarters:
        row = {'Quarter': qtr, 'Occupancy': None, 'NOI_Actual': None, 'NOI_UW': None}

        # Get performance data for this quarter
        perf = get_property_performance(vcode, qtr, isbs_df, None, occupancy_df)

        row['NOI_Actual'] = perf['noi']['ytd_actual'] if perf['noi']['ytd_actual'] != 0 else None
        row['NOI_UW'] = perf['noi']['uw_ye'] if perf['noi']['uw_ye'] != 0 else None
        row['Occupancy'] = perf['economic_occ']['ytd_actual']

        chart_data.append(row)

    return pd.DataFrame(chart_data)


# ============================================================
# COMMENTS CRUD
# ============================================================

def get_one_pager_comments(vcode: str, reporting_period: str) -> Dict[str, str]:
    """
    Get comments for a deal and reporting period

    Args:
        vcode: Deal vcode
        reporting_period: Quarter string

    Returns:
        Dictionary with comment fields
    """
    comments = {
        'econ_comments': '',
        'business_plan_comments': '',
        'accrued_pref_comment': '',
    }

    try:
        conn = get_db_connection()
        result = pd.read_sql(
            """SELECT econ_comments, business_plan_comments, accrued_pref_comment
               FROM one_pager_comments
               WHERE vcode = ? AND reporting_period = ?""",
            conn,
            params=(str(vcode), str(reporting_period))
        )
        conn.close()

        if not result.empty:
            row = result.iloc[0]
            comments['econ_comments'] = str(row['econ_comments']) if pd.notna(row['econ_comments']) else ''
            comments['business_plan_comments'] = str(row['business_plan_comments']) if pd.notna(row['business_plan_comments']) else ''
            comments['accrued_pref_comment'] = str(row['accrued_pref_comment']) if pd.notna(row['accrued_pref_comment']) else ''
    except Exception as e:
        pass  # Table may not exist yet

    return comments


def save_one_pager_comments(
    vcode: str,
    reporting_period: str,
    econ_comments: str = None,
    business_plan_comments: str = None,
    accrued_pref_comment: str = None
) -> bool:
    """
    Save comments for a deal and reporting period

    Args:
        vcode: Deal vcode
        reporting_period: Quarter string
        econ_comments: Economic comments text
        business_plan_comments: Business plan comments text
        accrued_pref_comment: Accrued pref comment text

    Returns:
        True if successful
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Check if record exists
        cursor.execute(
            "SELECT 1 FROM one_pager_comments WHERE vcode = ? AND reporting_period = ?",
            (str(vcode), str(reporting_period))
        )
        exists = cursor.fetchone() is not None

        if exists:
            # Update
            updates = []
            params = []
            if econ_comments is not None:
                updates.append("econ_comments = ?")
                params.append(econ_comments)
            if business_plan_comments is not None:
                updates.append("business_plan_comments = ?")
                params.append(business_plan_comments)
            if accrued_pref_comment is not None:
                updates.append("accrued_pref_comment = ?")
                params.append(accrued_pref_comment)

            if updates:
                updates.append("last_updated = CURRENT_TIMESTAMP")
                params.extend([str(vcode), str(reporting_period)])
                cursor.execute(
                    f"UPDATE one_pager_comments SET {', '.join(updates)} WHERE vcode = ? AND reporting_period = ?",
                    params
                )
        else:
            # Insert
            cursor.execute(
                """INSERT INTO one_pager_comments
                   (vcode, reporting_period, econ_comments, business_plan_comments, accrued_pref_comment, last_updated)
                   VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
                (str(vcode), str(reporting_period),
                 econ_comments or '', business_plan_comments or '', accrued_pref_comment or '')
            )

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        return False
