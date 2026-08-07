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
from utils import normalize_columns


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

    # ISBS is pre-normalized at load time — just filter
    df = isbs_df
    if 'vSource' in df.columns:
        df = df[df['vSource'] == 'Interim IS']

    if 'dtEntry_parsed' not in df.columns:
        return []

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
    normalize_columns(df)

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

    # Location is displayed as "City, State". City alone is ambiguous across the
    # portfolio (e.g. multiple Portland / Milford deals in different states).
    state = ''
    for col in ['State', 'state']:
        if col in row.index and pd.notna(row[col]):
            state = str(row[col]).strip()
            break
    city = info['location']
    if state and state.lower() not in city.lower():
        info['location'] = f"{city}, {state}" if city else state

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
    inv_map: pd.DataFrame,
    isbs_raw: pd.DataFrame = None,
    quarter_str: str = None,
    relationships: pd.DataFrame = None,
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
        'loan_terms_str': 'N/A',
        'second_loan_maturity': None,
        'second_loan_rate': 0.0,
        'second_loan_type': '',
        'second_loan_terms_str': 'N/A',
        'rate_cap': None,
        'debt': 0.0,
        'debt_pct': 0.0,
        'pref_equity': 0.0,
        'pref_equity_pct': 0.0,
        'partner_equity': 0.0,
        'partner_equity_pct': 0.0,
        'total_cap': 0.0,
        'current_valuation': 0.0,
        'valuation_year': '',
        'pe_exposure_on_cap': 0.0,
        'pe_exposure_on_value': 0.0,
        'pe_yield_on_exposure': 0.0,
        'committed_pe': 0.0,
    }

    vcode_str = str(vcode).strip()

    # Get debt from ISBS balance sheet (current outstanding for selected quarter)
    isbs_debt = None
    if isbs_raw is not None and not isbs_raw.empty:
        from compute import get_isbs_debt_balance
        as_of = None
        if quarter_str:
            _, q_end = quarter_to_date_range(quarter_str)
            as_of = q_end
        isbs_debt = get_isbs_debt_balance(isbs_raw, vcode, as_of_date=as_of, mri_loans=mri_loans)

    if isbs_debt is not None:
        cap['debt'] = isbs_debt
    elif mri_loans is not None and not mri_loans.empty:
        # Fallback: origination amounts from MRI_Loans
        loans = mri_loans.copy()
        normalize_columns(loans)
        if 'vCode' not in loans.columns and 'vcode' in loans.columns:
            loans = loans.rename(columns={'vcode': 'vCode'})
        if 'vCode' in loans.columns:
            loans['vCode'] = loans['vCode'].astype(str).str.strip()
            deal_loans = loans[loans['vCode'] == vcode_str]
            if not deal_loans.empty and 'mOrigLoanAmt' in deal_loans.columns:
                cap['debt'] = pd.to_numeric(deal_loans['mOrigLoanAmt'], errors='coerce').fillna(0).sum()

    # Loan terms — always from MRI_Loans (independent of debt source)
    if mri_loans is not None and not mri_loans.empty:
        loans = mri_loans.copy()
        normalize_columns(loans)
        if 'vCode' not in loans.columns and 'vcode' in loans.columns:
            loans = loans.rename(columns={'vcode': 'vCode'})
        if 'vCode' in loans.columns:
            loans['vCode'] = loans['vCode'].astype(str).str.strip()
            deal_loans = loans[loans['vCode'] == vcode_str]

            if not deal_loans.empty:
                def _parse_loan(row):
                    """Extract maturity, rate, type, index, spread from a loan row."""
                    maturity = None
                    rate = 0.0
                    ltype = ''
                    vindex = ''
                    vspread = ''
                    for mat_col in ['dtMaturity', 'dtEvent']:
                        if mat_col in row.index and pd.notna(row[mat_col]):
                            try:
                                maturity = pd.to_datetime(row[mat_col]).date()
                                break
                            except Exception:
                                pass
                    if 'nRate' in row.index:
                        r = pd.to_numeric(row['nRate'], errors='coerce')
                        if pd.notna(r):
                            rate = r if r < 1 else r / 100
                    if 'vIntType' in row.index:
                        ltype = str(row['vIntType']).strip()
                    if 'vIndex' in row.index and pd.notna(row['vIndex']):
                        vindex = str(row['vIndex']).strip()
                    if 'vSpread' in row.index:
                        s = pd.to_numeric(row['vSpread'], errors='coerce')
                        if pd.notna(s):
                            vspread = f"{s:.2%}" if s < 1 else f"{s:.2f}%"
                    return maturity, rate, ltype, vindex, vspread

                def _get_extension_options(row):
                    """Extract extension options from loan row (e.g. '2x12', '1X24')."""
                    for col in ['ExtensionOptions', 'extensionoptions', 'Extension Options']:
                        if col in row.index and pd.notna(row[col]):
                            val = str(row[col]).strip()
                            if val and val.upper() not in ('NA', 'NAN', 'NONE', ''):
                                return val
                    return ''

                def _format_loan_str(maturity, rate, ltype, vindex='', vspread='', extension=''):
                    """Format: '3.68% | Fixed | 8/1/2029 (+2x12)' or 'SOFR + 3.70% | 10/11/2026'."""
                    parts = []
                    if ltype.lower() == 'fixed':
                        if rate > 0:
                            parts.append(f"{rate:.2%}")
                        parts.append('Fixed')
                    elif ltype.lower() == 'variable' and vindex:
                        parts.append(f"{vindex} + {vspread}" if vspread else vindex)
                    else:
                        if rate > 0:
                            parts.append(f"{rate:.2%}")
                        if ltype:
                            parts.append(ltype)
                    if maturity:
                        mat_str = f"{maturity.month}/{maturity.day}/{maturity.year}"
                        if extension:
                            mat_str += f" (+{extension})"
                        parts.append(mat_str)
                    elif extension:
                        parts.append(f"(+{extension})")
                    return ' | '.join(parts) if parts else 'N/A'

                # Primary loan (largest by amount)
                deal_loans_sorted = deal_loans.copy()
                deal_loans_sorted['_amt'] = pd.to_numeric(deal_loans_sorted['mOrigLoanAmt'], errors='coerce').fillna(0)
                deal_loans_sorted = deal_loans_sorted.sort_values('_amt', ascending=False)

                loan_row = deal_loans_sorted.iloc[0]
                cap['loan_maturity'], cap['loan_rate'], cap['loan_type'], _idx, _sprd = _parse_loan(loan_row)
                _ext = _get_extension_options(loan_row)
                cap['loan_terms_str'] = _format_loan_str(cap['loan_maturity'], cap['loan_rate'], cap['loan_type'], _idx, _sprd, _ext)

                # Rate cap — from vHedged/vHedgedStrat on largest loan
                if 'vHedged' in loan_row.index and str(loan_row.get('vHedged', '')).strip().lower() == 'yes':
                    strat = str(loan_row.get('vHedgedStrat', '')).strip() if pd.notna(loan_row.get('vHedgedStrat')) else ''
                    cap['rate_cap'] = strat if strat else 'Yes'

                # Second loan
                if len(deal_loans_sorted) > 1:
                    loan2 = deal_loans_sorted.iloc[1]
                    cap['second_loan_maturity'], cap['second_loan_rate'], cap['second_loan_type'], _idx2, _sprd2 = _parse_loan(loan2)
                    _ext2 = _get_extension_options(loan2)
                    cap['second_loan_terms_str'] = _format_loan_str(cap['second_loan_maturity'], cap['second_loan_rate'], cap['second_loan_type'], _idx2, _sprd2, _ext2)

    # Get valuation from MRI_VAL
    if mri_val is not None and not mri_val.empty:
        val = mri_val.copy()
        normalize_columns(val)
        if 'vcode' not in val.columns and 'vCode' in val.columns:
            val = val.rename(columns={'vCode': 'vcode'})
        if 'vcode' in val.columns:
            val['vcode'] = val['vcode'].astype(str).str.strip()
            deal_val = val[val['vcode'] == vcode_str]

            if not deal_val.empty:
                # Sort by valuation date to get most recent
                for dt_col in ['dtValuation', 'dtVal', 'dtReported', 'dtEntry']:
                    if dt_col in deal_val.columns:
                        deal_val = deal_val.copy()
                        deal_val['_dt_parsed'] = pd.to_datetime(deal_val[dt_col], format='mixed', dayfirst=False, errors='coerce')
                        deal_val = deal_val.dropna(subset=['_dt_parsed']).sort_values('_dt_parsed', ascending=False)
                        break
                val_row = deal_val.iloc[0] if not deal_val.empty else None

                if val_row is not None and 'mIncomeCapConcludedValue' in deal_val.columns:
                    v = pd.to_numeric(val_row['mIncomeCapConcludedValue'], errors='coerce')
                    cap['current_valuation'] = float(v) if pd.notna(v) else 0.0

                # Valuation year from date column
                if val_row is not None:
                    for dt_col in ['dtValuation', 'dtVal', 'dtReported', 'dtEntry']:
                        if dt_col in val_row.index and pd.notna(val_row[dt_col]):
                            try:
                                cap['valuation_year'] = str(pd.to_datetime(val_row[dt_col]).year)
                            except:
                                pass
                            break

                # Purchase price from valuations table
                if val_row is not None:
                    for pp_col in ['mPurchasePrice', 'Acquisition_Price']:
                        if pp_col in deal_val.columns:
                            pp = pd.to_numeric(val_row[pp_col], errors='coerce')
                            if pd.notna(pp) and pp > 0:
                                cap['purchase_price'] = float(pp)
                                break

    # Fallback: purchase price from investment map (deals table)
    if cap['purchase_price'] == 0 and inv_map is not None and not inv_map.empty:
        im = inv_map.copy()
        normalize_columns(im)
        if 'vcode' not in im.columns and 'vCode' in im.columns:
            im = im.rename(columns={'vCode': 'vcode'})
        if 'vcode' in im.columns:
            im['vcode'] = im['vcode'].astype(str).str.strip()
            deal_im = im[im['vcode'] == vcode_str]
            if not deal_im.empty:
                for pp_col in ['Acquisition_Price', 'Purchase_Price', 'mPurchasePrice']:
                    if pp_col in deal_im.columns:
                        raw = deal_im.iloc[0][pp_col]
                        if pd.notna(raw):
                            clean = str(raw).replace(',', '').replace('$', '').strip()
                            pp = pd.to_numeric(clean, errors='coerce')
                            if pd.notna(pp) and pp > 0:
                                cap['purchase_price'] = float(pp)
                                break

    # Get PE coupon and participation from waterfalls
    if waterfalls is not None and not waterfalls.empty:
        wf = waterfalls.copy()
        normalize_columns(wf)
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
        normalize_columns(comm)
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
            normalize_columns(acct_norm)
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
        'Principal': ['7060'],
    },
    # Balance-sheet debt accounts for principal estimation from balance changes
    'DEBT_BS_ACCTS': ['2145', '2150', '2152', '2154', '2156'],
    # Underwriting total debt service account (Projected IS)
    'UW_DEBT_SERVICE': ['7010'],
}


def get_property_performance(
    vcode: str,
    quarter_str: str,
    isbs_df: pd.DataFrame,
    mri_val: pd.DataFrame,
    occupancy_df: pd.DataFrame = None,
    budget_econ_occ_df: pd.DataFrame = None,
    at_close_noi_df: pd.DataFrame = None,
    deal_terms_df: pd.DataFrame = None,
) -> Dict[str, Any]:
    """
    Get property performance metrics for a quarter

    Args:
        vcode: Deal vcode
        quarter_str: Quarter string (e.g., '2025-Q4')
        isbs_df: ISBS DataFrame with income statement data
        mri_val: Valuations DataFrame for At Close data
        occupancy_df: Occupancy DataFrame
        budget_econ_occ_df: Budget economic occupancy from ProjOccupancy
        at_close_noi_df: Pre-computed at-close NOI from Prop_Info_AtClose query
        deal_terms_df: Deal terms with econ_occ_at_close from txfinancial_IC

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

    # ISBS is pre-normalized at load time — just filter by deal
    if 'vcode' in isbs_df.columns:
        isbs = isbs_df[isbs_df['vcode'] == vcode_str]
    else:
        isbs = isbs_df

    if isbs.empty:
        return perf

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
    bs_data = isbs[isbs['vSource'] == 'Interim BS']
    budget_data = isbs[isbs['vSource'] == 'Budget IS']
    uw_data = isbs[isbs['vSource'] == 'Projected IS']

    # Helper: estimate YTD principal from Interim BS balance changes
    # Principal payments reduce loan balances, so principal = (prior_bal - current_bal)
    # extrapolated by months elapsed in the quarter period
    def _estimate_principal_from_bs(bs_df, qtr_end_ts, months_elapsed):
        """Estimate YTD principal from BS debt account balance changes."""
        if bs_df.empty:
            return 0
        bs_accts = IS_ACCOUNTS['DEBT_BS_ACCTS']
        debt_bs = bs_df[bs_df['vAccount'].isin(bs_accts)]
        if debt_bs.empty:
            return 0
        bs_periods = sorted(debt_bs['dtEntry_parsed'].dropna().unique())
        if not bs_periods:
            return 0
        # Find latest BS period on or before quarter end
        current_date = None
        for p in reversed(bs_periods):
            if pd.Timestamp(p) <= qtr_end_ts:
                current_date = pd.Timestamp(p)
                break
        if current_date is None:
            return 0
        current_bal = abs(debt_bs[debt_bs['dtEntry_parsed'] == current_date]['mAmount'].sum())
        # Find prior month's BS balance
        prior_date = None
        for p in reversed(bs_periods):
            if pd.Timestamp(p) < current_date:
                prior_date = pd.Timestamp(p)
                break
        if prior_date is None:
            return 0
        prior_bal = abs(debt_bs[debt_bs['dtEntry_parsed'] == prior_date]['mAmount'].sum())
        # Monthly principal = balance decrease; extrapolate to YTD
        monthly_principal = max(0, prior_bal - current_bal)
        return monthly_principal * months_elapsed

    # Determine months elapsed for the quarter
    qtr_num = int(quarter_str.split('-Q')[1]) if '-Q' in quarter_str else 1
    months_elapsed = qtr_num * 3  # Q1=3, Q2=6, Q3=9, Q4=12

    # Find the as-of date for YTD actual (last date in quarter or closest available)
    ytd_date = None
    if not actual_data.empty:
        actual_periods = sorted(actual_data['dtEntry_parsed'].dropna().unique())
        # Find period closest to or before quarter end
        for p in reversed(actual_periods):
            if pd.Timestamp(p).date() <= quarter_end:
                ytd_date = pd.Timestamp(p)
                break

        if ytd_date:
            rev, exp, noi, _ = calc_amounts(actual_data, as_of_date=ytd_date)
            perf['revenue']['ytd_actual'] = rev
            perf['expenses']['ytd_actual'] = exp
            perf['noi']['ytd_actual'] = noi

            # YTD Actual DSCR: Interest (5190) + Principal (7060 if available, else BS balance change)
            ytd_is = actual_data[actual_data['dtEntry_parsed'] == ytd_date]
            ytd_interest = abs(ytd_is[ytd_is['vAccount'] == '5190']['mAmount'].sum())
            ytd_principal_is = abs(ytd_is[ytd_is['vAccount'] == '7060']['mAmount'].sum())
            if ytd_principal_is > 0:
                ytd_principal = ytd_principal_is
            else:
                ytd_principal = _estimate_principal_from_bs(bs_data, pd.Timestamp(quarter_end), months_elapsed)
            ytd_actual_ds = ytd_interest + ytd_principal
            if ytd_actual_ds > 0:
                perf['dscr']['ytd_actual'] = noi / ytd_actual_ds

    # Get YTD budget — Budget IS is periodic, sum 5190+7060 over date range
    if not budget_data.empty:
        jan1 = pd.Timestamp(f"{quarter_str.split('-')[0]}-01-01") - pd.DateOffset(days=1)
        qtr_end = pd.Timestamp(quarter_end)
        rev, exp, noi, ds = calc_amounts(budget_data, sum_range=(jan1, qtr_end))
        perf['revenue']['ytd_budget'] = rev
        perf['expenses']['ytd_budget'] = exp
        perf['noi']['ytd_budget'] = noi
        if abs(ds) > 0:
            perf['dscr']['ytd_budget'] = noi / abs(ds)

    # Projected YE = YTD Actual + remainder-of-year Budget
    year = int(quarter_str.split('-')[0])
    if not actual_data.empty and not budget_data.empty:
        ytd_rev = perf['revenue']['ytd_actual']
        ytd_exp = perf['expenses']['ytd_actual']
        ytd_noi = perf['noi']['ytd_actual']

        # YTD actual debt service (interest + principal, same logic as ytd_actual DSCR above)
        ytd_ds = 0
        if ytd_date is not None:
            ytd_is = actual_data[actual_data['dtEntry_parsed'] == ytd_date]
            ytd_interest = abs(ytd_is[ytd_is['vAccount'] == '5190']['mAmount'].sum())
            ytd_principal_is = abs(ytd_is[ytd_is['vAccount'] == '7060']['mAmount'].sum())
            if ytd_principal_is > 0:
                ytd_ds = ytd_interest + ytd_principal_is
            else:
                ytd_ds = ytd_interest + _estimate_principal_from_bs(bs_data, pd.Timestamp(quarter_end), months_elapsed)

        # Remainder Budget: months after quarter end through Dec 31 (5190+7060)
        remainder_start = pd.Timestamp(quarter_end)
        dec31 = pd.Timestamp(f"{year}-12-31")
        rem_rev, rem_exp, rem_noi, rem_ds = calc_amounts(budget_data, sum_range=(remainder_start, dec31))

        perf['revenue']['actual_ye'] = ytd_rev + rem_rev
        perf['expenses']['actual_ye'] = ytd_exp + rem_exp
        perf['noi']['actual_ye'] = ytd_noi + rem_noi
        total_ds = abs(ytd_ds) + abs(rem_ds)
        if total_ds > 0:
            perf['dscr']['actual_ye'] = (ytd_noi + rem_noi) / total_ds
    elif not actual_data.empty:
        # No budget data — use YTD actual only
        perf['revenue']['actual_ye'] = perf['revenue']['ytd_actual']
        perf['expenses']['actual_ye'] = perf['expenses']['ytd_actual']
        perf['noi']['actual_ye'] = perf['noi']['ytd_actual']
        perf['dscr']['actual_ye'] = perf['dscr']['ytd_actual']

    # Get U/W YE (full year projected)
    # Underwriting (Projected IS) is YTD cumulative — use December snapshot
    if not uw_data.empty:
        year = int(quarter_str.split('-')[0])
        uw_periods = sorted(uw_data['dtEntry_parsed'].dropna().unique())
        dec_date = next((pd.Timestamp(p) for p in reversed(uw_periods)
                         if pd.Timestamp(p).year == year and pd.Timestamp(p).month == 12), None)
        if dec_date:
            rev, exp, noi, _ = calc_amounts(uw_data, as_of_date=dec_date)
            perf['revenue']['uw_ye'] = rev
            perf['expenses']['uw_ye'] = exp
            perf['noi']['uw_ye'] = noi

            # Fix 7: U/W DSCR uses account 7010 (total debt service) from Projected IS
            uw_dec = uw_data[uw_data['dtEntry_parsed'] == dec_date]
            uw_ds = abs(uw_dec[uw_dec['vAccount'].isin(IS_ACCOUNTS['UW_DEBT_SERVICE'])]['mAmount'].sum())
            if uw_ds > 0:
                perf['dscr']['uw_ye'] = noi / uw_ds

            # U/W YE Economic Occupancy from Projected IS: 1 - (vacancy / rental income)
            # 4010 = Rental Income (negative/credit), 4030 = Vacancy Loss (positive/debit)
            uw_dec = uw_data[uw_data['dtEntry_parsed'] == dec_date]
            if not uw_dec.empty:
                rental = uw_dec[uw_dec['vAccount'] == '4010']['mAmount'].sum()  # negative
                vacancy = uw_dec[uw_dec['vAccount'] == '4030']['mAmount'].sum()  # positive
                if rental != 0:
                    perf['economic_occ']['uw_ye'] = (1 - vacancy / abs(rental)) * 100

        # At Close: use pre-computed at_close_noi table if available (faster, dynamic dates)
        at_close_filled = False
        if at_close_noi_df is not None and not at_close_noi_df.empty:
            acn = at_close_noi_df.copy()
            normalize_columns(acn)
            if 'vcode' not in acn.columns and 'vCode' in acn.columns:
                acn = acn.rename(columns={'vCode': 'vcode'})
            if 'vcode' in acn.columns:
                acn['vcode'] = acn['vcode'].astype(str).str.strip().str.lower()
                acn_row = acn[acn['vcode'] == vcode_str]
                if not acn_row.empty:
                    r = acn_row.iloc[0]
                    acn_rev = pd.to_numeric(r.get('at_close_revenue'), errors='coerce')
                    acn_exp = pd.to_numeric(r.get('at_close_expenses'), errors='coerce')
                    acn_int = pd.to_numeric(r.get('at_close_interest'), errors='coerce')
                    acn_prin = pd.to_numeric(r.get('at_close_principal'), errors='coerce')
                    if pd.notna(acn_rev):
                        perf['revenue']['at_close'] = -float(acn_rev)
                    if pd.notna(acn_exp):
                        perf['expenses']['at_close'] = float(acn_exp)
                    acn_noi = pd.to_numeric(r.get('at_close_noi'), errors='coerce')
                    if pd.notna(acn_noi):
                        perf['noi']['at_close'] = -float(acn_noi)
                    ds = abs(float(acn_int or 0)) + abs(float(acn_prin or 0))
                    if ds > 0 and pd.notna(acn_noi):
                        perf['dscr']['at_close'] = -float(acn_noi) / ds
                    at_close_filled = True

        # Fallback: earliest December 31 in Projected IS = due diligence audit
        if not at_close_filled:
            dec_dates = [pd.Timestamp(p) for p in uw_periods if pd.Timestamp(p).month == 12]
            if dec_dates:
                at_close_date = min(dec_dates)
                rev, exp, noi, ds = calc_amounts(uw_data, as_of_date=at_close_date)
                perf['revenue']['at_close'] = rev
                perf['expenses']['at_close'] = exp
                perf['noi']['at_close'] = noi
                if ds > 0:
                    perf['dscr']['at_close'] = noi / ds

    # Economic Occupancy at Close from deal_terms (txfinancial_IC)
    # Outside the uw_data block — deal_terms is independent of ISBS data
    if deal_terms_df is not None and not deal_terms_df.empty:
        dtf = deal_terms_df.copy()
        normalize_columns(dtf)
        if 'vcode' not in dtf.columns and 'vCode' in dtf.columns:
            dtf = dtf.rename(columns={'vCode': 'vcode'})
        if 'vcode' in dtf.columns:
            dtf['vcode'] = dtf['vcode'].astype(str).str.strip().str.lower()
            dt_row = dtf[dtf['vcode'] == vcode_str]
            if not dt_row.empty:
                eoc = pd.to_numeric(dt_row.iloc[0].get('econ_occ_at_close'), errors='coerce')
                if pd.notna(eoc) and eoc > 0:
                    # Stored as decimal (0.909 = 90.9%), display as percentage
                    perf['economic_occ']['at_close'] = float(eoc) * 100 if eoc <= 1 else float(eoc)

    # Calculate variances
    for metric in ['revenue', 'expenses', 'noi']:
        perf[metric]['variance'] = perf[metric]['ytd_actual'] - perf[metric]['ytd_budget']

    if perf['dscr']['ytd_actual'] is not None and perf['dscr']['ytd_budget'] is not None:
        perf['dscr']['variance'] = perf['dscr']['ytd_actual'] - perf['dscr']['ytd_budget']

    # Economic Occupancy = avg physical occ (YTD months) - bad debt/concessions %
    if occupancy_df is not None and not occupancy_df.empty:
        occ = occupancy_df.copy()
        normalize_columns(occ)
        if 'vCode' in occ.columns or 'vcode' in occ.columns:
            vcode_col = 'vCode' if 'vCode' in occ.columns else 'vcode'
            occ[vcode_col] = occ[vcode_col].astype(str).str.strip().str.lower()
            occ = occ[occ[vcode_col] == vcode_str]

            if not occ.empty:
                # Parse dtReported to filter YTD months
                occ_col = 'Occ%' if 'Occ%' in occ.columns else 'OccupancyPercent'
                if occ_col in occ.columns and 'dtReported' in occ.columns:
                    occ['_dt'] = pd.to_datetime(occ['dtReported'], format='mixed', dayfirst=False, errors='coerce')
                    if occ['_dt'].isna().sum() > len(occ) * 0.5:
                        try:
                            num = pd.to_numeric(occ['dtReported'], errors='coerce')
                            occ.loc[occ['_dt'].isna(), '_dt'] = pd.to_datetime(
                                num[occ['_dt'].isna()], unit='D', origin='1899-12-30', errors='coerce')
                        except Exception:
                            pass

                    year = int(quarter_str.split('-')[0])
                    # YTD: Jan 1 through quarter end of selected year
                    ytd_occ = occ[(occ['_dt'].dt.year == year) & (occ['_dt'] <= pd.Timestamp(quarter_end))]
                    if not ytd_occ.empty:
                        avg_physical_occ = pd.to_numeric(ytd_occ[occ_col], errors='coerce').mean()

                        # Bad debt/concessions % from ISBS Interim IS YTD
                        bad_debt_pct = 0.0
                        if not actual_data.empty and ytd_date is not None:
                            ytd_is = actual_data[actual_data['dtEntry_parsed'] == ytd_date]
                            if not ytd_is.empty:
                                # 4040 (concessions) + 4043 (bad debt) — positive (debit, contra-revenue)
                                bd_amt = ytd_is[ytd_is['vAccount'].isin(['4040', '4043'])]['mAmount'].sum()
                                # 4010 (rental income) — negative (credit)
                                rental = ytd_is[ytd_is['vAccount'] == '4010']['mAmount'].sum()
                                if rental != 0:
                                    # bd_amt is positive, rental is negative → ratio is positive
                                    bad_debt_pct = bd_amt / abs(rental) * 100  # as percentage points

                        perf['economic_occ']['ytd_actual'] = avg_physical_occ - bad_debt_pct

    # Budget Economic Occupancy from ProjOccupancy (budget_econ_occ table)
    if budget_econ_occ_df is not None and not budget_econ_occ_df.empty:
        beo = budget_econ_occ_df.copy()
        normalize_columns(beo)
        vcode_col = 'VCODE' if 'VCODE' in beo.columns else 'vcode'
        if vcode_col in beo.columns:
            beo[vcode_col] = beo[vcode_col].astype(str).str.strip().str.lower()
            beo = beo[beo[vcode_col] == vcode_str]
            if not beo.empty:
                dt_col = 'DTPERIOD' if 'DTPERIOD' in beo.columns else 'dtperiod'
                beo['_dt'] = pd.to_datetime(beo[dt_col], format='mixed', dayfirst=False, errors='coerce')
                year = int(quarter_str.split('-')[0])
                q = int(quarter_str.split('Q')[1])
                end_month = q * 3
                # Find the last budget month in the quarter (YTD avg through quarter end)
                ytd_months = beo[(beo['_dt'].dt.year == year) & (beo['_dt'].dt.month <= end_month)]
                if not ytd_months.empty:
                    # Use the last row's YTDAvgPctOccupied (running avg through quarter end)
                    ytd_months = ytd_months.sort_values('_dt')
                    ytd_avg_col = 'YTDAvgPctOccupied' if 'YTDAvgPctOccupied' in ytd_months.columns else 'ytdavgpctoccupied'
                    if ytd_avg_col in ytd_months.columns:
                        val = pd.to_numeric(ytd_months.iloc[-1][ytd_avg_col], errors='coerce')
                        if pd.notna(val):
                            # ProjOccupancy stores as decimal (0.887 = 88.7%);
                            # actual Occ% is in percentage points (88.7)
                            budget_occ = val * 100
                            # Subtract bad debt/concessions % from Budget IS (matching Excel)
                            # Accounts: 4040 (Concessions), 4041 (Commercial Concessions), 4043 (Bad Debt)
                            bad_debt_pct_budget = 0
                            if not budget_data.empty:
                                jan1_b = pd.Timestamp(f"{year}-01-01") - pd.DateOffset(days=1)
                                qtr_end_b = pd.Timestamp(quarter_end)
                                bud_range = budget_data[
                                    (budget_data['dtEntry_parsed'] > jan1_b) &
                                    (budget_data['dtEntry_parsed'] <= qtr_end_b)
                                ]
                                if not bud_range.empty:
                                    bad_debt_sum = bud_range[bud_range['vAccount'].isin(['4040', '4041', '4043'])]['mAmount'].sum()
                                    rental_sum = bud_range[bud_range['vAccount'] == '4010']['mAmount'].sum()
                                    if rental_sum != 0:
                                        bad_debt_pct_budget = bad_debt_sum / abs(rental_sum) * 100
                            perf['economic_occ']['ytd_budget'] = budget_occ - bad_debt_pct_budget

    # Compute economic occupancy variance
    if perf['economic_occ']['ytd_actual'] is not None and perf['economic_occ']['ytd_budget'] is not None:
        perf['economic_occ']['variance'] = perf['economic_occ']['ytd_actual'] - perf['economic_occ']['ytd_budget']

    # Projected YE Economic Occupancy = avg of actual months + budget remaining months
    q_num = int(quarter_str.split('Q')[1])
    actual_months = q_num * 3  # e.g. Q1 = 3 actual months
    remaining_months = 12 - actual_months
    if perf['economic_occ']['ytd_actual'] is not None and remaining_months > 0:
        # Get budget occupancy for remaining months from budget_econ_occ
        budget_remaining_occ = None
        if budget_econ_occ_df is not None and not budget_econ_occ_df.empty:
            beo2 = budget_econ_occ_df.copy()
            normalize_columns(beo2)
            vc2 = 'VCODE' if 'VCODE' in beo2.columns else 'vcode'
            if vc2 in beo2.columns:
                beo2[vc2] = beo2[vc2].astype(str).str.strip().str.lower()
                beo2 = beo2[beo2[vc2] == vcode_str]
                if not beo2.empty:
                    dt2 = 'DTPERIOD' if 'DTPERIOD' in beo2.columns else 'dtperiod'
                    beo2['_dt'] = pd.to_datetime(beo2[dt2], format='mixed', dayfirst=False, errors='coerce')
                    # Remaining months: after quarter end through Dec
                    rem_months = beo2[(beo2['_dt'].dt.year == year) & (beo2['_dt'].dt.month > actual_months)]
                    if not rem_months.empty:
                        occ_col2 = 'PctOccupied' if 'PctOccupied' in rem_months.columns else 'pctoccupied'
                        if occ_col2 in rem_months.columns:
                            rem_vals = pd.to_numeric(rem_months[occ_col2], errors='coerce').dropna()
                            if len(rem_vals) > 0:
                                # ProjOccupancy stores as decimal; convert to percentage
                                budget_remaining_occ = rem_vals.mean() * 100

        if budget_remaining_occ is not None:
            # Weighted average: actual months at actual occ + remaining months at budget occ
            perf['economic_occ']['actual_ye'] = (
                perf['economic_occ']['ytd_actual'] * actual_months
                + budget_remaining_occ * remaining_months
            ) / 12
        else:
            perf['economic_occ']['actual_ye'] = perf['economic_occ']['ytd_actual']
    elif perf['economic_occ']['ytd_actual'] is not None:
        # Q4 — full year actual, no remainder needed
        perf['economic_occ']['actual_ye'] = perf['economic_occ']['ytd_actual']

    return perf


# ============================================================
# U/W PE DISTRIBUTIONS (for U/W ROE)
# ============================================================

UW_PE_DIST_ACCT = '7071'  # Underwritten distribution to preferred equity
UW_PE_ROC_ACCT = '7073'   # Underwritten return of capital to preferred equity


def _get_uw_pe_periodic(
    isbs_raw: pd.DataFrame,
    vcode: str,
    inception: date,
    end_date: date,
    account: str = UW_PE_DIST_ACCT,
) -> List[Tuple[date, float]]:
    """Extract periodic underwritten PE amounts from Projected IS.

    Accounts in ISBS vSource='Projected IS' store YTD cumulative
    amounts (same convention as Actuals).  This function converts to
    periodic amounts and returns them as (date, amount) tuples.

    Used for both distributions (7071) and return of capital (7073).

    The sign convention in MRI: these amounts are stored as positive
    (debit).  We take abs() to be safe.

    Returns list of (date, amount) for each period with a non-zero
    amount, covering inception through end_date.
    """
    if isbs_raw is None or isbs_raw.empty:
        return []

    # Filter to deal + Projected IS + target account
    df = isbs_raw.copy()
    if 'vcode' in df.columns:
        df = df[df['vcode'] == str(vcode).strip().lower()]
    if df.empty:
        return []

    if 'vSource' in df.columns:
        df = df[df['vSource'] == 'Projected IS']
    if df.empty or 'vAccount' not in df.columns:
        return []

    df = df[df['vAccount'] == account]
    if df.empty:
        return []

    if 'dtEntry_parsed' not in df.columns:
        return []

    df = df.dropna(subset=['dtEntry_parsed'])
    df = df.sort_values('dtEntry_parsed')

    # Get all available periods
    periods = sorted(df['dtEntry_parsed'].unique())
    if not periods:
        return []

    # Convert YTD cumulative → periodic amounts
    # Within a year: periodic = current_ytd - prior_ytd (same year)
    # First period of a year with no prior: pro-rate cumulative by month
    # (e.g. $70K cumulative at July → $10K/month × 1 month = $10K periodic)
    distributions = []
    prev_by_year = {}  # {year: last_cumulative_amount}

    for period in periods:
        period_ts = pd.Timestamp(period)
        period_date = period_ts.date()

        # Only include periods within our time range
        if period_date < inception or period_date > end_date:
            # Still track cumulative for YTD math
            period_data = df[df['dtEntry_parsed'] == period]
            cumulative = float(period_data['mAmount'].sum())
            prev_by_year[period_ts.year] = cumulative
            continue

        period_data = df[df['dtEntry_parsed'] == period]
        cumulative = float(period_data['mAmount'].sum())

        # Convert cumulative to periodic
        year = period_ts.year
        if year in prev_by_year:
            periodic = cumulative - prev_by_year[year]
        elif period_ts.month == 1:
            # January: cumulative IS the single-month amount
            periodic = cumulative
        else:
            # Mid-year start with no prior data: pro-rate cumulative
            # evenly across the months it covers
            periodic = cumulative / period_ts.month

        prev_by_year[year] = cumulative

        amount = abs(periodic)
        if amount > 0.01:
            distributions.append((period_date, amount))

    return distributions


def _get_uw_pe_distributions(
    isbs_raw: pd.DataFrame, vcode: str, inception: date, end_date: date,
) -> List[Tuple[date, float]]:
    """Backward-compatible wrapper — extracts UW PE distributions (7071)."""
    return _get_uw_pe_periodic(isbs_raw, vcode, inception, end_date, UW_PE_DIST_ACCT)


# ============================================================
# PREFERRED EQUITY PERFORMANCE
# ============================================================

def get_pe_performance(
    vcode: str,
    quarter_str: str,
    acct: pd.DataFrame,
    commitments: pd.DataFrame,
    waterfalls: pd.DataFrame,
    inv_map: pd.DataFrame,
    isbs_raw: pd.DataFrame = None,
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
        isbs_raw: ISBS DataFrame (for U/W ROE from Projected IS account 7071)

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
        normalize_columns(wf)
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
        normalize_columns(comm)
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
            normalize_columns(acct_norm)
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

                # Build cashflow lists for ROE calculation
                # capital_events: all cashflows (contributions negative, distributions positive)
                # cf_distributions: only CF (operating) distributions
                capital_events = []
                cf_distributions = []

                # Sum contributions (funded) and ROC for non-OP investors (PE investors)
                for _, row in deal_acct.iterrows():
                    investor_id = row["InvestorID"]
                    if investor_id.upper().startswith("OP"):
                        continue  # Skip operating partners

                    major_type = row["MajorType"].lower()
                    type_name = row["TypeName"].lower()
                    amt = float(row["Amt"])
                    evt_date = row["EffectiveDate"].date() if pd.notna(row["EffectiveDate"]) else None
                    if evt_date is None:
                        continue

                    if "contrib" in major_type:
                        pe['funded_to_date'] += abs(amt)
                        capital_events.append((evt_date, -abs(amt)))
                    elif "distri" in major_type:
                        if "return of capital" in type_name or "realized gain" in type_name:
                            # Capital return (or correction if amt < 0)
                            capital_events.append((evt_date, amt))
                            pe['return_of_capital'] += amt
                        elif "acquisition fee" not in type_name:
                            # CF (operating) distribution — preserve sign so
                            # negative corrections reduce ROE instead of inflating it.
                            # Only positive CF dists go to capital_events (negative
                            # corrections must not inflate weighted avg capital).
                            if amt >= 0:
                                capital_events.append((evt_date, amt))
                            cf_distributions.append((evt_date, amt))
                        else:
                            # Acquisition fee — still affects capital balance timeline
                            if amt >= 0:
                                capital_events.append((evt_date, amt))

                # Compute ROE to Date from actual accounting through quarter end
                if capital_events:
                    from metrics import calculate_roe
                    inception = min(d for d, _ in capital_events)
                    pe['roe_to_date'] = calculate_roe(
                        capital_events, cf_distributions, inception, quarter_end
                    )

                    # Compute U/W ROE to Date from Projected IS
                    # 7071 = underwritten PE distributions (ROE numerator)
                    # 7073 = underwritten return of capital (reduces denominator,
                    #         NOT counted as distributions)
                    # Same actual capital structure (contributions/returns), but
                    # substitute underwritten distributions for CF distributions.
                    # Build capital-only events (no CF dists) so calculate_roe
                    # doesn't try to match actual CF dates against UW dates.
                    if isbs_raw is not None and not isbs_raw.empty:
                        uw_dists = _get_uw_pe_distributions(
                            isbs_raw, vcode, inception, quarter_end
                        )
                        if uw_dists:
                            # capital_only: contributions (neg) + capital returns (pos)
                            # Exclude CF distributions from capital_events
                            cf_dates_amounts: dict = {}
                            for d, a in cf_distributions:
                                cf_dates_amounts[d] = cf_dates_amounts.get(d, 0.0) + a
                            capital_only = []
                            cf_remaining = dict(cf_dates_amounts)
                            for d, amt in capital_events:
                                if amt < 0:
                                    capital_only.append((d, amt))  # contribution
                                else:
                                    # Subtract CF portion to isolate capital returns
                                    cf_at = cf_remaining.get(d, 0.0)
                                    if cf_at > 0:
                                        consumed = min(cf_at, amt)
                                        cf_remaining[d] -= consumed
                                        cap_ret = amt - consumed
                                    else:
                                        cap_ret = amt
                                    if cap_ret > 0.005:
                                        capital_only.append((d, cap_ret))

                            # Add underwritten return of capital (7073)
                            # as positive capital events — reduces weighted
                            # avg capital but does NOT count as distributions
                            uw_roc = _get_uw_pe_periodic(
                                isbs_raw, vcode, inception, quarter_end,
                                UW_PE_ROC_ACCT
                            )
                            for d, amt in uw_roc:
                                capital_only.append((d, amt))

                            pe['uw_roe_to_date'] = calculate_roe(
                                capital_only, uw_dists, inception, quarter_end
                            )
        except Exception:
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
        'underlying_investors': '',
        'pe_cap_comment': '',
    }

    try:
        result = execute_query(
            """SELECT econ_comments, business_plan_comments, accrued_pref_comment, underlying_investors, pe_cap_comment
               FROM one_pager_comments
               WHERE vcode = ? AND reporting_period = ?""",
            (str(vcode), str(reporting_period))
        )

        def _extract_comments(row):
            """Extract comment fields from a query result row."""
            c = {}
            c['econ_comments'] = str(row['econ_comments']) if pd.notna(row['econ_comments']) else ''
            c['business_plan_comments'] = str(row['business_plan_comments']) if pd.notna(row['business_plan_comments']) else ''
            c['accrued_pref_comment'] = str(row['accrued_pref_comment']) if pd.notna(row['accrued_pref_comment']) else ''
            if 'underlying_investors' in row.index:
                c['underlying_investors'] = str(row['underlying_investors']) if pd.notna(row['underlying_investors']) else ''
            if 'pe_cap_comment' in row.index:
                c['pe_cap_comment'] = str(row['pe_cap_comment']) if pd.notna(row['pe_cap_comment']) else ''
            return c

        if not result.empty:
            extracted = _extract_comments(result.iloc[0])
            comments.update(extracted)

        # If the exact quarter has no econ/business_plan comments, fall back to
        # the most recent quarter that does — comments are entered per reporting
        # cycle and the viewed quarter may not have them yet.
        if not comments['econ_comments'].strip() and not comments['business_plan_comments'].strip():
            fallback_all = execute_query(
                """SELECT econ_comments, business_plan_comments, accrued_pref_comment, underlying_investors, pe_cap_comment
                   FROM one_pager_comments
                   WHERE vcode = ?
                     AND (econ_comments IS NOT NULL AND econ_comments != ''
                          OR business_plan_comments IS NOT NULL AND business_plan_comments != '')
                   ORDER BY reporting_period DESC LIMIT 1""",
                (str(vcode),)
            )
            if not fallback_all.empty:
                fb = _extract_comments(fallback_all.iloc[0])
                # Only fill in fields that are still empty
                for key in ('econ_comments', 'business_plan_comments', 'accrued_pref_comment'):
                    if not comments[key].strip() and fb.get(key, '').strip():
                        comments[key] = fb[key]

        # pe_cap_comment doesn't change quarter to quarter — fall back to most recent non-empty value for this vcode
        if not comments['pe_cap_comment']:
            fallback = execute_query(
                """SELECT pe_cap_comment FROM one_pager_comments
                   WHERE vcode = ? AND pe_cap_comment IS NOT NULL AND pe_cap_comment != ''
                   ORDER BY reporting_period DESC LIMIT 1""",
                (str(vcode),)
            )
            if not fallback.empty:
                val = fallback.iloc[0]['pe_cap_comment']
                comments['pe_cap_comment'] = str(val) if pd.notna(val) else ''
    except Exception as e:
        pass  # Table may not exist yet

    return comments


def save_one_pager_comments(
    vcode: str,
    reporting_period: str,
    econ_comments: str = None,
    business_plan_comments: str = None,
    accrued_pref_comment: str = None,
    pe_cap_comment: str = None
) -> bool:
    """
    Save comments for a deal and reporting period

    Args:
        vcode: Deal vcode
        reporting_period: Quarter string
        econ_comments: Economic comments text
        business_plan_comments: Business plan comments text
        accrued_pref_comment: Accrued pref comment text
        pe_cap_comment: Pref equity capitalization comment text

    Returns:
        True if successful
    """
    try:
        from database import _sa_engine
        use_pg = _sa_engine is not None

        # Check if record exists
        check = execute_query(
            "SELECT 1 FROM one_pager_comments WHERE vcode = ? AND reporting_period = ?",
            (str(vcode), str(reporting_period))
        )
        exists = not check.empty

        conn = get_db_connection()

        if use_pg:
            from sqlalchemy import text as sa_text
            if exists:
                sets = []
                p = {}
                if econ_comments is not None:
                    sets.append("econ_comments = :ec")
                    p['ec'] = econ_comments
                if business_plan_comments is not None:
                    sets.append("business_plan_comments = :bp")
                    p['bp'] = business_plan_comments
                if accrued_pref_comment is not None:
                    sets.append("accrued_pref_comment = :ap")
                    p['ap'] = accrued_pref_comment
                if pe_cap_comment is not None:
                    sets.append("pe_cap_comment = :pe")
                    p['pe'] = pe_cap_comment
                if sets:
                    sets.append("last_updated = CURRENT_TIMESTAMP")
                    p['v'] = str(vcode)
                    p['r'] = str(reporting_period)
                    conn.execute(sa_text(
                        f"UPDATE one_pager_comments SET {', '.join(sets)} WHERE vcode = :v AND reporting_period = :r"
                    ), p)
            else:
                conn.execute(sa_text(
                    """INSERT INTO one_pager_comments
                       (vcode, reporting_period, econ_comments, business_plan_comments, accrued_pref_comment, pe_cap_comment, last_updated)
                       VALUES (:v, :r, :ec, :bp, :ap, :pe, CURRENT_TIMESTAMP)"""
                ), {'v': str(vcode), 'r': str(reporting_period),
                    'ec': econ_comments or '', 'bp': business_plan_comments or '',
                    'ap': accrued_pref_comment or '', 'pe': pe_cap_comment or ''})
            conn.commit()
            conn.close()
        else:
            cursor = conn.cursor()
            if exists:
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
                if pe_cap_comment is not None:
                    updates.append("pe_cap_comment = ?")
                    params.append(pe_cap_comment)
                if updates:
                    updates.append("last_updated = CURRENT_TIMESTAMP")
                    params.extend([str(vcode), str(reporting_period)])
                    cursor.execute(
                        f"UPDATE one_pager_comments SET {', '.join(updates)} WHERE vcode = ? AND reporting_period = ?",
                        params
                    )
            else:
                cursor.execute(
                    """INSERT INTO one_pager_comments
                       (vcode, reporting_period, econ_comments, business_plan_comments, accrued_pref_comment, pe_cap_comment, last_updated)
                       VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
                    (str(vcode), str(reporting_period),
                     econ_comments or '', business_plan_comments or '', accrued_pref_comment or '', pe_cap_comment or '')
                )
            conn.commit()
            conn.close()
        return True
    except Exception as e:
        return False
