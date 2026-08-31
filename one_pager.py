"""
one_pager.py
Core data retrieval and calculation functions for One Pager Investor Report

Provides functions to extract and calculate:
- General information from investment_map
- Capitalization stack from MRI_Loans, MRI_VAL, waterfalls, accounting_feed
- Property performance from ISBS_Download
- PE performance from accounting_feed, waterfalls
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


def _canonical_vcode(vcode: str, inv_map: pd.DataFrame) -> str:
    """Resolve an input vcode to the casing used in the deals table.

    Routes, scripts and deep links may pass any casing, but the cap-stack and
    PE blocks compare vcode case-sensitively against inv_map, waterfalls and
    valuations.  A mismatched casing therefore matched nothing and rendered a
    zero-filled section instead of raising — a blank report that looks real.
    Resolving to the stored casing here leaves every downstream comparison
    untouched for callers that already pass the right case.
    """
    raw = str(vcode).strip()
    if inv_map is None or inv_map.empty:
        return raw
    col = 'vcode' if 'vcode' in inv_map.columns else ('vCode' if 'vCode' in inv_map.columns else None)
    if col is None:
        return raw
    match = inv_map[inv_map[col].astype(str).str.strip().str.upper() == raw.upper()]
    if match.empty:
        return raw
    return str(match.iloc[0][col]).strip()


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


def _quarter_sort_key(quarter_str: str) -> Tuple[int, int]:
    """(year, quarter number) for ordering — 2026-Q2 -> (2026, 2)."""
    return (int(quarter_str.split('-')[0]), int(quarter_str.split('Q')[1]))


def most_recent_completed_quarter(quarters: List[str],
                                  today: Optional[date] = None) -> Optional[str]:
    """Newest quarter in `quarters` that has fully ended.

    Mirrors getMostRecentCompletedQuarter() in OnePagerView.vue. The Vue first
    load deliberately sends no quarter and only labels the dropdown afterwards,
    so the server-side default has to land on the same quarter the label
    promises. Defaulting to the newest available quarter instead diverges the
    moment an in-progress quarter's actuals arrive — the page then renders that
    quarter's figures under the previous quarter's label.

    Falls back to the oldest entry when nothing has completed, matching the Vue
    helper's `quarters[quarters.length - 1]` (the list is sorted newest-first).
    Returns None only for an empty list.
    """
    if not quarters:
        return None
    today = today or date.today()
    current = (today.year, (today.month - 1) // 3 + 1)

    completed = []
    for q in quarters:
        try:
            if _quarter_sort_key(q) < current:
                completed.append(q)
        except (ValueError, IndexError):
            continue  # ignore malformed entries rather than fail the report

    if not completed:
        return quarters[-1]
    return max(completed, key=_quarter_sort_key)


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

def _lookup_event_date(event_dates: Optional[pd.DataFrame], vcode: str,
                       event_type: str, event: str,
                       date_type: str) -> Optional[date]:
    """Look up a date from the MRI event_dates table by event criteria.

    Filters to vCode + vEventType + vEvent + vDateType, returns the latest
    dtEvent.  Returns None when the table is absent, no row matches, or the
    date cannot be parsed.
    """
    if event_dates is None or getattr(event_dates, 'empty', True) or not vcode:
        return None

    # Column case varies by source: MRI/CSV gives vCode/vEventType/dtEvent,
    # PostgreSQL folds unquoted identifiers to lowercase.
    cols = {str(c).strip().lower(): c for c in event_dates.columns}
    if not all(k in cols for k in ('vcode', 'veventtype', 'vevent', 'vdatetype', 'dtevent')):
        return None

    df = event_dates
    mask = (df[cols['vcode']].astype(str).str.strip().str.lower()
            == str(vcode).strip().lower())
    for col_key, want in (('veventtype', event_type.lower()),
                          ('vevent', event.lower()),
                          ('vdatetype', date_type.lower())):
        mask &= df[cols[col_key]].astype(str).str.strip().str.lower() == want

    hits = pd.to_datetime(df.loc[mask, cols['dtevent']], errors='coerce').dropna()
    if hits.empty:
        return None
    return hits.max().date()


def get_current_anticipated_exit(event_dates: Optional[pd.DataFrame],
                                 vcode: str) -> Optional[date]:
    """Current anticipated exit: Disposition / Closing / Projected."""
    return _lookup_event_date(event_dates, vcode,
                              'Disposition', 'Closing', 'Projected')


def get_underwritten_exit(event_dates: Optional[pd.DataFrame],
                          vcode: str) -> Optional[date]:
    """Underwritten exit: Asset Management / U/W Exit / Actual."""
    return _lookup_event_date(event_dates, vcode,
                              'Asset Management', 'U/W Exit', 'Actual')


def get_general_information(inv_map: pd.DataFrame, vcode: str,
                            event_dates: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Get general deal information from investment_map

    Args:
        inv_map: Investment map DataFrame
        vcode: Deal vcode
        event_dates: Optional MRI event_dates DataFrame, source of
            current_anticipated_exit. Omitted -> that field stays None.

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
        'current_anticipated_exit': None,
        'investment_name': '',
    }

    # Both exit dates come from event_dates — available even without inv_map.
    info['anticipated_exit'] = get_underwritten_exit(event_dates, vcode)
    info['current_anticipated_exit'] = get_current_anticipated_exit(event_dates, vcode)

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
        # anticipated_exit is sourced from event_dates above, not from inv
        'investment_name': ['Investment_Name', 'InvestmentName', 'vName'],
    }

    for key, possible_cols in col_mappings.items():
        for col in possible_cols:
            if col in row.index and pd.notna(row[col]):
                val = row[col]
                if key in ['units', 'sqft']:
                    clean = str(val).replace(',', '').strip()
                    info[key] = int(float(clean)) if pd.notna(val) else 0
                elif key in ['date_closed']:
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

def _child_vcodes_for_parent(vcode: str, inv_map: pd.DataFrame) -> List[str]:
    """Child property vcodes for a parent deal; empty list if this is not a parent.

    A deal counts as a parent only when Property_Count >= 1 — every child property
    carries 0, which is what stops siblings from picking up each other's loans.

    Children are matched on Portfolio_Name against either the parent's Investment_Name
    (the usual convention, e.g. 'Berger Pittsburgh Portfolio') or the parent's own
    Portfolio_Name, which covers deals whose portfolio label differs from their name
    (e.g. 'Burton Retail Portfolio' sitting inside the 'Burton Portfolio' group).
    """
    if inv_map is None or inv_map.empty:
        return []

    df = inv_map.copy()
    normalize_columns(df)
    if 'vcode' not in df.columns and 'vCode' in df.columns:
        df = df.rename(columns={'vCode': 'vcode'})
    if 'vcode' not in df.columns or 'Portfolio_Name' not in df.columns:
        return []

    df['vcode'] = df['vcode'].astype(str).str.strip()
    df['Portfolio_Name'] = df['Portfolio_Name'].fillna('').astype(str).str.strip()
    df['Investment_Name'] = df['Investment_Name'].fillna('').astype(str).str.strip() \
        if 'Investment_Name' in df.columns else ''

    deal_row = df[df['vcode'] == str(vcode).strip()]
    if deal_row.empty:
        return []
    row = deal_row.iloc[0]

    prop_count = pd.to_numeric(row.get('Property_Count'), errors='coerce')
    if pd.isna(prop_count) or prop_count < 1:
        return []

    labels = {row['Investment_Name'], row['Portfolio_Name']} - {''}
    if not labels:
        return []

    children = df[(df['Portfolio_Name'].isin(labels)) & (df['vcode'] != row['vcode'])]
    return children['vcode'].tolist()


def _loans_share_terms(deal_loans: pd.DataFrame) -> bool:
    """True when every loan row carries the same rate, maturity and interest type.

    Only meaningful on the portfolio-parent inheritance path. A portfolio financed by
    one co-terminous facility split across its properties yields N child loan rows with
    identical terms (Burton: 3 rows, all 5.665% Fixed maturing 8/28/2032). Those are one
    loan, not a tranche stack, so they must render as a single term with no second loan.

    Loans that genuinely differ fail this and keep the existing primary + second
    behaviour — e.g. Berger Pittsburgh's children each carry a senior plus a mezz piece
    (6 distinct term sets across 8 rows).

    Maturity is read from dtMaturity with a dtEvent fallback, matching _parse_loan below;
    in the MRI feed dtMaturity is frequently blank and the date lives in dtEvent.
    """
    if deal_loans is None or len(deal_loans) < 2:
        return False

    keys = set()
    for _, row in deal_loans.iterrows():
        rate = pd.to_numeric(row.get('nRate'), errors='coerce')
        rate = round(float(rate), 6) if pd.notna(rate) else None
        maturity = None
        for col in ('dtMaturity', 'dtEvent'):
            if col in row.index and pd.notna(row[col]):
                try:
                    maturity = pd.to_datetime(row[col]).date()
                    break
                except Exception:
                    pass
        keys.add((rate, maturity, str(row.get('vIntType', '')).strip().lower()))
    return len(keys) == 1


def _is_dev_deal(vcode: str, inv_map: pd.DataFrame) -> bool:
    """True when a deal is a development deal, by the app's one definition.

    Field precedence mirrors ``portfolio_snapshot_operating.resolve_strategy``:
    ``Investment_Strategy`` wins wherever it is populated, and ``Lifecycle``
    fills the gap. That fallback is not decoration — Investment_Strategy is
    0/110 populated on live (no MRI query selects it and it is absent from
    mri_service.MRI_COLUMNS), so Lifecycle decides every classification today.
    Reading Investment_Strategy alone would make this branch dead code.

    The strategy set itself comes from config.DEV_STRATEGIES, shared with the
    Portfolio Snapshot, so the two pages cannot disagree about which deals are
    development deals even though they show different debt for them.
    """
    from config import is_dev_deal as _is_dev

    if inv_map is None or inv_map.empty:
        return False

    df = inv_map
    col = 'vcode' if 'vcode' in df.columns else ('vCode' if 'vCode' in df.columns else None)
    if col is None:
        return False

    match = df[df[col].astype(str).str.strip().str.upper() == str(vcode).strip().upper()]
    if match.empty:
        return False
    row = match.iloc[0]

    for field in ('Investment_Strategy', 'Lifecycle'):
        if field in row.index and pd.notna(row[field]):
            value = str(row[field]).strip()
            if value:
                return _is_dev(value)
    return False


def _dev_hard_costs(vcode: str, inspection: pd.DataFrame,
                    inv_map: pd.DataFrame) -> Optional[float]:
    """Hard costs drawn to date for a deal, or None when there is nothing on record.

    SUM of ``mHardCosts`` over the deal's inspection rows. The live table
    currently holds one row per deal, so sum and latest are the same figure —
    but ``queries/MRI_Inspection.sql`` returns one row per draw, so a sum is
    what survives the real feed landing. ``mHardCosts`` is matched
    case-insensitively; note the plural, ``mHardCost`` does not exist.

    None means "no inspection row", which the caller uses to leave the deal on
    its existing ISBS debt. **0.0 is a value, not an absence** — a development
    deal that has drawn nothing yet reports zero hard costs, and that is data.
    Same convention as ``portfolio_snapshot_debt._num``.

    Parent→child inheritance matches the loan fallback in
    ``get_capitalization_stack``: a portfolio parent carrying no inspection of
    its own uses its children's rows.
    """
    if inspection is None or getattr(inspection, 'empty', True):
        return None

    df = inspection
    col = next((c for c in df.columns if c.lower() == 'vcode'), None)
    hard_col = next((c for c in df.columns if c.lower() == 'mhardcosts'), None)
    if col is None or hard_col is None:
        return None

    codes = df[col].astype(str).str.strip().str.upper()
    rows = df[codes == str(vcode).strip().upper()]

    if rows.empty:
        child_vcodes = {c.strip().upper() for c in _child_vcodes_for_parent(vcode, inv_map)}
        if child_vcodes:
            rows = df[codes.isin(child_vcodes)]

    if rows.empty:
        return None

    vals = pd.to_numeric(rows[hard_col], errors='coerce').dropna()
    if vals.empty:
        return None
    return float(vals.sum())


def get_capitalization_stack(
    vcode: str,
    mri_loans: pd.DataFrame,
    mri_val: pd.DataFrame,
    waterfalls: pd.DataFrame,
    acct: pd.DataFrame,
    inv_map: pd.DataFrame,
    isbs_raw: pd.DataFrame = None,
    quarter_str: str = None,
    relationships: pd.DataFrame = None,
    inspection: pd.DataFrame = None,
) -> Dict[str, Any]:
    """
    Get capitalization stack and deal terms

    Args:
        vcode: Deal vcode
        mri_loans: Loans DataFrame
        mri_val: Valuations DataFrame
        waterfalls: Waterfalls DataFrame
        acct: Accounting feed DataFrame
        inv_map: Investment map DataFrame
        inspection: Construction draw inspections. When supplied, a development
            deal's ``debt`` is the hard costs drawn to date rather than the
            ISBS balance. See the debt-basis block below.

    Returns:
        Dictionary with cap stack data
    """
    cap = {
        'purchase_price': 0.0,
        'pe_coupon': 0.0,  # From waterfalls nPercent
        # None, NOT 0.0 — the One Pager must tell "no participation term on
        # this deal" (renders N/A) apart from "the term is nil" (renders 0.0%).
        # A present zero is a real answer: deal_terms.pe_split_capital == 0 says
        # the PE takes no share of the residual.  See _pe_terms_fallback().
        'pe_participation': None,  # From waterfalls FXRate where vState='Share'
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
        # The debt actually shown, plus the untouched ISBS basis beside it. The
        # Portfolio Snapshot reads the *_isbs twins so its Debt and Total Cap
        # columns do not move when a dev deal is rebased here. See the
        # debt-basis block below and portfolio_snapshot_debt.resolve_debt.
        'debt_basis': '',
        'debt_isbs': 0.0,
        'total_cap_isbs': 0.0,
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

    vcode_str = _canonical_vcode(vcode, inv_map)

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
            if deal_loans.empty:
                # Portfolio parent holding its debt on the children: when those child
                # loans are one co-terminous facility, the parent's loan amount is their
                # sum. Differing-terms parents are left alone (debt stays 0 here).
                child_vcodes = {c.upper() for c in _child_vcodes_for_parent(vcode_str, inv_map)}
                if child_vcodes:
                    inherited = loans[loans['vCode'].str.upper().isin(child_vcodes)]
                    if _loans_share_terms(inherited):
                        deal_loans = inherited
            if not deal_loans.empty and 'mOrigLoanAmt' in deal_loans.columns:
                cap['debt'] = pd.to_numeric(deal_loans['mOrigLoanAmt'], errors='coerce').fillna(0).sum()

    # ---- Debt basis: development deals report hard costs drawn to date ----
    #
    # A development deal's ISBS balance answers the wrong question on this page.
    # What the One Pager is reporting for a deal still under construction is how
    # much has actually been drawn, which is the Inspection table's mHardCosts,
    # summed over the deal's draws.
    #
    # Deliberately placed AFTER both branches above so it is an override of a
    # settled figure, not a third branch: a dev deal with no inspection row
    # (18 of the 26 on live) keeps exactly the debt it had before this existed.
    # Only `hard_costs is not None` overrides, so 0.0 — a dev deal that has
    # drawn nothing — is honoured as the real $0.0M it is.
    #
    # THE SNAPSHOT MUST NOT FOLLOW. `debt_isbs` is captured before the override
    # and `total_cap_isbs` is computed from it further down, because the
    # Portfolio Snapshot resolves dev debt to the full committed facility
    # (portfolio_snapshot_debt.resolve_debt, PDF footnote (6)) and reads
    # cap_stack for Total Cap. Without these twins, rebasing `debt` here would
    # silently move the Snapshot's Total Cap column and break its tie to the
    # published PDF.
    cap['debt_isbs'] = cap['debt']
    cap['debt_basis'] = 'ISBS Interim BS (as of quarter end)'
    if _is_dev_deal(vcode_str, inv_map):
        hard_costs = _dev_hard_costs(vcode_str, inspection, inv_map)
        if hard_costs is not None:
            cap['debt'] = hard_costs
            cap['debt_basis'] = 'mHardCosts (hard costs drawn to date)'

    # Loan terms — always from MRI_Loans (independent of debt source)
    if mri_loans is not None and not mri_loans.empty:
        loans = mri_loans.copy()
        normalize_columns(loans)
        if 'vCode' not in loans.columns and 'vcode' in loans.columns:
            loans = loans.rename(columns={'vcode': 'vCode'})
        if 'vCode' in loans.columns:
            loans['vCode'] = loans['vCode'].astype(str).str.strip()
            deal_loans = loans[loans['vCode'] == vcode_str]
            inherited_from_children = False

            if deal_loans.empty:
                # A parent portfolio deal can hold all its debt on the child properties
                # (e.g. Burton Retail Portfolio). Without this the parent renders 'N/A'
                # even though the loans are in the table.
                child_vcodes = {c.upper() for c in _child_vcodes_for_parent(vcode_str, inv_map)}
                if child_vcodes:
                    deal_loans = loans[loans['vCode'].str.upper().isin(child_vcodes)]
                    inherited_from_children = not deal_loans.empty

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

                # Second loan — suppressed when a portfolio parent has inherited its
                # children's loans and those are one co-terminous facility split across
                # the properties. Sorting by amount would otherwise promote a sibling
                # property's identical loan into the second slot as a phantom tranche
                # (Burton: Jubilee primary, Westwood "second", Foley dropped).
                # A single property with a real primary + second tranche is unaffected:
                # inherited_from_children is False for it.
                collapse_to_single = (inherited_from_children
                                      and _loans_share_terms(deal_loans_sorted))

                if len(deal_loans_sorted) > 1 and not collapse_to_single:
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

    # Get equity from accounting feed (filtered to quarter end when available)
    if acct is not None and not acct.empty and inv_map is not None:
        from loaders import build_investmentid_to_vcode

        try:
            inv_to_vcode = build_investmentid_to_vcode(inv_map)
            deal_investment_ids = [iid for iid, vc in inv_to_vcode.items() if str(vc) == vcode_str]

            acct_norm = acct.copy()
            normalize_columns(acct_norm)
            acct_norm["InvestmentID"] = acct_norm["InvestmentID"].astype(str).str.strip()
            deal_acct = acct_norm[acct_norm["InvestmentID"].isin(deal_investment_ids)].copy()

            # Filter to transactions on or before the quarter end date
            if not deal_acct.empty and quarter_str:
                _, q_end = quarter_to_date_range(quarter_str)
                deal_acct["EffectiveDate"] = pd.to_datetime(deal_acct["EffectiveDate"], errors="coerce")
                deal_acct = deal_acct[deal_acct["EffectiveDate"].dt.date <= q_end].copy()

            if not deal_acct.empty:
                deal_acct["MajorType"] = deal_acct["MajorType"].fillna("").astype(str).str.strip()
                deal_acct["Amt"] = pd.to_numeric(deal_acct["Amt"], errors="coerce").fillna(0.0)

                if "TypeName" not in deal_acct.columns and "Typename" in deal_acct.columns:
                    deal_acct["TypeName"] = deal_acct["Typename"]
                elif "TypeName" not in deal_acct.columns:
                    deal_acct["TypeName"] = ""
                deal_acct["TypeName"] = deal_acct["TypeName"].fillna("").astype(str).str.strip()
                deal_acct["InvestorID"] = deal_acct["InvestorID"].astype(str).str.strip()

                # Get committed PE from accounting commitment rows (non-OP only)
                if "is_commitment" in deal_acct.columns:
                    non_op_mask = ~deal_acct["InvestorID"].str.upper().str.startswith("OP")
                    commitment_rows = deal_acct[deal_acct["is_commitment"] & non_op_mask]
                    if not commitment_rows.empty:
                        cap['committed_pe'] = commitment_rows["Amt"].abs().sum()

                investor_balances = {}
                for _, row in deal_acct.iterrows():
                    investor_id = row["InvestorID"]
                    major_type = row["MajorType"].lower()
                    type_name = row["TypeName"].lower()
                    amt = float(row["Amt"])

                    # Skip commitment rows — pledges, not cash activity
                    if row.get("is_commitment", False):
                        continue

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

    # The same total on the pre-override debt basis, for the Portfolio Snapshot.
    # Identical to total_cap for every deal this page did not rebase.
    cap['total_cap_isbs'] = cap['debt_isbs'] + cap['pref_equity'] + cap['partner_equity']

    if cap['total_cap'] > 0:
        cap['debt_pct'] = cap['debt'] / cap['total_cap']
        cap['pref_equity_pct'] = cap['pref_equity'] / cap['total_cap']
        cap['partner_equity_pct'] = cap['partner_equity'] / cap['total_cap']
        cap['pe_exposure_on_cap'] = (cap['debt'] + cap['pref_equity']) / cap['total_cap'] * 100

    if cap['current_valuation'] > 0:
        cap['pe_exposure_on_value'] = (cap['debt'] + cap['pref_equity']) / cap['current_valuation'] * 100

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
        'Vacancy': ['4040', '4043', '4030', '4031', '4042'],
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
    # Tax abatement — in U/W (Projected IS) this is below NOI in acct 7070,
    # but in actuals it's netted into 5090 (Real Estate Taxes).  Include here
    # so calc_amounts() can fold it into expenses for apples-to-apples comparison.
    'TAX_ABATEMENT': ['7070'],
    # Balance-sheet debt accounts for principal from balance changes
    'DEBT_BS_ACCTS': ['2150', '2152', '2210'],
    # Underwriting total debt service account (Projected IS)
    'UW_DEBT_SERVICE': ['7010'],
}

#: Operating reserve releases that NET against the gross at-close 5xxx costs.
#: AT CLOSE ONLY — DO NOT "TIDY" THIS AWAY.
#:
#: 7083 is OPERATING RESERVE RELEASE: a credit (negative mAmount) funding the
#: gross operating costs booked in the same period, so adding it to expenses
#: leaves the real cost borne by the property. The at-close buckets key off the
#: 5xxx prefix, so the offset — a 7xxx account — was dropped entirely and the
#: column showed gross cost against zero revenue. 30 Bearfoot (P0000001) at
#: 2020-12-31 is the case: 5060 16,668.00 + 5090 37,152.00 + 5110 6,280.00 =
#: 60,100.00 against 7083 of -60,100.00, on a deal whose rows sum to exactly 0.
#:
#: A NAMED ACCOUNT, NOT A '7xxx' RULE, and deliberately only this one. 7083 is
#: on exactly one deal portfolio-wide. Every other 7xxx account reaching an
#: at-close date must stay out: 7073/7074 are capital proceeds (netting them
#: adds 22.2M to Jefferson Addison Heights), 7071/7072 distributions, 7050
#: capex, 7075 reserves for replacement (below NOI, 61 deals), and 7080 offsets
#: 7010 debt service, not operating cost.
#:
#: Mirrored in queries/Prop_Info_AtClose.sql, which nets the same account into
#: at_close_expenses for the primary path. Change both or the two disagree.
AT_CLOSE_RESERVE_RELEASE_ACCTS = ['7083']


def get_property_performance(
    vcode: str,
    quarter_str: str,
    isbs_df: pd.DataFrame,
    mri_val: pd.DataFrame,
    occupancy_df: pd.DataFrame = None,
    budget_econ_occ_df: pd.DataFrame = None,
    at_close_noi_df: pd.DataFrame = None,
    deal_terms_df: pd.DataFrame = None,
    mri_loans_all_df: pd.DataFrame = None,
    inv_map: pd.DataFrame = None,
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
        mri_loans_all_df: All MRI loans including Paid Off (for refi detection)

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

        # Tax abatement (7070): stored as negative (credit) in U/W Projected IS,
        # but in actuals it's already netted into 5090.  Adding it here reduces
        # expenses for U/W and is zero for actuals — makes columns comparable.
        abatement = period_data[
            period_data['vAccount'].isin(IS_ACCOUNTS['TAX_ABATEMENT'])
        ]['mAmount'].sum()
        expenses += abatement  # negative credit reduces expenses

        for category, acct_list in IS_ACCOUNTS['DEBT_SERVICE'].items():
            debt_service += period_data[period_data['vAccount'].isin(acct_list)]['mAmount'].sum()

        noi = revenue - expenses
        return revenue, expenses, noi, debt_service

    # Get actual data (Interim IS)
    actual_data = isbs[isbs['vSource'] == 'Interim IS']
    bs_data = isbs[isbs['vSource'] == 'Interim BS']
    budget_data = isbs[isbs['vSource'] == 'Budget IS']
    uw_data = isbs[isbs['vSource'] == 'Projected IS']

    # Budget fallback: when no budget rows exist for the report year, reuse
    # the prior year's budget (per LLC agreement).  Shift dates forward by
    # 12 months so downstream date-range filters work unchanged.
    #
    # NOT FOR DEVELOPMENT DEALS (creator decision, Jim, 2026-08-31).  A
    # stabilized deal's prior year is a fair stand-in for the current one.  A
    # development deal's is a LEASE-UP year that by definition will not repeat,
    # so republishing it as the current projection invents a Projected YE out
    # of a budget the property has already grown past.  Live examples this
    # rule exists for: Jefferson Addison Heights (P0000077) published a
    # -700K Projected YE NOI assembled entirely from its shifted 2025 lease-up
    # budget, and the four Brainerd buildings published a Projected YE with no
    # actual component in it at all.
    #
    # Declining to shift IS the whole fix.  The prior-year rows stay in
    # budget_data but fall outside every current-year date range below, so they
    # contribute 0 and a dev deal lands on exactly the path a deal with no
    # budget at all already takes: Projected YE = YTD Actual, YTD Budget = 0.
    # There is deliberately no second branch here to fall out of step.
    #
    # Detection is the app's one definition (config.DEV_STRATEGIES, via
    # _is_dev_deal) so this cannot disagree with the dev debt basis above or
    # the Portfolio Snapshot's "Dev" suppression.  inv_map is None only in
    # build_chart_data() at the bottom of this module, which reads ytd_actual,
    # uw_ye and economic_occ.ytd_actual — never a budget-derived field — so the
    # chart is unaffected either way.
    _report_year = int(quarter_str.split('-')[0])
    if not budget_data.empty and not _is_dev_deal(vcode_str, inv_map):
        has_report_year_budget = (budget_data['dtEntry_parsed'].dt.year == _report_year).any()
        if not has_report_year_budget:
            prior = budget_data[budget_data['dtEntry_parsed'].dt.year == _report_year - 1].copy()
            if not prior.empty:
                prior['dtEntry_parsed'] = prior['dtEntry_parsed'] + pd.DateOffset(years=1)
                budget_data = prior

    # Detect refi / payoff events from MRI_Loans for this deal.
    # When a loan has vDateType='Paid Off', the BS balance change in that month
    # reflects the payoff (not amortization) and must be excluded.
    _refi_months = set()  # set of (year, month) tuples where a payoff occurred
    if mri_loans_all_df is not None and not mri_loans_all_df.empty:
        vc_col = next((c for c in mri_loans_all_df.columns if c.lower() == 'vcode'), None)
        dt_col = next((c for c in mri_loans_all_df.columns if c.lower() == 'vdatetype'), None)
        ev_col = next((c for c in mri_loans_all_df.columns if c.lower() == 'dtevent'), None)
        if vc_col and dt_col and ev_col:
            deal_loans = mri_loans_all_df[
                mri_loans_all_df[vc_col].astype(str).str.strip().str.lower() == vcode_str
            ]
            paid_off = deal_loans[
                deal_loans[dt_col].astype(str).str.strip().str.lower() == 'paid off'
            ]
            for _, row in paid_off.iterrows():
                po_date = pd.to_datetime(row[ev_col], errors='coerce')
                if pd.notna(po_date):
                    _refi_months.add((po_date.year, po_date.month))

    def _get_bs_principal_change(bs_df, quarter_str, ytd_date):
        """Compute actual YTD principal from BS debt balance change.

        Principal paid = abs(balance at prior Dec 31) - abs(balance at ytd_date).
        This is the authoritative source for actual principal payments.

        A loan payoff inside that window is not amortization.  The subtraction
        spans every month from prior Dec 31 forward, so one payoff month
        contaminates the figure for the rest of the year — and the reported month
        moving past it does not clear it.  A month MRI_Loans flags as a payoff
        (vDateType='Paid Off', collected into _refi_months) therefore has its
        delta replaced by the nearest preceding non-payoff month's balance
        activity.

        Detection is by label only.  Sizing a drop against the deal's own
        amortization run rate would also catch the refis and curtailments MRI
        never labels (The Gallery, Pontchartrain, PMAT), but those are being
        corrected in MRI at the data layer instead — so this stays deliberately
        narrow rather than second-guessing a balance the source system reports as
        correct.

        Deals with no flagged month return the plain subtraction unchanged, so
        this is a no-op wherever MRI reports no payoff.
        """
        if bs_df.empty:
            return 0
        bs_accts = IS_ACCOUNTS['DEBT_BS_ACCTS']
        debt_bs = bs_df[bs_df['vAccount'].isin(bs_accts)]
        if debt_bs.empty:
            return 0

        year = int(quarter_str.split('-')[0])

        # Monthly balance series, oldest first — indexable so month-over-month
        # activity around an event month can be inspected.
        bal = debt_bs.groupby('dtEntry_parsed')['mAmount'].sum().abs().sort_index()
        if bal.empty:
            return 0
        bs_periods = list(bal.index)

        def _period_at(target_ts, direction='le'):
            candidates = [p for p in bs_periods
                          if (direction == 'le' and pd.Timestamp(p) <= target_ts)
                          or (direction == 'ge' and pd.Timestamp(p) >= target_ts)]
            if not candidates:
                return None
            return candidates[-1] if direction == 'le' else candidates[0]

        # Start balance: prior Dec 31 (or closest available before Jan 1)
        prior_dec = pd.Timestamp(f"{year - 1}-12-31")
        start_p = _period_at(prior_dec, 'le')

        # End balance: at or before the YTD date
        end_p = _period_at(pd.Timestamp(ytd_date), 'le')

        if start_p is None or end_p is None:
            return 0

        base = max(0, bal[start_p] - bal[end_p])

        # This deal's own amortization run rate, over its whole reported history.
        all_deltas = (bal.shift(1) - bal).dropna()
        paydowns = all_deltas[all_deltas > 0]
        median_paydown = float(paydowns.median()) if len(paydowns) else 0.0

        def _drop_at(idx):
            return float(bal.iloc[idx - 1] - bal.iloc[idx])

        def _is_payoff_month(idx):
            """True when MRI_Loans flags a loan Paid Off in month `idx`."""
            if _drop_at(idx) <= 0:
                return False   # balance rose or held — nothing to exclude
            ts = pd.Timestamp(bs_periods[idx])
            return (ts.year, ts.month) in _refi_months

        start_i, end_i = bs_periods.index(start_p), bs_periods.index(end_p)
        payoff_idx = [i for i in range(start_i + 1, end_i + 1) if _is_payoff_month(i)]
        if not payoff_idx:
            return base

        adjusted = base
        for i in payoff_idx:
            # Substitute the nearest preceding non-payoff month's activity; fall
            # back to the deal's median when no clean month precedes the payoff.
            substitute = None
            for j in range(i - 1, 0, -1):
                # Re-test rather than checking window membership: a payoff in the
                # month before the window (a December payoff, say) is outside
                # payoff_idx but is just as unusable as a substitute.
                if _is_payoff_month(j):
                    continue
                substitute = max(0.0, _drop_at(j))
                break
            if substitute is None:
                substitute = max(0.0, median_paydown)
            adjusted = adjusted - _drop_at(i) + substitute

        return max(0, adjusted)

    # NOTE: unreachable since b2b0a40 (2026-08-14) replaced both call sites with
    # _get_bs_principal_change().  Retained for reference only — the payoff guard
    # it carries (c1f6230) now lives in _get_bs_principal_change() above.  Do not
    # assume _refi_months is honoured here; it is not on any live path.
    def _estimate_principal_from_bs(bs_df, qtr_end_ts, months_elapsed):
        """Estimate YTD principal from BS debt account balance changes.

        Uses only the trailing 12 months of balance sheet data and filters
        out outlier deltas (> 3× median) that indicate loan restructuring
        or payoff events rather than normal amortization.
        """
        if bs_df.empty:
            return 0
        bs_accts = IS_ACCOUNTS['DEBT_BS_ACCTS']
        debt_bs = bs_df[bs_df['vAccount'].isin(bs_accts)]
        if debt_bs.empty:
            return 0
        bs_periods = sorted(debt_bs['dtEntry_parsed'].dropna().unique())
        if len(bs_periods) < 2:
            return 0

        # Only consider periods within the trailing 12 months up to quarter end
        lookback_start = qtr_end_ts - pd.DateOffset(months=12)

        # Build list of (period_date, balance) tuples in the lookback window
        period_bals = []
        for p in bs_periods:
            ts = pd.Timestamp(p)
            if lookback_start <= ts <= qtr_end_ts:
                bal = abs(debt_bs[debt_bs['dtEntry_parsed'] == p]['mAmount'].sum())
                period_bals.append((ts, bal))
        if len(period_bals) < 2:
            return 0

        # Compute month-over-month balance changes, skipping refi months
        raw_deltas = []
        for i in range(1, len(period_bals)):
            cur_ts, cur_bal = period_bals[i]
            prv_ts, prv_bal = period_bals[i - 1]
            # Skip if the current period's month had a loan payoff
            if (cur_ts.year, cur_ts.month) in _refi_months:
                continue
            delta = max(0, prv_bal - cur_bal)
            raw_deltas.append(delta)

        if not raw_deltas:
            return 0

        # Filter out outlier deltas (> 3× median) that indicate restructuring
        # events rather than normal amortization
        sorted_deltas = sorted(raw_deltas)
        median_delta = sorted_deltas[len(sorted_deltas) // 2]
        if median_delta > 0:
            clean_deltas = [d for d in raw_deltas if d <= 3 * median_delta]
        else:
            clean_deltas = raw_deltas

        if clean_deltas:
            monthly_principal = sum(clean_deltas) / len(clean_deltas)
        else:
            monthly_principal = 0

        return monthly_principal * months_elapsed

    # Determine months elapsed for the quarter
    qtr_num = int(quarter_str.split('-Q')[1]) if '-Q' in quarter_str else 1
    months_elapsed = qtr_num * 3  # Q1=3, Q2=6, Q3=9, Q4=12

    # Find the as-of date for YTD actual (last date in quarter's year, on or before quarter end)
    ytd_date = None
    report_year = int(quarter_str.split('-')[0])
    if not actual_data.empty:
        actual_periods = sorted(actual_data['dtEntry_parsed'].dropna().unique())
        # Find period closest to or before quarter end, but within the same year
        for p in reversed(actual_periods):
            pts = pd.Timestamp(p)
            if pts.year == report_year and pts.date() <= quarter_end:
                ytd_date = pts
                break

        if ytd_date:
            rev, exp, noi, _ = calc_amounts(actual_data, as_of_date=ytd_date)
            perf['revenue']['ytd_actual'] = rev
            perf['expenses']['ytd_actual'] = exp
            perf['noi']['ytd_actual'] = noi

            # YTD Actual DSCR: Interest (5190) + Principal from BS balance change
            # Principal comes from BS balance change (not 7060) to avoid double-counting
            ytd_is = actual_data[actual_data['dtEntry_parsed'] == ytd_date]
            ytd_interest = abs(ytd_is[ytd_is['vAccount'] == '5190']['mAmount'].sum())
            ytd_principal = _get_bs_principal_change(bs_data, quarter_str, ytd_date)
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
    # When no current-year actuals exist (ytd_date is None), use full-year budget.
    year = int(quarter_str.split('-')[0])
    has_current_year_actuals = ytd_date is not None
    if has_current_year_actuals and not budget_data.empty:
        ytd_rev = perf['revenue']['ytd_actual']
        ytd_exp = perf['expenses']['ytd_actual']
        ytd_noi = perf['noi']['ytd_actual']

        # YTD actual debt service (interest + principal from BS balance change)
        ytd_is = actual_data[actual_data['dtEntry_parsed'] == ytd_date]
        ytd_interest = abs(ytd_is[ytd_is['vAccount'] == '5190']['mAmount'].sum())
        ytd_principal = _get_bs_principal_change(bs_data, quarter_str, ytd_date)
        ytd_ds = ytd_interest + ytd_principal

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
    elif has_current_year_actuals:
        # No budget data — use YTD actual only
        perf['revenue']['actual_ye'] = perf['revenue']['ytd_actual']
        perf['expenses']['actual_ye'] = perf['expenses']['ytd_actual']
        perf['noi']['actual_ye'] = perf['noi']['ytd_actual']
        perf['dscr']['actual_ye'] = perf['dscr']['ytd_actual']
    elif not budget_data.empty:
        # No current-year actuals — Projected YE = full-year budget
        jan1 = pd.Timestamp(f"{year}-01-01") - pd.DateOffset(days=1)
        dec31 = pd.Timestamp(f"{year}-12-31")
        rev, exp, noi, ds = calc_amounts(budget_data, sum_range=(jan1, dec31))
        perf['revenue']['actual_ye'] = rev
        perf['expenses']['actual_ye'] = exp
        perf['noi']['actual_ye'] = noi
        if abs(ds) > 0:
            perf['dscr']['actual_ye'] = noi / abs(ds)

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
                # Detect partial-year debt service.  When the original U/W
                # exit date falls before year-end, account 7010's YTD
                # cumulative only covers months through the projected exit.
                # Dividing 12 months of NOI by < 12 months of DS inflates
                # the ratio (e.g. Ascent: 12 mo NOI / 3 mo DS → 10.69X
                # instead of ~2X).  Fix: find the last month the cumulative
                # was still growing and annualise.
                uw_7010 = uw_data[
                    (uw_data['vAccount'].isin(IS_ACCOUNTS['UW_DEBT_SERVICE']))
                    & (uw_data['dtEntry_parsed'].dt.year == year)
                ]
                if not uw_7010.empty:
                    monthly_cum = (
                        uw_7010.groupby(uw_7010['dtEntry_parsed'].dt.month)['mAmount']
                        .sum().abs().sort_index()
                    )
                    if len(monthly_cum) >= 2:
                        vals = list(monthly_cum.items())
                        months_active = vals[0][0]  # at least the first month
                        for i in range(len(vals) - 1, 0, -1):
                            if abs(vals[i][1] - vals[i - 1][1]) > 1.0:
                                months_active = vals[i][0]
                                break
                        if months_active < 12:
                            uw_ds = uw_ds * (12 / months_active)
                perf['dscr']['uw_ye'] = noi / uw_ds

            # U/W YE Economic Occupancy from Projected IS: 1 - (vacancy / rental income)
            # 4010 = Rental Income (negative/credit), 4030/4031 = Vacancy Loss (positive/debit;
            # commercial deals book vacancy to 4031, some exclusively)
            uw_dec = uw_data[uw_data['dtEntry_parsed'] == dec_date]
            if not uw_dec.empty:
                rental = uw_dec[uw_dec['vAccount'] == '4010']['mAmount'].sum()  # negative
                vacancy = uw_dec[uw_dec['vAccount'].isin(['4030', '4031'])]['mAmount'].sum()  # positive
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

                    # Adjust for tax abatement (7070) in Projected IS at close.
                    # The MRI pre-computed at_close_noi uses 5xxx for expenses,
                    # so 7070 (below-NOI abatement) is not included.  Find it
                    # from the Projected IS at the earliest December date.
                    if not uw_data.empty and at_close_filled:
                        uw_periods_all = sorted(uw_data['dtEntry_parsed'].dropna().unique())
                        ac_dec = next(
                            (pd.Timestamp(p) for p in uw_periods_all if pd.Timestamp(p).month == 12),
                            None,
                        )
                        if ac_dec is not None:
                            ac_period = uw_data[uw_data['dtEntry_parsed'] == ac_dec]
                            ac_abate = ac_period[
                                ac_period['vAccount'].isin(IS_ACCOUNTS['TAX_ABATEMENT'])
                            ]['mAmount'].sum()
                            if ac_abate != 0:
                                perf['expenses']['at_close'] += ac_abate
                                perf['noi']['at_close'] = perf['revenue']['at_close'] - perf['expenses']['at_close']
                                ds = abs(float(acn_int or 0)) + abs(float(acn_prin or 0))
                                if ds > 0:
                                    perf['dscr']['at_close'] = perf['noi']['at_close'] / ds

        # Fallback: earliest December 31 in Projected IS = due diligence audit
        if not at_close_filled:
            dec_dates = [pd.Timestamp(p) for p in uw_periods if pd.Timestamp(p).month == 12]
            if dec_dates:
                at_close_date = min(dec_dates)
                rev, exp, noi, ds = calc_amounts(uw_data, as_of_date=at_close_date)

                # Net the operating reserve release against the gross cost —
                # see AT_CLOSE_RESERVE_RELEASE_ACCTS.  Applied HERE rather than
                # inside calc_amounts() on purpose: that helper also serves the
                # YTD Actual, YTD Budget, Projected YE and U/W YE columns, and
                # this netting is an at-close-only correction.  Same shape as
                # the 7070 abatement fold-in above — the release is a credit,
                # so adding it reduces expenses.
                ac_release = uw_data[
                    (uw_data['dtEntry_parsed'] == at_close_date)
                    & (uw_data['vAccount'].isin(AT_CLOSE_RESERVE_RELEASE_ACCTS))
                ]['mAmount'].sum()
                exp += ac_release
                noi = rev - exp

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
            # Fallback: parent portfolio deal — try child vcodes
            if dt_row.empty and inv_map is not None:
                child_vcodes = _child_vcodes_for_parent(vcode_str, inv_map)
                if child_vcodes:
                    child_lower = {c.lower() for c in child_vcodes}
                    dt_row = dtf[dtf['vcode'].isin(child_lower)]
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
                        occ_col2 = next((c for c in rem_months.columns if c.lower() == 'pctoccupied'), None)
                        if occ_col2:
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

    # Separate supplement rows (periodic) from base ISBS rows (YTD cumulative)
    is_supp = df.get('_is_supplement', pd.Series(False, index=df.index)).fillna(False).astype(bool)
    base_df = df[~is_supp]
    supp_df = df[is_supp]

    distributions = []

    # Process base ISBS rows: YTD cumulative → periodic conversion
    if not base_df.empty:
        periods = sorted(base_df['dtEntry_parsed'].unique())
        prev_by_year = {}  # {year: last_cumulative_amount}

        for period in periods:
            period_ts = pd.Timestamp(period)
            period_date = period_ts.date()

            # Only include periods within our time range
            if period_date < inception or period_date > end_date:
                # Still track cumulative for YTD math
                period_data = base_df[base_df['dtEntry_parsed'] == period]
                cumulative = float(period_data['mAmount'].sum())
                prev_by_year[period_ts.year] = cumulative
                continue

            period_data = base_df[base_df['dtEntry_parsed'] == period]
            cumulative = float(period_data['mAmount'].sum())

            # Convert cumulative to periodic
            year = period_ts.year
            if year in prev_by_year:
                periodic = cumulative - prev_by_year[year]
            else:
                # First entry for this year — the cumulative IS the
                # periodic total so far, whether this is January or a
                # mid-year first appearance (e.g. a one-time ROC event
                # in November).  Dividing by month number incorrectly
                # assumed the cumulative was evenly spread across months.
                periodic = cumulative

            prev_by_year[year] = cumulative

            amount = abs(periodic)
            if amount > 0.01:
                distributions.append((period_date, amount))

    # Process supplement rows: already periodic, no cumulative conversion
    for _, row in supp_df.iterrows():
        period_date = pd.Timestamp(row['dtEntry_parsed']).date()
        if period_date < inception or period_date > end_date:
            continue
        amount = abs(float(row['mAmount']))
        if amount > 0.01:
            distributions.append((period_date, amount))

    distributions.sort(key=lambda x: x[0])
    return distributions


def _get_uw_pe_distributions(
    isbs_raw: pd.DataFrame, vcode: str, inception: date, end_date: date,
) -> List[Tuple[date, float]]:
    """Backward-compatible wrapper — extracts UW PE distributions (7071)."""
    return _get_uw_pe_periodic(isbs_raw, vcode, inception, end_date, UW_PE_DIST_ACCT)


def _get_uw_7073_signed(
    isbs_raw: pd.DataFrame,
    vcode: str,
    inception: date,
    end_date: date,
) -> List[Tuple[date, float]]:
    """Extract signed capital events from ISBS Projected IS account 7073.

    Sign convention in MRI: positive = contribution, negative = return of capital.
    Returns cashflow-sign tuples for calculate_roe():
      positive MRI → negative cashflow (contribution)
      negative MRI → positive cashflow (return of capital)
    """
    if isbs_raw is None or isbs_raw.empty:
        return []

    df = isbs_raw.copy()
    if 'vcode' in df.columns:
        df = df[df['vcode'] == str(vcode).strip().lower()]
    if df.empty:
        return []

    if 'vSource' in df.columns:
        df = df[df['vSource'] == 'Projected IS']
    if df.empty or 'vAccount' not in df.columns:
        return []

    df = df[df['vAccount'] == UW_PE_ROC_ACCT]
    if df.empty:
        return []

    if 'dtEntry_parsed' not in df.columns:
        return []

    df = df.dropna(subset=['dtEntry_parsed'])
    df = df.sort_values('dtEntry_parsed')

    # Separate supplement rows (periodic) from base ISBS rows (YTD cumulative)
    is_supp = df.get('_is_supplement', pd.Series(False, index=df.index)).fillna(False).astype(bool)
    base_df = df[~is_supp]
    supp_df = df[is_supp]

    events: List[Tuple[date, float]] = []

    # Process base ISBS rows: YTD cumulative → periodic conversion
    if not base_df.empty:
        periods = sorted(base_df['dtEntry_parsed'].unique())
        prev_by_year: dict = {}

        for period in periods:
            period_ts = pd.Timestamp(period)
            period_date = period_ts.date()

            period_data = base_df[base_df['dtEntry_parsed'] == period]
            cumulative = float(period_data['mAmount'].sum())

            year = period_ts.year
            if year in prev_by_year:
                periodic = cumulative - prev_by_year[year]
            else:
                # First entry for this year — the cumulative IS the
                # periodic total so far.  The old logic divided by
                # month number, which incorrectly annualised one-time
                # events (e.g. $5.77M ROC at Nov → $524K).
                periodic = cumulative

            prev_by_year[year] = cumulative

            if period_date < inception or period_date > end_date:
                continue

            if abs(periodic) < 0.01:
                continue

            events.append((period_date, -periodic))

    # Process supplement rows: already periodic, no cumulative conversion
    for _, row in supp_df.iterrows():
        period_date = pd.Timestamp(row['dtEntry_parsed']).date()
        if period_date < inception or period_date > end_date:
            continue
        periodic = float(row['mAmount'])
        if abs(periodic) < 0.01:
            continue
        # positive = contribution → negative cashflow; negative = ROC → positive cashflow
        events.append((period_date, -periodic))

    # Drop an event that arrived identically from both sources.
    #
    # The two paths above read the same account from two tables that overlap:
    # isbs_uw_supplements is a sparse CSV backfill for deals where MRI's
    # Projected IS was missing 7073, but MRI has since started reporting some
    # of them. Burton carries its 6/30/2025 contribution of 26,597,500 in
    # BOTH, so it was counted twice and its U/W ROE ran on double the capital.
    #
    # Keyed on (date, amount), NOT date alone. Seasons at Bel Air has two
    # genuinely different supplement contributions on 2025-06-30
    # (-3,562,882 and -2,518,202); collapsing by date would silently merge
    # them. Rounded to the cent because the base path reaches its value
    # through a cumulative subtraction and the supplement path reads it
    # directly, so the two can differ in the float tail.
    #
    # Verified against live: of the 61 deals carrying 7073 data, only Burton
    # changes (3 contributions -> 2), and every deal's returns of capital are
    # byte-identical either side.
    deduped: List[Tuple[date, float]] = []
    seen: set = set()
    for event_date, amount in events:
        key = (event_date, round(amount, 2))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((event_date, amount))
    events = deduped

    events.sort(key=lambda x: x[0])
    return events


def _get_uw_roc_events(
    isbs_raw: pd.DataFrame,
    vcode: str,
    inception: date,
    end_date: date,
    capital_balance: float,
) -> List[Tuple[date, float]]:
    """Extract underwritten return-of-capital events from Projected IS account 7073.

    Account 7073 in Projected IS stores YTD cumulative amounts.  Planned ROC
    events appear as a jump in the cumulative on the event date, then the same
    cumulative repeats each subsequent month through year-end.

    This function detects each jump (delta > $1) as a distinct ROC event on the
    date it first appears, ignoring the flat repeats.

    ``capital_balance`` is the PE capital outstanding at inception (total
    contributions minus actual returns of capital).  The returned events are
    capped so they never reduce the running capital below zero.

    Returns list of (date, amount) for each ROC event within inception..end_date.
    """
    if isbs_raw is None or isbs_raw.empty:
        return []

    df = isbs_raw.copy()
    if 'vcode' in df.columns:
        df = df[df['vcode'] == str(vcode).strip().lower()]
    if df.empty:
        return []

    if 'vSource' in df.columns:
        df = df[df['vSource'] == 'Projected IS']
    if df.empty or 'vAccount' not in df.columns:
        return []

    df = df[df['vAccount'] == UW_PE_ROC_ACCT]
    if df.empty:
        return []

    if 'dtEntry_parsed' not in df.columns:
        return []

    df = df.dropna(subset=['dtEntry_parsed'])
    df = df.sort_values('dtEntry_parsed')

    periods = sorted(df['dtEntry_parsed'].unique())
    if not periods:
        return []

    events = []
    prev_cumulative = None
    prev_year = None
    remaining_capital = capital_balance

    for period in periods:
        period_ts = pd.Timestamp(period)
        period_date = period_ts.date()
        year = period_ts.year

        period_data = df[df['dtEntry_parsed'] == period]
        cumulative = abs(float(period_data['mAmount'].sum()))

        # Reset tracking at year boundary
        if prev_year is not None and year != prev_year:
            prev_cumulative = None

        # Detect a jump: new cumulative exceeds prior by > $1
        if prev_cumulative is None:
            delta = cumulative
        else:
            delta = cumulative - prev_cumulative

        prev_cumulative = cumulative
        prev_year = year

        if delta < 1.0:
            continue
        if period_date < inception or period_date > end_date:
            continue
        if remaining_capital <= 0:
            continue

        # Cap so capital never goes negative
        capped = min(delta, remaining_capital)
        remaining_capital -= capped
        events.append((period_date, capped))

    return events


# ============================================================
# PREFERRED EQUITY PERFORMANCE
# ============================================================

def _pe_terms_fallback(pe: Dict[str, Any], deal_terms: pd.DataFrame,
                       vcode_str: str) -> None:
    """Fill a still-zero PE coupon / participation from the MRI deal terms.

    THE SAME TWO FIELDS the Capitalization block uses, so the two blocks agree:
    ``pe_coupon`` -> Coupon and ``pe_split_capital`` -> Participation, matching
    ``_enrich_cap_stack_from_deal_terms()`` in ``financials_service.py``.  Keep
    the mapping and the ``< 1`` percent normalisation identical in both places
    or one page will print two different numbers for the same term again.

    The one deliberate difference is precedence.  The cap stack lets deal_terms
    OVERRIDE the waterfall; this does not.  The waterfall is the structure that
    actually runs, so it stays primary and a deal that has one is untouched.
    """
    dt = deal_terms.copy()
    normalize_columns(dt)
    if 'vcode' not in dt.columns and 'vCode' in dt.columns:
        dt = dt.rename(columns={'vCode': 'vcode'})
    if 'vcode' not in dt.columns:
        return
    # Case-insensitive, unlike the cap stack's exact match.  Both live sources
    # store uppercase P-codes so the two resolve the same row today; this is
    # the safer side to err on if a casing ever diverges.
    row = dt[dt['vcode'].astype(str).str.strip().str.upper()
             == str(vcode_str).strip().upper()]
    if row.empty:
        return
    r = row.iloc[0]

    for key, col in (('coupon', 'pe_coupon'),
                     ('participation', 'pe_split_capital')):
        v = pd.to_numeric(r.get(col), errors='coerce')
        if pd.isna(v) or v < 0:
            continue                      # nothing to say about this field
        v = float(v)

        if key == 'coupon':
            # Unchanged: fill only when the waterfall gave nothing, and only
            # from a positive rate.  A 0% coupon is not a thing.
            if pe.get(key) or v == 0:
                continue
        else:
            # Participation carries ONE extra rule: an explicit zero in
            # deal_terms overrides whatever the waterfall produced.
            #
            # The waterfall reader takes the first vState='Share' row with no
            # PropCode filter, so on a deal where the PE has no Share row it
            # silently reports the OPERATING PARTNER's share instead — there is
            # no way for it to say "the PE participates in nothing", which is
            # exactly what that shape means.  pe_split_capital == 0 is MRI
            # stating the term affirmatively, so it wins.  Live today this is
            # Jefferson Eastchase (P0000085) and Jefferson Addison Heights
            # (P0000077): both carry two Share rows owned by OPJPI at
            # FXRate 1.0, which the `< 1` heuristic below renders as 1%.
            #
            # A POSITIVE deal_terms value still defers to the waterfall, so the
            # six deals where the two sources disagree on a real number are
            # untouched.  Keep it that way — picking a winner there is a
            # separate question about which source is right.
            if pe.get(key) is not None and v != 0:
                continue

        pe[key] = v if v < 1 else v / 100


def get_pe_performance(
    vcode: str,
    quarter_str: str,
    acct: pd.DataFrame,
    waterfalls: pd.DataFrame,
    inv_map: pd.DataFrame,
    isbs_raw: pd.DataFrame = None,
    deal_terms: pd.DataFrame = None,
) -> Dict[str, Any]:
    """
    Get Preferred Equity performance metrics

    Args:
        vcode: Deal vcode
        quarter_str: Quarter string
        acct: Accounting feed DataFrame
        waterfalls: Waterfalls DataFrame
        inv_map: Investment map DataFrame
        isbs_raw: ISBS DataFrame (for U/W ROE from Projected IS account 7071)
        deal_terms: Deal terms DataFrame (MRI txfinancial_IC).  Fallback only,
            for coupon / participation — see _pe_terms_fallback().

    Returns:
        Dictionary with PE performance metrics
    """
    pe = {
        'committed_pe': 0.0,
        'remaining_to_fund': 0.0,
        'coupon': 0.0,
        # None, NOT 0.0 — see the matching note in get_capitalization_stack().
        'participation': None,
        'funded_to_date': 0.0,
        'return_of_capital': 0.0,
        'roe_to_date': 0.0,
        'uw_roe_to_date': 0.0,
        'current_pe_balance': 0.0,
        'accrued_balance': 0.0,
    }

    vcode_str = _canonical_vcode(vcode, inv_map)
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

    # Whatever the waterfall did not supply, take from the MRI deal terms.
    #
    # Three live deals carry complete deal_terms and ZERO waterfall rows —
    # Plaza Del Mar (P0000116), Apple Self Storage (P0000003) and Jefferson
    # Stephens (P0000114).  The `waterfalls` table is in PROTECTED_TABLES and
    # is NOT MRI-fed; it is built by hand in Waterfall Setup, so no refresh
    # will ever populate it for them.  Until it is built, this block printed
    # "N/A" while the Capitalization block above it printed 8.50% from the same
    # contract — the same deal, the same page, two different answers.
    if deal_terms is not None and not deal_terms.empty:
        _pe_terms_fallback(pe, deal_terms, vcode_str)

    # Get funded, committed PE, and ROC from accounting feed
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

                # Get committed PE from accounting commitment rows (non-OP only)
                if "is_commitment" in deal_acct.columns:
                    non_op_mask = ~deal_acct["InvestorID"].str.upper().str.startswith("OP")
                    commitment_rows = deal_acct[deal_acct["is_commitment"] & non_op_mask]
                    if not commitment_rows.empty:
                        pe['committed_pe'] = commitment_rows["Amt"].abs().sum()

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

                    # Skip commitment rows — pledges, not cash activity
                    if row.get("is_commitment", False):
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
                        # else: acquisition fee — excluded from both capital_events
                        # and cf_distributions (no effect on ROE)

                # Include grace-period pref distributions in ROE.
                # Pref payments are contractually due within 30 days of
                # quarter close; a 45-day window captures timely payments
                # that arrived after quarter_end.  They are assigned to
                # quarter_end so annualisation uses the quarter boundary,
                # not the payment date.
                from datetime import timedelta
                grace_end = quarter_end + timedelta(days=45)
                grace_acct = acct_norm[
                    (acct_norm["InvestmentID"].isin(deal_investment_ids)) &
                    (acct_norm["EffectiveDate"].dt.date > quarter_end) &
                    (acct_norm["EffectiveDate"].dt.date <= grace_end)
                ].copy()
                if not grace_acct.empty:
                    grace_acct["MajorType"] = grace_acct["MajorType"].fillna("").astype(str).str.strip()
                    grace_acct["Amt"] = pd.to_numeric(grace_acct["Amt"], errors="coerce").fillna(0.0)
                    if "TypeName" not in grace_acct.columns and "Typename" in grace_acct.columns:
                        grace_acct["TypeName"] = grace_acct["Typename"]
                    elif "TypeName" not in grace_acct.columns:
                        grace_acct["TypeName"] = ""
                    grace_acct["TypeName"] = grace_acct["TypeName"].fillna("").astype(str).str.strip()
                    grace_acct["InvestorID"] = grace_acct["InvestorID"].astype(str).str.strip()
                    if "TypeID" in grace_acct.columns:
                        grace_acct["_tid"] = pd.to_numeric(grace_acct["TypeID"], errors="coerce").fillna(0)
                    else:
                        grace_acct["_tid"] = 0

                    for _, grow in grace_acct.iterrows():
                        if grow["InvestorID"].upper().startswith("OP"):
                            continue
                        if grow.get("is_commitment", False):
                            continue
                        g_major = grow["MajorType"].lower()
                        g_tname = grow["TypeName"].lower()
                        g_amt = float(grow["Amt"])
                        g_tid = float(grow["_tid"])
                        if "distri" not in g_major:
                            continue
                        is_pref = (
                            g_tid == 1019.0
                            or "preferred return" in g_tname
                            or "pref return" in g_tname
                        )
                        if is_pref:
                            # Assign to quarter_end — the payment services
                            # this quarter, it just landed late.
                            if g_amt >= 0:
                                capital_events.append((quarter_end, g_amt))
                            cf_distributions.append((quarter_end, g_amt))

                # Compute ROE to Date from actual accounting through quarter end
                if capital_events:
                    from metrics import calculate_roe
                    inception = min(d for d, _ in capital_events)
                    pe['roe_to_date'] = calculate_roe(
                        capital_events, cf_distributions, inception, quarter_end
                    )

                    # Compute U/W ROE to Date from ISBS Projected IS ONLY
                    # 7073: positive = contribution, negative = return of capital
                    # 7071: underwritten distributions (ROE numerator)
                    # No actual accounting data used.
                    if isbs_raw is not None and not isbs_raw.empty:
                        uw_capital = _get_uw_7073_signed(
                            isbs_raw, vcode, date(2000, 1, 1), quarter_end
                        )
                        uw_dists = _get_uw_pe_distributions(
                            isbs_raw, vcode, date(2000, 1, 1), quarter_end
                        )
                        if uw_capital or uw_dists:
                            all_dates = [d for d, _ in uw_capital] + [d for d, _ in uw_dists]
                            uw_inception = min(all_dates) if all_dates else inception
                            uw_capital = [(d, a) for d, a in uw_capital if d <= quarter_end]
                            uw_dists = [(d, a) for d, a in uw_dists if d >= uw_inception and d <= quarter_end]
                            pe['uw_roe_to_date'] = calculate_roe(
                                uw_capital, uw_dists, uw_inception, quarter_end
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
