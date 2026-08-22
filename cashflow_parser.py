"""
cashflow_parser.py
Generic parser for partner Excel/CSV cash flow models.

Reads a spreadsheet and extracts monthly or annual cash flow rows
(revenue, expenses, capex, NOI) into a list of dicts compatible
with the prospect_cashflows table schema.

No DB or Flask dependencies. Takes file bytes, returns structured dicts.
"""

import io
import re
from datetime import date
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np

from utils import month_end


# ============================================================
# COLUMN DETECTION PATTERNS
# ============================================================
# Partners label columns inconsistently. These patterns match
# common variants for each cash flow category.

_DATE_PATTERNS = [
    r'^date$', r'^period', r'^month', r'^year', r'^time',
]

_REVENUE_PATTERNS = [
    r'revenue', r'income', r'gross.?rev', r'egi', r'effective.?gross',
    r'rental.?income', r'total.?revenue', r'total.?income',
    r'potential.?rent', r'base.?rent', r'gross.?income',
    r'gross.?potential', r'collected.?rent',
]

_EXPENSE_PATTERNS = [
    r'expense', r'opex', r'operating.?exp', r'total.?exp',
    r'total.?opex', r'operating.?cost',
]

_NOI_PATTERNS = [
    r'\bnoi\b', r'net.?operating', r'net.?income',
]

_CAPEX_PATTERNS = [
    r'capex', r'capital.?exp', r'capital.?reserve', r'cap.?ex',
    r'tenant.?improve', r'leasing.?comm', r'\bti\b.*\blc\b',
    r'capital.?cost',
]


def _match_column(col_name: str, patterns: list) -> bool:
    """Check if a column name matches any of the given regex patterns."""
    col_lower = str(col_name).lower().strip()
    return any(re.search(p, col_lower) for p in patterns)


def _find_column(columns: list, patterns: list) -> Optional[str]:
    """Find the first column matching the patterns."""
    for col in columns:
        if _match_column(col, patterns):
            return col
    return None


def _detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """Find the date/period column, trying name patterns then dtype."""
    # Try name patterns first
    for col in df.columns:
        if _match_column(col, _DATE_PATTERNS):
            return col

    # Try the first column if it looks like dates
    first_col = df.columns[0]
    sample = df[first_col].dropna().head(5)
    if len(sample) > 0:
        try:
            parsed = pd.to_datetime(sample, format='mixed', dayfirst=False)
            if parsed.notna().sum() >= len(sample) * 0.6:
                return first_col
        except Exception:
            pass

    return None


def _is_annual(dates: pd.Series) -> bool:
    """Detect if dates represent annual periods (gaps > 300 days between consecutive)."""
    sorted_dates = dates.dropna().sort_values()
    if len(sorted_dates) < 2:
        return True
    diffs = sorted_dates.diff().dropna()
    median_gap = diffs.dt.days.median()
    return median_gap > 300


def _annual_to_monthly(rows: List[Dict], year: int) -> List[Dict]:
    """Spread an annual row across 12 monthly rows."""
    monthly = []
    rev = float(rows[0].get('revenue') or 0)
    exp = float(rows[0].get('expenses') or 0)
    capex = float(rows[0].get('capex') or 0)
    noi = float(rows[0].get('noi') or 0)

    for m in range(1, 13):
        monthly.append({
            'period_date': str(month_end(date(year, m, 1))),
            'revenue': round(rev / 12, 2),
            'expenses': round(exp / 12, 2),
            'capex': round(capex / 12, 2),
            'noi': round(noi / 12, 2),
        })
    return monthly


def parse_cashflow_excel(
    file_bytes: bytes,
    filename: str,
) -> Dict[str, Any]:
    """Parse a partner's Excel/CSV cash flow model.

    Auto-detects column mapping and date format. Handles both monthly
    and annual data (annual is spread to monthly).

    Returns:
        {
            'periods': int,
            'frequency': 'monthly' | 'annual',
            'cashflows': [{'period_date', 'revenue', 'expenses', 'capex', 'noi'}, ...],
            'columns_detected': {'revenue': col_name, ...},
            'metadata': {'filename': ..., 'sheet': ..., 'rows_parsed': ...},
        }
    """
    # Read file
    fname_lower = filename.lower()
    try:
        if fname_lower.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        else:
            # Try first sheet, skip common header rows
            xls = pd.ExcelFile(io.BytesIO(file_bytes))
            sheet_name = xls.sheet_names[0]

            # Try reading with different header offsets
            df = None
            for skip in range(6):
                candidate = pd.read_excel(
                    io.BytesIO(file_bytes), sheet_name=sheet_name,
                    header=skip, dtype=str,
                )
                # Check if we found numeric data columns
                num_cols = 0
                for col in candidate.columns:
                    sample = candidate[col].dropna().head(5)
                    try:
                        pd.to_numeric(sample.str.replace(',', '').str.replace('$', '').str.replace('(', '-').str.replace(')', ''))
                        num_cols += 1
                    except Exception:
                        pass
                if num_cols >= 2:
                    df = candidate
                    break

            if df is None:
                df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)
    except Exception as e:
        return {'error': f'Failed to read file: {e}'}

    if df.empty:
        return {'error': 'File is empty'}

    # Drop fully empty rows and columns
    df = df.dropna(how='all').dropna(axis=1, how='all')

    # Detect columns
    date_col = _detect_date_column(df)
    rev_col = _find_column(df.columns, _REVENUE_PATTERNS)
    exp_col = _find_column(df.columns, _EXPENSE_PATTERNS)
    noi_col = _find_column(df.columns, _NOI_PATTERNS)
    capex_col = _find_column(df.columns, _CAPEX_PATTERNS)

    columns_detected = {
        'date': date_col,
        'revenue': rev_col,
        'expenses': exp_col,
        'noi': noi_col,
        'capex': capex_col,
    }

    if not date_col:
        return {
            'error': 'Could not detect a date/period column',
            'columns_detected': columns_detected,
            'available_columns': list(df.columns),
        }

    # Need at least revenue+expenses OR noi
    if not rev_col and not noi_col:
        return {
            'error': 'Could not detect revenue or NOI columns',
            'columns_detected': columns_detected,
            'available_columns': list(df.columns),
        }

    # Parse numeric values
    def to_num(series):
        if series is None:
            return pd.Series([0.0] * len(df))
        s = series.astype(str).str.replace(',', '').str.replace('$', '')
        s = s.str.replace('(', '-', regex=False).str.replace(')', '', regex=False)
        return pd.to_numeric(s, errors='coerce').fillna(0)

    # Parse dates
    try:
        dates = pd.to_datetime(df[date_col], format='mixed', dayfirst=False)
    except Exception:
        try:
            # Try interpreting as years
            years = pd.to_numeric(df[date_col], errors='coerce')
            dates = years.apply(lambda y: pd.Timestamp(year=int(y), month=12, day=31) if pd.notna(y) and 1990 <= y <= 2060 else pd.NaT)
        except Exception:
            return {'error': f'Could not parse dates from column "{date_col}"'}

    valid_mask = dates.notna()
    df = df[valid_mask].copy()
    dates = dates[valid_mask]

    if df.empty:
        return {'error': 'No valid date rows found'}

    rev_data = to_num(df[rev_col]) if rev_col else None
    exp_data = to_num(df[exp_col]) if exp_col else None
    noi_data = to_num(df[noi_col]) if noi_col else None
    capex_data = to_num(df[capex_col]) if capex_col else None

    # Derive missing columns
    if rev_data is not None and exp_data is not None and noi_data is None:
        noi_data = rev_data - exp_data.abs()
    elif noi_data is not None and rev_data is None and exp_data is None:
        # Only NOI provided — estimate revenue as NOI/0.60, expenses as remainder
        rev_data = noi_data.abs() / 0.60
        exp_data = rev_data - noi_data.abs()

    # Normalize signs: revenue positive, expenses positive (stored as positive in prospect_cashflows)
    if rev_data is not None:
        rev_data = rev_data.abs()
    if exp_data is not None:
        exp_data = exp_data.abs()
    if capex_data is not None:
        capex_data = capex_data.abs()
    if noi_data is not None:
        # Keep NOI sign as-is (should be positive for profitable properties)
        pass

    # Detect frequency
    annual = _is_annual(dates)
    frequency = 'annual' if annual else 'monthly'

    # Build rows
    raw_rows = []
    for i in range(len(df)):
        dt = dates.iloc[i]
        raw_rows.append({
            'date': dt,
            'year': dt.year,
            'revenue': float(rev_data.iloc[i]) if rev_data is not None else 0,
            'expenses': float(exp_data.iloc[i]) if exp_data is not None else 0,
            'capex': float(capex_data.iloc[i]) if capex_data is not None else 0,
            'noi': float(noi_data.iloc[i]) if noi_data is not None else 0,
        })

    # Convert to monthly if annual
    cashflows = []
    if annual:
        for row in raw_rows:
            monthly = _annual_to_monthly([row], row['year'])
            cashflows.extend(monthly)
    else:
        for row in raw_rows:
            dt = row['date']
            period_date = month_end(date(dt.year, dt.month, 1))
            cashflows.append({
                'period_date': str(period_date),
                'revenue': round(row['revenue'], 2),
                'expenses': round(row['expenses'], 2),
                'capex': round(row['capex'], 2),
                'noi': round(row['noi'], 2),
            })

    return {
        'periods': len(raw_rows),
        'frequency': frequency,
        'cashflows': cashflows,
        'columns_detected': columns_detected,
        'metadata': {
            'filename': filename,
            'rows_parsed': len(raw_rows),
            'monthly_rows': len(cashflows),
        },
    }
