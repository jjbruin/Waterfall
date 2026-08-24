"""
argus_parser.py
Stateless parser for Argus Enterprise Excel exports.

Parses three export types:
1. Monthly Cash Flow — monthly line items mapped to COA accounts
2. Rent Roll Summary — tenant lease detail with escalations
3. Revenue Assumptions — market leasing profiles

No DB or Flask dependencies. Takes Excel bytes, returns structured dicts.
"""

import io
import re
from datetime import date
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np

from config import (
    GROSS_REVENUE_ACCTS, CONTRA_REVENUE_ACCTS,
    EXPENSE_ACCTS, ALL_EXCLUDED, TAX_ABATEMENT_ACCTS,
)


# ============================================================
# COA ACCOUNT MAPPING
# ============================================================
# Keyword-based matching: Argus users customize line item names,
# so we match on substrings rather than exact strings.
# Order matters — first match wins.

ARGUS_COA_MAP: List[Tuple[str, int, str]] = [
    # (keyword pattern, COA account, category)
    # Revenue
    ("potential base rent", 4010, "revenue"),
    ("base rent", 4010, "revenue"),
    ("scheduled rent", 4010, "revenue"),
    ("minimum rent", 4010, "revenue"),
    ("contract rent", 4010, "revenue"),
    ("rental income", 4010, "revenue"),
    ("absorption.*vacancy", 4030, "revenue"),
    ("turnover vacancy", 4030, "revenue"),
    ("vacancy", 4030, "revenue"),
    ("ret recovery", 4091, "revenue"),
    ("real estate tax recovery", 4091, "revenue"),
    ("tax recovery", 4091, "revenue"),
    ("ins recovery", 4092, "revenue"),
    ("insurance recovery", 4092, "revenue"),
    ("cam recovery", 4090, "revenue"),
    ("common area recovery", 4090, "revenue"),
    ("expense recovery", 4090, "revenue"),
    ("expense reimburse", 4090, "revenue"),
    ("percentage rent", 4075, "revenue"),
    ("pylon", 4075, "revenue"),
    ("other revenue", 4075, "revenue"),
    ("other income", 4075, "revenue"),
    ("miscellaneous income", 4075, "revenue"),
    ("parking", 4075, "revenue"),
    ("antenna", 4075, "revenue"),
    ("billboard", 4075, "revenue"),
    ("concession", 4040, "revenue"),
    ("credit loss", 4043, "revenue"),
    ("bad debt", 4043, "revenue"),
    # Expenses
    ("property management", 5040, "expense"),
    ("management fee", 5040, "expense"),
    ("real estate tax", 5090, "expense"),
    ("property tax", 5090, "expense"),
    ("insurance expense", 5110, "expense"),
    ("insurance", 5110, "expense"),
    ("cam expense", 5060, "expense"),
    ("common area maintenance", 5060, "expense"),
    ("maintenance", 5060, "expense"),
    ("repairs", 5060, "expense"),
    ("r&m", 5060, "expense"),
    ("utilities", 5060, "expense"),
    ("non-recoverable", 5020, "expense"),
    ("general & admin", 5020, "expense"),
    ("g&a", 5020, "expense"),
    ("administrative", 5020, "expense"),
    ("marketing", 5020, "expense"),
    ("payroll", 5020, "expense"),
    ("other expense", 5020, "expense"),
    # CapEx
    ("tenant improvement", 7050, "capex"),
    ("leasing commission", 7050, "capex"),
    ("capital reserve", 7050, "capex"),
    ("capital expenditure", 7050, "capex"),
    ("capex", 7050, "capex"),
    ("roof", 7050, "capex"),
    ("hvac", 7050, "capex"),
    ("structural", 7050, "capex"),
]


def map_to_coa(line_item: str) -> Tuple[Optional[int], Optional[str]]:
    """Map an Argus line item name to (COA account, category).

    Uses keyword matching against ARGUS_COA_MAP.
    Returns (None, None) if no match found.
    """
    if not line_item:
        return None, None
    item_lower = line_item.strip().lower()
    for pattern, account, category in ARGUS_COA_MAP:
        if re.search(pattern, item_lower):
            return account, category
    return None, None


# ============================================================
# MONTHLY CASH FLOW PARSER
# ============================================================

def parse_monthly_cashflow(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Parse Argus Enterprise Monthly Cash Flow Excel export.

    Argus exports are typically structured with:
    - Row headers in column A (line item names)
    - Monthly period columns (dates or "Month 1", "Month 2", etc.)
    - Possibly a header row with dates

    Returns:
        {
            'periods': [date, ...],
            'line_items': [
                {'name': str, 'coa_account': int|None, 'category': str|None,
                 'amounts': [float, ...], 'mapped': bool},
                ...
            ],
            'occupancy': [float, ...] or None,
            'metadata': {'filename': str, 'total_periods': int, 'mapped_count': int,
                         'unmapped_items': [str, ...]},
        }
    """
    df = _read_excel_flexible(file_bytes, filename)
    if df is None or df.empty:
        return _empty_cashflow_result(filename)

    # Find the header row with period dates
    periods, header_row_idx, data_start_col = _detect_periods(df)
    if not periods:
        return _empty_cashflow_result(filename, error="Could not detect monthly periods")

    # Parse line items
    line_items = []
    occupancy = None
    unmapped = []

    for idx in range(len(df)):
        if idx <= header_row_idx:
            continue

        row = df.iloc[idx]
        label = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
        if not label or label.lower() in ("", "nan", "none"):
            continue

        # Skip summary/total rows
        if _is_summary_row(label):
            continue

        # Extract amounts for each period
        amounts = []
        for col_idx in range(data_start_col, data_start_col + len(periods)):
            if col_idx < len(row):
                val = row.iloc[col_idx]
                amounts.append(_to_float(val))
            else:
                amounts.append(0.0)

        # Check for occupancy row
        if _is_occupancy_row(label):
            occupancy = amounts
            continue

        # Map to COA
        coa_account, category = map_to_coa(label)
        mapped = coa_account is not None
        if not mapped:
            unmapped.append(label)

        line_items.append({
            "name": label,
            "coa_account": coa_account,
            "category": category,
            "amounts": amounts,
            "mapped": mapped,
        })

    mapped_count = sum(1 for li in line_items if li["mapped"])

    return {
        "periods": [_date_to_str(p) for p in periods],
        "line_items": line_items,
        "occupancy": occupancy,
        "metadata": {
            "filename": filename,
            "total_periods": len(periods),
            "total_line_items": len(line_items),
            "mapped_count": mapped_count,
            "unmapped_items": unmapped,
        },
    }


# ============================================================
# RENT ROLL SUMMARY PARSER
# ============================================================

def parse_rent_roll_summary(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Parse Argus Enterprise Rent Roll Summary Excel export.

    Rent roll exports have multi-row tenant blocks with fields like:
    tenant name, suite, SF, lease dates, rent, recoveries, TI/LC, etc.

    Returns:
        {
            'tenants': [
                {'tenant_name': str, 'suite': str, 'square_feet': float,
                 'lease_start': str, 'lease_end': str, 'base_rent_annual': float,
                 'base_rent_psf': float, 'recovery_type': str, ...},
                ...
            ],
            'rent_steps': [
                {'tenant_index': int, 'effective_date': str, 'annual_rent': float,
                 'rent_psf': float, 'step_type': str, 'step_pct': float},
                ...
            ],
            'summary': {'total_sf': float, 'total_rent': float, 'tenant_count': int,
                        'occupied_sf': float, 'vacant_sf': float},
        }
    """
    df = _read_excel_flexible(file_bytes, filename)
    if df is None or df.empty:
        return {"tenants": [], "rent_steps": [], "summary": {}}

    # Normalize column names
    cols = [str(c).strip().lower() for c in df.columns]
    df.columns = cols

    # Try to find tenant-structured data
    tenants = []
    rent_steps = []

    # Strategy: look for columns that match tenant fields
    col_map = _detect_rent_roll_columns(cols)

    if col_map.get("tenant_name"):
        for idx, row in df.iterrows():
            tenant_name = _safe_str(row.get(col_map["tenant_name"]))
            if not tenant_name:
                continue

            tenant = {
                "tenant_name": tenant_name,
                "suite": _safe_str(row.get(col_map.get("suite", ""), "")),
                "square_feet": _to_float(row.get(col_map.get("square_feet", ""), 0)),
                "lease_type": _safe_str(row.get(col_map.get("lease_type", ""), "")),
                "lease_start": _safe_date_str(row.get(col_map.get("lease_start", ""))),
                "lease_end": _safe_date_str(row.get(col_map.get("lease_end", ""))),
                "base_rent_annual": _to_float(row.get(col_map.get("base_rent_annual", ""), 0)),
                "base_rent_psf": _to_float(row.get(col_map.get("base_rent_psf", ""), 0)),
                "recovery_type": _safe_str(row.get(col_map.get("recovery_type", ""), "")),
                "ret_recovery_psf": _to_float(row.get(col_map.get("ret_recovery_psf", ""), 0)),
                "ins_recovery_psf": _to_float(row.get(col_map.get("ins_recovery_psf", ""), 0)),
                "cam_recovery_psf": _to_float(row.get(col_map.get("cam_recovery_psf", ""), 0)),
                "ti_psf": _to_float(row.get(col_map.get("ti_psf", ""), 0)),
                "lc_psf": _to_float(row.get(col_map.get("lc_psf", ""), 0)),
                "renewal_probability": _to_float(row.get(col_map.get("renewal_probability", ""), 0)),
                "cpi_pct": _to_float(row.get(col_map.get("cpi_pct", ""), 0)),
                "free_rent_months": int(_to_float(row.get(col_map.get("free_rent_months", ""), 0))),
                "pct_rent_breakpoint": _to_float(row.get(col_map.get("pct_rent_breakpoint", ""), 0)),
                "pct_rent_rate": _to_float(row.get(col_map.get("pct_rent_rate", ""), 0)),
                "security_deposit": _to_float(row.get(col_map.get("security_deposit", ""), 0)),
                "is_vacant": _is_vacant_tenant(tenant_name),
            }
            tenants.append(tenant)

    # Summary
    total_sf = sum(t["square_feet"] for t in tenants)
    occupied_sf = sum(t["square_feet"] for t in tenants if not t["is_vacant"])
    vacant_sf = sum(t["square_feet"] for t in tenants if t["is_vacant"])
    total_rent = sum(t["base_rent_annual"] for t in tenants if not t["is_vacant"])

    return {
        "tenants": tenants,
        "rent_steps": rent_steps,
        "summary": {
            "total_sf": total_sf,
            "occupied_sf": occupied_sf,
            "vacant_sf": vacant_sf,
            "total_rent": total_rent,
            "tenant_count": len(tenants),
        },
    }


# ============================================================
# REVENUE ASSUMPTIONS PARSER
# ============================================================

def parse_revenue_assumptions(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Parse Argus Enterprise Revenue Assumptions Excel export.

    Contains market leasing profiles: base rent, term, renewal prob,
    vacancy months, TI/LC, step %, CPI, etc.

    Returns:
        {'profiles': [{'profile_name': str, 'base_rent_psf': float, ...}, ...]}
    """
    df = _read_excel_flexible(file_bytes, filename)
    if df is None or df.empty:
        return {"profiles": []}

    cols = [str(c).strip().lower() for c in df.columns]
    df.columns = cols

    col_map = _detect_market_profile_columns(cols)
    profiles = []

    if col_map.get("profile_name"):
        for idx, row in df.iterrows():
            name = _safe_str(row.get(col_map["profile_name"]))
            if not name:
                continue

            profiles.append({
                "profile_name": name,
                "base_rent_psf": _to_float(row.get(col_map.get("base_rent_psf", ""), 0)),
                "term_months": int(_to_float(row.get(col_map.get("term_months", ""), 0))),
                "renewal_probability": _to_float(row.get(col_map.get("renewal_probability", ""), 0)),
                "vacancy_months": int(_to_float(row.get(col_map.get("vacancy_months", ""), 0))),
                "ti_new_psf": _to_float(row.get(col_map.get("ti_new_psf", ""), 0)),
                "ti_renewal_psf": _to_float(row.get(col_map.get("ti_renewal_psf", ""), 0)),
                "lc_new_pct": _to_float(row.get(col_map.get("lc_new_pct", ""), 0)),
                "lc_renewal_pct": _to_float(row.get(col_map.get("lc_renewal_pct", ""), 0)),
                "fixed_step_pct": _to_float(row.get(col_map.get("fixed_step_pct", ""), 0)),
                "cpi_pct": _to_float(row.get(col_map.get("cpi_pct", ""), 0)),
                "recovery_type": _safe_str(row.get(col_map.get("recovery_type", ""), "")),
            })

    return {"profiles": profiles}


# ============================================================
# FORECAST DATAFRAME CONVERTER
# ============================================================

def cashflow_to_forecast_df(
    parsed: Dict[str, Any],
    vcode: str,
    pro_yr_base: int,
) -> pd.DataFrame:
    """Convert parsed monthly cashflow to forecast-compatible DataFrame.

    Produces the same schema as load_forecast() output:
    vcode, event_date, vAccount, mAmount, Pro_Yr, vAccountType, mAmount_norm

    Follows prospect_analysis.py:_fc_row() for sign normalization.
    """
    rows = []
    periods = parsed.get("periods", [])
    line_items = parsed.get("line_items", [])

    for li in line_items:
        coa_account = li.get("coa_account")
        if coa_account is None:
            continue  # Skip unmapped items

        amounts = li.get("amounts", [])
        for i, period_str in enumerate(periods):
            if i >= len(amounts):
                break

            amount = amounts[i]
            if amount == 0.0:
                continue

            period_date = pd.to_datetime(period_str).date()
            rows.append(_fc_row(vcode, period_date, coa_account, amount, pro_yr_base))

    if not rows:
        return pd.DataFrame(columns=[
            "vcode", "event_date", "vAccount", "mAmount", "Pro_Yr",
            "vAccountType", "mAmount_norm",
        ])

    return pd.DataFrame(rows)


def _fc_row(vcode: str, period_date, account: int, amount: float, pro_yr_base: int) -> dict:
    """Create a single forecast row with normalized amount.

    Same logic as prospect_analysis.py:_fc_row().
    """
    acct = int(account)
    raw = float(amount)

    if acct in GROSS_REVENUE_ACCTS:
        norm = abs(raw)
    elif acct in CONTRA_REVENUE_ACCTS:
        norm = -abs(raw)
    elif acct in EXPENSE_ACCTS:
        norm = -abs(raw)
    elif acct in ALL_EXCLUDED:
        norm = -abs(raw)
    elif acct in TAX_ABATEMENT_ACCTS:
        norm = abs(raw)
    else:
        norm = raw

    pro_yr = period_date.year - pro_yr_base
    return {
        "vcode": vcode,
        "event_date": period_date,
        "vAccount": acct,
        "mAmount": raw,
        "Pro_Yr": pro_yr,
        "vAccountType": "",
        "mAmount_norm": norm,
    }


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _read_excel_flexible(file_bytes: bytes, filename: str) -> Optional[pd.DataFrame]:
    """Read Excel file with flexible format detection."""
    try:
        buf = io.BytesIO(file_bytes)
        # Try openpyxl first (xlsx), then xlrd (xls)
        try:
            df = pd.read_excel(buf, header=None, engine="openpyxl")
        except Exception:
            buf.seek(0)
            df = pd.read_excel(buf, header=None, engine="xlrd")
        return df
    except Exception:
        return None


def _detect_periods(df: pd.DataFrame) -> Tuple[List[date], int, int]:
    """Detect monthly period dates from Excel header rows.

    Scans first 10 rows looking for a row with multiple date-like values.
    Returns (period_dates, header_row_index, data_start_column).
    """
    for row_idx in range(min(10, len(df))):
        row = df.iloc[row_idx]
        dates = []
        start_col = None

        for col_idx in range(len(row)):
            val = row.iloc[col_idx]
            parsed_date = _try_parse_date(val)
            if parsed_date:
                if start_col is None:
                    start_col = col_idx
                dates.append(parsed_date)

        # Need at least 6 dates to consider this a period header
        if len(dates) >= 6 and start_col is not None:
            return dates, row_idx, start_col

    # Fallback: look for "Month 1", "Year 1 Month 1" patterns
    for row_idx in range(min(10, len(df))):
        row = df.iloc[row_idx]
        month_cols = []
        start_col = None
        for col_idx in range(len(row)):
            val = str(row.iloc[col_idx]).strip().lower()
            if re.match(r"(month|mo\.?)\s*\d+", val) or re.match(r"(yr|year)\s*\d+\s*(mo|month)", val):
                if start_col is None:
                    start_col = col_idx
                month_cols.append(col_idx)

        if len(month_cols) >= 6 and start_col is not None:
            # Generate monthly dates starting from a reference
            base = date(2026, 1, 31)
            dates = []
            for i in range(len(month_cols)):
                m = (base.month + i - 1) % 12 + 1
                y = base.year + (base.month + i - 1) // 12
                dates.append(_month_end(y, m))
            return dates, row_idx, start_col

    return [], -1, 0


def _try_parse_date(val) -> Optional[date]:
    """Try to parse a value as a date."""
    if isinstance(val, date):
        return val
    if isinstance(val, pd.Timestamp):
        return val.date()
    if isinstance(val, str):
        val = val.strip()
        if not val or val.lower() in ("nan", "none", ""):
            return None
        try:
            return pd.to_datetime(val).date()
        except Exception:
            return None
    return None


def _month_end(year: int, month: int) -> date:
    """Get month-end date."""
    import calendar
    _, last_day = calendar.monthrange(year, month)
    return date(year, month, last_day)


def _is_summary_row(label: str) -> bool:
    """Check if a row label indicates a summary/total row."""
    lower = label.lower().strip()
    return lower in (
        "total", "total revenue", "total expenses", "total expense",
        "net operating income", "noi", "effective gross income",
        "effective gross revenue", "gross revenue",
        "total operating expenses", "cash flow before debt service",
        "cash flow after debt service", "net cash flow",
        "debt service", "total debt service",
        "total capital costs", "total leasing costs",
    )


def _is_occupancy_row(label: str) -> bool:
    """Check if a row is an occupancy metric."""
    lower = label.lower().strip()
    return any(kw in lower for kw in ("occupancy", "occupied %", "physical occ"))


def _to_float(val) -> float:
    """Convert a value to float, returning 0.0 on failure."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0.0
    try:
        # Handle string values with commas, parens (negative), %
        s = str(val).strip()
        if not s or s.lower() in ("nan", "none", "-", ""):
            return 0.0
        # Remove commas
        s = s.replace(",", "")
        # Handle parenthetical negatives: (1,234) -> -1234
        if s.startswith("(") and s.endswith(")"):
            s = "-" + s[1:-1]
        # Handle percentage
        if s.endswith("%"):
            return float(s[:-1]) / 100.0
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def _safe_str(val) -> str:
    """Convert to string, returning '' for NaN/None."""
    if val is None:
        return ""
    if isinstance(val, float) and np.isnan(val):
        return ""
    return str(val).strip()


def _safe_date_str(val) -> Optional[str]:
    """Convert to ISO date string, or None."""
    if val is None:
        return None
    parsed = _try_parse_date(val)
    if parsed:
        return parsed.isoformat()
    s = _safe_str(val)
    return s if s else None


def _date_to_str(d) -> str:
    """Convert date to ISO string."""
    if isinstance(d, date):
        return d.isoformat()
    return str(d)


def _is_vacant_tenant(name: str) -> bool:
    """Check if tenant name indicates a vacant unit."""
    lower = name.lower().strip()
    return lower in ("vacant", "vacancy", "available", "dark", "empty") or lower.startswith("vacant")


def _detect_rent_roll_columns(cols: List[str]) -> Dict[str, str]:
    """Map normalized column names to semantic fields for rent roll parsing."""
    mapping = {}
    patterns = {
        "tenant_name": ["tenant", "tenant name", "lessee", "occupant"],
        "suite": ["suite", "unit", "space", "suite/unit"],
        "square_feet": ["sf", "sq ft", "square feet", "sqft", "area", "gla", "rsf"],
        "lease_type": ["lease type", "type"],
        "lease_start": ["lease start", "start date", "commence", "commencement"],
        "lease_end": ["lease end", "end date", "expiration", "expiry"],
        "base_rent_annual": ["annual rent", "base rent", "annual base rent", "rent annual"],
        "base_rent_psf": ["rent psf", "rent/sf", "base rent psf", "$/sf"],
        "recovery_type": ["recovery", "recovery type", "reimburse type", "expense type"],
        "ret_recovery_psf": ["ret recovery", "tax recovery", "re tax"],
        "ins_recovery_psf": ["ins recovery", "insurance recovery"],
        "cam_recovery_psf": ["cam recovery", "cam reimburse"],
        "ti_psf": ["ti", "tenant improvement", "ti psf", "ti/sf"],
        "lc_psf": ["lc", "leasing commission", "lc psf", "lc/sf"],
        "renewal_probability": ["renewal prob", "renewal %", "renew prob"],
        "cpi_pct": ["cpi", "cpi %", "cpi pct"],
        "free_rent_months": ["free rent", "free months", "concession months"],
        "pct_rent_breakpoint": ["breakpoint", "pct rent breakpoint"],
        "pct_rent_rate": ["pct rent", "percentage rent", "overage"],
        "security_deposit": ["security deposit", "deposit"],
    }

    for field, keywords in patterns.items():
        for col in cols:
            for kw in keywords:
                if kw in col:
                    mapping[field] = col
                    break
            if field in mapping:
                break

    return mapping


def _detect_market_profile_columns(cols: List[str]) -> Dict[str, str]:
    """Map normalized column names to semantic fields for market profiles."""
    mapping = {}
    patterns = {
        "profile_name": ["profile", "name", "market", "space type", "use type"],
        "base_rent_psf": ["rent psf", "base rent", "market rent", "$/sf"],
        "term_months": ["term", "lease term", "months"],
        "renewal_probability": ["renewal prob", "renewal %", "renew"],
        "vacancy_months": ["vacancy months", "downtime", "vacancy"],
        "ti_new_psf": ["ti new", "new ti", "ti - new"],
        "ti_renewal_psf": ["ti renewal", "renewal ti", "ti - renewal"],
        "lc_new_pct": ["lc new", "new lc", "lc - new"],
        "lc_renewal_pct": ["lc renewal", "renewal lc", "lc - renewal"],
        "fixed_step_pct": ["step", "escalation", "increase", "bump"],
        "cpi_pct": ["cpi", "cpi %", "inflation"],
        "recovery_type": ["recovery", "reimburse", "expense type"],
    }

    for field, keywords in patterns.items():
        for col in cols:
            for kw in keywords:
                if kw in col:
                    mapping[field] = col
                    break
            if field in mapping:
                break

    return mapping


def _empty_cashflow_result(filename: str, error: str = None) -> Dict[str, Any]:
    """Return empty result for failed cashflow parse."""
    return {
        "periods": [],
        "line_items": [],
        "occupancy": None,
        "metadata": {
            "filename": filename,
            "total_periods": 0,
            "total_line_items": 0,
            "mapped_count": 0,
            "unmapped_items": [],
            "error": error,
        },
    }
