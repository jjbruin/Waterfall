"""Surveillance Service — portfolio monitoring with covenant tracking.

Reads from Waterfall's existing data sources (inv, occ, isbs_raw, loans)
and a thin editable table (surveillance_properties) for manual fields.
Computes live metrics (TTM NOI, DSCR, debt balance, loan maturity)
consistently with Property Financials and Deal Analysis tabs.
"""

import logging
from datetime import date, datetime, timedelta

import pandas as pd
from sqlalchemy import text

from config import IS_ACCOUNTS, DEBT_BS_ACCTS
from compute import get_isbs_debt_balance
from loans import build_loans_from_mri_loans
from utils import normalize_columns
from flask_app.db import get_engine
from flask_app.services import data_service
from flask_app.services.dashboard_service import (
    get_latest_occupancy as _dashboard_occ,
    get_child_vcodes,
    get_cached_caps_and_occ,
    get_portfolio_kpis,
)
from flask_app.services.financials_service import (
    _prepare_isbs,
    _get_cumulative_balances,
    _add_balances,
    _subtract_balances,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------

_SURVEILLANCE_DDL = [
    """
    CREATE TABLE IF NOT EXISTS surveillance_properties (
        vcode               TEXT PRIMARY KEY,
        dscr_val            DOUBLE PRECISION,
        dscr_min            DOUBLE PRECISION,
        dy_val              DOUBLE PRECISION,
        dy_min              DOUBLE PRECISION,
        ltv_val             DOUBLE PRECISION,
        ltv_min             DOUBLE PRECISION,
        working_capital     DOUBLE PRECISION,
        tax_due             TEXT,
        tax_status          TEXT,
        ins_renewal         TEXT,
        tenant_exp          TEXT,
        comments            TEXT,
        ground_lease_exp    TEXT,
        ground_lease_rent   DOUBLE PRECISION,
        ground_lease_status TEXT,
        escrow_tax          TEXT,
        escrow_insurance    TEXT,
        escrow_capex        TEXT,
        collateral_type     TEXT,
        collateral_value    DOUBLE PRECISION,
        collateral_notes    TEXT,
        updated_at          TIMESTAMP,
        updated_by          TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS insurance (
        id                  {pk_type},
        vcode               TEXT NOT NULL,
        ins_type            TEXT NOT NULL,
        carrier             TEXT,
        policy_number       TEXT,
        coverage_amount     DOUBLE PRECISION,
        expiration_date     TEXT,
        notes               TEXT,
        created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at          TIMESTAMP,
        UNIQUE(vcode, ins_type)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS surveillance_comments (
        id                  {pk_type},
        vcode               TEXT NOT NULL,
        comment_date        TEXT NOT NULL,
        comment_text        TEXT NOT NULL,
        created_by          TEXT,
        created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
]


def ensure_tables(engine=None):
    """Create surveillance tables if they don't exist."""
    if engine is None:
        engine = get_engine()

    is_pg = "postgresql" in str(engine.url)
    pk_type = "SERIAL PRIMARY KEY" if is_pg else "INTEGER PRIMARY KEY AUTOINCREMENT"

    with engine.begin() as conn:
        for ddl in _SURVEILLANCE_DDL:
            conn.execute(text(ddl.replace("{pk_type}", pk_type)))

        # Migration: add columns if missing
        for col_def in [
            "tax_status TEXT",
            "ground_lease_exp TEXT",
            "ground_lease_rent DOUBLE PRECISION",
            "ground_lease_status TEXT",
            "escrow_tax TEXT",
            "escrow_insurance TEXT",
            "escrow_capex TEXT",
            "collateral_type TEXT",
            "collateral_value DOUBLE PRECISION",
            "collateral_notes TEXT",
        ]:
            try:
                conn.execute(text(
                    f"ALTER TABLE surveillance_properties ADD COLUMN {col_def}"
                ))
            except Exception:
                pass  # column already exists


# ---------------------------------------------------------------------------
# Live metric helpers — consistent with Property Financials / Deal Analysis
# ---------------------------------------------------------------------------

def _compute_ttm_noi_and_dscr(isbs_raw, vcode):
    """Compute TTM NOI and DSCR for a deal, same formula as Property Financials.

    Uses ISBS Interim IS (actuals), YTD cumulative → TTM conversion.
    DSCR = NOI / abs(Debt Service).
    Returns dict with noi, dscr, revenue, expenses, debt_service, period.
    """
    isbs = _prepare_isbs(isbs_raw, vcode)
    if isbs.empty:
        return {}

    actual_data = isbs[isbs['vSource'] == 'Interim IS']
    if actual_data.empty:
        return {}

    actual_periods = sorted(actual_data['dtEntry_parsed'].dropna().unique())
    if not actual_periods:
        return {}

    ref_date = pd.Timestamp(actual_periods[-1])

    # TTM calculation — same as _calculate_is_amounts(TTM, Actual, ...)
    current_bal = _get_cumulative_balances(actual_data, ref_date, IS_ACCOUNTS)
    if not current_bal:
        return {}

    dec_prior = next(
        (pd.Timestamp(p) for p in actual_periods
         if pd.Timestamp(p).year == ref_date.year - 1 and pd.Timestamp(p).month == 12),
        None
    )
    same_month_ly = next(
        (pd.Timestamp(p) for p in actual_periods
         if pd.Timestamp(p).year == ref_date.year - 1 and pd.Timestamp(p).month == ref_date.month),
        None
    )

    if dec_prior and same_month_ly:
        dec_bal = _get_cumulative_balances(actual_data, dec_prior, IS_ACCOUNTS)
        ly_bal = _get_cumulative_balances(actual_data, same_month_ly, IS_ACCOUNTS)
        amounts = _subtract_balances(
            _add_balances(current_bal, dec_bal, IS_ACCOUNTS), ly_bal, IS_ACCOUNTS
        )
    elif dec_prior:
        dec_bal = _get_cumulative_balances(actual_data, dec_prior, IS_ACCOUNTS)
        amounts = _add_balances(current_bal, dec_bal, IS_ACCOUNTS)
    else:
        amounts = current_bal

    # Revenue (stored as credits/negative) → display positive
    rev_total = sum(-v for v in amounts.get('REVENUES', {}).values())
    exp_total = sum(v for v in amounts.get('EXPENSES', {}).values())
    noi = rev_total - exp_total

    # Debt service from IS_ACCOUNTS['DEBT_SERVICE']
    ds_total = sum(v for v in amounts.get('DEBT_SERVICE', {}).values())
    dscr = noi / abs(ds_total) if ds_total != 0 else None

    # Real estate taxes — account 5090 within EXPENSES
    expenses_detail = amounts.get('EXPENSES', {})
    re_tax = expenses_detail.get('Real Estate Taxes', 0.0)

    # Insurance expense — accounts 5110, 5114
    ins_exp = expenses_detail.get('Property & Liability Insurance', 0.0)

    return {
        "noi": round(noi, 2),
        "revenue": round(rev_total, 2),
        "expenses": round(exp_total, 2),
        "debt_service": round(ds_total, 2),
        "dscr": round(dscr, 2) if dscr is not None else None,
        "re_tax_ttm": round(abs(re_tax), 2) if re_tax else None,
        "ins_exp_ttm": round(abs(ins_exp), 2) if ins_exp else None,
        "period": ref_date.strftime("%Y-%m"),
    }


def _compute_debt_balances(isbs_raw, vcodes):
    """Compute ISBS debt balance for multiple vcodes at once.

    Returns dict {vcode_lower: balance}.
    """
    result = {}
    for vc in vcodes:
        bal = get_isbs_debt_balance(isbs_raw, vc)
        if bal is not None:
            result[vc.lower()] = round(bal, 2)
    return result


def _compute_loan_maturities(mri_loans_raw):
    """Get loan maturity dates per deal from MRI_Loans, same as Deal Analysis.

    Uses build_loans_from_mri_loans() for consistent parsing.
    Returns dict {vcode_lower: {maturity_date, loan_rate, loan_type, loan_balance}}.
    """
    if mri_loans_raw is None or mri_loans_raw.empty:
        return {}

    try:
        loans = build_loans_from_mri_loans(mri_loans_raw)
    except Exception:
        logger.exception("Failed to build loans from MRI_Loans")
        return {}

    result = {}
    for loan in loans:
        vc = str(loan.vcode).strip().lower()
        if vc not in result:
            result[vc] = {
                "maturity_date": None,
                "loan_rate": None,
                "loan_type": None,
                "loan_balance": 0.0,
            }
        entry = result[vc]
        entry["loan_balance"] += loan.orig_amount or 0
        # Use the latest maturity date across all loans for the deal
        if loan.maturity_date:
            mat_str = loan.maturity_date.isoformat()
            if entry["maturity_date"] is None or mat_str > entry["maturity_date"]:
                entry["maturity_date"] = mat_str
                entry["loan_rate"] = loan.fixed_rate
                entry["loan_type"] = loan.int_type

    return result


def _get_property_values(mri_val):
    """Get most recent property value per deal from valuations.

    Returns dict {vcode_lower: value}.
    """
    if mri_val is None or mri_val.empty:
        return {}

    df = mri_val.copy()
    col_map = {c.lower(): c for c in df.columns}
    vc_col = col_map.get("vcode")
    val_col = col_map.get("mincomecapconcludedvalue")
    dt_col = col_map.get("dtvaluation")

    if not vc_col or not val_col:
        return {}

    df["_vc"] = df[vc_col].astype(str).str.strip().str.lower()
    df["_val"] = pd.to_numeric(df[val_col].astype(str).str.replace(",", ""), errors="coerce")
    if dt_col:
        df["_dt"] = pd.to_datetime(df[dt_col], errors="coerce")
        df = df.dropna(subset=["_dt"]).sort_values("_dt", ascending=False)

    result = {}
    for vc, grp in df.groupby("_vc"):
        val = grp["_val"].dropna()
        if not val.empty:
            result[vc] = float(val.iloc[0])
    return result


def _compute_loan_covenants(mri_loans_raw):
    """Extract debt covenant requirements per deal from MRI_Loans.

    For deals with multiple loans, uses the most restrictive (highest
    minimum) covenant across all loans.  Tracks both general requirements
    and extension-specific requirements separately.

    Returns dict {vcode_lower: {
        dscr_min, dscr_ext,
        dy_min, dy_ext,
        ltv_max, ltv_ext,
        extension_options,
    }}.
    """
    if mri_loans_raw is None or mri_loans_raw.empty:
        return {}

    df = mri_loans_raw.copy()
    col_map = {c.lower(): c for c in df.columns}

    # Column name mapping — case-insensitive lookup
    vc_col = col_map.get("vcode", col_map.get("pcode"))
    dscr_min_col = col_map.get("nrequireddcr")
    dscr_ext_col = col_map.get("nreqdsr")
    dy_min_col = col_map.get("ndy")
    dy_ext_col = col_map.get("nrequireddy")
    ltv_max_col = col_map.get("nltv")
    ltv_ext_col = col_map.get("nrequiredltv")
    ext_options_col = col_map.get("extensionoptions", col_map.get("vamortamt"))

    if not vc_col:
        return {}

    # Filter to maturity rows only (avoid duplicates from multiple Loan_Date rows)
    vdt_col = col_map.get("vdatetype")
    if vdt_col:
        mat_df = df[df[vdt_col].fillna("").str.lower() == "maturity"]
        if not mat_df.empty:
            df = mat_df

    def _safe_float(val):
        try:
            v = float(val)
            if pd.isna(v):
                return None
            return v
        except (TypeError, ValueError):
            return None

    result = {}
    for _, row in df.iterrows():
        vc = str(row.get(vc_col, "")).strip().lower()
        if not vc:
            continue

        entry = result.get(vc, {
            "dscr_min": None, "dscr_ext": None,
            "dy_min": None, "dy_ext": None,
            "ltv_max": None, "ltv_ext": None,
            "extension_options": None,
        })

        # Most restrictive = highest minimum DSCR/DY, lowest max LTV
        def _update_max(cur, new):
            """Take the more restrictive (higher) value."""
            if new is None:
                return cur
            if cur is None:
                return new
            return max(cur, new)

        def _update_min(cur, new):
            """Take the more restrictive (lower) value for LTV."""
            if new is None:
                return cur
            if cur is None:
                return new
            return min(cur, new)

        if dscr_min_col:
            entry["dscr_min"] = _update_max(entry["dscr_min"], _safe_float(row.get(dscr_min_col)))
        if dscr_ext_col:
            entry["dscr_ext"] = _update_max(entry["dscr_ext"], _safe_float(row.get(dscr_ext_col)))
        if dy_min_col:
            entry["dy_min"] = _update_max(entry["dy_min"], _safe_float(row.get(dy_min_col)))
        if dy_ext_col:
            entry["dy_ext"] = _update_max(entry["dy_ext"], _safe_float(row.get(dy_ext_col)))
        if ltv_max_col:
            entry["ltv_max"] = _update_min(entry["ltv_max"], _safe_float(row.get(ltv_max_col)))
        if ltv_ext_col:
            entry["ltv_ext"] = _update_min(entry["ltv_ext"], _safe_float(row.get(ltv_ext_col)))
        if ext_options_col:
            opt = row.get(ext_options_col)
            if opt and not pd.isna(opt) and str(opt).strip() and str(opt).strip().upper() not in ("NA", "NAN"):
                entry["extension_options"] = str(opt).strip()

        result[vc] = entry

    return result


# ---------------------------------------------------------------------------
# Reporting completeness — trailing 12 months
# ---------------------------------------------------------------------------

def _get_due_months(as_of: date = None) -> list[tuple[int, int]]:
    """Return the 12 most recent (year, month) pairs whose reports are due.

    Reports are due 30 days after month end, so a month is 'due' if
    today >= month_end + 30 days.  E.g. on Jul 14 2026, May 2026 is due
    (due Jun 30) but Jun 2026 is not yet (due Jul 31).
    """
    if as_of is None:
        as_of = date.today()
    months = []
    yr, mo = as_of.year, as_of.month
    for _ in range(14):  # check up to 14 to ensure we get 12
        # month end of (yr, mo)
        if mo == 12:
            me = date(yr, 12, 31)
        else:
            me = date(yr, mo + 1, 1) - timedelta(days=1)
        due_by = me + timedelta(days=30)
        if as_of >= due_by:
            months.append((yr, mo))
        if len(months) == 12:
            break
        mo -= 1
        if mo == 0:
            mo = 12
            yr -= 1
    return months


def _build_reported_months_map(df, vcode_col, date_col):
    """Build {vcode_lower: set((year, month), ...)} from a DataFrame."""
    if df is None or df.empty:
        return {}
    result = {}
    for vc, grp in df.groupby(vcode_col):
        dates = pd.to_datetime(grp[date_col], errors="coerce").dropna()
        result[str(vc).strip().lower()] = {(d.year, d.month) for d in dates}
    return result


def _compute_reporting_completeness(occ, isbs_raw, tenants_raw, all_vcodes, is_commercial_set):
    """Compute reporting completeness for all deals.

    Returns dict {vcode_lower: {
        occ_latest, occ_missing,
        rent_roll_latest, rent_roll_missing,  (commercial only, None for residential)
        is_latest, is_missing,
        bs_latest, bs_missing,
    }}
    """
    due_months = _get_due_months()
    due_set = set(due_months)

    # --- Occupancy reported months ---
    occ_months = {}
    if occ is not None and not occ.empty:
        col_map = {c.lower(): c for c in occ.columns}
        vc_col = col_map.get("vcode", col_map.get("propcode", "vcode"))
        dt_col = col_map.get("dtreported", col_map.get("period", "dtReported"))
        df = occ.copy()
        df["_vc"] = df[vc_col].astype(str).str.strip().str.lower()
        df["_dt"] = pd.to_datetime(df[dt_col], errors="coerce")
        df = df.dropna(subset=["_dt"])
        occ_months = _build_reported_months_map(df, "_vc", "_dt")

    # --- Income Statement (Interim IS) reported months ---
    is_months = {}
    bs_months = {}
    if isbs_raw is not None and not isbs_raw.empty:
        if 'vSource' in isbs_raw.columns and 'dtEntry_parsed' in isbs_raw.columns:
            is_df = isbs_raw[isbs_raw['vSource'] == 'Interim IS']
            if not is_df.empty and 'vcode' in is_df.columns:
                is_months = _build_reported_months_map(is_df, 'vcode', 'dtEntry_parsed')

            bs_df = isbs_raw[isbs_raw['vSource'] == 'Interim BS']
            if not bs_df.empty and 'vcode' in bs_df.columns:
                bs_months = _build_reported_months_map(bs_df, 'vcode', 'dtEntry_parsed')

    # --- Rent Roll (tenants) reported months ---
    rr_months = {}
    if tenants_raw is not None and not tenants_raw.empty:
        t = tenants_raw.copy()
        normalize_columns(t)
        col_map_t = {c.lower(): c for c in t.columns}
        vc_col_t = col_map_t.get("code", col_map_t.get("vcode", None))
        dt_col_t = col_map_t.get("occupancy date", col_map_t.get("dtreported", None))
        if vc_col_t and dt_col_t:
            t["_vc"] = t[vc_col_t].astype(str).str.strip().str.lower()
            t["_dt"] = pd.to_datetime(t[dt_col_t], errors="coerce")
            t = t.dropna(subset=["_dt"])
            rr_months = _build_reported_months_map(t, "_vc", "_dt")

    def _fmt_latest(reported_set):
        """Format the most recent reported month as 'M/YY'."""
        if not reported_set:
            return None
        latest = max(reported_set)
        return f"{latest[1]}/{str(latest[0])[2:]}"

    def _count_missing(reported_set):
        """Count how many of the trailing 12 due months are missing."""
        if not reported_set:
            return len(due_set)
        return len(due_set - reported_set)

    result = {}
    for vc in all_vcodes:
        vc_lower = vc.lower()
        occ_set = occ_months.get(vc_lower, set())
        is_set = is_months.get(vc_lower, set())
        bs_set = bs_months.get(vc_lower, set())
        is_comm = vc in is_commercial_set or vc_lower in is_commercial_set

        entry = {
            "occ_latest": _fmt_latest(occ_set),
            "occ_missing": _count_missing(occ_set),
            "is_latest": _fmt_latest(is_set),
            "is_missing": _count_missing(is_set),
            "bs_latest": _fmt_latest(bs_set),
            "bs_missing": _count_missing(bs_set),
        }

        if is_comm:
            rr_set = rr_months.get(vc_lower, set())
            entry["rent_roll_latest"] = _fmt_latest(rr_set)
            entry["rent_roll_missing"] = _count_missing(rr_set)
        else:
            entry["rent_roll_latest"] = None
            entry["rent_roll_missing"] = None

        result[vc_lower] = entry

    return result


# ---------------------------------------------------------------------------
# Surveillance table — main query
# ---------------------------------------------------------------------------

def get_surveillance_table() -> list[dict]:
    """Build the surveillance table by joining Waterfall data sources.

    Returns one row per active deal with live occupancy, TTM NOI, DSCR,
    ISBS debt balance, loan maturity, and editable surveillance fields.
    """
    data = data_service.get_data()
    inv = data.get("inv", pd.DataFrame())
    occ = data.get("occupancy_raw", pd.DataFrame())
    isbs_raw = data.get("isbs_raw", pd.DataFrame())
    mri_loans_raw = data.get("mri_loans_raw", pd.DataFrame())
    tenants_raw = data.get("tenants_raw", pd.DataFrame())

    if inv.empty:
        return []

    # --- Filter to active deals (exclude sold) ---
    inv_col = {c.lower(): c for c in inv.columns}
    sale_col = inv_col.get("sale_status", inv_col.get("salestatus"))
    lifecycle_col = inv_col.get("lifecycle")
    mask = pd.Series(True, index=inv.index)
    if sale_col and sale_col in inv.columns:
        mask &= inv[sale_col].fillna("").str.upper() != "SOLD"
    if lifecycle_col and lifecycle_col in inv.columns:
        mask &= inv[lifecycle_col].fillna("").str.lower() != "sold"
    active = inv[mask].copy()

    vcode_col = _find_col(active, ["vcode", "Vcode", "PropCode"])
    name_col = _find_col(active, ["Investment_Name", "Property_Name", "PropertyName", "Deal_Name"])
    type_col = _find_col(active, ["Asset_Type", "AssetType", "asset_type"])
    city_col = _find_col(active, ["City", "city"])
    units_col = _find_col(active, ["Units", "units", "iResidentialUnits"])
    partner_col = _find_col(active, ["Partner", "partner", "PSC_Asset_Manager"])
    lifecycle_disp = _find_col(active, ["Lifecycle", "lifecycle"])
    portfolio_col = _find_col(active, ["Portfolio_Name", "portfolio_name"])

    # --- Collect all vcodes ---
    all_vcodes = [str(deal.get(vcode_col, "")).strip()
                  for _, deal in active.iterrows() if str(deal.get(vcode_col, "")).strip()]

    # --- Determine commercial deals (have tenant data) ---
    is_commercial_set = set()
    if tenants_raw is not None and not tenants_raw.empty:
        t_cols = {c.lower(): c for c in tenants_raw.columns}
        t_vc_col = t_cols.get("code", t_cols.get("vcode"))
        if t_vc_col:
            is_commercial_set = set(tenants_raw[t_vc_col].astype(str).str.strip().str.lower().unique())

    # --- Latest occupancy per deal ---
    occ_latest = _latest_occupancy(occ)

    # --- TTM NOI + DSCR per deal (computed like Property Financials) ---
    noi_dscr_map = {}
    for vc in all_vcodes:
        try:
            noi_dscr_map[vc.lower()] = _compute_ttm_noi_and_dscr(isbs_raw, vc)
        except Exception:
            logger.exception("TTM NOI/DSCR failed for %s", vc)

    # --- Debt balance from ISBS (same as Deal Analysis) ---
    debt_map = _compute_debt_balances(isbs_raw, all_vcodes)

    # --- Loan maturity from MRI_Loans (same as Deal Analysis) ---
    loan_map = _compute_loan_maturities(mri_loans_raw)

    # --- Loan covenant requirements from MRI_Loans ---
    covenant_map = _compute_loan_covenants(mri_loans_raw)

    # --- Property values from valuations (most recent per deal) ---
    mri_val = data.get("mri_val", pd.DataFrame())
    value_map = _get_property_values(mri_val)

    # --- Load editable surveillance fields ---
    surv_fields = _load_surveillance_properties()

    # --- Load insurance ---
    insurance = _load_insurance_summary()

    # --- Load latest comments ---
    comments_map = _load_latest_comments()

    # --- Reporting completeness ---
    reporting_map = _compute_reporting_completeness(
        occ, isbs_raw, tenants_raw, all_vcodes, is_commercial_set
    )

    # --- Build result rows ---
    rows = []
    for _, deal in active.iterrows():
        vc = str(deal.get(vcode_col, "")).strip()
        if not vc:
            continue
        vc_lower = vc.lower()

        occ_row = occ_latest.get(vc_lower, {})
        noi_row = noi_dscr_map.get(vc_lower, {})
        loan_row = loan_map.get(vc_lower, {})
        cov = covenant_map.get(vc_lower, {})
        surv = surv_fields.get(vc, {})
        ins = insurance.get(vc, {})
        comment_entry = comments_map.get(vc, {})
        rpt = reporting_map.get(vc_lower, {})

        # Debt: prefer ISBS balance, fall back to MRI_Loans origination total
        debt_balance = debt_map.get(vc_lower)
        if debt_balance is None:
            debt_balance = loan_row.get("loan_balance")

        # Computed Debt Yield = TTM NOI / Debt Balance
        noi = noi_row.get("noi")
        dy_val = None
        if noi and debt_balance and debt_balance > 0:
            dy_val = round(noi / debt_balance, 4)

        # Computed LTV = Debt Balance / Property Value
        prop_value = value_map.get(vc_lower)
        ltv_val = None
        if debt_balance and prop_value and prop_value > 0:
            ltv_val = round(debt_balance / prop_value, 4)

        row = {
            "vcode": vc,
            "name": deal.get(name_col, ""),
            "asset_type": deal.get(type_col, ""),
            "city": deal.get(city_col, ""),
            "units": _safe_int(deal.get(units_col)),
            "partner": deal.get(partner_col, ""),
            "lifecycle": deal.get(lifecycle_disp, ""),
            "portfolio_name": deal.get(portfolio_col, ""),
            # Live data — consistent with Property Financials
            "occ_pct": occ_row.get("occ_pct"),
            "occ_period": occ_row.get("period"),
            "noi_ttm": noi_row.get("noi"),
            "revenue_ttm": noi_row.get("revenue"),
            "dscr": noi_row.get("dscr"),
            "fin_period": noi_row.get("period"),
            # Debt — consistent with Deal Analysis (ISBS balance)
            "debt_balance": debt_balance,
            # Loan data — consistent with Deal Analysis
            "loan_rate": loan_row.get("loan_rate"),
            "maturity_date": loan_row.get("maturity_date"),
            "loan_type": loan_row.get("loan_type"),
            # Debt Covenants — computed actuals
            "dy_val": dy_val,
            "ltv_val": ltv_val,
            "prop_value": prop_value,
            # Debt Covenants — requirements from MRI Loan table
            "dscr_min": cov.get("dscr_min"),
            "dscr_ext": cov.get("dscr_ext"),
            "dy_min": cov.get("dy_min"),
            "dy_ext": cov.get("dy_ext"),
            "ltv_max": cov.get("ltv_max"),
            "ltv_ext": cov.get("ltv_ext"),
            "extension_options": cov.get("extension_options"),
            # Real Estate Taxes — TTM from ISBS
            "re_tax_ttm": noi_row.get("re_tax_ttm"),
            "tax_due": surv.get("tax_due"),
            "tax_status": surv.get("tax_status"),
            # Insurance expense — TTM from ISBS
            "ins_exp_ttm": noi_row.get("ins_exp_ttm"),
            # Ground Leases
            "ground_lease_exp": surv.get("ground_lease_exp"),
            "ground_lease_rent": surv.get("ground_lease_rent"),
            "ground_lease_status": surv.get("ground_lease_status"),
            # Escrows
            "escrow_tax": surv.get("escrow_tax"),
            "escrow_insurance": surv.get("escrow_insurance"),
            "escrow_capex": surv.get("escrow_capex"),
            # Add'l Collateral
            "collateral_type": surv.get("collateral_type"),
            "collateral_value": surv.get("collateral_value"),
            "collateral_notes": surv.get("collateral_notes"),
            # Other surveillance fields
            "working_capital": surv.get("working_capital"),
            "ins_renewal": ins.get("nearest_expiration"),
            "tenant_exp": surv.get("tenant_exp"),
            "updated_at": surv.get("updated_at"),
            # Insurance
            "has_property_ins": ins.get("has_property", False),
            "has_gl_ins": ins.get("has_gl", False),
            "property_carrier": ins.get("property_carrier"),
            "property_expiration": ins.get("property_expiration"),
            "gl_carrier": ins.get("gl_carrier"),
            "gl_expiration": ins.get("gl_expiration"),
            # Comments (latest)
            "comment_text": comment_entry.get("comment_text"),
            "comment_date": comment_entry.get("comment_date"),
            "comment_id": comment_entry.get("id"),
            # Reporting completeness
            "rpt_occ_latest": rpt.get("occ_latest"),
            "rpt_occ_missing": rpt.get("occ_missing"),
            "rpt_rent_roll_latest": rpt.get("rent_roll_latest"),
            "rpt_rent_roll_missing": rpt.get("rent_roll_missing"),
            "rpt_is_latest": rpt.get("is_latest"),
            "rpt_is_missing": rpt.get("is_missing"),
            "rpt_bs_latest": rpt.get("bs_latest"),
            "rpt_bs_missing": rpt.get("bs_missing"),
            "is_commercial": vc_lower in is_commercial_set,
        }

        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Dashboard KPIs
# ---------------------------------------------------------------------------

def get_dashboard() -> dict:
    """Portfolio surveillance KPIs and chart data.

    Debt and occupancy are sourced from the same shared caps/occ cache
    used by the Dashboard tab, ensuring identical numbers.
    """
    rows = get_surveillance_table()
    if not rows:
        return {"total": 0}

    # Use the shared caps/occ (same source as Dashboard KPIs)
    caps, occ_map, _, _ = get_cached_caps_and_occ()
    kpis = get_portfolio_kpis(caps, occ_map)

    total = len(rows)
    total_noi = sum(r["noi_ttm"] or 0 for r in rows)

    # Maturing within 12 months
    today = date.today()
    mat_12 = 0
    for r in rows:
        md = r.get("maturity_date")
        if md:
            try:
                mat = pd.to_datetime(md).date()
                if mat <= today.replace(year=today.year + 1):
                    mat_12 += 1
            except Exception:
                pass

    # By asset type
    by_type = {}
    for r in rows:
        t = r.get("asset_type") or "Unknown"
        by_type[t] = by_type.get(t, 0) + 1

    return {
        "total": total,
        "total_debt": kpis.get("debt_outstanding"),
        "avg_occ": round(kpis["portfolio_occupancy"], 1) if kpis.get("portfolio_occupancy") else None,
        "total_noi_ttm": total_noi,
        "maturing_12mo": mat_12,
        "by_type": by_type,
    }


# ---------------------------------------------------------------------------
# CRUD — surveillance properties (editable fields)
# ---------------------------------------------------------------------------

def update_surveillance_property(vcode: str, fields: dict, username: str = None) -> dict:
    """Upsert editable surveillance fields for a deal."""
    allowed = {
        "dscr_min", "dy_val", "dy_min",
        "ltv_val", "ltv_min", "working_capital",
        "tax_due", "tax_status", "ins_renewal", "tenant_exp",
        "ground_lease_exp", "ground_lease_rent", "ground_lease_status",
        "escrow_tax", "escrow_insurance", "escrow_capex",
        "collateral_type", "collateral_value", "collateral_notes",
    }
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return {"status": "no_changes"}

    updates["updated_at"] = datetime.utcnow().isoformat()
    if username:
        updates["updated_by"] = username

    engine = get_engine()
    with engine.begin() as conn:
        existing = conn.execute(
            text("SELECT vcode FROM surveillance_properties WHERE vcode = :vc"),
            {"vc": vcode}
        ).fetchone()

        if existing:
            set_clause = ", ".join(f"{k} = :{k}" for k in updates)
            conn.execute(
                text(f"UPDATE surveillance_properties SET {set_clause} WHERE vcode = :vcode"),
                {**updates, "vcode": vcode}
            )
        else:
            updates["vcode"] = vcode
            cols = ", ".join(updates.keys())
            vals = ", ".join(f":{k}" for k in updates.keys())
            conn.execute(
                text(f"INSERT INTO surveillance_properties ({cols}) VALUES ({vals})"),
                updates
            )

    return {"status": "ok", "vcode": vcode}


# ---------------------------------------------------------------------------
# CRUD — surveillance comments (date-based)
# ---------------------------------------------------------------------------

def get_comments(vcode: str) -> list[dict]:
    """Get all comments for a deal, newest first."""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT id, vcode, comment_date, comment_text, created_by, created_at "
                     "FROM surveillance_comments WHERE vcode = :vc ORDER BY comment_date DESC, id DESC"),
                {"vc": vcode}
            ).mappings().all()
        return [dict(r) for r in rows]
    except Exception:
        return []


def save_comment(vcode: str, comment_date: str, comment_text: str, username: str = None) -> dict:
    """Save a comment for a deal on a specific date.

    If a comment already exists for the same vcode+date, it is updated.
    Otherwise a new record is created.
    """
    engine = get_engine()
    with engine.begin() as conn:
        existing = conn.execute(
            text("SELECT id FROM surveillance_comments "
                 "WHERE vcode = :vc AND comment_date = :cd"),
            {"vc": vcode, "cd": comment_date}
        ).fetchone()

        if existing:
            conn.execute(
                text("UPDATE surveillance_comments "
                     "SET comment_text = :ct, created_by = :cb "
                     "WHERE id = :id"),
                {"ct": comment_text, "cb": username, "id": existing[0]}
            )
            return {"status": "updated", "id": existing[0]}
        else:
            conn.execute(
                text("INSERT INTO surveillance_comments "
                     "(vcode, comment_date, comment_text, created_by) "
                     "VALUES (:vc, :cd, :ct, :cb)"),
                {"vc": vcode, "cd": comment_date, "ct": comment_text, "cb": username}
            )
            return {"status": "created"}


def delete_comment(comment_id: int) -> dict:
    """Delete a comment by ID."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM surveillance_comments WHERE id = :id"), {"id": comment_id})
    return {"status": "deleted"}


# ---------------------------------------------------------------------------
# CRUD — insurance
# ---------------------------------------------------------------------------

def get_insurance_list() -> list[dict]:
    """All insurance records with days-to-expiration."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT * FROM insurance ORDER BY vcode, ins_type"
        )).mappings().all()

    today = date.today()
    result = []
    for r in rows:
        d = dict(r)
        exp = d.get("expiration_date")
        if exp:
            try:
                exp_date = pd.to_datetime(exp).date()
                d["days_to_expiration"] = (exp_date - today).days
            except Exception:
                d["days_to_expiration"] = None
        else:
            d["days_to_expiration"] = None
        result.append(d)
    return result


def upsert_insurance(vcode: str, ins_type: str, fields: dict) -> dict:
    """Create or update an insurance record."""
    allowed = {"carrier", "policy_number", "coverage_amount", "expiration_date", "notes"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    updates["updated_at"] = datetime.utcnow().isoformat()

    engine = get_engine()
    with engine.begin() as conn:
        existing = conn.execute(
            text("SELECT id FROM insurance WHERE vcode = :vc AND ins_type = :it"),
            {"vc": vcode, "it": ins_type}
        ).fetchone()

        if existing:
            set_clause = ", ".join(f"{k} = :{k}" for k in updates)
            conn.execute(
                text(f"UPDATE insurance SET {set_clause} WHERE vcode = :vc AND ins_type = :it"),
                {**updates, "vc": vcode, "it": ins_type}
            )
            return {"status": "updated", "id": existing[0]}
        else:
            updates["vcode"] = vcode
            updates["ins_type"] = ins_type
            cols = ", ".join(updates.keys())
            vals = ", ".join(f":{k}" for k in updates.keys())
            conn.execute(
                text(f"INSERT INTO insurance ({cols}) VALUES ({vals})"),
                updates
            )
            return {"status": "created"}


def delete_insurance(ins_id: int) -> dict:
    """Delete an insurance record by ID."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM insurance WHERE id = :id"), {"id": ins_id})
    return {"status": "deleted"}


# ---------------------------------------------------------------------------
# Helpers — data extraction from Waterfall sources
# ---------------------------------------------------------------------------

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    """Find the first matching column name (case-insensitive)."""
    col_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in col_map:
            return col_map[c.lower()]
    return candidates[0]  # fallback


def _safe_int(val):
    """Convert to int, return None on failure."""
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return None


def _latest_occupancy(occ: pd.DataFrame) -> dict:
    """Get latest occupancy per deal from occ DataFrame."""
    if occ.empty:
        return {}

    col_map = {c.lower(): c for c in occ.columns}
    vcode_col = col_map.get("vcode", col_map.get("propcode", "vcode"))
    period_col = col_map.get("dtreported", col_map.get("period", "dtReported"))
    occ_col = col_map.get("occ%", col_map.get("occ_pct", col_map.get("focc", "Occ%")))

    df = occ.copy()
    df["_vcode"] = df[vcode_col].astype(str).str.strip().str.lower()
    df["_period"] = pd.to_datetime(df[period_col], errors="coerce")
    df = df.dropna(subset=["_period"])

    # Latest period per deal
    idx = df.groupby("_vcode")["_period"].idxmax()
    latest = df.loc[idx]

    result = {}
    for _, row in latest.iterrows():
        try:
            occ_val = float(row[occ_col])
        except (TypeError, ValueError):
            occ_val = None
        result[row["_vcode"]] = {
            "occ_pct": round(occ_val, 1) if occ_val is not None else None,
            "period": row["_period"].strftime("%Y-%m"),
        }
    return result


def _load_surveillance_properties() -> dict:
    """Load all editable surveillance fields keyed by vcode."""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT * FROM surveillance_properties")).mappings().all()
        return {r["vcode"]: dict(r) for r in rows}
    except Exception:
        return {}


def _load_latest_comments() -> dict:
    """Load the most recent comment per vcode."""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            # Get the max comment_date per vcode, then fetch the row
            rows = conn.execute(text(
                "SELECT sc.id, sc.vcode, sc.comment_date, sc.comment_text, sc.created_by "
                "FROM surveillance_comments sc "
                "INNER JOIN ("
                "  SELECT vcode, MAX(comment_date) AS max_date "
                "  FROM surveillance_comments GROUP BY vcode"
                ") latest ON sc.vcode = latest.vcode AND sc.comment_date = latest.max_date"
            )).mappings().all()
        return {r["vcode"]: dict(r) for r in rows}
    except Exception:
        return {}


def _load_insurance_summary() -> dict:
    """Load insurance summary per vcode.

    Returns dict {vcode: {
        has_property, has_gl, nearest_expiration,
        property_carrier, property_expiration,
        gl_carrier, gl_expiration,
    }}.
    """
    engine = get_engine()
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT * FROM insurance")).mappings().all()
    except Exception:
        return {}

    result = {}
    for r in rows:
        vc = r["vcode"]
        if vc not in result:
            result[vc] = {
                "has_property": False, "has_gl": False,
                "nearest_expiration": None,
                "property_carrier": None, "property_expiration": None,
                "gl_carrier": None, "gl_expiration": None,
            }
        entry = result[vc]
        exp = r.get("expiration_date")
        if r["ins_type"] == "Property":
            entry["has_property"] = True
            entry["property_carrier"] = r.get("carrier")
            entry["property_expiration"] = exp
        elif r["ins_type"] == "General Liability":
            entry["has_gl"] = True
            entry["gl_carrier"] = r.get("carrier")
            entry["gl_expiration"] = exp
        if exp:
            if entry["nearest_expiration"] is None or exp < entry["nearest_expiration"]:
                entry["nearest_expiration"] = exp
    return result
