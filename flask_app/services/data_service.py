"""Data loading service — replaces _load_sqlite_data() from app.py.

Loads all tables via data adapters (database or API), caches in module-level dict.
No Streamlit dependency.
"""

import pandas as pd
from typing import Optional

import logging
from loaders import load_coa, load_forecast
from flask_app.services.data_adapters import get_adapter
from utils import normalize_columns

logger = logging.getLogger(__name__)

# Module-level cache — cleared by reload()
_cache: dict = {}


# ISBS split table names and their vSource values
# Both isbs_interim_is and isbs_interim_is_historical map to 'Interim IS'
_ISBS_SPLIT = {
    'isbs_interim_is': 'Interim IS',
    'isbs_interim_is_historical': 'Interim IS',
    'isbs_interim_bs': 'Interim BS',
    'isbs_budget_is': 'Budget IS',
    'isbs_projected_is': 'Projected IS',
    'isbs_valuation_is': 'Valuation IS',
}


def _filter_paid_off_loans(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude loans with vDateType = 'Paid Off' (case-insensitive)."""
    if df.empty:
        return df
    col = next((c for c in df.columns if c.lower() == "vdatetype"), None)
    if col:
        df = df[df[col].astype(str).str.strip().str.lower() != "paid off"].reset_index(drop=True)
    return df


def _enrich_acquisition_dates(inv: pd.DataFrame, acct: pd.DataFrame) -> None:
    """Derive Acquisition_Date from earliest accounting activity per deal.

    Overwrites inv['Acquisition_Date'] in-place. Falls back to existing value
    if a deal has no accounting activity.
    """
    if acct is None or acct.empty:
        return
    if "EffectiveDate" not in acct.columns or "InvestmentID" not in acct.columns:
        return
    if "InvestmentID" not in inv.columns:
        return
    acct_dates = acct[["InvestmentID", "EffectiveDate"]].copy()
    acct_dates["_dt"] = pd.to_datetime(acct_dates["EffectiveDate"], errors="coerce")
    earliest = acct_dates.dropna(subset=["_dt"]).groupby("InvestmentID")["_dt"].min()
    inv["Acquisition_Date"] = (
        inv["InvestmentID"].astype(str).str.strip().str.upper()
        .map(earliest).fillna(inv.get("Acquisition_Date"))
    )


def _normalize_waterfall_df(wf: pd.DataFrame) -> pd.DataFrame:
    """Fully normalize waterfall DataFrame at load time.

    Performs all normalization that load_waterfalls() in loaders.py would do:
    column stripping, vCode rename, string stripping, numeric conversions,
    nPercent_dec, date parsing, and sorting. This allows load_waterfalls()
    to short-circuit when it receives an already-normalized DataFrame.
    """
    if wf.empty:
        return wf
    normalize_columns(wf)
    if "vCode" in wf.columns and "vcode" not in wf.columns:
        wf = wf.rename(columns={"vCode": "vcode"})
    if "vcode" not in wf.columns:
        return wf
    for col in ("vcode", "vmisc", "PropCode", "vState"):
        if col in wf.columns:
            wf[col] = wf[col].astype(str).str.strip()
    # Numeric conversions (match loaders.py load_waterfalls)
    if "iOrder" in wf.columns:
        wf["iOrder"] = pd.to_numeric(wf["iOrder"], errors="coerce").fillna(9999).astype(int)
    if "FXRate" in wf.columns:
        wf["FXRate"] = pd.to_numeric(wf["FXRate"], errors="coerce").fillna(0.0).astype(float)
    if "nPercent" in wf.columns:
        import numpy as np
        p = pd.to_numeric(wf["nPercent"], errors="coerce").fillna(0.0).astype(float)
        wf["nPercent_dec"] = np.where(p > 1.0, p / 100.0, p)
    if "mAmount" in wf.columns:
        wf["mAmount"] = pd.to_numeric(wf["mAmount"], errors="coerce").fillna(0.0).astype(float)
    if "dteffective" in wf.columns:
        wf["dteffective"] = pd.to_datetime(wf["dteffective"], errors="coerce").dt.date
    wf = wf.sort_values(["vcode", "vmisc", "iOrder"]).reset_index(drop=True)
    return wf


def _normalize_accounting(acct: pd.DataFrame) -> pd.DataFrame:
    """Normalize accounting feed once at load time.

    Performs the same normalization as loaders.normalize_accounting_feed():
    column stripping, type coercion, MajorType filtering, Amt parsing,
    contribution/distribution classification. Consumers can use the result
    directly without calling normalize_accounting_feed() again.
    """
    if acct is None or acct.empty:
        return acct if acct is not None else pd.DataFrame()
    from loaders import normalize_accounting_feed
    return normalize_accounting_feed(acct)


def _assemble_forecasts(fc_raw: pd.DataFrame, isbs_raw: pd.DataFrame,
                        pro_yr_base: int) -> pd.DataFrame:
    """Assemble forecast data from multiple sources with priority:

    1. forecast_feed CSV (``forecasts`` table) — admin-uploaded overrides
    2. ISBS ``Valuation IS`` — annual valuation cash flow projections (periodic monthly)
    3. ISBS ``Projected IS`` — underwriting projections (YTD cumulative, converted to periodic)

    Deals present in the forecast_feed take full priority; ISBS sources fill
    in only for deals NOT already covered by the forecast_feed.
    """
    parts = []
    covered_vcodes: set[str] = set()

    # --- Priority 1: forecast_feed (the ``forecasts`` table) ---
    if fc_raw is not None and not fc_raw.empty:
        fc = fc_raw.copy()
        normalize_columns(fc)
        vc_col = "Vcode" if "Vcode" in fc.columns else (
            "vcode" if "vcode" in fc.columns else None)
        if vc_col:
            covered_vcodes = set(fc[vc_col].astype(str).str.strip().str.lower())
        parts.append(fc_raw)
        logger.info(
            "Forecast assembly: %d deals from forecast_feed", len(covered_vcodes))

    if isbs_raw is None or isbs_raw.empty:
        if parts:
            return pd.concat(parts, ignore_index=True)
        return fc_raw if fc_raw is not None else pd.DataFrame()

    # Build vcode case map from ISBS (lowercase → original) so downstream
    # case-sensitive matching works.  ISBS is already normalised (lowercase).
    # We restore original case from the raw dtEntry column (not ideal but the
    # vcode is the only grouping key we have).  Fallback: uppercase first char.
    def _restore_case(vc_lower: str) -> str:
        return "P" + vc_lower[1:] if vc_lower.startswith("p") else vc_lower.upper()

    def _isbs_to_forecast(isbs_subset: pd.DataFrame, source_label: str,
                          cumulative: bool = False) -> pd.DataFrame:
        """Convert an ISBS subset to forecast-compatible DataFrame.

        For periodic data (Valuation IS): direct column mapping.
        For cumulative data (Projected IS): convert YTD → periodic first.
        """
        if isbs_subset.empty:
            return pd.DataFrame()

        df = isbs_subset[["vcode", "dtEntry", "vAccount", "mAmount"]].copy()

        if cumulative:
            # Parse dates for cum→periodic conversion
            dt_parsed = pd.to_datetime(
                df["dtEntry"], format="mixed", dayfirst=False, errors="coerce")
            df["_dt"] = dt_parsed
            df = df.dropna(subset=["_dt"])
            if df.empty:
                return pd.DataFrame()

            # Convert cumulative YTD → periodic per (vcode, vAccount)
            periodic_rows = []
            for (vc, acct), grp in df.groupby(["vcode", "vAccount"]):
                grp = grp.sort_values("_dt")
                dates = grp["_dt"].tolist()
                amounts = grp["mAmount"].tolist()
                dt_entries = grp["dtEntry"].tolist()
                for i, (dt, amt, dt_str) in enumerate(
                        zip(dates, amounts, dt_entries)):
                    if dt.month == 1:
                        periodic_amt = amt
                    else:
                        # Find prior same-year row
                        periodic_amt = amt
                        for j in range(i - 1, -1, -1):
                            if dates[j].year == dt.year:
                                periodic_amt = amt - amounts[j]
                                break
                    periodic_rows.append({
                        "vcode": vc, "dtEntry": dt_str,
                        "vAccount": acct, "mAmount": periodic_amt,
                    })
            df = pd.DataFrame(periodic_rows)
            if df.empty:
                return pd.DataFrame()

        # Parse dates for Pro_Yr computation
        dt_parsed = pd.to_datetime(
            df["dtEntry"], format="mixed", dayfirst=False, errors="coerce")
        df["_year"] = dt_parsed.dt.year

        # Build output in forecasts-table format
        out = pd.DataFrame({
            "Vcode": df["vcode"].apply(_restore_case),
            "Date": df["dtEntry"],
            "vAccount": df["vAccount"],
            "mAmount": df["mAmount"],
            "Pro_Yr": df["_year"] - pro_yr_base,
            "vSource": source_label,
        })
        return out

    # --- Priority 2: ISBS Valuation IS (periodic monthly) ---
    val_is = isbs_raw[isbs_raw["vSource"] == "Valuation IS"]
    if not val_is.empty:
        val_vcodes = set(val_is["vcode"].astype(str).str.strip().str.lower())
        new_vcodes = val_vcodes - covered_vcodes
        if new_vcodes:
            subset = val_is[val_is["vcode"].isin(new_vcodes)]
            converted = _isbs_to_forecast(subset, "Valuation IS")
            if not converted.empty:
                parts.append(converted)
                covered_vcodes |= new_vcodes
                logger.info(
                    "Forecast assembly: %d deals from ISBS Valuation IS",
                    len(new_vcodes))

    # --- Priority 3: ISBS Projected IS (YTD cumulative → periodic) ---
    proj_is = isbs_raw[isbs_raw["vSource"] == "Projected IS"]
    if not proj_is.empty:
        proj_vcodes = set(proj_is["vcode"].astype(str).str.strip().str.lower())
        new_vcodes = proj_vcodes - covered_vcodes
        if new_vcodes:
            subset = proj_is[proj_is["vcode"].isin(new_vcodes)]
            converted = _isbs_to_forecast(
                subset, "Projected IS", cumulative=True)
            if not converted.empty:
                parts.append(converted)
                covered_vcodes |= new_vcodes
                logger.info(
                    "Forecast assembly: %d deals from ISBS Projected IS",
                    len(new_vcodes))

    if parts:
        return pd.concat(parts, ignore_index=True)
    return fc_raw if fc_raw is not None else pd.DataFrame()


def _normalize_isbs(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize ISBS DataFrame once at load time.

    Strips column names, lowercases vcode, strips vSource/vAccount,
    converts mAmount to numeric, and parses dtEntry dates (with Excel
    serial number fallback). Consumers can filter directly without
    repeating this work.
    """
    if df.empty:
        return df
    normalize_columns(df)
    if 'vcode' in df.columns:
        df['vcode'] = df['vcode'].astype(str).str.strip().str.lower()
    if 'vSource' in df.columns:
        df['vSource'] = df['vSource'].astype(str).str.strip()
    if 'vAccount' in df.columns:
        df['vAccount'] = df['vAccount'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    if 'mAmount' in df.columns:
        df['mAmount'] = pd.to_numeric(df['mAmount'], errors='coerce').fillna(0)
    if 'dtEntry' in df.columns:
        df['dtEntry_parsed'] = pd.to_datetime(
            df['dtEntry'], format='mixed', dayfirst=False, errors='coerce')
        nat_count = int(df['dtEntry_parsed'].isna().sum())
        if nat_count > len(df) * 0.5:
            try:
                numeric = pd.to_numeric(df['dtEntry'], errors='coerce')
                serial = pd.to_datetime(numeric, unit='D', origin='1899-12-30', errors='coerce')
                df.loc[df['dtEntry_parsed'].isna(), 'dtEntry_parsed'] = serial[df['dtEntry_parsed'].isna()]
            except Exception:
                pass
    return df


def _append_uw_supplements(assembled: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Append supplemental ISBS Projected IS records (isbs_uw_supplements table).

    These are admin-uploaded records (e.g. account 7073 capital contributions)
    that persist across MRI refreshes. The CSV already has vSource='Projected IS'.
    """
    try:
        supp = get_adapter("isbs_uw_supplements").load(config)
        if supp is not None and not supp.empty:
            supp = supp.copy()
            # Normalize column names to match ISBS convention (lowercase vcode)
            col_map = {c: c.lower() for c in supp.columns if c.lower() == 'vcode' and c != 'vcode'}
            if col_map:
                supp = supp.rename(columns=col_map)
            if 'vSource' not in supp.columns:
                supp['vSource'] = 'Projected IS'
            supp['_is_supplement'] = True
            assembled = pd.concat([assembled, supp], ignore_index=True)
            logger.info(f"ISBS appended {len(supp):,} UW supplement rows")
    except Exception as e:
        logger.debug(f"No UW supplements table: {e}")
    return assembled


def _assemble_isbs(config: dict) -> tuple:
    """Load split ISBS tables and assemble into a single DataFrame.

    Returns (assembled_df, split_dict) where split_dict maps table names
    to their individual DataFrames.
    Falls back to legacy monolithic isbs table if split tables are all empty.
    """
    parts = []
    split_dict = {}

    for table_name, vsource in _ISBS_SPLIT.items():
        df = get_adapter(table_name).load(config)
        split_dict[table_name] = df
        if not df.empty:
            if 'vSource' not in df.columns:
                df = df.copy()
                df['vSource'] = vsource
            parts.append(df)

    if parts:
        assembled = pd.concat(parts, ignore_index=True)
        # Check for vSource categories present in split tables
        split_vsources = set(assembled['vSource'].unique()) if 'vSource' in assembled.columns else set()
        all_vsources = set(_ISBS_SPLIT.values())
        missing_vsources = all_vsources - split_vsources

        # Supplement from legacy monolithic table for any missing vSource categories
        if missing_vsources:
            legacy = get_adapter("isbs").load(config)
            if not legacy.empty and 'vSource' in legacy.columns:
                legacy_supplement = legacy[legacy['vSource'].isin(missing_vsources)]
                if not legacy_supplement.empty:
                    logger.info(f"ISBS supplementing from legacy for missing vSources: {missing_vsources} ({len(legacy_supplement):,} rows)")
                    assembled = pd.concat([assembled, legacy_supplement], ignore_index=True)

        logger.info(f"ISBS assembled from split tables: {len(assembled):,} rows")
        assembled = _append_uw_supplements(assembled, config)
        return assembled, split_dict

    # Fallback: try legacy monolithic table
    legacy = get_adapter("isbs").load(config)
    if not legacy.empty:
        logger.info(f"ISBS fallback to legacy table: {len(legacy):,} rows")
        legacy = _append_uw_supplements(legacy, config)
        return legacy, split_dict

    return pd.DataFrame(), split_dict


def load_all(db_path: str, pro_yr_base: int = 2025) -> dict:
    """Load all data via adapters. Returns dict of DataFrames.

    Cached by (db_path, pro_yr_base). Call reload() to clear.
    Each table is loaded through its registered adapter (database by default,
    MRI API if configured).
    """
    cache_key = f"{db_path}|{pro_yr_base}"
    if cache_key in _cache:
        return _cache[cache_key]

    config = {"db_path": db_path, "pro_yr_base": pro_yr_base}

    # Required tables
    inv = get_adapter("deals").load(config)
    wf = _normalize_waterfall_df(get_adapter("waterfalls").load(config))

    coa_raw = get_adapter("coa").load(config)
    coa = load_coa(coa_raw)
    acct = get_adapter("accounting").load(config)

    # Load ISBS first — needed for forecast assembly
    isbs_raw, isbs_split = _assemble_isbs(config)
    isbs_raw = _normalize_isbs(isbs_raw)

    # Assemble forecasts: forecast_feed CSV > ISBS Valuation IS > ISBS Projected IS
    fc_feed_raw = get_adapter("forecasts").load(config)
    fc_assembled = _assemble_forecasts(fc_feed_raw, isbs_raw, pro_yr_base)
    fc = load_forecast(fc_assembled, coa, pro_yr_base)

    # Optional tables
    mri_loans_all = get_adapter("loans").load(config)  # unfiltered — includes Paid Off
    mri_loans_raw = _filter_paid_off_loans(mri_loans_all)
    mri_val = get_adapter("valuations").load(config)
    relationships_raw = get_adapter("relationships").load(config)
    capital_calls_raw = get_adapter("capital_calls").load(config)
    occupancy_raw = get_adapter("occupancy").load(config)
    budget_econ_occ = get_adapter("budget_econ_occ").load(config)
    commitments_raw = get_adapter("commitments").load(config)
    tenants_raw = get_adapter("tenants").load(config)
    # Normalize MRI column names to CSV-friendly names used by get_tenant_roster()
    _tenant_col_map = {
        "vcode": "Code", "vpropertyname": "Property Name",
        "iint": "Rentable SF", "vname": "Tenant Name",
        "dtleasest": "Lease Start", "dtleaseend": "Lease End",
        "nsfleased": "SF Leased", "mrent": "Rent",
        "ivacated": "Vacated?", "imonthtomonth": "Month to Month?",
        "vvendorcode": "Tenant Code", "icommsqft": "Occupancy SF",
        "dtreported": "Occupancy Date", "vpartnershipname": "Fund Ownership %",
        "fownership": "fOwnership", "vtype2": "Property Type",
    }
    if not tenants_raw.empty:
        renames = {c: _tenant_col_map[c.lower()] for c in tenants_raw.columns
                   if c.lower() in _tenant_col_map and c != _tenant_col_map.get(c.lower())}
        if renames:
            tenants_raw = tenants_raw.rename(columns=renames)
    prospective_loans_raw = get_adapter("prospective_loans").load(config)
    deal_terms_raw = get_adapter("deal_terms").load(config)
    at_close_noi_raw = get_adapter("at_close_noi").load(config)
    event_dates_raw = get_adapter("event_dates").load(config)

    # Normalize investment map
    normalize_columns(inv)
    if "vcode" not in inv.columns and "vCode" in inv.columns:
        inv = inv.rename(columns={"vCode": "vcode"})
    inv["vcode"] = inv["vcode"].astype(str)
    if "InvestmentID" in inv.columns:
        inv["InvestmentID"] = inv["InvestmentID"].astype(str).str.strip().str.upper()

    _enrich_acquisition_dates(inv, acct)

    # Normalize accounting once (avoids repeated normalize_accounting_feed() per deal)
    acct = _normalize_accounting(acct)

    # Normalize entity IDs in relationships to uppercase
    if not relationships_raw.empty:
        if "InvestmentID" in relationships_raw.columns:
            relationships_raw["InvestmentID"] = relationships_raw["InvestmentID"].astype(str).str.strip().str.upper()
        if "InvestorID" in relationships_raw.columns:
            relationships_raw["InvestorID"] = relationships_raw["InvestorID"].astype(str).str.strip().str.upper()

    # Replace empty DataFrames from optional reads with None where appropriate
    if relationships_raw.empty:
        relationships_raw = None
    if capital_calls_raw.empty:
        capital_calls_raw = None
    if isbs_raw.empty:
        isbs_raw = None
    if occupancy_raw.empty:
        occupancy_raw = None
    if budget_econ_occ.empty:
        budget_econ_occ = None
    if commitments_raw.empty:
        commitments_raw = None
    if tenants_raw.empty:
        tenants_raw = None
    if prospective_loans_raw.empty:
        prospective_loans_raw = None
    if deal_terms_raw.empty:
        deal_terms_raw = None
    if at_close_noi_raw.empty:
        at_close_noi_raw = None
    if event_dates_raw.empty:
        event_dates_raw = None

    data = {
        "inv": inv,
        "wf": wf,
        "acct": acct,
        "fc": fc,
        "coa": coa,
        "mri_loans_raw": mri_loans_raw,
        "mri_loans_all": mri_loans_all,
        "mri_supp": pd.DataFrame(),
        "mri_val": mri_val,
        "fund_deals_raw": pd.DataFrame(),
        "inv_wf_raw": pd.DataFrame(),
        "inv_acct_raw": pd.DataFrame(),
        "relationships_raw": relationships_raw,
        "capital_calls_raw": capital_calls_raw,
        "isbs_raw": isbs_raw,
        "isbs_interim_is": isbs_split.get("isbs_interim_is", pd.DataFrame()),
        "isbs_interim_is_historical": isbs_split.get("isbs_interim_is_historical", pd.DataFrame()),
        "isbs_interim_bs": isbs_split.get("isbs_interim_bs", pd.DataFrame()),
        "isbs_budget_is": isbs_split.get("isbs_budget_is", pd.DataFrame()),
        "isbs_projected_is": isbs_split.get("isbs_projected_is", pd.DataFrame()),
        "isbs_valuation_is": isbs_split.get("isbs_valuation_is", pd.DataFrame()),
        "occupancy_raw": occupancy_raw,
        "budget_econ_occ": budget_econ_occ,
        "commitments_raw": commitments_raw,
        "tenants_raw": tenants_raw,
        "prospective_loans_raw": prospective_loans_raw,
        "deal_terms_raw": deal_terms_raw,
        "at_close_noi_raw": at_close_noi_raw,
        "event_dates_raw": event_dates_raw,
    }

    _cache[cache_key] = data
    return data


def reload(db_path: Optional[str] = None):
    """Clear all cached data. If db_path given, clear only that key."""
    if db_path:
        keys_to_remove = [k for k in _cache if k.startswith(db_path)]
        for k in keys_to_remove:
            del _cache[k]
    else:
        _cache.clear()


def refresh_table(table_name: str):
    """Reload a single table in all cached data dicts.

    Much faster than reload() which nukes the entire cache (100MB+ of data).
    Only the changed table is re-read from the database.
    """
    # Map table adapter names to cache dict keys
    table_to_key = {
        "waterfalls": "wf",
        "deals": "inv",
        "accounting": "acct",
        "relationships": "relationships_raw",
        "prospective_loans": "prospective_loans_raw",
        "capital_calls": "capital_calls_raw",
        "loans": "mri_loans_raw",
        "event_dates": "event_dates_raw",
    }
    cache_key_name = table_to_key.get(table_name, table_name)

    # If an ISBS split table or UW supplements is refreshed, also reassemble isbs_raw
    is_isbs_split = table_name in _ISBS_SPLIT or table_name == 'isbs_uw_supplements'

    for cache_key, data in _cache.items():
        db_path = cache_key.split("|")[0]
        pro_yr_base = int(cache_key.split("|")[1]) if "|" in cache_key else 2025
        config = {"db_path": db_path, "pro_yr_base": pro_yr_base}
        try:
            adapter = get_adapter(table_name)
            fresh = adapter.load(config)
            # Apply same normalization as load_all
            if table_name == "deals":
                normalize_columns(fresh)
                if "vcode" not in fresh.columns and "vCode" in fresh.columns:
                    fresh = fresh.rename(columns={"vCode": "vcode"})
                fresh["vcode"] = fresh["vcode"].astype(str)
                if "InvestmentID" in fresh.columns:
                    fresh["InvestmentID"] = fresh["InvestmentID"].astype(str).str.strip().str.upper()
                _enrich_acquisition_dates(fresh, data.get("acct"))
            elif table_name == "relationships":
                if "InvestmentID" in fresh.columns:
                    fresh["InvestmentID"] = fresh["InvestmentID"].astype(str).str.strip().str.upper()
                if "InvestorID" in fresh.columns:
                    fresh["InvestorID"] = fresh["InvestorID"].astype(str).str.strip().str.upper()
            elif table_name == "waterfalls":
                fresh = _normalize_waterfall_df(fresh)
            if table_name == "accounting":
                fresh = _normalize_accounting(fresh)
            if table_name == "loans":
                fresh = _filter_paid_off_loans(fresh)
            data[cache_key_name] = fresh

            # Reassemble isbs_raw from split tables when a split table changes
            if is_isbs_split:
                assembled, _ = _assemble_isbs(config)
                assembled = _normalize_isbs(assembled)
                data["isbs_raw"] = assembled if not assembled.empty else None

            # Reassemble forecasts when forecasts table or ISBS sources change
            if table_name == "forecasts" or is_isbs_split or table_name == "isbs":
                fc_feed = get_adapter("forecasts").load(config)
                isbs_for_fc = data.get("isbs_raw")
                fc_assembled = _assemble_forecasts(fc_feed, isbs_for_fc, pro_yr_base)
                coa = data.get("coa")
                data["fc"] = load_forecast(fc_assembled, coa, pro_yr_base)
        except Exception:
            # If single-table refresh fails, fall back to full reload
            _cache.clear()
            break


def get_data() -> dict:
    """Load all data using current Flask app config. Replaces per-blueprint _get_data()."""
    from flask import current_app
    db_path = current_app.config["DB_PATH"]
    pro_yr_base = current_app.config["PRO_YR_BASE_DEFAULT"]
    return load_all(db_path, pro_yr_base)


def exclude_sold(inv: pd.DataFrame) -> pd.DataFrame:
    """Remove deals with Sale_Status=SOLD or Lifecycle=Sold."""
    if inv is None or inv.empty:
        return inv
    mask = inv["Sale_Status"].fillna("").str.upper() != "SOLD" if "Sale_Status" in inv.columns else pd.Series(True, index=inv.index)
    if "Lifecycle" in inv.columns:
        mask = mask & (inv["Lifecycle"].fillna("").str.strip().str.upper() != "SOLD")
    return inv[mask].copy()


def get_inv_display(inv: pd.DataFrame) -> pd.DataFrame:
    """Filter out child properties and deals sold before the current year.

    Sold deals are kept when they were owned during any part of the current
    calendar year (Sale_Date in current year), so they remain reportable.
    Deals sold in prior years are excluded — those belong in Sold Portfolio.
    """
    if inv is None or inv.empty:
        return pd.DataFrame()

    result = inv.copy()

    # Exclude deals sold before the current year
    if "Sale_Status" in result.columns and "Sale_Date" in result.columns:
        import datetime as _dt
        current_year = _dt.date.today().year
        is_sold = result["Sale_Status"].fillna("").str.upper() == "SOLD"
        if "Lifecycle" in result.columns:
            is_sold = is_sold | (result["Lifecycle"].fillna("").str.strip().str.upper() == "SOLD")
        sale_dt = pd.to_datetime(result["Sale_Date"], errors="coerce")
        sold_before_current_year = is_sold & (sale_dt.dt.year < current_year)
        # Also exclude sold deals with no Sale_Date — no way to confirm they
        # were active this year
        sold_no_date = is_sold & sale_dt.isna()
        result = result[~(sold_before_current_year | sold_no_date)].copy()

    # Exclude child properties (same logic as reports_service / review_service)
    if "Portfolio_Name" in result.columns and "Investment_Name" in result.columns:
        result["Portfolio_Name"] = result["Portfolio_Name"].fillna("").astype(str).str.strip()
        parent_names = set(result["Investment_Name"].str.strip())
        is_child = (
            result["Portfolio_Name"].isin(parent_names)
            & (result["Portfolio_Name"] != result["Investment_Name"].str.strip())
            & (result["Portfolio_Name"] != "")
        )
        result = result[~is_child].copy()

    # Sort alphabetically by Investment_Name
    if "Investment_Name" in result.columns:
        result = result.sort_values("Investment_Name", key=lambda s: s.str.lower()).reset_index(drop=True)

    return result
