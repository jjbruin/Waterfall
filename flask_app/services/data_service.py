"""Data loading service — replaces _load_sqlite_data() from app.py.

Loads all tables via data adapters (database or API), caches in module-level dict.
No Streamlit dependency.
"""

import pandas as pd
from typing import Optional

import logging
from loaders import load_coa, load_forecast
from flask_app.services.data_adapters import get_adapter

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
            df = df.copy()
            if 'vSource' not in df.columns:
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
        return assembled, split_dict

    # Fallback: try legacy monolithic table
    legacy = get_adapter("isbs").load(config)
    if not legacy.empty:
        logger.info(f"ISBS fallback to legacy table: {len(legacy):,} rows")
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
    wf = get_adapter("waterfalls").load(config)

    # Normalize waterfall columns (raw table may have vCode, unstripped vmisc, etc.)
    if not wf.empty:
        wf.columns = [str(c).strip() for c in wf.columns]
        if "vCode" in wf.columns and "vcode" not in wf.columns:
            wf = wf.rename(columns={"vCode": "vcode"})
        if "vcode" in wf.columns:
            wf["vcode"] = wf["vcode"].astype(str).str.strip()
        if "vmisc" in wf.columns:
            wf["vmisc"] = wf["vmisc"].astype(str).str.strip()
        if "PropCode" in wf.columns:
            wf["PropCode"] = wf["PropCode"].astype(str).str.strip()
        if "vState" in wf.columns:
            wf["vState"] = wf["vState"].astype(str).str.strip()

    coa_raw = get_adapter("coa").load(config)
    coa = load_coa(coa_raw)
    acct = get_adapter("accounting").load(config)
    fc_raw = get_adapter("forecasts").load(config)
    fc = load_forecast(fc_raw, coa, pro_yr_base)

    # Optional tables
    mri_loans_raw = get_adapter("loans").load(config)
    mri_val = get_adapter("valuations").load(config)
    relationships_raw = get_adapter("relationships").load(config)
    capital_calls_raw = get_adapter("capital_calls").load(config)
    isbs_raw, isbs_split = _assemble_isbs(config)
    occupancy_raw = get_adapter("occupancy").load(config)
    commitments_raw = get_adapter("commitments").load(config)
    tenants_raw = get_adapter("tenants").load(config)
    prospective_loans_raw = get_adapter("prospective_loans").load(config)

    # Normalize investment map
    inv.columns = [str(c).strip() for c in inv.columns]
    if "vcode" not in inv.columns and "vCode" in inv.columns:
        inv = inv.rename(columns={"vCode": "vcode"})
    inv["vcode"] = inv["vcode"].astype(str)

    # Derive Acquisition_Date from earliest accounting activity per deal.
    # This captures the true acquisition date (e.g., fee collected at closing)
    # even when first funding occurs much later (development deals).
    if not acct.empty and "EffectiveDate" in acct.columns and "InvestmentID" in acct.columns:
        acct_dates = acct[["InvestmentID", "EffectiveDate"]].copy()
        acct_dates["_dt"] = pd.to_datetime(acct_dates["EffectiveDate"], errors="coerce")
        earliest = (
            acct_dates.dropna(subset=["_dt"])
            .groupby("InvestmentID")["_dt"]
            .min()
        )
        if "InvestmentID" in inv.columns:
            inv["Acquisition_Date"] = (
                inv["InvestmentID"].astype(str).str.strip()
                .map(earliest)
                .fillna(inv.get("Acquisition_Date"))
            )

    # Replace empty DataFrames from optional reads with None where appropriate
    if relationships_raw.empty:
        relationships_raw = None
    if capital_calls_raw.empty:
        capital_calls_raw = None
    if isbs_raw.empty:
        isbs_raw = None
    if occupancy_raw.empty:
        occupancy_raw = None
    if commitments_raw.empty:
        commitments_raw = None
    if tenants_raw.empty:
        tenants_raw = None
    if prospective_loans_raw.empty:
        prospective_loans_raw = None

    data = {
        "inv": inv,
        "wf": wf,
        "acct": acct,
        "fc": fc,
        "coa": coa,
        "mri_loans_raw": mri_loans_raw,
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
        "commitments_raw": commitments_raw,
        "tenants_raw": tenants_raw,
        "prospective_loans_raw": prospective_loans_raw,
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
    }
    cache_key_name = table_to_key.get(table_name, table_name)

    # If an ISBS split table is refreshed, also reassemble isbs_raw
    is_isbs_split = table_name in _ISBS_SPLIT

    for cache_key, data in _cache.items():
        db_path = cache_key.split("|")[0]
        pro_yr_base = int(cache_key.split("|")[1]) if "|" in cache_key else 2025
        config = {"db_path": db_path, "pro_yr_base": pro_yr_base}
        try:
            adapter = get_adapter(table_name)
            fresh = adapter.load(config)
            # Apply same normalization as load_all
            if table_name == "deals":
                fresh.columns = [str(c).strip() for c in fresh.columns]
                if "vcode" not in fresh.columns and "vCode" in fresh.columns:
                    fresh = fresh.rename(columns={"vCode": "vcode"})
                fresh["vcode"] = fresh["vcode"].astype(str)
                # Re-derive Acquisition_Date from accounting
                acct_df = data.get("acct")
                if acct_df is not None and not acct_df.empty and "InvestmentID" in fresh.columns:
                    _ad = acct_df[["InvestmentID", "EffectiveDate"]].copy()
                    _ad["_dt"] = pd.to_datetime(_ad["EffectiveDate"], errors="coerce")
                    earliest = _ad.dropna(subset=["_dt"]).groupby("InvestmentID")["_dt"].min()
                    fresh["Acquisition_Date"] = (
                        fresh["InvestmentID"].astype(str).str.strip()
                        .map(earliest).fillna(fresh.get("Acquisition_Date"))
                    )
            elif table_name == "waterfalls" and not fresh.empty:
                fresh.columns = [str(c).strip() for c in fresh.columns]
                if "vCode" in fresh.columns and "vcode" not in fresh.columns:
                    fresh = fresh.rename(columns={"vCode": "vcode"})
                if "vcode" in fresh.columns:
                    fresh["vcode"] = fresh["vcode"].astype(str).str.strip()
                if "vmisc" in fresh.columns:
                    fresh["vmisc"] = fresh["vmisc"].astype(str).str.strip()
                if "PropCode" in fresh.columns:
                    fresh["PropCode"] = fresh["PropCode"].astype(str).str.strip()
                if "vState" in fresh.columns:
                    fresh["vState"] = fresh["vState"].astype(str).str.strip()
            data[cache_key_name] = fresh

            # Reassemble isbs_raw from split tables when a split table changes
            if is_isbs_split:
                assembled, _ = _assemble_isbs(config)
                data["isbs_raw"] = assembled if not assembled.empty else None
        except Exception:
            # If single-table refresh fails, fall back to full reload
            _cache.clear()
            break


def get_inv_display(inv: pd.DataFrame) -> pd.DataFrame:
    """Filter out sold deals — equivalent to inv_disp in app.py."""
    if inv is None or inv.empty:
        return pd.DataFrame()
    mask = inv["Sale_Status"].fillna("").str.upper() != "SOLD" if "Sale_Status" in inv.columns else pd.Series(True, index=inv.index)
    return inv[mask].copy()
