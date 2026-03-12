"""Data loading service — replaces _load_sqlite_data() from app.py.

Loads all tables via data adapters (database or API), caches in module-level dict.
No Streamlit dependency.
"""

import pandas as pd
from typing import Optional

from loaders import load_coa, load_forecast
from flask_app.services.data_adapters import get_adapter

# Module-level cache — cleared by reload()
_cache: dict = {}


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
    isbs_raw = get_adapter("isbs").load(config)
    occupancy_raw = get_adapter("occupancy").load(config)
    commitments_raw = get_adapter("commitments").load(config)
    tenants_raw = get_adapter("tenants").load(config)

    # Normalize investment map
    inv.columns = [str(c).strip() for c in inv.columns]
    if "vcode" not in inv.columns and "vCode" in inv.columns:
        inv = inv.rename(columns={"vCode": "vcode"})
    inv["vcode"] = inv["vcode"].astype(str)

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
        "occupancy_raw": occupancy_raw,
        "commitments_raw": commitments_raw,
        "tenants_raw": tenants_raw,
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
    }
    cache_key_name = table_to_key.get(table_name, table_name)

    for cache_key, data in _cache.items():
        db_path = cache_key.split("|")[0]
        pro_yr_base = int(cache_key.split("|")[1]) if "|" in cache_key else 2025
        config = {"db_path": db_path, "pro_yr_base": pro_yr_base}
        try:
            adapter = get_adapter(table_name)
            fresh = adapter.load(config)
            # Apply same normalization as load_all for deals
            if table_name == "deals":
                fresh.columns = [str(c).strip() for c in fresh.columns]
                if "vcode" not in fresh.columns and "vCode" in fresh.columns:
                    fresh = fresh.rename(columns={"vCode": "vcode"})
                fresh["vcode"] = fresh["vcode"].astype(str)
            data[cache_key_name] = fresh
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
