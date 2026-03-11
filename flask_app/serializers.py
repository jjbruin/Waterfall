"""DataFrame-to-JSON serialization helpers."""

import pandas as pd
import numpy as np
from datetime import date, datetime


def _convert_value(v):
    """Convert a single value to JSON-safe type."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    return v


def df_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame to list of dicts with JSON-safe values."""
    if df is None or df.empty:
        return []
    records = df.to_dict(orient="records")
    return [{k: _convert_value(v) for k, v in row.items()} for row in records]


def df_to_response(df: pd.DataFrame, name: str = "data") -> dict:
    """Convert DataFrame to a response dict with records + column info."""
    if df is None or df.empty:
        return {name: [], "columns": [], "count": 0}
    return {
        name: df_to_records(df),
        "columns": list(df.columns),
        "count": len(df),
    }


def series_to_dict(s: pd.Series) -> dict:
    """Convert a pandas Series to a JSON-safe dict."""
    if s is None:
        return {}
    return {str(k): _convert_value(v) for k, v in s.items()}


def safe_json(obj):
    """Make an arbitrary dict/list JSON-safe (recursive)."""
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json(v) for v in obj]
    return _convert_value(obj)
