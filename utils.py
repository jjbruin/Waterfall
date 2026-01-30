"""
utils.py
Utility functions for date handling and environment detection
"""

from datetime import date
from pathlib import Path
from typing import List
import pandas as pd


def is_streamlit_cloud() -> bool:
    """Check if running on Streamlit Cloud"""
    return Path("/mount/src").exists()


def as_date(x) -> date:
    """Convert various formats to date object"""
    return pd.to_datetime(x).date()


def month_end(d: date) -> date:
    """Get month-end date for given date"""
    return (pd.Timestamp(d) + pd.offsets.MonthEnd(0)).date()


def add_months(d: date, months: int) -> date:
    """Add months to a date"""
    return (pd.Timestamp(d) + pd.DateOffset(months=months)).date()


def month_ends_between(start_d: date, end_d: date) -> List[date]:
    """Generate list of month-end dates between start and end"""
    start_me = month_end(start_d)
    end_me = month_end(end_d)
    if end_me < start_me:
        return []
    rng = pd.date_range(start=start_me, end=end_me, freq="M")
    return [x.date() for x in rng]


def annual_360_to_365(r_annual: float) -> float:
    """Convert annual rate from ACT/360 to ACT/365 basis"""
    return r_annual * (365.0 / 360.0)


def fmt_date(x) -> str:
    """Format date for display"""
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "—"
        s = str(x).strip()
        if s == "":
            return "—"
        return pd.to_datetime(x).date().isoformat()
    except Exception:
        return "—"


def fmt_int(x) -> str:
    """Format integer with commas"""
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "—"
        s = str(x).strip()
        if s == "":
            return "—"
        return f"{int(float(x)):,}"
    except Exception:
        return "—"


def fmt_num(x) -> str:
    """Format number with commas, no decimals"""
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "—"
        s = str(x).strip()
        if s == "":
            return "—"
        return f"{float(x):,.0f}"
    except Exception:
        return "—"
