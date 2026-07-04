"""Shared ISBS helper functions used by dashboard_service and financials_service.

Extracted to eliminate duplicate implementations of cumulative-to-periodic
conversion and NOI computation from YTD ISBS data.
"""

import pandas as pd


def compute_cumulative_noi(data: pd.DataFrame, dates, rev_accounts: list, exp_accounts: list) -> dict:
    """Compute cumulative NOI from ISBS data at each date.

    Args:
        data: ISBS DataFrame with dtEntry_parsed, vAccount, mAmount columns
        dates: sorted list of dates to compute NOI at
        rev_accounts: list of revenue account codes
        exp_accounts: list of expense account codes

    Returns:
        dict mapping date -> cumulative NOI value
    """
    noi_by_date = {}
    for dt in dates:
        period = data[data["dtEntry_parsed"] == dt]
        rev = period[period["vAccount"].isin(rev_accounts)]["mAmount"].sum()
        exp = period[period["vAccount"].isin(exp_accounts)]["mAmount"].sum()
        noi_by_date[dt] = (-rev) - exp
    return noi_by_date


def cumulative_to_periodic(cum_dict: dict, sorted_dates) -> dict:
    """Convert YTD cumulative ISBS values to periodic monthly values.

    For January: periodic = cumulative (start of new year).
    For other months: periodic = cumulative - prior same-year cumulative.
    """
    periodic = {}
    for i, dt in enumerate(sorted_dates):
        dt_ts = pd.Timestamp(dt)
        if dt_ts.month == 1:
            periodic[dt_ts] = cum_dict[dt]
        else:
            prior = None
            for j in range(i - 1, -1, -1):
                p = pd.Timestamp(sorted_dates[j])
                if p.year == dt_ts.year:
                    prior = sorted_dates[j]
                    break
            if prior is not None:
                periodic[dt_ts] = cum_dict[dt] - cum_dict[prior]
            else:
                periodic[dt_ts] = cum_dict[dt]
    return periodic


def aggregate_periodic(periodic_dict: dict, freq: str) -> dict:
    """Aggregate periodic monthly values to the requested frequency.

    Args:
        periodic_dict: dict mapping Timestamp -> value (monthly periodic)
        freq: "Monthly", "Quarterly", or "Annually"

    Returns:
        dict mapping period-end Timestamp -> aggregated value.
        Incomplete periods (< 3 months for quarterly, < 12 for annual) are excluded.
    """
    if not periodic_dict:
        return {}
    if freq == "Monthly":
        return periodic_dict
    elif freq == "Quarterly":
        quarterly = {}
        month_counts = {}
        for dt, val in sorted(periodic_dict.items()):
            dt_ts = pd.Timestamp(dt)
            q_month = ((dt_ts.month - 1) // 3 + 1) * 3
            q_end = pd.Timestamp(year=dt_ts.year, month=q_month, day=1) + pd.offsets.MonthEnd(0)
            quarterly[q_end] = quarterly.get(q_end, 0) + val
            month_counts[q_end] = month_counts.get(q_end, 0) + 1
        return {k: v for k, v in quarterly.items() if month_counts.get(k, 0) == 3}
    else:  # Annually
        annual = {}
        month_counts = {}
        for dt, val in sorted(periodic_dict.items()):
            yr_end = pd.Timestamp(year=pd.Timestamp(dt).year, month=12, day=31)
            annual[yr_end] = annual.get(yr_end, 0) + val
            month_counts[yr_end] = month_counts.get(yr_end, 0) + 1
        return {k: v for k, v in annual.items() if month_counts.get(k, 0) == 12}
