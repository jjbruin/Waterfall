"""Dashboard service — extracts KPI calc, NOI pipeline, chart data from dashboard_ui.py.

All functions are pure Python (no Streamlit). They accept DataFrames and return
dicts/DataFrames suitable for JSON serialization.
"""

import pandas as pd
import numpy as np
from typing import Optional

from compute import get_deal_capitalization
from consolidation import get_property_vcodes_for_deal
from config import IS_ACCOUNTS


def get_portfolio_caps(inv_disp, inv, wf, acct, mri_val, mri_loans_raw,
                       on_progress=None) -> list[dict]:
    """Get capitalization data for all active deals.

    Mirrors dashboard_ui.py::_get_portfolio_caps() without st.session_state caching.
    on_progress(current, total, deal_name) is called after each deal if provided.
    """
    from compute import prepare_cap_lookups

    # Pre-compute lookups once (avoids 84x redundant DataFrame copies/normalization)
    lookups = prepare_cap_lookups(acct, inv, mri_val, mri_loans_raw)
    prop_map = lookups["prop_map"]

    caps = []
    total = len(inv_disp)
    for i, (_, row) in enumerate(inv_disp.iterrows()):
        vcode = str(row["vcode"])
        name = row.get("DealLabel", row.get("Investment_Name", vcode))
        if on_progress:
            on_progress(i, total, name)
        try:
            prop_vcodes = prop_map.get(vcode, []) or None
            cap = get_deal_capitalization(
                acct, inv, wf, mri_val, mri_loans_raw,
                deal_vcode=vcode,
                property_vcodes=prop_vcodes,
                lookups=lookups,
            )
            cap["vcode"] = vcode
            cap["name"] = name
            cap["asset_type"] = str(row.get("Asset_Type", "") or "")
            cap["total_units"] = float(pd.to_numeric(row.get("Total_Units", 0), errors="coerce") or 0)
            caps.append(cap)
        except Exception:
            continue
    if on_progress:
        on_progress(total, total, "")
    return caps


def get_latest_occupancy(inv_disp, occupancy_raw) -> dict:
    """Return dict {vcode: latest Occ%} for each deal.

    Mirrors dashboard_ui.py::_get_latest_occupancy().
    """
    if occupancy_raw is None or occupancy_raw.empty:
        return {}

    occ = occupancy_raw.copy()
    occ.columns = [str(c).strip() for c in occ.columns]

    if "vCode" not in occ.columns:
        return {}

    occ["vCode"] = occ["vCode"].astype(str).str.strip().str.lower()

    occ_col = "Occ%" if "Occ%" in occ.columns else (
        "OccupancyPercent" if "OccupancyPercent" in occ.columns else None)
    if occ_col is None or "dtReported" not in occ.columns:
        return {}

    occ["occ_val"] = pd.to_numeric(occ[occ_col], errors="coerce")
    try:
        occ["date_parsed"] = pd.to_datetime(
            occ["dtReported"], unit="D", origin="1899-12-30", errors="coerce")
    except Exception:
        occ["date_parsed"] = pd.to_datetime(occ["dtReported"], errors="coerce")
    occ = occ.dropna(subset=["date_parsed", "occ_val"])

    occ_map = {}
    for _, row in inv_disp.iterrows():
        vc = str(row["vcode"]).strip().lower()
        deal_occ = occ[occ["vCode"] == vc]
        if not deal_occ.empty:
            latest = deal_occ.loc[deal_occ["date_parsed"].idxmax()]
            occ_map[str(row["vcode"])] = float(latest["occ_val"])

    return occ_map


def get_portfolio_kpis(caps: list[dict], occ_map: dict) -> dict:
    """Compute 6 portfolio-level KPI values.

    Mirrors dashboard_ui.py::_render_kpi_cards() data logic.
    """
    if not caps:
        return {
            "portfolio_value": 0, "debt_outstanding": 0, "wtd_avg_cap_rate": 0,
            "portfolio_occupancy": 0, "deal_count": 0, "total_pref_equity": 0,
        }

    total_value = sum(c.get("current_valuation", 0) or 0 for c in caps)
    total_debt = sum(c.get("debt", 0) or 0 for c in caps)
    total_pref = sum(c.get("pref_equity", 0) or 0 for c in caps)
    total_partner = sum(c.get("partner_equity", 0) or 0 for c in caps)

    # Weighted avg cap rate (by valuation)
    cap_rate_num = sum(
        (c.get("cap_rate", 0) or 0) * (c.get("current_valuation", 0) or 0)
        for c in caps
    )
    wtd_cap_rate = cap_rate_num / total_value if total_value > 0 else 0.0

    # Weighted avg occupancy (by units)
    occ_num = 0.0
    occ_denom = 0.0
    for c in caps:
        vc = c.get("vcode", "")
        units = c.get("total_units", 0) or 0
        occ = occ_map.get(vc)
        if occ is not None and units > 0:
            occ_num += occ * units
            occ_denom += units
    wtd_occ = occ_num / occ_denom if occ_denom > 0 else 0.0

    return {
        "portfolio_value": total_value,
        "debt_outstanding": total_debt,
        "wtd_avg_cap_rate": wtd_cap_rate,
        "portfolio_occupancy": wtd_occ,
        "deal_count": len(caps),
        "total_pref_equity": total_pref,
    }


def get_capital_structure(caps: list[dict]) -> dict:
    """Compute capital structure data for stacked bar chart.

    Returns dict with debt_m, pref_m, partner_m (in millions), avg_ltv, pref_exposure.
    """
    total_debt = sum(c.get("debt", 0) or 0 for c in caps)
    total_pref = sum(c.get("pref_equity", 0) or 0 for c in caps)
    total_partner = sum(c.get("partner_equity", 0) or 0 for c in caps)
    total_cap = total_debt + total_pref + total_partner

    return {
        "debt_m": total_debt / 1_000_000,
        "pref_m": total_pref / 1_000_000,
        "partner_m": total_partner / 1_000_000,
        "avg_ltv": total_debt / total_cap if total_cap else 0,
        "pref_exposure": (total_debt + total_pref) / total_cap if total_cap else 0,
    }


def get_occupancy_by_type(caps: list[dict], occ_map: dict) -> list[dict]:
    """Weighted-average occupancy per asset type.

    Mirrors dashboard_ui.py::_render_occupancy_by_type() data logic.
    Returns list of dicts with asset_type, occupancy, color, plus portfolio_avg.
    """
    rows = []
    for c in caps:
        vc = c.get("vcode", "")
        occ = occ_map.get(vc)
        if occ is not None:
            at = (c.get("asset_type", "") or "").strip() or "Unknown"
            units = c.get("total_units", 0) or 1
            rows.append({"Asset_Type": at, "Occupancy": occ, "Units": units})

    if not rows:
        return []

    df = pd.DataFrame(rows)
    df["Weighted_Occ"] = df["Occupancy"] * df["Units"]
    type_agg = df.groupby("Asset_Type").agg(
        Total_Weighted_Occ=("Weighted_Occ", "sum"),
        Total_Units=("Units", "sum"),
    ).reset_index()
    type_agg["Occupancy"] = type_agg["Total_Weighted_Occ"] / type_agg["Total_Units"]
    type_agg = type_agg.sort_values("Occupancy", ascending=True)

    avg_occ = type_agg["Total_Weighted_Occ"].sum() / type_agg["Total_Units"].sum()

    result = []
    for _, r in type_agg.iterrows():
        result.append({
            "asset_type": r["Asset_Type"],
            "occupancy": float(r["Occupancy"]),
            "above_avg": bool(r["Occupancy"] >= avg_occ),
        })

    return result


def get_asset_allocation(caps: list[dict]) -> list[dict]:
    """Asset allocation by type sized by pref equity.

    Mirrors dashboard_ui.py::_render_asset_allocation() data logic.
    Returns list of dicts with asset_type, pref_equity, pct, count.
    """
    rows = []
    for c in caps:
        at = (c.get("asset_type", "") or "").strip() or "Unknown"
        rows.append({"Asset_Type": at, "Pref_Equity": c.get("pref_equity", 0) or 0})

    if not rows:
        return []

    df = pd.DataFrame(rows)
    agg = df.groupby("Asset_Type").agg(
        total_pref=("Pref_Equity", "sum"),
        count=("Pref_Equity", "size"),
    ).reset_index().sort_values("total_pref", ascending=False)

    grand_total = agg["total_pref"].sum()
    if grand_total == 0:
        return []

    result = []
    for _, r in agg.iterrows():
        result.append({
            "asset_type": r["Asset_Type"],
            "pref_equity": float(r["total_pref"]),
            "pct": float(r["total_pref"] / grand_total),
            "count": int(r["count"]),
        })
    return result


def get_loan_maturity_data(mri_loans_raw, inv_disp, inv) -> dict:
    """Loan maturity data for stacked bar chart.

    Mirrors dashboard_ui.py::_render_loan_maturity() data logic.
    Returns dict with yearly (list), fixed_rates (list), detail (list).
    """
    if mri_loans_raw is None or mri_loans_raw.empty:
        return {"yearly": [], "fixed_rates": [], "detail": []}

    loans = mri_loans_raw.copy()
    loans.columns = [str(c).strip() for c in loans.columns]

    if "vCode" not in loans.columns and "vcode" in loans.columns:
        loans = loans.rename(columns={"vcode": "vCode"})
    if "vCode" not in loans.columns:
        return {"yearly": [], "fixed_rates": [], "detail": []}

    loans["vCode"] = loans["vCode"].astype(str).str.strip()

    # Filter to known deal vcodes + child property vcodes
    parent_vcodes = set(inv_disp["vcode"].astype(str).str.strip())
    all_vcodes = set(parent_vcodes)
    for vc in parent_vcodes:
        children = get_property_vcodes_for_deal(vc, inv)
        for child_vc in children:
            all_vcodes.add(str(child_vc).strip())
    loans = loans[loans["vCode"].isin(all_vcodes)]

    # Parse maturity date
    mat_col = "dtEvent" if "dtEvent" in loans.columns else (
        "dtMaturity" if "dtMaturity" in loans.columns else None)
    if mat_col is None:
        return {"yearly": [], "fixed_rates": [], "detail": []}

    loans["maturity"] = pd.to_datetime(loans[mat_col], errors="coerce")
    loans = loans.dropna(subset=["maturity"])

    if "mOrigLoanAmt" not in loans.columns:
        return {"yearly": [], "fixed_rates": [], "detail": []}

    loans["mOrigLoanAmt"] = pd.to_numeric(loans["mOrigLoanAmt"], errors="coerce").fillna(0)
    loans["Year"] = loans["maturity"].dt.year.astype(str)

    if "nRate" in loans.columns:
        loans["nRate"] = pd.to_numeric(loans["nRate"], errors="coerce").fillna(0)
    else:
        loans["nRate"] = 0.0

    # Classify fixed vs floating
    if "vIntType" in loans.columns:
        loans["vIntType"] = loans["vIntType"].fillna("").astype(str).str.strip().str.lower()
        loans["Rate Type"] = loans["vIntType"].apply(
            lambda v: "Floating" if v in ("variable", "floating") else "Fixed")
    else:
        loans["Rate Type"] = "Fixed"

    yearly = loans.groupby(["Year", "Rate Type"])["mOrigLoanAmt"].sum().reset_index()
    yearly.columns = ["year", "rate_type", "amount"]

    # Weighted avg rate for fixed loans per year
    fixed_loans = loans[(loans["Rate Type"] == "Fixed") & (loans["mOrigLoanAmt"] > 0)]
    fixed_rates = []
    if not fixed_loans.empty:
        fixed_loans["weighted_rate"] = fixed_loans["nRate"] * fixed_loans["mOrigLoanAmt"]
        fixed_wtd = fixed_loans.groupby("Year").agg(
            total_weighted_rate=("weighted_rate", "sum"),
            total_amt=("mOrigLoanAmt", "sum"),
        ).reset_index()
        fixed_wtd["avg_rate"] = fixed_wtd["total_weighted_rate"] / fixed_wtd["total_amt"]
        fixed_rates = [
            {"year": r["Year"], "avg_rate": float(r["avg_rate"])}
            for _, r in fixed_wtd.iterrows()
        ]

    # Loan detail
    inv_tmp = inv.copy()
    inv_tmp["vcode"] = inv_tmp["vcode"].astype(str).str.strip()
    vcode_to_name = dict(zip(inv_tmp["vcode"], inv_tmp["Investment_Name"].fillna("")))

    detail_rows = []
    for _, r in loans.sort_values("maturity").iterrows():
        detail_rows.append({
            "property": vcode_to_name.get(r["vCode"], r["vCode"]),
            "loan_id": str(r.get("LoanID", "")) if "LoanID" in loans.columns else "",
            "maturity": r["maturity"].strftime("%Y-%m-%d") if pd.notna(r["maturity"]) else "",
            "amount": float(r["mOrigLoanAmt"]),
            "rate_type": r["Rate Type"],
            "rate": float(r["nRate"]),
        })

    return {
        "yearly": [{"year": r["year"], "rate_type": r["rate_type"], "amount": float(r["amount"])}
                    for _, r in yearly.iterrows()],
        "fixed_rates": fixed_rates,
        "detail": detail_rows,
    }


def compute_portfolio_noi(isbs_raw, inv_disp, frequency="Quarterly",
                          period_end_label=None, occupancy_raw=None) -> dict:
    """Aggregate NOI across all parent deals and return chart data.

    Mirrors dashboard_ui.py::_compute_portfolio_noi().
    Returns dict with periods, actual_noi, uw_noi, occupancy lists (parallel arrays).
    """
    if isbs_raw is None or isbs_raw.empty:
        return {"periods": [], "actual_noi": [], "uw_noi": [], "occupancy": []}

    isbs = isbs_raw.copy()
    isbs.columns = [str(c).strip() for c in isbs.columns]

    # Filter to parent deal vcodes only
    parent_vcodes = set(inv_disp["vcode"].astype(str).str.strip().str.lower())
    if "vcode" in isbs.columns:
        isbs["vcode"] = isbs["vcode"].astype(str).str.strip().str.lower()
        isbs = isbs[isbs["vcode"].isin(parent_vcodes)]

    if isbs.empty or "dtEntry" not in isbs.columns:
        return {"periods": [], "actual_noi": [], "uw_noi": [], "occupancy": []}

    # Parse dates
    try:
        isbs["dtEntry_parsed"] = pd.to_datetime(
            isbs["dtEntry"], unit="D", origin="1899-12-30", errors="coerce")
    except Exception:
        isbs["dtEntry_parsed"] = pd.to_datetime(isbs["dtEntry"], errors="coerce")
    null_dates = isbs["dtEntry_parsed"].isna()
    if null_dates.any():
        isbs.loc[null_dates, "dtEntry_parsed"] = pd.to_datetime(
            isbs.loc[null_dates, "dtEntry"], errors="coerce")

    if "vSource" in isbs.columns:
        isbs["vSource"] = isbs["vSource"].astype(str).str.strip()
    if "vAccount" in isbs.columns:
        isbs["vAccount"] = isbs["vAccount"].astype(str).str.strip()
    if "mAmount" in isbs.columns:
        isbs["mAmount"] = pd.to_numeric(isbs["mAmount"], errors="coerce").fillna(0)

    actual_data = isbs[isbs["vSource"] == "Interim IS"]
    uw_data = isbs[isbs["vSource"] == "Projected IS"]

    if actual_data.empty and uw_data.empty:
        return {"periods": [], "actual_noi": [], "uw_noi": [], "occupancy": []}

    # Flatten account codes
    rev_accounts = []
    for acct_list in IS_ACCOUNTS["REVENUES"].values():
        rev_accounts.extend(acct_list)
    exp_accounts = []
    for acct_list in IS_ACCOUNTS["EXPENSES"].values():
        exp_accounts.extend(acct_list)

    def _compute_cumulative_noi(data, dates):
        noi_by_date = {}
        for dt in dates:
            period = data[data["dtEntry_parsed"] == dt]
            rev = period[period["vAccount"].isin(rev_accounts)]["mAmount"].sum()
            exp = period[period["vAccount"].isin(exp_accounts)]["mAmount"].sum()
            noi_by_date[dt] = (-rev) - exp
        return noi_by_date

    def _cumulative_to_periodic(cum_dict, sorted_dates):
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

    def _aggregate_periodic(periodic_dict, freq):
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

    # Per-deal: cumulative -> periodic, then SUM across deals
    actual_periodic_total = {}
    uw_periodic_total = {}

    for vc in parent_vcodes:
        for source_data, target in [
            (actual_data, actual_periodic_total),
            (uw_data, uw_periodic_total),
        ]:
            deal_src = source_data[source_data["vcode"] == vc] if "vcode" in source_data.columns else source_data
            if deal_src.empty:
                continue
            dates = sorted(deal_src["dtEntry_parsed"].dropna().unique())
            cum = _compute_cumulative_noi(deal_src, dates)
            periodic = _cumulative_to_periodic(cum, dates)
            for dt, val in periodic.items():
                target[dt] = target.get(dt, 0) + val

    actual_agg = _aggregate_periodic(actual_periodic_total, frequency)
    uw_agg = _aggregate_periodic(uw_periodic_total, frequency)

    all_period_ends = sorted(set(actual_agg.keys()) | set(uw_agg.keys()))
    if not all_period_ends:
        return {"periods": [], "actual_noi": [], "uw_noi": [], "occupancy": []}

    # Cap at most recently ended quarter
    today = pd.Timestamp.today()
    current_q_month = ((today.month - 1) // 3) * 3
    if current_q_month == 0:
        last_q_end = pd.Timestamp(year=today.year - 1, month=12, day=31)
    else:
        last_q_end = pd.Timestamp(year=today.year, month=current_q_month, day=1) + pd.offsets.MonthEnd(0)
    all_period_ends = [d for d in all_period_ends if pd.Timestamp(d) <= last_q_end]
    if not all_period_ends:
        return {"periods": [], "actual_noi": [], "uw_noi": [], "occupancy": []}

    # Trailing 12 from period end
    display_dates = all_period_ends[-12:]

    # Occupancy per period
    occ_by_period = {}
    if occupancy_raw is not None and not occupancy_raw.empty:
        occ = occupancy_raw.copy()
        occ.columns = [str(c).strip() for c in occ.columns]
        if "vCode" in occ.columns:
            occ["vCode"] = occ["vCode"].astype(str).str.strip().str.lower()
            occ = occ[occ["vCode"].isin(parent_vcodes)]
        occ_col = "Occ%" if "Occ%" in occ.columns else (
            "OccupancyPercent" if "OccupancyPercent" in occ.columns else None)
        if occ_col and "dtReported" in occ.columns:
            occ["occ_val"] = pd.to_numeric(occ[occ_col], errors="coerce")
            try:
                occ["date_parsed"] = pd.to_datetime(
                    occ["dtReported"], unit="D", origin="1899-12-30", errors="coerce")
            except Exception:
                occ["date_parsed"] = pd.to_datetime(occ["dtReported"], errors="coerce")
            occ = occ.dropna(subset=["date_parsed", "occ_val"])

            if not occ.empty:
                monthly_occ = {}
                for _, r in occ.iterrows():
                    me = pd.Timestamp(r["date_parsed"]) + pd.offsets.MonthEnd(0)
                    monthly_occ.setdefault(me, []).append(r["occ_val"])
                monthly_occ_avg = {k: sum(v) / len(v) for k, v in monthly_occ.items()}

                for dt in display_dates:
                    dt_ts = pd.Timestamp(dt)
                    if frequency == "Monthly":
                        me = dt_ts + pd.offsets.MonthEnd(0)
                        if me in monthly_occ_avg:
                            occ_by_period[dt_ts] = monthly_occ_avg[me]
                    elif frequency == "Quarterly":
                        q_start_month = ((dt_ts.month - 1) // 3) * 3 + 1
                        vals = []
                        for m in range(q_start_month, q_start_month + 3):
                            me = pd.Timestamp(year=dt_ts.year, month=m, day=1) + pd.offsets.MonthEnd(0)
                            if me in monthly_occ_avg:
                                vals.append(monthly_occ_avg[me])
                        if vals:
                            occ_by_period[dt_ts] = sum(vals) / len(vals)
                    else:
                        yr_vals = [v for k, v in monthly_occ_avg.items()
                                   if pd.Timestamp(k).year == dt_ts.year]
                        if yr_vals:
                            occ_by_period[dt_ts] = sum(yr_vals) / len(yr_vals)

    # Build parallel arrays
    periods = []
    actual_noi = []
    uw_noi = []
    occupancy = []

    for dt in display_dates:
        dt_ts = pd.Timestamp(dt)
        label = (dt_ts.strftime("%b %Y") if frequency == "Monthly" else
                 f"Q{(dt_ts.month - 1) // 3 + 1} {dt_ts.year}" if frequency == "Quarterly" else
                 str(dt_ts.year))
        periods.append(label)

        # Convert to millions
        a = actual_agg.get(dt_ts, None)
        actual_noi.append(a / 1_000_000 if a is not None else None)
        u = uw_agg.get(dt_ts, None)
        uw_noi.append(u / 1_000_000 if u is not None else None)
        occupancy.append(occ_by_period.get(dt_ts, None))

    return {
        "periods": periods,
        "actual_noi": actual_noi,
        "uw_noi": uw_noi,
        "occupancy": occupancy,
        "frequency": frequency,
    }
