"""
planned_loans.py
Planned second mortgage sizing from MRI_Supp.csv

Sizes new loans based on:
- LTV constraint (projected value * LTV)
- DSCR constraint (NOI / DSCR / annual debt service)
- Takes the lesser of the two
"""

from datetime import date
from typing import Tuple
import pandas as pd
from scipy.optimize import brentq

from models import Loan
from config import GROSS_REVENUE_ACCTS, CONTRA_REVENUE_ACCTS, EXPENSE_ACCTS, INTEREST_ACCTS, PRINCIPAL_ACCTS
from utils import as_date, month_end, add_months, month_ends_between


def projected_cap_rate_at_date(mri_val: pd.DataFrame, vcode: str, proj_begin: date, asof_date: date) -> float:
    """
    Get projected cap rate at a specific date
    
    Uses base cap rate from MRI_Val + 0.05% increase per year
    """
    dv = mri_val.copy()
    dv.columns = [str(c).strip() for c in dv.columns]
    if "vcode" not in dv.columns and "vCode" in dv.columns:
        dv = dv.rename(columns={"vCode": "vcode"})
    if "vcode" not in dv.columns or "fCapRate" not in dv.columns:
        raise ValueError("MRI_Val.csv must include: vcode, fCapRate")

    dv["vcode"] = dv["vcode"].astype(str)
    sub = dv[dv["vcode"] == str(vcode)].copy()
    if sub.empty:
        raise ValueError(f"MRI_Val has no rows for vcode {vcode}")

    if "dtVal" in sub.columns:
        sub["dtVal"] = pd.to_datetime(sub["dtVal"], errors="coerce").dt.date
        sub = sub.sort_values("dtVal")

    fcap = float(pd.to_numeric(sub["fCapRate"], errors="coerce").dropna().iloc[-1])

    years = max(0.0, (pd.Timestamp(asof_date) - pd.Timestamp(proj_begin)).days / 365.0)
    return fcap + 0.0005 * years


def twelve_month_noi_after_date(fc_deal_full: pd.DataFrame, anchor_date: date) -> float:
    """
    Calculate NOI for 12 months FOLLOWING anchor_date
    
    Uses full forecast (not zeroed after sale)
    """
    start_me = month_end(anchor_date)
    end_me = add_months(start_me, 12)
    f = fc_deal_full[(fc_deal_full["event_date"] >= start_me) & (fc_deal_full["event_date"] < end_me)].copy()
    if f.empty:
        return 0.0
    is_rev = f["vAccount"].isin(GROSS_REVENUE_ACCTS | CONTRA_REVENUE_ACCTS)
    is_exp = f["vAccount"].isin(EXPENSE_ACCTS)
    return float(f.loc[is_rev | is_exp, "mAmount_norm"].sum())


def existing_debt_service_12m_after(fc_deal_full: pd.DataFrame, anchor_date: date) -> float:
    """Calculate existing debt service for 12 months after anchor_date"""
    start_me = month_end(anchor_date)
    end_me = add_months(start_me, 12)
    f = fc_deal_full[(fc_deal_full["event_date"] >= start_me) & (fc_deal_full["event_date"] < end_me)].copy()
    if f.empty:
        return 0.0
    ds = f.loc[f["vAccount"].isin(INTEREST_ACCTS | PRINCIPAL_ACCTS), "mAmount_norm"].sum()
    return float(abs(ds))


def ds_first_12_months_for_principal(
    principal: float, 
    rate: float, 
    term_years: int, 
    amort_years: int, 
    io_years: float, 
    orig_date: date
) -> float:
    """
    Calculate first 12 months of debt service for given principal
    
    Used for DSCR constraint sizing
    """
    r = rate / 100.0 if rate > 1.0 else rate
    r_m = r / 12.0

    term_m = int(round(term_years * 12))
    io_m = int(round(io_years * 12))
    amort_m = int(round(amort_years * 12))
    amort_after_io = max(1, amort_m - io_m)

    bal = principal
    dates = month_ends_between(orig_date, add_months(orig_date, term_m - 1))
    level_payment = None
    rows = []
    
    for i, _dte in enumerate(dates, start=1):
        interest = bal * r_m
        if i <= io_m:
            principal_pmt = 0.0
            payment = interest
        else:
            if level_payment is None:
                if r_m == 0:
                    level_payment = bal / amort_after_io
                else:
                    level_payment = bal * r_m / (1 - (1 + r_m) ** (-amort_after_io))
            payment = level_payment
            principal_pmt = max(0.0, payment - interest)
            if principal_pmt > bal:
                principal_pmt = bal
                payment = interest + principal_pmt
        bal = max(0.0, bal - principal_pmt)
        rows.append(payment)
        
    return float(sum(rows[:12]))


def solve_principal_from_annual_ds(
    target_ds_12m: float, 
    rate: float, 
    term: int, 
    amort: int, 
    io_period: float, 
    orig_date: date
) -> float:
    """
    Reverse-engineer principal amount that generates target annual debt service
    
    Uses binary search
    """
    if target_ds_12m <= 0:
        return 0.0

    def f(p):
        return ds_first_12_months_for_principal(p, rate, term, amort, io_period, orig_date) - target_ds_12m

    lo = 0.0
    hi = 1_000_000.0
    
    # Find upper bound
    for _ in range(30):
        if f(hi) >= 0:
            break
        hi *= 2.0
    else:
        return 0.0

    return float(brentq(f, lo, hi))


def size_planned_second_mortgage(
    inv: pd.DataFrame, 
    fc_deal_full: pd.DataFrame, 
    mri_supp_row: pd.Series, 
    mri_val: pd.DataFrame
) -> Tuple[float, dict]:
    """
    Size planned second mortgage from MRI_Supp parameters
    
    Args:
        inv: Investment map (for deal info)
        fc_deal_full: Full forecast for deal
        mri_supp_row: Row from MRI_Supp for this deal
        mri_val: Cap rate data
    
    Returns:
        (new_loan_amount, debug_dict)
    """
    orig_date = as_date(mri_supp_row["Orig_Date"])
    term = int(float(mri_supp_row["Term"]))
    amort = int(float(mri_supp_row["Amort"]))
    io_years = float(mri_supp_row["I/O Period"])
    dscr_req = float(mri_supp_row["DSCR"])
    ltv = float(mri_supp_row["LTV"])
    rate = float(mri_supp_row["Rate"])

    if ltv > 1.5:
        ltv = ltv / 100.0

    proj_begin = min(fc_deal_full["event_date"])
    noi_12 = twelve_month_noi_after_date(fc_deal_full, orig_date)
    cap_rate = projected_cap_rate_at_date(mri_val, str(fc_deal_full["vcode"].iloc[0]), proj_begin, orig_date)

    projected_value = (noi_12 / cap_rate) if cap_rate != 0 else 0.0

    # Placeholder for existing debt (upgrade later to use modeled balances)
    existing_bal = 0.0

    # LTV constraint
    max_total_debt = projected_value * ltv
    max_add_ltv = max(0.0, max_total_debt - existing_bal)

    # DSCR constraint
    max_total_ds = (noi_12 / dscr_req) if dscr_req > 0 else 0.0
    existing_ds = existing_debt_service_12m_after(fc_deal_full, orig_date)
    max_add_ds = max(0.0, max_total_ds - existing_ds)

    max_add_dscr = solve_principal_from_annual_ds(max_add_ds, rate, term, amort, io_years, orig_date)

    # Take lesser of two constraints
    new_loan_amt = min(max_add_ltv, max_add_dscr)

    dbg = {
        "Orig_Date": str(orig_date),
        "NOI_12m": noi_12,
        "CapRate": cap_rate,
        "ProjectedValue": projected_value,
        "ExistingDebtAssumed": existing_bal,
        "MaxAdd_LTV": max_add_ltv,
        "ExistingDS_12m_fromForecast": existing_ds,
        "MaxAddDS_12m": max_add_ds,
        "MaxAdd_DSCR_Principal": max_add_dscr,
        "NewLoanAmt": new_loan_amt,
    }
    return float(new_loan_amt), dbg


def planned_loan_as_loan_object(vcode: str, mri_supp_row: pd.Series, new_loan_amt: float) -> Loan:
    """
    Create Loan object for planned second mortgage
    
    Treated as fixed-rate loan
    """
    orig_date = as_date(mri_supp_row["Orig_Date"])
    term_y = int(float(mri_supp_row["Term"]))
    amort_y = int(float(mri_supp_row["Amort"]))
    io_y = float(mri_supp_row["I/O Period"])
    rate = float(mri_supp_row["Rate"])

    term_m = int(round(term_y * 12))
    amort_m = int(round(amort_y * 12))
    io_m = int(round(io_y * 12))

    maturity = month_end(add_months(orig_date, term_m))

    return Loan(
        vcode=str(vcode),
        loan_id="PLANNED_2ND",
        orig_date=orig_date,
        maturity_date=maturity,
        orig_amount=abs(float(new_loan_amt)),
        loan_term_m=term_m,
        amort_term_m=amort_m,
        io_months=io_m,
        int_type="Fixed",
        index_name="",
        fixed_rate=rate,
        spread=0.0,
        floor=0.0,
        cap=0.0,
    )
