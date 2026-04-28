"""
planned_loans.py
Planned second mortgage sizing from MRI_Supp.csv + prospective loan sizing

Sizes new loans based on:
- LTV constraint (projected value * LTV)
- DSCR constraint (NOI / DSCR / annual debt service)
- Debt Yield constraint (NOI / min_debt_yield)
- Takes the lesser of all constraints
"""

from datetime import date
from typing import Tuple, Dict, Any, Optional
import pandas as pd
from scipy.optimize import brentq

from models import Loan
from config import GROSS_REVENUE_ACCTS, CONTRA_REVENUE_ACCTS, EXPENSE_ACCTS, INTEREST_ACCTS, PRINCIPAL_ACCTS
from utils import as_date, month_end, add_months, month_ends_between


def projected_cap_rate_at_date(mri_val: pd.DataFrame, vcode: str, asof_date: date, proj_begin: date = None) -> float:
    """
    Get projected cap rate at a specific date

    Uses base cap rate from MRI_Val (fCapRate) + 0.05% increase per year.
    Escalation starts from the valuation date (dtVal) of the selected fCapRate row.
    Falls back to proj_begin if dtVal is not available.
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

    has_dtval = "dtVal" in sub.columns
    if has_dtval:
        sub["dtVal"] = pd.to_datetime(sub["dtVal"], errors="coerce").dt.date
        sub = sub.sort_values("dtVal")

    cap_series = pd.to_numeric(sub["fCapRate"], errors="coerce").dropna()
    fcap = float(cap_series.iloc[-1])

    # Determine escalation anchor: dtVal of the row that provided fCapRate
    anchor = None
    if has_dtval:
        anchor_val = sub.loc[cap_series.index[-1], "dtVal"]
        if pd.notna(anchor_val):
            anchor = anchor_val
    if anchor is None:
        anchor = proj_begin

    if anchor is None:
        return fcap

    years = max(0.0, (pd.Timestamp(asof_date) - pd.Timestamp(anchor)).days / 365.0)
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

    noi_12 = twelve_month_noi_after_date(fc_deal_full, orig_date)
    cap_rate = projected_cap_rate_at_date(mri_val, str(fc_deal_full["vcode"].iloc[0]), orig_date)

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


# ============================================================
# Prospective Loan Sizing (Refinance / New Mortgage)
# ============================================================

def size_prospective_loan(
    fc_deal_full: pd.DataFrame,
    mri_val: pd.DataFrame,
    prospect: Dict[str, Any],
    loan_sched: pd.DataFrame,
    vcode: str,
) -> Tuple[float, Dict[str, Any]]:
    """Size a prospective loan using LTV, DSCR, and Debt Yield constraints.

    Args:
        fc_deal_full: Full forecast for the deal (not zeroed after sale).
        mri_val: Cap rate data from MRI_Val.
        prospect: Dict from prospective_loans table row.
        loan_sched: Existing loan amortization schedule.
        vcode: Deal vcode.

    Returns:
        (estimated_loan_amount, sizing_detail_dict)
    """
    from loans import total_loan_balance_at

    refi_date = as_date(prospect["refi_date"])
    max_ltv = float(prospect.get("max_ltv") or 0)
    min_dscr = float(prospect.get("min_dscr") or 0)
    min_debt_yield = float(prospect.get("min_debt_yield") or 0)
    quoted_amount = float(prospect.get("loan_amount") or 0)
    lender_uw_noi = float(prospect.get("lender_uw_noi") or 0)
    rate = float(prospect.get("interest_rate") or 0)
    term = int(prospect.get("term_years") or 0)
    amort = int(prospect.get("amort_years") or 0)
    io_years = float(prospect.get("io_years") or 0)
    closing_costs = float(prospect.get("closing_costs") or 0)
    reserve_holdback = float(prospect.get("reserve_holdback") or 0)
    existing_loan_id_raw = prospect.get("existing_loan_id") or ""
    # Support comma-separated list of loan IDs being replaced
    replacing_ids = [s.strip() for s in str(existing_loan_id_raw).split(",") if s.strip()]

    if max_ltv > 1.5:
        max_ltv = max_ltv / 100.0
    if min_debt_yield > 1.0:
        min_debt_yield = min_debt_yield / 100.0

    # System projected NOI for 12 months from refi date
    system_noi = twelve_month_noi_after_date(fc_deal_full, refi_date)

    # Cap rate for projected value
    cap_rate = 0.0
    projected_value = 0.0
    try:
        cap_rate = projected_cap_rate_at_date(mri_val, str(vcode), refi_date)
        if cap_rate > 0:
            projected_value = system_noi / cap_rate
    except Exception:
        pass

    # Existing loan balance at refi date (use ONLY original loans, not prospective)
    existing_bal = 0.0
    if loan_sched is not None and not loan_sched.empty:
        # Filter to only the loans being replaced, or all loans if none specified
        if replacing_ids:
            existing_sched = loan_sched[loan_sched["LoanID"].isin(replacing_ids)]
        else:
            existing_sched = loan_sched
        if not existing_sched.empty:
            refi_me = month_end(refi_date)
            existing_bal = total_loan_balance_at(existing_sched, refi_me)

    # --- Constraint Analysis ---
    constraints = {}

    # LTV constraint
    ltv_max_loan = 0.0
    if max_ltv > 0 and projected_value > 0:
        ltv_max_loan = projected_value * max_ltv
    constraints["ltv"] = {
        "max_loan": ltv_max_loan,
        "projected_value": projected_value,
        "cap_rate": cap_rate,
        "max_ltv": max_ltv,
    }

    # DSCR constraint
    dscr_max_loan = 0.0
    dscr_max_ds = 0.0
    if min_dscr > 0 and system_noi > 0:
        dscr_max_ds = system_noi / min_dscr
        if rate > 0 and term > 0:
            dscr_max_loan = solve_principal_from_annual_ds(
                dscr_max_ds, rate, term, amort if amort > 0 else term, io_years, refi_date
            )
        elif dscr_max_ds > 0:
            # IO only: principal = max_ds / rate
            r = rate / 100.0 if rate > 1.0 else rate
            if r > 0:
                dscr_max_loan = dscr_max_ds / r
    constraints["dscr"] = {
        "max_loan": dscr_max_loan,
        "annual_ds": dscr_max_ds,
        "min_dscr": min_dscr,
    }

    # Debt Yield constraint (NEW)
    dy_max_loan = 0.0
    if min_debt_yield > 0 and system_noi > 0:
        dy_max_loan = system_noi / min_debt_yield
    constraints["debt_yield"] = {
        "max_loan": dy_max_loan,
        "min_debt_yield": min_debt_yield,
    }

    # Quoted amount constraint
    constraints["quoted"] = {"max_loan": quoted_amount}

    # Binding constraint — take minimum of all non-zero constraints
    candidates = []
    for name, c in constraints.items():
        if c["max_loan"] > 0:
            candidates.append((name, c["max_loan"]))

    if candidates:
        binding_name, estimated_amount = min(candidates, key=lambda x: x[1])
    else:
        binding_name = "none"
        estimated_amount = quoted_amount if quoted_amount > 0 else 0.0

    # Net proceeds
    net_proceeds = estimated_amount - existing_bal - closing_costs
    distributable = net_proceeds - reserve_holdback
    capital_call_required = distributable < 0

    # New sale date (maturity of new loan)
    new_maturity = month_end(add_months(refi_date, term * 12)) if term > 0 else None

    # NOI comparison
    noi_diff = system_noi - lender_uw_noi if lender_uw_noi > 0 else 0
    noi_diff_pct = (noi_diff / lender_uw_noi * 100) if lender_uw_noi > 0 else 0

    sizing = {
        "refi_date": str(refi_date),
        "system_noi_12m": system_noi,
        "lender_uw_noi": lender_uw_noi,
        "noi_difference": noi_diff,
        "noi_diff_pct": round(noi_diff_pct, 1),
        "constraints": constraints,
        "binding_constraint": binding_name,
        "estimated_loan_amount": estimated_amount,
        "existing_loan_balance": existing_bal,
        "replacing_loan_ids": replacing_ids,
        "closing_costs": closing_costs,
        "reserve_holdback": reserve_holdback,
        "net_refi_proceeds": net_proceeds,
        "distributable_proceeds": distributable,
        "capital_call_required": capital_call_required,
        "capital_call_amount": abs(distributable) if capital_call_required else 0,
        "new_maturity_date": str(new_maturity) if new_maturity else None,
        "projected_value": projected_value,
        "cap_rate": cap_rate,
    }

    return estimated_amount, sizing


def prospective_loan_as_loan_object(vcode: str, prospect: Dict[str, Any], final_amount: float) -> Loan:
    """Convert accepted prospective loan to Loan dataclass."""
    refi_date = as_date(prospect["refi_date"])
    term_y = int(prospect.get("term_years") or 0)
    amort_y = int(prospect.get("amort_years") or 0)
    io_y = float(prospect.get("io_years") or 0)
    rate = float(prospect.get("interest_rate") or 0)
    int_type = prospect.get("int_type", "Fixed") or "Fixed"

    term_m = int(round(term_y * 12))
    amort_m = int(round(amort_y * 12)) if amort_y > 0 else term_m
    io_m = int(round(io_y * 12))

    maturity = month_end(add_months(refi_date, term_m))
    loan_name = prospect.get("loan_name", "PROSPECTIVE") or "PROSPECTIVE"

    return Loan(
        vcode=str(vcode),
        loan_id=f"PROSP_{loan_name}",
        orig_date=refi_date,
        maturity_date=maturity,
        orig_amount=abs(float(final_amount)),
        loan_term_m=term_m,
        amort_term_m=amort_m,
        io_months=io_m,
        int_type=int_type,
        index_name=prospect.get("rate_index", "") or "",
        fixed_rate=rate,
        spread=float(prospect.get("rate_spread_bps", 0) or 0) / 10000.0,
        floor=0.0,
        cap=0.0,
    )
