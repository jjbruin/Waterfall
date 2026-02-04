"""
loans.py
Loan building and amortization schedule generation
"""

from datetime import date
from typing import List, Optional
import pandas as pd

from models import Loan
from utils import month_end, month_ends_between, add_months, annual_360_to_365


def build_loans_from_mri_loans(mri_loans: pd.DataFrame) -> List[Loan]:
    """
    Convert MRI_Loans DataFrame to list of Loan objects
    
    Args:
        mri_loans: DataFrame with loan data
    
    Returns:
        List of Loan objects
    """
    loans: List[Loan] = []
    if mri_loans is None or mri_loans.empty:
        return loans

    for _, r in mri_loans.iterrows():
        vcode = str(r["vCode"])
        loan_id = str(r["LoanID"])

        # Parse maturity date - handle string, datetime, or date
        maturity_raw = r["dtEvent"]
        if pd.isna(maturity_raw):
            maturity = None
        elif isinstance(maturity_raw, str):
            try:
                maturity = pd.to_datetime(maturity_raw).date()
            except Exception:
                maturity = None
        elif hasattr(maturity_raw, 'date'):
            maturity = maturity_raw.date()
        elif isinstance(maturity_raw, date):
            maturity = maturity_raw
        else:
            maturity = None

        loan_term_m = int(r["iLoanTerm"]) if pd.notna(r["iLoanTerm"]) and r["iLoanTerm"] > 0 else 0
        orig_date = add_months(maturity, -loan_term_m) if maturity and loan_term_m > 0 else (add_months(maturity, -120) if maturity else None)

        orig_amt = float(r["mOrigLoanAmt"]) if pd.notna(r["mOrigLoanAmt"]) else 0.0
        amort_m = int(r["iAmortTerm"]) if pd.notna(r["iAmortTerm"]) and r["iAmortTerm"] > 0 else loan_term_m
        io_m = int(r["mNominalPenalty"]) if pd.notna(r["mNominalPenalty"]) and r["mNominalPenalty"] > 0 else 0

        int_type = str(r.get("vIntType", "Fixed") or "Fixed")
        index_name = str(r.get("vIndex", "") or "").upper()
        fixed_rate = float(r.get("nRate", 0.0) or 0.0)
        spread = float(r.get("vSpread", 0.0) or 0.0)
        floor = float(r.get("nFloor", 0.0) or 0.0)
        cap = float(r.get("vIntRatereset", 0.0) or 0.0)

        loans.append(Loan(
            vcode=vcode,
            loan_id=loan_id,
            orig_date=orig_date,
            maturity_date=maturity,
            orig_amount=abs(orig_amt),
            loan_term_m=loan_term_m,
            amort_term_m=amort_m,
            io_months=io_m,
            int_type=int_type,
            index_name=index_name,
            fixed_rate=fixed_rate,
            spread=spread,
            floor=floor,
            cap=cap,
        ))

    return loans


def amortize_monthly_schedule(loan: Loan, schedule_start: date, schedule_end: date) -> pd.DataFrame:
    """
    Generate month-end amortization schedule for a loan

    Logic:
    - Fixed loans: IO period, then level-payment amortization
    - Variable loans: Interest-only for entire term (no amortization)

    Args:
        loan: Loan object
        schedule_start: Start date for schedule
        schedule_end: End date for schedule

    Returns:
        DataFrame with columns: vcode, LoanID, event_date, rate, interest,
                               principal, payment, ending_balance
    """
    # Handle missing dates - skip loans with invalid date data
    orig = loan.orig_date if pd.notna(loan.orig_date) else None
    maturity = loan.maturity_date if pd.notna(loan.maturity_date) else None

    if orig is None:
        # Cannot generate schedule without origination date
        return pd.DataFrame()

    if maturity is None:
        maturity = schedule_end

    all_dates = month_ends_between(orig, max(schedule_end, maturity))
    if not all_dates:
        return pd.DataFrame()

    # Get rate (convert from 360 to 365 basis for payment calculations)
    r_annual_360 = loan.rate_for_month()
    r_annual = annual_360_to_365(r_annual_360)
    r_m = r_annual / 12.0

    term_m = int(loan.loan_term_m) if loan.loan_term_m and loan.loan_term_m > 0 else len(all_dates)
    amort_m = int(loan.amort_term_m) if loan.amort_term_m and loan.amort_term_m > 0 else term_m
    io_m = int(loan.io_months) if loan.io_months and loan.io_months > 0 else 0

    # Variable loans: no amortization (interest-only for entire term)
    if loan.is_variable():
        io_m = term_m

    amort_after_io = max(1, amort_m - io_m)

    bal = float(loan.orig_amount)
    level_payment: Optional[float] = None

    rows = []
    for i, dte in enumerate(all_dates, start=1):
        if dte > maturity:
            break

        interest = bal * r_m

        if i <= io_m:
            # Interest-only period
            payment = interest
            principal = 0.0
        else:
            # Amortization period
            if level_payment is None:
                if r_m == 0:
                    level_payment = bal / amort_after_io
                else:
                    level_payment = bal * r_m / (1 - (1 + r_m) ** (-amort_after_io))

            payment = level_payment
            principal = max(0.0, payment - interest)

            if principal > bal:
                principal = bal
                payment = interest + principal

        bal = max(0.0, bal - principal)

        rows.append({
            "vcode": loan.vcode,
            "LoanID": loan.loan_id,
            "event_date": dte,
            "rate": float(r_annual),  # Store 365-basis decimal for display
            "interest": float(interest),
            "principal": float(principal),
            "payment": float(payment),
            "ending_balance": float(bal),
        })

        if bal <= 0.0:
            break

    sched = pd.DataFrame(rows)
    if sched.empty:
        return sched

    # Filter to requested date range
    sched = sched[
        (sched["event_date"] >= month_end(schedule_start)) &
        (sched["event_date"] <= month_end(schedule_end))
    ].copy()

    return sched


def total_loan_balance_at(loan_sched_df: pd.DataFrame, asof_me: date) -> float:
    """
    Get total loan balance across all loans at a specific month-end date
    
    Args:
        loan_sched_df: Amortization schedule DataFrame
        asof_me: Month-end date
    
    Returns:
        Total loan balance
    """
    if loan_sched_df is None or loan_sched_df.empty:
        return 0.0
    
    s = loan_sched_df.copy()
    s = s[s["event_date"] <= asof_me].copy()
    
    if s.empty:
        return 0.0
    
    # Get last record per LoanID (most recent balance)
    last = s.sort_values(["LoanID", "event_date"]).groupby("LoanID", as_index=False).tail(1)
    
    return float(last["ending_balance"].sum())
