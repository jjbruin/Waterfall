# app.py
# Waterfall + XIRR Forecast — Loan Modeling + Sale/Refi Proceeds (Capital Events)
#
# Updates in this version:
#  1) Select Deal dropdown now drives from Investment_Name; internally uses vcode (deal_vcode).
#  2) Deal header card above the forecast (vcode, InvestmentID, Operating_Partner, Asset_Type, Total_Units,
#     Size_Sqf, Acquisition_Date, Lifecycle).
#  3) Planned 2nd mortgage sizing detail shows ONLY when selected deal has MRI_Supp.
#  4) Debt service is modeled for existing loans (MRI_Loans) AND planned loans (MRI_Supp if qualifies).
#  5) Adds a "Proceeds from Sale or Refinancing" line below DSCR in the annual table:
#       - 98% of qualified new loan amount in its Orig_Date period (month-end)
#       - Net sale proceeds at Sale_Date (or end of 10-year horizon if Sale_Date missing)
#         using NOI-forward valuation + cap rate projection, less 2% selling costs and loan balances at sale.
#  6) After Sale_Date: display 0s for operating cash flows and debt service numbers,
#     BUT keep full forecast internally for NOI-forward valuation purposes.
#  7) Capital events (refi proceeds and sale proceeds) are built in a dataframe and shown in an expander
#     (intended to be fed to the Capital Waterfall next).
#
# Rate conventions:
#  - Variable base indices:
#       SOFR/LIBOR -> 0.043
#       WSJ        -> 0.075
#       else       -> 0.043
#  - Variable cap logic (cap applies to index component pre-spread):
#       effective_index = min(base, cap) if cap>0 else base
#       annual_rate = max(floor + spread, effective_index + spread)
#  - Variable loans: assume NO amortization (interest-only for entire term)
#  - ACT/360 -> ACT/365 conversion for amort/payment math:
#       r_annual_365 = r_annual_360 * 365/360

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import pandas as pd
import streamlit as st
import numpy as np

from scipy.optimize import brentq


# ============================================================
# ENV DETECTION
# ============================================================
def is_streamlit_cloud() -> bool:
    return Path("/mount/src").exists()


# ============================================================
# CONFIG
# ============================================================
DEFAULT_START_YEAR = 2026
DEFAULT_HORIZON_YEARS = 10
PRO_YR_BASE_DEFAULT = 2025

SELLING_COST_RATE = 0.02          # 2% explicit selling cost per user instruction
NEW_LOAN_NET_PROCEEDS = 0.98      # 98% net proceeds for qualified new loan

# Contra-revenue (vacancy / concessions) reduces revenue
CONTRA_REVENUE_ACCTS = {4040, 4043, 4030, 4042}

REVENUE_ACCTS = {
    4010, 4012, 4020, 4041, 4045, 4040, 4043, 4030, 4042, 4070,
    4091, 4092, 4090, 4097, 4093, 4094, 4096, 4095,
    4063, 4060, 4061, 4062, 4080, 4065
}
GROSS_REVENUE_ACCTS = REVENUE_ACCTS - CONTRA_REVENUE_ACCTS

EXPENSE_ACCTS = {
    5090, 5110, 5114, 5018, 5010, 5016, 5012, 5014,
    5051, 5053, 5050, 5052, 5054, 5055,
    5060, 5067, 5063, 5069, 5061, 5064, 5065, 5068, 5070, 5066,
    5020, 5022, 5021, 5023, 5025, 5026,
    5045, 5080, 5087, 5085, 5040,
    5096, 5095, 5091, 5100
}

# Debt & capex accounts (forecast feed)
INTEREST_ACCTS = {5190, 7030}
PRINCIPAL_ACCTS = {7060}
CAPEX_ACCTS = {7050}
OTHER_EXCLUDED_ACCTS = {4050, 5220, 5210, 5195, 7065, 5120, 5130, 5400}
ALL_EXCLUDED = INTEREST_ACCTS | PRINCIPAL_ACCTS | CAPEX_ACCTS | OTHER_EXCLUDED_ACCTS

# Base index assumptions
INDEX_BASE_RATES = {
    "SOFR": 0.043,
    "LIBOR": 0.043,
    "WSJ": 0.075,
}


# ============================================================
# DATE HELPERS
# ============================================================
def as_date(x) -> date:
    return pd.to_datetime(x).date()


def month_end(d: date) -> date:
    return (pd.Timestamp(d) + pd.offsets.MonthEnd(0)).date()


def add_months(d: date, months: int) -> date:
    return (pd.Timestamp(d) + pd.DateOffset(months=months)).date()


def month_ends_between(start_d: date, end_d: date) -> List[date]:
    start_me = month_end(start_d)
    end_me = month_end(end_d)
    if end_me < start_me:
        return []
    rng = pd.date_range(start=start_me, end=end_me, freq="M")
    return [x.date() for x in rng]


def annual_360_to_365(r_annual: float) -> float:
    return r_annual * (365.0 / 360.0)


# ============================================================
# XIRR (available for later integration)
# ============================================================
def xnpv(rate: float, cfs: List[Tuple[date, float]]) -> float:
    if rate <= -0.999999999:
        return float("inf")
    cfs = sorted(cfs, key=lambda t: t[0])
    t0 = cfs[0][0]
    return sum(a / ((1 + rate) ** ((d - t0).days / 365.0)) for d, a in cfs)


def xirr(cfs: List[Tuple[date, float]]) -> float:
    return brentq(lambda r: xnpv(r, cfs), -0.9999, 10.0)


# ============================================================
# LOADERS
# ============================================================
def load_coa(df: pd.DataFrame) -> pd.DataFrame:
    """
    coa headers:
      vcode, vdescription, vtype, iNOI, vMisc, vAccountType
    vcode == vAccount in forecast_feed and TypeID in accounting_feed
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "vcode" not in df.columns:
        raise ValueError("coa.csv must include column: vcode")

    df = df.rename(columns={"vcode": "vAccount"})
    df["vAccount"] = pd.to_numeric(df["vAccount"], errors="coerce").astype("Int64")

    if "vAccountType" not in df.columns:
        df["vAccountType"] = ""
    df["vAccountType"] = df["vAccountType"].fillna("").astype(str).str.strip()

    return df[["vAccount", "vAccountType"]]


def normalize_forecast_signs(fc: pd.DataFrame) -> pd.DataFrame:
    """
    Deal-agnostic sign normalization for operating lines:
      - Gross revenue: +abs
      - Contra revenue: -abs
      - Expenses: -abs
      - Interest/Principal/Capex/Excluded: -abs
      - Other: leave as-is
    """
    out = fc.copy()
    base = pd.to_numeric(out["mAmount"], errors="coerce").fillna(0.0)

    is_gross_rev = out["vAccount"].isin(GROSS_REVENUE_ACCTS)
    is_contra_rev = out["vAccount"].isin(CONTRA_REVENUE_ACCTS)
    is_exp = out["vAccount"].isin(EXPENSE_ACCTS)
    is_outflow = out["vAccount"].isin(ALL_EXCLUDED)

    amt = base.copy()
    amt = amt.where(~is_gross_rev, base.abs())
    amt = amt.where(~is_contra_rev, -base.abs())
    amt = amt.where(~is_exp, -base.abs())
    amt = amt.where(~is_outflow, -base.abs())

    out["mAmount_norm"] = amt
    return out


def load_forecast(df: pd.DataFrame, coa: pd.DataFrame, pro_yr_base: int) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={"Vcode": "vcode", "Date": "event_date"})
    required = {"vcode", "event_date", "vAccount", "mAmount", "Pro_Yr"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"forecast_feed.csv missing columns: {missing}")

    df["vcode"] = df["vcode"].astype(str)
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.date
    df["vAccount"] = pd.to_numeric(df["vAccount"], errors="coerce").astype("Int64")
    df["mAmount"] = pd.to_numeric(df["mAmount"], errors="coerce").fillna(0.0)

    df["Year"] = (int(pro_yr_base) + pd.to_numeric(df["Pro_Yr"], errors="coerce")).astype("Int64")

    df = df.merge(coa, on="vAccount", how="left")
    df["vAccountType"] = df["vAccountType"].fillna("").astype(str)

    df = normalize_forecast_signs(df)
    return df


def load_mri_loans(df: pd.DataFrame) -> pd.DataFrame:
    """
    MRI_Loans headers include:
      vCode, LoanID, mOrigLoanAmt, iAmortTerm, mNominalPenalty, iLoanTerm,
      vIntType, vIndex, nRate, vSpread, nFloor, vIntRatereset, dtEvent
    dtEvent contains loan maturity date.
    """
    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]
    if "vCode" not in d.columns:
        raise ValueError("MRI_Loans.csv missing column: vCode")
    if "LoanID" not in d.columns:
        raise ValueError("MRI_Loans.csv missing column: LoanID")
    if "dtEvent" not in d.columns and "dtMaturity" not in d.columns:
        raise ValueError("MRI_Loans.csv must include dtEvent (maturity date) or dtMaturity")

    if "dtEvent" not in d.columns:
        d = d.rename(columns={"dtMaturity": "dtEvent"})

    d["vCode"] = d["vCode"].astype(str)
    d["LoanID"] = d["LoanID"].astype(str)
    d["dtEvent"] = pd.to_datetime(d["dtEvent"]).dt.date

    for c in ["mOrigLoanAmt", "iAmortTerm", "mNominalPenalty", "iLoanTerm", "nRate", "vSpread", "nFloor", "vIntRatereset"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        else:
            d[c] = pd.NA

    d["vIntType"] = d.get("vIntType", "").fillna("").astype(str).str.strip()
    d["vIndex"] = d.get("vIndex", "").fillna("").astype(str).str.strip().str.upper()
    return d


# ============================================================
# LOAN ENGINE
# ============================================================
@dataclass
class Loan:
    vcode: str
    loan_id: str
    orig_date: date
    maturity_date: date
    orig_amount: float
    loan_term_m: int
    amort_term_m: int
    io_months: int
    int_type: str  # Fixed / Variable
    index_name: str
    fixed_rate: float  # nRate for fixed loans
    spread: float       # vSpread for variable loans
    floor: float        # nFloor for variable loans
    cap: float          # vIntRatereset used as cap on INDEX (pre-spread) if > 0

    @staticmethod
    def _as_decimal(x) -> float:
        if x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x):
            return 0.0
        x = float(x)
        return x / 100.0 if x > 1.0 else x

    def is_variable(self) -> bool:
        return (self.int_type or "").strip().lower().startswith("var")

    def rate_for_month(self) -> float:
        """
        Fixed:
          annual_rate = nRate (decimal)

        Variable:
          base index:
            SOFR/LIBOR -> 0.043
            WSJ        -> 0.075
            else       -> 0.043
          if cap exists (>0): effective_index = min(base, cap)
          else:              effective_index = base
          annual_rate = max(floor + spread, effective_index + spread)
        """
        idx = (self.index_name or "").strip().upper()

        if not self.is_variable():
            return self._as_decimal(self.fixed_rate)

        base = INDEX_BASE_RATES.get(idx, 0.043)

        cap = self._as_decimal(self.cap)
        spread = self._as_decimal(self.spread)
        floor = self._as_decimal(self.floor)

        effective_index = min(base, cap) if cap > 0 else base
        annual_rate = max(floor + spread, effective_index + spread)
        return annual_rate


def build_loans_from_mri_loans(mri_loans: pd.DataFrame) -> List[Loan]:
    loans: List[Loan] = []
    if mri_loans is None or mri_loans.empty:
        return loans

    for _, r in mri_loans.iterrows():
        vcode = str(r["vCode"])
        loan_id = str(r["LoanID"])
        maturity = r["dtEvent"]

        loan_term_m = int(r["iLoanTerm"]) if pd.notna(r["iLoanTerm"]) and r["iLoanTerm"] > 0 else 0
        orig_date = add_months(maturity, -loan_term_m) if loan_term_m > 0 else add_months(maturity, -120)

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
    Month-end schedule.

    Fixed loans:
      - IO months: payment = interest
      - After IO: level-payment amortization (constant payment), unless maturity cuts it off (balloon possible)

    Variable loans:
      - No amortization: interest-only for entire term
    """
    all_dates = month_ends_between(loan.orig_date, max(schedule_end, loan.maturity_date))
    if not all_dates:
        return pd.DataFrame()

    # rate_for_month returns decimal annual rate on a 360 basis; convert to 365 basis for payment math
    r_annual_360 = loan.rate_for_month()
    r_annual = annual_360_to_365(r_annual_360)
    r_m = r_annual / 12.0

    term_m = int(loan.loan_term_m) if loan.loan_term_m and loan.loan_term_m > 0 else len(all_dates)
    amort_m = int(loan.amort_term_m) if loan.amort_term_m and loan.amort_term_m > 0 else term_m
    io_m = int(loan.io_months) if loan.io_months and loan.io_months > 0 else 0

    # Variable: no amortization (interest-only for entire term)
    if loan.is_variable():
        io_m = term_m

    amort_after_io = max(1, amort_m - io_m)

    bal = float(loan.orig_amount)
    level_payment: Optional[float] = None

    rows = []
    for i, dte in enumerate(all_dates, start=1):
        if dte > loan.maturity_date:
            break

        interest = bal * r_m

        if i <= io_m:
            payment = interest
            principal = 0.0
        else:
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
            "rate": float(r_annual),  # store 365-basis decimal for display
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

    sched = sched[
        (sched["event_date"] >= month_end(schedule_start)) &
        (sched["event_date"] <= month_end(schedule_end))
    ].copy()

    return sched


# ============================================================
# PLANNED SECOND MORTGAGE SIZING (MRI_Supp)
# ============================================================
def projected_cap_rate_at_date(mri_val: pd.DataFrame, vcode: str, proj_begin: date, asof_date: date) -> float:
    dv = mri_val.copy()
    dv.columns = [str(c).strip() for c in dv.columns]
    if "vcode" not in dv.columns and "vCode" in dv.columns:
        dv = dv.rename(columns={"vCode": "vcode"})
    if "vcode" not in dv.columns or "fCapRate" not in dv.columns:
        raise ValueError("MRI_Val.csv must include columns: vcode (or vCode), fCapRate")

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
    NOI (revenues + expenses per mAmount_norm) for 12 months FOLLOWING anchor_date.
    Uses the full forecast (not zeroed after sale) per user instruction.
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
    start_me = month_end(anchor_date)
    end_me = add_months(start_me, 12)
    f = fc_deal_full[(fc_deal_full["event_date"] >= start_me) & (fc_deal_full["event_date"] < end_me)].copy()
    if f.empty:
        return 0.0
    ds = f.loc[f["vAccount"].isin(INTEREST_ACCTS | PRINCIPAL_ACCTS), "mAmount_norm"].sum()
    return float(abs(ds))


def ds_first_12_months_for_principal(
    principal: float, rate: float, term_years: int, amort_years: int, io_years: float, orig_date: date
) -> float:
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


def solve_principal_from_annual_ds(target_ds_12m: float, rate: float, term: int, amort: int, io_period: float, orig_date: date) -> float:
    if target_ds_12m <= 0:
        return 0.0

    def f(p):
        return ds_first_12_months_for_principal(p, rate, term, amort, io_period, orig_date) - target_ds_12m

    lo = 0.0
    hi = 1_000_000.0
    for _ in range(30):
        if f(hi) >= 0:
            break
        hi *= 2.0
    else:
        return 0.0

    return float(brentq(f, lo, hi))


def size_planned_second_mortgage(inv: pd.DataFrame, fc_deal_full: pd.DataFrame, mri_supp_row: pd.Series, mri_val: pd.DataFrame) -> Tuple[float, dict]:
    """
    Returns (new_loan_amt, dbg_dict).
    Existing debt balance at orig is left as placeholder 0 for now (will be upgraded to modeled balance later).
    DSCR uses existing debt service from forecast (or modeled if you later choose).
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

    # Placeholder (upgrade later to modeled existing balances at orig_date)
    existing_bal = 0.0

    max_total_debt = projected_value * ltv
    max_add_ltv = max(0.0, max_total_debt - existing_bal)

    max_total_ds = (noi_12 / dscr_req) if dscr_req > 0 else 0.0
    existing_ds = existing_debt_service_12m_after(fc_deal_full, orig_date)
    max_add_ds = max(0.0, max_total_ds - existing_ds)

    max_add_dscr = solve_principal_from_annual_ds(max_add_ds, rate, term, amort, io_years, orig_date)

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
    Create a modeled Loan object for the planned second mortgage so we can compute:
      - DS, principal, interest schedule
      - balance at sale
    Treat as FIXED rate loan using MRI_Supp: Rate, Term (years), Amort (years), I/O Period (years).
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
# WATERFALL ENGINE
# ============================================================

WF_REQUIRED_COLS = {"vcode", "vmisc", "iOrder", "PropCode", "dteffective", "mAmount", "nPercent", "FXRate", "vState"}

def load_waterfalls(df: pd.DataFrame) -> pd.DataFrame:
    w = df.copy()
    w.columns = [str(c).strip() for c in w.columns]
    # Normalize column names
    ren = {}
    if "vCode" in w.columns and "vcode" not in w.columns:
        ren["vCode"] = "vcode"
    w = w.rename(columns=ren)
    if "vcode" not in w.columns:
        raise ValueError("waterfalls.csv must include vcode (or vCode)")
    w["vcode"] = w["vcode"].astype(str)

    # Ensure required columns exist (allow some optional ones)
    missing = [c for c in WF_REQUIRED_COLS if c not in w.columns]
    if missing:
        raise ValueError(f"waterfalls.csv missing columns: {missing}")

    w["vmisc"] = w["vmisc"].astype(str).str.strip()
    w["PropCode"] = w["PropCode"].astype(str).str.strip()
    w["vState"] = w["vState"].astype(str).str.strip()

    w["iOrder"] = pd.to_numeric(w["iOrder"], errors="coerce").fillna(9999).astype(int)
    w["FXRate"] = pd.to_numeric(w["FXRate"], errors="coerce").fillna(0.0).astype(float)

    # nPercent stored as decimal or percent; normalize to decimal
    p = pd.to_numeric(w["nPercent"], errors="coerce").fillna(0.0).astype(float)
    w["nPercent_dec"] = np.where(p > 1.0, p / 100.0, p)

    # mAmount numeric, blank allowed
    w["mAmount"] = pd.to_numeric(w["mAmount"], errors="coerce").fillna(0.0).astype(float)

    # dates
    w["dteffective"] = pd.to_datetime(w["dteffective"], errors="coerce").dt.date

    # sort
    w = w.sort_values(["vcode", "vmisc", "iOrder"]).reset_index(drop=True)
    return w


def cashflows_monthly_fad(fc_deal_modeled_full: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly Funds Available for Distribution from modeled forecast.
    Uses your same operating definitions:
      FAD = NOI + Interest + Principal + Excluded + Capex  (all from mAmount_norm signs)
    """
    f = fc_deal_modeled_full.copy()
    f["event_date"] = pd.to_datetime(f["event_date"]).dt.date
    f["me"] = f["event_date"].apply(month_end)

    is_rev = f["vAccount"].isin(GROSS_REVENUE_ACCTS | CONTRA_REVENUE_ACCTS)
    is_exp = f["vAccount"].isin(EXPENSE_ACCTS)
    is_int = f["vAccount"].isin(INTEREST_ACCTS)
    is_prin = f["vAccount"].isin(PRINCIPAL_ACCTS)
    is_capex = f["vAccount"].isin(CAPEX_ACCTS)
    is_excl = f["vAccount"].isin(OTHER_EXCLUDED_ACCTS)

    f["rev"] = np.where(is_rev, f["mAmount_norm"], 0.0)
    f["exp"] = np.where(is_exp, f["mAmount_norm"], 0.0)
    f["noi"] = f["rev"] + f["exp"]
    f["int"] = np.where(is_int, f["mAmount_norm"], 0.0)
    f["prin"] = np.where(is_prin, f["mAmount_norm"], 0.0)
    f["capex"] = np.where(is_capex, f["mAmount_norm"], 0.0)
    f["excl"] = np.where(is_excl, f["mAmount_norm"], 0.0)

    g = f.groupby("me", as_index=False)[["noi", "int", "prin", "capex", "excl"]].sum()
    g["fad"] = g["noi"] + g["int"] + g["prin"] + g["capex"] + g["excl"]
    return g.rename(columns={"me": "event_date"})


def xirr_safe(cfs: List[Tuple[date, float]]) -> Optional[float]:
    # Need at least one negative and one positive
    if not cfs:
        return None
    vals = [a for _, a in cfs]
    if not (min(vals) < 0 and max(vals) > 0):
        return None
    try:
        return float(xirr(cfs))
    except Exception:
        return None


@dataclass
class InvestorState:
    propcode: str
    capital_outstanding: float = 0.0  # positive = capital owed back to investor
    pref_unpaid_compounded: float = 0.0  # unpaid pref carried into next year base
    pref_accrued_current_year: float = 0.0  # accrues during the year
    last_accrual_date: Optional[date] = None
    cashflows: List[Tuple[date, float]] = None  # investor perspective: contributions negative, dists positive

    def __post_init__(self):
        if self.cashflows is None:
            self.cashflows = []


def accrue_pref_to_date(
    ist: InvestorState,
    asof: date,
    annual_rate: float,
):
    """
    Act/365
    Compounds annually on 12/31:
      - within year: accrue on (capital_outstanding + pref_unpaid_compounded)
      - on 12/31: move current_year accrual into pref_unpaid_compounded
    Also: "pay first then compound remaining" is handled by paying pref buckets before compounding.
    """
    if ist.last_accrual_date is None:
        ist.last_accrual_date = asof
        return

    if asof <= ist.last_accrual_date:
        return

    # Step through year boundaries
    cur = ist.last_accrual_date
    while cur < asof:
        year_end = date(cur.year, 12, 31)
        next_stop = min(asof, year_end)
        days = (next_stop - cur).days
        if days > 0 and annual_rate > 0:
            base = max(0.0, ist.capital_outstanding + ist.pref_unpaid_compounded)
            ist.pref_accrued_current_year += base * annual_rate * (days / 365.0)

        # If we hit year-end exactly, compound (roll) the remaining current-year accrual
        if next_stop == year_end and asof > year_end:
            # At year end we will compound AFTER any payments on that date. Payments happen in waterfall.
            # So compounding should occur AFTER allocations for that period.
            pass

        cur = next_stop
        if cur == year_end and cur < asof:
            # advance one day to start next year's accrual window
            cur = cur  # keep at 12/31; the loop step will continue

            # We'll compound in a separate hook after distributions at 12/31 month-end.
            # (Handled in run_waterfall_period)
            break

    ist.last_accrual_date = asof


def compound_if_year_end(istates: Dict[str, InvestorState], period_date: date):
    """After paying pref on period_date, if period_date is 12/31, roll current-year accrual into compounded."""
    if period_date.month == 12 and period_date.day == 31:
        for s in istates.values():
            if s.pref_accrued_current_year > 0:
                s.pref_unpaid_compounded += s.pref_accrued_current_year
                s.pref_accrued_current_year = 0.0


def apply_distribution(ist: InvestorState, d: date, amt: float):
    """Record a distribution (positive cashflow from investor perspective)."""
    if amt == 0:
        return
    ist.cashflows.append((d, float(amt)))


def apply_contribution(ist: InvestorState, d: date, amt: float):
    """Record a contribution (negative cashflow from investor perspective)."""
    if amt == 0:
        return
    ist.cashflows.append((d, -abs(float(amt))))
    ist.capital_outstanding += abs(float(amt))


def pay_pref(ist: InvestorState, d: date, available: float) -> float:
    """Pay pref: first compounded unpaid, then current-year accrued."""
    pay = 0.0
    if available <= 0:
        return 0.0

    # Pay compounded unpaid first
    if ist.pref_unpaid_compounded > 0:
        x = min(available, ist.pref_unpaid_compounded)
        ist.pref_unpaid_compounded -= x
        available -= x
        pay += x

    if available <= 0:
        if pay:
            apply_distribution(ist, d, pay)
        return pay

    # Pay current-year accrued next
    if ist.pref_accrued_current_year > 0:
        x = min(available, ist.pref_accrued_current_year)
        ist.pref_accrued_current_year -= x
        available -= x
        pay += x

    if pay:
        apply_distribution(ist, d, pay)
    return pay


def pay_initial_capital(ist: InvestorState, d: date, available: float) -> float:
    """Return capital until outstanding capital is 0."""
    if available <= 0 or ist.capital_outstanding <= 0:
        return 0.0
    x = min(available, ist.capital_outstanding)
    ist.capital_outstanding -= x
    apply_distribution(ist, d, x)
    return x


def irr_needed_distribution(ist: InvestorState, d: date, target_irr: float, max_cash: float) -> float:
    """
    IRR lookback: find distribution amount (<= max_cash) to make IRR ~= target_irr.
    Uses brentq on IRR difference if possible.
    """
    if max_cash <= 0:
        return 0.0
    # Must have at least one negative already
    if not ist.cashflows or min(a for _, a in ist.cashflows) >= 0:
        return 0.0

    def irr_of(x):
        cfs = ist.cashflows + [(d, float(x))]
        r = xirr_safe(cfs)
        return r if r is not None else None

    r0 = irr_of(0.0)
    if r0 is None:
        return 0.0
    if r0 >= target_irr:
        return 0.0

    r1 = irr_of(max_cash)
    if r1 is None:
        return 0.0
    if r1 < target_irr:
        # Even all cash doesn't reach target; take all
        return max_cash

    # Root find x such that irr(x) - target = 0
    def f(x):
        rr = irr_of(x)
        if rr is None:
            return -1e9
        return rr - target_irr

    try:
        x = float(brentq(f, 0.0, max_cash))
        return max(0.0, min(max_cash, x))
    except Exception:
        return 0.0


def run_waterfall_period(
    steps: pd.DataFrame,
    istates: Dict[str, InvestorState],
    period_date: date,
    cash_available: float,
    pref_rates: Dict[str, float],
) -> Tuple[float, List[dict]]:
    """
    Returns (remaining_cash, allocations_rows)
    allocations_rows: list of dict with step details + allocated amount
    """
    alloc_rows = []

    # Accrue pref to this date for all investors in this waterfall using their per-investor pref rate
    for pc, stt in istates.items():
        rate = float(pref_rates.get(pc, 0.0))
        accrue_pref_to_date(stt, period_date, rate)

    remaining = float(cash_available)

    for _, step in steps.iterrows():
        pc = str(step["PropCode"])
        state = str(step["vState"]).strip()
        fx = float(step["FXRate"])
        rate = float(step["nPercent_dec"])

        if pc not in istates:
            istates[pc] = InvestorState(propcode=pc)

        stt = istates[pc]

        allocated = 0.0

        # NOTE: FXRate applies to this step’s share of *available in this step*
        step_cash = remaining

        if state == "Pref":
            # Use nPercent as the pref rate for this PropCode (stored)
            pref_rates[pc] = rate
            # Pay pref due to this investor, but do not exceed step_cash * fx (if fx < 1)
            cap = step_cash * fx if fx > 0 else 0.0
            allocated = pay_pref(stt, period_date, cap)
            remaining -= allocated

        elif state == "Initial":
            cap = step_cash * fx if fx > 0 else 0.0
            allocated = pay_initial_capital(stt, period_date, cap)
            remaining -= allocated

        elif state == "Share":
            cap = step_cash * fx if fx > 0 else 0.0
            allocated = cap
            if allocated > 0:
                apply_distribution(stt, period_date, allocated)
                remaining -= allocated

        elif state == "Tag":
            # Treat Tag as an explicit share using FXRate (lets your CSV control allocation precisely)
            cap = step_cash * fx if fx > 0 else 0.0
            allocated = cap
            if allocated > 0:
                apply_distribution(stt, period_date, allocated)
                remaining -= allocated

        elif state == "IRR":
            # Allocate enough to hit target IRR = nPercent (decimal)
            target = rate
            cap = step_cash * fx if fx > 0 else 0.0
            needed = irr_needed_distribution(stt, period_date, target, cap)
            allocated = needed
            if allocated > 0:
                apply_distribution(stt, period_date, allocated)
                remaining -= allocated

        elif state in {"Def_Int", "Default"}:
            # Placeholder: define precisely how these should compute (default interest/capital mechanics)
            allocated = 0.0

        else:
            allocated = 0.0

        alloc_rows.append({
            "event_date": period_date,
            "iOrder": int(step["iOrder"]),
            "PropCode": pc,
            "vState": state,
            "FXRate": fx,
            "nPercent": rate,
            "Allocated": float(allocated),
            "RemainingAfter": float(remaining),
        })

        if remaining <= 1e-9:
            remaining = 0.0
            break

    # After allocations on this period, if 12/31 compound remaining current-year accruals
    compound_if_year_end(istates, period_date)

    return remaining, alloc_rows


def run_waterfall(
    wf_steps: pd.DataFrame,
    vcode: str,
    wf_name: str,
    period_cash: pd.DataFrame,
    initial_states: Optional[Dict[str, InvestorState]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    wf_name in {"CF_WF", "Cap_WF"}; matches vmisc
    period_cash columns: event_date, cash_available
    Returns:
      allocations_df (step-by-step)
      investor_summary_df
    """
    steps = wf_steps[(wf_steps["vcode"] == str(vcode)) & (wf_steps["vmisc"] == wf_name)].copy()
    steps = steps.sort_values("iOrder")

    if steps.empty:
        return pd.DataFrame(), pd.DataFrame()

    istates = initial_states if initial_states is not None else {}
    pref_rates = {}  # per investor pref rate for accrual

    all_rows = []
    for _, r in period_cash.sort_values("event_date").iterrows():
        d = r["event_date"]
        cash = float(r["cash_available"])
        _rem, rows = run_waterfall_period(steps, istates, d, cash, pref_rates)
        all_rows.extend(rows)

    alloc_df = pd.DataFrame(all_rows)

    # investor summary
    inv_rows = []
    for pc, stt in istates.items():
        irr = xirr_safe(sorted(stt.cashflows, key=lambda t: t[0]))
        inv_rows.append({
            "PropCode": pc,
            "EndingCapitalOutstanding": stt.capital_outstanding,
            "UnpaidPrefCompounded": stt.pref_unpaid_compounded,
            "AccruedPrefCurrentYear": stt.pref_accrued_current_year,
            "XIRR": irr,
            "TotalDistributions": sum(a for _, a in stt.cashflows if a > 0),
            "TotalContributions": -sum(a for _, a in stt.cashflows if a < 0),
        })

    inv_sum = pd.DataFrame(inv_rows)
    if not inv_sum.empty:
        inv_sum = inv_sum.sort_values("PropCode")

    return alloc_df, inv_sum

# ============================================================
# ACCOUNTING FEED -> SEED INVESTOR STATES
# ============================================================

def normalize_accounting_feed(acct: pd.DataFrame) -> pd.DataFrame:
    """
    Expected headers:
      InvestmentID, InvestorID, EffectiveDate, TypeID, Amt, SubtypeUID,
      MajorType, Typename, Partner, Capital, ROE_Income, YEAR, Qtr

    Rules:
      - Ignore rows where MajorType is missing (quarter-end system rows)
      - Contribution has negative Amt
      - Capital = 'Y' => capital contribution/distribution (Cap_WF history)
        else => CF_WF history
    """
    a = acct.copy()
    a.columns = [str(c).strip() for c in a.columns]

    required = {"InvestmentID", "InvestorID", "EffectiveDate", "MajorType", "Amt", "Capital"}
    missing = [c for c in required if c not in a.columns]
    if missing:
        raise ValueError(f"accounting_feed.csv missing columns: {missing}")

    a["InvestmentID"] = a["InvestmentID"].astype(str).str.strip()
    a["InvestorID"] = a["InvestorID"].astype(str).str.strip()
    a["EffectiveDate"] = pd.to_datetime(a["EffectiveDate"], errors="coerce").dt.date

    a["MajorType"] = a["MajorType"].fillna("").astype(str).str.strip()
    a = a[a["MajorType"] != ""].copy()  # ignore system quarter-end rows

    a["Amt"] = pd.to_numeric(a["Amt"], errors="coerce").fillna(0.0).astype(float)

    a["Capital"] = a["Capital"].fillna("").astype(str).str.strip().str.upper()
    a["is_capital"] = a["Capital"].eq("Y")

    a["MajorTypeNorm"] = a["MajorType"].str.lower()
    a["is_contribution"] = a["MajorTypeNorm"].str.contains("contrib")
    a["is_distribution"] = a["MajorTypeNorm"].str.contains("distri")

    # keep only rows that are explicitly contrib or distr
    a = a[a["is_contribution"] | a["is_distribution"]].copy()

    return a


def build_investmentid_to_vcode(inv_map: pd.DataFrame) -> dict:
    inv = inv_map.copy()
    inv.columns = [str(c).strip() for c in inv.columns]
    if "InvestmentID" not in inv.columns:
        raise ValueError("investment_map.csv must include InvestmentID")
    if "vcode" not in inv.columns and "vCode" in inv.columns:
        inv = inv.rename(columns={"vCode": "vcode"})
    if "vcode" not in inv.columns:
        raise ValueError("investment_map.csv must include vcode (or vCode)")

    inv["InvestmentID"] = inv["InvestmentID"].astype(str).str.strip()
    inv["vcode"] = inv["vcode"].astype(str).str.strip()

    # If duplicates exist, last wins (you can make this stricter later)
    return dict(zip(inv["InvestmentID"], inv["vcode"]))


def pref_rates_from_waterfall_steps(wf_steps: pd.DataFrame, vcode: str) -> dict:
    """
    Pull pref rate per PropCode from any 'Pref' step (CF_WF or Cap_WF).
    If multiple Pref steps exist, we take the first by iOrder per vmisc.
    """
    s = wf_steps[wf_steps["vcode"].astype(str) == str(vcode)].copy()
    if s.empty:
        return {}
    pref = s[s["vState"].astype(str).str.strip() == "Pref"].copy()
    if pref.empty:
        return {}
    pref = pref.sort_values(["vmisc", "iOrder"])
    # nPercent_dec already normalized in load_waterfalls()
    out = {}
    for pc, grp in pref.groupby("PropCode"):
        out[str(pc)] = float(grp["nPercent_dec"].iloc[0])
    return out


def apply_accounting_event(
    ist: InvestorState,
    d: date,
    amt: float,
    is_capital: bool,
    is_contribution: bool,
    is_distribution: bool,
):
    """
    Accounting feed already reflects historical CF_WF/Cap_WF results.
    We replay it to establish:
      - capital outstanding (from capital events)
      - unpaid/compounded pref (by treating CF distributions as paying pref first)
      - investor cashflows (for XIRR)

    Sign conventions:
      - Contribution has negative Amt (from user); distribution could be positive.
    """
    amt = float(amt)

    # Normalize contribution / distribution amounts:
    # Contribution should be negative cashflow for investor
    if is_contribution:
        cf = amt  # should already be negative
        if cf > 0:
            cf = -abs(cf)

        ist.cashflows.append((d, cf))

        if is_capital:
            ist.capital_outstanding += abs(cf)  # increase capital outstanding
        # if non-capital contribution exists, we do NOT change capital_outstanding (rare)

        return

    if is_distribution:
        cf = amt
        if cf < 0:
            cf = abs(cf)  # ensure positive distribution

        if is_capital:
            # Capital distribution: return capital first, excess is just profit distribution
            cap_return = min(cf, max(0.0, ist.capital_outstanding))
            ist.capital_outstanding -= cap_return
            ist.cashflows.append((d, cf))
            return
        else:
            # CF distribution: pay pref first per your rule, remainder is residual distribution
            remaining = cf

            # Pay pref buckets FIRST but record as cashflow regardless
            # pay_pref() already appends distribution cashflow for the pref component
            paid_pref = pay_pref(ist, d, remaining)
            remaining -= paid_pref

            if remaining > 0:
                apply_distribution(ist, d, remaining)

            return


def seed_states_from_accounting(
    acct_raw: pd.DataFrame,
    inv_map: pd.DataFrame,
    wf_steps: pd.DataFrame,
    target_vcode: str,
) -> Dict[str, InvestorState]:
    """
    Build InvestorState per PropCode by replaying accounting events for the selected deal.

    Mapping:
      - InvestmentID (acct) -> InvestmentID (investment_map) -> vcode
      - InvestorID (acct) -> PropCode (waterfalls)

    Output:
      states dict keyed by PropCode
    """
    acct = normalize_accounting_feed(acct_raw)

    inv_to_vcode = build_investmentid_to_vcode(inv_map)

    # Restrict to selected deal via InvestmentID->vcode
    acct["vcode"] = acct["InvestmentID"].map(inv_to_vcode)
    acct = acct[acct["vcode"].astype(str) == str(target_vcode)].copy()

    # Pref rates per PropCode for accrual
    pref_rates = pref_rates_from_waterfall_steps(wf_steps, target_vcode)

    # Build states
    states: Dict[str, InvestorState] = {}
    acct = acct.sort_values(["EffectiveDate", "InvestorID"]).copy()

    for _, r in acct.iterrows():
        pc = str(r["InvestorID"])  # InvestorID == PropCode
        d = r["EffectiveDate"]
        if pd.isna(d):
            continue

        if pc not in states:
            states[pc] = InvestorState(propcode=pc)
            # start accrual clock at first event date
            states[pc].last_accrual_date = d

        stt = states[pc]

        # Accrue pref up to this EffectiveDate before applying the cash event
        rate = float(pref_rates.get(pc, 0.0))
        accrue_pref_to_date(stt, d, rate)

        apply_accounting_event(
            ist=stt,
            d=d,
            amt=r["Amt"],
            is_capital=bool(r["is_capital"]),
            is_contribution=bool(r["is_contribution"]),
            is_distribution=bool(r["is_distribution"]),
        )

        # If the event date is 12/31, compound AFTER payments that day
        compound_if_year_end(states, d)

    return states


# ============================================================
# ANNUAL AGGREGATION
# ============================================================
def annual_aggregation_table(
    fc_deal_display: pd.DataFrame,
    start_year: int,
    horizon_years: int,
    proceeds_by_year: Optional[pd.Series] = None,
) -> pd.DataFrame:
    years = list(range(int(start_year), int(start_year) + int(horizon_years)))
    f = fc_deal_display[fc_deal_display["Year"].isin(years)].copy()

    def sum_where(mask: pd.Series) -> pd.Series:
        if f.empty:
            return pd.Series(dtype=float)
        return f.loc[mask].groupby("Year")["mAmount_norm"].sum()

    revenues = sum_where(f["vAccount"].isin(GROSS_REVENUE_ACCTS | CONTRA_REVENUE_ACCTS))
    expenses = sum_where(f["vAccount"].isin(EXPENSE_ACCTS))

    interest = sum_where(f["vAccount"].isin(INTEREST_ACCTS))
    principal = sum_where(f["vAccount"].isin(PRINCIPAL_ACCTS))
    capex = sum_where(f["vAccount"].isin(CAPEX_ACCTS))
    excluded_other = sum_where(f["vAccount"].isin(OTHER_EXCLUDED_ACCTS))

    out = pd.DataFrame({"Year": years}).set_index("Year")
    out["Revenues"] = revenues
    out["Expenses"] = expenses
    out["NOI"] = out["Revenues"].fillna(0.0) + out["Expenses"].fillna(0.0)

    out["Interest"] = interest
    out["Principal"] = principal
    out["Total Debt Service"] = out["Interest"].fillna(0.0) + out["Principal"].fillna(0.0)

    out["Excluded Accounts"] = excluded_other
    out["Capital Expenditures"] = capex

    out["Funds Available for Distribution"] = (
        out["NOI"].fillna(0.0)
        + out["Interest"].fillna(0.0)
        + out["Principal"].fillna(0.0)
        + out["Excluded Accounts"].fillna(0.0)
        + out["Capital Expenditures"].fillna(0.0)
    )

    tds_abs = out["Total Debt Service"].abs().replace(0, pd.NA)
    out["Debt Service Coverage Ratio"] = out["NOI"] / tds_abs

    # Proceeds (capital events)
    if proceeds_by_year is None:
        proceeds_by_year = pd.Series(dtype=float)
    out["Proceeds from Sale or Refinancing"] = proceeds_by_year.reindex(out.index).fillna(0.0)

    return out.reset_index().fillna(0.0)


def pivot_annual_table(df: pd.DataFrame) -> pd.DataFrame:
    wide = df.set_index("Year").T
    wide.index.name = "Line Item"
    desired_order = [
        "Revenues", "Expenses", "NOI",
        "Interest", "Principal", "Total Debt Service",
        "Capital Expenditures", "Excluded Accounts",
        "Funds Available for Distribution", "Debt Service Coverage Ratio",
        "Proceeds from Sale or Refinancing",
    ]
    existing = [r for r in desired_order if r in wide.index]
    remainder = [r for r in wide.index if r not in existing]
    return wide.loc[existing + remainder]


def style_annual_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def money_fmt(x):
        if pd.isna(x):
            return ""
        return f"{x:,.0f}"

    def dscr_fmt(x):
        if pd.isna(x):
            return ""
        return f"{x:,.2f}"

    styler = df.style.format(money_fmt)

    if "Debt Service Coverage Ratio" in df.index:
        styler = styler.format(
            {col: dscr_fmt for col in df.columns},
            subset=pd.IndexSlice[["Debt Service Coverage Ratio"], :]
        )

    # Uniform column widths + right-aligned numbers
    styler = styler.set_table_styles(
        [
            {"selector": "th", "props": [("text-align", "left"), ("width", "240px")]},
            {"selector": "td", "props": [("text-align", "right"), ("width", "140px")]},
        ],
        overwrite=False,
    )

    # Underline expenses
    if "Expenses" in df.index:
        styler = styler.set_properties(subset=pd.IndexSlice[["Expenses"], :], **{"text-decoration": "underline"})

    # Double line under NOI
    if "NOI" in df.index:
        styler = styler.set_properties(
            subset=pd.IndexSlice[["NOI"], :],
            **{"border-bottom": "3px double black", "font-weight": "bold"}
        )

    # Line above FAD and bold FAD
    if "Funds Available for Distribution" in df.index:
        fad_idx = df.index.get_loc("Funds Available for Distribution")
        if fad_idx > 0:
            prev_row = df.index[fad_idx - 1]
            styler = styler.set_properties(subset=pd.IndexSlice[[prev_row], :], **{"border-bottom": "2px solid black"})
        styler = styler.set_properties(subset=pd.IndexSlice[["Funds Available for Distribution"], :], **{"font-weight": "bold"})

    # Light line above DSCR
    if "Debt Service Coverage Ratio" in df.index:
        styler = styler.set_properties(subset=pd.IndexSlice[["Debt Service Coverage Ratio"], :], **{"border-top": "1px solid #999"})

    # Capital proceeds row emphasis
    if "Proceeds from Sale or Refinancing" in df.index:
        styler = styler.set_properties(
            subset=pd.IndexSlice[["Proceeds from Sale or Refinancing"], :],
            **{"border-top": "2px solid black", "font-weight": "bold"}
        )

    return styler


# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(layout="wide")
st.title("Waterfall + XIRR Forecast — Loan Modeling")

CLOUD = is_streamlit_cloud()

with st.sidebar:
    st.header("Data Source")

    if CLOUD:
        mode = "Upload CSVs"
        st.info("Running on Streamlit Cloud — local folders are disabled. Please upload CSVs.")
    else:
        mode = st.radio("Load data from:", ["Local folder", "Upload CSVs"], index=0)

    folder = None
    uploads: Dict[str, Optional[object]] = {}

    if mode == "Local folder":
        folder = st.text_input("Data folder path", placeholder=r"C:\Path\To\Data")
        st.caption("Required: investment_map.csv, waterfalls.csv, coa.csv, accounting_feed.csv, forecast_feed.csv")
        st.caption("Optional: MRI_Loans.csv, MRI_Supp.csv, MRI_Val.csv")
    else:
        uploads["investment_map"] = st.file_uploader("investment_map.csv", type="csv")
        uploads["waterfalls"] = st.file_uploader("waterfalls.csv", type="csv")
        uploads["coa"] = st.file_uploader("coa.csv", type="csv")
        uploads["accounting_feed"] = st.file_uploader("accounting_feed.csv", type="csv")
        uploads["forecast_feed"] = st.file_uploader("forecast_feed.csv", type="csv")

        st.divider()
        st.subheader("Optional Inputs")
        uploads["MRI_Loans"] = st.file_uploader("MRI_Loans.csv (existing loans)", type="csv")
        uploads["MRI_Supp"] = st.file_uploader("MRI_Supp.csv (planned 2nd mortgages)", type="csv")
        uploads["MRI_Val"] = st.file_uploader("MRI_Val.csv (cap rates)", type="csv")

    st.divider()
    st.header("Report Settings")
    start_year = st.number_input("Start year", min_value=2000, max_value=2100, value=DEFAULT_START_YEAR, step=1)
    horizon_years = st.number_input("Horizon (years)", min_value=1, max_value=30, value=DEFAULT_HORIZON_YEARS, step=1)
    pro_yr_base = st.number_input("Pro_Yr base year", min_value=1900, max_value=2100, value=PRO_YR_BASE_DEFAULT, step=1)


def load_inputs():
    if CLOUD and mode == "Local folder":
        st.error("Local folder mode is disabled on Streamlit Cloud.")
        st.stop()

    if mode == "Local folder":
        if not folder:
            st.error("Please enter a data folder path.")
            st.stop()

        inv = pd.read_csv(f"{folder}/investment_map.csv")
        wf = pd.read_csv(f"{folder}/waterfalls.csv")
        coa = load_coa(pd.read_csv(f"{folder}/coa.csv"))
        acct = pd.read_csv(f"{folder}/accounting_feed.csv")
        fc = load_forecast(pd.read_csv(f"{folder}/forecast_feed.csv"), coa, int(pro_yr_base))

        mri_loans_raw = pd.read_csv(f"{folder}/MRI_Loans.csv") if Path(f"{folder}/MRI_Loans.csv").exists() else pd.DataFrame()
        mri_supp = pd.read_csv(f"{folder}/MRI_Supp.csv") if Path(f"{folder}/MRI_Supp.csv").exists() else pd.DataFrame()
        mri_val = pd.read_csv(f"{folder}/MRI_Val.csv") if Path(f"{folder}/MRI_Val.csv").exists() else pd.DataFrame()

    else:
        for k in ["investment_map", "waterfalls", "coa", "accounting_feed", "forecast_feed"]:
            if uploads.get(k) is None:
                st.warning(f"Please upload {k}.csv")
                st.stop()

        inv = pd.read_csv(uploads["investment_map"])
        wf = pd.read_csv(uploads["waterfalls"])
        coa = load_coa(pd.read_csv(uploads["coa"]))
        acct = pd.read_csv(uploads["accounting_feed"])
        fc = load_forecast(pd.read_csv(uploads["forecast_feed"]), coa, int(pro_yr_base))

        mri_loans_raw = pd.read_csv(uploads["MRI_Loans"]) if uploads.get("MRI_Loans") is not None else pd.DataFrame()
        mri_supp = pd.read_csv(uploads["MRI_Supp"]) if uploads.get("MRI_Supp") is not None else pd.DataFrame()
        mri_val = pd.read_csv(uploads["MRI_Val"]) if uploads.get("MRI_Val") is not None else pd.DataFrame()

    inv.columns = [str(c).strip() for c in inv.columns]
    if "vcode" not in inv.columns and "vCode" in inv.columns:
        inv = inv.rename(columns={"vCode": "vcode"})
    inv["vcode"] = inv["vcode"].astype(str)

    return inv, wf, acct, fc, mri_loans_raw, mri_supp, mri_val


inv, wf, acct, fc, mri_loans_raw, mri_supp, mri_val = load_inputs()

# --- Select by Investment_Name (drives vcode internally) ---
inv_disp = inv.copy()
if "Investment_Name" not in inv_disp.columns:
    st.error("investment_map.csv must include column: Investment_Name")
    st.stop()

inv_disp["Investment_Name"] = inv_disp["Investment_Name"].fillna("").astype(str)
inv_disp["vcode"] = inv_disp["vcode"].astype(str)

name_counts = inv_disp["Investment_Name"].value_counts()
inv_disp["DealLabel"] = inv_disp.apply(
    lambda r: f"{r['Investment_Name']} ({r['vcode']})" if name_counts.get(r["Investment_Name"], 0) > 1 else r["Investment_Name"],
    axis=1
)

labels_sorted = sorted(inv_disp["DealLabel"].dropna().unique().tolist(), key=lambda x: x.lower())
selected_label = st.selectbox("Select Deal", labels_sorted)

selected_row = inv_disp[inv_disp["DealLabel"] == selected_label].iloc[0]
deal_vcode = str(selected_row["vcode"])

if not st.button("Run Report", type="primary"):
    st.stop()


# --- Deal Header helpers ---
def fmt_date(x):
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "—"
        s = str(x).strip()
        if s == "":
            return "—"
        return pd.to_datetime(x).date().isoformat()
    except Exception:
        return "—"


def fmt_int(x):
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "—"
        s = str(x).strip()
        if s == "":
            return "—"
        return f"{int(float(x)):,}"
    except Exception:
        return "—"


def fmt_num(x):
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "—"
        s = str(x).strip()
        if s == "":
            return "—"
        return f"{float(x):,.0f}"
    except Exception:
        return "—"


# --- Deal Header ---
st.markdown("### Deal Summary")
st.markdown(
    f"""
    <div style="padding:14px 16px;border:1px solid #e6e6e6;border-radius:12px;background:#fafafa;">
      <div style="font-size:20px;font-weight:700;line-height:1.2;">{selected_row.get('Investment_Name','')}</div>
      <div style="margin-top:6px;color:#555;">
        <span style="margin-right:14px;"><b>vCode:</b> {deal_vcode}</span>
        <span style="margin-right:14px;"><b>InvestmentID:</b> {selected_row.get('InvestmentID','')}</span>
        <span style="margin-right:14px;"><b>Operating Partner:</b> {selected_row.get('Operating_Partner','')}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Asset Type", selected_row.get("Asset_Type", "—") or "—")
c2.metric("Total Units", fmt_int(selected_row.get("Total_Units", "")))
c3.metric("Size (Sqf)", fmt_num(selected_row.get("Size_Sqf", "")))
c4.metric("Lifecycle", selected_row.get("Lifecycle", "—") or "—")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Acquisition Date", fmt_date(selected_row.get("Acquisition_Date", "")))
c6.metric("", "")
c7.metric("", "")
c8.metric("", "")


# ============================================================
# Full forecast for selected deal (kept intact for NOI-forward valuation)
# ============================================================
fc_deal_full = fc[fc["vcode"].astype(str) == str(deal_vcode)].copy()
if fc_deal_full.empty:
    st.error(f"No forecast rows found for selected deal vCode {deal_vcode}.")
    st.stop()

# Full model window present in feed (needed for NOI-forward)
model_start = min(fc_deal_full["event_date"])
model_end_full = max(fc_deal_full["event_date"])


# ============================================================
# Determine Sale_Date
# - Use investment_map.Sale_Date if present
# - Else assume sale date is last day of the 10-year forecast horizon
# ============================================================
def horizon_end_date(start_year: int, horizon_years: int) -> date:
    y = int(start_year) + int(horizon_years) - 1
    return date(y, 12, 31)


sale_date_raw = selected_row.get("Sale_Date", None)
if sale_date_raw is None or (isinstance(sale_date_raw, float) and pd.isna(sale_date_raw)) or str(sale_date_raw).strip() == "":
    sale_date = month_end(horizon_end_date(int(start_year), int(horizon_years)))
else:
    sale_date = month_end(as_date(sale_date_raw))

# Ensure sale_date not before model_start
if sale_date < month_end(model_start):
    sale_date = month_end(model_start)


# ============================================================
# Build modeled loan schedules (existing + planned if qualifies)
# ============================================================
debug_msgs: List[str] = []
loan_sched = pd.DataFrame()
loans: List[Loan] = []

# Existing loans
if mri_loans_raw is not None and not mri_loans_raw.empty:
    mri_loans = load_mri_loans(mri_loans_raw)
    mri_loans = mri_loans[mri_loans["vCode"].astype(str) == str(deal_vcode)].copy()
    loans.extend(build_loans_from_mri_loans(mri_loans))
else:
    debug_msgs.append("MRI_Loans.csv not provided; existing loans will NOT be modeled.")

# Planned loan sizing + planned loan object
planned_dbg = None
planned_new_loan_amt = 0.0
planned_orig_date = None

if mri_supp is not None and not mri_supp.empty:
    ms = mri_supp.copy()
    ms.columns = [str(c).strip() for c in ms.columns]
    if "vCode" not in ms.columns and "vcode" in ms.columns:
        ms = ms.rename(columns={"vcode": "vCode"})
    ms["vCode"] = ms["vCode"].astype(str)

    if (ms["vCode"] == str(deal_vcode)).any():
        if mri_val is None or mri_val.empty:
            debug_msgs.append("MRI_Supp present for this deal, but MRI_Val missing — cannot size planned loan.")
        else:
            supp_row = ms[ms["vCode"] == str(deal_vcode)].iloc[0]
            try:
                planned_new_loan_amt, planned_dbg = size_planned_second_mortgage(inv, fc_deal_full, supp_row, mri_val)
                planned_orig_date = month_end(as_date(supp_row["Orig_Date"]))

                if planned_new_loan_amt > 0:
                    loans.append(planned_loan_as_loan_object(deal_vcode, supp_row, planned_new_loan_amt))
            except Exception as e:
                debug_msgs.append(f"Planned loan sizing failed for this deal: {e}")

# Build schedules for ALL loans through full model_end (to support sale balance deduction)
if loans:
    schedules = []
    for ln in loans:
        s = amortize_monthly_schedule(ln, model_start, model_end_full)
        if not s.empty:
            schedules.append(s)
    if schedules:
        loan_sched = pd.concat(schedules, ignore_index=True)
else:
    debug_msgs.append("No loans found to model for this deal.")


# ============================================================
# Replace forecast debt service with modeled debt service
# - We'll build a modeled version of the deal forecast (for reporting / NOI / DSCR)
# - Then we will zero-out values after sale_date for DISPLAY only
# ============================================================
fc_deal_modeled = fc_deal_full.copy()
fc_deal_modeled = fc_deal_modeled[~fc_deal_modeled["vAccount"].isin(INTEREST_ACCTS | PRINCIPAL_ACCTS)].copy()

if not loan_sched.empty:
    monthly = loan_sched.groupby(["vcode", "event_date"], as_index=False)[["interest", "principal"]].sum()

    add_rows = []
    for _, r in monthly.iterrows():
        dte = r["event_date"]
        intr = float(r["interest"])
        prin = float(r["principal"])

        # only add within available model window in feed (keeps clean)
        if dte < month_end(model_start) or dte > month_end(model_end_full):
            continue

        if intr != 0:
            add_rows.append({
                "vcode": str(deal_vcode),
                "event_date": dte,
                "vAccount": pd.Series([7030]).astype("Int64").iloc[0],
                "mAmount": intr,
                "mAmount_norm": -abs(intr),
                "vSource": "MODEL_LOANS",
                "Pro_Yr": None,
                "Year": pd.Timestamp(dte).year,
                "vAccountType": "Expenses",
            })
        if prin != 0:
            add_rows.append({
                "vcode": str(deal_vcode),
                "event_date": dte,
                "vAccount": pd.Series([7060]).astype("Int64").iloc[0],
                "mAmount": prin,
                "mAmount_norm": -abs(prin),
                "vSource": "MODEL_LOANS",
                "Pro_Yr": None,
                "Year": pd.Timestamp(dte).year,
                "vAccountType": "Liability",
            })

    if add_rows:
        fc_deal_modeled = pd.concat([fc_deal_modeled, pd.DataFrame(add_rows)], ignore_index=True)

# Ensure Year exists (some rows from feed already have it; we recompute safely)
fc_deal_modeled["Year"] = fc_deal_modeled["event_date"].apply(lambda d: pd.Timestamp(d).year).astype("Int64")


# ============================================================
# Capital events: proceeds from refinancing and sale
# ============================================================
capital_events = []

# (A) Refi proceeds if planned loan qualifies
if planned_new_loan_amt and planned_new_loan_amt > 0 and planned_orig_date is not None:
    capital_events.append({
        "vcode": str(deal_vcode),
        "event_date": planned_orig_date,
        "event_type": "Refinancing Proceeds (Planned 2nd Mortgage)",
        "amount": float(NEW_LOAN_NET_PROCEEDS * planned_new_loan_amt),
    })

# Helper: loan balances at a specific date (month-end), summed across loans
def total_loan_balance_at(loan_sched_df: pd.DataFrame, asof_me: date) -> float:
    if loan_sched_df is None or loan_sched_df.empty:
        return 0.0
    s = loan_sched_df.copy()
    s = s[s["event_date"] <= asof_me].copy()
    if s.empty:
        return 0.0
    # last record per LoanID
    last = s.sort_values(["LoanID", "event_date"]).groupby("LoanID", as_index=False).tail(1)
    return float(last["ending_balance"].sum())


# (B) Sale proceeds on sale_date
sale_me = month_end(sale_date)

sale_proceeds = 0.0
sale_dbg = None

if mri_val is not None and not mri_val.empty:
    try:
        # NOI forward 12 months AFTER sale date (full modeled forecast retained)
        noi_12_sale = twelve_month_noi_after_date(fc_deal_modeled, sale_me)
        proj_begin = min(fc_deal_modeled["event_date"])
        cap_rate_sale = projected_cap_rate_at_date(mri_val, str(deal_vcode), proj_begin, sale_me)

        value_sale = (noi_12_sale / cap_rate_sale) if cap_rate_sale != 0 else 0.0
        value_net_selling_cost = value_sale * (1.0 - SELLING_COST_RATE)

        loan_bal_sale = total_loan_balance_at(loan_sched, sale_me)

        sale_proceeds = max(0.0, value_net_selling_cost - loan_bal_sale)

        sale_dbg = {
            "Sale_Date": str(sale_me),
            "NOI_12m_After_Sale": noi_12_sale,
            "CapRate_Sale": cap_rate_sale,
            "Implied_Value": value_sale,
            "Less_Selling_Cost_2pct": value_sale * SELLING_COST_RATE,
            "Value_Net_Selling_Cost": value_net_selling_cost,
            "Less_Loan_Balances": loan_bal_sale,
            "Net_Sale_Proceeds": sale_proceeds,
        }

        capital_events.append({
            "vcode": str(deal_vcode),
            "event_date": sale_me,
            "event_type": "Proceeds from Sale",
            "amount": float(sale_proceeds),
        })
    except Exception as e:
        debug_msgs.append(f"Sale proceeds estimation failed (needs MRI_Val and NOI-forward): {e}")
else:
    debug_msgs.append("MRI_Val missing — cannot estimate sale proceeds.")


cap_events_df = pd.DataFrame(capital_events)
if not cap_events_df.empty:
    cap_events_df["Year"] = cap_events_df["event_date"].apply(lambda d: pd.Timestamp(d).year).astype("Int64")

# ============================================================
# Build period cash for CF and Capital waterfalls
# ============================================================

# Load waterfalls
wf_steps = load_waterfalls(wf)

# Monthly CF cash = monthly FAD from DISPLAY version (already zeroed post-sale for reporting)
fad_monthly = cashflows_monthly_fad(fc_deal_display)
cf_period_cash = fad_monthly.rename(columns={"fad": "cash_available"}).copy()

# Monthly capital cash = capital events by month-end
cap_period_cash = pd.DataFrame(columns=["event_date", "cash_available"])
if cap_events_df is not None and not cap_events_df.empty:
    ce = cap_events_df.copy()
    ce["event_date"] = pd.to_datetime(ce["event_date"]).dt.date
    ce["event_date"] = ce["event_date"].apply(month_end)
    cap_period_cash = ce.groupby("event_date", as_index=False)["amount"].sum().rename(columns={"amount": "cash_available"})

# Ensure we only run through sale month for distributions display (your rule)
cf_period_cash = cf_period_cash[cf_period_cash["event_date"] <= sale_me].copy()
cap_period_cash = cap_period_cash[cap_period_cash["event_date"] <= sale_me].copy()


# ============================================================
# DISPLAY ZEROES AFTER SALE DATE (operating + debt service lines)
# But keep full modeled data for NOI-forward already computed above.
# ============================================================
fc_deal_display = fc_deal_modeled.copy()
after_sale_mask = fc_deal_display["event_date"] > sale_me

# Zero mAmount_norm after sale for operating and debt-related accounts.
# (We keep the rows but set to 0 so the annual table shows zeros post-sale.)
fc_deal_display.loc[after_sale_mask, "mAmount_norm"] = 0.0


# ============================================================
# Annual aggregation display (includes proceeds by year)
# ============================================================
st.subheader("Annual Operating Forecast (Revenues → Funds Available for Distribution)")

proceeds_by_year = None
if not cap_events_df.empty:
    proceeds_by_year = cap_events_df.groupby("Year")["amount"].sum()

annual_df_raw = annual_aggregation_table(
    fc_deal_display,
    int(start_year),
    int(horizon_years),
    proceeds_by_year=proceeds_by_year
)
annual_df = pivot_annual_table(annual_df_raw)
styled = style_annual_table(annual_df)
st.dataframe(styled, use_container_width=True)

st.caption(
    f"Sale Date for reporting: {sale_me.isoformat()} — all operating cash flows and debt service display as 0 after this date. "
    f"Full forecast is still retained internally for NOI-forward valuation (sale and refi sizing)."
)

if debug_msgs:
    with st.expander("Diagnostics"):
        for m in debug_msgs:
            st.write("- " + m)
st.divider()
st.header("Waterfalls & Investor Returns")

# Seed from accounting history for this deal (InvestmentID -> vcode, InvestorID -> PropCode)
seed_states = seed_states_from_accounting(
    acct_raw=acct,      # accounting_feed df
    inv_map=inv,        # investment_map df
    wf_steps=wf_steps,  # loaded waterfalls df (after load_waterfalls)
    target_vcode=deal_vcode
)

# --- CF Waterfall ---
st.subheader("Cash Flow Waterfall (CF_WF)")
cf_alloc, cf_investors = run_waterfall(
    wf_steps=wf_steps,
    vcode=deal_vcode,
    wf_name="CF_WF",
    period_cash=cf_period_cash,
    initial_states=seed_states,
)


if cf_alloc.empty:
    st.info("No CF_WF steps found for this deal in waterfalls.csv.")
else:
    with st.expander("CF_WF Step-by-Step Allocations"):
        df = cf_alloc.copy()
        df["Allocated"] = df["Allocated"].map(lambda x: f"{x:,.0f}")
        df["RemainingAfter"] = df["RemainingAfter"].map(lambda x: f"{x:,.0f}")
        st.dataframe(df, use_container_width=True)

    st.markdown("**CF_WF Investor Summary**")
    if not cf_investors.empty:
        out = cf_investors.copy()
        out["EndingCapitalOutstanding"] = out["EndingCapitalOutstanding"].map(lambda x: f"{x:,.0f}")
        out["UnpaidPrefCompounded"] = out["UnpaidPrefCompounded"].map(lambda x: f"{x:,.0f}")
        out["AccruedPrefCurrentYear"] = out["AccruedPrefCurrentYear"].map(lambda x: f"{x:,.0f}")
        out["TotalDistributions"] = out["TotalDistributions"].map(lambda x: f"{x:,.0f}")
        out["TotalContributions"] = out["TotalContributions"].map(lambda x: f"{x:,.0f}")
        out["XIRR"] = out["XIRR"].map(lambda r: "" if r is None else f"{r*100:,.2f}%")
        st.dataframe(out, use_container_width=True)


# --- Capital Waterfall ---
st.subheader("Capital Waterfall (Cap_WF)")
cap_alloc, cap_investors = run_waterfall(
    wf_steps=wf_steps,
    vcode=deal_vcode,
    wf_name="Cap_WF",
    period_cash=cap_period_cash,
    initial_states=seed_states,  # same dict continues forward
)


if cap_alloc.empty:
    st.info("No Cap_WF steps found for this deal in waterfalls.csv.")
else:
    with st.expander("Cap_WF Step-by-Step Allocations"):
        df = cap_alloc.copy()
        df["Allocated"] = df["Allocated"].map(lambda x: f"{x:,.0f}")
        df["RemainingAfter"] = df["RemainingAfter"].map(lambda x: f"{x:,.0f}")
        st.dataframe(df, use_container_width=True)

    st.markdown("**Cap_WF Investor Summary**")
    if not cap_investors.empty:
        out = cap_investors.copy()
        out["EndingCapitalOutstanding"] = out["EndingCapitalOutstanding"].map(lambda x: f"{x:,.0f}")
        out["UnpaidPrefCompounded"] = out["UnpaidPrefCompounded"].map(lambda x: f"{x:,.0f}")
        out["AccruedPrefCurrentYear"] = out["AccruedPrefCurrentYear"].map(lambda x: f"{x:,.0f}")
        out["TotalDistributions"] = out["TotalDistributions"].map(lambda x: f"{x:,.0f}")
        out["TotalContributions"] = out["TotalContributions"].map(lambda x: f"{x:,.0f}")
        out["XIRR"] = out["XIRR"].map(lambda r: "" if r is None else f"{r*100:,.2f}%")
        st.dataframe(out, use_container_width=True)


# ============================================================
# Loan schedule display (formatted)
# ============================================================
if loan_sched is not None and not loan_sched.empty:
    with st.expander("Loan Schedule (modeled) — formatted"):
        show = loan_sched.sort_values(["LoanID", "event_date"]).copy()

        show_disp = show.copy()
        show_disp["rate"] = (show_disp["rate"] * 100.0).map(lambda x: f"{x:,.2f}%")
        for c in ["interest", "principal", "payment", "ending_balance"]:
            show_disp[c] = show_disp[c].map(lambda x: f"{x:,.0f}")

        st.dataframe(
            show_disp[["LoanID", "event_date", "rate", "interest", "principal", "payment", "ending_balance"]],
            use_container_width=True
        )

        totals = show.groupby("event_date", as_index=False)[["interest", "principal"]].sum()
        totals["debt_service"] = totals["interest"] + totals["principal"]

        totals_disp = totals.copy()
        for c in ["interest", "principal", "debt_service"]:
            totals_disp[c] = totals_disp[c].map(lambda x: f"{x:,.0f}")

        st.subheader("Total Modeled Debt Service by Month (all loans)")
        st.dataframe(totals_disp.sort_values("event_date"), use_container_width=True)


# ============================================================
# Planned second mortgage sizing detail (ONLY when selected deal has MRI_Supp)
# ============================================================
if mri_supp is not None and not mri_supp.empty:
    ms = mri_supp.copy()
    ms.columns = [str(c).strip() for c in ms.columns]
    if "vCode" not in ms.columns and "vcode" in ms.columns:
        ms = ms.rename(columns={"vcode": "vCode"})
    ms["vCode"] = ms["vCode"].astype(str)

    if (ms["vCode"] == str(deal_vcode)).any():
        st.subheader("Planned Second Mortgage — Sizing Detail (Selected Deal)")

        if mri_val is None or mri_val.empty or planned_dbg is None:
            st.info("MRI_Supp present, but sizing detail unavailable (ensure MRI_Val is uploaded and sizing succeeds).")
        else:
            def fmt_money(x):
                try:
                    return f"{float(x):,.0f}"
                except Exception:
                    return x

            def fmt_pct(x):
                try:
                    return f"{float(x) * 100:,.2f}%"
                except Exception:
                    return x

            pretty = {}
            for k, v in planned_dbg.items():
                if k in {"CapRate"}:
                    pretty[k] = fmt_pct(v)
                elif k in {
                    "NOI_12m", "ProjectedValue", "MaxAdd_LTV", "ExistingDS_12m_fromForecast",
                    "MaxAddDS_12m", "MaxAdd_DSCR_Principal", "NewLoanAmt", "ExistingDebtAssumed"
                }:
                    pretty[k] = fmt_money(v)
                else:
                    pretty[k] = v

            st.json(pretty)


# ============================================================
# Capital Events (to feed Capital Waterfall)
# ============================================================
with st.expander("Capital Events (Proceeds available to Capital Waterfall)"):
    if cap_events_df.empty:
        st.write("No capital events were generated for this deal.")
    else:
        disp = cap_events_df.sort_values("event_date").copy()
        disp2 = disp.copy()
        disp2["amount"] = disp2["amount"].map(lambda x: f"{x:,.0f}")
        st.dataframe(disp2[["event_date", "event_type", "amount"]], use_container_width=True)

    if sale_dbg is not None:
        st.markdown("**Sale Proceeds Detail**")
        def _fmt_money(x):
            try:
                return f"{float(x):,.0f}"
            except Exception:
                return x
        def _fmt_pct(x):
            try:
                return f"{float(x)*100:,.2f}%"
            except Exception:
                return x
        sale_pretty = {}
        for k, v in sale_dbg.items():
            if k in {"CapRate_Sale"}:
                sale_pretty[k] = _fmt_pct(v)
            elif k in {"NOI_12m_After_Sale", "Implied_Value", "Less_Selling_Cost_2pct", "Value_Net_Selling_Cost", "Less_Loan_Balances", "Net_Sale_Proceeds"}:
                sale_pretty[k] = _fmt_money(v)
            else:
                sale_pretty[k] = v
        st.json(sale_pretty)

st.success("Report generated successfully.")

