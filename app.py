# app.py
# Waterfall + XIRR Forecast
# Step: model debt service (interest/principal/balances) for all existing + planned loans
#
# Includes:
#  - Modeled loan schedules (existing loans from MRI_Loans.csv)
#  - Optional planned second mortgage sizing details (MRI_Supp.csv + MRI_Val.csv)
#  - Replaces forecast interest/principal lines with modeled values
#  - Annual aggregation table formatted (years across columns, right-justified)
#  - Loan schedule display formatted with commas + rate as % (2 decimals)
#  - Dropdown to display planned loan sizing math for deals with MRI_Supp entries
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
#
# NOTE: This file focuses on operating forecast + loan modeling and planned loan sizing debug.
#       Waterfall execution and XIRR integration are built elsewhere in your app; this is the
#       correctly functioning debt service engine + reporting shell.

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import pandas as pd
import streamlit as st
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
    """
    Return month-ends from month_end(start_d) through month_end(end_d), inclusive.
    """
    start_me = month_end(start_d)
    end_me = month_end(end_d)
    if end_me < start_me:
        return []
    rng = pd.date_range(start=start_me, end=end_me, freq="M")
    return [x.date() for x in rng]


def annual_360_to_365(r_annual: float) -> float:
    # r_annual in decimal form (e.g., 0.06 for 6%)
    return r_annual * (365.0 / 360.0)


# ============================================================
# XIRR
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
    coa headers provided:
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
    MRI_Loans headers:
      vCode, LoanID, vPropertyName, mOrigLoanAmt, iAmortTerm, mNominalPenalty, iLoanTerm,
      vIntType, vIndex, nRate, vSpread, nFloor, vIntRatereset, ..., dtEvent
    dtEvent contains loan maturity date
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

            # last-payment cleanup
            if principal > bal:
                principal = bal
                payment = interest + principal

        bal = max(0.0, bal - principal)

        rows.append({
            "vcode": loan.vcode,
            "LoanID": loan.loan_id,
            "event_date": dte,
            "rate": float(r_annual),  # store 365-basis decimal for display consistency
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
def projected_cap_rate_at_orig(mri_val: pd.DataFrame, vcode: str, proj_begin: date, orig_date: date) -> float:
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

    years = max(0.0, (pd.Timestamp(orig_date) - pd.Timestamp(proj_begin)).days / 365.0)
    return fcap + 0.0005 * years


def twelve_month_noi_after_date(fc_deal: pd.DataFrame, orig_date: date) -> float:
    start_me = month_end(orig_date)
    end_me = add_months(start_me, 12)
    f = fc_deal[(fc_deal["event_date"] >= start_me) & (fc_deal["event_date"] < end_me)].copy()
    if f.empty:
        return 0.0
    is_rev = f["vAccount"].isin(GROSS_REVENUE_ACCTS | CONTRA_REVENUE_ACCTS)
    is_exp = f["vAccount"].isin(EXPENSE_ACCTS)
    return float(f.loc[is_rev | is_exp, "mAmount_norm"].sum())


def existing_debt_service_12m_after(fc_deal: pd.DataFrame, orig_date: date) -> float:
    start_me = month_end(orig_date)
    end_me = add_months(start_me, 12)
    f = fc_deal[(fc_deal["event_date"] >= start_me) & (fc_deal["event_date"] < end_me)].copy()
    if f.empty:
        return 0.0
    ds = f.loc[f["vAccount"].isin(INTEREST_ACCTS | PRINCIPAL_ACCTS), "mAmount_norm"].sum()
    return float(abs(ds))


def ds_first_12_months_for_principal(principal: float, rate: float, term_years: int, amort_years: int, io_years: float, orig_date: date) -> float:
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


def size_planned_second_mortgage(inv: pd.DataFrame, fc_deal: pd.DataFrame, mri_supp_row: pd.Series, mri_val: pd.DataFrame) -> Tuple[float, dict]:
    orig_date = as_date(mri_supp_row["Orig_Date"])
    term = int(float(mri_supp_row["Term"]))
    amort = int(float(mri_supp_row["Amort"]))
    io_years = float(mri_supp_row["I/O Period"])
    dscr_req = float(mri_supp_row["DSCR"])
    ltv = float(mri_supp_row["LTV"])
    rate = float(mri_supp_row["Rate"])

    if ltv > 1.5:
        ltv = ltv / 100.0

    proj_begin = min(fc_deal["event_date"])
    noi_12 = twelve_month_noi_after_date(fc_deal, orig_date)
    cap_rate = projected_cap_rate_at_orig(mri_val, str(fc_deal["vcode"].iloc[0]), proj_begin, orig_date)

    inv2 = inv.copy()
    inv2.columns = [str(c).strip() for c in inv2.columns]
    inv2["vcode"] = inv2["vcode"].astype(str)
    inv_row = inv2[inv2["vcode"] == str(fc_deal["vcode"].iloc[0])]

    n_cost_sale = float(inv_row.iloc[0].get("nCostSaleRate", 0.0) or 0.0) if not inv_row.empty else 0.0
    projected_value = (noi_12 / cap_rate) * (1.0 - n_cost_sale) if cap_rate != 0 else 0.0

    # Placeholder until we wire modeled balances at orig_date:
    existing_bal = 0.0

    max_total_debt = projected_value * ltv
    max_add_ltv = max(0.0, max_total_debt - existing_bal)

    max_total_ds = (noi_12 / dscr_req) if dscr_req > 0 else 0.0
    existing_ds = existing_debt_service_12m_after(fc_deal, orig_date)
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


def planned_loan_debug_table(inv: pd.DataFrame, fc: pd.DataFrame, mri_supp: pd.DataFrame, mri_val: pd.DataFrame) -> pd.DataFrame:
    if mri_supp is None or mri_supp.empty:
        return pd.DataFrame()

    ms = mri_supp.copy()
    ms.columns = [str(c).strip() for c in ms.columns]
    if "vCode" not in ms.columns and "vcode" in ms.columns:
        ms = ms.rename(columns={"vcode": "vCode"})
    ms["vCode"] = ms["vCode"].astype(str)

    out_rows = []
    for vcode in sorted(ms["vCode"].dropna().unique()):
        fc_deal = fc[fc["vcode"].astype(str) == str(vcode)].copy()
        if fc_deal.empty:
            continue

        row = ms[ms["vCode"] == str(vcode)].iloc[0]
        try:
            new_amt, dbg = size_planned_second_mortgage(inv, fc_deal, row, mri_val)
            out_rows.append({"vCode": str(vcode), **dbg})
        except Exception as e:
            out_rows.append({"vCode": str(vcode), "Error": str(e)})

    return pd.DataFrame(out_rows)


# ============================================================
# ANNUAL AGGREGATION
# ============================================================
def annual_aggregation_table(fc_deal: pd.DataFrame, start_year: int, horizon_years: int) -> pd.DataFrame:
    years = list(range(int(start_year), int(start_year) + int(horizon_years)))
    f = fc_deal[fc_deal["Year"].isin(years)].copy()

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

    return out.reset_index().fillna(0.0)


def pivot_annual_table(df: pd.DataFrame) -> pd.DataFrame:
    wide = df.set_index("Year").T
    wide.index.name = "Line Item"
    desired_order = [
        "Revenues", "Expenses", "NOI",
        "Interest", "Principal", "Total Debt Service",
        "Capital Expenditures", "Excluded Accounts",
        "Funds Available for Distribution", "Debt Service Coverage Ratio",
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
            {"selector": "th", "props": [("text-align", "left"), ("width", "220px")]},
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

    # Line above FAD (last line before Funds Available), and bold FAD
    if "Funds Available for Distribution" in df.index:
        fad_idx = df.index.get_loc("Funds Available for Distribution")
        if fad_idx > 0:
            prev_row = df.index[fad_idx - 1]
            styler = styler.set_properties(subset=pd.IndexSlice[[prev_row], :], **{"border-bottom": "2px solid black"})
        styler = styler.set_properties(subset=pd.IndexSlice[["Funds Available for Distribution"], :], **{"font-weight": "bold"})

    # Light line above DSCR
    if "Debt Service Coverage Ratio" in df.index:
        styler = styler.set_properties(subset=pd.IndexSlice[["Debt Service Coverage Ratio"], :], **{"border-top": "1px solid #999"})

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
        st.caption("Debt modeling: MRI_Loans.csv (recommended). Planned loan sizing: MRI_Supp.csv + MRI_Val.csv (optional).")
    else:
        uploads["investment_map"] = st.file_uploader("investment_map.csv", type="csv")
        uploads["waterfalls"] = st.file_uploader("waterfalls.csv", type="csv")
        uploads["coa"] = st.file_uploader("coa.csv", type="csv")
        uploads["accounting_feed"] = st.file_uploader("accounting_feed.csv", type="csv")
        uploads["forecast_feed"] = st.file_uploader("forecast_feed.csv", type="csv")

        st.divider()
        st.subheader("Debt Modeling Inputs")
        uploads["MRI_Loans"] = st.file_uploader("MRI_Loans.csv (existing loans)", type="csv")
        uploads["MRI_Supp"] = st.file_uploader("MRI_Supp.csv (planned 2nd mortgages)", type="csv")
        uploads["MRI_Val"] = st.file_uploader("MRI_Val.csv (cap rates for sizing)", type="csv")

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

deal = st.selectbox("Select Deal", sorted(inv["vcode"].dropna().unique().tolist()))

if not st.button("Run Report", type="primary"):
    st.stop()


# ============================================================
# Model window from forecast
# ============================================================
fc_deal_base = fc[fc["vcode"].astype(str) == str(deal)].copy()
if fc_deal_base.empty:
    st.error(f"No forecast rows found for deal {deal}.")
    st.stop()

model_start = min(fc_deal_base["event_date"])
model_end = max(fc_deal_base["event_date"])


# ============================================================
# Build modeled loan schedules (existing loans)
# ============================================================
debug_msgs: List[str] = []
loan_sched = pd.DataFrame()
loans: List[Loan] = []

if mri_loans_raw is not None and not mri_loans_raw.empty:
    mri_loans = load_mri_loans(mri_loans_raw)
    mri_loans = mri_loans[mri_loans["vCode"].astype(str) == str(deal)].copy()
    loans.extend(build_loans_from_mri_loans(mri_loans))
else:
    debug_msgs.append("MRI_Loans.csv not provided; existing loans will NOT be modeled.")

if loans:
    schedules = []
    for ln in loans:
        s = amortize_monthly_schedule(ln, model_start, model_end)
        if not s.empty:
            schedules.append(s)
    if schedules:
        loan_sched = pd.concat(schedules, ignore_index=True)
else:
    debug_msgs.append("No loans found to model for this deal.")


# ============================================================
# Replace forecast debt service with modeled debt service
# ============================================================
fc_deal = fc_deal_base.copy()
fc_deal = fc_deal[~fc_deal["vAccount"].isin(INTEREST_ACCTS | PRINCIPAL_ACCTS)].copy()

if not loan_sched.empty:
    monthly = loan_sched.groupby(["vcode", "event_date"], as_index=False)[["interest", "principal"]].sum()

    add_rows = []
    for _, r in monthly.iterrows():
        dte = r["event_date"]
        intr = float(r["interest"])
        prin = float(r["principal"])

        if intr != 0:
            add_rows.append({
                "vcode": str(deal),
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
                "vcode": str(deal),
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
        fc_deal = pd.concat([fc_deal, pd.DataFrame(add_rows)], ignore_index=True)

fc_deal["Year"] = fc_deal["event_date"].apply(lambda d: pd.Timestamp(d).year).astype("Int64")


# ============================================================
# Annual aggregation display
# ============================================================
st.subheader("Annual Operating Forecast (Revenues → Funds Available for Distribution)")

annual_df_raw = annual_aggregation_table(fc_deal, int(start_year), int(horizon_years))
annual_df = pivot_annual_table(annual_df_raw)
styled = style_annual_table(annual_df)
st.dataframe(styled, use_container_width=True)

st.caption(
    "Debt service is MODEL-DRIVEN from MRI_Loans: interest/principal forecast rows are removed and replaced with "
    "modeled values. Variable loans are treated as interest-only for entire term."
)

if debug_msgs:
    with st.expander("Diagnostics"):
        for m in debug_msgs:
            st.write("- " + m)


# ============================================================
# Loan schedule display (formatted)
# ============================================================
if not loan_sched.empty:
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
# Planned loan sizing dropdown (MRI_Supp)
# ============================================================
if mri_supp is not None and not mri_supp.empty:
    st.subheader("Planned Second Mortgage — Maximum Additional Loan Size (MRI_Supp)")

    if mri_val is None or mri_val.empty:
        st.info("Upload/provide MRI_Val.csv to compute cap-rate based sizing details.")
    else:
        dbg_table = planned_loan_debug_table(inv, fc, mri_supp, mri_val)
        if dbg_table.empty:
            st.info("No planned loan rows could be evaluated (missing forecast rows or inputs).")
        else:
            vchoices = dbg_table["vCode"].dropna().astype(str).unique().tolist()
            chosen = st.selectbox("Select property for sizing detail", sorted(vchoices), key="planned_sizing_select")

            row = dbg_table[dbg_table["vCode"].astype(str) == str(chosen)].iloc[0].to_dict()

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
            for k, v in row.items():
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

st.success("Report generated successfully.")
