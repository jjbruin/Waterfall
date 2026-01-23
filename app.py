# app.py
# Waterfall + XIRR Forecast
# Step: model debt service (interest/principal/balances) for all existing + planned loans
#
# Updates per latest clarifications:
#  - Base index rates:
#       * SOFR/LIBOR -> 0.043
#       * WSJ        -> 0.075
#  - Variable-rate loans:
#       * Assume NO amortization (interest-only for entire term)
#       * Rate logic:
#            base = index_base (per above)
#            if cap exists (>0): effective_index = min(base, cap)
#            else:              effective_index = base
#            annual_rate = max(floor, effective_index + spread)
#       * (cap applies to the index component before spread, per your instruction)
#
# Inputs:
#  - investment_map.csv (required)
#  - waterfalls.csv (required)
#  - coa.csv (required)
#  - accounting_feed.csv (required)
#  - forecast_feed.csv (required)
#  - MRI_Loans.csv (optional but recommended to model existing loans)
#  - MRI_Supp.csv + MRI_Val.csv (optional) to size planned future second mortgages
#
# Key behavior:
#  - Forecast feed debt service can be wrong; model computes debt service monthly.
#  - Model replaces forecast debt service lines (interest/principal) with computed values.
#  - Monthly schedule includes ending balances so we can pick balances at sale later.

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
    # r_annual is a decimal annual rate (e.g., 0.06 for 6%)
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
      vCode, LoanID, mOrigLoanAmt, iAmortTerm, mNominalPenalty, iLoanTerm, vIntType, vIndex,
      nRate, vSpread, nFloor, vIntRatereset, dtMaturity OR dtEvent, ...
    dtEvent contains loan maturity date (per user)
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
          annual_rate = nRate

        Variable (per your rules):
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

        # Your requested change:
        annual_rate = max(floor + spread, effective_index + spread)
        return annual_rate

def rate_for_month(self) -> float:
    """
    Variable logic (updated):
      base = {SOFR/LIBOR:0.043, WSJ:0.075, else:0.043}
      if cap exists (>0): effective_index = min(base, cap)
      else:              effective_index = base
      annual_rate = max(floor + spread, effective_index + spread)
    Fixed logic:
      annual_rate = nRate
    """
    idx = (self.index_name or "").strip().upper()
    floor = self._as_decimal(self.floor)
    spread = self._as_decimal(self.spread)
    cap = self._as_decimal(self.cap)

    if not self.is_variable():
        return self._as_decimal(self.fixed_rate)

    base = INDEX_BASE_RATES.get(idx, 0.043)
    effective_index = min(base, cap) if cap > 0 else base

    r = max(floor + spread, effective_index + spread)
    return r

def amortize_monthly_schedule(loan: Loan, schedule_start: date, schedule_end: date) -> pd.DataFrame:
    """
    Month-end schedule.

    Fixed loans:
      - IO months: payment = interest
      - After IO: level-payment amortization (constant payment), unless maturity cuts it off (balloon possible)

    Variable loans (per your rules):
      - No amortization: interest-only for entire term
    """
    all_dates = month_ends_between(loan.orig_date, max(schedule_end, loan.maturity_date))
    if not all_dates:
        return pd.DataFrame()

   r_annual_360 = loan.rate_for_month()
   r_annual = annual_360_to_365(r_annual_360)   # convert to 365-basis annual
   r_m = r_annual / 12.0


    term_m = int(loan.loan_term_m) if loan.loan_term_m and loan.loan_term_m > 0 else len(all_dates)
    amort_m = int(loan.amort_term_m) if loan.amort_term_m and loan.amort_term_m > 0 else term_m
    io_m = int(loan.io_months) if loan.io_months and loan.io_months > 0 else 0

    # Variable: no amortization (interest-only for entire term)
    if loan.is_variable():
        io_m = term_m

    # remaining amort months after IO (used to set level payment ONCE)
    amort_after_io = max(1, amort_m - io_m)

    bal = float(loan.orig_amount)

    # We will set this at the first amort month and keep it constant thereafter (fixed loans only)
    level_payment: Optional[float] = None

    rows = []
    for i, dte in enumerate(all_dates, start=1):
        if dte > loan.maturity_date:
            break

        interest = bal * r_m

        if i <= io_m:
            # interest-only months (and all months for variable loans)
            payment = interest
            principal = 0.0
        else:
            # amortizing phase (fixed loans only)
            if level_payment is None:
                # compute once based on balance at amort start
                if r_m == 0:
                    level_payment = bal / amort_after_io
                else:
                    level_payment = bal * r_m / (1 - (1 + r_m) ** (-amort_after_io))

            payment = level_payment
            principal = max(0.0, payment - interest)

            # last-payment cleanup: don't pay more principal than remaining balance
            if principal > bal:
                principal = bal
                payment = interest + principal

        bal = max(0.0, bal - principal)

        rows.append({
            "vcode": loan.vcode,
            "LoanID": loan.loan_id,
            "event_date": dte,
            "rate": r_annual,
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
        "Excluded Accounts", "Capital Expenditures",
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

    styler = styler.set_table_styles(
        [
            {"selector": "th", "props": [("text-align", "left"), ("width", "220px")]},
            {"selector": "td", "props": [("text-align", "right"), ("width", "140px")]},
        ],
        overwrite=False,
    )

    if "Expenses" in df.index:
        styler = styler.set_properties(subset=pd.IndexSlice[["Expenses"], :], **{"text-decoration": "underline"})

    if "NOI" in df.index:
        styler = styler.set_properties(
            subset=pd.IndexSlice[["NOI"], :],
            **{"border-bottom": "3px double black", "font-weight": "bold"}
        )

    if "Funds Available for Distribution" in df.index:
        fad_idx = df.index.get_loc("Funds Available for Distribution")
        if fad_idx > 0:
            prev_row = df.index[fad_idx - 1]
            styler = styler.set_properties(subset=pd.IndexSlice[[prev_row], :], **{"border-bottom": "2px solid black"})
        styler = styler.set_properties(subset=pd.IndexSlice[["Funds Available for Distribution"], :], **{"font-weight": "bold"})

    if "Debt Service Coverage Ratio" in df.index:
        styler = styler.set_properties(subset=pd.IndexSlice[["Debt Service Coverage Ratio"], :], **{"border-top": "1px solid #999"})

    return styler


# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(layout="wide")
st.title("Waterfall + XIRR Forecast")

CLOUD = is_streamlit_cloud()

with st.sidebar:
    st.header("Data Source")

    if CLOUD:
        mode = "Upload CSVs"
        st.info("Running on Streamlit Cloud — local folders are disabled. Please upload CSVs.")
    else:
        mode = st.radio("Load data from:", ["Local folder", "Upload CSVs"], index=0)

    folder = None
    uploads = {}

    if mode == "Local folder":
        folder = st.text_input("Data folder path", placeholder=r"C:\Path\To\Data")
        st.caption("Required: investment_map.csv, waterfalls.csv, coa.csv, accounting_feed.csv, forecast_feed.csv")
        st.caption("Debt modeling: MRI_Loans.csv (recommended).")
    else:
        uploads["investment_map"] = st.file_uploader("investment_map.csv", type="csv")
        uploads["waterfalls"] = st.file_uploader("waterfalls.csv", type="csv")
        uploads["coa"] = st.file_uploader("coa.csv", type="csv")
        uploads["accounting_feed"] = st.file_uploader("accounting_feed.csv", type="csv")
        uploads["forecast_feed"] = st.file_uploader("forecast_feed.csv", type="csv")

        st.divider()
        st.subheader("Debt Modeling Inputs")
        uploads["MRI_Loans"] = st.file_uploader("MRI_Loans.csv (existing loans)", type="csv")

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

    inv.columns = [str(c).strip() for c in inv.columns]
    inv["vcode"] = inv["vcode"].astype(str)

    return inv, wf, acct, fc, mri_loans_raw


inv, wf, acct, fc, mri_loans_raw = load_inputs()

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

# Remove existing forecast interest/principal rows (replace with modeled)
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

# Ensure Year exists
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

if not loan_sched.empty:
    with st.expander("Loan Schedule (modeled) — monthly interest, principal, ending balance"):
        show = loan_sched.sort_values(["LoanID", "event_date"]).copy()
        st.dataframe(show, use_container_width=True)

        totals = loan_sched.groupby("event_date", as_index=False)[["interest", "principal"]].sum()
        totals["debt_service"] = totals["interest"] + totals["principal"]
        st.subheader("Total Modeled Debt Service by Month (all loans)")
        st.dataframe(totals.sort_values("event_date"), use_container_width=True)

st.success("Report generated successfully.")
