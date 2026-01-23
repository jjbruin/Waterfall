# app.py
# Waterfall + XIRR Forecast
# Adds planned future second mortgages from MRI_Supp.csv:
#   - sizes the new loan via Value/LTV and DSCR tests
#   - builds monthly amort schedule (IO then amortizing)
#   - appends interest/principal rows to the forecast feed for reporting
#
# NOTE: This version focuses on integrating the new debt into the forecast + annual table.
#       (Capital waterfall application is the next step after we get debt mechanics right.)

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Tuple, Optional
from pathlib import Path

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

# Explicit account definitions (NO iNOI)
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

# Existing forecast debt/capex/excluded accounts
INTEREST_ACCTS = {5190, 7030}
PRINCIPAL_ACCTS = {7060}
CAPEX_ACCTS = {7050}
OTHER_EXCLUDED_ACCTS = {4050, 5220, 5210, 5195, 7065, 5120, 5130, 5400}
ALL_EXCLUDED = INTEREST_ACCTS | PRINCIPAL_ACCTS | CAPEX_ACCTS | OTHER_EXCLUDED_ACCTS

# Existing loans from ISBS_Download.csv (liability balances)
EXISTING_LOAN_BAL_ACCTS = {2150, 2152, 2210}


# ============================================================
# DATE HELPERS
# ============================================================
def as_date(s) -> date:
    return pd.to_datetime(s).date()


def month_end(d: date) -> date:
    # move to end of month using pandas
    return (pd.Timestamp(d) + pd.offsets.MonthEnd(0)).date()


def add_months(d: date, months: int) -> date:
    return (pd.Timestamp(d) + pd.DateOffset(months=months)).date()


def month_ends_between(start_d: date, months: int) -> List[date]:
    # inclusive start at month end of start_d, for N months
    start_me = month_end(start_d)
    rng = pd.date_range(start= start_me, periods=months, freq="M")
    return [x.date() for x in rng]


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
# STATE (scaffolding for later waterfall execution)
# ============================================================
@dataclass
class PartnerState:
    principal: float = 0.0
    pref_accrued: float = 0.0
    pref_capitalized: float = 0.0
    irr_cashflows: List[Tuple[date, float]] = field(default_factory=list)

    def base(self) -> float:
        return self.principal + self.pref_capitalized


@dataclass
class DealState:
    vcode: str
    last_event_date: date
    partners: Dict[str, PartnerState] = field(default_factory=dict)


# ============================================================
# LOADERS (COA / FORECAST)
# ============================================================
def load_coa(df: pd.DataFrame) -> pd.DataFrame:
    """
    coa.csv headers:
      vcode, vdescription, vtype, iNOI, vMisc, vAccountType
    Join rule:
      coa.vcode == forecast_feed.vAccount == accounting_feed.TypeID
    iNOI is ignored.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "vcode" not in df.columns:
        raise ValueError("coa.csv is missing required column: vcode")

    df = df.rename(columns={"vcode": "vAccount"})
    df["vAccount"] = pd.to_numeric(df["vAccount"], errors="coerce").astype("Int64")

    if "vAccountType" not in df.columns:
        df["vAccountType"] = ""
    df["vAccountType"] = df["vAccountType"].fillna("").astype(str).str.strip()

    return df[["vAccount", "vAccountType"]]


def normalize_forecast_signs(fc: pd.DataFrame) -> pd.DataFrame:
    """
    Deal-agnostic normalization:
      - Gross revenue accounts: +abs(mAmount)
      - Contra-revenue (vacancy/concessions): -abs(mAmount)
      - Expense accounts: -abs(mAmount)
      - Interest/Principal/Capex/Other excluded: -abs(mAmount)
      - Other accounts: leave as-is (for future expansion)
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
    """
    Forecast feed columns:
      Vcode, dtEntry, vSource, vAccount, mAmount, Year, Qtr, Date, Pro_Yr
    """
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


# ============================================================
# NEW: PLANNED SECOND MORTGAGE LOGIC
# ============================================================
def require_cols(df: pd.DataFrame, cols: List[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def get_projection_begin_date(fc_deal: pd.DataFrame) -> date:
    # Use earliest forecast event_date for the deal as "projection begin"
    return min(fc_deal["event_date"])


def projected_cap_rate_at_orig(mri_val: pd.DataFrame, vcode: str, proj_begin: date, orig_date: date) -> float:
    """
    cap rate = most recent fCapRate (from MRI_Val) + 0.0005 per year between proj_begin and orig_date
    """
    require_cols(mri_val, ["vcode", "fCapRate"], "MRI_Val.csv")
    dv = mri_val.copy()
    dv["vcode"] = dv["vcode"].astype(str)

    sub = dv[dv["vcode"] == str(vcode)].copy()
    if sub.empty:
        raise ValueError(f"MRI_Val has no rows for vcode {vcode}")

    # "most recent" — if MRI_Val has a date column, use it; else take max fCapRate row order as last
    if "dtVal" in sub.columns:
        sub["dtVal"] = pd.to_datetime(sub["dtVal"], errors="coerce").dt.date
        sub = sub.sort_values("dtVal")
    fcap = float(sub["fCapRate"].dropna().iloc[-1])

    years = max(0.0, (pd.Timestamp(orig_date) - pd.Timestamp(proj_begin)).days / 365.0)
    return fcap + 0.0005 * years


def twelve_month_noi_after_date(fc_deal: pd.DataFrame, orig_date: date) -> float:
    """
    NOI for the 12 months FOLLOWING orig_date: sum monthly (revenues+expenses) for 12 month-ends.
    We rely on normalized mAmount_norm.
    """
    start_me = month_end(orig_date)
    end_me = add_months(start_me, 12)  # end exclusive

    f = fc_deal[(fc_deal["event_date"] >= start_me) & (fc_deal["event_date"] < end_me)].copy()
    if f.empty:
        return 0.0

    is_rev = f["vAccount"].isin(GROSS_REVENUE_ACCTS | CONTRA_REVENUE_ACCTS)
    is_exp = f["vAccount"].isin(EXPENSE_ACCTS)

    noi = f.loc[is_rev | is_exp, "mAmount_norm"].sum()
    return float(noi)


def existing_loan_balance_at_orig(
    isbs: pd.DataFrame,
    fc_deal: pd.DataFrame,
    vcode: str,
    proj_begin: date,
    orig_date: date
) -> float:
    """
    Start from Interim BS balances for loan accounts, then subtract principal paid between proj_begin and orig_date.
    """
    require_cols(isbs, ["vcode", "vAccount", "mAmount", "vSource"], "ISBS_Download.csv")
    d = isbs.copy()
    d["vcode"] = d["vcode"].astype(str)
    d["vAccount"] = pd.to_numeric(d["vAccount"], errors="coerce").astype("Int64")
    d["mAmount"] = pd.to_numeric(d["mAmount"], errors="coerce").fillna(0.0)

    bal = d[(d["vcode"] == str(vcode)) & (d["vSource"].astype(str) == "Interim BS") & (d["vAccount"].isin(EXISTING_LOAN_BAL_ACCTS))]["mAmount"].sum()

    # Treat liability balance as positive magnitude
    opening = float(abs(bal))

    # principal paid between proj_begin and orig_date (forecast already normalized; principal is negative outflow)
    f = fc_deal[(fc_deal["event_date"] >= proj_begin) & (fc_deal["event_date"] < orig_date)].copy()
    prin_paid = float(abs(f.loc[f["vAccount"].isin(PRINCIPAL_ACCTS), "mAmount_norm"].sum()))

    return max(0.0, opening - prin_paid)


def existing_debt_service_12m_after(fc_deal: pd.DataFrame, orig_date: date) -> float:
    """
    Existing debt service (interest+principal) for 12 months after orig_date from forecast.
    """
    start_me = month_end(orig_date)
    end_me = add_months(start_me, 12)
    f = fc_deal[(fc_deal["event_date"] >= start_me) & (fc_deal["event_date"] < end_me)].copy()
    if f.empty:
        return 0.0

    ds = f.loc[f["vAccount"].isin(INTEREST_ACCTS | PRINCIPAL_ACCTS), "mAmount_norm"].sum()
    return float(abs(ds))


def amort_schedule_monthly(
    principal: float,
    rate_annual: float,
    term_years: int,
    amort_years: int,
    io_years: float,
    orig_date: date
) -> pd.DataFrame:
    """
    Create monthly schedule from orig_date month-end for term_years*12 payments.
    IO for io_years (can be fractional, we'll round to months).
    After IO, amortize remaining balance over amort_years.

    Returns dataframe with columns: event_date, interest, principal, balance
    All amounts are POSITIVE magnitudes (we'll sign-normalize later).
    """
    r = float(rate_annual)
    if r > 1.0:
        r = r / 100.0
    r_m = r / 12.0

    term_m = int(round(term_years * 12))
    io_m = int(round(float(io_years) * 12))
    amort_m = int(round(amort_years * 12))
    if amort_m <= 0:
        raise ValueError("Amort must be > 0 years")

    pay_dates = month_ends_between(orig_date, term_m)
    bal = float(principal)

    rows = []
    for i, pdte in enumerate(pay_dates, start=1):
        interest = bal * r_m
        if i <= io_m:
            prin = 0.0
            pmt = interest
        else:
            # amortizing payment based on remaining amort schedule (standard mortgage)
            # payment uses original amort_m, not "remaining", which matches typical note payment setting
            # (If you want re-cast after IO, we can switch to remaining term.)
            if r_m == 0:
                pmt = bal / max(1, amort_m)
            else:
                pmt = bal * (r_m) / (1 - (1 + r_m) ** (-amort_m))
            prin = max(0.0, pmt - interest)

        bal = max(0.0, bal - prin)
        rows.append({"event_date": pdte, "interest": interest, "principal": prin, "balance": bal})

    return pd.DataFrame(rows)


def ds_first_12_months_for_principal(
    principal: float,
    rate: float,
    term: int,
    amort: int,
    io_period: float,
    orig_date: date
) -> float:
    sched = amort_schedule_monthly(principal, rate, term, amort, io_period, orig_date)
    first12 = sched.head(12)
    return float(first12["interest"].sum() + first12["principal"].sum())


def solve_principal_from_annual_ds(
    target_ds_12m: float,
    rate: float,
    term: int,
    amort: int,
    io_period: float,
    orig_date: date
) -> float:
    """
    Solve for principal that produces target debt service over first 12 months post-origination.
    If target is <= 0, returns 0.
    """
    if target_ds_12m <= 0:
        return 0.0

    def f(p):
        return ds_first_12_months_for_principal(p, rate, term, amort, io_period, orig_date) - target_ds_12m

    lo = 0.0
    hi = 1_000_000.0
    # grow hi until f(hi) >= 0
    for _ in range(30):
        if f(hi) >= 0:
            break
        hi *= 2.0
    else:
        # couldn't bracket; return conservative
        return 0.0

    return float(brentq(f, lo, hi))


def add_planned_second_mortgage_to_forecast(
    fc: pd.DataFrame,
    inv: pd.DataFrame,
    mri_supp: pd.DataFrame,
    mri_val: pd.DataFrame,
    isbs: pd.DataFrame,
    deal: str
) -> Tuple[pd.DataFrame, Optional[dict]]:
    """
    Returns updated forecast (rows appended for new mortgage interest/principal) + debug dict.
    """
    # Validate inputs (only if provided)
    if mri_supp is None or mri_supp.empty:
        return fc, None

    ms = mri_supp.copy()
    ms.columns = [str(c).strip() for c in ms.columns]
    ms["vCode"] = ms["vCode"].astype(str)

    row = ms[ms["vCode"] == str(deal)]
    if row.empty:
        return fc, None

    # Use first match (one planned second mortgage per deal for now)
    r = row.iloc[0]

    # Required fields in MRI_Supp per your description
    for col in ["Orig_Date", "Term", "Amort", "I/O Period", "DSCR", "LTV", "Rate"]:
        if col not in row.columns:
            raise ValueError(f"MRI_Supp.csv missing required column: {col}")

    orig_date = as_date(r["Orig_Date"])
    term = int(float(r["Term"]))
    amort = int(float(r["Amort"]))
    io_period = float(r["I/O Period"])
    dscr_req = float(r["DSCR"])
    ltv = float(r["LTV"])
    rate = float(r["Rate"])

    # Deal forecast subset (already normalized)
    fc_deal = fc[fc["vcode"].astype(str) == str(deal)].copy()
    if fc_deal.empty:
        raise ValueError(f"No forecast rows found for deal {deal} when sizing second mortgage.")

    proj_begin = get_projection_begin_date(fc_deal)

    # nCostSaleRate from investment_map if present, else 0
    inv2 = inv.copy()
    inv2.columns = [str(c).strip() for c in inv2.columns]
    inv2["vcode"] = inv2["vcode"].astype(str)
    inv_row = inv2[inv2["vcode"] == str(deal)]
    if inv_row.empty:
        n_cost_sale = 0.0
    else:
        n_cost_sale = float(inv_row.iloc[0].get("nCostSaleRate", 0.0) or 0.0)

    noi_12 = twelve_month_noi_after_date(fc_deal, orig_date)

    cap_rate = projected_cap_rate_at_orig(mri_val, deal, proj_begin, orig_date)
    if cap_rate <= 0:
        raise ValueError(f"Computed cap rate <= 0 for deal {deal}. Check MRI_Val fCapRate.")

    projected_value = (noi_12 / cap_rate) * (1.0 - n_cost_sale)

    # Existing loan balance at origination
    existing_bal = existing_loan_balance_at_orig(isbs, fc_deal, deal, proj_begin, orig_date)

    # LTV test
    max_total_debt = projected_value * ltv
    max_add_ltv = max(0.0, max_total_debt - existing_bal)

    # DSCR test
    if dscr_req <= 0:
        raise ValueError("MRI_Supp DSCR must be > 0")

    max_total_ds = noi_12 / dscr_req
    existing_ds = existing_debt_service_12m_after(fc_deal, orig_date)
    max_add_ds = max(0.0, max_total_ds - existing_ds)

    # Convert DS capacity to principal
    max_add_dscr = solve_principal_from_annual_ds(max_add_ds, rate, term, amort, io_period, orig_date)

    new_loan_amt = min(max_add_ltv, max_add_dscr)

    debug = {
        "Orig_Date": orig_date,
        "NOI_12m": noi_12,
        "CapRate": cap_rate,
        "ProjectedValue": projected_value,
        "ExistingLoanBal_at_Orig": existing_bal,
        "MaxAdd_LTV": max_add_ltv,
        "ExistingDS_12m": existing_ds,
        "MaxAddDS_12m": max_add_ds,
        "MaxAdd_DSCR_Principal": max_add_dscr,
        "NewLoanAmt": new_loan_amt,
    }

    if new_loan_amt <= 0:
        return fc, debug

    # Build schedule and append rows
    sched = amort_schedule_monthly(new_loan_amt, rate, term, amort, io_period, orig_date)

    # Create rows for Interest (7030) and Principal (7060)
    # We'll store mAmount as positive magnitude; normalization (outflow) would make negative,
    # BUT we are already normalized in fc. So we set mAmount_norm directly negative.
    new_rows = []

    for _, srow in sched.iterrows():
        dte = srow["event_date"]
        intr = float(srow["interest"])
        prin = float(srow["principal"])

        if intr != 0:
            new_rows.append({
                "vcode": str(deal),
                "event_date": dte,
                "vAccount": pd.Series([7030]).astype("Int64").iloc[0],
                "mAmount": intr,
                "mAmount_norm": -abs(intr),  # outflow
                "vSource": "MRI_Supp_NewLoan",
                "Pro_Yr": None,
                "Year": pd.Timestamp(dte).year,
                "vAccountType": "Expenses",
            })
        if prin != 0:
            new_rows.append({
                "vcode": str(deal),
                "event_date": dte,
                "vAccount": pd.Series([7060]).astype("Int64").iloc[0],
                "mAmount": prin,
                "mAmount_norm": -abs(prin),  # outflow
                "vSource": "MRI_Supp_NewLoan",
                "Pro_Yr": None,
                "Year": pd.Timestamp(dte).year,
                "vAccountType": "Liability",
            })

    if new_rows:
        fc_out = pd.concat([fc, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        fc_out = fc

    return fc_out, debug


# ============================================================
# ANNUAL AGGREGATION (Revenues → FAD by year)
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

    out = out.reset_index().fillna(0.0)
    return out


def pivot_annual_table(df: pd.DataFrame) -> pd.DataFrame:
    wide = df.set_index("Year").T
    wide.index.name = "Line Item"

    desired_order = [
        "Revenues",
        "Expenses",
        "NOI",
        "Interest",
        "Principal",
        "Total Debt Service",
        "Excluded Accounts",
        "Capital Expenditures",
        "Funds Available for Distribution",
        "Debt Service Coverage Ratio",
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
        styler = styler.set_properties(
            subset=pd.IndexSlice[["Expenses"], :],
            **{"text-decoration": "underline"}
        )

    if "NOI" in df.index:
        styler = styler.set_properties(
            subset=pd.IndexSlice[["NOI"], :],
            **{"border-bottom": "3px double black", "font-weight": "bold"}
        )

    if "Funds Available for Distribution" in df.index:
        fad_idx = df.index.get_loc("Funds Available for Distribution")
        if fad_idx > 0:
            prev_row = df.index[fad_idx - 1]
            styler = styler.set_properties(
                subset=pd.IndexSlice[[prev_row], :],
                **{"border-bottom": "2px solid black"}
            )

    if "Funds Available for Distribution" in df.index:
        styler = styler.set_properties(
            subset=pd.IndexSlice[["Funds Available for Distribution"], :],
            **{"font-weight": "bold"}
        )

    if "Debt Service Coverage Ratio" in df.index:
        styler = styler.set_properties(
            subset=pd.IndexSlice[["Debt Service Coverage Ratio"], :],
            **{"border-top": "1px solid #999"}
        )

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
        st.caption("Optional for future mortgages: MRI_Supp.csv, MRI_Val.csv, ISBS_Download.csv")
    else:
        uploads["investment_map"] = st.file_uploader("investment_map.csv", type="csv")
        uploads["waterfalls"] = st.file_uploader("waterfalls.csv", type="csv")
        uploads["coa"] = st.file_uploader("coa.csv", type="csv")
        uploads["accounting_feed"] = st.file_uploader("accounting_feed.csv", type="csv")
        uploads["forecast_feed"] = st.file_uploader("forecast_feed.csv", type="csv")

        st.divider()
        st.subheader("Optional: Planned Second Mortgages")
        uploads["MRI_Supp"] = st.file_uploader("MRI_Supp.csv (optional)", type="csv")
        uploads["MRI_Val"] = st.file_uploader("MRI_Val.csv (optional)", type="csv")
        uploads["ISBS_Download"] = st.file_uploader("ISBS_Download.csv (optional)", type="csv")

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

        # optional
        mri_supp = pd.read_csv(f"{folder}/MRI_Supp.csv") if Path(f"{folder}/MRI_Supp.csv").exists() else pd.DataFrame()
        mri_val = pd.read_csv(f"{folder}/MRI_Val.csv") if Path(f"{folder}/MRI_Val.csv").exists() else pd.DataFrame()
        isbs = pd.read_csv(f"{folder}/ISBS_Download.csv") if Path(f"{folder}/ISBS_Download.csv").exists() else pd.DataFrame()

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

        mri_supp = pd.read_csv(uploads["MRI_Supp"]) if uploads.get("MRI_Supp") is not None else pd.DataFrame()
        mri_val = pd.read_csv(uploads["MRI_Val"]) if uploads.get("MRI_Val") is not None else pd.DataFrame()
        isbs = pd.read_csv(uploads["ISBS_Download"]) if uploads.get("ISBS_Download") is not None else pd.DataFrame()

    inv.columns = [str(c).strip() for c in inv.columns]
    inv["vcode"] = inv["vcode"].astype(str)
    if "InvestmentID" in inv.columns:
        inv["InvestmentID"] = inv["InvestmentID"].astype(str)

    return inv, wf, coa, acct, fc, mri_supp, mri_val, isbs


inv, wf, coa, acct, fc, mri_supp, mri_val, isbs = load_inputs()

deal = st.selectbox("Select Deal", sorted(inv["vcode"].dropna().unique().tolist()))

if not st.button("Run Report", type="primary"):
    st.stop()


# ============================================================
# Append planned second mortgage (if applicable)
# ============================================================
debug_newloan = None
try:
    if mri_supp is not None and not mri_supp.empty:
        # Normalize MRI_Supp column names expected: vCode (not vcode)
        mri_supp = mri_supp.copy()
        mri_supp.columns = [str(c).strip() for c in mri_supp.columns]

        if "vCode" not in mri_supp.columns and "vcode" in mri_supp.columns:
            mri_supp = mri_supp.rename(columns={"vcode": "vCode"})

        # MRI_Val normalize: expect vcode
        if mri_val is not None and not mri_val.empty:
            mri_val = mri_val.copy()
            mri_val.columns = [str(c).strip() for c in mri_val.columns]
            if "vCode" in mri_val.columns and "vcode" not in mri_val.columns:
                mri_val = mri_val.rename(columns={"vCode": "vcode"})

        if isbs is not None and not isbs.empty:
            isbs = isbs.copy()
            isbs.columns = [str(c).strip() for c in isbs.columns]
            if "Vcode" in isbs.columns and "vcode" not in isbs.columns:
                isbs = isbs.rename(columns={"Vcode": "vcode"})

        # Only proceed if MRI_Val and ISBS are present (required by your steps)
        if (mri_val is None or mri_val.empty) or (isbs is None or isbs.empty):
            st.info("MRI_Supp.csv is present, but MRI_Val.csv and/or ISBS_Download.csv are missing; skipping planned mortgage sizing.")
        else:
            fc, debug_newloan = add_planned_second_mortgage_to_forecast(fc, inv, mri_supp, mri_val, isbs, deal)
except Exception as e:
    st.error(f"Failed to size/apply planned second mortgage for {deal}: {e}")


# ============================================================
# Annual report table (with new loan included if sized)
# ============================================================
st.subheader("Annual Operating Forecast (Revenues → Funds Available for Distribution)")

fc_deal = fc[fc["vcode"].astype(str) == str(deal)].copy()
if fc_deal.empty:
    st.error(f"No forecast rows found for deal {deal}.")
    st.stop()

annual_df_raw = annual_aggregation_table(fc_deal, int(start_year), int(horizon_years))
annual_df = pivot_annual_table(annual_df_raw)
styled = style_annual_table(annual_df)

st.dataframe(styled, use_container_width=True)

if debug_newloan:
    with st.expander("Planned Second Mortgage Sizing Details (debug)"):
        st.json(debug_newloan)

st.caption(
    "Normalization: Gross revenues +abs, vacancy/concessions -abs, expenses -abs, "
    "interest/principal/capex/excluded -abs. "
    "NOI = Revenues + Expenses. "
    "Funds Available for Distribution = NOI + Interest + Principal + Excluded + Capex. "
    "DSCR = NOI / |Total Debt Service|."
)

st.success("Report generated successfully.")

