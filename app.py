# app.py
# Waterfall + XIRR Forecast
# - Streamlit Cloud: forces Upload CSVs (no local C:\ paths)
# - Tight sign conventions for forecast using COA
# - Annual aggregation table: Revenues → FAD (by year)
# - COA join logic updated:
#     coa.csv: vcode == forecast_feed.vAccount == accounting_feed.TypeID

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Tuple
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

INTEREST_ACCTS = {5190, 7030}
PRINCIPAL_ACCTS = {7060}
CAPEX_ACCTS = {7050}
OTHER_EXCLUDED_ACCTS = {4050, 5220, 5210, 5195, 7065, 5120, 5130, 5400}
ALL_EXCLUDED = INTEREST_ACCTS | PRINCIPAL_ACCTS | CAPEX_ACCTS | OTHER_EXCLUDED_ACCTS


# ============================================================
# UTILITIES
# ============================================================
def to_date(x) -> date:
    return pd.to_datetime(x).date()


def is_year_end(d: date) -> bool:
    return d.month == 12 and d.day == 31


def year_ends_strictly_between(d0: date, d1: date) -> List[date]:
    if d1 <= d0:
        return []
    out = []
    y = d0.year
    while True:
        ye = date(y, 12, 31)
        if ye >= d1:
            break
        if ye > d0:
            out.append(ye)
        y += 1
    return out


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
# STATE
# ============================================================
@dataclass
class PartnerState:
    principal: float = 0.0
    pref_accrued: float = 0.0
    pref_capitalized: float = 0.0
    irr_cashflows: List[Tuple[date, float]] = field(default_factory=list)

    def base(self):
        return self.principal + self.pref_capitalized


@dataclass
class DealState:
    vcode: str
    last_event_date: date
    partners: Dict[str, PartnerState] = field(default_factory=dict)


# ============================================================
# ACCRUAL / COMPOUNDING
# ============================================================
def compound_year_end(deal: DealState):
    for ps in deal.partners.values():
        ps.pref_capitalized += ps.pref_accrued
        ps.pref_accrued = 0.0


def accrue_to(deal: DealState, new_date: date, pref_rates: Dict[str, float]):
    d0, d1 = deal.last_event_date, new_date
    if d1 <= d0:
        return

    splits = year_ends_strictly_between(d0, d1)
    dates = [d0] + splits + [d1]

    for i in range(len(dates) - 1):
        s, e = dates[i], dates[i + 1]
        yf = (e - s).days / 365.0

        for p, ps in deal.partners.items():
            r = pref_rates.get(p, 0.0)
            ps.pref_accrued += ps.base() * r * yf

        if is_year_end(e) and e != d1:
            compound_year_end(deal)


# ============================================================
# ACCOUNTING INGESTION (HISTORICAL)
# ============================================================
def map_bucket(flag):
    return "capital" if str(flag).upper() == "Y" else "pref"


def apply_txn(ps: PartnerState, d: date, amt: float, bucket: str):
    # NOTE: accounting-feed sign conventions will be finalized later.
    ps.irr_cashflows.append((d, amt))

    if bucket == "capital":
        # contributions are often negative in investor cashflow convention; this will be revisited
        ps.principal += -amt if amt < 0 else -min(amt, ps.principal)
    else:
        if amt > 0:
            pay = min(amt, ps.pref_accrued)
            ps.pref_accrued -= pay
            ps.pref_capitalized -= (amt - pay)
        else:
            ps.pref_accrued += -amt


# ============================================================
# LOADERS + SIGN NORMALIZATION (FORECAST)
# ============================================================
def load_coa(df: pd.DataFrame) -> pd.DataFrame:
    """
    coa.csv headers (per your feed):
      vcode, vdescription, vtype, iNOI, vMisc, vAccountType
    Joining rule:
      coa.vcode == forecast_feed.vAccount == accounting_feed.TypeID
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # vcode in COA is the account code
    if "vcode" not in df.columns:
        raise ValueError("coa.csv is missing required column: vcode")

    df = df.rename(columns={"vcode": "vAccount"})

    df["vAccount"] = pd.to_numeric(df["vAccount"], errors="coerce").astype("Int64")

    if "iNOI" not in df.columns:
        df["iNOI"] = 0
    df["iNOI"] = pd.to_numeric(df["iNOI"], errors="coerce").fillna(0).astype(int)

    if "vAccountType" not in df.columns:
        df["vAccountType"] = ""
    df["vAccountType"] = df["vAccountType"].fillna("").astype(str).str.strip()

    return df[["vAccount", "iNOI", "vAccountType"]]


def normalize_forecast_signs(fc: pd.DataFrame) -> pd.DataFrame:
    """
    Internal model standard:
      - Revenues: positive
      - Expenses: negative
      - Interest/Principal/Capex/Other excluded accounts: negative
    """
    out = fc.copy()
    atype = out["vAccountType"].astype(str).str.strip().str.lower()

    amt = pd.to_numeric(out["mAmount"], errors="coerce").fillna(0.0)

    # Revenues -> positive, Expenses -> negative
    amt = amt.where(~atype.eq("revenues"), amt.abs())
    amt = amt.where(~atype.eq("expenses"), -amt.abs())

    # Specific outflow accounts -> negative (even if feed presents as positive)
    amt = amt.where(~out["vAccount"].isin(ALL_EXCLUDED), -amt.abs())

    out["mAmount_norm"] = amt
    return out


def load_forecast(df: pd.DataFrame, coa: pd.DataFrame, pro_yr_base: int) -> pd.DataFrame:
    # Forecast feed columns (per your feed):
    # Vcode, dtEntry, vSource, vAccount, mAmount, Year, Qtr, Date, Pro_Yr
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    df = df.rename(columns={"Vcode": "vcode", "Date": "event_date"})
    df["vcode"] = df["vcode"].astype(str)

    df["event_date"] = pd.to_datetime(df["event_date"]).dt.date
    df["vAccount"] = pd.to_numeric(df["vAccount"], errors="coerce").astype("Int64")
    df["mAmount"] = pd.to_numeric(df["mAmount"], errors="coerce").fillna(0.0)

    df["Year"] = (int(pro_yr_base) + pd.to_numeric(df["Pro_Yr"], errors="coerce")).astype("Int64")

    # Join to COA on account code
    df = df.merge(coa, on="vAccount", how="left")
    df["iNOI"] = df["iNOI"].fillna(0).astype(int)
    df["vAccountType"] = df["vAccountType"].fillna("").astype(str)

    df = normalize_forecast_signs(df)
    return df


# ============================================================
# ANNUAL AGGREGATION (Revenues → FAD by year)
# ============================================================
def annual_aggregation_table(fc_deal: pd.DataFrame, start_year: int, horizon_years: int) -> pd.DataFrame:
    years = list(range(int(start_year), int(start_year) + int(horizon_years)))
    f = fc_deal[fc_deal["Year"].isin(years)].copy()

    f["acct_type"] = f["vAccountType"].astype(str).str.strip().str.lower()

    def sum_where(mask: pd.Series) -> pd.Series:
        if f.empty:
            return pd.Series(dtype=float)
        return f.loc[mask].groupby("Year")["mAmount_norm"].sum()

    revenues = sum_where(f["acct_type"].eq("revenues"))
    expenses = sum_where(f["acct_type"].eq("expenses"))
    noi = sum_where(f["iNOI"].eq(-1))

    interest = sum_where(f["vAccount"].isin(INTEREST_ACCTS))
    principal = sum_where(f["vAccount"].isin(PRINCIPAL_ACCTS))
    capex = sum_where(f["vAccount"].isin(CAPEX_ACCTS))
    excluded_other = sum_where(f["vAccount"].isin(OTHER_EXCLUDED_ACCTS))

    out = pd.DataFrame({"Year": years}).set_index("Year")

    out["Revenues"] = revenues
    out["Expenses"] = expenses
    out["NOI"] = noi

    out["Interest"] = interest
    out["Principal"] = principal
    out["Total Debt Service"] = out["Interest"].fillna(0.0) + out["Principal"].fillna(0.0)

    out["Excluded Accounts"] = excluded_other
    out["Capital Expenditures"] = capex

    # Because Interest/Principal/Excluded/Capex are normalized as negative outflows:
    out["Funds Available for Distribution"] = (
        out["NOI"].fillna(0.0)
        + out["Interest"].fillna(0.0)
        + out["Principal"].fillna(0.0)
        + out["Excluded Accounts"].fillna(0.0)
        + out["Capital Expenditures"].fillna(0.0)
    )

    out = out.fillna(0.0).reset_index()
    return out


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
    else:
        uploads["investment_map"] = st.file_uploader("investment_map.csv", type="csv")
        uploads["waterfalls"] = st.file_uploader("waterfalls.csv", type="csv")
        uploads["coa"] = st.file_uploader("coa.csv", type="csv")
        uploads["accounting_feed"] = st.file_uploader("accounting_feed.csv", type="csv")
        uploads["forecast_feed"] = st.file_uploader("forecast_feed.csv", type="csv")

    st.divider()
    st.header("Report Settings")
    start_year = st.number_input("Start year", min_value=2000, max_value=2100, value=DEFAULT_START_YEAR, step=1)
    horizon_years = st.number_input("Horizon (years)", min_value=1, max_value=30, value=DEFAULT_HORIZON_YEARS, step=1)
    pro_yr_base = st.number_input("Pro_Yr base year", min_value=1900, max_value=2100, value=PRO_YR_BASE_DEFAULT, step=1)


# ============================================================
# LOAD INPUTS
# ============================================================
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
    else:
        for k, f in uploads.items():
            if f is None:
                st.warning(f"Please upload {k}.csv")
                st.stop()

        inv = pd.read_csv(uploads["investment_map"])
        wf = pd.read_csv(uploads["waterfalls"])
        coa = load_coa(pd.read_csv(uploads["coa"]))
        acct = pd.read_csv(uploads["accounting_feed"])
        fc = load_forecast(pd.read_csv(uploads["forecast_feed"]), coa, int(pro_yr_base))

    # Normalize key fields we’ll use later
    inv["vcode"] = inv["vcode"].astype(str)

    if "InvestmentID" in inv.columns:
        inv["InvestmentID"] = inv["InvestmentID"].astype(str)

    # accounting_feed.TypeID aligns to COA account code (coa.vcode)
    if "TypeID" in acct.columns:
        acct["TypeID"] = pd.to_numeric(acct["TypeID"], errors="coerce").astype("Int64")

    return inv, wf, coa, acct, fc


inv, wf, coa, acct, fc = load_inputs()

deal = st.selectbox("Select Deal", sorted(inv["vcode"].dropna().unique().tolist()))

if not st.button("Run Report", type="primary"):
    st.stop()


# ============================================================
# CONTROL POPULATION (INNER JOIN ON InvestmentID → vcode)
# ============================================================
if "InvestmentID" in acct.columns and "InvestmentID" in inv.columns:
    acct["InvestmentID"] = acct["InvestmentID"].astype(str)
    acct = acct.merge(inv[["InvestmentID", "vcode"]], on="InvestmentID", how="inner")
    acct = acct[acct["vcode"] == deal].copy()
else:
    acct = pd.DataFrame()  # placeholder if not present yet

if acct.empty:
    st.warning("No accounting data found for the selected deal (after InvestmentID→vcode control join).")


# ============================================================
# INITIALIZE DEAL STATE FROM WATERFALL
# ============================================================
wf_d = wf[wf["vcode"].astype(str) == str(deal)].copy()
if wf_d.empty:
    st.error(f"No waterfall steps found for deal {deal}.")
    st.stop()

wf_d["dteffective"] = pd.to_datetime(wf_d["dteffective"]).dt.date
start_date = wf_d["dteffective"].min()

state = DealState(str(deal), start_date)
for p in wf_d["PropCode"].astype(str).unique():
    state.partners[p] = PartnerState()

# Pref rates from Pref steps (assumes nPercent is percent like 9 => 0.09)
pref_rates: Dict[str, float] = {}
if "vState" in wf_d.columns:
    pref_rows = wf_d[wf_d["vState"].astype(str).str.strip().str.lower().eq("pref")]
    for _, r in pref_rows.iterrows():
        rate = float(r.get("nPercent") or 0.0)
        if rate > 1.0:
            rate /= 100.0
        pref_rates[str(r["PropCode"])] = rate


# ============================================================
# APPLY HISTORICAL ACCOUNTING TO BUILD CURRENT STATE (placeholder)
# ============================================================
# NOTE: We'll finish accounting sign conventions & mapping later.
if not acct.empty and "EffectiveDate" in acct.columns and "InvestorID" in acct.columns:
    acct["EffectiveDate"] = pd.to_datetime(acct["EffectiveDate"]).dt.date
    acct["InvestorID"] = acct["InvestorID"].astype(str)

    for _, r in acct.sort_values("EffectiveDate").iterrows():
        d = r["EffectiveDate"]
        accrue_to(state, d, pref_rates)

        inv_id = r["InvestorID"]
        if inv_id in state.partners:
            apply_txn(
                state.partners[inv_id],
                d,
                float(r.get("Amt", 0.0)),
                map_bucket(r.get("Capital", "Y")),
            )

        state.last_event_date = d


# ============================================================
# ANNUAL AGGREGATION DISPLAY (Revenues → FAD)
# ============================================================
st.subheader("Annual Operating Forecast (Revenues → Funds Available for Distribution)")

fc_deal = fc[fc["vcode"].astype(str) == str(deal)].copy()
if fc_deal.empty:
    st.error(f"No forecast rows found for deal {deal}.")
    st.stop()

annual_df = annual_aggregation_table(fc_deal, int(start_year), int(horizon_years))

st.dataframe(
    annual_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Year": st.column_config.NumberColumn(format="%d"),
        "Revenues": st.column_config.NumberColumn(format="$%,.0f"),
        "Expenses": st.column_config.NumberColumn(format="$%,.0f"),
        "NOI": st.column_config.NumberColumn(format="$%,.0f"),
        "Interest": st.column_config.NumberColumn(format="$%,.0f"),
        "Principal": st.column_config.NumberColumn(format="$%,.0f"),
        "Total Debt Service": st.column_config.NumberColumn(format="$%,.0f"),
        "Excluded Accounts": st.column_config.NumberColumn(format="$%,.0f"),
        "Capital Expenditures": st.column_config.NumberColumn(format="$%,.0f"),
        "Funds Available for Distribution": st.column_config.NumberColumn(format="$%,.0f"),
    },
)

st.caption(
    "Sign conventions: Revenues normalized positive; Expenses/Interest/Principal/Capex/Excluded normalized negative. "
    "Funds Available for Distribution = NOI + Interest + Principal + Excluded + Capex."
)

st.success("Annual aggregation table generated successfully.")

