# app.py
# Waterfall + XIRR Forecast
# Tightened sign conventions + Streamlit Cloud protection

import io
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
            r = pref_rates.get(p, 0)
            ps.pref_accrued += ps.base() * r * yf

        if is_year_end(e) and e != d1:
            compound_year_end(deal)


# ============================================================
# ACCOUNTING INGESTION (HISTORICAL)
# ============================================================
def map_bucket(flag):
    return "capital" if str(flag).upper() == "Y" else "pref"


def apply_txn(ps: PartnerState, d: date, amt: float, bucket: str):
    ps.irr_cashflows.append((d, amt))
    if bucket == "capital":
        ps.principal += -amt if amt < 0 else -min(amt, ps.principal)
    else:
        if amt > 0:
            pay = min(amt, ps.pref_accrued)
            ps.pref_accrued -= pay
            ps.pref_capitalized -= (amt - pay)
        else:
            ps.pref_accrued += -amt


# ============================================================
# LOADERS + SIGN NORMALIZATION
# ============================================================
def load_coa(df):
    df = df.rename(columns={"vCode": "vAccount"})
    df["vAccount"] = pd.to_numeric(df["vAccount"], errors="coerce").astype("Int64")

    if "iNOI" not in df.columns and "NOI" in df.columns:
        df = df.rename(columns={"NOI": "iNOI"})
    df["iNOI"] = pd.to_numeric(df.get("iNOI", 0), errors="coerce").fillna(0).astype(int)

    if "vAccountType" not in df.columns:
        df["vAccountType"] = ""

    df["vAccountType"] = df["vAccountType"].astype(str).str.strip()
    return df[["vAccount", "iNOI", "vAccountType"]]


def normalize_forecast_signs(fc: pd.DataFrame) -> pd.DataFrame:
    out = fc.copy()
    atype = out["vAccountType"].str.lower()

    amt = pd.to_numeric(out["mAmount"], errors="coerce").fillna(0.0)

    # Revenues positive
    amt = amt.where(~atype.eq("revenues"), amt.abs())

    # Expenses negative
    amt = amt.where(~atype.eq("expenses"), -amt.abs())

    # Debt / capex / excluded always negative
    amt = amt.where(~out["vAccount"].isin(ALL_EXCLUDED), -amt.abs())

    out["mAmount_norm"] = amt
    return out


def load_forecast(df, coa, pro_yr_base):
    df = df.rename(columns={"Vcode": "vcode", "Date": "event_date"})
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.date
    df["Year"] = pro_yr_base + df["Pro_Yr"]
    df = df.merge(coa, on="vAccount", how="left")
    df = normalize_forecast_signs(df)
    return df


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
        st.info("Running on Streamlit Cloud — local folders are disabled.")
    else:
        mode = st.radio("Load data from:", ["Local folder", "Upload CSVs"])

    folder = None
    uploads = {}

    if mode == "Local folder":
        folder = st.text_input("Data folder path", placeholder=r"C:\Path\To\Data")
    else:
        uploads["investment_map"] = st.file_uploader("investment_map.csv")
        uploads["waterfalls"] = st.file_uploader("waterfalls.csv")
        uploads["coa"] = st.file_uploader("coa.csv")
        uploads["accounting_feed"] = st.file_uploader("accounting_feed.csv")
        uploads["forecast_feed"] = st.file_uploader("forecast_feed.csv")

    st.divider()
    start_year = st.number_input("Start year", value=DEFAULT_START_YEAR)
    horizon_years = st.number_input("Horizon (years)", value=DEFAULT_HORIZON_YEARS)
    pro_yr_base = st.number_input("Pro_Yr base year", value=PRO_YR_BASE_DEFAULT)


# ============================================================
# LOAD INPUTS
# ============================================================
def load_inputs():
    if CLOUD and mode == "Local folder":
        st.error("Local folder mode disabled on Streamlit Cloud.")
        st.stop()

    if mode == "Local folder":
        inv = pd.read_csv(f"{folder}/investment_map.csv")
        wf = pd.read_csv(f"{folder}/waterfalls.csv")
        coa = load_coa(pd.read_csv(f"{folder}/coa.csv"))
        acct = pd.read_csv(f"{folder}/accounting_feed.csv")
        fc = load_forecast(pd.read_csv(f"{folder}/forecast_feed.csv"), coa, pro_yr_base)
    else:
        for k, f in uploads.items():
            if f is None:
                st.warning(f"Upload {k}.csv")
                st.stop()
        inv = pd.read_csv(uploads["investment_map"])
        wf = pd.read_csv(uploads["waterfalls"])
        coa = load_coa(pd.read_csv(uploads["coa"]))
        acct = pd.read_csv(uploads["accounting_feed"])
        fc = load_forecast(pd.read_csv(uploads["forecast_feed"]), coa, pro_yr_base)

    return inv, wf, coa, acct, fc


inv, wf, coa, acct, fc = load_inputs()

inv["vcode"] = inv["vcode"].astype(str)
deal = st.selectbox("Select Deal", sorted(inv["vcode"].unique()))

if not st.button("Run Report"):
    st.stop()


# ============================================================
# CONTROL POPULATION (INNER JOIN)
# ============================================================
acct = acct.merge(inv[["InvestmentID", "vcode"]], on="InvestmentID", how="inner")
acct = acct[acct["vcode"] == deal]

if acct.empty:
    st.error("No accounting data for selected deal.")
    st.stop()


# ============================================================
# INITIALIZE DEAL
# ============================================================
wf_d = wf[wf["vcode"] == deal]
start_date = pd.to_datetime(wf_d["dteffective"].min()).date()

state = DealState(deal, start_date)
for p in wf_d["PropCode"].astype(str).unique():
    state.partners[p] = PartnerState()

pref_rates = {
    r["PropCode"]: r["nPercent"] / 100
    for _, r in wf_d[wf_d["vState"] == "Pref"].iterrows()
}


# ============================================================
# APPLY HISTORICAL ACCOUNTING
# ============================================================
acct["EffectiveDate"] = pd.to_datetime(acct["EffectiveDate"]).dt.date
for _, r in acct.sort_values("EffectiveDate").iterrows():
    accrue_to(state, r["EffectiveDate"], pref_rates)
    apply_txn(
        state.partners[r["InvestorID"]],
        r["EffectiveDate"],
        r["Amt"],
        map_bucket(r["Capital"])
    )
    state.last_event_date = r["EffectiveDate"]


st.success("Sign conventions normalized. Model state initialized successfully.")
st.write("Next: annual forecast + waterfall execution.")
