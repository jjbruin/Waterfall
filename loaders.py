"""
loaders.py - DATABASE VERSION
CSV data loading and normalization functions

NOW LOADS FROM DATABASE INSTEAD OF CSV FILES
"""

import pandas as pd
import numpy as np
from typing import Dict

from config import (GROSS_REVENUE_ACCTS, CONTRA_REVENUE_ACCTS, EXPENSE_ACCTS, TAX_ABATEMENT_ACCTS,
                    INTEREST_ACCTS, PRINCIPAL_ACCTS, CAPEX_ACCTS, ALL_EXCLUDED)
from database import execute_query
from utils import normalize_columns


def load_coa(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Load and normalize chart of accounts

    Args:
        df: Optional DataFrame with COA data. If None, loads from database.

    Returns:
        DataFrame with vAccount and vAccountType
    """
    if df is not None and not df.empty:
        coa = df.copy()
    else:
        coa = execute_query("SELECT * FROM coa")
    
    normalize_columns(coa)
    
    if "vcode" in coa.columns and "vAccount" not in coa.columns:
        coa = coa.rename(columns={"vcode": "vAccount"})
    
    coa["vAccount"] = pd.to_numeric(coa["vAccount"], errors="coerce").astype("Int64")
    
    if "vAccountType" not in coa.columns:
        coa["vAccountType"] = ""
    coa["vAccountType"] = coa["vAccountType"].fillna("").astype(str).str.strip()
    
    return coa[["vAccount", "vAccountType"]]


def normalize_forecast_signs(fc: pd.DataFrame) -> pd.DataFrame:
    """Normalize forecast amounts to consistent sign conventions"""
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

    # Tax abatements are credits that increase cash flow — force positive
    is_abate = out["vAccount"].isin(TAX_ABATEMENT_ACCTS)
    amt = amt.where(~is_abate, base.abs())

    out["mAmount_norm"] = amt
    return out


def load_forecast(df: pd.DataFrame = None, coa: pd.DataFrame = None, pro_yr_base: int = None) -> pd.DataFrame:
    """
    Load and normalize forecast data

    Args:
        df: Optional DataFrame with forecast data. If None, loads from database.
        coa: Chart of accounts (if None, loads from database)
        pro_yr_base: Base year for Pro_Yr calculations

    Returns:
        Normalized forecast DataFrame
    """
    if df is not None and not df.empty:
        fc = df.copy()
    elif df is None:
        fc = execute_query("SELECT * FROM forecasts")
    else:
        # df is an empty DataFrame — return it as-is rather than silently
        # loading all forecasts from the database (which would cause callers
        # to process unrelated deals' data).
        return df.copy()

    normalize_columns(fc)
    fc = fc.rename(columns={"Vcode": "vcode", "Date": "event_date"})
    
    required = {"vcode", "event_date", "vAccount", "mAmount", "Pro_Yr"}
    missing = [c for c in required if c not in fc.columns]
    if missing:
        raise ValueError(f"forecasts table missing columns: {missing}")

    fc["vcode"] = fc["vcode"].astype(str)
    fc["event_date"] = pd.to_datetime(fc["event_date"], format='mixed').dt.date
    fc["vAccount"] = pd.to_numeric(fc["vAccount"], errors="coerce").astype("Int64")
    fc["mAmount"] = pd.to_numeric(fc["mAmount"], errors="coerce").fillna(0.0)

    if pro_yr_base:
        fc["Year"] = (int(pro_yr_base) + pd.to_numeric(fc["Pro_Yr"], errors="coerce")).astype("Int64")

    # Load COA if not provided
    if coa is None:
        coa = load_coa()
    
    fc = fc.merge(coa, on="vAccount", how="left")
    fc["vAccountType"] = fc["vAccountType"].fillna("").astype(str)

    fc = normalize_forecast_signs(fc)
    return fc


def load_mri_loans(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Load and normalize loans data

    Args:
        df: Optional DataFrame with loan data. If None, loads from database.

    Returns:
        Normalized loans DataFrame
    """
    if df is not None and not df.empty:
        d = df.copy()
    else:
        d = execute_query("SELECT * FROM loans")
    normalize_columns(d)
    
    if "vCode" not in d.columns:
        raise ValueError("loans table missing column: vCode")
    if "LoanID" not in d.columns:
        raise ValueError("loans table missing column: LoanID")
    if "dtEvent" not in d.columns and "dtMaturity" not in d.columns:
        raise ValueError("loans table must include dtEvent or dtMaturity")

    if "dtEvent" not in d.columns:
        d = d.rename(columns={"dtMaturity": "dtEvent"})

    d["vCode"] = d["vCode"].astype(str)
    d["LoanID"] = d["LoanID"].astype(str)
    d["dtEvent"] = pd.to_datetime(d["dtEvent"]).dt.date

    for c in ["mOrigLoanAmt", "iAmortTerm", "mNominalPenalty", "iLoanTerm", 
              "nRate", "vSpread", "nFloor", "vIntRatereset"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        else:
            d[c] = pd.NA

    d["vIntType"] = d.get("vIntType", "").fillna("").astype(str).str.strip()
    d["vIndex"] = d.get("vIndex", "").fillna("").astype(str).str.strip().str.upper()
    return d


def load_waterfalls(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Normalize waterfall definitions for computation.

    Args:
        df: Pre-loaded waterfall DataFrame (from data_service cache).
            Falls back to database query if None.

    Returns:
        Normalized waterfalls DataFrame with numeric types and nPercent_dec.
    """
    if df is not None and not df.empty and "nPercent_dec" in df.columns:
        # Already fully normalized (by data_service at load time)
        return df

    wf = df if df is not None and not df.empty else execute_query("SELECT * FROM waterfalls")

    w = wf.copy()
    normalize_columns(w)
    
    # Normalize vCode -> vcode
    ren = {}
    if "vCode" in w.columns and "vcode" not in w.columns:
        ren["vCode"] = "vcode"
    w = w.rename(columns=ren)
    
    if "vcode" not in w.columns:
        raise ValueError("waterfalls table must include vcode")
    
    w["vcode"] = w["vcode"].astype(str)

    required = {"vcode", "vmisc", "iOrder", "PropCode", "dteffective", 
                "mAmount", "nPercent", "FXRate", "vState"}
    missing = [c for c in required if c not in w.columns]
    if missing:
        raise ValueError(f"waterfalls table missing columns: {missing}")

    w["vmisc"] = w["vmisc"].astype(str).str.strip()
    w["PropCode"] = w["PropCode"].astype(str).str.strip()
    w["vState"] = w["vState"].astype(str).str.strip()

    w["iOrder"] = pd.to_numeric(w["iOrder"], errors="coerce").fillna(9999).astype(int)
    w["FXRate"] = pd.to_numeric(w["FXRate"], errors="coerce").fillna(0.0).astype(float)

    # Normalize percentages to decimals
    p = pd.to_numeric(w["nPercent"], errors="coerce").fillna(0.0).astype(float)
    w["nPercent_dec"] = np.where(p > 1.0, p / 100.0, p)

    w["mAmount"] = pd.to_numeric(w["mAmount"], errors="coerce").fillna(0.0).astype(float)
    w["dteffective"] = pd.to_datetime(w["dteffective"], errors="coerce").dt.date

    w = w.sort_values(["vcode", "vmisc", "iOrder"]).reset_index(drop=True)
    return w


def normalize_accounting_feed(acct: pd.DataFrame = None) -> pd.DataFrame:
    """
    Normalize accounting feed from database.

    Args:
        acct: Accounting DataFrame or None to load from database.
            If already normalized (has 'is_contribution' column), returns as-is.

    Returns:
        Normalized accounting DataFrame
    """
    if acct is None:
        acct = execute_query("SELECT * FROM accounting")

    # Short-circuit if already normalized (by data_service at load time)
    if "is_contribution" in acct.columns:
        return acct

    a = acct.copy()
    normalize_columns(a)

    required = {"InvestmentID", "InvestorID", "EffectiveDate", "MajorType", "Amt", "Capital"}
    missing = [c for c in required if c not in a.columns]
    if missing:
        raise ValueError(f"accounting table missing columns: {missing}")

    a["InvestmentID"] = a["InvestmentID"].astype(str).str.strip().str.upper()
    a["InvestorID"] = a["InvestorID"].astype(str).str.strip().str.upper()
    a["EffectiveDate"] = pd.to_datetime(a["EffectiveDate"], errors="coerce").dt.date

    a["MajorType"] = a["MajorType"].fillna("").astype(str).str.strip()
    # Ignore system quarter-end rows
    a = a[a["MajorType"] != ""].copy()

    a["_raw_amt"] = a["Amt"].copy()
    a["Amt"] = pd.to_numeric(
        a["Amt"].astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False).str.replace("(", "-", regex=False).str.replace(")", "", regex=False).str.strip(),
        errors="coerce"
    ).fillna(0.0).astype(float)

    a["Capital"] = a["Capital"].fillna("").astype(str).str.strip().str.upper()
    a["is_capital"] = a["Capital"].eq("Y")

    a["MajorTypeNorm"] = a["MajorType"].str.lower()

    # Preserve Typename column for add-capital routing
    if "Typename" in a.columns:
        a["Typename"] = a["Typename"].fillna("").astype(str).str.strip()
    else:
        a["Typename"] = ""

    # Commitment rows are pledges, not cash activity — exclude from is_contribution
    # so they never enter waterfall, ROE, IRR, accrued pref, or MOIC calculations.
    # Marked with is_commitment for One Pager committed PE lookup.
    a["is_commitment"] = (
        a["MajorTypeNorm"].str.contains("contrib", na=False) &
        a["Typename"].str.lower().str.contains("commitment", na=False)
    )
    a["is_contribution"] = a["MajorTypeNorm"].str.contains("contrib") & ~a["is_commitment"]
    a["is_distribution"] = a["MajorTypeNorm"].str.contains("distri")

    # Include Partner column if available (for equity classification)
    if "Partner" in a.columns:
        a["Partner"] = a["Partner"].fillna("").astype(str).str.strip()
    else:
        a["Partner"] = ""

    # Keep contrib/distr/commitment rows
    a = a[a["is_contribution"] | a["is_distribution"] | a["is_commitment"]].copy()

    return a


# ══════════════════════════════════════════════════════════════════════════
# Capital outstanding — THE SIGN OF THE AMOUNT IS THE DIRECTION
# ══════════════════════════════════════════════════════════════════════════
#: One definition, used by every place that walks the accounting feed keeping a
#: running capital balance: the Deal Analysis waterfall seeding, ROE Summary's
#: accrued pref, the Pref Balance Detail report and Sold Portfolio.
#:
#: WHY THIS EXISTS. Each of those used to read the row's TYPE and then take
#: ``abs()`` of the amount:
#:
#:     if is_contribution:  capital += abs(amt)
#:     elif is_capital_return:  capital = max(0.0, capital - abs(amt))
#:
#: which discards the one piece of information that distinguishes a booking
#: from its REVERSAL. MRI reverses an entry by re-posting it with the opposite
#: sign under the same MajorType and Typename, so ``abs()`` applied the reversal
#: in the same direction as the original and moved capital by twice the amount
#: instead of leaving it unchanged.
#:
#: Measured on the live feed (2026-09-02): 66 such rows across 16 deals and 37
#: (deal, investor) pairs, mis-stating capital by ~$80.5M in total. 30 Bearfoot
#: carried $540,000 for OPMCCORD where the true figure was $960,000, and pref
#: accrued at 56% of the correct rate for eighteen months as a result; JB Fair
#: Park's $3.85M contribution was carried as $11.55M.
#:
#: THE RULE. The accounting feed's sign convention already encodes direction:
#: cash INTO the deal is negative (a contribution), cash OUT is positive (a
#: return of capital). So the change in capital outstanding is simply the
#: NEGATED amount, whatever the row's type says, and a reversal reverses itself
#: because its sign is flipped. The type is still what decides WHETHER a row
#: touches capital — it is no longer asked which way.
def capital_after(capital: float, amt: float) -> float:
    """Capital outstanding after one capital-affecting row. NOT floored.

    ``amt`` is the raw signed accounting amount: negative (cash in) increases
    capital, positive (cash out) decreases it.

    THE RUNNING TOTAL IS ALLOWED TO GO NEGATIVE, and that is deliberate. An
    earlier version of this floored at zero on every row, which made the answer
    depend on the order of same-day rows: JB Fair Park's 2020-12-18 reversal
    (+3,850,000) sorts BEFORE the contribution it reverses (-3,850,000), so the
    floor swallowed the reversal and the two remaining contributions both
    landed — 7,700,000 against a true 3,850,000. Netting without a floor is a
    plain sum, so it cannot depend on order at all.

    Floor at the point of USE instead: ``capital_outstanding`` for accrual and
    for display is ``max(0.0, running)``. A running total that ends BELOW zero
    is a real finding — the feed returned more capital than it ever contributed
    — and callers report it rather than absorbing it silently.
    """
    return float(capital) - float(amt)


def capital_outstanding(running: float) -> float:
    """The running total as a balance: floored at zero. See ``capital_after``."""
    return max(0.0, float(running))


def build_investmentid_to_vcode(inv_map: pd.DataFrame = None) -> dict:
    """
    Build mapping from InvestmentID to vcode
    
    Args:
        inv_map: Investment map DataFrame or None to load from database
    
    Returns:
        Dictionary mapping InvestmentID to vcode
    """
    if inv_map is None:
        inv_map = execute_query("SELECT * FROM deals")
    
    inv = inv_map.copy()
    normalize_columns(inv)
    
    if "InvestmentID" not in inv.columns:
        raise ValueError("deals table must include InvestmentID")
    if "vcode" not in inv.columns and "vCode" in inv.columns:
        inv = inv.rename(columns={"vCode": "vcode"})
    if "vcode" not in inv.columns:
        raise ValueError("deals table must include vcode")

    inv["InvestmentID"] = inv["InvestmentID"].astype(str).str.strip().str.upper()
    inv["vcode"] = inv["vcode"].astype(str).str.strip()

    return dict(zip(inv["InvestmentID"], inv["vcode"]))


# FUND-LEVEL LOADERS

def load_fund_deals(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Load fund-deal relationships from database
    
    Args:
        df: Ignored (kept for backward compatibility)
    
    Returns:
        Fund deals DataFrame
    """
    try:
        fd = execute_query("SELECT * FROM fund_deals")
    except:
        # Table doesn't exist yet
        return pd.DataFrame()
    
    normalize_columns(fd)
    
    required = {"FundID", "vcode", "PPI_PropCode", "Ownership_Pct"}
    missing = [c for c in required if c not in fd.columns]
    if missing:
        return pd.DataFrame()
    
    fd["FundID"] = fd["FundID"].astype(str).str.strip()
    fd["vcode"] = fd["vcode"].astype(str).str.strip()
    fd["PPI_PropCode"] = fd["PPI_PropCode"].astype(str).str.strip()
    fd["Ownership_Pct"] = pd.to_numeric(fd["Ownership_Pct"], errors="coerce").fillna(0.0)
    
    # Normalize percentages
    fd["Ownership_Pct"] = fd["Ownership_Pct"].apply(
        lambda x: x / 100.0 if x > 1.0 else x
    )
    
    return fd


def load_investor_waterfalls(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Load investor-level waterfall definitions from database
    
    Args:
        df: Ignored (kept for backward compatibility)
    
    Returns:
        Investor waterfalls DataFrame
    """
    try:
        iw = execute_query("SELECT * FROM investor_waterfalls")
    except:
        # Table doesn't exist yet
        return pd.DataFrame()
    
    normalize_columns(iw)
    
    required = {"FundID", "iOrder", "InvestorID", "vState"}
    missing = [c for c in required if c not in iw.columns]
    if missing:
        return pd.DataFrame()
    
    iw["FundID"] = iw["FundID"].astype(str).str.strip()
    iw["InvestorID"] = iw["InvestorID"].astype(str).str.strip().str.upper()
    iw["vState"] = iw["vState"].astype(str).str.strip()
    iw["iOrder"] = pd.to_numeric(iw["iOrder"], errors="coerce").fillna(9999).astype(int)
    
    for col in ["nPercent", "FXRate"]:
        if col in iw.columns:
            iw[col] = pd.to_numeric(iw[col], errors="coerce").fillna(0.0)
            iw[f"{col}_dec"] = iw[col].apply(lambda x: x / 100.0 if x > 1.0 else x)
        else:
            iw[f"{col}_dec"] = 0.0
    
    iw = iw.sort_values(["FundID", "iOrder"])
    return iw


def load_investor_accounting(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Load investor-level accounting history from database
    
    Args:
        df: Ignored (kept for backward compatibility)
    
    Returns:
        Investor accounting DataFrame
    """
    try:
        ia = execute_query("SELECT * FROM investor_accounting")
    except:
        # Table doesn't exist yet
        return pd.DataFrame()
    
    normalize_columns(ia)
    
    required = {"FundID", "InvestorID", "EffectiveDate", "TransType", "Amount"}
    missing = [c for c in required if c not in ia.columns]
    if missing:
        return pd.DataFrame()
    
    ia["FundID"] = ia["FundID"].astype(str).str.strip()
    ia["InvestorID"] = ia["InvestorID"].astype(str).str.strip().str.upper()
    ia["EffectiveDate"] = pd.to_datetime(ia["EffectiveDate"], errors="coerce").dt.date
    ia["TransType"] = ia["TransType"].astype(str).str.strip().str.lower()
    ia["Amount"] = pd.to_numeric(ia["Amount"], errors="coerce").fillna(0.0)
    
    ia["is_contribution"] = ia["TransType"].str.contains("contrib")
    ia["is_distribution"] = ia["TransType"].str.contains("distri")
    
    return ia
