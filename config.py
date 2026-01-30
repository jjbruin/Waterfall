"""
config.py
Configuration and constants for waterfall model
"""

# ============================================================
# DEFAULT SETTINGS
# ============================================================
DEFAULT_START_YEAR = 2026
DEFAULT_HORIZON_YEARS = 10
PRO_YR_BASE_DEFAULT = 2025

# Capital event parameters
SELLING_COST_RATE = 0.02  # 2% selling costs
NEW_LOAN_NET_PROCEEDS = 0.98  # 98% net proceeds from new loans

# ============================================================
# ACCOUNT CLASSIFICATIONS
# ============================================================
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

INTEREST_ACCTS = {5190, 7030}
PRINCIPAL_ACCTS = {7060}
CAPEX_ACCTS = {7050}
OTHER_EXCLUDED_ACCTS = {4050, 5220, 5210, 5195, 7065, 5120, 5130, 5400}

ALL_EXCLUDED = INTEREST_ACCTS | PRINCIPAL_ACCTS | CAPEX_ACCTS | OTHER_EXCLUDED_ACCTS

# ============================================================
# DEBT INDEX BASE RATES
# ============================================================
INDEX_BASE_RATES = {
    "SOFR": 0.043,
    "LIBOR": 0.043,
    "WSJ": 0.075,
}
