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

# ============================================================
# CAPITAL POOL ROUTING
# ============================================================

# Map accounting Typename keywords to capital pool names.
# Checked in order; first match wins.  Default is "initial".
TYPENAME_TO_POOL = {
    "operating capital": "operating",
    "cost overrun": "cost_overrun",
    "special capital": "special",
    "additional capital": "additional",
}


def typename_to_pool(typename: str) -> str:
    """Map an accounting Typename string to a capital pool name.

    Scans TYPENAME_TO_POOL for keyword matches (case-insensitive).
    Returns "initial" if no keyword matches.
    """
    t = (typename or "").strip().lower()
    for keyword, pool in TYPENAME_TO_POOL.items():
        if keyword in t:
            return pool
    return "initial"


def resolve_pool_and_action(vstate: str, vtranstype: str, is_capital_waterfall: bool) -> tuple:
    """Route a waterfall step to (pool_name, action).

    Used by the waterfall engine to decide which CapitalPool a step
    targets and what operation to perform.

    Actions returned:
        "pay_pref"          – pay preferred return from the pool
        "pay_capital"       – return capital from the pool
        "pay_capital_capped"– return capital with cumulative cap (operating)
        "skip"              – no action (e.g. capital return in CF waterfall)
        (None, None)        – step is not pool-routed (handled elsewhere)

    Args:
        vstate: Waterfall step vState (e.g. "Pref", "Initial", "Add")
        vtranstype: Waterfall step vtranstype text
        is_capital_waterfall: True for Cap_WF, False for CF_WF

    Returns:
        (pool_name, action) tuple
    """
    vt = (vtranstype or "").strip().lower()

    if vstate == "Pref":
        return ("initial", "pay_pref")

    if vstate == "Initial":
        if is_capital_waterfall:
            return ("initial", "pay_capital")
        return ("initial", "skip")

    if vstate == "Add":
        # Operating Capital — always capped return (CF or Cap)
        if "operating capital" in vt:
            return ("operating", "pay_capital_capped")

        # Cost Overrun
        if "cost overrun" in vt:
            if "pref" in vt:
                return ("cost_overrun", "pay_pref")
            if is_capital_waterfall:
                return ("cost_overrun", "pay_capital")
            return ("cost_overrun", "skip")

        # Special Capital
        if "special capital" in vt:
            if "pref" in vt:
                return ("special", "pay_pref")
            if is_capital_waterfall:
                return ("special", "pay_capital")
            return ("special", "skip")

        # Additional Capital (default Add bucket)
        if "pref" in vt:
            return ("additional", "pay_pref")
        if is_capital_waterfall:
            return ("additional", "pay_capital")
        return ("additional", "skip")

    # All other vStates (Share, IRR, Tag, Def&Int, etc.) — not pool-routed
    return (None, None)
