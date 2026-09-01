"""
config.py
Configuration and constants for waterfall model
"""
import math

def _isnan(v):
    """Safe NaN check for float values."""
    try:
        return math.isnan(v)
    except (TypeError, ValueError):
        return False

# ============================================================
# DEFAULT SETTINGS
# ============================================================
DEFAULT_START_YEAR = 2026
DEFAULT_HORIZON_YEARS = 10
PRO_YR_BASE_DEFAULT = 2025
DEFAULT_ACTUALS_THROUGH = "2026-07-31"  # None = full forecast; date = actuals cutoff

# Balance sheet debt accounts (Mortgages and Loans from ISBS)
DEBT_BS_ACCTS = {'2150', '2152', '2210'}

# Cash & reserve accounts for beginning cash balance (from ISBS Interim BS)
CASH_BALANCE_ACCTS = {
    '1010', '1012', '1014', '1070',
    '1090', '1091', '1092',
    '1100', '1120', '1130', '1140', '1141', '1142', '1144', '1145',
}

# Capital event parameters
SELLING_COST_RATE = 0.02  # 2% selling costs
NEW_LOAN_NET_PROCEEDS = 0.98  # 98% net proceeds from new loans

# ============================================================
# ACCOUNT CLASSIFICATIONS
# ============================================================
CONTRA_REVENUE_ACCTS = {4040, 4043, 4030, 4031, 4042}

REVENUE_ACCTS = {
    4010, 4012, 4020, 4041, 4045, 4040, 4043, 4030, 4031, 4042, 4070,
    4091, 4092, 4090, 4097, 4093, 4094, 4096, 4095,
    4063, 4060, 4061, 4062, 4080, 4065, 4075
}

GROSS_REVENUE_ACCTS = REVENUE_ACCTS - CONTRA_REVENUE_ACCTS

EXPENSE_ACCTS = {
    5090, 5110, 5114, 5018, 5010, 5016, 5012, 5014,
    5051, 5053, 5050, 5052, 5054, 5055,
    5060, 5067, 5063, 5069, 5061, 5064, 5065, 5068, 5070, 5066,
    5020, 5022, 5021, 5023, 5025, 5026,
    5045, 5080, 5087, 5085, 5040,
    5096, 5095, 5091, 5100, 5092
}

INTEREST_ACCTS = {5190, 7030}
PRINCIPAL_ACCTS = {7060}
CAPEX_ACCTS = {7050}
OTHER_EXCLUDED_ACCTS = {4050, 5220, 5210, 5195, 7065, 5400, 5160, 5165, 5120, 5130}
TAX_ABATEMENT_ACCTS = {7070}

ALL_EXCLUDED = INTEREST_ACCTS | PRINCIPAL_ACCTS | CAPEX_ACCTS | OTHER_EXCLUDED_ACCTS

TAX_ABATEMENT_DISCOUNT_RATE = 0.05

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
    t = typename
    if t is None or (isinstance(t, float) and _isnan(t)):
        t = ""
    t = str(t).strip().lower()
    for keyword, pool in TYPENAME_TO_POOL.items():
        if keyword in t:
            return pool
    return "initial"


# ============================================================
# UPSTREAM TYPENAME ROUTING
# ============================================================
# Maps typename keywords (case-insensitive substring) to target entity.
# When cash flows upstream and a typename matches, the cash bypasses
# intermediate waterfalls and routes directly to the target entity.
UPSTREAM_TYPENAME_ROUTING = {
    "acquisition fee": "PSC1",
}


def resolve_upstream_typename_route(typename: str, entity_id: str) -> str | None:
    """If typename matches a routing rule AND entity is NOT the target, return target. Else None.

    Case-insensitive substring matching (consistent with typename_to_pool).
    Returns None when no routing applies — caller continues normal waterfall.
    """
    t = typename
    if t is None or (isinstance(t, float) and _isnan(t)):
        t = ""
    t = str(t).strip().lower()
    if not t:
        return None
    for keyword, target in UPSTREAM_TYPENAME_ROUTING.items():
        if keyword in t and entity_id != target:
            return target
    return None


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
    vt = vtranstype
    if vt is None or (isinstance(vt, float) and _isnan(vt)):
        vt = ""
    vt = str(vt).strip().lower()

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


# ============================================================
# FINANCIAL STATEMENT ACCOUNT CLASSIFICATIONS
# ============================================================

BS_ACCOUNTS = {
    'ASSETS': {
        'Current Assets': {
            'Cash': ['1010', '1012'],
            'Misc Current Assets': ['1040', '1070'],
        },
        'Noncurrent Assets': {
            'Accounts Receivable': ['1020', '1025', '1030'],
            'Lender Held Reserves & Escrows': ['1145', '1092', '1091'],
            'Other Reserves & Escrows': ['1014', '1080', '1090', '1100', '1120', '1130', '1140'],
            'Prepaid': ['1050', '1060', '1075', '1151'],
            'Fixed Assets': ['1240', '1250', '1260', '1270', '1280', '1282', '1275'],
            'Depreciation & Amortization': ['1230', '1290'],
            'Other Assets': ['1150', '1224', '1220'],
        },
    },
    'LIABILITIES': {
        'Current Liabilities': {
            'Accounts Payable': ['2010', '2012', '2015', '2020'],
            'Accrued Interest Payable': ['2060'],
            'Accrued Taxes Payable': ['2110'],
            'Security Deposits': ['2090'],
            'Prepaid Revenues': ['2080'],
            'Other Accrued Liabilities': ['2115', '2120', '2124', '2130'],
        },
        'Noncurrent Liabilities': {
            'Mortgages and Loans': ['2150', '2152', '2210'],
            'Misc Long Term Liabilities': ['2300', '2310'],
            'Deferred Developer/AM Fee': ['2230'],
            'Notes Payable to GP': ['2280'],
            'Notes Payable to LP': ['2290'],
        },
    },
    'EQUITY': {
        'Equity': {
            'Equity': ['2520', '2530', '2534', '2536', '2540'],
            'Partner Equity': ['2525'],
            'PSC Pref Equity': ['2526'],
            'Distributions-2527': ['2527'],
            'Distributions-2528': ['2528'],
            'Net Income': ['2550'],
        },
    },
}

IS_ACCOUNTS = {
    'REVENUES': {
        'Rental Income': ['4010', '4012'],
        'Commercial': ['4020', '4041'],
        'Abated Apartments': ['4045'],
        'Vacancy': ['4040', '4043', '4030', '4031', '4042'],
        'RUBS': ['4070'],
        'RET': ['4091'],
        'INS': ['4092'],
        'CAM': ['4090', '4097', '4093', '4094', '4096', '4095'],
        'Other Income': ['4063', '4060', '4061', '4062', '4080', '4065', '4075'],
    },
    'EXPENSES': {
        'Real Estate Taxes': ['5090', '7070'],
        'Property & Liability Insurance': ['5110', '5114'],
        'Salary & Benefits': ['5018', '5010', '5016', '5012', '5014'],
        'Utilities': ['5051', '5053', '5050', '5052', '5054', '5055'],
        'Repairs & Maintenance': ['5060', '5067', '5063', '5069', '5061', '5064', '5065', '5068', '5070', '5066', '5092'],
        'Administrative': ['5020', '5022', '5021', '5023', '5025', '5026', '5080'],
        'Marketing & Advertising': ['5045'],
        'Legal & Professional': ['5087', '5085'],
        'Management Fee': ['5040'],
        'Other Expenses': ['5096', '5095', '5091', '5100'],
    },
    'DEBT_SERVICE': {
        'Interest': ['5190'],
        'Principal': [],  # Computed separately: BS balance change for Actuals, 7060 for Budget
    },
    'OTHER_BTL': {
        'Interest Income': ['4050'],
        'Other (Income) Expenses': ['5220', '5210', '5195', '7065'],
        'Capital Expenditures': ['7050'],
        'Partnership Expenses': ['5120', '5130'],
        'Depreciation & Amortization': ['5160', '5165'],
        'Extraordinary Expenses': ['5400'],
    },
}


# ============================================================
# DEV DEAL CLASSIFICATION
# ============================================================
#
# Lives here rather than in a flask service because two layers now need it and
# they sit on opposite sides of the app: ``one_pager.py`` (core, imported by
# scripts with no flask on the path) reads it to pick the cap-stack debt basis,
# and ``portfolio_snapshot_operating.py`` reads it for the "Dev" display, the
# mOrigLoanAmt debt path and the "Excluding Development Deals" subtotal.
#
# ONE definition, so those cannot disagree about what a development deal is.
# ``portfolio_snapshot_operating`` imports both names and re-exports them, which
# is why ``portfolio_snapshot_financial`` and ``portfolio_snapshot_loan`` still
# import them from there unchanged.

#: Strategy values that mark a development deal.
#:
#: "new construction" was REMOVED 2026-09-01. On this feed that Lifecycle value
#: describes the VINTAGE of the building, not that Peaceable is constructing it,
#: and it sat on exactly two deals — both finished assets bought after they were
#: built:
#:
#:   P0000006 Belleville Self Storage   Year_Built 2020, acquired 04/12/2021
#:   P0000066 Pegasus Life Storage      Year_Built 2020, acquired 05/11/2022
#:
#: Two independent registers agree they are not construction deals. Every
#: genuine one carries Year_Built = "To Be Built" (8 deals, all Lifecycle
#: "Development"), and the MRI ``inspection`` table — the construction-draw
#: register, 10 rows portfolio-wide — has no row for either. The reference PDF
#: also treats both as operating, printing real occupancy and NOI for Pegasus.
#:
#: The cost of the miscoding was real: Belleville's Operating row discarded a
#: genuine 93.2% occupancy and 1.14M NOI, and its Loan row printed "Dev" over a
#: computable 82.6% LTV, 0.708 DSCR and 5.44% Debt Yield.
#:
#: Removing it also drops Pegasus out of the At-Close Year-0 gate in
#: one_pager.py, which keys on this set rather than carrying its own list.
#: The Portfolio Snapshot Summary's DEAL_TYPE_MAP is a SEPARATE map and already
#: bucketed "new construction" as Income, so the allocation pie does not move.
#:
#: NOTE this leaves the set a single value. It stays a set because
#: Investment_Strategy is not yet populated (0/134) and will bring its own
#: vocabulary when MRI feeds it — at which point new development values are
#: added here and nowhere else.
DEV_STRATEGIES = {"development"}


def is_dev_deal(strategy):
    """True when a strategy string names a development deal."""
    return str(strategy or "").strip().lower() in DEV_STRATEGIES
