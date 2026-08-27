"""AI Assistant service — Claude API integration with app-aware tools."""

import os
import json
import logging
from datetime import date, datetime

import anthropic
import pandas as pd
from flask import current_app

from flask_app.services import data_service

logger = logging.getLogger(__name__)

# ── Tool definitions ─────────────────────────────────────────────────

TOOLS = [
    {
        "name": "resolve_deal",
        "description": "Resolve a deal name (or partial name) to its vcode. Use this when the user mentions a deal by name and you need the vcode for other tools. Supports fuzzy/partial matching.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Full or partial deal name to search for (e.g., '30 Bearfoot', 'bearfoot', '3rd Ave')",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "list_deals",
        "description": "List all deals (investments) in the portfolio. Returns deal name, vcode, asset type, status, acquisition date, operating partner, and other metadata. Use this to find a deal's vcode before querying detail. Can filter by status and/or operating partner.",
        "input_schema": {
            "type": "object",
            "properties": {
                "status_filter": {
                    "type": "string",
                    "description": "Optional filter: 'active', 'sold', or 'all' (default: 'all')",
                    "enum": ["active", "sold", "all"],
                },
                "operating_partner": {
                    "type": "string",
                    "description": "Optional: filter to deals with this operating partner (case-insensitive partial match)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "query_deal_data",
        "description": "Get detailed information about a specific deal by vcode. Returns investment metadata, capitalization, key dates, and summary metrics.",
        "input_schema": {
            "type": "object",
            "properties": {
                "vcode": {
                    "type": "string",
                    "description": "The deal's vcode identifier (e.g., 'P0000083')",
                },
            },
            "required": ["vcode"],
        },
    },
    {
        "name": "query_accounting",
        "description": "Query accounting entries (contributions and distributions) for a deal. Returns date, investor, type, amount, and capital balance.",
        "input_schema": {
            "type": "object",
            "properties": {
                "vcode": {
                    "type": "string",
                    "description": "The deal's vcode identifier",
                },
                "investor_id": {
                    "type": "string",
                    "description": "Optional: filter to a specific investor ID",
                },
            },
            "required": ["vcode"],
        },
    },
    {
        "name": "query_database",
        "description": "Run a read-only SQL query against the application database. Use for ad-hoc analysis. Available tables include: deals, accounting, waterfalls, commitments, relationships, isbs_interim_is, isbs_interim_bs, isbs_budget_is, isbs_projected_is, isbs_valuation_is, coa, occupancy, valuations, loans, tenants, capital_calls, planned_loans, prospective_loans, forecasts. Always use SELECT statements only.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "A SELECT SQL query to run against the database. Must be read-only (SELECT only).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max rows to return (default 100, max 500)",
                },
            },
            "required": ["sql"],
        },
    },
    {
        "name": "get_portfolio_summary",
        "description": "Get portfolio-level KPIs: total value, debt outstanding, deal count, weighted avg cap rate, portfolio occupancy, total preferred equity.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "compute_deal_returns",
        "description": "Compute IRR, ROE, and MOIC for a specific deal. Runs the full waterfall computation engine. Also returns sale date, sale proceeds breakdown (NOI, cap rate, implied value, selling costs, loan payoffs, net proceeds), and per-partner metrics.",
        "input_schema": {
            "type": "object",
            "properties": {
                "vcode": {
                    "type": "string",
                    "description": "The deal's vcode identifier",
                },
            },
            "required": ["vcode"],
        },
    },
    {
        "name": "get_loan_details",
        "description": "Get loan details for a deal: loan ID, origination amount, rate, type (fixed/floating), maturity date, current balance.",
        "input_schema": {
            "type": "object",
            "properties": {
                "vcode": {
                    "type": "string",
                    "description": "The deal's vcode identifier",
                },
            },
            "required": ["vcode"],
        },
    },
    {
        "name": "get_occupancy",
        "description": "Get occupancy data for a deal: date, physical occupancy %, unit count.",
        "input_schema": {
            "type": "object",
            "properties": {
                "vcode": {
                    "type": "string",
                    "description": "The deal's vcode identifier",
                },
            },
            "required": ["vcode"],
        },
    },
    {
        "name": "get_financial_statement",
        "description": "Get income statement data for a deal from ISBS. Returns revenue, expenses, NOI by period. Specify source: 'actual' (Interim IS), 'budget' (Budget IS), 'underwriting' (Projected IS).",
        "input_schema": {
            "type": "object",
            "properties": {
                "vcode": {
                    "type": "string",
                    "description": "The deal's vcode identifier",
                },
                "source": {
                    "type": "string",
                    "description": "Data source: 'actual', 'budget', or 'underwriting'",
                    "enum": ["actual", "budget", "underwriting"],
                },
                "year": {
                    "type": "integer",
                    "description": "Optional: filter to a specific year",
                },
            },
            "required": ["vcode"],
        },
    },
    {
        "name": "get_waterfall_structure",
        "description": "Get the waterfall distribution structure for an entity. Returns the CF and Capital waterfall steps with their allocation rules.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_code": {
                    "type": "string",
                    "description": "The entity's PropCode in the waterfall table",
                },
            },
            "required": ["entity_code"],
        },
    },
    {
        "name": "get_one_pager",
        "description": "Get the One Pager investor report data for a deal. Returns capitalization stack (debt, pref equity, partner equity, LTV, PE exposure), property performance (revenue, expenses, NOI, DSCR for YTD actual, budget, at close, projected year-end), PE performance (committed, funded, return of capital, current balance, accrued, ROE), and general deal information. This is the primary tool for answering questions about current NOI, PE exposure, deal terms, DSCR, and property performance.",
        "input_schema": {
            "type": "object",
            "properties": {
                "vcode": {
                    "type": "string",
                    "description": "The deal's vcode identifier",
                },
                "quarter": {
                    "type": "string",
                    "description": "Quarter to report on (e.g., '2026-Q2'). Defaults to the latest available quarter if omitted.",
                },
            },
            "required": ["vcode"],
        },
    },
    {
        "name": "get_annual_forecast",
        "description": "Get the annual forecast/projection table for a deal. Returns Revenue, Expenses, NOI, Tax Abatement, Debt Service (Interest + Principal), Capital Expenditures, FAD (Funds Available for Distribution), and DSCR by year. Use this to answer questions about projected NOI, when DSCR drops below a threshold, future cash flows, and year-over-year trends.",
        "input_schema": {
            "type": "object",
            "properties": {
                "vcode": {
                    "type": "string",
                    "description": "The deal's vcode identifier",
                },
            },
            "required": ["vcode"],
        },
    },
    {
        "name": "get_sold_returns",
        "description": "Get returns for sold deals computed from accounting history (not the waterfall engine). Returns IRR, ROE, MOIC, contributions, and distributions for each sold deal plus a portfolio total. Use this for any question about sold deals, historical returns, or realized performance. Optionally pass a vcode for a single deal's detailed activity.",
        "input_schema": {
            "type": "object",
            "properties": {
                "vcode": {
                    "type": "string",
                    "description": "Optional: specific sold deal vcode for detailed activity breakdown. Omit for portfolio summary.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_capitalization",
        "description": "Get the current capitalization stack for a deal: debt outstanding (from ISBS balance sheet), preferred equity, partner equity, total capitalization, LTV, current valuation, cap rate, and PE exposure on cap and value. Use this for questions about leverage, LTV, debt levels, equity balances, and capital structure.",
        "input_schema": {
            "type": "object",
            "properties": {
                "vcode": {
                    "type": "string",
                    "description": "The deal's vcode identifier",
                },
            },
            "required": ["vcode"],
        },
    },
    {
        "name": "compare_deals",
        "description": "Compare 2 or more deals side-by-side. Runs the waterfall computation for each deal and returns IRR, ROE, MOIC, contributions, distributions, sale date, and capitalization metrics in a comparison format. Use when the user asks to compare deals or rank them by a metric.",
        "input_schema": {
            "type": "object",
            "properties": {
                "vcodes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of 2+ deal vcodes to compare",
                },
            },
            "required": ["vcodes"],
        },
    },
    {
        "name": "get_debt_service",
        "description": "Get the debt service schedule for a deal: loan amortization by period with interest, principal, ending balance, and balloon payment detection. Returns a summary per loan (rate, maturity, balance) plus the schedule. Use for questions about loan payments, maturity dates, remaining balance, and balloon payoffs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "vcode": {
                    "type": "string",
                    "description": "The deal's vcode identifier",
                },
            },
            "required": ["vcode"],
        },
    },
    {
        "name": "get_cash_management",
        "description": "Get the cash management schedule for a deal: beginning cash, operating cash flow, CapEx funded from reserves, capital calls, distributable cash, and ending cash by period. Use for questions about cash reserves, distributable cash, CapEx funding, and cash flow timing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "vcode": {
                    "type": "string",
                    "description": "The deal's vcode identifier",
                },
            },
            "required": ["vcode"],
        },
    },
    {
        "name": "get_tenant_roster",
        "description": "Get the tenant roster for a commercial deal: tenant names, SF leased, lease dates, rent per SF, % of GLA/ABR, vacancy, and lease maturity rollover by year. Use for questions about tenants, lease expirations, occupancy, and rent rolls.",
        "input_schema": {
            "type": "object",
            "properties": {
                "vcode": {
                    "type": "string",
                    "description": "The deal's vcode identifier",
                },
            },
            "required": ["vcode"],
        },
    },
    {
        "name": "get_user_feedback",
        "description": "Get all user-submitted feedback requests (errors, improvements, report requests, analysis requests) with full message threads. Use this during design sessions to understand what users have reported and requested.",
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": "Filter by status: open, in_progress, resolved, closed. Omit for all.",
                },
                "request_type": {
                    "type": "string",
                    "description": "Filter by type: error, improvement, report, analysis. Omit for all.",
                },
            },
        },
    },
]

# ── System prompt ────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI assistant embedded in the Waterfall XIRR application — a real estate investment analysis platform used by Peaceable Street Capital.

You have access to the full portfolio database, deal computation engine, and financial data. You can help with:
- Looking up deal information, returns, and metrics
- Querying accounting data (contributions, distributions, capital balances)
- Analyzing financial statements (income, expenses, NOI)
- Checking loan details, occupancy, and valuations
- Running ad-hoc SQL queries for custom analysis
- Explaining how waterfall structures work
- Computing IRR, ROE, MOIC for specific deals
- Getting One Pager data (cap stack, property performance, PE metrics, deal terms)
- Getting annual forecast projections (Revenue, Expenses, NOI, Debt Service, FAD, DSCR by year)
- Checking capitalization stack, LTV, debt levels, PE exposure
- Comparing deals side-by-side on returns and metrics
- Analyzing sold deal returns and historical performance
- Reviewing debt service schedules, loan amortization, and balloon payments
- Checking cash reserves, distributable cash, and CapEx funding
- Looking up tenant rosters, lease expirations, and rent rolls (commercial deals)

Tool selection tips:
- Use get_one_pager for current NOI, PE exposure, DSCR, cap stack, deal terms, property performance
- Use get_annual_forecast for projected/future NOI, multi-year trends, DSCR forecasts, FAD projections
- Use compute_deal_returns for IRR, ROE, MOIC, sale date, sale proceeds
- Use get_capitalization for LTV, debt outstanding, equity balances, capital structure
- Use compare_deals when comparing 2+ deals (more efficient than calling compute_deal_returns multiple times)
- Use get_sold_returns for sold/historical deals — they use accounting data, NOT the waterfall engine
- Use get_debt_service for loan schedules, amortization, balloon payments, maturity dates
- Use get_cash_management for cash reserves, distributable cash, CapEx from reserves
- Use get_tenant_roster for tenant info, lease expirations, rent rolls (commercial deals only)

Key conventions:
- Cashflow signs: negative = contribution (money in), positive = distribution (money out)
- Rates are decimals (0.08 = 8%)
- Revenue accounts: 4xxx, Expense accounts: 5xxx
- ISBS sources: Interim IS = Actuals (YTD cumulative), Budget IS = periodic monthly, Projected IS = Underwriting
- "Sold" deals have Sale_Status = 'SOLD' in the deals table

When answering questions:
- Be concise and direct
- Format numbers as currency ($1,234,567) or percentages (12.34%) as appropriate
- When showing tables, use markdown formatting
- If a query returns too much data, summarize the key findings
- Always explain what the numbers mean in context

When the user asks a question, use the page context (provided below) to understand what they are looking at. If they ask about a deal without specifying which one, assume they mean the deal currently selected on their page. If their question requires data from a different tab (e.g., asking about expected returns while on the One Pager), use the current deal's vcode to fetch that data from the appropriate source (e.g., compute_deal_returns for Deal Analysis metrics).

Today's date is """ + date.today().isoformat() + "."


# ── Tool execution ───────────────────────────────────────────────────

def _get_data():
    return data_service.get_data()


def _df_to_json(df, limit=100):
    """Convert DataFrame to JSON-friendly list of dicts, truncated with summary."""
    if df is None or df.empty:
        return {"rows": [], "count": 0}
    total = len(df)
    truncated = total > limit
    df_out = df.head(limit).copy()
    # Convert dates and timestamps to strings
    for col in df_out.columns:
        if pd.api.types.is_datetime64_any_dtype(df_out[col]):
            df_out[col] = df_out[col].dt.strftime("%Y-%m-%d").fillna("")
        elif df_out[col].dtype == object:
            # Handle Python date/datetime objects in object-dtype columns
            df_out[col] = df_out[col].apply(
                lambda x: x.isoformat() if isinstance(x, (date, datetime)) else x
            )
    records = df_out.fillna("").to_dict(orient="records")
    result = {"rows": records, "count": total, "showing": min(limit, total)}
    if truncated:
        result["truncated"] = True
        result["note"] = f"Showing first {limit} of {total} rows. Use query_database with LIMIT/WHERE for more specific results."
    # Add numeric column summaries for large results
    if total > 20:
        summaries = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                vals = df[col].dropna()
                if len(vals) > 0:
                    summaries[col] = {
                        "sum": round(float(vals.sum()), 2),
                        "min": round(float(vals.min()), 2),
                        "max": round(float(vals.max()), 2),
                    }
        if summaries:
            result["column_summaries"] = summaries
    return result


def _tool_resolve_deal(inp):
    """Fuzzy match a deal name and return vcode + metadata."""
    query = str(inp["name"]).strip().lower()
    data = _get_data()
    inv = data["inv"]

    if "Investment_Name" not in inv.columns or "vcode" not in inv.columns:
        return json.dumps({"error": "Deal data not available"})

    # Build searchable frame
    names = inv[["vcode", "Investment_Name"]].dropna(subset=["Investment_Name"]).copy()
    names["name_lower"] = names["Investment_Name"].str.lower()

    # Exact match first
    exact = names[names["name_lower"] == query]
    if not exact.empty:
        row = exact.iloc[0]
        return json.dumps({"vcode": row["vcode"], "name": row["Investment_Name"], "match": "exact"})

    # Contains match
    contains = names[names["name_lower"].str.contains(query, na=False)]
    if len(contains) == 1:
        row = contains.iloc[0]
        return json.dumps({"vcode": row["vcode"], "name": row["Investment_Name"], "match": "contains"})
    elif len(contains) > 1:
        matches = [{"vcode": r["vcode"], "name": r["Investment_Name"]} for _, r in contains.head(10).iterrows()]
        return json.dumps({"matches": matches, "count": len(contains), "message": "Multiple matches found. Please specify."})

    # Token overlap match — each word in query checked against deal names
    query_tokens = set(query.split())
    scores = []
    for _, r in names.iterrows():
        name_tokens = set(r["name_lower"].split())
        overlap = len(query_tokens & name_tokens)
        if overlap > 0:
            scores.append((overlap, r["vcode"], r["Investment_Name"]))
    if scores:
        scores.sort(key=lambda x: -x[0])
        if scores[0][0] > 0:
            matches = [{"vcode": s[1], "name": s[2], "token_overlap": s[0]} for s in scores[:5]]
            if len(matches) == 1:
                return json.dumps({"vcode": matches[0]["vcode"], "name": matches[0]["name"], "match": "token"})
            return json.dumps({"matches": matches, "message": "Possible matches by keyword overlap."})

    return json.dumps({"error": f"No deals found matching '{inp['name']}'"})


def _enrich_deal_context(vcode: str) -> str:
    """Pre-fetch deal metadata for a vcode to inject into system prompt."""
    try:
        data = _get_data()
        inv = data["inv"]
        match = inv[inv["vcode"].astype(str).str.strip() == vcode.strip()]
        if match.empty:
            return ""

        row = match.iloc[0]
        parts = [f"\nDeal Details (pre-loaded for {vcode}):"]
        for field in ["Investment_Name", "Asset_Type", "Sale_Status", "Acquisition_Date",
                       "Sale_Date", "Acquisition_Price", "Portfolio_Name", "vCity",
                       "vState", "Units_SF", "Investment_Strategy", "Operating_Partner",
                       "Asset_Manager"]:
            if field in row.index and pd.notna(row[field]) and str(row[field]).strip():
                val = row[field]
                if isinstance(val, pd.Timestamp):
                    val = val.strftime("%Y-%m-%d")
                parts.append(f"  {field}: {val}")
        return "\n".join(parts)
    except Exception:
        return ""


def _error_hint(tool_name: str, tool_input: dict, exc: Exception) -> str:
    """Return a contextual hint for common errors."""
    msg = str(exc).lower()
    vcode = tool_input.get("vcode", tool_input.get("entity_code", ""))

    if "not found" in msg or "no data" in msg:
        return f"Deal '{vcode}' may not exist. Use resolve_deal to find the correct vcode."
    if "no waterfall" in msg:
        return f"Deal '{vcode}' has no waterfall steps configured. Check get_waterfall_structure."
    if "no forecast" in msg or "fc_deal" in msg:
        return f"Deal '{vcode}' has no forecast data. It may be a sold deal or missing valuation data."
    if "keyerror" in msg or "key error" in msg:
        return "A required data field is missing. The deal may have incomplete data in the database."
    if "timeout" in msg or "timed out" in msg:
        return "The computation timed out. Try a simpler query or a different deal."
    return ""


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool and return the result as a string."""
    try:
        if tool_name == "resolve_deal":
            return _tool_resolve_deal(tool_input)
        elif tool_name == "list_deals":
            return _tool_list_deals(tool_input)
        elif tool_name == "query_deal_data":
            return _tool_query_deal_data(tool_input)
        elif tool_name == "query_accounting":
            return _tool_query_accounting(tool_input)
        elif tool_name == "query_database":
            return _tool_query_database(tool_input)
        elif tool_name == "get_portfolio_summary":
            return _tool_portfolio_summary(tool_input)
        elif tool_name == "compute_deal_returns":
            return _tool_compute_deal_returns(tool_input)
        elif tool_name == "get_loan_details":
            return _tool_get_loans(tool_input)
        elif tool_name == "get_occupancy":
            return _tool_get_occupancy(tool_input)
        elif tool_name == "get_financial_statement":
            return _tool_get_financial_statement(tool_input)
        elif tool_name == "get_waterfall_structure":
            return _tool_get_waterfall(tool_input)
        elif tool_name == "get_one_pager":
            return _tool_get_one_pager(tool_input)
        elif tool_name == "get_annual_forecast":
            return _tool_get_annual_forecast(tool_input)
        elif tool_name == "get_sold_returns":
            return _tool_get_sold_returns(tool_input)
        elif tool_name == "get_capitalization":
            return _tool_get_capitalization(tool_input)
        elif tool_name == "compare_deals":
            return _tool_compare_deals(tool_input)
        elif tool_name == "get_debt_service":
            return _tool_get_debt_service(tool_input)
        elif tool_name == "get_cash_management":
            return _tool_get_cash_management(tool_input)
        elif tool_name == "get_tenant_roster":
            return _tool_get_tenant_roster(tool_input)
        elif tool_name == "get_user_feedback":
            return _tool_get_user_feedback(tool_input)
        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
    except Exception as e:
        logger.exception(f"Tool execution error: {tool_name}")
        hint = _error_hint(tool_name, tool_input, e)
        msg = f"Error in {tool_name}: {str(e)}"
        if hint:
            msg += f". Hint: {hint}"
        return json.dumps({"error": msg})


def _tool_list_deals(inp):
    data = _get_data()
    inv = data["inv"].copy()
    status_filter = inp.get("status_filter", "all")

    cols = ["Investment_Name", "vcode", "Asset_Type", "Sale_Status",
            "Acquisition_Date", "Sale_Date", "Portfolio_Name", "Operating_Partner"]
    available = [c for c in cols if c in inv.columns]
    result = inv[available].copy()

    if status_filter == "sold":
        result = result[result.get("Sale_Status", pd.Series()).str.upper() == "SOLD"]
    elif status_filter == "active":
        result = result[result.get("Sale_Status", pd.Series()).str.upper() != "SOLD"]

    partner = inp.get("operating_partner", "").strip()
    if partner and "Operating_Partner" in result.columns:
        result = result[result["Operating_Partner"].fillna("").str.lower().str.contains(partner.lower())]

    return json.dumps(_df_to_json(result, limit=200))


def _tool_query_deal_data(inp):
    data = _get_data()
    vcode = str(inp["vcode"]).strip()
    inv = data["inv"]
    match = inv[inv["vcode"].astype(str).str.strip() == vcode]
    if match.empty:
        return json.dumps({"error": f"Deal '{vcode}' not found. Use resolve_deal to find the correct vcode, or list_deals to see all deals."})
    row = match.iloc[0].to_dict()
    # Convert non-serializable types
    clean = {}
    for k, v in row.items():
        if pd.isna(v):
            clean[k] = None
        elif isinstance(v, pd.Timestamp):
            clean[k] = v.isoformat()
        elif hasattr(v, 'item'):  # numpy scalar
            clean[k] = v.item()
        else:
            clean[k] = v
    return json.dumps(clean, default=str)


def _tool_query_accounting(inp):
    data = _get_data()
    vcode = str(inp["vcode"]).strip()
    acct = data["acct"]

    # Map vcode to InvestmentID
    from loaders import build_investmentid_to_vcode
    id_map = build_investmentid_to_vcode(data["inv"])
    inv_ids = [k for k, v in id_map.items() if str(v).strip() == vcode]

    if not inv_ids:
        return json.dumps({"error": f"No InvestmentID found for vcode '{vcode}'. Use resolve_deal to verify the vcode."})

    filtered = acct[acct["InvestmentID"].isin(inv_ids)].copy()

    if "investor_id" in inp and inp["investor_id"]:
        filtered = filtered[filtered["InvestorID"].str.strip().str.upper() == inp["investor_id"].strip().upper()]

    cols = ["EffectiveDate", "InvestorID", "MajorType", "Typename", "Capital", "Amount"]
    available = [c for c in cols if c in filtered.columns]
    filtered = filtered[available].sort_values("EffectiveDate") if "EffectiveDate" in filtered.columns else filtered[available]

    return json.dumps(_df_to_json(filtered, limit=200))


def _tool_query_database(inp):
    sql = inp["sql"].strip()
    limit = min(inp.get("limit", 100), 500)

    # Safety: only allow SELECT
    first_word = sql.split()[0].upper() if sql.split() else ""
    if first_word not in ("SELECT", "WITH"):
        return json.dumps({"error": "Only SELECT queries are allowed"})

    # Block dangerous keywords
    sql_upper = sql.upper()
    for keyword in ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "CREATE", "GRANT", "REVOKE"]:
        if keyword in sql_upper.split():
            return json.dumps({"error": f"Query contains forbidden keyword: {keyword}"})

    from flask_app.db import get_engine
    engine = get_engine()
    try:
        df = pd.read_sql(sql, engine)
        return json.dumps(_df_to_json(df, limit=limit))
    except Exception as e:
        return json.dumps({"error": f"SQL error: {str(e)}"})


def _tool_portfolio_summary(inp):
    data = _get_data()
    inv = data["inv"]

    active = inv[inv.get("Sale_Status", pd.Series(dtype=str)).str.upper() != "SOLD"] if "Sale_Status" in inv.columns else inv

    summary = {
        "total_deals": len(active),
        "sold_deals": len(inv) - len(active),
        "asset_types": active["Asset_Type"].value_counts().to_dict() if "Asset_Type" in active.columns else {},
    }
    return json.dumps(summary, default=str)


def _tool_compute_deal_returns(inp):
    vcode = str(inp["vcode"]).strip()
    try:
        from flask_app.services.compute_service import get_cached_deal_result
        data = _get_data()
        start_year = current_app.config["DEFAULT_START_YEAR"]
        horizon_years = current_app.config["DEFAULT_HORIZON_YEARS"]
        pro_yr_base = current_app.config["PRO_YR_BASE_DEFAULT"]
        actuals_through = current_app.config.get("ACTUALS_THROUGH")
        result = get_cached_deal_result(
            vcode, start_year, horizon_years, pro_yr_base, data,
            actuals_through=actuals_through,
        )
        if not result or "partner_results" not in result:
            return json.dumps({"error": f"Could not compute returns for '{vcode}'. The deal may have no waterfall steps or no forecast data. Use get_waterfall_structure to check."})

        pr = result["partner_results"]
        # Extract summary metrics (keys are lowercase from compute.py)
        summary = []
        for row in pr:
            summary.append({
                "partner": row.get("partner", ""),
                "contributions": row.get("contributions", 0),
                "cf_distributions": row.get("cf_distributions", 0),
                "cap_distributions": row.get("cap_distributions", 0),
                "total_distributions": row.get("total_distributions", 0),
                "irr": row.get("irr", ""),
                "roe": row.get("roe", ""),
                "moic": row.get("moic", ""),
                "is_pref_equity": row.get("is_pref_equity", False),
                "capital_outstanding": row.get("capital_outstanding", 0),
                "unrealized_nav": row.get("unrealized_nav", 0),
            })
        # Deal-level summary from compute engine
        ds = result.get("deal_summary", {})
        deal_info = {
            "deal_irr": ds.get("deal_irr"),
            "deal_roe": ds.get("deal_roe"),
            "deal_moic": ds.get("deal_moic"),
            "total_contributions": ds.get("total_contributions", 0),
            "total_distributions": ds.get("total_distributions", 0),
        }
        if result.get("sale_me"):
            deal_info["sale_date"] = str(result["sale_me"])
        if result.get("sale_dbg"):
            dbg = result["sale_dbg"]
            if isinstance(dbg, dict):
                deal_info["sale_proceeds_breakdown"] = {
                    "noi_12m": dbg.get("NOI_12m_After_Sale"),
                    "cap_rate": dbg.get("CapRate_Sale"),
                    "implied_value": dbg.get("Implied_Value"),
                    "selling_cost": dbg.get("Less_Selling_Cost_2pct"),
                    "value_net_selling_cost": dbg.get("Value_Net_Selling_Cost"),
                    "loan_balances": dbg.get("Less_Loan_Balances"),
                    "tax_abatement_npv": dbg.get("Tax_Abatement_NPV"),
                    "net_sale_proceeds": dbg.get("Net_Sale_Proceeds"),
                }
        return json.dumps({"partner_returns": summary, "deal_summary": deal_info}, default=str)
    except Exception as e:
        return json.dumps({"error": f"Computation error: {str(e)}"})


def _tool_get_loans(inp):
    data = _get_data()
    vcode = str(inp["vcode"]).strip()
    loans = data.get("loans")
    if loans is None or loans.empty:
        return json.dumps({"rows": [], "count": 0})

    vcode_col = "PropCode" if "PropCode" in loans.columns else "vcode"
    filtered = loans[loans[vcode_col].astype(str).str.strip().str.upper() == vcode.upper()]

    cols = ["LoanID", "mOrigLoanAmt", "fIntRate", "vRateType", "dtMaturity", "vLenderName"]
    available = [c for c in cols if c in filtered.columns]
    return json.dumps(_df_to_json(filtered[available], limit=50))


def _tool_get_occupancy(inp):
    data = _get_data()
    vcode = str(inp["vcode"]).strip()
    occ = data.get("occ")
    if occ is None or occ.empty:
        return json.dumps({"rows": [], "count": 0})

    vcode_col = next((c for c in ["PropCode", "vcode", "PropertyCode"] if c in occ.columns), None)
    if not vcode_col:
        return json.dumps({"error": "No property code column in occupancy data"})

    filtered = occ[occ[vcode_col].astype(str).str.strip().str.upper() == vcode.upper()]
    cols = ["dtPeriod", "Occ%", "nTotalUnits", "nOccupiedUnits"]
    available = [c for c in cols if c in filtered.columns]
    result = filtered[available].sort_values("dtPeriod") if "dtPeriod" in filtered.columns else filtered[available]
    return json.dumps(_df_to_json(result, limit=50))


def _tool_get_financial_statement(inp):
    data = _get_data()
    vcode = str(inp["vcode"]).strip()
    source = inp.get("source", "actual")
    year = inp.get("year")

    isbs = data.get("isbs_raw")
    if isbs is None or isbs.empty:
        return json.dumps({"error": "No ISBS data available"})

    source_map = {
        "actual": "Interim IS",
        "budget": "Budget IS",
        "underwriting": "Projected IS",
    }
    vsource = source_map.get(source, "Interim IS")

    vcode_col = next((c for c in ["PropCode", "vcode"] if c in isbs.columns), None)
    if not vcode_col:
        return json.dumps({"error": "No property code column in ISBS"})

    filtered = isbs[
        (isbs[vcode_col].astype(str).str.strip().str.upper() == vcode.upper()) &
        (isbs["vSource"] == vsource)
    ].copy()

    if year and "dtPeriod" in filtered.columns:
        filtered["dtPeriod"] = pd.to_datetime(filtered["dtPeriod"], format="mixed", errors="coerce")
        filtered = filtered[filtered["dtPeriod"].dt.year == year]

    cols = ["dtPeriod", "vAccount", "vAccountType", "mAmount", "mAmount_norm"]
    available = [c for c in cols if c in filtered.columns]
    return json.dumps(_df_to_json(filtered[available], limit=200))


def _tool_get_waterfall(inp):
    data = _get_data()
    entity_code = str(inp["entity_code"]).strip()
    wf = data.get("wf")
    if wf is None or wf.empty:
        return json.dumps({"error": "No waterfall data available"})

    pc_col = "PropCode" if "PropCode" in wf.columns else "propcode"
    filtered = wf[wf[pc_col].astype(str).str.strip() == entity_code]

    if filtered.empty:
        return json.dumps({"error": f"No waterfall found for entity '{entity_code}'. Note: entity_code is the PropCode in the waterfall table, not always the deal vcode. Use query_database to check: SELECT DISTINCT PropCode FROM waterfalls"})

    cols = ["iOrder", "PropCode", "vState", "FXRate", "nPercent", "mAmount",
            "vtranstype", "vAmtType", "vNotes"]
    available = [c for c in cols if c in filtered.columns]
    return json.dumps(_df_to_json(filtered[available], limit=100))


def _tool_get_one_pager(inp):
    """Get One Pager investor report data for a deal."""
    vcode = str(inp["vcode"]).strip()
    quarter = inp.get("quarter")
    try:
        from flask_app.services.financials_service import get_one_pager_data
        data = _get_data()
        result = get_one_pager_data(
            vcode, quarter, data["inv"], data["isbs_raw"],
            data["mri_loans_raw"], data["mri_val"],
            data["wf"], data["acct"],
            occupancy_raw=data["occupancy_raw"],
            budget_econ_occ=data.get("budget_econ_occ"),
            deal_terms=data.get("deal_terms_raw"),
            at_close_noi=data.get("at_close_noi_raw"),
            event_dates=data.get("event_dates_raw"),
            relationships=data.get("relationships_raw"),
            inspection=data.get("inspection_raw"),
        )
        if not result:
            return json.dumps({"error": f"No One Pager data for '{vcode}'. The deal may lack ISBS data or valuations needed for the One Pager."})

        # Flatten for concise tool output
        out = {"vcode": vcode}
        if result.get("available_quarters"):
            out["available_quarters"] = result["available_quarters"][:8]
        if result.get("general"):
            out["general"] = result["general"]
        if result.get("cap_stack"):
            out["cap_stack"] = result["cap_stack"]
        if result.get("property_performance"):
            out["property_performance"] = result["property_performance"]
        if result.get("pe_performance"):
            out["pe_performance"] = result["pe_performance"]
        # Skip comments — not useful for AI queries
        return json.dumps(out, default=str)
    except Exception as e:
        return json.dumps({"error": f"One Pager error: {str(e)}"})


def _tool_get_annual_forecast(inp):
    """Get annual forecast/projection table for a deal."""
    vcode = str(inp["vcode"]).strip()
    try:
        from flask_app.services.compute_service import get_cached_deal_result
        from reporting import annual_aggregation_table
        data = _get_data()
        start_year = current_app.config["DEFAULT_START_YEAR"]
        horizon_years = current_app.config["DEFAULT_HORIZON_YEARS"]
        pro_yr_base = current_app.config["PRO_YR_BASE_DEFAULT"]
        actuals_through = current_app.config.get("ACTUALS_THROUGH")
        result = get_cached_deal_result(
            vcode, start_year, horizon_years, pro_yr_base, data,
            actuals_through=actuals_through,
        )
        if not result:
            return json.dumps({"error": f"Could not compute forecast for {vcode}"})

        fc_display = result.get("fc_deal_display")
        if fc_display is None or fc_display.empty:
            return json.dumps({"error": f"No forecast data for {vcode}"})

        cap_events = result.get("cap_events_df")
        proceeds_by_year = None
        if cap_events is not None and not cap_events.empty and "Year" in cap_events.columns:
            proceeds_by_year = cap_events.groupby("Year")["amount"].sum()

        table = annual_aggregation_table(
            fc_display, start_year, horizon_years,
            proceeds_by_year=proceeds_by_year,
            cf_alloc=result.get("cf_alloc"),
            cap_alloc=result.get("cap_alloc"),
            cash_schedule=result.get("cash_schedule"),
        )

        if table.empty or "Year" not in table.columns:
            return json.dumps({"error": f"No forecast data for {vcode}"})

        # Pivot to {row_label: {year: value}} for readability
        wide = table.set_index("Year").T
        # Filter to key rows only (skip blank separators and waterfall detail)
        key_rows = [
            "Revenues", "Expenses", "NOI", "Tax Abatement",
            "Interest", "Principal", "Total Debt Service",
            "Capital Expenditures", "Other Below-the-Line",
            "Funds Available for Distribution", "Debt Service Coverage Ratio",
        ]
        out = {"vcode": vcode, "years": [int(y) for y in wide.columns]}
        forecast = {}
        for label in key_rows:
            if label in wide.index:
                row_data = wide.loc[label]
                vals = {}
                for yr in wide.columns:
                    v = row_data.get(yr)
                    if pd.notna(v):
                        vals[str(int(yr))] = round(float(v), 2)
                if vals:
                    forecast[label] = vals
        out["forecast"] = forecast
        return json.dumps(out, default=str)
    except Exception as e:
        return json.dumps({"error": f"Forecast error: {str(e)}"})


def _tool_get_sold_returns(inp):
    """Get sold portfolio returns from accounting history."""
    vcode = inp.get("vcode", "").strip() if inp.get("vcode") else ""
    try:
        from flask_app.services.sold_service import get_sold_deals, compute_all_sold_returns, build_deal_detail
        data = _get_data()
        inv = data["inv"]
        acct = data["acct"]

        inv_sold = get_sold_deals(inv)
        if inv_sold.empty:
            return json.dumps({"error": "No sold deals found in the portfolio."})

        if vcode:
            # Single deal detail
            detail = build_deal_detail(vcode, inv_sold, acct, inv)
            if not detail or not detail.get("rows"):
                return json.dumps({"error": f"No sold deal activity found for '{vcode}'. Check that it has Sale_Status='SOLD'."})
            # Truncate rows for readability
            rows = detail["rows"]
            summary = detail.get("summary", {})
            out = {
                "vcode": vcode,
                "deal_name": summary.get("deal_name", vcode),
                "summary": {
                    "total_contributions": summary.get("Total Contributions", 0),
                    "total_distributions": summary.get("Total Distributions", 0),
                    "irr": summary.get("IRR"),
                    "roe": summary.get("ROE"),
                    "moic": summary.get("MOIC"),
                },
                "activity_rows": rows[:50],
                "total_rows": len(rows),
            }
            if len(rows) > 50:
                out["note"] = f"Showing first 50 of {len(rows)} activity rows."
            return json.dumps(out, default=str)
        else:
            # Portfolio summary
            df = compute_all_sold_returns(inv_sold, acct, inv)
            if df.empty:
                return json.dumps({"error": "Could not compute sold portfolio returns."})
            deals = []
            portfolio_total = None
            for _, row in df.iterrows():
                entry = {
                    "deal_name": row.get("Investment Name", ""),
                    "vcode": row.get("vcode", ""),
                    "acquisition_date": row.get("Acquisition Date", ""),
                    "sale_date": row.get("Sale Date", ""),
                    "contributions": row.get("Total Contributions", 0),
                    "distributions": row.get("Total Distributions", 0),
                    "irr": row.get("IRR"),
                    "roe": row.get("ROE"),
                    "moic": row.get("MOIC"),
                }
                if row.get("_is_deal_total"):
                    portfolio_total = entry
                else:
                    deals.append(entry)
            out = {"deals": deals}
            if portfolio_total:
                out["portfolio_total"] = portfolio_total
            return json.dumps(out, default=str)
    except Exception as e:
        return json.dumps({"error": f"Sold portfolio error: {str(e)}"})


def _tool_get_capitalization(inp):
    """Get capitalization stack for a deal."""
    vcode = str(inp["vcode"]).strip()
    try:
        from compute import get_deal_capitalization
        from consolidation import get_property_vcodes_for_deal
        data = _get_data()
        inv, acct, wf = data["inv"], data["acct"], data["wf"]
        mri_loans = data.get("mri_loans_raw")
        mri_val = data.get("mri_val")
        isbs_raw = data.get("isbs_raw")

        # Check deal exists
        match = inv[inv["vcode"].astype(str).str.strip() == vcode]
        if match.empty:
            return json.dumps({"error": f"Deal {vcode} not found. Use resolve_deal or list_deals to find the correct vcode."})

        property_vcodes = get_property_vcodes_for_deal(vcode, inv)
        cap = get_deal_capitalization(
            acct, inv, wf, mri_val, mri_loans, vcode,
            property_vcodes=property_vcodes, isbs_raw=isbs_raw,
        )

        # Compute LTV
        ltv = None
        if cap.get("current_valuation") and cap["current_valuation"] > 0:
            ltv = round(cap["debt"] / cap["current_valuation"], 4)

        result = {
            "vcode": vcode,
            "deal_name": match.iloc[0].get("Investment_Name", vcode),
            "debt": round(cap.get("debt", 0), 2),
            "pref_equity": round(cap.get("pref_equity", 0), 2),
            "partner_equity": round(cap.get("partner_equity", 0), 2),
            "total_cap": round(cap.get("total_cap", 0), 2),
            "current_valuation": round(cap.get("current_valuation", 0), 2),
            "cap_rate": cap.get("cap_rate", 0),
            "ltv": ltv,
            "pe_exposure_on_cap": round(cap.get("pe_exposure_cap", 0), 4),
            "pe_exposure_on_value": round(cap.get("pe_exposure_value", 0), 4),
        }
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": f"Capitalization error for {vcode}: {str(e)}"})


def _tool_compare_deals(inp):
    """Compare 2+ deals side-by-side on returns and capitalization."""
    vcodes = inp.get("vcodes", [])
    if len(vcodes) < 2:
        return json.dumps({"error": "Please provide at least 2 vcodes to compare."})
    if len(vcodes) > 10:
        return json.dumps({"error": "Maximum 10 deals can be compared at once."})

    from flask_app.services.compute_service import get_cached_deal_result
    from compute import get_deal_capitalization
    from consolidation import get_property_vcodes_for_deal

    data = _get_data()
    inv = data["inv"]
    start_year = current_app.config["DEFAULT_START_YEAR"]
    horizon_years = current_app.config["DEFAULT_HORIZON_YEARS"]
    pro_yr_base = current_app.config["PRO_YR_BASE_DEFAULT"]
    actuals_through = current_app.config.get("ACTUALS_THROUGH")

    comparisons = []
    for vcode in vcodes:
        vcode = str(vcode).strip()
        entry = {"vcode": vcode}

        # Deal name
        match = inv[inv["vcode"].astype(str).str.strip() == vcode]
        if match.empty:
            entry["error"] = f"Deal {vcode} not found"
            comparisons.append(entry)
            continue
        entry["deal_name"] = match.iloc[0].get("Investment_Name", vcode)
        entry["asset_type"] = match.iloc[0].get("Asset_Type", "")

        # Returns
        try:
            result = get_cached_deal_result(
                vcode, start_year, horizon_years, pro_yr_base, data,
                actuals_through=actuals_through,
            )
            if result and result.get("deal_summary"):
                ds = result["deal_summary"]
                entry["deal_irr"] = ds.get("deal_irr")
                entry["deal_roe"] = ds.get("deal_roe")
                entry["deal_moic"] = ds.get("deal_moic")
                entry["total_contributions"] = ds.get("total_contributions", 0)
                entry["total_distributions"] = ds.get("total_distributions", 0)
            if result and result.get("sale_me"):
                entry["sale_date"] = str(result["sale_me"])
        except Exception as e:
            entry["returns_error"] = str(e)

        # Capitalization
        try:
            property_vcodes = get_property_vcodes_for_deal(vcode, inv)
            cap = get_deal_capitalization(
                data["acct"], inv, data["wf"], data.get("mri_val"),
                data.get("mri_loans_raw"), vcode,
                property_vcodes=property_vcodes, isbs_raw=data.get("isbs_raw"),
            )
            entry["debt"] = round(cap.get("debt", 0), 2)
            entry["pref_equity"] = round(cap.get("pref_equity", 0), 2)
            entry["total_cap"] = round(cap.get("total_cap", 0), 2)
            if cap.get("current_valuation") and cap["current_valuation"] > 0:
                entry["ltv"] = round(cap["debt"] / cap["current_valuation"], 4)
        except Exception:
            pass

        comparisons.append(entry)

    return json.dumps({"comparisons": comparisons}, default=str)


def _tool_get_debt_service(inp):
    """Get debt service schedule for a deal."""
    vcode = str(inp["vcode"]).strip()
    try:
        from flask_app.services.compute_service import get_cached_deal_result
        data = _get_data()
        start_year = current_app.config["DEFAULT_START_YEAR"]
        horizon_years = current_app.config["DEFAULT_HORIZON_YEARS"]
        pro_yr_base = current_app.config["PRO_YR_BASE_DEFAULT"]
        actuals_through = current_app.config.get("ACTUALS_THROUGH")
        result = get_cached_deal_result(
            vcode, start_year, horizon_years, pro_yr_base, data,
            actuals_through=actuals_through,
        )
        if not result:
            return json.dumps({"error": f"Could not compute debt service for '{vcode}'. The deal may have no waterfall or forecast data."})

        # Loan summary
        loans = result.get("loans", [])
        loan_summary = []
        for ln in loans:
            loan_summary.append({
                "loan_id": getattr(ln, "loan_id", ""),
                "original_amount": getattr(ln, "original_amount", 0),
                "rate": getattr(ln, "rate", 0),
                "rate_type": getattr(ln, "rate_type", ""),
                "maturity": str(getattr(ln, "maturity_date", "")),
                "lender": getattr(ln, "lender", ""),
            })

        # Amortization schedule (annual summary)
        loan_sched = result.get("loan_sched")
        annual_debt = {}
        if loan_sched is not None and not loan_sched.empty:
            ls = loan_sched.copy()
            ls["event_date"] = pd.to_datetime(ls["event_date"], errors="coerce")
            ls["year"] = ls["event_date"].dt.year
            by_year = ls.groupby("year").agg(
                interest=("interest", "sum"),
                principal=("principal", "sum"),
                ending_balance=("ending_balance", "last"),
            ).reset_index()
            for _, row in by_year.iterrows():
                annual_debt[str(int(row["year"]))] = {
                    "interest": round(float(row["interest"]), 2),
                    "principal": round(float(row["principal"]), 2),
                    "ending_balance": round(float(row["ending_balance"]), 2),
                }

        out = {"vcode": vcode, "loans": loan_summary}
        if annual_debt:
            out["annual_debt_service"] = annual_debt
        if not loan_summary and not annual_debt:
            out["note"] = "No loans modeled for this deal."
        return json.dumps(out, default=str)
    except Exception as e:
        return json.dumps({"error": f"Debt service error for '{vcode}': {str(e)}"})


def _tool_get_cash_management(inp):
    """Get cash management schedule for a deal."""
    vcode = str(inp["vcode"]).strip()
    try:
        from flask_app.services.compute_service import get_cached_deal_result
        data = _get_data()
        start_year = current_app.config["DEFAULT_START_YEAR"]
        horizon_years = current_app.config["DEFAULT_HORIZON_YEARS"]
        pro_yr_base = current_app.config["PRO_YR_BASE_DEFAULT"]
        actuals_through = current_app.config.get("ACTUALS_THROUGH")
        result = get_cached_deal_result(
            vcode, start_year, horizon_years, pro_yr_base, data,
            actuals_through=actuals_through,
        )
        if not result:
            return json.dumps({"error": f"Could not compute cash schedule for '{vcode}'."})

        cash_schedule = result.get("cash_schedule")
        if cash_schedule is None or cash_schedule.empty:
            return json.dumps({"error": f"No cash schedule for '{vcode}'. The deal may lack forecast data."})

        # Annual summary
        cs = cash_schedule.copy()
        cs["event_date"] = pd.to_datetime(cs["event_date"], errors="coerce")
        cs["year"] = cs["event_date"].dt.year

        agg_cols = {}
        for col in ["operating_cf", "capex_paid", "capex_unpaid", "capital_call",
                     "deficit_covered", "distributable", "distributed"]:
            if col in cs.columns:
                agg_cols[col] = (col, "sum")
        agg_cols["ending_cash"] = ("ending_cash", "last")
        agg_cols["beginning_cash"] = ("beginning_cash", "first")

        by_year = cs.groupby("year").agg(**agg_cols).reset_index()
        annual = {}
        for _, row in by_year.iterrows():
            yr_data = {}
            for col in by_year.columns:
                if col != "year":
                    yr_data[col] = round(float(row[col]), 2)
            annual[str(int(row["year"]))] = yr_data

        # Cash summary
        cash_summary = result.get("cash_summary", {})

        out = {"vcode": vcode, "annual_cash_schedule": annual}
        if cash_summary:
            out["summary"] = {k: round(float(v), 2) if isinstance(v, (int, float)) else v
                              for k, v in cash_summary.items()}
        return json.dumps(out, default=str)
    except Exception as e:
        return json.dumps({"error": f"Cash management error for '{vcode}': {str(e)}"})


def _tool_get_tenant_roster(inp):
    """Get tenant roster for a commercial deal."""
    vcode = str(inp["vcode"]).strip()
    try:
        from flask_app.services.financials_service import get_tenant_roster
        data = _get_data()
        tenants_raw = data.get("tenants_raw")
        if tenants_raw is None or tenants_raw.empty:
            return json.dumps({"error": "No tenant data available in the database."})

        result = get_tenant_roster(tenants_raw, vcode, inv=data["inv"])
        if not result or not result.get("tenants"):
            return json.dumps({"error": f"No tenant data for '{vcode}'. This may be a residential deal (tenant rosters are for commercial properties)."})

        out = {"vcode": vcode}
        out["summary"] = result.get("summary", {})
        # Truncate tenant list for large rosters
        tenants = result.get("tenants", [])
        out["tenants"] = tenants[:30]
        if len(tenants) > 30:
            out["note"] = f"Showing first 30 of {len(tenants)} tenants."
        out["total_tenants"] = len(tenants)

        rollover = result.get("rollover", {})
        if rollover.get("maturity_by_year"):
            out["lease_maturity_by_year"] = rollover["maturity_by_year"]

        return json.dumps(out, default=str)
    except Exception as e:
        return json.dumps({"error": f"Tenant roster error for '{vcode}': {str(e)}"})


def _tool_get_user_feedback(inp):
    """Get user feedback requests for design sessions."""
    try:
        from flask_app.services.feedback_service import export_all_requests, list_requests
        status = inp.get("status")
        request_type = inp.get("request_type")

        items = list_requests(status=status, request_type=request_type)

        if not items:
            return json.dumps({"message": "No feedback requests found.", "count": 0})

        # Include full threads for up to 20 requests
        if len(items) <= 20:
            from flask_app.services.feedback_service import get_request
            detailed = []
            for item in items:
                detailed.append(get_request(item["id"]))
            return json.dumps({"requests": detailed, "count": len(detailed)}, default=str)

        # Summarize if too many
        return json.dumps({
            "requests": items[:20],
            "count": len(items),
            "note": f"Showing first 20 of {len(items)} requests. Filter by status or type for more focused results.",
        }, default=str)
    except Exception as e:
        return json.dumps({"error": f"Feedback query error: {str(e)}"})


# ── Chat session management ──────────────────────────────────────────

def get_client():
    """Get Anthropic client. API key from env var."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    return anthropic.Anthropic(api_key=api_key)


def _build_system_prompt(page_context: dict = None) -> str:
    """Build system prompt with optional page context and pre-loaded deal data."""
    prompt = SYSTEM_PROMPT
    if page_context:
        lines = ["\n\n--- Current Page Context ---"]
        if page_context.get("page"):
            lines.append(f"Page: {page_context['page']}")
        if page_context.get("path"):
            lines.append(f"Route: {page_context['path']}")
        vcode = page_context.get("current_vcode")
        if page_context.get("current_deal_name"):
            lines.append(f"Selected Deal: {page_context['current_deal_name']} (vcode: {vcode or 'unknown'})")
        elif vcode:
            lines.append(f"Selected Deal vcode: {vcode}")
        if page_context.get("selected_quarter"):
            lines.append(f"Selected Quarter: {page_context['selected_quarter']}")
        # Pre-load deal metadata so assistant doesn't need to call query_deal_data
        if vcode:
            deal_info = _enrich_deal_context(vcode)
            if deal_info:
                lines.append(deal_info)
                lines.append("(Use this pre-loaded deal info directly — no need to call query_deal_data or list_deals for this deal.)")
        lines.append("--- End Page Context ---")
        prompt += "\n".join(lines)
    return prompt


def chat_completion(messages: list, stream: bool = True, page_context: dict = None):
    """Run a chat completion with tool use loop.

    Args:
        messages: Conversation history [{role, content}, ...]
        stream: Whether to stream the response
        page_context: Current UI state (page, selected deal, quarter, etc.)

    Yields:
        dict events: {type: "text_delta"|"tool_use"|"done"|"error", ...}
    """
    client = get_client()
    system_prompt = _build_system_prompt(page_context)

    # Agentic loop — keep going until Claude stops calling tools
    max_iterations = 10
    for iteration in range(max_iterations):
        try:
            if stream:
                # Stream the response
                with client.messages.stream(
                    model="claude-sonnet-4-6",
                    max_tokens=4096,
                    system=system_prompt,
                    tools=TOOLS,
                    messages=messages,
                ) as response_stream:
                    # Collect text and tool use blocks as they stream
                    for event in response_stream:
                        if event.type == "content_block_delta":
                            if event.delta.type == "text_delta":
                                yield {"type": "text_delta", "text": event.delta.text}

                    final_message = response_stream.get_final_message()
            else:
                final_message = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=4096,
                    system=system_prompt,
                    tools=TOOLS,
                    messages=messages,
                )
        except anthropic.RateLimitError as e:
            # Rate limit hit — wait and retry once, or inform user
            import time
            if iteration < 2:  # Only retry on first couple iterations
                yield {"type": "text_delta", "text": "\n\n*Rate limit reached — waiting a moment before continuing...*\n\n"}
                time.sleep(15)
                continue
            else:
                yield {"type": "text_delta", "text": f"\n\n**Rate limit reached.** The API has a token-per-minute limit. Please wait about 30 seconds and try a simpler question, or break this into smaller queries."}
                yield {"type": "done"}
                return
        except anthropic.APIError as e:
            yield {"type": "text_delta", "text": f"\n\n**API error:** {str(e)}"}
            yield {"type": "done"}
            return

        # Check if Claude wants to use tools
        tool_use_blocks = [b for b in final_message.content if b.type == "tool_use"]

        if not tool_use_blocks:
            # No more tool calls — we're done
            yield {"type": "done"}
            return

        # Execute tools and continue the loop
        messages.append({"role": "assistant", "content": final_message.content})

        tool_results = []
        for tool_block in tool_use_blocks:
            yield {
                "type": "tool_use",
                "name": tool_block.name,
                "input": tool_block.input,
            }
            result = execute_tool(tool_block.name, tool_block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_block.id,
                "content": result,
            })

        messages.append({"role": "user", "content": tool_results})

    yield {"type": "error", "message": "Max tool iterations reached"}
