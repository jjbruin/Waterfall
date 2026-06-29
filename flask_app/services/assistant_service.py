"""AI Assistant service — Claude API integration with app-aware tools."""

import os
import json
import logging
from datetime import date

import anthropic
import pandas as pd
from flask import current_app

from flask_app.services import data_service

logger = logging.getLogger(__name__)

# ── Tool definitions ─────────────────────────────────────────────────

TOOLS = [
    {
        "name": "list_deals",
        "description": "List all deals (investments) in the portfolio. Returns deal name, vcode, asset type, status, acquisition date, and other metadata. Use this to find a deal's vcode before querying detail.",
        "input_schema": {
            "type": "object",
            "properties": {
                "status_filter": {
                    "type": "string",
                    "description": "Optional filter: 'active', 'sold', or 'all' (default: 'all')",
                    "enum": ["active", "sold", "all"],
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
        "description": "Compute IRR, ROE, and MOIC for a specific deal. Runs the full waterfall computation engine.",
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
    """Load app data (cached)."""
    db_path = current_app.config["DB_PATH"]
    pro_yr_base = current_app.config["PRO_YR_BASE_DEFAULT"]
    return data_service.load_all(db_path, pro_yr_base)


def _df_to_json(df, limit=100):
    """Convert DataFrame to JSON-friendly list of dicts, truncated."""
    if df is None or df.empty:
        return {"rows": [], "count": 0}
    total = len(df)
    df = df.head(limit)
    # Convert dates and timestamps to strings
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%d").fillna("")
    records = df.fillna("").to_dict(orient="records")
    return {"rows": records, "count": total, "showing": min(limit, total)}


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool and return the result as a string."""
    try:
        if tool_name == "list_deals":
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
        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
    except Exception as e:
        logger.exception(f"Tool execution error: {tool_name}")
        return json.dumps({"error": str(e)})


def _tool_list_deals(inp):
    data = _get_data()
    inv = data["inv"].copy()
    status_filter = inp.get("status_filter", "all")

    cols = ["Investment_Name", "vcode", "Asset_Type", "Sale_Status",
            "Acquisition_Date", "Sale_Date", "Portfolio_Name"]
    available = [c for c in cols if c in inv.columns]
    result = inv[available].copy()

    if status_filter == "sold":
        result = result[result.get("Sale_Status", pd.Series()).str.upper() == "SOLD"]
    elif status_filter == "active":
        result = result[result.get("Sale_Status", pd.Series()).str.upper() != "SOLD"]

    return json.dumps(_df_to_json(result, limit=200))


def _tool_query_deal_data(inp):
    data = _get_data()
    vcode = str(inp["vcode"]).strip()
    inv = data["inv"]
    match = inv[inv["vcode"].astype(str).str.strip() == vcode]
    if match.empty:
        return json.dumps({"error": f"Deal {vcode} not found"})
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
        return json.dumps({"error": f"No InvestmentID found for vcode {vcode}"})

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
            return json.dumps({"error": f"Could not compute returns for {vcode}"})

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
                deal_info["sale_price"] = dbg.get("sale_price") or dbg.get("value_net_selling_cost")
                deal_info["cap_rate_at_sale"] = dbg.get("cap_rate")
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
        return json.dumps({"error": f"No waterfall found for entity {entity_code}"})

    cols = ["iOrder", "PropCode", "vState", "FXRate", "nPercent", "mAmount",
            "vtranstype", "vAmtType", "vNotes"]
    available = [c for c in cols if c in filtered.columns]
    return json.dumps(_df_to_json(filtered[available], limit=100))


# ── Chat session management ──────────────────────────────────────────

def get_client():
    """Get Anthropic client. API key from env var."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    return anthropic.Anthropic(api_key=api_key)


def _build_system_prompt(page_context: dict = None) -> str:
    """Build system prompt with optional page context."""
    prompt = SYSTEM_PROMPT
    if page_context:
        lines = ["\n\n--- Current Page Context ---"]
        if page_context.get("page"):
            lines.append(f"Page: {page_context['page']}")
        if page_context.get("path"):
            lines.append(f"Route: {page_context['path']}")
        if page_context.get("current_deal_name"):
            lines.append(f"Selected Deal: {page_context['current_deal_name']} (vcode: {page_context.get('current_vcode', 'unknown')})")
        elif page_context.get("current_vcode"):
            lines.append(f"Selected Deal vcode: {page_context['current_vcode']}")
        if page_context.get("selected_quarter"):
            lines.append(f"Selected Quarter: {page_context['selected_quarter']}")
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
    for _ in range(max_iterations):
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
