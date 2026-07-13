# AI Assistant Enhancement Roadmap

## Status Key
- [ ] Not started
- [x] Complete
- [~] In progress

## Completed
- [x] Page context awareness (v47) — sends current page, deal, quarter to backend
- [x] Fix compute_deal_returns tool (v48-v49) — missing args + lowercase key mapping

## High Impact — Eliminate Common Friction Points

### 1. Deal Name Resolver + Pre-resolved Context
- [x] **Status**: Complete (v50, Jun 29, 2026)
- **resolve_deal tool**: Fuzzy name matching (exact → contains → token overlap), returns vcode + name
- **Pre-resolved context**: `_enrich_deal_context()` pre-fetches deal metadata into system prompt when page context has vcode. Includes name, asset type, dates, price, location, strategy, partners.
- **Saves**: 1-2 tool calls per conversation

### 2. One Pager Data Tool
- [x] **Status**: Complete (v52, Jun 29, 2026)
- **Tool**: `get_one_pager` — wraps `get_one_pager_data()` from `financials_service.py`
- **Answers**: "What's the current NOI?", "What's PE exposure?", "What are the deal terms?", "What's the DSCR?"
- **Returns**: cap stack (debt, PE, equity, LTV), property performance (YTD/budget/at-close/YE), PE performance, general info

### 3. Annual Forecast / Projection Tool
- [x] **Status**: Complete (v52, Jun 29, 2026)
- **Tool**: `get_annual_forecast` — wraps `annual_aggregation_table()` from compute result
- **Answers**: "What's projected NOI in 2028?", "When does DSCR drop below 1.2?", "What's FAD next year?"
- **Returns**: Key rows (Revenue, Expenses, NOI, Debt Service, CapEx, FAD, DSCR) by year

### 4. Sale Proceeds Breakdown Tool
- [x] **Status**: Complete (v52, Jun 29, 2026)
- **Approach**: Enriched existing `compute_deal_returns` tool instead of separate tool
- **Returns**: Full `sale_dbg` breakdown: NOI 12m, cap rate, implied value, selling cost, loan balances, tax abatement NPV, net sale proceeds
- **Answers**: "What are expected sale proceeds?", "What cap rate is used at sale?", "What's the loan payoff at sale?"

### 5. Capitalization Stack Tool
- [x] **Status**: Complete (Jun 30, 2026)
- **Tool**: `get_capitalization` — wraps `get_deal_capitalization()` from `compute.py`
- **Returns**: debt, pref equity, partner equity, total cap, valuation, cap rate, LTV, PE exposure on cap and value
- **Answers**: "What's the current LTV?", "How much debt is on this deal?", "What's the PE exposure?"

## Medium Impact — Richer Analysis

### 6. Deal Comparison Tool
- [x] **Status**: Complete (Jun 30, 2026)
- **Tool**: `compare_deals` — accepts 2+ vcodes, runs compute + capitalization for each
- **Returns**: Side-by-side IRR, ROE, MOIC, contributions, distributions, sale date, debt, pref equity, LTV
- **Answers**: "Compare Bearfoot and 3rd Ave", "Which deal has the highest IRR?"

### 7. Sold Portfolio Returns Tool
- [x] **Status**: Complete (Jun 30, 2026)
- **Tool**: `get_sold_returns` — wraps `compute_all_sold_returns()` and `build_deal_detail()` from `sold_service.py`
- **Summary mode** (no vcode): Returns all sold deals with IRR, ROE, MOIC, contributions, distributions + portfolio total
- **Detail mode** (with vcode): Returns accounting activity rows + summary metrics for a single sold deal
- **Answers**: "What are the sold portfolio returns?", "Show me activity for Homewood Commons"

### 8. Debt Service / Loan Schedule Tool
- [x] **Status**: Complete (Jun 30, 2026)
- **Tool**: `get_debt_service` — loan summary (rate, maturity, lender) + annual amortization (interest, principal, ending balance)
- **Data source**: `loan_sched` and `loans` from `get_cached_deal_result()`

### 9. Cash Management Tool
- [x] **Status**: Complete (Jun 30, 2026)
- **Tool**: `get_cash_management` — annual cash schedule (beginning/ending cash, operating CF, CapEx paid/unpaid, capital calls, distributable, distributed)
- **Data source**: `cash_schedule` and `cash_summary` from `get_cached_deal_result()`

### 10. Tenant Roster Tool
- [x] **Status**: Complete (Jun 30, 2026)
- **Tool**: `get_tenant_roster` — tenant list (name, SF, rent, lease dates), occupancy summary, lease maturity rollover by year
- **Data source**: `get_tenant_roster()` from `financials_service.py`

## Quality of Life — Speed & Accuracy

### 11. Smarter Error Messages
- [x] **Status**: Complete (Jun 30, 2026)
- **Approach**: `_error_hint()` function provides contextual suggestions on errors (missing deal, no waterfall, no forecast, timeout). All tool error returns now include actionable hints (e.g., "Use resolve_deal to verify the vcode").

### 12. Tool Result Truncation with Summary
- [x] **Status**: Complete (Jun 30, 2026)
- **Approach**: Enhanced `_df_to_json()` — adds `column_summaries` (sum/min/max for numeric columns) when >20 rows, `truncated` flag + guidance note when over limit

### 13. Conversation Memory / Persistence
- [x] **Status**: Complete (Jun 30, 2026)
- **Backend**: `chat_history` table (user_id PK, messages JSON, updated_at). Endpoints: GET/PUT/DELETE `/api/assistant/history`
- **Frontend**: Auto-loads history on first open, auto-saves after each response, clear button deletes server-side
- **Table creation**: `_ensure_chat_table()` in `assistant.py` (CREATE IF NOT EXISTS)

### 14. Suggested Questions
- [x] **Status**: Complete (Jun 30, 2026)
- **Frontend only**: `getSuggestedQuestions()` in `AiAssistant.vue` generates 3 context-aware questions based on current page + selected deal
- **Pages**: Dashboard, Deal Analysis, One Pager, Property Financials, Sold Portfolio, fallback
- **UX**: Clickable chips below greeting, disappear after first message sent
