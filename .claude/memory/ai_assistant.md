# AI Assistant — Embedded Claude Chat

## Overview
Embedded AI assistant in the waterfall app. Floating chat panel (bottom-right "AI" button) powered by Claude Sonnet 4.6 via the Anthropic API. Streaming SSE responses with tool activity indicators.

## Architecture

### Backend
- **Service**: `flask_app/services/assistant_service.py`
  - `TOOLS` — 19 tool definitions (JSON schema format)
  - `SYSTEM_PROMPT` — app-context system prompt with conventions, date, and usage guidance
  - `execute_tool()` — dispatcher for all tool implementations
  - `chat_completion()` — agentic loop (up to 10 tool iterations), streams SSE events
  - `get_client()` — creates Anthropic client from `ANTHROPIC_API_KEY` env var
- **API**: `flask_app/api/assistant.py`
  - `POST /api/assistant/chat` — streaming SSE chat endpoint (login_required)
  - `GET /api/assistant/status` — check if API key is configured (login_required)
- **Blueprint registered** in `flask_app/__init__.py` at `/api/assistant`

### Frontend
- **Component**: `vue_app/src/components/common/AiAssistant.vue`
  - Floating action button (FAB) in bottom-right corner
  - Slide-up chat panel (420px wide, 600px max height)
  - Streaming text via SSE (fetch + ReadableStream)
  - Tool activity chips with spinner animation
  - Typing dots indicator
  - Clear chat button
  - Enter to send, Shift+Enter for newline
  - Hidden when API key not configured (`/api/assistant/status` returns `available: false`)
- **Integrated** in `vue_app/src/App.vue` — rendered for all authenticated pages

### Dependencies
- `anthropic` Python package (added to `flask_app/requirements.txt`)
- `python-dotenv` (optional, for local `.env` loading in `flask_app/run.py`)

## Tools (19)

| Tool | Description | Data Source |
|------|-------------|-------------|
| `resolve_deal` | Fuzzy name → vcode matching | `inv` DataFrame |
| `list_deals` | Browse deals with status filter | `inv` DataFrame |
| `query_deal_data` | Deal metadata by vcode | `inv` DataFrame |
| `query_accounting` | Contributions/distributions | `acct` DataFrame |
| `query_database` | Ad-hoc read-only SQL | SQLAlchemy engine |
| `get_portfolio_summary` | Portfolio-level KPIs | `inv` DataFrame |
| `compute_deal_returns` | Full waterfall IRR/ROE/MOIC + sale proceeds | `get_cached_deal_result()` |
| `get_loan_details` | Loan data by deal | `loans` DataFrame |
| `get_occupancy` | Occupancy data by deal | `occ` DataFrame |
| `get_financial_statement` | ISBS income statement | `isbs_raw` DataFrame |
| `get_waterfall_structure` | Waterfall allocation rules | `wf` DataFrame |
| `get_one_pager` | One Pager investor report | `financials_service` |
| `get_annual_forecast` | Annual forecast projections | `annual_aggregation_table()` |
| `get_sold_returns` | Sold deal returns from accounting | `sold_service` |
| `get_capitalization` | Cap stack, debt, LTV, PE exposure | `get_deal_capitalization()` |
| `compare_deals` | Side-by-side deal comparison | multi-deal compute + cap |
| `get_debt_service` | Loan summary + annual amortization | compute result `loan_sched` |
| `get_cash_management` | Cash schedule (reserves, CapEx, distributable) | compute result `cash_schedule` |
| `get_tenant_roster` | Tenant list, occupancy, lease rollover | `financials_service` |

### Safety
- `query_database` enforces SELECT-only (blocks DROP, DELETE, UPDATE, INSERT, ALTER, etc.)
- Max 500 rows returned per query
- All tools wrapped in try/except with error logging

## Configuration

### Local Development
- API key in `.env` file (gitignored): `ANTHROPIC_API_KEY=sk-ant-...`
- `flask_app/run.py` loads `.env` via python-dotenv (optional import)

### Azure Production
- Env var set on container: `az containerapp update ... --set-env-vars ANTHROPIC_API_KEY=...`
- Set on revision v33 (Jun 16, 2026)

## Model Choice
- **Claude Sonnet 4.6** (`claude-sonnet-4-6`) — fast, cost-effective for interactive chat
- $3/M input, $15/M output tokens
- 4096 max_tokens per response
- Streaming enabled by default

## SSE Event Types
- `text_delta` — incremental text from Claude
- `tool_use` — Claude is calling a tool (name + input shown as chip)
- `done` — response complete
- `error` — error message

## Potential Enhancements
- Add more tools: forecasts, one-pager data, waterfall setup editing
- Conversation persistence (store chat history per user)
- Memory/context from CLAUDE.md and memory files
- Model selector (Sonnet for quick queries, Opus for complex analysis)
- File/chart generation capabilities
- Cost tracking per user
