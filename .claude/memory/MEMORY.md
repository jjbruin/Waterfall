# Waterfall XIRR Project Memory

## Topic Files
- [azure_deployment.md](azure_deployment.md) — Azure infrastructure (VNet, NAT Gateway, VPN Gateway), deployment workflow
- [review_workflow.md](review_workflow.md) — One Pager review/approval pipeline (tables, API, Vue, roles)
- [dashboard_perf.md](dashboard_perf.md) — Dashboard loading optimization (prepare_cap_lookups, 3.7x speedup)
- [isbs_split_migration.md](isbs_split_migration.md) — ISBS split into 6 tables by vSource
- [mri_databases.md](mri_databases.md) — MRI database sources, query service, Azure connectivity status
- [portfolio_analysis.md](portfolio_analysis.md) — Portfolio Analysis tab (upstream entity analysis, actual/proposed modes)
- [transfer_aware_returns.md](transfer_aware_returns.md) — Design doc: transfer-aware IRR/ROE/MOIC (scoped, not yet implemented)
- [onepager_audit_q1_2026.md](onepager_audit_q1_2026.md) — Q1 2026 audit: Azure vs Excel One Pager (959 discrepancies, 7 code bugs, data gaps)
- [session_handoff_may7b.md](session_handoff_may7b.md) — Session handoff: pref accrual fix, combined table, TGA23 step deletion
- [ai_assistant.md](ai_assistant.md) — Embedded AI assistant (Claude API, tools, streaming chat)
- [ai_assistant_roadmap.md](ai_assistant_roadmap.md) — AI assistant enhancement plan (15 items, prioritized)
- [vpn_tunnel_handoff.md](vpn_tunnel_handoff.md) — VPN tunnel to MRI: P1 DH21 negotiation issue, call scheduled Jun 23

## Project Overview
- Flask + Vue application for real estate investment waterfall calculations
- **Deployed to Azure Container Apps** (Apr 2026) — VNet-integrated (May 2026)
- PostgreSQL on Azure (psql-waterfall-dev), local dev still supports SQLite via DATABASE_URL toggle
- Data sourced from MRI (direct SQL via VPN locally, CSV upload as fallback)
- 11+ Vue routes, 13 Flask API blueprints (~115 routes)

## Architecture
- **flask_app/**: App factory, JWT auth, 12 API blueprints, 11 service modules, serializers
- **vue_app/**: Vue 3 + Vite + Pinia + Vue Router + ECharts + AG Grid
- **Core Python**: compute.py, waterfall.py, models.py, metrics.py, loaders.py, database.py, reporting.py, one_pager.py
- Run Flask: `python -m flask_app.run` (port 5000); Vue: `cd vue_app && npm run dev` (port 5173, proxies /api to Flask)
- **Azure admin**: admin / Qu@kers_12
- **Local admin**: admin / admin
- **App URL**: https://app-waterfall-dev-v2.icyplant-026fb2db.eastus.azurecontainerapps.io
- **Deploy**: `az acr build ... --no-logs .` then `az containerapp update ... -n app-waterfall-dev-v2 --revision-suffix vNN`

## Key Architecture Patterns
- `get_cached_deal_result()` — single entry for all deal computation consumers (compute_service.py)
- `build_partner_results()` — single source of truth for partner/deal metrics (compute.py)
- `prepare_cap_lookups()` — batch pre-computation for dashboard capitalization loop (3.7x faster)
- `get_cached_caps_and_occ()` — shared caps/occ cache in `dashboard_service.py`, used by both Dashboard and Surveillance for identical KPIs (debt, occupancy). Eliminates redundant computation and double-counting of child property debt.
- `run_interleaved_waterfalls()` — merges CF/Cap timelines chronologically with shared InvestorState
- `PROTECTED_TABLES` = waterfalls, one_pager_comments, waterfall_audit, review_roles, review_submissions, review_notes, prospective_loans, prospective_loans_audit, planned_loans, sale_overrides, user_requests, user_request_messages, surveillance_comments

## MRI Data Refresh (May 2026)
- **MRI Query Service**: `mri_service.py` + 7 API endpoints + Vue sidebar UI
- **15 queries** committed in `queries/` folder (copied from SharePoint). `_get_queries_folder()` checks `QUERIES_DIR` env var first (SharePoint/OneDrive), falls back to repo `queries/` folder. Dockerfile copies `queries/` into container.
- **Works from Azure** via S2S VPN tunnel (Jun 24, 2026) — both PMX (.9) and IM (.10) fully operational
- **Server IPs (tunnel)**: PMX=10.219.226.9, IM=10.219.226.10 (old FortiClient IPs .17/.18 no longer used)
- **Credentials**: UID=PSCVPN, PWD=NVc8MkB^PlRuv*
- **COA query** (Jun 2026): `COA` table on IM is permission-denied; uses `vCOA` view instead. Server=im. View columns differ from table (`vaccount` not `vcode`). Query: `select vaccount as vcode, vAccountType from vCOA where ISNUMERIC(vaccount)=1`. 176 rows.
- **UI**: "Refresh All Data from MRI" button in sidebar (admin only) refreshes all tables via VPN
- See [mri_databases.md](mri_databases.md) and [vpn_tunnel_handoff.md](vpn_tunnel_handoff.md) for full details

## Azure Infrastructure (May 2026)
- **VNet**: vnet-waterfall-dev (10.0.0.0/16) with NAT Gateway (static IP 20.127.96.240)
- **VPN Gateway**: vpngw-waterfall-dev (VpnGw1AZ, IP 48.194.101.189) — S2S tunnel to MRI FULLY OPERATIONAL
- **Old app** (app-waterfall-dev + cae-waterfall-dev) deleted Jul 1, 2026
- **PG credentials**: `wfadmin` / `Wf3d9097e0365c445456dcc52e!` on `waterfall_xirr` database
- **PG firewall**: Must add current public IP (`az postgres flexible-server firewall-rule create`). IPs added: local-dev (50.251.58.254), local-dev-2 (73.112.240.56), local-dev-3 (71.59.67.132), local-dev-4 (73.112.240.56)
- **Auth login endpoint**: `/auth/login` (not `/api/auth/login`), returns `token` key (not `access_token`)
- **Current revision**: v150 (deployed Jul 29, 2026) — PE cap comment box, Property Perf column swap, PG comments fix
- **Shared folders**: `DATA_DIR`, `QUERIES_DIR`, `DOWNLOADS_DIR` env vars (per-developer OneDrive paths)
- **Shared memory**: `.claude/memory/` in repo (committed, shared via git). Auto-memory redirects here.
- **Email**: SendGrid Web API v3 (replaces SMTP, blocked by O365 MFA). Env vars: `SENDGRID_API_KEY`, `SENDGRID_FROM`. Single Sender Verification on `jbruin@peaceablestreet.com`.
- See [azure_deployment.md](azure_deployment.md) for full resource list and commands

## Actuals Through Cutoff (Mar 2026)
- Global `actuals_through` setting: date or None (default None = full forecast)
- Partner cash flows: actuals from accounting through cutoff, waterfall only post-cutoff
- Dynamic defaults: `DEFAULT_START_YEAR = date.today().year`, `PRO_YR_BASE_DEFAULT = date.today().year - 1`

## Prospective Loans (Apr 2026)
- Supplemental vs Refinance radio selector + checkbox list of existing loans
- Multi-loan replacement: comma-separated IDs in `existing_loan_id`
- PostgreSQL: `ensure_pg_tables()` creates SERIAL id + proper column types at startup

## Sold Portfolio Net Returns (Apr 2026)
- Assumptions panel: Ownership%, AM Fee%, Hurdle Rate%, Promote%, Annual Expenses
- **Capital event identification**: By Typename ("Return of Capital" or "Realized Gain"), NOT the `Capital` flag
- **Hybrid promote**: Capital events use xnpv hurdle; CF distributions use pref as hurdle
- **Per-deal expense overrides** (Jun 2026): `expense_overrides` dict in assumptions (`{vcode: multiplier}`). Airport Plaza (P0000083) defaults to 50% — shared expenses with KOC portfolio. Collapsible UI in Vue below assumptions row.
- **Excel**: Per-deal sheets fully formula-driven (18 cols A-R), Portfolio Detail (19 cols A-S)
- **Excel _xlfn prefixes** (May 2026): openpyxl writes formulas verbatim but xlsx format requires `_xlfn.` for post-2003 functions (XIRR, XNPV, IFERROR) and `_xlfn._xlws.` for dynamic array functions (FILTER). Without prefixes, Excel removes all formulas on file open ("Removed Records: Formula"). Fixed in `sold_service.py`.
- **Net IRR formula**: `=_xlfn.IFERROR(_xlfn.XIRR(_xlfn._xlws.FILTER(...),_xlfn._xlws.FILTER(...)),"N/A")` — requires Excel 365/2021+
- **Sale_Status**: Comes from `deals` table (sourced from `investment_map.csv`), NOT from any MRI query. Must be manually set to "SOLD" for deals to appear in Sold Portfolio tab.

## AMFee Exclusions (Apr 2026)
- **vNotes syntax**: `SOURCE_PC;exclude:IID1,IID2` — excludes capital in specific investments from fee base
- `parse_amfee_vnotes()`, `build_amfee_exclusions()`, `get_amfee_excluded_capital()` in `waterfall.py`

## Sale Overrides (Jul 2026)
- **UI**: Contract Sale Price, Selling Costs (%/$), Sale Date inputs on Deal Analysis page
- **DB table**: `sale_overrides` (vcode PK, contract_sale_price, selling_cost_value, selling_cost_type, sale_date_override, updated_at, updated_by)
- **Workflow**: Enter values → Recompute (force=True bypasses cache) → Save (persists to DB) → Clear (deletes from DB)
- **Sale date override**: Threads through `compute_service.py` → `compute.py`, overrides `Sale_Date` from investment_map
- **ISBS-anchored loan payoff**: Scales modeled amortization balance by ratio of actual ISBS BS debt to modeled debt at anchor date. Prevents "few hundred thousand off" errors from origination-based amortization.
- **Cache**: Sale overrides don't create separate cache keys — `force=True` is set automatically when overrides present
- **PostgreSQL migration**: `sale_date_override` column added via SAVEPOINT-based ALTER TABLE (PG aborts transaction on failed SELECT)
- **P0000073 waterfall**: Copied from P0000049 (same deal structure). Must include `vmisc` column (CF_WF/Cap_WF) or waterfall won't run.

## Paid-Off Loan Exclusion (Jul 2026)
- **Filter**: Loans with `vDateType = "Paid Off"` excluded from all analysis at the data layer
- **Implementation**: `data_service.py:load_all()` and `refresh_table()` filter `mri_loans_raw` after loading
- **ISBS debt fallback**: When modeled loans aren't active at sale date (e.g. paid-off excluded, loan originates after sale), `compute.py` falls back to ISBS balance sheet debt for loan payoff
- **MRI refresh Sale_Date fix**: `Prop_Info_Core.sql` renamed `uw_exit` alias from `Sale_Date` to `Anticipated_Exit`; `MRI_COLUMNS` in `mri_service.py` updated accordingly. MRI refresh no longer overwrites `Sale_Date` (which is NOT available in MRI — preserved during upsert).
- **APP_URL fix**: Default `APP_URL` in `flask_app/config.py` updated from old deleted app to `app-waterfall-dev-v2`. Also set as Azure container env var.

## One Pager Enhancements (Jul 2026)
- **Loan Terms format**: `nRate% | Fixed | M/D/YYYY (+ext)` or `vIndex + vSpread% | M/D/YYYY (+ext)`. Uses `vIndex`, `vSpread` for variable, `nRate` for fixed. Primary = largest `mOrigLoanAmt`, 2nd = next largest. Loan terms populate regardless of debt source (ISBS or MRI_Loans). Extension options from `ExtensionOptions` column (exists on PG, not local SQLite).
- **PE Capitalization investor names**: Resolved via `underlying_investors` column in `one_pager_comments` table (human-readable, e.g. "PSC 69%, Declaration 31%"). Fallback: PPI→upstream entity resolution via `relationships` table. Fallback 2: raw PPI entity ID.
- **one_pager_comments table**: Added `underlying_investors TEXT` column (Jul 2026). 139 rows imported from Charlene's compiled spreadsheet (71 deals x Q4 2025 + Q1 2026). Protected table — not overwritten by CSV import.
- **PE Performance enrichment**: `_enrich_pe_from_deal_result()` in `financials_service.py` runs `get_cached_deal_result()` to pull:
  - **Accrued Balance**: from `seed_states` (pref_unpaid_compounded + pref_accrued_current_year) — represents current state at actuals boundary
  - **Current PE Balance**: from `seed_states` capital_outstanding — not terminal waterfall state
  - **Committed PE fallback**: total PE contributions from partner_results if commitments table empty
- **ROE to Date**: Computed in `get_pe_performance()` (`one_pager.py`) from actual accounting distributions through quarter end, using `calculate_roe()`. Does NOT include projected waterfall returns.
- **U/W ROE to Date**: Computed from ISBS Projected IS account 7071 (underwritten PE distributions). Shows 0% when acct 7071 data is missing.

## Forecast Assembly (Jul 2026)
- **Multi-source priority**: forecast_feed CSV > ISBS Valuation IS > ISBS Projected IS
- **`_assemble_forecasts()`** in `data_service.py` — per-deal priority; deals in forecast_feed override ISBS
- **Valuation IS**: Periodic monthly (direct use). Annual valuation exercise, approved by end of Feb.
- **Projected IS**: YTD cumulative → converted to periodic monthly (subtract prior same-year value)
- **Pro_Yr**: Derived as `date.year - pro_yr_base` for ISBS-sourced rows
- **vcode case**: ISBS lowercases; `_restore_case()` converts `p0000008` → `P0000008`
- **P0000008 (5-15 Broad St)**: Had no forecast on Azure because forecast_feed only had Val_IS_2025 (not Val_IS_2024). Now auto-filled from ISBS Valuation IS.
- **Occupancy >100% fix**: `.clip(upper=100)` on all `occ_val` reads in `dashboard_service.py`. Root cause: Prestige Storage (P0000080) had bad iResidentialUnits=364 in MRI (2023-2024), corrected to 3145 in 2025.

## Key Technical Notes
- `waterfall.db` exceeds GitHub 100MB — not committed, lives locally
- InvestmentID mapping: `build_investmentid_to_vcode()` returns `{InvestmentID: vcode}`
- **Acquisition Date**: Derived from earliest accounting activity per deal, enriched in `data_service.py:load_all()`
- **XIRR**: Filters zero-value cashflows before solving (metrics.py). Newton-Raphson primary, Brent fallback.
- **Sale date default**: Falls back to max loan maturity when no explicit sale date (compute.py)
- **Sub-portfolio**: Graceful fallback when child forecasts empty (compute.py)
- **PyJWT 2.11**: `sub` claim must be `str`, decode back with `int(payload["sub"])`
- **Git Bash path mangling**: Use `MSYS_NO_PATHCONV=1` prefix for Azure CLI commands with resource IDs

## User Preferences
- Reports: Both PDF and Excel
- Workflow: Mix of deal deep-dive, comparison, and reporting
- Approach: Incremental (extract one section at a time)
- Data access: Light editing (comments, assumptions, overrides); core data from MRI

## Ownership Chain Traversal (Jul 2026)
- **EndDate filtering**: All ownership traversals filter ended relationships (`EndDate` not empty). Must handle NaT/nan/None string representations from datetime-parsed columns.
- **Reports "By Partner"**: Uses `get_upstream_investor_deals()` with OP%/PPI% exclusion (replaced old waterfall PropCode-based list). Removed redundant "By Upstream Investor" option.
- **Hybrid deal discovery** (PSCKOC + Portfolio Analysis): Recursive downward traversal finds all intermediate entities, then filters to deals whose waterfall PropCode references one of those entities OR the entity itself. `match_entities = downstream | {entity_id}`. Including the entity itself catches deals that list it directly as PropCode (e.g., Life Storage lists TGA22). Prevents shared-entity false positives (e.g., TGA22 pulling in Pegasus via shared KOCTRS).
- **PE balance filtering**: `_build_deal_returns()` accepts `match_entities` and filters pref equity partners to only those in the entity's ownership chain. Without this, deals with multiple PE partners (e.g., TGA22 + PPILFS + OPPEGA) showed the full deal PE balance instead of the entity's share.
- **`find_entity_deals()` return type**: Returns `(deal_list, match_entities)` tuple — callers must unpack.
- **Review Tracking investor list**: SQL CTE in `get_investor_list()` — the authoritative upstream investor filter (excludes OP%, PPI%, sold deals, child properties).

## Portfolio Analysis Tab (May 2026)
- **Status**: Committed and deployed. Allocation table has enriched step labels, combined CF/Cap layout, pref accrual fix
- **Pref fix**: `_build_entity_seeded_states()` seeds InvestorStates with capital from accounting; `accrue_all_pools()` added to upstream waterfall period processing
- **TGA23 CF step 3 deleted** from both local SQLite and Azure PostgreSQL (modeling error, done May 7)
- See [portfolio_analysis.md](portfolio_analysis.md) for full details

## AI Assistant (Jun 2026)
- **Embedded chat panel**: Floating "AI" button (bottom-right), streaming SSE responses
- **Model**: Claude Sonnet 4.6 via Anthropic API (cost-effective for interactive use)
- **20 tools**: resolve_deal, list_deals, query_deal_data, query_accounting, query_database, get_portfolio_summary, compute_deal_returns, get_loan_details, get_occupancy, get_financial_statement, get_waterfall_structure, get_one_pager, get_annual_forecast, get_sold_returns, get_capitalization, compare_deals, get_debt_service, get_cash_management, get_tenant_roster, get_user_feedback
- **Chat persistence**: `chat_history` table (user_id PK, messages JSON), auto-save after each exchange, load on open
- **Backend**: `flask_app/services/assistant_service.py` (tools + agentic loop), `flask_app/api/assistant.py` (SSE endpoint)
- **Frontend**: `vue_app/src/components/common/AiAssistant.vue` (chat panel in App.vue)
- **Page context**: Sends current page, selected deal vcode/name, and quarter to backend with each message. System prompt includes context so assistant infers deal from user's current view.
- **Activation**: `ANTHROPIC_API_KEY` env var (`.env` for local dev, container env var for Azure)
- **API key stored**: `.env` file (gitignored), Azure container env var set on v33
- **Local .env loading**: `flask_app/run.py` uses python-dotenv (optional import, graceful skip)
- See [ai_assistant.md](ai_assistant.md) for full details

## Feedback & Request Tracking (Jul 2026)
- **Sidebar section**: "Feedback & Requests" between MRI Data and Database Tools
- **Request types**: error, improvement, report, analysis. Priorities: low, medium, high
- **Statuses**: open, in_progress, resolved, closed
- **DB tables**: `user_requests` (id, user_id, username, type, title, description, priority, status, page_context, deal_context, reply_token), `user_request_messages` (id, request_id, sender_type, sender_name, message, sent_via)
- **API**: `flask_app/api/feedback.py` — blueprint at `/api/feedback`. CRUD + admin email + inbound webhook
- **Service**: `flask_app/services/feedback_service.py` — CRUD, SendGrid email with reply links, export for design sessions
- **Email flow**: Admin sends via `POST /<id>/email` (SendGrid). Email has "View & Reply" link with unique reply_token → auto-opens sidebar feedback panel. SendGrid Inbound Parse webhook at `POST /inbound-email` for email replies (requires DNS MX setup).
- **AI assistant tool**: `get_user_feedback` — returns all requests with threads, filterable by status/type. For use during Claude design sessions.
- **Admin endpoints**: `PUT /<id>/status` (change status), `POST /<id>/email` (email user), `GET /export` (all requests for design sessions)

## Property Surveillance Tab (Jul 2026)
- **Backend**: `flask_app/services/surveillance_service.py` + `flask_app/api/surveillance.py`
- **Frontend**: `vue_app/src/views/SurveillanceView.vue` + `vue_app/src/stores/surveillance.ts`
- **Live metrics** (consistent with other tabs): TTM NOI (Property Financials formula), DSCR (NOI/abs(debt service)), debt balance (ISBS via `get_isbs_debt_balance()`), loan maturity (`build_loans_from_mri_loans()`)
- **KPI strip**: Debt and occupancy sourced from shared `get_cached_caps_and_occ()` — identical to Dashboard
- **Comments**: `surveillance_comments` table (vcode, comment_date, comment_text, created_by). Date-based, upsert on same date. Newest-first display with collapsible history.
- **7 expandable column groups**: Reporting, Debt Covenants, Real Estate Taxes, Insurance, Ground Leases, Escrows, Add'l Collateral
- **Debt Covenants**: DSCR/DY/LTV actuals (computed from ISBS/valuations) vs general and extension requirements (from MRI Loans: nRequiredDCR, nReqDSR, nDY, nRequiredDY, nLTV, nRequiredLTV). Most restrictive aggregation for multi-loan deals. Breach highlighting (red/green). Extension options text (vAmortAmt).
- **Real Estate Taxes**: TTM tax amount from ISBS account 5090 (extracted during NOI computation — no redundant query). Editable tax_due date and tax_status (Current/Paid/Pending/Delinquent/Appealed).
- **Insurance**: Carrier name + expiration date per policy type (Property, GL) with urgency highlighting (<30d orange, expired red). TTM insurance expense from ISBS accounts 5110/5114. Insurance CRUD table for policy details.
- **Ground Leases/Escrows/Collateral**: Editable fields in surveillance_properties table. Ground leases: expiration + maturity urgency, annual rent, status. Escrows: Yes/No/Waived/Partial badges for tax/insurance/capex. Collateral: type, value, notes.
- **Reporting completeness**: Latest reported period (M/YY) and missing count in trailing 12 months for Occupancy, Rent Roll (commercial only), Income Statement, Balance Sheet. Due date = month end + 30 days.
- **DB tables**: `surveillance_properties` (editable fields with column migration), `insurance`, `surveillance_comments` (all in PROTECTED_TABLES)

## Agreed Roadmap
- ~~Purchase 2nd MRI VPN license → configure Azure VPN Gateway tunnel → enable MRI queries from Azure~~ DONE (Jun 24, 2026)
- Phase 5: Additional report types (PDF generation, Debt Service Summary, Cash Flow Detail)
- Enhance AI assistant: add more tools (forecasts, one-pager data, waterfall setup), memory/context persistence

## Known Warnings
- ~~`utils.py:38` FutureWarning: `'M'` → `'ME'` for pandas date_range freq~~ FIXED (Jul 3, 2026)
- Python 3.14; specify encoding=utf-8 for file reads
- Vite build: ECharts split into own chunk via manualChunks; chunkSizeWarningLimit=650 (ECharts is ~620KB, can't be split further)
- `vue-tsc --noEmit` passes with zero errors (all 16 pre-existing TS errors fixed Jul 3, 2026)
