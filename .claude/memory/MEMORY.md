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
- [new_business_pipeline.md](new_business_pipeline.md) — New Business pipeline: prospect deals, properties, entities, lease review integration

## Project Overview
- Flask + Vue application for real estate investment waterfall calculations
- **Deployed to Azure Container Apps** (Apr 2026) — VNet-integrated (May 2026)
- PostgreSQL on Azure (psql-waterfall-dev), local dev still supports SQLite via DATABASE_URL toggle
- Data sourced from MRI (direct SQL via VPN locally, CSV upload as fallback)
- 13+ Vue routes, 13 Flask API blueprints (~120 routes)

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
- `PROTECTED_TABLES` = waterfalls, one_pager_comments, waterfall_audit, review_roles, review_submissions, review_notes, prospective_loans, prospective_loans_audit, planned_loans, sale_overrides, user_requests, user_request_messages, surveillance_comments, lease_reviews, lease_tenants, lease_documents, lease_rent_steps, lease_cotenancy, lease_cotenancy_refs, lease_exclusive_use, lease_options, lease_validation, prospect_deals, prospect_properties, prospect_entities, prospect_investors, prospect_assumptions, prospect_cashflows, prospect_activity

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
- **Current revision**: v235 (deployed Aug 12, 2026) — New Business Pipeline (Kanban + table + deal workspace), Lease Review Vue frontend, prospect property/entity data model
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
- **PE Capitalization**: Editable borderless comment box (`pe_cap_comment` column in `one_pager_comments`). Carries forward across quarters via vcode-level fallback. Replaced computed investor % formula (Jul 29, 2026). 178 rows imported from One_Pager_Comments.xlsx (89 deals x Q4 2025 + Q1 2026).
- **one_pager_comments table**: Columns: econ_comments, business_plan_comments, accrued_pref_comment, underlying_investors (legacy), pe_cap_comment. Protected table — not overwritten by CSV import. **PostgreSQL fix**: `get_one_pager_comments` uses `execute_query()` (handles `?`→`%s` conversion); `save_one_pager_comments` uses SQLAlchemy `text()` with named params on PG.
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

## NOI Account Mapping History (Aug 2026)
Three consecutive changes to which accounts sit in NOI. Read together — no single commit tells the whole story.
- **d152629** (Aug 4) added six accounts to NOI: 4075 (Other Revenue) to `REVENUE_ACCTS`; 5092 (Maintenance Flex), 5120 (AM Fee/Partnership), 5130 (Other Partnership Costs), 5160 (Depreciation), 5165 (Amortization) to `EXPENSE_ACCTS`. 5120/5130 were moved OUT of `OTHER_EXCLUDED_ACCTS` and the `Partnership Expenses` below-the-line row was deleted. 4075/5092/5160/5165 had previously appeared in NO list at all (silently dropped).
- **6af039f** (Aug 4, 8 minutes later) reverted depreciation: 5160/5165 out of `EXPENSE_ACCTS`, into `OTHER_EXCLUDED_ACCTS`, with a new `Depreciation & Amortization` below-the-line row. Not a true revert — pre-d152629 they were in no list, so they are now in `ALL_EXCLUDED` for the first time, which affects the forecast/FAD path in `loaders.py`/`reporting.py`, not just display.
- **This change** (Aug 5) reverts 5120 and 5130 out of NOI, restoring the `Partnership Expenses` below-the-line row. **5092 and 4075 remain in NOI by Charlene's decision.** Portfolio analysis flagged 5092 as also diverging from the investor model, but it was kept intentionally pending model confirmation — revisit.
- **Current state**: in NOI → 4075, 5092. Below-the-line → 5120, 5130, 5160, 5165.
- **Blast radius method**: only ISBS `Interim IS` (NOI ACT) and `Projected IS` (NOI U/W) feed the One Pager / Property Financials chart, so account changes only move the chart for deals with activity in those two sources. A deal can have activity in Budget IS / Valuation IS / forecast_feed and show no chart change — 8 of 61 such deals did. Per-account attribution is exact because the cumulative→periodic→quarterly pipeline is linear.
- **Guardrail pattern** (for future mapping changes, not committed — local scripts embed per-developer OneDrive paths): read the before-mapping from `git show HEAD:config.py` and the after-mapping from the working-tree `config.py`, then assert (a) the account set that left/joined NOI is exactly what was intended, (b) every moved deal-quarter belongs to a deal with activity in those accounts, (c) each move equals that account's own contribution. A typo then fails a check instead of being assumed away.
- **Deploy reality**: pushing to origin/main does NOT deploy (no GitHub Actions secrets). Live app keeps serving the prior image until someone runs `az acr build` + `az containerapp update`.

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

## One Pager Audit — Investigations 1 & 2 (Aug 5, 2026) — WORK HELD

Read-only diagnostics from `OnePager_WorkQueue_and_Report.docx`. **No code changed.**
Scripts: `scripts/inv1*.py`, `scripts/inv2*.py` (uncommitted — they embed per-developer
OneDrive paths). Source: local `accounting_feed.csv` snapshot **5/4/2026** + ISBS
Projected IS acct 7071. Azure may be more current — reconfirm before fixing.

### The root cause: `abs()` on opposite-signed rows

There is **no reversal mechanism anywhere in the codebase** (repo-wide grep for
`revers|adjust|void|cancel`: zero hits) and **no Typename marks a reversal**. A reversal
exists only as an opposite-signed row with the same Typename. Every consumer forces
magnitude, so reversals are **added instead of subtracted** — a 2x error.
- Contributions are stored **negative** (2,758 of 2,818 rows); 59 positive = $39,994,127.
- `one_pager.py:507` `+= abs(amt)`; `waterfall.py:1190,1197` `-abs(cf)` then `+= abs(cf)`.
- Portfolio-wide: abs() overstates contributions by $79,988,255 (exactly 2x the reversals).

### Investigation 1 — Total Capitalization

- **Reversal bug confirmed. Affects Partner Equity on 5 deals** (all reversing InvestorIDs
  start with `OP`; Pref Equity untouched on all 72): JB Fair Park 11,550,000 vs 3,850,000
  netted (+7,700,000) · Cocoplum 29,978,294 vs 23,350,000 (+6,628,294) · Belleville
  3,652,245 vs 1,702,415 (+1,949,830) · Adirondack RV 1,962,500 vs 762,500 (+1,200,000) ·
  Pegasus Life Storage 3,476,390 vs 2,573,473 (+902,916).
  **This explains 4 of the 6 Partner Equity discrepancy deals.** Brainerd Place and
  OREI Portfolio show zero delta — different cause, still unknown.
- **Only 12 of 59 positive rows pair exactly** with an offsetting negative (same
  deal+investor+typename+amount+date); 24 pair ignoring date. **The other 35 are
  fund-level InvestmentIDs (PSC3, PSCPGH, PSCKOC, INVF1/2/5) that map to no vcode and
  never reach a One Pager — OUT OF SCOPE.** Do not assume "positive contribution =
  reversal" globally; it is proven only for the 5 deals above.
- **Date attribution is BY DESIGN — no bug.** The feed has exactly one date column
  (`EffectiveDate`) plus a `Qtr` label, and `Qtr` **never** disagrees with the quarter
  derived from `EffectiveDate` (0 mismatches / 11,365 rows). The "entered in October,
  attributed to Q3" case does not exist. There is no second date to filter on.
- **Separate real finding: `get_capitalization_stack()` applies NO date filter to the
  equity block** — `quarter_str` feeds only the debt line (`one_pager.py:261-264`), while
  `get_pe_performance()` does filter `EffectiveDate <= quarter_end` (`:1194-1197`). The two
  One Pager sections are built on different populations. Five deals differ as of Q1 2026:
  Woodlands Square (9,700,000 → 0), **Trolley Square (5,932,489 → 4,190,216, on the Pref
  Equity list)**, Jefferson Stephens, Airport Plaza, Quakertown. Airport Plaza and
  Quakertown move the *wrong* way — the per-investor `max(0, balance)` floor at
  `:511-515` makes the result non-monotonic in the cutoff date.
- **Excel-serial dates = DATA ISSUE, owned by Jim.** 101 rows carry `43402`, `43448`, …
  in `EffectiveDate` instead of a date; `to_datetime(errors='coerce')` → NaT. They are
  *included* in Total Cap (no filter) and *dropped* from PE Performance. $20.18M: one
  $9.7M contribution (Woodlands Square's entire pref equity) + 91 Preferred Return
  distributions ($9.44M). Fix in the source data, not in code.

### Investigation 2 — ROE / U-W ROE (Q1 2026)

`ROE = CF distributions / wtd-avg capital / years`, i.e. **CF x 365 / dollar-days**.

- **Same `abs()` bug in the ROE numerator** (`one_pager.py:1233,1238`). Negative CF rows
  are abs()'d and added. **4 deals**: **Apple Self Storage (P0000003) 36 rows /
  $7,481,447 inflation, ROE 22.40% — by far the largest, and NOT on the audit list** ·
  Pontchartrain $252,400 (its cause) · Prestige Storage $226,400 · 30 Bearfoot $87,347.
- **30 Bearfoot (21.32%) decomposed — not the acquisition fee.** Three stacked behaviours:
  net the 4 reversal rows −0.99pp; keep Acq Fee out of the capital balance −0.44pp;
  drop CF received after payoff −2.75pp (~4.2pp total, residual unexplained without Excel).
- **Acquisition Fee shrinks the ROE denominator on 70 of 71 deals** ($8.96M). It is
  excluded from the numerator but still appended to `capital_events` at `:1233` *before*
  the typename check, so it acts as a return of capital.
- **CF received after capital hit zero — 4 deals.** Numerator with no denominator:
  **Berger Pittsburgh $2,993,146** (paid off 2024-07-10), 30 Bearfoot $160,612,
  Willowdale $38,318, Barnbeck $27,000.
- **Late distributions: real but small.** 5 of 8 deals have exactly one CF distribution
  dated 3–30 days after quarter end (the Q1 pref paid in April), dropped from Q1 ROE.
  Impact +0.03 to +0.67pp. The feed's own `Qtr` label puts them in Q2 — if Excel counts
  them as Q1 that is a definitional difference, not a data error.
- **U/W ROE: `abs()` fabricates distributions** in `_get_uw_pe_distributions()`. Sign
  flips in ISBS acct 7071 become fake income. **Court at Deptford**: Feb-2025 cumulative
  spikes to +108,170 among otherwise-negative periods → periodic +190,107, then Mar-2025
  −353,981; both abs()'d = **$544,087 fabricated, 16% of its $3.33M U/W total**.
  **Middle Island** oscillates sign repeatedly — most of its $578,891 is noise, against
  an actual ROE of 0.00%.
- **Pro-rate branch drops the first partial year's YTD** on 6 of 8 deals
  (`periodic = cumulative / month` keeps one month): Gallery 110,558 · Mount Prospect
  60,742 · Gathering 42,931 · Bearfoot 21,232 · Middle Island 9,552 · Pontchartrain 8,119.
- ~~**The Gathering is stale underwriting, not a bug**~~ — **WRONG, corrected Aug 6, 2026.**
  Its 7071 schedule flatlines at −31,551.35/month from Jan-2024 because that is the *correct*
  pref on the underwritten **post-disposition** balance of $3,559,057 ($378,616/yr = 10.6%,
  i.e. the coupon plus participation). The schedule is internally consistent. Real root cause:
  see **U/W ROE root cause: the 707x family** below.

### U/W ROE root cause — the 707x family (Aug 6, 2026)

`_get_uw_pe_distributions()` reads **only** `vAccount == '7071'` (`UW_PE_DIST_ACCT`,
`one_pager.py:1009`). The sibling accounts are identified by the **`vInput` column** in ISBS
(the `coa` table is useless here — it has only `vcode` + `vAccountType`, and types every 70xx
as `Expenses`):

| acct | `vInput` label | deals | read by U/W ROE? |
|------|----------------|-------|------------------|
| 7071 | PEACEABLE CASH FLOW | 50 | yes — the numerator |
| 7072 | SPONSOR CASH FLOW | 41 | no (correctly — sponsor side) |
| 7073 | **PEACEABLE DISPOSITION PROCEEDS** | 54 | **no — this is the bug** |
| 7074 | SPONSOR DISPOSITION PROCEEDS | 52 | no (correctly) |
| 7075 | RESERVES FOR REPLACEMENT | 75 | no (correctly) |

**The defect:** 7073 is PSC's underwritten capital return. Excluding it from the *numerator* is
right (ROE is operating-yield only), but the **denominator never applies it either** — the
denominator is built 100% from the `accounting` table (`one_pager.py:1180-1238`). So U/W ROE
divides a pref stream computed on a *reduced* underwritten balance by the *full actual* balance.
Affects the **54 deals** carrying 7073.

**The Gathering (P0000041) worked example:** 7073 = $5,768,443.30, one row, `dtEntry` 2022-11-30.
Verified to the cent against the source U/W workbook — `USE THIS The Gathering at University
Village Underwriting (3) v03.xlsm`, sheet **`PSC Investor Equity Structure`**, cell **G16**
("Capital Transactions", formula `=+PSC!E78-G15`); Total Equity row 7 steps $9,327,500 →
$4,206,846 at that event. Actual return of capital was only $983,287.41 on 2024-11-12.
As shipped U/W ROE = 4.76% (wtd cap 8,329,059); applying the underwritten return in place of the
actual one → **9.16%** (wtd cap 4,327,986), which matches row 18 of that same U/W sheet reading
**9.0%** in steady state. That agreement is the tell that this reading is correct.

**Trap for whoever fixes it:** 7073 is lumpy, not monthly. Routed through the existing
`one_pager.py:1092` pro-rate branch its 2022-11-30 cumulative would be divided by month 11,
keeping $524,404 and silently discarding $5,244,039. Disposition accounts need different
cumulative→periodic handling than monthly cash-flow accounts.

Audit workbook (formula-driven, Excel-verified): `~\Downloads\The_Gathering_UW_ROE_Audit_v2.xlsx`.
Builder script lives in the session scratchpad, pulls live from Azure PG — not committed
(embeds credentials + per-developer paths).
- **Ruled out:** 55 comma-formatted `Amt` values (Brainerd Place, 5-15 Broad St,
  Woodlands Square) look like they would parse to $0, but `data_service.load_all()`
  normalizes once via `loaders.normalize_accounting_feed()`, which strips commas/$/parens
  at `loaders.py:253`. Not a live bug — don't chase it.

## One Pager Audit — Investigations 3/4/5 + Prompt B (Aug 6, 2026)

Run against **live Azure PostgreSQL**, not the CSV snapshot — Azure had 12,300 accounting
rows (max EffectiveDate 2026-08-03) vs 11,466 in the 5/4/2026 CSV. Scripts:
`scripts/inv34_equity_composition.py`, `inv34b_basis_variants.py`, `inv5_jbfairpark_debt.py`,
`inv5b_stale_debt_scan.py`, `fixB_verify.py` (uncommitted — they embed the PG password).
Azure PG is reachable from local dev; firewall already allows it.

### Investigations 3+4 — Pref / Partner Equity composition

Equity block = `one_pager.py:496-515`. Rule is deliberately broad: any MajorType containing
`contrib` adds `abs(Amt)`; a `distri` row whose Typename contains `return of capital`
subtracts `abs(Amt)`. Keyed on `deals.InvestmentID` (1:1 to vcode). **No date filter.**
Bucketing is by InvestorID prefix `OP` → Partner, everything else → Pref.
- **Partner Equity: 4 of 6 confirmed as the abs() reversal bug** — JB Fair Park +7,700,000 ·
  Cocoplum +6,628,294 · Belleville +1,949,830 · Pegasus +902,916. Each is exactly 2x a
  single positive-signed `Contribution: Investments` row.
- **Brainerd Place and OREI Portfolio are NOT the abs() bug and NOT the date filter** —
  both are zero-delta on every mechanism tested. Remaining basis candidates (per
  `inv34b`): Brainerd's `Contribution: Others` 4,550,000 and its `Return of Capital`
  6,770,190 (Azure nets it; excluding it gives 18,777,868, excluding Others gives
  7,457,677); OREI's `Contribution: Operating Capital` 2,605,000 pref / 1,233,899 partner
  (excluding it gives pref 10,786,868 / partner 6,890,613). **Needs the model to say which
  basis it is on.** Child properties are NOT the cause — OREI's 2 children and Brainerd's
  9 children carry zero contribution/RoC rows.
- **Pref Equity: the abs() bug does not touch any of the 5 named deals.** Three are pure
  no-date-filter cases where a post-Q1 contribution is counted in the Q1 figure:
  Nottingham Village 2,923,427 (6/1/2026) · Trolley Square 2,544,582 (4/20 + 5/15/2026) ·
  Belleville 125,000 (6/10/2026). All three are **new on Azure vs the released Actual —
  Azure is more current, not wrong.** OREI Portfolio and 5-15 Broad St show no delta from
  any tested mechanism.
- Construction/pre-stab basis flag: Belleville, JB Fair Park, Trolley Square, Brainerd,
  Pegasus are Development / New Construction — the figure is funded-to-date, not a closing
  capitalisation.

### Investigation 5 — JB Fair Park debt: NOT mOrigLoanAmt

The reported "shows full loan amount" is wrong about the mechanism. `cap['debt']` =
**66,363,992**, not the 77,368,000 origination amount. It comes from a **single stale ISBS
Interim BS row on account 2150 dated 2022-12-31**. The deal's own BS data stops at
2025-06-30 (other deals run to 2026-06-30) and account 2150 never appears again.
`get_isbs_debt_balance()` detects the staleness (`compute.py:98-116`) but keeps the last
known balance because an active MRI loan exists (LoanID 335, vDateType='Maturity') — the
`mri_loans` cross-reference added in 024c29f is exactly what prevents the $0.
Interest expense (accts 5190/7030) is **0 across all periods**, consistent with nothing drawn.
**Portfolio-wide scan (`inv5b`): only 3 of 83 deals show a stale debt balance** — JB Fair
Park (30 months), Post Commons (1 month, benign lag), Pegasus (21 months but no active MRI
loan, so already forced to 0). This is a JB-Fair-Park-specific data gap, not a systemic bug.

### Prompt B — display fixes

- **Fix 12 DONE and pushed** (`09ec333`, main). `get_general_information()` appends State to
  City. 108 of 134 deals change; 20 stay blank (no City and no State in `deals`); skips
  append when the state string is already inside the city string (Town Fair Tire Portfolio
  has City=NaN, State='CT, RI' → renders 'CT, RI'). **Pushing does not deploy.**
- **Fix 1 — Town Fair Tire Portfolio (P0000107) was a false alarm**: it has loan 318 and
  renders `'SOFR + 3.50% | 2/14/2032'`. Blank extension is correct (ExtensionOptions NaN).
  **Burton Retail Portfolio (P0000109) is a real display bug — FLAGGED, NOT FIXED**: it has
  **zero** loan rows on its own vcode; all 3 loans (5.665% Fixed) sit on children
  P0000111/112/113. `get_capitalization_stack()` filters `vCode == vcode_str` with no child
  aggregation, so the parent renders 'N/A'. Fixing means aggregating child loans the way
  `consolidation.py` already does for the waterfall.
- **Fix 11 is already satisfied for 4 of the 5 deals** — Brainerd Place, Mount Prospect
  Plaza, Pontchartrain Landing and Poplar Prairie all render `second_loan_terms_str` today
  (the 2nd-largest-loan logic at `one_pager.py:367-371`). Live revision v153 (Jul 29)
  includes the Jul 27 loan-terms commit 972990d, so this is true on Azure too. **Only
  Ascent on Steamboat (P0000065) is blank, and that is arguably correct**: both its
  supplemental loans (LoanID 288, 289) have `vDateType='Paid Off'` and are dropped at the
  data layer by the portfolio-wide paid-off filter (3 of 89 rows). No branch created — the
  requested change would re-do work already shipped.

### Burton child-loan aggregation — DONE, on a branch (Aug 6, 2026)

Branch **`fix/burton-child-loans`** commit `e725134` — ~~not merged, not deployed~~ **SUPERSEDED
Aug 7, 2026: rebased to `2086ebc`, merged to main in `1a5e277`, and CONFIRMED DEPLOYED.** It
shipped with a flaw — see "Burton co-terminous child loans — Aug 7, 2026" at the end of this file.
New helper `_child_vcodes_for_parent()` in `one_pager.py`; the loan-terms block falls back
to child loans only when the deal has none of its own.

**Parent/child topology (from `scripts/burton_relmap.py`)** — 7 Portfolio_Name groups,
38 of 134 deals carry a Portfolio_Name. Two naming conventions coexist:
- usual: child.Portfolio_Name == parent.Investment_Name (Berger, Brainerd, Giant 7, OREI,
  Town Fair, Donald Lynch) — this is what `consolidation.get_property_vcodes_for_deal()`
  already matches.
- **Burton is the exception**: parent Investment_Name 'Burton Retail Portfolio' but the
  group is 'Burton Portfolio', so the existing helper finds nothing. The fix therefore
  matches Portfolio_Name against *either* the parent's Investment_Name or its own
  Portfolio_Name.
- **`Property_Count` is the parent discriminator**: parents have >= 1, every genuine child
  property has 0. Without it, a group-name match would wrongly give 22 child properties
  (7 Giant 7, 9 Brainerd buildings, 6 Town Fair stores) their sibling's loan terms.
  Only P0000109 looks like a child by name but has Property_Count 3 — that is Burton itself.

**Verified blast radius** (`burton_loandump.py` dumps all 134 deals through the production
path, before vs after): **131 byte-for-byte identical, 3 changed**, `debt` unchanged on
every deal (display-only, as intended).
- P0000109 Burton Retail Portfolio: 'N/A' → `5.67% | Fixed | 8/28/2032` (3 child loans)
- P0000007 Berger Pittsburgh Portfolio: 'N/A' → `2.93% | Fixed | 8/1/2030` (+2nd
  `2.95% | Fixed | 9/1/2030`; 8 child loans across 4 children)
- P0000049 Donald Lynch: 'N/A' → `3.95% | Fixed | 7/1/2028` (1 loan on P0000073)

All three are parents that previously rendered N/A while holding loans on children — the
intended population. No standalone deal changed.

### Status — HELD, nothing started

**NOT yet run:** `Debug_Progress.xlsx`.

**Open model-definition questions for Charlene (not bugs — Claude Code cannot see the model):**
- Brainerd Partner Equity: does the model include `Contribution: Others` 4,550,000?
  (excluding → 7,457,677 vs Azure 12,007,677)
- OREI: should `Contribution: Operating Capital` count? (excluding → pref 10,786,868 /
  partner 6,890,613 vs Azure 13,391,868 / 8,124,512)
- Development deals: does the model expect funded-to-date rather than closing capitalisation?

**For Jim:** JB Fair Park BS backfill (data stops 6/30/2025; debt reads a 12/31/2022 row);
refresh the released Actual for Nottingham / Trolley / Belleville recent contributions.

## Session Aug 6, 2026 (evening) — U/W ROE deep-dive + abs() row-level proof

**No code changed, no branch created, no commit authored in this session.** Everything below is
read-only diagnostics against **live Azure PostgreSQL** (`psql-waterfall-dev` / `waterfall_xirr`),
plus one read-only COM read of Charlene's open Excel model. The only file written was this
MEMORY.md (the "stale underwriting" correction), which **Charlene** committed as `72fa80d`.
Commit `6b3d54e` and branch `fix/burton-child-loans` (`e725134`) are from a **different** session —
not this one. Scripts live in the session scratchpad and are **not committed** (they embed the PG
password and per-developer paths).

### Verified against the source U/W workbook (new — this is the authority, not the ISBS feed)

File: `...\Asset Mgmt\MF - Berger\42. The Gathering\02. Budget, CF, UW\USE THIS The Gathering at
University Village Underwriting (3) v03.xlsm`, sheet **`PSC Investor Equity Structure`**.
Read live from Charlene's open Excel session via COM (openpyxl could not open it — Excel holds an
exclusive lock).
- **`G16` = 5,768,443.30**, row label "Capital Transactions", formula `=+PSC!E78-G15`. Ties to
  ISBS acct 7073 to the cent.
- Row 7 `Total Equity` = `+PSC!D62..L62`; row 16 D16/E16 = `PSC!B75`/`PSC!C75`. **The Excel model
  never reads actual contributions anywhere — it is 100% underwritten.**
- Row 2 headers are period ENDs on a 10/1 fiscal year (`=DATE(YEAR($D$2)+G4,...)`), so G16 sits in
  the year 10/1/2022–9/30/2023. ISBS pins the same figure to 11/30/2022.
- Row 22–26 terms: Coupon **8.0%**, Asset Management 0.75%, Promote 12.50%, Annual Costs 9,000.
  Note `deal_terms.pe_coupon` on the DB says **9.0%** — different vintages.

### THE BIG ONE: the Excel ROE is a different metric, not just a different denominator

**`H18 = H15/H7`** — single-period Distributable Cash Flow over that period's equity balance.
**Not time-weighted, not inception-to-date, not annualised.** G18 = 3.6% (yr 1), H18 = I18 = 9.0%.
`metrics.calculate_roe()` is ITD dollar-day weighted and annualised. They only agree on The
Gathering because its underwritten steady state is flat (−31,551.35/mo). **On a deal with a lumpy
7071 schedule the two definitions diverge even with a perfectly correct denominator.**
→ Decide which metric the One Pager is meant to report before touching code.

### Proposed-model testing (numerator held at the 7071 total 1,883,127.20; cutoff 2026-Q2)

| scenario | wtd avg capital | U/W ROE |
|---|---|---|
| as shipped (denominator 100% actual) | 8,329,059.44 | 4.7619% |
| **7073 replaces the actual ROC** | **4,421,261.26** | **8.9707%** |
| + acquisition fee left in | 4,327,986.26 | 9.1641% |
| only the pure equity step (5,120,654) reduces capital | 4,910,187.06 | 8.0775% |
| that + participation (647,789.30) in the numerator | 4,910,187.06 | 10.8561% |
| 7073 dated on the Excel yr-2 bucket 10/1/23 | 5,421,357.74 | 7.3159% |
| fully U/W timeline (U/W contrib dates + 10/1/23) | 5,692,095.04 | 6.9679% |

- **7073 decomposes**: Total Equity steps 9,327,500 → 4,206,846, so **5,120,654 is returned capital
  and 647,789.30 is profit above capital.** Scenario 2 hits 9.0% partly *because* it lets the whole
  5.77M reduce the balance — right answer, arguably wrong reason.
- **Date choice is worth 1.65pp** (ISBS 11/30/2022 vs Excel 10/1/2023).

### 707x identity + signs (extends the table in the earlier 707x section)

`vInput` is the only real label. All PSC-side accounts stored **negative (credit), zero sign
flips**; 7075 stored positive. **There is NO projected-contribution account anywhere in 707x** —
the "projected contributions increase capital" leg of the proposed model has no data behind it.
7072 = SPONSOR CASH FLOW and 7074 = SPONSOR DISPOSITION PROCEEDS are the **sponsor's** (Gencore/
Berger) side, not PSC's; 7075 = RESERVES FOR REPLACEMENT is not an equity flow.
**`7010` = `DEBT SERVICE PROPERTY`** (~1.6M/yr) — ruled out of the numerator.

### Numerator scope confirmed — 7071 only, and it ties to the Excel

Excel numerator = row 15 = row 13 (`+PSC!D27..L27`) + row 14 (0 on every column). Row 16 capital is
**excluded** (ROE = row15/row7), so the 647,789 participation is out of the Excel numerator too.
Reconciling 7071 (with a **corrected** cumulative→periodic conversion) to row 13 on 10/1–9/30 years:
**FY4 and FY5 tie to $0.17**; other years differ only from the calendar-vs-fiscal boundary.
**Adding 7072 destroys the tie** (FY4 833,030 vs 378,616) — independent proof the sponsor account
does not belong.

### GL accounts for PSC capital contributions (new topic)

- **The `accounting` table has NO GL account column.** Contributions are `MajorType='Contribution'`
  + TypeID: **1018 `Contribution: Investments` (2,597 rows)**, 1023 Partnership Expenses (400),
  1024 PSC KOC I LLC Expenses (70), 1020 Management Fees (34), 1019 In Kind (8), 1025 Operating
  Capital (7), 1022 Others (5), 1021 Organizational Costs (2).
  **TypeID 1018 is not unique** — it is also `Non Resident Withholding` on the Distribution side
  (33 rows), so TypeID alone is not a key.
  **`"contrib" in MajorType.lower()` sweeps all eight**, so partnership expenses and management
  fees land in `funded_to_date` alongside real capital. Same over-broad match on both the ROE and
  capitalisation paths.
- **GL side: `2526` = "PSC Pref Equity"** (`config.py:228`). Verified on The Gathering:
  `2526` / `344500-00 CAPITAL - PEACEABLE` = **−9,327,500.50**, matching PPIGPA funded-to-date to
  the cent, flat since Jan-2024 (gross, never netted for the 983,287.41 ROC).
- **2526 is unreliable — only 19 of 85 deals have a row in it.** PSC-labelled capital also sits in
  **2530 (22 deals)**, 2534 (1), 2540 (2). 18 deals have PSC-labelled capital and no 2526 row at
  all, incl. 30 Bearfoot, Apple Self Storage, Pontchartrain, Willowdale, Donald Lynch, Quakertown.
- **MRI has no standardised chart of accounts** — each property carries its own native GL string in
  `vInput` (`344500-00`, `3105 -`, `31051-000`, `320014 -`, `2730-0500`, `505610-460-00`). The 25xx
  number is PSC's normalised mapping bucket, applied inconsistently. **Filter on `vInput` text, not
  the account number.** The `coa` table is useless here: only `vcode` + `vAccountType`, and it types
  every 70xx as `Expenses`.
- **Gotchas**: the misspelling **`PEACABLE`** appears in live data (`CAPITAL CONTR-PEACABLE`,
  `325800-00 DRAWS - PEACABLE`, `ACCRUED PREF-PEACABLE`) — any text filter must include it.
  2530/2536 **mix contributions with draws** (`325800-00 DRAWS - PEACEABLE` sits beside
  `344500-00 CAPITAL - PEACEABLE`; on The Gathering that draws line is +4,589,104.78 = cumulative
  pref *and* capital). **`p0000073` and `P0000073` both exist** as distinct vcodes in
  `isbs_interim_bs` (Donald Lynch) — double-counts on any GL-side aggregation.

### 30 Bearfoot (P0000001) — the 2023-03 cluster is a RED HERRING

Full unfiltered ledger, 123 rows, 2 investors (`OPMCCORD`, `PPI27`). The 2023-03-01..2023-04-15
window has exactly 6 rows:
- `+210,000` (03-17), **`−210,000` (03-23)**, `+210,000` (03-23) — all **OPMCCORD**, TypeID 1016
  Return of Capital. Pattern is book → reverse → re-book same day; net +210,000.
- **`+315,000` (03-23) is a SEPARATE STANDALONE row** — PPI27, TypeID 1016, Return of Capital. Only
  row in all 123 with |Amt| = 315,000; no `Cum_Amt` equals it. Not a net, not a rollup.
- **This cluster contributes ZERO to ROE**: every 210,000 row is OPMCCORD, an operating partner,
  skipped at `one_pager.py:1219`; and Return of Capital is out of the numerator regardless.
- **The real offenders are 4 negative `Distribution: Excess Cash flow` rows, all PPI27, TypeID
  1020, `ROE_Income='Y'`**: 2023-06-23 −11,007.12 · 2023-07-24 −10,652.05 · 2023-08-21 −11,007.12 ·
  2023-09-22 −11,007.12. Sum −43,673.41 → **phantom 87,346.82** (6.80% of the shipped numerator).
  The same magnitudes recur positively on 9 other dates, so they read as reversals of a recurring
  monthly distribution — nothing marks them as such.
- **ROE to Date (cutoff 2026-06-30, inception 2021-01-12): 22.0236% as shipped → 20.5260%
  respecting signs, −1.4975pp.** Wtd avg capital **identical** at 988,843.47 both ways — zero
  denominator-side offenders on this deal. Fixes about half the gap to the audit's ~21.3%; the rest
  is the acquisition fee in the denominator and CF-after-payoff (separate from the sign bug).
- Method note: the sign-respecting walk was **asserted against `metrics.calculate_roe_detailed`**
  (match to 1e-12 on ROE, to the cent on wtd avg capital) before any output was written.

### Partner-equity abs() bug — row-level proof for 4 deals (confirms Investigations 3+4)

Replicates `one_pager.get_capitalization_stack():489-526`. **No date filter** (`:494`).
`max(0, balance)` floor per investor at `:524` was **not** in play on any of these four.

| deal | vcode | OP* rows | as shipped | corrected | overstatement | driving row |
|---|---|---|---|---|---|---|
| JB Fair Park | P0000021 | 4 | 11,550,000.00 | 3,850,000.00 | **7,700,000.00** | 2020-12-18 OPTREV **+3,850,000.00** |
| Cocoplum Apartments | P0000084 | 10 | 29,978,294.10 | 23,350,000.00 | **6,628,294.10** | 2023-09-06 OPASH **+3,314,147.05** |
| OREI Portfolio | P0000033 | 8 | 8,124,512.33 | 8,124,512.33 | **0.00** | none — zero sign flips |
| Pegasus Life Storage | P0000066 | 8 | 3,476,389.73 | 2,573,473.25 | **902,916.48** | 2023-08-08 OPPEGA **+451,458.24** |

Total 53,129,196.16 → 37,897,985.58 (**15,231,210.58** overstated across the four).
Every offender is a **positive `Contribution: Investments` (TypeID 1018)** on the operating partner;
**no negative Return-of-Capital row on any of the four.** Corrected totals land on round figures
(3,850,000 / 23,350,000), which is independent evidence the netting is right. JB Fair Park is the
starkest: 3 contribution rows netting −3,850,000 but summed by `abs()` to 11,550,000 = **3x** true
equity. **Belleville (+1,949,830) was NOT re-verified this session** — still only from `inv34`.

**OREI is purely definitional — confirmed, and one correction to the earlier note:** zero
sign-flipped rows, so `abs()` cannot be its cause. Partner-side breakdown is
`Contribution: Investments` 6,890,613.07 (2 rows) + **`Contribution: Operating Capital`
1,233,899.26 (6 rows)**. **There are NO `Contribution: Others` rows on OREI at all** — that
Typename does not appear on this deal, so it is not part of OREI's gap (it is Brainerd's).

**New flag — Pegasus investor bucketing:** Pegasus has **three** investors — `OPPEGA`, `PPILFS` and
**`TGA22`**. Only `OPPEGA` starts with `OP`, so **TGA22 is bucketed into PREF equity** by `:526`.
Given TGA22 is the PSCKOC JV entity, whether that is intended is worth a look. Does not affect the
902,916.48 above, but it does affect how Pegasus's cap stack splits pref vs partner.

### Deliverables (Charlene's Downloads folder, formula-driven where useful)

- `The_Gathering_UW_ROE_Audit_v2.xlsx` — 8 tabs; every formula recalculated in Excel and tied to the
  app's values (numerator 1,883,127.20, wtd cap 8,329,059.44, U/W ROE 4.7619%). v1 exists but was
  locked open in Excel, so v2 carries the corrected root cause.
- `30_Bearfoot_abs_bug_audit.xlsx` — Raw Rows / Numerator Build / Summary.
- `Cocoplum_partner_equity_abs_bug.xlsx` — Full Ledger / Offending Rows / Summary.
- `Partner_Equity_abs_bug_4_deals.xlsx` — Portfolio Summary + a tab per deal.

### Open — needs Charlene's decision

1. **Which metric is U/W ROE meant to be?** Match the Excel's per-period `row15/row7`, or a correct
   inception-to-date annualised ROE? They are not the same target and no denominator fix reconciles
   them on lumpy deals.
2. **Route 7073 into the denominator?** If yes: (a) full 5,768,443.30 or only the 5,120,654 equity
   step; (b) where does the 647,789.30 participation go; (c) ISBS date 11/30/2022 or the Excel's
   10/1/2023 bucket. Scope: **54 deals carry 7073.**
3. **Pro-rate trap on any 7073 fix** — 7073 is lumpy, not monthly. Through the existing
   `one_pager.py:1092` branch its cumulative would be divided by month 11, keeping 524,403.94 and
   silently discarding **5,244,039.36**. Disposition accounts need different handling from monthly
   cash-flow accounts.
4. **Reversal netting policy** (still open from the earlier session) — now with row-level proof:
   4 partner-equity rows and 4 numerator rows across the deals examined. Affects Total Cap, ROE and
   the waterfall seed simultaneously.
5. **OREI**: does `Contribution: Operating Capital` 1,233,899.26 count as partner equity?
   (excluding → 6,890,613.07)
6. **`"contrib"` over-broad match** — should Partnership Expenses / Management Fees / Organizational
   Costs really land in `funded_to_date` and in the ROE denominator?
7. **Pegasus TGA22** pref-vs-partner bucketing (above).
8. **Anchoring** (informational, no decision needed): actual vs U/W starting capital differ by
   **$0.50** in level, so anchoring year 0 to actuals is safe; the timing differs by **166 days** on
   tranche 2 (U/W 4/1/22 vs actual 9/14/22) = 469,186,965 dollar-days = **$270,737** on weighted-avg
   capital.

**Held pending exec decisions:**
1. **Reversal netting** — should opposite-signed rows net or keep current abs()? Affects
   Total Cap, ROE, and the waterfall seed simultaneously. Scope carefully: proven for 5
   deals, unproven for the 35 fund-level rows.
2. **CF after payoff** — should distributions received once capital is fully repaid still
   count in the ROE numerator when they add nothing to the denominator?

Investigation 1's fix also has to design around three interacting subtleties: the
unmatched fund-level rows, the missing date filter, and the `max(0, balance)` floor.

## Session log — Aug 6, 2026 (One Pager audit: Investigations 3/4/5, Prompt B, Burton fix, live NOI pull)

Driven by two handoff docs in `~/Downloads` — `OnePager_Resume_Investigations_and_Fixes.docx`
and `OnePager_Next_Steps.docx` — plus an ad-hoc live-NOI request. Fuller detail for each area
is in the two sections above; this is the end-to-end record of what was run and decided.

### Method change that mattered

Investigations 1 & 2 (previous session) used the local `accounting_feed.csv` snapshot dated
5/4/2026. **This session switched to live Azure PostgreSQL, and that changed conclusions.**
Azure `accounting` = 12,300 rows, max EffectiveDate **2026-08-03**, vs 11,466 rows /
2026-05-01 in the CSV — **+834 rows**. Azure PG is directly reachable from local dev (the
firewall already allows it); use the `wfadmin` credentials in the Azure Infrastructure
section above. Three of the five "Pref Equity discrepancies" turned out to be nothing but
this staleness.

### What was run (all read-only unless noted)

| script | purpose |
|---|---|
| `scripts/inv34_equity_composition.py` | per-investor / per-typename composition of the equity block, plus abs() and date-filter deltas |
| `scripts/inv34b_basis_variants.py` | recomputes each deal under 6 alternative bases to isolate definition differences |
| `scripts/inv5_jbfairpark_debt.py` | traces both branches of the One Pager debt figure for P0000021 |
| `scripts/inv5b_stale_debt_scan.py` | portfolio-wide scan for stale ISBS debt snapshots |
| `scripts/fixB_verify.py` | before/after for Fix 12, verification for Fix 1 and Fix 11 |
| `scripts/burton_relmap.py` | prints all 7 parent/child portfolio groups and where the loans sit |
| `scripts/burton_blastradius.py` | evaluates the proposed child-loan rule against all 134 deals |
| `scripts/burton_loandump.py` | dumps rendered loan display for all 134 deals to JSON; has a `--diff` mode |
| `scripts/pull_live_noi_requested.py` | live NOI ACT / U-W for named deal-quarter combinations |

**All of these remain UNTRACKED on purpose — they embed the Postgres password.** Confirmed
still untracked at end of session. Worth a `.gitignore` rule if this keeps recurring.

### Key findings, with numbers

**Equity block** — `one_pager.py:518` (`+= abs(amt)`) and the `max(0, balance)` floor at
`one_pager.py:526`. Keyed on `deals.InvestmentID` (1:1 map). **No date filter** —
`quarter_str` reaches only the debt line (`one_pager.py:272-274`).

- abs() reversal bug confirmed on exactly **4** Partner Equity deals, each exactly 2x a single
  positive-signed `Contribution: Investments` row: JB Fair Park +7,700,000 · Cocoplum
  +6,628,294 · Belleville +1,949,830 · Pegasus +902,916.
- **Brainerd Place and OREI Portfolio are zero-delta on every mechanism tested** — not abs(),
  not the date filter, not child properties (OREI's 2 and Brainerd's 9 children carry zero
  contribution / return-of-capital rows). Reduced to a basis question; see open items below.
- **Pref Equity: abs() touches none of the 5 named deals.** Nottingham 2,923,427 (6/1/2026),
  Trolley 2,544,582 (4/20 + 5/15/2026), Belleville 125,000 (6/10/2026) are all post-Q1 rows
  counted in the Q1 figure *and* newer than the released Actual. Azure is more current.

**JB Fair Park debt — the reported premise was wrong.** Not `mOrigLoanAmt` (77,368,000).
`cap['debt']` = **66,363,992**, from one stale ISBS Interim BS row on account 2150 dated
**12/31/2022**. The deal's BS data stops at 6/30/2025 while peers run to 6/30/2026.
`get_isbs_debt_balance()` detects the staleness at `compute.py:98-116` but keeps the
last-known balance because an active MRI loan exists (LoanID 335, `vDateType='Maturity'`).
Interest expense on accounts 5190/7030 = **0** in every period. Portfolio scan: only **3 of
83** deals show stale debt (JB Fair Park 30 months, Post Commons 1 month, Pegasus 21 months
but already forced to 0).

**Fix 1 and Fix 11 were largely false alarms.** Town Fair Tire Portfolio renders
`SOFR + 3.50% | 2/14/2032` correctly. Fix 11's 2nd-loan terms already render on 4 of 5 deals;
only Ascent on Steamboat is blank, and that is correct — both its supplementals are
`vDateType='Paid Off'` and are dropped by the data-layer filter (3 of 89 rows portfolio-wide).
Live revision v153 (Jul 29) includes the Jul 27 loan-terms commit 972990d, so this holds on
Azure too. **No branch was created for Fix 11** — it would have re-done shipped work.

### Commits made this session

| hash | branch | what |
|---|---|---|
| `09ec333` | main (pushed) | **Fix 12** — Location renders "City, State" (`one_pager.py:192-202`). 108 of 134 deals change; 20 stay blank (no City and no State); the append is skipped when the state string is already inside the city string (P0000107 → 'CT, RI'). |
| `72fa80d` | main (pushed) | MEMORY.md — Investigations 1-5 and the U/W ROE 707x findings |
| `6b3d54e` | main (pushed) | MEMORY.md — Burton topology, blast radius, open questions |
| `e725134` | ~~NOT merged, NOT deployed~~ **as of Aug 7: rebased `2086ebc`, merged `1a5e277`, DEPLOYED** | parent deals aggregate child-property loans: new `_child_vcodes_for_parent()` at `one_pager.py:210` (branch numbering), fallback wired in at `one_pager.py:348` |

`git pull` reported "Already up to date" early in the session, but the remote had moved by the
time Fix 12 was pushed — 4 commits (`a7d5e8d` plus 3 debug-endpoint commits) touching only
`flask_app/services/mri_service.py`. Rebased cleanly, no overlap.

**Nothing was deployed.** Live is still v153; pushing to main does not deploy.

### Burton fix — guardrail evidence

`Property_Count` is the parent discriminator (parents >= 1, every genuine child 0). Without it
a group-name match would have wrongly given **22 child properties** their sibling's loan terms
(7 Giant 7, 9 Brainerd buildings, 6 Town Fair stores). Burton is the one deal that looks like a
child by name — `Investment_Name` 'Burton Retail Portfolio' vs group 'Burton Portfolio' — but
carries Property_Count 3.

Verified blast radius across all 134 deals: **131 byte-for-byte identical, 3 changed, `debt`
unchanged everywhere** (display-only). P0000109 → `5.67% | Fixed | 8/28/2032`; P0000007 Berger
Pittsburgh → `2.93% | Fixed | 8/1/2030`; P0000049 Donald Lynch → `3.95% | Fixed | 7/1/2028`.
Berger and Donald Lynch were not named in the request but meet the stated criterion (a parent
showing N/A while holding loans on children).

### Live NOI pull (read-only, no code change)

49 deal/metric/quarter combinations pulled through the production pipeline verbatim:
`config.IS_ACCOUNTS` (config.py:236) → `data_service._normalize_isbs` → `isbs_helpers`
compute_cumulative_noi / cumulative_to_periodic / aggregate_periodic. NOI ACT = ISBS
`Interim IS`; NOI U/W = `Projected IS`. Mapping confirmed live as **4075 and 5092 IN NOI**,
**5120, 5130, 5160, 5165 below the line** — the post-`6c292ef` state.

48 of 49 returned a value. The one blank — **Town Fair Tire Portfolio NOI ACT Q1 2025 — is
suppressed by design**: the deal closed 2/14/2025, its actuals series starts 2025-03-31, and
`aggregate_periodic` drops any quarter without all 3 monthly rows
(`flask_app/services/isbs_helpers.py:80`). The lone March row carries the full Jan-Mar YTD
(405,627) because it is first in the series and hits the `prior is None` branch — that number
is NOT what the app displays and must not be pasted in as a quarterly value. First displayable
quarter is Q2 2025. The full 49-row result table was returned in chat, not duplicated here.

### Still open / needs Charlene's decision

1. **Brainerd Partner Equity basis** — does the model include `Contribution: Others`
   4,550,000? Azure shows 12,007,677; excluding it gives 7,457,677. (Return of Capital
   6,770,190 does net today.)
2. **OREI basis** — should `Contribution: Operating Capital` count? Azure pref 13,391,868 /
   partner 8,124,512; excluding it gives 10,786,868 / 6,890,613.
3. **Development-deal basis** — funded-to-date vs closing capitalisation for Belleville,
   JB Fair Park, Trolley Square, Brainerd, Pegasus.
4. **Burton branch** — review and decide whether to merge `fix/burton-child-loans`, including
   whether Berger and Donald Lynch also changing is acceptable.
5. **Ascent on Steamboat** — leave the paid-off supplemental hidden, or display it?
6. Still held from before: abs() reversal netting, CF-after-payoff in the ROE numerator, the
   U/W ROE mid-year pro-rate fix, Committed Pref Equity SQL (Jim), `Debug_Progress.xlsx`.

**For Jim:** backfill JB Fair Park balance-sheet data past 6/30/2025; refresh the released
Actual so Nottingham / Trolley / Belleville pick up the recent contributions; deploy to pick
up Fix 12.

**Security note:** a JWT for user `cbui` (admin) was pasted into the session chat. It was never
used — direct PG access covered everything. Recommend rotating it.

## Session Aug 6, 2026 (portfolio sweep) — abs() full map, U/W ROE pro-rate across all deals, 7076

**Read-only. No code changed, no branch, no commit authored in this session.** (`git log`
confirms nothing new since `6b3d54e`; `72fa80d`, `6b3d54e` and branch `fix/burton-child-loans`
belong to other sessions.) Sole artifact: **`abs_bug_evidence_2026-08-06.xlsx`** in the project
root, untracked. All DB access SELECT-only against live Azure PG (`psql-waterfall-dev` /
`waterfall_xirr`): accounting **12,403** rows, deals 134, `isbs_projected_is` acct 7071 **4,079**.
Local `accounting_feed.csv` is staler (11,466 rows) — prefer Azure.

This section is **portfolio-wide and complements the deal-level proofs in the "evening" section
above**; where we overlap the numbers agree (see Corroboration below). It does not supersede it.

### 1. Pro-rate branch — swept EVERY 7071 deal (the earlier note covered 6 of 8)

As-of 2026-03-31. 49 vcodes have 7071 data → 40 computable, **32 move if the
`one_pager.py:1090-1092` pro-rate branch is removed**, 8 unchanged, 9 have no PE accounting.
Numerator restored **$2,443,476.63** (excluding the bad Westbank hit: **$2,027,032.18**).

Discriminating test: compare the pro-rated period's cumulative to the following month's delta.
`cum/run-rate ≈ 1` → single month (pro-rate wrong); `≈ m` → real m-month YTD (pro-rate right).
**31 of 32 test as a single month.** Movers incl. Berger Pittsburgh 12.3355→12.5132% ·
Giant 7 12.1805→12.3244% · Camp Creek 8.2161→8.4336% · Gallery 7.8104→7.8919% ·
Mount Prospect 7.3840→7.5693% · **870 DLB 28.2011→28.6713% (+0.47pp, largest)** ·
30 Bearfoot 9.8070→10.1710%.

**STOP — Centre at Westbank (P0000010), hit 2022-03-31.** The one deal where removing the
pro-rate makes it worse. Series jumps 2021-12-31 (cum −156,166.67, flat since Mar-2021) straight
to 2022-03-31 cum **−624,666.67** with **no Jan/Feb 2022 rows**. 2022 run rate −58,266.87/mo, so
1 month ≈ 58k, 3 months ≈ 175k — observed is **10.72×** run rate. −624,666.67 is exactly the
full-year 2018/2019/2020 total (12 × 52,055.56): a stale prior-year value at the wrong date.
  as-coded (÷3)        numerator 5,627,633.12 → 10.3418%
  full fix             numerator 6,083,119.24 → 11.1789%   ← over-counts by ~416,444
  fix except this hit  numerator 5,666,674.79 → 10.4136%   ← recommended shape
**Untestable:** Asbury Commons (P0000004) — one row in its entire 7071 series (2023-03-31 =
909.99), no comparator. Impact $606.66; ROE 0.0030→0.0090%.
Cleared on the discriminating test (lease-up ramps, fail a naive equality test): JB Fair Park
(ratio 0.15), Addison Princeton (0.69), Dorsett Ridge (0.78), Belleville (1.62 / 0.08).

### 2. Complete abs() map — 138 production sites (228 total; 90 in untracked `scripts/`)

| Class | Sites | Verdict |
|---|---|---|
| A. Sign-forcing on accounting `Amt` | ~34 | BUG |
| B. `abs(periodic)` on ISBS 7071 (`one_pager.py:1107`) | 1 | BUG |
| C. Genuine magnitude (DSCR denominators, variance %, loan sizing, ISBS debt) | ~62 | OK |
| D. Defensive no-ops (capital-call amounts, waterfall `mAmount`) | ~25 | OK, but they normalise the idiom |
| E. Tolerance / date-distance | ~16 | OK |

Class A: `waterfall.py:97,99,841,1190,1197,1203,1210,1278,1358,1363,1365` ·
`compute.py:296,298` · `one_pager.py:518,520,1241,1242,1244,1246,1249` ·
`reports_service.py:330,342,353,356,445,446,448,450,452,733,736,744,768,791` ·
`sold_service.py:214,216,253,255,267,270,461,465,549,555,576,598,676,716,832,845` ·
`portfolio.py:114,116,122`.
**Line numbers in older notes have drifted** — memory cites `one_pager.py:507` and `:1233,1238`;
current code is `:518,520` and `:1241,1249`.
**Latent inconsistency:** `metrics.py:320` uses `abs(sum(a for _,a in contributions if a < 0))`
but `metrics.py:395` uses `abs(sum(a for _,a in contributions))` — **no `< 0` filter**. Two
different totals from one list if a positive row ever appears. Not currently reachable.

### 3. Distribution-side reversals, portfolio-wide — NEW (earlier notes were contributions only)

- Contributions positive: **59 rows / $39,994,127** → abs() overstates by **$79,988,255**
- **Distributions negative: 230 rows / −$17,277,262 → overstates by $34,554,524**

By Typename: Carried Interest 119 / −10,986,820 · **Non Resident Withholding 32 / −3,699,011** ·
Other 8 / −894,284 · Preferred Return 27 / −784,795 · Excess CF 6 / −283,073 · Return of Capital
7 / −270,556 · Acquisition Fee 4 / −257,997 · Income 20 / −55,970 · **Professional Fee Holdback
4 / −41,712** · Tax 3 / −3,046.

**Non Resident Withholding and Professional Fee Holdback are NOT reversals** — legitimately
negative (withheld *from* a distribution), no pairing question, no netting logic needed to fix.
abs() converts $3.83M of withholding into investor income. Also **TypeID 1019 has 35 negative
rows totalling −$8,625,980** that `waterfall.py:1278` and `reports_service.py:330` flip into pref
*payments*, wrongly clearing accrued balances.

### 4. Evidence workbook — `abs_bug_evidence_2026-08-06.xlsx` (project root, untracked)

Tabs: Deal Summary (31) · Offending Transactions (369) · Ambiguous - Review (209) · Validation
(22) · Method (33). As-of **2026-06-30** (last completed quarter, derived not hardcoded).
ROE replica asserted against `metrics.calculate_roe` to **0.00e+00**; two harness bugs had to be
fixed first — CF must not reduce the capital balance, and `one_pager` passes **all** cashflows to
`calculate_roe` and lets it net CF out itself.

369 offending rows / **$119,151,496 phantom**; (a) reversal pair 21, (b) adjustment 139,
(c) ambiguous 209. **All 209 ambiguous are fund-level InvestmentIDs with no vcode** ($85.6M:
PSCPGH 21.5M, PSC3 15.9M, PSCKOC 12.3M, INVF5 11.5M, PPISPM 11.3M, PSCDIF 5.1M) — they never
reach a One Pager. **Deal-scope phantom ≈ $33.5M.**
*Classifier caveat:* the fund-level test short-circuits before the pairing test, so some of the
209 may be clean reversal pairs never tested. Re-classify if a fix extends to fund-level entities.

**18 deals where a shipped figure actually moves.** Largest:
**Apple Self Storage P0000003 — Actual ROE 22.77 → 15.02% (−7.75pp, the largest single error
found, and NOT on the original audit list).** All 37 of its offending rows are Non Resident
Withholding + Professional Fee Holdback. Then: JB Fair Park PE 11,550,000→3,850,000 · Cocoplum
PE 29,978,294→23,350,000 · **Belleville P0000006 PE 3,652,245→1,702,415 (−1,949,830) and U/W ROE
1.92→0.26%** · Adirondack PADIRON PE 1,962,500→762,500 · 30 Bearfoot ROE 22.02→20.53% and U/W
9.81→6.44% · Pegasus PE 3,476,390→2,573,473 · Deptford U/W 10.73→9.59% · Pontchartrain ROE
4.88→3.96% · East Manchester U/W 10.73→9.79% · Prestige ROE 8.58→7.96% · Middle Island U/W
2.06→1.55% · Quakertown U/W 10.33→10.09% · Evergreen U/W 12.32→12.17% · Westbank U/W 10.43→10.39%.
13 further deals carry offending rows but nothing moves. **Donald Lynch P0000073 holds $2.18M of
7071 phantom that displays nowhere** — it surfaces the moment that vcode gets PE accounting.

**Corroboration with the "evening" section above:** 30 Bearfoot 22.02→20.53% and the four
partner-equity deals (JB Fair Park 7,700,000 · Cocoplum 6,628,294 · Pegasus 902,916 · OREI 0.00)
reproduce **exactly** via an independent code path. **Belleville, flagged there as not
re-verified, is now verified at −1,949,830.**

### 5. `ROE_Income` flag exists in the feed and is NEVER read by Python — NEW

`queries/accounting_feed.sql:43` sets `ROE_Income`, marking exactly TypeID 1019 (Preferred
Return) and 1020 (Excess Cash flow). Grep hits only .sql and memory docs — **zero hits in any
.py**. The code re-derives from `Typename` and disagrees. Deal scope (PE investors, vcode deals):
  code's Typename rule  **$169,654,101**
  ROE_Income='Y'        **$146,054,445**   difference **$23,599,656 (1.16×)**
Code is a strict superset. Extra: Distribution: Income 156 rows +27,186,460 · Distribution: Tax
9 +288,632 · Professional Fee Holdback 4 −41,712 · Non Resident Withholding 33 −3,833,721.
With abs() applied the effective numerator is **$177,404,967 (1.21×)** — and those 37
withholding/holdback rows are exactly the Apple Self Storage phantom ($7,750,866). Cross-checks.

### 6. Four different capital-event classifiers on the same rows — NEW

| Path | Driver | Effect |
|---|---|---|
| One Pager ROE, ROE Summary | `Typename` string | Realized Gain = capital event |
| Deal Analysis ROE Audit, partner ROE | `Capital` flag (`loaders.py:258`) | Realized Gain is `Capital='N'` → **operating income** |
| Sold Portfolio | `Typename` (deliberate — CLAUDE.md notes the flag is unreliable at sale) | capital event |
| Pref Balance Detail | `TypeID` 1019/1020 | narrowest |

**$73,574,410 of Realized Gain (41 rows, deal scope) is a capital return on the One Pager and
operating income in the Deal Analysis ROE Audit.** One deal legitimately shows two ROEs on two
tabs — 30 Bearfoot: Deal Analysis 24.62% (incl. projections) vs One Pager 22.02% (actuals only).

### 7. Account 7076 (TENANT IMPROVEMENTS) is completely unmodeled — NEW

Verified: `7076 in ALL_EXCLUDED / CAPEX_ACCTS / EXPENSE_ACCTS / REVENUE_ACCTS` = **False on all
four**. Every aggregation in `reporting.py:83-95` uses explicit set membership with no
"everything else" bucket → TI never reduces NOI, FAD, distributable cash, or the waterfall.

**Camp Creek (P0000075, Retail Non-Grocery)**, driver `forecast_feed` (which carries 2,784 rows
of 7076 portfolio-wide): TI ignored **$4,024,004** (2026-2036) vs modeled CapEx 7050 of only
**$2,902,452** — the unmodeled line is **1.4× the modeled one**. 4.6% of cumulative NOI, peaking
at **10% of NOI in 2035** ($902,011) as rollover hits. 37 deals carry 7076 in Valuation IS:
Camp Creek $8.20M · Woodlands Square $3.56M · Poplar Prairie $2.70M · Donald Lynch $2.69M ·
Merle Hay $2.68M · Deptford $2.60M · Evergreen $2.51M · Giant 7 $1.41M · Mount Prospect $1.37M.
**Sources disagree 2×:** Valuation IS says Camp Creek $8.20M, forecast_feed (priority 1, wins)
says $4.02M. Resolve before acting.
Recommended shape if included: add to `CAPEX_ACCTS`, not a flat deduction — reserve funding
(`capex_paid`, `cash_management.py`) and sign normalisation (`normalize_forecast_signs` forces
`-base.abs()` for `ALL_EXCLUDED`) both come free; one line in `config.py`. Almost no actuals
(1 Interim IS row portfolio-wide) so projections only; history and realised returns unaffected.
Also confirms the earlier 707x table: **7076 = TENANT IMPROVEMENTS** joins 7071-7075, and 7073
carries three `vInput` labels (DISPOSITION PROCEEDS / CAPITAL PROCEEDS / RELEASE COMMITTED
CAPITAL), 7074 two — do not key logic off the label string.

### 8. Why abs() exists at all — git archaeology

No commit ever reasoned about accounting-feed signs. The calls arrived incidentally inside larger
features: `waterfall.py` seed_states in **`3bd3c56`** (Feb 2026, "Add loan aggregation for
sub-portfolios and fix pref accrual bugs"); `one_pager.py` PE in **`950ae4e`** (Jul 2026, "Fix One
Pager ROE to Date to use actual payments only"). Neither message mentions signs.
There *was* a deliberate six-commit sign effort in Jan 2026 — **`ee9c5f3`** (`Implement
normalize_forecast_signs`) → `032c01e` → `18bf8f6` → **`5c7ef98`** (`Introduce NO_REVERSE_ACCTS`)
→ `134bd75` — but entirely on the **forecast/ISBS** side. The sub-ledger never got one.
Root cause: direction is carried by `MajorType` (`loaders.py:261-262`), never by `sign(Amt)`, so
`-abs(amt)` reads as "contributions are negative", not as paranoia. It is right **~97.7%** of the
time (289 offending rows / 12,403), so nothing surfaced it. The idiom is **correct** on the
forecast side (an account's nature is fixed; a wrong sign is an export artifact) and **wrong** on
the sub-ledger (there the sign is information). Same three characters, opposite correctness.
Confirms the earlier finding that no reversal concept exists anywhere: grep
`revers|adjust|void|cancel` = zero hits, and no Typename marks a reversal.

### Still open — needs Charlene / Jim decision (additions to the lists above)

A. **Sign authority (the real fix).** Make `sign(Amt)` authoritative and `MajorType` a label,
   then delete the ~34 class-A abs() calls — rather than 34 point edits. Moves Partner Equity,
   ROE, MOIC and IRR on 18 deals at once. Requires a before/after sweep first.
B. **Withholding rows can ship independently and far more safely** — Non Resident Withholding +
   Professional Fee Holdback are unambiguous, need no pairing logic, and are the entire Apple
   Self Storage −7.75pp error. Lowest-risk real win available.
C. **U/W ROE pro-rate:** remove it **plus a guard** (if first-period cumulative > ~2× the next
   month's delta, log and fall back) so Centre at Westbank surfaces instead of over-counting.
   Alternative: fix the MRI extract — Westbank's Jan/Feb 2022 rows are simply missing.
D. **`ROE_Income` flag vs the Typename rule** — $23.6M apart. Which is the house definition? Is
   `Distribution: Income` operating income? The flag exists and nobody is choosing.
E. **Realized Gain classification** — should the Deal Analysis path move off the `Capital` flag
   onto `Typename` to agree with the One Pager? $73.6M at stake.
F. **7076 Tenant Improvements** — include in FAD? First check whether 7075 Reserves for
   Replacement is the intended funding source (double-count risk) and whether a lender TI/LC
   holdback pays it.

**Housekeeping:** `abs_bug_evidence_2026-08-06.xlsx` sits untracked in the project root — commit,
move, or delete. During this session MEMORY.md grew 468 → 652 → 784 lines from a concurrent
session; this block was appended at the true end after re-reading. Nothing above it was altered.

## Burton co-terminous child loans — Aug 7, 2026

**Appended, nothing above altered.** Second of two Burton fixes; read with the
"Burton child-loan aggregation" section above, which it builds on and partly corrects.

### Problem

Burton Retail Portfolio (**P0000109**, `Property_Count` 3) is a portfolio parent that holds
**no loans of its own** — all three sit on the children: LoanID 324 Foley Square P0000111
($18,226,000), 325 Jubilee Square P0000112 ($30,166,500), 326 Westwood Plaza P0000113
($26,910,000). They are **one co-terminous financing split across the properties**: identical
5.665% Fixed, 84-month term, 360 amort, DCR 1.25, maturing 8/28/2032. Note `dtMaturity` is
**blank on all three** — the date lives in `dtEvent`.

The earlier fix (branch `fix/burton-child-loans`, commit **`2086ebc`**, pre-rebase `e725134`,
merged to main in `1a5e277`) is **merged AND DEPLOYED — confirmed live by Charlene on Aug 7,
2026**, the tell being that Burton renders the double-loan flaw on the live app (pre-fix it
showed 'N/A', so two identical terms can only come from this commit). It correctly stopped the
parent rendering 'N/A', but inherited a
primary/second selection rule written for a single property with a real capital stack. Sorting
the three inherited loans by `mOrigLoanAmt` descending gave Jubilee → primary, **Westwood →
phantom "second loan"**, Foley → silently dropped. Because all three are identical the second
line rendered the same string as the first, reading as a duplicate. Debt also stayed **0**: the
`MRI_Loans` fallback filtered on the parent's own vcode and found nothing.
*(A prior note cited this commit as `becea60` — that hash does not exist; the correct one is
`2086ebc`.)*

### Fix

Branch **`fix/burton-coterminous-loans`**, commit **`55686b0`** — pushed to origin,
**merged: no · deployed: no.** `one_pager.py`, +56/−2.

New `_loans_share_terms()` — true when every row shares rate, maturity and interest type
(maturity read `dtMaturity` then `dtEvent`, matching `_parse_loan`). On the
**parent-inheritance path only**, gated by a new `inherited_from_children` flag:
- the shared term renders as the single/primary loan
- the second slot stays **N/A**
- the debt fallback sums `mOrigLoanAmt` across the children: **0 → $75,302,500**

Burton's cap stack then recomputes: total cap 37,997,500 → **113,300,000**; debt **66.46%**,
pref 23.48%, partner 10.06% (sums to 100.0000%); PE exposure on cap 89.94%; **~69% LTV** on a
$109,000,000 purchase price. No negatives, nothing over 100%. Pref/partner dollars unchanged.
`current_valuation`, `pe_exposure_on_value` and `pe_yield_on_exposure` are 0 before *and* after —
`MRI_VAL` has no row for P0000109. Pre-existing, unrelated to this fix.

### Guardrail

110 deals compared, **only Burton changed, 109 byte-for-byte identical.** The **11 genuine
primary + second-tranche deals keep their second loan** (P0000003 3.20%/5.94%, P0000069
5.35%/SOFR+2.50%, …) — excluded by the same-terms guard, **not special-cased**. Berger
Pittsburgh, OREI and Donald Lynch untouched: Berger's 8 child loans carry **6 distinct term
sets** (senior ~2.9% plus mezz ~7.3% per property) so it fails the guard naturally; OREI has its
own loan so never reaches the inheritance path; Donald Lynch has 1 child loan. **Berger's
differing-terms parent case is a separate open follow-up** — it still shows a primary and a
second and silently drops six loans.

## One Pager first-load quarter default — Aug 13, 2026

**Appended, nothing above altered.**

### Problem

Reported as "dropdown set to 2026-Q2 but the numbers look like 26Q3 — the budget figures in
particular are way off," starting right after Jim loaded 26Q3 data.

The budget window math was **never wrong**. `one_pager.py` sums YTD Budget over
`(Jan 1 − 1 day, quarter_end]`, which is correctly capped at the selected quarter, and the YTD
Actual anchor cannot reach past `quarter_end` either — injecting Jul–Sep actuals leaves a Q2
anchor pinned. **Explicitly selecting a quarter always worked.**

The divergence is on **first load only**. `OnePagerView.vue:96-111` deliberately sends the first
request with **no `quarter` param**, then labels the dropdown via `getMostRecentCompletedQuarter()`
— **with no refetch**. The server filled that gap at `financials_service.py:963-964` with
`available[0]`, the **newest** quarter with actuals anywhere in the portfolio
(`get_available_quarters` runs on the full ISBS frame, **no vcode filter**). Two different
defaults: newest vs most-recent-completed.

While they coincided nothing showed. 26Q3 actuals split them, and a page **labelled 2026-Q2
rendered 2026-Q3 figures**. YTD Budget is a **cumulative Jan-to-date sum**, so it absorbed a whole
extra quarter — **Jan–Sep instead of Jan–Jun, ~51% overstated** — while YTD Actual, anchored to
each deal's last reported month, barely moved. That asymmetry is why the **budget column looked
wrong first**. The chart is requested only *after* the quarter resolves, so chart and tables on
the same page disagreed by one quarter — a useful tell.

Proven by running the real committed functions: before the Q3 load both defaults returned
`2026-Q2` (bug invisible); after, backend `2026-Q3` vs label `2026-Q2`.

### Fix

Branch **`fix/onepager-quarter-default`**, commit **`c67976b`** — pushed to origin,
**merged: no · deployed: no.** Backend only, no Vue change.

New `most_recent_completed_quarter(quarters, today=None)` in `one_pager.py`, a server-side port of
the Vue helper **including its fall back to the oldest entry when nothing has completed**. Call
site becomes `most_recent_completed_quarter(available) or available[0]`, keeping `available[0]`
only as a belt-and-braces fallback.

### Guardrail

`scripts/onepager_quarter_default_check.py` — **25 checks, all pass.** Three layers: **rule
parity** against an independent transcription of the JS (quarter boundaries Jun 30 / Jul 1 /
Dec 31 / Jan 1, lists with a hole at Q2, lists with nothing completed, empty and malformed input);
**real data** on 5 deals, each corrected Jan–Sep → Jan–Jun (p0000007 12,070,567 → 7,998,113;
p0000059 +52.2%, p0000088 +53.3%, p0000047 +50.5%, p0000053 +51.8% overstatement removed); and a
**wiring assertion** via `inspect.getsource` that fails if the call site reverts to `available[0]`.

**Caveat — the trigger is simulated.** The local ISBS snapshot is Apr 15 2026 (actuals stop
2026-03-31) and so contains **no in-progress quarter**, the exact condition that causes the bug.
Section 2 injects synthetic actuals for one donor deal to reproduce Jim's load; the rows are clones
of that deal's latest monthly snapshot, moving only *which periods exist* (all the default rule
reads), and no assertion depends on their amounts. **Needs one live confirmation after deploy:
open a One Pager fresh and check the figures match the label before touching the dropdown.**

**Residual:** minor timezone skew — server uses `date.today()`, browser uses local time, so on a
quarter boundary they can disagree for a few hours. The helper takes a `today` arg if pinning is
ever needed.

### Two related defects found, NOT fixed — still open

1. **Portfolio-wide dropdown.** `get_available_quarters` has no vcode filter, so one deal's newly
   loaded quarter appears on **every** deal's dropdown, and the list can have **holes** (observed
   offering Q3 but not Q2, since no deal had Q2 actuals). Confirmed behaviour.
2. **Projected YE drops months** between a deal's last actual and the selected quarter-end — they
   fall in neither the actual nor the remainder budget (remainder starts *after* `quarter_end`,
   nothing backfills the gap). p0000007's Projected YE NOI falls 13.4M → 9.3M → 5.3M as the
   dropdown advances; 81 of 81 deals in the Apr snapshot have actuals ending before 2026-06-30.
   **Do NOT treat this as a confirmed bug** — it may be intended (budget starts after the selected
   quarter, no backfill by design). Pending confirmation of whether loading the missing actuals
   picks those months up.

### Post-deploy check still needed — Burton's debt line

The 0 → $75.3M roll-up **only fires when ISBS returns no balance for P0000109**:
`get_isbs_debt_balance()` runs first and ISBS Interim BS takes precedence over the `MRI_Loans`
fallback. Verified here with `isbs_raw=None`, so the fallback ran. On Azure, if ISBS carries a
parent balance that value wins instead and may differ — including the possibility of a
**JB-Fair-Park-style stale row** kept alive by an active MRI loan. The **loan-term collapse is
independent of ISBS and holds either way.**

### Guardrail caveat

Ran against the **Apr-15 `MRI_Loans.csv` snapshot (78 loan rows, 110 deals)**; live Azure was
unreachable all session (PG firewall, no Azure CLI locally). Azure carries **~24 more deals** not
in that CSV, so another parent with co-terminous child loans would also collapse — correctly, but
it was not in the sweep. Re-run `scripts/burton_loandump.py` (which reads PG directly) once the
firewall allows it.

## Deploy handoff to Jim — Aug 7–8, 2026 (COMPLETED)

Both fixes merged, deployed (v197), and a follow-up cap stack fix deployed (v198).

### Fix 1 — One Pager chart window

Branch **`feat/onepager-chart-window`**, head **`8f59bb5`** — pushed to origin, based on
`b62135c` (origin/main). **merged: no · deployed: no.** Three commits:

- **`45539ff`** — the window change. Replaces and **deletes** `cap_to_last_actual` (from
  `becdf96`), which the new rule subsumes and would contradict.
- **`b9021ce`** — guardrail + CLAUDE.md.
- **`8f59bb5`** — Date-Closed pre-close zero-fill.

The chart now shows a **rolling 10-quarter window ending at the SELECTED dropdown quarter**
(26Q1 → 23Q4–26Q1; 26Q2 → 24Q1–26Q2), not at the latest actual. Every calendar quarter keeps
its x-axis slot; previously the window was index-sliced over the union of periods that *have
data*, so quarters a deal predated were never slots at all — Burton rendered a **one-bar
chart**.

**Quarter wiring confirmed end-to-end:** dropdown (`OnePagerView.vue`) → `?quarter=` →
`request.args` → `get_one_pager_chart` → `_quarter_window`. Proven by driving the real view
body inside a request context with `?quarter=` on the URL; with the param omitted the same
deals fall back to the old data-derived window and land elsewhere, which is what rules out a
hardcoded latest-actual. Before this, **no call site passed the quarter at all.**

**Pre-close quarters are forced to 0** on all three series (ACT, U/W, Occupancy) ahead of the
data lookups, from `inv['Acquisition_Date']`. Fixes **ReNew Glenmoore P0000099** (closed
2025-02-19), which showed a stray **0.78M U/W line in Q4 2024** off 60 pre-close Projected IS
rows — underwriting projections routinely predate closing. **Child properties inherit the
parent's Date Closed** (child `Portfolio_Name` == parent `Investment_Name`): Burton — Westwood
Plaza P0000113 went from unknown → 2025-08-28. Quarters **on or after** Date Closed keep the
data-driven zero-vs-gap split, so a quarter the deal existed for but that simply was not
reported is **not** force-zeroed (P0000085, closed 23Q4, no actuals until 25Q4).

**Guardrail** (`scripts/onepager_chart_window_check.py`, three-way capture: origin/main /
window-only / final): **6 deals × 2 quarters pass.** Only ReNew's **4 stray pre-close values**
changed. Burton intact — 1 bar → 10 slots, 7 zero-filled, 25Q4 actual 2.61M preserved. No
post-close quarter moved. Wiring proof in `scripts/onepager_chart_verify_wiring.py`.

### Fix 2 — Burton co-terminous loans

Branch **`fix/burton-coterminous-loans`**, commit **`55686b0`** — see the section above for
full detail. Replaces the **currently-live double-loan display**: the live commit is
**`2086ebc`** *(an earlier note cited `becea60`; that hash does not exist)*, which stopped the
parent showing 'N/A' but rendered the same loan twice. Collapses to **one term, 2nd = N/A**,
debt **0 → $75,302,500**.

### Handoff method

Sent Jim a Word doc — **`OnePager_Deploy_Handoff_for_Jim.docx`** — with branches, commits,
deploy commands and caveats, so **his Claude Code can execute the deploy**. The doc asks Jim's
Claude to send back a **"Deploy_Result"** doc containing build status, the new revision number,
and any errors.

### Deploy result — completed Aug 8, 2026

Both branches **merged and deployed** by Jim's Claude Code.
- ACR build **ca9c** succeeded (Vue/TS compiled cleanly). Revision **v197**.
- Merge commits: `f711e2e` (chart window), `f73ef23` (Burton loans). Final main: `f73ef23`.

### Follow-up fix — Cap stack equity quarter filtering (v198)

Burton had a **$27,630,000 PPI contribution on July 1, 2026** (4th property acquisition) — one
day after Q2 ends. The cap stack equity computation (`get_capitalization_stack()` in
`one_pager.py`) had **no date filter** on accounting transactions, so the contribution showed
in the Q2 2026 report. Debt was already filtered via `get_isbs_debt_balance(as_of_date=quarter_end)`.

**Fix** (`9086f16`, revision **v198**): Filter `deal_acct` to `EffectiveDate <= quarter_end`
when `quarter_str` is provided, before computing pref equity and partner equity balances.
Without `quarter_str` (non-One Pager callers), behavior unchanged.

### Remaining caveats

1. **Frozen snapshots keep their old arrays.** One Pager snapshots approved before the chart
   change still hold the old sparse quarter arrays and would need backfilling.

### Process note — concurrent Claude Code sessions

Multiple sessions running against the **same working tree** moved HEAD mid-run repeatedly this
session: an implementation commit landed on **`main`** instead of its feature branch, another
session's memory commit landed on the **feature branch**, and a new branch was created off the
**wrong base**. All repaired — `main` restored, commits relocated, the duplicate dropped — but
the fix cost real time and risked losing work. **Work one tab at a time, or give each session
its own `git worktree`.**

## One Pager NOI/Occupancy chart — dual-axis scaling fix (Aug 10, 2026)

Committed **straight to `main` as `34b8d92`** (rebased onto `origin/main` `6a5a43c` first, pushed
clean). **Not deployed** — handed to Jim, time-sensitive for Monday KOC reporting.

**The bug.** Occupancy is pinned 0-100 on the left axis, so a bar's top sits at `occ/100` of the
plot height. The right axis (NOI, $M — orange U/W and grey ACT lines) had **no `min` and no
`max`**, so ECharts auto-fit it to the NOI data: the peak NOI point always landed near the top of
the plot area and the auto *min* sat above zero, lifting the whole line off the baseline. The
lines rendered **above** the bar tops, reading as "NOI exceeds occupancy" even though the units
are unrelated. Worst case was OREI Portfolio (`p0000033`): the U/W line ran at 86-100% of plot
height while the bars fell to 68% — floating up to **32% of the plot** above them.

**The fix** (`noiAxisBounds()` + `niceCeil()` in `OnePagerView.vue`). Right-axis bounds are
computed from the window's own data, **never hardcoded**: the tallest NOI point may reach at most
`headroom` of the plot height, taken from the *shortest real occupancy bar* in the window. Two
things that are easy to get wrong:
- The bound must be solved against the **actual axis floor** — `(top - min) / (max - min) <=
  headroom`, not `max/headroom`. A deal with a negative-NOI quarter drops the floor below zero,
  and the naive form still overshot (Town Fair Tire - Avon `p0000101`, ACT -0.48 in 25Q3, was the
  one deal that failed the first formula).
- **Zero-occupancy slots are excluded** from the bar minimum — a pre-close quarter is 0 on all
  three series by definition (see the chart-window entry above), so there is no bar to sit under.

A 0.45 `headroom` floor keeps a lease-up window from crushing the lines into the axis. Worst
compression in the sample is Old Kinderhook (`p0000031`) at 37% of the axis: its occupancy feed
is 0.0 for nine of ten quarters with a single 38.2% reading, and it has negative-NOI quarters.
Both axes now get 5 intervals so their gridlines coincide.

**Guardrail**: `scripts/onepager_chart_axis_check.py` — runs the real `get_one_pager_chart` over
11 deals x 2 report quarters (~0.2M to ~2.6M quarterly NOI) and checks, **per quarter**, where
each NOI point lands as a fraction of plot height versus that quarter's *own* bar. PASS requires
no NOI point above its bar, nothing clipped, and peak NOI still using >=33% of the axis. Burton
(`p0000109`) keeps its 25Q4 actual of 2.61M at 87% under a 95% bar.

**Deploy note**: this is a **Vue/TS change**, so the `az acr build` step is the real check —
`vue_app/node_modules` is absent locally and the build only runs in Docker, so it was **not
typechecked here**. Verified on the local Apr-15 ISBS snapshot; live Azure data is more complete.

**Physical Occupancy relabel** (`78cbc29`, same deploy): the chart title became "Physical
Occupancy vs. NOI" and the bar series "Physical Occupancy" — the chart plots physical occupancy
from MRI_Occupancy_Download, while Property Performance above it reports *economic* occupancy
(physical less bad debt/concessions). Label-only; the series `name` is the single source for both
legend and tooltip.

## One Pager — Physical Occupancy relabel + Current Anticipated Exit (Aug 10, 2026)

Deploy handoff entry for the three changes now on `main`, head **`a2d242f`**.

**Physical Occupancy relabel** (`78cbc29`, `OnePagerView.vue`): chart title "Occupancy vs. NOI"
→ "Physical Occupancy vs. NOI", and the bar series "Occupancy" → "Physical Occupancy". The chart
plots *physical* occupancy from MRI_Occupancy_Download, while Property Performance on the same
page reports *economic* occupancy (physical less bad debt/concessions) — same word, two different
numbers. The series `name` is the single source for both legend and tooltip.

**Current Anticipated Exit field** (`a2d242f`): new date pipe-separated next to "Underwritten
Exit" on the One Pager, in **both** the single and batch/print views. Sourced from the
`event_dates` table: `vEventType='Disposition'` AND `vEvent='Closing'` AND `vDateType='Projected'`,
matched by `vCode`, value from `dtEvent`. Duplicates take **MAX(dtEvent)** (latest wins) — a
projected closing gets revised as an exit approaches, and this matches the
`Prop_Info_Core.sql:22` precedent for the U/W exit. Empty / no matching row / unparseable date →
`None` → renders **N/A**, same as Underwritten Exit already does. New
`get_current_anticipated_exit()` in `one_pager.py`; `event_dates` loads through the existing
adapter registry in `load_all`, is added to `refresh_table`'s `table_to_key` so a CSV re-import
invalidates it, and is threaded through **all four** `get_one_pager_data` call sites (single,
batch/print, assistant tool, snapshot freeze).

⚠️ **`event_dates` population in Azure was NOT verified** — the dev machine has no local copy of
the table (absent from `waterfall.db` and `csv_data/`), no `.env`/`DATABASE_URL`, no `az` CLI and
no `psql`. If the table lacks matching rows the field shows **N/A on every deal**; the rest of the
page is unaffected. Note the deals table's `Anticipated_Exit` comes from `Prop_Info_Core.sql` run
against MRI *directly*, so it is **not** evidence that this table has rows here.
`scripts/event_dates_exit_probe.py` (read-only, one SELECT) reports match counts, sample dates,
duplicate deals and active-deal coverage — run it with `DATABASE_URL` set, ideally *before*
spending a deploy.

**Post-deploy spot-check**: confirm real dates appear (not N/A), and that the pipe line does not
wrap awkwardly in print — the value cell is 28% wide (30% print) and now carries ~45 characters.
`white-space: nowrap` should widen the column instead of wrapping, but this could not be rendered
locally (`vue_app/node_modules` absent). Fallback if it wraps: give the exit pair its own
`colspan="4"` row.

**Known bug, deliberately deferred**: `fmtDate` displays dates **one day early** — JS parses the
`YYYY-MM-DD` the API sends as UTC midnight, then reads it back with local getters, shifting back a
day in any negative-UTC-offset zone (verified in Node at America/New_York: `2027-06-30` →
`6/29/2027`). Affects **Date Closed, Underwritten Exit, and the new Current Anticipated Exit**
consistently. Left untouched on purpose — to be fixed after KOC as a separate change, which would
correct all three date fields together.

**Deploy status**: all three merged to `main` — `34b8d92` (NOI axis scaling), `78cbc29` (relabel),
`a2d242f` (exit field) — head **`a2d242f`**, handed to Jim, time-sensitive for Monday KOC
reporting. **Not yet deployed.** All three are Vue/TS changes, so `az acr build` is the real
typecheck; nothing was typechecked locally.
