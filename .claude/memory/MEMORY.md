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
- **Current revision**: v153 (deployed Jul 29, 2026) — Print cleanup: suppress browser headers/footers, date/time stamp, business plan flex expand, chart anchored to bottom
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

### Status — HELD, nothing started

**NOT yet run:** `Debug_Progress.xlsx`.

**Held pending exec decisions:**
1. **Reversal netting** — should opposite-signed rows net or keep current abs()? Affects
   Total Cap, ROE, and the waterfall seed simultaneously. Scope carefully: proven for 5
   deals, unproven for the 35 fund-level rows.
2. **CF after payoff** — should distributions received once capital is fully repaid still
   count in the ROE numerator when they add nothing to the denominator?

Investigation 1's fix also has to design around three interacting subtleties: the
unmatched fund-level rows, the missing date filter, and the `max(0, balance)` floor.
