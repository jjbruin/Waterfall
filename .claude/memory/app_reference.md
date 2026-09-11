# Application Reference — tabs, views and the AI Assistant

What each tab of the app shows and which endpoints serve it. This is a **feature
catalogue**, not a set of working instructions: it was split out of `CLAUDE.md` on
Sep 11 2026 because it was half that file's bytes and was being loaded into every
session whether or not the work touched the UI.

Read it when you need to know what a view displays, what a section is called on
screen, or which endpoint backs it. For how to work in this repo — the deploy
procedure, the symptom-repair rule, domain conventions, account classifications —
stay in `CLAUDE.md`.

**Verify before relying on a detail here.** Descriptions of screens drift faster
than the code does; the endpoint list and the code are the authority.

## Application Tabs

### 1. Dashboard
Executive portfolio-level view with instant-load KPIs and charts. Vue: `DashboardView.vue`. Flask: `dashboard.py` + `dashboard_service.py`.
- **KPI Cards** (6): Portfolio Value, Debt Outstanding, Wtd Avg Cap Rate, Portfolio Occupancy, Deal Count, Total Preferred Equity
- **Portfolio NOI Trend** — Dual-axis Altair chart (occupancy bars + Actual/U/W NOI lines) aggregated across all deals. Values in $ million. Frequency and period-end selectors (defaults to most recently ended quarter), trailing 12 periods. Capped at last closed quarter.
- **Portfolio Capital Structure** — Consolidated vertical stacked bar (Debt blue / Pref Equity green / OP Equity grey) with Avg LTV and Pref Exposure annotations at dividing lines. Values in $ million.
- **Occupancy by Type** — Horizontal bars showing weighted-average occupancy per Asset_Type, colored above/below portfolio average with dashed reference line.
- **Asset Allocation** — Donut chart by Asset_Type sized by preferred equity, with % of total in legend table and hover tooltips.
- **Loan Maturities** — Stacked bar chart by maturity year (Fixed blue / Floating orange) with weighted avg rate labels on fixed sections, total dollar labels, and "Show Data" expander with loan-level detail table (includes child property loans).
- **Computed Returns** (button-gated) — Progress bar → IRR by Deal bar chart + formatted summary table (Contributions, Distributions, IRR, ROE, MOIC)

### 2. Deal Analysis
Main waterfall computation, partner returns, capital accounts, XIRR/MOIC metrics. Vue: `DealAnalysisView.vue`. Flask: `deals.py` + `compute_service.py`.

**Layout**: Deal Information + Capitalization → Parcel Sales (expandable) → Sale Assumptions Override → Deal-Level Summary (KPI cards) → Partner Returns (non-OP partners highlighted bold with blue-grey background) → Annual Forecast (whole-dollar formatting, DSCR as 2-decimal, blank spacer/header cells) → expandable sections (Diagnostics, Debt Service, Cash Management, Capital Calls, XIRR Cash Flows, ROE Audit, MOIC Audit).

**XIRR Cash Flows**: Merged side-by-side table with columns Date, Description (typename from `cashflow_details`), one amount column per partner, and Deal total column. Rows are keyed by (date, description); same-date/same-description cashflows for one partner are summed. Partner Returns IRR is computed from `combined_cfs` (the authoritative cashflow list); the XIRR Cash Flows table/Excel is a display-friendly pivot of the same data via `cashflow_details`.

**Acquisition Fee Exclusion**: "Distribution:Acquisition Fee" entries from accounting are excluded from all XIRR/ROE/MOIC calculations system-wide. Acquisition Fees are fees, not operating income — they must never inflate ROE. Exclusion points:
- `build_partner_results()` in `compute.py` — filters `metrics_cfs` and `cashflow_details` by label
- `seed_states_from_accounting()` in `waterfall.py` — excludes from `cf_distributions` (affects Deal Analysis ROE Audit, partner/deal ROE)
- `build_roe_summary_row()` in `reports_service.py` — excludes from `cf_distributions` (ROE Summary report)
- `get_pe_performance()` in `one_pager.py` — excludes from `cf_distributions` (One Pager ROE to Date)
- `irr_needed_distribution()` in `waterfall.py` — excluded from IRR hurdle lookback
- The **Sold Portfolio** tab is **not** affected — it computes returns directly from raw accounting data via `sold_service.py` and handles Acquisition Fees separately (fee consumed, investor gets $0).

**Annual Forecast Formatting**: Black border lines under Expenses, Capital Expenditures, and Other Below-the-Line rows (`underline-row` CSS class), black border above Total Distributions (`topline-row` CSS class). Row order: Revenues → Expenses → NOI → Tax Abatement → Interest → Principal → Total Debt Service → Capital Expenditures → Other Below-the-Line → FAD → DSCR → waterfall allocations.

**FAD and CapEx**: FAD = NOI + Tax Abatement + Interest + Principal + Other BTL + CapEx + CapEx paid from reserves. CapEx is initially fully deducted (negative), then the portion funded from cash reserves (`capex_paid` in cash schedule) is added back. Only unfunded CapEx reduces FAD. The cash schedule (`build_cash_flow_schedule_from_fad()` in `cash_management.py`) tracks `capex_paid` vs `capex_unpaid` per period; `annual_aggregation_table()` aggregates `capex_paid` by year from the cash schedule.

**Excel Downloads** (Vue): Per-section download buttons ("Excel") on each section header + "Download Full Deal Analysis (Excel)" button at top of results. Uses `fetch` + `Blob` with `Authorization: Bearer` header. Sections: Partner Returns, Annual Forecast, Debt Service, Cash Management, Capital Calls, XIRR Cash Flows, ROE Audit, MOIC Audit. Full workbook combines all 7 sheets.

**Excel API Endpoints** (`/api/deals/<vcode>/excel/`): `partner-returns`, `forecast`, `debt-service`, `cash-schedule`, `capital-calls`, `xirr-cashflows`, `full` (7-sheet workbook). All GET, login_required.

**Excel Generators** (`compute_service.py`): Shared helpers `_excel_styles()`, `_write_header_row()`, `_autosize_columns()`. Per-section: `generate_partner_returns_excel()`, `generate_forecast_excel()`, `generate_debt_service_excel()`, `generate_cash_schedule_excel()`, `generate_capital_calls_excel()`, `generate_xirr_cashflows_excel()`. Full: `generate_full_deal_excel()` — 7 sheets including ROE/MOIC audit via `load_workbook` copy.

**Audit Expanders** (after XIRR Cash Flows):
- **ROE Audit — Return on Equity Breakdown**: Unified event table with all cashflows (contributions, CF distributions, capital returns) in date order. Event labels use accounting Typename (e.g., "Distribution: Preferred Return", "Contribution: Investments"). Each row shows amount, running capital balance, days since prior event, and weighted capital. CF distributions do not reduce capital balance; only capital events do. Additional columns: Pref Due (cumulative pref owed at waterfall rate), Pref Paid (cumulative pref payments), Pref Accrued (due minus paid), ITD ROE (inception-to-date ROE at each event). Metric cards per partner include Pref Due, Pref Paid, Pref Accrued. Deal-level section with same breakdown. Excel download.
- **MOIC Audit — Multiple on Invested Capital**: Cashflow Breakdown table (Date, Description, Type, Amount), 6 metric cards per partner (Contributions, CF/Cap/Total Distributions, Unrealized NAV, MOIC). Deal-level section with note that deal MOIC uses realized distributions only. Excel download.

### 3. Property Financials
Vue: `PropertyFinancialsView.vue`. Flask: `financials.py` + `financials_service.py`. Sections in order:
- **Performance Chart** — Actual vs U/W NOI lines + occupancy bars (ECharts). Supports Monthly/Quarterly/Annually with configurable period window. Defaults to most recently completed actual period.
- **Income Statement** — Two-column comparison (TTM, YTD, Full Year, Estimate, Custom). Sources: Actual, Budget, Underwriting, Valuation. Independent left/right "As of Date" selectors for cross-period comparison. Valuation source negates MRI sign convention (`-mAmount_norm`).
- **Balance Sheet** — Two-period comparison with variance.
- **Tenant Roster** — Commercial lease data with rollover report, maturity chart, and printable HTML. $/SF displayed as currency with 2 decimal places.

### 4. One Pager
Standalone route at `/one-pager`. Vue: `OnePagerView.vue`. Flask: `financials.py` + `financials_service.py`. Professional investor report matching printed PDF layout.
- **Data Logic** (`one_pager.py`): `get_general_information()`, `get_capitalization_stack()`, `get_property_performance()`, `get_pe_performance()`, `get_one_pager_comments()`/`save_one_pager_comments()`.
- **General Information** — Left column: Partner, Location, Asset Type / Year Built (combined with `|` separator), # Units / SF. Right column: Investment Strategy, Date Closed, Underwritten Exit (from `event_dates`: `Asset Management / U/W Exit / Actual`), Current Anticipated Exit (from `event_dates`: `Disposition / Closing / Projected`). Both exit dates sourced from `event_dates` table via `_lookup_event_date()` — no longer from `inv.Sale_Date`.
- **Capitalization / Exposure / Deal Terms** — Purchase Price (from deals `Acquisition_Price` or valuations), P.E. Coupon/Participation (from waterfall Pref/Share steps), Loan Terms (from MRI_Loans: `nRate% | Fixed | M/D/YYYY` or `vIndex + vSpread% | M/D/YYYY`), 2nd Loan Terms (next largest loan by `mOrigLoanAmt`), Rate Cap, P.E. Yield on Exposure (NOI / (Debt + PE), computed in service layer). Loan terms always populated from MRI_Loans regardless of debt source (ISBS or MRI_Loans fallback). Parent portfolio deals with no direct loans aggregate child property loans via `_child_vcodes_for_parent()` in `one_pager.py`. Capitalization table: Debt (from ISBS balance sheet)/Pref. Equity/Partner Equity/Total Cap with %. All components are as-of the quarter end date: debt uses `get_isbs_debt_balance(as_of_date=quarter_end)`, equity filters accounting transactions to `EffectiveDate <= quarter_end`. Valuation sorted by date descending (most recent used). P.E. Exposure on Total Cap and on Value. Pref Equity capitalization (editable comment, persisted to `pe_cap_comment` in `one_pager_comments` table; carries forward across quarters via vcode-level fallback).
- **Property Performance** — Table with At Close (from `at_close_noi` MRI table or Projected IS earliest Dec 31 fallback), Projected YE (YTD Actual + remainder-of-year Budget), U/W YE on the left (Annual Financial Comparison); YTD (Actual), YTD (Budget), Variance (% of budget) on the right (As of quarter end). Rows: Economic Occ., Revenue, Expenses (underlined), NOI, DSCR. Amounts in $M, DSCR as X.XXX. Economic Occ sources: YTD = avg physical occ - bad debt %, Projected YE = weighted avg of actual + budget remaining months, U/W YE = 1 - (vacancy 4030 / rental income 4010) from Projected IS, At Close = from `deal_terms` table `econ_occ_at_close`. Editable performance comments.
- **Preferred Equity Performance** — Committed PE, Remaining to Fund, Funded to Date, Return of Capital, Current PE Balance, Accrued Balance, Coupon, Participation, ROE to Date, U/W ROE to Date. Editable accrued pref comment. Enriched from deal analysis waterfall via `_enrich_pe_from_deal_result()` in `financials_service.py`: Current PE Balance and Accrued Balance from `seed_states` (current state at actuals boundary, not terminal projected state); ROE to Date from actual accounting distributions through quarter end via `calculate_roe()` (not projected); Committed PE falls back to total PE contributions if commitments table empty. U/W ROE to Date uses ONLY ISBS Projected IS data — no actual accounting: account 7073 for capital events (positive = contribution, negative = return of capital) via `_get_uw_7073_signed()`, account 7071 for distributions via `_get_uw_pe_distributions()`. Supplemental 7073 records can be uploaded via `ISBS_UW_Supplements.csv` (protected table, persists across MRI refreshes).
- **Business Plan & Updates** — Editable free-text comments.
- **Occupancy vs. NOI Chart** — ECharts dual-axis. Occupancy bars + NOI U/W and NOI ACT lines. Values in $ millions. Rolling window of exactly 10 quarters ending at the selected report quarter (`window_end_quarter` in `get_one_pager_chart()`); quarters the deal predates keep their x-axis slot at 0, an in-progress quarter renders as a gap. Property Financials chart keeps its data-derived window and forward U/W lookahead.
- **Comments** — Four editable fields (performance, accrued pref, business plan, PE capitalization) persisted to `one_pager_comments` table per vcode + quarter. PE capitalization is a borderless inline textarea that carries forward across quarters. Comments are editable throughout the review process and locked (read-only) only after final approval.
- **Review Workflow** — Sequential approval pipeline: Asset Manager → Head of AM → President → CCO → CEO → Approved. `ReviewPanel.vue` component shows status indicator, approve/return buttons (role-gated), and threaded review notes. Return sends document back to Draft with a required note. Comments editable during review, locked only after final approval.
- **Snapshot System** — On CEO final approval, all computed One Pager data + chart frozen into `one_pager_snapshots` table as JSON via `_save_snapshot()` in `review_service.py`. "View Approved Version" / "View Live Data" toggle button renders frozen data with blue banner (approver + date). Comments read-only from snapshot. API: `GET /api/financials/<vcode>/one-pager/snapshot?quarter=X`. Review status includes `has_snapshot` flag.
- **Print** — `@media print` CSS produces clean single-page output matching the PDF template. Textareas render as plain text in print. ReviewPanel hidden in print. `@page { margin: 0 }` suppresses browser headers/footers (URL, page number); content padding on `.one-pager-page`. Page title temporarily blanked during print to remove "Waterfall XIRR". Business Plan section uses flex layout to expand and fill available space (scrollbar hidden); chart anchored to page bottom. Custom date/time stamp rendered in upper left via `printTimestamp` ref.
- **Date formatting** — `fmtDate()` in `OnePagerView.vue` parses ISO dates (`YYYY-MM-DD`) by regex to avoid JavaScript `new Date()` UTC→local timezone shift (midnight UTC displayed as prior day in US timezones). Non-ISO formats fall through to `new Date()`.
- **API Endpoints**: `GET /api/financials/<vcode>/one-pager` (all data), `GET /api/financials/<vcode>/one-pager/chart` (quarterly chart data), `PUT /api/financials/<vcode>/one-pager/comments` (save comments, blocked only when approved), `GET /api/financials/<vcode>/one-pager/snapshot?quarter=X` (frozen approved snapshot).
- **Review API Endpoints** (`/api/reviews`): `GET /<vcode>/<quarter>` (status + notes + permissions), `POST /<vcode>/<quarter>/submit` (submit for review), `POST /<vcode>/<quarter>/approve` (advance step), `POST /<vcode>/<quarter>/return` (return to draft), `POST /<vcode>/<quarter>/note` (add discussion note), `GET /tracking` (production pipeline data), `GET /roles` (list assignments), `POST /roles` (assign role), `DELETE /roles/<id>` (remove role).
- **Database Tables**: `review_roles` (user↔review_role, UNIQUE), `review_submissions` (vcode+quarter, status, current_step), `review_notes` (audit trail with action/note_text), `one_pager_snapshots` (vcode+quarter UNIQUE, snapshot_data JSON, approved_by, approved_at). All four in `PROTECTED_TABLES`.

### 4a. Review Tracking
Standalone view at `/review-tracking` (`ReviewTrackingView.vue`). Production pipeline dashboard for One Pager approval status across all active deals.
- **Summary Cards**: Draft / In Review / Returned / Approved counts (clickable to filter).
- **Filters**: Quarter (text input), Status (dropdown). Refresh button.
- **Table**: Deal name, Quarter, Status badge, Step label, Updated date. Click row → navigates to `/one-pager?vcode=X&quarter=Y`.
- **Data**: LEFT JOINs deals with `review_submissions` so unsubmitted deals show as "Draft". Excludes sold deals and child properties.

### Review Role Management (Settings)
Admin-only section in `SettingsView.vue`. Table of current review role assignments (username + role) with remove button. Add form: select user + select review role → "Assign Review Role". Available roles: `asset_manager`, `head_am`, `president`, `cco`, `ceo`. A user can hold multiple review roles.

### 5. Ownership & Partnerships
Ownership tree visualization and relationship data.

### 6. Waterfall Setup
View, edit, and create waterfall structures for any entity. Vue: `WaterfallSetupView.vue`. Flask: `waterfall_service.py`.
- **Entity Navigation** — Selectbox of all entities with waterfalls + entities from relationships.
- **Waterfall Editor** — Editable table for CF_WF and Cap_WF steps. Columns: iOrder, PropCode, vState, FXRate, nPercent, mAmount, vtranstype, vAmtType, vNotes.
- **Validation** — Inline warnings/errors: FXRate sums, Operating Capital Add vs Tag, Pref FX=1.0, lead/tag pairing, AMFee/Promote vNotes requirements.
- **New Waterfall** — Pre-fills template from relationships/accounting: Pref steps per investor, Initial steps for Cap_WF, residual Share+Tag.
- **Actions** — Save to Database (with audit trail + cache invalidation via `refresh_table` + `clear_cache`), Reset to Saved, Copy CF_WF->Cap_WF, Export CSV, Preview Waterfall ($100k test).
- **Guidance Panel** — Collapsible reference from `waterfall_setup_rules.txt`.

### Sidebar: Data Management Tools
Vue: `AppSidebar.vue` — tools are organized under the Data Management section dropdown. Flask: `data.py` API endpoints.
- **Import CSVs** (under Database Tools) — Browser file upload (no server-side folder scan — incompatible with Azure). Select CSV files → auto-matches filenames to table definitions → shows importable/protected/unmatched status → uploads one file at a time (sequential to avoid OOM on 2GB container) with progress indicator. Protected tables (`capital_calls`, `waterfalls`, `one_pager_comments`, `waterfall_audit`, `review_roles`, `review_submissions`, `review_notes`, `one_pager_snapshots`, `prospective_loans`, `prospective_loans_audit`, `planned_loans`, `sale_overrides`, `parcel_sales`, `user_requests`, `user_request_messages`) are never overwritten. Uses chunked import (`import_csv_stream()`, 50K rows/chunk, `dtype=object`) for large files like ISBS (800K+ rows). Clears data and computation caches.
- **Export Database** (under Database Tools) — Export all tables as `waterfall_db_export_{timestamp}.zip` containing `{table_name}_db_export.csv` for every table.
- **Reload Data** (under Data Management) — Reloads all cached data from the database.
- **Feedback & Requests** — Standalone expandable section below nav. Submit errors, improvements, report requests, and analysis requests. Submit form (type, title, description, priority) + scrollable list of past requests with status badges and message counts. Click a request to see its full threaded conversation and add replies. Auto-opens when navigating with a `?reply=TOKEN` query param (from email links).
- **Logout Button** — Full-width button at bottom of sidebar showing username + role. Clears auth store and redirects to login page.

### Data Explorer
Full-page database table browser. Vue: `DataExplorerView.vue`. Flask: `GET /api/data/tables/<table_name>/rows`.
- **Table List** — Left sidebar listing all database tables. Click to load.
- **Data Grid** — Sortable columns (click header), per-column text filter, pagination (configurable page size).
- **Column Visibility** — Checkbox dropdown to show/hide columns.
- **URL State** — Selected table tracked via `?table=` query param.
- **API** — Parameterized SQL with `sa.text()` for safety. Supports `page`, `page_size`, `sort`, `order`, `filter__<col>` query params.

### 10. Feedback & Request Tracking
Embedded request tracking system for users to report errors, suggest improvements, and request reports or analysis. Flask: `feedback.py` + `feedback_service.py`. Vue: sidebar section in `AppSidebar.vue`.
- **Request Types**: `error`, `improvement`, `report`, `analysis`. Priorities: `low`, `medium`, `high`.
- **Statuses**: `open` → `in_progress` → `resolved` / `closed`. Initiators can self-resolve via "Mark Resolved" button; admins can set any status.
- **Database Tables**: `user_requests` (id, user_id, username, request_type, title, description, priority, status, page_context, deal_context, reply_token, created_at, updated_at, resolved_at), `user_request_messages` (id, request_id, sender_type, sender_name, message, sent_via, created_at). Both in `PROTECTED_TABLES`.
- **Threaded Messages**: Each request has a conversation thread (user submissions, admin responses, system status changes, email replies). `sender_type`: `user`, `admin`, `system`. `sent_via`: `app`, `email`.
- **Email Communication**: Admin sends email to user via `POST /<id>/email` (SendGrid). Email includes "View & Reply" button with unique `reply_token` URL → opens app with sidebar auto-focused on that request. Reply token is per-request, generated at creation.
- **Inbound Email Webhook**: `POST /api/feedback/inbound-email` — SendGrid Inbound Parse endpoint. Extracts reply token from `to` address (`requests+TOKEN@domain.com`), strips quoted text, stores reply in thread. Requires DNS MX record setup for full email reply flow.
- **Design Session Export**: `GET /api/feedback/export` (admin) — returns all requests with full message threads for consumption during Claude design sessions.
- **AI Assistant Integration**: `get_user_feedback` tool allows the embedded Claude assistant to query all feedback requests, filterable by status and type.
- **API Endpoints** (`/api/feedback`): `POST /` (submit), `GET /` (list — admin sees all, users see own), `GET /<id>` (detail with thread), `POST /<id>/messages` (add reply), `POST /<id>/resolve` (initiator: mark own request resolved), `GET /reply/<token>` (lookup by email token), `PUT /<id>/status` (admin: change status), `POST /<id>/email` (admin: email user), `GET /export` (admin: all requests for design sessions), `POST /inbound-email` (webhook).
- **Page Context**: Automatically captures current route path when submitting, available for debugging context.

### User Authentication
- **JWT-based**: Login returns access token, stored in Pinia auth store, sent via Axios interceptor
- **Roles**: `admin`, `analyst`, `viewer` — role-gated endpoints via `@role_required()` decorator
- **Password Reset**: `ForgotPasswordView.vue` → email with reset token → `ResetPasswordView.vue`. Uses `flask_app/auth/email_utils.py` via SendGrid
- **Welcome Emails**: Admin creates user → sends welcome email with temporary password, login link, and desktop shortcut installer button
- **Forced Password Change**: Users with `must_change_password` flag are redirected to change password on login
- **Email**: SendGrid Web API v3 (`requests` library, no SMTP). Configured via `SENDGRID_API_KEY` and `SENDGRID_FROM` env vars. Single Sender Verification on `jbruin@peaceablestreet.com`.
- **Desktop Shortcut Installer**: One-click `.bat` download via `GET /auth/shortcut/install`. Downloads `waterfall_xirr.ico` from `GET /auth/shortcut/icon`, creates "Waterfall XIRR" desktop shortcut with custom icon. Uses base64-encoded PowerShell (`-EncodedCommand`) wrapped in a `.bat` file for non-technical users. `curl.exe` (built into Windows 10+) for corporate proxy compatibility. Green "Download Desktop Shortcut Setup" button included in welcome email.

### 7. Reports
Multi-report section with sidebar layout. Vue: `ReportsView.vue`. Flask: `reports.py` + `reports_service.py`.
- **Layout**: Left sidebar (report list + shared filters + Generate button) | Right main area (results table + Excel download). Report definitions are a registry array (`reportDefs`) in the Vue component — adding a new report requires one array entry + backend endpoints.
- **Custom View Reports**: Reports with `isCustomView: true` in `reportDefs` render their standalone Vue component (with `embedded` prop) instead of the standard filter/table layout. Currently: Sold Portfolio (`SoldPortfolioView`), PSCKOC (`PsckocView`), Portfolio Analysis (`PortfolioAnalysisView`).
- **Shared Filters**: Population (Current Deal, Select Deals, By Partner, All Deals). By Partner uses upstream investors via ownership chain (same as Review Tracking — excludes OP/PPI entities). Report-specific filters (e.g. As of Date) are conditionally shown per report definition. Filters are hidden for custom view reports.
- **API Endpoints**: `GET /api/reports/deal-lookup` (eligible deals), `GET /api/reports/partners` (upstream investors).

#### Report: Projected Returns Summary
- **Endpoint**: `POST /api/reports/projected-returns` (JSON), `POST /api/reports/projected-returns/excel`
- **Data Source**: Waterfall analysis via `get_cached_deal_result()` — projected returns through sale
- **Output**: Partner-level rows (Contributions, CF Distributions, Capital Distributions, IRR, ROE, MOIC) plus bold deal-level total row with solid top border
- **Excel**: Formatted workbook via openpyxl (currency/pct/multiple formats, auto-width, deal-total rows bold with top border)

#### Report: ROE Summary
- **Endpoint**: `POST /api/reports/roe-summary` (JSON), `POST /api/reports/roe-summary/excel`
- **Filter**: As of Date (defaults to today)
- **Data Source**: Actual accounting data through the report date (same formula as One Pager ROE to Date). Accrued Pref computed directly from accounting history + waterfall pref rates via `_compute_accrued_pref()` in `reports_service.py` (daily accrual at waterfall rate, year-end compounding with 45-day grace, TypeID 1019 pref payments reduce balance).
- **Output**: One row per deal — Total Funded, Return of Capital, Current Balance, Wtd Avg Balance, CF Received, Accrued Pref, ITD ROE, U/W ITD ROE
- **ROE Formula**: `(Total CF Distributions / Weighted Average Capital) / Years`. CF distributions = operating only (excludes Return of Capital, Realized Gain, and Acquisition Fee). Capital balance reduced by capital returns only, not CF distributions. Uses `calculate_roe_detailed()` in `metrics.py`.
- **Single-deal detail view**: When exactly one deal is selected, displays event-by-event weighted capital calculation table below the summary row with metric cards (Total Funded, Return of Capital, Current Balance, Wtd Avg Balance, CF Received, Days, Years, ITD ROE). Toggle button switches between ITD ROE (actual, default) and U/W ITD ROE (uses ONLY ISBS Projected IS data: 7073 for capital events with sign convention, 7071 for distributions — no actual accounting data). Detail data comes from `_detail_rows` and `_uw_detail_rows` in the API response.
- **Excel**: Formatted workbook (currency/pct formats, auto-width) with per-deal detail sheets

#### Report: Pref Balance Detail
- **Endpoint**: `POST /api/reports/pref-balance-detail` (JSON), `POST /api/reports/pref-balance-detail/excel`
- **Additional Endpoints**: `GET /api/reports/pref-balance-detail/investors/<vcode>` (PE investors for deal)
- **Filter**: Single deal + single investor selector, As of Date (defaults to today)
- **Data Source**: Actual accounting data through report date + generated quarter-end accrual markers
- **Pref Rate Priority**: `deal_terms.pe_coupon` (authoritative) > waterfall Pref step matching investor > waterfall any Pref rate
- **Day Count**: Act/Act — `days_in_year(event_date.year)`, no cross-year boundary splitting (matches Excel)
- **Calculation**: Row-by-row replication of Excel PE_Pref_Balances formulas: `Inv+Comp = prior(InvBal) + prior(CompPref)`, `CurrDue = Inv+Comp × rate / diy × days`, `AccrPref = prior(Remaining)`, `TotalDue = CurrDue + AccrPref`, `Remaining = max(0, TotalDue - PrefPaid)`
- **Compounding**: At 12/31 all remaining unpaid pref compounds (`CompPref = Remaining`). Mid-year, payments exceeding current accrual reduce CompPref.
- **Same-date ordering**: Contributions → Pref Return → Return of Capital → Excess CF → other → Generated (quarter-end markers)
- **Pref payment detection**: TypeID 1019 or Typename containing "Preferred Return"/"Pref Return"; excess CF via TypeID 1020 or "Excess Cash"
- **Output**: Header (investment balance, accrued pref, total, annual pref estimate, pref rate) + transaction detail rows with 13 columns (InvestmentID, InvestorID, Date, Amount, Typename, Investment Balance, Compounded Pref, Inv+Comp, Days, Current Due, Accrued Pref, Total Due, Pref Paid, Remaining Accrual)
- **Excel**: Formatted workbook with header section + detail table (currency formats, auto-width)
- **Validation**: 55/120 investor/deal pairs match Excel exactly; remaining differences are data vintage (different accounting exports), not calculation logic

### 8. Sold Portfolio
Historical returns for sold deals computed from accounting_feed (no forecast waterfalls). Accessed via Reports section (embedded as custom view). Vue: `SoldPortfolioView.vue` (accepts `embedded` prop). Flask: `sold_portfolio.py` + `sold_service.py`.
- **Data Source**: Accounting history only — contributions (`is_contribution`), distributions (`is_distribution`). Raw `acct` is normalised via `normalize_accounting_feed()` on first use.
- **Pref Equity Only**: Filters out OP partners (`InvestorID` starting with "OP"). Case-insensitive InvestorID grouping handles mixed-case entity IDs.
- **Capital event identification**: Uses **Typename** from accounting (not the `Capital` flag, which is unreliable for sale events). Events with Typename containing "Return of Capital" or "Realized Gain" (case-insensitive) are treated as capital events. Used in both gross and net ROE calculations to correctly separate operating income from capital activity.
- **ROE definition**: Annualized operating cash yield on invested capital. ROE numerator = CF distributions (operating income) only. Capital events (sale/refi proceeds including pref paid at sale and gain on sale) are **excluded** from the ROE numerator — they only affect the weighted average capital denominator. A development deal sold before stabilization with no operating distributions has 0% ROE.
- **Summary Table**: Inline `<table>` (not DataTable) with column group headers. One row per deal + bold Portfolio Total row. Gross columns: Investment Name, Acquisition Date, Sale Date, Total Contributions, Total Distributions, IRR, ROE, MOIC. Net columns (visible after computation): Net IRR, Net ROE, Net MOIC. Visual divider (4px border column) separates Gross and Net sections.
- **Deal Detail Drill-Down**: Selectbox to pick a deal → expander with every pref equity accounting row sorted by date. Columns: Date, InvestorID, MajorType, Typename, Capital, Amount, Cashflow (XIRR), Capital Balance (running). IRR/ROE/MOIC metric cards below. Download Activity Detail exports the table + summary metrics to Excel for independent return verification.
- **Net Returns**: Theoretical investor net-of-fees returns computed via simplified waterfall. User inputs assumptions (Ownership %, AM Fee %, Hurdle Rate %, Promote %, Annual Expenses) in a horizontal panel, clicks "Compute Net Returns". Results merged into summary table alongside gross columns.
  - **Per-deal waterfall** (`compute_net_waterfall_for_deal()`): Walks accounting events chronologically. Contributions scaled by ownership %. Distributions: deduct AM fee (on capital balance) and expenses, accrue pref at hurdle rate (Act/365 simple), pay accrued pref first, return capital (capital events only), split excess by promote %. Fees capped at distribution amount (prorated if AM Fee + Expenses > Scaled Distribution).
  - **Promote logic** (hybrid xnpv + pref):
    - **Capital events** (sale/refi): xnpv hurdle test. `hurdle_amount = -XNPV(hurdle_rate, prior_net_cfs) × (1+hurdle)^years`. Promote = `MAX(0, total_distribution - hurdle_amount) × promote_pct`. Promote is deducted from total payout (can reduce pref/capital portions).
    - **CF distributions**: Promote = `Excess × promote_pct`. Pref paid IS the hurdle return (accrued at hurdle rate on outstanding capital), so promote only applies to the profit portion after pref.
    - No state tracking needed — capital events use xnpv independently, CF distributions use the waterfall's pref mechanism.
  - **Acquisition Fees**: Typename "Acquisition Fee" — entire distribution consumed by fee, investor gets $0, pref still accrues. $0 cashflow appended for XIRR alignment with Excel.
  - **Net ROE**: Only "CF Distribution" events count as operating income. Capital event proceeds (including pref paid at sale and gain on sale) go entirely to capital_events — they reduce weighted avg capital but are NOT operating income. Net capital returned is capped at Net to Investor (promote reduces what investor actually gets back).
  - **Portfolio-level expenses**: Dynamic scaling based on active deal count. `Annual Expenses = (3 + 0.25 × active_deals) × per_deal_annual_expenses`. Active deals = deals with capital balance > 0. As deals sell, their proportion drops out. Per-deal sheets use flat per-deal expenses; portfolio-level uses the scaled formula.
  - **Portfolio total**: Full portfolio-level waterfall recomputation — AM fee on portfolio capital, pref accrual on portfolio capital, dynamic expenses with active deal tracking. All net cashflows pooled chronologically; IRR/ROE/MOIC computed from the combined pool. Portfolio-level promote uses same xnpv hurdle logic.
  - **Assumptions footnote**: Displayed below summary table when net returns are shown.
- **Excel Exports**:
  - Summary workbook (`sold_portfolio_returns.xlsx`) — gross returns only
  - Per-deal activity detail (`sold_activity_{name}.xlsx`) — accounting rows + metrics
  - **Net Returns workbook** (`sold_portfolio_net_returns.xlsx`) — multi-sheet:
    - Sheet 1 "Summary": Gross + Net side by side with assumptions footnote. Net columns reference per-deal detail sheets via cross-sheet formulas (IRR, ROE, MOIC, Contributions, Distributions); Portfolio Total IRR references Portfolio Detail sheet.
    - Per-deal sheets: Full waterfall detail with **Excel formulas** (not static values). Editable assumptions block in row 2 (B2=Ownership%, D2=AM Fee%, F2=Hurdle%, H2=Promote%, J2=Annual Expenses) — changing any assumption recalculates the entire sheet. 18 formula columns (A-R): Date, Event, Gross, Ownership%, Scaled, AcqFeePaid, AMFee, Expenses, Available, PrefAccrued, PrefPaid, CapReturned, Excess, **Hurdle Amount** (`=MAX(0,-XNPV(hurdle,prior_net_cfs)*(1+hurdle)^years)`), Promote (hybrid IF: capital events use hurdle, CF uses Excess), NetToInvestor, CapBalance, PrefBalance. Summary metrics: XIRR, MOIC, ROE (Python-computed), SUMPRODUCT for contributions/distributions.
    - "Portfolio Detail" sheet: All deals' events pooled chronologically. **Fully formula-driven** (19 cols A-S). Editable assumptions (B2=AM Fee%, D2=Hurdle%, F2=Promote%, H2=Annual Expenses per deal). Column S = `# Active Deals` (Python-computed from per-deal capital balances). Expense formula: `=(3+0.25×ActiveDeals)×AnnualExp×days/365`. AM Fee, Pref Accrued, Pref Paid, Capital Returned, Excess all calculated from portfolio-level Capital Balance and Pref Balance. XNPV-based Hurdle Amount and hybrid Promote formulas. Summary metrics: XIRR, MOIC, ROE.
- **API Endpoints** (`/api/sold-portfolio`): `GET /summary`, `GET /summary/excel`, `GET /detail/<vcode>`, `GET /detail/<vcode>/excel`, `POST /net-returns` (JSON), `POST /net-returns/excel` (multi-sheet workbook). Net endpoints accept `{ownership_pct, am_fee_pct, hurdle_rate, promote_pct, annual_expenses}` as decimals.
- **Acquisition Fees included**: Unlike the Deal Analysis waterfall path, Sold Portfolio includes all accounting entries (including Acquisition Fees) in its gross return calculations.

### 9. PSCKOC
Upstream waterfall analysis for the PSCKOC holding entity, showing how deal-level distributions flow through PPI entities to PSCKOC members. Accessed via Reports section (embedded as custom view). Vue: `PsckocView.vue` (accepts `embedded` prop). Flask: `psckoc_service.py`.
- **Members**: PSC1 (GP co-invest, Capital Units), KCREIT (LP, Capital Units), PCBLE (GP promote + AM fee recipient, Carry Units)
- **Deal Discovery**: Hybrid approach — recursive downward traversal from PSCKOC through ownership tree (`node.investments`) to find all intermediate entities, then filters to deals whose waterfall PropCode references one of those entities or PSCKOC itself. Prevents false matches from shared holding entities. Excludes sold deals (`Sale_Status=SOLD` or `Lifecycle=Sold`) and filters ended relationships (`EndDate`).
- **Computation**: Button-gated. Runs `get_cached_deal_result()` per deal + `run_recursive_upstream_waterfalls()` for CF and Cap. Results cached in `st.session_state['_psckoc_results']`.
- **Partner Returns**: KPI cards (IRR, ROE, MOIC) per member + styled summary table with deal-level totals.
- **Income Schedule**: PSCKOC's projected income by period and source deal (CF vs Cap).
- **Waterfall Allocations**: Allocation tables showing how income is distributed among PSC1/KCREIT/PCBLE.
- **AM Fee Schedule**: Quarterly AM fee amounts (date, KCREIT balance, fee amount) per Section 6.02.
- **XIRR Cash Flows**: Combined cashflow table per member (contributions + distributions).
- **Excel Export**: 4-sheet workbook (Partner Returns, Income Schedule, AM Fee Schedule, XIRR Cash Flows).
- **New Waterfall vStates** (in `waterfall.py`):
  - `AMFee`: Post-distribution fee deducted from source investor (vNotes), paid to recipient (PropCode). Pool-neutral — does NOT reduce remaining cash pool. `nPercent` = annual rate as percentage (e.g. 0.95 = 0.95%; raw value divided by 100 since `nPercent_dec` conversion is wrong for rates < 1.0). `mAmount` = periods/yr. Allocation rows display `amfee_actual` (the computed fee) for visibility, while `allocated` stays 0 for pool math. In upstream waterfalls, capped at one fee per quarter via `amt_quarterly_tracker` — prevents over-counting when multiple deal distributions trigger the same entity waterfall. Supports investment exclusions via `vNotes` syntax: `SOURCE_PC;exclude:IID1,IID2` — subtracts entity's capital in excluded investments (scaled by source investor's ownership %) from the fee base. Pre-computed via `build_amfee_exclusions()` from accounting data, threaded as `amfee_exclusions` dict through `run_waterfall()`, `run_upstream_waterfall_period()`, and `run_recursive_upstream_waterfalls()`.
  - `Promote`: Cumulative catch-up. `FXRate` = carry share, `nPercent` = target carry %. `vNotes` = comma-separated capital investors. Math: `E >= target/(1-target) * P`.
  - `Amt`: Fixed-amount distribution. `mAmount` = dollar amount per quarter. Reduces the remaining cash pool (unlike AMFee). Use for entity expenses off the top before waterfall splits. In upstream waterfalls, capped at `mAmount` per quarter via `amt_quarterly_tracker` — prevents over-counting when multiple deal distributions trigger the same entity waterfall in the same quarter (e.g., TGA22 $12,500/quarter for audit/tax prep).
  - `IRR`: IRR-targeted hurdle gate. `nPercent` = target IRR. Computes additional distribution needed for investor to reach target IRR using full cashflow history (net of AM fees and expenses). Used in Cap_WF to gate promote.
- **New InvestorState Fields** (`models.py`): `promote_base` (cumulative pref for catch-up denominator), `promote_carry` (cumulative carry from catch-up).
- **Waterfall Setup Guide** (`waterfall_setup_rules.txt`): Comprehensive modeling reference. Sections: vState vocabulary, Add vs Tag, pool routing, operating capital, FXRate, mAmount, deal patterns (A-E), expenses, promote/IRR structure, AMFee exclusions, TGA22 JV example, checklist, troubleshooting.

### 10a. Portfolio Analysis
Upstream waterfall analysis for any portfolio entity (generalized version of PSCKOC). Accessed via Reports section (embedded as custom view). Vue: `PortfolioAnalysisView.vue` (accepts `embedded` prop). Flask: `portfolio_analysis.py`.
- **Entity Selection** — Dropdown of portfolio entities from ownership tree.
- **Mode** — Actual or Proposed (with editable assumptions: AM fee, hurdle, promote, expenses).
- **Computation** — Button-gated. Runs deal-level waterfalls + recursive upstream waterfalls for the selected entity.
- **Output** — Partner returns, deal detail drill-down, investor-level metrics.

### 10b. Valuations
Annual valuation cycle: records → sign-off → committee approval → publish. Vue:
`ValuationsView.vue`. Flask: `valuations.py` + `valuation_service.py` (360 routes as of
`v440`). Documented Sep 11 2026 — this section had been missing entirely.

- **Cycle / record list** — filter by status, class, text. A record is one deal in one
  cycle, carrying the appraiser's Argus import (`argus_import_id`) once linked.
- **Assumptions tab** — entered valuation assumptions, and the linked Argus import.
- **Budget Review tab** — the comparison this module exists for:
  **Estimate | Budget | Valuation**, ~27 category rows from `config.IS_ACCOUNTS`, then
  Interest / Principal / Total Debt Service / DSCR and the below-the-line block.
  - *Estimate* = actuals through the last reported month + budget for the rest of the
    cycle year. *Budget* = next year's budget (`isbs_budget_is` + the app's supplement).
    *Valuation* = year 1 of the linked Argus forecast.
  - **Line mapping panels** (`v440`) — two tabs on one shared component,
    `LineMappingPanel.vue`: **Load Partner Budget** and **Review Argus Coding**. Upload →
    per-line category dropdown (ranked by the deal's own usage) → account within it
    (defaulted to the deal's most-used) → flip toggle → reconciliation panel
    (stated vs computed revenue, expenses, NOI) → warnings → commit. Budget writes
    `isbs_budget_is_supplements`; Argus writes COA overrides on its import. Committing
    rebuilds the comparison above.
  - **Debt service is modeled, not read from the files** (`v440`) — see the CLAUDE.md
    section. Budget and Valuation columns only; the panel note says so, names the loan
    count, and flags variable-rate loans.
  - Occupancy trend strip + analyst commentary.
- **Balance Sheet tab** — prior year end vs latest reported, with a note when the
  requested as-of month is not yet in ISBS.
- **Q&A / AI tabs** — appraisal upload and `claude-sonnet-4-6` extraction, reconciled
  against the entered assumptions. Completeness validation (`_missing_sections`, one
  targeted re-ask) added Sep 11 2026; an incomplete extraction says so on the panel.
- **NAV tab** — see `valuation_nav_module.md`; largely unbuilt (`open_items.md` §5.1).
- **Approval** — `president` / `ceo` / `cio` seats, per-cycle scope. An admin can
  **approve on behalf of** an outstanding seat; the vote records who it was cast for
  (amber chip on the UI). Publish writes `valuations` / `mri_val`.

### 11. New Business
Deal pipeline, lease due diligence, and deal evaluation workspace under the "New Business" sidebar section.

#### 11a. Pipeline
Vue: `PipelineView.vue`. Flask: `prospects.py` + `prospect_service.py`.
- **Deal Pipeline** — Kanban board + table view. Stages: Lead → Screening → LOI → DD → IC Review → Closing → Closed / Passed. Fields: deal name, location, asset type, GLA/units, partner, purchase price, assigned to, target close.
- **Deal Detail** — Right panel with properties, entities, activity log, and analysis tabs. Delete deal button at bottom.
- **Quick Deal Evaluator** — Assumptions form (acquisition, debt, equity structure, partnership terms, NOI, exit) → instant computed returns using existing engines. Results: PSC IRR/ROE/MOIC, Investor IRR/ROE/MOIC, property-level returns, annual summary table, capital stack visualization, sensitivity matrix.
- **Scenarios** — Multiple saved assumption sets per deal (Base Case, Downside, different hold periods). Side-by-side comparison view.
- **Engine reuse** — `build_prospect_analysis()` creates synthetic data structures from form inputs, then calls `compute_deal_analysis()` with the same waterfall/XIRR/ROE engines used by Deal Analysis. Accepts optional `waterfall_df` to use real DB waterfalls instead of synthetic `_build_waterfall()`.
- **Onboard to Portfolio** — One-click wizard converts a closed prospect to a portfolio deal (creates inv, waterfalls, loans, forecast entries). No re-keying.
- **Database tables** — `prospect_deals`, `prospect_properties`, `prospect_entities`, `prospect_investors`, `prospect_assumptions`, `prospect_cashflows`, `prospect_activity` (all in `PROTECTED_TABLES`).
- **CRUD endpoints** — Full REST: `GET/POST/PUT/DELETE /api/prospects`, `/api/prospects/<id>/properties`, `/api/prospects/<id>/entities`, `/api/prospects/<id>/investors`, `/api/prospects/<id>/assumptions`.
- **Waterfall endpoints** — `GET /<id>/waterfall` (retrieve steps), `POST /<id>/waterfall/build` (generate from investor inputs), `DELETE /<id>/waterfall` (clear).

#### 11a-0. Scenario Analysis
Named scenarios on Prospect Deal Analysis, selected from a dropdown above the results panel. A scenario binds: a cash flow source (pin an Argus import per property via `argus_import_ids`, else the normal cascade), assumption overrides (JSON overlay on `prospect_assumptions`), and income adjustments (`{label, start_date, end_date?, revenue: {acct: annual $}, expense: {...}}` — positive removes, negative adds back, applied to `mAmount_norm` by `apply_scenario_adjustments()` in `prospect_analysis.py`). Table `prospect_scenarios` (PROTECTED_TABLES). API: `GET/POST /api/prospects/<id>/scenarios`, `PUT/DELETE .../scenarios/<sid>`, `GET .../scenarios/risk-candidates` (seeds downsides from the linked lease review via `lease_reviews.prospect_property_id` — termination options, cotenancy dependents, material leases; stale termination dates fall back to lease end). `POST /analyze` accepts `scenario_id`. **Argus imports surface as scenarios automatically**: `ensure_import_scenarios()` in `scenario_service.py` runs when the scenario list loads (idempotent) — any import no scenario pins gets one named after the import, pinning it for its property plus the deal's active imports for the others; the first on a deal with no base becomes the Base Case when it pins the active import. Equity split honours declared `prospect_investors.planned_investor_id` records (ownership %, commitments) over waterfall-shape inference. Full plan: `.claude/memory/scenario_analysis.md`.

#### 11a-1. Prospect Deal Analysis
Standalone route at `/prospect-analysis`. Vue: `ProspectAnalysisView.vue`. Flask: endpoints in `prospects.py`, engine in `prospect_analysis.py`.
Full deal analysis view for New Business, mimicking the Asset Management Deal Analysis page with shared computation engines for consistent returns across the company.
- **Layout** — Left setup panel (420px) + right results panel. Setup panel: deal info, capital budget, operating assumptions, debt parameters, waterfall builder, action buttons.
- **Capital Budget** — Sources & Uses builder:
  - **Uses**: 13 default line items (Purchase Price, Sponsor Acq Fee, Loan Fees, Lender Orig Fee, Debt Broker Fee, Sponsor DD/Legal, Title, Sponsor Misc, Cap Ex Reserve, Prepaid Expenses, PSC Orig Fee, PSC DD Costs, Working Capital). Items support fixed $ or % of purchase_price / total_debt basis. Add/remove custom items. PSC Origination Fee auto-calculated (read-only).
  - **Sources**: Debt sources (First Mortgage, Future Fundings, Second Mortgage, Future 2nd Fundings) + computed equity gap split by PE/OP percentage (default 90/10).
  - **Persistence**: Serialized as JSON in `prospect_assumptions` table (`capital_uses_json`, `capital_sources_json`). Restored on deal load by merging saved amounts into default structure (forward-compatible with new items).
- **Assumption Fields** — 37 fields in `ASSUMPTION_FIELDS` list including loan terms (lender, rate_type, rate_index, rate_spread_bps, rate_cushion_bps), extension terms (count, months, conditions), prepay (type, schedule), sizing constraints (max_ltv, max_ltc, min_dscr, dscr_test_start, min_debt_yield, origination_fee_bps), and notes (earnout_notes, guarantor_notes). Dynamic SELECT/INSERT/UPDATE queries built from field list.
- **Waterfall Builder** — Two-tab interface:
  - **Builder tab**: Flexible step rows (Entity, Step Type, Rate/Amount). Step types: Preferred Return (rate %), Return of Capital, Cash Flow Split (share %), Fixed Amount ($/quarter). Add/remove steps + new entity. Share % validation warning when != 100%.
  - **Steps tab**: Preview of stored CF_WF and Cap_WF in separate bordered cards with descriptions ("Operating distributions — does NOT reduce capital outstanding" / "Refi / sale proceeds — DOES reduce capital outstanding"). Colored step badges (Pref/Initial/Share/Tag). Explanation panel describing Share/Tag simultaneous split.
- **Waterfall Generation** — `POST /api/prospects/<id>/waterfall/build` accepts `{investors, promote}`. Generates:
  - CF_WF: Pref steps (per investor with pref_rate > 0) → Share (lead) + Tag (followers) for residual split
  - Cap_WF: Pref steps → Initial steps (capital return per investor) → Share + Tag for residual
  - Saves to `waterfalls` table via `save_waterfall_steps()` with audit trail
- **Analysis Flow** — Loads real waterfalls from DB before computation. Falls back to synthetic `_build_waterfall()` if none stored. Uses same `compute_deal_analysis()` engine as AM. **Compute Returns reads the stored waterfall and never writes it** — only the explicit "Build & Save Waterfall" button replaces stored steps, and `_storedToInputs()` hydrates the Builder losslessly (iOrder as the Tie #, lead-first within a tie) so a rebuild round-trips identically.
- **Timing** — Close date: `target_close`, else the cash-flow source start (first Argus/uploaded month), else Jan 1 of the default year. Equity contributions seed at `close_date - 1 day` with `actuals_through` moving with them, so pref accrues from close, not a synthetic prior year-end. Sale = month-end preceding the hold anniversary (10/1/2026 + 5yr → 9/30/2031, exactly hold×12 months).
- **Capital Budget as source of truth** — `compute_capital_budget()` in `prospect_analysis.py` mirrors the app's Sources & Uses arithmetic (pct-based items, grossed-up PSC fee); the engine takes total cost/equity from it, so partner contributions tie to the on-screen Sources. The CapEx Reserve use seeds the cash schedule via `beginning_cash_override`.
- **Operating overrides on every source** — `mgmt_fee_pct` (replaces 5040 with gross revenue × pct) and `replacement_reserve_psf` (monthly 5092 = SF × $/SF / 12) apply to Argus forecasts via `_apply_operating_overrides_df()`, not just the uploaded/growth modes. The anniversary table shows the reserve as its own "Replacement Reserve" line under Management Fee (5092 split out of R&M on the NB table only).
- **Annual Forecast** — anniversary years from the close date; the extra column is "Terminal NOI": the forward 12 months the exit value is capped on, sourced from the un-truncated forecast so it ties to `sale_dbg.NOI_12m_After_Sale` exactly. Parcel-sale rent shows as "Less: {parcel name}" directly below Rental Income (backed out of the group lines from `result['parcel_revenue_detail']`; totals unchanged).
- **Audits** — ROE/MOIC audit dropdowns at full AM parity (per-event Pref Due/Paid/Accrued + ITD ROE, daily accrual at the deal's waterfall pref rates loaded by real vcode; deal level pref-free). XIRR Cash Flows sorted chronologically.
- **Audit Excel** — "Download Audit Excel" builds the workbook from the same computation via `_run_prospect_analysis()`/`_continue_analyze()`: Summary, Assumptions (term-sheet waterfall narrative, exit calc), Capital Budget (ties to the app), anniversary Annual Forecast, Debt Service, Cash Management, Waterfall Steps, Partner Cash Flows with live =XIRR formulas (`flask_app/services/prospect_excel.py`).
- **Results** — Sources & Uses summary, deal-level KPI cards (IRR, ROE, MOIC, sale price), Partner Returns table (PE partners highlighted), expandable Annual Forecast and Debt Service tables, Diagnostics expander.
- **Vcode convention** — Prospect deals use `N{deal_id:07d}` vcodes. Waterfalls stored with this vcode in the shared `waterfalls` table.
- **Cashflow Status API** — `GET /api/prospects/<id>/cashflow-status` returns per-property status with Argus/Excel badges and timestamps (batch endpoint, replaces per-property polling).

#### 11a-2. PPI Ownership Stack
The relationships between Peaceable and its investors upstream of the PE vehicle. Full plan + status: `.claude/memory/ppi_ownership_waterfalls.md`.
- **Declaration** — "PPI Ownership Stack" panel on Prospect Deal Analysis (below Waterfall Structure): PE vehicle (defaults to the deal-level PE participant), relationships (name, entity id — datalist of existing AM entities for JV additions — slice %), terms per relationship (`terms_json`: AM fee % on funded capital net of ROC + frequency, optional CF pref, investor minimum IRR, promote % and share passed to investors), participants (one PSC + one or many investors). Stored in `prospect_entities` (roles `pe_vehicle`/`ppi_relationship`) + `prospect_investors`. Placeholder ids `NR{deal}-{n}` until MRI assigns real ones.
- **Build** — `POST /api/prospects/<id>/ppi-stack/build` (the only write) generates steps into the shared `waterfalls` table following the TGA22 pattern: vehicle pro-rata Share/Tag split into relationship entities (single-relationship stacks sit on the vehicle id); per relationship CF = (pref) → pro-rata → AMFee rows at iOrder 900+ (vNotes = source investor, nPercent raw %, mAmount periods/yr); Cap = (pref) → Initial ROC → investor IRR gate (net of fees; largest LP gates multi-investor relationships, warned) → post-promote Share/Tag (PSC = co×(1−promote) + promote×(1−shared)). Pref/IRR rates as decimals, AMFee raw percent.
- **Computation** — `ppi_upstream_service.build_ppi_results()`: seeds participants at close−1 day (slice×share of the vehicle's actual deal contribution), interleaves the vehicle's CF+Cap allocations chronologically through `run_upstream_waterfall_period` with one shared state dict and one quarterly fee tracker (Cap IRR gates see CF history; fee cap spans both waterfalls). Payload `ppi_waterfalls` on `/analyze`. Engine fix: `state_credited` flag stops terminal recipients being double-credited in state cashflows (PSCKOC/Portfolio Analysis IRR gates became more correct).
- **Linked-JV chains** (a deal joining an existing venture, e.g. Windsor → TGA6): the run loads every stored waterfall plus the real `relationships` rows (active generations only — EndDate rows filtered), bounded by a closure walked downward from the vehicle. An entity's waterfall runs only if it is the vehicle, a declared relationship, or a relationship-passthrough target (a fund under a feeder); a waterfall-bearing entity reached as a distribution child is terminal (PSC1's house waterfall stays out of a deal's view) and fee/promote recipients are terminal (the manager's owners are out of scope). Capital cascades down the ownership tree at the active percentages (JV → feeder → fund → investors) so fee bases and IRRs exist at every level; a linked relationship seeds once at the entity (no double count). "Build & Save" never regenerates a linked entity's stored waterfall. Reference case with exact tie-outs: TGA6 in `.claude/memory/ppi_ownership_waterfalls.md`.
- **Display** — "PPI Ownership Waterfalls" dropdown between Annual Operating Forecast and Debt Service: PSC summary cards (fees/co-invest/distributions/IRR); participant table net of fees where **non-PSC fund investors pool into one "Fund Investors (n)" line** (pooled dated cashflows, so the group IRR is exact; PSC1, the outside JV partner and the manager stay individual); an **annual pivot** matching the operating forecast — anniversary-year columns (sale month-end clamped into the final hold year for mid-month closes), rows = Waterfall Step | Recipient | CF/Capital grouped per waterfall-bearing entity, AM Fee lines aggregated across sources; collapsible quarterly fee schedules per relationship. Audit Excel gains a "PPI Ownership" tab.
- **Migration at close** — `POST /api/prospects/<id>/ppi-stack/migrate` (admin; "Migrate at Close" in the panel): `id_map` re-keys placeholders across waterfall vcodes/PropCodes/AMFee vNotes + prospect rows; a final id that already carries a waterfall (existing JV) keeps its waterfall and the placeholder steps are removed (reported); writes ownership rows into `relationships` (a bridge — that table is MRI-refreshed) so the AM tree/Portfolio Analysis pick the stack up.
- **API**: `GET/PUT /ppi-stack`, `POST /ppi-stack/build`, `GET /ppi-stack/steps`, `POST /ppi-stack/migrate`, `GET /api/prospects/ppi-entities`.

#### 11b. Lease Review
Standalone route at `/lease-review`. Vue: `LeaseReviewView.vue`. Flask: `lease_review.py` + `lease_review_service.py`.
Commercial lease due diligence workflow with 7-step stepper UI.
- **Step 1: Setup** — Select or create a lease review (property name, review name, asset type).
- **Step 2: Import Rent Roll** — Upload seller's rent roll (Excel/CSV/PDF). Two modes: **Import (Merge)** — non-destructive, fuzzy-matches tenants by `(suite, tenant_name)`, updates fields without touching extraction data; **Replace All** — destructive full reset. Merge function: `merge_rent_roll_to_review()`.
- **Step 3: Upload Documents** — Multi-file PDF upload with SHA-256 hash dedup. Two buttons: **Select Files** (individual PDFs) and **Select Folder** (entire directory tree via `webkitdirectory`). Subfolder names sent as `folder_hints` for tenant matching — if files are in `Starbucks/Original Lease.pdf`, "Starbucks" is matched to tenants before falling back to filename matching. Function: `upload_documents_to_review()`, matching: `_match_file_to_tenant()`.
- **Step 4: AI Extraction** — Run Claude extraction on pending documents. Pulls rent steps, cotenancy clauses, exclusive use, options, key dates. Dedup on re-runs: checks `(tenant_id, effective_date)` for rent steps, `(tenant_id, source_doc)` for cotenancy/options.
- **Step 5: Validation** — Three-way comparison: seller rent roll vs lease extraction vs Argus (if provided). Summary cards (match/mismatch/pending counts) + per-tenant comparison table. Function: `validate_rent_roll()`.
- **Step 6: Analyst Review** — Per-tenant Approve/Flag/Reset buttons. Blocks completion until all non-vacant tenants approved. Function: `approve_tenant()`.
- **Step 7: Complete** — Excel download + summary charts.
- **Workflow persistence** — `workflow_step` and `step_data` columns on `lease_reviews`. Progress endpoint: `GET /api/lease-review/reviews/<id>/progress`.
- **Database tables** — `lease_reviews`, `lease_tenants`, `lease_documents`, `lease_rent_steps`, `lease_cotenancy`, `lease_cotenancy_refs`, `lease_exclusive_use`, `lease_options`, `lease_validation`, `lease_field_resolutions` (all in `PROTECTED_TABLES`).
- **Seed endpoint** — `POST /api/lease-review/seed` — bulk data import for portability (admin only).

#### 11c. Lease Risk Analysis
Standalone route at `/lease-risk-analysis`. Vue: `LeaseRiskAnalysisView.vue`. Flask: endpoints in `lease_review.py`, service functions in `lease_review_service.py`.
Analysis view using analyst-resolved data. Defaults to base data when no resolution exists; uses analyst's concluded value when a discrepancy has been resolved.
- **Field Resolution** — `lease_field_resolutions` table with `UNIQUE(tenant_id, field_name)`. Resolvable fields: `square_feet`, `annual_rent`, `monthly_rent`, `rent_per_sf`, `lease_start`, `lease_end`, `security_deposit`. UPSERT pattern (PG `ON CONFLICT`, SQLite `INSERT OR REPLACE`). Functions: `resolve_field()`, `clear_resolution()`, `get_resolved_tenants()`.
- **7 Tabs**:
  - **Overview** — KPI summary cards (total tenants, GLA, annual rent, avg rent/SF, co-tenancy count, resolution count) + tenant roster with inline field editing (double-click to override, "R" badge marks resolved fields, revert button).
  - **Lease Expirations** — Dual-axis bar chart (expiring SF + % of total rent by year) + yearly table + material lease detail per expiration year with co-tenancy implications.
  - **Validation** — Match/mismatch/pending summary + per-tenant validation details with one-click "Use Seller" or "Use Lease" buttons on mismatches.
  - **Co-Tenancy Risk** — Horizontal bar chart of rent at risk by named co-tenant + clause detail table + rent-at-risk summary.
  - **Scenario Analysis** — Expandable departure scenario cards showing cascading impacts if a named co-tenant departs.
  - **Exclusive Use** — Restriction table (tenant, suite, restricted use, restriction text).
  - **Options** — Renewal/termination options table (type, term, notice period, deadline, rent terms, auto-renewal).
- **API Endpoints**: `GET /api/lease-review/reviews/<id>/risk-analysis` (complete data bundle), `PUT /api/lease-review/reviews/<id>/tenants/<tid>/resolve` (resolve field), `DELETE /api/lease-review/reviews/<id>/tenants/<tid>/resolve/<field>` (clear resolution).

### 11d. Cash Flow Imports (Argus + Generic Excel)
Property-level cash flow import hub supporting two sources: Argus Enterprise Excel exports and generic partner Excel/CSV models. Both roll up to deal-level for waterfall analysis.

#### Pipeline Workflow
1. Deal comes in → analyst creates deal + adds properties
2. Per property → load cash flows via property card buttons:
   - **"Import Argus"** → Argus Excel export → `argus_cashflows` (vcode = `NP{property_id:06d}`)
   - **"Upload Cash Flows"** → partner Excel/CSV → `prospect_cashflows` (with `property_id` FK)
3. Deal Analysis tab → "Run Analysis" with cascade: Argus > Excel cashflows > NOI growth assumptions
4. Test waterfall structures → iterate → quote term sheet
5. Term sheet accepted → move to DD/verification

#### Argus Parser (`argus_parser.py`)
- Stateless, no DB/Flask deps. 56 keyword-to-COA mappings
- Three parsers: `parse_monthly_cashflow()`, `parse_rent_roll_summary()`, `parse_revenue_assumptions()`
- `cashflow_to_forecast_df()` converts to compute-compatible DataFrame
- COA keyword matching: Revenue → 4010/4030/4075/4090-92, Expense → 5020/5040/5060/5090/5110, CapEx → 7050
- Unmapped items shown in UI for manual assignment

#### Generic Parser (`cashflow_parser.py`)
- Stateless, no DB/Flask deps. Auto-detects columns via regex patterns
- **Two layout modes**: Vertical (standard columns) and Horizontal (dates across columns, line items down rows)
- Horizontal detection: `_detect_horizontal_dates()` scans first 20 rows for date-like values across columns (60%+ threshold). Row label matching via `_ROW_REVENUE_PATTERNS`, `_ROW_EXPENSE_PATTERNS`, `_ROW_NOI_PATTERNS`, `_ROW_CAPEX_PATTERNS`. Skip patterns exclude false positives (recovery/reimbursement rows).
- Handles annual and monthly data (annual auto-spread to 12 monthly rows)
- Normalizes signs, derives missing columns (NOI from rev-exp, or rev/exp from NOI)
- Handles dollar signs, commas, parenthetical negatives, messy header rows

#### Service Layer
- **`argus_service.py`**: Import with SHA-256 dedup, projection CRUD, forecast generation, COA override, NB→AM migration. `get_property_rollup_forecast_df()` aggregates active Argus forecasts from multiple properties into one deal-level forecast.
- **`prospect_service.py`**: `import_property_cashflows()`, `get_property_cashflows()`, `delete_property_cashflows()`, `get_deal_cashflows_by_property()`.

#### Projection Toggle (Asset Management)
- Dropdown on Deal Analysis switches between Default and Argus projections
- `projection_id` in compute cache key; `get_cached_deal_result()` substitutes `fc` DataFrame

#### Vue Components
- `ArgusImport.vue` — shared upload component (file zones, COA mapping, tenant preview). Used in Deal Analysis modal and Pipeline property cards.
- Pipeline property cards — "Import Argus" and "Upload Cash Flows" buttons per property. Green "CF" badge when data loaded.

#### Database Tables
- 5 Argus tables (all in `PROTECTED_TABLES`): `argus_imports`, `argus_cashflows`, `argus_tenants`, `argus_rent_steps`, `argus_market_profiles`
- `prospect_cashflows` (existing) — now used for property-level Excel imports with `property_id` FK and `source` column

#### API Endpoints
- **Argus** (`/api/argus`): 11 endpoints for upload, CRUD, forecast preview, COA mapping, migration
- **Cashflows** (`/api/prospects`): `POST /<deal_id>/properties/<prop_id>/cashflows/upload`, `GET .../cashflows`, `DELETE .../cashflows`
- **Analysis cascade** in `POST /<deal_id>/analyze`: checks Argus property rollup → prospect_cashflows → NOI growth assumptions

## AI Assistant

Embedded Claude-powered chat panel for natural-language queries against the portfolio database.

### Architecture
- **Backend**: `flask_app/services/assistant_service.py` (tools + agentic loop), `flask_app/api/assistant.py` (SSE endpoint + history CRUD)
- **Frontend**: `vue_app/src/components/common/AiAssistant.vue` (floating chat panel in App.vue)
- **Model**: Claude Sonnet 4.6 (`claude-sonnet-4-6`), 4096 max tokens, streaming SSE
- **Activation**: `ANTHROPIC_API_KEY` env var (`.env` for local, container env var for Azure)
- **Agentic loop**: Up to 10 tool iterations per query via `chat_completion()`

### Tools (20)

| Tool | Description | Data Source |
|------|-------------|-------------|
| `resolve_deal` | Fuzzy name → vcode matching | `inv` DataFrame |
| `list_deals` | Browse deals with status filter | `inv` DataFrame |
| `query_deal_data` | Deal metadata by vcode | `inv` DataFrame |
| `query_accounting` | Contributions/distributions | `acct` DataFrame |
| `query_database` | Ad-hoc read-only SQL (SELECT only) | SQLAlchemy engine |
| `get_portfolio_summary` | Portfolio-level KPIs | `inv` DataFrame |
| `compute_deal_returns` | Full waterfall IRR/ROE/MOIC + sale proceeds | `get_cached_deal_result()` |
| `get_loan_details` | Loan metadata (rate, maturity, lender) | `loans` DataFrame |
| `get_occupancy` | Occupancy data by deal | `occ` DataFrame |
| `get_financial_statement` | ISBS income statement (actual/budget/UW) | `isbs_raw` DataFrame |
| `get_waterfall_structure` | Waterfall allocation rules | `wf` DataFrame |
| `get_one_pager` | One Pager investor report (cap stack, perf, PE) | `financials_service` |
| `get_annual_forecast` | Annual projections (NOI, DSCR, FAD by year) | `annual_aggregation_table()` |
| `get_sold_returns` | Sold deal returns from accounting history | `sold_service` |
| `get_capitalization` | Cap stack, debt, LTV, PE exposure | `get_deal_capitalization()` |
| `compare_deals` | Side-by-side 2-10 deal comparison | multi-deal compute + cap |
| `get_debt_service` | Loan summary + annual amortization schedule | compute result `loan_sched` |
| `get_cash_management` | Cash schedule (reserves, CapEx, distributable) | compute result `cash_schedule` |
| `get_tenant_roster` | Tenant list, occupancy, lease maturity rollover | `financials_service` |
| `get_user_feedback` | User feedback requests with threads (for design sessions) | `feedback_service` |

### Features
- **Page context awareness**: Sends current page, selected deal vcode/name, and quarter to backend. System prompt includes pre-loaded deal metadata so assistant infers context from user's current view.
- **Suggested questions**: Context-aware clickable chips based on current page (Dashboard, Deal Analysis, One Pager, Property Financials, Sold Portfolio) shown on first open.
- **Conversation persistence**: `chat_history` table (user_id PK, messages JSON). Auto-loads on open, auto-saves after each response, clear button deletes server-side. Survives page refresh.
- **Result truncation**: Large results include numeric column summaries (sum/min/max) and truncation notes with guidance.
- **Smarter error messages**: `_error_hint()` maps common errors to actionable suggestions (missing deal → "use resolve_deal", no waterfall → "check get_waterfall_structure").
- **Safety**: `query_database` enforces SELECT-only (blocks DROP/DELETE/UPDATE/INSERT/ALTER). Max 500 rows per query. All tools wrapped in try/except with error logging.

### API Endpoints
- `POST /api/assistant/chat` — Streaming SSE chat (login_required)
- `GET /api/assistant/status` — Check if API key is configured
- `GET /api/assistant/history` — Load saved chat history for current user
- `PUT /api/assistant/history` — Save chat history
- `DELETE /api/assistant/history` — Clear chat history

### SSE Event Types
- `text_delta` — Incremental text from Claude
- `tool_use` — Tool call in progress (name + input shown as chip)
- `done` — Response complete
- `error` — Error message
