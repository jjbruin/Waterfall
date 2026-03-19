# Session Handoff: Flask API + Vue 3 SPA — Waterfall XIRR

**Date**: 2026-03-19 (updated)
**Status**: Application fully operational on Flask + Vue. Streamlit removed. Azure deployment blocked on IT.

---

## Current Status

| Area | Status | Notes |
|------|--------|-------|
| Flask API | **COMPLETE** | 105 routes across 10 blueprints, JWT auth, role-based access |
| Vue 3 SPA | **COMPLETE** | 10+ routes, Pinia stores, ECharts, AG Grid, responsive layout |
| Streamlit | **REMOVED** | 9 UI files deleted (8,799 lines). Zero `import streamlit` remaining |
| Core Engine | **CLEAN** | 14+ Python modules with no UI dependency (compute, waterfall, metrics, etc.) |
| Actuals Through | **COMPLETE** | Global cutoff setting, dynamic year defaults, threaded through all layers |
| Excel Downloads | **COMPLETE** | Per-section buttons + full 7-sheet Deal Analysis workbook |
| Capital Calls | **COMPLETE** | CRUD endpoints, mixed date handling, refi shortfall auto-clear, table auto-creation |
| Balloon Payoff | **COMPLETE** | Detection, forecast exclusion, pre-balloon balance deduction from sale proceeds |
| Prospective Loans | **COMPLETE** | Refi sizing with refi_date, sale date extension, balloon-aware net proceeds |
| Desktop Launcher | **COMPLETE** | `launch_app.bat` + custom icon, starts both servers + opens browser |
| Responsive Layout | **COMPLETE** | Sidebar collapse updates main content width, views fill available screen |
| Review Workflow | **COMPLETE** | Sequential approval pipeline (Draft → Head AM → President → CCO → CEO → Approved) |
| Multi-User Auth | **COMPLETE** | Role-based (viewer/analyst/admin), user management, SSO ready |
| PostgreSQL | **READY** | Migration script, SQLAlchemy abstraction, data adapters |
| Azure Deployment | **BLOCKED** | Dockerfile, Bicep infra, CI/CD pipeline built — waiting on IT for Azure subscription |

---

## Architecture

```
waterfall-xirr/
├── config.py                    # Constants, account classifications, dynamic defaults
├── compute.py                   # Deal computation engine (core logic, no UI dependency)
├── one_pager.py                 # One Pager data logic (perf, cap stack, PE metrics, comments)
├── models.py                    # Data classes (InvestorState, Loan)
├── waterfall.py                 # Waterfall calculation engine
├── metrics.py                   # XIRR, XNPV, ROE, MOIC calculations
├── loaders.py                   # Data loading from database/CSV
├── database.py                  # SQLite management, migrations, table definitions, CSV import/export
├── loans.py                     # Debt service modeling
├── planned_loans.py             # Future loan projections
├── capital_calls.py             # Capital call handling
├── cash_management.py           # Cash flow management
├── consolidation.py             # Sub-portfolio aggregation
├── portfolio.py                 # Fund/portfolio aggregation
├── reporting.py                 # Annual aggregation tables, formatting utilities
├── ownership_tree.py            # Investor ownership structures
├── utils.py                     # Helper utilities
├── launch_app.bat               # Desktop launcher (starts Flask + Vue + opens browser)
├── waterfall_xirr.ico           # Custom app icon for desktop shortcut
├── waterfall.db                 # SQLite database (not in git, >100MB)
│
├── flask_app/                   # Flask REST API (105 routes across 10 blueprints)
│   ├── __init__.py              # App factory, SPA serving (static/ in prod, JSON in dev)
│   ├── config.py                # DB_PATH, DATABASE_URL, secrets, defaults, ACTUALS_THROUGH
│   ├── db.py                    # SQLAlchemy engine (SQLite or PostgreSQL via DATABASE_URL)
│   ├── run.py                   # Entry point (threaded=True)
│   ├── serializers.py           # df_to_records(), safe_json() for numpy/pandas types
│   ├── migrate_to_postgres.py   # One-command SQLite → PostgreSQL data migration
│   ├── auth/
│   │   ├── routes.py            # JWT login/logout/me, login_required, role_required, user CRUD
│   │   ├── models.py            # User model via SQLAlchemy (SQLite or PostgreSQL)
│   │   └── sso.py               # OAuth2/OIDC SSO (Azure AD + Okta) via authlib
│   ├── api/                     # 10 blueprint modules (thin route handlers calling services)
│   │   ├── dashboard.py         # 8 routes including SSE /init-stream
│   │   ├── deals.py             # 22 routes (compute, header, forecast, debt, cash, xirr, audits, 7 excel downloads)
│   │   ├── financials.py        # 7 routes (perf chart, IS, BS, tenants, one-pager + chart + comments)
│   │   ├── data.py              # 8 routes (deals, reload, import, export, tables, config GET/PUT, sources)
│   │   ├── waterfall_setup.py   # 6 routes (entities, steps CRUD, validate, preview, copy, export CSV)
│   │   ├── reports.py           # 2 routes (projected returns + excel)
│   │   ├── sold_portfolio.py    # 3 routes (summary, excel, detail)
│   │   ├── ownership.py         # 5 routes (tree, entity tree, investors, requirements, upstream analysis)
│   │   ├── psckoc.py            # 4 routes (deals, compute, results, excel)
│   │   └── reviews.py           # 9 routes (status, submit, approve, return, note, tracking, roles CRUD)
│   └── services/                # Business logic (imports core Python modules directly)
│       ├── data_service.py      # load_all() via adapters, module-level cache, reload()
│       ├── data_adapters.py     # Pluggable per-table adapters (DatabaseAdapter, MriApiAdapter)
│       ├── compute_service.py   # Deal cache, ROE/MOIC audit builders, 8 Excel generators
│       ├── dashboard_service.py # Portfolio caps, KPIs, NOI trend, charts
│       ├── financials_service.py# Perf chart NOI pipeline, IS, BS, tenants, one-pager
│       ├── waterfall_service.py # Validation, CRUD, entity listing
│       ├── reports_service.py   # Partner returns builder, Excel generator
│       ├── sold_service.py      # Sold returns computation, Excel
│       ├── ownership_service.py # Tree building, upstream analysis
│       ├── psckoc_service.py    # Deal discovery, computation, results, Excel
│       └── review_service.py    # Review workflow (approval pipeline, tracking, roles)
│
├── vue_app/                     # Vue 3 SPA (Vite + Pinia + Vue Router + ECharts + AG Grid)
│   ├── src/
│   │   ├── api/client.ts        # Axios with JWT interceptor, 5min timeout
│   │   ├── router/index.ts      # 10+ routes with auth guard
│   │   ├── stores/              # 6 Pinia stores (auth, data, deals, dashboard, waterfall, psckoc)
│   │   ├── views/               # 10+ view components
│   │   │   ├── DashboardView.vue         # KPIs, 5 charts, progress overlay
│   │   │   ├── DealAnalysisView.vue      # All sections with lazy-loading, Excel downloads
│   │   │   ├── PropertyFinancialsView.vue# Chart, IS, BS, tenants
│   │   │   ├── OnePagerView.vue          # Investor report + review panel + print
│   │   │   ├── ReviewTrackingView.vue    # Production pipeline dashboard
│   │   │   ├── WaterfallSetupView.vue    # AG Grid editors with tabs
│   │   │   ├── ReportsView.vue           # Generate + excel download
│   │   │   ├── SoldPortfolioView.vue     # Summary table + drill-down + excel
│   │   │   ├── OwnershipView.vue         # Tree overview, entity explorer, upstream analysis
│   │   │   ├── PsckocView.vue            # Deal portfolio, KPIs, returns, income, allocations, Excel
│   │   │   ├── SettingsView.vue          # User management (admin), review roles, password change
│   │   │   └── LoginView.vue             # Username/password + SSO button
│   │   └── components/
│   │       ├── charts/          # DualAxisChart, StackedBarChart, DonutChart, BarChart
│   │       ├── common/          # KpiCard, DataTable, MetricCards, ProgressOverlay, ToastNotifications
│   │       ├── layout/          # AppSidebar (collapsible, DB tools, report settings), AppHeader
│   │       └── waterfall/       # WaterfallEditor (AG Grid)
│
├── azure/                       # Azure deployment infrastructure
│   ├── main.bicep               # IaC: PostgreSQL Flexible Server, App Service, Container Registry
│   └── deploy.sh                # One-command deployment script
│
├── .github/
│   ├── CODEOWNERS               # PR review enforcement
│   ├── pull_request_template.md # Standardized PR template
│   └── workflows/deploy.yml     # CI/CD: build Docker → push ACR → deploy App Service
│
├── Dockerfile                   # Multi-stage: Node builds Vue → Python serves with gunicorn
├── .dockerignore                # Excludes .venv, node_modules, .db, secrets
├── .env.example                 # Template: SECRET_KEY, JWT, DATABASE_URL, MRI_API, SSO config
└── .gitignore                   # Comprehensive: *.db, .env, .claude/, dist/, static/, credentials
```

---

## Recent Changes (March 2026)

### Desktop Launcher & Responsive Layout (Mar 19)

**Desktop Launcher**:
- `launch_app.bat` starts Flask API + Vue dev server (minimized), then opens browser to `http://localhost:5173`
- Custom `waterfall_xirr.ico` icon (waterfall bar chart design) for desktop shortcut
- Desktop shortcut created at user's Desktop folder

**Responsive Sidebar & Full-Width Views**:
- Sidebar collapse/expand now updates the `--sidebar-width` CSS variable on the document root
- `main-content` margin animates with the sidebar via `transition: margin-left 0.2s`
- Removed `max-width` constraints from Property Financials (was 1200px), Review Tracking (was 900px), and Settings (was 700px)
- One Pager retains 960px `max-width` for print/PDF layout
- All views now fill available screen width regardless of sidebar state

### Capital Calls, Balloon Payoff & Prospective Loans (Mar 18)

**Capital Call CRUD & Data Pipeline**:
- `POST/PUT/DELETE /api/deals/<vcode>/capital-calls` endpoints with full cache invalidation (`data_service.reload()`)
- `capital_calls` table auto-created via `CREATE TABLE IF NOT EXISTS` in `database.py`
- `load_capital_calls()` uses `pd.to_datetime(format='mixed', dayfirst=False)` for mixed CSV/HTML date formats
- Null row filtering (`dropna`) handles empty rows from CSV imports (5,130 → 3 rows)
- Column mapping: `entityid` → `deal_name`, `propcode` → `investor_id`, `calldate` → `call_date`
- Refi shortfall auto-clear: when capital calls cover the gap within $1 tolerance, warning flag is removed

**Balloon Loan Payoff at Sale**:
- Detection: loan schedule last row `ending_balance < $1` (float tolerance), `principal > 0`, prior row balance > 0
- Balloon principal excluded from forecast debt service (not an operating expense)
- Sale proceeds deduct pre-balloon loan balance: `total_loan_balance_at(sale_date) + balloon_total`
- Loan schedule grouped by `["vcode", "LoanID", "event_date"]` to distinguish overlapping loans

**Prospective Loan Refinancing**:
- `size_prospective_loan()` returns `refi_date` in sizing dict for Vue form pre-fill
- Sale date extends to new maturity when `Sale_ME < new_maturity`
- Net sale proceeds: `sale_price - loan_balances - balloon_total + cash_reserves`

### Streamlit Removal (Mar 16)
- Deleted 9 Streamlit UI files: `app.py`, `dashboard_ui.py`, `debt_service_ui.py`, `property_financials_ui.py`, `reports_ui.py`, `sold_portfolio_ui.py`, `psckoc_ui.py`, `waterfall_setup_ui.py`, `one_pager_ui.py`
- Cleaned `compute.py`: removed Streamlit `get_cached_deal_result()` (Flask has its own in `compute_service.py`)
- Cleaned `reporting.py`: removed `import streamlit` and `show_waterfall_matrix()` display function
- Zero `import streamlit` references remain in the codebase

### Actuals Through Cutoff (Mar 16)
- Global `actuals_through` setting: sidebar checkbox + month-end date selector
- **Partner cash flows**: Actual distributions from `accounting_feed` through cutoff; waterfall only post-cutoff
- **Operating forecast**: Rev+Exp rows for months <= cutoff removed from `fc_deal_full`
- **Cache key**: Includes `actuals_through` so toggling triggers recomputation
- Dynamic defaults: `DEFAULT_START_YEAR = date.today().year`, `PRO_YR_BASE_DEFAULT = date.today().year - 1`
- Threaded through: `config.py`, `flask_app/config.py`, `compute.py`, `compute_service.py`, all Flask API endpoints, Vue sidebar + data store

### Deal Analysis Excel Downloads (Mar 16)
- Per-section "Excel" buttons on: Partner Returns, Annual Forecast, Debt Service, Cash Management, Capital Calls, XIRR Cash Flows
- "Download Full Deal Analysis (Excel)" button at top — 7-sheet comprehensive workbook
- 7 new API endpoints at `GET /api/deals/<vcode>/excel/{section}`
- 8 generator functions in `compute_service.py` with shared helpers (`_excel_styles`, `_write_header_row`, `_autosize_columns`)
- Vue `downloadExcel()` uses `fetch` + `Blob` with `Authorization: Bearer` header

### Bug Fixes (Mar 16)
- **Capitalization table all zeros**: `get_deal_capitalization` non-lookup path didn't normalize `Typename` → `TypeName`. Fixed with full column normalization matching `prepare_cap_lookups`.
- **500 error on deal switch**: `_params()` in `deals.py` returns 4 values after adding `actuals_through`, but one call site still unpacked 3. Fixed all unpack sites.
- **Vue config persistence**: `updateConfig` in `data.ts` wasn't including `actuals_through` when rebuilding config from API response. Fixed.

### Annual Forecast Formatting (Mar 16)
- Black border lines under Expenses and Capital Expenditures rows (`underline-row` CSS)
- Black border above Total Distributions (`topline-row` CSS)

### One Pager Review Workflow (Mar 11)
- Sequential approval: Draft → Head AM → President → CCO → CEO → Approved
- `ReviewPanel.vue` with status indicator, approve/return buttons (role-gated), threaded notes
- `ReviewTrackingView.vue` at `/review-tracking` — production pipeline dashboard
- Review role management in Settings (admin only)
- Comments locked when document is in review or approved
- 9 API endpoints at `/api/reviews`

---

## Phase 9: Azure Deployment (BLOCKED — Waiting on IT)

### What's Built
- **Dockerfile**: Multi-stage (Node 20 builds Vue → Python 3.12 slim + gunicorn serves everything)
- **azure/main.bicep**: Infrastructure-as-code for PostgreSQL Flexible Server (B1ms), App Service (B1, Linux container), Container Registry (Basic)
- **azure/deploy.sh**: One-command script — creates resource group, deploys Bicep, builds Docker, pushes to ACR, sets secrets, restarts app
- **.github/workflows/deploy.yml**: CI/CD — on push to main, builds image, pushes to ACR, deploys to App Service
- **SPA serving**: Flask serves Vue's `static/index.html` for all non-API routes in production; Vue Router handles client-side routing
- **Estimated cost**: ~$43/month (App Service B1 $13 + PostgreSQL B1ms $25 + ACR Basic $5)

### What's Blocking
1. **Azure subscription**: `jbruin@peaceablestreet.com` has no Azure subscription — IT needs to assign one
2. **Conditional Access**: Azure AD tenant (`peaceablestreet.com`) blocks CLI refresh tokens (AADSTS530036) — IT needs to whitelist Azure CLI or create a service principal
3. **WSL2 restart**: WSL2 installed but requires machine restart for Docker Desktop to function

### Deployment Steps (After IT Access Granted + Restart)
```bash
# 1. Login to Azure
az login

# 2. Deploy infrastructure + container (creates everything)
cd C:\Users\jbruin\Documents\GitHub\waterfall-xirr
PG_ADMIN_PASSWORD='YourSecurePassword123!' bash azure/deploy.sh

# 3. Migrate data from local SQLite to Azure PostgreSQL
DATABASE_URL="postgresql://pgadmin:YourPass@waterfall-xirr-pg.postgres.database.azure.com:5432/waterfall_xirr?sslmode=require" \
    python -m flask_app.migrate_to_postgres

# 4. Open the URL printed by deploy.sh
# 5. Login as admin/admin, change password immediately in Settings
```

### CI/CD Setup (After First Deploy)
Add 6 secrets to GitHub repo Settings → Secrets:
- `AZURE_CREDENTIALS` — from `az ad sp create-for-rbac --sdk-auth`
- `ACR_LOGIN_SERVER` — e.g. `waterfallxirracr.azurecr.io`
- `ACR_USERNAME` / `ACR_PASSWORD` — from ACR admin credentials
- `AZURE_WEBAPP_NAME` — e.g. `waterfall-xirr`
- `AZURE_RG` — e.g. `waterfall-xirr-rg`

Then every push to `main` auto-deploys.

---

## Installed Prerequisites

| Tool | Version | Status |
|------|---------|--------|
| Azure CLI | 2.84.0 | Installed, needs `az login` after IT grants subscription |
| Docker Desktop | 29.2.1 | Installed, needs restart (WSL2 dependency) |
| WSL2 | 2.6.3 | Installed, needs restart to enable VM platform |
| Node.js | (existing) | Used for Vue build in Dockerfile |
| Python 3.12 | (existing) | Used in .venv and Dockerfile |

---

## Running Locally (Development)

**Desktop shortcut** (easiest): Double-click **"Waterfall XIRR"** on the desktop — starts both servers and opens browser.

**Or manually**:
```bash
# Flask API (from repo root)
cd waterfall-xirr
.venv\Scripts\activate
python -m flask_app.run
# → http://localhost:5000, 105 routes, login: admin/admin

# Vue SPA (from vue_app/, separate terminal)
cd vue_app
npm run dev
# → http://localhost:5173, proxies /api → :5000, /auth → :5000
```

---

## Key Technical Patterns

- **Lazy-loading**: Vue sections fetch data only when expanded (toggle function in DealAnalysisView)
- **Module-level caching**: Flask services cache in plain dicts (`_deal_cache` in compute_service.py)
- **SSE for long operations**: Dashboard init uses Server-Sent Events with token-in-query-param auth
- **Role-based access**: `@login_required` + `@role_required("admin", "analyst")` decorator stacking
- **Database abstraction**: SQLAlchemy engine auto-selects SQLite or PostgreSQL from config
- **Data adapter pattern**: Each table can independently source from database or API
- **SSO optional**: Disabled by default, one env var (`SSO_CLIENT_ID`) enables it
- **SPA serving**: Flask serves `static/index.html` for non-API routes in production
- **Excel downloads**: openpyxl bytes served via `send_file(BytesIO(bytes))`, fetch + Blob on client
- **Actuals through**: Global cutoff threads through config → compute → cache key → API → Vue sidebar
- **Review workflow**: Sequential approval with role-gated actions, comment lockout during review
- **Capital call dates**: `format='mixed'` handles CSV US dates ("6/30/2026") and HTML ISO dates ("2026-06-01")
- **Cache invalidation**: Capital call CRUD uses `data_service.reload()` (full clear), not `refresh_table()` (single table patch)
- **Balloon detection**: Float tolerance (`< $1`) instead of exact zero comparison for loan ending balances
- **Refi shortfall tolerance**: $1 threshold for floating point rounding in capital call coverage checks
- **Responsive sidebar**: Toggle updates `--sidebar-width` CSS variable on `:root`; `main-content` transitions smoothly

---

## What IT Needs To Provide

1. **Azure subscription** — Pay-As-Go or Dev/Test (~$43/month)
2. **Azure CLI access** — Whitelist "Microsoft Azure CLI" in Conditional Access policy (error `AADSTS530036`), or provide a service principal with Contributor role on the subscription
3. **SSO app registration** (optional) — If using Azure AD SSO: register an app in Azure AD portal, provide Client ID + Client Secret + Tenant ID
