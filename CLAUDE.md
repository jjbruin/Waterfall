# Waterfall XIRR - Multi-Layer Waterfall Model

## Shared Memory

Shared project memory files are in `.claude/memory/`. Read `MEMORY.md` there at the start of every conversation for project context, conventions, and architecture notes. Update these files as you work — they are committed to git and shared across all developers.

**Date-stamp anything describing work in flight, and delete it when the work ships.**
A section that says a branch is unmerged reads as authoritative for as long as it sits
here, and nothing distinguishes it from a current one. On Sep 11 2026 this file still
carried a 69-line section declaring `feat/onepager-chart-window` "not merged, not
deployed" — it had been on main for over a month (`window_end_quarter` at
`financials_service.py:177`). Shipped work belongs in the deploy history, not in
standing instructions.

## Project Overview

A Flask + Vue financial modeling application for calculating investment waterfalls, XIRR, and related performance metrics for real estate investments. The application supports multi-layer distribution waterfalls with preferred returns, capital accounts, and investor-level tracking.

## Tech Stack

- **Python 3.x** with virtual environment (`.venv/`)
- **Flask** - REST API backend (`flask_app/`)
- **Vue 3 + Vite** - Modern frontend (`vue_app/`)
- **pandas/numpy** - Data manipulation
- **scipy** - XIRR/NPV calculations (Newton-Raphson primary, Brent's method fallback)
- **ECharts** - Interactive charts (Vue)
- **PostgreSQL** - Azure-hosted database (local dev: SQLite via `waterfall.db`)
- **SQLAlchemy** - Database abstraction (`flask_app/db.py`), switches via `DATABASE_URL` env var
- **JWT** - Authentication (Flask + Vue)
- **Docker** - Multi-stage build (Vue → Python 3.12-slim + Gunicorn)
- **Azure Container Apps** - Production hosting

## Project Structure

```
waterfall-xirr/
├── config.py                 # Constants, account classifications, rates, dynamic defaults
├── compute.py                # Deal computation logic (core engine)
├── one_pager.py              # One Pager data logic (general info, cap stack, property perf, PE metrics, comments)
├── models.py                 # Data classes (InvestorState, Loan)
├── waterfall.py              # Waterfall calculation engine
├── metrics.py                # XIRR, XNPV, ROE, MOIC calculations
├── loaders.py                # Data loading from database/CSV
├── database.py               # Database management (SQLite + PostgreSQL), migrations, CSV import/export
├── loans.py                  # Debt service modeling
├── planned_loans.py          # Future loan projections
├── capital_calls.py          # Capital call handling
├── cash_management.py        # Cash flow management
├── consolidation.py          # Sub-portfolio aggregation
├── portfolio.py              # Fund/portfolio aggregation
├── reporting.py              # Annual aggregation tables, formatting utilities
├── ownership_tree.py         # Investor ownership structures
├── utils.py                  # Helper utilities
├── argus_parser.py           # Stateless Argus Enterprise Excel parser (COA mapping, forecast conversion)
├── cashflow_parser.py        # Generic Excel/CSV parser for partner cash flow models (auto-detect columns, annual→monthly)
├── Dockerfile                # Multi-stage Docker build (Vue + Flask + Gunicorn)
├── launch_app.bat            # Desktop launcher (opens Azure app in browser)
├── waterfall_xirr.ico        # Custom app icon for desktop shortcut
├── waterfall.db              # SQLite database for local dev (not in git, >100MB)
│
├── flask_app/                # Flask REST API backend
│   ├── __init__.py           # App factory (create_app), SPA serving with cache headers
│   ├── run.py                # Dev server entry point
│   ├── db.py                 # SQLAlchemy engine management (SQLite/PostgreSQL)
│   ├── config.py             # Flask configuration (DATABASE_URL, dynamic defaults, ACTUALS_THROUGH)
│   ├── extensions.py         # Flask extensions
│   ├── serializers.py        # JSON serialization helpers (NumpyEncoder, safe_json)
│   ├── auth/                 # JWT authentication (login, SSO config, password reset, welcome emails)
│   │   ├── routes.py         # Auth routes (login, users, desktop shortcut installer)
│   │   └── email_utils.py    # SendGrid email sending (welcome emails, password reset)
│   ├── api/                  # API blueprints
│   │   ├── dashboard.py      # Dashboard endpoints (KPIs, charts, SSE init-stream)
│   │   ├── data.py           # Data endpoints (deals, upload-import, export, config)
│   │   ├── deals.py          # Deal analysis endpoints + Excel downloads
│   │   ├── financials.py     # Property Financials + One Pager endpoints
│   │   ├── reports.py        # Report generation endpoints
│   │   ├── reviews.py        # Review workflow endpoints (status, submit, approve, return, tracking, roles)
│   │   ├── feedback.py       # Feedback & request tracking endpoints (submit, list, messages, email, webhook)
│   │   ├── lease_review.py   # Lease review & risk analysis endpoints (DD workflow, document upload, field resolution)
│   │   ├── prospects.py      # Pipeline prospect CRUD (deals, properties, entities, investors, assumptions)
│   │   ├── argus.py          # Argus Enterprise import, projection management, COA mapping, forecast preview
│   │   └── ...               # Additional route blueprints
│   └── services/             # Business logic (reuses compute.py, database.py, etc.)
│       ├── dashboard_service.py  # KPI calculations, NOI pipeline, chart data
│       ├── data_service.py       # Data loading and caching
│       ├── data_adapters.py      # Pluggable data source adapters (DB or MRI API)
│       ├── compute_service.py    # Deal computation cache, ROE/MOIC audit builders, Excel generators
│       ├── review_service.py     # Review workflow business logic (approval pipeline)
│       ├── financials_service.py # Property Financials + One Pager data aggregation
│       ├── feedback_service.py   # Feedback & request tracking (CRUD, email, export)
│       ├── reports_service.py    # Report builders (projected returns, ROE summary, pref balance detail)
│       ├── lease_review_service.py  # Lease review DD workflow, document upload, extraction, field resolution
│       ├── prospect_service.py      # Pipeline prospect CRUD, lease review creation, deal evaluation
│       ├── argus_service.py         # Argus Enterprise import, projection CRUD, forecast generation, NB→AM migration
│       └── ...
│
├── scripts/                  # Azure migration and setup scripts
│   ├── migrate_to_postgres.py    # Bulk SQLite → PostgreSQL migration
│   ├── fix_tables.py             # Fix tables with type mismatches
│   └── azure-complete-setup.sh   # Reference doc of provisioned infrastructure
│
└── vue_app/                  # Vue 3 + Vite frontend
    ├── src/
    │   ├── api/client.ts     # Axios instance with JWT interceptors
    │   ├── stores/           # Pinia stores (auth, data, dashboard, deals)
    │   ├── views/            # Page components (DashboardView, DealAnalysisView, OnePagerView, DataExplorerView, ForgotPasswordView, ResetPasswordView, etc.)
    │   └── components/       # Shared components (KpiCard, DataTable, ReviewPanel, AppSidebar)
    ├── vite.config.ts        # Vite config (proxies /api to Flask)
    └── package.json
```

## Documentation

- **DOCUMENTATION.md** - Complete project documentation (setup, data files, concepts, troubleshooting)
- **waterfall_setup_rules.txt** - Waterfall step configuration guide for deal modeling team
- **typename_rules.txt** - Capital pool routing rules based on Typename field
- **.claude/memory/app_reference.md** - What each app tab displays + AI Assistant tools/endpoints (split out of this file Sep 11 2026)

## Running the Application

### Production (Azure)
**Desktop shortcut**: Double-click **"Waterfall XIRR"** on the desktop — opens the Azure app in the browser.
- **URL**: https://app-waterfall-dev-v2.icyplant-026fb2db.eastus.azurecontainerapps.io
- Login: real user accounts. `admin` / `admin` is the LOCAL dev seed only — it
  returns 401 against Azure (verified Sep 2 2026). Do not script against it.

### Deploying Changes

**BEFORE DEPLOYING ANY COMMIT: review it for symptom repair, and tell Jim first.**
(Jim's standing instruction, Sep 1 2026, after `50695d9` shipped an override that
zeroed correct data on 12 deals.)

Fix problems, not symptoms. Read the diff and ask what the commit is actually doing:

- Does it **suppress, zero, blank, force or special-case** a value rather than correct
  the computation that produced it? A value overridden downstream is still computed
  wrong upstream, and every other consumer keeps reading the wrong one.
- Is the **trigger a proxy** for the thing it claims to detect? (`50695d9` keyed off
  the presence of a `2015-12-31` row in MRI's export — an export artifact — to infer
  "this deal has no baseline".)
- Does the stated rationale **hold for every deal it touches**? Enumerate the affected
  rows from live data and check. `50695d9`'s rationale held for 2 of the 12 it acted on.
- Does it use a **sentinel value** where the answer is "unknown"? Return `None`, never
  `0` — a `0` that means "no data" is indistinguishable from a real zero to every
  consumer, and only renders as a dash because `fmtMil()` happens to treat `0` as `—`.
- Is it a **per-deal hardcode** (a vcode in a constant)? Those are always symptom
  repairs, however well documented.

Verifying that a commit does what its message says is NOT the same as verifying the
premise. Both are required. When a commit is a symptom repair — even a well-reasoned,
well-documented one — say so to Jim, with the affected deals and figures, and get his
call BEFORE building the image. Deploy is not the place to discover the question.

All deploys use Azure CLI (GitHub Actions secrets are not configured).

**Tag every image with the commit SHA it was built from, and deploy that tag — never `:latest`.**
`:latest` is mutable, so a revision pointing at it cannot be traced back to a commit once the
next build overwrites the tag. Deploy the SHA tag and the running revision names its own source.

```bash
# 0. Build from a known commit — the ACR build uploads the local working tree, not a git ref
git rev-parse --short HEAD          # confirm you are on the commit you intend to ship
git status --porcelain              # must be empty, or the image contains uncommitted work

# 1. Build in ACR, tagged with the commit SHA (--no-logs avoids a unicode crash)
SHA=$(git rev-parse --short HEAD)
az acr build --registry acrwaterfalldev -g rg-waterfall-dev --image waterfall-xirr:$SHA --image waterfall-xirr:latest --no-logs .

# 2. Lock the SHA tag so a later build cannot overwrite it (delete stays enabled for cleanup)
az acr repository update -n acrwaterfalldev --image waterfall-xirr:$SHA --write-enabled false

# 3. Deploy the SHA tag (incrementing suffix forces a new revision)
az containerapp update -g rg-waterfall-dev -n app-waterfall-dev-v2 --image acrwaterfalldev.azurecr.io/waterfall-xirr:$SHA --revision-suffix v350
```

Pick the revision suffix by bumping the current one — reusing a suffix is rejected. This same
query answers "what commit is live?", since the image tag is the SHA:
```bash
az containerapp revision list -g rg-waterfall-dev -n app-waterfall-dev-v2 --query "[?properties.active].{name:name,image:properties.template.containers[0].image}" -o table
```

**Notes**:
- ACR build agent has transient failures (5-second runs) — retry if it fails. Confirm the run actually built rather than failing fast: `az acr task show-run --registry acrwaterfalldev --run-id <id> --query "{status:status,start:startTime,finish:finishTime}" -o tsv`
- Use `--no-logs` to avoid Azure CLI unicode crash (`✓` character).
- To pin an already-deployed `:latest` revision after the fact, retag its digest without rebuilding and redeploy that tag — same digest, so the content is provably identical: `az acr import -n acrwaterfalldev --source acrwaterfalldev.azurecr.io/waterfall-xirr@sha256:<digest> --image waterfall-xirr:<sha>`
- **Deploy history (SHA-pinned)** — newest first; `vNNN` is the revision suffix and the
  backticked SHA is the commit its image was built from, so the running revision names its
  own source. Entries marked † have a full post-mortem archived verbatim in
  `.claude/memory/deploy_history.md`. **Read that before assuming a revision shipped what
  its SHA suggests** — several did not (`v424` was a merge, not the commit that was asked
  for; `v378` was superseded minutes later; `v418`/`v417` shipped only part of a branch).

  - `v429` = `33a4bf5`
  - `v428` = `dfc38df` †
  - `v427` = `bf093c2` †
  - `v426` = `9f0708d` †
  - `v425` = `54fe25a` †
  - `v424` = `3a21c3c` †
  - `v423` = `eb34396` †
  - `v422` = `2ea1186` †
  - `v421` = `c8c367b` †
  - `v420` = `d1259bb` †
  - `v419` = `a7cb4c4` †
  - `v418` = `dcefc29` †
  - `v417` = `6dac97f` †
  - `v416` = `eb7520d` †
  - `v415` = `639f023` †
  - `v414` = `bd001e8` †
  - `v413` = `7d21769` †
  - `v412` = `da354a3` †
  - `v411` = `8534916` †
  - `v410` = `019b592` †
  - `v409` = `7dc7bd8` †
  - `v408` = `e5ef21d` †
  - `v407` = `50695d9` †
  - `v406` = `5e5a3b7` †
  - `v405` = `7827c6f` †
  - `v404` = `864e834` †
  - `v403` = `edf4a3a` †
  - `v402` = `1495d61` †
  - `v401` = `9432f87` †
  - `v400` = `b8bdfea` †
  - `v399` = `bf29707` †
  - `v398` = `ffd19b1` †
  - `v397` = `c6083f0` †
  - `v396` = `557af5e`
  - `v395` = `726f708`
  - `v394` = `228f440`
  - `v393` = `791cbef`
  - `v392` = `3264771`
  - `v391` = `1c68a37`
  - `v390` = `635f0ff`
  - `v389` = `f20e195`
  - `v388` = `6f3d4e7`
  - `v387` = `f0b8d00`
  - `v386` = `dfccb3f`
  - `v385` = `e15333c`
  - `v384` = `d847410`
  - `v383` = `4ecc15b`
  - `v382` = `30c3833`
  - `v381` = `9987dba`
  - `v380` = `5b314e0`
  - `v379` = `1759185`
  - `v378` = `3cb8bb0` †
  - `v377` = `b113806`
  - `v376` = `e1e94f7` †
  - `v375` = `c5d0b81`
  - `v374` = `f051cf3`
  - `v373` = `6bbfbf6`
  - `v372` = `93e3a19`
  - `v371` = `644af7d`
  - `v370` = `0f9918c`
  - `v369` = `a31f3ec`
  - `v368` = `b3e1bc1`
  - `v367` = `515d5b1`
  - `v366` = `c571d53`
  - `v365` = `35fc12f`
  - `v364` = `e22dd84`
  - `v363` = `be08b30`
  - `v362` = `0290d6b`
  - `v361` = `770363b`
  - `v360` = `54951b8`
  - `v359` = `0070ffd`
  - `v358` = `feea43d`
  - `v357` = `eb02954`
  - `v356` = `71068a9`
  - `v355` = `d58a797`
  - `v354` = `0b28731`
  - `v353` = `f754919`
  - `v352` = `6918db3`
  - `v351` = `826df0e`
  - `v350` = `87202b1`
  - `v349` = `2700c99` †

  `v348` and earlier point at `:latest` and are not traceable by tag.

### Local Development
```bash
# Activate virtual environment
.venv\Scripts\activate

# Run Flask API backend
python -m flask_app.run          # API on http://localhost:5000

# Run Vue frontend (separate terminal)
cd vue_app && npm run dev        # Frontend on http://localhost:5173
# Default login: admin / admin
```

### Azure Infrastructure
- **Container App**: app-waterfall-dev-v2 (1 CPU, 2GB RAM, 2 Gunicorn workers)
- **PostgreSQL**: psql-waterfall-dev.postgres.database.azure.com (B1ms, v16)
- **Container Registry**: acrwaterfalldev.azurecr.io
- **Resource Group**: rg-waterfall-dev (eastus)
- View logs: `az containerapp logs show -g rg-waterfall-dev -n app-waterfall-dev-v2 --type console --tail 50`

### Caching
- `index.html` served with `Cache-Control: no-cache` — browser always checks for new version on deploy
- Hashed assets (`/assets/*`) cached for 1 year with `immutable` — Vite generates new hashes on each build

## Key Concepts

### Acquisition Date
- **Derived from accounting feed**: `min(EffectiveDate)` per `InvestmentID` from the accounting table, mapped to vcode via investment map
- **Enriched at load time**: `data_service.py:load_all()` overwrites `Acquisition_Date` in `inv` DataFrame after loading both `inv` and `acct`. Also re-applied in `refresh_table()` when the deals table is refreshed.
- **Date parsing**: Raw `EffectiveDate` in the accounting adapter is strings (e.g., `"11/21/2016 0:00"`). Must parse via `pd.to_datetime()` before `groupby().min()` — string comparison is alphabetical, not chronological.
- **Rationale**: The MRI `Acquisition_Date` field may not reflect the true closing date. The earliest accounting activity (e.g., acquisition fee collected at closing) is the authoritative date. For development deals, first funding may occur months after acquisition (e.g., 3rd Ave & Indian School: acquisition fee 11/21/2016, first funding 5/15/2017).
- **Fallback**: If a deal has no accounting activity, the original MRI `Acquisition_Date` is preserved
- **Consumers**: Deal Analysis metadata, Sold Portfolio summary, One Pager general info — all use `inv["Acquisition_Date"]` automatically

### Sale Date & Anticipated Exit
- **Sale date priority**: (1) Sale date override from UI, (2) `event_dates` projected disposition closing (`vEventType='Disposition', vEvent='Closing', vDateType='Projected'`), (3) horizon end / max loan maturity. The `Sale_Date` column in the deals table is no longer consulted.
- **Underwritten Exit**: From MRI `event_dates` table (`vEventType='Asset Management', vEvent='U/W Exit', vDateType='Actual'`). Latest `dtEvent` wins. Function: `get_underwritten_exit()` in `one_pager.py`. Previously sourced from `inv.Sale_Date` — now sourced directly from `event_dates`.
- **Current Anticipated Exit**: From `event_dates` table (`vEventType='Disposition', vEvent='Closing', vDateType='Projected'`). Latest `dtEvent` wins when multiple rows exist. Used on One Pager and as default sale date for Deal Analysis projections. Function: `get_current_anticipated_exit()` in `one_pager.py`.
- **Shared helper**: Both functions use `_lookup_event_date()` in `one_pager.py` — generic event_dates query by vCode + vEventType + vEvent + vDateType.
- **MRI refresh safety**: `MRI_COLUMNS` in `mri_service.py` lists only columns that MRI refresh may overwrite. `Sale_Date`, `Sale_Status`, `InvestmentID`, and `Portfolio_Name` are explicitly excluded — they are preserved during upsert.
- **Sale date override**: Separate `sale_overrides` table (see Sale Overrides section) — completely independent of MRI refresh.
- **Event dates refresh**: `MRI_Event_Dates` and `MRI_Inspection` queries are included in the "Refresh All Data from MRI" process (`QUERY_REGISTRY` in `mri_service.py`).

### Waterfall Types
- **CF Waterfall**: Operating cash distributions (does NOT reduce capital outstanding)
- **Capital Waterfall**: Refi/sale proceeds (DOES reduce capital outstanding)

### Preferred Returns
- Daily accrual using Act/365 Fixed day count
- Compounds annually on 12/31 (with 45-day grace period)
- Tracked per investor via `InvestorState`

### Capital Calls
- **Data pipeline**: app entry → `capital_calls` table → `load_capital_calls()` → `build_capital_call_schedule()` → `apply_capital_calls_to_states()`
- **`capital_calls` IS IN `PROTECTED_TABLES`** (Sep 10 2026, Jim's instruction). The CSV import runs `to_sql(if_exists="replace")`, which DROPS the table, so a single `MRI_Capital_Calls.csv` upload silently destroyed every call typed into Deal Analysis. **Capital calls are now app-entered only** — added, edited and deleted per row via `/api/deals/<vcode>/raw-capital-calls`. Cost is only the bulk upload: the table is NOT in `mri_service.QUERY_REGISTRY`, so "Refresh All Data from MRI" never touched it and no automated feed is interrupted. The upload UI badges a protected file "locked" and excludes it from the importable count, so a skipped import announces itself. **Known consequence**: rows with a blank `Vcode` (5,127 of 5,130 locally — empty trailing rows from a past CSV) are invisible to the vcode-filtered GET and the replace that used to clear them is now blocked, so they cannot be cleaned from the UI. Harmless to every computation (`load_capital_calls` drops them via `dropna`), but permanent; purging them is a separate deliberate act. Guardrail: `scripts/capital_call_crud_check.py` asserts all four import entry points refuse the table AND that the rows survive, plus that app add/edit/delete still work.
- **Date handling**: Uses `pd.to_datetime(format='mixed', dayfirst=False)` to handle both CSV US dates ("6/30/2026") and HTML ISO dates ("2026-06-01")
- **Null filtering**: `dropna(subset=['deal_name', 'call_date', 'amount'])` removes empty rows from CSV imports
- **Column mapping**: Supports `entityid` → `deal_name`, `propcode` → `investor_id`, `calldate` → `call_date`
- **Table auto-creation**: `database.py` creates `capital_calls` table via `CREATE TABLE IF NOT EXISTS` in `create_additional_tables()`
- **CRUD endpoints**: `POST/PUT/DELETE /api/deals/<vcode>/capital-calls` in `deals.py`. All use `data_service.reload()` for full cache invalidation
- **Refi shortfall auto-clear**: After capital calls are applied, if total capital calls cover the refi shortfall (within $1 tolerance), `refi_capital_call_required` is set to False

### Tax Abatements
- **Account**: `TAX_ABATEMENT_ACCTS = {7070}` in `config.py` — stored in `forecast_feed` / `forecasts` table
- **Sign convention**: MRI stores abatements as negative (credit); `normalize_forecast_signs()` in `loaders.py` forces `base.abs()` (positive) since abatements increase cash flow
- **Annual Forecast**: Displayed as "Tax Abatement" row below NOI, included in FAD calculation. FAD adds back CapEx funded from cash reserves (`reporting.py`)
- **NPV at Sale**: Remaining abatement payments after sale date are discounted to PV at `TAX_ABATEMENT_DISCOUNT_RATE` (5%) and added to net sale proceeds (`compute.py`). Shown as "NPV (@5%) Tax Abatements" in Sale Proceeds Calculation (Debt Service section)
- **Below-the-line items**: Former "Excluded Accounts" renamed to "Other Below-the-Line" (`OTHER_EXCLUDED_ACCTS`): Interest Income (4050), Other Income/Expenses (5220, 5210, 5195, 7065), Partnership Expenses (5120, 5130), Depreciation & Amortization (5160, 5165), Extraordinary Expenses (5400). Other Revenue (4075) and Maintenance Flex (5092) are in NOI.
- **Conditional display**: Tax Abatement NPV line only appears in Sale Proceeds Calculation when deal has 7070 data
- **Comparability fix**: In U/W Projected IS, 7070 appears below NOI as a separate line item, but in actuals the abatement is netted into account 5090 (Real Estate Taxes). To ensure apples-to-apples comparison across all columns:
  - **One Pager** (`one_pager.py`): `calc_amounts()` folds 7070 into expenses (negative credit reduces expenses). `TAX_ABATEMENT: ['7070']` added to `IS_ACCOUNTS`. Separate adjustment for the pre-computed `at_close_noi_df` path.
  - **Property Financials** (`config.py`): `'7070'` added to `Real Estate Taxes` account list in `IS_ACCOUNTS['EXPENSES']`, so the income statement includes it alongside 5090 in all source comparisons.

### Paid-Off Loan Exclusion
- **Filter**: Loans with `vDateType = "Paid Off"` are excluded from all analysis at the data layer
- **Implementation**: `data_service.py:load_all()` filters `mri_loans_raw` after loading; same filter applied in `refresh_table()` for the `loans` table
- **Column detection**: Case-insensitive lookup (`c.lower() == "vdatetype"`) for robustness across CSV/MRI sources
- **Scope**: Affects all consumers — waterfall, capitalization, dashboard, debt service, assistant tools

### Balloon Loan Payoff at Sale
- **Detection**: Loan schedule's last row has `ending_balance < 1.0` (float tolerance), `principal > 0`, and prior row's `ending_balance > 0`
- **Forecast exclusion**: Balloon principal payments are excluded from forecast debt service rows (they are NOT operating expenses)
- **Sale proceeds**: Net sale proceeds deduct `total_loan_balance_at(sale_date) + balloon_total` — uses pre-balloon balance since balloon is paid at sale
- **Sale proceeds formula**: `max(0, value_net_selling_cost - loan_balances + tax_abatement_npv)`
- **Groupby**: Loan schedule grouped by `["vcode", "LoanID", "event_date"]` to distinguish loans with same dates
- **ISBS debt fallback**: When modeled loans aren't active at sale date (e.g. all loans paid off, or loan originates after sale), `compute.py` falls back to ISBS balance sheet debt (`get_isbs_debt_balance()`) for loan payoff amount. Also handles case where `loan_sched` is completely empty but ISBS shows outstanding debt.

### Sale Overrides
- **UI**: Contract Sale Price, Selling Costs (% or $), Sale Date — input row on Deal Analysis below capitalization table
- **Contract Sale Price**: Overrides the NOI/cap rate implied value when populated
- **Selling Costs**: Overrides the default 2% selling cost; supports percentage (`pct`) or fixed dollar (`fixed`) modes
- **Sale Date**: Overrides the default sale date (from `event_dates` projected disposition); threads through `compute_service.py` as `sale_date_override`
- **Database**: `sale_overrides` table (vcode PK, contract_sale_price, selling_cost_value, selling_cost_type, sale_date_override, updated_at, updated_by). In `PROTECTED_TABLES`.
- **Cache behavior**: Sale overrides do NOT create separate cache keys — `force=True` is set automatically when any override is present, overwriting the existing cache entry
- **Workflow**: Enter values → Recompute → optionally Save (persists to DB) or Clear. Saved overrides auto-load on deal select.
- **API**: `GET/PUT/DELETE /api/deals/<vcode>/sale-override`. POST `/compute` accepts overrides in body; falls back to saved overrides if not provided.
- **ISBS-anchored loan payoff**: Modeled amortization balance at sale date is scaled by `(ISBS_actual_debt / modeled_debt_at_anchor_date)` ratio. The ISBS Interim BS gives the real current outstanding; the scale factor corrects for differences between MRI origination amount and actual paydown. Diagnostic message shows anchor date, actual/modeled balances, and scale factor.

### Parcel Sales (Interim Sales)
Sales of part of a property before the final disposition. A deal can carry
several. Expandable box on Deal Analysis, directly above the Sale row.

- **Inputs per sale**: label, projected sale date, sale price, cost of sale (% or $), debt paydown allocated across mortgages, amount held in the CapEx reserve, distribution mode, and the revenue/expenses that leave with the parcel.
- **Database**: `parcel_sales` table (id, vcode, label, sale_date, sale_price, cost_of_sale_value/type, debt_application, capex_reserve_hold, distribution_mode, distribution_fixed, lost_revenue, lost_expense, notes, sort_order). The four structured columns are JSON blobs, following `prospect_assumptions`. In `PROTECTED_TABLES`.
- **Money flow** (`compute_economics()` in `parcel_sale_service.py`, the single definition the UI and engine share): price → less cost of sale → less debt paydown → less reserve hold → remainder distributed.
- **Reserve hold**: passed to `build_cash_flow_schedule_from_fad()` as `reserve_deposits`; joins the cash balance *before* CapEx is funded, then is spent by the ordinary reserve rules. Anything unspent returns at the final sale through the existing `get_sale_period_total_cash()` path — no separate logic. Note the reserve can also fund operating deficits that were previously unfunded, so a parcel sale may improve returns by more than its proceeds.
- **Debt paydown**: `Loan.curtailments` + the amortization builder. Convention is **re-amortize** — the payment is recalculated over the remaining term and the maturity date does not move, so debt service and DSCR improve from the paydown date. An interest-only loan simply drops its interest.
- **Distribution modes**: `waterfall` feeds the remainder into `cap_period_cash` as a Cap event at the parcel date (exact — `run_interleaved_waterfalls` processes it in date order); `pro_rata` splits on contributed capital; `fixed` uses per-partner amounts. All three are a **return of capital** — they reduce capital outstanding, so weighted average capital falls and ROE rises gradually rather than spiking.
- **Lost income**: `apply_parcel_income_loss()` spreads annual amounts across the months from the sale date, adjusting `mAmount_norm` only (the column every NOI/FAD/DSCR/terminal-value path reads). Revenue subtracts; expense adds back. Because the exit value is forward NOI / cap rate, **removing revenue automatically lowers the final sale price** — correct, and stated in diagnostics so it is not read as an error.
- **Tenant picker**: `get_deal_tenants()` reads the MRI roster and seeds Rental Income (4010); the figure stays editable because a point-in-time rent roll will not tie to a forecast carrying growth and rollover. Identical roster rows are collapsed — the roster returns every tenant twice and summing raw rows doubles the rent removed.
- **Cache**: parcel sales are loaded *inside* `get_cached_deal_result()` and fingerprinted into the cache key, so Deal Analysis, the reports and the assistant share one set of assumptions. The fingerprint covers only fields that change the projection, so a rename does not force a recompute.
- **Reporting**: four lines on the Annual Forecast below DSCR — Net Proceeds, Debt Paydown, To CapEx Reserve, Distributed — added only when the deal has a sale, and picked up by the Excel export automatically. `result['parcel_sales_applied']` is a JSON-safe audit record of what was applied.
- **Guards** (each reported rather than silent): a parcel date after the deal's sale date is ignored; a paydown at or before the actuals boundary is not modelled, since it is already in the reported balance and would otherwise corrupt the ISBS anchoring; a paydown allocated to a loan that is not modelled is not applied; a loan retired early by a paydown is not misread as a balloon and deducted again at sale; fixed amounts that miss the remainder are distributed as entered with the difference reported; an amount naming a partner not on the deal is not distributed.
- **All three modes process in date order**: `pro_rata`/`fixed` remainders are runner events (`build_manual_parcel_events()` + `manual_events` on `run_interleaved_waterfalls`) — pref accrues to the parcel date on the pre-cut balance, then the return of capital reduces the pools, so pref for every later period accrues on the reduced capital, same as the `waterfall` mode.
- **API**: `GET/POST /api/deals/<vcode>/parcel-sales`, `PUT/DELETE /api/deals/<vcode>/parcel-sales/<id>`, `GET /api/deals/<vcode>/parcel-sales/tenants`. Validation runs server-side on create and update; errors block the save, warnings do not.

### Cap Rate at Sale / Refinance
- **Source column**: `fCapRate` from `valuations` table (MRI_Val)
- **Date column**: `dtValuation` — the date each cap rate was assessed
- **Function**: `projected_cap_rate_at_date()` in `planned_loans.py` — sorts by `dtValuation`, takes the most recent `fCapRate` as the base rate
- **Escalation**: +0.05% (0.0005) per year from the `dtValuation` of the selected row to the target date (sale or refi)
- **Example**: fCapRate=0.0575 at 12/31/2025, sale at 12/31/2030 → 0.0575 + 5×0.0005 = 0.06 (6.00%)
- **Sale value**: `NOI_12_months / cap_rate` — uses `twelve_month_noi_after_date()` for forward 12-month NOI from forecast
- **Callers**: `compute_deal_analysis()` for sale proceeds, `size_planned_second_mortgage()` and `size_prospective_loan()` for loan sizing

### Prospective Loans (Refinancing)
- `size_prospective_loan()` in `planned_loans.py` returns sizing dict including `refi_date` for Vue form pre-fill
- Prospective loan extends sale date to new maturity when `Sale_ME` < new maturity
- Net sale proceeds formula: `sale_price - loan_balances - balloon_total + cash_reserves`
- **Loan type**: Radio selector — Supplemental (keep all existing loans) or Refinance (replace selected loans via checkbox list)
- **Multi-loan replacement**: `existing_loan_id` stores comma-separated LoanIDs; `compute.py` filters with `not in replacing_ids`; `planned_loans.py` uses `.isin(replacing_ids)` for balance lookup
- **Sizing constraints**: LTV (value x max LTV), DSCR (NOI / min DSCR → solve principal), Debt Yield (NOI / min yield), Quoted amount — binding = minimum
- **Workflow**: Draft → Save & Analyze (sizing only) → Accept (full waterfall recompute) → Revert. Editing an accepted loan auto-recomputes waterfall on save.
- **CRUD endpoints**: `GET/POST/PUT/DELETE /api/deals/<vcode>/prospective-loans`, plus `/accept`, `/revert`, `/sizing` per loan
- **PostgreSQL**: `ensure_pg_tables()` in `database.py` creates tables (prospective_loans, prospective_loans_audit, waterfall_audit) with `SERIAL` id + proper column types at startup; `_pg_fix_column_types()` migrates TEXT→INTEGER/DOUBLE PRECISION. Column names with mixed case (iOrder, PropCode, FXRate) must be double-quoted in SQL statements.

### ISBS Data Formats
- **Source column**: `vSource` in ISBS table (`ISBS_Download.csv`)
- **Interim IS** (Actuals): YTD cumulative trial balance snapshots — use `_get_cumulative_balances()` at a single date
- **Interim BS** (Balance Sheet): Current outstanding balances — used for debt via `get_isbs_debt_balance()`
- **Budget IS**: Periodic monthly amounts — use `_get_budget_sum()` over date range
- **Projected IS** (Underwriting): YTD cumulative trial balance snapshots — use `_get_cumulative_balances()` (same as Actuals)
- **Valuation**: Periodic monthly from `forecast_feed` — use `_get_valuation_sum()` with negated `mAmount_norm`
- **TTM from cumulative**: Current YTD + prior year Dec YTD - prior year same-month YTD
- **Performance chart / Dashboard**: Both correctly convert cumulative→periodic via `_cumulative_to_periodic()`

### ISBS Split Tables
MRI's query record limits make exporting the monolithic `ISBS_Download.csv` (800K+ rows) impractical. The ISBS data is split into 6 tables by `vSource`:

| Table | CSV Filename | vSource | Description |
|-------|-------------|---------|-------------|
| `isbs_interim_is` | `ISBS_Interim_IS.csv` | Interim IS | Actuals — YTD cumulative (2025+) |
| `isbs_interim_is_historical` | `ISBS_Interim_IS_Historical.csv` | Interim IS | Actuals — YTD cumulative (pre-2025) |
| `isbs_interim_bs` | `ISBS_Interim_BS.csv` | Interim BS | Balance Sheet |
| `isbs_budget_is` | `ISBS_Budget_IS.csv` | Budget IS | Budget — periodic monthly |
| `isbs_projected_is` | `ISBS_Projected_IS.csv` | Projected IS | Underwriting — YTD cumulative |
| `isbs_valuation_is` | `ISBS_Valuation_IS.csv` | Valuation IS | Valuation — periodic monthly |
| `isbs_uw_supplements` | `ISBS_UW_Supplements.csv` | Projected IS | Supplemental UW records (e.g. 7073 capital contributions) — importable via CSV upload, persists across MRI refreshes (not in QUERY_REGISTRY) |

- **Assembly**: `_assemble_isbs()` in `data_service.py` loads split tables, restores `vSource` column, concatenates into `isbs_raw`. Supplements from legacy monolithic `isbs` table for any missing vSource categories. Falls back entirely to legacy table if no split tables exist. After assembly, `_append_uw_supplements()` appends rows from `isbs_uw_supplements` (defaults `vSource='Projected IS'` if not present in CSV).
- **CSV upload**: Auto-detects split table filenames via `TABLE_DEFINITIONS` in `database.py`
- **Cache**: `refresh_table()` reassembles `isbs_raw` when any split table or `isbs_uw_supplements` is updated
- **Migration**: `split_isbs_table()` in `database.py` can migrate legacy monolithic table into splits (idempotent)
- **Consumers unchanged**: All code continues to use `isbs_raw` with `vSource` filtering
- **Pending**: Direct MRI database access via VPN (requested Apr 2026) to bypass record limits entirely

### ISBS Debt Balance
- **Config**: `DEBT_BS_ACCTS = {'2150', '2152', '2210'}` in `config.py` — Balance Sheet debt accounts
- **Function**: `get_isbs_debt_balance(isbs_raw, vcode, as_of_date=None)` in `compute.py`
- **Logic**: Filters ISBS to `vSource='Interim BS'`, deal vcode (case-insensitive), debt accounts; picks most recent period (or specific `as_of_date`); returns `abs(sum(mAmount))`
- **Hierarchy**: ISBS current outstanding preferred over MRI_Loans `mOrigLoanAmt` (static origination). Falls back to MRI_Loans if ISBS unavailable.
- **Usage**: Dashboard (`get_portfolio_caps()`), Deal Analysis (`get_deal_capitalization()`), One Pager (`get_capitalization_stack()` with quarter-specific date)
- **Date parsing**: Uses `pd.to_datetime(format='mixed')` first; Excel serial fallback only when >50% NaT

### At Close Data (One Pager)
- **Primary source**: `at_close_noi` table (from `Prop_Info_AtClose.sql` MRI query) — pre-computed per deal with dynamic date selection
- **Fallback**: ISBS `vSource='Projected IS'` at the earliest December 31 date per deal
- **Meaning**: Due diligence audit performed at original closing — represents underwritten expectations
- **Fields**: Revenue, Expenses, NOI, DSCR (debt service from Interest + Principal accounts)
- **Sign convention**: MRI stores revenue as negative (credit); negated to positive for display
- **Deal terms**: `deal_terms` table (from `Prop_Info_DealTerms.sql`) provides `econ_occ_at_close` and PE coupon/participation overrides
- **Implementation**: `get_property_performance()` in `one_pager.py` checks `at_close_noi_df` first, falls back to ISBS Projected IS scan

### Economic Occupancy (One Pager)
- **Formula**: `avg(physical occupancy YTD months) - bad_debt_concessions_pct`
- **Physical occupancy**: Average of `Occ%` from MRI_Occupancy_Download for YTD months of current year through quarter end
- **Bad debt/concessions %**: `(sum of vAccounts 4040 + 4043) / abs(sum of vAccount 4010) × 100` from ISBS Interim IS YTD
- **Accounts**: 4040 = Residential Concessions, 4043 = Bad Debt & Collection Loss, 4010 = Rental Income

### Sub-Portfolio Aggregation
- Deals can have child properties linked via `Portfolio_Name`
- Loans aggregate UP from properties to parent deal level
- See `consolidation.py` for implementation

### Forecast Assembly (Multi-Source)
- **Priority 1**: `forecast_feed.csv` (admin-uploaded via CSV import → `forecasts` table). Overrides everything for deals it covers. Used for draft/updated valuations before MRI approval.
- **Priority 2**: ISBS `vSource = 'Valuation IS'` — annual valuation cash flow projections (periodic monthly). These are the MRI-approved 10-year projections from the December 31 valuation exercise.
- **Priority 3**: ISBS `vSource = 'Projected IS'` — underwriting projections (YTD cumulative, auto-converted to periodic monthly).
- **Assembly**: `_assemble_forecasts()` in `data_service.py` — identifies deals covered by forecast_feed, then fills gaps from ISBS Valuation IS, then ISBS Projected IS. Combined result passed to `load_forecast()`.
- **Cumulative→Periodic**: Projected IS conversion subtracts prior same-year cumulative value; January = start of new year (no subtraction).
- **Pro_Yr derivation**: For ISBS-sourced rows, `Pro_Yr = date.year - pro_yr_base`.
- **vcode case**: ISBS normalizes vcodes to lowercase; `_restore_case()` converts back to original case (e.g., `p0000008` → `P0000008`) for case-sensitive matching in `compute_deal_analysis()`.
- **Cache refresh**: `refresh_table()` reassembles forecasts when `forecasts`, `isbs`, or any ISBS split table changes.

### Forecast Date Filtering
- **Anomalous dates**: MRI valuation exports can include "Year 0" base entries with dates far in the past (e.g. 2015-12-31 for a 2025 deal). `compute_deal_analysis()` filters out forecast rows with `event_date` before `start_year - 2` to prevent `model_start` from being set to an unreasonable date.
- **Beginning cash date selection**: `load_beginning_cash_balance()` uses the **most recent** available Interim BS date (not restricted to pre-model_start). This ensures account reclassifications (e.g. security deposits moved from 1010→1080) are reflected in the beginning cash. Previously restricted to dates before `model_start`, which could include misclassified balances from older periods.

### Actuals Through Cutoff
- Global setting (`actuals_through`): date or None (default None = full forecast)
- **Actuals/Forecast boundary**: Always enforced. If `actuals_through` is set, that date is the cutoff. Otherwise, defaults to Dec 31 of `start_year - 1`. XIRR cash flows come from `accounting_feed` only before the boundary, and from `forecast_feed` (waterfall) only after.
- **Partner cash flows**: `seed_states_from_accounting()` accepts `cutoff_date` parameter — only accounting entries on or before cutoff are used to seed InvestorState. Waterfall-computed distributions cover periods AFTER cutoff only.
- **Operating forecast**: Forecast Rev+Exp rows for months <= cutoff are removed from `fc_deal_full`
- **Waterfall**: `cf_period_cash` and `cap_period_cash` filtered to post-cutoff periods only (always, not just when `actuals_through` is set)
- **Date comparison safety**: Period filtering uses `pd.to_datetime()` on `event_date`/`EffectiveDate` columns before comparing with `pd.Timestamp` cutoff to avoid `Timestamp vs datetime.date` errors
- **Cache key**: includes `actuals_through` so toggling triggers recomputation
- **Defaults**: `DEFAULT_START_YEAR = 2026`, `DEFAULT_HORIZON_YEARS = 10`, `PRO_YR_BASE_DEFAULT = 2025`, `DEFAULT_ACTUALS_THROUGH = "2026-07-31"` (YTD Actuals enabled through July 2026)
- **UI**: Vue sidebar in Report Settings (checkbox + month-end selector)
- **Flask**: `ACTUALS_THROUGH` in config, passed via query params / request body, included in `/api/data/config`

## Sidebar Navigation

The sidebar (`AppSidebar.vue`) is organized into major sections with expandable dropdowns. Section headers are uppercase bold; child items are indented. Sections auto-expand when navigating to a child route.

| Section | Type | Children |
|---------|------|----------|
| **Dashboard** | Standalone link | `/dashboard` |
| **Asset Management** | Expandable | Deal Analysis, Property Financials, Surveillance, One Pager, Review Tracking, Ownership, Waterfall Setup, Report Settings (expandable config panel) |
| **Accounting** | Future (dimmed) | — |
| **New Business** | Expandable | Pipeline, Deal Analysis, Lease Review, Lease Risk Analysis |
| **Investment Management** | Future (dimmed) | — |
| **Reports** | Standalone link | `/reports` (Projected Returns, ROE Summary, Pref Balance Detail, Sold Portfolio, PSCKOC, Portfolio Analysis) |
| **Data Management** | Expandable | Data Explorer, MRI Data (expandable panel), Database Tools (expandable panel), Reload Data, Settings |
| **Feedback & Requests** | Expandable | Submit form + request list (standalone section below nav) |

- **Report Settings** under Asset Management: expandable inline config panel (Start Year, Horizon, Pro_Yr Base, YTD Actuals + Apply Settings button)
- **MRI Data** under Data Management: expandable panel with server status, query list, per-query download/run/import buttons, admin "Refresh All Data from MRI"
- **Database Tools** under Data Management: expandable panel with Import CSVs (file upload + match), Export Database (.zip download)
- **Sold Portfolio**, **PSCKOC**, and **Portfolio Analysis** are embedded as custom view reports inside the Reports page (selected from the report list sidebar). Their Vue components accept an `embedded` prop that hides their standalone headers.

## Application Tabs & AI Assistant

Moved to `.claude/memory/app_reference.md` (Sep 11 2026) — what every tab displays,
section by section, plus the embedded AI Assistant's tool table and endpoints. It was
half of this file's bytes and loaded into every session regardless of whether the work
touched the UI. Read it when you need to know what a view shows or which endpoint backs
it; the sidebar map above is kept here as a quick orientation.

## Key Functions

### Core Engine
- `compute_deal_analysis()` - Main deal computation orchestration (compute.py)
- `build_partner_results()` - Single source of truth for all partner & deal metrics (compute.py)
- `run_interleaved_waterfalls()` - Merges CF/Cap timelines chronologically with shared InvestorState (compute.py)
- `prepare_cap_lookups()` - Pre-compute normalized DataFrames and lookup dicts for batch capitalization (compute.py)
- `xirr(cfs)` - Calculate IRR with irregular dates; Newton-Raphson (guess=0.1) matching Excel XIRR, Brent fallback (metrics.py)
- `accrue_pref_to_date()` - Daily pref accrual (waterfall.py)
- `parse_amfee_vnotes()` - Parse AMFee vNotes for source investor and exclusions (waterfall.py)
- `build_amfee_exclusions()` - Pre-compute net capital by (InvestmentID, InvestorID) for AMFee exclusions (waterfall.py)
- `get_amfee_excluded_capital()` - Compute capital to exclude from AMFee base, scaled by ownership % (waterfall.py)
- `seed_states_from_accounting()` - Build InvestorState from historical accounting; accepts `cutoff_date` to limit to actuals boundary. Non-capital distributions recorded in `cf_distributions` for ROE audit (waterfall.py)
- `InvestorState` - Tracks capital, pref, cashflows per investor (models.py)
- `Loan` - Debt structure with fixed/variable rates (models.py)
- `get_property_vcodes_for_deal()` - Get child properties for aggregation (consolidation.py)
- `cashflows_monthly_fad()` - Monthly FAD from modeled forecast; includes CapEx (reporting.py)
- `annual_aggregation_table()` - Annual pivot table for forecast display; accepts cash_schedule for reserve-adjusted FAD (reporting.py)
- `build_cash_flow_schedule_from_fad()` - Transforms FAD into distributable; funds CapEx from reserves, tracks shortfalls (cash_management.py)
- `get_isbs_debt_balance()` - Current debt from ISBS balance sheet, fallback to MRI_Loans (compute.py)
- `load_beginning_cash_balance()` - Beginning cash balance from ISBS Interim BS using CASH_BALANCE_ACCTS; uses most recent available BS date for accurate account classification (cash_management.py)
- `get_deal_capitalization()` - Deal cap stack with ISBS debt support (compute.py)
- `projected_cap_rate_at_date()` - Cap rate with annual escalation from valuation date (planned_loans.py)
- `twelve_month_noi_after_date()` - Forward 12-month NOI from forecast for sale/refi valuation (planned_loans.py)

### One Pager Data
- `get_general_information()` - Deal general info from investment_map + event_dates (one_pager.py)
- `_lookup_event_date()` - Generic event_dates query by vCode/vEventType/vEvent/vDateType (one_pager.py)
- `get_underwritten_exit()` - U/W Exit from event_dates (Asset Management / U/W Exit / Actual) (one_pager.py)
- `get_current_anticipated_exit()` - Current exit from event_dates (Disposition / Closing / Projected) (one_pager.py)
- `get_capitalization_stack()` - Cap stack, loan terms, PE exposure, investor breakdown (one_pager.py)
- `get_property_performance()` - YTD/Budget/Variance/AtClose/YE metrics per quarter (one_pager.py)
- `get_pe_performance()` - PE funding, ROE, balances from accounting; U/W ROE from ISBS Projected IS only (one_pager.py)
- `_get_uw_7073_signed()` - Sign-preserving 7073 YTD→periodic: positive MRI = contribution (neg CF), negative MRI = ROC (pos CF) (one_pager.py)
- `get_one_pager_comments()` / `save_one_pager_comments()` - Comments CRUD per vcode+quarter (one_pager.py)

### Capital Calls
- `load_capital_calls()` - Load and normalize capital calls with mixed date format handling (capital_calls.py)
- `build_capital_call_schedule()` - Build list of capital call events, optionally filtered by deal (capital_calls.py)
- `apply_capital_calls_to_states()` - Apply capital calls to investor states with pool routing (capital_calls.py)

### Parcel Sales
- `compute_economics()` - Price less cost of sale, paydown and reserve; the remainder to distribute (parcel_sale_service.py)
- `validate()` - Per-sale checks, errors vs warnings (parcel_sale_service.py)
- `get_deal_loans()` / `get_deal_tenants()` - Paydown targets and roster rows for the pickers; loan_id kept in the raw form `loans.py` builds ids with, tenant rows deduped (parcel_sale_service.py)
- `get_prospect_deal_loans()` / `get_prospect_tenants()` - NB (N-vcode) equivalents: loans from the latest saved Capital Budget mirroring `_build_loans` ids ({vcode}-{source_id} / {vcode}-L1 blended); tenants from the scenario-pinned Argus import (requested scenario, else Base Case), else the active import, else the lease-review roster with analyst resolutions winning (parcel_sale_service.py). `deals.py` falls back to both for N-vcodes; the tenants endpoint accepts `?scenario_id`.
- `apply_parcel_income_loss()` - Remove the revenue/expenses that leave with a parcel, from the sale date forward; records per-parcel/account/month removals in `result['parcel_revenue_detail']` for the display back-out (compute.py)
- `build_manual_parcel_events()` - Resolves pro-rata/fixed shares + validation; the runner applies them in date order via `manual_events` (compute.py)

### Database
- `save_waterfall_steps()` - Replace all waterfall steps for a vcode with audit trail; quotes column names for PostgreSQL (database.py)
- `create_additional_tables()` - Creates capital_calls and other tables if they don't exist (database.py)
- `import_csv_dataframe()` - Import a DataFrame into a table, works with SQLite and PostgreSQL (database.py)
- `import_csvs_to_database()` - Refresh all tables from CSVs, protecting DB-managed tables (database.py)
- `export_all_tables_to_zip()` - Export all tables as labeled CSVs in a zip archive (database.py)
- `set_engine()` - Wire SQLAlchemy engine for PostgreSQL support (database.py)
- `import_csv_stream()` - Chunked CSV import (50K rows, dtype=object) for large files (database.py)
- `split_isbs_table()` - Migrate monolithic isbs table into 6 split tables by vSource; idempotent (database.py)
- `_assemble_isbs()` - Load split ISBS tables, restore vSource column, concatenate; fallback to legacy (data_service.py)
- `_append_uw_supplements()` - Append isbs_uw_supplements rows to assembled ISBS; defaults vSource='Projected IS' (data_service.py)
- `_assemble_forecasts()` - Merge forecast_feed CSV > ISBS Valuation IS > ISBS Projected IS with per-deal priority (data_service.py)

### Cash Flow Imports (Argus + Generic)
- `parse_monthly_cashflow()` - Parse Argus cash flow Excel, auto-detect periods, map line items to COA (argus_parser.py)
- `parse_rent_roll_summary()` - Parse tenant lease detail from Argus rent roll export (argus_parser.py)
- `cashflow_to_forecast_df()` - Convert parsed Argus data to forecast DataFrame (same schema as load_forecast) (argus_parser.py)
- `map_to_coa()` - Keyword-based line item → COA account mapping (argus_parser.py)
- `parse_cashflow_excel()` - Generic Excel/CSV parser, auto-detect columns, annual→monthly conversion, horizontal layout support (cashflow_parser.py)
- `_detect_horizontal_dates()` - Detect horizontal Excel layout (dates across columns) (cashflow_parser.py)
- `_parse_horizontal_cashflow()` - Parse transposed cash flow (line items as rows, dates as columns) (cashflow_parser.py)
- `import_argus_cashflow()` - Import with SHA-256 dedup, parse + store cashflows (argus_service.py)
- `get_active_forecast_df()` / `get_forecast_df_by_id()` - Forecast DataFrame from Argus projection (argus_service.py)
- `get_property_rollup_forecast_df()` - Aggregate Argus forecasts from multiple properties into deal-level forecast (argus_service.py)
- `migrate_projection_to_forecast()` - Re-key vcode + insert into forecasts table for AM onboarding (argus_service.py)
- `import_property_cashflows()` - Store Excel-parsed cash flows by property+version (prospect_service.py)
- `get_deal_cashflows_by_property()` - All cashflows grouped by property_id for deal-level rollup (prospect_service.py)

### Flask Services
- `get_cached_deal_result()` - Shared multi-deal cache wrapper (compute_service.py)
- `build_roe_audit()` / `build_moic_audit()` - Audit data builders (compute_service.py)
- `generate_roe_audit_excel()` / `generate_moic_audit_excel()` - Audit Excel workbooks (compute_service.py)
- `generate_partner_returns_excel()` - Partner Returns Excel with deal total row (compute_service.py)
- `generate_forecast_excel()` - Annual Forecast Excel pivoted by year (compute_service.py)
- `generate_debt_service_excel()` - Loan Summary + Amortization Schedule (2-sheet) (compute_service.py)
- `generate_cash_schedule_excel()` - Cash flow schedule Excel (compute_service.py)
- `generate_capital_calls_excel()` - Capital calls Excel (compute_service.py)
- `generate_xirr_cashflows_excel()` - Merged XIRR cashflows by partner; sums same-date/same-description entries (compute_service.py)
- `generate_full_deal_excel()` - 7-sheet comprehensive Deal Analysis workbook (compute_service.py)
- `get_one_pager_data()` - Aggregates all one-pager sections + computes pe_yield_on_exposure; accepts `full_data` for PE enrichment (financials_service.py)
- `_enrich_pe_from_deal_result()` - Enriches PE performance from deal analysis seed_states (accrued balance, capital outstanding) (financials_service.py)
- `get_one_pager_chart()` - Quarterly NOI chart data for one-pager (financials_service.py)
- `get_submission()` - Get or create review submission with notes and permissions (review_service.py)
- `submit_for_review()` - Submit draft/returned document for review (review_service.py)
- `approve()` - Approve at current step and advance to next (review_service.py)
- `return_to_draft()` - Return document to draft with required note (review_service.py)
- `is_editable()` - Check if comments can be edited based on review status (review_service.py)
- `get_tracking_data()` - Production tracking data with filters, LEFT JOINs deals with submissions (review_service.py)
- `_save_snapshot()` - Freeze all computed One Pager data + chart as JSON on CEO approval (review_service.py)
- `get_snapshot()` - Retrieve and deserialize a frozen snapshot by vcode+quarter (review_service.py)
- `compute_net_waterfall_for_deal()` - Per-deal net returns waterfall with fees/promote (sold_service.py)
- `compute_all_net_returns()` - Orchestrator: loops sold deals, pools net cashflows for portfolio metrics (sold_service.py)
- `generate_net_returns_excel()` - Multi-sheet workbook with formula-driven waterfall detail (sold_service.py)
- `build_pref_balance_detail()` - Per-investor pref accrual detail matching Excel PE_Pref_Balances (reports_service.py)
- `get_deal_pe_investors()` - List PE investors for a deal from accounting (reports_service.py)
- `generate_pref_balance_excel()` - Pref balance detail Excel with header + transaction table (reports_service.py)
- `create_request()` - Create a new user feedback request with reply token (feedback_service.py)
- `list_requests()` - List requests with optional user/status/type filters (feedback_service.py)
- `send_request_email()` - Send email to request submitter via SendGrid with reply link (feedback_service.py)
- `handle_inbound_email()` - Process inbound email reply, match token, store in thread (feedback_service.py)
- `export_all_requests()` - Export all requests with full threads for design sessions (feedback_service.py)

## Account Classifications

- Revenue accounts: 4xxx series (positive in `mAmount_norm`)
- Expense accounts: 5xxx series (negative in `mAmount_norm`)
- Interest: `INTEREST_ACCTS` {5190, 7030}
- Principal: `PRINCIPAL_ACCTS` {7060}
- CapEx: `CAPEX_ACCTS` {7050}
- Tax Abatement: `TAX_ABATEMENT_ACCTS` {7070} — forced positive (income)
- Other Below-the-Line: `OTHER_EXCLUDED_ACCTS` {4050, 5220, 5210, 5195, 7065, 5120, 5130, 5400}
- `ALL_EXCLUDED` = Interest | Principal | CapEx | Other Below-the-Line (does NOT include Tax Abatement — separate sign handling)
- Debt (Balance Sheet): `DEBT_BS_ACCTS` {2150, 2152, 2210} — ISBS Interim BS accounts for current debt outstanding
- Cash & Reserves (Balance Sheet): `CASH_BALANCE_ACCTS` {1010, 1012, 1014, 1070, 1090, 1091, 1092, 1100, 1120, 1130, 1140, 1141, 1142, 1144, 1145} — ISBS Interim BS accounts for beginning cash balance in Cash Management section
- Bad Debt/Concessions: 4040 (Residential Concessions), 4043 (Bad Debt & Collection Loss), 4010 (Rental Income) — used in Economic Occupancy calculation
- Underwritten PE (Projected IS): `UW_PE_DIST_ACCT` 7071 (PE distributions — ROE numerator), `UW_PE_ROC_ACCT` 7073 (PE capital events — sign convention: positive = contribution, negative = return of capital). 7071 is YTD cumulative converted to periodic by `_get_uw_pe_periodic()`. 7073 uses `_get_uw_7073_signed()` for sign-preserving YTD→periodic conversion (positive MRI → negative cashflow/contribution, negative MRI → positive cashflow/ROC). Supplemental 7073 records loaded from `isbs_uw_supplements` table (importable via CSV upload, persists across MRI refreshes).

## Conventions

- Cashflow signs: negative = contribution, positive = distribution
- Rates as decimals (0.08 = 8%)
- Use Python date objects for dates
- **InvestorID / InvestmentID case normalization**: All entity IDs are uppercased at data load time (`.str.strip().str.upper()`) in `loaders.py`, `data_service.py`, and `ownership_tree.py`. MRI accounting data occasionally has mixed-case entries (e.g. "Centre" instead of "CENTRE") which caused journal entries to be silently dropped from groupby/filter operations. Normalization happens at the lowest layer so all downstream consumers get consistent IDs.
