# Session Handoff — Aug 22, 2026

## What Was Done

### Session 1: Argus Enterprise Loader (Full Implementation)
Implemented the complete Argus Enterprise Excel import pipeline across all 5 phases.

**Commit**: `4af6b34` — pushed to main, deployed as revision v302

#### New Files
1. **`argus_parser.py`** — Stateless parser (no DB/Flask deps)
   - `ARGUS_COA_MAP`: 56 keyword-to-COA mappings (revenue → 4010/4030/4075/4090-92, expense → 5020/5040/5060/5090/5110, capex → 7050)
   - `parse_monthly_cashflow()`: Auto-detects period dates from Excel headers, maps line items, filters summary rows
   - `parse_rent_roll_summary()`: Flexible column detection for tenant blocks
   - `parse_revenue_assumptions()`: Market leasing profiles
   - `cashflow_to_forecast_df()`: Converts parsed data to compute-compatible DataFrame
   - Sign normalization follows `prospect_analysis.py:_fc_row()` pattern

2. **`flask_app/services/argus_service.py`** — DB service layer
   - Import: SHA-256 dedup, parse + store with sign normalization
   - CRUD: `get_projection_scenarios()`, `set_active_projection()`, `delete_projection()` (cascade)
   - Forecast: `get_active_forecast_df()`, `get_forecast_df_by_id()` — returns forecast DataFrame from stored Argus cashflows
   - COA mapping: `get_coa_mapping()`, `update_coa_mapping()` — review/override unmapped items
   - Migration: `migrate_projection_to_forecast()` — re-keys vcode (N→P series)
   - **`get_property_rollup_forecast_df()`** — aggregates active Argus forecasts from multiple properties (NP vcodes) into one deal-level forecast

3. **`flask_app/api/argus.py`** — 11 REST endpoints at `/api/argus`

4. **`vue_app/src/components/common/ArgusImport.vue`** — Shared upload component

#### Modified Files
5. **`database.py`** — 5 new tables: argus_imports, argus_cashflows, argus_tenants, argus_rent_steps, argus_market_profiles
6. **`flask_app/__init__.py`** — Registered `argus_bp` at `/api/argus`
7. **`flask_app/services/compute_service.py`** — `projection_id` in cache key, Argus forecast substitution
8. **`flask_app/api/deals.py`** — Threads `projection_id` through `/compute`
9. **`vue_app/src/stores/deals.ts`** — `computeDeal()` accepts `{ force, projection_id }`
10. **`vue_app/src/views/DealAnalysisView.vue`** — Projection dropdown + "Import Argus" button + modal

---

### Session 2: Property-Level Cash Flow Imports + Generic Excel Parser
Moved Argus import from deal-level to property-level. Added generic Excel/CSV parser for partner models. Both paths store property-level data that rolls up to deal-level analysis.

**Commit**: `fe54e06` — pushed to main, deployed as revision v305

#### New Files
1. **`cashflow_parser.py`** — Generic Excel/CSV parser for partner cash flow models
   - Auto-detects columns via regex patterns (revenue, expenses, NOI, capex, date)
   - Handles both annual and monthly data (annual auto-spread to 12 monthly rows)
   - Normalizes signs, derives missing columns (NOI from rev-exp, or rev/exp from NOI)
   - Handles dollar signs, commas, parenthetical negatives, messy header rows
   - Tested with both annual and monthly synthetic Excel files

#### Modified Files
2. **`flask_app/services/prospect_service.py`** — 4 new CRUD functions:
   - `import_property_cashflows()` — replaces existing rows for a property+version
   - `get_property_cashflows()` — retrieve stored rows
   - `delete_property_cashflows()` — clear stored rows
   - `get_deal_cashflows_by_property()` — all cashflows grouped by property_id

3. **`flask_app/api/prospects.py`** — 3 new endpoints + unified analysis cascade:
   - `POST /<deal_id>/properties/<prop_id>/cashflows/upload` — file upload + parse + store
   - `GET /<deal_id>/properties/<prop_id>/cashflows` — retrieve
   - `DELETE /<deal_id>/properties/<prop_id>/cashflows` — clear
   - `analyze_deal` now checks: Argus → prospect_cashflows → NOI growth assumptions

4. **`flask_app/services/argus_service.py`** — Added `get_property_rollup_forecast_df()`

5. **`prospect_analysis.py`** — Added `argus_forecast_df` parameter; uses it when available

6. **`vue_app/src/views/PipelineView.vue`** — Property card is now the cash flow hub:
   - Removed deal-level Argus `<details>` from Analysis tab
   - "Import Argus" button → Argus import modal (scoped to property via NP vcode)
   - "Upload Cash Flows" button → generic Excel upload modal with column detection preview
   - Green "CF" badge on properties with loaded cash flows
   - Cash flow status loads on deal open

---

### Session 3: Prospect Deal Analysis View + Waterfall Builder
Added standalone Deal Analysis page for New Business, mimicking AM Deal Analysis with shared computation engines. Includes streamlined waterfall builder for modeling proposed partnership structures.

**Commit**: `ce13fb7` — pushed to main, deployed as revision v306

#### New Files
1. **`vue_app/src/views/ProspectAnalysisView.vue`** (933 lines) — Full prospect deal analysis view:
   - Left panel (420px): Deal selector, acquisition form (purchase price, closing costs), operating assumptions (NOI, growth, hold period), debt parameters (LTV, rate, amort, IO), waterfall builder, action buttons
   - Right panel: Sources & Uses summary, deal KPI cards (IRR, ROE, MOIC, sale price), Partner Returns table (PE partners highlighted), expandable Annual Forecast + Debt Service tables, Diagnostics expander
   - Waterfall Builder with two tabs:
     - **Builder tab**: Investor rows (ID, Name, Pref Rate, Residual %, PE checkbox), add/remove, share % validation (must sum to 100%), "Build & Save Waterfall" button
     - **Steps tab**: Preview of stored CF_WF and Cap_WF steps in compact tables
   - Initializes investors from prospect entities or from stored waterfall steps
   - Calls `POST /api/prospects/<id>/analyze` for computation
   - Calls `POST /api/prospects/<id>/waterfall/build` for waterfall generation

#### Modified Files
2. **`flask_app/api/prospects.py`** — 3 new waterfall endpoints:
   - `GET /<deal_id>/waterfall` — retrieve waterfall steps for prospect deal vcode
   - `POST /<deal_id>/waterfall/build` — generate CF_WF + Cap_WF from investor inputs (pref_rate, share_pct, is_pe) + optional promote, save via `save_waterfall_steps()`
   - `DELETE /<deal_id>/waterfall` — delete all waterfall steps for the vcode
   - `analyze_deal()` modified to load real waterfalls from DB before falling back to synthetic `_build_waterfall()`

3. **`prospect_analysis.py`** — Added `waterfall_df` parameter to `build_prospect_analysis()`:
   - When `waterfall_df` is provided and non-empty, uses it directly instead of calling `_build_waterfall()`
   - Enables real DB-stored waterfalls to flow through the computation engine

4. **`vue_app/src/router/index.ts`** — Added `/prospect-analysis` route between Pipeline and Lease Review

5. **`vue_app/src/components/layout/AppSidebar.vue`** — Added "Deal Analysis" link to New Business section between Pipeline and Lease Review

## Pipeline Workflow (As Designed)

```
1. Deal comes in → analyst creates deal + adds properties
2. Per property → analyst loads cash flows via:
   ├─ "Import Argus" (Argus Excel → argus_cashflows, vcode = NP{property_id})
   └─ "Upload Cash Flows" (partner Excel/CSV → prospect_cashflows with property_id)
3. Deal Analysis page (/prospect-analysis) → select deal → configure assumptions
4. Build waterfall: add investors with pref rates + residual splits → "Build & Save"
   └─ Generates CF_WF + Cap_WF steps, saves to waterfalls table
5. Run Analysis → returns computed with same engine as AM Deal Analysis
6. Iterate assumptions/waterfall → determine viable deal structure → quote term sheet
7. Term sheet accepted → move to DD/verification stage
```

## Next Steps

### Testing
- **Real Argus Monthly Cash Flow Excel export** needed from an analyst
- Parser tested with synthetic data only; needs real file validation
- Test generic Excel parser with actual partner Excel models
- Test waterfall builder → analysis flow end-to-end with real prospect data

### Database Tables
| Table | Description |
|-------|-------------|
| `argus_imports` | Import session metadata (vcode, label, type, file hash, active flag) |
| `argus_cashflows` | Monthly line items with COA mapping + normalized amounts |
| `argus_tenants` | Full lease detail from rent roll (25 columns) |
| `argus_rent_steps` | Escalation schedule per tenant |
| `argus_market_profiles` | Revenue assumptions / market leasing profiles |
| `prospect_cashflows` | Property-level cash flows from any source (existing table, now used) |
| `waterfalls` | Prospect deal waterfalls stored with N-series vcodes (shared table) |

### Architecture Notes
- Argus parser is stateless (`argus_parser.py`) — no DB/Flask deps
- Cashflow parser is stateless (`cashflow_parser.py`) — no DB/Flask deps
- Property vcodes: `NP{property_id:06d}` for Argus imports
- Prospect deal vcodes: `N{deal_id:07d}` — used for waterfalls stored in shared `waterfalls` table
- Forecast substitution in `compute_service.py` for AM; in `prospects.py:analyze_deal` for NB
- `prospect_cashflows.source` column tracks origin: 'excel', 'manual', 'argus'
- `build_prospect_analysis()` accepts `waterfall_df` and `argus_forecast_df` for real data override
- Waterfall builder: `POST /<id>/waterfall/build` takes `{investors: [{id, name, pref_rate, share_pct, is_pe}], promote: {enabled, pct}}` → generates Pref + Initial + Share/Tag steps

## Deployed State
- **Revision**: v306
- **Commits**: `4af6b34` (Argus loader) + `fe54e06` (property-level + generic parser) + `ce13fb7` (Prospect Deal Analysis + waterfall builder)
