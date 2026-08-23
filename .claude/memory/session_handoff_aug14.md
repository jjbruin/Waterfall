# Session Handoff — Aug 14, 2026

## What Was Done

### 1. Lease Risk Analysis View (new feature)
- **New Vue component**: `LeaseRiskAnalysisView.vue` at `/lease-risk-analysis`
- **Sidebar**: Added under New Business, below Lease Review
- **Purpose**: All the analysis content (expirations, validation, co-tenancy, scenarios) that was previously in Lease Review Step 7, now in a standalone tab using analyst-resolved data
- **7 tabs**: Overview, Lease Expirations, Validation, Co-Tenancy Risk, Scenario Analysis, Exclusive Use, Options
- **Field resolution system**:
  - `lease_field_resolutions` table with `UNIQUE(tenant_id, field_name)`
  - UPSERT pattern (PG `ON CONFLICT`, SQLite `INSERT OR REPLACE`)
  - Resolvable fields: `square_feet`, `annual_rent`, `monthly_rent`, `rent_per_sf`, `lease_start`, `lease_end`, `security_deposit`
  - `get_resolved_tenants()` overlays analyst resolutions on base tenant data
  - Inline editing: double-click any resolvable field, "R" badge marks resolved, revert button to clear
  - Validation tab: one-click "Use Seller" / "Use Lease" buttons on mismatches
- **Service functions** (appended to `lease_review_service.py`):
  - `ensure_resolution_table()`, `resolve_field()`, `clear_resolution()`
  - `get_resolved_tenants()`, `get_resolved_expiration_histogram()`, `get_resolved_cotenancy_matrix()`, `get_resolved_scenario_analysis()`
  - `get_risk_analysis_data()` — complete bundle
- **API endpoints** (in `lease_review.py`):
  - `GET /reviews/<id>/risk-analysis`
  - `PUT /reviews/<id>/tenants/<tid>/resolve`
  - `DELETE /reviews/<id>/tenants/<tid>/resolve/<field_name>`
- **Commit**: `5a07f7b`

### 2. Folder Upload with Subfolder Tenant Matching
- **Select Folder button** added to Lease Review Step 3 (alongside existing Select Files)
- Uses `webkitdirectory` attribute — browser returns all files from directory tree recursively
- Non-PDF files filtered out client-side
- **Subfolder name as tenant hint**: `webkitRelativePath` parsed to extract immediate parent folder name (e.g., `Starbucks/Original Lease.pdf` → hint "Starbucks")
- Hints sent as `folder_hints` JSON in form data
- `_match_file_to_tenant()` tries folder hint first (containment match against tenant names), falls back to filename matching
- **Commit**: `1b896f7`

### 3. InvestorID/InvestmentID Case Normalization Fix
- **Problem**: MRI accounting data occasionally has mixed-case entity IDs (e.g., "Centre" instead of "CENTRE"). These entries were silently dropped from `groupby`/filter operations, causing missing distributions in ROE, Accrued Pref, and all downstream reports.
- **Example**: Centre at Westbank 1/3/2024 distribution with InvestorID "Centre" instead of "CENTRE"
- **Fix**: Added `.str.upper()` at every data loading entry point:
  - `loaders.py`: `normalize_accounting_feed()` (lines 243-244), `build_investment_map()` (line 313), `load_investor_waterfalls()` (line 381), `load_investor_accounting()` (line 420)
  - `ownership_tree.py`: `load_relationships()` (lines 58-59)
  - `data_service.py`: `load_all()` (inv + relationships), `_enrich_acquisition_dates()`, `refresh_table()` (deals + relationships paths)
- **Commit**: `efc300d`

### 4. Documentation Updates
- CLAUDE.md: Full Lease Review/Risk Analysis docs, sidebar nav table, project structure, ID normalization convention
- Shared memory: Updated `new_business_pipeline.md` and `MEMORY.md`
- **Commit**: `323e3aa`

## Deployed State
- **Revision**: v259
- **All 4 commits pushed to main and deployed**

## Pipeline Delete Buttons
- User asked about deleting duplicate Windsor Square records
- Delete buttons already exist: `×` on property cards, "Delete Deal" at bottom of deal detail panel
- Both have confirmation dialogs — no code change needed

## Open Items / Future Work
- **DD Workflow Plan** (plan file `reflective-stirring-cat.md`): Phase 2 (field-level provenance, side-by-side comparison UI) and Phase 3 (pipeline integration, auto-create lease review on DD stage, vetted rent roll → Deal Evaluator) not yet started
- **One Pager chart window branch** (`feat/onepager-chart-window`): Not merged, not deployed — documented in CLAUDE.md
- **New Business Deal Analysis**: Shared engine architecture designed but not implemented (see `new_business_pipeline.md`)
