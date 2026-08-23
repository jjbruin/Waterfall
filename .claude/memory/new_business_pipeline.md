# New Business Pipeline — Session Handoff (Aug 13, 2026)

## What Was Built

### Lease Review System (prior sessions + Aug 12-13)
- **Backend**: 10-table schema (`lease_reviews`, `lease_tenants`, `lease_documents`, `lease_rent_steps`, `lease_cotenancy`, `lease_cotenancy_refs`, `lease_exclusive_use`, `lease_options`, `lease_validation`, `lease_field_resolutions`)
- **PDF Extraction**: Claude API extracts lease terms from PDFs via `extract_lease_terms_via_api()` in `lease_review_service.py`
- **Three-way Validation**: Rent Roll vs Lease, Argus vs Lease, Cotenancy Schedule vs Lease
- **Vue Frontend**: `LeaseReviewView.vue` — 5 tabs (Overview, Expirations, Validation, Co-Tenancy, Scenarios) with ECharts
- **Windsor Square data**: review_id=1, 49 tenants, 128 documents extracted, 680 rent steps, 208 validations (local only)
- **Rent Roll Upload** (Aug 12-13): `parse_rent_roll_flexible()` handles Argus-format Excel (two-row headers), standard Excel, CSV with fuzzy column matching. `import_rent_roll_to_review()` replaces tenants. Upload endpoint: `POST /reviews/<id>/upload-rent-roll`
- **Manual Review Creation**: `POST /reviews/create` — no folder scanning needed (Azure-compatible)
- **Pipeline Property Linking** (Aug 13): New Review modal has "Link to Pipeline Property" dropdown. Lists all prospect properties without existing reviews. Auto-fills name/address/GLA. Stores `prospect_property_id` FK.
  - API: `GET /api/lease-review/prospect-properties` — joins prospect_properties + prospect_deals

### Prospect Pipeline (Aug 12)
- **7 new tables**: `prospect_deals`, `prospect_properties`, `prospect_entities`, `prospect_investors`, `prospect_assumptions`, `prospect_cashflows`, `prospect_activity`
- **Service**: `flask_app/services/prospect_service.py` — full CRUD + DDL + activity logging
- **API**: `flask_app/api/prospects.py` — registered as `prospects_bp` at `/api/prospects`
- **Vue Pipeline**: `PipelineView.vue` — Kanban board + table view + deal workspace slide-out
- **Lease Review Link**: `lease_reviews.prospect_property_id` FK connects reviews to properties
- **All tables in PROTECTED_TABLES** in `database.py`

### Bug Fix (Aug 13)
- `prospects.py` and `lease_review.py` used `request.user` but auth decorator sets `g.current_user` — caused 500 on deal creation. Fixed, deployed v245.

### Design Doc
- `docs/New_Business_Design_Phase1.md` — updated with property layer, entity/investor structure, onboarding mapping, API endpoints

## Data Model Summary

```
prospect_deals (deal/portfolio level)
  └── prospect_properties (individual properties)
       └── lease_reviews (via prospect_property_id FK)
  └── prospect_entities (ownership structure)
       └── prospect_investors (investor participation)
  └── prospect_assumptions (scenario versions)
  └── prospect_cashflows (property or deal level)
  └── prospect_activity (log)
```

Mirrors AM: `prospect_deals` → parent `inv`, `prospect_properties` → child `inv` with `Portfolio_Name`

## Current State (deployed v310)
- Pipeline view live at `/pipeline` under New Business sidebar section
- Kanban drag-and-drop, table view, new deal modal, deal workspace with Properties/Entities/Activity tabs
- Properties have "Start Lease Review" / "View Lease Review" links
- Entities support planned EntityID/InvestorID pre-assignment
- Lease Review 7-step workflow stepper (Setup → Import Rent Roll → Upload Documents → AI Extraction → Validation → Analyst Review → Complete)
- Non-destructive rent roll merge + destructive replace option
- Document upload: **Select Files** (individual) and **Select Folder** (entire directory tree via `webkitdirectory`). Subfolder names used as tenant matching hints. SHA-256 hash dedup.
- Extraction dedup on re-runs (composite key checks)
- Per-tenant approval workflow (approve/flag/reset)
- Lease Risk Analysis (`/lease-risk-analysis`): 7-tab analysis view using analyst-resolved data
  - Field resolution via `lease_field_resolutions` table (UPSERT pattern)
  - Overview (KPIs + editable tenant roster), Expirations, Validation, Co-Tenancy Risk, Scenarios, Exclusive Use, Options
  - Inline field editing: double-click to override, "R" badge, revert button
  - One-click "Use Seller" / "Use Lease" on validation mismatches

## Prospect Deal Analysis (IMPLEMENTED — v310)

Standalone route at `/prospect-analysis`. Full deal analysis with shared compute engines.

### What's Built
- **Capital Budget**: 13-item Uses table + 4 debt Sources + computed equity gap (PE/OP split)
- **37 Assumption Fields**: Loan terms, extension, prepay, sizing constraints, earnout/guarantor notes
- **Capital Budget Persistence**: `capital_uses_json` + `capital_sources_json` in prospect_assumptions
- **Waterfall Builder**: Flexible step types (Pref, ROC, Residual, Fixed Amount) with entity selector
- **Steps Preview**: CF_WF and Cap_WF shown in separate bordered cards with colored badges and descriptions
- **Shared Engine**: `build_prospect_analysis()` → `compute_deal_analysis()` with same waterfall/XIRR/ROE
- **Cashflow Status**: `GET /api/prospects/<id>/cashflow-status` batch endpoint with Argus/Excel badges + timestamps
- **Horizontal Parser**: `cashflow_parser.py` detects and parses transposed Excel layouts (dates across columns)

### Key Architecture
The engines in `compute.py`, `waterfall.py`, `metrics.py` expect specific DataFrame schemas. The mapping layer (`build_prospect_analysis()` in `prospect_analysis.py`) produces conforming DataFrames from prospect inputs. This is a translation layer, NOT a new engine.

## What's Further Out
- Scenario side-by-side comparison view
- Sensitivity matrix (varying cap rate × hold period)
- IC memo generation
- One-click onboarding (prospect → portfolio deal)
- Cross-portfolio analysis
- Term sheet generator

## Key Files
- `flask_app/services/prospect_service.py` — all business logic + DDL
- `flask_app/api/prospects.py` — REST endpoints
- `flask_app/services/lease_review_service.py` — lease review (9 tables, extraction, validation)
- `flask_app/api/lease_review.py` — lease review API
- `vue_app/src/views/PipelineView.vue` — pipeline UI (Kanban + table + deal workspace)
- `vue_app/src/views/LeaseReviewView.vue` — lease review UI (7-step workflow stepper)
- `vue_app/src/views/LeaseRiskAnalysisView.vue` — risk analysis UI (7 tabs, field resolution)
- `docs/New_Business_Design_Phase1.md` — design document

## Windsor Square Model Analysis (Aug 12)
- **Three-layer waterfall**: Deal Level (OP WPG 30% ↔ PE PSC 70%, 9% pref, 40/60 CF, IRR hurdles 9%→13.5%), PSC/TIAA JV (PSC 10% ↔ TIAA 90%, 9% coupon, 0.95% AM fee, 20% promote), Ambassador's Fund (PSC 4.27% ↔ Fund 5.73%)
- **Argus rent roll**: Two-row headers (row 14 = category prefixes, row 15 = column names), 45 tenants, 656K SF, $7.4M annual rent
- **Tabs**: PSC (OP↔PE waterfall), TIAA_AMB (PE↔Investor waterfall), Tenant Rent Roll, Inputs (Argus monthly D180-ER251), Bifurcated CFs (monthly by phase), Dashboard (sources/uses)
- **UW files location**: `C:\Users\jbruin\OneDrive - peaceablestreet.com\Documents - Peaceable Street Capital\New Business\Windsor Square - Matthews, NC\UW`
