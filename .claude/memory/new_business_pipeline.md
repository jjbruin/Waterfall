# New Business Pipeline — Session Handoff (Aug 12, 2026)

## What Was Built

### Lease Review System (prior sessions + this one)
- **Backend**: 9-table schema (`lease_reviews`, `lease_tenants`, `lease_documents`, `lease_rent_steps`, `lease_cotenancy`, `lease_cotenancy_refs`, `lease_exclusive_use`, `lease_options`, `lease_validation`)
- **PDF Extraction**: Claude API extracts lease terms from PDFs via `extract_lease_terms_via_api()` in `lease_review_service.py`
- **Three-way Validation**: Rent Roll vs Lease, Argus vs Lease, Cotenancy Schedule vs Lease
- **Vue Frontend**: `LeaseReviewView.vue` — 5 tabs (Overview, Expirations, Validation, Co-Tenancy, Scenarios) with ECharts
- **Windsor Square data**: review_id=1, 49 tenants, 128 documents extracted, 680 rent steps, 208 validations

### Prospect Pipeline (this session)
- **7 new tables**: `prospect_deals`, `prospect_properties`, `prospect_entities`, `prospect_investors`, `prospect_assumptions`, `prospect_cashflows`, `prospect_activity`
- **Service**: `flask_app/services/prospect_service.py` — full CRUD + DDL + activity logging
- **API**: `flask_app/api/prospects.py` — registered as `prospects_bp` at `/api/prospects`
- **Vue Pipeline**: `PipelineView.vue` — Kanban board + table view + deal workspace slide-out
- **Lease Review Link**: `lease_reviews.prospect_property_id` FK connects reviews to properties
- **All tables in PROTECTED_TABLES** in `database.py`

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

## Current State (deployed v235)
- Pipeline view live at `/pipeline` under New Business sidebar section
- Kanban drag-and-drop, table view, new deal modal, deal workspace with Properties/Entities/Activity tabs
- Properties have "Start Lease Review" / "View Lease Review" links
- Entities support planned EntityID/InvestorID pre-assignment

## What's Next (not started)
1. **Quick Deal Evaluator** — input panel (acquisition, debt, equity, operating, exit) + results panel (IRR/ROE/MOIC, annual summary, cap stack, sensitivity table). Uses existing engines via `build_prospect_analysis()` wrapper
2. **Scenario management** — save/compare multiple assumption sets per deal
3. **Excel export** of evaluation results
4. **Onboarding wizard** — convert closed prospect to portfolio deal (create inv rows, waterfalls, relationships)
5. **Cash flow import** (Phase 2) — upload partner/Argus Excel models

## Key Files
- `flask_app/services/prospect_service.py` — all business logic + DDL
- `flask_app/api/prospects.py` — REST endpoints
- `flask_app/services/lease_review_service.py` — lease review (9 tables, extraction, validation)
- `flask_app/api/lease_review.py` — lease review API
- `vue_app/src/views/PipelineView.vue` — pipeline UI (Kanban + table + deal workspace)
- `vue_app/src/views/LeaseReviewView.vue` — lease review UI (5 tabs)
- `docs/New_Business_Design_Phase1.md` — design document
