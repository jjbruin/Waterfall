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

## Single source of truth (Jim's rule, Aug 26 2026)
One home per fact; everything else derives. Applied so far:
- **Purchase price**: the property is the atom (`prospect_properties.property_price`). The deal's purchase price and the Capital Budget line derive as the sum when any property is priced (read-only, tagged "Sigma properties" in the UI); `prospect_deals.purchase_price` is only a fallback for deals with no priced properties yet.
- **Cash flows**: per-property (Argus `NP{property_id}` imports / Excel versions) roll up to deal level via `get_property_rollup_forecast_df()`; waterfalls always run against the deal-level rollup.
- **Investor identity**: `prospect_investors.planned_investor_id` + ownership % is the declaration; waterfall-shape inference is the fallback, never the master.
- **Naming**: the assumptions-version picker in the header is labelled "Assumptions", not "Scenario" -- Scenario is exclusively the results-panel scenario system.
Known remaining duplicates to consolidate: `prospect_deals.purchase_price` still written from Pipeline forms; assumption versions vs scenarios overlap (versions may eventually fold into scenarios).

## Planned refinancing + property-bound parcels (Aug 26 2026)
- **Planned refi (NB)**: `prospect_assumptions.planned_refi_json` {enabled, refi_date, loan_amount, rate, term/amort/io years, closing_costs, holdback}. Builds a synthetic accepted `prospective_loans_raw` row; the AM machinery does the rest (loan replacement, proceeds through Cap_WF, capital-call flag, sale extended to new maturity). Warns loudly when the refi maturity passes the forecast end (terminal value would compute from empty data).
- **Pre-existing AM bug fixed in compute.py**: an accepted refi dropped the replaced loan's entire schedule, losing all pre-refi debt service (FAD/DSCR overstated from close to refi). Replaced loans' pre-refi rows are now restored and closed with a curtailment-marked retirement row (excluded from balloon/sale-payoff double counting).
- **Parcels belong to a property, not a deal** (Jim): `parcel_sales.property_vcode` binds each sale to its property; the AM card offers the deal's child properties. Effects still aggregate at deal level where the waterfall runs.
- Parcel Sales + Sale Override boxes on AM Deal Analysis now render on deal select (were gated behind hasResult, i.e. invisible until Compute).

## Waterfall step pairing + gated promote tiers (Aug 26 2026)
The engine pairs a Share with its Tags only on IDENTICAL iOrder (or shared vAmtType) -- AM convention, e.g. P0000006 step 12/12. The NB builder was writing residual steps at sequential orders, so the lead took its share alone and every Tag allocated zero. Fixed in _convert_step_inputs: consecutive residual steps share one level order; any other step type ends the run.
**Gated two-tier promote** ("X/Y until partner hits IRR target, then A/B"): encode tier 1 as the gated partner's IRR step AS LEAD with FXRate = its share (step_max = level * fx, capped by irr_needed_distribution), partner Tag at the same iOrder for the other share; tier 2 is a plain Share/Tag pair at the next order. Windsor Square N0000003 Cap_WF is the reference: IRR 9 / IRR 9 / [IRR 13.5 fx.4 + Tag .6]@30 / [Share .75 + Tag .25]@40. Verified: 40/60 exact pre-gate, 75/25 exact when the 13.5 binds. Builder UI expresses the gate since Aug 26: the IRR Lookback step has a 'share %' field (FXRate on the lead); residual steps after a gated IRR tag at its level until shares reach 100%, then the next residual opens a fresh level. Verified against the AM cascade pattern (30/70@14.5 -> 24/76@14.6 -> 21/79 final).

## Waterfall persistence: Compute Returns never writes (Aug 26 2026)
The stored waterfall kept scrambling after page reloads: runAnalysis() auto-built
and SAVED the Builder rows on every Compute Returns, and _storedToInputs() hydrated
those rows lossily (dropped iOrder/Tie #, collapsed Share+Tag to generic residuals),
so the close-at-100 heuristic regrouped the ties on the next save. One reload + one
Compute Returns mispaired multi-tier structures (e.g. 40/60-until-13.5 + 75/25).
Fixed in c5d0b81 (v375):
- Compute Returns READS the stored waterfall only. The explicit "Build & Save
  Waterfall" button is the sole write path. Do not reintroduce auto-save on analyze.
- _storedToInputs() is lossless: level = iOrder on every row, lead-first order
  within a tie (save path makes the first residual at a level the Share). Round-trip
  through /waterfall/build verified as an identity on Windsor's 10 steps.
If a scrambled structure is found: correct rows are in the waterfall_audit backups
and scratchpad windsor_waterfall_backup*.json; the reference Cap_WF shape is in the
"Waterfall step pairing" section above.

## Aug 27 2026: timing, overrides, parcel + scenario integration (v380-v385)
- **Equity funds at close**: seed_date = close_date - 1 day, actuals_through moves
  with it (prospect_analysis). Backdating to Dec 31 of the prior year accrued ~9
  months of phantom pref and stretched ROE years. Close date falls back to the
  cash-flow source start when target_close is blank (Windsor = 10/1/2026).
- **Sale = month-end preceding the hold anniversary** (exactly hold*12 months).
  Terminal NOI column in the annual forecast = forward 12 months from the
  un-truncated forecast; ties to sale_dbg.NOI_12m_After_Sale by construction.
- **compute_capital_budget()** is the single source for total cost/equity
  (mirrors the app's Sources & Uses incl. grossed-up PSC fee); CapEx Reserve
  use seeds beginning cash via beginning_cash_override.
- **Operating overrides run on Argus too** (_apply_operating_overrides_df):
  mgmt_fee_pct replaces 5040 with gross rev x pct; replacement_reserve_psf adds
  monthly 5092 (own "Replacement Reserve" line under Management Fee, NB only).
- **Manual parcel distributions are runner events** (build_manual_parcel_events
  + manual_events on run_interleaved_waterfalls): pref accrues to the parcel
  date on the pre-cut balance, then the ROC reduces pools — pref for all later
  periods accrues on reduced capital, same as waterfall mode. The old
  "pref slightly overstated" limitation is gone.
- **NB parcel context**: loans from the saved Capital Budget
  (get_prospect_deal_loans, ids match _build_loans), tenants from the
  scenario-pinned/active Argus rent roll else the lease-review roster with
  analyst resolutions (get_prospect_tenants). Multi-tenant picker sums into
  4010. Parcel rent shows as "Less: {name}" under Rental Income via
  result['parcel_revenue_detail'].
- **Argus imports auto-surface as scenarios** (ensure_import_scenarios,
  idempotent on scenario listing; first active-pinning scenario becomes Base
  Case — also anchors the parcel tenant picker's rent roll).
- **NB ROE/MOIC audits at AM parity**; XIRR Cash Flows sorted by date.
- **Deploy race lesson**: background `git push` in a compound command can be
  rejected while the ACR build proceeds from a stale local branch — always
  `git pull --rebase` BEFORE building, and verify the built SHA is on
  origin/main before deploying (v376/v378 were built off-main; v377/v379
  superseded them minutes later).
