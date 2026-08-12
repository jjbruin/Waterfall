# New Business Section - Design Document

## Phase 1: Deal Pipeline & Quick Evaluation

**Date:** August 12, 2026
**Status:** Design Review - Pending Team Input
**Author:** Jim Bruin / Engineering

---

## 1. Overview

### Purpose

The New Business section connects the front end of the deal sourcing and underwriting process to the existing asset management platform. Today, prospective deals are evaluated entirely in Excel — partner-provided Argus models, custom return spreadsheets, and manually assembled IC memos. When a deal closes, the underwritten data is re-keyed into MRI and the Projected IS file is manually loaded.

This design brings the evaluation workflow into the app so that:

- New deals can be quickly modeled using the same calculation engines that power asset management (waterfall, IRR, ROE, MOIC, debt service)
- Multiple scenarios can be saved and compared side-by-side
- When a deal closes, it onboards to the portfolio with one click — no re-keying
- The team has a shared pipeline view showing all prospective deals and their status
- Many more deals are evaluated than closed; the system accommodates high-volume screening

### Scope (Phase 1)

Phase 1 focuses on two core capabilities:

1. **Deal Pipeline** — Track prospective deals through stages from Lead to Closed/Passed
2. **Quick Deal Evaluator** — "Does this pencil?" calculator using existing engines

Future phases will add rent roll analysis, lease testing, Excel cash flow import, IC memo generation, and cross-portfolio analysis.

---

## 2. Deal Pipeline

### Pipeline Stages

| Stage | Description |
|-------|-------------|
| **Lead** | Opportunity identified; minimal info (name, location, price, broker) |
| **Screening** | Quick evaluation run; initial returns assessment |
| **LOI** | Letter of intent submitted or under negotiation |
| **Due Diligence** | PCA, environmental, title, legal, empire analysis underway |
| **IC Review** | Investment committee memo prepared and under review |
| **Closing** | Approved; legal documentation and funding in progress |
| **Closed** | Transaction completed; ready to onboard to portfolio |
| **Passed** | Deal declined (with reason captured) |

### Deal Record Fields

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| Deal Name | Text | Yes | e.g., "Vestavia Hills City Center" |
| Location | Text | Yes | City, State (deal-level summary) |
| Asset Type | Select | Yes | Retail, Multifamily, Office, Industrial, Mixed-Use |
| Partner Name | Text | No | e.g., "Burton Property Group" |
| Stage | Select | Yes | See stages above |
| Assigned To | Select | No | Team member |
| Target Close Date | Date | No | |
| Purchase Price | Currency | No | Total deal price; allocates down to properties |
| Source / Broker | Text | No | |
| Pass Reason | Text | No | Required when stage = Passed |
| Notes | Text | No | Free-form |

### Property Fields (per property within a deal)

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| Property Name | Text | Yes | e.g., "Windsor Square", "Westwood Plaza" |
| Address | Text | No | Street address |
| City / State / Zip | Text | No | |
| Asset Type | Select | No | Property-level; may differ from deal for mixed-use |
| GLA (SF) | Number | No | Gross leasable area |
| Units | Number | No | For multifamily |
| Year Built | Number | No | |
| Acreage | Number | No | |
| Allocated Price | Currency | No | This property's share of total purchase price |
| Occupancy | Percent | No | Current occupancy at evaluation time |
| In-Place NOI | Currency | No | Current NOI for this property |

A single-property deal has one property row that inherits the deal name. Multi-property
portfolio deals (like Burton with 7 properties) have multiple rows — GLA, NOI, and price
roll up to the deal level automatically.

### Entity & Investor Structure

The ownership structure being formed for the deal. Pre-planned IDs become real EntityIDs
and InvestorIDs on onboarding.

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| Entity Name | Text | Yes | e.g., "PPI-WS Holdings LLC" |
| Entity Type | Select | Yes | deal_jv, gp, lp, holding, property |
| Planned EntityID | Text | No | Pre-assigned ID for MRI (becomes InvestmentID) |
| Parent Entity | Select | No | For nested structures (JV → holding → property) |
| Ownership % | Percent | No | This entity's ownership share in parent |
| Role | Select | No | sponsor, investor, co_investor, manager |

Each entity contains investors:

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| Investor Name | Text | Yes | e.g., "PSC1", "KCREIT" |
| Planned InvestorID | Text | No | Pre-assigned ID (flows to waterfall PropCode) |
| Commitment | Currency | No | Capital commitment amount |
| Ownership % | Percent | No | Share within this entity |
| Investor Type | Select | No | pref_equity, op_equity, co_invest |

### Pipeline Views

**Kanban Board** — Cards organized by stage in columns. Drag-and-drop to advance stage. Cards show deal name, location, purchase price, and partner. Color-coded by asset type.

**Table View** — Sortable, filterable list with all fields. Toggle between Kanban and Table with a button. Default sort by last updated.

**Activity Log** — Every deal has a chronological log of stage changes, notes, and evaluation runs. Visible in the deal workspace.

---

## 3. Quick Deal Evaluator

### Concept

A single-page workspace where the user enters deal assumptions on the left and sees instant computed returns on the right. This replaces the Summary + Returns sheets from the current Excel workflow.

The evaluator uses the app's existing calculation engines — the same code that powers Deal Analysis, waterfall distributions, and partner returns. No separate calculation logic to maintain.

### Input Panel (Left Side)

#### Acquisition

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| Purchase Price | Currency | — | Required |
| Closing Costs | % or $ | 2% | Toggle between percentage and fixed dollar |
| CapEx at Close | Currency | 0 | Upfront capital expenditure budget |
| Total Cost Basis | Computed | — | Purchase + Closing Costs + CapEx (auto-calculated) |

#### Debt Assumptions

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| Loan Amount | $ or LTV% | 65% | Toggle between dollar amount and LTV percentage |
| Interest Rate | % | — | Fixed rate, OR enter as UST Yield + Spread |
| Term | Years | 7 | Loan term |
| I/O Period | Months | 60 | Interest-only period |
| Amortization | Years | 30 | After I/O period |
| Origination Fee | % | 0.25% | Of loan amount |

#### Equity Structure

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| PSC Equity % | % | 90% | PSC share of total equity |
| Partner Equity % | % | 10% | Auto-calculated as complement |
| Partner Name | Text | — | Pre-filled from deal record |

#### Partnership Terms

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| Preferred Return | % | 8.0% | Cumulative compounded annual pref rate |
| PSC Promote | % | 20% | PSC participation after pref is current |
| AM Fee | % | 0.95% | Annual fee on invested PSC capital |
| Annual Expenses | $ | 7,500 | Venture/partnership expenses |
| CF Distribution | Select | Pari-passu | How pref is distributed (pari-passu or PSC priority) |

#### Operating Assumptions

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| Year 1 NOI | Currency | — | Required |
| NOI Growth Rate | % | 2.0% | Annual growth applied to NOI |
| CapEx Reserve | $/SF/yr | $0.80 | Annual replacement reserves |
| *— OR —* | | | |
| Annual NOI Schedule | Table | — | Override: enter NOI per year manually |

**Note:** Future Phase will add Excel cash flow import to replace manual NOI entry with parsed Argus/partner model data.

#### Exit Assumptions

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| Hold Period | Years | 7 | Matches loan term by default |
| Exit Cap Rate | % | — | Required |
| Selling Costs | % | 2% | Of sale price |

### Results Panel (Right Side)

All results compute instantly when the user clicks "Evaluate" (or auto-compute on field change).

#### Key Metrics (KPI Cards)

| Metric | Description |
|--------|-------------|
| **Property IRR** | Unlevered IRR on total cost basis |
| **Levered IRR** | IRR on equity after debt service |
| **PSC IRR** | IRR to PSC after waterfall (pref, AM fee, promote) |
| **PSC Avg ROE** | Average annual return on equity to PSC |
| **PSC MOIC** | Multiple on invested capital to PSC |
| **Investor IRR** | IRR to investor (partner) net of PSC fees/promote |
| **Investor MOIC** | Multiple to investor net of fees |
| **Ops / Residual %** | Percentage of PSC returns from operations vs capital event |

#### Annual Summary Table

| Year | NOI | Debt Service | CFAD | Pref Due | Pref Paid | Promote | PSC CF | ROE |
|------|-----|-------------|------|----------|-----------|---------|--------|-----|
| 1 | ... | ... | ... | ... | ... | ... | ... | ... |
| 2 | ... | ... | ... | ... | ... | ... | ... | ... |
| ... | | | | | | | | |
| Exit | | | | Sale Proceeds | | | | |

#### Capital Stack Visualization

Horizontal stacked bar showing:
- Debt (blue) with LTV %
- PSC Preferred Equity (green) with exposure %
- Partner Equity (gray)
- PSC Exposure line (Debt + PSC PE as % of value)

#### Sensitivity Table

Matrix showing PSC IRR at different combinations of:
- **Rows:** Exit Cap Rate (e.g., 6.5%, 7.0%, 7.5%, 8.0%, 8.5%)
- **Columns:** Hold Period (e.g., 5yr, 6yr, 7yr, 8yr, 9yr, 10yr)

Highlights the base case cell. Color-codes cells (green > target IRR, yellow near target, red below).

#### Cash Flow Chart

Dual-axis chart:
- Bars: Annual NOI
- Line: PSC cash flow (cumulative or annual)
- Marker: Capital event (sale proceeds)

### Scenarios

Users can save multiple assumption sets per deal:

- **"Base Case"** — primary underwriting
- **"Downside - AMC Vacates"** — stress test
- **"7-Year Hold"** vs **"10-Year Hold"** — hold period comparison
- **"Standalone"** vs **"Crossed Portfolio"** — future phase

Each scenario saves all inputs and computed results. A comparison view shows scenarios side-by-side with delta highlighting.

---

## 4. Engine Integration

### How Existing Code Is Reused

The Quick Evaluator does NOT duplicate calculation logic. It builds the same data structures that `compute_deal_analysis()` expects, populated from form inputs instead of database tables:

| App Engine | Evaluator Usage |
|------------|----------------|
| `Loan` model (`models.py`) | Built from debt assumptions (rate, term, I/O, amort) |
| `build_loan_schedule()` (`loans.py`) | Generates amortization schedule for debt service |
| `InvestorState` (`models.py`) | Tracks PSC and partner capital, pref accrual |
| Waterfall steps (`waterfall.py`) | Dynamically built from partnership terms (Pref, Share, Promote) |
| `run_interleaved_waterfalls()` (`compute.py`) | Runs CF + Capital waterfalls on synthetic cash flows |
| `build_partner_results()` (`compute.py`) | Computes IRR, ROE, MOIC per partner |
| `xirr()` (`metrics.py`) | IRR calculation with irregular dates |
| `calculate_roe_detailed()` (`metrics.py`) | Weighted average capital ROE |
| `build_cash_flow_schedule_from_fad()` | Cash management / distributable cash |
| `projected_cap_rate_at_date()` | Exit valuation |

A new function, `build_prospect_analysis()`, will:

1. Generate a synthetic forecast DataFrame from NOI inputs (Year 1 + growth, or manual schedule)
2. Create a `Loan` object from debt assumptions
3. Build waterfall steps from partnership terms
4. Call `compute_deal_analysis()` with these synthetic inputs
5. Return the same result structure that Deal Analysis uses

This means the Evaluator's returns will always match what Deal Analysis shows after the deal is onboarded.

---

## 5. Onboarding to Portfolio (Deal Closes)

When a prospect's stage is set to "Closed," a wizard converts it to a portfolio deal:

| Step | Action | Target Table |
|------|--------|--------------|
| 1 | Assign parent vcode (auto or manual) | — |
| 2 | Create parent deal record | `inv` (deals) — `Investment_Name` = deal_name |
| 3 | Create property records (if multi-property) | `inv` (deals) — child rows with `Portfolio_Name` = deal_name |
| 4 | Assign property vcodes | Each property gets a unique vcode |
| 5 | Create entity relationships | `relationships` — from `prospect_entities` + `prospect_investors` |
| 6 | Create waterfall steps | `waterfalls` — from partnership terms + entity structure |
| 7 | Create loan record(s) | `loans` — from debt assumptions |
| 8 | Generate Projected IS entries | `forecast_feed` or `isbs_projected_is` — from prospect cashflows |
| 9 | Link prospect to portfolio deal | `prospect_deals.onboarded_vcode`, `prospect_properties.onboarded_vcode` |

**Single-property deal:** Steps 2-3 collapse — one `inv` row created, no `Portfolio_Name` linkage needed.

**Multi-property deal:** Parent row has `Property_Count > 1`. Each child row's `Portfolio_Name` = parent's `Investment_Name`. This mirrors existing portfolios like Burton (parent P0000109, children P0000113-P0000119 all with `Portfolio_Name = "Burton"`).

The wizard pre-fills all fields from the prospect's assumptions, properties, and entities. The user reviews and adjusts before confirming. After onboarding, the deal appears in Dashboard, Deal Analysis, One Pager, and all reports. Lease reviews remain linked to their properties — the `prospect_property_id` FK persists as a historical reference even after the property is onboarded to `inv`.

---

## 6. Data Architecture

### Design Principles

The new business data model mirrors the asset management structure so that onboarding is a
direct mapping rather than a re-keying exercise:

| New Business | Asset Management | Relationship |
|-------------|-----------------|--------------|
| `prospect_deals` | `inv` (parent deal) | 1:1 on close — creates parent deal row with vcode |
| `prospect_properties` | `inv` (child properties) | 1:1 per property — creates child rows with `Portfolio_Name` = deal name |
| `prospect_entities` | `relationships` + `waterfalls` | Entity/investor IDs flow to waterfall PropCode and relationship InvestorID |
| `lease_reviews` | — (no AM equivalent) | Due diligence artifact; linked to property via `prospect_property_id` |

A **single-property deal** has one `prospect_properties` row (the deal IS the property).
A **multi-property portfolio** has multiple rows — NOI, GLA, and purchase price roll up
from properties to the deal level, just as AM aggregates child properties into the parent.

### New Database Tables

```
prospect_deals
    id              INTEGER PRIMARY KEY AUTOINCREMENT
    deal_name       TEXT NOT NULL
    location        TEXT                -- City, State (deal-level summary)
    asset_type      TEXT                -- Primary asset type; properties may differ for mixed-use
    partner_name    TEXT
    source_broker   TEXT
    assigned_to     TEXT
    stage           TEXT DEFAULT 'lead'
    pass_reason     TEXT
    target_close    TEXT                -- ISO date
    purchase_price  REAL                -- Total deal purchase price
    closing_cost_pct REAL DEFAULT 0.02
    capex_at_close  REAL DEFAULT 0
    notes           TEXT
    onboarded_vcode TEXT                -- Links to parent inv row after close
    created_by      TEXT
    created_at      TEXT
    updated_at      TEXT

prospect_properties
    id              INTEGER PRIMARY KEY AUTOINCREMENT
    prospect_id     INTEGER NOT NULL REFERENCES prospect_deals(id)
    property_name   TEXT NOT NULL       -- e.g., "Windsor Square" or "Westwood Plaza"
    address         TEXT                -- Street address
    city            TEXT
    state           TEXT
    zip             TEXT
    asset_type      TEXT                -- Property-level type (may differ from deal for mixed-use)
    gla_sf          REAL                -- Gross leasable area (square feet)
    units           INTEGER             -- For multifamily
    year_built      INTEGER
    acreage         REAL
    property_price  REAL                -- Allocated purchase price for this property
    occupancy_pct   REAL                -- Current occupancy at time of evaluation
    noi_in_place    REAL                -- In-place NOI for this property
    notes           TEXT
    onboarded_vcode TEXT                -- Links to child inv row after close
    sort_order      INTEGER DEFAULT 0   -- Display ordering
    created_at      TEXT
    updated_at      TEXT

prospect_entities
    id              INTEGER PRIMARY KEY AUTOINCREMENT
    prospect_id     INTEGER NOT NULL REFERENCES prospect_deals(id)
    entity_name     TEXT NOT NULL       -- e.g., "PPI-WS Holdings LLC"
    entity_type     TEXT                -- 'deal_jv', 'gp', 'lp', 'holding', 'property'
    planned_entity_id TEXT              -- Pre-planned EntityID (becomes InvestmentID on onboard)
    parent_entity_id INTEGER            -- Self-ref FK for entity hierarchy (NULL = top-level)
    ownership_pct   REAL                -- Ownership percentage in parent entity
    role            TEXT                -- 'sponsor', 'investor', 'co_investor', 'manager'
    notes           TEXT
    created_at      TEXT
    updated_at      TEXT

prospect_investors
    id              INTEGER PRIMARY KEY AUTOINCREMENT
    entity_id       INTEGER NOT NULL REFERENCES prospect_entities(id)
    investor_name   TEXT NOT NULL       -- e.g., "PSC1", "KCREIT"
    planned_investor_id TEXT            -- Pre-planned InvestorID (flows to waterfall PropCode)
    commitment      REAL                -- Committed capital amount
    ownership_pct   REAL                -- Share within this entity
    investor_type   TEXT                -- 'pref_equity', 'op_equity', 'co_invest'
    notes           TEXT
    created_at      TEXT
    updated_at      TEXT

prospect_assumptions
    id              INTEGER PRIMARY KEY AUTOINCREMENT
    prospect_id     INTEGER REFERENCES prospect_deals(id)
    version         INTEGER DEFAULT 1
    version_label   TEXT DEFAULT 'Base Case'
    debt_amount     REAL
    debt_rate       REAL
    debt_term_months INTEGER DEFAULT 84
    io_months       INTEGER DEFAULT 60
    amort_months    INTEGER DEFAULT 360
    origination_fee REAL DEFAULT 0.0025
    psc_equity_pct  REAL DEFAULT 0.90
    pref_rate       REAL DEFAULT 0.08
    promote_pct     REAL DEFAULT 0.20
    am_fee_pct      REAL DEFAULT 0.0095
    annual_expenses REAL DEFAULT 7500
    exit_cap_rate   REAL
    selling_cost_pct REAL DEFAULT 0.02
    hold_years      INTEGER DEFAULT 7
    capex_reserve_psf REAL DEFAULT 0.80
    noi_year1       REAL
    noi_growth_rate REAL DEFAULT 0.02
    crossed_vcodes  TEXT                -- Future: comma-separated vcodes for cross-portfolio
    created_at      TEXT
    updated_at      TEXT

prospect_cashflows
    id              INTEGER PRIMARY KEY AUTOINCREMENT
    prospect_id     INTEGER REFERENCES prospect_deals(id)
    property_id     INTEGER REFERENCES prospect_properties(id)  -- NULL = deal-level
    version         INTEGER DEFAULT 1
    period_date     TEXT                -- ISO date
    revenue         REAL
    expenses        REAL
    noi             REAL
    capex           REAL
    other           REAL
    source          TEXT DEFAULT 'manual'

prospect_activity
    id              INTEGER PRIMARY KEY AUTOINCREMENT
    prospect_id     INTEGER REFERENCES prospect_deals(id)
    username        TEXT
    action          TEXT                -- created, stage_change, note, evaluated, property_added, etc.
    note            TEXT
    created_at      TEXT
```

#### Lease Review Integration

The existing `lease_reviews` table gains a nullable FK to `prospect_properties`:

```
lease_reviews (MODIFIED — add column)
    prospect_property_id  INTEGER REFERENCES prospect_properties(id)  -- nullable
```

This links a lease review to a specific property within a prospect deal. A property
may have zero or one lease review. When `prospect_property_id` is NULL, the review is
standalone (e.g., existing Windsor Square review created before the prospect pipeline).

The lease review's `property_name` and `property_address` are denormalized copies —
they match the prospect property at creation time but can be updated independently
(the review is a snapshot of due diligence, not a live reference).

#### Entity/Investor → Onboarding Mapping

On deal close, the onboard wizard maps prospect entities to AM structures:

| prospect_entities field | AM target |
|------------------------|-----------|
| `entity_name` | `relationships.Investment_Name` or entity label |
| `planned_entity_id` | `relationships.InvestmentID` (= `inv.InvestmentID`) |
| `entity_type='deal_jv'` | Parent deal row in `inv` |
| `entity_type='property'` | Child property rows in `inv` |
| `ownership_pct` | `relationships.Percent` |

| prospect_investors field | AM target |
|-------------------------|-----------|
| `planned_investor_id` | `waterfalls.PropCode` and `relationships.InvestorID` |
| `investor_name` | `relationships.Investor_Name` |
| `commitment` | Seeded into `capital_calls` or `accounting` |
| `investor_type='pref_equity'` | Waterfall Pref steps auto-generated |

All prospect tables will be added to `PROTECTED_TABLES` to prevent accidental overwrite during CSV imports.

### API Endpoints

```
Pipeline:
  GET    /api/prospects                        List all (filterable by stage, assigned_to)
  POST   /api/prospects                        Create new prospect
  GET    /api/prospects/<id>                    Get detail (includes properties + entities)
  PUT    /api/prospects/<id>                    Update (including stage changes)
  DELETE /api/prospects/<id>                    Archive/delete

Properties:
  GET    /api/prospects/<id>/properties         List properties for deal
  POST   /api/prospects/<id>/properties         Add property
  PUT    /api/prospects/<id>/properties/<pid>   Update property
  DELETE /api/prospects/<id>/properties/<pid>   Remove property

Entities & Investors:
  GET    /api/prospects/<id>/entities           List entities + investors
  POST   /api/prospects/<id>/entities           Add entity
  PUT    /api/prospects/<id>/entities/<eid>     Update entity
  DELETE /api/prospects/<id>/entities/<eid>     Remove entity
  POST   /api/prospects/<id>/entities/<eid>/investors    Add investor to entity
  PUT    /api/prospects/<id>/investors/<iid>              Update investor
  DELETE /api/prospects/<id>/investors/<iid>              Remove investor

Assumptions & Evaluation:
  GET    /api/prospects/<id>/assumptions        List scenarios
  POST   /api/prospects/<id>/assumptions        Save new scenario
  PUT    /api/prospects/<id>/assumptions/<v>     Update scenario
  POST   /api/prospects/<id>/evaluate           Run returns calculation
  POST   /api/prospects/<id>/evaluate/excel     Download results as Excel

Cash Flow Import (Future Phase):
  POST   /api/prospects/<id>/import-cf          Upload Excel cash flow

Lease Review (links to existing lease review system):
  POST   /api/prospects/<id>/properties/<pid>/lease-review   Create review for property
  GET    /api/prospects/<id>/lease-reviews                    List reviews for all properties

Onboarding:
  POST   /api/prospects/<id>/onboard            Convert to portfolio deal

Activity:
  GET    /api/prospects/<id>/activity           Activity log
  POST   /api/prospects/<id>/activity           Add note
```

---

## 7. User Interface Layout

### Pipeline View (default landing for New Business)

```
+------------------------------------------------------------------+
|  New Business                                    [+ New Deal]     |
|  [Kanban] [Table]                   [Filter: Stage] [Assigned To] |
+------------------------------------------------------------------+
| Lead       | Screening  | LOI        | DD         | IC    | ...  |
|            |            |            |            |       |      |
| +--------+ | +--------+ | +--------+ |            |       |      |
| |Vestavia| | |Park Pl | | |Meridian| |            |       |      |
| |Birm,AL | | |Atl, GA | | |Hou, TX | |            |       |      |
| |$76M    | | |$42M    | | |$31M    | |            |       |      |
| |Burton  | | |CBRE    | | |Greystar| |            |       |      |
| +--------+ | +--------+ | +--------+ |            |       |      |
|            |            |            |            |       |      |
+------------------------------------------------------------------+
```

### Deal Workspace (click into a deal)

```
+------------------------------------------------------------------+
|  < Back to Pipeline    Vestavia Hills City Center     [Passed v]  |
|  Birmingham, AL | Retail | 389,471 SF | Burton Property Group    |
+------------------------------------------------------------------+
|  ASSUMPTIONS                    |  RESULTS                       |
|                                 |                                |
|  Acquisition                    |  PSC IRR    PSC ROE   PSC MOIC |
|  Purchase Price   $76,000,000   |  [13.2%]   [9.3%]    [2.02x]  |
|  Closing Costs    2.0%          |                                |
|  CapEx at Close   $4,000,000    |  Inv IRR   Inv ROE   Inv MOIC |
|  Total Basis      $82,700,000   |  [11.3%]   [8.2%]    [1.84x]  |
|                                 |                                |
|  Debt                           |  Property IRR (Unlev)  [9.4%]  |
|  LTV              65%           |  Property IRR (Lev)   [13.9%]  |
|  Rate             6.15%         |  Ops/Residual      62% / 38%  |
|  Term             7 yrs         |                                |
|  I/O              60 mo         |  Annual Summary                |
|  Amort            30 yr         |  +---------------------------+ |
|                                 |  | Yr | NOI   | CFAD  | ROE | |
|  Equity                         |  | 1  | 5.6M  | 5.2M  |10.4%| |
|  PSC              90%           |  | 2  | 6.1M  | 5.4M  | 8.5%| |
|  Partner           10%          |  | 3  | 6.4M  | 5.5M  | 8.5%| |
|                                 |  | .. | ...   | ...   | ... | |
|  Partnership Terms              |  +---------------------------+ |
|  Pref Rate         8.0%         |                                |
|  Promote          20%           |  Capital Stack                 |
|  AM Fee           0.95%         |  [===Debt 63%===|=PSC 33%=|P4%]|
|  Expenses         $7,500        |  LTV: 65%  Exposure: 87.9%    |
|                                 |                                |
|  Operating                      |  Sensitivity (PSC IRR)         |
|  Year 1 NOI       $5,643,885   |  +---------------------------+ |
|  Growth            2.0%         |  |     | 6yr |7yr |8yr |9yr | |
|  CapEx Rsv        $0.80/SF     |  |7.0% |14.1 |13.8|13.5|13.2| |
|                                 |  |7.5% |13.5 |13.2|12.9|12.6| |
|  Exit                           |  |8.0% |12.9 |12.5|12.2|11.9| |
|  Hold Period       7 yrs        |  |8.5% |12.2 |11.9|11.5|11.2| |
|  Exit Cap          7.5%         |  +---------------------------+ |
|  Selling Costs     2.0%         |                                |
|                                 |                                |
|  [Evaluate]                     |  [Download Excel]              |
|                                 |                                |
|  Scenarios: [Base Case v] [+]   |                                |
+------------------------------------------------------------------+
|  Activity Log                                                    |
|  Aug 12 - Deal created by jbruin                                 |
|  Aug 12 - Stage: Lead -> Screening                               |
|  Aug 12 - Evaluated: Base Case (PSC IRR 13.2%)                  |
+------------------------------------------------------------------+
```

---

## 8. What This Replaces

| Current Excel Workflow | New Business App |
|----------------------|------------------|
| Summary sheet (manual entry of price, debt, equity) | Quick Evaluator form with auto-computed basis |
| TIAA/PSC Returns sheets (complex waterfall formulas) | Waterfall engine computes automatically |
| Multiple scenario tabs (4-Pack, Standalone, 7-Yr) | Saved scenarios with labels, side-by-side compare |
| Cash Flow sheet (Argus paste, 829 rows x 117 cols) | Future: Excel import with column mapping |
| Closing Costs sheet | Closing cost % input + CapEx budget |
| Debt sheet (amortization schedule) | Loan model with full amortization |
| Exit NOI Calculation sheet | NOI growth + exit cap rate |
| Manual copying of results to IC Memo tables | Export returns to Excel for IC Memo |
| Manual re-keying into MRI after close | One-click onboard wizard |
| No shared pipeline visibility | Kanban board visible to entire team |

---

## 9. Future Phases

### Phase 2: Cash Flow Import & Advanced Modeling
- Upload partner-provided Excel/Argus models
- Auto-detect and map revenue, expense, NOI rows
- Monthly cash flow support (not just annual)
- Tenant-level revenue assumptions
- CapEx schedule (itemized, not just $/SF)

### Phase 3: Rent Roll & Lease Testing
- Import rent roll from Excel
- Lease expiration analysis and rollover schedule
- Mark-to-market analysis (in-place vs market rents)
- Vacancy and downtime modeling
- Recovery/reimbursement audit

### Phase 4: IC Memo Generation
- Auto-populate memo template from deal data
- Sources & Uses table from assumptions
- Returns summary with sensitivity
- Capital stack diagram
- Export to formatted PDF

### Phase 5: Cross-Portfolio Analysis
- Cross a prospect with existing portfolio deals
- Combined capital accounts and waterfalls
- Portfolio-level exposure analysis
- Matches the "Four Pack" scenario from the current Excel

### Phase 6: Term Sheet Generator
- Generate term sheets from partnership terms
- Standard templates by deal type
- Version tracking and approval workflow

---

## 10. Implementation Priority

| Priority | Component | Effort | Dependencies |
|----------|-----------|--------|-------------|
| 1 | Database tables + API (CRUD) | 1 sprint | None |
| 2 | Pipeline view (Kanban + Table) | 1 sprint | API |
| 3 | Quick Evaluator (form + engine) | 2 sprints | API + existing engines |
| 4 | Scenario management | 0.5 sprint | Evaluator |
| 5 | Excel export of results | 0.5 sprint | Evaluator |
| 6 | Onboard wizard | 1 sprint | Evaluator + all tables |

---

## 11. Questions for the Team

1. **Pipeline stages** — Are the 8 stages (Lead through Closed/Passed) right? Are any missing? Should there be sub-stages (e.g., DD has environmental, title, legal)?

2. **Default assumptions** — Are these defaults reasonable for most deals? PSC 90%, Pref 8%, Promote 20%, AM Fee 0.95%, Expenses $7,500, Closing 2%, Selling 2%, LTV 65%, Hold 7yr, Growth 2%, CapEx $0.80/SF.

3. **Evaluation frequency** — Do you re-run returns frequently as you refine assumptions, or is it more of a "run once, save, move on" workflow? This affects whether we auto-compute on every field change or use an explicit "Evaluate" button.

4. **Scenario naming** — What are the most common scenarios you model? (e.g., Base/Downside/Upside, different hold periods, crossed vs standalone, different exit caps)

5. **Who accesses this?** — Should all users see the pipeline, or only New Business team members? Should analysts be able to edit assumptions, or view-only?

6. **Crossing timing** — At what stage do you typically model the crossed portfolio returns? Is it always at IC stage, or earlier during screening?

7. **What data from the Excel model do you wish you didn't have to re-enter?** What are the biggest pain points in the current process?
