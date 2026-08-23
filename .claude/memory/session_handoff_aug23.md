# Session Handoff — Aug 23, 2026

## What Was Done

### 1. Committed & Deployed Prior Session Work
- Prior session froze ("Lollygagging...") with uncommitted changes on disk
- Verified all 5 modified files intact, committed as `8af6cee`, pushed, deployed v307
- Changes: capital budget builder, horizontal cashflow parser, enhanced pipeline property cards

### 2. Fixed Assumptions Save 500 Error
- **Problem**: Vue sent 19 new loan/sizing fields (lender, rate_type, extension_count, prepay_schedule, sizing constraints, etc.) not in DB table or `ASSUMPTION_FIELDS`
- **Fix**: Added 19 columns to `prospect_assumptions` DDL + migration ALTER TABLE + updated `ASSUMPTION_FIELDS` (now 37 fields)
- Refactored `list_assumptions()` and `get_assumption()` to use `ASSUMPTION_FIELDS` dynamically instead of fragile positional indexing (`_assumption_row_to_dict()` helper)
- **Commit**: `53044d7`, deployed v308

### 3. Enhanced Waterfall Steps View
- Steps tab now shows **CF_WF** and **Cap_WF** in separate bordered cards with:
  - Section headers with descriptions ("Operating distributions — does NOT reduce capital outstanding" / "Refi / sale proceeds — DOES reduce capital outstanding")
  - Colored step type badges (Pref=green, Initial=blue, Share=orange, Tag=red)
  - Explanation panel describing how Share/Tag split works
- **Commit**: `44f2cbe`, deployed v309

### 4. Capital Budget Persistence
- **Problem**: Capital budget (Uses + Sources) lived only in Vue refs, lost on page refresh
- **Fix**: Added `capital_uses_json` and `capital_sources_json` TEXT columns to `prospect_assumptions`
- Save serializes all Uses line items + debt Sources + equity split as JSON
- Load merges saved amounts into default structure (forward-compatible with new items added later)
- **Commit**: `1162ee6`, deployed v310

### 5. Documentation Updates
- CLAUDE.md: Updated Prospect Deal Analysis section with capital budget, assumption fields, waterfall builder details, cashflow status API
- CLAUDE.md: Updated cashflow_parser.py docs with horizontal layout detection
- CLAUDE.md: Added new Key Functions entries
- MEMORY.md: Updated revision to v310
- new_business_pipeline.md: Marked Prospect Deal Analysis as IMPLEMENTED, updated status

## Deployed State
- **Revision**: v310 (deployed Aug 23, 2026)
- **All 4 code commits + 1 doc commit pushed to main and deployed**

## Key Files Modified
- `cashflow_parser.py` — horizontal Excel layout detection + parsing (~255 lines added)
- `flask_app/services/prospect_service.py` — 21 new assumption columns, migration, dynamic queries
- `flask_app/api/prospects.py` — `/cashflow-status` batch endpoint
- `vue_app/src/views/ProspectAnalysisView.vue` — capital budget, waterfall builder, persistence (~1,800 lines)
- `vue_app/src/views/PipelineView.vue` — enhanced property cards with Argus/Excel badges

## Open Items / Future Work
- Scenario side-by-side comparison view
- Sensitivity matrix (cap rate × hold period)
- One-click onboarding (prospect → portfolio deal)
- IC memo generation
- One Pager chart window branch (`feat/onepager-chart-window`) — not merged, not deployed
