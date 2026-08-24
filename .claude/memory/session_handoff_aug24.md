# Session Handoff — Aug 24, 2026

## What Was Done

### 1. Independent CF and Capital Waterfall Builders
- **Problem**: Single waterfall step list generated both CF_WF and Cap_WF by mirroring, with no way to customize them independently (e.g., Cap_WF needs Return of Capital steps, CF_WF doesn't)
- **Fix**: Split `wfStepInputs` into `cfStepInputs` and `capStepInputs` refs. Two-tab builder UI with independent editing per waterfall type.
- **"Copy from CF" helper**: Copies CF steps to Cap and auto-inserts Return of Capital steps for each entity before the residual split
- **Backend**: `build_deal_waterfall()` now accepts `{cf_steps, cap_steps}` directly instead of converting from investor objects. Each step has `{entity_id, step_type, rate, amount}`. Legacy `investors` format still accepted for backward compat.
- **`_convert_step_inputs()` helper**: Converts UI step inputs to waterfall DataFrame rows — handles pref, return_of_capital, residual, fixed_amount, irr_lookback
- **Separate residual validation**: `cfResidualPct` and `capResidualPct` computed separately with independent warnings
- **Commit**: `c29f210`, files: `prospects.py`, `ProspectAnalysisView.vue`

### 2. IRR Lookback Step Type
- **Feature**: New step type `irr_lookback` for laddered promote structures with IRR hurdle gates between residual split tiers (e.g., 90/10 up to 9% IRR, 80/20 to 13%, 70/30 above)
- **UI**: Added to `STEP_TYPES` array with label "IRR Lookback", input "Target IRR (%)"
- **Backend**: Maps to `vState='IRR'`, `nPercent=rate`, `vtranstype='IRR Hurdle'`
- **Stored→input conversion**: `_storedToInputs()` recognizes `vState='IRR'` and converts back to `irr_lookback` inputs
- **Commit**: `edce5b1`, files: `prospects.py`, `ProspectAnalysisView.vue`

### 3. Auto-Save Assumptions on Build & Analyze
- **Problem**: Capital budget (Sources & Uses) was only persisted when clicking "Save Assumptions" explicitly. Building waterfall or running analysis without saving first meant data was lost on page refresh.
- **Fix**: `buildAndSaveWaterfall()` and `runAnalysis()` now call `await saveAssumptions()` before proceeding
- **Commit**: `1e47ad6`, file: `ProspectAnalysisView.vue`

### 4. Vue Ref Auto-Unwrap Fix
- **Problem**: Add/Remove step buttons stopped working after splitting into `cfStepInputs`/`capStepInputs`. Vue 3 templates auto-unwrap refs, so `addWfStep(cfStepInputs)` passed a plain array, but the function tried to access `.value`.
- **Fix**: Changed `addWfStep()` and `removeWfStep()` parameter types from `Ref<WfStepInput[]>` to `WfStepInput[]`, removed `.value` access
- **Commit**: `9bfe9d3`, file: `ProspectAnalysisView.vue`

## Deployed State
- **Revision**: v314 (deployed Aug 24, 2026)
- **All 4 commits pushed to main and deployed**

## Key Files Modified
- `flask_app/api/prospects.py` — `build_deal_waterfall()` rewritten for independent CF/Cap step inputs + IRR Lookback
- `vue_app/src/views/ProspectAnalysisView.vue` — split step lists, two-tab builder, Copy from CF, IRR Lookback, auto-save, ref fix

## Open Items / Future Work
- Scenario side-by-side comparison view
- Sensitivity matrix (cap rate x hold period)
- One-click onboarding (prospect -> portfolio deal)
- IC memo generation
- One Pager chart window branch (`feat/onepager-chart-window`) — not merged, not deployed
