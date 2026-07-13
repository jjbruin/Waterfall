# Session Handoff — May 7, 2026 (Session 2)

## What Was Done

### 1. Deleted TGA23 CF Waterfall Step 3 (Default)
- **What**: Removed iOrder=3 rows from `waterfalls` table in local SQLite for vcode=TGA23, vAmtType=6.02(c)
  - TGAM Default FXRate=0.9
  - INV23 Tag FXRate=0.1
- **Why**: User identified as modeling error — Default capital return step in CF waterfall doesn't belong
- **Cap waterfall step 3 (6.03(c)) left intact** — only CF was the error
- **IMPORTANT**: Only deleted from local `waterfall.db`. Azure PostgreSQL still has the old rows. Need to run equivalent DELETE on Azure PG before deploying.

### 2. Fixed Entity Pref Accrual ($0 → Real Values)
- **Root cause**: `run_recursive_upstream_waterfalls()` created empty InvestorState objects with `capital_outstanding=0`. With no capital, `accrue_pool_pref()` computes zero accrual, so Pref/Def_Int steps always pay $0.
- **Fix (3 parts)**:
  - `_build_entity_seeded_states(entity_id, acct)` in `portfolio_analysis_service.py` — new function that builds InvestorStates with net capital from entity accounting (TGAM=$102M, INV23=$11.6M for TGA23)
  - `pre_seeded_states` parameter added to `run_recursive_upstream_waterfalls()` in `waterfall.py` line 1885 — uses seeded dict instead of empty `{}`
  - `accrue_all_pools(stt, p_date)` call added in `run_upstream_waterfall_period()` at waterfall.py ~line 1553 — accrues pref to period date before processing steps
- **Result**: TGA23 Def_Int 18% step now allocates ~$1.7-1.9M/yr to TGAM and ~$190K/yr to INV23. Pref 8% step is $0 because the 18% default pref (higher priority) consumes all available CF — correct waterfall mechanics.

### 3. Combined CF/Cap Allocation Tables
- **What**: Replaced two separate `<table>` elements with one combined table in `PortfolioAnalysisView.vue`
- **Layout**: Single `<table>` with `<thead>` (Step + year columns), then two `<tbody>` sections:
  - "CF Waterfall (Section 6.02)" header row (grey background)
  - CF data rows
  - "Capital Waterfall (Section 6.03)" header row
  - Cap data rows
- **Year columns aligned vertically** between CF and Cap sections
- **CSS**: `.wf-section-header` (grey bg), `.wf-section-label` (bold 13px left-aligned)

## Modified Files (Uncommitted)
1. **`waterfall.py`** — `pre_seeded_states` param + `accrue_all_pools()` call (~11 lines changed)
2. **`flask_app/services/portfolio_analysis_service.py`** — `_build_entity_seeded_states()` function + wiring (~302 lines changed, includes prior session's allocation table enrichments)
3. **`vue_app/src/views/PortfolioAnalysisView.vue`** — Combined table + CSS (~138 lines changed, includes prior session's step labels and row styling)

## What Needs To Happen Next
1. **Browser test** the combined allocation table (user hasn't visually verified yet)
2. **Commit** all 3 modified files
3. ~~Delete TGA23 CF step 3 from Azure PostgreSQL~~ — **DONE** (deleted 2 rows, verified)
4. **Deploy to Azure**: `az acr build ... --no-logs .` then `az containerapp update ...`

## Running State
- Flask: `source .venv/Scripts/activate && python -m flask_app.run` (port 5000)
- Vue: `cd vue_app && npm run dev` (port 5173, proxies to Flask)
- Auth endpoint: `POST /auth/login` (not `/api/auth/login` — no `/api` prefix in Flask dev)
- Login: admin/admin

## Key Context for Continuing
- TGA23 entity waterfall has 18% Def_Int (step 1) as highest priority, 8% Pref (step 2) below it. With $102M TGAM capital, the 18% accrual vastly exceeds annual CF, so Pref 8% gets $0 and residual Share steps get very little. This is correct behavior.
- The `pre_seeded_states` parameter is backward-compatible (defaults to None). PSCKOC and other callers of `run_recursive_upstream_waterfalls` are unaffected.
- Proposed mode doesn't pass `pre_seeded_states` — it replaces the entity waterfall entirely, so seeding isn't needed there.
