# Valuations & NAV Audit Packages — Design + Phase 1 (Aug 28, 2026)

**Status: Phase 1 BUILT on branch `feat/valuations-phase1` — not merged, not deployed.**

## Phase 1 implementation (Aug 28, 2026)
- **Tables** (SQLite + PG via `ensure_valuation_tables()`, all in PROTECTED_TABLES):
  `valuation_cycles`, `valuation_records` (UNIQUE(cycle_id, vcode)),
  `valuation_documents` (blob store, SHA-256 dedup per record), `valuation_comments`
  (UNIQUE(record_id, section); sections: budget_review, balance_sheet, general).
- **Service** `flask_app/services/valuation_service.py`: cycle create/seed (idempotent
  reseed picks up new deals), policy classification (`_classify`: dev→cost, <12mo→cost,
  PE>=5M→third_party, else internal; PE funded nets signed contributions — no abs()),
  children inherit parent classification via `_child_parent_map` (Property_Count>=1 =
  parent; Portfolio_Name matches parent Investment_Name OR parent Portfolio_Name —
  Burton exception), record CRUD + status actions (open/signed_off/excluded),
  documents, comments, `import_argus()` (wraps argus_service with
  import_type='valuation', label '{year} Valuation', links import_id to record),
  `get_budget_review()` (3-col: cycle-yr Estimate = YTD actual + budget remainder /
  next-yr Budget / Argus Val yr 1 — reuses financials_service._calculate_is_amounts +
  IS_ACCOUNTS; principal per-source: BS-change+budget / 7060 / argus 7060),
  `get_balance_sheet()` (prior Dec-31 vs latest Interim BS by vAccountType/vAccount/
  vDescription).
- **API** `flask_app/api/valuations.py` at /api/valuations (registered in __init__.py;
  ensure hooks in both PG and SQLite startup blocks).
- **Vue** `ValuationsView.vue` at /valuations (sidebar: AM section between Surveillance
  and One Pager). Dashboard (cycle selector, clickable summary cards, parent/child
  indented table) + workspace (?record=id): Assumptions & Docs / Budget Review /
  Balance Sheet tabs, doc upload one-file-per-request, Argus upload, comments, analyst
  sign-off, print (all tabs forced visible via .tab-panel !important over v-show).
- **Verified locally**: 2025 test cycle seeded 84 records (53 third_party/16 internal/
  15 cost pre-override); Asbury budget-review Budget column ties the printed 2025 form
  EXACTLY (816,863 / 1,140,073 / 733,416 / 264,000 / 106,800); BS ties (AR 150,860,
  mortgage 5,351,478). Test cycle left in local waterfall.db (records reset clean).
- **.claude/launch.json** added (flask-api :5000, vue-dev :5173).
- **Notes**: `valuations` table name was taken (MRI_Val feed) — new tables prefixed
  `valuation_*`. Argus upload of the *audit package* workbook fails parse ("Could not
  detect monthly periods") — correct behavior; upload the raw Argus export.
- **Phase 2 next**: committee analyses 1&2, parallel unanimous approvals
  (valuation_approvals), CCO recorder, snapshots, committee workbook export.

Full design proposal published as artifact:
https://claude.ai/code/artifact/b30ce007-5715-4bd6-b3a1-5f7206843144
Source documents reviewed: Valuation Policy (9/18/25), 2025 Valuation Summary Report - LIVE.xlsx,
Asbury Commons appraisal (25-16252-000-A2-NDA), Asbury Review Form 2025.pdf,
"asbury cashflows and loader 2025.xlsx" (the auditor NAV package). All under
`OneDrive - peaceablestreet.com\Documents - Peaceable Street Capital\Asset Mgmt\4. Consolidated Asset Documents\Valuations\2025\`.

## What it is
New Asset Management section to manage annual property valuation review/approval
(Valuation Committee, ASC 820) and produce per-deal NAV audit package Excels for the auditors.

## Policy rules (encode as per-cycle classification, overridable with note)
- 3rd-party appraisal required: invested PE ≥ $5M at initial investment AND held ≥ 12 months.
- Cost basis: held < 12 months, in development, or lease-up (full valuation at earlier of
  stabilization or 12 months from substantial completion).
- Internal valuation (income cap / DCF): < $5M PE.
- Overrides need work papers + committee approval; all decisions documented.

## Key mapping of existing engines → new uses
- **NAV calc = Cap_WF run once at valuation date** on hypothetical liquidation proceeds:
  value − ISBS BS debt + current assets − current liabilities = net proceeds →
  `seed_states_from_accounting()` through 12/31 → `run_waterfall()` on Cap_WF
  (accrued pref, ROC, IRR lookback vState, residual split). Asbury 2025 verified:
  9,400,000 − 5,257,764.58 + 534,372.68 − 135,801.64 = 4,540,806.46 → PSC NAV 2,340,159.27.
- **Accrued_Pref / OP_Pref tabs = `build_pref_balance_detail()`** — identical columns,
  Act/Act, quarter-end markers, 12/31 compounding. Asbury accrued 26,452.60 ties.
- **IRR_Lookbacks tab** = cashflow history + terminal pref+capital row solved at lookback
  rate(s) (12%/15% on Asbury); live =XIRR formulas like sold_service workbooks.
- **Argus import** already exists; new = tag imports to cycle, stage `Val_IS_{year}` rows,
  publish to forecast_feed only on approval (replaces the manual "Loader" Excel).
- **Review Form (3 pages)** = Property Financials I/S comparison (val yr-1 vs current-yr
  estimate vs proposed budget + analyst comments) + One Pager BS/cap-stack/ownership +
  Tenant Roster lease maturity profile. Print layout per One Pager pattern.
- **Committee workbook** = two live analyses: (1) pref balance + accrual → Pref NAV vs
  prior year; (2) method/caps/discount/NOI/value/debt/net-proceeds YoY with up/down flags.
  Prior-year columns come from `valuations` table (MRI_Val already has every column:
  vMethod, fCapRate, nTermCapRate, nDiscountRateForEquityInterest, mAnnualNOI,
  mIncomeCapConcludedValue, mDebtValue, mEquityValue, mMezzanineValue = PE NAV, nCostSaleRate).
- **Approval workflow** = reuse review_submissions/review_notes/review_roles with a
  `doc_type='valuation'` discriminator + valuation committee roles + snapshot on approval.

## Proposed pipeline (per deal per cycle)
Classified → Docs & Assumptions → Argus Imported → Review Form → Analyst Sign-off →
Committee Review → Approved → NAV Package → Published (valuations row + forecast_feed).

## New tables (all PROTECTED)
`valuation_cycles`, `valuation_records` (unit of work: classification, method, value, caps,
discount, NOI, cost-of-sale, appraiser, argus_import_id, status), `valuation_documents`
(appraisal/argus/llc_excerpt/bs_support, SHA-256 dedup), `valuation_nav_results`
(walk JSON + psc_nav/op_nav), `valuation_comments`, forecast staging.
Add optional `agreement_ref` per waterfall step (LLC section citations like 8.2(a)–(g))
for the auditor walk.

## Phases
1. Foundation: tables, classification, dashboard, workspace, Argus-to-cycle, review form.
2. Committee: analyses 1&2, review-engine extension, snapshots, committee workbook export.
3. NAV: compute_nav(), agreement refs, OP pref, package Excel (8 tabs), publish step.
4. AI assist: appraisal PDF extraction (lease-review pattern), tie-out checks, assistant tools.

## Decisions (Jim, Aug 28, 2026)
1. **App is system of record** for valuations — MRI has no valuation entry screen, no
   write-back. `valuations` → PROTECTED_TABLES, remove from MRI refresh registry.
2. **Current assets/liabilities**: app SUGGESTS inclusion set (default classification),
   asset manager curates per BS line item (add/remove checkboxes) — OP reporting varies.
   Selections persist on the NAV result, carry forward to next cycle, YoY treatment
   changes flagged. Consistency is the stated priority.
3. **Pref engine**: mimic the ROE Audit's DATA SPINE, not its math (see recommendation
   below — reviewed and accepted direction with one amendment).
4. **IRR lookbacks live in the `waterfalls` table** as IRR steps — NAV reads the deal's
   own structure. Rollout includes completeness audit vs the MRI Rates feed.
5. **Cost basis = accounting capital balances + accrued pref at valuation date** —
   computed, not entered.
6. **Portfolios**: appraisals evaluated per CHILD property (each child gets a valuation
   record); NAV rolled up and run ONCE at the PARENT (consolidated BS + parent waterfall).
7. **Committee = President, CEO, CIO — unanimous, NOT sequential. CCO records.**
   → parallel `valuation_approvals` table (approved when all 3 active approvals; return
   clears them); CCO = recorder role (notes, closes record), not an approver.
8. **Pegasus dual-tranche**: one property valuation; waterfall order does the split —
   Pref A takes value to its 10% IRR (IRR step), remainder flows to Pref B NAV.
   No special-case code; package shows one NAV line per tranche.

## Pref recommendation (reviewed against code, Aug 28)
Three implementations exist:
- **ROE Audit** (`build_roe_timeline()` compute_service.py:202): spine = `cashflow_details`
  from `get_cached_deal_result()` — ONLY one that merges actual accounting + projected
  waterfall distributions (needed for projected 12/31 balances when valuing in November).
  But math is weak for audit: simple daily accrual bal×rate×days/365, NO annual
  compounding, Act/365F, pref payments detected by Description text. Rates per partner
  from waterfall Pref steps via `_get_pref_rates()`.
- **Pref Balance Detail** (`build_pref_balance_detail()` reports_service.py:972):
  Excel-exact — Act/Act (`_days_in_year`), 12/31 compounding (Inv+Comp), same-date
  ordering, rate priority deal_terms.pe_coupon > waterfall. Accounting-only, no projection.
- **Waterfall engine** (InvestorState): Act/365F, 12/31 compounding with 45-day grace.

**The auditor package's Accrued_Pref tab is the Pref Balance Detail math** (verified:
Asbury 2020 leap-year rows divide by 366; Compounded Pref column present; ends 26,452.60
PSC / 14,243.71 OP at 12/31/25).

**Recommendation (accepted direction): refactor `build_pref_balance_detail()` to accept an
event list.** Report keeps passing accounting events (unchanged); NAV service passes the
ROE Audit's merged actual+projected event list through 12/31. One walk serves PSC and OP
(OP rate from OP's waterfall Pref step). NAV walk injects these balances into seeded
InvestorStates before `run_waterfall()` so NAV_Calc ties to the pref tabs to the penny —
do NOT let the engine re-accrue under its own convention (Act/365F + grace ≠ Excel Act/Act).

## Follow-up tasks
1. IRR-step completeness audit (every deal with a lookback in its package has the Cap_WF
   IRR step at the right rate; Asbury solves 12% AND 15% — check multi-tier handling).
2. Pref walk refactor + tie-out vs 2025 packages (Asbury first test case).
3. OP Pref step coverage audit (OP walk needs a Pref step on the OP investor).
4. Draft default BS classification map (account ranges + vInput text patterns) from the
   2025 packages; curation absorbs exceptions.
5. Protect `valuations` (PROTECTED_TABLES + MRI refresh registry + CSV importer) before
   first cycle publishes.

## Notes / gotchas discovered
- Excel package PSC NAV (2,340,159.27) vs MRI mMezzanineValue (2,332,693.20) differ from
  data vintage — computing both in-app from one source removes that discrepancy class.
- Valuation Policy PDF is scanned (no text layer) — 3 pages, image extraction needed.
- Summary workbook "Rates" tab = MRI feed of PE Coupon / PE Split / IRR Lookback /
  Purchase & UW Exit Cap Rate per deal (vtranstype rows) — possible authoritative rate source.
