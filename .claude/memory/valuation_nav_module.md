# Valuations & NAV Audit Packages — Design + Phases 1-4 (Aug 28, 2026)

**Status: ALL FOUR PHASES DEPLOYED — merged to main as `c6083f0` (with Charlene's
`a060c59` One Pager PE terms fallback) and live on Azure as revision `v397`
(Aug 28, 2026). PG tables created cleanly on first boot; /api/valuations live.
No cycle opened in production yet — create the cycle from the Valuations page
and assign the committee roles (president/ceo/cio) in Settings.**

## Phase 4 implementation (Aug 28, 2026)
- **Tie-out checks** (`run_record_checks` in valuation_nav_service): Cap_WF present,
  Pref step present, IRR step vs deal_terms.irr_lookback (the completeness audit,
  automated per record), assumptions complete (children-rollup aware), children values
  complete, appraisal doc + Argus link required for third_party, LLC excerpt reminder,
  open questions, AI cross-check mismatches, value vs NOI/cap sanity (±10%), Argus
  year-1 NOI vs entered NOI (±5%, REVENUE_ACCTS|EXPENSE_ACCTS over first 12 months of
  the import), BS snapshot staleness (>92 days), BS treatment changed vs prior cycle,
  NAV staleness vs record edits, PSC NAV vs prior-year published mezz (>25% swing →
  info). Endpoint GET /records/<id>/checks; Vue: expandable "Tie-out checks" strip in
  the workspace header, auto-loaded per record.
- **Apply extracted values**: AI Summary cross-check table gains per-row "apply" +
  "Apply All Extracted Values" (also fills appraiser firm + appraisal date). Uses the
  existing PUT; only while the record is open.
- **Assistant tools**: `get_valuation_cycle` (status board + NAVs) and
  `get_valuation_detail` (record + NAV walk + checks + Q&A) added to
  assistant_service (22 tools now); system prompt mentions valuation cycles.

## Phase 3 implementation (Aug 28, 2026)
- **NAV engine** (`flask_app/services/valuation_nav_service.py`): value − ISBS BS debt
  + curated current assets − curated current liabilities = net proceeds →
  `run_waterfall(Cap_WF)` at the cycle as-of on states seeded from accounting with the
  **Excel-exact pref walk injected** (`build_pref_balance_detail` per Cap_WF Pref
  investor; new `pref_rate_override` param so OP investors use their own step rate, not
  deal_terms.pe_coupon; tiers zeroed + `last_accrual_date = as_of` so the engine can't
  re-accrue under Act/365F+grace). PSC NAV = non-OP allocations; walk persisted to
  `valuation_nav_results` (inputs_json + walk_json).
- **VERIFIED vs the Asbury Excel package** (net proceeds 4,540,806.46): pref 26,452.60 /
  capital 1,490,000 / OP pref 14,243.71 / OP capital 802,308 all EXACT. IRR lookback
  223,912.82 vs Excel 230,522.89 — **date convention, not an error**: the manual
  workbook discounts distributions at month-end "IRR Dates"; the engine uses actual
  payment dates (acquisition fees excluded per app-wide policy). XNPV closed form
  reproduces both conventions exactly. A note to this effect is emitted in every NAV
  result with IRR steps.
- **`irr_needed_distribution()` in waterfall.py REPLACED with the closed form**
  `x = −XNPV(target, cfs) × (1+target)^t` (metrics.xnpv, Act/365). Equivalent to the
  old brentq-over-xirr search (verified to the penny on Asbury) but exact and immune to
  solver noise. Blast radius: PSCKOC / Portfolio Analysis / prospect IRR gates may
  shift by solver-precision amounts (improvement).
- **BS curation**: `valuation_bs_selections` (record_id, account, included). Defaults:
  assets 1000-1199 in; liabilities 2000-2149 in except DEBT_BS_ACCTS; equity never.
  Stored > prior-cycle carry-forward > default; `changed_vs_prior` flag per line; debt
  rows locked. Consolidated across parent+child vcodes at each vcode's latest BS ≤ as-of.
- **Cost-basis derivation**: when classification=cost and no value entered, value =
  debt + Σ seeded capital_outstanding + Σ accrued pref (components reported).
  Portfolio parents: entered value > Σ children concluded values; walk runs at parent.
- **Agreement refs**: `valuation_step_refs` (vcode, wf_type, iorder UNIQUE) — survives
  waterfall re-saves; editable inline on the NAV walk (e.g. 8.2(a)…8.2(g) on Asbury).
- **Auditor package Excel** (`generate_nav_package`): NAV_Calc / Bal_Sht (with Included
  + changed flags) / Accrued_Pref + OP_Pref (Excel-exact columns) / IRR_Lookbacks
  (cashflow history + NAV-walk terminal rows + live `_xlfn.XIRR` check) / LLC_Waterfall
  (steps + refs + uploaded excerpt list) / Loader (Val_IS rows). Cycle-level zip
  endpoint. Committee view gains "Download NAV Packages (zip)".
- **Publish** (`publish_record`, admin, requires approved + computed NAV): writes the
  `valuations` row (mEquityValue = net proceeds, mMezzanineValue = PSC NAV; deletes
  same-date rows by PARSED date — MRI rows store '12/31/2025 0:00', publishes store
  ISO) + inserts Val_IS_{year} rows into `forecasts` (Date 'M/D/YYYY', mAmount =
  −amount_norm → MRI sign convention, Pro_Yr = year − PRO_YR_BASE; verified signs match
  the real Loader tab) + stamps published_at/by + refresh_table('valuations'/'forecasts')
  + clear_cache(vcode). **System-of-record cutover shipped**: 'valuations' added to
  PROTECTED_TABLES; MRI_VAL query set to download-only (target_table=None) in
  mri_service QUERY_REGISTRY.
- **Committee integration**: Analysis 1 pref_nav = stored psc_nav; Analysis 2
  net_proceeds = stored NAV net proceeds when computed (has_nav flag), else
  value−debt estimate. Snapshots now freeze the NAV too.
- **API adds**: GET/records/<id>/nav (inputs+result), PUT bs-selections,
  POST nav/compute, PUT /step-refs, GET nav-package (xlsx), GET cycles/<id>/nav-packages
  (zip), POST records/<id>/publish (admin).
- **Vue**: NAV tab (summary facts, walk with editable ref column, notes, pref balances,
  BS curation checkboxes with 'changed' badges + Save & Recompute), Publish button in
  the header (admin, approved, shows Published stamp after).
- **Follow-ups**: verify a real Argus export end-to-end (synthetic rows tested the
  publish path); IRR-step completeness audit across deals (Rates feed vs Cap_WF);
  multi-tier lookbacks (Asbury Excel also shows a 15% solve — only the 12% step is in
  the waterfall); PG smoke test of new DDL on first Azure deploy.

## Phase 2 implementation (Aug 28, 2026)
- **New tables** (added to `_VALUATION_DDL`, protected): `valuation_questions`
  (Q→single editable answer, status open/answered/resolved), `valuation_approvals`
  (member_role, action approve/return, active flag — history kept), `valuation_snapshots`
  (record_id UNIQUE, full JSON freeze), `valuation_ai_summaries` (record_id UNIQUE).
- **Committee = President/CEO/CIO, parallel unanimous; CCO = recorder** (approves
  nothing, can return). Roles reuse the shared `review_roles` table — **'cio' added to
  REVIEW_ROLE_NAMES** in review_service.py so admins assign it in Settings. A user
  holding multiple committee roles approves all of them in one click. All 3 active
  approvals → status 'approved' + snapshot (record/budget-review/balance-sheet JSON);
  approved records are locked (`_require_not_approved` on every write); return requires
  a note, clears approvals, deletes the snapshot, back to 'open'. Batch approve-all
  endpoint per cycle.
- **Q&A**: any logged-in user asks (asker's committee/recorder role labeled);
  analyst/admin answers (re-answer updates); answered→resolved by AM or committee.
  Open-question counts on dashboard + committee views. Editable through review,
  frozen at approval (comments too — `commentsEditable` = not approved).
- **Committee summary** (`get_committee_summary`): Analysis 1 (pref balance + accrued
  pref via `build_roe_summary_row` from reports_service at the cycle as-of; prior Pref
  NAV from mri_val mMezzanineValue; current Pref NAV = Phase 3) + Analysis 2 (method/
  cap/exit cap/disc/NOI/value YoY vs mri_val prior year, debt via get_isbs_debt_balance,
  net proceeds = value − debt estimate pending NAV walk, up/down + rate deltas).
  **Ties verified to the 2025 workbook**: Asbury 1,490,000/26,452.60/1,516,452.60 and
  Giant 7 20,200,000/352,808.22 exact. 2-sheet Excel export (`generate_committee_workbook`).
- **AI appraisal summary** (`valuation_ai_service.py`): pulls latest appraisal blob,
  PyMuPDF text extraction (pdfplumber fallback; scanned PDFs rejected with clear msg,
  cap 350k chars), claude-sonnet-4-6 → structured JSON (exec summary, approach,
  key_assumptions, value_conclusion, market/rent/positives/risks bullets, extraordinary
  assumptions, in-place income). Stored per record; regenerate replaces. Server-side
  **cross-checks extracted vs entered** assumptions (6 fields, tolerance-based).
  JSON parse via `raw_decode` of first object (model may append commentary).
  **Verified on the real 170-page Asbury appraisal: all 6 checks match** (9.4M, 7.25%,
  7.5%, 8.5%, 2%, NOI 706,041≈709,790), narrative correctly captures DCF-primary
  reconciliation + Winn-Dixie flat-option NOI suppression. ~75s per run.
- **API adds**: /permissions, records/<id>/questions + questions/<id>/answer|resolve,
  records/<id>/approve|return|snapshot, cycles/<id>/committee-summary|committee-excel|
  approve-all, records/<id>/ai-summary GET/POST.
- **Vue**: Records|Committee Summary toggle on dashboard (analyses tables, workbook
  download, approve-all, click-through); workspace committee chips (✓/○ per role) +
  Approve/Return (role-gated via /permissions); Q&A tab; AI Summary tab (facts grid,
  cross-check table with match/differs badges, regenerate). Q/Appr. columns on records
  table.
- **Local test state**: admin holds 'cio' review role in local waterfall.db; Asbury
  record carries the real appraisal PDF + generated AI summary + real 2025 assumptions.
- **Phase 3 next**: NAV engine (compute_nav, BS curation UI, pref walk refactor,
  agreement refs), auditor package Excel, publish step (valuations row + forecast_feed
  + protect `valuations` from MRI refresh).

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
