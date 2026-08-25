# PORTFOLIO SNAPSHOT — FINAL STATE / DEPLOY HANDOFF

> **STATUS: MERGED TO `main` AND DEPLOYED AND LIVE.**
> Superseded the earlier "handed to Jim for deploy" state. The feature was
> merged at **`41c8e4a`** ("Merge wip/portfolio-snapshot-performance into main")
> and all snapshot work is now in `origin/main`. Live builds were rotating
> rapidly through the evening of 2026-08-24.
> Read "SESSION 2026-08-24 (post-deploy)" immediately below first; the rest of
> the file is the original build record and remains accurate as history.

Written 2026-08-24. Read `portfolio_snapshot_build_spec.md` (commit `010d1e9`)
for the original plan; this file is the build record, the deploy handoff, and now
the post-deploy log.

---

## SESSION 2026-08-24 (post-deploy) — READ THIS FIRST

### Deployed state

- **Merged and live.** `origin/main` carries the feature via merge `41c8e4a`.
  Jim then pushed a run of small fixes on top (`65f45e9`, `1f679c9`, `13da6ba`,
  `12484df` …). **Every one of my 16 snapshot commits is contained in
  `origin/main`** — verified with `git merge-base --is-ancestor`.
- Live container builds rotated repeatedly that evening
  (`f7692533e77c` → `3a0426c1bd03` → `7b0aed6fdb0c` → `e4e80de6410f` →
  `150a89f70ace` …). **Always re-check `/api/data/version` and confirm it is
  stable across several probes before trusting any captured figure.** One probe
  window hit read timeouts mid-redeploy.
- The earlier scare that this branch "deleted Jim's cash-flow work" was a
  **stale-base artefact, not an isolation violation**: my 16 commits were
  17 added / 4 modified / **0 deleted**, and `cashflow_parser.py` /
  `ProspectAnalysisView.vue` never existed on my base. Diffing *newer main →
  older branch* reports files that only exist on the newer side as deletions.
  The real merge preserved everything. Only genuine overlap was the 4 additive
  shared touches, with one one-line `PROTECTED_TABLES` union to resolve.

### Efficiency optimisations

- **#1 lean One Pager provider — DONE by Jim (`1f679c9`).** Replaces
  `get_one_pager_data` with direct `get_capitalization_stack` +
  `get_property_performance` calls. **Audited output-neutral**: the snapshot
  reads only `cap_stack` (financial/loan/summary) and `property_performance`
  (loan/operating); `pe_performance`, `general` and the payload's `comments` are
  read **nowhere**. The one field Jim omits, `pe_yield_on_exposure`, is also read
  nowhere. Arg lists match, including `inv_map=`.
- **#3 batch `pe_cap_comments` — DONE by Jim (`1f679c9`).** Pure efficiency. Two
  behaviours that must stay: only non-empty comments are emitted, and failure
  must degrade to `{}` (so the cross-check reports it could not run) rather than
  raising.
- **#2 direct-ISBS quarterly NOI — NOT DONE. Jim is building/verifying it.**
  `_quarterly_noi_provider` is still the Property Financials chart pipeline.
  **Baseline handed over**:
  `~/Downloads/debt_yield_baseline_TIAA_20260824-200432_CLEAN.csv` plus its
  `_README.txt` (build `e4e80de6410f`, git HEAD `13da6ba`).
  **Gate: must match 26Q1 AND 26Q2 byte-for-byte, including every `None`.**
  - Q1 alone is **not** a sufficient test: at Q1 (`months_elapsed=3`)
    single-period NOI **equals** YTD for all 18 populated deals, so a YTD-shaped
    implementation passes by accident. At Q2 (`months_elapsed=6`) the two differ
    on **all 19** populated deals, by up to **+5.88pp** (Plaza Del Mar; Mount
    Prospect +4.52, Poplar Prairie +3.30, Evergreen +2.47).
  - 11 `None` rows per quarter, and **the None sets differ between quarters**.
    **Giant 7 (`P0000019`) is the key trap** — portfolio parent whose NOI sits on
    child vcodes with nothing under its own, so the chart pipeline correctly
    returns `None`; a naive three-month sum under the parent returns 0 or invents
    a value. Negative dev-deal NOI (Jefferson Eastchase/Waters Creek/Addison
    Heights) must not be clamped.

### Comma-parse bug — FIXED AND DEPLOYED

Five Summary deals were failing on `could not convert string to float:
'266,698'` and losing funded pref + commitment. **All 5 now parse, and zero
flags remain across the 32 Summary rows**:

| Deal | funded / committed (deal level) |
|---|---|
| Green Valley Ranch & Telluride | 20,000,000 / 20,000,000 |
| ReNew Glenmoore | 20,750,000 / 20,750,000 |
| Town Fair Tire Portfolio | 9,050,000 / 9,050,000 |
| Burton Portfolio | 26,597,500 / 54,227,500 |
| Trolley Square | 4,190,216 / 6,750,000 |

Verified TIAA/26Q1 only — other investors and quarters not re-checked.

### PDF reconciliation — TIAA 26Q1

**Funded now essentially ties: `402,096,813` live vs `404,200,000` PDF
(−0.52%)**, up from −$74.6M before the comma fix.

| Asset type | Live funded | PDF | Delta |
|---|---|---|---|
| Multifamily | 240,411,006 | 240,410,995 | **+$11** |
| Retail | 117,414,277 | 117,414,277 | **exact** |
| Office | 12,243,598 | 12,243,598 | **exact** |
| **Self-Storage** | **32,027,932** | **34,172,689** | **−2,144,757** |

**← NEXT TASK.** The **entire** remaining funded gap is the Self-Storage bucket,
2 deals — **likely Pegasus Life Storage, which also shows `n/a` debt on the Loan
subtab.** Small, well-bounded chase.

**Committed is +$31.0M over the PDF (476.1M vs 445.1M) — PRE-EXISTING, NOT A
BUG.** Already documented in the `portfolio_snapshot_summary.py` docstring under
"WHAT DOES NOT TIE, AND IS FLAGGED RATHER THAN FUDGED". Driven by **unfunded
commitments on dev deals**; `diagnostics['committed_gap_attribution']` gives
per-deal numbers (Burton 24.87M, JB Fair Park 19.42M, Jefferson Stephens 17.88M,
Brainerd 8.41M; East Manchester runs −2.72M the other way). By asset type:
Retail +22.1M, Multifamily +11.0M, Self-Storage −2.1M. **This is a definitional
question for Charlene + Jim — how should unfunded dev commitments count — not a
defect.** Reported as computed, never bent toward the PDF.

**Trap when reconciling:** the PDF's Deal Type pie ($153.1M / $131.3M /
$119.8M) is **FUNDED dollars, not committed** — they sum to $404.2M. Only the
**asset-type** committed comparison is valid. Deal Type funded ties well:
Value-Add −2.95M (contains the Self-Storage gap), Income +0.85M, New
Construction −$1,988.

### Open threads at 2026-08-24 close

1. **NEXT: the Self-Storage −$2,144,757 funded gap.** 2 deals, likely Pegasus
   Life Storage (also `n/a` debt on Loan). Now the only funded discrepancy.
2. **Camp Creek YTD-DSCR: live 1.854x vs PDF 1.7x.** Separate, unrelated to the
   efficiency work, comes from `property_performance.dscr` (which optimisation #1
   keeps). Worth a look.
3. **Accrued pref fix** — see the RESOLVED/NEXT TASK section further down. OREI
   is settled as correct; Apple Self Storage + Plaza Del Mar need their own terms
   checked before any code change.
4. **Full PDF figure-by-figure reconciliation + flag cleanup — DEFERRED.** Needs
   a PDF reader in the venv: there is **no poppler and no `fitz`/`pymupdf`/
   `pdfplumber`/`pypdf`**, so the 26Q1 PDF could not be read at all. Every PDF
   figure used today came from anchors quoted in conversation, not extracted.
   Candidate file: `~/Downloads/3-31-26 - Portfolio Report Appendix- TIAA_revised.pdf`.
   **Also note: snapshot "flags" are DERIVED, not stored** — there are only
   `portfolio_snapshot_{comments,footnotes,values,frozen}` tables and no flag
   table. Flags are recomputed per row from data conditions ("no loan record",
   "no commitment row", "One Pager unavailable"). They are **data-availability
   and provenance notes, not reconciliation-vs-PDF claims**, so "clear the flag
   because the figure matches the PDF" is not a supported operation and would
   destroy real lineage information. Fix the data or the guardrail logic instead.
5. **Print function matching the PDF format — DEFERRED.**
6. **Only TIAA is validated and name-resolved** (`TGAM` → "TIAA" alias).
   **KOC and Declaration still need the MRI name query + validation.**

### Working process with Jim (agreed)

Both push to `origin` and leave a one-line ping. **Pull latest before starting
anything**, and sync every session — `origin/main` moved **five times** during
this one session (`986881f` → … → `12484df`), and Jim edits the same snapshot
Vue components. `git pull --ff-only origin main` is the safe form.

---

## Branch stack

```
main  986881f
└─ wip/portfolio-snapshot-spec   010d1e9   build spec (docs only)
   └─ wip/portfolio-snapshot-step1  4d2e90f   foundation service
      └─ wip/portfolio-snapshot-step2  cbe23bf   persistence + approval pipeline
         └─ wip/portfolio-snapshot-step3  0b7a809   Operating subtab assembly
            └─ wip/portfolio-snapshot-step4  a0fe820   Loan subtab assembly
               └─ wip/portfolio-snapshot-step5  13ff99f   Financial subtab assembly
                  └─ (resume note, a18881a)
                     └─ wip/portfolio-snapshot-step6   b771995  Step 6a Summary
                                                                + Loan dev-display
                                                       1301b2a  Step 1 acquisition gate
                        └─ wip/portfolio-snapshot-step7  a462c01  UI shell + 4 Vue
                           └─ wip/portfolio-snapshot-step8   36eb326  guardrails
                              └─ wip/portfolio-snapshot-freeze  77bbafc  snapshot freeze
                                                                0206edf  reopen + freeze UI
                                 └─ wip/portfolio-snapshot-performance
                                                                c4c2cbc  perf  ← DEPLOY THIS
```

**One linear lineage, 15 commits off `main`, zero merges.** Verified: every one of
the 11 `wip/portfolio-snapshot-*` branches is an ancestor of `c4c2cbc`, so the
other ten names are just bookmarks into the same chain. Nothing needs
consolidating — deploy `c4c2cbc` and you have everything.

| Step | Commit | What landed | Self-test |
|---|---|---|---|
| Spec | `010d1e92ce5fca5ff11e27b42af096be9bf8401c` | `portfolio_snapshot_build_spec.md` | — |
| 1 | `4d2e90f7d07344e05609fa960546c83fbc79a7b4` | `portfolio_snapshot_service.py` — deal resolution, fund grouping, look-through %, quarter-aware sold exclusion, child roll-up, investor names | 14/14 |
| 2 | `cbe23bf82c4db3a1a80c11ba287eb619f9f33dfa` | `portfolio_snapshot_persistence.py` + 3 tables + `PROTECTED_TABLES` (additive, 32→35) | 35/35 |
| 3a | `0b7a8097211a494e202caac16ad4c6b06f09607a` | `portfolio_snapshot_operating.py` — Econ Occ, NOI ×3, Expected/Actual Growth | 10/10 structure |
| 4a | `a0fe820bac6154b0f1112689b3e96b9bcc9b1b87` | `portfolio_snapshot_loan.py` — Debt, LTV, YTD DSCR, Debt Yield, Rate/Maturity | 21/26 (see below) |
| 5a | `13ff99f7937848a9805dd0f87027ad84ddef401e` | `portfolio_snapshot_financial.py` — cap-stack zone, 4 scaled TIAA columns, manual Net ROE/ITD, footnotes | 28/28 |
| 6a | `b771995a4e2cb2904417730623a860817bf10396` | `portfolio_snapshot_summary.py` — asset + deal-type allocations, 2 blank narratives; plus Loan dev-display refinements (Waters Creek LTV exception, Lifecycle dev detection, Pegasus debt-None n/a gate) | Summary 41/43, Loan 45/46 |
| 1-fix | `1301b2aadbe044b8c19a542f0501438e0e6e919d` | acquisition-date gate in `portfolio_snapshot_service.py` — the ownership window's missing half | 25/25 |
| **7** | `a462c01d935e75229ff8718ae2970f51c155c415` | **UI shell + 4 Vue components + blueprint + nav — the feature is now DEPLOYABLE** | app starts, 13 routes; frontend unverified (see below) |
| 8 | `36eb32674990d2239cb4c390fce466305f6a9c63` | `portfolio_snapshot_guardrails.py` — missing-data audit, aggregate integrity, pref-equity cross-check, ownership completeness, quarter window | 22/22 |
| freeze | `77bbafc4eafe5392254fd7fb0dd2a2a8c04767fb` | `portfolio_snapshot_freeze.py` + `portfolio_snapshot_frozen` table — approved reports become immutable; `reopen()` added | 34/34 |
| freeze UI | `0206edf449e25d3cfa3ec7b0976bc1b0f0557cce` | `/reopen` route, `can_reopen`, frozen-vs-live banner, reopen button | route verified |
| **10** | `c4c2cbcc698ff0c5efd012e83a4bbc19e733b58f` | **performance — 128 → 32 One Pager computations per `/bundle` (4.0x), `_pe_cap_comments` scoped, 71 lines of duplicated assembly removed** | outputs byte-identical |

**Step 7 is DONE and the feature is deployable.** All four subtab backends, the
foundation, persistence, the API and the UI are in place. Nothing is deployed —
`az acr build` + `az containerapp update` have not been run, and per the current
decision everything is hardened *before* the first deploy.

---

## Step 5a TODOs — both CLOSED

Both adjustments requested after the first 5a pass are **in commit `13ff99f`**, not
outstanding:

- **Net ROE / ITD text-box accessors — DONE.** `get_net_roe()` / `get_itd()` are the
  only read path; the assembly never touches the values table. Each row carries
  `net_roe_source` / `itd_source` = `"manual entry (formula TBD)"`. Round-trip
  verified: enter 0.0912 / 1,250,000 → accessors and assembly return them, other
  deals stay `"pending entry"`, an approved page refuses the edit, the stored value
  stays readable.
- **Excluding-dev subtotal removed from Financial — DONE.** `total_excluding_dev` is
  gone and asserted absent. The portfolio total covers all 35 deals (asserted two
  ways). That subtotal is Loan-tab only.

---

## File layout — what exists now

**New files (the whole feature, except the three additive edits below):**

```
flask_app/services/portfolio_snapshot_service.py        Step 1  foundation
flask_app/services/portfolio_snapshot_persistence.py    Step 2  elements + approval
flask_app/services/portfolio_snapshot_operating.py      Step 3a Operating assembly
                                                                + resolve_strategy()
flask_app/services/portfolio_snapshot_loan.py           Step 4a Loan assembly
flask_app/services/portfolio_snapshot_financial.py      Step 5a Financial assembly
flask_app/services/portfolio_snapshot_summary.py        Step 6a Summary assembly
flask_app/api/portfolio_snapshot.py                     Step 7  blueprint, 13 routes
vue_app/src/views/PortfolioSnapshotView.vue             Step 7  shell
vue_app/src/components/snapshot/SnapshotSummary.vue     Step 7
vue_app/src/components/snapshot/SnapshotFinancial.vue   Step 7
vue_app/src/components/snapshot/SnapshotOperating.vue   Step 7
vue_app/src/components/snapshot/SnapshotLoan.vue        Step 7
vue_app/src/components/snapshot/format.ts               Step 7  shared formatters
portfolio_snapshot_build_spec.md                        the plan
portfolio_snapshot_resume.md                            this file
```

**Shared files touched — four edits total, all additive:**

| File | Step | Change |
|---|---|---|
| `database.py` | 2 | 3 tables added to `PROTECTED_TABLES` (32 → 35) |
| `flask_app/__init__.py` | 7 | 2 lines: import + `register_blueprint(url_prefix="/api/portfolio-snapshot")` |
| `vue_app/src/router/index.ts` | 7 | one lazy route `/portfolio-snapshot` |
| `vue_app/src/components/layout/AppSidebar.vue` | 7 | one `router-link` under Asset Management + the path in `amRoutes` |

**Final count across the whole feature: 17 files added** (15 code + 2 docs),
**4 shared files modified by 17 lines total**, 8,854 insertions / 2 deletions.
Both deletions are the two lines above that *grew* (the `PROTECTED_TABLES` set and
the `amRoutes` array). Nothing was deleted and no existing behaviour was changed —
every touch appends to a list, a set or a route table.

Files added after this table was first written: `portfolio_snapshot_guardrails.py`
(Step 8), `portfolio_snapshot_freeze.py` (freeze). Routes grew 13 → 14 with
`/reopen`. `PROTECTED_TABLES` grew 35 → 36 with `portfolio_snapshot_frozen`.

**Key API shapes** (the Vue components are written against these live field
names, so renaming any of them breaks the UI silently):
`GET /bundle` → `{subtabs: {summary, financial, operating, loan}, errors, resolution, review}`.
Financial's `groups` are `{deals, subtotal}` per fund; Operating and Loan `groups`
are plain arrays. `*_display` fields are polymorphic — number, `"Dev"`,
`"pending entry"`, or null — and `disp()` in `format.ts` passes strings through
verbatim so the UI never recomputes a backend decision.

---

## Deploy handoff

**Deploy `wip/portfolio-snapshot-performance` @ `c4c2cbcc698ff0c5efd012e83a4bbc19e733b58f`.**

```
az acr build --registry acrwaterfalldev -g rg-waterfall-dev     --image waterfall-xirr:latest --no-logs .
az containerapp update -g rg-waterfall-dev -n app-waterfall-dev-v2     --image acrwaterfalldev.azurecr.io/waterfall-xirr:latest --revision-suffix vNNN
```

Merging alone ships nothing. `main` stays untouched until someone decides to merge.

### What is being deployed

- **17 files added** — 8 Python services, 1 API blueprint, 6 Vue files, 2 docs
- **4 shared files modified, 17 lines, all additive** — `database.py` (+7/−1),
  `flask_app/__init__.py` (+3), `router/index.ts` (+5), `AppSidebar.vue` (+2/−1)
- **4 new tables**, all in `PROTECTED_TABLES` (35 → 36) and all self-creating on
  first use — **no migration step**: `portfolio_snapshot_comments`,
  `_footnotes`, `_values`, `_frozen`
- **14 API routes** at `/api/portfolio-snapshot`: `bundle`, `<subtab>`, `deals`,
  `investors`, `quarters`, `elements`, `comment`, `value`, `footnote`,
  `footnote/<id>`, `submit`, `approve`, `return`, `reopen`
- **1 nav entry** — "Portfolio Snapshot" under Asset Management, between One
  Pager and Review Tracking
- App starts clean: **245 routes**, all 8 service modules import without error

### Deploy-time known items — only verifiable in Jim's build / on Azure

1. **No local frontend build verification.** `vue_app/node_modules` is absent, so
   neither `vue-tsc --noEmit` nor `vite build` has ever run against the 6 Vue
   files. Unused imports and locals were removed by hand, but **a TypeScript
   error would surface for the first time in the Docker build.** Biggest risk.
2. **The investor dropdown depends on `review_service.get_investor_list()`**
   (returns bare ID strings). If it throws there is a relationships-derived
   fallback; if that is also empty the dropdown is empty and nothing renders.
   **Check it populates first thing after deploy.**
3. **Review actions are role-gated.** `can_submit` / `can_approve` /
   `can_return` / `can_reopen` need a `review_roles` row (or admin). With none
   assigned the status strip reads "no action available for your role" —
   correct, not a fault.
4. **The 4x performance win is a call-count measurement, not wall-clock.**
   128 → 32 `get_one_pager_data` calls per `/bundle` is exact and deterministic;
   what that is worth in seconds could not be measured locally (empty SQLite,
   unrepresentative REST latency). **Spot-check `/bundle` timing on Azure.**
5. **No threaded review notes.** Step 2 has no notes table, so Return/Reopen
   enforce their required note but have nowhere to persist it. The inline status
   strip was chosen over `ReviewPanel.vue` for exactly this reason.

### Validation caveat — only TIAA is PDF-verified

Every number in this build was validated against the **26Q1 TIAA** PDF. TIAA
works end to end partly because `INVESTOR_NAME_ALIASES` hardcodes
`TGAM -> "TIAA"` — the literal string "TIAA" appears nowhere in MRI.

**KOC and Declaration are NOT PDF-verified and their names will not resolve.**
`MRI_IA_Investor` (MRI **IM** server, 374 rows) is what names `KOCINV`,
`DCXVIA`/`DCXVIB` and `PSC1/2/3`, and it **is not in the app database** — it needs
a new `QUERY_REGISTRY` entry plus VPN. Until then `get_investor_name()` falls
through to the raw code, so those investors render as `KOCINV` etc., and none of
their figures has been checked against a reference report. Expect to validate them
separately.

### Final self-test status

| module | result | note |
|---|---|---|
| service (1) | **ALL PASS** | |
| persistence (2) | **35/35** | |
| financial (5a) | **28/28** | |
| guardrails (8) | **22/22** | |
| freeze | **34/34** | includes the key anti-drift test |
| loan (4a) | 45/46 | Nottingham LTV debt vintage — known, documented |
| summary (6a) | 40/43 | 45th & Main 90-vs-100 + Multifamily/Self-Storage ~1pp — known |
| operating (3a) | **13/28** | see the correction below |

### CORRECTION — Operating is 13/28, not the 15/28 reported earlier

An earlier report and an earlier revision of this note said Operating was 15/28
"unchanged". **That number was stale.** Operating was not in the loop re-run
during Step 8, so a pre-45th-&-Main-fix figure was carried forward.

Re-measured at deploy state: **13/28**. The two extra failures are
`"45th & Main still appears (ownership-flagged)"` and
`"45th & Main carries its ownership flag"` — the same **silently-disarmed
assertion** class fixed in `service`, `financial`, `loan` and `summary` during
Step 8, missed in `operating`.

`portfolio_snapshot_operating.py` is **byte-unchanged** since that run, so this is
pure live-data drift, **not a regression and not shipped behaviour** — self-test
assertions only. **Non-blocking for deploy.** The fix is two lines: assert the
property (a flagged deal withholds its figures) rather than the identity
(45th & Main is flagged), exactly as the other four modules now do.

---

## Open items — all non-blocking, all post-deploy

1. **TGAM2 fund-vs-SPV flip.** `_classify_entities` calls an entity a fund on a
   deal-count threshold, so TGAM2 is a fund at 26Q1 (East Manchester + Giant 7)
   and an SPV at 26Q2 (East Manchester sold 2026-06-25), moving Giant 7 in and out
   of Individual Investments. **Only visible across a multi-quarter series.**
   Pre-existing behaviour, not caused by the acquisition gate — verified against
   the pre-change Q1 result.
2. **Committed basis +7.39%** ($477.99M vs the PDF's $445.1M) and
   **Self-Storage −1.07pp / Multifamily +0.97pp**. Both are *documented
   differences*, not bugs: the committed figures are plausible real unfunded
   commitments on development/staged deals (the `abs()` bug was ruled out —
   `abs sum == |signed sum|` on all 32 deals), and the asset-type offset is ~$4.35M
   sitting in a different bucket in the PDF with no located cause.
3. **Net ROE / ITD are manual entry.** Typed from Excel; both carry
   `"manual entry (formula TBD)"`. Automating them is a separate piece of work.
4. **Admin review bypass.** `_review_payload` lets `role == "admin"` pass every
   review gate. The One Pager has **no** such bypass. Kept because with no
   `review_roles` rows on live the pipeline would otherwise be unusable —
   **reconsider once real roles are assigned.**
5. **`reopen()` is gated on `("ceo",)`.** Derived from the One Pager's own rule
   ("the role at the step being reversed may reverse it"), not copied — the One
   Pager has no reopen path at all. Confirm the authority is right.
6. **East Manchester has no commitment row** (funded $3.6M, committed $0, un-funded
   computes to −$2.7M). Source-data fix. Low priority — the deal sold this quarter.

---

## Step 6a (Summary) — landed

`flask_app/services/portfolio_snapshot_summary.py`, self-test **41/43** against the
26Q1 PDF page 1. Nothing imports it.

**Scaling basis: LOOK-THROUGH, settled empirically.** The PDF's page-1 dollars are
TIAA's share, like the four scaled columns on Financial:

| variant | funded | vs PDF $404.2M |
|---|---|---|
| look-through, 45th & Main at 100% | **403.95M** | **−0.06%** |
| look-through, flagged deal excluded | 385.40M | −4.65% |
| full deal-level | 540.96M | +33.83% |

`funded` here equals Financial's `invested` **to the cent** (385,401,813) — same
`cap_stack.pref_equity` through the same %, asserted so the subtabs cannot drift.
Note the total only ties with **45th & Main at 100%**, corroborating that its PMX
0% edges are the data bug (MRI's IM copy shows 100%). Production still excludes
and flags it; the 100% figure appears only in the self-test.

**Asset Allocation** — Retail 29.07% vs 29%, Office 3.03% vs 3% (both <0.1pp).
Multifamily 59.97% vs 59% (+0.97) and Self-Storage 7.93% vs 9% (−1.07) are the two
failing checks; they exactly offset, i.e. ~4.3M sits in a different type in the
PDF. Lead: Citizen Storage Swartz Creek (P0000120, Self-Storage) has `total_cap`
4.6M but `pref_equity` 0 — at 90% that is 4.14M, which would bring all four types
into line but push the total ~1% above the PDF. Data question, not a formula one.

**Deal Type Allocation** — all three inside 0.7pp:
Value-Add 150.15M / 37.17% (PDF 153.1M / 38%) · Income 132.15M / 32.71%
(PDF 131.3M / 32%) · New Construction 121.65M / 30.12% (PDF 119.8M / 30%).

`DEAL_TYPE_MAP` is **calibrated, not literal**: an exhaustive search over all 243
assignments of the five Lifecycle values in TIAA's set picked it at 5.65M total
absolute error vs 24.95M for the runner-up (4.4× margin). `Development → New
Construction`, `Stable → Income`, and `New Construction → Income` — that last edge
is driven solely by Pegasus and reflects that Lifecycle is a build state while the
PDF's pie is an investment thesis. `DEAL_TYPE_MAP_LITERAL` is returned alongside
under `alternate_deal_type_literal` for audit.

**Part B** — two boxes, `narrative_1` / `narrative_2`, `scope='report'`, through
Step 2's existing save/approval path. Both load blank with
`source = "blank (no auto-generation)"`; round-trip verified.

---

## Step 8 (guardrails) — landed

`flask_app/services/portfolio_snapshot_guardrails.py`, self-test **22/22**, all
negative tests (each detector must fire on injected bad data). Wired into
`GET /bundle` as an advisory `guardrails` block — a finding never blanks a metric
or fails the page, so one bad deal cannot hide the whole report. Severities:
`error` = a figure is wrong or a missing value renders as real; `warn` = needs a
human; `info` = a count worth surfacing.

**Live 26Q1 result: 0 errors, 0 warnings, 3 info.** The page is clean.

### The one real leak found and FIXED

`_rollup` in the Summary assembly initialised its accumulators at `0.0` and only
ever added, so **a bucket where every deal was missing reported `0.0`, not
`None`** — and `0.00` is indistinguishable from a genuine zero once formatted.
Its own docstring claimed the opposite. Fixed by tracking whether any deal
contributed, matching what Financial's `_subtotal` already did
(`sum(vals) if vals else None`). `check_aggregate_integrity` now guards both.

### The leak that is NOT ours to fix — detected instead

**`one_pager.py` initialises every `cap_stack` money field to `0.0`, not `None`**
(`debt`, `pref_equity`, `ptr_equity`, `total_cap`, `committed_pe`, …). A deal the
One Pager has no data for therefore returns `0.0`, which formats to "0.00" and
reads as a real figure. That contract is shared with other tabs, so
`detect_empty_cap_stack` flags the condition rather than changing it: a *single*
zero can be genuine (Citizen Storage really had $0 pref at 26Q1, pre-closing),
but an **entire** Zone A of zeros means no data. Live 26Q1: no deal trips it.

### Missing-data audit — the `zero` column is the dangerous one

| subtab | field | real | zero | missing |
|---|---|---|---|---|
| financial | debt | 31 | **1** (Pegasus) | 0 |
| financial | ptr_equity | 31 | **1** (Jefferson Stephens) | 0 |
| financial | unfunded | 0 | **32** | 0 |
| financial | total_pref / total_cap / invested / total_commitment | 32 | 0 | 0 |
| loan | debt | 31 | 0 | 1 |
| loan | valuation | 28 | 0 | 4 |
| loan | ltv | 20 | 0 | 12 |
| loan | ytd_dscr | 18 | 0 | 14 |
| operating | expected / actual growth | 23 | 0 | 9 |

**`unfunded` is 0 on all 32 deals** — a real number that means nothing. It is the
direct consequence of `COMMITMENT_BASIS = "funded"`, which makes Total Commitment
identically equal to Invested. It renders as "0.00" everywhere and a reader
cannot tell it from a computed zero. **This is the one remaining place bad data
can pass as real**, and it is a pending creator decision, not a bug.
(`loan.debt_yield` shows missing on all 32 only because the self-test stubs the
quarterly-NOI provider; in the app it is populated.)

### 45th & Main is FIXED in MRI — RESOLVED — and it moved the PDF tie

Live build changed mid-build from `09fe220ae0da` to **`f7692533e77c`**. The PMX
`IA_Relationship` row was corrected, so 45th & Main now resolves at **90.0% into
TGA24** via `TGAM → TGA24 → PPI45M → 45MAIN`. **Zero deals are ownership-flagged.**
That closes the long-standing blocker.

**The 100% and the 90% are different hops — both are correct.** Traced live
(`OwnershipPct` on `relationships`), and worth keeping because the two numbers
look contradictory until you see the chain:

| hop | edge | `OwnershipPct` | normalised |
|---|---|---|---|
| deal | **`PPI45M → 45MAIN`** — *the record that was corrected, 0% → 100%* | 100.0 | 100% |
| | `OPEVGR → 45MAIN` (operating partner, carried) | 0.0 | 0% |
| SPV | `TGA24 → PPI45M` | 100.0 | 100% |
| fund | **`TGAM → TGA24`** | 90.0 | **90%** |
| | `INV24 → TGA24` | 10.0 | 10% |

Look-through = 0.90 × 1.00 × 1.00 = **90.0%**. The 100% is asset-level ownership;
the 90% is TIAA's share of the *fund*, and **every TGA24 deal inherits that same
90% hop** — Flats at Dorsett Ridge, Green Valley Ranch, ReNew Glenmoore and Town
Fair Tire are all 90.0000%, and Seasons at Bel Air is 51.2676% × 90% = 46.1408%,
which is the arithmetic proof.

So 45th & Main at 90% is the *only* value consistent with its fund siblings, and
the PDF's implied ~100% for it alone would be internally inconsistent. **That
weakens the "45th & Main should be 100%" reading of the PDF gap** — the residual
~$2.1M more likely sits elsewhere, possibly bound up with the unexplained ~$4.35M
Multifamily/Self-Storage offset (45th & Main is Multifamily).

Consequence: the Summary funded total is now **402.10M vs the PDF's 404.2M
(−0.52%)**, where the assumed-100% variant used to give 403.95M (−0.06%). The
1.855M gap is exactly 10% of the deal's 18,550,000 funded pref — **the PDF was
built with 45th & Main at 100% while MRI now says 90%.** Which is right is a
creator/data question. The self-test check is left **knowingly red** and
self-explaining rather than having its tolerance widened, because widening it
would hide a $2.1M discrepancy. Same precedent as the Nottingham LTV check.

**Four self-tests had hardcoded 45th & Main as flagged and silently disarmed
themselves when it resolved.** All four were rewritten to assert the *property*
(a flagged deal is absent from `groups`; flagged deals withhold Zone B) rather
than the *identity*, so they survive the data changing. The Step 8 self-test
itself hit the same trap during the build and is now fully synthetic for exactly
this reason.

### Self-test status after Step 8

| module | result | remaining failures |
|---|---|---|
| service (Step 1) | **ALL PASS** | — |
| financial (5a) | **28/28** | — |
| guardrails (8) | **22/22** | — |
| loan (4a) | 45/46 | Nottingham LTV debt vintage (known) |
| summary (6a) | 40/43 | 45th & Main 90-vs-100 (new, above) + Multifamily/Self-Storage ~1pp (known) |

### Isolation re-verified after Step 7 wiring

Shared files touched vs `main`: **4 files, 14 insertions, 2 deletions** —
`database.py` (Step 2 `PROTECTED_TABLES`), `flask_app/__init__.py`,
`vue_app/src/router/index.ts`, `AppSidebar.vue`. All additive. App starts, 244
routes. A snapshot failure cannot take the app down: the guardrail block is
wrapped, `run_guardrails` catches per-check exceptions and reports them as
findings, and `_data_or_error()` turns a data-load failure into a clean 503.

---

## Snapshot freeze — landed

`flask_app/services/portfolio_snapshot_freeze.py`, self-test **34/34**, fully
synthetic (a scratch SQLite plus an injected assembler), so the drift test is
deterministic. **This closes the gap that was the most consequential open item.**

**The problem it removes.** An approved report used to re-render live on every
view, so an approved 26Q1 report would silently change when MRI data moved — and
it moved during this build: 45th & Main's look-through went 100% → 90% on
2026-08-24. THE KEY TEST reproduces exactly that: approve a report, then move the
underlying figure 1.00 → 0.90, and confirm the approved report still serves
**1.00 / $18,550,000**, not the live 0.90 / $16,695,000.

**Mirrors the One Pager** (`review_service._save_snapshot` / `get_snapshot`) on
storage, trigger, upsert and failure containment:

| | |
|---|---|
| storage | `portfolio_snapshot_frozen`, UNIQUE(investor_code, quarter), JSON payload + approved_by/at + `data_version`. **In `PROTECTED_TABLES` (35 → 36)** — a CSV import overwriting it would rewrite what was signed off. |
| trigger | the FINAL transition into `approved` only, from `approve()` via a lazy import, so persistence stays free of the assemblies |
| upsert | DELETE-then-INSERT in one transaction — cross-DB, and *is* the re-approval overwrite (asserted: exactly one row per investor+quarter) |
| failure | wrapped by the caller — a freeze failure logs and **the approval still stands** (asserted by injecting a failure) |
| unfreeze | free. `_set_status` already nulls `approved_at` on any non-approved transition, so the read path's `status == "approved"` test falls back to live by itself |

**The one deliberate divergence, per creator decision:** the One Pager defaults to
*live* with the frozen copy behind a manual "View Approved Version" toggle; the
Portfolio Snapshot serves **frozen by default** when approved. Every payload
carries `source: "frozen" | "live"` plus a `source_note` for the UI. Flagged in
the module docstring — do not "align" the two tabs without reading it.

**`assemble_full_report` is the single assembly path** used by both the freeze and
the live read. If the freeze assembled differently, a frozen payload would differ
from live even with unchanged data and every comparison would be noise.

**Frozen content is the whole report:** all four subtabs' assembled payloads AND
the approved editable content — comments, footnotes, and the manual Net ROE / ITD
values — as they stood at approval.

**Guardrails are skipped on a frozen payload** (`guardrails.skipped`): they audit
a live computation, and re-auditing what was already approved would surface
findings nobody can now act on.

### A gap this exposed — `reopen()` had to be added

**`reject()` cannot reopen an approved page.** At the approved step
`_step_for(5)["role"]` is `None`, so its role check raises
`PermissionError: You need the 'None' role to return at this step`. An approved
report therefore had **no route back**, which also meant no way to reach the
re-approval that replaces a frozen payload — the requirement described an
impossible flow.

New `portfolio_snapshot_persistence.reopen()` (additive): approved → returned,
note required, gated on `REOPEN_ROLES = ("ceo",)` since the CEO is the approver at
the final step. Deliberately **not** a relaxation of `reject()` — returning a page
mid-review and unwinding a completed approval are different acts with different
authority, and collapsing them would let any reviewer reverse a CEO sign-off.
**The role gate is a guess and needs creator confirmation.** It is not yet exposed
on any route; the blueprint has `/submit`, `/approve`, `/return` but no `/reopen`.

The frozen row is left in place on reopen — it becomes unreachable while the
status is not `approved`, and the next approval overwrites it. Deleting it would
throw away the only record of what was approved while the correction is in flight.

### Also verified

Approved-but-unfrozen (approved before this existed, or a failed freeze) falls
back to live **and says so** in `source_note` rather than showing an empty report.
A corrupt payload returns `None` rather than raising. Isolation unchanged: the
only shared app file touched is `database.py`, additively, for the one new table.
App starts, 244 routes; persistence 35/35 and guardrails 22/22 still pass.

---

## SEPARATE APP-SIDE BUG — accrued pref. OREI part RESOLVED (see below). NOT this feature.

Found while working on the snapshot but **entirely outside it**: this is Jim's
app-side fix in `reports_service.py` / the ROE Summary path. **Nothing has been
applied. No code changed. Pending decisions.** It does not block the snapshot
deploy and no snapshot module touches it.

### Symptoms

- **OREI Portfolio, Apple Self Storage and Plaza Del Mar show $0 accrued pref.**
- **Cocoplum accrues at the wrong rate.**

### Two independent root causes

**(a) The `deal_terms.pe_coupon` read fails silently on live PostgreSQL.**
`reports_service.py:~1002` does
`pd.read_sql("SELECT pe_coupon FROM deal_terms WHERE UPPER(vcode) = :vc", _sa_engine, params={"vc": ...})`
— a raw string with a `:name` binding, which needs `sa.text()` on SQLAlchemy/PG —
wrapped in a **bare `except Exception: pass`**. The query raises, the exception is
swallowed, and `pref_rate` silently stays `0.0`. **Same class of bug as the
`get_one_pager_comments` PG fix already recorded in MEMORY.md.**

**(b) OREI's waterfall pref step is labelled `vState='Default'`, not `'Pref'`,**
so the waterfall-based rate lookup finds nothing either. Both paths miss, so the
rate is zero.

### RESOLVED 2026-08-24 — OREI's $0 IS CORRECT. DO NOT "FIX" OREI.

**Per Matt: the JV amendment last year ENDED OREI's pref accrual.** The deal is
**cash-flow-split by percentages** now. So `vState='Default'` is a faithful
reflection of the **amended** structure, not a mislabel, and the $0 accrued
balance is the right answer. Manufacturing 8.5% pref here would have invented
income the amendment deliberately removed — which is exactly why the question
gated the fix.

**The underlying code bug (a) is still real, and still needs fixing** — it just
is not OREI's problem. It affects:

- **Apple Self Storage** (~$8.06M PE capital) and **Plaza Del Mar** (~$13.60M)
  — both show $0 accrued. **CHECK THEIR OWN TERMS FIRST.** They may also be
  amended, in which case $0 is correct for them too. Do not assume a bug from
  the symptom; OREI just proved the symptom is ambiguous.
- **Cocoplum**, which accrues at 5% where `deal_terms` says 8.5% — and is
  **genuinely two-tier (5% / 8.5%)**, so it needs a business decision, not a
  flat rate.

For reference, the accrual mechanics were walked end to end and verified against
live: `build_pref_balance_detail()` is **simple within the year, compounded
annually at 12/31**; base `H = prior InvBal + prior CompPref`;
`CurrDue = H x rate x days/days_in_year(row year)` (Act/Act, no cross-year
split); distributions pay **pref only** and never principal (surplus is truncated
by `max(0, …)`); quarter-ends are synthesised as zero-amount rows. With
`rate = 0` the whole chain collapses to 0 regardless of capital — which is why
OREI shows 33 event rows, $13.39M of capital, and $0.

### The naive fix ships a REGRESSION

Simply wrapping the query in `sa.text()` **breaks Pegasus / TGA22, moving them
10% → 9%**, because `deal_terms` is **investor-blind** while Pegasus has
**per-investor rates**. The correct order is:

> **investor-aware waterfall match → `deal_terms` → any waterfall rate**

**Cocoplum is genuinely two-tier (5% / 8.5%)** and needs a *business decision*, not
a flat rate.

### Wider data-completeness question

**9 further deals (~$50.7M of capital) accrue no pref at all** — they have neither
a `deal_terms` row nor a `Pref` waterfall step. Whether that is correct or a data
gap is unresolved.

### Artefacts

Diagnostics are **untracked, read-only**, in `scripts/`:
`orei_accrued_diag.py` … `orei_accrued_diag5.py`, `pref_fix_impact.py`,
`pref_fix_impact_detail.py`, `pref_mechanics_walkthrough.py`. They are not
committed (they follow the existing convention for per-developer diagnostic
scripts) — **they will be lost if the working tree is cleaned.** Commit them
first if they need to survive.

`pref_fix_impact.py` is the one worth keeping: `pull` / `simulate` / `validate` /
`report` stages, and its `validate` stage proved the simulated "before" matched
live on **78 investor-level and all 110 deal-level comparisons, zero
mismatches** — which is what licensed the "after" numbers.

### NEXT TASK when this is picked up

1. **Check Apple Self Storage and Plaza Del Mar's actual deal terms** — amended
   or not? That decides whether their $0 is a bug at all.
2. Only then implement the **reordered** lookup
   (investor-aware waterfall match → `deal_terms` → any waterfall rate). The
   naive `sa.text()` fix alone **ships a Pegasus/TGA22 regression** (10% → 9%,
   −$435,165) because `deal_terms` is investor-blind.
3. Decide Cocoplum's two-tier treatment separately.
4. Narrow the bare `except Exception: pass` at `reports_service.py:~1008` — it
   hid this indefinitely.

---

## Open creator / data items

### RESOLVED — the strategy field. `Lifecycle` IS the strategy field.

This supersedes the earlier "populate `Investment_Strategy`" item, which was
based on a wrong premise. Settled against live build `09fe220ae0da`:

- **`deals.Investment_Strategy` is 0/110 populated, and is UNPOPULATABLE.** Four
  independent live probes agree: `/api/data/deals/all`, the raw `deals` table, a
  dump of every column in that table (no column anywhere has 99 populated), and
  server-side `filter__Investment_Strategy=` for six substrings including the bare
  letters `a` and `e` — **0 rows every time**, while the identical filter on
  `Lifecycle` returns 25 and 31, proving the filter mechanism works.
  **There is no MRI source column to populate it from.** The only strategy-ish MRI
  field is `meta.vStatus`, already mapped to `Lifecycle` in both
  `queries/Prop_Info_Core.sql:65` and `queries/Prop_Info.sql:175`. So this is NOT
  "add a column to the query" — it would need a new field on MRI's side or manual
  entry. Do not re-open this as an extraction task.
- **A prior session recorded it as "99/110 populated, the field the One Pager
  displays". That was a measurement of the wrong thing.** `one_pager.py:282`
  coalesces `['Investment_Strategy', 'Lifecycle', 'InvestmentStrategy',
  'Strategy', 'vStrategy']`, first non-null wins — and the last three **exist in
  no live table at all** (checked across all 77 tables). Since
  `Investment_Strategy` is empty, the One Pager's displayed "Investment Strategy"
  is **always `Lifecycle`**. Ascent shows "Value-Add" because
  `Lifecycle='Value-Add'`; its `Investment_Strategy` is `None`.
- **Database-wide scan: exactly ONE column anywhere holds investment-strategy
  values — `deals.Lifecycle`.** 65 strategy-ish-named columns across all tables
  were value-probed against six strategy terms; every other one returned nothing.
  `deals.Investment_Category` and `deals.Property_Description` are also 0/110.
  `Sub-Asset_Type` (22/110) holds asset sub-types, not strategies. `deal_terms`
  has no strategy column. (Method limit: for the `deals` table coverage is
  complete — all 30 columns dumped; other tables were name-hinted, and
  `lease_documents` 500s on schema fetch and was not inspected.)
- **Therefore dev detection via `Lifecycle` is correct, not a workaround** — it
  reads the same field the One Pager displays. `resolve_strategy()` in
  `portfolio_snapshot_operating.py` is the single definition, shared by the
  Summary, Operating and Loan subtabs: it prefers `Investment_Strategy` and falls
  through to `Lifecycle`, i.e. `one_pager.py:282`'s coalesce minus the three
  aliases that do not exist. `DEV_STRATEGY_ALLOW_LIFECYCLE_FALLBACK = True`
  enables the fallback; set it to `False` only if `Investment_Strategy` is ever
  fed. `Lifecycle` is 97/110: Value-Add 31 / Development 24 / Income 22 /
  Stable 16 / New Construction 2 / Redevelopment 1 / Lease up 1.
- **This is what lifted Loan from 21/26 to 45/46.** 10 of TIAA's 35 deals now
  classify as dev, JB Fair Park renders "Dev" instead of a 457.7% LTV off stale
  debt, and its Debt uses the 77,368,000 committed facility.
- **45th & Main (P0000089) — MRI fix pending.** Both owner edges are 0% in PMX
  (`PPI45M → 45MAIN` and `OPEVGR → 45MAIN`), while MRI's IM copy shows 100%. The
  deal therefore has no look-through % and is flagged "ownership % unavailable".
  One row in PMX `IA_Relationship` fixes it; the guardrail should stay regardless.
  Note the Waterfall Setup screen shows 0.00% for it, and `/investors` separately
  fabricates a 50/50 split (see below).

### RESOLVED — Step 1 acquisition-date gate (ownership window)

Step 1 gated the **sold** end of the ownership window but not the **acquired**
end, so deals that had not closed yet still rendered rows. The Loan subtab
carried their debt as if real: at 26Q1 **Presidential Arms alone contributed
$98,980,000 of phantom debt** against zero equity (the One Pager's equity block
is quarter-filtered; the debt line is not), plus Citizen Storage $4,600,000 and
Fairview Heights $13,250,000 — **$116,830,000 total**.

`is_acquired_as_of()` now mirrors `is_sold_as_of()`. Full test: **acquired on or
before `quarter_end` AND not sold on or before it**, both applied before fund
classification so a deal outside the window cannot inflate an entity's tally.

**Missing `Acquisition_Date` fails OPEN (deal kept), deliberately asymmetric with
the sold gate.** Including a disposed deal reports something no longer owned, so
that gate excludes on a missing date; excluding here would silently drop a deal
genuinely held and understate the portfolio, the worse failure. 34 of 110 deals
carry no date — almost all child properties already removed on
`Property_Count == 0` — and every kept case is listed in
`diagnostics['acquisition_date_missing']` (0 in TIAA's set).

26Q1 population **35 → 32** (31 grouped + 1 ownership-flagged), exactly the 3
deals; all 3 reappear in 26Q2 and 26Q3. **Every PDF figure is byte-identical**,
because all three had `pref_equity` 0 and `committed_pe` 0 at 26Q1: funded
403.95M, committed 477.99M, all four asset types, all three deal types, the
Summary↔Financial identity at 385,401,813, and the four Loan anchors.
Self-tests after the change: Step 1 **25/25**, Summary 41/43, Loan 45/46,
Financial 28/28, Operating 15/28 — all unchanged bar Step 1's new checks.

Two grouping consequences, both correct:
- **TGA6 disappears from 26Q1** (7 groups → 6). Its only two deals are Fairview
  Heights and Presidential Arms, neither owned in Q1, so the fund held nothing.
- **TGAM2 is a fund at 26Q1 and an SPV at 26Q2, moving Giant 7 in and out of
  Individual Investments.** This is NOT caused by the gate — verified against the
  pre-change Q1 result, where TGAM2 was already a fund. It is pre-existing
  behaviour of `_classify_entities`, which calls an entity a fund on a deal-count
  threshold: TGAM2 holds East Manchester + Giant 7 in Q1, but East Manchester
  sells 2026-06-25, leaving one deal in Q2. **Worth a decision** — fund identity
  currently shifts quarter to quarter with holdings.

### RESOLVED — Loan dev-display rules (this branch)

- **Waters Creek LTV exception — TEMPORARY hardcode, verified.**
  `WATERS_CREEK_LTV_EXCEPTION = {"P0000078"}` in `portfolio_snapshot_loan.py`:
  Jefferson Waters Creek renders a **real LTV** (income-based valuation on entering
  lease-up) while DSCR and Debt Yield stay "Dev". Computed **57.51% vs the PDF's
  57.4%** (+0.11pp). That tie also settles the numerator: the dev basis
  `mOrigLoanAmt` 51,667,000 is right, since the ISBS drawn balance 48,416,160 gives
  53.89% and would not match. The rule it stands in for — a dev deal shows a real
  LTV when `mIncomeCapConcludedValue` is a true income-based valuation — needs a
  valuation-method/basis column that is not currently extracted. Delete the
  constant and `_ltv_exception()` when that lands.
- **Pegasus Life Storage — RESOLVED, reads n/a via the debt-None gate.**
  Its `Lifecycle = 'New Construction'` is **correct** (it is what the One Pager
  displays), so the label alone would render "Dev" — but the PDF shows **n/a** for
  its LTV, DSCR and Debt Yield. Cause: an empty loan block — ISBS debt 0.0, no
  `mOrigLoanAmt`, `loan_count` 0, so `debt is None`. `dev_no_data = dev and debt is
  None` is gated **ahead of** the dev checks and forces n/a on all three columns.
  **Exactly one deal changes** (asserted: `dev_no_data == ["P0000066"]`); the other
  9 dev deals keep "Dev".
  *Gated on `debt is None` — the "empty loan block" signal — deliberately NOT on
  per-column ratio availability, which over-flips:* no dev deal carries a real YTD
  DSCR (raw `ytd_actual` is None on 9 of 10), so per-column gating would drop DSCR
  to n/a almost everywhere and JB Fair Park would stop reading "Dev". The gate is
  also scoped to dev deals so a non-dev deal never loses its real DSCR, which is
  NOI ÷ debt service and does not depend on the balance.
  This closes the earlier Pegasus question — no remapping of
  `New Construction → dev` was needed, and note the Step 6a allocation calibration
  separately places Pegasus in the **Income** pie bucket. Both are now consistent:
  Lifecycle is a build state, the pie is a thesis.

### Pending creator decisions

- **Debt Yield NOI basis — flagged pending.** Implemented as *single-quarter
  Interim IS NOI × 4*, which reproduces the PDF at Q1 (Camp Creek 13.47% vs 13.5,
  Post Commons 10.69% vs 10.7). The source Excel's SUMIFS actually returns the YTD
  cumulative, so its flat ×4 over-counts 2× at Q2, 3× at Q3, 4× at Q4. Also note
  single-quarter×4 (run-rate) and YTD÷months×12 (average-to-date) are **not**
  interchangeable beyond Q1; both are returned per row.
- **`COMMITMENT_BASIS`** — `"funded"` reproduces the PDF but makes Un-funded
  identically 0 on every deal. `"committed_pe"` populates it (Nottingham $1.2M; 12
  deals portfolio-wide) but breaks the PDF match. One constant in
  `portfolio_snapshot_financial.py`.
- **Nottingham LTV vintage.** Computed 79.12% (26Q1 debt 38,850,000) vs the PDF's
  ~75%, which equals the **26Q2** debt 37,000,000 / 49,100,000 = 75.36%. Formula is
  right; the PDF appears to carry a debt balance later than its own quarter.
- ~~**TGA6 zero-pref deals.**~~ **CLOSED** — Fairview Heights and Presidential Arms
  read $0 at 26Q1 because they had not closed yet (2026-06-30 and 2026-05-13).
  The acquisition-date gate now removes them from that quarter entirely, so there
  is no zero to render. They carry real figures from 26Q2 on.
- **East Manchester has no commitment row — fix at source.** Funded $3,600,000 but
  no `Contribution`/"Commitment" row in accounting, so `committed_pe` is 0 and
  un-funded computes to **−$2,723,400** (scaled). A deal cannot be funded above
  its commitment; the pledge row is missing. Verified live 2026-08-24.
- **Committed total runs +7.39% over the PDF** ($477.99M vs $445.1M). NOT the
  `abs()` bug — `abs sum == |signed sum|` on all 32 deals with commitment rows,
  each a single consistently-negative pledge dated at closing, and 22 of 32 have
  committed == funded exactly. The un-funded is dominated by Development deals
  mid-draw (JB Fair Park 19.4M, Jefferson Stephens 17.9M, Brainerd 8.4M, Trolley
  2.3M) plus staged portfolio Burton 24.9M, all plausible. Leading hypothesis:
  the PDF excludes un-funded commitment for assets **not yet acquired or built** —
  dropping Burton + Brainerd gives 40.77M vs the PDF's implied 40.9M (0.33%), the
  only subset within $1M of 129 one-to-three-deal combinations tested. Both are
  phased/multi-property (Burton 3 properties; Brainerd two same-date tranches
  across 9 buildings). A fit with a mechanism, not proof — needs confirmation.
- **Asset allocation: Multifamily +0.97pp / Self-Storage −1.07pp, unexplained.**
  They exactly offset: ~$4.35M sits in a different bucket in the PDF. Rounding
  does not absorb it (Multifamily is 1.8M outside a ±0.5pp band, Self-Storage
  2.4M). Arithmetically one ~$3.8M deal bucketed differently closes all four
  types, and only Trolley Square (3.77M) and Nottingham Village (3.76M) are that
  size — but neither is plausibly self-storage. **The earlier Citizen Storage lead
  is dead:** its pref is legitimately $0 at 26Q1 and it is now gated out.
  Resolving this needs the PDF's per-type dollars; only percentages are known.
- **Hanestowne Waterstone** is `Lifecycle = Stable`, so it does not show "Dev".
  Expected as dev — check the MRI value. Since `Lifecycle` is now confirmed as the
  only strategy field there is, fixing this means correcting `vStatus` in MRI.
- **Redevelopment is NOT dev** (creator decision). Only P0000014 Crowne Plaza
  carries it, and it is not in TIAA's set, so the distinction is latent.

### Data gaps to extract

- **Investor names for KOC / Declaration / PSC — IM query pending.** `MRI_IA_Investor`
  on the MRI **IM** server (374 rows) resolves **266 of 275** relationship investor
  codes and names `KOCINV` = "Knights of Columbus - REIT Investor",
  `DCXVIA`/`DCXVIB` = "Declaration Capital PE SPV XVIA/XVIB LLC", `PSC1/2/3` =
  "Peaceable Street Capital LLC / II / III". It is **not in the app database** — it
  needs a new `QUERY_REGISTRY` entry (IM server) and VPN access.
  `get_investor_name()` already accepts an `investor_names` mapping; pass it in once
  the table lands. `TGAM → "TIAA"` stays a hardcoded alias: the literal string
  "TIAA" appears nowhere in MRI's investor master (0 of 374 rows), only as an
  `ENTITY` row named TIAA keyed to project `PTGA`, which joins to nothing.
- **Commitment data.** Live accounting has 113 `Commitment` rows across 55
  investments (53 deals); 12 deals show genuine un-funded. The separate
  `commitments` table is loaded by `data_service` and **read by nothing**, and its
  deal-level amounts are unreliable (Seasons at Bel Air $699,847 against a
  $35,453,000 commitment). Prefer the accounting rows.

### Known app-side issues, not ours to fix

- **JB Fair Park stale debt.** ISBS Interim BS runs quarterly to 2025-06-30 but the
  debt line (acct 2150 "SENIOR FINANCING") appears at **one date only**,
  2022-12-31, for 66,363,992. Assets ≈ equity ≈ $10.9M without it, accounts
  5190/7030/7060 have **no rows at all**, and reserve accounts 1130/1140 are empty
  — the facility was never drawn. `get_isbs_debt_balance` keeps the stale figure
  alive because an active MRI loan exists (LoanID 335). True drawn ≈ $0.
- **`loan_terms_str` renders a literal `nan`** for JB Fair Park
  (`'6.40% | nan | 7/1/2031 (+1X12)'`) where `vIntType` is null.
- **`/api/waterfall-setup/<id>/investors` fabricates ownership.** With all owners at
  0% it returns an equal split — 45MAIN comes back 50/50. Not user-visible today
  (the tree display prefers the truthful 0.00%, and the template buttons only render
  when a deal has no waterfall), and only 3 entities portfolio-wide could trigger
  it, all single-investor. At exactly 0.5 the prefill would also mislabel an
  operating partner as "Pref".
- **`get_available_quarters()` has no vcode filter**, so one deal's newly loaded
  quarter appears on every deal's dropdown.

---

## Method notes worth keeping

- **Live access** is a Bearer token against
  `app-waterfall-dev-v2.icyplant-026fb2db.eastus.azurecontainerapps.io`, via
  `scripts/live_api.py` (`WF_TOKEN` env var, raises `TokenExpired` on 401/403).
  Direct Postgres and MRI were reachable earlier in the session but the VPN dropped;
  the REST API is the reliable route. Always prove live first:
  `/api/data/version`, `/api/data/config` (`actuals_through`), and
  `available_quarters` containing 2026-Q2/Q3.
- **Pagination trap.** `/api/data/tables/<t>/rows` caps `page_size` at 500 and pages
  with LIMIT/OFFSET and no tiebreaker, so multi-page pulls silently duplicate and
  drop rows. Narrow every request until the match set fits one page — for
  relationships that means one request per entity; for `isbs_interim_bs` it means
  filtering by `vcode` **and** `vAccount`. `filter__` is case-insensitive
  *contains*, so post-filter to exact matches (`TGAM` also matches `TGAM2`/`TGAM3`).
  This bit the analysis twice; both times the app was fine and the harness was wrong.
- **`float('nan') or ""` returns NaN, not `""`.** Cost two real bugs. Use a
  NaN-safe `_s()` helper on every deals-table string field.
- **`InvestmentID` is not unique.** `ASTONC` and `MCCORD` are each shared by two
  deals. Key on `vcode`.
- All five service modules take **DataFrames and injected providers**, never HTTP, so
  in-app there is no pagination at all and each module is testable standalone. Each
  has a `__main__` self-test that pulls live and asserts against the PDF.
