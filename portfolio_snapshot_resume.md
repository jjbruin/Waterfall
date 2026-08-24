# PORTFOLIO SNAPSHOT — RESUME NOTE

Written 2026-08-24. Read `portfolio_snapshot_build_spec.md` (commit `010d1e9`) first
for the full plan; this file is only "where we are and what's next".

**Nothing is merged, nothing is pushed, `main` is untouched at `986881f`.**
No route, no blueprint registration, no Vue. **None of the five service modules is
imported by anything** — the whole feature is inert until Step 7 wires it up.

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
                  └─ (this note, a18881a)
                     └─ wip/portfolio-snapshot-step6   Step 6a Summary
                                                       + Loan dev-display   ← you are here
```

| Step | Commit | What landed | Self-test |
|---|---|---|---|
| Spec | `010d1e92ce5fca5ff11e27b42af096be9bf8401c` | `portfolio_snapshot_build_spec.md` | — |
| 1 | `4d2e90f7d07344e05609fa960546c83fbc79a7b4` | `portfolio_snapshot_service.py` — deal resolution, fund grouping, look-through %, quarter-aware sold exclusion, child roll-up, investor names | 14/14 |
| 2 | `cbe23bf82c4db3a1a80c11ba287eb619f9f33dfa` | `portfolio_snapshot_persistence.py` + 3 tables + `PROTECTED_TABLES` (additive, 32→35) | 35/35 |
| 3a | `0b7a8097211a494e202caac16ad4c6b06f09607a` | `portfolio_snapshot_operating.py` — Econ Occ, NOI ×3, Expected/Actual Growth | 10/10 structure |
| 4a | `a0fe820bac6154b0f1112689b3e96b9bcc9b1b87` | `portfolio_snapshot_loan.py` — Debt, LTV, YTD DSCR, Debt Yield, Rate/Maturity | 21/26 (see below) |
| 5a | `13ff99f7937848a9805dd0f87027ad84ddef401e` | `portfolio_snapshot_financial.py` — cap-stack zone, 4 scaled TIAA columns, manual Net ROE/ITD, footnotes | 28/28 |
| 6a | this branch | `portfolio_snapshot_summary.py` — asset + deal-type allocations, 2 blank narratives; plus Loan dev-display refinements | Summary 41/43, Loan 45/46 |

Existing files touched: `database.py` (additive, `PROTECTED_TABLES`), and in Step 6a
`portfolio_snapshot_loan.py` + `portfolio_snapshot_operating.py` — both prior
steps' own files, not app code. Nothing outside the feature has been modified, and
no module is imported by any blueprint or route.

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

## What's next

1. ~~**Step 6a — Summary subtab backend.**~~ **DONE on this branch** — see
   "Step 6a (Summary) — landed" below. Backend is now complete for all four subtabs.
2. **Step 7 — UI shell + Vue components.** Mirror `ReportsView.vue`:
   `PortfolioSnapshotView.vue` (header controls + subtab bar + one bundle fetch)
   plus `components/snapshot/Snapshot{Summary,Financial,Operating,Loan}.vue`,
   presentational only. Then the route, the blueprint
   (`/api/portfolio-snapshot`), the additive `AppSidebar.vue` nav entry under
   Asset Management (+ path in `amRoutes`), and paired `/excel` endpoints.
3. **Guardrails + isolation verification.**
4. **Snapshot freeze** — deliberately NOT copied in Step 2. The One Pager fires
   `_save_snapshot()` on CEO approval; the equivalent needs the four subtabs to
   exist. Until it is built, an approved page re-renders live data rather than
   what was approved.
5. **Polish manual-entry UX**, then the **performance pass** (bundle fetch,
   `get_cached_deal_result`, keep `full_data` out of the One Pager provider).

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
- **TGA6 zero-pref deals.** Fairview Heights and Presidential Arms have Total Pref
  0, so Invested/Commitment are a real $0 rather than pending. Render `0` or `—`?
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
