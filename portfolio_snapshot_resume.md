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
                                                <this note>   ← you are here
```

| Step | Commit | What landed | Self-test |
|---|---|---|---|
| Spec | `010d1e92ce5fca5ff11e27b42af096be9bf8401c` | `portfolio_snapshot_build_spec.md` | — |
| 1 | `4d2e90f7d07344e05609fa960546c83fbc79a7b4` | `portfolio_snapshot_service.py` — deal resolution, fund grouping, look-through %, quarter-aware sold exclusion, child roll-up, investor names | 14/14 |
| 2 | `cbe23bf82c4db3a1a80c11ba287eb619f9f33dfa` | `portfolio_snapshot_persistence.py` + 3 tables + `PROTECTED_TABLES` (additive, 32→35) | 35/35 |
| 3a | `0b7a8097211a494e202caac16ad4c6b06f09607a` | `portfolio_snapshot_operating.py` — Econ Occ, NOI ×3, Expected/Actual Growth | 10/10 structure |
| 4a | `a0fe820bac6154b0f1112689b3e96b9bcc9b1b87` | `portfolio_snapshot_loan.py` — Debt, LTV, YTD DSCR, Debt Yield, Rate/Maturity | 21/26 (see below) |
| 5a | `13ff99f7937848a9805dd0f87027ad84ddef401e` | `portfolio_snapshot_financial.py` — cap-stack zone, 4 scaled TIAA columns, manual Net ROE/ITD, footnotes | 28/28 |

Only one existing file has ever been touched: `database.py`, additively, to add the
three new tables to `PROTECTED_TABLES`.

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

1. **Step 6a — Summary subtab backend.** Charts + two blank editable narrative
   boxes (`portfolio_snapshot_comments`, `scope='report'`). Allocations by asset
   type and deal type; per the spec the "Excluding Development Deals" subtotal
   lives here and on Loan — decide the deal-type field first (see open items).
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

## Open creator / data items

### Blocking correctness

- **`deals.Investment_Strategy` is empty (0/110 live, build `09fe220ae0da`).**
  Dev detection was switched to pure `Investment_Strategy ∈ {Development,
  New Construction}` per creator decision, so **nothing is currently classified as
  dev**. Consequences today: all 9 development deals render numeric ratios on the
  Loan subtab instead of "Dev", JB Fair Park shows **LTV 457.7%** off its stale
  debt, and its Debt reverts to the stale ISBS 66,363,992 instead of the
  77,368,000 committed facility. This is why Step 4a's self-test is 21/26.
  *Fix either way:* (a) populate the column — no MRI query selects it and it is
  absent from `mri_service.MRI_COLUMNS`, so it needs an extraction change to
  `Prop_Info_Core.sql` + `MRI_COLUMNS`; or (b) point the two call sites back at
  `strategy` (= `Investment_Strategy or Lifecycle`, the One Pager's own
  precedence), which works today and prefers `Investment_Strategy` once present.
  `Lifecycle` is 97/110 and holds Development 24 / New Construction 2 / Value-Add
  31 / Income 22 / Stable 16 / Redevelopment 1 / Lease up 1.
- **45th & Main (P0000089) — MRI fix pending.** Both owner edges are 0% in PMX
  (`PPI45M → 45MAIN` and `OPEVGR → 45MAIN`), while MRI's IM copy shows 100%. The
  deal therefore has no look-through % and is flagged "ownership % unavailable".
  One row in PMX `IA_Relationship` fixes it; the guardrail should stay regardless.
  Note the Waterfall Setup screen shows 0.00% for it, and `/investors` separately
  fabricates a 50/50 split (see below).

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
- **Hanestowne Waterstone** is `Lifecycle = Stable`, so it will not show "Dev" even
  once the strategy field is fed. Expected as dev — check the MRI value.
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
