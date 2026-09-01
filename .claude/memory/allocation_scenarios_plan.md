# Deal Allocation Scenarios + Net Returns PDF — PLAN (Sep 1 2026, nothing built)

Artifact: https://claude.ai/code/artifact/07eae40b-cc8e-4d99-b2c9-0113d3549714
Baseline outputs: `TIAA - Windsor Square Net Returns 9.1.2026.pdf` (one-pager) +
the emailed "PSC TGA VI LLC" allocation table (JV roll-up), both in Jim's OneDrive.

## The ask (Jim, Sep 1 2026)
"Can we share $5M of Windsor Square with PSCKOC I LLC? What would the returns look
like?" Historically: paste an Excel waterfall page into the model, divert $5M to a
PSCKOC tab, run it **as if Peaceable were already fully caught up in the promote**.
Requirements: KOC may decline, so the original (TIAA-only) scenario must survive;
any amount of any deal to any JV, on the fly; print a one-page net-returns PDF per
investor (baseline: `TIAA - Windsor Square Net Returns 9.1.2026.pdf`) plus the
emailed JV roll-up table showing how the new deal changes the JV's blended returns.

## What already exists (verified against Azure PG, Sep 1)
- **Multi-relationship stacks already work.** `prospect_entities` role='ppi_relationship'
  with `ownership_pct` = pro-rata slice, `terms_json` = term sheet; `validate_stack()`
  already enforces slices summing to 100 and per-relationship participant/fee/promote
  sanity. A second relationship at 26.11% IS the $5M slice, structurally.
- **Windsor's stack**: vehicle PPIWIND (commitment 19,150,000 = 70% of equity, OPWATER
  30%); one relationship TGA6 @ slice 100 (`am_fee 0.95, min_irr 9, promote 20,
  promote_shared 80, existing_entity true`); participants TGAM 90 lp / INV6 10 psc.
- **JV topology** (`relationships`): PPIBET/PPIFVH/PPIPAR/PPIPRE/PPISPA -> TGA6 (the five
  deals in the emailed table); PPIWIND -> TGA6 would be the sixth. TGA6 -> TGAM 90 / INV6 10.
- **PSCKOC members**: KCREIT 85%, PSC1 15%, PCBLE 0% (carry units). PSCKOC's own deals
  route via INVCOC / INVPR1 / INVSPM / PIG6 (65.96%) / PPIAIR.
- **`ppi_upstream_service.build_ppi_results()`** already returns per-participant
  contributions/distributions/fees/promote/IRR/MOIC, an anniversary-year `annual_table`
  (step | recipient | CF/Cap per entity), per-relationship fee schedules, and the
  PSC1-consolidated summary. The PDF is a **re-pivot of this payload**, not new math.
- **Print-to-PDF house pattern**: One Pager uses `@media print` + `@page { margin: 0 }`
  plus a page-title blank during print. **No PDF library is installed and none is needed.**

## The four real gaps
1. **The stack is per-DEAL, not per-scenario.** `prospect_entities` has no `scenario_id`;
   `prospect_scenarios` binds only cash-flow source + assumption overrides + income
   adjustments. Declaring KOC today OVERWRITES the TIAA-only base case. This is the
   core blocker for "KOC might decline."
2. **Slices are percentages.** Jim works in dollars ("$5M of this investment").
3. **No "already caught up on promote" mode.** And **PSCKOC has NO waterfall rows** in
   the shared `waterfalls` table (checked: 104 vcodes, no PSCKOC) — the slice has
   nothing to run through.
4. **No net-returns one-pager and no JV roll-up report.**

## Plan

### Phase 0 — tie the app to the Excel FIRST (blocking for investor-facing output)
App says TGAM **12.32% IRR / 1.59x**; the 9/1/2026 PDF and the emailed table say
**11.8% / 8.0% ROE / 1.57x**. Do not print investor PDFs from the app until this ties.
Leading suspect: **TGA6's stored waterfall has no partnership-cost `Amt` step**, yet the
PDF shows "PSC & Investor Partnership Costs ($7,500)/yr" (TGA22 carries exactly this
shape: `Amt` PSCMAN 12,500/qtr). Also check month-end vs actual payment dates and the
0.72 -> 0.70 promote tie change (worth about -0.09pp on TGAM).

### Phase 1 — Allocation scenarios (stack becomes scenario-scoped)
- `scenario_id` (nullable FK) on `prospect_entities` + `prospect_investors`. NULL = the
  deal's base stack. `get_stack(prospect_id, scenario_id)` returns the scenario's rows
  when any exist, else the base — same fallback shape as the Argus cascade.
- **Copy-on-write**: "Create allocation scenario" clones the base stack rows into the
  scenario. The stack is a handful of rows, so a full clone keeps CRUD, validation and
  the Builder round-trip unchanged.
- **Waterfall steps stay NOT scenario-scoped** — and that is correct: TGA6's and PSCKOC's
  stored waterfalls are real entities' real terms. Scenario scoping covers only the
  DECLARATION (who participates, at what slice). Placeholder JVs keep getting distinct
  `NR{deal}-{n}` ids, so two hypothetical JVs cannot collide.
- Thread `scenario_id` through `/ppi-stack` GET/PUT/build/steps and `build_ppi_results`.
- UI: the PPI panel gains the scenario selector already on the page plus a banner
  ("Stack: Base" / "Stack: KOC $5M slice") and a "Copy stack from..." action.

### Phase 2 — dollar slices + the caught-up promote
- Relationship gains `slice_basis` ('pct' | 'amount') + `slice_amount`. On 'amount',
  `slice_pct = amount / vehicle PE commitment` and the **host relationship auto-absorbs
  the residual** (enter $5M once; TGA6 falls to 73.89% by itself). Validation: amounts
  under the vehicle PE; exactly one residual holder.
- `promote_state: 'caught_up'` term. Two implementations, because a linked entity's
  stored waterfall must never be rewritten:
  (a) **placeholder/new JV** -> `build_relationship_steps` emits the post-promote
      Share/Tag tier at the residual iOrder and omits the IRR gate;
  (b) **linked existing entity** (PSCKOC) -> a run option
      `gates_satisfied={'PSCKOC'}` on `run_upstream_waterfall_period`: IRR/Promote steps
      for those entities allocate 0 and cash flows straight to the residual tier.
- **Build PSCKOC's permanent waterfall once** (KCREIT 85 / PSC1 15 / PCBLE carry, AM fee
  plus expense steps per the LLC agreement) via Waterfall Setup, so the slice has
  something to run through and the AM-side PSCKOC report picks it up too.

### Phase 3 — Net returns one-pager (print -> PDF)
- New `net_returns_service.build_net_returns(deal, scenario, relationship, investor)`
  re-pivoting the existing `alloc` rows + seeded states into exactly the baseline PDF's
  lines: capital account units at close / full funding + ownership %; JV-level investor /
  PSC / total equity by year; **JV IRR before fees, avg ROE, MOIC**; CF available from the
  PE investment; partnership costs; investor pro-rata CF share; PSC AM fee (pro-rata);
  PSC CF promote; operating CF net of fees/promote; capital transactions before fees; PSC
  promote capital event; investor capital-event participation before promote; CF from
  capital events; investor total CF; investor ROE by year; summary IRR / avg ROE / MOIC.
  The gross-vs-net pairing ("before fees" alongside net) is the one genuinely new
  computation — everything else is a re-pivot.
- New print view + route (`/net-returns-print?deal=&scenario=&relationship=&investor=`),
  landscape `@page`, 11 year columns, One Pager print conventions. Browser -> PDF.
- Same payload feeds a new Excel tab, so the emailed workbook and the PDF cannot disagree.

### Phase 4 — JV Allocation Summary (the emailed table)
One row per deal in a JV entity: Type | Strategy | Sale Yr | Total Pref Equity | PSC JV
Equity | % Owned | {Investor} Equity | IRR | ROE | MOIC plus a bold total row.
- Deals already in the JV: reached by walking `relationships` (PPIxxx -> JV), computed on
  the existing AM path (`get_cached_deal_result` + upstream waterfalls) — the same
  traversal Portfolio Analysis already does.
- The NB prospect blends in as one synthetic row from the selected allocation scenario.
- Type/Strategy from `deals.Asset_Type` / investment strategy; Sale Yr from the sale-date
  cascade. Runs for TGA6 and for PSCKOC (KOC's own $5M view). Print + Excel.

## ANSWERED by Jim, Sep 1 2026 (supersedes the open decisions below)
1. **"Fully caught up in the promote" = the deal stands on its own**, not factoring in the
   portfolio's history/performance. Combined with the Belair PDF's catch-up note
   ("estimated based upon this investment's contribution to the PSC/KofC JV"), this means a
   STANDALONE run: promote_base/promote_carry accumulate from this deal only.
   **The engine already does this when running one deal — Phase 2(b) `gates_satisfied` is
   NOT needed and is dropped from the plan.**
2. **The $5M comes off the top**, not out of TIAA's share. So the TGA6 relationship falls
   to 73.89% and its own 90/10 applies to the reduced base: TGAM 12,735,000 /
   INV6 1,415,000 / PSCKOC 5,000,000 (KCREIT 4,250,000 / PSC1 750,000). PSC's co-invest
   shrinks pro-rata alongside TIAA's.
3. **Underwritten basis while building the portfolio** — use the underwritten projections,
   not the AM engine's actual+forecast blend. `isbs_uw_supplements` is the load path.
   See "TGA6 underwriting gap" below for exactly what is missing.
4. **PSCKOC has its own fee structure** and the $5M runs completely through it — AM fees,
   deal expenses and catch-up. Terms recovered in [psckoc_structure.md](psckoc_structure.md).

## TGA6 underwriting gap (measured against Azure PG, Sep 1 2026)
| Emailed row | PPI entity | InvestmentID | In `deals`? | UW rows | Waterfall |
|---|---|---|---|---|---|
| Presidential Arms | PPIPRE | PRES | yes P0000119 | **none** (108 Interim IS actual rows at 6/30/2026 only) | **none** |
| Fairview Heights | PPIFVH | FAIRVH | yes P0000117 | **none** | **none** |
| Elme Bethesda | PPIBET | Elme | **no row** | none | none |
| Parma | PPIPAR | PARMA | **no row** | none | none |
| Prestige American SS | PPISPA | AMERI | **no row** | none | none |

- **`isbs_uw_supplements` today holds ONLY account 7073** (56 rows, PE disposition
  proceeds) — it was built narrowly for the U/W ROE denominator. It has no revenue,
  expense, NOI or debt-service accounts.
- **But the pipe is already general**: `_append_uw_supplements()` stamps
  `vSource='Projected IS'` on whatever rows the table holds, and those flow into
  `isbs_raw` and the forecast assembly. So full UW projections CAN be loaded through it
  with no code change.
- **Lightest sufficient ask** — the roll-up needs each deal's PE cash flow stream, not a
  full property model. Accounts **7071 (PEACEABLE CASH FLOW)** and **7073 (PEACEABLE
  DISPOSITION PROCEEDS)** already carry exactly that, and the U/W ROE path already reads
  both with correct sign conventions. Loading 7071 + 7073 per deal means **no deal-level
  waterfall is required** for these five — feed the PE stream straight into the TGA6
  upstream waterfall.
- Also needed for the display columns: `deals` rows for Elme / PARMA / AMERI, and each
  deal's total pref commitment, asset type, strategy and projected sale year.
- **Underwritten-basis run mode**: the AM path always blends actuals before
  `actuals_through`. A roll-up "as underwritten" must run with `actuals_through=None` and
  the forecast forced to the Projected IS source — a per-run option, new but small.

## Superseded open decisions (kept for the record)
1. **"Fully caught up in the promote" — confirm the mechanics.** Reading taken above:
   the gate is treated as already met, so the slice splits at post-promote percentages
   from period one (PCBLE takes its carry share immediately). Alternative reading: the
   catch-up runs but seeded with the JV's existing cumulative promote balance.
2. **Does the $5M come out of TIAA's dollars or off the top?** Plan assumes it reduces
   the TGA6 slice (TIAA keeps 90/10 of a smaller base) — i.e. the vehicle's PE is fixed
   at 19,150,000 and the slices re-split it.
3. **"As underwritten" in the roll-up** — recompute the five existing TGA6 deals live
   from current projections, or freeze the underwriting figures that were emailed?
4. **Does the KOC slice pay the same 0.95% AM fee / 20% promote terms**, or PSCKOC's own?


## PHASE 0 RESULT (Sep 1 2026) — the waterfall ties; the gap is convention + projection

Target: `TIAA - Windsor Square Net Returns 9.1.2026.pdf`. Its own printed cash flows solve
to **11.740%** (the page prints "11.8%"), MOIC 1.565x on contributions of 17,241,750 and
distributions of 26,987,075.

| step | TGAM IRR | delta |
|---|---|---|
| app as shipped (before this session) | 12.319% | |
| + TGA6 venture-expense step added (the real fix) | 12.281% | -0.038 |
| + annual buckets, i.e. the PDF's convention | 11.516% | **-0.766** |
| + startup cost inside the equity basis (17,241,750) | 11.504% | -0.011 |
| **residual vs the PDF's 11.740%** | | **-0.236** |

**Distribution frequency is the whole story (-0.766pp).** The app models actual monthly
distributions; the Excel models one annual distribution per anniversary. Same dollars,
earlier receipt, higher IRR. This is a presentation convention, NOT an error — and it
yields the rule the one-pager must follow:

> **The net-returns page must compute IRR / ROE / MOIC from the same annual figures it
> prints.** Print 11 annual columns and derive the metrics from monthly cash flows and the
> page will not reproduce its own numbers — an investor can catch that. Deriving them from
> the printed columns makes the app tie to the Excel by construction.

**The -0.236pp residual is the underlying projection, not the waterfall.** Cash reaching
PPIWIND over the hold: operating app 7,534,821 vs PDF 7,377,607 (+157,214); capital app
23,823,037 vs PDF 24,206,135 (-383,098); net **-225,884**, of which TIAA's 90% is -203,295.
The app's Argus-based forecast is simply a different vintage from the Excel's. Do not
"fix" it — the one-pager should name its cash-flow source so the difference is explainable.

### DONE — TGA6 venture-expense step (was genuinely missing)
`Amt` at iOrder 1 on both CF_WF and Cap_WF, PropCode **`TGA6_EXP`**, `mAmount` **1875**,
vNotes "Quarterly: $7,500/yr venture expense (pari-passu)". Saved on Azure through
`save_waterfall_steps` (audit trail written). Notes:
- `mAmount` on `Amt` is capped **per quarter** (tracker key `(entity, iOrder, quarter)`), so
  an annual $7,500 is entered as 1,875. Same iOrder on both waterfalls so CF and Cap share
  one quarterly cap and cannot double-charge.
- Recipient is `TGA6_EXP`, following the `{entity}_EXP` convention PSC1/PSC3 already use.
  **Deliberate**: routing it to PSCMAN (the TGA22 convention) would flow a third-party
  expense into PSCMAN's income and therefore into PSC1's consolidated return.
- Verified: yr5 charges exactly 7,500. A quarter with no distribution event charges nothing
  (the waterfall does not run) — same limitation as the `;accrue` fee.
- **`save_waterfall_steps()` wrote NULL into `dteffective`** for the 11 pre-existing rows;
  restored by hand. `dteffective` is a required column in `loaders.load_waterfalls`. Worth a
  look before the next programmatic save.

### STILL OPEN — TGA6 has no 9% coupon, and the Excel appears to have one
The PDF's structure legend reads "Coupon **9.00%**", but TGA6's stored Cap_WF has **no
`Pref` step** — only a 9% **IRR gate** at tie 20. Different mechanics: a coupon accrues and
compounds, an IRR gate is a lookback test.
Arithmetic evidence from the PDF's own sale year: PSC promote 689,625 = 20% of 3,448,125,
so **1,095,647 was paid ahead of the promote**. A 9% coupon on the declining balance
(17,241,750 then 14,493,239) totals 6,769,324 against 5,891,178 of operating cash actually
received net of fees — an **878,146 shortfall before compounding**, which compounds toward
the implied 1,095,647. The app's IRR gate instead allocated only **352,885**.
So the Excel is running an accruing 9% coupon the app is not. Material, and it restructures
TGA6's tie order, so it needs Jim's decision rather than a unilateral fix.
**Jim is sending the source Excel model**, which will settle this exactly rather than by
inference.


## PHASE 0 CLOSED (Sep 1 2026) — the source Excel resolves it exactly

Model: `~\OneDrive\Documents\TIAA - Windsor Square - Base Case - 8.19.2026.xlsx`,
sheet **`TIAA_AMB`** (the sheet that prints the PDF). Read with openpyxl after stripping a
corrupt `<definedNames>` block whose value is the literal text "Formula removed, name can be
deleted." — openpyxl refuses the file otherwise; copy it and delete that block first.

### The promote is the closed form the app ALREADY has
`TIAA_AMB` rows 152-154 and 43:
```
r152  TIAA CF before promote = participation (r44) + ROC (r42) + operating net of fees (r40)
r153  each column / (1+9%)^((date - D11)/365)        <- discounted at the COUPON
O153  = SUM(r153)                                    = XNPV(9%, pre-promote CFs)
O154  = O153 * (1+9%)^((MAX(dates) - D11)/365)       = the excess, compounded to exit
r43   promote = -O154 * 20%
```
Reproduced **to the penny**: O153 2,240,516.40 vs sheet 2,240,516.39; O154 3,448,126.22 vs
3,448,126.21; promote 689,625.24 exact. **Discount base is `D11` = 2026-09-30**
(`=PSC!B84`), not the close date — every period is then a clean 365-day year.

This is exactly `x = XNPV(target, cfs) x (1+target)^t`, i.e. the same closed form that
replaced `irr_needed_distribution()`'s brentq search during the valuation NAV work — only
the sign convention differs (the engine solves the amount NEEDED to reach the target; the
Excel takes the EXCESS above it).

### The tie-out
| | TIAA IRR |
|---|---|
| app, monthly, TGA6 as structured today | 12.281% |
| app, annual buckets (the Excel's convention) | 11.568% |
| **app, annual + the Excel's promote structure** | **11.748%** |
| **Excel as printed** | **11.740%** |

**0.008pp apart.** Phase 0's earlier -0.236pp "projection" residual largely cancels: the
Excel-shape promote is smaller than the app's current one, which offsets the app's lower
exit pool. The two models agree.

### The one real structural difference
- **Excel**: the investor keeps its FULL pro-rata share; PSC then takes 20% of the excess
  above a 9% IRR. On the app's cash flows that promote is **670,736**.
- **App today**: tie 20 tops TGAM up to a 9% IRR (`IRR` vState), then tie 30 splits the
  residual TGAM .70 / INV6 .26 / PSCMAN .04. That concedes 20 points of the tie-30 pool =
  **856,189**, i.e. **185,452 too much promote**, routed 4/20 to PSCMAN and 16/20 to INV6.
- The two shapes are only *approximately* equivalent and the gap drifts with the cash-flow
  shape, so calibrating tie-30 FXRates is not a fix.

**Recommendation — a new `IRRPromote` vState.** `FXRate` = promote share (0.20),
`nPercent` = the coupon (0.09), `vNotes` = the investor whose IRR is measured; allocates
`FXRate x max(0, xnpv(nPercent, investor_cfs) x (1+nPercent)^t)` to the recipient at the
sale tie. ~15 lines reusing `metrics.xnpv`, and it makes the app tie to the Excel **by
construction** rather than by calibration. Then TGA6 Cap_WF becomes:
tie 10 ROC 90/10 -> tie 20 `IRRPromote` PSC 20% @ 9% -> tie 30 residual 90/10.

### CORRECTION to the earlier session note: TGA6 is RIGHT to have no Pref step
The earlier inference — that the Excel pays an accrued 9% coupon at the sale ahead of the
promote — was **wrong**. Rows 62-64 do accrue an "Investor Due / Paid / Accrued" coupon, but
the accrued balance (166,801 at year 5) is **never paid**: it only (i) caps the operating
distribution via `r63 = MIN(due, distributable x 90%)` and (ii) supplies the 9% discount
rate in the promote calc. The coupon is a **hurdle rate, not a payable**. So the absence of
a `Pref` step on TGA6 is correct, and no Pref step should be added. Good thing the model was
read rather than the arithmetic trusted.

### Two smaller confirmations
- **AM fee base**: Excel charges 0.95% on **row 16 = TOTAL** equity, then allocates 90% to
  TIAA (`r38 = -r69 x 90%`). The app charges `AMFee` on TGAM's own capital, which is
  0.95% x 90% x total — arithmetically identical. No change needed.
- **The Excel's AM fee is subordinated and accrues**: `r69 = MIN(due, Investor Paid)` with
  `r70` carrying the shortfall forward. TGA6's `AMFee` has no `;accrue`. Row 70 is zero
  throughout this deal so it does not bite here, but **TGA6's AMFee should carry `;accrue`**
  to match the model on a weaker deal.
