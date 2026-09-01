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

## Open decisions for Jim
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
