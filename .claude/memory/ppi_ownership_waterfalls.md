# PPI Ownership Waterfalls (New Business) — Plan (Aug 27 2026)

## What Jim asked for
Peaceable raises outside capital for part of each preferred equity investment.
A PE position can be shared pro-rata across two or more **investor
relationships**, each with its own waterfall between Peaceable (PSC) and its
investor(s). PSC earns: its pro-rata share, an annual **asset management fee on
invested capital balances**, and **promotes gated on investor IRR lookbacks
computed NET of AM fees paid**. Promotes are sometimes shared with investors,
sometimes 100% PSC. Model the stack on a new deal; results in a dropdown
between Annual Operating Forecast and Debt Service; on closing, the waterfalls
migrate to the permanent database and are visible on the AM side.

## Key insight: the engine already exists
`run_recursive_upstream_waterfalls()` (waterfall.py:1977) powers PSCKOC and
Portfolio Analysis today. It takes deal-level allocations (cf_alloc/cap_alloc),
all waterfall steps, and relationships, and recurses upstream through any
entity that has a waterfall. The vStates match the term sheet exactly:
- `AMFee` — annual rate (nPercent, raw %), mAmount = periods/yr, vNotes =
  source investor; pool-neutral; capped once per quarter via
  amt_quarterly_tracker; supports exclusions.
- `IRR` — hurdle gate computing the distribution needed to reach a target IRR
  from full cashflow history **net of AM fees and expenses** (built for
  exactly this promote convention). Works as gated lead with FXRate.
- `Promote` — cumulative catch-up (FXRate = carry share, nPercent = target).
- `Share`/`Tag` at one iOrder — pro-rata splits, incl. shared promotes.
NB work = declaration + seeding + wiring + display, NOT new waterfall math.

## Architecture

### Stack shape
```
Deal (N0000003)
 └─ PPI vehicle (PPIWIND)              <- deal-level Cap/CF waterfall participant
     ├─ Relationship A (e.g. PPIW-KC)   pro-rata slice a%   [own waterfall]
     │    ├─ PSC co-invest
     │    └─ Investor (e.g. KCREIT)
     └─ Relationship B (PPIW-TIAA)      slice b%            [own waterfall]
          └─ ...
```
- The **vehicle's waterfall** (PropCode = vehicle ID in shared `waterfalls`
  table) is a single-tier Share/Tag pro-rata split into relationship entity
  IDs (a% / b%). Single-relationship deals skip the intermediate: the
  relationship waterfall sits directly on the vehicle ID.
- Each **relationship waterfall** (keyed by relationship entity ID) encodes:
  AMFee (PSC recipient, investor source), Pref and/or gated IRR tiers (net of
  fee by engine construction), promote tiers with FXRates expressing
  shared-vs-100%-PSC promote, residual pro-rata.
- Stored in the shared `waterfalls` table from day one, keyed by permanent
  entity IDs (planned_entity_id / MRI-convention IDs where known). "Migration"
  is then mostly a no-op for the stack itself.

### Seeding
Upstream IRR/pref/fee math needs invested capital per participant.
`run_recursive_upstream_waterfalls(pre_seeded_states=...)` accepts synthetic
InvestorStates: build contributions at the deal's seed date (close - 1 day)
from `prospect_investors` (commitment / ownership_pct) — investor slice of the
vehicle's PE amount from compute_capital_budget, PSC co-invest as declared.
Reuses the declared-beats-inferred rule.

### Relationships (ownership tree)
The engine consults `relationships` for entities without waterfalls. NB builds
a small synthetic relationships DataFrame from the declared stack (vehicle ->
relationship -> participants) — no writes to the AM relationships table until
onboarding.

## Data model
- `prospect_entities`: role='ppi_relationship' rows, parent_entity_id = the
  vehicle's entity row, planned_entity_id = permanent ID; vehicle row
  role='pe_vehicle'. ownership_pct = pro-rata slice.
- `prospect_investors`: participants per relationship entity (PSC co-invest +
  outside investors), ownership_pct within the relationship, commitment,
  investor_type ('psc' | 'lp').
- Relationship waterfall steps: shared `waterfalls` table, vcode = the
  relationship entity ID (PROTECTED_TABLES, audit trail via
  save_waterfall_steps — all existing).
- Per-relationship terms JSON on the entity row (notes or a new column):
  am_fee_pct, fee_period (Q/A), hurdles, promote splits — the Builder's source
  so a rebuild round-trips (same lossless principle as the deal Builder).

## Computation flow
1. `_run_prospect_analysis` computes the deal as today.
2. New `ppi_upstream_service.build_ppi_results(result, stack)`:
   - filter cf_alloc/cap_alloc rows to the vehicle PropCode(s),
   - run run_recursive_upstream_waterfalls twice (CF_WF, Cap_WF) with stack
     steps + synthetic relationships + pre-seeded states + amfee tracking,
   - assemble per participant: contributions, AM fees paid/received, pref,
     ROC, promote, total distributions, net IRR / MOIC / ROE (net of fees for
     investors; PSC economics = co-invest + fees + promote),
   - AM fee schedule (date, fee base balance, fee) per relationship,
   - per-participant XIRR cashflow lists.
3. Payload `ppi_waterfalls` on the /analyze response; audit Excel gains a
   "PPI Ownership" tab from the same data.

## UI
- **Setup panel** (left column, below Waterfall Builder): "PPI Ownership
  Stack" — add relationships (name, permanent ID, slice %), participants
  (PSC % / investor %), terms (AM fee % + frequency, hurdle(s), promote % and
  shared-with-investor toggle). "Build & Save PPI Waterfalls" generates and
  saves the vehicle split + per-relationship steps (explicit write, same
  reads-never-writes rule as the deal Builder). Steps tab preview.
- **Results dropdown** "PPI Ownership Waterfalls" between Annual Operating
  Forecast and Debt Service: per relationship — participant returns table
  (Contribs, AM Fees, Pref, ROC, Promote, Total, IRR, MOIC), fee schedule
  expander, XIRR cashflows expander. PSC summary card across relationships
  (total fees, total promote, blended IRR on co-invest).

## Migration at close (onboarding wizard additions)
- Stack waterfalls are already in the permanent table under permanent IDs —
  survive untouched (excluded from any N-vcode re-key except rows actually
  keyed to the N-vcode).
- Write real `relationships` rows (vehicle -> relationships -> investors) so
  the AM ownership tree and Portfolio Analysis pick the stack up natively —
  AM-side visibility comes free through the existing Portfolio Analysis view.
- Deal-level N->P re-key already exists; extend to the vehicle split's vcode
  reference if it was N-keyed.

## Phases
1. **Stack model + CRUD** — entity roles, terms JSON, endpoints, validation
   (slices sum to 100, participant shares sum, fee/hurdle sanity).
2. **Builder + save** — templates that emit the vehicle split and
   relationship steps (AMFee/IRR/Promote patterns from PSCKOC, e.g. TGA22).
3. **Computation** — seeding, upstream runs, results assembly, diagnostics
   (fee base trace, promote gate trace).
4. **UI dropdown + Excel tab.**
5. **Onboarding migration + AM verification** (Portfolio Analysis renders the
   onboarded stack).

Verification: hand-modeled two-relationship Windsor case (e.g. 60/40 slices,
0.95% fee, 9% net-IRR gate, 20% promote shared on one side only) cross-checked
against the PSCKOC engine's known-good behaviors.

## Answers (Jim, Aug 27) — design resolutions

1. **AM fee base = funded capital reduced by ROC, paid quarterly.** Matches
   AMFee's balance-based behavior (capital_outstanding falls with returns of
   capital) with mAmount = 4 periods/yr and the per-quarter cap. Builder
   default: quarterly.
2. **PSCKOC's catch-up is unique — do not template it.** The STANDARD deal:
   CF and capital shared pro-rata between PSC and the investor(s); AM fee paid
   out of the investor's distribution (AMFee: vNotes = investor, PropCode =
   PSC — pool-neutral deduction, exactly the engine semantics); PSC earns a
   promote AFTER the investor achieves a minimum IRR. Standard template per
   relationship:
     - Tier 1: pro-rata Share/Tag (PSC co-invest % / investor %)
     - AMFee step (investor source -> PSC), quarterly on funded balance
     - Cap_WF gate: investor IRR-as-lead (nPercent = min IRR, net of fees by
       engine construction, FXRate = investor share) + PSC Tag; then the
       post-promote tier Share/Tag with PSC's promoted share
     - "Promote shared with investors" = FXRates on the post-gate tier
   The Promote (catch-up) vState stays available as an "advanced" step for
   PSCKOC-style deals but is not in the default template.
3. **Entity IDs are assigned at closing — but a relationship may already
   exist.** Builder offers two bindings per relationship: (a) link to an
   EXISTING entity (picker over the AM ownership tree / entities that already
   have waterfalls) — the NB deal is modeled as an addition to that JV, and at
   close the deal simply routes into the existing relationship (its permanent
   waterfall may already exist; offer reuse vs override); (b) a PLACEHOLDER ID
   (e.g. NR{deal}{n}) re-keyed to the MRI-assigned ID by the onboarding
   wizard — the wizard gains a rename step (waterfalls vcode/PropCode rows +
   prospect rows), mirroring the N->P deal re-key.
4. **One relationship can have one or many investors.** Participant table is
   per-relationship with N investor rows + PSC; pro-rata inside the
   relationship by ownership_pct; the IRR promote gate applies per investor
   (each investor's lookback gates its own share) — engine supports multiple
   gated leads at separate tie levels; validation warns when investor shares
   within a relationship do not sum to 100% net of PSC.

