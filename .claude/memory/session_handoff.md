# Session Handoff — through Sep 2 2026 (v416 live)

Rolling handoff for the next session/developer. Update in place; keep only what is still
live. Deploy history lives in CLAUDE.md (SHA-pinned). Supersedes the Sep 1 / v408 handoff.

## Where things stand
- **Live**: `v416` = `eb7520d`, healthy, 100% traffic. **main == origin/main**, all pushed.
- Sep 2 ran **v409 → v416, eight revisions**. Six were engine corrections found by pulling
  on one question from Jim about a $300,000 contribution; two were Charlene's snapshot work.
- **Seven new guardrail scripts** now cover this ground, all passing:
  `capital_reversal_sign_check` 19, `pref_excess_cf_check` 6, `sale_date_month_end_check` 10,
  `snapshot_kept_sold_stack_check` 38, `snapshot_east_manchester_check` 26,
  `snapshot_itd_roe_check` 30, `snapshot_footnotes_freeform_check` 28,
  `snapshot_print_formatting_check` 44.

## STANDING RULE — read before deploying anything
CLAUDE.md "Deploying Changes" carries Jim's pre-deploy symptom-repair check (added Sep 1
after `50695d9` shipped an override that zeroed correct data on 12 deals).
**Verifying that a commit does what its message says is NOT verifying its premise.** Flag a
symptom repair to Jim, with affected deals and figures, BEFORE building the image. It fired
for real on Sep 2 against `eb7520d` and Jim took the stopgap knowingly — see below.

## A guardrail habit worth keeping
Several before/after scripts asserted **"BEFORE the feature was absent/broken"**. Those fail
the moment the baseline is a commit that already has the fix, which happened three times in
one day as the chain moved. **State the invariant about the AFTER side and report what the
baseline happened to show.** All known cases are converted; write new ones that way.

---

# TRACK 1 — Investor groups around a deal (the KOC slice)

Plan: `.claude/memory/allocation_scenarios_plan.md` ·
artifact https://claude.ai/code/artifact/07eae40b-cc8e-4d99-b2c9-0113d3549714
Agreement analysis: `.claude/memory/tga6_amb6_agreement_review.md` ·
PSC KOC terms: `.claude/memory/psckoc_structure.md`

**The ask**: share $5M of Windsor Square with PSCKOC without losing the TIAA-only case;
make it repeatable for any amount / any deal / any JV; produce the one-page net-returns PDF
and the JV allocation roll-up.

### Phase 0 — tie the app to the Excel ✅ DONE
Reconciled to the source model (`TIAA - Windsor Square - Base Case - 8.19.2026.xlsx`, sheet
`TIAA_AMB`). Waterfall mechanics tie; the residual is cash-flow vintage.
- Added TGA6's missing venture-expense step (`Amt` TGA6_EXP 1,875/qtr = $7,500/yr).
- Rebuilt TGA6 to the executed LLC agreement + Jim's commercial terms — see Track 1 detail
  below. Live TGAM 12.178% monthly / 11.480% annual-bucket vs the Excel's 11.740%; the gap
  is 0.236pp projection vintage + ~0.03pp (now fixed) engine defect.
- **`IRRPromote` vState: NOT needed, retired.** The existing `IRR` step in the right
  position is the mechanism.
- **The one-pager must compute IRR/ROE/MOIC from the annual figures it prints** — the app
  runs monthly, the Excel annually, worth 0.766pp. Print the columns and derive the metrics
  from them or the page will not reproduce its own numbers.

### Phases 1–4 — NOT STARTED
1. **Allocation scenarios** — `scenario_id` (nullable) on `prospect_entities` +
   `prospect_investors`, NULL = the deal's base stack, copy-on-write clone. Waterfall steps
   stay un-scoped (they are real entities' real terms). This is the blocker for "KOC might
   decline".
2. **Dollar slices** — `slice_basis`/`slice_amount`; enter $5M once and the host
   relationship absorbs the residual. ("Off the top": TGAM 12,735,000 / INV6 1,415,000 /
   PSCKOC 5,000,000 → KCREIT 4,250,000 / PSC1 750,000.) The "caught-up promote" needs NO
   flag — it is PSC KOC §6.06(d), and a standalone single-deal run already does it.
   PSCKOC still has **no waterfall rows anywhere**; terms are recovered in
   `psckoc_structure.md` (8% coupon pari-passu 85/15, 1.50% AM fee subordinated + accruing,
   20% catch-up, $15k venture costs) and map onto existing vStates.
   **Note**: PSCKOC's pref compounds **9/30**, the engine compounds 12/31 — configurable
   compounding month is a correctness gate before any PSCKOC page ships.
   **Also**: new PSCKOC deals go in at 90/10 while older ones keep 85/15, so pref/ROC use
   the per-deal ratio while catch-up/residual use the blended Capital Unit ratio.
3. **Net returns one-pager** — re-pivot of the existing payload; print-to-PDF via the One
   Pager `@media print` pattern (no PDF library needed). Both TIAA and KOC layouts.
4. **JV allocation roll-up** — the emailed table. **Blocked on data**: all five existing
   TGA6 deals lack underwriting rows and waterfalls, three lack `deals` rows entirely.
   Lightest ask is **7071 + 7073 per deal** into `isbs_uw_supplements` — then no deal-level
   waterfall is needed, the PE stream feeds the upstream waterfall directly.

### TGA6 — live structure (rebuilt Sep 1 to the executed agreement)
```
CF_WF : 1 Amt TGA6_EXP 1875/qtr · 10 Pref 9% .90/.10 · 20 Share .70/.30 · 900/901 AMFee
Cap_WF: 1 Amt · 10 Pref 9% · 20 Initial ROC .90/.10 · 25 IRR 9% · 30 Share .70/.30 · 900/901 AMFee
```
Removed PSCMAN's .04 promote share — the agreement's "PSC" is **PSC Investment TGA VI LLC =
INV6**, not PSC Manager LLC. Jim's original .70/.30 instruction was right. AMFee now charges
**both** members per §5.06 (the single TGAM row collected only 90% of the fee).
**Hurdle convention**: 8% pre-TGA6 (TGA22/23/24/25), 9% from TGA6. Do not "correct" TGA22.
**TGA22 needs NO changes** — the 10% LifeStorage yield dragging the blended 8% test is the
bargained-for economics, and the fee waiver is already `;exclude:PEGASU`.

### AMB6 §8.4(b) — mechanism PROVEN, configuration blocked
The provenance feature **already exists**: `run_upstream_waterfall_period` routes to a
`Promote_WF` when the source tier's vtranstype contains "Promote", and forwards the tag
through passthrough hops. Verified in `scratchpad/promote_wf_test.py`.
Remaining: derive the Exhibit C band from **funded** non-PSC1 capital (accounting is booking
the missing AMB6 contributions — see that section in the agreement review), configure the
TGA6 tie-30 split + AMB6 `Promote_WF` with PSC1 omitted, and find a durable route for INV6PU
that survives MRI refresh. **Expect AM-side movement when the contributions land**: PSCMAN's
AMB6 fee is currently charged on ~zero seeded capital.

---

# TRACK 3 — Sep 2 engine corrections (start here if returns look wrong)

All six came from ONE question: Jim asked why OPMCCORD's $300,000 into 30 Bearfoot showed
$5,388 of pref and no return of capital. Each answer exposed the next. Full detail in
`.claude/memory/capital_reversal_and_psc3.md`.

| # | Fix | Revision |
|---|---|---|
| 1 | **A reversed entry no longer moves capital twice.** MRI reverses by re-posting with the OPPOSITE SIGN under the same MajorType/Typename; six sites took `abs()`. Rule now in `loaders.capital_after`. | v412 |
| 2 | **Excess Cash Flow pays pref down.** It sits BELOW pref, so a partner cannot receive it while pref is outstanding; the seeding counted only TypeID 1019. `reports_service` always applied both — the two paths had been disagreeing. | v412 |
| 3 | **A mid-month sale is not a month-end sale.** `month_end()` was applied at PARSE time, so the real date was gone. Now `sale_actual` (pref stops, loan repaid, closing settles) vs `sale_me` (monthly grid). | v414 |
| 4 | **Terminal NOI starts the month AFTER the sale month** — it used to begin WITH it while the cash schedule also gave the seller that month. | v414 |
| 5 | **A kept-sold deal reports the stack it had while held** (`last_held_quarter`). | v414 |
| 6 | ITD stored in MILLIONS; footnote removal; print separators and cell font. | v411, v412 |

**Two rules that are easy to break again**, both learned the hard way:
- **Do not floor a running capital total per row.** JB Fair Park's reversal sorts BEFORE the
  entry it reverses; a per-row `max(0.0, ...)` swallows it and both remaining contributions
  land. Floor at the point of USE (`capital_outstanding`).
- **Do not add a magnitude heuristic** to decide a unit. 0.9 is a legitimate $0.9M and a
  legitimate 0.9%. Manual figures store the unit their column DISPLAYS.

**Confirmed against the published page**: the corrected partner equity ties the 26Q1
reference PDF on three deals that did not tie before — JB Fair Park 3.9, Pegasus 2.6,
Cocoplum 23.4, and Cocoplum Total Cap 108.9 to the decimal.

**Portfolio effect**: accrued pref 159.9M → 152.5M (−7.4M) across 37 of 101 pairs. Five
pairs went UP, correctly — they have reversed pref payments that `abs()` had been counting
as extra payments.

### Found but NOT fixed
- **One day of pref is dropped per investor per year.** `accrue_to_date` splits at year end
  then resumes at `date(year+1, 1, 1)`, skipping 31 Dec → 1 Jan. Visible as a 30-day January
  accrual where 31 is due (~$74/yr on $300k; order of $37k/yr portfolio-wide, understated).
- **`cap_stack.pref_equity` is capital OUTSTANDING** while three columns describe it as
  funded / invested / committed. A LIVE deal with a PARTIAL return of capital would already
  understate Invested. None on the 26Q2 page has one — dormant, not safe.

---

# TRACK 2 — Charlene's updates

Review of all 83 of her commits: `.claude/memory/commit_review_charlene.md`. Headline —
**the great majority are genuine root-cause fixes with live-data guardrails**; the
exceptions cluster on one missing concept (an asset stabilisation state) worked around
independently in six modules.

### Deployed Sep 1
| commit | what | revision |
|---|---|---|
| `0cb14ba` + `150db60` | nil participation renders 0%; prior-year budget fallback skips dev deals | v403 |
| `97d3945` | Review Tracking quarter on the JOIN | v404 |
| `7827c6f` | Snapshot Financial Total Pref = committed tranche | v405 |
| `d48469b` | PUT /value returns display/source, no bundle refetch | v406 |
| `50695d9` | At-Close zeroed for dev deals with no Year-0 row | v407 |
| `8de3d53` + `9043c92` | Waters Creek LTV exception retired; dev-tag correction | v408 |

### Deployed Sep 2
| commit | what | revision |
|---|---|---|
| `7dc7bd8` | Eastchase override withdrawn; East Manchester kept after sale; `SOLD_NA_CELLS` | v409 |
| `019b592` | Prompts A/B/C — funding row out, ITD summed, Net ROE manual everywhere; footnotes free-form; per-loan Rate/Maturity, one page per subtab, separators | v410 |
| `8534916` | ITD stored in MILLIONS (v410 printed `$0.00M` on every live cell); vertical separators; comment/input cells print in the table's font | v411 |
| `da354a3` | Capital reversals; excess-CF pref; clearing a footnote removes it | v412 |
| `7d21769` | East Manchester Net ROE typeable (`SOLD_NA_CELLS` → `{"debt"}`) | v413 |
| `bd001e8` | Mid-month sale dates; terminal NOI window; kept-sold cap stack | v414 |
| `639f023` | Footnote 2 → City West only | v415 |
| `eb7520d` | **STOPGAP** — six-deal `MANUAL_RATIO_SEEDS` + subtotals weight what the row displays | v416 |

### Open with Charlene
1. **`MANUAL_RATIO_SEEDS` is a stopgap with an expiry.** Six vcodes carry hand-typed LTV /
   DSCR / Debt Yield, and since `eb7520d` those typed figures **drive investor-facing
   totals** (26Q2 Portfolio LTV 63.9%, DSCR 1.75x, TGA 6 DSCR 1.27x). Jim approved it
   knowingly so quarter-end reports could go out. The sharpest item is **Presidential Arms'
   typed 1.1x replacing a computed 3.8x** — that single override is what moves TGA 6.
   Root cause is three structural data gaps, none fixed: no valuation on/before year-end,
   no full YTD Interim IS + BS principal movement, no complete quarter of actual NOI.
   A weekday 09:00 reminder runs as scheduled task `retire-manual-ratio-seeds`; delete it
   when the dict empties. Retiring is one deletion per deal.
2. **`scripts/live_api.py` IS STILL NOT COMMITTED.** About a dozen guardrails import it,
   including `snapshot_financial_pdf_check`, `snapshot_pe_basis_check`, the module
   self-tests and the 126-check script behind `eb7520d`. **Nobody but Charlene can
   reproduce them**, and that blocked independent verification three times on Sep 2. This
   is the single highest-value thing she could commit.
3. **Known defect shipped in v416**: the Loan row's warning flag reads "the computed
   figures … still feed the subtotals" — true when written in `2a3fabe`, made false by
   `eb7520d` an hour later. Visible on the row tooltip. One line.
4. **Footnote 2 is settled** — City West only, decided by Jim Sep 2. East Manchester came
   out because its Net ROE is typeable and its ITD shows.
5. **East Manchester Debt stays n/a DELIBERATELY**, and Total Cap 6.0M rather than 15.6M.
   Jim raised the inconsistency (the row is read at the last held quarter, where the
   9,641,912 WAS outstanding); Charlene's instruction is to keep it blank and Jim confirmed
   following it. Reasoning is written at the `SOLD_NA_CELLS` definition. **Not an oversight
   — reopening it means changing what the row is for.**
6. **Jefferson Eastchase `GROUP_OVERRIDES` — RESOLVED Sep 2.** The Sep 1 override was
   withdrawn in `7dc7bd8`; the work order had meant East MANCHESTER. The ordinary rule had
   Eastchase in TGA 2023 correctly all along.
7. **`DEV_DISPLAY_EXCEPTIONS` Pegasus entry** still carries Charlene's own
   **"FLAG FOR CONFIRMATION"** in the code, unanswered.
8. **`9043c92`'s two per-deal hardcodes** remain debt: `AT_CLOSE_FORCE_SUPPRESS` and
   `DEBT_FREE_DEALS`, both `{"P0000066"}`.

### Resolved Sep 2 — no longer open
- The print PDF **has** now been rendered and inspected end to end. `snapshot_print_check.mjs`
  takes `WF_UPSTREAM`, so it can print against a LOCAL Flask instead of live — that is how an
  unmerged backend change gets a PDF before it ships.
- The footnote rework shipped: free-form text, every note editable and deletable including
  the code-defined standing ones, clearing the text removes the note and its marker.

### Settled — do not reopen
- **At-Close Year-0 gate stays as deployed** (Jim, Sep 1). All 12 affected deals have
  complete data; 10 lost real figures (Brainerd Bldg E 1,662,811, Pegasus 624,689). A
  Brainerd/Pegasus At-Close reconciling to a real `at_close_noi` behind a blank column is
  **expected**. Kill switch if ever revisited: `AT_CLOSE_REQUIRE_YEAR0_ROW = False`.

---

## Known defects / debt worth carrying forward
- **One day of pref is dropped per investor per year** — `accrue_to_date` skips
  31 Dec → 1 Jan when it splits at the year boundary. See Track 3.
- **`cap_stack.pref_equity` is capital OUTSTANDING** while three columns call it
  funded/invested/committed. Dormant; bites the first live deal with a partial ROC.
- **The `deals.Sale_Date` column is not what the model uses.** Priority is sale override →
  `event_dates` projected disposition → horizon/max maturity. The local snapshot has no
  `event_dates` table at all, so a local run falls back and will NOT reproduce a live sale
  date — inject one, or use `sale_date_override`, when testing anything sale-related.
- **The local `waterfall.db` accounting feed ends 2026-06-02.** Anything that turns on a
  later event — East Manchester's 6/25/2026 sale, the 8/13/2026 contribution — cannot be
  seen locally and must be simulated. Say so when reporting; do not conclude "no defect"
  from a snapshot that predates the data.
- **`save_waterfall_steps()` writes NULL into `dteffective`** — restore it after any
  programmatic save. `dteffective` is required by `loaders.load_waterfalls`.
- **`accounting_feed.sql` has no `TRIM()`** — 3,641 of 12,827 rows carry untrimmed IDs
  (`'TGA22'` and `'TGA22 '` both exist). The app strips at load; raw exports do not.
- **`accounting_feed.sql` LEFT JOIN has `AND S.MajorType = ...` in the ON clause** — an
  unclassified row survives with NULL MajorType and then vanishes silently from every
  consumer. Only 4 rows today, but it is the mechanism for invisible loss.
- **The Flask dev server drops connections on heavy computes** (Portfolio Analysis, PSCKOC)
  non-deterministically. Measure in-process instead — `scratchpad/blast_inproc.py` is the
  pattern; `compute_portfolio_actual(eid, data, start_year, horizon_years, pro_yr_base)`.
- **`Investment_Strategy` is 0 of 134 populated** on live, so the dev classification runs
  entirely off the `Lifecycle` proxy. Populating it is the cheap half of retiring the
  dev-special-case debt; a real stabilisation state is the full fix.
