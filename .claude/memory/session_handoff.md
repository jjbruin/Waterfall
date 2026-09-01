# Session Handoff — through Sep 1 2026 (v408 live)

Rolling handoff for the next session/developer. Update in place; keep only what is still
live. Deploy history lives in CLAUDE.md (SHA-pinned). Supersedes the Aug 28 / v396 handoff.

## Where things stand
- **Live**: `v408` = `e5ef21d`, healthy, 100% traffic. Merge of five Jim-approved items —
  Charlene's `8de3d53` + `9043c92`, the upstream `Pref` first-period fix `16bcf8b`, the
  Total Current Funding row + scope-placed footnotes, and the Jefferson Eastchase override.
- **main == origin/main**, everything pushed.
- The Sep 1 chain ran v405 → v408. See CLAUDE.md for the SHA-pinned list.

## STANDING RULE — read before deploying anything
CLAUDE.md "Deploying Changes" now carries Jim's pre-deploy symptom-repair check (added
Sep 1 after `50695d9` shipped an override that zeroed correct data on 12 deals).
**Verifying that a commit does what its message says is NOT verifying its premise.** Flag a
symptom repair to Jim, with affected deals and figures, BEFORE building the image.

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

### Open with Charlene
1. **Prompt C item 4** — "remove the auto-generated footnotes from both views" is ambiguous.
   After the footnote rework the Financial list holds exactly the two notes she wants kept;
   the remaining system-written notes are methodology paragraphs on
   `SnapshotOperating.vue`, `SnapshotLoan.vue` and `SnapshotSummary.vue`. **She must pick.**
2. **The print PDF has not been visually re-rendered** (`node_modules` absent in the agent
   worktree). The marker-suppression *rules* are proven by a static guardrail; the paper is
   not. **Needs a visual pass before anything goes out.**
3. **`9043c92`'s two new per-deal hardcodes**, accepted by Jim as-is but still debt:
   `one_pager.AT_CLOSE_FORCE_SUPPRESS = {"P0000066"}` (writes `0` as an unknown-sentinel over
   a complete footing measurement — Pegasus revenue 973,606 − expenses 348,917 = NOI 624,689)
   and `portfolio_snapshot_loan.DEBT_FREE_DEALS = {"P0000066"}` (defended as underivable, but
   `Sale_Status` and valuation-row coverage both separate Pegasus from City West).
4. **Jefferson Eastchase `GROUP_OVERRIDES`** — deployed at Jim's direction. It is a per-deal
   hardcode that disagrees with both the ownership data (61.2% via TGA23, one route) and the
   reference PDF (which prints 61% inside "Total PSC TGA 2023 LLC"). The durable fix is an
   MRI ownership row; delete the entry when that exists.
5. **`DEV_DISPLAY_EXCEPTIONS` Pegasus entry** still carries Charlene's own
   **"FLAG FOR CONFIRMATION"** in the code, unanswered.

### Settled — do not reopen
- **At-Close Year-0 gate stays as deployed** (Jim, Sep 1). All 12 affected deals have
  complete data; 10 lost real figures (Brainerd Bldg E 1,662,811, Pegasus 624,689). A
  Brainerd/Pegasus At-Close reconciling to a real `at_close_noi` behind a blank column is
  **expected**. Kill switch if ever revisited: `AT_CLOSE_REQUIRE_YEAR0_ROW = False`.

---

## Known defects / debt worth carrying forward
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
