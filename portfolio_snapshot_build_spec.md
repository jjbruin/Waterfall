# PORTFOLIO SNAPSHOT — BUILD SPEC

**Goal:** An isolated Portfolio Snapshot tab under Asset Management. User picks investor + quarter
→ walks the ownership chain to find that investor's deals → groups by fund → renders 4 subtabs,
reusing One Pager data read-only, scaling only the 4 TIAA columns by ownership, with
approval-gated editable comments/footnotes/values. **If it breaks, it breaks only itself.**

---

## Architecture (isolation is core — deploy-blind)

- Own files, components, services, tables
- Read-only reuse of One Pager functions (call, never modify)
- Copy-pasted approval pipeline (mirror One Pager's states) — **NOT shared code**
- New tables; no changes to existing tables
- Only shared-code touch: additive nav entry under Asset Management (One Pager nav pattern —
  router-link in the AM section body at `AppSidebar.vue` ~429-436 + path in `amRoutes` ~25)
- UI mirrors `ReportsView.vue`: one view component, `activeTab` state + conditional rendering for
  subtabs, shared dropdowns above the subtab bar

---

## Structure

```
Asset Management → Portfolio Snapshot
├── [Investor dropdown] [Quarter dropdown]  (shared, persist across subtabs)
├── Subtab 1: Summary   (charts + 2 blank editable narrative boxes)
├── Subtab 2: Financial (cap-stack deal-level + 4 TIAA columns scaled + Net ROE/ITD manual
│                        + anchored footnotes)
├── Subtab 3: Operating (Econ Occ, NOI ×3, Growth + per-deal comments)
└── Subtab 4: Loan      (Rate/Maturity/Debt/YTD DSCR/LTV/Debt Yield + per-deal comments)
```

---

## Foundation logic

- **Deal resolution:** walk the relationships chain investor → funds → deals (**NOT** the
  waterfall-based finder — new deals lack waterfalls and would vanish). New deals appear
  automatically via relationships.
- **Investor names:** IM's `MRI_IA_Investor` joined to PMX relationships, LEFT JOIN with fallback
  to the raw code; hardcoded alias `TGAM` → "TIAA". (KOC=`KOCINV`, Declaration=`DCXVIA`/`DCXVIB`,
  PSC=`PSC1/2/3` resolve from data. Azure can reach both servers.)
- **Fund grouping:** falls out of the traversal (path to a deal = its fund). Groups: Individual
  Investments (the 5 non-fund TGAM deals) + TGA22/23/24/25.
- **Multi-hop ownership:** product of normalized ownership % along the chain to the investor,
  excluding 0%-holders (`PSCMAN`, carried-interest). This produces "% of Pref."
- **Sold exclusion:** exclude if `Sale_Status='SOLD'` AND `Sale_Date <= quarter_end`. **MUST be
  quarter-aware** — do NOT reuse `get_inv_display` (today-keyed, breaks historical reports).

---

## Metric map

**Reused from One Pager (read-only):** Debt, Total Pref, Ptr Equity, Total Cap, %, Econ Occ,
NOI (At Close / U/W YE / Projected YE), YTD DSCR.

- **% of Pref** = product of normalized ownership % along chain
  (verified: Nottingham 48.4851% × 85% = 41.21% ≈ 41%)
- **Invested** = % of Pref × funded pref
- **Total Commitment** = % of Pref × Total Pref (Total Pref = the committed/pledged total)
- **Un-funded** = Total Commitment − Invested
- **Expected Growth** = (U/W YE NOI − At Close NOI) / At Close NOI
- **Actual Growth** = (Projected YE NOI − At Close NOI) / At Close NOI
- **LTV** = Debt ÷ `mIncomeCapConcludedValue` (Valuation table, 26 YE valuation; if `dtValuation`
  is in 2026 but not year-end, fall back to the 2025 valuation)
- **Debt Yield** = TTM NOI ÷ current debt; for **DEVELOPMENT** deals use `mOrigLoanAmt`
  (Loan Customs / full commitment) instead of current drawn balance
- **Rate / Maturity** = use loan data but apply OWN logic: count the deal's loans, output
  "Various" for 2+ differing-term loans (like Brainerd); do not read a literal "Various" from
  source
- **Net ROE** = **MANUAL ENTRY for v1** (typed from Acct Excel; method is net of fund-level
  expenses weighted by dollars invested and time; automate later)
- **ITD Distributions** = **MANUAL ENTRY for v1** (typed from Excel; footnote-1 fee allocation;
  automate later). Raw sum is available from the One Pager if wanted.
- **"Excluding Development Deals" / dev detection** = Investment Strategy field

**Scaling applies ONLY to the 4 TIAA columns** (% of Pref, Invested, Total Commitment, Un-funded).
All other columns (cap-stack columns on Subtab 2, all of Subtabs 3-4) are full deal-level, **NOT
scaled**. Operating metrics (Occ/NOI/DSCR) are property-level and never scaled.

---

## Editable elements

All through the copy-pasted approval pipeline: draft → pending → approved, gated.

- **Subtab 1:** 2 narrative text boxes, blank by default (no auto-generation)
- **Subtab 2:** anchored auto-numbered footnotes — analyst attaches a footnote to a column
  (e.g. ITD Distributions) → system auto-assigns "(n)" at the anchor AND an editable box at the
  bottom → add/remove re-sequences all numbers (anchor markers + bottom list stay synced).
  PLUS Net ROE and ITD as manual number-entry fields.
- **Subtabs 3 & 4:** per-deal comment boxes (Subtab 3 = operating comment, Subtab 4 = loan
  comment — same deal, two independent fields)

**Tables:** `portfolio_snapshot_comments` (narrative + per-deal, via scope/scope_key/field),
`portfolio_snapshot_footnotes` (anchored, auto-numbered), `portfolio_snapshot_values`
(manual Net ROE / ITD per investor/quarter/deal/field). Add all to `PROTECTED_TABLES`.
Build a shared editable-element pattern handling text, footnotes, and numbers through the same
save/approval mechanism.

---

## Guardrails

- Flag any missing/pending metric — never fake or show wrong numbers
- Pref-equity-comment cross-check: compare computed investor mapping against `pe_cap_comment`
  where it exists; flag disagreements for human review (soft, never fail the build)
- New deal with incomplete ownership chain → flag ("ownership % unavailable"), don't show
  unscaled numbers

---

## File layout

- `vue_app/src/views/PortfolioSnapshotView.vue` (shell: header controls + subtab bar +
  bundle fetch + 4 tab bodies)
- `vue_app/src/components/snapshot/Snapshot{Summary,Financial,Operating,Loan}.vue`
  (presentational, props in, no API calls)
- `vue_app/src/router/index.ts` (+ one lazy route)
- `vue_app/src/components/layout/AppSidebar.vue` (+ router-link in AM body; + `amRoutes` path)
- `flask_app/api/portfolio_snapshot.py` (`portfolio_snapshot_bp` + `_get_data()`)
- `flask_app/services/portfolio_snapshot_service.py` (all computation)
- `flask_app/__init__.py` (+ `register_blueprint`, `url_prefix "/api/portfolio-snapshot"`)
- Paired `/excel` endpoint per section (Reports convention)

---

## 10-step build sequence (each step VERIFIED before implementing)

**Pre-build:** Size_Sqf fix deployed ✓; confirm LTV = Debt ÷ Value, and dev-deal detection is
consistent between "Excluding Dev" and Debt Yield's `mOrigLoanAmt` path

1. **Foundation** (resolution/names/ownership/sold) — test: TIAA + 26Q2 → right 26 deals, right
   funds, City West excluded, Nottingham % of Pref = 41%
2. **Editable persistence** (3 tables) + copy-pasted approval pipeline + shared
   text/footnote/number pattern
3. **Subtab 3 (Operating) FIRST** — easiest, proves the pattern
4. **Subtab 4 (Loan)** — incl. LTV + Debt Yield dev handling
5. **Subtab 2 (Financial)** — 4 TIAA columns + Net ROE/ITD manual entry + anchored footnotes
6. **Subtab 1 (Summary)**
7. **UI shell** — mirror `ReportsView.vue` (activeTab subtabs, shared dropdowns above)
8. **Guardrails** + verify isolation
9. **Polish manual-entry UX** (Net ROE / ITD typed from Excel)
10. **Performance pass** (26 waterfalls per load → `get_cached_deal_result` cache or drop
    `full_data` where accrued pref isn't needed)

---

## Verifies during build (non-blocking)

- LTV: confirm Debt ÷ Value (the valuation-table source implies it)
- Dev-deal detection consistent between "Excluding Dev" subtotal and Debt Yield's `mOrigLoanAmt`
  treatment

---

## Resolved (no action)

Foundation (grouping, names, ownership, sold, 4 TIAA columns, cap-stack reuse), all metric
formulas, architecture, all build decisions, UI pattern, inherited bugs (resolved via session
fixes). Size_Sqf blocker deployed.

---

## Workflow

Build step by step. Each step: detailed instructions → verify the approach (data/logic steps
verified against live data BEFORE implementing) → implement → confirm it works → next step.
**Do not build ahead. Start with Step 1 only when asked.**
