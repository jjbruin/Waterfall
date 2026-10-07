# Open items — live work, with evidence and an owner

Every item below was **re-verified against the working tree on Sep 11 2026**, not copied
forward on trust. Several things the Aug 2026 session notes described as open turned out
to be fixed; those are listed at the bottom so nobody re-opens them.

**The discipline this file exists to enforce.** These items were buried inside dated
session narratives in `MEMORY.md`, where a live gap and a story about a Thursday looked
identical and carried equal authority. An item here is a claim about the code as it
stands. When you touch one:

- **Re-verify before acting.** Each item carries the check that proves it — run it. A
  line number is a hint, not a fact; they drift (the Aug notes cite `one_pager.py:507`
  for code that now sits at `:518`).
- **Move it to "Resolved" when it ships, with the commit.** Do not delete it — a reader
  needs to know the question was answered, not just that it vanished.
- **A decision is not a fix.** Items in §2 are blocked on a human answer, not on code.

Full narrative and the underlying diagnostics: `session_log_aug2026.md` (the Aug 5–13
One Pager audit). Related topic files: `capital_reversal_and_psc3.md` (the sign bug,
fixed, plus `MANUAL_RATIO_SEEDS` and its expiry), `onepager_audit_q1_2026.md`.

---

## 21. Sold deals still carry stale debt / loans — HELD, off the 26Q3 critical path (Oct 6 2026)

**Held by Charlene's call, Oct 6 2026:** none of this changes Investment Metrics or the One
Pager's PRINTED figures, and (except 21.6) not the Portfolio Snapshot, so it waits — but it has to be solved,
because the leaks below sit in Deal Analysis, the assistant, the One Pager's loan-terms text
and the valuation cycles. Source: `payoffs_to_book.csv` (Downloads): 10 sold / paid-off deals
plus the Berger parent roll-up, whose ISBS 2150/2152 balances and MRI loan rows are still
live — **$249.99M of balance-sheet debt on the 10 deals** (Berger's $190.8M is the sum of its
four children and is not counted twice). Audited Oct 6 2026 in-process against the local copy
(pulled Oct 5) AND by read-only GETs against production at `v585`; **local and production agree
on every screen compared.**

The 10 deals: Bear Run, Heritage Hills, Lindenbrooke, Stonecliffe (Berger children, sold
2026-04-22), 30 Bearfoot and Donald Lynch (2026-09-04), Quakertown and Airport Plaza
(2026-03-04), Clima Secur (2026-07-01), East Manchester (2026-06-25, the only one whose MRI
Paid Off date is present).

**Already correct, verified, no action:** Dashboard (KPIs, Capitalization, Maturity buckets —
KPI debt outstanding $2,042.4M, none of the 11 in any list), Surveillance, both Dashboard Excel
exports, Portfolio Snapshot (for 10 of the 11 — EXCEPT Donald Lynch's parent, see 21.6, which it still lists), the
One Pager's Debt cell (`debt_display` None once sold as of the quarter), Reports / PE exposure
and Sold Portfolio (read no debt), Ownership, Review Tracking (lists the deals, no debt fields),
Investment Metrics (First Lien is the loan at ORIGINATION, not a balance — Berger $197.8M,
Bearfoot $12.6M, Quakertown $12.2M, Clima $10.7M, East Manchester $10.0M, Airport $6.5M), and
the deal's own balance sheet / Property Financials Excel (its statement of record, by design).
**Pre-sale quarters are right** — the same `is_sold_as_of` test gives debt before the sale and a
blank after: Berger $190,763,346 at 26Q1 on both the One Pager and the Snapshot, Quakertown
$11,611,949 at 25Q4, Airport Plaza $5,762,332 at 25Q4, East Manchester $9,641,912 at 26Q1,
Clima Secur $10,577,673 at 26Q2, Bear Run $80,043,352 at 26Q1.

**THE REAL FIX is data, not code:** book each payoff in MRI — a `Paid Off` Loan_Date event on
every loan (or a Disposition in `event_dates`). That removes the stale loans from 21.2, 21.3
and 21.4 at once. Berger has no loan row at all (its four children carry them).

### 21.1 One Pager still prints Loan Terms for a sold deal
`OnePagerView.vue` prints `loan_terms_str` / `second_loan_terms_str` with no sold gate, so 26Q3
shows e.g. Bear Run "2.93% | Fixed | 8/1/2030" and "7.28% | Fixed | 8/1/2030" beside a blank Debt
cell. The Snapshot blanks rate and maturity (`portfolio_snapshot_loan.py`); the One Pager does
not. The API payload also still carries the raw `cap_stack.debt` beside `debt_display=None`.
**Fix:** gate the loan-terms text on `cap.sold_suppressed`. **Closed when:** a 26Q3 One Pager
for Bear Run prints no loan terms, and 26Q1 still prints them.

### 21.2 Deal Analysis models the sale at the stale loan's maturity
`Sale_Date` is deliberately not consulted (priority: UI override, `event_dates`, then horizon
end / max loan maturity), so with the loan live the modeled sale is Bear Run 2030-08-31 (actual
2026-04-22), Heritage Hills / Lindenbrooke / Stonecliffe / Berger 2030-09-30, Quakertown
2028-01-31 (2026-03-04), Clima Secur 2031-09-30 (2026-07-01), East Manchester 2031-01-31
(2026-06-25), Airport Plaza 2026-11-30 (2026-03-04). Only 30 Bearfoot and Donald Lynch are right
(2026-09-30), via `sale_overrides`. Debt-service schedules run past the sale. **Fix:** booking
the payoffs (above). **Closed when:** `/api/deals/<vcode>/debt-service` lists no loan for these
deals and the modeled sale is the actual one.

### 21.3 `_filter_paid_off_loans` works per ROW, not per loan
East Manchester's Paid Off row (2026-06-25) is dropped, but its Origination and Maturity rows
keep LoanID 257 alive (maturity 2031-01-11) — so the loan is still modeled. The comment on
`_collapse_loan_date_events` says a repaid loan carries only a Paid Off event; it does not.
**Fix:** drop the whole LoanID when any of its rows is Paid Off. NOTE: `loaders.py` has a bare
`to_datetime` that raises on East Manchester's `T`-format `dtEvent` — **local only; production
does NOT hit it** (Deal Analysis returns normally there for East Manchester, Quakertown, Bear
Run and Berger).

### 21.4 Assistant tools
* `get_loan_details` reads `data["loans"]`, a key that does not exist, and returns nothing for
  every deal.
* `get_debt_service` reads `original_amount` / `rate` / `rate_type` / `lender`, which the Loan
  object does not have (0 / blank), and its annual `ending_balance` is one loan's last row, not
  the deal's sum.
* `get_capitalization` has no sold filter and double-counts a parent roll-up: **Berger returns
  $381.5M of debt, LTV 1.50, against $190.8M** — the one number here that is actively
  misleading. Bear Run alone returns $80.0M, LTV 0.84.
* `get_one_pager` returns the loan terms and the raw debt beside `debt_display=None`.
**Closed when:** each tool is correct on a live deal, refuses or flags a sold deal, and Berger
returns $190.8M.

### 21.5 Valuation cycles seed with `exclude_sold` at seed time
Wrong in both directions. Cycle 1 (as of 2026-12-31) holds **30 Bearfoot and Donald Lynch, both
sold 2026-09-04**, and Bearfoot's stored NAV carries **$11,973,158.81 of debt**
(`valuation_nav_results`). Cycle 2 (as of 2025-12-31) is MISSING deals that were owned then —
all 11 were — because they were marked sold by the time it was seeded (production holds only
Donald Lynch of the 11). **Fix:** seed by the cycle's own as-of (sold on or before it, not sold
today) — the Snapshot's `is_sold_as_of` is the test to reuse — then reseed.

### 21.6 Donald Lynch's PARENT (P0000049) is not marked SOLD — the Snapshot still lists it — DO NOT HOLD
Donald Lynch is two vcodes under InvestmentID MCCORD: **P0000073** (the child, Property_Count 0)
is `Sale_Status = SOLD`, Sale_Date 9/4/2026; **P0000049** (the parent, Property_Count 1) carries
the same Sale_Date but a BLANK `Sale_Status` (local copy pulled Oct 5; **production NOT checked —
the admin token had expired**, so confirm in Data Explorer: `deals`, vcode P0000049).
`payoffs_to_book.csv` lists only the child, so the audit above followed P0000073 and missed this.
What the blank marker does at 26Q3:
* **Portfolio Snapshot lists Donald Lynch as a live deal** on all four subtabs (summary,
  financial, operating, loan) — for at least BCA, BRECO, DCXVIA, DCXVIB, FNKI and INVF9 (the list
  was truncated at six) — with `debt` / `debt_display` **$0.0**, a **7/1/2028 maturity**
  inherited from the child's loan, 63 months owned and a look-through %. `is_sold_as_of` returns
  False for P0000049 at 26Q3 and True for P0000073. Verified by the same rule at 26Q2 (correctly
  still listed).
* **One Pager, P0000049:** 26Q3 `debt_display` = 0.0, `sold_suppressed` False, loan terms
  "3.95% | Fixed | 7/1/2028" (the child's). P0000073's own page is right (blank at 26Q3, $9,684,943
  at 26Q2).
* **Dashboard** hides it only because `get_child_vcodes` classes BOTH vcodes as children — not
  because of the sold rule; fragile.
* **Investment Metrics is right** (it pairs the two rows and takes SOLD from the child).
**Cause:** `Sale_Status` is not in any MRI feed — it lives in the `deals` table (from
`investment_map.csv`) and an MRI refresh PRESERVES it, so it has to be set by hand on BOTH rows.
**Fix (data, one field):** set `Sale_Status = SOLD` on P0000049. **Closed when:** the Snapshot at
26Q3 no longer lists Donald Lynch for any investor while 26Q2 still does. Unlike the rest of this
item it changes what the Snapshot prints, so it should not wait with the others.

**Owner:** Charlene / Jim for the MRI payoffs; code fixes unassigned.

---

## 20. Board package section -- planned, not started (Oct 5 2026)

**Oct 7 2026: Phase 1 pp.26, 27, 29-31 live at `v598`** (
reconciliation in `board.md`). Open from it:
- **MRI `Property_Count`** -- Apple Self Storage P0000003 1 -> 16, PMAT Midwest P0000036
  1 -> 3, Prestige P0000080 1 -> 12. Then p.26 gives 98 properties, 23 wholly owned.
  Owner: whoever maintains MRI deal records.
- **p.27's mixed dates** -- the deck shows proceeds and CoC through 9/30 against a 12/31
  population. The app answers one as-of per schedule. Decision: keep that (set p.27 to
  the last CLOSED quarter), or give the schedule a second "cash through" date. Owner: **Jim**.
- **Deal count** -- the deck counts Brainerd I and II as two deals (54); MRI has one (53).
  Owner: **Jim**.

Development plan (a shared doc, private until shared): https://claude.ai/code/artifact/71e5c89f-933f-4b88-9e3e-f1b2e1d7a5b9
-- built from the Jan 14 2026 board deck (33 pages). Seven phases: 0 foundation (an
OPT-IN section -- today every new section is ticked for everyone -- plus a separate
Compensation permission, server-side aggregation, compensation tables restricted and
excluded from the assistant and from `pull_production_db.py`, an audit log), 1 portfolio
overview, 2 track record, 3 originations (pipeline holds 2 prospects; the 339 reviewed are
elsewhere), 4 projected sales returns, 5 platform model + payroll plan (nothing exists),
6 package assembly. Each phase accepted by reproducing the January deck at 12/31/25.
Decisions listed in the doc. Owner: **Jim**.

---

## 19. Local data, rates, and the PE exposure tracker (Oct 5 2026)

- **19.1 Database Tools "Export Database" does not export production.**
  `database.export_all_tables_to_zip` reads a SQLite file (`sqlite_master`), so on
  Azure it zips the container's empty local file and none of PostgreSQL. The button
  looks like it works. Use `scripts/pull_production_db.py` for a local copy; the
  button needs a PostgreSQL branch or removing. Owner: **Jim's call**.
- **19.2 Investment Metrics carries its own CAD rate** (`investment_metrics_config.
  CAD_TO_USD = 0.73`) now that `market_rates` holds Bank of Canada's published rate
  (6/30/26: 1.4210 CAD per USD = 0.7037). Two sources for one number; switching it
  MOVES Investment Metrics figures, so it needs measuring and Jim's call. Owner: **Jim**.
- **19.4 27 entities in `relationships` do not close to 100%** (found Oct 5 2026, the
  first time `ownership_pct_closure_check` ran on production's feed): the BRN-1..9,
  BURT-1..4 and TFT-1 sub-entities, INV23-P / INV24-P, PSCIF1, PPI2 and PSCMAN carry
  0.00%. Affects every RELATIONSHIPS-based trace (Upstream Analysis, Portfolio
  Analysis, PPI upstream, PSCKOC) -- not the PE exposure report, which walks
  commitments. Owner: **accounting / MRI data**.
- **19.5 Forward yield curves for refinancing estimates** -- Jim asked, Oct 5. The
  free official source is Treasury's daily par yield curve; implied forwards can be
  bootstrapped from it. It is a TREASURY curve, so a refi rate = forward + a spread
  that would be an input. SOFR swap / Term SOFR forwards are licensed. Not built.
  Owner: **Jim's call**.
- **19.3 Apple's cost convention** -- historical rate per contribution, or the
  quarter-end rate? The tracker converts at the quarter-end Bank of Canada rate until
  accounting says otherwise. Owner: **accounting**.

---

## 20. Items recovered from the deploy history (Oct 5 2026)

CLAUDE.md's inline deploy history was moved to `deploy_history.md` on Oct 5 2026.
Reading it through first turned up these, each stated as "STILL OPEN" or "NOT DONE"
in a revision note and tracked nowhere else. Every one was **re-checked against the
working tree** before being written down here; the ones that turned out to be closed
are recorded as closed rather than carried forward.

**Standing: eight open — 20.1-20.5, plus 20.8-20.10 (housekeeping, low priority). 20.11 closed Oct 5 2026.**
20.7 is not an item but the record of four things the archive called open and that
turned out to be shipped, kept so nobody re-opens them.

**20.6 is deliberately absent.** It tracked `v547` having moved nine reported figures,
three of them investor-facing coupons, without the "tell Jim first" step being closed.
**Jim reviewed them on Oct 5 2026 and confirmed MRI deal terms as the source of truth
for coupon and participation**, so the item is closed and removed; the resolution is
recorded against `v547` in `deploy_history.md`. Numbers are never reused here — a gap
means an item closed, and renumbering would break every reference written before it.

### 20.1 Investment Metrics: UW Proj. IRR and Proj Yr-1 CoC — SHIPPED `v573`, WAITING ON A deal_terms REFRESH (updated Oct 6 2026)

MRI now holds both fields (`'U/W IRR'`, `'Projected Yr 1 CoC Returns'`; NOT the older
look-alikes `'UW IRR'` / `'Projected Yr 1 CoC'`). `queries/Prop_Info_DealTerms.sql` pivots them
into `deal_terms.uw_irr` / `deal_terms.proj_yr1_coc` (latest `dtEffective` per deal, undated
figures, fractions) and `UNLOADED_FIGURES` reads them in mode `mri`. Charlene proved the SQL
in SSMS: 89 rows, the five original columns unchanged, 76 `uw_irr` and 55 `proj_yr1_coc`.
**Until `deal_terms` is refreshed on production the columns do not exist there, so every
UW IRR / Proj Yr-1 cell is still an em dash** (`unloaded_figure_field_absent`: 152).

- **Waiting on:** Charlene/Jim holding the refresh until the PRECISION question with Alay is
  answered — MRI holds whole percents (0.11) where the reference PDF shows 10.7%. Nothing
  is rounded or altered. Run only `POST /api/data/mri/refresh/Prop_Info_DealTerms`.
- **After the refresh, verify:** `unloaded_figure_field_absent` is gone and
  `unloaded_figure_value_null` shows ~21 `proj_yr1_coc` deals; seven of the nine footnote-(5)
  deals take the projected figure in three cells; Current Total Proj Yr-1 ~8.0%, UW IRR
  ~15.4%. Preview without refreshing: `scripts/investment_metrics_preview_new_fields.py`.
- **Two data questions for MRI (not code):** Plaza Del Mar (P0000116, a young deal) has
  `proj_yr1_coc` = 0.0, the only zero of 55, and footnote (5) would print it in three cells;
  30 Bearfoot (P0000001) has `uw_irr` = 0.33 against a next-highest of 0.24.
- **Owner:** Charlene (refresh), Alay (precision), whoever maintains the Investment Checklist
  (the two values).

### 20.2 Investment Metrics: first lien reproduces the reference on 42 of 76 — unassigned (the ISBS fallback was removed at `v579`; see 20.12)

Recorded at `v545`/`v546` and still stated in `investment_metrics_config.py`'s own
header comment. `v551` improved the SOURCE — origination dates are read off the raw
`mri_loans_all` frame where they exist, and maturity is never used as a proxy — but
**no printed figure moved**, because the whole live table holds four Origination rows
and all four sit on single-loan deals. Ten deals are named in the
`first_lien_origination_missing` diagnostic.

- **The check:** the config header comment, and the diagnostic on a live payload.
- **Owner:** unassigned. It is a DATA gap (MRI carries no origination date for most
  loans) before it is a code gap, so it may belong with Alay too.


### 20.12 Investment Metrics: the Total / Average shares do not add to 100% when a deal's first lien is unknown (Oct 6 2026) — decision needed, Jim / Charlene

`_total_row` / `_grand_total` use `total_of`, which sums each column on its own and skips
blanks. A deal with an unknown first lien has `total_size` None, so it leaves the Total Size
denominator, but its Pref and First-Loss DOLLARS stay in those totals — so the three shares
(`first_lien_pct + pref_pct + first_loss_pct`) are measured against different populations and
do not add to 100%. After `v579` switched the ISBS fallback off (16 first-lien dashes, was 9):
Current 101.3%, Sold 127.5%, Grand 104.8%; before it, 100.2% / 134.4% / 105.9% — the Sold row
was already wrong from nine dashed deals. The reference's own rows add to 100.0%.

- **The fix to decide on:** total the Pref and First-Loss dollars, and the shares, only over
  deals where all three pieces are known (or footnote the deals left out). It changes the
  Total rows by more than `v579` intended, so it was NOT bundled and needs a measure-first
  pass against the reference's Total / Average and Grand Total rows.
- **Owner:** Charlene / Jim.


### 20.13 Investment Metrics: a deal sold AFTER the as-of — CLOSED at `v585` (Oct 6 2026); one diagnostic follow-up

Decided by Charlene: a deal is Sold at a quarter only once its sale date is on or before the
as-of; otherwise it is Current with its figures as of that quarter. Shipped at `v585`
(`classify` in `investment_metrics.py`). At 26Q2 this moves Clima Secur, 30 Bearfoot and
870 Donald Lynch from Sold to Current, the one deliberate departure from the printed reference
page; the Sold page no longer has a footnote (4). Named per build in
`diagnostics.sold_after_as_of_shown_current`.

- **Still true, by design:** First Lien (MRI has an origination date on only 4 loans) and the
  deal terms and underwriting figures (coupon, split, lookback, UW IRR, Proj Yr-1) are static
  attributes of the deal and are not cut by quarter.
- **Follow-up, diagnostic only, unassigned:** at an early quarter (25Q4: 10, 26Q1: 6) the
  stale-config check lists config entries for deals NOT YET INVESTED at that quarter. The entries
  are correct and no value is affected; `_check_config_population` should treat a vcode in
  `diagnostics.not_yet_invested` as present. Check: `diagnostics.config_entries_without_a_deal`
  at `as_of=2025-12-31` is empty when this is done.

### 20.3 Freezing is switched off, and 26Q2 has never been frozen — Charlene

`FREEZE_ENABLED` defaults FALSE (`flask_app/config.py:51`) and has never been set on
the container — verified absent (`[]`) before and after `v530`, `v544` and `v548`.
`freeze_gate` makes `freeze_part` raise 503, which covers both batch buttons, the
published-overlay freeze, re-freeze and the Snapshot approval chain. Unfreeze is
deliberately NOT gated.

It was switched off at `v530` after an all-investors batch was run as a SINGLE request
over ~145 investors and the app was unavailable for about 35 minutes. **The freeze
itself was correct** — 145 rows written, cleanly unfrozen, 0 left frozen. The SHAPE of
the request was the problem, and `v530` then posted in slices of 10 while leaving the
flag off; `v539` added the background freeze job.

- **What is still outstanding:** (a) a decision that the background job is what the
  flag was waiting for, and (b) **26Q2 has still never been frozen from the published
  PDF overlay.** Deploying the buttons did not press them.
- **The check:** `az containerapp show ... --query "properties.template.containers[0].env"`
  for `FREEZE_ENABLED`; `portfolio_snapshot_frozen` row count for 2026-Q2.
- **Owner:** Charlene.

### 20.4 The June bank statements have not been loaded on production — accounting

`v508` closed the mechanism (filing a statement opens the chain, the statements are
listed, the PDF is kept) and recorded the load itself as not done: of the 64 files in
`2026\06.2026`, **49 file, 14 are held for an unregistered account and 1 is refused**
(a Wells Fargo statement in the PNC folder). Two of the 14 hold real money — PPI Life
Storage NY 119,701.35 and PSC Ambassadors Fund TGA VI 629,125.04. It is now two steps:
upload the folder, answer the 14.

- **The check:** `GET /api/treasury/statements` row count, and the pending list.
- **Owner:** accounting. Background in `treasury.md`.

### 20.5 `KEEP_DESPITE_SOLD` is a live per-deal hardcode — unassigned

`portfolio_snapshot_service.py:214` names four vcodes (PCITWES, P0000017, PCAMARI,
POUTLOO). The code comment states the problem plainly: *"This is a per-deal exception,
not a rule: 'sold but still reported' is an editorial judgement with no field behind
it. Should MRI ever carry a disposition-type or still-reporting flag, drive it off that
and delete this."*

CLAUDE.md's symptom-repair checklist says a vcode in a constant is **always** a symptom
repair, so this is on the list by that rule alone. It was flagged with Jim at `v459`
and deployed on his call; the `v459` note left it "Unresolved" because the stated
rationale — *"the page is meant to carry every sold deal"* — described 4 of 27 sold
deals, and two DROPPED deals sold later than two kept ones.

- **The precedent that it CAN be derived:** `DEBT_FREE_DEALS` was exactly this shape
  and `v544` replaced it with a rule read off the row's own data. See §12.
- **Owner:** unassigned. Needs the MRI field question put to Alay first.

### 20.7 CLOSED, recorded so it is not re-opened

Three of the four below are shipped and deployed, so they are simply closed. The FIRST
is in flight and carries a date stamp and a deletion instruction, per the standing rule
at the top of CLAUDE.md: a "fixed but not deployed" note reads as authoritative for as
long as it sits here.

- **IN FLIGHT — Investment Metrics shared the committed-pref FUNCTION but not the
  as-of RULE** (`v548`, `v549`). Neither `capitalization_sources` call site passed
  `as_of`, so the report read the CURRENT commitments row while the One Pager read the
  row in force at the quarter — Burton 26Q2 $54.23M here against $26.60M there, with
  the reference printing $26.60M.

  **Status as of Oct 5 2026: FIXED IN A COMMIT, NOT MERGED, NOT DEPLOYED, AND NOT
  RIDING ALONG WITH THE DOCS COMPACTION.**
  - **It lives on `feat/investment-metrics-quarter-dropdown`**, whose tip `a610267` IS
    the commit: "Investment Metrics: committed equity is read AS OF the quarter". It
    adds `committed_as_of()` and makes `_build_row` pass
    `as_of=committed_as_of(ident, as_of)`; both sides of the stack move together, and a
    deal sold on or before the as-of is read at its last held quarter. Two files:
    `investment_metrics.py` and `scripts/investment_metrics_check.py`.
  - **`a610267` pushed to `origin/feat/investment-metrics-quarter-dropdown` on Oct 5
    2026 (was local-only).** Until then it existed in one clone and nowhere else, so
    losing that machine would have lost it. Pushing changes NOTHING else: it is still
    unmerged, still undeployed, and **still needs the measurement and Jim's call
    below.** No pull request has been opened, deliberately.
  - **`docs/compact-claude-md-clean` does NOT contain it.** The first compaction branch
    did, because it was cut from `a610267` and would have carried a runtime change into
    a docs merge; the clean branch is cut from `origin/main` and is docs-only. **Merging
    the compaction does not ship this fix, and must not be read as having done so.**
  - **Not on main**: `git merge-base --is-ancestor a610267 origin/main` fails, and
    `git show origin/main:investment_metrics.py | grep -c committed_as_of` returns **0**.
  - **`fix/investment-metrics-committed-as-of` WAS A DECOY AND IS DELETED** (local,
    Oct 5 2026). It sounded like this work and was not: its tip was `06061b0`, a stale
    copy of main, an ancestor of `origin/main` with no commit of its own and **zero**
    occurrences of `committed_as_of`. It never existed on the remote, so nothing was
    deleted there. Recorded in case anyone recreates it from an old clone.
  - **IT NEEDS ITS OWN FIGURE MEASUREMENT BEFORE IT MERGES.** The commit message claims
    8 values move on live and 0 get worse; that was measured by its author and has not
    been re-measured since. It changes a REPORTED figure, so CLAUDE.md's standing rule
    applies: measure the affected deals against live data, report the count and the
    deltas, and get Jim's call BEFORE building. Do not fold it into a docs merge.

  **Delete this bullet when it ships**, and record the revision in
  `deploy_history.md` instead.
- **`nReqDSR` 1.10 vs `nLTV` 0.55 — which is the extension test** (`v481`, "still
  open"). Settled by Jim Sep 17 2026 and shipped at `v482`: `nReqDSR` is the EXTENSION
  test, `nRequiredDCR` the ongoing covenant. Deployed; nothing in flight.
- **The two portfolio summary SCREENS** (`v502`, "still to build"). Shipped at `v503`
  (`f151e5a` — confirmed present in git, Oct 5 2026). Deployed; nothing in flight.
- **No rent step carries `period_start_month`** (`v511`, "still open"). The `v512`
  re-extraction took it 0 -> 305, with 208 steps dated from the term. Deployed;
  nothing in flight.

### 20.8 Two memory files describe the same rules twice — LOW PRIORITY, unassigned

`treasury.md` and `accounting_workpapers.md` each now carry **two** descriptions of the
same rules: their original long-form section, and a "rule digest that used to live in
CLAUDE.md" appended beneath it on Oct 5 2026. The digest was appended rather than merged
line-by-line because merging by hand risked dropping a rule, and losslessness was the
higher priority that day.

Nothing is wrong today — both descriptions agree, because one was written FROM the
other. The hazard is drift: a rule corrected in one copy and not the other leaves two
confident, contradictory statements with nothing saying which is current, which is the
same failure ONE NUMBER ONE ENGINE exists to prevent, one level up.

- **The work:** reconcile to one description per rule in each file, keeping the fuller
  wording where they differ, and leave a line saying the digest was folded in.
- **The check:** no rule stated twice in either file; `CLAUDE.md`'s pointer still
  resolves; the line-presence check still passes against the pre-compaction original.
- **Why it is low priority:** it costs a reader some repetition, not a wrong answer.
- **Owner:** unassigned.

### 20.9 Delete `claude_md_prose_archive.md` once the compaction merges — unassigned

`.claude/memory/claude_md_prose_archive.md` holds CLAUDE.md's pre-compaction prose for
the sections that were rewritten rather than moved. It exists so "the compaction lost
nothing" is checkable without going to git, which matters while the branch is in review
and not after: once `docs/compact-claude-md` is on main, git history holds the same text
and a second copy is just another thing to keep in step.

**DONE Oct 5 2026.** The compaction merged to main as `3e55316`, and the archive was
verified against history before being deleted: all four extracted ranges of
`a610267:CLAUDE.md` (the parent of `bb91c1d`, and the last commit holding the
pre-compaction CLAUDE.md) are present in it verbatim. `pr_docs_compaction.md` went
with it, both MEMORY.md index lines and CLAUDE.md's pointer row with them. The
pre-compaction text remains reachable at **`a610267`**.

### 20.10 Two contradicting comments about Loan subtotals in one file — docs-only, unassigned

`flask_app/services/portfolio_snapshot_loan.py:313` says of the manual ratio seeds:

> the raw computed `ltv` / `ytd_dscr` / `debt_yield` are LEFT ALONE, so the fund
> subtotals, the portfolio total, every guardrail and any frozen payload keep
> reading the computed truth. **A typed cell contributes to no aggregate**

`aggregation_value` at `:595` — the code that actually runs — says the opposite,
and says why:

> a subtotal aggregates the figure the row DISPLAYS — the typed entry where there
> is one, the computed figure otherwise — so a fund total can always be
> re-derived from the rows printed above it.

**`:595` is right and `:313` is stale.** The behaviour changed deliberately on
2026-09-02, because weighting the computed figure alone left the TGA25 and TGA6 LTV
totals blank while their members displayed typed LTVs, and left TGA6's DSCR reading
3.81x above a Presidential Arms row printing 1.1x. The comment at `:313` was not
updated with it.

**IT HAS ALREADY MISLED ONCE.** A hardcode inventory of the Loan subtab on Oct 5
2026 quoted `:313` as authoritative and stated that the seeds are excluded from
subtotals. They are not. `v569` then moved four subtotals by removing one seed —
PSC1 1.5946575 -> 1.5954366, OWPSC total 1.6237423 -> 1.6243495, TGA25 1.8514554 ->
1.8578197, TGAM total 1.7578363 -> 1.7586545 — which the inventory had said could
not happen. The figures were right; the explanation given for them was not.

- **The work:** correct `:313` to match `:595`, and say that a typed cell DOES enter
  the aggregate in the displayed unit (percentages divided by 100, DSCR passed
  through), with the three skip cases `aggregation_value` lists.
- **Do NOT "fix" it by changing the behaviour** — `:595` is the deliberate rule and
  the one the fund totals are auditable against.
- **Docs-only.** Deliberately not bundled into `v569`, which was a behaviour change.
- **Owner:** unassigned.

### 20.11 A standing PostgreSQL firewall rule exists for Charlene's machine — CLOSED Oct 5 2026: Jim confirmed it stands

`local-dev-cbui` → `76.99.107.120/32` was added to `psql-waterfall-dev` on **Oct 5
2026** and **deliberately left in place**, matching the per-machine convention already
on the server (`local-dev`, `local-dev-2`, `local-dev-3`, `local-dev-4`,
`local-dev-current` — six rules before this one). Jim's own guide,
`docs/Refresh_Local_Database_Instructions.md`, names adding the IP as the normal
remedy for the connection timeout.

**Why it was needed:** `scripts/pull_production_db.py` connects directly to
PostgreSQL, and the Q3 readiness investigation is impractical any other way.
`az containerapp exec` is the alternative and it does not scale for this: its
`--command` payload 404s above roughly 1.5 KB, and it rate-limits to **HTTP 429 with
`retry-after: 600`** after a handful of connections. Six execs were enough to trigger
a ten-minute lockout.

- **It is a single /32**, not a range, and grants nothing beyond what an `az login`
  with secret-read already allows.
- **Remove it** with `az postgres flexible-server firewall-rule delete -g
  rg-waterfall-dev -s psql-waterfall-dev -n local-dev-cbui --yes` if the convention
  changes, or when the machine does.
- **Note `--rule-name` is not valid on this CLI version** — the flag is `-n`.
- **Decided Oct 5 2026:** Jim gave Charlene the OK -- the rule stays. Remove it only
  if the convention or her machine changes (command above).

---

## 18. Production runs pandas 3; every guardrail runs on pandas 2 (Oct 2 2026)

`requirements.txt` has `pandas>=2.3`. Pandas 3.0.0 shipped Jan 21 2026, so every
production image since has resolved pandas 3 (3.0.6 at `v553`-`v555`), while the local
`.venv` -- where every guardrail runs -- is pandas 2.3.3. Found when `intercompany_check`
failed inside the container: pandas 3 reads a SQL NULL as NaN, and the intercompany
reconciliation had an entity called NAN (fixed at `v555`).

- **18.1 DONE (Oct 2):** all 134 guardrails run one at a time under pandas 2.3.3 and
  3.0.6 (scratch venv with the container's resolution): **0 differ.** 50 fail under
  BOTH -- 16 import the absent `live_api` harness, the rest need real local data or
  files (§15) -- so they say nothing about pandas 3 either way. The first, parallel
  run was confounded by checks sharing fixed temp-db names; run them sequentially.
- **18.2 DONE (Oct 2, `v557`):** `pandas>=2.3,<3.1`, resolving 3.0.6.
- **18.3 DONE (Oct 2):** the local `.venv` matches production -- pandas 3.0.6, numpy
  2.5.3, SQLAlchemy 2.0.54 -- and the leftover `streamlit` (unused, not in requirements)
  and the `pyarrow` it pulled in are REMOVED: with pyarrow present pandas 3 stores strings
  differently from production, which has none. A SQL NULL now reads as `nan` locally, as
  in the container. All 135 guardrails re-run one at a time: identical to the scratch
  pandas-3 run (85 pass; the 50 are §15's environmental ones). Previous package list:
  the session scratchpad's `venv_before_pandas3.txt`.

---

## 17. Employee expense reports (Oct 2 2026) — LIVE at `v553`

Design, phases and measurements: `expense_reporting.md`.

- **17.1 The intercompany ownership rule — accounting's call.** Right for Fairview,
  Nottingham and Woodlands (and Ascent's split); wrong for Pontchartrain, Gallery and
  Belleville because the investee funds INVF7 / INVF2 / INVF11 keep their own
  `MR15000002` while accounting books their parent PSC3. Do investee funds always pass
  through to PSC3? Needs a rule from accounting before the walk changes. Owner:
  **Jim / accounting**.
- **17.2 Apple and Brainerd shares are data questions.** Commitments carry PSS1 17.93%
  into APPLE beside PPI2 (accounting books PPI2 100%); Brainerd's commitment dollars give
  TGA22 82.68 / PSC1 17.32 against accounting's ~62.4 / 37.6. Owner: **accounting**.
- **17.3 Accounting sets PPI2 to CAD** (Expense Coding > Currencies & recurring). Nothing
  seeds it. Owner: **accounting**.
- **17.4 The admin unticks Expenses** for anyone who should not see it, and sets every
  employee's name on reports and approver -- since `v564` only the `admin` LOGIN can set
  approvers. Owner: **admin**.
- **17.7 Untick Accounting for Charlene (`cbui`)** -- and `anaik` if appropriate. `v564`
  makes accounting's expense rights need the Accounting SECTION, not just the admin
  role, but nothing changes for her until the box is unticked. Owner: **Jim**.
- **17.8 Treasury descriptions on screen (`fix/treasury-mri-text`, `8ac9fe9`, pushed,
  not merged).** Built, not verified in the app, not deployed; wire descriptions need a
  payee-first default (Jim's call). See the handoff. Owner: **Claude, then Jim**.
- **17.6 DECIDED (Jim, Oct 2 2026): Expense Coding's Void stays as it is.** It does not
  check whether MRI already posted the batch (intercompany's void does). So it is safe
  only for a file that was NEVER uploaded: voiding an uploaded batch releases its
  reports to be batched and paid again. The batch list shows "posted" once the GL
  carries the payroll credit; an uploaded batch is corrected in MRI with a reversing
  entry, not voided. Told to the CFO via Jim. Do not change without Jim.
- **17.5 No mileage rate is set** — mileage lines are refused until accounting sets one.
  Owner: **accounting**.
- Until 17.1-17.2 are settled, accounting should check every intercompany row; every
  proposal is editable and a hand-set split is kept.

---

## 16. Section access by username (Oct 1 2026) — LIVE at `v552`

Built and verified locally (`scripts/section_access_check.py` 252/0, proved
non-vacuous against eleven injected defects; accounting_access 54/0,
gl_ia_query 123/0, treasury_api 46/0 unchanged). See CLAUDE.md "Section access
by username". Open:

- **16.1 DONE (Oct 1):** `tr_*`, `wp_*`, `ic_*` restricted to Accounting by
  prefix. Only the `admin` username assigns access; Asset Management and New
  Business are linked "for now" -- revisit when they should separate.
- **16.2 Settled (Jim, Oct 1):** review roles in Settings stay behind Asset
  Management; shared screens (`/api/deals`, `/api/argus`) stay shared.
- **16.3 Feedback & Requests is not a section** — every user can report a
  problem. It sits below the nav, not in it.
- **16.4 Deploy note:** creates `user_section_access` on first gated request
  (once per process). No backfill; with no rows every user keeps every
  section, so the deploy itself changes nobody's access.
- **16.5 DONE (Oct 2):** deployed as `v552` = `11c3455` after origin/main was merged in.
- **16.6 DONE (Oct 2):** a user named `admin` exists on production (verified in-process:
  `can_assign` true for it, 403 for the admin-role users `anaik` and `cbui`).

---

## 15. Guardrails that are RED for a known reason (Sep 29 2026)

A check that has been red for a while stops being read, and then the day it goes
red for a NEW reason nobody notices. Each one here is proved pre-existing —
`git stash` the working tree and it fails identically on the unmodified code —
and carries an owner and a fix, so "known red" never becomes "ignored".

### 15.1 `onepager_missing_vs_zero_check` — 15 of 16, unassigned

`scripts/onepager_missing_vs_zero_check.py` fails on **"the Economic Occ. row
routes through `fmtOccVariance`, not a raw `toFixed`"**.

**PROVED PRE-EXISTING** on Sep 29 2026: stashing the `feat/freeze-per-quarter`
working tree and re-running gives the same `15/16 checks passed` on clean
`origin/main` (`27e1dd8`). It is not related to the freeze or quarter work; it
was simply found while sweeping the suites around them.

The rule it defends is real — the One Pager must not print a raw `toFixed` for a
variance row, because a missing value and a zero then render identically, which
is the whole point of that check file. Either the Economic Occ. row genuinely
stopped using the shared formatter (a defect to fix in the view) or the check is
matching on a call shape the view has since changed (a defect to fix in the
check). **Nobody has yet established which**, and that is the work.

**Do not "fix" it by loosening the assertion** — same standing rule as §11.1.

See also §11.1, `budget_import_check`'s one pre-existing failure, and the
container-only failure of `katex_render_check` recorded in the `v527` deploy
note (it reads `vue_app/package.json`, which the runtime image does not ship).

---

## 16. Rent roll objectives and the IC exhibit (Sep 29 2026)

New business's 31-section rent-roll specification and their Market at Poplar exhibit
(B2:H111), measured against the app. **Full detail, the exhibit's exact formatting
and the five-step plan: `rent_roll_exhibit.md`.**

Headline, 27 exhibit tenants matched: SF 26, current rent 26, expiration 21-22,
Start 12, future step dates 21, option count 19, **option rent 0 of 39**. The causes:
exercised options never applied to the term; later documents overwrite the lease
start; 115 duplicated option rows; option rent stored only as text; no continuous
timeline; no analyst settlement of steps or options.

### 16.1 Plan -- step 1 BUILT (the three re-read failures), steps 2-5 open
Owner: build, in order. Step 2 governing terms; 3 one timeline engine; 4 analyst
settlement of timeline rows; 5 the exhibit, accepted by reproducing theirs cell by cell.

### 16.2 Also found
- 7 exhibit tenants did not match an app tenant by name -- reconcile before step 5.
- The re-read moved BooYa's and Outback's lease start to a later document's date
  (see the Start row above) -- step 2 fixes the cause; until then the Start field
  on those two is wrong.

---

## 15. Co-tenancy and exclusives -- the analysts' corrections (Sep 29 2026)

New business corrected Market at Poplar (review 3) by hand in a workbook and a
correction log. Measured against production first: **the app already had most of
it right.** Firehouse's pizza/hamburger/dairy/shoe-repair rows were marked "Bound
by" with the real holder named; Hobby Lobby's 200-ft radius was there; the
Starbucks carve-out names CiCi's because Exhibit H does. What was wrong was what
the export and screens HID: the Exclusive Use sheet carried only restriction text,
so every "bound by" row read as the tenant's own -- the likely source of the
Firehouse error.

Built (not yet deployed): a re-read replaces a document's clause rows instead of
adding copies (`_write_document_clause_rows`); `rebuild_clause_rows` clears the
existing duplicates from stored extractions with no API calls; the export and the
Exclusive Use tab show Holds/Bound by, radius, carve-outs, source; a per-tenant
review (`lease_clause_reviews`: unreviewed/confirmed/flagged + note, marked when
re-read); a failed reading is recorded as `error` WITH its reason
(`extraction_error`) instead of being logged "Extracted"; one retry when no JSON
comes back; output room 16K -> 64K tokens and the text cap 180K chars -> 2M.
Guardrails `lease_clause_rows_check.py` (35), `lease_scan_extraction_check.py` 25 -> 33.

### 15.1 AFTER DEPLOY -- both steps RUN at `v532` (Sep 29 2026)
Result: rebuild 291 -> 92 rows (idempotent); GNC 8 of 9 read, the 1996 lease
still failed at `v532`. **RESOLVED at `v533`**: the retry now sends a scan's
pages as rendered images, and the 1996 lease read that way -- the
vitamins/supplements exclusive is in the app, all 9 GNC documents read.

Original steps:
1. `POST /api/lease-review/reviews/3/clause-rows/rebuild` (admin) -- measured
   before: 291 exclusive rows for 35 tenant/document pairs, up to 25 from one lease.
2. Re-read GNC (tenant 159). 8 of its 9 documents were read on Sep 29; the
   **1996 original lease** (41-page scan) came back with 30 tokens -- the first
   sentence of the prompt -- and a normal stop, so no vitamins/supplements
   exclusive is in the app yet. The new retry is the remedy; confirm it lands.

### 15.2 Still open
- **Conflict flags** (one tenant's exclusive vs another's use) exist only in the
  analysts' workbook. A feature, not a fix -- not built.
- **Co-tenancy is empty for review 3**: 0 of 138 documents report a clause.
  Plausible; the analysts should confirm on the Co-Tenancy review list.
- The lease export raises if a review has no GLA/rent totals yet
  (`generate_lease_review_excel`, the `review[3]:,.0f` format). Latent.
- Renewal options still dedupe on `(source_doc, option_number)` and never
  replace -- the same stale-row shape co-tenancy had.

---

## 14. MRI commitments as the ownership source — checked Sep 29 2026

Jim updated `IA_Commitment` in MRI to serve as the source of capital ownership
percentages and asked whether every chain now reaches the beneficial owners at
100%. **Not yet.** Measured on the 556 current rows (`EndDate IS NULL`), read
live from MRI and identical to the app's copy (refresh finished Sep 28 14:56).
The entity-by-entity list went to Jim as
`Downloads\Commitments_Ownership_Check_2026-09-29.xlsx` (7 tabs), for whoever
maintains MRI.

**The app never reads `CapitalPercent`.** `ownership_chain_service` and the
treasury investor split derive the share from committed DOLLARS (see that
module's docstring for why: PPIECH/EASTCH read 0.00 on both rows while the
amounts were right). So what can break the app is the POPULATION and the
AMOUNTS, not blank percentages. Keep the two apart when this is re-checked.

### 14.1 Two missing links — MRI (Jim / whoever maintains commitments)
Walking up from each of the 77 deal investments that have commitments, every
chain ends at OWPSC (its six owners sum to 100.00%) or at one of 107 outside
investor IDs — except two entities with owners in `relationships` and NONE in
commitments:

| Entity | Invests into | Owner per relationships | Deals affected |
|---|---|---|---|
| `INV24-P` | TGA24 | AMB24 | 45MAIN, BELAIR, DORS, GLENM, GVRTEL, TFTP |
| `PSS1` | APPLE, PPI2LP | PSC2 | APPLE, BALES |

### 14.2 Amounts to confirm — MRI
- **PPI11, PPI17, PPI18**: one owner each (PSCKOC) stored at 50%. A second owner
  missing, or should be 100%? Dollars say 100%.
- **24 rows carry $0 committed**, so they derive 0% — mostly PSC3's partners
  (DBH, FNKI, JJB, EAJI), PSCMAN/PSC1 rows, KCREIT. **INVPLY <- PIG3** has a $0
  row at 100% and a $5,784,031 row at 0%.

### 14.3 `CapitalPercent` itself — only if MRI's column is to be the source
157 of 228 entities total 100%; **71 do not** — 67 are all 0.00 (most property
entities, plus PSC1 in 66 deals' chains, PCBLE 23, PSCMAN 22), and four total
the wrong figure (PPI11/17/18 at 50, PIG6 at 34.04 with PSCKOC's $6.2M at 0%).
Eleven JV deals store 100% for the PPI vehicle and 0% for the OP partner, which
is not the dollar split (e.g. WINDSO: PPIWIN 70% / OPWATER 30% by dollars).
Only **4 of 77 deals** carry a correct stored percentage at every level.

### 14.4 Deals with no current commitments — 32, mostly expected
Child properties held through a parent (BRN-1..9, BURT-1..3, TFT-1..6, PMATNO)
plus 13 others — ASTONC, AYRTOW, BEARUN, CREEK1, HERITA, LINDEN, PARKPL, SCOTTO,
SPRNGM, STONEC, STONES, WHITMA, WSTCHA. **Presumed sold, NOT verified** — the
query drops ended commitments.

**Re-check**: rerun the extraction (live `run_query("MRI_Commitments",
save_csv=False)` plus `relationships`, `entities`, `deals` via
`az containerapp exec`), then walk up from each `deals.InvestmentID`, stopping
at OWPSC.

---

## 12. Loan subtotals carry Pegasus's debt while its row prints a dash (Sep 25 2026)

**RESOLVED at `v544` (`1da00ca`, Sep 30 2026). Closed Oct 5 2026; the original
statement is kept below because the shape of the bug is worth recognising again.**

`DEBT_FREE_DEALS` is **removed, not emptied** — three guardrails assert the attribute
cannot come back (`debt_free_rule_check`, `freeze_as_sent_check`, `q3_cleanup_check`).
The N/A display is now derived from the row's own data (`_debt_free` in
`portfolio_snapshot_loan.py`), and the footing hole is closed **by construction, twice
over**:

- `debt` and `debt_display` are BOTH bound through one `debt_field()`, which returns
  `None` when the deal is held debt free — so the figure the subtotal sums and the
  figure the cell prints are decided once and cannot drift apart. `loan_subtotal()`
  cannot pick up a number the page does not show.
- The derived rule requires `debt == 0.0` exactly (a measured zero on the ISBS basis;
  `None` is "no reading" and is declined). So the combination the bug needed — a
  non-zero raw `debt` sitting behind a blanked display — is no longer reachable.

The population was measured read-only on production at 26Q1/26Q2/26Q3 before it
shipped: the rule fires on `P0000066` and nothing else, and `CHANGED vs old vcode list
= NONE` at all three. **The only thing that moved was the SOURCE of Pegasus's dash.**

*Original statement (Sep 25 2026), retained:*

`DEBT_FREE_DEALS = {"P0000066"}` (Pegasus Life Storage,
`portfolio_snapshot_loan.py:192`) blanks `debt_display` so the row prints an em
dash, but leaves the raw `debt` alone. `loan_subtotal()` sums the RAW `debt` and
excludes only `sold_suppressed` rows — debt-free is not an exclusion — so
**$25.2M sits inside Portfolio Totals, the fund subtotal and the
excluding-development row with no row on the page to account for it.**

The in-code justification is itself the defect: *"the raw `debt` stays 0.0 and
still feeds the subtotals, where it contributes nothing either way"* — true only
while the ISBS balance is zero, and it is not.

The Financial subtab has no equivalent hole: `PDF_NA_CELLS["P0000066"] =
{"debt"}` makes `debt_summable` None, so the figure leaves that total. **So
Financial equals the rows it displays and Loan is the total that does not foot
to its own page.**

- **The check:** compare the Loan subtab's Portfolio Totals debt against the sum
  of its displayed rows for any quarter where Pegasus carries a balance.
- **Why it is not fixed here:** it is a live-engine defect, not a freeze defect.
  Fixing it on `feat/freeze-as-sent` would put two unrelated concerns in one
  review.
- **Why 26Q2 is safe:** the frozen page stores the PDF's own printed subtotal,
  so the sent quarter keeps the figure that was sent however the live engine
  changes afterwards. Proved by `scripts/freeze_as_sent_check.py` section Q,
  which removes all five Loan-tab hardcodes and asserts the frozen payload is
  byte-identical.
- **Owner:** unassigned. Earlier measurement (Sep 16 2026) in
  `loan-financial-debt-footing-wip`.

## 1. Code gaps — verified present in the tree

### 1.1 Account 7076 (Tenant Improvements) is completely unmodeled — LARGEST ITEM
**Verified Sep 11:** `grep 7076 config.py` → no match. It is in no account set, and
`reporting.py` aggregates by explicit set membership with no "everything else" bucket, so
TI never reduces NOI, FAD, distributable cash, or the waterfall.

**Scale:** 37 deals carry 7076 in Valuation IS. Camp Creek (`P0000075`) ignores
**$4,024,004** (2026–2036) against modeled CapEx 7050 of only $2,902,452 — the unmodeled
line is **1.4× the modeled one**, 4.6% of cumulative NOI, peaking at **10% of NOI in 2035**
($902,011) as rollover hits. Others: Woodlands Square $3.56M · Poplar Prairie $2.70M ·
Donald Lynch $2.69M · Merle Hay $2.68M · Deptford $2.60M · Evergreen $2.51M.

**Blocked on a question first:** the two sources disagree by 2× — Valuation IS says Camp
Creek $8.20M, `forecast_feed` (priority 1, wins) says $4.02M. Resolve that before acting.
Also confirm whether 7075 (Reserves for Replacement) is the intended funding source, or a
lender TI/LC holdback pays it — either would make a naive inclusion a double-count.

**Recommended shape if included:** add to `CAPEX_ACCTS` in `config.py`, not a flat
deduction. Reserve funding (`capex_paid`, `cash_management.py`) and sign normalisation
(`normalize_forecast_signs` forces `-base.abs()` for `ALL_EXCLUDED`) then come free. Almost
no actuals exist (1 Interim IS row portfolio-wide), so this touches projections only —
history and realised returns are unaffected.

**Owner:** Jim (source question) → then a code change.

### 1.2 `ROE_Income` is set in SQL and read by no Python
**Verified Sep 11:** `grep -rn ROE_Income --include=*.py` → zero hits.
`queries/accounting_feed.sql:43` sets the flag, marking TypeID 1019 (Preferred Return) and
1020 (Excess Cash flow). The code re-derives the same concept from `Typename` and
**disagrees by $23,599,656** (code $169,654,101 vs flag $146,054,445, deal scope). The code
is a strict superset; the extra is `Distribution: Income` (156 rows, +$27.2M),
`Distribution: Tax`, Professional Fee Holdback and Non Resident Withholding.

A flag exists, the code ignores it, and nobody has chosen which is the house definition.
See §2.2.

### 1.3 Four different capital-event classifiers on the same rows
**Verified Sep 11** (all four paths still distinct):

| Path | Driver | Realized Gain treated as |
|---|---|---|
| One Pager ROE, ROE Summary | `Typename` string | capital event |
| Deal Analysis ROE Audit, partner ROE | `Capital` flag (`loaders.py`) | **operating income** |
| Sold Portfolio | `Typename` (deliberate — CLAUDE.md notes the flag is unreliable at sale) | capital event |
| Pref Balance Detail | `TypeID` 1019/1020 | narrowest |

**$73,574,410 of Realized Gain (41 rows, deal scope)** is a capital return on one tab and
operating income on another. One deal legitimately shows two ROEs: 30 Bearfoot, Deal
Analysis 24.62% vs One Pager 22.02%. See §2.3.

### 1.4 The quarter dropdown is portfolio-wide, and can have holes
**Verified Sep 11:** `one_pager.py:87` — `get_available_quarters(isbs_df)` takes no vcode
and applies no per-deal filter. One deal's newly loaded quarter therefore appears on
**every** deal's dropdown, and the list can skip quarters (observed offering Q3 but not Q2,
because no deal had Q2 actuals). Confirmed behaviour, not theory.

This is the sibling of the first-load default bug that WAS fixed (`54b4700`); that fix made
the label and the data agree, it did not make the list per-deal.

### 1.5 Berger Pittsburgh silently drops six of eight child loans
**Verified Sep 11:** `_loans_share_terms()` exists in `one_pager.py` and correctly guards
the co-terminous case (Burton). Berger's 8 child loans carry **6 distinct term sets**
(senior ~2.9% plus mezz ~7.3% per property), so it fails that guard — correctly — and falls
back to the primary/second selection rule written for a single property with a real capital
stack. It renders a primary and a second and **drops the other six without saying so**.

A parent with genuinely differing child loans has no display that tells the truth. Needs a
decision on what it should show before it can be coded.

### 1.6 U/W ROE pro-rate branch keeps one month of a multi-month YTD
**Still present** at the `one_pager.py` pro-rate branch. Swept across every 7071 deal:
49 vcodes have data → 40 computable, **32 move if the branch is removed**, restoring
**$2,443,476.63** of numerator ($2,027,032.18 excluding the bad Westbank hit). The
discriminating test (period cumulative vs the next month's delta) says **31 of 32 are a
single month**, i.e. the pro-rate is wrong on them.

**Do not simply remove it.** Centre at Westbank (`P0000010`) is the one deal where removal
makes things worse — its 2022-03-31 cumulative is 10.72× run rate, a stale prior-year value
at the wrong date, and removing the branch over-counts by ~$416,444. Ship removal **plus a
guard**: if a first-period cumulative exceeds ~2× the next month's delta, log it and fall
back. Alternative is fixing the extract — see §3.3.

Untestable: Asbury Commons (`P0000004`), one row in its entire 7071 series; impact $606.66.

### 1.7 `"contrib" in MajorType` is over-broad
**Verified Sep 11:** live at `compute.py:298`, `reports_service.py:355, 451, 592, 1101, 1145`.
The match sweeps all eight contribution TypeIDs, so **Partnership Expenses (400 rows),
Management Fees (34), Organizational Costs (2)** land in `funded_to_date` and in the ROE
denominator alongside real capital. Same over-broad match on the ROE and capitalisation
paths. See §2.5 — whether that is wrong is a definition question, not a bug, until someone
says so.

### 1.8 Pegasus: a JV entity is bucketed into Pref equity
**Verified Sep 11:** `one_pager.py:894, 926` (and `:2533, :2547`) bucket on
`InvestorID.startswith("OP")` — everything else is Pref. Pegasus has three investors:
`OPPEGA`, `PPILFS` and **`TGA22`**. Only `OPPEGA` starts with OP, so **TGA22 — the PSCKOC JV
entity — is counted as Pref equity.** Does not affect the (now-fixed) reversal numbers, but
it does change how Pegasus's cap stack splits pref vs partner. See §2.6.

---

### 1.9 A reset password is the literal string `password`, emailed in plaintext
**Verified Sep 11:** `flask_app/auth/routes.py:358` — `temp_pw = "password"`, hardcoded.

There is **no admin "set a password" endpoint**. The only admin path to reset someone is
`POST /auth/users/<id>/send-welcome` (the "Send Welcome" button in Settings → Users), which
sets every reset user to that same literal, flags `must_change_password`, and **emails the
password in plaintext**.

Why it matters: the value is identical for every user and every reset, so it is guessable
by anyone who has ever been onboarded. `must_change_password` narrows the window to the
user's next login — it does not close it, and the email persists in a mailbox indefinitely.

Recommended: generate a random temporary password per reset, and prefer the existing
`/auth/forgot-password` flow (a one-hour single-use token, no password in the email) as the
default path for an existing user. `send-welcome` then only matters for genuine onboarding.

The trap: `change_password(user_id, temp_pw, clear_must_change=False)` and the
`must_change_password` UPDATE are two separate statements — if a random password is
generated, make sure a failure between them cannot leave an account on an unknown password.

**This is no longer hypothetical, and §3.12 makes it worse.** On Sep 15 2026 a new user
(`jstewart`) was onboarded while SendGrid was dead. The reset ran — it happens *before* the
send is attempted — so the account sat on the literal `password` from 09:26 until Jim
delivered the credentials by hand through Outlook and the user logged in. That is the
designed behaviour and the account was usable throughout, which is the point of the `v451`
wording. But note the interaction: **while email is down, every onboarding leaves an account
on the guessable literal for as long as it takes a human to make contact**, and the
credentials then travel through ordinary mail rather than the app's own channel. Another
reason to finish §3.12 rather than keep hand-delivering.

### 1.10 A leaked JWT cannot be revoked — it is live for up to 24 hours
**Verified Sep 11:** `JWT_EXPIRATION_HOURS = 24` (`flask_app/config.py:15`), HS256 signed
with `JWT_SECRET`. A repo-wide grep for `revoke|blocklist|blacklist|token_version` returns
**nothing**.

Tokens are self-contained: nothing re-checks the password on a request. So **changing a
user's password does NOT invalidate their existing token**, and there is no per-user way to
kill one. The only lever is rotating `JWT_SECRET`, which signs everyone out at once.

Why it matters: any leaked token — pasted in a chat, captured in a log, copied from
DevTools — is valid for up to 24 hours with no way to intervene. This is exactly what made
the Aug 6 `cbui` incident (§4) unanswerable at the time: nothing could be done, and nothing
recorded that.

Recommended: a `token_version` integer on `users`, included in the JWT payload and compared
on decode. Bumping it invalidates that user's tokens only, and a password change can bump
it automatically. Cheap, and it turns "wait it out" into an action.

The trap: `/auth/me` and every `@login_required` route decode on each request, so the
comparison needs the user row — check the cost before adding a query per request.

### 1.11 An approved valuation gives no sign that it is still unpublished
**Verified Sep 11 2026.** `committee_approve` sets `status='approved'` and freezes a
snapshot. It does NOT write to `valuations` — that is `publish_record`, a separate step.
Nothing in the UI says so. The status reads "approved", which sounds finished, and the
figure silently never reaches the One Pager.

Why it matters: valuations are annual and low-volume, so a record can sit
approved-but-unpublished indefinitely with no signal. This cost a full afternoon on Sep 11
— the record read `approved`, the One Pager kept showing the prior year, and nothing
connected the two.

Recommended: surface "approved, not published" on the cycle dashboard, and mark it on the
record. Publishing automatically on final approval is the other option, but the separation
looks deliberate — showing the gap is the safer change.

The trap: `published_at` is the field that actually answers this, not `status`.

### 1.12 `publish_record` can publish a NULL valuation
**Verified Sep 11 2026.** The guard is `nav = get_nav(...)` then `if not nav: raise`. `nav`
is a **dict**, which is truthy even when `nav["value"]` is `None` — so the insert writes
`"val": nav.get("value")` as NULL and reports success.

Why it matters: a published row with no concluded value is indistinguishable from a
successful publish, and every consumer then falls back to the previous year (correctly —
see §4, the blank-column fix). The publish "worked" and nothing changed.

Recommended: refuse the publish when `nav.get("value")` is not a positive number, with the
same message shape as the existing "Compute the NAV before publishing".

The trap: `0` must be refused as well as `None` — a valuation of zero is not a
measurement, which is the rule the rest of this file already applies.

### 1.13 `valuation_records` has no structured cost basis
**Verified Sep 11 2026.** The table holds `concluded_value` and a free-text
`override_note`, and nothing else about how a cost-basis figure was built.

Why it matters: Town Fair's 12/31/2025 value is 33,910,000 against an `Acquisition_Price`
of 30,750,000 — a 10.3% gap that looks like a market write-up and is not one. "Cost" here
means purchase + capital prefunded for improvements + closing costs + accrued pref through
the reporting date. **Anyone reconciling the two hits that wall**, and the only thing
standing between them and the wrong conclusion is whatever a human typed in the note. This
file's author reached the wrong conclusion on exactly this and had to be corrected.

Recommended: structured components on the record (purchase, improvements, closing costs,
accrued pref), so a cost basis explains itself and the total is checkable.

The trap: the free-text note is still worth keeping — the components will not cover every
case.

### 1.14 Pref Equity capitalization may truncate in print — UNCONFIRMED
9 deals reportedly lose the tail of their ownership split in print: P0000006 prints "KOC
43%, PSC 41%, Declaration" and drops "16%"; P0000081 drops "F&F 12%". Reported in
`9eff832` as pre-existing and present in the 180px baseline.

**Do not change print CSS until this is measured.** Two reasons to doubt it:

1. `onepager_print_geometry.py:146` `_covers()` documents this exact case as a READING
   ORDER artifact, naming "16%" specifically: *"a value that re-wraps onto its own line
   moves in pdfplumber's reading ORDER … '16%' reads back as '%16' … an IDENTICAL
   character count is what identifies it as reordering rather than loss."*
2. Tracing the chain found no mechanism that would clip it: the textarea is genuinely
   hidden (`.print-hide { display: none !important }`), the print twin has
   `white-space: pre-wrap; overflow: visible; height: auto`, and the cell has no
   `overflow: hidden`, no fixed row height and no `nowrap`. A table row grows to its
   tallest cell.

Settle it by printing P0000006 and looking at the cell, or by running the sweep (needs
`WF_TOKEN`). If real, the likely fix is `table-layout: fixed` on `.cap-table` in print so
the declared 18% is enforced and wrapping is predictable — with the sweep to confirm no
other column regresses.

*Jim's interim call (Sep 11 2026): the asset manager will abbreviate "Declaration" so the
text fits. That removes the symptom on one deal; it does not answer whether the defect is
real.*

## 2. Decisions needed — blocked on a human, not on code

### 2.1 What metric is U/W ROE meant to be? — ANSWER THIS FIRST
The source U/W workbook computes **`H18 = H15/H7`**: single-period distributable cash flow
over that period's equity balance. **Not time-weighted, not inception-to-date, not
annualised.** `metrics.calculate_roe()` is ITD dollar-day weighted and annualised.

They agree on The Gathering only because its underwritten steady state is flat
(−31,551.35/mo). **On any deal with a lumpy 7071 schedule the two definitions diverge even
with a perfectly correct denominator.** Most of the remaining U/W ROE work is downstream of
this one answer — decide the target before writing code against it.

*Owner: Charlene.*

### 2.2 `ROE_Income` flag or the Typename rule? ($23.6M apart)
Is `Distribution: Income` (156 rows, $27.2M) operating income for ROE purposes? See §1.2.
*Owner: Charlene.*

### 2.3 Should Deal Analysis move off the `Capital` flag onto `Typename`? ($73.6M)
Would make it agree with the One Pager and Sold Portfolio. See §1.3. *Owner: Charlene.*

### 2.4 Brainerd and OREI equity basis
Both are zero-delta on every mechanism tested — not the sign bug, not a date filter, not
child properties. Reduced to a definition question:
- **Brainerd**: does the model include `Contribution: Others` $4,550,000? (Azure shows
  12,007,677; excluding gives 7,457,677.)
- **OREI**: does `Contribution: Operating Capital` $1,233,899.26 count as partner equity?
  (Azure pref 13,391,868 / partner 8,124,512; excluding gives 10,786,868 / 6,890,613.)
  Note: OREI has **no** `Contribution: Others` rows at all — that Typename is Brainerd's.

*Owner: Charlene.*

### 2.5 Development-deal basis, and the over-broad `contrib` match
Do development deals report **funded-to-date** or **closing capitalisation**? Affects
Belleville, JB Fair Park, Trolley Square, Brainerd, Pegasus. And should partnership
expenses / management fees / organizational costs really count as funded capital (§1.7)?
*Owner: Charlene.*

### 2.6 Pegasus TGA22 pref-vs-partner bucketing
See §1.8. *Owner: Charlene.*

### 2.7 CF received after capital is fully repaid
Should distributions that arrive once capital is at zero still count in the ROE numerator,
when they add nothing to the denominator? **4 deals**: Berger Pittsburgh **$2,993,146**
(paid off 2024-07-10), 30 Bearfoot $160,612, Willowdale $38,318, Barnbeck $27,000.
*Owner: Charlene.*

### 2.8 Projected YE drops months — confirm whether it is even a defect
Months between a deal's last actual and the selected quarter-end fall in neither the actual
nor the remainder-budget window (remainder starts *after* `quarter_end`; nothing backfills).
`p0000007`'s Projected YE NOI falls 13.4M → 9.3M → 5.3M as the dropdown advances. 81 of 81
deals in the Apr snapshot have actuals ending before 2026-06-30.

**Explicitly flagged NOT a confirmed bug** — it may be intended. Pending confirmation of
whether loading the missing actuals picks those months up. Do not "fix" it first.
*Owner: Jim/Charlene.*

---

## 3. Data and operations — Jim

### 3.2 JB Fair Park balance-sheet backfill
BS data stops 6/30/2025 while peers run to 6/30/2026, so `cap['debt']` reads a stale
**12/31/2022** row on account 2150 → $66,363,992. `get_isbs_debt_balance()` detects the
staleness but keeps the last-known balance because an active MRI loan exists (LoanID 335) —
that cross-reference is what prevents a wrong $0, so this is a data gap, not a code bug.
Interest expense (5190/7030) is 0 in every period, consistent with nothing drawn.
Portfolio-wide only 3 of 83 deals are stale: JB Fair Park (30 months), Post Commons
(1 month, benign), Pegasus (21 months, already forced to 0).

### 3.3 Centre at Westbank is missing Jan/Feb 2022 rows in the MRI extract
Root cause of the one deal that breaks the §1.6 pro-rate fix. Fixing the extract is the
alternative to coding the guard.

### 3.4 Audit logging coverage on the rest of the infrastructure
**Prompted by:** the Postgres server had `log_connections` on but three days of
**undownloadable** log files and **no diagnostic settings at all** — so when it mattered
(§4, the five-month credential exposure) there was nothing to read. Fixed for Postgres on
Sep 11 2026; **nobody has checked whether the same is true elsewhere.**

Not checked: the container app `app-waterfall-dev-v2`, the registry `acrwaterfalldev`
(who pulled or pushed an image), the storage account, and the container app environment.
Four Log Analytics workspaces exist in `rg-waterfall-dev` at 30-day retention, but they
were created automatically by Container Apps — their existence is not evidence anything is
being shipped to them.

Recommended: `az monitor diagnostic-settings list` per resource, and wire anything
security-relevant into the existing workspace the way `pg-logs` now is.

The trap: an empty log query reads like "nothing happened". It usually means the log was
never collected. Check that a category is **enabled and flowing** before treating its
silence as evidence.

### 3.5 Five deals carry a valuation with NO cap rate on any row
**Verified Sep 11 2026** after the blank-column fix (§4): P0000021 ($14.5M), P0000085
($67.8M), P0000089 ($46.6M), P0000100 ($43.6M), P0000110 ($8.4M) still read a 0 cap rate,
because no valuation row for those deals carries one at all.

Why it matters: the Dashboard's weighted-average cap rate is
`sum(cap_rate x valuation) / sum(valuation)`, so each of these puts its **full valuation
into the denominator contributing nothing to the numerator**. $181M of valuation is
currently diluting a reported KPI. This is a data gap, not a code one — the fallback has
nowhere to fall back to.

Recommended: fill in `fCapRate` for those five. Expect the portfolio figure to rise again
when they land, as it did (+8.9 bps) when the six partial rows were fixed.

### 3.9 No write path is exercised against PostgreSQL before it is needed
**The process gap behind two of Sep 11's four deploys.** Both were invisible locally:

* `v435` — unquoted mixed-case SQL. SQLite is case-insensitive, PostgreSQL is not, so the
  valuation publish path had **never once succeeded against Azure** and looked tested.
* `v436` — `refresh_table('valuations')` invalidated a key nothing reads. It returned
  success. No error anywhere; a published figure simply never appeared.

Each was hidden behind the one before it, and neither could be found without running the
real path against the real database.

Recommended: a smoke test that exercises the app's WRITE paths against a PostgreSQL
instance — publish a valuation, save a waterfall, add a capital call — and asserts the
change is visible on a subsequent read. Two guardrails now cover these specific classes
(`scripts/sql_mixedcase_identifier_check.py`, `scripts/refresh_table_key_check.py`) and
both fail when the bug is reintroduced, but they are static checks and cannot catch the
next thing that only breaks on the real engine.

The trap: "covered by tests" is not the same as "has ever run in production". The publish
path had a service function, an endpoint, a UI button and a workflow around it.

### 3.6 One Pager snapshots frozen before the chart-window change
They still hold the old sparse quarter arrays and would need backfilling to match what the
live chart now renders.

### 3.7 `Debug_Progress.xlsx` was never run
Carried as outstanding since Aug 5. Confirm whether it is still wanted.

### 3.8 Burton debt roll-up — post-deploy check never recorded
The 0 → $75,302,500 fallback **only fires when ISBS returns no balance for P0000109**;
`get_isbs_debt_balance()` runs first and takes precedence. Verified locally with
`isbs_raw=None`, so the fallback ran. On Azure, if ISBS carries a parent balance that value
wins and may differ — including a possible JB-Fair-Park-style stale row. The loan-term
collapse is independent of ISBS and holds either way. Also: the guardrail swept the Apr-15
`MRI_Loans.csv` (78 loans, 110 deals); Azure carries ~24 more deals, so another parent with
co-terminous child loans would also collapse — correctly, but untested. Re-run
`scripts/burton_loandump.py` against PG.

### 3.10 The Azure app admin password was committed in plaintext for two months
`.claude/memory/MEMORY.md:45` carried `admin / Qu@kers_12` from **Jul 13 2026**
(`670902e`, "Share Claude Code memory files via repo") until it was removed on **Sep 11
2026**. It sat in the file every session is instructed to read first, and in every clone
of the repo.

Same class as the `wfadmin` Postgres credential (§4, five months public). The line is
gone from the working tree; **it is still in git history and cannot be removed from it
without a rewrite, so the credential must be treated as exposed.**

Owner: **Jim** — rotate the account password, and check the app's login audit for
sign-ins that were not the team's. Whether it was ever used is not knowable from here,
which is exactly what was true of `wfadmin`.

Recommended, beyond rotating: the pre-commit hook at `scripts/hooks/pre-commit` blocks
`://user:secret@` URLs but not a bare `user / password` line. Widening it is cheap.

**Re-checked Sep 15 2026, after Charlene reported the cleanup as incomplete.** Her
specific finding — *"`scripts/azure-complete-setup.sh:40` still has a wfadmin password in
a DATABASE_URL … a real 10-char credential (not a placeholder)"* — is a **FALSE POSITIVE,
and should not be re-raised.** The value is the literal string `<password>`, angle
brackets included, in a commented-out line. Confirmed by hash rather than by eye:
`sha256("<password>")[:12] == dd81ca61fb57`, matching the file. It has been that
placeholder in the only commit that ever touched the file. Nothing to remove, nothing to
rotate. Her scanner appears to measure length and character classes without special-casing
`<...>`; worth telling her, because it will keep firing.

**What IS real, and what is still open.** Scanning every historical version of all three
files from `838c966`:

| File | In history | At HEAD |
|---|---|---|
| `azure-complete-setup.sh` | `<password>` placeholder only | clean, always was |
| `fix_tables.py` | **real `wfadmin` password** (`sha256[:12] = 2c68d9574663`) | clean (`USER:PASS`) |
| `migrate_to_postgres.py` | same real credential | clean (`USER:PASS`) |

Removed in `3a2bfdf` (Sep 11, *"SECURITY: purge the committed wfadmin password"*), but
`838c966` is **on `origin/main`**, so the credential is in public history permanently and
removal from HEAD did nothing for it. Rotation is the only fix — which is what `v430` was
doing when it moved `DATABASE_URL` to a secret ref.

**The open question is whether that rotation actually changed the password.** Do not test
a live credential to find out. Jim can settle it without exposing the value:

```bash
read -rsp 'current wfadmin password: ' PW; echo; printf '%s' "$PW" | sha256sum | cut -c1-12
```

`2c68d9574663` means the Apr 10 credential is **still live and public** — rotate that day.
Anything else closes the incident. **Unanswered as of Sep 15 2026.**

Charlene's structural point stands and is the durable lesson: the pre-commit hook only
inspects *staged* lines, so it stops the next leak and can never see an existing one. A
repo-wide scan of tracked content was run Sep 15 — **354 files, zero real secrets** (the
one hit was a regex matching `token=')[1]?.split('` in `LoginView.vue:38`). That scan is
not automated; re-run it by hand after any incident.

### 3.11 `isbs_budget_is_supplements` has never been created on PostgreSQL — the next 3.9
The budget import creates its table on first write (`budget_import_validate._ensure_table`)
and every column is double-quoted, which is exactly the defect `v435` shipped. But that
code path **has only ever run against local SQLite.** The table does not exist on Azure
and will be created by whoever imports the first partner budget.

`scripts/sql_mixedcase_identifier_check.py` passes on it, but that is a static check —
§3.9 is the standing lesson that a static check is not a run. Recommended: import one
small budget on Azure and confirm the rows land and the comparison reads them, before the
team relies on it. Cheap now, expensive during a valuation cycle.

### 3.12 Email moved from SendGrid to Azure Communication Services — DONE, mail is JUNKED
**Resolved Sep 15 2026, same day.** Outbound email had failed since ~Aug 1 with SendGrid's
`Maximum credits exceeded` — wording that reads as *you sent too much* on an account that had
sent nothing. `/v3/user/credits` returned `total: 0, used: 0` with the reset frozen at
2026-08-01: **Twilio retired the free plan in 2025 and ours lapsed.** The allowance was zero,
not the usage. Nothing about the app, the key or the recipient was ever wrong.

**What is live** (`v458`, a config-only revision on image `9db5923`):

| | |
|---|---|
| `ecs-waterfall-dev` | Email Communication Service — owns the domain |
| `acs-waterfall-dev` | Communication Services — owns the connection string |
| `notify.peaceablestreet.com` | CustomerManaged; Domain/SPF/DKIM/DKIM2 all **Verified** |
| sender | `noreply@notify.peaceablestreet.com`, display name "Waterfall XIRR" |
| `ACS_CONNECTION_STRING` | container app **secret ref**, alongside `db-url` |
| `ACS_SENDER` | plain env var |

A real password-reset email was delivered and logged
`Email sent to jbruin@peaceablestreet.com via ACS (Succeeded)` — `Succeeded` is ACS's
terminal status, polled to completion, not merely accepted.

**THE REMAINING PROBLEM IS SPAM FILTERING, NOT CONFIGURATION.** The message landed in Junk.
The headers settle where and why, and the answer is not Azure:

- Leaving Azure: **`spf=pass`, `dkim=pass`**.
- At the gateway's own check (`mx.avanan.net`): **`spf=pass`, `dkim=pass`**.
- At final delivery to M365: `spf=fail`, `dkim=fail (body hash did not verify)` — from IP
  `35.174.145.124` = `us.cloud-sec-av.com` = **Avanan / Check Point**, the security gateway
  in front of the tenant. It modified the body (breaking the DKIM body hash) and re-injected
  from its own IP (which is not in our SPF). **That is an artifact of having a gateway, not a
  misconfiguration** — and Microsoft compensated correctly, recovering the original results
  from the ARC chain: `arc=pass`, `compauth=pass reason=130`.

The junking came from Avanan itself: `X-CLOUD-SEC-AV-SPAM-LOW: true`, `X-CLOUD-SEC-AV-SCL:
true`. Exchange then **deferred** to that verdict rather than filtering independently —
`SCL:6`, `SFV:SKS` (filtering skipped), `CAT:SPM`, `RF:JunkEmail`.

**So an Exchange-only allow rule cannot fix this.** Avanan is upstream and its verdict is
what M365 obeyed. Requested from IT Sep 15 2026:

1. Allow `notify.peaceablestreet.com` as a sender domain **in the Check Point/Avanan policy**.
2. Add DMARC — `_dmarc.notify` TXT = `v=DMARC1; p=none; rua=mailto:dmarc@peaceablestreet.com`.
   Microsoft logged `dmarc=none` and fell back to `bestguesspass`. This was deliberately left
   out of the original DNS request to avoid a policy conversation delaying the four records
   that unblocked sending; that was right for getting mail flowing and wrong for getting it
   into inboxes. IT was also asked to report any existing `_dmarc.peaceablestreet.com`, since
   a parent policy may already inherit down.

**Still open, and Jim's:**

- **Revoke the SendGrid API key.** It was stored as a PLAINTEXT env var (not a secret ref) and
  was printed into a session transcript on Sep 15 by an `az containerapp show --query value`.
  Low practical risk — the account cannot send — but it is a live credential in a transcript.
- **Then remove `SENDGRID_API_KEY` and `SENDGRID_FROM` from the container app**, which deletes
  the plaintext credential outright. Hold until ACS has carried real mail for a while: the
  SendGrid path in `email_utils.py` is the rollback, and removing the vars is what disarms it.

**Inbound is out of scope and still SendGrid.** Feedback replies use SendGrid Inbound Parse
(`POST /api/feedback/inbound-email`), which needs an MX record. Whether that was ever
configured is **unverified** — DNS lookups were blocked from the dev sandbox. If it is live it
needs re-pointing separately, and it is not covered by anything above.

**Runbook**, still accurate for a rebuild or a second domain:
<https://claude.ai/artifact/MZd8VHR5zgAFtue9yLKBDA>. Two things it records that cost time:
every ACS record name needs the `.notify` suffix because the zone sits one level above the
sending subdomain (Azure prints them without it, assuming a delegated zone), and
`az communication update --linked-domains` fails from Git Bash because MSYS rewrites the
leading `/subscriptions/...` into a Windows path — `MSYS_NO_PATHCONV=1` fixes it.

### 3.13 The ownership tree is live and has never been read against MRI
`v457`. Sidebar → Investment Management → Ownership. Derives each owner's share from the
**current** commitment — the latest `StartDate` with no `EndDate`, one row — walking up from
`deals.InvestmentID` to OWPSC, with waterfall status and a setup link at every level.

**One defect has already been found this way and fixed** (`9db5923`, see the handoff): the
first version summed every not-yet-ended row, which inflated any amended owner and
understated everyone else at that level while still totalling 100%. Jim found it by reading
the deployed tree against MRI. Assume there are more.

Specifically never exercised, because the local database holds three commitment rows:

- The **OWPSC stop**. The code handles it; no local data reaches it.
- **Chains deeper than three levels**, and the cycle guard.
- **Performance** across 101 investments at production table sizes.
- Whether **`deals.InvestmentID` is the right starting set**. Strong inference — EASTCH is
  one, PPIECH is not — but an inference.

Two things to read on real data: `superseded_rows` in the health block should be non-zero if
amendments exist, and the amber `CapitalPercent` disagreement flags should be RARE. If they
light up broadly, either commitments have real gaps worth chasing or the 0.5pp tolerance is
too tight.

Owner: **Jim** — the check is reading a few deep chains against MRI. Nobody should set up a
waterfall from this screen until that has happened once.

**Updated through `v466` (Sep 15 2026 evening).** The tree reads real data now; getting
there took four deploys and the sequence is the lesson:

- `v461` — the table was loading 601 rows and the open-commitment test discarded every
  one, because it matched a RENDERED null against a list of spellings and `pd.NA` renders
  as `<NA>`. PostgreSQL produces pd.NA where SQLite produces None. See `deploy_history.md`
  under v460 for the full post-mortem; the short version is that two fixes shipped on
  reasoning first and neither was the bug.
- `v462` — connector arrows (measured from the DOM, since the data does not know how the
  browser laid out the cards), capital balances, beneficial owners, bounded scroll.
- `v463` — **look-through**. An upper-level commitment is not this deal's money: OWPSC's
  $64M into PSC3 is not its share of the $3M PPI27 put into 30BEAR. Every node carries
  `look_through` and `effective_pct`, the percentages multiplied down from level 1. The
  direct figure is kept, subordinate, naming the entity it went into.
- `v466` — the same defect one layer down, which `v463` missed: the balance BREAKDOWN
  still described the owner's whole relationship with the entity below. Dropped above
  level 1 and replaced by the derivation.

**The repeating shape, worth recognising before the next screen is built**: a figure that
is correct about the relationship it was computed from, displayed in a context asking a
different question. It appeared three times here — commitment dollars, balance dollars,
balance breakdown — and each time it looked right and read wrong.

Still unexercised: the OWPSC stop, chains deeper than three levels, the cycle guard, and
performance at production size.

### 3.14 The MRI SQL Server password is in the public repo — ROTATE
**Found Sep 15 2026.** `flask_app/services/mri_service.py` hardcodes the MRI SQL Server
password in plaintext, with the username `PSCVPN` and both server IPs. Committed
**2026-05-05** in `08df897`, on `origin/main`, **public, four months**.

This is the most serious of the three credential exposures found this session, by a
distance. The others were a dev database (§3.10) and a dead SendGrid account (§3.12).
This is read access to **MRI itself** — accounting, commitments, relationships, ISBS,
valuations, loans, occupancy, tenants. The source of record.

The VPN requirement limits reachability. It is not access control, and a credential in a
public repo must be assumed compromised.

Owner: **Jim.** Order matters:

1. **Rotate `PSCVPN`** with whoever administers MRI.
2. **Then** move it to `MRI_USERNAME` / `MRI_PASSWORD` env vars with a container-app
   secret ref, matching `db-url` and `acs-connection-string`. Moving it first would only
   relocate a credential that is already compromised.
3. History cannot be scrubbed without a rewrite, so the old value stays exposed
   regardless. Rotation is the only real fix.

**All three exposures got through the same gap**: `scripts/hooks/pre-commit` blocked
`://user:secret@` URLs and not a bare `NAME = "value"` assignment.

**The hook was widened Sep 15 2026 (`eaff54f`) and this is verified, not assumed**:
replaying `08df897`'s `mri_service.py` verbatim — the real file, the real credential —
now exits 1 with `MRI_PASSWORD = "********"`. Five rules: the URI form, a secret-named
variable assigned a literal, self-identifying tokens (SendGrid, Anthropic, AWS, GitHub,
Slack, Azure AccountKey), a `user / password` pair, and the wfadmin name-match.
`scripts/precommit_secret_check.py` runs the real hook against a scratch repo with
synthetic secrets and NINE legitimate idioms from this codebase, because a hook that
cries wolf gets `--no-verify`'d by reflex. It caught three false positives of mine before
they shipped, one of which blocked `wfadmin:<password>@` — the very placeholder line
reported as a live credential that morning.

**It changes nothing about the three already committed.** The hook reads only staged,
added lines: it stops the next leak and can never see an existing one. All three still
need rotating.

**It only protects clones that have run** `git config core.hooksPath scripts/hooks`.
Worth confirming Charlene has.

### 3.15 The $1,347,797 that should be zero — 30BEAR / PPI27
**Open Sep 15 2026.** Jim: 30BEAR had its equity fully returned before the sale, and the
ownership tree shows PPI27 with a capital balance of $1,347,797.

**Not reproducible locally**: the same rows net to EXACTLY 0.00 under both the
`reports_service` classifier this column borrows AND the `Capital` flag — the two agree
to the cent on that pair here. Whatever produces $1,347,797 is in rows this machine does
not have.

`v464` made the figure auditable rather than guessing again: clicking a level-1 balance
opens a breakdown by Typename with each line's effect on capital outstanding, and the
card compares (never uses) what the `Capital` flag gives for the same rows.

**The hypothesis to test first, from BRECO's own breakdown**: `Contribution: Partnership
Expenses` is flagged `Capital=Y` and matched by `"contrib" in MajorType`, but it is not
equity and never returns as `Return of Capital`. It would leave exactly this kind of
residue on a deal whose equity came back in full. If that is what the panel shows, this
is **§1.7** (`"contrib" in MajorType` is over-broad) surfacing on a specific deal — not a
new defect, and the fix belongs upstream in the shared rule, not in this column.

Owner: **Jim** to read the breakdown; then whoever settles §1.7 / §2.3.

---

## 10. The FS mapping incident (Sep 19 2026) — one item still open

Fixed and deployed at `v506`. The full account is in the deploy history; this is
what is left.

### 10.1 202 of the 553 restored mappings are a FALLBACK caption — ACCOUNTING TO REVIEW
The restore wrote `consolidated_mapping()`: **190 `accounting`** (the PPI Eastchase
package's own FS Tagging), **161 `routed`** (name-matched into the same 56 captions)
and **202 `other`** — accounts that matched nothing and fell to *Other assets*,
*Other income* or *Other expenses*.

The 202 are right to within "this is an asset"; they are not right to the caption.
On a statement they read as one large Other line per section, which is a real
presentation problem and not a wrong total.

**The origin is NOT stored.** `wp_fs_map` holds acctnum / statement / fs_line /
cf_category / sort_order / updated_by / updated_at — so nothing on the mapping
screen can say which 202 to look at, and the reviewer faces all 553.

Adding an `origin` column turns "review 553" into "review 202". One migration, one
column written by `set_fs_map` from the proposal, one badge on the screen. Not done:
it is beyond what Jim asked for and it changes a table that had just been lost.
Owner: unassigned.

### 10.2 wp_fs_map has no history, so a bad edit is unrecoverable — OPEN
`set_fs_map` replaces wholesale. It now refuses an EMPTY replace, but a replace with
553 *wrong* rows is accepted and the previous mapping is gone with no way back —
there is no audit table, unlike `waterfall_audit` and `prospective_loans_audit`
which exist for exactly this shape of edit.

The restore was only possible because the seed is checked into the repo and matches
the specimen workbook exactly. A mapping the CFO had since hand-tuned would not have
been recoverable at all.

`wp_fs_map` is small (553 rows) and edited rarely, so keeping every version is cheap.
Owner: unassigned.

### 10.3 Nothing tells anyone the mapping is empty — OPEN
The engine behaved correctly throughout: no mapping, so every account is `unmapped`,
so no lines. `build()` returns `unmapped` and `unmapped_total` and the workbench has
always shown them. But an entity with **zero** mapped accounts renders as a statement
with no lines rather than as a statement that cannot be drawn, and the count that
would have explained it sits in a payload nobody reads when the page looks empty.

A statement with no lines AND a non-empty `unmapped` list should say so on its face:
"no accounts are mapped to this statement — N accounts are unmapped". That single
sentence would have turned a diagnosis into a glance. Owner: unassigned.


## 13. Valuation section — the Argus cash flow, AM's third list (Sep 28 2026)

Live at `v528` (Sep 28 2026). On production 4 records link an Argus import, none shared.

### 13.1 One upload — DONE `v528`
The Assumptions & Documents Argus upload, its route and `valuation_service.import_argus`
are removed; the tab says where the cash flow is loaded now. Budget Review's second tab is
"Load Valuation Cash Flow", and applying it CREATES / REPLACES the record's import. The
appraisal PDF upload on Assumptions is untouched.

### 13.2 Map by the account in the file — DONE `v528`
No keyword pre-fill for Argus. AND THE ACCOUNT BESIDE THE DESCRIPTION WAS NEVER READ:
`parse_budget_workbook` only recognised the account when it sat LEFT of the description,
so AM's stated layout (description, then the four-digit account) would have pre-filled
nothing. Measured on fixtures of all four layouts before and after; a column of annual
totals is still NOT taken for accounts.

### 13.3 Overturn a subtotal — DONE `v528`
"not a subtotal" / "it is a subtotal" per row. The dropdown was never actually locked on
a subtotal row -- it was greyed, tagged and not pre-filled, which read as final.

### 13.4 Found building it: the Partnership costs tick box never did anything
From `v502` to Sep 28 2026 `acceptedProposals` stayed in the browser -- not sent with the
check, the draft or the commit. Fixed. **It also means the v525 record was wrong** where
it said the $20K 5130 line put a phantom $20,000 into the tie-out: it never reached the
tie-out. The v525 fix stands (interest, partnership costs mapped from the file,
depreciation, interest income WERE inside the tie-out's NOI); that example did not.
Corrected in place in CLAUDE.md, the handoff, §12.1 and the code comments.

### 13.5 Existing linked imports — WATCH
Records already linked to an import made by the old Assumptions upload keep it until the
analyst applies a mapping, which then replaces it in place (or makes a new one if another
cycle shares it). Nothing migrates by itself.

Guardrail `scripts/argus_single_load_check.py` (42), proved against six injected defects;
`line_mapping_check` 27/9 -> 35/0 with the reversed assertions recorded; one
`mapping_draft_check` assertion reversed (it had gone vacuous -- it matched a comment).

## 12. Valuation section — asset management's second list (Sep 25 2026)

From AM (Matt + colleague) via Jim. 12.1 and 12.2 shipped in `v525`.

### 12.1 Tie-out said NOI did not tie when it did — DONE, `v525`
`budget_import_validate.reconcile` classified by PREFIX (any 4xxx revenue, any 5xxx
expense) — a second NOI definition. The Budget column sums `IS_ACCOUNTS` REVENUES /
EXPENSES, which puts 5190, 5120/5130, 5160/5165, 5195/5210/5220/5400 and 4050 BELOW
NOI and folds 7070 into expenses. ~~The proposed $20K 5130 line alone made every import
that accepted it "not tie" by $20,000.~~ [CORRECTED Sep 28 2026: the proposed $20K 5130 line could NOT have caused it -- the tick box never left the browser, so it never reached the tie-out, or anything else. The prefix defect is real; that example of it was not.] Now reads the same sections, and lists what was
mapped below NOI. Guardrail `budget_import_mapping_check` 25 -> 33; the old code fails
the new fixture at exactly -35,000.

### 12.2 Checks panel: critical only — DONE, `v525`
`CRITICAL_WARNINGS` = sign opposite to history, magnitude, negative NOI. Everything
else is folded behind "Show N notes". The per-line "below NOI" warning is replaced by
the reconcile's outside-NOI list (its hand-kept set `{5190,7030,7060,7050}` missed 5130).

### 12.3 Budgeted occupancy row -> 2027 bars in a second colour — DONE, `v526`
Read off the budget import by LABEL ("Occupancy", "Budgeted Occupancy", "Occ %"), above
or below the month header, and pulled OUT of the mappable lines. A row only counts if its
figures read as percentages -- a hotel "Occupancy Tax" in dollars stays a line. 0.95 / 95
/ "95%" all read 95; one figure on the row is an annual budget applied to every month;
out-of-range months are refused, not clipped. Saved on commit to
`valuation_budget_occupancy` (vcode, period), replace-by-months. Orange bars
(`--color-floating`) beside the blue history, with a legend.

### 12.4 MRI loaders for budget, valuation, budgeted occupancy — OPEN, needs MRI's
accepted loader files (same method as treasury `v492`: rebuild an accepted file
byte-identical). Supersedes §5.9. `valuation_budget_occupancy` is keyed the way a loader
will want it.

### 12.5 Replacement reserves above/below NOI — DECISION (Jim/AM)
Facts for the decision: UW (Projected IS) carries replacements on **7075 "Total Recurring
Replacements"**, below NOI, on 74 deals locally. And **the chart of accounts names 7030
"Replacement Reserve Deposit"** while `config.INTEREST_ACCTS = {5190, 7030}` treats it as
INTEREST and `compute.py` writes modeled interest to it (§5.8) -- so in the AM forecast a
reserve deposit and modeled interest can share an account. Not changed; it moves Property
Financials on every deal and needs its own look before the reserves policy is set.

### 12.6 Valuation / UW toggle on the third column — DONE, `v526`
`get_budget_review(compare="underwriting")`: the same `_calculate_is_amounts` with
`source="Underwriting"`, Full Year at Dec of the budget year. **UW records debt service
as ONE figure, 7010 "Hard Debt (P&I)"** -- not 5190/7060 (UW carries neither) -- so the UW
column's Interest/Principal are BLANK and Total Debt Service carries 7010. Read through
`one_pager.uw_debt_service_for_year`, extracted from One Pager's UW DSCR (proved
identical on 428 deal-years, 51 partial-year). A deal whose UW does not reach Dec of the
budget year says so. **On production (v526): UW reaches the budget year on 131 of 162
records, but only 78 carry 7010** -- the other 53 show UW debt service BLANK. Whether
those UWs genuinely had no debt or 7010 was never loaded is a data question for AM.

### 12.7 Debt service — UW override DONE `v526`; the rest OPEN
Per-record `valuation_records.debt_service_basis` ('modeled' default, 'underwriting').
Underwriting puts UW's 7010 total in the Budget column; when UW has none for the year it
is NOT applied, the modeled figure stays, and a note says so. A partial-year UW figure is
reported, not annualised (that is what UW assumed for the year). STILL OPEN: development
deals at commitment x (treasury + spread) -- no treasury rate source exists in the app --
variable-with-cap, two loans, assumed loans. The Budget column's debt service is MODELED
from MRI loan terms (§5.3), not a figure pulled from MRI.

### 12.8 Override any 2026 Estimate line — DONE, `v526`
`valuation_estimate_overrides` (record_id, row_label). LINE ITEMS ONLY: totals, NOI,
Total Debt Service and DSCR recompute from them and are marked `*` as including one --
overriding a total would leave a column that no longer adds up. Double-click to edit;
highlighted with `*`; computed figure, who and why on hover; Revert restores it. Refused on
an approved record. Overrides on rows no longer rendered are listed, not hidden.

### 12.9 Fixed in passing: the Budget Review 500'd for a deal with no ISBS
`get_budget_review` built empty frames with NO COLUMNS and the budget helpers index
`dtEntry_parsed` unconditionally -- KeyError, HTTP 500, for any deal whose ISBS has not
arrived (a new acquisition). 44 of 84 local records. Pre-existing; the old code fails
identically.

Guardrail `scripts/budget_review_inputs_check.py` (50), proved against injected defects
(UW split, overrides ignored, occupancy left in the lines, no percentage test, totals
overridable). One fixture was vacuous first -- "93%" as TEXT never became a line whatever
the code did; a % cell arrives from Excel as 0.93.

## 11. Budget import, and a vacant suite reported as un-extracted (Sep 22-23 2026)

Shipped in `v523`. What is left, with an owner on each.

### 11.1 `budget_import_check` has ONE pre-existing failure — unassigned

`scripts/budget_import_check.py` fails on **"the account list is the deal's own
recent accounts, not the whole COA"** (`got 0`), then raises `StopIteration` at
line 159 looking for account 4010 in the choices.

**PROVED PRE-EXISTING**, not introduced by `v523`: `git stash` the working tree and
it fails identically on the unmodified code. It is a local-data artefact — the
fixture's vcode has no 4010 history in `waterfall.db` — so `account_choices()`
correctly returns nothing and the check has nothing to assert on.

Two honest options: point the fixture at a vcode that does have history, or have
the check SKIP with a reason when the deal has no actuals (the pattern
`treasury_pending_check` uses). **Do not "fix" it by loosening the assertion** —
the rule it defends is real and was measured (Asbury Commons uses 23 accounts of
169).

### 11.2 Jack has not re-run his import — NEEDS HIM

The fix is live and measured against all three of his real files offline (all
import with zero blocking errors). **Nobody has driven it through the screen on
production.** Until he does, "it works" rests on offline parsing plus a guardrail,
not on the round trip that failed before.

Tell him he no longer needs the helper column joining the account number and the
description — that was a workaround for a defect, not a requirement.

### 11.3 A screen import for P0000019 would REPLACE its CSV-loaded rows — WATCH

`isbs_budget_is_supplements` holds **324 rows for P0000019** covering 2026-01-31 to
2026-12-31, loaded by **CSV** (their `vInput` is a bare account number, not the
`"label [username]"` this importer writes; the table also carries `vCode` and a
column literally named `statement _id`, both CSV-creation fingerprints).

`commit()` replaces by (vcode, the periods in THIS file), so importing a budget for
P0000019 through the screen removes those rows for any overlapping month. **That is
the designed scope and it is correct** — a budget is re-imported until final — but
nobody would expect a screen import to clear rows that arrived by CSV. Owner: Jim,
if P0000019's budget is ever loaded through the screen.

### 11.4 "7 tenants with data not yet extracted" — NOT YET IDENTIFIED

**New business analyst, Sep 23 2026 via Jim:** *"Is there a reason why once I get to
analyst review, it shows 7 tenants with data not yet extracted?"*

**A FIRST ANSWER WAS GIVEN AND WAS WRONG. Recorded because the mistake is the
lesson.** Seven ACTIVE tenants do carry `extraction_status != 'extracted'`, and all
seven are vacant suites with no documents — so the count matched and I reported it as
the cause. **It is not**: `occupiedTenants` in the Vue filters `!t.is_vacant`, and
`validate_against_leases` selects `WHERE is_vacant = false`, so those rows reach
neither the extraction KPI nor the validation. **Matching on the number and stopping
is the error** — the same shape as counting a field that does not exist. Check that
the rows you found actually reach the screen being described.

**What IS true, measured on production before an `az containerapp exec` rate limit
cut the session short:**

| Fact | Value |
|---|---|
| Reviews that exist | TWO — Windsor Square (id 2), Market at Poplar (id 3), both at `validation` |
| Extraction KPI, review 2 | **45 / 45 — zero not extracted** |
| Extraction KPI, review 3 | **33 / 33 — zero not extracted** |
| `lease_validation` rows with status `pending` | **0** |
| Unread documents, review 2 | 10 docs, 4 term-bearing, **6 distinct tenants** |
| Unread documents, review 3 | 2 docs, both term-bearing, **2 distinct tenants** |

So the screen the analyst describes does not derive its number from tenant
`extraction_status` — that reads 0 on both reviews.

**The best remaining candidate is the UNREAD DOCUMENTS box** (`v519`), which says "N
document(s) in this review were never read". Windsor Square has 10 across 6 tenants,
and those ARE genuinely un-extracted data: Outback's Letter Agreement and Option
Letter, Sam's Club's CenturyLink document (`error`), an unassigned abstracts PDF, plus
COIs for Green Zone, O'Reilly and Velva Nail. **Six, not seven** — which is why this
is a candidate and not an answer.

**NOTHING SHOULD BE BUILT UNTIL THE SCREEN IS IDENTIFIED.** The vacant-suite gate I
proposed would have changed nothing the analyst sees. Jim to ask: which property, and
which panel is the number on?

**Still worth knowing regardless:** `lease_tenants.is_vacant` is a real **BOOLEAN on
PostgreSQL** and an integer locally, so `COALESCE(is_vacant, -1)` raises
`DatatypeMismatch` on production and works fine in dev. Cast it.

**And confirmed good:** the three Market at Poplar debris rows are
`tenant_status='disregarded'` and correctly excluded — `v501`'s reading is holding.

**Method note:** `az containerapp exec` rate-limits with `retry-after: 600` after
repeated calls. Batch production questions into ONE script; a sequence of small probes
costs a ten-minute lockout, which is what ended this investigation early.

## 9. Lease review and the GL / IA query tool (Sep 19 2026)

Shipped in `v503` and `v504`. What is left, with an owner on each.

### 9.1 Lease amendment ordering: MEASURED, and it is clean — CLOSED

Measured against production on Sep 20 2026 at `v509`, on Jim's instruction ("if
you want that measured, please run a read against the container"). **530
documents across 71 tenants, one property (Windsor Square).**

| | |
|---|---|
| documents carrying a date (stored, or parsed from the filename) | 487 |
| documents carrying an amendment ordinal | 97 |
| documents carrying **neither** | **42** |
| **AMENDMENTS carrying neither** | **0 of 100** |

**Every amendment on production can be ordered.** That was the open question and
the answer is none — so the "best-effort, reported per tenant" path exists and is
currently never taken by a document that layers rent.

The 42 are 21 typed `Original Lease` and 21 typed `Other`, and their filenames
say what they are: certificates of insurance, a reciprocal easement agreement, a
move-in form, an option letter, a landlord consent, a change-of-notices address.
None of them carries a rent step.

**Caveat, and it is §9.8's subject:** `doc_type` is not trustworthy for this
population, so "0 of 100" is 0 of the documents whose NAME says amendment. A
document misfiled under another type would not be counted here. That is a real
gap but a different one, and it is recorded separately.

### 9.9 The document classifier read the FOLDER — FIXED at `v510`

`classify_document` matched `DOC_TYPE_PATTERNS` against the whole stored path,
and every production document sits under `Tenant Leases/…`, so pattern 0
(`lease`) matched the FOLDER and short-circuited. It now reads the basename,
with no fallback to the path — that fallback would re-admit the 126 documents
the fix corrects.

Production types after the backfill, which ran on first request at `v510`:

| | before | after |
|---|---|---|
| Original Lease | 409 | **77** |
| COI | 0 | **111** |
| Other | 21 | 147 |
| Commencement Letter | 0 | 23 |
| Consent Letter | 0 | 17 |
| Option Letter / SNDA | 0 | 12 / 12 |
| Amendment | 100 | **100** (untouched, as predicted) |

**The gate widened with it, and that was the necessary half.** Extraction was
gated on `('Original Lease', 'Amendment')`; correcting the types alone would
have pushed 328 documents out of it and stripped the rent commencement date from
16 of the 38 tenants that have one. `is_term_bearing` now gates extraction AND
filters the consolidation — one function, two call sites, because two spellings
of one rule is how they come to disagree. `NON_TERM_TYPES` is `{'COI'}` alone,
measured: excluding it costs no tenant any field and removes 108 certificates of
insurance from the layering, two of which were supplying a `rent_commencement`.

Guardrail: `scripts/lease_doc_type_check.py` (37), both directions, proved
non-vacuous against each defect.

**Measurement base:** one property, Windsor Square — the only lease review on
production. The rule is general (a certificate of insurance is not a lease
anywhere) but the evidence is one roster.

### 9.2 Re-extraction and re-consolidation are DONE — CLOSED (twice)

**A SECOND full re-extraction ran Sep 21 2026 on `v517`**, to pick up the
`cam_fixed` schedules and `escalation_pct` the first run predated: **419 documents
in three hours, no failures.** Coverage HELD on every field (rent_commencement 53,
lease_expiration 57, square_feet 61, suite 71; security deposit +2, escalation −1)
and 92 fields moved — 32 escalation descriptions, 15 rent-step counts, 11 rent
commencements, 11 lease commencements, 7 suites, **one** lease expiration.
Tenants with a fixed-recovery schedule went 4 → 9.

**RUN IT WITH A DIFF.** The diff is what found the defect `v518` fixed: nine
recovery findings, three of them comparing a pro-rata ESTIMATE to the rent roll as
though the lease had capped it. A run reported as "419 documents, no errors" would
have looked like a success.

The original run's record follows.

---

Finished Sep 20 2026 at 16:27 on `v512`: **417 documents re-extracted, zero
errors**, then consolidated (the extraction ends by consolidating each review, so
that half ran itself). A separate second consolidation pass afterwards left
**71 of 71 blobs byte-identical and moved no field** — the convergence proof.

**COVERAGE ROSE ON EVERY FIELD**, which is the signature that the 219 scans are
now contributing where they previously contributed nothing:

| field | before | after |
|---|---|---|
| rent_commencement | 38 | **53** |
| lease_commencement | 34 | **51** |
| lease_expiration | 45 | **57** |
| square_feet | 43 | **61** |
| escalation_structure | 45 | **65** |
| security_deposit | 31 | **53** |
| suite | 69 | **71** |

- **110 fields newly populated** where there had been nothing.
- 65 of 71 tenants' terms moved; **one tenant (Monterrey Mexican Restaurant) had
  no terms at all before** and now has them.
- Documents applied 392 → 414; tenants with stored terms 70 → 71.
- `lease_tenants.rent_commencement` 37 → **54**.

**THE `v503` FEATURE IS LIVE FOR THE FIRST TIME.** `period_start_month` was 0 of
346 before this run and is now **305**, with **208 rent steps dated from the term**
(`effective_date_basis` set). Those only arrive from an extraction under the
post-`v503` prompt, so a step written "Months 1-12" is now placed against the
tenant's own rent commencement date rather than left unresolved.

**Spot-checked the largest move rather than trusting the aggregate.** Style
Studio's rent commencement went 2021-06-01 → 2026-06-01, which looks alarming and
is correct: the original term ran to 2026-05-31 and the 2026 First Amendment
renews 2026-06-01 → 2031-05-31. The amendment GOVERNS (last applied, `v511`
ordering), the address that had been sitting in `suite` is gone and reads P765,
and both COIs are extracted but NOT in `_documents_applied`. Several other tenants
show the same shape — Outback Steakhouse's expiry 2006-12-09 → 2026-12-09, Cici's
2017-02-28 → 2027-02-28 — stale decade-old terms replaced by the renewal that
superseded them.

**Two things worth a later look, neither blocking:**

- **`security_deposit` moved `None` → `0` on several tenants.** Zero and unknown
  are different facts, and this is the model asserting there is no deposit rather
  than declining to say. It is extraction output, not our code, but it is the
  sentinel-vs-unknown shape this repo is careful about.
- 417 of 419 term-bearing documents extracted; the 2 that did not are worth
  identifying if anything downstream looks thin.

### 9.3 `gl_detail` may never have been imported on production — CHECK FIRST
`MRI_GL_Detail` is LAST in `QUERY_REGISTRY` and its own description says "never yet
executed": unbounded GHIS on a 2GB container is the query that could kill the worker.
The GL tab of the query tool reads `gl_detail`, and says which MRI query to run when
the table is absent rather than showing an empty grid — but the CFO should not meet
that message as his first experience of the tool.

Check before pointing him at it, and if it has not run, run it deliberately and watch
the container. Owner: Jim / whoever runs the refresh.

### 9.4 The CFO's workbook is TRUNCATED — confirm nothing else was lost
In `GL & IA Queries with Filters - 09182026.xlsx`, row 49 — the third `UNION ALL`
branch of the IA query, covering `ia_noncashtrans` — is cut off mid-statement at 124
characters. Our `MRI_IA_Transactions.sql` does include non-cash transactions, so the
tool covers that branch, but if he pasted from a longer original something else may
have been lost with it. Worth one look at his source. Owner: Jim to ask the CFO.

### 9.5 The IA date bound differs from his sheet ON PURPOSE — DECISION NEEDED (Jim)
His query is `contributiondate < &SPARM03` — strictly before. The tool's To date is
**inclusive**, and says so on screen, because "to 6/30" excluding 6/30 surprises
people. A transaction dated exactly on the end date is therefore IN the tool and OUT
of his spreadsheet.

Setting To one day earlier reproduces his figure exactly. If he would rather the tool
tie by default, it is a one-character change (`<=` to `<`) plus the label. Not picked
unilaterally because it changes which rows a saved query returns.

### 9.6 GL / IA Query access is READ-OPEN — DECISION NEEDED (Jim)
Reads are open to any signed-in user, matching the rest of the accounting section
(`CLAUDE.md`, "Who may edit the Accounting section": reads open, writes gated).
Nothing in the tool writes.

But this is a **bulk export of entity-level GL**, which is a wider read than viewing
one entity's statement, and it exports to a workbook that gets mailed around.
Narrowing it to `ACCOUNTING_ROLES` is one decorator on six endpoints. Raised Sep 19
2026; left consistent with the section rather than narrowed on my own judgement.

### 9.7 The MRI VPN password is committed in git — JIM ROTATES
Hardcoded at `flask_app/services/mri_service.py:41` as `MRI_PASSWORD`, with **no
environment fallback**, and repeated in three `.claude/memory/` files
(`vpn_tunnel_handoff.md`, `mri_databases.md`, `MEMORY.md`). In the repo since
`670902e`.

**Jim rotates it; I never handle the secret.** Once rotated, the fix is the pattern
already used for `DATABASE_URL` and `ACS_CONNECTION_STRING` — a container app secret
ref plus `os.environ.get`, and the literal stripped from all four files. Note that
removing it from the working tree does not remove it from history; whether to rewrite
history or treat rotation as sufficient is Jim's call. Raised Sep 19 2026.


### 9.8 "Both sides of the journal entry" — ANSWERED, and the grid ships instead

Jim asked twice for a filter that would show one side of an entry. Two candidates
were measured and both refused; the third answer is that no single column says
which line is the substance.

| candidate | why it fails |
|---|---|
| `ITEM = 1` | a LINE NUMBER — **13,493 distinct values** across 79,074 rows. Keeps 8.4% of rows and 5.5% of the money, and takes the on-screen net from **10,797 to 2,102,385,065**. Nothing is duplicated: 0 duplicate rows on any key, all 8,809 open-period entries balance |
| sign of `AMT` | Jim's own objection, and correct. Sign tracks the ACCOUNT'S NATURE: it keeps the expense on an expense entry but keeps the **CASH** and drops the **INCOME** on a revenue entry. Cash - PNC appears 2,786 times on the debit side and 4,360 on the credit side |

**What actually differentiates them is the ACCOUNT**, and `gl_accounts.TYPE`
carries it — the same classification the financial statements use:

| TYPE | rows | |
|---|---|---|
| B | 21,574 | balance sheet — IA Suspense, Distributions, Intercompany |
| I | 14,970 | income statement — Investment Income, Management Fee Expense |
| **C** | **7,538** | **cash** — Cash - PNC, Cash - Canada, Cash - Bank |

Jim's call was to skip the opinionated filter: "take the query results that we
are currently receiving and allow the user to filter or sort by any of the column
headers." Shipped at `v513`.

**Still available if wanted**, and this is the column to key it off: a one-click
**Hide cash lines** (exclude TYPE `C`) removes 7,538 of 44,082 open-period rows —
the literal offset on most cash transactions. Measured alternatives: income
statement only (TYPE `I`) collapses **3,615 of 5,043 entries to a single line**.
Neither is universally "the real line" — a distribution's substance is
`MR31000001 Distributions`, which is **B** — which is why no default was chosen.

### 9.11 O'Reilly Auto Parts #6716 has no rent commencement date — NEEDS THE FILE

Its lease fixes Common Area Expenses as a **35-year monthly schedule**, Lease
Year 1 $10,429.29/mo rising to Lease Year 35 $28,491.83/mo, and all 35 rows are
captured. It cannot be priced because **lease year 1 begins on the rent
commencement date and the tenant has none** — so the validation says exactly that
rather than approximating it from the calendar year, which would be right by luck
only for a January commencement.

The rent roll says $3.52/SF. For scale, $10,429.29/mo over 36,594 SF is $3.42/SF
in Lease Year 1, so the answer turns entirely on which lease year is in force.

**To close it:** find the commencement date (a commencement letter usually carries
it) and set it, or add the document and let the tenant re-read. Nothing to build.

### 9.12 The pro-rata / fixed boundary is read from ONE field — watch it

`v518` gates the recovery comparison on `cam_structure` starting with `fixed`,
because under a pro-rata lease the monthly figure is an ESTIMATE trued up at the
annual reconciliation. That is the right rule and it removed three wrong findings
— but it rests on the model's own one-word classification of the lease.

Two things to keep an eye on:

- **A misread structure now silently removes a comparison** rather than producing
  a wrong one. Safer direction, but still silent. A lease that genuinely fixes its
  recoveries and is typed `pro rata` will simply raise no finding.
- **`cam_fixed` is CAM only.** Pure Barre returned "$25.00 per month ... for water
  and sewer" as a fixed recovery; the prompt now excludes utility reimbursements,
  but no structural rule can tell one from a CAM charge, so this is a
  prompt-quality boundary, not a checked one.

Both are worth a look the next time the corpus is re-extracted: compare the count
of `fixed` structures before and after, and read any that changed.

### 9.10 Five debris rows still carry `lease_end = 'NaN'` — cosmetic

Market at Poplar (review 3) holds five rows whose `lease_end` is the literal
string `'NaN'`: the building banner `Market @ Poplar`, `Sub-total for
Building: 925`, `Grand Total for Report`, and two `Vacant`. They are the same
debris `v501` read as "No lease" and hid from the roster.

`v514` made them harmless — the expiration histogram now filters on
`tenant_status` and guards `pd.isna` — so this is tidiness, not a defect.
**Worth knowing before deleting them:** the readings are deliberate and
reversible, and `v501`'s whole point was that a row read as not-a-tenant is
HIDDEN, never deleted. Anything that purges them should respect that.

Windsor Square (review 2) has none.

## 8. One number, one engine — the sweep (Sep 18 2026)

**Two of the duplicates found in this sweep shipped in `v503`** — accrued pref
(the ROE Summary and Committee Summary were on a second implementation that
lost a day at every year end) and the Committee tab's `value - debt` net
proceeds estimate. The standing rule is in `CLAUDE.md` under **ONE NUMBER, ONE
ENGINE**, enforced by `scripts/one_engine_per_number_check.py`. The three below
are still open.

Jim's standing instruction is in `CLAUDE.md` under **ONE NUMBER, ONE ENGINE**. Two
duplicates were collapsed the same day (`accrued pref`, `net proceeds`). These three
were found in the same sweep and deliberately **not** changed, because each needs a
decision or data this environment cannot supply. Each one is a place where two parts
of the app can print different numbers for the same fact.

### 8.1 Capital balance: one floored, one not — DECISION NEEDED (Jim)
Both paths accumulate identically (`funded - roc` is arithmetically the same running
total `loaders.capital_after` produces). They differ only in what they do when the
running total ends **below zero**:

* **ROE Summary** `Current Balance` prints the raw running total, which can be negative.
* **Pref Balance Detail / valuation tabs** print `capital_outstanding(running)`, i.e.
  `max(0, running)`.

Measured at 2025-12-31: of 68 deals both price, **53 agree and 15 differ** — in every
one of the 15 the ROE Summary shows a negative balance where the walk shows 0.00. The
largest are PWILLOW −8,044,374.08, POUTLOO −8,008,062.00, P3RDAVE −6,777,786.00,
PLANCS1 −6,329,063.10, PCAMARI −5,091,231.39. All 15 are property-level vcodes.

A negative balance arises because `realized gain` is accumulated into return of capital
alongside actual return of capital, so a sold deal returns more than it contributed.
`capital_after`'s docstring is explicit that this is intended and that **callers should
report the below-zero case rather than absorb it** — the walk currently absorbs it by
flooring, and the ROE Summary shows it raw. Neither reports it.

**The question for Jim:** is a below-zero running balance a finding to surface (a deal
that returned more capital than it took), or is `realized gain` simply misfiled as
return of capital in the ledger? The answer decides whether both paths floor, both
report, or the classification changes. Do not pick one silently — it moves a
user-facing balance on 15 deals.

### 8.2 Debt: two implementations, UNMEASURED
* `compute.get_isbs_debt_balance(isbs_raw, vcode, as_of)` — Dashboard, Deal Analysis,
  One Pager cap stack, Committee Summary's Debt column.
* `valuation_nav_service._bs_snapshot` summed over `DEBT_BS_ACCTS` — the NAV, and so
  the Debt column on the valuation summary tab.

Both read ISBS Interim BS and the same `DEBT_BS_ACCTS`, by different code. The NAV path
also consolidates **child vcodes** and snapshots at the record's `as_of`; the other
takes a single vcode and, when called without a date, the most recent period. Those are
real behavioural differences, not just duplication.

**Not measured**, and the reason matters: the local `waterfall.db` carries **61 rows** of
`isbs_raw` in total, one of them Interim BS. Both paths return nothing for all 128 deals
locally, so a local comparison proves nothing. This needs
`scripts/` run against production (the image ships `scripts/` since `v474`). Until then,
treat the two Debt columns as potentially different numbers.

### 8.4 Two capitalization engines, measured — OPEN (Oct 6 2026)
**READ FIRST -- Jim, Oct 7 2026: "We solved the parent child relationship early on ...
some deals with the loan at the portfolio level and others with individual loans at the
property levels. We already have logic to handle the difference."** That logic is
`consolidation.py` (Feb 4 2026, `3bd3c56` / `363bb31`; DOCUMENTATION.md "Sub-Portfolio
Structure"): a child's Portfolio_Name = the parent's Investment_Name; LOANS ROLL UP FROM
PARENT AND PROPERTIES ("All loans will be included in our MRI_Loans table and should roll
to the deal level"); a sub-portfolio's FORECASTS come from the properties only. **Do not
propose a new child rule or a "properties-else-parent" loan rule** -- both were proposed
below and Jim declined. `fix/one-child-lookup` is DROPPED (left unmerged). Under the
February logic the findings below are DATA problems:
- **OREI: DONE Oct 7 2026.** LoanID 313 on the parent duplicated the two property loans;
  Jim deleted it in MRI and ran Refresh All Data from MRI. Production's `loans` table now
  holds 285 + 286 only -> 34,196,000 in both engines.
- **Burton (Jim, Oct 7 2026): "The Burton Properties will each have their own forecasts
  and budgets which will have to be rolled up to the portfolio level to calculate the
  waterfall and NAV calculation."** That IS the February rule; it applies once the
  buildings' Portfolio_Name reads "Burton Retail Portfolio" (today "Burton Portfolio";
  app-managed). ORDER: load the three buildings' forecasts FIRST (production Oct 7: 60
  rows, all on the parent P0000109, none on P0000111-113), then rename -- renaming
  first blanks Burton's projection. No code change needed.
- `ownership_service`'s reversed call: fixed on `fix/ownership-child-call` (`c5111fe`).
- **P0000049 IS marked SOLD on production** (9/4/2026). The "blocker" recorded above
  came from the LOCAL copy pulled Oct 5, where Sale_Status was still blank -- stated as
  fact without checking production. Not a blocker.
- The One Pager's Aug 6-7 helper (`_child_vcodes_for_parent`, the "Burton exception")
  and `valuation_service._child_parent_map` are workarounds for the Burton naming; with
  Burton's data aligned they become redundant, not wrong.

Measurement that led here (Oct 6):
`compute.get_deal_capitalization` (Dashboard KPIs, Deal Analysis header, the assistant's
`get_capitalization`) vs `one_pager.get_capitalization_stack` (One Pager, both Snapshot
pages, Valuation Summary). Measured on local data (recent production copy) over the
Dashboard's own 62 deals, the cap stack at 2026-Q4 (after every transaction):
**pref and partner equity agree on all 62; debt on 60.** The two that differ, both deals
with no ISBS debt, so both fall back to MRI origination amounts:

| Deal | compute | cap stack | Cause |
|---|---|---|---|
| Burton (P0000109) | 0 | 75,302,500 | compute's child lookup finds no children |
| OREI (P0000033) | 69,047,000 | 34,851,000 | compute adds the parent loan AND both child loans |

- **Burton is a CHILD-LOOKUP defect, and it reaches past capitalization.**
  `consolidation.get_property_vcodes_for_deal` / `build_property_map` match children on
  the parent's `Investment_Name` only; `one_pager._child_vcodes_for_parent` also matches
  the parent's own `Portfolio_Name` ("Burton Retail Portfolio" in "Burton Portfolio").
  The narrow one feeds the Dashboard's child exclusion (`get_child_vcodes`) -- so
  Burton's 3 buildings are listed as Dashboard deals of their own, carrying its 75.3M
  of debt; the portfolio total is right only by accident -- plus occupancy roll-ups,
  `valuation_debt_service`, ownership, financials and the assistant. Across all 134
  deals the two lookups disagree on 2: Burton, and P0000073 (consolidation calls the
  OTHER "Donald Lynch", P0000049, its child; no debt moves). Fix: ONE child lookup.
  Measure every consumer before and after -- the Dashboard deal count drops by 3.
- **OREI ANSWERED (Jim, Oct 7 2026): "OREI has two properties and each has its own
  loan. This is the same situation with the Burton portfolio."** So OREI's debt is the
  property loans, 285 + 286 = **34,196,000** -- and BOTH engines are wrong: compute
  69,047,000 (parent + properties), the cap stack 34,851,000 (parent's loan 313 instead
  of the properties'). Loan 313 carries the same 2.73% and sits 655,000 above the
  properties' total: a portfolio-level record of the same financing. Question for
  Charlene: is 313 a duplicate to remove in MRI?
  Every portfolio measured (local, Oct 7): loans on the PROPERTIES only -- Burton
  (3, 75,302,500), Berger (8, sold); on the PARENT only -- Giant 7 (97.0M), Brainerd
  (2, 64.4M), Town Fair Tire (20.0M); on BOTH -- OREI alone. **Proposed rule: a
  portfolio's loans are its properties' when the properties carry any, else the
  parent's; never both.** On today's data it moves only OREI (to 34,196,000 in both
  engines). It should also decide which loans Deal Analysis models (§8.5).
- The Dashboard's `/init-stream` calls compute WITHOUT `isbs_raw` while
  `dashboard_service` passes it, into the same cache key; identical figures on this
  data, but whichever runs first after a restart decides.
- Otherwise the two differ only in code paths no current data exercises (abs() vs
  sign-netted reversals, no sold-deal debt suppression, no date). After the lookup fix
  and the OREI answer, retire one; nothing measured argues for keeping two.

### 8.3 Prior-year figures: two SOURCES (not two engines) — DECIDED Oct 6 2026
**Jim, Oct 6 2026: "use MRI valuations as the prior-year source."** The two summary tabs
now read last year through `valuation_service._prior_rows` -- the Committee Summary's own
reader -- at the prior year-end (`valuation_summary_service._prior_from_mri`), with net
proceeds = mEquityValue and pref NAV = mMezzanineValue. On local data the 2026 cycle's
prior value fills on 48 of 82 deals (from ~1, read off the near-empty 2025 cycle); the
28 with no 2025 MRI valuation are named on the tab with their vcode. Built on branch
`feat/valuation-summary-mri-prior`, not deployed as of Oct 6 2026.
**Still open:** the Committee Summary's PRIOR net proceeds is `value - debt`, while
MRI carries mEquityValue -- the same class as the Sep 18 fix to its current-year column.
Measure both across every deal before changing either.

Original note:
The Committee Summary reads last year from MRI's `valuations` feed (`_prior_rows`, using
`mIncomeCapConcludedValue`, `mDebtValue`, `mMezzanineValue`). The valuation summary tabs
read last year from **the prior cycle's own `valuation_records`**. These should be the
same fact — the app publishes its concluded values to MRI — but they are two reads of
two stores, and nothing reconciles them.

**Not measured:** there is only one cycle (2025) in local data, so there is no prior
cycle to compare against. Needs production, and probably needs Jim to say which is the
source of record for a prior year once a cycle has been published.


## 7. Treasury and accounting access (Sep 17 2026, updated Sep 19)

### 7.1 The PNC connection — not built, and it needs Jim's banker

`v488`–`v490` shipped the bank side reading **files**: a PNC activity CSV and a
statement PDF, imported by hand. The automated feed is the next phase and is
blocked on PNC, not on code.

Researched Sep 17 2026:
- **PINACLE Connect** has a fixed ERP connector list; this app is not on it.
- **developer.pnc.com** offers OAuth 2.0 + mTLS APIs, gated behind a
  Relationship Manager / Treasury Management Officer conversation.
- **BAI2 over SFTP** is the fallback and is how most firms this size do it.

**No screen-scraping of PINACLE.** Credentials stay with PNC's own mechanism.

**Owner: Jim.** Until then the import tab is the feed, and it works.

**PNC answered, Sep 29 2026** (three guides in Jim's Downloads: Information
Reporting API v4.0, BAI Download & Transmission, OAuth). Existing TMSA covers
all options, no addendum. Pricing on top of current PINACLE IR charges:

| Option | PNC price | At our size (63 PNC accounts, ~240 items/mo) | Lead time |
|---|---|---|---|
| Info Reporting API | $750/mo per PINACLE ID, any number of APIs | ~$9,000/yr | avg 150 days (40–288), scales with number of APIs |
| BAI2 over SFTP | $195/mo + $30/account + $0.19/item PDR, $500 setup | ~$25,600/yr IF $30 is monthly — **ask** | 4–6 weeks |
| EDI statements | $45/mo + $25/file + $150/hr | — | 4–6 weeks |
| Manual BAI2 Export in PINACLE | none quoted (part of IR) — **confirm** | $0 | now |

What the guides settle:
- **The API answers §7.2 and §7.7.** `/accounts` returns the FULL `accountNumber`
  for every entitled account (so quiet accounts register themselves) plus
  `availableBalance`, `currentLedgerBalance`, `closingLedgerBalance`. Up to
  **two years** of history, not 90 days. CAD Canadian-branch accounts included.
- Auth is 2-legged OAuth (`client_credentials`, scope `ViewFinancialInformationIR`
  = read-only), 15-minute tokens, `apiKey` header, plus mTLS or IP allowlist.
  **mTLS** for us: Container Apps outbound IPs are not a guarantee.
  PNC recommends a dedicated API-only PINACLE user.
- Sandbox responses are MOCKED; real data first appears in Pre-Production.
- Client must throttle; RPS limits are set with the implementation team.
- **"Statements by transmission" is ACCOUNT ANALYSIS (fee) statements in X12**,
  not the monthly DDA statements the three-way tie files. DDA statements are
  scheduled EMAIL only. Not worth buying.

Recommendation given to Jim: API, scoped to Accounts + Account Transactions
only (not recondetails, not card), keep the CSV import running during
onboarding. Open questions for PNC listed in the Sep 29 session reply.

### 7.2 `current_available` is `None` until that connection exists

Deliberate, not a gap, but it will be asked about. Available is ledger less
holds, float and pending debits, which exist only at the bank. The column is on
the accounts tab with the reason printed under it. **Do not fill it with the
ledger figure** — guardrail `treasury_api_check.py` asserts it stays `None`.

### 7.3 GL and IA upload files — DONE (v492), including the coding screen

**Correction, Sep 17 2026:** an earlier version of this item said the templates
were outstanding. They were not — Jim supplied all four with the August batch
(blank GL, blank IA, and the filled AMB6 August examples of each), and the
reconciliation guardrail had been reading one of them the whole time.

`flask_app/services/treasury_upload.py` writes both files. The contract, read
off the accepted August files rather than from a specification:

```
GL upload, all lines          49 rows   sums to      0.00   a balanced JE
GL cash lines MR10005000      24 rows        -560,022.54    = the bank's own
                                                              net movement
GL distributions MR31000001   13 rows          12,580.47
IA upload                     13 rows          12,580.47    the same thirteen
```

One coded distribution produces BOTH a GL line and an IA row; the GL line's
description names the investor whose ID the IA row carries.

**The IA file is written into a COPY of MRI's own template**, vendored at
`flask_app/mri_templates/`. Column 12 of `Transaction Values` is "Number of
Shares" and **its header cell is blank** in MRI's template — pandas reads it as
`Unnamed: 11`, and a workbook rebuilt from column names would write that string
into a header MRI parses. Copying also keeps the Guide sheet, so the file still
explains its own rules.

Refused, not repaired: an unbalanced entry, a zero amount, a period that is not
YYYYMM, a transaction type MRI's Guide disallows (its *Validation* column is
narrower than its *Available Values* column), and an IA total that does not tie
to the GL lines it mirrors.

Guardrail `scripts/treasury_upload_check.py` (41) **rebuilds both accepted
files from their own contents and asserts the result is byte-identical.** A
format check written from a specification proves only that the code agrees with
itself.

**The coding screen shipped in `v492`** as Treasury's fourth tab. One row per
bank transaction; the cash side is never typed, so the entry balances by
construction and a partly coded month cannot produce a file. The investor split
is COMPUTED and shown as an editable proposal (Jim: "compute it and show it as
an editable proposal"), from commitment AMOUNTS — see `treasury.md` for why the
stored percentages are wrong on 5 of 13 investors.

**Nothing in treasury posts to MRI, and nothing here will** — this produces two
files a person uploads.

### 7.7 One bank account still needs its number — Jim

**PPI Life Storage NY LLC holds 119,701.35** at 6/30/2026 and has had no
activity in PNC's 90-day window, so it never registered and its June statement
cannot file. `create_account` (v494) registers one by hand, but **the full
account number has to come from PINACLE** — the statement prints only
`XX-XXXX-7891`, and the form refuses a masked number because a guessed one
silently splits the account in two the moment real activity arrives.

Twelve further June statements are for accounts in the same position but holding
**0.00**, so nothing is lost by leaving them until they transact.

Since `v507` the statement is no longer merely refused: it is **held** in
`tr_pending_statements` with everything that was read, and the Import tab lists
it with its balance and a link to the PDF — which is where the full number is
printed. Typing the number there validates it against the mask, registers the
account, files the statement and routes every later pull for it.

**The thirteen others are held the same way**; twelve are 0.00 and one, **PSC
Ambassadors Fund TGA VI, holds 629,125.04**, so two of the fourteen hold real
money, not one.

**Owner: Jim.** Fourteen numbers, typed on the Import tab once §7.10 is done.

### 7.8 Bank-account to cash-account mapping — Jim

All 50 accounts are mapped to an entity (Jim, Sep 17 2026) but every one still
sits on the `MR10005000` default. Two groups need changing before their
reconciliations can match:

- **The three CAD accounts** (`7900016982`, `7900017029`, `7900021255`) belong
  on `MR10006000` — *Cash - Canada (PNC)*. On the USD default the matcher looks
  in the wrong ledger account and finds nothing.
- **PPI2, PSS1 and PIG5 hold two bank accounts each.** Left both on the default,
  the matcher pairs both months' bank lines against the same GL account.

Not a defect — the default is right for the single-account majority, which is
why Jim chose it. Context: `MR10001000` *Cash - Operating* appears in the GL for
44 entity/account pairs and is a DIFFERENT BANK, correctly absent from the PNC
dropdown, so ledger cash activity with no PNC counterpart is expected.

### 7.9 A Wells Fargo statement sits in the PNC folder

`30 June 2026 Peaceable Street Capital LLC Wells...` (`5825-5092`) is in
`Bank Statements6.2026`. The parser refuses it, which is correct — it is
a different bank with a different layout. Noted so it is not mistaken for a
parser gap. **No action unless Wells Fargo accounts need reconciling too**, which
would be a separate parser.

### 7.10 NOTHING HAS BEEN LOADED INTO TREASURY ON PRODUCTION — Jim, two clicks

**Read this before diagnosing anything in treasury.** Checked against the live
database at `v507` (Sep 19 2026): **49 accounts, 716 activity rows, 0 statements,
0 periods, 0 matches.**

The 64 June 2026 statements were parsed **locally** during the `v493`/`v494` work
to prove the parser — that is where "45 filed before, 50 after" came from. They
were never uploaded to the app. Jim recalls uploading them; the database does not.

**Consequence, and it has already been hit.** *Seed openings from statements*
returns `0 of 49 opened`, every line reading *"No statement is on file for
202606."* That is the seeder working correctly — it reads a filed statement and
will not invent an opening balance — but it reads like a failure.

**The order, and it cannot be reordered:**

1. Import tab, bulk card → the 64 PDFs in `OneDrive/Documents/2026/06.2026`.
   12 MB total, inside the 50 MB request cap and the 200-file limit.
   Measured against the real folder and the 49 registered accounts:
   **49 file** (32 with a balance, 17 dormant at 0.00), **14 held**, **1 refused**
   (§7.9, the Wells Fargo statement). Total filed ending balance
   **$32,450,887.35**. None fails its own arithmetic check.
   **Since `v508` this also opens each account at `202607`** — seeding is part of
   filing, so there is no separate step and no way to do one without the other.
2. Answer the 14 held (§7.7). Resolving one files it, which opens its chain too.
3. Reconcile July. (Not June: the activity export opens 6/22, so June can never
   be reconciled; July and August are complete.)

**Owner: Jim.** No code is needed for any of it.

### 7.4 Six accounting writes had NO role check — FIXED, and worth knowing why

`v490`. `transition`, `assign`, exhibit upload, **exhibit DELETE**, schedule
signoff and step writes were `@login_required` only: any signed-in user,
**viewers included**, could delete a workpaper exhibit or sign off a tracker
cell.

They survived a guardrail that had been reporting green, because that check
grepped for a decorator's exact text. **A string that is absent looks exactly
like a rule that does not apply.** Found by rewriting the check to enumerate the
section's routes from the Flask app and call each one as each role.

**The lesson generalises:** any access rule asserted by reading source text is
blind to the routes that never had the text. `scripts/accounting_access_check.py`
is the pattern to copy — it covers new endpoints the day they are written.

### 7.5 The role model cannot express "the CFO but not an accountant" by level

`ROLE_LEVELS` puts `analyst`, `accountant`, `accounting_manager` and `cfo` all
at level 1. `role_required` is a level comparison, so naming any one of them
admits all four. The accounting section now uses `roles_exactly` (membership).

**Still true everywhere else in the app.** 104 endpoints use `role_required`. If
a future rule needs to separate two level-1 roles outside accounting, it needs
`roles_exactly` too — the decorator's wording will otherwise read as a
restriction that is not there.

### 7.6 Deadline validation is write-time only — unchanged, restated

`validate_due_date()` runs when a deadline is written. A bad deadline stored
before the rule existed stays stored. Now that deadlines are CFO-only, fewer
people can introduce one, but nothing sweeps the existing rows.

---

## 6. Accounting workpapers — the statement engine (Sep 14 2026)

### 6.1 MR22000002 is tagged to the wrong side of the balance sheet — ASK ACCOUNTING

The example PPI Eastchase package tags account `MR22000002`, named **Other
Liabilities** in the MRI chart, to the asset caption **Due from Manager**. The
app inherited that tagging with the rest of accounting's 192-account map.

Harmless in the example workbook because the account is zero for PPIECH. It is
not harmless generally: on AMB6 the same mapping puts **-629,125.04** into
assets. The balance sheet still ties out — a negative asset and a positive
liability net identically — so no tie-out can catch it. Only a reader can, and
only if they notice a minus sign in a column of positives.

What shipped meanwhile (`v452`, `5323de3`): `build()` returns
`balance_sheet.sign_anomalies`, every line facing the wrong way for its
section, with the accounts behind it. It reports; it does not resolve. Guessing
at accounting's intent inside the app is how a wrong statement gets produced
confidently.

**Owner: Jim, to ask accounting.** Either the account's name is wrong or its FS
tag is. When they answer, correct `ACCOUNT_LINE` in
`flask_app/services/fs_line_seed.py` — not the statement output.

### 6.2 The close cycle has never been created in production — OPEN

`wp_roles` has no assignments, no close cycle exists, and the step owners and
CFO deadlines are unset. The feature is deployed and idle until a CFO session
walks through: assign roles, create the first cycle, set deadlines per step.
`MC_TYPENAME_ROW` (members' capital row routing) also wants accounting's eye
before the first real package goes out.

### 6.3 Close deadlines: two things the rule does not do — KNOWN, not defects

`validate_due_date()` ships in `v455`. Two deliberate gaps, recorded so neither
is rediscovered as a bug:

**It is write-time only.** A deadline typed before the rule existed stays
stored — the local fixture still carries `GL detail reviewed = 2020-01-01` on a
period ended 2026-06-30. Correctable through the UI (clearing is always
allowed), and production has no cycles yet, so no migration was written.
`scripts/workpaper_deadline_check.py` prints a NOTE naming any it finds rather
than letting a cycle look clean.

**A pre-close deadline warns about sequence.** All 12 steps in `STEP_TEMPLATE`
are post-close work, so a date inside the period is legitimately "before a step
that comes earlier in the close". It saves; it just says so. If accounting adds
a real prep step it belongs first in `STEP_TEMPLATE` and the warning stops.
Not worth engineering for a step that does not exist.

---

---

## 5. Valuation section — asset management's six comments (Sep 11 2026)

Feedback from AM on the first pass at the valuation section, with what shipped against
each. Three are done and live in `v440`; the rest are here so they are not lost.

### 5.1 Build out the NAV display — OPEN
Design exists in `valuation_nav_module.md`. Not started.

### 5.2 NOI basis in the summary chart — DECISION NEEDED (Jim)
AM asked for either **2025 actual vs 2026 Projected YE**, or **2026 Projected YE vs 2027
Budget**. These answer different questions — the first is "how did we do", the second is
"what are we underwriting" — and the chart can only carry one as its default. Nothing to
build until this is settled.

### 5.3 Debt service and partnership costs in the Argus cashflow, automated
**Debt service: DONE — `v440` (`01f9e47`).** An Argus download is unlevered, so the
Valuation column showed 0 interest, 0 principal and a blank DSCR.
`valuation_debt_service.py` builds the monthly schedule from the deal's own loan terms —
the same engine behind Deal Analysis and the waterfall — and `get_budget_review`
substitutes it into the **Budget and Valuation columns only**. The Estimate column is
left alone: it means actuals, and its interest was actually paid. Balloons excluded,
child-property loans included, "cannot model" returns unavailable with a reason rather
than a zero. Guardrail `scripts/valuation_debt_service_check.py`, 34 checks.

**Partnership costs: OPEN.** Accounts 5120/5130, below the line, not a debt-engine
concern. They would come from the budget or from actuals; nothing built.

### 5.4 A spot to load the budget with GL mapping — DONE, `v440`
Budget Review tab → "Load Partner Budget". Excel in, category-then-account mapping,
quality checks, writes `isbs_budget_is_supplements`. Re-importable until final: a commit
replaces the same months rather than stacking revisions. See §5.9 for the MRI export,
which is the part of Jim's original request that is NOT built.

### 5.5 Cannot see or interact with code mapping for valuation — DONE, `v440`
Budget Review tab → "Review Argus Coding". The 56 keyword rules in
`argus_parser.ARGUS_COA_MAP` had always been applied silently at import; the same screen
now shows each guess tagged "keyword guess" and lets the analyst change it before it
feeds the Valuation column. Same component as 5.4 — `LineMappingPanel.vue`, one flow,
`source` is the only difference.

### 5.6 Extract key valuation assumptions — CLOSED, not a defect
AM reported the 30 Bearfoot AI extraction "did not get the same details". Checking the
records showed the extraction had not been run with the completeness check, not that it
was broken. Completeness validation shipped (`valuation_ai_service._missing_sections`,
one targeted re-ask, `scripts/valuation_ai_completeness_check.py`). Remaining action is
to **show AM the AI tab** — a walkthrough, not a build.

### 5.7 Prospective and refi loans are not in the valuation debt layer — OPEN
`valuation_debt_service` models loans from `mri_loans_raw` only. A modeled refinance
lives in `planned_loans.py` and is not consulted. Immaterial for a 12-month budget;
material across an appraiser's ten-year horizon, which is where the Valuation column
comes from. Same note applies to variable-rate loans, which `loans.py:123` models
interest-only for their full term — the screen warns about this, it is not silent.

### 5.8 Interest lands on 7030 in the AM forecast and 5190 in the valuation comparison
`INTEREST_ACCTS = {7030, 5190}`, and `compute.py` writes `list(INTEREST_ACCTS)[0]`, which
yields **7030**. The budget comparison reads `IS_ACCOUNTS['DEBT_SERVICE']['Interest']`,
which is **`['5190']`**. So the two disagree about where modeled interest belongs.

`v440` writes 5190 in the valuation section (Jim's call, Sep 11 2026) and deliberately
left the AM forecast alone — changing which account it writes would move Property
Financials rows on every deal, which is not a side effect to make while fixing a
valuation screen. **The divergence is now intentional and documented, which is better
than accidental, but it is still a divergence someone will trip over.** Worth settling
deliberately. Note also that `list(a_set)[0]` is a fragile way to pick an account.

### 5.9 Export `isbs_budget_is_supplements` to MRI — NOT STARTED
The last step of Jim's original budget request: once a budget is final in the app, send
the whole table to MRI to load. Nothing built. Note this is the reason the table is in
`PROTECTED_TABLES` while the other four supplements are not — the app is its writer and,
until this export exists, its only copy.

### 5.11 Valuation Summary tabs vs Jack's 2025 workbook — OPEN (Oct 6 2026)
Jack re-sent `2025 Valuation Summary Report - LIVE.xlsx` (Jim's OneDrive) asking for
tabs `2025_Val_Summary_1/2`. **Both already exist** (§5.10, `v503`); measured against his
workbook Oct 6 2026, the gaps are:

- **Total Capitalization: BUILT Oct 6 2026** (LIVE at v576).
  Jim: debt balance + preferred equity balance + the operating partner's equity balance,
  as of the valuation date. It is `one_pager.get_capitalization_stack`'s
  `total_cap_isbs` (ISBS debt + funded pref + funded OP equity), called exactly as the
  Portfolio Snapshot calls it, at the quarter holding the cycle date -- NOT the
  Snapshot's printed `total_cap`, which re-foots dev deals to committed figures.
  Parent/standalone rows only; a child row points at its parent. Pegasus 34,908,128 vs
  the workbook's hand-rounded 34,910,000. `compute.get_deal_capitalization` is a
  SECOND, dateless implementation of the same three legs (abs()-based, no sale
  suppression) -- not used here; measure it against the cap stack before touching it.
- **Pegasus split: BUILT Oct 6 2026**, by rule (any deal with 2+ PSC-side pref
  investors -- today only Pegasus): each investor's line is its own pref walk and the
  NAV waterfall's ALLOCATION to it, detail under the deal row, never in a subtotal.
  Balances tie to the workbook (TGA22 24,150,000; PPILFS 8,184,654.75). The tranche
  NAV needs Pegasus's NAV to be run on the cycle; MRI holds no prior-year split.
- **Grand totals must not count a row twice** (Jim, Oct 6 2026). His sheet's formulas
  overlap; the app's `_sections()` straight sum is the rule. Child property rows
  (Giant 7's, Berger's, OREI's, PMAT's) must not be added on top of their parent.
- **Prior year is blank for 2026 vs 2025**: the 2025 cycle's records are nearly empty;
  the real 2025 figures are in MRI `valuations`. Source of record is §8.3 -- Jim's call.
- No group labels are set on any record; Up/Down, rate deltas, % change, prior-year NOI
  and an Excel export are not on the tabs. Questions sent to Jack Oct 6 2026 (Jim).
- Pref spot check, 70 deals at 12/31/2025: 37 within $1, 13 differ (mostly accrual,
  app lower -- likely later payments; unproven), 20 have no app figure (no Cap_WF or no
  PSC pref steps). Pegasus TGA22: workbook 2,857,750.00 vs app 1,909,023.05.

### 5.10 Jack Day's valuation list (Sep 17 2026) — DONE, live at `v503`
Nine asks from asset management, shipped in `v500`–`v502`.

**Done and live:** the mapping draft that survives a reload (`v500`); the account number
READ FROM THE FILE rather than guessed, with a "as mapped before" column showing how the
same label was coded on this deal previously; the whole chart of accounts in statement
order in the dropdown; the 3+ digit row filter; the $20K partnership expense as a
VISIBLE proposed line to GL 5130, pro-rated by months and withheld when the file already
carries 5130 — a proposal the analyst accepts, never a silent injection; the debt service
question answered (already modelled, §5.3); portfolio groups LABELLED BY JACK rather than
inferred.

**The two summary report screens** shipped in `v503` (`f151e5a`). Two entries beside
Records and Committee Summary, subtotals tying to the rendered rows, grouping
round-tripped through the real controls.

Two defects the screens exposed, both fixed in the same commit:

* **The deal was never named.** `_names` looked for `deal_name` / `property_name` /
  `name`; the deals table calls it `Investment_Name`. A missing column does not raise,
  so every row fell through to the vcode. Found on screen — no test would have caught
  it, because the fallback is a legitimate code path.
* **`prior_debt` was emitted and never rendered**, so the comparison tab had no
  comparison on the debt row. Found by the API-to-screen seam check, not by looking.

Guardrail 46 → 100, with a seam section scoped to the summary block and PROVED
non-vacuous: a key typo inside the block fails it, the same typo elsewhere does not.

**Known gap, not a defect:** on local data 40 of 84 valuation records produce no pref
figure — 32 have no Cap_WF waterfall configured and 8 have no PSC pref steps. The tab
says so at the top, broken down by reason, rather than printing a zero or leaving
somebody counting dashes. Whether those 40 should have a waterfall is a data question
for the valuation cycle, not a code one. **Not yet measured on production**, where more
waterfalls may be configured than in local dev.

**The Committee tab answers the same question a different way, and agrees.**
`get_committee_summary`'s Analysis 1 reaches the pref balance through
`reports_service.build_roe_summary_row`; the summary tab goes through
`valuation_nav_service._pref_walks` → `build_pref_balance_detail`. Measured Sep 18 2026
on the 2025 cycle: of the 56 deals both cover, **43 agree to within $1 and none
disagree**. The committee path answers 3 more (P0000003, P0000033, P0000087) because it
does not require a Cap_WF waterfall. Two paths to one fact is the shape of a future
disagreement even when today's answers match — worth collapsing onto the vetted engine,
but not urgent while they agree. Owner: unassigned.

---

## 4. Resolved since the Aug 2026 notes — do NOT re-open

Each verified fixed on Sep 11 2026 against the working tree.

### 1.15 A typed NAV step ref overrode every step sharing that iOrder — RESOLVED
Filed and closed Sep 14 2026. `valuation_step_refs` was keyed
`UNIQUE(vcode, wf_type, iorder)`, but an iOrder is not one step: 5 of 689 Cap_WF
(vcode, iOrder) groups carry two or three citations — `P0000099` iOrder 5 is 8.2(c),
8.2(d) AND 8.2(e) — so one typed ref was read back by all of them.

Closed by retiring the override rather than re-keying it. The NAV walk takes each step's
citation from the waterfall setup's own `vAmtType`, which is per row and cannot be
ambiguous; the Ref cell is read-only and a wrong citation is fixed in Waterfall Setup,
where it reaches the walk, the auditor package and the setup screen at once. Verified:
those three P0000099 rows now render 8.2(c) / 8.2(d) / 8.2(e).

The table is kept but read by nothing, so the 7 rows on `P0000004` are not destroyed by a
deploy. They were character-identical to that deal's `vAmtType` — hand-copying of data the
app already held — so dropping the table is safe whenever someone wants the cleanup.

### A published valuation now actually appears — four defects, each hiding the next (Sep 11 2026)
Publishing Town Fair's 12/31/2025 valuation took **four deploys**, because four separate
defects sat in a line. Recorded together because the sequence is the lesson.

| | |
|---|---|
| `v433` | The admin committee override existed only over the API — no button (`bc07e75` shipped the parameter, nothing could press it) |
| `v434` | Gave it a button (`46b5649`), plus an amber "cast on behalf" chip so an overridden seat never reads as a normal approval |
| `v435` | Unquoted mixed-case SQL (`776bdfc`) — publish had **never once succeeded on PostgreSQL** |
| `v436` | `refresh_table('valuations')` invalidated a key nothing reads (`a043351`) — a SUCCESSFUL publish stayed invisible |

**`v435`, the SQL.** PostgreSQL folds unquoted identifiers to lower case; the table was
created by pandas `to_sql`, which quotes, so the columns really are `dtValuation`/`vCode`.
Fixed on `valuations` AND on the `forecasts` block further down the same function, which
would have been the next failure for any record with a linked Argus import. Guardrail:
`scripts/sql_mixedcase_identifier_check.py`.

**`v436`, the cache.** `refresh_table(name)` resolves `table_to_key.get(name, name)`. The
cache holds valuations under `mri_val` and `"valuations"` was not in the map, so the
invalidation wrote a key nothing reads and returned success. The row was correct in the
database the whole time. Guardrail: `scripts/refresh_table_key_check.py`.

**Both guardrails are proven to fail when the bug is reintroduced**, not merely to pass on
a clean tree. `refresh_table_key_check` also flagged `forecasts` on its first run — a false
positive, since forecasts are reassembled into `fc`; rather than exempt reassembled tables
it now follows them to the key the reassembly writes and asserts that exists.

**What stays open from this:** §1.11 (nothing shows an approved record is unpublished),
§1.12 (a NULL value can still be published), §1.13 (no structured cost basis), §3.9 (no
write path is exercised against PostgreSQL).

### One Pager and cap-stack valuation selection — fixed Sep 11 2026
Two sibling defects, same shape, found while tracing why Town Fair showed no 2025 value.

**A blank column discarded a good valuation.** Both `one_pager.get_capitalization_stack`
(`9d9e545`) and `compute.get_deal_capitalization` (`9d9fa1d`) took the newest valuation row
unconditionally and fell back to `0.0` on a blank. Each field now falls back independently
to the most recent row that carries it — the same thing
`planned_loans.projected_cap_rate_at_date` already did.

**Six deals were already wrong**, not hypothetically: P0000077's cap rate read 0 against a
real 4.5% one row behind it, and five deals had a zeroed cost-of-sale. The portfolio
weighted-average cap rate moved **5.7969% -> 5.8859% (+8.9 bps)** — the old figure was
understated by a data gap, not a market view. Five deals still read 0 because no row has a
cap rate at all; that is §3.5.

**The valuation is now as of the report quarter** (`9d9e545`). Debt used `as_of_date=q_end`
and equity filtered `EffectiveDate <= q_end`, but the valuation always took the newest row
on file — so P.E. Exposure on Value was a ratio between two different dates. Measured
across all 93 deals carrying valuations: **0 change at the current quarter**, 53 historical
quarters corrected.

Guardrails: `scripts/onepager_valuation_selection_check.py` (10),
`scripts/capitalization_valuation_fields_check.py` (9).

### Valuation committee: override recorded, requirement per-cycle (Sep 11 2026)
Jim needed to correct a 2025 valuation when `president` and `ceo` are **held by nobody**.
His design, adopted over the one first proposed: keep the control and record the exception,
rather than narrow the committee.

`committee_approve` takes `on_behalf_of`; each seat voted by a non-holder is written with
`cast_on_behalf`, `cast_by_admin`, the admin's username and a **mandatory** reason, and the
note carries `[CAST ON BEHALF OF <ROLE> BY ADMIN <user>]`. The UI shows those seats amber
with a bullet — `• CEO (by jim)` — never the green tick of a role-holder's own vote.
Expressly preferable to giving one person all three roles, which reaches the same outcome
with a trail that shows three independent approvals. 22 checks.

Also shipped and **inert unless set**: `valuation_cycles.required_roles` (`c4173e6`), a
per-cycle subset that fails safe to the full committee on an empty, unknown or typo'd
value. It is the tool for a POLICY change; the override is the tool for an EXCEPTION. 13
checks.

### The `wfadmin` Postgres password was public for five months — INCIDENT, closed Sep 11 2026
This was carried as "no `.gitignore` rule for the password-bearing scripts", a risk to
prevent. **It had already happened.** Recording it properly, because the ticket framing
would have buried it.

**What leaked:** the live `wfadmin` password, hardcoded in `scripts/fix_tables.py` and
`scripts/migrate_to_postgres.py` (identical string, matching SHA-256). `wfadmin` is
confirmed via Azure as the `administratorLogin` of `psql-waterfall-dev`, which had
`publicNetworkAccess: Enabled` and a `0.0.0.0` firewall rule for Azure services.

**For how long:** committed in `838c966` on **2026-04-10**, found **2026-09-11** — five
months. `jjbruin/Waterfall` returns HTTP 200 to an unauthenticated request: the repo is
**public**.

**Done** (`3a2bfdf`, and the rotation): password rotated; container app moved to
`DATABASE_URL=secretref:db-url` so it is no longer plaintext in the template (revision
`v430`, verified — a login probe returns 401 from a real users-table query); both scripts
now read `DATABASE_URL` from the environment and fail closed; `.gitignore` gained a
`scripts/local_*` convention plus the named one-off diagnostics, deliberately NOT a blanket
`scripts/` ignore since the committed `*_check.py` guardrails live there; and
`scripts/hooks/pre-commit` now inspects staged **content** for a URI with an inline
password, tested both directions.

**The lesson, and why the hook is the real fix:** a `.gitignore` only protects files
someone thought to name. Nobody named `fix_tables.py` — it looked like an ordinary
migration script. Content checking catches the file nobody predicted, which is the only
kind that gets through.

**Still true, deliberately not "fixed":** the old password remains in git history forever.
Rotation is what closed the exposure; rewriting history cannot un-leak five months of
public readability. **Never verified:** whether anyone used the credential during that
window. Postgres logs would show authentication attempts from unexpected IPs. "We rotated
it" and "nothing happened" are different claims and only the first is supported.

**Open, and larger than this incident:** the repo is public — an internal financial
modeling app with deal vcodes, investor entity IDs and infrastructure names throughout.
That is a decision, not a defect, and it is still Jim's to make.

### Excel serials in `EffectiveDate` — gone from live data (closed Sep 11 2026)
The Aug 2026 audit found 101 rows / $20.18M carrying Excel serials instead of dates,
including **Woodlands Square's entire $9.7M pref equity contribution**. **They are no
longer there.**

**Measured on live Azure through the Data Explorer** (which reads the app's own database,
so no credential was handled): **13,052 accounting rows**, up from 12,403 on Aug 6 — current
data, not a stale snapshot. A `434` contains-filter on `EffectiveDate` returns nothing, and
a **descending sort returns 2026 dates at the top with no bare numbers**. That sort is the
exhaustive test: the column is text, so any 5-digit serial starting with `4` would sort
above every `19xx`/`20xx` date regardless of which year it encoded. The `434` filter alone
was not sufficient — it only covers the `43xxx` range, roughly Oct 2018 – Jan 2019, which
happened to be the two examples the audit named.

**WHY they are gone is unknown.** Nobody fixed this deliberately. Most likely a re-export
replaced the rows, since `accounting` is in `mri_service.QUERY_REGISTRY` and every refresh
overwrites the table. **That means it can recur** — if the cause was a CSV passing through
Excel (which converts dates to serials on save), the next export by the same route
reintroduces it, silently.

**The detector is `scripts/effectivedate_serial_check.py`** — read-only, reproduces the
app's exact `pd.to_datetime(errors="coerce")` parse, decodes serials to real dates, and
warns that a database fix would not survive a refresh. Run it after any accounting
re-import.

**Two things learned here that outlived the item:**
1. **The v198 filter changed the defect's shape.** In August the rows were *included* in
   Total Cap (no date filter) and *dropped* from PE Performance. `9086f16` added the filter
   to the cap stack (`one_pager.py:878`), so both paths now drop an unparseable row — the
   money would be invisible everywhere rather than inconsistently counted. A disagreement
   between two figures is noticeable; a silent drop is not. If this recurs, it recurs
   quieter than it did.
2. **A clean result from the wrong database looks exactly like a clean one from the right
   database.** The first run of the detector fell back to local SQLite because
   `DATABASE_URL` was unset, and reported "CLEAN, 11,886 rows" — which was quoted as the
   live answer. The row count was what gave it away. The script now names its source **in
   the verdict line**, exits `2` on a local clean result, and takes `--require-postgres` to
   refuse to run at all without a live connection.

### Postgres had no usable log retention — fixed Sep 11 2026
Asked of the credential incident above: did anyone actually use it during the five months?
**The answer is unknown and unknowable**, and the reason is worth recording.

What was found: `log_connections` and `log_disconnections` were **on**, so connections were
being logged — but `logfiles.retention_days = 3`, `logfiles.download_enable = off`, and
`az postgres flexible-server server-logs list` returned nothing. Diagnostic settings on the
server were `[]`: logs had **never** been shipped to any of the four Log Analytics
workspaces in the resource group (those exist only because Container Apps created them).
Azure Activity Log caps at 90 days and covers control-plane operations, not database
connections. Every retention mechanism was shorter than the five-month window, and the one
with the longest retention was not connected to the database.

Fixed: `logfiles.retention_days` 3 → **7**, and a `pg-logs` diagnostic setting now ships
`PostgreSQLLogs` to `workspace-rgwaterfalldev5uCa` at **30-day** retention. Verified both.

Query for authentication activity from here on:

    AzureDiagnostics
    | where Category == "PostgreSQLLogs"
    | where Message contains "connection authorized" or Message contains "authentication failed"
    | project TimeGenerated, Message
    | order by TimeGenerated desc

**This makes the NEXT window observable. It recovers nothing about the last one.** The only
backward-looking check left is persistent artifacts — an unexpected login role, or a table
owned by someone other than `wfadmin`:

    SELECT rolname, rolsuper, rolcreaterole, rolcanlogin FROM pg_roles WHERE rolcanlogin ORDER BY rolname;
    SELECT schemaname, tablename, tableowner FROM pg_tables
     WHERE schemaname NOT IN ('pg_catalog','information_schema') AND tableowner <> 'wfadmin';

Neither is conclusive — reading data leaves no trace — and **as of Sep 11 2026 neither had
been run.** Whether the same logging gap exists on the container app, the registry and
storage is open as §3.4.

### The `cbui` admin JWT — expired on its own, nothing to rotate (closed Sep 11 2026)
Carried as "rotate it, whether that happened is unknown." It could not have been rotated,
and did not need to be.

**Verified Sep 11:** `JWT_EXPIRATION_HOURS = 24` (`flask_app/config.py:15`). The token was
pasted on **Aug 6 2026** — roughly five weeks before it was reviewed. It expired on Aug 7
and has been inert since. No action was available or required.

Note what this is NOT: a password change would not have helped. Tokens are self-contained
and there is no revocation path — see §1.10, which is the durable finding this incident
actually produced. The exposure window for any leaked token is up to 24 hours with no way
to intervene, and that is worth fixing; this particular token is not.

| Was open | Status |
|---|---|
| **abs() reversal sign bug** (partner equity 2× on 4 deals, ROE numerator on 4) | **Fixed.** The rule now lives once in `loaders.capital_after` — sign is direction, type decides only *whether* capital moves. Guardrail `scripts/capital_reversal_sign_check.py`. Detail + the two traps in `capital_reversal_and_psc3.md`. |
| **7073 missing from the U/W ROE denominator** (54 deals) | **Fixed.** `_get_uw_7073_signed()` in `one_pager.py`, with the lumpy-vs-monthly trap handled and supplement dedup keyed on (date, amount). Verified live on 61 deals. |
| **Acquisition fee shrinking the ROE denominator** (70 of 71 deals, $8.96M) | **Fixed.** `one_pager.py:2573-2581` excludes it from both `capital_events` and the numerator; the exclusion is system-wide (`compute.py`, `waterfall.py`, `reports_service.py`). |
| **`metrics.py` two contribution totals** (`:320` vs `:395`, one with a `< 0` filter) | **Gone.** One site remains (`metrics.py:405`); the inconsistency no longer exists. |
| **One Pager first-load quarter default** (label said Q2, data was Q3; budget ~51% overstated) | **Fixed** on main as `54b4700` (rebase of `c67976b`). |
| **`fmtDate` one day early** (Date Closed, U/W Exit, Anticipated Exit) | **Fixed** — ISO dates parsed by regex, no UTC round-trip. |
| **Cap-stack equity had no date filter** (Burton's $27.63M July-1 contribution in the Q2 report) | **Fixed** (`9086f16`, v198). |
| **Burton child-loan aggregation and co-terminous double-loan display** | **Fixed and live** (`2086ebc`, then `55686b0`). |
| **Chart window, NOI dual-axis scaling, Physical Occupancy relabel, Current Anticipated Exit** | All on main (`8f59bb5`, `34b8d92`, `78cbc29`, `a2d242f`). |
| **At-Close Year-0 gate** (v407) | **Decided by Jim, Sep 1 2026: leave as deployed.** Settled — do not revert, narrow, or re-raise without Jim asking. At-Close reads as an em dash on all 12 deals, including the 10 whose underlying figures are complete. Anyone reconciling a Brainerd or Pegasus At-Close to `at_close_noi` will find a real number behind a blank column; that is expected. |
| Fix 1 (Town Fair Tire), Fix 11 (2nd loan terms), Fix 12 (City, State) | False alarms or shipped (`09ec333`). |
