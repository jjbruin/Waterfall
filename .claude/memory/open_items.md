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

### 3.1 101 rows carry Excel serials in `EffectiveDate` — $20.18M
Values like `43402`, `43448` instead of dates; `to_datetime(errors='coerce')` → NaT. They
are *included* in Total Cap (no filter) and *dropped* from PE Performance. Includes
**Woodlands Square's entire $9.7M pref equity contribution** plus 91 Preferred Return
distributions ($9.44M). **Source-data fix, not a code fix.**

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

### 3.4 Rotate the `cbui` admin JWT
A JWT for user `cbui` (admin) was pasted into a session chat on **Aug 6 2026**. The note
says it was never used — direct PG access covered everything — and recommends rotating it.
**Whether that rotation happened is unknown.** Confirm or rotate.

### 3.5 No `.gitignore` rule for the password-bearing scripts
**Verified Sep 11:** no rule covers them. `scripts/inv34_*.py`, `inv5*.py`, `fixB_verify.py`,
`burton_*.py`, `pull_live_noi_requested.py` embed the Postgres password and per-developer
paths, and are kept untracked **by hand**. The Aug note already flagged that this "keeps
recurring." One rule removes the risk of an accidental `git add -A`.

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

---

## 4. Resolved since the Aug 2026 notes — do NOT re-open

Each verified fixed on Sep 11 2026 against the working tree.

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
