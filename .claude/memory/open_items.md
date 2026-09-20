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

### 9.2 Consolidation RE-RUN at `v511`; re-EXTRACTION RUNNING at `v512`

Was "the extraction has not been re-run since the prompt changed". Measured at
`v510` and it is broader than that.

**All 70 stored consolidated records predate `v503`.** Checked directly: not one
of the 70 `lease_tenants.extraction_json` blobs contains `_documents_applied`,
the field `v503` added. So every stored set of consolidated terms was built by
the OLD consolidation — before the amendment ordering fix, before rent steps
were resolved against the commencement date, and with the certificates of
insurance layered in. 43 tenants own at least one now-excluded COI.

**Consolidation does not re-run by itself** — `consolidate_review_extractions`
is called only at the end of `extract_all_documents`, so nothing refreshes until
an extraction run happens.

**A METHOD NOTE WORTH KEEPING.** The first attempt to size this asked how many
stored blobs carried a now-excluded document in `_documents_applied` and got
**0**, which reads as "nothing to do". It was vacuous: the key is absent from all
70, so the comparison could only ever return zero. A check against a field that
does not exist returns the same answer as a clean bill of health. The follow-up
counted the blobs that HAVE the field first, which is what exposed it.

**RE-CONSOLIDATION IS DONE** (`v511`, Sep 20 2026, on Jim's instruction). It
re-layers the existing per-document extractions and makes NO API calls, so it
cost nothing and is re-runnable. 70 of 70 tenants, zero errors, 392 documents
applied — the figure predicted before any of it was built.

The diff earned its keep: the first pass moved three tenants the WRONG way and
exposed an ordering defect introduced by the `v510` classifier fix. See `v511`
in the deploy history. After the ordering fix:

| | |
|---|---|
| lease expirations corrected | 9 |
| rent commencements corrected | 5 |
| suites / lease commencements / escalations | 2 / 1 / 1 |
| `lease_tenants.rent_commencement` populated | **0 -> 37** |
| coverage lost on any field | **none** (lease_commencement 35 -> 34, and that one came from a COI) |
| second full pass | 70 of 70 blobs identical, nothing moved — converged |

Several tenants had been showing terms that expired years ago and now show live
ones: Office Depot 2022-01-31 -> 2027-01-31, DSW 2024-01-31 -> 2029-01-31,
SalonCentric 2024-09-30 -> 2029-09-30, Peak Potential 2019-01-30 -> 2029-01-31.

**RE-EXTRACTION IS RUNNING** (launched Sep 20 2026 on Jim's instruction, on
revision `v512`). Detached: `/app/reex.py`, log `/app/reex.log`. **Resumable** —
anything finished is marked `extracted` and is not repeated — so if it died,
re-launching is safe and cheap.

What it does, and why it is not just "run the button": `extract_all_documents`
selects `extraction_status IN ('pending','text_extracted')`, so it SKIPS anything
already extracted — running it untouched processes ~26 documents, not the 419
whose extractions predate the `v503` prompt. The runner therefore resets the
term-bearing extracted documents to `text_extracted` first (which keeps
`extracted_text`, so no PDF is parsed twice), then extracts, then consolidates.
COIs are deliberately not reset.

Scale, measured before launching: 419 documents, 219 of them scans going through
the `v512` PDF route, on `claude-opus-5`. ~37s per document observed, so 3-5
hours; roughly $50, which Jim approved explicitly ("I'm not price sensitive for
this task").

**WHEN IT FINISHES — this is the outstanding work:**

1. Re-run `consolidate_tenant_extractions` for all 70 tenants (no API calls).
2. **Diff the terms before against after.** Snapshot `extraction_json` and
   `rent_commencement` per tenant first. The first consolidation run is what
   caught the `v511` ordering regression; a success count catches nothing.
3. Run it a SECOND time and assert nothing moves — convergence is the proof it
   settled rather than oscillating.

Two things to check specifically in that diff: the 219 scans should ADD coverage
(they contributed nothing before, so `lease_expiration` / `rent_commencement` /
`square_feet` counts should rise, not merely shuffle), and rent steps stated as
"Months 1-12" should appear for the first time — `period_start_month` was 0 of
346 before this run, since it only arrives from an extraction under the `v503`
prompt.

**Also seen, and not an ordering problem:** Style Studio's `suite` was
`9623-F East Independence Blvd., Matthews, NC 28105` — an address the extractor
put in the suite field. Extraction quality, cosmetic (suite feeds no
calculation). Worth re-checking after the run above, which is on a much stronger
model than the one that produced it.

`period_start_month` / `period_end_month` only arrive from extractions run AFTER
`v503`. Existing rows carry the period as text in `effective_date`, which
`resolve_rent_steps` still reads — verified — so nothing is broken and no backfill is
required. But a tenant extracted before `v503` gets its period parsed from prose
rather than from a field the model filled deliberately, which is the weaker path.

Re-extracting a review is an existing button; worth doing on Market at Poplar and
whichever review carries the Hobby Lobby lease, and comparing the rent in force
before and after. Owner: unassigned.

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


### 9.8 `ITEM = 1` would not show one side of an entry — DECISION NEEDED (Jim)

Jim, Sep 19 2026: *"The CFO's GL query tool is bringing both sides of the journal
entries into the results. Let's default the filter to pull entries where
Item = 1."*

**Not shipped.** The observation is right and the remedy would be wrong, so this
needs his call. Measured against all 79,074 production `gl_detail` rows:

| | |
|---|---|
| distinct `ITEM` values | **13,493** — it is a LINE NUMBER, not a side |
| rows with `ITEM = 1` | **6,618 of 79,074** (8.4%) |
| share of the money | **5.5%** (2.65bn of 48.5bn absolute) |
| net `AMT`, all rows | **10,797** — a balanced ledger |
| net `AMT`, `ITEM = 1` | **2,102,385,065** |
| duplicate rows, any key | **0** |
| open-period entries | 8,809, **all 8,809 balance to zero** |
| lines per entry | median **2**, mean 5.0, max **173** |

**There is nothing being duplicated.** A general ledger carries both sides
because that is what a ledger is, and the median entry having exactly 2 lines is
why every transaction appears to show up twice. `ITEM` orders the lines within an
entry; on a 173-line entry, `ITEM = 1` keeps one line and drops 172.

Defaulting to it would leave every on-screen total wrong — and wrong in the way
that does not announce itself, since the figure is still a plausible, correctly
formatted number.

**What already answers the real need:** the ACCOUNT filter. Picking the accounts
in question returns only the lines hitting them, and the offsetting cash side
drops out with no data lost. It is multi-select and already works.

**Open question for Jim:** if what he wants is *one row per journal entry* rather
than one side, that is a grouped view (entry, date, ref, description, net), not a
filter — worth building, but it is a different thing and would be a new engine
for a figure, so it needs saying out loud first.

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

### 8.3 Prior-year figures: two SOURCES (not two engines)
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

**Owner: Jim.** A TMO email was offered and not requested. Until then the
import tab is the feed, and it works.

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
