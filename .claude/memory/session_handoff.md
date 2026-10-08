# Session Handoff — through Oct 8 2026 (v613 live)

## Oct 8 2026 — v601-v613: THE BOARD PACKAGE, PDF FILLS, MICROSOFT SIGN-IN, SHAREPOINT DOWNLOADS

**Live: `v613` = `7b93298`. `main` = origin, level with live + docs.** Every revision has an entry in
`deploy_history.md`; this is the map.

### What shipped (mine unless marked)
| Rev | What |
|---|---|
| v601-v604 | (Charlene) Portfolio Snapshot / ROE Summary: Donald Lynch reported under P0000073 (`REPORT_VCODE_PROMOTE` -- a per-deal one-off, flagged to Jim) |
| v605 | Board pp. 9, 23, 24, 28 + the RESULTS PACKAGE: cover (official logo), contents with printed page numbers, parts I-V, full / abbreviated, narrative attachments (images; PDFs per page) and measured pagination, footnotes and disclosures on every page, Print / PDF. NEW `loan_caps.cap_terms` (rate-cap reader; one-engine row) |
| v606 | (Charlene) Print rules gated: my global Board print CSS had blanked every later print in the session |
| v607 | (Charlene) Sold Total realized IRR = pooled XIRR; Reports counts each investment once |
| v608 | (Charlene) Snapshot Loan: at-close ratios for new deals; Giant 7 label and retirement guard |
| v609 | Board PDF keeps its fills (`print-color-adjust: exact`) -- Save as PDF dropped every background |
| v610 | CONFIG: `SSO_REDIRECT_URL` corrected to `/login` -- Microsoft sign-in had NEVER worked in production (Git Bash had stored `C:/Program Files/Git/login` since v580). Accountants confirmed it works |
| v611 | SSO guard: the callback only redirects to an app path or a web address |
| v612 | Board p. 5 year in review (+ Charlene's local pull-script backup pruning, Jim approved) |
| v613 | SharePoint picker: a file with no download link is read through Graph's /content -- Jim verified `investment_map.csv` |

### Decisions and facts recorded today
- MRI moved overnight: Burton now `Retail - Grocery`; Merle Hay and 5-15 Broad `Retail - Non-Grocery` (a
  second spelling -- both mapped). `Property_Count` still NOT fixed. `Investment_Strategy` still blank.
- The official logo is in `docs/brand/` (`brand.md`); never re-extract one.
- Board page numbers are POSITIONS in the printed package; one as-of per schedule; Brainerd one deal.

### Waiting on someone else
- **Jim -- Board p. 4**: four decisions, listed in `open_items` section 20 (he closed the question box).
- **AM / Jack** -- p. 24 partner assignments; p. 28 maturity buckets, JB Fair / Town Fair Tire floating
  vs swapped, Mount Prospect's partial cap, Middle Island's spread (open_items 20).
- **MRI** -- `Property_Count` (Apple 16, PMAT Midwest 3, Prestige 12); one spelling per asset type.
- **Accounting** -- the five future-funding questions (from Oct 7).

### Next, in the Board plan
p. 4 once Jim answers; pp. 20-21 projected sales (no input needed); p. 25 map on Jim's OK to geocode.
Narrative sections are ready for text and attachments in the Narrative tab.

### Lessons today
1. **Git Bash rewrites `/path` arguments** -- an `az --set-env-vars X=/login` stored a Windows path and
   broke Microsoft sign-in for two days unnoticed. PowerShell, or `MSYS_NO_PATHCONV=1`; read the value back.
2. **A global stylesheet outlives its route** (Charlene's v606 catch): Vite leaves a lazy chunk's CSS in
   the page, so a print rule in it acts on every later print. Gate print rules on the element being present.
3. **Browsers drop backgrounds when printing** unless asked: `print-color-adjust: exact`. Proved by
   printing through Edge's DevTools protocol with `printBackground` off (scratchpad `cdp_print.py`).
4. **Wait for the other deployer's MERGE, not just their revision.** Charlene deploys from a branch and
   merges after; building before her merge means merging her branch yourself and pushing twice.
5. **Investment Metrics' "proceeds to date" exclude returned capital** -- never infer a loss from them;
   realized losses come from `noncash_by_holding`, and only at the deal's own investment (the same
   write-off is booked again at each upstream fund).
6. **Pushes to main are refused for me here** -- commit, ask Jim to push, deploy from the pushed SHA.

### Local state
- The local test meeting has seeded narrative text and attachments (Strategy, Year in review) -- LOCAL only.
- The flask-api / vue-dev preview servers may still be running.

---


## Oct 7 2026 — v589-v600: EXPENSES, VALUATION IMPORT RULES, BOARD PHASE 1 AND THE DECK VIEWER

**Live: `v600` = `c04bfe4`. `main` = `f82e002` (v600 + its docs), level with live.** Every
revision today has an entry in `deploy_history.md`; this is the map, not the detail.

### What shipped

| Rev | What | Detail in |
|---|---|---|
| v589 | Expenses: "Add an expense" as three equal options; invoice extraction inside the add pop-up | `expense_reporting.md` |
| v590, v591 | iPhone/iPad home-screen icon "PSC Expenses"; missing static files now 404 instead of the SPA (that is what made iOS offer it) | deploy_history |
| v592 | Jack's import rules: only 4-digit accounts import; the 2027 Budget column's debt service is the budget's; loss to lease (4042) nets into 4010 | `valuation_budget.md` |
| v593 | Receipts list shows only receipts needing attention, deletable; the CFO's asks -- emails on submit/decide/approve/coding, a coding REVIEW step (CFO approves), account names, a real dropdown, payroll totals | `expense_reporting.md` |
| v594 | PE exposure future funding: variant D (split by what is still owed only at an override); Brainerd overrides entered on production | `board.md`, `pe_exposure.md` |
| v595 | (Charlene) U/W ROE un-gated; no-capital deal a dash | deploy_history |
| v596 | Cell phone reimbursement: $50/month, CFO-set rate, paid in the payroll batch automatically from Oct 2026, cell bills declined at entry | `expense_reporting.md` |
| v597 | (Charlene) Portfolio Snapshot Loan: typed LTV / DSCR / Debt Yield seeds removed | deploy_history |
| v598 | Board Phase 1: pp. 26, 27, 29-31 views; Investment Metrics' Apple exclusion dated from its 28 Jan 2026 realization | `board.md` |
| v599 | (Charlene) Investment Metrics reads a two-vcode investment's deal terms across both (Donald Lynch) | deploy_history |
| v600 | Board: an open meeting shows as the DECK -- pages on the January deck's canvas, loaded once, flipped without recalculating | `board.md` |

### Decisions Jim made today (recorded where they live)
- Board p.27 keeps ONE as-of per schedule -- no "cash through" date (`board.md`).
- Brainerd I and II are ONE deal on the Board pages (`board.md`).
- Future funding: variant D (`board.md`).
- Cell phone: everyone by default, CFO can exclude; from October 2026; cell bills only.
- The coding review step: the CFO approves.

### Waiting on someone else
- **MRI data owner -- `Property_Count`**: Apple Self Storage P0000003 1 -> 16, PMAT Midwest
  P0000036 1 -> 3, Prestige P0000080 1 -> 12. Then p.26 gives 98 properties / 23 wholly
  owned, the deck's figures.
- **MRI -- asset types and deal types**: workbook `docs/board/MRI classification changes -
  Jan 2026 board deck.xlsx` (git-excluded, local only). Three `Asset_Type` changes (Burton,
  Merle Hay, 5-15 Broad); `Investment_Strategy` is blank on all 199 deals. When it is
  populated, check the Portfolio Snapshot's `DEAL_TYPE_MAP` -- it has no "Opportunistic".
- **Accounting -- five questions** (email drafted in session for Jim to send): may
  remaining-to-fund go negative (Apple Bales, Middle Island); do Bearfoot / Lynch commitments
  count at 12/31/25; why Pontchartrain shows $2.25M unfunded in their tracker and none in
  the app; plus the rest of the draft. JB Fair / Nottingham need their own DATED override
  sets for accounting's later corrections.
- **AM -- a canonical operating-partner list**: needed for p.24, and it would also make
  pp. 29-30 readable -- Investment Metrics carries full legal names ("Evergreen DevCo,
  Inc.") so the deck table sets at ~9.4px on the canvas.
- **Charlene -- her `v597` docs** (`31b0a97`, on `fix/snapshot-loan-remove-typed-ratios`,
  NOT on main). When merged it should REPLACE my "as observed" `v597` line in
  deploy_history; expect a conflict there.
- Drafts handed to Jim this session (not sent by me): the email to Jack on the import
  rules, the reply to the CFO, the accounting questions, the expense rollout email with
  new screenshots (`docs/expense_rollout/`).

### Next, in the Board plan
Phase 1 still open: **p.23 exposure by asset class** (status "Ready now" -- PE exposure x
`Asset_Type`; reconciled in `board.md`, differs only by MRI classification), **p.24** (needs
the partner list), **p.25 map**, **p.28 debt / occupancy / DSCR**. Each new view: compose
engines in `board_views_service`, add a slide component under `components/board/`, set
`"view": True` in `SCHEDULES`, extend `board_views_check`. The deck picks it up by itself.
Phase 2 (track record) can start any time; plan doc linked at the top of `board.md`.

### Lessons today
1. **Two people deployed tonight, twice in the gap.** Charlene's v597 went live unmerged
   between my P1 and my build; her v599 between my P1 and my update. Re-running P1 caught
   both (v597 was merged so v598 kept it; v599 made my build redundant, so I did not
   deploy it). Say "deploying" to each other first.
2. **`git push origin main` is refused by the permission classifier in this setup.** Commit
   locally, ask Jim to push, then deploy from the pushed SHA -- never from an unpushed one.
3. **Windows Python cannot read Git Bash's `/tmp`** -- one split commit went wrong that way
   (redone before push). Use the scratchpad path for anything Python reads.
4. **A full `data_service.get_data()` in the container is OOM-killed** -- check imports,
   routes and config there; reconcile figures locally against production read-only.
5. **The Browser pane**: zoom is unsupported and an emulated viewport fires no `resize`;
   verify layout by measuring the DOM (clipping, overflow, fit) rather than by eye.

### Local state
- A local-only Board meeting "Jan 2026 Board (local test)" exists in `waterfall.db`.
- The flask-api and vue-dev preview servers may still be running (stopping them was refused).
- `feat/board-phase1-views` and `feat/board-deck-viewer` are merged; delete when convenient.

---


**Later Oct 5: `v566` = `96534b3`** added U.S. Treasury par yields (`UST_1M`..`UST_30Y`;
the 10-year is `UST_10Y`, 4.44% at 6/30/26) to the rates table and batched its inserts
(production load 29,889 rows in 9.1s). It also carried Charlene's CLAUDE.md compaction:
CLAUDE.md is now ~370 lines under a budget check, and the per-revision deploy record
lives in `.claude/memory/deploy_history.md`. Forward curves (from these par yields)
are the natural next step if wanted -- open_items 19.5.

## Oct 5 2026 — v565: THE PE EXPOSURE TRACKER, MARKET RATES, LOCAL DATA FROM PRODUCTION

**Live: `v565` = `032fa27`. `main` = `bd1a48a` (v565 + its docs), level with live.**
`fix/treasury-mri-text` (`8ac9fe9`) is still pushed and unmerged -- see the Oct 2
section's "Next". Full detail of v565 is in CLAUDE.md's deploy history and its two new
Key Concepts sections ("PSC Preferred Equity Exposure", "Market rates").

### What shipped (v565)

| Feature | Where | The point |
|---|---|---|
| **PSC Preferred Equity Exposure** | Reports, open to everyone | Accounting's `PSC Preferred Equity Tracker - <date>.xlsx` from our MRI copy, Spreadsheet Server links and typed percentages gone |
| **Market Rates** | Data Management | Bank of Canada USD/CAD, CORRA, policy rate; NY Fed SOFR (+30/90/180-day, index), EFFR, OBFR. Free, official, no key |
| `scripts/pull_production_db.py` | (on main since `40c6796`) | Copies production PostgreSQL into a local `waterfall.db`; instructions in CLAUDE.md "Refreshing local data from production" |

**The tracker reuses engines; the only new arithmetic is summing IA non-cash rows.**
Cost = Pref Balance Detail capital balance + realized losses; FMV = Cost + unrealized;
the seven investor columns = `ownership_chain_service.group_shares` (commitments in force
on the date, amounts multiplied down -- a new as-of mode of the Ownership walker; its
default path is unchanged); Future Funding = the One Pager's `remaining_to_fund`; CAD via
`market_rates`. Layout as the workbook: current exposure, Net Invested Equity, future
funding below, Total Equity Invested / Committed. Any quarter end or **Live** (today).

**Measured against accounting's 26Q2 tracker on production data** (and re-verified on
production after deploy, equal to the dollar): Cost 52/53, FMV 53/53, investor split
51/53, grand total **725,204,338 vs 724,660,963**. Every difference is the tracker's own
typed input: Nottingham (its Cost view omits a $2.92M June contribution its FMV view
includes), Brainerd (typed funded-to-date split; its side note gives ours 25.59/74.41),
Bel Air (two typed constants). Apple is now derived (MRI's figures, converted at Bank of
Canada 1.4210), not typed.

### Lessons -- each cost something today

1. **`groupby` silently drops null keys** -- a measurement script made 1,767 real IA rows
   (non-cash, no Effective Date) look missing from production, "a $81.2M import gap",
   traced as far as querying MRI before seeing the rows were never missing. Measure with
   `dropna=False`; count grouped vs source rows (memory `groupby-drops-null-keys`).
2. **SQLite compares dates as TEXT.** The first pull wrote `2026-06-30T00:00:00`, which
   sorts after `2026-06-30 23:59:59` -- all 879 IA rows dated 6/30 fell outside "through
   6/30". Fixed in the script (plain date / space before a time) and repaired in place
   (`--repair-dates`, 1,124,271 values).
3. **`git add -A` swept a 186 MB database backup into a commit.** Caught at pre-flight P2's
   `--stat`; never pushed; the branch was rebuilt as one commit on main and `.gitignore`
   now excludes `waterfall.db.bak-*` / `*.db.pulling`. Read the `--stat` of the whole span.
4. **The app's "Export Database" exports production's EMPTY SQLite**, not PostgreSQL --
   it never could produce a local copy (open_items 19.1).
5. **A write-off is a non-cash realized loss.** The `accounting` feed has no non-cash rows,
   so the pref engine alone carries Adirondack and City West (sold) at full capital;
   accounting's Cost counts the loss. Cost = balance + realized losses, stated as a column.
6. **InvestmentID is not unique** (Donald Lynch = MCCORD under two vcodes); ask the pref
   engine under the vcode `build_investmentid_to_vcode` gives, or it answers 0.
7. **Each PowerShell tab has its own variables** -- the pull failed once because the
   `DATABASE_URL` line ran in another tab. The documented command is one line.

### Open -- with owners

- **Jim -- the investor-group STOPS** (`pe_exposure_service.STOPS`, from accounting's
  Mapping tab) are a fixed list of entity ids, and Ambassadors is recognised by an `AMB`
  name prefix. A new TIAA vehicle or PSC fund must be added by hand or lands in F&F.
  Suggested: a small table accounting maintains.
- **Jim -- Investment Metrics' own CAD rate** (`CAD_TO_USD = 0.73`) vs the published
  rate; switching moves its figures (open_items 19.2).
- **Accounting -- Apple's cost convention**, historical vs quarter-end rate (19.3).
- **Accounting / MRI -- 27 `relationships` entities at 0%** (BRN-1..9, BURT-1..4, TFT-1,
  INV23-P/INV24-P, PSCIF1, PPI2, PSCMAN): affects the RELATIONSHIPS-based traces
  (Upstream, Portfolio Analysis, PSCKOC), not this report (19.4).
- **Jim -- forward yield curves** for refi estimates: Treasury's free par curve +
  bootstrapped forwards + a spread input; SOFR forwards are licensed. Not built (19.5).
- **Charlene** -- instructions written (`docs/Refresh_Local_Database_Instructions.md`)
  for Jim to send; her IP may need the PostgreSQL firewall.
- **Possible**: Future Funding for a deal with several holders (Pegasus) is shown whole,
  not split; in Live it is the current quarter's figure (labelled on screen).
- **Possible**: a first FULL rates load took 200s on PostgreSQL (row-by-row); fine now
  (incremental ~80 rows) but a fresh database's first Refresh button press would approach
  the 240s ingress limit.
- **Jim's call**: rotate the Google Maps key seen in Terminal 1's scrollback.

### Where things are

- Guardrails added: `scripts/pe_exposure_check.py` (28; two injected defects each caught),
  `scripts/market_rates_check.py` (18; the unsorted-fetch bug reproduces). Both run in the
  container (28/0, 18/0 on v565).
- Local `waterfall.db` now holds production data as of Oct 5 morning (~1.4 GB); previous
  file kept as `waterfall.db.bak-20261005-095933` (untracked).
- Review copy of the 6/30 workbook: `docs/review/PSC_PE_Exposure_2026-06-30.xlsx`
  (untracked).


## Oct 2 2026 — v552 -> v564: SECTION ACCESS, EXPENSES, INTERCOMPANY PAY; ONE OUTAGE; PANDAS 3; MRI DESCRIPTION RULE; ONE ADMIN

**Live: `v564` = `d6adabe`. `main` = `a1ad346` (v564 + its docs), level with live.**
ONE BRANCH IS NOT ON MAIN: `fix/treasury-mri-text` (`8ac9fe9`, pushed, not merged),
see "Next" below. Main was fast-forwarded and pushed by Claude today without a block. Full per-revision
detail is in `deploy_history.md` (moved out of CLAUDE.md on Oct 5 2026); this is what a
reader needs to carry forward.

### What shipped

| Rev | SHA | What |
|---|---|---|
| v552 | `11c3455` | Section access by username (held a day: Charlene's v549/v550 were deployed from an unpushed clone) |
| v553 | `79d21a2` | Employee expense reports, all four phases: reports/approval, receipts read by the lease model, accounting's coding + TriNet payroll batch, recurring lines / duplicates / accounting's return |
| v554 | `baf619e` | Intercompany phase 2 -- Pay and the reimbursement JE, with a DOUBLE-BATCH LOCK added in pre-flight (two simultaneous generates could both reimburse one entity; reproduced 1 run in 3 without the lock) |
| v555 | `effab11` | A NULL segment is blank under pandas 3 -- the intercompany reconciliation had carried an entity called NAN since v531 |
| v556 | -- | FAILED; ~4 min outage (below) |
| v556r | `effab11` | Restore |
| v557 | `68ffce3` | pandas pinned `<3.1`; the CFO may sign off any employee's expense report |
| v558 | `7626b3e` | Purpose and Deal as dropdowns on the expense table; receipts list below |
| v559 | `05b6f5f` | Distance wizard (Google Routes + Geocoding); `GOOGLE_MAPS_API_KEY` = secret `google-maps-key` |
| v560 | `1455419` | Charlene's -- TGA VI on the Snapshot |
| v561 | `9adfddd` | The expense line opens as a pop-up again (entry + receipt), read-only for approvers; Edit/View moved to the first column |
| v562 | `fae6013` | Expense JE descriptions to MRI's rule: `ER FK <deal> <comment>`, `IC ER ...`, initials, 80 chars, no punctuation |
| v563 | `577511b` | EVERY MRI JE description to the rule, in `build_gl_csv` (treasury, intercompany, expense); Expense Coding's receipt link opens the pop-up (it had shown a PDF as a broken image) |
| v564 | `d6adabe` | ONE ADMIN: the admin ROLE is not accounting -- approved expense reports, accounting's return, the employee list and mileage rates need an accounting role AND the Accounting section (or the `admin` USERNAME); setting approvers is the `admin` username only |

### Lessons -- each cost something today

1. **THE OUTAGE (v556, ~16:41-16:45 UTC).** `az acr build` failed mid-upload because a
   background guardrail sweep was using the local SQLite db in the same checkout; the
   chained script deployed the missing tag anyway; DEACTIVATING the broken revision then
   deprovisioned the healthy one (single-revision mode: traffic follows the LATEST).
   Rules now, in memory `deploy-gate-on-build-success`: **build only from the clean
   deploy worktree (`../waterfall-xirr-deploy`, checked out detached at the target);
   gate `containerapp update` on the tag existing AND the run succeeding for that SHA;
   back out by rolling FORWARD to the last good locked tag, never by deactivating.**
2. **RE-RUN P1 IMMEDIATELY BEFORE THE UPDATE.** It caught Charlene twice: v549/v550
   (unpushed clone) and v560 (deployed between this work's span and its update).
   Merge origin/main and recompute the span; never deploy over a live SHA you lack.
3. **PRODUCTION RUNS PANDAS 3 (since 3.0.0, Jan 21 2026); local dev ran 2.3.3.** Found
   only because `intercompany_check` failed INSIDE THE CONTAINER while passing locally.
   Pandas 3 reads a SQL NULL as NaN. A sequential sweep of all guardrails under both
   found NO other difference. Local `.venv` now matches production exactly (pandas
   3.0.6, numpy 2.5.3, SQLAlchemy 2.0.54, NO pyarrow -- with pyarrow present pandas 3
   stores strings differently). Pinned `<3.1`. **Run guardrails in the container
   after a deploy, not only locally.** Never run the sweep in parallel: checks share
   fixed temp-db names.
4. **MRI's description rule (Jim): 80 characters, letters, digits and spaces only,
   for EVERY journal-entry description.** Enforced in `treasury_upload.build_gl_csv` --
   the one writer -- via `mri_description` (comment trimmed first, then the deal; long
   common words abbreviated only when over 80; a suffix keeps its room). Any new MRI
   upload path MUST go through `build_gl_csv`. `validate_gl` warns what it cleaned.
5. **THE ADMIN ROLE IS A DEVELOPER ROLE; THE `admin` USERNAME IS JIM** (Jim, Oct 2:
   "There is only 1 Admin for the system ... and that is me. Charlene has admin rights
   to make enhancements ... if she is blocked from accounting, she should not be able to
   view or update accounting tables or screens"). Charlene saw Jim's approved expense
   report because `can_view` admitted any ACCOUNTING_ROLES role and `admin` is one.
   **Never grant accounting rights on role alone** -- use
   `sections.has_accounting_authority(user)` (username `admin`, or accounting role AND
   the Accounting section). Anything only the Admin should do checks the USERNAME
   (`sections.SUPERUSER`), like section assignment and now approvers. The screen mirrors
   it: `auth.canEditAccounting` requires `hasSection('accounting')`; `auth.canAssignSections`
   is the username test. Admin-role users today: `admin`, `cbui`, `anaik`.
6. **Assert the premise, not the screen.** "The edit screen was removed" was a link
   pushed off the right edge by the new dropdowns; "the key doesn't work" was a
   `^` pasted on the end, then the literal placeholder `PASTE_KEY_HERE` in the secret.

### Open -- with owners (`open_items.md` §17-18)

- **Accounting -- the intercompany ownership rule.** The expense coding proposal is
  right for Fairview, Nottingham, Woodlands (and Ascent's split) and WRONG for
  Pontchartrain, Gallery, Belleville: investee funds INVF7/INVF2/INVF11 keep their own
  MR15000002, so the walk stops there, while accounting books their parent PSC3. Do
  investee funds always pass through? Apple (PSS1 17.93% beside PPI2) and Brainerd
  (82.68/17.32 vs ~62.4/37.6) are data questions. Proposals are editable meanwhile;
  accounting should check every intercompany row.
- **Accounting -- setup**: PPI2 to CAD in Expense Coding; a mileage rate (mileage lines
  are refused until one is set).
- **Accounting -- the MRI rule's evidence**: MRI ACCEPTED the AMB6 August GL file with
  punctuation in 23 of 51 descriptions, and accounting's Sep 24 expense upload carries
  punctuation on all 119 lines and 12 over 80. Ask whether the restriction is specific
  characters or upload types; the app strips all punctuation regardless, as Jim asked.
- **Possible**: two employees with the same initials would read alike in MRI -- not handled.
- **Jim -- UNTICK ACCOUNTING FOR CHARLENE (`cbui`)** in Settings > User Management, and
  for `anaik` if they should not see it. v564 does nothing for her until then: with
  Accounting ticked her admin role still reads approved/batched expense reports.
- **Admin -- setup**: every employee's name on reports and approver (only the `admin`
  login can now); untick Expenses for anyone who should not see it.
- **DECIDED, do not change without Jim**: Expense Coding's Void does NOT check whether
  MRI posted the batch (§17.6). Safe only for a never-uploaded file; an uploaded batch
  is reversed in MRI.
- **Possible next**: Google Places API if loose landmark names resolve poorly
  ("Pontchartrain Landing, New Orleans" resolved to Pontchartrain Blvd).

### Next -- the treasury branch (`fix/treasury-mri-text`, `8ac9fe9`, NOT deployed)

Jim asked: "Make treasury descriptions follow the MRI rule." The FILE already does (v563);
this makes the SCREEN show it. `summarise` returns `mri_text` (each description as the file
will carry it), the Journal Entry tab swaps each row's text to it and re-previews, and the
description input is `maxlength="80"`. `treasury_upload_check` 61 -> 65.
- **Still to do:** verify the JE tab in the running app (month input = 2026-08, Load the
  month, every description <=80 and punctuation-free); then DELETE the AMB6 August treasury
  rows imported into the LOCAL db for that test.
- **Raise with Jim first -- a better default for wires.** PNC wire descriptions are
  hundreds of characters with the payee at the END ("...CREDITOR:CCGS Investco LLC"), so
  cutting to 80 keeps a useless bank header. Propose extracting the payee (and reference)
  as the default text; the accountant can still edit it.
- Deploy only on Jim's approval, with the full pre-flight.

### Small things noticed, not fixed

- `vue-tsc` reports three `'total' is possibly 'null'` errors at `ExpensesView.vue`
  179-183 (pre-existing; the build does not type-check, so it ships fine).
- Preview servers from today were stopped at the end of the session.

### Where things are

- Design and measurements: `.claude/memory/expense_reporting.md` (all phases, the
  production ownership comparison, the wizard). Intercompany: `intercompany.md`.
- Guardrails added this session: `expense_report_check` 98 (§12: the admin role vs the
  Accounting section, and approvers by username; 98/0 in the v564 container), `expense_receipt_check` 53,
  `expense_coding_check` 53 (the Sep 24 acceptance now compares against accounting's text
  held to MRI's rule; every batch line <=80 and punctuation-free),
  `treasury_upload_check` 61 (accepted file identical in every field but the description),
  `expense_phase4_check` 23, `expense_distance_check` 26; `intercompany_check` 88.
- Scratchpad (session-local, not in the repo): the pandas-3 venv and sweep results,
  `venv_before_pandas3.txt` (the old package list, to roll back).

## Sep 28 2026 — THE ARGUS CASH FLOW, AM'S THIRD LIST

**`v528` = `799239a`**, live and verified on production. Status in `open_items.md` §13.
AM confirmed the whole `v526` batch works (UW dropdown, debt-service toggle, Estimate
double-click, orange budgeted-occupancy bars) and asked for the budget's framework on
the Argus file.

| Ask | What shipped |
|---|---|
| One upload | Assumptions-tab upload, its route and `valuation_service.import_argus` REMOVED. Budget Review > **Load Valuation Cash Flow** is the one way in; applying it writes the Valuation cash flow |
| Map by the file's account | Keyword pre-fill removed; the file's account, then "as mapped before" -- same as the budget |
| Overturn a subtotal | "not a subtotal" per row, pre-filling the file's account |

### What was actually wrong, and it was not what it looked like

1. **Two parsers, joined by label.** The Assumptions upload read the file with
   `argus_parser.parse_monthly_cashflow` (its own labels, keyword accounts); the mapping
   panel read it with the budget parser and wrote its mapping back BY LABEL onto that
   import. Any line the two parsers named differently took no mapping. Now the panel's
   reading IS the cash flow: `_commit_argus` creates the import, replaces it in place, or
   makes a new one when another cycle's record shares it. Signs come from the account
   (`_normalize_amount`), so Argus shows no flip box.
2. **AM's layout was never read.** `parse_budget_workbook` only found an account column
   LEFT of the description. AM said "description, then the four-digit account" -- every
   line would have come in blank. `_account_column_beside` reads it on the right, by
   membership of our chart; a column of annual totals is still refused.
3. **The subtotal was never locked** -- the dropdown worked; the row was greyed, tagged
   and not pre-filled, which read as final.

### Found building it: a tick box that did nothing, and a claim of mine that was false

The **Partnership costs** proposal ($20K, 5130) never left the browser from `v502`:
`acceptedProposals` was not sent with the check, the draft or the commit. It now rides
on the parsed file (`accepted_proposals`) and `with_accepted_proposals` makes it a line.

**That falsified my own v525 write-up**, which blamed that line for a phantom $20,000 in
the tie-out -- it never reached the tie-out. The v525 fix stands (interest, partnership
costs mapped FROM THE FILE, depreciation and 4050 really were inside the tie-out's
NOI); that example did not. Corrected in place everywhere it appeared. **Lesson: before
citing a control as a cause, trace it to the server.**

### Verification

- `argus_single_load_check` (42 locally, 35 in the container with 7 screen checks
  skipping), proved against six injected defects. One of its own checks had `or True`
  in it and was vacuous until rewritten.
- `line_mapping_check` 27/9 -> 35/0 and one `mapping_draft_check` assertion REVERSED,
  reasons recorded -- they asserted the keyword behaviour AM asked to remove. The
  mapping_draft one had gone vacuous: "keyword guess" matched a comment saying the
  guesses were gone.
- Production (PostgreSQL): every column the new commit writes exists; the old route is
  gone; **4 records link an Argus import, 0 shared**. They keep it until someone applies
  a mapping on the new tab -- nothing migrates by itself.

### Method notes

- **`npm install` after pulling** -- the traceability merge added `katex`, and the local
  build failed until installed. It also rewrote `package-lock.json`; restored, not
  shipped.
- **Heredocs mangled escapes AGAIN** (`\n` in a Python splice). Write edit scripts to
  the scratchpad with the Write tool.

### Still open

- **MRI loaders** (budget, valuation, budgeted occupancy) — need one accepted loader
  file of each from AM (§12.4).
- **Development-deal debt service** — no treasury rate source (§12.7).
- **Replacement reserves** — Jim/AM's decision; 7030 is "Replacement Reserve Deposit"
  in the COA yet treated as interest (§12.5).


## Sep 25 2026 — ASSET MANAGEMENT'S SECOND VALUATION LIST

AM (via Jim) sent eight items on the valuation section. Six shipped across `v525`
and `v526`; three need someone else before they can be built. Status lives in
`open_items.md` §12; this is what to carry forward.

### What shipped

| | Revision | What |
|---|---|---|
| "NOI not tying" when it did | `v525` | `reconcile()` classified by PREFIX (any 4xxx/5xxx) — a second NOI definition. Now reads `IS_ACCOUNTS`, same as the Budget column. ~~The proposed $20K 5130 line alone was a $20,000 phantom difference~~ — WRONG, corrected Sep 28: that tick box never reached the server |
| Checks panel noise | `v525` | Critical only (sign opposite history, magnitude, negative NOI); the rest folded behind a count |
| Valuation / UW toggle | `v526` | Header dropdown; `?compare=underwriting`; same engine, different source |
| UW debt service in the Budget column | `v526` | Per-record basis; not applied (and said so) when UW has none for the year |
| Override a 2026 Estimate cell | `v526` | Line items only; totals recompute and are marked `*`; computed figure kept |
| Budgeted occupancy bars | `v526` | "Occupancy" row read off the budget file, orange bars beside the blue history |

### Four things that are not obvious

1. **UW records debt service as ONE figure — 7010 "Hard Debt (P&I)".** It carries no
   5190 and no 7060. So a UW column or a UW-based Budget column has Interest and
   Principal BLANK and Total Debt Service = 7010. Never split it. Read through
   `one_pager.uw_debt_service_for_year`, extracted from One Pager's UW DSCR and proved
   identical on 428 deal-years (51 partial-year, 0 differences).
2. **On production, UW reaches 2027 on 131 of 162 records but only 78 carry 7010.** The
   other 53 show UW debt service blank. Correct — blank, not zero — but AM should hear it
   before reading it as "no debt". Whether those UWs had no debt or 7010 was never loaded
   is a data question.
3. **7030 is "Replacement Reserve Deposit" in the chart of accounts**, yet
   `config.INTEREST_ACCTS = {5190, 7030}` and `compute.py` writes modeled interest to it.
   Found while scoping the reserves question (§12.5). NOT changed — it moves Property
   Financials on every deal.
4. **The Budget Review 500'd for any deal with no ISBS** — empty frames had no columns and
   the helpers index `dtEntry_parsed`. 44 of 84 local records. Fixed in `v526`; the old
   code failed identically, so it had been live all along.

### v524 never took traffic — and it was not the commit

Every gunicorn worker died on boot with `No module named 'psycopg'`. **SQLAlchemy 2.1.0
was released Sep 24 2026**; `requirements.txt` said `>=2.0`, and 2.1 makes a bare
`postgresql://` URL load psycopg v3 instead of psycopg2. v523 kept serving (site 200).
**Any rebuild after Sep 24 would have failed identically, whatever it carried.** Pinned
`<2.1` in `0d68e53`. When a revision fails on boot right after a rebuild, check what
PyPI released since the last good build before reading the diff — only SQLAlchemy had
crossed a major boundary. Moving to 2.1 is a deliberate upgrade, still to do.

### Method notes

- **One fixture was vacuous and injection found it.** "The occupancy row is not a
  mappable line" passed with the exclusion deleted, because the fixture wrote `"93%"` as
  TEXT, which the line reader cannot parse — the row never became a line whatever the
  code did. A % cell arrives from Excel as the NUMBER 0.93. Five defects injected; the
  corrected check fails on the one that matters.
- **Local data cannot test UW.** Locally `isbs_raw` is 61 rows and the split
  `isbs_projected_is` has 1, so UW mode was proven on a synthetic fixture and first
  measured for real on production. The regression (old vs new, default mode) was 0
  differences on the 40 local records that rendered at all.
- **Endpoints were driven in-process** with a JWT minted from `JWT_SECRET`, the
  `accounting_access_check` method — no password typed anywhere. 16/16, including viewer
  refusals, and the script cleaned up after itself.

### Still open — each needs someone else

- **MRI loaders** (budget, valuation, budgeted occupancy): one ACCEPTED loader file of
  each from AM, then rebuild byte-identical (the treasury `v492` method).
- **Development-deal debt service** (commitment × (treasury + spread)): no treasury rate
  source exists in the app. Rate caps, two loans, assumed loans also open.
- **Replacement reserves** above/below NOI: Jim and AM's decision; the 7030 finding is
  the input.
- **Nobody has used the v526 screens yet.** AM offered to test.


## Sep 22–23 2026 — THE BUDGET IMPORT HAD NEVER WORKED ON PRODUCTION

**`v523` = `07272c6`.** Shipped and verified on production. Full entry in
`CLAUDE.md`; this is what a reader needs to carry forward.

Jack (asset management) sent feedback on the Evergreen Plaza budget upload with
**three attached spreadsheets — his versions 1, 5 and 8 of the same budget**. He
wanted to "take our Excel version of the budget and upload it directly with no
manipulation". Four asks, and a crash report at the end.

### The crash report was the whole story, and it reframed the other three

```
(psycopg2.errors.UndefinedColumn) column "vcode" does not exist
HINT: Perhaps you meant to reference the column "isbs_budget_is_supplements.vCode".
```

`commit()` quoted `"vcode"`. Production's supplement table carries **`vCode`**. A
double-quoted identifier is case-**sensitive** on PostgreSQL and case-**insensitive**
on SQLite — so it passed every local test and raised on every real import: the
DELETE raises, the transaction rolls back, and the analyst sees an empty Budget
column after doing all the mapping work.

**So the budget import had never once succeeded on production, for any file, since
it was written.** All eight of Jack's spreadsheet versions were doomed at the last
step whatever their layout. No spreadsheet could have fixed it. That is the thing
to lead with if this comes up again — the layout problems below were real, but they
were not why his imports failed.

The column names are now READ FROM THE TABLE (`_supplement_columns`, via
`inspect()`). **Quoting is not enough**, and the old docstring — which asserted the
columns "really are `vcode`" — was itself the mistake: true of the table pandas
creates locally, false of the one production has. Corrected rather than left as a
passing claim. This is the `v435` / `v496` family: PostgreSQL-only, locally
unreproducible.

**Verified on production after deploy**, because the premise is the fix:
```
ACTUAL  : ['vCode','dtEntry','vSource','vAccount','mAmount','vInput','statement _id']
RESOLVED: {'vcode': 'vCode', ...}
```

### The 324 rows that would have contradicted the claim

The table already held 324 rows, which reads as "so it HAS worked". Checked rather
than assumed: they are **P0000019**, and their `vInput` holds bare account numbers
(`5064`, `4092`, `5190`) — not the `"label [username]"` this importer writes. They
came from a **CSV upload**, which is also why the table carries `vCode` and a column
named `statement _id` with a space in it. Nothing in there came through this path.

**Flagged, not changed:** a screen import for P0000019 would REPLACE those rows for
any overlapping month, since commit is scoped to (vcode, the periods in this file).
That is the designed behaviour and it is correct, but nobody would expect it.

### It only ever read column A

The detector finds ONE label column and took the **account** column as the label —
which is exactly why Jack had built a helper column joining the number and the
description by hand. Now: a label column that is essentially all numbers is
recognised as the account with the description taken from beside it; an account
**leading** the label (`"4010 - Rental Income"`, his v8) is read; and the label's own
account outranks a separate column.

### The label and the amounts must come from the same block

His **v5 puts two independent tables side by side** — a 19-row roll-up in A–B, and
the 50-row detail it was rolled up FROM in D–G with the months beside the *detail*.
Every line read its NAME from one and its FIGURES from the other: **"5051 - Water"
carrying Property Management's 366,157.78.** Nothing about that looks wrong. The
labels are real, the amounts are real, and they belong to different lines.

A second block **announces itself with a second account column**. Without that
evidence nothing is re-based — a sheet whose labels merely have a sub-description
beside them must not be read off the sub-description, and that negative case is in
the guardrail.

**Finding that column by SHAPE got it wrong immediately**, and this is the lesson
worth keeping: a roll-up column of annual totals (1200, 240, 120) matches
"3–6 digits" perfectly, so it was read as the account column and every line came
back with account `1200`. `_find_account_column`'s own docstring had already said
this — *"an account number and a monthly amount are both 3-6 digits"* — and answers
it with a header match, which a block boundary does not have. The test is now
**membership of our chart of accounts**.

### The account decides the category; the dropdown is gone

Jack: *"right now it's two separate steps and they fight each other… the category
dropdown should come out entirely and just display whatever the account dictates."*

**Measured before reversing it:** all 80 accounts in `category_accounts()` belong to
exactly one category, so a separately-chosen category could only ever agree with the
account or contradict it — and contradicting was **blocking**. The server derives it
and IGNORES what the screen sends; the screen displays it read-only. An account on
**no** category still blocks, because a rule that only ever corrects would accept
anything. The whole chart is offered unconditionally now: with nothing narrowing the
list, a tick box would leave most accounts unreachable.

### Many lines may share one account

23 of Evergreen's repair lines are 5060. They combine, and the combining is
**reported** with the lines named and the combined figure. Simply not-blocking would
be satisfied by dropping every line after the first, which is worse than the refusal
it replaced — both directions are asserted.

### What it does on his real files

| File | Result |
|---|---|
| Original | 231 lines, 56 auto-mapped, 23 repair lines rolling into 5060 |
| v5 | **19 mis-paired lines → 52 correct ones** |
| v8 | 19 lines, all auto-mapped |

All three import with **zero blocking errors**. Two accounts are genuinely not on
our comparison (**7076** Tenant Improvements, **5019** Leasing Commissions) and are
NAMED rather than dropped — worth telling Jack. And his own Total column agrees with
the sum of its months on **228 of 229 rows** (the exception is the DSCR row, a
ratio), so the months ARE the total and no separate total column needs importing.

### Three method notes

1. **A verification of mine was vacuous.** I grepped the deployed bundle for the new
   strings and got "gone" for every one — from a **551-byte SPA shell that
   references no chunk**. `index.html` names only the ENTRY bundle; the lazy chunk
   name lives inside it. Resolved properly (`ValuationsView-rlSXpPnk.js`, 95,086
   bytes) the new strings are present and the old dropdown's are gone. Checking for
   a *server-side* string in a Vue chunk proves nothing either way, in either
   direction.
2. **`v522` was live and unrecorded** — found by pre-flight P1 while deploying
   v523. It is `e4bb231`, the traceability-tools merge. Both are recorded now. The
   history is the only thing mapping a running revision to a commit, so a missing
   entry is the same failure as an untagged image.
3. **Guardrail `budget_import_mapping_check.py` (25)**, on fixtures of all three
   real shapes plus the negative case; 25/25 in the container. Proved non-vacuous
   against **nine** injected defects including both opposite failures (combining
   blocked again; only the first line of a shared account written). Two older
   guardrails asserted the behaviour Jack asked to change and were reversed with the
   reason recorded — `mapping_draft_check` now asserts the category is displayed
   AND not selectable, since "no dropdown" alone is satisfied by deleting the column.

### Still open from this

- **`budget_import_check` has one PRE-EXISTING failure** — "the account list is the
  deal's own recent accounts" — which fails on local data (no 4010 history for the
  fixture's vcode) and fails **identically on the unmodified tree**. Not introduced
  here, not fixed here. `open_items.md` §11.1.
- **Jack has not re-run his import yet.** The fix is live and measured against his
  files offline; nobody has driven it through the screen on production.

---

# Session Handoff — through Sep 21 2026 (v518 live)

## Sep 20–21 2026 — THE VALIDATION SCREEN BECAME USABLE, and CAM got checked

**`v515` = `a21494e`, `v516` = `d27fd9f`, `v517` = `8393b25`, `v518` = `03e93aa`.**
All shipped and verified on production. Full entries in `CLAUDE.md`; this is what
a reader needs to carry forward.

### What Jim asked for, in order, and what each one turned up

**The rent roll date, and why the page looked empty.** Market at Poplar showed 23
tenants whose rent "could not be determined from the lease" while holding 148 rent
steps. The review simply had no `rent_roll_date`, so no rent could be placed in
force — and the message blamed the lease, which is what sent a reader hunting for
data that was not missing. The date is now settable on the page AND asked for at
upload (requested, not enforced). With 2026-09-01 set: 0 of 23 comparisons became
22 of 23, six real disagreements.

**Clearing a mismatch.** There was no control at all — the only thing near a
finding was a per-TENANT approve/flag two steps later, recording no value, no
reason and no document. Settle now records the figure, a REQUIRED reason and a
CHECKED document citation, survives re-validation, and feeds a change report
(rent roll → applies → difference → reason → document → who) that downloads.

**The CAM question, which was the biggest finding.** A fixed CAM charge was not
validated at all: the extraction stored the word `fixed` and never the amount.
Now captured, dated (calendar years, lease years against rent commencement, or a
verbatim period), escalations compounded by the app, and compared — as a QUESTION
where the lease passes tax or insurance through separately, since our rent roll
column is one combined figure.

**A document arriving for a scanned tenant** re-reads that tenant's WHOLE set,
after asking whether more files are coming. Which exposed that the abstract had
been frozen the moment anyone saved it.

### The three things worth remembering

1. **RUN A RE-EXTRACTION WITH A DIFF.** The second full run (419 documents, three
   hours, no failures) reported nine recovery findings and three were comparing a
   pro-rata ESTIMATE to the rent roll as though the lease had capped it. "No
   errors" would have read as success.
2. **THE TARGETED RUN BEFORE THE FULL ONE EARNED ITS PLACE** — Jim asked for it,
   and it found two more shapes (a stated escalation with no amounts, and a fixed
   amount with no period) that would otherwise have been discovered across the
   whole corpus.
3. **FOUR OF MY OWN CHECKS WERE WEAKER THAN THEY LOOKED**, each found by injecting
   the defect rather than by reading: an undated row sitting second in a fixture
   so "the first row is not used" proved nothing; two assertions raising TypeError
   instead of failing and hiding every check after them; and a section key outside
   the abstract template that never reaches the screen. Inject the defect.

### Live open items after this

`open_items.md` **§9.11** (O'Reilly needs a rent commencement date — nothing to
build), **§9.12** (the pro-rata/fixed boundary rests on one model-classified
field), **§9.3** (whether `gl_detail` was imported — it was, can probably close),
**§9.5 / §9.6** (two GL-tool decisions for Jim), **§9.7** (the MRI password in
git, Jim rotates), **§9.10** (five cosmetic debris rows).

## Sep 20 2026 — THE LEASE CORPUS, and a regression the diff caught

**`v509` = `49d2120`, `v510` = `a12f98a`, `v511` = `07a83ed`, `v512` = `e473a07`,
`v513` = `b3bebdf`, `v514` = `42d053d`.**
Everything in this section is SHIPPED and verified on production. The live open
items after it are **`open_items.md` §9.3** (whether `gl_detail` was ever imported
— it was, so this can probably close), **§9.5 / §9.6** (two decisions for Jim on
the GL tool), **§9.7** (the MRI password in git, Jim rotates) and **§9.10** (five
cosmetic debris rows).

### Two more, after the extraction landed

**`v513` — the GL grid slices.** Jim asked twice for a filter showing one side of
a journal entry, and both candidates were measured and refused: `ITEM = 1` is a
line number (13,493 distinct values; it takes the net from 10,797 to
2,102,385,065), and the SIGN of `AMT` — his own objection — keeps the expense on
an expense entry but keeps the CASH and drops the INCOME on a revenue entry.
Which line is the substance is a property of the ACCOUNT (`gl_accounts.TYPE`),
so no default was baked in and the grid became sortable and filterable on every
column instead. `open_items.md` §9.8 carries the measurements and the one-click
"hide cash lines" option if it is ever wanted.

**`v514` — "that page is blank" was three bugs, not a missing screen.** Jim asked
whether to relocate the Lease Risk validation screen into Lease Review. The Lease
Review screen was complete and its 23 rows were in the database; Windsor Square
rendered its 175 the whole time, which is why it read as missing rather than
broken. The chain: five debris rows carry the STRING `'NaN'` as `lease_end`;
`pd.to_datetime('NaN')` returns NaT WITHOUT raising so the try/except never
fired; `.year` is `nan` and BOTH range comparisons are False, so **a range guard
cannot catch NaN**; `yearly[nan]` raised and the endpoint 500'd; `Promise.all`
rejected and validation — assigned LAST — was never set; and the catch called it
"(expected for new reviews)".

**The generalisable half:** `v501` taught the roster and the headline totals to
respect `tenant_status`, and the expiration histogram was simply never updated.
When a reading is added, find every consumer of the rows it governs.

### The re-extraction is DONE (finished 16:27, Sep 20)

417 documents, zero errors, consolidated and converged. Coverage rose on every
field — rent_commencement 38 -> 53, square_feet 43 -> 61, escalation 45 -> 65,
security_deposit 31 -> 53 — with 110 fields newly populated and one tenant
gaining terms it never had. `period_start_month` went 0 -> 305 and 208 rent steps
are now dated from the term, so the `v503` month-of-term feature is live on
production for the first time. Detail and the two follow-ups in
`open_items.md` §9.2.

### The lease work, in the order it actually happened

It started as Jim asking two unrelated UI things and turned into four deploys,
because each measurement exposed the next thing.

**`v509` — the input column folds away.** One `CollapsiblePanel` on the five pages
with a genuine inputs-left / analysis-right split. The lesson worth keeping: **a
GRID parent sets the column track, so a child narrowing itself reclaims nothing** —
the arrow works, the panel goes, and the empty space stays. Each page states its own
collapsed track and the guardrail asserts it per page. Also in `v509`: the GL grid
hides four columns and clips Description, **hidden on screen but kept in the
export**.

**The CFO's `ITEM = 1` was refused, with figures.** `ITEM` is a LINE NUMBER, not a
debit/credit side — 13,493 distinct values across 79,074 rows. Filtering to it keeps
8.4% of rows and 5.5% of the money and takes the on-screen net from 10,797 to
2,102,385,065. Nothing is duplicated (0 duplicate rows on any key; all 8,809
open-period entries balance). Answered in full at `v513` — see §9.8 and the
`v513` note above.

**`v510` — the classifier read the FOLDER.** `classify_document` matched
`DOC_TYPE_PATTERNS` against the whole stored path, and every document sits under
`Tenant Leases/`, so the `lease` pattern matched the folder and short-circuited. 409
of 530 typed `Original Lease`; only 77 had "lease" in the file name. Because
extraction is gated on the type, 108 certificates of insurance were being sent to
the extraction API as leases and layered into tenant terms.

**And the obvious fix would have been worse than the bug.** Correcting the classifier
ALONE pushes 328 documents out of the `('Original Lease','Amendment')` gate and
strips the rent commencement date from 16 of the 38 tenants that have one — 15 of
those sources being Commencement Letters. So the gate widened to `is_term_bearing`,
with `NON_TERM_TYPES = {'COI'}` **measured** against all 500 extracted documents
rather than chosen.

**`v511` — everything after the base lease is layered in DATE order.** Found by
running the re-consolidation and diffing it. `order_lease_documents` returned
`originals + amendments + others`, applying every non-amendment AFTER every
amendment — invisible while `others` was nearly empty, and exposed the moment `v510`
put 147 documents in it. Three tenants moved the wrong way; Style Studio's expiry
went 2031 → 2026 because a 2021 Acceptance of Premises overwrote a 2026 First
Amendment.

**The re-consolidation then landed properly**: 70 of 70, 392 documents applied, 9
lease expirations and 5 rent commencements corrected, coverage unchanged on every
field, `lease_tenants.rent_commencement` populated 0 → 37, and a second pass moving
nothing at all.

**`v512` — a scanned lease is read from the PDF.** Jim: "is there anything we can do
to extract from the pdfs that produce no text extractions? I'm sure it will be a
common problem when scanning bulk files of pdfs." **219 of the 419 term-bearing
documents yield under 200 characters of text**, including 47 amendments and 28
original leases — their pages are images, so the prompt was being filled with an
empty string and the document contributed nothing, silently.

The API reads a PDF as images, so below `SCAN_TEXT_THRESHOLD` the PDF itself goes as
a document block. **No OCR stack — no Tesseract, no poppler, nothing new in the
image.** Jim: "I'm not price sensitive for this task. build it with the best model
suites for all scenarios", so both routes moved from a date-pinned Haiku to
`claude-opus-5`.

Verified on a real scan rather than a fixture: the 1987 Sam's Club short form lease,
5 pages, **4 characters of extractable text**, returned 10 populated fields
including 103,060 SF and six 5-year renewal options.

### Three things that would have broken quietly on the new model

Worth carrying to any other call site that moves to Opus:

- **`content[0].text` raises.** Thinking is ON by default, so the first block is a
  thinking block. Join the `type == "text"` blocks instead. The guardrail's stub
  returns a thinking block FIRST so a reintroduced indexed read fails.
- **A refusal returns HTTP 200 with no text.** Check `stop_reason` BEFORE reading
  content, and report it as a refusal rather than as a parse failure.
- **Base64 must carry no newlines.** `b64encode` adds none; `encodebytes` would.

### The method note, and it is the one to keep

**A check against a field that does not exist returns the same answer as a clean
bill of health.** Sizing how many stored consolidations were stale, I counted the
ones carrying a now-excluded document in `_documents_applied` and got **0** — which
reads as "nothing to do". The field was added in `v503` and is absent from all 70
records, so the comparison could only ever return zero. Counting how many rows
*have* the field first is one line and is what exposed it.

And: **the diff is what catches your own mistake.** "70 of 70, zero errors" was true
and useless; the before/after comparison is what surfaced the `v511` ordering
regression.

## Sep 19 2026 — A STATEMENT WITH NO ACCOUNT IS HELD, and the June load still has not happened

**`v507` = `b1d9197`, `v508` = `b75cf92`.** Open items: **`open_items.md` §7.9**.

**`v508` moved the seeding into the load** (Jim: "shouldn't the seeding process
be integrated into loading the statements function?"), after he hit the `0 of 49`
above. Filing a statement now opens that account's chain, so step 3 below is gone;
an account already carrying its balances forward is left alone and its statement
simply kept. It also LISTED the filed statements — the PDFs had been stored since
v507 and nothing linked them once a statement left the held list — and stopped the
accounts tab calling a seeded period "closed", which would have been wrong on all
49 rows the moment the June folder landed.

### Read this before touching treasury: production has ZERO statements

Checked on `v507` against the live database: **49 accounts, 716 activity rows,
0 statements, 0 periods, 0 matches.** Jim's recollection that "we uploaded all
the june 30, 2026 statements" is of the `v493`/`v494` work, where the 64 real
June PDFs were parsed **locally** to prove the parser — that is where "45 filed
before, 50 after" came from. Nothing was ever uploaded to the app.

So **Seed openings from statements** correctly returns `0 of 49 opened`, every
line saying *"No statement is on file for 202606"*. That is the seeder working:
it reads a filed statement and refuses to invent an opening balance. Jim hit
this on Sep 19 and read it as a failure; it is the feature.

The order, and it cannot be reordered:

1. **Import tab → the bulk card** → the 64 PDFs in
   `OneDrive/Documents/2026/06.2026`. 12 MB total, well under the 50 MB request
   cap and the 200-file limit. **49 file, 14 are held, 1 is refused** (a Wells
   Fargo statement sitting in the PNC folder).
2. **Answer the 14 held** — the new prompt on the Import tab. Each links its PDF,
   which is the only place the full number is printed.
3. Reconcile **July** — not June: the activity export opens 6/22, so June can
   never be reconciled; July and August are complete. (Seeding at `202607` is no
   longer a step; filing does it.)

### What `v507` built, and the part worth carrying

Jim: *"for the statements without a production account, I would like you to
create a record and prompt the user to find and input the account number for
future matching of the data pulls."* And: *"give the accountants the ability to
pull up a copy of the statement from the treasury screen."*

**An account registers itself only from an activity import, and PNC serves 90
days.** An account quiet longer than that has a statement showing real money and
no transaction anywhere to introduce it — a REGISTRATION gap, not a parse gap.
The parser reads all 14 correctly, `790-XXXXX47` included. The old behaviour
parsed them, said so in a result row, and **kept nothing**.

**The mask check is what makes the hold worth anything.** A typed number is
matched against the printed pattern — `XX-XXXX-7891` says ten digits ending 7891
— and refused if it does not fit. Without that, one transposed digit registers a
plausible new account, the statement files against it, and when the real account
later arrives under its true number the balance is split across two records with
nothing saying so. That failure is silent and permanent; the refusal is neither.

**Verified against a real PDF, not a fixture** — PPI Life Storage June: parses at
119,701.35, held, prompt shows it, re-import does not duplicate the question, the
PDF opens while pending, `9999999999` refused with nothing registered,
`8517897891` accepted, account created, statement filed with its balance and its
PDF, and a later statement for it routes by itself.

**All six `tr_*` tables joined `PROTECTED_TABLES`** on the `wp_fs_map` rule: the
app is the writer and holds the only copy. `tr_periods` is the reconciliation
CHAIN — each closed period's ending becomes the next one's opening — so losing it
loses the thread, not a report. Checked against the `isbs_uw_supplements` lesson
first (protection without a write path is a lockout); the guardrail asserts BOTH
the membership and the write path.

Guardrail: `scripts/treasury_pending_check.py` (34), which SKIPS with a reason
where the real PDF is absent so it still runs in the container.

## Sep 18-19 2026 — ONE NUMBER ONE ENGINE, the lease rent in force, and the CFO's query tool

**`v502` = `0d9eaee`, `v503` = `2bb9138`, `v504` = `93ce506`.**
Open items: **`open_items.md` §8** (three duplicates still open) and **§5.10**.

### The standing rule that came out of this, and why it is in CLAUDE.md

Jim, Sep 18 2026: *"We should not have conflicting calculation results. It will
cause doubt in the accuracy of the entire work. Make sure the vetted calculation
engines are used consistently and we do not have separate calculation engines
for the same number. The only differences in results should come from changes in
time frames or projections that we are running through the engines."*

He had said a version of this twice before and it kept recurring, so it is now a
standing rule in `CLAUDE.md` under **ONE NUMBER, ONE ENGINE**, with a table of
which engine owns which number and `scripts/one_engine_per_number_check.py`
enforcing it. **Before writing any calculation, find out whether the app already
answers it.**

### What the sweep actually found

**Accrued pref had two implementations in ONE FILE.** `_compute_accrued_pref`
(ROE Summary, Committee Summary) and `build_pref_balance_detail` (everything
else) walked the same ledger at the same rate and disagreed on **34 of the 68
deals both could price**, the ROE path **$633,807.54 low** in aggregate at
2025-12-31.

The cause: it accrued `cur -> 31 Dec`, compounded, then resumed at `1 Jan`, so
**31 Dec -> 1 Jan was never accrued**. One lost day per year end, always short,
worse the older the deal — P0000068 lost ~$102,000 over nine of them.

**It never looked wrong.** A slightly low accrual is still a plausible accrual.
That is the lesson worth carrying: *a second implementation is most dangerous
when it is nearly right*, because nothing on screen and nothing in the logs
distinguishes it from the answer.

**Which one was correct was settled by Jim's own figures**, not by which was
newer — the vetted walk reproduces P0000044 51,926.54 and P0000031 37,394.57
exactly as he stated them; the other gave 26,489.03 for P0000031. When you find
a duplicate, **measure both across every deal before changing either**, and say
which of his figures the candidate reproduces.

**A "temporary estimate" is a second engine.** The Committee tab's Net Proceeds
fell back to `value - debt` when no NAV had run — scaffolding from before the
NAV engine, left in after it shipped. Removed.

**A guardrail was pointed at the wrong engine** — it asserted "no grace period"
against the DELETED function's docstring while claiming to describe the one the
NAV uses. It would have kept passing while the real engine drifted. Now proven
behaviourally.

### Lease review: the rent in force is resolved, not guessed (`v503`)

New business via Jim, Sep 19. Two asks, and a worse defect underneath them.

* **Rent PSF is annual rent over SF.** A monthly rent is annualised before
  dividing. Same 12x shape as `v495`.
* **The most recent amendment governs.** Consolidation ordered by `doc_date`,
  and `parse_doc_date` only matched a date at the START of a filename — so
  "First/Second/Third/Fourth Amendment.pdf" had no dates and fell through to
  UPLOAD ORDER. Measured applying **4, 1, 3, 2**: the First Amendment's
  superseded rent overwrote the Fourth's. The ordinal was already being matched
  by `DOC_TYPE_PATTERNS` and thrown away.
* **"Months 1-12" is placed against the rent commencement date**, which now
  lives on `lease_tenants` instead of only inside `extraction_json`. Month 1
  begins ON commencement, so month N is the anniversary.
* **THE DEFECT NEITHER ASK NAMED:** when a step would not resolve, the
  validation picked the step whose annual rent was **closest to the rent roll's
  own figure**. The rent roll was checked against whichever lease number already
  agreed with it — **it could not report a mismatch**. A validation that always
  passes is worse than none, because it reads as confirmation.

### The FS mapping was empty, and the statements went with it (`v506`)

Jim: after a refresh the statements stopped appearing. **`wp_fs_map` was at 0 rows**
against 583 accounts and 79,074 GL rows. No mapping means every account is
`unmapped`, so every statement for every entity renders empty **while the API still
answers 200** — the logs showed 200 in 2,712 bytes, not a 500.

**I was wrong first.** I thought v505's wider balance query had hit a missing column.
The logs said 200 and all fourteen columns were there. Worth keeping the habit that
caught it: the log line discriminated between the two candidates in one glance,
because they fail differently (500 vs an empty 200).

**The cause was a destructive default**: `set_fs_map` deletes before inserting and
the endpoint passed `entries or []`, so a PUT carrying nothing wiped it and answered
`{status: ok}`. Now refused; `allow_empty` clears it deliberately.
`wp_fs_map` is in PROTECTED_TABLES — checked against the `isbs_uw_supplements`
lesson first: it HAS an app write path, so protection is a safeguard not a lockout.

**Restored and proved.** `fs_line_seed` really is the FS Tagging column of the PPI
Eastchase 06.30.2026 package — re-extracted from the file, 192 accounts and 56
captions, exact match. 553 rows restored; PPIECH and AMB6 balance with net income
-11,745.08 and -16,282.49, the same figures recorded at `v455`.

Open: 202 of the 553 are a fallback caption and nothing records which
(`open_items.md` §10).

### Statement drilldown on the workbench (`v505`)

Click any figure on a workbench statement, see the GL entries behind it. It does
not re-query — `_balances` already reads `gl_detail`, so the row selection moved
into one function both the builder and the drilldown call. **A drilldown that does
not reconcile makes a correct statement look wrong**, which is worse than none.

Two things the rendered line did not say, either of which would have made a correct
drilldown look broken: WHICH MEASURE it is (balance sheet `closing`, income
statement `ytd`) and THE PRESENTATION SIGN (a liability shown as 5,000 is -5,000 in
the GL). And the one balance-sheet line with no accounts of its own — the period
result carried into members' capital — would have opened an empty drawer; it now
carries the income accounts.

The refactor touches every statement, so it was proved behaviour-preserving before
the build: the pre-change module run side by side with the new one, 50 figures
across three period ends, zero differences.

### The CFO's GL / IA query tool (`v504`)

His workbook `GL & IA Queries with Filters - 09182026.xlsx`. **It does not
re-run his SQL** — `queries/MRI_GL_Detail.sql` already IS that query with the
&SPARM parameters stripped, so the job was putting the parameters back against
the tables we already import (Jim: *"since we are already pulling these tables
into our database we can have the query hit our tables"*).

Note the UI lesson: he asked whether he could select several entities and
accounts in the same query. **He already could** — a native `<select multiple>`
needs ctrl-click and nothing said so. Replaced with `MultiPicker.vue`. When
somebody asks whether a thing is possible, check whether it is already possible
and merely invisible.

---

## STILL OPEN — carry these forward

### Needs Jim's decision
1. **Capital balance: floored in one path, raw in the other** (`open_items.md`
   §8.1). Same accumulation; the ROE Summary shows the raw running total, the
   pref walk shows `max(0, running)`. **15 deals show a negative balance in one
   place and 0.00 in the other** — PWILLOW −8,044,374.08, POUTLOO −8,008,062.00,
   P3RDAVE −6,777,786.00 the largest. It arises because `realized gain` is
   accumulated as return of capital. Is below-zero a finding to surface, or is
   realized gain misfiled? Do not pick one silently.
2. **The MRI VPN password is committed in git**, hardcoded at
   `flask_app/services/mri_service.py:41` with no env fallback, and repeated in
   three `.claude/memory/` files. In the repo since `670902e`. **Jim rotates; I
   surface and remove afterwards.** Raised Sep 19 2026, not yet actioned.
3. **GL / IA Query access** — reads are open to any signed-in user, matching the
   rest of Accounting, but this is a bulk export of entity GL. One decorator
   narrows it to `ACCOUNTING_ROLES`.
4. **The IA date bound** — his sheet uses strictly-before; the tool's To date is
   inclusive and says so. Flip it if he wants his figures to tie exactly.

### Needs production data to settle
5. **Two debt implementations** (`open_items.md` §8.2) —
   `compute.get_isbs_debt_balance` vs `valuation_nav_service._bs_snapshot`
   summed over `DEBT_BS_ACCTS`. They also differ on child consolidation.
   **Unmeasurable locally**: `isbs_raw` is a 61-row stub here, so both return
   nothing for all 128 deals.
6. **Prior-year figures come from two sources** (`open_items.md` §8.3) — the
   Committee tab reads MRI's `valuations` feed, the summary tabs read the prior
   cycle's own records. Only one cycle exists locally.
7. **Lease amendment ordering coverage** — how many real amendments carry
   neither a date nor a number is unknown; there are no lease documents in local
   data and a production read was refused by a permission gate. That is the
   population where ordering is still best-effort, and it is reported per tenant.
8. **`gl_detail` may never have been imported on production.** `MRI_GL_Detail`
   is last in the refresh registry and its own description says "never yet
   executed" — unbounded GHIS on a 2GB container. The query tool says which
   query to run rather than showing an empty grid, but check before the CFO
   opens it expecting data.
9. **His workbook is truncated** — the IA query's third branch (non-cash
   transactions) is cut off mid-statement at 124 characters in row 49. Our
   import covers non-cash so the tool does, but something else may have been
   lost in his paste.

### Working practice that keeps paying
* **Measure before building, and measure the thing that would be wrong.** Every
  defect above was found by running against real artefacts, not fixtures.
* **Make a guardrail fail on purpose before trusting it.** Two checks this week
  passed vacuously: a fixture whose ids happened to ascend with the amendment
  ordinals, and a seam check whose window was wide enough to catch an unrelated
  mention. Both were caught by injecting the defect and confirming a failure.
* **Production reads are gated for me.** `az containerapp exec` was refused this
  session. Anything needing production measurement has to be asked for.

---

## Sep 17-18 2026 — TREASURY END TO END, and three bugs the real files found

**`v492` = `8fc4947`, `v493` = `cab414c`, `v494` = `d72c46b`.**
Topic file: **`treasury.md`**. Open items: **`open_items.md` §7**.

The whole chain is live: import the PNC activity export, reconcile it against
the statement and the ledger, code the month, download the GL and IA upload
files. Nothing posts to MRI — it produces two files a person uploads.

### The part worth carrying forward: real files found three bugs a spec could not

Every one of these was found by running the code against Jim's ACTUAL files
before he relied on it, and every one would have looked like somebody else's
problem:

1. **`.00`** — PNC prints a zero balance with no leading digit. 46 of his 64
   June statements were refused, including rows carrying real money, because
   one column held `.00`. It looked like "PNC layouts vary".
2. **The mask is not always a tail.** `XX-XXXX-5765` hides the front,
   `790-XXXXX55` hides the MIDDLE. Reading the last four visible digits off the
   second gives an account that exists nowhere, so five REGISTERED accounts were
   reported as unknown. **That is the dangerous shape** — it invites you to
   "fix" it by entering data that was never missing, creating duplicates.
3. **The vendored MRI template was never committed** — `.gitignore` blocks
   `*.xlsx`. Pre-flight P4 caught it; the guardrail had passed on an untracked
   file sitting in the working tree. **Present locally is not shipped.**

The pattern: assert against an artefact the real system ACCEPTED, not against a
description of one. `treasury_upload_check.py` rebuilds both accepted MRI upload
files from their own contents and demands byte-identical output.

### Two rules in the module that look like details and are not

**Seeding is not re-basing.** `opening_balance()` never reads a statement; it
carries the prior close, so a break surfaces as a difference instead of being
hidden. `seed_from_statement` exists only to START the chain and refuses once
anything is reconciled.

**The cash side is never typed.** Each bank transaction becomes its own GL cash
line at the amount the bank reported; the accountant supplies only the offset.
The entry balances BY CONSTRUCTION, and a partly coded month cannot produce a
file — no separate rule that could drift from it.

### What Jim needs to do next, in order

1. Import the 90-day activity export (50 accounts register themselves).
2. Drop the `06.2026` statement folder into the bulk card — 50 of 64 file.
3. **Seed openings at `202607`** — not 202606; the export opens 6/22 so June can
   never be reconciled, while July and August are complete.
4. Set the three CAD accounts to `MR10006000`, and give PPI2/PSS1/PIG5's second
   accounts their own cash accounts (§7.8).
5. Type the one missing account number for PPI Life Storage NY (§7.7).
6. Reconcile July.

### Watch out: that OneDrive folder is Files On-Demand

Reading the statement folder directly from the command line pulls each PDF from
the cloud — read times climbed 0.0s to 3.1s and then stalled. Parsing itself is
0.03s per file. Uploading through the browser is unaffected. If it needs reading
locally again, "Always keep on this device" first.

## Sep 17 2026 — TREASURY, and who owns the accounting section

**`v488` = `492fe04`, `v489` = `ce80ba5`, `v490` = `1496daa`.**
(`426633b`, the order-number gate, shipped in `v491` — see the entry above.)

Topic files: **`treasury.md`** (new — the module in full) and
`accounting_workpapers.md` (a new "Who may edit any of this" section).
Open items: **`open_items.md` §7**.

### Treasury is live at `/treasury`

Three tabs — accounts, import, reconciliation. Built from Jim's real August
AMB6 files, and the figures were measured before any code was written:

```
beginning (PNC statement)   571,750.04
net movement (PNC export)  -560,022.54
computed ending              11,727.50
ending (PNC statement)       11,727.50   ties
MRI's September opens at     11,727.50   carries forward
```

**`current_available` is `None` and the screen says why.** Available is ledger
less holds, float and pending debits — bank facts absent from any export. Do not
be tempted to fill that column; a guardrail asserts it stays empty. Current
ledger IS shown, carried from the last close, and says which period it came from.

The PNC API is **not** built — §7.1, blocked on Jim's banker, no screen-scraping.
GL/IA upload templates are Phase 3 — nothing here posts to MRI.

### The thing worth carrying forward: how the access bug was found

Jim asked for accounting to be editable only by accounting. Two discoveries:

1. **`role_required` compares LEVELS, and analyst/accountant/accounting_manager/
   cfo are all level 1.** No arrangement of role names could exclude analysts.
   Needed a membership check (`roles_exactly`). This is still true everywhere
   else in the app — §7.5.

2. **Six accounting writes had NO role check at all**, including exhibit
   DELETE and tracker sign-off, reachable by any signed-in user. They had
   survived a green guardrail because that guardrail **grepped for a
   decorator's text**. A string that is absent looks exactly like a rule that
   does not apply. Rewriting it to **enumerate routes from the Flask app** and
   call each as each role found them immediately — §7.4. Copy that pattern.

### Two gates now

| | Who |
|---|---|
| `ACCOUNTING_ROLES` — edit anything in the section | admin, cfo, accounting_manager, accountant |
| `CLOSE_PLAN_ROLES` — when the close opens, when things are due, what order | admin, cfo |
| read | everyone signed in |

Renumber and carry-forward sit in the narrower gate **because they write
`sort_order`** — gating the order cell alone would be defeated by a different
button. Fill properties does NOT, because it writes only the Property column.

**Assert every narrowing in BOTH directions.** A rule tested only in the
refusing direction is satisfied by locking everyone out — which here would stop
the close. The screen was wrong in exactly that way twice: `v480` hid controls
from the CFO whose writes the API would have taken, and its fix then locked out
the accountants.

### Recurring mechanical trap, wasted time three times today

Writing Python through a bash heredoc **collapses a backslash-n inside a string
literal into a real newline**, producing `print("` followed by an actual line
break — a syntax error, and only at parse time. Quoting the heredoc (`<<'EOF'`)
does not prevent it. Use the Write/Edit tools for any content containing escape
sequences.

This note is itself an example: the first attempt to write this paragraph
through a heredoc was mangled by the behaviour it describes.

## Sep 15 2026, evening — THREE CREDENTIALS, and the ownership tree

**`v466` = `be27c1a`.** Read `open_items.md` §3.14 before anything else.

**THREE PLAINTEXT CREDENTIALS SURFACED IN ONE DAY, all through the same gap.**
The pre-commit hook blocks `://user:secret@` URLs and not a bare
`NAME = "value"` assignment, which is how every one of them got in.

1. **The MRI SQL Server password** (§3.14) — `mri_service.py`, public on
   `origin/main` since May 5 2026. Read access to the source of record. **The
   urgent one.** Rotate, THEN move to a secret ref; the other order just
   relocates a compromised value.
2. **The SendGrid API key** (§3.12) — a plaintext env var on the container app,
   printed into a session transcript by my own `--query value`. Dead account,
   low practical risk, still a live credential.
3. **The wfadmin Postgres password** (§3.10) — Charlene reported the cleanup as
   incomplete; her specific finding was a FALSE POSITIVE (the literal
   placeholder `<password>`) but the real credential is in public history from
   April and whether the Sep 11 rotation covered it is still unanswered. The
   hash check that settles it without exposing anything is in §3.10.

**The MRI refresh was broken by one clause, not by the VPN.**
`Connection Timeout=30;` is an ADO/OLE DB keyword; ODBC Driver 18 rejects the
whole string with 08001 before touching the network. Every MRI query has failed
identically since May 5 whether or not the tunnel was up — and 08001 is the
same SQLSTATE a dead VPN produces, so it read as connectivity every time.
Diagnosed by varying one clause at a time against the real driver. Fixed in
`v463`; the VPN is separately down, so this removes one of two reasons.

**The ownership tree** (§3.13) went from showing nothing to working, across four
deploys. The defect that reached production was a rendered-string null test that
missed `pd.NA` — PostgreSQL's flavour, which no three-row SQLite fixture can
produce. `deploy_history.md` under v460 has the post-mortem and the three
lessons; the most transferable is that when you cannot reproduce, instrumenting
the running system beat three rounds of hypothesis.

**One shape repeated three times** and is worth recognising early next time: a
figure correct about the relationship it was computed from, shown in a context
asking a different question. Commitment dollars, balance dollars, and the
balance breakdown each looked right and read wrong.

**Two latent defects found by building on top of old code, both now fixed and
neither reported by anyone** — worth noting as a pattern, since both had been
live for months and produced plausible-looking output the whole time:

- `run_upstream_analysis` hardcoded `wf_type="CF_WF"` at BOTH levels, so a sale
  or refinancing ran the operating waterfall instead of the Capital one. Same
  dollar, different split, capital outstanding not reduced, and nothing on
  screen naming which had run. Fixed in `v467`; the type is now the caller's and
  an invalid value is refused rather than defaulted.
- `Connection Timeout=30` in the MRI connection string (above). Four months.

Both share a shape: a wrong value that the surrounding code accepts without
complaint, so the only symptom is output that looks reasonable. Neither a test
nor a reviewer would have caught them; both turned up because somebody built on
the code and had to read it.

**Open and unanswered**: the $1,347,797 on 30BEAR/PPI27 (§3.15), whether "every
sold deal" means 4 or 27 (`v459` note in CLAUDE.md), and Charlene's three
guardrails that import an uncommitted `live_api` and therefore run for nobody.

---


## Sep 15 2026 — email provider, app roles, ownership. LIVE at `v457`.

  - `v457` = `9db5923`  ownership: the CURRENT commitment, not the sum of open ones
  - `v456` = `b00ed5d`  roles, ACS email, ownership rebuilt (seven commits — see below)

  - `9db5923`  ownership: current commitment = latest open StartDate, one row
  - `b00ed5d`  ownership: upstream analysis restored as a second tab
  - `01777cf`  ownership: the commitment chain above each PE investment
  - `07fd035`  memory
  - `b72ea9b`  send through Azure Communication Services when configured
  - `1e0ebad`  CFO / accounting manager / accountant roles + a real privilege hierarchy

**`v456` shipped seven commits when two were asked for** — the live image was five
behind `origin/main`. Pre-flight P2 caught it *before* the build, which is the whole
reason that step exists; the five were reviewed and the two docs commits verified to
touch no runtime file. The `v429` post-mortem said this would happen again, and it did.

**THE OWNERSHIP DEFECT IS THE LESSON FROM THIS SESSION.** `01777cf` derived each
owner's share by summing every commitment row that had not yet ended. The current
commitment is **one row** — the latest `StartDate` with no `EndDate` — because MRI does
not reliably close the superseded row, so several rows for the same pair sit open at
once. Summing inflated the amended owner AND understated every other owner at the
level, since each share is that owner's amount over the level total. **The level still
summed to 100%, so nothing looked wrong.** Jim found it by reading the deployed tree
against MRI; no check would have.

Two things worth carrying forward from how it was fixed:

- **My first regression test could not fail against the bug.** I closed the superseded
  rows with past EndDates, which the broken filter already removed, so both scenarios
  passed against the broken code. Always run a new guardrail against the commit before
  the fix and confirm it FAILS — `scripts/ownership_commitment_currency_check.py` does,
  on the all-open-rows and future-EndDate cases.
- **The local database cannot produce this shape at all** — three commitment rows, all
  open, one per pair. Everything in `01777cf` was "verified" against that. A fixture that
  cannot express the defect is not coverage.

**Roles.** `role_required()` matched role strings exactly while the comment above
`ROLES` claimed a hierarchy. That was harmless for viewer/analyst/admin — for those
three, exact matching and a hierarchy agree — and would have stopped being harmless
the moment a fourth role existed, since 104 endpoints name only `admin`/`analyst`.
The three accounting roles sit at **analyst** level (Jim's call: every analytical and
workpaper screen, but no user management, MRI refresh or CSV import). `WP_ROLE_FOR_LOGIN`
lets a `cfo` login approve as CFO without a duplicate `wp_roles` row. Guardrail
`scripts/role_hierarchy_check.py` fails 6 assertions against the previous commit.

**Worth remembering from that change**: the first version computed its threshold with
`min(role_level(r) for r in allowed_roles)`, and `role_level()` returns 0 for unknown
names — so a decorator typo (`role_required("Admin")`) would have dropped the bar to 0
and admitted **a viewer to an admin-only endpoint**. The exact-match code it replaced
failed *closed* on that same typo. Verified by simulation, not reasoning, before the fix.
A hierarchy that turns a harmless typo into a silent auth bypass is worse than the
problem it solves — the guardrail now covers it.

**Email.** See `open_items.md` §3.12. Decision made (ACS, not Resend/Brevo), code
committed, **provisioning not started**. Runbook, including the four DNS records and
why they must go on a subdomain rather than the root:
<https://claude.ai/artifact/MZd8VHR5zgAFtue9yLKBDA>

**Charlene's credential report was a false positive** — `azure-complete-setup.sh:40` is
the literal placeholder `<password>`, not a credential. `open_items.md` §3.10 has the
proof and, more usefully, the part that IS still open: whether the Sep 11 rotation
actually changed the `wfadmin` password, with a hash check Jim can run without exposing
the value. **Do not re-raise the setup-script finding.**

**Not verified**: nothing in either commit has been exercised on Azure. The roles change
in particular wants a real login per new role before anyone is assigned one — Jim has
admin access and offered; it was not used this session.


## Latest: `v451`–`v455` (Sep 14–15 2026) — ACCOUNTING WORKPAPERS

A new section of the app, built across two sessions. **Read
`accounting_workpapers.md` before touching any of it**; `open_items.md` §6 has what
is still open.

  - `v455` = `841e92b`  a pre-close step may be due before period end
  - `v454` = `c0a53f2`  refuse a close deadline that cannot be true
  - `v453` = `5274e83`  statement line order + dormant-line suppression
  - `v452` = `5323de3`  flag a statement line facing the wrong way
  - `v451` = `041827b`  say why an email failed, and that the account works anyway

**What exists now.** Sidebar → Accounting → Workpaper Packages. A close cycle
generates one package per entity tagged `ENTGRPID='REP'` in MRI. Each package
carries five drafted statements with their tie-outs on screen, a 12-step
checklist with CFO deadlines, step-scoped exhibit upload, an approval chain
(accountant → manager → CFO) and a 17-tab download with the exhibits placed
inside it.

**The design decision that matters most**: the statements are ONE engine
(`statement_service.py`) serving any entity, and the package is a caller rather
than the owner. A figure in a downloaded workbook cannot differ from the one an
auditor is shown elsewhere, because there is no second implementation.

**Three things to pick up, in order:**

1. **MR22000002 — with accounting, unanswered** (`open_items.md` §6.1). The
   example package tags an account named "Other Liabilities" to the asset line
   "Due from Manager". Zero for PPIECH, so harmless in the specimen; on AMB6 it
   puts **-629,125.04** into assets. The balance sheet still ties out — a
   negative asset and a positive liability net identically — so only
   `sign_anomalies` catches it. **Fix `ACCOUNT_LINE` in `fs_line_seed.py` when
   they answer, not the statement output.**

2. **Nothing is set up in production** (`open_items.md` §6.2). No close cycle,
   no `wp_roles` assignments, no step owners, no deadlines. The feature is
   deployed and idle. Needs a CFO session. `MC_TYPENAME_ROW` (members' capital
   row routing) also wants accounting's eye before the first real package.

3. **SendGrid's free plan lapsed** — **RESOLVED Sep 15 2026.** Moved to Azure
   Communication Services; `v458` sends as `noreply@notify.peaceablestreet.com`
   and a real email was delivered. What remains is not configuration: the
   message was **junked by Avanan/Check Point**, the security gateway in front
   of the tenant, and Exchange deferred to that verdict. `open_items.md` §3.12
   has the header evidence and the two asks with IT. Also still open there: the
   SendGrid API key was a plaintext env var and needs revoking.

**Verified how**: both guardrails run clean
(`scripts/statement_presentation_check.py`,
`scripts/workpaper_deadline_check.py`, each of which fails against the commit
before it), the deployed frontend chunk was fetched from production and
confirmed to carry the change, and the tab was driven in the browser locally.
**Not verified**: the rendered statements on the Azure instance — that needs a
login.

**Two things the browser found that unit checks could not**, worth remembering
as a method rather than as facts:

- A refused deadline was **silent**. `setDue` set `error`, then called
  `loadTracker()`, which clears `error` on entry — so the field snapped back
  with no explanation. Every unit check passed throughout.
- The local statement fixture had **one line per section and no zero lines**, so
  neither new presentation rule was visible until four fixture accounts were
  seeded (and removed again). A green guardrail was not the same as having
  looked.

---

## Previously: v440 = `0ad313a` (Sep 11 2026, evening)
The valuation section's budget work. **Read `open_items.md` §5 for the full picture** —
asset management's six comments and what shipped against each.

- **One line-mapping screen for both spreadsheet sources** — the partner's budget
  workbook and the appraiser's Argus download, under Budget Review. Argus arrives
  pre-filled from the keyword rules and every guess is tagged as one; that mapping had
  been applied silently at import since it was written, which was AM's complaint.
- **Modeled debt service in the Budget and Valuation columns.** An Argus download is
  unlevered, so the Valuation DSCR was blank. Interest → **5190** (not the AM forecast's
  7030 — see §5.8), balloons excluded, Estimate column untouched.
- **`isbs_budget_is_supplements` is protected; the other four supplements are NOT.**
  Protect what the app writes. Protecting `isbs_uw_supplements`, which has no app write
  path, froze its 56 rows instead of protecting them — caught in the deploy pre-flight
  for this revision, before the image was built.

**Two things to pick up:**
1. **§3.10 — the Azure app admin password was committed in plaintext** from Jul 13 to
   Sep 11 2026 in `MEMORY.md`, the file every session reads first. Removed from the tree,
   still in git history. **Jim: rotate it.** Second credential exposure in as many weeks.
2. **§3.11 — `isbs_budget_is_supplements` has never been created on PostgreSQL.** The
   first partner budget imported on Azure creates it. Same shape as the `v435` defect;
   worth one small import before a cycle depends on it.

---

## Previously: v429

Rolling handoff for the next session/developer. Update in place; keep only what is still
live. Per-revision post-mortems live in `.claude/memory/deploy_history.md` (CLAUDE.md keeps
the deploy rule and a one-line SHA index).

**Supersedes the Sep 2 / v416 handoff, now archived at `session_handoff_sep2.md`.** That
file still holds TRACK 1 (the KOC slice / investor groups around a deal), TRACK 2
(Charlene's stream through v416) and TRACK 3 (the Sep 2 engine corrections). **None of
those was touched on Sep 10 and their state is unchanged — read that file for them.** The
durable defect list is carried forward here so it does not get lost behind an archive.

## Where things stand
- **Live**: `v429` = `33a4bf5`, deployed Sep 11 2026, healthy, 100% traffic, HTTP 200.
- **main == origin/main**, everything pushed.
- **THE THREE ONE PAGER PRINT COMMITS ARE NOW LIVE.** They shipped in `v429` as ancestors
  of the requested SHA, not because they were deployed deliberately: `e5699c6`,
  `62161a9`, `948be26`. Also live now: `608ca8b` (the read-only ownership reconciliation
  script) and `29a1463` (One Pager em dash + Pref Equity capitalization print fix).
  **The Sep 10 handoff said these three "need the standing symptom-repair review before
  anyone builds an image". That review did not happen** — the pre-build review covered
  only `33a4bf5` and `29a1463`, the span against local HEAD. They are live and unreviewed,
  and they change what prints on an investor document. **Not spot-checked on Azure** —
  they were verified locally only.
- **The delta that matters is against the RUNNING IMAGE, not local HEAD.** `v429` was
  asked for as "deploy 33a4bf5" and shipped seven commits, because local main was two
  behind origin and the live image was five behind that. Run
  `git log <live-sha>..<target>` before every build and review the whole span; the
  post-mortem in `deploy_history.md` records how this one was under-reported at the time.
- **OPEN QUESTION FOR JIM (live, unanswered)**: `33a4bf5` moved the Portfolio Totals
  "% of Pref" 75.53% -> 76.39% at 26Q2. Fund-group subtotals now tie to the 26Q1
  baseline PDF exactly; the grand total deliberately does not, because the PDF computes
  that one row on a basis it uses nowhere else. If the published total must read 68%,
  that is a two-line exception still to be made.

## STANDING RULE — read before deploying anything
CLAUDE.md "Deploying Changes" carries Jim's pre-deploy symptom-repair check.
**Verifying that a commit does what its message says is NOT verifying its premise.** Flag a
symptom repair to Jim, with affected deals and figures, BEFORE building the image.

## The habit that paid on Sep 10
Every headline figure in this session was **measured against the real function on real
data** before it was reported, and three of the first four conclusions were wrong. The
pattern that caught them: state the claim, then try to reproduce it from the database. See
"Corrections made mid-session" below — they are recorded because each one was reported to
Jim confidently first.

---

# TRACK A — Brainerd / TIAA look-through (THE OPEN ITEM)

**Status: root cause found and proved. Data fix NOT made. Nothing deployed.**

## The finding
TIAA's Total Commitment on Brainerd Place Apartments is understated because the ownership
graph is missing one edge. The legal org chart (`PPI Brainerd (CT) LLC - Org Chart w TIAA
24.11.21 Transfer PSC Investee Brainerd (CT).pdf`, in the deal's Legal/Org Chart folder)
records a **11/21/2024 transfer of 53.975% of PSC Investee Brainerd (CT) LLC [INVBPA] from
Peaceable Street Capital to PSC TGA 2022 LLC [TGA22]**. It was never recorded in MRI.

Because `relationships` has INVBPA as PSC1 100%, TGAM's third route to PPIBPA does not
exist:

| Route | Today | Corrected |
|---|---|---|
| TGA22 → PPIBPA | 44.4513% | 44.9100% |
| TGA22 → INVBPS → PPIBPA | 18.6907% | 11.2274% |
| TGA22 → **INVBPA** → INVBPS → PPIBPA | **missing** | **18.2773%** |
| **TIAA total** | **63.1420%** | **74.4147%** |

Jim's independent figure was 74.41%. Chart cross-checks tie to four decimals: TGA22 58.435%
vs the chart's 58.434% Borrower, TIAA 52.591% vs 52.591%.

**Total Commitment $11,622,976 → $13,698,026** on the funded-pref basis.

## THE ENGINE IS CORRECT — do not "fix" the walk
`lookthrough_pct` in `portfolio_snapshot_service.py` already sums every distinct route; it
returned 2 routes and was fed an incomplete graph. Patch the three hops into the
relationships frame and the unmodified function returns **74.4147%** via 3 routes. This was
verified, not assumed.

## The data fix, not yet made
Must land **in MRI** — `relationships` is MRI-refreshed, so a direct DB edit is overwritten.

| Entity | Current | Org chart |
|---|---|---|
| INVBPA | PSC1 100% | PSC1 46.025% / **TGA22 53.975%** |
| INVBPS | INVBPA 58.9654 / TGA22 41.0345 | INVBPA 75.10 / TGA22 24.90 |
| PPIBPA | INVBPS 50.6097 / TGA22 49.3903 | INVBPS 50.10 / TGA22 49.90 |

Blast radius is contained: INVBPA and INVBPS reach only Brainerd and its nine child
buildings, which the report already excludes.

## Why nobody caught it, and why the obvious detectors do not work
**TGAM 63.142% + PSC1 36.858% = exactly 100%.** A transfer moves ownership *between*
owners, so the total is preserved and the books balance while 11.27 points sit with the
wrong party. Both candidate detectors were tested and neither finds it:
- **A conservation check passes today.** That is why the reconciliation report deliberately
  ships none — including one would imply a guarantee it cannot give.
- **The commitments cross-check does not flag INVBPA**: both feeds say PSC1 100%, both
  predate the transfer. **Agreement between the feeds is not evidence of correctness.**

The only source that knows is the legal org chart. Proposed durable fix (designed, not
built): a protected `ownership_attestations` table — deal, investor, attested %, as-of
date, source document — with the Snapshot flagging any deal whose computed look-through
differs beyond a tolerance, and rows auto-retiring once the feed agrees so it cannot ossify
the way `MANUAL_RATIO_SEEDS` has.

## Second, independent understatement on the same deal
Brainerd's **Total Pref is funded pref, not committed**. The family has zero accounting
`is_commitment` rows, so `resolve_committed_pref` falls back to funded ($18,407,677.40) and
labels it `funded (no commitment row)`. The `commitments` table shows PPIBPA at
**$31,721,927.29** across two generations — which matches the org chart's figure to the
cent, confirming those generations are **additive, not superseding**. Fixing the percentage
alone leaves this understated. `commitments_raw` is loaded at `data_service.py:586` and
**read by nothing**.

---

# TRACK B — Ownership reconciliation report (`608ca8b`, shipped, not deployed)

`scripts/ownership_reconciliation.py` — read-only, touches no engine path. Self-test 15/15
via `--selftest`.

```
.venv/Scripts/python.exe scripts/ownership_reconciliation.py
```

**Current queue**: 30 of 200 entities disagree between `relationships` and `commitments`;
20 carry economics, governing **$424m** of funded pref (TGA23 $125m, OWPSC $108m, TGA24
$100m). Separately, **38 rows / $69.7m of commitment money funded against a 0% or absent
ownership row** — that check needs only ONE feed to contradict itself, so it is firmer
evidence than a split disagreement and is the better place to start.

**Both feeds are stale, in opposite directions** — which is why the report names no
authority. Brainerd: `commitments` matches the chart at INVBPS, `relationships` does not.
OWPSC: the reverse — `relationships` correctly ends BPH's 48.9688% on 2025-12-31 and starts
WOFC on 2026-01-01, while `commitments` still names BPH.

## Corrections made mid-session — read before re-deriving any of this
1. **TGA23/TGA24 are NOT a TIAA overstatement.** Reported as one; retracted. Both feeds
   state TGAM at 90% and the money agrees: `TGAM/(TGAM+INV23)` and `TGAM/(TGAM+AMB24)` are
   each **exactly 90.000000%**. The apparent dilution was a second-closing sleeve
   (`INV23-P`, $1,475,409 at a stated 0%) and a member recorded one level up (`AMB24`).
   *Still genuinely open, and small*: is INV23-P's $1.47m matched by a TGAM increment? If
   not TGAM really is 88.95% on TGA23. Two of three signals say 90%.
2. **`relationships.Name` names the INVESTMENT, not the investor.** Every row of TGA23
   carries "PSC TGA 2023 LLC". Reading it as an investor name made INV23/INV23-P look like
   one legal entity; a merge built on that collapsed four distinct OWPSC members into one.
   Removed. The self-test pins the column's meaning.
3. **Reachability is not ownership.** PSCMAN ranked first at $472m through a **0% edge**
   into TGA22. Exposure is now weighted by the entity's own look-through and it scores zero.
4. **32 apparent "missing owners" are the same member a level up** — the whole
   DCXVIA/DCXVIB family sits under PSC3. Resolved by a reachability walk, not reported as
   breaks.
5. **Deal entities are not holding vehicles** — `relationships` carries the PE vehicle at
   100% while `commitments` includes the OP. Reported separately, not dropped.

**Checked and clean**: no active (investment, investor) pair carries more than one row, so
the engine's graph — which appends every row as an edge — cannot double-count. The 0.0000%
rows visible on OWPSC are ended generations.

---

# TRACK C — Waterfall Setup (all shipped and live in v426–v428)

Three defects, all found from one report by Jim that copying a waterfall "said it copied"
and produced nothing.

1. **`bf093c2` (v427) — "Copy from deal" copied nothing on 68 of 92 deals.** A blank
   `nPercent` reaches `json.dumps` as a bare `NaN`, which is not valid JSON. **Axios does
   not reject that**: with default `silentJSONParsing` a body that fails to parse comes back
   as the raw STRING, so `res.data.cf_wf` is undefined, the store writes `[]`, and nothing
   throws. **This shipped twice** — `e6858b5` fixed it in `get_waterfall_steps` by writing
   the scrub INLINE, so the two copy paths kept the unscrubbed line. Now one definition,
   `steps_to_records`. Guardrail `waterfall_copy_json_check.py` 12/12.
2. **`02023d1` (v428) — the UI announces what it actually got.** `copyFromDeal` awaited and
   then reported success without looking at the result, which is what made the above
   *silent*. One guard, `readStepsPayload`, on all three step-loading paths; both handlers
   now print step counts.
3. **`dfc38df` (v428) — an active deal is selectable before it has a waterfall.** The entity
   list was `rel_vcodes | wf_vcodes`, so a deal with neither could not be selected — and a
   deal cannot be given its FIRST waterfall without being selectable. Jefferson Stephens
   (P0000114) sat in that gap. 12 deals became reachable; verified purely additive by
   diffing the payload (254 → 266, nothing lost, `has_wf` identical at 92). Guardrail
   `waterfall_entity_nav_check.py` 16/16.

**Not browser-verified** — dev servers cannot be started from a session the harness flags
unattended. A minute on live closes it: open Waterfall Setup, confirm Jefferson Stephens is
listed, copy Eastchase into it, expect 4 CF / 8 Cap rows **with those counts in the
message**.

Also live: **`89c39a3` (v426) — `capital_calls` joined PROTECTED_TABLES** at Jim's
instruction; the CSV import's `to_sql(if_exists="replace")` was dropping every hand-typed
call. Capital calls are app-entered only now. Known consequence: 5,127 blank-Vcode rows can
no longer be cleaned from the UI (harmless to every computation via `load_capital_calls`'
dropna, but permanent).

---

## Known defects / debt worth carrying forward
Carried from the Sep 2 handoff; all still true unless marked.

- **One day of pref is dropped per investor per year** — `accrue_to_date` skips
  31 Dec → 1 Jan when it splits at the year boundary. See TRACK 3 in `session_handoff_sep2.md`.
- **`cap_stack.pref_equity` is capital OUTSTANDING** while three columns call it
  funded/invested/committed. Dormant; bites the first live deal with a partial ROC.
- **`commitments_raw` is loaded and read by nothing** (`data_service.py:586`, payload line
  682). 542 rows of real commitment data unused. NEW Sep 10.
- **`relationships.Name` is the investment's name, not the investor's.** NEW Sep 10.
- **The `deals.Sale_Date` column is not what the model uses.** Priority is sale override →
  `event_dates` projected disposition → horizon/max maturity. The local snapshot has no
  `event_dates` table, so a local run will NOT reproduce a live sale date.
- **The local `waterfall.db` accounting feed ends 2026-06-02.** Anything turning on a later
  event cannot be seen locally and must be simulated. Say so; do not conclude "no defect"
  from a snapshot that predates the data.
- **P0000116–P0000120 and P0000114 are absent from the local snapshot**, so anything about
  recent acquisitions must be proved against injected rows at their real dates.
- **`save_waterfall_steps()` writes NULL into `dteffective`** — restore it after any
  programmatic save; it is required by `loaders.load_waterfalls`.
- **`accounting_feed.sql` has no `TRIM()`** — 3,641 of 12,827 rows carry untrimmed IDs.
- **`accounting_feed.sql` LEFT JOIN has `AND S.MajorType = ...` in the ON clause** — an
  unclassified row survives with NULL MajorType and vanishes silently from every consumer.
- **The Flask dev server drops connections on heavy computes** (Portfolio Analysis, PSCKOC).
  Measure in-process; `scratchpad/blast_inproc.py` is the pattern.
- **`Investment_Strategy` is 0 of 134 populated** on live, so dev classification runs
  entirely off the `Lifecycle` proxy.
- **Dev servers cannot be started from an unattended session** (a scheduled-task run). Vue
  changes then reach only function/payload level, never the screen. Say so explicitly.
- **Three per-deal hardcodes remain on the Portfolio Snapshot** — `MANUAL_RATIO_SEEDS` (6
  deals), `PROJECTED_YE_NOI_FALLBACK` (Giant 7), `TEMP_OPERATING_SUPPRESS` (Hanestowne, who
  carries one on two subtabs). Tracked by the weekday `retire-manual-ratio-seeds` task.
  `scripts/live_api.py` is still uncommitted, so the guardrail behind the seeds cannot be
  reproduced by anyone but Charlene.

## Suggested next steps, in order
1. **Review and deploy Charlene's three One Pager print commits** — they are investor-facing
   and sitting undeployed on main.
2. **The Brainerd MRI correction** (TRACK A) — the one item with a known-right answer.
3. **Work the $69.7m "capital with no ownership" list** before the $424m split queue; it is
   firmer evidence and a shorter list.
4. **Spot-check Waterfall Setup on live** (TRACK C) — one minute, closes the only
   unverified part of v427/v428.
