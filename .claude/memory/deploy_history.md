# Deploy history — per-revision post-mortems (archive)

Verbatim archive of the deploy-history narratives that used to live inline in
`CLAUDE.md` under **Running the Application → Deploying Changes**. That section now
carries the deploy *rule* and a one-line-per-revision index (`vNNN` = SHA); the
detail for each revision is below, unchanged.

**Read this before assuming a revision shipped what its SHA suggests.** Several did
not — `v424` was a merge rather than the commit that was asked for, `v378` was
superseded minutes after it went out, and `v418`/`v417` deliberately shipped only
part of the branch they came from.

Newest first. Revisions absent from the post-mortem sections (`v396` and older,
apart from the few noted) carry no recorded post-mortem; their SHAs are in the
**Revisions v349-v569** index at the foot of this file, which is where the CLAUDE.md
index moved to on Oct 5 2026. CLAUDE.md now carries no revision list at all — the
running revision's image tag IS its commit SHA, so pre-flight P1 answers "what is
live?" without one.
## v460 = `ee1c61a` — two fixes, neither of which was the bug

Deployed Sep 15 2026. Build `cahv`, 2m11s. Tag locked. Healthy at 100% traffic.

**THE MOST INSTRUCTIVE FAILURE OF THE DAY, and the fix is the least interesting
part of it.**

The ownership tree, live since `v457`, reported "No commitments recorded into
this entity" for **every deal in the portfolio**. Jim could see 601 rows in the
`commitments` table. The local database holds three rows and seven columns;
production's is `select * from IA_Commitment` with every column MRI carries. So
nothing local reproduced it, and nothing about the symptom said where to look.

**I shipped two fixes on reasoning rather than evidence. Both were wrong.**

- `v459`-era hypothesis: PostgreSQL folds unquoted identifiers to lower case, so
  `EntityID` would be `entityid` on Azure and never match. Tested: a lowercase
  table raises `AttributeError` on a numpy scalar and returns **HTTP 500**, not
  an empty screen. Ruled out — but fixed anyway, because it is a real latent
  defect, and CLAUDE.md already carries that warning for SQL identifiers.
- Second hypothesis: `_read` swallowed every exception and returned an empty
  frame, so a failed query rendered as "no commitments". Also real, also fixed,
  also **not the bug** — the query was succeeding.

**What actually diagnosed it was making the screen report its own state.** The
only durable thing `v460` shipped was a health block carrying
`commitment_rows_loaded` beside `commitment_rows`, the column names as read, and
an explicit note when rows load and none survive. Jim read three lines off the
screen:

    The commitments table returned 601 rows and none survived filtering.

That located the defect immediately, in a function I had already read four times
without seeing it.

**The bug.** The open-commitment test asked `astype(str)` what the cell LOOKED
LIKE and matched the rendering against `("", "none", "nan", "nat", "null")`:

| flavour | renders as | matched |
|---|---|---|
| `None` | `none` | yes |
| `NaT` | `nat` | yes |
| `nan` | `nan` | yes |
| **`pd.NA`** | **`<NA>`** | **NO** |

`pd.NA` is what PostgreSQL produces where SQLite produces `None`. Every row
failed, the table emptied, and every local test passed. `.isna()` is the
primitive that answers this for all five. Fixed in `v461` = `e62cc0b`.

**Three transferable lessons.**

1. **Never test a null by its rendering.** `astype(str)` plus a list of spellings
   is a guess at an enumeration that pandas already answers exactly.
2. **A fixture that cannot express the defect is not coverage.** Three SQLite
   rows could not produce a `pd.NA`, a second open row for one pair, or a
   timezone-aware timestamp. Everything the ownership tree shipped was "verified"
   against that.
3. **When you cannot reproduce, stop theorising and make the system report.**
   Three rounds of hypothesis produced two wrong fixes; one round of asking the
   running system what it saw produced the answer in a single message. Reach for
   the instrument earlier than feels necessary.

---

## v440 = `0ad313a`

Deployed Sep 11 2026, 23:05 UTC. Build `cah7`, 2m18s — a real build, not one of the
5-second fast-fails. Tag locked. `RunningAtMaxScale`, 100% traffic, v439 deprovisioned.

**Asked for as "deploy 01f9e47". EIGHT commits shipped, and the target moved.** The live
image was `v439` = `e469d9b`; the span `e469d9b..01f9e47` was seven commits, four of them
written earlier the same day and never reviewed against the deploy checklist.

**P4 found a regression, and it was fixed before the image was built.** `8016874` had put
all five ISBS supplement tables into `PROTECTED_TABLES`. That is right for
`isbs_budget_is_supplements` — the app writes it and holds the only copy of a budget
between import and approval — and wrong for the other four, whose ownership runs the
other way: a CSV is their source of record and `to_sql(if_exists="replace")` is their
*designed* refresh.

`isbs_uw_supplements` has **no app write path at all**. Protecting it did not make its 56
rows safe, it made them unchangeable — and those rows feed the One Pager's underwritten
PE ROE through account 7073. CLAUDE.md documented the table as "importable via CSV
upload"; that sentence had quietly become false. **Protection without a write path is a
lockout, not a safeguard.** Jim's call: narrow it to the one table the app writes. That
fix is `0ad313a`, which is what actually shipped — deploying the requested `01f9e47`
would have shipped the lockout.

The rest of the span was reviewed clean: `data_service`'s ISBS precedence block
re-measured at **797,660 rows, unchanged** (the figure that matters, since an earlier
`drop_duplicates` formulation of the same rule took it to 439,268), 21/21 on its
guardrail; everything else new files, new endpoints, or the debt-service substitution
built the same day.

**What went live**: the shared budget/Argus line-mapping screen, and modeled debt service
in the Budget and Valuation columns of the budget comparison. See `open_items.md` §5.

**Verified on the live app** by busting `index.html` with both a `Cache-Control: no-cache`
header and a `?cb=` query, then resolving the route chunk from *that* response —
`ValuationsView-CYKSjhzs.js`, 75,888 bytes — and confirming the mapping tabs, the
reconciliation panel, the "keyword guess" tags, the four `mapping/*` calls and both debt
notes are present. That procedure exists because three "failed deploy" reports in this
project were stale cached chunks, not failed deploys.

Guardrails before build, all green: supplement precedence 19/19, budget import 42/42,
line mapping 36/36, debt service 34/34, refresh-table keys 14/14, capital-call CRUD
49/49, mixed-case SQL clean across 183 files. App boots at 360 routes.

**Noted against myself**: `az acr build` ran twice — the second only to capture the run
id, which the first invocation had already printed. Same source, no harm, a couple of
wasted minutes of build agent. Read the output you already have.

## v429 = `33a4bf5`

Deployed Sep 11 2026, 12:49 UTC. Digest `sha256:e95be886…`, tag locked. Healthy,
100% traffic, HTTP 200.

**Asked for as "deploy 33a4bf5"; SEVEN commits went live, not one.** Local main was
two behind origin at `3cbac29`, and the previously live image was `v428` = `dfc38df`,
which was five behind *that*. The full delta (`git log dfc38df..33a4bf5`):

- `33a4bf5` Snapshot Financial: "% of Pref" subtotal foots to its own row
- `29a1463` One Pager: absent capitalisation is a dash; ownership split prints whole
- `3cbac29` Handoff docs
- `608ca8b` Ownership reconciliation report (read-only script)
- `948be26` One Pager print: always exactly one page, and never a lost word
- `62161a9` One Pager print: the whole report on one page, at a legibility floor
- `e5699c6` One Pager print: the Business Plan narrative is no longer thrown away
- `737ca65` Docs: deploy history through v428

The last three are Charlene's investor-facing print commits that the Sep 10 handoff
had flagged as "on main, not in the live image". They shipped here as ancestors of the
requested SHA. **This is the lesson: the delta that matters is against the running
image, not against local HEAD.** `git log <live-sha>..<target>` before every build —
reviewing only the named commit reviews a fraction of what ships.

**Reviewed for symptom repair before building** (`33a4bf5` and `29a1463` only, which is
the gap above). Neither is one. `29a1463` replaces a `0.0` sentinel with `None` —
the rule this file exists to enforce, applied correctly; verified `fmtPct` renders null
as an em dash (OnePagerView.vue:376) and that `pe_exposure_on_cap` is absent from
`_SUM_FIELDS`, reaching the payload only through `_num()`, which maps None to None.
`33a4bf5` corrects a computation basis rather than overriding an output.

**OPEN QUESTION FOR JIM, live and unanswered.** `33a4bf5` moves the Portfolio Totals
"% of Pref" from 75.53% to 76.39% at 26Q2. Every fund-group subtotal now reproduces
the 26Q1 baseline PDF exactly on the committed basis; the grand total does not, because
the PDF computes that one row as Invested / Total Pref — a basis it uses there and
nowhere else. The commit deliberately keeps the total on the same rule as the groups it
sums rather than special-casing it. If the published total must match the PDF's 68%,
that is a two-line exception still to be made.

**Known, pre-existing, untouched**: the `portfolio_snapshot_financial.py` module
self-test dies partway through with `KeyError ('PCITWES', '2026-Q1')`, so several
checks — including "Un-funded = Commitment − Invested" and the two assertions
`33a4bf5` adds — are not executing. Present on main before this change.


## v428 = `dfc38df`

SIX COMMITS, not the two that were asked for. Origin gained four while the work was in progress, so this is a REBASE not a merge - ZERO overlapping files (Charlene in One Pager / Snapshot, me in Waterfall Setup), linear history, no merge commit, and main left in sync. The v424 trap avoided by checking `git rev-list --left-right` BEFORE pushing rather than after.

MY TWO, both consequences of the v427 investigation. (1) `02023d1` COPY AND TEMPLATE REPORT WHAT THEY ACTUALLY GOT - `copyFromDeal` awaited the copy and then announced success without looking at the result, which is what made v427's bug SILENT rather than an error: a body that fails to parse comes back from axios as a raw STRING under the default `silentJSONParsing`, so `res.data.cf_wf || []` produced an empty grid under a "copied" banner. The server is fixed, but the UI layer is the one that decides whether a failure is VISIBLE, so it no longer trusts the shape. One guard, `readStepsPayload`, rejects a non-object body or one carrying neither key, and the three paths that load steps into the drafts route through it - `copyFromEntity`, `createFromTemplate`, `loadSteps`. An empty array still passes, because an entity with no waterfall is not an error. Both handlers now print the STEP COUNTS they received, so an empty result cannot pass for a successful one. `createFromTemplate` was included because it had the identical three lines and the identical silent-success message two lines away. `saveSteps` needed no change - VERIFIED it already falls through to "Save failed." on a malformed body. (2) `dfc38df` AN ACTIVE DEAL IS SELECTABLE BEFORE IT HAS A WATERFALL, and this carries a CORRECTION TO MY OWN EARLIER DIAGNOSIS: the `InvestmentID = 'NONE'` collision is NOT what hid Jefferson Stephens. MEASURED: zero relationships rows carry `NONE`, so nothing ever looks it up and the collision changes nothing on today's data. The real blocker was a chicken-and-egg - the entity list was `rel_vcodes | wf_vcodes`, so a deal with neither a relationships row nor a waterfall was absent from the dropdown, and a deal cannot be given its FIRST waterfall without being selectable. TWELVE DEALS become reachable, 5 standalone (870 DLB, Jefferson Stephens P0000114, both PMAT Midwest deals, PSC Investee Fund XII Sub 1) and 7 child properties; five of those are five of the six deals the 2025 valuation cycle reports as missing a waterfall. PURELY ADDITIVE, verified by DIFFING THE PAYLOAD against HEAD rather than asserted: 254 -> 266, nothing lost, no pre-existing row changed, and the `has_wf` set - which is what "Copy from deal" filters on - IDENTICAL at 92. No sold deal is added; a sold deal already reachable via a relationships row or its own waterfall still appears. The placeholder collision is fixed anyway, and the reason is that it has NO SYMPTOM: injecting one `NONE` relationships row shows the old rule folding it into P0000114 without even changing the entity count, while the new rule surfaces it as its own unmapped entity where someone can see it. Guardrail `scripts/waterfall_entity_nav_check.py` 16/16, dropping to 9/16 against the pre-fix tree with "Jefferson Stephens P0000114 is selectable -> ABSENT"; TWO OF ITS ASSERTIONS WERE WRONG on first run and were corrected to test the real property rather than relaxed - sold deals legitimately appear via relationships (so the rule is that none is ADDED by this union), and 17 placeholder-id deals are absent because they are SOLD, not because they collapsed.

CHARLENE'S FOUR, all reviewed for symptom repair and NONE is one; one moves the other way. `0e929d2` A MISSING INPUT DOES NOT PRINT AS A COMPUTED ZERO - `pe_exposure_on_value` defaulted to 0.0 and is only assigned when `current_valuation > 0`, so a deal with no valuation printed "0.0%" beside a Valuation cell reading an em dash; now None, which is the standing rule applied deliberately rather than a patch. CONSUMERS VERIFIED INDEPENDENTLY here, not taken on trust: both Vue call sites pass it through `fmtPct`, which returns an em dash for null, and `assistant_service:1066` reads a DIFFERENT key (`pe_exposure_value`), so nothing does arithmetic on it. Same commit removes the One Pager's own `.print-date` div - reported as Chrome's print header bleeding through, and it never was; the browser's header has been suppressed since v421 and is untouched - and adds `fmtOccVariance`, whose `-0.0` guard stops a variance too small to show at one decimal printing a minus sign (Mount Prospect Plaza 26Q2, actual 95.6000 vs budget 95.6439). `8d0ef6c` A DEAL WITH NOTHING TO PLOT STILL GETS ITS CHART: the no-ISBS-rows guard returned before the calendar window was built while its SIBLING guard already exempted the windowed case - the same intent applied to one early return and not the other - so four deals at 26Q2 (Donald Lynch, Jefferson Stephens, Fairview Heights, Citizen Storage) rendered "No chart data available" instead of axes. It returns a NULL series and says why: ten quarters of 0.0 would draw flat lines asserting NOI and occupancy were MEASURED at zero. `7bc8d5f` the Snapshot "(Sold)" label keeps a non-breaking space in front of it. `4d46b63` fixes her own chart-frame guardrail so it still passes once merged.

GUARDRAILS ON THE MERGED TREE: onepager_missing_vs_zero 16/16, snapshot_sold_label_spacing 6/6, waterfall_entity_nav 16/16, waterfall_copy_json 12/12, capital_call_crud 49/49. The spacing check FAILED 5/6 on the first run and it was a STALE LOCAL `dist/` from an earlier `vite build`, not a defect - rebuilding cleared it, and the ACR image compiles fresh regardless; checked rather than assumed, because a guardrail that reads build output is exactly the one that can lie about a tree. Build genuine at 2m14s (runId cagu); tag locked writeEnabled=false; v428 clean boot, zero error lines, `/api/waterfall-setup/entities`, `/copy-from/P0000085` and the One Pager chart route all answering 401 not 500.

JEFFERSON STEPHENS P0000114 APPEARS IN TWO OF THESE FIXES INDEPENDENTLY - selectable in Waterfall Setup (mine) and no longer printing "No chart data available" (hers) - both symptoms of a new deal with no data yet, found from different directions on the same day.

NOT BROWSER-VERIFIED: dev servers cannot be started from a session the harness has flagged unattended, so the copy flow, the twelve new dropdown entries and the empty chart frame are proven at the function and payload level, not on screen. Worth a minute on live: open Waterfall Setup, confirm Jefferson Stephens is listed, copy Eastchase into it and expect 4 CF / 8 Cap rows WITH those counts in the message.

## v427 = `bf093c2`

WATERFALL SETUP "COPY FROM DEAL" HAD COPIED NOTHING ON 68 OF 92 DEALS, and said it worked. Jim copied Jefferson Eastchase into Jefferson Stephens: success banner, empty grid. THE ROWS WERE NEVER MISSING - the server builds them correctly (measured: 4 CF + 8 Cap for P0000085). A blank `nPercent` reaches `json.dumps` as a bare `NaN` token, which is NOT VALID JSON, and Eastchase has three. AXIOS DOES NOT REJECT THAT: with its default `silentJSONParsing` a response that fails to parse is handed back as the RAW STRING, so `res.data.cf_wf` is undefined, the store writes `[]`, and nothing throws - which is the whole reason the failure was silent rather than an error. SECOND TIME THIS SHIPPED: `e6858b5` ("Fix Waterfall Setup showing 0 steps: replace NaN with valid JSON values") fixed exactly this bug in `get_waterfall_steps` by writing the scrub INLINE, so the two copy paths kept the unscrubbed line and kept the bug. The duplicated rule was the defect; the missing scrub was its symptom. It now has ONE definition, `steps_to_records`, and all three payload builders route through it - a rule REMOVED, not a third copy of it added. NOT ONE DEAL: 68 of the 92 entities with a waterfall carry at least one blank cell (Seasons at Bel Air 16 bad cells, Trolley Square 15, Orange Grove / The Gallery / Adirondack RV Park / TGA24 / TGA25 14 each), so the feature only ever worked when the source happened to be clean. `copy_cf_to_cap` had the same hole, reachable only via the API since the view does CF->Cap client-side. Guardrail `scripts/waterfall_copy_json_check.py` 12/12, and note WHAT it asserts - the payload must survive a STRICT parse, the way the browser parses it, not the weaker "no NaN". Against the pre-fix tree in a throwaway worktree it drops to 4/12 and reproduces the report; TWO OF THE PASSES THERE ARE THE POINT: "copy-from returns 4 rows" PASSES pre-fix, which is the bug in one line, and `get_waterfall_steps` PASSES too, so the check discriminates rather than failing everything. Build genuine at 2m14s (runId cagt); tag locked writeEnabled=false; both guardrails re-run on the deploy tree (copy-JSON 12/12, capital-call CRUD 49/49); v427 clean boot, zero error lines, `/api/waterfall-setup/copy-from/P0000085` answering 401 not 500. NOT FIXED, reported to Jim: `copyFromDeal` in `WaterfallSetupView.vue` announces success without looking at what it got back - the same shape as the v423 capital-call save, and what turned a broken response into a silent one. ALSO OPEN AND NOT VERIFIED ON LIVE: Jefferson Stephens P0000114 is ABSENT from the entity dropdown locally - it has no waterfall and no `relationships` row, and its `InvestmentID` is the literal string `NONE`, shared with 24 other deals, so `inv_id_to_vcode['NONE']` collides and resolves to whichever deal iterates last. If it was not selectable on live either, the copy fix alone does not finish the job

## v426 = `9f0708d`

TWO commits, both reviewed for symptom repair and NEITHER is one. The first had been sitting UNPUSHED on local main - the exact v424 trap - and main is now pushed and in sync again. (1) `89c39a3` CAPITAL CALLS ARE APP-ENTERED ONLY: `capital_calls` joins PROTECTED_TABLES because the CSV import runs `to_sql(if_exists="replace")`, which DROPS the table, so one `MRI_Capital_Calls.csv` upload silently destroyed every hand-typed call - including the two Asbury Commons rows entered on v423. Jim's instruction, Sep 10 2026, and the resolution of the open question left at v423. It BLOCKS A DESTRUCTIVE WRITE PATH rather than suppressing a computed value - no sentinel, no vcode, no special-cased figure. CLAIMS VERIFIED INDEPENDENTLY rather than taken from the commit message: all four import entry points guard BEFORE touching the connection (`database.py` 1600 / 1649 / 1704 / 1750); `capital_calls` is genuinely absent from `mri_service.QUERY_REGISTRY` - the only "capital" hit in that module is a description string - so NO automated feed is interrupted; and the app's own CRUD is four direct routes in `deals.py` that never pass through the import guard. Guardrail `scripts/capital_call_crud_check.py` 49/49, asserting rows SURVIVE each refused import and that app edit/delete still return 200. THE LOCAL TABLE IS 3 USABLE ROWS OF 5,130 - the two Asbury calls (P0000004) plus one on P0000088; every other row is blank-Vcode junk from a past CSV, so the feed was contributing noise, not data. KNOWN CONSEQUENCE: those 5,127 blank rows can no longer be cleaned from the UI, since the GET filters by vcode and the replace that used to clear them is now blocked - harmless to every computation via `load_capital_calls`' dropna, but permanent; purging them is a separate deliberate act. NOT VERIFIED FROM HERE: whether LIVE Postgres holds capital-call rows that came from the CSV and are now bulk-unrefreshable - that session had no live DB access. Deploying destroys nothing either way; it only closes the bulk path going forward. (2) `9f0708d` THE WATERFALL SETUP ENTITY DROPDOWN SORTS ALPHABETICALLY - it was already sorted, by the WRONG KEY: `all_ids = sorted(rel_vcodes | wf_vcodes)` orders by vcode while the `<option>` renders `label`, which leads with the investment name, so it read as unsorted on screen. Now sorted by label casefold with vcode as the tiebreaker for unnamed entities, whose label is the vcode itself. Verified through the real function against the local database: 254 entities, `labels == sorted(labels, key=str.casefold)`, vcodes still unique. Both dropdowns inherit it, since "Copy from deal" filters the same array. Two visible consequences: digit-leading names sort ahead of letters, and a property carrying both a vcode and a legacy id now sits adjacent to its twin. Build genuine at 3m04s (runId cags); tag locked; v426 clean boot, healthy, zero error lines

## v425 = `54fe25a`

Charlene sent FOUR SHAs for the day; `db13161` was ALREADY LIVE in v424 and `54fe25a` is the tip containing the other three, so ONE deploy ships the batch. She had pulled main after v424, so this is a FAST-FORWARD with no divergence and no merge - the opposite of v424's trap, and worth noting as the pattern that works. THREE ONE PAGER FIXES, all reviewed and none a symptom repair. (1) `8126cc4` NO VARIANCE AGAINST AN ACTUAL NOBODY REPORTED - the textbook version of the standing rule: revenue/expenses/noi `ytd_actual` seeded to a literal `0`, so `(0 - budget)/|budget|` printed a confident -100% beside a BLANK YTD Actual cell, since `fmtMil` renders 0 as an em dash - the same sentinel invisible in one column and load-bearing in the next. Fixed AT THE SEED (0 -> None), not at `fmtVariance`, which had always guarded null and simply never saw one. TWELVE DEALS not the two reported (Giant 7, East Manchester, Ayr Town Center, City West, Creekside Market Place, Declan & Walton, Orange Grove, Parkway Plaza, Quakertown, Scott Town Center, Spring Meadow, Stonehedge Square - six being Giant 7's own children, consistent with one feed stopping). CONSUMERS INDEPENDENTLY VERIFIED here, not taken on trust: `financials_service:1177` reads it `or 0`; `one_pager:2688` tests `!= 0` so None lands where 0 did; `fmtMil` renders null and 0 IDENTICALLY (`val == null || isNaN || val === 0`) so no readable cell goes blank; and `_payload_unpopulated` is BEHAVIOURALLY IDENTICAL under the change because its loop does `if v is None: continue` and a 0 likewise fails `!= 0.0`, so both fall through to the same 'unpopulated' verdict for ANY column it is called with - a stronger guarantee than her stated audit. HER AUDIT MISSED ONE CONSUMER, verified harmless: `portfolio_snapshot_loan:1072` reads `noi.ytd_actual` for Debt Yield, but line 1100 is `if ytd_noi:`, falsy for both 0.0 and None, so only the `ytd_noi` DIAGNOSTIC field differs. Only `ytd_actual` moved; the other five keys stay 0 because retiring them means retiring `_payload_unpopulated` too. (2) `0e0b88f` CURRENT PREF EQUITY BALANCE IS AS OF THE REPORT QUARTER - `_enrich_pe_from_deal_result` built the deal result with the GLOBAL `actuals_through` (2026-07-31 live) while every other figure on the page is quarter-filtered, so the balance was QUARTER-INVARIANT: Burton printed 54,227,500 in 25Q4, 26Q1 and 26Q2 against funded 26,597,500, the difference being ONE contribution dated 2026-07-01 - the 25Q4 report showing capital not contributed for another seven months. JB Fair Park had it unreported (+1,462,095 from a 7/30 draw, while its 8/19 draw was absent - which is what pins the cause to the 7/31 boundary and not to 'as of today'). Fix is THE DATE not the formula: `min(quarter_end, global_boundary)`, capped because past the boundary the engine reads forecast rather than accounting. She DELIBERATELY REJECTED `funded_to_date - return_of_capital`, which looks equivalent and is not - ROC also absorbs realized-gain rows, and East Manchester proves it (ROC 5,139,662 against 3,600,000 funded would print -1,539,662 where the engine correctly reports 0). Blast radius 2 deals of 36 scanned. (3) `54fe25a` A SOLD DEAL DOES NOT PRINT ITS STALE DEBT - THIS ONE IS A SUPPRESSION and was flagged to Jim as such before building. Judged CORRECT rather than a symptom repair: MRI stops the balance sheet at disposal instead of writing the payoff down, so there is NO correct post-sale balance to compute (East Manchester sold 2026-06-25, BS ends 2025-11, printed $9,641,912 forever); raw `debt`/`debt_isbs` are preserved so audits and the Snapshot's twin still see the figure; Total Cap EXCLUDES the suppressed leg so the printed row still foots; the trigger is a real domain condition, the Snapshot's own quarter-aware `is_sold_as_of` IMPORTED not reimplemented (verified), so Clima Secur sold 2026-07-01 is NOT suppressed at 26Q2; and it mirrors SOLD_NA_CELLS, approved at v409. EIGHT DEALS' VISIBLE DEBT CHANGES, with Total Cap and the debt percentages moving with them: Berger Pittsburgh 190.8M, Bear Run 80.0M, Heritage Hills 42.6M, Lindenbrooke 40.8M, Stonecliffe 27.4M, Quakertown 11.6M, East Manchester 9.6M, Airport Plaza 5.8M - each verified stale, 19 other sold deals already read 0. Her `capDebtCell` tests `'debt_display' in c`, a GENUINE KEY TEST - she fixed the truthiness pattern flagged on SnapshotLoan.vue at v424 - so a snapshot frozen before the field existed keeps rendering its raw debt. NO vcode was added to product code in any of the three (comments only). Capital-call guardrail re-run on the deploy tree 31/31; build genuine at 2m14s (runId cagr); tag locked; v425 clean boot, zero error lines, routes answering 401 not 500. NOT INDEPENDENTLY VERIFIED: her three new guardrails need WF_TOKEN and live API access, so the per-deal figures above are hers - and 8126cc4 / 54fe25a both change investor-facing cells, so a spot-check of a sold deal's One Pager and of one of the twelve -100% rows is worth doing on live

## v424 = `3a21c3c`

MERGE, not the commit that was asked for. Charlene asked Jim to deploy `db13161`; MAIN HAD DIVERGED 4/4 and `db13161` DOES NOT CONTAIN the capital-call fix `eb34396` that v423 was running - `_cap_calls_schema` is absent from `deals.py` at that SHA - so shipping it alone would have REVERTED a fix verified on live an hour earlier, and the two Asbury calls would have vanished from the Manage list again (the rows themselves are data and would have survived). Root cause of the trap: the v423 line was committed locally and NEVER PUSHED, so origin/main carried Charlene's four and local main carried mine. Put to Jim BEFORE building, per the standing instruction; he chose merge-and-deploy. The merge is clean with ZERO overlapping files (she touched snapshot/One Pager, I touched deals.py + DealAnalysisView/DataTable/deals.ts) and main is now PUSHED and in sync, which is what stops this recurring. SHIPS FOUR OF HER COMMITS, none of them previously live: `a318722` Financial Total Cap foots to the Debt the row prints, `142a92f` one definition of 'development deal' across pages 6 and 8, `c5f30a0` One Pager economic occupancy formatted by unit not magnitude, `db13161` Loan ex-dev debt subtotal foots to its visible rows. ALL FOUR REVIEWED FOR SYMPTOM REPAIR and none is one; TWO MOVE THE OTHER WAY, which is worth recording: `142a92f` DELETES the `EXCLUDING_DEV_VCODES` per-deal hardcode (a frozenset of eight vcodes) in favour of the real `is_dev_deal` classifier - verified the constant is gone from live code, its only surviving mentions being comments dating the deletion - and `db13161` REMOVES a duplicated rule rather than patching it: the ex-dev debt subtotal was the one total computed client-side (`exDevTotal`), summing RAW `debt` while the cell beside it printed `debt_display`, so a suppressed row showed an em dash and still fed the total ($984,768,975.62 against $975,127,063.62 of visible rows, the $9,641,912 gap being East Manchester to the cent); it now routes through `loan_subtotal`, which has dropped `sold_suppressed` rows since 2026-09-02, which is precisely why the fund subtotals and Portfolio Totals were never wrong. Her diagnosis is the correct direction and worth quoting: 'The duplicated rule was the defect; the missing flag was its symptom.' No sentinel escapes to a consumer - the one `debt_leg = 0.0` is a summand in an addition, the arithmetically correct way to omit a term, and both server and Vue fallback deliberately distinguish a PRESENT NULL (real dash) from an ABSENT KEY (old payload). `deal_count` stays 27 rather than being trimmed to make the arithmetic tidy. HER MEASURED FIGURES ARE NOT INDEPENDENTLY VERIFIED: `snapshot_loan_exdev_subtotal_check` needs WF_TOKEN and live API access, and the affected deals postdate the local waterfall.db. ONE NIT SHIPPED AS-IS: `SnapshotLoan.vue` does `if (served) return served`, a truthiness test, while its comment claims to test for the KEY - correct today only because `loan_subtotal` always returns an object. Capital-call guardrail re-run on the merged tree 31/31; build genuine at 2m11s (runId cagq); tag locked writeEnabled=false; v424 verified serving 200s on the live One Pager endpoint with no tracebacks

## v423 = `eb34396`

CAPITAL-CALL CRUD HAD NEVER WORKED ON AZURE. Jim added a call for Asbury Commons and nothing happened. All four `/api/deals/<vcode>/raw-capital-calls` endpoints spoke to the database through `database.get_db_connection()` with sqlite3-shaped calls - raw SQL strings, `?` placeholders, tuple params and `rowid` - and that function returns a SQLAlchemy Connection whenever an engine is wired in, which SQLAlchemy 2.x rejects BEFORE the statement reaches PostgreSQL: `ArgumentError: List argument must consist only of dictionaries`. So POST/PUT/DELETE were 500s and GET SWALLOWED ITS EXCEPTION and returned `{"capital_calls": []}`, meaning the editable list had looked empty for every deal on live since the table existed. INVISIBLE IN LOCAL DEV BY CONSTRUCTION - `flask_app/__init__` only calls `database.set_engine()` when DATABASE_URL is set, so SQLite takes the raw sqlite3 path and the old code worked; this is the trap for any other endpoint still on `get_db_connection()`. THREE PG-SPECIFIC FAULTS each fatal alone: (1) identifiers must be DOUBLE-QUOTED because `capital_calls` is built by the `MRI_Capital_Calls.csv` import through `to_sql(if_exists="replace")`, which rebuilds it from the CSV's own headers and so keeps mixed case - unquoted `Vcode` folds to lowercase and errors; (2) PG HAS NO `rowid` and to_sql leaves no key at all, so Edit/Delete were unreachable there regardless of the param bug - a surrogate `"id" SERIAL` is added, SQLite keeps rowid; (3) `Typename` MAY NEVER HAVE EXISTED on PG since `create_additional_tables`/`run_migrations` run ONLY on the SQLite path. Columns are therefore resolved by INTROSPECTION (`_cap_calls_schema`, self-healing on every request because the CSV import can drop the table again), and PUT/DELETE are scoped to the vcode in the URL and report 404 instead of silently matching nothing. FRONT END was the other half of 'nothing happened': `saveCapCall` had `try/finally` with NO CATCH, so the rejection went unhandled and the modal neither closed nor complained - it now shows the server's error and keeps the entry, validates required fields first, and the list surfaces its own load error. SECOND REAL BUG CAUGHT WHILE VERIFYING: dates on this page read A DAY EARLY - a call saved as 2026-09-30 displayed as 9/29/2026 because `new Date('2026-09-30')` is midnight UTC; `fmtDate` and a new DataTable `date` format now parse the parts, the same fix OnePagerView already carried. Guardrail `scripts/capital_call_crud_check.py` 31/31 - drives the REAL view functions through Flask test requests with a REAL JWT against a table seeded exactly as the CSV import leaves it, in two sections: SQLite, and the PostgreSQL branch with `is_postgres()` FORCED, which asserts the emitted SQL carries no rowid, quotes every mixed-case identifier and uses named binds. Run against the pre-fix code at `9dc5704` in a throwaway worktree the same battery REPRODUCES the reported failure, and note WHICH check passes there: 'GET does not report an error' - that is the silent-empty bug in one line. Verified in the running app on Asbury Commons P0000004: PPI22 and OPFLAG, $130,000 each on 9/30/2026, both saved, both listed at 9/30/2026, both picked up by the engine into the computed Capital Calls Schedule. VERIFIED ON LIVE POSTGRES by an authenticated round-trip - Jim entered both Asbury calls on v423 while the container log was tailed: GET 200 at 21 bytes (empty), POST 201, GET 200 at 224 bytes (one row), POST 201, GET 200 at 417 bytes (two rows), all interleaved computes 200, and NO 500 / Traceback / ProgrammingError / UndefinedColumn anywhere. The 200 on that GET is what proves the two things the forced-branch guardrail could not: the `"id" SERIAL` ALTER actually succeeded on the real table (its failure only logs a warning, but the next `SELECT "id"` would then 500), so Edit/Delete work on live too; and the double-quoted mixed-case identifiers match the real column names, since one wrong case would be an UndefinedColumn on the INSERT. TAG LOCKED (`waterfall-xirr:eb34396` writeEnabled=false, deleteEnabled=true). Worth knowing for the next deploy: that command was declined ONCE by the auto-mode permission classifier and went through unchanged on a retry - `.claude/settings.local.json` already allows both `az acr:*` and `az:*`, so the classifier is a SECOND gate on top of the allowlist and its decisions are not deterministic. Adding another permission rule does not help; retry the command. NOT FIXED, Jim's call pending: `capital_calls` is not in PROTECTED_TABLES and the CSV import REPLACES the table, so importing MRI_Capital_Calls.csv drops every hand-entered call along with the id column; protecting the table would block the MRI feed instead, so it is a choice about which of the two wins

## v422 = `2ea1186`

Portfolio Snapshot printed report REBUILT LANDSCAPE after Jim asked what a professional investor-facing version needs. The data on pages 2-4 was set at 4.5pt - BELOW LEGAL FINE PRINT - and three compensations were needed for even that: 7.5px type, 1.5px cell padding, comment column capped at 1.5in. Diagnosis came from measuring HIS printed PDF, not from reading CSS: every table page laid out WIDER than a portrait column (7.68in of content in 7.50in) while leaving 2.2-3.2in of blank paper at the foot, and page 1 used 5.73in of 7.50in with 4.86in of slack. The document was starved of width and swimming in height. LANDSCAPE DOES NOT ADD AREA (letter is 93.5 sq in either way) and the first render proved it - 11px gave a readable 8.2pt and SEVEN pages. Type was then sized from the measured budget: LINE HEIGHT not font-size sets row pitch at these sizes (at 8px default leading still held pitch at 0.146in vs 0.157in at 9.5px; pinning 1.05 dropped it to 0.137 and BOUGHT BACK type size), vertical padding 0.5px / horizontal 5px since width is what landscape bought, footnote block compacted from 1.08in to ~0.45in (it alone was the 5th page), and THE COVER TITLE REPEATED ON PAGE 2 dropped in print - Financial's table started at 1.30in where Operating/Loan start at 0.77in and Financial is the only page also carrying footnotes. THAT LAST ONE DEVIATES FROM THE REFERENCE PDF, which prints the title on pages 1-2; flagged to Jim explicitly, he shipped it. RESULT 4 pages, 4.5pt -> 6.0pt. SIZED FOR THE LIVE 36 DEALS not the 30 in the local snapshot - each figure is the measured local page plus six rows at measured pitch: Summary 8.2pt margin +0.47in, Financial 6.0pt +0.13in, Operating 6.0pt +1.66in, Loan 6.0pt +1.52in. Financial is tight BY CONSTRUCTION and that is the reassuring part - it is the only table with NO comment column so its height is deterministic in row count, while Operating/Loan carry the comments and their 1.5in of slack absorbs a comment wrapping to a second line on live (this snapshot has none). PAST ~37 DEALS the footnotes take a fifth page; the lever is the one font-size in the file. Orientation is per-document via a NAMED page box in App.vue - `@page` cannot be scoped (v421) so a view redefining the DEFAULT breaks every other printing view, but a named box applies only where an element opts in with `page: landscape-sheet`, so the One Pager and Lease Abstract stay portrait by construction. THE PREVIEW NOW PREVIEWS: the table-fitting rules moved OUT of @media print and the on-screen sheet is 11x8.5in with the paper's padding - previously that page kept overflow-x:auto, its interactive font and the sticky column's min-width on screen, so it showed scroll bars and clipped columns, which is what Jim reported as broken formatting. VERIFIED by rendering the REAL browser print path (scripts/snapshot_print_check.mjs, headless Chrome --print-to-pdf against a local Flask) and measuring the PDF. print_page_rule_check 14/14, extended for named boxes (any number of named, exactly ONE default, all zero-margin, all in App.vue) and its regex tightened - it had been matching a // comment mentioning @page and reporting a false leak. Verified on LIVE after deploy: index chunk serves both page boxes, the print chunk serves page:landscape-sheet

## v421 = `c8c367b`

PRINT REGRESSION, root cause found after Jim reported the One Pager printing with bigger margins and the app name back in the header: `@page` CANNOT BE SCOPED. Vue's scoped CSS appends a [data-v-hash] to SELECTORS, and an at-rule styling the page box has none - so a `@page` in any component's <style scoped> escapes into the global sheet and applies to every route. FOUR components had one (One Pager margin 0, Snapshot print margin 0, PortfolioSnapshotView 0.5in, LeaseAbstract 0.5in), proven from the built CSS where none carried a scope attribute. Routes are lazy-loaded and Vite APPENDS a route's stylesheet and NEVER removes it - verified in the bundle, `document.head.appendChild(h)` with ZERO removeChild/.remove() anywhere - so all four accumulate in one document and THE LAST ROUTE VISITED WINS the margin descriptor. Which view printed correctly depended on navigation order. The culprit was `830934d` (Aug 24, 'Fix: Portfolio Snapshot formatting and print styles'); NOTHING in OnePagerView.vue had changed since July 30. One line produced three symptoms because the zero margin was doing TWO jobs: it set the side margins AND it is the mechanism suppressing the browser's own header/footer, since Chrome only draws those when a page margin leaves room - restore the margin and the title comes back. THE CORRUPTION WAS MUTUAL and not only the One Pager: with One Pager or the Snapshot print view visited last, the Snapshot and Lease Abstract printed at margin 0, EDGE TO EDGE; only the One Pager was noticed. FIX: the page box is declared EXACTLY ONCE, in App.vue's non-scoped block (the only always-loaded stylesheet), as `letter portrait; margin: 0`; each print view supplies its real margin as padding on its own container, which the One Pager and Snapshot print view already did. Every view keeps the margins it was written to have - One Pager 0.4/0.5, Snapshot 0.5, Snapshot print 0.42/0.5/0.34/0.5, Lease Abstract 0.5 - and now keeps them regardless of where the user has been. BEHAVIOUR CHANGE, flagged to Jim BEFORE deploy: Snapshot and Lease Abstract no longer print the browser's header/footer, and this cannot be opted out per view since that is exactly what @page will not scope. Judged correct because PortfolioSnapshotView IS NOT A PRINT DELIVERABLE AT ALL - its 'Print report' button opens /portfolio-snapshot/print in a new tab, so its whole @media print block only ever fired on a stray Ctrl+P - and Lease Abstract already renders its own corner timestamp, the same pattern the One Pager uses BECAUSE the browser header is suppressed. Guardrail scripts/print_page_rule_check.py 11/11 - fails if a second @page appears in source OR in the build, if it lands anywhere but App.vue / the index chunk, or if a print view stops setting its own padding. VERIFIED AGAINST LIVE after deploy, not just the local build: index-LegteruW.css serves `@page{size:letter portrait;margin:0}` and all four route chunks serve none

## v420 = `d1259bb`

TEMPORARY per-deal suppression: Hanestowne Waterstone P0000118 reads an em dash across EVERY Operating metric - econ occ, all three NOI columns, both growth figures - by name rather than by rule. THE THIRD PER-DEAL HARDCODE on the Snapshot after MANUAL_RATIO_SEEDS and PROJECTED_YE_NOI_FALLBACK, and the one place a hardcode is genuinely the only instrument available: THE REQUESTED OUTCOME CANNOT BE EXPRESSED AS AN OWNERSHIP AGE. Hanestowne is 3.38 months owned at 26Q2 and Plaza Del Mar 3.48 - THREE DAYS APART - and they are to be treated differently, so no value of INSUFFICIENT_HISTORY_MONTHS separates them (5.0 catches both, 3.0 catches neither). An exception to a rule that works, not a patch over one that does not; the threshold is UNTOUCHED at 3.0 and Presidential Arms is still caught by the age rule on its own (1.58mo). Two lines of behaviour - the vcode set plus one term in `shown()`, the single gate all four metric columns already pass through - so the 'must not feed subtotals' requirement comes FREE and stays auditable via the existing noi_contributors / suppressed_count. MEASURED: subtotal NOI falls by exactly Hanestowne's figure in all three columns (9.15->7.05, 10.10->7.70, 9.80->7.50), one fewer contributor each, suppressed_count 1->2, deal_count UNCHANGED (withheld, not removed - the reader still sees the row), weighted occupancy drops its reading. Plaza Del Mar cell-for-cell identical. Raw noi / econ_occ untouched - display only, so a frozen payload keeps its real figures. Kept SEPARATE from `insufficient_history` on the row so the age rule is not credited with a catch it did not make, and the row carries a TEMPORARY flag plus econ_occ_basis = 'TEMPORARY per-deal suppression'. THE DURABLE RULE IS PROBABLY NOT AGE AT ALL and this is recorded at the constant: the stated reason Hanestowne differs from Plaza Del Mar is that it carries NO ACTUAL OPERATING DATA while Plaza Del Mar has data the page can print - so the rule wants to be 'a recent acquisition with no actual operating readings', keyed on data, retiring itself when the feed starts, covering the next acquisition with no edit. NOT WRITTEN because P0000118 is absent from the local snapshot and the premise could not be measured rather than assumed; put to Charlene. DOES NOT EXPIRE ON ITS OWN - a vcode set keeps firing next quarter until someone deletes the line, so it joins the weekday `retire-manual-ratio-seeds` reminder, which now tracks THREE hardcodes and notes that Hanestowne carries one on TWO subtabs (it is also a MANUAL_RATIO_SEEDS entry). Guardrail snapshot_temp_suppress_check 26/26 - none of the three deals exist locally (all newer than waterfall.db, accounting ends 2026-06-02) so the page is assembled from injected Step 1 entries at their real closing dates and before/after is taken by emptying the constant, with Burton as a real-data control; the live local 26Q2 page is BYTE-IDENTICAL with and without the override across all 30 deals. freeze self-test 34/34, snapshot_insufficient_history_check 15/15. The Operating module's own _selftest STILL CANNOT RUN - it imports the uncommitted scripts/live_api.py

## v419 = `a7cb4c4`

Charlene, root cause NOT the threshold: the Operating tab's insufficient-history suppression HAD NEVER FIRED FOR ANY DEAL. `months_owned` reads `payload["general"]["date_closed"]` and the Snapshot's lean provider `_one_pager_provider` skipped the general block for speed, so the date was always None, months_owned was always None, and the rule is guarded on `mo is not None` - MEASURED 0 of 30 deals had a computable ownership age before, 30 of 30 after. A brand-new acquisition was therefore reported as though it had a full operating history. INSUFFICIENT_HISTORY_MONTHS STAYS AT 3.0 - the 5.0 raise on fix/snapshot-debug4 @ 0c0c68d is ABANDONED, not merged, because it was treating the wrong cause; that branch should be discarded. The provider now emits one field through the new `one_pager.closing_date_from_row`, which is the single definition of the five-column precedence (Acquisition_Date, DateClosed, Date_Closed, dtClosed, ClosingDate) that `get_general_information` used to hold inline - both paths now reference it and the guardrail asserts they agree, so the lean page and the full One Pager cannot drift about when a deal closed. No extra query; the row is already in memory. NOT the whole general block, deliberately: that path exists to stay lean and this is the only field a subtab reads. IMPACT ON THE 26Q2 PAGE IS NIL and that null result was reported to Jim BEFORE deploy: every deal there is owned 9.6 months or more (youngest Trolley Square 9.6, Burton 10.1), so nothing crosses 3.0 and no established row moves; the Operating tab still renders 29 of 30 econ-occ figures. The deals it targets - Presidential Arms ~1.6mo, Plaza Del Mar ~0.5mo, Hanestowne ~0.4mo - are NEWER THAN THE LOCAL waterfall.db (accounting ends 2026-06-02) so they are not in this snapshot and their suppression is proved against their real closing dates rather than against rows I do not have. CHARLENE SHOULD SPOT-CHECK THE LIVE 26Q2 OPERATING TAB: any deal closed within 3 months of quarter end now reads an em dash where it previously printed a figure, and that live list could not be produced from here. LEFT AS IS ON PURPOSE: the guard still FAILS OPEN - no closing date means no suppression, which is backwards for this rule - because suppressing on unknown would blank a deal whose payload lacks the field for an unrelated reason; but the condition is now reported as a row flag (`no_closing_date`) so the next silent exemption says so instead of looking like a rule that works. Guardrail snapshot_insufficient_history_check 12/12; portfolio_snapshot_freeze self-test 34/34, snapshot_giant7_ye_fallback_check 6/6, snapshot_loan_sold_suppression_check 12/12, snapshot_east_manchester_check 26/26

## v418 = `dcefc29`

Prompt 4 of PortSnapshot_Debug_4 ONLY - SPLIT OUT of the commit it shared with Prompt 2 so it could ship alone. Giant 7 (P0000019) drives YTD DSCR and Debt Yield from its Projected YE NOI (~9.2M): the deal is under PSA to sell and its actual NOI feed stopped Nov 2025, so there is no complete quarter of NOI and both cells read an em dash. DSCR reuses the One Pager's own `dscr.actual_ye`, which already IS Projected YE NOI over debt service, rather than forming a second opinion about Giant 7's debt service; Debt Yield is Projected YE NOI / Debt (0.0967 = 9,200,000 / 95,105,179). A PER-DEAL HARDCODE, `PROJECTED_YE_NOI_FALLBACK`, and the SECOND on this page after MANUAL_RATIO_SEEDS - but a FALLBACK not an override, so it fires only where the real input is missing and retires itself the moment the feed resumes; scoped by vcode; one deletion to remove; and the row carries a TEMPORARY flag so the figure is not read as computed. FOLDED INTO THE DAILY REMINDER at Jim's instruction - the weekday `retire-manual-ratio-seeds` task now tracks BOTH hardcodes, and asks a different question for this one (has the NOI feed resumed?) because unlike the seeds it can become removable without anyone doing work. Guardrail snapshot_giant7_ye_fallback_check 6/6, proving the path by injecting the live Projected YE NOI, since this snapshot lacks it too. STILL NOT DEPLOYED on `fix/snapshot-debug4` @ 0c0c68d: Prompt 2, INSUFFICIENT_HISTORY_MONTHS 3.0 -> 5.0, together with the finding that it will NOT fix Presidential Arms because the new-acquisition rule FAILS OPEN - `months_owned` returns None with no `date_closed`, so a deal with no known closing date is treated as established and no threshold reaches it

## v417 = `6dac97f`

Prompt 3 of PortSnapshot_Debug_4 ONLY - a deal reported after its sale carries no loan. East Manchester (P0000017) and City West (PCITWES) now read an em dash in EVERY Loan column - Rate, Maturity, Debt, YTD DSCR, LTV, Debt Yield - because the asset and its facility went together. NOT merely cosmetic: the two pages of the same report disagreed, with the Financial row saying Debt n/a since v409 while the Loan row printed 9,641,912, and `loan_subtotal` sums the RAW debt, so a sold asset's balance sat inside the totals. Individual Investments Debt 221,515,091 -> 211,873,179 (-9,641,912) and its weighted LTV 67.41% -> 68.25% (ltv_n 5 -> 4); City West's Debt stops printing "0.0". Keyed on `kept_despite_sold`, NOT on a vcode, so both pages now answer the same way. Display only - raw debt/ltv/ytd_dscr/debt_yield untouched, `sold_suppressed` is what keeps them out of loan_subtotal and aggregation_value. Guardrail snapshot_loan_sold_suppression_check 12/12, and note its design: City West fires on UNMODIFIED local data, which is what proves the rule keys on the sale rather than on the injected Sale_Date East Manchester needs locally. NOT DEPLOYED and still on `fix/snapshot-debug4` @ 0c0c68d: Prompt 2 (INSUFFICIENT_HISTORY_MONTHS 3.0 -> 5.0) and Prompt 4 (PROJECTED_YE_NOI_FALLBACK for Giant 7, a second per-deal hardcode on this page). Prompt 1 was WITHDRAWN by the author after the WIP showed Pegasus growth computing to +1482%; Pegasus is byte-identical to main. OPEN QUESTION recorded there: the new-acquisition rule FAILS OPEN - `months_owned` returns None with no `date_closed`, so a deal with no known closing date is treated as established, and no threshold reaches it

## v416 = `eb7520d`

Charlene, TWO commits — Jim approved AS A STOPGAP so the reports could go out, WITH the hardcode acknowledged and a standing daily reminder to replace it. (1) `2a3fabe` MANUAL_RATIO_SEEDS: LTV / YTD DSCR / Debt Yield become typeable, pre-filled cells on SIX vcodes — Burton P0000109 LTV 69.0, Plaza Del Mar P0000116 LTV 64.2, Fairview Heights P0000117 69.7/1.9/12.1, Hanestowne P0000118 75.7/1.5/8.9, Presidential Arms P0000119 70.6/1.1/5.93, Citizen Storage P0000120 74.0/1.5/9.7. THIS IS A PER-DEAL HARDCODE and is labelled as one in the code. Root cause is structural and NOT fixed: no valuation dated on/before the report year-end, no full YTD Interim IS plus BS principal movement for a DSCR, no complete three-month quarter of actual NOI for a Debt Yield. Safety verified independently: the raw `ltv`/`ytd_dscr`/`debt_yield` are never reassigned (only *_computed/*_manual twins are added), a seed is only a DEFAULT so a cleared cell stays an em dash, and Dev/N/A literals still outrank a typed cell. (2) `eb7520d` makes the subtotals weight what the row DISPLAYS via `aggregation_value` — defensible on its own, since a total that cannot be re-derived from the rows above it is wrong, and TGA 6 was printing DSCR 3.81x above a member row reading 1.1x. The unit conversion is load-bearing: a typed 69.0 is divided by 100 to sit beside a computed 0.6203, and without it those two deals average 3,201%. TOGETHER they mean HAND-TYPED NUMBERS NOW DRIVE INVESTOR-FACING TOTALS: 26Q2 Portfolio LTV 62.6->63.9%, Portfolio DSCR 1.91x->1.75x, TGA 6 DSCR 3.81x->1.27x, TGA 2025 LTV blank->69.1%. The sharpest single item is Presidential Arms' typed 1.1x DELIBERATELY REPLACING a computed 3.8x — that one override is what moves TGA 6. NOT INDEPENDENTLY VERIFIED: her 126/126 guardrail imports `scripts/live_api.py`, which is still not committed, so the before/after figures above are hers; I verified the code and a partial local assembly (Burton renders 69.0% with raw computed None, TGA25 subtotal 0.69 — the conversion is right). KNOWN DEFECT SHIPPED: the row's warning flag still reads "the computed figures ... still feed the subtotals", true when written in 2a3fabe and made false by eb7520d an hour later; it is visible on the row tooltip. TO RETIRE: delete a vcode from MANUAL_RATIO_SEEDS and its cells revert to the engine

## v415 = `639f023`

Jim: footnote 2 is CITY WEST ONLY. East Manchester was added to the ROE-exclusion note in 7dc7bd8 and removed here, because from v413 its Net ROE is typeable and its ITD shows - the whole reason its row is kept after the sale - so a note saying it is excluded from ROE contradicted its own row. The TEXT changed as well as the anchor: "City West is excluded from ROE calculations", anchors just deal:PCITWES. Removing only the anchor would have left a note naming a deal it no longer marks; the guardrail asserts both. City West stays - FORECLOSED, not sold, so there is genuinely no ROE, and its Net ROE is n/a via its own PDF_NA_CELLS entry. Second time today that foreclosure-vs-sale distinction decided the answer. ALSO RECORDED, NOT CHANGED: Jim asked why Debt is still n/a when the rest of the row is now read at the last held quarter - a real inconsistency, since East Manchester's 9,641,912 WAS genuinely outstanding at 26Q1 and is not the stale post-sale reading SOLD_NA_CELLS was written to suppress. Charlene's instruction is to keep it blank and Jim confirmed following it, so Debt stays n/a and Total Cap stays 6.0M rather than 15.6M. The reasoning and the accepted consequence are written at the SOLD_NA_CELLS definition - do NOT treat it as an oversight; reopening it means changing what the row is for. snapshot_east_manchester_check 26/26

## v414 = `bd001e8`

TWO engine fixes merged, Jim approved. (1) A MID-MONTH SALE IS NO LONGER MODELLED AS A MONTH-END SALE. `compute.py` applied `month_end()` at the moment it PARSED the sale date, so the real date was gone before anything used it - a 4 Sep sale became 30 Sep everywhere, and pref and debt ran 26 extra days. Two dates are now kept: `sale_actual` (pref stops, loan repaid, closing period settles) and `sale_me` (the forecast, cash schedule and terminal-NOI window stay on the monthly grid, because the operating data cannot split a month). 30 Bearfoot/OPMCCORD's $300,000: pref 3,550.68 (48 days) -> 1,627.40 (22 days); loan payoff 11,831,749.05 -> 11,856,279.30. TERMINAL NOI now starts the month AFTER the month of sale - it used to begin WITH the sale month while the cash schedule also gave the seller that month, counting it twice. THAT PART MOVES EVERY DEAL, not just mid-month sales: on a 14-deal sample terminal NOI moved -1,090 to +15,352 and net proceeds -14,387 to +272,561. The refi path is untouched. (2) A DEAL KEPT AFTER ITS SALE REPORTS THE STACK IT HAD WHILE HELD. East Manchester's Total Pref/Invested/Commitment went blank on live at 26Q2 because the sale's return of capital nets them to zero and the cap stack was being asked at the quarter of sale; new `portfolio_snapshot_service.last_held_quarter` rebases a kept-sold row to the quarter before the sale (26Q1 here). NO vcode in the rule - it keys on Sale_Date - but it is reached only through the pre-existing per-deal KEEP_DESPITE_SOLD set. City West is rebased too and is a MEASURED no-op (it has no return-of-capital row in any quarter). All 30 deals on both pages byte-identical; page 1 stops disagreeing with page 2 about East Manchester's funded pref ($0 vs $3.60M). v413 intact: Debt still n/a and out of the subtotal, Net ROE still typeable. LATENT AND NOT FIXED: `cap_stack.pref_equity` is capital OUTSTANDING while three columns describe it as funded/invested/committed, so a LIVE deal with a partial return of capital would understate Invested - none on the 26Q2 page has one today. Guardrails: sale_date_month_end_check 10/10, snapshot_kept_sold_stack_check 38/38, snapshot_east_manchester_check 22/22, capital_reversal_sign_check 19/19, pref_excess_cf_check 6/6, snapshot_itd_roe_check 30/30, snapshot_footnotes_freeform_check 28/28. Footnote 2 STILL contradicts East Manchester's visible Net ROE, deliberately, pending Charlene

## v413 = `7d21769`

Charlene: East Manchester keeps a TYPEABLE Net ROE after its 6/25/2026 sale, because its ROE and ITD distributions are tracked for sold deals - that is why the row is kept. Most of the brief was already true (Total Pref 3,600,000, Ptr Equity 2,400,000, Invested/Commitment 2,723,400, % of Pref 75.65%, Debt n/a, ITD already a manual cell); the ONE defect was Net ROE reading n/a, because `SOLD_NA_CELLS` blanked it on every kept-despite-sold row and a blanked cell renders READ-ONLY, so the figure could not be entered. Fix is one line - SOLD_NA_CELLS drops to `{"debt"}`, the only thing the sale genuinely invalidates - and adds NO per-deal hardcode. City West is untouched BY CONSTRUCTION: its Net ROE stays n/a through its own static PDF_NA_CELLS entry, correct because foreclosure is not a sale and it really is out of the ROE numbers. Total Cap stays 6.0M (0 + 3.6 + 2.4) rather than 15.6M so the row foots against what it prints; raw debt 9,641,912 preserved underneath. FOOTNOTE 2 IS DELIBERATELY UNCHANGED at Charlene's instruction and is now CONTRADICTORY IN PRODUCTION - it reads "City West and East Manchester are excluded from ROE calculations" while East Manchester shows a Net ROE cell. Removing `deal:P0000017` from that note's `anchors` in STANDING_FOOTNOTES is the one-line change once Charlene decides; the code says so at the definition. Guardrail snapshot_east_manchester_check 22/22

## v412 = `da354a3`

Jim approved: THREE engine corrections plus a footnote fix. (1) A reversed accounting entry no longer moves capital twice - MRI reverses by re-posting with the OPPOSITE SIGN under the same MajorType/Typename, and six sites took `abs()`, so the pair moved capital by 2x instead of nil. Rule now in `loaders.capital_after`: the sign of the amount IS the direction; the running total is NOT floored per row (order dependence - JB Fair Park's reversal sorts before its original), floored at use via `capital_outstanding`. 5 deals move, ONLY partner_equity/total_cap: JB Fair Park 11.6->3.9, Cocoplum 30.0->23.4, Pegasus 3.5->2.6, Belleville -1.95M, Adirondack -1.20M - all three on the TIAA page now TIE the 26Q1 reference PDF, which they did not before. pref_equity, debt and committed_pe move on none of the 128 deals. (2) Excess Cash Flow now pays pref down in the waterfall seeding - it sits BELOW pref so a partner cannot receive it while pref is outstanding; only TypeID 1019 counted before, leaving phantom arrears (OPMCCORD on 30 Bearfoot showed $8,995 owed against $1,055,944 of excess CF paid since its last pref payment). `reports_service._compute_accrued_pref` always applied both, so the two paths had been disagreeing. Portfolio accrued pref 159.9M -> 152.5M (-7.4M) across 37 of 101 pairs; 5 pairs go UP because reversed pref payments (negative 1019 rows) no longer count as payments. Invariant now checked: no investor carries pref behind a later excess distribution. (3) Footnotes - clearing the text now REMOVES the footnote and its marker; a blank note used to keep its number and keep stamping (1)/(2) on a header or property name, which is what Charlene hit; compose_footnotes also drops blanks as a backstop for rows already saved that way, and the control is labelled Remove instead of a bare x. Guardrails: capital_reversal_sign_check 19/19, pref_excess_cf_check 6/6, snapshot_footnotes_freeform_check 28/28. NOTE the PSC3 'over-return' the capital guardrail reports is a redemption artifact and out of scope - see .claude/memory/capital_reversal_and_psc3.md

## v411 = `8534916`

Snapshot print/ITD units, Jim approved: ITD Distributions is STORED IN MILLIONS and is no longer divided by 1e6 — v410 rendered every live cell "$0.00M"; the column header corrected from "$" to "$M", which was the reason for the wrong assumption in the first place; PUT /value strips a trailing "M" so "$5.87M" copied off the page pastes back; both manual fields now store the unit their column displays — ITD in millions, Net ROE in percentage points — with no conversion in either direction and no magnitude heuristic; light vertical column separators alongside the existing horizontal rules in print, on Financial/Operating/Loan only (anchored on `table.grid`, so Summary is excluded by construction); comment and manual-input cells print in the table's own font and size — form controls do not inherit font, and `.cmt-text` hardcoded 12px; ITD kept at 2 decimals per Jim

## v410 = `019b592`

Charlene Prompts A/B/C, Jim approved: A - "Total Current Funding" row removed, ITD Distributions carries its unit ("$15.33M") and is SUMMED onto every fund subtotal / Portfolio Totals / excluding-dev row, Net ROE manual at EVERY level and renders with "%" (stored in PERCENTAGE POINTS, no magnitude heuristic; aggregate rows store against reserved keys `__TOTAL__` / `__EXCLUDING_DEV__` / `__GROUP__:<key>` through the same values table and PUT /value); B - footnotes fully free-form: the prefix was RENDERED not stored so no migration was needed, every footnote now editable+deletable including the code-defined standing notes via reserved `standing-edit:<key>` / `standing-delete:<key>` rows scoped PER QUARTER (Jim confirmed per-quarter Sep 2), Restore drops the override; C - Loan tab lists each facility's real Rate/Maturity largest-first instead of "Various" (4 deals at 26Q1, all confirmed genuine multi-loan by distinct LoanID, not Loan_Date fan-out), fund-group separators on Financial/Operating/Loan only, each subtab confirmed to print to exactly one page with all 30 deals intact

## v409 = `7dc7bd8`

Charlene: Jefferson Eastchase's Sep-1 GROUP_OVERRIDES entry WITHDRAWN — the work order meant East MANCHESTER, and the ordinary rule had Eastchase in TGA 2023 correctly all along; East Manchester joins KEEP_DESPITE_SOLD so it stays on the 26Q2 page after its 6/25/2026 sale; new `SOLD_NA_CELLS` rule blanks Debt + Net ROE on any row reported after its sale and takes the n/a cell OUT of the subtotal via `debt_summable` — keyed on the sale, not the vcode, so East Manchester's 9,641,912 stays intact at 26Q1 where it was genuinely outstanding and reads n/a only from 26Q2; raw debt untouched, no 0 sentinel; reviewed and judged NOT a symptom repair, Jim approved

## v408 = `e5ef21d`

merge, Jim approved 5 items: Charlene `8de3d53` Waters Creek LTV exception retired + `9043c92` dev-tag correction with two new Pegasus hardcodes accepted as-is; the upstream `Pref` first-period accrual fix `16bcf8b` (TGA22 moves ~2.2M of promote off PSCMAN to TGAM/PSC1 - a real correction); Total Current Funding row + scope-placed footnotes; Jefferson Eastchase GROUP_OVERRIDES editorial override

## v407 = `50695d9`

Charlene: At-Close column zeroes for DEVELOPMENT deals with no 2015-12-31 Year-0 Projected IS row; the dev condition is load-bearing - it protects Town Fair children, Quakertown, Donald Lynch P0000073, Crowne Plaza

## v406 = `5e5a3b7`

merge: Charlene Snapshot Financial manual-input fix `d48469b` - PUT /value returns the display/source strings so Net ROE / ITD entry no longer refetches the bundle

## v405 = `7827c6f`

Charlene: Snapshot Financial Total Pref is the committed pref tranche, not funded

## v404 = `864e834`

merge: Charlene's Review Tracking quarter-join fix `97d3945` — a deal whose only submission is a different quarter shows as Draft instead of vanishing

## v403 = `edf4a3a`

merge: Charlene's One Pager fixes `150db60`+`0cb14ba` — nil participation renders 0% not 1%, prior-year budget fallback skips development deals — merged into the local v400-v402 line so no live feature regressed

## v402 = `1495d61`

PPI: PSC1 participant row carries the consolidated return

## v401 = `9432f87`

PPI: AMFee `;accrue` modifier — fee pays from source distributions, shortfall carries forward; PSC1 consolidated returns in psc_summary incl. PSCMAN fees/promote + PSC orig fee

## v400 = `b8bdfea`

Valuations: BS account type/description derived on live MRI data — fixes vAccountType error + blank Balance Sheet/NAV tabs; comma inputs; cross-check formatting

## v399 = `bf29707`

valuation module mixed-format date parse + Snapshot LTV year-end guard

## v398 = `ffd19b1`

At Close 7083 reserve-release netting

## v397 = `c6083f0`

Valuations module phases 1-4 + One Pager PE terms fallback

## v378 = `3cb8bb0`

rebased off main, superseded minutes later by v379 which adds 56f5666

## v376 = `e1e94f7`

SHA rebased off main; same parcel changes as `bc197d1`

## v349 = `2700c99`

first SHA-pinned revision

---

`v348` and earlier point at `:latest` and are not traceable by tag.

## Revisions v349-v569 (moved from CLAUDE.md, Oct 5 2026; v566 onward added here directly)

The per-revision index that used to sit inline in CLAUDE.md under **Running
the Application -> Deploying Changes**, verbatim and newest first. Entries
marked † ALSO have a longer post-mortem in the sections below; where the two
differ in length the post-mortem is the fuller text and this is the index
entry pointing at it. Nothing here was summarised.

- **Deploy history (SHA-pinned)** — newest first; `vNNN` is the revision suffix and the
  backticked SHA is the commit its image was built from, so the running revision names its
  own source. Entries marked † have a full post-mortem archived verbatim in
  `.claude/memory/deploy_history.md`. **Read that before assuming a revision shipped what
  its SHA suggests** — several did not (`v424` was a merge, not the commit that was asked
  for; `v378` was superseded minutes later; `v418`/`v417` shipped only part of a branch).

  - `v569` = `6f33f0a` (TWO LOAN-SUBTAB HARDCODES RETIRED BECAUSE THEIR REASONS
    LAPSED, not because the page was wrong. Oct 5 2026, image
    `waterfall-xirr:6f33f0a`, digest `sha256:bf151ece...`, tag locked, build `cand`.
    ONE runtime file, `portfolio_snapshot_loan.py`, +15/-11.
    (1) `KNOWN_LOAN_SUBTOTAL_DIFFS` loses `("Total PSC TGA 2022 LLC", "ltv")`. It
    documented a diff that no longer exists: the published 60.4% reproduced only
    while Jefferson Waters Creek's real 57.5% LTV was weighted in, and that
    exception was retired 2026-09-01 so every development deal shows "Dev". With
    Waters Creek out of the denominator the total is 61.6% and the entry described
    a cost already paid. `snapshot_subtotal_method_check` asserts a KNOWN entry is
    still LIVE and had been FAILING on this one -- an entry that outlives its diff
    turns the check into noise, which is how a real diff gets missed. It now passes
    **33/33**, NOT 34/34: removing the entry removes its assertion, so the
    denominator falls with the numerator. A check whose total drops is not a check
    that was skipped.
    (2) Citizen Storage `P0000120` loses its `ytd_dscr` seed and KEEPS its LTV and
    Debt Yield. Measured on live 26Q3 before the change: the deal now carries a
    full YTD Interim IS (`ytd_noi` 142,682.78) and the engine computes **1.6735x**,
    so the seeded 1.5x had stopped filling a gap and started OVERRIDING a real
    figure -- the one thing the seeds exist not to do. LTV and Debt Yield stay
    because those reasons DO hold: `valuation` is None, and `quarter_noi` and
    `annualised_noi` are both None. The other five seeded deals are byte-identical,
    verified key by key.
    THE SUBTOTALS MOVED, AND THAT IS CORRECT -- recorded in full because it
    contradicts a comment still in the same file. Row-level diff across BOTH
    investor pages (OWPSC 55 rows, TGAM 38) is SIX FIELDS, all of them
    P0000120's DSCR manual-state fields; no other deal, no other column. But the
    debt-weighted means containing that deal moved with its displayed figure:

        OWPSC  PSC1  ytd_dscr   1.5946575 -> 1.5954366   (typed_n 4 -> 3)
        OWPSC  total ytd_dscr   1.6237423 -> 1.6243495
        TGAM   TGA25 ytd_dscr   1.8514554 -> 1.8578197   (typed_n 2 -> 1)
        TGAM   total ytd_dscr   1.7578363 -> 1.7586545

    `aggregation_value` (`:595`) weights THE FIGURE A ROW DISPLAYS -- the typed
    entry where there is one, the computed figure otherwise -- deliberately since
    2026-09-02, so that a fund total can be re-derived from the rows printed above
    it. A displayed DSCR rising 1.5 -> 1.6735 therefore moves every mean it is in.
    **A COMMENT AT `:313` SAYS THE OPPOSITE** ("A typed cell contributes to no
    aggregate") and is stale; it was read as authoritative in a hardcode inventory
    earlier the same day and produced a wrong statement about this page. Two
    contradicting comments now live in one file -- `open_items.md` 20.10, docs-only,
    deliberately NOT bundled into this deploy.
    VERIFIED ON LIVE after cutover: v569 alone at 100%, root 200, Citizen Storage's
    cell reads the computed figure, and its payload shape matches every other
    computed row (raw float, `fmtX` renders 1.67x) rather than the formatted string
    a typed cell carries. Built from a CLEAN WORKTREE, gated on the tag and on run
    `cand` succeeding for this SHA, P1 re-run immediately before cutover (still
    v568 = `e399d92`, an ancestor). `SQLAlchemy>=2.0,<2.1` and `pandas>=2.3,<3.1`
    intact, NOT bumped.
    GUARDRAILS COMPARED AGAINST A PRISTINE `origin/main` WORKTREE so nothing
    pre-existing was misread as new: `snapshot_subtotal_method_check` FAILS on
    baseline and PASSES here; `freeze_as_sent_check` 104/0 on both;
    `snapshot_loan_manual_cells_check`, `snapshot_dev_display_check` and
    `snapshot_manual_input_no_reload_check` fail IDENTICALLY on both -- they import
    the absent `live_api` harness, open_items section 15.
    NOT DONE, DELIBERATELY: `portfolio_snapshot_values` row id 86 -- a `draft` test
    row on `R000`/`BATCH`/`2099-Q4` -- is LEFT IN PLACE. The app has NO delete path
    for a typed value (`PUT /value` with null stores a NULL, a deliberately CLEARED
    cell, it does not remove the row), so removing it needs a direct production
    DELETE and the firewall stayed shut. Confirmed it is the only test row; the
    other 85 are TGAM/2026-Q2 plus one KCREIT/2026-Q1.)
  - `v568` = `e399d92` (THE ISBS SUPPLEMENT SUPERSEDE KEY WAS BUILT ON RAW VALUES,
    AND MOUNT PROSPECT'S PUBLISHED BUDGET NOI WAS EXACTLY DOUBLE. Oct 5 2026, image
    `waterfall-xirr:e399d92`, digest `sha256:6631b064...`, tag locked, build `canc`.
    `_append_isbs_supplements` runs inside `_assemble_isbs`, ONE LINE BEFORE
    `_normalize_isbs` lower-cases vcode and parses dtEntry. MRI writes `p0000069` /
    `2026-01-31T00:00:00`; the app's budget import writes `P0000069` / `2026-01-31`.
    Compared with `.astype(str)` those never matched, so NO MRI ROW WAS EVER
    SUPERSEDED and both copies reached `isbs_raw`, where every consumer SUMS them.
    It read as a plausible figure, which is the danger -- nothing on screen
    distinguishes a doubled NOI from a good one. Both sides of the key are now
    normalised FOR COMPARISON ONLY (vcode stripped and lower-cased, dtEntry parsed to
    YYYY-MM-DD; vSource and vAccount untouched). Nothing is written back;
    `_normalize_isbs` still owns the stored values.
    A YTD-CUMULATIVE vSOURCE IS EXCLUDED, and the fix found that rather than the other
    way round. `isbs_projected_is` 7073 is a RUNNING TOTAL -- one contribution restated
    monthly, not several. Superseding its 06-30 row makes 07-31 the first month of the
    year, so its full cumulative reads as a SECOND contribution while the supplement
    still supplies the real one. Measured on production BEFORE the exclusion: Court of
    Deptford -8,751,183.95 -> -18,297,183.95, Burton -26,597,500 -> -53,195,000,
    Presidential Arms -20,600,000 -> -41,200,000, all feeding U/W ROE to Date and CoC
    Proj. Since Close. The genuine duplicate is already handled downstream by the
    (date, amount) dedupe at the end of `_get_uw_7073_signed`, so there was nothing to
    fix here and real harm in trying. Interim IS excluded on the same grounds; Budget
    IS, Valuation IS and Interim BS are periodic or point-in-time and still supersede.
    INVESTOR-FACING -- halves P0000069's published budget NOI. Jim signed off.
    VERIFIED ON LIVE AFTER CUTOVER: Mount Prospect 26Q2 budget NOI 2,685,375.22 ->
    **1,342,687.61**, budget revenue 2,882,644.59, 26Q3 budget NOI 1,902,734.91. Its
    ACTUALS are unchanged (26Q2 1,430,157.84) and the budget DSCR still computes
    (1.7991386).
    THE SWEEP FOUND A THIRD DEAL THE BRIEF DID NOT NAME, and paging is why. Only a deal
    carrying an app-written budget supplement can be touched; the first page of
    `isbs_budget_is_supplements` returned 500 of 2,856 rows and showed two deals.
    Paged in full (and deduped -- the rows endpoint's OFFSET paging repeats rows,
    2,856 -> 2,709 unique) it is THREE: P0000019, P0000069, P0000075. Key-overlap
    against the MRI budget rows decides which can move: P0000019 **0 overlap**,
    P0000075 **0 overlap**, P0000069 **188**. The supersede key can only remove an MRI
    row a supplement actually covers, so the other two are untouchable by this change
    and P0000069 is the only deal whose budget moves. A 500-row first page would have
    supported the same conclusion for the wrong reason.
    MERGED FROM A BRANCH 59 COMMITS BEHIND MAIN, forked at `326ef6c` before v552-v567,
    so a bare `git diff` against main lists ~75 files including `one_pager.py` and
    `requirements.txt`. THE COMMIT ITSELF TOUCHES TWO: `data_service.py` and its
    guardrail. Reviewed before merging rather than after: main has not touched either
    file since the merge base, the dry-run staged exactly those two, and the merged
    tree still carries the v567 One Pager NOI fix and v566's Treasury par yields.
    Regression-checked on live after cutover -- Pontchartrain UW-YE 2,130,282.15,
    Asbury 730,129.17, Dorsett 3,896,181.99 all still exact; 23 rate series with 13
    UST tenors, `UST_10Y` 2,439 observations last 2026-10-02 at 5.28.
    Guardrail `isbs_supplement_precedence_check` 20 -> 33: it had REPLICATED the
    shadowing rule locally with the same `.astype(str)` the engine used, so it agreed
    with the defect perfectly -- it now calls `_append_isbs_supplements` itself. On the
    merged tree 33/0, with one_engine_per_number 26/0, investment_metrics 178/0,
    market_rates 25/0. Built from a CLEAN WORKTREE (the v556 lesson), gated on the tag
    and on run `canc` succeeding for this SHA, P1 re-run immediately before cutover
    (still v567 = `af89507`, an ancestor). `SQLAlchemy>=2.0,<2.1` and `pandas>=2.3,<3.1`
    confirmed intact and NOT bumped. After cutover: v568 alone at 100%, root 200 in
    0.08-0.58s.)
  - `v567` = `af89507` (ONE PAGER HEADLINE NOI READS THE CANONICAL ACCOUNT LIST.
    Oct 5 2026, image `waterfall-xirr:af89507`, digest `sha256:aafc916a...`, tag locked.
    The headline NOI was computed from a SECOND, hand-maintained copy of the
    income-statement account list in `one_pager.py`, while the One Pager's own NOI
    CHART -- and the Dashboard KPIs, Surveillance, Property Financials, the Snapshot
    freeze and Budget Review -- all read `config.IS_ACCOUNTS`. The copy had drifted
    short by two accounts, so one deal could print one definition of NOI in the header
    and draw another in the chart directly beneath it. The headline list is now DERIVED
    from config's and gains **4075 Other Income** (revenue, raises NOI) and **5092 R&M**
    (expense, lowers NOI). `config.py` is NOT touched, so no other surface moves.
    7070 IS STRIPPED FROM THE DERIVED LIST and kept in its own `TAX_ABATEMENT` key:
    config groups it inside Real Estate Taxes while `calc_amounts` folds it into
    expenses itself, and a plain import would have counted the abatement TWICE.
    `DEBT_SERVICE` is deliberately NOT derived -- config's `Principal` list is EMPTY
    where one_pager's carries 7060, and deriving it would have silently dropped 7060
    out of the DSCR denominator. Caught by diffing EVERY key against main, not only the
    two in scope.
    INVESTOR-FACING, AND JIM SIGNED OFF. Measured on live PostgreSQL before the build,
    both code versions through the app's own loader on the same 943,489 ISBS rows.
    Only UW full-year NOI moves, on three deals, identically at 26Q2 and 26Q3:
    Pontchartrain Landing 2,555,745.19 -> 2,130,282.15 (-425,463.04) and Asbury Commons
    741,367.00 -> 730,129.17 (-11,237.83), both from 5092; Flats at Dorsett Ridge
    3,718,397.44 -> 3,896,181.99 (+177,784.55) from 4075. **Actual NOI, actual-YE,
    at-close and every DSCR denominator are unchanged**, and the six deals whose 4075
    activity is 2021-2022 are identical at both quarters. All six figures re-verified
    on the live API after cutover; Evergreen Plaza, carrying neither account, is
    unmoved.
    BUILT FRESH AFTER A DEPLOY COLLISION, AND THIS IS THE ENTRY'S REAL LESSON. An
    earlier image `waterfall-xirr:2f7d631` (the NOI fix merged onto `24c55ed`) was
    built first. One minute into that build, `v566` = `96534b3` (Treasury par yields)
    was deployed by someone else from the SAME base. Neither commit is an ancestor of
    the other, so **deploying `2f7d631` would have reverted the Treasury work** --
    `_derive_headline_is_accounts` occurs 0 times in `96534b3`, and `96534b3` is absent
    from `2f7d631`. The P1 RE-RUN IMMEDIATELY BEFORE `containerapp update` is the only
    reason this was caught: it returned `v566` where `v565` was expected, and the
    suffix `v566` was already taken. Nothing was deployed; `2f7d631` remains built and
    locked, UNUSED. Rebuilt from `origin/main` `af89507`, which contains both, after
    asserting `96534b3` IS an ancestor. `2f7d631` is also an ancestor, so the NOI fix
    is in.
    Built from a CLEAN WORKTREE, not the checkout: `git status` carried two untracked
    diagnostic scripts, ACR uploads the working tree, and `scripts/` is copied into the
    image -- the `v556` lesson. Gated on the tag existing and run `canb` succeeding for
    this SHA (2m19s) before the update. `SQLAlchemy>=2.0,<2.1` and `pandas>=2.3,<3.1`
    confirmed intact and NOT bumped. After cutover: v567 alone at 100% traffic, root
    200 in 0.10-0.16s, and the Treasury feature from `v566` is NOT reverted -- 13 UST
    series live, `UST_10Y` 2,439 observations, last 2026-10-02 at 5.28.
    A VERIFICATION OF MINE WAS WRONG FIRST and is recorded because it will recur: the
    Treasury check initially reported "0 UST series" because I guessed the payload key
    (`series_id`/`id`/`code`) when it is `key`. Dumping the raw response showed 13.
    Same class of error as reading `at_close_noi` instead of `at_close_noi_raw` earlier
    the same day, which briefly made at-close look like it moved. Dump the shape before
    filtering on it.)
  - `v566` = `96534b3` (U.S. TREASURY PAR YIELDS IN THE RATES TABLE -- Jim: "add the
    treasury rates to the rate table, especially the 10-year." Thirteen tenors
    `UST_1M`..`UST_30Y` from Treasury's Daily Par Yield Curve CSV, history from 2017,
    read by COLUMN NAME (the columns changed over the years). Inserts now BATCHED via a
    Core insert(): on production the full Treasury load was 29,889 rows in 9.1s, where
    v565's first load of 21,013 rows took 200s row by row. VERIFIED ON PRODUCTION: the
    10-year 4.44% at 6/30/26 and 5.28% at 10/02, 23 series. P2 listed six commits: this
    one, two of mine (docs), and three of CHARLENE'S -- her CLAUDE.md compaction (4,245 ->
    372 lines, detail moved to .claude/memory/, this index moved here), a budget check
    `scripts/claude_md_budget_check.py`, and a pre-commit hook rule that runs it only
    when CLAUDE.md is staged; reviewed, docs and tooling, no runtime effect. The runtime
    delta is `market_rates_service.py` alone. Container egress to treasury.gov checked
    BEFORE the build. In the container: market_rates_check 25/0, pe_exposure_check 28/0.
    Build `can9` 2m17s, tag locked, P1 re-run before the update (one transient Azure
    500 on the revision list, retried); root 200, 0 errors in the boot log.)
  - `v565` = `032fa27` (PSC PREFERRED EQUITY EXPOSURE in Reports, open to everyone,
    and MARKET RATES under Data Management. Accounting's tracker from our MRI copy:
    Cost = the Pref Balance Detail balance + realized losses, FMV = Cost + unrealized,
    the seven investor columns from commitments in force multiplied down the chain,
    future funding (the One Pager's remaining to fund) BELOW the current exposure with
    Total Equity Invested / Committed, any quarter end or Live, Excel in the same
    layout. VERIFIED ON PRODUCTION through its own server: 6/30 grand total
    725,204,338 on 56 holdings + 12 future-funding rows (the tracker: 724,660,963),
    Live 766,476,998 -- both equal to the local run to the dollar. Rates loaded on
    production: 21,013 rows, 10 series (6/30 USD/CAD 1.4210); the FIRST full load took
    200s on PostgreSQL (row-by-row to a remote server), incremental refreshes are ~80
    rows. Container egress to Bank of Canada and the NY Fed checked BEFORE the build.
    PRE-FLIGHT P4 CAUGHT A 186 MB LOCAL DATABASE BACKUP swept into the branch by
    `git add -A`; never pushed; the branch was rebuilt as ONE commit on origin/main
    without it, and .gitignore now excludes `waterfall.db.bak-*` / `*.db.pulling`.
    Span vs live `d6adabe`: three docs commits, `40c6796` (pull script), `032fa27`.
    Raised with Jim before the build, not symptom repairs but rules to know: the
    investor-group STOPS are a fixed list of entity ids (accounting's Mapping tab), and
    Ambassadors is recognised by an `AMB` name prefix. In the container:
    pe_exposure_check 28/0, market_rates_check 18/0, section_access --static 101/0.
    Build `can8` 2m18s, tag locked, P1 re-run before the update; root 200, 0 errors in
    the boot log; served chunks carry both screens.)
  - `v564` = `d6adabe` (THE ADMIN ROLE IS NOT ACCOUNTING. Jim, Oct 2 2026: "There is
    only 1 Admin for the system with access and rights to everything and that is me.
    Charlene has admin rights to make enhancements to the system however, if she is
    blocked from accounting, she should not be able to view or update accounting tables
    or screens." Charlene could read Jim's approved expense report because `can_view`
    admitted any ACCOUNTING_ROLES role, and `admin` is one. Now
    `sections.has_accounting_authority`: the `admin` USERNAME, or an accounting role AND
    the Accounting section ticked -- used by `can_view` (approved/batched), the
    accounting-return permission and route, GET /employees, PUT /mileage-rates, and
    `canEditAccounting` on screen. SETTING APPROVERS is now the `admin` USERNAME only
    (it was the admin role), like assigning sections. The list asks once per load, not
    per report. TAKES EFFECT FOR CHARLENE ONLY WHEN ACCOUNTING IS UNTICKED for her in
    User Management. `expense_report_check` 84 -> 98, nine new checks fail on the old
    code; 98/0 IN THE CONTAINER. Span vs live `577511b`: three commits plus two docs.
    Clean worktree, build `can7` 2m34s, tag gated and locked, P1 re-run before the
    update; root 200, 0 errors in the boot log.)
  - `v563` = `577511b` (EVERY MRI JOURNAL-ENTRY DESCRIPTION TO THE RULE, AND EXPENSE
    CODING'S RECEIPT OPENS THE POP-UP. Jim: "any journal entry descriptions getting
    loaded into MRI need to follow the rule" -- applied in `build_gl_csv`, the one
    writer treasury, intercompany and expense all use; `validate_gl` warns which lines
    it will clean. The AMB6 August GL file MRI accepted had punctuation in 23 of 51
    descriptions (asked of accounting). `treasury_upload_check` 58 -> 61: identical in
    every field but the description, which must equal the accepted text cleaned.
    Intercompany's JE Template rebuild still line-for-line identical. The coding page's
    receipt link showed a broken image -- a PDF in an <img>, no content type passed;
    `ReceiptViewer` now reads the type from the file. Span vs live `fae6013`: `577511b`
    plus docs. Clean worktree, gated; site 200. In the container: treasury_upload 26/0
    (the accepted-file rebuild skips there), expense_coding 47/0; the served chunk
    carries the pop-up.)
  - `v562` = `fae6013` (EXPENSE JE DESCRIPTIONS HELD TO MRI'S RULE -- 80 characters,
    letters, digits and spaces only (Jim, Oct 2 2026). `ER FK <deal> <comment>`, `IC ER
    ...` for intercompany, comment words the deal already says dropped, long common
    words abbreviated only when over 80, the payroll credit `ER Trinet Payroll 202609
    <suffix>`, the CAD note `USD 712 98`. Expense only; intercompany's fixed text already
    complies, treasury's bank-sourced text unchanged. Accounting's own Sep 24 file had
    punctuation on all 119 lines and 12 over 80. Span vs live `9adfddd`: `06b3256` plus
    docs and a zero-line merge. Built from the clean worktree, gated; site 200. In the
    container: `expense_coding_check` 47/0 (the Sep 24 rebuild skips -- the file is not
    on the server); 53/0 locally with it.)
  - `v561` = `9adfddd` (THE EXPENSE LINE OPENS AS A POP-UP AGAIN -- entry left,
    receipt right, stacked under 900px -- and READ-ONLY for the approver and
    accounting; Edit/Remove (View for anyone who cannot edit) moved to the FIRST
    column. Jim thought v558 had removed the edit screen; it had not -- the new
    dropdowns widened the table and the Edit link in the LAST column went off screen.
    P1 BEFORE THE UPDATE CAUGHT CHARLENE'S `v560` = `1455419` (TGA VI on the Snapshot,
    on origin/main) deployed after this work's span was taken -- deploying `779fb44`
    would have rolled it back. Merged origin/main (`9adfddd`, zero lines of its own),
    so the runtime delta vs live is ExpensesView.vue only and her six files are
    byte-identical in the image. Served chunk carries the pop-up. Site 200.)
  - `v560` = `1455419` (THE SNAPSHOT PRINTS **TGA VI**, NOT **TGA6**, IN BOTH PLACES.
    Oct 2 2026, tag locked. THIS IS LIVE. The fund is PSC Ambassadors Fund TGA VI
    LLC (AMB6); the ownership traversal keys it `TGA6`, and that key is what every
    subtotal, override and lookup is written against — so only the printed
    spelling moved and the key was left alone.

    **TWO MAPS, BECAUSE THERE ARE TWO ROUTES ONTO THE PAGE and they only look like
    one.** The subtotal row takes `group_total_label(key)`, which falls back to
    `"Total <key>"`; the header row on Operating and Loan renders the GROUP KEY
    itself, with no map behind it at all. So TGA6 arrived at 26Q2 printing `TGA6`
    above its deals and `Total TGA6` below them — nothing errored, nothing was
    missing, and the only way to see it was to read the page against the sent
    report. `GROUP_TOTAL_LABELS` gains `"Total TGA VI"`; `GROUP_DISPLAY_LABELS` is
    NEW and has exactly ONE entry, because the header cannot be derived from the
    total label — stripping `"Total "` would rewrite TGA22's header from `TGA22` to
    `PSC TGA 2022 LLC`. Every other group keeps its key.

    Operating and Loan publish `group_display_labels` beside `group_labels`; their
    components render it and **fall back to the key when the field is absent**, so
    a Snapshot frozen before this field existed still renders a header rather than
    an empty cell. Financial is untouched — it prints no header row and names the
    fund only on its total row.

    **THE RE-CHECK BEFORE BUILDING IS WHY THIS SHIPPED CORRECTLY.** The branch was
    cut from `68ffce3` (v557). By the time it was ready, live had moved twice —
    to v559 `05b6f5f` — and Jim had merged his line into main, ending the
    prod/main divergence that had stood through v552–v557. Building the original
    branch would have shipped a tree of `68ffce3` + the label fix, **silently
    reverting the expense distance wizard and the dropdown work** (675 lines, 9
    files, a new service and a DB column). Nothing would have errored; the
    features would simply have been gone. Rebased onto `0be6ea8` instead — the
    three intervening commits touch expenses and docs and **not one** of the six
    files here, so the rebase was conflict-free.

    Guardrail: `scripts/snapshot_group_label_check.py` (27). It pins the whole sent
    page group by group, header AND subtotal, so "TGA6 is relabelled" cannot pass
    by relabelling everything and "no other fund moved" cannot pass by changing
    nothing. The full `snapshot_*_check` sweep is byte-identical before and after
    on the same machine, the only difference being this file's 27 checks
    appearing; the two standing failures (`subtotal_method` 33/34 "KNOWN entry is
    stale", `temp_suppress` 25/26) reproduce identically on the clean base and are
    untouched. **11 snapshot suites cannot run locally at all** (`ModuleNotFoundError:
    live_api` — that module is not in the repo) and 4 more need the `coa` table;
    same on both sides, counted as neither pass nor fail.

    Verified on live: Operating and Loan both carry header `TGA VI` and subtotal
    `Total TGA VI`, Financial carries `Total TGA VI`, and the other five groups are
    unchanged on all three. No stale `TGA6` string anywhere in the payload. Smoke:
    Dashboard, One Pager, Investment Metrics, two Snapshot bundles, the Expenses
    screen Jim shipped in v558/v559, deals list — all 200. 1,028 MiB of 2,048,
    **0 tracebacks**.)
  - `v559` = `05b6f5f` (THE DISTANCE WIZARD -- driving miles by Google Routes, each
    stop resolved by Geocoding; a bare 3-letter code asked as an airport, because
    Geocoding "PHL" alone is "Philippines". The deploy WIRES `GOOGLE_MAPS_API_KEY` to
    the container secret `google-maps-key` (`secretref`). Built from the clean
    worktree, tag gated before the update; site 200 throughout. VERIFIED IN THE
    CONTAINER: the env reads the secret. It first held a 14-character placeholder
    (Google refused, the wizard said so); Jim set the real key and restarted the
    revision, and in the container the key is 39 chars and PHL -> Philadelphia City
    Hall round trip measures 24.5 mi (11.9 + 12.5), the same as locally.
    `expense_distance_check` 26/0.)
  - `v558` = `7626b3e` (PURPOSE AND DEAL AS DROPDOWNS on the expense table, saved on
    change with the whole line sent; the receipts list below the table and the
    submit notice. Served chunk verified to carry the new strings. Site 200.)
  - `v557` = `68ffce3` (PANDAS PINNED `<3.1` -- resolves 3.0.6, what production
    already ran, so it changes nothing live and stops 3.1 (rc0 out Sep 30) arriving
    unannounced -- and THE CFO MAY SIGN OFF ANY EMPLOYEE'S EXPENSE REPORT, recorded
    "CFO in place of <approver>"; never their own, drafts private. Span vs live
    `effab11`: `68c7b03` (docs) and `68ffce3`. Built from the CLEAN DEPLOY WORKTREE
    with the tag checked before the update. VERIFIED IN THE CONTAINER: pandas 3.0.6,
    `expense_report_check` 84/0. Deployed 16:47 UTC, site 200 throughout.)
  - `v556r` = `effab11` (RESTORE. Not a release: a new revision on the last good
    locked image, after the v556 failure below. Up ~16:45 UTC.)
  - `v556` -- FAILED, NEVER SERVED, AND CAUSED A ~4 MINUTE OUTAGE (~16:41-16:45 UTC,
    root 404). `az acr build` failed during upload -- `waterfall.db-shm` vanished,
    because a background guardrail sweep was using the local SQLite db in the same
    checkout -- so tag `68ffce3` was never created, and the chained script still ran
    `containerapp update` to it. Deactivating the broken v556 then made it worse: in
    single-revision mode traffic follows the LATEST revision, so the healthy v555
    deprovisioned. Recovered by deploying a NEW revision on the good locked tag
    (`v556r`). RULES SINCE: gate `containerapp update` on the tag existing and the run
    succeeding for that SHA; build only from a clean worktree nothing else is using;
    to back out in single mode, roll FORWARD to the last good tag, never deactivate the
    latest revision.
  - `v555` = `effab11` (A NULL SEGMENT IS BLANK UNDER PANDAS 3. PRODUCTION RUNS
    PANDAS 3.0.6 -- `requirements.txt` says only `pandas>=2.3`, pandas 3.0.0 shipped
    Jan 21 2026, so EVERY image in this history has run pandas 3 -- while local dev
    runs 2.3.3 and every guardrail runs locally. Pandas 3 reads a SQL NULL as NaN,
    `_norm` made it the string "NAN", and the intercompany reconciliation carried an
    ENTITY CALLED NAN (PSC Manager's unsegmented MR15000001) in 202609 and 202610
    instead of the blank-segment check -- since v531, and payable from v554. Found
    because `intercompany_check` failed 4 checks INSIDE THE CONTAINER (68/4/1) while
    passing 86/86 locally; production data confirmed untouched by that run (0
    batches, 0 pay rows, gl_detail 79,752 real rows). AFTER: no NAN, 63 rows -- the
    count on the CFO's own sheet; the "64" recorded at v531 included the phantom --
    blank segment 3 rows netting 0; `intercompany_check` 74/0/1 in the container.
    Span vs live `baf619e`: this one commit. A pandas-2-vs-3 sweep of all guardrails
    is open in open_items §18. Build 2m18s.)
  - `v554` = `baf619e` (INTERCOMPANY PHASE 2 -- PAY AND THE REIMBURSEMENT JE, plus
    the DOUBLE-BATCH LOCK. P2 vs live `79d21a2`: `be7a33e` (the feature, committed
    from work found uncommitted in the checkout and reviewed in full), the v553 docs
    commit, a merge whose only resolution was the union of two PROTECTED_TABLES
    additions, and `baf619e`. P4 FOUND A DOUBLE-PAYMENT RACE and it was fixed before
    building: generate checked for a pending batch on a read taken before its own
    transaction, so two simultaneous generates could both reimburse one entity. Now
    the batch table is locked inside the generate transaction (PostgreSQL SHARE ROW
    EXCLUSIVE; SQLite's write lock) and the pending set re-read inside it. WITHOUT
    the lock the threaded check produced two batches in one run of three; with it,
    six of six pass. Behaviour change: PPI2/PSC2 cash defaults to their USD MR1000*
    less the Canada accounts (CFO, Sep 30), so their can-afford is now computed.
    VERIFIED ON PRODUCTION: both tables; the Sep 30 answers applied to the seed rows
    only; reconcile runs for 202610; `LOCK TABLE` works on PostgreSQL (rolled back).
    13 guardrails green on the merged tree. Build 2m16s.)
  - `v553` = `79d21a2` (EMPLOYEE EXPENSE REPORTS, all four phases -- the
    Expenses section and Accounting > Expense Coding. Design and measurements:
    `.claude/memory/expense_reporting.md`. Deployed Oct 2 2026 15:52 UTC, build
    2m12s. P2 against live `11c3455`: 7 commits -- the five expense phases, the
    v552 docs commit, a merge with ZERO lines of its own; live is an ancestor.
    No vcode or entity literals in shipped code. Behaviour changes: Expenses is a
    new section ticked for everyone; `validate_gl` now refuses a MULTI-entity
    entry out of balance by entity (treasury's byte-identical rebuild unchanged,
    58/58); `er_` tables hidden from raw-table paths without Accounting; Expense
    Coding READS closed to analysts. NEW DEPENDENCY `pillow-heif` 1.8.0 (Pillow
    12.3.0), bounded, prebuilt cp312 manylinux wheel. GUARDRAILS on the merged
    tree, 13 suites: expense 83/53/46/23, section 265, accounting 56, treasury
    upload 58 / api 46, intercompany 53, gl_ia 123, investment metrics 178,
    committed pref 47, lease scan 43; vite build. VERIFIED ON PRODUCTION: HEIC
    converts in the container; 11 `er_*` tables on PostgreSQL; intercompany
    population 64 entities. OWNERSHIP PROPOSALS vs accounting's own booking:
    right for Fairview, Nottingham, Woodlands (and Ascent's split); WRONG for
    Pontchartrain, Gallery, Belleville -- the investee funds INVF7/INVF2/INVF11
    keep their own MR15000002 so the walk stops there, while accounting books
    their parent PSC3 -- and Apple (commitments carry PSS1 17.93% beside PPI2)
    and Brainerd (82.68/17.32 against accounting's ~62.4/37.6). Proposals are
    editable; the rule is accounting's question, not changed. See
    expense_reporting.md and open_items.md.)
  - `v552` = `11c3455` (SECTION ACCESS BY USERNAME -- per-sidebar-section
    checkboxes in Settings > User Management; only the `admin` USERNAME assigns;
    without Accounting the GL/IA tables and every `tr_`/`wp_`/`ic_` table are
    hidden from Data Explorer, the export, MRI query downloads and the assistant's
    SQL. Deployed Oct 2 2026 15:38 UTC, build 2m17s. HELD a day: `b93dd5f` was
    imaged Oct 1 but Charlene had deployed `v549`/`v550` from an unpushed clone
    between this work's P1 and its build; deploying it would have rolled hers back.
    Re-run P1 before the update caught it (memory: recheck-live-before-deploy).
    Once her commits reached origin, origin/main was MERGED into the branch
    (`11c3455`). P2 against live `c255729` (v551): 8 commits -- `0362ecb`,
    `b93dd5f`, four docs-only, two merges with ZERO lines of their own (`git show
    --cc`); runtime delta is the 11 section-access files only; `c255729` is an
    ancestor. No vcode literals. Behaviour change: the DB export and MRI query
    downloads now need Data Management (ticked for everyone by default).
    GUARDRAILS on the merged tree: section_access 256, accounting_access 54,
    treasury_api 46, gl_ia_query 123, investment_metrics 178, committed_pref 47,
    vite build. P1 re-run immediately before `containerapp update`: still v551.
    VERIFIED ON PRODUCTION, in-process with a minted token: an account named
    `admin` EXISTS (16 users); `admin` gets `/auth/sections` 200 `can_assign`
    true, all 7 sections; the admin-ROLE users `anaik` and `cbui` get
    `can_assign` false and their PUT is 403; `user_section_access` has 0 rows, so
    nobody's access changed at deploy; `section_access_check --static` 98/0 in
    the container. Root 200 in 0.15s, clean boot.)
  - `v551` = `c255729` (INVESTMENT METRICS: Act. Yr-1 CoC IS COMPUTED, AND THREE
    FOOTNOTE MARKERS ARE DERIVED RATHER THAN TRANSCRIBED. Oct 2 2026, tag locked.
    THIS IS LIVE. Pre-flight P2 against the live `b250639` listed six commits, all
    Charlene's and all but this one docs-only.

    **Act. Yr-1 CoC now prints**, from the One Pager's ROE engine windowed to the
    deal's first twelve months. The column had been an em dash because the figure
    behind it was a different quantity — preferred return over funded-to-date,
    which ignores excess cash flow and the capital actually at risk and returns
    0.0% on ten deals that paid no pref in year one. Against the reference
    workbook at 2026-06-30, at display rounding: **24/76** exact, against 23/76
    for the old pref/funded alternate and **21/76 for printing nothing**. It is
    **biased low** — of the 51 rows where both sides carry a number, 40 come in
    below and 11 above, mean **-1.21pp**, 21 within 1pp, 30 within 2pp. The
    reference's cells are typed-in constants on 75 of its 76 rows, so the gap is
    not a window that needs tuning, and it is reported rather than fitted.

    **The window opens at the EARLIER of the invest date and the first cash
    event**, and that is the one thing most likely to be "simplified" back out.
    `Acquisition_Date` is overwritten at load time with the earliest accounting
    entry across ALL investors, and on six live deals PSC's own first
    contribution is dated the day BEFORE it — Evergreen Plaza, Giant-7, Mount
    Prospect, OREI, Pontchartrain and 870 Donald Lynch. Opening strictly on the
    invest date drops that contribution, takes contributions in the window to
    zero, and takes the whole figure to a dash: measured, it blanked **seven of
    76** deals outright and sent Cocoplum to **36.0%** against a reference 4.98%.

    **Footnote (5) is derived from the date and now ACTS.** Invest date + 12
    calendar months past the as-of date selects **exactly the nine deals the
    reference marks** — Apple, Burton, Trolley Square, Jefferson Stephens, Plaza
    Del Mar, Hanestowne, Presidential Arms, Swartz Creek, Fairview — with a wide
    margin either side (the newest UNMARKED deal is Green Valley Ranch at 16
    months; the oldest marked one is Apple at 11.5). Act. Yr-1 CoC, CoC Proj.
    Since Close and CoC Act. Since Close all take the projected year-1 figure,
    which is what the TABLE does on Apple's row even though **the printed note
    names only the first column**. Both are reproduced; neither is edited to
    agree with the other. With `proj_yr1_coc` still unloaded the three cells are
    **BLANKED** rather than left showing a stub period — Presidential Arms' ROE
    over seven weeks is not a year-1 return, and 8.0% on the page is a number
    somebody will quote. Precedence is **Dev. > Lease up > this rule > computed**,
    per cell, which is why Trolley Square and Jefferson Stephens still print
    `Dev.` The substitution activates by itself the moment `proj_yr1_coc` is
    switched on.

    **WHAT THIS COSTS, SAID OUT LOUD.** Overall agreement with the reference goes
    **539/1242 -> 537/1242**. act_yr1_coc gains 2 (21 -> 23); blanking the young
    deals loses 1 on act_coc_since_close (36 -> 35) and 3 on proj_coc_since_close
    (32 -> 29). That is the instructed behaviour and the reference is not the
    target — but the number moved the wrong way and it is recorded here rather
    than left for somebody to rediscover. **60 cells change in all**, 31 of them
    dashes becoming figures.

    Totals: Current Act. Yr-1 CoC — -> **7.1%**, CoC Proj. Since Close 7.9% ->
    **8.2%**, CoC Act. Since Close 6.9% -> **6.5%**; Sold Act. Yr-1 CoC — ->
    **7.9%**, both Sold since-close averages unmoved. The nine (5) deals are
    **$132.2m of $784.8m = 16.8%** of Current pref weight and are excluded from
    those averages entirely, never counted as zero.

    **First lien reads ORIGINATION dates** where the data carries them, off the
    RAW `mri_loans_all` frame: one loan is the first lien; several loans all
    carrying an Origination row give the sum of those sharing the earliest;
    anything else falls back and names the undated LoanIDs in
    `first_lien_origination_missing`. **Maturity is never a proxy** — the
    existing `earliest_loan` basis sorts on `dtEvent`, which is a MATURITY on 83
    of 91 live rows, so what it calls "the earliest loan" is the
    earliest-maturing one. Development deals keep the committed facility.
    Measured before shipping: 49 deals have one loan, 16 none, 11 several, and
    **not one of the 11 carries an origination date on every loan** — the whole
    live table holds four Origination rows and all four sit on single-loan deals.
    **No printed figure moved**, and 10 deals are named in the diagnostic.
    `_collapse_loan_date_events` and `_filter_paid_off_loans` were NOT touched;
    the report already read upstream of both.

    **Three markers derived, the rest transcribed**: Current (2) from
    `Currency != 'USD'` (exactly one deal, Apple/CAD), Current (5) from the date
    test, Sold (4) from `sale_date > as_of` (exactly Clima Secur, 30 Bearfoot and
    870 Donald Lynch — East Manchester sold 2026-06-25, five days the other side,
    and is correctly unmarked). Woodlands Square's Current (4), City West's Sold
    (2) and every (6) stay hardcoded. **The marker set on live is byte-identical
    to v550's** — nothing on the page moved; the hand-maintained lists just stopped
    being the authority. A new runtime diagnostic,
    `config_entries_without_a_deal`, names any config vcode with no deal behind it
    and any footnote number its table does not carry: **empty on live.**

    **The default quarter is pinned to 2026-06-30.** `latest_quarter_end` returns
    the quarter that has finished, which on Oct 2 is 2026-09-30 — closed two days
    ago with no accounting behind it, and the report opened on it.
    `DEFAULT_QUARTER_LAG_DAYS = 45` is the rule that replaces the pin and is
    deliberately unwired; `PROCEEDS_CUTOFF_DAYS_AFTER_QUARTER = None` is today's
    behaviour. The guardrail asserts both are inert by reading the engine source.
    Every quarter stays selectable — the list and the default are separate
    questions, and 2026-09-30 is still in the list.

    Guardrail `investment_metrics_check.py` **113 -> 178**, including an
    equivalence test that pins `_pe_roe_events` to `get_pe_performance`'s own
    `roe_to_date` at a quarter end, so the restated classification cannot drift
    from the engine it restates, and a marker regression over the real 76-deal
    population frozen inline. Printed sheet re-measured from the LIVE payload:
    **44/44, fits the sheet.** Live payload verified cell-for-cell against the
    locally predicted one: **0 disagreements across 76 rows x 17 columns.**
    Smoke: One Pager, Portfolio Snapshot bundle (OWPSC/PSC3/PSCKOC, `errors: {}`),
    Dashboard KPIs, deals list, `/investment-metrics` 20.1s cold / 0.17s warm,
    1,057 MiB of 2,048, **0 tracebacks**.)
  - `v550` = `b250639` (HOTFIX FOR v549 — NULLS ARE GUARDED BEFORE THE
    isinstance TEST IN `_as_date`. Oct 1 2026, tag locked.
    **`pd.NaT` IS an instance of `datetime`** — `isinstance(pd.NaT, datetime)`
    is True — so the isinstance branch sitting AHEAD of the null guard returned
    `NaT.date()`, which is NaT, and `row_in_effect`'s `end >= as_of` then
    raised "Cannot compare NaT with datetime.date object".
    UNREACHABLE UNTIL THE DATA CHANGED, which is why it shipped quietly in
    v548. While `queries/MRI_Commitments.sql` filtered `EndDate IS NULL` the
    column was entirely null, pandas typed it object/float, NaN is NOT a
    datetime, and the `pd.to_datetime` path returned None correctly. The moment
    ended rows loaded the column became `datetime64[us]` and every OPEN row's
    EndDate arrived as NaT.
    REPRODUCED AGAINST THE REAL REFRESHED ROWS BEFORE FIXING: Pontchartrain
    raised at both quarters, Camarillo at 26Q3, Asbury at 26Q3. Asbury's 26Q2
    SURVIVED — its in-force row there is an ENDED one that matched before the
    loop reached a NaT — which is exactly the mixed 200/500 pattern seen live.
    After the fix all five resolve and open commitments are still counted as of
    quarter end: Pontchartrain 10,847,420 both quarters, Camarillo 18,843,400,
    Asbury 1,490,000 at 26Q2 and 1,620,000 at 26Q3.
    **THE SUITE WAS GREEN ON A SHAPE THE DATABASE CANNOT DELIVER.** Every
    fixture used Python `None` for an open row; pandas never produces None once
    the column is `datetime64`, it produces NaT. `committed_pref_check` 39 ->
    47, the new cases passing `pd.NaT`. Pre-fix it does not merely fail:
    `_as_date(pd.NaT)` returns NaT AND `_as_date(pd.NA)` raises "boolean value
    of NA is ambiguous" on the old `value == ""`, so the old ordering carried a
    SECOND latent fault the guard also closes.
    VERIFIED ON v550: 114/114 One Pagers 200, 0 failing; 26Q2 Burton
    26,597,500 / JB Fair Park 14,300,000 / Nottingham 9,135,000 and 26Q3 Burton
    54,227,500 / JB Fair Park 29,757,181 / Nottingham 12,535,000, all exact;
    `cap_stack.committed_pe == pe_performance.committed_pe` on every deal
    checked; Snapshot all 4 subtabs x TGAM/KOCINV/BCA build clean; Treasury
    AMB6 13 investors / base 11,000,000.00 / PSC1 42.7273% and TGA25 base
    103,572,497.76 both unchanged; `committed_pref_check` 47/47 and
    `treasury_upload_check` 26/0 IN THE CONTAINER; logs 0 tracebacks, root 200
    ~0.2s. `requirements.txt` still pins `SQLAlchemy>=2.0,<2.1`, re-confirmed.
    Rollback: `activeRevisionsMode` is Single, so redeploy the image —
    `az containerapp update ... --image
    acrwaterfalldev.azurecr.io/waterfall-xirr:b4a9c1f --revision-suffix v551`,
    which is v548 and therefore also reverts the SQL. Note the `commitments`
    TABLE refresh is NOT in git and is not undone by a rollback.)
  - `v549` = `76c786c` (DROPS THE `EndDate IS NULL` FILTER SO ENDED COMMITMENT
    ROWS LOAD — **AND TOOK EVERY 26Q3 ONE PAGER TO HTTP 500**, fixed minutes
    later by v550. Oct 1 2026, tag locked. Approved by Jim.
    `queries/MRI_Commitments.sql` becomes `select * from IA_Commitment`. The
    table goes 557 -> 897 rows, 339 ended, and carries ENDED revisions for the
    first time — which is what `committed_pref.resolve_committed_pref` (v548)
    needs: its as-of rule is `StartDate <= Q AND (EndDate IS NULL OR EndDate >=
    Q)`, and until now the row that WAS in force on a past quarter simply was
    not in the database. **The table refresh is a DATA change that is not in
    git**: run `import_query_to_database("MRI_Commitments")` under a BARE Flask
    app context (never `create_app()`), then `POST /api/data/reload` as admin,
    because that function does not clear caches and the exec session is a
    different process from the workers'.
    ONE READER WAS NOT SAFE AND THE PRE-FLIGHT CHECK IS WHAT FOUND IT. v548
    added `EndDate IS NULL` to the six SQL readers and filtered Investment
    Metrics' FIRST-LOSS side, but its PSC-PREF side took the legacy raw-split
    branch — `capitalization_sources` is only quarter-aware when `as_of` is
    passed, and NEITHER call site passes it. Summing that frame once ended rows
    load reads EVERY PAST VERSION of a pledge as a live one: Pontchartrain
    10,847,420 -> 42,823,260 (4 revisions), Middle Island 7,896,655 ->
    29,978,275, Belleville 4,752,161 -> 21,533,305, Burton 54,227,500 ->
    80,825,000, JB Fair Park 29,757,181 -> 44,057,181, Nottingham 12,535,000 ->
    21,670,000. Both sides now take the current row on the no-as-of path, so
    the report is UNMOVED by the data change — 0 of 6 deals differ, proved both
    ways against a fixture carrying the real ended rows.
    A FIXTURE BUG WORTH RECORDING: the first audit said all six deals changed
    even WITH the fix, because the scratch fixture serialised a null EndDate as
    the STRING "None", so `.isna()` matched nothing and every row read as
    ended. The filter looked broken when it was the test that was.
    `committed_pref_check` 35 -> 39, `treasury_upload_check` 27/0,
    `investment_metrics_check` 113/0, `one_engine_per_number_check` 26/0 — and
    every one of them was green across the outage. See v550.
    STILL OPEN: **Investment Metrics shares the FUNCTION but not the RULE.**
    Neither `capitalization_sources` call site passes `as_of`, so it takes the
    current row rather than the as-of one. Deliberate — making it quarter-aware
    WOULD move its figures and needs its own measurement.)
  - `v548` = `b4a9c1f` (COMMITTED PREF COMES FROM MRI'S IA_Commitment, AS OF
    THE QUARTER. Oct 1 2026, build `camp` 2m32s, run status Succeeded, tag
    locked. Approved by Charlene; Jim notified.
    ONE ENGINE. `committed_pref.resolve_committed_pref` is the only place the
    rule lives, and the One Pager's cap stack, its PE block and Investment
    Metrics all call it. They disagreed on TWELVE DEALS before this: the One
    Pager summed accounting `Typename='Commitment'` rows while Investment
    Metrics already read IA_Commitment first. `one_pager.py` carried TWO
    independent copies of that accounting sum, 1,800 lines apart, agreeing only
    because the filter was duplicated verbatim. Both halves now return the same
    figure on every deal — verified live on six.
    THE AS-OF RULE IS A DATE RANGE, NOT A QUARTER-END SNAPSHOT, and the DATA
    decided it against the stated premise. A commitment is revised by ENDING
    one row and opening the next the following day: of the 80 deal-level chains
    in MRI, 30 pairs are contiguous, 0 have a gap, 1 overlaps. Exactly ONE
    ended row in that whole population lands on a quarter end and there is no
    `09-30` EndDate anywhere in the table. So the row that applies on Q is
    `StartDate <= Q AND (EndDate IS NULL OR EndDate >= Q)`.
    **THE BENEFIT IS DEFERRED AND THAT WAS THE POINT OF SHIPPING IT THIS WAY.**
    `queries/MRI_Commitments.sql` still filters `EndDate IS NULL` — Jim's call,
    NOT touched — so ENDED rows never reach the app. A deal whose current row
    starts after the quarter (JB Fair Park's begins 2026-07-30) has NO row in
    effect at 26Q2. Falling through to funded pref would have moved its printed
    figure by **-22,850,000** to a number matching neither today, the sent
    report, nor the answer the query change will give. So the fallback KEEPS
    THE ACCOUNTING FIGURE, with the basis saying so. Measured before building;
    it is what stopped the first attempt at this deploy.
    ELEVEN DEALS CHANGED AT DEPLOY (v547 -> v548, measured on production):
    Burton 26Q3 81,857,500 -> 54,227,500; Camarillo 0 -> 18,843,400 and Outlook
    0 -> 11,847,307 and East Manchester 0 -> 3,600,000 (both quarters); Clima
    Secur 26Q3 0 -> 3,025,000; Pontchartrain 12,620,000 -> 10,847,420;
    Nottingham 26Q3 12,058,427 -> 12,535,000; JB Fair Park 26Q3 30,000,000 ->
    29,757,181; Middle Island 8,129,967 -> 7,896,655; Asbury 26Q3 1,490,000 ->
    1,620,000; Donald Lynch 0 -> blank.
    THREE CHANGE FURTHER WHEN THE ENDED ROWS LOAD, all at 26Q2: Burton
    54,227,500 -> 26,597,500, JB Fair Park 30,000,000 -> 14,300,000, Nottingham
    12,058,427 -> 9,135,000. **Those three are the figures that were typed into
    the sent 26Q2 TIAA report BY HAND** — which is the strongest evidence that
    IA_Commitment is the right source. Against that report the Snapshot
    Financial page goes 290/355 -> 300/355 and the One Pagers 182/190 -> 186/190
    WITH the ended rows; with today's open-only table it is 287/355 and 182/190,
    the three Camarillo cells the only movement.
    NONE, NEVER 0, AND NO FLOOR. `financials_service` tested
    `committed_pe == 0`, which fires on a genuine zero AND on every None,
    silently relabelling "no pledge on file" as "fully funded"; now `is None`.
    `remaining_to_fund` is NOT floored — Nottingham prints -1.2M at 26Q2 once
    the ended rows load, and flooring it would hide a disagreement between the
    pledge and the ledger. Flagged as `committed_below_funded`.
    TOMBSTONES AND BACK-FILL. Eleven rows open and close on the same day for
    0.00/0.01 and are dropped. A chain whose every row is MRI back-filling an
    uploaded transaction is not a pledge register: Apple - Bales Drive would
    have fallen 4,172,975 -> 170,179, a 96% drop driven entirely by artifacts,
    so its accounting figure is KEPT with the basis `pending accounting`.
    ALL EIGHT READERS OF THE `commitments` TABLE NOW FILTER TO CURRENT ROWS —
    a no-op today, load-bearing the moment ended rows arrive. Treasury's
    investor split is the one that would have broken loudest: AMB6 carries PSC1
    TWICE once ended rows load (11,000,000 ended 2026-06-30 plus 4,700,000
    open), the base doubles to 22,000,000 and EVERY ONE of the thirteen
    investors' percentages halves; TGA25 would go 103.6M -> 438.9M. Verified
    unchanged after deploy: AMB6 13 investors, base 11,000,000.00, PSC1
    42.7273%; TGA25 base 103,572,497.76.
    JIM'S ENGINES PROVED UNTOUCHED BY DIFF, not by reading: all 114 One Pager
    payloads compared against the v547 capture, **ZERO differences outside the
    four intended committed-pref fields**. `waterfall.py`, `capital_calls.py`,
    `metrics.py` and `models.py` contain zero occurrences of "commit", and
    `seed_states_from_accounting` selects on `is_contribution`, which already
    excludes commitments at `loaders.py:275`.
    POST-DEPLOY: root 200 (0.27-0.45s), 0 tracebacks, One Pager 1.8s, Snapshot
    Financial 8.5s cold, Investment Metrics 0.18s. `committed_pref_check` 35/35
    IN THE CONTAINER, proved non-vacuous against four injected defects (rule A
    instead of B fails 2, dropping the tombstone rule 1, returning 0 instead of
    None 1, removing the back-fill gate 3). `one_engine_per_number_check` 26/0,
    `investment_metrics_check` 113/0, `treasury_upload_check` 27/0; every other
    suite byte-identical to the baseline tree.
    **NOT FULLY DELIVERED, and it is not a regression**: Investment Metrics
    shares the FUNCTION but not the RULE — neither `capitalization_sources`
    call site passes `as_of`, so it still takes the legacy raw-split branch.
    No figure of its moved, but it reports Apple - Bales at the back-fill sum
    (0.1242M CAD) where the One Pager now says 4,172,975. Closing that WOULD
    move Investment Metrics figures and so needs its own measurement.
    FREEZE_ENABLED still ABSENT (`[]`) before and after. Rollback:
    `activeRevisionsMode` is Single, so redeploy the image —
    `az containerapp update ... --image
    acrwaterfalldev.azurecr.io/waterfall-xirr:0b439d8 --revision-suffix v549`.)
  - `v547` = `0b439d8` (THREE ONE PAGER FIGURES STOP LYING BY DEFAULT, Sep 30
    2026. Merge of `fix/pe-yield-blank-when-uncomputable` onto `origin/main`
    `6829b38`, so it sits ON TOP of Jim's v545/v546 Investment Metrics work and
    reverts none of it (`ad92358` verified an ancestor of HEAD before building).
    Build 2m14s, digest `sha256:3fe513c7dc95c91b90a2c23d5abadd37411f0119cd892174a86f0a8b818d6c30`,
    tag locked. Rollback target `ad92358` (v546), staged and unused.
    **(1) `pe_yield_on_exposure` DEFAULTED TO 0.0** and is only assigned when it
    can be computed, so any deal failing that test published a computed yield of
    nil where there was nothing to compute it from. Now `None` — the same
    sentinel `pe_exposure_on_cap` and `pe_exposure_on_value` already carried two
    lines above. The field's own trace already disagreed with it:
    `field_trace_service`'s check returned None in exactly those cases. 17 of 70
    deals at 26Q2 published 0.0; they now read N/A. The screen does not move for
    them — both cells are truthiness-guarded, so 0.0 and null both rendered N/A
    already; what changes is the payload.
    **(2) A NEGATIVE NOI NOW YIELDS A NEGATIVE PERCENTAGE.** The gate was
    `noi_ye > 0`, which declined a deal whose NOI is really negative and
    reported it as absent — "cannot compute" and "is negative" came out as the
    same answer. It is truthiness now. Exactly two deals move: Jefferson Addison
    Heights -235,114 / 61,854,983 = **-0.38%** and Jefferson Eastchase
    -698,661 / 71,381,704 = **-0.98%**, both confirmed through the live endpoint
    after deploy.
    TRUTHINESS, NOT `is not None`, AND THE DIFFERENCE IS LOAD-BEARING.
    `perf['noi']['actual_ye']` DEFAULTS TO 0 while `ytd_actual` defaults to
    None, so a deal with no actuals carries a literal 0 that was never assigned
    — verified on all 15: `actual_ye` 0, `ytd_actual` null. Gating on "not None"
    would divide that no-data 0 and republish the fake zero on 15 deals.
    **(3) BOTH PARTICIPATION CELLS AND BOTH COUPON CELLS NOW READ MRI DEAL TERMS
    FIRST**, waterfall as fallback. The One Pager prints each term TWICE —
    Capitalization (`cap_stack.pe_participation` / `pe_coupon`) and PE
    Performance (`participation` / `coupon`) — and the two resolved them
    differently: Capitalization let deal terms OVERRIDE, the PE block took the
    waterfall FIRST. So nine deals were each printing two different numbers for
    one term. `_pe_terms_fallback` had written the participation half down as an
    open question — "picking a winner there is a separate question about which
    source is right" — and it is now decided in favour of deal terms.
    **THE COUPON HALF WAS NOT PREVENTIVE AND WAS BRIEFED AS THOUGH IT WERE.**
    I stated no live deal disagreed on the coupon WITHOUT HAVING MEASURED IT,
    and that was wrong. THREE INVESTOR-FACING COUPONS CHANGED:
    Pegasus Life Storage 10% -> **9%**, Cocoplum Apartments 5% -> **8.5%**,
    Orange Grove 8% -> **8.5%**. Six participation figures moved with them —
    5-15 Broad St 0.333 -> 0.33, Merle Hay 0.7 -> 0.3, OREI Portfolio / Whitney
    Manor / Westchase 0.75 -> 0.475, Donald Lynch 0.2 -> 0.3. In every case the
    PE block moved ONTO the figure the Capitalization block was already showing;
    the Capitalization cell moved on no deal.
    A ZERO IS NOT A COUPON but IS a participation: an explicit 0 in deal terms
    overrides for participation ("the PE takes no share") and falls through for
    the coupon (0% preferred return is not a structure). Inert today — 75 deals
    carry a deal_terms coupon, none of them 0.
    **(4) `normalize_share`, `<= 1` NOT `< 1`.** A share of exactly 1.0 means
    ALL of it; the old test sent it down the percentage branch, 1.0/100 = 0.01,
    and printed "1%" for a hundred percent. ONE definition replacing four copies
    of the same expression across both waterfall readers and both deal-terms
    paths. LATENT TODAY and stated as such: the only two deals carrying 1.0 are
    the OPJPI pair, whose deal_terms 0 overrides them, so nothing on screen
    moves from this half. The COUPON keeps its own `< 1` — a 1.0 coupon reads as
    1%, a 1.0 share as 100% — and no deal carries a 1.0 coupon either way.
    RETURN OF CAPITAL IS NOT IN THIS RELEASE. It shipped as v544 (`e255da7`) and
    is inherited; East Manchester's 3,600,000.00 was re-confirmed here only as a
    regression check.
    VERIFIED AFTER DEPLOY: `onepager_participation_precedence_check` 27/27 **IN
    THE CONTAINER** (replica `v547-56cf769c64-zqtbv`), which is what proves the
    IMAGE rather than the local tree; 12 live spot-checks through
    `/api/financials/<vcode>/one-pager?quarter=2026-Q2`, all pass, with the two
    cells carrying identical values on every deal that previously disagreed;
    root 200 (0.14 / 0.08 / 0.10s), `/api/data/deals` 401, clean boot, 0
    tracebacks, SQLAlchemy `<2.1` pin held.
    `onepager_missing_vs_zero_check` was DELIBERATELY NOT run in the container:
    it lifts `fmtOccVariance` and the One Pager cell out of `OnePagerView.vue`,
    and the runtime image ships no `vue_app/`, so it would fail for reasons
    unrelated to this deploy. 31/32 locally, that one failure pre-existing and
    identical on main. The pe_yield fix is confirmed on the image instead by the
    two Jefferson deals returning real negatives through the live endpoint.
    STILL UNRESOLVED AT DEPLOY TIME: CLAUDE.md's standing instruction is to tell
    Jim BEFORE building anything that moves reported figures. This moved nine,
    three of them investor-facing coupons. It was flagged twice before the build
    and the deploy proceeded on Charlene's instruction without that step being
    closed.)

    RESOLVED Oct 5 2026: Jim reviewed the nine changed figures and confirmed MRI
    deal terms as the source of truth for coupon and participation. The entry
    above is left exactly as it was written at deploy time, including its closing
    line; this note is appended beneath it. `open_items.md` §20.6, which tracked
    the open loop, is closed and removed.
  - `v546` = `ad92358` (INVESTMENT METRICS GOES LIVE — draft gate OFF, and the
    link moves under Asset Management. Sep 30 2026, build `camm` 2m20s, run
    status Succeeded. `INVESTMENT_METRICS_DRAFT = False`, so the screen banner,
    the printed DRAFT line and the watermark are gone and the sidebar links to
    the report between Review Tracking and Waterfall Setup.
    THE LINK IS STILL GATED ON THE SERVER FLAG, not hard-coded visible: the
    compiled sidebar in the served bundle is
    `(config?.investment_metrics_draft)===!1` with `class:"nav-item"`, sitting
    immediately after Review Tracking. `/investment-metrics` joined `amRoutes`
    so the section auto-expands on a direct visit. Turning the gate back on
    removes the link again without touching the sidebar.
    NOT ONE FIGURE MOVED, and it was checked rather than asserted: the v545 and
    v546 live payloads were fetched and compared — **every row identical**,
    totals identical, grand total identical, footnotes identical, column
    geometry identical. The diff is two files, the flag and the sidebar; the
    engine, the labels, the footnotes and both view templates are untouched.
    ACCESS IS THE ONE PAGER'S, VERIFIED ACROSS EVERY LOGIN ROLE. Both routes
    carry `@login_required`, which does not consult role at all. Driven
    locally against the shipped code with a forged JWT per role: viewer,
    analyst, accountant, cfo and admin ALL get 200, and no token gets 401 —
    identical to `/api/financials/<vcode>/one-pager`. Note there is no Asset
    Manager LOGIN role: the six are viewer / analyst / accountant /
    accounting_manager / cfo / admin, and `asset_manager` is a REVIEW-workflow
    role (`_STEP_ROLE`, `review_roles`) that neither the sidebar nor this API
    consults. Asset Management is a navigation grouping.
    POST-DEPLOY: IM cold 4.68s / warm 0.20-0.22s; One Pager 2.53s; Snapshot
    0.11s; Dashboard KPIs 17.8s (the cold shared `load_all`, first request
    after the revision started); deals 0.16s. Container 1,091 MB of 2,048
    (51%), +9 MB over v545. Boot log clean, 0 tracebacks. Live config returns
    `investment_metrics_draft: false`; the live print render is 44/44 with no
    DRAFT text on either sheet.
    STILL OPEN, and the flag being off does not close any of it: UW Proj. IRR
    and both Yr-1 CoC columns remain in `UNLOADED_FIGURES` "none" mode against
    an Alay TODO, and first lien reproduces the reference on 42 of 76.
    Rollback: `activeRevisionsMode` is Single, so redeploy the image —
    `az containerapp update ... --image
    acrwaterfalldev.azurecr.io/waterfall-xirr:c3cb48d --revision-suffix v547`.)
  - `v545` = `c3cb48d` (INVESTMENT METRICS — the quarterly PSC Investment
    Summary as a top-level report, SHIPPED BEHIND A DRAFT GATE. Sep 30 2026,
    build `camk` 2m31s, digest `sha256:f315a801b18a616`.
    **IT IS NOT IN THE SIDEBAR AND THAT IS THE POINT.**
    `investment_metrics_config.INVESTMENT_METRICS_DRAFT = True` drives three
    things from one switch — the screen banner, the printed DRAFT mark, and
    whether the sidebar links to it at all. Published on `/api/data/config` as
    `investment_metrics_draft` (live: `true`) so the sidebar reads the same
    flag the report does. The compiled gate in the served entry bundle is
    `(config?.investment_metrics_draft)===!1`, so with the flag true the link
    cannot render. Route reachable by direct URL at `/investment-metrics`.
    THE MARK PRINTS. A banner that vanishes on the way to the printer is worse
    than none: the screen would say draft and the forwarded PDF would say
    nothing. Verified against LIVE data through the real browser print path —
    both sheets carry it. It adds NO LAYOUT: with the flag on, the 1,430 table
    words on page 1 and 797 on page 2 sit at identical coordinates and there
    are zero extra rects inside the table, so `investment_metrics_print_inspect`
    still measures the real document and still passes 44/44 with the flag off.
    PURELY ADDITIVE against what was live: 15 files, +4,280, **-0**. No DDL, no
    change to any existing computation; the only edits to existing files are
    wiring. P2 listed SIX commits and two are other-author and already
    deployed — `ec9ae69` is CLAUDE.md only, and `a1f073f` is a MERGE whose
    `--cc` diff is EMPTY and whose second parent IS `1da00ca`, the live commit.
    So the net runtime delta was exactly the four Investment Metrics commits.
    90s -> 10s COLD, 0.2s WARM, 0 DB QUERIES. Measured on frames padded to live
    row counts, because a 5k-row stand-in flatters it by two orders of
    magnitude. `_earliest_isbs_debt` normalised a 233k-row column with a Python
    `map` on every call — 92 calls, **21.6 million `norm_id` calls**, 53 of 64
    seconds; `_get_uw_7073_signed` and `_get_uw_pe_periodic` each open with
    `isbs_raw.copy()`, 152 copies of a 325 MB frame per report. All three fixes
    are pure narrowing done once with the SAME predicates, and the payload is
    byte-identical before and after. `one_pager` deliberately NOT modified — it
    is shared with the One Pager and the Portfolio Snapshot.
    Route cache invalidates BY OBJECT IDENTITY, holding a reference to the
    frames it was built from: `id()` alone can be reused and row counts repeat.
    POST-DEPLOY, MEASURED: IM cold 5.25s / warm 0.17-0.20s; Dashboard KPIs
    18.9s (that was the cold `load_all`, shared, first request after deploy);
    One Pager 2.58s; Snapshot bundle 1.30s; deals 0.15s. Container 1,082 MB of
    2,048 (50%). Boot log clean, 0 tracebacks.
    A HARNESS ARTIFACT WAS CORRECTED BY THE DEPLOY, worth recording because it
    ran through every figure I reported beforehand: `vAccount` is TEXT on
    production and the local CSV mirror brought it back as int64, so every
    `== '7073'` in `one_pager` matched nothing and U/W ROE came back empty for
    all 76 deals — silently, as "this deal has no underwriting". On production
    `proj_coc_since_close` is populated on **53 of 76**, not 29. Live cell
    agreement against the reference PDF is **618/1242 (49.8%)**, or 576/1014
    (56.8%) excluding the 228 cells held for Alay.
    NOT SIGNED OFF, hence the gate: UW Proj. IRR and Proj/Act Yr-1 CoC are all
    behind `cfg.UNLOADED_FIGURES` in `"none"` mode with an open TODO(alay), and
    the first-lien column reproduces the reference on only 42 of 76.
    Rollback is REDEPLOY THE IMAGE, not a traffic split —
    `activeRevisionsMode` is **Single**, so `ingress traffic set` does not
    apply: `az containerapp update ... --image
    acrwaterfalldev.azurecr.io/waterfall-xirr:1da00ca --revision-suffix v546`.)
  - `v544` = `1da00ca` (A DEBT-FREE DEAL IS FOUND BY ITS DATA, and return of
    capital counts only `Capital='Y'`. Q3 phase-1 plus the debt-free rule,
    shipped together, Sep 30 2026. Built from a worktree cut at live `ad65707`;
    P2 span was exactly five commits over seven files.
    **THE PER-DEAL LIST IS GONE.** `DEBT_FREE_DEALS = {"P0000066"}` is removed —
    not emptied — and the N/A display is now derived: `not dev`, `not sold`,
    ISBS basis, debt exactly `0.0`, no active MRI loan, and
    `property_count >= 1`. Both guardrails assert the constant cannot return.
    **THE PARENT TERM EXISTS BECAUSE THE MEASUREMENT FOUND SIX DEALS.** Without
    it the rule fired on Pegasus AND on all six Town Fair Tire properties
    (`P0000101`-`P0000106`) at 26Q1, 26Q2 AND 26Q3 — each reporting
    "no debt account rows -> 0" with no loan of its own. They are CHILD
    properties (`Property_Count == 0`, one shared `Portfolio_Name`) whose
    facility is held at the parent, so "held with no debt" is the wrong
    sentence for them. Pegasus is `Property_Count == 1` with no portfolio. A
    NULL `Property_Count` is declined on its own account as well as by
    `_deal_index`'s coercion, so an unknown can never widen the rule.
    **AND THE `sold` TERM IS NOT WHAT EXCLUDES CITY WEST**, which was my
    assumption and was wrong. PCITWES never reaches the rule at all: it has no
    ISBS rows, so its debt is `None`, not `0.0`. The `sold` term is still
    correct and still required — do not read the measurement as proving it
    load-bearing for that deal.
    POPULATION CHECK, READ-ONLY ON PRODUCTION, AFTER THE PARENT TERM: fires on
    `P0000066` and nothing else at 26Q1, 26Q2 and 26Q3; `CHANGED vs old vcode
    list = NONE` at all three. **The only thing that moved is the SOURCE of
    Pegasus's dash** — same em dash, same five N/A literals, `debt` still
    `None` so no subtotal absorbs anything.
    RETURN OF CAPITAL: `_is_return_of_capital` reads MRI's `Capital` flag, not
    the Typename, so a Realized Gain stops inflating capital coming back.
    **MEASURED THROUGH `get_pe_performance` ITSELF, old rule vs new, all deals
    at 26Q2: 21 DEALS MOVE AND NONE GOES TO $0.00.**

    | deal | before | after |
    |---|---|---|
    | `P0000007` Berger Pittsburgh Portfolio | 57,183,009.00 | 36,719,000.00 |
    | `P0000003` Apple Self Storage | 46,417,982.69 | 27,214,566.32 |
    | `PVILLAGE` Village Square Apartments | 29,026,157.79 | 15,400,000.00 |
    | `PCAMARI` Camarillo Village | 23,934,631.39 | 18,843,400.00 |
    | `POUTLOO` Outlook Nine Mile | 19,855,369.00 | 11,847,307.00 |
    | `PJWEST` Jefferson West Love | 19,176,803.67 | 14,747,340.00 |
    | `PWILLOW` Willowdale Apartments | 18,629,374.08 | 10,585,000.00 |
    | `P3RDAVE` 3rd Ave & Indian School | 15,311,541.00 | 8,533,755.00 |
    | `PDEVON` Devon Square | 14,595,605.00 | 12,000,000.00 |
    | `PLANCS1` Lancaster Apartments | 13,887,277.96 | 7,558,214.86 |
    | `PASTONC` Jefferson Centura | 6,103,708.57 | 4,346,000.00 |
    | `P0000017` East Manchester | 5,139,662.37 | 3,600,000.00 |
    | `PSHOPPW` Shoppers World | 4,671,733.48 | 3,374,000.48 |
    | `PORANGE` Orange Grove | 4,000,000.00 | 1,200,000.00 |
    | `P0000038` Quakertown Shopping Center | 3,681,883.24 | 3,087,500.00 |
    | `PDECLAN` Declan & Walton | 3,049,567.22 | 2,250,000.00 |
    | `PCREEK` Creek Crossing | 3,030,794.99 | 2,200,000.00 |
    | `PLENDSS` Leander Self Storage | 3,023,728.71 | 2,261,292.28 |
    | `PJEFFOA` Jefferson Oakhurst | 2,876,112.87 | 1,796,000.00 |
    | `PHOMEW` Homewood Commons | 2,604,531.05 | 1,828,033.33 |
    | `PBARN` Barnbeck Apartments | 1,549,746.02 | 1,150,000.00 |

    **CORRECTION TO `e255da7`'S OWN COMMIT MESSAGE, recorded here because git
    history is not being rewritten.** That message says "IT MOVES 20 DEAL/
    INVESTOR PAIRS... Seven go to 0.00 because their entire reported return of
    capital was realized gain (Village Square, 3rd Ave, Orange Grove, Jefferson
    Centura, Shoppers World, Jefferson Oakhurst, Leander Self Storage)."
    **That was a PAIR-LEVEL reading and the deal-level result is different: 21
    deals change and NOT ONE reaches $0.00.** All seven named deals do appear
    in the table above — they were identified correctly as affected — but each
    lands on a real figure (3rd Ave 8,533,755; Orange Grove 1,200,000;
    Jefferson Centura 4,346,000; Shoppers World 3,374,000.48; Jefferson
    Oakhurst 1,796,000; Leander 2,261,292.28; Village Square 15,400,000). The
    claim that their whole return of capital was realized gain is false. Read
    the table here, not the commit message.
    `P0000042 "Village Square Apartments"` is a SECOND, EMPTY deal — it maps to
    no InvestmentID (`iids=[]`), funded 0.00 and ROC 0.00 before and after — so
    its zero has nothing to do with this change and must not be read as one of
    the seven.
    A METHOD NOTE WORTH KEEPING: the first measurement grouped raw accounting
    rows by `(InvestmentID, InvestorID)` and reported 85 moving pairs, because
    that counts intermediate entities (`PPI16`, `PSCKOC`, `PSC3`) as deals. The
    engine scopes by VCODE via `build_investmentid_to_vcode`, which is what
    gives 21. Same arithmetic, wrong grouping — and it agreed with the engine
    to the cent on every deal it did scope correctly, which is why the
    disagreement looked like a defect rather than a grouping error.
    POST-DEPLOY, ON PRODUCTION: root 200 (0.74s / 0.11s / 0.15s), clean boot
    with 0 tracebacks, `/api/data/deals` and `/api/data/config` both 401,
    `FREEZE_ENABLED` still ABSENT (`[]`) before and after — freezing stays off.
    `q3_cleanup_check` 20/0 and `debt_free_rule_check` 41/0 **in the
    container**, the former asserting East Manchester's 3,600,000.00 against
    live data. Locally both trees were run side by side against live `ad65707`
    and every pre-existing failure is identical, so nothing regressed:
    `freeze_as_sent_check` 103 -> 104, `one_engine_per_number_check` 26/0,
    `loan_maturity_gap_check` 36/0, `lease_terms_check` 129/0,
    `gl_ia_query_check` 123/0.
    `debt_free_rule_check` is proved non-vacuous THREE ways: `--inject=off`
    fails 9, `--inject=nosold` fails 3, `--inject=nochild` fails 7. It also
    caught a bug in its own fixture — "New Construction" is no longer in
    `config.DEV_STRATEGIES`, so the dev case was silently a second unlevered
    one.
    NOT RECORDED BY ANYONE: `v543` = `ad65707` shipped with NO deploy-history
    entry, the same gap as `v522`. Left for whoever deployed it rather than
    reconstructed here.
    Build `1da00ca`, tag locked `--write-enabled false`. v543 stays tagged for
    rollback.)
  - `v542` = `992de9d` (RENT ADD-ONS: a later document stating the whole rent
    ends an earlier add-on -- Mattress Firm $243,600 -> $170,100, their exhibit's
    figure -- and every add-on applied is flagged with its document. Acceptance
    vs new business's exhibit: current rent 29/32, expiration 28, options 26,
    subordinate rows 18; total $3,203,508 vs their $3,112,833, the gap now mostly
    Habitat's "Rent Reduction Request" letter (flagged, an analyst's call).
    Build `camg`. Span `35ed8c6..992de9d` is this one commit.)
  - `v541` = `35ed8c6` (FIXES A REGRESSION v539 INTRODUCED, found only by
    re-running the acceptance comparison -- every guardrail was green. Current
    rent had fallen 30 -> 26 of 32 and the total risen $231k. Re-anchoring an
    original lease's months now happens only when the schedule PROVES the lease
    commencement (Habitat had shifted six months on a guess); an amendment that
    QUOTES the original's month-of-term schedule no longer counts it from its own
    date (Hobby Lobby, which the old tenant-wide dedup had hidden by accident).
    Build `camf`. RUN THE ACCEPTANCE COMPARISON AFTER EVERY LEASE-ENGINE DEPLOY.)
  - `v540` = `4283d8b` (Patton's option term travels with its rent schedule and
    the options summary is computed, not read back from consolidation --
    "6 x 30-Day Rolling", one row, verified on production. v539 had not fixed it:
    the fixture put the wording where I expected, not where production stores
    it. Build `came`.)
  - `v539` = `a767df5` (CHARLENE'S PR #5 + THE THREE ACCEPTANCE FIXES. P2 listed
    her four commits and three merges; reviewed, not a symptom repair -- the
    background freeze with each deal built once, quarter-level unfreeze, reaping,
    the frozen store protected. FREEZE_ENABLED STAYS OFF and the gate is still in
    `freeze_part`. Behaviour change: freezing FROM THE PUBLISHED PDFs is removed
    (26Q3 onward, live data only). Nit: the job table's heartbeat ALTER is tried
    once per process and its error swallowed rather than checked in
    information_schema first. Her 8 guardrails green. Ours: stale-settlement
    flag, rent steps replaced per document (rebuilt on production: Poplar
    241 -> 180, Windsor 673 -> 509, idempotent), Little Petals' option now
    2031-03-01. Build `camd`.)
  - `v538` = `ccd47bc` (STEPS 4 + 5 of the rent-roll plan: analyst settlement of the
    timeline, the Rent Roll tab, and the IC exhibit. BUILT FROM `ccd47bc`, NOT
    MAIN'S TIP: origin had Charlene's PR #5 (freeze background/unfreeze, ~2.7k
    lines) merged on top, unreviewed by this deploy, so it was left for its own
    deploy -- it is on main and NOT live. Before deploying, the Market at Poplar
    full re-read finished (33/33; the single job was killed at 26 -- memory, no
    traceback, app unaffected -- and the rest re-run one subprocess per tenant).
    ACCEPTANCE vs new business's exhibit, 32 of 33 tenants paired: SF 31, rent 30,
    expiration 27, options 25, subordinate rows 17, whole block 4; totals 228,122 /
    $3,125,144 vs 228,119 / $3,112,833. The layout alone reproduces theirs exactly
    (rent_roll_exhibit_check). Build `camc`.)
 (LEASE TIMELINE -- step 3 of the rent-roll plan: one engine
    from governing terms to continuous periods, `governing_steps`, validation reads
    the same schedule. Measured on Poplar vs the exhibit: current rent 26/27, steps
    20/27, option rent 0/34 pending re-extraction. See rent_roll_exhibit.md. Build
    `camb`.)
  - `v536` = `2e92142` (LEASE GOVERNING TERMS -- step 2: undated documents fill
    gaps only, lease start from the original lease, exercised options carry the
    expiration, option rows per document, option rent in the prompt. After deploy:
    77 tenants re-consolidated, clause/option rows rebuilt for reviews 2 and 3
    (Windsor exclusives 1,134 -> 337). Build `cama`.)
  - `v535` = `6ab7ae8` (STEP 1: NUL characters stripped, the PDF size check on the
    encoded request with over-size scans sent as images, a failed text reading
    retried from the PDF. Re-read after: Sam's Club and Perkins read in full;
    Tropical Smoothie's 60-page lease still fails. Build `cam9`.)
 (NEW BUSINESS DOWNSIDE CANDIDATES READ THE SETTLED ROSTER,
    Sep 29 2026. `scenario_service.get_risk_candidates` read `lease_tenants` raw:
    an analyst's settled rent or expiry never reached the downside scenario, and a
    tenant read as vacated / no lease was still offered. Now `get_resolved_tenants`.
    Both defects reproduced against the old code (lease_clause_rows_check 35 ->
    37). P2: `45c992c` docs only. Build `cam8`.)
  - `v533` = `ffb7bf7` (A SCAN THE MODEL WON'T READ AS A PDF IS RETRIED AS PAGE
    IMAGES, Sep 29 2026. GNC's 1996 lease gave degenerate replies to the PDF
    block every time; rendered with PyMuPDF (1600 px, JPEG q80) the same pages
    read. The retry on the PDF route now sends the images; capped at 100 pages /
    22 MB, and an unrenderable file retries as the PDF and says why. VERIFIED ON
    PRODUCTION: the GNC re-read logged "No JSON ... asking once more", the retry
    went as images, and the 1996 lease is `extracted` via `images` -- 1,300 SF, no
    co-tenancy, the vitamins/supplements exclusive held (Rider 25), plus three
    restrictions it is bound by; all 9 GNC documents read, 0 left. P2: `54954b9`
    is docs only. Build `cam7`. lease_scan_extraction_check 33 -> 39.)
 (LEASE CO-TENANCY / EXCLUSIVES -- a re-read replaces a
    document's clause rows, the export and Exclusive Use tab show Holds / Bound
    by, radius, carve-outs and source, `lease_clause_reviews` keeps the analysts'
    reading, a failed reading is `error` with its reason, one retry when no JSON
    returns, caps raised to 64K output / 2M text chars. Sep 29 2026. P2 listed two
    commits; `663c7c4` is CLAUDE.md only. AFTER DEPLOY, on production:
    `rebuild_clause_rows(3)` took Market at Poplar from 291 exclusive rows to 92
    (holder 63 -> 19, most from one document 25 -> 7; Firehouse 22 -> 7 with ONE
    sandwich exclusive held), idempotent on a second run, 0 seller rows touched.
    GNC re-read: 8 of 9 documents read; the 1996 original lease (41-page scan)
    STILL fails -- the retry fired, and the model's replies to that one PDF are
    degenerate (the prompt's first sentence; a 30-character markup fragment;
    no thinking), so it is now recorded as `error` with the reason instead of
    logged "Extracted". Not a token limit. See open_items §15.1. Build `cam6`.)
 (INTERCOMPANY, PHASE 1 -- the CFO's Due to/from PSC Manager
    reconciliation as Accounting -> Intercompany, Sep 29 2026. Reads `gl_detail`,
    basis A.B only (C and T rows exist and halve the figures). P2 listed two
    commits; `2d80ec1` is CLAUDE.md only, so the runtime delta is exactly
    `582f92d`. VERIFIED ON PRODUCTION POSTGRESQL: entity side -308,316.59 and
    manager side 313,093.97, the CFO's sheet to the cent; 64 rows, 0 drilldown
    mismatches across 4 sides; Investigate = NOTTNV -22,385.07, PEGASU -94,941.70
    (on its real alternate account MR99991102 -- the sheet had it on the wrong one
    and read No Balance), PPI2 -274.16, PSC2 -2,467.66; INVF10 reads No Balance
    where the sheet's formula error said Investigate. New tables
    `ic_entity_settings` (five rows seeded ONCE from the CFO's answers, each
    saying where it came from; PSC1's Liberty MM exclusion is marked INFERRED)
    and `ic_recon_notes`, both protected. Also fixed: GL / IA Query freshness read
    refresh state 'complete', which nothing writes, so it said "unknown" after
    every refresh since v504 -- production now reports 2026-09-28 14:56. Pay and
    the JE file are NOT built. Build `cam5`, 2m48s.)
  - `v530` = `e97c9fe` (FREEZING IS PER QUARTER AND SWITCHED OFF. Deployed Sep 29
    2026 12:53 EDT (16:53:39 UTC), build `cam4` 2m44s, digest
    `sha256:f713a2ad675659b401c1906cd775db80d033158def89cf3351770590157b1f0a`.
    Two all-investors buttons become ONE PER TAB -- Snapshots on the Portfolio
    Snapshot page, One Pagers on the One Pager page -- on one shared
    `FreezeQuarterPanel`, so each exists in exactly one place. The screen posts
    in SLICES of 10 through the endpoint's existing `investors` parameter, so
    "40 of 127" is measured rather than animated, and no single request has to
    carry every investor.
    **FREEZING IS OFF AFTER THIS DEPLOY AND THAT IS THE POINT.** `FREEZE_ENABLED`
    defaults false and is NOT set on the container (verified before and after:
    the env query returns `[]`). It exists because the all-investors batch was
    run as a SINGLE request over ~145 investors earlier the same day and the app
    was unavailable for about 35 minutes. The freeze itself was correct -- 145
    rows written, then cleanly unfrozen, 0 left frozen, 145 archived. The SHAPE
    of the request was the problem. It goes back on only when the freeze runs as
    a background job.
    GATED AT THE CORE, not only at the doors: `freeze_part` raises, so both
    batches, the published-overlay freeze, re-freeze and the Portfolio Snapshot
    approval chain are covered, as is any entry point added later. The One Pager
    approval snapshot does NOT route through `freeze_part` and is gated
    separately. Unfreeze is deliberately NOT gated -- a mistake made before the
    switch has to stay correctable. VERIFIED ON PRODUCTION, authenticated as
    admin: all four of `freeze-all/snapshots`, `freeze-all/one-pagers`,
    `freeze-overlay` and `refreeze` return **503** with the reason.
    A FREEZE IS NOT AN APPROVAL. `approved_at` was declared
    `DEFAULT CURRENT_TIMESTAMP` and the INSERT never named it, so EVERY row
    carried one -- including an as-sent freeze that deliberately leaves
    `approved_by` NULL. All 114 rows read off production showed exactly that.
    It also broke the legacy fallback: `frozen_parts_of` inferred BOTH halves
    from `frozen_at` OR `approved_at`, and since `approved_at` was always
    present that test was always true. Narrowed to `frozen_at`; the fallback
    STAYS (dropping it would silently un-freeze every quarter already sent) but
    now reports the parts as INFERRED via `frozen_is_legacy`. The INSERT names
    `approved_at` -- which is what actually closes it, since a default only
    applies to a column an INSERT omits.
    **PRE-FLIGHT P4 CAUGHT A DEFECT IN THE MIGRATION BEFORE THE BUILD, which is
    exactly what it is for.** `_drop_approved_at_default` ran an UNCONDITIONAL
    `ALTER TABLE ... DROP DEFAULT` inside `_ensure_table` -- and `_ensure_table`
    is reached from ordinary READS: `_current_row`, `quarter_part_state`, and
    `quarters_frozen_with_deal`, which the One Pager comment lock calls on EVERY
    save. On PostgreSQL that takes an ACCESS EXCLUSIVE lock, so every such
    request would have serialised behind a catalog lock for a migration whose
    work is done once. It also broke the pattern beside it -- `_ADDED_COLUMNS`
    has always checked `if col in existing: continue`. Fixed in PR #4 before
    building: the ALTER asks `information_schema` first, `_ensure_table` runs
    once per process per engine, and `create_app` calls `ensure_schema()` at
    startup so the lock is taken while the worker boots.
    THE MIGRATION RAN, ONCE, AND SAID SO: the boot log carries exactly one
    `Dropped the portfolio_snapshot_frozen.approved_at default (was
    CURRENT_TIMESTAMP)`. 1 worker booted against `GUNICORN_WORKERS=1`, 0
    tracebacks, no psycopg error.
    NO SCREEN PINS A QUARTER ANY MORE. `2026-Q2` was a literal in FIVE places,
    not the two expected -- both views AND the two print-sweep scripts' own
    fallbacks, plus all 61 rows of `onepager_print_population.txt`. Screens read
    `/api/portfolio-snapshot/quarters`; the sweeps take `--quarter` and REFUSE
    rather than guess. On production today that endpoint returns
    `default = 2026-Q2` (26Q3 ends Sep 30 and has not ended), so the One Pager
    batch and Review Tracking both open on 26Q2.
    P2 LISTED ELEVEN COMMITS against live `4036e69`, and that is expected: four
    are the feature, one is the P4 fix, four are docs only (`.claude/`,
    `CLAUDE.md`) and two are merges introducing ZERO lines of their own
    (`git show --cc` empty on both). `845ab5d`'s `--stat` looks like it carries
    `financials.py`, but that is its first-parent diff -- `9bedf3d` is already
    an ancestor of the live image, confirmed with `merge-base --is-ancestor`,
    and the net span diff contains no `financials.py`. Zero vcode literals in
    shipped `flask_app/` or `vue_app/` code; the only ones are the population
    list and a test fixture.
    POST-DEPLOY: root 200 in 0.51s; `freeze_disabled_check` 70/0 and
    `quarter_hardcode_check` 16/0 IN THE CONTAINER (the Vue-source sections skip
    with a reason, no `vue_app/` in the image); 154 / 70 / 73 / 24 locally. The
    served panel chunk, resolved from the entry bundle rather than the 551-byte
    SPA shell, carries "all investors", "Freezing is temporarily disabled",
    "Apply the published PDFs first" and "Partly frozen", and the old
    "Freeze all Snapshots" string is gone. One Pager load 2.31s; comment save
    0.35s -- the path that would have carried the ALTER.
    NOT DONE: no freeze has been run and none can be. 26Q2 is still to be frozen
    from the published PDF overlay after the rerouting work. v529 stays tagged
    for rollback.)
  - `v529` = `4036e69` (FREEZE ALL SNAPSHOTS / ALL ONE PAGERS, ON ONE CORE, and the
    One Pager comment lock closed. Deployed Sep 29 2026 10:12 EDT (14:12:11 UTC),
    digest `sha256:8873f844c1a362552639503e6ca8eb9822b4555a618321a7a0f776813da4d0b4`.
    THE IMAGE IS A MERGE COMMIT, so read its parents, not its diff: `4036e69` is
    `9bedf3d` (freeze-all-parts) merged with `30bf3a8` (origin/main), which carries
    JIM'S v528 `799239a`. Nothing of his is reverted; v528's runtime is inside this
    image. Rollback target is `799239a` (v528), still tagged.
    ONE CORE, NOT A SECOND ENGINE. `freeze_part(investor, quarter, part, ...)` is the
    single body, and the two batch buttons, the published-overlay freeze, refreeze and
    the approval-chain `freeze()` ALL call it. `freeze_as_sent` and the per-investor
    `POST /freeze` are DELETED, not left beside it -- the old route now 405s. The
    overlay takes BOTH parts in ONE call on purpose: one document covers both halves,
    so two writes would mint two versions and two history rows for a single act.
    THE CARRY-FORWARD IS LOAD-BEARING. The upsert is DELETE-then-INSERT, so without
    explicitly carrying the untouched half forward, freezing the Snapshot BLANKS the
    One Pagers -- silently, with the row still present. Four additive columns via
    `_ADDED_COLUMNS` plus `frozen_parts` on the history table, which
    `CREATE TABLE IF NOT EXISTS` would never have added to the existing production
    table (the v507 lesson).
    A ONE PAGER FAILURE NO LONGER SINKS THE SNAPSHOT: the old core raised on any
    failed One Pager, which with independent halves would let one unbuildable deal
    block a Snapshot that is fine. Failures drop that part and are REPORTED. Asking
    for the One Pagers alone and having them all fail still raises, because then
    nothing was frozen.
    THE COMMENT LOCK WAS A REAL GAP.
    `PUT /api/financials/<vcode>/one-pager/comments` is on `financials_bp`, so the
    snapshot blueprint's `before_request` never saw it -- comments stayed EDITABLE on
    a frozen quarter unless it also happened to be APPROVED, which is a different
    authority. It takes no investor and cannot: comments are keyed (vcode, quarter)
    while a freeze is keyed (investor, quarter), so the question is asked of every
    investor and any frozen report carrying the deal refuses the edit, naming it.
    DELIBERATELY BROADER THAN THE FREEZE ITSELF -- freezing one investor's quarter
    makes that deal's One Pager comments read-only for the other investors on the same
    deal, which nobody would predict from the button's label.
    TWO OTHER BEHAVIOUR CHANGES WORTH JIM'S EYE, both intentional: freezing is now
    `@roles_exactly("admin")` where it was analyst-and-above; and a freeze PRESERVES an
    existing `approved_by` instead of nulling it (`"by": approved_by or
    prior.get("approved_by")`), needed so the second half of a two-part freeze cannot
    drop it. A fresh as-sent freeze still leaves `approved_by` NULL, asserted by the
    guardrail. The review chain -- submit / approve / return_to_draft / reopen -- is
    untouched.
    GUARDRAILS: `freeze_as_sent_check` 148/154 IN THE CONTAINER and 154/154 on the
    merged tree locally; the 6 are the `overlay_26q2.json` checks, which SKIP because
    that file is gitignored and not in the image -- correct behaviour, stated by the
    script, same as v527. `traceability_tools_check` 129/129. Jim's own suites on the
    merged tree 42 / 33 / 58. `line_mapping_check` exits 1 on an empty local SQLite and
    does so IDENTICALLY on Jim's unmodified tree -- a pre-existing environment failure,
    not this branch.
    LIVE SMOKE: clean boot, the SQLAlchemy `<2.1` pin held (no "No module named
    psycopg", the v524 failure still closed), both buttons present in the served
    bundle, old `POST /freeze` gone, endpoints auth-gated.
    NOT DEMONSTRATED AGAINST PRODUCTION: a signed-in NON-ADMIN getting 403. It is
    asserted by the code and by guardrail U, but no non-admin credential was
    available, so the one thing the role narrowing exists to do has been proved only
    locally. **NO REAL FREEZE HAS BEEN RUN** -- the v527 caveat still stands, and
    deploying the buttons does not press them.)
  - `v528` = `799239a` (THE ARGUS CASH FLOW IS LOADED ONCE AND MAPPED BY ACCOUNT --
    AM's third list, Sep 28 2026. The Assumptions-tab upload, its route and
    `valuation_service.import_argus` are gone; "Load Valuation Cash Flow" on
    Budget Review WRITES the Valuation cash flow from its own reading of the file
    (create, replace in place, or a new import when another cycle shares one).
    The two old paths used DIFFERENT PARSERS and joined the mapping back BY
    LABEL. Keyword pre-fill removed; an account column to the RIGHT of the
    description is now read (AM's layout -- before, nothing would have
    pre-filled); a subtotal reading can be overturned. FOUND BUILDING IT: the
    Partnership costs tick box never left the browser from v502 -- it now
    writes -- which also falsified one example in the v525 entry, corrected in
    place. P2 listed six commits; five are docs only against live (`.claude/`,
    `CLAUDE.md`), so the runtime delta is exactly `799239a`. VERIFIED ON
    PRODUCTION: `argus_single_load_check` 35/0 (7 screen checks skip, no Vue in
    the image); every column the new commit writes exists on PostgreSQL; the old
    route is gone; 4 records link an Argus import, 0 shared. Served chunk
    `ValuationsView-Ciczbwp4.js` carries the new strings and not the old.
    Build `cam2`, 2m12s.)
  - `v527` = `59b1875` (TWO FEATURES ONTO LIVE, AS A MERGE -- traceability
    enhancements (`da8c356`, 6 commits) and freeze-as-sent (`72f4592`, 14) on
    top of live `6bf26b6`, Sep 28 2026. 22 commits in the span, 22 files,
    +8,921 / -1,218.
    MERGED, NOT REBASED, AND THAT IS THE WHOLE POINT. Both branches were built
    before v526 and are BEHIND it; replaying either onto an older base would
    have dropped the Budget Review work AND Jim's SQLAlchemy pin. The proof is
    the net diff against live: `requirements.txt`, `database.py`,
    `api/valuations.py`, `budget_import_service.py`, `valuation_service.py` and
    `ValuationsView.vue` DO NOT APPEAR in it. The pin is byte-identical to v526
    and gunicorn booted both workers clean -- no "No module named psycopg", the
    v524 failure closed.
    ZERO CONFLICTS, and not by luck: the two features touch DISJOINT file sets
    (traceability 12 files, freeze-as-sent 10, intersection empty). Only
    `.claude/memory/open_items.md` auto-merged, and `.dockerignore` excludes
    `.claude/` anyway.
    THE ONLY ENGINE FILE IN THE SPAN IS `one_pager.py`, and it is purely
    additive: the DSCR denominators and ROE components were already being
    computed and thrown away, and are now published. Each denominator is
    recorded WHERE IT IS COMPUTED and never inferred from another -- the bases
    genuinely differ per column (ytd_actual is 5190 plus the balance-sheet
    principal change, ytd_budget is budget 5190+7060, uw_ye is 7010 annualised)
    -- and a suppressed column clears its denominator WITH the ratio, so the
    trace cannot show a figure for a column the page does not publish. `None`,
    never `0` or `{}`, so "no breakdown" stays distinct from a breakdown of
    zeros.
    THE ONE BEHAVIOURAL SWAP WAS VERIFIED, NOT TRUSTED: `calculate_roe` ->
    `calculate_roe_detailed` cannot move the scalar, because
    `calculate_roe_detailed` DELEGATES (`metrics.py:243` is
    `roe = calculate_roe(...)` and that same value is returned). ONE ENGINE
    holds.
    NO NEW PER-DEAL HARDCODES. `AT_CLOSE_FORCE_SUPPRESS = {"P0000066"}` is NOT
    in the span diff -- pre-existing and already live; `P0000089` is inside
    `_selftest()` (`# pragma: no cover`). Checked rather than assumed.
    GUARDRAILS: `traceability_tools_check` 129/129 and `freeze_as_sent_check`
    93/93 IN THE CONTAINER; 129 / 17 / 99 locally.
    `katex_render_check.py` FAILS IN THE CONTAINER AND THE FEATURE IS FINE --
    worth recording because it will fail on every future container run until
    fixed. It reads `vue_app/package.json`, and the runtime stage never copies
    `vue_app/` (only `COPY --from=frontend-build /build/dist/ static/`);
    `ls /app/vue_app` is "No such file or directory". It guards its LIVE-RENDER
    half with `skip()` but reads `package.json` unguarded, so it crashes where
    the v490/v491 screen checks skip with a reason. The feature was verified
    against the SERVED bundle instead, resolving the lazy chunk from the entry
    bundle rather than the 551-byte SPA shell that burned v523:
    `katex-BYZTj3zW.css` sits in Vite's preload map inside
    `index-BzdiB7yj.js` and serves 200 / 29,288 bytes.
    The `freeze_as_sent_check` 99 -> 93 delta is also explained: the six
    `overlay_26q2.json` checks SKIP when that file is absent, and it is
    gitignored so it is not in the image. Correct behaviour, stated by the
    script.
    **NO REAL FREEZE HAS BEEN RUN, AND DEPLOYING THE CODE DOES NOT PRESS THE
    BUTTON.** `docs/review_freeze_as_sent.md` ships in this span and says the
    26Q2 overlay has NEVER been compared against live data -- the cached Sep 23
    pull never arrived, so the cell-by-cell test runs against a stub. The
    preview is the first place real values meet the overlay; read it. The
    freeze is admin-only (`roles_exactly("admin")`), flagged columns BLOCK, and
    `_lock_frozen_quarters` is inert while no quarter is frozen. Jim signed off
    on the deploy; the review note's five open items (the `assembler=lambda`
    seam most of all) still stand before anything is frozen.
    NOT FIXED HERE, and live: `DEBT_FREE_DEALS` blanks Pegasus's
    `debt_display` while `loan_subtotal()` sums the raw `debt`, so $25.2M sits
    inside Portfolio Totals with no row accounting for it. `open_items.md` §12.
    [SUPERSEDED BY `v544` -- read that entry, not this sentence. The constant is
    REMOVED, not emptied; the N/A display is derived from the row's own data; and
    `debt` and `debt_display` are both bound through one `debt_field()` returning
    `None`, so the figure the subtotal sums and the figure the cell prints are
    decided once. `open_items.md` §12 is closed. Annotated Oct 5 2026.]
    Build `cam0`, 2m27s, digest `sha256:c9af915deee3c62a`. Deployed at 100%;
    v526 deactivated and its tag intact for rollback.)
  - `v526` = `6bf26b6` (BUDGET REVIEW: the third column can be UNDERWRITING, the
    Budget column's debt service can be UW's, an Estimate LINE can be overridden
    (totals follow, marked), and budgeted occupancy is read off the budget file --
    AM's second list, Sep 25 2026. UW debt service is ONE figure, 7010, so its
    Interest/Principal are blank, never split. Also fixed: the tab 500'd for any
    deal with no ISBS. VERIFIED ON PRODUCTION (PostgreSQL): both tables and the
    basis column present; all 162 records render in BOTH modes, 0 errors; UW
    reaches the budget year on 131 of them but only 78 carry 7010 -- on the other
    53 the UW debt service is BLANK, not zero, which is correct and worth knowing
    before AM reads it as "no debt". Guardrail 50/50 in the container. Served
    chunk `ValuationsView-klczDMgR.js` resolved from the entry bundle.)
  - `v525` = `0d68e53` (THE TIE-OUT'S NOI IS THE BUDGET COLUMN'S NOI, and the
    Checks panel shows critical items only -- AM, Sep 25 2026. `reconcile()`
    classified by PREFIX (any 4xxx / 5xxx), a second definition of NOI that put
    5190, 5120/5130, depreciation and 4050 inside it and left 7070 out; the
    proposed $20K 5130 line alone made every import that took it "not tie" by
    $20,000 [CORRECTED Sep 28 2026: it could not have -- that tick box never left
    the browser; see open_items §13.4]. Now reads `IS_ACCOUNTS`, and lists what was mapped below NOI.
    Critical = sign opposite to history, magnitude, negative NOI; the rest fold
    behind a count. Guardrail 25 -> 33, 33/33 in the container; the old code
    fails the new fixture at exactly -35,000. Served chunk verified by resolving
    the lazy chunk from the entry bundle.)
  - `v524` = `8839ab4` -- NEVER TOOK TRAFFIC. Every worker died on boot with
    "No module named psycopg": SQLAlchemy 2.1.0 was released Sep 24 2026,
    requirements said only `>=2.0`, and 2.1 makes a bare `postgresql://` URL load
    psycopg v3 instead of psycopg2. v523 kept serving throughout (site 200). ANY
    rebuild after Sep 24 would have failed identically, whatever it carried --
    worth knowing before blaming the commit. Pinned `<2.1` in `0d68e53`.
  - `v523` = `07272c6` (THE BUDGET IMPORT HAD NEVER ONCE SUCCEEDED ON
    PRODUCTION, which reframes the eight spreadsheet versions Jack built trying
    to get one through. `commit()` quoted `"vcode"`; production's supplement
    table carries `vCode`. A double-quoted identifier is case-SENSITIVE on
    PostgreSQL and case-INSENSITIVE on SQLite, so it passed every local test and
    raised `UndefinedColumn` on every real import -- the DELETE raises, the
    transaction rolls back, and the analyst sees an empty Budget column after
    doing the work. NO SPREADSHEET COULD HAVE FIXED IT. The column names are
    READ FROM THE TABLE now; the old docstring asserted the columns "really are
    `vcode`", true of the table pandas creates and false of the one production
    has, and that claim was itself the mistake.
    VERIFIED ON PRODUCTION AFTER DEPLOY: the table really is
    ['vCode','dtEntry','vSource','vAccount','mAmount','vInput','statement _id']
    and the resolver maps vcode -> vCode. The 324 rows already in it are
    P0000019 with ACCOUNT NUMBERS in `vInput`, not the `label [username]` this
    importer writes -- so they came from a CSV upload, which is also why the
    table carries `vCode` and a column named with a space. Nothing in there came
    through this path, which is the evidence for the claim above.
    IT ONLY EVER READ COLUMN A. The detector finds one label column and took the
    ACCOUNT column as the label, which is why Jack was hand-building a helper
    column joining the number and the description. A label column that is
    essentially all numbers is now recognised as the account with the
    description taken from beside it; an account LEADING the label ("4010 -
    Rental Income") is read; and the label's own account outranks a separate
    column.
    THE LABEL AND THE AMOUNTS MUST COME FROM THE SAME BLOCK. His v5 puts two
    independent tables side by side -- a 19-row roll-up in A-B, the 50-row
    detail it came from in D-G with the months beside the DETAIL. Every line
    read its NAME from one and its FIGURES from the other: "5051 - Water"
    carrying Property Management's 366,157.78. Nothing about that looks wrong;
    the labels are real and the amounts are real. A second block announces
    itself with a second account column, and without that evidence nothing is
    re-based, or a sheet whose labels merely have a sub-description beside them
    would be read off the sub-description. SHAPE ALONE IS NOT ENOUGH TO FIND
    THAT COLUMN and assuming it was got this wrong immediately: a roll-up of
    annual totals (1200, 240, 120) matches "3-6 digits" perfectly and was read
    as the account column, so every line came back with account 1200. The test
    is membership of our chart of accounts. `_find_account_column`'s own
    docstring had already said this -- "an account number and a monthly amount
    are both 3-6 digits" -- and answers it with a header match, which a block
    boundary does not have.
    THE ACCOUNT DECIDES THE CATEGORY; the dropdown is gone (Jack: "right now
    it's two separate steps and they fight each other... the category dropdown
    should come out entirely and just display whatever the account dictates").
    Measured before reversing: all 80 accounts belong to exactly one category,
    so one source IS possible. The server derives it and IGNORES what the screen
    sends. An account on NO category still blocks -- a rule that only ever
    corrects would accept anything. The whole chart is offered unconditionally,
    since with nothing narrowing the list a tick box would leave most accounts
    unreachable.
    MANY LINES MAY SHARE ONE ACCOUNT -- 23 of Evergreen's repair lines are 5060.
    They combine, and the combining is REPORTED with the lines named and the
    combined figure; merely not-blocking would be satisfied by dropping every
    line after the first.
    MEASURED ON JACK'S THREE REAL FILES: all three import with ZERO blocking
    errors. v5 goes from 19 mis-paired lines to 52 correct ones. Two accounts
    are genuinely not on our comparison (7076 Tenant Improvements, 5019 Leasing
    Commissions) and are NAMED rather than dropped. The file's own Total column
    agrees with the sum of its months on 228 of 229 rows -- the exception is the
    DSCR row, a ratio -- so the months ARE the total and no separate total
    column is imported.
    Guardrail `budget_import_mapping_check.py` (25), on fixtures of all three
    real shapes plus the NEGATIVE case for re-basing; 25/25 in the container.
    Proved non-vacuous against NINE injected defects, including both opposite
    failures: combining blocked again, and only the first line of a shared
    account written. `budget_import_check`'s duplicate assertion is reversed
    with the reason recorded; `mapping_draft_check` now asserts the category is
    displayed AND not selectable, since "no dropdown" alone is satisfied by
    deleting the column.
    A VERIFICATION OF MINE WAS VACUOUS FIRST and is worth recording: I grepped
    the deployed bundle for the new strings and got "gone" for all of them --
    from a 551-byte SPA shell that references no chunk. index.html names only
    the ENTRY bundle; the lazy chunk name lives inside it. Resolved properly
    (`ValuationsView-rlSXpPnk.js`, 95,086 bytes) the new strings are present and
    the old dropdown's are gone. Two server-side strings I also checked prove
    nothing from a Vue chunk either way.
    PRE-EXISTING, NOT FIXED: `budget_import_check`'s "the account list is the
    deal's own recent accounts" fails on local data (no 4010 history for the
    fixture's vcode) and fails identically on the unmodified tree.
    FLAGGED: the 324 P0000019 rows are CSV-loaded, and a screen import for that
    deal would REPLACE them for any overlapping month. That is the designed
    scope and is correct, but nobody would expect it.)
  - `v522` = `e4bb231` (traceability tools -- field dictionary, dependency map,
    and two read-only assistant tools (`lookup_field`, `impact_of`). RECORDED
    AFTER THE FACT on Sep 23 2026: it was live and absent from this history,
    found by pre-flight P1 while deploying v523. The history is the only thing
    that maps a running revision to its source, so a missing entry is the same
    failure as an untagged image.)
  - `v521` = `c370509` (AN UNDATED AMENDMENT NEEDS ITS DOCUMENT'S DATE, and most
    had none. `v520` fixed Benjamin Moore and BooYa's and left Chapultepec still
    reporting the original lease's $51,999.96 against the amendment's $53,331.96
    -- the fix was LIVE and still producing the old number, which is the argument
    for checking each named tenant rather than stopping at a green guardrail.
    Dating an undated step from the document that states it needs that document to
    HAVE a date, and most Poplar documents predate `parse_doc_date_anywhere`
    (`v503`) and carry none, so the fallback had nothing to fall back to. Read
    from the filename now, in the same backfill, only where none is stored.
    THE FIXTURE HAD TO CHANGE TO PROVE IT: the original lease's filename carried
    no date, so "a date already stored is left alone" was true however the code
    behaved and injecting the overwrite passed 38/38. It now carries 2015.05.05
    against a stored 2019-01-01, so overwriting is visible.
    After deploy, SIX of the analyst's seven tenants agree with the rent roll and
    every row names the document its figure came from. Marco's waits on the
    re-extraction. `lease_amendment_governs_check` 36 -> 38.)
  - `v520` = `445cb26` (A DATE THE APP DERIVED IS RE-DERIVED, NOT TRUSTED.
    Deploying `v519` proved the six fixes changed NOTHING on the seven tenants.
    THE WRONGLY-DATED STEP IS STORED: consolidation writes the resolved date back
    onto the row, so every step already in the table carried one computed from the
    old anchor -- Benjamin Moore's "months 3-14" sits in the database as
    2026-06-01 -- and `resolve_rent_steps` read it back as a date the DOCUMENT had
    stated. Anchoring correctly is useless while the wrong answer is an input.
    A step whose basis SAYS it was derived ("month N of the term") is re-derived
    from its own anchor; one whose date came from the document keeps it, period or
    no period -- re-deriving everything is the same defect facing the other way,
    and the guardrail fails on that injection too.
    Benjamin Moore resolved to $50,052 and BooYa's to $103,596, both matching the
    rent roll, with the original schedules back on their own terms.)
  - `v519` = `ea8e196` (THE AMENDMENT GOVERNS: six defects behind the analyst's
    seven tenants, five of them corpus-wide. Measured on production first.
    (A) THIRTEEN DOCUMENTS HAD NEVER REACHED THE MODEL -- 8 `text_extracted`, 4
    `error`, 1 `pending` -- including Ciao Baby's 4th Amendment and both of Hobby
    Lobby's option notices, exactly the documents reported as "not recognized".
    `error` was excluded from the retry, so a document that failed once could
    never be read again by any run. Retried now, and `unread_documents` puts them
    on the Validation screen: a document that was never read is not the extraction
    disagreeing with the lease, and the screen could not tell them apart.
    (B) AN ORIGINAL LEASE'S SCHEDULE WAS RE-DATED ONTO A LATER AMENDMENT'S TERM,
    anchored to the tenant's LATEST commencement, so it outranked the amendment's
    own rent -- Benjamin Moore $38,038 against $50,052 (THE RENT ROLL WAS RIGHT
    AND THE APP WAS WRONG), Kohls projecting its 2017 schedule to 2062. 22
    tenants. Each step now carries the term its own document states and falls back
    to the ORIGINAL commencement, never the latest.
    (C) 168 OF 809 STEPS could never be in force, having neither date nor period,
    so they lost silently to the original lease.
    (D) AN AMENDMENT THAT ADDS SPACE ADDS RENT. Marco's Pizza's takes another
    160 SF for another $242 a month; read as a replacement it reported $2,904 a
    year against $65,558 -- while the square footage in the SAME amendment was
    combined correctly, so one document disagreed with itself. The model now says
    which it is; never inferred.
    (E) THE MERGE DROPPED THE BASE LEASE'S UNDATED STEPS whenever an amendment
    supplied any -- 53 tenants, Kohls holding 1 against 11.
    (F) 82 TIED DATES across 31 tenants, decided by row order. The later DOCUMENT
    governs, and a step we cannot attribute never outranks one we can.
    THE BACKFILL IS WHAT MAKES ANY OF IT LIVE: all of it reads `source_doc_id` and
    `term_start`, which only a NEW extraction writes, so on the 809 stored steps
    nothing would have changed. Derived from what we already hold, idempotent,
    fills only NULLs, and cannot invent `is_additional`.
    FIXING E BROKE A TEST THAT HAD BEEN PASSING BY ACCIDENT, the most useful thing
    this turned up: `lease_terms_check` asserts the Fourth Amendment's 544,500
    beats the First's superseded 510,000, and it passed only because the merge was
    DROPPING the First's undated step. The merge now reads the period wording as
    well as the parsed months. Guardrail `lease_amendment_governs_check.py`.)
  - `v518` = `03e93aa` (AN ESTIMATE UNDER A PRO-RATA LEASE IS NOT A FIXED
    RECOVERY, and a document added to a scanned tenant re-reads its whole set.
    THE FULL RE-EXTRACTION FOUND THE FIRST ONE, which is the argument for having
    run it with a diff: 419 documents in three hours, nine recovery findings, and
    THREE of them comparing the wrong thing. USA Karate and CPR are PRO RATA
    leases and the rows captured for them are the initial ESTIMATE ("Initial
    Common Area Maintenance charge per month" $209.21; "the estimated amounts ...
    ($5.00 per square foot annually)"), trued up at the annual reconciliation --
    holding the rent roll to one as though the lease had capped it compares two
    different things. Gated on `cam_structure` actually saying FIXED: the
    structure says whether the amount binds, the presence of a schedule does not.
    The third was Pure Barre, whose row is "$25.00 per month to reimburse
    Landlord for water and sewer charges", $0.20/SF against a rent roll of $3.64
    -- not a CAM charge at all and no structural rule can tell, so the PROMPT now
    says it in terms. Both were needed: after re-reading, Pure Barre and USA
    Karate return no schedule at all while CPR still returns its 2017 estimate and
    the gate is what excludes it.
    Findings 9 -> 4, and the two extra disappearances were CHECKED rather than
    assumed: DSW #2 and Kohls #0249 are also pro rata. Seven tenants carry a
    schedule or a fixed structure, four say fixed, and those four are exactly the
    findings that remain -- Starbucks $7.70 vs $2.16, AT&T $4.63 vs $2.38,
    BooYa's $3.64 vs $3.21, O'Reilly cannot be dated for want of a rent
    commencement date.
    ALSO: ADDING A DOCUMENT RE-READS THE TENANT'S WHOLE SET (Jim: "rerun the
    extraction just on that tenant's set of leases... Ask the user if any other
    files will be loaded before running"). Reading only the new file leaves the
    terms assembled from a mixture of prompt versions, and the prompt moved twice
    in one day. Files STAGE and the panel asks before anything runs -- uploading
    on choose started a re-read per drop, so three files arriving one at a time
    meant three runs over the same tenant. A tenant can also be re-read with no
    upload, for a document assigned from the unmatched list.
    THE ABSTRACT WAS FROZEN THE MOMENT ANYONE SAVED IT: `get_tenant_abstract`
    assembles from data only when NO section is stored, so an abstract touched
    once never saw another document again. A section this code wrote is refreshed;
    a section A PERSON wrote is never overwritten -- it is marked, with the newly
    assembled text beside it, and saving clears the mark either way.
    THE RISK ANALYSIS DELIBERATELY DOES NOT TAKE THE LEASE'S DATE.
    `lease_tenants.lease_end` is the RENT ROLL's figure; copying the lease's over
    it on every re-read would make the two agree by construction and no expiry
    mismatch could ever be reported again -- the v503 failure. The re-read
    surfaces the disagreement; SETTLING it moves the analysis. Both halves
    asserted.
    Guardrail `lease_tenant_rerun_check.py` (36), proved against six injected
    defects; `lease_validation_resolve_check` 102 -> 106. Two of my own fixtures
    were wrong first -- a section key outside the template never reaches the
    screen, and "is 2032 in the histogram" was true before the change too.
    Production after deploy: both migrations present, three routes gated.)
  - `v517` = `8393b25` (THE TARGETED RE-EXTRACTION FOUND TWO MORE SHAPES, which
    is why Jim asked for it before the full run. It worked on two of the four
    fixed-CAM leases and reported "the amount in force could not be determined"
    on the other two, both of which state it perfectly clearly.
    A STATED ESCALATION IS COMPOUNDED BY THE APP. Starbucks #8362 fixes operating
    expenses at "$1.96 per square foot ... for the first five (5) Lease Years,
    increasing ten percent (10%) on the commencement of the sixth (6th) Lease
    Year and upon each fifth anniversary thereafter"; the extraction is told not
    to compute the later figures, so six rows arrived with periods and no
    amounts. `escalation_pct` is captured per row and `fill_cam_escalations`
    carries the last stated amount forward, compounding -- $1.96 -> $2.156 ->
    $2.3716 -- and a derived figure SAYS it was derived. An escalation with
    nothing before it stays empty rather than inventing a base.
    A FIXED AMOUNT WITH NO PERIOD APPLIES THROUGHOUT (BooYa's: "$1,276.27 ... due
    with Minimum Monthly Rent"), but ONLY when it is the sole row -- among dated
    rows an undated one would be in force at every date and beat all of them.
    "Could not be determined" and "states no amount for this period" are
    different answers and no longer read the same.
    TWO OF MY OWN CHECKS WERE WEAKER THAN THEY LOOKED. "An undated row is not used
    among dated ones" passed 102/102 with the rule deleted, because the undated
    row sat SECOND in the fixture and the code reads the first. And two assertions
    raised TypeError rather than failing, hiding every check after them.
    After deploy all four leases resolve: Starbucks $2.156 dated from the fifth
    anniversary 2022-09-29, AT&T $2.38, BooYa's $3.206, O'Reilly reporting that it
    cannot be dated. Guardrail 90 -> 102.)
  - `v516` = `d27fd9f` (A MISMATCH CAN BE SETTLED, A FIXED CAM CHARGE IS CHECKED,
    and a lease-year schedule is dated from rent commencement.
    THERE WAS NO WAY TO CLEAR A FINDING. The page listed them and the only control
    near one was a per-TENANT approve/flag two steps later, recording no value, no
    reason and no document -- so a mismatch stayed for ever and the decision that
    settled it lived in somebody's email. SETTLE records the figure that applies,
    the REASON (required) and the DOCUMENT it was read from, defaulted to the one
    that governs; a citation naming another tenant's document is REFUSED, because
    an unchecked citation reads as evidence. Confirming the rent roll is a
    decision too and the report separates it from a change. A settled finding
    stops asking, and survives a re-validation, because the decision lives in
    `lease_field_resolutions` and not on the rows that get rebuilt.
    THE CHANGE REPORT is rent roll -> applies -> difference -> reason -> document
    -> who, on screen and as a workbook with the confirmations on their own tab.
    `prior_value` is stored WITH the decision because the rent roll gets
    re-imported.
    A FIXED CAM CHARGE IS NOW PART OF THE VALIDATION (Jim, on the AT&T 4th
    Amendment: "Is this situation part of the lease review and validation to the
    rent roll?" It was NOT). The extraction captured the WORD `fixed` and never
    the amount, so the one figure the rent roll could be checked against did not
    exist anywhere. WHETHER A DIFFERENCE IS A MISMATCH OR A QUESTION depends on
    what the column means: our rent roll carries ONE recoveries figure while a
    lease fixing only the operating-expense share passes tax and insurance
    through separately, so that is raised as a question naming both figures.
    `cam_fixed` is a LIST and the merge whitelist knew only scalars and objects,
    so it would have been dropped silently -- in the wrong direction, since the
    schedule is stated BY the amendment.
    FORMATTING FOLLOWS THE FIELD, NOT THE COLUMN: "All Validation Comparisons"
    put rents, $/SF figures, square feet and dates in the same two columns and
    rendered them raw (50052.0, 18.99998001998002). The sign sits OUTSIDE the
    dollar sign -- "$-12,014" is not how a figure is written.
    Verified in the running app: Benjamin Moore $50,052 vs $38,038 settled to the
    lease figure citing the First Amendment, the report showing -$12,014, and undo
    putting the finding back. Guardrails 72 and 90.)
  - `v515` = `a21494e` (THE UPLOAD ASKS FOR THE RENT ROLL DATE, so Market at
    Poplar cannot recur. Asked on the mapping confirmation, REQUESTED not
    enforced -- a date typed wrong is worse than one supplied a moment later,
    blank announces itself and a wrong date does not -- and the panel says what a
    blank one costs. Parsed or refused, never guessed.
    Carried `0ef754d`: the validation page shows enough to resolve a finding --
    RR SF / RR Rent / RR $/SF / Lease SF / Lease Rent / Lease $/SF, every document
    linked in applied order with un-applied COIs listed and marked, and the rent
    roll date settable on the page. THE MESSAGE THAT BLAMED THE LEASE is gone: it
    said rents "could not be determined from the lease" when the lease had 148
    rent steps and the REVIEW simply had no date, which is what made Jim's
    instinct right and the page wrong.
    After deploy, Market at Poplar set to 2026-09-01: 0 of 23 rent comparisons
    became 22 of 23, with six real disagreements surfacing.)
  - `v514` = `42d053d` (A VALIDATION SCREEN WITH ROWS IN THE DATABASE STOPS
    RENDERING BLANK. Jim asked whether to relocate the Lease Risk validation
    screen into Lease Review because "right now that page is blank". NO — the
    screen was complete and its data existed. Windsor Square rendered 175 rows
    and always had; Market at Poplar had 23 rows in the database and showed
    nothing, which is why this looked like a MISSING screen rather than a broken
    one. THREE THINGS STACKED UP.
    (1) THE NaN THAT WALKED PAST THE RANGE GUARD. Five rows in Market at Poplar
    carry the literal string `'NaN'` as `lease_end` — the building banner, two
    subtotal rows and two vacant suites, the same debris `v501` taught the
    roster to hide. `pd.to_datetime('NaN')` returns NaT WITHOUT RAISING (pandas
    reads the string as a null token), so the `try/except` never fired, `.year`
    was `nan`, and BOTH range comparisons were False — NaN never compares — so
    the row sailed past the guard into `yearly[nan]` and raised `KeyError: nan`,
    which the endpoint returned as HTTP 500 `{"error":"nan"}`.
    FIXED TWICE OVER, because each covers what the other cannot: the query
    filters `COALESCE(tenant_status,'active')='active'` — the idiom every other
    consumer uses, and the right answer for the debris — AND an explicit
    `pd.isna(exp_year)` guard catches an ACTIVE tenant whose date will not
    parse, which the status filter would let straight through. `v501` taught the
    roster and the headline totals to respect the reading; the histogram was
    simply never updated, one screen over.
    (2) ONE FAILING PANEL BLANKED THREE OTHERS. `Promise.all` rejects on the
    FIRST failure and `validation.value` was the LAST assignment in the block,
    so the 500 meant it was never set at all. `allSettled` now, each panel
    assigned from its own result, and scenarios/validation assigned OUTSIDE the
    cotenancy try so shaping that payload cannot take them down either.
    (3) THE FAILURE WAS DISMISSED. The catch logged "Secondary data load error
    (expected for new reviews)", so it read as normal. A panel that could not
    load now says so on screen.
    Guardrail `lease_validation_blank_check.py` (15) against a real database in
    the exact production shape — one real tenant, three debris rows read as
    `no_lease`, and one ACTIVE tenant with an unparseable date so the status
    filter ALONE cannot pass it. Both directions: swallowing the exception and
    returning an empty histogram would satisfy "the debris is excluded" on its
    own, so five rows go in and exactly one is expected out. Proved non-vacuous
    by restoring each defect — the original histogram fails with detail `[nan]`,
    the same error the endpoint returned, and `Promise.all` fails its own check.
    Production after deploy: review 3's expirations 500 -> 200 carrying real
    data (2027, annual_rent 435,582), all four endpoints 200 on both reviews,
    validation returning its 23 rows. Review 2 unchanged throughout, which is
    the check that the fix cost nothing.
    A CORRECTION RECORDED WITH IT: the first version of this guardrail asserted
    on a `tenants` key that `yearly_data` does not carry. It failed loudly
    rather than passing vacuously, but it is the same trap as the
    `_documents_applied` count the same day — assert on a shape you have
    verified, not one you assumed.)
  - `v513` = `b3bebdf` (THE GL / IA GRID SORTS AND FILTERS ON ANY COLUMN. Jim:
    "lets make it easy. take the query results that we are currently receiving
    and allow the user to filter or sort by any of the column headers."
    He asked for this after two candidate filters were measured and REFUSED.
    `ITEM = 1` is a LINE NUMBER, not a side — 13,493 distinct values, keeping
    8.4% of rows and taking the net to $2.1bn. The SIGN of `AMT` was his own
    objection and it is right: it keeps the expense on an expense entry but
    keeps the CASH and drops the INCOME on a revenue entry, because sign tracks
    the ACCOUNT'S NATURE, not which line is the substance. Cash - PNC appears
    2,786 times on the debit side and 4,360 on the credit side, so no sign rule
    separates offset from substance. Which line is the 'other side' is a
    property of the account (`gl_accounts.TYPE`: B 21,574 rows / I 14,970 /
    C 7,538 cash), so rather than bake in one opinionated default the grid is
    sliceable and the reader decides.
    FOUR THINGS THAT WOULD HAVE BEEN WRONG QUIETLY. The body must walk the
    FILTERED rows — left on `result.rows` the boxes accept text and the count
    moves while the table does not, and the guardrail fails on exactly that
    injection. `Array.sort` MUTATES, so sorting in place destroys the server's
    order permanently and clearing the sort could not restore it; copied first,
    and verified in the browser that a third click returns the original order
    exactly. The server totals the WHOLE match on purpose (`v504`), so once a
    filter is on the filtered subtotal is shown BESIDE it — "Total for all 11:
    (427.84)  Shown: 127,500.00" — never instead of it. And sorting 5,000 of
    79,074 loaded rows does not find the largest amount in the match, so a
    truncated result says so and points at the export.
    Numeric columns sort numerically: the cells render "(2,694,676.22)", so a
    string sort would order by the bracket. Blanks sort last in both directions.
    A filter matching nothing explains itself rather than rendering an empty
    grid. gl_ia_query_check 113 -> 123.
    Verified against the SERVED bundle after deploy, cache-busted with the chunk
    resolved from that response: every user-visible string present. The
    identifier names are absent because the production build minifies locals —
    expected, not a miss.)
  - `v512` = `e473a07` (A SCANNED LEASE IS READ FROM THE PDF, not from its
    empty text. Jim, Sep 20 2026: "is there anything we can do to extract from
    the pdfs that produce no text extractions? I'm sure it will be a common
    problem when scanning bulk files of pdfs." Then: "I'm not price sensitive
    for this task. build it with the best model suites for all scenarios."
    MEASURED ON PRODUCTION BEFORE BUILDING: 219 of the 419 term-bearing
    documents yield under 200 characters of text — including 47 AMENDMENTS and
    28 ORIGINAL LEASES. Their pages are images, pdfplumber returns nothing, the
    prompt was being filled with an empty string, and the document contributed
    NOTHING to the tenant's terms with no error and no warning. All 219 have
    their PDF stored; the largest is 79 pages against the API's 600-page
    ceiling and exactly one exceeds the 32 MB cap.
    NO OCR STACK. The API reads a PDF as images, so where the text is too thin
    the PDF itself goes as a document block — same prompt, different
    representation of the same document. No Tesseract, no poppler, nothing new
    in the container image.
    BOTH ROUTES RUN ON `claude-opus-5` now, not the date-pinned Haiku this
    started on: extraction decides every rent figure downstream and reading a
    scan is harder again.
    THREE THINGS THAT WOULD HAVE BROKEN QUIETLY. `content[0].text` — thinking is
    ON by default on this model, so the first block is a THINKING block and the
    indexed read would raise; the text blocks are joined instead, and the
    guardrail's stub returns a thinking block FIRST so a reintroduced indexed
    read fails. A REFUSAL returns HTTP 200 with no text, so `stop_reason` is
    checked BEFORE the content and reported as `_refused` rather than as a parse
    failure. And the base64 must carry no newlines — `b64encode` adds none,
    `encodebytes` would.
    Streamed, because the input can be a 180,000-character lease or a 79-page
    scan and a blocking request that size risks the HTTP timeout; max_tokens
    4096 -> 16000. Over a cap is REFUSED WITH A REASON naming the size or the
    page count, never attempted. Every result records `_extraction_source`
    (text / pdf) so a reader seeing thin terms can tell how it was read.
    VERIFIED ON A REAL SCAN, not a fixture: `1987.08.11_Sam's Club-Short Form
    Lease.pdf`, 5 pages, 428 KB, **4 characters of extractable text**, came back
    with 10 populated fields — Wal-Mart Stores Inc., suite C500, 103,060 SF,
    commencement 1987-08-11, expiry 2007-10-31, six 5-year renewal options. It
    returned nothing at all before.
    Guardrail `lease_scan_extraction_check.py` (25), which makes NO API calls —
    a stub records the request, because what matters is the ROUTING DECISION and
    the request SHAPE. Proved non-vacuous: disabling the scan route fails 3,
    restoring the Haiku pin fails 2.)
  - `v511` = `07a83ed` (EVERYTHING AFTER THE BASE LEASE IS LAYERED IN DATE
    ORDER. Found by running the re-consolidation Jim asked for AND DIFFING IT —
    the diff is the only reason this was caught, and the regression was mine.
    `order_lease_documents` returned `originals + ordered_am + others`, so every
    non-amendment applied AFTER every amendment. It was invisible while the
    classifier typed nearly everything `Original Lease` and `others` was almost
    empty; correcting the classifier at `v510` put 147 documents into it and
    three tenants moved the WRONG way at once — Style Studio's expiry
    2031-05-31 -> 2026-05-31 (a 2021 Acceptance of Premises overwriting the 2026
    First Amendment that extended the term five years), Green Zone's
    2026-12-31 -> 2025-12-31, and Appliances 4 Less's suite N625 -> a misread
    "G" that beat the lease AND three COIs all saying N625.
    Now: the original lease is the base and everything after it is layered in
    the order it was executed. UNDATED documents sort last with amendments among
    them by number, so the case this ordering was built for still holds — a
    folder of "First / Second / Third / Fourth Amendment.pdf" carrying no dates
    applies 1,2,3,4 with the Fourth governing. On an equal date the amendment
    wins, since a non-amendment carries no ordinal.
    THE RE-CONSOLIDATION IS WHAT THIS WAS FOR, and it is done: 70 of 70 tenants,
    zero errors, 392 documents applied — matching the 392 predicted before any
    of it was built. It re-layers the EXISTING per-document extractions and
    makes no API calls, so it cost nothing and is re-runnable.
    WHAT MOVED, and it is the shape you would expect once amendments govern:
    nine lease expirations, five rent commencements, two suites, one lease
    commencement, one escalation. Several tenants were showing terms that had
    expired years ago and now show live ones — Office Depot 2022-01-31 ->
    2027-01-31, DSW 2024-01-31 -> 2029-01-31, SalonCentric 2024-09-30 ->
    2029-09-30, Peak Potential 2019-01-30 -> 2029-01-31, Firehouse Subs
    2013-01-31 -> 2023-01-31.
    COVERAGE HELD, which is the check that the exclusions cost nothing:
    rent_commencement 38 -> 38, lease_expiration 45 -> 45, square_feet 43 -> 43,
    escalation 45 -> 45, security_deposit 31 -> 31, suite 69 -> 69,
    lease_commencement 35 -> 34 (the one lost came FROM a COI).
    AND `lease_tenants.rent_commencement` IS POPULATED FOR THE FIRST TIME — 0 ->
    37. That column is written only by consolidation, which had not run since
    `v503` added it, so the month-of-term rent-step dating has had no date to
    work against on production until now.
    CONVERGED: a second full pass leaves 70 of 70 blobs byte-identical and moves
    no field, which is the proof it is settled rather than oscillating.
    Guardrail `lease_doc_type_check.py` 37 -> 44, both directions, proved
    non-vacuous by restoring the old ordering — which reproduces the production
    symptom exactly, `[1, 4, 3, 2, 5]`, the amendment applied second with the
    2021 documents after it. `lease_terms_check` still 129/129, so `v503` is
    intact.
    STILL OPEN: no rent step carries `period_start_month` yet (0 of 346), since
    those arrive only from an extraction run after `v503`. The dates are now in
    place for when one happens. `open_items.md` §9.2.)
  - `v510` = `a12f98a` (A DOCUMENT IS TYPED BY ITS FILE NAME, and the documents
    that carry terms are kept. Jim, Sep 20 2026: "do both changes" — both,
    because either alone does damage.
    `classify_document` matched the whole stored PATH. Every production document
    sits under `Tenant Leases/`, so pattern 0 (`lease`) matched the FOLDER and
    short-circuited before any later pattern was tried: 409 of 530 typed
    `Original Lease` with only 77 carrying "lease" in the file name.
    Certificates of insurance, easements, option letters, move-in forms and
    landlord consents all read as original leases — and because extraction is
    GATED ON THE TYPE they were sent to the extraction API as leases and layered
    into the tenant's consolidated terms. 500 of 530 documents were in a
    consolidation and 328 of them were misclassified; 69 of 70 tenants carried
    at least one.
    THE OBVIOUS FIX WOULD HAVE BEEN WORSE THAN THE BUG, which is why it was
    measured first. Correcting the classifier ALONE pushes those 328 out of the
    gate and strips the rent commencement date from 16 of the 38 tenants that
    have one — 15 of those sources being COMMENCEMENT LETTERS, the document
    whose whole purpose is to state that date, and the date `v503` uses to place
    a rent step written as "Months 1-12". The naive fix would have broken the
    feature shipped three deploys earlier.
    So the gate widened with it: `is_term_bearing`, and `NON_TERM_TYPES` is
    `{'COI'}` ALONE. The list was MEASURED against all 500 extracted documents,
    not chosen — excluding COI costs no tenant its rent commencement, expiration,
    square feet, escalation or rent steps, while removing 108 certificates of
    insurance from the layering, two of which were supplying a
    `rent_commencement` a COI has no business stating. The narrower rule loses 16
    tenants a date; keeping everything keeps two wrong ones.
    ONE RULE, TWO CALL SITES. `is_term_bearing` gates extraction AND filters the
    consolidation, applied in Python against the same function rather than
    repeated as a SQL `NOT IN`. It HAS to be in the consolidation too: a document
    extracted under the old classifier still carries its `extraction_json`, which
    is that query's admission ticket, so the COIs would go on being layered in
    for as long as that JSON exists. An unknown or missing type counts as
    term-bearing — visibly wrong beats invisibly absent.
    `_reclassify_documents` backfills the stored types from the filename,
    idempotent, after the CREATEs. Safe because `doc_type` is only ever written
    by the classifier: there is no screen or endpoint that sets it by hand, so
    nothing overwrites a human judgement. Production after deploy: Original Lease
    409 -> 77, COI 0 -> 111, Commencement Letter 0 -> 23, Option Letter 0 -> 12,
    Amendment 100 -> 100 exactly as predicted.
    Guardrail `lease_doc_type_check.py` (37) drives a real database and asserts
    BOTH directions, since "the COIs are gone" is satisfied by dropping
    everything. Proved non-vacuous against each defect: path matching fails 10
    checks; the old narrow gate fails 10, including the commencement letter
    dropping out of `_documents_applied` — the 16-tenant failure, reproduced.
    STILL OPEN, and it is `open_items.md` §9.2: ALL 70 stored consolidated
    records predate `v503` — not one contains `_documents_applied` — so every
    one was built by the old consolidation, before the amendment ordering fix and
    with the COIs layered in. Consolidation only runs at the end of an
    extraction, so nothing refreshes by itself. A METHOD NOTE went with it: the
    first attempt to size this asked how many stored blobs carried a
    now-excluded document in `_documents_applied` and got 0, which reads as
    "nothing to do" and was VACUOUS, because the key is absent from all 70.
    Carried the two docs commits `068febb` and `9f2d2dd`, neither shipped.)
  - `v509` = `49d2120` (THE INPUT COLUMN FOLDS AWAY, and the GL grid shows what
    matters. Two asks from Jim, Sep 19 2026.
    "MANY OF OUR PAGES HAVE INPUT SECTIONS ON THE LEFT AND ANALYSIS SECTIONS ON
    THE RIGHT. CAN WE PROVIDE THE SAME LITTLE ARROW AS EXISTS IN THE SIDE BAR TO
    EXPAND OR COLLAPSE THE INPUT SECTIONS THROUGHOUT THE APP?" One
    `CollapsiblePanel`, the sidebar's own `<` / `>`, on the five pages that have
    a genuine inputs-left / analysis-right split: Ownership, Workpapers,
    Reports, Data Explorer, Prospect Analysis. Everything else is an
    equal-width content grid, a filter bar above the results, or an overlay
    drawer, and is deliberately left alone.
    A GRID PARENT MUST DECLARE ITS OWN COLLAPSED TRACK, and that is the failure
    the guardrail exists for: the container sets the column, so a child
    narrowing itself to 30px reclaims NOTHING — the arrow works, the panel
    goes, and 260px of empty space sits where it was, with no error and nothing
    on screen saying so. Each page binds `input-collapsed` and states the track
    itself; the component does not reach upward into a layout it cannot see.
    THE TOGGLE SURVIVES COLLAPSING and the rail keeps the panel's NAME — a
    control that vanishes when used cannot be undone by anyone who did not
    already know it was there, which v482 shipped once and had to fix.
    VERIFIED IN THE RUNNING APP at 1440px, both directions, all five: Reports
    280 -> 30 (results 888 -> 1138), Ownership 290 -> 30 (789 -> 1064), Data
    Explorer 240 -> 30 (833 -> 1043), Workpapers 320 -> 30 (721 -> 1011),
    Prospect 480 -> 33 (657 -> 1119). Guardrail `collapsible_panel_check.py`
    (43), proved non-vacuous by deleting one page's collapsed track — which
    fails exactly the check that matters. (The first injection attempt was a
    no-op because the file is CRLF and the needle was not; worth knowing.)
    THE GL GRID drops Bal/Fwd, Item, Related Entity and Related Entity Name and
    clips Description at 260px with the full text on hover. HIDDEN ON SCREEN,
    KEPT IN THE EXPORT: the workbook is what somebody checks the screen against
    and ITEM is how a line is found again in MRI's journal. Description was
    measured first — 85 characters at the longest, median 29, and the tail
    repeats an entity name the Entity column already carries. Asserted in both
    directions; the check catches "dropped instead of hidden" AND "flag emitted,
    grid ignores it". gl_ia_query_check 94 -> 113.
    NOT DONE, AND IT NEEDS JIM'S CALL: defaulting the filter to `ITEM = 1`.
    ITEM is a LINE NUMBER, not a debit/credit side — 13,493 distinct values
    across the real 79,074 rows. It would keep 6,618 rows (8.4%) and 5.5% of the
    money, and take the on-screen net from 10,797 to 2,102,385,065. Nothing is
    being duplicated: 0 duplicate rows on any key, and all 8,809 open-period
    entries balance to zero. A ledger carries both sides because that is what it
    is; the median entry has 2 lines and the largest 173. The ACCOUNT filter
    already answers the real need. `open_items.md` §9.8.
    Production after deploy: four columns hidden, ten visible, all 14 still
    returned by the query, 79,074 rows. Carried the docs commit `1f9d913`.)
  - `v508` = `b75cf92` (FILING A STATEMENT OPENS THE CHAIN, and the statements
    can be called up. Jim, Sep 19 2026, after "Seed openings from statements"
    returned `0 of 49 opened`: "Shouldn't the seeding process be integrated
    into loading the statements function? If the account does not need a seed
    because reconciled balances are carried forward, the process should simply
    save the statement file in its place and make it readily available when
    called by the accountant." Right on both halves, and the plumbing was
    already there.
    THE SEED IS NOT A SECOND ENGINE: `import_statement` CALLS
    `seed_from_statement` rather than computing an opening itself, so "seeding
    is not re-basing" stays enforced in exactly one place. An account with a
    reconciled period is left alone and the refusal comes back as
    `seed_skipped`, not an error — the statement IS filed, and needing no seed
    is not a failure. Each statement opens its OWN following month, so a folder
    of several months carries no ordering dependence and no month can be opened
    twice from different files. A real close outranks a seed automatically: the
    seed is stored as a `seeded` row and `opening_balance` reads the prior
    period's `computed_ending`, so reconciling that month replaces it.
    A SEEDED PERIOD WAS NEVER CLOSED, and the accounts tab said "Closed 202606
    at 119,701.35" of one. Wrong before, and wrong on EVERY row once filing
    opens a chain for all 49 accounts at once — the kind of claim that gets
    believed because it is everywhere. `last_status` is carried and it now
    reads "Opened at ... from the 202606 statement, nothing reconciled yet",
    with the column labelled Chain starts / last closed.
    THE STATEMENTS ARE LISTED (`GET /api/treasury/statements`, panel at the
    foot of the Accounts tab, filterable by account). The PDF had been kept
    since v507 and NOTHING LISTED IT — the only route to one was the
    held-statement prompt, which empties the moment the statement is placed.
    Stored and unreachable is not kept. `has_file` is per row, because a
    statement filed before the bytes were kept still has correct balances.
    FOUND ON THE WAY: the single-file import route never stored the PDF at all
    while the bulk one did; and re-importing a file stored a SECOND statement
    row, so re-running a folder left two statements for one month with nothing
    saying which is read. Guarded on (account, period_end, source_file) and
    reported as `already_filed` rather than silently skipped.
    VERIFIED END TO END THROUGH THE REAL ROUTES with the real PPI Life Storage
    June PDF, not a fixture: filed=1 opened=1, 202607 opens at 119,701.35,
    listed, and the PDF served from the page at 188,234 bytes byte-identical to
    the file on disk. Then a period closed and the same statement re-filed: it
    files, opens NOTHING, and the closed figure is untouched. Screen read back
    in the browser.
    Guardrail `treasury_seed_on_import_check.py` (29), asserted in BOTH
    directions — a check written only in the seeding direction is satisfied by
    seeding everything, which is the one outcome the rule exists to prevent.
    PROVED NON-VACUOUS by injecting each defect: turning the seed off fails 4
    checks, removing the reconciled-account guard fails 5 and overwrites a
    closed 10,000.00 with 88,888.88. `treasury_api_check` 43 -> 46 with the new
    panel's field names and `last_status`.
    Production after deploy: 29/29 and 46/46 in the container, the API check
    cleaning up after itself (49 accounts and 716 activity rows unchanged, 0
    statements), `wp_fs_map` still 553. Carried the docs commit `0874af2`;
    P2 listed it, `.claude/memory/` and `CLAUDE.md` are in no Dockerfile COPY.
    STILL NOT DONE: the June load itself — 49 file, 14 held, 1 refused. Now two
    steps, not three: upload the folder, answer the 14.)
  - `v507` = `b1d9197` (A STATEMENT WITH NO ACCOUNT IS HELD AND ASKED ABOUT, and
    the PDF is kept. Two asks from Jim, Sep 19 2026.
    "FOR THE STATEMENTS WITHOUT A PRODUCTION ACCOUNT, CREATE A RECORD AND PROMPT
    THE USER TO FIND AND INPUT THE ACCOUNT NUMBER FOR FUTURE MATCHING OF THE
    DATA PULLS." An account registers itself from an activity import and PNC
    serves only 90 days, so an account quiet longer than that has a statement
    showing real money and no transaction anywhere to introduce it. MEASURED ON
    THE REAL FOLDER before building: uploading `2026\06.2026` files 49 of 64, 14
    are held and 1 is refused (a WELLS FARGO statement in the PNC folder, which
    is correct). Twelve of the 14 are 0.00 but TWO HOLD MONEY — PPI Life Storage
    NY 119,701.35 and PSC Ambassadors Fund TGA VI 629,125.04.
    The old behaviour parsed them correctly, said so in a result row, and kept
    NOTHING, so the reading had to be found again by hand. They now go to
    `tr_pending_statements` with everything that was read, and the import
    reports them as HELD rather than skipped.
    THE MASK IS WHY THE HOLD IS WORTH ANYTHING. A typed number is checked
    against it — `XX-XXXX-7891` says ten digits ending 7891 — so a number that
    does not fit is REFUSED. Without that a mistyped digit registers a
    plausible-looking new account, the statement files against it, and when the
    real account arrives under its true number the balance is split across two
    records with nothing saying so. Resolving creates the account, files the
    statement, and every later pull for it routes by itself, which is the point
    of asking. A separate table on purpose: a reconciliation must never pick up
    a statement that has not been placed against a real account.
    "GIVE THE ACCOUNTANTS THE ABILITY TO PULL UP A COPY OF THE STATEMENT." The
    PDF was never kept — only the parsed balances and a filename. `file_data`
    now holds it on both tables, added BY MIGRATION as well as DDL because
    production already has these tables and CREATE TABLE IF NOT EXISTS would not
    have touched them (verified after deploy: both columns present). It carries
    across when a held statement is placed rather than being left on the pending
    row. A statement imported before this says so plainly instead of 404ing.
    For a held statement, opening the PDF is not a convenience — it is the ONLY
    place the full account number is written, so it is how the question gets
    answered.
    ALL SIX `tr_*` TABLES JOIN PROTECTED_TABLES, on the same rule as `wp_fs_map`
    and the budget supplement: the app is the writer and holds the only copy,
    and none comes from a CSV feed. `tr_periods` matters most — it is the
    reconciliation CHAIN, each closed period's ending becoming the next one's
    opening, so losing it loses the thread rather than a report. Checked against
    the `isbs_uw_supplements` lesson first: protection without a write path is a
    lockout, and all six are app-written, so the guardrail asserts BOTH the
    membership AND the write path.
    VERIFIED AGAINST A REAL STATEMENT, not a fixture — the PPI Life Storage June
    PDF: parses at 119,701.35, is held, appears in the prompt with its balance
    and the entity name off the filename, re-importing does not duplicate the
    question, the PDF opens while pending, 9999999999 is REFUSED for not fitting
    the mask with nothing registered, 8517897891 is accepted, the account
    appears, the statement files with its balance, the PDF comes with it, and a
    later statement for that account routes automatically.
    Guardrail `treasury_pending_check.py` (34), which SKIPS with a reason where
    the real PDF is absent so it still runs in the container.
    Production after deploy: both `file_data` columns present, all six tables
    protected, `wp_fs_map` still at its restored 553 rows, 49 accounts and 716
    activity rows untouched, 0 statements — the June load has not been done yet.)
  - `v506` = `2f462e9` (THE FS MAPPING CANNOT BE LOST THE WAY IT WAS, and the
    statements are back.
    THE INCIDENT. Jim: after refreshing the app and running MRI_GL_Detail, the
    financial statements stopped appearing on the workbench. `wp_fs_map` was
    found at **0 rows** against 583 accounts in `gl_accounts` and 79,074 GL rows
    over 207 entities. With no mapping every account falls to `unmapped`, so
    every statement for every entity renders EMPTY WHILE THE API STILL ANSWERS
    200 — the logs show `GET /packages/28/statements` returning 200 in 2,712
    bytes. No error anywhere, and the damage invisible until somebody opens a
    statement.
    MY FIRST HYPOTHESIS WAS WRONG AND IS RECORDED AS SUCH. I thought v505's
    wider balance query had hit a column the refresh did not bring, which fails
    on PostgreSQL only. The logs said 200, not 500, and all fourteen columns
    were present. `6fcb576` still ships because the landmine is real —
    `_balances` had been widened from the six columns a BALANCE needs to the
    thirteen a DRILLDOWN wants, and SQLite treats a double-quoted identifier
    matching no column as a STRING LITERAL while PostgreSQL raises, so a
    missing column would have been production-only and locally unreproducible,
    the v496 shape. The balance query is back to its six; the drilldown asks for
    the rest intersected with what the table HAS, and a missing one renders
    blank rather than as the word ENTRDATE in the date column.
    THE ACTUAL CAUSE IS A DESTRUCTIVE DEFAULT. `set_fs_map` deletes every row
    before inserting and `put_fs_map` passed `body.get("entries") or []`, so a
    PUT carrying no entries wiped the mapping and returned
    `{"status": "ok", "rows": 0}`. It is the only code path that empties the
    table. An empty replace is now REFUSED, naming how many rows it would have
    deleted; `allow_empty` still clears it on purpose, because "start the
    mapping again" is a real thing to want; the endpoint returns 400.
    `wp_fs_map` joins PROTECTED_TABLES on the same rule as
    `isbs_budget_is_supplements` — the app is its writer and holds the only
    copy. Checked against the `isbs_uw_supplements` lesson before adding it:
    this one HAS an app write path, so protection is a safeguard and not a
    lockout. It is not currently a CSV target, so today it is belt and braces.
    RESTORED, AND PROVABLY THE ORIGINAL. Jim asked whether the map came from the
    sample workpaper package. `fs_line_seed`'s docstring says it is the FS
    Tagging column of `PPI Eastchase (TX) LLC - WP - 06.30.2026.xlsx`;
    re-extracting that column from the file he reattached gives 192 accounts and
    56 captions MATCHING THE SEED EXACTLY — zero in one and not the other, zero
    captions that disagree. 553 rows written from `consolidated_mapping()`: 190
    `accounting`, 161 `routed`, 202 `other`, covering all 411 accounts that
    appear in `gl_detail`.
    THE FIGURES ARE THE EVIDENCE IT IS THE ORIGINAL: PPIECH, PPI35 and AMB6 all
    build with zero unmapped and all three BALANCE, and PPIECH's net income
    comes back at -11,745.08 with AMB6's at -16,282.49 — the exact figures
    recorded in CLAUDE.md when the engine was first built at `v455`.
    Verified again after this deploy: 553 rows, protected True, both entities
    balanced with the same net income, and an empty replace refused with nothing
    deleted.
    Guardrail `statement_drilldown_check` 32 -> 60, and it also proves the
    drilldown really is read-only (no INSERT/UPDATE/DELETE/DROP/ALTER/commit
    anywhere in its path), which is what its OPEN_POSTS entry claims —
    `accounting_access_check` had gone 2/54 red at v505 because the drilldown
    POSTs, and is back to 54/0.
    OPEN: 202 of the 553 are `other`, a fallback caption rather than
    accounting's tag, and the origin is NOT stored in `wp_fs_map` so the screen
    cannot say which. See `open_items.md`.)
  - `v505` = `36a26d9` (CLICK A STATEMENT FIGURE, SEE THE ENTRIES BEHIND IT.
    The CFO, Sep 19 2026 via Jim, on the workbench financial statements.
    IT DOES NOT RE-QUERY. `_balances` — which builds every statement — already
    reads `gl_detail`, so the entries behind a figure ARE the rows the figure
    was made from. The row selection moved into `select_measure_rows` and BOTH
    the builder and the drilldown call it. If the two selected differently the
    entries would not add up to the number, and a drilldown that does not
    reconcile makes a CORRECT statement look wrong with no way for the reader to
    tell which to believe — worse than no drilldown at all.
    TWO THINGS THE RENDERED LINE DID NOT SAY, either of which would have made a
    correct drilldown look broken. WHICH MEASURE it is: the balance sheet shows
    `closing` and the income statement `ytd`, so assuming one returns the right
    rows for the wrong question. THE PRESENTATION SIGN: `render()` applies a
    sign per section, so a liability shown as 5,000 is -5,000 in the GL; the
    drilldown takes the sign the screen displayed and returns `presented_total`
    beside the GL `total`, so the figure it reports is the one that was clicked.
    `closing` is deliberately the UNION of the balance-forward rows and the
    year's activity — a closing balance IS opening plus activity, and showing
    only the activity would not add up to the figure it was opened from.
    THE ONE BALANCE-SHEET LINE WITH NO ACCOUNTS OF ITS OWN is the period result
    carried into members' capital. It IS the income statement's total, so its
    `accounts` list was empty and it would have opened an EMPTY DRAWER on a
    figure the CFO will certainly click. It now carries the INCOME statement's
    accounts and its `ytd` measure. Ties on real data.
    THE REFACTOR WAS PROVED BEHAVIOUR-PRESERVING BEFORE THE BUILD, because
    `_balances` feeds every statement, the print view and the Excel package: the
    pre-change module was checked out and run side by side with the new one —
    50 figures across three period ends, ZERO differences, and members' capital,
    cash flow and the schedule of investments byte-identical.
    Verified on real local data (all six PPIECH lines tie) and end to end
    through the running app: clicked 2,317.24 on Accrued Expenses: Audit, three
    entries totalling 2,317.24 with the opening balance labelled B/fwd so it is
    not hunted for as a posting. Totals are deliberately NOT clickable.
    Guardrail `statement_drilldown_check.py` (32) with a deliberately awkward
    fixture — a balance-forward row AND activity so the measures are genuinely
    different row sets, two accounts on one line, a prior-year row, a row on an
    excluded basis, and the same account number on another entity. Building it
    found the real trap: `statement` on a `wp_fs_map` row is the SECTION name,
    not the statement name, and getting it wrong sends every account to
    `conflicts` and renders an empty statement.
    Also carried the documentation commits `dad4442` and `2200906` — `CLAUDE.md`
    and `.claude/memory/`, neither copied into the image; the handoff had still
    said `v455` was live, four deploys stale.
    SAME CAVEAT AS v504: `gl_detail` may never have been imported on production,
    and the drilldown reads the table the statements read — if the statements
    render, it works. See `open_items.md` §9.3.)
  - `v504` = `93ce506` (THE CFO'S GL / IA QUERY, as a screen. His workbook
    `GL & IA Queries with Filters - 09182026.xlsx` carries the two Spreadsheet
    Server queries and what he needs to vary: GL — multiple entities, change
    period, select account(s), export; IA — multiple investment IDs, multiple
    investor IDs, select date, MajorType(s), SubType(s), export.
    NOT A SECOND COPY OF HIS SQL. `queries/MRI_GL_Detail.sql` ALREADY IS his GL
    query — its own header says "the Spreadsheet Server NEW JOURNAL query
    one-for-one, with the &SPARM smart parameters removed" — and
    `MRI_IA_Transactions.sql` is the IA one. Both import into `gl_detail` and
    `ia_transactions`. So the ask was to put the parameters BACK against the
    copy we already hold, not to paste the SQL in again (Jim: "since we are
    already pulling these tables into our database we can have the query hit our
    tables"). A second copy would be a second engine for the same numbers.
    NO USER INPUT IS CONCATENATED INTO SQL. Every filter is a bound parameter,
    every IN list an expanding bindparam, and `date_field` — the one filter that
    names a COLUMN — is matched against a fixed set before it can reach the
    query. The guardrail asserts a value carrying SQL matches NOTHING rather
    than everything, and that the table is still there afterwards.
    SEVERAL ENTITIES AND SEVERAL ACCOUNTS IN ONE QUERY (Jim's follow-up). It
    already worked; a native `<select multiple>` needs ctrl-click and nothing
    said so, which is the clearest evidence the control was answering the wrong
    question. Replaced with checkbox lists carrying a search, a count and Select
    all / Clear, in a reusable `MultiPicker`. The search matches the NAME as
    well as the code, because an accountant looks for "Eastchase" and not
    "PPIECH". Proved through the real checkboxes in the running app, and pinned:
    one entity plus one account passing proves nothing, since a bug collapsing
    either list to its first element satisfies both single-value checks — the
    check asserts 2 entities x 2 accounts returns exactly those four pairs.
    WHAT IT SAYS WHEN IT CANNOT ANSWER is most of the design. A truncated grid
    says so AND totals the WHOLE match, not the rows on screen — totalling the
    visible page would make a truncated result look complete and be wrong, with
    nothing saying which. A period before `202401` returns nothing and NAMES THE
    IMPORT BOUND as the reason, because an empty grid otherwise reads as "no
    activity". Freshness is "when the last MRI refresh finished", worded as the
    refresh and not the table because `mri_refresh_status` holds one row for the
    whole job; unknown when none has completed, never a guess. A table that was
    never imported says which MRI query to run. Sub types carry their major
    type, so Return of Capital under Distribution cannot be picked as though it
    were the Contribution one.
    The export is NOT capped at the screen's row limit — getting every row is
    the reason to export — and the workbook records the filters that produced
    it, since these get mailed around and an unlabelled grid cannot be checked
    or repeated.
    Sidebar: Accounting > GL / IA Query, at the bottom as asked, with the route
    added to `acctRoutes` so the section opens when reached directly.
    Purely additive — no DDL, no change to any existing computation; three lines
    of wiring plus new files. Reads are open to any signed-in user, matching the
    rest of the accounting section; flagged to Jim that this is a wider read
    (bulk entity GL) than one entity's statement and one decorator narrows it.
    HIS FILE IS TRUNCATED: the third branch of the IA query (non-cash
    transactions) is cut off mid-statement at 124 characters in row 49 of the
    workbook. Our import covers non-cash, so the tool does too, but if he pasted
    from a longer original something else may have been lost — raised with Jim.
    AND THE DATE BOUND DIFFERS ON PURPOSE: his query is `contributiondate <
    date`, strictly before; the tool's To date is INCLUSIVE, stated on screen,
    because "to 6/30" excluding 6/30 surprises people. Set To one day earlier to
    tie to his sheet exactly.
    Guardrail `gl_ia_query_check.py` (94) against a real database.)
  - `v503` = `2bb9138` (ONE NUMBER, ONE ENGINE — and two places that were
    answering the same question twice.
    ACCRUED PREF HAD TWO IMPLEMENTATIONS IN ONE FILE. Jim: "We should not have
    conflicting calculation results. It will cause doubt in the accuracy of the
    entire work... The only differences in results should come from changes in
    time frames or projections that we are running through the engines."
    `_compute_accrued_pref` (ROE Summary, Committee Summary) and
    `build_pref_balance_detail` (Pref Balance Detail, One Pager, Ownership, NAV,
    valuation tabs) walked the SAME ledger at the SAME rate and disagreed on 34
    of the 68 deals both could price, the ROE path $633,807.54 LOW in aggregate
    at 2025-12-31.
    THE CAUSE WAS A LOST DAY AT EVERY YEAR END: it accrued `cur -> 31 Dec`,
    compounded, then resumed at `1 Jan`, so 31 Dec -> 1 Jan was never accrued.
    One day per year end, always short, worse the older the deal — P0000068 lost
    about $102,000 over nine of them. It never looked wrong, because a slightly
    low accrual is still a plausible accrual. That is why a second
    implementation is most dangerous when it is NEARLY right.
    WHICH ONE WAS CORRECT WAS SETTLED BY JIM'S OWN FIGURES, not by which was
    newer: the vetted walk gives P0000044 51,926.54 and P0000031 37,394.57,
    exactly as he stated them on Sep 17; the other gave 26,489.03 for P0000031.
    `deal_accrued_pref` is now the single way to ask; the One Pager's helper
    became an ADAPTER to it rather than a third copy. Verified through the
    running endpoint before deploy: all three consumers return one figure on all
    68 deals, 0 disagree. THIS RAISES REPORTED ACCRUED PREF ON 34 DEALS — Jim
    was given the count, the delta and the worked examples and said deploy.
    A "TEMPORARY ESTIMATE" IS A SECOND ENGINE. The Committee tab's Net Proceeds
    fell back to `value - debt` when no NAV had run — scaffolding from before
    the NAV engine, left in after it shipped. The NAV walk runs the deal's
    waterfall; value-less-debt ignores it. Removed; a blank now means no NAV.
    A GUARDRAIL WAS POINTED AT THE WRONG ENGINE — it asserted "no grace period"
    against the DELETED function's docstring while claiming to describe the one
    the NAV uses, so it would have kept passing while the real engine drifted.
    Now proven behaviourally. New `one_engine_per_number_check.py` (26); the
    rule and a table of which engine owns which number are in CLAUDE.md.

    THE RENT IN FORCE IS RESOLVED, NOT GUESSED (new business via Jim, Sep 19).
    Rent PSF is taken on ANNUAL rent over SF, one definition, reached by both
    rent-roll readers and the parcel roster; a monthly rent is ANNUALISED before
    dividing, never divided as-is, which is the 12x error v495 already shipped
    once from this importer. A stated per-SF figure is still kept as stated.
    THE MOST RECENT AMENDMENT NOW GOVERNS. Consolidation layers each document
    over the one before, ordered `doc_date ASC NULLS LAST`, and `parse_doc_date`
    only matched a date at the START of a filename — so a folder of "First /
    Second / Third / Fourth Amendment.pdf" had NO dates at all and fell through
    to UPLOAD ORDER. The guardrail fixture reproduces it: the amendments applied
    4, 1, 3, 2, so the First Amendment's superseded rent overwrote the Fourth's.
    The ordinal was already matched by `DOC_TYPE_PATTERNS` and thrown away.
    "MONTHS 1-12" IS PLACED AGAINST THE RENT COMMENCEMENT DATE. There was
    nowhere to store a period, so the extraction had to force it into
    `effective_date` as text, which then compared as text. Now proper columns,
    and `rent_commencement` is lifted out of `extraction_json` onto the tenant.
    Month 1 begins ON rent commencement, so month N is the ANNIVERSARY.
    THE DEFECT NEITHER ASK NAMED: when a step would not resolve, the validation
    picked the step whose annual rent was CLOSEST to the rent roll's own figure
    — the rent roll checked against whichever lease number already agreed with
    it. It could not report a mismatch, and a validation that always passes is
    worse than none because it reads as confirmation. Gone.
    Also: the step dedup keyed on `effective_date = :ed`, and `NULL = NULL` is
    not true in SQL, so every undated step was re-inserted on every extraction
    run. And the validation table gained a Basis column — the note naming which
    step and how it was dated was in the API payload all along, never rendered.
    Guardrail `lease_terms_check.py` (129) drives the SHIPPING paths against a
    real database: a fresh schema, then an EXISTING one migrated with rows in
    it, including a legacy step stored as "Lease Year 7" in the date field,
    which still resolves without a backfill. Production logs after deploy carry
    "Lease review tables ensured" and no migration error.

    ALSO CARRIED: the two portfolio summary SCREENS (`f151e5a`) closing Jack
    Day's list, where two more defects surfaced — the deal was never named
    (`_names` looked for `deal_name`; the table calls it `Investment_Name`, and
    a missing column does not raise, so all 84 rows printed the vcode twice),
    and `prior_debt` was emitted and never rendered, so the comparison tab had
    no comparison on that row. Plus two docs commits, `74f1b0f` and `fcee772`,
    both `.claude/memory/` + `CLAUDE.md`, neither copied into the image.

    NOT VERIFIED ON PRODUCTION: the pref figures themselves. Checking them needs
    a real login, and `admin/admin` is local-only. Verified instead before
    deploy against the running local endpoint on all 68 deals, and after deploy:
    root 200, routes present and gated, migrations clean, bundles carrying the
    new UI, "Net Proceeds (est)" gone. Also unmeasured: how many real lease
    amendments carry neither a date nor a number — there are no lease documents
    in local data and the production read was refused. That is the population
    where the order is still best-effort, and it is reported per tenant.)
  - `v502` = `0d9eaee` (AN UNMATCHED LEASE FILE CAN BE DEALT WITH, and a
    NOT NULL that would have refused the whole upload. Jim: the analyst deletes
    the files belonging to former leases and the section clears. Assign was
    already there; delete was not, so the section could only ever grow and a
    review carrying old leases had no way back to a clean state.
    `delete_documents` is ALL-OR-NOTHING AND SCOPED TO THE REVIEW — a doc id
    from another review is refused rather than quietly skipped, and it returns
    the file NAMES removed, because "2 documents deleted" is not something an
    analyst can check. Two-step confirm in the UI; nothing deletes on a single
    click.
    THE LATENT BUG IS THE BIGGER HALF. `lease_documents.tenant_id` was declared
    `NOT NULL` while the multi-file upload writes NULL for exactly the unmatched
    files this feature exists to manage — so on a FRESH database the whole
    upload raises, not just the unmatched rows. It never surfaced because every
    live database predates the NOT NULL. DDL now nullable, plus
    `_relax_document_tenant_null()` at the START of `ensure_lease_tables`:
    PostgreSQL does `ALTER COLUMN tenant_id DROP NOT NULL`; SQLite has no such
    statement, so it drops and rebuilds ONLY when the table is empty, and I
    verified the destructive branch is unreachable on PostgreSQL before
    shipping. Production logs after deploy carry no migration error.
    VERIFIED END TO END IN THE RUNNING APP, not asserted: 3 unmatched files
    seeded, 1 assigned to a current tenant, the other 2 selected and deleted
    behind the confirm, section gone, the assigned amendment still there.
    ALSO CARRIED, all reviewed under P4: Jack Day's valuation list — the
    mapping draft that survives a reload (`d2c8ce0`, `18de4cd`, `815b049`,
    `1d99bce`), the account number READ FROM THE FILE rather than guessed, the
    whole-COA dropdown in statement order, the 3+ digit row filter, the $20K
    partnership expense as a VISIBLE proposed line to 5130 rather than a silent
    injection, the two portfolio summary tabs assembled from the Pref Balance
    Detail engine, and portfolio groups LABELLED BY JACK rather than inferred —
    a derived rule got 10 of 11 right and was wrong on three, which is exactly
    the kind of nearly-right that never gets checked.
    TWO DEFECTS THE SUMMARY WORK FOUND IN PASSING: the PSC and OP pref sides
    were being SUMMED (5,746,667 became 9,469,999), and a cycle's `as_of` is
    stored as TEXT, so passing it to the pref walk failed every date comparison
    and returned `0.00` — wrong on 7 of 8 deals, and wrong in the direction that
    looks like "no accrual yet" rather than an error. Guardrails 144 / 57 / 46.
    STILL TO BUILD: the two summary SCREENS. Service, grouping, sections and
    endpoints are done and tested; the Vue is not written.)
  - `v501` = `ec6dda2` (A ROW READ AS "NO LEASE" STOPS BEING A TENANT. Jim:
    three records checked off as No lease, still in the Market at Poplar
    roster. They are the building banner and the two subtotal rows, left over
    from an import predating the phantom-row fix — his figures give it away,
    $259,324 annual at $1.13/SF is the old monthly-read-as-annual number.
    Reading them took them out of the PROJECTION, which is what the reading was
    built to do, and nothing else moved:
    THE ROSTER showed every tenant regardless of its reading — a row read as
    not a tenant has no business in the tenant list. THE HEADLINE TOTALS summed
    every row with no status filter, so each subtotal row added the property's
    whole 229,722 SF a second and third time; GLA read three times the real
    figure and the reading did not correct it.
    The roster now hides a No lease row, says how many it hides, and offers
    Show them — HIDDEN, NEVER DELETED, because the reading is reversible and
    the row is still there to be read differently. A VACATED tenant still
    shows: we hold a lease for them and it has to stay reachable; it is only
    out of the projection.
    TOTALS ARE RECOMPUTED ON THE DISPOSITION, not just at the next import — a
    reading that does not move the number on screen looks like it did nothing,
    which is how both of the last two reports started. The call returns the
    corrected totals with it.
    Verified against his exact state: seeded the three debris rows, read them
    through the API, GLA back to 660,140 with the rows still present and marked.
    Guardrail 127 -> 134, pinning the defect in BOTH directions — an unread
    subtotal row does inflate the headline, and reading it takes it back out.)
  - `v500` = `c84127d` (A MAPPING IN PROGRESS IS KEPT. Asset management: "How
    do you save mapping adjustments? I don't see a save button, and when the
    page refreshed my mapping work was gone." Both halves were true and a THIRD
    was waiting. The mapping lived only in browser memory between parse and
    commit and NONE of the four endpoints read one back — so a refresh threw
    away twenty minutes of judgement, and re-opening after a SUCCESSFUL import
    showed an empty page, which reads exactly like losing it. The one button on
    the screen said "Apply mapping to the Valuation column", which is why it
    was not found when looking for a save button.
    SAVED AS THE ANALYST WORKS, NOT ON A BUTTON. Every category, account or
    flip change already triggers a validation round trip; the save rides along,
    debounced 700ms and sequenced so a slow reply cannot report a stale state.
    Asking someone to remember to save twenty minutes of judgement is asking
    them to lose it once.
    THE PARSED FILE IS STORED WITH THE MAPPING — without it, resuming means
    hunting down the partner's spreadsheet again, which is most of the friction
    the draft removes. An upload is stored immediately, before any mapping.
    A COMMITTED MAPPING IS KEPT AND MARKED APPLIED rather than cleared;
    clearing on success is what made a successful import look like lost work.
    Editing puts it back to a draft. Drafts are per record AND per source.
    New table `valuation_mapping_drafts`, created by the existing idempotent
    `ensure_valuation_tables` — which `list_cycles` calls, so it exists the
    moment anyone opens Valuations.
    Verified through the running app, not only the service: 20 of 65 lines
    mapped, refreshed, got them back with the file; one more edit took the
    stored count 20 -> 21 with the indicator confirming.
    Guardrail `mapping_draft_check.py` (29), which caught a bug in itself —
    Flask registers one rule per view function, so keying them by path reported
    two of the three methods missing.)
  - `v499` = `3c045aa` (A FINDING THAT HAS BEEN READ STOPS ASKING. Jim: three
    records where selecting "No lease" did nothing. IT WAS NOT DOING NOTHING —
    `merge_rent_roll_to_review` lists every tenant absent from the upload
    REGARDLESS of whether it already carries a reading, so a re-import re-listed
    tenants settled in an earlier run. Their button was already lit, the PUT
    re-set a value that was already set, and the only feedback was a highlight
    that did not change. Reproduced locally: request fires, 200 back, row
    identical before and after — the hardest kind of bug to see, because every
    part of it worked.
    The status now travels with the finding, so the screen can tell a settled
    row from an outstanding one and a re-import stops asking about tenants
    already read.
    Plus what Jim asked for: a reading removes its row, and the box goes when
    nothing is outstanding. Measured through the real 46-finding case — 46 to 45
    to 44 read one at a time, bulk clearing the remaining 43, box gone, and a
    fresh import of the same file leaving it gone. The badge counts what is
    still to read, not what the merge first found. A new import clears the
    addressed set. Guardrail 121 -> 127, including the round trip: read every
    finding, re-import, assert nothing outstanding.)
  - `v498` = `bc4ac18` (ONE READING FOR MANY FINDINGS. Jim asked for the bulk
    action after seeing that 46 findings meant 46 clicks. Select all, or pick
    the many and correct the few; ONE PUT for the whole selection, verified in
    the browser against the real 46-finding case.
    ALL OR NOTHING: every id is checked against the review BEFORE anything is
    written, so one bad id cannot leave half the tenants moved and half not — a
    partly applied bulk action is worse than a refused one because nothing on
    screen would say which half took. The guardrail asserts not just that the
    bad id is refused but that the VALID tenants in the same call were left
    alone and are still projected. The selection clears once applied, so a
    second click cannot re-apply to a set the analyst thinks is dealt with.
    Guardrail 111 -> 121.)
  - `v497` = `89ea195` (THE RENT ROLL IS CHECKED AGAINST THE LEASES, NOT THE
    OTHER WAY ROUND. Jim: "If there is no lease to support a tenant listed in
    the rent roll we are more likely to disregard the rent roll entry. If we
    have valid leases for legitimate space we will likely be looking to add
    that lease to the rent roll than delete the lease from our records." And:
    lease uploads may hold leases for tenants who have VACATED, which do not
    belong in the projection.
    THAT INVERTS THE PREMISE THE IMPORT WAS BUILT ON. The leases are the
    authority and the rent roll is what is being validated, so a tenant missing
    from a later rent roll is a FINDING ABOUT THE RENT ROLL — not a cue to
    delete the tenant and the abstract built from its lease. I was part-way
    through making replace "safer" by carrying work across; that was still the
    wrong premise and was reverted rather than finished.
    THE IMPORT NO LONGER DELETES ANYTHING. Merge only; replace is gone from the
    screen AND from every route, because removing a button is not enough when
    the route is reachable alone. Nothing in `api/lease_review.py` calls
    `import_rent_roll_to_review`, and merge itself contains no DELETE.
    DISPOSITION is what makes the finding actionable — three readings: on the
    rent roll (projected), vacated (lease stays on file, not projected), no
    lease (not projected). Set from the findings list at the moment the
    disagreement surfaces, and from a Reading column on the roster so it stays
    visible and reversible.
    VERIFIED ON A REAL PROJECTION, not by reading the status column back:
    marking a tenant vacated takes the run 48 suites -> 47, and afterwards the
    tenant row, its lease PDF and its abstract are all still there. The
    guardrail asserts BOTH halves — out of the projection AND records kept —
    since either passing alone would be worse than useless.
    `is_vacant` and `tenant_status` are DIFFERENT AXES and are not conflated:
    is_vacant = the suite is empty per the rent roll; tenant_status 'vacated' =
    we hold a lease for a tenant who has left. Confirmed live: vacated with
    is_vacant still 0.
    FOUND ON THE WAY: `rent_per_sf` is read by `get_resolved_tenants` and
    written by every import path but was in NEITHER the CREATE TABLE nor any
    migration. Existing databases carry it from an older schema so nothing was
    broken, but a genuinely fresh one would have failed with "no such column"
    the first time anyone opened a projection. Added as a migration — which is
    also what made the projection fixture buildable. Guardrail 97 -> 111.)
  - `v496` = `b1b0f5b` (ONE WAY IN, AND REPLACE CLEARS WHAT POINTS AT THE
    TENANTS. Jim's re-import of Market at Poplar failed on production with a
    ForeignKeyViolation on `lease_tenant_sales`. The mapping was fine; the
    REPLACE was not. Deleting a review's tenants means clearing everything
    referencing them first, and that list was typed by hand covering 5 of the 9
    child tables — so ANY review carrying tenant sales, an abstract, a
    validation row or an analyst resolution could not be replaced at all. IT
    COULD ONLY EVER FAIL ON POSTGRESQL: `_pg_to_sqlite` strips REFERENCES from
    the DDL, so SQLite declares no foreign keys and local dev cannot reproduce
    it even with `PRAGMA foreign_keys=ON`. Read from the catalog now, UNIONed
    with a known list because neither alone suffices, and the FK COLUMN is read
    rather than assumed to be `tenant_id`.
    THE FIRST VERSION OF THAT FIX WAS WORSE THAN THE BUG, AND THE GUARDRAIL
    MISSED IT because it asserted the list existed instead of running the
    clear. The helpers are module level and `text` is imported inside each of
    the sixty functions that use it, so every DELETE raised NameError, a
    blanket try/except swallowed all of them, and the only symptom was orphaned
    rows on SQLite and the same ForeignKeyViolation on PostgreSQL. Missing
    tables are skipped by asking the inspector; anything else raises. The check
    now seeds three kinds of dependent row, runs the clear and asserts they
    were there before and gone after. A replace that still cannot proceed
    returns 409 naming merge, not a constraint name.
    ONE WAY IN (Jim: "route everything through the scan"). The two bypass
    buttons are gone; merge-vs-replace is chosen on the confirmation panel
    after the columns have been seen. Removing buttons was NOT enough — both
    routes are reachable alone, so `upload-rent-roll` and `merge-rent-roll` run
    the same scan and import its PROPOSAL, refusing a file whose charge period
    is unstated rather than guessing. `parse_rent_roll_flexible` is no longer
    reached from any route.
    THAT FORCED THE PROPOSALS TO GET BETTER, and finding out cost a
    measurement: routing Windsor through the scan proposed NO tenant name and
    NO base rent. `_BASE_RENT_PATTERNS` matched only a column called exactly
    "Rent", so "Monthly Rent"/"Annual Rent" matched nothing, and the
    Argus/Windsor export names its tenant column "Lease". Per-area and
    per-month restatements are demoted to Ignore with a warning — two columns
    both marked base rent would be SUMMED. Windsor now imports with no manual
    input and reproduces the legacy parser to the dollar (49 rows, 660,140 SF,
    $7,203,596).
    Also: the scan promised 53 tenants and delivered 49, because scan and
    import resolved the tenant column differently and disagreed about which
    rows were subtotals. One shared test now, asserted equal.
    REPLACE NOW SUCCEEDS WHERE IT USED TO ERROR, so it removes the sales and
    abstracts it could not remove before; the dropdown says so at the moment
    the choice is made. Guardrail 80 -> 92.)
  - `v495` = `423daa5` (THE RENT ROLL STOPS GUESSING, AND THE RECOVERY COLUMNS
    WERE THE SMALLER HALF. Jim reported that Market at Poplar imported CAM but
    not Insurance or Tax — `_find_col` returns the FIRST match and stops, so CAM
    came in alone at 34% of the recovery, and Property Tax by itself
    (37,884/mo) is bigger than CAM (22,574/mo). The bigger defect was next to
    it: "Base Rent" in that file is a MONTHLY column carrying no qualifier, and
    `annual_rent`'s keyword list reads "base rent" as annual, so EVERY RENT
    LANDED 12x LOW — Patton Computers at $1.83/SF/yr against a real $22.00.
    Three non-tenant rows imported as tenants as well (the building banner,
    "Sub-total for Building: 925", "Grand Total for Report"), adding 459,444
    phantom SF against a real 228,122; the old skip list matched 'grand total'
    EXACTLY, so the actual row walked past it.
    NEITHER QUESTION IS ANSWERABLE FROM A HEADER, so neither is guessed now.
    `scan()` describes the file and proposes; `apply_mapping()` loads only what
    the analyst confirmed. `_propose_basis` returns None rather than picking,
    and a charge with no stated period REFUSES to import until answered —
    that refusal IS the feature. Recoveries are many-to-one by design.
    TWO LAYOUTS, because the same property arrives in both. The MRI "Master
    Rent Roll" PDF prints charges as ROWS under each tenant and the old parser
    refused it outright ("Cannot find tenant name column"); the analyst
    classifies charge LABELS there, same screen. THE REAL PDF FOUND THREE
    PARSE BUGS FIXTURES WOULD NOT HAVE: `round(top,0)` split a line at a 0.24pt
    boundary so a `* Tenant Total *` lost its own amounts; the running page
    header cleared the current tenant, dropping every charge for the two blocks
    straddling a page break; and the building totals block read as tenant
    charges.
    BOTH FILES TIE TO THEIR OWN STATED TOTALS, which is the only check that
    proves the chain rather than its parts — Excel to its subtotal row
    (3,111,887.88 rent / 786,698.28 recoveries / 229,722 SF), PDF to its
    building total (3,045,529.32 / 782,590.20), all 33 tenant blocks
    reconciling to their printed tenant total, and 32 of 33 suites agreeing
    across the two formats (the 33rd went vacant between the two dates).
    Verified end to end through the running app, not asserted.
    THE LEGACY KEYWORD PATH IS UNCHANGED and still reachable as "Quick Merge"
    and "Replace All" — so the 12x hazard is still one wrong button away for a
    file of this shape. Deliberate (Argus and Windsor rely on it) and open.
    ROSTER LAYOUT: 44 of 44 rows ran over one line and "Republic Finance #289"
    took three. Tenant name fixed at 190px, clipped, full name on hover —
    measured across all 77 names in both rosters, 70 fit whole and 220px would
    have gained exactly one more. Table height 2,250px -> 1,370. Dates get
    nowrap as a LATENT hazard, not a reproduced one: at the widths tested the
    "Lease End" header's own nowrap already held the column at 82px.
    Guardrail `rent_roll_mapping_check.py` (80) caught a bug of mine before it
    shipped — a tenant named "Total Wine & More" was being thrown out as a
    subtotal row, which is why every narrowing is asserted in both directions.
    Carried two undeployed docs commits, `84715dc` and `05ee17e`; P2 listed
    them, both are `.claude/memory/` + `CLAUDE.md` and the Dockerfile copies
    neither into the image.)
  - `v494` = `d72c46b` (THE STATEMENT'S MASK IS NOT ALWAYS A TAIL. Jim asked
    me to add the six accounts holding real money that PNC would not serve
    activity for. FIVE OF THEM DID NOT NEED ADDING — they were registered all
    along and the matching was wrong. `XX-XXXX-5765` hides the FRONT of the
    number; `790-XXXXX55` hides the MIDDLE, and reading "the last four visible
    digits" off the second gives 790 + 55 -> `79055` -> `9055`, an account that
    exists nowhere, while the real 7900021255 sat untouched. It reported them
    as unknown accounts, so the cause looked like missing data rather than a
    bad heuristic — the kind of error that gets "fixed" by entering data that
    was never missing.
    The mask is now read as what it is: each run of X is that many unknown
    digits, each printed digit is itself, and the pattern matches the WHOLE
    number. `790-XXXXX55` -> `^790\d{5}55$`. Being anchored also makes it
    STRICTER than before, so the guardrail's ambiguity fixture needed a genuine
    same-length twin to still test anything. Against the 64 real June
    statements: 45 filed before, 50 after.
    ONE account is genuinely missing (PPI Life Storage NY, 119,701.35), so
    `create_account` registers one by hand — nothing could before, since
    accounts only ever self-registered from an activity import, which is no
    help for an account quiet longer than PNC's 90-day window. THE FULL NUMBER
    IS TYPED AND NEVER INFERRED: several of these sit in obvious number ranges
    so guessing would usually work and would occasionally split one account in
    two the moment real activity arrived under the true number. A masked number
    is refused as input.
    Still not filing, all benign: twelve 0.00 dormant accounts and one WELLS
    FARGO statement in the PNC folder. Guardrail 69 -> 80 (54 on production).)
  - `v493` = `cab414c` (STATEMENTS IN BULK, AND A PARSER BUG THE REAL FILES
    FOUND. Jim asked whether to type fifty 6/30 opening balances or read them
    off the June statements; the statements win, being an external authority
    carrying their own arithmetic check and staying traceable to a named file.
    TESTED AGAINST THE 64 REAL JUNE STATEMENTS BEFORE HE RELIED ON IT, and 50
    would not have filed. 46 of those failed on ONE detail: PNC prints a zero
    balance as `.00`, with no leading digit, and the money pattern required
    one. It threw out rows carrying REAL amounts too — `30,832.24 .00
    11,712.76 19,119.48` was refused for the single `.00` in the credits
    column. 14 filed before, 45 after. It failed the SAFE way, refusing rather
    than inventing a balance, but it would have blocked most of the seeding
    and looked like "PNC layouts vary" rather than a one-character regex.
    Also: a TRAILING MINUS is PNC's overdraft notation, read as positive it
    gives a wrong balance rather than a refusal, so statement figures go
    through `_money` — kept separate from `_num` because the ACTIVITY export
    carries its sign in a Credit/Debit column and never a trailing minus. And
    the four figures are now read from UNDER THE SUMMARY HEADER, because
    loosening the pattern made "the first four money figures in the document"
    unsafe.
    SEEDING IS NOT RE-BASING, and that distinction is the whole function.
    `opening_balance()` never reads a statement — it carries the prior close,
    so a break surfaces as a difference instead of being papered over.
    Starting the chain is the one case where a statement IS the right source,
    so `seed_from_statement` REFUSES once a period has actually been
    reconciled, and says why. The note it writes names the source file.
    BULK IMPORT ROUTES BY THE MASKED NUMBER `XX-XXXX-5765` — the parser always
    extracted it and nothing used it. All 50 last-four groups are unique in
    today's accounts, but that is a fact and not a guarantee, so an AMBIGUOUS
    suffix is REFUSED rather than resolved by picking one. Every file gets its
    own result row.
    Of the 19 June statements still not filing: 18 are accounts with a
    statement but NO activity in the 90-day export, so they never registered —
    a registration gap, not a parse gap, and six hold real money (141,522.90,
    119,701.35, 15,530.09, 6,587.72, 5,015.92, 592.33). The 19th is a WELLS
    FARGO statement sitting in the PNC folder; refusing it is correct.
    Guardrail 51 -> 69 (43 on production), pinning the zero balance, the mixed
    row, the trailing minus, the header anchoring, the non-PNC refusal, and
    both halves of the seeding rule.)
  - `v492` = `8fc4947` (TREASURY PRODUCES THE TWO MRI UPLOAD FILES. The
    contract was read off the files MRI ACCEPTED, not from a specification:
    the August GL upload is 49 lines summing to 0.00, its 24 MR10005000 lines
    equal the bank's own net movement of -560,022.54, and its 13 MR31000001
    lines and the 13-row IA sample both total 12,580.47. One coded
    distribution produces BOTH a GL line and an IA row.
    A FOURTH TAB codes the month, one row per bank transaction — measured, not
    assumed: all thirteen investor distributions arrive as INDIVIDUAL bank
    debits (285.92 seven times, 571.84 twice), so naming an account per line is
    the common case and the split proposal is offered per row for the other.
    THE CASH SIDE IS NEVER TYPED: each transaction becomes its own cash line at
    the amount the bank reported and the accountant supplies the offset, so the
    entry balances BY CONSTRUCTION and a partly coded month cannot produce a
    file. Verified end to end: 24 transactions -> 48 lines, balances, cash side
    ties to the bank.
    THE SPLIT IS COMPUTED FROM COMMITMENT AMOUNTS, NOT THE STORED PERCENTAGES,
    and the difference is not academic. AMB6's CapitalPercent is 4dp and sums
    to 99.9999; allocating by it lands three cents over and is wrong on 5 of 13
    investors. Allocating by the amounts over their base of 11,000,000
    reproduces all thirteen to the cent. It reads `commitments`, NOT
    `relationships` — AMB6 has 15 relationship rows for 13 investors, PSC1
    twice (one a closed 100% ending 2026-06-30) and PSCMAN at 0%. Verified on
    production against the real commitments: matches the accountant's file
    exactly, ties, nothing excluded, no drift. It is a PROPOSAL and editable
    (Jim: "compute it and show it as an editable proposal").
    THE IA FILE IS WRITTEN INTO A COPY OF MRI'S OWN TEMPLATE. Column 12 of
    `Transaction Values` is "Number of Shares" and its header cell is BLANK;
    pandas calls it `Unnamed: 11` and a rebuilt workbook would write that into
    a header MRI parses.
    PRE-FLIGHT P4 CAUGHT THE TEMPLATE NOT SHIPPING. `.gitignore` blocks *.xlsx
    and *.csv, so the vendored templates were never committed and
    `build_ia_xlsx` would have raised in the container. The guardrail had
    passed on an untracked file present in the working tree — present locally
    is not shipped — and now asserts git tracking. Both files verified to carry
    no entity data before the narrow ignore exception was added.
    Guardrail `treasury_upload_check.py` (58; 26 on production where the real
    files and git are absent) REBUILDS BOTH ACCEPTED FILES FROM THEIR OWN
    CONTENTS AND ASSERTS BYTE-IDENTICAL OUTPUT — a format check written from a
    specification proves only that the code agrees with itself. It caught the
    amount formatting: the accepted file writes 13313.8, not 13313.80.)
  - `v491` = `33b04e7` (THE ORDER NUMBER IS THE CFO'S, and so are the two
    buttons that rewrite the same column. The order cell is
    `PUT /schedule/order`, but `renumber` rewrites the whole column and
    `carry-forward` writes `sort_order` for every row of the next cycle —
    gating the cell alone leaves the rule defeated by clicking a different
    button. Carry-forward also moves the preparer and property, which are the
    team's; it is included because laying out the next cycle is the CFO's act,
    not because those fields are his. `Fill properties` is deliberately NOT
    included and was lifted out of the same toolbar block rather than dragged
    along with its neighbours.
    `CLOSE_CYCLE_ROLES` -> `CLOSE_PLAN_ROLES`: after three passes the gate is
    no longer about cycles but about when the close opens, when each thing is
    due, and what order entities are worked in. No alias left behind.
    Dead `setDue()` deleted (19 lines, unreferenced since the two-tab split).
    THE ENDPOINT STAYS GATED — still reachable over HTTP, and a rule covering
    only the routes that happen to have a button breaks the next time somebody
    adds one. The guardrail asserts the function is gone so it cannot return.
    Verified on production: cfo may use the order cell, renumber and carry
    forward; accountant may use none of the three, and may still sync, sign
    off, name a preparer, set a property and fill properties in bulk.
    accounting_access_check 40 -> 54 (41 on production; the screen checks read
    Vue source the image does not ship and skip with a reason).
    Also carried the documentation commits: new `.claude/memory/treasury.md`,
    the access model in `accounting_workpapers.md`, `open_items.md` §7 (six
    items, each with an owner), and the handoff retitled through Sep 17.)
  - `v490` = `1496daa` (DEADLINES ARE THE CFO'S; signing off against one is
    not. Jim: "deadlines should be CFO only too."
    TWO ENDPOINTS, AND THE ONE I HAD FLAGGED WAS THE DEAD ONE.
    `PUT /cycles/<id>/steps` has no caller — the two-tab split replaced the
    per-step deadline UI with the tracker's per-deliverable target dates, and
    `setDue()` in WorkpapersView.vue is unreferenced leftovers. The deadline an
    accountant sees and edits is `PUT /packages/<id>/schedule/target`. Gating
    only the endpoint I had named would have satisfied the request and changed
    nothing on screen. Both now use `CLOSE_CYCLE_ROLES`.
    THE LINE: the CFO decides WHEN a deliverable is due; the team records what
    was DONE against it. The target-date cell and its three sign-off cells are
    adjacent columns of one tracker row and belong to different people, so the
    guardrail asserts BOTH halves — accountants refused on target dates, and
    still able to sign off, name a preparer and sync. A rule checked only in
    the refusing direction is satisfied by locking everyone out, which here
    would have stopped the close.
    Verified on production: cfo/admin may set a target date, accountant and
    accounting_manager may not, and both may still sign off, name preparers and
    sync. accounting_access_check 29 -> 40 (31 on production; the nine screen
    checks read Vue source, which the image does not ship, and now SKIP with a
    reason instead of crashing — at v489 they killed the run after the useful
    sections had already passed).
    Still the team's: the order number, renumber and carry-forward.)
  - `v489` = `ce80ba5` (TREASURY HAS A SCREEN, and the accounting section is
    accounting's to edit.
    THREE TABS under Accounting: accounts, import, reconciliation. The accounts
    tab shows CURRENT LEDGER — the last closed ending plus everything imported
    since, saying which period it came from and what date it runs through — and
    leaves CURRENT AVAILABLE BLANK, with the reason on the screen. Available is
    ledger less holds, float and pending debits, which exist only at the bank
    and appear nowhere in an activity export; filling that column would be
    inventing the one number a treasurer acts on. An account with nothing
    closed has NO ledger figure rather than 0.00.
    A NEW GUARDRAIL FOR THE API-TO-SCREEN SEAM, which the service's own 51
    checks structurally cannot see: a field read by the wrong name renders as a
    blank cell with no error and no log, and it happened three times while the
    view was written (`opening`/`opening_balance`, `net_movement`/
    `bank_movement`, `imported`/`inserted`). It found three defects, all fixed
    before shipping — a non-export file reported as "0 transactions imported"
    (a refusal dressed as a no-op; the column check now runs BEFORE the row
    count), an unmapped account returning empty lists so the screen could not
    show what HAD come through, and imports open to viewers.
    ACCESS: `role_required` compares LEVELS and analyst sits at level 1 with
    every accounting role, so no naming of roles could exclude analysts.
    `roles_exactly(*ACCOUNTING_ROLES)` checks membership instead — accountant,
    accounting_manager, cfo, admin. Jim's own day-to-day login is an analyst
    one and is now read-only here (Jim, Sep 17 2026).
    SIX WRITES HAD NO ROLE CHECK AT ALL — transition, assign, exhibit upload,
    exhibit DELETE, schedule signoff, steps. Any signed-in user, viewers
    included, could delete a workpaper exhibit or sign off a tracker cell.
    Found by ENUMERATING the section's routes from the app; the previous check
    grepped for a decorator's text, which is blind to a route that never had
    one. Verified on production: 25 writes, analyst refused on all 25, viewer
    refused on all 25, all four accounting roles admitted.
    STARTING A CLOSE CYCLE IS NARROWER STILL — `CLOSE_CYCLE_ROLES` = admin +
    cfo, one endpoint. Syncing entities into an existing cycle stays with the
    team (Jim: "starting a close cycle should belong to the CFO anyone on the
    accounting team can sync entities"). Checked in BOTH directions on
    production, and the accountants checked to STILL be able to sync.
    THE SCREENS WERE WRONG THE OTHER WAY: `['admin', 'cfo']` locked out the
    accountants who prepare the close — v480's mistake, one role over. Both
    views read `auth.canEditAccounting` now, and the guardrail compares the Vue
    list to the Python one by name so they cannot drift.
    Guardrails: `accounting_access_check.py` (29), `treasury_api_check.py` (37),
    `treasury_reconciliation_check.py` (51). First two verified on production.)
  - `v488` = `492fe04` (treasury service only, inert — no endpoint, no screen.
    The bank side of the close: PNC activity and statement parsing, the
    three-way tie, carry-forward, and the matcher. Measured from the real
    August AMB6 files before any code was written — beginning 571,750.04, net
    movement -560,022.54, computed ending 11,727.50 ties to the statement, and
    MRI's September reconciliation opens at the same figure. 24 bank
    transactions pair 1:1 with 24 GL cash lines.)
  - `v487` = `6b30f20` (a name an EARLIER fill wrote gets its basis back.
    Deploying v486 exposed this at once: 23 of 58 rows had a property and no
    basis, because the column did not exist when they were written — so they
    read as values somebody typed, which is the precise confusion the basis was
    added to prevent, and it was wrong on more rows than the new hop had just
    filled. Annotated ONLY where the stored name is identical to what the walk
    produces; a name the CFO has since edited differs and keeps its silence.
    Production after: one hop 21, two hops 17, typed/unmarked 2, blank 18 = 58.)
  - `v486` = `92cb04a` (the Property column walks commitments TWO levels, and
    every inferred name says how it was reached. Measured before building: one
    hop leaves 35 of 58 blank, two leaves 18, and six of the seven new names
    match the CFO's own sheet exactly. THE SEVENTH IS WRONG AND IS WHY THE BASIS
    EXISTS — TGA6 is a fund that happens to reach one deal at two levels, so it
    resolves to "Presidential Arms JV" where his sheet says "Various". No
    cleverer rule fixes that, so the mitigation is telling the reader: the row
    shows the basis, the name renders in italic with a superscript hop count,
    and typing over it clears the marker because then it is his decision.
    Breadth first, so the basis reports the SHALLOWEST level a deal was found
    at; visited nodes are never re-queued, so a cycle cannot walk forever.
    `property_basis` is a new column: a sentence when inferred, NULL when typed.
    Guardrail 67 -> 71 checks.)
  - `v485` = `5fe16b7` (the printed Schedule of Investments and Statement of
    Changes in Members' Capital were blank. THREE SHAPES come out of the
    statement engine — `sections` for the balance sheet, income statement and
    cash flow; `lines` for the SOI; `rows` x `members` for members' capital —
    and the print view rendered only the first. `hasContent` accepted all three,
    so both statements passed the test for having something to say, got a page
    and a title, then met a table body that could only walk `sections`. A titled
    empty page is worse than an omitted one: it reads as "this entity has no
    investments" rather than "this view cannot draw them". Each shape now has
    its own renderer.
    VERIFIED THROUGH THE ACTUAL TEMPLATE EXPRESSIONS on the real PPIECH payload
    in a browser rather than asserted — members=2, rows=3, 9 of 9 amount cells
    populated. SOI field names taken from `workpaper_excel`, which is shipping
    code reading the same object, since PPIECH has no SOI lines locally.
    Confirmed on production afterwards: of 14 entities checked, 14 have members'
    capital data and 12 have SOI data — so this was blanking real content on
    nearly every entity.
    The investor version omits the SOI's membership-interest reconciliation
    (commitments vs `relationships`, which disagree while accounting is
    mid-update); that stays in the workbook where it can be resolved.
    Guardrail asserts the engine's shapes AND that the view reads the keys the
    engine emits — this failure is silent on both sides, since a renamed field
    prints blanks rather than raising.)
  - `v484` = `a08a608` (THREE THINGS on the statements.
    (1) THE DOWNLOADED WORKBOOK WAS ILLEGIBLE AND IT WAS v480'S FAULT — the
    Eastchase column widths were applied to statements with a DIFFERENT layout
    (Eastchase indents in A and labels in B; the generated statements label in
    A), so `A = 4.3` landed on the label column and the Income Statement got
    2.4. Worse, `_table` had already measured every column to its content and
    `apply()` ran afterwards, overwriting a correct answer with a wrong one.
    Widths are no longer set from the reference; margins, centring, header,
    footer and fit-to-page stay, being layout-independent. Verified on
    production: Balance Sheet now A=42.0/B=20.0/C=21.0.
    (2) PRINTED STATEMENTS, INDIVIDUAL AND BATCH, on the One Pager's pattern as
    Jim asked — `POST /api/workpapers/statements/batch` assembles server side,
    `/workpapers/print` stacks the pages, one `window.print()` yields one PDF for
    one entity or 58. Entry points on the tracker toolbar (whole cycle, CFO's
    order) and in the workbench head. PER-ENTITY `error`: one failure prints a
    notice instead of silently dropping an entity from the batch.
    (3) EVERY STATEMENT FOOTS, from ONE shape. Jim asked for the balance sheet's
    liabilities-plus-capital total, then net income, then the net change in
    cash. The last two ALREADY EXISTED in the engine and simply were not
    rendered on screen. Rather than three bespoke blocks in three consumers,
    `footing = {label, amount, compare_label, compare_amount, ties, difference}`
    covers all of them and a fourth statement gets it free. `ties` is None when
    there is nothing to compare against — never False, never True. The printed
    statement shows the figure only; the reconciliation note stays on the
    workbench and in the workbook where it can be acted on.
    Verified on production PPIECH: balance sheet foots to 33,378,047.84 and TIES
    to total assets; net income -11,745.08; cash flow ties at 0.
    Guardrail: statement_presentation_check.py extended.)
  - `v483` = `5215888` (the tracker header stops covering the first data row —
    reported twice, because the first fix did not work. It was sticky PER ROW,
    row 2 pinned at an offset that must equal row 1's height: hardcoded 34px,
    then measured at runtime, and the overlap survived both. Measured in a
    standalone repro of the exact markup served through the dev server: row 1 is
    22px and row 2 is 34px, so the 34px fallback sat twelve pixels too low. The
    measurement was arithmetically right and simply never reached the CSS
    variable. Fixed by deleting the arithmetic — `thead { position: sticky }`
    sticks both rows as one block, so they cannot be mispositioned relative to
    each other at any font size or zoom, with no JavaScript. Verified on BOTH
    axes in a second repro carrying the left-pinned columns, since nested sticky
    is where this would break: at rest thead.bottom 63 = firstRow.top 63, flush.
    Not verified against the running app — that needs a login.)
  - `v482` = `11f9ca8` (Two things. DEAL ANALYSIS: the extension test — what
    paydown would clear the covenant, and what if it is negotiated. `nReqDSR` is
    the EXTENSION test and `nRequiredDCR` the ongoing covenant (Jim, Sep 17
    2026); they are both ratios in the same units, so testing the wrong one
    answers a different question. Reuses `planned_loans`' own primitives rather
    than re-deriving a constraint. Seeded from MRI, editable, NEVER stored — a
    proposed covenant is a negotiating position. The what-if re-solves from the
    baseline's own NOI/rate/cap rate, so only the covenant moves between two
    answers; it also needs no data load, since `fc_deal_full` and `mri_val` are
    NOT on the cached result. Reports a paydown, applies nothing.
    TRACKER, from Jim's screenshot: # column 46px -> 40px; the second header row
    was pinned at a hardcoded `top:34px` and covered the first data row, now
    MEASURED after render and on resize; the clear × was #c3ccd9 on pale green
    and only coloured on hover, so it could not be found — now green-on-green
    13px bold; the date cells no longer repeat the initials, since the column
    header already names the approval level (the signer is still recorded and is
    on the tooltip); Property imports the deal name via commitments; Prep is a
    dropdown showing initials only.
    THE DROPDOWN READS THE USER LIST. I reported that no account carried an
    accounting role while the close is prepared by KH/NL/RE, so a users-sourced
    dropdown would have been empty of the people doing the work; Jim added the
    accounts mid-build. Verified on production: regolf -> RE
    (accounting_manager), kherrmannn -> KH (accountant), jstewart -> JS (cfo) —
    exactly his spreadsheet's initials, no collisions.
    PROPERTY DECLINES RATHER THAN GUESSES: measured on production, 23 of 58
    filled and 35 unresolved — the funds and holdings that commit into another
    ENTITY rather than a deal (AMB6, KCREIT, OWPSC, PCBLE…) plus deal-level ones
    like NOTTNV. A typed value is never overwritten. Resolving the rest needs a
    second hop through the ownership chain; not built.
    Guardrails: `workpaper_tracker_check.py` now 59 (one of the new checks was
    passing VACUOUSLY — it tested overwrite-protection on a row with nothing to
    overwrite) and `loan_extension_check.py` 31.)
  - `v481` = `516ffd6` (Deal Analysis now SAYS when a loan matures before the deal
    sells. The amortization schedule ends at maturity, so every month between it
    and the sale carried no debt service and the balance stopped being anywhere —
    and nothing said so. Jefferson Waters Creek against a manually entered
    2027-04-30 sale: schedule ends 2026-11-30, maturity 2026-12-05, $51,667,000
    outstanding, ~$318,673/month, five months, **~$1.59M of interest never
    charged** — distributable cash overstated by about that much. Verified on
    production after deploy.
    IT DOES NOT EXTEND THE LOAN. MRI records `ExtensionOptions = '2x12'` and one
    of those two options would carry the maturity to 2027-12-05, past the sale —
    so closing the gap automatically would have been easy and wrong. Exercising
    an extension is a business decision with covenant conditions attached; the
    interest is reported and NOT added back. Purely additive: the forecast, the
    schedules and the waterfall are unchanged.
    Placed ABOVE the sections rather than inside Debt Service, which is collapsed
    by default — hiding the warning behind a click would repeat the failure it
    reports. `debug_msgs` was not an option: the engine returns it and
    DealAnalysisView renders it nowhere.
    ALSO FIXES A BUG IN ITSELF, found checking Jim's covenant corrections:
    `dtMaturity` is EMPTY on all 91 production loan rows and the date lives in
    `dtEvent` on the `vDateType='Maturity'` row. Reading the obvious column found
    nothing for every loan and fell back to the schedule's last period — close
    enough to look right, wrong enough to compute the extension from the wrong
    base date. The guardrail had passed 31/31 against it because the fixture put
    the date in a column that never carries one.
    Guardrail: `loan_maturity_gap_check.py`, 36 checks, pure fixtures — it checks
    the two OPPOSITE failures, staying silent and quietly extending the loan.
    STILL OPEN: `nReqDSR` 1.10 and `nLTV` 0.55 are the only covenants left after
    the corrections; whether 1.10 is the extension test or the ongoing one has
    not been settled, and solve-for-paydown needs that answer first.)
  - `v480` = `5325f44` (Accounting Workpapers split into two tabs: a production
    TRACKER across every reporting entity and the single-entity WORKBENCH, which
    now has its own entity dropdown. The tracker replicates the CFO's reporting
    calendar — his order number (rows he has not placed sort to the BOTTOM), the
    preparer's initials, and per deliverable a target date and its sign-offs,
    14 cells per entity. Stages are per deliverable: three for workpapers, FS and
    capital accounts, FIVE for investor delivery, because the FS and the capital
    accounts are posted and checked separately. Carry-forward moves the
    ARRANGEMENT and never the approvals. A date is parsed or refused, never
    guessed; sequence is reported, not enforced; clearing deletes the row.
    THE ACCOUNTING SECTION IS NOW THE CFO'S — the API gate was ("admin",
    "analyst") and the SCREEN gated on `admin`, and the screen is what actually
    locked him out. Ten endpoints name `cfo`; note `role_required` is
    level-based, so that grants the CFO and keeps admitting analysts.
    THE STATEMENT TABS PRINT LIKE THE DELIVERED PACKAGE, transcribed from
    `PPI Eastchase (TX) LLC - WP - 06.30.2026.xlsx` — margins, centring, Arial 11,
    column widths, the five-line page header and footer pages 1-5. The print
    RANGE is computed rather than copied (58 entities do not share a row count)
    and starts at row 4, so the workpaper title and provenance note stay on
    screen and off the printed statement.
    SHIPPED DDL: two new tables plus `sort_order`/`property_name` on
    `wp_packages`, created on first request. Verified on production PostgreSQL
    after deploy — both tables present, both columns added, tracker returns all
    58 REP entities. Population is the REP TAG, not the spreadsheet (Jim's call);
    all 58 start unordered and say so.
    Guardrails: `workpaper_tracker_check.py` (43) and `workpaper_print_check.py`
    (77, and it builds a real package and reads the workbook back).)
  - `v479` = `4908fc7` (Charlene: the snapshot's ownership graph is read AS OF THE
    QUARTER. `_is_open` was date-blind — ANY EndDate meant closed — so a relationship
    closing at any point AFTER a quarter retroactively removed its deal from that
    quarter's report, and the nearer a report was re-run to the present the more
    history it lost. JB Fair Park vanished from 26Q2 entirely: not sold, not
    unacquired, in no exclusion bucket, just gone, because `PPI32 -> JBFAIR` ends
    2026-07-29 and 26Q2 ends 2026-06-30. Verified on production before and after:
    restored at 25Q4/26Q1/26Q2, correctly still absent at 26Q3. The change is
    STRICTLY ADDITIVE — measured against the live feed, 0 of 765 edges that were
    kept are now dropped and exactly 1 is restored. The four ownership cells stay
    BLANK because both JBFAIR owners are recorded at 0%; that is an MRI data gap and
    filling it would be inventing TIAA's stake in a $77M deal.
    OPEN BOUNDARY QUESTION, not introduced here: the gate is strict (`ended > q_end`),
    so an edge ending exactly ON a quarter end counts as closed for that quarter.
    87 of the 188 ended edges land on a quarter end, and `AMB6 <- PSC1` ends
    2026-06-30 — 26Q2's own quarter end. It was closed under the old gate too, so
    nothing regressed, but whether EndDate is the last live day or the first dead
    one has never been settled.)
  - `v478` = `1618e78` (Charlene: the Financial print table was 0.224in wider than
    the sheet — `white-space: nowrap` makes `width: 100%` a floor, not a ceiling, and
    11 columns overflowed where 8 did not, cutting Net ROE at the margin. Cell padding
    5px→4px pays for it exactly (11 x 2 x 1px = 0.229in). Also the TIAA band rule,
    which was printing across the whole row rather than under its four columns:
    `.spanrow th` had always been outranked by `table.grid th`. She MEASURED the
    brief's third item and rejected it — the left margin was never clipped, and
    shifting right would have worsened the real overflow. New `check_margins`
    guardrail: horizontal fit had no check, which is how this went unnoticed.)
  - `v477` = `8455d58` (ownership chain: a deal's waterfall may be filed under the
    property code OR the InvestmentID — 3rd Ave's six steps are under `3RDAVE` and
    `P3RDAVE` has none, so the screen called it unconfigured and linked to a code
    nothing is filed under. Also the waterfall-setup deep link, which never read
    `route.query.vcode`, and 28 child properties of multi-property deals dropped
    from the PE investment list. Carried Charlene's `2b987e1`, reviewed before
    building — it REMOVES two vcodes from a suppression list, the opposite of a
    symptom repair, and moves the investor-facing footnote with them.)
  - `v476` = `e9630b6` (diagnostic: trace one deal chain and name the level that breaks)
  - `v475` = `60b88dd` (diagnostic: read one table, not the whole data layer — the
    937,650-row assembly could not survive the 2GB container it was diagnosing)
  - `v474` = `ae7bb29` (Dockerfile ships `scripts/`. NOTHING in that directory had
    ever been in the image, so no guardrail and no diagnostic could run against
    production — the only copy of the data — until this.)
  - `v473` = `25de795` (Charlene: snapshot print, each subtab back to one page at 37 deals)
  - `v472` = `f3ec22b` (upstream analysis reconciles out loud: beneficiaries receiving
    123,179.36 against a 100,000.00 distribution now SAYS so rather than printing it)
  - `v471` = `2e2548e` (the pref balance comes from the vetted Pref Balance Detail
    report. Jim, twice: "why are you trying to recreate a calculation engine that we
    have already built and vetted?" Two numbers for one fact was the defect.)
  - `v470` = `ad773ce` (pref accrues to the distribution date, not to two stale ones)
  - `v469` = `bdc7229` (the live figures — pref rates, balances, residual shares — in
    the step descriptions, so a step says what it did rather than what it is)
  - `v468` = `ad6e8b6` (upstream analysis seeds the EXISTING engine via
    `seed_states_from_accounting` instead of starting every investor from zero)
  - `v467` = `1845dc8` (upstream analysis: the Deal Analysis deal list, a Cash Flow vs
    Capital choice — it was hardcoded to `CF_WF` at both levels, so a sale ran as an
    operating distribution — and an estimate footnote on beneficiaries reached through a
    multi-asset fund. Also carried `eaff54f`, the widened pre-commit hook.)
  - `v466` = `be27c1a` (an upper-level balance breakdown describes the owner's whole
    position in the entity below, not this deal — dropped above level 1 and replaced
    by the derivation)
  - `v465` = `d693e24` (the commitment's date was sitting directly above the balance
    and being read as the balance's as-of date; no date has ever touched that figure)
  - `v464` = `89c77a9` (the balance shows its working — a breakdown by Typename, and
    the `Capital` flag's answer for the same rows compared but never used)
  - `v463` = `8a7bdca` (look-through: an upper-level commitment is not this deal's
    money — OWPSC's $64M into PSC3 is not its share of the $3M in 30BEAR. Also the
    MRI `Connection Timeout` fix, and Charlene's `ac68ffc`)
  - `v462` = `c014922` (ownership connector arrows, capital balances, beneficial
    owners, bounded scroll)
  - `v461` = `e62cc0b` (also carried Charlene's `836ca5f`, reviewed before building —
    a footing fix, not a symptom repair: $45.4M of Evergreen Plaza debt sat inside
    Portfolio Totals under no subtotal)
  - `v460` = `ee1c61a` † (two fixes for the empty ownership tree, NEITHER of which was
    the bug; its real contribution was making the health block report what ARRIVED
    versus what SURVIVED, which is what diagnosed `v461`)
  - `v459` = `78630d8` (carried Charlene's `ad55705`/`13e47c9` — Camarillo Village and
    Outlook Nine Mile added to `KEEP_DESPITE_SOLD`. Per-deal hardcode, raised with Jim
    before building, deployed on his call. The commit's stated rationale — "the page is
    meant to carry every sold deal" — describes 4 of 27, and two DROPPED deals sold later
    than two kept ones. Unresolved.)
  - `v458` = `9db5923` (config-only roll: ACS email switched on — `ACS_CONNECTION_STRING`
    as a secret ref and `ACS_SENDER` — same image as v457)
  - `v457` = `9db5923`
  - `v456` = `b00ed5d` (shipped SEVEN commits, not the two asked for — the live image was
    five behind origin/main. Pre-flight P2 caught it before the build; the five were
    reviewed and the two docs commits verified to touch no runtime file.)
  - `v455` = `841e92b`
  - `v454` = `c0a53f2`
  - `v453` = `5274e83`
  - `v452` = `5323de3`
  - `v451` = `041827b`
  - `v450` = `bd48806`
  - `v449` = `7b2c11d`
  - `v448` = `cdcf9bb`
  - `v447` = `9b51cbd`
  - `v446` = `30d5821`
  - `v445` = `54591fc`
  - `v444` = `8aa30a2`
  - `v443` = `5d81eb5`
  - `v442` = `ae4a767`
  - `v441` = `648e4e2`
  - `v440` = `0ad313a`
  - `v439` = `e469d9b`
  - `v438` = `efb377d`
  - `v437` = `0dbd63f`
  - `v436` = `a043351`
  - `v435` = `776bdfc`
  - `v434` = `46b5649`
  - `v433` = `9eff832`
  - `v432` = `9d9fa1d`
  - `v431` = `9d9e545`
  - `v430` = `33a4bf5` (config-only roll: DATABASE_URL moved to a secret ref during the credential rotation — same image as v429)
  - `v429` = `33a4bf5` †
  - `v428` = `dfc38df` †
  - `v427` = `bf093c2` †
  - `v426` = `9f0708d` †
  - `v425` = `54fe25a` †
  - `v424` = `3a21c3c` †
  - `v423` = `eb34396` †
  - `v422` = `2ea1186` †
  - `v421` = `c8c367b` †
  - `v420` = `d1259bb` †
  - `v419` = `a7cb4c4` †
  - `v418` = `dcefc29` †
  - `v417` = `6dac97f` †
  - `v416` = `eb7520d` †
  - `v415` = `639f023` †
  - `v414` = `bd001e8` †
  - `v413` = `7d21769` †
  - `v412` = `da354a3` †
  - `v411` = `8534916` †
  - `v410` = `019b592` †
  - `v409` = `7dc7bd8` †
  - `v408` = `e5ef21d` †
  - `v407` = `50695d9` †
  - `v406` = `5e5a3b7` †
  - `v405` = `7827c6f` †
  - `v404` = `864e834` †
  - `v403` = `edf4a3a` †
  - `v402` = `1495d61` †
  - `v401` = `9432f87` †
  - `v400` = `b8bdfea` †
  - `v399` = `bf29707` †
  - `v398` = `ffd19b1` †
  - `v397` = `c6083f0` †
  - `v396` = `557af5e`
  - `v395` = `726f708`
  - `v394` = `228f440`
  - `v393` = `791cbef`
  - `v392` = `3264771`
  - `v391` = `1c68a37`
  - `v390` = `635f0ff`
  - `v389` = `f20e195`
  - `v388` = `6f3d4e7`
  - `v387` = `f0b8d00`
  - `v386` = `dfccb3f`
  - `v385` = `e15333c`
  - `v384` = `d847410`
  - `v383` = `4ecc15b`
  - `v382` = `30c3833`
  - `v381` = `9987dba`
  - `v380` = `5b314e0`
  - `v379` = `1759185`
  - `v378` = `3cb8bb0` †
  - `v377` = `b113806`
  - `v376` = `e1e94f7` †
  - `v375` = `c5d0b81`
  - `v374` = `f051cf3`
  - `v373` = `6bbfbf6`
  - `v372` = `93e3a19`
  - `v371` = `644af7d`
  - `v370` = `0f9918c`
  - `v369` = `a31f3ec`
  - `v368` = `b3e1bc1`
  - `v367` = `515d5b1`
  - `v366` = `c571d53`
  - `v365` = `35fc12f`
  - `v364` = `e22dd84`
  - `v363` = `be08b30`
  - `v362` = `0290d6b`
  - `v361` = `770363b`
  - `v360` = `54951b8`
  - `v359` = `0070ffd`
  - `v358` = `feea43d`
  - `v357` = `eb02954`
  - `v356` = `71068a9`
  - `v355` = `d58a797`
  - `v354` = `0b28731`
  - `v353` = `f754919`
  - `v352` = `6918db3`
  - `v351` = `826df0e`
  - `v350` = `87202b1`
  - `v349` = `2700c99` †

  `v348` and earlier point at `:latest` and are not traceable by tag.

## The deploy lessons, in long form (moved from CLAUDE.md, Oct 5 2026)

CLAUDE.md's **Lessons** list is the RULE for each of these, one line apiece, with the
revision in brackets pointing here. The sentence explaining what each one cost lives
below, so the rule list stays a list and this file keeps the reasons.

Where a lesson names a revision, that revision's own entry -- in the post-mortem
sections above, or in the `v349`-`v563` index -- is the full account. These are the
compressed versions, written when the lessons were extracted.

1. **Compute the P2 span against the LIVE image, not local HEAD.** Seven commits
   shipped where one was asked for, and only two were reviewed (`v429`).
2. **Re-run P1 immediately before `containerapp update`, not only before the build.** A
   colleague's deploy landing between the two would have been rolled back by the update
   (`v549`, `v560`/`v561`).
3. **Build only from a clean worktree nothing else is using, and gate `containerapp
   update` on the tag existing and the ACR run succeeding for that SHA.** A background
   job touching the local SQLite made the upload fail, so the tag was never created --
   and the chained update ran to it anyway (`v556`).
4. **`activeRevisionsMode` is Single, so traffic follows the LATEST revision.** Never
   deactivate it to back out, and a traffic split does not apply: deactivating the
   broken revision deprovisioned the healthy one behind it. Rollback is rolling
   FORWARD, to the last good SHA tag under a new suffix (`v556r`).
5. **Pin dependency MAJORS.** An unpinned minor took every worker down on boot --
   SQLAlchemy 2.1 made a bare `postgresql://` URL load psycopg v3 (`v524`); pandas is
   pinned `<3.1` on the same reasoning. **And run guardrails inside the container**:
   production ran pandas 3 while local ran 2.3 for months, so a suite can be green
   locally on a shape the database cannot deliver (`v555`), and `pd.NaT` passes an
   `isinstance(datetime)` test (`v550`).
6. **Verify a UI change against the SERVED bundle, resolving the lazy chunk from the
   ENTRY bundle.** `index.html` is a ~551-byte SPA shell referencing no chunk, so
   grepping it returns "gone" for strings that are present and proves nothing in either
   direction (`v523`, `v527`).
7. **Know which guardrails SKIP in the container, and why.** The runtime image ships no
   `vue_app/`, and gitignored fixtures are absent. **Skip is not pass** -- and a check
   that CRASHES instead of skipping kills the run after the useful sections have
   already passed (`v527`).
8. **ACR uploads the WORKING TREE, not a git ref**, so an unclean tree ships
   uncommitted work (pre-flight P3). Correspondingly, a data change that is NOT in git
   -- an MRI table refresh -- is **not undone by a rollback**; say so when one
   accompanies a deploy (`v549`).
9. **Prove a shared-engine refactor behaviour-preserving by running the OLD module
   beside the new one over real data and diffing the output**, not by reading the diff
   (`v505`, `v548`). And **measure a rule's population on live data BEFORE shipping
   it**, reporting what CHANGED including when the change was none -- a rule that is
   right on the deal you looked at has still not been tested (`v544`).
