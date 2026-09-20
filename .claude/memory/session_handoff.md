# Session Handoff — through Sep 20 2026 (v512 live)

## Sep 20 2026 — THE LEASE CORPUS, and a regression the diff caught

**`v509` = `49d2120`, `v510` = `a12f98a`, `v511` = `07a83ed`, `v512` = `e473a07`.**
Open items: **`open_items.md` §9.2** (a re-extraction is RUNNING as this is written)
and **§9.8** (the CFO's `ITEM = 1`, still needing Jim's call).

### Read this first: a re-extraction may still be in flight

At the time of writing, a full re-extraction of all 419 term-bearing lease documents
is running detached on production (`/app/reex.py`, log `/app/reex.log`, revision
`v512`). It is resumable — anything finished is marked `extracted` and is not
repeated — so if it died, re-launching is safe. **When it completes, the
consolidation must be re-run and DIFFED**, which is the outstanding half of §9.2.
`scripts/` holds no runner for it; the script is in the session scratch and is
reproduced in shape by `open_items.md` §9.2.

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
open-period entries balance). §9.8, awaiting Jim.

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
