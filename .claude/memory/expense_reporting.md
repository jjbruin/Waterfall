# Employee expense reporting (Oct 2 2026; LIVE at `v553` = `79d21a2`)

**Live at `v553` (Oct 2 2026).** Built on branch `feat/expense-reports` (off `feat/section-access`, which it
needs for the Expenses section), not merged or deployed as of Oct 2 2026. Delete this
paragraph when it ships. What is built and how, below under "Phase 1 as built".

Jim, Oct 2 2026: employees start a report for a period, upload their receipts, the
app reads them (as it reads leases), the employee corrects and adds no-receipt items,
the manager approves, accounting codes it, and the app produces the MRI JE upload.
Section access (`feat/section-access`) exists so expense-only employees see nothing
else.

Accounting's inputs, read in full:
- `Expense Reimbursement - Example.xlsx` — the employee form. Columns: Date/Period,
  Accounting Category, Purpose, Deal, Vendor, Comments, Total, Receipt Y/N,
  no-receipt reason. Lists: 15 categories, 6 purposes, ~50 deals.
- `Expense Reimbursement - Accounting Coding - 09302026.xlsx` — accounting's
  aggregation: JE description formula, GL account, interco entity, interco debit /
  credit accounts, per-employee check against the submitted total, an
  `Interco Ownership Splits` tab and a COA tab.
- `2026-09-24 Payroll ERs Entries - JE Upload.csv` — the accepted MRI upload, 119
  lines across PSCMAN, PPI2, PSC3, TGA6. THIS IS THE ACCEPTANCE TARGET.

## What the Sep 24 file teaches (measured, not assumed)

| Finding | Consequence for the design |
|---|---|
| **PPI2 is out of balance by 0.30**: cab charges debit 159.66, credit -159.96. The file totals -0.30 | A hand-built file posts typos. The per-entity balance check committed today in `treasury_upload.validate_gl` (`be7a33e`) refuses exactly this |
| Fred's 85 miles is 61.61, 61.62 and 61.63 on different lines | Mileage is computed once (miles x rate, rounded once), never typed |
| Elaine's mileage is $0.76/mi, everyone else's $0.725 | ONE rate, set by accounting with an effective date |
| Booking account follows the DEAL, not the category: Operations -> the category's MR5* account; a pipeline deal (Windsor Square, Market Poplar, Pine Tree) -> `MR11000012` Deal Cost Receivable; an owned deal (Apple, Pontchartrain, Fairview) -> `MR15000001` Due to/from Intercompany with `RLTDENTITY` = the owning entity | A rule, defaulted and editable — to be confirmed with accounting |
| An interco line creates a PAIR at the entity: debit the expense (`RLTDENTITY` = the investee, e.g. INVF7, PPIFVH), credit `MR15000002` Due to Manager | Generated, never typed |
| Accounting recodes some categories (Pontchartrain dinner Travel -> Meals at PSC3; Bob's phone Office -> Telephone) | Accounting's account outranks the employee's, and the employee's stays beside it |
| PPI2 is CAD: 712.98 USD -> 1,007.75 CAD, the USD in the description | The app holds no FX rate (same as intercompany): typed per batch, stated on the line |
| The credit is ONE line, `MR20000001` "Expense Reimbursement Activity - Trinet Payroll - 202609 End of Month" | Reimbursed through payroll; a batch is per payroll run |
| Charlene's lines are a PERIOD ("Sep 1 - Sep 21"), not a date | A line carries a date OR a date range |

## The flow

```
Employee: draft -> submitted ----> Manager: approved -> Accounting: coded -> batched -> posted
                     ^                 |                       |
                     +--- returned ----+-----------------------+   (a note is required)
```

1. **Report** — employee, period start/end. One open report per employee per period.
2. **Receipts** — the SAME upload as Lease Review's documents panel: pick files
   (`multiple`) or pick a folder (`webkitdirectory`, `LeaseReviewView.vue:1815`) —
   the browser sends every file in it. There is no live link to a folder in either. PDF, JPG, PNG,
   HEIC (iPhone default — needs `pillow-heif` to render), WEBP, TIFF, GIF. Bytes stored
   in the database like treasury statements (`file_data`), SHA-256 kept so the same
   receipt cannot be claimed twice — across reports and employees, not just within one.
3. **Extraction** — the lease engine's route, not a second extractor: `claude-opus-5`;
   a text PDF as text, a scan or photo as an image/document block; join text blocks
   (thinking comes first), check `stop_reason` for refusals, record
   `_extraction_source`. Fields: date, vendor, total, currency, tax, tip, card last 4,
   a suggested category, and WHETHER THE FILE HOLDS SEVERAL RECEIPTS (a month of Uber
   in one PDF) — then one line per receipt.
4. **The form** is built from the extraction. Every field the employee corrects keeps
   the extracted value beside it (the lease "settle" pattern: figure, prior value, who).
   A line with no receipt REQUIRES a reason. Mileage: miles x the rate in force on the
   line's date, tolls a separate line. Split: one receipt across several deals by % or
   amount, which must foot to the receipt. Recurring items (the $50 phone line ~20
   times a month) copy from the prior report.
5. **Submit** to the employee's approver. **Every submitter must have one** — submit is
   refused otherwise, and nobody approves their own report. Decided (Jim, Oct 2 2026):
   - **an approver's own report goes to the CFO** (the `cfo` role);
   - **the CFO may approve ANY report too** (Jim, Oct 2 2026) — recorded as "CFO in
     place of …" when it was sent to someone else;
   - **the CEO or the President may approve ANY report** when its approver is out —
     the `ceo` / `president` review roles that already exist in `review_roles`
     (One Pager chain), not new roles. Whoever approved is recorded, and an approval
     by someone other than the named approver says so on the report.
6. **Manager review** — the report with each receipt image beside its line; approve,
   or return with a note. Tracking screen on the Review Tracking pattern: who is
   waiting on whom, and for how long.
7. **Accounting coding** (Accounting section, `ACCOUNTING_ROLES`): every approved line
   for the period in one grid, PRE-CODED by the rules above:
   - description `ER - {Employee} - {Deal} - {Comment}` (`Interco - ` prefixed),
     editable, original kept;
   - booking account from the deal rule; the category's MR5* account at the entity;
   - **ownership splits from commitment AMOUNTS** — the same allocation treasury uses
     for the investor split (CapitalPercent is 4dp and does not foot), walking a PPI
     pass-through to its owners: Gallery (PPI25) -> PSCKOC 70% / PSC3 30%.
     Accounting's `Interco Ownership Splits` tab is the check: rebuild it and compare.
   - per-employee total vs what was submitted, shown rather than typed.
8. **Batch** — pick the payroll date; preview; generate the GL CSV through
   `treasury_upload`'s writer and `validate_gl` (balances in total AND per entity, one
   period). Stored like `ic_je_batches`, so a reimbursement cannot be batched twice
   before MRI shows it; "posted" is read off `gl_detail`. Void allowed.

## Who sees what

Two axes, as section access already separates them:
- **Section**: a new sidebar section **Expenses** (`/expenses`), registered in
  `flask_app/auth/sections.py`. The accounting coding screen sits under Accounting.
- **Row-level, server-side**: an employee sees their own reports; an approver sees
  reports submitted to them; accounting sees everything at or past approval. Receipt
  images follow the same rule — "on demand" for the approver and accounting, nobody
  else. This is NEW: nothing in the app is per-record today, and the guardrail must
  call each endpoint as each of owner / approver / accountant / other employee.

**The section access process is NOT changed** (Jim, Oct 2 2026). Expenses is one more
entry in `SECTIONS`, ticked by default like every other; the Admin sets each
employee's sections. Row-level visibility is a rule about records inside the section,
not a change to who can open it.

## Data

`users` gains `full_name` (the JE names "Bob Pfeiffer", not `bpfeiffer`) and
`approver_user_id`. New tables, all `PROTECTED_TABLES` (app-written, only copy):
`er_reports`, `er_lines` (date or range, category, purpose, deal, vendor, comment,
amount, miles, receipt link, no-receipt reason, extracted values), `er_line_splits`,
`er_receipts` (bytes, hash, extraction JSON), `er_events` (submit/approve/return with
notes), `er_coding` (accounting's account, entity, description, original kept),
`er_batches`, `er_mileage_rates`.

## Build order

| Phase | What | Proves it |
|---|---|---|
| 1 | Names + approvers, report + lines typed by hand, submit / approve / return, tracking | Row-level access guardrail, both directions |
| 2 | Receipt upload, extraction, corrections, images beside lines | Real receipts from this month, incl. a phone photo and a multi-receipt PDF |
| 3 | Accounting coding, deal rule, commitment splits, FX, batch + CSV | **Rebuild the Sep 24 upload from the coding workbook's lines** — identical except the 0.30 PPI2 typo, which must be refused |
| 4 | Posted detection, recurring lines, duplicate receipts, mileage rate history | |

## Open questions

Answered Oct 2 2026 (Jim):
- Approvers -> CFO; CEO or President as backup for any report; access stays the
  Admin's; receipts upload exactly as lease documents do.
- **Owned vs pipeline is the employee's choice**: an owned property from asset
  management's list, otherwise the pipeline deal's name typed. Phase 3's rule reads
  `deal_kind`: `deal` -> interco to the owning entity, `pipeline` -> Deal Cost
  Receivable (`MR11000012`), `operations` -> the category's account.
- **Every report is reimbursed through TriNet payroll, `MR20000001`, ONE BATCH PER
  PAYROLL DATE.**
- **No caps or policy limits.** Everything is subject to manager approval.
- **"FK - Benefits" is part of the process**: Fred's benefits reimbursement, recurring
  monthly, added manually by accounting today. Phase 3/4: a recurring reimbursement
  accounting adds to a batch (not an employee line), carried to each payroll batch.

Still open: nothing blocking.

## Phase 1 as built (Oct 2 2026)

`flask_app/services/expense_service.py`, `flask_app/api/expenses.py`,
`vue_app/src/views/ExpensesView.vue`, guardrail `scripts/expense_report_check.py` (74,
proved against five injected defects).

- **Section**: `expenses` in `SECTIONS`, `/api/expenses` in `API_SECTIONS`, a
  standalone sidebar link after Reports. Nothing else about section access changed.
- **`er_` is a restricted table prefix (Accounting)** and all six `er_*` tables are
  protected. Without the prefix, Data Explorer would show every employee's reports to
  anyone with Data Management, past the per-report rule.
- **Who approves is COMPUTED at submit** (`route_for`) and stored on the report:
  approver of others -> CFO (any `cfo`-role user); the CFO -> CEO or President;
  otherwise the named approver. The setup tab shows the same computation. CEO /
  President are the existing `review_roles`; they may decide any submitted report
  except their own, and an approval in place of the named approver says so
  ("President in place of Max Manager").
- **A report you may not read is 404**, never 403. Drafts: owner only. Submitted:
  plus whoever may decide it. Approved: plus `ACCOUNTING_ROLES`.
- **Setup**: name on reports and approver — admin role writes; admin + accounting
  read. Mileage rate — accounting writes. **No rate is seeded**: mileage lines are
  refused until accounting sets one (Sep 24's file implies $0.725, but Elaine used
  $0.76, so the rate is theirs to state).
- **Categories are the FIFTEEN in accounting's template** (Jim, Oct 2 2026), named in
  `CATEGORY_NAMES` and numbered from `gl_accounts` by name, so the chart stays the one
  source of the number. All MR5* had been 52, carrying Payroll, Interest and
  Depreciation. A template name the chart lacks is reported, not dropped. Measured
  locally: all 15 resolve.
- **Owned deals are asset management's own list** — `data_service.get_inv_display`,
  the source behind Deal Analysis's dropdown, with its "Name (vcode)" label.
  **ONE CODE PER DEAL**: where a name carries a P000 code and an entity-id code, only
  the P000 is offered; an entity-id-only deal keeps its code (Jim: "If most are
  P000... then use the P000 otherwise use the entity id version"). Measured locally:
  that list is already 67 deals, all P000, so the rule changes nothing today — the
  duplicates first seen came from reading the raw `deals` table, which this no
  longer does. The rule stays as the stated policy.
- **A pipeline deal is TYPED** (Jim: "otherwise, allow the employee to type the name
  of the pipeline deal"). Stored `deal_kind = 'pipeline'`, no code, the name as typed;
  the screen sends `PIPELINE` as the choice, never stored as a code. `prospect_deals`
  is not read.
- A line is a date or a date range; a split must foot to the line (percent entry is
  converted to amounts once, the last row taking the remainder); receipt Y/N with a
  reason required for N, exactly as accounting's template — phase 2 replaces Y with
  the attached file.
- Verified in the running app (local): report opened, a line saved and totalled,
  submit disabled with the reason, setup tab rendered, draft deleted.

## Phase 2 as built (Oct 2 2026): receipts, read and shown beside the line

Jim: the employee sees "the image of the uploaded invoice related to the line they
are completing", to correct handwritten amounts the reader missed.
`flask_app/services/expense_receipts.py`, `vue_app/src/components/expenses/ReceiptViewer.vue`,
guardrail `scripts/expense_receipt_check.py` (53, no API calls; injection-proved).

- **Upload is Lease Review's**: pick files or a folder. **One file per request** —
  `MAX_CONTENT_LENGTH` is 50 MB and a folder of phone photos passes it; counting
  through them is the progress. `.DS_Store`-type files are skipped and said so.
- **Stored as uploaded** (`er_receipts.file_data`, protected). HEIC/TIFF/BMP also get a
  JPEG `view_data` a browser can show. Needs `pillow-heif` (added to requirements,
  bounded `<2`, with `Pillow<13`).
- **The reader is the lease engine's route**: `EXTRACTION_MODEL`, a PDF as a document
  block, page images on a retry (`_render_pdf_pages`), `_pdf_fits`; text blocks
  joined (thinking first), `stop_reason` checked for a refusal. Its own prompt.
  An image is sent UPRIGHT (EXIF rotation applied — a phone portrait is stored
  sideways) and at most 2000 px.
- **One line per receipt in the file**, on its page, with `extracted_json` kept beside
  the employee's figures. The total is the amount PAID — a handwritten tip and total
  win over the printed total, and the reader says when an amount is handwritten.
- **A receipt is now an ATTACHED FILE**: "receipt Y" with nothing attached means
  nothing; a line needs a file or "no receipt" with a reason.
- **The image is fetched with the token** and shown from a blob URL: an `<img src>`
  cannot send the Authorization header, and the file is per-record (owner, approver,
  accounting once approved; 404 otherwise).
- **Duplicates**: the same file twice on one report is not stored; the same file on
  another report is stored and flagged on the line (a shared dinner is one receipt).
- **A file with lines from it cannot be re-read** — remove its lines first, so a re-read
  never overwrites what the employee settled.
- **Two defects found by the new check**: `save_employee` never created the tables (a
  fresh database 500'd on the first approver set — phase 1's check called `options`
  first and hid it); and a failed reading built its reply INSIDE the transaction, so
  the screen showed the receipt still unread after the row said otherwise.
- **Verified with the real model locally** on a drawn Arnaud's receipt: printed total
  161.99, tip 30.00 and total 191.99 written in: read as 191.99, printed 161.99,
  handwritten flagged, vendor, date, category and "Dinner, 3 guests"; the form showed
  the image beside it with that reading above. Test data deleted.

## Phase 3 as built (Oct 2 2026): accounting's coding and the payroll batch

Accounting > **Expense Coding** (`/expense-coding`), `flask_app/services/expense_coding.py`,
`flask_app/api/expense_coding.py`, `vue_app/src/views/ExpenseCodingView.vue`, guardrail
`scripts/expense_coding_check.py` (46, four injected defects each caught).

- **Every approved line arrives PRE-CODED**; only what accounting CHANGES is stored
  (`er_coding`, per line or per split), so a corrected chart or ownership record still
  reaches every row nobody decided by hand. Changed fields are starred; the employee's
  category stays beside a recode.
- **The booking follows `deal_kind`**: operations -> the category's account; pipeline ->
  `MR11000012`; owned -> `MR15000001` at PSCMAN (RLTDENTITY = owning entity) plus, at the
  entity, the expense against `MR15000002`.
- **WHICH ENTITY OWNS THE EXPENSE IS READ FROM THE DATA**: `ownership_chain_service.build_chain`
  (commitment dollars), walked up each branch to the FIRST owner that keeps an
  intercompany account with PSC Manager (carries `MR15000002`, or is a `MR15000001`
  segment on PSCMAN, basis A/B). That one rule gives Gallery -> PSCKOC 70 / PSC3 30
  (PPI25 keeps no such account: "a simple pass through") AND Apple -> PPI2 (it does).
  A name-based rule could not give both. Branches never reaching one (an operating
  partner) are dropped and the rest RE-BASED, said in the basis text. RLTDENTITY at the
  entity = the entity below it on the path (INVF7 for Pontchartrain, as in the file).
  **NOT YET RUN AGAINST PRODUCTION OWNERSHIP** — locally `commitments` has 3 rows.
  Before accounting relies on a proposal, compare it on production for the Sep 24
  deals: Apple -> PPI2, Pontchartrain -> PSC3 (INVF7), Fairview -> TGA6 (PPIFVH),
  and the six splits on accounting's `Interco Ownership Splits` tab.
- **Splits use `treasury_upload.allocate`** (largest remainder, to the cent); **the file is
  `build_gl_csv` / `validate_gl`**, with the per-entity balance check brought over from
  `feat/intercompany-pay-je` VERBATIM so the two branches merge cleanly.
- **One batch per payroll date**: one credit at PSCMAN to `MR20000001`, "Expense
  Reimbursement Activity - Trinet Payroll - {YYYYMM} {suffix}". Reports are CLAIMED by a
  conditional UPDATE (status `batched`, `batch_id`), so two people cannot batch one
  report. "Posted" = the GL shows that credit. Void releases the reports, coding kept.
- **A non-USD entity** is listed in `er_entity_currency`; the batch requires a USD-to-X
  rate for it, books the entity side in that currency and appends "(712.98 USD)". PSC
  Manager's side stays USD. **Accounting must list PPI2 as CAD** — nothing seeds it.
- **Recurring reimbursements** (`er_recurring`, e.g. FK - Benefits) are ticked per batch.
- **Reads are closed to analysts too** — the grid is everyone's spending. The one named
  exception in `accounting_access_check` (`READ_CLOSED_PREFIXES`), asserted both ways.

### Acceptance — accounting's Sep 24 upload, rebuilt

The file's own coding was fed back through the app (its accounts, entities, related
entities and descriptions as accounting's decisions) and the batch compared line by line:
**114 of 119 lines identical** (entity, account, amount, description, related entity) —
every PSCMAN, PSC3 and TGA6 line. The five that differ are all PPI2: the 0.30 cab typo
(the file credits 159.96 against a 159.66 debit; the app balances, and REFUSES the file as
written), one CAD cent (613.44 vs 613.45 — accounting's rates vary 1.41341-1.41350 line
to line, the app uses one per batch), and accounting typing a shorter description at
PPI2 than at PSCMAN for one mileage line. This proves the ASSEMBLY; the ownership
PROPOSAL is the production check above.

## Phase 4 as built (Oct 2 2026)

Guardrail `scripts/expense_phase4_check.py` (23), every rule asserted both ways and proved
against injected defects.

- **Recurring lines.** "recurring every month" on a line; "Copy recurring lines from my
  last report" brings forward the marked lines of the employee's most recent other report
  that has any: category, purpose, deal (or split), vendor, comment, amount and the mark.
  NOT the receipt (this month's is a different file) and NOT the date (set to the new
  period's start). A line already present is not copied twice. The comment is copied as
  written, so "- August" has to be edited; the toast says so.
- **Duplicates: same vendor (normalised: case, spaces, punctuation), same date, same
  amount on ANOTHER report** -- all three, because twenty people claim $50 of phone on the
  1st. Counts the employee's own other reports at any status (a report started twice is
  the commonest case) and colleagues' only once SUBMITTED (a draft is private). Names the
  report only when it is the employee's own. A WARNING, never a block; shown to the
  employee, the approver and on accounting's coding grid. Separate from phase 2's
  identical-file flag.
- **Accounting returns an APPROVED, UNBATCHED report** with a required note (report
  screen, or inline on the batch tab). It goes back to the employee as `returned` and
  **must be approved again**, so an approved figure cannot be changed and paid unseen;
  accounting's coding is kept. A batched report is refused -- voiding the batch is how --
  by two independent guards (the status check and the conditional UPDATE); an injection
  removing one was still refused by the other.
- `er_reports.batch_id` is now added by the base table setup, not only by the coding
  module: the return reads it whether or not coding has ever run.

Mileage-rate history is the existing table (effective date, rate, basis, who); nothing
more was needed.

## The ownership proposal against production (Oct 2 2026, `v553`)

Run in the container against the real commitments and GL (intercompany population: 64
entities). Accounting's booking (Sep 24 upload and their `Interco Ownership Splits` tab)
against the app's proposal:

| Deal | Accounting | App | |
|---|---|---|---|
| Fairview (FAIRVH) | TGA6, rltd PPIFVH | TGA6 100%, rltd PPIFVH | match |
| Nottingham (NOTTNV) | TGANOT 48.49 / PSCKOC 51.51 | same | match |
| Woodlands (WOODSQ) | PSCKOC 67.51 / PSC1 32.49 | same | match |
| Ascent (ASCENT) | TGAAS 76.82 / PSCKOC 23.18 | same split; PSCKOC's rltd INVASC not PPIAS | split matches |
| Pontchartrain (PONTCH) | PSC3, rltd INVF7 | INVF7 100%, rltd PPI31 | WRONG |
| Gallery (THEGAL) | PSCKOC 70 / PSC3 30 | PSCKOC 70 / INVF2 30 | WRONG |
| Belleville (BELLES) | KOCTRS 50 / PSC3 50 | KOCTRS 50 / INVF11 50 | WRONG |
| Apple (APPLE) | PPI2 100% | PPI2 82.07 / PSS1 17.93 | WRONG |
| Brainerd (BRNERD) | TGA22 ~62.4 / PSC1 ~37.6 | TGA22 82.68 / PSC1 17.32 | WRONG |

- **The three PSC3 rows are ONE cause.** The investee funds INVF7, INVF2 and INVF11
  carry their own `MR15000002`, so "the first owner with an intercompany account" stops
  at them; accounting treats them as pass-throughs and books PSC3. Whether investee
  funds always pass through to their parent is accounting's rule to state -- a data
  criterion for it is not known yet, and a name rule (`INVF*`) would be a guess.
- **Apple and Brainerd are data questions**: the commitments carry PSS1 beside PPI2 into
  APPLE, and Brainerd's commitment dollars give different shares from accounting's tab.
- Brainerd's nine building properties (BRN-1..9) find no owner -- not reachable from the
  employee's deal list, which offers the parent deal only.
- Not changed: every proposal is editable and a hand-set split is kept. Until the rule
  is settled accounting should CHECK every intercompany row.

## The distance wizard (Oct 2 2026, built; ships with the GOOGLE_MAPS_API_KEY wiring)

`flask_app/services/expense_distance.py`, `POST /api/expenses/distance`, the "Measure
route…" panel under mileage. Guardrail `scripts/expense_distance_check.py` (26, stubs
Google; injection-proved).

- **Google, on Jim's call**: Routes API for the drive, Geocoding API for what each stop
  was understood as. Key = container secret `google-maps-key`, read server-side as
  `GOOGLE_MAPS_API_KEY`; never reaches the browser. Local key in `.env`.
- **A bare 3-letter code is asked as an airport** -- measured: Geocoding "PHL" alone is
  "Philippines". A code Google matches to a non-airport stops and says so.
- **Every stop shows what it resolved to.** Measured: "Pontchartrain Landing, New
  Orleans" resolved to Pontchartrain Blvd, a street -- the street address fixes it.
  Places API is the option if loose landmark names prove common.
- **The measurement is the server's** (`er_routes`, protected): a line points at it
  (`route_id`), only the employee's own, only on a mileage line. Miles typed over it are
  kept and the line warns "differ from the measured route".
- Round trip drives back as its own leg. Using a route pre-fills "no receipt — Mileage,
  measured route", as accounting's template records mileage.
- Real key, locally: PHL -> City Hall round trip 24.5 mi (11.9 + 12.5); EWR -> JFK 34.7;
  Philadelphia -> 5050 Poplar Ave, Memphis 1,006.7.
- The inline Purpose/Deal save carries `route_id`, verified -- otherwise a quick change
  would drop the route.

## MRI description rule (Oct 2 2026)

Jim: MRI's JE description is at most 80 characters and allows NO punctuation.
`treasury_upload.mri_description` (beside the GL writer) cleans to letters, digits and
spaces and trims to 80 on a word boundary -- the comment first, then the deal, never the
prefix or initials; common long words (Reimbursement -> Reimb, Conference -> Conf, ...)
are abbreviated ONLY when over the limit; a suffix (the CAD note "USD 712 98") keeps its
room. Expense descriptions: `ER FK <deal> <comment>`, `IC ER FK ...` for intercompany,
comment words the deal already says dropped; the payroll credit is
`ER Trinet Payroll 202609 <suffix>`. Accounting's typed text is cleaned when saved AND
at the batch. Applies to EXPENSE only: intercompany's fixed text already complies;
treasury's descriptions come from the bank and were not changed.
Note: accounting's accepted Sep 24 file has punctuation on all 119 lines and 12 over 80
-- either MRI accepted them then or accounting cleaned the file; the acceptance check now
compares against their text cleaned to the rule. Two employees with the same initials
would read alike -- not handled.

## Phones, and the three ways to add (Oct 7 2026, live in `v589`)
- **iPhone/iPad** (b0cf454): the picker's `accept` carries `image/*` so iOS offers Photos
  and the camera; touch gets "Take a photo" instead of "Upload a folder"; iOS's
  `image.jpg` is renamed "Photo <date time>". Guardrail `expense_mobile_check.py`.
- **Three equal cards above the lines table** -- Add an expense / Upload receipts (Photos
  or files on touch) / Upload a folder (Take a photo on touch). Employees clicked
  "+ Add an expense" and never saw the upload buttons lower down; the Receipts block now
  only LISTS files.
- **The pop-up reads the receipt itself** (`uploadInvoice`), through the SAME `/receipts`
  and `/extract` calls -- one reader. New expense: the reader's line becomes the line
  being edited, the employee's typing wins, the receipt fills blanks, a differing amount
  is SAID. Existing line: it takes the receipt and the reader's line is deleted. Busy
  blocks Save/Cancel/Close; Cancel asks, then removes what the pop-up added. A duplicate
  upload returns `receipt_id` so a re-picked photo (new name) is found.
  Guardrail `expense_add_options_check.py` (12 injections).
- **"PSC Expenses" on the home screen** (`v590`): `apple-touch-icon` in index.html; the
  Expenses page sets `apple-mobile-web-app-title` and removes it on leaving. Not full-screen
  mode (no popups there, and Microsoft sign-in needs one). `expense_mobile_check.py` section 6.

## Emails and the coding review (Oct 7 2026, the CFO's asks; live `v593`)
- **Email** (`expense_notify.py`, via ACS, sent in a background thread AFTER the action
  commits): submit -> whoever may decide (approver; CFO for an approver's own; CEO/President
  for the CFO's own); approve -> accounting + the employee; return (approver or accounting)
  -> the employee with the reason; coding submitted -> reviewers; coding returned -> its
  submitter. **Accounting = roles** cfo / accounting_manager / accountant WITH accounting
  authority; reviewers = cfo / accounting_manager. Nobody is emailed about their own report
  or submission. EVERY send, failure, missing address or "nobody to email" is written to
  the report's History. Links: `/expenses?report=<id>` (ExpensesView opens it) and
  `/expense-coding`. Local `.env` has no ACS, so local runs record "not sent".
- **Coding review** (`expense_coding.review_state`): not submitted -> submitted (accounting)
  -> reviewed (cfo / accounting_manager, `REVIEWER_ROLES`, `roles_exactly`); return needs a
  reason. ONLY REVIEWED reports batch. A coding change after review sends it back to
  submitted. The state counts only if submitted after the report's latest approval
  (`decided_at`), so a re-approved report starts over. Columns on `er_reports`, added by
  `ensure_tables` -- which RACED (lines + settings load at once, second ALTER "duplicate
  column", 500 + empty grid) and now tolerates a column that appeared in between.
- **Screen**: account NAMES on the coding grid, a real account dropdown (name -- number),
  payroll totals (all / selected). The receipts list on a report shows only receipts no
  line uses (or also on another report); Delete asks first.
- **The CFO's JE-sign question**: the batch already writes credits negative (production's
  one batch ER-20261031-C7FB66: MR20000001 -3,628.65, every MR15000002 negative);
  `expense_coding_check` now asserts it on the file.
- Guardrail: `expense_coding_check.py` 81 (`--inject=nogate|noacct|selfemail|silentnobody`),
  `expense_add_options_check.py` 42.

## The monthly cell phone reimbursement (Oct 7 2026; branch `feat/cell-phone-allowance`)
`expense_phone.py`. Jim: a fixed monthly reimbursement for personal cell phones, $50 now,
the CFO controls the rate, paid by an automatic batch, and cell phone bills declined on
reports. Decided: everyone set up in Expenses (a name on reports; never the admin account)
is on by default and the CFO can switch one off or bound the months; from October 2026;
cell phone BILLS only -- internet on Telephone & Internet is still claimed.
- **Rate**: `er_phone_rates`, dated by month, seeded once at $50 from 2026-10 (Jim's
  figure); set by `roles_exactly("cfo")` + accounting authority only. A rate for a month
  already paid is refused.
- **Paid by the payroll batch**: `build_batch` adds every (employee, month) owed through the
  payroll month, MR53000015 at PSCMAN, "ER <initials> Cell Phone Reimbursement <Month YYYY>",
  inside the payroll credit; `er_phone_paid` is the ledger (key user_id + month), written
  in the batch's transaction, released by voiding. No report needed. A month with no rate
  is reported, not paid at $0. Untick "Include the monthly cell phone reimbursements" to
  leave them out of one batch.
- **Not `er_recurring`** (accounting's standing items, ticked per batch): those pay per
  BATCH, not per month -- two batches in a month would pay twice, a month with no batch
  would pay nothing -- carry no CFO-owned rate, and would need one row per employee edited
  at every rate change.
- **Filter**: `expense_phone.declined` -- at `save_line` AND in `_check` (a line read off a
  receipt is never saved). Carrier names or cell-phone words; internet, chargers, cases and
  repairs are not declined. Lines dated before Oct 2026 stand (on production: Alay's
  approved T-Mobile $50 for Aug; Joseph's two draft Verizon $103.46 for Jul/Aug).
- Guardrail `scripts/expense_phone_check.py` (32; `--inject=double|nofilter|everyone`).
