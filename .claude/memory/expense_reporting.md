# Employee expense reporting — design (Oct 2 2026; PHASE 1 BUILT, not deployed)

**Phase 1 is on branch `feat/expense-reports`** (off `feat/section-access`, which it
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

Answered Oct 2 2026: approvers -> CFO; CEO or President as backup for any report;
access stays the Admin's, via the existing section checkboxes; receipts upload
exactly as lease documents do.

1. **FOR ACCOUNTING** — the deal rule (pipeline -> Deal Cost Receivable, owned -> interco, Operations ->
   expense) — accounting to confirm, and the source of "owned" vs "pipeline".
2. Always reimbursed via TriNet payroll (`MR20000001`)? Batch per payroll date?
3. One approval step, or manager then accounting manager? Any thresholds or policy
   limits (per-meal, alcohol), or out of scope?
4. "FK - Benefits" $3,960.60 is a manual add in the coding workbook's totals — part of
   this process or outside it?

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
- **Categories are every MR5* account of TYPE I, as accounting asked** — which on
  real data includes Payroll, Interest, Depreciation, Management Fee and Professional
  Fees (52 accounts). Their template lists 15. Ask whether to narrow.
- **The deals table carries some deals under two codes with one name** (Adirondack
  RV Park = PADIRON and P0000064; City West; Orange Grove). Repeated names show the
  code. Which code accounting books to is a phase 3 question.
- A line is a date or a date range; a split must foot to the line (percent entry is
  converted to amounts once, the last row taking the remainder); receipt Y/N with a
  reason required for N, exactly as accounting's template — phase 2 replaces Y with
  the attached file.
- Verified in the running app (local): report opened, a line saved and totalled,
  submit disabled with the reason, setup tab rendered, draft deleted.
