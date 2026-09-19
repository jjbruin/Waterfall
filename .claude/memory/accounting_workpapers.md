# Accounting Workpapers & the Statement Engine

Built Sep 14–15 2026. The workpaper engine itself is unchanged since `v455`; the
ACCESS MODEL below is current to `v504`. Replicates the quarterly workpaper package
accounting produces by hand in Spreadsheet Server (the PPI Eastchase 06.30.2026
workbook was the specimen), sourced from MRI directly instead.

Sidebar: **Accounting → Workpaper Packages** (`/workpapers`,
`vue_app/src/views/WorkpapersView.vue`). 25 routes in `flask_app/api/workpapers.py`.

**The Accounting section now has three screens**, and the access model below covers
all of them: Workpaper Packages, **Treasury** (`treasury.md`), and **GL / IA Query**
(`v504`, `/gl-ia-query`) — the CFO's two Spreadsheet Server queries with his filters,
reading the app's imported `gl_detail` and `ia_transactions`. The query tool is
READ-ONLY and its reads are open to any signed-in user like the rest of the section;
that it is a bulk export of entity GL rather than one entity's statement is raised in
`open_items.md` §9.6.

---

## The shape of it

**One engine, many entities.** The statements are NOT a workpaper feature that
happens to draw numbers. `statement_service.py` builds Balance Sheet, Income
Statement, Members' Capital, Cash Flow and Schedule of Investments for ANY
entity and period; the package is one caller. A figure in a downloaded workbook
therefore cannot differ from the one an auditor is shown anywhere else — there
is no second implementation to drift.

**The population is MRI's, not ours.** `entity_groups` (ENTITYGRPD) with
`ENTGRPID = 'REP'` defines which entities need a package. Creating a close cycle
generates one package per REP entity; `sync_packages()` adds any that appear
later.

**The statements are the product; the checklist is how you get there.** They sit
at the TOP of a package with their tie-outs visible, each carrying a chip saying
whether the system's own check passed. Clicking a step loads exactly what that
step asserts — its guidance, the checks, the accounts or figures behind it, and
the exhibit slots it expects. The step to evidence mapping is accounting
knowledge and lives on the server (`workpaper_workbench.py`), not in the Vue.

---

## Data

Five MRI queries feed this, all added to `QUERY_REGISTRY` and the Data Explorer:

| Query | Table | What |
|---|---|---|
| `MRI_Entities` | `entities` | ENTITY master |
| `MRI_Entity_GroupID` | `entity_groups` | ENTITYGRPD; `ENTGRPID='REP'` = needs a package |
| `MRI_GL_Accounts` | `gl_accounts` | GACC — the ENTITY chart, **not** the property COA |
| `MRI_IA_Transactions` | `ia_transactions` | IA_Contribution/Distribution/NonCashTrans, unfiltered |
| `MRI_GL_Detail` | `gl_detail` | JOURNAL (open) + GHIS (closed, incl. balance-forward), period >= 202401 |

`MRI_GL_Detail` is deliberately **LAST** in the registry. `refresh_all()` walks the
dict in order and holds each result in a DataFrame before writing; GHIS is full
company history on a 1 CPU / 2GB container. A killed worker is not an exception
the per-query try/except can catch, so everything cheap commits first.

All five `.sql` files use `UNION ALL`, never `UNION` — Spreadsheet Server's own
queries use `UNION`, which dedupes, and the GL is a **journal**: one key
legitimately carries many rows that consumers SUM.

### The balance model

`GACC.TYPE`: `B` balance sheet (194), `C` cash (87), `I` income (269), `L`/`M`
roll-up headers that are **not** statement lines. Basis codes A/B/C/T.

    opening  = BALFOR 'B' row at period YYYY01
    YTD      = sum of BALFOR 'N' rows
    closing  = opening + YTD

---

## Statement mapping

`fs_line_seed.py` holds **accounting's own vocabulary**, lifted from the example
workbook's FS Tagging column — not invented here:

- `ACCOUNT_LINE` — 192 accounts to their caption
- `LINE_SECTION` — 56 captions to their section
- `LINE_ORDER` — 75 ranked captions, most-liquid-first (cash 10, restricted cash
  11, receivables 20s, due-from 30s, prepaid 40s, investment 50s, other assets
  90; liabilities 110–141; capital 210–220; income 310–390; expenses 410–490)

`wp_fs_map` is the live per-account mapping, seeded from `consolidated_mapping()`
and editable. **Accounts are never guessed onto a statement**: an account with no
GACC row, or a type outside B/C/I, is reported as `untyped`; one whose mapping
names a section that does not belong to the statement its type implies is
reported as a `conflict`. An account visibly missing from both statements is
better than one silently on the wrong one.

### Three corrections worth not repeating

**The period result belongs in Members' Capital.** Income accounts close to
equity at YEAR END, so at any date before that the equity accounts hold opening
capital plus capital movements and NOT the year's profit. Every entity came out
of balance by exactly its net income until `build()` carried the income
statement's own total into the capital section. PPIECH then landed on
33,317,764.46, matching the example workbook's rollforward.

**A line facing the wrong way still balances.** A negative asset and a positive
liability net identically, so the tie-out cannot catch a misclassification —
only a reader can, and only if they notice a minus sign in a column of
positives. `balance_sheet.sign_anomalies` reports them. See `open_items.md`
§6.1 for the live case (MR22000002).

**Membership interest comes from commitments, not `relationships`.**
`relationships.OwnershipPct` said PPIECH owns 100% of EASTCH. It does not —
29,390,000 of 44,085,000 = 66.67%, two thirds from PPI and one third from the
operating partner. The engine derives from commitment amounts, shows both, and
flags disagreement.

### Presentation (`v453`)

`line_sort_key()` orders by `LINE_ORDER`; an unranked caption sorts after the
ranked ones rather than displacing them. `is_dormant()` marks a line with NO
balance AND NO movement — the screen and the workbook leave those off and state
how many they suppressed. A zero line that HAD movement is kept: it went out and
came back, and hiding it would make the statement disagree with the trial
balance behind it.

**The engine only flags; it never drops.** The API returns every line.
Suppression is the consumer's call, so a tie-out cannot change because of how a
page is printed. Guardrail: `scripts/statement_presentation_check.py`.

---

## Workflow

`wp_cycles` to `wp_cycle_steps` (CFO deadlines) to `wp_packages` to
`wp_package_steps`, plus `wp_exhibits`, `wp_events`, `wp_roles`, `wp_fs_map`.

**12 steps**: `tb_load, gl_review, cash_rec, intercompany, accruals, investments,
capital_activity, exhibits, fs_draft, preparer_signoff, manager_review,
cfo_approval`.

**8 exhibit slots**: `valuation_support, cash_support, capital_rec,
cap_call_support, tax_pricing, fee_allocation, org_chart, other`.

**State chain**: `not_started` to `in_progress` to `submitted` to
`manager_approved` to `cfo_approved`, plus `returned` (note required).

### Deadlines (`v454`, relaxed `v455`)

`validate_due_date()` — **reject what cannot be true, warn what is merely odd.**

- **Refused**: not a date; earlier than the period BEGAN; unknown cycle
- **Warned but saved**: more than a year after period end; out of sequence
  against deadlines already set
- **Allowed**: clearing, always; the period end; the period start; equal dates
  on two steps

The bound is the period START, not the end — a pre-close prep step (bank
statements requested) is legitimately due while the quarter is still running.
`period_start()` derives it: the previous cycle's `period_end` is authoritative
where one exists, else a quarter end starts that quarter, another month end
starts that month, anything else falls back to a year.

Prompted by a real `2020-01-01` sitting on a period ended 2026-06-30, which
rendered that step overdue in red on every package from the moment it was typed.
**Validation is write-time only** — a deadline typed before the rule existed
stays stored (correctable through the UI, since clearing is always allowed).
Guardrail: `scripts/workpaper_deadline_check.py`, which NOTEs any it finds.

A pre-close deadline will fire the out-of-sequence warning, because all 12
current steps are post-close work. If accounting adds a genuine prep step it
belongs first in `STEP_TEMPLATE` and the warning stops.

---

## The download

`workpaper_excel.py` builds a 17-tab workbook: Cover, Index, the five
statements, Trial Balance, GL Detail, Account Summary, Investor/Investment
Detail, IA Rollforward, Commitments, Exhibits, exhibit tabs, Sign-off.

**Exhibits are PLACED, not attached** — an auditor opens one file, not a workbook
plus a folder. `.xlsx`/`.csv` are copied in as tabs, images embedded on their own
tab. A PDF cannot be inlined by openpyxl, so it is listed on the Exhibits tab
with its name, size and uploader and shipped alongside rather than lost.

Every tab names the table, period and basis it came from. A workpaper whose
provenance is "the app produced it" is not reviewable.

---

## Who may edit any of this (Sep 17 2026)

**Two gates, both MEMBERSHIP checks, both in `flask_app/auth/routes.py`.**

| | Who |
|---|---|
| `ACCOUNTING_ROLES` — edit anything in the section | admin, cfo, accounting_manager, accountant |
| `CLOSE_PLAN_ROLES` — the plan of the close | admin, cfo |
| read | everyone signed in |

`CLOSE_PLAN_ROLES` covers **when the close opens, when each thing is due, and
what order entities are worked in**: `POST /cycles`, `PUT /cycles/<id>/steps`,
`PUT /packages/<id>/schedule/target`, `PUT /packages/<id>/schedule/order`,
`POST /schedule/renumber`, `POST /schedule/carry-forward`. The team records what
it has DONE against that plan — syncing entities, naming a preparer, setting a
property, filling properties in bulk, and every sign-off.

Renumber and carry-forward are in the narrower gate **because they write
`sort_order`**. A rule covering the order cell but not the two buttons that
rewrite the same column is defeated by clicking a different button.

**WHY `roles_exactly` AND NOT `role_required`.** `role_required` compares
LEVELS, and `analyst`, `accountant`, `accounting_manager` and `cfo` are ALL
level 1 — so any level-based gate naming one of them admits all four. Jim's own
day-to-day login is an analyst one and he wants it read-only here (Sep 17 2026),
which levels cannot express. `roles_exactly` checks the name and fails closed on
an unknown one.

**THE SCREEN AND THE SERVER MUST AGREE.** `v480` gated the screen on `admin`
while the API would have accepted the CFO's writes, so the buttons simply were
not rendered — indistinguishable, to the person using it, from having no access.
Then the fix went one role too far: `['admin', 'cfo']` locked out the
accountants who prepare the close. Both views now read `auth.canEditAccounting`
/ `auth.canSetClosePlan` from the store, and the guardrail compares the Vue
lists to the Python ones **by name**.

**`scripts/accounting_access_check.py` (54) ENUMERATES THE SECTION'S ROUTES FROM
THE APP** and calls each as each role. The check it replaced grepped for a
decorator's exact text, which proves a string is present and is blind to a route
that never had one — that blindness was hiding **six writes with no role check
at all**, including exhibit deletion and tracker sign-off, reachable by any
signed-in user including viewers. Every narrowing is asserted in BOTH
directions, because a rule tested only in the refusing direction is satisfied by
locking everyone out, which here would stop the close.

## Not done yet

See `open_items.md` §6. In short: the MR22000002 tagging question is with
accounting; no close cycle, `wp_roles` assignment or step owners exist in
production yet; and `MC_TYPENAME_ROW` (members' capital row routing) wants
accounting's eye before the first real package goes out.
