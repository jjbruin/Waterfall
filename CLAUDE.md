# Waterfall XIRR - Multi-Layer Waterfall Model

## Shared Memory

Shared project memory files are in `.claude/memory/`. Read `MEMORY.md` there at the start of every conversation for project context, conventions, and architecture notes. Update these files as you work — they are committed to git and shared across all developers.

**The live work queue is `.claude/memory/open_items.md`** — what is still open, with evidence and an owner. Run `/open-items` to triage it. Close items there as they ship rather than letting them drift back into session narratives.

**Date-stamp anything describing work in flight, and delete it when the work ships.**
A section that says a branch is unmerged reads as authoritative for as long as it sits
here, and nothing distinguishes it from a current one. On Sep 11 2026 this file still
carried a 69-line section declaring `feat/onepager-chart-window` "not merged, not
deployed" — it had been on main for over a month (`window_end_quarter` at
`financials_service.py:177`). Shipped work belongs in the deploy history, not in
standing instructions.

## Project Overview

A Flask + Vue financial modeling application for calculating investment waterfalls, XIRR, and related performance metrics for real estate investments. The application supports multi-layer distribution waterfalls with preferred returns, capital accounts, and investor-level tracking.

## Tech Stack

- **Python 3.x** with virtual environment (`.venv/`)
- **Flask** - REST API backend (`flask_app/`)
- **Vue 3 + Vite** - Modern frontend (`vue_app/`)
- **pandas/numpy** - Data manipulation
- **scipy** - XIRR/NPV calculations (Newton-Raphson primary, Brent's method fallback)
- **ECharts** - Interactive charts (Vue)
- **PostgreSQL** - Azure-hosted database (local dev: SQLite via `waterfall.db`)
- **SQLAlchemy** - Database abstraction (`flask_app/db.py`), switches via `DATABASE_URL` env var
- **JWT** - Authentication (Flask + Vue)
- **Docker** - Multi-stage build (Vue → Python 3.12-slim + Gunicorn)
- **Azure Container Apps** - Production hosting

## Project Structure

```
waterfall-xirr/
├── config.py                 # Constants, account classifications, rates, dynamic defaults
├── compute.py                # Deal computation logic (core engine)
├── one_pager.py              # One Pager data logic (general info, cap stack, property perf, PE metrics, comments)
├── models.py                 # Data classes (InvestorState, Loan)
├── waterfall.py              # Waterfall calculation engine
├── metrics.py                # XIRR, XNPV, ROE, MOIC calculations
├── loaders.py                # Data loading from database/CSV
├── database.py               # Database management (SQLite + PostgreSQL), migrations, CSV import/export
├── loans.py                  # Debt service modeling
├── planned_loans.py          # Future loan projections
├── capital_calls.py          # Capital call handling
├── cash_management.py        # Cash flow management
├── consolidation.py          # Sub-portfolio aggregation
├── portfolio.py              # Fund/portfolio aggregation
├── reporting.py              # Annual aggregation tables, formatting utilities
├── ownership_tree.py         # Investor ownership structures
├── utils.py                  # Helper utilities
├── argus_parser.py           # Stateless Argus Enterprise Excel parser (COA mapping, forecast conversion)
├── cashflow_parser.py        # Generic Excel/CSV parser for partner cash flow models (auto-detect columns, annual→monthly)
├── Dockerfile                # Multi-stage Docker build (Vue + Flask + Gunicorn)
├── launch_app.bat            # Desktop launcher (opens Azure app in browser)
├── waterfall_xirr.ico        # Custom app icon for desktop shortcut
├── waterfall.db              # SQLite database for local dev (not in git, >100MB)
│
├── flask_app/                # Flask REST API backend
│   ├── __init__.py           # App factory (create_app), SPA serving with cache headers
│   ├── run.py                # Dev server entry point
│   ├── db.py                 # SQLAlchemy engine management (SQLite/PostgreSQL)
│   ├── config.py             # Flask configuration (DATABASE_URL, dynamic defaults, ACTUALS_THROUGH)
│   ├── extensions.py         # Flask extensions
│   ├── serializers.py        # JSON serialization helpers (NumpyEncoder, safe_json)
│   ├── auth/                 # JWT authentication (login, SSO config, password reset, welcome emails)
│   │   ├── routes.py         # Auth routes (login, users, desktop shortcut installer)
│   │   └── email_utils.py    # SendGrid email sending (welcome emails, password reset)
│   ├── api/                  # API blueprints
│   │   ├── dashboard.py      # Dashboard endpoints (KPIs, charts, SSE init-stream)
│   │   ├── data.py           # Data endpoints (deals, upload-import, export, config)
│   │   ├── deals.py          # Deal analysis endpoints + Excel downloads
│   │   ├── financials.py     # Property Financials + One Pager endpoints
│   │   ├── reports.py        # Report generation endpoints
│   │   ├── reviews.py        # Review workflow endpoints (status, submit, approve, return, tracking, roles)
│   │   ├── feedback.py       # Feedback & request tracking endpoints (submit, list, messages, email, webhook)
│   │   ├── lease_review.py   # Lease review & risk analysis endpoints (DD workflow, document upload, field resolution)
│   │   ├── prospects.py      # Pipeline prospect CRUD (deals, properties, entities, investors, assumptions)
│   │   ├── argus.py          # Argus Enterprise import, projection management, COA mapping, forecast preview
│   │   └── ...               # Additional route blueprints
│   └── services/             # Business logic (reuses compute.py, database.py, etc.)
│       ├── dashboard_service.py  # KPI calculations, NOI pipeline, chart data
│       ├── data_service.py       # Data loading and caching
│       ├── data_adapters.py      # Pluggable data source adapters (DB or MRI API)
│       ├── compute_service.py    # Deal computation cache, ROE/MOIC audit builders, Excel generators
│       ├── review_service.py     # Review workflow business logic (approval pipeline)
│       ├── financials_service.py # Property Financials + One Pager data aggregation
│       ├── feedback_service.py   # Feedback & request tracking (CRUD, email, export)
│       ├── reports_service.py    # Report builders (projected returns, ROE summary, pref balance detail)
│       ├── statement_service.py  # THE statement engine — BS, IS, Members' Capital, Cash Flow, SOI for any entity
│       ├── fs_line_seed.py       # Accounting's own FS vocabulary: 192 accounts, 56 captions, 75 ranked
│       ├── workpaper_service.py  # Close cycles, packages, steps, exhibits, approvals, deadlines
│       ├── workpaper_data.py     # Trial balance, GL detail, IA/commitment rollforwards
│       ├── workpaper_workbench.py # Step → evidence mapping (accounting knowledge, server-side)
│       ├── workpaper_excel.py    # 17-tab downloadable package; exhibits placed, not attached
│       ├── lease_review_service.py  # Lease review DD workflow, document upload, extraction, field resolution
│       ├── prospect_service.py      # Pipeline prospect CRUD, lease review creation, deal evaluation
│       ├── argus_service.py         # Argus Enterprise import, projection CRUD, forecast generation, NB→AM migration
│       └── ...
│
├── scripts/                  # Azure migration and setup scripts
│   ├── migrate_to_postgres.py    # Bulk SQLite → PostgreSQL migration
│   ├── fix_tables.py             # Fix tables with type mismatches
│   └── azure-complete-setup.sh   # Reference doc of provisioned infrastructure
│
└── vue_app/                  # Vue 3 + Vite frontend
    ├── src/
    │   ├── api/client.ts     # Axios instance with JWT interceptors
    │   ├── stores/           # Pinia stores (auth, data, dashboard, deals)
    │   ├── views/            # Page components (DashboardView, DealAnalysisView, OnePagerView, DataExplorerView, ForgotPasswordView, ResetPasswordView, etc.)
    │   └── components/       # Shared components (KpiCard, DataTable, ReviewPanel, AppSidebar)
    ├── vite.config.ts        # Vite config (proxies /api to Flask)
    └── package.json
```

## Documentation

- **DOCUMENTATION.md** - Complete project documentation (setup, data files, concepts, troubleshooting)
- **waterfall_setup_rules.txt** - Waterfall step configuration guide for deal modeling team
- **typename_rules.txt** - Capital pool routing rules based on Typename field
- **.claude/memory/accounting_workpapers.md** - The workpaper packages + statement engine: data, mapping, workflow, deadlines, the download (Sep 15 2026)
- **.claude/memory/app_reference.md** - What each app tab displays + AI Assistant tools/endpoints (split out of this file Sep 11 2026)
- **.claude/memory/treasury.md** - The bank side of the close: PNC import, the three-way tie, the matcher, what `current_available` cannot say (Sep 17 2026)

## Running the Application

### Production (Azure)
**Desktop shortcut**: Double-click **"Waterfall XIRR"** on the desktop — opens the Azure app in the browser.
- **URL**: https://app-waterfall-dev-v2.icyplant-026fb2db.eastus.azurecontainerapps.io
- Login: real user accounts. `admin` / `admin` is the LOCAL dev seed only — it
  returns 401 against Azure (verified Sep 2 2026). Do not script against it.

### Deploying Changes

**BEFORE DEPLOYING ANY COMMIT: review it for symptom repair, and tell Jim first.**
(Jim's standing instruction, Sep 1 2026, after `50695d9` shipped an override that
zeroed correct data on 12 deals.)

Fix problems, not symptoms. Read the diff and ask what the commit is actually doing:

- Does it **suppress, zero, blank, force or special-case** a value rather than correct
  the computation that produced it? A value overridden downstream is still computed
  wrong upstream, and every other consumer keeps reading the wrong one.
- Is the **trigger a proxy** for the thing it claims to detect? (`50695d9` keyed off
  the presence of a `2015-12-31` row in MRI's export — an export artifact — to infer
  "this deal has no baseline".)
- Does the stated rationale **hold for every deal it touches**? Enumerate the affected
  rows from live data and check. `50695d9`'s rationale held for 2 of the 12 it acted on.
- Does it use a **sentinel value** where the answer is "unknown"? Return `None`, never
  `0` — a `0` that means "no data" is indistinguishable from a real zero to every
  consumer, and only renders as a dash because `fmtMil()` happens to treat `0` as `—`.
- Is it a **per-deal hardcode** (a vcode in a constant)? Those are always symptom
  repairs, however well documented.

Verifying that a commit does what its message says is NOT the same as verifying the
premise. Both are required. When a commit is a symptom repair — even a well-reasoned,
well-documented one — say so to Jim, with the affected deals and figures, and get his
call BEFORE building the image. Deploy is not the place to discover the question.

#### Pre-flight — run this BEFORE `az acr build`, every time

Step 0 below is not "check you are on the right commit". It is **establish what is
actually shipping, and review all of it.** These four steps are the deploy; the `az`
commands are just what you type afterwards.

```bash
# P1. What is live RIGHT NOW? The image tag is the SHA, so this answers it.
az containerapp revision list -g rg-waterfall-dev -n app-waterfall-dev-v2 \
  --query "[?properties.active].{name:name,image:properties.template.containers[0].image}" -o table

# P2. What would ship? THE SPAN IS AGAINST THE LIVE IMAGE, NOT AGAINST LOCAL HEAD.
git fetch origin
git log --oneline <live-sha>..<target-sha>      # <live-sha> from P1

# P3. The tree must be clean and at the target — ACR uploads the WORKING TREE, not a git ref.
git rev-parse --short HEAD
git status --porcelain                          # must be empty

# P4. Read the diff of EVERY commit P2 listed, against the checklist above.
git show <sha>                                  # for each one
```

**P2 is the step that gets skipped, and skipping it is how unreviewed code ships.**
On `v429` (Sep 11 2026) the request was "deploy 33a4bf5". Local main was two commits
behind origin, and the live image was five behind that — so **seven commits shipped, not
one**, including three of Charlene's investor-facing One Pager print commits that the
previous handoff had explicitly flagged as needing this review first. Only the two-commit
local span was reviewed. Nothing broke, but nobody had looked.

A commit is not exempt because someone else wrote it, because it is "only" a docs or
script commit, or because it was already on main. If P2 lists it, it ships, and you own
reviewing it.

**If P2 lists anything you did not expect, stop and reconcile before building.** That is
the signal, not a formality.

Then: symptom repair found → tell Jim, with affected deals and figures, and get his call.
Clean → build.

All deploys use Azure CLI (GitHub Actions secrets are not configured). `.github/workflows/deploy.yml`
exists but is **deliberately not wired up** — it triggers on push to main, which would ship
code without this pre-flight, and it deploys `:latest`, which is untraceable. Do not enable it
without changing both.

**Who can deploy**: Jim, and Charlene (`cbui@peaceablestreet.com`) as of Sep 11 2026 —
Contributor scoped to the registry `acrwaterfalldev` and the container app
`app-waterfall-dev-v2` only, not the resource group. Note `AcrPush` is NOT sufficient for
`az acr build`: it grants only `pull/read` and `push/write`, while the build needs
`scheduleRun/action` and `listBuildSourceUploadUrl/action`.

**Tag every image with the commit SHA it was built from, and deploy that tag — never `:latest`.**
`:latest` is mutable, so a revision pointing at it cannot be traced back to a commit once the
next build overwrites the tag. Deploy the SHA tag and the running revision names its own source.

```bash
# 0. Pre-flight P1-P4 above is done and clean. Do not start here.
#    (P3 already proved HEAD is the target and the tree is clean — ACR uploads the
#    WORKING TREE, not a git ref, so an unclean tree ships uncommitted work.)

# 1. Build in ACR, tagged with the commit SHA (--no-logs avoids a unicode crash)
SHA=$(git rev-parse --short HEAD)
az acr build --registry acrwaterfalldev -g rg-waterfall-dev --image waterfall-xirr:$SHA --image waterfall-xirr:latest --no-logs .

# 2. Lock the SHA tag so a later build cannot overwrite it (delete stays enabled for cleanup)
az acr repository update -n acrwaterfalldev --image waterfall-xirr:$SHA --write-enabled false

# 3. Deploy the SHA tag (incrementing suffix forces a new revision)
az containerapp update -g rg-waterfall-dev -n app-waterfall-dev-v2 --image acrwaterfalldev.azurecr.io/waterfall-xirr:$SHA --revision-suffix v350
```

Pick the revision suffix by bumping the current one — reusing a suffix is rejected. This same
query answers "what commit is live?", since the image tag is the SHA:
```bash
az containerapp revision list -g rg-waterfall-dev -n app-waterfall-dev-v2 --query "[?properties.active].{name:name,image:properties.template.containers[0].image}" -o table
```

**Notes**:
- ACR build agent has transient failures (5-second runs) — retry if it fails. Confirm the run actually built rather than failing fast: `az acr task show-run --registry acrwaterfalldev --run-id <id> --query "{status:status,start:startTime,finish:finishTime}" -o tsv`
- Use `--no-logs` to avoid Azure CLI unicode crash (`✓` character).
- To pin an already-deployed `:latest` revision after the fact, retag its digest without rebuilding and redeploy that tag — same digest, so the content is provably identical: `az acr import -n acrwaterfalldev --source acrwaterfalldev.azurecr.io/waterfall-xirr@sha256:<digest> --image waterfall-xirr:<sha>`
- **Deploy history (SHA-pinned)** — newest first; `vNNN` is the revision suffix and the
  backticked SHA is the commit its image was built from, so the running revision names its
  own source. Entries marked † have a full post-mortem archived verbatim in
  `.claude/memory/deploy_history.md`. **Read that before assuming a revision shipped what
  its SHA suggests** — several did not (`v424` was a merge, not the commit that was asked
  for; `v378` was superseded minutes later; `v418`/`v417` shipped only part of a branch).

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

### Local Development
```bash
# Activate virtual environment
.venv\Scripts\activate

# Run Flask API backend
python -m flask_app.run          # API on http://localhost:5000

# Run Vue frontend (separate terminal)
cd vue_app && npm run dev        # Frontend on http://localhost:5173
# Default login: admin / admin
```

### Azure Infrastructure
- **Container App**: app-waterfall-dev-v2 (1 CPU, 2GB RAM, 2 Gunicorn workers)
- **PostgreSQL**: psql-waterfall-dev.postgres.database.azure.com (B1ms, v16)
- **Container Registry**: acrwaterfalldev.azurecr.io
- **Resource Group**: rg-waterfall-dev (eastus)
- View logs: `az containerapp logs show -g rg-waterfall-dev -n app-waterfall-dev-v2 --type console --tail 50`

### Caching
- `index.html` served with `Cache-Control: no-cache` — browser always checks for new version on deploy
- Hashed assets (`/assets/*`) cached for 1 year with `immutable` — Vite generates new hashes on each build

## Key Concepts

### Acquisition Date
- **Derived from accounting feed**: `min(EffectiveDate)` per `InvestmentID` from the accounting table, mapped to vcode via investment map
- **Enriched at load time**: `data_service.py:load_all()` overwrites `Acquisition_Date` in `inv` DataFrame after loading both `inv` and `acct`. Also re-applied in `refresh_table()` when the deals table is refreshed.
- **Date parsing**: Raw `EffectiveDate` in the accounting adapter is strings (e.g., `"11/21/2016 0:00"`). Must parse via `pd.to_datetime()` before `groupby().min()` — string comparison is alphabetical, not chronological.
- **Rationale**: The MRI `Acquisition_Date` field may not reflect the true closing date. The earliest accounting activity (e.g., acquisition fee collected at closing) is the authoritative date. For development deals, first funding may occur months after acquisition (e.g., 3rd Ave & Indian School: acquisition fee 11/21/2016, first funding 5/15/2017).
- **Fallback**: If a deal has no accounting activity, the original MRI `Acquisition_Date` is preserved
- **Consumers**: Deal Analysis metadata, Sold Portfolio summary, One Pager general info — all use `inv["Acquisition_Date"]` automatically

### Sale Date & Anticipated Exit
- **Sale date priority**: (1) Sale date override from UI, (2) `event_dates` projected disposition closing (`vEventType='Disposition', vEvent='Closing', vDateType='Projected'`), (3) horizon end / max loan maturity. The `Sale_Date` column in the deals table is no longer consulted.
- **Underwritten Exit**: From MRI `event_dates` table (`vEventType='Asset Management', vEvent='U/W Exit', vDateType='Actual'`). Latest `dtEvent` wins. Function: `get_underwritten_exit()` in `one_pager.py`. Previously sourced from `inv.Sale_Date` — now sourced directly from `event_dates`.
- **Current Anticipated Exit**: From `event_dates` table (`vEventType='Disposition', vEvent='Closing', vDateType='Projected'`). Latest `dtEvent` wins when multiple rows exist. Used on One Pager and as default sale date for Deal Analysis projections. Function: `get_current_anticipated_exit()` in `one_pager.py`.
- **Shared helper**: Both functions use `_lookup_event_date()` in `one_pager.py` — generic event_dates query by vCode + vEventType + vEvent + vDateType.
- **MRI refresh safety**: `MRI_COLUMNS` in `mri_service.py` lists only columns that MRI refresh may overwrite. `Sale_Date`, `Sale_Status`, `InvestmentID`, and `Portfolio_Name` are explicitly excluded — they are preserved during upsert.
- **Sale date override**: Separate `sale_overrides` table (see Sale Overrides section) — completely independent of MRI refresh.
- **Event dates refresh**: `MRI_Event_Dates` and `MRI_Inspection` queries are included in the "Refresh All Data from MRI" process (`QUERY_REGISTRY` in `mri_service.py`).

### Waterfall Types
- **CF Waterfall**: Operating cash distributions (does NOT reduce capital outstanding)
- **Capital Waterfall**: Refi/sale proceeds (DOES reduce capital outstanding)

### Preferred Returns
- Daily accrual using Act/365 Fixed day count
- Compounds annually on 12/31 (with 45-day grace period)
- Tracked per investor via `InvestorState`

### Capital Calls
- **Data pipeline**: app entry → `capital_calls` table → `load_capital_calls()` → `build_capital_call_schedule()` → `apply_capital_calls_to_states()`
- **`capital_calls` IS IN `PROTECTED_TABLES`** (Sep 10 2026, Jim's instruction). The CSV import runs `to_sql(if_exists="replace")`, which DROPS the table, so a single `MRI_Capital_Calls.csv` upload silently destroyed every call typed into Deal Analysis. **Capital calls are now app-entered only** — added, edited and deleted per row via `/api/deals/<vcode>/raw-capital-calls`. Cost is only the bulk upload: the table is NOT in `mri_service.QUERY_REGISTRY`, so "Refresh All Data from MRI" never touched it and no automated feed is interrupted. The upload UI badges a protected file "locked" and excludes it from the importable count, so a skipped import announces itself. **Known consequence**: rows with a blank `Vcode` (5,127 of 5,130 locally — empty trailing rows from a past CSV) are invisible to the vcode-filtered GET and the replace that used to clear them is now blocked, so they cannot be cleaned from the UI. Harmless to every computation (`load_capital_calls` drops them via `dropna`), but permanent; purging them is a separate deliberate act. Guardrail: `scripts/capital_call_crud_check.py` asserts all four import entry points refuse the table AND that the rows survive, plus that app add/edit/delete still work.
- **Date handling**: Uses `pd.to_datetime(format='mixed', dayfirst=False)` to handle both CSV US dates ("6/30/2026") and HTML ISO dates ("2026-06-01")
- **Null filtering**: `dropna(subset=['deal_name', 'call_date', 'amount'])` removes empty rows from CSV imports
- **Column mapping**: Supports `entityid` → `deal_name`, `propcode` → `investor_id`, `calldate` → `call_date`
- **Table auto-creation**: `database.py` creates `capital_calls` table via `CREATE TABLE IF NOT EXISTS` in `create_additional_tables()`
- **CRUD endpoints**: `POST/PUT/DELETE /api/deals/<vcode>/capital-calls` in `deals.py`. All use `data_service.reload()` for full cache invalidation
- **Refi shortfall auto-clear**: After capital calls are applied, if total capital calls cover the refi shortfall (within $1 tolerance), `refi_capital_call_required` is set to False

### Tax Abatements
- **Account**: `TAX_ABATEMENT_ACCTS = {7070}` in `config.py` — stored in `forecast_feed` / `forecasts` table
- **Sign convention**: MRI stores abatements as negative (credit); `normalize_forecast_signs()` in `loaders.py` forces `base.abs()` (positive) since abatements increase cash flow
- **Annual Forecast**: Displayed as "Tax Abatement" row below NOI, included in FAD calculation. FAD adds back CapEx funded from cash reserves (`reporting.py`)
- **NPV at Sale**: Remaining abatement payments after sale date are discounted to PV at `TAX_ABATEMENT_DISCOUNT_RATE` (5%) and added to net sale proceeds (`compute.py`). Shown as "NPV (@5%) Tax Abatements" in Sale Proceeds Calculation (Debt Service section)
- **Below-the-line items**: Former "Excluded Accounts" renamed to "Other Below-the-Line" (`OTHER_EXCLUDED_ACCTS`): Interest Income (4050), Other Income/Expenses (5220, 5210, 5195, 7065), Partnership Expenses (5120, 5130), Depreciation & Amortization (5160, 5165), Extraordinary Expenses (5400). Other Revenue (4075) and Maintenance Flex (5092) are in NOI.
- **Conditional display**: Tax Abatement NPV line only appears in Sale Proceeds Calculation when deal has 7070 data
- **Comparability fix**: In U/W Projected IS, 7070 appears below NOI as a separate line item, but in actuals the abatement is netted into account 5090 (Real Estate Taxes). To ensure apples-to-apples comparison across all columns:
  - **One Pager** (`one_pager.py`): `calc_amounts()` folds 7070 into expenses (negative credit reduces expenses). `TAX_ABATEMENT: ['7070']` added to `IS_ACCOUNTS`. Separate adjustment for the pre-computed `at_close_noi_df` path.
  - **Property Financials** (`config.py`): `'7070'` added to `Real Estate Taxes` account list in `IS_ACCOUNTS['EXPENSES']`, so the income statement includes it alongside 5090 in all source comparisons.

### Paid-Off Loan Exclusion
- **Filter**: Loans with `vDateType = "Paid Off"` are excluded from all analysis at the data layer
- **Implementation**: `data_service.py:load_all()` filters `mri_loans_raw` after loading; same filter applied in `refresh_table()` for the `loans` table
- **Column detection**: Case-insensitive lookup (`c.lower() == "vdatetype"`) for robustness across CSV/MRI sources
- **Scope**: Affects all consumers — waterfall, capitalization, dashboard, debt service, assistant tools

### Balloon Loan Payoff at Sale
- **Detection**: Loan schedule's last row has `ending_balance < 1.0` (float tolerance), `principal > 0`, and prior row's `ending_balance > 0`
- **Forecast exclusion**: Balloon principal payments are excluded from forecast debt service rows (they are NOT operating expenses)
- **Sale proceeds**: Net sale proceeds deduct `total_loan_balance_at(sale_date) + balloon_total` — uses pre-balloon balance since balloon is paid at sale
- **Sale proceeds formula**: `max(0, value_net_selling_cost - loan_balances + tax_abatement_npv)`
- **Groupby**: Loan schedule grouped by `["vcode", "LoanID", "event_date"]` to distinguish loans with same dates
- **ISBS debt fallback**: When modeled loans aren't active at sale date (e.g. all loans paid off, or loan originates after sale), `compute.py` falls back to ISBS balance sheet debt (`get_isbs_debt_balance()`) for loan payoff amount. Also handles case where `loan_sched` is completely empty but ISBS shows outstanding debt.

### Sale Overrides
- **UI**: Contract Sale Price, Selling Costs (% or $), Sale Date — input row on Deal Analysis below capitalization table
- **Contract Sale Price**: Overrides the NOI/cap rate implied value when populated
- **Selling Costs**: Overrides the default 2% selling cost; supports percentage (`pct`) or fixed dollar (`fixed`) modes
- **Sale Date**: Overrides the default sale date (from `event_dates` projected disposition); threads through `compute_service.py` as `sale_date_override`
- **Database**: `sale_overrides` table (vcode PK, contract_sale_price, selling_cost_value, selling_cost_type, sale_date_override, updated_at, updated_by). In `PROTECTED_TABLES`.
- **Cache behavior**: Sale overrides do NOT create separate cache keys — `force=True` is set automatically when any override is present, overwriting the existing cache entry
- **Workflow**: Enter values → Recompute → optionally Save (persists to DB) or Clear. Saved overrides auto-load on deal select.
- **API**: `GET/PUT/DELETE /api/deals/<vcode>/sale-override`. POST `/compute` accepts overrides in body; falls back to saved overrides if not provided.
- **ISBS-anchored loan payoff**: Modeled amortization balance at sale date is scaled by `(ISBS_actual_debt / modeled_debt_at_anchor_date)` ratio. The ISBS Interim BS gives the real current outstanding; the scale factor corrects for differences between MRI origination amount and actual paydown. Diagnostic message shows anchor date, actual/modeled balances, and scale factor.

### Parcel Sales (Interim Sales)
Sales of part of a property before the final disposition. A deal can carry
several. Expandable box on Deal Analysis, directly above the Sale row.

- **Inputs per sale**: label, projected sale date, sale price, cost of sale (% or $), debt paydown allocated across mortgages, amount held in the CapEx reserve, distribution mode, and the revenue/expenses that leave with the parcel.
- **Database**: `parcel_sales` table (id, vcode, label, sale_date, sale_price, cost_of_sale_value/type, debt_application, capex_reserve_hold, distribution_mode, distribution_fixed, lost_revenue, lost_expense, notes, sort_order). The four structured columns are JSON blobs, following `prospect_assumptions`. In `PROTECTED_TABLES`.
- **Money flow** (`compute_economics()` in `parcel_sale_service.py`, the single definition the UI and engine share): price → less cost of sale → less debt paydown → less reserve hold → remainder distributed.
- **Reserve hold**: passed to `build_cash_flow_schedule_from_fad()` as `reserve_deposits`; joins the cash balance *before* CapEx is funded, then is spent by the ordinary reserve rules. Anything unspent returns at the final sale through the existing `get_sale_period_total_cash()` path — no separate logic. Note the reserve can also fund operating deficits that were previously unfunded, so a parcel sale may improve returns by more than its proceeds.
- **Debt paydown**: `Loan.curtailments` + the amortization builder. Convention is **re-amortize** — the payment is recalculated over the remaining term and the maturity date does not move, so debt service and DSCR improve from the paydown date. An interest-only loan simply drops its interest.
- **Distribution modes**: `waterfall` feeds the remainder into `cap_period_cash` as a Cap event at the parcel date (exact — `run_interleaved_waterfalls` processes it in date order); `pro_rata` splits on contributed capital; `fixed` uses per-partner amounts. All three are a **return of capital** — they reduce capital outstanding, so weighted average capital falls and ROE rises gradually rather than spiking.
- **Lost income**: `apply_parcel_income_loss()` spreads annual amounts across the months from the sale date, adjusting `mAmount_norm` only (the column every NOI/FAD/DSCR/terminal-value path reads). Revenue subtracts; expense adds back. Because the exit value is forward NOI / cap rate, **removing revenue automatically lowers the final sale price** — correct, and stated in diagnostics so it is not read as an error.
- **Tenant picker**: `get_deal_tenants()` reads the MRI roster and seeds Rental Income (4010); the figure stays editable because a point-in-time rent roll will not tie to a forecast carrying growth and rollover. Identical roster rows are collapsed — the roster returns every tenant twice and summing raw rows doubles the rent removed.
- **Cache**: parcel sales are loaded *inside* `get_cached_deal_result()` and fingerprinted into the cache key, so Deal Analysis, the reports and the assistant share one set of assumptions. The fingerprint covers only fields that change the projection, so a rename does not force a recompute.
- **Reporting**: four lines on the Annual Forecast below DSCR — Net Proceeds, Debt Paydown, To CapEx Reserve, Distributed — added only when the deal has a sale, and picked up by the Excel export automatically. `result['parcel_sales_applied']` is a JSON-safe audit record of what was applied.
- **Guards** (each reported rather than silent): a parcel date after the deal's sale date is ignored; a paydown at or before the actuals boundary is not modelled, since it is already in the reported balance and would otherwise corrupt the ISBS anchoring; a paydown allocated to a loan that is not modelled is not applied; a loan retired early by a paydown is not misread as a balloon and deducted again at sale; fixed amounts that miss the remainder are distributed as entered with the difference reported; an amount naming a partner not on the deal is not distributed.
- **All three modes process in date order**: `pro_rata`/`fixed` remainders are runner events (`build_manual_parcel_events()` + `manual_events` on `run_interleaved_waterfalls`) — pref accrues to the parcel date on the pre-cut balance, then the return of capital reduces the pools, so pref for every later period accrues on the reduced capital, same as the `waterfall` mode.
- **API**: `GET/POST /api/deals/<vcode>/parcel-sales`, `PUT/DELETE /api/deals/<vcode>/parcel-sales/<id>`, `GET /api/deals/<vcode>/parcel-sales/tenants`. Validation runs server-side on create and update; errors block the save, warnings do not.

### Valuation Budget Comparison — loading it, and levering it
The Budget Review tab's comparison is **Estimate | Budget | Valuation**. Two of those
columns are loaded from somebody else's spreadsheet, and the debt rows are ours.

- **One screen for both sources** (`LineMappingPanel.vue`, `line_mapping_service.py`,
  four endpoints under `/api/valuations/records/<id>/mapping/`). `source` is `budget`
  (partner's workbook → `isbs_budget_is_supplements` → Budget column) or `argus`
  (appraiser's download → COA overrides via `argus_service.update_coa_mapping` →
  Valuation column). Same job, same rules, same screen.
- **Category first, then an account within it.** The categories are the ~27 rows the
  comparison renders; a bare account number asks the analyst to translate from a
  169-item list into a row they cannot see. The account is still required — the
  supplement stores `vAccount` and NOI/FAD/DSCR/waterfall all read accounts — and
  defaults to the one that deal used most in the last 12 months.
- **The flip default is PER ACCOUNT, from the deal's own history**, never from the
  4xxx/5xxx prefix: 4030 Residential Vacancy and 4042 Loss to Lease are 4xxx stored
  POSITIVE, 5220 Other (Income) Expense is 5xxx stored NEGATIVE.
- **Unmapped lines never block.** Spreadsheets carry subtotals and skipping them is
  correct; `reconcile()` shows stated-vs-computed revenue, expense and NOI so the analyst
  can tell a skipped subtotal from a missed line. Anything announcing itself as a total
  is flagged and never pre-filled — "Total Capital Expenditures" matched the Argus
  keyword rules and would have double-counted capex.
- **Commit REPLACES, scoped to (vcode, the periods in THIS file)** — a budget is
  re-imported until final and appending would stack every revision.
- **Debt service is MODELED, Budget and Valuation columns only**
  (`valuation_debt_service.py`). An Argus download is unlevered, so those columns showed
  0 interest, 0 principal and a blank DSCR. Same strip-and-replace `compute.py` applies
  to the AM forecast. **The Estimate column is never substituted** — it means actuals,
  and its interest was actually paid. Levered only when an Argus forecast exists:
  modeled debt over a zero NOI turns a blank DSCR into a hard `0.00`, which reads as
  "cannot cover its debt" instead of "no forecast loaded".
- **Interest goes to 5190 here, 7030 in the AM forecast** — see `open_items.md` §5.8.
  Deliberate as of Sep 11 2026, not accidental, and still worth settling.

### Accounting Workpapers & the Statement Engine
**Full detail in `.claude/memory/accounting_workpapers.md`.** Live at `v455`.

- **ONE ENGINE, MANY ENTITIES.** `statement_service.py` builds Balance Sheet, Income
  Statement, Members' Capital, Cash Flow and Schedule of Investments for any entity and
  period. The workpaper package is one caller, not the owner — so a figure in a
  downloaded workbook cannot differ from the one shown anywhere else.
- **The population is MRI's**: `entity_groups` (ENTITYGRPD) with `ENTGRPID='REP'`.
- **Five MRI queries**: `MRI_Entities`, `MRI_Entity_GroupID`, `MRI_GL_Accounts`,
  `MRI_IA_Transactions`, `MRI_GL_Detail`. **`MRI_GL_Detail` is LAST in `QUERY_REGISTRY`
  on purpose** — unbounded GHIS on a 2GB container is the one that could kill the
  worker, and a killed process is not an exception the per-query try/except can catch.
  All five `.sql` files use `UNION ALL`, never `UNION`: the GL is a journal and one key
  legitimately carries many rows that consumers SUM.
- **Balance model**: `opening` = BALFOR 'B' at YYYY01, `YTD` = BALFOR 'N' rows,
  `closing` = opening + YTD. `GACC.TYPE` B/C/I are statement accounts; L/M are roll-up
  headers and are NOT lines.
- **Accounts are never guessed onto a statement.** No GACC row, or a type outside
  B/C/I → reported as `untyped`. A mapping naming a section that does not belong to the
  statement its type implies → reported as a `conflict`. Visibly missing beats silently
  wrong.
- **THE PERIOD RESULT BELONGS IN MEMBERS' CAPITAL.** Income closes to equity at YEAR
  END, so before then the equity accounts hold no profit and every entity came out of
  balance by exactly its net income. `build()` carries the income statement's own total
  across, so the two statements cannot disagree.
- **A line facing the wrong way still balances** — a negative asset and a positive
  liability net identically, so no tie-out can catch it. `balance_sheet.sign_anomalies`
  reports them. Live case in `open_items.md` §6.1.
- **The engine flags; it never drops.** `dormant` (no balance AND no movement) is
  returned on every line and the screen and workbook suppress them, saying how many. A
  zero line WITH movement is kept — hiding it would make the statement disagree with the
  trial balance behind it.
- **Deadlines: reject what cannot be true, warn what is merely odd.** Refused — not a
  date, or earlier than the period BEGAN (`period_start()` derives the bound; a
  pre-close prep step may be due inside the period). Warned but saved — over a year out,
  or out of sequence. Clearing is always allowed. Write-time only: a bad deadline typed
  before the rule stays stored.
- **Guardrails**: `scripts/statement_presentation_check.py`,
  `scripts/workpaper_deadline_check.py`.

### Who may edit the Accounting section
**Full detail in `.claude/memory/accounting_workpapers.md`.** Live at `v490`.

| Gate | Who | What |
|---|---|---|
| `ACCOUNTING_ROLES` | admin, cfo, accounting_manager, accountant | every write in `/api/workpapers` and `/api/treasury` |
| `CLOSE_PLAN_ROLES` | admin, cfo | when the close opens, when things are due, what order entities are worked in |
| — | everyone signed in | reads |

- **`roles_exactly`, NOT `role_required`.** `role_required` compares LEVELS and
  `analyst`, `accountant`, `accounting_manager` and `cfo` are ALL level 1 — so any
  level gate naming one admits all four, and no arrangement of names excludes
  analysts. Jim's day-to-day login is an analyst one and is read-only here. The
  rest of the app (104 endpoints) still uses `role_required`; if a rule ever needs
  to separate two level-1 roles elsewhere, it needs `roles_exactly` too.
- **`CLOSE_PLAN_ROLES` includes renumber and carry-forward because they write
  `sort_order`.** A rule covering the order cell but not the buttons that rewrite
  the same column is defeated by clicking a different button. `Fill properties`
  is deliberately NOT included — it writes only the Property column.
- **The screen must agree with the server, and has been wrong BOTH ways.** `v480`
  gated the screen on `admin` while the API would have taken the CFO's writes, so
  the buttons were simply not rendered; the fix then went one role too far and
  locked out the accountants. Views read `auth.canEditAccounting` /
  `auth.canSetClosePlan`; the guardrail compares the Vue lists to the Python ones
  by name.
- **`scripts/accounting_access_check.py` (54) ENUMERATES ROUTES FROM THE APP** and
  calls each as each role, so a new endpoint is covered the day it is written. The
  check it replaced grepped for a decorator's text and was therefore blind to
  **six writes that had no gate at all** — including exhibit DELETE and tracker
  sign-off, reachable by any signed-in user. Every narrowing is asserted in BOTH
  directions: a rule tested only in the refusing direction is satisfied by
  locking everyone out.

### Treasury — the bank side of the close
**Full detail in `.claude/memory/treasury.md`.** Live at `v490`, screen `/treasury`.

- **Three tabs**: accounts, import (PNC activity CSV + statement PDF),
  reconciliation (the three-way tie, the matcher, the reconciling items).
- **`current_ledger` is CARRIED** from the last closed period plus activity since,
  and says which period and through what date. **`current_available` is `None`** —
  it is ledger less holds, float and pending debits, which exist only at the bank.
  Never fill it from the ledger; a guardrail asserts it stays empty.
- **Each leg of the tie is reported separately** (statement, ledger). Which leg
  disagrees is the only thing the difference is for. `ties_to_statement` is `None`
  with no statement filed, never `False`.
- **The matcher PAIRS, it does not set-compare** — August carries `285.92` seven
  times. What is left over IS the reconciliation: deposits in transit, outstanding
  payments, activity not yet recorded. A manual pairing outranks the matcher.
- **`MR10005000` is the default cash account**; four others in a dropdown, anything
  else refused. An account registers itself on first import, before it is mapped.
- **A file that is not an activity export is REFUSED**, not reported as "0
  imported" — the column check runs before the row count.
- **Nothing here posts to MRI.** GL/IA upload templates are the next phase.

### Cap Rate at Sale / Refinance
- **Source column**: `fCapRate` from `valuations` table (MRI_Val)
- **Date column**: `dtValuation` — the date each cap rate was assessed
- **Function**: `projected_cap_rate_at_date()` in `planned_loans.py` — sorts by `dtValuation`, takes the most recent `fCapRate` as the base rate
- **Escalation**: +0.05% (0.0005) per year from the `dtValuation` of the selected row to the target date (sale or refi)
- **Example**: fCapRate=0.0575 at 12/31/2025, sale at 12/31/2030 → 0.0575 + 5×0.0005 = 0.06 (6.00%)
- **Sale value**: `NOI_12_months / cap_rate` — uses `twelve_month_noi_after_date()` for forward 12-month NOI from forecast
- **Callers**: `compute_deal_analysis()` for sale proceeds, `size_planned_second_mortgage()` and `size_prospective_loan()` for loan sizing

### Prospective Loans (Refinancing)
- `size_prospective_loan()` in `planned_loans.py` returns sizing dict including `refi_date` for Vue form pre-fill
- Prospective loan extends sale date to new maturity when `Sale_ME` < new maturity
- Net sale proceeds formula: `sale_price - loan_balances - balloon_total + cash_reserves`
- **Loan type**: Radio selector — Supplemental (keep all existing loans) or Refinance (replace selected loans via checkbox list)
- **Multi-loan replacement**: `existing_loan_id` stores comma-separated LoanIDs; `compute.py` filters with `not in replacing_ids`; `planned_loans.py` uses `.isin(replacing_ids)` for balance lookup
- **Sizing constraints**: LTV (value x max LTV), DSCR (NOI / min DSCR → solve principal), Debt Yield (NOI / min yield), Quoted amount — binding = minimum
- **Workflow**: Draft → Save & Analyze (sizing only) → Accept (full waterfall recompute) → Revert. Editing an accepted loan auto-recomputes waterfall on save.
- **CRUD endpoints**: `GET/POST/PUT/DELETE /api/deals/<vcode>/prospective-loans`, plus `/accept`, `/revert`, `/sizing` per loan
- **PostgreSQL**: `ensure_pg_tables()` in `database.py` creates tables (prospective_loans, prospective_loans_audit, waterfall_audit) with `SERIAL` id + proper column types at startup; `_pg_fix_column_types()` migrates TEXT→INTEGER/DOUBLE PRECISION. Column names with mixed case (iOrder, PropCode, FXRate) must be double-quoted in SQL statements.

### ISBS Data Formats
- **Source column**: `vSource` in ISBS table (`ISBS_Download.csv`)
- **Interim IS** (Actuals): YTD cumulative trial balance snapshots — use `_get_cumulative_balances()` at a single date
- **Interim BS** (Balance Sheet): Current outstanding balances — used for debt via `get_isbs_debt_balance()`
- **Budget IS**: Periodic monthly amounts — use `_get_budget_sum()` over date range
- **Projected IS** (Underwriting): YTD cumulative trial balance snapshots — use `_get_cumulative_balances()` (same as Actuals)
- **Valuation**: Periodic monthly from `forecast_feed` — use `_get_valuation_sum()` with negated `mAmount_norm`
- **TTM from cumulative**: Current YTD + prior year Dec YTD - prior year same-month YTD
- **Performance chart / Dashboard**: Both correctly convert cumulative→periodic via `_cumulative_to_periodic()`

### ISBS Split Tables
MRI's query record limits make exporting the monolithic `ISBS_Download.csv` (800K+ rows) impractical. The ISBS data is split into 6 tables by `vSource`:

| Table | CSV Filename | vSource | Description |
|-------|-------------|---------|-------------|
| `isbs_interim_is` | `ISBS_Interim_IS.csv` | Interim IS | Actuals — YTD cumulative. **ALL of it**, not just 2025+ — 2018-03-31 to 2026-08-31 as of Sep 14 2026 |
| `isbs_interim_is_historical` | — (see below) | Interim IS | **Empty, and normally stays that way.** Extension point, not a second source |
| `isbs_interim_bs` | `ISBS_Interim_BS.csv` | Interim BS | Balance Sheet |
| `isbs_budget_is` | `ISBS_Budget_IS.csv` | Budget IS | Budget — periodic monthly |
| `isbs_projected_is` | `ISBS_Projected_IS.csv` | Projected IS | Underwriting — YTD cumulative |
| `isbs_valuation_is` | `ISBS_Valuation_IS.csv` | Valuation IS | Valuation — periodic monthly |
| `isbs_uw_supplements` | `ISBS_UW_Supplements.csv` | Projected IS | Supplemental UW records (e.g. 7073 capital contributions) — importable via CSV upload, persists across MRI refreshes (not in QUERY_REGISTRY) |

| `isbs_budget_is_supplements` | — (app-written) | Budget IS | Partner budgets imported and vetted in the valuation section, before MRI has them |

- **`isbs_interim_is_historical` does NOT hold pre-2025 actuals** (corrected Sep 14
  2026 — the earlier description here said it did, and sent a session hunting for data
  that was never missing). The refresh's split branch sends EVERY `Interim IS` row to
  `isbs_interim_is`; there is no date filter in that path. The 2025 cutoff exists only in
  `split_isbs_table()`, the one-time migration off the legacy monolithic `isbs` table.
  The `ISBS_Interim_IS_Historical.csv` name in `TABLE_DEFINITIONS` is a mapping with no
  file and no import behind it.
  **Keep the table anyway**: `_ISBS_SPLIT` in `data_service.py` concatenates it into
  `isbs_raw`, so rows put there DO reach every ISBS consumer, and the legacy migration
  still writes to it. It is a working extension point that happens to be empty — an
  MRI refresh reports it as `preserved` rather than importing it (`mri_service.py`,
  guardrail `scripts/isbs_historical_preserve_check.py`).

- **Supplement ownership decides protection** (Sep 11 2026). `isbs_budget_is_supplements` is in `PROTECTED_TABLES` because the APP writes it and holds the only copy of a budget between import and approval. The other four supplements are **deliberately not protected**: a CSV is their source of record and `replace` is their designed refresh. Protecting `isbs_uw_supplements`, which has no app write path, briefly froze its 56 rows (they feed One Pager PE ROE via 7073) — protection without a write path is a lockout, not a safeguard. Guardrail: `scripts/isbs_supplement_precedence_check.py`.
- **Where MRI and an app supplement share a key, the app wins** — `(vcode, dtEntry, vSource, vAccount)`. Written as "remove MRI rows whose key a supplement covers", NOT `drop_duplicates`: ISBS is a **journal**, one key legitimately carries many rows that consumers SUM, and the `drop_duplicates` formulation was measured taking `isbs_raw` from 797,660 to 439,268 rows.
- **Assembly**: `_assemble_isbs()` in `data_service.py` loads split tables, restores `vSource` column, concatenates into `isbs_raw`. Supplements from legacy monolithic `isbs` table for any missing vSource categories. Falls back entirely to legacy table if no split tables exist. After assembly, `_append_uw_supplements()` appends rows from `isbs_uw_supplements` (defaults `vSource='Projected IS'` if not present in CSV).
- **CSV upload**: Auto-detects split table filenames via `TABLE_DEFINITIONS` in `database.py`
- **Cache**: `refresh_table()` reassembles `isbs_raw` when any split table or `isbs_uw_supplements` is updated
- **Migration**: `split_isbs_table()` in `database.py` can migrate legacy monolithic table into splits (idempotent)
- **Consumers unchanged**: All code continues to use `isbs_raw` with `vSource` filtering
- **Pending**: Direct MRI database access via VPN (requested Apr 2026) to bypass record limits entirely

### ISBS Debt Balance
- **Config**: `DEBT_BS_ACCTS = {'2150', '2152', '2210'}` in `config.py` — Balance Sheet debt accounts
- **Function**: `get_isbs_debt_balance(isbs_raw, vcode, as_of_date=None)` in `compute.py`
- **Logic**: Filters ISBS to `vSource='Interim BS'`, deal vcode (case-insensitive), debt accounts; picks most recent period (or specific `as_of_date`); returns `abs(sum(mAmount))`
- **Hierarchy**: ISBS current outstanding preferred over MRI_Loans `mOrigLoanAmt` (static origination). Falls back to MRI_Loans if ISBS unavailable.
- **Usage**: Dashboard (`get_portfolio_caps()`), Deal Analysis (`get_deal_capitalization()`), One Pager (`get_capitalization_stack()` with quarter-specific date)
- **Date parsing**: Uses `pd.to_datetime(format='mixed')` first; Excel serial fallback only when >50% NaT

### At Close Data (One Pager)
- **Primary source**: `at_close_noi` table (from `Prop_Info_AtClose.sql` MRI query) — pre-computed per deal with dynamic date selection
- **Fallback**: ISBS `vSource='Projected IS'` at the earliest December 31 date per deal
- **Meaning**: Due diligence audit performed at original closing — represents underwritten expectations
- **Fields**: Revenue, Expenses, NOI, DSCR (debt service from Interest + Principal accounts)
- **Sign convention**: MRI stores revenue as negative (credit); negated to positive for display
- **Deal terms**: `deal_terms` table (from `Prop_Info_DealTerms.sql`) provides `econ_occ_at_close` and PE coupon/participation overrides
- **Implementation**: `get_property_performance()` in `one_pager.py` checks `at_close_noi_df` first, falls back to ISBS Projected IS scan

### Economic Occupancy (One Pager)
- **Formula**: `avg(physical occupancy YTD months) - bad_debt_concessions_pct`
- **Physical occupancy**: Average of `Occ%` from MRI_Occupancy_Download for YTD months of current year through quarter end
- **Bad debt/concessions %**: `(sum of vAccounts 4040 + 4043) / abs(sum of vAccount 4010) × 100` from ISBS Interim IS YTD
- **Accounts**: 4040 = Residential Concessions, 4043 = Bad Debt & Collection Loss, 4010 = Rental Income

### Sub-Portfolio Aggregation
- Deals can have child properties linked via `Portfolio_Name`
- Loans aggregate UP from properties to parent deal level
- See `consolidation.py` for implementation

### Forecast Assembly (Multi-Source)
- **Priority 1**: `forecast_feed.csv` (admin-uploaded via CSV import → `forecasts` table). Overrides everything for deals it covers. Used for draft/updated valuations before MRI approval.
- **Priority 2**: ISBS `vSource = 'Valuation IS'` — annual valuation cash flow projections (periodic monthly). These are the MRI-approved 10-year projections from the December 31 valuation exercise.
- **Priority 3**: ISBS `vSource = 'Projected IS'` — underwriting projections (YTD cumulative, auto-converted to periodic monthly).
- **Assembly**: `_assemble_forecasts()` in `data_service.py` — identifies deals covered by forecast_feed, then fills gaps from ISBS Valuation IS, then ISBS Projected IS. Combined result passed to `load_forecast()`.
- **Cumulative→Periodic**: Projected IS conversion subtracts prior same-year cumulative value; January = start of new year (no subtraction).
- **Pro_Yr derivation**: For ISBS-sourced rows, `Pro_Yr = date.year - pro_yr_base`.
- **vcode case**: ISBS normalizes vcodes to lowercase; `_restore_case()` converts back to original case (e.g., `p0000008` → `P0000008`) for case-sensitive matching in `compute_deal_analysis()`.
- **Cache refresh**: `refresh_table()` reassembles forecasts when `forecasts`, `isbs`, or any ISBS split table changes.

### Forecast Date Filtering
- **Anomalous dates**: MRI valuation exports can include "Year 0" base entries with dates far in the past (e.g. 2015-12-31 for a 2025 deal). `compute_deal_analysis()` filters out forecast rows with `event_date` before `start_year - 2` to prevent `model_start` from being set to an unreasonable date.
- **Beginning cash date selection**: `load_beginning_cash_balance()` uses the **most recent** available Interim BS date (not restricted to pre-model_start). This ensures account reclassifications (e.g. security deposits moved from 1010→1080) are reflected in the beginning cash. Previously restricted to dates before `model_start`, which could include misclassified balances from older periods.

### Actuals Through Cutoff
- Global setting (`actuals_through`): date or None (default None = full forecast)
- **Actuals/Forecast boundary**: Always enforced. If `actuals_through` is set, that date is the cutoff. Otherwise, defaults to Dec 31 of `start_year - 1`. XIRR cash flows come from `accounting_feed` only before the boundary, and from `forecast_feed` (waterfall) only after.
- **Partner cash flows**: `seed_states_from_accounting()` accepts `cutoff_date` parameter — only accounting entries on or before cutoff are used to seed InvestorState. Waterfall-computed distributions cover periods AFTER cutoff only.
- **Operating forecast**: Forecast Rev+Exp rows for months <= cutoff are removed from `fc_deal_full`
- **Waterfall**: `cf_period_cash` and `cap_period_cash` filtered to post-cutoff periods only (always, not just when `actuals_through` is set)
- **Date comparison safety**: Period filtering uses `pd.to_datetime()` on `event_date`/`EffectiveDate` columns before comparing with `pd.Timestamp` cutoff to avoid `Timestamp vs datetime.date` errors
- **Cache key**: includes `actuals_through` so toggling triggers recomputation
- **Defaults**: `DEFAULT_START_YEAR = 2026`, `DEFAULT_HORIZON_YEARS = 10`, `PRO_YR_BASE_DEFAULT = 2025`, `DEFAULT_ACTUALS_THROUGH = "2026-07-31"` (YTD Actuals enabled through July 2026)
- **UI**: Vue sidebar in Report Settings (checkbox + month-end selector)
- **Flask**: `ACTUALS_THROUGH` in config, passed via query params / request body, included in `/api/data/config`

## Sidebar Navigation

The sidebar (`AppSidebar.vue`) is organized into major sections with expandable dropdowns. Section headers are uppercase bold; child items are indented. Sections auto-expand when navigating to a child route.

| Section | Type | Children |
|---------|------|----------|
| **Dashboard** | Standalone link | `/dashboard` |
| **Asset Management** | Expandable | Deal Analysis, Property Financials, Surveillance, One Pager, Review Tracking, Ownership, Waterfall Setup, Report Settings (expandable config panel) |
| **Accounting** | Expandable | Workpaper Packages, Treasury |
| **New Business** | Expandable | Pipeline, Deal Analysis, Lease Review, Lease Risk Analysis |
| **Investment Management** | Future (dimmed) | — |
| **Reports** | Standalone link | `/reports` (Projected Returns, ROE Summary, Pref Balance Detail, Sold Portfolio, PSCKOC, Portfolio Analysis) |
| **Data Management** | Expandable | Data Explorer, MRI Data (expandable panel), Database Tools (expandable panel), Reload Data, Settings |
| **Feedback & Requests** | Expandable | Submit form + request list (standalone section below nav) |

- **Report Settings** under Asset Management: expandable inline config panel (Start Year, Horizon, Pro_Yr Base, YTD Actuals + Apply Settings button)
- **MRI Data** under Data Management: expandable panel with server status, query list, per-query download/run/import buttons, admin "Refresh All Data from MRI"
- **Database Tools** under Data Management: expandable panel with Import CSVs (file upload + match), Export Database (.zip download)
- **Sold Portfolio**, **PSCKOC**, and **Portfolio Analysis** are embedded as custom view reports inside the Reports page (selected from the report list sidebar). Their Vue components accept an `embedded` prop that hides their standalone headers.

## Application Tabs & AI Assistant

Moved to `.claude/memory/app_reference.md` (Sep 11 2026) — what every tab displays,
section by section, plus the embedded AI Assistant's tool table and endpoints. It was
half of this file's bytes and loaded into every session regardless of whether the work
touched the UI. Read it when you need to know what a view shows or which endpoint backs
it; the sidebar map above is kept here as a quick orientation.

## Key Functions

### Core Engine
- `compute_deal_analysis()` - Main deal computation orchestration (compute.py)
- `build_partner_results()` - Single source of truth for all partner & deal metrics (compute.py)
- `run_interleaved_waterfalls()` - Merges CF/Cap timelines chronologically with shared InvestorState (compute.py)
- `prepare_cap_lookups()` - Pre-compute normalized DataFrames and lookup dicts for batch capitalization (compute.py)
- `xirr(cfs)` - Calculate IRR with irregular dates; Newton-Raphson (guess=0.1) matching Excel XIRR, Brent fallback (metrics.py)
- `accrue_pref_to_date()` - Daily pref accrual (waterfall.py)
- `parse_amfee_vnotes()` - Parse AMFee vNotes for source investor and exclusions (waterfall.py)
- `build_amfee_exclusions()` - Pre-compute net capital by (InvestmentID, InvestorID) for AMFee exclusions (waterfall.py)
- `get_amfee_excluded_capital()` - Compute capital to exclude from AMFee base, scaled by ownership % (waterfall.py)
- `seed_states_from_accounting()` - Build InvestorState from historical accounting; accepts `cutoff_date` to limit to actuals boundary. Non-capital distributions recorded in `cf_distributions` for ROE audit (waterfall.py)
- `InvestorState` - Tracks capital, pref, cashflows per investor (models.py)
- `Loan` - Debt structure with fixed/variable rates (models.py)
- `get_property_vcodes_for_deal()` - Get child properties for aggregation (consolidation.py)
- `cashflows_monthly_fad()` - Monthly FAD from modeled forecast; includes CapEx (reporting.py)
- `annual_aggregation_table()` - Annual pivot table for forecast display; accepts cash_schedule for reserve-adjusted FAD (reporting.py)
- `build_cash_flow_schedule_from_fad()` - Transforms FAD into distributable; funds CapEx from reserves, tracks shortfalls (cash_management.py)
- `get_isbs_debt_balance()` - Current debt from ISBS balance sheet, fallback to MRI_Loans (compute.py)
- `load_beginning_cash_balance()` - Beginning cash balance from ISBS Interim BS using CASH_BALANCE_ACCTS; uses most recent available BS date for accurate account classification (cash_management.py)
- `get_deal_capitalization()` - Deal cap stack with ISBS debt support (compute.py)
- `projected_cap_rate_at_date()` - Cap rate with annual escalation from valuation date (planned_loans.py)
- `twelve_month_noi_after_date()` - Forward 12-month NOI from forecast for sale/refi valuation (planned_loans.py)

### One Pager Data
- `get_general_information()` - Deal general info from investment_map + event_dates (one_pager.py)
- `_lookup_event_date()` - Generic event_dates query by vCode/vEventType/vEvent/vDateType (one_pager.py)
- `get_underwritten_exit()` - U/W Exit from event_dates (Asset Management / U/W Exit / Actual) (one_pager.py)
- `get_current_anticipated_exit()` - Current exit from event_dates (Disposition / Closing / Projected) (one_pager.py)
- `get_capitalization_stack()` - Cap stack, loan terms, PE exposure, investor breakdown (one_pager.py)
- `get_property_performance()` - YTD/Budget/Variance/AtClose/YE metrics per quarter (one_pager.py)
- `get_pe_performance()` - PE funding, ROE, balances from accounting; U/W ROE from ISBS Projected IS only (one_pager.py)
- `_get_uw_7073_signed()` - Sign-preserving 7073 YTD→periodic: positive MRI = contribution (neg CF), negative MRI = ROC (pos CF) (one_pager.py)
- `get_one_pager_comments()` / `save_one_pager_comments()` - Comments CRUD per vcode+quarter (one_pager.py)

### Capital Calls
- `load_capital_calls()` - Load and normalize capital calls with mixed date format handling (capital_calls.py)
- `build_capital_call_schedule()` - Build list of capital call events, optionally filtered by deal (capital_calls.py)
- `apply_capital_calls_to_states()` - Apply capital calls to investor states with pool routing (capital_calls.py)

### Parcel Sales
- `compute_economics()` - Price less cost of sale, paydown and reserve; the remainder to distribute (parcel_sale_service.py)
- `validate()` - Per-sale checks, errors vs warnings (parcel_sale_service.py)
- `get_deal_loans()` / `get_deal_tenants()` - Paydown targets and roster rows for the pickers; loan_id kept in the raw form `loans.py` builds ids with, tenant rows deduped (parcel_sale_service.py)
- `get_prospect_deal_loans()` / `get_prospect_tenants()` - NB (N-vcode) equivalents: loans from the latest saved Capital Budget mirroring `_build_loans` ids ({vcode}-{source_id} / {vcode}-L1 blended); tenants from the scenario-pinned Argus import (requested scenario, else Base Case), else the active import, else the lease-review roster with analyst resolutions winning (parcel_sale_service.py). `deals.py` falls back to both for N-vcodes; the tenants endpoint accepts `?scenario_id`.
- `apply_parcel_income_loss()` - Remove the revenue/expenses that leave with a parcel, from the sale date forward; records per-parcel/account/month removals in `result['parcel_revenue_detail']` for the display back-out (compute.py)
- `build_manual_parcel_events()` - Resolves pro-rata/fixed shares + validation; the runner applies them in date order via `manual_events` (compute.py)

### Database
- `save_waterfall_steps()` - Replace all waterfall steps for a vcode with audit trail; quotes column names for PostgreSQL (database.py)
- `create_additional_tables()` - Creates capital_calls and other tables if they don't exist (database.py)
- `import_csv_dataframe()` - Import a DataFrame into a table, works with SQLite and PostgreSQL (database.py)
- `import_csvs_to_database()` - Refresh all tables from CSVs, protecting DB-managed tables (database.py)
- `export_all_tables_to_zip()` - Export all tables as labeled CSVs in a zip archive (database.py)
- `set_engine()` - Wire SQLAlchemy engine for PostgreSQL support (database.py)
- `import_csv_stream()` - Chunked CSV import (50K rows, dtype=object) for large files (database.py)
- `split_isbs_table()` - Migrate monolithic isbs table into 6 split tables by vSource; idempotent (database.py)
- `_assemble_isbs()` - Load split ISBS tables, restore vSource column, concatenate; fallback to legacy (data_service.py)
- `_append_uw_supplements()` - Append isbs_uw_supplements rows to assembled ISBS; defaults vSource='Projected IS' (data_service.py)
- `_assemble_forecasts()` - Merge forecast_feed CSV > ISBS Valuation IS > ISBS Projected IS with per-deal priority (data_service.py)

### Cash Flow Imports (Argus + Generic)
- `parse_monthly_cashflow()` - Parse Argus cash flow Excel, auto-detect periods, map line items to COA (argus_parser.py)
- `parse_rent_roll_summary()` - Parse tenant lease detail from Argus rent roll export (argus_parser.py)
- `cashflow_to_forecast_df()` - Convert parsed Argus data to forecast DataFrame (same schema as load_forecast) (argus_parser.py)
- `map_to_coa()` - Keyword-based line item → COA account mapping (argus_parser.py)
- `parse_cashflow_excel()` - Generic Excel/CSV parser, auto-detect columns, annual→monthly conversion, horizontal layout support (cashflow_parser.py)
- `_detect_horizontal_dates()` - Detect horizontal Excel layout (dates across columns) (cashflow_parser.py)
- `_parse_horizontal_cashflow()` - Parse transposed cash flow (line items as rows, dates as columns) (cashflow_parser.py)
- `import_argus_cashflow()` - Import with SHA-256 dedup, parse + store cashflows (argus_service.py)
- `get_active_forecast_df()` / `get_forecast_df_by_id()` - Forecast DataFrame from Argus projection (argus_service.py)
- `get_property_rollup_forecast_df()` - Aggregate Argus forecasts from multiple properties into deal-level forecast (argus_service.py)
- `migrate_projection_to_forecast()` - Re-key vcode + insert into forecasts table for AM onboarding (argus_service.py)
- `import_property_cashflows()` - Store Excel-parsed cash flows by property+version (prospect_service.py)
- `get_deal_cashflows_by_property()` - All cashflows grouped by property_id for deal-level rollup (prospect_service.py)

### Flask Services
- `get_cached_deal_result()` - Shared multi-deal cache wrapper (compute_service.py)
- `build_roe_audit()` / `build_moic_audit()` - Audit data builders (compute_service.py)
- `generate_roe_audit_excel()` / `generate_moic_audit_excel()` - Audit Excel workbooks (compute_service.py)
- `generate_partner_returns_excel()` - Partner Returns Excel with deal total row (compute_service.py)
- `generate_forecast_excel()` - Annual Forecast Excel pivoted by year (compute_service.py)
- `generate_debt_service_excel()` - Loan Summary + Amortization Schedule (2-sheet) (compute_service.py)
- `generate_cash_schedule_excel()` - Cash flow schedule Excel (compute_service.py)
- `generate_capital_calls_excel()` - Capital calls Excel (compute_service.py)
- `generate_xirr_cashflows_excel()` - Merged XIRR cashflows by partner; sums same-date/same-description entries (compute_service.py)
- `generate_full_deal_excel()` - 7-sheet comprehensive Deal Analysis workbook (compute_service.py)
- `get_one_pager_data()` - Aggregates all one-pager sections + computes pe_yield_on_exposure; accepts `full_data` for PE enrichment (financials_service.py)
- `_enrich_pe_from_deal_result()` - Enriches PE performance from deal analysis seed_states (accrued balance, capital outstanding) (financials_service.py)
- `get_one_pager_chart()` - Quarterly NOI chart data for one-pager (financials_service.py)
- `get_submission()` - Get or create review submission with notes and permissions (review_service.py)
- `submit_for_review()` - Submit draft/returned document for review (review_service.py)
- `approve()` - Approve at current step and advance to next (review_service.py)
- `return_to_draft()` - Return document to draft with required note (review_service.py)
- `is_editable()` - Check if comments can be edited based on review status (review_service.py)
- `get_tracking_data()` - Production tracking data with filters, LEFT JOINs deals with submissions (review_service.py)
- `_save_snapshot()` - Freeze all computed One Pager data + chart as JSON on CEO approval (review_service.py)
- `get_snapshot()` - Retrieve and deserialize a frozen snapshot by vcode+quarter (review_service.py)
- `compute_net_waterfall_for_deal()` - Per-deal net returns waterfall with fees/promote (sold_service.py)
- `compute_all_net_returns()` - Orchestrator: loops sold deals, pools net cashflows for portfolio metrics (sold_service.py)
- `generate_net_returns_excel()` - Multi-sheet workbook with formula-driven waterfall detail (sold_service.py)
- `build_pref_balance_detail()` - Per-investor pref accrual detail matching Excel PE_Pref_Balances (reports_service.py)
- `get_deal_pe_investors()` - List PE investors for a deal from accounting (reports_service.py)
- `generate_pref_balance_excel()` - Pref balance detail Excel with header + transaction table (reports_service.py)
- `parse_budget_workbook()` - Partner budget Excel → lines × months, subtotal rows FLAGGED not dropped (budget_import_service.py)
- `account_choices()` / `category_choices()` - The deal's own last-12-months accounts, and the ~27 comparison categories each with its accounts, ranked by the deal's usage with a default account (budget_import_service.py)
- `category_accounts()` - {category: accounts} as the IMPORT sees it — config plus `_CATEGORY_ACCOUNTS_FOR_BUDGET`, so the dropdown and the not-in-category check cannot drift (budget_import_service.py)
- `reconcile()` / `validate()` / `commit()` - Stated-vs-computed revenue/expense/NOI; blocking vs warnings; replace-by-(vcode, periods) write to `isbs_budget_is_supplements` (budget_import_validate.py)
- `parse()` / `check()` / `commit()` - One line-mapping flow for `source` in ("budget", "argus"); Argus pre-fills from `argus_parser.map_to_coa` as a visible, editable suggestion, a budget never guesses (line_mapping_service.py)
- `monthly_schedule()` / `for_year()` - Modeled interest (5190) and principal (7060) from the deal's own loan terms, balloons excluded, child-property loans included; returns unavailable-with-a-reason, never a zero (valuation_debt_service.py)
- `build()` - Balance Sheet + Income Statement for one entity/period, with tie-out, `sign_anomalies`, unmapped/untyped/conflicts (statement_service.py)
- `build_members_capital()` / `build_cash_flow()` / `build_schedule_of_investments()` - The other three statements (statement_service.py)
- `line_sort_key()` / `is_dormant()` - Statement order from `LINE_ORDER`; no-balance-and-no-movement flag (statement_service.py)
- `consolidated_mapping()` / `seed_mapping_from_names()` - Proposed account → FS line mappings; proposals, never applied automatically (statement_service.py)
- `create_cycle()` / `sync_packages()` - Close cycle + one package per REP entity (workpaper_service.py)
- `validate_due_date()` / `period_start()` - Deadline rule and the period bound it uses (workpaper_service.py)
- `step_evidence()` / `statements_summary()` - What a step asserts and how to verify it, server-side (workpaper_workbench.py)
- `build_package()` - The 17-tab workbook; exhibits placed into it, not attached (workpaper_excel.py)
- `create_request()` - Create a new user feedback request with reply token (feedback_service.py)
- `list_requests()` - List requests with optional user/status/type filters (feedback_service.py)
- `send_request_email()` - Send email to request submitter via SendGrid with reply link (feedback_service.py)
- `handle_inbound_email()` - Process inbound email reply, match token, store in thread (feedback_service.py)
- `export_all_requests()` - Export all requests with full threads for design sessions (feedback_service.py)

## Account Classifications

- Revenue accounts: 4xxx series (positive in `mAmount_norm`)
- Expense accounts: 5xxx series (negative in `mAmount_norm`)
- Interest: `INTEREST_ACCTS` {5190, 7030}
- Principal: `PRINCIPAL_ACCTS` {7060}
- CapEx: `CAPEX_ACCTS` {7050}
- Tax Abatement: `TAX_ABATEMENT_ACCTS` {7070} — forced positive (income)
- Other Below-the-Line: `OTHER_EXCLUDED_ACCTS` {4050, 5220, 5210, 5195, 7065, 5120, 5130, 5400}
- `ALL_EXCLUDED` = Interest | Principal | CapEx | Other Below-the-Line (does NOT include Tax Abatement — separate sign handling)
- Debt (Balance Sheet): `DEBT_BS_ACCTS` {2150, 2152, 2210} — ISBS Interim BS accounts for current debt outstanding
- Cash & Reserves (Balance Sheet): `CASH_BALANCE_ACCTS` {1010, 1012, 1014, 1070, 1090, 1091, 1092, 1100, 1120, 1130, 1140, 1141, 1142, 1144, 1145} — ISBS Interim BS accounts for beginning cash balance in Cash Management section
- Bad Debt/Concessions: 4040 (Residential Concessions), 4043 (Bad Debt & Collection Loss), 4010 (Rental Income) — used in Economic Occupancy calculation
- Underwritten PE (Projected IS): `UW_PE_DIST_ACCT` 7071 (PE distributions — ROE numerator), `UW_PE_ROC_ACCT` 7073 (PE capital events — sign convention: positive = contribution, negative = return of capital). 7071 is YTD cumulative converted to periodic by `_get_uw_pe_periodic()`. 7073 uses `_get_uw_7073_signed()` for sign-preserving YTD→periodic conversion (positive MRI → negative cashflow/contribution, negative MRI → positive cashflow/ROC). Supplemental 7073 records loaded from `isbs_uw_supplements` table (importable via CSV upload, persists across MRI refreshes).

## Conventions

- Cashflow signs: negative = contribution, positive = distribution
- Rates as decimals (0.08 = 8%)
- Use Python date objects for dates
- **InvestorID / InvestmentID case normalization**: All entity IDs are uppercased at data load time (`.str.strip().str.upper()`) in `loaders.py`, `data_service.py`, and `ownership_tree.py`. MRI accounting data occasionally has mixed-case entries (e.g. "Centre" instead of "CENTRE") which caused journal entries to be silently dropped from groupby/filter operations. Normalization happens at the lowest layer so all downstream consumers get consistent IDs.
