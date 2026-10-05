# CLAUDE.md before the Oct 5 2026 compaction — the prose that was rewritten

CLAUDE.md was 4,133 lines and is now under 350. Most of that was a **move**: the deploy
history, the per-feature rules and the function catalogue went to the topic files listed
in CLAUDE.md's own pointer table, verbatim and unedited.

This file holds the remainder — the sections that were **rewritten rather than moved**,
as they stood immediately before. Nothing below is current guidance; CLAUDE.md is.
It is here so that "the compaction lost nothing" is a claim anyone can check without
going to git, and so the worked examples behind a rule stay readable after the rule was
shortened to its statement.

What changed in the rewrite, so you know what to look for:

- The **deploy history narratives** became an 9-item *Lessons* list. Each lesson keeps
  the revision in brackets as a pointer into `deploy_history.md`.
- **ONE NUMBER, ONE ENGINE** kept its rule, its instruction from Jim and its engine
  table; the accrued-pref worked example (the lost day at each year end, the 34 deals,
  the $633,807.54) was cut to a sentence. The full version is below.
- **Project Structure** lost its per-service one-line descriptions; those services are
  described where they are documented.
- The **engine sections** (acquisition date, ISBS, capital calls, the cutoff, account
  classifications …) became one-line invariants in CLAUDE.md, with the working detail
  in `engine_reference.md`. Their original text is there, not here.

---

## The head of the file, as it stood (overview, stack, structure, running, deploying)

Everything from the title to the deploy-history bullet. The deploy history itself
is in `deploy_history.md`; it started at the line this extract ends on.

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
│   │   ├── gl_ia_query.py    # GL / IA Query endpoints (options, run, Excel)
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
│       ├── lease_terms.py        # Rent PSF, amendment order, rent steps stated as months of the term (pure)
│       ├── gl_ia_query_service.py # The CFO's GL/IA Spreadsheet Server filters, against our imported tables
│       ├── valuation_summary_service.py # The two portfolio summary tabs, assembled from vetted figures
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
- **.claude/memory/rent_roll_exhibit.md** - New business's rent-roll specification, the gaps against it, the IC exhibit's exact formatting, and the five-step build plan (Sep 29 2026)
- **.claude/memory/expense_reporting.md** - Employee expense reports: accounting's process and files, the design, phase 1 as built (Oct 2 2026)
- **.claude/memory/intercompany.md** - Due to/from PSC Manager reconciliation from `gl_detail`, basis A.B only; phase 1 built Sep 29 2026, the Pay/JE step is not

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

**RE-RUN P1 IMMEDIATELY BEFORE `az containerapp update`, not only before the build.**
On Oct 1 2026 Charlene deployed `v549` = `76c786c` eight minutes before the
section-access build started and after its P1 -- from a clone she had not pushed, so
`git fetch` could not have shown it. Deploying the new image would have rolled her
work back; it was caught only because the suffix `v549` was already taken. If the
active image is no longer the one P2 was computed against, STOP: get it pushed,
merge it, and run the pre-flight again.

Pick the revision suffix by bumping the HIGHEST existing one — reusing a suffix is
rejected, and an INACTIVE revision still holds its name, so list with `--all`
(`az containerapp revision list ... --all`). This same query answers "what commit is
live?", since the image tag is the SHA:
```bash
az containerapp revision list -g rg-waterfall-dev -n app-waterfall-dev-v2 --query "[?properties.active].{name:name,image:properties.template.containers[0].image}" -o table
```

**Notes**:
- ACR build agent has transient failures (5-second runs) — retry if it fails. Confirm the run actually built rather than failing fast: `az acr task show-run --registry acrwaterfalldev --run-id <id> --query "{status:status,start:startTime,finish:finishTime}" -o tsv`
- Use `--no-logs` to avoid Azure CLI unicode crash (`✓` character).
- To pin an already-deployed `:latest` revision after the fact, retag its digest without rebuilding and redeploy that tag — same digest, so the content is provably identical: `az acr import -n acrwaterfalldev --source acrwaterfalldev.azurecr.io/waterfall-xirr@sha256:<digest> --image waterfall-xirr:<sha>`

## Local development, Azure, caching, and ONE NUMBER ONE ENGINE in full

This is the block that sat immediately after the deploy history.


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

### ONE NUMBER, ONE ENGINE

**Jim's standing instruction, Sep 18 2026:** "We should not have conflicting
calculation results. It will cause doubt in the accuracy of the entire work. Make
sure the vetted calculation engines are used consistently and we do not have
separate calculation engines for the same number. The only differences in results
should come from changes in time frames or projections that we are running through
the engines. The calculations should be reliable."

Before writing any calculation, find out whether the app already answers it. If it
does, **call that engine** — do not re-derive, do not "simplify for this screen", do
not write a fallback that computes it a cheaper way. A date, a horizon or a scenario
is a parameter; the arithmetic is not.

**The vetted engines, and what they own:**

| Number | Engine | Reached by |
|---|---|---|
| Accrued pref, pref balance | `reports_service.build_pref_balance_detail` | `reports_service.deal_accrued_pref` for a deal total; `valuation_nav_service._pref_walks` per investor |
| Deal projection, waterfall, XIRR/ROE/MOIC | `compute.compute_deal_analysis` | `compute_service.get_cached_deal_result` |
| NAV, net proceeds | `valuation_nav_service.compute_nav` | stored in `valuation_nav_results` |
| Modeled debt service | `valuation_debt_service.monthly_schedule` | Budget/Valuation columns |
| Statements | `statement_service.build` + siblings | workpapers, print, Excel |

**Why this is not a style preference.** "Accrued pref" had two implementations in
ONE FILE. `_compute_accrued_pref` (ROE Summary, Committee Summary) and
`build_pref_balance_detail` (everything else) walked the same ledger at the same
rate and disagreed on **34 of the 68 deals both could price**, with the ROE path
**$633,807.54 low** in aggregate at 2025-12-31. The cause: it accrued `cur -> 31 Dec`,
compounded, then resumed at `1 Jan`, so **31 Dec -> 1 Jan was never accrued** — one
lost day per year end, always short, worse the older the deal. On P0000031 it gave
26,489.03 where Jim's own workbook says 37,394.57.

It never looked wrong. A slightly low accrual is still a plausible accrual. **That
is the whole danger: a second implementation is most dangerous when it is nearly
right**, because nothing on screen and nothing in the logs distinguishes it from
the answer.

**A "temporary estimate" is a second engine.** The Committee tab's Net Proceeds
column fell back to `value - debt` when the NAV had not been run — scaffolding from
before the NAV engine existed, left in after it shipped. The NAV walk runs the
deal's waterfall; value-less-debt ignores it. One column, two calculations, nothing
saying which. Removed: unavailable now reads as unavailable.

**Guardrail: `scripts/one_engine_per_number_check.py`.** It asserts the deleted
engine cannot return, that every consumer reaches the identical figure, and that
only the as-of date moves the answer. Add a row to the table above and a check here
whenever a new engine takes ownership of a number.

**When you find a duplicate: measure both across every deal before changing either.**
Which one is right is a question for the data, not for whichever is newer. Report the
count that disagree and the aggregate delta to Jim, and say which figures of his the
candidate reproduces.

## The Application Tabs pointer, as it stood

## Application Tabs & AI Assistant

Moved to `.claude/memory/app_reference.md` (Sep 11 2026) — what every tab displays,
section by section, plus the embedded AI Assistant's tool table and endpoints. It was
half of this file's bytes and loaded into every session regardless of whether the work
touched the UI. Read it when you need to know what a view shows or which endpoint backs
it; the sidebar map above is kept here as a quick orientation.

## Conventions, as they stood

## Conventions

- Cashflow signs: negative = contribution, positive = distribution
- Rates as decimals (0.08 = 8%)
- Use Python date objects for dates
- **InvestorID / InvestmentID case normalization**: All entity IDs are uppercased at data load time (`.str.strip().str.upper()`) in `loaders.py`, `data_service.py`, and `ownership_tree.py`. MRI accounting data occasionally has mixed-case entries (e.g. "Centre" instead of "CENTRE") which caused journal entries to be silently dropped from groupby/filter operations. Normalization happens at the lowest layer so all downstream consumers get consistent IDs.

