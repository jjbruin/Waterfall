# Waterfall XIRR — Multi-Layer Waterfall Model

## How this file works

**CLAUDE.md holds invariants and procedures only. No version numbers, no counts, no
incident narratives. Put history in `.claude/memory/`.** This file is loaded into every
session, so anything sitting here is paid for whether or not the work needs it.
Guardrail: `scripts/claude_md_budget_check.py`, run by the pre-commit hook.

Shared project memory is `.claude/memory/`. **Read `MEMORY.md` there at the start of
every conversation.** Update those files as you work — they are committed and shared.
The pointer table at the foot of this file says which one holds what.

- **The live work queue is `.claude/memory/open_items.md`** — what is still open, with
  evidence and an owner. Run `/open-items` to triage it. Close items there as they ship
  rather than letting them drift back into session narratives.
- **Date-stamp anything describing work in flight, and delete it when the work ships.**
  A section that says a branch is unmerged reads as authoritative for as long as it sits
  here, and nothing distinguishes it from a current one. On Sep 11 2026 this file still
  carried a 69-line section declaring `feat/onepager-chart-window` "not merged, not
  deployed" — it had been on main for over a month (`window_end_quarter` at
  `financials_service.py:177`). It happened again on Oct 1 2026: a "NOT deployed" stamp
  on section access, which shipped the next day and then sat stale.
  **Shipped work belongs in the deploy history, not in standing instructions.**

## Project Overview

A Flask + Vue application for investment waterfalls, XIRR and related performance
metrics on real estate investments: multi-layer distribution waterfalls with preferred
returns, capital accounts and investor-level tracking. **Python 3.x** (`.venv/`) with
pandas/numpy and scipy (XIRR by Newton-Raphson, Brent fallback); **Vue 3 + Vite** with
ECharts; **JWT** auth; **Docker** multi-stage (Vue → Python 3.12-slim + Gunicorn) on
**Azure Container Apps**. **PostgreSQL** on Azure, SQLite (`waterfall.db`) locally,
**SQLAlchemy** switching on `DATABASE_URL` (`flask_app/db.py`). **Pin dependency MAJORS
in `requirements.txt`.**

The engine is at the repo root (`compute.py`, `waterfall.py`, `metrics.py`, …);
`flask_app/` holds the app factory, `auth/` with the section registry, `api/` blueprints
and `services/` that reuse the root engine; `vue_app/src/` the frontend; `queries/` the
MRI `.sql`; `scripts/` the guardrails, diagnostics and the git `hooks/`. **The runtime
image ships no `vue_app/` source.** Full tree: `.claude/memory/MEMORY.md` § Architecture.

Also in the repo root: **DOCUMENTATION.md** (setup, data files, troubleshooting),
**waterfall_setup_rules.txt** (waterfall step configuration, for the modeling team) and
**typename_rules.txt** (capital pool routing by Typename). Everything else is in the
pointer table at the foot of this file.

## Running the Application

### Production (Azure)
- **URL**: https://app-waterfall-dev-v2.icyplant-026fb2db.eastus.azurecontainerapps.io
  (desktop shortcut "Waterfall XIRR" opens it)
- Login with a real account. `admin` / `admin` is the LOCAL dev seed only — it returns
  401 against Azure. Do not script against it.

### Local Development
```bash
.venv\Scripts\activate
python -m flask_app.run          # API on http://localhost:5000
cd vue_app && npm run dev        # Frontend on http://localhost:5173
```

**Refresh `waterfall.db` from production** — it is not in git, so every clone drifts.
Stop the Flask server first, then in ONE PowerShell terminal:
```powershell
$env:DATABASE_URL = (az containerapp secret show -g rg-waterfall-dev -n app-waterfall-dev-v2 --secret-name db-url --query value -o tsv); .venv\Scripts\python scripts\pull_production_db.py; Remove-Item Env:DATABASE_URL
```
Rows are replaced and the local schema kept; local logins are not copied; the previous
file is backed up and swapped only when every row count matches. The cautions that
matter — the date format that silently broke a "through 6/30" filter, `--repair-dates`,
`--no-files`, and why the Export Database button is NOT a substitute — are in
`.claude/memory/azure_deployment.md`.

### Azure Infrastructure
- **Container App** app-waterfall-dev-v2 (1 CPU, 2GB RAM, 2 Gunicorn workers) —
  **Registry** acrwaterfalldev.azurecr.io — **Resource group** rg-waterfall-dev (eastus)
  — **PostgreSQL** psql-waterfall-dev.postgres.database.azure.com (B1ms, v16)
- Logs: `az containerapp logs show -g rg-waterfall-dev -n app-waterfall-dev-v2 --type console --tail 50`
- **The subscription is ThriveCSP-Dev-01**, not the default active one; the wrong one
  fails as `ResourceGroupNotFound`.
- **Caching**: `index.html` is served `Cache-Control: no-cache`, so a browser picks up a
  deploy; hashed assets (`/assets/*`) are cached a year `immutable`, rehashed per build.

## Deploying Changes

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

### Pre-flight — run this BEFORE `az acr build`, every time

This is not "check you are on the right commit". It is **establish what is actually
shipping, and review all of it** — these four steps ARE the deploy; the `az` commands
are just what you type afterwards.

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

**P2 is the step that gets skipped, and skipping it is how unreviewed code ships.** A
commit is not exempt because someone else wrote it, because it is "only" a docs or
script commit, or because it was already on main. If P2 lists it, it ships, and you own
reviewing it. **If P2 lists anything you did not expect, stop and reconcile before
building** — that is the signal, not a formality. Then: symptom repair found → tell Jim,
with affected deals and figures, and get his call. Clean → build.

**Who can deploy**: Jim, and Charlene (`cbui@peaceablestreet.com`) — Contributor scoped
to the registry `acrwaterfalldev` and the container app `app-waterfall-dev-v2` only, not
the resource group. `AcrPush` is NOT sufficient for `az acr build`: it grants only
`pull/read` and `push/write`, while the build needs `scheduleRun/action` and
`listBuildSourceUploadUrl/action`. All deploys use Azure CLI (GitHub Actions secrets are
not configured). `.github/workflows/deploy.yml` exists but is **deliberately not wired
up** — it triggers on push to main, which would ship code without this pre-flight, and it
deploys `:latest`, which is untraceable. **Do not enable it without changing both.**

### Build, lock, deploy
**Tag every image with the commit SHA it was built from, and deploy that tag — never
`:latest`**, which is mutable, so a revision pointing at it stops being traceable to a
commit the moment the next build overwrites the tag.

```bash
# 0. Pre-flight P1-P4 above is done and clean. Do not start here.

# 1. Build in ACR, tagged with the commit SHA (--no-logs avoids a unicode crash)
SHA=$(git rev-parse --short HEAD)
az acr build --registry acrwaterfalldev -g rg-waterfall-dev --image waterfall-xirr:$SHA --image waterfall-xirr:latest --no-logs .

# 2. GATE: the tag must exist and the run must have succeeded for THIS SHA.
az acr repository show -n acrwaterfalldev --image waterfall-xirr:$SHA
az acr task show-run --registry acrwaterfalldev --run-id <id> --query "{status:status,start:startTime,finish:finishTime}" -o tsv

# 3. Lock the SHA tag so a later build cannot overwrite it (delete stays enabled)
az acr repository update -n acrwaterfalldev --image waterfall-xirr:$SHA --write-enabled false

# 4. RE-RUN P1. Then deploy the SHA tag; the suffix forces a new revision.
az containerapp update -g rg-waterfall-dev -n app-waterfall-dev-v2 --image acrwaterfalldev.azurecr.io/waterfall-xirr:$SHA --revision-suffix <next-suffix>
```

Pick the suffix by bumping the HIGHEST existing one — reusing one is rejected, and an
INACTIVE revision still holds its name, so list with `--all`. **Re-run P1 immediately
before `az containerapp update`, not only before the build**: if the active image is no
longer the one P2 was computed against, STOP, get the other work pushed and merged, and
run the pre-flight again.

**Notes**: the ACR build agent has transient failures (5-second runs) — retry, and
confirm the run actually built rather than failing fast. To pin an already-deployed
`:latest` revision after the fact, `az acr import` its digest under a SHA tag — same
digest, so the content is provably identical. **Deploy-history entries go in
`.claude/memory/deploy_history.md`, never CLAUDE.md.**

### Lessons

The rule only. What each cost is in `deploy_history.md` under *The deploy lessons, in
long form*; the bracketed revision points at that revision's own entry there.

1. Compute the P2 span against the LIVE image, not local HEAD (`v429`).
2. Re-run P1 immediately before `containerapp update`, not only before the build
   (`v549`, `v560`).
3. Build only from a clean worktree nothing else is using, and gate `containerapp
   update` on the tag existing and the ACR run succeeding for that SHA (`v556`).
4. `activeRevisionsMode` is **Single**: never deactivate the latest revision to back
   out, and a traffic split does not apply. **Rollback is rolling FORWARD**, to the
   last good SHA tag under a new suffix (`v556r`).
5. Pin dependency MAJORS (`v524`), and **run guardrails inside the container** — a
   suite can be green locally on a shape the database cannot deliver (`v555`, `v550`).
6. Verify a UI change against the SERVED bundle, resolving the lazy chunk from the
   ENTRY bundle; `index.html` is an SPA shell and proves nothing (`v523`, `v527`).
7. Know which guardrails SKIP in the container and why. **Skip is not pass** (`v527`).
8. ACR uploads the WORKING TREE, not a git ref (P3) — and a data change that is not in
   git is correspondingly not undone by a rollback (`v549`).
9. Prove a shared-engine refactor behaviour-preserving by diffing OLD against new over
   real data (`v505`, `v548`), and measure a rule's population on live data before
   shipping it, reporting what changed even when nothing did (`v544`).

## Standing rules

### ONE NUMBER, ONE ENGINE
**Jim's standing instruction, Sep 18 2026:** "We should not have conflicting
calculation results. It will cause doubt in the accuracy of the entire work. Make
sure the vetted calculation engines are used consistently and we do not have
separate calculation engines for the same number. The only differences in results
should come from changes in time frames or projections that we are running through
the engines. The calculations should be reliable."

Before writing any calculation, find out whether the app already answers it. If it does,
**call that engine** — do not re-derive, do not "simplify for this screen", do not write
a cheaper fallback. A date, a horizon or a scenario is a parameter; the arithmetic is
not, and **a "temporary estimate" is a second engine.**

| Number | Engine | Reached by |
|---|---|---|
| Accrued pref, pref balance | `reports_service.build_pref_balance_detail` | `reports_service.deal_accrued_pref` for a deal total; `valuation_nav_service._pref_walks` per investor |
| Deal projection, waterfall, XIRR/ROE/MOIC | `compute.compute_deal_analysis` | `compute_service.get_cached_deal_result` |
| NAV, net proceeds | `valuation_nav_service.compute_nav` | stored in `valuation_nav_results` |
| Modeled debt service | `valuation_debt_service.monthly_schedule` | Budget/Valuation columns |
| Committed pref | `committed_pref.resolve_committed_pref` | One Pager cap stack and PE block, Investment Metrics, Portfolio Snapshot |
| Statements | `statement_service.build` + siblings | workpapers, print, Excel |
| Exchange and reference rates (USD/CAD, SOFR, CORRA, EFFR...) | `market_rates_service.rate_on` (the `market_rates` table, from Bank of Canada / NY Fed) | PE exposure; NOT yet Investment Metrics, which still carries `CAD_TO_USD = 0.73` (open_items 19.2) |
| Ultimate ownership by investor group, as of a date | `ownership_chain_service.group_shares` (commitments in force, amounts multiplied down) | PE exposure |
| Unrealized gain/loss and realized losses per holding | `pe_exposure_service.noncash_by_holding` (ia_transactions -- the app's `accounting` feed has no non-cash rows) | PE exposure |

**A second implementation is most dangerous when it is NEARLY right** — nothing on
screen and nothing in the logs distinguishes it from the answer. **When you find a
duplicate, measure both across every deal before changing either**; which is right is a
question for the data, not for whichever is newer. Report the count that disagree and
the aggregate delta to Jim. Guardrail: `scripts/one_engine_per_number_check.py` — add a
row above and a check there whenever a new engine takes ownership of a number.

### The rest
- **A NEW SECTION GOES IN THE REGISTRY** (Jim, Oct 1 2026). Adding a sidebar section
  means adding it to `SECTIONS` in `flask_app/auth/sections.py` and gating its block on
  `auth.hasSection('<key>')`; it then appears in Settings > User Management, ticked for
  everyone. Every new Vue route, `/api` route and assistant tool must be assigned too.
  `scripts/section_access_check.py` and the pre-commit hook fail until they are.
- **Every file in `queries/` uses `UNION ALL`, never `UNION`: the GL is a journal and
  one key legitimately carries many rows that consumers SUM.** A `UNION` silently
  de-duplicates them and the figures come out low.
- **`PROTECTED_TABLES` is for a table the APP writes and holds the only copy of.**
  Protection without a write path is a LOCKOUT, not a safeguard; where a CSV is the
  source of record, `replace` is the designed refresh.
- **`None`, never `0`, for "unknown"** — a sentinel zero is indistinguishable from a
  real zero to every consumer downstream. Likewise `None`, not `False`, for "cannot be
  determined" as against "does not tie".
- **A fix ships with a guardrail** in `scripts/*_check.py`, proved NON-VACUOUS by
  re-injecting the defect and watching it fail. A check that passes on the broken code
  is worse than none. **Assert in BOTH directions**: a rule tested only in the refusing
  direction is satisfied by locking everyone out, and one tested only in the admitting
  direction by admitting everything.
- **The engine flags; it never drops.** Report what was excluded, suppressed, truncated
  or combined, and why — silent truncation reads as "covered everything". And **visibly
  missing beats silently wrong**: never guess a value onto a report.
- **Reject what cannot be true; warn what is merely odd.** Refuse an impossible input
  with the reason named; save an implausible one with a warning beside it.
- **`roles_exactly`, not `role_required`, when two level-1 roles must be separated** —
  `analyst`, `accountant`, `accounting_manager` and `cfo` are ALL level 1, so any level
  gate naming one admits all four. **The screen must agree with the server**, and has
  been wrong both ways: gate the Vue on the same list the Python uses, with a guardrail
  comparing them by name.

## Domain invariants

Full detail — column names, fallbacks, why each rule has the shape it has — is in
`.claude/memory/engine_reference.md`.

- **Acquisition date** is `min(EffectiveDate)` per `InvestmentID` from the accounting
  feed, overwritten onto `inv` at load time — MRI's own field may not be the true
  closing date. Parse before `groupby().min()`: string comparison is alphabetical, not
  chronological. No accounting activity → keep MRI's.
- **Sale date priority**: (1) UI override, (2) `event_dates` projected disposition
  closing, (3) horizon end / max loan maturity. The `Sale_Date` COLUMN is never
  consulted; `MRI_COLUMNS` excludes it, `Sale_Status`, `InvestmentID` and
  `Portfolio_Name`, so an MRI refresh preserves them.
- **Waterfall types**: CF = operating distributions, does NOT reduce capital
  outstanding. Capital = refi/sale proceeds, DOES reduce it. **Preferred returns**
  accrue daily, Act/365 Fixed, compounding annually on 12/31 with a 45-day grace
  period, tracked per investor via `InvestorState`.
- **Capital calls are app-entered only** and `capital_calls` is in `PROTECTED_TABLES`: a
  CSV import runs `to_sql(if_exists="replace")`, which DROPS the table, and one upload
  destroyed every call typed into Deal Analysis. No MRI feed is interrupted.
- **The actuals/forecast boundary is ALWAYS enforced**, set or not (default Dec 31 of
  `start_year - 1`). XIRR cash flows come from accounting before it and the waterfall
  after it, never both. It is in the cache key.
- **ISBS formats differ by `vSource` and mixing them is silent**: Interim IS (actuals)
  and Projected IS (underwriting) are YTD CUMULATIVE; Budget IS and Valuation IS are
  PERIODIC monthly; Interim BS is current balances. It lives in **six split tables**
  assembled into `isbs_raw` by `_assemble_isbs()`, and every consumer still filters on
  `vSource`. **ISBS is a JOURNAL** — one key legitimately carries many rows that
  consumers SUM, so never `drop_duplicates` it. Where an app supplement shares a key
  with MRI the app wins, written as "remove the MRI rows the supplement covers".
- **Forecast priority**: `forecast_feed` > ISBS Valuation IS > ISBS Projected IS, per
  deal, assembled by `_assemble_forecasts()`.
- **Sign conventions**: negative = contribution, positive = distribution; MRI stores
  revenue as a negative (credit) and tax abatements as a negative that is forced
  POSITIVE. Rates are decimals (0.08 = 8%); use Python `date` objects.
- **Account classifications** (`config.py`): revenue 4xxx, expense 5xxx;
  `INTEREST_ACCTS` {5190, 7030}; `PRINCIPAL_ACCTS` {7060}; `CAPEX_ACCTS` {7050};
  `TAX_ABATEMENT_ACCTS` {7070}; `OTHER_EXCLUDED_ACCTS` {4050, 5120, 5130, 5195, 5210,
  5220, 5400, 7065}; `DEBT_BS_ACCTS` {2150, 2152, 2210}; UW PE 7071 (distributions) and
  7073 (capital events, sign-bearing). `ALL_EXCLUDED` does NOT include the tax
  abatement — it has its own sign handling.
- **Entity IDs are uppercased at load time** (`.str.strip().str.upper()`) in
  `loaders.py`, `data_service.py` and `ownership_tree.py` — MRI occasionally sends mixed
  case, which silently dropped journal entries from groupby and filter operations.
  **Paid-off loans** (`vDateType = "Paid Off"`) are likewise dropped at the DATA layer,
  so any loan row still present is an active facility.

## Where things are written down

| Topic | File (`.claude/memory/`) |
|---|---|
| **The live work queue** | `open_items.md` |
| Per-revision deploy post-mortems, and the revision index | `deploy_history.md` |
| Engine detail: dates, pref, capital calls, abatements, loans, parcel sales, sale overrides, ISBS, At Close, occupancy, forecasts, the cutoff, account classes | `engine_reference.md` |
| Which function answers which question — read before writing any calculation | `function_index.md` |
| What each app tab displays, the AI assistant's tools, the sidebar map | `app_reference.md` |
| Accounting workpapers, the statement engine, WHO MAY EDIT the section, and the `queries/` rules (`UNION ALL`, never `UNION`) | `accounting_workpapers.md` |
| Treasury — PNC import, the three-way tie, the matcher, the JE files | `treasury.md` |
| Intercompany — Due to/from PSC Manager | `intercompany.md` |
| Employee expense reports | `expense_reporting.md` |
| GL / IA Query — the CFO's filters | `gl_ia_query.md` |
| Section access by username | `section_access.md` |
| Valuation Budget Review — line mapping, the levered columns | `valuation_budget.md` |
| Lease review AND lease risk analysis — extraction, terms, validation, recoveries, exclusives ("bound by" is not "holds"), and why the risk analysis must NOT take the lease's date | `lease_review.md` |
| Rent roll specification and the IC exhibit | `rent_roll_exhibit.md` |
| Shared UI patterns | `ui_patterns.md` |
| This file as it read before the Oct 2026 compaction, for the prose that was cut | `claude_md_prose_archive.md` |
