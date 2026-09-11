# Session Handoff — through Sep 11 2026 (v429 live)

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
