# Review note — `feat/freeze-as-sent`

For Jim, before merge. Written Sep 25 2026. **Nothing here is deployed and no
real freeze has been run.** Branch is rebased onto `origin/main` (`a3c9459`);
guardrail `scripts/freeze_as_sent_check.py` is 99/99.

## What the feature is

A quarter that has been **sent to an investor** can be frozen, so the Snapshot
and the One Pagers stop following live data and serve the stored copy instead.
Separate from the CEO approval chain: freezing records that something *was
sent*, approving records a decision somebody *made*, and rolling them together
would have the button write an approval nobody gave.

For 26Q2 there is a second half: the values actually printed in the sent PDFs
are laid over the computed ones, so the stored copy reproduces the document
rather than the engine's opinion of it.

---

## 1. The request path

**New endpoint.** `POST /api/portfolio-snapshot/freeze-overlay`, admin only
(`roles_exactly("admin")`). Everything else on the blueprint is unchanged.

**One endpoint, two modes.** `confirm: false` previews and writes nothing;
`confirm: true` repeats the identical resolution and writes. Two endpoints
could drift, and a preview describing a different resolution from the freeze it
precedes is worse than no preview — the reader would have approved something
other than what ran.

**The frozen-write gate.** `_lock_frozen_quarters` is a blueprint-level
`before_request` that **denies by default**: any non-GET on the snapshot
blueprint is refused (409) once the quarter is frozen, unless its endpoint is
named in `_FROZEN_WRITE_ALLOWED`. The new endpoint is named there because it
refuses an already-frozen quarter on its own terms. The deny-by-default
direction is deliberate: an allowlist of endpoints that may *not* write would
need extending every time someone adds an editable field, and the day it is
forgotten a sent quarter silently becomes editable again.

**Cost.** The preview assembles the full report and reads every One Pager in
the overlay to compare against live. That is heavier than a page load, and it
is a button an admin presses deliberately, not something on a render path.

---

## 2. Database

**No new tables and no schema change on this branch.** The two tables
(`portfolio_snapshot_frozen`, `portfolio_snapshot_frozen_history`) and the
added columns came in with commit `1d5e901`, created by the existing idempotent
`_ensure_table()` on first request.

Worth knowing about that design, since it is the part that cannot be undone
casually:

- **`UNIQUE(investor_code, quarter)` is deliberately untouched.** Superseded
  versions go to a **separate history table** rather than becoming extra rows.
  Dropping a UNIQUE constraint means a table rebuild on SQLite, and this data
  is the only record of what an investor was sent — the migration that cannot
  lose it is the one that never rewrites it.
- **Columns are added one at a time**, each in its own transaction, because a
  failed `ALTER` on one must not abandon the rest and SQLite has no
  `ADD COLUMN IF NOT EXISTS`.
- **One writer.** `_write_frozen` serves both freeze paths, so there is no
  second way for a row to reach the table.

---

## 3. `assembler=lambda` — the question I flagged

In the endpoint the freeze is called as:

```python
assembled = FZ.assemble_full_report(inv, quarter)
...
FZ.freeze_as_sent(..., assembler=lambda _i, _q, _p=assembled: _p)
```

**Why.** The Snapshot overlay is keyed by printed **row label** — that is all
the sent page knows — and has to be resolved to `groups.<G>.deals[i]` paths,
which only exist once the report is assembled. If the endpoint resolved against
one assembly and `freeze_as_sent` then built another, an index that shifted
between the two would publish a whole row of figures **against the wrong
property**, and nothing afterwards would show it.

So the report is assembled **once** and that same object is both resolved
against and stored.

**What it costs.** `freeze_as_sent`'s default assembler
(`assemble_full_report`) is bypassed for this one call path. The ordinary
"Freeze as sent" button still uses the default. `assembler` was already a
parameter — the guardrail has used it since the first commit — so this is a
documented seam, not a new hole.

**What I would want checked:** that nothing inside `freeze_as_sent` depends on
having assembled the report itself rather than being handed one. I read it and
believe not, but it is the single call in this branch where an outside object
becomes the stored record, so it deserves a second pair of eyes.

---

## 4. The parser, and why it is self-checked rather than eyeballed

The 26Q2 overlay is read out of the sent PDFs by
`scripts/build_26q2_overlay.py` — offline, no auth, no live call. This replaced
a token-driven seeder that could never run, because `WF_TOKEN` has never
reached a tool shell.

**Six defects were found in that parser, all after I had spot-checked it and
reported it working.** Every one produced a plausible wrong number:

| # | Defect | Effect |
|---|---|---|
| 1 | Label taken from the left | `Giant 7` → label `Giant`, **7 read as Debt**; `Total PSC TGA 2022 LLC` → `Total PSC TGA`, **four fund subtotals collided, three lost** |
| 2 | `-$0.20M` — sign before the `$` | Row parsed one cell short, **label absorbed a value**; every negative-ITD deal |
| 3 | Two-loan rows | `SOFR + 650 \| 4.2% fixed` — pipe and second date **read as Debt** |
| 4 | Loan page has no left anchor | `Mount Prospect Plaza 5.3%` swallowed its own rate |
| 5 | Comment prose read as cells | `sale now expected 11/2026` is a date — **9 of 40 operating rows dropped** |
| 6 | Comment cut fired mid-label | `2022` cell-like + `LLC` a word — **all four fund subtotals dropped entirely** |

Note the pattern: **five of the six come from a row label containing a number.**

**What replaced the eyeballing.** Every money column on all three Snapshot
pages now **foots to its own printed subtotal** within rounding — financial
debt/pref/equity/cap/invested/unfunded/commitment/ITD, loan debt, and all three
operating NOI columns. Before the fixes the operating columns were **45% short
and nothing would have shown it**. The footing check is what found most of the
six.

Also asserted offline: 41 printed row labels normalise to 41 distinct keys,
**zero collisions**; and all 40 distinct One Pager overlay paths land in the
real payload shape, **zero unapplied**.

---

## 5. What the preview checks guard against

Three faults that look perfectly fine cell by cell, which is exactly why a cell
count is not enough:

- **A units error** — every value plausible, all of them 1e6 out.
- **A column shift** — every value plausible, all in the wrong column.
- **A printed dash over a live figure** — the page printed `—`, the app holds
  3.65%, and the frozen row would show a number that was never sent.

`compare_overlay_to_live` runs server-side against the payloads about to be
written over, and reports per page how many cells actually move, per column how
many match and differ and the **typical overlay/live ratio**. A column is
flagged when most of its rows differ (a shift looks exactly like that) or when
the median ratio is outside 0.5–2.0, with 1e6 named. **Flagged columns block
the freeze** until an admin acknowledges each.

**`expected ≈114` is printed, not enforced.** It sits beside the real count and
the screen says so loudly when the two are more than 40 apart. A reader who
sees 900 where 114 was expected stops and asks why; a hard threshold would just
be edited the first time it fired.

Also blocking: any printed cell that **would not land** in the assembled report
(`dry_run_unapplied`, computed by applying the overlay to a deep copy — the same
code the freeze uses, so the prediction cannot differ from the write), and any
printed page whose deal title cannot be matched. A title matching two deals is
left unresolved rather than resolved to the first.

---

## 6. Things to look at, and one live bug found on the way

1. **The `assembler=lambda` seam** — §3 above.
2. **The `$ millions` assumption** on Snapshot pages 6–8, read off the page
   header. If any column there is not in millions, those cells are 1e6 out. The
   preview's ratio check would catch it, but only if someone reads it.
3. **Row-label normalisation** drops parenthetical suffixes so
   `Portfolio Totals (38)` survives a changing count. Zero collisions today.
   `Citizen Storage Swartz Creek Holdings` and `… Holdings LLC` normalise to
   *different* keys — they do not collide, but they are probably the same deal
   printed two ways.
4. **Nothing has been verified against live data.** The cached Sep 23 pull never
   arrived, so the cell-by-cell test runs against the overlay and a stub. The
   preview is the first place real values are compared — read it carefully.
5. **KOC freezes One Pagers only** (`snapshot_pages: []`), since no KOC Snapshot
   was sent. Confirm that is still right.

**A live bug, not introduced here and not fixed here:** `DEBT_FREE_DEALS`
blanks Pegasus's `debt_display` but leaves the raw `debt`, and
`loan_subtotal()` sums the raw figure — so **$25.2M is inside Portfolio Totals,
the fund subtotal and the excluding-development row with no row on the page to
account for it**. Financial has no such hole. Recorded as `open_items.md` §12
and left for a separate change, because it is a live-engine defect and fixing it
here would put two unrelated concerns in one review. **26Q2 is protected** — the
frozen page stores the PDF's own printed subtotal, asserted by guardrail
section Q, which strips all five Loan-tab hardcodes and requires the stored
payload to come back byte-identical.

---

# Post-deploy test plan — turning FREEZE_ENABLED on, once

Added Sep 29 2026 with the sustainable freeze (dedup + background job +
set-based quarter unfreeze). **The branch ships with `FREEZE_ENABLED` off.**
This is the plan for switching it on for one controlled run, measuring it, and
switching it back off until 26Q3.

**Why a rehearsal at all.** The freeze has never completed on production. The
one attempt, on Sep 29 2026, ran the whole batch inside a single request and
the app was unavailable for about 35 minutes. Everything since — building each
deal once, running it on a background thread — is a response to that, and none
of it has met real data. A deliberate rehearsal against 26Q2, which is already
sent and can be unfrozen afterwards, is how that stops being a guess.

**26Q2 is the subject on purpose.** It is a quarter whose reports have already
gone out, so freezing it changes nothing anybody will read, and the quarter
unfreeze puts it straight back. **The freeze is for 26Q3 onward**; this run is
a measurement, not the real thing.

## Before you start

- [ ] The deploy is done and the revision is serving (`site root 200`).
- [ ] `portfolio_snapshot_frozen` is **0 rows** — read it, do not assume.
- [ ] Nobody is mid-quarter in the app. **Outside working hours.**
- [ ] You can reach `az containerapp exec`, for the row counts and the memory
      reading.

## 1. Turn the flag on — configuration only, no rebuild

```bash
az containerapp update -g rg-waterfall-dev -n app-waterfall-dev-v2 \
  --set-env-vars FREEZE_ENABLED=1 --revision-suffix vNNN
```

Same image, new revision. **Record the revision name and the image digest** —
they must match the revision you were already running, or you have changed two
things at once.

- [ ] Confirm the buttons are live: the panel no longer shows "Freezing is
      temporarily disabled".
- [ ] Confirm the gate really flipped, over HTTP: `POST /freeze-all/one-pagers`
      as a **non-admin** must still be `403`, not `503`.

## 2. Baseline, before freezing anything

Take these first; without them the numbers during the run mean nothing.

- [ ] `GET /` — record milliseconds.
- [ ] `GET /api/financials/<vcode>/one-pager?quarter=2026-Q2` — record
      milliseconds. Use the same vcode throughout (P0000001 was 2.31s on
      Sep 29 2026).
- [ ] Container memory: `az containerapp exec … "cat /sys/fs/cgroup/memory.current"`
      (or `memory.usage_in_bytes` on cgroup v1). Record bytes.

## 3. Freeze 26Q2 One Pagers

One Pagers only — it is the expensive half and the one the dedup targets.

- [ ] Press **Freeze 2026-Q2 One Pagers — all investors**. Note the wall-clock
      start.
- [ ] Confirm the POST returned **at once** with a job id. If the page hangs,
      stop: the background path is not working and the rest of the plan is void.
- [ ] Watch the panel: investors done / total should advance steadily.

**While it runs**, every 60 seconds:

- [ ] `GET /` — record ms.
- [ ] `GET /api/financials/<vcode>/one-pager?quarter=2026-Q2` — record ms.
- [ ] Container memory — record bytes.

**What "responsive" means here, so the result can be judged.** One gunicorn
sync worker serves one request at a time; the freeze runs on a thread beside
it. The work is mostly pandas and SQLAlchemy, which release the GIL, so
requests are served throughout — but CPU-bound stretches contend, so some
slowdown is expected and is not a failure. **A doubling is fine. A page that
does not return is not.**

## 4. When it finishes

- [ ] Wall clock, start to finish. **Expected ~3 minutes** (84 distinct deals at
      the 2.31s measured per One Pager). Anything near 28 minutes means the
      dedup is not working — check the job's `deal_builds` vs `deal_reuses`.
- [ ] Record `deal_builds` and `deal_reuses` from the job row. Expected roughly
      84 and 638.
- [ ] Peak container memory over the run, against the baseline.
- [ ] Worst `GET /` and worst One Pager response time during the run.
- [ ] Investors frozen / skipped / failed. **Any failure: record the investor
      and the error** — per-investor isolation means the rest still froze.
- [ ] Row count: `portfolio_snapshot_frozen` should be the investor count.

## 5. Unfreeze, and verify it is really gone

- [ ] Press **Unfreeze 2026-Q2 One Pagers**, reason: `rehearsal, 26Q2 not
      being frozen yet`.
- [ ] It should complete in **seconds**, not minutes — it is four set-based
      statements, not a loop.
- [ ] `portfolio_snapshot_frozen` — **0 rows for 2026-Q2**. Read it.
- [ ] `portfolio_snapshot_frozen_history` — one row **per investor**, carrying
      the reason and your username.
- [ ] Open a One Pager for a deal that was in the freeze and confirm its
      comments are **editable again** — the comment lock must have released.

## 6. Turn the flag back off

```bash
az containerapp update -g rg-waterfall-dev -n app-waterfall-dev-v2 \
  --set-env-vars FREEZE_ENABLED=false --revision-suffix vNNN
```

- [ ] Confirm the panel says freezing is disabled again.
- [ ] Confirm `POST /freeze-all/one-pagers` as an **admin** returns **503**.

**It stays off until 26Q3 is ready to be frozen.** Leaving it on is how a
quarter gets frozen by accident, and the skip-if-frozen rule means an accidental
freeze is not silently corrected by a later, correct one.

## If it goes wrong

- **The app becomes unresponsive.** The job is a thread in the worker; there is
  no way to cancel it from the screen. Restart the revision — the job is marked
  **interrupted** at the next boot, and investors frozen before the restart stay
  frozen. Then quarter-unfreeze to clear them.
- **The job says interrupted.** Expected after any container restart. Read the
  count it reached, quarter-unfreeze, and start again.
- **A container restart mid-job** loses the in-memory deal cache and the
  remaining investors, nothing else: every investor frozen so far was committed
  on its own.
- **Rolling back the flag is a config change**, not a deploy: set
  `FREEZE_ENABLED=false` and the buttons refuse again, with the same image.

## Record the result here

Add a dated section under this one with the numbers, whatever they are. A
rehearsal whose result is not written down has to be run again.
