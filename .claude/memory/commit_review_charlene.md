# Commit review — symptom vs root cause (Sep 1 2026)

Jim's ask after `50695d9`: review Charlene's submissions with the lens *fix problems, not
symptoms*, and flag a symptom repair **before** deploying. The deploy-time rule now lives
in CLAUDE.md under "Deploying Changes". This file is the review itself.

Scope: 83 commits authored by Charlene Bui, Aug 24 – Sep 1 2026.

## Headline

**The great majority are genuine root-cause fixes, and the work is unusually well
evidenced** — most commits ship a `scripts/*_check.py` guardrail that runs the real
committed function against live data, and the messages quantify blast radius. That is
better practice than the repo's baseline.

**The exceptions cluster in one place, and they share one cause**: the app has no data
concept for *where an asset sits in its lease-up / stabilisation lifecycle*. Six modules
now branch on a development classification, each inventing its own answer to "what should
this column show for a deal that isn't operating yet." No commit created that gap; each
one works around it locally, which is why it keeps recurring.

## A. Root-cause fixes (representative, all verified by reading the diff)

| commit | what it actually fixed |
|---|---|
| `97d3945` | Review Tracking: quarter moved from WHERE to the JOIN — a deal whose only submission is another quarter matched neither branch and vanished. Real SQL defect. |
| `d48469b` | `PUT /value` returns the display/source strings instead of the client re-fetching the whole bundle. Removes the cause of the "page refresh", doesn't mask it. |
| `0cb14ba` | Participation reader took the first `vState='Share'` row with **no PropCode filter**, so a deal where the PE has no Share row reported the operating partner's. Genuine reader bug. |
| `4351f94` | Econ Occ multiplied by 100 twice (9223% → 92.2%). |
| `fc30c4f` | MRI refresh died on a float in a str column. |
| `6bbfbf6` | Collapsed MRI's `Loan_Date` fan-out so one facility is one row. |
| `e6d9ca5` | Netted the 7083 operating reserve release into the At Close expense line. |
| `7827c6f` | Total Pref re-based to the committed tranche, established three independent ways against the reference PDF. Definitional correction, not a nudge. |
| `567aec2` | Financial and Loan disagreed on Debt for the same deal; extracted `portfolio_snapshot_debt.py` so **one** module owns the rule. Textbook root-cause fix. |

## B. Symptom repairs Charlene declares as such (good practice, still debt)

Both are per-deal vcode hardcodes. Both carry a comment naming the real fix and warning
that the constant "will silently keep overriding whatever real rule ships."

1. **`portfolio_snapshot_operating.DEV_DISPLAY_EXCEPTIONS`**
   `{"P0000078": {"noi"}, "P0000066": {"econ_occ","noi"}}` — per-deal, per-metric
   suppression overrides. Her own note: the rule these stand in for is "a dev deal shows
   the columns its stabilisation stage supports", which "needs a lease-up/stabilisation
   state in the data that does not currently exist." Pegasus is additionally marked
   **FLAG FOR CONFIRMATION** — it was not in the brief; she added it to match the PDF.
2. **`portfolio_snapshot_loan.WATERS_CREEK_LTV_EXCEPTION`** `{"P0000078"}` — needs a
   valuation-method/basis column to distinguish a genuine income-based valuation from a
   cost-basis placeholder; that column is not extracted today.

## C. Symptom repair NOT declared as such — `50695d9`, deployed as v407

Zeroes the One Pager At-Close column when a deal is development **and** its Projected IS
lacks a `2015-12-31` row. Three problems:

- **The rationale does not hold.** "No underwritten Year-0 baseline, so the column is not
  a measurement" — measured on live data, **all 12** affected deals had a complete
  At-Close measurement that foots to the cent. Nine Brainerd buildings and Pegasus lost
  real figures (Bldg E **1,662,811**, F(2) 954,662, Pegasus 624,689).
- **The trigger is a proxy.** Presence of a 2015-12-31 row is an artifact of MRI's export
  (626 of 108,963 rows, 52 of 134 deals, including deals that closed years later). The
  commit's own comment concedes this, then uses its absence as the signal.
- **`0` as a "no data" sentinel.** The page only shows a dash because `fmtMil()` treats
  `0` as `—`. The API returns a real `0`; Portfolio Snapshot needs a separate all-zero
  heuristic to guess, and Excel / assistant / any future report read zero economics.
  Should be `None`.

**Jim's domain input (Sep 1)**: for a development deal placing units into service,
expenses before revenue is expected — *the data is correct*. That also retires my own
earlier suggestion of triggering on "revenue == 0 with expenses > 0" as a defect test; it
is the normal shape, not a defect. Any suppression here is an editorial choice about what
the column is for, not a data fix.

**DECIDED (Jim, Sep 1 2026): LEAVE IT AS DEPLOYED.** The override stays in force on v407.
This is a settled decision, not an open item — do not revert it, narrow it, or re-raise it
without Jim asking. The accepted state is that At-Close reads as an em dash on all 12
deals, including the 10 whose underlying figures are complete and correct (9 Brainerd
buildings, Pegasus Life Storage). Anyone reconciling a Brainerd or Pegasus At-Close to
`at_close_noi` will find a real number behind a blank column — that is expected, not a bug
to chase.

Kill switch, recorded only so it is findable if that decision is ever revisited:
`AT_CLOSE_REQUIRE_YEAR0_ROW = False` reverts exactly.

## The architectural note worth acting on

Two commits (`568e9eb`, `50695d9`) explicitly place their logic *after* the existing
branches "so it overrides a settled figure rather than adding a third branch." As a
blast-radius technique that is sound. As architecture it means **the wrong value is still
computed and then masked** — the derivation is never corrected, and every consumer that
doesn't go through that exact display path still sees the unmasked figure.

Six modules now branch on the dev classification: `config`, `one_pager`,
`portfolio_snapshot_financial`, `portfolio_snapshot_loan`, `portfolio_snapshot_operating`,
`valuation_service`. The one fix that would retire most of this debt is a **stabilisation
state on the deal** — pre-construction / in-construction / lease-up / stabilised, with a
first-revenue date — sourced once and read everywhere. Charlene names this exact fix
independently in two different constants' comments. Until it exists the special-cases will
keep accumulating, one page at a time.

Secondary, cheaper: populate `Investment_Strategy` (it is empty on live, so the dev
classification runs entirely off the `Lifecycle` proxy — which is why Pegasus is caught at
all, via `Lifecycle = "New Construction"`, one of only two such rows in the feed).


---

## Sep 2 2026 — a second day of her commits, reviewed the same way

Eight revisions shipped (v409-v416). Her work in that chain: `7dc7bd8` (Eastchase override
withdrawn, East Manchester kept after sale), `019b592` (Prompts A/B/C), and the two commits
behind v416.

**The pattern from Sep 1 held**: the great majority are genuine root-cause fixes carrying
live-data guardrails, and `7dc7bd8` is a good example — it REMOVES a hardcode (the Sep 1
Eastchase `GROUP_OVERRIDES` entry, added on a work order that had meant East MANCHESTER) and
replaces a would-be vcode entry with `SOLD_NA_CELLS`, a rule keyed on the sale date. Reviewed
and judged not a symptom repair; deployed.

**`eb7520d` / `2a3fabe` is the exception, and the standing rule worked.** `MANUAL_RATIO_SEEDS`
is a per-deal hardcode of six vcodes with hand-typed LTV / DSCR / Debt Yield. Charlene
labelled it a symptom repair herself and both commits ended "NOT DEPLOYED … goes to Jim
before any image is built" — which is exactly right, and is the behaviour the Sep 1 rule was
written to produce. It was flagged with the affected deals and figures, and **Jim took it
knowingly as a stopgap** so quarter-end reports could go out, with a standing daily reminder
to replace it. See the handoff and `capital_reversal_and_psc3.md`.

**What I could NOT verify, three times over**: her guardrails import `scripts/live_api.py`,
which is still not committed. `snapshot_financial_pdf_check`, `snapshot_pe_basis_check`, the
module self-tests and the 126-check script behind `eb7520d` all fail at import for anyone
else. Her measured before/after figures had to be taken on trust; I verified the code and
partial local assemblies instead. **Committing that file is the single highest-value thing
she could do for reviewability.**

**One habit worth passing back to her**: several before/after scripts assert "BEFORE the
feature was absent", which breaks as soon as the baseline moves forward — it produced three
false failures in one day. State the invariant about the AFTER side instead.
