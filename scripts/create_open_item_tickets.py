#!/usr/bin/env python
"""Create one Feedback & Requests ticket per open item in .claude/memory/open_items.md.

WHY A SCRIPT AND NOT A DIRECT WRITE: `user_requests` has no delete path — not in
`flask_app/api/feedback.py`, not in `feedback_service.py`. A ticket created here is
permanent short of hand-written SQL. So this runs DRY BY DEFAULT and needs an explicit
--commit, and it is idempotent: it reads the existing tickets first and skips any whose
title already exists, so a second run adds nothing.

AUTH — you supply it, the script never asks for or stores a credential:

    # preferred: through the app, so it gets reply tokens and normal app behaviour
    export WATERFALL_API=https://app-waterfall-dev-v2.icyplant-026fb2db.eastus.azurecontainerapps.io
    export WATERFALL_TOKEN=<paste a JWT from your logged-in browser session>

    python scripts/create_open_item_tickets.py            # dry run — prints what it would do
    python scripts/create_open_item_tickets.py --commit   # actually creates

Get the JWT from the browser: DevTools > Application > Local Storage > the app origin >
whatever the auth store holds. It is short-lived; that is fine, this runs once.

Tickets are attributed to whoever the token belongs to. Run it as yourself.

--only <n>  create just one ticket by number, to check the shape before the rest.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

# --------------------------------------------------------------------------------------
# The tickets. Each description is the six-line brief the /open-items skill uses:
# what's wrong / evidence / why it matters / recommended fix / the trap / how to verify.
# Numbered §-refs point back into .claude/memory/open_items.md for the full entry.
#
# `n` IS A STABLE LABEL, NOT AN INDEX — gaps are deliberate. 13, 14 and 15 were resolved on
# Sep 11 2026 (the cbui JWT had long since expired; the committed wfadmin password was
# purged and rotated; the Excel serials are gone from live data — see open_items.md §4) and
# their numbers were retired rather than reused, so "ticket 7" still means the same thing it
# meant in the conversation where these were reviewed. Renumbering would silently re-point
# every earlier reference. New items take the next unused number.
# --------------------------------------------------------------------------------------

TICKETS: list[dict] = [
    # ---------------------------------------------------------------- decide first
    dict(
        n=1, type="analysis", priority="high",
        title="DECIDE FIRST: which metric is U/W ROE meant to be? (open_items 2.1)",
        body="""ANSWER THIS BEFORE ANY OTHER U/W ROE ITEM — it gates most of them.

What's wrong: the source U/W workbook and the app compute different metrics, not
different denominators. The workbook is H18 = H15/H7 — single-period distributable cash
flow over that period's equity balance. Not time-weighted, not inception-to-date, not
annualised. metrics.calculate_roe() is ITD dollar-day weighted AND annualised.

Evidence: verified against 'USE THIS The Gathering at University Village Underwriting (3)
v03.xlsm', sheet 'PSC Investor Equity Structure'. G18 = 3.6% (yr 1), H18 = I18 = 9.0%.

Why it matters: they agree on The Gathering only because its underwritten steady state is
flat (-31,551.35/mo). On ANY deal with a lumpy 7071 schedule the two definitions diverge
even with a perfectly correct denominator. Scope: every deal carrying 7071 (49 vcodes).

Recommended: pick the target and write it down. If the One Pager is meant to reproduce the
investor workbook, the app needs a per-period metric, not a corrected ITD one.

The trap: fixing denominators first feels productive and cannot converge — you would be
tuning one metric to match a different one.

Verify: state the chosen definition in open_items.md 2.1 and close it; the dependent items
(2.7 CF-after-payoff, 1.6 pro-rate) can then be scoped against it.""",
    ),
    # ---------------------------------------------------------------- code gaps
    dict(
        n=2, type="error", priority="high",
        title="Account 7076 (Tenant Improvements) is completely unmodeled (open_items 1.1)",
        body="""What's wrong: 7076 is in no account set, and reporting.py aggregates by explicit set
membership with no catch-all bucket. TI therefore never reduces NOI, FAD, distributable
cash, or the waterfall — anywhere.

Evidence: `grep 7076 config.py` returns nothing (re-verified Sep 11 2026). Not in
CAPEX_ACCTS, ALL_EXCLUDED, EXPENSE_ACCTS or REVENUE_ACCTS.

Why it matters: Camp Creek (P0000075) ignores $4,024,004 across 2026-2036 against modeled
CapEx 7050 of only $2,902,452 — the unmodeled line is 1.4x the modeled one. 4.6% of
cumulative NOI, peaking at 10% of NOI in 2035 ($902,011) as rollover hits. 37 deals carry
7076: Woodlands Square $3.56M, Poplar Prairie $2.70M, Donald Lynch $2.69M, Merle Hay
$2.68M, Deptford $2.60M, Evergreen $2.51M.

Recommended: add to CAPEX_ACCTS in config.py — not a flat deduction. Reserve funding
(capex_paid in cash_management.py) and sign normalisation (normalize_forecast_signs forces
-base.abs() for ALL_EXCLUDED) then come free, and it is one line.

The trap — TWO QUESTIONS FIRST, do not code before answering:
 (a) the sources disagree 2x. Valuation IS says Camp Creek $8.20M; forecast_feed (priority
     1, which wins) says $4.02M. Which is right?
 (b) is 7075 Reserves for Replacement already the intended funding source, or does a lender
     TI/LC holdback pay it? Either would make naive inclusion a double-count.

Verify: almost no actuals exist (1 Interim IS row portfolio-wide), so this moves projections
only — history and realised returns must be byte-identical. Diff a full deal computation
either side and confirm nothing pre-actuals-boundary moves.""",
    ),
    dict(
        n=3, type="error", priority="medium",
        title="ROE_Income is set in SQL and read by no Python — $23.6M apart (open_items 1.2 + 2.2)",
        body="""What's wrong: queries/accounting_feed.sql:43 sets a ROE_Income flag marking TypeID 1019
(Preferred Return) and 1020 (Excess Cash flow). No Python reads it. The code re-derives the
same concept from Typename and disagrees.

Evidence: `grep -rn ROE_Income --include=*.py` returns zero hits (re-verified Sep 11 2026).

Why it matters: $23,599,656 apart on deal scope — code's Typename rule $169,654,101 vs
ROE_Income='Y' $146,054,445. The code is a strict superset; the extra is Distribution:
Income (156 rows, +$27.2M), Distribution: Tax, Professional Fee Holdback and Non Resident
Withholding.

Recommended: this is a definition question before it is a code change. Decide whether
'Distribution: Income' is operating income for ROE purposes. Then either read the flag or
delete it from the SQL — having both, disagreeing, is the worst of the three states.

The trap: none known, but note the flag and the Typename rule are not a subset/superset by
design — nobody chose this, it drifted.

Verify: whichever wins, one definition should exist in one place and the other should be
gone.""",
    ),
    dict(
        n=4, type="error", priority="medium",
        title="Four capital-event classifiers disagree on $73.6M of Realized Gain (open_items 1.3 + 2.3)",
        body="""What's wrong: four code paths classify the same accounting rows four different ways.

Evidence (all four still distinct, re-verified Sep 11 2026):
 - One Pager ROE / ROE Summary     -> Typename string  -> Realized Gain = capital event
 - Deal Analysis ROE Audit, ptr ROE -> Capital flag     -> Realized Gain = OPERATING INCOME
 - Sold Portfolio                   -> Typename (deliberate; CLAUDE.md notes the flag is
                                       unreliable at sale)
 - Pref Balance Detail              -> TypeID 1019/1020 -> narrowest

Why it matters: $73,574,410 of Realized Gain (41 rows, deal scope) is a capital return on
one tab and operating income on another. 30 Bearfoot legitimately shows two different ROEs:
Deal Analysis 24.62% vs One Pager 22.02%.

Recommended: move the Deal Analysis path off the `Capital` flag onto Typename so it agrees
with the One Pager and Sold Portfolio. Sold Portfolio already made this choice deliberately
and documented why.

The trap: 30 Bearfoot's two ROEs are not ONLY this — Deal Analysis includes projections and
the One Pager is actuals-only. Do not expect them to converge fully; expect the Realized
Gain treatment to stop being a second, hidden reason.

Verify: enumerate the 41 rows before and after and confirm only the classification moved.""",
    ),
    dict(
        n=5, type="error", priority="medium",
        title="U/W ROE pro-rate branch keeps one month of a multi-month YTD (open_items 1.6)",
        body="""What's wrong: the pro-rate branch in one_pager.py divides a first-period cumulative by the
month number, which is right for a real YTD and wrong for a single month's figure.

Evidence: swept every 7071 deal — 49 vcodes have data, 40 computable, 32 move if the branch
is removed. The discriminating test (period cumulative vs the next month's delta) says 31 of
those 32 are a single month, i.e. the pro-rate is wrong on them.

Why it matters: $2,443,476.63 of numerator restored ($2,027,032.18 excluding the bad
Westbank hit). Movers include 870 DLB 28.2011 -> 28.6713% (+0.47pp, largest), Berger
Pittsburgh, Giant 7, Camp Creek, Mount Prospect, 30 Bearfoot.

Recommended: remove the branch PLUS a guard — if a first-period cumulative exceeds ~2x the
next month's delta, log it and fall back to the current behaviour.

The trap — DO NOT JUST REMOVE IT. Centre at Westbank (P0000010) is the one deal where
removal makes things worse: its 2022-03-31 cumulative is 10.72x run rate, a stale
prior-year value at the wrong date, and removing the branch over-counts by ~$416,444. The
alternative is fixing the MRI extract (Westbank's Jan/Feb 2022 rows are simply missing) —
see the separate data ticket.

Verify: Asbury Commons (P0000004) is untestable — one row in its whole 7071 series, impact
$606.66. Expect it unchanged either way.""",
    ),
    dict(
        n=6, type="error", priority="medium",
        title="Berger Pittsburgh silently drops six of its eight child loans (open_items 1.5)",
        body="""What's wrong: a portfolio parent whose child loans have DIFFERENT terms renders a primary
and a second loan and drops the rest without saying so.

Evidence: _loans_share_terms() exists and correctly handles the co-terminous case (Burton).
Berger's 8 child loans carry 6 distinct term sets (senior ~2.9% plus mezz ~7.3% per
property), so it fails that guard — correctly — and falls back to the primary/second
selection rule, which was written for a single property with a real capital stack.

Why it matters: 6 of 8 loans invisible on an investor-facing page. Berger is the known
case; any parent with differing child loans has the same hole.

Recommended: needs a display decision before code — what SHOULD a parent with six distinct
term sets show? Options: a count with a link, the largest N, or a roll-up weighted rate.

The trap: do not widen _loans_share_terms() to cover this. It is doing its job; the gap is
that no display exists for the differing-terms case. Widening it would break Burton.

Verify: 11 genuine primary + second-tranche deals must keep their second loan (P0000003
3.20%/5.94%, P0000069 5.35%/SOFR+2.50%, ...). They are excluded by the same-terms guard, not
special-cased — keep it that way.""",
    ),
    dict(
        n=7, type="error", priority="medium",
        title="Quarter dropdown is portfolio-wide and can have holes (open_items 1.4)",
        body="""What's wrong: the One Pager quarter dropdown lists quarters from the whole portfolio, not
from the deal being viewed.

Evidence: one_pager.py:87 — get_available_quarters(isbs_df) takes no vcode and applies no
per-deal filter (re-verified Sep 11 2026).

Why it matters: one deal's newly loaded quarter appears on EVERY deal's dropdown, and the
list can skip quarters — observed offering Q3 but not Q2, because no deal had Q2 actuals.
Confirmed behaviour, not theory.

Recommended: take a vcode and filter to that deal's own reported periods.

The trap: this is the SIBLING of the first-load default bug, which is already fixed
(54b4700). That fix made the label and the data agree; it did not make the list per-deal. Do
not assume the area was handled.

Verify: a deal with no Q2 actuals should not offer Q2. Re-check that the first-load default
still lands on the most recent COMPLETED quarter after the change.""",
    ),
    dict(
        n=8, type="error", priority="medium",
        title="'contrib' MajorType match is over-broad — expenses land in funded capital (open_items 1.7 + 2.5)",
        body="""What's wrong: `"contrib" in major_type` sweeps all eight contribution TypeIDs, so
non-capital items are counted as funded capital.

Evidence: live at compute.py:298 and reports_service.py:355, 451, 592, 1101, 1145
(re-verified Sep 11 2026).

Why it matters: Partnership Expenses (400 rows), Management Fees (34) and Organizational
Costs (2) land in funded_to_date AND in the ROE denominator, on both the ROE and
capitalisation paths.

Recommended: decide first whether that is wrong. It may be intentional — these are real
cash the partner put in. If it is wrong, match on the specific TypeIDs rather than a
substring.

Related question in the same area: do DEVELOPMENT deals report funded-to-date or closing
capitalisation? Affects Belleville, JB Fair Park, Trolley Square, Brainerd, Pegasus.

The trap: TypeID 1018 is not unique — it is 'Contribution: Investments' on the contribution
side AND 'Non Resident Withholding' on the distribution side (33 rows). TypeID alone is not
a key; pair it with MajorType.

Verify: enumerate what moves out of funded_to_date before changing it — the figure is
investor-facing.""",
    ),
    dict(
        n=9, type="improvement", priority="low",
        title="Pegasus: the TGA22 JV entity is bucketed into Pref equity (open_items 1.8 + 2.6)",
        body="""What's wrong: pref-vs-partner bucketing is a string prefix test, so an entity that is
neither an OP nor a pref investor lands in pref by default.

Evidence: one_pager.py:894 and :926 (also :2533, :2547) bucket on
InvestorID.str.startswith("OP"); everything else is Pref (re-verified Sep 11 2026).

Why it matters: Pegasus Life Storage has three investors — OPPEGA, PPILFS and TGA22. Only
OPPEGA starts with OP, so TGA22 — the PSCKOC JV entity — is counted as Pref equity, changing
how Pegasus's cap stack splits pref vs partner.

Recommended: confirm the intended treatment first. If TGA22 should not be pref, the fix is a
real entity-role lookup rather than a prefix test.

The trap: the prefix test is load-bearing in several places. Changing it globally will move
more than Pegasus — scope the blast radius before touching it.

Verify: Pegasus pref/partner split before and after; every other deal unchanged.""",
    ),
    # ---------------------------------------------------------------- decisions
    dict(
        n=10, type="analysis", priority="medium",
        title="Brainerd and OREI equity basis — a definition question, not a bug (open_items 2.4)",
        body="""What's wrong: nothing, possibly. Both deals are zero-delta on every MECHANISM tested — not
the sign bug, not a date filter, not child properties. What remains is a basis question only
the model can answer.

Evidence: inv34b basis-variant sweep, run against live Azure PG.

The two questions:
 - BRAINERD: does the model include `Contribution: Others` $4,550,000? Azure shows
   12,007,677; excluding it gives 7,457,677.
 - OREI: does `Contribution: Operating Capital` $1,233,899.26 count as partner equity? Azure
   shows pref 13,391,868 / partner 8,124,512; excluding gives 10,786,868 / 6,890,613.

Recommended: answer both, then the figures either reconcile or a real defect is isolated.

The trap: OREI has NO `Contribution: Others` rows at all — that Typename is Brainerd's only.
An earlier note conflated them; do not go looking for Others on OREI.

Verify: whichever basis is chosen, apply it consistently to the ROE denominator and the cap
stack — they are built on different populations today.""",
    ),
    dict(
        n=11, type="analysis", priority="medium",
        title="Should CF received after capital is fully repaid count in ROE? (open_items 2.7)",
        body="""What's wrong: possibly nothing — a policy question.

The question: when capital has been fully returned and distributions keep arriving, they add
to the ROE numerator while adding nothing to the denominator. Should they count?

Evidence / scope — 4 deals: Berger Pittsburgh $2,993,146 (paid off 2024-07-10), 30 Bearfoot
$160,612, Willowdale $38,318, Barnbeck $27,000.

Why it matters: Berger is the material one. On 30 Bearfoot this behaviour is worth about
-2.75pp of the ~4.2pp gap between the app and the audit figure.

Recommended: decide the policy, then apply it in one place.

The trap: this interacts with item 1 (what metric U/W ROE is). A per-period metric makes the
question nearly moot; an ITD annualised one makes it material. Answer item 1 first.

Verify: the 4 deals above are the whole affected population — confirm no fifth appears after
any change.""",
    ),
    dict(
        n=12, type="analysis", priority="medium",
        title="Projected YE drops months — confirm whether it is even a defect (open_items 2.8)",
        body="""READ THIS BEFORE 'FIXING' IT — it may be correct behaviour.

What's observed: months between a deal's last actual and the selected quarter-end fall into
neither window. YTD Actual stops at the last actual; remainder-Budget starts AFTER
quarter_end; nothing backfills the gap.

Evidence: p0000007's Projected YE NOI falls 13.4M -> 9.3M -> 5.3M as the dropdown advances.
81 of 81 deals in the Apr snapshot have actuals ending before 2026-06-30.

Why it matters: it looks alarming and is easy to 'fix' into something wrong.

Recommended: first confirm whether loading the missing actuals picks those months up. If it
does, this is a data-currency artifact and there is nothing to fix.

The trap: the original investigator explicitly flagged this as NOT a confirmed bug — budget
starting after the selected quarter with no backfill may be by design. Do not code against
it until someone confirms the intent.

Verify: pick one deal, load its missing actuals, re-read Projected YE.""",
    ),
    dict(
        n=13, type="error", priority="medium",
        title="A reset password is the literal string 'password', emailed in plaintext (open_items 1.9)",
        body="""What's wrong: there is no admin "set a password" endpoint. The only admin reset path is
POST /auth/users/<id>/send-welcome (the "Send Welcome" button in Settings > Users), and it
sets every user to the same hardcoded literal.

Evidence: flask_app/auth/routes.py:358 — `temp_pw = "password"` (verified Sep 11 2026). It
then flags must_change_password and EMAILS the password in plaintext.

Why it matters: the value is identical for every user and every reset, so it is guessable
by anyone who has ever been onboarded. must_change_password narrows the window to that
user's next login — it does not close it, and the email persists in a mailbox indefinitely.

Recommended: generate a random temporary password per reset, and make the existing
/auth/forgot-password flow (one-hour single-use token, no password in the email) the
default for an EXISTING user. send-welcome then only matters for genuine onboarding.

The trap: `change_password(user_id, temp_pw, clear_must_change=False)` and the
must_change_password UPDATE are two separate statements. If the password becomes random,
a failure between them leaves an account on a password nobody knows.

Verify: reset a test user twice and confirm the two temporary passwords differ, and that
neither appears in the email body.""",
    ),
    dict(
        n=14, type="error", priority="medium",
        title="A leaked JWT cannot be revoked — it stays valid up to 24h (open_items 1.10)",
        body="""What's wrong: there is no way to invalidate a single user's token.

Evidence (verified Sep 11 2026): JWT_EXPIRATION_HOURS = 24 (flask_app/config.py:15), HS256
signed with JWT_SECRET. A repo-wide grep for revoke|blocklist|blacklist|token_version
returns NOTHING.

Why it matters: tokens are self-contained — nothing re-checks the password on a request —
so changing a user's password does NOT invalidate their existing token. The only lever is
rotating JWT_SECRET, which signs out every user at once. Any leaked token (pasted in chat,
captured in a log, copied from DevTools) is live for up to 24 hours with no intervention
possible.

This is exactly what made the Aug 6 `cbui` incident unanswerable at the time: nothing could
be done, and nothing recorded that. That token has long since expired on its own.

Recommended: a `token_version` integer on `users`, included in the JWT payload and compared
on decode. Bumping it kills that user's tokens only, and a password change can bump it
automatically. Turns "wait it out" into an action.

The trap: /auth/me and every @login_required route decode on each request, so the
comparison needs the user row — measure the cost before adding a query per request.

Verify: issue a token, bump token_version, confirm the token is rejected while another
user's still works.""",
    ),
    dict(
        n=22, type="error", priority="high",
        title="An approved valuation gives no sign it is still unpublished (open_items 1.11)",
        body="""What's wrong: committee_approve sets status='approved' and freezes a snapshot. It does
NOT write to the valuations table — that is publish_record, a separate step. Nothing in the
UI says so.

Evidence: verified Sep 11 2026. The status reads "approved", which sounds finished, and the
published figure silently never reaches the One Pager.

Why it matters: valuations are annual and low-volume, so a record can sit
approved-but-unpublished indefinitely with nothing flagging it. This cost a full afternoon
on Sep 11 — the record read approved, the One Pager kept showing the prior year, and
nothing connected the two.

Recommended: surface "approved, not published" on the cycle dashboard and mark it on the
record. Auto-publishing on final approval is the alternative, but the separation looks
deliberate, so showing the gap is the safer change.

The trap: `published_at` is the field that answers this, not `status`.

Verify: an approved, unpublished record is visibly distinct from a published one without
opening it.""",
    ),
    dict(
        n=23, type="error", priority="medium",
        title="publish_record can publish a NULL valuation and report success (open_items 1.12)",
        body="""What's wrong: the guard is `nav = get_nav(...)` then `if not nav: raise`. `nav` is a DICT,
which is truthy even when nav["value"] is None — so the insert writes NULL into
mIncomeCapConcludedValue and the publish reports success.

Evidence: flask_app/services/valuation_nav_service.py, publish_record (verified Sep 11 2026).

Why it matters: a published row with no value is indistinguishable from a successful
publish. Every consumer then correctly falls back to the previous year (see the
blank-column fix), so the publish "worked" and nothing changed — which is a very expensive
thing to debug.

Recommended: refuse the publish when nav.get("value") is not a positive number, with the
same message shape as the existing "Compute the NAV before publishing".

The trap: 0 must be refused as well as None. A valuation of zero is not a measurement —
that is the rule applied everywhere else in this queue.

Verify: publishing a record whose NAV has no value raises instead of writing.""",
    ),
    dict(
        n=24, type="improvement", priority="medium",
        title="valuation_records has no structured cost basis (open_items 1.13)",
        body="""What's wrong: the table holds concluded_value and a free-text override_note, and nothing
else about how a cost-basis figure was built.

Why it matters: Town Fair's 12/31/2025 value is 33,910,000 against an Acquisition_Price of
30,750,000 — a 10.3% gap that LOOKS like a market write-up and is not one. "Cost" here
means purchase + capital prefunded for improvements + closing costs + accrued pref through
the reporting date. Anyone reconciling the two hits that wall, and the only thing standing
between them and the wrong conclusion is whatever a human typed in the note. The author of
this ticket reached the wrong conclusion on exactly this and had to be corrected.

Recommended: structured components on the record (purchase, improvements, closing costs,
accrued pref) so a cost basis explains itself and the total is checkable.

The trap: keep the free-text note — the components will not cover every case.

Verify: a cost-classified record shows its build-up without anyone having to ask.""",
    ),
    dict(
        n=25, type="analysis", priority="low",
        title="UNCONFIRMED: Pref Equity capitalization may truncate in print (open_items 1.14)",
        body="""READ BEFORE CHANGING ANY PRINT CSS — this defect may not be real.

What's reported: 9 deals lose the tail of their ownership split in print. P0000006 prints
"KOC 43%, PSC 41%, Declaration" and drops "16%"; P0000081 drops "F&F 12%".

Two reasons to doubt it:
 (a) scripts/onepager_print_geometry.py:146 `_covers()` documents this EXACT case as a
     reading-order artifact, naming "16%": "a value that re-wraps onto its own line moves
     in pdfplumber's reading ORDER … '16%' reads back as '%16' … an IDENTICAL character
     count is what identifies it as reordering rather than loss."
 (b) Tracing the chain found NO mechanism that would clip it: the textarea is genuinely
     hidden, the print twin has white-space: pre-wrap / overflow: visible / height: auto,
     and the cell has no overflow:hidden, no fixed row height, no nowrap. A table row grows
     to its tallest cell.

Recommended: settle it first — print P0000006 and look at the cell, or run the print sweep
(needs WF_TOKEN). Only then decide.

The trap: if it IS real, the likely fix is `table-layout: fixed` on .cap-table in print so
the declared 18% is enforced — but that changes every column on that table, so it needs the
sweep to confirm nothing else regresses.

Note: Jim's interim call was to have the asset manager abbreviate "Declaration" so it fits.
That removes the symptom on one deal; it does not answer whether the defect is real.""",
    ),
]

# ---- Jim's data / ops items. Same shape, different owner. -----------------------------
TICKETS += [
    dict(
        n=26, type="error", priority="high",
        title="DATA: five deals have a valuation but NO cap rate on any row (open_items 3.5)",
        body="""What's wrong: P0000021 ($14.5M), P0000085 ($67.8M), P0000089 ($46.6M), P0000100
($43.6M) and P0000110 ($8.4M) read a 0 cap rate because no valuation row for them carries
one at all.

Evidence: measured Sep 11 2026, after the blank-column fix removed every case where a real
cap rate existed on an older row. These five have nowhere to fall back to.

Why it matters: the Dashboard's weighted-average cap rate is
sum(cap_rate x valuation) / sum(valuation), so each of these puts its FULL valuation into
the denominator contributing nothing to the numerator. $181M of valuation is diluting a
reported KPI right now.

Recommended: fill in fCapRate for the five. This is a data fix, not a code one.

The trap: none — but expect the portfolio figure to RISE when they land, as it did
(+8.9 bps, 5.7969% -> 5.8859%) when the six partial rows were corrected. That is the gap
closing, not a market move.

Verify: none of the five reads 0 afterwards, and the weighted average moves once.""",
    ),
    dict(
        n=27, type="improvement", priority="high",
        title="No write path is exercised against PostgreSQL before it is needed (open_items 3.9)",
        body="""What's wrong: a process gap, and the cause of two of Sep 11's four deploys.

Evidence — both were invisible locally:
 * v435: unquoted mixed-case SQL. SQLite is case-insensitive, PostgreSQL is not, so the
   valuation publish path had NEVER ONCE SUCCEEDED against Azure and looked tested.
 * v436: refresh_table('valuations') invalidated a key nothing reads. It returned success.
   No error anywhere; a published figure simply never appeared.

Each was hidden behind the one before it, and neither was findable without running the real
path against the real database.

Why it matters: the publish path had a service function, an endpoint, a UI button and a
whole workflow around it. "Covered" is not "has ever run in production".

Recommended: a smoke test exercising the app's WRITE paths against a PostgreSQL instance —
publish a valuation, save a waterfall, add a capital call — asserting each change is
visible on a subsequent READ.

The trap: two guardrails now cover these specific classes
(scripts/sql_mixedcase_identifier_check.py, scripts/refresh_table_key_check.py) and both
fail when the bug is reintroduced. They are STATIC checks and cannot catch the next thing
that only breaks on the real engine.

Verify: the smoke test fails if either Sep 11 bug is reintroduced.""",
    ),
    dict(
        n=16, type="error", priority="medium",
        title="DATA: JB Fair Park balance sheet stops 6/30/2025; debt reads a 12/31/2022 row (open_items 3.2)",
        body="""What's wrong: a data gap, not a code bug.

Evidence: JB Fair Park's BS data stops 6/30/2025 while peers run to 6/30/2026, so cap['debt']
reads a stale 12/31/2022 row on account 2150 -> $66,363,992. Interest expense (5190/7030) is
0 in every period, consistent with nothing drawn.

Why it matters: a stale debt figure on an investor-facing cap stack.

Note the code is behaving correctly: get_isbs_debt_balance() DETECTS the staleness but keeps
the last-known balance because an active MRI loan exists (LoanID 335). That cross-reference
is what prevents a wrong $0 — do not 'fix' it.

Portfolio scan: only 3 of 83 deals are stale — JB Fair Park (30 months), Post Commons
(1 month, benign lag), Pegasus (21 months but no active MRI loan, so already forced to 0).

Recommended: backfill the balance-sheet data past 6/30/2025.

Verify: cap['debt'] picks up a current balance without any code change.""",
    ),
    dict(
        n=17, type="error", priority="medium",
        title="DATA: Centre at Westbank is missing its Jan/Feb 2022 rows (open_items 3.3)",
        body="""What's wrong: Westbank's 7071 series jumps from 2021-12-31 straight to 2022-03-31 with no
Jan or Feb rows, so the 03-31 cumulative reads as 10.72x run rate.

Why it matters: this single gap is the ONLY reason the U/W ROE pro-rate fix (separate
ticket) cannot simply remove the branch — without a guard it over-counts Westbank by
~$416,444.

Recommended: fixing the extract is the clean alternative to coding a guard. If the rows can
be recovered, the code fix gets simpler.

The trap: -624,666.67 at 2022-03-31 is exactly the full-year 2018/2019/2020 total
(12 x 52,055.56) — a stale prior-year value landing at the wrong date, not a real Q1 figure.

Verify: after the extract fix, Westbank's cumulative/run-rate ratio should land near 3 for a
genuine Q1, and the pro-rate fix can ship without its guard.""",
    ),
    dict(
        n=18, type="improvement", priority="low",
        title="One Pager snapshots frozen before the chart-window change need backfilling (open_items 3.6)",
        body="""What's wrong: snapshots approved before the chart-window change still hold the OLD sparse
quarter arrays, so an approved historical One Pager renders a different chart from the same
deal's live view.

Why it matters: an approved investor document that no longer matches what the app shows. Low
urgency, but it is a silent divergence.

Recommended: decide whether to backfill or to accept that a frozen snapshot is frozen — a
defensible position, since the snapshot is the record of what was approved.

The trap: backfilling rewrites approved documents. If that is done, record that it happened
and when.

Verify: pick one pre-change approved snapshot and compare its chart array length against a
current one (should be 10 quarters).""",
    ),
    dict(
        n=19, type="improvement", priority="low",
        title="Burton debt roll-up: the post-deploy check was never recorded (open_items 3.8)",
        body="""What's wrong: a verification gap, not a known defect.

The 0 -> $75,302,500 debt fallback for Burton (P0000109) only fires when ISBS returns NO
balance for that vcode — get_isbs_debt_balance() runs first and takes precedence. It was
verified locally with isbs_raw=None, so the fallback ran. On Azure, if ISBS carries a parent
balance, that value wins instead and may differ — including the possibility of a
JB-Fair-Park-style stale row kept alive by an active MRI loan.

The loan-TERM collapse is independent of ISBS and holds either way; only the debt figure is
in question.

Also: the guardrail swept the Apr-15 MRI_Loans.csv (78 loans, 110 deals). Azure carries ~24
more deals, so another parent with co-terminous child loans would also collapse — correctly,
but untested.

Recommended: open Burton's One Pager on Azure and read the debt line. Re-run
scripts/burton_loandump.py against PG for the wider sweep.

Verify: debt is a current figure, not 0 and not a years-old balance.""",
    ),
    dict(
        n=21, type="error", priority="medium",
        title="Audit logging coverage on the rest of the infrastructure (open_items 3.4)",
        body="""What's wrong: unknown, and that is the point. Nobody has checked.

Evidence (why this was raised): the Postgres server had log_connections ON but
logfiles.retention_days = 3, logfiles.download_enable = off, and NO diagnostic settings at
all — so when it mattered (a credential public for five months) there was nothing to read.
Fixed for Postgres Sep 11 2026: retention 3 -> 7, and a pg-logs diagnostic setting now
ships PostgreSQLLogs to workspace-rgwaterfalldev5uCa at 30-day retention.

Not checked: the container app app-waterfall-dev-v2, the registry acrwaterfalldev (who
pulled or pushed an image), the storage account, and the container app environment.

Why it matters: four Log Analytics workspaces exist in rg-waterfall-dev at 30-day
retention, but they were created automatically by Container Apps. Their existence is not
evidence that anything is being shipped to them.

Recommended: `az monitor diagnostic-settings list` per resource, and wire anything
security-relevant into the existing workspace the way pg-logs now is.

The trap: an empty log query reads like "nothing happened". It usually means the log was
never collected. Confirm a category is enabled AND flowing before treating its silence as
evidence of anything.

Verify: for each resource, a non-empty diagnostic-settings list naming an enabled category
and a real workspace.""",
    ),
    dict(
        n=20, type="analysis", priority="low",
        title="Debug_Progress.xlsx was never run (open_items 3.7)",
        body="""What's wrong: possibly nothing — an outstanding request from Aug 5 2026 that was never
actioned and never withdrawn.

Recommended: confirm whether it is still wanted. If not, close this and it comes off the
queue permanently.

The trap: none. The only cost here is carrying an open question nobody intends to answer.""",
    ),
]


# --------------------------------------------------------------------------------------

def _api(method: str, path: str, token: str, base: str, payload: dict | None = None):
    url = base.rstrip("/") + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:400]
        raise SystemExit(f"\nHTTP {e.code} on {method} {path}\n{body}\n"
                         f"(401 usually means the token expired — grab a fresh one.)")
    except urllib.error.URLError as e:
        raise SystemExit(f"\nCould not reach {url}: {e.reason}")


def main() -> int:
    # Windows consoles default to cp1252 and mangle the em dashes in these briefs.
    # Only affects what you SEE on a dry run — the POST body is UTF-8 regardless.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true",
                    help="actually create the tickets (default is a dry run)")
    ap.add_argument("--only", type=int, metavar="N",
                    help="create just ticket N, to check the shape first")
    args = ap.parse_args()

    todo = [t for t in TICKETS if args.only is None or t["n"] == args.only]
    if args.only is not None and not todo:
        raise SystemExit(f"No ticket numbered {args.only} (have 1-{len(TICKETS)}).")

    if not args.commit:
        print(f"DRY RUN — {len(todo)} ticket(s) would be created. "
              f"Nothing is written. Add --commit to create.\n")
        for t in todo:
            print("=" * 78)
            print(f"[{t['n']:>2}] {t['type']:<12} {t['priority']:<7} {t['title']}")
            print("-" * 78)
            print(t["body"])
            print()
        print("=" * 78)
        print(f"{len(todo)} ticket(s). Re-run with --commit to create them.")
        return 0

    base = os.environ.get("WATERFALL_API", "").strip()
    token = os.environ.get("WATERFALL_TOKEN", "").strip()
    if not base or not token:
        raise SystemExit(
            "Set WATERFALL_API and WATERFALL_TOKEN first — see the docstring at the top "
            "of this file. The script never prompts for a credential.")

    # Idempotency: user_requests has NO delete path, so never create a duplicate.
    existing = _api("GET", "/api/feedback", token, base)
    rows = existing if isinstance(existing, list) else existing.get("requests", [])
    seen = {str(r.get("title", "")).strip() for r in rows}
    print(f"{len(seen)} existing ticket(s) found; titles already present will be skipped.\n")

    created = skipped = 0
    for t in todo:
        if t["title"].strip() in seen:
            print(f"  skip  [{t['n']:>2}] already exists — {t['title'][:60]}")
            skipped += 1
            continue
        res = _api("POST", "/api/feedback", token, base, {
            "request_type": t["type"],
            "title": t["title"],
            "description": t["body"],
            "priority": t["priority"],
            "page_context": ".claude/memory/open_items.md",
        })
        rid = res.get("id") or (res.get("request") or {}).get("id", "?")
        print(f"  OK    [{t['n']:>2}] #{rid}  {t['title'][:60]}")
        created += 1

    print(f"\nCreated {created}, skipped {skipped}.")
    print("These cannot be deleted through the app — only status-changed to resolved/closed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
