# Reversed capital entries, and why PSC3 looks over-returned (Sep 2 2026)

## The reversal bug — fixed, see `scripts/capital_reversal_sign_check.py`

MRI reverses a booking by **re-posting it with the opposite sign under the same
MajorType and Typename**. Six places in this codebase used to read the row's
TYPE and then take `abs()` of the amount, which threw away the only thing
distinguishing a reversal from the entry it reverses, so the pair moved capital
by twice the amount instead of leaving it unchanged.

The rule now lives once, in `loaders.capital_after`: **the sign of the amount is
the direction.** Cash in is negative, cash out positive, so the change in
capital is the negated amount and a reversal reverses itself. The row's type
still decides *whether* it touches capital — never which way.

Two things that are easy to get wrong when touching this again:

* **Do not floor the running total per row.** JB Fair Park's reversal sorts
  BEFORE the contribution it reverses; a per-row `max(0.0, ...)` swallows the
  reversal and both remaining contributions land (7,700,000 against a true
  3,850,000). Floor at the point of use — `loaders.capital_outstanding`.
* **Do not add a magnitude heuristic.** The sign is authoritative; the size of
  the number is not.

Confirmation the netted reading is right: the 26Q1 reference PDF was produced
outside this app and prints the netted figures. JB Fair Park 3.9, Pegasus 2.6,
Cocoplum 23.4 and Cocoplum Total Cap 108.9 all tie after the fix and did not
before.

## PSC3 over-returns — ANSWERED, do not re-raise

The guardrail reports pairs whose capital account goes negative. Four of them
are PSC3 as the investor into a fund vehicle (INVF10, INVF5, INVF3, PSCPGH),
over-returning by 15-39%.

**Jim, Sep 2 2026:** PSC3 had a redemption event — PSC1 acquired PSCMAN from
PSC3, while the Investee Funds that had been owned by PSC3's *members* were
assigned from the members to PSC3 directly. The mismatch is journal entries
from that redemption and acquisition.

**It does not reach any reported number: returns are run for the individual
assets inside PSC3, never for PSC3 itself.** None of these entities is in the
`deals` table.

## A (deal, investor) pair is not always a capital account

Three more flagged pairs have **no contributions at all** at that grain —
PCBLE/PSC1, OWPSC/WOFC, PPI2/PSC2. PCBLE never receives contributions from
anyone: it is the promote / AM-fee vehicle, and its payouts are earned income
carried under the Typename **"Distribution: Return of Capital"**. OWPSC was
funded by six investors while WOFC, which draws from it, has contributed
nothing anywhere.

So "returns more capital than contributed" is only a meaningful measure for a
pair that was actually funded under that same pair. I asserted it as a finding
once before checking, and it was an artifact of my own grouping.

---

# MANUAL_RATIO_SEEDS — a stopgap with an expiry (Sep 2 2026)

`v416` (`eb7520d`) shipped six hand-typed LTV / YTD DSCR / Debt Yield figures in
`portfolio_snapshot_loan.MANUAL_RATIO_SEEDS` (P0000109, P0000116, P0000117,
P0000118, P0000119, P0000120). Jim approved it **knowingly, as a stopgap**, so
quarter-end reports could go out — not as a fix.

Since the same release the subtotals weight what the row *displays*, so those
typed figures now drive **investor-facing totals**: 26Q2 Portfolio LTV 63.9%,
Portfolio DSCR 1.75x, TGA 6 DSCR 1.27x. The sharpest single item is
Presidential Arms' typed **1.1x replacing a computed 3.8x**, which is what
moves TGA 6.

**What actually needs fixing** — the ratios are uncomputable for structural
reasons, none addressed:

* no valuation dated on or before the report year-end → `_latest_valuation`
  finds nothing → no LTV
* no full YTD Interim IS plus balance-sheet principal movement → no DSCR
* no complete three-month quarter of actual NOI → no Debt Yield

Retiring it is one deletion per deal: remove a vcode from the dict and its
cells revert to the engine. A weekday reminder runs at 09:00 under the
scheduled task `retire-manual-ratio-seeds`; delete that task once the dict is
empty.

**Known defect shipped with it:** the row's warning flag still reads "the
computed figures … still feed the subtotals", true when written in `2a3fabe`
and made false by `eb7520d` an hour later. Visible on the row tooltip.

**Not independently verified:** the 126-check guardrail behind this imports
`scripts/live_api.py`, which is still not committed, so the before/after
figures are Charlene's. See [[MEMORY]] — that gap has now blocked verification
on several changes.
