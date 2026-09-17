# Treasury — the bank side of the close

**Live at `v490`.** Screen at `/treasury`, under Accounting. Service
`flask_app/services/treasury_service.py`, API `flask_app/api/treasury.py`, view
`vue_app/src/views/TreasuryView.vue`.

Built Sep 17 2026 from Jim's real August AMB6 files: a PNC statement PDF, a PNC
activity CSV export, MRI's bank reconciliation screen, and the GL/IA upload
templates. **Every figure below was measured from those files before any code
was written**, which is why the module knows what a correct August looks like.

```
beginning balance (PNC statement)     571,750.04
net movement (PNC activity export)   -560,022.54
computed ending                        11,727.50
ending balance (PNC statement)         11,727.50   ties
MRI's September reconciliation opens   11,727.50   carries forward
```
24 bank transactions pair 1:1 against 24 GL cash lines.

## The three tabs

**Accounts** — every account, its entity, its GL cash account, and its position.
**Import** — the PNC activity CSV and the statement PDF.
**Reconciliation** — the three-way tie, the matcher, and the reconciling items.

## What the accounts tab can and cannot say

**CURRENT LEDGER is carried, not read.** It is the last closed period's computed
ending plus every transaction imported since, and the row says which period it
came from and what date the activity runs through. It is a position computed
from what the app holds, **not a reading of PNC's ledger balance**.

**CURRENT AVAILABLE IS `None` AND THE SCREEN SAYS WHY.** Available is ledger
less holds, float and pending debits — facts that exist only at the bank and
appear nowhere in an activity export. It is the one number a treasurer acts on,
so filling that column with the ledger figure would be inventing it. The column
exists because Jim asked for it; it fills in when the PNC connection does.

**An account with nothing closed has NO ledger figure, not `0.00`.** A real zero
balance and an unknown one are different facts. Guardrail asserts both.

## The three-way tie

`reconcile()` reports each leg **separately** — opening + bank movement against
the statement, and against the ledger. Rolling them into one difference would
hide which leg disagrees, and which leg it is happens to be the only thing the
number is for. `ties_to_statement` is `None` when no statement has been filed,
never `False`.

**The opening balance chains deliberately.** A month that has never been closed
has NO opening figure and says so. Opening at zero would report the whole
balance as a difference; silently re-basing on a fresh statement figure would
hide a break in the chain. Jim, Sep 17 2026: "Carry the prior period's computed
ending forward." `seed_opening()` starts the chain for a first period.

## The matcher

**PAIRED, NOT SET-COMPARED.** August carries the amount `285.92` seven times;
comparing sets of amounts would call all seven matched the moment one was. Items
are consumed as they are used. Within an amount the nearest date wins, and a
pairing further apart than `NEAR_DAYS` (5) is still offered but flagged.

**What is left over IS the reconciliation** — ledger entries with no bank line
are deposits in transit or outstanding payments; bank lines with no ledger entry
are activity not yet recorded. Both lists are returned in full, named the way an
accountant names them, rather than counted.

A **manual pairing outranks the matcher** and is never re-decided: the
accountant looked at both sides, the matcher only looked at the amount.

## Cash accounts

`MR10005000` is the default — the generic cash account for an entity with a
single bank account (Jim's call). `MR10006000`, `MR10007000`, `MR10008000`,
`MR10008100` are the dropdown. An account not in that list is refused.

An account **registers itself** the first time its activity is imported, before
anybody maps it to an entity: an unmapped account is a question to answer on
screen, not a row to leave out. An unmapped account's match explains itself AND
still carries the bank side, so "map this account first" is actionable.

## Refusals that are not silence

- A file that is not an activity export is **refused**, not reported as "0
  transactions imported". The column check runs BEFORE the row count, because a
  non-export often parses to an empty frame — a refusal dressed as a successful
  no-op is how a month goes missing unnoticed.
- Rows that could not be read come back in the response and are listed on
  screen. A transaction dropped in silence makes a period tie for the wrong
  reason.
- A statement whose own figures do not agree is not stored, and the refusal
  carries what WAS read — a scanned PDF is a different problem from a statement
  that does not foot.
- Re-importing the same export adds nothing (identity is a row hash), because
  re-pulling a month after a correction is normal.

## Guardrails

- `scripts/treasury_reconciliation_check.py` — 51 locally against the real
  files, 17 on production where those files are absent (the file-dependent
  sections skip by design).
- `scripts/treasury_api_check.py` — 37. **Asserts every field name the screen
  reads against a live response.** This exists because the service's own checks
  structurally cannot see that seam: a field read by the wrong name renders as a
  blank cell with no error and no log. It happened three times while the view
  was being written (`opening`/`opening_balance`, `net_movement`/
  `bank_movement`, `imported`/`inserted`) and found three real defects.

## Not done yet

1. **GL and IA upload template generation** from the accountant's coding. Jim
   has the templates and will supply them. Nothing here posts to MRI.
2. **Statements as workpaper exhibits** — attach the imported PDF to the
   related workpaper.
3. **The automated PNC feed.** Researched Sep 17 2026: PINACLE Connect has a
   fixed ERP connector list; `developer.pnc.com` offers OAuth 2.0 + mTLS APIs
   gated behind a Relationship Manager / Treasury Management Officer
   conversation; BAI2 over SFTP is the fallback and is how most firms this size
   do it. **No screen-scraping of PINACLE** — credentials stay with PNC's own
   mechanism. A TMO email was offered and not yet requested.
