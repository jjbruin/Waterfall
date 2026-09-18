# Treasury — the bank side of the close

**Live at `v494`.** Screen at `/treasury`, under Accounting. Service
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

## The four tabs

**Accounts** — every account, its entity, its GL cash account, and its position.
**Import** — the PNC activity CSV, one statement PDF, or a whole folder of them.
**Reconciliation** — the three-way tie, the matcher, and the reconciling items.
**Journal entry** — code the month, get the GL and IA upload files.

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

- `scripts/treasury_reconciliation_check.py` — 80 locally against the real
  files, 54 on production where those files are absent (the file-dependent
  sections skip by design).
- `scripts/treasury_upload_check.py` — 58. **Rebuilds both accepted MRI
  upload files from their own contents and asserts byte-identical output.**
  A format check written from a specification proves only that the code
  agrees with itself; this one caught the amount formatting (`13313.8`, not
  `13313.80`).
- `scripts/treasury_api_check.py` — 37. **Asserts every field name the screen
  reads against a live response.** This exists because the service's own checks
  structurally cannot see that seam: a field read by the wrong name renders as a
  blank cell with no error and no log. It happened three times while the view
  was being written (`opening`/`opening_balance`, `net_movement`/
  `bank_movement`, `imported`/`inserted`) and found three real defects.

## The journal entry tab — coding the month

Fourth tab. One row per bank transaction, which is how a month is actually
coded: measured from the real August file, all thirteen investor distributions
arrive as **individual bank debits** (285.92 seven times, 571.84 twice), so the
common case is naming an account per line, not splitting a lump.

**THE CASH SIDE IS NEVER TYPED.** Each bank transaction becomes its own cash
line at the amount the bank reported; the accountant supplies only the OFFSET.
The entry therefore balances **by construction** rather than by arithmetic, and
a coded month cannot silently disagree with the bank. A partly coded month does
not balance, which is what keeps the download disabled.

**The split proposal is per row, for the other case** — one payment covering
several investors. Basis is `SPLIT_BASIS`: commitment AMOUNTS, not the stored
`CapitalPercent`. See below; this is the part most likely to be "simplified"
back by someone who has not measured it.

Downloads: `/upload/gl` (CSV) and `/upload/ia` (the template copy), both gated
to `ACCOUNTING_ROLES`. `/upload/preview` is open — it totals what is on screen
and says whether it balances, storing nothing. **Nothing is saved**: a
half-coded journal entry is a draft, not a record.

## Why the split uses commitment amounts

`CapitalPercent` is held to four decimals and, for AMB6, sums to **99.9999**.

| basis | result |
|---|---|
| stored percentages | 12,580.50 — three cents over, wrong on **5 of 13** investors |
| commitment amounts over their base of 11,000,000 | **all thirteen to the cent, ties exactly** |

The percentages are a rounded VIEW of the amounts; the amounts are the fact.
`treasury_upload_check.py` asserts the percentage basis *would* be wrong on
exactly five, so the shortcut cannot be reintroduced quietly.

**It reads `commitments`, not `relationships`.** AMB6 has FIFTEEN relationship
rows for thirteen investors — PSC1 appears twice, once at a closed 100% that
ended 2026-06-30 before the restructure, and PSCMAN sits at 0%. Summing that
column doubles one investor and invents a fourteenth.

`allocate()` floors every share then hands leftover cents to whoever was
rounded down hardest, ties broken on investor id so the same inputs always give
the same file. The remainder is placed deliberately and reported per row.

## Statements in bulk, and starting the chain from them

Jim, Sep 17 2026, asking whether to type fifty 6/30 opening balances or read
them off the June statements. **The statements win**: an external authority,
carrying their own arithmetic check, and the figure stays traceable to a named
file instead of to somebody's typing.

**SEEDING IS NOT RE-BASING**, and that distinction is the whole of
`seed_from_statement`. `opening_balance()` never reads a statement — it carries
the prior close, so a break surfaces as a difference instead of being papered
over. Starting the chain is the one case where a statement IS the right source,
because there is nothing behind it to contradict. So it **refuses the moment a
period has actually been reconciled**, and says why. The note it writes names
the source file, replacing "opening balance entered by hand".

Seed **202607**, not 202606: the 90-day activity export opens 6/22, so June can
never be reconciled from it, while July and August are complete.

### The mask is not always a tail

PNC prints a masked account number and it is the ONLY routing key a statement
carries. `XX-XXXX-5765` hides the FRONT; **`790-XXXXX55` hides the MIDDLE.**
Reading "the last four visible digits" off the second gives `790` + `55` ->
`79055` -> `9055`, an account that exists nowhere — while the real 7900021255
sits in the table untouched. That reported five of Jim's June statements as
unknown accounts, so the cause looked like missing data rather than a bad
heuristic: the kind of error that gets "fixed" by entering data that was never
missing, creating duplicates.

`_mask_pattern` reads the mask as what it is — each run of X is that many
unknown digits, each printed digit is itself, matched against the WHOLE number.
`790-XXXXX55` -> `^790\d{5}55$`. Being anchored also makes it stricter than a
tail match, so 8612199055 cannot collide with it. An **ambiguous** mask is
refused rather than resolved by picking one.

### A zero balance prints `.00`

Tested against the 64 real June 2026 statements before Jim relied on any of it.
**50 would not have filed, and 46 of those failed on one character**: PNC writes
a zero balance as `.00`, with no leading digit, and the money pattern required
one. It threw out rows carrying real amounts too — `30,832.24 .00 11,712.76
19,119.48` was refused for the single `.00` in the credits column.

14 filed before the fix, 45 after, 50 once the mask bug went too.

Also fixed there: a **trailing minus** is PNC's overdraft notation, and read as
positive it gives a WRONG balance rather than a refusal, so statement figures go
through `_money` — kept separate from `_num` because the activity export carries
its sign in a Credit/Debit column and never a trailing minus. And the four
figures are now read from **under the summary header**, because loosening the
number pattern made "the first four money figures in the document" unsafe.

### Registering an account by hand

`create_account`. An account normally registers itself on its first activity
import, which is no help when **PNC will not serve activity beyond 90 days** and
an account has been quiet longer: its June statement shows real money and no
transaction anywhere introduces it.

**THE FULL NUMBER IS TYPED AND NEVER INFERRED.** The statement prints only a
mask and several of these sit in obvious number ranges, so guessing the hidden
digits would usually work and would occasionally be wrong — and a wrong number
silently splits one account in two the moment real activity arrives under the
true one. A masked number is refused as input.

## Not done yet

1. **Statements as workpaper exhibits** — attach the imported PDF to the
   related workpaper.
3. **The automated PNC feed.** Researched Sep 17 2026: PINACLE Connect has a
   fixed ERP connector list; `developer.pnc.com` offers OAuth 2.0 + mTLS APIs
   gated behind a Relationship Manager / Treasury Management Officer
   conversation; BAI2 over SFTP is the fallback and is how most firms this size
   do it. **No screen-scraping of PINACLE** — credentials stay with PNC's own
   mechanism. A TMO email was offered and not yet requested.
