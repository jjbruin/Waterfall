# PSC Preferred Equity Exposure (Reports, open to everyone)

Moved verbatim out of CLAUDE.md on Oct 5 2026 by the compaction, the same day it was
written (`v565` = `032fa27`). CLAUDE.md keeps every RULE below as a one-line bullet
under Domain invariants; the narrative, the measurement against accounting's own
tracker and the full stops list are here.

Code: `flask_app/services/pe_exposure_service.py`, `vue_app/src/views/PeExposureView.vue`,
`/api/reports/pe-exposure[/quarters|/excel]`. Guardrail: `scripts/pe_exposure_check.py`.

---

### PSC Preferred Equity Exposure (Reports, open to everyone)
Built Oct 5 2026. Replaces accounting's `PSC Preferred Equity Tracker - <date>.xlsx`,
whose Spreadsheet Server (`GEXD("IA Query.edq", ...)`) links and typed percentages
become references to our MRI copy. `pe_exposure_service.py`, `PeExposureView.vue`,
`/api/reports/pe-exposure[/quarters|/excel]`.

- **Every figure is an existing engine's**: Cost = the Pref Balance Detail capital
  balance + realized losses (ia_transactions, below zero); FMV = Cost + unrealized
  marks; the investor split = `group_shares` over commitments in force; Future
  Funding = the One Pager's `remaining_to_fund`; CAD via `market_rates`.
- **Measured against accounting's 26Q2 tracker on production data**: Cost 52/53,
  FMV 53/53, the seven investor splits 51/53, grand total within 0.1%. Every
  difference is the tracker's own typed input -- Nottingham (its Cost view omits a
  $2.92M June contribution its FMV view includes), Brainerd (a typed funded-to-date
  split; its own side note gives ours), Bel Air (two typed constants).
- **The stops** (where the walk ends and what it is called) are accounting's
  classification, from the tracker's Mapping tab: PSC = PSC1/PSC2/OWPSC/PSCMAN/
  PSL1/PSS1, KOC = KCREIT, TIAA = TGAM, Declaration = DCXVIA/B, Clarion = DIFPP; an
  AMB fund's outside investors are Ambassadors; every other outside investor F&F.
- **Holders come from commitments** (non-OP investors into the deal), under the
  vcode `build_investmentid_to_vcode` maps the InvestmentID to -- InvestmentID is
  not unique (MCCORD has two vcodes) and the pref engine answers 0 under the other.
- **A sale booked as a realized loss** (Adirondack, City West) takes cost to 0; the
  pref engine alone would carry the full capital, since `accounting` has no
  non-cash rows.
- **IA is cut on TRANSACTION date**: 1,767 non-cash rows carry no Effective Date.
- **Live** is any date; Future Funding alone answers by quarter, and says so.
- Guardrail `scripts/pe_exposure_check.py` (28).

### Accounting's allocation overrides (built Oct 5 2026, NOT deployed)
**Status:** on `feat/pe-allocation-overrides`. Delete this line when it ships.

Accounting's review of the report raised three structures where MRI's ENTITY-level
commitment ratios misstate who funded ONE investment:

| Investment | Entity | Why |
|---|---|---|
| BRNERD | PPIBPA and INVBPS | Restructured mid-hold; the ratios were modified and the commitments are not yet fully funded |
| JBFAIR | PSC3 | DCXVIA/DCXVIB opted out of it and their initial commitment was returned |
| NOTTNV | PSC3 | An increase funded by all except DCXVIA/DCXVIB; OWPSC funded their share |

The rules (accounting's answers):
- **Entered as AMOUNTS funded.** The share is amount / total.
- **Applies on the effective date and after.** The latest set on or before the date wins.
- **One entity, one investment.** The walk returns to commitments above that entity.
- **Never deleted.** Removing a set marks it removed and keeps the record.

`allocation_override_service.py` stores the sets (`ownership_overrides` and
`_lines`, both PROTECTED). `ownership_chain_service.group_shares(..., investment=)`
applies them, so it stays the one engine. Every route through an override names it,
and so do the row, a report note and the workbook's routes sheet.

Refused, with the reason:
- an entity not in the investment's chain on the effective date (it would change
  nothing);
- an entity the report already stops at;
- negative amounts, or amounts that total zero;
- an investor named twice;
- no reason given;
- a second set on the same date.

An investor MRI doesn't carry for that entity is saved with a warning.

Writes are `has_accounting_authority`; reads are open to the report's users.
Guardrail `scripts/allocation_override_check.py` (38 checks). It fails on each
injected defect: `ignore` (10 failures), `nodate` (7), `global` (1).

**Accounting enters the actual sets**: they have the funded amounts. Nothing was
entered on their behalf.

### Future funding at an override is split by what is still OWED (variant D, Oct 7 2026)
`ownership_chain_service.group_shares(..., override_weights="remaining")` weighs an
override entity's investors by commitment in force less the override's funded amount;
`pe_exposure_service` uses it for FUTURE FUNDING only (funded rows keep the funded split).
No override -> identical to before (verified on production at 12/31/25: not one deal
moved). Why only at overrides: measured four ways against accounting's 12/31/25 tracker
(see `board.md`); a general "who still owes" walk mis-split Bel Air, where PSC1 funded
beyond its commitment at I1BAS2 and PIG5 to carry other investors. With Brainerd's two
sets supplied in memory, Brainerd's funded (PSC 6,784,705 / TIAA 11,622,972) and future
(PSC 1,331,425 / TIAA 11,982,825) match the tracker to the dollar, and nothing else moves.
Nothing remaining at the entity -> funded split, said in the route; a committed investor
the set does not name owes its whole commitment. Guardrail
`scripts/future_funding_split_check.py` (13; `--inject=ignore|nofallback|unnamed|wiring`).
