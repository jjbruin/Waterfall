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
