# Valuation Budget Comparison — the budget/Argus line mapping and the debt rows

Moved verbatim out of CLAUDE.md on Oct 5 2026. Screen: Valuations > Budget
Review. Code: `LineMappingPanel.vue`, `line_mapping_service.py`,
`budget_import_service.py`, `budget_import_validate.py`,
`valuation_debt_service.py`, `valuation_budget_inputs.py`.

## Valuation Budget Comparison — loading it, and levering it

### Valuation Budget Comparison — loading it, and levering it
The Budget Review tab's comparison is **Estimate | Budget | Valuation**. Two of those
columns are loaded from somebody else's spreadsheet, and the debt rows are ours.

- **One screen for both sources** (`LineMappingPanel.vue`, `line_mapping_service.py`,
  four endpoints under `/api/valuations/records/<id>/mapping/`). `source` is `budget`
  (partner's workbook → `isbs_budget_is_supplements` → Budget column) or `argus`
  (appraiser's Argus download → the record's `argus_imports` / `argus_cashflows`, WRITTEN
  by `_commit_argus` from this screen's own reading since `v528` → Valuation column).
  Same job, same rules, same screen.
- **THE ACCOUNT IS THE MAPPING; the category is derived from it** (Jack, Sep 22 2026,
  reversing "category first"). `budget_import_service.category_for_account` is the one
  lookup, the server ignores whatever category the screen sends, and the screen
  displays it read-only. Measured before reversing: all 80 accounts in
  `category_accounts()` belong to exactly one category, so a separately-chosen category
  could only ever agree or contradict — and contradicting was BLOCKING, which is the
  "two separate steps and they fight each other" he reported. An account on **no**
  category still blocks; a rule that only ever corrects would accept anything. The
  whole chart is offered unconditionally, since with nothing narrowing the list a tick
  box would leave most accounts unreachable.
- **MANY LINES MAY SHARE ONE ACCOUNT** — 23 of Evergreen's repair lines are 5060. They
  combine, and the combining is reported with the lines named and the combined figure.
  Simply not-blocking would be satisfied by dropping every line after the first.
- **THE LABEL AND THE AMOUNTS MUST COME FROM THE SAME BLOCK.** A sheet can carry two
  independent tables side by side (Evergreen v5: a roll-up in A–B, the detail it came
  from in D–G with the months beside the detail); reading names from one and figures
  from the other produced "5051 - Water" carrying Property Management's 366,157.78, and
  nothing about it looks wrong. A second block announces itself with a second account
  column — **found by membership of our chart of accounts, not by shape**, because a
  roll-up of annual totals (1200, 240, 120) matches "3–6 digits" perfectly. Without
  that evidence nothing is re-based, or a sheet whose labels merely have a
  sub-description beside them would be read off the sub-description.
- **The account may be its own column or lead the label** ("4010 - Rental Income"), and
  the label's own account outranks a separate column. Before this the detector took the
  account column AS the label and read nothing else, which is why Jack was building a
  helper column joining the number and the description by hand.
- **`commit()` READS THE COLUMN NAMES FROM THE TABLE.** Production's supplement tables
  carry `vCode`; the MRI-created ISBS tables carry `vcode`. A double-quoted identifier
  is case-SENSITIVE on PostgreSQL and case-INSENSITIVE on SQLite, so `WHERE "vcode"`
  passed every local test and raised `UndefinedColumn` on every real import — **the
  budget import had never once succeeded on production**, for any file, since it was
  written. Quoting is not enough, and the docstring that claimed the columns "really
  are `vcode`" was itself the mistake.
- **The flip default is PER ACCOUNT, from the deal's own history**, never from the
  4xxx/5xxx prefix: 4030 Residential Vacancy and 4042 Loss to Lease are 4xxx stored
  POSITIVE, 5220 Other (Income) Expense is 5xxx stored NEGATIVE.
- **Guardrail**: `scripts/budget_import_mapping_check.py` (25), on fixtures of all
  three real file shapes plus the negative case for block re-basing; proved
  non-vacuous against nine injected defects including both opposite failures.
- **ONLY A ROW WITH A 4-DIGIT ACCOUNT IS IMPORTED** (Jack, Oct 7 2026, superseding the
  subtotal reading): `budget_import_service.file_account` -- the account the parser read,
  else one leading/ending the label. Every such row is pre-filled from it, INCLUDING one
  whose label reads like a total (tagged "reads like a total" on screen, since it can
  double-count); a row without one is not offered, and `validate` refuses it by name
  (`no_file_account`). The account stays EDITABLE: measured on production's 9 stored
  mappings, 8 lines were deliberately mapped off the file's account (partner 7076 TI ->
  our 7050; 4045 -> 4060) and 7076/5019/5180 are not on our chart. Clearing an accounted
  row is allowed and recorded (`left_out` in the mapping) and listed beside the tie-out.
  **A draft saved before the rule** is read by it on load: rows with no account drop
  (said), accounted rows with no decision are filled from the file (said) -- Camp Creek's
  saved budget took interest from unnumbered "CIBC" rows and left "5190 Total Interest"
  out, so applying it as saved would have imported no interest. 3 of 4 stored Argus
  files (P0000069, P0000075, P0000081) carry NO account column and import nothing now.
  `reconcile()` still shows stated-vs-computed revenue, expense and NOI. The "as mapped
  before" pre-fill and the "not a subtotal" toggle are gone (both only ever served rows
  with no account). Guardrail: `scripts/am_import_rules_check.py`.
- **Commit REPLACES, scoped to (vcode, the periods in THIS file)** — a budget is
  re-imported until final and appending would stack every revision.
- **Debt service: the Budget column is the BUDGET'S OWN by default** (5190 / 7060; Jack,
  Oct 7 2026 -- `valuation_budget_inputs.DEBT_BASES = ("budget", "modeled",
  "underwriting")`, default "budget"; all 162 production records were on the default). A
  budget carrying neither is BLANK with a note, never $0. A chosen basis with nothing for
  the year falls back to the budget's own, else modeled, else blank, and says which.
  The payload's `debt_service.budget_from` says what the column holds.
- **The Valuation column's debt service is MODELED** (`valuation_debt_service.py`). An Argus download is unlevered, so those columns showed
  0 interest, 0 principal and a blank DSCR. Same strip-and-replace `compute.py` applies
  to the AM forecast. **The Estimate column is never substituted** — it means actuals,
  and its interest was actually paid. Levered only when an Argus forecast exists:
  modeled debt over a zero NOI turns a blank DSCR into a hard `0.00`, which reads as
  "cannot cover its debt" instead of "no forecast loaded".
- **The third column is Valuation Yr 1 OR underwriting** (`?compare=underwriting`), the
  same `_calculate_is_amounts` with a different source. **UW records debt service as ONE
  figure, 7010 "Hard Debt (P&I)"** -- it carries no 5190 or 7060 -- so wherever UW is the
  source, Interest/Principal are blank and Total Debt Service is 7010, read through
  `one_pager.uw_debt_service_for_year` (shared with One Pager's UW DSCR). Never split it.
- **The Budget column's debt service may be UW's** (`valuation_records.debt_service_basis`).
  Chosen but unavailable -> not applied, and said so.
- **LOSS TO LEASE (4042) NETS INTO RENTAL INCOME on the Budget Review** (Jack, Oct 7 2026),
  in all three columns, via `budget_import_service.review_is_accounts()` -- config's
  grouping with `_REVIEW_MOVES`. The review, the import's category lookup and the tie-out
  read it; `config.IS_ACCOUNTS` (Financials, One Pager) keeps 4042 under Vacancy. On
  production 15 deals' 2026 actuals and 10 MRI budgets carry 4042; total revenue does
  not move. Argus signs 4042 as contra-revenue (a debit), whichever sign the file shows.
  P0000081's Argus "Loss to lease" is coded 4043, not 4042.
- **An Estimate line may be overridden** (`valuation_estimate_overrides`); LINE ITEMS ONLY,
  totals recompute and are marked. The computed figure is kept beside it.
- **Budgeted occupancy is read off the budget file** by label, only if its figures read as
  percentages, kept OUT of the mappable lines, and stored in `valuation_budget_occupancy`.
- **7030 is "Replacement Reserve Deposit" in the chart of accounts**, yet `INTEREST_ACCTS`
  treats it as interest. See `open_items.md` §12.5 before touching either.
- **THE ARGUS CASH FLOW IS LOADED ONCE**, in Budget Review > Load Valuation Cash Flow
  (AM, Sep 28 2026). The Assumptions-tab upload and its route are GONE. It used to be
  read by `argus_parser.parse_monthly_cashflow` there and by the budget parser here, with
  the mapping written back BY LABEL onto the first import -- a line the two parsers named
  differently took no mapping. `line_mapping_service._commit_argus` now WRITES the
  Valuation cash flow from the panel's own reading: creates the import if the record has
  none, replaces it in place if only this record links it, and makes a NEW one if another
  record (another cycle) shares it. Signs come from the account via
  `argus_service._normalize_amount` -- and since Oct 6 2026 a per-line **Flip sign box
  for Argus too** (Jack), stored as `reverse`, never `flip`: the screen had been
  pre-filling the budget's `flip` on Argus lines, unseen, and 14 production lines carry
  one. Honouring `flip` would have inverted them on the next apply.
- **ONE FUNCTION FOR WHAT A MAPPING WRITES** -- `budget_import_validate.imported_amounts`
  (Oct 6 2026). Both commits, the tie-out, the "as imported" column (`check()` returns
  `imported`) and the Excel export read it. Before, the Argus tie-out applied the unseen
  `flip` the Argus import ignored, and showed NOI off by $2,839,732 (P0000069) and
  $8,684,788 (P0000075) when what was imported tied to $10 and $60. The Valuation column
  itself was never wrong.
- **$0 lines are set aside** (`without_zero_lines`, Oct 6 2026): every month $0, NOT
  "nets to $0" (a +500/-500 line moves months). Reported as `zero_lines`, applied to
  stored drafts too. One production budget was 221 of 309 lines $0.
- **The warnings are off the screen** (Jack, Oct 6 2026: "a couple hundred lines ...
  Take those out entirely. Keep the does-it-tie check"). `validate()` still computes
  them and the API still returns them; only BLOCKING items render. "Export mapping to
  Excel" (`/mapping/export`) gives sheet row, account, flip, sheet total vs imported total
  and the months, plus By account and Tie-out sheets.
  Guardrail: `scripts/mapping_feedback_check.py`.
- **Argus is mapped like the budget** -- the file's account, then "as mapped before",
  never keywords. The keyword pre-fill ran FIRST and outranked the file's own account.
- **An account column to the RIGHT of the description is read** (`_account_column_beside`),
  by membership of our chart, headed or not. Before this only the account-on-the-left
  layout was read, and AM's Argus layout is description then account.
- **The Partnership costs proposal now WRITES.** From `v502` the tick box never left the
  browser. It rides on the parsed file (`accepted_proposals`) and
  `with_accepted_proposals` turns it into a line; only offered accounts, Argus only.
- **Interest goes to 5190 here, 7030 in the AM forecast** — see `open_items.md` §5.8.
  Deliberate as of Sep 11 2026, not accidental, and still worth settling.
