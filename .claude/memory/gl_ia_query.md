# GL / IA Query — the CFO's Spreadsheet Server filters, against our tables

Moved verbatim out of CLAUDE.md on Oct 5 2026. Screen `/gl-ia-query`, bottom of
the Accounting section. Code: `gl_ia_query_service.py`, `api/gl_ia_query.py`,
`MultiPicker.vue`. Guardrail: `scripts/gl_ia_query_check.py`.

## GL / IA Query — the CFO's Spreadsheet Server filters

### GL / IA Query — the CFO's Spreadsheet Server filters
Live at `v504`, screen `/gl-ia-query`, bottom of the Accounting section.

- **It does NOT re-run his SQL.** `queries/MRI_GL_Detail.sql` already IS his GL query
  ("the Spreadsheet Server NEW JOURNAL query one-for-one, with the &SPARM smart
  parameters removed") and `MRI_IA_Transactions.sql` is the IA one. Both import into
  `gl_detail` and `ia_transactions`. The tool puts the PARAMETERS back against the copy
  we hold (Jim, Sep 19 2026). A second copy of the SQL would be a second engine.
- **Filters** are exactly his list. GL: entities, period from/to, accounts, basis.
  IA: investments, investors, a date range on Transaction or Effective date, major
  types, sub types. All multi-select, all ANDed, all optional.
- **No user input is concatenated into SQL.** Bound parameters throughout, expanding
  bindparams for IN lists, and `date_field` — the one filter naming a COLUMN — matched
  against a fixed set before it reaches the query.
- **A truncated grid totals the WHOLE match**, not the rows shown, and says so.
  Totalling the visible page makes a truncated result look complete and be wrong.
- **A period before `202401` says the import bound is why**, because our GL copy starts
  there and an empty grid otherwise reads as "no activity".
- **Freshness is the last completed MRI refresh**, worded as the refresh and not the
  table (`mri_refresh_status` holds one row for the whole job), and unknown when none
  has completed.
- **Sub types carry their major type** — Return of Capital under Distribution is not
  the same line as one under Contribution.
- **`MultiPicker.vue`** (checkbox list + search + select all/clear) replaced a native
  `<select multiple>`: multi-select already worked and nothing on screen said so.
  Reusable anywhere the same problem appears.
- **FOUR COLUMNS ARE HIDDEN ON SCREEN AND KEPT IN THE EXPORT** (Jim, Sep 19
  2026): Bal/Fwd, Item, Related Entity, Related Entity Name. The workbook is what
  somebody checks the screen against and `ITEM` is how a line is found again in
  MRI's journal, so they are flagged (`GL_SCREEN_HIDDEN`), not dropped from the
  query. **Description is CLIPPED at 260px with the full text on hover** — across
  the real 79,074 rows it runs to 85 characters but the median is 29 and the tail
  repeats an entity name the Entity column already carries.
- **`ITEM` IS A LINE NUMBER, NOT A DEBIT/CREDIT SIDE**, and defaulting the filter
  to `ITEM = 1` would be wrong. Measured on production: 13,493 distinct values,
  ITEM=1 is 6,618 of 79,074 rows (8.4%) and 5.5% of the money, and the on-screen
  net would go from **10,797** to **2,102,385,065**. There is no duplication to
  remove — 0 duplicate rows on any key, and all 8,809 open-period entries balance
  to zero. A general ledger carries both sides because that is what it is; the
  median entry has 2 lines and the largest has 173. The way to see one side is
  the ACCOUNT filter, which already works. See `open_items.md` §9.8.
- **THE GRID SORTS AND FILTERS ON ANY COLUMN** (`v513`). Which line of an
  entry is the 'other side' is a property of the ACCOUNT, not of `ITEM` and not
  of the SIGN of `AMT` — sign keeps the expense on an expense entry and the CASH
  on a revenue entry. So no default was baked in; the reader slices it. Sorting
  copies before it sorts (`Array.sort` mutates, and clearing must restore the
  server's order), numeric columns sort numerically, and a truncated result says
  the slicing covers only the rows loaded.
- **A filter shows its subtotal BESIDE the whole-match total, never instead.**
  `v504` totals the whole match on purpose; once a filter is on that number no
  longer describes the screen, so both are shown.
- Guardrail: `scripts/gl_ia_query_check.py` (123).
- **Open**: his workbook's IA query is truncated mid-statement in row 49 (the non-cash
  branch); the To date is INCLUSIVE here and strictly-before in his sheet; reads are
  open to any signed-in user, which is a wider read than one entity's statement.
