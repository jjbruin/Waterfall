# Market rates (Data Management > Market Rates)

Moved verbatim out of CLAUDE.md on Oct 5 2026 by the compaction, the same day it was
written (`v565` = `032fa27`). CLAUDE.md keeps the RULES as bullets under Domain
invariants and the `rate_on` row in the ONE NUMBER table; this is the section as it
was written.

---

### Market rates (Data Management > Market Rates)
Built Oct 5 2026. `market_rates_service.py`, the `market_rates` table (protected):
Bank of Canada USD/CAD, CORRA and policy rate; NY Fed SOFR, its 30/90/180-day
averages and index, EFFR, OBFR -- free, official, no API key. Refreshed from the
publishers on the screen and at the end of Refresh All Data from MRI. `rate_on`
never interpolates: the last publication on or before the date, saying which, None
past 7 days. **Term SOFR is CME's and licensed -- not here.** Forward curves are not
built: the free official curve is Treasury's par yield curve (open). Guardrail
`scripts/market_rates_check.py` (18).

### U.S. Treasury par yields (added Oct 5 2026, after the compaction)
Jim: "add the treasury rates to the rate table, especially the 10-year." Thirteen
series `UST_1M` .. `UST_30Y` (the 10-year is `UST_10Y`) from Treasury's Daily Treasury
Par Yield Curve Rates, history from 2017, free, no key.

- **One CSV per calendar year, and its COLUMNS CHANGED OVER TIME** -- the 4-month
  began in Oct 2022 and a 1.5-month column was added later -- so a tenor is read by
  its column NAME, never by position. A tenor a year did not publish contributes
  nothing; a blank cell is not a zero.
- **All thirteen tenors share one download per year** (the refresh's cache), so a full
  backfill is ten downloads, not 130.
- **Inserts are batched now** (a Core `insert()`, which SQLAlchemy batches on
  PostgreSQL). The first full load of the original ten series took 200s on production
  writing row by row through `text()`; Treasury alone adds ~30,000 rows.
- 6/30/2026: 2-year 4.14, 10-year 4.44, 30-year 4.91.
- Still not built: FORWARD curves. Implied forwards could be bootstrapped from these
  par yields, but it is a Treasury curve -- a refinancing rate would be forward plus a
  spread that is an input, not market data (open_items 19.5).
- Guardrail now 25: section 7 pins the by-name read, the missing tenor, the blank
  cell and the shared download; reading a fixed column fails it.
