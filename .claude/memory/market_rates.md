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
