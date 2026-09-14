---
name: ISBS Split Migration
description: ISBS table split into 6 tables by vSource to work around MRI 1,000-record query limit; pending VPN access for direct MRI database queries
type: project
---

## Problem
MRI reduced query record limit to 1,000 rows, making it impossible to export the monolithic ISBS_Download.csv (800K+ rows).

## Solution: Split Tables (code complete, Apr 2026)
Split ISBS into 6 tables by vSource type:
- `isbs_interim_is` ← ISBS_Interim_IS.csv (Actuals 2025+)
- `isbs_interim_is_historical` ← ISBS_Interim_IS_Historical.csv (Actuals pre-2025)
- `isbs_interim_bs` ← ISBS_Interim_BS.csv (Balance Sheet)
- `isbs_budget_is` ← ISBS_Budget_IS.csv (Budget)
- `isbs_projected_is` ← ISBS_Projected_IS.csv (Underwriting)
- `isbs_valuation_is` ← ISBS_Valuation_IS.csv (Valuation)

> **Correction, Sep 14 2026 — the `isbs_interim_is_historical` line above is no longer
> true, and misled a session into hunting for data that was never missing.** Direct MRI
> access works now: `ISBS_Download` pulled 800K+ rows successfully today, so the refresh
> is the live source rather than the per-table CSVs this plan assumed. In that path
> EVERY `Interim IS` row lands in `isbs_interim_is` — no date filter — which is why it
> holds 2018-03-31 to 2026-08-31, not "2025+". `isbs_interim_is_historical` is empty,
> no `ISBS_Interim_IS_Historical.csv` has ever been imported, and the only thing that
> ever writes it is `split_isbs_table()` here. It is still worth keeping: `_ISBS_SPLIT`
> concatenates it into `isbs_raw`, so rows put there do reach every consumer.

Architecture in place: TABLE_DEFINITIONS, _assemble_isbs(), indexes, cache invalidation, CSV upload auto-detection. Consumers unchanged — assembly restores vSource column.

## Next Step: Direct MRI Database Access (VPN pending May 2026)
- VPN access requested — awaiting connection details from MRI technicians
- **Database confirmed**: SQL Server, IM database at psc.investment.mrisoftware.com
- **Query ready**: `queries/ISBS_Download.sql` — queries `vstaging_journal_entry` + `coa`, returns all vSource types
- **Source tables**: `vstaging_journal_entry` (journal entries), `coa` (chart of accounts)
- Once VPN is live: single query replaces all 6 split CSVs — filter by vSource in Python or at query level
- Recommended: sync into our split tables (not live query) to stay decoupled from MRI uptime
- See `queries/README.md` for VPN testing checklist
