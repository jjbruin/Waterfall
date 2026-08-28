# Session Handoff — through Aug 28 2026 (v396 live)

Rolling handoff for the next session/developer. Update in place; keep only
what is still live. Deploy history lives in CLAUDE.md (SHA-pinned).

## Where things stand
- **Live**: v396 = `557af5e` (Charlene: U/W ROE 7073 dedup +
  fc30c4f Prop_Info_Core refresh crash fix). Aug 26–28 chain shipped
  v369→v396: audit
  workbook rebuild, 4031 Commercial Vacancy fix (11 deals), NB timing
  (equity funds at close−1d, sale = month-end before the hold anniversary,
  Terminal NOI column), operating overrides on Argus (mgmt fee %,
  replacement reserve as its own line), parcel-sale rent as "Less: {name}"
  under Rental Income, manual parcel ROC processed in date order (pref on
  post-ROC balances), NB parcel loans/tenants (Capital Budget ids, Argus
  rent roll via scenario), Argus imports auto-surface as scenarios,
  waterfall persistence (Compute reads only; lossless Builder hydration),
  NB ROE/MOIC audits at AM parity, and the full PPI ownership waterfall
  feature (phases 1–5 + linked-JV chains + annual pivot + Fund Investors
  grouping). Details: `new_business_pipeline.md`,
  `ppi_ownership_waterfalls.md`.
- **Windsor Square (prospect 3 / N0000003)** is the reference deal:
  target_close 2026-09-24 (Jim set it), exit cap 7.87%, replacement
  reserve $0.22/SF, Sam's Club parcel sale (12/31/2026, pro-rata), Base
  Case = auto-scenario pinning Argus import 1, PPI stack linked to the
  real TGA6 JV (TGAM 90 / INV6 10; 95bps to PSCMAN; 20% promote after
  TGAM's 9% net IRR with PSCMAN keeping 20% of it; AMB6's 13 investors at
  relationship pcts paying 50bps). TGA6 (11 rows) and AMB6 (52 rows)
  waterfalls live in the shared table.

## Open items
1. **TGA6 pref**: modeled with NO preferred return (straight 90/10 + 9%
   net-IRR gate at capital events). Jim to confirm; if the JV carries a
   current pref, add the tier to TGA6's stored waterfall.
2. **Purchase prices blank on 8 deals** (Plaza Del Mar etc.): the query
   fix is deployed (v392 reads both MRI interest labels) AND the refresh
   crash is fixed (v396, fc30c4f: Prop_Info_Core died on a float in a str
   column — the likely root cause of Charlene's failed refreshes). Next
   step: Charlene retries the Prop_Info_Core refresh from an
   MRI-connected machine; prices should then fill. MRI is a direct SQL
   Server connection (mri_service.py, servers 10.219.226.9/.10).
3. **PPI32 padded InvestorID rows** (JB Fair Park): 3 accounting rows +
   1 commitments + 1 relationships row carry trailing spaces. Loaders
   strip them (nothing computed is wrong). Durable fix belongs in the MRI
   source; an import-time strip in mri_service is the offered fallback.
4. **PSCKOC / Portfolio Analysis spot-check**: the `state_credited`
   engine fix (terminal recipients were double-credited in state
   cashflows) makes their IRR gates more correct — numbers may shift
   slightly on next run; confirm against last saved figures.
5. **Onboarding**: PPI migrate endpoint exists
   (`POST /ppi-stack/migrate`); the broader one-click "Onboard to
   Portfolio" wizard in CLAUDE.md is still aspirational — only the Argus
   NB→AM migration and the PPI migrate step exist.

## Process rules that keep biting
- **Deploys**: `git pull --rebase` BEFORE `az acr build`; verify the SHA
  is on origin/main before `az containerapp update`. Push races produced
  off-main images twice (v376/v378, both superseded within minutes).
- Azure management endpoint has transient connect timeouts from this
  machine — the operation usually landed; re-query before retrying.
- Prod Postgres writes from scripts get blocked by the permission
  classifier — route through app endpoints (test client) or ask Jim.
- All repo files CRLF except a few LF strays; patch scripts must detect
  per-file (`'\r\n' if '\r\n' in raw else '\n'`).
- Windsor lives ONLY in live Postgres — local waterfall.db is a stale
  April snapshot. Set DATABASE_URL for any NB verification.
