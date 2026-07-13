# Transfer-Aware Returns — Design Document (May 2026)

## Problem Statement

Investors transfer ownership interests mid-life (404 detectable transfer pairs across the portfolio). The current system assumes one investor per PropCode for the life of a deal. Transfers break this — departing investors get phantom forward distributions, arriving investors get incorrect inception dates.

**Example**: PSC1 owned 10% of TGA23 from 2023-02-21 to 2024-03-31. INV23 replaced them on 2024-04-01. PSC1 should see a terminal positive cashflow (transfer proceeds); new investors' IRR starts 4/1/2024.

## Data Available (UPDATED May 7)

**Critical gap RESOLVED**: Transfer amounts exist in MRI's `IA_noncashtrans` table.

| Data Element | Source | Status |
|---|---|---|
| Transfer date | Relationships StartDate/EndDate | AVAILABLE |
| Departing/arriving investor IDs | Relationships | AVAILABLE |
| OwnershipPct | Relationships | AVAILABLE |
| **Transfer amount** | **`IA_noncashtrans` table in MRI** | **AVAILABLE** |

### IA_noncashtrans Discovery

MRI stores non-cash journal entries (ownership transfers) in `IA_noncashtrans`, separate from `IA_Contribution` and `IA_Distribution`. Example for TGA23 on 4/1/2024:

| InvestmentID | InvestorID | EffectiveDate | MajorType | Sub Type | Amount |
|---|---|---|---|---|---|
| TGA23 | PSC1 | 4/1/2024 | Other | Transfer of Ownership - P&L | 8,566.46 |
| TGA23 | INV23 | 4/1/2024 | Other | Transfer of Ownership - P&L | (8,566.46) |
| TGA23 | INV23 | 4/1/2024 | Other | Transfer of Ownership - Capital | 9,234,110.63 |
| TGA23 | PSC1 | 4/1/2024 | Other | Transfer of Ownership - Capital | (9,234,110.63) |

**Sign convention**: Positive = capital moving IN to investor (INV23 receives capital balance). Negative = capital moving OUT (PSC1 surrenders capital balance). This matches the MRI sign convention for contributions (negative in our feed) and distributions (positive).

**Transfer types**:
- `Transfer of Ownership - Capital`: Capital balance transfer (the main amount, $9.2M)
- `Transfer of Ownership - P&L`: P&L/income allocation transfer (smaller amount, $8.5K)

### Query Update

`queries/accounting_feed.sql` updated to UNION ALL from `IA_noncashtrans`:
- FK column: `NoncashTransTypeID` (verified via INFORMATION_SCHEMA) joins to `IA_Subtype.SubtypeUID`
- Capital flag: `Y` when Typename contains "Capital", else `N`
- ROE_Income: always `N` (transfers are not operating income)
- Cum_Amt: NULL (not meaningful for transfers)
- **COMPLETED**: Schema verified, query corrected, data pulled (127 transfer entries across portfolio)

## Revised Impact Assessment

With transfer data in the accounting feed, the implementation simplifies significantly:

### What Changes

1. **`seed_states_from_accounting()`** — Transfer entries arrive as normal accounting rows. "Transfer of Ownership - Capital" with negative amount for departing investor naturally reduces their capital balance. Positive amount for arriving investor establishes theirs. **No synthetic cashflows needed** — MRI provides the actual journal entries.

2. **IRR treatment** — The transfer entries should be included in each investor's cashflow stream:
   - PSC1 sees: contributions (negative) + distributions (positive) + Transfer-Capital out (positive, $9.2M) = terminal cashflow
   - INV23 sees: Transfer-Capital in (negative, -$9.2M) + subsequent contributions + distributions
   - This gives correct IRR for both parties automatically

3. **`build_partner_results()`** — Departed investors (PSC1) will have no forward waterfall distributions because they're not in the current waterfall steps. Their combined_cfs naturally end at the transfer date. **No special termination logic needed** if waterfall steps are current.

4. **`run_waterfall_period()`** — No change needed. Departed investors don't appear in current waterfall steps, so they don't receive forward distributions.

### What Stays the Same

- **`metrics.py`** — No changes. Pure functions on cashflow lists.
- **`InvestorState`** — No new fields needed for Phase 1. The accounting entries handle capital balance transitions.
- **`run_upstream_waterfall_period()`** — No change for Phase 1 (deal-level only).
- **Dashboard/Reports** — Inherit correct results.
- **Sold Portfolio** — Will automatically include transfer entries in its accounting-based calculations.

### Remaining Considerations

1. **Departed investors in partner_results**: `build_partner_results()` iterates waterfall step PropCodes + seed state PropCodes. PSC1 will appear in seed states (from accounting) but NOT in waterfall steps. Need to verify it still gets a partner_results entry with correct metrics (seed-only, no forward waterfall).

2. **Cum_Amt subquery**: The running capital balance subquery in the existing query doesn't include noncash transactions. For accuracy, the Cum_Amt calculation should also account for "Transfer of Ownership - Capital" entries. This is a query enhancement, not a code change.

3. **Display**: Transfer entries will show in XIRR Cash Flows with Typename "Transfer of Ownership - Capital". Vue may want to format these distinctly.

4. **Waterfall step consistency**: If someone enters a transfer in MRI but doesn't update the waterfall steps, the departed investor still receives forward distributions. The relationship EndDate check can detect this — flag as a warning.

## Revised Phasing

### Phase 1: Data Integration (LOW RISK)
1. Verify `IA_noncashtrans` column names via VPN (especially FK to IA_Subtype)
2. Finalize `accounting_feed.sql` query with UNION ALL for noncash transactions
3. Run query via MRI service to pull transfer data into accounting table
4. Verify transfer entries appear correctly in local + Azure databases
5. Test that existing deal computation handles transfers correctly (PSC1/INV23 in TGA23)

**Estimated effort**: 1-2 hours once VPN is connected
**Risk**: LOW — additive data, no engine changes

### Phase 2: Validation & Display
1. Verify `seed_states_from_accounting()` correctly processes transfer entries
2. Verify `build_partner_results()` includes departed investors with correct metrics
3. Add "Transfer" as a recognized Typename in display formatting (Vue)
4. Verify Sold Portfolio handles transfer entries
5. Add transfer detection from relationships as a validation check

### Phase 3: Upstream Transfer Awareness
- Portfolio Analysis: include historical investors in entity results
- Time-aware upstream waterfall tracing (if needed)

## Status — PHASE 1 COMPLETE (May 7)
- Query verified and corrected: FK=`NoncashTransTypeID`, sign flip (`*-1`), Transfer-only WHERE clause
- Data pulled from MRI: 11,600 total accounting rows including 127 transfer entries
- Imported to both local SQLite and Azure PostgreSQL
- TGA23 waterfall corrected: AMB23 replaced with INV23 in both databases
- **One code change**: `_build_entity_results()` now includes departed investors (those with accounting history but no current waterfall entries) in `all_recipient_ids`
- **TGA23 verified**:
  - PSC1 (departed): IRR=3.65%, MOIC=1.02x — invested $9.25M, received $231K + $9.23M transfer = $9.47M
  - INV23 (arrived): IRR=9.08%, MOIC=1.68x — bought in at $9.23M, projected $21M distributions
  - Transfer cashflows correctly signed: PSC1 +$9.2M (received), INV23 -$9.2M (paid)
- **No engine changes needed** — data integration + one small departed-investor inclusion fix
