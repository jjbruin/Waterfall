# Portfolio Snapshot Performance Optimizations

## Context
The `/api/portfolio-snapshot/bundle` endpoint computes all four subtabs live for ~32 deals (TIAA).
Load time was slow. Three bottlenecks identified (Aug 2026):

## Optimization 1: Trim the One Pager provider (biggest win) — DONE
**Commit**: `1f679c9` (v342, Aug 24 2026)
**File**: `flask_app/services/portfolio_snapshot_freeze.py` — `_one_pager_provider()`

Replaced `get_one_pager_data()` with direct calls to only `get_capitalization_stack()` and
`get_property_performance()`. Skips `pe_performance`, `general`, and `comments` — the 3
sections no subtab reads. Eliminates ~60% of per-deal work. Does NOT modify the shared
`get_one_pager_data`; lean path calls the two functions directly with the same args.

## Optimization 2: Direct ISBS lookup for quarterly NOI — VERIFIED, READY TO DEPLOY
**File**: `flask_app/services/portfolio_snapshot_freeze.py` — `_quarterly_noi_provider()`

Replaced full `get_performance_chart_data()` call (12-quarter pipeline) with direct use of
the same shared `isbs_helpers` functions: `compute_cumulative_noi`, `cumulative_to_periodic`,
`aggregate_periodic`. Uses same `IS_ACCOUNTS` from `config.py`. Only processes Interim IS
data needed for the one requested quarter.

**Verification**: 210/210 matched (35 vcodes × 6 quarters) — old chart pipeline vs new
provider produce identical results on same data, including None cases (Giant 7, East
Manchester, incomplete quarters). Script: `scripts/verify_quarterly_noi.py --local`.

**Note**: Baseline CSV mode requires Azure-level ISBS data (through July 2026); local DB
only has through Feb 2026. The `--local` mode proves equivalence without needing that data.

## Optimization 3: Batch pe_cap_comments query — DONE
**Commit**: `1f679c9` (v342, Aug 24 2026)
**File**: `flask_app/api/portfolio_snapshot.py` — `_pe_cap_comments()`

Replaced per-deal loop (32 DB round-trips) with a single batch SQL query.
Preserves behaviors: only non-empty comments returned, degrades to `{}` on failure.
