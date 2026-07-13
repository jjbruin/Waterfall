---
name: Dashboard Performance Optimization
description: prepare_cap_lookups() eliminates redundant DataFrame copies in 84-deal capitalization loop — 3.7x speedup
type: project
---

## Dashboard Loading Optimization (Mar 2026)

### Problem
Dashboard SSE init-stream loops over 84 deals calling `get_deal_capitalization()` per deal. Each call was:
- Rebuilding `build_investmentid_to_vcode()` mapping (84x)
- Copying + normalizing 7,024-row accounting DataFrame (84x)
- Copying + normalizing loans and valuations DataFrames (84x each)
- `get_property_vcodes_for_deal()` copying 84-row inv DataFrame + normalizing (84x)
- Total: ~143 MB of redundant DataFrame copies per dashboard load

### Solution: `prepare_cap_lookups()` in compute.py
Pre-computes all lookups once before the loop:
- `vcode_to_iids`: reverse mapping of InvestmentID→vcode (built once from `build_investmentid_to_vcode`)
- `acct_norm`: normalized accounting DataFrame (columns stripped, types converted, TypeName resolved)
- `loans_norm`: normalized MRI loans DataFrame
- `val_norm`: normalized MRI valuations DataFrame
- `prop_map`: pre-computed deal→child property vcode mapping (replaces per-call `get_property_vcodes_for_deal`)

Pass `lookups=prepare_cap_lookups(...)` to `get_deal_capitalization()`. Backward compatible — omitting `lookups` falls back to per-call behavior.

### Result
- **3.7x speedup** on capitalization loop (1.08s → 0.29s for 84 deals)
- Applied to both `dashboard.py` init-stream and `dashboard_service.py` `get_portfolio_caps()`

### Key Detail
- `build_investmentid_to_vcode()` returns `{InvestmentID: vcode}` e.g. `{"30BEAR": "P0000001"}`
- Dashboard uses `vcode` (P0000001), not InvestmentID (30BEAR)
- `vcode_to_iids` reverses: `{"P0000001": ["30BEAR"]}` for accounting lookups
