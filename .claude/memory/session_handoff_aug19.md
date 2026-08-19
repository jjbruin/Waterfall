# Session Handoff — August 19, 2026

## What was completed this session
- Committed and pushed the Burton at-close economic occupancy fix (v293, already deployed)
  - `one_pager.py`: Added `inv_map` param to `get_property_performance()`, falls back to child vcodes via `_child_vcodes_for_parent()` when parent portfolio vcode not found in deal_terms
  - `financials_service.py`: Passes `inv=inv` to the call
- Updated CLAUDE.md revision suffix from v288 to v293
- Both commits pushed to origin/main

## Current deploy state
- **Azure**: v293 deployed and running
- **Git**: Fully synced — no uncommitted changes

## First thing tomorrow — ASK THE USER:
**"What fields are currently included in the AI lease extraction (Step 4), and is that everything you want to pull from the lease documents?"**

Context: Windsor Square has 644 lease PDFs uploaded. The next step is running AI extraction (Step 4 in the lease review workflow). Before running it, the user wants to review what data fields the extraction pulls and confirm whether additional fields should be added.

The extraction logic is in `lease_review_service.py` — look for the Claude API call that processes lease PDFs. Key fields currently extracted include rent steps, cotenancy clauses, exclusive use restrictions, options (renewal/termination), and key dates. The user may want additional fields extracted.

## Other pending items
- 36 unmatched documents from Windsor Square upload need manual tenant assignment via the UI (unmatched docs management feature was deployed in v291)
- Windsor Square extraction hasn't been run yet (waiting on field review)
- Lease Abstract Phases 2-4: awaiting user direction
- One Pager chart window branch (`feat/onepager-chart-window`) is NOT merged/deployed — documented in CLAUDE.md
