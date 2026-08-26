# Scenario Analysis — New Business Deal Analysis (plan + build record)

Requested by Jim (Aug 26, 2026): named, saved scenarios on the Prospect Deal
Analysis page, selected from a Scenario dropdown, each running the full
waterfall. Scenario types: the operating partner's model, Base Case (our
DD-adjusted underwriting), and downside what-ifs driven by lease risk
(cotenancy cascades, tenant termination rights).

## What a scenario is

A named binding of three things, saved per prospect deal:

1. **Cash flow source** — which Argus import (per property) drives the
   forecast; `null` follows the normal cascade (active Argus > Excel >
   NOI-growth assumptions). "OP Model" vs "Base Case UW" are two Argus
   imports of the same property bound to two scenarios.
2. **Assumption overrides** — JSON overlay merged onto `prospect_assumptions`
   before analysis (hold_years, exit_cap_rate, debt terms, etc.).
3. **Adjustment events** — income deltas applied to the forecast from a date:
   `{label, start_date, end_date?, revenue: {acct: annual $}, expense:
   {acct: annual $}}`. Positive = removed, negative = added back (a re-lease
   after downtime is a removal event bounded by end_date, or an addition
   from the re-lease date). Same engine family as parcel-sale lost income.

## Schema

`prospect_scenarios` (PROTECTED_TABLES): id, prospect_id, name, description,
is_base, argus_import_ids JSON ({property_id: import_id} or null),
assumption_overrides JSON, adjustments JSON, sort_order, created_at,
updated_at, updated_by. Both DDL paths (Postgres `ensure_pg_tables`, SQLite
`create_additional_tables`).

## Engine

- `apply_scenario_adjustments(fc, adjustments, debug_msgs)` in
  prospect_analysis.py — like `apply_parcel_income_loss` but with signed
  deltas and an optional end_date window. Adjusts `mAmount_norm` only.
- `build_prospect_analysis(..., scenario=None)` — resolves the source,
  merges overrides, applies adjustments, reports each in diagnostics.
- Prospect analysis is computed fresh per Run (no cache), so no cache-key
  work was needed.

## API (prospects.py)

- `GET/POST /api/prospects/<id>/scenarios`, `PUT/DELETE .../scenarios/<sid>`
- `POST /api/prospects/<id>/analyze` accepts `scenario_id` in the body
- `GET /api/prospects/<id>/scenarios/risk-candidates` — seeds downside
  scenarios from the linked lease review (via
  `prospect_properties.lease_review_id`): tenants with termination options
  or cotenancy exposure, their annual rent, suggested start dates
  (earliest_termination_date / lease_end), months vacant default.

## UI (ProspectAnalysisView.vue)

- Scenario dropdown at the top of the results panel; Run Analysis computes
  the selected scenario. "+ New" / duplicate / rename / delete.
- Scenario editor: source per property, override fields, adjustment rows
  (label, dates, per-account annual deltas), and a "From lease risk" picker
  that turns risk candidates into adjustment rows.
- Comparison strip: after computing 2+ scenarios in a session, a table of
  deal + partner IRR/ROE/MOIC across them.

## Context that made this necessary (Aug 26)

- Windsor Square live deal is `N0000003` (prospect id 3, property id 3,
  Argus under `NP000003`). Investors: PPIWIND (pref, 70%, $19.15M),
  OPWATER (op, 30%, $8.2M) as `prospect_investors.planned_investor_id`.
- The waterfall was saved under placeholder IDs and re-keyed to
  PPIWIND/OPWATER on Aug 26 (backup: session scratchpad
  `windsor_waterfall_backup.json`).
- Equity split now honors declared investor records over waterfall-shape
  inference (`declared beats inferred`) — inference fails when both partners
  hold IRR steps, as Windsor's structure does.
