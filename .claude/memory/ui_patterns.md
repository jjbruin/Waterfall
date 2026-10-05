# Shared UI patterns

Moved verbatim out of CLAUDE.md on Oct 5 2026. Patterns that apply across
screens rather than to one feature.

## The input column folds away

### The input column folds away
Live at `v509`. `vue_app/src/components/common/CollapsiblePanel.vue`.

Jim, Sep 19 2026: the same little arrow as the sidebar, to give the analysis the
screen. Applied to the five pages that have a genuine inputs-left /
analysis-right split — Ownership (`picker`), Workpapers (`steps`), Reports
(`reports-sidebar`), Data Explorer (`table-list-panel`), Prospect Analysis
(`setup-panel`). Everything else is an equal-width content grid, a filter bar
above the results, or an overlay drawer, and is deliberately untouched.

- **A GRID parent must declare its own collapsed track.** The container sets the
  column, so a child narrowing itself to 30px reclaims NOTHING and leaves 260px
  of empty space where the panel was — the arrow works, the panel goes, and the
  analysis is exactly as cramped as before, with no error and nothing on screen
  saying so. Each page binds `input-collapsed` on its layout element and states
  `grid-template-columns: 30px …` itself. The guardrail asserts this per page;
  the component deliberately does not reach upward into a layout it cannot see.
- **The toggle survives collapsing**, and the rail keeps the panel's NAME. A
  control that vanishes when used cannot be undone by anyone who did not already
  know it was there — `v482` shipped exactly that and had to fix it.
- **The choice is remembered per page** in `localStorage`, per browser, wrapped
  in try/catch both ways: it is a convenience, never load-bearing, and a private
  window simply opens the panel.
- Measured in the running app at 1440px, both directions: Reports 280→30 (results
  888→1138), Ownership 290→30 (789→1064), Data Explorer 240→30 (833→1043),
  Workpapers 320→30 (721→1011), Prospect 480→33 (657→1119).
- Guardrail: `scripts/collapsible_panel_check.py` (43).
