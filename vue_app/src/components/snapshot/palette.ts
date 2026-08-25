/**
 * Categorical palette for the Portfolio Snapshot page-1 charts.
 *
 * Hues are assigned in FIXED ORDER and never cycled: slot 1 always goes to the
 * first bucket the backend returns, slot 2 to the second, and so on. That means
 * a filter which changes how many buckets appear cannot repaint the survivors —
 * Retail stays amber whether or not Office is present.
 *
 * VALIDATED, not eyeballed. Run against the six checks (lightness band, chroma
 * floor, CVD separation, normal-vision floor, contrast) on 2026-08-25:
 *
 *   4 slots — #4472C4,#E8A33D,#2E9E8F,#8E5FA8   -> ALL CHECKS PASS
 *     lightness  all 4 inside L 0.43-0.77
 *     chroma     all 4 >= 0.1
 *     CVD        worst adjacent #8E5FA8/#2E9E8F dE 10.2 (deutan), 17.6 (tritan)
 *     normal     worst adjacent dE 21.2
 *     contrast   WARN on #E8A33D at 2.1:1
 *
 *   3 slots — the first three of the same list -> ALL CHECKS PASS
 *     CVD        worst adjacent #2E9E8F/#E8A33D dE 13.8 (protan)
 *
 * The contrast WARN is not dismissable: it obliges visible labels or a table
 * view. Both charts carry direct value labels AND a "Show data" table beneath,
 * so identity never rests on the fill alone.
 *
 * Slot 1 is the app's own --color-accent (#4472C4) so the charts sit inside the
 * existing visual language rather than beside it.
 *
 * The app has no dark theme (no prefers-color-scheme or data-theme anywhere in
 * vue_app/src), so there is no dark-surface stepping to validate. If one is ever
 * added these must be re-stepped against that surface — an automatic flip is not
 * a dark palette.
 */

/** Fixed-order categorical hues. Index = bucket position, never rank. */
export const CATEGORICAL = ['#4472C4', '#E8A33D', '#2E9E8F', '#8E5FA8'] as const

/**
 * A 9th series is never a generated hue. Four buckets is all the backend's
 * asset-type and deal-type rollups produce today; if a fifth ever appears it
 * folds into "Other" upstream rather than inventing a colour here.
 */
export const MAX_SLOTS = CATEGORICAL.length

/** Muted ink for a residual bucket, and for anything past the last slot. */
export const INK_RESIDUAL = '#9aa0a6'

/**
 * Bucket labels that are a leftover, not a category.
 *
 * The backend emits "Unclassified" when a deal's Lifecycle maps to no deal type
 * — City West at 26Q1, whose Lifecycle is null because it was foreclosed. A
 * residual must not wear a categorical hue: that would present "we could not
 * classify this" as a peer of Value-Add and Income, and it would also mean the
 * hue a real fourth category is entitled to had been spent on a gap in the data.
 */
export const RESIDUAL_LABELS = new Set(['Unclassified', 'Other', 'Unknown'])

export function isResidual(label: string | null | undefined): boolean {
  return RESIDUAL_LABELS.has(String(label || '').trim())
}

/**
 * Hue for bucket `i`. A residual label always gets the muted ink regardless of
 * position, so it cannot consume a categorical slot; anything past the last slot
 * gets it too.
 */
export function hueFor(i: number, label?: string | null): string {
  if (isResidual(label)) return INK_RESIDUAL
  return CATEGORICAL[i] ?? INK_RESIDUAL
}

/** 2px surface gap between stacked segments and adjacent bars. */
export const SURFACE = '#ffffff'
export const SEGMENT_GAP = 2

/** Recessive axis/grid ink — never a series colour. */
export const INK_AXIS = '#6c757d'
export const INK_GRID = '#eceff1'
