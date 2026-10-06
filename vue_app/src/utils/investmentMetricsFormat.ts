/**
 * How one Investment Metrics cell is written as text.
 *
 * ONE FORMATTER for the table on screen, the printed sheet and the CSV export.
 * It used to live inside `InvestmentMetricsTable.vue`; the CSV needs exactly the
 * same text, and a second copy is how a file comes to disagree with the page.
 */

/** Columns that carry a figure rather than a word; only these ever show a dash. */
export const NUMERIC = new Set([
  'total_size', 'first_lien', 'first_lien_pct', 'pref', 'pref_pct',
  'first_loss', 'first_loss_pct', 'uw_irr', 'realized_irr', 'proceeds',
  'proj_yr1_coc', 'act_yr1_coc', 'proj_coc_since_close',
  'act_coc_since_close', 'pref_coupon', 'residual_cf_split', 'irr_lookback',
])
export const MONEY = new Set([
  'total_size', 'first_lien', 'pref', 'first_loss', 'proceeds',
])

/**
 * FORMAT BY FIELD, NEVER BY MAGNITUDE. The same two columns carry dollars in
 * millions and percentages, and a value of 8.5 is "$8.5" in one and "8.5%" in
 * the next. Deciding from the number would get both wrong on the deals where
 * they happen to coincide.
 *
 * `null` is NOT zero. A dash means the app has no figure; "0.0%" means it has
 * one and it is zero. Collapsing the two is how a deal with no data comes to
 * read as a deal that returned nothing.
 */
export function fmt(row: any, key: string): string {
  const label = row?.labels?.[key]
  if (label) return label
  const v = row?.[key]
  if (v === null || v === undefined) return NUMERIC.has(key) ? '—' : ''
  if (MONEY.has(key)) {
    const sign = v < 0 ? '-' : ''
    return `${sign}$${Math.abs(v).toLocaleString('en-US', {
      minimumFractionDigits: 1, maximumFractionDigits: 1,
    })}`
  }
  if (NUMERIC.has(key)) return `${(v * 100).toFixed(1)}%`
  return String(v)
}

export function cellText(row: any, col: any): string {
  if (col.key === '_spacer') return ''
  if (col.key === 'name') return ''          // rendered with its markers
  if (col.key === 'invest_date') return row.invest_date_display || ''
  return fmt(row, col.key)
}

export function totalText(total: any, col: any): string {
  if (!total || col.key === '_spacer') return ''
  if (col.key === 'name') return total.label || ''
  if (['asset_class', 'dma', 'invest_date', 'partner'].includes(col.key)) return ''
  return fmt(total, col.key)
}
