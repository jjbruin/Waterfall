/**
 * Display formatting for the board slides. DISPLAY ONLY: every figure arrives
 * from the server already computed; these only write it the way the deck does.
 *
 * `null` is "the engine has no figure": a dash, never 0.
 */
export function money(v: number | null | undefined, dp = 1): string {
  if (v === null || v === undefined) return '—'
  const sign = v < 0 ? '-' : ''
  return sign + '$' + Math.abs(v / 1e6).toLocaleString('en-US', {
    minimumFractionDigits: dp, maximumFractionDigits: dp,
  })
}

export function pct(v: number | null | undefined, dp = 1): string {
  return v === null || v === undefined ? '—' : (v * 100).toFixed(dp) + '%'
}

/** "29-31" -> 29: where a schedule sits in the deck. */
export function firstPage(pages: string | number): number {
  const n = parseInt(String(pages), 10)
  return Number.isFinite(n) ? n : 999
}
