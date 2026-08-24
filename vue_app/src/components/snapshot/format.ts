/**
 * Shared formatters for the Portfolio Snapshot subtabs.
 *
 * The backend hands us `*_display` fields that are deliberately polymorphic: a
 * number when the metric computed, the literal string "Dev" when a development
 * deal suppresses it, the string "pending entry" for an un-entered manual
 * figure, or null when there is genuinely no data. `disp()` passes strings
 * through untouched so the UI never recomputes or second-guesses a backend
 * decision — that rule is what keeps "Dev" and n/a honest.
 */

export const DASH = '—' // em dash for "no value"

/** $M with two decimals — the unit the reference PDF reports. */
export function fmtM(v: number | null | undefined): string {
  if (v == null) return DASH
  return (v / 1e6).toFixed(2)
}

/** Whole dollars with thousands separators. */
export function fmtCurr(v: number | null | undefined): string {
  if (v == null) return DASH
  return '$' + Math.round(v).toLocaleString()
}

/** A decimal rate as a percentage (0.5751 -> "57.51%"). */
export function fmtPct(v: number | null | undefined, dp = 2): string {
  if (v == null) return DASH
  return (v * 100).toFixed(dp) + '%'
}

/** A ratio like DSCR (1.8536 -> "1.854"). */
export function fmtX(v: number | null | undefined, dp = 3): string {
  if (v == null) return DASH
  return v.toFixed(dp)
}

/** ISO date -> M/D/YYYY, parsed by regex.
 *
 * `new Date('2026-03-31')` is midnight UTC, which renders as the previous day
 * in US timezones. Same fix as `fmtDate` in OnePagerView.
 */
export function fmtDate(v: string | null | undefined): string {
  if (!v) return DASH
  const m = String(v).match(/^(\d{4})-(\d{2})-(\d{2})/)
  if (m) return `${parseInt(m[2])}/${parseInt(m[3])}/${m[1]}`
  return String(v)
}

type Kind = 'pct' | 'x' | 'currency' | 'm' | 'raw'

/**
 * Render a backend `*_display` value.
 *
 * Strings ("Dev", "pending entry") pass through verbatim — they are the
 * backend's decision, not ours to reformat. null becomes an em dash.
 */
export function disp(v: unknown, kind: Kind = 'raw', dp?: number): string {
  if (typeof v === 'string') return v
  if (v == null) return DASH
  const n = Number(v)
  if (!isFinite(n)) return DASH
  switch (kind) {
    case 'pct': return fmtPct(n, dp ?? 2)
    case 'x': return fmtX(n, dp ?? 3)
    case 'currency': return fmtCurr(n)
    case 'm': return fmtM(n)
    default: return String(v)
  }
}

/** True when a display value is a backend literal rather than a number. */
export function isLiteral(v: unknown): boolean {
  return typeof v === 'string'
}
