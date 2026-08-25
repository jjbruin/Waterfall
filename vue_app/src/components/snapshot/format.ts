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

/** $M with one decimal — the unit the reference PDF reports.
 *
 * The regex strips a sign that survives only because of rounding. Jefferson
 * Waters Creek's At Close NOI is a few cents below zero, so `toFixed(1)` gives
 * "-0.0" — which reads as a loss where the PDF prints its accounting dash for
 * nothing. Done on the string, not via `+ 0`, because the case to catch is any
 * small negative that rounds to zero at this precision, not just literal -0.
 */
export function fmtM(v: number | null | undefined): string {
  if (v == null) return DASH
  return (v / 1e6).toFixed(1).replace(/^-(0\.0*)$/, '$1')
}

/** $M with dollar sign prefix — for first data row and total row.
 *
 * Delegates so it inherits `fmtM`'s rounded-negative-zero handling; two
 * near-identical formatters that disagreed on "-0.0" would be a trap. */
export function fmtM$(v: number | null | undefined): string {
  if (v == null) return DASH
  return '$' + fmtM(v)
}

/** Whole dollars with thousands separators. */
export function fmtCurr(v: number | null | undefined): string {
  if (v == null) return DASH
  return '$' + Math.round(v).toLocaleString()
}

/** A decimal rate as a percentage (0.5751 -> "57.5%"). */
export function fmtPct(v: number | null | undefined, dp = 1): string {
  if (v == null) return DASH
  return (v * 100).toFixed(dp) + '%'
}

/**
 * A value ALREADY in percentage points as a percentage (92.23 -> "92.2%").
 *
 * Almost every percentage on this page is a decimal ratio, so `fmtPct` is the
 * default and this is the exception. It exists for figures the backend copies
 * verbatim out of the One Pager `property_performance` payload, where the unit
 * is percentage points, not a ratio — `economic_occ` is the one in use today
 * (`one_pager.get_property_performance` scales every branch of it to 0-100).
 *
 * Deliberately NOT the tolerant `v > 1 ? v : v * 100` heuristic that
 * `OnePagerView.fmtPct` uses: 0.98 is a legitimate 0.98% reading on a ratio
 * field, so guessing the unit from the magnitude is a bug waiting for the deal
 * that sits near the boundary. Pick the formatter that matches the field.
 */
export function fmtPctPts(v: number | null | undefined, dp = 1): string {
  if (v == null) return DASH
  return v.toFixed(dp) + '%'
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

type Kind = 'pct' | 'pctpts' | 'x' | 'currency' | 'm' | 'raw'

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
    case 'pct': return fmtPct(n, dp ?? 1)
    case 'pctpts': return fmtPctPts(n, dp ?? 1)
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
