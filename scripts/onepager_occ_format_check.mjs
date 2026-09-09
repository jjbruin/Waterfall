/**
 * Guardrail: the One Pager's economic-occupancy cells are formatted by unit,
 * not by magnitude.
 *
 * THE DEFECT. `fmtPct` infers the unit from the size of the number:
 *
 *     const pct = val > 1 ? val : val * 100
 *
 * Economic occupancy always arrives in percentage points, so that guess is
 * wrong whenever a reading fails `> 1` — and it fails for every negative and
 * for every genuinely-low occupancy. Jefferson Waters Creek's stored -12.8468
 * printed as "-1284.7%"; a real 0.5% would print as "50%".
 *
 * `fmtPct` is NOT changed — it is correct for the genuinely-decimal fields it
 * also serves (pe_coupon 0.08 -> "8%"). The occupancy row uses `fmtOcc`, which
 * fixes the unit by contract.
 *
 * NO VALUE BOUNDING. This checks display only. -12.8% is what the calculation
 * produced and the page now says so; whether that figure ought to be negative
 * is a data question about Waters Creek's budgeted account 4043, not a
 * formatting one.
 *
 * The two functions are read OUT OF THE COMPONENT rather than reimplemented,
 * so the guardrail cannot drift from the code it is checking.
 *
 * Usage
 *     node scripts/onepager_occ_format_check.mjs
 */
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..')
// Newlines normalised: the working tree is CRLF on Windows, and every pattern
// below anchors on "\n" — without this the row and function matchers silently
// find nothing and the guardrail reports a defect that is not there.
const SRC = readFileSync(
  join(ROOT, 'vue_app/src/views/OnePagerView.vue'), 'utf8').replace(/\r\n/g, '\n')

/** Lift a formatter's body straight out of the component. */
function extract(name) {
  const re = new RegExp(
    `function ${name}\\(val: number \\| null \\| undefined\\): string \\{([\\s\\S]*?)\\n\\}`)
  const m = SRC.match(re)
  if (!m) throw new Error(`${name} not found in OnePagerView.vue`)
  return new Function('val', m[1].replace(/: number \| null \| undefined/g, ''))
}

const fmtOcc = extract('fmtOcc')
const fmtPct = extract('fmtPct')

// The occupancy row must call fmtOcc, and must not call fmtPct.
const rowMatch = SRC.match(/\{ label: 'Economic Occ\.',[\s\S]*?\},\n/)
if (!rowMatch) throw new Error('Economic Occ. row not found')
const row = rowMatch[0]

const results = []
const chk = (name, ok) => results.push([name, !!ok])

chk('the Economic Occ. row formats with fmtOcc', /fmtOcc\(/.test(row))
chk('the Economic Occ. row no longer calls fmtPct', !/fmtPct\(/.test(row))
chk('fmtPct itself is untouched (still guesses by magnitude)',
  /val > 1 \? val : val \* 100/.test(
    SRC.match(/function fmtPct[\s\S]*?\n\}/)[0]))
chk('fmtOcc does NOT scale by 100', !/\* *100/.test(
  SRC.match(/function fmtOcc[\s\S]*?\n\}/)[0]))
// No bounding: this fix is display-only and must not withhold a value.
chk('fmtOcc does not withhold out-of-range values (no bounding)',
  fmtOcc(-12.8468) !== '—' && fmtOcc(150) !== '—')

// ── the real stored values, transcribed from live on 2026-09-09 ──────────
// vcode -> [label, stored, what fmtPct printed, what must print now]
const CASES = [
  ['Waters Creek  ytd_budget', -12.8468, '-1284.7%', '-12.8%'],
  ['Waters Creek  ytd_actual', 72.3, '72.3%', '72.3%'],
  ['Waters Creek  actual_ye', 80.936, '80.9%', '80.9%'],
  ['Waters Creek  uw_ye', 74.8173, '74.8%', '74.8%'],
  ['Evergreen     ytd_actual', 95.4, '95.4%', '95.4%'],
  ['Evergreen     ytd_budget', 95.5237, '95.5%', '95.5%'],
  ['Camp Creek    ytd_actual', 93.5333, '93.5%', '93.5%'],
  ['Camp Creek    ytd_budget', 93.7918, '93.8%', '93.8%'],
]

console.log('STORED VALUE -> DISPLAY')
console.log('  ' + 'cell'.padEnd(28) + 'stored'.padStart(12)
  + 'before (fmtPct)'.padStart(18) + 'after (fmtOcc)'.padStart(17))
for (const [label, stored, before, want] of CASES) {
  const gotBefore = fmtPct(stored)
  const gotAfter = fmtOcc(stored)
  const moved = gotBefore !== gotAfter
  console.log('  ' + label.padEnd(28) + String(stored).padStart(12)
    + gotBefore.padStart(18) + gotAfter.padStart(17)
    + (moved ? '   << FIXED' : ''))
  chk(`${label}: fmtPct really did print ${before}`, gotBefore === before)
  chk(`${label}: now prints ${want}`, gotAfter === want)
}

// The magnitude trap, in the direction nobody reported: a real low occupancy.
console.log('\nTHE OTHER DIRECTION (unreported, same root cause)')
for (const v of [0.5, 0.9, 1.0]) {
  console.log(`  stored ${String(v).padEnd(5)} fmtPct -> ${fmtPct(v).padStart(8)}`
    + `    fmtOcc -> ${fmtOcc(v).padStart(8)}`)
  chk(`a genuine ${v}% is not centupled`, fmtOcc(v) === v.toFixed(1) + '%')
}

chk('null renders as an em dash', fmtOcc(null) === '—')
chk('NaN renders as an em dash', fmtOcc(NaN) === '—')

console.log('\n' + '-'.repeat(72))
for (const [name, ok] of results) console.log(`  [${ok ? 'PASS' : 'FAIL'}] ${name}`)
const bad = results.filter(([, ok]) => !ok).length
console.log(`\n${bad ? 'FAIL' : 'PASS'} — ${results.length - bad}/${results.length} checks`)
process.exit(bad ? 1 : 0)
