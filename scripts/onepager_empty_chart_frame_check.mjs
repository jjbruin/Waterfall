/**
 * Guardrail: the One Pager chart is always a FRAME, never a "no data" message.
 *
 * THE DEFECT. `buildChartOption` returned null when `periods` was empty, and the
 * template rendered "No chart data available." in place of the chart. A deal
 * with nothing to plot yet is not a deal with a broken chart — the axes,
 * gridlines, labels and legend are the report's structure and belong on the page
 * whether or not there is a line to draw on them. Four deals at 26Q2 hit it:
 * Donald Lynch, Jefferson Stephens, Fairview Heights, Citizen Storage — all new
 * or development deals with zero rows in every ISBS table and zero occupancy
 * readings.
 *
 * WHY THE OPTION AND NOT THE PIXELS. The chart is an ECharts <canvas>, and the
 * printed PDF is not reproducible pixel for pixel: rendering the SAME build
 * twice differs by 5.1% in the chart area, because the capture lands at a
 * different point in the load animation. So a pixel diff cannot answer "did the
 * deals with data change" — this compares the option object the component
 * builds, which is deterministic.
 *
 * The functions are EXECUTED after being lifted out of the component — both the
 * committed version and the one on the merge base — so the check cannot drift
 * from the code, and "unchanged for deals with data" is proved rather than
 * asserted.
 *
 * Usage
 *   node scripts/onepager_empty_chart_frame_check.mjs [payloads.json]
 *
 * payloads.json: {vcode: <chart endpoint response>}. Falls back to a built-in
 * pair of synthetic payloads (one with data, one empty) when absent, so the
 * check still runs with no network.
 */
import { execFileSync } from 'node:child_process'
import { existsSync, readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { dirname, join } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..')
const REL = 'vue_app/src/views/OnePagerView.vue'
const BASE_REF = process.env.WF_BASE_REF || 'main'

let pass = 0, fail = 0
const chk = (label, ok, detail) => {
  (ok ? pass++ : fail++)
  console.log(`  [${ok ? 'PASS' : 'FAIL'}] ${label}`)
  if (detail && !ok) console.log(`         ${detail}`)
}

/** Lift niceCeil + noiAxisBounds + buildChartOption out of a source string.
 *
 * Transpiled with esbuild (the one Vite already depends on) rather than by
 * stripping type annotations with regexes — a hand-rolled stripper silently
 * mangles a form it does not know about, which is exactly the kind of drift
 * this guardrail exists to prevent.
 */
async function compile(src, tag) {
  const text = src.replace(/\r\n/g, '\n')
  const grab = (name) => {
    const re = new RegExp(`\\nfunction ${name}\\([\\s\\S]*?\\n\\}`)
    const m = text.match(re)
    if (!m) throw new Error(`${name} not found in ${tag}`)
    return m[0]
  }
  const ts = [grab('niceCeil'), grab('noiAxisBounds'), grab('buildChartOption')]
    .join('\n')
  // esbuild lives in vue_app/node_modules; a bare import resolves from this
  // file's own directory (scripts/), which has none.
  const { transform } = await import(
    pathToFileURL(createRequire(join(ROOT, 'vue_app/package.json'))
      .resolve('esbuild')).href)
  const { code } = await transform(ts, { loader: 'ts', format: 'cjs' })
  // eslint-disable-next-line no-new-func
  return new Function(`${code}\nreturn buildChartOption`)()
}

const nowSrc = readFileSync(join(ROOT, REL), 'utf8')
let baseSrc = null
try {
  baseSrc = execFileSync('git', ['show', `${BASE_REF}:${REL}`],
    { cwd: ROOT, encoding: 'utf8', maxBuffer: 1 << 26 })
} catch (e) {
  console.log(`  [SKIP] cannot read ${BASE_REF}:${REL} — before/after comparison skipped`)
}

const build = await compile(nowSrc, 'working tree')
const buildBase = baseSrc ? await compile(baseSrc, BASE_REF) : null

// ---- payloads -----------------------------------------------------------
const file = process.argv[2] || join(ROOT, 'vue_app/.chartcheck/live_charts.json')
let payloads
if (existsSync(file)) {
  payloads = JSON.parse(readFileSync(file, 'utf8'))
  console.log(`payloads: ${Object.keys(payloads).length} from ${file}\n`)
} else {
  payloads = {
    SYNTH_DATA: {
      periods: ['Q1 2026', 'Q2 2026'], actual_noi: [1e6, 1.2e6],
      uw_noi: [1.1e6, 1.15e6], occupancy: [92.5, 93.1],
    },
    SYNTH_EMPTY: {
      periods: ['Q1 2026', 'Q2 2026'], actual_noi: [null, null],
      uw_noi: [null, null], occupancy: [null, null],
    },
    SYNTH_NOPERIODS: { periods: [], actual_noi: [], uw_noi: [], occupancy: [] },
  }
  console.log('payloads: built-in synthetic (no live file found)\n')
}

const hasData = (c) => (c.actual_noi || []).concat(c.uw_noi || [], c.occupancy || [])
  .some((v) => v != null)

console.log('1. every payload — and null — produces a frame, never null')
chk('null input still builds a frame', build(null) != null)
chk('undefined input still builds a frame', build(undefined) != null)
let nulls = 0
for (const [vc, c] of Object.entries(payloads)) if (build(c) == null) nulls++
chk('no payload produces a null option', nulls === 0, `${nulls} produced null`)

console.log('\n2. the frame is complete — axes, gridlines, legend, series')
const empties = Object.entries(payloads).filter(([, c]) => !hasData(c))
const sample = empties.length ? empties[0] : Object.entries(payloads)[0]
const opt = build(sample[1])
chk(`${sample[0]}: has a title`, !!opt.title?.text)
chk(`${sample[0]}: has a category x-axis`, opt.xAxis?.type === 'category')
chk(`${sample[0]}: has two y-axes`, Array.isArray(opt.yAxis) && opt.yAxis.length === 2)
chk(`${sample[0]}: left axis is the fixed 0-100 occupancy scale`,
  opt.yAxis?.[0]?.min === 0 && opt.yAxis?.[0]?.max === 100)
chk(`${sample[0]}: right axis bounds are finite`,
  Number.isFinite(opt.yAxis?.[1]?.min) && Number.isFinite(opt.yAxis?.[1]?.max)
  && Number.isFinite(opt.yAxis?.[1]?.interval),
  JSON.stringify(opt.yAxis?.[1]))
chk(`${sample[0]}: right axis is the clean 0-1 empty default`,
  !hasData(sample[1]) ? (opt.yAxis[1].min === 0 && opt.yAxis[1].max === 1) : true,
  JSON.stringify(opt.yAxis?.[1]))
chk(`${sample[0]}: both axes keep 5 intervals so gridlines align`,
  Math.abs((opt.yAxis[1].max - opt.yAxis[1].min) / opt.yAxis[1].interval - 5) < 1e-9)
chk(`${sample[0]}: has a legend`, !!opt.legend)
chk(`${sample[0]}: all three series declared`,
  opt.series?.length === 3
  && ['Physical Occupancy', 'NOI U/W', 'NOI ACT']
    .every((n, i) => opt.series[i].name === n))
chk(`${sample[0]}: no series plots a point`,
  opt.series.every((s) => (s.data || []).every((v) => v == null)))
chk(`${sample[0]}: x-axis carries its labels`,
  (opt.xAxis.data || []).length === (sample[1].periods || []).length)

console.log('\n3. deals WITH data are byte-for-byte unchanged')
// Once this change is ON the base ref, "before" and "after" are the same code
// and the comparison has nothing left to say — in particular no payload can be
// previously-null any more. That is not a regression, so it must not read as
// one: a guardrail that only passes until it is merged is broken. Sections 1,
// 2 and 4 are the permanent invariants; this section is a migration check and
// retires itself, and can be re-run against the real merge base with
// WF_BASE_REF=<sha>.
// Ask the BASE what it does, rather than whether its file text matches ours.
// Comparing the sources was too narrow: any LATER edit to OnePagerView.vue —
// the Business Plan print fix was the first — makes the text differ again while
// the base still already frames, so the migration check came back and failed a
// second time. What actually matters is behaviour: if the base already returns
// a frame for a no-data payload, there is nothing left to migrate.
const NO_DATA_PROBE = { periods: [], actual_noi: [], uw_noi: [], occupancy: [] }
if (buildBase && buildBase(NO_DATA_PROBE) != null) {
  console.log(`  [SKIP] ${BASE_REF} already frames a no-data payload — the`
    + ' migration this section checks has happened. Re-run against a base from'
    + ' before it to exercise the comparison:')
  console.log('         WF_BASE_REF=<sha before the change> node'
    + ' scripts/onepager_empty_chart_frame_check.mjs <payloads.json>')
} else if (!buildBase) {
  console.log('  [SKIP] no base revision available')
} else {
  // Only payloads that actually carry a reading. A payload with no data is
  // exactly what this commit changes — the old code either returned null (live
  // payloads, empty `periods`) or, once the backend supplies the calendar
  // window, drew a 0.00-0.05 right axis where the new code draws 0.00-1.00.
  // Including those here would report the intended change as a regression.
  let same = 0, moved = [], became = [], skipped = 0
  for (const [vc, c] of Object.entries(payloads)) {
    const before = buildBase(c)
    const after = build(c)
    if (before == null) { became.push(vc); continue }
    if (!hasData(c)) { skipped++; continue }
    if (JSON.stringify(before) === JSON.stringify(after)) same++
    else moved.push(vc)
  }
  if (skipped) console.log(`         (${skipped} no-data payload(s) excluded — see note)`)
  chk(`every payload the old code charted is unchanged (${same} identical)`,
    moved.length === 0, `moved: ${moved.join(', ')}`)
  chk(`the old code returned null for ${became.length} payload(s), now framed`,
    became.length > 0 || Object.values(payloads).every((c) => (c.periods || []).length),
    'expected at least one previously-null payload when live payloads are used')
  if (became.length) console.log(`         now framed: ${became.join(', ')}`)
}

console.log('\n4. the message is gone from the template')
chk('no "No chart data available" outside a comment',
  !nowSrc.replace(/\/\*[\s\S]*?\*\//g, '').replace(/<!--[\s\S]*?-->/g, '')
    .includes('No chart data available'))
chk('the chart element is not gated on a v-if',
  !/v-if="chartOption"/.test(nowSrc) && !/v-if="buildChartOption\(/.test(nowSrc))

console.log(`\n${pass}/${pass + fail} checks passed`)
process.exit(fail ? 1 : 0)
