/**
 * Guardrail: the Portfolio Snapshot page-1 charts, against live data and the PDF.
 *
 * Imports the REAL option builders from
 * vue_app/src/components/snapshot/chartOptions.ts (bundled on the fly by
 * esbuild, which ships with vite) — not a replica — then:
 *
 *   1. builds both option objects from the live /summary payload,
 *   2. asserts every segment value, label and colour against that payload,
 *   3. checks the PDF's page-1 figures where they are meant to tie,
 *   4. renders both charts through ECharts SSR to SVG, which proves they draw
 *      without throwing and lets label geometry be inspected as text rather
 *      than guessed at,
 *   5. writes the two SVGs so they can be opened and looked at.
 *
 * Read-only against live: GET only, Bearer token from WF_TOKEN.
 *
 * Usage
 *   cd vue_app && npm install          # once, if node_modules is absent
 *   WF_TOKEN=<jwt> node scripts/snapshot_summary_charts_check.mjs
 */

import { mkdtempSync, writeFileSync, mkdirSync, readFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, dirname } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const ROOT = join(HERE, '..')
const VUE = join(ROOT, 'vue_app')
const BASE = 'https://app-waterfall-dev-v2.icyplant-026fb2db.eastus.azurecontainerapps.io'

const TOKEN = process.env.WF_TOKEN
if (!TOKEN) {
  console.error('WF_TOKEN not set')
  process.exit(2)
}

// ── PDF page 1, transcribed ──────────────────────────────────────────────
const PDF = {
  funded: {
    Multifamily: 240410995,
    Retail: 117414277,
    'Self-Storage': 34172689,
    Office: 12243598,
  },
  committed: {
    Multifamily: 280853025,
    Retail: 117868982,
    'Self-Storage': 34172689,
    Office: 12243598,
  },
  totalFunded: 404200000,
  totalCommitted: 445100000,
  dealTypePct: { 'Value-Add': 38, Income: 32, 'New Construction': 30 },
}
//: Allocation buckets that do NOT tie to the PDF, with the reason. Reported,
//: not scored — neither is a charting fault.
//:
//: EMPTY, and both former entries were real bugs rather than data differences —
//: which is the argument for keeping this list short and distrusting it.
//:
//: Multifamily was here first: City West joined the report via
//: KEEP_DESPITE_SOLD (d8cd2f9) and pushed it to $245,396,390, over the
//: published figure by exactly City West's look-through share. The allocation
//: rollups now exclude KEEP_DESPITE_SOLD deals, so it ties.
//:
//: Self-Storage was here next, at "live 32.03M vs PDF 34.17M — pre-existing
//: data difference, unrelated to the chart work". That was WRONG. The gap was
//: entirely Pegasus Life Storage, whose PE-only `pref_equity` was being scaled
//: by the whole-deal look-through %, subtracting OPPEGA's 7.37% twice; it is
//: $2,144,757.65, and the bucket ties the PDF to 28 cents once the PE basis is
//: used. See scripts/snapshot_pe_basis_check.py and `lookthrough_pct`.
//:
//: The staleness guard below fails if an entry here starts matching the PDF, so
//: a stale excuse cannot outlive the thing it was excusing.
const KNOWN_ALLOC_DIFFS = {}
const TOL_USD = 350_000      // $0.35M, same tolerance the Financial check uses
const TOL_PCT = 1.5          // percentage points

const checks = []
const ck = (label, ok, note = '') => checks.push({ label, ok, note })

async function get(path, params) {
  const url = new URL(BASE + path)
  for (const [k, v] of Object.entries(params || {})) url.searchParams.set(k, v)
  const r = await fetch(url, { headers: { Authorization: `Bearer ${TOKEN}` } })
  if (r.status === 401 || r.status === 403) throw new Error(`token rejected (${r.status})`)
  if (!r.ok) throw new Error(`HTTP ${r.status} on ${path}`)
  return r.json()
}

// ── bundle the real .ts module so we import it, not a copy ───────────────
// esbuild's JS API rather than its CLI: node_modules/.bin/esbuild is a POSIX
// shim that Windows spawnSync cannot execute (only esbuild.cmd is runnable), so
// the API keeps this script cross-platform.
async function loadChartOptions() {
  const esbuild = await import(
    'file://' + join(VUE, 'node_modules', 'esbuild', 'lib', 'main.js')
      .replace(/\\/g, '/'))
  const out = join(mkdtempSync(join(tmpdir(), 'chartopts-')), 'chartOptions.mjs')
  await (esbuild.default ?? esbuild).build({
    entryPoints: [join(VUE, 'src', 'components', 'snapshot', 'chartOptions.ts')],
    bundle: true, format: 'esm', platform: 'node', outfile: out, logLevel: 'silent',
  })
  return import('file://' + out.replace(/\\/g, '/'))
}

const usd = (v) => (v == null ? '—' : '$' + Math.round(v).toLocaleString('en-US'))

function main(mod, echarts, summary) {
  const asset = summary.asset_allocation
  const dealType = summary.deal_type_allocation
  const { assetAllocationOption, dealTypeOption, LABEL_MIN_PCT,
        PIE_LABEL_MIN_PCT, segmentLabel } = mod
  const RESIDUAL = new Set(['Unclassified', 'Other', 'Unknown'])

  // ---------- Chart 1 ----------
  const o1 = assetAllocationOption(asset)
  console.log('='.repeat(96))
  console.log('CHART 1 — Asset Allocation (100% stacked, 2 bars)')
  console.log('='.repeat(96))
  console.log(`  x categories: ${JSON.stringify(o1.xAxis.data)}`)
  console.log(`  y axis: ${o1.yAxis.min}-${o1.yAxis.max} step ${o1.yAxis.interval}`
            + `  formatter ${o1.yAxis.axisLabel.formatter}`)
  console.log(`  legend at bottom: ${o1.legend.bottom === 0}`)
  console.log(`  ${o1.series.length} stacked series\n`)
  console.log(`  ${'segment'.padEnd(16)}${'colour'.padEnd(10)}`
            + `${'funded $'.padStart(15)}${'% bar'.padStart(8)}${'label'.padStart(15)}`
            + `${'PDF funded'.padStart(14)}${'Δ'.padStart(10)}`)
  console.log('  ' + '-'.repeat(92))

  let sumFundedPct = 0, sumCommittedPct = 0
  for (const s of o1.series) {
    const [f, c] = s.data
    sumFundedPct += f.value
    sumCommittedPct += c.value
    const lbl = s.label.formatter({ dataIndex: 0 })
    const pdf = PDF.funded[s.name]
    const d = pdf == null ? null : f.usd - pdf
    console.log(`  ${s.name.padEnd(16)}${s.itemStyle.color.padEnd(10)}`
      + `${usd(f.usd).padStart(15)}${f.value.toFixed(1).padStart(8)}`
      + `${(lbl || '(suppressed)').padStart(15)}`
      + `${(pdf == null ? '—' : usd(pdf)).padStart(14)}`
      + `${(d == null ? '' : (d > 0 ? '+' : '') + (d / 1e6).toFixed(2) + 'M').padStart(10)}`)

    // value integrity: the series value must be this bucket's share of its bar
    const bucket = asset.buckets.find((b) => b.label === s.name)
    ck(`${s.name}: funded segment = bucket funded`, Math.abs(f.usd - bucket.funded) < 1)
    ck(`${s.name}: committed segment = bucket committed`,
       Math.abs(c.usd - bucket.committed) < 1)
    ck(`${s.name}: funded % = funded / total_funded`,
       Math.abs(f.value - (bucket.funded / asset.total_funded) * 100) < 0.01)
    // label rule
    const shouldLabel = f.value >= LABEL_MIN_PCT
    // The inside label is $M, not full dollars — see segmentLabel. The
    // reference PDF prints full dollars there; this is a deliberate deviation.
    ck(`${s.name}: label ${shouldLabel ? 'shown' : 'suppressed'} per LABEL_MIN_PCT`,
       shouldLabel ? lbl === segmentLabel(f.usd) : lbl === '')
    // 2px surface gap between stacked fills
    ck(`${s.name}: 2px surface gap`,
       s.itemStyle.borderWidth === 2 && s.itemStyle.borderColor === '#ffffff')
    // text token, not the series colour
    ck(`${s.name}: label ink is a text token`, s.label.color === '#ffffff')
    if (pdf != null) {
      const known = KNOWN_ALLOC_DIFFS[s.name]
      if (known) {
        console.log(`      known difference — ${known}`)
        // Staleness guard: an excused bucket that now ties means the excuse is
        // obsolete. Fail, so it gets deleted rather than shielding a real
        // regression later. Both former entries turned out to be bugs.
        ck(`${s.name}: KNOWN_ALLOC_DIFFS entry is still needed`,
           Math.abs(d) > TOL_USD,
           `now within tolerance (${usd(f.usd)} vs ${usd(pdf)}) — delete the entry`)
      } else {
        ck(`${s.name}: funded within tolerance of PDF`, Math.abs(d) <= TOL_USD,
           `${usd(f.usd)} vs ${usd(pdf)}`)
      }
    }
  }
  console.log(`\n  bars sum to 100%: funded ${sumFundedPct.toFixed(2)}  `
            + `committed ${sumCommittedPct.toFixed(2)}`)
  ck('funded bar sums to 100%', Math.abs(sumFundedPct - 100) < 0.01)
  ck('committed bar sums to 100%', Math.abs(sumCommittedPct - 100) < 0.01)
  ck('colours are the fixed-order categorical hues, not cycled',
     new Set(o1.series.map((s) => s.itemStyle.color)).size === o1.series.length)
  ck('y axis is 0-100%', o1.yAxis.min === 0 && o1.yAxis.max === 100)
  ck('legend present (>=2 series)', !!o1.legend && o1.series.length >= 2)
  ck('every segment reachable on hover', typeof o1.tooltip.formatter === 'function')
  ck('axis labels carry each bar total',
     o1.xAxis.data[0].includes('$') && o1.xAxis.data[1].includes('$'))
  // A deal held back from the allocation must be named, not just netted out.
  const excl = summary.excluded_from_allocation || []
  console.log(`
  excluded from the allocation: `
    + (excl.length ? excl.map((e) => `${e.name} (${e.vcode})`).join(', ')
                   : 'none'))
  ck('any allocation exclusion is named in the payload',
     excl.every((e) => e.vcode && e.name && e.reason))
  ck('no excluded deal leaks into a bucket',
     !summary.asset_allocation.buckets.some((b) => b.deal_count === 0))

  // ---------- Chart 2 ----------
  const o2 = dealTypeOption(dealType)
  const slices = o2.series[0].data
  const tot = slices.reduce((a, s) => a + s.value, 0)
  console.log('\n' + '='.repeat(96))
  console.log('CHART 2 — Deal Type (pie, funded dollars)')
  console.log('='.repeat(96))
  console.log(`  ${'slice'.padEnd(20)}${'colour'.padEnd(10)}${'funded $'.padStart(15)}`
            + `${'%'.padStart(8)}${'PDF %'.padStart(8)}${'Δpp'.padStart(8)}${'deals'.padStart(7)}`)
  console.log('  ' + '-'.repeat(76))
  for (const s of slices) {
    const pct = (s.value / tot) * 100
    const pdf = PDF.dealTypePct[s.name]
    console.log(`  ${s.name.padEnd(20)}${s.itemStyle.color.padEnd(10)}`
      + `${usd(s.value).padStart(15)}${pct.toFixed(1).padStart(8)}`
      + `${(pdf ?? '—').toString().padStart(8)}`
      + `${(pdf == null ? '' : (pct - pdf > 0 ? '+' : '') + (pct - pdf).toFixed(1)).padStart(8)}`
      + `${String(s.deals ?? '—').padStart(7)}`)
    const bucket = dealType.buckets.find((b) => b.label === s.name)
    ck(`${s.name}: slice = bucket funded`, Math.abs(s.value - bucket.funded) < 1)
    if (pdf != null) {
      ck(`${s.name}: share within ${TOL_PCT}pp of PDF`, Math.abs(pct - pdf) <= TOL_PCT,
         `${pct.toFixed(1)}% vs ${pdf}%`)
    }
  }
  ck('pie totals the funded total', Math.abs(tot - dealType.total_funded) < 1)
  ck('pie is on FUNDED dollars, not committed',
     Math.abs(tot - dealType.total_funded) < 1
     && Math.abs(tot - (dealType.total_committed || 0)) > 1)
  // <= MAX_SLOTS, not exactly 3: the backend legitimately emits a residual
  // "Unclassified" bucket (City West at 26Q1). What must hold is that a pie
  // stays inside the slot count and that slivers do not print colliding labels.
  ck(`slice count within MAX_SLOTS (${slices.length})`, slices.length <= 4)
  ck('percentage labels shown inside', o2.series[0].label.position === 'inside'
     && typeof o2.series[0].label.formatter === 'function')
  for (const s of slices) {
    const pct = (s.value / tot) * 100
    const lbl = o2.series[0].label.formatter({ percent: pct })
    ck(`${s.name}: pie label ${pct >= PIE_LABEL_MIN_PCT ? 'shown' : 'suppressed'}`,
       pct >= PIE_LABEL_MIN_PCT ? lbl.endsWith('%') : lbl === '')
    if (RESIDUAL.has(s.name)) {
      ck(`${s.name}: residual wears the muted ink, not a categorical hue`,
         s.itemStyle.color === '#9aa0a6')
    }
  }
  ck('legend below', o2.legend.bottom === 0)
  ck('2px surface ring on slices', o2.series[0].itemStyle.borderWidth === 2)
  ck('slice colours match the bar chart for shared slot order',
     slices[0].itemStyle.color === o1.series[0].itemStyle.color)

  // ---------- render, so it is proven to draw ----------
  console.log('\n' + '='.repeat(96))
  console.log('SSR RENDER')
  console.log('='.repeat(96))
  // NOT inside vue_app/dist: 'vite build' empties that directory, and writing
  // here mid-build made the build itself fail. Gitignored scratch instead.
  const outDir = join(ROOT, 'vue_app', '.chartcheck')
  mkdirSync(outDir, { recursive: true })
  for (const [name, opt, w, h] of [
    ['asset_allocation', o1, 620, 380], ['deal_type', o2, 460, 380],
  ]) {
    const chart = echarts.init(null, null, {
      renderer: 'svg', ssr: true, width: w, height: h,
    })
    // animation:false is REQUIRED for a static SSR render. ECharts emits its
    // marks with entry animations, and CSS animations do not run inside an
    // <img> or a rasteriser — the shapes stay at their zero-size initial frame,
    // so the axes and labels appear and the bars and slices are invisible. The
    // app itself animates normally; this is a rendering-harness concern only.
    chart.setOption({ ...opt, animation: false })
    const svg = chart.renderToSVGString()
    chart.dispose()
    const p = join(outDir, `${name}.svg`)
    writeFileSync(p, svg)
    const texts = [...svg.matchAll(/<text[^>]*>([^<]*)<\/text>/g)].map((m) => m[1])
    console.log(`  ${name}: ${svg.length} bytes, ${texts.length} text nodes -> ${p}`)
    console.log(`      texts: ${JSON.stringify(texts)}`)
    // Count shapes filled with a PALETTE hue — that is a data mark. Counting
    // every <path> would have passed on gridlines alone, and an earlier version
    // of this check used a  word-boundary regex that matched nothing and
    // reported 0 marks on a perfectly good chart. Count the thing you mean.
    const HUES = ['#4472C4', '#E8A33D', '#2E9E8F', '#8E5FA8', '#9aa0a6']
    const marks = (svg.match(/<(?:path|rect|polygon)[^>]*>/g) || [])
      .filter((el) => HUES.some((h) => el.includes(`fill="${h}"`))).length
    // Real geometry, not a zero-size initial animation frame.
    const flat = /<path[^>]*d="[^"]*l0 -?0(?:\.0+)?l/.test(svg)
    console.log(`      ${marks} data marks (palette-filled shapes), `
              + `zero-height marks: ${flat}`)
    ck(`${name}: renders to SVG`, svg.startsWith('<svg') && svg.length > 1000)
    ck(`${name}: has drawn text`, texts.length > 0)
    // The empty-picture guard: 59 checks passed once while the bars were
    // invisible, because every one of them read the option object and none
    // read the output. Count the shapes.
    // 4 asset types x 2 bars + 4 legend swatches = 12; pie 4 slices + 4 swatches.
    ck(`${name}: data marks are actually drawn`, marks >= 6)
    ck(`${name}: marks have real geometry, not a zero-size frame`, !flat)
  }

  // Every non-suppressed dollar label must actually appear in the SVG.
  const svg1 = String(
    (() => { const c = echarts.init(null, null, { renderer: 'svg', ssr: true, width: 620, height: 380 })
             c.setOption({ ...o1, animation: false })
             const s = c.renderToSVGString(); c.dispose(); return s })())
  for (const s of o1.series) {
    const lbl = s.label.formatter({ dataIndex: 0 })
    if (lbl) ck(`${s.name}: funded label "${lbl}" is in the rendered SVG`, svg1.includes(lbl))
  }
}

// PAYLOAD SOURCE. `--payload <file>` reads a summary payload assembled by the
// LOCAL Python backend; without it this falls back to the live endpoint.
//
// The distinction is not cosmetic. The live endpoint runs the DEPLOYED build, so
// a local change to the allocation rollups is invisible through it — this check
// reported pre-fix allocation numbers for exactly that reason while the fix sat
// in the working tree. Produce the local payload with:
//
//   python scripts/snapshot_payload_dump.py summary TGAM 2026-Q1 > p.json
//   node scripts/snapshot_summary_charts_check.mjs --payload p.json
const pi = process.argv.indexOf('--payload')
const payloadFile = pi > -1 ? process.argv[pi + 1] : null

const [mod, echarts, summary] = await Promise.all([
  loadChartOptions(),
  import(pathToFileURL(join(VUE, 'node_modules', 'echarts', 'index.js')).href),
  payloadFile
    ? Promise.resolve(JSON.parse(readFileSync(payloadFile, 'utf8')))
    : get('/api/portfolio-snapshot/summary',
          { investor: 'TGAM', quarter: '2026-Q1' }),
])

const ti = await get('/api/data/version', {})
console.log(`LIVE build=${ti.version}  investor=TGAM quarter=2026-Q1`)
console.log(`payload source: ${payloadFile
  ? `LOCAL assemble_summary (${payloadFile})`
  : 'LIVE /summary — the DEPLOYED backend, so local backend changes will NOT '
    + 'appear; pass --payload to test local code'}\n`)
main(mod, echarts, summary)

const failed = checks.filter((c) => !c.ok)
console.log('\n' + '='.repeat(96))
console.log(`${checks.length - failed.length}/${checks.length} checks passed`)
for (const f of failed) console.log(`    [FAIL] ${f.label}${f.note ? '  — ' + f.note : ''}`)
process.exit(failed.length ? 1 : 0)
