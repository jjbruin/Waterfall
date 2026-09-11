/**
 * Guardrail: print the REAL One Pager PDF for EVERY deal in one run, so a claim
 * about "all N deals still fit one page" is measured rather than extrapolated
 * from a sample.
 *
 * This is scripts/onepager_print_check.mjs widened to a sweep. Same mechanics —
 * serve the real `vue_app/dist` build over http, reverse-proxy /api and /auth to
 * the LIVE app with a Bearer token, seed localStorage['token'] so the router
 * guard lets the page through — but it keeps ONE server up and drives headless
 * Chrome once per deal instead of once per invocation. Rendering 56 deals
 * through the single-deal script means 56 server start/stops and 56 fresh proxy
 * connections; this run is the same work in a tenth of the wall clock.
 *
 * SCOPE. The proxy reaches the DEPLOYED backend, so this is authoritative for
 * TEMPLATE and CSS changes (which is what print geometry is) and NOT for
 * one_pager.py changes. The PDFs it writes are inspected by the companion
 * scripts/onepager_print_geometry.py, which does the measuring.
 *
 * Read-only: GET only against live. Nothing is written outside .chartcheck/.
 *
 * Usage
 *   cd vue_app && npx vite build       # the harness serves dist/, not src/
 *   WF_TOKEN=<jwt> node scripts/onepager_print_sweep.mjs \
 *       --deals scripts/onepager_print_population.txt --tag before
 *
 * The deals file is "<vcode> <quarter>" per line, blank lines and # ignored.
 */
import { createServer } from 'node:http'
import { existsSync, mkdirSync, readFileSync, statSync } from 'node:fs'
import { execFile } from 'node:child_process'
import { join, dirname, extname, normalize } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const ROOT = join(HERE, '..')
const DIST = process.env.WF_DIST || join(ROOT, 'vue_app', 'dist')
const OUT_DIR = join(ROOT, 'vue_app', '.chartcheck')
const UPSTREAM = process.env.WF_UPSTREAM
  || ('https://app-waterfall-dev-v2.icyplant-026fb2db'
      + '.eastus.azurecontainerapps.io')

const TOKEN = process.env.WF_TOKEN
if (!TOKEN) {
  console.error('WF_TOKEN not set')
  process.exit(2)
}
if (!existsSync(join(DIST, 'index.html'))) {
  console.error(`no build at ${DIST} — run: cd vue_app && npx vite build`)
  process.exit(2)
}

const arg = (name, dflt) => {
  const i = process.argv.indexOf(`--${name}`)
  return i > -1 && process.argv[i + 1] ? process.argv[i + 1] : dflt
}
const TAG = arg('tag', 'out')
const DEALS_FILE = arg('deals', join(HERE, 'onepager_print_population.txt'))
const ONLY = arg('only', null)          // comma-separated vcodes, for a spot check

// THE WINDOW WIDTH IS PART OF THE MEASUREMENT, NOT A HARNESS DETAIL.
// The chart is an ECharts canvas at `width: 100%`. ECharts writes the canvas's
// pixel size from the SCREEN container and Chrome runs no JS between applying
// print CSS and painting, so that pixel width is what prints. A narrow window
// therefore produces a narrow canvas that fits the print column, and a wide one
// produces a canvas that overflows it and makes Chrome shrink the whole page.
// Headless Chrome defaults to 800x600, which is narrow enough to hide the
// overflow completely — a sweep run at the default reports a page that no real
// user ever prints. 1280x1024 is the smallest ordinary desktop window and is
// the default here for that reason. Anything wider reproduces identically.
const WINDOW = arg('window', '1280,1024')

const deals = readFileSync(DEALS_FILE, 'utf8')
  .split('\n')
  .map((l) => l.trim())
  .filter((l) => l && !l.startsWith('#'))
  .map((l) => {
    const [vcode, quarter] = l.split(/\s+/)
    return { vcode, quarter: quarter || '2026-Q2' }
  })
  .filter((d) => !ONLY || ONLY.split(',').includes(d.vcode))

if (!deals.length) {
  console.error(`no deals to render from ${DEALS_FILE}`)
  process.exit(2)
}

const CHROME = [
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
].find((p) => existsSync(p))
if (!CHROME) {
  console.error('no Chrome or Edge found')
  process.exit(2)
}

const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css',
  '.json': 'application/json', '.svg': 'image/svg+xml', '.png': 'image/png',
  '.ico': 'image/x-icon', '.woff2': 'font/woff2', '.woff': 'font/woff',
  '.map': 'application/json',
}

function indexHtml() {
  const html = readFileSync(join(DIST, 'index.html'), 'utf8')
  const seed = `<script>localStorage.setItem('token',${JSON.stringify(TOKEN)})`
    + `</script>`
  return html.includes('<head>')
    ? html.replace('<head>', `<head>${seed}`)
    : seed + html
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url, 'http://localhost')
  const path = url.pathname

  if (path.startsWith('/api') || path.startsWith('/auth')) {
    try {
      const up = await fetch(UPSTREAM + path + (url.search || ''), {
        method: req.method,
        headers: { Authorization: `Bearer ${TOKEN}`, Accept: 'application/json' },
      })
      const body = Buffer.from(await up.arrayBuffer())
      res.writeHead(up.status, {
        'content-type': up.headers.get('content-type') || 'application/json',
      })
      return res.end(body)
    } catch (e) {
      res.writeHead(502, { 'content-type': 'application/json' })
      return res.end(JSON.stringify({ error: String(e) }))
    }
  }

  const rel = normalize(path).replace(/^([/\\])+/, '')
  const file = join(DIST, rel)
  if (rel && existsSync(file) && statSync(file).isFile()) {
    res.writeHead(200, {
      'content-type': MIME[extname(file)] || 'application/octet-stream',
    })
    return res.end(readFileSync(file))
  }
  res.writeHead(200, { 'content-type': 'text/html' })
  return res.end(indexHtml())
})

await new Promise((r) => server.listen(0, '127.0.0.1', r))
const port = server.address().port

console.log(`serving ${DIST}`)
console.log(`proxying /api -> ${UPSTREAM}`)
console.log(`${deals.length} deals -> tag "${TAG}"\n`)

mkdirSync(OUT_DIR, { recursive: true })

function printOne({ vcode, quarter }) {
  const pdf = join(OUT_DIR, `onepager_${vcode}_${quarter}_${TAG}.pdf`)
  const target = `http://127.0.0.1:${port}/one-pager`
    + `?vcode=${encodeURIComponent(vcode)}&quarter=${encodeURIComponent(quarter)}`
  // NOT --no-pdf-header-footer: App.vue's `@page { margin: 0 }` is what has to
  // leave Chrome no room for its own header, and that must do the work here too.
  const args = [
    '--headless=new', '--disable-gpu', '--no-sandbox', '--hide-scrollbars',
    `--window-size=${WINDOW}`,
    '--virtual-time-budget=25000',
    '--run-all-compositor-stages-before-draw',
    `--print-to-pdf=${pdf}`,
    target,
  ]
  return new Promise((resolve) => {
    execFile(CHROME, args, { timeout: 180000 }, (err) => {
      const ok = existsSync(pdf)
      resolve({ vcode, quarter, pdf, ok, err: ok ? null : (err && err.message) })
    })
  })
}

// Chrome instances are independent processes; a small pool keeps the wall clock
// down without letting 56 renderers fight over the machine.
const POOL = Number(arg('pool', '4'))
const results = []
let next = 0
await Promise.all(Array.from({ length: Math.min(POOL, deals.length) }, async () => {
  for (;;) {
    const i = next++
    if (i >= deals.length) return
    const r = await printOne(deals[i])
    results.push(r)
    console.log(`  [${results.length}/${deals.length}] ${r.vcode} `
      + (r.ok ? `${statSync(r.pdf).size.toLocaleString()} bytes` : `FAILED ${r.err}`))
  }
}))

const failed = results.filter((r) => !r.ok)
console.log(`\n${results.length - failed.length}/${deals.length} rendered`)
if (failed.length) {
  console.error('FAILED: ' + failed.map((f) => f.vcode).join(', '))
  process.exitCode = 1
}
server.close()
