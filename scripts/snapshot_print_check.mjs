/**
 * Guardrail: produce the REAL printed PDF of the consolidated Portfolio
 * Snapshot document, so pagination can be inspected instead of assumed.
 *
 * Print CSS cannot be verified by reading it. Page breaks, clipped tables and
 * a trailing blank page only exist once a paginating renderer has run, so this
 * drives the actual browser print path end to end:
 *
 *   1. serves the real `vue_app/dist` build over http (a file:// origin cannot
 *      hold localStorage or resolve the SPA's absolute asset paths),
 *   2. reverse-proxies /api and /auth to the LIVE app with the Bearer token, so
 *      the page loads real data,
 *   3. injects one line into index.html seeding localStorage['token'], because
 *      the router guard redirects an unauthenticated visit to /login and the
 *      print route would never mount,
 *   4. drives headless Chrome with --print-to-pdf against the print route.
 *
 * The PDF then gets inspected — page count, and the text on each page — by
 * `scripts/snapshot_print_inspect.py`.
 *
 * Read-only: GET only against live. Nothing is written outside .chartcheck/.
 *
 * Usage
 *   cd vue_app && npm run build          # the harness serves dist/, not src/
 *   WF_TOKEN=<jwt> node scripts/snapshot_print_check.mjs [--investor TGAM] \
 *       [--quarter 2026-Q1] [--keep-open]
 */
import { createServer } from 'node:http'
import { existsSync, mkdirSync, readFileSync, statSync } from 'node:fs'
import { execFile } from 'node:child_process'
import { join, dirname, extname, normalize } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const ROOT = join(HERE, '..')
const DIST = join(ROOT, 'vue_app', 'dist')
const OUT_DIR = join(ROOT, 'vue_app', '.chartcheck')
const UPSTREAM = 'https://app-waterfall-dev-v2.icyplant-026fb2db'
  + '.eastus.azurecontainerapps.io'

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
const INVESTOR = arg('investor', 'TGAM')
const QUARTER = arg('quarter', '2026-Q1')
const KEEP_OPEN = process.argv.includes('--keep-open')

// Produce one with:
//   python scripts/snapshot_payload_dump.py bundle TGAM 2026-Q1 > bundle.json
const bundleFile = arg('bundle', null)
const LOCAL_BUNDLE = bundleFile ? readFileSync(bundleFile, 'utf8') : null

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

/** index.html with the auth token seeded before the SPA boots. */
function indexHtml() {
  const html = readFileSync(join(DIST, 'index.html'), 'utf8')
  const seed = `<script>localStorage.setItem('token',${JSON.stringify(TOKEN)})`
    + `</script>`
  // Before the first module script, so the store reads it on construction.
  return html.includes('<head>')
    ? html.replace('<head>', `<head>${seed}`)
    : seed + html
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url, 'http://localhost')
  const path = url.pathname

  // ---- a locally-assembled bundle, served in place of the proxied one ----
  // The proxy reaches the DEPLOYED backend, which cannot show local assembly
  // work — new subtotals and a changed allocation both went unverified in the
  // printed document that way. --bundle short-circuits that one route.
  if (LOCAL_BUNDLE && path === '/api/portfolio-snapshot/bundle') {
    res.writeHead(200, { 'content-type': 'application/json' })
    return res.end(LOCAL_BUNDLE)
  }

  // ---- proxy the API to live ----
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

  // ---- static, with SPA fallback ----
  const rel = normalize(path).replace(/^([/\\])+/, '')
  const file = join(DIST, rel)
  if (rel && existsSync(file) && statSync(file).isFile()) {
    res.writeHead(200, { 'content-type': MIME[extname(file)] || 'application/octet-stream' })
    return res.end(readFileSync(file))
  }
  res.writeHead(200, { 'content-type': 'text/html' })
  return res.end(indexHtml())
})

await new Promise((r) => server.listen(0, '127.0.0.1', r))
const port = server.address().port
const target = `http://127.0.0.1:${port}/portfolio-snapshot/print`
  + `?investor=${encodeURIComponent(INVESTOR)}`
  + `&quarter=${encodeURIComponent(QUARTER)}`

console.log(`serving ${DIST}`)
console.log(`proxying /api -> ${UPSTREAM}`)
console.log(`printing ${target}\n`)

mkdirSync(OUT_DIR, { recursive: true })
const pdf = join(OUT_DIR, `snapshot_${INVESTOR}_${QUARTER}.pdf`)

// --virtual-time-budget lets the SPA fetch /bundle, mount four subtabs and lay
// out two charts before the snapshot is taken. Too small and pages come out
// empty; this is generous on purpose.
const args = [
  '--headless=new', '--disable-gpu', '--no-sandbox', '--hide-scrollbars',
  '--no-pdf-header-footer',
  '--virtual-time-budget=25000',
  '--run-all-compositor-stages-before-draw',
  `--print-to-pdf=${pdf}`,
  target,
]

await new Promise((resolve) => {
  execFile(CHROME, args, { timeout: 180000 }, (err, stdout, stderr) => {
    const noise = String(stderr || '').split('\n')
      .filter((l) => l.trim() && !/DevTools|Fontconfig|GPU|Vulkan|dbus|voice/i.test(l))
    if (noise.length) console.log(noise.slice(0, 6).join('\n'))
    if (err && !existsSync(pdf)) console.error('chrome failed:', err.message)
    resolve()
  })
})

if (existsSync(pdf)) {
  console.log(`wrote ${pdf} (${statSync(pdf).size.toLocaleString()} bytes)`)
  console.log(`\nnow inspect it:\n  python scripts/snapshot_print_inspect.py "${pdf}"`)
} else {
  console.error('no PDF produced')
}

if (!KEEP_OPEN) server.close()
else console.log(`\nserver still up at http://127.0.0.1:${port} (--keep-open)`)
