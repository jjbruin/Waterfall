/**
 * Guardrail: produce the REAL printed PDF of the Investment Metrics report, so
 * the sheet can be measured against the reference instead of assumed.
 *
 * Print CSS cannot be verified by reading it. Column widths, the page break
 * between Current and Sold, a table that overruns the right margin and a
 * trailing blank sheet only exist once a paginating renderer has run, so this
 * drives the browser print path end to end:
 *
 *   1. serves the real `vue_app/dist` build over http (a file:// origin cannot
 *      hold localStorage or resolve the SPA's absolute asset paths),
 *   2. answers `/api/investment-metrics` either from a LOCAL payload file or by
 *      proxying to a running backend,
 *   3. injects one line into index.html seeding localStorage['token'], because
 *      the router guard would otherwise redirect to /login and the print route
 *      would never mount,
 *   4. drives headless Chrome with --print-to-pdf against the print route.
 *
 * The PDF is then measured by `scripts/investment_metrics_print_inspect.py`,
 * which is where the pass/fail lives.
 *
 * --payload IS THE POINT, not a convenience. The proxy reaches a DEPLOYED
 * backend, which by definition cannot serve an endpoint that has not shipped
 * yet — so without it a new report can never have its printed page checked
 * before it goes out, which is the one moment the check is worth anything.
 *
 * Read-only: GET only, and nothing is written outside vue_app/.chartcheck/.
 *
 * Usage
 *   cd vue_app && npx vite build          # the harness serves dist/, not src/
 *   node scripts/investment_metrics_print_check.mjs --payload out.json
 *   WF_TOKEN=<jwt> node scripts/investment_metrics_print_check.mjs   # proxied
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

const arg = (name, dflt) => {
  const i = process.argv.indexOf(`--${name}`)
  return i > -1 && process.argv[i + 1] ? process.argv[i + 1] : dflt
}

const PAYLOAD_FILE = arg('payload', null)
const PAYLOAD = PAYLOAD_FILE ? readFileSync(PAYLOAD_FILE, 'utf8') : null
const AS_OF = arg('as-of', '')
const KEEP_OPEN = process.argv.includes('--keep-open')
const UPSTREAM = process.env.WF_UPSTREAM
  || ('https://app-waterfall-dev-v2.icyplant-026fb2db'
      + '.eastus.azurecontainerapps.io')
// A token is only needed when the API is proxied; a local payload needs none
// beyond something for the router guard to find.
const TOKEN = process.env.WF_TOKEN || 'local-harness'

if (!PAYLOAD && !process.env.WF_TOKEN) {
  console.error('set WF_TOKEN to proxy the API, or pass --payload <file.json>')
  process.exit(2)
}
if (!existsSync(join(DIST, 'index.html'))) {
  console.error(`no build at ${DIST} — run: cd vue_app && npx vite build`)
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

/** index.html with the auth token seeded before the SPA boots. */
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

  // THE ROUTER GUARD CALLS /auth/me BEFORE THE ROUTE MOUNTS, and `fetchMe`
  // logs out on any failure — so with the API proxied to a live backend and a
  // placeholder token, the guard silently redirects and the harness prints the
  // LOGIN PAGE. It looks like a print-CSS problem and is not one.
  if (PAYLOAD && path === '/auth/me') {
    res.writeHead(200, { 'content-type': 'application/json' })
    return res.end(JSON.stringify({
      user: { id: 0, username: 'print-harness', role: 'admin' },
    }))
  }

  if (PAYLOAD && path === '/api/investment-metrics') {
    res.writeHead(200, { 'content-type': 'application/json' })
    return res.end(PAYLOAD)
  }
  if (PAYLOAD && path === '/api/investment-metrics/quarters') {
    const d = JSON.parse(PAYLOAD).as_of
    res.writeHead(200, { 'content-type': 'application/json' })
    return res.end(JSON.stringify({ quarters: [d], default: d }))
  }

  // IN PAYLOAD MODE NOTHING MAY 401. The axios response interceptor turns any
  // 401 into `window.location.href = '/login'`, so a single unrelated call
  // that the payload does not cover — a config fetch, a store warming itself —
  // navigates the whole page away and the harness prints the sign-in form.
  // That failure is silent: the PDF is produced, it is one portrait page, and
  // nothing says the print route never mounted.
  if (PAYLOAD && (path.startsWith('/api') || path.startsWith('/auth'))) {
    res.writeHead(200, { 'content-type': 'application/json' })
    return res.end('{}')
  }

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
// autoprint=0 — the view prints itself on mount, which races Chrome's own
// --print-to-pdf and can capture a half-laid-out page.
const target = `http://127.0.0.1:${port}/investment-metrics/print?autoprint=0`
  + (AS_OF ? `&as_of=${encodeURIComponent(AS_OF)}` : '')

console.log(`serving ${DIST}`)
console.log(PAYLOAD ? `payload  ${PAYLOAD_FILE}` : `proxying /api -> ${UPSTREAM}`)
console.log(`printing ${target}\n`)

mkdirSync(OUT_DIR, { recursive: true })
const pdf = join(OUT_DIR, 'investment_metrics.pdf')

const args = [
  '--headless=new', '--disable-gpu', '--no-sandbox', '--hide-scrollbars',
  '--no-pdf-header-footer',
  '--virtual-time-budget=20000',
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
  console.log('\nnow measure it:\n  python scripts/investment_metrics_print_inspect.py '
    + `"${pdf}"`)
} else {
  console.error('no PDF produced')
  process.exitCode = 1
}

if (!KEEP_OPEN) server.close()
else console.log(`\nserver still up at http://127.0.0.1:${port} (--keep-open)`)
