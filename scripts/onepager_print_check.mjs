/**
 * Guardrail: produce the REAL printed PDF of a One Pager, so what lands on
 * paper can be inspected instead of assumed.
 *
 * Print CSS cannot be verified by reading it. `.print-date` is the reason this
 * script exists: it is `display:none` on screen and `display:block !important`
 * under `@media print`, so a timestamp that is invisible in the app appeared on
 * every printed sheet — in Chrome's own header format, which is why it was
 * reported as the browser's header bleeding through rather than as ours.
 * Nothing short of rendering the print path shows that.
 *
 * Mirrors scripts/snapshot_print_check.mjs:
 *   1. serves the real `vue_app/dist` build over http (a file:// origin cannot
 *      hold localStorage or resolve the SPA's absolute asset paths),
 *   2. reverse-proxies /api and /auth to the LIVE app with the Bearer token, so
 *      the page loads real data,
 *   3. injects one line into index.html seeding localStorage['token'], because
 *      the router guard would otherwise redirect to /login,
 *   4. drives headless Chrome with --print-to-pdf against /one-pager.
 *
 * SCOPE. The proxy reaches the DEPLOYED backend, so a payload change that is
 * only on this branch will NOT appear in the PDF — the PDF shows the deployed
 * server's numbers rendered by the local build. That makes it authoritative for
 * template and CSS changes and NOT authoritative for one_pager.py changes; use
 * a direct call to the changed function for those.
 *
 * Read-only: GET only against live. Nothing is written outside .chartcheck/.
 *
 * Usage
 *   cd vue_app && npx vite build       # the harness serves dist/, not src/
 *   WF_TOKEN=<jwt> node scripts/onepager_print_check.mjs \
 *       --vcode P0000086 [--quarter 2026-Q2] [--tag after]
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
const VCODE = arg('vcode', 'P0000086')
const QUARTER = arg('quarter', '2026-Q2')
const TAG = arg('tag', 'out')

// Produce one with the companion python script that calls the real
// get_one_pager_data() against live PostgreSQL.
const opFile = arg('onepager', null)
const LOCAL_OP = opFile ? readFileSync(opFile, 'utf8') : null

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

// Chrome's --print-to-pdf prints the page as it stands; it never runs the app's
// own Print handler. That matters here, because `printOnePager()` is what fills
// the `.print-date` div — leave it uninvoked and the div renders empty, so the
// PDF comes out clean and the defect looks fixed when it is not. With
// --click-print the harness stubs window.print (so nothing blocks on a native
// dialog) and clicks the real button, putting the component through exactly the
// path a user takes. Without it the PDF shows a plain Ctrl+P instead.
const CLICK_PRINT = process.argv.includes('--click-print')
const CLICK_JS = `
(function () {
  window.print = function () { document.documentElement.dataset.printCalled = '1' }
  var tries = 0
  var t = setInterval(function () {
    var b = Array.prototype.find.call(
      document.querySelectorAll('button'),
      function (x) { return /^Print/.test((x.textContent || '').trim()) })
    if (b) { clearInterval(t); b.click() }
    else if (++tries > 200) clearInterval(t)
  }, 100)
})()`

function indexHtml() {
  const html = readFileSync(join(DIST, 'index.html'), 'utf8')
  let seed = `<script>localStorage.setItem('token',${JSON.stringify(TOKEN)})`
    + `</script>`
  if (CLICK_PRINT) seed += `<script>${CLICK_JS}</script>`
  return html.includes('<head>')
    ? html.replace('<head>', `<head>${seed}`)
    : seed + html
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url, 'http://localhost')
  const path = url.pathname

  // ---- a locally-computed one-pager payload, served in place of the proxy ----
  // The proxy reaches the DEPLOYED backend, which cannot show a one_pager.py
  // change that is only on this branch. --onepager short-circuits that one
  // route with a payload produced by running the REAL get_one_pager_data
  // locally against the live database, so the PDF still shows the real
  // function's output and not a hand-written number.
  if (LOCAL_OP && path === `/api/financials/${VCODE}/one-pager`) {
    res.writeHead(200, { 'content-type': 'application/json' })
    return res.end(LOCAL_OP)
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
    res.writeHead(200, { 'content-type': MIME[extname(file)] || 'application/octet-stream' })
    return res.end(readFileSync(file))
  }
  res.writeHead(200, { 'content-type': 'text/html' })
  return res.end(indexHtml())
})

await new Promise((r) => server.listen(0, '127.0.0.1', r))
const port = server.address().port
const target = `http://127.0.0.1:${port}/one-pager`
  + `?vcode=${encodeURIComponent(VCODE)}`
  + `&quarter=${encodeURIComponent(QUARTER)}`

console.log(`serving ${DIST}`)
console.log(`proxying /api -> ${UPSTREAM}`)
console.log(`printing ${target}\n`)

mkdirSync(OUT_DIR, { recursive: true })
const pdf = join(OUT_DIR, `onepager_${VCODE}_${QUARTER}_${TAG}.pdf`)

// NOT --no-pdf-header-footer. The whole question this harness answers is what
// appears at the top of the sheet, and passing that flag would suppress the
// browser's header in the harness only — hiding the very distinction being
// checked (ours vs the browser's). App.vue's `@page { margin: 0 }` is what
// leaves Chrome no room to draw one, and that has to do the work here too.
const args = [
  '--headless=new', '--disable-gpu', '--no-sandbox', '--hide-scrollbars',
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
} else {
  console.error('no PDF produced')
  process.exitCode = 1
}
server.close()
