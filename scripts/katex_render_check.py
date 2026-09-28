"""Guardrail: a $$...$$ formula actually renders, and the wiring that makes it.

THE FAILURE THIS EXISTS FOR IS SILENT. If the KaTeX wiring is removed or broken,
nothing errors — the assistant simply prints `$$\\text{Total Cap} = ...$$` as
literal text in the chat bubble. No log, no exception, and it only looks wrong
to someone who knows what it should have looked like.

TWO LAYERS, because the cheap one catches almost everything:

  STATIC (always runs, no browser, no build) — the six things that must be true
  for math to render at all. Each is a real regression someone could introduce
  while tidying:
    * katex declared in package.json
    * the directive declared in `<script setup>` — a `v`-prefixed const in the
      plain `<script>` block sits at module scope, which `resolveDirective` does
      not consult, so `v-typeset` would bind to nothing and do nothing
    * `v-typeset` actually bound on the message bubble
    * the $$ and \\( delimiters configured
    * the typeset gated on stream completion, or a half-arrived `$$` flashes
    * `.katex-display { overflow-x: auto }` — the panel is 420px and display
      math is the one thing that will exceed it

  LIVE (runs only when node + playwright + vue_app/dist are all present, and
  SKIPS with a reason otherwise, so this still runs in the container). Drives
  the real built bundle with a canned SSE stream: asserts an arithmetic answer
  produces a .katex-display node with no literal "$$" left, and that a
  NON-arithmetic answer produces no math at all.

  The second half is not optional padding. "math renders" is satisfied by
  typesetting everything, including the economic-occupancy answer that has no
  equation — which would be fabrication with a stylesheet on it.

    python scripts/katex_render_check.py
"""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VUE = os.path.join(ROOT, "vue_app")
COMPONENT = os.path.join(VUE, "src", "components", "common", "AiAssistant.vue")

PASSED = FAILED = SKIPPED = 0
FAILURES: list = []


def chk(label: str, cond: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  PASS  {label}")
    else:
        FAILED += 1
        FAILURES.append(label)
        print(f"  FAIL  {label}" + (f"  [{detail}]" if detail else ""))


def skip(label: str, why: str) -> None:
    global SKIPPED
    SKIPPED += 1
    print(f"  SKIP  {label}  — {why}")


# ── Static wiring ────────────────────────────────────────────────────────
print("\nStatic wiring")

pkg_path = os.path.join(VUE, "package.json")
pkg = json.load(open(pkg_path, encoding="utf-8"))
chk("katex is a declared dependency",
    "katex" in (pkg.get("dependencies") or {}),
    str(list((pkg.get("dependencies") or {}).keys())))

src = open(COMPONENT, encoding="utf-8").read()
setup_block = src.split("</script>", 1)[0] if "<script setup" in src else ""

chk("the v-typeset directive is declared in <script setup>, not module scope",
    "const vTypeset" in setup_block)
chk("v-typeset is bound on the message bubble", "v-typeset=" in src)
chk("the $$ block delimiter is configured",
    "'$$'" in src or '"$$"' in src)
chk("the inline \\( delimiter is configured", "\\\\(" in src)
chk("typesetting is gated on the message being complete (no mid-stream flash)",
    "isLoading" in src and "v-typeset=" in src
    and "messages.length - 1" in src)
chk("display math scrolls inside the 420px panel rather than widening it",
    ".katex-display" in src and "overflow-x" in src)
chk("KaTeX is lazy-imported, not bundled into the entry chunk",
    "import('katex/contrib/auto-render')" in src
    or 'import("katex/contrib/auto-render")' in src)
chk("a failed import degrades to source text instead of losing the message",
    "catch" in src.split("auto-render", 1)[-1][:900])

# ── Live render ──────────────────────────────────────────────────────────
print("\nLive render")

dist = os.path.join(VUE, "dist")
node = shutil.which("node")
have_pw = False
if node:
    have_pw = subprocess.run(
        [node, "-e", "require.resolve('playwright')"],
        cwd=VUE, capture_output=True).returncode == 0

if not node:
    skip("live render", "node is not on PATH")
elif not os.path.isdir(dist):
    skip("live render", "vue_app/dist absent — run `npm run build` first")
elif not have_pw:
    skip("live render", "playwright not installed under vue_app/")
else:
    driver = r"""
import { chromium } from 'playwright';
import { createServer } from 'http';
import { readFileSync, existsSync } from 'fs';
import { join, extname } from 'path';
const DIST = process.argv[2];
const TYPES = {'.js':'text/javascript','.css':'text/css','.html':'text/html',
  '.woff2':'font/woff2','.woff':'font/woff','.ttf':'font/ttf','.json':'application/json'};
const srv = createServer((q, s) => {
  let p = join(DIST, q.url.split('?')[0]);
  if (!existsSync(p) || q.url === '/') p = join(DIST, 'index.html');
  try { s.writeHead(200, {'Content-Type': TYPES[extname(p)] || 'text/html'});
        s.end(readFileSync(p)); } catch { s.writeHead(404); s.end(); }
}).listen(0);
const port = srv.address().port;
const ARITH = `**Formula**\n$$\\text{Total Cap} = \\text{Debt} + \\text{Total Pref}$$\n\n**Inputs**\ndebt - See debt field`;
const PROSE = `**Formula**\nytd_actual = Avg Physical Occ (YTD) - Bad-Debt %\n(no single-expression arithmetic)\n\n**Inputs**\nytd_actual - MRI`;
function sse(t){let b='';for(let i=0;i<t.length;i+=24)
  b+=`data: ${JSON.stringify({type:'text_delta',text:t.slice(i,i+24)})}\n\n`;
  b+=`data: ${JSON.stringify({type:'done'})}\n\n`;return b;}
const br = await chromium.launch();
const ctx = await br.newContext();
await ctx.addInitScript(() => localStorage.setItem('token','guardrail-local'));
const page = await ctx.newPage();
let answer = ARITH; const flashes = [];
await page.route('**/api/**', r => { const u = r.request().url();
  if (u.includes('/assistant/chat')) return r.fulfill({status:200,
      headers:{'Content-Type':'text/event-stream'}, body: sse(answer)});
  if (u.includes('/assistant/status')) return r.fulfill({status:200,
      contentType:'application/json', body:'{"available":true}'});
  return r.fulfill({status:200, contentType:'application/json', body:'{}'}); });
async function ask(q){
  await page.goto(`http://localhost:${port}/dashboard`,{waitUntil:'domcontentloaded'});
  await page.locator('.ai-fab').click();
  await page.locator('.ai-panel').waitFor({timeout:10000});
  flashes.length = 0;
  const poll = setInterval(async () => { try {
    const el = page.locator('.ai-message--assistant .ai-message-bubble').last();
    const t = await el.innerText();
    if (/\\text\{/.test(t) && (await el.locator('.katex').count()) === 0) flashes.push(1);
  } catch {} }, 30);
  const box = page.locator('.ai-panel textarea, .ai-panel input[type=text]').first();
  await box.fill(q); await box.press('Enter');
  await page.waitForTimeout(3000); clearInterval(poll);
  const el = page.locator('.ai-message--assistant .ai-message-bubble').last();
  return {text: await el.innerText(), katex: await el.locator('.katex').count(),
          disp: await el.locator('.katex-display').count(), flashes: flashes.length};
}
answer = ARITH; const a = await ask('arith');
answer = PROSE; const p2 = await ask('prose');
await br.close(); srv.close();
console.log(JSON.stringify({a, p: p2}));
"""
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", dir=VUE,
                                     delete=False, encoding="utf-8") as fh:
        fh.write(driver)
        drv = fh.name
    try:
        out = subprocess.run([node, drv, dist], cwd=VUE,
                             capture_output=True, text=True, timeout=180)
        line = [l for l in out.stdout.splitlines() if l.startswith("{")]
        if not line:
            chk("the live driver produced a result", False,
                (out.stderr or out.stdout)[-300:])
        else:
            res = json.loads(line[-1])
            a, p = res["a"], res["p"]
            chk("arithmetic answer renders a .katex node", a["katex"] > 0)
            chk("it is DISPLAY math", a["disp"] > 0)
            chk("no literal '$$' is left visible", "$$" not in a["text"])
            chk("no raw \\text{ is left visible", "\\text{" not in a["text"])
            chk("nothing flashed as raw LaTeX mid-stream", a["flashes"] == 0,
                f"{a['flashes']} frame(s)")
            chk("NON-arithmetic answer renders NO math", p["katex"] == 0)
            chk("and leaves no stray '$$'", "$$" not in p["text"])
            chk("and keeps its prose", "Avg Physical Occ" in p["text"])
    finally:
        os.unlink(drv)

print("\n" + "=" * 62)
print(f"RESULT: {PASSED} passed, {FAILED} failed, {SKIPPED} skipped")
if FAILURES:
    for f in FAILURES:
        print(f"  - {f}")
sys.exit(1 if FAILED else 0)
