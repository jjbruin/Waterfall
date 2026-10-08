<script setup lang="ts">
/**
 * The meeting as the package the board will read: a cover, the contents, and
 * each part -- its divider, then its pages in catalog order. ABBREVIATED carries
 * the schedules the engines complete; FULL adds the narrative sections, their
 * attachments, and as many pages as their text needs.
 *
 * PAGE NUMBERS ARE POSITIONS. A page's number is where it sits in the package
 * as printed -- worked out after narrative sections are paginated and after the
 * investment summaries know how many pages they take -- and the contents page
 * prints those same numbers. Switching version renumbers both.
 *
 * LOADED ONCE, FLIPPED FREELY. Each schedule's figures are fetched in the
 * background -- the page on screen first -- and kept, keyed by meeting, schedule
 * and as-of date; turning a page never goes back to the server. Attachment pages
 * are fetched once and kept as images.
 *
 * FOOTNOTES AND DISCLOSURES live on the frame of every page. A page shows its
 * defaults (what the engine says about its figures) until an editor changes
 * them in the panel below the page; the edit previews live and is saved per
 * page of this meeting. Nothing here computes a figure.
 */
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import api from '@/api/client'
import DeckPage from './DeckPage.vue'
import { attachmentBlocks, measureNotes, paginate, parseBody } from './narrative'
import { firstPage } from './slideFormat'

const props = defineProps<{ meeting: any; canEdit: boolean }>()
const emit = defineEmits<{ (e: 'notes-saved', key: string, value: any): void }>()

// ---------------------------------------------------------------- version
const version = ref<'full' | 'abbr'>('full')
try { const v = localStorage.getItem('board.version'); if (v === 'abbr' || v === 'full') version.value = v } catch { /* none */ }
watch(version, (v) => { try { localStorage.setItem('board.version', v) } catch { /* none */ } })

// ---------------------------------------------------------------- figures
const CACHE: Map<string, any> = (globalThis as any).__boardDeckCache ||= new Map()
const status = reactive<Record<string, 'loading' | 'ready' | 'error'>>({})
const errors = reactive<Record<string, string>>({})
const cacheKey = (s: any) => `${props.meeting.id}|${s.key}|${s.as_of}`
const views = computed(() => props.meeting.schedules.filter((s: any) => s.included && s.view))
const pending = computed(() => props.meeting.schedules.filter((s: any) => s.included && !s.view))
function viewOf(s: any) { void status[cacheKey(s)]; return CACHE.get(cacheKey(s)) }

let running = false
async function load(s: any) {
  const k = cacheKey(s)
  if (CACHE.has(k)) { status[k] = 'ready'; return }
  status[k] = 'loading'
  try {
    const r = await api.get(`/api/board/meetings/${props.meeting.id}/schedules/${s.key}/view`)
    CACHE.set(k, r.data); status[k] = 'ready'
  } catch (e: any) {
    status[k] = 'error'
    errors[k] = e?.response?.data?.error || e?.response?.data?.message || String(e)
  }
}
async function prefetch() {
  if (running) return
  running = true
  try {
    for (;;) {
      const want = [current.value?.sched, ...views.value].filter(Boolean)
        .find((s: any) => !CACHE.has(cacheKey(s)) && status[cacheKey(s)] !== 'error')
      if (!want) break
      await load(want)
    }
  } finally { running = false }
}
function refresh() {
  for (const k of [...CACHE.keys()]) if (k.startsWith(props.meeting.id + '|')) CACHE.delete(k)
  for (const k of Object.keys(status)) if (k.startsWith(props.meeting.id + '|')) delete status[k]
  prefetch()
}
const loaded = computed(() => views.value.filter((s: any) => status[cacheKey(s)] === 'ready').length)

// ---------------------------------------------------------------- attachments
const IMAGES: Map<string, string> = (globalThis as any).__boardDeckImages ||= new Map()
const images = reactive<Record<string, string>>({})
async function loadImages() {
  for (const n of props.meeting.narratives || []) {
    for (const a of n.attachments || []) {
      for (const pg of a.pages || []) {
        const k = `${a.id}:${pg.n}`
        if (images[k]) continue
        const ck = `${props.meeting.id}|${k}`
        if (IMAGES.has(ck)) { images[k] = IMAGES.get(ck) as string; continue }
        try {
          const r = await api.get(`/api/board/meetings/${props.meeting.id}/attachments/${a.id}/pages/${pg.n}`,
            { responseType: 'blob' })
          const url = URL.createObjectURL(r.data)
          IMAGES.set(ck, url); images[k] = url
        } catch { /* the page keeps its "Loading attachment…" box */ }
      }
    }
  }
}

// ---------------------------------------------------------------- footnotes
const editing = ref(false)
const draft = reactive<{ key: string; footnotes: string[]; disclosure: string }>({ key: '', footnotes: [], disclosure: '' })
const saving = ref(false)
const notesError = ref('')

function defaults(sl: any): string[] {
  if (!sl.sched) return []
  const v = viewOf(sl.sched)
  if (!v) return []
  if (sl.kind === 'investment_summaries' && sl.spec) {
    const t = v.investment_metrics?.[sl.spec.table]
    return [sl.spec.footnote, ...(sl.spec.is_last ? (t?.footnotes || []).map((f: any) => `(${f.n}) ${f.text}`) : [])]
  }
  return v.footnotes || []
}
function stored(key: string) { return (props.meeting.page_notes || {})[key] }
function notesFor(sl: any): { footnotes: string[]; disclosure: string } {
  if (sl.showNotes === false) return { footnotes: [], disclosure: '' }
  if (editing.value && draft.key === sl.noteKey) {
    return { footnotes: draft.footnotes.filter((f) => f.trim()), disclosure: draft.disclosure }
  }
  const st = stored(sl.noteKey)
  return { footnotes: st?.footnotes ?? defaults(sl), disclosure: st?.disclosure || '' }
}

// ---------------------------------------------------------------- the package
const fontsReady = ref(0)
interface Slide { id: string; kind: string; title: string; noteKey: string; section: string; part?: string
  roman?: string; sched?: any; spec?: any; blocks?: any[]; showNotes?: boolean }

const sections = computed(() => {
  const parts = (props.meeting.parts || []).map((p: any) => ({ ...p, items: [] as any[] }))
  const byKey: Record<string, any> = Object.fromEntries(parts.map((p: any) => [p.key, p]))
  for (const s of views.value) byKey[s.part]?.items.push({ kind: 'sched', s, page: firstPage(s.pages) })
  if (version.value === 'full') {
    for (const n of props.meeting.narratives || []) {
      if ((n.body || '').trim() || (n.attachments || []).length) byKey[n.part]?.items.push({ kind: 'narr', n, page: firstPage(n.pages) })
    }
  }
  for (const p of parts) p.items.sort((a: any, b: any) => a.page - b.page || (a.kind === 'narr' ? 1 : -1))
  return parts.filter((p: any) => p.items.length)
})

const slides = computed<Slide[]>(() => {
  void fontsReady.value
  const out: Slide[] = [
    { id: 'cover', kind: 'cover', title: 'Cover', noteKey: 'cover', section: 'cover' },
    { id: 'contents', kind: 'contents', title: 'Table of Contents', noteKey: 'contents', section: 'contents' },
  ]
  for (const p of sections.value) {
    out.push({ id: 'part:' + p.key, kind: 'divider', title: p.title, roman: p.key, noteKey: 'part:' + p.key,
      section: 'part:' + p.key, part: p.key })
    for (const it of p.items) {
      if (it.kind === 'sched') {
        const s = it.s
        const v = viewOf(s)
        if (s.key === 'investment_summaries' && v?.deck) {
          v.deck.slides.forEach((spec: any, i: number) => out.push({ id: `${s.key}:${i}`, kind: s.key,
            title: spec.title.replace(/\*$/, ''), noteKey: `${s.key}:${i}`, section: s.key, part: p.key, sched: s, spec }))
        } else {
          out.push({ id: s.key, kind: s.key, title: v?.slide_title || s.title, noteKey: s.key, section: s.key,
            part: p.key, sched: s })
        }
      } else {
        const n = it.n
        const key = 'n:' + n.key
        const st = stored(key)
        const last = editing.value && draft.key === key
          ? { f: draft.footnotes.filter((f) => f.trim()), d: draft.disclosure }
          : { f: st?.footnotes ?? [], d: st?.disclosure || '' }
        const blocks = [...parseBody(n.body || ''), ...attachmentBlocks(n.attachments || [])]
        const pages = paginate(blocks, measureNotes(last.f, last.d))
        pages.forEach((b, i) => out.push({ id: `${key}:${i}`, kind: 'narrative',
          title: i === 0 ? n.title : `${n.title} (cont’d)`, noteKey: key, section: key, part: p.key,
          blocks: b, showNotes: i === pages.length - 1 }))
      }
    }
  }
  return out
})

const contents = computed(() => {
  const at = (sec: string) => slides.value.findIndex((s) => s.section === sec) + 1
  return sections.value.map((p: any) => ({
    roman: p.key, title: p.title, page: at('part:' + p.key),
    items: p.items.map((it: any) => it.kind === 'sched'
      ? { title: it.s.title, page: at(it.s.key) }
      : { title: it.n.title, page: at('n:' + it.n.key) }),
  }))
})

const index = ref(0)
const current = computed(() => slides.value[Math.min(index.value, slides.value.length - 1)])
function go(i: number) { index.value = Math.max(0, Math.min(slides.value.length - 1, i)); cancelEdit() }
const statusOf = (sl: any) => (sl?.sched ? status[cacheKey(sl.sched)] : 'ready')
const sig = (sl: any) => { const n = notesFor(sl); return sl.id + '|' + n.footnotes.join('\u0001') + '|' + n.disclosure }

watch(() => views.value.map(cacheKey).join(','), () => {
  for (const s of views.value) if (CACHE.has(cacheKey(s))) status[cacheKey(s)] = 'ready'
  prefetch()
}, { immediate: true })
watch(() => [version.value, JSON.stringify((props.meeting.narratives || []).map((n: any) => (n.attachments || []).map((a: any) => a.id)))],
  () => { if (version.value === 'full') loadImages() }, { immediate: true })
watch(() => slides.value.length, () => { if (index.value >= slides.value.length) index.value = slides.value.length - 1 })

// ---------------------------------------------------------------- the editor
function startEdit() {
  const sl = current.value
  if (!sl) return
  const n = notesFor(sl.showNotes === false ? { ...sl, showNotes: true } : sl)
  draft.key = sl.noteKey; draft.footnotes = [...n.footnotes]; draft.disclosure = n.disclosure || ''
  notesError.value = ''; editing.value = true
}
function cancelEdit() { editing.value = false; notesError.value = '' }
function move(i: number, d: number) {
  const j = i + d
  if (j < 0 || j >= draft.footnotes.length) return
  const f = draft.footnotes;[f[i], f[j]] = [f[j], f[i]]
}
async function saveNotes(reset = false) {
  saving.value = true; notesError.value = ''
  try {
    const body = reset ? { reset: true } : { footnotes: draft.footnotes, disclosure: draft.disclosure }
    const r = await api.put(`/api/board/meetings/${props.meeting.id}/pages/${encodeURIComponent(draft.key)}/notes`, body)
    emit('notes-saved', draft.key, reset ? null : r.data)
    editing.value = false
  } catch (e: any) {
    notesError.value = e?.response?.data?.error || String(e)
  } finally { saving.value = false }
}
const isDefault = computed(() => !stored(current.value?.noteKey || ''))

// ---------------------------------------------------------------- stage, keys, full screen
const stage = ref<HTMLElement | null>(null)
const scale = ref(1)
const isFull = ref(false)
const offset = ref({ x: 0, y: 0 })
let ro: ResizeObserver | null = null
function fit() {
  const el = stage.value
  if (!el) return
  const full = document.fullscreenElement === el
  isFull.value = full
  const w = el.clientWidth - (full ? 0 : 2)
  const h = full ? el.clientHeight : Math.max(320, window.innerHeight - el.getBoundingClientRect().top - 24)
  scale.value = Math.max(0.2, Math.min(w / 1100, h / 825))
  offset.value = { x: Math.max(0, (w - 1100 * scale.value) / 2), y: full ? Math.max(0, (h - 825 * scale.value) / 2) : 0 }
}
function fullscreen() {
  if (document.fullscreenElement) document.exitFullscreen()
  else stage.value?.requestFullscreen?.()
}
function onKey(e: KeyboardEvent) {
  const t = e.target as HTMLElement
  if (t && /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName)) return
  if (['ArrowRight', 'PageDown'].includes(e.key)) { go(index.value + 1); e.preventDefault() }
  else if (['ArrowLeft', 'PageUp'].includes(e.key)) { go(index.value - 1); e.preventDefault() }
  else if (e.key === 'Home') go(0)
  else if (e.key === 'End') go(slides.value.length - 1)
}

// ---------------------------------------------------------------- print
const printing = ref(false)
const printReady = computed(() => loaded.value === views.value.length)
async function printPackage() {
  cancelEdit()
  printing.value = true
  await nextTick()
  await (document as any).fonts?.ready
  const imgs = Array.from(document.querySelectorAll('.bd-print img')) as HTMLImageElement[]
  await Promise.all(imgs.map((im) => (im.complete ? null : new Promise((r) => { im.onload = r; im.onerror = r }))))
  await new Promise((r) => setTimeout(r, 400))       // the tables that fit their type settle
  window.print()
}
function afterPrint() { printing.value = false }

onMounted(async () => {
  window.addEventListener('keydown', onKey)
  window.addEventListener('resize', fit)
  window.addEventListener('afterprint', afterPrint)
  document.addEventListener('fullscreenchange', fit)
  await nextTick()
  if (stage.value) { ro = new ResizeObserver(fit); ro.observe(stage.value) }
  fit()
  ;(document as any).fonts?.ready?.then(() => { fontsReady.value++ })
})
onBeforeUnmount(() => {
  window.removeEventListener('keydown', onKey)
  window.removeEventListener('resize', fit)
  window.removeEventListener('afterprint', afterPrint)
  document.removeEventListener('fullscreenchange', fit)
  ro?.disconnect()
})

const showReview = ref(false)
const reviewNotes = computed<string[]>(() => (current.value?.sched ? viewOf(current.value.sched)?.notes || [] : []))
</script>

<template>
  <div class="deck">
    <aside class="rail">
      <div class="ver">
        <button :class="{ on: version === 'full' }" @click="version = 'full'">Full</button>
        <button :class="{ on: version === 'abbr' }" @click="version = 'abbr'">Abbreviated</button>
      </div>
      <button v-for="(s, i) in slides" :key="s.id" class="thumb" :class="{ on: i === index, sub: s.kind !== 'divider' && s.kind !== 'cover' && s.kind !== 'contents' }"
              @click="go(i)">
        <span class="pg">{{ i + 1 }}</span>
        <span class="tt">{{ s.kind === 'divider' ? s.roman + '. ' + s.title : s.title }}</span>
        <span v-if="s.sched" class="dot" :class="status[cacheKey(s.sched)] || 'loading'"></span>
      </button>
      <div v-if="pending.length" class="muted small later"
           :title="pending.map((s: any) => 'p. ' + s.pages + '  ' + s.title).join('\n')">
        {{ pending.length }} included schedule{{ pending.length === 1 ? '' : 's' }} not built yet</div>
    </aside>

    <section class="main">
      <div class="bar">
        <button class="btn" :disabled="index === 0" @click="go(index - 1)">‹ Prev</button>
        <span class="count">{{ index + 1 }} / {{ slides.length }}</span>
        <button class="btn" :disabled="index >= slides.length - 1" @click="go(index + 1)">Next ›</button>
        <span class="muted small">
          <template v-if="loaded < views.length">Loading figures… {{ loaded }} of {{ views.length }}</template>
          <template v-else>All figures loaded; pages turn without recalculating.</template>
        </span>
        <span class="grow"></span>
        <span v-if="current?.sched" class="muted small">as of {{ current.sched.as_of }}</span>
        <button class="btn" title="Fetch every schedule again (after a data refresh)" @click="refresh">Refresh figures</button>
        <button class="btn" @click="fullscreen">Full screen</button>
        <button class="btn primary" :disabled="!printReady" :title="printReady ? 'Print, or choose Save as PDF' : 'Waiting for figures'"
                @click="printPackage">Print / PDF</button>
      </div>

      <div ref="stage" class="stage" :style="isFull ? {} : { height: 825 * scale + 2 + 'px' }">
        <div class="canvas" :style="{ transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})` }">
          <DeckPage v-if="current" :key="sig(current)" :slide="current" :page="index + 1"
                    :view="current.sched ? viewOf(current.sched) : null" :status="statusOf(current)"
                    :error="current.sched ? errors[cacheKey(current.sched)] : ''" :notes="notesFor(current)"
                    :images="images" :contents="contents" :meeting-date="meeting.meeting_date" :version="version"
                    @go="(p: number) => go(p - 1)" @retry="load(current.sched)" />
        </div>
      </div>

      <!-- FOOTNOTES & DISCLOSURE for the page on screen -->
      <div class="notes-panel">
        <div class="np-head">
          <strong>Footnotes &amp; disclosure</strong>
          <span class="muted small">page {{ index + 1 }}
            <template v-if="current?.kind === 'narrative'"> · shown on the last page of this section</template>
            · {{ isDefault ? 'default' : 'edited' + (stored(current?.noteKey || '')?.updated_by ? ' by ' + stored(current?.noteKey || '').updated_by : '') }}</span>
          <span class="grow"></span>
          <button v-if="canEdit && !editing" class="btn" @click="startEdit">Edit</button>
          <span v-else-if="!canEdit" class="muted small">read only</span>
        </div>
        <template v-if="editing">
          <div v-for="(f, i) in draft.footnotes" :key="i" class="np-row">
            <textarea v-model="draft.footnotes[i]" rows="1" placeholder="Footnote text"></textarea>
            <button class="ico" title="Move up" :disabled="i === 0" @click="move(i, -1)">↑</button>
            <button class="ico" title="Move down" :disabled="i === draft.footnotes.length - 1" @click="move(i, 1)">↓</button>
            <button class="ico" title="Remove" @click="draft.footnotes.splice(i, 1)">✕</button>
          </div>
          <button class="link" @click="draft.footnotes.push('')">+ Add footnote</button>
          <label class="np-disc">Disclosure (small type at the foot of the page)
            <textarea v-model="draft.disclosure" rows="2" placeholder="e.g. Returns are gross of fees…"></textarea></label>
          <div class="np-actions">
            <button class="btn primary" :disabled="saving" @click="saveNotes(false)">Save</button>
            <button class="btn" :disabled="saving" @click="cancelEdit">Cancel</button>
            <button v-if="!isDefault" class="btn" :disabled="saving" title="Back to the page's own footnotes, no disclosure"
                    @click="saveNotes(true)">Reset to defaults</button>
            <span class="muted small">The page above previews your edit.</span>
            <span v-if="notesError" class="err small">{{ notesError }}</span>
          </div>
        </template>
      </div>

      <div v-if="reviewNotes.length" class="review">
        <button class="link" @click="showReview = !showReview">
          {{ showReview ? '▾' : '▸' }} Reviewer notes for this schedule ({{ reviewNotes.length }}) — not shown on the page</button>
        <ul v-if="showReview"><li v-for="(n, i) in reviewNotes" :key="i">{{ n }}</li></ul>
      </div>
    </section>

    <!-- The printed package: every page at full size, drawn by the same component. -->
    <Teleport to="body">
      <div v-if="printing" class="bd-print">
        <div v-for="(s, i) in slides" :key="'p' + s.id" class="bd-page">
          <DeckPage :slide="s" :page="i + 1" :view="s.sched ? viewOf(s.sched) : null" :status="statusOf(s)"
                    :notes="notesFor(s)" :images="images" :contents="contents" :meeting-date="meeting.meeting_date"
                    :version="version" />
        </div>
      </div>
    </Teleport>
  </div>
</template>

<style scoped>
.deck { display: flex; gap: 14px; align-items: flex-start; }
.rail { width: 220px; flex: none; display: flex; flex-direction: column; gap: 3px; max-height: calc(100vh - 200px);
  overflow-y: auto; }
.ver { display: flex; margin-bottom: 6px; border: 1px solid var(--color-border, #d1d5db); border-radius: 6px; overflow: hidden; flex: none; }
.ver button { flex: 1; padding: 5px; border: none; background: var(--color-surface, #fff); color: inherit; cursor: pointer; font-size: 12.5px; }
.ver button.on { background: #00b274; color: #fff; font-weight: 600; }
.thumb { display: grid; grid-template-columns: 26px 1fr 10px; align-items: center; gap: 6px; text-align: left;
  padding: 6px 8px; border: 1px solid var(--color-border, #e5e7eb); border-radius: 6px; cursor: pointer;
  background: var(--color-surface, #fff); color: inherit; font-size: 12.5px; }
.thumb.sub { margin-left: 10px; }
.thumb.on { border-color: #00b274; box-shadow: inset 3px 0 0 #00b274; font-weight: 600; }
.thumb .pg { color: var(--color-text-muted, #6b7280); font-variant-numeric: tabular-nums; }
.thumb .tt { line-height: 1.25; }
.dot { width: 8px; height: 8px; border-radius: 50%; background: #d1d5db; }
.dot.ready { background: #00b274; }
.dot.loading { background: #f59e0b; }
.dot.error { background: #dc2626; }
.later { margin-top: 8px; }
.main { flex: 1; min-width: 0; width: 100%; }
.bar { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; flex-wrap: wrap; }
.grow { flex: 1; }
.count { font-variant-numeric: tabular-nums; min-width: 48px; text-align: center; }
.btn { padding: 4px 10px; border-radius: 6px; font-size: 13px; cursor: pointer; border: 1px solid var(--color-border, #d1d5db);
  background: var(--color-surface, #fff); color: inherit; }
.btn.primary { background: #00b274; border-color: #00b274; color: #fff; font-weight: 600; }
.btn:disabled { opacity: .45; cursor: default; }
.stage { position: relative; width: 100%; overflow: hidden; background: #e5e7eb; border: 1px solid #d1d5db; }
.stage:fullscreen { background: #111; border: none; }
.canvas { position: absolute; left: 0; top: 0; width: 1100px; height: 825px; transform-origin: top left;
  box-shadow: 0 1px 6px rgba(0, 0, 0, .25); }
.notes-panel { margin-top: 10px; border: 1px solid var(--color-border, #e5e7eb); border-radius: 6px; padding: 8px 10px; }
.np-head { display: flex; align-items: center; gap: 10px; }
.np-row { display: flex; gap: 6px; align-items: flex-start; margin-top: 6px; }
.np-row textarea { flex: 1; font: inherit; font-size: 13px; padding: 4px 6px; resize: vertical; }
.ico { width: 26px; height: 26px; border: 1px solid var(--color-border, #d1d5db); border-radius: 4px; background: var(--color-surface, #fff);
  color: inherit; cursor: pointer; }
.ico:disabled { opacity: .35; }
.np-disc { display: flex; flex-direction: column; gap: 3px; margin-top: 8px; font-size: 12.5px; }
.np-disc textarea { font: inherit; font-size: 13px; padding: 4px 6px; }
.np-actions { display: flex; align-items: center; gap: 8px; margin-top: 8px; flex-wrap: wrap; }
.review { margin-top: 8px; font-size: 13px; }
.review ul { margin: 6px 0 0; padding-left: 20px; color: var(--color-text-muted, #4b5563); }
.link { background: none; border: none; padding: 0; color: inherit; cursor: pointer; font-size: 13px; margin-top: 6px; }
.muted { color: var(--color-text-muted, #6b7280); }
.small { font-size: 12px; }
.err { color: #b42318; }
@media (max-width: 1200px) {
  .deck { flex-direction: column; }
  .rail { width: 100%; max-height: none; flex-direction: row; overflow-x: auto; overflow-y: hidden; padding-bottom: 4px; }
  .ver { width: 170px; margin: 0 6px 0 0; }
  .thumb { flex: none; width: 180px; }
  .thumb.sub { margin-left: 0; }
  .later { margin: 0 0 0 8px; align-self: center; white-space: nowrap; }
}
</style>

<style>
/* The printed package. Off-screen (but laid out) while it is built, so the
   tables that size their type can measure themselves; alone on the page when printed. */
@media screen { .bd-print { position: fixed; left: -40000px; top: 0; } }
@media print {
  @page { size: 1100px 825px; margin: 0; }
  /* The fills ARE the formatting -- the green rule, banded rows, header shading,
     the bars, the banner. Browsers drop backgrounds when printing unless the
     dialog's "Background graphics" is ticked (it is not, by default); this makes
     them print regardless. Measured Oct 8 2026: without it, Save as PDF lost all
     of them. */
  .bd-print, .bd-print * { -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }
  body > *:not(.bd-print) { display: none !important; }
  html, body { background: #fff !important; }
  .bd-print { position: static; }
  .bd-print .bd-page { width: 1100px; height: 825px; overflow: hidden; break-after: page; page-break-after: always; }
  .bd-print .bd-page:last-child { break-after: auto; page-break-after: auto; }
}
</style>
