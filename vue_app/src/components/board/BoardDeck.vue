<script setup lang="ts">
/**
 * The meeting as the board will see it: every schedule that has a view, plus
 * every narrative with text, as deck pages in page order.
 *
 * LOADED ONCE, FLIPPED FREELY. When the deck opens, each schedule's figures are
 * fetched in the background -- the page on screen first, the rest one after
 * another (the server has one worker, so asking in parallel only queues) -- and
 * kept here, keyed by meeting, schedule and as-of date. Turning a page never
 * goes back to the server. Changing a schedule's as-of date changes its key, so
 * only that schedule is fetched again; "Refresh figures" drops this meeting's
 * copies after a data reload.
 *
 * The canvas is the deck's (1100 x 825) and is SCALED to the space available,
 * so the layout on screen is the layout of the page. Nothing here computes a
 * figure; the engines' notes on what they could not supply are shown beside the
 * page for the reviewer, never on it.
 */
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import api from '@/api/client'
import AssetClassSlide from './AssetClassSlide.vue'
import CapitalizationSlide from './CapitalizationSlide.vue'
import PerformanceSlide from './PerformanceSlide.vue'
import InvestmentSummarySlide from './InvestmentSummarySlide.vue'
import NarrativeSlide from './NarrativeSlide.vue'
import PortfolioMetricsSlide from './PortfolioMetricsSlide.vue'
import { firstPage } from './slideFormat'

const props = defineProps<{ meeting: any }>()

// Module scope: survives switching tabs and meetings for as long as the app is open.
const CACHE: Map<string, any> = (globalThis as any).__boardDeckCache ||= new Map()
const status = reactive<Record<string, 'loading' | 'ready' | 'error'>>({})
const errors = reactive<Record<string, string>>({})

const cacheKey = (s: any) => `${props.meeting.id}|${s.key}|${s.as_of}`
const views = computed(() => props.meeting.schedules.filter((s: any) => s.included && s.view))
const pending = computed(() => props.meeting.schedules.filter((s: any) => s.included && !s.view))

interface Slide { id: string; page: number; label: string; title: string; kind: string;
  sched?: any; spec?: any; narrative?: any }

const slides = computed<Slide[]>(() => {
  const out: Slide[] = []
  for (const s of views.value) {
    const p = firstPage(s.pages)
    const v = CACHE.get(cacheKey(s))
    void status[cacheKey(s)]          // re-run when a load lands
    if (s.key === 'investment_summaries' && v?.deck) {
      v.deck.slides.forEach((spec: any, i: number) => out.push({
        id: `${s.key}:${i}`, page: p + i, label: String(p + i), title: spec.title.replace(/\*$/, ''),
        kind: s.key, sched: s, spec }))
    } else {
      out.push({ id: s.key, page: p, label: String(p), title: v?.slide_title || s.title, kind: s.key, sched: s })
    }
  }
  for (const n of props.meeting.narratives || []) {
    if ((n.body || '').trim()) {
      const p = firstPage(n.pages)
      out.push({ id: 'n:' + n.key, page: p, label: String(p), title: n.title, kind: 'narrative', narrative: n })
    }
  }
  return out.sort((a, b) => a.page - b.page || a.id.localeCompare(b.id))
})

const index = ref(0)
const current = computed(() => slides.value[Math.min(index.value, slides.value.length - 1)])
const currentView = computed(() => {
  const s = current.value?.sched
  if (!s) return null
  void status[cacheKey(s)]           // CACHE is a plain Map; status is what changes
  return CACHE.get(cacheKey(s))
})
const currentStatus = computed(() => current.value?.sched ? status[cacheKey(current.value.sched)] : 'ready')
const loaded = computed(() => views.value.filter((s: any) => status[cacheKey(s)] === 'ready').length)

function go(i: number) { index.value = Math.max(0, Math.min(slides.value.length - 1, i)) }

// ---- loading: the page on screen first, then the rest, one at a time
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
watch(() => views.value.map(cacheKey).join(','), () => {
  for (const s of views.value) if (CACHE.has(cacheKey(s))) status[cacheKey(s)] = 'ready'
  prefetch()
}, { immediate: true })

// ---- scaling the deck canvas to the stage
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
  // Centred in the space it has; the canvas is out of the flow, so it never
  // widens the page it sits on.
  offset.value = { x: Math.max(0, (w - 1100 * scale.value) / 2),
                   y: full ? Math.max(0, (h - 825 * scale.value) / 2) : 0 }
}
function fullscreen() {
  if (document.fullscreenElement) document.exitFullscreen()
  else stage.value?.requestFullscreen?.()
}
function onKey(e: KeyboardEvent) {
  const t = e.target as HTMLElement
  if (t && /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName)) return
  if (['ArrowRight', 'PageDown', ' '].includes(e.key)) { go(index.value + 1); e.preventDefault() }
  else if (['ArrowLeft', 'PageUp'].includes(e.key)) { go(index.value - 1); e.preventDefault() }
  else if (e.key === 'Home') go(0)
  else if (e.key === 'End') go(slides.value.length - 1)
}
onMounted(async () => {
  window.addEventListener('keydown', onKey)
  window.addEventListener('resize', fit)
  document.addEventListener('fullscreenchange', fit)
  await nextTick()
  if (stage.value) { ro = new ResizeObserver(fit); ro.observe(stage.value) }
  fit()
})
onBeforeUnmount(() => {
  window.removeEventListener('keydown', onKey)
  window.removeEventListener('resize', fit)
  document.removeEventListener('fullscreenchange', fit)
  ro?.disconnect()
})

const showNotes = ref(false)
const notes = computed<string[]>(() => currentView.value?.notes || [])
</script>

<template>
  <div class="deck">
    <aside class="rail">
      <button v-for="(s, i) in slides" :key="s.id" class="thumb" :class="{ on: i === index }" @click="go(i)">
        <span class="pg">{{ s.label }}</span>
        <span class="tt">{{ s.title }}</span>
        <span v-if="s.sched" class="dot" :class="status[cacheKey(s.sched)] || 'loading'"
              :title="status[cacheKey(s.sched)] === 'ready' ? 'Loaded' : status[cacheKey(s.sched)] === 'error'
                ? 'Could not load' : 'Loading'"></span>
      </button>
      <div v-if="!slides.length" class="muted small">Nothing to show yet: include a schedule that has a view,
        or write a narrative.</div>
      <div v-if="pending.length" class="muted small later"
           :title="pending.map((s: any) => 'p. ' + s.pages + '  ' + s.title).join('\n')">
        {{ pending.length }} included schedule{{ pending.length === 1 ? '' : 's' }} not built yet</div>
    </aside>

    <section class="main">
      <div class="bar">
        <button class="btn" :disabled="index === 0" @click="go(index - 1)">‹ Prev</button>
        <span class="count">{{ slides.length ? index + 1 : 0 }} / {{ slides.length }}</span>
        <button class="btn" :disabled="index >= slides.length - 1" @click="go(index + 1)">Next ›</button>
        <span class="muted small">
          <template v-if="loaded < views.length">Loading figures… {{ loaded }} of {{ views.length }}</template>
          <template v-else-if="views.length">All figures loaded; pages turn without recalculating.</template>
        </span>
        <span class="grow"></span>
        <span v-if="current?.sched" class="muted small">as of {{ current.sched.as_of }}</span>
        <button class="btn" title="Fetch every schedule again (after a data refresh)" @click="refresh">Refresh figures</button>
        <button class="btn" @click="fullscreen">Full screen</button>
      </div>

      <div ref="stage" class="stage" :style="isFull ? {} : { height: 825 * scale + 2 + 'px' }">
        <div class="canvas" :style="{ transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})` }">
          <template v-if="current">
            <template v-if="current.kind === 'narrative'">
              <NarrativeSlide :title="current.narrative.title" :body="current.narrative.body" :page="current.label" />
            </template>
            <template v-else-if="currentStatus === 'ready' && currentView">
              <AssetClassSlide v-if="current.kind === 'exposure_asset_class'" :view="currentView" :page="current.label" />
              <CapitalizationSlide v-else-if="current.kind === 'capitalization'" :view="currentView" :page="current.label" />
              <PerformanceSlide v-else-if="current.kind === 'performance'" :view="currentView" :page="current.label" />
              <PortfolioMetricsSlide v-else-if="current.kind === 'debt'" :view="currentView" :page="current.label" />
              <InvestmentSummarySlide v-else-if="current.kind === 'investment_summaries'" :view="currentView"
                                      :slide="current.spec" :page="current.label" />
            </template>
            <div v-else class="placeholder">
              <div class="ph-title">p. {{ current.sched.pages }} · {{ current.sched.title }}</div>
              <div v-if="currentStatus === 'error'" class="ph-err">Could not load: {{ errors[cacheKey(current.sched)] }}
                <button class="btn" @click="load(current.sched)">Try again</button></div>
              <div v-else>Building the figures from the engines…</div>
            </div>
          </template>
        </div>
      </div>

      <div v-if="notes.length" class="review">
        <button class="link" @click="showNotes = !showNotes">
          {{ showNotes ? '▾' : '▸' }} Reviewer notes for this schedule ({{ notes.length }}) — not shown on the page</button>
        <ul v-if="showNotes"><li v-for="(n, i) in notes" :key="i">{{ n }}</li></ul>
      </div>
    </section>
  </div>
</template>

<style scoped>
.deck { display: flex; gap: 14px; align-items: flex-start; }
.rail { width: 210px; flex: none; display: flex; flex-direction: column; gap: 4px; max-height: calc(100vh - 220px);
  overflow-y: auto; }
.thumb { display: grid; grid-template-columns: 30px 1fr 10px; align-items: center; gap: 6px; text-align: left;
  padding: 7px 8px; border: 1px solid var(--color-border, #e5e7eb); border-radius: 6px; cursor: pointer;
  background: var(--color-surface, #fff); color: inherit; font-size: 12.5px; }
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
.btn:disabled { opacity: .45; cursor: default; }
.stage { position: relative; width: 100%; overflow: hidden; background: #e5e7eb; border: 1px solid #d1d5db; }
.stage:fullscreen { background: #111; border: none; }
.canvas { position: absolute; left: 0; top: 0; width: 1100px; height: 825px; transform-origin: top left;
  box-shadow: 0 1px 6px rgba(0, 0, 0, .25); }
.placeholder { width: 1100px; height: 825px; background: #fff; color: #374151; display: flex; flex-direction: column;
  align-items: center; justify-content: center; gap: 12px; font-size: 20px; font-family: Calibri, 'Segoe UI', sans-serif; }
.ph-title { font-size: 28px; }
.ph-err { color: #b42318; font-size: 16px; }
.review { margin-top: 8px; font-size: 13px; }
.review ul { margin: 6px 0 0; padding-left: 20px; color: var(--color-text-muted, #4b5563); }
.link { background: none; border: none; padding: 0; color: inherit; cursor: pointer; font-size: 13px; }
.muted { color: var(--color-text-muted, #6b7280); }
.small { font-size: 12px; }
/* A narrow window: the page list runs across the top so the page keeps the width. */
@media (max-width: 1200px) {
  .deck { flex-direction: column; }
  .rail { width: 100%; max-height: none; flex-direction: row; overflow-x: auto; overflow-y: hidden;
    padding-bottom: 4px; }
  .thumb { flex: none; width: 190px; }
  .later { margin: 0 0 0 8px; align-self: center; white-space: nowrap; }
}
</style>
