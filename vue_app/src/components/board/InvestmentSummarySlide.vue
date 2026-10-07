<script setup lang="ts">
/**
 * One page of the investment summaries (deck pp. 29-31): the deck's narrower
 * column set, the rows the server put on this page, and the Total on a table's
 * last page. Every cell is written by Investment Metrics' own formatter
 * (`investmentMetricsFormat`), so the slide cannot word a figure differently
 * from the Investment Metrics tab.
 */
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import BoardSlide from './BoardSlide.vue'
import { cellText, totalText } from '@/utils/investmentMetricsFormat'

const props = defineProps<{ view: any; slide: any; page: string | number }>()

const im = computed(() => props.view.investment_metrics)
const table = computed(() => im.value[props.slide.table])
const cols = computed<any[]>(() => props.view.deck.columns[props.slide.table] || [])
const rows = computed<any[]>(() => (table.value.rows || []).slice(props.slide.first, props.slide.last))

// The grouped capitalization heading and its three pair headings, re-anchored
// onto the deck's columns by KEY (the payload's spans index the full column list).
const fullKeys = computed<string[]>(() => (table.value.columns || []).map((c: any) => c.key))
const capKeys = computed(() => {
  const g = table.value.cap_group || { start: 0, span: 0 }
  return new Set(fullKeys.value.slice(g.start, g.start + g.span))
})
const pairStarts = computed(() => {
  const out: Record<string, { label: string; span: number }> = {}
  for (const p of table.value.cap_pairs || []) {
    const keys = fullKeys.value.slice(p.start, p.start + p.span)
    const present = keys.filter((k) => cols.value.some((c) => c.key === k))
    if (present.length) out[present[0]] = { label: p.label, span: present.length }
  }
  return out
})
const capFirst = computed(() => cols.value.findIndex((c) => capKeys.value.has(c.key)))
const capSpan = computed(() => cols.value.filter((c) => capKeys.value.has(c.key)).length)
const inPair = (key: string) => capKeys.value.has(key)
// Thin rules between column groups, as the deck draws them.
const RULE_BEFORE = new Set(['proceeds', 'pref_coupon'])

// FIT, NEVER CLIP. Columns size to their contents and nothing is cut off; when
// the table is wider (long partner names) or taller than the page allows, the
// table's type steps down until it fits -- the deck sets this table small too.
const BASE_PX = 11
const fontPx = ref(BASE_PX)
const wrap = ref<HTMLElement | null>(null)
const tbl = ref<HTMLElement | null>(null)
async function fit() {
  fontPx.value = BASE_PX
  await nextTick()
  const w = wrap.value, t = tbl.value
  if (!w || !t) return
  // Step down until it truly fits: padding and rules do not shrink with the
  // type, so one proportional step can still leave it a little too wide.
  while (fontPx.value > 7 && (t.scrollWidth > w.clientWidth || t.scrollHeight > w.clientHeight)) {
    fontPx.value = Math.round((fontPx.value - 0.2) * 10) / 10
    await nextTick()
  }
}
onMounted(fit)
watch(() => [props.slide, props.view], fit)
</script>

<template>
  <BoardSlide :title="slide.title" :page="page">
    <div class="head">
      <span class="b">{{ table.title }}</span><span class="units">{{ im.units_note }}</span>
    </div>
    <div ref="wrap" class="fit">
    <table ref="tbl" class="ims" :style="{ fontSize: fontPx + 'px' }">
      <thead>
        <tr>
          <template v-for="(c, i) in cols" :key="'a' + c.key">
            <th v-if="i === capFirst" :colspan="capSpan" class="grp"><span>{{ table.cap_group?.heading }}</span></th>
            <th v-else-if="!inPair(c.key)" :class="{ vr: RULE_BEFORE.has(c.key) }">{{ c.row1 }}</th>
          </template>
        </tr>
        <tr>
          <template v-for="c in cols" :key="'b' + c.key">
            <th v-if="pairStarts[c.key]" :colspan="pairStarts[c.key].span" class="grp">
              <span>{{ pairStarts[c.key].label }}</span></th>
            <th v-else-if="!inPair(c.key)" :class="{ vr: RULE_BEFORE.has(c.key) }">{{ c.row2 }}</th>
          </template>
        </tr>
        <tr class="h3">
          <th v-for="c in cols" :key="'c' + c.key" :class="[c.align === 'left' ? 'l' : '', { vr: RULE_BEFORE.has(c.key) }]">
            <span>{{ c.row3 }}</span></th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(r, ri) in rows" :key="r.vcode" :class="{ band: ri % 2 === 0 }">
          <td v-for="c in cols" :key="c.key" :class="[c.align === 'left' ? 'l' : '', { vr: RULE_BEFORE.has(c.key) }]">
            <template v-if="c.key === 'name'">{{ r.name }}<span v-if="r.markers?.length" class="mk">
              {{ r.markers.map((m: number) => `(${m})`).join('') }}</span></template>
            <template v-else>{{ cellText(r, c) }}</template>
          </td>
        </tr>
        <template v-if="slide.is_last">
          <tr class="tot">
            <td v-for="c in cols" :key="c.key" :class="[c.align === 'left' ? 'l' : '', { vr: RULE_BEFORE.has(c.key) }]">
              {{ totalText(table.total, c) }}</td>
          </tr>
          <tr v-if="slide.table === 'sold' && im.grand_total" class="tot grand">
            <td v-for="c in cols" :key="c.key" :class="[c.align === 'left' ? 'l' : '', { vr: RULE_BEFORE.has(c.key) }]">
              {{ totalText(im.grand_total, c) }}</td>
          </tr>
        </template>
      </tbody>
    </table>
    </div>
    <div class="notes">
      <div class="star">{{ slide.footnote }}</div>
      <div v-if="slide.is_last" class="fn">
        <span v-for="f in table.footnotes" :key="f.n">({{ f.n }}) {{ f.text }}&nbsp; </span>
      </div>
    </div>
  </BoardSlide>
</template>

<style scoped>
.head { display: flex; justify-content: space-between; align-items: baseline; border-bottom: 1.5px solid #000;
  font-family: Garamond, 'EB Garamond', 'Times New Roman', serif; font-size: 12px; padding-bottom: 2px; }
.head .b { font-weight: 700; }
.head .units { font-style: italic; }
.fit { position: absolute; left: 0; right: 0; top: 22px; bottom: 66px; overflow: hidden; }
.ims { width: 100%; border-collapse: collapse; color: #000;
  font-family: Garamond, 'EB Garamond', 'Times New Roman', serif; margin-top: 6px; }
.ims th { font-weight: 400; text-align: center; padding: 0 3px; line-height: 1.15; white-space: nowrap; }
.ims th.grp span { display: block; border-bottom: 1px solid #000; margin: 0 4px 1px; }
.ims tr.h3 th span { display: block; border-bottom: 1px solid #000; margin: 0 2px; min-height: 13px; }
.ims td { text-align: center; padding: 1px 3px; height: 1.6em; white-space: nowrap; }
.ims .l { text-align: left; }
.ims tbody tr:first-child td { padding-top: 5px; }
.ims tr.band td { background: #f2f2f2; }
.ims tr.tot td { font-weight: 700; border-top: 1px solid #000; padding-top: 3px; }
.ims tr.grand td { border-top: none; }
.ims .vr { border-left: 1px solid #000; }
.ims .mk { font-size: .8em; }
.notes { position: absolute; left: 0; right: 0; bottom: 4px; }
.star { font-size: 13px; font-style: italic; }
.fn { margin-top: 3px; font-size: 9.5px; color: #333; line-height: 1.3;
  font-family: Garamond, 'EB Garamond', 'Times New Roman', serif; }
</style>
