<script setup lang="ts">
/**
 * Page 9, laid out as the January 2026 deck lays it out: one stacked bar a year --
 * everything invested before it (navy) and that year's new preferred equity
 * (green). Figures from the server; the SVG only draws them.
 */
import { computed } from 'vue'
import BoardSlide from './BoardSlide.vue'

const props = defineProps<{ view: any; page: string | number }>()

const W = 820, H = 470, L = 80, B = 40, T = 70
const years = computed<any[]>(() => props.view.years || [])
const top = computed(() => {
  const m = Math.max(1, ...years.value.map((y: any) => y.cumulative / 1e6))
  const step = m > 500 ? 100 : m > 200 ? 50 : 20
  return Math.ceil(m / step) * step
})
const ticks = computed(() => {
  const step = top.value > 500 ? 100 : top.value > 200 ? 50 : 20
  return Array.from({ length: top.value / step + 1 }, (_, i) => i * step)
})
const y = (v: number) => T + (H - T - B) * (1 - v / top.value)
const slot = computed(() => (W - L - 20) / Math.max(1, years.value.length))
const bw = computed(() => slot.value * 0.66)
const xOf = (i: number) => L + slot.value * i + (slot.value - bw.value) / 2
const m = (v: number) => '$' + Math.round(v / 1e6).toLocaleString('en-US')
</script>

<template>
  <BoardSlide :title="view.slide_title" :page="page">
    <div class="frame">
      <svg :viewBox="`0 0 ${W} ${H + 30}`" width="100%">
        <text :x="W / 2" y="26" text-anchor="middle" class="ttl">New &amp; Cumulative</text>
        <text :x="W / 2" y="50" text-anchor="middle" class="ttl">Preferred Equity Investments ($M)</text>
        <g v-for="t in ticks" :key="t">
          <line :x1="L" :x2="W - 20" :y1="y(t)" :y2="y(t)" class="grid" />
          <text :x="L - 8" :y="y(t) + 4" text-anchor="end" class="ax">${{ t.toLocaleString('en-US') }}</text>
        </g>
        <text :x="18" :y="(T + H - B) / 2" class="axl" text-anchor="middle"
              :transform="`rotate(-90 18 ${(T + H - B) / 2})`">Pref Equity Volume</text>
        <g v-for="(yr, i) in years" :key="yr.year">
          <rect v-if="yr.prior > 0" :x="xOf(i)" :y="y(yr.prior / 1e6)" :width="bw"
                :height="y(0) - y(yr.prior / 1e6)" class="cum" />
          <rect :x="xOf(i)" :y="y(yr.cumulative / 1e6)" :width="bw"
                :height="y(yr.prior / 1e6) - y(yr.cumulative / 1e6)" class="new" />
          <text v-if="yr.prior > 0" :x="xOf(i) + bw / 2" :y="(y(yr.prior / 1e6) + y(0)) / 2 + 4"
                text-anchor="middle" class="lbl w">{{ m(yr.prior) }}</text>
          <text :x="xOf(i) + bw / 2" :y="(y(yr.cumulative / 1e6) + y(yr.prior / 1e6)) / 2 + 4"
                text-anchor="middle" class="lbl w">{{ m(yr.new) }}</text>
          <text :x="xOf(i) + bw / 2" :y="H - B + 20" text-anchor="middle" class="ax b">{{ yr.year }}</text>
        </g>
        <g :transform="`translate(${W / 2 - 90}, ${H + 14})`">
          <rect x="0" y="-9" width="10" height="10" class="cum" /><text x="14" y="0" class="ax b">Cumulative</text>
          <rect x="120" y="-9" width="10" height="10" class="new" /><text x="134" y="0" class="ax b">New</text>
        </g>
      </svg>
    </div>
    <div class="fnote"><div v-for="(f, i) in view.footnotes" :key="i">{{ f }}</div></div>
  </BoardSlide>
</template>

<style scoped>
.frame { margin: 0 auto; width: 900px; border: 1px solid #d9d9d9; padding: 6px 10px; }
.ttl { font-family: 'Times New Roman', serif; font-weight: 700; font-size: 19px; }
.grid { stroke: #d9d9d9; stroke-width: 1; }
.ax { font-family: 'Times New Roman', serif; font-size: 14px; }
.ax.b { font-weight: 700; }
.axl { font-family: 'Times New Roman', serif; font-size: 16px; font-weight: 700; }
.cum { fill: #002060; }
.new { fill: #00b050; }
.lbl { font-family: 'Times New Roman', serif; font-size: 13px; font-weight: 700; }
.lbl.w { fill: #fff; }
.fnote { position: absolute; left: 0; bottom: 4px; font-size: 12px; font-style: italic; }
</style>
