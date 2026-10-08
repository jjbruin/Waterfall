<script setup lang="ts">
/**
 * Page 28, laid out as the January 2026 deck lays it out: occupancy and DSCR by
 * asset class, fixed-rate debt by years to maturity, floating-rate debt by cap,
 * each floating loan's maximum interest, and the summary line. Every figure and
 * every sentence comes from the server; the bars only draw the server's amounts.
 */
import { computed } from 'vue'
import BoardSlide from './BoardSlide.vue'
import { money, pct } from './slideFormat'

const props = defineProps<{ view: any; page: string | number; notes?: { footnotes: string[]; disclosure?: string | null } }>()

const m = computed(() => props.view.metrics)
const fixed = computed(() => props.view.debt.fixed)
const floating = computed(() => props.view.debt.floating)
const exposure = computed<any[]>(() => props.view.debt.exposure || [])
const half = computed(() => Math.ceil(exposure.value.length / 2))

const bars = computed(() => [{ label: 'Total', amount: fixed.value.total, avg_rate: fixed.value.avg_rate },
  ...fixed.value.buckets])
const maxBar = computed(() => Math.max(1, ...bars.value.map((b: any) => b.amount || 0)))
const dscr = (v: number | null | undefined) => (v === null || v === undefined ? '—' : v.toFixed(1))
const asOf = computed(() => {
  const [y, mo, d] = String(props.view.schedule?.as_of || '').split('-')
  return y ? `${+mo}/${+d}/${y.slice(2)}` : ''
})
</script>

<template>
  <BoardSlide :title="view.slide_title" :page="page" :footnotes="notes?.footnotes" :disclosure="notes?.disclosure">
    <div class="top">
      <table class="met">
        <thead><tr><th class="l units">(as of {{ asOf }})</th><th>Occupancy</th><th>DSCR</th></tr></thead>
        <tbody>
          <tr v-for="(r, i) in m.rows" :key="r.label" :class="{ alt: i % 2 === 1 }">
            <td class="l">{{ r.label }}</td><td>{{ pct(r.occupancy, 0) }}</td><td>{{ dscr(r.dscr) }}</td>
          </tr>
          <tr class="tot"><td class="l">Portfolio</td><td>{{ pct(m.portfolio.occupancy, 0) }}</td>
            <td>{{ dscr(m.portfolio.dscr) }}</td></tr>
        </tbody>
      </table>

      <div class="fixed">
        <div class="boxhead">Fixed Rate Debt Exposure</div>
        <div class="chart">
          <div v-for="b in bars" :key="b.label" class="col">
            <div class="val">{{ money(b.amount) }}</div>
            <div class="bar" :style="{ height: (140 * (b.amount || 0) / maxBar) + 'px' }"></div>
            <div class="lab">{{ b.label }}</div>
            <div class="rate">{{ pct(b.avg_rate, 2) }}</div>
          </div>
          <div class="ratelab">Avg. Rate:</div>
        </div>
      </div>
    </div>

    <div class="float">
      <div class="fhead">FLOATING RATE DEBT EXPOSURE</div>
      <table class="ft">
        <colgroup><col style="width: 25%" /><col style="width: 12%" /><col style="width: 8%" />
          <col style="width: 9%" /><col style="width: 46%" /></colgroup>
        <thead><tr><th class="l">Floating Rate</th><th>Amount</th><th>% of<br />Port. Debt</th><th>Index</th>
          <th>Properties</th></tr></thead>
        <tbody>
          <tr v-for="r in floating.rows" :key="r.key">
            <td class="l">{{ r.label }}</td><td>{{ money(r.amount) }}</td><td>{{ pct(r.share, 0) }}</td>
            <td>{{ r.index }}</td><td class="props">{{ r.deals.join(', ') }}</td>
          </tr>
          <tr class="tot"><td class="l">Floating Rate Total</td><td>{{ money(floating.total) }}</td>
            <td>{{ pct(floating.share, 0) }}</td><td></td><td></td></tr>
        </tbody>
      </table>
      <div class="mx">
        <div class="mxhead">Max Interest Exposure</div>
        <div class="cols">
          <div v-for="part in [exposure.slice(0, half), exposure.slice(half)]" :key="part[0]?.deal || 'e'" class="mxcol">
            <div v-for="e in part" :key="e.deal" class="mxrow"><span class="d">{{ e.deal }}</span>
              <span class="t">{{ e.line }}</span></div>
          </div>
        </div>
      </div>
    </div>
    <div v-if="view.banner" class="banner">{{ view.banner }}</div>
  </BoardSlide>
</template>

<style scoped>
.top { display: flex; gap: 40px; align-items: flex-start; height: 262px; }
.met { border-collapse: collapse; width: 430px; font-size: 15px; color: #000; border: 1.5px solid #000; }
.met th { font-weight: 700; padding: 3px 8px; border-bottom: 1.5px solid #000; background: #e2efda; }
.met th.units { font-weight: 400; font-style: italic; font-size: 12px; }
.met td { padding: 1px 8px; text-align: center; border-left: 1px solid #000; }
.met td.l, .met th.l { text-align: left; border-left: none; }
.met th:not(.l) { border-left: 1px solid #000; }
.met tr.alt td { background: #f2f2f2; }
.met tr.tot td { font-weight: 700; border-top: 1.5px solid #000; }
.fixed { flex: 1; }
.boxhead { border: 2px solid #000; background: #e2efda; text-align: center; font-weight: 700; font-size: 17px;
  padding: 4px; margin: -70px 0 10px 40px; }
.chart { position: relative; display: flex; justify-content: space-around; align-items: flex-end; height: 214px;
  padding-left: 60px; border-bottom: 1px solid #d9d9d9; }
.col { display: flex; flex-direction: column; align-items: center; width: 70px; }
.val { font-weight: 700; font-size: 14px; margin-bottom: 4px; }
.bar { width: 34px; background: #5b9bd5; }
.lab { position: absolute; bottom: -22px; font-weight: 700; font-size: 13px; }
.rate { position: absolute; bottom: -40px; font-size: 12px; }
.col { position: relative; }
.ratelab { position: absolute; left: 0; bottom: -40px; font-size: 12px; font-weight: 700; text-decoration: underline; }
.float { border: 1px solid #000; margin-top: 30px; padding: 0 0 2px; }
.fhead { background: #e2efda; text-align: center; font-weight: 700; font-size: 17px; text-decoration: underline;
  padding: 4px; border-bottom: 1px solid #000; }
.ft { width: 100%; border-collapse: collapse; font-size: 11.5px; color: #000; margin-top: 4px; }
.ft th { font-weight: 700; text-decoration: underline; padding: 1px 4px; text-align: center; vertical-align: bottom; }
.ft td { padding: 1px 4px; text-align: center; }
.ft .l { text-align: left; }
.ft td.props { font-size: 10.5px; line-height: 1.2; }
.ft tr.tot td { border-top: 1px solid #999; }
.mx { padding: 4px 4px 0; font-size: 11px; }
.mxhead { font-weight: 700; text-decoration: underline; margin-bottom: 2px; }
.cols { display: flex; gap: 16px; }
.mxcol { flex: 1; }
.mxrow { display: grid; grid-template-columns: 44% 56%; line-height: 1.35; }
.mxrow .d { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.banner { background: #217346; color: #fff; text-align: center; font-weight: 700; font-size: 16px; padding: 6px;
  margin-top: 4px; }
</style>
