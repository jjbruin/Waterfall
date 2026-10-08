<script setup lang="ts">
/**
 * Page 23, laid out as the January 2026 deck lays it out: the two boxed columns
 * (total and PSC net preferred equity, unfunded included) and a pie for each.
 * Figures and shares come from the server; the pies only draw the server's shares.
 */
import { computed } from 'vue'
import BoardSlide from './BoardSlide.vue'
import { money, pct } from './slideFormat'

const props = defineProps<{ view: any; page: string | number }>()

// The deck's palette, by class -- one colour per class in the table, both pies and the legend.
const PALETTE: Record<string, string> = {
  'Multifamily': '#5b9bd5', 'Non-Grocery Retail': '#ed7d31', 'Grocery-Anchored Retail': '#a5a5a5',
  'Self Storage': '#ffc000', 'Other': '#4472c4',
}
const SPARE = ['#70ad47', '#264478', '#9e480e', '#636363']
const colour = (label: string, i: number) => PALETTE[label] || SPARE[i % SPARE.length]

const rows = computed<any[]>(() => props.view.rows || [])

interface Slice { label: string; path: string; colour: string; share: number; lx: number; ly: number; outside: boolean }
function slices(field: 'total_share' | 'psc_share'): Slice[] {
  const R = 112, CX = 130, CY = 130
  let a = -Math.PI / 2
  const out: Slice[] = []
  rows.value.forEach((r: any, i: number) => {
    const share = r[field]
    if (!share || share <= 0) return
    const b = a + share * 2 * Math.PI
    const large = b - a > Math.PI ? 1 : 0
    const [x1, y1, x2, y2] = [CX + R * Math.cos(a), CY + R * Math.sin(a), CX + R * Math.cos(b), CY + R * Math.sin(b)]
    const path = share > 0.9999
      ? `M ${CX - R} ${CY} A ${R} ${R} 0 1 1 ${CX + R} ${CY} A ${R} ${R} 0 1 1 ${CX - R} ${CY} Z`
      : `M ${CX} ${CY} L ${x1} ${y1} A ${R} ${R} 0 ${large} 1 ${x2} ${y2} Z`
    const mid = (a + b) / 2
    const outside = share < 0.04
    const lr = outside ? R + 14 : R * 0.68
    out.push({ label: r.label, path, colour: colour(r.label, i), share,
      lx: CX + lr * Math.cos(mid), ly: CY + lr * Math.sin(mid), outside })
    a = b
  })
  return out
}
const pieTotal = computed(() => slices('total_share'))
const piePsc = computed(() => slices('psc_share'))
</script>

<template>
  <BoardSlide :title="view.slide_title" :page="page">
    <table class="ac">
      <colgroup><col style="width: 40%" /><col style="width: 14%" /><col style="width: 9%" />
        <col style="width: 4%" /><col style="width: 14%" /><col style="width: 9%" /></colgroup>
      <thead>
        <tr>
          <th class="l units band">$ millions</th>
          <th colspan="2" class="bx band">Total Net<br />Preferred Equity<sup>*</sup></th>
          <th class="gapc"></th>
          <th colspan="2" class="bx band">PSC Net<br />Preferred Equity<sup>*</sup></th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(r, i) in rows" :key="r.label" :class="{ alt: i % 2 === 1 }">
          <td class="l name">{{ r.label }}</td>
          <td class="bl">{{ money(r.total) }}</td><td class="br">{{ pct(r.total_share, 0) }}</td>
          <td class="gapc"></td>
          <td class="bl">{{ money(r.psc) }}</td><td class="br">{{ pct(r.psc_share, 0) }}</td>
        </tr>
        <tr class="tot">
          <td class="l">Total</td>
          <td colspan="2" class="bl br bb c">{{ money(view.total.total) }}</td>
          <td class="gapc"></td>
          <td colspan="2" class="bl br bb c">{{ money(view.total.psc) }}</td>
        </tr>
      </tbody>
    </table>
    <div class="fnote"><div v-for="(f, i) in view.footnotes" :key="i">{{ f }}</div></div>

    <div class="pies">
      <figure v-for="pie in [{ t: 'Preferred Net Equity', s: pieTotal }, { t: 'PSC Net', s: piePsc }]" :key="pie.t">
        <figcaption>{{ pie.t }}</figcaption>
        <svg viewBox="-20 -20 300 300" width="250" height="250">
          <path v-for="sl in pie.s" :key="sl.label" :d="sl.path" :fill="sl.colour" stroke="#fff" stroke-width="2" />
          <text v-for="sl in pie.s" :key="'t' + sl.label" :x="sl.lx" :y="sl.ly" text-anchor="middle"
                dominant-baseline="middle" :class="sl.outside ? 'out' : 'in'">{{ Math.round(sl.share * 100) }}%</text>
        </svg>
      </figure>
    </div>
    <div class="legend">
      <span v-for="(r, i) in rows" :key="r.label"><i :style="{ background: colour(r.label, i) }"></i>{{ r.label }}</span>
    </div>
  </BoardSlide>
</template>

<style scoped>
.ac { border-collapse: collapse; width: 620px; margin: 22px auto 0; font-size: 16px; color: #000; }
.ac th { font-weight: 700; text-align: center; vertical-align: bottom; padding: 6px 8px; line-height: 1.2; }
.ac th.units { font-weight: 400; font-style: italic; font-size: 13px; text-align: left; border: 1px solid #000;
  border-right: none; }
.ac th.band { background: #e2efda; }
.ac th.bx { border: 2px solid #000; }
.ac td { padding: 4px 8px; text-align: right; height: 26px; }
.ac .l { text-align: left; }
.ac .c { text-align: center; }
.ac td.name { padding-left: 14px; font-size: 17px; }
.ac tr.alt td:not(.gapc) { background: #f2f2f2; }
.ac .gapc { width: 22px; background: #fff; border: none; }
.ac .bl { border-left: 2px solid #000; }
.ac .br { border-right: 2px solid #000; }
.ac .bb { border-bottom: 2px solid #000; }
.ac tr.tot td { border-top: 2px solid #000; font-weight: 700; }
.ac tr.tot td.gapc { border-top: none; }
.ac tbody tr:first-child td { padding-top: 10px; }
.fnote { width: 620px; margin: 4px auto 0; font-size: 11px; font-style: italic; }
.pies { display: flex; justify-content: center; gap: 90px; margin-top: 18px; }
figure { margin: 0; text-align: center; }
figcaption { font-size: 18px; text-decoration: underline; margin-bottom: 2px; }
text.in { font-size: 15px; font-weight: 700; fill: #fff; }
text.out { font-size: 14px; font-weight: 700; fill: #404040; }
.legend { display: flex; justify-content: center; gap: 22px; font-size: 14px; color: #404040; margin-top: 4px; }
.legend i { display: inline-block; width: 10px; height: 10px; margin-right: 6px; }
</style>
