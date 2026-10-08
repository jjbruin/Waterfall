<script setup lang="ts">
/**
 * Page 24, laid out as the January 2026 deck lays it out: one row per operating
 * partner -- deals, properties, total and PSC net preferred equity (unfunded
 * included) with shares -- and the Grand Total. Figures from the server. The list
 * grows with the portfolio, so the table's type steps down until it fits the page.
 */
import { nextTick, onMounted, ref, watch } from 'vue'
import BoardSlide from './BoardSlide.vue'
import { money, pct } from './slideFormat'

const props = defineProps<{ view: any; page: string | number; notes?: { footnotes: string[]; disclosure?: string | null } }>()

const BASE = 15
const fontPx = ref(BASE)
const wrap = ref<HTMLElement | null>(null)
const tbl = ref<HTMLElement | null>(null)
async function fit() {
  fontPx.value = BASE
  await nextTick()
  while (wrap.value && tbl.value && fontPx.value > 8 && tbl.value.scrollHeight > wrap.value.clientHeight) {
    fontPx.value = Math.round((fontPx.value - 0.25) * 100) / 100
    await nextTick()
  }
}
onMounted(fit)
watch(() => props.view, fit)
</script>

<template>
  <BoardSlide :title="view.slide_title" :page="page" :footnotes="notes?.footnotes" :disclosure="notes?.disclosure">
    <div ref="wrap" class="fit">
      <table ref="tbl" class="pt" :style="{ fontSize: fontPx + 'px' }">
        <colgroup><col style="width: 25%" /><col style="width: 10%" /><col style="width: 12%" />
          <col style="width: 15%" /><col style="width: 11%" /><col style="width: 15%" /><col style="width: 12%" /></colgroup>
        <thead>
          <tr>
            <th class="l"><i class="units">$ in Millions</i><br />Operating Partner</th>
            <th>Deals</th><th>Properties</th>
            <th colspan="2" class="bl">Total Net<br />Preferred Equity<sup>*</sup></th>
            <th colspan="2" class="bl">PSC Net<br />Preferred Equity<sup>*</sup></th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(r, i) in view.rows" :key="r.label" :class="{ alt: i % 2 === 1 }">
            <td class="l">{{ r.label }}</td><td>{{ r.deals }}</td><td>{{ r.properties ?? '—' }}</td>
            <td class="bl r">{{ money(r.total) }}</td><td class="r">{{ pct(r.total_share, 0) }}</td>
            <td class="bl r">{{ money(r.psc) }}</td><td class="r">{{ pct(r.psc_share, 0) }}</td>
          </tr>
          <tr class="tot">
            <td class="l">Grand Total</td><td>{{ view.total.deals }}</td><td>{{ view.total.properties }}</td>
            <td colspan="2" class="bl">{{ money(view.total.total) }}</td>
            <td colspan="2" class="bl">{{ money(view.total.psc) }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </BoardSlide>
</template>

<style scoped>
.fit { position: absolute; left: 90px; right: 90px; top: 0; bottom: 0; overflow: hidden; }
.pt { width: 100%; border-collapse: collapse; color: #000; border: 1.5px solid #000; }
.pt th { background: #d9e1f2; font-weight: 700; padding: 3px 6px; text-align: center; vertical-align: bottom;
  border-bottom: 1.5px solid #000; line-height: 1.15; }
.pt th.l { text-align: left; }
.pt .units { font-weight: 400; font-size: .8em; }
.pt td { padding: 1px 8px; text-align: center; line-height: 1.25; }
.pt td.l { text-align: left; padding-left: 12px; }
.pt td.r { text-align: right; }
.pt .bl { border-left: 1.5px solid #000; }
.pt tr.alt td { background: #f2f2f2; }
.pt tr.tot td { background: #b4c6e7; font-weight: 700; border-top: 1.5px solid #000; padding: 4px 8px; }
</style>
