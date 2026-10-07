<script setup lang="ts">
/** Page 27, laid out as the January 2026 deck lays it out. Figures from the server. */
import BoardSlide from './BoardSlide.vue'
import { money, pct } from './slideFormat'

defineProps<{ view: any; page: string | number }>()
</script>

<template>
  <BoardSlide :title="view.slide_title" :page="page">
    <table class="perf">
      <colgroup>
        <col style="width: 21%" /><col style="width: 12%" /><col style="width: 12%" /><col style="width: 9%" />
        <col style="width: 14%" /><col style="width: 16%" /><col style="width: 16%" />
      </colgroup>
      <thead>
        <tr>
          <th></th><th></th><th>Pref. Equity</th><th><i>Proj.</i> IRR</th>
          <th class="bl bt">Final Realized<br />Gross IRR</th><th class="bt">Proceeds-to-Date</th>
          <th class="br bt">CoC Act.<br />Rtns. Since Close</th>
        </tr>
      </thead>
      <tbody>
        <template v-for="r in view.rows" :key="r.key">
          <tr class="band">
            <td class="l b">{{ r.label }}</td><td></td>
            <td :title="r.pref_basis">{{ money(r.pref) }}{{ r.key === 'current' ? '*' : '' }}</td>
            <td :class="{ i: r.key === 'current' }">{{ r.key === 'exited' ? 'N/A' : pct(r.proj_irr) }}</td>
            <td class="bl">{{ r.key === 'current' ? 'N/A' : pct(r.realized_irr) }}</td>
            <td>{{ money(r.proceeds) }}</td><td class="br">{{ pct(r.coc) }}</td>
          </tr>
          <tr class="gap"><td></td><td></td><td></td><td></td><td class="bl"></td><td></td><td class="br"></td></tr>
        </template>
        <tr class="tot">
          <td colspan="4" class="l note">
            <div v-for="(f, i) in view.footnotes" :key="i">{{ f }}</div>
          </td>
          <td class="bl bb"></td><td class="bb b">{{ money(view.total.proceeds) }}</td>
          <td class="br bb b">{{ pct(view.total.coc) }}</td>
        </tr>
      </tbody>
    </table>
  </BoardSlide>
</template>

<style scoped>
.perf { border-collapse: collapse; width: 100%; font-size: 16px; color: #000; margin-top: 70px; }
.perf th { font-weight: 700; text-align: center; vertical-align: bottom; padding: 4px 6px; line-height: 1.15;
  border-bottom: 1.5px solid #000; }
.perf td { text-align: center; padding: 5px 6px; height: 24px; }
.perf .l { text-align: left; }
.perf .b { font-weight: 700; }
.perf .i { font-style: italic; }
.perf tr.band td { background: #e2efda; }
.perf tr.gap td { height: 22px; padding: 0; }
.perf tr.tot td { border-top: 1.5px solid #000; padding-top: 8px; padding-bottom: 8px; }
.perf td.note { font-size: 11px; font-style: italic; vertical-align: top; line-height: 1.35; }
.perf .bl { border-left: 2px solid #000; }
.perf .br { border-right: 2px solid #000; }
.perf .bt { border-top: 2px solid #000; }
.perf .bb { border-bottom: 2px solid #000; }
</style>
