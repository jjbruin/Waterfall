<script setup lang="ts">
/** Page 26, laid out as the January 2026 deck lays it out. Figures from the server. */
import BoardSlide from './BoardSlide.vue'
import { money, pct } from './slideFormat'

defineProps<{ view: any; page: string | number }>()
</script>

<template>
  <BoardSlide :title="view.slide_title" :page="page">
    <table class="cap">
      <colgroup>
        <col style="width: 22%" /><col style="width: 10%" /><col style="width: 11%" /><col style="width: 13%" />
        <col style="width: 14%" /><col style="width: 15%" /><col style="width: 15%" />
      </colgroup>
      <thead>
        <tr>
          <th class="l units">$ in millions</th><th>Deals</th><th>Properties</th>
          <th>Total Gross<br />Capitalization</th>
          <th class="bx bl bt">PSC Capital</th><th class="bx bt">3rd Party Capital</th>
          <th class="bx br bt">Total Net<br />Pref. Equity<sup>*</sup></th>
        </tr>
      </thead>
      <tbody>
        <template v-for="l in view.lines" :key="l.key">
          <tr class="band">
            <td class="l b">{{ l.label }}</td><td>{{ l.deals }}</td><td>{{ l.properties }}</td>
            <td>{{ money(l.gross_cap, 2) }}</td>
            <td class="bl">{{ money(l.psc) }}</td><td>{{ money(l.third_party) }}</td>
            <td class="br">{{ money(l.total) }}</td>
          </tr>
          <tr class="gap"><td></td><td></td><td></td><td></td><td class="bl"></td><td></td><td class="br"></td></tr>
        </template>
        <tr class="tot">
          <td></td><td>{{ view.total.deals }}</td><td>{{ view.total.properties }}</td>
          <td>{{ money(view.total.gross_cap, 2) }}</td>
          <td class="bl bb">{{ money(view.total.psc) }}</td><td class="bb">{{ money(view.total.third_party) }}</td>
          <td class="br bb">{{ money(view.total.total) }}</td>
        </tr>
      </tbody>
    </table>

    <table class="src">
      <thead>
        <tr><th colspan="3" class="cap-head">3<sup>rd</sup> Party Capital Sources</th></tr>
        <tr class="sub"><th class="l">Investor</th><th colspan="2" class="vl">Current AUM**</th></tr>
      </thead>
      <tbody>
        <tr v-for="x in view.third_party_sources" :key="x.group">
          <td class="l">{{ x.label }}</td><td class="vl">{{ money(x.amount) }}</td><td>{{ pct(x.share) }}</td>
        </tr>
        <tr class="tot"><td class="l">TOTAL</td><td class="vl">{{ money(view.third_party_total) }}</td>
          <td><i>100%</i></td></tr>
      </tbody>
    </table>

    <div class="notes">
      <div v-for="(f, i) in view.footnotes" :key="i">{{ f }}</div>
    </div>
  </BoardSlide>
</template>

<style scoped>
table { border-collapse: collapse; color: #000; }
.cap { width: 100%; font-size: 16px; margin-top: 30px; }
.cap th { font-weight: 700; text-align: center; vertical-align: bottom; padding: 4px 6px; line-height: 1.15; }
.cap th.units { font-weight: 400; font-style: italic; vertical-align: top; }
.cap td { text-align: center; padding: 5px 6px; height: 24px; }
.cap .l { text-align: left; }
.cap .b { font-weight: 700; }
.cap thead th { border-bottom: 1.5px solid #000; }
.cap tr.band td { background: #e2efda; }
.cap tr.gap td { height: 12px; padding: 0; }
.cap tr.tot td { border-top: 1.5px solid #000; font-weight: 700; padding-top: 8px; padding-bottom: 8px; }
.cap .bl { border-left: 2px solid #000; }
.cap .br { border-right: 2px solid #000; }
.cap .bt { border-top: 2px solid #000; }
.cap .bb { border-bottom: 2px solid #000; }

.src { margin: 70px auto 0; width: 400px; font-size: 15px; border: 1.5px solid #000; }
.src th, .src td { padding: 6px 10px; text-align: right; }
.src .l { text-align: left; }
.src .cap-head { text-align: center; background: #e7e6e6; font-weight: 700; font-size: 16px;
  border-bottom: 1.5px solid #000; padding: 10px; }
.src tr.sub th { font-weight: 700; border-bottom: 1.5px solid #000; }
.src tr.sub th.vl { text-align: center; }
.src .vl { border-left: 1.5px solid #000; }
.src tr.tot td { border-top: 1.5px solid #000; font-weight: 700; font-size: 16px; }

.notes { position: absolute; left: 30px; bottom: 26px; font-size: 12px; font-style: italic; line-height: 1.4; }
</style>
