<script setup lang="ts">
/**
 * One Investment Metrics table — Current or Sold.
 *
 * ONE COMPONENT FOR BOTH THE SCREEN AND THE PRINTED DOCUMENT. The two differ
 * only in CSS: the print sheet sets Garamond at 3.6pt on point-measured column
 * widths, the screen sets a legible size and lets the columns breathe. Giving
 * the printed page its own markup would mean maintaining the reference's
 * wording, its column order and its footnote markers twice, and the printed
 * copy is the one nobody looks at until it is wrong.
 *
 * Every piece of geometry — column widths in points, alignment, which columns
 * carry a vertical rule, the grouped heading spans — comes from the server
 * payload. It is measured from the reference document and lives in
 * `investment_metrics_config.py`; duplicating it here would put the two out of
 * step the first time a column moves.
 */
import { computed } from 'vue'
import { cellText, totalText } from '@/utils/investmentMetricsFormat'

const props = defineProps<{
  table: any
  /** Marker numbers printed after the Total row's own cells, e.g. {realized_irr: [3]} */
  totalMarkers?: Record<string, number[]>
  /** Appended under the total row; only the Sold table has one. */
  grandTotal?: any
  grandTotalLabel?: string
}>()

const cols = computed<any[]>(() => props.table?.columns || [])
const rules = computed<number[]>(() => props.table?.vertical_rules || [])

function ruleClass(i: number) {
  return rules.value.includes(i) ? 'vrule' : ''
}

/** Row 1 and row 2 cells that the capitalization spans cover are drawn by the
 *  span itself, so the per-column cell must not also emit one. */
const capGroup = computed(() => props.table?.cap_group || { start: 6, span: 6 })
const capPairs = computed<any[]>(() => props.table?.cap_pairs || [])
const capStart = computed(() => capGroup.value.start)
const capEnd = computed(() => capGroup.value.start + capGroup.value.span)

function inCap(i: number) {
  return i >= capStart.value && i < capEnd.value
}
</script>

<template>
  <table class="im-grid">
    <colgroup>
      <col v-for="(c, i) in cols" :key="i" :style="{ width: c.width + 'pt' }" />
    </colgroup>
    <thead>
      <!-- The reference leaves a blank row between the rule under the title
           and the first heading row. Measured: the title rule sits at y=63.00
           and the first heading's text at y=68, one full 4.68pt row below. -->
      <tr class="gap headlead"><td :colspan="cols.length"></td></tr>
      <!-- Row 1: single words, plus the grouped capitalization heading -->
      <tr class="h1">
        <template v-for="(c, i) in cols" :key="'a' + i">
          <th
            v-if="i === capStart"
            :colspan="capGroup.span"
            class="grouped"
          ><span class="rule">{{ table.cap_group?.heading }}</span></th>
          <th v-else-if="!inCap(i)" :class="[c.align, ruleClass(i)]">{{ c.row1 }}</th>
        </template>
      </tr>
      <!-- Row 2: the three capitalization pair headings sit here -->
      <tr class="h2">
        <template v-for="(c, i) in cols" :key="'b' + i">
          <th
            v-if="capPairs.some((p) => p.start === i)"
            :colspan="capPairs.find((p) => p.start === i).span"
            :class="['grouped', ruleClass(i)]"
          ><span class="rule">{{ capPairs.find((p) => p.start === i).label }}</span></th>
          <th v-else-if="!inCap(i)" :class="[c.align, ruleClass(i)]">{{ c.row2 }}</th>
        </template>
      </tr>
      <!--
        Row 3: the per-column labels, each with its own underline.

        THE RULE IS ON AN INNER SPAN, not on the cell. In the reference each
        column's rule stops 1.08pt short of the next one, and that gap is what
        makes twenty rules read as twenty columns; drawn as a cell border they
        butt together into a single line across the whole table and the
        grouping disappears. A cell border cannot be inset, so the rule lives
        on a block span that is 1.08pt narrower than its cell.
      -->
      <tr class="h3">
        <th
          v-for="(c, i) in cols"
          :key="'c' + i"
          :class="[c.align, ruleClass(i)]"
        ><span :class="c.row3 ? 'rule' : ''">{{ c.row3 }}</span></th>
      </tr>
    </thead>
    <tbody>
      <!-- The reference leaves one blank row between the column rules and the
           first deal. Without it the table reads as starting inside its own
           heading. -->
      <tr class="gap lead"><td :colspan="cols.length"></td></tr>
      <tr
        v-for="(row, ri) in table.rows"
        :key="row.vcode"
        :class="{ band: ri % 2 === 1 }"
      >
        <td
          v-for="(c, i) in cols"
          :key="i"
          :class="[c.align, ruleClass(i), c.key === '_spacer' ? 'spacer' : '']"
        >
          <template v-if="c.key === 'name'">{{ row.name
            }}<span v-if="row.markers && row.markers.length" class="mark">{{
              ' ' + row.markers.map((m: number) => `(${m})`).join('')
            }}</span></template>
          <template v-else>{{ cellText(row, c) }}</template>
        </td>
      </tr>
      <tr class="gap"><td :colspan="cols.length"></td></tr>
      <tr class="total">
        <td
          v-for="(c, i) in cols"
          :key="i"
          :class="[c.align, ruleClass(i), c.key === '_spacer' ? 'spacer' : '']"
        >{{ totalText(table.total, c)
          }}<span v-if="totalMarkers && totalMarkers[c.key]" class="mark">{{
            ' ' + totalMarkers[c.key].map((m: number) => `(${m})`).join('')
          }}</span></td>
      </tr>
      <tr v-if="grandTotal" class="gap"><td :colspan="cols.length"></td></tr>
      <tr v-if="grandTotal" class="total">
        <td
          v-for="(c, i) in cols"
          :key="i"
          :class="[c.align, ruleClass(i), c.key === '_spacer' ? 'spacer' : '']"
        >{{ c.key === 'name' ? (grandTotalLabel || grandTotal.label)
             : totalText(grandTotal, c) }}</td>
      </tr>
    </tbody>
  </table>
</template>

<style scoped>
/* Structure only. Size, face and rules come from the sheet that embeds this —
   the screen wants something legible and the printed document wants the
   reference's 3.6pt Garamond, and neither belongs in a shared component. */
.im-grid {
  border-collapse: collapse;
  table-layout: fixed;
  width: 100%;
}
.im-grid th,
.im-grid td {
  padding: 0;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: clip;
  vertical-align: bottom;
}
.left { text-align: left; }
.center { text-align: center; }
.grouped { text-align: center; }
.gap td { border: none; background: none; }
/* The rule span fills its cell so the text still centres within the column;
   the sheet that embeds this decides how wide the inset is and whether the
   span carries a border at all. */
.im-grid .rule { display: block; }
</style>
