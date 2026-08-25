<script setup lang="ts">
/**
 * Asset Allocation — Currently Funded vs Total Commitment, PDF page 1.
 *
 * Two 100%-stacked bars segmented by asset type, the dollar printed inside each
 * segment and each bar's total under its axis label.
 *
 * WHY PERCENTAGES ON THE AXIS AND DOLLARS IN THE LABELS. The bars are different
 * sizes ($404.2M funded against $445.1M committed) and the page's question is
 * about MIX — "multifamily is 59% of funded assets". Normalising both to 100%
 * answers that directly, and the dollar labels keep the absolute figures so no
 * reader has to multiply a percentage by a total in their head.
 *
 * Selective labels, per the mark spec: a segment under LABEL_MIN_PCT of its bar
 * cannot hold 10px type without colliding with its neighbour (Office is ~3%,
 * about 11px in a 380px plot). Those keep their hover tooltip and their row in
 * "Show data" instead of printing a number that overlaps one.
 *
 * The option object is built by ./chartOptions.ts so it can be asserted against
 * real payloads headlessly — see scripts/snapshot_summary_charts_check.mjs.
 */
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { BarChart } from 'echarts/charts'
import {
  GridComponent, TooltipComponent, LegendComponent,
} from 'echarts/components'
import { assetAllocationOption, type Alloc } from './chartOptions'

use([CanvasRenderer, BarChart, GridComponent, TooltipComponent, LegendComponent])

const props = defineProps<{ alloc: Alloc | null }>()

const hasData = computed(() => (props.alloc?.buckets || []).length > 0)
const option = computed(() => assetAllocationOption(props.alloc))
</script>

<template>
  <v-chart
    v-if="hasData"
    :option="option"
    style="height: 380px; width: 100%"
    autoresize
  />
  <p v-else class="empty">No allocation data.</p>
</template>

<style scoped>
.empty {
  font-size: 12px;
  font-style: italic;
  color: var(--color-text-secondary);
  text-align: center;
  padding: 30px 0;
}
</style>
