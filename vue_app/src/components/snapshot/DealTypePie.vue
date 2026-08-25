<script setup lang="ts">
/**
 * Deal Type Allocation — PDF page 1's pie.
 *
 * A pie is usually the wrong form; this is the narrow case where it is not.
 * Three slices, one part-to-whole question ("how is the portfolio split across
 * value-add, income and new construction?"), and a reference document that
 * already renders it this way. Three is inside the limit where angle comparison
 * still works — a fourth category would make this a bar chart.
 *
 * FUNDED dollars, not committed: the published slices sum to the funded total
 * and the narrative above quotes funded figures ("value-add deals represent 38%
 * ($153.1M)"). Committed would answer a different question with the same picture.
 *
 * Option object from ./chartOptions.ts so it is assertable headlessly.
 */
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { PieChart } from 'echarts/charts'
import { TooltipComponent, LegendComponent } from 'echarts/components'
import { dealTypeOption, type Alloc } from './chartOptions'

use([CanvasRenderer, PieChart, TooltipComponent, LegendComponent])

const props = defineProps<{ alloc: Alloc | null }>()

const hasData = computed(
  () => (props.alloc?.buckets || []).length > 0 && !!props.alloc?.total_funded)
const option = computed(() => dealTypeOption(props.alloc))
</script>

<template>
  <v-chart
    v-if="hasData"
    :option="option"
    style="height: 380px; width: 100%"
    autoresize
  />
  <p v-else class="empty">No deal-type data.</p>
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
