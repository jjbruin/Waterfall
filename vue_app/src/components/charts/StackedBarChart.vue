<script setup lang="ts">
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { BarChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent } from 'echarts/components'

use([CanvasRenderer, BarChart, GridComponent, TooltipComponent, LegendComponent])

const props = defineProps<{
  categories: string[]
  series: Array<{ name: string; data: number[]; color?: string }>
  horizontal?: boolean
  showLabels?: boolean
}>()

const option = computed(() => {
  const axis = { type: 'category' as const, data: props.categories }
  const valueAxis = { type: 'value' as const }

  return {
    tooltip: { trigger: 'axis' },
    legend: { bottom: 0 },
    grid: { left: 60, right: 20, top: 20, bottom: 40 },
    xAxis: props.horizontal ? valueAxis : axis,
    yAxis: props.horizontal ? axis : valueAxis,
    series: props.series.map((s) => ({
      name: s.name,
      type: 'bar',
      stack: 'total',
      data: s.data,
      itemStyle: { color: s.color },
      label: props.showLabels ? { show: true, position: 'inside', fontSize: 10 } : undefined,
    })),
  }
})
</script>

<template>
  <v-chart :option="option" style="height: 350px; width: 100%" autoresize />
</template>
