<script setup lang="ts">
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { BarChart, LineChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent } from 'echarts/components'

use([CanvasRenderer, BarChart, LineChart, GridComponent, TooltipComponent, LegendComponent])

const props = defineProps<{
  categories: string[]
  barData: { name: string; data: number[]; color?: string }
  lineData: Array<{ name: string; data: number[]; color?: string }>
  barAxisLabel?: string
  lineAxisLabel?: string
}>()

const option = computed(() => ({
  tooltip: { trigger: 'axis' },
  legend: { bottom: 0 },
  grid: { left: 60, right: 60, top: 20, bottom: 40 },
  xAxis: { type: 'category', data: props.categories },
  yAxis: [
    {
      type: 'value',
      name: props.barAxisLabel || '',
      position: 'left',
      axisLabel: { formatter: '{value}%' },
    },
    {
      type: 'value',
      name: props.lineAxisLabel || '',
      position: 'right',
      axisLabel: { formatter: '${value}' },
    },
  ],
  series: [
    {
      name: props.barData.name,
      type: 'bar',
      yAxisIndex: 0,
      data: props.barData.data,
      itemStyle: { color: props.barData.color || '#93c5fd' },
      barMaxWidth: 40,
    },
    ...props.lineData.map((line) => ({
      name: line.name,
      type: 'line',
      yAxisIndex: 1,
      data: line.data,
      lineStyle: { color: line.color },
      itemStyle: { color: line.color },
    })),
  ],
}))
</script>

<template>
  <v-chart :option="option" style="height: 350px; width: 100%" autoresize />
</template>
