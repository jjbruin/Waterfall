<script setup lang="ts">
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { BarChart as EBarChart } from 'echarts/charts'
import { GridComponent, TooltipComponent } from 'echarts/components'

use([CanvasRenderer, EBarChart, GridComponent, TooltipComponent])

const props = defineProps<{
  categories: string[]
  data: number[]
  color?: string
  horizontal?: boolean
  formatLabel?: 'percent' | 'currency' | 'number'
}>()

const option = computed(() => {
  const axis = { type: 'category' as const, data: props.categories }
  const valueAxis = { type: 'value' as const }

  return {
    tooltip: { trigger: 'axis' },
    grid: { left: props.horizontal ? 120 : 40, right: 20, top: 10, bottom: 30 },
    xAxis: props.horizontal ? valueAxis : axis,
    yAxis: props.horizontal ? axis : valueAxis,
    series: [
      {
        type: 'bar',
        data: props.data,
        itemStyle: { color: props.color || 'var(--color-accent)' },
        barMaxWidth: 50,
        label: {
          show: true,
          position: props.horizontal ? 'right' : 'top',
          fontSize: 11,
          formatter: (params: any) => {
            const val = params.value
            if (props.formatLabel === 'percent') return (val * 100).toFixed(1) + '%'
            if (props.formatLabel === 'currency') return '$' + (val / 1e6).toFixed(1) + 'M'
            return String(val)
          },
        },
      },
    ],
  }
})
</script>

<template>
  <v-chart :option="option" style="height: 300px; width: 100%" autoresize />
</template>
