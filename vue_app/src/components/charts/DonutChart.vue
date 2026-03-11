<script setup lang="ts">
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { PieChart } from 'echarts/charts'
import { TooltipComponent, LegendComponent } from 'echarts/components'

use([CanvasRenderer, PieChart, TooltipComponent, LegendComponent])

const props = defineProps<{
  data: Array<{ name: string; value: number; color?: string }>
}>()

const option = computed(() => ({
  tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
  legend: { orient: 'vertical', right: 10, top: 'center' },
  series: [
    {
      type: 'pie',
      radius: ['45%', '70%'],
      center: ['40%', '50%'],
      data: props.data.map((d) => ({
        name: d.name,
        value: d.value,
        itemStyle: d.color ? { color: d.color } : undefined,
      })),
      label: { show: false },
      emphasis: { label: { show: true, fontSize: 14, fontWeight: 'bold' } },
    },
  ],
}))
</script>

<template>
  <v-chart :option="option" style="height: 350px; width: 100%" autoresize />
</template>
