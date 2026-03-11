<script setup lang="ts">
import KpiCard from './KpiCard.vue'

defineProps<{
  metrics: Array<{ label: string; value: string | number; format?: string }>
}>()
</script>

<template>
  <div class="metric-cards">
    <KpiCard
      v-for="m in metrics"
      :key="m.label"
      :label="m.label"
      :value="m.value"
      :format="m.format as any"
    />
  </div>
</template>

<style scoped>
.metric-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
}
</style>
