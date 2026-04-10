<script setup lang="ts">
defineProps<{
  label: string
  value: string | number
  format?: 'currency' | 'currency2' | 'currency-millions' | 'percent' | 'number' | 'integer'
}>()

function formatValue(value: string | number, format?: string): string {
  if (typeof value === 'string') return value
  if (value == null || isNaN(value)) return '—'

  switch (format) {
    case 'currency':
      return new Intl.NumberFormat('en-US', {
        style: 'currency', currency: 'USD', minimumFractionDigits: 0, maximumFractionDigits: 0,
      }).format(value)
    case 'currency-millions':
      return new Intl.NumberFormat('en-US', {
        style: 'currency', currency: 'USD', minimumFractionDigits: 1, maximumFractionDigits: 1,
      }).format(value / 1_000_000) + 'M'
    case 'currency2':
      return new Intl.NumberFormat('en-US', {
        style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: 2,
      }).format(value)
    case 'percent':
      return (value * 100).toFixed(1) + '%'
    case 'number':
      return new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 }).format(value)
    case 'integer':
      return new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(value)
    default:
      return String(value)
  }
}
</script>

<template>
  <div class="kpi-card">
    <div class="kpi-label">{{ label }}</div>
    <div class="kpi-value">{{ formatValue(value, format) }}</div>
  </div>
</template>

<style scoped>
.kpi-card {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: 8px;
  padding: 16px 20px;
}

.kpi-label {
  font-size: 12px;
  color: var(--color-text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 4px;
}

.kpi-value {
  font-size: 24px;
  font-weight: 700;
  color: var(--color-text);
}
</style>
