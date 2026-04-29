<script setup lang="ts">
defineProps<{
  columns: Array<{ key: string; label: string; format?: string; align?: string }>
  rows: Array<Record<string, any>>
  highlightTotal?: boolean
  rowClass?: (row: Record<string, any>) => string | Record<string, boolean> | undefined
}>()

function formatCell(value: any, format?: string): string {
  if (value == null) return '—'
  switch (format) {
    case 'currency':
      return new Intl.NumberFormat('en-US', {
        style: 'currency', currency: 'USD', minimumFractionDigits: 0, maximumFractionDigits: 0,
      }).format(value)
    case 'currency2':
      return new Intl.NumberFormat('en-US', {
        style: 'currency', currency: 'USD', minimumFractionDigits: 2,
      }).format(value)
    case 'percent':
      return typeof value === 'number' ? (value * 100).toFixed(2) + '%' : String(value)
    case 'multiple':
      return typeof value === 'number' ? value.toFixed(2) + 'x' : String(value)
    default:
      return String(value)
  }
}
</script>

<template>
  <div class="data-table-wrapper">
    <table class="data-table">
      <thead>
        <tr>
          <th v-for="col in columns" :key="col.key" :style="{ textAlign: col.align || 'left' }">
            {{ col.label }}
          </th>
        </tr>
      </thead>
      <tbody>
        <tr
          v-for="(row, idx) in rows"
          :key="idx"
          :class="[{ 'total-row': highlightTotal && row._is_deal_total }, rowClass?.(row)]"
        >
          <td v-for="col in columns" :key="col.key" :style="{ textAlign: col.align || 'left' }">
            {{ formatCell(row[col.key], col.format) }}
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<style scoped>
.data-table-wrapper {
  overflow-x: auto;
}

.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

.data-table th {
  padding: 8px 12px;
  background: var(--color-accent);
  color: white;
  font-weight: 600;
  white-space: nowrap;
}

.data-table td {
  padding: 6px 12px;
  border-bottom: 1px solid var(--color-border);
}

.data-table tbody tr:hover {
  background: #f0f4f8;
}

.total-row {
  font-weight: 700;
}

.total-row td {
  border-top: 2px solid var(--color-text);
}

.row-highlight {
  font-weight: 700;
  background: #edf2f7;
}

.row-highlight td {
  background: #edf2f7;
}
</style>
