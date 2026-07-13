<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useSurveillanceStore } from '../stores/surveillance'
import type { SurveillanceRow } from '../stores/surveillance'

const store = useSurveillanceStore()

// Inline editing
const editingVcode = ref<string | null>(null)
const editFields = ref<Record<string, any>>({})

onMounted(async () => {
  await Promise.all([store.loadTable(), store.loadDashboard()])
})

function startEdit(row: SurveillanceRow) {
  editingVcode.value = row.vcode
  editFields.value = {
    dscr_val: row.dscr_val,
    dscr_min: row.dscr_min,
    ltv_val: row.ltv_val,
    ltv_min: row.ltv_min,
    comments: row.comments || '',
  }
}

async function saveEdit() {
  if (!editingVcode.value) return
  try {
    await store.updateProperty(editingVcode.value, editFields.value)
    editingVcode.value = null
  } catch (e: any) {
    alert('Save failed: ' + e.message)
  }
}

function cancelEdit() {
  editingVcode.value = null
}

function formatCurrency(val: number | null): string {
  if (val == null) return '\u2014'
  if (Math.abs(val) >= 1_000_000) return '$' + (val / 1_000_000).toFixed(1) + 'M'
  if (Math.abs(val) >= 1_000) return '$' + (val / 1_000).toFixed(0) + 'K'
  return '$' + val.toFixed(0)
}

function formatPct(val: number | null, decimals = 1): string {
  if (val == null) return '\u2014'
  return val.toFixed(decimals) + '%'
}

function formatDate(dt: string | null): string {
  if (!dt) return '\u2014'
  const d = new Date(dt)
  if (isNaN(d.getTime())) return dt
  return `${d.getMonth() + 1}/${d.getDate()}/${d.getFullYear()}`
}

function formatRatio(val: number | null): string {
  if (val == null) return '\u2014'
  return val.toFixed(2) + 'x'
}

function occupancyClass(val: number | null): string {
  if (val == null) return ''
  if (val >= 90) return 'occ-good'
  if (val >= 75) return 'occ-warn'
  return 'occ-bad'
}

function maturityClass(dt: string | null): string {
  if (!dt) return ''
  const days = Math.floor((new Date(dt).getTime() - Date.now()) / 86400000)
  if (days <= 180) return 'mat-urgent'
  if (days <= 365) return 'mat-soon'
  return ''
}

function handleExportCsv() {
  const headers = ['Property', 'Asset Type', 'Occupancy', 'NOI Monthly', 'Loan Balance', 'DSCR', 'LTV', 'Maturity', 'Flagged', 'Comments']
  const csvRows = store.filteredRows.map(r => [
    r.name, r.asset_type,
    r.occ_pct != null ? r.occ_pct.toFixed(1) : '',
    r.noi_monthly != null ? r.noi_monthly.toFixed(0) : '',
    r.loan_balance != null ? r.loan_balance.toFixed(0) : '',
    r.dscr_val != null ? r.dscr_val.toFixed(2) : '',
    r.ltv_val != null ? r.ltv_val.toFixed(1) : '',
    r.maturity_date || '',
    r.flagged ? 'Yes' : 'No',
    r.comments || '',
  ])
  const csv = [headers, ...csvRows].map(row => row.map(c => `"${String(c).replace(/"/g, '""')}"`).join(',')).join('\n')
  const blob = new Blob([csv], { type: 'text/csv' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = `surveillance_${new Date().toISOString().split('T')[0]}.csv`
  a.click()
  URL.revokeObjectURL(a.href)
}
</script>

<template>
  <div class="surveillance">
    <div class="page-header">
      <h2>Property Surveillance</h2>
      <button class="btn-export" @click="handleExportCsv">Export CSV</button>
    </div>

    <!-- KPI Strip -->
    <div class="kpi-strip" v-if="store.dashboard">
      <div class="kpi-card">
        <span class="kpi-value">{{ store.dashboard.total }}</span>
        <span class="kpi-label">Active Deals</span>
      </div>
      <div class="kpi-card">
        <span class="kpi-value">{{ formatPct(store.dashboard.avg_occ) }}</span>
        <span class="kpi-label">Avg Occupancy</span>
      </div>
      <div class="kpi-card">
        <span class="kpi-value">{{ formatCurrency(store.dashboard.total_debt) }}</span>
        <span class="kpi-label">Total Debt</span>
      </div>
      <div class="kpi-card">
        <span class="kpi-value">{{ formatCurrency(store.dashboard.total_noi_monthly) }}</span>
        <span class="kpi-label">NOI (Monthly)</span>
      </div>
      <div class="kpi-card" :class="{ 'kpi-alert': (store.dashboard.flagged ?? 0) > 0 }">
        <span class="kpi-value">{{ store.dashboard.flagged }}</span>
        <span class="kpi-label">Flagged</span>
      </div>
      <div class="kpi-card" :class="{ 'kpi-warn': (store.dashboard.maturing_12mo ?? 0) > 0 }">
        <span class="kpi-value">{{ store.dashboard.maturing_12mo }}</span>
        <span class="kpi-label">Maturing 12mo</span>
      </div>
    </div>

    <!-- Filters -->
    <div class="filters">
      <input
        v-model="store.searchQuery"
        type="text"
        class="filter-input search-input"
        placeholder="Search by name or vcode..."
      />
      <select v-model="store.flagFilter" class="filter-select">
        <option value="">All</option>
        <option value="flagged">Flagged Only</option>
        <option value="clear">Clear Only</option>
      </select>
      <select v-model="store.assetTypeFilter" class="filter-select">
        <option value="">All Asset Types</option>
        <option v-for="t in store.assetTypes" :key="t" :value="t">{{ t }}</option>
      </select>
      <span class="filter-count">{{ store.filteredRows.length }} deals</span>
    </div>

    <!-- Error -->
    <div v-if="store.error" class="error-banner">
      {{ store.error }}
      <button @click="store.error = null">Dismiss</button>
    </div>

    <!-- Loading -->
    <div v-if="store.loading" class="loading-text">Loading surveillance data...</div>

    <!-- Table -->
    <div v-else class="table-wrap">
      <table class="surv-table">
        <thead>
          <tr>
            <th class="sticky-col">Property</th>
            <th>Type</th>
            <th>Occupancy</th>
            <th>NOI (Mo)</th>
            <th>Debt</th>
            <th>DSCR</th>
            <th>LTV</th>
            <th>Maturity</th>
            <th>Flag</th>
            <th>Comments</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="row in store.filteredRows"
            :key="row.vcode"
            :class="{ 'row-flagged': row.flagged }"
          >
            <td class="sticky-col deal-name" :title="row.vcode">{{ row.name || row.vcode }}</td>
            <td>{{ row.asset_type || '\u2014' }}</td>
            <td :class="occupancyClass(row.occ_pct)">
              <div class="occ-cell" v-if="row.occ_pct != null">
                <div class="occ-bar">
                  <div class="occ-fill" :style="{ width: Math.min(row.occ_pct, 100) + '%' }"></div>
                </div>
                <span class="occ-text">{{ row.occ_pct.toFixed(1) }}%</span>
              </div>
              <span v-else>&mdash;</span>
            </td>
            <td class="num">{{ formatCurrency(row.noi_monthly) }}</td>
            <td class="num">{{ formatCurrency(row.loan_balance) }}</td>

            <!-- DSCR — inline edit when active -->
            <template v-if="editingVcode === row.vcode">
              <td>
                <input v-model.number="editFields.dscr_val" class="edit-num" placeholder="Val" />
              </td>
              <td>
                <input v-model.number="editFields.ltv_val" class="edit-num" placeholder="LTV" />
              </td>
              <td :class="maturityClass(row.maturity_date)">{{ formatDate(row.maturity_date) }}</td>
              <td>
                <span v-if="row.flagged" class="flag-indicator">!</span>
              </td>
              <td class="edit-actions">
                <input v-model="editFields.comments" class="edit-input" placeholder="Comment..." />
                <div class="edit-btns">
                  <button class="btn-save" @click="saveEdit">Save</button>
                  <button class="btn-cancel" @click="cancelEdit">Cancel</button>
                </div>
              </td>
            </template>
            <template v-else>
              <td class="num" :class="{ 'breach': row.dscr_val != null && row.dscr_min != null && row.dscr_val < row.dscr_min }">
                {{ formatRatio(row.dscr_val) }}
              </td>
              <td class="num" :class="{ 'breach': row.ltv_val != null && row.ltv_min != null && row.ltv_val > row.ltv_min }">
                {{ formatPct(row.ltv_val) }}
              </td>
              <td :class="maturityClass(row.maturity_date)">{{ formatDate(row.maturity_date) }}</td>
              <td>
                <span v-if="row.flagged" class="flag-indicator">!</span>
              </td>
              <td class="comment-cell" @click="startEdit(row)" :title="row.comments || 'Click to edit'">
                {{ row.comments || '' }}
              </td>
            </template>
          </tr>
          <tr v-if="!store.filteredRows.length && !store.loading">
            <td colspan="10" class="empty-row">No deals match the current filters.</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<style scoped>
.surveillance {
  padding: 0 0 40px 0;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.page-header h2 {
  font-size: 18px;
  margin: 0;
}

.btn-export {
  padding: 6px 14px;
  background: var(--color-accent);
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
}
.btn-export:hover { opacity: 0.9; }

/* KPI Strip */
.kpi-strip {
  display: flex;
  gap: 12px;
  margin-bottom: 16px;
  flex-wrap: wrap;
}

.kpi-card {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: 8px;
  padding: 10px 20px;
  text-align: center;
  min-width: 100px;
  flex: 1;
}

.kpi-value {
  font-size: 20px;
  font-weight: 700;
  display: block;
  color: var(--color-text);
}

.kpi-label {
  font-size: 11px;
  color: var(--color-text-secondary);
  text-transform: uppercase;
}

.kpi-alert { border-color: #ef5350; }
.kpi-alert .kpi-value { color: #c62828; }
.kpi-warn { border-color: #ffb74d; }
.kpi-warn .kpi-value { color: #e65100; }

/* Filters */
.filters {
  display: flex;
  gap: 12px;
  align-items: center;
  margin-bottom: 16px;
  flex-wrap: wrap;
}

.filter-input {
  padding: 6px 10px;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  font-size: 13px;
}

.search-input { width: 240px; }

.filter-select {
  padding: 6px 10px;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  font-size: 13px;
}

.filter-count {
  font-size: 12px;
  color: var(--color-text-secondary);
  margin-left: auto;
}

/* Table */
.table-wrap {
  border: 1px solid var(--color-border);
  border-radius: 8px;
  overflow-x: auto;
}

.surv-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
  min-width: 900px;
}

.surv-table th {
  padding: 10px 12px;
  background: var(--color-accent);
  color: white;
  font-weight: 600;
  text-align: left;
  white-space: nowrap;
  position: sticky;
  top: 0;
  z-index: 2;
}

.surv-table td {
  padding: 7px 12px;
  border-bottom: 1px solid var(--color-border);
  white-space: nowrap;
}

.sticky-col {
  position: sticky;
  left: 0;
  z-index: 1;
  background: inherit;
}

th.sticky-col { z-index: 3; }

.surv-table tbody tr { background: white; }
.surv-table tbody tr:hover { background: #f5f5f5; }

.row-flagged { background: #fff8e1 !important; }
.row-flagged:hover { background: #fff3c4 !important; }

.deal-name {
  font-weight: 500;
  max-width: 220px;
  overflow: hidden;
  text-overflow: ellipsis;
}

.num {
  text-align: right;
  font-variant-numeric: tabular-nums;
}

/* Covenant breach */
.breach {
  color: #c62828;
  font-weight: 600;
}

/* Occupancy bar */
.occ-cell {
  display: flex;
  align-items: center;
  gap: 6px;
}

.occ-bar {
  width: 50px;
  height: 8px;
  background: #e0e0e0;
  border-radius: 4px;
  overflow: hidden;
  flex-shrink: 0;
}

.occ-fill {
  height: 100%;
  border-radius: 4px;
  background: #66bb6a;
  transition: width 0.3s;
}

.occ-warn .occ-fill { background: #ffb74d; }
.occ-bad .occ-fill { background: #ef5350; }

.occ-text {
  font-size: 12px;
  min-width: 42px;
  text-align: right;
}

/* Maturity urgency */
.mat-urgent { color: #c62828; font-weight: 600; }
.mat-soon { color: #e65100; }

/* Flag indicator */
.flag-indicator {
  display: inline-block;
  background: #ffb74d;
  color: #fff;
  font-size: 10px;
  font-weight: 700;
  width: 18px;
  height: 18px;
  line-height: 18px;
  text-align: center;
  border-radius: 50%;
}

/* Comment cell — clickable to edit */
.comment-cell {
  cursor: pointer;
  max-width: 180px;
  overflow: hidden;
  text-overflow: ellipsis;
  color: var(--color-text-secondary);
  font-size: 12px;
}
.comment-cell:hover { background: #f0f0f0; }

/* Inline editing */
.edit-num {
  width: 60px;
  padding: 2px 4px;
  font-size: 12px;
  border: 1px solid var(--color-border);
  border-radius: 3px;
  text-align: right;
}

.edit-input {
  width: 140px;
  padding: 2px 6px;
  font-size: 12px;
  border: 1px solid var(--color-border);
  border-radius: 3px;
}

.edit-actions {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.edit-btns {
  display: flex;
  gap: 4px;
}

.btn-save {
  padding: 2px 8px;
  background: #2e7d32;
  color: white;
  border: none;
  border-radius: 3px;
  cursor: pointer;
  font-size: 11px;
}

.btn-cancel {
  padding: 2px 8px;
  background: #757575;
  color: white;
  border: none;
  border-radius: 3px;
  cursor: pointer;
  font-size: 11px;
}

/* Shared */
.error-banner {
  background: #fef2f2;
  border: 1px solid #fca5a5;
  color: #991b1b;
  padding: 8px 14px;
  border-radius: 6px;
  margin-bottom: 12px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 13px;
}
.error-banner button {
  background: none;
  border: 1px solid #fca5a5;
  color: #991b1b;
  padding: 3px 10px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
}

.empty-row {
  text-align: center;
  color: var(--color-text-secondary);
  font-style: italic;
  padding: 24px !important;
}

.loading-text {
  text-align: center;
  padding: 40px;
  color: var(--color-text-secondary);
  font-style: italic;
}
</style>
