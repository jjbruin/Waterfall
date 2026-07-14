<script setup lang="ts">
import { ref, onMounted, nextTick } from 'vue'
import { useSurveillanceStore } from '../stores/surveillance'
import type { SurveillanceRow } from '../stores/surveillance'

const store = useSurveillanceStore()

// Comment editing
const commentVcode = ref<string | null>(null)
const commentText = ref('')
const commentDate = ref(new Date().toISOString().split('T')[0])
const showHistory = ref(false)
const commentTextarea = ref<HTMLTextAreaElement | null>(null)

// Expandable column groups
const expandedGroups = ref<Record<string, boolean>>({
  reporting: false,
  covenants: false,
  taxes: false,
  insurance: false,
  ground_leases: false,
  escrows: false,
  collateral: false,
})

function toggleGroup(group: string) {
  expandedGroups.value[group] = !expandedGroups.value[group]
}

onMounted(async () => {
  await Promise.all([store.loadTable(), store.loadDashboard()])
})

function openCommentEditor(row: SurveillanceRow) {
  commentVcode.value = row.vcode
  commentText.value = ''
  commentDate.value = new Date().toISOString().split('T')[0]
  showHistory.value = false
  store.loadComments(row.vcode)
  nextTick(() => commentTextarea.value?.focus())
}

function closeCommentEditor() {
  commentVcode.value = null
  showHistory.value = false
}

async function saveComment() {
  if (!commentVcode.value || !commentText.value.trim()) return
  try {
    await store.saveComment(commentVcode.value, commentDate.value, commentText.value.trim())
    commentText.value = ''
  } catch (e: any) {
    alert('Save failed: ' + e.message)
  }
}

function formatCurrency(val: number | null): string {
  if (val == null) return '\u2014'
  if (Math.abs(val) >= 1_000_000) {
    const m = (val / 1_000_000).toFixed(1)
    return '$' + Number(m).toLocaleString('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 }) + 'M'
  }
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

function formatShortDate(dt: string | null): string {
  if (!dt) return ''
  const d = new Date(dt + 'T00:00:00')
  if (isNaN(d.getTime())) return dt
  return `${d.getMonth() + 1}/${d.getDate()}/${String(d.getFullYear()).slice(2)}`
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

function dscrClass(row: SurveillanceRow): string {
  if (row.dscr != null && row.dscr_min != null && row.dscr < row.dscr_min) return 'breach'
  return ''
}

function missingClass(val: number | null): string {
  if (val == null) return ''
  if (val === 0) return 'rpt-complete'
  if (val <= 2) return 'rpt-warn'
  return 'rpt-bad'
}

// Expandable group definitions: key, label, sub-column headers
const groupDefs = [
  { key: 'reporting', label: 'Reporting', cols: ['Occ', 'Rent Roll', 'Inc. Stmt', 'Bal. Sheet'] },
  { key: 'covenants', label: 'Debt Covenants', cols: ['DSCR Min', 'Debt Yield', 'DY Min', 'LTV', 'LTV Max'] },
  { key: 'taxes', label: 'Real Estate Taxes', cols: ['Tax Due', 'Status', 'Amount'] },
  { key: 'insurance', label: 'Insurance', cols: ['Property', 'GL', 'Renewal'] },
  { key: 'ground_leases', label: 'Ground Leases', cols: ['Lease Exp', 'Rent', 'Status'] },
  { key: 'escrows', label: 'Escrows', cols: ['Tax', 'Insurance', 'CapEx'] },
  { key: 'collateral', label: "Add'l Collateral", cols: ['Type', 'Value', 'Notes'] },
]

// Total columns for colspan on empty row
function totalCols(): number {
  let count = 8 // base columns
  for (const g of groupDefs) {
    count += expandedGroups.value[g.key] ? g.cols.length : 1
  }
  return count
}

function handleExportCsv() {
  const headers = [
    'Property', 'Asset Type', 'Occupancy', 'NOI (TTM)', 'DSCR', 'Debt Balance', 'Maturity', 'Comments',
    'Occ Latest', 'Occ Missing', 'Rent Roll Latest', 'Rent Roll Missing',
    'IS Latest', 'IS Missing', 'BS Latest', 'BS Missing',
  ]
  const csvRows = store.filteredRows.map(r => [
    r.name, r.asset_type,
    r.occ_pct != null ? r.occ_pct.toFixed(1) : '',
    r.noi_ttm != null ? r.noi_ttm.toFixed(0) : '',
    r.dscr != null ? r.dscr.toFixed(2) : '',
    r.debt_balance != null ? r.debt_balance.toFixed(0) : '',
    r.maturity_date || '',
    r.comment_text || '',
    r.rpt_occ_latest || '', r.rpt_occ_missing ?? '',
    r.rpt_rent_roll_latest || '', r.rpt_rent_roll_missing ?? '',
    r.rpt_is_latest || '', r.rpt_is_missing ?? '',
    r.rpt_bs_latest || '', r.rpt_bs_missing ?? '',
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
        <span class="kpi-value">{{ formatCurrency(store.dashboard.total_noi_ttm) }}</span>
        <span class="kpi-label">NOI (TTM)</span>
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
          <!-- Group header row -->
          <tr class="group-header-row">
            <th :colspan="8" class="group-spacer"></th>
            <th
              v-for="g in groupDefs"
              :key="g.key"
              :colspan="expandedGroups[g.key] ? g.cols.length : 1"
              class="group-header"
              :class="'group-' + g.key"
              @click="toggleGroup(g.key)"
            >
              <span class="group-toggle">{{ expandedGroups[g.key] ? '\u25BC' : '\u25B6' }}</span>
              {{ g.label }}
            </th>
          </tr>
          <!-- Column header row -->
          <tr>
            <th class="sticky-col col-property">Property</th>
            <th class="col-type">Type</th>
            <th class="col-occ">Occupancy</th>
            <th class="col-noi">NOI (TTM)</th>
            <th class="col-dscr">DSCR</th>
            <th class="col-debt">Debt Balance</th>
            <th class="col-mat">Maturity</th>
            <th class="col-comment">Comments</th>
            <!-- Expandable group sub-columns -->
            <template v-for="g in groupDefs" :key="'hdr-' + g.key">
              <template v-if="expandedGroups[g.key]">
                <th v-for="col in g.cols" :key="g.key + '-' + col" class="col-rpt">{{ col }}</th>
              </template>
              <template v-else>
                <th class="col-rpt-collapsed">&nbsp;</th>
              </template>
            </template>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="row in store.filteredRows"
            :key="row.vcode"
          >
            <td class="sticky-col deal-name" :title="row.vcode">{{ row.name || row.vcode }}</td>
            <td class="col-type-cell">{{ row.asset_type || '\u2014' }}</td>
            <td :class="occupancyClass(row.occ_pct)">
              <div class="occ-cell" v-if="row.occ_pct != null">
                <div class="occ-bar">
                  <div class="occ-fill" :style="{ width: Math.min(row.occ_pct, 100) + '%' }"></div>
                </div>
                <span class="occ-text">{{ row.occ_pct.toFixed(1) }}%</span>
              </div>
              <span v-else>&mdash;</span>
            </td>
            <td class="centered">{{ formatCurrency(row.noi_ttm) }}</td>
            <td class="centered" :class="dscrClass(row)">
              {{ formatRatio(row.dscr) }}
            </td>
            <td class="centered">{{ formatCurrency(row.debt_balance) }}</td>
            <td class="centered" :class="maturityClass(row.maturity_date)">{{ formatDate(row.maturity_date) }}</td>
            <td class="comment-cell" @click="openCommentEditor(row)" :title="row.comment_text || 'Click to add comment'">
              <div class="comment-preview" v-if="row.comment_text">
                <span class="comment-date-badge">{{ formatShortDate(row.comment_date) }}</span>
                <span class="comment-snippet">{{ row.comment_text }}</span>
              </div>
              <span v-else class="comment-empty">+</span>
            </td>
            <!-- Reporting columns -->
            <template v-if="expandedGroups.reporting">
              <td class="rpt-cell" :class="missingClass(row.rpt_occ_missing)">
                <span class="rpt-period">{{ row.rpt_occ_latest || '\u2014' }}</span>
                <span v-if="row.rpt_occ_missing" class="rpt-missing" :title="row.rpt_occ_missing + ' missing in trailing 12mo'">{{ row.rpt_occ_missing }}</span>
              </td>
              <td class="rpt-cell" :class="row.is_commercial ? missingClass(row.rpt_rent_roll_missing) : ''">
                <template v-if="row.is_commercial">
                  <span class="rpt-period">{{ row.rpt_rent_roll_latest || '\u2014' }}</span>
                  <span v-if="row.rpt_rent_roll_missing" class="rpt-missing" :title="row.rpt_rent_roll_missing + ' missing in trailing 12mo'">{{ row.rpt_rent_roll_missing }}</span>
                </template>
                <span v-else class="rpt-na">n/a</span>
              </td>
              <td class="rpt-cell" :class="missingClass(row.rpt_is_missing)">
                <span class="rpt-period">{{ row.rpt_is_latest || '\u2014' }}</span>
                <span v-if="row.rpt_is_missing" class="rpt-missing" :title="row.rpt_is_missing + ' missing in trailing 12mo'">{{ row.rpt_is_missing }}</span>
              </td>
              <td class="rpt-cell" :class="missingClass(row.rpt_bs_missing)">
                <span class="rpt-period">{{ row.rpt_bs_latest || '\u2014' }}</span>
                <span v-if="row.rpt_bs_missing" class="rpt-missing" :title="row.rpt_bs_missing + ' missing in trailing 12mo'">{{ row.rpt_bs_missing }}</span>
              </td>
            </template>
            <template v-else>
              <td class="rpt-summary-cell" :title="'Click Reporting header to expand'">
                <span v-if="(row.rpt_occ_missing || 0) + (row.rpt_is_missing || 0) + (row.rpt_bs_missing || 0) === 0" class="rpt-ok-dot"></span>
                <span v-else class="rpt-missing-total">{{ (row.rpt_occ_missing || 0) + (row.rpt_is_missing || 0) + (row.rpt_bs_missing || 0) + (row.is_commercial ? (row.rpt_rent_roll_missing || 0) : 0) }}</span>
              </td>
            </template>

            <!-- Debt Covenants -->
            <template v-if="expandedGroups.covenants">
              <td class="rpt-cell">{{ row.dscr_min != null ? row.dscr_min.toFixed(2) + 'x' : '\u2014' }}</td>
              <td class="rpt-cell">{{ row.dy_val != null ? formatPct(row.dy_val) : '\u2014' }}</td>
              <td class="rpt-cell">{{ row.dy_min != null ? formatPct(row.dy_min) : '\u2014' }}</td>
              <td class="rpt-cell">{{ row.ltv_val != null ? formatPct(row.ltv_val) : '\u2014' }}</td>
              <td class="rpt-cell">{{ row.ltv_min != null ? formatPct(row.ltv_min) : '\u2014' }}</td>
            </template>
            <td v-else class="rpt-summary-cell"><span class="placeholder-dot"></span></td>

            <!-- Real Estate Taxes -->
            <template v-if="expandedGroups.taxes">
              <td class="rpt-cell">{{ row.tax_due ? formatShortDate(row.tax_due) : '\u2014' }}</td>
              <td class="rpt-cell">&mdash;</td>
              <td class="rpt-cell">&mdash;</td>
            </template>
            <td v-else class="rpt-summary-cell"><span class="placeholder-dot"></span></td>

            <!-- Insurance -->
            <template v-if="expandedGroups.insurance">
              <td class="rpt-cell">
                <span v-if="row.has_property_ins" class="rpt-ok-dot"></span>
                <span v-else>&mdash;</span>
              </td>
              <td class="rpt-cell">
                <span v-if="row.has_gl_ins" class="rpt-ok-dot"></span>
                <span v-else>&mdash;</span>
              </td>
              <td class="rpt-cell">{{ row.ins_renewal ? formatShortDate(row.ins_renewal) : '\u2014' }}</td>
            </template>
            <td v-else class="rpt-summary-cell"><span class="placeholder-dot"></span></td>

            <!-- Ground Leases -->
            <template v-if="expandedGroups.ground_leases">
              <td class="rpt-cell">&mdash;</td>
              <td class="rpt-cell">&mdash;</td>
              <td class="rpt-cell">&mdash;</td>
            </template>
            <td v-else class="rpt-summary-cell"><span class="placeholder-dot"></span></td>

            <!-- Escrows -->
            <template v-if="expandedGroups.escrows">
              <td class="rpt-cell">&mdash;</td>
              <td class="rpt-cell">&mdash;</td>
              <td class="rpt-cell">&mdash;</td>
            </template>
            <td v-else class="rpt-summary-cell"><span class="placeholder-dot"></span></td>

            <!-- Add'l Collateral -->
            <template v-if="expandedGroups.collateral">
              <td class="rpt-cell">&mdash;</td>
              <td class="rpt-cell">&mdash;</td>
              <td class="rpt-cell">&mdash;</td>
            </template>
            <td v-else class="rpt-summary-cell"><span class="placeholder-dot"></span></td>
          </tr>
          <tr v-if="!store.filteredRows.length && !store.loading">
            <td :colspan="totalCols()" class="empty-row">No deals match the current filters.</td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Comment Editor Overlay -->
    <div v-if="commentVcode" class="comment-overlay" @click.self="closeCommentEditor">
      <div class="comment-panel">
        <div class="comment-panel-header">
          <h3>Comments — {{ store.rows.find(r => r.vcode === commentVcode)?.name || commentVcode }}</h3>
          <button class="btn-close" @click="closeCommentEditor">&times;</button>
        </div>

        <!-- New comment form -->
        <div class="comment-form">
          <div class="comment-form-row">
            <label>Date:</label>
            <input type="date" v-model="commentDate" class="comment-date-input" />
          </div>
          <textarea
            ref="commentTextarea"
            v-model="commentText"
            class="comment-textarea"
            placeholder="Enter comment..."
            rows="3"
          ></textarea>
          <button class="btn-save" @click="saveComment" :disabled="!commentText.trim()">Save Comment</button>
        </div>

        <!-- Comment history -->
        <div class="comment-history">
          <div class="comment-history-header" @click="showHistory = !showHistory">
            <span>{{ showHistory ? 'Hide' : 'Show' }} History ({{ store.comments.length }})</span>
            <span class="toggle-icon">{{ showHistory ? '\u25B2' : '\u25BC' }}</span>
          </div>
          <div v-if="showHistory && store.comments.length" class="comment-list">
            <div v-for="c in store.comments" :key="c.id" class="comment-item">
              <div class="comment-item-header">
                <span class="comment-item-date">{{ formatShortDate(c.comment_date) }}</span>
                <span class="comment-item-by" v-if="c.created_by">{{ c.created_by }}</span>
              </div>
              <div class="comment-item-text">{{ c.comment_text }}</div>
            </div>
          </div>
          <div v-if="showHistory && !store.comments.length" class="comment-list-empty">
            No comments yet.
          </div>
        </div>
      </div>
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

/* Group header row */
.group-header-row th {
  padding: 6px 12px;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  border-bottom: 2px solid rgba(255, 255, 255, 0.3);
}

.group-spacer {
  background: var(--color-accent) !important;
  border: none;
}

.group-header {
  cursor: pointer;
  user-select: none;
  text-align: center;
  background: #37474f !important;
  border-left: 2px solid rgba(255, 255, 255, 0.2);
}
.group-header:hover { background: #455a64 !important; }

.group-toggle {
  font-size: 9px;
  margin-right: 4px;
}

/* Column widths */
.col-property { width: 220px; min-width: 180px; }
.col-type { width: 130px; min-width: 100px; }
.col-occ { width: 130px; min-width: 110px; text-align: center; }
.col-noi { width: 110px; min-width: 90px; text-align: center; }
.col-dscr { width: 80px; min-width: 65px; text-align: center; }
.col-debt { width: 120px; min-width: 95px; text-align: center; }
.col-mat { width: 110px; min-width: 90px; text-align: center; }
.col-comment { min-width: 200px; }

/* Reporting sub-column headers */
.col-rpt {
  width: 85px;
  min-width: 75px;
  text-align: center;
  font-size: 11px;
  background: #455a64 !important;
  border-left: 1px solid rgba(255, 255, 255, 0.15);
}

.col-rpt-collapsed {
  width: 36px;
  min-width: 36px;
  text-align: center;
  background: #455a64 !important;
  border-left: 2px solid rgba(255, 255, 255, 0.2);
}

.surv-table tbody tr { background: white; }
.surv-table tbody tr:hover { background: #f5f5f5; }

.deal-name {
  font-weight: 500;
  max-width: 220px;
  overflow: hidden;
  text-overflow: ellipsis;
}

.col-type-cell {
  font-size: 12px;
  color: var(--color-text-secondary);
}

.centered {
  text-align: center;
  font-variant-numeric: tabular-nums;
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
  justify-content: center;
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

/* Reporting cells */
.rpt-cell {
  text-align: center;
  font-size: 12px;
  border-left: 1px solid #e8e8e8;
  padding: 5px 6px !important;
}

.rpt-period {
  color: var(--color-text);
  font-variant-numeric: tabular-nums;
}

.rpt-missing {
  display: inline-block;
  background: #ef5350;
  color: white;
  font-size: 10px;
  font-weight: 700;
  min-width: 16px;
  height: 16px;
  line-height: 16px;
  text-align: center;
  border-radius: 8px;
  margin-left: 4px;
  cursor: default;
}

.rpt-complete .rpt-period { color: #2e7d32; }
.rpt-warn .rpt-missing { background: #ff9800; }
.rpt-bad .rpt-missing { background: #ef5350; }

.rpt-na {
  color: #bdbdbd;
  font-size: 11px;
  font-style: italic;
}

/* Collapsed reporting summary */
.rpt-summary-cell {
  text-align: center;
  border-left: 2px solid #e0e0e0;
  width: 36px;
  padding: 5px 4px !important;
}

.rpt-ok-dot {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: #66bb6a;
}

.rpt-missing-total {
  display: inline-block;
  background: #ef5350;
  color: white;
  font-size: 10px;
  font-weight: 700;
  min-width: 18px;
  height: 18px;
  line-height: 18px;
  text-align: center;
  border-radius: 9px;
}

.placeholder-dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #e0e0e0;
}

/* Alternating group header colors for visual separation */
.group-covenants { background: #4a5568 !important; }
.group-covenants:hover { background: #5a6778 !important; }
.group-taxes { background: #37474f !important; }
.group-taxes:hover { background: #455a64 !important; }
.group-insurance { background: #4a5568 !important; }
.group-insurance:hover { background: #5a6778 !important; }
.group-ground_leases { background: #37474f !important; }
.group-ground_leases:hover { background: #455a64 !important; }
.group-escrows { background: #4a5568 !important; }
.group-escrows:hover { background: #5a6778 !important; }
.group-collateral { background: #37474f !important; }
.group-collateral:hover { background: #455a64 !important; }

/* Comment cell — clickable */
.comment-cell {
  cursor: pointer;
  max-width: 260px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.comment-cell:hover { background: #f0f0f0; }

.comment-preview {
  display: flex;
  align-items: center;
  gap: 6px;
}

.comment-date-badge {
  background: #e3f2fd;
  color: #1565c0;
  font-size: 10px;
  font-weight: 600;
  padding: 1px 5px;
  border-radius: 3px;
  white-space: nowrap;
  flex-shrink: 0;
}

.comment-snippet {
  color: var(--color-text-secondary);
  font-size: 12px;
  overflow: hidden;
  text-overflow: ellipsis;
}

.comment-empty {
  color: #bdbdbd;
  font-size: 16px;
  font-weight: 600;
}

/* Comment Overlay */
.comment-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.3);
  z-index: 100;
  display: flex;
  align-items: center;
  justify-content: center;
}

.comment-panel {
  background: white;
  border-radius: 10px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
  width: 480px;
  max-height: 80vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.comment-panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 14px 18px;
  border-bottom: 1px solid var(--color-border);
}

.comment-panel-header h3 {
  margin: 0;
  font-size: 15px;
  font-weight: 600;
}

.btn-close {
  background: none;
  border: none;
  font-size: 22px;
  cursor: pointer;
  color: var(--color-text-secondary);
  line-height: 1;
  padding: 0 4px;
}
.btn-close:hover { color: var(--color-text); }

/* Comment form */
.comment-form {
  padding: 14px 18px;
  border-bottom: 1px solid var(--color-border);
}

.comment-form-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.comment-form-row label {
  font-size: 13px;
  font-weight: 500;
  color: var(--color-text-secondary);
}

.comment-date-input {
  padding: 4px 8px;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  font-size: 13px;
}

.comment-textarea {
  width: 100%;
  padding: 8px;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  font-size: 13px;
  font-family: inherit;
  resize: vertical;
  margin-bottom: 8px;
  box-sizing: border-box;
}

.btn-save {
  padding: 6px 16px;
  background: #2e7d32;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 13px;
}
.btn-save:hover { opacity: 0.9; }
.btn-save:disabled { opacity: 0.5; cursor: default; }

/* Comment history */
.comment-history {
  overflow-y: auto;
  flex: 1;
}

.comment-history-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 18px;
  cursor: pointer;
  font-size: 13px;
  font-weight: 500;
  color: var(--color-text-secondary);
  user-select: none;
}
.comment-history-header:hover { background: #fafafa; }

.toggle-icon { font-size: 10px; }

.comment-list {
  padding: 0 18px 14px;
}

.comment-item {
  padding: 8px 0;
  border-bottom: 1px solid #f0f0f0;
}
.comment-item:last-child { border-bottom: none; }

.comment-item-header {
  display: flex;
  gap: 8px;
  align-items: center;
  margin-bottom: 4px;
}

.comment-item-date {
  font-size: 12px;
  font-weight: 600;
  color: #1565c0;
}

.comment-item-by {
  font-size: 11px;
  color: var(--color-text-secondary);
}

.comment-item-text {
  font-size: 13px;
  color: var(--color-text);
  white-space: pre-wrap;
  line-height: 1.4;
}

.comment-list-empty {
  padding: 12px 18px;
  font-size: 13px;
  color: var(--color-text-secondary);
  font-style: italic;
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
