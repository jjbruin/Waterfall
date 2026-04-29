<script setup lang="ts">
import { ref, onMounted, computed, watch } from 'vue'
import DataTable from '../components/common/DataTable.vue'
import KpiCard from '../components/common/KpiCard.vue'
import api from '../api/client'
import { useDataStore } from '../stores/data'

const dataStore = useDataStore()

const summary = ref<any[]>([])
const dealNames = ref<any[]>([])
const loading = ref(false)
const selectedDealVcode = ref('')

// Deal detail state
const detail = ref<any[]>([])
const detailSummary = ref<any>({})
const detailLoading = ref(false)

// Net returns state
const assumptions = ref({
  ownership_pct: 90,
  am_fee_pct: 1.0,
  hurdle_rate: 8.0,
  promote_pct: 20,
  annual_expenses: 50000,
})
const netResults = ref<any>(null)
const netLoading = ref(false)
const showNet = computed(() => netResults.value !== null)

const detailColumns = [
  { key: 'Date', label: 'Date' },
  { key: 'InvestorID', label: 'InvestorID' },
  { key: 'MajorType', label: 'MajorType' },
  { key: 'Typename', label: 'Typename' },
  { key: 'Capital', label: 'Capital' },
  { key: 'Amount', label: 'Amount', format: 'currency', align: 'right' },
  { key: 'Cashflow (XIRR)', label: 'Cashflow (XIRR)', format: 'currency', align: 'right' },
  { key: 'Capital Balance', label: 'Capital Balance', format: 'currency', align: 'right' },
]

onMounted(async () => {
  loading.value = true
  try {
    const res = await api.get('/api/sold-portfolio/summary')
    summary.value = res.data.rows
    dealNames.value = res.data.deal_names || []
  } catch (e: any) {
    dataStore.addToast('Failed to load sold portfolio: ' + (e.response?.data?.error || e.message), 'error')
  } finally {
    loading.value = false
  }
})

async function loadDealDetail() {
  if (!selectedDealVcode.value) return
  detailLoading.value = true
  detail.value = []
  detailSummary.value = {}
  try {
    const res = await api.get(`/api/sold-portfolio/detail/${selectedDealVcode.value}`)
    detail.value = res.data.rows || []
    detailSummary.value = res.data.summary || {}
  } catch (e: any) {
    dataStore.addToast('Failed to load deal detail: ' + (e.response?.data?.error || e.message), 'error')
  } finally {
    detailLoading.value = false
  }
}

const selectedDealName = computed(() => {
  const d = dealNames.value.find((n) => n.vcode === selectedDealVcode.value)
  return d ? d.name : ''
})

const hasDetail = computed(() => detail.value.length > 0)

// Merge gross + net for summary table display
const mergedSummary = computed(() => {
  if (!showNet.value) return summary.value
  const netByVcode: Record<string, any> = {}
  for (const dr of netResults.value.deal_results) {
    netByVcode[dr.vcode] = dr
  }
  return summary.value.map((row: any) => {
    const isTotal = row._is_deal_total
    if (isTotal) {
      return {
        ...row,
        'Net Contributions': netResults.value.portfolio['Net Contributions'],
        'Net Distributions': netResults.value.portfolio['Net Distributions'],
        'Net IRR': netResults.value.portfolio['Net IRR'],
        'Net ROE': netResults.value.portfolio['Net ROE'],
        'Net MOIC': netResults.value.portfolio['Net MOIC'],
      }
    }
    const nd = netByVcode[row.vcode]
    if (nd) {
      return {
        ...row,
        'Net Contributions': nd['Net Contributions'],
        'Net Distributions': nd['Net Distributions'],
        'Net IRR': nd['Net IRR'],
        'Net ROE': nd['Net ROE'],
        'Net MOIC': nd['Net MOIC'],
      }
    }
    return row
  })
})

// Formatters
function fmtCur(v: any): string {
  if (v == null || v !== v) return '--'
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 0, maximumFractionDigits: 0 }).format(v)
}
function fmtPct(v: any): string {
  if (v == null || v !== v) return 'N/A'
  return (v * 100).toFixed(2) + '%'
}
function fmtMult(v: any): string {
  if (v == null || v !== v) return '--'
  return v.toFixed(2) + 'x'
}
function fmtCell(val: any, format?: string): string {
  if (format === 'currency') return fmtCur(val)
  if (format === 'percent') return fmtPct(val)
  if (format === 'multiple') return fmtMult(val)
  if (val == null) return ''
  return String(val)
}

// Net returns computation
async function computeNetReturns() {
  netLoading.value = true
  try {
    const res = await api.post('/api/sold-portfolio/net-returns', {
      ownership_pct: assumptions.value.ownership_pct / 100,
      am_fee_pct: assumptions.value.am_fee_pct / 100,
      hurdle_rate: assumptions.value.hurdle_rate / 100,
      promote_pct: assumptions.value.promote_pct / 100,
      annual_expenses: assumptions.value.annual_expenses,
    })
    netResults.value = res.data
  } catch (e: any) {
    dataStore.addToast('Failed to compute net returns: ' + (e.response?.data?.error || e.message), 'error')
  } finally {
    netLoading.value = false
  }
}

const assumptionsFootnote = computed(() => {
  if (!showNet.value) return ''
  const a = assumptions.value
  return `Net Returns Assumptions: Investor Ownership ${a.ownership_pct.toFixed(1)}%, AM Fee ${a.am_fee_pct.toFixed(2)}%, Hurdle Rate ${a.hurdle_rate.toFixed(2)}%, Promote ${a.promote_pct.toFixed(1)}%, Annual Expenses $${a.annual_expenses.toLocaleString()}`
})

// Downloads
async function downloadSummaryExcel() {
  const res = await api.get('/api/sold-portfolio/summary/excel', { responseType: 'blob' })
  const url = URL.createObjectURL(new Blob([res.data]))
  const a = document.createElement('a')
  a.href = url
  a.download = 'sold_portfolio_returns.xlsx'
  a.click()
  URL.revokeObjectURL(url)
}

async function downloadNetExcel() {
  const res = await api.post('/api/sold-portfolio/net-returns/excel', {
    ownership_pct: assumptions.value.ownership_pct / 100,
    am_fee_pct: assumptions.value.am_fee_pct / 100,
    hurdle_rate: assumptions.value.hurdle_rate / 100,
    promote_pct: assumptions.value.promote_pct / 100,
    annual_expenses: assumptions.value.annual_expenses,
  }, { responseType: 'blob' })
  const url = URL.createObjectURL(new Blob([res.data]))
  const a = document.createElement('a')
  a.href = url
  a.download = 'sold_portfolio_net_returns.xlsx'
  a.click()
  URL.revokeObjectURL(url)
}

async function downloadDetailExcel() {
  if (!selectedDealVcode.value) return
  const res = await api.get(`/api/sold-portfolio/detail/${selectedDealVcode.value}/excel`, { responseType: 'blob' })
  const url = URL.createObjectURL(new Blob([res.data]))
  const a = document.createElement('a')
  a.href = url
  a.download = `sold_activity_${selectedDealName.value.replace(/ /g, '_')}.xlsx`
  a.click()
  URL.revokeObjectURL(url)
}
</script>

<template>
  <div class="sold-portfolio">
    <h2>Sold Portfolio — Historical Returns</h2>

    <!-- Toolbar -->
    <div class="toolbar">
      <button v-if="summary.length" class="btn-download" @click="downloadSummaryExcel">
        Download Summary Excel
      </button>
      <button v-if="showNet" class="btn-download btn-net-excel" @click="downloadNetExcel">
        Download Net Returns (Excel)
      </button>
    </div>

    <!-- Assumptions Panel -->
    <div v-if="summary.length > 0" class="assumptions-panel">
      <div class="assumptions-row">
        <label class="assumption-field">
          <span>Ownership %</span>
          <input type="number" v-model.number="assumptions.ownership_pct" step="1" min="0" max="100" />
        </label>
        <label class="assumption-field">
          <span>AM Fee %</span>
          <input type="number" v-model.number="assumptions.am_fee_pct" step="0.1" min="0" max="100" />
        </label>
        <label class="assumption-field">
          <span>Hurdle Rate %</span>
          <input type="number" v-model.number="assumptions.hurdle_rate" step="0.1" min="0" max="100" />
        </label>
        <label class="assumption-field">
          <span>Promote %</span>
          <input type="number" v-model.number="assumptions.promote_pct" step="1" min="0" max="100" />
        </label>
        <label class="assumption-field">
          <span>Annual Expenses</span>
          <input type="number" v-model.number="assumptions.annual_expenses" step="1000" min="0" />
        </label>
        <button class="btn-compute" @click="computeNetReturns" :disabled="netLoading">
          {{ netLoading ? 'Computing...' : 'Compute Net Returns' }}
        </button>
      </div>
    </div>

    <!-- Summary Table -->
    <div v-if="loading" class="placeholder">Loading sold portfolio data...</div>
    <div v-else-if="summary.length" class="summary-table-wrapper">
      <table class="summary-table">
        <thead>
          <tr class="group-header" v-if="showNet">
            <th colspan="3"></th>
            <th colspan="2"></th>
            <th colspan="3" class="group-gross">Gross Returns</th>
            <th class="spacer-col"></th>
            <th colspan="3" class="group-net">Net Returns</th>
          </tr>
          <tr>
            <th>Investment Name</th>
            <th>Acquisition Date</th>
            <th>Sale Date</th>
            <th class="right">Total Contributions</th>
            <th class="right">Total Distributions</th>
            <th class="right">IRR</th>
            <th class="right">ROE</th>
            <th class="right">MOIC</th>
            <template v-if="showNet">
              <th class="spacer-col"></th>
              <th class="right">Net IRR</th>
              <th class="right">Net ROE</th>
              <th class="right">Net MOIC</th>
            </template>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(row, idx) in mergedSummary" :key="idx"
              :class="{ 'total-row': row._is_deal_total }">
            <td>{{ row['Investment Name'] }}</td>
            <td>{{ row['Acquisition Date'] }}</td>
            <td>{{ row['Sale Date'] }}</td>
            <td class="right">{{ fmtCur(row['Total Contributions']) }}</td>
            <td class="right">{{ fmtCur(row['Total Distributions']) }}</td>
            <td class="right">{{ fmtPct(row['IRR']) }}</td>
            <td class="right">{{ fmtPct(row['ROE']) }}</td>
            <td class="right">{{ fmtMult(row['MOIC']) }}</td>
            <template v-if="showNet">
              <td class="spacer-col"></td>
              <td class="right">{{ fmtPct(row['Net IRR']) }}</td>
              <td class="right">{{ fmtPct(row['Net ROE']) }}</td>
              <td class="right">{{ fmtMult(row['Net MOIC']) }}</td>
            </template>
          </tr>
        </tbody>
      </table>

      <!-- Assumptions Footnote -->
      <p v-if="showNet" class="assumptions-footnote">{{ assumptionsFootnote }}</p>
    </div>
    <p v-else class="placeholder">No sold deals found.</p>

    <!-- Deal Detail Drill-Down -->
    <template v-if="dealNames.length > 0">
      <div class="detail-section">
        <h3>Deal Activity Detail</h3>
        <div class="detail-controls">
          <select v-model="selectedDealVcode" @change="loadDealDetail" class="deal-select">
            <option value="">-- Select a deal --</option>
            <option v-for="d in dealNames" :key="d.vcode" :value="d.vcode">
              {{ d.name }}
            </option>
          </select>
        </div>

        <div v-if="detailLoading" class="placeholder">Loading activity detail...</div>

        <template v-if="hasDetail">
          <!-- Detail Table -->
          <DataTable :columns="detailColumns" :rows="detail" />

          <!-- Metric Cards -->
          <div v-if="detailSummary" class="metric-cards">
            <div class="metric-card">
              <span class="metric-label">IRR</span>
              <span class="metric-value">{{ fmtPct(detailSummary.IRR) }}</span>
            </div>
            <div class="metric-card">
              <span class="metric-label">ROE</span>
              <span class="metric-value">{{ fmtPct(detailSummary.ROE) }}</span>
            </div>
            <div class="metric-card">
              <span class="metric-label">MOIC</span>
              <span class="metric-value">{{ fmtMult(detailSummary.MOIC) }}</span>
            </div>
            <div class="metric-card">
              <span class="metric-label">Contributions</span>
              <span class="metric-value">{{ fmtCur(detailSummary['Total Contributions']) }}</span>
            </div>
            <div class="metric-card">
              <span class="metric-label">Distributions</span>
              <span class="metric-value">{{ fmtCur(detailSummary['Total Distributions']) }}</span>
            </div>
          </div>

          <!-- Detail Excel Download -->
          <button class="btn-download detail-download" @click="downloadDetailExcel">
            Download Activity Detail
          </button>
        </template>

        <p v-else-if="selectedDealVcode && !detailLoading" class="placeholder">
          No pref equity accounting activity found for this deal.
        </p>
      </div>
    </template>
  </div>
</template>

<style scoped>
.sold-portfolio { padding: 0 0 40px 0; }
h2 { font-size: 20px; margin-bottom: 16px; }
h3 { font-size: 16px; margin: 0 0 12px 0; }

.toolbar {
  margin-bottom: 12px;
  display: flex;
  gap: 8px;
}

.btn-download {
  padding: 8px 20px;
  background: var(--color-pref);
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
}

.btn-download:hover { opacity: 0.9; }

.btn-net-excel {
  background: #2e7d32;
}

.placeholder {
  color: var(--color-text-secondary);
  font-style: italic;
  text-align: center;
  padding: 32px 0;
}

/* Assumptions Panel */
.assumptions-panel {
  background: #f8f9fa;
  border: 1px solid var(--color-border);
  border-radius: 8px;
  padding: 12px 16px;
  margin-bottom: 16px;
}

.assumptions-row {
  display: flex;
  align-items: flex-end;
  gap: 16px;
  flex-wrap: wrap;
}

.assumption-field {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.assumption-field span {
  font-size: 11px;
  color: var(--color-text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.assumption-field input {
  padding: 6px 10px;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  font-size: 13px;
  width: 120px;
}

.btn-compute {
  padding: 8px 20px;
  background: #1565c0;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
  white-space: nowrap;
}
.btn-compute:hover { opacity: 0.9; }
.btn-compute:disabled { opacity: 0.6; cursor: not-allowed; }

/* Summary Table */
.summary-table-wrapper {
  overflow-x: auto;
}

.summary-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
  white-space: nowrap;
}

.summary-table th,
.summary-table td {
  padding: 8px 12px;
  text-align: left;
  border-bottom: 1px solid var(--color-border);
}

.summary-table th {
  background: #f1f3f5;
  font-weight: 600;
  font-size: 12px;
}

.summary-table th.right,
.summary-table td.right {
  text-align: right;
}

.summary-table .group-header th {
  border-bottom: none;
  padding-bottom: 2px;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.group-gross {
  text-align: center !important;
  color: var(--color-text-secondary);
}

.group-net {
  text-align: center !important;
  color: #2e7d32;
  font-weight: 700 !important;
}

.spacer-col {
  width: 4px !important;
  min-width: 4px !important;
  max-width: 4px !important;
  padding: 0 !important;
  border-left: 2px solid #999;
  background: transparent;
}

.total-row td {
  font-weight: 700;
  border-top: 2px solid #333;
}

.assumptions-footnote {
  font-size: 11px;
  color: var(--color-text-secondary);
  margin-top: 8px;
  font-style: italic;
}

/* Detail Section */
.detail-section {
  margin-top: 24px;
  padding-top: 20px;
  border-top: 1px solid var(--color-border);
}

.detail-controls {
  margin-bottom: 12px;
}

.deal-select {
  padding: 8px 12px;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  font-size: 14px;
  min-width: 350px;
}

/* Metric Cards */
.metric-cards {
  display: flex;
  gap: 16px;
  margin: 16px 0;
  flex-wrap: wrap;
}

.metric-card {
  display: flex;
  flex-direction: column;
  padding: 12px 20px;
  background: #f8f9fa;
  border: 1px solid var(--color-border);
  border-radius: 8px;
  min-width: 130px;
}

.metric-label {
  font-size: 11px;
  color: var(--color-text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 4px;
}

.metric-value {
  font-size: 18px;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
}

.detail-download {
  margin-top: 8px;
}
</style>
