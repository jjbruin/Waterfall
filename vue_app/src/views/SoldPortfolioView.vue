<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
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

const summaryColumns = [
  { key: 'Investment Name', label: 'Investment Name' },
  { key: 'Acquisition Date', label: 'Acquisition Date' },
  { key: 'Sale Date', label: 'Sale Date' },
  { key: 'Total Contributions', label: 'Total Contributions', format: 'currency', align: 'right' },
  { key: 'Total Distributions', label: 'Total Distributions', format: 'currency', align: 'right' },
  { key: 'IRR', label: 'IRR', format: 'percent', align: 'right' },
  { key: 'ROE', label: 'ROE', format: 'percent', align: 'right' },
  { key: 'MOIC', label: 'MOIC', format: 'multiple', align: 'right' },
]

const detailColumns = [
  { key: 'Date', label: 'Date' },
  { key: 'InvestorID', label: 'InvestorID' },
  { key: 'MajorType', label: 'MajorType' },
  { key: 'Typename', label: 'Typename' },
  { key: 'Capital', label: 'Capital' },
  { key: 'Amount', label: 'Amount', format: 'currency2', align: 'right' },
  { key: 'Cashflow (XIRR)', label: 'Cashflow (XIRR)', format: 'currency2', align: 'right' },
  { key: 'Capital Balance', label: 'Capital Balance', format: 'currency2', align: 'right' },
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

// Formatters
function fmtCur(v: any): string {
  if (v == null || v !== v) return '--'
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 0 }).format(v)
}
function fmtPct(v: any): string {
  if (v == null || v !== v) return 'N/A'
  return (v * 100).toFixed(2) + '%'
}
function fmtMult(v: any): string {
  if (v == null || v !== v) return '--'
  return v.toFixed(2) + 'x'
}

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

    <!-- Summary Table -->
    <div class="toolbar">
      <button v-if="summary.length" class="btn-download" @click="downloadSummaryExcel">
        Download Summary Excel
      </button>
    </div>

    <div v-if="loading" class="placeholder">Loading sold portfolio data...</div>
    <DataTable v-else-if="summary.length" :columns="summaryColumns" :rows="summary" :highlight-total="true" />
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

.toolbar { margin-bottom: 12px; }

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

.placeholder {
  color: var(--color-text-secondary);
  font-style: italic;
  text-align: center;
  padding: 32px 0;
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
