<script setup lang="ts">
import { ref, onMounted, computed, watch } from 'vue'
import { useDataStore } from '../stores/data'
import { useDealsStore } from '../stores/deals'
import DataTable from '../components/common/DataTable.vue'
import SoldPortfolioView from './SoldPortfolioView.vue'
import api from '../api/client'

const data = useDataStore()
const deals = useDealsStore()

// --- Report registry ---
interface ReportDef {
  value: string
  label: string
  description: string
  endpoint: string
  excelEndpoint: string
  excelFilename: string
  columns: { key: string; label: string; format?: string; align?: string }[]
  hasReportDate?: boolean
  highlightTotal?: boolean
  hasDealInvestorSelector?: boolean
  isCustomView?: boolean
}

const reportDefs: ReportDef[] = [
  {
    value: 'projected-returns',
    label: 'Projected Returns Summary',
    description: 'Partner-level projected IRR, ROE, and MOIC from waterfall analysis',
    endpoint: '/api/reports/projected-returns',
    excelEndpoint: '/api/reports/projected-returns/excel',
    excelFilename: 'projected_returns.xlsx',
    highlightTotal: true,
    columns: [
      { key: 'Deal Name', label: 'Deal Name' },
      { key: 'Partner', label: 'Partner' },
      { key: 'Contributions', label: 'Contributions', format: 'currency', align: 'right' },
      { key: 'CF Distributions', label: 'CF Distributions', format: 'currency', align: 'right' },
      { key: 'Capital Distributions', label: 'Capital Distributions', format: 'currency', align: 'right' },
      { key: 'IRR', label: 'IRR', format: 'percent', align: 'right' },
      { key: 'ROE', label: 'ROE', format: 'percent', align: 'right' },
      { key: 'MOIC', label: 'MOIC', format: 'multiple', align: 'right' },
    ],
  },
  {
    value: 'roe-summary',
    label: 'ROE Summary',
    description: 'Inception-to-date return on equity by deal from actual accounting data',
    endpoint: '/api/reports/roe-summary',
    excelEndpoint: '/api/reports/roe-summary/excel',
    excelFilename: 'roe_summary.xlsx',
    hasReportDate: true,
    columns: [
      { key: 'Deal Name', label: 'Deal Name' },
      { key: 'Total Funded', label: 'Total Funded', format: 'currency', align: 'right' },
      { key: 'Return of Capital', label: 'Return of Capital', format: 'currency', align: 'right' },
      { key: 'Current Balance', label: 'Current Balance', format: 'currency', align: 'right' },
      { key: 'Wtd Avg Balance', label: 'Wtd Avg Balance', format: 'currency', align: 'right' },
      { key: 'CF Received', label: 'CF Received', format: 'currency', align: 'right' },
      { key: 'Accrued Pref', label: 'Accrued Pref', format: 'currency', align: 'right' },
      { key: 'ITD ROE', label: 'ITD ROE', format: 'percent', align: 'right' },
      { key: 'U/W ITD ROE', label: 'U/W ITD ROE', format: 'percent', align: 'right' },
    ],
  },
  {
    value: 'pref-balance-detail',
    label: 'Pref Balance Detail',
    description: 'Accrued preferred return detail by deal and investor (Act/Act)',
    endpoint: '/api/reports/pref-balance-detail',
    excelEndpoint: '/api/reports/pref-balance-detail/excel',
    excelFilename: 'pref_balance_detail.xlsx',
    hasReportDate: true,
    hasDealInvestorSelector: true,
    columns: [
      { key: 'EffectiveDate', label: 'Date' },
      { key: 'Typename', label: 'Event' },
      { key: 'Amt', label: 'Amount', format: 'currency', align: 'right' },
      { key: 'Investment_Balance', label: 'Inv Balance', format: 'currency', align: 'right' },
      { key: 'Compounded Pref', label: 'Compounded', format: 'currency', align: 'right' },
      { key: 'Inv + Comp', label: 'Inv + Comp', format: 'currency', align: 'right' },
      { key: 'DaysSinceLast', label: 'Days', align: 'right' },
      { key: 'Current Due', label: 'Current Due', format: 'currency', align: 'right' },
      { key: 'Accrued Pref', label: 'Accrued Pref', format: 'currency', align: 'right' },
      { key: 'Total Due', label: 'Total Due', format: 'currency', align: 'right' },
      { key: 'Pref Paid', label: 'Pref Paid', format: 'currency', align: 'right' },
      { key: 'Remaining Accrual', label: 'Remaining', format: 'currency', align: 'right' },
    ],
  },
  {
    value: 'sold-portfolio',
    label: 'Sold Portfolio',
    description: 'Historical returns for sold deals from accounting data',
    isCustomView: true,
    endpoint: '',
    excelEndpoint: '',
    excelFilename: '',
    columns: [],
  },
]

// --- State ---
const selectedReport = ref('')
const population = ref<'current' | 'select' | 'partner' | 'all'>('all')
const selectedVcodes = ref<string[]>([])
const results = ref<any[]>([])
const errors = ref<any[]>([])
const loading = ref(false)
const showErrors = ref(false)
const reportDate = ref(new Date().toISOString().slice(0, 10))

const eligibleDeals = ref<any[]>([])
const partners = ref<any[]>([])
const selectedPartner = ref('')

// Pref Balance Detail state
const prefDealVcode = ref('')
const prefInvestorId = ref('')
const prefInvestors = ref<any[]>([])
const prefHeader = ref<any>(null)

const activeReport = computed(() => reportDefs.find((r) => r.value === selectedReport.value))
const isPrefDetail = computed(() => activeReport.value?.hasDealInvestorSelector ?? false)
const isCustomView = computed(() => activeReport.value?.isCustomView ?? false)

onMounted(async () => {
  if (data.deals.length === 0) await data.loadDeals()
  try {
    const res = await api.get('/api/reports/deal-lookup')
    eligibleDeals.value = res.data.eligible
  } catch { /* ignore */ }
})

watch(population, async (val) => {
  if (val === 'partner' && partners.value.length === 0) {
    try {
      const res = await api.get('/api/reports/partners')
      partners.value = res.data.partners
    } catch { /* ignore */ }
  }
})

// Clear results when report changes
watch(selectedReport, () => {
  results.value = []
  errors.value = []
  prefHeader.value = null
  prefInvestorId.value = ''
  prefInvestors.value = []
  prefDealVcode.value = ''
})

// Load investors when deal changes for pref balance detail
watch(prefDealVcode, async (vcode) => {
  prefInvestorId.value = ''
  prefInvestors.value = []
  prefHeader.value = null
  results.value = []
  if (!vcode) return
  try {
    const res = await api.get(`/api/reports/pref-balance-detail/investors/${vcode}`)
    prefInvestors.value = res.data.investors || []
  } catch { /* ignore */ }
})

const resolvedVcodes = computed(() => {
  switch (population.value) {
    case 'current':
      return deals.currentVcode ? [deals.currentVcode] : []
    case 'select':
      return selectedVcodes.value
    case 'partner': {
      const p = partners.value.find((p) => p.partner === selectedPartner.value)
      return p ? p.vcodes : []
    }
    case 'all':
      return eligibleDeals.value.map((d) => d.vcode)
    default:
      return []
  }
})

const populationLabel = computed(() => {
  const count = resolvedVcodes.value.length
  switch (population.value) {
    case 'current':
      return deals.currentVcode ? `Current Deal: ${data.getDealName(deals.currentVcode)}` : 'No deal selected'
    case 'select':
      return `${count} deal(s) selected`
    case 'partner':
      return selectedPartner.value ? `${selectedPartner.value} — ${count} deal(s)` : 'Select a partner'
    case 'all':
      return `${count} deals with waterfalls`
    default:
      return ''
  }
})

const canGenerate = computed(() => {
  if (isPrefDetail.value) {
    return prefDealVcode.value && prefInvestorId.value
  }
  return resolvedVcodes.value.length > 0
})

function buildPayload() {
  if (isPrefDetail.value) {
    return {
      vcode: prefDealVcode.value,
      investor_id: prefInvestorId.value,
      report_date: reportDate.value,
    }
  }
  const payload: any = { vcodes: resolvedVcodes.value }
  if (activeReport.value?.hasReportDate) {
    payload.report_date = reportDate.value
  }
  return payload
}

async function generate() {
  if (!activeReport.value || !canGenerate.value) return
  loading.value = true
  errors.value = []
  prefHeader.value = null
  try {
    const res = await api.post(activeReport.value.endpoint, buildPayload())
    if (isPrefDetail.value) {
      results.value = res.data.rows || []
      prefHeader.value = res.data.header || null
    } else {
      results.value = res.data.rows
      errors.value = res.data.errors || []
    }
  } finally {
    loading.value = false
  }
}

async function downloadExcel() {
  if (!activeReport.value || !canGenerate.value) return
  const res = await api.post(activeReport.value.excelEndpoint, buildPayload(), { responseType: 'blob' })
  const url = URL.createObjectURL(new Blob([res.data]))
  const a = document.createElement('a')
  a.href = url
  const fn = isPrefDetail.value
    ? `pref_balance_${prefDealVcode.value}_${prefInvestorId.value}.xlsx`
    : activeReport.value.excelFilename
  a.download = fn
  a.click()
  URL.revokeObjectURL(url)
}

function toggleDeal(vcode: string) {
  const idx = selectedVcodes.value.indexOf(vcode)
  if (idx >= 0) selectedVcodes.value.splice(idx, 1)
  else selectedVcodes.value.push(vcode)
}

function fmtCurr(v: number | null | undefined): string {
  if (v == null) return '-'
  return '$' + Math.round(v).toLocaleString()
}

function fmtPct(v: number | null | undefined): string {
  if (v == null) return '-'
  return (v * 100).toFixed(2) + '%'
}

function fmtDate(val: string | null | undefined): string {
  if (!val) return ''
  const m = String(val).match(/^(\d{4})-(\d{2})-(\d{2})/)
  if (m) return `${parseInt(m[2])}/${parseInt(m[3])}/${m[1]}`
  return String(val)
}

// --- ROE Detail (single-deal drill-down) ---
const isRoeSummary = computed(() => activeReport.value?.value === 'roe-summary')
const roeDetailMode = ref<'actual' | 'uw'>('actual')

const roeDetailRow = computed(() => {
  if (!isRoeSummary.value || results.value.length !== 1) return null
  const row = results.value[0]
  if (!row._detail_rows || !row._detail_rows.length) return null
  return row
})

const hasUwDetail = computed(() => {
  if (!roeDetailRow.value) return false
  const uw = roeDetailRow.value._uw_detail_rows
  return uw && uw.length > 0
})

const roeDetailColumns = [
  { key: 'Date', label: 'Date' },
  { key: 'Event', label: 'Event' },
  { key: 'Amount', label: 'Amount', format: 'currency', align: 'right' },
  { key: 'Days', label: 'Days', align: 'right' },
  { key: 'Capital Balance', label: 'Capital Balance', format: 'currency', align: 'right' },
  { key: 'Weighted Capital', label: 'Weighted Capital', format: 'currency', align: 'right' },
  { key: 'New Balance', label: 'New Balance', format: 'currency', align: 'right' },
]

const roeDetailRows = computed(() => {
  if (!roeDetailRow.value) return []
  const source = roeDetailMode.value === 'uw'
    ? roeDetailRow.value._uw_detail_rows || []
    : roeDetailRow.value._detail_rows || []
  return source.map((r: any) => ({
    ...r,
    Date: fmtDate(r.Date),
  }))
})

const roeDetailTitle = computed(() => {
  if (!roeDetailRow.value) return ''
  const name = roeDetailRow.value['Deal Name']
  return roeDetailMode.value === 'uw'
    ? `U/W ITD ROE Calculation — ${name}`
    : `ITD ROE Calculation — ${name}`
})

const roeDetailROE = computed(() => {
  if (!roeDetailRow.value) return 0
  return roeDetailMode.value === 'uw'
    ? roeDetailRow.value['U/W ITD ROE']
    : roeDetailRow.value['ITD ROE']
})

const roeDetailCF = computed(() => {
  if (!roeDetailRow.value) return 0
  return roeDetailMode.value === 'uw'
    ? roeDetailRow.value._uw_cf_total
    : roeDetailRow.value['CF Received']
})
</script>

<template>
  <div class="reports">
    <h2>Reports</h2>

    <div class="reports-layout">
      <!-- Left: report list + filters -->
      <div class="reports-sidebar">
        <!-- Report list -->
        <div class="section-label">Select Report</div>
        <div class="report-list">
          <div
            v-for="rt in reportDefs"
            :key="rt.value"
            :class="['report-item', { active: selectedReport === rt.value }]"
            @click="selectedReport = rt.value"
          >
            <div class="report-item-label">{{ rt.label }}</div>
            <div class="report-item-desc">{{ rt.description }}</div>
          </div>
        </div>

        <!-- Filters (hidden for custom view reports like Sold Portfolio) -->
        <template v-if="selectedReport && !isCustomView">
          <div class="section-label">Filters</div>

          <!-- Standard population selector (non-pref-detail reports) -->
          <template v-if="!isPrefDetail">
            <div class="filter-group">
              <label>Population</label>
              <select v-model="population">
                <option value="current">Current Deal</option>
                <option value="select">Select Deals</option>
                <option value="partner">By Partner</option>
                <option value="all">All Deals</option>
              </select>
            </div>

            <div v-if="population === 'partner'" class="filter-group">
              <label>Partner</label>
              <select v-model="selectedPartner">
                <option value="">-- Select partner --</option>
                <option v-for="p in partners" :key="p.partner" :value="p.partner">
                  {{ p.display || p.partner }} ({{ p.deal_count }} deals)
                </option>
              </select>
            </div>

            <!-- Deal picker inline -->
            <div v-if="population === 'select'" class="deal-picker">
              <div class="deal-picker-header">
                <span>{{ selectedVcodes.length }} / {{ eligibleDeals.length }}</span>
                <button class="btn-sm" @click="selectedVcodes = eligibleDeals.map(d => d.vcode)">All</button>
                <button class="btn-sm" @click="selectedVcodes = []">Clear</button>
              </div>
              <div class="deal-picker-list">
                <label v-for="d in eligibleDeals" :key="d.vcode" class="deal-checkbox">
                  <input type="checkbox" :checked="selectedVcodes.includes(d.vcode)" @change="toggleDeal(d.vcode)" />
                  {{ d.label }}
                </label>
              </div>
            </div>

            <div class="population-label">{{ populationLabel }}</div>
          </template>

          <!-- Deal + Investor selectors (pref balance detail) -->
          <template v-if="isPrefDetail">
            <div class="filter-group">
              <label>Deal</label>
              <select v-model="prefDealVcode">
                <option value="">-- Select deal --</option>
                <option v-for="d in eligibleDeals" :key="d.vcode" :value="d.vcode">
                  {{ d.label }}
                </option>
              </select>
            </div>

            <div v-if="prefDealVcode" class="filter-group">
              <label>Investor</label>
              <select v-model="prefInvestorId">
                <option value="">-- Select investor --</option>
                <option v-for="inv in prefInvestors" :key="inv.investor_id" :value="inv.investor_id">
                  {{ inv.investor_id }}
                </option>
              </select>
            </div>
          </template>

          <div v-if="activeReport?.hasReportDate" class="filter-group">
            <label>As of Date</label>
            <input type="date" v-model="reportDate" />
          </div>

          <button
            class="btn-generate"
            @click="generate"
            :disabled="loading || !canGenerate"
          >
            {{ loading ? 'Generating...' : 'Generate Report' }}
          </button>
        </template>
      </div>

      <!-- Right: results -->
      <div class="reports-main">
        <template v-if="!selectedReport">
          <p class="placeholder">Select a report from the list to get started.</p>
        </template>
        <template v-else-if="isCustomView">
          <SoldPortfolioView v-if="activeReport?.value === 'sold-portfolio'" :embedded="true" />
        </template>
        <template v-else>
          <div class="results-header">
            <h3>{{ activeReport?.label }}</h3>
            <button v-if="results.length" class="btn-download" @click="downloadExcel">
              Download Excel
            </button>
          </div>

          <!-- Pref Balance Header -->
          <div v-if="prefHeader && isPrefDetail" class="pref-header">
            <div class="pref-header-grid">
              <div class="pref-kv"><span class="pref-label">Property #:</span> {{ prefHeader.vcode }}</div>
              <div class="pref-kv"><span class="pref-label">As of:</span> {{ prefHeader.report_date }}</div>
              <div class="pref-kv"><span class="pref-label">Investment ID:</span> {{ prefHeader.investment_id }}</div>
              <div class="pref-kv"><span class="pref-label">Investment Balance:</span> {{ fmtCurr(prefHeader.investment_balance) }}</div>
              <div class="pref-kv"><span class="pref-label">Investor:</span> {{ prefHeader.investor_id }}</div>
              <div class="pref-kv"><span class="pref-label">Accrued:</span> {{ fmtCurr(prefHeader.accrued_pref) }}</div>
              <div class="pref-kv"><span class="pref-label">Pref Rate:</span> {{ fmtPct(prefHeader.pref_rate) }}</div>
              <div class="pref-kv"><span class="pref-label">Total:</span> <strong>{{ fmtCurr(prefHeader.total) }}</strong></div>
              <div class="pref-kv"></div>
              <div class="pref-kv"><span class="pref-label">Annual Pref Est:</span> {{ fmtCurr(prefHeader.annual_pref_est) }}</div>
            </div>
          </div>

          <!-- Errors -->
          <div v-if="errors.length > 0" class="error-section">
            <button class="btn-sm" @click="showErrors = !showErrors">
              {{ errors.length }} deal(s) skipped {{ showErrors ? '&#9662;' : '&#9656;' }}
            </button>
            <div v-if="showErrors" class="error-list">
              <p v-for="(e, i) in errors" :key="i">{{ e.deal_name || e.vcode }}: {{ e.error }}</p>
            </div>
          </div>

          <DataTable
            v-if="results.length"
            :columns="activeReport?.columns || []"
            :rows="results"
            :highlight-total="activeReport?.highlightTotal ?? false"
          />

          <!-- ROE Detail: event-by-event breakdown for single deal -->
          <template v-if="roeDetailRow">
            <div class="roe-detail-section">
              <div class="roe-detail-header">
                <h4>{{ roeDetailTitle }}</h4>
                <div v-if="hasUwDetail" class="roe-toggle">
                  <button
                    :class="['roe-toggle-btn', { active: roeDetailMode === 'actual' }]"
                    @click="roeDetailMode = 'actual'"
                  >ITD ROE</button>
                  <button
                    :class="['roe-toggle-btn', { active: roeDetailMode === 'uw' }]"
                    @click="roeDetailMode = 'uw'"
                  >U/W ITD ROE</button>
                </div>
              </div>
              <div class="roe-metrics">
                <div class="roe-metric">
                  <span class="roe-metric-label">Total Funded</span>
                  <span class="roe-metric-value">{{ fmtCurr(roeDetailRow['Total Funded']) }}</span>
                </div>
                <div class="roe-metric">
                  <span class="roe-metric-label">Return of Capital</span>
                  <span class="roe-metric-value">{{ fmtCurr(roeDetailRow['Return of Capital']) }}</span>
                </div>
                <div class="roe-metric">
                  <span class="roe-metric-label">Current Balance</span>
                  <span class="roe-metric-value">{{ fmtCurr(roeDetailRow['Current Balance']) }}</span>
                </div>
                <div class="roe-metric">
                  <span class="roe-metric-label">Wtd Avg Balance</span>
                  <span class="roe-metric-value">{{ fmtCurr(roeDetailRow['Wtd Avg Balance']) }}</span>
                </div>
                <div class="roe-metric">
                  <span class="roe-metric-label">{{ roeDetailMode === 'uw' ? 'U/W CF (7071)' : 'CF Received' }}</span>
                  <span class="roe-metric-value">{{ fmtCurr(roeDetailCF) }}</span>
                </div>
                <div class="roe-metric">
                  <span class="roe-metric-label">Days</span>
                  <span class="roe-metric-value">{{ roeDetailRow._total_days?.toLocaleString() }}</span>
                </div>
                <div class="roe-metric">
                  <span class="roe-metric-label">Years</span>
                  <span class="roe-metric-value">{{ roeDetailRow._years?.toFixed(4) }}</span>
                </div>
                <div class="roe-metric highlight">
                  <span class="roe-metric-label">{{ roeDetailMode === 'uw' ? 'U/W ITD ROE' : 'ITD ROE' }}</span>
                  <span class="roe-metric-value">{{ fmtPct(roeDetailROE) }}</span>
                </div>
              </div>
              <DataTable
                :columns="roeDetailColumns"
                :rows="roeDetailRows"
              />
            </div>
          </template>

          <p v-else-if="!results.length && !loading" class="placeholder">
            {{ isPrefDetail ? 'Select a deal and investor, then click Generate Report.' : 'Set filters and click Generate Report.' }}
          </p>
          <p v-if="loading" class="placeholder">Generating report...</p>
        </template>
      </div>
    </div>
  </div>
</template>

<style scoped>
.reports { padding: 0 0 40px 0; }
h2 { font-size: 20px; margin-bottom: 16px; }

.reports-layout {
  display: flex;
  gap: 24px;
  align-items: flex-start;
}

/* --- Sidebar --- */
.reports-sidebar {
  width: 280px;
  flex-shrink: 0;
}

.section-label {
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--color-text-secondary);
  margin-bottom: 6px;
  margin-top: 16px;
}

.section-label:first-child { margin-top: 0; }

.report-list {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.report-item {
  padding: 10px 12px;
  border-radius: 6px;
  cursor: pointer;
  border: 1px solid transparent;
  transition: background 0.12s;
}

.report-item:hover { background: #f5f5f5; }

.report-item.active {
  background: var(--color-accent);
  color: white;
  border-color: var(--color-accent);
}

.report-item.active .report-item-desc { color: rgba(255,255,255,0.8); }

.report-item-label {
  font-size: 13px;
  font-weight: 600;
}

.report-item-desc {
  font-size: 11px;
  color: var(--color-text-secondary);
  margin-top: 2px;
  line-height: 1.3;
}

/* --- Filters --- */
.filter-group {
  margin-top: 8px;
}

.filter-group label {
  display: block;
  font-size: 12px;
  font-weight: 600;
  margin-bottom: 3px;
}

.filter-group select,
.filter-group input {
  width: 100%;
  padding: 7px 10px;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  font-size: 13px;
  box-sizing: border-box;
}

.population-label {
  font-size: 11px;
  color: var(--color-text-secondary);
  font-style: italic;
  margin-top: 8px;
}

.btn-generate {
  width: 100%;
  margin-top: 12px;
  padding: 10px 20px;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
  font-weight: 600;
  background: var(--color-accent);
  color: white;
}

.btn-generate:hover:not(:disabled) { background: #3a63ad; }
.btn-generate:disabled { opacity: 0.6; cursor: not-allowed; }

/* --- Deal picker --- */
.deal-picker {
  margin-top: 8px;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  padding: 8px;
}

.deal-picker-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
  font-size: 11px;
  color: var(--color-text-secondary);
}

.deal-picker-list {
  max-height: 180px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.deal-checkbox {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  cursor: pointer;
  padding: 2px 4px;
  border-radius: 3px;
}

.deal-checkbox:hover { background: #f5f5f5; }
.deal-checkbox input { cursor: pointer; }

.btn-sm {
  padding: 2px 8px;
  border: 1px solid var(--color-border);
  background: var(--color-surface);
  border-radius: 4px;
  cursor: pointer;
  font-size: 11px;
}

.btn-sm:hover { background: #eee; }

/* --- Pref Balance Header --- */
.pref-header {
  background: #f8f9fa;
  border: 1px solid var(--color-border);
  border-radius: 8px;
  padding: 16px 20px;
  margin-bottom: 16px;
}

.pref-header-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px 32px;
  font-size: 13px;
}

.pref-label {
  font-weight: 600;
  color: var(--color-text-secondary);
  margin-right: 6px;
}

/* --- Main results --- */
.reports-main {
  flex: 1;
  min-width: 0;
}

.results-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}

.results-header h3 {
  font-size: 16px;
  margin: 0;
}

.btn-download {
  padding: 7px 16px;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
  background: var(--color-pref);
  color: white;
}

/* Errors */
.error-section { margin-bottom: 12px; }
.error-list {
  margin-top: 4px;
  padding: 8px 12px;
  background: #fff8e1;
  border: 1px solid #ffe082;
  border-radius: 4px;
  font-size: 12px;
  color: #856404;
}
.error-list p { margin: 2px 0; }

.placeholder {
  color: var(--color-text-secondary);
  font-style: italic;
  text-align: center;
  padding: 40px 0;
}

/* --- ROE Detail --- */
.roe-detail-section {
  margin-top: 24px;
  border-top: 2px solid var(--color-border);
  padding-top: 16px;
}

.roe-detail-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}

.roe-detail-header h4 {
  font-size: 14px;
  margin: 0;
}

.roe-toggle {
  display: flex;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  overflow: hidden;
}

.roe-toggle-btn {
  padding: 5px 14px;
  border: none;
  background: var(--color-surface);
  cursor: pointer;
  font-size: 12px;
  font-weight: 600;
  color: var(--color-text-secondary);
}

.roe-toggle-btn + .roe-toggle-btn {
  border-left: 1px solid var(--color-border);
}

.roe-toggle-btn.active {
  background: var(--color-accent);
  color: white;
}

.roe-toggle-btn:hover:not(.active) {
  background: #eee;
}

.roe-metrics {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 16px;
}

.roe-metric {
  background: #f8f9fa;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  padding: 8px 14px;
  min-width: 120px;
}

.roe-metric.highlight {
  background: var(--color-accent);
  border-color: var(--color-accent);
}

.roe-metric.highlight .roe-metric-label,
.roe-metric.highlight .roe-metric-value {
  color: white;
}

.roe-metric-label {
  display: block;
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.3px;
  color: var(--color-text-secondary);
  margin-bottom: 2px;
}

.roe-metric-value {
  font-size: 14px;
  font-weight: 700;
}
</style>
