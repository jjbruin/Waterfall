<script setup lang="ts">
import { ref, onMounted, computed, reactive } from 'vue'
import DataTable from '../components/common/DataTable.vue'
import ProgressOverlay from '../components/common/ProgressOverlay.vue'
import api from '../api/client'
import { useDataStore } from '../stores/data'

const dataStore = useDataStore()

// State
const entities = ref<any[]>([])
const selectedEntity = ref('')
const mode = ref<'actual' | 'proposed'>('actual')
const loading = ref(false)
const computing = ref(false)
const results = ref<any>(null)

// Deals for selected entity
const deals = ref<any[]>([])
const investors = ref<any[]>([])
const dealsLoading = ref(false)

// Deal detail drill-down
const selectedDealVcode = ref('')
const dealDetail = ref<any>(null)
const dealDetailLoading = ref(false)

// Proposed assumptions
const assumptions = reactive({
  am_fee_pct: 1.0,
  hurdle_rate: 8.0,
  promote_pct: 20,
  annual_expenses: 50000,
})

// Load entities on mount
onMounted(async () => {
  loading.value = true
  try {
    const res = await api.get('/api/portfolio-analysis/entities')
    entities.value = res.data.entities || []
  } catch (e: any) {
    dataStore.addToast('Failed to load entities: ' + (e.response?.data?.error || e.message), 'error')
  } finally {
    loading.value = false
  }
})

// Load deals when entity changes
async function onEntityChange() {
  deals.value = []
  investors.value = []
  results.value = null
  dealDetail.value = null
  selectedDealVcode.value = ''
  if (!selectedEntity.value) return
  dealsLoading.value = true
  try {
    const res = await api.get(`/api/portfolio-analysis/entities/${selectedEntity.value}/deals`)
    deals.value = res.data.deals || []
    investors.value = res.data.investors || []
  } catch (e: any) {
    dataStore.addToast('Failed to load deals: ' + (e.response?.data?.error || e.message), 'error')
  } finally {
    dealsLoading.value = false
  }
}

// Build request body
function buildRequestBody() {
  const body: any = { mode: mode.value }
  if (mode.value === 'proposed') {
    body.assumptions = {
      am_fee_pct: assumptions.am_fee_pct / 100,
      hurdle_rate: assumptions.hurdle_rate / 100,
      promote_pct: assumptions.promote_pct / 100,
      annual_expenses: assumptions.annual_expenses,
    }
  }
  return body
}

// Compute
async function compute() {
  if (!selectedEntity.value) return
  computing.value = true
  results.value = null
  dealDetail.value = null
  selectedDealVcode.value = ''
  try {
    const res = await api.post(
      `/api/portfolio-analysis/entities/${selectedEntity.value}/compute`,
      buildRequestBody()
    )
    results.value = res.data
  } catch (e: any) {
    dataStore.addToast('Computation failed: ' + (e.response?.data?.error || e.message), 'error')
  } finally {
    computing.value = false
  }
}

// Deal detail drill-down
async function loadDealDetail() {
  dealDetail.value = null
  if (!selectedDealVcode.value || !selectedEntity.value) return
  dealDetailLoading.value = true
  try {
    const res = await api.get(
      `/api/portfolio-analysis/entities/${selectedEntity.value}/deals/${selectedDealVcode.value}/detail`
    )
    dealDetail.value = res.data
  } catch (e: any) {
    dataStore.addToast('Failed to load deal detail: ' + (e.response?.data?.error || e.message), 'error')
  } finally {
    dealDetailLoading.value = false
  }
}

// Excel downloads
async function downloadSummaryExcel() {
  try {
    const res = await api.post(
      `/api/portfolio-analysis/entities/${selectedEntity.value}/excel`,
      buildRequestBody(),
      { responseType: 'blob' }
    )
    const url = URL.createObjectURL(new Blob([res.data]))
    const a = document.createElement('a')
    a.href = url
    const name = entityName.value.replace(/\s+/g, '_')
    a.download = `portfolio_analysis_${name}_${mode.value}.xlsx`
    a.click()
    URL.revokeObjectURL(url)
  } catch (e: any) {
    dataStore.addToast('Failed to download Excel: ' + (e.response?.data?.error || e.message), 'error')
  }
}

// Computed
const entityName = computed(() => {
  const e = entities.value.find((x: any) => x.entity_id === selectedEntity.value)
  return e ? (e.name || e.entity_id) : selectedEntity.value
})

const dealReturns = computed(() => results.value?.deal_returns || [])
const partnerResults = computed(() => results.value?.partner_results || [])
const dealSummary = computed(() => results.value?.deal_summary || {})
const prefEquitySummary = computed(() => results.value?.pref_equity_summary || {})
const incomeSchedule = computed(() => results.value?.income_schedule || [])
const waterfallDetail = computed(() => results.value?.waterfall_detail || [])
const computeErrors = computed(() => results.value?.errors || [])

const allocationTable = computed(() => results.value?.allocation_table || {})
const cfAllocTable = computed(() => allocationTable.value.cf || { rows: [], years: [] })
const capAllocTable = computed(() => allocationTable.value.cap || { rows: [], years: [] })

// Summary table rows (deal pref equity returns + portfolio total)
const summaryRows = computed(() => {
  const rows = dealReturns.value.map((dr: any) => ({ ...dr, _is_deal_total: false }))
  const ps = prefEquitySummary.value
  rows.push({
    name: `${entityName.value} — Pref Equity Total`,
    pref_partner: '',
    vcode: '',
    asset_type: '',
    computed: true,
    contributions: ps.total_contributions || 0,
    distributions: ps.total_distributions || 0,
    irr: ps.irr,
    roe: null,
    moic: ps.moic || 0,
    _is_deal_total: true,
  })
  return rows
})

// Deal names for drill-down dropdown (only computed deals)
const computedDealNames = computed(() =>
  dealReturns.value
    .filter((d: any) => d.computed)
    .map((d: any) => ({ vcode: d.vcode, name: d.name }))
)

const selectedDealName = computed(() => {
  const d = computedDealNames.value.find((n: any) => n.vcode === selectedDealVcode.value)
  return d ? d.name : ''
})

// Detail partner results table
const detailPartnerColumns = [
  { key: 'partner', label: 'Partner' },
  { key: 'contributions', label: 'Contributions', format: 'currency', align: 'right' },
  { key: 'cf_distributions', label: 'CF Distributions', format: 'currency', align: 'right' },
  { key: 'cap_distributions', label: 'Cap Distributions', format: 'currency', align: 'right' },
  { key: 'total_distributions', label: 'Total Distributions', format: 'currency', align: 'right' },
  { key: 'irr', label: 'IRR', format: 'percent', align: 'right' },
  { key: 'roe', label: 'ROE', format: 'percent', align: 'right' },
  { key: 'moic', label: 'MOIC', format: 'multiple', align: 'right' },
]

// Income summary grouped by Source Deal
const incomeSummary = computed(() => {
  const groups: Record<string, any> = {}
  for (const row of incomeSchedule.value) {
    const key = `${row['Source Deal']}|${row.Type}`
    if (!groups[key]) {
      groups[key] = { 'Source Deal': row['Source Deal'], Type: row.Type, Amount: 0 }
    }
    groups[key].Amount += row.Amount
  }
  return Object.values(groups)
})

// Expandable sections
const sections = reactive({
  partners: false,
  income: false,
  allocations: false,
  waterfall: false,
  xirr: false,
})

// Column defs
const waterfallDetailColumns = [
  { key: 'Date', label: 'Date' },
  { key: 'Event', label: 'Event' },
  { key: 'Type', label: 'Type' },
  { key: 'Gross', label: 'Gross', format: 'currency', align: 'right' },
  { key: 'AM Fee', label: 'AM Fee', format: 'currency', align: 'right' },
  { key: 'Expenses', label: 'Expenses', format: 'currency', align: 'right' },
  { key: 'Pref Paid', label: 'Pref Paid', format: 'currency', align: 'right' },
  { key: 'Capital Returned', label: 'Capital Returned', format: 'currency', align: 'right' },
  { key: 'Excess', label: 'Excess', format: 'currency', align: 'right' },
  { key: 'Promote', label: 'Promote', format: 'currency', align: 'right' },
  { key: 'Net Available', label: 'Net to Investors', format: 'currency', align: 'right' },
  { key: 'Capital Balance', label: 'Capital Balance', format: 'currency', align: 'right' },
  { key: 'Pref Balance', label: 'Pref Balance', format: 'currency', align: 'right' },
]

const incomeSummaryColumns = [
  { key: 'Source Deal', label: 'Source Deal' },
  { key: 'Type', label: 'Type' },
  { key: 'Amount', label: 'Amount', format: 'currency2', align: 'right' },
]

const incomeDetailColumns = [
  { key: 'Date', label: 'Date' },
  { key: 'Source Entity', label: 'Source Entity' },
  { key: 'Source Deal', label: 'Source Deal' },
  { key: 'Type', label: 'Type' },
  { key: 'vState', label: 'vState' },
  { key: 'Amount', label: 'Amount', format: 'currency2', align: 'right' },
]

// Merged XIRR Cash Flows (partners as columns, dates down rows)
const xirrMerged = computed(() => {
  const prs = partnerResults.value
  if (!prs.length) return null

  const partners = prs.map((p: any) => p.partner)

  // Build map keyed by "date|type" -> { date, type, partner1: amt, ..., _total: amt }
  const rowMap = new Map<string, Record<string, any>>()
  for (const pr of prs) {
    for (const cf of (pr.combined_cashflows || [])) {
      const desc = cf.amount < 0 ? 'Contribution' : 'Distribution'
      const key = `${cf.date}|${desc}`
      if (!rowMap.has(key)) {
        rowMap.set(key, { date: cf.date, type: desc, _total: 0 })
      }
      const row = rowMap.get(key)!
      row[pr.partner] = (row[pr.partner] || 0) + cf.amount
      row._total += cf.amount
    }
  }

  const rows = [...rowMap.values()].sort((a, b) => a.date.localeCompare(b.date))

  const columns = [
    { key: 'date', label: 'Date' },
    { key: 'type', label: 'Type' },
    ...partners.map((p: string) => ({ key: p, label: p, format: 'currency', align: 'right' })),
    { key: '_total', label: 'Total', format: 'currency', align: 'right' },
  ]

  return { columns, rows }
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
function fmtInt(v: any): string {
  if (v == null || v !== v) return ''
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(v)
}
</script>

<template>
  <div class="portfolio-analysis">
    <h2>Portfolio Analysis</h2>
    <p class="subtitle">Upstream waterfall analysis — traces deal cash flows through to portfolio entity investors.</p>

    <ProgressOverlay :visible="computing" message="Running portfolio computation..." />

    <!-- Controls -->
    <div class="section controls-section">
      <div class="controls-row">
        <div class="control-group">
          <label>Portfolio Entity</label>
          <select v-model="selectedEntity" @change="onEntityChange" :disabled="loading">
            <option value="">-- Select Entity --</option>
            <option v-for="e in entities" :key="e.entity_id" :value="e.entity_id">
              {{ e.name || e.entity_id }} ({{ e.deal_count }} deals)
            </option>
          </select>
        </div>
        <div class="control-group">
          <label>Waterfall Mode</label>
          <select v-model="mode">
            <option value="actual">Actual (Saved Waterfalls)</option>
            <option value="proposed">Proposed (Simplified Assumptions)</option>
          </select>
        </div>
        <div class="control-group" style="justify-content: flex-end">
          <button
            class="btn-compute"
            @click="compute"
            :disabled="computing || !selectedEntity || !deals.length"
          >
            {{ computing ? 'Computing...' : `Compute ${mode === 'actual' ? 'Actual' : 'Proposed'} Returns` }}
          </button>
        </div>
      </div>

      <!-- Proposed Assumptions Panel -->
      <div v-if="mode === 'proposed'" class="assumptions-panel">
        <div class="assumptions-row">
          <div class="assumption-field">
            <label>AM Fee %</label>
            <input type="number" v-model.number="assumptions.am_fee_pct" step="0.25" min="0" max="10" />
          </div>
          <div class="assumption-field">
            <label>Hurdle Rate %</label>
            <input type="number" v-model.number="assumptions.hurdle_rate" step="0.5" min="0" max="25" />
          </div>
          <div class="assumption-field">
            <label>Promote %</label>
            <input type="number" v-model.number="assumptions.promote_pct" step="5" min="0" max="100" />
          </div>
          <div class="assumption-field">
            <label>Annual Expenses ($)</label>
            <input type="number" v-model.number="assumptions.annual_expenses" step="10000" min="0" />
          </div>
        </div>
        <p class="assumptions-note">
          Proposed assumptions apply only at the {{ entityName || 'entity' }} level.
          All lower-tier waterfalls (deals and PPIs) use their actual saved waterfalls.
        </p>
      </div>
    </div>

    <!-- Errors -->
    <div v-if="computeErrors.length" class="error-section">
      <div v-for="(err, i) in computeErrors" :key="i" class="error-msg">{{ err }}</div>
    </div>

    <!-- Results -->
    <template v-if="results">

      <!-- Toolbar -->
      <div class="toolbar">
        <button class="btn-download" @click="downloadSummaryExcel">
          Download Summary Excel
        </button>
        <span class="compute-status">
          {{ results.deals_computed }} of {{ dealReturns.length }} deals computed ({{ results.mode }} mode)
        </span>
      </div>

      <!-- Pref Equity Returns Summary Table (like Sold Portfolio) -->
      <div class="summary-table-wrapper">
        <table class="summary-table">
          <thead>
            <tr>
              <th>Investment Name</th>
              <th>Pref Equity Partner</th>
              <th class="right">Contributions</th>
              <th class="right">Distributions</th>
              <th class="right">IRR</th>
              <th class="right">ROE</th>
              <th class="right">MOIC</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(row, idx) in summaryRows" :key="idx"
                :class="{ 'total-row': row._is_deal_total, 'no-data': !row.computed && !row._is_deal_total }">
              <td>{{ row.name }}</td>
              <td>{{ row.pref_partner || '' }}</td>
              <td class="right">{{ row.computed || row._is_deal_total ? fmtCur(row.contributions) : '--' }}</td>
              <td class="right">{{ row.computed || row._is_deal_total ? fmtCur(row.distributions) : '--' }}</td>
              <td class="right">{{ row.computed || row._is_deal_total ? fmtPct(row.irr) : '--' }}</td>
              <td class="right">{{ row.computed || row._is_deal_total ? fmtPct(row.roe) : '--' }}</td>
              <td class="right">{{ row.computed || row._is_deal_total ? fmtMult(row.moic) : '--' }}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Deal Detail Drill-Down -->
      <div class="detail-section">
        <h3>Deal Detail</h3>
        <div class="detail-controls">
          <select v-model="selectedDealVcode" @change="loadDealDetail" class="deal-select">
            <option value="">-- Select a deal to view partner returns --</option>
            <option v-for="d in computedDealNames" :key="d.vcode" :value="d.vcode">
              {{ d.name }}
            </option>
          </select>
        </div>

        <div v-if="dealDetailLoading" class="placeholder">Loading deal detail...</div>

        <template v-if="dealDetail">
          <!-- Deal partner returns -->
          <DataTable :columns="detailPartnerColumns" :rows="dealDetail.partner_results" />

          <!-- Deal summary metrics -->
          <div class="metric-cards">
            <div class="metric-card">
              <span class="metric-label">Deal IRR</span>
              <span class="metric-value">{{ fmtPct(dealDetail.deal_summary?.deal_irr) }}</span>
            </div>
            <div class="metric-card">
              <span class="metric-label">Deal ROE</span>
              <span class="metric-value">{{ fmtPct(dealDetail.deal_summary?.deal_roe) }}</span>
            </div>
            <div class="metric-card">
              <span class="metric-label">Deal MOIC</span>
              <span class="metric-value">{{ fmtMult(dealDetail.deal_summary?.deal_moic) }}</span>
            </div>
            <div class="metric-card">
              <span class="metric-label">Contributions</span>
              <span class="metric-value">{{ fmtCur(dealDetail.deal_summary?.total_contributions) }}</span>
            </div>
            <div class="metric-card">
              <span class="metric-label">Distributions</span>
              <span class="metric-value">{{ fmtCur(dealDetail.deal_summary?.total_distributions) }}</span>
            </div>
          </div>
        </template>

        <p v-else-if="selectedDealVcode && !dealDetailLoading" class="placeholder">
          No data available for this deal.
        </p>
      </div>

      <!-- Entity Waterfall Allocations (actual mode) -->
      <template v-if="results.mode === 'actual'">
        <div class="section expandable" @click="sections.allocations = !sections.allocations">
          <h3 class="expand-header">
            <span class="expand-icon">{{ sections.allocations ? '▾' : '▸' }}</span>
            Entity Waterfall Allocations — {{ entityName }}
          </h3>
        </div>
        <div v-if="sections.allocations" class="section-body">
          <template v-if="cfAllocTable.rows.length || capAllocTable.rows.length">
            <template v-if="cfAllocTable.rows.length">
              <h4>CF Waterfall</h4>
              <div class="forecast-table-wrapper">
                <table class="forecast-table">
                  <thead>
                    <tr>
                      <th class="label-col">Step</th>
                      <th v-for="y in cfAllocTable.years" :key="y" class="year-col">{{ y }}</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="(row, i) in cfAllocTable.rows" :key="'cf'+i"
                        :class="{
                          'section-header-row': row.label.endsWith(':'),
                          'blank-row': row.label.trim() === '',
                          'topline-row': row.label.trim() === 'Total Distributions',
                        }">
                      <td class="label-col">{{ row.label }}</td>
                      <td v-for="y in cfAllocTable.years" :key="y" class="year-col">
                        {{ (row.label.trim() === '' || row.label.endsWith(':')) ? '' : row.values[String(y)] != null ? fmtInt(row.values[String(y)]) : '' }}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </template>
            <template v-if="capAllocTable.rows.length">
              <h4>Capital Waterfall</h4>
              <div class="forecast-table-wrapper">
                <table class="forecast-table">
                  <thead>
                    <tr>
                      <th class="label-col">Step</th>
                      <th v-for="y in capAllocTable.years" :key="y" class="year-col">{{ y }}</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="(row, i) in capAllocTable.rows" :key="'cap'+i"
                        :class="{
                          'section-header-row': row.label.endsWith(':'),
                          'blank-row': row.label.trim() === '',
                          'topline-row': row.label.trim() === 'Total Distributions',
                        }">
                      <td class="label-col">{{ row.label }}</td>
                      <td v-for="y in capAllocTable.years" :key="y" class="year-col">
                        {{ (row.label.trim() === '' || row.label.endsWith(':')) ? '' : row.values[String(y)] != null ? fmtInt(row.values[String(y)]) : '' }}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </template>
          </template>
          <p v-else class="placeholder">No member allocations found.</p>
        </div>
      </template>

      <!-- Proposed Waterfall Detail (proposed mode) -->
      <template v-if="results.mode === 'proposed' && waterfallDetail.length">
        <div class="section expandable" @click="sections.waterfall = !sections.waterfall">
          <h3 class="expand-header">
            <span class="expand-icon">{{ sections.waterfall ? '▾' : '▸' }}</span>
            Proposed Waterfall Detail — {{ entityName }}
          </h3>
        </div>
        <div v-if="sections.waterfall" class="section-body">
          <DataTable :columns="waterfallDetailColumns" :rows="waterfallDetail" />
          <div class="assumptions-footnote">
            Assumptions: AM Fee {{ assumptions.am_fee_pct }}%,
            Hurdle {{ assumptions.hurdle_rate }}%,
            Promote {{ assumptions.promote_pct }}%,
            Annual Expenses ${{ assumptions.annual_expenses.toLocaleString() }}
          </div>
        </div>
      </template>

      <!-- Partner Returns (expandable) -->
      <div class="section expandable" @click="sections.partners = !sections.partners">
        <h3 class="expand-header">
          <span class="expand-icon">{{ sections.partners ? '▾' : '▸' }}</span>
          Entity Partner Returns — {{ entityName }} Investors
        </h3>
      </div>
      <div v-if="sections.partners" class="section-body">
        <template v-if="partnerResults.length">
          <div class="kpi-row">
            <div v-for="pr in partnerResults" :key="pr.partner" class="kpi-group">
              <div class="kpi-group-title">{{ pr.partner }}</div>
              <div class="kpi-cards">
                <div class="mini-kpi">
                  <span class="mini-label">IRR</span>
                  <span class="mini-value">{{ fmtPct(pr.irr) }}</span>
                </div>
                <div class="mini-kpi">
                  <span class="mini-label">MOIC</span>
                  <span class="mini-value">{{ fmtMult(pr.moic) }}</span>
                </div>
                <div class="mini-kpi">
                  <span class="mini-label">Distributions</span>
                  <span class="mini-value">{{ fmtCur(pr.total_distributions) }}</span>
                </div>
              </div>
            </div>
          </div>
        </template>
        <p v-else class="placeholder">No partner returns available.</p>
      </div>

      <!-- Income Schedule (expandable — actual mode only) -->
      <template v-if="results.mode === 'actual'">
        <div class="section expandable" @click="sections.income = !sections.income">
          <h3 class="expand-header">
            <span class="expand-icon">{{ sections.income ? '▾' : '▸' }}</span>
            Income Schedule — Cash Arriving at {{ entityName }}
          </h3>
        </div>
        <div v-if="sections.income" class="section-body">
          <template v-if="incomeSummary.length">
            <h4>Summary by Source Deal</h4>
            <DataTable :columns="incomeSummaryColumns" :rows="incomeSummary" />
            <h4>Full Detail</h4>
            <DataTable :columns="incomeDetailColumns" :rows="incomeSchedule" />
          </template>
          <p v-else class="placeholder">No income allocated to this entity.</p>
        </div>
      </template>

      <!-- XIRR Cash Flows (expandable) -->
      <div class="section expandable" @click="sections.xirr = !sections.xirr">
        <h3 class="expand-header">
          <span class="expand-icon">{{ sections.xirr ? '▾' : '▸' }}</span>
          XIRR Cash Flows — Entity Investors
        </h3>
      </div>
      <div v-if="sections.xirr" class="section-body">
        <template v-if="xirrMerged">
          <DataTable :columns="xirrMerged.columns" :rows="xirrMerged.rows" />
        </template>
        <p v-else class="placeholder">No cashflow data available.</p>
      </div>

    </template>
  </div>
</template>

<style scoped>
.portfolio-analysis { padding: 0 0 40px 0; }
h2 { font-size: 20px; margin-bottom: 4px; }
h3 { font-size: 15px; margin: 0; }
h4 { font-size: 13px; margin: 16px 0 8px 0; font-weight: 600; }
.subtitle { font-size: 13px; color: var(--color-text-secondary); margin-bottom: 16px; }

.section {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 12px;
}

.section-body {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-top: none;
  border-radius: 0 0 8px 8px;
  padding: 16px;
  margin-top: -12px;
  margin-bottom: 12px;
}

.expandable { cursor: pointer; }
.expandable:hover { background: #f8f9fa; }

.expand-header {
  display: flex;
  align-items: center;
  gap: 8px;
  user-select: none;
}

.expand-icon { font-size: 12px; width: 16px; }

.placeholder {
  color: var(--color-text-secondary);
  font-style: italic;
  text-align: center;
  padding: 20px 0;
}

.placeholder-sm {
  color: var(--color-text-secondary);
  font-style: italic;
  font-size: 13px;
  padding: 8px 0;
}

/* Controls */
.controls-section { padding: 16px 20px; }

.controls-row {
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
  align-items: flex-end;
}

.control-group {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 220px;
}

.control-group label {
  font-size: 12px;
  font-weight: 600;
  color: var(--color-text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.3px;
}

.control-group select {
  padding: 6px 10px;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  font-size: 14px;
  background: white;
}

/* Assumptions */
.assumptions-panel {
  margin-top: 14px;
  padding-top: 14px;
  border-top: 1px solid var(--color-border);
}

.assumptions-row {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}

.assumption-field {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 140px;
}

.assumption-field label {
  font-size: 11px;
  font-weight: 600;
  color: var(--color-text-secondary);
}

.assumption-field input {
  padding: 5px 8px;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  font-size: 13px;
  width: 120px;
}

.assumptions-note {
  font-size: 12px;
  color: var(--color-text-secondary);
  font-style: italic;
  margin-top: 10px;
}

.assumptions-footnote {
  font-size: 12px;
  color: var(--color-text-secondary);
  margin-top: 12px;
  padding: 8px 0;
  border-top: 1px solid var(--color-border);
}

/* Toolbar */
.toolbar {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}

.compute-status {
  font-size: 13px;
  color: var(--color-text-secondary);
}

.btn-compute {
  padding: 8px 20px;
  background: var(--color-accent, #4472C4);
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
  white-space: nowrap;
}
.btn-compute:hover { opacity: 0.9; }
.btn-compute:disabled { opacity: 0.6; cursor: not-allowed; }

.btn-download {
  padding: 8px 20px;
  background: var(--color-pref, #548235);
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
}
.btn-download:hover { opacity: 0.9; }

.error-section { margin-bottom: 12px; }
.error-msg {
  color: #d32f2f;
  background: #ffebee;
  padding: 6px 12px;
  border-radius: 6px;
  font-size: 13px;
  margin-bottom: 4px;
}

/* Summary Table (Sold Portfolio style) */
.summary-table-wrapper {
  overflow-x: auto;
  margin-bottom: 16px;
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

.total-row td {
  font-weight: 700;
  border-top: 2px solid #333;
}

.no-data td {
  color: var(--color-text-secondary);
}

/* Deal Detail Section */
.detail-section {
  margin-bottom: 16px;
  padding: 16px;
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: 8px;
}

.detail-controls {
  margin-bottom: 12px;
}

.deal-select {
  padding: 8px 12px;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  font-size: 14px;
  min-width: 400px;
}

/* Metric Cards */
.metric-cards {
  display: flex;
  gap: 16px;
  margin: 16px 0 0 0;
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
  font-size: 16px;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
}

/* KPI Row */
.kpi-row {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}

.kpi-group {
  background: #f8f9fa;
  border: 1px solid var(--color-border);
  border-radius: 8px;
  padding: 12px 16px;
  min-width: 160px;
}

.kpi-group-title {
  font-size: 13px;
  font-weight: 600;
  margin-bottom: 8px;
}

.kpi-cards { display: flex; flex-direction: column; gap: 4px; }

.mini-kpi { display: flex; justify-content: space-between; gap: 12px; }

.mini-label {
  font-size: 11px;
  color: var(--color-text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.3px;
}

.mini-value {
  font-size: 14px;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}

.cf-member-section { margin-bottom: 16px; }

/* Forecast / Allocation Table (yearly pivot) */
.forecast-table-wrapper { overflow-x: auto; }

.forecast-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
  white-space: nowrap;
}

.forecast-table th,
.forecast-table td {
  padding: 6px 12px;
  border-bottom: 1px solid var(--color-border);
}

.forecast-table th {
  background: #f1f3f5;
  font-weight: 600;
  font-size: 12px;
}

.forecast-table .label-col {
  text-align: left;
  min-width: 240px;
  font-family: monospace;
  font-size: 12px;
}

.forecast-table .year-col {
  text-align: right;
  font-variant-numeric: tabular-nums;
  min-width: 90px;
}

.forecast-table .section-header-row td {
  font-weight: 700;
  background: #f0f4f8;
  font-family: inherit;
}

.forecast-table .blank-row td {
  height: 8px;
  padding: 0;
  border: none;
}

.forecast-table .topline-row td {
  border-top: 2px solid #333;
  font-weight: 700;
}
</style>
