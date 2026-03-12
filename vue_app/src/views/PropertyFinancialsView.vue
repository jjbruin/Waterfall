<script setup lang="ts">
import { onMounted, ref, watch, computed } from 'vue'
import { useDataStore } from '../stores/data'
import { useDealsStore } from '../stores/deals'
import KpiCard from '../components/common/KpiCard.vue'
import DualAxisChart from '../components/charts/DualAxisChart.vue'
import api from '../api/client'

const data = useDataStore()
const deals = useDealsStore()

onMounted(async () => {
  if (data.deals.length === 0) await data.loadDeals()
})

// ============================================================
// Deal selection
// ============================================================
const loading = ref(false)
const error = ref<string | null>(null)

async function onDealSelect(event: Event) {
  const vcode = (event.target as HTMLSelectElement).value
  if (!vcode) return
  deals.selectDeal(vcode)
  // Reset all sections
  chartData.value = null
  isData.value = null
  bsData.value = null
  tenantData.value = null
  onePagerData.value = null
  expanded.value = {}
  // Auto-load performance chart
  await loadChart(vcode)
}

// ============================================================
// Expandable sections
// ============================================================
const expanded = ref<Record<string, boolean>>({})
function toggle(section: string) {
  expanded.value[section] = !expanded.value[section]
  const vc = deals.currentVcode
  if (expanded.value[section] && vc) {
    if (section === 'is' && !isData.value) loadIS(vc)
    if (section === 'bs' && !bsData.value) loadBS(vc)
    if (section === 'tenants' && !tenantData.value) loadTenants(vc)
    if (section === 'onepager' && !onePagerData.value) loadOnePager(vc)
  }
}

watch(() => deals.currentVcode, () => {
  expanded.value = {}
  chartData.value = null
  isData.value = null
  isLeftAsOfDate.value = ''
  isRightAsOfDate.value = ''
  bsData.value = null
  tenantData.value = null
  onePagerData.value = null
})

// ============================================================
// Performance Chart
// ============================================================
interface ChartResponse {
  periods: string[]
  actual_noi: (number | null)[]
  uw_noi: (number | null)[]
  occupancy: (number | null)[]
  frequency: string
  available_period_ends: string[]
}
const chartData = ref<ChartResponse | null>(null)
const chartFreq = ref('Quarterly')
const chartPeriods = ref(12)
const chartLoading = ref(false)

async function loadChart(vcode: string) {
  chartLoading.value = true
  try {
    const res = await api.get(`/api/financials/${vcode}/performance-chart`, {
      params: { freq: chartFreq.value, periods: chartPeriods.value },
    })
    chartData.value = res.data
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    chartLoading.value = false
  }
}

async function refreshChart() {
  if (deals.currentVcode) await loadChart(deals.currentVcode)
}

const chartBarData = computed(() => ({
  name: 'Occupancy',
  // Occ% is already 0–100; do NOT multiply by 100
  data: (chartData.value?.occupancy || []).map(v => v != null ? +v.toFixed(1) : 0),
  color: '#93c5fd',
}))

const chartLineData = computed(() => [
  {
    name: 'Actual NOI',
    // Convert to thousands for cleaner axis labels
    data: (chartData.value?.actual_noi || []).map(v => v != null ? +(v / 1000).toFixed(1) : 0),
    color: '#1F4E79',
  },
  {
    name: 'U/W NOI',
    data: (chartData.value?.uw_noi || []).map(v => v != null ? +(v / 1000).toFixed(1) : 0),
    color: '#ED7D31',
  },
])

// ============================================================
// Income Statement
// ============================================================
interface ISRow {
  account: string
  level: number
  is_header?: boolean
  is_total?: boolean
  is_calc?: boolean
  left: number | null
  right: number | null
  var_usd: number | null
  var_pct: number | null
}
interface ISResponse {
  rows: ISRow[]
  available_dates: Record<string, string[]>
  left_label: string
  right_label: string
}
const isData = ref<ISResponse | null>(null)
const isLeftSource = ref('Actual')
const isLeftPeriod = ref('TTM')
const isRightSource = ref('Budget')
const isRightPeriod = ref('YTD')
const isLeftAsOfDate = ref('')
const isRightAsOfDate = ref('')
const isLoading = ref(false)

/** All unique dates across every source, sorted ascending. */
const isAllDates = computed(() => {
  if (!isData.value?.available_dates) return []
  const dateSet = new Set<string>()
  for (const dates of Object.values(isData.value.available_dates)) {
    for (const d of dates) dateSet.add(d)
  }
  return [...dateSet].sort()
})

function _pickDefaultDate(available: Record<string, string[]>): string {
  const actuals = available?.Actual || []
  if (actuals.length) return actuals[actuals.length - 1]
  const all = [...new Set(Object.values(available || {}).flat() as string[])].sort()
  return all.length ? all[all.length - 1] : ''
}

async function loadIS(vcode: string) {
  isLoading.value = true
  try {
    const params: any = {
      left_source: isLeftSource.value,
      left_period: isLeftPeriod.value,
      right_source: isRightSource.value,
      right_period: isRightPeriod.value,
    }
    if (isLeftAsOfDate.value) params.left_as_of_date = isLeftAsOfDate.value
    if (isRightAsOfDate.value) params.right_as_of_date = isRightAsOfDate.value
    const res = await api.get(`/api/financials/${vcode}/income-statement`, { params })
    isData.value = res.data
    // Set defaults on first load
    if (!isLeftAsOfDate.value) {
      isLeftAsOfDate.value = _pickDefaultDate(res.data.available_dates)
    }
    if (!isRightAsOfDate.value) {
      // Default right to prior-year same month when possible
      const leftTs = isLeftAsOfDate.value
      if (leftTs) {
        const d = new Date(leftTs)
        const priorYear = `${d.getFullYear() - 1}-${String(d.getMonth() + 1).padStart(2, '0')}`
        const allDates = [...new Set(Object.values(res.data.available_dates || {}).flat() as string[])].sort()
        const match = allDates.find(dt => dt.startsWith(priorYear))
        isRightAsOfDate.value = match || _pickDefaultDate(res.data.available_dates)
      } else {
        isRightAsOfDate.value = _pickDefaultDate(res.data.available_dates)
      }
    }
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    isLoading.value = false
  }
}

async function refreshIS() {
  if (deals.currentVcode) await loadIS(deals.currentVcode)
}

// ============================================================
// Balance Sheet
// ============================================================
interface BSRow {
  account: string
  level: number
  is_header?: boolean
  is_total?: boolean
  period1: number | null
  period2: number | null
  var_usd: number | null
  var_pct: number | null
}
interface BSResponse {
  rows: BSRow[]
  available_periods: string[]
  period1_label: string
  period2_label: string
}
const bsData = ref<BSResponse | null>(null)
const bsPeriod1 = ref('')
const bsPeriod2 = ref('')
const bsLoading = ref(false)

async function loadBS(vcode: string) {
  bsLoading.value = true
  try {
    const params: any = {}
    if (bsPeriod1.value) params.period1 = bsPeriod1.value
    if (bsPeriod2.value) params.period2 = bsPeriod2.value
    const res = await api.get(`/api/financials/${vcode}/balance-sheet`, { params })
    bsData.value = res.data
    if (!bsPeriod1.value && res.data.period1_label) bsPeriod1.value = res.data.period1_label
    if (!bsPeriod2.value && res.data.period2_label) bsPeriod2.value = res.data.period2_label
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    bsLoading.value = false
  }
}

async function refreshBS() {
  if (deals.currentVcode) await loadBS(deals.currentVcode)
}

// ============================================================
// Tenant Roster
// ============================================================
interface TenantRow {
  tenant_name: string
  sf_leased: number
  lease_start: string | null
  lease_end: string | null
  annual_rent: number
  rpsf: number
  pct_gla: number
  pct_abr: number
  expiring_soon: boolean
  is_vacant: boolean
}
interface TenantResponse {
  tenants: TenantRow[]
  summary: { occupied_sf: number; occupancy_pct: number; wtd_avg_rpsf: number }
  rollover: {
    property_info?: Record<string, any>
    maturity_by_year?: Array<{ year: string; sf: number; annual_rent: number; avg_rpsf: number; pct_revenue: number }>
    exposure_2yr?: { gla: number; gla_pct: number; abr: number; abr_pct: number; rpsf: number }
  }
}
const tenantData = ref<TenantResponse | null>(null)
const tenantLoading = ref(false)

async function loadTenants(vcode: string) {
  tenantLoading.value = true
  try {
    const res = await api.get(`/api/financials/${vcode}/tenants`)
    tenantData.value = res.data
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    tenantLoading.value = false
  }
}

// ============================================================
// One Pager
// ============================================================
const onePagerData = ref<Record<string, any> | null>(null)
const onePagerQuarter = ref('')
const onePagerLoading = ref(false)

async function loadOnePager(vcode: string) {
  onePagerLoading.value = true
  try {
    const params: any = {}
    if (onePagerQuarter.value) params.quarter = onePagerQuarter.value
    const res = await api.get(`/api/financials/${vcode}/one-pager`, { params })
    onePagerData.value = res.data
    if (!onePagerQuarter.value && res.data.available_quarters?.length) {
      onePagerQuarter.value = res.data.available_quarters[res.data.available_quarters.length - 1]
    }
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    onePagerLoading.value = false
  }
}

async function refreshOnePager() {
  if (deals.currentVcode) await loadOnePager(deals.currentVcode)
}

// ============================================================
// Formatting helpers
// ============================================================
function fmtCurrency(val: number | null | undefined): string {
  if (val == null || isNaN(val)) return '—'
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 0, maximumFractionDigits: 0 }).format(val)
}
function fmtPct(val: number | null | undefined): string {
  if (val == null || isNaN(val)) return '—'
  return (val * 100).toFixed(1) + '%'
}
function fmtNumber(val: number | null | undefined, decimals = 0): string {
  if (val == null || isNaN(val)) return '—'
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: decimals }).format(val)
}
function fmtDate(val: string | null | undefined): string {
  if (!val) return '—'
  const d = new Date(val)
  return isNaN(d.getTime()) ? val : d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
}
</script>

<template>
  <div class="property-financials">
    <!-- Error banner -->
    <div v-if="error" class="error-banner">
      {{ error }}
      <button @click="error = null">Dismiss</button>
    </div>

    <!-- Deal selector -->
    <div class="deal-selector">
      <label>Select Deal:</label>
      <select :value="deals.currentVcode" @change="onDealSelect">
        <option value="">-- Choose a deal --</option>
        <option v-for="d in data.deals" :key="d.vcode" :value="d.vcode">
          {{ d.Investment_Name || d.vcode }}
        </option>
      </select>
    </div>

    <template v-if="deals.currentVcode">
      <!-- ======================================== -->
      <!-- Performance Chart (always shown) -->
      <!-- ======================================== -->
      <div class="section">
        <h3>Performance Chart</h3>
        <div class="chart-controls">
          <div class="control-group">
            <label>Frequency:</label>
            <select v-model="chartFreq" @change="refreshChart">
              <option>Monthly</option>
              <option>Quarterly</option>
              <option>Annually</option>
            </select>
          </div>
          <div class="control-group">
            <label>Periods:</label>
            <input type="number" v-model.number="chartPeriods" min="4" max="60" @change="refreshChart" style="width:60px" />
          </div>
        </div>

        <div v-if="chartLoading" class="loading">Loading chart data...</div>
        <template v-else-if="chartData && chartData.periods.length > 0">
          <DualAxisChart
            :categories="chartData.periods"
            :bar-data="chartBarData"
            :line-data="chartLineData"
            bar-axis-label="Occupancy %"
            line-axis-label="NOI ($000s)"
          />
        </template>
        <p v-else class="empty">No performance data available for this deal.</p>
      </div>

      <!-- ======================================== -->
      <!-- Income Statement -->
      <!-- ======================================== -->
      <div class="section expandable" :class="{ open: expanded['is'] }">
        <h3 class="section-header" @click="toggle('is')">
          <span class="caret">{{ expanded['is'] ? '&#9660;' : '&#9654;' }}</span>
          Income Statement
        </h3>
        <div v-if="expanded['is']" class="section-body">
          <div class="is-controls">
            <div class="col-control">
              <label>Left Column:</label>
              <select v-model="isLeftSource"><option>Actual</option><option>Budget</option><option>Underwriting</option><option>Valuation</option></select>
              <select v-model="isLeftPeriod"><option>TTM</option><option>YTD</option><option>Full Year</option><option>Estimate</option><option>Custom</option></select>
              <select v-model="isLeftAsOfDate">
                <option v-for="d in isAllDates" :key="d" :value="d">{{ d }}</option>
              </select>
            </div>
            <div class="col-control">
              <label>Right Column:</label>
              <select v-model="isRightSource"><option>Actual</option><option>Budget</option><option>Underwriting</option><option>Valuation</option></select>
              <select v-model="isRightPeriod"><option>TTM</option><option>YTD</option><option>Full Year</option><option>Estimate</option><option>Custom</option></select>
              <select v-model="isRightAsOfDate">
                <option v-for="d in isAllDates" :key="d" :value="d">{{ d }}</option>
              </select>
            </div>
            <button class="btn btn-sm" @click="refreshIS">Apply</button>
          </div>

          <div v-if="isLoading" class="loading">Loading income statement...</div>
          <template v-else-if="isData && isData.rows.length > 0">
            <div class="table-wrapper">
              <table class="fin-table">
                <thead>
                  <tr>
                    <th>Account</th>
                    <th class="right">{{ isData.left_label }}</th>
                    <th class="right">{{ isData.right_label }}</th>
                    <th class="right">Var ($)</th>
                    <th class="right">Var (%)</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="(row, idx) in isData.rows" :key="idx"
                    :class="{ 'header-row': row.is_header, 'total-row': row.is_total, 'calc-row': row.is_calc }">
                    <td :style="{ paddingLeft: (row.level * 20 + 8) + 'px' }">{{ row.account }}</td>
                    <template v-if="row.account === 'DSCR'">
                      <td class="right">{{ row.left != null ? row.left.toFixed(2) + 'x' : '' }}</td>
                      <td class="right">{{ row.right != null ? row.right.toFixed(2) + 'x' : '' }}</td>
                      <td class="right"></td>
                      <td class="right"></td>
                    </template>
                    <template v-else>
                      <td class="right">{{ row.left != null ? fmtCurrency(row.left) : '' }}</td>
                      <td class="right">{{ row.right != null ? fmtCurrency(row.right) : '' }}</td>
                      <td class="right" :class="{ positive: row.var_usd && row.var_usd > 0, negative: row.var_usd && row.var_usd < 0 }">
                        {{ row.var_usd != null ? fmtCurrency(row.var_usd) : '' }}
                      </td>
                      <td class="right">{{ row.var_pct != null ? fmtPct(row.var_pct) : '' }}</td>
                    </template>
                  </tr>
                </tbody>
              </table>
            </div>
          </template>
          <p v-else class="empty">No income statement data available.</p>
        </div>
      </div>

      <!-- ======================================== -->
      <!-- Balance Sheet -->
      <!-- ======================================== -->
      <div class="section expandable" :class="{ open: expanded['bs'] }">
        <h3 class="section-header" @click="toggle('bs')">
          <span class="caret">{{ expanded['bs'] ? '&#9660;' : '&#9654;' }}</span>
          Balance Sheet
        </h3>
        <div v-if="expanded['bs']" class="section-body">
          <div class="bs-controls">
            <div class="col-control">
              <label>Period 1:</label>
              <select v-model="bsPeriod1">
                <option v-for="p in (bsData?.available_periods || [])" :key="p" :value="p">{{ p }}</option>
              </select>
            </div>
            <div class="col-control">
              <label>Period 2:</label>
              <select v-model="bsPeriod2">
                <option v-for="p in (bsData?.available_periods || [])" :key="p" :value="p">{{ p }}</option>
              </select>
            </div>
            <button class="btn btn-sm" @click="refreshBS">Apply</button>
          </div>

          <div v-if="bsLoading" class="loading">Loading balance sheet...</div>
          <template v-else-if="bsData && bsData.rows.length > 0">
            <div class="table-wrapper">
              <table class="fin-table">
                <thead>
                  <tr>
                    <th>Account</th>
                    <th class="right">{{ bsData.period1_label }}</th>
                    <th class="right">{{ bsData.period2_label }}</th>
                    <th class="right">Var ($)</th>
                    <th class="right">Var (%)</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="(row, idx) in bsData.rows" :key="idx"
                    :class="{ 'header-row': row.is_header, 'total-row': row.is_total }">
                    <td :style="{ paddingLeft: (row.level * 20 + 8) + 'px' }">{{ row.account }}</td>
                    <td class="right">{{ row.period1 != null ? fmtCurrency(row.period1) : '' }}</td>
                    <td class="right">{{ row.period2 != null ? fmtCurrency(row.period2) : '' }}</td>
                    <td class="right" :class="{ positive: row.var_usd && row.var_usd > 0, negative: row.var_usd && row.var_usd < 0 }">
                      {{ row.var_usd != null ? fmtCurrency(row.var_usd) : '' }}
                    </td>
                    <td class="right">{{ row.var_pct != null ? fmtPct(row.var_pct) : '' }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </template>
          <p v-else class="empty">No balance sheet data available.</p>
        </div>
      </div>

      <!-- ======================================== -->
      <!-- Tenant Roster -->
      <!-- ======================================== -->
      <div class="section expandable" :class="{ open: expanded['tenants'] }">
        <h3 class="section-header" @click="toggle('tenants')">
          <span class="caret">{{ expanded['tenants'] ? '&#9660;' : '&#9654;' }}</span>
          Tenant Roster
        </h3>
        <div v-if="expanded['tenants']" class="section-body">
          <div v-if="tenantLoading" class="loading">Loading tenant data...</div>
          <template v-else-if="tenantData">
            <!-- Summary KPIs -->
            <div class="kpi-row" v-if="tenantData.summary">
              <KpiCard label="Occupied SF" :value="tenantData.summary.occupied_sf" format="integer" />
              <KpiCard label="Occupancy" :value="tenantData.summary.occupancy_pct" format="percent" />
              <KpiCard label="Wtd Avg RPSF" :value="tenantData.summary.wtd_avg_rpsf" format="currency" />
            </div>

            <!-- Tenant table -->
            <div class="table-wrapper" v-if="tenantData.tenants.length > 0">
              <table class="fin-table tenant-table">
                <thead>
                  <tr>
                    <th>Tenant</th>
                    <th class="right">SF Leased</th>
                    <th>Lease Start</th>
                    <th>Lease End</th>
                    <th class="right">Annual Rent</th>
                    <th class="right">$/SF</th>
                    <th class="right">% GLA</th>
                    <th class="right">% ABR</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="(t, idx) in tenantData.tenants" :key="idx"
                    :class="{ 'vacant-row': t.is_vacant, 'expiring-row': t.expiring_soon }">
                    <td>{{ t.tenant_name }}</td>
                    <td class="right">{{ fmtNumber(t.sf_leased) }}</td>
                    <td>{{ fmtDate(t.lease_start) }}</td>
                    <td>{{ fmtDate(t.lease_end) }}</td>
                    <td class="right">{{ fmtCurrency(t.annual_rent) }}</td>
                    <td class="right">{{ fmtNumber(t.rpsf, 2) }}</td>
                    <td class="right">{{ fmtPct(t.pct_gla) }}</td>
                    <td class="right">{{ fmtPct(t.pct_abr) }}</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <!-- Rollover: 2-Year Exposure -->
            <div v-if="tenantData.rollover?.exposure_2yr" class="rollover-section">
              <h4>2-Year Lease Exposure</h4>
              <div class="kpi-row">
                <KpiCard label="Expiring GLA" :value="tenantData.rollover.exposure_2yr.gla" format="integer" />
                <KpiCard label="% of GLA" :value="tenantData.rollover.exposure_2yr.gla_pct" format="percent" />
                <KpiCard label="Expiring ABR" :value="tenantData.rollover.exposure_2yr.abr" format="currency" />
                <KpiCard label="% of ABR" :value="tenantData.rollover.exposure_2yr.abr_pct" format="percent" />
              </div>
            </div>

            <!-- Rollover: Maturity by Year -->
            <div v-if="tenantData.rollover?.maturity_by_year?.length" class="rollover-section">
              <h4>Lease Maturity Schedule</h4>
              <div class="table-wrapper">
                <table class="fin-table">
                  <thead>
                    <tr>
                      <th>Year</th>
                      <th class="right">SF</th>
                      <th class="right">Annual Rent</th>
                      <th class="right">Avg $/SF</th>
                      <th class="right">% Revenue</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="m in tenantData.rollover.maturity_by_year" :key="m.year">
                      <td>{{ m.year }}</td>
                      <td class="right">{{ fmtNumber(m.sf) }}</td>
                      <td class="right">{{ fmtCurrency(m.annual_rent) }}</td>
                      <td class="right">{{ fmtNumber(m.avg_rpsf, 2) }}</td>
                      <td class="right">{{ fmtPct(m.pct_revenue) }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </template>
          <p v-else class="empty">No tenant data available for this deal.</p>
        </div>
      </div>

      <!-- ======================================== -->
      <!-- One Pager -->
      <!-- ======================================== -->
      <div class="section expandable" :class="{ open: expanded['onepager'] }">
        <h3 class="section-header" @click="toggle('onepager')">
          <span class="caret">{{ expanded['onepager'] ? '&#9660;' : '&#9654;' }}</span>
          One Pager — Investor Report
        </h3>
        <div v-if="expanded['onepager']" class="section-body">
          <div class="op-controls">
            <div class="col-control">
              <label>Quarter:</label>
              <select v-model="onePagerQuarter" @change="refreshOnePager">
                <option v-for="q in (onePagerData?.available_quarters || [])" :key="q" :value="q">{{ q }}</option>
              </select>
            </div>
          </div>

          <div v-if="onePagerLoading" class="loading">Loading one pager...</div>
          <template v-else-if="onePagerData">
            <!-- General Information -->
            <div v-if="onePagerData.general" class="op-section">
              <h4>General Information</h4>
              <div class="detail-grid">
                <template v-for="(val, key) in onePagerData.general" :key="key">
                  <div class="detail-label">{{ key }}</div>
                  <div class="detail-value">{{ val ?? '—' }}</div>
                </template>
              </div>
            </div>

            <!-- Capitalization Stack -->
            <div v-if="onePagerData.cap_stack" class="op-section">
              <h4>Capitalization Stack</h4>
              <div class="detail-grid">
                <template v-for="(val, key) in onePagerData.cap_stack" :key="key">
                  <div class="detail-label">{{ key }}</div>
                  <div class="detail-value">{{ typeof val === 'number' ? fmtCurrency(val) : (val ?? '—') }}</div>
                </template>
              </div>
            </div>

            <!-- Property Performance -->
            <div v-if="onePagerData.property_performance" class="op-section">
              <h4>Property Performance</h4>
              <div class="detail-grid">
                <template v-for="(val, key) in onePagerData.property_performance" :key="key">
                  <div class="detail-label">{{ key }}</div>
                  <div class="detail-value">{{ typeof val === 'number' ? fmtCurrency(val) : (val ?? '—') }}</div>
                </template>
              </div>
            </div>

            <!-- PE Performance -->
            <div v-if="onePagerData.pe_performance" class="op-section">
              <h4>PE Performance</h4>
              <div class="detail-grid">
                <template v-for="(val, key) in onePagerData.pe_performance" :key="key">
                  <div class="detail-label">{{ key }}</div>
                  <div class="detail-value">{{ typeof val === 'number' ? fmtCurrency(val) : (val ?? '—') }}</div>
                </template>
              </div>
            </div>

            <!-- Comments -->
            <div v-if="onePagerData.comments" class="op-section">
              <h4>Comments</h4>
              <div v-for="(val, key) in onePagerData.comments" :key="key" class="comment-block">
                <div class="comment-label">{{ key }}</div>
                <div class="comment-text">{{ val || '(none)' }}</div>
              </div>
            </div>
          </template>
          <p v-else class="empty">No one pager data available.</p>
        </div>
      </div>
    </template>

    <p v-else class="empty">Select a deal to view property financials.</p>
  </div>
</template>

<style scoped>
.property-financials { max-width: 1200px; }

/* Error banner */
.error-banner { background: #fef2f2; border: 1px solid #fca5a5; color: #991b1b; padding: 10px 16px; border-radius: 8px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; }
.error-banner button { background: none; border: 1px solid #fca5a5; color: #991b1b; padding: 4px 12px; border-radius: 4px; cursor: pointer; }

/* Deal selector */
.deal-selector { display: flex; align-items: center; gap: 12px; margin-bottom: 20px; }
.deal-selector select { padding: 8px 12px; border: 1px solid var(--color-border); border-radius: 6px; font-size: 14px; min-width: 350px; }

/* Sections */
.section { background: var(--color-surface); border: 1px solid var(--color-border); border-radius: 8px; padding: 16px; margin-bottom: 16px; }
.section h3 { font-size: 14px; margin-bottom: 12px; }

/* Expandable */
.expandable .section-header { cursor: pointer; user-select: none; display: flex; align-items: center; gap: 8px; margin-bottom: 0; }
.expandable .section-header:hover { color: var(--color-accent); }
.expandable.open .section-header { margin-bottom: 12px; }
.caret { font-size: 10px; width: 14px; display: inline-block; }

/* Controls */
.chart-controls, .is-controls, .bs-controls, .op-controls {
  display: flex; gap: 16px; align-items: flex-end; flex-wrap: wrap; margin-bottom: 16px;
}
.col-control { display: flex; flex-direction: column; gap: 4px; }
.col-control label { font-size: 11px; color: var(--color-text-secondary); text-transform: uppercase; letter-spacing: 0.3px; }
.col-control select, .col-control input { padding: 6px 10px; border: 1px solid var(--color-border); border-radius: 4px; font-size: 13px; }
.btn { padding: 6px 16px; border: none; border-radius: 4px; background: var(--color-accent); color: white; cursor: pointer; font-size: 13px; }
.btn:hover { opacity: 0.9; }
.btn-sm { padding: 6px 12px; font-size: 12px; }

/* Financial tables */
.table-wrapper { overflow-x: auto; }
.fin-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.fin-table th { padding: 8px 12px; background: var(--color-accent); color: white; font-weight: 600; white-space: nowrap; text-align: left; }
.fin-table td { padding: 6px 12px; border-bottom: 1px solid var(--color-border); }
.fin-table .right { text-align: right; }
.fin-table th.right { text-align: right; }
.fin-table tbody tr:hover { background: #f0f4f8; }
.header-row { background: #f8fafc; }
.header-row td { font-weight: 700; font-size: 12px; text-transform: uppercase; letter-spacing: 0.3px; color: var(--color-text-secondary); }
.total-row td { font-weight: 700; border-top: 2px solid var(--color-text); }
.calc-row td { font-weight: 700; background: #eff6ff; }
.positive { color: #16a34a; }
.negative { color: #dc2626; }

/* Tenant-specific */
.vacant-row td { color: var(--color-text-secondary); font-style: italic; }
.expiring-row td { background: #fef3c7; }

/* KPI row */
.kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 16px; }

/* Rollover section */
.rollover-section { margin-top: 20px; }
.rollover-section h4 { font-size: 13px; font-weight: 600; margin-bottom: 12px; }

/* One Pager sections */
.op-section { margin-bottom: 20px; padding-bottom: 16px; border-bottom: 1px solid var(--color-border); }
.op-section:last-child { border-bottom: none; }
.op-section h4 { font-size: 13px; font-weight: 600; margin-bottom: 12px; }
.detail-grid { display: grid; grid-template-columns: 200px 1fr; gap: 4px 16px; font-size: 13px; }
.detail-label { color: var(--color-text-secondary); font-weight: 500; }
.detail-value { color: var(--color-text); }
.comment-block { margin-bottom: 12px; }
.comment-label { font-size: 12px; color: var(--color-text-secondary); font-weight: 500; margin-bottom: 4px; }
.comment-text { font-size: 13px; white-space: pre-wrap; }

/* Loading / Empty */
.loading { text-align: center; padding: 24px; color: var(--color-text-secondary); font-style: italic; }
.empty { text-align: center; padding: 24px; color: var(--color-text-secondary); font-style: italic; }
</style>
