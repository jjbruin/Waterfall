<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import api from '../api/client'

const router = useRouter()
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { BarChart, PieChart } from 'echarts/charts'
import {
  GridComponent, TooltipComponent, LegendComponent,
  MarkLineComponent,
} from 'echarts/components'

use([CanvasRenderer, BarChart, PieChart, GridComponent, TooltipComponent, LegendComponent, MarkLineComponent])

const CLR_DARK = '#1F4E79'
const CLR_ACCENT = '#ED7D31'
const CLR_GREEN = '#548235'
const CLR_RED = '#C00000'
const CLR_GREY = '#808080'

// State
const reviews = ref<any[]>([])
const selectedReviewId = ref<number | null>(null)
const loading = ref(false)
const activeTab = ref('overview')
const resolvingField = ref<string | null>(null)

// Risk analysis data
const riskData = ref<any>(null)
const tenants = ref<any[]>([])
const validation = ref<any[]>([])
const expirations = ref<any>(null)
const cotenancy = ref<any>(null)
const scenarios = ref<any[]>([])
const exclusiveUse = ref<any[]>([])
const options = ref<any[]>([])

// Edit state for field resolution
const editingTenantId = ref<number | null>(null)
const editingField = ref<string | null>(null)
const editValue = ref('')
const editSource = ref('analyst')

const TABS = [
  { key: 'overview', label: 'Overview' },
  { key: 'expirations', label: 'Lease Expirations' },
  { key: 'validation', label: 'Validation' },
  { key: 'cotenancy', label: 'Co-Tenancy Risk' },
  { key: 'scenarios', label: 'Scenario Analysis' },
  { key: 'exclusive', label: 'Exclusive Use' },
  { key: 'options', label: 'Options' },
]

const RESOLVABLE_FIELDS = [
  'square_feet', 'annual_rent', 'monthly_rent',
  'rent_per_sf', 'lease_start', 'lease_end', 'security_deposit',
]

// Load reviews list
async function loadReviews() {
  try {
    const res = await api.get('/api/lease-review/reviews')
    reviews.value = res.data
    if (reviews.value.length && !selectedReviewId.value) {
      selectedReviewId.value = reviews.value[0].id
    }
  } catch (e: any) {
    console.error('Failed to load reviews:', e)
  }
}

// Load risk analysis data
async function loadRiskData() {
  if (!selectedReviewId.value) return
  loading.value = true
  try {
    const res = await api.get(`/api/lease-review/reviews/${selectedReviewId.value}/risk-analysis`)
    riskData.value = res.data
    tenants.value = res.data.tenants || []
    validation.value = res.data.validation || []
    expirations.value = res.data.expirations || null
    cotenancy.value = res.data.cotenancy || null
    scenarios.value = res.data.scenarios || []
    exclusiveUse.value = res.data.exclusive_use || []
    options.value = res.data.options || []
  } catch (e: any) {
    console.error('Failed to load risk analysis:', e)
  } finally {
    loading.value = false
  }
}

// Resolve a field
async function resolveField(tenantId: number, fieldName: string, value: string, source: string = 'analyst') {
  try {
    resolvingField.value = `${tenantId}-${fieldName}`
    await api.put(`/api/lease-review/reviews/${selectedReviewId.value}/tenants/${tenantId}/resolve`, {
      field_name: fieldName,
      value,
      source,
    })
    editingTenantId.value = null
    editingField.value = null
    await loadRiskData()
  } catch (e: any) {
    alert('Failed to resolve field: ' + (e.response?.data?.error || e.message))
  } finally {
    resolvingField.value = null
  }
}

// Clear a resolution
async function clearResolution(tenantId: number, fieldName: string) {
  try {
    resolvingField.value = `${tenantId}-${fieldName}`
    await api.delete(`/api/lease-review/reviews/${selectedReviewId.value}/tenants/${tenantId}/resolve/${fieldName}`)
    await loadRiskData()
  } catch (e: any) {
    alert('Failed to clear resolution: ' + (e.response?.data?.error || e.message))
  } finally {
    resolvingField.value = null
  }
}

// Toggle exercised status on an option
async function toggleExercised(optionId: number, current: boolean) {
  try {
    await api.put(`/api/lease-review/reviews/${selectedReviewId.value}/options/${optionId}/exercised`, {
      exercised: !current,
    })
    await loadRiskData()
  } catch (e: any) {
    alert('Failed to update option: ' + (e.response?.data?.error || e.message))
  }
}

// Group options by tenant for display
const optionsByTenant = computed(() => {
  const groups: Record<string, any[]> = {}
  for (const o of options.value) {
    const key = `${o.tenant_name} — ${o.suite || 'N/A'}`
    if (!groups[key]) groups[key] = []
    groups[key].push(o)
  }
  // Sort each group: renewal first, then termination, then by option_number
  for (const key of Object.keys(groups)) {
    groups[key].sort((a: any, b: any) => {
      if (a.option_type !== b.option_type) return a.option_type === 'renewal' ? -1 : 1
      return (a.option_number || 0) - (b.option_number || 0)
    })
  }
  return groups
})

function startEdit(tenantId: number, field: string, currentValue: any) {
  editingTenantId.value = tenantId
  editingField.value = field
  editValue.value = currentValue ?? ''
  editSource.value = 'analyst'
}

function cancelEdit() {
  editingTenantId.value = null
  editingField.value = null
}

function submitEdit(tenantId: number, field: string) {
  resolveField(tenantId, field, editValue.value, editSource.value)
}

// Summary stats
const summaryStats = computed(() => {
  if (!tenants.value.length) return null
  const occupied = tenants.value.filter(t => !t.is_vacant)
  const totalSf = occupied.reduce((s, t) => s + (t.square_feet || 0), 0)
  const totalRent = occupied.reduce((s, t) => s + (t.annual_rent || 0), 0)
  const approved = tenants.value.filter(t => t.approval_status === 'approved').length
  const resolved = tenants.value.filter(t => Object.keys(t.resolutions || {}).length > 0).length
  const withCotenancy = tenants.value.filter(t => t.has_cotenancy).length
  const withExclusive = tenants.value.filter(t => t.has_exclusive_use).length
  return {
    total: tenants.value.length,
    occupied: occupied.length,
    totalSf,
    totalRent,
    approved,
    resolved,
    withCotenancy,
    withExclusive,
    avgRentPerSf: totalSf > 0 ? totalRent / totalSf : 0,
  }
})

// Validation summary
const validationSummary = computed(() => {
  if (!validation.value.length) return null
  const match = validation.value.filter(v => v.status === 'match').length
  const mismatch = validation.value.filter(v => v.status === 'mismatch').length
  const pending = validation.value.filter(v => v.status === 'pending').length
  return { match, mismatch, pending, total: validation.value.length }
})

// Grouped validation by tenant
const validationByTenant = computed(() => {
  const grouped: Record<string, any[]> = {}
  for (const v of validation.value) {
    const key = `${v.tenant} (${v.suite || 'N/A'})`
    if (!grouped[key]) grouped[key] = []
    grouped[key].push(v)
  }
  return grouped
})

// Expiration chart
const expirationChartOpts = computed(() => {
  if (!expirations.value?.yearly_data) return null
  const data = expirations.value.yearly_data
  return {
    tooltip: { trigger: 'axis' },
    legend: { data: ['Expiring SF', '% of Total Rent'] },
    grid: { left: 60, right: 60, bottom: 30 },
    xAxis: { type: 'category', data: data.map((d: any) => d.year) },
    yAxis: [
      { type: 'value', name: 'Square Feet', position: 'left' },
      { type: 'value', name: '% of Rent', position: 'right', max: 100 },
    ],
    series: [
      {
        name: 'Expiring SF',
        type: 'bar',
        data: data.map((d: any) => d.expiring_sf),
        itemStyle: { color: CLR_DARK },
      },
      {
        name: '% of Total Rent',
        type: 'bar',
        yAxisIndex: 1,
        data: data.map((d: any) => d.pct_of_total_rent),
        itemStyle: { color: CLR_ACCENT },
      },
    ],
  }
})

// Co-tenancy risk chart
const cotenancyChartOpts = computed(() => {
  if (!cotenancy.value?.rent_at_risk) return null
  const entries = Object.entries(cotenancy.value.rent_at_risk) as [string, any][]
  if (!entries.length) return null
  const sorted = entries.sort((a, b) => b[1].total_dependent_rent - a[1].total_dependent_rent).slice(0, 10)
  return {
    tooltip: { trigger: 'axis' },
    grid: { left: 150, right: 40, bottom: 30 },
    xAxis: { type: 'value', name: 'Dependent Rent at Risk ($)' },
    yAxis: { type: 'category', data: sorted.map(e => e[0]).reverse(), axisLabel: { width: 130, overflow: 'truncate' } },
    series: [{
      type: 'bar',
      data: sorted.map(e => e[1].total_dependent_rent).reverse(),
      itemStyle: { color: CLR_RED },
    }],
  }
})

// Format helpers
function fmt$(v: any) {
  if (v == null || isNaN(v)) return '-'
  return '$' + Number(v).toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })
}
function fmt$c(v: any) {
  if (v == null || isNaN(v)) return '-'
  return '$' + Number(v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}
const DOLLAR_FIELDS = new Set(['annual_rent', 'monthly_rent', 'security_deposit'])
const DOLLAR_CENTS_FIELDS = new Set(['rent_per_sf'])
const NUMBER_FIELDS = new Set(['square_feet'])
function fmtValidationVal(field: string, v: any) {
  if (v == null) return '-'
  if (DOLLAR_FIELDS.has(field)) return fmt$(v)
  if (DOLLAR_CENTS_FIELDS.has(field)) return fmt$c(v)
  if (NUMBER_FIELDS.has(field) && !isNaN(v)) return Number(v).toLocaleString('en-US')
  return v
}
function fmtSf(v: any) {
  if (v == null || isNaN(v)) return '-'
  return Number(v).toLocaleString('en-US') + ' SF'
}
function fmtPct(v: any) {
  if (v == null || isNaN(v)) return '-'
  return Number(v).toFixed(1) + '%'
}
function fmtDate(v: any) {
  if (!v) return '-'
  try {
    const m = String(v).match(/^(\d{4})-(\d{2})-(\d{2})/)
    if (m) return `${parseInt(m[2])}/${parseInt(m[3])}/${m[1]}`
    return new Date(v).toLocaleDateString()
  } catch { return String(v) }
}

const expandedScenario = ref<string | null>(null)

watch(selectedReviewId, () => { loadRiskData() })

onMounted(() => {
  loadReviews()
})
</script>

<template>
  <div class="lease-risk-analysis">
    <div class="page-header">
      <h1>Lease Risk Analysis</h1>
      <div class="header-controls">
        <select v-model="selectedReviewId" class="review-select">
          <option v-for="r in reviews" :key="r.id" :value="r.id">
            {{ r.property_name }} — {{ r.review_name || 'Review #' + r.id }}
          </option>
        </select>
      </div>
    </div>

    <div v-if="loading" class="loading-bar">Loading risk analysis data...</div>

    <div v-else-if="!selectedReviewId" class="empty-state">
      Select a lease review to view risk analysis.
    </div>

    <template v-else-if="riskData">
      <!-- Tab Navigation -->
      <div class="tab-bar">
        <button
          v-for="tab in TABS" :key="tab.key"
          class="tab-btn"
          :class="{ active: activeTab === tab.key }"
          @click="activeTab = tab.key"
        >{{ tab.label }}</button>
      </div>

      <!-- ═══ OVERVIEW TAB ═══ -->
      <div v-if="activeTab === 'overview'" class="tab-content">
        <!-- Summary Cards -->
        <div v-if="summaryStats" class="kpi-row">
          <div class="kpi-card">
            <div class="kpi-value">{{ summaryStats.total }}</div>
            <div class="kpi-label">Total Tenants</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-value">{{ fmtSf(summaryStats.totalSf) }}</div>
            <div class="kpi-label">Total GLA</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-value">{{ fmt$(summaryStats.totalRent) }}</div>
            <div class="kpi-label">Total Annual Rent</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-value">{{ fmt$(summaryStats.avgRentPerSf) }}/SF</div>
            <div class="kpi-label">Avg Rent/SF</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-value">{{ summaryStats.withCotenancy }}</div>
            <div class="kpi-label">Co-Tenancy Clauses</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-value">{{ summaryStats.resolved }}</div>
            <div class="kpi-label">Tenants w/ Resolutions</div>
          </div>
        </div>

        <!-- Tenant Roster with Field Resolution -->
        <h3>Tenant Roster (Resolved Data)</h3>
        <div class="table-wrapper">
          <table class="data-table">
            <thead>
              <tr>
                <th>Tenant</th>
                <th>Suite</th>
                <th>SF</th>
                <th>Annual Rent</th>
                <th>Rent/SF</th>
                <th>Lease Start</th>
                <th>Lease End</th>
                <th>Approval</th>
                <th>Resolutions</th>
                <th>Abstract</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="t in tenants" :key="t.id"
                  :class="{ vacant: t.is_vacant, resolved: Object.keys(t.resolutions || {}).length > 0 }">
                <td>{{ t.tenant_name }}</td>
                <td>{{ t.suite }}</td>
                <td class="num-cell" :class="{ 'has-resolution': t.resolutions?.square_feet }">
                  <template v-if="editingTenantId === t.id && editingField === 'square_feet'">
                    <input v-model="editValue" class="inline-edit" type="number" @keyup.enter="submitEdit(t.id, 'square_feet')" @keyup.escape="cancelEdit" />
                    <button class="btn-xs btn-save" @click="submitEdit(t.id, 'square_feet')">Save</button>
                    <button class="btn-xs btn-cancel" @click="cancelEdit">X</button>
                  </template>
                  <template v-else>
                    <span @dblclick="startEdit(t.id, 'square_feet', t.square_feet)" :title="t.resolutions?.square_feet ? 'Resolved by analyst' : 'Double-click to override'">
                      {{ t.square_feet?.toLocaleString() || '-' }}
                    </span>
                    <span v-if="t.resolutions?.square_feet" class="resolution-badge" title="Analyst resolved">R</span>
                    <button v-if="t.resolutions?.square_feet" class="btn-xs btn-clear" @click="clearResolution(t.id, 'square_feet')" title="Revert to original">&#x21A9;</button>
                  </template>
                </td>
                <td class="num-cell" :class="{ 'has-resolution': t.resolutions?.annual_rent }">
                  <template v-if="editingTenantId === t.id && editingField === 'annual_rent'">
                    <input v-model="editValue" class="inline-edit" type="number" @keyup.enter="submitEdit(t.id, 'annual_rent')" @keyup.escape="cancelEdit" />
                    <button class="btn-xs btn-save" @click="submitEdit(t.id, 'annual_rent')">Save</button>
                    <button class="btn-xs btn-cancel" @click="cancelEdit">X</button>
                  </template>
                  <template v-else>
                    <span @dblclick="startEdit(t.id, 'annual_rent', t.annual_rent)" :title="t.resolutions?.annual_rent ? 'Resolved by analyst' : 'Double-click to override'">
                      {{ fmt$(t.annual_rent) }}
                    </span>
                    <span v-if="t.resolutions?.annual_rent" class="resolution-badge" title="Analyst resolved">R</span>
                    <button v-if="t.resolutions?.annual_rent" class="btn-xs btn-clear" @click="clearResolution(t.id, 'annual_rent')" title="Revert to original">&#x21A9;</button>
                  </template>
                </td>
                <td class="num-cell" :class="{ 'has-resolution': t.resolutions?.rent_per_sf }">
                  <span @dblclick="startEdit(t.id, 'rent_per_sf', t.rent_per_sf)" :title="'Double-click to override'">
                    {{ t.rent_per_sf ? '$' + Number(t.rent_per_sf).toFixed(2) : '-' }}
                  </span>
                  <span v-if="t.resolutions?.rent_per_sf" class="resolution-badge">R</span>
                </td>
                <td :class="{ 'has-resolution': t.resolutions?.lease_start }">
                  <span @dblclick="startEdit(t.id, 'lease_start', t.lease_start)">{{ fmtDate(t.lease_start) }}</span>
                  <span v-if="t.resolutions?.lease_start" class="resolution-badge">R</span>
                </td>
                <td :class="{ 'has-resolution': t.resolutions?.lease_end }">
                  <span @dblclick="startEdit(t.id, 'lease_end', t.lease_end)">{{ fmtDate(t.lease_end) }}</span>
                  <span v-if="t.resolutions?.lease_end" class="resolution-badge">R</span>
                </td>
                <td>
                  <span class="status-badge" :class="t.approval_status">{{ t.approval_status }}</span>
                </td>
                <td class="num-cell">{{ Object.keys(t.resolutions || {}).length || '-' }}</td>
                <td>
                  <button
                    class="btn-xs btn-abstract"
                    @click.stop="router.push({ path: '/lease-abstract', query: { review: String(selectedReviewId), tenant: String(t.id) } })"
                  >View</button>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- ═══ LEASE EXPIRATIONS TAB ═══ -->
      <div v-if="activeTab === 'expirations'" class="tab-content">
        <template v-if="expirations">
          <h3>Lease Expiration Schedule</h3>
          <div v-if="expirationChartOpts" class="chart-container">
            <v-chart :option="expirationChartOpts" style="height:350px" autoresize />
          </div>
          <div class="table-wrapper">
            <table class="data-table">
              <thead>
                <tr>
                  <th>Year</th>
                  <th>Tenants</th>
                  <th>Expiring SF</th>
                  <th>Expiring Rent</th>
                  <th>% of Total Rent</th>
                  <th>Avg Rent/SF</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="d in expirations.yearly_data" :key="d.year"
                    :class="{ highlight: d.pct_of_total_rent > 20 }">
                  <td>{{ d.year }}</td>
                  <td class="num-cell">{{ d.tenant_count }}</td>
                  <td class="num-cell">{{ d.expiring_sf.toLocaleString() }}</td>
                  <td class="num-cell">{{ fmt$(d.expiring_rent) }}</td>
                  <td class="num-cell" :class="{ danger: d.pct_of_total_rent > 25 }">{{ fmtPct(d.pct_of_total_rent) }}</td>
                  <td class="num-cell">{{ d.avg_rent_per_sf ? '$' + d.avg_rent_per_sf.toFixed(2) : '-' }}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <!-- Material leases expiring per year -->
          <template v-for="(leases, year) in (expirations.material_leases || {})" :key="year">
            <h4 class="material-header">Material Leases Expiring in {{ year }}</h4>
            <div class="table-wrapper">
              <table class="data-table compact">
                <thead>
                  <tr><th>Tenant</th><th>Suite</th><th>SF</th><th>Annual Rent</th><th>Rent/SF</th><th>Lease End</th><th>Co-Tenancy</th></tr>
                </thead>
                <tbody>
                  <tr v-for="(l, i) in leases" :key="i">
                    <td>{{ l.tenant_name }}</td>
                    <td>{{ l.suite }}</td>
                    <td class="num-cell">{{ l.square_feet?.toLocaleString() }}</td>
                    <td class="num-cell">{{ fmt$(l.annual_rent) }}</td>
                    <td class="num-cell">{{ l.rent_per_sf ? '$' + l.rent_per_sf.toFixed(2) : '-' }}</td>
                    <td>{{ fmtDate(l.lease_end) }}</td>
                    <td>
                      <span v-if="l.has_cotenancy" class="flag-warn">Yes</span>
                      <span v-else>-</span>
                    </td>
                  </tr>
                </tbody>
              </table>
              <p v-for="(l, i) in leases.filter((x: any) => x.cotenancy_implication)" :key="'imp-'+i" class="implication-note">
                {{ l.tenant_name }}: {{ l.cotenancy_implication }}
              </p>
            </div>
          </template>
        </template>
        <div v-else class="empty-state">No expiration data available. Run validation in Lease Review first.</div>
      </div>

      <!-- ═══ VALIDATION TAB ═══ -->
      <div v-if="activeTab === 'validation'" class="tab-content">
        <template v-if="validationSummary">
          <div class="kpi-row">
            <div class="kpi-card match"><div class="kpi-value">{{ validationSummary.match }}</div><div class="kpi-label">Matches</div></div>
            <div class="kpi-card mismatch"><div class="kpi-value">{{ validationSummary.mismatch }}</div><div class="kpi-label">Mismatches</div></div>
            <div class="kpi-card pending"><div class="kpi-value">{{ validationSummary.pending }}</div><div class="kpi-label">Pending</div></div>
            <div class="kpi-card"><div class="kpi-value">{{ validationSummary.total }}</div><div class="kpi-label">Total Checks</div></div>
          </div>

          <h3>Validation Details by Tenant</h3>
          <div v-for="(items, tenantKey) in validationByTenant" :key="tenantKey" class="validation-group">
            <h4>{{ tenantKey }}</h4>
            <div class="table-wrapper">
              <table class="data-table compact">
                <thead>
                  <tr>
                    <th>Field</th>
                    <th>Source</th>
                    <th>Seller Value</th>
                    <th>Lease Value</th>
                    <th>Status</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="v in items" :key="v.tenant_id + '-' + v.field + '-' + v.source_type"
                      :class="{ 'row-mismatch': v.status === 'mismatch', 'row-match': v.status === 'match' }">
                    <td>{{ v.field }}</td>
                    <td>{{ v.source_type }}</td>
                    <td>{{ fmtValidationVal(v.field, v.seller_value) }}</td>
                    <td>{{ fmtValidationVal(v.field, v.lease_value) }}</td>
                    <td><span class="status-badge" :class="v.status">{{ v.status }}</span></td>
                    <td>
                      <template v-if="v.status === 'mismatch' && RESOLVABLE_FIELDS.includes(v.field)">
                        <button class="btn-xs btn-resolve" @click="resolveField(v.tenant_id, v.field, v.seller_value, 'seller')"
                                title="Use seller value">Seller</button>
                        <button class="btn-xs btn-resolve" @click="resolveField(v.tenant_id, v.field, v.lease_value, 'lease')"
                                title="Use lease value">Lease</button>
                      </template>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </template>
        <div v-else class="empty-state">No validation data available. Run validation in Lease Review first.</div>
      </div>

      <!-- ═══ CO-TENANCY RISK TAB ═══ -->
      <div v-if="activeTab === 'cotenancy'" class="tab-content">
        <template v-if="cotenancy?.clauses?.length">
          <h3>Co-Tenancy Risk Overview</h3>
          <div v-if="cotenancyChartOpts" class="chart-container">
            <v-chart :option="cotenancyChartOpts" style="height:350px" autoresize />
          </div>

          <h3>Co-Tenancy Clauses</h3>
          <div class="table-wrapper">
            <table class="data-table">
              <thead>
                <tr>
                  <th>Tenant</th>
                  <th>Suite</th>
                  <th>Named Co-Tenants</th>
                  <th>Trigger</th>
                  <th>Alt Rent</th>
                  <th>Can Terminate</th>
                  <th>Cure Period</th>
                  <th>Curable</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(c, i) in cotenancy.clauses" :key="i">
                  <td>{{ c.tenant_name }}</td>
                  <td>{{ c.suite }}</td>
                  <td>{{ (c.named_cotenants || []).join(', ') || '-' }}</td>
                  <td>{{ c.trigger_description || '-' }}</td>
                  <td>{{ c.alt_rent_formula || '-' }}</td>
                  <td><span :class="c.termination_right ? 'flag-danger' : ''">{{ c.termination_right ? 'Yes' : 'No' }}</span></td>
                  <td>{{ c.cure_period_days ? c.cure_period_days + ' days' : '-' }}</td>
                  <td>{{ c.is_curable ? 'Yes' : 'No' }}</td>
                </tr>
              </tbody>
            </table>
          </div>

          <h3>Rent at Risk by Named Co-Tenant</h3>
          <div class="table-wrapper">
            <table class="data-table">
              <thead><tr><th>Co-Tenant</th><th>Dependent Tenants</th><th>Dependent Rent</th><th>Termination Eligible</th></tr></thead>
              <tbody>
                <tr v-for="(risk, name) in (cotenancy.rent_at_risk || {})" :key="name as string">
                  <td>{{ name }}</td>
                  <td class="num-cell">{{ risk.dependent_count }}</td>
                  <td class="num-cell">{{ fmt$(risk.total_dependent_rent) }}</td>
                  <td class="num-cell" :class="{ danger: risk.termination_eligible_count > 0 }">{{ risk.termination_eligible_count }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </template>
        <div v-else class="empty-state">No co-tenancy clauses found for this review.</div>
      </div>

      <!-- ═══ SCENARIO ANALYSIS TAB ═══ -->
      <div v-if="activeTab === 'scenarios'" class="tab-content">
        <template v-if="scenarios.length">
          <h3>Departure Scenario Analysis</h3>
          <p class="section-desc">Analyze the impact if a named co-tenant departs. Shows which dependent tenants would be affected and the total rent at risk.</p>
          <div v-for="s in scenarios" :key="s.departing_tenant" class="scenario-card">
            <div class="scenario-header" @click="expandedScenario = expandedScenario === s.departing_tenant ? null : s.departing_tenant">
              <div class="scenario-title">
                <strong>{{ s.departing_tenant }}</strong> departs
              </div>
              <div class="scenario-metrics">
                <span class="metric">{{ s.dependent_count }} dependent{{ s.dependent_count > 1 ? 's' : '' }}</span>
                <span class="metric danger">{{ fmt$(s.total_dependent_rent) }} at risk</span>
                <span v-if="s.termination_eligible > 0" class="metric danger">{{ s.termination_eligible }} can terminate</span>
              </div>
              <span class="chevron">{{ expandedScenario === s.departing_tenant ? '&#x25BE;' : '&#x25B8;' }}</span>
            </div>
            <div v-if="expandedScenario === s.departing_tenant" class="scenario-body">
              <table class="data-table compact">
                <thead><tr><th>Affected Tenant</th><th>Annual Rent</th><th>Alt Rent Formula</th><th>Can Terminate</th><th>Cure Days</th><th>Curable</th></tr></thead>
                <tbody>
                  <tr v-for="(imp, j) in s.impacts" :key="j"
                      :class="{ 'row-danger': imp.can_terminate }">
                    <td>{{ imp.tenant }}</td>
                    <td class="num-cell">{{ fmt$(imp.annual_rent) }}</td>
                    <td>{{ imp.alt_rent_formula || '-' }}</td>
                    <td><span :class="imp.can_terminate ? 'flag-danger' : ''">{{ imp.can_terminate ? 'Yes' : 'No' }}</span></td>
                    <td>{{ imp.cure_days ?? '-' }}</td>
                    <td>{{ imp.is_curable ? 'Yes' : 'No' }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </template>
        <div v-else class="empty-state">No scenarios available. Co-tenancy data is needed for scenario analysis.</div>
      </div>

      <!-- ═══ EXCLUSIVE USE TAB ═══ -->
      <div v-if="activeTab === 'exclusive'" class="tab-content">
        <template v-if="exclusiveUse.length">
          <h3>Exclusive Use Restrictions</h3>
          <div class="table-wrapper">
            <table class="data-table">
              <thead><tr><th>Tenant</th><th>Suite</th><th>Restricted Use</th><th>Restriction Text</th></tr></thead>
              <tbody>
                <tr v-for="(e, i) in exclusiveUse" :key="i">
                  <td>{{ e.tenant_name }}</td>
                  <td>{{ e.suite }}</td>
                  <td>{{ e.restricted_use || '-' }}</td>
                  <td class="wrap-cell">{{ e.restriction_text || '-' }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </template>
        <div v-else class="empty-state">No exclusive use restrictions found.</div>
      </div>

      <!-- ═══ OPTIONS TAB ═══ -->
      <div v-if="activeTab === 'options'" class="tab-content">
        <template v-if="options.length">
          <h3>Lease Options</h3>
          <div v-for="(items, tenantKey) in optionsByTenant" :key="tenantKey" class="options-group">
            <h4>{{ tenantKey }}</h4>
            <div class="table-wrapper">
              <table class="data-table compact">
                <thead>
                  <tr>
                    <th>Type</th>
                    <th>Option #</th>
                    <th>Start</th>
                    <th>End</th>
                    <th>Term (Years)</th>
                    <th>Notice (Days)</th>
                    <th>Notice Deadline</th>
                    <th>Rent Terms / Conditions</th>
                    <th>Auto-Renewal</th>
                    <th>Exercised</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="(o, i) in items" :key="i"
                      :class="{ 'row-exercised': o.exercised }">
                    <td><span class="option-type-badge" :class="o.option_type">{{ o.option_type }}</span></td>
                    <td class="num-cell">{{ o.option_number ?? '-' }}</td>
                    <td>{{ fmtDate(o.option_start) }}</td>
                    <td>{{ fmtDate(o.option_end) }}</td>
                    <td class="num-cell">{{ o.term_years ?? '-' }}</td>
                    <td class="num-cell">{{ o.notice_days ?? '-' }}</td>
                    <td>{{ fmtDate(o.notice_deadline) }}</td>
                    <td class="wrap-cell">{{ o.rent_terms || '-' }}</td>
                    <td>{{ o.option_type === 'renewal' ? (o.auto_renewal ? 'Yes' : 'No') : '-' }}</td>
                    <td>
                      <button class="btn-xs" :class="o.exercised ? 'btn-exercised' : 'btn-not-exercised'"
                              @click="toggleExercised(o.id, o.exercised)">
                        {{ o.exercised ? 'Yes' : 'No' }}
                      </button>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </template>
        <div v-else class="empty-state">No lease options found.</div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.lease-risk-analysis {
  padding: 24px 32px;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  color: #1a1a1a;
}
.page-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 20px;
}
.page-header h1 {
  font-size: 1.5rem;
  font-weight: 600;
  color: #1F4E79;
  margin: 0;
}
.review-select {
  padding: 6px 12px;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 0.9rem;
  min-width: 300px;
}
.loading-bar {
  padding: 40px;
  text-align: center;
  color: #666;
  font-style: italic;
}
.empty-state {
  padding: 40px;
  text-align: center;
  color: #999;
}

/* Tabs */
.tab-bar {
  display: flex;
  gap: 2px;
  border-bottom: 2px solid #1F4E79;
  margin-bottom: 20px;
}
.tab-btn {
  padding: 8px 16px;
  border: none;
  background: #f0f0f0;
  color: #555;
  cursor: pointer;
  font-size: 0.85rem;
  border-radius: 4px 4px 0 0;
  transition: background 0.15s;
}
.tab-btn:hover { background: #e0e0e0; }
.tab-btn.active {
  background: #1F4E79;
  color: #fff;
  font-weight: 600;
}
.tab-content { min-height: 300px; }

/* KPIs */
.kpi-row {
  display: flex;
  gap: 16px;
  margin-bottom: 24px;
  flex-wrap: wrap;
}
.kpi-card {
  flex: 1;
  min-width: 140px;
  background: #f8f9fa;
  border: 1px solid #dee2e6;
  border-radius: 6px;
  padding: 16px;
  text-align: center;
}
.kpi-card.match { border-left: 4px solid #548235; }
.kpi-card.mismatch { border-left: 4px solid #C00000; }
.kpi-card.pending { border-left: 4px solid #ED7D31; }
.kpi-value { font-size: 1.4rem; font-weight: 700; color: #1F4E79; }
.kpi-label { font-size: 0.78rem; color: #666; margin-top: 4px; }

/* Tables */
.table-wrapper { overflow-x: auto; margin-bottom: 24px; }
.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.82rem;
}
.data-table th {
  background: #1F4E79;
  color: #fff;
  padding: 8px 10px;
  text-align: left;
  font-weight: 600;
  white-space: nowrap;
}
.data-table td {
  padding: 6px 10px;
  border-bottom: 1px solid #e8e8e8;
}
.data-table tbody tr:hover { background: #f0f4f8; }
.data-table.compact td, .data-table.compact th { padding: 4px 8px; }
.num-cell { text-align: right; font-variant-numeric: tabular-nums; }
.wrap-cell { max-width: 300px; word-wrap: break-word; white-space: normal; }
tr.vacant { opacity: 0.5; }
tr.resolved { background: #f0faf0; }
tr.highlight td { background: #fff3cd; }
.row-mismatch td { background: #fde8e8; }
.row-match td { background: #e8fde8; }
.row-danger td { background: #fde8e8; }
.danger { color: #C00000; font-weight: 600; }

/* Status badges */
.status-badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 10px;
  font-size: 0.72rem;
  font-weight: 600;
  text-transform: uppercase;
}
.status-badge.pending { background: #fff3cd; color: #856404; }
.status-badge.approved { background: #d4edda; color: #155724; }
.status-badge.flagged { background: #f8d7da; color: #721c24; }
.status-badge.match { background: #d4edda; color: #155724; }
.status-badge.mismatch { background: #f8d7da; color: #721c24; }

/* Resolution indicators */
.has-resolution { background: #e6f3ff; }
.resolution-badge {
  display: inline-block;
  width: 16px;
  height: 16px;
  line-height: 16px;
  text-align: center;
  background: #1F4E79;
  color: #fff;
  border-radius: 50%;
  font-size: 0.6rem;
  font-weight: 700;
  margin-left: 4px;
  vertical-align: middle;
}

/* Inline edit */
.inline-edit {
  width: 100px;
  padding: 2px 4px;
  border: 1px solid #1F4E79;
  border-radius: 3px;
  font-size: 0.82rem;
}
.btn-xs {
  padding: 1px 6px;
  font-size: 0.7rem;
  border: none;
  border-radius: 3px;
  cursor: pointer;
  margin-left: 2px;
}
.btn-save { background: #548235; color: #fff; }
.btn-cancel { background: #999; color: #fff; }
.btn-clear { background: transparent; color: #999; font-size: 0.8rem; }
.btn-clear:hover { color: #C00000; }
.btn-resolve { background: #1F4E79; color: #fff; }
.btn-resolve:hover { background: #16395a; }

/* Flags */
.flag-warn { color: #ED7D31; font-weight: 600; }
.flag-danger { color: #C00000; font-weight: 600; }

/* Chart */
.chart-container { margin-bottom: 24px; }

/* Scenario cards */
.scenario-card {
  border: 1px solid #dee2e6;
  border-radius: 6px;
  margin-bottom: 12px;
  overflow: hidden;
}
.scenario-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  background: #f8f9fa;
  cursor: pointer;
  gap: 16px;
}
.scenario-header:hover { background: #e9ecef; }
.scenario-title { flex: 1; }
.scenario-metrics { display: flex; gap: 16px; }
.metric { font-size: 0.82rem; color: #555; }
.metric.danger { color: #C00000; font-weight: 600; }
.chevron { font-size: 1rem; color: #999; }
.scenario-body { padding: 12px 16px; }

.material-header { margin-top: 20px; color: #1F4E79; }
.implication-note { font-size: 0.82rem; color: #856404; background: #fff3cd; padding: 6px 12px; border-radius: 4px; margin: 4px 0; }
.section-desc { color: #666; font-size: 0.85rem; margin-bottom: 16px; }
.validation-group { margin-bottom: 20px; }
.validation-group h4 { color: #1F4E79; margin: 12px 0 8px; }

h3 { color: #1F4E79; margin: 20px 0 12px; font-size: 1.1rem; }

/* Options tab */
.options-group { margin-bottom: 20px; }
.options-group h4 { color: #1F4E79; margin: 12px 0 8px; }
.option-type-badge {
  display: inline-block; padding: 2px 8px; border-radius: 4px;
  font-size: 0.8rem; font-weight: 600; text-transform: capitalize;
}
.option-type-badge.renewal { background: #E2EFDA; color: #375623; }
.option-type-badge.termination { background: #FCE4EC; color: #C00000; }
.row-exercised { background: #f0f7ff; }
.btn-exercised {
  background: #28a745; color: #fff; border: none; border-radius: 4px;
  padding: 2px 10px; cursor: pointer; font-weight: 600;
}
.btn-exercised:hover { background: #218838; }
.btn-not-exercised {
  background: #e9ecef; color: #6c757d; border: none; border-radius: 4px;
  padding: 2px 10px; cursor: pointer;
}
.btn-not-exercised:hover { background: #dee2e6; }
.btn-abstract { background: #1F4E79; color: #fff; font-weight: 600; }
.btn-abstract:hover { background: #16395a; }
</style>
