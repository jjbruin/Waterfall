<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import api from '../api/client'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { BarChart } from 'echarts/charts'
import {
  GridComponent, TooltipComponent, LegendComponent,
  MarkLineComponent,
} from 'echarts/components'

use([CanvasRenderer, BarChart, GridComponent, TooltipComponent, LegendComponent, MarkLineComponent])

const route = useRoute()

const CLR_DARK = '#1F4E79'
const CLR_ACCENT = '#ED7D31'
const CLR_GREEN = '#548235'
const CLR_RED = '#C00000'

// State
const reviews = ref<any[]>([])
const selectedReviewId = ref<number | null>(null)
const review = ref<any>(null)
const tenants = ref<any[]>([])
const expirations = ref<any>(null)
const cotenancy = ref<any>(null)
const scenarios = ref<any[]>([])
const validation = ref<any[]>([])
const loading = ref(false)
const activeTab = ref('overview')
const expandedTenant = ref<number | null>(null)
const tenantDocs = ref<any[]>([])
const expandedScenario = ref<string | null>(null)

// New review creation
const showNewReview = ref(false)
const newReviewName = ref('')
const newReviewAddress = ref('')
const newReviewGla = ref<number | null>(null)
const creatingReview = ref(false)

// Rent roll upload
const uploadingRentRoll = ref(false)
const uploadMessage = ref('')

// Load
onMounted(async () => {
  const res = await api.get('/api/lease-review/reviews')
  reviews.value = res.data

  // Honor ?id= query param (from pipeline navigation)
  const qid = Number(route.query.id)
  if (qid && reviews.value.some(r => r.id === qid)) {
    selectedReviewId.value = qid
    await loadReview(qid)
  } else if (reviews.value.length) {
    selectedReviewId.value = reviews.value[0].id
    await loadReview(reviews.value[0].id)
  }
})

async function loadReview(id: number) {
  loading.value = true
  try {
    const revRes = await api.get(`/api/lease-review/reviews/${id}`)
    review.value = revRes.data.review
    tenants.value = revRes.data.tenants

    // Load secondary data — these may fail if no tenants yet
    if (tenants.value.length) {
      try {
        const [expRes, cotRes, scenRes, valRes] = await Promise.all([
          api.get(`/api/lease-review/reviews/${id}/expirations`),
          api.get(`/api/lease-review/reviews/${id}/cotenancy`),
          api.get(`/api/lease-review/reviews/${id}/scenarios`),
          api.get(`/api/lease-review/reviews/${id}/validation`),
        ])
        expirations.value = expRes.data
        const cotData = cotRes.data
        const clauses: any[] = []
        if (cotData.details) {
          for (const [tenantName, detail] of Object.entries(cotData.details) as any) {
            clauses.push({
              tenant_name: tenantName,
              ...detail,
              trigger_description: detail.trigger,
              alt_rent_formula: detail.alt_rent,
              cure_period_days: detail.cure_days,
              named_cotenants: cotData.forward?.[tenantName] || [],
            })
          }
        }
        cotenancy.value = { ...cotData, clauses }
        scenarios.value = scenRes.data.scenarios || []
        validation.value = valRes.data
      } catch (e2: any) {
        console.warn('Secondary data load error (expected for new reviews)', e2)
      }
    } else {
      expirations.value = null
      cotenancy.value = null
      scenarios.value = []
      validation.value = []
    }
  } catch (e: any) {
    console.error('Load error', e)
  } finally {
    loading.value = false
  }
}

async function onReviewChange() {
  if (selectedReviewId.value) await loadReview(selectedReviewId.value)
}

async function toggleTenantDocs(tid: number) {
  if (expandedTenant.value === tid) {
    expandedTenant.value = null
    return
  }
  expandedTenant.value = tid
  const res = await api.get(`/api/lease-review/reviews/${selectedReviewId.value}/tenants/${tid}/documents`)
  tenantDocs.value = res.data
}

async function downloadExcel() {
  if (!selectedReviewId.value) return
  const res = await api.get(`/api/lease-review/reviews/${selectedReviewId.value}/excel`, { responseType: 'blob' })
  const url = URL.createObjectURL(res.data)
  const a = document.createElement('a')
  a.href = url
  a.download = `Lease_Review_${review.value?.property_name?.replace(/ /g, '_')}.xlsx`
  a.click()
  URL.revokeObjectURL(url)
}

async function createNewReview() {
  if (!newReviewName.value.trim()) return
  creatingReview.value = true
  try {
    const res = await api.post('/api/lease-review/reviews/create', {
      property_name: newReviewName.value.trim(),
      property_address: newReviewAddress.value.trim(),
      total_gla: newReviewGla.value || 0,
    })
    // Reload reviews list and select the new one
    const listRes = await api.get('/api/lease-review/reviews')
    reviews.value = listRes.data
    selectedReviewId.value = res.data.review_id
    await loadReview(res.data.review_id)
    showNewReview.value = false
    newReviewName.value = ''
    newReviewAddress.value = ''
    newReviewGla.value = null
  } catch (e: any) {
    console.error('Create review error', e)
    alert(e.response?.data?.error || 'Failed to create review')
  } finally {
    creatingReview.value = false
  }
}

async function onRentRollUpload(event: Event) {
  const input = event.target as HTMLInputElement
  if (!input.files?.length || !selectedReviewId.value) return

  const file = input.files[0]
  const formData = new FormData()
  formData.append('file', file)

  uploadingRentRoll.value = true
  uploadMessage.value = ''
  try {
    const res = await api.post(
      `/api/lease-review/reviews/${selectedReviewId.value}/upload-rent-roll`,
      formData,
      { headers: { 'Content-Type': 'multipart/form-data' } }
    )
    uploadMessage.value = `Imported ${res.data.tenant_count} tenants (${(res.data.total_gla || 0).toLocaleString()} SF)`
    // Reload the review data
    await loadReview(selectedReviewId.value!)
  } catch (e: any) {
    uploadMessage.value = ''
    console.error('Upload error', e)
    alert(e.response?.data?.error || 'Failed to upload rent roll')
  } finally {
    uploadingRentRoll.value = false
    input.value = ''
  }
}

// Computed
const occupiedTenants = computed(() => tenants.value.filter(t => !t.is_vacant))
const vacantSuites = computed(() => tenants.value.filter(t => t.is_vacant))
const materialTenants = computed(() => occupiedTenants.value.filter(t => t.is_material))
const cotenancyTenants = computed(() => occupiedTenants.value.filter(t => t.has_cotenancy))
const extractedCount = computed(() => occupiedTenants.value.filter(t => t.extraction_status === 'extracted').length)

const valSummary = computed(() => {
  const bySource: Record<string, { match: number; minor: number; mismatch: number; review: number; pending: number }> = {}
  for (const v of validation.value) {
    const src = v.source_type || 'rent_roll'
    if (!bySource[src]) bySource[src] = { match: 0, minor: 0, mismatch: 0, review: 0, pending: 0 }
    bySource[src][v.status as keyof typeof bySource[typeof src]]++
  }
  return bySource
})

const annualRentValidation = computed(() =>
  validation.value.filter(v => v.field === 'annual_rent' && v.source_type === 'rent_roll')
)

// Expiration chart
const expChartOption = computed(() => {
  if (!expirations.value?.yearly_data) return null
  const data = expirations.value.yearly_data.filter((y: any) => y.tenant_count > 0 || y.year <= 2036)
  return {
    tooltip: {
      trigger: 'axis',
      formatter: (params: any) => {
        const p = params[0]
        const yr = data[p.dataIndex]
        return `<b>${yr.year}</b><br/>` +
          `Expiring Rent: $${(yr.expiring_rent / 1000).toFixed(0)}K<br/>` +
          `Expiring SF: ${yr.expiring_sf.toLocaleString()}<br/>` +
          `% of Total: ${yr.pct_of_total_rent.toFixed(1)}%<br/>` +
          `Tenants: ${yr.tenant_count}`
      },
    },
    grid: { left: 80, right: 30, top: 40, bottom: 40 },
    xAxis: { type: 'category', data: data.map((y: any) => y.year), name: 'Year' },
    yAxis: { type: 'value', name: 'Annual Rent ($)', axisLabel: { formatter: (v: number) => '$' + (v / 1000).toFixed(0) + 'K' } },
    series: [{
      type: 'bar',
      data: data.map((y: any) => ({
        value: y.expiring_rent,
        itemStyle: { color: y.pct_of_total_rent > 15 ? CLR_ACCENT : CLR_DARK },
      })),
      label: {
        show: true,
        position: 'top',
        formatter: (p: any) => data[p.dataIndex].tenant_count > 0 ? data[p.dataIndex].tenant_count + '' : '',
        fontSize: 10,
        color: '#666',
      },
    }],
  }
})

// Risk chart (rent at risk by departing tenant)
const riskChartOption = computed(() => {
  if (!cotenancy.value?.rent_at_risk) return null
  const entries = Object.entries(cotenancy.value.rent_at_risk)
    .map(([name, risk]: [string, any]) => ({ name, ...risk }))
    .sort((a: any, b: any) => b.total_dependent_rent - a.total_dependent_rent)
  if (!entries.length) return null
  return {
    tooltip: {
      trigger: 'axis',
      formatter: (params: any) => {
        const p = params[0]
        const e = entries[p.dataIndex]
        return `<b>If ${e.name} departs:</b><br/>` +
          `Rent at Risk: $${(e.total_dependent_rent / 1000).toFixed(0)}K<br/>` +
          `Tenants Affected: ${e.dependent_count}<br/>` +
          `Can Terminate: ${e.termination_eligible_count}`
      },
    },
    grid: { left: 120, right: 60, top: 30, bottom: 30 },
    xAxis: { type: 'value', name: 'Rent at Risk ($)', axisLabel: { formatter: (v: number) => '$' + (v / 1000).toFixed(0) + 'K' } },
    yAxis: { type: 'category', data: entries.map((e: any) => e.name), inverse: true },
    series: [{
      type: 'bar',
      data: entries.map((e: any) => ({
        value: e.total_dependent_rent,
        itemStyle: { color: e.termination_eligible_count > 0 ? CLR_RED : CLR_ACCENT },
      })),
      label: {
        show: true,
        position: 'right',
        formatter: (p: any) => {
          const e = entries[p.dataIndex]
          return `${e.dependent_count} tenants, ${e.termination_eligible_count} can terminate`
        },
        fontSize: 10,
      },
    }],
  }
})

function fmtCurrency(val: number | null): string {
  if (val == null) return '\u2014'
  return '$' + val.toLocaleString('en-US', { maximumFractionDigits: 0 })
}
function fmtSF(val: number | null): string {
  if (val == null) return '\u2014'
  return val.toLocaleString('en-US', { maximumFractionDigits: 0 })
}
function fmtPct(val: number | null): string {
  if (val == null) return '\u2014'
  return val.toFixed(1) + '%'
}
function fmtDate(val: string | null): string {
  if (!val) return '\u2014'
  return val
}
function statusClass(s: string): string {
  if (s === 'match') return 'status-match'
  if (s === 'mismatch') return 'status-mismatch'
  if (s === 'minor') return 'status-minor'
  if (s === 'review') return 'status-review'
  return 'status-pending'
}
</script>

<template>
  <div class="lease-review-page">
    <!-- Header -->
    <div class="page-header">
      <div class="header-left">
        <h1>Lease Review</h1>
        <select v-if="reviews.length" v-model="selectedReviewId" @change="onReviewChange" class="review-select">
          <option v-for="r in reviews" :key="r.id" :value="r.id">{{ r.property_name }}</option>
        </select>
        <span v-else-if="review" class="property-name">{{ review.property_name }}</span>
      </div>
      <div class="header-right">
        <button class="btn-new" @click="showNewReview = true">+ New Review</button>
        <button class="btn-excel" @click="downloadExcel" :disabled="!selectedReviewId || !tenants.length">
          Download Excel
        </button>
      </div>
    </div>

    <!-- New Review Modal -->
    <div v-if="showNewReview" class="modal-overlay" @click.self="showNewReview = false">
      <div class="modal-box">
        <h3>New Lease Review</h3>
        <div class="form-field">
          <label>Property Name *</label>
          <input v-model="newReviewName" placeholder="e.g. Windsor Square" />
        </div>
        <div class="form-field">
          <label>Address</label>
          <input v-model="newReviewAddress" placeholder="e.g. Matthews, NC" />
        </div>
        <div class="form-field">
          <label>Total GLA (SF)</label>
          <input v-model.number="newReviewGla" type="number" placeholder="0" />
        </div>
        <div class="modal-actions">
          <button class="btn-cancel" @click="showNewReview = false">Cancel</button>
          <button class="btn-primary" @click="createNewReview" :disabled="!newReviewName.trim() || creatingReview">
            {{ creatingReview ? 'Creating...' : 'Create Review' }}
          </button>
        </div>
      </div>
    </div>

    <div v-if="loading" class="loading">Loading lease review data...</div>

    <!-- Empty state -->
    <div v-if="!loading && !reviews.length" class="empty-state">
      <div class="empty-icon">&#128196;</div>
      <h3>No Lease Reviews Yet</h3>
      <p>
        Create a lease review to get started, or use the <strong>Pipeline</strong> tab
        to create one linked to a deal.
      </p>
      <button class="btn-primary" style="margin-top: 1rem" @click="showNewReview = true">+ New Review</button>
    </div>

    <!-- Rent roll upload bar (when review exists but no tenants) -->
    <div v-if="review && !loading && !tenants.length" class="upload-bar">
      <div class="upload-prompt">
        <strong>{{ review.property_name }}</strong> has no tenant data yet.
        Upload a rent roll to populate tenants.
      </div>
      <label class="btn-upload">
        {{ uploadingRentRoll ? 'Uploading...' : 'Upload Rent Roll' }}
        <input type="file" accept=".xlsx,.xls,.csv" @change="onRentRollUpload" :disabled="uploadingRentRoll" hidden />
      </label>
    </div>

    <template v-if="review && !loading">
      <!-- KPI Cards -->
      <div class="kpi-row">
        <div class="kpi-card">
          <div class="kpi-label">Total GLA</div>
          <div class="kpi-value">{{ fmtSF(review.total_gla) }} SF</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Annual Rent</div>
          <div class="kpi-value">{{ fmtCurrency(review.total_annual_rent) }}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Tenants</div>
          <div class="kpi-value">{{ occupiedTenants.length }} <span class="kpi-sub">/ {{ vacantSuites.length }} vacant</span></div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Material Leases</div>
          <div class="kpi-value">{{ materialTenants.length }}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Co-Tenancy Clauses</div>
          <div class="kpi-value">{{ cotenancyTenants.length }}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Extracted</div>
          <div class="kpi-value">{{ extractedCount }} / {{ occupiedTenants.length }}</div>
        </div>
      </div>

      <!-- Tabs -->
      <div class="tabs">
        <button :class="{ active: activeTab === 'overview' }" @click="activeTab = 'overview'">Overview</button>
        <button :class="{ active: activeTab === 'expirations' }" @click="activeTab = 'expirations'">Lease Expirations</button>
        <button :class="{ active: activeTab === 'validation' }" @click="activeTab = 'validation'">Validation</button>
        <button :class="{ active: activeTab === 'cotenancy' }" @click="activeTab = 'cotenancy'">Co-Tenancy Risk</button>
        <button :class="{ active: activeTab === 'scenarios' }" @click="activeTab = 'scenarios'">Scenario Analysis</button>
      </div>

      <!-- TAB: Overview (Tenant Roster) -->
      <div v-if="activeTab === 'overview'" class="tab-content">
        <div class="section-header-row">
          <h2>Tenant Roster</h2>
          <div class="section-actions">
            <span v-if="uploadMessage" class="upload-msg">{{ uploadMessage }}</span>
            <label class="btn-upload-sm">
              {{ uploadingRentRoll ? 'Uploading...' : 'Upload Rent Roll' }}
              <input type="file" accept=".xlsx,.xls,.csv" @change="onRentRollUpload" :disabled="uploadingRentRoll" hidden />
            </label>
          </div>
        </div>
        <div class="table-scroll">
          <table class="data-table">
            <thead>
              <tr>
                <th>Tenant</th>
                <th>Suite</th>
                <th class="r">SF</th>
                <th>Start</th>
                <th>End</th>
                <th class="r">Annual Rent</th>
                <th class="r">$/SF</th>
                <th class="c">Material</th>
                <th class="c">Co-Ten</th>
                <th class="c">Status</th>
                <th class="c">Docs</th>
              </tr>
            </thead>
            <tbody>
              <template v-for="t in occupiedTenants" :key="t.id">
                <tr :class="{ 'row-material': t.is_material, 'row-cotenancy': t.has_cotenancy }" @click="toggleTenantDocs(t.id)" style="cursor:pointer">
                  <td class="tenant-name">{{ t.tenant_name }}</td>
                  <td>{{ t.suite }}</td>
                  <td class="r">{{ fmtSF(t.square_feet) }}</td>
                  <td>{{ fmtDate(t.lease_start) }}</td>
                  <td>{{ fmtDate(t.lease_end) }}</td>
                  <td class="r">{{ fmtCurrency(t.annual_rent) }}</td>
                  <td class="r">{{ t.rent_per_sf?.toFixed(2) ?? '\u2014' }}</td>
                  <td class="c">{{ t.is_material ? 'Yes' : '' }}</td>
                  <td class="c">{{ t.has_cotenancy ? 'Yes' : '' }}</td>
                  <td class="c"><span :class="'badge badge-' + t.extraction_status">{{ t.extraction_status }}</span></td>
                  <td class="c">{{ t.documents.extracted }}/{{ t.documents.total }}</td>
                </tr>
                <!-- Expanded docs row -->
                <tr v-if="expandedTenant === t.id" class="doc-row">
                  <td colspan="11">
                    <div class="doc-list">
                      <div v-for="d in tenantDocs" :key="d.id" class="doc-item">
                        <span class="doc-type">{{ d.doc_type }}</span>
                        <span class="doc-name">{{ d.filename }}</span>
                        <span class="doc-pages">{{ d.page_count ? d.page_count + ' pg' : '' }}</span>
                        <span :class="'badge badge-' + d.extraction_status">{{ d.extraction_status }}</span>
                      </div>
                      <div v-if="!tenantDocs.length" class="doc-empty">No documents cataloged</div>
                    </div>
                  </td>
                </tr>
              </template>
            </tbody>
          </table>
        </div>

        <div v-if="vacantSuites.length" style="margin-top:1.5rem">
          <h3>Vacant Suites ({{ vacantSuites.length }})</h3>
          <div class="table-scroll">
            <table class="data-table compact">
              <thead><tr><th>Suite</th><th class="r">SF</th></tr></thead>
              <tbody>
                <tr v-for="v in vacantSuites" :key="v.id">
                  <td>{{ v.suite }}</td><td class="r">{{ fmtSF(v.square_feet) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <!-- TAB: Lease Expirations -->
      <div v-if="activeTab === 'expirations'" class="tab-content">
        <h2>Lease Expiration Schedule</h2>
        <div v-if="expChartOption" class="chart-container">
          <v-chart :option="expChartOption" style="height:350px" autoresize />
        </div>

        <div v-if="expirations?.yearly_data" class="table-scroll" style="margin-top:1rem">
          <table class="data-table">
            <thead>
              <tr>
                <th>Year</th><th class="r">Expiring SF</th><th class="r">Expiring Rent</th>
                <th class="r">% of Total</th><th class="r">Avg $/SF</th><th class="r">Tenants</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="y in expirations.yearly_data.filter((y: any) => y.tenant_count > 0 || y.year <= 2036)" :key="y.year"
                  :class="{ 'row-heavy': y.pct_of_total_rent > 15 }">
                <td>{{ y.year }}</td>
                <td class="r">{{ fmtSF(y.expiring_sf) }}</td>
                <td class="r">{{ fmtCurrency(y.expiring_rent) }}</td>
                <td class="r">{{ fmtPct(y.pct_of_total_rent) }}</td>
                <td class="r">{{ y.avg_rent_per_sf > 0 ? '$' + y.avg_rent_per_sf.toFixed(2) : '\u2014' }}</td>
                <td class="r">{{ y.tenant_count }}</td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- Material Leases by Year -->
        <div v-if="expirations?.material_leases" style="margin-top:2rem">
          <h2>Material Leases Maturing by Year</h2>
          <div v-for="(leases, year) in expirations.material_leases" :key="year" style="margin-bottom:1rem">
            <h3>{{ year }}</h3>
            <div class="table-scroll">
              <table class="data-table compact">
                <thead>
                  <tr><th>Tenant</th><th>Suite</th><th class="r">SF</th><th class="r">Annual Rent</th><th class="r">$/SF</th><th>Co-Tenancy</th></tr>
                </thead>
                <tbody>
                  <tr v-for="l in leases" :key="l.tenant_name" :class="{ 'row-cotenancy': l.has_cotenancy }">
                    <td>{{ l.tenant_name }}</td>
                    <td>{{ l.suite }}</td>
                    <td class="r">{{ fmtSF(l.square_feet) }}</td>
                    <td class="r">{{ fmtCurrency(l.annual_rent) }}</td>
                    <td class="r">${{ l.rent_per_sf?.toFixed(2) }}</td>
                    <td>{{ l.cotenancy_implication || '\u2014' }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <!-- TAB: Validation -->
      <div v-if="activeTab === 'validation'" class="tab-content">
        <h2>Seller Document Validation vs Lease Terms</h2>
        <p class="subtitle">Lease PDFs are the authoritative source. Rent Roll and Argus are seller representations validated against actual lease terms.</p>

        <!-- Validation summary cards -->
        <div class="val-summary">
          <div v-for="(stats, source) in valSummary" :key="source" class="val-card">
            <div class="val-card-title">{{ source === 'rent_roll' ? 'Rent Roll' : source === 'argus' ? 'Argus' : 'Co-Tenancy Schedule' }}</div>
            <div class="val-stats">
              <span class="status-match">{{ stats.match }} match</span>
              <span class="status-minor" v-if="stats.minor">{{ stats.minor }} minor</span>
              <span class="status-mismatch" v-if="stats.mismatch">{{ stats.mismatch }} mismatch</span>
              <span class="status-review" v-if="stats.review">{{ stats.review }} review</span>
            </div>
          </div>
        </div>

        <!-- Annual rent comparison table -->
        <h3 style="margin-top:1.5rem">Annual Rent: Rent Roll vs Lease</h3>
        <div class="table-scroll">
          <table class="data-table">
            <thead>
              <tr><th>Tenant</th><th>Suite</th><th class="r">Rent Roll</th><th class="r">Lease</th><th class="c">Status</th><th>Notes</th></tr>
            </thead>
            <tbody>
              <tr v-for="v in annualRentValidation" :key="v.tenant + v.suite" :class="statusClass(v.status)">
                <td>{{ v.tenant }}</td>
                <td>{{ v.suite }}</td>
                <td class="r">{{ v.seller_value ? fmtCurrency(parseFloat(v.seller_value)) : '\u2014' }}</td>
                <td class="r">{{ v.lease_value ? fmtCurrency(parseFloat(v.lease_value)) : '\u2014' }}</td>
                <td class="c"><span :class="'badge badge-' + v.status">{{ v.status }}</span></td>
                <td class="notes">{{ v.notes || '' }}</td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- Full validation detail -->
        <h3 style="margin-top:1.5rem">All Validation Comparisons ({{ validation.length }})</h3>
        <div class="table-scroll">
          <table class="data-table compact">
            <thead>
              <tr><th>Tenant</th><th>Suite</th><th>Source</th><th>Field</th><th class="r">Seller</th><th class="r">Lease</th><th class="c">Status</th></tr>
            </thead>
            <tbody>
              <tr v-for="(v, i) in validation" :key="i" :class="statusClass(v.status)">
                <td>{{ v.tenant }}</td>
                <td>{{ v.suite }}</td>
                <td>{{ v.source_type }}</td>
                <td>{{ v.field }}</td>
                <td class="r">{{ v.seller_value ?? '\u2014' }}</td>
                <td class="r">{{ v.lease_value ?? '\u2014' }}</td>
                <td class="c"><span :class="'badge badge-' + v.status">{{ v.status }}</span></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- TAB: Co-Tenancy Risk -->
      <div v-if="activeTab === 'cotenancy'" class="tab-content">
        <h2>Co-Tenancy Risk Analysis</h2>

        <div v-if="riskChartOption" class="chart-container">
          <v-chart :option="riskChartOption" style="height:300px" autoresize />
        </div>

        <!-- Co-tenancy clause details -->
        <div v-if="cotenancy?.clauses?.length" style="margin-top:1.5rem">
          <h3>Clause Details</h3>
          <div class="table-scroll">
            <table class="data-table">
              <thead>
                <tr>
                  <th>Tenant</th><th>Suite</th><th class="r">Annual Rent</th>
                  <th>Trigger</th><th>Cure</th><th>Alt Rent</th>
                  <th class="c">Terminate?</th><th class="c">Curable?</th>
                  <th>Named Co-Tenants</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="c in cotenancy.clauses" :key="c.tenant_name" :class="{ 'row-uncurable': !c.is_curable }">
                  <td class="tenant-name">{{ c.tenant_name }}</td>
                  <td>{{ c.suite }}</td>
                  <td class="r">{{ fmtCurrency(c.annual_rent) }}</td>
                  <td class="wrap">{{ c.trigger_description || '\u2014' }}</td>
                  <td>{{ c.cure_period_days != null ? c.cure_period_days + 'd' : '\u2014' }}</td>
                  <td class="wrap">{{ c.alt_rent_formula || '\u2014' }}</td>
                  <td class="c">{{ c.termination_right ? 'Yes' : 'No' }}</td>
                  <td class="c"><span :class="c.is_curable ? '' : 'uncurable'">{{ c.is_curable ? 'Yes' : 'UNCURABLE' }}</span></td>
                  <td class="wrap">{{ (c.named_cotenants || []).join(', ') || '\u2014' }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- Departing tenant risk detail -->
        <div v-if="cotenancy?.rent_at_risk" style="margin-top:1.5rem">
          <h3>Departing Tenant Impact</h3>
          <div class="table-scroll">
            <table class="data-table">
              <thead>
                <tr><th>Departing Tenant</th><th class="r">Affected</th><th class="r">Rent at Risk</th><th class="r">Can Terminate</th></tr>
              </thead>
              <tbody>
                <tr v-for="(risk, name) in cotenancy.rent_at_risk" :key="name">
                  <td class="tenant-name">{{ name }}</td>
                  <td class="r">{{ risk.dependent_count }}</td>
                  <td class="r">{{ fmtCurrency(risk.total_dependent_rent) }}</td>
                  <td class="r">{{ risk.termination_eligible_count }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <!-- TAB: Scenario Analysis -->
      <div v-if="activeTab === 'scenarios'" class="tab-content">
        <h2>Cascading Scenario Analysis</h2>
        <p class="subtitle">Models the impact of each anchor tenant departing on co-tenancy clauses across the property.</p>

        <div v-for="s in scenarios" :key="s.departing_tenant" class="scenario-card">
          <div class="scenario-header" @click="expandedScenario = expandedScenario === s.departing_tenant ? null : s.departing_tenant">
            <div class="scenario-title">
              <span class="scenario-icon">{{ expandedScenario === s.departing_tenant ? '\u25BC' : '\u25B6' }}</span>
              If <strong>{{ s.departing_tenant }}</strong> departs
            </div>
            <div class="scenario-summary">
              <span class="scenario-metric">{{ fmtCurrency(s.total_dependent_rent) }} at risk</span>
              <span class="scenario-metric">{{ s.termination_eligible }} can terminate</span>
            </div>
          </div>
          <div v-if="expandedScenario === s.departing_tenant" class="scenario-detail">
            <table class="data-table compact">
              <thead>
                <tr><th>Tenant</th><th class="r">Annual Rent</th><th>Alt Rent Formula</th><th>Cure</th><th>Termination</th></tr>
              </thead>
              <tbody>
                <tr v-for="imp in s.impacts" :key="imp.tenant" :class="{ 'row-uncurable': !imp.is_curable }">
                  <td>{{ imp.tenant }}</td>
                  <td class="r">{{ fmtCurrency(imp.annual_rent) }}</td>
                  <td>{{ imp.alt_rent_formula || '\u2014' }}</td>
                  <td>{{ !imp.is_curable ? 'UNCURABLE' : (imp.cure_days ? imp.cure_days + 'd' : '\u2014') }}</td>
                  <td>{{ imp.can_terminate ? 'CAN TERMINATE' : 'No' }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
        <div v-if="!scenarios.length" class="empty">No scenario data available.</div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.lease-review-page {
  padding: 1.5rem;
  max-width: 1400px;
}
.empty-state {
  text-align: center;
  padding: 60px 20px;
  color: #666;
}
.empty-state .empty-icon {
  font-size: 48px;
  margin-bottom: 12px;
}
.empty-state h3 {
  margin: 0 0 8px;
  color: #333;
}
.empty-state p {
  max-width: 400px;
  margin: 0 auto;
  line-height: 1.5;
}
.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
}
.header-left { display: flex; align-items: center; gap: 1rem; }
.header-left h1 { margin: 0; font-size: 1.5rem; color: #1F4E79; }
.property-name { font-size: 1.1rem; color: #555; }
.review-select { padding: 0.4rem 0.8rem; border: 1px solid #ccc; border-radius: 4px; font-size: 0.9rem; }
.btn-excel {
  padding: 0.5rem 1rem; background: #1F4E79; color: #fff; border: none;
  border-radius: 4px; cursor: pointer; font-size: 0.85rem;
}
.btn-excel:hover { background: #163a5c; }
.btn-excel:disabled { opacity: 0.5; cursor: default; }
.loading { text-align: center; padding: 3rem; color: #888; }
.empty { text-align: center; padding: 2rem; color: #888; }

/* KPI cards */
.kpi-row { display: flex; gap: 0.75rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.kpi-card {
  flex: 1; min-width: 140px; padding: 0.75rem 1rem;
  background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 6px;
}
.kpi-label { font-size: 0.75rem; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }
.kpi-value { font-size: 1.25rem; font-weight: 600; color: #1F4E79; margin-top: 0.25rem; }
.kpi-sub { font-size: 0.8rem; color: #888; font-weight: 400; }

/* Tabs */
.tabs {
  display: flex; gap: 0; border-bottom: 2px solid #e0e0e0; margin-bottom: 1.5rem;
}
.tabs button {
  padding: 0.6rem 1.2rem; border: none; background: none; cursor: pointer;
  font-size: 0.85rem; color: #666; border-bottom: 2px solid transparent;
  margin-bottom: -2px; transition: all 0.15s;
}
.tabs button:hover { color: #1F4E79; }
.tabs button.active { color: #1F4E79; border-bottom-color: #1F4E79; font-weight: 600; }

.tab-content h2 { margin: 0 0 0.75rem; font-size: 1.15rem; color: #1F4E79; }
.tab-content h3 { margin: 0 0 0.5rem; font-size: 0.95rem; color: #333; }
.subtitle { color: #666; font-size: 0.85rem; margin: -0.5rem 0 1rem; }

/* Tables */
.table-scroll { overflow-x: auto; }
.data-table {
  width: 100%; border-collapse: collapse; font-size: 0.82rem;
}
.data-table th {
  background: #1F4E79; color: #fff; padding: 0.5rem 0.6rem;
  text-align: left; font-weight: 500; white-space: nowrap;
  position: sticky; top: 0;
}
.data-table td {
  padding: 0.4rem 0.6rem; border-bottom: 1px solid #eee;
}
.data-table tbody tr:hover { background: #f5f8fc; }
.data-table.compact td { padding: 0.3rem 0.5rem; }
.r { text-align: right; }
.c { text-align: center; }
.wrap { max-width: 220px; white-space: normal; word-break: break-word; }
.tenant-name { font-weight: 500; }
.notes { font-size: 0.78rem; color: #666; max-width: 250px; white-space: normal; }

.row-material { background: #fafafa; }
.row-cotenancy td:first-child { border-left: 3px solid #ED7D31; }
.row-heavy { background: #fff8f0; }
.row-uncurable { background: #fff0f0; }

/* Badges */
.badge {
  display: inline-block; padding: 0.15rem 0.5rem; border-radius: 10px;
  font-size: 0.72rem; font-weight: 500;
}
.badge-extracted { background: #c6efce; color: #006100; }
.badge-pending { background: #ffeb9c; color: #9c6500; }
.badge-error { background: #ffc7ce; color: #9c0006; }
.badge-match { background: #c6efce; color: #006100; }
.badge-mismatch { background: #ffc7ce; color: #9c0006; }
.badge-minor { background: #ffeb9c; color: #9c6500; }
.badge-review { background: #b4d4f0; color: #1F4E79; }
.badge-text_extracted { background: #e0e0e0; color: #333; }

/* Validation status row colors */
.status-match { }
.status-mismatch td { background: #fff5f5; }
.status-minor td { background: #fffcf0; }
.status-review td { background: #f0f6ff; }

/* Validation summary */
.val-summary { display: flex; gap: 1rem; margin-bottom: 1rem; flex-wrap: wrap; }
.val-card {
  flex: 1; min-width: 200px; padding: 0.75rem 1rem;
  border: 1px solid #e0e0e0; border-radius: 6px; background: #f8f9fa;
}
.val-card-title { font-weight: 600; color: #1F4E79; margin-bottom: 0.4rem; font-size: 0.85rem; }
.val-stats { display: flex; gap: 0.5rem; flex-wrap: wrap; }
.val-stats span { font-size: 0.78rem; padding: 0.1rem 0.4rem; border-radius: 8px; }

/* Documents expansion */
.doc-row td { background: #f8f9fc; padding: 0; }
.doc-list { padding: 0.5rem 1rem 0.5rem 2rem; }
.doc-item {
  display: flex; gap: 0.75rem; align-items: center; padding: 0.25rem 0;
  font-size: 0.8rem; border-bottom: 1px solid #eee;
}
.doc-type { font-weight: 500; min-width: 120px; color: #1F4E79; }
.doc-name { flex: 1; color: #555; }
.doc-pages { color: #888; min-width: 40px; }
.doc-empty { color: #aaa; font-size: 0.8rem; }

/* Chart */
.chart-container { border: 1px solid #e0e0e0; border-radius: 6px; padding: 0.5rem; background: #fff; }

/* Scenario cards */
.scenario-card {
  border: 1px solid #e0e0e0; border-radius: 6px; margin-bottom: 0.75rem;
  overflow: hidden;
}
.scenario-header {
  display: flex; justify-content: space-between; align-items: center;
  padding: 0.75rem 1rem; cursor: pointer; background: #f8f9fa;
}
.scenario-header:hover { background: #f0f3f7; }
.scenario-title { font-size: 0.9rem; }
.scenario-icon { margin-right: 0.5rem; font-size: 0.7rem; }
.scenario-summary { display: flex; gap: 1.5rem; }
.scenario-metric { font-size: 0.82rem; color: #666; }
.scenario-detail { padding: 0 1rem 0.75rem; }

.uncurable { color: #9c0006; font-weight: 700; }

/* New review / upload */
.btn-new {
  padding: 0.5rem 1rem; background: #548235; color: #fff; border: none;
  border-radius: 4px; cursor: pointer; font-size: 0.85rem; margin-right: 0.5rem;
}
.btn-new:hover { background: #3d6127; }
.btn-primary {
  padding: 0.5rem 1rem; background: #1F4E79; color: #fff; border: none;
  border-radius: 4px; cursor: pointer; font-size: 0.85rem;
}
.btn-primary:hover { background: #163a5c; }
.btn-primary:disabled { opacity: 0.5; cursor: default; }
.btn-cancel {
  padding: 0.5rem 1rem; background: #e0e0e0; color: #333; border: none;
  border-radius: 4px; cursor: pointer; font-size: 0.85rem;
}

/* Modal */
.modal-overlay {
  position: fixed; inset: 0; background: rgba(0,0,0,0.4); z-index: 1000;
  display: flex; align-items: center; justify-content: center;
}
.modal-box {
  background: #fff; border-radius: 8px; padding: 1.5rem; width: 400px;
  max-width: 90vw; box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}
.modal-box h3 { margin: 0 0 1rem; color: #1F4E79; font-size: 1.1rem; }
.form-field { margin-bottom: 0.75rem; }
.form-field label { display: block; font-size: 0.8rem; color: #555; margin-bottom: 0.25rem; }
.form-field input {
  width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px;
  font-size: 0.9rem; box-sizing: border-box;
}
.modal-actions { display: flex; justify-content: flex-end; gap: 0.5rem; margin-top: 1rem; }

/* Upload bar */
.upload-bar {
  display: flex; align-items: center; justify-content: space-between;
  padding: 1rem 1.25rem; background: #f0f6ff; border: 1px solid #b4d4f0;
  border-radius: 6px; margin-bottom: 1.5rem;
}
.upload-prompt { font-size: 0.9rem; color: #333; }
.btn-upload, .btn-upload-sm {
  display: inline-block; padding: 0.5rem 1rem; background: #1F4E79; color: #fff;
  border-radius: 4px; cursor: pointer; font-size: 0.85rem;
}
.btn-upload:hover, .btn-upload-sm:hover { background: #163a5c; }
.btn-upload-sm { padding: 0.35rem 0.75rem; font-size: 0.8rem; }

/* Section header row */
.section-header-row {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 0.75rem;
}
.section-header-row h2 { margin: 0; }
.section-actions { display: flex; align-items: center; gap: 0.75rem; }
.upload-msg { font-size: 0.8rem; color: #548235; }
</style>
