<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
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
const expandedTenant = ref<number | null>(null)
const tenantDocs = ref<any[]>([])
const expandedScenario = ref<string | null>(null)

// Workflow stepper
const STEPS = [
  { key: 'setup', label: 'Setup', num: 1 },
  { key: 'rent_roll', label: 'Import Rent Roll', num: 2 },
  { key: 'documents', label: 'Upload Documents', num: 3 },
  { key: 'extraction', label: 'AI Extraction', num: 4 },
  { key: 'validation', label: 'Validation', num: 5 },
  { key: 'review', label: 'Analyst Review', num: 6 },
  { key: 'complete', label: 'Complete', num: 7 },
]
const activeStep = ref('setup')
const progress = ref<any>(null)

// New review creation
const showNewReview = ref(false)
const newReviewName = ref('')
const newReviewAddress = ref('')
const newReviewGla = ref<number | null>(null)
const creatingReview = ref(false)
const prospectProperties = ref<any[]>([])
const selectedProspectPropId = ref<number | null>(null)

// Rent roll upload / merge
const uploadingRentRoll = ref(false)
const mergingRentRoll = ref(false)
const uploadMessage = ref('')
const mergeReport = ref<any>(null)

// Document upload
const uploadingDocs = ref(false)
const docUploadReport = ref<any>(null)

// Extraction
const extracting = ref(false)
const extractionMessage = ref('')

// Validation
const validating = ref(false)

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
    activeStep.value = review.value.workflow_step || 'setup'

    // Load progress
    try {
      const progRes = await api.get(`/api/lease-review/reviews/${id}/progress`)
      progress.value = progRes.data
    } catch { progress.value = null }

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

// Step navigation
function stepIndex(key: string): number {
  return STEPS.findIndex(s => s.key === key)
}

function isStepUnlocked(key: string): boolean {
  // All steps up to and including current step +1 are unlocked
  const current = stepIndex(activeStep.value)
  const target = stepIndex(key)
  return target <= current + 1
}

async function goToStep(key: string) {
  if (!isStepUnlocked(key)) return
  activeStep.value = key
  // Persist to server
  if (selectedReviewId.value) {
    try {
      await api.put(`/api/lease-review/reviews/${selectedReviewId.value}/workflow-step`, { step: key })
    } catch (e) {
      console.warn('Failed to persist step', e)
    }
  }
}

// Tenant docs
async function toggleTenantDocs(tid: number) {
  if (expandedTenant.value === tid) {
    expandedTenant.value = null
    return
  }
  expandedTenant.value = tid
  const res = await api.get(`/api/lease-review/reviews/${selectedReviewId.value}/tenants/${tid}/documents`)
  tenantDocs.value = res.data
}

// Excel download
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

// New review
async function openNewReviewModal() {
  showNewReview.value = true
  try {
    const res = await api.get('/api/lease-review/prospect-properties')
    prospectProperties.value = res.data.filter((p: any) => !p.lease_review_id)
  } catch {
    prospectProperties.value = []
  }
}

function onProspectPropertySelect() {
  const prop = prospectProperties.value.find(p => p.id === selectedProspectPropId.value)
  if (prop) {
    newReviewName.value = prop.property_name || ''
    const addr = [prop.address, prop.city, prop.state].filter(Boolean).join(', ')
    newReviewAddress.value = addr
    newReviewGla.value = prop.gla_sf || null
  }
}

async function createNewReview() {
  if (!newReviewName.value.trim()) return
  creatingReview.value = true
  try {
    const res = await api.post('/api/lease-review/reviews/create', {
      property_name: newReviewName.value.trim(),
      property_address: newReviewAddress.value.trim(),
      total_gla: newReviewGla.value || 0,
      prospect_property_id: selectedProspectPropId.value || undefined,
    })
    const listRes = await api.get('/api/lease-review/reviews')
    reviews.value = listRes.data
    selectedReviewId.value = res.data.review_id
    await loadReview(res.data.review_id)
    showNewReview.value = false
    newReviewName.value = ''
    newReviewAddress.value = ''
    newReviewGla.value = null
    selectedProspectPropId.value = null
  } catch (e: any) {
    console.error('Create review error', e)
    alert(e.response?.data?.error || 'Failed to create review')
  } finally {
    creatingReview.value = false
  }
}

// Destructive rent roll upload (original)
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

// Non-destructive rent roll merge
async function onRentRollMerge(event: Event) {
  const input = event.target as HTMLInputElement
  if (!input.files?.length || !selectedReviewId.value) return

  const file = input.files[0]
  const formData = new FormData()
  formData.append('file', file)

  mergingRentRoll.value = true
  mergeReport.value = null
  uploadMessage.value = ''
  try {
    const res = await api.post(
      `/api/lease-review/reviews/${selectedReviewId.value}/merge-rent-roll`,
      formData,
      { headers: { 'Content-Type': 'multipart/form-data' } }
    )
    mergeReport.value = res.data
    uploadMessage.value = `Merged: ${res.data.matched} updated, ${res.data.added} added, ${res.data.not_in_upload} not in upload`
    await loadReview(selectedReviewId.value!)
  } catch (e: any) {
    console.error('Merge error', e)
    alert(e.response?.data?.error || 'Failed to merge rent roll')
  } finally {
    mergingRentRoll.value = false
    input.value = ''
  }
}

// Document upload
async function onDocumentUpload(event: Event) {
  const input = event.target as HTMLInputElement
  if (!input.files?.length || !selectedReviewId.value) return

  const formData = new FormData()
  const folderHints: string[] = []
  for (const f of input.files) {
    if (!f.name.toLowerCase().endsWith('.pdf')) continue
    formData.append('files', f)
    // webkitRelativePath gives "FolderName/SubFolder/file.pdf"
    // Extract the parent folder name as a tenant matching hint
    const relPath = (f as any).webkitRelativePath || ''
    const parts = relPath.split('/')
    // Use the immediate parent folder (not the root folder selected)
    const hint = parts.length > 2 ? parts[parts.length - 2] : (parts.length === 2 ? parts[0] : '')
    folderHints.push(hint)
  }
  if (!formData.has('files')) {
    alert('No PDF files found in the selection.')
    return
  }
  formData.append('folder_hints', JSON.stringify(folderHints))

  uploadingDocs.value = true
  docUploadReport.value = null
  try {
    const res = await api.post(
      `/api/lease-review/reviews/${selectedReviewId.value}/upload-documents`,
      formData,
      { headers: { 'Content-Type': 'multipart/form-data' } }
    )
    docUploadReport.value = res.data
    await loadReview(selectedReviewId.value!)
  } catch (e: any) {
    console.error('Doc upload error', e)
    alert(e.response?.data?.error || 'Failed to upload documents')
  } finally {
    uploadingDocs.value = false
    input.value = ''
  }
}

// Run extraction
async function runExtraction() {
  if (!selectedReviewId.value) return
  extracting.value = true
  extractionMessage.value = 'Running AI extraction on pending documents...'
  try {
    await api.post(`/api/lease-review/reviews/${selectedReviewId.value}/extract`)
    extractionMessage.value = 'Extraction complete.'
    await loadReview(selectedReviewId.value!)
  } catch (e: any) {
    extractionMessage.value = e.response?.data?.error || 'Extraction failed'
  } finally {
    extracting.value = false
  }
}

// Run validation
async function runValidation() {
  if (!selectedReviewId.value) return
  validating.value = true
  try {
    await api.post(`/api/lease-review/reviews/${selectedReviewId.value}/validate`)
    await loadReview(selectedReviewId.value!)
  } catch (e: any) {
    alert(e.response?.data?.error || 'Validation failed')
  } finally {
    validating.value = false
  }
}

// Tenant approval
async function setTenantApproval(tid: number, status: string) {
  if (!selectedReviewId.value) return
  try {
    await api.put(`/api/lease-review/reviews/${selectedReviewId.value}/tenants/${tid}/approve`, { status })
    // Update local state
    const t = tenants.value.find(x => x.id === tid)
    if (t) t.approval_status = status
  } catch (e: any) {
    alert(e.response?.data?.error || 'Failed to update approval')
  }
}

// Computed
const occupiedTenants = computed(() => tenants.value.filter(t => !t.is_vacant))
const vacantSuites = computed(() => tenants.value.filter(t => t.is_vacant))
const materialTenants = computed(() => occupiedTenants.value.filter(t => t.is_material))
const cotenancyTenants = computed(() => occupiedTenants.value.filter(t => t.has_cotenancy))
const extractedCount = computed(() => occupiedTenants.value.filter(t => t.extraction_status === 'extracted').length)
const approvedCount = computed(() => occupiedTenants.value.filter(t => t.approval_status === 'approved').length)
const flaggedCount = computed(() => occupiedTenants.value.filter(t => t.approval_status === 'flagged').length)

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

// Risk chart
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
        <button class="btn-new" @click="openNewReviewModal">+ New Review</button>
        <button class="btn-excel" @click="downloadExcel" :disabled="!selectedReviewId || !tenants.length">
          Download Excel
        </button>
      </div>
    </div>

    <!-- New Review Modal -->
    <div v-if="showNewReview" class="modal-overlay" @click.self="showNewReview = false">
      <div class="modal-box">
        <h3>New Lease Review</h3>
        <div v-if="prospectProperties.length" class="form-field">
          <label>Link to Pipeline Property</label>
          <select v-model="selectedProspectPropId" @change="onProspectPropertySelect">
            <option :value="null">— Enter manually —</option>
            <option v-for="p in prospectProperties" :key="p.id" :value="p.id">
              {{ p.deal_name }} — {{ p.property_name }}
            </option>
          </select>
        </div>
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
      <button class="btn-primary" style="margin-top: 1rem" @click="openNewReviewModal">+ New Review</button>
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
          <div class="kpi-label">Extracted</div>
          <div class="kpi-value">{{ extractedCount }} / {{ occupiedTenants.length }}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Approved</div>
          <div class="kpi-value">
            {{ approvedCount }} / {{ occupiedTenants.length }}
            <span v-if="flaggedCount" class="kpi-sub kpi-flagged">{{ flaggedCount }} flagged</span>
          </div>
        </div>
      </div>

      <!-- Workflow Stepper -->
      <div class="stepper">
        <div
          v-for="step in STEPS"
          :key="step.key"
          class="step"
          :class="{
            active: activeStep === step.key,
            completed: stepIndex(step.key) < stepIndex(activeStep),
            locked: !isStepUnlocked(step.key),
          }"
          @click="goToStep(step.key)"
        >
          <div class="step-number">
            <span v-if="stepIndex(step.key) < stepIndex(activeStep)" class="step-check">&#10003;</span>
            <span v-else>{{ step.num }}</span>
          </div>
          <div class="step-label">{{ step.label }}</div>
        </div>
      </div>

      <!-- STEP 1: Setup -->
      <div v-if="activeStep === 'setup'" class="step-content">
        <h2>Deal Setup</h2>
        <p class="subtitle">Review created: <strong>{{ review.property_name }}</strong> — {{ review.property_address || 'No address' }}</p>
        <div class="setup-info">
          <div><strong>Created by:</strong> {{ review.created_by }}</div>
          <div><strong>Status:</strong> {{ review.status }}</div>
          <div><strong>Total GLA:</strong> {{ fmtSF(review.total_gla) }} SF</div>
        </div>
        <button class="btn-primary" style="margin-top: 1rem" @click="goToStep('rent_roll')">
          Continue to Import Rent Roll &rarr;
        </button>
      </div>

      <!-- STEP 2: Import Rent Roll -->
      <div v-if="activeStep === 'rent_roll'" class="step-content">
        <h2>Import Seller's Rent Roll</h2>
        <p class="subtitle">Upload the rent roll received from the operating partner. Use <strong>Import (Merge)</strong> to safely update existing data, or <strong>Replace All</strong> to start fresh.</p>

        <div class="upload-actions">
          <label class="btn-primary btn-upload-label">
            {{ mergingRentRoll ? 'Merging...' : 'Import Rent Roll (Merge)' }}
            <input type="file" accept=".xlsx,.xls,.csv,.pdf" @change="onRentRollMerge" :disabled="mergingRentRoll" hidden />
          </label>

          <label class="btn-secondary btn-upload-label">
            {{ uploadingRentRoll ? 'Replacing...' : 'Replace All (Destructive)' }}
            <input type="file" accept=".xlsx,.xls,.csv,.pdf" @change="onRentRollUpload" :disabled="uploadingRentRoll" hidden />
          </label>
        </div>

        <div v-if="uploadMessage" class="upload-msg">{{ uploadMessage }}</div>

        <!-- Merge report -->
        <div v-if="mergeReport" class="merge-report">
          <h3>Merge Results</h3>
          <div class="merge-stats">
            <span class="badge badge-extracted">{{ mergeReport.matched }} updated</span>
            <span class="badge badge-match">{{ mergeReport.added }} added</span>
            <span v-if="mergeReport.not_in_upload" class="badge badge-minor">{{ mergeReport.not_in_upload }} not in upload</span>
          </div>
          <div v-if="mergeReport.not_in_upload_tenants?.length" class="merge-missing">
            <strong>Tenants not in uploaded rent roll:</strong>
            <ul>
              <li v-for="t in mergeReport.not_in_upload_tenants" :key="t.suite">
                {{ t.tenant }} ({{ t.suite }})
              </li>
            </ul>
          </div>
        </div>

        <!-- Tenant roster preview -->
        <div v-if="tenants.length" style="margin-top: 1.5rem">
          <h3>Current Tenant Roster ({{ occupiedTenants.length }} tenants)</h3>
          <div class="table-scroll">
            <table class="data-table compact">
              <thead>
                <tr>
                  <th>Tenant</th><th>Suite</th><th class="r">SF</th>
                  <th class="r">Annual Rent</th><th>Source</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="t in occupiedTenants" :key="t.id">
                  <td class="tenant-name">{{ t.tenant_name }}</td>
                  <td>{{ t.suite }}</td>
                  <td class="r">{{ fmtSF(t.square_feet) }}</td>
                  <td class="r">{{ fmtCurrency(t.annual_rent) }}</td>
                  <td><span class="badge badge-pending">{{ t.rent_roll_source || 'original' }}</span></td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <button class="btn-primary" style="margin-top: 1rem" @click="goToStep('documents')" :disabled="!tenants.length">
          Continue to Upload Documents &rarr;
        </button>
      </div>

      <!-- STEP 3: Upload Documents -->
      <div v-if="activeStep === 'documents'" class="step-content">
        <h2>Upload Lease Documents</h2>
        <p class="subtitle">Upload lease PDFs (Original Lease, Amendments, etc.). Documents are auto-classified and matched to tenants by filename. Duplicates are automatically skipped.</p>

        <div class="upload-actions">
          <label class="btn-primary btn-upload-label">
            {{ uploadingDocs ? 'Uploading...' : 'Select Files' }}
            <input type="file" accept=".pdf" multiple @change="onDocumentUpload" :disabled="uploadingDocs" hidden />
          </label>
          <label class="btn-primary btn-upload-label">
            {{ uploadingDocs ? 'Uploading...' : 'Select Folder' }}
            <input type="file" webkitdirectory @change="onDocumentUpload" :disabled="uploadingDocs" hidden />
          </label>
        </div>

        <!-- Upload report -->
        <div v-if="docUploadReport" class="merge-report">
          <h3>Upload Results</h3>
          <div class="merge-stats">
            <span class="badge badge-extracted">{{ docUploadReport.added }} added</span>
            <span v-if="docUploadReport.skipped_duplicate" class="badge badge-pending">{{ docUploadReport.skipped_duplicate }} duplicates skipped</span>
            <span v-if="docUploadReport.unmatched" class="badge badge-minor">{{ docUploadReport.unmatched }} unmatched</span>
          </div>
          <div v-if="docUploadReport.details?.filter((d: any) => d.action === 'unmatched').length" class="merge-missing">
            <strong>Unmatched documents (need manual assignment):</strong>
            <ul>
              <li v-for="d in docUploadReport.details.filter((d: any) => d.action === 'unmatched')" :key="d.filename">
                {{ d.filename }} ({{ d.doc_type }})
              </li>
            </ul>
          </div>
        </div>

        <!-- Document summary by tenant -->
        <div v-if="tenants.length" style="margin-top: 1.5rem">
          <h3>Documents by Tenant</h3>
          <div class="table-scroll">
            <table class="data-table compact">
              <thead>
                <tr><th>Tenant</th><th>Suite</th><th class="c">Documents</th><th class="c">Extracted</th></tr>
              </thead>
              <tbody>
                <tr v-for="t in occupiedTenants" :key="t.id" @click="toggleTenantDocs(t.id)" style="cursor:pointer">
                  <td class="tenant-name">{{ t.tenant_name }}</td>
                  <td>{{ t.suite }}</td>
                  <td class="c">{{ t.documents.total }}</td>
                  <td class="c">{{ t.documents.extracted }}</td>
                </tr>
                <template v-if="expandedTenant">
                  <tr v-if="tenantDocs.length || expandedTenant" class="doc-row">
                    <td colspan="4">
                      <div class="doc-list">
                        <div v-for="d in tenantDocs" :key="d.id" class="doc-item">
                          <span class="doc-type">{{ d.doc_type }}</span>
                          <span class="doc-name">{{ d.filename }}</span>
                          <span class="doc-pages">{{ d.page_count ? d.page_count + ' pg' : '' }}</span>
                          <span :class="'badge badge-' + d.extraction_status">{{ d.extraction_status }}</span>
                        </div>
                        <div v-if="!tenantDocs.length" class="doc-empty">No documents</div>
                      </div>
                    </td>
                  </tr>
                </template>
              </tbody>
            </table>
          </div>
        </div>

        <button class="btn-primary" style="margin-top: 1rem" @click="goToStep('extraction')">
          Continue to AI Extraction &rarr;
        </button>
      </div>

      <!-- STEP 4: AI Extraction -->
      <div v-if="activeStep === 'extraction'" class="step-content">
        <h2>AI Extraction</h2>
        <p class="subtitle">Run Claude extraction on pending documents to pull rent steps, cotenancy clauses, exclusive use, and renewal options.</p>

        <div class="extraction-status">
          <div v-if="progress">
            <strong>{{ progress.docs_extracted }}</strong> of <strong>{{ progress.docs_uploaded }}</strong> documents extracted
            <span v-if="progress.docs_pending"> ({{ progress.docs_pending }} pending)</span>
          </div>
        </div>

        <button class="btn-primary" @click="runExtraction" :disabled="extracting || !progress?.docs_pending">
          {{ extracting ? 'Extracting...' : 'Run Extraction' }}
        </button>
        <div v-if="extractionMessage" class="upload-msg" style="margin-top: 0.5rem">{{ extractionMessage }}</div>

        <button class="btn-primary" style="margin-top: 1rem" @click="goToStep('validation')">
          Continue to Validation &rarr;
        </button>
      </div>

      <!-- STEP 5: Validation -->
      <div v-if="activeStep === 'validation'" class="step-content">
        <h2>Three-Way Validation</h2>
        <p class="subtitle">Compare seller rent roll vs AI-extracted lease terms vs Argus (if provided). Flags matches and mismatches.</p>

        <button class="btn-primary" @click="runValidation" :disabled="validating" style="margin-bottom: 1rem">
          {{ validating ? 'Validating...' : 'Run Validation' }}
        </button>

        <!-- Validation summary cards -->
        <div v-if="Object.keys(valSummary).length" class="val-summary">
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

        <!-- Annual rent comparison -->
        <div v-if="annualRentValidation.length">
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
        </div>

        <!-- Full validation detail -->
        <div v-if="validation.length">
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

        <button class="btn-primary" style="margin-top: 1rem" @click="goToStep('review')">
          Continue to Analyst Review &rarr;
        </button>
      </div>

      <!-- STEP 6: Analyst Review -->
      <div v-if="activeStep === 'review'" class="step-content">
        <h2>Analyst Review &amp; Approval</h2>
        <p class="subtitle">Review each tenant. Approve or flag tenants based on validation results. All non-vacant tenants must be approved to complete the review.</p>

        <div class="approval-summary">
          <span class="badge badge-extracted">{{ approvedCount }} approved</span>
          <span v-if="flaggedCount" class="badge badge-mismatch">{{ flaggedCount }} flagged</span>
          <span class="badge badge-pending">{{ occupiedTenants.length - approvedCount - flaggedCount }} pending</span>
        </div>

        <div class="table-scroll" style="margin-top: 1rem">
          <table class="data-table">
            <thead>
              <tr>
                <th>Tenant</th><th>Suite</th><th class="r">SF</th>
                <th class="r">Annual Rent</th><th class="c">Extraction</th>
                <th class="c">Approval</th><th>Actions</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="t in occupiedTenants" :key="t.id"
                  :class="{ 'row-approved': t.approval_status === 'approved', 'row-flagged': t.approval_status === 'flagged' }">
                <td class="tenant-name">{{ t.tenant_name }}</td>
                <td>{{ t.suite }}</td>
                <td class="r">{{ fmtSF(t.square_feet) }}</td>
                <td class="r">{{ fmtCurrency(t.annual_rent) }}</td>
                <td class="c"><span :class="'badge badge-' + t.extraction_status">{{ t.extraction_status }}</span></td>
                <td class="c">
                  <span :class="'badge badge-' + (t.approval_status === 'approved' ? 'extracted' : t.approval_status === 'flagged' ? 'mismatch' : 'pending')">
                    {{ t.approval_status || 'pending' }}
                  </span>
                </td>
                <td>
                  <button v-if="t.approval_status !== 'approved'" class="btn-sm btn-approve" @click="setTenantApproval(t.id, 'approved')">Approve</button>
                  <button v-if="t.approval_status !== 'flagged'" class="btn-sm btn-flag" @click="setTenantApproval(t.id, 'flagged')">Flag</button>
                  <button v-if="t.approval_status !== 'pending'" class="btn-sm btn-reset" @click="setTenantApproval(t.id, 'pending')">Reset</button>
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <button class="btn-primary" style="margin-top: 1rem" @click="goToStep('complete')"
                :disabled="approvedCount < occupiedTenants.length">
          {{ approvedCount >= occupiedTenants.length ? 'Complete Review \u2192' : `Approve all tenants to continue (${approvedCount}/${occupiedTenants.length})` }}
        </button>
      </div>

      <!-- STEP 7: Complete / Deliverables -->
      <div v-if="activeStep === 'complete'" class="step-content">
        <h2>Review Complete</h2>
        <p class="subtitle">Due diligence review is complete. Download the comprehensive workbook or review analysis below.</p>

        <button class="btn-primary" @click="downloadExcel" style="margin-bottom: 1.5rem">
          Download DD Workbook (Excel)
        </button>

        <!-- Lease Expirations -->
        <div v-if="expChartOption" style="margin-bottom: 2rem">
          <h3>Lease Expiration Schedule</h3>
          <div class="chart-container">
            <v-chart :option="expChartOption" style="height:350px" autoresize />
          </div>
        </div>

        <!-- Co-Tenancy Risk -->
        <div v-if="riskChartOption" style="margin-bottom: 2rem">
          <h3>Co-Tenancy Risk</h3>
          <div class="chart-container">
            <v-chart :option="riskChartOption" style="height:300px" autoresize />
          </div>
        </div>

        <!-- Co-tenancy clause details -->
        <div v-if="cotenancy?.clauses?.length" style="margin-bottom: 2rem">
          <h3>Co-Tenancy Clause Details</h3>
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

        <!-- Scenario Analysis -->
        <div v-if="scenarios.length" style="margin-bottom: 2rem">
          <h3>Cascading Scenario Analysis</h3>
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
        </div>
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
.kpi-flagged { color: #C00000; }

/* Workflow Stepper */
.stepper {
  display: flex;
  margin-bottom: 1.5rem;
  border: 1px solid #e0e0e0;
  border-radius: 6px;
  overflow: hidden;
  background: #f8f9fa;
}
.step {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem 1rem;
  cursor: pointer;
  transition: all 0.15s;
  border-right: 1px solid #e0e0e0;
  font-size: 0.82rem;
}
.step:last-child { border-right: none; }
.step:hover:not(.locked) { background: #e8f0f8; }
.step.active {
  background: #1F4E79;
  color: #fff;
}
.step.completed {
  background: #e8f5e9;
  color: #2e7d32;
}
.step.locked {
  opacity: 0.4;
  cursor: not-allowed;
}
.step-number {
  width: 24px; height: 24px;
  border-radius: 50%;
  background: #ccc;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.75rem; font-weight: 600;
  color: #fff;
  flex-shrink: 0;
}
.step.active .step-number { background: #fff; color: #1F4E79; }
.step.completed .step-number { background: #2e7d32; color: #fff; }
.step-check { font-size: 0.7rem; }
.step-label { font-weight: 500; white-space: nowrap; }

/* Step content */
.step-content { margin-bottom: 2rem; }
.step-content h2 { margin: 0 0 0.5rem; font-size: 1.15rem; color: #1F4E79; }
.step-content h3 { margin: 0 0 0.5rem; font-size: 0.95rem; color: #333; }
.subtitle { color: #666; font-size: 0.85rem; margin: 0 0 1rem; }

/* Setup info */
.setup-info {
  display: flex; gap: 2rem; font-size: 0.9rem; color: #555;
  padding: 0.75rem 1rem; background: #f8f9fa; border-radius: 6px;
}

/* Upload actions */
.upload-actions {
  display: flex; gap: 0.75rem; margin-bottom: 1rem; flex-wrap: wrap;
}
.btn-upload-label {
  display: inline-block; padding: 0.5rem 1rem;
  border-radius: 4px; cursor: pointer; font-size: 0.85rem;
}
.btn-primary { padding: 0.5rem 1rem; background: #1F4E79; color: #fff; border: none; border-radius: 4px; cursor: pointer; font-size: 0.85rem; }
.btn-primary:hover { background: #163a5c; }
.btn-primary:disabled { opacity: 0.5; cursor: default; }
.btn-secondary { padding: 0.5rem 1rem; background: #e0e0e0; color: #333; border: none; border-radius: 4px; cursor: pointer; font-size: 0.85rem; }
.btn-secondary:hover { background: #d0d0d0; }
.btn-cancel { padding: 0.5rem 1rem; background: #e0e0e0; color: #333; border: none; border-radius: 4px; cursor: pointer; font-size: 0.85rem; }
.upload-msg { font-size: 0.85rem; color: #548235; margin-top: 0.5rem; }

/* Merge report */
.merge-report {
  padding: 0.75rem 1rem; background: #f0f8ff; border: 1px solid #b4d4f0;
  border-radius: 6px; margin-top: 0.75rem;
}
.merge-stats { display: flex; gap: 0.5rem; margin-top: 0.5rem; }
.merge-missing { margin-top: 0.5rem; font-size: 0.82rem; color: #666; }
.merge-missing ul { margin: 0.25rem 0 0 1.5rem; padding: 0; }

/* Extraction status */
.extraction-status {
  padding: 0.75rem 1rem; background: #f8f9fa; border: 1px solid #e0e0e0;
  border-radius: 6px; margin-bottom: 1rem; font-size: 0.9rem;
}

/* Approval */
.approval-summary { display: flex; gap: 0.5rem; margin-bottom: 0.5rem; }
.row-approved { background: #f0fff0; }
.row-flagged { background: #fff5f5; }
.btn-sm {
  padding: 0.2rem 0.5rem; font-size: 0.75rem; border: none;
  border-radius: 3px; cursor: pointer; margin-right: 0.25rem;
}
.btn-approve { background: #c6efce; color: #006100; }
.btn-approve:hover { background: #a8e4b0; }
.btn-flag { background: #ffc7ce; color: #9c0006; }
.btn-flag:hover { background: #ffb0b8; }
.btn-reset { background: #e0e0e0; color: #333; }
.btn-reset:hover { background: #d0d0d0; }

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
.form-field input, .form-field select {
  width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px;
  font-size: 0.9rem; box-sizing: border-box;
}
.modal-actions { display: flex; justify-content: flex-end; gap: 0.5rem; margin-top: 1rem; }
</style>
