<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
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
const uploadMessage = ref('')
const mergeReport = ref<any>(null)

// Sales import
const uploadingSales = ref(false)
const salesUploadMessage = ref('')
const salesData = ref<Record<string, any>>({})
const hasSales = ref(false)
const editingSalesTenantId = ref<number | null>(null)
const editingSalesValue = ref('')

// Document upload
const uploadingDocs = ref(false)
const docUploadReport = ref<any>(null)

// Unmatched document assignment
const unmatchedDocs = ref<any[]>([])
const unmatchedAssignments = ref<Record<number, number>>({})  // doc_id -> tenant_id

// Extraction
const extracting = ref(false)
const extractionMessage = ref('')
const resettingExtraction = ref(false)

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

  // Resume polling if extraction is already running (e.g. page reload)
  if (selectedReviewId.value) {
    try {
      const st = await api.get(`/api/lease-review/reviews/${selectedReviewId.value}/extract-status`)
      if (st.data.status === 'running') {
        extracting.value = true
        extractionMessage.value = `Extracting ${st.data.extracted} of ${st.data.total}...`
        pollExtractionStatus()
      }
    } catch { /* ignore */ }
  }
})

onUnmounted(() => {
  if (extractionPollTimer) { clearInterval(extractionPollTimer); extractionPollTimer = null }
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

    // Load sales data
    if (tenants.value.length) {
      try {
        const salesRes = await api.get(`/api/lease-review/reviews/${id}/sales`)
        salesData.value = salesRes.data.tenants || {}
        hasSales.value = salesRes.data.has_sales || false
      } catch { salesData.value = {}; hasSales.value = false }
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
  // Load unmatched docs when entering documents step
  if (key === 'documents') loadUnmatchedDocs()
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

// --- Tenant disposition -------------------------------------------------------
// After the rent roll is checked against the leases, a tenant row means one of
// three things. None of them deletes the record: the lease and the abstract built
// from it stay on file, and only the reading changes whether it is projected.
const dispositionOptions = [
  { value: 'active', label: 'On the rent roll',
    meaning: 'Supported by a lease and on the rent roll — counted in the projection.' },
  { value: 'vacated', label: 'Vacated',
    meaning: 'Tenant has left. The lease stays on file; not counted in the projection.' },
  { value: 'disregarded', label: 'No lease',
    meaning: 'Rent roll entry with no lease to support it — not counted in the projection.' },
]
const savingDisposition = ref<number | null>(null)
const dispositionError = ref<Record<number, string>>({})
const localDispositions = ref<Record<number, string>>({})

// Bulk: a review can throw off dozens of findings, and most of them usually get
// the same reading. Pick the many, then correct the few.
const selectedFindings = ref<number[]>([])
const bulkSaving = ref(false)
const bulkError = ref('')

// Read once, gone. A finding is outstanding only while nobody has given it a
// reading: either the analyst just did (addressedFindings), or it was settled in an
// earlier run and came back carrying its status. Re-listing a settled tenant is what
// made clicking its reading look like it did nothing — the reading was already set.
const addressedFindings = ref<number[]>([])

const pendingFindings = computed<any[]>(() =>
  (mergeReport.value?.not_in_upload_tenants || []).filter((t: any) =>
    !addressedFindings.value.includes(t.id) &&
    (t.tenant_status || 'active') === 'active'))

const settledFindingCount = computed(() =>
  (mergeReport.value?.not_in_upload_tenants || []).length - pendingFindings.value.length)

const findingIds = computed<number[]>(() =>
  pendingFindings.value.map((t: any) => t.id).filter((id: any) => id != null))

function markAddressed(ids: number[]) {
  addressedFindings.value = [...new Set([...addressedFindings.value, ...ids])]
  selectedFindings.value = selectedFindings.value.filter(id => !ids.includes(id))
}
const allFindingsSelected = computed(() =>
  findingIds.value.length > 0 &&
  selectedFindings.value.length === findingIds.value.length)

function toggleAllFindings(on: boolean) {
  selectedFindings.value = on ? [...findingIds.value] : []
}
function toggleFinding(id: number, on: boolean) {
  const set = new Set(selectedFindings.value)
  on ? set.add(id) : set.delete(id)
  selectedFindings.value = [...set]
}

async function setDispositionBulk(status: string) {
  if (!selectedFindings.value.length || !selectedReviewId.value) return
  bulkSaving.value = true
  bulkError.value = ''
  try {
    const res = await api.put(
      `/api/lease-review/reviews/${selectedReviewId.value}/tenants/dispositions`,
      { tenant_ids: selectedFindings.value, status }
    )
    const done = res.data.tenant_ids || selectedFindings.value
    for (const id of done) localDispositions.value[id] = status
    markAddressed(done)
    selectedFindings.value = []
    await loadReview(selectedReviewId.value!)
  } catch (e: any) {
    bulkError.value = e.response?.data?.error || 'Could not apply'
  } finally {
    bulkSaving.value = false
  }
}

function dispositionOf(t: any): string {
  if (t?.id != null && localDispositions.value[t.id]) return localDispositions.value[t.id]
  const row = tenants.value.find((x: any) => x.id === t?.id)
  return row?.tenant_status || 'active'
}

async function setDisposition(t: any, status: string) {
  if (!t?.id || !selectedReviewId.value) return
  savingDisposition.value = t.id
  delete dispositionError.value[t.id]
  try {
    await api.put(
      `/api/lease-review/reviews/${selectedReviewId.value}/tenants/${t.id}/disposition`,
      { status }
    )
    localDispositions.value[t.id] = status
    markAddressed([t.id])
    await loadReview(selectedReviewId.value!)
  } catch (e: any) {
    dispositionError.value[t.id] = e.response?.data?.error || 'Could not save'
  } finally {
    savingDisposition.value = null
  }
}

// --- Rent roll column mapping -------------------------------------------------
// The file is scanned first and nothing is written until the analyst has said which
// columns are recoveries and whether each charge is monthly or annual. Neither is
// answerable from the header text: Market at Poplar prints CAM, Insurance and Tax as
// three separate columns, and its "Base Rent" is a monthly figure.
const scanResult = ref<any>(null)
const scanFile = ref<File | null>(null)

const scanning = ref(false)
const committing = ref(false)
const mapReport = ref<any>(null)

const scanEntries = computed<any[]>(() =>
  scanResult.value ? (scanResult.value.columns || scanResult.value.charges || []) : []
)
// Every periodic charge needs a period before the import can run.
const unansweredPeriods = computed<string[]>(() => {
  if (!scanResult.value) return []
  const periodic = ['base_rent', 'recovery', 'misc']
  return scanEntries.value
    .filter(e => periodic.includes(scanResult.value.mapping.roles[e.key]))
    .filter(e => !scanResult.value.mapping.bases[e.key])
    .map(e => e.label)
})
const mappedRecoveries = computed<string[]>(() =>
  scanEntries.value
    .filter(e => scanResult.value?.mapping.roles[e.key] === 'recovery')
    .map(e => e.label)
)
function needsPeriod(key: string): boolean {
  return ['base_rent', 'recovery', 'misc'].includes(scanResult.value?.mapping.roles[key])
}
function onRoleChange(key: string) {
  // Dropping a column from a periodic role retires the period question with it.
  if (!needsPeriod(key)) delete scanResult.value.mapping.bases[key]
}

async function onRentRollScan(event: Event) {
  const input = event.target as HTMLInputElement
  if (!input.files?.length || !selectedReviewId.value) return
  const file = input.files[0]
  const formData = new FormData()
  formData.append('file', file)

  scanning.value = true
  scanResult.value = null
  mapReport.value = null
  uploadMessage.value = ''
  try {
    const res = await api.post(
      `/api/lease-review/reviews/${selectedReviewId.value}/rent-roll/scan`,
      formData, { headers: { 'Content-Type': 'multipart/form-data' } }
    )
    scanResult.value = res.data
    scanFile.value = file
  } catch (e: any) {
    console.error('Scan error', e)
    alert(e.response?.data?.error || 'Could not read that rent roll')
  } finally {
    scanning.value = false
    input.value = ''
  }
}

async function commitRentRoll() {
  if (!scanFile.value || !selectedReviewId.value) return
  const formData = new FormData()
  formData.append('file', scanFile.value)
  formData.append('mapping', JSON.stringify(scanResult.value.mapping))

  committing.value = true
  try {
    const res = await api.post(
      `/api/lease-review/reviews/${selectedReviewId.value}/rent-roll/commit`,
      formData, { headers: { 'Content-Type': 'multipart/form-data' } }
    )
    mapReport.value = res.data
    mergeReport.value = res.data.status === 'merged' ? res.data : null
    addressedFindings.value = []
    selectedFindings.value = []
    uploadMessage.value =
      `${res.data.status === 'merged' ? 'Merged' : 'Imported'} — ` +
      `${(res.data.total_gla || 0).toLocaleString()} SF, ` +
      `$${Math.round(res.data.total_annual_rent || 0).toLocaleString()} annual rent, ` +
      `$${Math.round(res.data.total_annual_recoveries || 0).toLocaleString()} recoveries`
    scanResult.value = null
    scanFile.value = null
    await loadReview(selectedReviewId.value!)
  } catch (e: any) {
    console.error('Commit error', e)
    alert(e.response?.data?.error || 'Failed to import rent roll')
  } finally {
    committing.value = false
  }
}

function cancelScan() {
  scanResult.value = null
  scanFile.value = null
}



// Sales import
async function onSalesUpload(event: Event) {
  const input = event.target as HTMLInputElement
  if (!input.files?.length || !selectedReviewId.value) return

  const file = input.files[0]
  const formData = new FormData()
  formData.append('file', file)

  uploadingSales.value = true
  salesUploadMessage.value = 'Extracting sales data with AI...'
  try {
    const res = await api.post(
      `/api/lease-review/reviews/${selectedReviewId.value}/upload-sales`,
      formData,
      { headers: { 'Content-Type': 'multipart/form-data' }, timeout: 120000 }
    )
    const ext = res.data.extraction
    salesUploadMessage.value = `Imported: ${ext.matched} tenants matched, ${ext.unmatched} unmatched of ${ext.total}`
    salesData.value = res.data.sales?.tenants || {}
    hasSales.value = res.data.sales?.has_sales || false
  } catch (e: any) {
    console.error('Sales upload error', e)
    salesUploadMessage.value = ''
    alert(e.response?.data?.error || 'Failed to import sales data')
  } finally {
    uploadingSales.value = false
    input.value = ''
  }
}

function getTenantSales(tenantId: number) {
  return salesData.value[String(tenantId)] || null
}

function tenantTTMSales(t: any): number | null {
  const sd = getTenantSales(t.id)
  if (sd) return sd.ttm_sales || null
  return null
}

function tenantSalesPerSF(t: any): number | null {
  const sd = getTenantSales(t.id)
  if (sd && sd.sales_per_sf) return sd.sales_per_sf
  return null
}

function tenantOccCost(t: any): string {
  const rentPSF = t.annual_rent_per_sf || t.rent_per_sf || 0
  const recPSF = t.annual_recoveries_per_sf || 0
  const salesPSF = tenantSalesPerSF(t)
  if (!salesPSF || salesPSF <= 0) return '\u2014'
  const cost = (rentPSF + recPSF) / salesPSF
  return (cost * 100).toFixed(1) + '%'
}

function startEditSales(tenantId: number) {
  editingSalesTenantId.value = tenantId
  const sd = getTenantSales(tenantId)
  editingSalesValue.value = sd?.ttm_sales ? String(Math.round(sd.ttm_sales)) : ''
}

async function saveSalesEdit(tenantId: number) {
  if (!selectedReviewId.value) return
  const val = editingSalesValue.value.replace(/[,$]/g, '').trim()
  const numVal = val ? parseFloat(val) : null

  try {
    const res = await api.put(
      `/api/lease-review/reviews/${selectedReviewId.value}/tenants/${tenantId}/sales`,
      { annual_sales: numVal }
    )
    salesData.value = res.data.sales?.tenants || salesData.value
    hasSales.value = Object.keys(salesData.value).length > 0
  } catch (e: any) {
    alert(e.response?.data?.error || 'Failed to save')
  } finally {
    editingSalesTenantId.value = null
  }
}

function cancelSalesEdit() {
  editingSalesTenantId.value = null
}

// Document upload — one file at a time for reliability, with retry
const docUploadProgress = ref('')
const docUploadFailed = ref<string[]>([])
const uploadCancelled = ref(false)

async function cancelUpload() {
  uploadCancelled.value = true
}

async function uploadOneFile(
  reviewId: number,
  item: { file: File; hint: string },
  retries = 2
): Promise<any> {
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const formData = new FormData()
      formData.append('files', item.file)
      formData.append('folder_hints', JSON.stringify([item.hint]))
      const res = await api.post(
        `/api/lease-review/reviews/${reviewId}/upload-documents`,
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      )
      return res.data
    } catch (e: any) {
      if (attempt < retries) {
        // Wait before retry (1s, then 3s)
        await new Promise(r => setTimeout(r, (attempt + 1) * 2000))
        continue
      }
      throw e
    }
  }
}

async function onDocumentUpload(event: Event) {
  const input = event.target as HTMLInputElement
  if (!input.files?.length || !selectedReviewId.value) return

  // Collect PDF files and their folder hints
  const pdfFiles: { file: File; hint: string }[] = []
  for (const f of input.files) {
    if (!f.name.toLowerCase().endsWith('.pdf')) continue
    const relPath = (f as any).webkitRelativePath || ''
    const parts = relPath.split('/')
    const hint = parts.length > 2 ? parts[parts.length - 2] : (parts.length === 2 ? parts[0] : '')
    pdfFiles.push({ file: f, hint })
  }
  if (!pdfFiles.length) {
    alert('No PDF files found in the selection.')
    return
  }

  uploadingDocs.value = true
  uploadCancelled.value = false
  docUploadReport.value = null
  docUploadProgress.value = ''
  docUploadFailed.value = []

  const totals = { added: 0, skipped_duplicate: 0, unmatched: 0, details: [] as any[] }
  const totalFiles = pdfFiles.length
  const reviewId = selectedReviewId.value!

  for (let i = 0; i < totalFiles; i++) {
    if (uploadCancelled.value) {
      docUploadProgress.value = `Cancelled after ${i} of ${totalFiles} files.`
      break
    }
    const item = pdfFiles[i]
    const sizeMB = (item.file.size / (1024 * 1024)).toFixed(1)
    const parts: string[] = []
    if (totals.added) parts.push(`${totals.added} added`)
    if (totals.unmatched) parts.push(`${totals.unmatched} unmatched`)
    if (totals.skipped_duplicate) parts.push(`${totals.skipped_duplicate} skipped`)
    if (docUploadFailed.value.length) parts.push(`${docUploadFailed.value.length} failed`)
    const stats = parts.length ? ` [${parts.join(', ')}]` : ''
    docUploadProgress.value = `Uploading ${i + 1} of ${totalFiles}: ${item.file.name} (${sizeMB} MB)${stats}...`

    try {
      const d = await uploadOneFile(reviewId, item)
      totals.added += d.added || 0
      totals.skipped_duplicate += d.skipped_duplicate || 0
      totals.unmatched += d.unmatched || 0
      if (d.details) totals.details.push(...d.details)
    } catch (e: any) {
      console.error(`Failed: ${item.file.name}`, e)
      docUploadFailed.value.push(item.file.name)
    }
    if (i < totalFiles - 1) await new Promise(r => setTimeout(r, 200))
  }

  docUploadReport.value = totals
  if (docUploadFailed.value.length > 0 && !uploadCancelled.value) {
    alert(`Upload complete. ${totals.added} added, ${docUploadFailed.value.length} failed:\n${docUploadFailed.value.slice(0, 10).join('\n')}${docUploadFailed.value.length > 10 ? `\n...and ${docUploadFailed.value.length - 10} more` : ''}`)
  }
  await loadReview(reviewId)
  await loadUnmatchedDocs()
  uploadingDocs.value = false
  input.value = ''
}

// Unmatched document management
async function loadUnmatchedDocs() {
  if (!selectedReviewId.value) return
  try {
    const res = await api.get(`/api/lease-review/reviews/${selectedReviewId.value}/unmatched-documents`)
    unmatchedDocs.value = res.data.documents || []
    unmatchedAssignments.value = {}
  } catch { unmatchedDocs.value = [] }
}

// Deleting the leases of tenants who have gone is how the unmatched pile actually
// clears. Selection + one confirm, because it is the bulk case: assigning a former
// tenant's lease to a current one just to empty the list would be worse than leaving it.
const selectedDocs = ref<number[]>([])
const deletingDocs = ref(false)
const confirmDeleteDocs = ref(false)

const allDocsSelected = computed(() =>
  unmatchedDocs.value.length > 0 &&
  selectedDocs.value.length === unmatchedDocs.value.length)

function toggleAllDocs(on: boolean) {
  selectedDocs.value = on ? unmatchedDocs.value.map(d => d.id) : []
}
function toggleDoc(id: number, on: boolean) {
  const set = new Set(selectedDocs.value)
  on ? set.add(id) : set.delete(id)
  selectedDocs.value = [...set]
}

async function deleteDocs(ids: number[]) {
  if (!ids.length || !selectedReviewId.value) return
  deletingDocs.value = true
  try {
    const res = await api.delete(
      `/api/lease-review/reviews/${selectedReviewId.value}/documents`,
      { data: { doc_ids: ids } })
    const gone = new Set((res.data.documents || []).map((d: any) => d.id))
    unmatchedDocs.value = unmatchedDocs.value.filter(d => !gone.has(d.id))
    selectedDocs.value = selectedDocs.value.filter(id => !gone.has(id))
    confirmDeleteDocs.value = false
    docUploadReport.value = null
    await loadReview(selectedReviewId.value!)
  } catch (e: any) {
    alert('Could not delete: ' + (e.response?.data?.error || e.message))
  } finally {
    deletingDocs.value = false
  }
}

async function assignDoc(docId: number) {
  const tenantId = unmatchedAssignments.value[docId]
  if (!tenantId || !selectedReviewId.value) return
  try {
    await api.post(`/api/lease-review/reviews/${selectedReviewId.value}/documents/${docId}/assign-tenant`, { tenant_id: tenantId })
    unmatchedDocs.value = unmatchedDocs.value.filter(d => d.id !== docId)
    delete unmatchedAssignments.value[docId]
    await loadReview(selectedReviewId.value!)
  } catch (e: any) {
    alert('Failed to assign: ' + (e.response?.data?.error || e.message))
  }
}

async function assignAllDocs() {
  if (!selectedReviewId.value) return
  const entries = Object.entries(unmatchedAssignments.value).filter(([, tid]) => tid)
  for (const [docId, tenantId] of entries) {
    try {
      await api.post(`/api/lease-review/reviews/${selectedReviewId.value}/documents/${Number(docId)}/assign-tenant`, { tenant_id: tenantId })
      unmatchedDocs.value = unmatchedDocs.value.filter(d => d.id !== Number(docId))
      delete unmatchedAssignments.value[Number(docId)]
    } catch { /* skip failures */ }
  }
  await loadReview(selectedReviewId.value!)
}

// Run extraction
let extractionPollTimer: ReturnType<typeof setInterval> | null = null

async function runExtraction() {
  if (!selectedReviewId.value) return
  extracting.value = true
  extractionMessage.value = 'Starting AI extraction...'
  try {
    const res = await api.post(`/api/lease-review/reviews/${selectedReviewId.value}/extract`)
    if (res.data.status === 'already_running') {
      extractionMessage.value = 'Extraction already in progress...'
    }
    // Start polling for progress
    pollExtractionStatus()
  } catch (e: any) {
    extractionMessage.value = e.response?.data?.error || 'Extraction failed to start'
    extracting.value = false
  }
}

function pollExtractionStatus() {
  if (extractionPollTimer) clearInterval(extractionPollTimer)
  extractionPollTimer = setInterval(async () => {
    if (!selectedReviewId.value) return
    try {
      const res = await api.get(`/api/lease-review/reviews/${selectedReviewId.value}/extract-status`)
      const job = res.data
      if (job.status === 'running') {
        extractionMessage.value = `Extracting ${job.extracted} of ${job.total}... ${job.current_file || ''}`
      } else if (job.status === 'complete') {
        clearInterval(extractionPollTimer!)
        extractionPollTimer = null
        extractionMessage.value = `Extraction complete. ${job.extracted} of ${job.total} documents processed.`
        extracting.value = false
        await loadReview(selectedReviewId.value!)
      } else if (job.status === 'failed') {
        clearInterval(extractionPollTimer!)
        extractionPollTimer = null
        extractionMessage.value = `Extraction failed: ${job.error || 'unknown error'}`
        extracting.value = false
      }
    } catch {
      // Network error during poll — keep trying
    }
  }, 3000)
}

// Reset extraction data for re-extraction
async function resetExtraction() {
  if (!selectedReviewId.value) return
  if (!confirm('This will clear all AI-extracted data (rent steps, options, cotenancy, exclusive use, validation) and reset documents for re-extraction.\n\nTenant roster, uploaded documents, field resolutions, and abstracts are preserved.\n\nContinue?')) return
  resettingExtraction.value = true
  extractionMessage.value = 'Resetting extraction data...'
  try {
    const res = await api.post(`/api/lease-review/reviews/${selectedReviewId.value}/reset-extraction`)
    const d = res.data
    extractionMessage.value = `Reset complete: ${d.rent_steps} rent steps, ${d.options} options, ${d.cotenancy} cotenancy, ${d.exclusive_use} exclusive use, ${d.validation} validation cleared. ${d.documents_reset} documents ready for re-extraction.`
    await loadReview(selectedReviewId.value!)
  } catch (e: any) {
    extractionMessage.value = e.response?.data?.error || 'Reset failed'
  } finally {
    resettingExtraction.value = false
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
// A row read as "No lease" is not a tenant, so it does not belong in the tenant
// roster. Reported by asset management: three leftover rows -- the building banner and
// two subtotal rows from an import that predates the phantom-row fix -- were checked
// off as No lease and kept appearing anyway.
//
// A VACATED tenant still shows. We hold a lease for them and that lease has to stay
// reachable; it is simply out of the projection. The two readings mean different
// things and the roster treats them differently.
const disregardedTenants = computed(() =>
  tenants.value.filter(t => t.tenant_status === 'disregarded'))
const showDisregarded = ref(false)

const occupiedTenants = computed(() => tenants.value.filter(t =>
  !t.is_vacant && (showDisregarded.value || t.tenant_status !== 'disregarded')))
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
function fmtPerSF(val: number | null | undefined): string {
  if (val == null || val === 0) return '\u2014'
  return '$' + val.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
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
        <p class="subtitle">
          Upload the rent roll received from the operating partner. You confirm the
          columns before anything is written, and the import never deletes a tenant —
          the leases are the record, so a tenant the rent roll no longer lists is
          reported for a reading rather than removed.
        </p>

        <!-- One way in. Merge vs replace is chosen on the confirmation panel, after
             the columns have been seen, rather than by picking a button before the
             file has even been read. -->
        <div class="upload-actions">
          <label class="btn-primary btn-upload-label">
            {{ scanning ? 'Reading file...' : 'Select Rent Roll...' }}
            <input type="file" accept=".xlsx,.xls,.csv,.pdf" @change="onRentRollScan" :disabled="scanning" hidden />
          </label>
        </div>

        <!-- Column mapping — confirm before anything is written -->
        <div v-if="scanResult" class="map-panel">
          <div class="map-head">
            <h3>
              Confirm the columns
              <span class="map-sub">
                {{ scanResult.layout === 'stacked' ? 'stacked layout — one line per charge' : 'one row per tenant' }}
                · {{ scanResult.row_count }} tenant rows
              </span>
            </h3>
            <!-- No destructive mode. The leases are the authority here and the rent
                 roll is what is being checked against them, so a tenant we hold a
                 lease for that is missing from a later rent roll is a finding about
                 the rent roll -- not a reason to delete the tenant and the abstract
                 built from its lease. Removing a tenant stays a deliberate, one-at-
                 a-time act. -->
            <div class="map-actions">
              <button class="btn-primary" :disabled="committing || unansweredPeriods.length > 0"
                @click="commitRentRoll">
                {{ committing ? 'Importing...' : 'Import' }}
              </button>
              <button class="btn-secondary" @click="cancelScan" :disabled="committing">Cancel</button>
            </div>
          </div>

          <div v-if="scanResult.warnings?.length" class="map-warnings">
            <div v-for="w in scanResult.warnings" :key="w">{{ w }}</div>
          </div>
          <div v-if="unansweredPeriods.length" class="map-blocker">
            The file does not say whether these are monthly or annual amounts — set each
            one before importing: <strong>{{ unansweredPeriods.join(', ') }}</strong>
          </div>
          <div v-if="mappedRecoveries.length > 1" class="map-note">
            {{ mappedRecoveries.length }} recovery columns will be summed:
            <strong>{{ mappedRecoveries.join(' + ') }}</strong>
          </div>

          <div class="table-scroll">
            <table class="data-table compact">
              <thead>
                <tr>
                  <th>{{ scanResult.layout === 'stacked' ? 'Charge' : 'Column' }}</th>
                  <th>Sample values</th>
                  <th>Treat as</th>
                  <th>Monthly or annual</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="e in scanEntries" :key="e.key"
                  :class="{ 'map-row-off': scanResult.mapping.roles[e.key] === 'ignore' }">
                  <td class="map-label">
                    {{ e.label }}
                    <span v-if="e.count" class="map-count">{{ e.count }} rows</span>
                  </td>
                  <td class="map-samples">{{ (e.samples || []).join('  ·  ') }}</td>
                  <td>
                    <select v-model="scanResult.mapping.roles[e.key]" class="map-select"
                      @change="onRoleChange(e.key)">
                      <option v-for="o in scanResult.role_options" :key="o.value" :value="o.value">
                        {{ o.label }}
                      </option>
                    </select>
                  </td>
                  <td>
                    <select v-if="needsPeriod(e.key)" v-model="scanResult.mapping.bases[e.key]"
                      class="map-select" :class="{ 'map-missing': !scanResult.mapping.bases[e.key] }">
                      <option :value="undefined">— pick one —</option>
                      <option v-for="o in scanResult.basis_options" :key="o.value" :value="o.value">
                        {{ o.label }}
                      </option>
                    </select>
                    <span v-else class="map-na">—</span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          <div v-if="scanResult.excluded_rows?.length" class="map-excluded">
            <strong>Excluded as non-tenant rows:</strong>
            <span v-for="x in scanResult.excluded_rows" :key="x.index">
              {{ x.name }} <em>({{ x.reason }})</em>;
            </span>
          </div>
          <div v-if="scanResult.skipped" class="map-excluded">
            <strong>Also skipped:</strong>
            {{ scanResult.skipped.increase }} future rent increases,
            {{ scanResult.skipped.vacant }} vacant units kept as vacant.
          </div>
        </div>

        <!-- What the last mapped import actually did -->
        <div v-if="mapReport?.mapping_report" class="map-report">
          <h3>Import Applied</h3>
          <div class="map-report-row">
            <strong>Base rent:</strong>
            {{ (mapReport.mapping_report.base_rent_columns || mapReport.mapping_report.base_rent_charges || []).join(', ') || '—' }}
          </div>
          <div class="map-report-row">
            <strong>Recoveries:</strong>
            {{ (mapReport.mapping_report.recovery_columns || mapReport.mapping_report.recovery_charges || []).join(' + ') || '—' }}
          </div>
          <div v-if="mapReport.mapping_report.excluded?.length" class="map-report-row">
            <strong>Rows excluded:</strong> {{ mapReport.mapping_report.excluded.join('; ') }}
          </div>
          <div v-if="mapReport.mapping_report.tie_out?.length" class="map-report-row map-blocker">
            <strong>Charges do not add to the stated tenant total:</strong>
            <span v-for="t in mapReport.mapping_report.tie_out" :key="t.tenant">
              {{ t.tenant }} (stated {{ t.stated }}, summed {{ t.summed }});
            </span>
          </div>
        </div>

        <div style="margin-top: 0.75rem">
          <label class="btn-secondary btn-upload-label" style="background: #e8f0fe; color: #1a73e8; border-color: #1a73e8">
            {{ uploadingSales ? 'Extracting...' : 'Import Tenant Sales (AI)' }}
            <input type="file" accept=".pdf" @change="onSalesUpload" :disabled="uploadingSales || !tenants.length" hidden />
          </label>
          <span v-if="salesUploadMessage" class="upload-msg" style="margin-left: 0.75rem">{{ salesUploadMessage }}</span>
        </div>

        <div v-if="uploadMessage" class="upload-msg">{{ uploadMessage }}</div>

        <!-- Merge report -->
        <div v-if="mergeReport" class="merge-report">
          <h3>Merge Results</h3>
          <div class="merge-stats">
            <span class="badge badge-extracted">{{ mergeReport.matched }} updated</span>
            <span class="badge badge-match">{{ mergeReport.added }} added</span>
            <span v-if="pendingFindings.length" class="badge badge-minor">{{ pendingFindings.length }} to read</span>
            <span v-if="settledFindingCount" class="badge badge-match">{{ settledFindingCount }} read</span>
          </div>
          <!-- These are the rows where a lease we hold disagrees with the rent roll.
               Nothing was deleted; each one needs a reading, and that reading is what
               decides whether it belongs in the projection. -->
          <div v-if="pendingFindings.length" class="merge-missing">
            <strong>In our records but not on this rent roll — how should each be read?</strong>
            <p class="disp-help">
              Leases stay on file either way. Only <em>On the rent roll</em> is counted
              in the projection.
            </p>
            <div class="disp-bulk">
              <label class="disp-bulk-all">
                <input type="checkbox" :checked="allFindingsSelected"
                  @change="toggleAllFindings(($event.target as HTMLInputElement).checked)" />
                Select all {{ findingIds.length }}
              </label>
              <template v-if="selectedFindings.length">
                <span class="disp-bulk-count">{{ selectedFindings.length }} selected —
                  read them all as:</span>
                <button v-for="opt in dispositionOptions" :key="opt.value"
                  class="btn-xs disp-btn" :disabled="bulkSaving"
                  :title="opt.meaning" @click="setDispositionBulk(opt.value)">
                  {{ opt.label }}
                </button>
                <button class="btn-xs disp-btn" :disabled="bulkSaving"
                  @click="selectedFindings = []">Clear</button>
              </template>
              <span v-if="bulkSaving" class="disp-bulk-count">applying…</span>
              <span v-if="bulkError" class="disp-err">{{ bulkError }}</span>
            </div>
            <table class="data-table compact disp-table">
              <thead>
                <tr><th class="disp-check"></th><th>Tenant</th><th>Suite</th><th>Reading</th></tr>
              </thead>
              <tbody>
                <tr v-for="t in pendingFindings" :key="t.id ?? t.suite">
                  <td class="disp-check">
                    <input type="checkbox" :disabled="!t.id"
                      :checked="selectedFindings.includes(t.id)"
                      @change="toggleFinding(t.id, ($event.target as HTMLInputElement).checked)" />
                  </td>
                  <td class="tenant-name" :title="t.tenant"><span class="tname">{{ t.tenant }}</span></td>
                  <td class="nowrap-cell">{{ t.suite }}</td>
                  <td class="nowrap-cell">
                    <button v-for="opt in dispositionOptions" :key="opt.value"
                      class="btn-xs disp-btn"
                      :class="{ 'disp-on': (dispositionOf(t) === opt.value) }"
                      :disabled="!t.id || savingDisposition === t.id"
                      :title="opt.meaning"
                      @click="setDisposition(t, opt.value)">{{ opt.label }}</button>
                    <span v-if="dispositionError[t.id]" class="disp-err">{{ dispositionError[t.id] }}</span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- Tenant roster preview -->
        <div v-if="tenants.length" style="margin-top: 1.5rem">
          <h3>Current Tenant Roster ({{ occupiedTenants.length }} tenants)</h3>
          <!-- Hidden, not deleted: the reading is reversible and the rows are still
               here to be read differently. -->
          <p v-if="disregardedTenants.length" class="roster-hidden">
            {{ disregardedTenants.length }} row(s) read as <strong>No lease</strong>
            {{ showDisregarded ? 'are shown below' : 'are hidden' }} and excluded from
            the totals.
            <button class="btn-xs disp-btn" @click="showDisregarded = !showDisregarded">
              {{ showDisregarded ? 'Hide them' : 'Show them' }}
            </button>
          </p>
          <div class="table-scroll">
            <table class="data-table compact">
              <thead>
                <tr>
                  <th>Tenant</th><th>Suite</th><th class="r">SF</th>
                  <th>Lease Type</th><th>Lease Start</th><th>Lease End</th>
                  <th class="r">Term</th>
                  <th class="r">Monthly Rent</th><th class="r">Monthly $/SF</th>
                  <th class="r">Annual Rent</th><th class="r">Annual $/SF</th>
                  <th class="r">Recoveries $/SF</th><th class="r">Misc $/SF</th>
                  <th class="r">Annual Sales</th><th class="r">Sales $/SF</th>
                  <th class="r">Occ. Cost</th>
                  <th>Reading</th>
                  <th>Source</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="t in occupiedTenants" :key="t.id">
                  <td class="tenant-name" :title="t.tenant_name"><span class="tname">{{ t.tenant_name }}</span></td>
                  <td class="nowrap-cell">{{ t.suite }}</td>
                  <td class="r">{{ fmtSF(t.square_feet) }}</td>
                  <td class="nowrap-cell">{{ t.lease_type || '\u2014' }}</td>
                  <td class="date-cell">{{ fmtDate(t.lease_start) }}</td>
                  <td class="date-cell">{{ fmtDate(t.lease_end) }}</td>
                  <td class="r">{{ t.term_months || '\u2014' }}</td>
                  <td class="r">{{ fmtCurrency(t.monthly_rent) }}</td>
                  <td class="r">{{ fmtPerSF(t.monthly_rent_per_sf) }}</td>
                  <td class="r">{{ fmtCurrency(t.annual_rent) }}</td>
                  <td class="r">{{ fmtPerSF(t.annual_rent_per_sf || t.rent_per_sf) }}</td>
                  <td class="r">{{ fmtPerSF(t.annual_recoveries_per_sf) }}</td>
                  <td class="r">{{ fmtPerSF(t.annual_misc_per_sf) }}</td>
                  <td class="r editable-cell" @dblclick="startEditSales(t.id)">
                    <template v-if="editingSalesTenantId === t.id">
                      <input type="text" class="inline-edit" v-model="editingSalesValue"
                        @keyup.enter="saveSalesEdit(t.id)" @keyup.escape="cancelSalesEdit()"
                        @blur="saveSalesEdit(t.id)" ref="salesEditInput" />
                    </template>
                    <template v-else>
                      <span :class="{'override-val': getTenantSales(t.id)?.has_override}">
                        {{ tenantTTMSales(t) != null ? fmtCurrency(tenantTTMSales(t)) : '\u2014' }}
                      </span>
                    </template>
                  </td>
                  <td class="r">{{ tenantSalesPerSF(t) != null ? fmtPerSF(tenantSalesPerSF(t)) : '\u2014' }}</td>
                  <td class="r">{{ tenantOccCost(t) }}</td>
                  <td class="nowrap-cell">
                    <select class="map-select disp-select"
                      :class="{ 'disp-off': (t.tenant_status || 'active') !== 'active' }"
                      :value="t.tenant_status || 'active'"
                      :disabled="savingDisposition === t.id"
                      @change="setDisposition(t, ($event.target as HTMLSelectElement).value)">
                      <option v-for="opt in dispositionOptions" :key="opt.value"
                        :value="opt.value">{{ opt.label }}</option>
                    </select>
                  </td>
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
          <span v-if="docUploadProgress" class="upload-progress-text">{{ docUploadProgress }}</span>
          <button v-if="uploadingDocs && !uploadCancelled" class="btn-cancel" @click="cancelUpload">Cancel</button>
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

        <!-- Unmatched documents — manual tenant assignment -->
        <div v-if="unmatchedDocs.length" style="margin-top: 1.5rem">
          <h3 style="color: #C00000">Unmatched Documents ({{ unmatchedDocs.length }})</h3>
          <p class="subtitle">
            These could not be matched to a tenant. Assign the ones that belong to a
            current tenant; delete the rest — a lease for a tenant who has gone belongs
            nowhere, and assigning it to somebody else to clear the list is worse than
            leaving it.
          </p>
          <div class="doc-bulk">
            <label class="doc-bulk-all">
              <input type="checkbox" :checked="allDocsSelected"
                @change="toggleAllDocs(($event.target as HTMLInputElement).checked)" />
              Select all {{ unmatchedDocs.length }}
            </label>
            <template v-if="selectedDocs.length">
              <span class="doc-bulk-count">{{ selectedDocs.length }} selected</span>
              <button v-if="!confirmDeleteDocs" class="btn-xs doc-del"
                :disabled="deletingDocs" @click="confirmDeleteDocs = true">
                Delete selected
              </button>
              <template v-else>
                <span class="doc-warn">Delete {{ selectedDocs.length }} file(s)? This cannot be undone.</span>
                <button class="btn-xs doc-del-yes" :disabled="deletingDocs"
                  @click="deleteDocs(selectedDocs)">
                  {{ deletingDocs ? 'Deleting...' : 'Yes, delete' }}
                </button>
                <button class="btn-xs" :disabled="deletingDocs"
                  @click="confirmDeleteDocs = false">Cancel</button>
              </template>
            </template>
          </div>
          <div class="table-scroll">
            <table class="data-table compact">
              <thead>
                <tr><th class="disp-check"></th><th>Filename</th><th>Type</th><th>Assign to Tenant</th><th class="c">Action</th></tr>
              </thead>
              <tbody>
                <tr v-for="d in unmatchedDocs" :key="d.id">
                  <td class="disp-check">
                    <input type="checkbox" :checked="selectedDocs.includes(d.id)"
                      @change="toggleDoc(d.id, ($event.target as HTMLInputElement).checked)" />
                  </td>
                  <td style="font-size: 0.85rem">{{ d.filename }}</td>
                  <td>{{ d.doc_type }}</td>
                  <td>
                    <select v-model="unmatchedAssignments[d.id]" style="width: 100%; padding: 0.3rem">
                      <option :value="undefined">-- Select tenant --</option>
                      <option v-for="t in occupiedTenants" :key="t.id" :value="t.id">
                        {{ t.suite ? t.suite + ' — ' : '' }}{{ t.tenant_name }}
                      </option>
                    </select>
                  </td>
                  <td class="c">
                    <button class="btn-primary" style="padding: 0.2rem 0.6rem; font-size: 0.8rem" @click="assignDoc(d.id)" :disabled="!unmatchedAssignments[d.id]">Assign</button>
                    <button class="btn-xs doc-del" style="margin-left: 4px"
                      :disabled="deletingDocs" @click="deleteDocs([d.id])">Delete</button>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <button class="btn-primary" style="margin-top: 0.5rem" @click="assignAllDocs" :disabled="!Object.values(unmatchedAssignments).some(v => v)">
            Assign All Selected
          </button>
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
                  <td class="tenant-name" :title="t.tenant_name"><span class="tname">{{ t.tenant_name }}</span></td>
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

        <div class="extraction-actions">
          <button class="btn-primary" @click="runExtraction" :disabled="extracting || resettingExtraction || !progress?.docs_pending">
            {{ extracting ? 'Extracting...' : 'Run Extraction' }}
          </button>
          <button class="btn-danger" @click="resetExtraction" :disabled="extracting || resettingExtraction"
                  title="Clear all extracted data and re-run with updated prompt">
            {{ resettingExtraction ? 'Resetting...' : 'Reset Extraction' }}
          </button>
        </div>
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
                <td class="tenant-name" :title="t.tenant_name"><span class="tname">{{ t.tenant_name }}</span></td>
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
                  <td class="tenant-name" :title="c.tenant_name"><span class="tname">{{ c.tenant_name }}</span></td>
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
  display: flex; gap: 0.75rem; margin-bottom: 1rem; flex-wrap: wrap; align-items: center;
}
.upload-progress-text {
  font-size: 0.85rem; color: #666; font-style: italic;
}
.btn-upload-label {
  display: inline-block; padding: 0.5rem 1rem;
  border-radius: 4px; cursor: pointer; font-size: 0.85rem;
}
.btn-primary { padding: 0.5rem 1rem; background: #1F4E79; color: #fff; border: none; border-radius: 4px; cursor: pointer; font-size: 0.85rem; }
.btn-primary:hover { background: #163a5c; }
.btn-primary:disabled { opacity: 0.5; cursor: default; }
.btn-cancel { padding: 0.3rem 0.8rem; background: #C00000; color: #fff; border: none; border-radius: 4px; cursor: pointer; font-size: 0.8rem; margin-left: 0.5rem; }
.btn-cancel:hover { background: #900; }
.btn-secondary { padding: 0.5rem 1rem; background: #e0e0e0; color: #333; border: none; border-radius: 4px; cursor: pointer; font-size: 0.85rem; }
.btn-secondary:hover { background: #d0d0d0; }
.btn-danger { padding: 0.5rem 1rem; background: #C00000; color: #fff; border: none; border-radius: 4px; cursor: pointer; font-size: 0.85rem; }
.btn-danger:hover { background: #a00; }
.btn-danger:disabled { opacity: 0.5; cursor: default; }
.extraction-actions { display: flex; gap: 12px; align-items: center; }
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
/* ISO dates offer a break opportunity at each hyphen, so a narrow column
   splits 2028-05-31 over two lines and doubles the row height. */
.date-cell { white-space: nowrap; }

/* Rent roll column mapping */
.map-panel {
  margin-top: 1.25rem; border: 1px solid #1F4E79; border-radius: 6px;
  padding: 1rem; background: #fbfcfe;
}
.map-head { display: flex; justify-content: space-between; align-items: flex-start;
  gap: 1rem; flex-wrap: wrap; margin-bottom: 0.75rem; }
.map-head h3 { margin: 0; }
.map-sub { display: block; font-weight: 400; font-size: 0.8rem; color: #666; }
.map-actions { display: flex; gap: 0.5rem; align-items: center; }
.map-select { padding: 3px 6px; font-size: 0.8rem; border: 1px solid #c3ccd9;
  border-radius: 3px; background: #fff; }
.map-select.map-missing { border-color: #c0392b; background: #fff5f4; }
.map-warnings { font-size: 0.82rem; color: #7a5c00; background: #fff8e5;
  border-left: 3px solid #e0a800; padding: 0.5rem 0.75rem; margin-bottom: 0.5rem; }
.map-blocker { font-size: 0.82rem; color: #8a1c14; background: #fff1f0;
  border-left: 3px solid #c0392b; padding: 0.5rem 0.75rem; margin-bottom: 0.5rem; }
.map-note { font-size: 0.82rem; color: #14507a; background: #eef5fc;
  border-left: 3px solid #1a73e8; padding: 0.5rem 0.75rem; margin-bottom: 0.5rem; }
.map-row-off { opacity: 0.5; }
.map-samples { font-size: 0.78rem; color: #555; }
.map-count { font-size: 0.72rem; color: #888; font-weight: 400; margin-left: 0.4rem; }
.map-na { color: #bbb; }
.map-excluded { font-size: 0.78rem; color: #666; margin-top: 0.6rem; }
.map-report { margin-top: 1rem; border: 1px solid #d7e3ee; border-radius: 6px;
  padding: 0.75rem 1rem; background: #f7fbff; }
.map-report h3 { margin: 0 0 0.5rem; font-size: 0.95rem; }
.map-report-row { font-size: 0.82rem; margin-bottom: 0.25rem; }
.wrap { max-width: 220px; white-space: normal; word-break: break-word; }
/* Tenant names are the tallest thing in the roster — "Republic Finance #289" wraps
   to three lines and takes the whole row with it. Fixed width, clipped with an
   ellipsis: the identifying part of a retail name is the front of it, so
   "DSW Designer Shoe Warehouse #29460" still reads as DSW. Full name on hover.
   max-width alone is advisory under table-layout:auto, so the width is set
   outright and the cell is a block to make overflow apply. */
.tenant-name { font-weight: 500; max-width: 190px; }
/* The clip lives on an inner block, not the td: max-width on a table-cell is
   advisory under table-layout:auto, and display:block on a td would drop it out
   of the row. */
.tenant-name .tname {
  display: block;
  max-width: 190px;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}
/* Short identifier fields. "Specialty Lease" and a two-suite tenant like
   "N630, N640-A" were the only rows left at double height once the tenant name
   was capped; neither reads better broken across two lines. */
.nowrap-cell { white-space: nowrap; }
.doc-bulk {
  display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
  padding: 6px 10px; margin: 0.35rem 0 0.6rem; font-size: 0.82rem;
  background: #fff1f0; border-left: 3px solid #C00000; border-radius: 3px;
}
.doc-bulk-all { display: flex; align-items: center; gap: 6px; cursor: pointer; }
.doc-bulk-count { color: #8a1c14; font-weight: 500; }
.doc-warn { color: #8a1c14; font-weight: 600; }
.doc-del {
  border: 1px solid #c3ccd9; background: #fff; color: #8a1c14;
  padding: 2px 8px; border-radius: 3px; cursor: pointer; font-size: 0.74rem;
}
.doc-del:hover:not(:disabled) { background: #fff1f0; border-color: #C00000; }
.doc-del-yes {
  border: 1px solid #C00000; background: #C00000; color: #fff;
  padding: 2px 10px; border-radius: 3px; cursor: pointer; font-size: 0.74rem; font-weight: 600;
}
.roster-hidden {
  font-size: 0.8rem; color: #7a5c00; background: #fff8e5;
  border-left: 3px solid #e0a800; padding: 6px 10px; border-radius: 3px;
  margin: 0.25rem 0 0.6rem; display: flex; align-items: center; gap: 8px;
}

/* Tenant disposition — the reading, not the rent roll's own vacancy flag */
.disp-help { font-size: 0.78rem; color: #666; margin: 0.15rem 0 0.5rem; }
.disp-table { margin-top: 0.25rem; background: #fff; }
.disp-btn {
  border: 1px solid #c3ccd9; background: #fff; color: #444;
  padding: 2px 8px; margin-right: 4px; border-radius: 3px;
  font-size: 0.74rem; cursor: pointer;
}
.disp-btn:hover:not(:disabled) { background: #eef5fc; border-color: #1a73e8; }
.disp-btn:disabled { opacity: 0.5; cursor: default; }
.disp-btn.disp-on { background: #1F4E79; border-color: #1F4E79; color: #fff; font-weight: 600; }
.disp-err { color: #c0392b; font-size: 0.74rem; margin-left: 0.4rem; }
.disp-select { font-size: 0.76rem; }
.disp-bulk {
  display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;
  padding: 0.4rem 0.6rem; margin-top: 0.25rem;
  background: #eef5fc; border-left: 3px solid #1a73e8; border-radius: 3px;
  font-size: 0.8rem;
}
.disp-bulk-all { display: flex; align-items: center; gap: 0.35rem; cursor: pointer; }
.disp-bulk-count { color: #14507a; font-weight: 500; }
.disp-check { width: 28px; text-align: center; }
/* A reading other than "on the rent roll" means the row is out of the projection,
   so it should not look like the rows that are in it. */
.disp-select.disp-off { background: #fff6e5; border-color: #e0a800; color: #7a5c00; }
.map-label { font-weight: 500; }
.editable-cell { cursor: pointer; }
.editable-cell:hover { background: #e8f0fe; }
.inline-edit {
  width: 90px; padding: 2px 4px; font-size: 0.82rem;
  text-align: right; border: 1px solid #1a73e8; border-radius: 3px;
}
.override-val { color: #1a73e8; font-weight: 500; }
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
