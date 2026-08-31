<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import api from '../api/client'
import { useAuthStore } from '../stores/auth'

const route = useRoute()
const router = useRouter()
const auth = useAuthStore()

// ------------------------------------------------------------
// Types
// ------------------------------------------------------------
interface Cycle {
  id: number
  year: number
  as_of_date: string
  status: string
  record_count: number
}
interface RecordRow {
  id: number
  vcode: string
  deal_name: string
  asset_type: string
  is_child: boolean
  parent_vcode: string | null
  effective_classification: string
  classification_reason: string
  classification_override: string | null
  method: string | null
  concluded_value: number | null
  prior_value: number | null
  value_change: number | null
  status: string
  doc_count: number
  has_appraisal: boolean
  has_argus: boolean
}

// ------------------------------------------------------------
// State
// ------------------------------------------------------------
const cycles = ref<Cycle[]>([])
const selectedCycleId = ref<number | null>(null)
const records = ref<RecordRow[]>([])
const loading = ref(false)
const error = ref<string | null>(null)
const statusFilter = ref('')
const classFilter = ref('')
const searchText = ref('')

const selectedRecordId = ref<number | null>(null)
const record = ref<any | null>(null)
const recordLoading = ref(false)
const activeTab = ref<'assumptions' | 'budget' | 'balance' | 'qa' | 'ai' | 'nav'>('assumptions')
const saving = ref(false)
const saveMsg = ref('')

// Phase 2 state
const perms = ref<{ committee_roles: string[]; can_approve: boolean; is_recorder: boolean }>({
  committee_roles: [], can_approve: false, is_recorder: false,
})
const viewMode = ref<'records' | 'committee'>('records')
const committee = ref<any | null>(null)
const committeeLoading = ref(false)
const newQuestion = ref('')
const answerDrafts = ref<Record<number, string>>({})
const qaBusy = ref(false)
const aiSummary = ref<any | null>(null)
const aiLoading = ref(false)
const aiGenerating = ref(false)

// Phase 4 — checks + apply-extracted
const checksData = ref<any | null>(null)
const checksExpanded = ref(false)
const applying = ref(false)

// Phase 3 — NAV
const navData = ref<any | null>(null)
const navLoading = ref(false)
const navComputing = ref(false)
const navSelectionsDirty = ref(false)
const publishing = ref(false)
const refDrafts = ref<Record<number, string>>({})

const budgetReview = ref<any | null>(null)
const budgetLoading = ref(false)
const balanceSheet = ref<any | null>(null)
const balanceLoading = ref(false)

const uploadingDocs = ref(false)
const uploadDocType = ref('appraisal')
const uploadingArgus = ref(false)
const comments = ref<Record<string, string>>({ budget_review: '', balance_sheet: '', general: '' })
const commentSaving = ref<Record<string, boolean>>({})

// Assumption form
const form = ref<Record<string, any>>({})
const FORM_FIELDS = [
  'method', 'concluded_value', 'cap_rate', 'term_cap_rate', 'discount_rate',
  'direct_cap_noi', 'cost_of_sale_pct', 'appraiser', 'appraisal_date',
  'classification_override', 'override_note',
]

// Comma-formatted dollar inputs (Concluded Value, Direct Cap NOI): the form
// keeps the numeric value; these hold the display string shown in the input.
const CURRENCY_FIELDS = ['concluded_value', 'direct_cap_noi'] as const
const currencyDisplay = ref<Record<string, string>>({ concluded_value: '', direct_cap_noi: '' })

function fmtInputNumber(v: any): string {
  if (v === null || v === undefined || v === '' || isNaN(Number(v))) return ''
  return Number(v).toLocaleString('en-US', { maximumFractionDigits: 2 })
}

function syncCurrencyDisplays() {
  for (const f of CURRENCY_FIELDS) currencyDisplay.value[f] = fmtInputNumber(form.value[f])
}

function onCurrencyInput(field: string, e: Event) {
  const raw = (e.target as HTMLInputElement).value
  currencyDisplay.value[field] = raw
  const n = parseFloat(raw.replace(/[$,\s]/g, ''))
  form.value[field] = isNaN(n) ? null : n
}

function onCurrencyBlur(field: string) {
  currencyDisplay.value[field] = fmtInputNumber(form.value[field])
}

// ------------------------------------------------------------
// Computed
// ------------------------------------------------------------
const selectedCycle = computed(() => cycles.value.find(c => c.id === selectedCycleId.value) || null)

const filteredRecords = computed(() => {
  let rows = records.value
  if (statusFilter.value) rows = rows.filter(r => r.status === statusFilter.value)
  if (classFilter.value) rows = rows.filter(r => r.effective_classification === classFilter.value)
  if (searchText.value.trim()) {
    const q = searchText.value.trim().toLowerCase()
    rows = rows.filter(r => r.deal_name.toLowerCase().includes(q) || r.vcode.toLowerCase().includes(q))
  }
  return rows
})

const countBy = (fn: (r: RecordRow) => boolean) => records.value.filter(fn).length
const openCount = computed(() => countBy(r => r.status === 'open'))
const signedCount = computed(() => countBy(r => r.status === 'signed_off'))
const thirdPartyCount = computed(() => countBy(r => r.effective_classification === 'third_party'))
const internalCount = computed(() => countBy(r => r.effective_classification === 'internal'))
const costCount = computed(() => countBy(r => r.effective_classification === 'cost'))

const canEdit = computed(() => auth.isAnalyst)
const recordEditable = computed(() => record.value && record.value.status === 'open' && canEdit.value)
// Comments and Q&A answers stay editable through committee review, lock on approval
const commentsEditable = computed(() => record.value && record.value.status !== 'approved' && canEdit.value)
const openQuestions = computed(() =>
  (record.value?.questions || []).filter((q: any) => q.status === 'open').length)

// ------------------------------------------------------------
// Loaders
// ------------------------------------------------------------
async function loadCycles() {
  try {
    const res = await api.get('/api/valuations/cycles')
    cycles.value = res.data.cycles || []
    if (!selectedCycleId.value && cycles.value.length) {
      const fromQuery = Number(route.query.cycle)
      selectedCycleId.value = cycles.value.find(c => c.id === fromQuery)?.id ?? cycles.value[0].id
    }
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  }
}

async function loadDashboard() {
  if (!selectedCycleId.value) return
  loading.value = true
  error.value = null
  try {
    const res = await api.get(`/api/valuations/cycles/${selectedCycleId.value}/dashboard`)
    records.value = res.data.records || []
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    loading.value = false
  }
}

async function openRecord(id: number) {
  selectedRecordId.value = id
  router.replace({ query: { ...route.query, cycle: String(selectedCycleId.value), record: String(id) } })
  record.value = null
  budgetReview.value = null
  balanceSheet.value = null
  aiSummary.value = null
  navData.value = null
  checksData.value = null
  activeTab.value = 'assumptions'
  recordLoading.value = true
  try {
    const res = await api.get(`/api/valuations/records/${id}`)
    record.value = res.data
    for (const f of FORM_FIELDS) form.value[f] = res.data[f]
    syncCurrencyDisplays()
    comments.value = {
      budget_review: res.data.comments?.budget_review?.comment_text || '',
      balance_sheet: res.data.comments?.balance_sheet?.comment_text || '',
      general: res.data.comments?.general?.comment_text || '',
    }
    loadChecks() // non-blocking tie-out checks
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    recordLoading.value = false
  }
}

function closeRecord() {
  selectedRecordId.value = null
  record.value = null
  const q = { ...route.query }
  delete q.record
  router.replace({ query: q })
  loadDashboard()
}

async function loadBudgetReview() {
  if (!selectedRecordId.value || budgetReview.value) return
  budgetLoading.value = true
  try {
    const res = await api.get(`/api/valuations/records/${selectedRecordId.value}/budget-review`)
    budgetReview.value = res.data
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    budgetLoading.value = false
  }
}

async function loadBalanceSheet() {
  if (!selectedRecordId.value || balanceSheet.value) return
  balanceLoading.value = true
  try {
    const res = await api.get(`/api/valuations/records/${selectedRecordId.value}/balance-sheet`)
    balanceSheet.value = res.data
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    balanceLoading.value = false
  }
}

async function loadAiSummary(force = false) {
  if (!selectedRecordId.value || (aiSummary.value && !force)) return
  aiLoading.value = true
  try {
    const res = await api.get(`/api/valuations/records/${selectedRecordId.value}/ai-summary`)
    aiSummary.value = res.data.exists ? res.data : { exists: false }
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    aiLoading.value = false
  }
}

async function loadChecks() {
  if (!selectedRecordId.value) return
  try {
    const res = await api.get(`/api/valuations/records/${selectedRecordId.value}/checks`)
    checksData.value = res.data
  } catch { /* non-blocking */ }
}

const CHECK_FIELD_MAP: Record<string, string> = {
  'Concluded Value': 'concluded_value',
  'Going-in Cap Rate': 'cap_rate',
  'Terminal Cap Rate': 'term_cap_rate',
  'Discount Rate': 'discount_rate',
  'Cost of Sale %': 'cost_of_sale_pct',
  'Direct Cap NOI': 'direct_cap_noi',
}

async function applyExtracted(check: any) {
  const field = CHECK_FIELD_MAP[check.field]
  if (!field || check.extracted == null || !selectedRecordId.value) return
  applying.value = true
  try {
    await api.put(`/api/valuations/records/${selectedRecordId.value}`, { [field]: check.extracted })
    await openRecord(selectedRecordId.value)
    activeTab.value = 'ai'
    await loadAiSummary(true)
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    applying.value = false
  }
}

async function applyAllExtracted() {
  if (!selectedRecordId.value || !aiSummary.value?.exists) return
  const body: Record<string, any> = {}
  for (const c of aiSummary.value.checks || []) {
    const field = CHECK_FIELD_MAP[c.field]
    if (field && c.extracted != null) body[field] = c.extracted
  }
  const s = aiSummary.value.summary || {}
  if (s.appraiser?.firm) body.appraiser = s.appraiser.firm
  const apprDate = s.value_conclusion?.value_date || s.appraiser?.appraisal_date
  if (apprDate && /^\d{4}-\d{2}-\d{2}/.test(apprDate)) body.appraisal_date = apprDate.slice(0, 10)
  if (!Object.keys(body).length) return
  applying.value = true
  try {
    await api.put(`/api/valuations/records/${selectedRecordId.value}`, body)
    saveMsg.value = `Applied ${Object.keys(body).length} extracted value(s) to the record`
    setTimeout(() => (saveMsg.value = ''), 4000)
    await openRecord(selectedRecordId.value)
    activeTab.value = 'ai'
    await loadAiSummary(true)
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    applying.value = false
  }
}

async function loadNav(force = false) {
  if (!selectedRecordId.value || (navData.value && !force)) return
  navLoading.value = true
  try {
    const res = await api.get(`/api/valuations/records/${selectedRecordId.value}/nav`)
    navData.value = res.data
    navSelectionsDirty.value = false
    refDrafts.value = {}
    for (const l of navData.value?.result?.walk || []) {
      if (refDrafts.value[l.iorder] === undefined) refDrafts.value[l.iorder] = l.agreement_ref || ''
    }
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    navLoading.value = false
  }
}

watch(activeTab, (tab) => {
  if (tab === 'budget') loadBudgetReview()
  if (tab === 'balance') loadBalanceSheet()
  if (tab === 'ai') loadAiSummary()
  if (tab === 'nav') loadNav()
})

// ------------------------------------------------------------
// Phase 3 — NAV actions
// ------------------------------------------------------------
async function computeNav() {
  if (!selectedRecordId.value) return
  navComputing.value = true
  error.value = null
  try {
    if (navSelectionsDirty.value) await saveBsSelections(false)
    await api.post(`/api/valuations/records/${selectedRecordId.value}/nav/compute`, {})
    await loadNav(true)
    committee.value = null // committee summary now stale
    saveMsg.value = 'NAV computed'
    setTimeout(() => (saveMsg.value = ''), 3000)
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    navComputing.value = false
  }
}

function toggleBsLine(line: any) {
  if (!line.selectable) return
  line.included = !line.included
  navSelectionsDirty.value = true
}

async function saveBsSelections(reload = true) {
  if (!selectedRecordId.value || !navData.value?.inputs) return
  const selections: Record<string, boolean> = {}
  for (const l of navData.value.inputs.lines) {
    if (l.selectable) selections[l.account] = !!l.included
  }
  await api.put(`/api/valuations/records/${selectedRecordId.value}/bs-selections`, { selections })
  navSelectionsDirty.value = false
  if (reload) await loadNav(true)
}

async function saveStepRef(line: any) {
  if (!navData.value) return
  const ref_ = (refDrafts.value[line.iorder] || '').trim()
  try {
    await api.put('/api/valuations/step-refs', {
      vcode: navData.value.inputs.vcode, iorder: line.iorder, agreement_ref: ref_,
    })
    for (const l of navData.value.result?.walk || []) {
      if (l.iorder === line.iorder) l.agreement_ref = ref_
    }
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  }
}

function downloadNavPackage() {
  if (!selectedRecordId.value) return
  const token = localStorage.getItem('token')
  window.open(`/api/valuations/records/${selectedRecordId.value}/nav-package?token=${token}`, '_blank')
}

function downloadCyclePackages() {
  if (!selectedCycleId.value) return
  const token = localStorage.getItem('token')
  window.open(`/api/valuations/cycles/${selectedCycleId.value}/nav-packages?token=${token}`, '_blank')
}

async function publishRecord() {
  if (!selectedRecordId.value) return
  if (!confirm('Publish this approved valuation? It becomes the valuation of record '
    + '(valuations table + Val_IS forecast) for every downstream report.')) return
  publishing.value = true
  try {
    const res = await api.post(`/api/valuations/records/${selectedRecordId.value}/publish`, {})
    saveMsg.value = `Published — valuations row written`
      + (res.data.forecast_rows ? `, ${res.data.forecast_rows} forecast rows staged as Val_IS` : '')
    setTimeout(() => (saveMsg.value = ''), 6000)
    await openRecord(selectedRecordId.value)
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    publishing.value = false
  }
}

// ------------------------------------------------------------
// Phase 2 — permissions, Q&A, approvals, committee, AI
// ------------------------------------------------------------
async function loadPermissions() {
  try {
    const res = await api.get('/api/valuations/permissions')
    perms.value = res.data
  } catch { /* viewer defaults are fine */ }
}

async function loadCommittee(force = false) {
  if (!selectedCycleId.value || (committee.value && !force)) return
  committeeLoading.value = true
  try {
    const res = await api.get(`/api/valuations/cycles/${selectedCycleId.value}/committee-summary`)
    committee.value = res.data
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    committeeLoading.value = false
  }
}

watch(viewMode, (m) => { if (m === 'committee') loadCommittee() })

async function askQuestion() {
  if (!selectedRecordId.value || !newQuestion.value.trim()) return
  qaBusy.value = true
  try {
    await api.post(`/api/valuations/records/${selectedRecordId.value}/questions`, {
      text: newQuestion.value,
    })
    newQuestion.value = ''
    await openRecord(selectedRecordId.value)
    activeTab.value = 'qa'
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    qaBusy.value = false
  }
}

async function answerQuestion(q: any) {
  const text = (answerDrafts.value[q.id] || '').trim()
  if (!text || !selectedRecordId.value) return
  qaBusy.value = true
  try {
    await api.put(`/api/valuations/questions/${q.id}/answer`, { text })
    delete answerDrafts.value[q.id]
    await openRecord(selectedRecordId.value)
    activeTab.value = 'qa'
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    qaBusy.value = false
  }
}

async function resolveQuestion(q: any) {
  if (!selectedRecordId.value) return
  try {
    await api.post(`/api/valuations/questions/${q.id}/resolve`)
    await openRecord(selectedRecordId.value)
    activeTab.value = 'qa'
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  }
}

async function approveRecord() {
  if (!selectedRecordId.value) return
  try {
    const res = await api.post(`/api/valuations/records/${selectedRecordId.value}/approve`, {})
    saveMsg.value = res.data.status === 'approved'
      ? 'Unanimously approved — valuation locked and snapshot saved'
      : `Approval recorded — waiting on: ${res.data.missing_roles.join(', ')}`
    setTimeout(() => (saveMsg.value = ''), 5000)
    await openRecord(selectedRecordId.value)
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  }
}

async function returnRecord() {
  if (!selectedRecordId.value) return
  const note = prompt('Return to the asset manager — what needs to change? (required)')
  if (!note?.trim()) return
  try {
    await api.post(`/api/valuations/records/${selectedRecordId.value}/return`, { note })
    await openRecord(selectedRecordId.value)
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  }
}

async function approveAllSigned() {
  if (!selectedCycleId.value) return
  if (!confirm('Approve every signed-off valuation in this cycle as your committee role(s)?')) return
  try {
    const res = await api.post(`/api/valuations/cycles/${selectedCycleId.value}/approve-all`)
    saveMsg.value = `Approved ${res.data.approved_by_member} record(s); ${res.data.fully_approved} now fully approved`
    setTimeout(() => (saveMsg.value = ''), 5000)
    await loadCommittee(true)
    await loadDashboard()
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  }
}

async function downloadCommitteeExcel() {
  if (!selectedCycleId.value) return
  try {
    const res = await api.get(`/api/valuations/cycles/${selectedCycleId.value}/committee-excel`, {
      responseType: 'blob',
    })
    const url = URL.createObjectURL(res.data)
    const a = document.createElement('a')
    a.href = url
    a.download = 'valuation_committee_summary.xlsx'
    a.click()
    URL.revokeObjectURL(url)
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  }
}

async function generateAiSummary() {
  if (!selectedRecordId.value) return
  aiGenerating.value = true
  error.value = null
  try {
    const res = await api.post(`/api/valuations/records/${selectedRecordId.value}/ai-summary`, {})
    aiSummary.value = res.data
    saveMsg.value = 'AI appraisal summary generated'
    setTimeout(() => (saveMsg.value = ''), 4000)
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    aiGenerating.value = false
  }
}

// ------------------------------------------------------------
// Actions
// ------------------------------------------------------------
async function createCycle() {
  const year = prompt('Valuation year (e.g. 2026):', String(new Date().getFullYear()))
  if (!year) return
  try {
    await api.post('/api/valuations/cycles', { year: Number(year) })
    await loadCycles()
    const created = cycles.value.find(c => c.year === Number(year))
    if (created) selectedCycleId.value = created.id
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  }
}

async function reseedCycle() {
  if (!selectedCycleId.value) return
  try {
    const res = await api.post(`/api/valuations/cycles/${selectedCycleId.value}/reseed`)
    saveMsg.value = `Seeded ${res.data.seeded} new record(s)`
    setTimeout(() => (saveMsg.value = ''), 4000)
    loadDashboard()
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  }
}

async function saveAssumptions() {
  if (!selectedRecordId.value) return
  saving.value = true
  saveMsg.value = ''
  try {
    const body: Record<string, any> = {}
    for (const f of FORM_FIELDS) body[f] = form.value[f] ?? null
    await api.put(`/api/valuations/records/${selectedRecordId.value}`, body)
    saveMsg.value = 'Saved'
    setTimeout(() => (saveMsg.value = ''), 3000)
    await openRecord(selectedRecordId.value)
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    saving.value = false
  }
}

async function recordAction(action: string) {
  if (!selectedRecordId.value) return
  try {
    await api.post(`/api/valuations/records/${selectedRecordId.value}/action`, { action })
    await openRecord(selectedRecordId.value)
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  }
}

async function onDocumentUpload(event: Event) {
  const input = event.target as HTMLInputElement
  if (!input.files?.length || !selectedRecordId.value) return
  uploadingDocs.value = true
  try {
    // One file per request — matches the app-wide pattern (avoids server OOM)
    for (const file of Array.from(input.files)) {
      const formData = new FormData()
      formData.append('files', file)
      formData.append('doc_type', uploadDocType.value)
      await api.post(`/api/valuations/records/${selectedRecordId.value}/documents`, formData)
    }
    await openRecord(selectedRecordId.value)
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    uploadingDocs.value = false
    input.value = ''
  }
}

async function deleteDocument(docId: number) {
  if (!selectedRecordId.value) return
  if (!confirm('Remove this document?')) return
  try {
    await api.delete(`/api/valuations/records/${selectedRecordId.value}/documents/${docId}`)
    await openRecord(selectedRecordId.value)
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  }
}

function viewDocument(docId: number) {
  if (!selectedRecordId.value) return
  const token = localStorage.getItem('token')
  window.open(
    `/api/valuations/records/${selectedRecordId.value}/documents/${docId}/view?token=${token}`,
    '_blank',
  )
}

async function onArgusUpload(event: Event) {
  const input = event.target as HTMLInputElement
  if (!input.files?.length || !selectedRecordId.value) return
  uploadingArgus.value = true
  try {
    const formData = new FormData()
    formData.append('file', input.files[0])
    const res = await api.post(`/api/valuations/records/${selectedRecordId.value}/argus`, formData)
    saveMsg.value = `Argus imported (#${res.data.import_id}) — ${res.data.mapped_count ?? '?'} line items mapped`
    setTimeout(() => (saveMsg.value = ''), 5000)
    budgetReview.value = null
    await openRecord(selectedRecordId.value)
  } catch (e: any) {
    if (e.response?.status === 409) {
      saveMsg.value = 'File already imported — linked existing import'
      setTimeout(() => (saveMsg.value = ''), 4000)
      await openRecord(selectedRecordId.value)
    } else {
      error.value = e.response?.data?.error || e.response?.data?.message || e.message
    }
  } finally {
    uploadingArgus.value = false
    input.value = ''
  }
}

async function saveComment(section: string) {
  if (!selectedRecordId.value) return
  commentSaving.value[section] = true
  try {
    await api.put(`/api/valuations/records/${selectedRecordId.value}/comments`, {
      section, text: comments.value[section] || '',
    })
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    commentSaving.value[section] = false
  }
}

async function printForm() {
  await Promise.all([loadBudgetReview(), loadBalanceSheet()])
  setTimeout(() => window.print(), 300)
}

function filterByStatus(s: string) {
  statusFilter.value = statusFilter.value === s ? '' : s
}
function filterByClass(c: string) {
  classFilter.value = classFilter.value === c ? '' : c
}

// ------------------------------------------------------------
// Formatting
// ------------------------------------------------------------
function fmtCurrency(v: number | null | undefined): string {
  if (v === null || v === undefined) return '—'
  const neg = v < 0
  const s = Math.abs(v).toLocaleString('en-US', { maximumFractionDigits: 0 })
  return neg ? `(${s})` : s
}
function fmtPct(v: number | null | undefined): string {
  if (v === null || v === undefined) return '—'
  return (v * 100).toFixed(2) + '%'
}
function fmtRatio(v: number | null | undefined): string {
  if (v === null || v === undefined) return '—'
  return v.toFixed(2) + 'x'
}
// Assumption Cross-Check: rate fields as percentages, dollar fields with commas
const PCT_CHECK_FIELDS = new Set(['Going-in Cap Rate', 'Terminal Cap Rate', 'Discount Rate', 'Cost of Sale %'])
function fmtCheckValue(field: string, v: any): string {
  if (v === null || v === undefined || v === '') return '—'
  const n = Number(v)
  if (isNaN(n)) return String(v)
  return PCT_CHECK_FIELDS.has(field) ? fmtPct(n) : fmtCurrency(n)
}
function fmtDate(d: string | null | undefined): string {
  if (!d) return '—'
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(d)
  if (m) return `${Number(m[2])}/${Number(m[3])}/${m[1]}`
  return d
}
const CLASS_LABELS: Record<string, string> = {
  third_party: '3rd Party', internal: 'Internal', cost: 'Cost',
}
const STATUS_LABELS: Record<string, string> = {
  open: 'Open', signed_off: 'Signed Off', approved: 'Approved', excluded: 'Excluded',
}
const ROLE_LABELS: Record<string, string> = { president: 'President', ceo: 'CEO', cio: 'CIO' }
const COMMITTEE_ROLES = ['president', 'ceo', 'cio']
function classBadge(c: string) {
  return { third_party: 'badge-third', internal: 'badge-internal', cost: 'badge-cost' }[c] || 'badge-internal'
}
function statusBadge(s: string) {
  return {
    open: 'status-open', signed_off: 'status-signed',
    approved: 'status-approved', excluded: 'status-excluded',
  }[s] || 'status-open'
}
function fmtOcc(v: number | null | undefined): string {
  if (v === null || v === undefined) return '—'
  // The model may return a fraction (1.0) or a percentage (100)
  return (v <= 1 ? v * 100 : v).toFixed(0) + '%'
}
function fmtDateTime(d: string | null | undefined): string {
  if (!d) return ''
  return fmtDate(d.slice(0, 10))
}

onMounted(async () => {
  loadPermissions()
  await loadCycles()
  await loadDashboard()
  const rec = Number(route.query.record)
  if (rec) openRecord(rec)
})
watch(selectedCycleId, () => {
  router.replace({ query: { ...route.query, cycle: String(selectedCycleId.value) } })
  committee.value = null
  loadDashboard()
  if (viewMode.value === 'committee') loadCommittee(true)
})
</script>

<template>
  <div class="valuations">
    <div v-if="error" class="error-banner no-print">
      {{ error }}
      <button @click="error = null">Dismiss</button>
    </div>
    <div v-if="saveMsg" class="save-banner no-print">{{ saveMsg }}</div>

    <!-- ================= Dashboard ================= -->
    <template v-if="!selectedRecordId">
      <div class="header-row no-print">
        <h2>Valuations</h2>
        <div class="header-controls">
          <div class="view-toggle">
            <button :class="{ active: viewMode === 'records' }" @click="viewMode = 'records'">Records</button>
            <button :class="{ active: viewMode === 'committee' }" @click="viewMode = 'committee'">Committee Summary</button>
          </div>
          <select v-model.number="selectedCycleId" class="cycle-select">
            <option v-for="c in cycles" :key="c.id" :value="c.id">
              {{ c.year }} Cycle (as of {{ fmtDate(c.as_of_date) }})
            </option>
          </select>
          <button v-if="auth.isAdmin" class="btn-secondary" @click="reseedCycle" :disabled="!selectedCycleId">
            Reseed
          </button>
          <button v-if="auth.isAdmin" class="btn-primary" @click="createCycle">New Cycle</button>
        </div>
      </div>

      <div v-if="!cycles.length && !loading" class="empty-state">
        No valuation cycles yet.
        <span v-if="auth.isAdmin">Click "New Cycle" to open one — every active deal gets a record, pre-classified per the valuation policy.</span>
      </div>

      <template v-else>
        <template v-if="viewMode === 'records'">
        <div class="summary-cards no-print">
          <div class="summary-card" :class="{ active: statusFilter === 'open' }" @click="filterByStatus('open')">
            <span class="card-count">{{ openCount }}</span>
            <span class="card-label">Open</span>
          </div>
          <div class="summary-card card-signed" :class="{ active: statusFilter === 'signed_off' }" @click="filterByStatus('signed_off')">
            <span class="card-count">{{ signedCount }}</span>
            <span class="card-label">Signed Off</span>
          </div>
          <div class="summary-card card-third" :class="{ active: classFilter === 'third_party' }" @click="filterByClass('third_party')">
            <span class="card-count">{{ thirdPartyCount }}</span>
            <span class="card-label">3rd Party</span>
          </div>
          <div class="summary-card" :class="{ active: classFilter === 'internal' }" @click="filterByClass('internal')">
            <span class="card-count">{{ internalCount }}</span>
            <span class="card-label">Internal</span>
          </div>
          <div class="summary-card card-cost" :class="{ active: classFilter === 'cost' }" @click="filterByClass('cost')">
            <span class="card-count">{{ costCount }}</span>
            <span class="card-label">Cost</span>
          </div>
          <input v-model="searchText" class="search-input" placeholder="Search deals..." />
        </div>

        <div v-if="loading" class="loading-text">Loading valuation records...</div>
        <div class="records-table-wrap" v-else>
          <table class="records-table">
            <thead>
              <tr>
                <th>Deal</th>
                <th>Classification</th>
                <th>Method</th>
                <th class="num">Concluded Value</th>
                <th class="num">Prior Value</th>
                <th class="num">&Delta;</th>
                <th>Docs</th>
                <th>Argus</th>
                <th title="Open reviewer questions">Q</th>
                <th title="Committee approvals">Appr.</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="r in filteredRecords" :key="r.id" class="clickable-row"
                  :class="{ 'child-row': r.is_child }" @click="openRecord(r.id)">
                <td class="deal-name">
                  <span v-if="r.is_child" class="child-indent">&#x21B3;</span>
                  {{ r.deal_name }}
                  <span class="vcode-tag">{{ r.vcode }}</span>
                </td>
                <td>
                  <span class="badge" :class="classBadge(r.effective_classification)"
                        :title="r.classification_reason">
                    {{ CLASS_LABELS[r.effective_classification] || r.effective_classification }}
                    <span v-if="r.classification_override">*</span>
                  </span>
                </td>
                <td>{{ r.method || '—' }}</td>
                <td class="num">{{ fmtCurrency(r.concluded_value) }}</td>
                <td class="num">{{ fmtCurrency(r.prior_value) }}</td>
                <td class="num" :class="{ pos: (r.value_change ?? 0) > 0, neg: (r.value_change ?? 0) < 0 }">
                  {{ fmtCurrency(r.value_change) }}
                </td>
                <td>
                  <span v-if="r.has_appraisal" class="mini-badge" title="Appraisal uploaded">A</span>
                  <span v-if="r.doc_count" class="doc-count">{{ r.doc_count }}</span>
                </td>
                <td><span v-if="r.has_argus" class="mini-badge argus">CF</span></td>
                <td><span v-if="(r as any).open_questions" class="mini-badge qbadge">{{ (r as any).open_questions }}</span></td>
                <td>
                  <span v-if="r.status === 'signed_off' || r.status === 'approved'" class="appr-count">
                    {{ (r as any).approval_count || 0 }}/3
                  </span>
                </td>
                <td><span class="status-badge" :class="statusBadge(r.status)">{{ STATUS_LABELS[r.status] || r.status }}</span></td>
              </tr>
              <tr v-if="!filteredRecords.length">
                <td colspan="11" class="empty-row">No records match.</td>
              </tr>
            </tbody>
          </table>
        </div>
        </template>

        <!-- ===== Committee Summary view ===== -->
        <template v-else>
          <div class="committee-actions no-print">
            <button class="btn-secondary" @click="downloadCommitteeExcel">Download Committee Workbook</button>
            <button class="btn-secondary" @click="downloadCyclePackages">Download NAV Packages (zip)</button>
            <button v-if="perms.can_approve" class="btn-primary" @click="approveAllSigned">
              Approve All Signed-Off
            </button>
            <span v-if="perms.can_approve" class="perm-note">
              You approve as: {{ perms.committee_roles.map(r => ROLE_LABELS[r] || r).join(', ') }}
            </span>
            <span v-else-if="perms.is_recorder" class="perm-note">Recorder (CCO)</span>
          </div>
          <div v-if="committeeLoading" class="loading-text">Building committee analyses...</div>
          <template v-if="committee">
            <div class="panel">
              <h3>Analysis 1 — Preferred Equity NAV</h3>
              <p class="panel-note">Pref balance and accrued pref computed from accounting through
                {{ fmtDate(committee.cycle?.as_of_date) }}. Current-cycle Pref NAV arrives with the NAV engine (Phase 3).</p>
              <div class="table-scroll">
                <table class="data-table">
                  <thead>
                    <tr>
                      <th>Deal</th><th>Status</th>
                      <th class="num">Pref Balance</th><th class="num">Accrued Pref</th>
                      <th class="num">Balance w/ Accrual</th>
                      <th class="num">Pref NAV</th><th class="num">Prior Pref NAV</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="r in committee.analysis1" :key="r.vcode" class="clickable-row"
                        @click="openRecord(records.find(x => x.vcode === r.vcode)?.id || 0)">
                      <td class="deal-name">{{ r.deal_name }}</td>
                      <td><span class="status-badge" :class="statusBadge(r.status)">{{ STATUS_LABELS[r.status] || r.status }}</span></td>
                      <td class="num">{{ fmtCurrency(r.pref_balance) }}</td>
                      <td class="num">{{ fmtCurrency(r.accrued_pref) }}</td>
                      <td class="num">{{ fmtCurrency(r.balance_with_accrual) }}</td>
                      <td class="num">{{ fmtCurrency(r.pref_nav) }}</td>
                      <td class="num">{{ fmtCurrency(r.prior_pref_nav) }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <div class="panel" style="margin-top:16px">
              <h3>Analysis 2 — Methods &amp; Values, Year over Year</h3>
              <div class="table-scroll">
                <table class="data-table">
                  <thead>
                    <tr>
                      <th>Deal</th><th>Status</th><th class="num">Q</th>
                      <th>Method (Prior &rarr; Now)</th>
                      <th class="num">Cap (Prior &rarr; Now)</th>
                      <th class="num">Exit Cap</th>
                      <th class="num">Disc</th>
                      <th class="num">NOI</th>
                      <th class="num">Prior Value</th><th class="num">Value</th>
                      <th class="num">Var</th><th class="num">Var %</th>
                      <th class="num">Debt</th><th class="num">Net Proceeds (est)</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="r in committee.analysis2" :key="r.vcode" class="clickable-row"
                        :class="{ 'child-row': r.is_child }"
                        @click="openRecord(records.find(x => x.vcode === r.vcode)?.id || 0)">
                      <td class="deal-name">
                        <span v-if="r.is_child" class="child-indent">&#x21B3;</span>{{ r.deal_name }}
                      </td>
                      <td><span class="status-badge" :class="statusBadge(r.status)">{{ STATUS_LABELS[r.status] || r.status }}</span></td>
                      <td class="num"><span v-if="r.open_questions" class="mini-badge qbadge">{{ r.open_questions }}</span></td>
                      <td>{{ r.prior_method || '—' }} &rarr; {{ r.method || '—' }}</td>
                      <td class="num">{{ fmtPct(r.prior_cap_rate) }} &rarr; {{ fmtPct(r.cap_rate) }}</td>
                      <td class="num">{{ fmtPct(r.prior_term_cap) }} &rarr; {{ fmtPct(r.term_cap) }}</td>
                      <td class="num">{{ fmtPct(r.prior_discount) }} &rarr; {{ fmtPct(r.discount) }}</td>
                      <td class="num">{{ fmtCurrency(r.noi) }}</td>
                      <td class="num">{{ fmtCurrency(r.prior_value) }}</td>
                      <td class="num">{{ fmtCurrency(r.value) }}</td>
                      <td class="num" :class="{ pos: (r.value_var ?? 0) > 0, neg: (r.value_var ?? 0) < 0 }">{{ fmtCurrency(r.value_var) }}</td>
                      <td class="num" :class="{ pos: (r.value_var ?? 0) > 0, neg: (r.value_var ?? 0) < 0 }">{{ fmtPct(r.value_var_pct) }}</td>
                      <td class="num">{{ fmtCurrency(r.debt) }}</td>
                      <td class="num">{{ fmtCurrency(r.net_proceeds) }}</td>
                      <td>
                        <span v-if="r.direction === 'Up'" class="dir-up">&#9650;</span>
                        <span v-else-if="r.direction === 'Down'" class="dir-down">&#9660;</span>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </template>
        </template>
      </template>
    </template>

    <!-- ================= Workspace ================= -->
    <template v-else>
      <div class="header-row no-print">
        <div class="header-left">
          <button class="btn-back" @click="closeRecord">&larr; Back</button>
          <h2 v-if="record">{{ record.deal?.name || record.vcode }}
            <span class="vcode-tag">{{ record.vcode }}</span>
          </h2>
        </div>
        <div class="header-controls" v-if="record">
          <div v-if="record.status === 'signed_off' || record.status === 'approved'" class="committee-chips">
            <span v-for="role in COMMITTEE_ROLES" :key="role" class="role-chip"
                  :class="{ done: record.approval_state?.approved_roles?.includes(role) }">
              {{ record.approval_state?.approved_roles?.includes(role) ? '✓' : '○' }} {{ ROLE_LABELS[role] }}
            </span>
          </div>
          <span class="status-badge" :class="statusBadge(record.status)">{{ STATUS_LABELS[record.status] || record.status }}</span>
          <button v-if="record.status === 'open' && canEdit" class="btn-primary" @click="recordAction('sign_off')">
            Analyst Sign-off
          </button>
          <button v-else-if="record.status === 'signed_off' && canEdit" class="btn-secondary" @click="recordAction('reopen')">
            Reopen
          </button>
          <button v-if="perms.can_approve && record.status === 'signed_off'
                        && !COMMITTEE_ROLES.every(r => !perms.committee_roles.includes(r) || record.approval_state?.approved_roles?.includes(r))"
                  class="btn-primary" @click="approveRecord">
            Approve
          </button>
          <button v-if="(perms.can_approve || perms.is_recorder) && (record.status === 'signed_off' || record.status === 'approved')"
                  class="btn-secondary" @click="returnRecord">
            Return
          </button>
          <span v-if="record.published_at" class="mini-badge argus" :title="'Published by ' + record.published_by">
            Published {{ fmtDateTime(record.published_at) }}
          </span>
          <button v-else-if="auth.isAdmin && record.status === 'approved'" class="btn-primary"
                  @click="publishRecord" :disabled="publishing">
            {{ publishing ? 'Publishing...' : 'Publish' }}
          </button>
          <button class="btn-secondary" @click="printForm">Print</button>
        </div>
      </div>

      <div v-if="recordLoading" class="loading-text">Loading record...</div>

      <template v-if="record">
        <div class="deal-strip">
          <div class="strip-item"><span class="strip-label">Cycle</span>{{ record.cycle?.year }} (as of {{ fmtDate(record.cycle?.as_of_date) }})</div>
          <div class="strip-item"><span class="strip-label">Asset Type</span>{{ record.deal?.asset_type || '—' }}</div>
          <div class="strip-item"><span class="strip-label">Location</span>{{ [record.deal?.city, record.deal?.state].filter(Boolean).join(', ') || '—' }}</div>
          <div class="strip-item"><span class="strip-label">Operating Partner</span>{{ record.deal?.operating_partner || '—' }}</div>
          <div class="strip-item"><span class="strip-label">Acquired</span>{{ fmtDate(record.deal?.acquisition_date) }}</div>
          <div class="strip-item"><span class="strip-label">PE Funded</span>${{ fmtCurrency(record.pe_funded) }}</div>
          <div class="strip-item">
            <span class="strip-label">Classification</span>
            <span class="badge" :class="classBadge(record.effective_classification)" :title="record.classification_reason">
              {{ CLASS_LABELS[record.effective_classification] || record.effective_classification }}
              <span v-if="record.classification_override">*</span>
            </span>
          </div>
        </div>
        <div class="class-reason">{{ record.classification_reason }}</div>

        <div v-if="checksData" class="checks-strip no-print">
          <button class="checks-toggle" @click="checksExpanded = !checksExpanded">
            {{ checksExpanded ? '▾' : '▸' }} Tie-out checks:
            <span v-if="checksData.counts.fail" class="mini-badge mismatch">{{ checksData.counts.fail }} failed</span>
            <span v-if="checksData.counts.warn" class="mini-badge qbadge">{{ checksData.counts.warn }} warning{{ checksData.counts.warn > 1 ? 's' : '' }}</span>
            <span v-if="checksData.counts.info" class="mini-badge">{{ checksData.counts.info }} info</span>
            <span class="mini-badge argus">{{ checksData.counts.ok }} passed</span>
            <span class="btn-link" style="margin-left:8px" @click.stop="loadChecks">refresh</span>
          </button>
          <div v-if="checksExpanded" class="checks-list">
            <div v-for="(c, i) in checksData.checks" :key="i" class="check-row">
              <span class="mini-badge" :class="{
                mismatch: c.severity === 'fail',
                qbadge: c.severity === 'warn',
                argus: c.severity === 'ok',
              }">{{ c.severity }}</span>
              <span>{{ c.message }}</span>
            </div>
          </div>
        </div>

        <div class="tabs no-print">
          <button :class="{ active: activeTab === 'assumptions' }" @click="activeTab = 'assumptions'">Assumptions &amp; Documents</button>
          <button :class="{ active: activeTab === 'budget' }" @click="activeTab = 'budget'">Budget Review</button>
          <button :class="{ active: activeTab === 'balance' }" @click="activeTab = 'balance'">Balance Sheet</button>
          <button :class="{ active: activeTab === 'qa' }" @click="activeTab = 'qa'">
            Q&amp;A <span v-if="openQuestions" class="mini-badge qbadge">{{ openQuestions }}</span>
          </button>
          <button :class="{ active: activeTab === 'ai' }" @click="activeTab = 'ai'">AI Summary</button>
          <button :class="{ active: activeTab === 'nav' }" @click="activeTab = 'nav'">NAV</button>
        </div>

        <!-- ===== Tab: Assumptions & Documents ===== -->
        <div v-show="activeTab === 'assumptions'" class="tab-panel">
          <div class="panel-grid">
            <div class="panel">
              <h3>Valuation Assumptions</h3>
              <div class="form-grid">
                <label>Method
                  <select v-model="form.method" :disabled="!recordEditable">
                    <option :value="null">—</option>
                    <option>DCF</option>
                    <option>Direct Cap</option>
                    <option>Sales Comp</option>
                    <option>Cost</option>
                    <option>Various</option>
                  </select>
                </label>
                <label>Concluded Value
                  <input type="text" inputmode="decimal" :value="currencyDisplay.concluded_value"
                         @input="onCurrencyInput('concluded_value', $event)"
                         @blur="onCurrencyBlur('concluded_value')" :disabled="!recordEditable" />
                </label>
                <label>Going-in Cap Rate
                  <input type="number" step="0.0001" v-model.number="form.cap_rate" :disabled="!recordEditable" placeholder="0.0725" />
                </label>
                <label>Terminal Cap Rate
                  <input type="number" step="0.0001" v-model.number="form.term_cap_rate" :disabled="!recordEditable" placeholder="0.0750" />
                </label>
                <label>Discount Rate
                  <input type="number" step="0.0001" v-model.number="form.discount_rate" :disabled="!recordEditable" placeholder="0.0850" />
                </label>
                <label>Direct Cap NOI
                  <input type="text" inputmode="decimal" :value="currencyDisplay.direct_cap_noi"
                         @input="onCurrencyInput('direct_cap_noi', $event)"
                         @blur="onCurrencyBlur('direct_cap_noi')" :disabled="!recordEditable" />
                </label>
                <label>Cost of Sale %
                  <input type="number" step="0.001" v-model.number="form.cost_of_sale_pct" :disabled="!recordEditable" placeholder="0.02" />
                </label>
                <label>Appraiser
                  <input type="text" v-model="form.appraiser" :disabled="!recordEditable" />
                </label>
                <label>Appraisal Date
                  <input type="date" v-model="form.appraisal_date" :disabled="!recordEditable" />
                </label>
                <label>Classification Override
                  <select v-model="form.classification_override" :disabled="!recordEditable">
                    <option :value="null">None (use policy default)</option>
                    <option value="third_party">3rd Party</option>
                    <option value="internal">Internal</option>
                    <option value="cost">Cost</option>
                  </select>
                </label>
                <label class="span-2" v-if="form.classification_override">Override Note (required)
                  <input type="text" v-model="form.override_note" :disabled="!recordEditable"
                         placeholder="Why this deal departs from the policy default" />
                </label>
              </div>
              <div class="form-actions no-print" v-if="recordEditable">
                <button class="btn-primary" @click="saveAssumptions" :disabled="saving">
                  {{ saving ? 'Saving...' : 'Save Assumptions' }}
                </button>
              </div>
            </div>

            <div class="panel">
              <h3>Documents</h3>
              <div class="doc-controls no-print" v-if="recordEditable">
                <select v-model="uploadDocType">
                  <option value="appraisal">Appraisal</option>
                  <option value="llc_excerpt">LLC Agreement Excerpt</option>
                  <option value="bs_support">Balance Sheet Support</option>
                  <option value="other">Other</option>
                </select>
                <label class="btn-primary btn-upload-label">
                  {{ uploadingDocs ? 'Uploading...' : 'Upload Files' }}
                  <input type="file" multiple @change="onDocumentUpload" :disabled="uploadingDocs" hidden />
                </label>
              </div>
              <table class="doc-table" v-if="record.documents?.length">
                <thead><tr><th>Type</th><th>File</th><th>By</th><th></th></tr></thead>
                <tbody>
                  <tr v-for="d in record.documents" :key="d.id">
                    <td><span class="mini-badge">{{ d.doc_type }}</span></td>
                    <td class="doc-file" @click="viewDocument(d.id)">{{ d.filename }}</td>
                    <td>{{ d.uploaded_by }}</td>
                    <td class="no-print">
                      <button v-if="recordEditable" class="btn-link danger" @click="deleteDocument(d.id)">remove</button>
                    </td>
                  </tr>
                </tbody>
              </table>
              <div v-else class="empty-note">No documents uploaded.</div>

              <h3 class="mt">Argus Projection</h3>
              <div v-if="record.argus_import_id" class="argus-linked">
                <span class="mini-badge argus">CF</span>
                Import #{{ record.argus_import_id }} linked — year-1 flows appear in the Budget Review tab.
              </div>
              <div v-else class="empty-note">No Argus import linked.</div>
              <div class="doc-controls no-print" v-if="recordEditable">
                <label class="btn-secondary btn-upload-label">
                  {{ uploadingArgus ? 'Importing...' : (record.argus_import_id ? 'Replace Argus Export' : 'Import Argus Export') }}
                  <input type="file" accept=".xlsx,.xls" @change="onArgusUpload" :disabled="uploadingArgus" hidden />
                </label>
              </div>
            </div>
          </div>

          <div class="panel" v-if="record.valuation_history?.length">
            <h3>Valuation History</h3>
            <div class="table-scroll">
              <table class="data-table">
                <thead>
                  <tr>
                    <th>Date</th><th>Method</th><th class="num">NOI</th><th class="num">Cap</th>
                    <th class="num">Term Cap</th><th class="num">Discount</th><th class="num">Value</th>
                    <th class="num">Debt</th><th class="num">PE NAV</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="(h, i) in record.valuation_history" :key="i">
                    <td>{{ fmtDate(h.date) }}</td>
                    <td>{{ h.method || '—' }}</td>
                    <td class="num">{{ fmtCurrency(h.noi) }}</td>
                    <td class="num">{{ fmtPct(h.cap_rate) }}</td>
                    <td class="num">{{ fmtPct(h.term_cap_rate) }}</td>
                    <td class="num">{{ fmtPct(h.discount_rate) }}</td>
                    <td class="num">{{ fmtCurrency(h.value) }}</td>
                    <td class="num">{{ fmtCurrency(h.debt) }}</td>
                    <td class="num">{{ fmtCurrency(h.pe_nav) }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <div class="panel">
            <h3>General Comments</h3>
            <textarea v-model="comments.general" rows="3" :disabled="!commentsEditable"
                      placeholder="Notes for the committee record..."></textarea>
            <div class="form-actions no-print" v-if="commentsEditable">
              <button class="btn-secondary" @click="saveComment('general')" :disabled="commentSaving.general">
                {{ commentSaving.general ? 'Saving...' : 'Save Comment' }}
              </button>
            </div>
          </div>
        </div>

        <!-- ===== Tab: Budget Review ===== -->
        <div v-show="activeTab === 'budget'" class="tab-panel">
          <div v-if="budgetLoading" class="loading-text">Building comparison...</div>
          <template v-if="budgetReview">
            <div class="panel">
              <h3>{{ budgetReview.budget_year }} Budget Review</h3>
              <p class="panel-note">
                {{ budgetReview.estimate_year }} Estimate = actuals through
                {{ fmtDate(budgetReview.last_actual_month) || 'n/a' }} plus budget for the remaining months.
                <span v-if="!budgetReview.has_argus" class="warn-note">
                  No Argus import linked — the Valuation column is empty until one is imported on the first tab.
                </span>
              </p>
              <div class="table-scroll">
                <table class="data-table budget-table">
                  <thead>
                    <tr>
                      <th></th>
                      <th class="num">{{ budgetReview.estimate_year }} Estimate</th>
                      <th class="num">{{ budgetReview.budget_year }} Budget</th>
                      <th class="num">Variance</th>
                      <th class="num">Valuation Yr 1</th>
                      <th class="num">Var to Budget</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="(row, i) in budgetReview.rows" :key="i"
                        :class="{ 'row-total': row.is_total, 'row-calc': row.is_calc }">
                      <td :class="{ indent: row.level === 1 }">{{ row.account }}</td>
                      <template v-if="row.is_ratio">
                        <td class="num">{{ fmtRatio(row.estimate) }}</td>
                        <td class="num">{{ fmtRatio(row.budget) }}</td>
                        <td class="num"></td>
                        <td class="num">{{ fmtRatio(row.valuation) }}</td>
                        <td class="num"></td>
                      </template>
                      <template v-else>
                        <td class="num">{{ fmtCurrency(row.estimate) }}</td>
                        <td class="num">{{ fmtCurrency(row.budget) }}</td>
                        <td class="num" :class="{ neg: (row.var_est_bud ?? 0) < 0 }">{{ fmtCurrency(row.var_est_bud) }}</td>
                        <td class="num">{{ budgetReview.has_argus ? fmtCurrency(row.valuation) : '—' }}</td>
                        <td class="num" :class="{ neg: (row.var_bud_val ?? 0) < 0 }">
                          {{ budgetReview.has_argus ? fmtCurrency(row.var_bud_val) : '—' }}
                        </td>
                      </template>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <div class="panel" v-if="budgetReview.occupancy_trend?.length">
              <h3>Occupancy Trend</h3>
              <div class="occ-strip">
                <div v-for="q in budgetReview.occupancy_trend" :key="q.quarter" class="occ-col">
                  <div class="occ-bar-wrap"><div class="occ-bar" :style="{ height: q.occupancy + '%' }"></div></div>
                  <div class="occ-val">{{ q.occupancy.toFixed(0) }}%</div>
                  <div class="occ-label">{{ q.quarter }}</div>
                </div>
              </div>
            </div>

            <div class="panel">
              <h3>Analyst Commentary</h3>
              <textarea v-model="comments.budget_review" rows="5" :disabled="!commentsEditable"
                        placeholder="Variance drivers: lease assumptions, credit loss, expense differences vs budget..."></textarea>
              <div class="form-actions no-print" v-if="commentsEditable">
                <button class="btn-secondary" @click="saveComment('budget_review')" :disabled="commentSaving.budget_review">
                  {{ commentSaving.budget_review ? 'Saving...' : 'Save Commentary' }}
                </button>
              </div>
            </div>
          </template>
        </div>

        <!-- ===== Tab: Balance Sheet ===== -->
        <div v-show="activeTab === 'balance'" class="tab-panel">
          <div v-if="balanceLoading" class="loading-text">Loading balance sheet...</div>
          <template v-if="balanceSheet">
            <div class="panel">
              <h3>Balance Sheet Analysis</h3>
              <p class="panel-note">
                {{ fmtDate(balanceSheet.prior_date) }} (prior year end) vs {{ fmtDate(balanceSheet.current_date) }} (latest reported).
              </p>
              <p class="panel-note warn-note" v-if="balanceSheet.current_date && balanceSheet.current_is_as_of === false">
                The {{ fmtDate(balanceSheet.as_of_date) }} balance sheet is not yet in ISBS —
                showing the most recent month available ({{ fmtDate(balanceSheet.current_date) }}).
              </p>
              <div class="table-scroll">
                <table class="data-table">
                  <thead>
                    <tr>
                      <th>Acct</th><th>Line Item</th>
                      <th class="num">{{ fmtDate(balanceSheet.prior_date) }}</th>
                      <th class="num">{{ fmtDate(balanceSheet.current_date) }}</th>
                      <th class="num">Variance</th>
                    </tr>
                  </thead>
                  <tbody>
                    <template v-for="section in ['Assets', 'Liabilities', 'Equity']" :key="section">
                      <tr class="row-section" v-if="balanceSheet.rows.some((r: any) => r.account_type === section)">
                        <td colspan="5">{{ section.toUpperCase() }}</td>
                      </tr>
                      <tr v-for="(r, i) in balanceSheet.rows.filter((r: any) => r.account_type === section)" :key="section + i">
                        <td class="mono">{{ r.account }}</td>
                        <td>{{ r.description }}</td>
                        <td class="num">{{ fmtCurrency(r.prior) }}</td>
                        <td class="num">{{ fmtCurrency(r.current) }}</td>
                        <td class="num" :class="{ neg: r.variance < 0 }">{{ fmtCurrency(r.variance) }}</td>
                      </tr>
                    </template>
                    <tr v-if="!balanceSheet.rows.length">
                      <td colspan="5" class="empty-row">No balance sheet data reported for this deal.</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <div class="panel">
              <h3>Balance Sheet Commentary</h3>
              <textarea v-model="comments.balance_sheet" rows="4" :disabled="!commentsEditable"
                        placeholder="AR/AP movements, reserve balances, anything the committee should see..."></textarea>
              <div class="form-actions no-print" v-if="commentsEditable">
                <button class="btn-secondary" @click="saveComment('balance_sheet')" :disabled="commentSaving.balance_sheet">
                  {{ commentSaving.balance_sheet ? 'Saving...' : 'Save Commentary' }}
                </button>
              </div>
            </div>
          </template>
        </div>

        <!-- ===== Tab: Q&A ===== -->
        <div v-show="activeTab === 'qa'" class="tab-panel">
          <div class="panel no-print" v-if="record.status !== 'approved'">
            <h3>Ask a Question</h3>
            <p class="panel-note">Reviewers and committee members can ask about the valuation or its
              assumptions; the asset manager answers here and the thread becomes part of the record.</p>
            <textarea v-model="newQuestion" rows="2"
                      placeholder="e.g. What supports the terminal cap tightening vs the survey midpoint?"></textarea>
            <div class="form-actions">
              <button class="btn-primary" @click="askQuestion" :disabled="qaBusy || !newQuestion.trim()">
                {{ qaBusy ? 'Posting...' : 'Post Question' }}
              </button>
            </div>
          </div>

          <div class="panel" v-for="q in (record.questions || [])" :key="q.id">
            <div class="qa-head">
              <span class="status-badge" :class="{
                'status-open': q.status === 'open',
                'status-signed': q.status === 'answered',
                'status-approved': q.status === 'resolved',
              }">{{ q.status }}</span>
              <span class="qa-meta">{{ q.asked_by }} &middot; {{ fmtDateTime(q.asked_at) }}</span>
            </div>
            <div class="qa-question">{{ q.question_text }}</div>
            <div v-if="q.answer_text" class="qa-answer">
              <div class="qa-meta">Answered by {{ q.answered_by }} &middot; {{ fmtDateTime(q.answered_at) }}</div>
              {{ q.answer_text }}
            </div>
            <div v-if="canEdit && q.status !== 'resolved'" class="qa-reply no-print">
              <textarea v-model="answerDrafts[q.id]" rows="2"
                        :placeholder="q.answer_text ? 'Revise the answer...' : 'Provide an answer...'"></textarea>
              <div class="form-actions">
                <button class="btn-secondary" @click="answerQuestion(q)" :disabled="qaBusy || !(answerDrafts[q.id] || '').trim()">
                  {{ q.answer_text ? 'Update Answer' : 'Post Answer' }}
                </button>
                <button v-if="q.status === 'answered'" class="btn-secondary" @click="resolveQuestion(q)">
                  Mark Resolved
                </button>
              </div>
            </div>
            <div v-else-if="(perms.can_approve || perms.is_recorder) && q.status === 'answered'" class="form-actions no-print">
              <button class="btn-secondary" @click="resolveQuestion(q)">Mark Resolved</button>
            </div>
          </div>
          <div v-if="!(record.questions || []).length" class="empty-note" style="padding: 8px 2px">
            No questions yet.
          </div>
        </div>

        <!-- ===== Tab: AI Summary ===== -->
        <div v-show="activeTab === 'ai'" class="tab-panel">
          <div v-if="aiLoading" class="loading-text">Checking for a stored summary...</div>
          <div class="panel no-print" v-if="aiSummary && !aiSummary.exists">
            <h3>AI Appraisal Summary</h3>
            <p class="panel-note">Condenses the uploaded appraisal PDF to a few key pages: how the
              appraiser approached the valuation, the assumptions that drive the value, market context,
              and risks — then cross-checks the extracted assumptions against what was entered.</p>
            <button class="btn-primary" @click="generateAiSummary" :disabled="aiGenerating || !canEdit">
              {{ aiGenerating ? 'Reading the appraisal (about a minute)...' : 'Generate Summary' }}
            </button>
          </div>

          <template v-if="aiSummary?.exists">
            <div class="panel ai-meta-panel no-print">
              <span class="panel-note">
                Generated {{ fmtDateTime(aiSummary.created_at) }} by {{ aiSummary.created_by }}
                from {{ aiSummary.summary?._meta?.source_document }}
                ({{ aiSummary.summary?._meta?.page_count }} pages, {{ aiSummary.model }}).
                AI-generated — verify against the report before relying on it.
              </span>
              <button class="btn-secondary" @click="generateAiSummary" :disabled="aiGenerating || !canEdit">
                {{ aiGenerating ? 'Regenerating...' : 'Regenerate' }}
              </button>
            </div>

            <div class="panel">
              <h3>Executive Summary</h3>
              <p class="ai-text">{{ aiSummary.summary.executive_summary }}</p>
              <div class="ai-facts" v-if="aiSummary.summary.value_conclusion">
                <div class="fact"><span class="strip-label">As-Is Value</span>{{ fmtCurrency(aiSummary.summary.value_conclusion.as_is_value) }}</div>
                <div class="fact"><span class="strip-label">Per SF</span>{{ fmtCurrency(aiSummary.summary.value_conclusion.per_sf) }}</div>
                <div class="fact"><span class="strip-label">Prior Value</span>{{ fmtCurrency(aiSummary.summary.value_conclusion.prior_value) }}</div>
                <div class="fact"><span class="strip-label">Change</span>
                  <span :class="{ pos: (aiSummary.summary.value_conclusion.change_amount ?? 0) > 0, neg: (aiSummary.summary.value_conclusion.change_amount ?? 0) < 0 }">
                    {{ fmtCurrency(aiSummary.summary.value_conclusion.change_amount) }}
                    ({{ fmtPct(aiSummary.summary.value_conclusion.change_pct) }})
                  </span>
                </div>
                <div class="fact"><span class="strip-label">Interest</span>{{ aiSummary.summary.value_conclusion.interest_appraised || '—' }}</div>
                <div class="fact"><span class="strip-label">Occupancy</span>{{ fmtOcc(aiSummary.summary.property?.occupancy_pct) }}</div>
              </div>
            </div>

            <div class="panel" v-if="aiSummary.checks?.length">
              <div class="ai-meta-panel">
                <h3 style="margin:0">Assumption Cross-Check <span class="panel-note" style="font-weight:400">entered on the record vs extracted from the appraisal</span></h3>
                <button v-if="recordEditable" class="btn-secondary no-print" @click="applyAllExtracted" :disabled="applying">
                  {{ applying ? 'Applying...' : 'Apply All Extracted Values' }}
                </button>
              </div>
              <div class="table-scroll">
                <table class="data-table">
                  <thead><tr><th>Field</th><th class="num">Entered</th><th class="num">From Appraisal</th><th>Match</th><th class="no-print"></th></tr></thead>
                  <tbody>
                    <tr v-for="c in aiSummary.checks" :key="c.field">
                      <td>{{ c.field }}</td>
                      <td class="num">{{ fmtCheckValue(c.field, c.entered) }}</td>
                      <td class="num">{{ fmtCheckValue(c.field, c.extracted) }}</td>
                      <td>
                        <span v-if="c.match === true" class="mini-badge argus">match</span>
                        <span v-else-if="c.match === false" class="mini-badge mismatch">differs</span>
                        <span v-else class="mini-badge">n/a</span>
                      </td>
                      <td class="no-print">
                        <button v-if="recordEditable && c.extracted != null && c.match !== true"
                                class="btn-link" @click="applyExtracted(c)" :disabled="applying">
                          apply
                        </button>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <p class="panel-note" v-if="!recordEditable" style="margin-top:8px">
                Values can be applied while the record is open (before analyst sign-off).
              </p>
            </div>

            <div class="panel">
              <h3>Valuation Approach</h3>
              <p class="ai-text">{{ aiSummary.summary.valuation_approach }}</p>
            </div>

            <div class="panel" v-if="aiSummary.summary.key_assumptions">
              <h3>Key Assumptions</h3>
              <div class="ai-facts">
                <div class="fact"><span class="strip-label">Going-in Cap</span>{{ fmtPct(aiSummary.summary.key_assumptions.overall_cap_rate) }}</div>
                <div class="fact"><span class="strip-label">Terminal Cap</span>{{ fmtPct(aiSummary.summary.key_assumptions.terminal_cap_rate) }}</div>
                <div class="fact"><span class="strip-label">Discount Rate</span>{{ fmtPct(aiSummary.summary.key_assumptions.discount_rate) }}</div>
                <div class="fact"><span class="strip-label">Rent Growth</span>{{ fmtPct(aiSummary.summary.key_assumptions.market_rent_growth) }}</div>
                <div class="fact"><span class="strip-label">Expense Growth</span>{{ fmtPct(aiSummary.summary.key_assumptions.expense_growth) }}</div>
                <div class="fact"><span class="strip-label">Selling Costs</span>{{ fmtPct(aiSummary.summary.key_assumptions.selling_costs_at_reversion) }}</div>
                <div class="fact"><span class="strip-label">Vacancy / Credit</span>{{ fmtPct(aiSummary.summary.key_assumptions.vacancy_credit_loss) }}</div>
                <div class="fact"><span class="strip-label">DCF Hold</span>{{ aiSummary.summary.key_assumptions.dcf_hold_period_years != null ? aiSummary.summary.key_assumptions.dcf_hold_period_years + ' yrs' : '—' }}</div>
              </div>
              <p class="ai-text" v-if="aiSummary.summary.key_assumptions.notes" style="margin-top:10px">
                {{ aiSummary.summary.key_assumptions.notes }}
              </p>
            </div>

            <div class="panel-grid">
              <div class="panel" v-if="aiSummary.summary.market_overview?.length">
                <h3>Market Overview</h3>
                <ul class="ai-list"><li v-for="(b, i) in aiSummary.summary.market_overview" :key="i">{{ b }}</li></ul>
              </div>
              <div class="panel" v-if="aiSummary.summary.rent_and_leasing?.length">
                <h3>Rent &amp; Leasing</h3>
                <ul class="ai-list"><li v-for="(b, i) in aiSummary.summary.rent_and_leasing" :key="i">{{ b }}</li></ul>
              </div>
              <div class="panel" v-if="aiSummary.summary.positives?.length">
                <h3>Positives</h3>
                <ul class="ai-list"><li v-for="(b, i) in aiSummary.summary.positives" :key="i">{{ b }}</li></ul>
              </div>
              <div class="panel" v-if="aiSummary.summary.risks?.length">
                <h3>Risks</h3>
                <ul class="ai-list"><li v-for="(b, i) in aiSummary.summary.risks" :key="i">{{ b }}</li></ul>
              </div>
            </div>

            <div class="panel" v-if="aiSummary.summary.extraordinary_assumptions?.length">
              <h3>Extraordinary Assumptions &amp; Hypothetical Conditions</h3>
              <ul class="ai-list"><li v-for="(b, i) in aiSummary.summary.extraordinary_assumptions" :key="i">{{ b }}</li></ul>
            </div>
          </template>
        </div>

        <!-- ===== Tab: NAV ===== -->
        <div v-show="activeTab === 'nav'" class="tab-panel">
          <div v-if="navLoading" class="loading-text">Loading NAV inputs...</div>
          <template v-if="navData">
            <div class="panel">
              <div class="ai-meta-panel no-print">
                <span class="panel-note" v-if="navData.result">
                  Computed {{ fmtDateTime(navData.result.computed_at) }} by {{ navData.result.computed_by }}
                  <span v-if="navSelectionsDirty" class="warn-note"> — balance sheet selections changed, recompute to apply</span>
                </span>
                <span class="panel-note" v-else>No NAV computed yet for this record.</span>
                <span style="display:flex; gap:8px">
                  <button class="btn-primary" @click="computeNav" :disabled="navComputing || !canEdit">
                    {{ navComputing ? 'Computing...' : (navData.result ? 'Recompute NAV' : 'Compute NAV') }}
                  </button>
                  <button v-if="navData.result" class="btn-secondary" @click="downloadNavPackage">
                    Download Auditor Package
                  </button>
                </span>
              </div>

              <template v-if="navData.result">
                <div class="ai-facts" style="margin-top:14px">
                  <div class="fact"><span class="strip-label">Value ({{ navData.result.value_source === 'cost_derived' ? 'cost derived' : navData.result.value_source === 'children_rollup' ? 'children rollup' : 'entered' }})</span>{{ fmtCurrency(navData.result.value) }}</div>
                  <div class="fact"><span class="strip-label">Less Debt</span>({{ fmtCurrency(navData.result.debt) }})</div>
                  <div class="fact"><span class="strip-label">Current Assets</span>{{ fmtCurrency(navData.result.current_assets) }}</div>
                  <div class="fact"><span class="strip-label">Current Liabilities</span>({{ fmtCurrency(navData.result.current_liabilities) }})</div>
                  <div class="fact"><span class="strip-label">Net Proceeds</span><strong>{{ fmtCurrency(navData.result.net_proceeds) }}</strong></div>
                  <div class="fact"><span class="strip-label">PSC NAV</span><strong class="pos">{{ fmtCurrency(navData.result.psc_nav) }}</strong></div>
                  <div class="fact"><span class="strip-label">OP NAV</span>{{ fmtCurrency(navData.result.op_nav) }}</div>
                </div>

                <h3 style="margin-top:18px">Liquidation Waterfall</h3>
                <div class="table-scroll">
                  <table class="data-table">
                    <thead>
                      <tr><th>Ref</th><th>Recipient</th><th>Step</th><th class="num">Rate</th>
                          <th class="num">Amount</th><th class="num">Remaining</th></tr>
                    </thead>
                    <tbody>
                      <tr v-for="(l, i) in navData.result.walk" :key="i">
                        <td>
                          <input v-if="canEdit" class="ref-input" v-model="refDrafts[l.iorder]"
                                 placeholder="8.2(a)" @change="saveStepRef(l)" />
                          <span v-else>{{ l.agreement_ref }}</span>
                        </td>
                        <td>{{ l.recipient }}</td>
                        <td>{{ l.step }} <span class="qa-meta">{{ l.label }}</span></td>
                        <td class="num">{{ l.rate != null && (l.step === 'Pref' || l.step === 'IRR') ? fmtPct(l.rate) : '' }}</td>
                        <td class="num">{{ fmtCurrency(l.allocated) }}</td>
                        <td class="num">{{ fmtCurrency(l.remaining_after) }}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
                <ul class="ai-list" v-if="navData.result.notes?.length">
                  <li v-for="(n, i) in navData.result.notes" :key="i" class="qa-meta">{{ n }}</li>
                </ul>
                <div class="ai-facts" v-if="navData.result.pref" style="margin-top:8px">
                  <div class="fact" v-for="(p, pc) in navData.result.pref" :key="pc">
                    <span class="strip-label">{{ pc }} accrued pref @ {{ fmtPct(p.pref_rate) }}</span>
                    {{ fmtCurrency(p.accrued_pref) }}
                  </div>
                </div>
              </template>
            </div>

            <div class="panel">
              <h3>Balance Sheet Adjustment — Current Assets &amp; Liabilities</h3>
              <p class="panel-note">
                The app suggests the inclusion set; adjust per your knowledge of the property's books.
                Selections carry forward to next year's cycle
                <span v-if="navData.inputs?.has_prior_selections">(carried forward from the prior cycle)</span>.
                Snapshot: <span v-for="(d, v) in navData.inputs?.snapshot_dates" :key="v">{{ v }} {{ fmtDate(d) }} </span>
              </p>
              <div class="table-scroll">
                <table class="data-table">
                  <thead>
                    <tr><th class="no-print">Include</th><th>Acct</th><th>Line Item</th>
                        <th class="num">Amount</th><th>Note</th></tr>
                  </thead>
                  <tbody>
                    <template v-for="section in ['Assets', 'Liabilities']" :key="section">
                      <tr class="row-section"><td colspan="5">{{ section.toUpperCase() }}</td></tr>
                      <tr v-for="(l, i) in (navData.inputs?.lines || []).filter((x: any) => x.account_type === section)"
                          :key="section + i" :class="{ 'line-excluded': !l.included && l.selectable }">
                        <td class="no-print">
                          <input type="checkbox" :checked="l.included" :disabled="!l.selectable || !canEdit"
                                 @change="toggleBsLine(l)" />
                        </td>
                        <td class="mono">{{ l.account }}</td>
                        <td>{{ l.description }}</td>
                        <td class="num">{{ fmtCurrency(l.amount) }}</td>
                        <td>
                          <span v-if="l.is_debt" class="mini-badge">debt</span>
                          <span v-else-if="l.changed_vs_prior" class="mini-badge mismatch" title="Treatment differs from the prior cycle">changed</span>
                        </td>
                      </tr>
                    </template>
                  </tbody>
                </table>
              </div>
              <div class="form-actions no-print" v-if="canEdit">
                <button class="btn-primary" @click="computeNav" :disabled="navComputing">
                  {{ navComputing ? 'Computing...' : 'Save Selections & Recompute' }}
                </button>
              </div>
            </div>
          </template>
        </div>
      </template>
    </template>
  </div>
</template>

<style scoped>
.valuations { padding: 0 0 40px 0; }
h2 { font-size: 18px; margin: 0; display: inline-flex; align-items: center; gap: 10px; }
h3 { font-size: 14px; margin: 0 0 10px; }
.mt { margin-top: 18px; }

.header-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; flex-wrap: wrap; gap: 10px; }
.header-left { display: flex; align-items: center; gap: 12px; }
.header-controls { display: flex; align-items: center; gap: 8px; }
.cycle-select { padding: 6px 10px; border: 1px solid var(--color-border); border-radius: 6px; background: var(--color-surface); font-size: 13px; }

.btn-primary, .btn-secondary, .btn-back {
  padding: 6px 14px; border-radius: 6px; font-size: 13px; cursor: pointer;
  border: 1px solid var(--color-border); background: var(--color-surface); color: var(--color-text);
}
.btn-primary { background: var(--color-accent); border-color: var(--color-accent); color: #fff; }
.btn-primary:disabled, .btn-secondary:disabled { opacity: 0.6; cursor: default; }
.btn-back { font-weight: 600; }
.btn-upload-label { display: inline-block; }
.btn-link { background: none; border: none; cursor: pointer; font-size: 12px; color: var(--color-accent); padding: 0; }
.btn-link.danger { color: #b3402f; }

.error-banner { background: #fdecea; color: #b3402f; border: 1px solid #f5c6c0; border-radius: 6px; padding: 10px 14px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; }
.error-banner button { border: none; background: none; color: inherit; cursor: pointer; font-weight: 600; }
.save-banner { background: #e8f5e9; color: #2e7d32; border: 1px solid #c8e6c9; border-radius: 6px; padding: 8px 14px; margin-bottom: 12px; }
.loading-text { color: var(--color-text-secondary); padding: 20px 0; }
.empty-state { color: var(--color-text-secondary); padding: 30px 0; }
.empty-row { text-align: center; color: var(--color-text-secondary); padding: 18px; }
.empty-note { color: var(--color-text-secondary); font-size: 13px; }

/* summary cards */
.summary-cards { display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; align-items: center; }
.summary-card {
  background: var(--color-surface); border: 1px solid var(--color-border); border-radius: 8px;
  padding: 10px 20px; text-align: center; cursor: pointer; transition: all 0.15s; min-width: 90px;
}
.summary-card:hover { border-color: var(--color-accent); }
.summary-card.active { border-color: var(--color-accent); background: #e3f2fd; }
.card-count { font-size: 22px; font-weight: 700; display: block; }
.card-label { font-size: 11px; color: var(--color-text-secondary); text-transform: uppercase; }
.card-signed .card-count { color: #2e7d32; }
.card-third .card-count { color: #1565c0; }
.card-cost .card-count { color: #e65100; }
.search-input { margin-left: auto; padding: 7px 12px; border: 1px solid var(--color-border); border-radius: 6px; font-size: 13px; min-width: 200px; }

/* records table */
.records-table-wrap { border: 1px solid var(--color-border); border-radius: 8px; overflow: hidden; background: var(--color-surface); overflow-x: auto; }
.records-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.records-table th {
  text-align: left; padding: 10px 12px; background: var(--color-bg);
  border-bottom: 2px solid var(--color-border); font-size: 11px; text-transform: uppercase;
  color: var(--color-text-secondary); white-space: nowrap;
}
.records-table td { padding: 8px 12px; border-bottom: 1px solid var(--color-border); }
.records-table th.num, .records-table td.num { text-align: right; font-variant-numeric: tabular-nums; }
.clickable-row { cursor: pointer; }
.clickable-row:hover { background: var(--color-bg); }
.child-row td.deal-name { padding-left: 26px; color: var(--color-text-secondary); }
.child-indent { margin-right: 4px; }
.deal-name { font-weight: 600; }
.vcode-tag { font-size: 11px; font-weight: 400; color: var(--color-text-secondary); margin-left: 6px; }
.pos { color: #2e7d32; }
.neg { color: #b3402f; }

/* badges */
.badge, .status-badge, .mini-badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; }
.badge-third { background: #e3f2fd; color: #1565c0; }
.badge-internal { background: #ede7f6; color: #5e35b1; }
.badge-cost { background: #fff3e0; color: #e65100; }
.status-open { background: #eeeeee; color: #666; }
.status-signed { background: #e8f5e9; color: #2e7d32; }
.status-excluded { background: #fdecea; color: #b3402f; }
.mini-badge { background: #eeeeee; color: #555; padding: 1px 7px; }
.mini-badge.argus { background: #e8f5e9; color: #2e7d32; }
.doc-count { font-size: 11px; color: var(--color-text-secondary); margin-left: 4px; }

/* workspace */
.deal-strip { display: flex; flex-wrap: wrap; gap: 18px; background: var(--color-surface); border: 1px solid var(--color-border); border-radius: 8px; padding: 12px 16px; margin-bottom: 4px; font-size: 13px; }
.strip-label { display: block; font-size: 10px; text-transform: uppercase; color: var(--color-text-secondary); margin-bottom: 2px; }
.class-reason { font-size: 12px; color: var(--color-text-secondary); margin: 6px 2px 14px; }

.tabs { display: flex; gap: 4px; border-bottom: 2px solid var(--color-border); margin-bottom: 16px; }
.tabs button {
  padding: 8px 16px; border: none; background: none; cursor: pointer; font-size: 13px;
  color: var(--color-text-secondary); border-bottom: 2px solid transparent; margin-bottom: -2px;
}
.tabs button.active { color: var(--color-accent); border-bottom-color: var(--color-accent); font-weight: 600; }

.tab-panel { display: flex; flex-direction: column; gap: 16px; }
.panel { background: var(--color-surface); border: 1px solid var(--color-border); border-radius: 8px; padding: 16px; }
.panel-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 1000px) { .panel-grid { grid-template-columns: 1fr; } }
.panel-note { font-size: 12px; color: var(--color-text-secondary); margin: 0 0 12px; }
.warn-note { color: #e65100; }

.form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px 14px; }
.form-grid label { display: flex; flex-direction: column; font-size: 12px; color: var(--color-text-secondary); gap: 3px; }
.form-grid input, .form-grid select {
  padding: 6px 8px; border: 1px solid var(--color-border); border-radius: 5px;
  font-size: 13px; background: var(--color-surface); color: var(--color-text);
}
.form-grid .span-2 { grid-column: span 2; }
.form-actions { margin-top: 12px; display: flex; gap: 8px; }

.doc-controls { display: flex; gap: 8px; margin-bottom: 10px; align-items: center; }
.doc-controls select { padding: 6px 8px; border: 1px solid var(--color-border); border-radius: 5px; font-size: 13px; }
.doc-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.doc-table th { text-align: left; font-size: 11px; text-transform: uppercase; color: var(--color-text-secondary); padding: 6px 8px; border-bottom: 1px solid var(--color-border); }
.doc-table td { padding: 6px 8px; border-bottom: 1px solid var(--color-border); }
.doc-file { color: var(--color-accent); cursor: pointer; }
.doc-file:hover { text-decoration: underline; }
.argus-linked { font-size: 13px; margin-bottom: 8px; }

.table-scroll { overflow-x: auto; }
.data-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.data-table th { text-align: left; padding: 8px 10px; border-bottom: 2px solid var(--color-border); font-size: 11px; text-transform: uppercase; color: var(--color-text-secondary); white-space: nowrap; }
.data-table td { padding: 6px 10px; border-bottom: 1px solid var(--color-border); }
.data-table th.num, .data-table td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.data-table .indent { padding-left: 24px; }
.row-total td { font-weight: 700; border-top: 2px solid var(--color-text); }
.row-calc td { font-weight: 700; }
.row-section td { font-weight: 700; background: var(--color-bg); font-size: 11px; letter-spacing: 0.05em; }
.mono { font-family: monospace; font-size: 12px; }

textarea { width: 100%; padding: 8px 10px; border: 1px solid var(--color-border); border-radius: 6px; font-size: 13px; font-family: inherit; resize: vertical; box-sizing: border-box; background: var(--color-surface); color: var(--color-text); }

/* occupancy strip */
.occ-strip { display: flex; gap: 8px; align-items: flex-end; padding: 6px 2px; overflow-x: auto; }
.occ-col { text-align: center; min-width: 52px; }
.occ-bar-wrap { height: 90px; display: flex; align-items: flex-end; justify-content: center; }
.occ-bar { width: 26px; background: var(--color-accent); border-radius: 3px 3px 0 0; min-height: 2px; }
.occ-val { font-size: 11px; font-weight: 600; margin-top: 2px; }
.occ-label { font-size: 10px; color: var(--color-text-secondary); white-space: nowrap; }

/* phase 2 */
.view-toggle { display: inline-flex; border: 1px solid var(--color-border); border-radius: 6px; overflow: hidden; }
.view-toggle button { padding: 6px 12px; border: none; background: var(--color-surface); font-size: 13px; cursor: pointer; color: var(--color-text-secondary); }
.view-toggle button.active { background: var(--color-accent); color: #fff; }
.committee-actions { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; flex-wrap: wrap; }
.perm-note { font-size: 12px; color: var(--color-text-secondary); }
.status-approved { background: #e8f5e9; color: #1b5e20; border: 1px solid #a5d6a7; }
.qbadge { background: #fff3e0; color: #e65100; }
.mini-badge.mismatch { background: #fdecea; color: #b3402f; }
.appr-count { font-size: 12px; font-weight: 600; color: var(--color-text-secondary); }
.committee-chips { display: inline-flex; gap: 6px; }
.role-chip {
  font-size: 11px; font-weight: 600; padding: 2px 9px; border-radius: 12px;
  background: #eeeeee; color: #777;
}
.role-chip.done { background: #e8f5e9; color: #2e7d32; }
.dir-up { color: #2e7d32; }
.dir-down { color: #b3402f; }
.qa-head { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
.qa-meta { font-size: 11.5px; color: var(--color-text-secondary); }
.qa-question { font-weight: 600; font-size: 13.5px; margin-bottom: 8px; white-space: pre-wrap; }
.qa-answer {
  background: var(--color-bg); border-left: 3px solid var(--color-accent);
  border-radius: 0 6px 6px 0; padding: 8px 12px; font-size: 13.5px;
  margin-bottom: 8px; white-space: pre-wrap;
}
.qa-answer .qa-meta { margin-bottom: 4px; }
.qa-reply textarea { margin-bottom: 0; }
.ai-meta-panel { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
.ai-text { font-size: 13.5px; line-height: 1.55; margin: 4px 0; max-width: 90ch; white-space: pre-wrap; }
.ai-facts { display: flex; flex-wrap: wrap; gap: 16px 24px; margin-top: 10px; font-size: 13.5px; }
.ai-facts .fact { min-width: 110px; }
.ai-list { margin: 4px 0 0; padding-left: 18px; font-size: 13.5px; line-height: 1.5; }
.ai-list li { margin: 4px 0; }

/* phase 3 — NAV */
.ref-input {
  width: 72px; padding: 3px 6px; border: 1px solid var(--color-border);
  border-radius: 4px; font-size: 12px; font-family: monospace;
  background: var(--color-surface); color: var(--color-text);
}
.line-excluded td { color: var(--color-text-secondary); }
.line-excluded td.num { text-decoration: line-through; }

/* phase 4 — checks */
.checks-strip {
  background: var(--color-surface); border: 1px solid var(--color-border);
  border-radius: 8px; padding: 8px 14px; margin-bottom: 14px;
}
.checks-toggle {
  border: none; background: none; cursor: pointer; font-size: 13px;
  display: flex; align-items: center; gap: 8px; padding: 0; color: var(--color-text);
}
.checks-list { margin-top: 10px; display: flex; flex-direction: column; gap: 6px; }
.check-row { display: flex; gap: 10px; align-items: baseline; font-size: 13px; }
.check-row .mini-badge { flex: none; min-width: 40px; text-align: center; }

/* print */
@media print {
  * { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  .no-print { display: none !important; }
  .tab-panel { display: flex !important; }
  .tabs { display: none; }
  .panel { border: none; padding: 8px 0; }
  textarea { border: none; padding: 0; }
}
</style>
