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
const activeTab = ref<'assumptions' | 'budget' | 'balance'>('assumptions')
const saving = ref(false)
const saveMsg = ref('')

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
  activeTab.value = 'assumptions'
  recordLoading.value = true
  try {
    const res = await api.get(`/api/valuations/records/${id}`)
    record.value = res.data
    for (const f of FORM_FIELDS) form.value[f] = res.data[f]
    comments.value = {
      budget_review: res.data.comments?.budget_review?.comment_text || '',
      balance_sheet: res.data.comments?.balance_sheet?.comment_text || '',
      general: res.data.comments?.general?.comment_text || '',
    }
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

watch(activeTab, (tab) => {
  if (tab === 'budget') loadBudgetReview()
  if (tab === 'balance') loadBalanceSheet()
})

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
  open: 'Open', signed_off: 'Signed Off', excluded: 'Excluded',
}
function classBadge(c: string) {
  return { third_party: 'badge-third', internal: 'badge-internal', cost: 'badge-cost' }[c] || 'badge-internal'
}
function statusBadge(s: string) {
  return { open: 'status-open', signed_off: 'status-signed', excluded: 'status-excluded' }[s] || 'status-open'
}

onMounted(async () => {
  await loadCycles()
  await loadDashboard()
  const rec = Number(route.query.record)
  if (rec) openRecord(rec)
})
watch(selectedCycleId, () => {
  router.replace({ query: { ...route.query, cycle: String(selectedCycleId.value) } })
  loadDashboard()
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
                <td><span class="status-badge" :class="statusBadge(r.status)">{{ STATUS_LABELS[r.status] || r.status }}</span></td>
              </tr>
              <tr v-if="!filteredRecords.length">
                <td colspan="9" class="empty-row">No records match.</td>
              </tr>
            </tbody>
          </table>
        </div>
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
          <span class="status-badge" :class="statusBadge(record.status)">{{ STATUS_LABELS[record.status] || record.status }}</span>
          <button v-if="record.status === 'open' && canEdit" class="btn-primary" @click="recordAction('sign_off')">
            Analyst Sign-off
          </button>
          <button v-else-if="record.status === 'signed_off' && canEdit" class="btn-secondary" @click="recordAction('reopen')">
            Reopen
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

        <div class="tabs no-print">
          <button :class="{ active: activeTab === 'assumptions' }" @click="activeTab = 'assumptions'">Assumptions &amp; Documents</button>
          <button :class="{ active: activeTab === 'budget' }" @click="activeTab = 'budget'">Budget Review</button>
          <button :class="{ active: activeTab === 'balance' }" @click="activeTab = 'balance'">Balance Sheet</button>
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
                  <input type="number" step="any" v-model.number="form.concluded_value" :disabled="!recordEditable" />
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
                  <input type="number" step="any" v-model.number="form.direct_cap_noi" :disabled="!recordEditable" />
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
            <textarea v-model="comments.general" rows="3" :disabled="!recordEditable"
                      placeholder="Notes for the committee record..."></textarea>
            <div class="form-actions no-print" v-if="recordEditable">
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
              <textarea v-model="comments.budget_review" rows="5" :disabled="!recordEditable"
                        placeholder="Variance drivers: lease assumptions, credit loss, expense differences vs budget..."></textarea>
              <div class="form-actions no-print" v-if="recordEditable">
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
              <textarea v-model="comments.balance_sheet" rows="4" :disabled="!recordEditable"
                        placeholder="AR/AP movements, reserve balances, anything the committee should see..."></textarea>
              <div class="form-actions no-print" v-if="recordEditable">
                <button class="btn-secondary" @click="saveComment('balance_sheet')" :disabled="commentSaving.balance_sheet">
                  {{ commentSaving.balance_sheet ? 'Saving...' : 'Save Commentary' }}
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
