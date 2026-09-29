<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import api from '../api/client'
import { useAuthStore } from '../stores/auth'

const router = useRouter()
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { BarChart, PieChart, LineChart } from 'echarts/charts'
import {
  GridComponent, TooltipComponent, LegendComponent,
  MarkLineComponent,
} from 'echarts/components'

use([CanvasRenderer, BarChart, PieChart, LineChart, GridComponent, TooltipComponent, LegendComponent, MarkLineComponent])

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
const documents = ref<Record<number, any[]>>({})

// Edit state for field resolution
const editingTenantId = ref<number | null>(null)
const editingField = ref<string | null>(null)
const editValue = ref('')
const editSource = ref('analyst')

// Space planning state
const selectedTenantIds = ref<Set<number>>(new Set())
const showAddTenant = ref(false)
const newTenant = ref<any>({})
const showMergeModal = ref(false)
const showSplitModal = ref(false)
const mergeForm = ref({ merged_name: '', merged_suite: '' })
const splitForm = ref<{ splits: any[] }>({ splits: [{ tenant_name: '', suite: '', square_feet: 0 }, { tenant_name: '', suite: '', square_feet: 0 }] })
const splitSourceTenant = ref<any>(null)
const showSuccessionModal = ref(false)
const successionSourceId = ref<number | null>(null)
const successionForm = ref<any>({})
const successionChain = ref<any[]>([])
const showChainFor = ref<number | null>(null)
const showReplacedTenants = ref(false)
const spaceEvents = ref<any[]>([])
const overviewMode = ref<'current' | 'timeline'>('current')
const showPlanEventModal = ref(false)
const planEventForm = ref<any>({ event_type: 'vacate', effective_date: '', source_tenant_ids: [], results: [{ tenant_name: '', suite: '', square_feet: 0 }] })
const confirmDeleteId = ref<number | null>(null)

// ── Analyst review of each tenant's exclusives and co-tenancy ──
// Per tenant and section, kept apart from the clause rows (which a re-read
// rebuilds), so a note or a sign-off survives re-extraction. A re-read after
// sign-off is marked rather than silently kept.
const auth = useAuthStore()
const canReview = computed(() => auth.isAnalyst)
const clauseReviews = ref<Record<string, Record<string, any>>>({})
const reviewEdit = ref<{ tenantId: number, section: string } | null>(null)
const reviewForm = ref({ status: 'unreviewed', notes: '' })
const reviewSaving = ref(false)
const reviewError = ref<string | null>(null)
const showBoundBy = ref(true)
const exclusiveFilter = ref<'all' | 'with' | 'unreviewed' | 'flagged'>('all')

function reviewOf(tenantId: number, section: string) {
  return clauseReviews.value[String(tenantId)]?.[section] || null
}
function startReview(tenantId: number, section: string) {
  const r = reviewOf(tenantId, section)
  reviewEdit.value = { tenantId, section }
  reviewForm.value = { status: r?.status || 'unreviewed', notes: r?.notes || '' }
  reviewError.value = null
}
async function saveReview() {
  if (!reviewEdit.value) return
  reviewSaving.value = true
  reviewError.value = null
  const { tenantId, section } = reviewEdit.value
  try {
    const res = await api.put(
      `/api/lease-review/reviews/${selectedReviewId.value}/tenants/${tenantId}/clause-review`,
      { section, status: reviewForm.value.status, notes: reviewForm.value.notes })
    const key = String(tenantId)
    clauseReviews.value = {
      ...clauseReviews.value,
      [key]: { ...(clauseReviews.value[key] || {}), [section]: res.data },
    }
    reviewEdit.value = null
  } catch (e: any) {
    reviewError.value = e.response?.data?.error || e.message
  } finally {
    reviewSaving.value = false
  }
}

// Every active tenant, with its exclusive-use rows grouped under it -- so a
// tenant with NO rows can still be reviewed ("confirmed: no exclusive" is an
// answer), and a lease's disclosure of other tenants' exclusives sits visibly
// under "Bound by", not mixed in with what the tenant holds.
const exclusiveGroups = computed(() => {
  const byTenant: Record<string, any[]> = {}
  for (const e of exclusiveUse.value) (byTenant[String(e.tenant_id)] ||= []).push(e)
  const seen = new Set<string>()
  const groups: any[] = []
  for (const t of tenants.value) {
    const k = String(t.id)
    seen.add(k)
    const rows = byTenant[k] || []
    groups.push({
      tenant_id: t.id, tenant_name: t.tenant_name, suite: t.suite,
      holds: rows.filter(r => r.clause_role === 'holder'),
      bound: rows.filter(r => r.clause_role === 'subject'),
      other: rows.filter(r => r.clause_role !== 'holder' && r.clause_role !== 'subject'),
      review: reviewOf(t.id, 'exclusive_use'),
    })
  }
  // Rows for a tenant not in the active list (vacated / not a tenant) still show.
  for (const [k, rows] of Object.entries(byTenant)) {
    if (seen.has(k)) continue
    groups.push({
      tenant_id: Number(k), tenant_name: rows[0].tenant_name, suite: rows[0].suite,
      holds: rows.filter(r => r.clause_role === 'holder'),
      bound: rows.filter(r => r.clause_role === 'subject'),
      other: rows.filter(r => r.clause_role !== 'holder' && r.clause_role !== 'subject'),
      review: reviewOf(Number(k), 'exclusive_use'), inactive: true,
    })
  }
  return groups.filter(g => {
    const n = g.holds.length + g.bound.length + g.other.length
    if (exclusiveFilter.value === 'with') return n > 0
    if (exclusiveFilter.value === 'unreviewed') return !g.review || g.review.status === 'unreviewed'
    if (exclusiveFilter.value === 'flagged') return g.review?.status === 'flagged' || !!g.review?.reread_at
    return true
  })
})

// ── Rent Roll: the lease timeline in the IC exhibit's layout (steps 3-4) ──
// One engine (GET /rent-roll-timeline) -- the same timeline the exhibit prints.
// The analyst settles what the documents cannot settle: start and expiration
// through the field resolutions, the future schedule and the options as a whole,
// each with a reason and a cited document.
const rentRoll = ref<any>(null)
const rentRollLoading = ref(false)
const rentRollError = ref<string | null>(null)
const openFlags = ref<Set<number>>(new Set())
const settleFor = ref<any>(null)
const settleForm = ref<any>({})
const settleSaving = ref(false)
const settleMsg = ref<string | null>(null)

async function loadRentRoll() {
  if (!selectedReviewId.value) return
  rentRollLoading.value = true
  rentRollError.value = null
  try {
    const res = await api.get(`/api/lease-review/reviews/${selectedReviewId.value}/rent-roll-timeline`)
    rentRoll.value = res.data
  } catch (e: any) {
    rentRollError.value = e.response?.data?.error || e.message
  } finally {
    rentRollLoading.value = false
  }
}
watch(activeTab, (t) => { if (t === 'rentroll' && !rentRoll.value) loadRentRoll() })
watch(selectedReviewId, () => { rentRoll.value = null; settleFor.value = null
  if (activeTab.value === 'rentroll') loadRentRoll() })

function rrDate(v: any) {
  if (!v) return ''
  const [y, m, d] = String(v).slice(0, 10).split('-')
  return `${m}-${d}-${y.slice(2)}`
}
function rrMoney(v: any) {
  return v === null || v === undefined ? '' : Math.round(Number(v)).toLocaleString('en-US')
}
function rrPsf(v: any) {
  return v === null || v === undefined ? '' : Number(v).toFixed(2)
}
function toggleFlags(id: number) {
  const s = new Set(openFlags.value)
  s.has(id) ? s.delete(id) : s.add(id)
  openFlags.value = s
}
function tenantDocs(tid: number) {
  return (documents.value as any)[tid] || (documents.value as any)[String(tid)] || []
}

function openSettle(t: any) {
  settleMsg.value = null
  settleFor.value = t
  settleForm.value = {
    start: t.start || '', expiration: t.expiration || '',
    schedule: (t.future || []).map((p: any) => ({ start: p.start, end: p.end,
      annual_rent: p.annual_rent })),
    options: (t.options || []).flatMap((o: any, i: number) =>
      (o.periods || []).map((p: any) => ({ option: i + 1, start: p.start, end: p.end,
        annual_rent: p.annual_rent, term_years: o.term_years }))),
    reason: '', doc: (tenantDocs(t.tenant_id)[0] || {}).id || null,
  }
}
async function settleField(field: 'lease_start' | 'lease_end', value: string) {
  const t = settleFor.value
  await api.put(`/api/lease-review/reviews/${selectedReviewId.value}/validation/resolve`, {
    tenant_id: t.tenant_id, field, value, reason: settleForm.value.reason,
    source_doc_id: settleForm.value.doc,
    prior_value: field === 'lease_start' ? t.start : t.expiration,
  })
}
async function saveSettlement(what: 'dates' | 'schedule' | 'options') {
  const t = settleFor.value
  const f = settleForm.value
  settleSaving.value = true
  settleMsg.value = null
  try {
    if (!f.reason?.trim()) throw new Error('A reason is required.')
    if (!f.doc) throw new Error('Cite the document the figures come from.')
    if (what === 'dates') {
      if (f.start && f.start !== t.start) await settleField('lease_start', f.start)
      if (f.expiration && f.expiration !== t.expiration) await settleField('lease_end', f.expiration)
    } else {
      let rows: any[]
      if (what === 'schedule') {
        rows = f.schedule.filter((r: any) => r.start || r.end)
      } else {
        const byOpt: Record<string, any> = {}
        for (const r of f.options.filter((r: any) => r.start || r.end)) {
          const k = String(r.option || 1)
          byOpt[k] ||= { term_years: r.term_years || null, periods: [] }
          byOpt[k].periods.push({ start: r.start, end: r.end,
            annual_rent: r.annual_rent === '' ? null : r.annual_rent })
        }
        rows = Object.keys(byOpt).sort((a, b) => Number(a) - Number(b)).map(k => byOpt[k])
      }
      const res = await api.put(
        `/api/lease-review/reviews/${selectedReviewId.value}/tenants/${t.tenant_id}/timeline/settlement`,
        { section: what, rows, reason: f.reason, source_doc_id: f.doc })
      if (res.data.warnings?.length) settleMsg.value = res.data.warnings.join(' ')
    }
    await loadRentRoll()
    settleFor.value = (rentRoll.value?.tenants || []).find((x: any) => x.tenant_id === t.tenant_id) || null
    if (!settleMsg.value) settleMsg.value = 'Saved.'
  } catch (e: any) {
    settleMsg.value = e.response?.data?.error || e.message
  } finally {
    settleSaving.value = false
  }
}
async function clearSettlement(section: 'schedule' | 'options') {
  const t = settleFor.value
  await api.delete(
    `/api/lease-review/reviews/${selectedReviewId.value}/tenants/${t.tenant_id}/timeline/settlement`,
    { params: { section } })
  await loadRentRoll()
  settleFor.value = (rentRoll.value?.tenants || []).find((x: any) => x.tenant_id === t.tenant_id) || null
}

function radiusText(v: any) {
  return v === null || v === undefined || v === '' ? '' : `${Number(v).toLocaleString()} ft`
}

// Alias state
const aliases = ref<any[]>([])
const aliasSuggestions = ref<any[]>([])
const showAliasModal = ref(false)
const aliasForm = ref({ alias_name: '', canonical_name: '' })
const aliasLoading = ref(false)

// Projections state
const marketAssumptions = ref<any[]>([])
const projectionData = ref<any>(null)
const projectionScenario = ref<'weighted' | 'renewal' | 'new_tenant'>('weighted')
const projectionLoading = ref(false)
const projStartDate = ref('')
const projEndDate = ref('')
const revenueSummary = ref<any>(null)

const TABS = [
  { key: 'overview', label: 'Overview' },
  { key: 'expirations', label: 'Lease Expirations' },
  { key: 'validation', label: 'Validation' },
  { key: 'cotenancy', label: 'Co-Tenancy Risk' },
  { key: 'scenarios', label: 'Scenario Analysis' },
  { key: 'exclusive', label: 'Exclusive Use' },
  { key: 'rentroll', label: 'Rent Roll' },
  { key: 'options', label: 'Options' },
  { key: 'projections', label: 'Projections' },
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
    clauseReviews.value = res.data.clause_reviews || {}
    options.value = res.data.options || []
    documents.value = res.data.documents || {}
    loadAliases()
  } catch (e: any) {
    console.error('Failed to load risk analysis:', e)
  } finally {
    loading.value = false
  }
}

// ── Tenant Alias Management ──
async function loadAliases() {
  if (!selectedReviewId.value) return
  try {
    const [a, s] = await Promise.all([
      api.get('/api/lease-review/tenant-aliases'),
      api.get(`/api/lease-review/reviews/${selectedReviewId.value}/tenant-aliases/suggestions`),
    ])
    aliases.value = a.data
    aliasSuggestions.value = s.data
  } catch { /* ignore */ }
}

function openAliasModal(aliasName: string = '', canonicalName: string = '') {
  aliasForm.value = { alias_name: aliasName, canonical_name: canonicalName }
  showAliasModal.value = true
}

async function saveAlias() {
  if (!selectedReviewId.value || !aliasForm.value.alias_name || !aliasForm.value.canonical_name) return
  aliasLoading.value = true
  try {
    await api.post('/api/lease-review/tenant-aliases', aliasForm.value)
    showAliasModal.value = false
    await Promise.all([loadRiskData(), loadAliases()])
  } catch (e: any) {
    console.error('Failed to save alias:', e)
  } finally {
    aliasLoading.value = false
  }
}

async function removeAlias(aliasId: number) {
  if (!selectedReviewId.value) return
  try {
    await api.delete(`/api/lease-review/tenant-aliases/${aliasId}`)
    await Promise.all([loadRiskData(), loadAliases()])
  } catch (e: any) {
    console.error('Failed to delete alias:', e)
  }
}

async function applyAllSuggestions() {
  if (!selectedReviewId.value) return
  aliasLoading.value = true
  try {
    for (const s of aliasSuggestions.value) {
      const canon = s.suggested_canonical
      for (const v of s.variants) {
        if (v !== canon) {
          await api.post('/api/lease-review/tenant-aliases', {
            alias_name: v, canonical_name: canon,
          })
        }
      }
    }
    await Promise.all([loadRiskData(), loadAliases()])
  } catch (e: any) {
    console.error('Failed to apply suggestions:', e)
  } finally {
    aliasLoading.value = false
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

function tenantDocsForGroup(items: any[]): any[] {
  if (!items.length) return []
  const tid = items[0].tenant_id
  return documents.value[tid] || []
}

async function viewDocument(docId: number) {
  try {
    const res = await api.get(
      `/api/lease-review/reviews/${selectedReviewId.value}/documents/${docId}/view`,
      { responseType: 'blob' }
    )
    const blob = new Blob([res.data], { type: res.headers['content-type'] || 'application/pdf' })
    const url = URL.createObjectURL(blob)
    window.open(url, '_blank')
  } catch (e: any) {
    console.error('Failed to open document:', e)
  }
}

function openAbstract(tenantId: number) {
  router.push({
    path: '/lease-abstract',
    query: { review: String(selectedReviewId.value), tenant: String(tenantId) },
  })
}

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

// --- Tenant CRUD ---
async function addTenantSubmit() {
  if (!selectedReviewId.value) return
  try {
    await api.post(`/api/lease-review/reviews/${selectedReviewId.value}/tenants`, newTenant.value)
    showAddTenant.value = false
    newTenant.value = {}
    await loadRiskData()
  } catch (e: any) {
    alert('Failed to add tenant: ' + (e.response?.data?.error || e.message))
  }
}

async function deleteTenantConfirm(tenantId: number) {
  if (!selectedReviewId.value) return
  try {
    await api.delete(`/api/lease-review/reviews/${selectedReviewId.value}/tenants/${tenantId}`)
    confirmDeleteId.value = null
    await loadRiskData()
  } catch (e: any) {
    alert('Failed to delete tenant: ' + (e.response?.data?.error || e.message))
  }
}

async function toggleVacant(tenantId: number, currentVacant: boolean) {
  if (!selectedReviewId.value) return
  try {
    await api.put(`/api/lease-review/reviews/${selectedReviewId.value}/tenants/${tenantId}/vacant`, { vacant: !currentVacant })
    await loadRiskData()
  } catch (e: any) {
    alert('Failed to toggle vacant: ' + (e.response?.data?.error || e.message))
  }
}

function toggleSelection(id: number) {
  const s = new Set(selectedTenantIds.value)
  if (s.has(id)) s.delete(id); else s.add(id)
  selectedTenantIds.value = s
}

// --- Space Mutations ---
async function submitMerge() {
  if (!selectedReviewId.value) return
  try {
    await api.post(`/api/lease-review/reviews/${selectedReviewId.value}/space/merge`, {
      source_ids: Array.from(selectedTenantIds.value),
      merged_name: mergeForm.value.merged_name,
      merged_suite: mergeForm.value.merged_suite,
    })
    showMergeModal.value = false
    selectedTenantIds.value = new Set()
    mergeForm.value = { merged_name: '', merged_suite: '' }
    await loadRiskData()
    await loadSpaceEvents()
  } catch (e: any) {
    alert('Merge failed: ' + (e.response?.data?.error || e.message))
  }
}

function openSplitModal() {
  const sel = Array.from(selectedTenantIds.value)
  if (sel.length !== 1) return
  const t = tenants.value.find((x: any) => x.id === sel[0])
  splitSourceTenant.value = t
  splitForm.value = { splits: [
    { tenant_name: '', suite: '', square_feet: 0 },
    { tenant_name: '', suite: '', square_feet: 0 },
  ] }
  showSplitModal.value = true
}

function addSplitRow() {
  splitForm.value.splits.push({ tenant_name: '', suite: '', square_feet: 0 })
}

function removeSplitRow(i: number) {
  if (splitForm.value.splits.length > 2) splitForm.value.splits.splice(i, 1)
}

const splitSfTotal = computed(() => splitForm.value.splits.reduce((s: number, r: any) => s + (Number(r.square_feet) || 0), 0))
const splitSfValid = computed(() => {
  if (!splitSourceTenant.value?.square_feet) return true
  return Math.abs(splitSfTotal.value - splitSourceTenant.value.square_feet) <= 1
})

async function submitSplit() {
  if (!selectedReviewId.value || !splitSourceTenant.value) return
  try {
    await api.post(`/api/lease-review/reviews/${selectedReviewId.value}/space/split`, {
      source_id: splitSourceTenant.value.id,
      splits: splitForm.value.splits,
    })
    showSplitModal.value = false
    selectedTenantIds.value = new Set()
    await loadRiskData()
    await loadSpaceEvents()
  } catch (e: any) {
    alert('Split failed: ' + (e.response?.data?.error || e.message))
  }
}

// --- Succession ---
function openSuccessionModal(tenantId: number) {
  successionSourceId.value = tenantId
  const t = tenants.value.find((x: any) => x.id === tenantId)
  successionForm.value = { tenant_name: '', suite: t?.suite || '', square_feet: t?.square_feet || 0, effective_date: '' }
  showSuccessionModal.value = true
}

async function submitSuccession() {
  if (!selectedReviewId.value || !successionSourceId.value) return
  try {
    await api.post(`/api/lease-review/reviews/${selectedReviewId.value}/tenants/${successionSourceId.value}/succession`, {
      new_tenant: successionForm.value,
      effective_date: successionForm.value.effective_date,
    })
    showSuccessionModal.value = false
    await loadRiskData()
    await loadSpaceEvents()
  } catch (e: any) {
    alert('Succession failed: ' + (e.response?.data?.error || e.message))
  }
}

async function loadSuccessionChain(tenantId: number) {
  if (showChainFor.value === tenantId) { showChainFor.value = null; return }
  try {
    const res = await api.get(`/api/lease-review/reviews/${selectedReviewId.value}/tenants/${tenantId}/succession-chain`)
    successionChain.value = res.data
    showChainFor.value = tenantId
  } catch (e: any) {
    console.error('Chain load error:', e)
  }
}

// --- Space Events & Timeline ---
async function loadSpaceEvents() {
  if (!selectedReviewId.value) return
  try {
    const res = await api.get(`/api/lease-review/reviews/${selectedReviewId.value}/space-events`)
    spaceEvents.value = res.data
  } catch (e: any) {
    console.error('Space events load error:', e)
  }
}

async function submitPlanEvent() {
  if (!selectedReviewId.value) return
  try {
    await api.post(`/api/lease-review/reviews/${selectedReviewId.value}/space-events`, planEventForm.value)
    showPlanEventModal.value = false
    planEventForm.value = { event_type: 'vacate', effective_date: '', source_tenant_ids: [], results: [{ tenant_name: '', suite: '', square_feet: 0 }] }
    await loadSpaceEvents()
  } catch (e: any) {
    alert('Plan event failed: ' + (e.response?.data?.error || e.message))
  }
}

async function applyEvent(eventId: number) {
  if (!selectedReviewId.value) return
  try {
    await api.post(`/api/lease-review/reviews/${selectedReviewId.value}/space-events/${eventId}/apply`)
    await loadRiskData()
    await loadSpaceEvents()
  } catch (e: any) {
    alert('Apply failed: ' + (e.response?.data?.error || e.message))
  }
}

async function cancelEvent(eventId: number) {
  if (!selectedReviewId.value) return
  try {
    await api.delete(`/api/lease-review/reviews/${selectedReviewId.value}/space-events/${eventId}`)
    await loadRiskData()
    await loadSpaceEvents()
  } catch (e: any) {
    alert('Cancel failed: ' + (e.response?.data?.error || e.message))
  }
}

// --- Projections ---
async function loadAssumptions() {
  if (!selectedReviewId.value) return
  try {
    const res = await api.get(`/api/lease-review/reviews/${selectedReviewId.value}/market-assumptions`)
    marketAssumptions.value = res.data
  } catch (e: any) {
    console.error('Assumptions load error:', e)
  }
}

async function saveAssumptions() {
  if (!selectedReviewId.value) return
  try {
    await api.post(`/api/lease-review/reviews/${selectedReviewId.value}/market-assumptions`, { assumptions: marketAssumptions.value })
    alert('Assumptions saved')
  } catch (e: any) {
    alert('Save failed: ' + (e.response?.data?.error || e.message))
  }
}

function addAssumptionRow() {
  marketAssumptions.value.push({
    lease_type: '', market_rent_psf: 0, annual_rent_growth: 0.03,
    renewal_probability: 0.70, renewal_downtime_months: 0,
    renewal_ti_psf: 5, renewal_lc_pct: 0.04, renewal_rent_spread: 0,
    renewal_term_years: 5, new_downtime_months: 6, new_ti_psf: 15,
    new_lc_pct: 0.06, new_rent_spread: 0, new_term_years: 10,
    free_rent_months: 0, annual_expense_growth: 0.02,
  })
}

async function computeProjections() {
  if (!selectedReviewId.value || !projStartDate.value || !projEndDate.value) return
  projectionLoading.value = true
  try {
    const [cfRes, sumRes] = await Promise.all([
      api.get(`/api/lease-review/reviews/${selectedReviewId.value}/projected-cash-flow?start=${projStartDate.value}&end=${projEndDate.value}`),
      api.get(`/api/lease-review/reviews/${selectedReviewId.value}/projected-revenue-summary?start=${projStartDate.value}&end=${projEndDate.value}`),
    ])
    projectionData.value = cfRes.data
    revenueSummary.value = sumRes.data
  } catch (e: any) {
    alert('Projection failed: ' + (e.response?.data?.error || e.message))
  } finally {
    projectionLoading.value = false
  }
}

const revenueChartOpts = computed(() => {
  if (!revenueSummary.value?.summaries) return null
  const data = revenueSummary.value.summaries[projectionScenario.value] || []
  if (!data.length) return null
  return {
    tooltip: { trigger: 'axis' },
    legend: { data: ['Gross Rent', 'Vacancy Loss', 'TI/LC Costs', 'Net Effective'] },
    grid: { left: 80, right: 40, bottom: 30 },
    xAxis: { type: 'category', data: data.map((d: any) => d.year) },
    yAxis: { type: 'value', name: '$' },
    series: [
      { name: 'Gross Rent', type: 'bar', stack: 'rev', data: data.map((d: any) => Math.round(d.gross_rent)), itemStyle: { color: CLR_DARK } },
      { name: 'Vacancy Loss', type: 'bar', stack: 'rev', data: data.map((d: any) => -Math.round(d.vacancy_loss)), itemStyle: { color: CLR_RED } },
      { name: 'TI/LC Costs', type: 'bar', stack: 'rev', data: data.map((d: any) => -Math.round(d.ti_costs + d.lc_costs)), itemStyle: { color: CLR_ACCENT } },
      { name: 'Net Effective', type: 'line', data: data.map((d: any) => Math.round(d.net_effective)), itemStyle: { color: CLR_GREEN }, lineStyle: { width: 2 } },
    ],
  }
})

// Distinct lease types from current tenants
const leaseTypes = computed(() => {
  const types = new Set<string>()
  for (const t of tenants.value) {
    if (t.lease_type) types.add(t.lease_type)
  }
  return Array.from(types).sort()
})

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

// Projection helpers
const projectionYears = computed(() => {
  if (!projectionData.value?.months?.length) return []
  const years = new Set<string>()
  for (const m of projectionData.value.months) years.add(m.substring(0, 4))
  return Array.from(years).sort()
})

function projAnnualRent(suite: any, year: string): number {
  const entries = (suite[projectionScenario.value] || []).filter((e: any) => e.month.startsWith(year))
  return entries.reduce((s: number, e: any) => s + (e.effective_rent || 0), 0)
}

function projCellClass(suite: any, year: string): string {
  const entries = (suite[projectionScenario.value] || []).filter((e: any) => e.month.startsWith(year))
  if (!entries.length) return ''
  const phases = new Set(entries.map((e: any) => e.phase))
  if (phases.has('vacancy')) return 'proj-vacancy'
  if (phases.has('new_tenant')) return 'proj-new'
  if (phases.has('renewal')) return 'proj-renewal'
  return 'proj-inplace'
}

const expandedScenario = ref<string | null>(null)

watch(selectedReviewId, () => {
  loadRiskData()
  loadSpaceEvents()
  loadAssumptions()
})

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

        <!-- View Toggle + Action Buttons -->
        <div class="toolbar">
          <div class="toolbar-left">
            <button class="btn-sm" :class="{ active: overviewMode === 'current' }" @click="overviewMode = 'current'">Current</button>
            <button class="btn-sm" :class="{ active: overviewMode === 'timeline' }" @click="overviewMode = 'timeline'">Timeline</button>
          </div>
          <div class="toolbar-right">
            <button class="btn-sm btn-primary" @click="showAddTenant = true">+ Add Tenant</button>
            <button class="btn-sm btn-primary" :disabled="selectedTenantIds.size < 2" @click="showMergeModal = true">Merge</button>
            <button class="btn-sm btn-primary" :disabled="selectedTenantIds.size !== 1" @click="openSplitModal()">Split</button>
            <button class="btn-sm btn-secondary" @click="showPlanEventModal = true">Plan Event</button>
          </div>
        </div>

        <!-- Add Tenant Inline Form -->
        <div v-if="showAddTenant" class="inline-form">
          <h4>Add New Tenant</h4>
          <div class="form-row">
            <input v-model="newTenant.tenant_name" placeholder="Tenant Name" class="form-input" />
            <input v-model="newTenant.suite" placeholder="Suite" class="form-input sm" />
            <input v-model.number="newTenant.square_feet" placeholder="SF" type="number" class="form-input sm" />
            <input v-model.number="newTenant.annual_rent" placeholder="Annual Rent" type="number" class="form-input sm" />
            <input v-model="newTenant.lease_start" placeholder="Lease Start" type="date" class="form-input sm" />
            <input v-model="newTenant.lease_end" placeholder="Lease End" type="date" class="form-input sm" />
            <label class="form-check"><input type="checkbox" v-model="newTenant.is_vacant" /> Vacant</label>
            <button class="btn-sm btn-save" @click="addTenantSubmit">Save</button>
            <button class="btn-sm btn-cancel" @click="showAddTenant = false">Cancel</button>
          </div>
        </div>

        <!-- ═══ CURRENT VIEW ═══ -->
        <template v-if="overviewMode === 'current'">
          <h3>Tenant Roster (Resolved Data)</h3>
          <div class="table-wrapper">
            <table class="data-table">
              <thead>
                <tr>
                  <th class="chk-col"><input type="checkbox" @change="(e: any) => { if (e.target.checked) tenants.forEach((t: any) => selectedTenantIds.add(t.id)); else selectedTenantIds = new Set() }" /></th>
                  <th>Tenant</th>
                  <th>Suite</th>
                  <th>SF</th>
                  <th>Annual Rent</th>
                  <th>Rent/SF</th>
                  <th>Lease Start</th>
                  <th>Lease End</th>
                  <th>Vacant</th>
                  <th>Approval</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="t in tenants" :key="t.id"
                    :class="{ vacant: t.is_vacant, resolved: Object.keys(t.resolutions || {}).length > 0 }">
                  <td class="chk-col"><input type="checkbox" :checked="selectedTenantIds.has(t.id)" @change="toggleSelection(t.id)" /></td>
                  <td>
                    {{ t.tenant_name }}
                    <span v-if="t.successor_tenant_id" class="chain-link" @click="loadSuccessionChain(t.id)" title="View succession chain">&#x1F517;</span>
                  </td>
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
                  <td class="date-cell" :class="{ 'has-resolution': t.resolutions?.lease_start }">
                    <span @dblclick="startEdit(t.id, 'lease_start', t.lease_start)">{{ fmtDate(t.lease_start) }}</span>
                    <span v-if="t.resolutions?.lease_start" class="resolution-badge">R</span>
                  </td>
                  <td class="date-cell" :class="{ 'has-resolution': t.resolutions?.lease_end }">
                    <span @dblclick="startEdit(t.id, 'lease_end', t.lease_end)">{{ fmtDate(t.lease_end) }}</span>
                    <span v-if="t.resolutions?.lease_end" class="resolution-badge">R</span>
                  </td>
                  <td>
                    <button class="btn-xs" :class="t.is_vacant ? 'btn-exercised' : 'btn-not-exercised'" @click="toggleVacant(t.id, t.is_vacant)">
                      {{ t.is_vacant ? 'Yes' : 'No' }}
                    </button>
                  </td>
                  <td>
                    <span class="status-badge" :class="t.approval_status">{{ t.approval_status }}</span>
                  </td>
                  <td class="actions-cell">
                    <button class="btn-xs btn-abstract" @click.stop="router.push({ path: '/lease-abstract', query: { review: String(selectedReviewId), tenant: String(t.id) } })">Abstract</button>
                    <button class="btn-xs btn-succession" @click="openSuccessionModal(t.id)" title="Replace with successor">&#x27A1;</button>
                    <button v-if="confirmDeleteId !== t.id" class="btn-xs btn-delete" @click="confirmDeleteId = t.id" title="Delete tenant">&#x1F5D1;</button>
                    <template v-else>
                      <button class="btn-xs btn-delete-confirm" @click="deleteTenantConfirm(t.id)">Confirm</button>
                      <button class="btn-xs btn-cancel" @click="confirmDeleteId = null">X</button>
                    </template>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          <!-- Succession Chain Display -->
          <div v-if="showChainFor && successionChain.length" class="succession-chain">
            <h4>Succession Chain</h4>
            <div class="chain-items">
              <div v-for="(c, i) in successionChain" :key="c.id" class="chain-item" :class="{ replaced: c.tenant_status === 'replaced', current: c.id === showChainFor }">
                <div class="chain-name">{{ c.tenant_name }}</div>
                <div class="chain-detail">{{ c.suite }} &middot; {{ c.square_feet?.toLocaleString() }} SF &middot; {{ fmt$(c.annual_rent) }}</div>
                <div class="chain-dates">{{ fmtDate(c.lease_start) }} - {{ fmtDate(c.lease_end) }}</div>
                <span v-if="i < successionChain.length - 1" class="chain-arrow">&#x2192;</span>
              </div>
            </div>
            <button class="btn-xs btn-cancel" @click="showChainFor = null">Close</button>
          </div>

          <!-- Show replaced tenants toggle -->
          <div class="replaced-toggle">
            <label><input type="checkbox" v-model="showReplacedTenants" /> Show replaced/deleted tenants</label>
          </div>

          <!-- Space Changes History -->
          <div v-if="spaceEvents.length" class="space-history">
            <h3>Space Changes History</h3>
            <div v-for="e in spaceEvents" :key="e.id" class="event-row" :class="e.status">
              <span class="event-badge" :class="e.event_type">{{ e.event_type }}</span>
              <span class="event-date">{{ fmtDate(e.effective_date) }}</span>
              <span class="event-desc">{{ e.description }}</span>
              <span class="event-status status-badge" :class="e.status">{{ e.status }}</span>
              <button v-if="e.status === 'planned'" class="btn-xs btn-save" @click="applyEvent(e.id)">Apply</button>
              <button v-if="e.status !== 'cancelled'" class="btn-xs btn-delete" @click="cancelEvent(e.id)">Cancel</button>
            </div>
          </div>
        </template>

        <!-- ═══ TIMELINE VIEW ═══ -->
        <template v-if="overviewMode === 'timeline'">
          <h3>Space Planning Timeline</h3>
          <div v-if="!spaceEvents.length" class="empty-state">No space events planned. Use the toolbar buttons to merge, split, or plan future events.</div>
          <div v-else class="timeline-list">
            <div v-for="e in spaceEvents" :key="e.id" class="timeline-event" :class="{ future: e.status === 'planned', cancelled: e.status === 'cancelled' }">
              <div class="timeline-marker" :class="e.status"></div>
              <div class="timeline-content">
                <div class="timeline-header">
                  <span class="event-badge" :class="e.event_type">{{ e.event_type }}</span>
                  <span class="event-date">{{ fmtDate(e.effective_date) }}</span>
                  <span class="event-status status-badge" :class="e.status">{{ e.status }}</span>
                </div>
                <div class="timeline-desc">{{ e.description }}</div>
                <div v-if="e.source_tenants?.length" class="timeline-sources">
                  Source: {{ e.source_tenants.map((s: any) => `${s.name} (${s.suite})`).join(', ') }}
                </div>
                <div v-if="e.results?.length" class="timeline-results">
                  <span v-for="r in e.results" :key="r.id" class="result-chip">
                    {{ r.tenant_name }} &middot; {{ r.suite }} &middot; {{ r.square_feet?.toLocaleString() }} SF
                  </span>
                </div>
                <div v-if="e.status === 'planned'" class="timeline-actions">
                  <button class="btn-xs btn-save" @click="applyEvent(e.id)">Apply</button>
                  <button class="btn-xs btn-delete" @click="cancelEvent(e.id)">Cancel</button>
                </div>
              </div>
            </div>
          </div>
        </template>
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
            <div class="validation-group-header">
              <h4>{{ tenantKey }}</h4>
              <div class="validation-group-links">
                <button class="btn-xs btn-abstract" @click="openAbstract(items[0].tenant_id)" title="View lease abstract">Abstract</button>
                <template v-for="doc in tenantDocsForGroup(items)" :key="doc.id">
                  <button
                    v-if="doc.has_file"
                    class="btn-xs btn-doc"
                    @click="viewDocument(doc.id)"
                    :title="doc.filename"
                  >{{ doc.doc_type || 'PDF' }}</button>
                </template>
              </div>
            </div>
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

          <!-- Alias suggestions banner -->
          <div v-if="aliasSuggestions.length" class="alias-suggestions">
            <strong>Possible duplicates detected:</strong>
            <span v-for="(s, i) in aliasSuggestions" :key="i" class="alias-suggestion-item">
              {{ s.variants.join(' / ') }} → <em>{{ s.suggested_canonical }}</em>
            </span>
            <button class="btn-sm btn-primary" @click="applyAllSuggestions" :disabled="aliasLoading" style="margin-left: 0.5rem">
              {{ aliasLoading ? 'Applying...' : 'Apply All Suggestions' }}
            </button>
          </div>

          <div class="table-wrapper">
            <table class="data-table">
              <thead><tr><th>Co-Tenant</th><th>Dependent Tenants</th><th>Dependent Rent</th><th>Termination Eligible</th><th style="width:40px"></th></tr></thead>
              <tbody>
                <tr v-for="(risk, name) in (cotenancy.rent_at_risk || {})" :key="name as string">
                  <td>{{ name }}</td>
                  <td class="num-cell">{{ risk.dependent_count }}</td>
                  <td class="num-cell">{{ fmt$(risk.total_dependent_rent) }}</td>
                  <td class="num-cell" :class="{ danger: risk.termination_eligible_count > 0 }">{{ risk.termination_eligible_count }}</td>
                  <td><button class="btn-xs" title="Create alias for this name" @click="openAliasModal(name as string)">Map</button></td>
                </tr>
              </tbody>
            </table>
          </div>

          <!-- Active aliases -->
          <div v-if="aliases.length" style="margin-top: 1.5rem">
            <h4>Active Tenant Aliases <span class="badge badge-info">Global</span></h4>
            <div class="table-wrapper">
              <table class="data-table compact">
                <thead><tr><th>Alias (Extracted Name)</th><th>Canonical Name</th><th>Created By</th><th style="width:40px"></th></tr></thead>
                <tbody>
                  <tr v-for="a in aliases" :key="a.id">
                    <td>{{ a.alias_name }}</td>
                    <td><strong>{{ a.canonical_name }}</strong></td>
                    <td>{{ a.created_by || '-' }}</td>
                    <td><button class="btn-xs btn-danger" @click="removeAlias(a.id)" title="Remove alias">×</button></td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <!-- Alias modal -->
          <div v-if="showAliasModal" class="modal-overlay" @click.self="showAliasModal = false">
            <div class="modal-content" style="max-width: 500px">
              <h3>Map Tenant Name</h3>
              <p class="section-desc">Map an extracted name to its canonical tenant name. This alias applies across all properties.</p>
              <div class="form-row">
                <label>Extracted Name (Alias)</label>
                <input v-model="aliasForm.alias_name" />
              </div>
              <div class="form-row">
                <label>Canonical Name</label>
                <input v-model="aliasForm.canonical_name" list="tenant-names-list" />
                <datalist id="tenant-names-list">
                  <option v-for="t in tenants" :key="t.id" :value="t.tenant_name" />
                </datalist>
              </div>
              <div class="modal-actions">
                <button class="btn-primary" @click="saveAlias" :disabled="aliasLoading || !aliasForm.alias_name || !aliasForm.canonical_name">
                  {{ aliasLoading ? 'Saving...' : 'Save Alias' }}
                </button>
                <button class="btn-secondary" @click="showAliasModal = false">Cancel</button>
              </div>
            </div>
          </div>
        </template>
        <div v-else class="empty-state">No co-tenancy clauses found for this review.</div>

        <!-- Every active tenant, so "no co-tenancy provision" can be confirmed as
             an answer and not just inferred from an empty table. -->
        <h3>Co-Tenancy Review by Tenant</h3>
        <div class="table-wrapper">
          <table class="data-table">
            <thead><tr><th>Tenant</th><th>Suite</th><th>Clause extracted</th><th>Review</th></tr></thead>
            <tbody>
              <template v-for="t in tenants" :key="t.id">
                <tr>
                  <td>{{ t.tenant_name }}</td>
                  <td>{{ t.suite }}</td>
                  <td>{{ (cotenancy?.clauses || []).some((c: any) => c.tenant_name === t.tenant_name) ? 'Yes' : '—' }}</td>
                  <td>
                    <div class="review-cell">
                      <span class="rv-chip" :class="'rv-' + (reviewOf(t.id, 'cotenancy')?.status || 'unreviewed')">
                        {{ reviewOf(t.id, 'cotenancy')?.status || 'unreviewed' }}</span>
                      <span v-if="reviewOf(t.id, 'cotenancy')?.reread_at" class="rv-reread">re-read since</span>
                      <span v-if="reviewOf(t.id, 'cotenancy')?.notes" class="rv-notes">
                        {{ reviewOf(t.id, 'cotenancy').notes }}</span>
                      <button v-if="canReview" class="rv-btn" @click="startReview(t.id, 'cotenancy')">Review</button>
                    </div>
                  </td>
                </tr>
                <tr v-if="reviewEdit && reviewEdit.tenantId === t.id && reviewEdit.section === 'cotenancy'">
                  <td colspan="4">
                    <div class="rv-form">
                      <select v-model="reviewForm.status">
                        <option value="unreviewed">unreviewed</option>
                        <option value="confirmed">confirmed</option>
                        <option value="flagged">flagged</option>
                      </select>
                      <textarea v-model="reviewForm.notes" rows="2"
                                placeholder="What was checked, what was corrected, what is still open"></textarea>
                      <button class="btn-primary" :disabled="reviewSaving" @click="saveReview">Save</button>
                      <button class="btn-secondary" @click="reviewEdit = null">Cancel</button>
                      <span v-if="reviewError" class="rv-error">{{ reviewError }}</span>
                    </div>
                  </td>
                </tr>
              </template>
            </tbody>
          </table>
        </div>
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
        <h3>Exclusive Use Restrictions</h3>
        <div class="excl-toolbar">
          <label>Show
            <select v-model="exclusiveFilter">
              <option value="all">every tenant</option>
              <option value="with">tenants with restrictions</option>
              <option value="unreviewed">not yet reviewed</option>
              <option value="flagged">flagged or re-read since review</option>
            </select>
          </label>
          <label class="cb"><input type="checkbox" v-model="showBoundBy" />
            Show restrictions a tenant is <em>bound by</em></label>
          <span class="hint-inline">
            <b>Holds</b> = this tenant's own exclusive. <b>Bound by</b> = another tenant's
            exclusive or a prohibited-use list this lease discloses, such as an exhibit
            of existing exclusives.
          </span>
        </div>
        <div v-if="!exclusiveGroups.length" class="empty-state">No tenant matches.</div>
        <div v-for="g in exclusiveGroups" :key="g.tenant_id" class="excl-group">
          <div class="excl-head">
            <div class="excl-name">
              <b>{{ g.tenant_name }}</b> <span class="muted">{{ g.suite }}</span>
              <span v-if="g.inactive" class="muted"> · not an active tenant</span>
              <span class="muted"> · holds {{ g.holds.length }}, bound by {{ g.bound.length }}</span>
            </div>
            <div class="review-cell">
              <span class="rv-chip" :class="'rv-' + (g.review?.status || 'unreviewed')">
                {{ g.review?.status || 'unreviewed' }}</span>
              <span v-if="g.review?.reread_at" class="rv-reread"
                    title="The tenant's documents were re-read after this review">re-read since</span>
              <span v-if="g.review?.notes" class="rv-notes">{{ g.review.notes }}</span>
              <button v-if="canReview" class="rv-btn"
                      @click="startReview(g.tenant_id, 'exclusive_use')">Review</button>
            </div>
          </div>
          <div v-if="reviewEdit && reviewEdit.tenantId === g.tenant_id && reviewEdit.section === 'exclusive_use'"
               class="rv-form">
            <select v-model="reviewForm.status">
              <option value="unreviewed">unreviewed</option>
              <option value="confirmed">confirmed</option>
              <option value="flagged">flagged</option>
            </select>
            <textarea v-model="reviewForm.notes" rows="2"
                      placeholder="What was checked, what was corrected, what is still open"></textarea>
            <button class="btn-primary" :disabled="reviewSaving" @click="saveReview">Save</button>
            <button class="btn-secondary" @click="reviewEdit = null">Cancel</button>
            <span v-if="reviewError" class="rv-error">{{ reviewError }}</span>
          </div>
          <table v-if="g.holds.length + g.other.length + (showBoundBy ? g.bound.length : 0)"
                 class="data-table excl-table">
            <thead><tr><th>Type</th><th>Restricted Use</th><th>Radius</th><th>Carve-Outs</th>
              <th>Restriction Text</th><th>Source</th></tr></thead>
            <tbody>
              <tr v-for="(e, i) in [...g.holds, ...g.other, ...(showBoundBy ? g.bound : [])]" :key="i">
                <td>
                  <span v-if="e.clause_role === 'holder'" class="role-badge role-holder"
                        title="This tenant holds the exclusive">Holds</span>
                  <span v-else-if="e.clause_role === 'subject'" class="role-badge role-subject"
                        title="This tenant is bound by a restriction">Bound by</span>
                  <span v-else class="role-badge role-unknown" title="Not classified">—</span>
                </td>
                <td>{{ e.restricted_use || '-' }}</td>
                <td class="nowrap">{{ radiusText(e.radius_feet) }}</td>
                <td class="wrap-cell">{{ e.carve_outs || '-' }}</td>
                <td class="wrap-cell">{{ e.restriction_text || '-' }}</td>
                <td class="src-cell">{{ e.source_doc || '-' }}</td>
              </tr>
            </tbody>
          </table>
          <div v-else-if="!(g.holds.length + g.bound.length + g.other.length)" class="muted excl-none">
            No exclusive-use rows extracted.</div>
        </div>
      </div>

      <!-- ═══ RENT ROLL TAB — the lease timeline, the IC exhibit's layout ═══ -->
      <div v-if="activeTab === 'rentroll'" class="tab-content">
        <h3>Rent Roll</h3>
        <div v-if="rentRollLoading" class="empty-state">Building the timeline…</div>
        <div v-else-if="rentRollError" class="rv-error">{{ rentRollError }}</div>
        <template v-else-if="rentRoll">
          <div class="rr-meta">
            As of <b>{{ rrDate(rentRoll.as_of) }}</b> ·
            {{ rentRoll.tenants.length }} tenants ·
            {{ rentRoll.flag_count }} flag(s) to review ·
            current rent, future steps and remaining options from the governing
            documents, or as settled by the analyst.
          </div>
          <table class="rr-table">
            <thead><tr><th>Tenant</th><th>SF</th><th>Start</th><th>Expiration</th>
              <th>Annual Rent</th><th>Annual PSF</th><th>Option(s)</th><th></th></tr></thead>
            <tbody v-for="t in rentRoll.tenants" :key="t.tenant_id" class="rr-group">
              <tr v-for="(r, i) in t.rows" :key="i" :class="'rr-' + r.kind">
                <td :class="r.kind === 'tenant' ? 'rr-name' : 'rr-label'">
                  {{ r.kind === 'tenant' ? t.tenant_name : (r.label || '') }}</td>
                <td class="num">{{ r.kind === 'tenant' ? rrMoney(r.sf) : '' }}</td>
                <td class="num">{{ rrDate(r.start) }}</td>
                <td class="num">{{ rrDate(r.end) }}</td>
                <td class="num" :class="{ 'rr-blank': r.annual_rent === null && r.kind !== 'tenant' }">
                  {{ r.annual_rent === null && r.kind.startsWith('option') ? 'not stated' : rrMoney(r.annual_rent) }}
                  <span v-if="r.derived" class="rr-derived" title="Compounded from a stated percentage">d</span></td>
                <td class="num">{{ rrPsf(r.psf) }}</td>
                <td class="num">{{ r.kind === 'tenant' ? r.options : '' }}</td>
                <td class="rr-actions">
                  <template v-if="r.kind === 'tenant'">
                    <button v-if="t.flags.length" class="rr-flagbtn" @click="toggleFlags(t.tenant_id)">
                      {{ t.flags.length }} flag{{ t.flags.length === 1 ? '' : 's' }}</button>
                    <span v-if="t.schedule_basis !== 'lease schedule' || t.options_basis !== 'governing documents'
                                || t.start_basis === 'settled by the analyst' || t.expiration_basis === 'settled by the analyst'"
                          class="rv-chip rv-confirmed">settled</span>
                    <button v-if="canReview" class="rv-btn" @click="openSettle(t)">Settle</button>
                  </template>
                </td>
              </tr>
              <tr v-if="openFlags.has(t.tenant_id)" class="rr-flags">
                <td colspan="8"><ul><li v-for="(f, j) in t.flags" :key="j">
                  <b>{{ f.code.replace(/_/g, ' ') }}</b> — {{ f.message }}</li></ul></td>
              </tr>
              <tr v-if="settleFor && settleFor.tenant_id === t.tenant_id" class="rr-settle">
                <td colspan="8">
                  <div class="rr-settle-box">
                    <div class="rr-settle-head"><b>Settle {{ t.tenant_name }}</b>
                      <button class="btn-secondary" @click="settleFor = null">Close</button></div>
                    <div class="rr-settle-common">
                      <label>Reason (required)
                        <input v-model="settleForm.reason" placeholder="What the documents show, and why" /></label>
                      <label>Document cited
                        <select v-model="settleForm.doc">
                          <option v-for="d in tenantDocs(t.tenant_id)" :key="d.id" :value="d.id">
                            {{ String(d.filename).split('/').pop() }}</option>
                        </select></label>
                    </div>
                    <div class="rr-settle-sec">
                      <b>Start / Expiration</b>
                      <input type="date" v-model="settleForm.start" />
                      <input type="date" v-model="settleForm.expiration" />
                      <button class="btn-primary" :disabled="settleSaving" @click="saveSettlement('dates')">Save dates</button>
                    </div>
                    <div class="rr-settle-sec">
                      <b>Future rent steps</b>
                      <div v-for="(r, k) in settleForm.schedule" :key="'s' + k" class="rr-row">
                        <input type="date" v-model="r.start" /><input type="date" v-model="r.end" />
                        <input type="number" v-model.number="r.annual_rent" placeholder="Annual rent" />
                        <button class="rv-btn" @click="settleForm.schedule.splice(k, 1)">×</button>
                      </div>
                      <button class="rv-btn" @click="settleForm.schedule.push({ start: '', end: '', annual_rent: null })">+ step</button>
                      <button class="btn-primary" :disabled="settleSaving" @click="saveSettlement('schedule')">Save schedule</button>
                      <button v-if="t.schedule_basis === 'settled by the analyst'" class="btn-secondary"
                              @click="clearSettlement('schedule')">Revert to documents</button>
                    </div>
                    <div class="rr-settle-sec">
                      <b>Options</b> <span class="muted">(rows with the same option number are that option's rent steps; leave rent blank for fair market)</span>
                      <div v-for="(r, k) in settleForm.options" :key="'o' + k" class="rr-row">
                        <input type="number" v-model.number="r.option" class="rr-optno" title="Option number" />
                        <input type="date" v-model="r.start" /><input type="date" v-model="r.end" />
                        <input type="number" v-model="r.annual_rent" placeholder="Annual rent" />
                        <button class="rv-btn" @click="settleForm.options.splice(k, 1)">×</button>
                      </div>
                      <button class="rv-btn" @click="settleForm.options.push({ option: (settleForm.options.at(-1)?.option || 0) + 1, start: '', end: '', annual_rent: '' })">+ row</button>
                      <button class="btn-primary" :disabled="settleSaving" @click="saveSettlement('options')">Save options</button>
                      <button v-if="t.options_basis === 'settled by the analyst'" class="btn-secondary"
                              @click="clearSettlement('options')">Revert to documents</button>
                    </div>
                    <div v-if="settleMsg" class="rr-msg">{{ settleMsg }}</div>
                  </div>
                </td>
              </tr>
            </tbody>
            <tfoot><tr class="rr-total"><td>Total / Wtd. Avg.</td>
              <td class="num">{{ rrMoney(rentRoll.totals.square_feet) }}</td><td></td><td></td>
              <td class="num">{{ rrMoney(rentRoll.totals.annual_rent) }}</td>
              <td class="num">{{ rrPsf(rentRoll.totals.psf) }}</td><td></td><td></td></tr></tfoot>
          </table>
        </template>
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

      <!-- ═══ PROJECTIONS TAB ═══ -->
      <div v-if="activeTab === 'projections'" class="tab-content">
        <!-- Market Assumptions -->
        <h3>Leasing Assumptions by Type</h3>
        <div class="table-wrapper">
          <table class="data-table compact">
            <thead>
              <tr>
                <th>Lease Type</th>
                <th>Market Rent/SF</th>
                <th>Annual Growth</th>
                <th>Renewal Prob.</th>
                <th>Renewal Down (mo)</th>
                <th>Renewal TI/SF</th>
                <th>Renewal LC %</th>
                <th>Renewal Term (yr)</th>
                <th>New Down (mo)</th>
                <th>New TI/SF</th>
                <th>New LC %</th>
                <th>New Term (yr)</th>
                <th>Free Rent (mo)</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(a, i) in marketAssumptions" :key="i">
                <td><input v-model="a.lease_type" class="form-input xs" :list="'lt-list'" /></td>
                <td><input v-model.number="a.market_rent_psf" type="number" step="0.01" class="form-input xs" /></td>
                <td><input v-model.number="a.annual_rent_growth" type="number" step="0.01" class="form-input xs" /></td>
                <td><input v-model.number="a.renewal_probability" type="number" step="0.05" min="0" max="1" class="form-input xs" /></td>
                <td><input v-model.number="a.renewal_downtime_months" type="number" class="form-input xs" /></td>
                <td><input v-model.number="a.renewal_ti_psf" type="number" step="0.5" class="form-input xs" /></td>
                <td><input v-model.number="a.renewal_lc_pct" type="number" step="0.01" class="form-input xs" /></td>
                <td><input v-model.number="a.renewal_term_years" type="number" class="form-input xs" /></td>
                <td><input v-model.number="a.new_downtime_months" type="number" class="form-input xs" /></td>
                <td><input v-model.number="a.new_ti_psf" type="number" step="0.5" class="form-input xs" /></td>
                <td><input v-model.number="a.new_lc_pct" type="number" step="0.01" class="form-input xs" /></td>
                <td><input v-model.number="a.new_term_years" type="number" class="form-input xs" /></td>
                <td><input v-model.number="a.free_rent_months" type="number" class="form-input xs" /></td>
              </tr>
            </tbody>
          </table>
          <datalist id="lt-list">
            <option v-for="lt in leaseTypes" :key="lt" :value="lt" />
          </datalist>
        </div>
        <div class="form-row" style="margin-bottom:16px">
          <button class="btn-sm btn-secondary" @click="addAssumptionRow">+ Add Type</button>
          <button class="btn-sm btn-save" @click="saveAssumptions">Save Assumptions</button>
        </div>

        <!-- Projection Controls -->
        <h3>Generate Projections</h3>
        <div class="form-row">
          <label>Start: <input v-model="projStartDate" type="month" class="form-input sm" /></label>
          <label>End: <input v-model="projEndDate" type="month" class="form-input sm" /></label>
          <button class="btn-sm btn-primary" @click="computeProjections" :disabled="projectionLoading || !projStartDate || !projEndDate">
            {{ projectionLoading ? 'Computing...' : 'Compute' }}
          </button>
        </div>

        <template v-if="revenueSummary">
          <!-- Scenario Toggle -->
          <div class="scenario-toggle">
            <button class="btn-sm" :class="{ active: projectionScenario === 'renewal' }" @click="projectionScenario = 'renewal'">Renewal Case</button>
            <button class="btn-sm" :class="{ active: projectionScenario === 'new_tenant' }" @click="projectionScenario = 'new_tenant'">New Tenant Case</button>
            <button class="btn-sm" :class="{ active: projectionScenario === 'weighted' }" @click="projectionScenario = 'weighted'">Probability-Weighted</button>
          </div>

          <!-- Revenue Chart -->
          <div v-if="revenueChartOpts" class="chart-container">
            <v-chart :option="revenueChartOpts" style="height:350px" autoresize />
          </div>

          <!-- Revenue Summary Table -->
          <h3>Annual Revenue Summary ({{ projectionScenario.replace('_', ' ') }})</h3>
          <div class="table-wrapper">
            <table class="data-table">
              <thead>
                <tr>
                  <th>Year</th>
                  <th>Gross Rent</th>
                  <th>Vacancy Loss</th>
                  <th>TI Costs</th>
                  <th>LC Costs</th>
                  <th>Net Effective</th>
                  <th>Vacancy Rate</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="d in (revenueSummary.summaries[projectionScenario] || [])" :key="d.year">
                  <td>{{ d.year }}</td>
                  <td class="num-cell">{{ fmt$(d.gross_rent) }}</td>
                  <td class="num-cell danger">{{ fmt$(d.vacancy_loss) }}</td>
                  <td class="num-cell">{{ fmt$(d.ti_costs) }}</td>
                  <td class="num-cell">{{ fmt$(d.lc_costs) }}</td>
                  <td class="num-cell" style="font-weight:600">{{ fmt$(d.net_effective) }}</td>
                  <td class="num-cell">{{ fmtPct(d.vacancy_rate * 100) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </template>

        <!-- Projected Rent Roll (suite-level) -->
        <template v-if="projectionData?.suites">
          <h3>Projected Rent Roll by Suite ({{ projectionScenario.replace('_', ' ') }})</h3>
          <div class="table-wrapper" style="max-height:500px;overflow:auto">
            <table class="data-table compact">
              <thead>
                <tr>
                  <th>Suite</th>
                  <th>Tenant</th>
                  <th>SF</th>
                  <th>Lease End</th>
                  <th v-for="yr in projectionYears" :key="yr">{{ yr }}</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(suite, key) in projectionData.suites" :key="key">
                  <td>{{ suite.suite }}</td>
                  <td>{{ suite.tenant_name }}</td>
                  <td class="num-cell">{{ suite.sf?.toLocaleString() }}</td>
                  <td>{{ fmtDate(suite.lease_end) }}</td>
                  <td v-for="yr in projectionYears" :key="yr" class="num-cell"
                      :class="projCellClass(suite, yr)">
                    {{ fmt$(projAnnualRent(suite, yr)) }}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </template>
      </div>

      <!-- Modals -->
      <!-- Merge Modal -->
      <div v-if="showMergeModal" class="modal-overlay" @click.self="showMergeModal = false">
        <div class="modal-box">
          <h3>Merge {{ selectedTenantIds.size }} Tenants</h3>
          <div class="form-row"><label>Merged Name: <input v-model="mergeForm.merged_name" class="form-input" /></label></div>
          <div class="form-row"><label>Merged Suite: <input v-model="mergeForm.merged_suite" class="form-input sm" /></label></div>
          <div class="form-row">
            <button class="btn-sm btn-save" @click="submitMerge">Merge</button>
            <button class="btn-sm btn-cancel" @click="showMergeModal = false">Cancel</button>
          </div>
        </div>
      </div>

      <!-- Split Modal -->
      <div v-if="showSplitModal" class="modal-overlay" @click.self="showSplitModal = false">
        <div class="modal-box wide">
          <h3>Split {{ splitSourceTenant?.tenant_name }} ({{ splitSourceTenant?.square_feet?.toLocaleString() }} SF)</h3>
          <table class="data-table compact">
            <thead><tr><th>Tenant Name</th><th>Suite</th><th>SF</th><th></th></tr></thead>
            <tbody>
              <tr v-for="(s, i) in splitForm.splits" :key="i">
                <td><input v-model="s.tenant_name" class="form-input" /></td>
                <td><input v-model="s.suite" class="form-input sm" /></td>
                <td><input v-model.number="s.square_feet" type="number" class="form-input sm" /></td>
                <td><button v-if="splitForm.splits.length > 2" class="btn-xs btn-delete" @click="removeSplitRow(i)">X</button></td>
              </tr>
            </tbody>
          </table>
          <div class="split-validation">
            Total SF: {{ splitSfTotal.toLocaleString() }}
            <span v-if="!splitSfValid" class="danger"> (must match source {{ splitSourceTenant?.square_feet?.toLocaleString() }} SF)</span>
            <span v-else class="match-ok"> &#x2713;</span>
          </div>
          <div class="form-row">
            <button class="btn-sm btn-secondary" @click="addSplitRow">+ Row</button>
            <button class="btn-sm btn-save" :disabled="!splitSfValid" @click="submitSplit">Split</button>
            <button class="btn-sm btn-cancel" @click="showSplitModal = false">Cancel</button>
          </div>
        </div>
      </div>

      <!-- Succession Modal -->
      <div v-if="showSuccessionModal" class="modal-overlay" @click.self="showSuccessionModal = false">
        <div class="modal-box">
          <h3>Replace Tenant with Successor</h3>
          <div class="form-row"><label>New Tenant Name: <input v-model="successionForm.tenant_name" class="form-input" /></label></div>
          <div class="form-row"><label>Suite: <input v-model="successionForm.suite" class="form-input sm" /></label></div>
          <div class="form-row"><label>SF: <input v-model.number="successionForm.square_feet" type="number" class="form-input sm" /></label></div>
          <div class="form-row"><label>Annual Rent: <input v-model.number="successionForm.annual_rent" type="number" class="form-input sm" /></label></div>
          <div class="form-row"><label>Effective Date: <input v-model="successionForm.effective_date" type="date" class="form-input sm" /></label></div>
          <div class="form-row"><label>Lease Start: <input v-model="successionForm.lease_start" type="date" class="form-input sm" /></label></div>
          <div class="form-row"><label>Lease End: <input v-model="successionForm.lease_end" type="date" class="form-input sm" /></label></div>
          <div class="form-row">
            <button class="btn-sm btn-save" @click="submitSuccession">Create Succession</button>
            <button class="btn-sm btn-cancel" @click="showSuccessionModal = false">Cancel</button>
          </div>
        </div>
      </div>

      <!-- Plan Event Modal -->
      <div v-if="showPlanEventModal" class="modal-overlay" @click.self="showPlanEventModal = false">
        <div class="modal-box">
          <h3>Plan Future Space Event</h3>
          <div class="form-row">
            <label>Type:
              <select v-model="planEventForm.event_type" class="form-input sm">
                <option value="vacate">Vacate</option>
                <option value="renew">Renew</option>
                <option value="new_tenant">New Tenant</option>
                <option value="succession">Succession</option>
                <option value="resize">Resize</option>
              </select>
            </label>
          </div>
          <div class="form-row"><label>Effective Date: <input v-model="planEventForm.effective_date" type="date" class="form-input sm" /></label></div>
          <div class="form-row">
            <label>Source Tenants:
              <select v-model="planEventForm.source_tenant_ids" multiple class="form-input" style="height:80px">
                <option v-for="t in tenants" :key="t.id" :value="t.id">{{ t.tenant_name }} ({{ t.suite }})</option>
              </select>
            </label>
          </div>
          <div class="form-row"><label>Description: <input v-model="planEventForm.description" class="form-input" /></label></div>
          <h4>Result Tenant(s)</h4>
          <div v-for="(r, i) in planEventForm.results" :key="i" class="form-row">
            <input v-model="r.tenant_name" placeholder="Name" class="form-input sm" />
            <input v-model="r.suite" placeholder="Suite" class="form-input xs" />
            <input v-model.number="r.square_feet" placeholder="SF" type="number" class="form-input xs" />
          </div>
          <div class="form-row">
            <button class="btn-sm btn-save" @click="submitPlanEvent">Save Event</button>
            <button class="btn-sm btn-cancel" @click="showPlanEventModal = false">Cancel</button>
          </div>
        </div>
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
.role-badge { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 0.72rem; font-weight: 600; white-space: nowrap; }
.role-holder  { background: #e8eff6; color: #1f4e79; }
.role-subject { background: #f8f0de; color: #9a6a18; }
.role-unknown { background: #eee; color: #888; }
.excl-toolbar { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; margin: 6px 0 12px; font-size: 0.82rem; }
.excl-toolbar select { margin-left: 6px; }
.excl-toolbar .cb { display: flex; gap: 5px; align-items: center; }
.hint-inline { color: #6b7280; font-size: 0.78rem; flex-basis: 100%; }
.excl-group { border: 1px solid #e5e7eb; border-radius: 6px; margin-bottom: 10px; padding: 8px 10px; }
.excl-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; flex-wrap: wrap; }
.excl-table { margin-top: 6px; }
.excl-none { font-size: 0.8rem; margin-top: 4px; }
.muted { color: #6b7280; }
.nowrap { white-space: nowrap; }
.review-cell { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; font-size: 0.8rem; max-width: 560px; }
.rv-chip { padding: 1px 8px; border-radius: 10px; font-size: 0.72rem; font-weight: 600; }
.rv-unreviewed { background: #eef0f2; color: #6b7280; }
.rv-confirmed { background: #e6f3e8; color: #2f7a3d; }
.rv-flagged { background: #fdecd6; color: #9a5200; }
.rv-reread { background: #fde8e8; color: #b52b27; padding: 1px 6px; border-radius: 3px; font-size: 0.7rem; }
.rv-notes { color: #374151; font-style: italic; }
.rv-btn { font-size: 0.75rem; padding: 1px 8px; border: 1px solid #cbd5e1; background: #fff; border-radius: 4px; cursor: pointer; }
.rv-form { display: flex; gap: 8px; align-items: flex-start; flex-wrap: wrap; margin: 8px 0; }
.rv-form textarea { flex: 1; min-width: 280px; font-size: 0.8rem; padding: 4px 6px; }
.rv-error { color: #b52b27; font-size: 0.8rem; }
/* Rent Roll: the IC exhibit's look -- Times New Roman, grey header and total,
   a frame, a thin rule between tenants, italic subordinate labels. */
.rr-meta { font-size: 0.8rem; color: #6b7280; margin: 4px 0 10px; }
.rr-table { border-collapse: collapse; font-family: 'Times New Roman', Times, serif; font-size: 0.9rem;
  border: 2px solid #111; width: 100%; max-width: 1100px; }
.rr-table th { background: #f2f2f2; font-weight: bold; text-align: center; padding: 3px 8px;
  border-bottom: 1px solid #111; }
.rr-table td { padding: 2px 8px; }
.rr-table .num { text-align: right; white-space: nowrap; }
.rr-group { border-bottom: 1px solid #111; }
.rr-name { text-align: left; }
.rr-label { text-align: right; font-style: italic; }
.rr-blank { color: #9a5200; font-style: italic; }
.rr-derived { font-size: 0.7rem; color: #9a5200; vertical-align: super; }
.rr-total td { background: #f2f2f2; font-weight: bold; border-top: 1px solid #111; }
.rr-actions { white-space: nowrap; font-family: system-ui, sans-serif; }
.rr-flagbtn { font-size: 0.72rem; border: 1px solid #d9a441; background: #fdf7ea; color: #9a6700;
  border-radius: 10px; padding: 0 7px; cursor: pointer; margin-right: 4px; }
.rr-flags td, .rr-settle td { font-family: system-ui, sans-serif; font-size: 0.78rem; background: #fafafa; }
.rr-flags ul { margin: 4px 0; padding-left: 18px; }
.rr-settle-box { display: flex; flex-direction: column; gap: 8px; padding: 6px 0; }
.rr-settle-head { display: flex; justify-content: space-between; align-items: center; }
.rr-settle-common { display: flex; gap: 12px; flex-wrap: wrap; }
.rr-settle-common label { display: flex; flex-direction: column; gap: 2px; min-width: 260px; flex: 1; }
.rr-settle-sec { display: flex; gap: 6px; align-items: center; flex-wrap: wrap;
  border-top: 1px solid #e5e7eb; padding-top: 6px; }
.rr-row { display: flex; gap: 4px; flex-basis: 100%; }
.rr-row input { font-size: 0.78rem; padding: 2px 4px; }
.rr-optno { width: 48px; }
.rr-msg { font-size: 0.8rem; color: #374151; }
.src-cell { font-size: 0.75rem; color: #666; }
.wrap-cell { max-width: 300px; word-wrap: break-word; white-space: normal; }
/* ISO dates offer a break opportunity at each hyphen, so a narrow column
   splits 2028-05-31 over two lines and doubles the row height. */
.date-cell { white-space: nowrap; }
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
.validation-group h4 { color: #1F4E79; margin: 0; }
.validation-group-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin: 12px 0 8px;
  gap: 8px;
}
.validation-group-links {
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
  align-items: center;
}
.btn-doc {
  background: #f0f4f8;
  color: #1F4E79;
  border: 1px solid #ccc;
  font-weight: 500;
  max-width: 120px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.btn-doc:hover { background: #e3eef8; border-color: #1F4E79; }

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

/* Toolbar */
.toolbar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; gap: 8px; flex-wrap: wrap; }
.toolbar-left, .toolbar-right { display: flex; gap: 6px; }
.btn-sm {
  padding: 5px 12px; font-size: 0.82rem; border: 1px solid #ccc;
  border-radius: 4px; cursor: pointer; background: #f0f0f0;
}
.btn-sm.active { background: #1F4E79; color: #fff; border-color: #1F4E79; }
.btn-sm.btn-primary { background: #1F4E79; color: #fff; border-color: #1F4E79; }
.btn-sm.btn-primary:hover { background: #16395a; }
.btn-sm.btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-sm.btn-secondary { background: #f8f9fa; color: #333; }
.btn-sm.btn-secondary:hover { background: #e9ecef; }
.btn-sm.btn-save { background: #548235; color: #fff; border-color: #548235; }
.btn-sm.btn-cancel { background: #999; color: #fff; border-color: #999; }
.chk-col { width: 30px; text-align: center; }
.actions-cell { white-space: nowrap; }

/* Inline form */
.inline-form {
  background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px;
  padding: 12px 16px; margin-bottom: 16px;
}
.inline-form h4 { margin: 0 0 8px; color: #1F4E79; font-size: 0.9rem; }
.form-row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 8px; }
.form-input { padding: 4px 8px; border: 1px solid #ccc; border-radius: 3px; font-size: 0.82rem; }
.form-input.sm { width: 120px; }
.form-input.xs { width: 80px; }
.form-check { font-size: 0.82rem; display: flex; align-items: center; gap: 4px; }

/* Delete / succession buttons */
.btn-delete { background: transparent; color: #C00000; font-size: 0.8rem; }
.btn-delete:hover { background: #fde8e8; }
.btn-delete-confirm { background: #C00000; color: #fff; }
.btn-succession { background: #f0f4f8; color: #1F4E79; border: 1px solid #ccc; }
.btn-succession:hover { background: #e3eef8; }

/* Succession chain */
.succession-chain { background: #f0f4f8; border: 1px solid #cce; border-radius: 6px; padding: 12px; margin: 12px 0; }
.chain-items { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 8px; }
.chain-item {
  background: #fff; border: 1px solid #dee2e6; border-radius: 4px; padding: 8px 12px;
  min-width: 140px; position: relative;
}
.chain-item.replaced { opacity: 0.6; border-style: dashed; }
.chain-item.current { border-color: #1F4E79; border-width: 2px; }
.chain-name { font-weight: 600; font-size: 0.85rem; }
.chain-detail, .chain-dates { font-size: 0.75rem; color: #666; }
.chain-arrow { font-size: 1.2rem; color: #1F4E79; }
.chain-link { cursor: pointer; font-size: 0.7rem; }

/* Replaced toggle */
.replaced-toggle { margin: 12px 0; font-size: 0.82rem; color: #666; }

/* Space history */
.space-history { margin-top: 20px; }
.event-row { display: flex; gap: 10px; align-items: center; padding: 6px 0; border-bottom: 1px solid #eee; }
.event-row.cancelled { opacity: 0.5; }
.event-badge {
  display: inline-block; padding: 2px 8px; border-radius: 4px;
  font-size: 0.72rem; font-weight: 600; text-transform: capitalize;
  background: #e9ecef; color: #333;
}
.event-badge.merge { background: #E2EFDA; color: #375623; }
.event-badge.split { background: #DEEBF7; color: #1F4E79; }
.event-badge.succession { background: #FFF2CC; color: #856404; }
.event-badge.resize { background: #F2F2F2; color: #333; }
.event-badge.vacate { background: #FCE4EC; color: #C00000; }
.event-badge.new_tenant { background: #E8F5E9; color: #2E7D32; }
.event-badge.renew { background: #E2EFDA; color: #375623; }
.event-date { font-size: 0.82rem; color: #555; }
.event-desc { flex: 1; font-size: 0.82rem; }

/* Timeline */
.timeline-list { position: relative; padding-left: 24px; }
.timeline-event { position: relative; margin-bottom: 16px; padding: 10px 16px; background: #fff; border: 1px solid #dee2e6; border-radius: 6px; }
.timeline-event.future { border-style: dashed; border-color: #1F4E79; }
.timeline-event.cancelled { opacity: 0.4; }
.timeline-marker {
  position: absolute; left: -32px; top: 14px; width: 12px; height: 12px;
  border-radius: 50%; border: 2px solid #1F4E79; background: #fff;
}
.timeline-marker.applied { background: #548235; }
.timeline-marker.planned { background: #fff; }
.timeline-marker.cancelled { background: #C00000; }
.timeline-header { display: flex; gap: 10px; align-items: center; margin-bottom: 4px; }
.timeline-desc { font-size: 0.85rem; color: #333; }
.timeline-sources, .timeline-results { font-size: 0.78rem; color: #666; margin-top: 4px; }
.timeline-actions { margin-top: 6px; display: flex; gap: 6px; }
.result-chip {
  display: inline-block; padding: 2px 8px; background: #f0f4f8;
  border-radius: 3px; font-size: 0.75rem; margin: 2px;
}

/* Modals */
.modal-overlay {
  position: fixed; top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(0,0,0,0.4); display: flex; align-items: center;
  justify-content: center; z-index: 1000;
}
.modal-box {
  background: #fff; border-radius: 8px; padding: 24px; min-width: 400px;
  max-width: 600px; max-height: 80vh; overflow-y: auto; box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}
.modal-box.wide { min-width: 600px; max-width: 800px; }
.modal-box h3 { margin: 0 0 16px; color: #1F4E79; }
.modal-box h4 { margin: 12px 0 8px; color: #333; font-size: 0.9rem; }

/* Split validation */
.split-validation { font-size: 0.85rem; margin: 8px 0 12px; }
.match-ok { color: #548235; font-weight: 600; }

/* Projection colors */
.proj-inplace { background: #DEEBF7; }
.proj-renewal { background: #E2EFDA; }
.proj-new { background: #FFF2CC; }
.proj-vacancy { background: #FCE4EC; }

/* Scenario toggle */
.scenario-toggle { display: flex; gap: 4px; margin: 16px 0; }

/* Alias suggestions */
.alias-suggestions {
  background: #FFF8E1; border: 1px solid #FFD54F; border-radius: 6px;
  padding: 10px 14px; margin-bottom: 12px; font-size: 0.85rem;
}
.alias-suggestion-item {
  display: inline-block; margin: 2px 8px; padding: 2px 8px;
  background: #FFF3E0; border-radius: 4px; font-size: 0.82rem;
}
.badge-info {
  background: #1F4E79; color: #fff; font-size: 0.7rem; padding: 1px 6px;
  border-radius: 3px; margin-left: 4px; vertical-align: middle;
}
</style>
