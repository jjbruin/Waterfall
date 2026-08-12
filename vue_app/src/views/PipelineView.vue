<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import api from '../api/client'

const router = useRouter()
const route = useRoute()

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ProspectDeal {
  id: number
  deal_name: string
  location: string
  asset_type: string
  partner_name: string
  stage: string
  assigned_to: string
  target_close: string | null
  purchase_price: number | null
  source_broker: string
  onboarded_vcode: string | null
  created_by: string
  created_at: string | null
  updated_at: string | null
  property_count: number
}

interface Property {
  id: number
  property_name: string
  address: string
  city: string
  state: string
  zip: string
  asset_type: string
  gla_sf: number | null
  units: number | null
  year_built: number | null
  acreage: number | null
  property_price: number | null
  occupancy_pct: number | null
  noi_in_place: number | null
  notes: string
  onboarded_vcode: string | null
  sort_order: number
  lease_review_id: number | null
}

interface Entity {
  id: number
  entity_name: string
  entity_type: string
  planned_entity_id: string
  parent_entity_id: number | null
  ownership_pct: number | null
  role: string
  notes: string
  investors: Investor[]
}

interface Investor {
  id: number
  entity_id: number
  investor_name: string
  planned_investor_id: string
  commitment: number | null
  ownership_pct: number | null
  investor_type: string
  notes: string
}

interface Activity {
  id: number
  username: string
  action: string
  note: string
  created_at: string | null
}

// ---------------------------------------------------------------------------
// Pipeline stages
// ---------------------------------------------------------------------------

const STAGES = [
  { key: 'lead', label: 'Lead', color: '#9e9e9e' },
  { key: 'screening', label: 'Screening', color: '#42a5f5' },
  { key: 'loi', label: 'LOI', color: '#7e57c2' },
  { key: 'due_diligence', label: 'Due Diligence', color: '#ff9800' },
  { key: 'ic_review', label: 'IC Review', color: '#ef5350' },
  { key: 'closing', label: 'Closing', color: '#66bb6a' },
  { key: 'closed', label: 'Closed', color: '#2e7d32' },
  { key: 'passed', label: 'Passed', color: '#bdbdbd' },
]

const ASSET_TYPES = ['Retail', 'Multifamily', 'Office', 'Industrial', 'Mixed-Use']

function stageLabel(key: string): string {
  return STAGES.find(s => s.key === key)?.label || key
}
function stageColor(key: string): string {
  return STAGES.find(s => s.key === key)?.color || '#9e9e9e'
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const deals = ref<ProspectDeal[]>([])
const loading = ref(false)
const error = ref<string | null>(null)
const viewMode = ref<'kanban' | 'table'>('kanban')

// Filters
const stageFilter = ref('')
const assignedFilter = ref('')

// New deal modal
const showNewDeal = ref(false)
const newDeal = ref({
  deal_name: '',
  location: '',
  asset_type: '',
  partner_name: '',
  source_broker: '',
  purchase_price: null as number | null,
  target_close: '',
  notes: '',
})
const saving = ref(false)

// Deal detail
const selectedDealId = ref<number | null>(null)
const dealDetail = ref<any>(null)
const detailLoading = ref(false)
const detailTab = ref<'properties' | 'entities' | 'activity'>('properties')
const activity = ref<Activity[]>([])

// Property form
const showPropertyForm = ref(false)
const editingProperty = ref<Property | null>(null)
const propertyForm = ref({
  property_name: '',
  address: '',
  city: '',
  state: '',
  zip: '',
  asset_type: '',
  gla_sf: null as number | null,
  units: null as number | null,
  year_built: null as number | null,
  acreage: null as number | null,
  property_price: null as number | null,
  occupancy_pct: null as number | null,
  noi_in_place: null as number | null,
  notes: '',
})

// Entity form
const showEntityForm = ref(false)
const entityForm = ref({
  entity_name: '',
  entity_type: 'deal_jv',
  planned_entity_id: '',
  ownership_pct: null as number | null,
  role: 'sponsor',
  notes: '',
})

// Investor form
const showInvestorForm = ref(false)
const investorEntityId = ref<number | null>(null)
const investorForm = ref({
  investor_name: '',
  planned_investor_id: '',
  commitment: null as number | null,
  ownership_pct: null as number | null,
  investor_type: 'pref_equity',
  notes: '',
})

// Activity note
const newNote = ref('')

// Drag state
const dragDealId = ref<number | null>(null)

// ---------------------------------------------------------------------------
// Computed
// ---------------------------------------------------------------------------

const filteredDeals = computed(() => {
  let result = deals.value
  if (stageFilter.value) {
    result = result.filter(d => d.stage === stageFilter.value)
  }
  if (assignedFilter.value) {
    result = result.filter(d => d.assigned_to === assignedFilter.value)
  }
  return result
})

const dealsByStage = computed(() => {
  const map: Record<string, ProspectDeal[]> = {}
  for (const s of STAGES) {
    map[s.key] = filteredDeals.value.filter(d => d.stage === s.key)
  }
  return map
})

const assignedOptions = computed(() => {
  const names = new Set(deals.value.map(d => d.assigned_to).filter(Boolean))
  return Array.from(names).sort()
})

const stageCounts = computed(() => {
  const counts: Record<string, number> = {}
  for (const s of STAGES) {
    counts[s.key] = deals.value.filter(d => d.stage === s.key).length
  }
  return counts
})

const deal = computed(() => dealDetail.value?.deal)
const properties = computed<Property[]>(() => dealDetail.value?.properties || [])
const entities = computed<Entity[]>(() => dealDetail.value?.entities || [])

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

async function loadDeals() {
  loading.value = true
  error.value = null
  try {
    const res = await api.get('/api/prospects')
    deals.value = res.data
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    loading.value = false
  }
}

async function createDeal() {
  if (!newDeal.value.deal_name.trim()) return
  saving.value = true
  try {
    const res = await api.post('/api/prospects', newDeal.value)
    showNewDeal.value = false
    resetNewDeal()
    await loadDeals()
    // Open the new deal
    openDeal(res.data.id)
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    saving.value = false
  }
}

function resetNewDeal() {
  newDeal.value = {
    deal_name: '', location: '', asset_type: '', partner_name: '',
    source_broker: '', purchase_price: null, target_close: '', notes: '',
  }
}

async function openDeal(id: number) {
  selectedDealId.value = id
  detailLoading.value = true
  detailTab.value = 'properties'
  try {
    const [detailRes, actRes] = await Promise.all([
      api.get(`/api/prospects/${id}`),
      api.get(`/api/prospects/${id}/activity`),
    ])
    dealDetail.value = detailRes.data
    activity.value = actRes.data
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    detailLoading.value = false
  }
}

function closeDeal() {
  selectedDealId.value = null
  dealDetail.value = null
  activity.value = []
}

async function updateStage(dealId: number, newStage: string) {
  try {
    await api.put(`/api/prospects/${dealId}`, { stage: newStage })
    await loadDeals()
    if (selectedDealId.value === dealId) {
      await openDeal(dealId)
    }
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  }
}

async function saveDealField(field: string, value: any) {
  if (!selectedDealId.value) return
  try {
    await api.put(`/api/prospects/${selectedDealId.value}`, { [field]: value })
    await loadDeals()
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  }
}

async function deleteDeal() {
  if (!selectedDealId.value) return
  if (!confirm('Delete this deal and all related data?')) return
  try {
    await api.delete(`/api/prospects/${selectedDealId.value}`)
    closeDeal()
    await loadDeals()
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  }
}

// Properties
async function saveProperty() {
  if (!selectedDealId.value || !propertyForm.value.property_name.trim()) return
  saving.value = true
  try {
    if (editingProperty.value) {
      await api.put(
        `/api/prospects/${selectedDealId.value}/properties/${editingProperty.value.id}`,
        propertyForm.value,
      )
    } else {
      await api.post(
        `/api/prospects/${selectedDealId.value}/properties`,
        propertyForm.value,
      )
    }
    showPropertyForm.value = false
    editingProperty.value = null
    await openDeal(selectedDealId.value)
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    saving.value = false
  }
}

function editProperty(p: Property) {
  editingProperty.value = p
  propertyForm.value = { ...p }
  showPropertyForm.value = true
}

function addProperty() {
  editingProperty.value = null
  propertyForm.value = {
    property_name: '', address: '', city: '', state: '', zip: '',
    asset_type: deal.value?.asset_type || '', gla_sf: null, units: null,
    year_built: null, acreage: null, property_price: null,
    occupancy_pct: null, noi_in_place: null, notes: '',
  }
  showPropertyForm.value = true
}

async function removeProperty(propId: number) {
  if (!selectedDealId.value) return
  if (!confirm('Remove this property?')) return
  try {
    await api.delete(`/api/prospects/${selectedDealId.value}/properties/${propId}`)
    await openDeal(selectedDealId.value)
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  }
}

async function createLeaseReview(propId: number) {
  if (!selectedDealId.value) return
  try {
    const res = await api.post(
      `/api/prospects/${selectedDealId.value}/properties/${propId}/lease-review`,
    )
    await openDeal(selectedDealId.value)
    // Navigate to lease review
    router.push({ path: '/lease-review', query: { id: res.data.review_id } })
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  }
}

// Entities
async function saveEntity() {
  if (!selectedDealId.value || !entityForm.value.entity_name.trim()) return
  saving.value = true
  try {
    await api.post(`/api/prospects/${selectedDealId.value}/entities`, entityForm.value)
    showEntityForm.value = false
    await openDeal(selectedDealId.value)
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    saving.value = false
  }
}

async function removeEntity(entityId: number) {
  if (!selectedDealId.value) return
  if (!confirm('Remove this entity and its investors?')) return
  try {
    await api.delete(`/api/prospects/${selectedDealId.value}/entities/${entityId}`)
    await openDeal(selectedDealId.value)
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  }
}

// Investors
async function saveInvestor() {
  if (!selectedDealId.value || !investorEntityId.value) return
  if (!investorForm.value.investor_name.trim()) return
  saving.value = true
  try {
    await api.post(
      `/api/prospects/${selectedDealId.value}/entities/${investorEntityId.value}/investors`,
      investorForm.value,
    )
    showInvestorForm.value = false
    await openDeal(selectedDealId.value)
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    saving.value = false
  }
}

function addInvestor(entityId: number) {
  investorEntityId.value = entityId
  investorForm.value = {
    investor_name: '', planned_investor_id: '', commitment: null,
    ownership_pct: null, investor_type: 'pref_equity', notes: '',
  }
  showInvestorForm.value = true
}

async function removeInvestor(investorId: number) {
  if (!selectedDealId.value) return
  if (!confirm('Remove this investor?')) return
  try {
    await api.delete(`/api/prospects/${selectedDealId.value}/investors/${investorId}`)
    await openDeal(selectedDealId.value)
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  }
}

// Activity
async function addNote() {
  if (!selectedDealId.value || !newNote.value.trim()) return
  try {
    await api.post(`/api/prospects/${selectedDealId.value}/activity`, {
      note: newNote.value.trim(),
    })
    newNote.value = ''
    const res = await api.get(`/api/prospects/${selectedDealId.value}/activity`)
    activity.value = res.data
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  }
}

// Drag & drop
function onDragStart(e: DragEvent, dealId: number) {
  dragDealId.value = dealId
  if (e.dataTransfer) {
    e.dataTransfer.effectAllowed = 'move'
  }
}

function onDragOver(e: DragEvent) {
  e.preventDefault()
  if (e.dataTransfer) e.dataTransfer.dropEffect = 'move'
}

function onDrop(e: DragEvent, stage: string) {
  e.preventDefault()
  if (dragDealId.value !== null) {
    updateStage(dragDealId.value, stage)
    dragDealId.value = null
  }
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

function fmtCurrency(v: number | null): string {
  if (v == null) return '—'
  if (Math.abs(v) >= 1_000_000) return '$' + (v / 1_000_000).toFixed(1) + 'M'
  if (Math.abs(v) >= 1_000) return '$' + (v / 1_000).toFixed(0) + 'K'
  return '$' + v.toLocaleString()
}

function fmtPct(v: number | null): string {
  if (v == null) return '—'
  return (v * 100).toFixed(1) + '%'
}

function fmtDate(dt: string | null): string {
  if (!dt) return '—'
  const d = new Date(dt)
  if (isNaN(d.getTime())) return dt
  return `${d.getMonth() + 1}/${d.getDate()}/${d.getFullYear()}`
}

function fmtNum(v: number | null): string {
  if (v == null) return '—'
  return v.toLocaleString()
}

function actionIcon(action: string): string {
  const icons: Record<string, string> = {
    created: '+', stage_change: '\u2192', note: '\u270E',
    property_added: '\u2302', lease_review_created: '\u2611',
    evaluated: '\u2713',
  }
  return icons[action] || '\u2022'
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

onMounted(() => {
  loadDeals()
  // If URL has ?deal=id, open it
  if (route.query.deal) {
    openDeal(Number(route.query.deal))
  }
})
</script>

<template>
  <div class="pipeline-view">
    <!-- Header -->
    <div class="pipeline-header">
      <h2>New Business Pipeline</h2>
      <div class="header-actions">
        <div class="view-toggle">
          <button
            :class="{ active: viewMode === 'kanban' }"
            @click="viewMode = 'kanban'"
          >Kanban</button>
          <button
            :class="{ active: viewMode === 'table' }"
            @click="viewMode = 'table'"
          >Table</button>
        </div>
        <div class="filter-group" v-if="viewMode === 'table'">
          <select v-model="stageFilter" class="filter-select">
            <option value="">All Stages</option>
            <option v-for="s in STAGES" :key="s.key" :value="s.key">{{ s.label }}</option>
          </select>
        </div>
        <div class="filter-group">
          <select v-model="assignedFilter" class="filter-select">
            <option value="">All Users</option>
            <option v-for="a in assignedOptions" :key="a" :value="a">{{ a }}</option>
          </select>
        </div>
        <button class="btn-primary" @click="showNewDeal = true">+ New Deal</button>
      </div>
    </div>

    <!-- Error -->
    <div v-if="error" class="error-banner">
      {{ error }}
      <button @click="error = null">Dismiss</button>
    </div>

    <!-- Stage summary bar -->
    <div class="stage-summary">
      <div
        v-for="s in STAGES"
        :key="s.key"
        class="stage-pill"
        :style="{ borderColor: s.color }"
        :class="{ active: stageFilter === s.key }"
        @click="stageFilter = stageFilter === s.key ? '' : s.key"
      >
        <span class="pill-count" :style="{ color: s.color }">{{ stageCounts[s.key] }}</span>
        <span class="pill-label">{{ s.label }}</span>
      </div>
    </div>

    <!-- Loading -->
    <div v-if="loading" class="loading-text">Loading pipeline...</div>

    <!-- ============ KANBAN VIEW ============ -->
    <div v-if="viewMode === 'kanban' && !loading" class="kanban-board">
      <div
        v-for="stage in STAGES"
        :key="stage.key"
        class="kanban-column"
        @dragover="onDragOver"
        @drop="onDrop($event, stage.key)"
      >
        <div class="column-header" :style="{ borderTopColor: stage.color }">
          <span class="column-title">{{ stage.label }}</span>
          <span class="column-count">{{ dealsByStage[stage.key].length }}</span>
        </div>
        <div class="column-body">
          <div
            v-for="d in dealsByStage[stage.key]"
            :key="d.id"
            class="kanban-card"
            draggable="true"
            @dragstart="onDragStart($event, d.id)"
            @click="openDeal(d.id)"
          >
            <div class="card-name">{{ d.deal_name }}</div>
            <div class="card-location" v-if="d.location">{{ d.location }}</div>
            <div class="card-meta">
              <span v-if="d.purchase_price" class="card-price">{{ fmtCurrency(d.purchase_price) }}</span>
              <span v-if="d.asset_type" class="card-type" :title="d.asset_type">{{ d.asset_type }}</span>
            </div>
            <div class="card-footer">
              <span v-if="d.partner_name" class="card-partner">{{ d.partner_name }}</span>
              <span v-if="d.property_count" class="card-props">{{ d.property_count }} prop{{ d.property_count > 1 ? 's' : '' }}</span>
            </div>
          </div>
          <div v-if="!dealsByStage[stage.key].length" class="column-empty">
            Drop deals here
          </div>
        </div>
      </div>
    </div>

    <!-- ============ TABLE VIEW ============ -->
    <div v-if="viewMode === 'table' && !loading" class="table-wrap">
      <table class="pipeline-table">
        <thead>
          <tr>
            <th>Deal Name</th>
            <th>Location</th>
            <th>Asset Type</th>
            <th>Partner</th>
            <th>Stage</th>
            <th class="r">Purchase Price</th>
            <th class="r">Properties</th>
            <th>Target Close</th>
            <th>Updated</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="d in filteredDeals"
            :key="d.id"
            class="clickable-row"
            @click="openDeal(d.id)"
          >
            <td class="deal-name">{{ d.deal_name }}</td>
            <td>{{ d.location || '—' }}</td>
            <td>{{ d.asset_type || '—' }}</td>
            <td>{{ d.partner_name || '—' }}</td>
            <td>
              <span class="stage-badge" :style="{ background: stageColor(d.stage) }">
                {{ stageLabel(d.stage) }}
              </span>
            </td>
            <td class="r">{{ fmtCurrency(d.purchase_price) }}</td>
            <td class="r">{{ d.property_count || 0 }}</td>
            <td>{{ d.target_close || '—' }}</td>
            <td>{{ fmtDate(d.updated_at) }}</td>
          </tr>
          <tr v-if="!filteredDeals.length">
            <td colspan="9" class="empty-row">No deals found.</td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- ============ NEW DEAL MODAL ============ -->
    <div v-if="showNewDeal" class="modal-overlay" @click.self="showNewDeal = false">
      <div class="modal-content">
        <div class="modal-header">
          <h3>New Prospect Deal</h3>
          <button class="modal-close" @click="showNewDeal = false">&times;</button>
        </div>
        <div class="modal-body">
          <div class="form-grid">
            <div class="form-field span-2">
              <label>Deal Name *</label>
              <input v-model="newDeal.deal_name" placeholder="e.g., Vestavia Hills City Center" />
            </div>
            <div class="form-field">
              <label>Location</label>
              <input v-model="newDeal.location" placeholder="City, State" />
            </div>
            <div class="form-field">
              <label>Asset Type</label>
              <select v-model="newDeal.asset_type">
                <option value="">Select...</option>
                <option v-for="t in ASSET_TYPES" :key="t" :value="t">{{ t }}</option>
              </select>
            </div>
            <div class="form-field">
              <label>Partner Name</label>
              <input v-model="newDeal.partner_name" placeholder="e.g., Burton Property Group" />
            </div>
            <div class="form-field">
              <label>Source / Broker</label>
              <input v-model="newDeal.source_broker" />
            </div>
            <div class="form-field">
              <label>Purchase Price</label>
              <input v-model.number="newDeal.purchase_price" type="number" placeholder="$" />
            </div>
            <div class="form-field">
              <label>Target Close</label>
              <input v-model="newDeal.target_close" type="date" />
            </div>
            <div class="form-field span-2">
              <label>Notes</label>
              <textarea v-model="newDeal.notes" rows="2"></textarea>
            </div>
          </div>
        </div>
        <div class="modal-footer">
          <button class="btn-secondary" @click="showNewDeal = false">Cancel</button>
          <button class="btn-primary" @click="createDeal" :disabled="saving || !newDeal.deal_name.trim()">
            {{ saving ? 'Creating...' : 'Create Deal' }}
          </button>
        </div>
      </div>
    </div>

    <!-- ============ DEAL DETAIL PANEL ============ -->
    <div v-if="selectedDealId" class="detail-overlay" @click.self="closeDeal">
      <div class="detail-panel">
        <div v-if="detailLoading" class="loading-text" style="padding:40px">Loading deal...</div>
        <template v-else-if="deal">
          <!-- Detail header -->
          <div class="detail-header">
            <button class="btn-back" @click="closeDeal">&larr; Back to Pipeline</button>
            <div class="detail-title-row">
              <h2>{{ deal.deal_name }}</h2>
              <select
                class="stage-select"
                :value="deal.stage"
                @change="updateStage(deal.id, ($event.target as HTMLSelectElement).value)"
                :style="{ borderColor: stageColor(deal.stage) }"
              >
                <option v-for="s in STAGES" :key="s.key" :value="s.key">{{ s.label }}</option>
              </select>
            </div>
            <div class="detail-subtitle">
              <span v-if="deal.location">{{ deal.location }}</span>
              <span v-if="deal.asset_type"> | {{ deal.asset_type }}</span>
              <span v-if="deal.partner_name"> | {{ deal.partner_name }}</span>
            </div>
          </div>

          <!-- Deal info cards -->
          <div class="info-cards">
            <div class="info-card">
              <span class="info-label">Purchase Price</span>
              <span class="info-value">{{ fmtCurrency(deal.purchase_price) }}</span>
            </div>
            <div class="info-card">
              <span class="info-label">Properties</span>
              <span class="info-value">{{ properties.length }}</span>
            </div>
            <div class="info-card">
              <span class="info-label">Total GLA</span>
              <span class="info-value">{{ fmtNum(properties.reduce((s: number, p: Property) => s + (p.gla_sf || 0), 0)) }} SF</span>
            </div>
            <div class="info-card">
              <span class="info-label">Target Close</span>
              <span class="info-value">{{ deal.target_close || '—' }}</span>
            </div>
            <div class="info-card">
              <span class="info-label">Source</span>
              <span class="info-value">{{ deal.source_broker || '—' }}</span>
            </div>
          </div>

          <!-- Tab bar -->
          <div class="tab-bar">
            <button
              v-for="tab in [
                { key: 'properties', label: `Properties (${properties.length})` },
                { key: 'entities', label: `Entities (${entities.length})` },
                { key: 'activity', label: 'Activity' },
              ]"
              :key="tab.key"
              :class="{ active: detailTab === tab.key }"
              @click="detailTab = tab.key as any"
            >{{ tab.label }}</button>
          </div>

          <!-- Properties tab -->
          <div v-if="detailTab === 'properties'" class="tab-content">
            <div class="tab-actions">
              <button class="btn-primary btn-sm" @click="addProperty">+ Add Property</button>
            </div>
            <div v-if="!properties.length" class="empty-state">
              No properties yet. Add the first property for this deal.
            </div>
            <div v-else class="property-grid">
              <div v-for="p in properties" :key="p.id" class="property-card">
                <div class="prop-header">
                  <h4>{{ p.property_name }}</h4>
                  <div class="prop-actions">
                    <button class="btn-icon" @click="editProperty(p)" title="Edit">&#x270E;</button>
                    <button class="btn-icon btn-danger" @click="removeProperty(p.id)" title="Remove">&times;</button>
                  </div>
                </div>
                <div class="prop-address" v-if="p.address || p.city">
                  {{ [p.address, p.city, p.state, p.zip].filter(Boolean).join(', ') }}
                </div>
                <div class="prop-details">
                  <span v-if="p.asset_type"><strong>Type:</strong> {{ p.asset_type }}</span>
                  <span v-if="p.gla_sf"><strong>GLA:</strong> {{ fmtNum(p.gla_sf) }} SF</span>
                  <span v-if="p.units"><strong>Units:</strong> {{ p.units }}</span>
                  <span v-if="p.year_built"><strong>Built:</strong> {{ p.year_built }}</span>
                  <span v-if="p.property_price"><strong>Price:</strong> {{ fmtCurrency(p.property_price) }}</span>
                  <span v-if="p.occupancy_pct != null"><strong>Occ:</strong> {{ fmtPct(p.occupancy_pct) }}</span>
                  <span v-if="p.noi_in_place"><strong>NOI:</strong> {{ fmtCurrency(p.noi_in_place) }}</span>
                </div>
                <div class="prop-footer">
                  <button
                    v-if="p.lease_review_id"
                    class="btn-link"
                    @click="router.push({ path: '/lease-review', query: { id: p.lease_review_id } })"
                  >View Lease Review</button>
                  <button
                    v-else
                    class="btn-link"
                    @click="createLeaseReview(p.id)"
                  >Start Lease Review</button>
                </div>
              </div>
            </div>
          </div>

          <!-- Entities tab -->
          <div v-if="detailTab === 'entities'" class="tab-content">
            <div class="tab-actions">
              <button class="btn-primary btn-sm" @click="showEntityForm = true">+ Add Entity</button>
            </div>
            <div v-if="!entities.length" class="empty-state">
              No entities defined. Add the ownership structure for this deal.
            </div>
            <div v-for="ent in entities" :key="ent.id" class="entity-card">
              <div class="entity-header">
                <div>
                  <h4>{{ ent.entity_name }}</h4>
                  <span class="entity-type">{{ ent.entity_type || 'entity' }}</span>
                  <span v-if="ent.planned_entity_id" class="entity-id">ID: {{ ent.planned_entity_id }}</span>
                  <span v-if="ent.role" class="entity-role">{{ ent.role }}</span>
                  <span v-if="ent.ownership_pct != null" class="entity-pct">{{ fmtPct(ent.ownership_pct) }}</span>
                </div>
                <div class="prop-actions">
                  <button class="btn-icon" @click="addInvestor(ent.id)" title="Add Investor">+</button>
                  <button class="btn-icon btn-danger" @click="removeEntity(ent.id)" title="Remove">&times;</button>
                </div>
              </div>
              <div v-if="ent.investors.length" class="investor-table">
                <table>
                  <thead>
                    <tr>
                      <th>Investor</th>
                      <th>Planned ID</th>
                      <th>Type</th>
                      <th class="r">Commitment</th>
                      <th class="r">Ownership</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="inv in ent.investors" :key="inv.id">
                      <td class="deal-name">{{ inv.investor_name }}</td>
                      <td>{{ inv.planned_investor_id || '—' }}</td>
                      <td>{{ inv.investor_type || '—' }}</td>
                      <td class="r">{{ fmtCurrency(inv.commitment) }}</td>
                      <td class="r">{{ inv.ownership_pct != null ? fmtPct(inv.ownership_pct) : '—' }}</td>
                      <td>
                        <button class="btn-icon btn-danger btn-xs" @click="removeInvestor(inv.id)">&times;</button>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <div v-else class="entity-empty">No investors. Click + to add.</div>
            </div>
          </div>

          <!-- Activity tab -->
          <div v-if="detailTab === 'activity'" class="tab-content">
            <div class="note-input">
              <input
                v-model="newNote"
                placeholder="Add a note..."
                @keyup.enter="addNote"
              />
              <button class="btn-primary btn-sm" @click="addNote" :disabled="!newNote.trim()">Add</button>
            </div>
            <div v-if="!activity.length" class="empty-state">No activity yet.</div>
            <div v-for="a in activity" :key="a.id" class="activity-item">
              <span class="activity-icon">{{ actionIcon(a.action) }}</span>
              <div class="activity-body">
                <span class="activity-note">{{ a.note }}</span>
                <span class="activity-meta">{{ a.username }} &middot; {{ fmtDate(a.created_at) }}</span>
              </div>
            </div>
          </div>

          <!-- Delete -->
          <div class="detail-footer">
            <button class="btn-danger-text" @click="deleteDeal">Delete Deal</button>
          </div>
        </template>
      </div>
    </div>

    <!-- ============ PROPERTY FORM MODAL ============ -->
    <div v-if="showPropertyForm" class="modal-overlay" @click.self="showPropertyForm = false">
      <div class="modal-content">
        <div class="modal-header">
          <h3>{{ editingProperty ? 'Edit Property' : 'Add Property' }}</h3>
          <button class="modal-close" @click="showPropertyForm = false">&times;</button>
        </div>
        <div class="modal-body">
          <div class="form-grid">
            <div class="form-field span-2">
              <label>Property Name *</label>
              <input v-model="propertyForm.property_name" />
            </div>
            <div class="form-field span-2">
              <label>Address</label>
              <input v-model="propertyForm.address" />
            </div>
            <div class="form-field">
              <label>City</label>
              <input v-model="propertyForm.city" />
            </div>
            <div class="form-field">
              <label>State</label>
              <input v-model="propertyForm.state" maxlength="2" style="width:60px" />
            </div>
            <div class="form-field">
              <label>ZIP</label>
              <input v-model="propertyForm.zip" maxlength="10" style="width:90px" />
            </div>
            <div class="form-field">
              <label>Asset Type</label>
              <select v-model="propertyForm.asset_type">
                <option value="">Select...</option>
                <option v-for="t in ASSET_TYPES" :key="t" :value="t">{{ t }}</option>
              </select>
            </div>
            <div class="form-field">
              <label>GLA (SF)</label>
              <input v-model.number="propertyForm.gla_sf" type="number" />
            </div>
            <div class="form-field">
              <label>Units</label>
              <input v-model.number="propertyForm.units" type="number" />
            </div>
            <div class="form-field">
              <label>Year Built</label>
              <input v-model.number="propertyForm.year_built" type="number" />
            </div>
            <div class="form-field">
              <label>Acreage</label>
              <input v-model.number="propertyForm.acreage" type="number" step="0.01" />
            </div>
            <div class="form-field">
              <label>Allocated Price</label>
              <input v-model.number="propertyForm.property_price" type="number" />
            </div>
            <div class="form-field">
              <label>Occupancy %</label>
              <input v-model.number="propertyForm.occupancy_pct" type="number" step="0.01" min="0" max="1" />
            </div>
            <div class="form-field">
              <label>In-Place NOI</label>
              <input v-model.number="propertyForm.noi_in_place" type="number" />
            </div>
            <div class="form-field span-2">
              <label>Notes</label>
              <textarea v-model="propertyForm.notes" rows="2"></textarea>
            </div>
          </div>
        </div>
        <div class="modal-footer">
          <button class="btn-secondary" @click="showPropertyForm = false">Cancel</button>
          <button class="btn-primary" @click="saveProperty" :disabled="saving || !propertyForm.property_name.trim()">
            {{ saving ? 'Saving...' : (editingProperty ? 'Update' : 'Add Property') }}
          </button>
        </div>
      </div>
    </div>

    <!-- ============ ENTITY FORM MODAL ============ -->
    <div v-if="showEntityForm" class="modal-overlay" @click.self="showEntityForm = false">
      <div class="modal-content modal-sm">
        <div class="modal-header">
          <h3>Add Entity</h3>
          <button class="modal-close" @click="showEntityForm = false">&times;</button>
        </div>
        <div class="modal-body">
          <div class="form-stack">
            <div class="form-field">
              <label>Entity Name *</label>
              <input v-model="entityForm.entity_name" placeholder="e.g., PPI-WS Holdings LLC" />
            </div>
            <div class="form-field">
              <label>Entity Type</label>
              <select v-model="entityForm.entity_type">
                <option value="deal_jv">Deal JV</option>
                <option value="gp">GP</option>
                <option value="lp">LP</option>
                <option value="holding">Holding</option>
                <option value="property">Property</option>
              </select>
            </div>
            <div class="form-field">
              <label>Planned EntityID</label>
              <input v-model="entityForm.planned_entity_id" placeholder="e.g., PPI-WS" />
            </div>
            <div class="form-field">
              <label>Role</label>
              <select v-model="entityForm.role">
                <option value="sponsor">Sponsor</option>
                <option value="investor">Investor</option>
                <option value="co_investor">Co-Investor</option>
                <option value="manager">Manager</option>
              </select>
            </div>
            <div class="form-field">
              <label>Ownership %</label>
              <input v-model.number="entityForm.ownership_pct" type="number" step="0.01" min="0" max="1" />
            </div>
          </div>
        </div>
        <div class="modal-footer">
          <button class="btn-secondary" @click="showEntityForm = false">Cancel</button>
          <button class="btn-primary" @click="saveEntity" :disabled="saving || !entityForm.entity_name.trim()">
            {{ saving ? 'Saving...' : 'Add Entity' }}
          </button>
        </div>
      </div>
    </div>

    <!-- ============ INVESTOR FORM MODAL ============ -->
    <div v-if="showInvestorForm" class="modal-overlay" @click.self="showInvestorForm = false">
      <div class="modal-content modal-sm">
        <div class="modal-header">
          <h3>Add Investor</h3>
          <button class="modal-close" @click="showInvestorForm = false">&times;</button>
        </div>
        <div class="modal-body">
          <div class="form-stack">
            <div class="form-field">
              <label>Investor Name *</label>
              <input v-model="investorForm.investor_name" placeholder="e.g., PSC1" />
            </div>
            <div class="form-field">
              <label>Planned InvestorID</label>
              <input v-model="investorForm.planned_investor_id" placeholder="e.g., PSC1" />
            </div>
            <div class="form-field">
              <label>Type</label>
              <select v-model="investorForm.investor_type">
                <option value="pref_equity">Preferred Equity</option>
                <option value="op_equity">OP Equity</option>
                <option value="co_invest">Co-Invest</option>
              </select>
            </div>
            <div class="form-field">
              <label>Commitment ($)</label>
              <input v-model.number="investorForm.commitment" type="number" />
            </div>
            <div class="form-field">
              <label>Ownership %</label>
              <input v-model.number="investorForm.ownership_pct" type="number" step="0.01" min="0" max="1" />
            </div>
          </div>
        </div>
        <div class="modal-footer">
          <button class="btn-secondary" @click="showInvestorForm = false">Cancel</button>
          <button class="btn-primary" @click="saveInvestor" :disabled="saving || !investorForm.investor_name.trim()">
            {{ saving ? 'Saving...' : 'Add Investor' }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.pipeline-view { padding: 0 0 40px 0; }

/* Header */
.pipeline-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
  flex-wrap: wrap;
  gap: 8px;
}
.pipeline-header h2 { font-size: 18px; margin: 0; }
.header-actions { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }

.view-toggle {
  display: flex;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  overflow: hidden;
}
.view-toggle button {
  padding: 5px 14px;
  border: none;
  background: var(--color-surface);
  font-size: 12px;
  cursor: pointer;
}
.view-toggle button.active {
  background: var(--color-accent);
  color: white;
}
.view-toggle button + button { border-left: 1px solid var(--color-border); }

.filter-select {
  padding: 5px 10px;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  font-size: 12px;
}

.btn-primary {
  padding: 6px 16px;
  background: var(--color-accent);
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
  font-weight: 600;
}
.btn-primary:hover { opacity: 0.9; }
.btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-primary.btn-sm { padding: 4px 12px; font-size: 11px; }

.btn-secondary {
  padding: 6px 16px;
  background: var(--color-surface);
  color: #333;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
}

/* Stage summary */
.stage-summary {
  display: flex;
  gap: 6px;
  margin-bottom: 14px;
  flex-wrap: wrap;
}
.stage-pill {
  border: 2px solid #ccc;
  border-radius: 20px;
  padding: 3px 14px;
  display: flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
  transition: all 0.15s;
  background: var(--color-surface);
}
.stage-pill:hover { transform: translateY(-1px); box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.stage-pill.active { background: #f0f4ff; }
.pill-count { font-weight: 700; font-size: 14px; }
.pill-label { font-size: 11px; color: #666; }

/* Error */
.error-banner {
  background: #fef2f2;
  border: 1px solid #fca5a5;
  color: #991b1b;
  padding: 8px 14px;
  border-radius: 6px;
  margin-bottom: 12px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 13px;
}
.error-banner button {
  background: none;
  border: 1px solid #fca5a5;
  color: #991b1b;
  padding: 3px 10px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
}

.loading-text {
  text-align: center;
  padding: 40px;
  color: #999;
  font-style: italic;
}

/* ===== KANBAN ===== */
.kanban-board {
  display: flex;
  gap: 10px;
  overflow-x: auto;
  padding-bottom: 12px;
}
.kanban-column {
  min-width: 200px;
  max-width: 240px;
  flex: 1;
  display: flex;
  flex-direction: column;
}
.column-header {
  padding: 8px 10px;
  background: #f8f9fa;
  border-top: 3px solid #ccc;
  border-radius: 6px 6px 0 0;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.column-title { font-size: 12px; font-weight: 600; }
.column-count {
  background: #e0e0e0;
  color: #666;
  font-size: 11px;
  font-weight: 700;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
}
.column-body {
  flex: 1;
  background: #f8f9fa;
  border-radius: 0 0 6px 6px;
  padding: 6px;
  min-height: 100px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.column-empty {
  text-align: center;
  color: #bbb;
  font-size: 11px;
  padding: 20px 0;
  font-style: italic;
}

.kanban-card {
  background: white;
  border: 1px solid #e0e0e0;
  border-radius: 6px;
  padding: 10px;
  cursor: pointer;
  transition: all 0.15s;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.kanban-card:hover {
  border-color: var(--color-accent);
  box-shadow: 0 2px 6px rgba(0,0,0,0.1);
  transform: translateY(-1px);
}
.kanban-card[draggable="true"] { cursor: grab; }
.kanban-card[draggable="true"]:active { cursor: grabbing; }

.card-name { font-weight: 600; font-size: 13px; margin-bottom: 2px; }
.card-location { font-size: 11px; color: #888; margin-bottom: 4px; }
.card-meta {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 4px;
}
.card-price { font-size: 12px; font-weight: 600; color: #333; }
.card-type {
  font-size: 10px;
  background: #e3f2fd;
  color: #1565c0;
  padding: 1px 6px;
  border-radius: 8px;
}
.card-footer {
  display: flex;
  justify-content: space-between;
  font-size: 10px;
  color: #999;
}
.card-partner { font-style: italic; }
.card-props { color: #666; }

/* ===== TABLE ===== */
.table-wrap {
  border: 1px solid var(--color-border);
  border-radius: 8px;
  overflow: hidden;
}
.pipeline-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
.pipeline-table th {
  padding: 10px 14px;
  background: var(--color-accent);
  color: white;
  font-weight: 600;
  text-align: left;
}
.pipeline-table td {
  padding: 8px 14px;
  border-bottom: 1px solid var(--color-border);
}
.pipeline-table .r { text-align: right; }
.clickable-row { cursor: pointer; }
.clickable-row:hover { background: #f5f5f5; }
.deal-name { font-weight: 500; }
.empty-row {
  text-align: center;
  color: #999;
  font-style: italic;
  padding: 24px;
}
.stage-badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 12px;
  font-size: 11px;
  font-weight: 600;
  color: white;
}

/* ===== MODALS ===== */
.modal-overlay {
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(0,0,0,0.4);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}
.modal-content {
  background: white;
  border-radius: 10px;
  width: 600px;
  max-width: 95vw;
  max-height: 90vh;
  overflow-y: auto;
  box-shadow: 0 8px 30px rgba(0,0,0,0.2);
}
.modal-content.modal-sm { width: 420px; }
.modal-header {
  padding: 16px 20px;
  border-bottom: 1px solid var(--color-border);
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.modal-header h3 { margin: 0; font-size: 16px; }
.modal-close {
  background: none;
  border: none;
  font-size: 22px;
  cursor: pointer;
  color: #999;
  line-height: 1;
}
.modal-body { padding: 16px 20px; }
.modal-footer {
  padding: 12px 20px;
  border-top: 1px solid var(--color-border);
  display: flex;
  justify-content: flex-end;
  gap: 8px;
}

.form-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}
.form-grid .span-2 { grid-column: 1 / -1; }
.form-stack { display: flex; flex-direction: column; gap: 12px; }
.form-field label {
  display: block;
  font-size: 12px;
  font-weight: 600;
  color: #555;
  margin-bottom: 4px;
}
.form-field input,
.form-field select,
.form-field textarea {
  width: 100%;
  padding: 7px 10px;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  font-size: 13px;
  box-sizing: border-box;
}
.form-field textarea { resize: vertical; }

/* ===== DETAIL PANEL ===== */
.detail-overlay {
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(0,0,0,0.35);
  display: flex;
  justify-content: flex-end;
  z-index: 999;
}
.detail-panel {
  width: 720px;
  max-width: 95vw;
  height: 100vh;
  background: white;
  box-shadow: -4px 0 20px rgba(0,0,0,0.15);
  overflow-y: auto;
  padding: 20px 24px;
  display: flex;
  flex-direction: column;
}

.btn-back {
  background: none;
  border: none;
  color: var(--color-accent);
  cursor: pointer;
  font-size: 13px;
  padding: 0;
  margin-bottom: 8px;
}
.btn-back:hover { text-decoration: underline; }

.detail-title-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 4px;
}
.detail-title-row h2 { margin: 0; font-size: 20px; }
.stage-select {
  padding: 4px 10px;
  border: 2px solid;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
}
.detail-subtitle {
  font-size: 13px;
  color: #666;
  margin-bottom: 16px;
}

/* Info cards */
.info-cards {
  display: flex;
  gap: 10px;
  margin-bottom: 16px;
  flex-wrap: wrap;
}
.info-card {
  background: #f8f9fa;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  padding: 8px 14px;
  min-width: 100px;
}
.info-label { display: block; font-size: 10px; color: #888; text-transform: uppercase; font-weight: 600; }
.info-value { font-size: 15px; font-weight: 600; }

/* Tab bar */
.tab-bar {
  display: flex;
  border-bottom: 2px solid var(--color-border);
  margin-bottom: 14px;
}
.tab-bar button {
  padding: 8px 18px;
  border: none;
  background: none;
  font-size: 13px;
  cursor: pointer;
  color: #666;
  border-bottom: 2px solid transparent;
  margin-bottom: -2px;
}
.tab-bar button.active {
  color: var(--color-accent);
  border-bottom-color: var(--color-accent);
  font-weight: 600;
}

.tab-content { flex: 1; }
.tab-actions { margin-bottom: 12px; }
.empty-state {
  text-align: center;
  color: #999;
  font-style: italic;
  padding: 30px 0;
}

/* Property cards */
.property-grid {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.property-card {
  border: 1px solid var(--color-border);
  border-radius: 8px;
  padding: 12px 14px;
}
.prop-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 4px;
}
.prop-header h4 { margin: 0; font-size: 14px; }
.prop-actions { display: flex; gap: 4px; }
.btn-icon {
  background: none;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  width: 26px;
  height: 26px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  font-size: 14px;
  color: #666;
}
.btn-icon:hover { background: #f0f0f0; }
.btn-icon.btn-danger { color: #e53935; border-color: #ffcdd2; }
.btn-icon.btn-danger:hover { background: #ffebee; }
.btn-icon.btn-xs { width: 22px; height: 22px; font-size: 12px; }

.prop-address { font-size: 12px; color: #888; margin-bottom: 6px; }
.prop-details {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  font-size: 12px;
  color: #555;
  margin-bottom: 6px;
}
.prop-footer { margin-top: 6px; }
.btn-link {
  background: none;
  border: none;
  color: var(--color-accent);
  cursor: pointer;
  font-size: 12px;
  padding: 0;
  text-decoration: underline;
}

/* Entity cards */
.entity-card {
  border: 1px solid var(--color-border);
  border-radius: 8px;
  padding: 12px 14px;
  margin-bottom: 10px;
}
.entity-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 8px;
}
.entity-header h4 { margin: 0 8px 0 0; font-size: 14px; display: inline; }
.entity-type {
  font-size: 10px;
  background: #e8eaf6;
  color: #3949ab;
  padding: 1px 8px;
  border-radius: 8px;
  margin-right: 6px;
}
.entity-id {
  font-size: 10px;
  background: #f5f5f5;
  color: #666;
  padding: 1px 8px;
  border-radius: 8px;
  margin-right: 6px;
  font-family: monospace;
}
.entity-role {
  font-size: 10px;
  color: #888;
  margin-right: 6px;
}
.entity-pct {
  font-size: 12px;
  font-weight: 600;
  color: #333;
}
.entity-empty {
  font-size: 12px;
  color: #bbb;
  font-style: italic;
}

.investor-table table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}
.investor-table th {
  padding: 5px 8px;
  text-align: left;
  font-weight: 600;
  color: #888;
  font-size: 11px;
  border-bottom: 1px solid var(--color-border);
}
.investor-table td {
  padding: 5px 8px;
  border-bottom: 1px solid #f0f0f0;
}
.investor-table .r { text-align: right; }

/* Activity */
.note-input {
  display: flex;
  gap: 8px;
  margin-bottom: 14px;
}
.note-input input {
  flex: 1;
  padding: 7px 10px;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  font-size: 13px;
}
.activity-item {
  display: flex;
  gap: 10px;
  padding: 8px 0;
  border-bottom: 1px solid #f0f0f0;
}
.activity-icon {
  width: 24px;
  height: 24px;
  background: #f0f0f0;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  flex-shrink: 0;
}
.activity-body { flex: 1; }
.activity-note { font-size: 13px; display: block; }
.activity-meta { font-size: 11px; color: #999; }

/* Detail footer */
.detail-footer {
  margin-top: 24px;
  padding-top: 12px;
  border-top: 1px solid var(--color-border);
}
.btn-danger-text {
  background: none;
  border: none;
  color: #e53935;
  cursor: pointer;
  font-size: 12px;
  padding: 0;
}
.btn-danger-text:hover { text-decoration: underline; }
</style>
