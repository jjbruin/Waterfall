<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import api from '../api/client'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ProspectDeal {
  id: number
  vcode: string | null
  deal_name: string
  stage: string
  purchase_price: number | null
  partner_name: string
}

interface WfInvestor {
  id: string
  name: string
  pref_rate: number
  share_pct: number
  is_pe: boolean
}

interface WfStep {
  vcode: string
  vmisc: string
  iOrder: number
  PropCode: string
  vState: string
  FXRate: number
  nPercent: number
  mAmount: number
  vtranstype: string
  vAmtType: string
  vNotes: string
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const deals = ref<ProspectDeal[]>([])
const selectedDealId = ref<number | null>(null)
const dealDetail = ref<any>(null)
const loading = ref(false)
const error = ref('')

// Assumptions form
const assumptionVersions = ref<any[]>([])
const selectedAssumptionId = ref<number | null>(null)
const acqForm = ref({
  purchase_price: null as number | null,
  closing_cost_pct: 0.02,
  capex_at_close: 0,
})
const assumptionForm = ref({
  version_label: 'Base Case',
  debt_amount: null as number | null,
  debt_rate: 0.05,
  debt_term_months: 84,
  io_months: 60,
  amort_months: 360,
  psc_equity_pct: 0.90,
  pref_rate: 0.08,
  promote_pct: 0.20,
  exit_cap_rate: 0.06,
  selling_cost_pct: 0.02,
  hold_years: 7,
  capex_reserve_psf: 0.80,
  noi_year1: null as number | null,
  noi_growth_rate: 0.02,
})
const savingAssumptions = ref(false)

// Analysis results
const analysisLoading = ref(false)
const analysisResult = ref<any>(null)
const analysisError = ref('')

// Waterfall builder
const wfInvestors = ref<WfInvestor[]>([])
const wfPromote = ref({ enabled: false, pct: 0.20 })
const wfSteps = ref<WfStep[]>([])
const wfHasStored = ref(false)
const wfSaving = ref(false)
const wfTab = ref<'builder' | 'steps'>('builder')

// Expandable sections
const expanded = ref<Record<string, boolean>>({})

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

async function loadDeals() {
  loading.value = true
  try {
    const res = await api.get('/api/prospects')
    deals.value = (res.data || []).filter((d: any) => d.stage !== 'passed')
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    loading.value = false
  }
}

async function selectDeal(id: number) {
  selectedDealId.value = id
  analysisResult.value = null
  analysisError.value = ''
  expanded.value = {}

  try {
    const [detailRes, assumRes, wfRes] = await Promise.all([
      api.get(`/api/prospects/${id}`),
      api.get(`/api/prospects/${id}/assumptions`),
      api.get(`/api/prospects/${id}/waterfall`),
    ])
    dealDetail.value = detailRes.data
    assumptionVersions.value = assumRes.data || []
    wfSteps.value = wfRes.data.steps || []
    wfHasStored.value = wfRes.data.has_cf || wfRes.data.has_cap

    // Pre-fill acquisition from deal
    const d = detailRes.data.deal || {}
    acqForm.value.purchase_price = d.purchase_price
    acqForm.value.closing_cost_pct = d.closing_cost_pct || 0.02
    acqForm.value.capex_at_close = d.capex_at_close || 0

    // Pre-fill investors from entities
    const entities = detailRes.data.entities || []
    if (!wfHasStored.value) {
      initInvestorsFromEntities(entities)
    } else {
      loadInvestorsFromSteps()
    }

    // Load first assumption version if exists
    if (assumptionVersions.value.length) {
      selectAssumption(assumptionVersions.value[0])
    }
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  }
}

function initInvestorsFromEntities(entities: any[]) {
  wfInvestors.value = []
  for (const ent of entities) {
    const id = ent.planned_entity_id || ent.entity_name?.toUpperCase().replace(/\s+/g, '_') || `ENT_${ent.id}`
    const isPe = (ent.role || '').toLowerCase().includes('pe') ||
                 (ent.role || '').toLowerCase().includes('pref') ||
                 (ent.entity_type || '').toLowerCase().includes('pe')
    wfInvestors.value.push({
      id,
      name: ent.entity_name,
      pref_rate: isPe ? (assumptionForm.value.pref_rate || 0.08) : 0,
      share_pct: ent.ownership_pct || (isPe ? 0.90 : 0.10),
      is_pe: isPe,
    })
  }
  // Default 2-partner if no entities
  if (!wfInvestors.value.length) {
    wfInvestors.value = [
      { id: 'PSC_PE', name: 'PSC Preferred Equity', pref_rate: 0.08, share_pct: 0.90, is_pe: true },
      { id: 'OP_PARTNER', name: 'Operating Partner', pref_rate: 0, share_pct: 0.10, is_pe: false },
    ]
  }
}

function loadInvestorsFromSteps() {
  // Derive investor list from stored waterfall steps
  const seen = new Map<string, WfInvestor>()
  for (const s of wfSteps.value) {
    if (!s.PropCode || seen.has(s.PropCode)) continue
    const hasPref = wfSteps.value.some(x => x.PropCode === s.PropCode && x.vState === 'Pref')
    const prefStep = wfSteps.value.find(x => x.PropCode === s.PropCode && x.vState === 'Pref')
    const shareStep = wfSteps.value.find(x =>
      x.PropCode === s.PropCode && (x.vState === 'Share' || x.vState === 'Tag') && x.vmisc === 'CF_WF'
    )
    seen.set(s.PropCode, {
      id: s.PropCode,
      name: s.PropCode,
      pref_rate: prefStep ? (prefStep.nPercent > 1 ? prefStep.nPercent / 100 : prefStep.nPercent) : 0,
      share_pct: shareStep ? shareStep.FXRate : 0,
      is_pe: hasPref,
    })
  }
  wfInvestors.value = Array.from(seen.values())
}

function addInvestor() {
  const n = wfInvestors.value.length + 1
  wfInvestors.value.push({
    id: `INV_${n}`,
    name: `Investor ${n}`,
    pref_rate: 0,
    share_pct: 0,
    is_pe: false,
  })
}

function removeInvestor(idx: number) {
  wfInvestors.value.splice(idx, 1)
}

// ---------------------------------------------------------------------------
// Assumptions
// ---------------------------------------------------------------------------

function selectAssumption(a: any) {
  if (!a) return
  selectedAssumptionId.value = a.id
  for (const key of Object.keys(assumptionForm.value)) {
    if (a[key] != null) (assumptionForm.value as any)[key] = a[key]
  }
}

async function saveAssumptions() {
  if (!selectedDealId.value) return
  savingAssumptions.value = true
  try {
    const payload = {
      ...(selectedAssumptionId.value ? { id: selectedAssumptionId.value } : {}),
      ...assumptionForm.value,
    }
    const res = await api.post(`/api/prospects/${selectedDealId.value}/assumptions`, payload)
    const { data: versions } = await api.get(`/api/prospects/${selectedDealId.value}/assumptions`)
    assumptionVersions.value = versions || []
    selectedAssumptionId.value = res.data.id
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    savingAssumptions.value = false
  }
}

// ---------------------------------------------------------------------------
// Waterfall build + save
// ---------------------------------------------------------------------------

async function buildAndSaveWaterfall() {
  if (!selectedDealId.value || !wfInvestors.value.length) return
  wfSaving.value = true
  try {
    const res = await api.post(`/api/prospects/${selectedDealId.value}/waterfall/build`, {
      investors: wfInvestors.value,
      promote: wfPromote.value,
    })
    wfSteps.value = res.data.steps || []
    wfHasStored.value = true
    wfTab.value = 'steps'
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    wfSaving.value = false
  }
}

async function deleteWaterfall() {
  if (!selectedDealId.value) return
  try {
    await api.delete(`/api/prospects/${selectedDealId.value}/waterfall`)
    wfSteps.value = []
    wfHasStored.value = false
    wfTab.value = 'builder'
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  }
}

// ---------------------------------------------------------------------------
// Run Analysis
// ---------------------------------------------------------------------------

async function runAnalysis() {
  if (!selectedDealId.value) return
  analysisLoading.value = true
  analysisError.value = ''
  analysisResult.value = null

  try {
    const payload: any = {
      ...assumptionForm.value,
      purchase_price_override: acqForm.value.purchase_price,
      closing_cost_pct_override: acqForm.value.closing_cost_pct,
      capex_at_close_override: acqForm.value.capex_at_close,
    }
    if (selectedAssumptionId.value) {
      payload.assumption_id = selectedAssumptionId.value
    }
    const res = await api.post(`/api/prospects/${selectedDealId.value}/analyze`, payload)
    analysisResult.value = res.data
  } catch (e: any) {
    analysisError.value = e.response?.data?.error || e.message
  } finally {
    analysisLoading.value = false
  }
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

function fmtCurrency(v: any): string {
  if (v == null || v === '' || isNaN(v)) return '—'
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(v)
}
function fmtPct(v: any): string {
  if (v == null || v === '' || isNaN(v)) return '—'
  return (Number(v) * 100).toFixed(2) + '%'
}
function fmtNum(v: any): string {
  if (v == null || v === '' || isNaN(v)) return '—'
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(v)
}
function fmtDec(v: any, d = 2): string {
  if (v == null || v === '' || isNaN(v)) return '—'
  return Number(v).toFixed(d)
}
function fmtVal(v: any, isPct: boolean): string {
  if (v == null || v === '' || (typeof v === 'number' && isNaN(v))) return ''
  if (isPct) return fmtDec(v, 2)
  return fmtNum(v)
}

// Computed
const deal = computed(() => dealDetail.value?.deal || null)
const properties = computed(() => dealDetail.value?.properties || [])
const entities = computed(() => dealDetail.value?.entities || [])

const cfSteps = computed(() => wfSteps.value.filter(s => s.vmisc === 'CF_WF'))
const capSteps = computed(() => wfSteps.value.filter(s => s.vmisc === 'Cap_WF'))

const sharePctTotal = computed(() =>
  wfInvestors.value.reduce((sum, inv) => sum + (inv.share_pct || 0), 0)
)

// Init
loadDeals()
</script>

<template>
  <div class="prospect-analysis">
    <!-- ============ DEAL SELECTOR ============ -->
    <div class="selector-bar">
      <label>Select Deal:</label>
      <select @change="e => { const id = Number((e.target as HTMLSelectElement).value); if (id) selectDeal(id) }">
        <option value="">— Select a prospect deal —</option>
        <option v-for="d in deals" :key="d.id" :value="d.id" :selected="d.id === selectedDealId">
          {{ d.deal_name }} {{ d.stage ? `(${d.stage})` : '' }}
        </option>
      </select>
      <span v-if="assumptionVersions.length" class="version-selector">
        <label>Scenario:</label>
        <select @change="e => selectAssumption(assumptionVersions.find((a: any) => a.id === Number((e.target as HTMLSelectElement).value)))">
          <option v-for="v in assumptionVersions" :key="v.id" :value="v.id" :selected="v.id === selectedAssumptionId">
            {{ v.version_label }} (v{{ v.version }})
          </option>
        </select>
      </span>
    </div>

    <div v-if="error" class="error-msg">{{ error }}
      <button @click="error = ''" class="dismiss">&times;</button>
    </div>

    <div v-if="!selectedDealId" class="empty-state">
      Select a prospect deal to begin analysis.
    </div>

    <!-- ============ MAIN CONTENT ============ -->
    <template v-if="deal">
      <div class="analysis-layout">
        <!-- LEFT: Setup Panel -->
        <div class="setup-panel">

          <!-- Deal Info -->
          <div class="section">
            <div class="section-header">Deal Information</div>
            <div class="info-grid">
              <div><strong>Deal:</strong> {{ deal.deal_name }}</div>
              <div><strong>Stage:</strong> {{ deal.stage }}</div>
              <div><strong>Partner:</strong> {{ deal.partner_name || '—' }}</div>
              <div><strong>Properties:</strong> {{ properties.length }}</div>
              <div v-if="deal.location"><strong>Location:</strong> {{ deal.location }}</div>
              <div v-if="deal.asset_type"><strong>Type:</strong> {{ deal.asset_type }}</div>
            </div>
          </div>

          <!-- Acquisition -->
          <div class="section">
            <div class="section-header">Acquisition</div>
            <div class="form-grid-3">
              <div class="form-group">
                <label>Purchase Price ($)</label>
                <input type="number" v-model.number="acqForm.purchase_price" step="10000" />
              </div>
              <div class="form-group">
                <label>Closing Cost %</label>
                <input type="number" v-model.number="acqForm.closing_cost_pct" step="0.005" />
              </div>
              <div class="form-group">
                <label>Reserves / CapEx ($)</label>
                <input type="number" v-model.number="acqForm.capex_at_close" step="10000" />
              </div>
            </div>
          </div>

          <!-- Operating Assumptions -->
          <div class="section">
            <div class="section-header">Operating & Exit Assumptions</div>
            <div class="form-grid-3">
              <div class="form-group">
                <label>Year 1 NOI ($)</label>
                <input type="number" v-model.number="assumptionForm.noi_year1" step="1000" />
              </div>
              <div class="form-group">
                <label>NOI Growth Rate</label>
                <input type="number" v-model.number="assumptionForm.noi_growth_rate" step="0.005" />
              </div>
              <div class="form-group">
                <label>Hold Period (years)</label>
                <input type="number" v-model.number="assumptionForm.hold_years" />
              </div>
              <div class="form-group">
                <label>Exit Cap Rate</label>
                <input type="number" v-model.number="assumptionForm.exit_cap_rate" step="0.0025" />
              </div>
              <div class="form-group">
                <label>Selling Cost %</label>
                <input type="number" v-model.number="assumptionForm.selling_cost_pct" step="0.005" />
              </div>
              <div class="form-group">
                <label>CapEx Reserve ($/SF)</label>
                <input type="number" v-model.number="assumptionForm.capex_reserve_psf" step="0.10" />
              </div>
            </div>
          </div>

          <!-- Debt -->
          <div class="section">
            <div class="section-header">Debt</div>
            <div class="form-grid-3">
              <div class="form-group">
                <label>Loan Amount ($)</label>
                <input type="number" v-model.number="assumptionForm.debt_amount" step="10000" />
              </div>
              <div class="form-group">
                <label>Interest Rate</label>
                <input type="number" v-model.number="assumptionForm.debt_rate" step="0.0025" />
              </div>
              <div class="form-group">
                <label>Term (months)</label>
                <input type="number" v-model.number="assumptionForm.debt_term_months" />
              </div>
              <div class="form-group">
                <label>IO Period (months)</label>
                <input type="number" v-model.number="assumptionForm.io_months" />
              </div>
              <div class="form-group">
                <label>Amort Period (months)</label>
                <input type="number" v-model.number="assumptionForm.amort_months" />
              </div>
            </div>
          </div>

          <!-- Waterfall Builder -->
          <div class="section">
            <div class="section-header">
              Waterfall Structure
              <span v-if="wfHasStored" class="badge-saved">Saved</span>
            </div>

            <div class="wf-tabs">
              <button :class="{ active: wfTab === 'builder' }" @click="wfTab = 'builder'">Builder</button>
              <button :class="{ active: wfTab === 'steps' }" @click="wfTab = 'steps'"
                      :disabled="!wfSteps.length">Steps ({{ wfSteps.length }})</button>
            </div>

            <!-- Builder tab -->
            <div v-if="wfTab === 'builder'" class="wf-builder">
              <div class="wf-investor-header">
                <strong>Investors</strong>
                <button class="btn-sm btn-secondary" @click="addInvestor">+ Add</button>
              </div>

              <div v-for="(inv, i) in wfInvestors" :key="i" class="wf-investor-row">
                <div class="form-group">
                  <label>Entity ID</label>
                  <input v-model="inv.id" placeholder="ENTITY_ID" />
                </div>
                <div class="form-group">
                  <label>Name</label>
                  <input v-model="inv.name" />
                </div>
                <div class="form-group">
                  <label>Pref Rate</label>
                  <input type="number" v-model.number="inv.pref_rate" step="0.005" />
                </div>
                <div class="form-group">
                  <label>Residual %</label>
                  <input type="number" v-model.number="inv.share_pct" step="0.05" min="0" max="1" />
                </div>
                <div class="form-group wf-pe-check">
                  <label><input type="checkbox" v-model="inv.is_pe" /> PE</label>
                </div>
                <button class="btn-icon btn-danger btn-xs" @click="removeInvestor(i)"
                        v-if="wfInvestors.length > 1">&times;</button>
              </div>

              <div v-if="Math.abs(sharePctTotal - 1.0) > 0.02" class="wf-warning">
                Residual shares sum to {{ fmtPct(sharePctTotal) }} (should be ~100%)
              </div>

              <div class="wf-actions">
                <button class="btn-primary" @click="buildAndSaveWaterfall"
                        :disabled="wfSaving || !wfInvestors.length">
                  {{ wfSaving ? 'Saving...' : (wfHasStored ? 'Rebuild & Save Waterfall' : 'Build & Save Waterfall') }}
                </button>
                <button v-if="wfHasStored" class="btn-danger-text" @click="deleteWaterfall">
                  Delete Waterfall
                </button>
              </div>
            </div>

            <!-- Steps tab (preview of stored waterfall) -->
            <div v-if="wfTab === 'steps' && wfSteps.length" class="wf-steps-view">
              <div v-for="wfType in ['CF_WF', 'Cap_WF']" :key="wfType" class="wf-type-section">
                <h5>{{ wfType === 'CF_WF' ? 'Cash Flow Waterfall' : 'Capital Waterfall' }}</h5>
                <table class="compact-table">
                  <thead>
                    <tr>
                      <th>Order</th>
                      <th>Investor</th>
                      <th>Type</th>
                      <th class="r">FXRate</th>
                      <th class="r">Rate</th>
                      <th>Pool</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="s in wfSteps.filter(x => x.vmisc === wfType)" :key="`${s.iOrder}-${s.PropCode}`">
                      <td>{{ s.iOrder }}</td>
                      <td>{{ s.PropCode }}</td>
                      <td>{{ s.vState }}</td>
                      <td class="r">{{ s.FXRate?.toFixed(2) }}</td>
                      <td class="r">{{ s.nPercent ? (s.nPercent > 1 ? s.nPercent.toFixed(1) + '%' : fmtPct(s.nPercent)) : '—' }}</td>
                      <td>{{ s.vtranstype || '—' }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <!-- Action buttons -->
          <div class="action-bar">
            <button class="btn-primary btn-lg" @click="runAnalysis"
                    :disabled="analysisLoading || !assumptionForm.noi_year1">
              {{ analysisLoading ? 'Computing...' : 'Compute Returns' }}
            </button>
            <button class="btn-secondary" @click="saveAssumptions" :disabled="savingAssumptions">
              {{ savingAssumptions ? 'Saving...' : 'Save Assumptions' }}
            </button>
          </div>
        </div>

        <!-- RIGHT: Results Panel -->
        <div class="results-panel">
          <div v-if="analysisLoading" class="computing-overlay">
            <div class="spinner"></div>
            <span>Running analysis...</span>
          </div>

          <div v-if="analysisError" class="error-msg">{{ analysisError }}</div>

          <template v-if="analysisResult">
            <!-- Sources & Uses -->
            <div class="section">
              <div class="section-header">Sources & Uses</div>
              <div class="su-grid">
                <div class="su-col">
                  <h5>Uses</h5>
                  <div class="su-row"><span>Purchase Price</span><span class="r">{{ fmtCurrency(analysisResult.prospect_assumptions?.purchase_price) }}</span></div>
                  <div class="su-row"><span>Closing Costs</span><span class="r">{{ fmtCurrency(analysisResult.prospect_assumptions?.closing_costs) }}</span></div>
                  <div class="su-row"><span>Reserves / CapEx</span><span class="r">{{ fmtCurrency(analysisResult.prospect_assumptions?.capex_at_close) }}</span></div>
                  <div class="su-row su-total"><span>Total Cost</span><span class="r">{{ fmtCurrency(analysisResult.prospect_assumptions?.total_cost) }}</span></div>
                </div>
                <div class="su-col">
                  <h5>Sources</h5>
                  <div class="su-row"><span>Debt</span><span class="r">{{ fmtCurrency(analysisResult.prospect_assumptions?.debt_amount) }} ({{ fmtPct(analysisResult.prospect_assumptions?.ltv) }} LTV)</span></div>
                  <div class="su-row"><span>Pref. Equity</span><span class="r">{{ fmtCurrency(analysisResult.prospect_assumptions?.pe_equity) }}</span></div>
                  <div class="su-row"><span>OP Equity</span><span class="r">{{ fmtCurrency(analysisResult.prospect_assumptions?.op_equity) }}</span></div>
                  <div class="su-row su-total"><span>Total Sources</span><span class="r">{{ fmtCurrency(analysisResult.prospect_assumptions?.total_cost) }}</span></div>
                </div>
              </div>
            </div>

            <!-- Deal Summary KPIs -->
            <div class="section" v-if="analysisResult.deal_summary">
              <div class="section-header">Deal-Level Summary</div>
              <div class="metrics-row">
                <div class="metric-card" v-if="analysisResult.deal_summary.irr != null">
                  <div class="metric-label">Deal IRR</div>
                  <div class="metric-value">{{ fmtPct(analysisResult.deal_summary.irr) }}</div>
                </div>
                <div class="metric-card" v-if="analysisResult.deal_summary.roe != null">
                  <div class="metric-label">Deal ROE</div>
                  <div class="metric-value">{{ fmtPct(analysisResult.deal_summary.roe) }}</div>
                </div>
                <div class="metric-card" v-if="analysisResult.deal_summary.moic != null">
                  <div class="metric-label">Deal MOIC</div>
                  <div class="metric-value">{{ fmtDec(analysisResult.deal_summary.moic, 2) }}x</div>
                </div>
                <div class="metric-card" v-if="analysisResult.prospect_assumptions?.exit_value">
                  <div class="metric-label">Exit Value</div>
                  <div class="metric-value">{{ fmtCurrency(analysisResult.prospect_assumptions.exit_value) }}</div>
                </div>
                <div class="metric-card" v-if="analysisResult.prospect_assumptions?.terminal_noi">
                  <div class="metric-label">Terminal NOI</div>
                  <div class="metric-value">{{ fmtCurrency(analysisResult.prospect_assumptions.terminal_noi) }}</div>
                </div>
              </div>
            </div>

            <!-- Partner Returns -->
            <div class="section" v-if="analysisResult.partner_results?.length">
              <div class="section-header">Partner Returns</div>
              <div class="table-scroll">
                <table class="results-table">
                  <thead>
                    <tr>
                      <th>Partner</th>
                      <th class="r">Contributions</th>
                      <th class="r">CF Distributions</th>
                      <th class="r">Capital Distributions</th>
                      <th class="r">Total Distributions</th>
                      <th class="r">IRR</th>
                      <th class="r">ROE</th>
                      <th class="r">MOIC</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="p in analysisResult.partner_results" :key="p.investor_id"
                        :class="{ 'pe-row': p.is_pe }">
                      <td class="partner-name">{{ p.investor_id }}</td>
                      <td class="r">{{ fmtCurrency(p.contributions) }}</td>
                      <td class="r">{{ fmtCurrency(p.cf_distributions) }}</td>
                      <td class="r">{{ fmtCurrency(p.cap_distributions) }}</td>
                      <td class="r">{{ fmtCurrency(p.total_distributions) }}</td>
                      <td class="r">{{ p.irr != null ? fmtPct(p.irr) : '—' }}</td>
                      <td class="r">{{ p.roe != null ? fmtPct(p.roe) : '—' }}</td>
                      <td class="r">{{ p.moic != null ? fmtDec(p.moic, 2) + 'x' : '—' }}</td>
                    </tr>
                    <tr v-if="analysisResult.deal_summary" class="deal-total-row">
                      <td><strong>Deal Total</strong></td>
                      <td class="r"><strong>{{ fmtCurrency(analysisResult.deal_summary.contributions) }}</strong></td>
                      <td class="r"><strong>{{ fmtCurrency(analysisResult.deal_summary.cf_distributions) }}</strong></td>
                      <td class="r"><strong>{{ fmtCurrency(analysisResult.deal_summary.cap_distributions) }}</strong></td>
                      <td class="r"><strong>{{ fmtCurrency(analysisResult.deal_summary.total_distributions) }}</strong></td>
                      <td class="r"><strong>{{ analysisResult.deal_summary.irr != null ? fmtPct(analysisResult.deal_summary.irr) : '—' }}</strong></td>
                      <td class="r"><strong>{{ analysisResult.deal_summary.roe != null ? fmtPct(analysisResult.deal_summary.roe) : '—' }}</strong></td>
                      <td class="r"><strong>{{ analysisResult.deal_summary.moic != null ? fmtDec(analysisResult.deal_summary.moic, 2) + 'x' : '—' }}</strong></td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <!-- Annual Forecast -->
            <div class="section expandable" v-if="analysisResult.annual_forecast">
              <div class="section-header" @click="expanded.forecast = !expanded.forecast">
                Annual Forecast
                <span class="chevron">{{ expanded.forecast ? '\u25BE' : '\u25B8' }}</span>
              </div>
              <div v-if="expanded.forecast" class="table-scroll">
                <table class="forecast-table">
                  <thead>
                    <tr>
                      <th class="row-label">Account</th>
                      <th v-for="yr in analysisResult.annual_forecast.years" :key="yr" class="r">{{ yr }}</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="row in analysisResult.annual_forecast.rows" :key="row.label"
                        :class="{
                          'section-header-row': row.is_header,
                          'underline-row': row.underline,
                          'topline-row': row.topline,
                        }">
                      <td class="row-label">{{ row.label }}</td>
                      <td v-for="yr in analysisResult.annual_forecast.years" :key="yr" class="r">
                        {{ fmtVal(row.values?.[yr], row.is_pct) }}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <!-- Debt Service -->
            <div class="section expandable" v-if="analysisResult.debt_service?.length">
              <div class="section-header" @click="expanded.debt = !expanded.debt">
                Debt Service
                <span class="chevron">{{ expanded.debt ? '\u25BE' : '\u25B8' }}</span>
              </div>
              <div v-if="expanded.debt">
                <table class="compact-table">
                  <thead>
                    <tr>
                      <th>Year</th>
                      <th class="r">Interest</th>
                      <th class="r">Principal</th>
                      <th class="r">Total DS</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="ds in analysisResult.debt_service" :key="ds.Year">
                      <td>{{ ds.Year }}</td>
                      <td class="r">{{ fmtCurrency(ds.interest) }}</td>
                      <td class="r">{{ fmtCurrency(ds.principal) }}</td>
                      <td class="r">{{ fmtCurrency(ds.total) }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <!-- Diagnostics -->
            <details v-if="analysisResult.debug_msgs?.length" class="debug-section">
              <summary>Diagnostics ({{ analysisResult.debug_msgs.length }})</summary>
              <ul class="debug-list">
                <li v-for="(msg, i) in analysisResult.debug_msgs" :key="i">{{ msg }}</li>
              </ul>
            </details>

          </template>

          <div v-if="!analysisResult && !analysisLoading && !analysisError" class="empty-results">
            Configure assumptions and click "Compute Returns" to see results.
          </div>
        </div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.prospect-analysis { padding: 0; height: 100%; display: flex; flex-direction: column; }

/* Selector bar */
.selector-bar {
  display: flex; align-items: center; gap: 12px;
  padding: 12px 20px; background: #f0f4f8;
  border-bottom: 1px solid #dee2e6;
  flex-shrink: 0;
}
.selector-bar label { font-weight: 600; font-size: 13px; white-space: nowrap; }
.selector-bar select { padding: 6px 10px; border: 1px solid #ccc; border-radius: 4px; font-size: 13px; min-width: 240px; }
.version-selector { display: flex; align-items: center; gap: 6px; margin-left: 16px; }

.empty-state { padding: 40px; text-align: center; color: #999; }

.error-msg {
  background: #fbe9e7; color: #d32f2f; padding: 8px 16px;
  margin: 8px 16px; border-radius: 4px; font-size: 13px;
  display: flex; align-items: center; gap: 8px;
}
.dismiss { background: none; border: none; cursor: pointer; font-size: 16px; color: #d32f2f; }

/* Layout */
.analysis-layout {
  display: flex; flex: 1; overflow: hidden;
}
.setup-panel {
  width: 420px; min-width: 380px; max-width: 480px;
  overflow-y: auto; padding: 12px 16px;
  border-right: 1px solid #dee2e6;
  background: #fafbfc;
  flex-shrink: 0;
}
.results-panel {
  flex: 1; overflow-y: auto; padding: 12px 20px;
  position: relative;
}

/* Sections */
.section {
  background: #fff; border: 1px solid #e0e0e0;
  border-radius: 6px; padding: 12px 14px;
  margin-bottom: 10px;
}
.section-header {
  font-weight: 700; font-size: 13px; color: #333;
  margin-bottom: 8px; display: flex; align-items: center; gap: 8px;
}
.expandable .section-header { cursor: pointer; user-select: none; }
.chevron { font-size: 11px; color: #999; }

/* Info grid */
.info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 12px; font-size: 12px; }

/* Form grids */
.form-grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; }
.form-group label { display: block; font-size: 11px; font-weight: 500; color: #666; margin-bottom: 2px; }
.form-group input, .form-group select {
  width: 100%; padding: 5px 8px; border: 1px solid #ccc;
  border-radius: 4px; font-size: 12px;
}

/* Waterfall builder */
.wf-tabs { display: flex; gap: 4px; margin-bottom: 8px; }
.wf-tabs button {
  padding: 4px 12px; border: 1px solid #ccc; border-radius: 4px;
  background: #f5f5f5; font-size: 12px; cursor: pointer;
}
.wf-tabs button.active { background: #1976d2; color: #fff; border-color: #1976d2; }
.wf-tabs button:disabled { opacity: 0.4; cursor: default; }

.wf-investor-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; font-size: 12px; }
.wf-investor-row {
  display: flex; gap: 6px; align-items: flex-end;
  padding: 6px 0; border-bottom: 1px solid #f0f0f0;
}
.wf-investor-row .form-group { flex: 1; min-width: 0; }
.wf-investor-row .form-group input { font-size: 11px; padding: 4px 6px; }
.wf-investor-row .form-group label { font-size: 10px; }
.wf-pe-check { flex: 0 0 40px !important; }
.wf-pe-check label { font-size: 11px; display: flex; align-items: center; gap: 3px; }

.wf-warning {
  color: #e65100; font-size: 11px; padding: 4px 8px;
  background: #fff3e0; border-radius: 3px; margin: 6px 0;
}
.wf-actions { display: flex; gap: 8px; margin-top: 8px; align-items: center; }

.wf-type-section { margin-bottom: 10px; }
.wf-type-section h5 { font-size: 12px; margin: 0 0 4px; color: #555; }

.badge-saved {
  font-size: 10px; font-weight: 600; color: #2e7d32;
  background: #e8f5e9; padding: 1px 6px; border-radius: 3px;
}

/* Action bar */
.action-bar {
  display: flex; gap: 8px; padding: 10px 0;
  border-top: 1px solid #e0e0e0; margin-top: 4px;
}

/* Computing overlay */
.computing-overlay {
  display: flex; align-items: center; gap: 10px;
  padding: 20px; color: #1976d2; font-size: 14px;
}
.spinner {
  width: 20px; height: 20px; border: 3px solid #e0e0e0;
  border-top-color: #1976d2; border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* Sources & Uses */
.su-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.su-col h5 { margin: 0 0 6px; font-size: 12px; color: #555; }
.su-row { display: flex; justify-content: space-between; font-size: 12px; padding: 2px 0; }
.su-total { border-top: 1px solid #333; font-weight: 700; margin-top: 4px; padding-top: 4px; }

/* Metrics */
.metrics-row { display: flex; flex-wrap: wrap; gap: 10px; }
.metric-card {
  background: #f0f4f8; border-radius: 6px; padding: 10px 16px;
  min-width: 120px; text-align: center;
}
.metric-label { font-size: 11px; color: #666; margin-bottom: 2px; }
.metric-value { font-size: 18px; font-weight: 700; color: #1a237e; }

/* Tables */
.table-scroll { overflow-x: auto; }
.results-table, .forecast-table, .compact-table {
  width: 100%; border-collapse: collapse; font-size: 12px;
  font-variant-numeric: tabular-nums;
}
.results-table th, .results-table td,
.forecast-table th, .forecast-table td,
.compact-table th, .compact-table td {
  padding: 5px 10px; border-bottom: 1px solid #eee; white-space: nowrap;
}
.results-table th, .forecast-table th, .compact-table th {
  background: #f0f4f8; font-weight: 600; position: sticky; top: 0;
}
.r { text-align: right; }
.row-label { white-space: nowrap; font-weight: 500; }
.pe-row { background: #e8eaf6; font-weight: 600; }
.deal-total-row td { border-top: 2px solid #333; }
.partner-name { font-weight: 500; }
.section-header-row td { font-weight: 700; background: #f5f5f5; }
.underline-row td { border-bottom: 2px solid #333; }
.topline-row td { border-top: 2px solid #333; }

.empty-results { padding: 40px; text-align: center; color: #bbb; font-size: 14px; }

/* Buttons */
.btn-primary {
  background: #1976d2; color: #fff; border: none; border-radius: 4px;
  padding: 7px 16px; font-size: 13px; font-weight: 600; cursor: pointer;
}
.btn-primary:hover { background: #1565c0; }
.btn-primary:disabled { opacity: 0.5; cursor: default; }
.btn-primary.btn-lg { padding: 9px 24px; font-size: 14px; }
.btn-secondary {
  background: #f5f5f5; color: #333; border: 1px solid #ccc; border-radius: 4px;
  padding: 7px 16px; font-size: 13px; cursor: pointer;
}
.btn-secondary:disabled { opacity: 0.5; cursor: default; }
.btn-sm { padding: 3px 10px; font-size: 11px; }
.btn-icon { background: none; border: none; cursor: pointer; font-size: 14px; padding: 2px 4px; }
.btn-danger { color: #d32f2f; }
.btn-xs { font-size: 12px; }
.btn-danger-text {
  background: none; border: none; color: #d32f2f;
  cursor: pointer; font-size: 12px; padding: 0;
}
.btn-danger-text:hover { text-decoration: underline; }

/* Debug */
.debug-section { margin-top: 12px; }
.debug-section summary { cursor: pointer; font-size: 12px; color: #666; }
.debug-list { font-size: 11px; color: #666; padding-left: 20px; }
</style>
