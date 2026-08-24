<script setup lang="ts">
import { ref, computed } from 'vue'
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

interface UseItem {
  id: string
  label: string
  amount: number | null
  pct: number | null         // percentage as whole number (e.g. 1.0 = 1%)
  pctBase: 'purchase_price' | 'total_debt' | null
  isFixed: boolean           // PSC Origination Fee — auto-calc, not editable
  removable: boolean
}

interface SourceItem {
  id: string
  label: string
  amount: number | null
  isDebt: boolean
  removable: boolean
  level: 'portfolio' | 'property'
  propertyId: number | null  // which property (if level === 'property')
}

interface WfStepInput {
  entity_id: string
  step_type: 'pref' | 'return_of_capital' | 'residual' | 'fixed_amount' | 'irr_lookback'
  rate: number | null
  amount: number | null
  wf_type?: 'CF_WF' | 'Cap_WF'  // used when sending to backend
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

const STEP_TYPES = [
  { value: 'pref', label: 'Preferred Return', inputLabel: 'Rate (%)', inputType: 'rate' },
  { value: 'return_of_capital', label: 'Return of Capital', inputLabel: null, inputType: null },
  { value: 'residual', label: 'Cash Flow Split', inputLabel: 'Share (%)', inputType: 'rate' },
  { value: 'fixed_amount', label: 'Fixed Amount', inputLabel: 'Per Quarter ($)', inputType: 'amount' },
  { value: 'irr_lookback', label: 'IRR Lookback', inputLabel: 'Target IRR (%)', inputType: 'rate' },
] as const

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const deals = ref<ProspectDeal[]>([])
const selectedDealId = ref<number | null>(null)
const dealDetail = ref<any>(null)
const loading = ref(false)
const error = ref('')

// Capital Budget — Uses
const capitalUses = ref<UseItem[]>([])
const expandedBudget = ref(true)

// Capital Budget — Sources (Debt)
const debtSources = ref<SourceItem[]>([])
const additionalEquitySources = ref<SourceItem[]>([])
const peEquityPct = ref<number>(90) // as whole number (90 = 90%)

// Property-level prices (for portfolio deals)
const propertyPrices = ref<Record<number, number | null>>({})

// Assumptions form (operating + exit)
const assumptionVersions = ref<any[]>([])
const selectedAssumptionId = ref<number | null>(null)
const assumptionForm = ref({
  version_label: 'Base Case',
  noi_year1: null as number | null,
  noi_growth_rate: 0.02,
  hold_years: 7,
  exit_cap_rate: 0.06,
  selling_cost_pct: 0.02,
  capex_reserve_psf: 0.80,
  // Loan terms (for first mortgage)
  lender: '' as string,
  debt_rate: 0.063,
  rate_type: 'fixed' as 'fixed' | 'floating',
  rate_index: 'UST' as string,
  rate_index_term: '5yr' as string,
  rate_spread_bps: 170 as number | null,
  rate_cushion_bps: 25 as number | null,
  debt_term_months: 60,
  io_months: 24,
  amort_months: 360,
  extension_count: null as number | null,
  extension_months: 12 as number | null,
  extension_conditions: '' as string,
  prepay_type: '' as string,
  prepay_schedule: '' as string, // e.g. "3%, 2%, 1%, 1%, 0%"
  // Sizing constraints
  max_ltv: null as number | null,
  max_ltc: null as number | null,
  min_dscr: null as number | null,
  dscr_test_start: '' as string,
  min_debt_yield: null as number | null,
  origination_fee_bps: null as number | null,
  // Earnout / future funding
  earnout_notes: '' as string,
  // Guarantor
  guarantor_notes: '' as string,
  // Equity split (kept for backward compat with assumptions table)
  psc_equity_pct: 0.90,
  pref_rate: 0.08,
  promote_pct: 0.20,
  // Loan proposal fields (future)
  debt_amount: null as number | null,
})
const savingAssumptions = ref(false)

// Analysis results
const analysisLoading = ref(false)
const analysisResult = ref<any>(null)
const analysisError = ref('')

// Waterfall builder — independent CF and Cap step lists
const cfStepInputs = ref<WfStepInput[]>([])
const capStepInputs = ref<WfStepInput[]>([])
const wfSteps = ref<WfStep[]>([])
const wfHasStored = ref(false)
const wfSaving = ref(false)
const wfTab = ref<'builder' | 'steps'>('builder')

// Expandable sections
const expanded = ref<Record<string, boolean>>({})

// ---------------------------------------------------------------------------
// Default line items
// ---------------------------------------------------------------------------

function defaultUses(): UseItem[] {
  return [
    { id: 'purchase_price', label: 'Purchase Price', amount: null, pct: null, pctBase: null, isFixed: false, removable: false },
    { id: 'sponsor_acq_fee', label: 'Sponsor Acquisition Fee', amount: null, pct: null, pctBase: 'purchase_price', isFixed: false, removable: false },
    { id: 'loan_fees', label: 'Loan Fees', amount: null, pct: null, pctBase: null, isFixed: false, removable: false },
    { id: 'lender_orig_fee', label: 'Lender Origination Fee', amount: null, pct: null, pctBase: 'total_debt', isFixed: false, removable: false },
    { id: 'debt_broker_fee', label: 'Debt Broker Fee', amount: null, pct: null, pctBase: 'total_debt', isFixed: false, removable: false },
    { id: 'sponsor_dd_legal', label: 'Sponsor Due Diligence / Legal', amount: null, pct: null, pctBase: null, isFixed: false, removable: false },
    { id: 'title', label: 'Title', amount: null, pct: null, pctBase: null, isFixed: false, removable: false },
    { id: 'sponsor_misc', label: 'Sponsor Misc Closing Costs', amount: null, pct: null, pctBase: null, isFixed: false, removable: false },
    { id: 'capex_reserve', label: 'Cap Ex Reserve', amount: null, pct: null, pctBase: null, isFixed: false, removable: false },
    { id: 'prepaid_expenses', label: 'Prepaid Expenses', amount: null, pct: null, pctBase: null, isFixed: false, removable: false },
    { id: 'psc_orig_fee', label: 'PSC Origination Fee', amount: null, pct: null, pctBase: null, isFixed: true, removable: false },
    { id: 'psc_dd_costs', label: 'PSC Due Diligence Costs', amount: null, pct: null, pctBase: null, isFixed: false, removable: false },
    { id: 'working_capital', label: 'Working Capital', amount: null, pct: null, pctBase: null, isFixed: false, removable: false },
  ]
}

function defaultDebtSources(): SourceItem[] {
  return [
    { id: 'first_mortgage', label: 'First Mortgage', amount: null, isDebt: true, removable: false, level: 'portfolio', propertyId: null },
    { id: 'future_mtg_fundings', label: 'Future Mortgage Fundings', amount: null, isDebt: true, removable: false, level: 'portfolio', propertyId: null },
    { id: 'second_mortgage', label: 'Second Mortgage', amount: null, isDebt: true, removable: false, level: 'portfolio', propertyId: null },
    { id: 'future_2nd_fundings', label: 'Future Second Mortgage Fundings', amount: null, isDebt: true, removable: false, level: 'portfolio', propertyId: null },
  ]
}

function defaultCfSteps(): WfStepInput[] {
  return [
    { entity_id: 'PSC_PE', step_type: 'pref', rate: 8.0, amount: null },
    { entity_id: 'PSC_PE', step_type: 'residual', rate: 90, amount: null },
    { entity_id: 'OP_PARTNER', step_type: 'residual', rate: 10, amount: null },
  ]
}

function defaultCapSteps(): WfStepInput[] {
  return [
    { entity_id: 'PSC_PE', step_type: 'pref', rate: 8.0, amount: null },
    { entity_id: 'PSC_PE', step_type: 'return_of_capital', rate: null, amount: null },
    { entity_id: 'OP_PARTNER', step_type: 'return_of_capital', rate: null, amount: null },
    { entity_id: 'PSC_PE', step_type: 'residual', rate: 90, amount: null },
    { entity_id: 'OP_PARTNER', step_type: 'residual', rate: 10, amount: null },
  ]
}

// ---------------------------------------------------------------------------
// Computed — Capital Budget
// ---------------------------------------------------------------------------

const deal = computed(() => dealDetail.value?.deal || null)
const properties = computed(() => dealDetail.value?.properties || [])
const entities = computed(() => dealDetail.value?.entities || [])

const purchasePrice = computed(() => {
  const pp = capitalUses.value.find(u => u.id === 'purchase_price')
  return pp?.amount || 0
})

const totalGla = computed(() =>
  properties.value.reduce((sum: number, p: any) => sum + (p.gla_sf || 0), 0)
)

const totalNoi = computed(() => {
  // Sum noi_in_place from properties, fall back to assumptionForm.noi_year1
  const propNoi = properties.value.reduce((sum: number, p: any) => sum + (p.noi_in_place || 0), 0)
  return propNoi || assumptionForm.value.noi_year1 || 0
})

const capRateDisplay = computed(() => {
  if (!purchasePrice.value || !totalNoi.value) return 'n/a'
  return ((totalNoi.value / purchasePrice.value) * 100).toFixed(1) + '%'
})

const psfDisplay = computed(() => {
  if (!purchasePrice.value || !totalGla.value) return 'n/a'
  return '$' + Math.round(purchasePrice.value / totalGla.value).toLocaleString()
})

const totalDebt = computed(() =>
  debtSources.value.reduce((sum, s) => sum + (s.amount || 0), 0)
)

/** Get effective amount for a use item (respecting pct-based auto-calc) */
function getUseAmount(item: UseItem): number {
  if (item.isFixed) return pscOrigFee.value
  if (item.pctBase && item.pct != null && item.pct !== 0) {
    const pctDec = item.pct / 100
    if (item.pctBase === 'purchase_price') return pctDec * purchasePrice.value
    if (item.pctBase === 'total_debt') return pctDec * totalDebt.value
  }
  return item.amount || 0
}

/** Sum of all uses EXCEPT the PSC Origination Fee */
const baseUsesTotal = computed(() => {
  let sum = 0
  for (const item of capitalUses.value) {
    if (item.isFixed) continue  // skip PSC Orig Fee
    sum += getUseAmount(item)
  }
  return sum
})

/**
 * PSC Origination Fee = 1% of PE commitment, solved without circular loop.
 * fee = origPct * pePct * (baseUses - totalDebt) / (1 - origPct * pePct)
 */
const pscOrigFee = computed(() => {
  const origPct = 0.01  // always 1%
  const pePct = peEquityPct.value / 100
  const denom = 1 - origPct * pePct
  if (denom <= 0 || baseUsesTotal.value <= totalDebt.value) return 0
  return (origPct * pePct * (baseUsesTotal.value - totalDebt.value)) / denom
})

const totalUses = computed(() => baseUsesTotal.value + pscOrigFee.value)

const totalEquity = computed(() => Math.max(0, totalUses.value - totalDebt.value))

const peAmount = computed(() => (peEquityPct.value / 100) * totalEquity.value)

const partnerEquity = computed(() => totalEquity.value - peAmount.value)

const partnerEquityPct = computed(() =>
  totalEquity.value > 0 ? ((partnerEquity.value / totalEquity.value) * 100) : 0
)

const totalSources = computed(() =>
  totalDebt.value + peAmount.value + partnerEquity.value +
  additionalEquitySources.value.reduce((s, e) => s + (e.amount || 0), 0)
)

/** Computed all-in rate from index spread + cushion (for display) */
const computedAllInRate = computed(() => {
  if (assumptionForm.value.rate_type !== 'fixed' || !assumptionForm.value.rate_spread_bps) return null
  const spread = (assumptionForm.value.rate_spread_bps || 0) / 10000
  const cushion = (assumptionForm.value.rate_cushion_bps || 0) / 10000
  return spread + cushion  // index rate not known, just show spread+cushion contribution
})

/** Extension display string */
const extensionDisplay = computed(() => {
  const c = assumptionForm.value.extension_count
  const m = assumptionForm.value.extension_months
  if (!c || !m) return 'None'
  const yrs = m >= 12 ? `${m / 12}-year` : `${m}-month`
  return `(${c}) ${yrs} extensions`
})

const sourcesBalanced = computed(() =>
  Math.abs(totalSources.value - totalUses.value) < 1.0
)

// Cap Ex Reserve item (for backend compat)
const capexReserve = computed(() => {
  const item = capitalUses.value.find(u => u.id === 'capex_reserve')
  return getUseAmount(item || { id: '', label: '', amount: null, pct: null, pctBase: null, isFixed: false, removable: false })
})

// Sum of per-property prices
const propertyPriceSum = computed(() => {
  let sum = 0
  for (const v of Object.values(propertyPrices.value)) {
    sum += (v as number) || 0
  }
  return sum
})

// ---------------------------------------------------------------------------
// Computed — Waterfall Builder
// ---------------------------------------------------------------------------

const entityOptions = computed(() => {
  const opts: { value: string; label: string }[] = []
  const seen = new Set<string>()

  for (const ent of entities.value) {
    const id = ent.planned_entity_id || ent.entity_name?.toUpperCase().replace(/\s+/g, '_') || `ENT_${ent.id}`
    if (!seen.has(id)) {
      opts.push({ value: id, label: ent.entity_name || id })
      seen.add(id)
    }
  }
  // Also include entities from existing waterfall steps
  for (const s of [...cfStepInputs.value, ...capStepInputs.value]) {
    if (s.entity_id && !seen.has(s.entity_id)) {
      opts.push({ value: s.entity_id, label: s.entity_id })
      seen.add(s.entity_id)
    }
  }
  // Defaults if empty
  if (!opts.length) {
    opts.push({ value: 'PSC_PE', label: 'PSC Preferred Equity' })
    opts.push({ value: 'OP_PARTNER', label: 'Operating Partner' })
  }
  return opts
})

const cfResidualPct = computed(() =>
  cfStepInputs.value
    .filter(s => s.step_type === 'residual')
    .reduce((sum, s) => sum + (s.rate || 0), 0)
)

const capResidualPct = computed(() =>
  capStepInputs.value
    .filter(s => s.step_type === 'residual')
    .reduce((sum, s) => sum + (s.rate || 0), 0)
)

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

    // Init capital budget from deal data
    initCapitalBudget(detailRes.data)

    // Init waterfall builder
    if (wfHasStored.value) {
      loadWfStepsFromStored()
    } else {
      initDefaultWfSteps()
    }

    // Load first assumption version if exists
    if (assumptionVersions.value.length) {
      selectAssumption(assumptionVersions.value[0])
    }
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  }
}

function initCapitalBudget(data: any) {
  const d = data?.deal || {}
  const props = data?.properties || []

  // Reset uses to defaults
  capitalUses.value = defaultUses()
  debtSources.value = defaultDebtSources()
  additionalEquitySources.value = []

  // Pre-fill purchase price from deal
  const ppItem = capitalUses.value.find(u => u.id === 'purchase_price')
  if (ppItem) ppItem.amount = d.purchase_price || null

  // Pre-fill property prices
  propertyPrices.value = {}
  for (const p of props) {
    propertyPrices.value[p.id] = p.property_price || null
  }

  // Pre-fill PE equity pct from deal
  peEquityPct.value = (d.closing_cost_pct != null && d.closing_cost_pct > 0)
    ? 90 : 90  // Default 90%, can be overridden
}

function initDefaultWfSteps() {
  const ents = entities.value
  if (ents.length) {
    const cfSteps: WfStepInput[] = []
    const capSteps: WfStepInput[] = []
    // Pref steps for PE entities
    for (const ent of ents) {
      const id = ent.planned_entity_id || ent.entity_name?.toUpperCase().replace(/\s+/g, '_') || `ENT_${ent.id}`
      const isPe = (ent.role || '').toLowerCase().includes('pe') ||
                   (ent.role || '').toLowerCase().includes('pref') ||
                   (ent.entity_type || '').toLowerCase().includes('pe')
      if (isPe) {
        cfSteps.push({ entity_id: id, step_type: 'pref', rate: 8.0, amount: null })
        capSteps.push({ entity_id: id, step_type: 'pref', rate: 8.0, amount: null })
      }
    }
    // Cap_WF gets return of capital for all entities
    for (const ent of ents) {
      const id = ent.planned_entity_id || ent.entity_name?.toUpperCase().replace(/\s+/g, '_') || `ENT_${ent.id}`
      capSteps.push({ entity_id: id, step_type: 'return_of_capital', rate: null, amount: null })
    }
    // Residual split for both
    for (const ent of ents) {
      const id = ent.planned_entity_id || ent.entity_name?.toUpperCase().replace(/\s+/g, '_') || `ENT_${ent.id}`
      const share = (ent.ownership_pct || 0) * 100
      cfSteps.push({ entity_id: id, step_type: 'residual', rate: share || 50, amount: null })
      capSteps.push({ entity_id: id, step_type: 'residual', rate: share || 50, amount: null })
    }
    cfStepInputs.value = cfSteps.length ? cfSteps : defaultCfSteps()
    capStepInputs.value = capSteps.length ? capSteps : defaultCapSteps()
  } else {
    cfStepInputs.value = defaultCfSteps()
    capStepInputs.value = defaultCapSteps()
  }
}

function _storedToInputs(steps: WfStep[]): WfStepInput[] {
  const inputs: WfStepInput[] = []
  for (const s of steps) {
    if (s.vState === 'Pref') {
      const rate = s.nPercent > 1 ? s.nPercent : s.nPercent * 100
      inputs.push({ entity_id: s.PropCode, step_type: 'pref', rate, amount: null })
    } else if (s.vState === 'Initial') {
      inputs.push({ entity_id: s.PropCode, step_type: 'return_of_capital', rate: null, amount: null })
    } else if (s.vState === 'Share' || s.vState === 'Tag') {
      inputs.push({ entity_id: s.PropCode, step_type: 'residual', rate: s.FXRate * 100, amount: null })
    } else if (s.vState === 'Amt') {
      inputs.push({ entity_id: s.PropCode, step_type: 'fixed_amount', rate: null, amount: s.mAmount })
    } else if (s.vState === 'IRR') {
      inputs.push({ entity_id: s.PropCode, step_type: 'irr_lookback', rate: s.nPercent, amount: null })
    }
  }
  return inputs
}

function loadWfStepsFromStored() {
  const cfStored = wfSteps.value.filter(s => s.vmisc === 'CF_WF')
  const capStored = wfSteps.value.filter(s => s.vmisc === 'Cap_WF')
  cfStepInputs.value = cfStored.length ? _storedToInputs(cfStored) : defaultCfSteps()
  capStepInputs.value = capStored.length ? _storedToInputs(capStored) : defaultCapSteps()
}

// ---------------------------------------------------------------------------
// Capital Budget actions
// ---------------------------------------------------------------------------

function addUseLine() {
  const n = capitalUses.value.length + 1
  capitalUses.value.push({
    id: `custom_${n}_${Date.now()}`,
    label: '',
    amount: null,
    pct: null,
    pctBase: null,
    isFixed: false,
    removable: true,
  })
}

function removeUseLine(idx: number) {
  capitalUses.value.splice(idx, 1)
}

function addDebtLine() {
  const n = debtSources.value.length + 1
  debtSources.value.push({
    id: `debt_custom_${n}_${Date.now()}`,
    label: '',
    amount: null,
    isDebt: true,
    removable: true,
    level: 'portfolio',
    propertyId: null,
  })
}

function removeDebtLine(idx: number) {
  debtSources.value.splice(idx, 1)
}

function addEquityLine() {
  const n = additionalEquitySources.value.length + 1
  additionalEquitySources.value.push({
    id: `equity_custom_${n}_${Date.now()}`,
    label: '',
    amount: null,
    isDebt: false,
    removable: true,
    level: 'portfolio',
    propertyId: null,
  })
}

function removeEquityLine(idx: number) {
  additionalEquitySources.value.splice(idx, 1)
}

// ---------------------------------------------------------------------------
// Waterfall builder actions
// ---------------------------------------------------------------------------

function addWfStep(list: WfStepInput[]) {
  list.push({
    entity_id: entityOptions.value[0]?.value || '',
    step_type: 'residual',
    rate: null,
    amount: null,
  })
}

function removeWfStep(list: WfStepInput[], idx: number) {
  list.splice(idx, 1)
}

function addNewEntity() {
  const name = prompt('Enter Entity ID (e.g., NEWCO_PE):')
  if (!name) return
  const id = name.toUpperCase().replace(/\s+/g, '_')
  // Add a residual step to both waterfalls
  cfStepInputs.value.push({ entity_id: id, step_type: 'residual', rate: null, amount: null })
  capStepInputs.value.push({ entity_id: id, step_type: 'residual', rate: null, amount: null })
}

function copyCfToCap() {
  // Copy CF steps to Cap, inserting Return of Capital for each entity before the residual split
  const copied: WfStepInput[] = []
  const entities = new Set<string>()
  // First pass: copy prefs and collect entity IDs
  for (const s of cfStepInputs.value) {
    copied.push({ ...s })
    if (s.entity_id) entities.add(s.entity_id)
  }
  // Insert Return of Capital steps before residual split
  const residualIdx = copied.findIndex(s => s.step_type === 'residual')
  if (residualIdx >= 0) {
    const rocSteps: WfStepInput[] = []
    for (const eid of entities) {
      if (!copied.some(s => s.entity_id === eid && s.step_type === 'return_of_capital')) {
        rocSteps.push({ entity_id: eid, step_type: 'return_of_capital', rate: null, amount: null })
      }
    }
    copied.splice(residualIdx, 0, ...rocSteps)
  }
  capStepInputs.value = copied
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
  // Sync PE equity pct from assumption
  if (a.psc_equity_pct != null) {
    peEquityPct.value = (a.psc_equity_pct > 1 ? a.psc_equity_pct : a.psc_equity_pct * 100)
  }
  // Restore capital budget from saved JSON
  if (a.capital_uses_json) {
    try {
      const saved = JSON.parse(a.capital_uses_json)
      if (Array.isArray(saved) && saved.length) {
        // Merge saved amounts into default structure (preserves any new fields added later)
        const base = defaultUses()
        const savedMap = new Map(saved.map((u: any) => [u.id, u]))
        for (const item of base) {
          const s = savedMap.get(item.id)
          if (s) {
            item.amount = s.amount
            if (s.pct != null) item.pct = s.pct
          }
        }
        capitalUses.value = base
      }
    } catch { /* ignore parse errors */ }
  }
  if (a.capital_sources_json) {
    try {
      const saved = JSON.parse(a.capital_sources_json)
      if (saved.debt && Array.isArray(saved.debt)) {
        // Merge saved debt amounts into default structure
        const base = defaultDebtSources()
        const savedMap = new Map(saved.debt.map((s: any) => [s.id, s]))
        for (const item of base) {
          const s = savedMap.get(item.id)
          if (s) item.amount = s.amount
        }
        debtSources.value = base
      }
      if (saved.equity && Array.isArray(saved.equity)) {
        additionalEquitySources.value = saved.equity
      }
      if (saved.pe_pct != null) {
        peEquityPct.value = saved.pe_pct
      }
    } catch { /* ignore parse errors */ }
  } else if (a.debt_amount != null) {
    // Fallback: sync debt amount to first mortgage source (legacy)
    const fm = debtSources.value.find(s => s.id === 'first_mortgage')
    if (fm) fm.amount = a.debt_amount
  }
}

async function saveAssumptions() {
  if (!selectedDealId.value) return
  savingAssumptions.value = true
  try {
    const payload = {
      ...(selectedAssumptionId.value ? { id: selectedAssumptionId.value } : {}),
      ...assumptionForm.value,
      debt_amount: totalDebt.value,
      psc_equity_pct: peEquityPct.value / 100,
      capital_uses_json: JSON.stringify(capitalUses.value),
      capital_sources_json: JSON.stringify({
        debt: debtSources.value,
        equity: additionalEquitySources.value,
        pe_pct: peEquityPct.value,
      }),
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
  if (!selectedDealId.value || (!cfStepInputs.value.length && !capStepInputs.value.length)) return
  wfSaving.value = true
  // Auto-save assumptions (persists capital budget alongside waterfall)
  await saveAssumptions()
  try {
    const res = await api.post(`/api/prospects/${selectedDealId.value}/waterfall/build`, {
      cf_steps: cfStepInputs.value.filter(s => s.entity_id),
      cap_steps: capStepInputs.value.filter(s => s.entity_id),
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

  // Auto-save assumptions (persists capital budget, debt params, etc.)
  await saveAssumptions()

  try {
    // Compute effective closing_cost_pct so backend formula reproduces totalUses
    // total_cost = PP + PP * closing_cost_pct + capex_at_close = totalUses
    // closing_cost_pct = (totalUses - PP - capex) / PP
    const pp = purchasePrice.value
    const capex = capexReserve.value
    const effectiveClosingPct = pp > 0 ? (totalUses.value - pp - capex) / pp : 0

    const payload: any = {
      ...assumptionForm.value,
      purchase_price_override: pp,
      closing_cost_pct_override: effectiveClosingPct,
      capex_at_close_override: capex,
      debt_amount: totalDebt.value,
      psc_equity_pct: peEquityPct.value / 100,
    }
    if (selectedAssumptionId.value) {
      payload.assumption_id = selectedAssumptionId.value
    }

    // Include property-level prices
    if (properties.value.length > 1) {
      payload.property_prices = propertyPrices.value
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

/** Format a raw number as currency string for display in inputs */
function fmtInputCurrency(v: number | null): string {
  if (v == null || v === 0) return ''
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(v)
}

/** Parse formatted currency string back to number */
function parseInputCurrency(s: string): number | null {
  const cleaned = s.replace(/[$,\s]/g, '')
  if (!cleaned) return null
  const n = Number(cleaned)
  return isNaN(n) ? null : n
}

// Formatted-input focus/blur handlers for UseItem amount fields
const focusedUseIdx = ref<number | null>(null)
const focusedDebtIdx = ref<number | null>(null)
const focusedEquityIdx = ref<number | null>(null)
const focusedPropPriceId = ref<number | null>(null)

function onUseAmountBlur(item: UseItem, e: Event) {
  focusedUseIdx.value = null
  const raw = (e.target as HTMLInputElement).value
  item.amount = parseInputCurrency(raw)
}

function onDebtAmountBlur(item: SourceItem, e: Event) {
  focusedDebtIdx.value = null
  const raw = (e.target as HTMLInputElement).value
  item.amount = parseInputCurrency(raw)
}

function onEquityAmountBlur(item: SourceItem, e: Event) {
  focusedEquityIdx.value = null
  const raw = (e.target as HTMLInputElement).value
  item.amount = parseInputCurrency(raw)
}

function onPropPriceBlur(propId: number, e: Event) {
  focusedPropPriceId.value = null
  const raw = (e.target as HTMLInputElement).value
  propertyPrices.value[propId] = parseInputCurrency(raw)
}

// Collapsible Operating Assumptions
const showOperatingAssumptions = ref(false)

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

          <!-- ============ CAPITAL BUDGET ============ -->
          <div class="section">
            <div class="section-header clickable" @click="expandedBudget = !expandedBudget">
              Capital Budget
              <span class="chevron">{{ expandedBudget ? '\u25BE' : '\u25B8' }}</span>
            </div>

            <template v-if="expandedBudget">

              <!-- USES -->
              <div class="budget-subsection">
                <h5 class="budget-title">Capital Uses</h5>
                <table class="budget-table">
                  <thead>
                    <tr>
                      <th>Item</th>
                      <th class="r col-pct">%</th>
                      <th class="r col-amt">Amount</th>
                      <th class="col-action"></th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="(item, i) in capitalUses" :key="item.id"
                        :class="{ 'fixed-row': item.isFixed }">
                      <td>
                        <input v-if="item.removable" v-model="item.label"
                               class="inline-label-input" placeholder="Line item name" />
                        <span v-else>{{ item.label }}</span>
                        <div v-if="item.id === 'purchase_price'" class="sub-metrics">
                          Cap Rate: <strong>{{ capRateDisplay }}</strong> &nbsp;|&nbsp;
                          PSF: <strong>{{ psfDisplay }}</strong>
                        </div>
                      </td>
                      <td class="r col-pct">
                        <template v-if="item.pctBase && !item.isFixed">
                          <input type="text" class="fmt-input fmt-pct"
                                 :value="focusedUseIdx === i ? (item.pct ?? '') : (item.pct != null ? item.pct.toFixed(2) + '%' : '')"
                                 :placeholder="item.pctBase === 'purchase_price' ? '% PP' : '% Debt'"
                                 @focus="focusedUseIdx = i; ($event.target as HTMLInputElement).value = item.pct != null ? String(item.pct) : ''"
                                 @blur="focusedUseIdx = null; item.pct = parseFloat(($event.target as HTMLInputElement).value) || null" />
                        </template>
                        <span v-else-if="item.isFixed" class="fmt-display">1.00%</span>
                      </td>
                      <td class="r col-amt">
                        <!-- If pct is set, show computed amount (read-only display) -->
                        <template v-if="item.isFixed">
                          <span class="fmt-display">{{ fmtCurrency(pscOrigFee) }}</span>
                        </template>
                        <template v-else-if="item.pctBase && item.pct">
                          <span class="fmt-display">{{ fmtCurrency(getUseAmount(item)) }}</span>
                        </template>
                        <template v-else>
                          <input type="text" class="fmt-input fmt-amt"
                                 :value="focusedUseIdx === i ? (item.amount ?? '') : fmtInputCurrency(item.amount)"
                                 placeholder="$0"
                                 @focus="focusedUseIdx = i; ($event.target as HTMLInputElement).value = item.amount != null ? String(item.amount) : ''"
                                 @blur="onUseAmountBlur(item, $event)" />
                        </template>
                      </td>
                      <td class="col-action">
                        <button v-if="item.removable" @click="removeUseLine(i)"
                                class="btn-icon btn-danger btn-xs">&times;</button>
                      </td>
                    </tr>
                  </tbody>
                  <tfoot>
                    <tr class="total-row">
                      <td><strong>Total Uses</strong></td>
                      <td></td>
                      <td class="r"><strong>{{ fmtCurrency(totalUses) }}</strong></td>
                      <td></td>
                    </tr>
                  </tfoot>
                </table>
                <button class="btn-sm btn-secondary mt-4" @click="addUseLine">+ Add Line</button>
              </div>

              <!-- Per-Property Purchase Prices (portfolio deals) -->
              <div v-if="properties.length > 1" class="budget-subsection">
                <h5 class="budget-title">Per-Property Purchase Prices</h5>
                <table class="budget-table compact">
                  <tbody>
                    <tr v-for="p in properties" :key="p.id">
                      <td>{{ p.property_name }}</td>
                      <td class="r col-amt">
                        <input type="text" class="fmt-input fmt-amt"
                               :value="focusedPropPriceId === p.id ? (propertyPrices[p.id] ?? '') : fmtInputCurrency(propertyPrices[p.id])"
                               placeholder="$0"
                               @focus="focusedPropPriceId = p.id; ($event.target as HTMLInputElement).value = propertyPrices[p.id] != null ? String(propertyPrices[p.id]) : ''"
                               @blur="onPropPriceBlur(p.id, $event)" />
                      </td>
                    </tr>
                  </tbody>
                  <tfoot>
                    <tr class="subtotal-row">
                      <td>Sum</td>
                      <td class="r">
                        {{ fmtCurrency(propertyPriceSum) }}
                        <span v-if="Math.abs(propertyPriceSum - purchasePrice) > 1 && purchasePrice > 0"
                              class="mismatch-warn">
                          (does not match Purchase Price)
                        </span>
                      </td>
                    </tr>
                  </tfoot>
                </table>
              </div>

              <!-- SOURCES — Debt -->
              <div class="budget-subsection">
                <h5 class="budget-title">Capital Sources — Debt</h5>
                <table class="budget-table">
                  <thead>
                    <tr>
                      <th>Item</th>
                      <th class="r col-pct">Level</th>
                      <th class="r col-amt">Amount</th>
                      <th class="col-action"></th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="(item, i) in debtSources" :key="item.id">
                      <td>
                        <input v-if="item.removable" v-model="item.label"
                               class="inline-label-input" placeholder="Debt source name" />
                        <span v-else>{{ item.label }}</span>
                      </td>
                      <td class="r col-pct">
                        <select v-if="properties.length > 1" v-model="item.level" class="level-select">
                          <option value="portfolio">Portfolio</option>
                          <option value="property">Property</option>
                        </select>
                        <span v-else class="level-badge">Property</span>
                        <select v-if="item.level === 'property' && properties.length > 1"
                                v-model.number="item.propertyId" class="prop-select">
                          <option :value="null">— Select —</option>
                          <option v-for="p in properties" :key="p.id" :value="p.id">{{ p.property_name }}</option>
                        </select>
                      </td>
                      <td class="r col-amt">
                        <input type="text" class="fmt-input fmt-amt"
                               :value="focusedDebtIdx === i ? (item.amount ?? '') : fmtInputCurrency(item.amount)"
                               placeholder="$0"
                               @focus="focusedDebtIdx = i; ($event.target as HTMLInputElement).value = item.amount != null ? String(item.amount) : ''"
                               @blur="onDebtAmountBlur(item, $event)" />
                      </td>
                      <td class="col-action">
                        <button v-if="item.removable" @click="removeDebtLine(i)"
                                class="btn-icon btn-danger btn-xs">&times;</button>
                      </td>
                    </tr>
                  </tbody>
                  <tfoot>
                    <tr class="subtotal-row">
                      <td><strong>Total Debt</strong></td>
                      <td></td>
                      <td class="r"><strong>{{ fmtCurrency(totalDebt) }}</strong></td>
                      <td></td>
                    </tr>
                  </tfoot>
                </table>
                <button class="btn-sm btn-secondary mt-4" @click="addDebtLine">+ Add Debt Line</button>
              </div>

              <!-- SOURCES — Equity -->
              <div class="budget-subsection">
                <h5 class="budget-title">Capital Sources — Equity</h5>
                <table class="budget-table">
                  <thead>
                    <tr>
                      <th>Item</th>
                      <th class="r col-pct">% of Equity</th>
                      <th class="r col-amt">Amount</th>
                      <th class="col-action"></th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>PSC Preferred Equity</td>
                      <td class="r col-pct">
                        <input type="text" class="fmt-input fmt-pct"
                               :value="focusedEquityIdx === -1 ? peEquityPct : peEquityPct.toFixed(1) + '%'"
                               @focus="focusedEquityIdx = -1; ($event.target as HTMLInputElement).value = String(peEquityPct)"
                               @blur="focusedEquityIdx = null; peEquityPct = parseFloat(($event.target as HTMLInputElement).value) || 0" />
                      </td>
                      <td class="r col-amt">
                        <span class="fmt-display">{{ fmtCurrency(peAmount) }}</span>
                      </td>
                      <td class="col-action"></td>
                    </tr>
                    <tr>
                      <td>Partner Equity</td>
                      <td class="r col-pct">
                        <span class="fmt-display">{{ partnerEquityPct.toFixed(1) }}%</span>
                      </td>
                      <td class="r col-amt">
                        <span class="fmt-display">{{ fmtCurrency(partnerEquity) }}</span>
                      </td>
                      <td class="col-action"></td>
                    </tr>
                    <tr v-for="(item, i) in additionalEquitySources" :key="item.id">
                      <td>
                        <input v-model="item.label" class="inline-label-input" placeholder="Equity source" />
                      </td>
                      <td class="r col-pct"></td>
                      <td class="r col-amt">
                        <input type="text" class="fmt-input fmt-amt"
                               :value="focusedEquityIdx === i ? (item.amount ?? '') : fmtInputCurrency(item.amount)"
                               placeholder="$0"
                               @focus="focusedEquityIdx = i; ($event.target as HTMLInputElement).value = item.amount != null ? String(item.amount) : ''"
                               @blur="onEquityAmountBlur(item, $event)" />
                      </td>
                      <td class="col-action">
                        <button @click="removeEquityLine(i)" class="btn-icon btn-danger btn-xs">&times;</button>
                      </td>
                    </tr>
                  </tbody>
                  <tfoot>
                    <tr class="subtotal-row">
                      <td><strong>Total Equity</strong></td>
                      <td></td>
                      <td class="r"><strong>{{ fmtCurrency(totalEquity) }}</strong></td>
                      <td></td>
                    </tr>
                  </tfoot>
                </table>
                <button class="btn-sm btn-secondary mt-4" @click="addEquityLine">+ Add Equity Line</button>
              </div>

              <!-- Balance Check -->
              <div class="balance-check" :class="{ balanced: sourcesBalanced, unbalanced: !sourcesBalanced && totalUses > 0 }">
                <div class="balance-row">
                  <span>Total Sources</span>
                  <strong>{{ fmtCurrency(totalSources) }}</strong>
                </div>
                <div class="balance-row">
                  <span>Total Uses</span>
                  <strong>{{ fmtCurrency(totalUses) }}</strong>
                </div>
                <div v-if="totalUses > 0" class="balance-status">
                  {{ sourcesBalanced ? 'Balanced' : 'Sources and Uses do not match' }}
                </div>
              </div>

              <button class="btn-sm btn-primary mt-4" @click="saveAssumptions" :disabled="!selectedDealId">
                Save Capital Budget
              </button>

            </template>
          </div>

          <!-- ============ FIRST MORTGAGE TERMS ============ -->
          <div class="section">
            <div class="section-header">First Mortgage Terms</div>
            <table class="terms-table">
              <tbody>
                <tr>
                  <td class="terms-label">Lender</td>
                  <td><input type="text" v-model="assumptionForm.lender" class="terms-text" placeholder="e.g. United Bank" /></td>
                </tr>
                <tr>
                  <td class="terms-label">Interest Rate</td>
                  <td>
                    <div class="rate-builder">
                      <select v-model="assumptionForm.rate_type" class="terms-select-sm">
                        <option value="fixed">Fixed</option>
                        <option value="floating">Floating</option>
                      </select>
                      <span class="rate-desc">priced</span>
                      <select v-model="assumptionForm.rate_index_term" class="terms-select-sm">
                        <option value="1yr">1-yr</option>
                        <option value="3yr">3-yr</option>
                        <option value="5yr">5-yr</option>
                        <option value="7yr">7-yr</option>
                        <option value="10yr">10-yr</option>
                      </select>
                      <select v-model="assumptionForm.rate_index" class="terms-select-sm">
                        <option value="UST">UST</option>
                        <option value="SOFR">SOFR</option>
                        <option value="Prime">Prime</option>
                      </select>
                      <span class="rate-desc">+</span>
                      <input type="number" v-model.number="assumptionForm.rate_spread_bps"
                             class="terms-num-sm" placeholder="170" /> <span class="rate-desc">bps</span>
                      <span class="rate-desc">+ cushion</span>
                      <input type="number" v-model.number="assumptionForm.rate_cushion_bps"
                             class="terms-num-sm" placeholder="25" /> <span class="rate-desc">bps</span>
                      <span class="rate-desc">=</span>
                      <input type="number" v-model.number="assumptionForm.debt_rate" step="0.0025"
                             class="terms-num-sm" />
                      <span class="rate-desc">{{ assumptionForm.debt_rate ? (assumptionForm.debt_rate * 100).toFixed(2) + '%' : '' }}</span>
                    </div>
                  </td>
                </tr>
                <tr>
                  <td class="terms-label">Term</td>
                  <td>
                    <div class="terms-inline">
                      <input type="number" v-model.number="assumptionForm.debt_term_months"
                             class="terms-num-sm" /> <span class="rate-desc">months</span>
                      <span class="rate-desc">with</span>
                      <input type="number" v-model.number="assumptionForm.extension_count"
                             class="terms-num-xs" placeholder="0" />
                      <span class="rate-desc">×</span>
                      <input type="number" v-model.number="assumptionForm.extension_months"
                             class="terms-num-sm" placeholder="12" />
                      <span class="rate-desc">month extensions</span>
                    </div>
                  </td>
                </tr>
                <tr>
                  <td class="terms-label">Amortization</td>
                  <td>
                    <div class="terms-inline">
                      <input type="number" v-model.number="assumptionForm.io_months"
                             class="terms-num-sm" /> <span class="rate-desc">months I/O, thereafter</span>
                      <input type="number" v-model.number="assumptionForm.amort_months"
                             class="terms-num-sm" /> <span class="rate-desc">month ({{ assumptionForm.amort_months ? (assumptionForm.amort_months / 12).toFixed(0) + '-yr' : '' }}) amortization</span>
                    </div>
                  </td>
                </tr>
                <tr>
                  <td class="terms-label">Origination Fee</td>
                  <td>
                    <div class="terms-inline">
                      <input type="number" v-model.number="assumptionForm.origination_fee_bps"
                             class="terms-num-sm" placeholder="50" /> <span class="rate-desc">bps</span>
                      <span v-if="assumptionForm.origination_fee_bps && totalDebt > 0" class="rate-desc">
                        ({{ fmtCurrency(totalDebt * (assumptionForm.origination_fee_bps / 10000)) }})
                      </span>
                    </div>
                  </td>
                </tr>
                <tr>
                  <td class="terms-label">Sizing Constraints</td>
                  <td>
                    <div class="constraints-grid">
                      <div class="constraint-item">
                        <label>Max LTV</label>
                        <div class="terms-inline">
                          <input type="number" v-model.number="assumptionForm.max_ltv" step="0.5"
                                 class="terms-num-sm" placeholder="70" /> <span class="rate-desc">%</span>
                        </div>
                      </div>
                      <div class="constraint-item">
                        <label>Max LTC</label>
                        <div class="terms-inline">
                          <input type="number" v-model.number="assumptionForm.max_ltc" step="0.5"
                                 class="terms-num-sm" placeholder="72" /> <span class="rate-desc">%</span>
                        </div>
                      </div>
                      <div class="constraint-item">
                        <label>Min DSCR</label>
                        <div class="terms-inline">
                          <input type="number" v-model.number="assumptionForm.min_dscr" step="0.05"
                                 class="terms-num-sm" placeholder="1.25" /> <span class="rate-desc">x</span>
                        </div>
                      </div>
                      <div class="constraint-item">
                        <label>Min Debt Yield</label>
                        <div class="terms-inline">
                          <input type="number" v-model.number="assumptionForm.min_debt_yield" step="0.25"
                                 class="terms-num-sm" placeholder="10.5" /> <span class="rate-desc">%</span>
                        </div>
                      </div>
                    </div>
                  </td>
                </tr>
                <tr>
                  <td class="terms-label">Prepayment</td>
                  <td>
                    <div class="terms-inline">
                      <select v-model="assumptionForm.prepay_type" class="terms-select-sm">
                        <option value="">None</option>
                        <option value="open">Open</option>
                        <option value="lockout">Lockout</option>
                        <option value="defeasance">Defeasance</option>
                        <option value="yield_maint">Yield Maintenance</option>
                        <option value="step_down">Step-Down Penalty</option>
                      </select>
                    </div>
                    <div v-if="assumptionForm.prepay_type === 'step_down' || assumptionForm.prepay_type === 'open'" class="terms-sub">
                      <label>Penalty schedule (by year)</label>
                      <input type="text" v-model="assumptionForm.prepay_schedule"
                             class="terms-text" placeholder="e.g. 3%, 2%, 1%, 1%, 0%" />
                    </div>
                  </td>
                </tr>
                <tr>
                  <td class="terms-label">Earnout / Future Funding</td>
                  <td>
                    <textarea v-model="assumptionForm.earnout_notes" class="terms-textarea"
                              rows="2" placeholder="e.g. Between month 18-36, borrower may request increase subject to..."></textarea>
                  </td>
                </tr>
                <tr>
                  <td class="terms-label">Guarantor</td>
                  <td>
                    <input type="text" v-model="assumptionForm.guarantor_notes" class="terms-text"
                           placeholder="e.g. Sponsor guarantees 25% of outstanding balance" />
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          <!-- ============ OPERATING ASSUMPTIONS (collapsible) ============ -->
          <div class="section">
            <div class="section-header clickable" @click="showOperatingAssumptions = !showOperatingAssumptions">
              Operating Assumptions
              <span class="chevron">{{ showOperatingAssumptions ? '\u25BE' : '\u25B8' }}</span>
              <span class="section-hint">(used when no imported cash flows)</span>
            </div>
            <div v-if="showOperatingAssumptions" class="form-grid-3">
              <div class="form-group">
                <label>Year 1 NOI ($)</label>
                <input type="number" v-model.number="assumptionForm.noi_year1" step="1000" />
              </div>
              <div class="form-group">
                <label>NOI Growth Rate</label>
                <input type="number" v-model.number="assumptionForm.noi_growth_rate" step="0.005" />
              </div>
              <div class="form-group">
                <label>CapEx Reserve ($/SF)</label>
                <input type="number" v-model.number="assumptionForm.capex_reserve_psf" step="0.10" />
              </div>
            </div>
          </div>

          <!-- ============ SALE / EXIT ============ -->
          <div class="section">
            <div class="section-header">Sale / Exit</div>
            <div class="form-grid-3">
              <div class="form-group">
                <label>Hold Period (years)</label>
                <input type="number" v-model.number="assumptionForm.hold_years" />
              </div>
              <div class="form-group">
                <label>Exit Cap Rate</label>
                <input type="number" v-model.number="assumptionForm.exit_cap_rate" step="0.0025" />
              </div>
              <div class="form-group">
                <label>Cost of Sale %</label>
                <input type="number" v-model.number="assumptionForm.selling_cost_pct" step="0.005" />
              </div>
            </div>

            <!-- Sale Proceeds Summary (shown after analysis runs) -->
            <div v-if="analysisResult?.sale_dbg" class="sale-proceeds-summary">
              <h5 class="budget-title">Net Sale Proceeds</h5>
              <table class="proceeds-table">
                <tbody>
                  <tr>
                    <td>Sale Price ({{ analysisResult.sale_dbg.Sale_Price_Source === 'contract' ? 'Contract' : 'NOI / Cap Rate' }})</td>
                    <td class="r">{{ fmtCurrency(analysisResult.sale_dbg.Implied_Value) }}</td>
                  </tr>
                  <tr>
                    <td>Less: Cost of Sale ({{ analysisResult.sale_dbg.Selling_Cost_Label }})</td>
                    <td class="r neg">{{ fmtCurrency(-analysisResult.sale_dbg.Selling_Cost_Amount) }}</td>
                  </tr>
                  <tr class="subtotal-row">
                    <td>Value Net of Selling Costs</td>
                    <td class="r">{{ fmtCurrency(analysisResult.sale_dbg.Value_Net_Selling_Cost) }}</td>
                  </tr>
                  <tr>
                    <td>Less: Loan Payoff</td>
                    <td class="r neg">{{ fmtCurrency(-analysisResult.sale_dbg.Less_Loan_Balances) }}</td>
                  </tr>
                  <tr v-if="analysisResult.sale_dbg.Tax_Abatement_NPV > 0">
                    <td>Plus: NPV Tax Abatements</td>
                    <td class="r">{{ fmtCurrency(analysisResult.sale_dbg.Tax_Abatement_NPV) }}</td>
                  </tr>
                  <tr class="total-row">
                    <td><strong>Net Sale Proceeds</strong></td>
                    <td class="r"><strong>{{ fmtCurrency(analysisResult.sale_dbg.Net_Sale_Proceeds) }}</strong></td>
                  </tr>
                </tbody>
              </table>

              <!-- Per-partner capital distributions from waterfall -->
              <div v-if="analysisResult.partner_results?.length" class="partner-proceeds">
                <h5 class="budget-title" style="margin-top: 8px;">Capital Distributions to Partners</h5>
                <table class="proceeds-table">
                  <tbody>
                    <tr v-for="p in analysisResult.partner_results" :key="p.partner || p.investor_id"
                        :class="{ 'pe-row-light': p.is_pref_equity || p.is_pe }">
                      <td>{{ p.partner || p.investor_id }}</td>
                      <td class="r">{{ fmtCurrency(p.capital_distributions) }}</td>
                    </tr>
                    <tr v-if="analysisResult.deal_summary" class="subtotal-row">
                      <td><strong>Total</strong></td>
                      <td class="r"><strong>{{ fmtCurrency(analysisResult.deal_summary.total_capital_distributions || analysisResult.deal_summary.cap_distributions) }}</strong></td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <div class="sale-meta">
                Sale Date: {{ analysisResult.sale_dbg.Sale_Date }}
                &nbsp;|&nbsp; Terminal NOI: {{ fmtCurrency(analysisResult.sale_dbg.NOI_12m_After_Sale) }}
                &nbsp;|&nbsp; Exit Cap: {{ analysisResult.sale_dbg.CapRate_Sale ? (analysisResult.sale_dbg.CapRate_Sale * 100).toFixed(2) + '%' : 'n/a' }}
              </div>
            </div>
          </div>

          <!-- ============ WATERFALL STRUCTURE ============ -->
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
              <!-- CF Waterfall Builder -->
              <div class="wf-builder-section">
                <div class="wf-builder-header">
                  <h5>Cash Flow Waterfall (CF_WF)</h5>
                  <span class="wf-type-desc">Operating distributions — does NOT reduce capital outstanding</span>
                </div>
                <table class="budget-table wf-table">
                  <thead>
                    <tr>
                      <th class="col-num">#</th>
                      <th>Entity</th>
                      <th>Step Type</th>
                      <th class="r col-rate">Rate / Amount</th>
                      <th class="col-action"></th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="(step, i) in cfStepInputs" :key="'cf-'+i">
                      <td class="col-num">{{ i + 1 }}</td>
                      <td>
                        <select v-model="step.entity_id" class="wf-select">
                          <option value="">— Select —</option>
                          <option v-for="e in entityOptions" :key="e.value" :value="e.value">{{ e.label }}</option>
                        </select>
                      </td>
                      <td>
                        <select v-model="step.step_type" class="wf-select">
                          <option v-for="st in STEP_TYPES" :key="st.value" :value="st.value">{{ st.label }}</option>
                        </select>
                      </td>
                      <td class="r col-rate">
                        <template v-if="step.step_type === 'pref'">
                          <input type="number" v-model.number="step.rate" step="0.25" class="rate-input" placeholder="8.0" /><span class="rate-suffix">%</span>
                        </template>
                        <template v-else-if="step.step_type === 'residual'">
                          <input type="number" v-model.number="step.rate" step="1" class="rate-input" placeholder="90" /><span class="rate-suffix">%</span>
                        </template>
                        <template v-else-if="step.step_type === 'fixed_amount'">
                          <span class="rate-suffix">$</span><input type="number" v-model.number="step.amount" step="1000" class="rate-input" placeholder="0" />
                        </template>
                        <template v-else-if="step.step_type === 'irr_lookback'">
                          <input type="number" v-model.number="step.rate" step="0.5" class="rate-input" placeholder="9.0" /><span class="rate-suffix">%</span>
                        </template>
                        <template v-else><span class="muted">—</span></template>
                      </td>
                      <td class="col-action">
                        <button class="btn-icon btn-danger btn-xs" @click="removeWfStep(cfStepInputs, i)">&times;</button>
                      </td>
                    </tr>
                  </tbody>
                </table>
                <div class="wf-add-row">
                  <button class="btn-sm btn-secondary" @click="addWfStep(cfStepInputs)">+ Add Step</button>
                </div>
                <div v-if="cfResidualPct > 0 && Math.abs(cfResidualPct - 100) > 0.5" class="wf-warning">
                  CF split shares sum to {{ cfResidualPct.toFixed(1) }}% (should be 100%)
                </div>
              </div>

              <!-- Cap Waterfall Builder -->
              <div class="wf-builder-section">
                <div class="wf-builder-header">
                  <h5>Capital Event Waterfall (Cap_WF)</h5>
                  <span class="wf-type-desc">Refi / sale proceeds — DOES reduce capital outstanding</span>
                  <button class="btn-xs btn-link" @click="copyCfToCap" title="Copy CF steps and add Return of Capital">Copy from CF</button>
                </div>
                <table class="budget-table wf-table">
                  <thead>
                    <tr>
                      <th class="col-num">#</th>
                      <th>Entity</th>
                      <th>Step Type</th>
                      <th class="r col-rate">Rate / Amount</th>
                      <th class="col-action"></th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="(step, i) in capStepInputs" :key="'cap-'+i">
                      <td class="col-num">{{ i + 1 }}</td>
                      <td>
                        <select v-model="step.entity_id" class="wf-select">
                          <option value="">— Select —</option>
                          <option v-for="e in entityOptions" :key="e.value" :value="e.value">{{ e.label }}</option>
                        </select>
                      </td>
                      <td>
                        <select v-model="step.step_type" class="wf-select">
                          <option v-for="st in STEP_TYPES" :key="st.value" :value="st.value">{{ st.label }}</option>
                        </select>
                      </td>
                      <td class="r col-rate">
                        <template v-if="step.step_type === 'pref'">
                          <input type="number" v-model.number="step.rate" step="0.25" class="rate-input" placeholder="8.0" /><span class="rate-suffix">%</span>
                        </template>
                        <template v-else-if="step.step_type === 'residual'">
                          <input type="number" v-model.number="step.rate" step="1" class="rate-input" placeholder="90" /><span class="rate-suffix">%</span>
                        </template>
                        <template v-else-if="step.step_type === 'fixed_amount'">
                          <span class="rate-suffix">$</span><input type="number" v-model.number="step.amount" step="1000" class="rate-input" placeholder="0" />
                        </template>
                        <template v-else-if="step.step_type === 'irr_lookback'">
                          <input type="number" v-model.number="step.rate" step="0.5" class="rate-input" placeholder="9.0" /><span class="rate-suffix">%</span>
                        </template>
                        <template v-else><span class="muted">—</span></template>
                      </td>
                      <td class="col-action">
                        <button class="btn-icon btn-danger btn-xs" @click="removeWfStep(capStepInputs, i)">&times;</button>
                      </td>
                    </tr>
                  </tbody>
                </table>
                <div class="wf-add-row">
                  <button class="btn-sm btn-secondary" @click="addWfStep(capStepInputs)">+ Add Step</button>
                </div>
                <div v-if="capResidualPct > 0 && Math.abs(capResidualPct - 100) > 0.5" class="wf-warning">
                  Cap split shares sum to {{ capResidualPct.toFixed(1) }}% (should be 100%)
                </div>
              </div>

              <!-- Shared actions -->
              <div class="wf-add-row" style="margin-top: 4px;">
                <button class="btn-sm btn-secondary" @click="addNewEntity">+ New Entity (both)</button>
              </div>

              <div class="wf-actions">
                <button class="btn-primary" @click="buildAndSaveWaterfall"
                        :disabled="wfSaving || (!cfStepInputs.length && !capStepInputs.length)">
                  {{ wfSaving ? 'Saving...' : (wfHasStored ? 'Rebuild & Save' : 'Build & Save Waterfall') }}
                </button>
                <button v-if="wfHasStored" class="btn-danger-text" @click="deleteWaterfall">
                  Delete Waterfall
                </button>
              </div>
            </div>

            <!-- Steps tab (preview of stored waterfall) -->
            <div v-if="wfTab === 'steps' && wfSteps.length" class="wf-steps-view">
              <div v-for="wfType in ['CF_WF', 'Cap_WF']" :key="wfType" class="wf-type-section">
                <div class="wf-type-header">
                  <h5>{{ wfType === 'CF_WF' ? 'Cash Flow Waterfall (CF_WF)' : 'Capital Event Waterfall (Cap_WF)' }}</h5>
                  <span class="wf-type-desc">{{
                    wfType === 'CF_WF'
                      ? 'Operating distributions — does NOT reduce capital outstanding'
                      : 'Refi / sale proceeds — DOES reduce capital outstanding'
                  }}</span>
                </div>
                <table class="compact-table" v-if="wfSteps.filter(x => x.vmisc === wfType).length">
                  <thead>
                    <tr>
                      <th>Order</th>
                      <th>Investor</th>
                      <th>Step</th>
                      <th class="r">FXRate</th>
                      <th class="r">Rate</th>
                      <th>Description</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="s in wfSteps.filter(x => x.vmisc === wfType)" :key="`${s.iOrder}-${s.PropCode}`"
                        :class="{ 'step-pref': s.vState === 'Pref', 'step-initial': s.vState === 'Initial', 'step-split': s.vState === 'Share' || s.vState === 'Tag' }">
                      <td>{{ s.iOrder }}</td>
                      <td>{{ s.PropCode }}</td>
                      <td><span class="step-badge" :class="'step-' + s.vState.toLowerCase()">{{ s.vState }}</span></td>
                      <td class="r">{{ s.FXRate ? s.FXRate.toFixed(2) : '—' }}</td>
                      <td class="r">{{ s.nPercent ? (s.nPercent > 1 ? s.nPercent.toFixed(1) + '%' : fmtPct(s.nPercent)) : '—' }}</td>
                      <td>{{ s.vtranstype || '—' }}</td>
                    </tr>
                  </tbody>
                </table>
                <div v-else class="muted" style="padding: 8px 0;">No steps</div>
              </div>

              <div class="wf-explanation">
                <strong>How it works:</strong>
                <span v-if="wfSteps.some(s => s.vState === 'Pref')">
                  Pref steps pay the preferred return first.
                </span>
                <span v-if="wfSteps.some(s => s.vState === 'Initial')">
                  Initial steps return capital (Cap_WF only).
                </span>
                <span v-if="wfSteps.some(s => s.vState === 'Share' || s.vState === 'Tag')">
                  Share/Tag steps split remaining cash — Share (lead) and Tag (followers)
                  receive their FXRate percentage of the pool simultaneously.
                </span>
              </div>
            </div>
          </div>

          <!-- Action buttons -->
          <div class="action-bar">
            <button class="btn-primary btn-lg" @click="runAnalysis"
                    :disabled="analysisLoading || !purchasePrice">
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
            <!-- Sources & Uses Summary -->
            <div class="section">
              <div class="section-header">Sources & Uses</div>
              <div class="su-grid">
                <div class="su-col">
                  <h5>Uses</h5>
                  <template v-for="item in capitalUses" :key="item.id">
                    <div v-if="getUseAmount(item) > 0" class="su-row">
                      <span>{{ item.label }}</span>
                      <span class="r">{{ fmtCurrency(getUseAmount(item)) }}</span>
                    </div>
                  </template>
                  <div class="su-row su-total"><span>Total Uses</span><span class="r">{{ fmtCurrency(totalUses) }}</span></div>
                </div>
                <div class="su-col">
                  <h5>Sources</h5>
                  <template v-for="item in debtSources" :key="item.id">
                    <div v-if="(item.amount || 0) > 0" class="su-row">
                      <span>{{ item.label }}</span>
                      <span class="r">{{ fmtCurrency(item.amount) }}</span>
                    </div>
                  </template>
                  <div v-if="totalDebt > 0" class="su-row su-subtotal">
                    <span>Total Debt</span>
                    <span class="r">{{ fmtCurrency(totalDebt) }} ({{ purchasePrice > 0 ? ((totalDebt / purchasePrice) * 100).toFixed(1) + '% LTV' : '' }})</span>
                  </div>
                  <div class="su-row"><span>PSC Preferred Equity</span><span class="r">{{ fmtCurrency(peAmount) }}</span></div>
                  <div class="su-row"><span>Partner Equity</span><span class="r">{{ fmtCurrency(partnerEquity) }}</span></div>
                  <div class="su-row su-total"><span>Total Sources</span><span class="r">{{ fmtCurrency(totalSources) }}</span></div>
                </div>
              </div>
            </div>

            <!-- Deal Summary KPIs -->
            <div class="section" v-if="analysisResult.deal_summary">
              <div class="section-header">Deal-Level Summary</div>
              <div class="metrics-row">
                <div class="metric-card" v-if="analysisResult.deal_summary.deal_irr != null">
                  <div class="metric-label">Deal IRR</div>
                  <div class="metric-value">{{ fmtPct(analysisResult.deal_summary.deal_irr) }}</div>
                </div>
                <div class="metric-card" v-if="analysisResult.deal_summary.deal_roe != null">
                  <div class="metric-label">Deal ROE</div>
                  <div class="metric-value">{{ fmtPct(analysisResult.deal_summary.deal_roe) }}</div>
                </div>
                <div class="metric-card" v-if="analysisResult.deal_summary.deal_moic != null">
                  <div class="metric-label">Deal MOIC</div>
                  <div class="metric-value">{{ fmtDec(analysisResult.deal_summary.deal_moic, 2) }}x</div>
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
                    <tr v-for="p in analysisResult.partner_results" :key="p.partner || p.investor_id"
                        :class="{ 'pe-row': p.is_pref_equity || p.is_pe }">
                      <td class="partner-name">{{ p.partner || p.investor_id }}</td>
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
                      <td class="r"><strong>{{ fmtCurrency(analysisResult.deal_summary.total_contributions) }}</strong></td>
                      <td class="r"><strong>{{ fmtCurrency(analysisResult.deal_summary.total_cf_distributions) }}</strong></td>
                      <td class="r"><strong>{{ fmtCurrency(analysisResult.deal_summary.total_cap_distributions) }}</strong></td>
                      <td class="r"><strong>{{ fmtCurrency(analysisResult.deal_summary.total_distributions) }}</strong></td>
                      <td class="r"><strong>{{ analysisResult.deal_summary.deal_irr != null ? fmtPct(analysisResult.deal_summary.deal_irr) : '—' }}</strong></td>
                      <td class="r"><strong>{{ analysisResult.deal_summary.deal_roe != null ? fmtPct(analysisResult.deal_summary.deal_roe) : '—' }}</strong></td>
                      <td class="r"><strong>{{ analysisResult.deal_summary.deal_moic != null ? fmtDec(analysisResult.deal_summary.deal_moic, 2) + 'x' : '—' }}</strong></td>
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
.analysis-layout { display: flex; flex: 1; overflow: hidden; }
.setup-panel {
  width: 480px; min-width: 440px; max-width: 540px;
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
.section-header.clickable { cursor: pointer; user-select: none; }
.expandable .section-header { cursor: pointer; user-select: none; }
.chevron { font-size: 11px; color: #999; }
.section-hint { font-size: 10px; font-weight: 400; color: #999; margin-left: 4px; }

/* Info grid */
.info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 12px; font-size: 12px; }

/* Form grids */
.form-grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; }
.form-group label { display: block; font-size: 11px; font-weight: 500; color: #666; margin-bottom: 2px; }
.form-group input, .form-group select {
  width: 100%; padding: 5px 8px; border: 1px solid #ccc;
  border-radius: 4px; font-size: 12px;
}

/* ========== Capital Budget ========== */
.budget-subsection { margin-bottom: 12px; }
.budget-title {
  font-size: 11px; font-weight: 600; color: #555;
  text-transform: uppercase; letter-spacing: 0.5px;
  margin: 0 0 6px; padding-bottom: 4px;
  border-bottom: 1px solid #eee;
}

.budget-table {
  width: 100%; border-collapse: collapse; font-size: 12px;
  font-variant-numeric: tabular-nums;
}
.budget-table th {
  font-size: 10px; font-weight: 600; color: #888;
  text-transform: uppercase; letter-spacing: 0.3px;
  padding: 3px 6px; border-bottom: 1px solid #ddd;
  text-align: left;
}
.budget-table td {
  padding: 4px 6px; border-bottom: 1px solid #f0f0f0;
  vertical-align: middle;
}
.budget-table.compact td { padding: 3px 6px; }

.col-pct { width: 72px; }
.col-amt { width: 110px; }
.col-action { width: 24px; }
.col-num { width: 24px; text-align: center; color: #999; }
.col-rate { width: 100px; }

.inline-label-input {
  border: none; border-bottom: 1px dashed #ccc;
  background: transparent; font-size: 12px;
  padding: 0 2px; width: 100%;
  outline: none;
}
.inline-label-input:focus { border-bottom-color: #1976d2; }

/* Unified formatted inputs — match computed value styling */
.fmt-input {
  padding: 2px 4px; border: 1px solid transparent;
  border-radius: 3px; text-align: right;
  font-size: 12px; font-weight: 500; color: #333;
  background: transparent;
  font-variant-numeric: tabular-nums;
  outline: none;
}
.fmt-input:hover { border-color: #ccc; }
.fmt-input:focus {
  border-color: #1976d2; background: #fff;
  box-shadow: 0 0 0 1px #1976d2;
}
.fmt-pct { width: 66px; }
.fmt-amt { width: 110px; }

.fmt-display { font-size: 12px; font-weight: 500; color: #333; }

/* Legacy compat for waterfall rate inputs */
.pct-input {
  width: 60px; padding: 2px 4px; border: 1px solid #ccc;
  border-radius: 3px; font-size: 12px; text-align: right;
}
.amt-input {
  width: 100px; padding: 2px 4px; border: 1px solid #ccc;
  border-radius: 3px; font-size: 12px; text-align: right;
}
.fixed-row { background: #f8f9fa; }

.sub-metrics {
  display: block; font-size: 10px; color: #777;
  margin-top: 1px; font-weight: 400;
}
.sub-metrics strong { color: #333; }

.total-row td {
  border-top: 2px solid #333;
  padding-top: 6px; font-size: 12px;
}
.subtotal-row td {
  border-top: 1px solid #999;
  padding-top: 4px; font-size: 12px;
}

.mismatch-warn { color: #e65100; font-size: 10px; font-weight: 500; }

.level-badge {
  font-size: 9px; background: #e3f2fd; color: #1565c0;
  padding: 1px 5px; border-radius: 3px; font-weight: 500;
}
.level-select {
  font-size: 10px; padding: 1px 2px; border: 1px solid #ccc;
  border-radius: 3px; background: #fff; width: 68px;
}
.prop-select {
  display: block; font-size: 10px; padding: 1px 2px;
  border: 1px solid #ccc; border-radius: 3px;
  background: #fff; width: 100%; margin-top: 2px;
}

.mt-4 { margin-top: 4px; }

/* Balance check */
.balance-check {
  margin-top: 8px; padding: 8px 10px;
  border-radius: 4px; font-size: 12px;
  background: #f5f5f5; border: 1px solid #e0e0e0;
}
.balance-check.balanced { background: #e8f5e9; border-color: #a5d6a7; }
.balance-check.unbalanced { background: #fff3e0; border-color: #ffcc80; }
.balance-row { display: flex; justify-content: space-between; padding: 2px 0; }
.balance-status {
  text-align: center; font-weight: 600; margin-top: 4px; padding-top: 4px;
  border-top: 1px solid rgba(0,0,0,0.1);
}
.balanced .balance-status { color: #2e7d32; }
.unbalanced .balance-status { color: #e65100; }

/* ========== Waterfall Builder ========== */
.wf-tabs { display: flex; gap: 4px; margin-bottom: 8px; }
.wf-tabs button {
  padding: 4px 12px; border: 1px solid #ccc; border-radius: 4px;
  background: #f5f5f5; font-size: 12px; cursor: pointer;
}
.wf-tabs button.active { background: #1976d2; color: #fff; border-color: #1976d2; }
.wf-tabs button:disabled { opacity: 0.4; cursor: default; }

.wf-table td { padding: 4px 4px; }
.wf-select {
  width: 100%; padding: 3px 4px; border: 1px solid #ccc;
  border-radius: 3px; font-size: 11px; background: #fff;
}
.rate-input {
  width: 56px; padding: 3px 4px; border: 1px solid #ccc;
  border-radius: 3px; font-size: 11px; text-align: right;
}
.rate-suffix { font-size: 11px; color: #888; margin-left: 2px; }
.muted { color: #ccc; }

.wf-add-row { display: flex; gap: 6px; margin: 6px 0; }
.wf-warning {
  color: #e65100; font-size: 11px; padding: 4px 8px;
  background: #fff3e0; border-radius: 3px; margin: 6px 0;
}
.wf-actions { display: flex; gap: 8px; margin-top: 8px; align-items: center; }

.wf-type-section { margin-bottom: 16px; border: 1px solid #e0e0e0; border-radius: 6px; padding: 10px; background: #fafafa; }
.wf-builder-section { margin-bottom: 14px; border: 1px solid #e0e0e0; border-radius: 6px; padding: 10px; background: #fafafa; }
.wf-builder-header { margin-bottom: 6px; display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; }
.wf-builder-header h5 { font-size: 13px; font-weight: 700; margin: 0; color: #333; }
.wf-type-header { margin-bottom: 6px; }
.wf-type-header h5 { font-size: 13px; font-weight: 700; margin: 0; color: #333; }
.wf-type-desc { font-size: 11px; color: #777; }
.step-badge {
  display: inline-block; font-size: 10px; font-weight: 600; padding: 1px 6px;
  border-radius: 3px; text-transform: uppercase;
}
.step-pref { background: #e8f5e9; color: #2e7d32; }
.step-initial { background: #e3f2fd; color: #1565c0; }
.step-share { background: #fff3e0; color: #e65100; }
.step-tag { background: #fce4ec; color: #c62828; }
.wf-explanation {
  font-size: 11px; color: #666; background: #f5f5f5; padding: 8px 10px;
  border-radius: 4px; margin-top: 8px; line-height: 1.5;
}
.wf-explanation strong { color: #333; }

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
.su-subtotal { border-top: 1px solid #bbb; margin-top: 4px; padding-top: 4px; font-weight: 500; }
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

/* ========== Sale Proceeds Summary ========== */
.sale-proceeds-summary {
  margin-top: 10px; padding-top: 8px;
  border-top: 1px solid #eee;
}
.proceeds-table {
  width: 100%; border-collapse: collapse; font-size: 12px;
  font-variant-numeric: tabular-nums;
}
.proceeds-table td {
  padding: 3px 6px; border-bottom: 1px solid #f0f0f0;
}
.proceeds-table .neg { color: #d32f2f; }
.proceeds-table .total-row td {
  border-top: 2px solid #333; padding-top: 5px; border-bottom: none;
}
.proceeds-table .subtotal-row td {
  border-top: 1px solid #999; padding-top: 4px; font-weight: 500;
}
.pe-row-light { background: #f0f4f8; }
.partner-proceeds { margin-top: 4px; }
.sale-meta {
  font-size: 10px; color: #888; margin-top: 6px;
  padding-top: 4px; border-top: 1px solid #f0f0f0;
}

/* ========== Term Sheet Table ========== */
.terms-table {
  width: 100%; border-collapse: collapse; font-size: 12px;
}
.terms-table td {
  padding: 5px 6px; border-bottom: 1px solid #f0f0f0;
  vertical-align: top;
}
.terms-label {
  font-weight: 600; color: #555; white-space: nowrap;
  width: 120px; padding-right: 8px;
}
.terms-text {
  width: 100%; padding: 3px 6px; border: 1px solid #ddd;
  border-radius: 3px; font-size: 12px; background: #fff;
}
.terms-textarea {
  width: 100%; padding: 4px 6px; border: 1px solid #ddd;
  border-radius: 3px; font-size: 11px; resize: vertical;
  font-family: inherit; background: #fff;
}
.terms-select-sm {
  padding: 2px 4px; border: 1px solid #ddd; border-radius: 3px;
  font-size: 11px; background: #fff;
}
.terms-num-sm {
  width: 52px; padding: 2px 4px; border: 1px solid #ddd;
  border-radius: 3px; font-size: 11px; text-align: right;
}
.terms-num-xs {
  width: 32px; padding: 2px 4px; border: 1px solid #ddd;
  border-radius: 3px; font-size: 11px; text-align: center;
}
.rate-builder {
  display: flex; align-items: center; gap: 4px; flex-wrap: wrap;
}
.rate-desc { font-size: 11px; color: #888; white-space: nowrap; }
.terms-inline {
  display: flex; align-items: center; gap: 4px; flex-wrap: wrap;
}
.terms-sub {
  margin-top: 4px;
}
.terms-sub label {
  display: block; font-size: 10px; color: #888; margin-bottom: 2px;
}
.constraints-grid {
  display: grid; grid-template-columns: 1fr 1fr; gap: 6px 12px;
}
.constraint-item label {
  display: block; font-size: 10px; font-weight: 500; color: #888; margin-bottom: 1px;
}

/* Debug */
.debug-section { margin-top: 12px; }
.debug-section summary { cursor: pointer; font-size: 12px; color: #666; }
.debug-list { font-size: 11px; color: #666; padding-left: 20px; }
</style>
