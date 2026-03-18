import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import api from '../api/client'

// ============================================================
// Interfaces
// ============================================================

export interface PartnerResult {
  partner: string
  is_pref_equity: boolean
  contributions: number
  cf_distributions: number
  cap_distributions: number
  total_distributions: number
  irr: number | null
  roe: number | null
  moic: number | null
  unrealized_nav: number
  capital_outstanding: number
  pref_unpaid_compounded: number
  pref_accrued_current_year: number
  [key: string]: any
}

export interface DealSummary {
  deal_irr: number | null
  deal_roe: number | null
  deal_moic: number | null
  total_contributions: number
  total_cf_distributions: number
  total_cap_distributions: number
  total_distributions: number
  [key: string]: any
}

export interface DealHeader {
  metadata: Record<string, any>
  cap_data: Record<string, any>
  consolidation: Record<string, any>
  sub_portfolio_msg: string | null
}

export interface LoanInfo {
  vcode: string
  loan_id: string
  orig_amount: number
  orig_date: string | null
  maturity_date: string | null
  int_type: string
  fixed_rate: number
  loan_term_m: number
  amort_term_m: number
  io_months: number
}

export interface DebtServiceData {
  loans: LoanInfo[]
  loan_schedule: Record<string, any>[]
  sale_proceeds: Record<string, any> | null
}

export interface CashData {
  schedule: Record<string, any>[]
  summary: Record<string, any>
  beginning_cash: number
}

export interface AnnualForecast {
  rows: Array<{ label: string; values: Record<string, number | null> }>
  years: number[]
}

export interface RoePartnerAudit {
  partner: string
  timeline: Record<string, any>[]
  cf_distributions: Record<string, any>[]
  summary: Record<string, any>
}

export interface RoeAudit {
  partners: RoePartnerAudit[]
  deal_level: {
    timeline: Record<string, any>[]
    cf_distributions: Record<string, any>[]
    summary: Record<string, any>
  }
}

export interface MoicPartnerAudit {
  partner: string
  breakdown: Record<string, any>[]
  summary: Record<string, any>
}

export interface MoicAudit {
  partners: MoicPartnerAudit[]
  deal_level: Record<string, any>
}

export interface XirrCashflows {
  [partner: string]: Array<{ date: string; amount: number }>
}

export interface ProspectiveLoan {
  id: number
  vcode: string
  loan_name: string | null
  status: string
  refi_date: string
  existing_loan_id: string | null
  loan_amount: number | null
  lender_uw_noi: number | null
  max_ltv: number | null
  min_dscr: number | null
  min_debt_yield: number | null
  interest_rate: number | null
  rate_spread_bps: number | null
  rate_index: string | null
  term_years: number | null
  amort_years: number | null
  io_years: number | null
  int_type: string | null
  closing_costs: number | null
  reserve_holdback: number | null
  notes: string | null
  created_at: string | null
  [key: string]: any
}

export interface SizingResult {
  system_noi_12m: number
  lender_uw_noi: number
  noi_difference: number
  noi_diff_pct: number
  constraints: Record<string, any>
  binding_constraint: string
  estimated_loan_amount: number
  existing_loan_balance: number
  closing_costs: number
  reserve_holdback: number
  net_refi_proceeds: number
  distributable_proceeds: number
  capital_call_required: boolean
  capital_call_amount: number
  new_maturity_date: string | null
  [key: string]: any
}

export interface RawCapitalCall {
  id: number
  Vcode: string
  PropCode: string
  CallDate: string
  Amount: number
  CallType: string
  FundingSource: string
  Notes: string
  Typename: string
}

// ============================================================
// Store
// ============================================================

export const useDealsStore = defineStore('deals', () => {
  const currentVcode = ref<string>('')
  const computing = ref(false)
  const error = ref<string | null>(null)

  // Core compute results (keyed by vcode)
  const partnerResults = ref<Record<string, PartnerResult[]>>({})
  const dealSummaries = ref<Record<string, DealSummary>>({})
  const debugMsgs = ref<Record<string, string[]>>({})

  // Lazy-loaded sections (keyed by vcode)
  const headers = ref<Record<string, DealHeader>>({})
  const forecasts = ref<Record<string, AnnualForecast>>({})
  const debtService = ref<Record<string, DebtServiceData>>({})
  const cashData = ref<Record<string, CashData>>({})
  const capitalCalls = ref<Record<string, any[]>>({})
  const xirrCashflows = ref<Record<string, XirrCashflows>>({})
  const roeAudits = ref<Record<string, RoeAudit>>({})
  const moicAudits = ref<Record<string, MoicAudit>>({})

  // Prospective loans
  const prospectiveLoans = ref<Record<string, ProspectiveLoan[]>>({})
  const sizingResults = ref<Record<number, SizingResult>>({})

  // Raw capital calls (for editing)
  const rawCapitalCalls = ref<Record<string, RawCapitalCall[]>>({})

  // Refi info from last compute
  const refiDbg = ref<Record<string, any>>({})
  const refiCapitalCallRequired = ref<Record<string, boolean>>({})
  const refiCapitalCallAmount = ref<Record<string, number>>({})

  // Section loading states
  const loadingSection = ref<string | null>(null)

  // Computed
  const currentPartners = computed(() => partnerResults.value[currentVcode.value] || [])
  const currentSummary = computed(() => dealSummaries.value[currentVcode.value] || null)
  const currentHeader = computed(() => headers.value[currentVcode.value] || null)
  const currentForecast = computed(() => forecasts.value[currentVcode.value] || null)
  const currentDebt = computed(() => debtService.value[currentVcode.value] || null)
  const currentCash = computed(() => cashData.value[currentVcode.value] || null)
  const currentCapCalls = computed(() => capitalCalls.value[currentVcode.value] || null)
  const currentXirr = computed(() => xirrCashflows.value[currentVcode.value] || null)
  const currentRoe = computed(() => roeAudits.value[currentVcode.value] || null)
  const currentMoic = computed(() => moicAudits.value[currentVcode.value] || null)
  const currentDebugMsgs = computed(() => debugMsgs.value[currentVcode.value] || [])
  const hasResult = computed(() => !!dealSummaries.value[currentVcode.value])
  const currentProspectiveLoans = computed(() => prospectiveLoans.value[currentVcode.value] || [])
  const currentRawCapitalCalls = computed(() => rawCapitalCalls.value[currentVcode.value] || [])
  const currentRefiDbg = computed(() => refiDbg.value[currentVcode.value] || null)
  const currentRefiCapCallRequired = computed(() => refiCapitalCallRequired.value[currentVcode.value] || false)
  const currentRefiCapCallAmount = computed(() => refiCapitalCallAmount.value[currentVcode.value] || 0)

  // ============================================================
  // Actions
  // ============================================================

  function _clearLazySections(vcode: string) {
    // Clear lazy-loaded sections so they reload with fresh data
    delete forecasts.value[vcode]
    delete debtService.value[vcode]
    delete cashData.value[vcode]
    delete capitalCalls.value[vcode]
    delete xirrCashflows.value[vcode]
    delete roeAudits.value[vcode]
    delete moicAudits.value[vcode]
  }

  async function computeDeal(vcode: string, force = false) {
    computing.value = true
    error.value = null
    currentVcode.value = vcode
    try {
      // Fire compute + header in parallel
      const [compRes, headerRes] = await Promise.all([
        api.post('/api/deals/compute', { vcode, force }),
        api.get(`/api/deals/${vcode}/header`),
      ])
      partnerResults.value[vcode] = compRes.data.partner_results
      dealSummaries.value[vcode] = compRes.data.deal_summary
      debugMsgs.value[vcode] = compRes.data.debug_msgs || []
      headers.value[vcode] = headerRes.data

      // Store refi info
      refiDbg.value[vcode] = compRes.data.refi_dbg || null
      refiCapitalCallRequired.value[vcode] = compRes.data.refi_capital_call_required || false
      refiCapitalCallAmount.value[vcode] = compRes.data.refi_capital_call_amount || 0

      // Clear lazy sections so they reload fresh
      _clearLazySections(vcode)
    } catch (e: any) {
      error.value = e.response?.data?.error || e.message
    } finally {
      computing.value = false
    }
  }

  async function loadForecast(vcode: string) {
    if (forecasts.value[vcode]) return
    loadingSection.value = 'forecast'
    try {
      const res = await api.get(`/api/deals/${vcode}/annual-forecast`)
      forecasts.value[vcode] = res.data
    } finally { loadingSection.value = null }
  }

  async function loadDebtService(vcode: string) {
    if (debtService.value[vcode]) return
    loadingSection.value = 'debt'
    try {
      const res = await api.get(`/api/deals/${vcode}/debt-service`)
      debtService.value[vcode] = res.data
    } finally { loadingSection.value = null }
  }

  async function loadCashSchedule(vcode: string) {
    if (cashData.value[vcode]) return
    loadingSection.value = 'cash'
    try {
      const res = await api.get(`/api/deals/${vcode}/cash-schedule`)
      cashData.value[vcode] = res.data
    } finally { loadingSection.value = null }
  }

  async function loadCapitalCalls(vcode: string) {
    if (capitalCalls.value[vcode]) return
    loadingSection.value = 'capcalls'
    try {
      const res = await api.get(`/api/deals/${vcode}/capital-calls`)
      capitalCalls.value[vcode] = res.data.capital_calls
    } finally { loadingSection.value = null }
  }

  async function loadXirrCashflows(vcode: string) {
    if (xirrCashflows.value[vcode]) return
    loadingSection.value = 'xirr'
    try {
      const res = await api.get(`/api/deals/${vcode}/xirr-cashflows`)
      xirrCashflows.value[vcode] = res.data.cashflows
    } finally { loadingSection.value = null }
  }

  async function loadRoeAudit(vcode: string) {
    if (roeAudits.value[vcode]) return
    loadingSection.value = 'roe'
    try {
      const res = await api.get(`/api/deals/${vcode}/roe-audit`)
      roeAudits.value[vcode] = res.data
    } finally { loadingSection.value = null }
  }

  async function loadMoicAudit(vcode: string) {
    if (moicAudits.value[vcode]) return
    loadingSection.value = 'moic'
    try {
      const res = await api.get(`/api/deals/${vcode}/moic-audit`)
      moicAudits.value[vcode] = res.data
    } finally { loadingSection.value = null }
  }

  // ============================================================
  // Prospective Loans
  // ============================================================

  async function loadProspectiveLoans(vcode: string) {
    try {
      const res = await api.get(`/api/deals/${vcode}/prospective-loans`)
      prospectiveLoans.value[vcode] = res.data.loans
    } catch { /* ignore */ }
  }

  async function createProspectiveLoan(vcode: string, data: Record<string, any>) {
    const res = await api.post(`/api/deals/${vcode}/prospective-loans`, data)
    await loadProspectiveLoans(vcode)
    return res.data
  }

  async function updateProspectiveLoan(vcode: string, loanId: number, data: Record<string, any>) {
    await api.put(`/api/deals/${vcode}/prospective-loans/${loanId}`, data)
    await loadProspectiveLoans(vcode)
  }

  async function deleteProspectiveLoan(vcode: string, loanId: number) {
    const loan = (prospectiveLoans.value[vcode] || []).find(l => l.id === loanId)
    await api.delete(`/api/deals/${vcode}/prospective-loans/${loanId}`)
    await loadProspectiveLoans(vcode)
    // Only recompute if we deleted an accepted loan
    if (loan && loan.status === 'accepted') {
      await computeDeal(vcode, true)
    }
  }

  async function acceptProspectiveLoan(vcode: string, loanId: number) {
    const res = await api.post(`/api/deals/${vcode}/prospective-loans/${loanId}/accept`)
    await loadProspectiveLoans(vcode)
    // Recompute to get fresh results with the new loan
    await computeDeal(vcode, true)
    return res.data
  }

  async function revertProspectiveLoan(vcode: string, loanId: number) {
    await api.post(`/api/deals/${vcode}/prospective-loans/${loanId}/revert`)
    await loadProspectiveLoans(vcode)
    await computeDeal(vcode, true)
  }

  async function runSizingAnalysis(vcode: string, loanId: number) {
    const res = await api.get(`/api/deals/${vcode}/prospective-loans/${loanId}/sizing`)
    sizingResults.value[loanId] = res.data
    return res.data
  }

  // ============================================================
  // Raw Capital Calls CRUD
  // ============================================================

  async function loadRawCapitalCalls(vcode: string) {
    try {
      const res = await api.get(`/api/deals/${vcode}/raw-capital-calls`)
      rawCapitalCalls.value[vcode] = res.data.capital_calls
    } catch { /* ignore */ }
  }

  async function createRawCapitalCall(vcode: string, data: Record<string, any>) {
    await api.post(`/api/deals/${vcode}/raw-capital-calls`, data)
    await loadRawCapitalCalls(vcode)
  }

  async function updateRawCapitalCall(vcode: string, callId: number, data: Record<string, any>) {
    await api.put(`/api/deals/${vcode}/raw-capital-calls/${callId}`, data)
    await loadRawCapitalCalls(vcode)
  }

  async function deleteRawCapitalCall(vcode: string, callId: number) {
    await api.delete(`/api/deals/${vcode}/raw-capital-calls/${callId}`)
    await loadRawCapitalCalls(vcode)
  }

  function selectDeal(vcode: string) {
    currentVcode.value = vcode
  }

  function clearAll() {
    partnerResults.value = {}
    dealSummaries.value = {}
    debugMsgs.value = {}
    headers.value = {}
    forecasts.value = {}
    debtService.value = {}
    cashData.value = {}
    capitalCalls.value = {}
    xirrCashflows.value = {}
    roeAudits.value = {}
    moicAudits.value = {}
    prospectiveLoans.value = {}
    sizingResults.value = {}
    rawCapitalCalls.value = {}
    refiDbg.value = {}
    refiCapitalCallRequired.value = {}
    refiCapitalCallAmount.value = {}
    currentVcode.value = ''
  }

  return {
    currentVcode, computing, error, loadingSection,
    partnerResults, dealSummaries, debugMsgs,
    headers, forecasts, debtService, cashData, capitalCalls,
    xirrCashflows, roeAudits, moicAudits,
    prospectiveLoans, sizingResults, rawCapitalCalls,
    refiDbg, refiCapitalCallRequired, refiCapitalCallAmount,
    // Computed
    currentPartners, currentSummary, currentHeader, currentForecast,
    currentDebt, currentCash, currentCapCalls, currentXirr,
    currentRoe, currentMoic, currentDebugMsgs, hasResult,
    currentProspectiveLoans, currentRawCapitalCalls,
    currentRefiDbg, currentRefiCapCallRequired, currentRefiCapCallAmount,
    // Actions
    computeDeal, loadForecast, loadDebtService, loadCashSchedule,
    loadCapitalCalls, loadXirrCashflows, loadRoeAudit, loadMoicAudit,
    selectDeal, clearAll,
    // Prospective loans
    loadProspectiveLoans, createProspectiveLoan, updateProspectiveLoan,
    deleteProspectiveLoan, acceptProspectiveLoan, revertProspectiveLoan,
    runSizingAnalysis,
    // Raw capital calls
    loadRawCapitalCalls, createRawCapitalCall, updateRawCapitalCall,
    deleteRawCapitalCall,
  }
})
