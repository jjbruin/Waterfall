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

  // ============================================================
  // Actions
  // ============================================================

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
    currentVcode.value = ''
  }

  return {
    currentVcode, computing, error, loadingSection,
    partnerResults, dealSummaries, debugMsgs,
    headers, forecasts, debtService, cashData, capitalCalls,
    xirrCashflows, roeAudits, moicAudits,
    // Computed
    currentPartners, currentSummary, currentHeader, currentForecast,
    currentDebt, currentCash, currentCapCalls, currentXirr,
    currentRoe, currentMoic, currentDebugMsgs, hasResult,
    // Actions
    computeDeal, loadForecast, loadDebtService, loadCashSchedule,
    loadCapitalCalls, loadXirrCashflows, loadRoeAudit, loadMoicAudit,
    selectDeal, clearAll,
  }
})
