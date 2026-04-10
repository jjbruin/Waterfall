import { defineStore } from 'pinia'
import { ref } from 'vue'
import api from '../api/client'

type LogLevel = 'log' | 'warn' | 'error'
function log(level: LogLevel, ...args: any[]) {
  console[level]('[dashboard]', ...args)
}

interface KpiData {
  portfolio_value: number
  debt_outstanding: number
  wtd_avg_cap_rate: number
  portfolio_occupancy: number
  property_count: number
  total_pref_equity: number
}

interface CapStructure {
  debt_m: number
  pref_m: number
  partner_m: number
  avg_ltv: number
  pref_exposure: number
}

interface NoiData {
  periods: string[]
  actual_noi: (number | null)[]
  uw_noi: (number | null)[]
  occupancy: (number | null)[]
  frequency: string
}

interface OccByType {
  data: Array<{ asset_type: string; occupancy: number; above_avg: boolean }>
  portfolio_avg: number
}

interface AssetAlloc {
  asset_type: string
  pref_equity: number
  pct: number
  count: number
}

interface LoanMaturity {
  yearly: Array<{ year: string; rate_type: string; amount: number }>
  fixed_rates: Array<{ year: string; avg_rate: number }>
  detail: Array<Record<string, any>>
}

interface DealReturn {
  vcode: string
  name: string
  contributions: number
  cf_distributions: number
  cap_distributions: number
  irr: number | null
  roe: number | null
  moic: number | null
}

export const useDashboardStore = defineStore('dashboard', () => {
  const kpis = ref<KpiData | null>(null)
  const capStructure = ref<CapStructure | null>(null)
  const noiData = ref<NoiData | null>(null)
  const occByType = ref<OccByType | null>(null)
  const assetAlloc = ref<AssetAlloc[]>([])
  const loanMaturities = ref<LoanMaturity | null>(null)
  const returns = ref<DealReturn[]>([])
  const returnErrors = ref<any[]>([])
  const loading = ref(false)
  const computingReturns = ref(false)
  const noiFreq = ref('Quarterly')

  const loadError = ref<string | null>(null)
  const initProgress = ref<{ current: number; total: number; deal: string } | null>(null)

  /**
   * Phase 1: stream progress while server computes caps for all deals.
   * Phase 2: fire 6 parallel data requests (caps now cached server-side).
   */
  async function loadAll() {
    loading.value = true
    loadError.value = null
    initProgress.value = null

    try {
      // Phase 1 — SSE init-stream (pre-computes caps with progress)
      log('log', 'loadAll() phase 1 — init-stream SSE')
      await initCaps()

      // Phase 2 — parallel data fetch (caps cached, all fast now)
      log('log', 'loadAll() phase 2 — 6 parallel requests')
      initProgress.value = null
      await fetchDashboardData()
      log('log', 'loadAll() complete')
    } catch (err: any) {
      const msg = err?.response?.data?.error || err?.message || 'Unknown error'
      log('warn', 'loadAll() first attempt failed, retrying:', msg)
      // Auto-retry once — the first attempt may have warmed the server cache
      try {
        await fetchDashboardData()
        log('log', 'loadAll() retry succeeded')
      } catch (retryErr: any) {
        const retryMsg = retryErr?.response?.data?.error || retryErr?.message || 'Unknown error'
        log('error', 'loadAll() retry also failed:', retryMsg, retryErr)
        loadError.value = `Dashboard load failed: ${retryMsg}`
      }
    } finally {
      loading.value = false
      initProgress.value = null
    }
  }

  async function fetchDashboardData() {
    const [kpiRes, capRes, noiRes, occRes, allocRes, loanRes] = await Promise.all([
      api.get('/api/dashboard/kpis'),
      api.get('/api/dashboard/capitalization'),
      api.get('/api/dashboard/noi-trend', { params: { freq: noiFreq.value } }),
      api.get('/api/dashboard/occupancy-by-type'),
      api.get('/api/dashboard/asset-allocation'),
      api.get('/api/dashboard/loan-maturities'),
    ])
    kpis.value = kpiRes.data
    capStructure.value = capRes.data
    noiData.value = noiRes.data
    occByType.value = occRes.data
    assetAlloc.value = allocRes.data.allocation
    loanMaturities.value = loanRes.data
  }

  /** Connect to SSE init-stream endpoint. Resolves when done. */
  function initCaps(): Promise<void> {
    return new Promise((resolve, reject) => {
      const token = localStorage.getItem('token')
      const url = `/api/dashboard/init-stream?token=${encodeURIComponent(token || '')}`
      const es = new EventSource(url)

      es.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data)
          if (msg.done) {
            es.close()
            log('log', `init-stream done — ${msg.total} deals processed${msg.cached ? ' (cached)' : ''}`)
            resolve()
          } else {
            initProgress.value = { current: msg.current, total: msg.total, deal: msg.deal }
          }
        } catch (e) {
          log('warn', 'SSE parse error', e)
        }
      }

      es.onerror = () => {
        es.close()
        log('warn', 'SSE connection error — falling back to direct load')
        // Fall back: just resolve and let the parallel requests compute caps
        resolve()
      }
    })
  }

  async function loadNoi(freq: string) {
    noiFreq.value = freq
    const res = await api.get('/api/dashboard/noi-trend', { params: { freq } })
    noiData.value = res.data
  }

  async function computeReturns() {
    computingReturns.value = true
    try {
      const res = await api.post('/api/dashboard/computed-returns')
      returns.value = res.data.returns
      returnErrors.value = res.data.errors
    } finally {
      computingReturns.value = false
    }
  }

  function clearAll() {
    kpis.value = null
    capStructure.value = null
    noiData.value = null
    occByType.value = null
    assetAlloc.value = []
    loanMaturities.value = null
    returns.value = []
    returnErrors.value = []
  }

  return {
    kpis, capStructure, noiData, occByType, assetAlloc, loanMaturities,
    returns, returnErrors, loading, loadError, initProgress, computingReturns, noiFreq,
    loadAll, loadNoi, computeReturns, clearAll,
  }
})
