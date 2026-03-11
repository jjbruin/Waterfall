import { defineStore } from 'pinia'
import { ref } from 'vue'
import api from '../api/client'

interface PsckocDeal {
  vcode: string
  name: string
  asset_type?: string
  sale_status?: string
  ppi_entity?: string
  psckoc_pct?: number
}

interface PartnerReturn {
  partner: string
  contributions: number
  cf_distributions: number
  cap_distributions: number
  total_distributions: number
  irr: number | null
  roe: number | null
  moic: number
  combined_cashflows: Array<{ date: string; amount: number }>
}

interface PsckocResults {
  income_schedule: any[]
  member_allocations: any[]
  partner_results: PartnerReturn[]
  deal_summary: {
    deal_irr: number | null
    deal_moic: number
    total_contributions: number
    total_distributions: number
  }
  am_fee_schedule: any[]
  deal_vcodes: string[]
  deals_computed: number
  errors: string[]
}

export const usePsckocStore = defineStore('psckoc', () => {
  const deals = ref<PsckocDeal[]>([])
  const results = ref<PsckocResults | null>(null)
  const computing = ref(false)
  const loadingDeals = ref(false)

  async function loadDeals() {
    loadingDeals.value = true
    try {
      const res = await api.get('/api/psckoc/deals')
      deals.value = res.data.deals
    } finally {
      loadingDeals.value = false
    }
  }

  async function compute() {
    computing.value = true
    try {
      const res = await api.post('/api/psckoc/compute')
      results.value = res.data
    } finally {
      computing.value = false
    }
  }

  async function loadCachedResults() {
    try {
      const res = await api.get('/api/psckoc/results')
      results.value = res.data
    } catch {
      // No cached results — that's fine
    }
  }

  function clearAll() {
    deals.value = []
    results.value = null
  }

  return { deals, results, computing, loadingDeals, loadDeals, compute, loadCachedResults, clearAll }
})
