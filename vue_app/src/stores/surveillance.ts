import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import api from '../api/client'

export interface SurveillanceRow {
  vcode: string
  name: string
  asset_type: string
  city: string
  units: number | null
  partner: string
  lifecycle: string
  portfolio_name: string
  // Live data
  occ_pct: number | null
  occ_period: string | null
  noi_monthly: number | null
  revenue_monthly: number | null
  fin_period: string | null
  // Loan data
  loan_balance: number | null
  loan_rate: number | null
  maturity_date: string | null
  loan_type: string | null
  // Editable surveillance fields
  dscr_val: number | null
  dscr_min: number | null
  dy_val: number | null
  dy_min: number | null
  ltv_val: number | null
  ltv_min: number | null
  working_capital: number | null
  tax_due: string | null
  ins_renewal: string | null
  tenant_exp: string | null
  comments: string | null
  updated_at: string | null
  // Insurance
  has_property_ins: boolean
  has_gl_ins: boolean
  // Computed
  flagged: boolean
}

export interface SurveillanceDashboard {
  total: number
  total_debt: number | null
  avg_occ: number | null
  total_noi_monthly: number | null
  flagged: number
  maturing_12mo: number
  by_type: Record<string, number>
}

export interface InsuranceRecord {
  id?: number
  vcode: string
  ins_type: string
  carrier: string | null
  policy_number: string | null
  expiration_date: string | null
  coverage_amount: number | null
  notes: string | null
}

export const useSurveillanceStore = defineStore('surveillance', () => {
  const rows = ref<SurveillanceRow[]>([])
  const dashboard = ref<SurveillanceDashboard | null>(null)
  const insurance = ref<InsuranceRecord[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)

  // Filters
  const searchQuery = ref('')
  const flagFilter = ref('')
  const assetTypeFilter = ref('')

  const assetTypes = computed(() => {
    const types = new Set(rows.value.map(r => r.asset_type).filter(Boolean))
    return Array.from(types).sort()
  })

  const filteredRows = computed(() => {
    let result = rows.value
    if (searchQuery.value) {
      const q = searchQuery.value.toLowerCase()
      result = result.filter(r =>
        r.name?.toLowerCase().includes(q) || r.vcode.toLowerCase().includes(q)
      )
    }
    if (flagFilter.value) {
      if (flagFilter.value === 'flagged') {
        result = result.filter(r => r.flagged)
      } else if (flagFilter.value === 'clear') {
        result = result.filter(r => !r.flagged)
      }
    }
    if (assetTypeFilter.value) {
      result = result.filter(r => r.asset_type === assetTypeFilter.value)
    }
    return result
  })

  async function loadTable() {
    loading.value = true
    error.value = null
    try {
      const res = await api.get('/api/surveillance/')
      rows.value = res.data
    } catch (e: any) {
      error.value = e.response?.data?.error || e.message
    } finally {
      loading.value = false
    }
  }

  async function loadDashboard() {
    try {
      const res = await api.get('/api/surveillance/dashboard')
      dashboard.value = res.data
    } catch (e: any) {
      error.value = e.response?.data?.error || e.message
    }
  }

  async function updateProperty(vcode: string, fields: Record<string, any>) {
    try {
      await api.patch(`/api/surveillance/${vcode}`, fields)
      const row = rows.value.find(r => r.vcode === vcode)
      if (row) Object.assign(row, fields)
    } catch (e: any) {
      throw new Error(e.response?.data?.error || e.message)
    }
  }

  async function loadInsurance() {
    try {
      const res = await api.get('/api/surveillance/insurance')
      insurance.value = res.data
    } catch (e: any) {
      error.value = e.response?.data?.error || e.message
    }
  }

  async function saveInsurance(record: Partial<InsuranceRecord>) {
    try {
      await api.post('/api/surveillance/insurance', record)
      await loadInsurance()
    } catch (e: any) {
      throw new Error(e.response?.data?.error || e.message)
    }
  }

  async function deleteInsurance(id: number) {
    try {
      await api.delete(`/api/surveillance/insurance/${id}`)
      insurance.value = insurance.value.filter(r => r.id !== id)
    } catch (e: any) {
      throw new Error(e.response?.data?.error || e.message)
    }
  }

  return {
    rows, dashboard, insurance, loading, error,
    searchQuery, flagFilter, assetTypeFilter,
    assetTypes, filteredRows,
    loadTable, loadDashboard, updateProperty,
    loadInsurance, saveInsurance, deleteInsurance,
  }
})
