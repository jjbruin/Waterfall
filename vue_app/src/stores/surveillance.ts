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
  // Live data — consistent with Property Financials / Deal Analysis
  occ_pct: number | null
  occ_period: string | null
  noi_ttm: number | null
  revenue_ttm: number | null
  dscr: number | null
  fin_period: string | null
  // Debt — ISBS balance (Deal Analysis consistent)
  debt_balance: number | null
  // Loan data — Deal Analysis consistent
  loan_rate: number | null
  maturity_date: string | null
  loan_type: string | null
  // Debt Covenants — computed actuals
  dy_val: number | null
  ltv_val: number | null
  prop_value: number | null
  // Debt Covenants — requirements from MRI Loan table
  dscr_min: number | null
  dscr_ext: number | null
  dy_min: number | null
  dy_ext: number | null
  ltv_max: number | null
  ltv_ext: number | null
  extension_options: string | null
  // Other surveillance fields
  working_capital: number | null
  tax_due: string | null
  ins_renewal: string | null
  tenant_exp: string | null
  updated_at: string | null
  // Insurance
  has_property_ins: boolean
  has_gl_ins: boolean
  // Comments (latest)
  comment_text: string | null
  comment_date: string | null
  comment_id: number | null
  // Reporting completeness
  rpt_occ_latest: string | null
  rpt_occ_missing: number | null
  rpt_rent_roll_latest: string | null
  rpt_rent_roll_missing: number | null
  rpt_is_latest: string | null
  rpt_is_missing: number | null
  rpt_bs_latest: string | null
  rpt_bs_missing: number | null
  is_commercial: boolean
}

export interface SurveillanceDashboard {
  total: number
  total_debt: number | null
  avg_occ: number | null
  total_noi_ttm: number | null
  maturing_12mo: number
  by_type: Record<string, number>
}

export interface SurveillanceComment {
  id: number
  vcode: string
  comment_date: string
  comment_text: string
  created_by: string | null
  created_at: string | null
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

  // Comments for a selected deal
  const commentsDealVcode = ref<string | null>(null)
  const comments = ref<SurveillanceComment[]>([])

  // Filters
  const searchQuery = ref('')
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

  // --- Comments ---
  async function loadComments(vcode: string) {
    commentsDealVcode.value = vcode
    try {
      const res = await api.get(`/api/surveillance/${vcode}/comments`)
      comments.value = res.data
    } catch (e: any) {
      error.value = e.response?.data?.error || e.message
    }
  }

  async function saveComment(vcode: string, commentDate: string, commentText: string) {
    try {
      await api.post(`/api/surveillance/${vcode}/comments`, {
        comment_date: commentDate,
        comment_text: commentText,
      })
      // Reload comments and update the row's latest comment
      await loadComments(vcode)
      await loadTable()
    } catch (e: any) {
      throw new Error(e.response?.data?.error || e.message)
    }
  }

  async function deleteComment(commentId: number) {
    try {
      await api.delete(`/api/surveillance/comments/${commentId}`)
      if (commentsDealVcode.value) {
        await loadComments(commentsDealVcode.value)
        await loadTable()
      }
    } catch (e: any) {
      throw new Error(e.response?.data?.error || e.message)
    }
  }

  // --- Insurance ---
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
    searchQuery, assetTypeFilter,
    assetTypes, filteredRows,
    commentsDealVcode, comments,
    loadTable, loadDashboard, updateProperty,
    loadComments, saveComment, deleteComment,
    loadInsurance, saveInsurance, deleteInsurance,
  }
})
