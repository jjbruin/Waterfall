<script setup lang="ts">
import { onMounted, ref, computed, watch, nextTick } from 'vue'
import { useDataStore } from '../stores/data'
import { useDealsStore } from '../stores/deals'
import { useRoute } from 'vue-router'
import api from '../api/client'
import ReviewPanel from '../components/common/ReviewPanel.vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { BarChart, LineChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent, TitleComponent } from 'echarts/components'

use([CanvasRenderer, BarChart, LineChart, GridComponent, TooltipComponent, LegendComponent, TitleComponent])

const data = useDataStore()
const deals = useDealsStore()
const route = useRoute()

// Review workflow state
const reviewStatus = ref<Record<string, any> | null>(null)
const reviewLoading = ref(false)

onMounted(async () => {
  if (data.deals.length === 0) await data.loadDeals()
  // Handle query params (from Review Tracking navigation)
  const qVcode = route.query.vcode as string
  const qQuarter = route.query.quarter as string
  if (qVcode) {
    deals.selectDeal(qVcode)
    if (qQuarter) selectedQuarter.value = qQuarter
    await loadOnePager(qVcode)
  }
})

// ============================================================
// Mode: 'single' (one deal) or 'batch' (investor batch)
// ============================================================
const mode = ref<'single' | 'batch'>('single')

// ============================================================
// Deal selection (single mode)
// ============================================================
const error = ref<string | null>(null)

async function onDealSelect(event: Event) {
  const vcode = (event.target as HTMLSelectElement).value
  if (!vcode) return
  deals.selectDeal(vcode)
  opData.value = null
  chartResult.value = null
  selectedQuarter.value = ''
  batchPages.value = []
  await loadOnePager(vcode)
}

// ============================================================
// Data (single mode)
// ============================================================
const opData = ref<Record<string, any> | null>(null)
const chartResult = ref<Record<string, any> | null>(null)
const selectedQuarter = ref('')
const loading = ref(false)
const saving = ref(false)

// Editable comments
const econComments = ref('')
const businessPlanComments = ref('')
const accruedPrefComment = ref('')

function parseQuarter(q: string): [number, number] {
  const [yStr, qStr] = q.split('-')
  return [parseInt(yStr), parseInt(qStr.replace('Q', ''))]
}

function getMostRecentCompletedQuarter(quarters: string[]): string {
  const now = new Date()
  const curYear = now.getFullYear()
  const curQuarter = Math.ceil((now.getMonth() + 1) / 3)
  const completed = quarters.filter(q => {
    const [y, qn] = parseQuarter(q)
    return y < curYear || (y === curYear && qn < curQuarter)
  })
  if (!completed.length) return quarters[quarters.length - 1]
  return completed.reduce((a, b) => {
    const [ay, aq] = parseQuarter(a)
    const [by, bq] = parseQuarter(b)
    return (by > ay || (by === ay && bq > aq)) ? b : a
  })
}

async function loadOnePager(vcode: string) {
  loading.value = true
  try {
    const params: any = {}
    if (selectedQuarter.value) params.quarter = selectedQuarter.value
    const [opRes, chartRes] = await Promise.all([
      api.get(`/api/financials/${vcode}/one-pager`, { params }),
      api.get(`/api/financials/${vcode}/one-pager/chart`),
    ])
    opData.value = opRes.data
    chartResult.value = chartRes.data
    if (!selectedQuarter.value && opRes.data.available_quarters?.length) {
      selectedQuarter.value = getMostRecentCompletedQuarter(opRes.data.available_quarters)
    }
    const c = opRes.data.comments || {}
    econComments.value = c.econ_comments || ''
    businessPlanComments.value = c.business_plan_comments || ''
    accruedPrefComment.value = c.accrued_pref_comment || ''
    // Load review status after quarter is known
    await loadReviewStatus()
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    loading.value = false
  }
}

async function refreshQuarter() {
  if (deals.currentVcode) {
    await loadOnePager(deals.currentVcode)
  }
}

async function saveComments() {
  if (!deals.currentVcode || !selectedQuarter.value) return
  saving.value = true
  try {
    await api.put(`/api/financials/${deals.currentVcode}/one-pager/comments`, {
      quarter: selectedQuarter.value,
      econ_comments: econComments.value,
      business_plan_comments: businessPlanComments.value,
      accrued_pref_comment: accruedPrefComment.value,
    })
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    saving.value = false
  }
}

// ============================================================
// Review workflow
// ============================================================
async function loadReviewStatus() {
  if (!deals.currentVcode || !selectedQuarter.value) return
  reviewLoading.value = true
  try {
    const res = await api.get(`/api/reviews/${deals.currentVcode}/${selectedQuarter.value}`)
    reviewStatus.value = res.data
  } catch {
    reviewStatus.value = null
  } finally {
    reviewLoading.value = false
  }
}

const commentsLocked = computed(() => {
  if (!reviewStatus.value) return false
  return !reviewStatus.value.is_editable
})

async function handleReviewSubmit(note: string) {
  reviewLoading.value = true
  try {
    const res = await api.post(
      `/api/reviews/${deals.currentVcode}/${selectedQuarter.value}/submit`,
      { note: note || undefined }
    )
    reviewStatus.value = res.data
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    reviewLoading.value = false
  }
}

async function handleReviewApprove(note: string) {
  reviewLoading.value = true
  try {
    const res = await api.post(
      `/api/reviews/${deals.currentVcode}/${selectedQuarter.value}/approve`,
      { note: note || undefined }
    )
    reviewStatus.value = res.data
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    reviewLoading.value = false
  }
}

async function handleReviewReturn(note: string) {
  reviewLoading.value = true
  try {
    const res = await api.post(
      `/api/reviews/${deals.currentVcode}/${selectedQuarter.value}/return`,
      { note }
    )
    reviewStatus.value = res.data
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    reviewLoading.value = false
  }
}

async function handleReviewNote(note: string) {
  reviewLoading.value = true
  try {
    const res = await api.post(
      `/api/reviews/${deals.currentVcode}/${selectedQuarter.value}/note`,
      { note }
    )
    reviewStatus.value = res.data
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    reviewLoading.value = false
  }
}

// ============================================================
// Batch mode
// ============================================================
interface Investor {
  investor_id: string
  name: string
  display: string
  deal_count: number
  vcodes: string[]
}

interface BatchPage {
  vcode: string
  data: Record<string, any> | null
  chart: Record<string, any> | null
  error?: string
}

const investors = ref<Investor[]>([])
const selectedInvestor = ref('')
const batchQuarter = ref('')
const batchPages = ref<BatchPage[]>([])
const batchLoading = ref(false)
const batchProgress = ref('')

async function loadInvestors() {
  try {
    const res = await api.get('/api/financials/one-pager/investors')
    investors.value = res.data.investors || []
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  }
}

function onModeChange() {
  if (mode.value === 'batch' && investors.value.length === 0) {
    loadInvestors()
  }
}

const selectedInvestorInfo = computed(() =>
  investors.value.find(i => i.investor_id === selectedInvestor.value)
)

async function loadBatch() {
  const inv = selectedInvestorInfo.value
  if (!inv) return
  batchLoading.value = true
  batchPages.value = []
  batchProgress.value = `Loading ${inv.vcodes.length} deals...`
  error.value = null
  try {
    const res = await api.post('/api/financials/one-pager/batch', {
      vcodes: inv.vcodes,
      quarter: batchQuarter.value || undefined,
    })
    batchPages.value = res.data.pages || []
    // Auto-set quarter from first page with available_quarters
    if (!batchQuarter.value) {
      for (const pg of batchPages.value) {
        const aq = pg.data?.available_quarters
        if (aq?.length) {
          batchQuarter.value = getMostRecentCompletedQuarter(aq)
          break
        }
      }
    }
    batchProgress.value = ''
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
    batchProgress.value = ''
  } finally {
    batchLoading.value = false
  }
}

async function refreshBatch() {
  if (selectedInvestorInfo.value) await loadBatch()
}

// ============================================================
// Shortcut accessors (single mode)
// ============================================================
const gen = computed(() => opData.value?.general || {})
const cap = computed(() => opData.value?.cap_stack || {})
const perf = computed(() => opData.value?.property_performance || {})
const pe = computed(() => opData.value?.pe_performance || {})

// ============================================================
// Formatting helpers
// ============================================================
function fmtMil(val: number | null | undefined): string {
  if (val == null || isNaN(val) || val === 0) return '—'
  return '$' + (val / 1_000_000).toFixed(1) + 'M'
}
function fmtMil0(val: number | null | undefined): string {
  if (val == null || isNaN(val)) return '—'
  return '$' + (val / 1_000_000).toFixed(1) + 'M'
}
function fmtPct(val: number | null | undefined): string {
  if (val == null || isNaN(val)) return '—'
  const pct = val > 1 ? val : val * 100
  return pct.toFixed(1) + '%'
}
function fmtPctInt(val: number | null | undefined): string {
  if (val == null || isNaN(val)) return '—'
  const pct = val > 1 ? val : val * 100
  return Math.round(pct) + '%'
}
function fmtDscr(val: number | null | undefined): string {
  if (val == null || isNaN(val)) return ''
  return val.toFixed(2) + 'X'
}
function fmtDate(val: string | null | undefined): string {
  if (!val) return 'N/A'
  const d = new Date(val)
  if (isNaN(d.getTime())) return String(val)
  return `${d.getMonth() + 1}/${d.getDate()}/${d.getFullYear()}`
}
function fmtVariance(actual: number | null | undefined, budget: number | null | undefined): string {
  if (actual == null || budget == null || budget === 0) return ''
  const pct = ((actual - budget) / Math.abs(budget)) * 100
  return Math.round(pct) + '%'
}

// Property Performance table rows (single mode)
const perfRows = computed(() => {
  if (!perf.value || !perf.value.revenue) return []
  const p = perf.value
  return buildPerfRows(p)
})

function buildPerfRows(p: any) {
  return [
    { label: 'Economic Occ.', ytdA: fmtPct(p.economic_occ?.ytd_actual), ytdB: fmtPct(p.economic_occ?.ytd_budget),
      variance: fmtVariance(p.economic_occ?.ytd_actual, p.economic_occ?.ytd_budget),
      atClose: fmtPct(p.economic_occ?.at_close), actualYE: fmtPct(p.economic_occ?.actual_ye), uwYE: fmtPct(p.economic_occ?.uw_ye) },
    { label: 'Revenue', ytdA: fmtMil(p.revenue?.ytd_actual), ytdB: fmtMil(p.revenue?.ytd_budget),
      variance: fmtVariance(p.revenue?.ytd_actual, p.revenue?.ytd_budget),
      atClose: fmtMil(p.revenue?.at_close), actualYE: fmtMil(p.revenue?.actual_ye), uwYE: fmtMil(p.revenue?.uw_ye) },
    { label: 'Expenses', ytdA: fmtMil(p.expenses?.ytd_actual), ytdB: fmtMil(p.expenses?.ytd_budget),
      variance: fmtVariance(p.expenses?.ytd_actual, p.expenses?.ytd_budget),
      atClose: fmtMil(p.expenses?.at_close), actualYE: fmtMil(p.expenses?.actual_ye), uwYE: fmtMil(p.expenses?.uw_ye) },
    { label: 'NOI', ytdA: fmtMil(p.noi?.ytd_actual), ytdB: fmtMil(p.noi?.ytd_budget),
      variance: fmtVariance(p.noi?.ytd_actual, p.noi?.ytd_budget),
      atClose: fmtMil(p.noi?.at_close), actualYE: fmtMil(p.noi?.actual_ye), uwYE: fmtMil(p.noi?.uw_ye) },
    { label: 'DSCR', ytdA: fmtDscr(p.dscr?.ytd_actual), ytdB: fmtDscr(p.dscr?.ytd_budget),
      variance: '',
      atClose: fmtDscr(p.dscr?.at_close), actualYE: fmtDscr(p.dscr?.actual_ye), uwYE: fmtDscr(p.dscr?.uw_ye) },
  ]
}

// As-of date from quarter string
function getAsOfDate(q: string): string {
  if (!q) return ''
  const [yearStr, qStr] = q.split('-')
  const qNum = parseInt(qStr.replace('Q', ''))
  const month = qNum * 3
  const lastDay = new Date(parseInt(yearStr), month, 0).getDate()
  return `${month}/${lastDay}/${yearStr}`
}

const asOfDate = computed(() => getAsOfDate(selectedQuarter.value))

// ============================================================
// Chart option builder
// ============================================================
function buildChartOption(cr: Record<string, any> | null) {
  if (!cr || !cr.periods?.length) return null
  const labels = cr.periods.map((p: string) => {
    const m = p.match(/^Q(\d)\s+(\d{4})$/)
    return m ? `${m[2]}-Q${m[1]}` : p
  })
  const actualNoi = cr.actual_noi.map((v: number | null) => v != null ? +(v / 1_000_000).toFixed(2) : null)
  const uwNoi = cr.uw_noi.map((v: number | null) => v != null ? +(v / 1_000_000).toFixed(2) : null)
  const occ = cr.occupancy.map((v: number | null) => v != null ? +v.toFixed(1) : null)
  return {
    title: { text: 'Occupancy vs. NOI', subtext: '($ Millions)', left: 'center', top: 0,
      textStyle: { fontSize: 13, fontWeight: 'bold' }, subtextStyle: { fontSize: 11 } },
    tooltip: { trigger: 'axis' },
    legend: { bottom: 0, textStyle: { fontSize: 10 } },
    grid: { left: 55, right: 55, top: 55, bottom: 45 },
    xAxis: { type: 'category', data: labels, axisLabel: { fontSize: 10, rotate: 0 } },
    yAxis: [
      { type: 'value', name: '', position: 'left', min: 0, max: 100,
        axisLabel: { formatter: '{value}.0%', fontSize: 10 } },
      { type: 'value', name: '', position: 'right',
        axisLabel: { formatter: (v: number) => v.toFixed(2), fontSize: 10 } },
    ],
    series: [
      { name: 'Occupancy', type: 'bar', yAxisIndex: 0, data: occ,
        itemStyle: { color: '#5B9BD5' }, barMaxWidth: 45,
        label: { show: true, position: 'top', formatter: (p: any) => p.value != null ? p.value.toFixed(1) + '%' : '', fontSize: 9 } },
      { name: 'NOI U/W', type: 'line', yAxisIndex: 1, data: uwNoi,
        lineStyle: { color: '#ED7D31', width: 2 }, itemStyle: { color: '#ED7D31' }, symbol: 'circle', symbolSize: 5 },
      { name: 'NOI ACT', type: 'line', yAxisIndex: 1, data: actualNoi,
        lineStyle: { color: '#A5A5A5', width: 2 }, itemStyle: { color: '#A5A5A5' }, symbol: 'circle', symbolSize: 5 },
    ],
  }
}

const chartOption = computed(() => buildChartOption(chartResult.value))

// ============================================================
// Print
// ============================================================
function printOnePager() {
  window.print()
}
</script>

<template>
  <div class="one-pager-page">
    <!-- Controls bar (hidden in print) -->
    <div class="controls-bar no-print">
      <!-- Mode toggle -->
      <div class="mode-toggle">
        <label>
          <input type="radio" value="single" v-model="mode" @change="onModeChange" /> Single Deal
        </label>
        <label>
          <input type="radio" value="batch" v-model="mode" @change="onModeChange" /> Batch by Investor
        </label>
      </div>

      <!-- Single deal controls -->
      <template v-if="mode === 'single'">
        <div class="deal-selector">
          <label>Deal:</label>
          <select :value="deals.currentVcode" @change="onDealSelect">
            <option value="">-- Choose a deal --</option>
            <option v-for="d in data.deals" :key="d.vcode" :value="d.vcode">
              {{ d.Investment_Name || d.vcode }}
            </option>
          </select>
        </div>
        <div v-if="opData" class="quarter-selector">
          <label>Quarter:</label>
          <select v-model="selectedQuarter" @change="refreshQuarter">
            <option v-for="q in (opData.available_quarters || [])" :key="q" :value="q">{{ q }}</option>
          </select>
        </div>
        <button v-if="opData" class="btn btn-sm" @click="printOnePager">Print</button>
        <button v-if="opData && !commentsLocked" class="btn btn-sm btn-save" @click="saveComments" :disabled="saving">
          {{ saving ? 'Saving...' : 'Save Comments' }}
        </button>
      </template>

      <!-- Batch controls -->
      <template v-if="mode === 'batch'">
        <div class="deal-selector">
          <label>Investor:</label>
          <select v-model="selectedInvestor">
            <option value="">-- Choose an investor --</option>
            <option v-for="inv in investors" :key="inv.investor_id" :value="inv.investor_id">
              {{ inv.display }} ({{ inv.deal_count }} deals)
            </option>
          </select>
        </div>
        <div v-if="selectedInvestor" class="quarter-selector">
          <label>Quarter:</label>
          <input type="text" v-model="batchQuarter" placeholder="e.g. 2025-Q4" class="quarter-input" />
        </div>
        <button v-if="selectedInvestor" class="btn btn-sm" @click="loadBatch" :disabled="batchLoading">
          {{ batchLoading ? 'Loading...' : 'Load All' }}
        </button>
        <button v-if="batchPages.length" class="btn btn-sm" @click="printOnePager">Print All</button>
      </template>
    </div>

    <div v-if="error" class="error-banner no-print">
      {{ error }}
      <button @click="error = null">Dismiss</button>
    </div>

    <!-- Review Panel (single mode only) -->
    <ReviewPanel
      v-if="mode === 'single' && opData && selectedQuarter"
      :review="reviewStatus"
      :loading="reviewLoading"
      @submit="handleReviewSubmit"
      @approve="handleReviewApprove"
      @return="handleReviewReturn"
      @add-note="handleReviewNote"
    />

    <!-- ==================== SINGLE DEAL MODE ==================== -->
    <template v-if="mode === 'single'">
      <div v-if="loading" class="loading">Loading one pager...</div>

      <div v-else-if="opData" class="op-sheet">
        <h1 class="op-title">{{ gen.investment_name || deals.currentVcode }}</h1>

        <!-- GENERAL INFORMATION -->
        <div class="section-header">GENERAL INFORMATION</div>
        <table class="info-table">
          <tbody>
            <tr>
              <td class="lbl">Partner:</td><td class="val">{{ gen.partner || '—' }}</td>
              <td class="lbl">Asset Type:</td><td class="val">{{ gen.asset_type || '—' }}</td>
            </tr>
            <tr>
              <td class="lbl">Location:</td><td class="val">{{ gen.location || '—' }}</td>
              <td class="lbl">Investment Strategy:</td><td class="val">{{ gen.investment_strategy || '—' }}</td>
            </tr>
            <tr>
              <td class="lbl"># Units | SF:</td>
              <td class="val">{{ gen.units ? gen.units.toLocaleString() : '—' }}{{ gen.sqft ? ' | ' + (gen.sqft >= 1000 ? Math.round(gen.sqft / 1000).toLocaleString() + 'K' : gen.sqft.toLocaleString()) : '' }}</td>
              <td class="lbl">Date Closed:</td><td class="val">{{ fmtDate(gen.date_closed) }}</td>
            </tr>
            <tr>
              <td class="lbl">Year Built:</td><td class="val">{{ gen.year_built || '—' }}</td>
              <td class="lbl">Underwritten Exit:</td><td class="val">{{ fmtDate(gen.anticipated_exit) }}</td>
            </tr>
          </tbody>
        </table>

        <!-- CAPITALIZATION -->
        <div class="section-header">CAPITALIZATION / EXPOSURE / DEAL TERMS</div>
        <table class="info-table cap-table">
          <tbody>
            <tr>
              <td class="lbl">Purchase Price:</td><td class="val">{{ fmtMil(cap.purchase_price) }}</td>
              <td class="lbl cap-hdr" colspan="2">Capitalization</td><td></td>
            </tr>
            <tr>
              <td class="lbl">P.E. Coupon:</td><td class="val">{{ cap.pe_coupon ? fmtPct(cap.pe_coupon) : 'N/A' }}</td>
              <td class="lbl">Debt:</td><td class="val right">{{ fmtMil(cap.debt) }}</td>
              <td class="val right">{{ fmtPctInt(cap.debt_pct) }}</td>
            </tr>
            <tr>
              <td class="lbl">P.E. Participation:</td><td class="val">{{ cap.pe_participation ? fmtPct(cap.pe_participation) : 'N/A' }}</td>
              <td class="lbl">Pref. Equity:</td><td class="val right">{{ fmtMil(cap.pref_equity) }}</td>
              <td class="val right">{{ fmtPctInt(cap.pref_equity_pct) }}</td>
            </tr>
            <tr>
              <td class="lbl">Loan Terms:</td><td class="val">{{ cap.loan_terms_str || 'N/A' }}</td>
              <td class="lbl">Partner Equity:</td><td class="val right">{{ fmtMil(cap.partner_equity) }}</td>
              <td class="val right">{{ fmtPctInt(cap.partner_equity_pct) }}</td>
            </tr>
            <tr>
              <td class="lbl">2nd Loan Terms:</td><td class="val">{{ cap.second_loan_terms_str || 'N/A' }}</td>
              <td class="lbl">Total Cap:</td><td class="val right">{{ fmtMil(cap.total_cap) }}</td><td></td>
            </tr>
            <tr>
              <td class="lbl">Rate Cap:</td><td class="val">{{ cap.rate_cap || 'N/A' }}</td>
              <td class="lbl">{{ cap.valuation_year || '' }} Valuation:</td>
              <td class="val right">{{ fmtMil(cap.current_valuation) }}</td><td></td>
            </tr>
            <tr>
              <td class="lbl">P.E. Yield on Exposure:</td><td class="val">{{ cap.pe_yield_on_exposure ? fmtPct(cap.pe_yield_on_exposure) : 'N/A' }}</td>
              <td class="lbl">P.E. Expos. on Total Cap:</td><td></td>
              <td class="val right">{{ fmtPct(cap.pe_exposure_on_cap) }}</td>
            </tr>
            <tr>
              <td class="lbl">Pref Equity capitalization:</td><td class="val">{{ cap.pref_equity_capitalization || 'N/A' }}</td>
              <td class="lbl">P.E. Expos. on {{ cap.valuation_year ? cap.valuation_year.slice(-2) + '/' + selectedQuarter.split('-')[1] : '' }} Value:</td>
              <td></td><td class="val right">{{ fmtPct(cap.pe_exposure_on_value) }}</td>
            </tr>
          </tbody>
        </table>

        <!-- PROPERTY PERFORMANCE -->
        <div class="section-header">Property Performance</div>
        <table class="perf-table">
          <thead>
            <tr>
              <th></th>
              <th class="sub-header" colspan="3"><span class="sub-label">As of: {{ asOfDate }}</span></th>
              <th class="spacer-col"></th>
              <th class="sub-header" colspan="3"><span class="sub-label">Annual Financial Comparison</span></th>
            </tr>
            <tr>
              <th></th><th class="col-hdr">YTD (Actual)</th><th class="col-hdr">YTD (Budget)</th><th class="col-hdr">Variance</th>
              <th class="spacer-col"></th><th class="col-hdr">At Close</th><th class="col-hdr">Actual YE</th><th class="col-hdr">U/W YE</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in perfRows" :key="row.label">
              <td class="row-label">{{ row.label }}</td>
              <td class="val right">{{ row.ytdA }}</td><td class="val right">{{ row.ytdB }}</td>
              <td class="val right">{{ row.variance }}</td><td class="spacer-col"></td>
              <td class="val right">{{ row.atClose }}</td><td class="val right">{{ row.actualYE }}</td>
              <td class="val right">{{ row.uwYE }}</td>
            </tr>
          </tbody>
        </table>

        <!-- Comments -->
        <table class="comments-row-table">
          <tbody><tr>
            <td class="lbl" style="vertical-align: top; width: 80px;">Comments:</td>
            <td><textarea v-model="econComments" class="comment-input" rows="3" placeholder="Property performance comments..." spellcheck="true" lang="en" :readonly="commentsLocked"></textarea></td>
          </tr></tbody>
        </table>

        <!-- PE PERFORMANCE -->
        <div class="section-header">Preferred Equity Performance</div>
        <table class="pe-table">
          <tbody>
            <tr>
              <td class="lbl">Committed Pref Equity:</td><td class="val">{{ fmtMil0(pe.committed_pe) }}</td>
              <td class="lbl">Coupon:</td><td class="val">{{ pe.coupon ? fmtPct(pe.coupon) : 'N/A' }}</td><td></td><td></td>
            </tr>
            <tr>
              <td class="lbl">Remaining to Fund:</td><td class="val">{{ fmtMil0(pe.remaining_to_fund) }}</td>
              <td class="lbl">Participation:</td><td class="val">{{ pe.participation ? fmtPct(pe.participation) : 'N/A' }}</td><td></td><td></td>
            </tr>
            <tr><td colspan="6" style="height: 6px;"></td></tr>
            <tr>
              <td class="lbl">Funded to Date:</td><td class="val">{{ fmtMil0(pe.funded_to_date) }}</td>
              <td></td><td></td><td></td><td></td>
            </tr>
            <tr>
              <td class="lbl">Return of Capital:</td><td class="val">{{ fmtMil0(pe.return_of_capital) }}</td>
              <td class="lbl">ROE to Date:</td><td class="val">{{ pe.roe_to_date ? fmtPct(pe.roe_to_date) : '—' }}</td>
              <td class="lbl">U/W ROE to Date:</td><td class="val">{{ pe.uw_roe_to_date ? fmtPct(pe.uw_roe_to_date) : '—' }}</td>
            </tr>
            <tr>
              <td class="lbl">Current Pref Equity Balance:</td><td class="val">{{ fmtMil0(pe.current_pe_balance) }}</td>
              <td class="lbl">Accrued Balance:</td><td class="val">{{ fmtMil0(pe.accrued_balance) }}</td>
              <td colspan="2"><textarea v-model="accruedPrefComment" class="comment-input small" rows="2" placeholder="Accrued pref comment..." spellcheck="true" lang="en" :readonly="commentsLocked"></textarea></td>
            </tr>
          </tbody>
        </table>

        <!-- BUSINESS PLAN -->
        <div class="section-header">Business Plan &amp; Updates</div>
        <div class="bp-section">
          <textarea v-model="businessPlanComments" class="comment-input bp-input" rows="6" placeholder="Business plan and updates..." spellcheck="true" lang="en" :readonly="commentsLocked"></textarea>
        </div>

        <!-- CHART -->
        <div class="chart-section">
          <v-chart v-if="chartOption" :option="chartOption" style="width: 100%; height: 300px;" autoresize />
          <p v-else class="empty">No chart data available.</p>
        </div>
      </div>

      <p v-else-if="!loading" class="empty no-print">Select a deal to view the one pager.</p>
    </template>

    <!-- ==================== BATCH MODE ==================== -->
    <template v-if="mode === 'batch'">
      <div v-if="batchLoading" class="loading">{{ batchProgress || 'Loading...' }}</div>

      <p v-else-if="!batchPages.length && selectedInvestor" class="empty no-print">
        Select an investor and click "Load All" to generate one pagers for all {{ selectedInvestorInfo?.deal_count }} deals.
      </p>
      <p v-else-if="!selectedInvestor" class="empty no-print">
        Select an investor to generate batch one pagers.
      </p>

      <!-- Batch pages — one op-sheet per deal, page-break between -->
      <template v-for="(pg, idx) in batchPages" :key="pg.vcode">
        <div v-if="pg.data" class="op-sheet" :class="{ 'page-break': idx < batchPages.length - 1 }">
          <h1 class="op-title">{{ pg.data.general?.investment_name || pg.vcode }}</h1>

          <!-- GENERAL INFORMATION -->
          <div class="section-header">GENERAL INFORMATION</div>
          <table class="info-table">
            <tbody>
              <tr>
                <td class="lbl">Partner:</td><td class="val">{{ pg.data.general?.partner || '—' }}</td>
                <td class="lbl">Asset Type:</td><td class="val">{{ pg.data.general?.asset_type || '—' }}</td>
              </tr>
              <tr>
                <td class="lbl">Location:</td><td class="val">{{ pg.data.general?.location || '—' }}</td>
                <td class="lbl">Investment Strategy:</td><td class="val">{{ pg.data.general?.investment_strategy || '—' }}</td>
              </tr>
              <tr>
                <td class="lbl"># Units | SF:</td>
                <td class="val">{{ pg.data.general?.units ? pg.data.general.units.toLocaleString() : '—' }}{{ pg.data.general?.sqft ? ' | ' + (pg.data.general.sqft >= 1000 ? Math.round(pg.data.general.sqft / 1000).toLocaleString() + 'K' : pg.data.general.sqft.toLocaleString()) : '' }}</td>
                <td class="lbl">Date Closed:</td><td class="val">{{ fmtDate(pg.data.general?.date_closed) }}</td>
              </tr>
              <tr>
                <td class="lbl">Year Built:</td><td class="val">{{ pg.data.general?.year_built || '—' }}</td>
                <td class="lbl">Underwritten Exit:</td><td class="val">{{ fmtDate(pg.data.general?.anticipated_exit) }}</td>
              </tr>
            </tbody>
          </table>

          <!-- CAPITALIZATION -->
          <div class="section-header">CAPITALIZATION / EXPOSURE / DEAL TERMS</div>
          <table class="info-table cap-table">
            <tbody>
              <tr>
                <td class="lbl">Purchase Price:</td><td class="val">{{ fmtMil(pg.data.cap_stack?.purchase_price) }}</td>
                <td class="lbl cap-hdr" colspan="2">Capitalization</td><td></td>
              </tr>
              <tr>
                <td class="lbl">P.E. Coupon:</td><td class="val">{{ pg.data.cap_stack?.pe_coupon ? fmtPct(pg.data.cap_stack.pe_coupon) : 'N/A' }}</td>
                <td class="lbl">Debt:</td><td class="val right">{{ fmtMil(pg.data.cap_stack?.debt) }}</td>
                <td class="val right">{{ fmtPctInt(pg.data.cap_stack?.debt_pct) }}</td>
              </tr>
              <tr>
                <td class="lbl">P.E. Participation:</td><td class="val">{{ pg.data.cap_stack?.pe_participation ? fmtPct(pg.data.cap_stack.pe_participation) : 'N/A' }}</td>
                <td class="lbl">Pref. Equity:</td><td class="val right">{{ fmtMil(pg.data.cap_stack?.pref_equity) }}</td>
                <td class="val right">{{ fmtPctInt(pg.data.cap_stack?.pref_equity_pct) }}</td>
              </tr>
              <tr>
                <td class="lbl">Loan Terms:</td><td class="val">{{ pg.data.cap_stack?.loan_terms_str || 'N/A' }}</td>
                <td class="lbl">Partner Equity:</td><td class="val right">{{ fmtMil(pg.data.cap_stack?.partner_equity) }}</td>
                <td class="val right">{{ fmtPctInt(pg.data.cap_stack?.partner_equity_pct) }}</td>
              </tr>
              <tr>
                <td class="lbl">2nd Loan Terms:</td><td class="val">{{ pg.data.cap_stack?.second_loan_terms_str || 'N/A' }}</td>
                <td class="lbl">Total Cap:</td><td class="val right">{{ fmtMil(pg.data.cap_stack?.total_cap) }}</td><td></td>
              </tr>
              <tr>
                <td class="lbl">Rate Cap:</td><td class="val">{{ pg.data.cap_stack?.rate_cap || 'N/A' }}</td>
                <td class="lbl">{{ pg.data.cap_stack?.valuation_year || '' }} Valuation:</td>
                <td class="val right">{{ fmtMil(pg.data.cap_stack?.current_valuation) }}</td><td></td>
              </tr>
              <tr>
                <td class="lbl">P.E. Yield on Exposure:</td><td class="val">{{ pg.data.cap_stack?.pe_yield_on_exposure ? fmtPct(pg.data.cap_stack.pe_yield_on_exposure) : 'N/A' }}</td>
                <td class="lbl">P.E. Expos. on Total Cap:</td><td></td>
                <td class="val right">{{ fmtPct(pg.data.cap_stack?.pe_exposure_on_cap) }}</td>
              </tr>
              <tr>
                <td class="lbl">Pref Equity capitalization:</td><td class="val">{{ pg.data.cap_stack?.pref_equity_capitalization || 'N/A' }}</td>
                <td class="lbl">P.E. Expos. on {{ pg.data.cap_stack?.valuation_year ? pg.data.cap_stack.valuation_year.slice(-2) + '/' + (batchQuarter ? batchQuarter.split('-')[1] : '') : '' }} Value:</td>
                <td></td><td class="val right">{{ fmtPct(pg.data.cap_stack?.pe_exposure_on_value) }}</td>
              </tr>
            </tbody>
          </table>

          <!-- PROPERTY PERFORMANCE -->
          <div class="section-header">Property Performance</div>
          <table class="perf-table">
            <thead>
              <tr>
                <th></th>
                <th class="sub-header" colspan="3"><span class="sub-label">As of: {{ getAsOfDate(batchQuarter) }}</span></th>
                <th class="spacer-col"></th>
                <th class="sub-header" colspan="3"><span class="sub-label">Annual Financial Comparison</span></th>
              </tr>
              <tr>
                <th></th><th class="col-hdr">YTD (Actual)</th><th class="col-hdr">YTD (Budget)</th><th class="col-hdr">Variance</th>
                <th class="spacer-col"></th><th class="col-hdr">At Close</th><th class="col-hdr">Actual YE</th><th class="col-hdr">U/W YE</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="row in buildPerfRows(pg.data.property_performance || {})" :key="row.label">
                <td class="row-label">{{ row.label }}</td>
                <td class="val right">{{ row.ytdA }}</td><td class="val right">{{ row.ytdB }}</td>
                <td class="val right">{{ row.variance }}</td><td class="spacer-col"></td>
                <td class="val right">{{ row.atClose }}</td><td class="val right">{{ row.actualYE }}</td>
                <td class="val right">{{ row.uwYE }}</td>
              </tr>
            </tbody>
          </table>

          <!-- Comments (read-only in batch) -->
          <table class="comments-row-table">
            <tbody><tr>
              <td class="lbl" style="vertical-align: top; width: 80px;">Comments:</td>
              <td class="comment-text">{{ pg.data.comments?.econ_comments || '' }}</td>
            </tr></tbody>
          </table>

          <!-- PE PERFORMANCE -->
          <div class="section-header">Preferred Equity Performance</div>
          <table class="pe-table">
            <tbody>
              <tr>
                <td class="lbl">Committed Pref Equity:</td><td class="val">{{ fmtMil0(pg.data.pe_performance?.committed_pe) }}</td>
                <td class="lbl">Coupon:</td><td class="val">{{ pg.data.pe_performance?.coupon ? fmtPct(pg.data.pe_performance.coupon) : 'N/A' }}</td><td></td><td></td>
              </tr>
              <tr>
                <td class="lbl">Remaining to Fund:</td><td class="val">{{ fmtMil0(pg.data.pe_performance?.remaining_to_fund) }}</td>
                <td class="lbl">Participation:</td><td class="val">{{ pg.data.pe_performance?.participation ? fmtPct(pg.data.pe_performance.participation) : 'N/A' }}</td><td></td><td></td>
              </tr>
              <tr><td colspan="6" style="height: 6px;"></td></tr>
              <tr>
                <td class="lbl">Funded to Date:</td><td class="val">{{ fmtMil0(pg.data.pe_performance?.funded_to_date) }}</td>
                <td></td><td></td><td></td><td></td>
              </tr>
              <tr>
                <td class="lbl">Return of Capital:</td><td class="val">{{ fmtMil0(pg.data.pe_performance?.return_of_capital) }}</td>
                <td class="lbl">ROE to Date:</td><td class="val">{{ pg.data.pe_performance?.roe_to_date ? fmtPct(pg.data.pe_performance.roe_to_date) : '—' }}</td>
                <td class="lbl">U/W ROE to Date:</td><td class="val">{{ pg.data.pe_performance?.uw_roe_to_date ? fmtPct(pg.data.pe_performance.uw_roe_to_date) : '—' }}</td>
              </tr>
              <tr>
                <td class="lbl">Current Pref Equity Balance:</td><td class="val">{{ fmtMil0(pg.data.pe_performance?.current_pe_balance) }}</td>
                <td class="lbl">Accrued Balance:</td><td class="val">{{ fmtMil0(pg.data.pe_performance?.accrued_balance) }}</td>
                <td colspan="2" class="comment-text" style="font-size: 9px;">{{ pg.data.comments?.accrued_pref_comment || '' }}</td>
              </tr>
            </tbody>
          </table>

          <!-- BUSINESS PLAN -->
          <div class="section-header">Business Plan &amp; Updates</div>
          <div class="bp-section">
            <div class="comment-text bp-text">{{ pg.data.comments?.business_plan_comments || '' }}</div>
          </div>

          <!-- CHART -->
          <div class="chart-section">
            <v-chart v-if="buildChartOption(pg.chart)" :option="buildChartOption(pg.chart)!" style="width: 100%; height: 300px;" autoresize />
            <p v-else class="empty">No chart data available.</p>
          </div>
        </div>

        <!-- Error page -->
        <div v-else class="op-sheet page-break">
          <h1 class="op-title">{{ pg.vcode }}</h1>
          <p class="empty">{{ pg.error || 'Failed to load data' }}</p>
        </div>
      </template>
    </template>
  </div>
</template>

<style scoped>
/* ============================================================
   SCREEN STYLES
   ============================================================ */
.one-pager-page {
  max-width: 960px;
  margin: 0 auto;
  font-family: 'Calibri', 'Segoe UI', Arial, sans-serif;
  font-size: 11px;
  color: #000;
  line-height: 1.35;
}

/* Controls bar */
.controls-bar {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 16px;
  flex-wrap: wrap;
}
.mode-toggle {
  display: flex;
  gap: 12px;
  font-size: 13px;
  padding-right: 8px;
  border-right: 1px solid #ccc;
}
.mode-toggle label {
  display: flex;
  align-items: center;
  gap: 4px;
  cursor: pointer;
  font-weight: 500;
}
.deal-selector, .quarter-selector {
  display: flex;
  align-items: center;
  gap: 8px;
}
.deal-selector label, .quarter-selector label {
  font-size: 13px;
  font-weight: 500;
}
.deal-selector select, .quarter-selector select {
  padding: 6px 10px;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 13px;
  min-width: 280px;
}
.quarter-selector select { min-width: 120px; }
.quarter-input {
  padding: 6px 10px;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 13px;
  width: 120px;
}
.btn { padding: 6px 16px; border: none; border-radius: 4px; background: #1F4E79; color: white; cursor: pointer; font-size: 12px; }
.btn:hover { opacity: 0.9; }
.btn-sm { padding: 5px 12px; }
.btn-save { background: #548235; }
.btn-save:disabled { opacity: 0.5; cursor: default; }
.error-banner { background: #fef2f2; border: 1px solid #fca5a5; color: #991b1b; padding: 8px 14px; border-radius: 6px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; font-size: 13px; }
.error-banner button { background: none; border: 1px solid #fca5a5; color: #991b1b; padding: 3px 10px; border-radius: 4px; cursor: pointer; font-size: 12px; }
.loading { text-align: center; padding: 40px; color: #666; font-style: italic; font-size: 14px; }
.empty { text-align: center; padding: 40px; color: #999; font-style: italic; font-size: 14px; }

/* ============================================================
   ONE PAGER SHEET (printable area)
   ============================================================ */
.op-sheet {
  background: #fff;
  border: 1px solid #ddd;
  padding: 20px 28px;
  margin-bottom: 16px;
}

/* Title */
.op-title {
  text-align: center;
  font-size: 20px;
  font-weight: 700;
  margin: 0 0 2px 0;
  border-bottom: 2px solid #000;
  padding-bottom: 4px;
}

/* Section headers */
.section-header {
  text-align: center;
  font-weight: 700;
  font-size: 11px;
  border-bottom: 1px solid #000;
  padding: 5px 0 2px 0;
  margin: 2px 0 2px 0;
}

/* Generic info table */
.info-table {
  width: 100%;
  border-collapse: collapse;
  margin-bottom: 0;
}
.info-table td {
  padding: 1px 6px 1px 0;
  vertical-align: top;
}
.info-table .lbl {
  font-weight: 700;
  font-style: italic;
  white-space: nowrap;
  width: 22%;
}
.info-table .val {
  width: 28%;
}

/* Cap table overrides */
.cap-table .lbl { width: 18%; }
.cap-table .val { width: 18%; }
.cap-table .cap-hdr {
  font-weight: 700;
  font-style: italic;
  text-decoration: underline;
}
.cap-table .right { text-align: right; }
.cap-table td:nth-child(3) { padding-left: 20px; }

/* Performance table */
.perf-table {
  width: 100%;
  border-collapse: collapse;
  margin-bottom: 0;
}
.perf-table th, .perf-table td {
  padding: 1px 6px;
  font-size: 11px;
}
.perf-table .sub-header {
  text-align: center;
  border-bottom: none;
  padding-bottom: 0;
}
.perf-table .sub-label {
  font-weight: 400;
  font-size: 10px;
  font-style: italic;
}
.perf-table .col-hdr {
  text-align: right;
  font-weight: 700;
  font-size: 10px;
  text-decoration: underline;
  padding-bottom: 2px;
}
.perf-table .row-label {
  font-weight: 700;
  white-space: nowrap;
  padding-right: 12px;
}
.perf-table .val { font-size: 11px; }
.perf-table .right { text-align: right; }
.perf-table .spacer-col { width: 20px; }

/* Comments row */
.comments-row-table {
  width: 100%;
  border-collapse: collapse;
  margin: 2px 0;
}
.comments-row-table td { padding: 2px 4px; }
.comment-input {
  width: 100%;
  border: 1px solid #ccc;
  border-radius: 3px;
  padding: 4px 6px;
  font-family: inherit;
  font-size: 10px;
  line-height: 1.35;
  resize: vertical;
  color: #000;
}
.comment-input.small { font-size: 9px; }
.comment-text {
  font-size: 10px;
  line-height: 1.35;
  white-space: pre-wrap;
  min-height: 2.8em;
  border-bottom: 1px solid #eee;
}
.comment-text.bp-text {
  min-height: 5.5em;
}

/* PE table */
.pe-table {
  width: 100%;
  border-collapse: collapse;
  margin-bottom: 0;
}
.pe-table td {
  padding: 1px 6px 1px 0;
  vertical-align: top;
  font-size: 11px;
}
.pe-table .lbl {
  font-weight: 700;
  font-style: italic;
  white-space: nowrap;
}
.pe-table .val { }

/* Business plan section */
.bp-section {
  margin: 2px 0;
}
.bp-input {
  min-height: 80px;
}

/* Chart */
.chart-section {
  margin-top: 6px;
  border-top: 1px solid #ccc;
  padding-top: 4px;
}

/* ============================================================
   PRINT STYLES
   ============================================================ */
@media print {
  * { -webkit-print-color-adjust: exact; print-color-adjust: exact; }

  .no-print { display: none !important; }

  @page {
    size: letter portrait;
    margin: 0.4in 0.5in;
  }

  body, html {
    margin: 0 !important;
    padding: 0 !important;
    font-size: 10px !important;
  }

  .one-pager-page {
    max-width: none;
    margin: 0;
    padding: 0;
  }

  .op-sheet {
    border: none;
    padding: 0;
    box-shadow: none;
    margin-bottom: 0;
  }

  .op-sheet.page-break {
    page-break-after: always;
  }

  .op-title { font-size: 18px; }
  .section-header { font-size: 10px; }

  .info-table td, .cap-table td, .perf-table th, .perf-table td, .pe-table td {
    font-size: 10px;
    padding: 1px 4px 1px 0;
  }

  /* Make textareas look like plain text in print */
  .comment-input {
    border: none !important;
    padding: 0 !important;
    resize: none !important;
    background: transparent !important;
    font-size: 10px !important;
    overflow: visible;
    height: auto !important;
  }

  .chart-section {
    break-inside: avoid;
  }

  /* Force chart to print */
  .chart-section canvas {
    max-width: 100% !important;
  }
}
</style>
