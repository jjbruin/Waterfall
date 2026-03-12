<script setup lang="ts">
import { onMounted, ref, computed } from 'vue'
import { useDataStore } from '../stores/data'
import { useDealsStore } from '../stores/deals'
import api from '../api/client'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { BarChart, LineChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent, TitleComponent } from 'echarts/components'

use([CanvasRenderer, BarChart, LineChart, GridComponent, TooltipComponent, LegendComponent, TitleComponent])

const data = useDataStore()
const deals = useDealsStore()

onMounted(async () => {
  if (data.deals.length === 0) await data.loadDeals()
})

// ============================================================
// Deal selection
// ============================================================
const error = ref<string | null>(null)

async function onDealSelect(event: Event) {
  const vcode = (event.target as HTMLSelectElement).value
  if (!vcode) return
  deals.selectDeal(vcode)
  opData.value = null
  chartResult.value = null
  selectedQuarter.value = ''
  await loadOnePager(vcode)
}

// ============================================================
// Data
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
      selectedQuarter.value = opRes.data.available_quarters[opRes.data.available_quarters.length - 1]
    }
    // Populate comment fields
    const c = opRes.data.comments || {}
    econComments.value = c.econ_comments || ''
    businessPlanComments.value = c.business_plan_comments || ''
    accruedPrefComment.value = c.accrued_pref_comment || ''
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    loading.value = false
  }
}

async function refreshQuarter() {
  if (deals.currentVcode) await loadOnePager(deals.currentVcode)
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
// Shortcut accessors
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
  // Like fmtMil but shows $0.0M for zero (for PE section)
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
  return (pct >= 0 ? '' : '') + Math.round(pct) + '%'
}

// Property Performance table rows
const perfRows = computed(() => {
  if (!perf.value || !perf.value.revenue) return []
  const p = perf.value
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
})

// As-of date from quarter
const asOfDate = computed(() => {
  const q = selectedQuarter.value
  if (!q) return ''
  const [yearStr, qStr] = q.split('-')
  const qNum = parseInt(qStr.replace('Q', ''))
  const month = qNum * 3
  const lastDay = new Date(parseInt(yearStr), month, 0).getDate()
  return `${month}/${lastDay}/${yearStr}`
})

// ============================================================
// Chart option
// ============================================================
const chartOption = computed(() => {
  if (!chartResult.value) return null
  const cr = chartResult.value
  if (!cr.periods?.length) return null

  // Convert "Q4 2025" → "2025-Q4" for display
  const labels = cr.periods.map((p: string) => {
    const m = p.match(/^Q(\d)\s+(\d{4})$/)
    return m ? `${m[2]}-Q${m[1]}` : p
  })

  // NOI in millions
  const actualNoi = cr.actual_noi.map((v: number | null) => v != null ? +(v / 1_000_000).toFixed(2) : null)
  const uwNoi = cr.uw_noi.map((v: number | null) => v != null ? +(v / 1_000_000).toFixed(2) : null)
  const occ = cr.occupancy.map((v: number | null) => v != null ? +v.toFixed(1) : null)

  return {
    title: {
      text: 'Occupancy vs. NOI',
      subtext: '($ Millions)',
      left: 'center',
      top: 0,
      textStyle: { fontSize: 13, fontWeight: 'bold' },
      subtextStyle: { fontSize: 11 },
    },
    tooltip: { trigger: 'axis' },
    legend: { bottom: 0, textStyle: { fontSize: 10 } },
    grid: { left: 55, right: 55, top: 55, bottom: 45 },
    xAxis: {
      type: 'category',
      data: labels,
      axisLabel: { fontSize: 10, rotate: 0 },
    },
    yAxis: [
      {
        type: 'value',
        name: '',
        position: 'left',
        min: 0,
        max: 100,
        axisLabel: { formatter: '{value}.0%', fontSize: 10 },
      },
      {
        type: 'value',
        name: '',
        position: 'right',
        axisLabel: { formatter: (v: number) => v.toFixed(2), fontSize: 10 },
      },
    ],
    series: [
      {
        name: 'Occupancy',
        type: 'bar',
        yAxisIndex: 0,
        data: occ,
        itemStyle: { color: '#5B9BD5' },
        barMaxWidth: 45,
        label: {
          show: true,
          position: 'top',
          formatter: (p: any) => p.value != null ? p.value.toFixed(1) + '%' : '',
          fontSize: 9,
        },
      },
      {
        name: 'NOI U/W',
        type: 'line',
        yAxisIndex: 1,
        data: uwNoi,
        lineStyle: { color: '#ED7D31', width: 2 },
        itemStyle: { color: '#ED7D31' },
        symbol: 'circle',
        symbolSize: 5,
      },
      {
        name: 'NOI ACT',
        type: 'line',
        yAxisIndex: 1,
        data: actualNoi,
        lineStyle: { color: '#A5A5A5', width: 2 },
        itemStyle: { color: '#A5A5A5' },
        symbol: 'circle',
        symbolSize: 5,
      },
    ],
  }
})

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
      <div class="deal-selector">
        <label>Select Deal:</label>
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
      <button v-if="opData" class="btn btn-sm btn-save" @click="saveComments" :disabled="saving">
        {{ saving ? 'Saving...' : 'Save Comments' }}
      </button>
    </div>

    <div v-if="error" class="error-banner no-print">
      {{ error }}
      <button @click="error = null">Dismiss</button>
    </div>

    <div v-if="loading" class="loading">Loading one pager...</div>

    <!-- ==================== PRINTABLE ONE PAGER ==================== -->
    <div v-else-if="opData" class="op-sheet">

      <!-- TITLE -->
      <h1 class="op-title">{{ gen.investment_name || deals.currentVcode }}</h1>

      <!-- GENERAL INFORMATION -->
      <div class="section-header">GENERAL INFORMATION</div>
      <table class="info-table">
        <tbody>
          <tr>
            <td class="lbl">Partner:</td>
            <td class="val">{{ gen.partner || '—' }}</td>
            <td class="lbl">Asset Type:</td>
            <td class="val">{{ gen.asset_type || '—' }}</td>
          </tr>
          <tr>
            <td class="lbl">Location:</td>
            <td class="val">{{ gen.location || '—' }}</td>
            <td class="lbl">Investment Strategy:</td>
            <td class="val">{{ gen.investment_strategy || '—' }}</td>
          </tr>
          <tr>
            <td class="lbl"># Units | SF:</td>
            <td class="val">{{ gen.units ? gen.units.toLocaleString() : '—' }}{{ gen.sqft ? ' | ' + (gen.sqft >= 1000 ? Math.round(gen.sqft / 1000).toLocaleString() + 'K' : gen.sqft.toLocaleString()) : '' }}</td>
            <td class="lbl">Date Closed:</td>
            <td class="val">{{ fmtDate(gen.date_closed) }}</td>
          </tr>
          <tr>
            <td class="lbl">Year Built:</td>
            <td class="val">{{ gen.year_built || '—' }}</td>
            <td class="lbl">Underwritten Exit:</td>
            <td class="val">{{ fmtDate(gen.anticipated_exit) }}</td>
          </tr>
        </tbody>
      </table>

      <!-- CAPITALIZATION / EXPOSURE / DEAL TERMS -->
      <div class="section-header">CAPITALIZATION / EXPOSURE / DEAL TERMS</div>
      <table class="info-table cap-table">
        <tbody>
          <tr>
            <td class="lbl">Purchase Price:</td>
            <td class="val">{{ fmtMil(cap.purchase_price) }}</td>
            <td class="lbl cap-hdr" colspan="2">Capitalization</td>
            <td></td>
          </tr>
          <tr>
            <td class="lbl">P.E. Coupon:</td>
            <td class="val">{{ cap.pe_coupon ? fmtPct(cap.pe_coupon) : 'N/A' }}</td>
            <td class="lbl">Debt:</td>
            <td class="val right">{{ fmtMil(cap.debt) }}</td>
            <td class="val right">{{ fmtPctInt(cap.debt_pct) }}</td>
          </tr>
          <tr>
            <td class="lbl">P.E. Participation:</td>
            <td class="val">{{ cap.pe_participation ? fmtPct(cap.pe_participation) : 'N/A' }}</td>
            <td class="lbl">Pref. Equity:</td>
            <td class="val right">{{ fmtMil(cap.pref_equity) }}</td>
            <td class="val right">{{ fmtPctInt(cap.pref_equity_pct) }}</td>
          </tr>
          <tr>
            <td class="lbl">Loan Terms:</td>
            <td class="val">{{ cap.loan_terms_str || 'N/A' }}</td>
            <td class="lbl">Partner Equity:</td>
            <td class="val right">{{ fmtMil(cap.partner_equity) }}</td>
            <td class="val right">{{ fmtPctInt(cap.partner_equity_pct) }}</td>
          </tr>
          <tr>
            <td class="lbl">2nd Loan Terms:</td>
            <td class="val">{{ cap.second_loan_terms_str || 'N/A' }}</td>
            <td class="lbl">Total Cap:</td>
            <td class="val right">{{ fmtMil(cap.total_cap) }}</td>
            <td></td>
          </tr>
          <tr>
            <td class="lbl">Rate Cap:</td>
            <td class="val">{{ cap.rate_cap || 'N/A' }}</td>
            <td class="lbl">{{ cap.valuation_year || '' }} Valuation:</td>
            <td class="val right">{{ fmtMil(cap.current_valuation) }}</td>
            <td></td>
          </tr>
          <tr>
            <td class="lbl">P.E. Yield on Exposure:</td>
            <td class="val">{{ cap.pe_yield_on_exposure ? fmtPct(cap.pe_yield_on_exposure) : 'N/A' }}</td>
            <td class="lbl">P.E. Expos. on Total Cap:</td>
            <td></td>
            <td class="val right">{{ fmtPct(cap.pe_exposure_on_cap) }}</td>
          </tr>
          <tr>
            <td class="lbl">Pref Equity capitalization:</td>
            <td class="val">{{ cap.pref_equity_capitalization || 'N/A' }}</td>
            <td class="lbl">P.E. Expos. on {{ cap.valuation_year ? cap.valuation_year.slice(-2) + '/' + selectedQuarter.split('-')[1] : '' }} Value:</td>
            <td></td>
            <td class="val right">{{ fmtPct(cap.pe_exposure_on_value) }}</td>
          </tr>
        </tbody>
      </table>

      <!-- PROPERTY PERFORMANCE -->
      <div class="section-header">Property Performance</div>
      <table class="perf-table">
        <thead>
          <tr>
            <th></th>
            <th class="sub-header" colspan="3">
              <span class="sub-label">As of: {{ asOfDate }}</span>
            </th>
            <th class="spacer-col"></th>
            <th class="sub-header" colspan="3">
              <span class="sub-label">Annual Financial Comparison</span>
            </th>
          </tr>
          <tr>
            <th></th>
            <th class="col-hdr">YTD (Actual)</th>
            <th class="col-hdr">YTD (Budget)</th>
            <th class="col-hdr">Variance</th>
            <th class="spacer-col"></th>
            <th class="col-hdr">At Close</th>
            <th class="col-hdr">Actual YE</th>
            <th class="col-hdr">U/W YE</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in perfRows" :key="row.label">
            <td class="row-label">{{ row.label }}</td>
            <td class="val right">{{ row.ytdA }}</td>
            <td class="val right">{{ row.ytdB }}</td>
            <td class="val right">{{ row.variance }}</td>
            <td class="spacer-col"></td>
            <td class="val right">{{ row.atClose }}</td>
            <td class="val right">{{ row.actualYE }}</td>
            <td class="val right">{{ row.uwYE }}</td>
          </tr>
        </tbody>
      </table>

      <!-- Comments row -->
      <table class="comments-row-table">
        <tbody>
          <tr>
            <td class="lbl" style="vertical-align: top; width: 80px;">Comments:</td>
            <td>
              <textarea v-model="econComments" class="comment-input" rows="3"
                placeholder="Property performance comments..."></textarea>
            </td>
          </tr>
        </tbody>
      </table>

      <!-- PREFERRED EQUITY PERFORMANCE -->
      <div class="section-header">Preferred Equity Performance</div>
      <table class="pe-table">
        <tbody>
          <tr>
            <td class="lbl">Committed Pref Equity:</td>
            <td class="val">{{ fmtMil0(pe.committed_pe) }}</td>
            <td class="lbl">Coupon:</td>
            <td class="val">{{ pe.coupon ? fmtPct(pe.coupon) : 'N/A' }}</td>
            <td></td>
            <td></td>
          </tr>
          <tr>
            <td class="lbl">Remaining to Fund:</td>
            <td class="val">{{ fmtMil0(pe.remaining_to_fund) }}</td>
            <td class="lbl">Participation:</td>
            <td class="val">{{ pe.participation ? fmtPct(pe.participation) : 'N/A' }}</td>
            <td></td>
            <td></td>
          </tr>
          <tr><td colspan="6" style="height: 6px;"></td></tr>
          <tr>
            <td class="lbl">Funded to Date:</td>
            <td class="val">{{ fmtMil0(pe.funded_to_date) }}</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
          </tr>
          <tr>
            <td class="lbl">Return of Capital:</td>
            <td class="val">{{ fmtMil0(pe.return_of_capital) }}</td>
            <td class="lbl">ROE to Date:</td>
            <td class="val">{{ pe.roe_to_date ? fmtPct(pe.roe_to_date) : '—' }}</td>
            <td class="lbl">U/W ROE to Date:</td>
            <td class="val">{{ pe.uw_roe_to_date ? fmtPct(pe.uw_roe_to_date) : '—' }}</td>
          </tr>
          <tr>
            <td class="lbl">Current Pref Equity Balance:</td>
            <td class="val">{{ fmtMil0(pe.current_pe_balance) }}</td>
            <td class="lbl">Accrued Balance:</td>
            <td class="val">{{ fmtMil0(pe.accrued_balance) }}</td>
            <td colspan="2">
              <textarea v-model="accruedPrefComment" class="comment-input small" rows="2"
                placeholder="Accrued pref comment..."></textarea>
            </td>
          </tr>
        </tbody>
      </table>

      <!-- BUSINESS PLAN & UPDATES -->
      <div class="section-header">Business Plan &amp; Updates</div>
      <div class="bp-section">
        <textarea v-model="businessPlanComments" class="comment-input bp-input" rows="6"
          placeholder="Business plan and updates..."></textarea>
      </div>

      <!-- OCCUPANCY vs NOI CHART -->
      <div class="chart-section">
        <v-chart v-if="chartOption" :option="chartOption" style="width: 100%; height: 300px;" autoresize />
        <p v-else class="empty">No chart data available.</p>
      </div>

    </div>

    <p v-else-if="!loading" class="empty no-print">Select a deal to view the one pager.</p>
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
