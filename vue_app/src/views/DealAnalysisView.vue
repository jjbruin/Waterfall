<script setup lang="ts">
import { onMounted, computed, ref, watch } from 'vue'
import { useDataStore } from '../stores/data'
import { useDealsStore } from '../stores/deals'
import KpiCard from '../components/common/KpiCard.vue'
import DataTable from '../components/common/DataTable.vue'
import ProgressOverlay from '../components/common/ProgressOverlay.vue'

const data = useDataStore()
const deals = useDealsStore()

onMounted(async () => {
  if (data.deals.length === 0) await data.loadDeals()
})

async function onDealSelect(event: Event) {
  const vcode = (event.target as HTMLSelectElement).value
  if (vcode) await deals.computeDeal(vcode)
}

// Lazy-load sections when they expand
const expanded = ref<Record<string, boolean>>({})
function toggle(section: string) {
  expanded.value[section] = !expanded.value[section]
  const vc = deals.currentVcode
  if (expanded.value[section] && vc) {
    if (section === 'forecast') deals.loadForecast(vc)
    if (section === 'debt') deals.loadDebtService(vc)
    if (section === 'cash') deals.loadCashSchedule(vc)
    if (section === 'capcalls') deals.loadCapitalCalls(vc)
    if (section === 'xirr') deals.loadXirrCashflows(vc)
    if (section === 'roe') deals.loadRoeAudit(vc)
    if (section === 'moic') deals.loadMoicAudit(vc)
  }
}

// Reset expansions when deal changes
watch(() => deals.currentVcode, () => { expanded.value = {} })

// ============================================================
// Merged XIRR Cash Flows (side-by-side partners)
// ============================================================

const xirrMerged = computed(() => {
  const xirr = deals.currentXirr
  if (!xirr) return null

  const partners = Object.keys(xirr)
  if (partners.length === 0) return null

  // Build a map keyed by "date|description" -> { date, description, partner1: amt, partner2: amt, deal: amt }
  const rowMap = new Map<string, Record<string, any>>()

  for (const partner of partners) {
    for (const cf of xirr[partner]) {
      const key = `${cf.date}|${cf.description}`
      if (!rowMap.has(key)) {
        rowMap.set(key, { date: cf.date, description: cf.description, _deal: 0 })
      }
      const row = rowMap.get(key)!
      row[partner] = (row[partner] || 0) + cf.amount
      row._deal += cf.amount
    }
  }

  // Sort by date
  const rows = [...rowMap.values()].sort((a, b) => a.date.localeCompare(b.date))

  // Build columns
  const columns = [
    { key: 'date', label: 'Date' },
    { key: 'description', label: 'Description' },
    ...partners.map(p => ({ key: p, label: p, format: 'currency', align: 'right' })),
    { key: '_deal', label: 'Deal', format: 'currency', align: 'right' },
  ]

  return { columns, rows }
})

// ============================================================
// Formatters
// ============================================================

function fmtInt(v: any): string {
  if (v == null || v !== v) return '—'
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 0, maximumFractionDigits: 0 }).format(v)
}
function fmtCur(v: any): string {
  if (v == null || v !== v) return '—'
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 0 }).format(v)
}
function fmtPct(v: any): string {
  if (v == null || v !== v) return '—'
  return (v * 100).toFixed(2) + '%'
}
function fmtMult(v: any): string {
  if (v == null || v !== v) return '—'
  return v.toFixed(2) + 'x'
}
function fmtNum(v: any): string {
  if (v == null || v !== v) return '—'
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(v)
}
function fmtDate(v: any): string {
  if (!v) return '—'
  const d = new Date(v)
  return isNaN(d.getTime()) ? String(v) : d.toLocaleDateString('en-US')
}

// ============================================================
// Partner Returns columns
// ============================================================

const partnerCols = [
  { key: 'partner', label: 'Partner' },
  { key: 'contributions', label: 'Contributions', format: 'currency', align: 'right' },
  { key: 'cf_distributions', label: 'CF Distributions', format: 'currency', align: 'right' },
  { key: 'cap_distributions', label: 'Cap Distributions', format: 'currency', align: 'right' },
  { key: 'total_distributions', label: 'Total Distributions', format: 'currency', align: 'right' },
  { key: 'unrealized_nav', label: 'Unrealized NAV', format: 'currency', align: 'right' },
  { key: 'capital_outstanding', label: 'Capital O/S', format: 'currency', align: 'right' },
  { key: 'pref_unpaid_compounded', label: 'Pref Unpaid', format: 'currency', align: 'right' },
  { key: 'pref_accrued_current_year', label: 'Pref Accrued CY', format: 'currency', align: 'right' },
  { key: 'irr', label: 'IRR', format: 'percent', align: 'right' },
  { key: 'roe', label: 'ROE', format: 'percent', align: 'right' },
  { key: 'moic', label: 'MOIC', format: 'multiple', align: 'right' },
]

// ============================================================
// Deal metrics
// ============================================================

const dealMetrics = computed(() => {
  const ds = deals.currentSummary
  if (!ds) return []
  return [
    { label: 'Deal IRR', value: ds.deal_irr, format: 'percent' },
    { label: 'Deal ROE', value: ds.deal_roe, format: 'percent' },
    { label: 'Deal MOIC', value: ds.deal_moic, format: 'number' },
    { label: 'Contributions', value: ds.total_contributions, format: 'currency' },
    { label: 'CF Distributions', value: ds.total_cf_distributions, format: 'currency' },
    { label: 'Cap Distributions', value: ds.total_cap_distributions, format: 'currency' },
  ]
})

// ============================================================
// Loan schedule columns
// ============================================================

const loanCols = [
  { key: 'loan_id', label: 'Loan ID' },
  { key: 'int_type', label: 'Rate Type' },
  { key: 'orig_amount', label: 'Original Amount', format: 'currency', align: 'right' },
  { key: 'orig_date', label: 'Origination' },
  { key: 'maturity_date', label: 'Maturity' },
  { key: 'fixed_rate', label: 'Rate', format: 'percent', align: 'right' },
  { key: 'loan_term_m', label: 'Term (mo)', align: 'right' },
  { key: 'amort_term_m', label: 'Amort (mo)', align: 'right' },
  { key: 'io_months', label: 'IO (mo)', align: 'right' },
]

// ============================================================
// Excel download helper
// ============================================================

async function downloadExcel(url: string, filename: string) {
  const token = localStorage.getItem('token')
  const resp = await fetch(url, {
    headers: { 'Authorization': `Bearer ${token}` },
  })
  if (!resp.ok) return
  const blob = await resp.blob()
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = filename
  a.click()
  URL.revokeObjectURL(a.href)
}
</script>

<template>
  <div class="deal-analysis">
    <ProgressOverlay :visible="deals.computing" message="Computing deal..." />

    <!-- Deal Selector -->
    <div class="deal-selector">
      <label>Select Deal:</label>
      <select @change="onDealSelect" :value="deals.currentVcode">
        <option value="">-- Choose a deal --</option>
        <option v-for="d in data.deals" :key="d.vcode" :value="d.vcode">
          {{ d.Investment_Name || d.vcode }} ({{ d.vcode }})
        </option>
      </select>
    </div>

    <p v-if="deals.error" class="error">{{ deals.error }}</p>

    <!-- ============================================================ -->
    <!-- Results -->
    <!-- ============================================================ -->
    <template v-if="deals.hasResult">

      <!-- Full Workbook Download -->
      <div class="full-download">
        <button class="btn-download" @click="downloadExcel(`/api/deals/${deals.currentVcode}/excel/full`, `deal_analysis_${deals.currentVcode}.xlsx`)">
          Download Full Deal Analysis (Excel)
        </button>
      </div>

      <!-- Deal Header -->
      <div v-if="deals.currentHeader" class="section header-section">
        <div class="header-grid">
          <div class="header-col">
            <h3>Deal Information</h3>
            <table class="info-table">
              <tr><td>vCode</td><td>{{ deals.currentHeader.metadata.vcode }}</td></tr>
              <tr><td>Investment Name</td><td>{{ deals.currentHeader.metadata.Investment_Name }}</td></tr>
              <tr><td>InvestmentID</td><td>{{ deals.currentHeader.metadata.InvestmentID }}</td></tr>
              <tr><td>Asset Type</td><td>{{ deals.currentHeader.metadata.Asset_Type }}</td></tr>
              <tr><td>Lifecycle</td><td>{{ deals.currentHeader.metadata.Lifecycle }}</td></tr>
              <tr><td>Units</td><td>{{ deals.currentHeader.metadata.Total_Units }}</td></tr>
              <tr><td>SQF</td><td>{{ fmtNum(deals.currentHeader.metadata.Total_SQF) }}</td></tr>
              <tr><td>Acquisition Date</td><td>{{ deals.currentHeader.metadata.Acquisition_Date }}</td></tr>
            </table>
            <p v-if="deals.currentHeader.sub_portfolio_msg" class="sub-msg">
              {{ deals.currentHeader.sub_portfolio_msg }}
            </p>
          </div>
          <div class="header-col">
            <h3>Capitalization</h3>
            <table class="info-table" v-if="deals.currentHeader.cap_data">
              <tr><td>Debt</td><td class="right">{{ fmtCur(deals.currentHeader.cap_data.debt) }}</td></tr>
              <tr><td>Pref Equity</td><td class="right">{{ fmtCur(deals.currentHeader.cap_data.pref_equity) }}</td></tr>
              <tr class="total-border"><td>Ptr Equity</td><td class="right">{{ fmtCur(deals.currentHeader.cap_data.partner_equity) }}</td></tr>
              <tr><td class="bold">Total Cap</td><td class="right bold">{{ fmtCur(deals.currentHeader.cap_data.total_cap) }}</td></tr>
              <tr><td>Valuation</td><td class="right">{{ fmtCur(deals.currentHeader.cap_data.current_valuation) }}</td></tr>
              <tr><td>Cap Rate</td><td class="right">{{ fmtPct(deals.currentHeader.cap_data.cap_rate) }}</td></tr>
              <tr><td>PE Exp (Cap)</td><td class="right">{{ fmtPct(deals.currentHeader.cap_data.pe_exposure_cap) }}</td></tr>
              <tr><td>PE Exp (Val)</td><td class="right">{{ fmtPct(deals.currentHeader.cap_data.pe_exposure_value) }}</td></tr>
            </table>
          </div>
        </div>
      </div>

      <!-- Deal-Level Summary -->
      <div class="section metrics-section">
        <h3>Deal-Level Summary</h3>
        <div class="metrics-row">
          <KpiCard v-for="m in dealMetrics" :key="m.label" :label="m.label" :value="m.value" :format="m.format" />
        </div>
      </div>

      <!-- Partner Returns -->
      <div class="section">
        <div class="section-title-row">
          <h3>Partner Returns</h3>
          <button class="btn-download-sm" @click="downloadExcel(`/api/deals/${deals.currentVcode}/excel/partner-returns`, `partner_returns_${deals.currentVcode}.xlsx`)">Excel</button>
        </div>
        <DataTable :columns="partnerCols" :rows="deals.currentPartners"
          :rowClass="(row: any) => !String(row.partner || '').startsWith('OP') ? 'row-highlight' : undefined" />
      </div>

      <!-- Annual Forecast -->
      <div class="section expandable" :class="{ open: expanded.forecast }">
        <div class="section-title-row">
          <h3 @click="toggle('forecast')" class="section-header">
            <span class="chevron">{{ expanded.forecast ? '▾' : '▸' }}</span>
            Annual Operating Forecast & Waterfall Summary
          </h3>
          <button v-if="expanded.forecast && deals.currentForecast" class="btn-download-sm" @click.stop="downloadExcel(`/api/deals/${deals.currentVcode}/excel/forecast`, `annual_forecast_${deals.currentVcode}.xlsx`)">Excel</button>
        </div>
        <div v-if="expanded.forecast" class="section-body">
          <p v-if="deals.loadingSection === 'forecast'" class="loading-msg">Loading forecast...</p>
          <div v-else-if="deals.currentForecast" class="forecast-table-wrapper">
            <table class="forecast-table" v-if="deals.currentForecast.rows.length">
              <thead>
                <tr>
                  <th class="label-col">Line Item</th>
                  <th v-for="y in deals.currentForecast.years" :key="y" class="year-col">{{ y }}</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(row, i) in deals.currentForecast.rows" :key="i"
                    :class="{ 'section-header-row': row.label.endsWith(':'), 'blank-row': row.label.trim() === '', 'underline-row': ['Expenses', 'Capital Expenditures'].includes(row.label.trim()), 'topline-row': row.label.trim() === 'Total Distributions' }">
                  <td class="label-col">{{ row.label }}</td>
                  <td v-for="y in deals.currentForecast.years" :key="y" class="year-col">
                    {{ (row.label.trim() === '' || row.label.endsWith(':')) ? '' : row.values[String(y)] != null ? (row.label === 'Debt Service Coverage Ratio' ? row.values[String(y)].toFixed(2) : fmtInt(row.values[String(y)])) : '' }}
                  </td>
                </tr>
              </tbody>
            </table>
            <p v-else class="placeholder">No forecast data available</p>
          </div>
        </div>
      </div>

      <!-- Diagnostics -->
      <div v-if="deals.currentDebugMsgs.length" class="section expandable" :class="{ open: expanded.diag }">
        <h3 @click="expanded.diag = !expanded.diag" class="section-header">
          <span class="chevron">{{ expanded.diag ? '▾' : '▸' }}</span>
          Diagnostics ({{ deals.currentDebugMsgs.length }})
        </h3>
        <div v-if="expanded.diag" class="section-body">
          <ul class="diag-list">
            <li v-for="(msg, i) in deals.currentDebugMsgs" :key="i">{{ msg }}</li>
          </ul>
        </div>
      </div>

      <!-- Debt Service -->
      <div class="section expandable" :class="{ open: expanded.debt }">
        <div class="section-title-row">
          <h3 @click="toggle('debt')" class="section-header">
            <span class="chevron">{{ expanded.debt ? '▾' : '▸' }}</span>
            Debt Service
          </h3>
          <button v-if="expanded.debt && deals.currentDebt" class="btn-download-sm" @click.stop="downloadExcel(`/api/deals/${deals.currentVcode}/excel/debt-service`, `debt_service_${deals.currentVcode}.xlsx`)">Excel</button>
        </div>
        <div v-if="expanded.debt" class="section-body">
          <p v-if="deals.loadingSection === 'debt'" class="loading-msg">Loading debt service...</p>
          <template v-else-if="deals.currentDebt">
            <!-- Loan Summary -->
            <h4>Loan Summary</h4>
            <DataTable v-if="deals.currentDebt.loans.length" :columns="loanCols" :rows="deals.currentDebt.loans" />
            <p v-else class="placeholder">No loans</p>

            <!-- Amortization Schedule -->
            <template v-for="loan in deals.currentDebt.loans" :key="loan.loan_id">
              <details class="amort-detail">
                <summary>Amortization Schedule — {{ loan.loan_id }}</summary>
                <div class="amort-table-wrapper">
                  <DataTable
                    :columns="[
                      { key: 'event_date', label: 'Date' },
                      { key: 'rate', label: 'Rate', format: 'percent', align: 'right' },
                      { key: 'interest', label: 'Interest', format: 'currency', align: 'right' },
                      { key: 'principal', label: 'Principal', format: 'currency', align: 'right' },
                      { key: 'payment', label: 'Payment', format: 'currency', align: 'right' },
                      { key: 'ending_balance', label: 'Ending Balance', format: 'currency', align: 'right' },
                    ]"
                    :rows="deals.currentDebt.loan_schedule.filter((r: any) => r.LoanID === loan.loan_id)"
                  />
                </div>
              </details>
            </template>

            <!-- Sale Proceeds -->
            <template v-if="deals.currentDebt.sale_proceeds">
              <h4>Sale Proceeds Calculation</h4>
              <div class="sale-grid">
                <div>
                  <h5>Inputs</h5>
                  <table class="info-table">
                    <tr><td>Sale Date</td><td>{{ deals.currentDebt.sale_proceeds.Sale_Date }}</td></tr>
                    <tr><td>NOI (12mo after sale)</td><td class="right">{{ fmtCur(deals.currentDebt.sale_proceeds.NOI_12m_After_Sale) }}</td></tr>
                    <tr><td>Exit Cap Rate</td><td class="right">{{ fmtPct(deals.currentDebt.sale_proceeds.CapRate_Sale) }}</td></tr>
                  </table>
                </div>
                <div>
                  <h5>Calculation</h5>
                  <table class="info-table">
                    <tr><td>Implied Value</td><td class="right">{{ fmtCur(deals.currentDebt.sale_proceeds.Implied_Value) }}</td></tr>
                    <tr><td>Less Selling Costs (2%)</td><td class="right">{{ fmtCur(deals.currentDebt.sale_proceeds.Less_Selling_Cost_2pct) }}</td></tr>
                    <tr><td>Net of Costs</td><td class="right">{{ fmtCur(deals.currentDebt.sale_proceeds.Value_Net_Selling_Cost) }}</td></tr>
                    <tr><td>Less Loan Payoff</td><td class="right">{{ fmtCur(deals.currentDebt.sale_proceeds.Less_Loan_Balances) }}</td></tr>
                    <tr><td class="bold">Net Sale Proceeds</td><td class="right bold">{{ fmtCur(deals.currentDebt.sale_proceeds.Net_Sale_Proceeds) }}</td></tr>
                  </table>
                </div>
              </div>
            </template>
          </template>
        </div>
      </div>

      <!-- Cash Management -->
      <div class="section expandable" :class="{ open: expanded.cash }">
        <div class="section-title-row">
          <h3 @click="toggle('cash')" class="section-header">
            <span class="chevron">{{ expanded.cash ? '▾' : '▸' }}</span>
            Cash Management & Reserves
          </h3>
          <button v-if="expanded.cash && deals.currentCash" class="btn-download-sm" @click.stop="downloadExcel(`/api/deals/${deals.currentVcode}/excel/cash-schedule`, `cash_schedule_${deals.currentVcode}.xlsx`)">Excel</button>
        </div>
        <div v-if="expanded.cash" class="section-body">
          <p v-if="deals.loadingSection === 'cash'" class="loading-msg">Loading cash data...</p>
          <template v-else-if="deals.currentCash">
            <div class="metrics-row">
              <KpiCard label="Beginning Cash" :value="deals.currentCash.beginning_cash" format="currency" />
              <KpiCard label="Total CapEx Paid" :value="deals.currentCash.summary.total_capex_paid" format="currency" />
              <KpiCard label="Total Distributable" :value="deals.currentCash.summary.total_distributable" format="currency" />
              <KpiCard label="Ending Cash" :value="deals.currentCash.summary.ending_cash" format="currency" />
            </div>
            <details v-if="deals.currentCash.schedule.length">
              <summary>Cash Flow Schedule</summary>
              <DataTable
                :columns="[
                  { key: 'event_date', label: 'Date' },
                  { key: 'beginning_cash', label: 'Begin Cash', format: 'currency', align: 'right' },
                  { key: 'capital_call', label: 'Capital Call', format: 'currency', align: 'right' },
                  { key: 'operating_cf', label: 'Operating CF', format: 'currency', align: 'right' },
                  { key: 'distributable', label: 'Distributable', format: 'currency', align: 'right' },
                  { key: 'ending_cash', label: 'End Cash', format: 'currency', align: 'right' },
                ]"
                :rows="deals.currentCash.schedule"
              />
            </details>
          </template>
        </div>
      </div>

      <!-- Capital Calls -->
      <div class="section expandable" :class="{ open: expanded.capcalls }">
        <div class="section-title-row">
          <h3 @click="toggle('capcalls')" class="section-header">
            <span class="chevron">{{ expanded.capcalls ? '▾' : '▸' }}</span>
            Capital Calls Schedule
          </h3>
          <button v-if="expanded.capcalls && deals.currentCapCalls" class="btn-download-sm" @click.stop="downloadExcel(`/api/deals/${deals.currentVcode}/excel/capital-calls`, `capital_calls_${deals.currentVcode}.xlsx`)">Excel</button>
        </div>
        <div v-if="expanded.capcalls" class="section-body">
          <p v-if="deals.loadingSection === 'capcalls'" class="loading-msg">Loading capital calls...</p>
          <template v-else-if="deals.currentCapCalls">
            <DataTable v-if="deals.currentCapCalls.length"
              :columns="[
                { key: 'call_date', label: 'Date' },
                { key: 'investor_id', label: 'Investor' },
                { key: 'typename', label: 'Type' },
                { key: 'amount', label: 'Amount', format: 'currency', align: 'right' },
              ]"
              :rows="deals.currentCapCalls"
            />
            <p v-else class="placeholder">No capital calls</p>
          </template>
        </div>
      </div>

      <!-- XIRR Cash Flows -->
      <div class="section expandable" :class="{ open: expanded.xirr }">
        <div class="section-title-row">
          <h3 @click="toggle('xirr')" class="section-header">
            <span class="chevron">{{ expanded.xirr ? '▾' : '▸' }}</span>
            XIRR Cash Flows
          </h3>
          <button v-if="expanded.xirr && xirrMerged" class="btn-download-sm" @click.stop="downloadExcel(`/api/deals/${deals.currentVcode}/excel/xirr-cashflows`, `xirr_cashflows_${deals.currentVcode}.xlsx`)">Excel</button>
        </div>
        <div v-if="expanded.xirr" class="section-body">
          <p v-if="deals.loadingSection === 'xirr'" class="loading-msg">Loading XIRR data...</p>
          <template v-else-if="xirrMerged">
            <DataTable :columns="xirrMerged.columns" :rows="xirrMerged.rows" />
          </template>
        </div>
      </div>

      <!-- ROE Audit -->
      <div class="section expandable" :class="{ open: expanded.roe }">
        <h3 @click="toggle('roe')" class="section-header">
          <span class="chevron">{{ expanded.roe ? '▾' : '▸' }}</span>
          ROE Audit — Return on Equity Breakdown
        </h3>
        <div v-if="expanded.roe" class="section-body">
          <p v-if="deals.loadingSection === 'roe'" class="loading-msg">Loading ROE audit...</p>
          <template v-else-if="deals.currentRoe">
            <button class="btn-download" @click="downloadExcel(`/api/deals/${deals.currentVcode}/roe-audit/excel`, `roe_audit_${deals.currentVcode}.xlsx`)">
              Download Excel
            </button>

            <div v-for="pa in deals.currentRoe.partners" :key="pa.partner" class="audit-partner">
              <h4>{{ pa.partner }}</h4>

              <!-- Timeline -->
              <h5>Capital Balance Timeline</h5>
              <DataTable
                :columns="[
                  { key: 'Date', label: 'Date' },
                  { key: 'Event', label: 'Event' },
                  { key: 'Amount', label: 'Amount', format: 'currency', align: 'right' },
                  { key: 'Capital Balance', label: 'Capital Balance', format: 'currency', align: 'right' },
                  { key: 'Days at Balance', label: 'Days', align: 'right' },
                  { key: 'Weighted Capital', label: 'Weighted Capital', format: 'currency', align: 'right' },
                ]"
                :rows="pa.timeline"
              />

              <!-- CF Distributions -->
              <h5>CF Distributions</h5>
              <DataTable
                :columns="[
                  { key: 'Date', label: 'Date' },
                  { key: 'Amount', label: 'Amount', format: 'currency', align: 'right' },
                ]"
                :rows="pa.cf_distributions"
              />

              <!-- Summary -->
              <div class="metrics-row" v-if="pa.summary">
                <KpiCard label="Inception" :value="pa.summary.inception" />
                <KpiCard label="End" :value="pa.summary.end" />
                <KpiCard label="Years" :value="pa.summary.years?.toFixed(2)" />
                <KpiCard label="CF Distributions" :value="pa.summary.total_cf_dist" format="currency" />
                <KpiCard label="Wtd Avg Capital" :value="pa.summary.wac" format="currency" />
                <KpiCard label="ROE" :value="pa.summary.roe" format="percent" />
              </div>
            </div>

            <!-- Deal Level -->
            <div class="audit-partner deal-level">
              <h4>Deal-Level ROE</h4>
              <DataTable
                :columns="[
                  { key: 'Date', label: 'Date' },
                  { key: 'Event', label: 'Event' },
                  { key: 'Amount', label: 'Amount', format: 'currency', align: 'right' },
                  { key: 'Capital Balance', label: 'Capital Balance', format: 'currency', align: 'right' },
                  { key: 'Days at Balance', label: 'Days', align: 'right' },
                  { key: 'Weighted Capital', label: 'Weighted Capital', format: 'currency', align: 'right' },
                ]"
                :rows="deals.currentRoe.deal_level.timeline"
              />
              <div class="metrics-row" v-if="deals.currentRoe.deal_level.summary">
                <KpiCard label="Years" :value="deals.currentRoe.deal_level.summary.years?.toFixed(2)" />
                <KpiCard label="CF Distributions" :value="deals.currentRoe.deal_level.summary.total_cf_dist" format="currency" />
                <KpiCard label="Wtd Avg Capital" :value="deals.currentRoe.deal_level.summary.wac" format="currency" />
                <KpiCard label="ROE" :value="deals.currentRoe.deal_level.summary.roe" format="percent" />
              </div>
            </div>
          </template>
        </div>
      </div>

      <!-- MOIC Audit -->
      <div class="section expandable" :class="{ open: expanded.moic }">
        <h3 @click="toggle('moic')" class="section-header">
          <span class="chevron">{{ expanded.moic ? '▾' : '▸' }}</span>
          MOIC Audit — Multiple on Invested Capital
        </h3>
        <div v-if="expanded.moic" class="section-body">
          <p v-if="deals.loadingSection === 'moic'" class="loading-msg">Loading MOIC audit...</p>
          <template v-else-if="deals.currentMoic">
            <button class="btn-download" @click="downloadExcel(`/api/deals/${deals.currentVcode}/moic-audit/excel`, `moic_audit_${deals.currentVcode}.xlsx`)">
              Download Excel
            </button>

            <div v-for="pa in deals.currentMoic.partners" :key="pa.partner" class="audit-partner">
              <h4>{{ pa.partner }}</h4>

              <DataTable
                :columns="[
                  { key: 'Date', label: 'Date' },
                  { key: 'Description', label: 'Description' },
                  { key: 'Type', label: 'Type' },
                  { key: 'Amount', label: 'Amount', format: 'currency', align: 'right' },
                ]"
                :rows="pa.breakdown"
              />

              <div class="metrics-row">
                <KpiCard label="Contributions" :value="pa.summary.contributions" format="currency" />
                <KpiCard label="CF Distributions" :value="pa.summary.cf_dist" format="currency" />
                <KpiCard label="Cap Distributions" :value="pa.summary.cap_dist" format="currency" />
                <KpiCard label="Total Distributions" :value="pa.summary.total_dist" format="currency" />
                <KpiCard label="Unrealized NAV" :value="pa.summary.unrealized" format="currency" />
                <KpiCard label="MOIC" :value="pa.summary.moic" format="number" />
              </div>
            </div>

            <!-- Deal Level -->
            <div class="audit-partner deal-level">
              <h4>Deal-Level MOIC</h4>
              <p class="note">{{ deals.currentMoic.deal_level.note }}</p>
              <div class="metrics-row">
                <KpiCard label="Contributions" :value="deals.currentMoic.deal_level.total_contributions" format="currency" />
                <KpiCard label="CF Distributions" :value="deals.currentMoic.deal_level.total_cf_distributions" format="currency" />
                <KpiCard label="Cap Distributions" :value="deals.currentMoic.deal_level.total_cap_distributions" format="currency" />
                <KpiCard label="Total Distributions" :value="deals.currentMoic.deal_level.total_distributions" format="currency" />
                <KpiCard label="MOIC" :value="deals.currentMoic.deal_level.deal_moic" format="number" />
              </div>
            </div>
          </template>
        </div>
      </div>

    </template>
  </div>
</template>

<style scoped>
.deal-analysis {
  padding: 0;
}

.deal-selector {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 20px;
}
.deal-selector label {
  font-weight: 600;
  font-size: 14px;
}
.deal-selector select {
  padding: 8px 12px;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  font-size: 14px;
  min-width: 350px;
}

.error {
  color: #dc3545;
  margin-bottom: 12px;
}

.section {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 12px;
}
.section h3 {
  font-size: 14px;
  margin-bottom: 12px;
}
.section h4 {
  font-size: 13px;
  margin: 16px 0 8px;
  color: var(--color-text-secondary);
}
.section h5 {
  font-size: 12px;
  margin: 12px 0 6px;
  color: var(--color-text-secondary);
}

/* Expandable sections */
.expandable .section-header {
  cursor: pointer;
  user-select: none;
  margin-bottom: 0;
}
.expandable .section-body {
  margin-top: 12px;
}
.chevron {
  font-size: 12px;
  margin-right: 6px;
  display: inline-block;
  width: 14px;
}

/* Header grid */
.header-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
}
.info-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
.info-table td {
  padding: 4px 8px;
  border-bottom: 1px solid var(--color-border);
}
.info-table td:first-child {
  font-weight: 600;
  width: 40%;
  color: var(--color-text-secondary);
}
.right { text-align: right; }
.bold { font-weight: 800 !important; }
.total-border td { border-bottom: 2px solid #000; }
.sub-msg {
  font-size: 12px;
  color: var(--color-text-secondary);
  font-style: italic;
  margin-top: 8px;
}

/* Metrics */
.metrics-row {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin: 12px 0;
}
.metrics-section {
  margin-top: 12px;
}

/* Forecast table */
.forecast-table-wrapper {
  overflow-x: auto;
}
.forecast-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}
.forecast-table th,
.forecast-table td {
  padding: 4px 8px;
  border-bottom: 1px solid var(--color-border);
  white-space: nowrap;
}
.forecast-table th {
  background: var(--color-accent);
  color: white;
  font-weight: 600;
  text-align: right;
}
.forecast-table th.label-col {
  text-align: left;
  min-width: 200px;
}
.forecast-table td.year-col {
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.forecast-table td.label-col {
  font-weight: 500;
}
.section-header-row td {
  font-weight: 700 !important;
  background: #f0f4f8;
}
.blank-row td {
  height: 8px;
  border-bottom: none;
}
.underline-row td {
  border-bottom: 2px solid #000;
}
.topline-row td {
  border-top: 2px solid #000;
}

/* Sale grid */
.sale-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
}

/* Amortization details */
.amort-detail {
  margin: 8px 0;
}
.amort-detail summary {
  cursor: pointer;
  font-weight: 600;
  font-size: 13px;
  color: var(--color-accent);
}
.amort-table-wrapper {
  max-height: 400px;
  overflow-y: auto;
}

/* Audit sections */
.audit-partner {
  border-top: 1px solid var(--color-border);
  padding-top: 12px;
  margin-top: 12px;
}
.audit-partner:first-child {
  border-top: none;
  padding-top: 0;
  margin-top: 0;
}
.deal-level {
  border-top: 2px solid var(--color-accent);
  margin-top: 20px;
  padding-top: 16px;
}

.xirr-partner {
  margin-bottom: 16px;
}

/* Buttons */
.btn-download {
  padding: 6px 14px;
  background: var(--color-accent);
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 13px;
  margin-bottom: 12px;
}
.btn-download:hover {
  opacity: 0.9;
}
.btn-download-sm {
  padding: 3px 10px;
  background: var(--color-accent);
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 11px;
  white-space: nowrap;
}
.btn-download-sm:hover {
  opacity: 0.9;
}

/* Section title with download button */
.section-title-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}
.section-title-row h3 {
  flex: 1;
}

/* Full workbook download */
.full-download {
  margin-bottom: 12px;
  text-align: right;
}

.loading-msg {
  color: var(--color-text-secondary);
  font-style: italic;
  padding: 8px 0;
}
.placeholder {
  color: var(--color-text-secondary);
  font-style: italic;
  padding: 20px 0;
  text-align: center;
}
.note {
  font-size: 12px;
  color: var(--color-text-secondary);
  font-style: italic;
  margin-bottom: 8px;
}

.diag-list {
  font-size: 12px;
  color: var(--color-text-secondary);
  padding-left: 20px;
}
.diag-list li {
  margin-bottom: 4px;
}
</style>
