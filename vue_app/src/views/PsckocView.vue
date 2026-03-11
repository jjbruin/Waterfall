<script setup lang="ts">
import { onMounted, computed } from 'vue'
import { usePsckocStore } from '../stores/psckoc'
import DataTable from '../components/common/DataTable.vue'
import KpiCard from '../components/common/KpiCard.vue'
import ProgressOverlay from '../components/common/ProgressOverlay.vue'
import api from '../api/client'
import { useDataStore } from '../stores/data'

const psckoc = usePsckocStore()
const dataStore = useDataStore()

onMounted(async () => {
  if (psckoc.deals.length === 0) {
    await psckoc.loadDeals()
  }
  // Try to load cached results
  if (!psckoc.results) {
    await psckoc.loadCachedResults()
  }
})

// Computed helpers
const partnerResults = computed(() => psckoc.results?.partner_results || [])
const dealSummary = computed(() => psckoc.results?.deal_summary || {})
const incomeSchedule = computed(() => psckoc.results?.income_schedule || [])
const memberAllocations = computed(() => psckoc.results?.member_allocations || [])
const amFeeSchedule = computed(() => psckoc.results?.am_fee_schedule || [])
const computeErrors = computed(() => psckoc.results?.errors || [])

// Income summary grouped by Source Deal + Type
const incomeSummary = computed(() => {
  const groups: Record<string, { 'Source Deal': string; Type: string; Amount: number }> = {}
  for (const row of incomeSchedule.value) {
    const key = `${row['Source Deal']}|${row.Type}`
    if (!groups[key]) {
      groups[key] = { 'Source Deal': row['Source Deal'], Type: row.Type, Amount: 0 }
    }
    groups[key].Amount += row.Amount
  }
  return Object.values(groups)
})

// CF and Cap allocation summaries
const cfAllocations = computed(() =>
  memberAllocations.value.filter((r: any) => r.Type === 'CF')
)
const capAllocations = computed(() =>
  memberAllocations.value.filter((r: any) => r.Type === 'Cap')
)

// Partner returns table rows (including total)
const returnsTableRows = computed(() => {
  const rows = partnerResults.value.map((pr: any) => ({
    Partner: pr.partner,
    Contributions: pr.contributions,
    'CF Distributions': pr.cf_distributions,
    'Cap Distributions': pr.cap_distributions,
    'Total Distributions': pr.total_distributions,
    IRR: pr.irr,
    ROE: pr.roe,
    MOIC: pr.moic,
    _is_deal_total: false,
  }))
  // Total row
  rows.push({
    Partner: 'PSCKOC Total',
    Contributions: dealSummary.value.total_contributions || 0,
    'CF Distributions': partnerResults.value.reduce((s: number, p: any) => s + p.cf_distributions, 0),
    'Cap Distributions': partnerResults.value.reduce((s: number, p: any) => s + p.cap_distributions, 0),
    'Total Distributions': dealSummary.value.total_distributions || 0,
    IRR: dealSummary.value.deal_irr,
    ROE: null,
    MOIC: dealSummary.value.deal_moic || 0,
    _is_deal_total: true,
  })
  return rows
})

// Expanded section state
const sections = {
  income: { expanded: false },
  allocations: { expanded: false },
  amfee: { expanded: false },
  xirr: { expanded: false },
}

// Table columns
const dealColumns = [
  { key: 'name', label: 'Investment Name' },
  { key: 'vcode', label: 'Vcode' },
  { key: 'asset_type', label: 'Asset Type' },
  { key: 'sale_status', label: 'Status' },
  { key: 'ppi_entity', label: 'PPI Entity' },
  { key: 'psckoc_pct', label: 'PSCKOC %', format: 'percent', align: 'right' },
]

const returnsColumns = [
  { key: 'Partner', label: 'Partner' },
  { key: 'Contributions', label: 'Contributions', format: 'currency', align: 'right' },
  { key: 'CF Distributions', label: 'CF Distributions', format: 'currency', align: 'right' },
  { key: 'Cap Distributions', label: 'Cap Distributions', format: 'currency', align: 'right' },
  { key: 'Total Distributions', label: 'Total Distributions', format: 'currency', align: 'right' },
  { key: 'IRR', label: 'IRR', format: 'percent', align: 'right' },
  { key: 'ROE', label: 'ROE', format: 'percent', align: 'right' },
  { key: 'MOIC', label: 'MOIC', format: 'multiple', align: 'right' },
]

const incomeSummaryColumns = [
  { key: 'Source Deal', label: 'Source Deal' },
  { key: 'Type', label: 'Type' },
  { key: 'Amount', label: 'Amount', format: 'currency2', align: 'right' },
]

const incomeDetailColumns = [
  { key: 'Date', label: 'Date' },
  { key: 'Source Entity', label: 'Source Entity' },
  { key: 'Source Deal', label: 'Source Deal' },
  { key: 'Type', label: 'Type' },
  { key: 'vState', label: 'vState' },
  { key: 'Amount', label: 'Amount', format: 'currency2', align: 'right' },
]

const allocColumns = [
  { key: 'iOrder', label: 'Order', align: 'right' },
  { key: 'Member', label: 'Member' },
  { key: 'vState', label: 'vState' },
  { key: 'vtranstype', label: 'Type' },
  { key: 'Amount', label: 'Amount', format: 'currency2', align: 'right' },
]

const amFeeColumns = [
  { key: 'Date', label: 'Date' },
  { key: 'Type', label: 'Type' },
  { key: 'Recipient', label: 'Recipient' },
  { key: 'Amount', label: 'Amount', format: 'currency2', align: 'right' },
]

const cashflowColumns = [
  { key: 'date', label: 'Date' },
  { key: 'amount', label: 'Amount', format: 'currency2', align: 'right' },
]

// Formatters
function fmtCur(v: any): string {
  if (v == null || v !== v) return '--'
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 0 }).format(v)
}
function fmtPct(v: any): string {
  if (v == null || v !== v) return 'N/A'
  return (v * 100).toFixed(2) + '%'
}
function fmtMult(v: any): string {
  if (v == null || v !== v) return '--'
  return v.toFixed(2) + 'x'
}

// Total AM fees
const totalAmFees = computed(() =>
  amFeeSchedule.value.reduce((s: number, r: any) => s + (r.Amount || 0), 0)
)

// Excel download
async function downloadExcel() {
  try {
    const res = await api.get('/api/psckoc/excel', { responseType: 'blob' })
    const url = URL.createObjectURL(new Blob([res.data]))
    const a = document.createElement('a')
    a.href = url
    a.download = 'psckoc_analysis.xlsx'
    a.click()
    URL.revokeObjectURL(url)
  } catch (e: any) {
    dataStore.addToast('Failed to download Excel: ' + (e.response?.data?.error || e.message), 'error')
  }
}
</script>

<template>
  <div class="psckoc">
    <h2>PSCKOC Entity Analysis</h2>
    <p class="subtitle">Upstream waterfall analysis — traces deal cash flows through to PSC1, KCREIT, and PCBLE.</p>

    <ProgressOverlay :visible="psckoc.computing" message="Running PSCKOC computation..." />

    <!-- Deal Portfolio -->
    <div class="section">
      <h3>Deal Portfolio</h3>
      <div v-if="psckoc.loadingDeals" class="placeholder">Loading deals...</div>
      <template v-else-if="psckoc.deals.length">
        <p class="deal-count">{{ psckoc.deals.length }} deals discovered linked to PSCKOC</p>
        <DataTable :columns="dealColumns" :rows="psckoc.deals" />
      </template>
      <p v-else class="placeholder">No PSCKOC deals found. Ensure PSCKOC waterfall steps or relationship records exist.</p>
    </div>

    <!-- Compute Button -->
    <div class="section compute-section">
      <button
        class="btn-compute"
        @click="psckoc.compute()"
        :disabled="psckoc.computing || psckoc.deals.length === 0"
      >
        {{ psckoc.computing ? 'Computing...' : 'Compute PSCKOC Returns' }}
      </button>
      <span v-if="psckoc.results" class="compute-status">
        {{ psckoc.results.deals_computed }} deals computed
      </span>
    </div>

    <!-- Errors -->
    <div v-if="computeErrors.length" class="error-section">
      <div v-for="(err, i) in computeErrors" :key="i" class="error-msg">{{ err }}</div>
    </div>

    <!-- Results -->
    <template v-if="psckoc.results">

      <!-- Partner Returns KPI Cards -->
      <div class="section">
        <h3>Partner Returns</h3>
        <div class="kpi-row">
          <div v-for="pr in partnerResults" :key="pr.partner" class="kpi-group">
            <div class="kpi-group-title">{{ pr.partner }}</div>
            <div class="kpi-cards">
              <div class="mini-kpi">
                <span class="mini-label">IRR</span>
                <span class="mini-value">{{ fmtPct(pr.irr) }}</span>
              </div>
              <div class="mini-kpi">
                <span class="mini-label">Distributions</span>
                <span class="mini-value">{{ fmtCur(pr.total_distributions) }}</span>
              </div>
              <div class="mini-kpi">
                <span class="mini-label">MOIC</span>
                <span class="mini-value">{{ fmtMult(pr.moic) }}</span>
              </div>
            </div>
          </div>
          <!-- Total -->
          <div class="kpi-group total-group">
            <div class="kpi-group-title">PSCKOC Total</div>
            <div class="kpi-cards">
              <div class="mini-kpi">
                <span class="mini-label">IRR</span>
                <span class="mini-value">{{ fmtPct(dealSummary.deal_irr) }}</span>
              </div>
              <div class="mini-kpi">
                <span class="mini-label">Distributions</span>
                <span class="mini-value">{{ fmtCur(dealSummary.total_distributions) }}</span>
              </div>
              <div class="mini-kpi">
                <span class="mini-label">MOIC</span>
                <span class="mini-value">{{ fmtMult(dealSummary.deal_moic) }}</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Returns Table -->
        <DataTable :columns="returnsColumns" :rows="returnsTableRows" :highlight-total="true" />
      </div>

      <!-- Income Schedule (expandable) -->
      <div class="section expandable" @click="sections.income.expanded = !sections.income.expanded">
        <h3 class="expand-header">
          <span class="expand-icon">{{ sections.income.expanded ? '▾' : '▸' }}</span>
          Income Schedule — Cash arriving at PSCKOC
        </h3>
      </div>
      <div v-if="sections.income.expanded" class="section-body">
        <template v-if="incomeSummary.length">
          <h4>Summary by Source Deal</h4>
          <DataTable :columns="incomeSummaryColumns" :rows="incomeSummary" />
          <h4>Full Detail</h4>
          <DataTable :columns="incomeDetailColumns" :rows="incomeSchedule" />
        </template>
        <p v-else class="placeholder">No income allocated to PSCKOC.</p>
      </div>

      <!-- Waterfall Allocations (expandable) -->
      <div class="section expandable" @click="sections.allocations.expanded = !sections.allocations.expanded">
        <h3 class="expand-header">
          <span class="expand-icon">{{ sections.allocations.expanded ? '▾' : '▸' }}</span>
          Waterfall Allocations — PSCKOC to Members
        </h3>
      </div>
      <div v-if="sections.allocations.expanded" class="section-body">
        <template v-if="memberAllocations.length">
          <template v-if="cfAllocations.length">
            <h4>CF Waterfall</h4>
            <DataTable :columns="allocColumns" :rows="cfAllocations" />
          </template>
          <template v-if="capAllocations.length">
            <h4>Cap Waterfall</h4>
            <DataTable :columns="allocColumns" :rows="capAllocations" />
          </template>
        </template>
        <p v-else class="placeholder">No member allocations found.</p>
      </div>

      <!-- AM Fee Schedule (expandable) -->
      <div class="section expandable" @click="sections.amfee.expanded = !sections.amfee.expanded">
        <h3 class="expand-header">
          <span class="expand-icon">{{ sections.amfee.expanded ? '▾' : '▸' }}</span>
          AM Fee Schedule
        </h3>
      </div>
      <div v-if="sections.amfee.expanded" class="section-body">
        <template v-if="amFeeSchedule.length">
          <DataTable :columns="amFeeColumns" :rows="amFeeSchedule" />
          <div class="am-total">Total AM Fees: <strong>{{ fmtCur(totalAmFees) }}</strong></div>
        </template>
        <p v-else class="placeholder">No AM fee entries found. (AMFee steps will appear after waterfall is configured with AMFee vState.)</p>
      </div>

      <!-- XIRR Cash Flows (expandable) -->
      <div class="section expandable" @click="sections.xirr.expanded = !sections.xirr.expanded">
        <h3 class="expand-header">
          <span class="expand-icon">{{ sections.xirr.expanded ? '▾' : '▸' }}</span>
          XIRR Cash Flows — Combined per Member
        </h3>
      </div>
      <div v-if="sections.xirr.expanded" class="section-body">
        <template v-if="partnerResults.length">
          <div v-for="pr in partnerResults" :key="pr.partner" class="cf-member-section">
            <h4>{{ pr.partner }}</h4>
            <template v-if="pr.combined_cashflows && pr.combined_cashflows.length">
              <DataTable :columns="cashflowColumns" :rows="pr.combined_cashflows" />
            </template>
            <p v-else class="placeholder-sm">No cashflows recorded.</p>
          </div>
        </template>
        <p v-else class="placeholder">No cashflow data available.</p>
      </div>

      <!-- Excel Export -->
      <div class="section">
        <button class="btn-download" @click="downloadExcel">
          Download PSCKOC Analysis (Excel)
        </button>
      </div>
    </template>
  </div>
</template>

<style scoped>
.psckoc { padding: 0 0 40px 0; }
h2 { font-size: 20px; margin-bottom: 4px; }
h3 { font-size: 15px; margin: 0; }
h4 { font-size: 13px; margin: 16px 0 8px 0; font-weight: 600; }
.subtitle { font-size: 13px; color: var(--color-text-secondary); margin-bottom: 16px; }

.section {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 12px;
}

.section-body {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-top: none;
  border-radius: 0 0 8px 8px;
  padding: 16px;
  margin-top: -12px;
  margin-bottom: 12px;
}

.expandable { cursor: pointer; }
.expandable:hover { background: #f8f9fa; }

.expand-header {
  display: flex;
  align-items: center;
  gap: 8px;
  user-select: none;
}

.expand-icon { font-size: 12px; width: 16px; }

.placeholder {
  color: var(--color-text-secondary);
  font-style: italic;
  text-align: center;
  padding: 20px 0;
}

.placeholder-sm {
  color: var(--color-text-secondary);
  font-style: italic;
  font-size: 13px;
  padding: 8px 0;
}

.deal-count {
  font-size: 13px;
  color: var(--color-text-secondary);
  margin-bottom: 8px;
}

.compute-section {
  display: flex;
  align-items: center;
  gap: 16px;
}

.btn-compute {
  padding: 10px 24px;
  background: var(--color-accent);
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
}
.btn-compute:hover { background: #3a63ad; }
.btn-compute:disabled { opacity: 0.6; cursor: not-allowed; }

.compute-status {
  font-size: 13px;
  color: var(--color-text-secondary);
}

.error-section { margin-bottom: 12px; }
.error-msg {
  color: #d32f2f;
  background: #ffebee;
  padding: 6px 12px;
  border-radius: 6px;
  font-size: 13px;
  margin-bottom: 4px;
}

/* KPI Row */
.kpi-row {
  display: flex;
  gap: 16px;
  margin-bottom: 16px;
  flex-wrap: wrap;
}

.kpi-group {
  background: #f8f9fa;
  border: 1px solid var(--color-border);
  border-radius: 8px;
  padding: 12px 16px;
  min-width: 160px;
}

.total-group {
  background: #e8edf5;
  border-color: var(--color-accent);
}

.kpi-group-title {
  font-size: 13px;
  font-weight: 600;
  margin-bottom: 8px;
}

.kpi-cards { display: flex; flex-direction: column; gap: 4px; }

.mini-kpi { display: flex; justify-content: space-between; gap: 12px; }

.mini-label {
  font-size: 11px;
  color: var(--color-text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.3px;
}

.mini-value {
  font-size: 14px;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}

.cf-member-section { margin-bottom: 16px; }

.am-total {
  font-size: 14px;
  margin-top: 8px;
  padding: 8px 0;
}

.btn-download {
  padding: 8px 20px;
  background: var(--color-pref, #548235);
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
}
.btn-download:hover { opacity: 0.9; }
</style>
