<script setup lang="ts">
import { onMounted, computed, watch } from 'vue'
import { useDashboardStore } from '../stores/dashboard'
import KpiCard from '../components/common/KpiCard.vue'
import DataTable from '../components/common/DataTable.vue'
import ProgressOverlay from '../components/common/ProgressOverlay.vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { BarChart, LineChart, PieChart } from 'echarts/charts'
import {
  GridComponent, TooltipComponent, LegendComponent,
  MarkLineComponent,
} from 'echarts/components'

use([CanvasRenderer, BarChart, LineChart, PieChart, GridComponent,
     TooltipComponent, LegendComponent, MarkLineComponent])

const CLR_DARK = '#1F4E79'
const CLR_ACCENT = '#ED7D31'
const CLR_LIGHT = '#B4D4F0'
const CLR_PREF = '#548235'
const CLR_OP = '#A6A6A6'

const dashboard = useDashboardStore()

onMounted(async () => {
  if (!dashboard.kpis) await dashboard.loadAll()
})

// --- NOI Trend Chart ---
const noiOption = computed(() => {
  const d = dashboard.noiData
  if (!d || !d.periods.length) return null
  const series: any[] = []
  if (d.occupancy.some((v: any) => v != null)) {
    series.push({
      name: 'Occupancy',
      type: 'bar',
      yAxisIndex: 0,
      data: d.occupancy,
      itemStyle: { color: CLR_LIGHT, opacity: 0.6 },
      barMaxWidth: 40,
    })
  }
  if (d.actual_noi.some((v: any) => v != null)) {
    series.push({
      name: 'Actual NOI',
      type: 'line',
      yAxisIndex: 1,
      data: d.actual_noi,
      lineStyle: { color: CLR_DARK },
      itemStyle: { color: CLR_DARK },
      symbol: 'circle',
      symbolSize: 6,
    })
  }
  if (d.uw_noi.some((v: any) => v != null)) {
    series.push({
      name: 'Underwritten NOI',
      type: 'line',
      yAxisIndex: 1,
      data: d.uw_noi,
      lineStyle: { color: CLR_ACCENT, type: 'dashed' },
      itemStyle: { color: CLR_ACCENT },
      symbol: 'circle',
      symbolSize: 6,
    })
  }
  return {
    tooltip: { trigger: 'axis' },
    legend: { bottom: 0 },
    grid: { left: 60, right: 60, top: 45, bottom: 40 },
    xAxis: { type: 'category', data: d.periods },
    yAxis: [
      { type: 'value', name: 'Occupancy %', position: 'left', min: 0, max: 100,
        nameTextStyle: { padding: [0, 0, 0, 0] } },
      { type: 'value', name: 'NOI ($ million)', position: 'right',
        axisLabel: { formatter: (v: number) => v.toFixed(1) },
        nameTextStyle: { padding: [0, 0, 0, 0] } },
    ],
    series,
  }
})

// --- Capital Structure Chart ---
const capOption = computed(() => {
  const c = dashboard.capStructure
  if (!c) return null
  return {
    tooltip: { trigger: 'axis', formatter: (params: any) => {
      return params.map((p: any) => `${p.seriesName}: $${p.value.toFixed(1)}M`).join('<br/>')
    }},
    legend: { bottom: 0 },
    grid: { left: 60, right: 100, top: 45, bottom: 40 },
    xAxis: { type: 'category', data: ['Portfolio'] },
    yAxis: { type: 'value', name: '$ million', axisLabel: { formatter: '{value}' } },
    series: [
      { name: 'Debt', type: 'bar', stack: 'total', data: [c.debt_m],
        itemStyle: { color: CLR_DARK }, barWidth: 100,
        markLine: { data: [{ yAxis: c.debt_m }],
          lineStyle: { color: CLR_DARK, type: 'dashed' },
          label: { formatter: `LTV: ${(c.avg_ltv * 100).toFixed(1)}%`, position: 'insideEndTop', fontSize: 11 },
          symbol: 'none' } },
      { name: 'Pref Equity', type: 'bar', stack: 'total', data: [c.pref_m],
        itemStyle: { color: CLR_PREF },
        markLine: { data: [{ yAxis: c.debt_m + c.pref_m }],
          lineStyle: { color: CLR_PREF, type: 'dashed' },
          label: { formatter: `Pref: ${(c.pref_exposure * 100).toFixed(1)}%`, position: 'insideEndTop', fontSize: 11 },
          symbol: 'none' } },
      { name: 'OP Equity', type: 'bar', stack: 'total', data: [c.partner_m],
        itemStyle: { color: CLR_OP } },
    ],
  }
})

// --- Occupancy by Type Chart ---
const occOption = computed(() => {
  const d = dashboard.occByType
  if (!d || !d.data.length) return null
  const sorted = [...d.data].sort((a, b) => a.occupancy - b.occupancy)
  return {
    tooltip: { trigger: 'axis' },
    grid: { left: 120, right: 60, top: 35, bottom: 35 },
    xAxis: { type: 'value', name: 'Occupancy %', nameLocation: 'center', nameGap: 20, min: 0, max: 100 },
    yAxis: { type: 'category', data: sorted.map(d => d.asset_type) },
    series: [{
      type: 'bar',
      data: sorted.map(d => ({
        value: d.occupancy,
        itemStyle: { color: d.above_avg ? CLR_DARK : CLR_ACCENT },
      })),
      label: { show: true, position: 'right', formatter: (p: any) => p.value.toFixed(1) + '%' },
      markLine: {
        data: [{ xAxis: d.portfolio_avg }],
        lineStyle: { color: '#333', type: 'dashed' },
        label: { formatter: `Avg: ${d.portfolio_avg.toFixed(1)}%` },
        symbol: 'none',
      },
    }],
  }
})

// --- Asset Allocation Donut ---
const allocOption = computed(() => {
  const d = dashboard.assetAlloc
  if (!d.length) return null
  return {
    tooltip: { trigger: 'item', formatter: '{b}: ${c} ({d}%)' },
    legend: { orient: 'vertical', right: 10, top: 'center' },
    series: [{
      type: 'pie',
      radius: ['45%', '70%'],
      center: ['40%', '50%'],
      data: d.map(item => ({ name: item.asset_type, value: Math.round(item.pref_equity) })),
      label: { show: false },
      emphasis: { label: { show: true, fontSize: 14, fontWeight: 'bold' } },
    }],
  }
})

// --- Loan Maturities Chart ---
const loanOption = computed(() => {
  const d = dashboard.loanMaturities
  if (!d || !d.yearly.length) return null
  const years = [...new Set(d.yearly.map(r => r.year))].sort()
  const fixedData = years.map(y => {
    const r = d.yearly.find(r => r.year === y && r.rate_type === 'Fixed')
    return r ? r.amount : 0
  })
  const floatingData = years.map(y => {
    const r = d.yearly.find(r => r.year === y && r.rate_type === 'Floating')
    return r ? r.amount : 0
  })
  const totals = years.map((_, i) => fixedData[i] + floatingData[i])

  return {
    tooltip: { trigger: 'axis', valueFormatter: (v: number) => '$' + (v / 1e6).toFixed(1) + 'M' },
    legend: { bottom: 0 },
    grid: { left: 70, right: 20, top: 30, bottom: 40 },
    xAxis: { type: 'category', data: years, name: 'Maturity Year' },
    yAxis: { type: 'value', name: 'Loan Amount ($)',
      axisLabel: { formatter: (v: number) => '$' + (v / 1e6).toFixed(0) + 'M' } },
    series: [
      { name: 'Fixed', type: 'bar', stack: 'total', data: fixedData,
        itemStyle: { color: CLR_DARK },
        label: {
          show: true, position: 'inside', color: '#fff', fontSize: 10,
          formatter: (p: any) => {
            const rate = d.fixed_rates.find(r => r.year === years[p.dataIndex])
            return rate ? (rate.avg_rate * 100).toFixed(2) + '%' : ''
          },
        } },
      { name: 'Floating', type: 'bar', stack: 'total', data: floatingData,
        itemStyle: { color: CLR_ACCENT },
        label: {
          show: true, position: 'top', color: CLR_DARK, fontSize: 11,
          formatter: (p: any) => {
            const total = totals[p.dataIndex]
            return '$' + (total / 1e6).toFixed(1) + 'M'
          },
        } },
    ],
  }
})

// --- Computed Returns Chart ---
const returnsOption = computed(() => {
  const d = dashboard.returns
  if (!d.length) return null
  const sorted = [...d].filter(r => r.irr != null).sort((a, b) => (a.irr ?? 0) - (b.irr ?? 0))
  return {
    tooltip: { trigger: 'axis' },
    grid: { left: 160, right: 40, top: 10, bottom: 30 },
    xAxis: { type: 'value', name: 'IRR (%)', axisLabel: { formatter: '{value}%' } },
    yAxis: { type: 'category', data: sorted.map(r => r.name) },
    series: [{
      type: 'bar',
      data: sorted.map(r => ({ value: ((r.irr ?? 0) * 100).toFixed(2), itemStyle: { color: CLR_DARK } })),
      label: { show: true, position: 'right', formatter: '{c}%', fontSize: 11 },
    }],
  }
})

const returnsColumns = [
  { key: 'name', label: 'Deal Name' },
  { key: 'contributions', label: 'Contributions', format: 'currency', align: 'right' },
  { key: 'cf_distributions', label: 'CF Distributions', format: 'currency', align: 'right' },
  { key: 'cap_distributions', label: 'Cap Distributions', format: 'currency', align: 'right' },
  { key: 'irr', label: 'IRR', format: 'percent', align: 'right' },
  { key: 'roe', label: 'ROE', format: 'percent', align: 'right' },
  { key: 'moic', label: 'MOIC', format: 'multiple', align: 'right' },
]

function onFreqChange(e: Event) {
  dashboard.loadNoi((e.target as HTMLSelectElement).value)
}
</script>

<template>
  <div class="dashboard">
    <ProgressOverlay :visible="dashboard.loading"
      :message="dashboard.initProgress ? 'Computing portfolio data...' : 'Loading dashboard...'"
      :progress="dashboard.initProgress" />
    <div v-if="dashboard.loadError" class="error-banner">
      {{ dashboard.loadError }}
      <button @click="dashboard.loadAll()">Retry</button>
    </div>

    <!-- KPI Cards -->
    <div class="kpi-grid" v-if="dashboard.kpis">
      <KpiCard label="Portfolio Value" :value="dashboard.kpis.portfolio_value" format="currency" />
      <KpiCard label="Debt Outstanding" :value="dashboard.kpis.debt_outstanding" format="currency" />
      <KpiCard label="Wtd Avg Cap Rate" :value="dashboard.kpis.wtd_avg_cap_rate" format="percent" />
      <KpiCard label="Portfolio Occupancy" :value="dashboard.kpis.portfolio_occupancy / 100" format="percent" />
      <KpiCard label="Deal Count" :value="dashboard.kpis.deal_count" format="integer" />
      <KpiCard label="Total Preferred Equity" :value="dashboard.kpis.total_pref_equity" format="currency" />
    </div>

    <!-- Charts Row 1: NOI Trend + Capital Structure -->
    <div class="chart-row">
      <div class="chart-card chart-wide">
        <div class="chart-header">
          <h3>Portfolio NOI Trend</h3>
          <select @change="onFreqChange" :value="dashboard.noiFreq" class="freq-select">
            <option>Monthly</option>
            <option>Quarterly</option>
            <option>Annually</option>
          </select>
        </div>
        <v-chart v-if="noiOption" :option="noiOption" style="height: 350px" autoresize />
        <p v-else class="placeholder">No NOI data available</p>
      </div>
      <div class="chart-card chart-narrow">
        <h3>Capital Structure</h3>
        <v-chart v-if="capOption" :option="capOption" style="height: 350px" autoresize />
        <p v-else class="placeholder">No data</p>
      </div>
    </div>

    <!-- Charts Row 2: Occupancy by Type + Asset Allocation -->
    <div class="chart-row">
      <div class="chart-card">
        <h3>Occupancy by Type</h3>
        <v-chart v-if="occOption" :option="occOption"
          :style="{ height: Math.max(200, (dashboard.occByType?.data.length ?? 0) * 35) + 'px' }"
          autoresize />
        <p v-else class="placeholder">No occupancy data</p>
      </div>
      <div class="chart-card">
        <h3>Asset Allocation</h3>
        <v-chart v-if="allocOption" :option="allocOption" style="height: 300px" autoresize />
        <p v-else class="placeholder">No allocation data</p>
      </div>
    </div>

    <!-- Loan Maturities -->
    <div class="chart-card">
      <h3>Loan Maturities</h3>
      <v-chart v-if="loanOption" :option="loanOption" style="height: 300px" autoresize />
      <p v-else class="placeholder">No loan data</p>
    </div>

    <!-- Computed Returns -->
    <div class="chart-card" style="margin-top: 16px">
      <div class="chart-header">
        <h3>Portfolio Computed Returns</h3>
        <button
          class="btn-compute"
          @click="dashboard.computeReturns()"
          :disabled="dashboard.computingReturns"
        >
          {{ dashboard.computingReturns ? 'Computing...' : 'Compute All Returns' }}
        </button>
      </div>

      <div v-if="dashboard.returnErrors.length" class="error-list">
        <details>
          <summary>{{ dashboard.returnErrors.length }} deal(s) skipped</summary>
          <p v-for="(err, i) in dashboard.returnErrors" :key="i">{{ err.name }}: {{ err.error }}</p>
        </details>
      </div>

      <v-chart v-if="returnsOption" :option="returnsOption"
        :style="{ height: Math.max(250, dashboard.returns.length * 28) + 'px' }"
        autoresize />

      <DataTable v-if="dashboard.returns.length" :columns="returnsColumns" :rows="dashboard.returns"
        style="margin-top: 16px" />
    </div>
  </div>
</template>

<style scoped>
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 12px;
  margin-bottom: 20px;
}

.chart-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-bottom: 16px;
}

.chart-row .chart-wide { grid-column: 1; }
.chart-row .chart-narrow { grid-column: 2; }

.chart-card {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: 8px;
  padding: 16px;
}

.chart-card h3 {
  font-size: 14px;
  margin-bottom: 12px;
}

.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.chart-header h3 { margin-bottom: 0; }

.freq-select {
  padding: 4px 8px;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  font-size: 13px;
}

.placeholder {
  color: var(--color-text-secondary);
  font-style: italic;
  padding: 40px 0;
  text-align: center;
}
.error-banner {
  background: #fef2f2;
  border: 1px solid #ef4444;
  color: #991b1b;
  padding: 12px 16px;
  border-radius: 6px;
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  gap: 12px;
}
.error-banner button {
  background: #ef4444;
  color: #fff;
  border: none;
  padding: 4px 12px;
  border-radius: 4px;
  cursor: pointer;
}

.btn-compute {
  padding: 6px 16px;
  background: var(--color-accent);
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
}

.btn-compute:hover { background: #3a63ad; }
.btn-compute:disabled { opacity: 0.6; cursor: not-allowed; }

.error-list {
  background: #fff8e1;
  border: 1px solid #ffe082;
  border-radius: 6px;
  padding: 8px 12px;
  margin-bottom: 12px;
  font-size: 13px;
}

.error-list summary { cursor: pointer; color: #856404; }
.error-list p { margin: 4px 0; color: #666; }

@media (max-width: 1200px) {
  .kpi-grid { grid-template-columns: repeat(3, 1fr); }
  .chart-row { grid-template-columns: 1fr; }
}
</style>
