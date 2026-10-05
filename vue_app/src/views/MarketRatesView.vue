<script setup lang="ts">
/**
 * Market Rates — Bank of Canada and the NY Fed, stored once for every screen.
 *
 * Jim, Oct 5 2026: "build the rates table with Bank of Canada and NY Fed."
 * Every value names its publisher and the date it was published for; nothing
 * here is typed or interpolated. Refresh pulls from the publishers and also
 * runs at the end of "Refresh All Data from MRI".
 */
import { ref, computed, onMounted } from 'vue'
import api from '@/api/client'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, DataZoomComponent } from 'echarts/components'

use([CanvasRenderer, LineChart, GridComponent, TooltipComponent, DataZoomComponent])

const series = ref<any[]>([])
const selected = ref('USDCAD')
const history = ref<any[]>([])
const loading = ref(false)
const refreshing = ref(false)
const refreshResult = ref<Record<string, any> | null>(null)
const error = ref<string | null>(null)

const spec = computed(() => series.value.find(s => s.key === selected.value))

function fmt(v: number | null | undefined, unit: string) {
  if (v === null || v === undefined) return '—'
  if (unit === 'percent') return v.toFixed(2) + '%'
  if (unit === 'index') return v.toFixed(8)
  return v.toFixed(4)
}

async function loadSeries() {
  const { data } = await api.get('/api/market-rates/series')
  series.value = data.series || []
}

async function loadHistory() {
  loading.value = true
  error.value = null
  try {
    const { data } = await api.get('/api/market-rates/observations', { params: { series: selected.value } })
    history.value = data.observations || []
  } catch (e: any) {
    error.value = e?.response?.data?.error || 'Could not load the history'
  } finally {
    loading.value = false
  }
}

function pick(key: string) {
  selected.value = key
  loadHistory()
}

async function refresh() {
  refreshing.value = true
  error.value = null
  refreshResult.value = null
  try {
    const { data } = await api.post('/api/market-rates/refresh')
    refreshResult.value = data.results || {}
    await loadSeries()
    await loadHistory()
  } catch (e: any) {
    error.value = e?.response?.data?.error || 'Refresh failed'
  } finally {
    refreshing.value = false
  }
}

// A publisher that could not be reached is named, not folded into "done".
const refreshErrors = computed<[string, any][]>(() =>
  Object.entries(refreshResult.value || {}).filter(([, r]: any) => r.status !== 'ok'))
const refreshRows = computed(() =>
  Object.values(refreshResult.value || {}).reduce((n: number, r: any) => n + (r.rows || 0), 0))

const chartOption = computed(() => ({
  grid: { left: 60, right: 20, top: 20, bottom: 60 },
  tooltip: { trigger: 'axis', valueFormatter: (v: number) => fmt(v, spec.value?.unit || '') },
  xAxis: { type: 'category', data: history.value.map(o => o.date) },
  yAxis: { type: 'value', scale: true },
  dataZoom: [{ type: 'inside' }, { type: 'slider', height: 18, bottom: 10 }],
  series: [{ type: 'line', showSymbol: false, data: history.value.map(o => o.value),
             lineStyle: { width: 1.5 } }],
}))

onMounted(async () => {
  try {
    await loadSeries()
    await loadHistory()
  } catch (e: any) {
    error.value = e?.response?.data?.error || 'Could not load market rates'
  }
})
</script>

<template>
  <div class="rates">
    <div class="header-row">
      <div>
        <h2>Market Rates</h2>
        <div class="subtitle">
          Published rates from the Bank of Canada and the Federal Reserve Bank of New York.
          A date with no publication (a weekend or holiday) uses the prior business day, and says so.
        </div>
      </div>
      <button class="btn" :disabled="refreshing" @click="refresh">
        {{ refreshing ? 'Refreshing…' : 'Refresh from publishers' }}
      </button>
    </div>

    <div v-if="error" class="error-banner">{{ error }}</div>
    <div v-if="refreshResult" class="note" :class="{ warn: refreshErrors.length }">
      Refreshed {{ refreshRows.toLocaleString() }} observations.
      <template v-if="refreshErrors.length">
        Could not reach:
        <span v-for="[k, r] in refreshErrors" :key="k">{{ k }} ({{ r.error }}) </span>
      </template>
    </div>

    <table class="grid">
      <thead>
        <tr><th>Series</th><th>Source</th><th>Unit</th><th class="num">Latest</th><th>As of</th>
            <th>History</th><th>Fetched</th></tr>
      </thead>
      <tbody>
        <tr v-for="s in series" :key="s.key" :class="{ sel: s.key === selected }" @click="pick(s.key)">
          <td><strong>{{ s.key }}</strong><div class="lbl">{{ s.label }}</div></td>
          <td>{{ s.source }}</td>
          <td>{{ s.unit }}</td>
          <td class="num">{{ fmt(s.latest_value, s.unit) }}</td>
          <td>{{ s.last || '—' }}</td>
          <td>{{ s.count ? `${s.count.toLocaleString()} days from ${s.first}` : 'not loaded' }}</td>
          <td class="muted">{{ s.fetched_at ? s.fetched_at.replace('T', ' ').slice(0, 16) + ' UTC' : '—' }}</td>
        </tr>
      </tbody>
    </table>

    <div v-if="spec" class="chart-card">
      <div class="chart-title">{{ spec.label }} <span class="muted">({{ spec.unit }})</span></div>
      <div v-if="loading" class="muted">Loading…</div>
      <div v-else-if="!history.length" class="muted">No observations stored yet — refresh to load them.</div>
      <v-chart v-else :option="chartOption" autoresize class="chart" />
    </div>
  </div>
</template>

<style scoped>
.rates { padding: 20px; }
.header-row { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; }
.header-row h2 { margin: 0; }
.subtitle { font-size: 12.5px; color: var(--color-text-secondary); margin-top: 3px; max-width: 760px; }
.btn {
  padding: 6px 14px; border: 1px solid var(--color-primary, #2f6f4f); border-radius: 6px;
  background: var(--color-primary, #2f6f4f); color: #fff; cursor: pointer; font-size: 13px;
}
.btn:disabled { opacity: .6; cursor: default; }
.error-banner { margin: 12px 0; padding: 8px 12px; border-radius: 6px; background: #fdeaea; color: #8a1f1f; font-size: 13px; }
.note { margin: 12px 0; padding: 8px 12px; border-radius: 6px; background: #eaf5ee; font-size: 13px; }
.note.warn { background: #fff4e0; }
.grid { width: 100%; border-collapse: collapse; margin-top: 14px; font-size: 12.5px; }
.grid th, .grid td { padding: 6px 8px; border-bottom: 1px solid var(--color-border); text-align: left; vertical-align: top; }
.grid th { font-size: 11px; text-transform: uppercase; letter-spacing: .03em; color: var(--color-text-secondary); }
.grid tbody tr { cursor: pointer; }
.grid tbody tr:hover { background: var(--color-surface-hover, rgba(0,0,0,.03)); }
.grid tbody tr.sel { background: rgba(47, 111, 79, .08); }
.num { text-align: right !important; font-variant-numeric: tabular-nums; }
.lbl { font-size: 11.5px; color: var(--color-text-secondary); }
.muted { color: var(--color-text-secondary); font-size: 12px; }
.chart-card { margin-top: 18px; border: 1px solid var(--color-border); border-radius: 8px; padding: 12px; }
.chart-title { font-weight: 600; margin-bottom: 6px; }
.chart { height: 320px; }
</style>
