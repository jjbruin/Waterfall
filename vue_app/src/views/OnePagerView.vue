<script setup lang="ts">
import { onMounted, ref, watch } from 'vue'
import { useDataStore } from '../stores/data'
import { useDealsStore } from '../stores/deals'
import KpiCard from '../components/common/KpiCard.vue'
import api from '../api/client'

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
  onePagerData.value = null
  onePagerQuarter.value = ''
  await loadOnePager(vcode)
}

watch(() => deals.currentVcode, () => {
  onePagerData.value = null
  onePagerQuarter.value = ''
})

// ============================================================
// One Pager
// ============================================================
const onePagerData = ref<Record<string, any> | null>(null)
const onePagerQuarter = ref('')
const onePagerLoading = ref(false)

async function loadOnePager(vcode: string) {
  onePagerLoading.value = true
  try {
    const params: any = {}
    if (onePagerQuarter.value) params.quarter = onePagerQuarter.value
    const res = await api.get(`/api/financials/${vcode}/one-pager`, { params })
    onePagerData.value = res.data
    if (!onePagerQuarter.value && res.data.available_quarters?.length) {
      onePagerQuarter.value = res.data.available_quarters[res.data.available_quarters.length - 1]
    }
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    onePagerLoading.value = false
  }
}

async function refreshOnePager() {
  if (deals.currentVcode) await loadOnePager(deals.currentVcode)
}

// ============================================================
// Formatting helpers
// ============================================================
function fmtCurrency(val: number | null | undefined): string {
  if (val == null || isNaN(val)) return '—'
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 0, maximumFractionDigits: 0 }).format(val)
}
</script>

<template>
  <div class="one-pager">
    <!-- Error banner -->
    <div v-if="error" class="error-banner">
      {{ error }}
      <button @click="error = null">Dismiss</button>
    </div>

    <!-- Deal selector -->
    <div class="deal-selector">
      <label>Select Deal:</label>
      <select :value="deals.currentVcode" @change="onDealSelect">
        <option value="">-- Choose a deal --</option>
        <option v-for="d in data.deals" :key="d.vcode" :value="d.vcode">
          {{ d.Investment_Name || d.vcode }}
        </option>
      </select>
    </div>

    <template v-if="deals.currentVcode">
      <div class="section">
        <h3>One Pager — Investor Report</h3>

        <div class="op-controls">
          <div class="col-control">
            <label>Quarter:</label>
            <select v-model="onePagerQuarter" @change="refreshOnePager">
              <option v-for="q in (onePagerData?.available_quarters || [])" :key="q" :value="q">{{ q }}</option>
            </select>
          </div>
          <button v-if="!onePagerData && !onePagerLoading" class="btn btn-sm" @click="refreshOnePager">Load</button>
        </div>

        <div v-if="onePagerLoading" class="loading">Loading one pager...</div>
        <template v-else-if="onePagerData">
          <!-- General Information -->
          <div v-if="onePagerData.general" class="op-section">
            <h4>General Information</h4>
            <div class="detail-grid">
              <template v-for="(val, key) in onePagerData.general" :key="key">
                <div class="detail-label">{{ key }}</div>
                <div class="detail-value">{{ val ?? '—' }}</div>
              </template>
            </div>
          </div>

          <!-- Capitalization Stack -->
          <div v-if="onePagerData.cap_stack" class="op-section">
            <h4>Capitalization Stack</h4>
            <div class="detail-grid">
              <template v-for="(val, key) in onePagerData.cap_stack" :key="key">
                <div class="detail-label">{{ key }}</div>
                <div class="detail-value">{{ typeof val === 'number' ? fmtCurrency(val) : (val ?? '—') }}</div>
              </template>
            </div>
          </div>

          <!-- Property Performance -->
          <div v-if="onePagerData.property_performance" class="op-section">
            <h4>Property Performance</h4>
            <div class="detail-grid">
              <template v-for="(val, key) in onePagerData.property_performance" :key="key">
                <div class="detail-label">{{ key }}</div>
                <div class="detail-value">{{ typeof val === 'number' ? fmtCurrency(val) : (val ?? '—') }}</div>
              </template>
            </div>
          </div>

          <!-- PE Performance -->
          <div v-if="onePagerData.pe_performance" class="op-section">
            <h4>PE Performance</h4>
            <div class="detail-grid">
              <template v-for="(val, key) in onePagerData.pe_performance" :key="key">
                <div class="detail-label">{{ key }}</div>
                <div class="detail-value">{{ typeof val === 'number' ? fmtCurrency(val) : (val ?? '—') }}</div>
              </template>
            </div>
          </div>

          <!-- Comments -->
          <div v-if="onePagerData.comments" class="op-section">
            <h4>Comments</h4>
            <div v-for="(val, key) in onePagerData.comments" :key="key" class="comment-block">
              <div class="comment-label">{{ key }}</div>
              <div class="comment-text">{{ val || '(none)' }}</div>
            </div>
          </div>
        </template>
        <p v-else class="empty">Select a deal and click Load to view the investor report.</p>
      </div>
    </template>

    <p v-else class="empty">Select a deal to view the one pager.</p>
  </div>
</template>

<style scoped>
.one-pager { max-width: 1200px; }

/* Error banner */
.error-banner { background: #fef2f2; border: 1px solid #fca5a5; color: #991b1b; padding: 10px 16px; border-radius: 8px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; }
.error-banner button { background: none; border: 1px solid #fca5a5; color: #991b1b; padding: 4px 12px; border-radius: 4px; cursor: pointer; }

/* Deal selector */
.deal-selector { display: flex; align-items: center; gap: 12px; margin-bottom: 20px; }
.deal-selector select { padding: 8px 12px; border: 1px solid var(--color-border); border-radius: 6px; font-size: 14px; min-width: 350px; }

/* Section */
.section { background: var(--color-surface); border: 1px solid var(--color-border); border-radius: 8px; padding: 16px; margin-bottom: 16px; }
.section h3 { font-size: 14px; margin-bottom: 12px; }

/* Controls */
.op-controls { display: flex; gap: 16px; align-items: flex-end; flex-wrap: wrap; margin-bottom: 16px; }
.col-control { display: flex; flex-direction: column; gap: 4px; }
.col-control label { font-size: 11px; color: var(--color-text-secondary); text-transform: uppercase; letter-spacing: 0.3px; }
.col-control select { padding: 6px 10px; border: 1px solid var(--color-border); border-radius: 4px; font-size: 13px; }
.btn { padding: 6px 16px; border: none; border-radius: 4px; background: var(--color-accent); color: white; cursor: pointer; font-size: 13px; }
.btn:hover { opacity: 0.9; }
.btn-sm { padding: 6px 12px; font-size: 12px; }

/* One Pager sections */
.op-section { margin-bottom: 20px; padding-bottom: 16px; border-bottom: 1px solid var(--color-border); }
.op-section:last-child { border-bottom: none; }
.op-section h4 { font-size: 13px; font-weight: 600; margin-bottom: 12px; }
.detail-grid { display: grid; grid-template-columns: 200px 1fr; gap: 4px 16px; font-size: 13px; }
.detail-label { color: var(--color-text-secondary); font-weight: 500; }
.detail-value { color: var(--color-text); }
.comment-block { margin-bottom: 12px; }
.comment-label { font-size: 12px; color: var(--color-text-secondary); font-weight: 500; margin-bottom: 4px; }
.comment-text { font-size: 13px; white-space: pre-wrap; }

/* Loading / Empty */
.loading { text-align: center; padding: 24px; color: var(--color-text-secondary); font-style: italic; }
.empty { text-align: center; padding: 24px; color: var(--color-text-secondary); font-style: italic; }
</style>
