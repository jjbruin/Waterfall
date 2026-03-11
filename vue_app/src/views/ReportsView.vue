<script setup lang="ts">
import { ref, onMounted, computed, watch } from 'vue'
import { useDataStore } from '../stores/data'
import { useDealsStore } from '../stores/deals'
import DataTable from '../components/common/DataTable.vue'
import api from '../api/client'

const data = useDataStore()
const deals = useDealsStore()

const population = ref<'current' | 'select' | 'partner' | 'upstream' | 'all'>('all')
const selectedVcodes = ref<string[]>([])
const results = ref<any[]>([])
const errors = ref<any[]>([])
const loading = ref(false)
const showErrors = ref(false)

// Population selector data
const eligibleDeals = ref<any[]>([])
const partners = ref<any[]>([])
const upstreamInvestors = ref<any[]>([])
const selectedPartner = ref('')
const selectedInvestor = ref('')

onMounted(async () => {
  if (data.deals.length === 0) await data.loadDeals()
  // Load eligible deals for selectors
  try {
    const res = await api.get('/api/reports/deal-lookup')
    eligibleDeals.value = res.data.eligible
  } catch { /* ignore */ }
})

// Load partner/investor data when population changes
watch(population, async (val) => {
  if (val === 'partner' && partners.value.length === 0) {
    try {
      const res = await api.get('/api/reports/partners')
      partners.value = res.data.partners
    } catch { /* ignore */ }
  }
  if (val === 'upstream' && upstreamInvestors.value.length === 0) {
    try {
      const res = await api.get('/api/reports/upstream-investors')
      upstreamInvestors.value = res.data.investors
    } catch { /* ignore */ }
  }
})

const resolvedVcodes = computed(() => {
  switch (population.value) {
    case 'current':
      return deals.currentVcode ? [deals.currentVcode] : []
    case 'select':
      return selectedVcodes.value
    case 'partner': {
      const p = partners.value.find((p) => p.partner === selectedPartner.value)
      return p ? p.vcodes : []
    }
    case 'upstream': {
      const inv = upstreamInvestors.value.find((i) => i.investor_id === selectedInvestor.value)
      return inv ? inv.vcodes : []
    }
    case 'all':
      return eligibleDeals.value.map((d) => d.vcode)
    default:
      return []
  }
})

const populationLabel = computed(() => {
  const count = resolvedVcodes.value.length
  switch (population.value) {
    case 'current':
      return deals.currentVcode ? `Current Deal: ${data.getDealName(deals.currentVcode)}` : 'No deal selected'
    case 'select':
      return `${count} deal(s) selected`
    case 'partner':
      return selectedPartner.value ? `${selectedPartner.value} — ${count} deal(s)` : 'Select a partner'
    case 'upstream':
      return selectedInvestor.value ? `${count} deal(s)` : 'Select an investor'
    case 'all':
      return `${count} deals with waterfalls`
    default:
      return ''
  }
})

const columns = [
  { key: 'Deal Name', label: 'Deal Name' },
  { key: 'Partner', label: 'Partner' },
  { key: 'Contributions', label: 'Contributions', format: 'currency', align: 'right' },
  { key: 'CF Distributions', label: 'CF Distributions', format: 'currency', align: 'right' },
  { key: 'Capital Distributions', label: 'Capital Distributions', format: 'currency', align: 'right' },
  { key: 'IRR', label: 'IRR', format: 'percent', align: 'right' },
  { key: 'ROE', label: 'ROE', format: 'percent', align: 'right' },
  { key: 'MOIC', label: 'MOIC', format: 'multiple', align: 'right' },
]

async function generate() {
  const vcodes = resolvedVcodes.value
  if (vcodes.length === 0) return

  loading.value = true
  errors.value = []
  try {
    const res = await api.post('/api/reports/projected-returns', { vcodes })
    results.value = res.data.rows
    errors.value = res.data.errors || []
  } finally {
    loading.value = false
  }
}

async function downloadExcel() {
  const vcodes = resolvedVcodes.value
  if (vcodes.length === 0) return

  const res = await api.post('/api/reports/projected-returns/excel', { vcodes }, { responseType: 'blob' })
  const url = URL.createObjectURL(new Blob([res.data]))
  const a = document.createElement('a')
  a.href = url
  a.download = 'projected_returns.xlsx'
  a.click()
  URL.revokeObjectURL(url)
}

function toggleDeal(vcode: string) {
  const idx = selectedVcodes.value.indexOf(vcode)
  if (idx >= 0) selectedVcodes.value.splice(idx, 1)
  else selectedVcodes.value.push(vcode)
}
</script>

<template>
  <div class="reports">
    <h2>Reports</h2>

    <div class="controls">
      <!-- Population Selector -->
      <div class="control-group">
        <label>Population:</label>
        <select v-model="population">
          <option value="current">Current Deal</option>
          <option value="select">Select Deals</option>
          <option value="partner">By Partner</option>
          <option value="upstream">By Upstream Investor</option>
          <option value="all">All Deals</option>
        </select>
      </div>

      <!-- Partner selector -->
      <div v-if="population === 'partner'" class="control-group">
        <label>Partner:</label>
        <select v-model="selectedPartner">
          <option value="">-- Select partner --</option>
          <option v-for="p in partners" :key="p.partner" :value="p.partner">
            {{ p.partner }} ({{ p.deal_count }} deals)
          </option>
        </select>
      </div>

      <!-- Upstream investor selector -->
      <div v-if="population === 'upstream'" class="control-group">
        <label>Investor:</label>
        <select v-model="selectedInvestor">
          <option value="">-- Select investor --</option>
          <option v-for="inv in upstreamInvestors" :key="inv.investor_id" :value="inv.investor_id">
            {{ inv.display }} ({{ inv.deal_count }} deals)
          </option>
        </select>
      </div>

      <span class="population-label">{{ populationLabel }}</span>

      <button class="btn-generate" @click="generate" :disabled="loading || resolvedVcodes.length === 0">
        {{ loading ? 'Generating...' : 'Generate Report' }}
      </button>
      <button v-if="results.length" class="btn-download" @click="downloadExcel">
        Download Excel
      </button>
    </div>

    <!-- Multi-select deal picker -->
    <div v-if="population === 'select'" class="deal-picker">
      <div class="deal-picker-header">
        <span>{{ selectedVcodes.length }} of {{ eligibleDeals.length }} deals selected</span>
        <button class="btn-sm" @click="selectedVcodes = eligibleDeals.map(d => d.vcode)">Select All</button>
        <button class="btn-sm" @click="selectedVcodes = []">Clear</button>
      </div>
      <div class="deal-picker-list">
        <label v-for="d in eligibleDeals" :key="d.vcode" class="deal-checkbox">
          <input type="checkbox" :checked="selectedVcodes.includes(d.vcode)" @change="toggleDeal(d.vcode)" />
          {{ d.label }}
        </label>
      </div>
    </div>

    <!-- Errors -->
    <div v-if="errors.length > 0" class="error-section">
      <button class="btn-sm" @click="showErrors = !showErrors">
        {{ errors.length }} deal(s) skipped {{ showErrors ? '▾' : '▸' }}
      </button>
      <div v-if="showErrors" class="error-list">
        <p v-for="(e, i) in errors" :key="i">{{ e.deal_name || e.vcode }}: {{ e.error }}</p>
      </div>
    </div>

    <!-- Results Table -->
    <DataTable v-if="results.length" :columns="columns" :rows="results" :highlight-total="true" />
    <p v-else-if="!loading" class="placeholder">Generate a report to see projected returns.</p>
  </div>
</template>

<style scoped>
.reports { padding: 0 0 40px 0; }
h2 { font-size: 20px; margin-bottom: 16px; }

.controls {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
  flex-wrap: wrap;
}

.control-group {
  display: flex;
  align-items: center;
  gap: 6px;
}

.control-group label {
  font-size: 13px;
  font-weight: 600;
}

.control-group select {
  padding: 7px 10px;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  font-size: 13px;
}

.population-label {
  font-size: 12px;
  color: var(--color-text-secondary);
  font-style: italic;
}

.btn-generate, .btn-download {
  padding: 8px 20px;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
}

.btn-generate { background: var(--color-accent); color: white; }
.btn-generate:hover:not(:disabled) { background: #3a63ad; }
.btn-generate:disabled { opacity: 0.6; cursor: not-allowed; }
.btn-download { background: var(--color-pref); color: white; }

.btn-sm {
  padding: 3px 10px;
  border: 1px solid var(--color-border);
  background: var(--color-surface);
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
}

.btn-sm:hover { background: #eee; }

/* Deal picker */
.deal-picker {
  margin-bottom: 16px;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  padding: 10px;
}

.deal-picker-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
  font-size: 12px;
  color: var(--color-text-secondary);
}

.deal-picker-list {
  max-height: 200px;
  overflow-y: auto;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 4px;
}

.deal-checkbox {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  cursor: pointer;
  padding: 2px 4px;
  border-radius: 3px;
}

.deal-checkbox:hover { background: #f5f5f5; }
.deal-checkbox input { cursor: pointer; }

/* Errors */
.error-section { margin-bottom: 12px; }
.error-list {
  margin-top: 4px;
  padding: 8px 12px;
  background: #fff8e1;
  border: 1px solid #ffe082;
  border-radius: 4px;
  font-size: 12px;
  color: #856404;
}
.error-list p { margin: 2px 0; }

.placeholder {
  color: var(--color-text-secondary);
  font-style: italic;
  text-align: center;
  padding: 40px 0;
}
</style>
