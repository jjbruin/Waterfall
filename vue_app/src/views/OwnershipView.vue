<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import DataTable from '../components/common/DataTable.vue'
import api from '../api/client'
import { useDataStore } from '../stores/data'

const dataStore = useDataStore()

// Tree state
const treeData = ref<any>(null)
const treeLoading = ref(false)

// Entity selector
const selectedEntity = ref('')
const entityTree = ref<any>(null)
const entityLoading = ref(false)

// Upstream analysis
const distributionAmount = ref(100000)
const upstreamResult = ref<any>(null)
const upstreamLoading = ref(false)
const upstreamError = ref('')

// Entities for the selector (from tree nodes)
const entities = computed(() => {
  if (!treeData.value || !treeData.value.nodes) return []
  return treeData.value.nodes
    .map((n: any) => ({ id: n.entity_id, name: n.name || n.entity_id }))
    .sort((a: any, b: any) => a.name.localeCompare(b.name))
})

const requirementsData = ref<any[]>([])

onMounted(async () => {
  treeLoading.value = true
  try {
    const [treeRes, reqRes] = await Promise.all([
      api.get('/api/ownership/tree'),
      api.get('/api/ownership/requirements'),
    ])
    treeData.value = treeRes.data
    requirementsData.value = reqRes.data.requirements || []
  } catch (e: any) {
    dataStore.addToast('Failed to load ownership data: ' + (e.response?.data?.error || e.message), 'error')
  } finally {
    treeLoading.value = false
  }
})

async function loadEntityTree() {
  if (!selectedEntity.value) return
  entityLoading.value = true
  entityTree.value = null
  upstreamResult.value = null
  upstreamError.value = ''
  try {
    const res = await api.get(`/api/ownership/tree/${selectedEntity.value}`)
    entityTree.value = res.data
  } catch (e: any) {
    dataStore.addToast('Failed to load entity tree: ' + (e.response?.data?.error || e.message), 'error')
  } finally {
    entityLoading.value = false
  }
}

async function runUpstreamAnalysis() {
  if (!selectedEntity.value) return
  upstreamLoading.value = true
  upstreamResult.value = null
  upstreamError.value = ''
  try {
    const res = await api.post('/api/ownership/upstream-analysis', {
      entity_id: selectedEntity.value,
      distribution_amount: distributionAmount.value,
    })
    if (res.data.error) {
      upstreamError.value = res.data.error
    } else {
      upstreamResult.value = res.data
    }
  } catch (e: any) {
    upstreamError.value = e.response?.data?.error || 'Upstream analysis failed'
  } finally {
    upstreamLoading.value = false
  }
}

// Table columns
const requirementColumns = [
  { key: 'entity_id', label: 'Entity ID' },
  { key: 'entity_name', label: 'Entity Name' },
  { key: 'num_investors', label: 'Investors', align: 'right' },
  { key: 'deal_vcode', label: 'Deal Vcode' },
]

const dealAllocColumns = [
  { key: 'PropCode', label: 'PropCode' },
  { key: 'vState', label: 'vState' },
  { key: 'Allocated', label: 'Allocated', format: 'currency2', align: 'right' },
]

const upstreamColumns = [
  { key: 'Entity', label: 'Entity' },
  { key: 'PropCode', label: 'PropCode' },
  { key: 'vState', label: 'vState' },
  { key: 'Allocated', label: 'Allocated', format: 'currency2', align: 'right' },
  { key: 'Level', label: 'Level', align: 'right' },
  { key: 'Path', label: 'Path' },
]

const beneficiaryColumns = [
  { key: 'entity_id', label: 'Entity' },
  { key: 'amount', label: 'Amount', format: 'currency2', align: 'right' },
  { key: 'pct_of_total', label: '% of Total', format: 'percent', align: 'right' },
]

const investorColumns = [
  { key: 'investor_id', label: 'Investor ID' },
  { key: 'name', label: 'Name' },
  { key: 'ownership_pct', label: 'Ownership %', format: 'percent', align: 'right' },
]

function fmtCur(v: any): string {
  if (v == null) return '--'
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 0 }).format(v)
}
</script>

<template>
  <div class="ownership">
    <h2>Ownership & Partnerships</h2>

    <!-- Overview -->
    <div class="section">
      <h3>Ownership Tree Overview</h3>
      <div v-if="treeLoading" class="placeholder">Loading ownership data...</div>
      <template v-else-if="treeData">
        <div class="stats">
          <span class="stat">Entities: <strong>{{ treeData.entity_count }}</strong></span>
          <span class="stat">Relationships: <strong>{{ treeData.relationship_count }}</strong></span>
        </div>
      </template>
    </div>

    <!-- Waterfall Requirements -->
    <div v-if="requirementsData.length" class="section">
      <h3>Entities Needing Waterfalls ({{ requirementsData.length }})</h3>
      <DataTable :columns="requirementColumns" :rows="requirementsData" />
    </div>

    <!-- Entity Explorer -->
    <div class="section">
      <h3>Entity Explorer</h3>
      <div class="entity-controls">
        <select v-model="selectedEntity" @change="loadEntityTree" class="entity-select">
          <option value="">-- Select an entity --</option>
          <option v-for="e in entities" :key="e.id" :value="e.id">
            {{ e.name }} ({{ e.id }})
          </option>
        </select>
      </div>

      <div v-if="entityLoading" class="placeholder">Loading entity tree...</div>

      <template v-if="entityTree">
        <!-- Entity Info -->
        <div v-if="entityTree.entity_info" class="entity-info">
          <span class="info-badge">{{ entityTree.entity_info.entity_id }}</span>
          <span v-if="entityTree.entity_info.is_passthrough" class="info-tag passthrough">Passthrough</span>
          <span v-if="entityTree.entity_info.needs_waterfall" class="info-tag needs-wf">Needs Waterfall</span>
          <span class="info-detail">{{ entityTree.entity_info.investor_count }} investor(s)</span>
        </div>

        <!-- ASCII Tree -->
        <div v-if="entityTree.tree_text" class="tree-text">
          <pre>{{ entityTree.tree_text }}</pre>
        </div>

        <!-- Ultimate Investors -->
        <div v-if="entityTree.ultimate_investors && entityTree.ultimate_investors.length">
          <h4>Ultimate Investors</h4>
          <DataTable :columns="investorColumns" :rows="entityTree.ultimate_investors" />
        </div>
      </template>
    </div>

    <!-- Upstream Analysis -->
    <div class="section">
      <h3>Upstream Analysis</h3>
      <p class="section-desc">Run a test distribution through the entity's waterfall and trace cash flows upstream.</p>

      <div class="upstream-controls">
        <select v-model="selectedEntity" class="entity-select">
          <option value="">-- Select an entity --</option>
          <option v-for="e in entities" :key="e.id" :value="e.id">
            {{ e.name }} ({{ e.id }})
          </option>
        </select>

        <div class="amount-input">
          <label>Distribution Amount ($)</label>
          <input type="number" v-model.number="distributionAmount" min="0" step="10000" />
        </div>

        <button
          class="btn-run"
          @click="runUpstreamAnalysis"
          :disabled="!selectedEntity || upstreamLoading"
        >
          {{ upstreamLoading ? 'Running...' : 'Run Upstream Analysis' }}
        </button>
      </div>

      <div v-if="upstreamError" class="error-msg">{{ upstreamError }}</div>

      <template v-if="upstreamResult">
        <div class="result-summary">
          <span>Distribution: <strong>{{ fmtCur(upstreamResult.distribution_amount) }}</strong></span>
          <span>Total Allocated: <strong>{{ fmtCur(upstreamResult.total_allocated) }}</strong></span>
        </div>

        <!-- Deal Allocations -->
        <h4>Deal-Level Allocations</h4>
        <DataTable :columns="dealAllocColumns" :rows="upstreamResult.deal_allocations" />

        <!-- Terminal Beneficiaries -->
        <h4>Terminal Beneficiaries</h4>
        <DataTable :columns="beneficiaryColumns" :rows="upstreamResult.beneficiaries" />

        <!-- Upstream Allocations -->
        <div v-if="upstreamResult.upstream_allocations && upstreamResult.upstream_allocations.length">
          <h4>Upstream Allocation Detail</h4>
          <DataTable :columns="upstreamColumns" :rows="upstreamResult.upstream_allocations" />
        </div>
      </template>
    </div>
  </div>
</template>

<style scoped>
.ownership { padding: 0 0 40px 0; }
h2 { font-size: 20px; margin-bottom: 16px; }
h3 { font-size: 15px; margin: 0 0 12px 0; }
h4 { font-size: 13px; margin: 16px 0 8px 0; font-weight: 600; }

.section {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 16px;
}

.section-desc {
  font-size: 13px;
  color: var(--color-text-secondary);
  margin-bottom: 12px;
}

.placeholder {
  color: var(--color-text-secondary);
  font-style: italic;
  text-align: center;
  padding: 32px 0;
}

.stats { display: flex; gap: 24px; }
.stat { font-size: 14px; color: var(--color-text-secondary); }
.stat strong { color: var(--color-text); }

.entity-controls { margin-bottom: 12px; }

.entity-select {
  padding: 8px 12px;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  font-size: 14px;
  min-width: 350px;
}

.entity-info {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 12px;
  flex-wrap: wrap;
}

.info-badge {
  background: var(--color-accent);
  color: white;
  padding: 2px 10px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 600;
}

.info-tag {
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 500;
}

.info-tag.passthrough { background: #e8f5e9; color: #2e7d32; }
.info-tag.needs-wf { background: #fff3e0; color: #e65100; }

.info-detail {
  font-size: 13px;
  color: var(--color-text-secondary);
}

.tree-text {
  background: #f8f9fa;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  padding: 12px;
  margin-bottom: 12px;
  overflow-x: auto;
}

.tree-text pre {
  margin: 0;
  font-size: 12px;
  line-height: 1.5;
  white-space: pre;
}

.upstream-controls {
  display: flex;
  gap: 12px;
  align-items: flex-end;
  margin-bottom: 16px;
  flex-wrap: wrap;
}

.amount-input {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.amount-input label {
  font-size: 12px;
  color: var(--color-text-secondary);
}

.amount-input input {
  padding: 8px 12px;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  font-size: 14px;
  width: 160px;
}

.btn-run {
  padding: 8px 20px;
  background: var(--color-accent);
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
}

.btn-run:hover { background: #3a63ad; }
.btn-run:disabled { opacity: 0.6; cursor: not-allowed; }

.error-msg {
  color: #d32f2f;
  background: #ffebee;
  padding: 8px 12px;
  border-radius: 6px;
  font-size: 13px;
  margin-bottom: 12px;
}

.result-summary {
  display: flex;
  gap: 24px;
  margin-bottom: 12px;
  font-size: 14px;
}
</style>
