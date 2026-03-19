<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import api from '../api/client'

const router = useRouter()

interface TrackingItem {
  vcode: string
  deal_name: string
  quarter: string
  status: string
  current_step: number
  step_label: string
  updated_at: string | null
}

const items = ref<TrackingItem[]>([])
const loading = ref(false)
const error = ref<string | null>(null)

// Filters
const quarterFilter = ref('')
const statusFilter = ref('')

// Summary counts
const draftCount = computed(() => items.value.filter(i => i.status === 'draft').length)
const inReviewCount = computed(() => items.value.filter(i =>
  i.status.startsWith('pending_')
).length)
const returnedCount = computed(() => items.value.filter(i => i.status === 'returned').length)
const approvedCount = computed(() => items.value.filter(i => i.status === 'approved').length)

onMounted(() => loadTracking())

async function loadTracking() {
  loading.value = true
  error.value = null
  try {
    const params: Record<string, string> = {}
    if (quarterFilter.value) params.quarter = quarterFilter.value
    if (statusFilter.value) params.status = statusFilter.value
    const res = await api.get('/api/reviews/tracking', { params })
    items.value = res.data.items || []
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    loading.value = false
  }
}

function navigateToOnePager(item: TrackingItem) {
  const query: Record<string, string> = { vcode: item.vcode }
  if (item.quarter) query.quarter = item.quarter
  router.push({ path: '/one-pager', query })
}

function statusClass(status: string): string {
  if (status === 'approved') return 'status-approved'
  if (status === 'returned') return 'status-returned'
  if (status === 'draft') return 'status-draft'
  return 'status-pending'
}

function statusLabel(status: string): string {
  if (status === 'draft') return 'Draft'
  if (status === 'returned') return 'Returned'
  if (status === 'approved') return 'Approved'
  return 'In Review'
}

function formatDate(dt: string | null): string {
  if (!dt) return '—'
  const d = new Date(dt)
  if (isNaN(d.getTime())) return dt
  return `${d.getMonth() + 1}/${d.getDate()}`
}

function filterByStatus(status: string) {
  statusFilter.value = statusFilter.value === status ? '' : status
  loadTracking()
}
</script>

<template>
  <div class="review-tracking">
    <h2>Review Tracking — One Pager Production Pipeline</h2>

    <!-- Filters -->
    <div class="filters">
      <div class="filter-group">
        <label>Quarter:</label>
        <input
          type="text"
          v-model="quarterFilter"
          placeholder="e.g. 2025-Q4"
          class="filter-input"
          @keyup.enter="loadTracking"
        />
      </div>
      <div class="filter-group">
        <label>Status:</label>
        <select v-model="statusFilter" @change="loadTracking" class="filter-select">
          <option value="">All</option>
          <option value="draft">Draft</option>
          <option value="pending_head_am">Pending Head AM</option>
          <option value="pending_president">Pending President</option>
          <option value="pending_cco">Pending CCO</option>
          <option value="pending_ceo">Pending CEO</option>
          <option value="returned">Returned</option>
          <option value="approved">Approved</option>
        </select>
      </div>
      <button class="btn-refresh" @click="loadTracking" :disabled="loading">
        {{ loading ? 'Loading...' : 'Refresh' }}
      </button>
    </div>

    <!-- Summary cards -->
    <div class="summary-cards">
      <div class="summary-card" :class="{ active: statusFilter === 'draft' }" @click="filterByStatus('draft')">
        <span class="card-count">{{ draftCount }}</span>
        <span class="card-label">Draft</span>
      </div>
      <div class="summary-card card-pending" :class="{ active: statusFilter.startsWith('pending') }" @click="filterByStatus('')">
        <span class="card-count">{{ inReviewCount }}</span>
        <span class="card-label">In Review</span>
      </div>
      <div class="summary-card card-returned" :class="{ active: statusFilter === 'returned' }" @click="filterByStatus('returned')">
        <span class="card-count">{{ returnedCount }}</span>
        <span class="card-label">Returned</span>
      </div>
      <div class="summary-card card-approved" :class="{ active: statusFilter === 'approved' }" @click="filterByStatus('approved')">
        <span class="card-count">{{ approvedCount }}</span>
        <span class="card-label">Approved</span>
      </div>
    </div>

    <!-- Error -->
    <div v-if="error" class="error-banner">
      {{ error }}
      <button @click="error = null">Dismiss</button>
    </div>

    <!-- Table -->
    <div class="tracking-table-wrap">
      <table class="tracking-table">
        <thead>
          <tr>
            <th>Deal</th>
            <th>Quarter</th>
            <th>Status</th>
            <th>Step</th>
            <th>Updated</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="item in items"
            :key="item.vcode + item.quarter"
            class="clickable-row"
            @click="navigateToOnePager(item)"
          >
            <td class="deal-name">{{ item.deal_name || item.vcode }}</td>
            <td>{{ item.quarter || '—' }}</td>
            <td>
              <span class="status-badge" :class="statusClass(item.status)">
                {{ statusLabel(item.status) }}
              </span>
            </td>
            <td>{{ item.step_label }}</td>
            <td>{{ formatDate(item.updated_at) }}</td>
          </tr>
          <tr v-if="!items.length && !loading">
            <td colspan="5" class="empty-row">No deals found.</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div v-if="loading" class="loading-text">Loading tracking data...</div>
  </div>
</template>

<style scoped>
.review-tracking {
  padding: 0 0 40px 0;
}

h2 {
  font-size: 18px;
  margin-bottom: 16px;
}

/* Filters */
.filters {
  display: flex;
  gap: 16px;
  align-items: center;
  margin-bottom: 16px;
  flex-wrap: wrap;
}

.filter-group {
  display: flex;
  align-items: center;
  gap: 6px;
}

.filter-group label {
  font-size: 13px;
  font-weight: 500;
}

.filter-input {
  padding: 6px 10px;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  font-size: 13px;
  width: 120px;
}

.filter-select {
  padding: 6px 10px;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  font-size: 13px;
}

.btn-refresh {
  padding: 6px 16px;
  background: var(--color-accent);
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
}
.btn-refresh:hover { opacity: 0.9; }
.btn-refresh:disabled { opacity: 0.5; cursor: not-allowed; }

/* Summary cards */
.summary-cards {
  display: flex;
  gap: 12px;
  margin-bottom: 16px;
}

.summary-card {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: 8px;
  padding: 10px 20px;
  text-align: center;
  cursor: pointer;
  transition: all 0.15s;
  min-width: 90px;
}
.summary-card:hover { border-color: var(--color-accent); }
.summary-card.active { border-color: var(--color-accent); background: #e3f2fd; }

.card-count { font-size: 22px; font-weight: 700; display: block; }
.card-label { font-size: 11px; color: var(--color-text-secondary); text-transform: uppercase; }

.summary-card .card-count { color: #666; }
.card-pending .card-count { color: #1565c0; }
.card-returned .card-count { color: #e65100; }
.card-approved .card-count { color: #2e7d32; }

/* Table */
.tracking-table-wrap {
  border: 1px solid var(--color-border);
  border-radius: 8px;
  overflow: hidden;
}

.tracking-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

.tracking-table th {
  padding: 10px 14px;
  background: var(--color-accent);
  color: white;
  font-weight: 600;
  text-align: left;
}

.tracking-table td {
  padding: 8px 14px;
  border-bottom: 1px solid var(--color-border);
}

.clickable-row { cursor: pointer; }
.clickable-row:hover { background: #f5f5f5; }

.deal-name { font-weight: 500; }

.status-badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 12px;
  font-size: 11px;
  font-weight: 600;
}

.status-draft { background: #eeeeee; color: #666; }
.status-pending { background: #e3f2fd; color: #1565c0; }
.status-returned { background: #fff3e0; color: #e65100; }
.status-approved { background: #e8f5e9; color: #2e7d32; }

.empty-row {
  text-align: center;
  color: var(--color-text-secondary);
  font-style: italic;
  padding: 24px;
}

.error-banner {
  background: #fef2f2;
  border: 1px solid #fca5a5;
  color: #991b1b;
  padding: 8px 14px;
  border-radius: 6px;
  margin-bottom: 12px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 13px;
}
.error-banner button {
  background: none;
  border: 1px solid #fca5a5;
  color: #991b1b;
  padding: 3px 10px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
}

.loading-text {
  text-align: center;
  padding: 20px;
  color: var(--color-text-secondary);
  font-style: italic;
}
</style>
