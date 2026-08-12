<script setup lang="ts">
import { ref, computed, watch, onMounted, type Directive } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import api from '../api/client'

// Click-outside directive
const vClickOutside: Directive = {
  mounted(el, binding) {
    el._clickOutside = (e: MouseEvent) => {
      if (!el.contains(e.target as Node)) binding.value()
    }
    document.addEventListener('click', el._clickOutside)
  },
  unmounted(el) {
    document.removeEventListener('click', el._clickOutside)
  },
}

const route = useRoute()
const router = useRouter()

interface TableInfo { name: string; rows: number; description: string }

const tables = ref<TableInfo[]>([])
const selectedTable = ref('')
const columns = ref<string[]>([])
const rows = ref<Record<string, any>[]>([])
const total = ref(0)
const page = ref(1)
const pageSize = ref(100)
const totalPages = ref(1)
const sortCol = ref('')
const sortOrder = ref<'asc' | 'desc'>('asc')
const filters = ref<Record<string, string>>({})
const loading = ref(false)
const error = ref<string | null>(null)
const tableSearch = ref('')

const filteredTables = computed(() => {
  if (!tableSearch.value) return tables.value
  const q = tableSearch.value.toLowerCase()
  return tables.value.filter(t =>
    t.name.toLowerCase().includes(q) || (t.description || '').toLowerCase().includes(q)
  )
})

onMounted(async () => {
  try {
    const res = await api.get('/api/data/tables')
    tables.value = (res.data.tables || []).sort((a: TableInfo, b: TableInfo) =>
      a.name.localeCompare(b.name)
    )
  } catch (e: any) {
    error.value = 'Failed to load tables'
  }
  // Auto-select from query param
  if (route.query.table) {
    selectedTable.value = route.query.table as string
  }
})

watch(selectedTable, () => {
  page.value = 1
  sortCol.value = ''
  sortOrder.value = 'asc'
  filters.value = {}
  hiddenCols.value = new Set()
  if (selectedTable.value) {
    router.replace({ query: { table: selectedTable.value } })
    loadRows()
  }
})

async function loadRows() {
  if (!selectedTable.value) return
  loading.value = true
  error.value = null
  try {
    const params: Record<string, string | number> = {
      page: page.value,
      page_size: pageSize.value,
    }
    if (sortCol.value) {
      params.sort = sortCol.value
      params.order = sortOrder.value
    }
    for (const [col, val] of Object.entries(filters.value)) {
      if (val.trim()) params[`filter__${col}`] = val.trim()
    }
    const res = await api.get(`/api/data/tables/${selectedTable.value}/rows`, { params })
    columns.value = res.data.columns || []
    rows.value = res.data.rows || []
    total.value = res.data.total
    page.value = res.data.page
    totalPages.value = res.data.total_pages
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    loading.value = false
  }
}

function handleSort(col: string) {
  if (sortCol.value === col) {
    sortOrder.value = sortOrder.value === 'asc' ? 'desc' : 'asc'
  } else {
    sortCol.value = col
    sortOrder.value = 'asc'
  }
  page.value = 1
  loadRows()
}

function sortIcon(col: string) {
  if (sortCol.value !== col) return '↕'
  return sortOrder.value === 'asc' ? '↑' : '↓'
}

let filterTimer: ReturnType<typeof setTimeout> | null = null
function handleFilter(col: string, val: string) {
  filters.value = { ...filters.value, [col]: val }
  if (filterTimer) clearTimeout(filterTimer)
  filterTimer = setTimeout(() => {
    page.value = 1
    loadRows()
  }, 400)
}

function clearFilters() {
  filters.value = {}
  page.value = 1
  loadRows()
}

function goPage(p: number) {
  page.value = p
  loadRows()
}

const tableInfo = computed(() =>
  tables.value.find(t => t.name === selectedTable.value)
)

const hasActiveFilters = computed(() =>
  Object.values(filters.value).some(v => v.trim())
)

const pageSizeOptions = [50, 100, 250, 500]

function handlePageSize(size: number) {
  pageSize.value = size
  page.value = 1
  loadRows()
}

// Column visibility
const hiddenCols = ref<Set<string>>(new Set())
const showColPicker = ref(false)

const visibleColumns = computed(() =>
  columns.value.filter(c => !hiddenCols.value.has(c))
)

function toggleColumn(col: string) {
  const next = new Set(hiddenCols.value)
  if (next.has(col)) next.delete(col)
  else next.add(col)
  hiddenCols.value = next
}

function showAllColumns() {
  hiddenCols.value = new Set()
}

</script>

<template>
  <div class="data-explorer">
    <header class="page-header">
      <h1>Data Explorer</h1>
      <p class="subtitle">Browse and search database tables</p>
    </header>

    <div class="explorer-layout">
      <!-- Table selector panel -->
      <aside class="table-list-panel">
        <input
          v-model="tableSearch"
          class="table-search"
          placeholder="Search tables..."
        />
        <div class="table-list">
          <button
            v-for="t in filteredTables"
            :key="t.name"
            class="table-item"
            :class="{ active: selectedTable === t.name }"
            @click="selectedTable = t.name"
            :title="t.description"
          >
            <span class="table-name">{{ t.name }}</span>
            <span class="table-row-count">{{ t.rows.toLocaleString() }}</span>
          </button>
        </div>
      </aside>

      <!-- Data panel -->
      <main class="data-panel">
        <div v-if="!selectedTable" class="empty-state">
          Select a table from the left to view its data.
        </div>

        <template v-else>
          <!-- Toolbar -->
          <div class="toolbar">
            <div class="toolbar-left">
              <span class="table-title">{{ selectedTable }}</span>
              <span class="table-desc" v-if="tableInfo?.description">{{ tableInfo.description }}</span>
            </div>
            <div class="toolbar-right">
              <span class="row-info">
                {{ total.toLocaleString() }} row{{ total !== 1 ? 's' : '' }}
                <template v-if="hasActiveFilters">(filtered)</template>
              </span>
              <div class="col-picker-dropdown" v-click-outside="() => showColPicker = false">
                <button class="btn-clear" @click="showColPicker = !showColPicker">
                  Columns ({{ visibleColumns.length }}/{{ columns.length }})
                </button>
                <div v-if="showColPicker" class="col-picker-panel">
                  <div class="col-picker-header">
                    <span>Show/Hide Columns</span>
                    <button class="col-picker-reset" @click="showAllColumns">Show All</button>
                  </div>
                  <div class="col-picker-list">
                    <label
                      v-for="col in columns"
                      :key="col"
                      class="col-picker-item"
                    >
                      <input
                        type="checkbox"
                        :checked="!hiddenCols.has(col)"
                        @change="toggleColumn(col)"
                      />
                      {{ col }}
                    </label>
                  </div>
                </div>
              </div>
              <button
                v-if="hasActiveFilters"
                class="btn-clear"
                @click="clearFilters"
              >Clear Filters</button>
              <select
                class="page-size-select"
                :value="pageSize"
                @change="handlePageSize(Number(($event.target as HTMLSelectElement).value))"
              >
                <option v-for="s in pageSizeOptions" :key="s" :value="s">{{ s }} / page</option>
              </select>
            </div>
          </div>

          <!-- Loading -->
          <div v-if="loading" class="loading-bar">Loading...</div>
          <div v-if="error" class="error-msg">{{ error }}</div>

          <!-- Table -->
          <div class="table-scroll">
            <table class="explorer-table">
              <thead>
                <tr>
                  <th class="row-num-header">#</th>
                  <th
                    v-for="col in visibleColumns"
                    :key="col"
                    @click="handleSort(col)"
                    class="sortable-header"
                    :class="{ 'sort-active': sortCol === col }"
                  >
                    <span class="header-text">{{ col }}</span>
                    <span class="sort-icon">{{ sortIcon(col) }}</span>
                  </th>
                </tr>
                <!-- Filter row -->
                <tr class="filter-row">
                  <th></th>
                  <th v-for="col in visibleColumns" :key="'f-' + col">
                    <input
                      class="filter-input"
                      :placeholder="'Filter...'"
                      :value="filters[col] || ''"
                      @input="handleFilter(col, ($event.target as HTMLInputElement).value)"
                    />
                  </th>
                </tr>
              </thead>
              <tbody>
                <tr v-if="rows.length === 0 && !loading">
                  <td :colspan="visibleColumns.length + 1" class="no-data">No data found</td>
                </tr>
                <tr v-for="(row, idx) in rows" :key="idx">
                  <td class="row-num">{{ (page - 1) * pageSize + idx + 1 }}</td>
                  <td v-for="col in visibleColumns" :key="col" :title="String(row[col] ?? '')">
                    {{ row[col] ?? '' }}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          <!-- Pagination -->
          <div v-if="totalPages > 1" class="pagination">
            <button :disabled="page <= 1" @click="goPage(1)">«</button>
            <button :disabled="page <= 1" @click="goPage(page - 1)">‹</button>
            <span class="page-info">Page {{ page }} of {{ totalPages }}</span>
            <button :disabled="page >= totalPages" @click="goPage(page + 1)">›</button>
            <button :disabled="page >= totalPages" @click="goPage(totalPages)">»</button>
          </div>
        </template>
      </main>
    </div>
  </div>
</template>

<style scoped>
.data-explorer {
  padding: 24px;
  height: calc(100vh - 48px);
  display: flex;
  flex-direction: column;
}

.page-header {
  margin-bottom: 16px;
}
.page-header h1 {
  font-size: 20px;
  font-weight: 700;
  margin: 0;
}
.subtitle {
  color: #888;
  font-size: 13px;
  margin: 2px 0 0;
}

.explorer-layout {
  display: flex;
  gap: 16px;
  flex: 1;
  min-height: 0;
}

/* Table list panel */
.table-list-panel {
  width: 240px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  background: #f8f9fa;
  border: 1px solid #dee2e6;
  border-radius: 6px;
  overflow: hidden;
}
.table-search {
  padding: 8px 10px;
  border: none;
  border-bottom: 1px solid #dee2e6;
  font-size: 13px;
  outline: none;
  background: white;
}
.table-list {
  overflow-y: auto;
  flex: 1;
}
.table-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
  padding: 6px 10px;
  border: none;
  background: none;
  cursor: pointer;
  font-size: 12px;
  text-align: left;
  border-bottom: 1px solid #eee;
}
.table-item:hover {
  background: #e9ecef;
}
.table-item.active {
  background: #1a5276;
  color: white;
}
.table-item.active .table-row-count {
  color: rgba(255,255,255,0.8);
}
.table-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  flex: 1;
}
.table-row-count {
  font-size: 11px;
  color: #888;
  margin-left: 6px;
  flex-shrink: 0;
}

/* Data panel */
.data-panel {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
}

.empty-state {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 200px;
  color: #888;
  font-size: 14px;
}

/* Toolbar */
.toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  flex-wrap: wrap;
  gap: 8px;
}
.toolbar-left {
  display: flex;
  align-items: baseline;
  gap: 10px;
}
.table-title {
  font-size: 16px;
  font-weight: 700;
}
.table-desc {
  font-size: 12px;
  color: #888;
}
.toolbar-right {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 13px;
}
.row-info {
  color: #666;
}
.btn-clear {
  padding: 3px 10px;
  border: 1px solid #ccc;
  border-radius: 4px;
  background: white;
  cursor: pointer;
  font-size: 12px;
}
.btn-clear:hover {
  background: #f0f0f0;
}
.page-size-select {
  padding: 3px 6px;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 12px;
}

.loading-bar {
  padding: 8px;
  text-align: center;
  color: #1a5276;
  font-size: 13px;
}
.error-msg {
  padding: 8px;
  color: #c62828;
  font-size: 13px;
}

/* Table */
.table-scroll {
  overflow: auto;
  flex: 1;
  border: 1px solid #dee2e6;
  border-radius: 4px;
}
.explorer-table {
  width: max-content;
  min-width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}
.explorer-table thead {
  position: sticky;
  top: 0;
  z-index: 2;
}
.sortable-header {
  padding: 6px 10px;
  background: #1a5276;
  color: white;
  font-weight: 600;
  white-space: nowrap;
  cursor: pointer;
  user-select: none;
  border-right: 1px solid rgba(255,255,255,0.15);
}
.sortable-header:hover {
  background: #1e6a96;
}
.sortable-header.sort-active {
  background: #0d3b54;
}
.header-text {
  margin-right: 4px;
}
.sort-icon {
  font-size: 10px;
  opacity: 0.6;
}
.sort-active .sort-icon {
  opacity: 1;
}
.row-num-header {
  padding: 6px 8px;
  background: #1a5276;
  color: rgba(255,255,255,0.6);
  font-weight: 600;
  font-size: 10px;
  width: 40px;
  text-align: center;
}

.filter-row th {
  padding: 3px 2px;
  background: #f0f4f7;
  border-bottom: 2px solid #1a5276;
}
.filter-input {
  width: 100%;
  padding: 3px 6px;
  border: 1px solid #ccc;
  border-radius: 3px;
  font-size: 11px;
  box-sizing: border-box;
}
.filter-input:focus {
  border-color: #1a5276;
  outline: none;
}

.explorer-table td {
  padding: 4px 10px;
  border-bottom: 1px solid #eee;
  white-space: nowrap;
  max-width: 300px;
  overflow: hidden;
  text-overflow: ellipsis;
}
.explorer-table tbody tr:hover {
  background: #f5f8fa;
}
.row-num {
  text-align: center;
  color: #aaa;
  font-size: 11px;
  border-right: 1px solid #eee;
}
.no-data {
  text-align: center;
  padding: 24px;
  color: #888;
}

/* Pagination */
.pagination {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 10px 0 4px;
}
.pagination button {
  padding: 4px 10px;
  border: 1px solid #ccc;
  border-radius: 4px;
  background: white;
  cursor: pointer;
  font-size: 13px;
}
.pagination button:hover:not(:disabled) {
  background: #f0f0f0;
}
.pagination button:disabled {
  opacity: 0.4;
  cursor: default;
}
.page-info {
  font-size: 13px;
  color: #555;
  margin: 0 8px;
}

/* Column picker */
.col-picker-dropdown {
  position: relative;
}
.col-picker-panel {
  position: absolute;
  top: 100%;
  right: 0;
  z-index: 10;
  background: white;
  border: 1px solid #ccc;
  border-radius: 6px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
  width: 220px;
  margin-top: 4px;
}
.col-picker-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 10px;
  border-bottom: 1px solid #eee;
  font-size: 12px;
  font-weight: 600;
}
.col-picker-reset {
  border: none;
  background: none;
  color: #1a5276;
  cursor: pointer;
  font-size: 11px;
  padding: 2px 6px;
}
.col-picker-reset:hover {
  text-decoration: underline;
}
.col-picker-list {
  max-height: 300px;
  overflow-y: auto;
  padding: 4px 0;
}
.col-picker-item {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 3px 10px;
  font-size: 12px;
  cursor: pointer;
  white-space: nowrap;
}
.col-picker-item:hover {
  background: #f5f8fa;
}
.col-picker-item input[type="checkbox"] {
  margin: 0;
}
</style>
