<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useAuthStore } from '../../stores/auth'
import { useDataStore } from '../../stores/data'

const route = useRoute()
const router = useRouter()
const auth = useAuthStore()
const data = useDataStore()

function handleLogout() {
  auth.logout()
  router.push('/login')
}

const navItems = [
  { path: '/dashboard', label: 'Dashboard', icon: 'grid' },
  { path: '/deal-analysis', label: 'Deal Analysis', icon: 'trending-up' },
  { path: '/property-financials', label: 'Property Financials', icon: 'building' },
  { path: '/one-pager', label: 'One Pager', icon: 'file' },
  { path: '/review-tracking', label: 'Review Tracking', icon: 'check-square' },
  { path: '/ownership', label: 'Ownership', icon: 'git-branch' },
  { path: '/waterfall-setup', label: 'Waterfall Setup', icon: 'layers' },
  { path: '/reports', label: 'Reports', icon: 'file-text' },
  { path: '/sold-portfolio', label: 'Sold Portfolio', icon: 'archive' },
  { path: '/psckoc', label: 'PSCKOC', icon: 'share-2' },
  { path: '/settings', label: 'Settings', icon: 'settings' },
]

// MRI Data tools
const showMriTools = ref(false)
const mriQueries = ref<Array<{
  name: string; server: string | null; description: string;
  importable: boolean; sql_exists: boolean; target_table: string | null
}>>([])
const mriServers = ref<Record<string, { status: string; latency_ms?: number; error?: string }>>({})
const mriLoading = ref(false)
const mriRunning = ref<string | null>(null)
const mriRefreshing = ref(false)
const mriRefreshResults = ref<Record<string, any> | null>(null)
const mriRunResult = ref<{ query: string; rows: number; csv_path?: string; elapsed_seconds: number } | null>(null)

// Database tools
const showDbTools = ref(false)
const uploadFiles = ref<File[]>([])
const uploadMatches = ref<Array<{ file: File; table_name: string | null; csv_file: string; description: string; protected: boolean }>>([])
const uploading = ref(false)

// Config
const showConfig = ref(false)
const localStartYear = ref(new Date().getFullYear())
const localHorizon = ref(10)
const localProYrBase = ref(new Date().getFullYear() - 1)
const localActualsThrough = ref<string | null>(null)
const useActuals = ref(false)

// Build month-end options for current year
const monthEndOptions = (() => {
  const today = new Date()
  const year = today.getFullYear()
  const options: { value: string; label: string }[] = []
  for (let m = 0; m < today.getMonth(); m++) {
    const lastDay = new Date(year, m + 1, 0)
    const iso = lastDay.toISOString().split('T')[0]
    const label = lastDay.toLocaleDateString('en-US', { month: 'long', year: 'numeric' })
    options.push({ value: iso, label })
  }
  return options
})()

onMounted(async () => {
  if (!auth.isAuthenticated) return
  await data.loadConfig()
  if (data.config) {
    localStartYear.value = data.config.start_year
    localHorizon.value = data.config.horizon_years
    localProYrBase.value = data.config.pro_yr_base
    localActualsThrough.value = data.config.actuals_through
    useActuals.value = !!data.config.actuals_through
  }
})

async function handleReload() {
  await data.reloadData()
}

async function handleFileSelect(event: Event) {
  const input = event.target as HTMLInputElement
  if (!input.files || input.files.length === 0) return
  uploadFiles.value = Array.from(input.files)

  // Load table defs if not yet loaded, then match files
  await data.loadTableDefs()
  const csvToTable: Record<string, { table_name: string; csv_file: string; description: string; protected: boolean }> = {}
  for (const td of data.tableDefs) {
    csvToTable[td.csv_file.toLowerCase()] = td
  }

  uploadMatches.value = uploadFiles.value.map(f => {
    const match = csvToTable[f.name.toLowerCase()]
    return {
      file: f,
      table_name: match?.table_name || null,
      csv_file: match?.csv_file || f.name,
      description: match?.description || '',
      protected: match?.protected || false,
    }
  })
}

const uploadProgress = ref('')

async function handleUploadImport() {
  const importable = uploadMatches.value.filter(m => m.table_name && !m.protected)
  if (importable.length === 0) return
  uploading.value = true
  data.importResult = null
  const allResults: Record<string, any> = {}
  const client = (await import('../../api/client')).default

  // Upload one file at a time to avoid OOM on the server
  for (let i = 0; i < importable.length; i++) {
    const m = importable[i]
    uploadProgress.value = `${i + 1}/${importable.length}: ${m.table_name}`
    try {
      const formData = new FormData()
      formData.append('files', m.file, m.file.name)
      const res = await client.post(
        '/api/data/upload-import', formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      )
      Object.assign(allResults, res.data.results)
    } catch (e: any) {
      allResults[m.table_name || m.file.name] = {
        status: 'error',
        error: e.response?.data?.error || e.message,
      }
    }
  }

  data.importResult = allResults
  const ok = Object.values(allResults).filter((v: any) => v.status === 'success').length
  const err = Object.values(allResults).filter((v: any) => v.status === 'error').length
  data.addToast(
    `Upload complete: ${ok} imported${err > 0 ? ', ' + err + ' errors' : ''}`,
    err > 0 ? 'error' : 'success'
  )
  uploadFiles.value = []
  uploadMatches.value = []
  uploadProgress.value = ''
  uploading.value = false
  await data.loadDeals()
}

async function handleExport() {
  await data.exportDatabase()
}

async function handleConfigSave() {
  await data.updateConfig({
    start_year: localStartYear.value,
    horizon_years: localHorizon.value,
    pro_yr_base: localProYrBase.value,
    actuals_through: useActuals.value ? localActualsThrough.value : null,
  })
}

// MRI functions
async function loadMriQueries() {
  if (mriQueries.value.length > 0) return
  mriLoading.value = true
  try {
    const client = (await import('../../api/client')).default
    const [qRes, sRes] = await Promise.all([
      client.get('/api/data/mri/queries'),
      client.get('/api/data/mri/status'),
    ])
    mriQueries.value = qRes.data.queries
    mriServers.value = sRes.data.servers
  } catch (e: any) {
    data.addToast('Failed to load MRI queries: ' + (e.response?.data?.error || e.message), 'error')
  } finally {
    mriLoading.value = false
  }
}

async function handleMriRun(queryName: string) {
  mriRunning.value = queryName
  mriRunResult.value = null
  try {
    const client = (await import('../../api/client')).default
    const res = await client.post(`/api/data/mri/queries/${queryName}/run`, {})
    mriRunResult.value = res.data
    data.addToast(`${queryName}: ${res.data.rows.toLocaleString()} rows downloaded`, 'success')
  } catch (e: any) {
    data.addToast(`${queryName} failed: ${e.response?.data?.error || e.message}`, 'error')
  } finally {
    mriRunning.value = null
  }
}

async function handleMriDownload(queryName: string) {
  mriRunning.value = queryName
  try {
    const client = (await import('../../api/client')).default
    const res = await client.get(`/api/data/mri/queries/${queryName}/download`, { responseType: 'blob' })
    const url = window.URL.createObjectURL(new Blob([res.data]))
    const a = document.createElement('a')
    a.href = url
    a.download = `${queryName}.csv`
    a.click()
    window.URL.revokeObjectURL(url)
    data.addToast(`${queryName}: downloaded`, 'success')
  } catch (e: any) {
    data.addToast(`${queryName} download failed: ${e.response?.data?.error || e.message}`, 'error')
  } finally {
    mriRunning.value = null
  }
}

async function handleMriRefresh() {
  if (!confirm('Refresh ALL app data from MRI? This replaces current database tables.')) return
  mriRefreshing.value = true
  mriRefreshResults.value = null
  try {
    const client = (await import('../../api/client')).default
    const res = await client.post('/api/data/mri/refresh', {})
    mriRefreshResults.value = res.data.queries
    const ok = Object.values(res.data.queries).filter((r: any) => r.status === 'ok').length
    const err = Object.values(res.data.queries).filter((r: any) => r.status === 'error').length
    data.addToast(
      `MRI refresh: ${ok} tables updated in ${res.data.elapsed_seconds}s${err > 0 ? ', ' + err + ' errors' : ''}`,
      err > 0 ? 'error' : 'success'
    )
    await data.loadDeals()
  } catch (e: any) {
    data.addToast('MRI refresh failed: ' + (e.response?.data?.error || e.message), 'error')
  } finally {
    mriRefreshing.value = false
  }
}

async function handleMriRefreshSingle(queryName: string) {
  mriRunning.value = queryName
  try {
    const client = (await import('../../api/client')).default
    const res = await client.post(`/api/data/mri/refresh/${queryName}`, {})
    const tables = res.data.tables || {}
    const totalRows = Object.values(tables).reduce((s: number, t: any) => s + t.rows, 0)
    data.addToast(`${queryName}: ${totalRows.toLocaleString()} rows imported (${res.data.elapsed_seconds}s)`, 'success')
    await data.loadDeals()
  } catch (e: any) {
    data.addToast(`${queryName} import failed: ${e.response?.data?.error || e.message}`, 'error')
  } finally {
    mriRunning.value = null
  }
}

// Sidebar collapse toggle
const collapsed = ref(false)

function toggleCollapsed() {
  collapsed.value = !collapsed.value
  document.documentElement.style.setProperty(
    '--sidebar-width',
    collapsed.value ? '40px' : '240px'
  )
}
</script>

<template>
  <aside class="sidebar" :class="{ collapsed }">
    <div class="sidebar-header">
      <h2 v-if="!collapsed">Waterfall XIRR</h2>
      <button class="toggle-btn" @click="toggleCollapsed()">
        {{ collapsed ? '>' : '<' }}
      </button>
    </div>

    <nav class="sidebar-nav" v-show="!collapsed">
      <router-link
        v-for="item in navItems"
        :key="item.path"
        :to="item.path"
        class="nav-item"
        :class="{ active: route.path === item.path }"
      >
        {{ item.label }}
      </router-link>
    </nav>

    <div class="sidebar-tools" v-show="!collapsed">
      <!-- Reload -->
      <button class="btn btn-sm btn-full" @click="handleReload">
        Reload Data
      </button>

      <!-- Report Settings -->
      <div class="tool-section">
        <button class="section-toggle" @click="showConfig = !showConfig">
          {{ showConfig ? '▾' : '▸' }} Report Settings
        </button>
        <div v-if="showConfig" class="section-body">
          <div class="config-row">
            <label>Start Year</label>
            <input type="number" v-model.number="localStartYear" min="2000" max="2100" />
          </div>
          <div class="config-row">
            <label>Horizon (yrs)</label>
            <input type="number" v-model.number="localHorizon" min="1" max="30" />
          </div>
          <div class="config-row">
            <label>Pro_Yr Base</label>
            <input type="number" v-model.number="localProYrBase" min="1900" max="2100" />
          </div>
          <div class="config-row">
            <label>
              <input type="checkbox" v-model="useActuals" style="margin-right: 4px;" />
              YTD Actuals
            </label>
          </div>
          <div v-if="useActuals && monthEndOptions.length" class="config-row">
            <label>Through</label>
            <select v-model="localActualsThrough" class="config-select">
              <option v-for="opt in monthEndOptions" :key="opt.value" :value="opt.value">
                {{ opt.label }}
              </option>
            </select>
          </div>
          <div v-if="useActuals && !monthEndOptions.length" class="config-caption">
            No completed months yet this year.
          </div>
          <button class="btn btn-xs btn-full" @click="handleConfigSave">
            Apply Settings
          </button>
        </div>
      </div>

      <!-- MRI Data -->
      <div class="tool-section">
        <button class="section-toggle" @click="showMriTools = !showMriTools; if (showMriTools) loadMriQueries()">
          {{ showMriTools ? '▾' : '▸' }} MRI Data
        </button>
        <div v-if="showMriTools" class="section-body">
          <!-- Connection status -->
          <div v-if="mriLoading" class="db-caption">Loading queries...</div>
          <div v-else-if="Object.keys(mriServers).length" class="mri-status-row">
            <span
              v-for="(info, key) in mriServers"
              :key="key"
              class="mri-server-badge"
              :class="info.status"
              :title="info.status === 'ok' ? `${info.latency_ms}ms` : info.error"
            >{{ key.toUpperCase() }} {{ info.status === 'ok' ? '●' : '✕' }}</span>
          </div>

          <!-- Query list -->
          <div v-if="mriQueries.length" class="mri-query-list">
            <div
              v-for="q in mriQueries"
              :key="q.name"
              class="mri-query-row"
              :title="q.description"
            >
              <span class="mri-query-name">{{ q.name }}</span>
              <span class="mri-query-actions">
                <button
                  class="mri-action-btn"
                  title="Download CSV"
                  @click="handleMriDownload(q.name)"
                  :disabled="mriRunning !== null || mriRefreshing"
                >↓</button>
                <button
                  class="mri-action-btn"
                  title="Run & save to network folder"
                  @click="handleMriRun(q.name)"
                  :disabled="mriRunning !== null || mriRefreshing"
                >▶</button>
                <button
                  v-if="q.importable && auth.user?.role === 'admin'"
                  class="mri-action-btn import"
                  title="Import to database"
                  @click="handleMriRefreshSingle(q.name)"
                  :disabled="mriRunning !== null || mriRefreshing"
                >⇪</button>
              </span>
            </div>
          </div>

          <!-- Running indicator -->
          <div v-if="mriRunning" class="mri-running">
            Running {{ mriRunning }}...
          </div>

          <!-- Run result -->
          <div v-if="mriRunResult && !mriRunning" class="mri-run-result">
            {{ mriRunResult.query }}: {{ mriRunResult.rows.toLocaleString() }} rows ({{ mriRunResult.elapsed_seconds }}s)
          </div>

          <!-- Admin: Full Refresh -->
          <div v-if="auth.user?.role === 'admin'" class="db-sub" style="margin-top: 6px">
            <button
              class="btn btn-xs btn-full mri-refresh-btn"
              @click="handleMriRefresh"
              :disabled="mriRefreshing || mriRunning !== null"
            >
              {{ mriRefreshing ? 'Refreshing all tables...' : 'Refresh All Data from MRI' }}
            </button>
          </div>

          <!-- Refresh results -->
          <div v-if="mriRefreshResults" class="import-results">
            <div
              v-for="(res, qname) in mriRefreshResults"
              :key="qname"
              class="import-row"
              :class="res.status"
            >
              <span class="import-table">{{ qname }}</span>
              <span class="import-status">{{ res.status === 'ok'
                ? Object.values(res.tables || {}).reduce((s: number, t: any) => s + t.rows, 0).toLocaleString() + ' rows'
                : 'error' }}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Database Tools -->
      <div class="tool-section">
        <button class="section-toggle" @click="showDbTools = !showDbTools">
          {{ showDbTools ? '▾' : '▸' }} Database Tools
        </button>
        <div v-if="showDbTools" class="section-body">
          <!-- Import CSVs -->
          <div class="db-sub">
            <span class="db-label">Import CSVs</span>
            <p class="db-caption">Select MRI CSV files to upload and import.</p>
            <input
              type="file"
              multiple
              accept=".csv"
              class="db-file-input"
              @change="handleFileSelect"
            />

            <!-- Matched file list -->
            <div v-if="uploadMatches.length" class="csv-list" style="margin-top: 4px">
              <div class="csv-list-header">
                <span class="csv-count">{{ uploadMatches.filter(m => m.table_name && !m.protected).length }} importable</span>
              </div>
              <div
                v-for="m in uploadMatches"
                :key="m.file.name"
                class="csv-row"
                :class="{ 'csv-missing': !m.table_name, 'csv-protected': m.protected }"
              >
                <span class="csv-label" :title="m.description || m.file.name">
                  <span class="csv-name">{{ m.table_name || m.file.name }}</span>
                  <span v-if="m.protected" class="csv-badge protected">locked</span>
                  <span v-else-if="!m.table_name" class="csv-badge missing">no match</span>
                  <span v-else class="csv-badge" style="color: #81c784">{{ (m.file.size / 1024).toFixed(0) }}KB</span>
                </span>
              </div>
            </div>

            <p class="db-caption">Protected tables (waterfalls, comments) are never overwritten.</p>
            <button
              class="btn btn-xs btn-full"
              style="margin-top: 4px"
              @click="handleUploadImport"
              :disabled="uploading || uploadMatches.filter(m => m.table_name && !m.protected).length === 0"
            >
              {{ uploading ? `Importing ${uploadProgress}` : uploadMatches.filter(m => m.table_name && !m.protected).length === 0 ? 'Select CSV Files' : `Upload & Import ${uploadMatches.filter(m => m.table_name && !m.protected).length} CSV${uploadMatches.filter(m => m.table_name && !m.protected).length > 1 ? 's' : ''}` }}
            </button>
          </div>

          <!-- Import Results -->
          <div v-if="data.importResult" class="import-results">
            <div
              v-for="(res, table) in data.importResult"
              :key="table"
              class="import-row"
              :class="res.status"
            >
              <span class="import-table">{{ table }}</span>
              <span class="import-status">{{ res.status }}</span>
            </div>
          </div>

          <div class="db-divider"></div>

          <!-- Export Database -->
          <div class="db-sub">
            <span class="db-label">Export Database</span>
            <button
              class="btn btn-xs btn-full"
              @click="handleExport"
              :disabled="data.exporting"
            >
              {{ data.exporting ? 'Preparing...' : 'Download Export (.zip)' }}
            </button>
          </div>

          <!-- DB Info -->
          <div v-if="data.config" class="db-info">
            <span>DB: {{ data.config.db_path }}</span>
          </div>
        </div>
      </div>
    </div>

    <div class="sidebar-footer" v-show="!collapsed">
      <div class="user-info" v-if="auth.user">
        <span>{{ auth.user.username }}</span>
        <span class="user-role">{{ auth.user.role }}</span>
      </div>
      <button class="btn btn-logout" @click="handleLogout">
        Logout
      </button>
    </div>
  </aside>
</template>

<style scoped>
.sidebar {
  position: fixed;
  left: 0;
  top: 0;
  bottom: 0;
  width: var(--sidebar-width);
  background: var(--color-primary);
  color: white;
  display: flex;
  flex-direction: column;
  z-index: 100;
  transition: width 0.2s;
  overflow-y: auto;
}

.sidebar.collapsed {
  width: 40px;
}

.sidebar-header {
  padding: 12px 16px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  display: flex;
  align-items: center;
  justify-content: space-between;
  min-height: 48px;
}

.sidebar-header h2 {
  font-size: 15px;
  font-weight: 600;
  white-space: nowrap;
}

.toggle-btn {
  background: none;
  border: none;
  color: rgba(255, 255, 255, 0.6);
  cursor: pointer;
  font-size: 14px;
  padding: 2px 6px;
}

.toggle-btn:hover {
  color: white;
}

.sidebar-nav {
  padding: 4px 0;
}

.nav-item {
  display: block;
  padding: 9px 16px;
  color: rgba(255, 255, 255, 0.7);
  text-decoration: none;
  font-size: 13px;
  transition: all 0.15s;
}

.nav-item:hover {
  color: white;
  background: rgba(255, 255, 255, 0.1);
}

.nav-item.active {
  color: white;
  background: rgba(255, 255, 255, 0.15);
  border-left: 3px solid white;
}

/* Tools Section */
.sidebar-tools {
  padding: 8px 12px;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
}

.tool-section {
  margin-top: 8px;
}

.section-toggle {
  background: none;
  border: none;
  color: rgba(255, 255, 255, 0.8);
  cursor: pointer;
  font-size: 12px;
  padding: 4px 0;
  width: 100%;
  text-align: left;
}

.section-toggle:hover { color: white; }

.section-body {
  padding: 6px 0 4px 0;
}

/* Config */
.config-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 4px;
}

.config-row label {
  font-size: 11px;
  color: rgba(255, 255, 255, 0.7);
}

.config-row input[type="number"] {
  width: 70px;
  padding: 2px 6px;
  border: 1px solid rgba(255, 255, 255, 0.3);
  border-radius: 3px;
  background: rgba(255, 255, 255, 0.1);
  color: white;
  font-size: 12px;
  text-align: right;
}

.config-select {
  width: 120px;
  padding: 2px 4px;
  border: 1px solid rgba(255, 255, 255, 0.3);
  border-radius: 3px;
  background: rgba(255, 255, 255, 0.1);
  color: white;
  font-size: 11px;
}

.config-caption {
  font-size: 10px;
  color: rgba(255, 255, 255, 0.5);
  margin: 2px 0;
}

/* MRI Data tools */
.mri-status-row {
  display: flex;
  gap: 8px;
  margin-bottom: 6px;
}

.mri-server-badge {
  font-size: 10px;
  font-weight: 600;
  padding: 1px 6px;
  border-radius: 3px;
}
.mri-server-badge.ok { color: #81c784; }
.mri-server-badge.error { color: #ef5350; }

.mri-query-list {
  max-height: 240px;
  overflow-y: auto;
  margin-bottom: 4px;
}

.mri-query-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 3px 6px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}
.mri-query-row:last-child { border-bottom: none; }
.mri-query-row:hover { color: white; }

.mri-query-name {
  font-size: 10px;
  color: rgba(255, 255, 255, 0.8);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  flex: 1;
}

.mri-query-actions {
  display: flex;
  gap: 2px;
  flex-shrink: 0;
}

.mri-action-btn {
  background: none;
  border: 1px solid rgba(255, 255, 255, 0.2);
  color: rgba(255, 255, 255, 0.7);
  border-radius: 3px;
  cursor: pointer;
  font-size: 10px;
  padding: 1px 5px;
  line-height: 1.2;
}
.mri-action-btn:hover { color: white; }
.mri-action-btn:disabled { opacity: 0.3; cursor: not-allowed; }
.mri-action-btn.import { color: #81c784; border-color: rgba(129, 199, 132, 0.3); }
.mri-action-btn.import:hover { color: #a5d6a7; }

.mri-running {
  font-size: 10px;
  color: #90caf9;
  padding: 4px 0;
  animation: pulse 1.5s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.mri-run-result {
  font-size: 10px;
  color: #81c784;
  padding: 2px 0;
}

.mri-refresh-btn {
  border-color: rgba(129, 199, 132, 0.3) !important;
}
.mri-refresh-btn:hover:not(:disabled) {
  color: #a5d6a7;
}

/* Database tools */
.db-sub { margin-bottom: 8px; }

.db-label {
  font-size: 11px;
  font-weight: 600;
  color: rgba(255, 255, 255, 0.9);
  display: block;
  margin-bottom: 4px;
}

/* CSV file list */
.csv-list {
  max-height: 200px;
  overflow-y: auto;
  margin: 6px 0;
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 3px;
  padding: 2px 0;
}

.csv-list-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 2px 6px 4px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.csv-count {
  font-size: 10px;
  color: rgba(255, 255, 255, 0.6);
}

.csv-actions { display: flex; gap: 6px; }

.link-btn {
  background: none;
  border: none;
  color: rgba(255, 255, 255, 0.6);
  font-size: 10px;
  cursor: pointer;
  padding: 0;
  text-decoration: underline;
}

.link-btn:hover { color: white; }

.csv-row {
  padding: 1px 6px;
}

.csv-row:hover { background: rgba(255, 255, 255, 0.05); }

.csv-label {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 11px;
  color: rgba(255, 255, 255, 0.8);
  cursor: pointer;
}

.csv-label input[type="checkbox"] {
  margin: 0;
  accent-color: #81c784;
}

.csv-name {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.csv-badge {
  font-size: 9px;
  padding: 0 4px;
  border-radius: 2px;
  flex-shrink: 0;
}

.csv-badge.protected {
  background: rgba(255, 183, 77, 0.3);
  color: #ffb74d;
}

.csv-badge.missing {
  color: rgba(255, 255, 255, 0.35);
}

.csv-missing .csv-name { color: rgba(255, 255, 255, 0.35); }
.csv-protected .csv-name { color: rgba(255, 183, 77, 0.7); }

.db-caption {
  font-size: 10px;
  color: rgba(255, 255, 255, 0.5);
  margin-bottom: 4px;
}

.db-file-input {
  width: 100%;
  font-size: 11px;
  color: rgba(255, 255, 255, 0.8);
  margin-bottom: 2px;
}
.db-file-input::file-selector-button {
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  color: rgba(255, 255, 255, 0.8);
  border-radius: 3px;
  padding: 2px 8px;
  font-size: 11px;
  cursor: pointer;
}
.db-file-input::file-selector-button:hover {
  background: rgba(255, 255, 255, 0.2);
}

.db-divider {
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  margin: 8px 0;
}

.db-info {
  font-size: 10px;
  color: rgba(255, 255, 255, 0.4);
  margin-top: 6px;
  word-break: break-all;
}

/* Import results */
.import-results {
  max-height: 120px;
  overflow-y: auto;
  margin: 4px 0;
}

.import-row {
  display: flex;
  justify-content: space-between;
  font-size: 10px;
  padding: 1px 0;
}

.import-table { color: rgba(255, 255, 255, 0.7); }
.import-status { font-weight: 500; }
.import-row.success .import-status { color: #81c784; }
.import-row.protected .import-status { color: #ffb74d; }
.import-row.skipped .import-status { color: rgba(255, 255, 255, 0.5); }
.import-row.error .import-status { color: #ef5350; }

/* Buttons */
.btn {
  padding: 4px 12px;
  border: 1px solid rgba(255, 255, 255, 0.3);
  background: transparent;
  color: white;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
}

.btn:hover { background: rgba(255, 255, 255, 0.1); }
.btn:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-full { width: 100%; }
.btn-xs { font-size: 11px; padding: 3px 8px; }

.btn-text {
  border: none;
  text-decoration: underline;
  padding: 0;
}

/* Footer */
.sidebar-footer {
  padding: 10px 12px;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  margin-top: auto;
}

.user-info {
  font-size: 12px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.user-role {
  font-size: 10px;
  color: rgba(255, 255, 255, 0.5);
  text-transform: capitalize;
}

.btn-logout {
  width: 100%;
  padding: 6px 12px;
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(255, 255, 255, 0.2);
  color: rgba(255, 255, 255, 0.85);
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
  transition: all 0.15s;
}

.btn-logout:hover {
  background: rgba(239, 83, 80, 0.25);
  border-color: rgba(239, 83, 80, 0.5);
  color: white;
}
</style>
