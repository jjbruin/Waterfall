<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { useAuthStore } from '../../stores/auth'
import { useDataStore } from '../../stores/data'

const route = useRoute()
const auth = useAuthStore()
const data = useDataStore()

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

// Database tools
const showDbTools = ref(false)
const csvFolder = ref('')
const selectedCsvs = ref<Set<string>>(new Set())

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

async function handleScanFolder() {
  if (!csvFolder.value.trim()) return
  selectedCsvs.value = new Set()
  await data.scanCsvs(csvFolder.value.trim())
  // Auto-select all found, non-protected CSVs
  for (const csv of data.availableCsvs) {
    if (csv.found && !csv.protected) {
      selectedCsvs.value.add(csv.table_name)
    }
  }
}

function toggleCsv(tableName: string) {
  if (selectedCsvs.value.has(tableName)) {
    selectedCsvs.value.delete(tableName)
  } else {
    selectedCsvs.value.add(tableName)
  }
}

function selectAllCsvs() {
  for (const csv of data.availableCsvs) {
    if (csv.found && !csv.protected) selectedCsvs.value.add(csv.table_name)
  }
}

function selectNoneCsvs() {
  selectedCsvs.value = new Set()
}

async function handleImportSelected() {
  if (!csvFolder.value.trim() || selectedCsvs.value.size === 0) return
  const selected = [...selectedCsvs.value]
  if (selected.length === data.availableCsvs.filter(c => c.found && !c.protected).length) {
    // All importable CSVs selected — use bulk import
    await data.importCsvs(csvFolder.value.trim())
  } else {
    // Import one at a time
    for (const tableName of selected) {
      await data.importCsvs(csvFolder.value.trim(), tableName)
    }
  }
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

      <!-- Database Tools -->
      <div class="tool-section">
        <button class="section-toggle" @click="showDbTools = !showDbTools">
          {{ showDbTools ? '▾' : '▸' }} Database Tools
        </button>
        <div v-if="showDbTools" class="section-body">
          <!-- Import CSVs -->
          <div class="db-sub">
            <span class="db-label">Import CSVs</span>
            <div class="folder-row">
              <input
                type="text"
                v-model="csvFolder"
                placeholder="C:\Path\To\MRI_Exports"
                class="db-input"
                @keyup.enter="handleScanFolder"
              />
              <button
                class="btn btn-xs btn-scan"
                @click="handleScanFolder"
                :disabled="data.scanningCsvs || !csvFolder.trim()"
                title="Scan folder for CSVs"
              >
                {{ data.scanningCsvs ? '...' : 'Scan' }}
              </button>
            </div>

            <!-- CSV file list -->
            <div v-if="data.availableCsvs.length" class="csv-list">
              <div class="csv-list-header">
                <span class="csv-count">{{ data.availableCsvs.filter(c => c.found).length }} found</span>
                <span class="csv-actions">
                  <button class="link-btn" @click="selectAllCsvs">All</button>
                  <button class="link-btn" @click="selectNoneCsvs">None</button>
                </span>
              </div>
              <div
                v-for="csv in data.availableCsvs"
                :key="csv.table_name"
                class="csv-row"
                :class="{ 'csv-missing': !csv.found, 'csv-protected': csv.protected }"
              >
                <label class="csv-label" :title="csv.description + ' (' + csv.csv_file + ')'">
                  <input
                    type="checkbox"
                    :checked="selectedCsvs.has(csv.table_name)"
                    :disabled="!csv.found || csv.protected"
                    @change="toggleCsv(csv.table_name)"
                  />
                  <span class="csv-name">{{ csv.table_name }}</span>
                  <span v-if="csv.protected" class="csv-badge protected">locked</span>
                  <span v-else-if="!csv.found" class="csv-badge missing">not found</span>
                </label>
              </div>
            </div>

            <p class="db-caption">Protected tables (waterfalls, comments) are never overwritten.</p>
            <button
              class="btn btn-xs btn-full"
              @click="handleImportSelected"
              :disabled="data.importing || selectedCsvs.size === 0"
            >
              {{ data.importing ? 'Importing...' : selectedCsvs.size === 0 ? 'Select CSVs to Import' : `Import ${selectedCsvs.size} CSV${selectedCsvs.size > 1 ? 's' : ''}` }}
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
        <button class="btn btn-sm btn-text" @click="auth.logout()">Logout</button>
      </div>
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

/* Database tools */
.db-sub { margin-bottom: 8px; }

.db-label {
  font-size: 11px;
  font-weight: 600;
  color: rgba(255, 255, 255, 0.9);
  display: block;
  margin-bottom: 4px;
}

.db-input {
  width: 100%;
  padding: 4px 6px;
  border: 1px solid rgba(255, 255, 255, 0.3);
  border-radius: 3px;
  background: rgba(255, 255, 255, 0.1);
  color: white;
  font-size: 11px;
  margin-bottom: 4px;
}

.db-input::placeholder { color: rgba(255, 255, 255, 0.4); }

.folder-row {
  display: flex;
  gap: 4px;
  margin-bottom: 4px;
}

.folder-row .db-input {
  flex: 1;
  margin-bottom: 0;
}

.btn-scan {
  flex-shrink: 0;
  padding: 4px 8px;
  font-size: 11px;
  border: 1px solid rgba(255, 255, 255, 0.3);
  background: rgba(255, 255, 255, 0.1);
  color: white;
  border-radius: 3px;
  cursor: pointer;
  white-space: nowrap;
}

.btn-scan:hover { background: rgba(255, 255, 255, 0.2); }
.btn-scan:disabled { opacity: 0.5; cursor: not-allowed; }

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
  padding: 8px 16px;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  margin-top: auto;
}

.user-info {
  font-size: 12px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
</style>
