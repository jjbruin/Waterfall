import { defineStore } from 'pinia'
import { ref } from 'vue'
import api from '../api/client'

interface Deal {
  vcode: string
  Investment_Name?: string
  Asset_Type?: string
  Sale_Status?: string
  [key: string]: any
}

interface AppConfig {
  start_year: number
  horizon_years: number
  pro_yr_base: number
  actuals_through: string | null
  db_path: string
}

interface ImportResult {
  [table: string]: { status: string; rows?: number; error?: string }
}

interface CsvInfo {
  table_name: string
  csv_file: string
  description: string
  found: boolean
  protected: boolean
}

interface TableDef {
  table_name: string
  csv_file: string
  description: string
  protected: boolean
}

export const useDataStore = defineStore('data', () => {
  const deals = ref<Deal[]>([])
  const allDeals = ref<Deal[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)

  // Config
  const config = ref<AppConfig | null>(null)

  // Database tools
  const importing = ref(false)
  const importResult = ref<ImportResult | null>(null)
  const exporting = ref(false)
  const availableCsvs = ref<CsvInfo[]>([])
  const scanningCsvs = ref(false)
  const tableDefs = ref<TableDef[]>([])
  const tableDefsLoaded = ref(false)

  // Toast notifications
  const toasts = ref<Array<{ id: number; message: string; type: 'success' | 'error' | 'info' }>>([])
  let toastId = 0

  function addToast(message: string, type: 'success' | 'error' | 'info' = 'info') {
    const id = ++toastId
    toasts.value.push({ id, message, type })
    setTimeout(() => {
      toasts.value = toasts.value.filter((t) => t.id !== id)
    }, 5000)
  }

  function dismissToast(id: number) {
    toasts.value = toasts.value.filter((t) => t.id !== id)
  }

  async function loadDeals() {
    loading.value = true
    error.value = null
    try {
      const [active, all] = await Promise.all([
        api.get('/api/data/deals'),
        api.get('/api/data/deals/all'),
      ])
      deals.value = active.data.deals
      allDeals.value = all.data.deals
    } catch (e: any) {
      error.value = e.response?.data?.error || e.message
      addToast('Failed to load deals: ' + (error.value || 'Unknown error'), 'error')
    } finally {
      loading.value = false
    }
  }

  async function loadConfig() {
    try {
      const res = await api.get('/api/data/config')
      config.value = res.data
    } catch {
      // Config endpoint may not be available
    }
  }

  async function updateConfig(updates: Partial<AppConfig>) {
    try {
      const res = await api.put('/api/data/config', updates)
      config.value = {
        start_year: res.data.start_year,
        horizon_years: res.data.horizon_years,
        pro_yr_base: res.data.pro_yr_base,
        actuals_through: res.data.actuals_through || null,
        db_path: config.value?.db_path || '',
      }
      addToast('Configuration updated', 'success')
    } catch (e: any) {
      addToast('Failed to update config: ' + (e.response?.data?.error || e.message), 'error')
    }
  }

  async function reloadData() {
    try {
      await api.post('/api/data/reload')
      await loadDeals()
      addToast('All caches cleared and data reloaded', 'success')
    } catch (e: any) {
      addToast('Reload failed: ' + (e.response?.data?.error || e.message), 'error')
    }
  }

  async function loadTableDefs() {
    if (tableDefsLoaded.value) return
    try {
      const res = await api.get('/api/data/table-definitions')
      tableDefs.value = res.data.tables
      tableDefsLoaded.value = true
    } catch {
      // endpoint may not exist yet
    }
  }

  async function scanCsvs(folderPath: string) {
    scanningCsvs.value = true
    availableCsvs.value = []
    try {
      const res = await api.post('/api/data/list-csvs', { folder_path: folderPath })
      availableCsvs.value = res.data.csvs
    } catch (e: any) {
      addToast('Scan failed: ' + (e.response?.data?.error || e.message), 'error')
    } finally {
      scanningCsvs.value = false
    }
  }

  async function importCsvs(folderPath: string, tableName?: string) {
    importing.value = true
    importResult.value = null
    try {
      const payload: any = { folder_path: folderPath }
      if (tableName) payload.table_name = tableName
      const res = await api.post('/api/data/import', payload)
      importResult.value = res.data.results

      // Summarize
      const results = res.data.results
      if (results._error) {
        addToast('Import error: ' + results._error.error, 'error')
      } else {
        const ok = Object.values(results).filter((v: any) => v.status === 'success').length
        const prot = Object.values(results).filter((v: any) => v.status === 'protected').length
        const skip = Object.values(results).filter((v: any) => v.status === 'skipped').length
        const err = Object.values(results).filter((v: any) => v.status === 'error').length
        addToast(`Import complete: ${ok} updated, ${prot} protected, ${skip} skipped, ${err} errors`,
          err > 0 ? 'error' : 'success')
      }

      // Reload deals after import
      await loadDeals()
    } catch (e: any) {
      addToast('Import failed: ' + (e.response?.data?.error || e.message), 'error')
    } finally {
      importing.value = false
    }
  }

  async function exportDatabase() {
    exporting.value = true
    try {
      const res = await api.get('/api/data/export', { responseType: 'blob' })
      const url = URL.createObjectURL(new Blob([res.data]))
      const a = document.createElement('a')
      a.href = url
      const ts = new Date().toISOString().replace(/[:-]/g, '').slice(0, 15)
      a.download = `waterfall_db_export_${ts}.zip`
      a.click()
      URL.revokeObjectURL(url)
      addToast('Database export downloaded', 'success')
    } catch (e: any) {
      addToast('Export failed: ' + (e.response?.data?.error || e.message), 'error')
    } finally {
      exporting.value = false
    }
  }

  function getDealName(vcode: string): string {
    const deal = allDeals.value.find((d) => d.vcode === vcode)
    return deal?.Investment_Name || vcode
  }

  return {
    deals, allDeals, loading, error, config,
    importing, importResult, exporting,
    availableCsvs, scanningCsvs,
    tableDefs, tableDefsLoaded,
    toasts,
    loadDeals, loadConfig, updateConfig, reloadData,
    loadTableDefs, scanCsvs, importCsvs, exportDatabase,
    getDealName, addToast, dismissToast,
  }
})
