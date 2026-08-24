<template>
  <div class="argus-import">
    <div class="import-section">
      <h4>Argus Enterprise Import</h4>

      <div class="import-label-row">
        <label>Projection Label</label>
        <input v-model="importLabel" type="text" placeholder="e.g. Partner Projection" />
      </div>

      <div class="file-zones">
        <!-- Monthly Cash Flow -->
        <div class="file-zone" :class="{ active: cashflowFile }">
          <div class="zone-header">
            <span class="zone-icon">📊</span>
            <span>Monthly Cash Flow</span>
            <span v-if="cashflowFile" class="file-name">{{ cashflowFile.name }}</span>
          </div>
          <input type="file" ref="cashflowInput" accept=".xlsx,.xls"
                 @change="e => cashflowFile = e.target.files[0]" />
          <button class="btn-select" @click="$refs.cashflowInput.click()">
            {{ cashflowFile ? 'Change File' : 'Select File' }}
          </button>
        </div>

        <!-- Rent Roll Summary -->
        <div class="file-zone" :class="{ active: rentRollFile }">
          <div class="zone-header">
            <span class="zone-icon">🏢</span>
            <span>Rent Roll Summary</span>
            <span v-if="rentRollFile" class="file-name">{{ rentRollFile.name }}</span>
          </div>
          <input type="file" ref="rentRollInput" accept=".xlsx,.xls"
                 @change="e => rentRollFile = e.target.files[0]" />
          <button class="btn-select" @click="$refs.rentRollInput.click()">
            {{ rentRollFile ? 'Change File' : 'Select File' }}
          </button>
        </div>

        <!-- Revenue Assumptions -->
        <div class="file-zone" :class="{ active: revenueFile }">
          <div class="zone-header">
            <span class="zone-icon">📈</span>
            <span>Revenue Assumptions</span>
            <span v-if="revenueFile" class="file-name">{{ revenueFile.name }}</span>
          </div>
          <input type="file" ref="revenueInput" accept=".xlsx,.xls"
                 @change="e => revenueFile = e.target.files[0]" />
          <button class="btn-select" @click="$refs.revenueInput.click()">
            {{ revenueFile ? 'Change File' : 'Select File' }}
          </button>
        </div>
      </div>

      <button class="btn-import" :disabled="!canImport || importing" @click="runImport">
        {{ importing ? 'Importing...' : 'Import' }}
      </button>

      <!-- Import result -->
      <div v-if="importResult" class="import-result" :class="importResult.status">
        <div v-if="importResult.status === 'success'">
          <strong>Import successful</strong> — {{ importResult.mapped_count }}/{{ importResult.total_line_items }} line items mapped,
          {{ importResult.total_periods }} periods
        </div>
        <div v-else-if="importResult.status === 'duplicate'">
          <strong>Duplicate</strong> — {{ importResult.message }}
        </div>
        <div v-else>
          <strong>Error</strong> — {{ importResult.message }}
        </div>
      </div>

      <!-- Unmapped items -->
      <div v-if="unmappedItems.length" class="unmapped-section">
        <h5>Unmapped Line Items ({{ unmappedItems.length }})</h5>
        <p class="unmapped-hint">These line items could not be auto-mapped to COA accounts. You can assign them manually below.</p>
        <table class="unmapped-table">
          <thead>
            <tr><th>Line Item</th><th>COA Account</th></tr>
          </thead>
          <tbody>
            <tr v-for="(item, idx) in unmappedItems" :key="idx">
              <td>{{ item }}</td>
              <td>
                <select v-model="manualMappings[item]">
                  <option :value="null">— Skip —</option>
                  <option v-for="acct in coaOptions" :key="acct.value" :value="acct.value">
                    {{ acct.label }}
                  </option>
                </select>
              </td>
            </tr>
          </tbody>
        </table>
        <button class="btn-apply-mapping" :disabled="!hasManualMappings" @click="applyMappings">
          Apply Mappings
        </button>
      </div>

      <!-- Tenant preview -->
      <div v-if="tenantPreview.length" class="tenant-preview">
        <h5>Tenants ({{ tenantPreview.length }})</h5>
        <table class="tenant-table">
          <thead>
            <tr><th>Tenant</th><th>Suite</th><th>SF</th><th>Rent/SF</th><th>Lease End</th></tr>
          </thead>
          <tbody>
            <tr v-for="t in tenantPreview" :key="t.id">
              <td>{{ t.tenant_name }}</td>
              <td>{{ t.suite }}</td>
              <td>{{ fmtNum(t.square_feet) }}</td>
              <td>{{ fmtCurrency(t.base_rent_psf) }}</td>
              <td>{{ t.lease_end }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import api from '@/api/client'

const props = defineProps({
  vcode: { type: String, required: true },
  importType: { type: String, default: 'asset_management' },
  onImportComplete: { type: Function, default: null },
})

const importLabel = ref('Argus Projection')
const cashflowFile = ref(null)
const rentRollFile = ref(null)
const revenueFile = ref(null)
const importing = ref(false)
const importResult = ref(null)
const unmappedItems = ref([])
const manualMappings = ref({})
const tenantPreview = ref([])
const currentImportId = ref(null)

const canImport = computed(() => !!cashflowFile.value)
const hasManualMappings = computed(() =>
  Object.values(manualMappings.value).some(v => v !== null)
)

const coaOptions = [
  { value: 4010, label: '4010 - Rental Income' },
  { value: 4030, label: '4030 - Vacancy' },
  { value: 4040, label: '4040 - Concessions' },
  { value: 4043, label: '4043 - Bad Debt' },
  { value: 4075, label: '4075 - Other Revenue' },
  { value: 4090, label: '4090 - CAM Recovery' },
  { value: 4091, label: '4091 - RET Recovery' },
  { value: 4092, label: '4092 - INS Recovery' },
  { value: 5020, label: '5020 - G&A / Other Expense' },
  { value: 5040, label: '5040 - Management Fee' },
  { value: 5060, label: '5060 - CAM / Maintenance' },
  { value: 5090, label: '5090 - Real Estate Tax' },
  { value: 5110, label: '5110 - Insurance' },
  { value: 7050, label: '7050 - CapEx / TI / LC' },
]

async function runImport() {
  importing.value = true
  importResult.value = null
  unmappedItems.value = []
  tenantPreview.value = []

  try {
    // 1. Import cashflow (required)
    const cfForm = new FormData()
    cfForm.append('file', cashflowFile.value)
    cfForm.append('import_label', importLabel.value)
    cfForm.append('import_type', props.importType)

    const cfRes = await api.post(`/api/argus/${props.vcode}/import/cashflow`, cfForm)
    importResult.value = cfRes.data
    currentImportId.value = cfRes.data.import_id

    if (cfRes.data.status !== 'success') {
      return
    }

    unmappedItems.value = cfRes.data.unmapped_items || []
    // Reset manual mappings
    manualMappings.value = {}
    unmappedItems.value.forEach(item => { manualMappings.value[item] = null })

    const importId = cfRes.data.import_id

    // 2. Import rent roll (optional)
    if (rentRollFile.value) {
      const rrForm = new FormData()
      rrForm.append('file', rentRollFile.value)
      rrForm.append('import_id', importId)

      await api.post(`/api/argus/${props.vcode}/import/rent-roll`, rrForm)

      // Load tenant preview
      const tenRes = await api.get(`/api/argus/${props.vcode}/projections/${importId}/tenants`)
      tenantPreview.value = tenRes.data || []
    }

    // 3. Import revenue assumptions (optional)
    if (revenueFile.value) {
      const raForm = new FormData()
      raForm.append('file', revenueFile.value)
      raForm.append('import_id', importId)

      await api.post(`/api/argus/${props.vcode}/import/revenue-assumptions`, raForm)
    }

    if (props.onImportComplete) {
      props.onImportComplete(importId)
    }
  } catch (err) {
    importResult.value = { status: 'error', message: err.response?.data?.error || err.message }
  } finally {
    importing.value = false
  }
}

async function applyMappings() {
  if (!currentImportId.value) return

  const mappings = Object.entries(manualMappings.value)
    .filter(([, v]) => v !== null)
    .map(([lineItem, coaAccount]) => ({
      line_item: lineItem,
      coa_account: coaAccount,
      category: coaAccount < 5000 ? 'revenue' : coaAccount < 7000 ? 'expense' : 'capex',
    }))

  if (!mappings.length) return

  try {
    await api.put(
      `/api/argus/${props.vcode}/projections/${currentImportId.value}/mapping`,
      { mappings }
    )
    // Refresh unmapped list
    const mapRes = await api.get(
      `/api/argus/${props.vcode}/projections/${currentImportId.value}/mapping`
    )
    unmappedItems.value = (mapRes.data.unmapped || []).map(m => m.line_item)
    manualMappings.value = {}
    unmappedItems.value.forEach(item => { manualMappings.value[item] = null })
  } catch (err) {
    console.error('Failed to apply mappings:', err)
  }
}

function fmtNum(v) {
  if (v == null) return ''
  return Number(v).toLocaleString('en-US', { maximumFractionDigits: 0 })
}

function fmtCurrency(v) {
  if (v == null) return ''
  return '$' + Number(v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}
</script>

<style scoped>
.argus-import {
  border: 1px solid #dee2e6;
  border-radius: 6px;
  padding: 16px;
  background: #fff;
}

.import-section h4 {
  margin: 0 0 12px 0;
  font-size: 14px;
  font-weight: 600;
}

.import-label-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 12px;
}

.import-label-row label {
  font-size: 12px;
  font-weight: 500;
  white-space: nowrap;
}

.import-label-row input {
  flex: 1;
  padding: 4px 8px;
  border: 1px solid #ced4da;
  border-radius: 4px;
  font-size: 12px;
}

.file-zones {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-bottom: 12px;
}

.file-zone {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border: 1px dashed #ced4da;
  border-radius: 4px;
  background: #f8f9fa;
}

.file-zone.active {
  border-color: #28a745;
  border-style: solid;
  background: #f0fff4;
}

.file-zone input[type="file"] {
  display: none;
}

.zone-header {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
}

.zone-icon {
  font-size: 16px;
}

.file-name {
  color: #28a745;
  font-style: italic;
  margin-left: auto;
}

.btn-select {
  padding: 3px 10px;
  border: 1px solid #6c757d;
  border-radius: 3px;
  background: #fff;
  font-size: 11px;
  cursor: pointer;
  white-space: nowrap;
}

.btn-select:hover {
  background: #f8f9fa;
}

.btn-import {
  width: 100%;
  padding: 6px 0;
  border: none;
  border-radius: 4px;
  background: #0d6efd;
  color: #fff;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
}

.btn-import:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-import:hover:not(:disabled) {
  background: #0b5ed7;
}

.import-result {
  margin-top: 8px;
  padding: 8px 12px;
  border-radius: 4px;
  font-size: 12px;
}

.import-result.success {
  background: #d4edda;
  color: #155724;
}

.import-result.duplicate {
  background: #fff3cd;
  color: #856404;
}

.import-result.error {
  background: #f8d7da;
  color: #721c24;
}

.unmapped-section {
  margin-top: 12px;
}

.unmapped-section h5 {
  font-size: 12px;
  font-weight: 600;
  margin: 0 0 4px 0;
}

.unmapped-hint {
  font-size: 11px;
  color: #6c757d;
  margin: 0 0 8px 0;
}

.unmapped-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 11px;
}

.unmapped-table th,
.unmapped-table td {
  padding: 4px 6px;
  border: 1px solid #dee2e6;
  text-align: left;
}

.unmapped-table select {
  width: 100%;
  font-size: 11px;
  padding: 2px;
}

.btn-apply-mapping {
  margin-top: 6px;
  padding: 4px 12px;
  border: 1px solid #28a745;
  border-radius: 3px;
  background: #28a745;
  color: #fff;
  font-size: 11px;
  cursor: pointer;
}

.btn-apply-mapping:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.tenant-preview {
  margin-top: 12px;
}

.tenant-preview h5 {
  font-size: 12px;
  font-weight: 600;
  margin: 0 0 4px 0;
}

.tenant-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 11px;
}

.tenant-table th,
.tenant-table td {
  padding: 3px 6px;
  border: 1px solid #dee2e6;
  text-align: left;
}

.tenant-table th {
  background: #f8f9fa;
  font-weight: 600;
}
</style>
