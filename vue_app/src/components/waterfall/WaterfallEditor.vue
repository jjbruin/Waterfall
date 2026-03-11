<script setup lang="ts">
import { ref, watch, computed } from 'vue'
import { AgGridVue } from 'ag-grid-vue3'
import type { WaterfallStep } from '../../stores/waterfall'

const props = defineProps<{
  steps: WaterfallStep[]
  wfType: string
  vstateOptions: string[]
  saving: boolean
  dirty: boolean
}>()

const emit = defineEmits<{
  (e: 'update', steps: WaterfallStep[]): void
  (e: 'save'): void
  (e: 'validate'): void
  (e: 'preview'): void
  (e: 'reset'): void
  (e: 'copy-cf-to-cap'): void
  (e: 'export-csv'): void
}>()

const gridApi = ref<any>(null)
const rowData = ref<WaterfallStep[]>([...props.steps])

watch(() => props.steps, (newSteps) => {
  rowData.value = [...newSteps]
}, { deep: true })

const rowCount = computed(() => rowData.value.length)

const columnDefs = computed(() => [
  {
    field: 'iOrder', headerName: 'Order', width: 80, editable: true,
    valueParser: (params: any) => Number(params.newValue) || 0,
  },
  { field: 'PropCode', headerName: 'PropCode', width: 120, editable: true },
  {
    field: 'vState', headerName: 'vState', width: 120, editable: true,
    cellEditor: 'agSelectCellEditor',
    cellEditorParams: { values: props.vstateOptions },
  },
  {
    field: 'FXRate', headerName: 'FXRate', width: 100, editable: true,
    valueParser: (params: any) => parseFloat(params.newValue) || 0,
    valueFormatter: (params: any) => params.value != null ? params.value.toFixed(4) : '',
  },
  {
    field: 'nPercent', headerName: 'nPercent', width: 100, editable: true,
    valueParser: (params: any) => parseFloat(params.newValue) || 0,
    valueFormatter: (params: any) => params.value != null ? params.value.toFixed(4) : '',
  },
  {
    field: 'mAmount', headerName: 'mAmount', width: 110, editable: true,
    valueParser: (params: any) => parseFloat(params.newValue) || 0,
    valueFormatter: (params: any) => {
      if (params.value == null || params.value === 0) return ''
      return new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(params.value)
    },
  },
  { field: 'vtranstype', headerName: 'vtranstype', width: 180, editable: true },
  { field: 'vAmtType', headerName: 'vAmtType', width: 100, editable: true },
  { field: 'vNotes', headerName: 'vNotes', width: 160, editable: true },
])

const defaultColDef = {
  sortable: false,
  resizable: true,
  suppressMovable: true,
}

function onGridReady(params: any) {
  gridApi.value = params.api
}

function onCellValueChanged() {
  emit('update', [...rowData.value])
}

function addRow() {
  const maxOrder = rowData.value.reduce((max, r) => Math.max(max, r.iOrder || 0), 0)
  rowData.value = [...rowData.value, {
    iOrder: maxOrder + 1,
    PropCode: '',
    vState: '',
    FXRate: 0,
    nPercent: 0,
    mAmount: 0,
    vtranstype: '',
    vAmtType: '',
    vNotes: '',
  }]
  emit('update', rowData.value)
}

function removeSelected() {
  if (!gridApi.value) return
  const selected = gridApi.value.getSelectedRows()
  if (selected.length === 0) return
  rowData.value = rowData.value.filter((r) => !selected.includes(r))
  emit('update', rowData.value)
}

function moveRow(direction: 'up' | 'down') {
  if (!gridApi.value) return
  const selected = gridApi.value.getSelectedNodes()
  if (selected.length !== 1) return
  const idx = selected[0].rowIndex
  const newIdx = direction === 'up' ? idx - 1 : idx + 1
  if (newIdx < 0 || newIdx >= rowData.value.length) return
  const rows = [...rowData.value]
  const [moved] = rows.splice(idx, 1)
  rows.splice(newIdx, 0, moved)
  rowData.value = rows
  emit('update', rowData.value)
}
</script>

<template>
  <div class="waterfall-editor">
    <div class="editor-header">
      <div class="header-left">
        <span class="wf-type-label">{{ wfType }}</span>
        <span class="row-count">{{ rowCount }} steps</span>
        <span v-if="dirty" class="dirty-badge">unsaved</span>
      </div>
      <div class="header-actions">
        <button class="btn btn-sm" @click="addRow" title="Add row">+ Add Row</button>
        <button class="btn btn-sm" @click="removeSelected" title="Remove selected rows">- Remove</button>
        <button class="btn btn-sm" @click="moveRow('up')" title="Move selected row up">Up</button>
        <button class="btn btn-sm" @click="moveRow('down')" title="Move selected row down">Down</button>
        <span class="separator"></span>
        <button class="btn btn-sm" @click="emit('validate')">Validate</button>
        <button class="btn btn-sm" @click="emit('preview')">Preview</button>
        <button v-if="wfType === 'CF_WF'" class="btn btn-sm" @click="emit('copy-cf-to-cap')" title="Copy CF_WF steps to Cap_WF">
          CF -> Cap
        </button>
        <button class="btn btn-sm" @click="emit('export-csv')">Export CSV</button>
        <button class="btn btn-sm" @click="emit('reset')">Reset</button>
        <button class="btn btn-sm btn-primary" @click="emit('save')" :disabled="saving">
          {{ saving ? 'Saving...' : 'Save' }}
        </button>
      </div>
    </div>
    <ag-grid-vue
      class="ag-theme-alpine"
      style="height: 400px; width: 100%"
      :rowData="rowData"
      :columnDefs="columnDefs"
      :defaultColDef="defaultColDef"
      rowSelection="multiple"
      :stopEditingWhenCellsLoseFocus="true"
      :undoRedoCellEditing="true"
      @gridReady="onGridReady"
      @cellValueChanged="onCellValueChanged"
    />
  </div>
</template>

<style scoped>
.waterfall-editor {
  margin-bottom: 20px;
}

.editor-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
  flex-wrap: wrap;
  gap: 8px;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 10px;
}

.wf-type-label {
  font-weight: 700;
  font-size: 15px;
  color: var(--color-text);
}

.row-count {
  font-size: 12px;
  color: var(--color-text-secondary);
}

.dirty-badge {
  font-size: 11px;
  color: #c00;
  background: #fee;
  border: 1px solid #fcc;
  border-radius: 4px;
  padding: 1px 6px;
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}

.separator {
  width: 1px;
  height: 20px;
  background: var(--color-border);
  margin: 0 4px;
}

.btn {
  padding: 4px 10px;
  border: 1px solid var(--color-border);
  background: var(--color-surface);
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
  white-space: nowrap;
}

.btn:hover {
  background: #eee;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-primary {
  background: var(--color-accent);
  color: white;
  border-color: var(--color-accent);
}

.btn-primary:hover:not(:disabled) {
  background: #3a63ad;
}
</style>
