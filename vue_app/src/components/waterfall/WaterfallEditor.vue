<script setup lang="ts">
import { ref, watch, computed, nextTick } from 'vue'
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

const rows = ref<WaterfallStep[]>([])
const selectedRows = ref<Set<number>>(new Set())

watch(() => props.steps, (newSteps) => {
  rows.value = newSteps.map((s) => ({ ...s }))
  selectedRows.value = new Set()
}, { deep: true, immediate: true })

const rowCount = computed(() => rows.value.length)
const allSelected = computed(() => rows.value.length > 0 && selectedRows.value.size === rows.value.length)

function emitUpdate() {
  emit('update', rows.value.map((r) => ({ ...r })))
}

function onCellChange(rowIdx: number, field: keyof WaterfallStep, value: any) {
  const row = rows.value[rowIdx]
  if (field === 'iOrder') {
    ;(row as any)[field] = parseInt(value) || 0
  } else if (field === 'FXRate' || field === 'nPercent' || field === 'mAmount') {
    ;(row as any)[field] = parseFloat(value) || 0
  } else {
    ;(row as any)[field] = value
  }
  emitUpdate()
}

function toggleRow(idx: number) {
  if (selectedRows.value.has(idx)) {
    selectedRows.value.delete(idx)
  } else {
    selectedRows.value.add(idx)
  }
}

function toggleAll() {
  if (allSelected.value) {
    selectedRows.value = new Set()
  } else {
    selectedRows.value = new Set(rows.value.map((_, i) => i))
  }
}

function addRow() {
  const maxOrder = rows.value.reduce((max, r) => Math.max(max, r.iOrder || 0), 0)
  rows.value.push({
    iOrder: maxOrder + 1,
    PropCode: '',
    vState: '',
    FXRate: 0,
    nPercent: 0,
    mAmount: 0,
    vtranstype: '',
    vAmtType: '',
    vNotes: '',
  })
  emitUpdate()
}

function removeSelected() {
  if (selectedRows.value.size === 0) return
  rows.value = rows.value.filter((_, i) => !selectedRows.value.has(i))
  selectedRows.value = new Set()
  emitUpdate()
}

function moveRow(direction: 'up' | 'down') {
  if (selectedRows.value.size !== 1) return
  const idx = [...selectedRows.value][0]
  const newIdx = direction === 'up' ? idx - 1 : idx + 1
  if (newIdx < 0 || newIdx >= rows.value.length) return
  const temp = rows.value[idx]
  rows.value[idx] = rows.value[newIdx]
  rows.value[newIdx] = temp
  selectedRows.value = new Set([newIdx])
  emitUpdate()
}

function fmtNum(v: number, decimals: number): string {
  if (v == null) return ''
  return v.toFixed(decimals)
}
</script>

<template>
  <div class="waterfall-editor">
    <!-- Header with label and action buttons -->
    <div class="editor-header">
      <div class="header-left">
        <span class="wf-type-label">{{ wfType }}</span>
        <span class="row-count">{{ rowCount }} steps</span>
        <span v-if="dirty" class="dirty-badge">unsaved</span>
      </div>
      <div class="header-actions">
        <button class="btn btn-sm" @click="addRow" title="Add row">+ Add Row</button>
        <button class="btn btn-sm" @click="removeSelected" :disabled="selectedRows.size === 0" title="Remove selected rows">- Remove</button>
        <button class="btn btn-sm" @click="moveRow('up')" :disabled="selectedRows.size !== 1" title="Move selected row up">Up</button>
        <button class="btn btn-sm" @click="moveRow('down')" :disabled="selectedRows.size !== 1" title="Move selected row down">Down</button>
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

    <!-- Editable table -->
    <div class="table-wrapper">
      <table class="wf-table">
        <thead>
          <tr>
            <th class="col-sel"><input type="checkbox" :checked="allSelected" @change="toggleAll" /></th>
            <th class="col-order">iOrder</th>
            <th class="col-propcode">PropCode</th>
            <th class="col-vstate">vState</th>
            <th class="col-num">FXRate</th>
            <th class="col-num">nPercent</th>
            <th class="col-num">mAmount</th>
            <th class="col-text">vtranstype</th>
            <th class="col-amttype">vAmtType</th>
            <th class="col-notes">vNotes</th>
          </tr>
        </thead>
        <tbody>
          <tr v-if="rows.length === 0">
            <td colspan="10" class="empty-msg">No steps. Click "+ Add Row" to begin.</td>
          </tr>
          <tr
            v-for="(row, idx) in rows"
            :key="idx"
            :class="{ selected: selectedRows.has(idx) }"
          >
            <td class="col-sel">
              <input type="checkbox" :checked="selectedRows.has(idx)" @change="toggleRow(idx)" />
            </td>
            <td class="col-order">
              <input
                type="number"
                :value="row.iOrder"
                @change="onCellChange(idx, 'iOrder', ($event.target as HTMLInputElement).value)"
                min="0" max="30" step="1"
              />
            </td>
            <td class="col-propcode">
              <input
                type="text"
                :value="row.PropCode"
                @change="onCellChange(idx, 'PropCode', ($event.target as HTMLInputElement).value)"
              />
            </td>
            <td class="col-vstate">
              <select
                :value="row.vState"
                @change="onCellChange(idx, 'vState', ($event.target as HTMLSelectElement).value)"
              >
                <option value="">--</option>
                <option v-for="opt in vstateOptions" :key="opt" :value="opt">{{ opt }}</option>
              </select>
            </td>
            <td class="col-num">
              <input
                type="number"
                :value="row.FXRate"
                @change="onCellChange(idx, 'FXRate', ($event.target as HTMLInputElement).value)"
                step="0.0001" min="0" max="1"
              />
            </td>
            <td class="col-num">
              <input
                type="number"
                :value="row.nPercent"
                @change="onCellChange(idx, 'nPercent', ($event.target as HTMLInputElement).value)"
                step="0.0001" min="0" max="1"
              />
            </td>
            <td class="col-num">
              <input
                type="number"
                :value="row.mAmount"
                @change="onCellChange(idx, 'mAmount', ($event.target as HTMLInputElement).value)"
                step="1" min="0"
              />
            </td>
            <td class="col-text">
              <input
                type="text"
                :value="row.vtranstype"
                @change="onCellChange(idx, 'vtranstype', ($event.target as HTMLInputElement).value)"
              />
            </td>
            <td class="col-amttype">
              <input
                type="text"
                :value="row.vAmtType"
                @change="onCellChange(idx, 'vAmtType', ($event.target as HTMLInputElement).value)"
              />
            </td>
            <td class="col-notes">
              <input
                type="text"
                :value="row.vNotes"
                @change="onCellChange(idx, 'vNotes', ($event.target as HTMLInputElement).value)"
              />
            </td>
          </tr>
        </tbody>
      </table>
    </div>
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

/* Table */
.table-wrapper {
  overflow-x: auto;
  border: 1px solid var(--color-border);
  border-radius: 6px;
}

.wf-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
  min-width: 900px;
}

.wf-table th {
  background: #f5f5f5;
  font-weight: 600;
  text-align: left;
  padding: 6px 4px;
  border-bottom: 2px solid var(--color-border);
  white-space: nowrap;
  font-size: 12px;
}

.wf-table td {
  padding: 2px 2px;
  border-bottom: 1px solid #eee;
  vertical-align: middle;
}

.wf-table tr.selected {
  background: #e3f2fd;
}

.wf-table tr:hover:not(.selected) {
  background: #fafafa;
}

.empty-msg {
  text-align: center;
  color: var(--color-text-secondary);
  font-style: italic;
  padding: 24px !important;
}

/* Column widths */
.col-sel { width: 30px; text-align: center; }
.col-order { width: 65px; }
.col-propcode { width: 110px; }
.col-vstate { width: 110px; }
.col-num { width: 90px; }
.col-text { width: 170px; }
.col-amttype { width: 80px; }
.col-notes { width: 140px; }

/* Inputs inside cells */
.wf-table input[type="text"],
.wf-table input[type="number"],
.wf-table select {
  width: 100%;
  box-sizing: border-box;
  border: 1px solid transparent;
  background: transparent;
  padding: 4px 6px;
  font-size: 13px;
  font-family: inherit;
  border-radius: 3px;
}

.wf-table input[type="text"]:focus,
.wf-table input[type="number"]:focus,
.wf-table select:focus {
  border-color: var(--color-accent, #4a7cc9);
  outline: none;
  background: white;
}

.wf-table input[type="text"]:hover,
.wf-table input[type="number"]:hover,
.wf-table select:hover {
  border-color: #ccc;
}

.wf-table input[type="number"] {
  text-align: right;
  -moz-appearance: textfield;
}

.wf-table input[type="number"]::-webkit-inner-spin-button,
.wf-table input[type="number"]::-webkit-outer-spin-button {
  -webkit-appearance: none;
  margin: 0;
}

.wf-table input[type="checkbox"] {
  cursor: pointer;
}

.wf-table select {
  cursor: pointer;
}

/* Buttons */
.btn {
  padding: 4px 10px;
  border: 1px solid var(--color-border);
  background: var(--color-surface);
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
  white-space: nowrap;
}

.btn:hover:not(:disabled) {
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
