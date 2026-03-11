<script setup lang="ts">
import { onMounted, ref, computed, watch } from 'vue'
import { useWaterfallStore } from '../stores/waterfall'
import type { WaterfallStep } from '../stores/waterfall'
import WaterfallEditor from '../components/waterfall/WaterfallEditor.vue'
import api from '../api/client'

const wf = useWaterfallStore()

const selectedEntity = ref('')
const activeTab = ref<'CF_WF' | 'Cap_WF'>('CF_WF')
const showGuidance = ref(false)
const copySourceVcode = ref('')
const statusMessage = ref<{ type: 'success' | 'error' | 'info'; text: string } | null>(null)

// Local draft steps (decoupled from store to avoid feedback loops)
const cfDraft = ref<WaterfallStep[]>([])
const capDraft = ref<WaterfallStep[]>([])

onMounted(async () => {
  await wf.loadEntities()
})

// Sync store steps to drafts when entity loads
watch(() => [wf.cfSteps, wf.capSteps], () => {
  cfDraft.value = wf.cfSteps.map((s) => ({ ...s }))
  capDraft.value = wf.capSteps.map((s) => ({ ...s }))
}, { deep: true })

const currentDraft = computed(() => activeTab.value === 'CF_WF' ? cfDraft.value : capDraft.value)
const hasAnySteps = computed(() => wf.hasWaterfall || cfDraft.value.length > 0 || capDraft.value.length > 0)

const entitiesWithWf = computed(() => wf.entities.filter((e) => e.has_wf))

// ── Entity Selection ─────────────────────────────────────────────────────

async function onEntitySelect(event: Event) {
  const vcode = (event.target as HTMLSelectElement).value
  selectedEntity.value = vcode
  clearStatus()
  if (vcode) {
    await wf.loadSteps(vcode)
  }
}

// ── Draft Updates ────────────────────────────────────────────────────────

function handleCfUpdate(steps: WaterfallStep[]) {
  cfDraft.value = steps
  wf.markDirty('CF_WF')
}

function handleCapUpdate(steps: WaterfallStep[]) {
  capDraft.value = steps
  wf.markDirty('Cap_WF')
}

// ── Save ─────────────────────────────────────────────────────────────────

async function handleSave(wfType: string) {
  if (!selectedEntity.value) return
  clearStatus()

  const steps = wfType === 'CF_WF' ? cfDraft.value : capDraft.value

  // Validate first
  const val = await wf.validateSteps(steps, wfType)
  if (val.errors.length > 0) {
    setStatus('error', 'Fix validation errors before saving.')
    return
  }

  const result = await wf.saveSteps(selectedEntity.value, wfType, steps)
  if (result?.success) {
    setStatus('success', `Saved ${steps.length} ${wfType} steps for ${selectedEntity.value}.`)
  } else {
    setStatus('error', result?.message || 'Save failed.')
  }
}

// ── Validate ─────────────────────────────────────────────────────────────

async function handleValidate(wfType: string) {
  clearStatus()
  const steps = wfType === 'CF_WF' ? cfDraft.value : capDraft.value
  const res = await wf.validateSteps(steps, wfType)
  if (res.errors.length === 0 && res.warnings.length === 0) {
    setStatus('success', 'All validation checks passed.')
  }
}

// ── Preview ──────────────────────────────────────────────────────────────

async function handlePreview(wfType: string) {
  if (!selectedEntity.value) return
  clearStatus()
  const steps = wfType === 'CF_WF' ? cfDraft.value : capDraft.value
  await wf.previewWaterfall(selectedEntity.value, wfType, steps)
}

// ── Copy CF -> Cap ───────────────────────────────────────────────────────

function handleCopyCfToCap() {
  wf.copyCfToCap(selectedEntity.value, cfDraft.value)
  capDraft.value = cfDraft.value.map((s) => ({ ...s }))
  activeTab.value = 'Cap_WF'
  setStatus('info', 'Copied CF_WF steps to Cap_WF draft.')
}

// ── Reset ────────────────────────────────────────────────────────────────

async function handleReset() {
  if (!selectedEntity.value) return
  clearStatus()
  await wf.resetSteps(selectedEntity.value)
  setStatus('info', 'Reset to saved state.')
}

// ── Export CSV ───────────────────────────────────────────────────────────

function handleExportCsv() {
  if (!selectedEntity.value) return
  window.open(`/api/waterfall-setup/${selectedEntity.value}/export-csv?token=${localStorage.getItem('token')}`, '_blank')
}

// ── New Waterfall Creation ───────────────────────────────────────────────

async function createNew(template: 'new' | 'pari-passu') {
  if (!selectedEntity.value) return
  clearStatus()
  await wf.createFromTemplate(selectedEntity.value, template)
  setStatus('info', `Created ${template === 'new' ? 'standard' : 'pari passu'} waterfall template. Review and save.`)
}

async function copyFromDeal() {
  if (!copySourceVcode.value) return
  clearStatus()
  await wf.copyFromEntity(copySourceVcode.value)
  setStatus('info', `Copied waterfall from ${copySourceVcode.value}. Review and save.`)
}

// ── Status Messages ──────────────────────────────────────────────────────

function setStatus(type: 'success' | 'error' | 'info', text: string) {
  statusMessage.value = { type, text }
}

function clearStatus() {
  statusMessage.value = null
  wf.validation.errors = []
  wf.validation.warnings = []
  wf.previewResult = null
  wf.previewError = ''
}

// ── Formatters ───────────────────────────────────────────────────────────

function fmtCur(v: number): string {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2 }).format(v)
}

function fmtPct(v: number): string {
  return (v * 100).toFixed(2) + '%'
}
</script>

<template>
  <div class="waterfall-setup">
    <h2>Waterfall Setup</h2>

    <!-- Entity Selector -->
    <div class="entity-selector">
      <div class="selector-row">
        <label>Entity:</label>
        <select @change="onEntitySelect" :value="selectedEntity" class="entity-select">
          <option value="">-- Select entity --</option>
          <option v-for="e in wf.entities" :key="e.vcode" :value="e.vcode">
            {{ e.label }}
          </option>
        </select>
      </div>

      <!-- Investor List -->
      <div v-if="selectedEntity && wf.investors.length > 0" class="investor-list">
        <details>
          <summary>Investors ({{ wf.investors.length }})</summary>
          <div class="investor-items">
            <div v-for="inv in wf.investors" :key="inv.investor_id" class="investor-item">
              <span class="inv-id">{{ inv.investor_id }}</span>
              <span class="inv-pct">{{ fmtPct(inv.ownership_pct) }}</span>
            </div>
          </div>
        </details>
      </div>
    </div>

    <!-- Loading -->
    <div v-if="wf.loading" class="loading">Loading waterfall steps...</div>

    <!-- Status Messages -->
    <div v-if="statusMessage" :class="['status-msg', statusMessage.type]">
      {{ statusMessage.text }}
    </div>

    <!-- Validation Messages -->
    <div v-if="wf.validation.errors.length" class="validation-box errors">
      <strong>Errors:</strong>
      <p v-for="(err, i) in wf.validation.errors" :key="i">{{ err }}</p>
    </div>
    <div v-if="wf.validation.warnings.length" class="validation-box warnings">
      <strong>Warnings:</strong>
      <p v-for="(warn, i) in wf.validation.warnings" :key="i">{{ warn }}</p>
    </div>

    <!-- No Waterfall: Create Options -->
    <template v-if="selectedEntity && !wf.loading && !hasAnySteps">
      <div class="no-waterfall">
        <p class="placeholder-text">No waterfall defined for this entity.</p>
        <div class="create-actions">
          <button class="btn" @click="createNew('new')">Create New Waterfall</button>
          <button class="btn" @click="createNew('pari-passu')">Create Pari Passu</button>
        </div>
        <div class="copy-from-section">
          <label>Copy from existing deal:</label>
          <select v-model="copySourceVcode" class="copy-select">
            <option value="">-- Select --</option>
            <option v-for="e in entitiesWithWf" :key="e.vcode" :value="e.vcode">
              {{ e.label }}
            </option>
          </select>
          <button class="btn" @click="copyFromDeal" :disabled="!copySourceVcode">Copy</button>
        </div>
      </div>
    </template>

    <!-- Waterfall Editors -->
    <template v-if="selectedEntity && !wf.loading && hasAnySteps">
      <!-- Tab Switcher -->
      <div class="tab-bar">
        <button
          :class="['tab-btn', { active: activeTab === 'CF_WF' }]"
          @click="activeTab = 'CF_WF'"
        >
          CF_WF
          <span v-if="wf.cfDirty" class="tab-dirty">*</span>
        </button>
        <button
          :class="['tab-btn', { active: activeTab === 'Cap_WF' }]"
          @click="activeTab = 'Cap_WF'"
        >
          Cap_WF
          <span v-if="wf.capDirty" class="tab-dirty">*</span>
        </button>
      </div>

      <!-- CF_WF Editor -->
      <div v-show="activeTab === 'CF_WF'">
        <WaterfallEditor
          :steps="cfDraft"
          wf-type="CF_WF"
          :vstate-options="wf.vstateOptions"
          :saving="wf.saving"
          :dirty="wf.cfDirty"
          @update="handleCfUpdate"
          @save="handleSave('CF_WF')"
          @validate="handleValidate('CF_WF')"
          @preview="handlePreview('CF_WF')"
          @reset="handleReset"
          @copy-cf-to-cap="handleCopyCfToCap"
          @export-csv="handleExportCsv"
        />
      </div>

      <!-- Cap_WF Editor -->
      <div v-show="activeTab === 'Cap_WF'">
        <WaterfallEditor
          :steps="capDraft"
          wf-type="Cap_WF"
          :vstate-options="wf.vstateOptions"
          :saving="wf.saving"
          :dirty="wf.capDirty"
          @update="handleCapUpdate"
          @save="handleSave('Cap_WF')"
          @validate="handleValidate('Cap_WF')"
          @preview="handlePreview('Cap_WF')"
          @reset="handleReset"
          @export-csv="handleExportCsv"
        />
      </div>

      <!-- Preview Results -->
      <div v-if="wf.previewing" class="preview-section">
        <p class="loading">Running preview...</p>
      </div>
      <div v-if="wf.previewResult && wf.previewResult.length > 0" class="preview-section">
        <h4>Preview: $100,000 Allocation</h4>
        <table class="preview-table">
          <thead>
            <tr><th>PropCode</th><th>vState</th><th>Allocated</th></tr>
          </thead>
          <tbody>
            <tr v-for="(row, i) in wf.previewResult" :key="i">
              <td>{{ row.PropCode }}</td>
              <td>{{ row.vState }}</td>
              <td class="num">{{ fmtCur(row.Allocated) }}</td>
            </tr>
            <tr class="total-row">
              <td colspan="2"><strong>Total</strong></td>
              <td class="num"><strong>{{ fmtCur(wf.previewTotal) }}</strong></td>
            </tr>
          </tbody>
        </table>
      </div>
      <div v-if="wf.previewError" class="validation-box errors">
        <strong>Preview Error:</strong> {{ wf.previewError }}
      </div>
    </template>

    <!-- Placeholder -->
    <p v-if="!selectedEntity" class="placeholder-text">
      Select an entity to view or edit its waterfall structure.
    </p>

    <!-- Guidance Panel -->
    <div class="guidance-section">
      <button class="btn btn-guidance" @click="showGuidance = !showGuidance">
        {{ showGuidance ? 'Hide' : 'Show' }} Waterfall Setup Guide
      </button>
      <div v-if="showGuidance" class="guidance-panel">
        <h3>vState Reference</h3>
        <table class="guide-table">
          <thead><tr><th>vState</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td><strong>Pref</strong></td><td>Pay accrued preferred return. <code>nPercent</code> = annual rate. Accrues daily Act/365.</td></tr>
            <tr><td><strong>Initial</strong></td><td>Return initial capital. Cap_WF reduces capital; CF_WF skips.</td></tr>
            <tr><td><strong>Add</strong></td><td>Route to capital pool per <code>vtranstype</code>. Independent cap/tracking.</td></tr>
            <tr><td><strong>Tag</strong></td><td>Proportional follower of lead step at same iOrder. No pool routing.</td></tr>
            <tr><td><strong>Share</strong></td><td>Residual distribution. <code>FXRate</code> = sharing percentage.</td></tr>
            <tr><td><strong>IRR</strong></td><td>IRR-targeted distribution (hurdle gate). <code>nPercent</code> = target IRR.</td></tr>
            <tr><td><strong>Amt</strong></td><td>Fixed-amount distribution. <code>mAmount</code> = dollar amount per period.</td></tr>
            <tr><td><strong>Def&amp;Int</strong></td><td>Default interest + principal. CF_WF: interest only; Cap_WF: both.</td></tr>
            <tr><td><strong>Def_Int</strong></td><td>Default interest only. Does not reduce capital.</td></tr>
            <tr><td><strong>Default</strong></td><td>Default principal return. Reduces capital_outstanding.</td></tr>
            <tr><td><strong>AMFee</strong></td><td>Post-distribution AM fee. Deducts from source (vNotes), pays to PropCode. Pool-neutral.</td></tr>
            <tr><td><strong>Promote</strong></td><td>Cumulative catch-up. <code>FXRate</code> = carry share, <code>nPercent</code> = target carry %.</td></tr>
          </tbody>
        </table>

        <div class="guide-warning">
          <strong>ADD vs TAG Rule:</strong> If an investor contributed capital to a pool that needs tracked and returned,
          that investor MUST have an <strong>Add</strong> step, NEVER a Tag.
        </div>
        <div class="guide-error">
          <strong>Operating Capital Rule:</strong> Operating capital MUST always use <strong>Add</strong> for each investor.
          Tag steps bypass pool routing — capital_outstanding is never reduced and cumulative caps are never enforced.
        </div>

        <h3>Pool Routing Table</h3>
        <table class="guide-table">
          <thead><tr><th>vtranstype contains</th><th>Pool</th><th>CF_WF</th><th>Cap_WF</th></tr></thead>
          <tbody>
            <tr><td>Operating Capital</td><td>operating</td><td>pay_capital_capped</td><td>pay_capital_capped</td></tr>
            <tr><td>Cost Overrun + Pref</td><td>cost_overrun</td><td>pay_pref</td><td>pay_pref</td></tr>
            <tr><td>Cost Overrun</td><td>cost_overrun</td><td>skip</td><td>pay_capital</td></tr>
            <tr><td>Special Capital + Pref</td><td>special</td><td>pay_pref</td><td>pay_pref</td></tr>
            <tr><td>Special Capital</td><td>special</td><td>skip</td><td>pay_capital</td></tr>
            <tr><td>Pref (other)</td><td>additional</td><td>pay_pref</td><td>pay_pref</td></tr>
            <tr><td>(default)</td><td>additional</td><td>skip</td><td>pay_capital</td></tr>
          </tbody>
        </table>

        <h3>Common Patterns</h3>
        <div class="guide-info">
          <p><strong>Pattern A — Simple two-partner deal:</strong><br>
          iOrder 1: Lead + Tag for Default/Def&amp;Int (prorata)<br>
          iOrder 3-4: Separate Pref steps per investor (FX=1.0 each)<br>
          iOrder 5: Share (lead) + Tag (partner) for residual</p>
          <p><strong>Pattern B — Operating capital deal:</strong><br>
          iOrder 1: Default step(s)<br>
          iOrder 2: Add + Add for operating capital (separate mAmount caps)<br>
          iOrder 3-4: Pref steps per investor<br>
          iOrder 5: Share + Tag for residual</p>
        </div>

        <h3>Modeling Checklist</h3>
        <ol class="guide-checklist">
          <li>Identify all capital pools (initial, operating, additional, special, cost overrun, default)</li>
          <li>For each pool with multiple investors: independent cap -> separate Add; shared prorata -> Add + Tag</li>
          <li>Operating capital MUST always use Add (never Tag) for each investor</li>
          <li>Set <code>mAmount</code> correctly for operating capital = cumulative cap</li>
          <li>Verify FXRates at each iOrder sum to ~1.0</li>
          <li>Pref steps: own iOrder per investor, FX=1.0</li>
          <li>Residual sharing: lead = Share, partner = Tag</li>
          <li>Set up both CF_WF and Cap_WF</li>
          <li>Verify <code>vtranstype</code> text matches routing rules</li>
        </ol>
      </div>
    </div>
  </div>
</template>

<style scoped>
.waterfall-setup {
  padding: 0 0 40px 0;
}

h2 {
  font-size: 20px;
  margin-bottom: 16px;
}

/* Entity selector */
.entity-selector {
  margin-bottom: 16px;
}

.selector-row {
  display: flex;
  align-items: center;
  gap: 12px;
}

.selector-row label {
  font-weight: 600;
  font-size: 14px;
}

.entity-select {
  padding: 8px 12px;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  font-size: 14px;
  min-width: 450px;
  background: var(--color-surface);
}

.investor-list {
  margin-top: 8px;
}

.investor-list details {
  font-size: 13px;
}

.investor-list summary {
  cursor: pointer;
  color: var(--color-text-secondary);
  font-weight: 500;
}

.investor-items {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 6px;
}

.investor-item {
  display: flex;
  gap: 6px;
  padding: 3px 8px;
  background: #f5f5f5;
  border-radius: 4px;
  font-size: 12px;
}

.inv-id { font-weight: 600; }
.inv-pct { color: var(--color-text-secondary); }

/* Tabs */
.tab-bar {
  display: flex;
  gap: 0;
  margin-bottom: 12px;
  border-bottom: 2px solid var(--color-border);
}

.tab-btn {
  padding: 8px 20px;
  border: none;
  background: none;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  color: var(--color-text-secondary);
  border-bottom: 2px solid transparent;
  margin-bottom: -2px;
}

.tab-btn.active {
  color: var(--color-accent);
  border-bottom-color: var(--color-accent);
}

.tab-btn:hover:not(.active) {
  color: var(--color-text);
}

.tab-dirty {
  color: #c00;
  font-weight: 700;
}

/* Status & Validation */
.status-msg {
  padding: 8px 14px;
  border-radius: 6px;
  margin-bottom: 12px;
  font-size: 13px;
}

.status-msg.success { background: #e8f5e9; border: 1px solid #a5d6a7; color: #2e7d32; }
.status-msg.error { background: #fee; border: 1px solid #fcc; color: #c00; }
.status-msg.info { background: #e3f2fd; border: 1px solid #90caf9; color: #1565c0; }

.validation-box {
  padding: 8px 14px;
  border-radius: 6px;
  margin-bottom: 12px;
  font-size: 13px;
}

.validation-box p { margin: 2px 0; }

.validation-box.errors { background: #fee; border: 1px solid #fcc; color: #c00; }
.validation-box.warnings { background: #fff8e1; border: 1px solid #ffe082; color: #856404; }

/* No Waterfall */
.no-waterfall {
  padding: 24px;
  text-align: center;
  background: #fafafa;
  border: 1px dashed var(--color-border);
  border-radius: 8px;
  margin-bottom: 16px;
}

.create-actions {
  display: flex;
  gap: 12px;
  justify-content: center;
  margin: 16px 0;
}

.copy-from-section {
  display: flex;
  align-items: center;
  gap: 8px;
  justify-content: center;
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid var(--color-border);
}

.copy-from-section label {
  font-size: 13px;
  color: var(--color-text-secondary);
}

.copy-select {
  padding: 6px 10px;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  font-size: 13px;
  min-width: 300px;
}

/* Preview */
.preview-section {
  margin-bottom: 16px;
  padding: 12px 16px;
  background: #f8f9fa;
  border: 1px solid var(--color-border);
  border-radius: 6px;
}

.preview-section h4 {
  margin: 0 0 8px 0;
  font-size: 14px;
}

.preview-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

.preview-table th, .preview-table td {
  padding: 4px 10px;
  text-align: left;
  border-bottom: 1px solid #e0e0e0;
}

.preview-table .num { text-align: right; font-variant-numeric: tabular-nums; }
.preview-table .total-row { border-top: 2px solid #333; background: #f0f0f0; }

/* Guidance */
.guidance-section {
  margin-top: 24px;
}

.btn-guidance {
  padding: 6px 14px;
  border: 1px solid var(--color-border);
  background: var(--color-surface);
  border-radius: 4px;
  cursor: pointer;
  font-size: 13px;
}

.btn-guidance:hover { background: #eee; }

.guidance-panel {
  margin-top: 12px;
  padding: 16px 20px;
  background: #fafafa;
  border: 1px solid var(--color-border);
  border-radius: 8px;
  font-size: 13px;
  line-height: 1.5;
}

.guidance-panel h3 {
  font-size: 15px;
  margin: 16px 0 8px 0;
}

.guidance-panel h3:first-child {
  margin-top: 0;
}

.guide-table {
  width: 100%;
  border-collapse: collapse;
  margin-bottom: 12px;
}

.guide-table th, .guide-table td {
  padding: 4px 10px;
  text-align: left;
  border-bottom: 1px solid #e0e0e0;
}

.guide-table th {
  background: #eeeeee;
  font-weight: 600;
}

.guide-table code {
  background: #e3e3e3;
  padding: 1px 4px;
  border-radius: 3px;
  font-size: 12px;
}

.guide-warning {
  padding: 8px 12px;
  background: #fff8e1;
  border: 1px solid #ffe082;
  border-radius: 4px;
  color: #856404;
  margin: 8px 0;
}

.guide-error {
  padding: 8px 12px;
  background: #fee;
  border: 1px solid #fcc;
  border-radius: 4px;
  color: #c00;
  margin: 8px 0;
}

.guide-info {
  padding: 8px 12px;
  background: #e3f2fd;
  border: 1px solid #90caf9;
  border-radius: 4px;
  color: #1565c0;
  margin: 8px 0;
}

.guide-info p { margin: 4px 0; }

.guide-checklist {
  padding-left: 20px;
  margin: 8px 0;
}

.guide-checklist li {
  margin: 4px 0;
}

.guide-checklist code {
  background: #e3e3e3;
  padding: 1px 4px;
  border-radius: 3px;
  font-size: 12px;
}

/* Common */
.placeholder-text {
  color: var(--color-text-secondary);
  font-style: italic;
  text-align: center;
  padding: 32px 0;
}

.loading {
  color: var(--color-text-secondary);
  text-align: center;
  padding: 20px 0;
}

.btn {
  padding: 6px 14px;
  border: 1px solid var(--color-border);
  background: var(--color-surface);
  border-radius: 4px;
  cursor: pointer;
  font-size: 13px;
}

.btn:hover { background: #eee; }
.btn:disabled { opacity: 0.5; cursor: not-allowed; }
</style>
