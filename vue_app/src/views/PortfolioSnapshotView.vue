<script setup lang="ts">
/**
 * Portfolio Snapshot — shell.
 *
 * Header controls (investor + quarter + refresh) and an inline review status
 * strip, then a horizontal subtab bar over four presentational bodies. One
 * `GET /bundle` per (investor, quarter) builds all four subtabs server-side, so
 * switching tabs is free and the dropdowns persist across them.
 *
 * All persistence lives here: the four components are props-in and emit save
 * events, so none of them talks to the API directly.
 */
import { ref, computed, onMounted, watch } from 'vue'
import api from '../api/client'
import SnapshotSummary from '../components/snapshot/SnapshotSummary.vue'
import SnapshotFinancial from '../components/snapshot/SnapshotFinancial.vue'
import SnapshotOperating from '../components/snapshot/SnapshotOperating.vue'
import SnapshotLoan from '../components/snapshot/SnapshotLoan.vue'

const BASE = '/api/portfolio-snapshot'

type TabKey = 'summary' | 'financial' | 'operating' | 'loan'

const TABS: { key: TabKey; label: string }[] = [
  { key: 'summary', label: 'Summary' },
  { key: 'financial', label: 'Financial' },
  { key: 'operating', label: 'Operating' },
  { key: 'loan', label: 'Loan' },
]

const activeTab = ref<TabKey>('summary')

const investors = ref<{ code: string; name: string }[]>([])
const quarters = ref<string[]>([])
const selectedInvestor = ref('')
const selectedQuarter = ref('')

const bundle = ref<any>(null)
const loading = ref(false)
const loadError = ref('')

const review = ref<any>(null)
const savingCount = ref(0)
const savedFlash = ref('')
const saveError = ref('')

const editable = computed(() => review.value?.editable !== false)
const subtabs = computed(() => bundle.value?.subtabs || {})
const subtabErrors = computed(() => bundle.value?.errors || {})
const resolution = computed(() => bundle.value?.resolution || null)
const saving = computed(() => savingCount.value > 0)

const canLoad = computed(() => !!selectedInvestor.value && !!selectedQuarter.value)

onMounted(async () => {
  try {
    const [inv, qs] = await Promise.all([
      api.get(`${BASE}/investors`),
      api.get(`${BASE}/quarters`),
    ])
    investors.value = inv.data.investors || []
    quarters.value = qs.data.quarters || []
    if (!selectedQuarter.value) selectedQuarter.value = qs.data.default || ''
  } catch (e: any) {
    loadError.value = e?.response?.data?.error || 'Could not load selectors'
  }
})

watch([selectedInvestor, selectedQuarter], () => {
  if (canLoad.value) load()
})

async function load() {
  if (!canLoad.value) return
  loading.value = true
  loadError.value = ''
  saveError.value = ''
  try {
    const res = await api.get(`${BASE}/bundle`, {
      params: { investor: selectedInvestor.value, quarter: selectedQuarter.value },
    })
    bundle.value = res.data
    review.value = res.data.review || null
  } catch (e: any) {
    bundle.value = null
    loadError.value = e?.response?.data?.error || 'Failed to load snapshot'
  } finally {
    loading.value = false
  }
}

function flash(msg: string) {
  savedFlash.value = msg
  window.setTimeout(() => { if (savedFlash.value === msg) savedFlash.value = '' }, 2000)
}

/** Wrap a save so the strip shows progress and 409-locked reads clearly. */
async function withSave(fn: () => Promise<any>, okMsg = 'Saved') {
  savingCount.value++
  saveError.value = ''
  try {
    await fn()
    flash(okMsg)
  } catch (e: any) {
    const d = e?.response?.data
    saveError.value = d?.locked
      ? 'This snapshot is approved and locked — edits are no longer accepted.'
      : (d?.error || 'Save failed')
    if (d?.locked) await refreshReview()
  } finally {
    savingCount.value--
  }
}

const ctx = () => ({ investor: selectedInvestor.value, quarter: selectedQuarter.value })

function onSaveComment(p: { scope: string; field: string; scope_key?: string; text: string }) {
  withSave(async () => {
    await api.put(`${BASE}/comment`, { ...ctx(), ...p })
  })
}

function onSaveValue(p: { vcode: string; field: string; value: string | number | null }) {
  withSave(async () => {
    await api.put(`${BASE}/value`, { ...ctx(), ...p })
    // Manual figures feed the Financial rows, so re-read to pick up the
    // backend's own display/source strings rather than guessing them here.
    await load()
  })
}

function onAddFootnote(p: { anchor: string; text: string }) {
  withSave(async () => {
    const res = await api.post(`${BASE}/footnote`, { ...ctx(), ...p })
    if (bundle.value?.subtabs?.financial) {
      bundle.value.subtabs.financial.footnotes = res.data.footnotes || []
    }
  }, 'Footnote added')
}

function onRemoveFootnote(id: number) {
  withSave(async () => {
    const res = await api.delete(`${BASE}/footnote/${id}`, { params: ctx() })
    if (bundle.value?.subtabs?.financial) {
      bundle.value.subtabs.financial.footnotes = res.data.footnotes || []
    }
  }, 'Footnote removed')
}

async function refreshReview() {
  try {
    const res = await api.get(`${BASE}/elements`, { params: ctx() })
    review.value = res.data.review || null
  } catch { /* leave the previous status in place */ }
}

const returnNote = ref('')
const showReturn = ref(false)
const reopenNote = ref('')
const showReopen = ref(false)

type Action = 'submit' | 'approve' | 'return' | 'reopen'

const ACTION_LABEL: Record<Action, string> = {
  submit: 'Submitted', approve: 'Approved',
  return: 'Returned', reopen: 'Reopened',
}

async function transition(action: Action) {
  const body: any = { ...ctx() }
  // Both backward actions require a note, matching the backend's own rule.
  if (action === 'return' || action === 'reopen') {
    const note = action === 'return' ? returnNote.value : reopenNote.value
    if (!note.trim()) {
      saveError.value = `A note is required to ${action} the snapshot.`
      return
    }
    body.note = note
  }
  await withSave(async () => {
    const res = await api.post(`${BASE}/${action}`, body)
    review.value = res.data.review || null
    showReturn.value = false
    showReopen.value = false
    returnNote.value = ''
    reopenNote.value = ''
    // Reload: reopening an approved report switches the payload from frozen
    // back to live, so the body must be refetched, not just the status.
    await load()
  }, ACTION_LABEL[action])
}

// --- frozen vs live ---
const isFrozen = computed(() => bundle.value?.source === 'frozen')
const sourceNote = computed(() => bundle.value?.source_note || '')

const approvedAsOf = computed(() => {
  const raw = bundle.value?.approved_at
  if (!raw) return ''
  const m = String(raw).match(/^(\d{4})-(\d{2})-(\d{2})/)
  return m ? `${parseInt(m[2])}/${parseInt(m[3])}/${m[1]}` : String(raw).slice(0, 10)
})

const statusColor = computed(() => {
  switch (review.value?.status) {
    case 'draft': return '#666'
    case 'returned': return '#e65100'
    case 'approved': return '#2e7d32'
    default: return '#1565c0'
  }
})
</script>

<template>
  <div class="snapshot">
    <div class="snap-header">
      <h2>Portfolio Snapshot</h2>

      <div class="snap-controls">
        <div class="ctl">
          <label>Investor</label>
          <select v-model="selectedInvestor">
            <option value="">-- Select investor --</option>
            <option v-for="i in investors" :key="i.code" :value="i.code">
              {{ i.name || i.code }}
            </option>
          </select>
        </div>
        <div class="ctl">
          <label>Quarter</label>
          <select v-model="selectedQuarter">
            <option v-for="q in quarters" :key="q" :value="q">{{ q }}</option>
          </select>
        </div>
        <button class="btn-refresh" :disabled="!canLoad || loading" @click="load">
          {{ loading ? 'Loading…' : 'Refresh' }}
        </button>
        <span v-if="saving" class="save-note">Saving…</span>
        <span v-else-if="savedFlash" class="save-note ok">{{ savedFlash }}</span>
      </div>
    </div>

    <!-- Review status strip -->
    <div v-if="review && !review.error" class="review-strip">
      <span class="dot" :style="{ background: statusColor }"></span>
      <strong>{{ review.label || review.status }}</strong>
      <span v-if="review.approver" class="review-meta">
        — {{ review.approver }}<span v-if="review.approved_at">, {{ String(review.approved_at).slice(0, 10) }}</span>
      </span>
      <span v-if="!editable" class="locked-badge">locked</span>
      <span v-if="review.mixed" class="warn-badge" title="Elements disagree on status; the least advanced wins">mixed status</span>

      <span class="strip-spacer"></span>

      <button v-if="review.can_submit" class="btn-sm primary" @click="transition('submit')">Submit for review</button>
      <button v-if="review.can_approve" class="btn-sm primary" @click="transition('approve')">Approve</button>
      <button v-if="review.can_return" class="btn-sm" @click="showReturn = !showReturn">Return</button>
      <button v-if="review.can_reopen" class="btn-sm warn" @click="showReopen = !showReopen"
              :title="`Unwind the approval so the report can be corrected (${(review.reopen_roles || []).join(', ')})`">
        Reopen
      </button>
      <span v-if="!review.can_submit && !review.can_approve && !review.can_return && !review.can_reopen"
            class="review-meta">no action available for your role</span>
    </div>

    <div v-if="showReturn" class="return-form">
      <input v-model="returnNote" placeholder="Reason for returning (required)" />
      <button class="btn-sm" @click="transition('return')">Confirm return</button>
      <button class="btn-sm" @click="showReturn = false">Cancel</button>
    </div>

    <div v-if="showReopen" class="return-form">
      <input v-model="reopenNote" placeholder="Reason for reopening this approved report (required)" />
      <button class="btn-sm warn" @click="transition('reopen')">Confirm reopen</button>
      <button class="btn-sm" @click="showReopen = false">Cancel</button>
    </div>

    <p v-if="saveError" class="banner err">{{ saveError }}</p>
    <p v-if="loadError" class="banner err">{{ loadError }}</p>

    <!-- Frozen vs live. An approved report serves the payload frozen at
         approval, so the reader must never be left guessing which they have. -->
    <div v-if="bundle && isFrozen" class="banner frozen">
      <strong>Approved version{{ approvedAsOf ? ` — as of ${approvedAsOf}` : '' }}</strong>
      <span>
        Frozen at approval and not recomputed, so it cannot shift if MRI data
        changes.
        <template v-if="bundle.approved_by">Approved by {{ bundle.approved_by }}.</template>
      </span>
      <span v-if="bundle.data_version" class="banner-meta">{{ bundle.data_version }}</span>
    </div>
    <div v-else-if="bundle" class="banner live">
      <strong>Live data</strong>
      <span>{{ sourceNote || 'In progress — computed from current data and will change as data changes.' }}</span>
    </div>

    <!-- Population diagnostics -->
    <details v-if="resolution" class="diag">
      <summary>
        {{ resolution.diagnostics?.deal_count }} deals in
        {{ resolution.diagnostics?.group_count }} groups
        <template v-if="resolution.flagged?.length">
          · {{ resolution.flagged.length }} ownership-flagged
        </template>
        <template v-if="resolution.excluded_not_acquired?.length">
          · {{ resolution.excluded_not_acquired.length }} not yet acquired
        </template>
        <template v-if="resolution.excluded_sold?.length">
          · {{ resolution.excluded_sold.length }} sold
        </template>
      </summary>
      <div class="diag-body">
        <div v-if="resolution.flagged?.length">
          <strong>Ownership % unavailable</strong>
          <p v-for="f in resolution.flagged" :key="f.vcode">{{ f.vcode }} {{ f.name }} — {{ f.detail }}</p>
        </div>
        <div v-if="resolution.excluded_not_acquired?.length">
          <strong>Not yet acquired at quarter end</strong>
          <p v-for="d in resolution.excluded_not_acquired" :key="d.vcode">
            {{ d.vcode }} {{ d.name }} — acquired {{ d.acquisition_date }}
          </p>
        </div>
        <div v-if="resolution.excluded_sold?.length">
          <strong>Sold on or before quarter end</strong>
          <p v-for="d in resolution.excluded_sold" :key="d.vcode">
            {{ d.vcode }} {{ d.name }} — sold {{ d.sale_date }}
          </p>
        </div>
      </div>
    </details>

    <!-- Subtab bar -->
    <div class="tabbar">
      <button
        v-for="t in TABS"
        :key="t.key"
        :class="['tab', { active: activeTab === t.key }]"
        @click="activeTab = t.key"
      >
        {{ t.label }}
        <span v-if="subtabErrors[t.key]" class="tab-err" title="This subtab failed to build">!</span>
      </button>
    </div>

    <div class="tabbody">
      <p v-if="!canLoad" class="placeholder">Select an investor and quarter to build the snapshot.</p>
      <p v-else-if="loading" class="placeholder">Building snapshot…</p>
      <template v-else-if="bundle">
        <p v-if="subtabErrors[activeTab]" class="banner err">
          {{ TABS.find(t => t.key === activeTab)?.label }} failed: {{ subtabErrors[activeTab] }}
        </p>
        <template v-else>
          <SnapshotSummary
            v-if="activeTab === 'summary'"
            :data="subtabs.summary"
            :editable="editable"
            @save-comment="onSaveComment"
          />
          <SnapshotFinancial
            v-else-if="activeTab === 'financial'"
            :data="subtabs.financial"
            :editable="editable"
            @save-value="onSaveValue"
            @add-footnote="onAddFootnote"
            @remove-footnote="onRemoveFootnote"
          />
          <SnapshotOperating
            v-else-if="activeTab === 'operating'"
            :data="subtabs.operating"
            :editable="editable"
            @save-comment="onSaveComment"
          />
          <SnapshotLoan
            v-else-if="activeTab === 'loan'"
            :data="subtabs.loan"
            :editable="editable"
            @save-comment="onSaveComment"
          />
        </template>
      </template>
    </div>
  </div>
</template>

<style scoped>
.snapshot { padding: 0 0 40px 0; }
h2 { font-size: 20px; margin: 0 0 12px 0; }

.snap-controls {
  display: flex;
  align-items: flex-end;
  gap: 14px;
  flex-wrap: wrap;
  margin-bottom: 12px;
}

.ctl label {
  display: block;
  font-size: 12px;
  font-weight: 600;
  margin-bottom: 3px;
}

.ctl select {
  padding: 7px 10px;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  font-size: 13px;
  min-width: 200px;
  box-sizing: border-box;
}

.btn-refresh {
  padding: 8px 18px;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
  font-weight: 600;
  background: var(--color-accent);
  color: white;
}
.btn-refresh:hover:not(:disabled) { background: #3a63ad; }
.btn-refresh:disabled { opacity: 0.6; cursor: not-allowed; }

.save-note { font-size: 12px; color: var(--color-text-secondary); font-style: italic; }
.save-note.ok { color: #2e7d32; font-style: normal; font-weight: 600; }

/* --- review strip --- */
.review-strip {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 14px;
  background: #f8f9fa;
  border: 1px solid var(--color-border);
  border-radius: 8px;
  font-size: 13px;
  margin-bottom: 10px;
}
.dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
.review-meta { color: var(--color-text-secondary); font-size: 12px; }
.strip-spacer { flex: 1; }

.locked-badge, .warn-badge {
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.3px;
  padding: 2px 7px;
  border-radius: 10px;
}
.locked-badge { background: #eceff1; color: #455a64; }
.warn-badge { background: #fff8e1; color: #856404; }

.btn-sm {
  padding: 4px 12px;
  border: 1px solid var(--color-border);
  background: var(--color-surface);
  border-radius: 5px;
  cursor: pointer;
  font-size: 12px;
  font-weight: 600;
}
.btn-sm:hover { background: #eee; }
.btn-sm.primary {
  background: var(--color-accent);
  border-color: var(--color-accent);
  color: white;
}
.btn-sm.primary:hover { background: #3a63ad; }

.return-form {
  display: flex;
  gap: 8px;
  margin-bottom: 10px;
}
.return-form input {
  flex: 1;
  padding: 6px 10px;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  font-size: 13px;
}

.banner {
  padding: 8px 12px;
  border-radius: 6px;
  font-size: 12px;
  margin: 0 0 10px 0;
}
.banner.err { background: #fdecea; border: 1px solid #f5c6cb; color: #a12622; }

/* Frozen / live indicator */
.banner.frozen, .banner.live {
  display: flex;
  align-items: baseline;
  gap: 8px;
  flex-wrap: wrap;
  font-size: 12px;
}
.banner.frozen {
  background: #e8f5e9;
  border: 1px solid #a5d6a7;
  color: #1b5e20;
}
.banner.live {
  background: #f8f9fa;
  border: 1px solid var(--color-border);
  color: var(--color-text-secondary);
}
.banner-meta {
  margin-left: auto;
  font-size: 10px;
  opacity: 0.75;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}

.btn-sm.warn {
  background: #fff8e1;
  border-color: #ffcc80;
  color: #8a4b00;
}
.btn-sm.warn:hover { background: #ffecb3; }

/* --- diagnostics --- */
.diag {
  font-size: 12px;
  margin-bottom: 12px;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  padding: 6px 12px;
  background: var(--color-surface);
}
.diag summary { cursor: pointer; color: var(--color-text-secondary); }
.diag-body { margin-top: 8px; }
.diag-body strong { display: block; margin-top: 6px; font-size: 11px; text-transform: uppercase; letter-spacing: 0.3px; }
.diag-body p { margin: 2px 0; color: var(--color-text-secondary); }

/* --- tabs --- */
.tabbar {
  display: flex;
  gap: 2px;
  border-bottom: 2px solid var(--color-border);
  margin-bottom: 16px;
}

.tab {
  padding: 9px 20px;
  border: none;
  background: transparent;
  cursor: pointer;
  font-size: 13px;
  font-weight: 600;
  color: var(--color-text-secondary);
  border-bottom: 2px solid transparent;
  margin-bottom: -2px;
}
.tab:hover:not(.active) { color: var(--color-text); background: #f5f5f5; }
.tab.active {
  color: var(--color-accent);
  border-bottom-color: var(--color-accent);
}
.tab-err {
  display: inline-block;
  margin-left: 5px;
  color: #a12622;
  font-weight: 800;
}

.placeholder {
  color: var(--color-text-secondary);
  font-style: italic;
  text-align: center;
  padding: 40px 0;
}
</style>
