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
import { fmtItd } from '../components/snapshot/format'
import { useAuthStore } from '../stores/auth'

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

/** The client line on the PDF header — the investor's display name. */
const investorName = computed(() =>
  resolution.value?.investor_name
  || investors.value.find((i: any) => i.code === selectedInvestor.value)?.name
  || selectedInvestor.value)

/** Open the consolidated 4-page document for the current selection. */
function openPrint() {
  const q = new URLSearchParams({
    investor: selectedInvestor.value,
    quarter: selectedQuarter.value,
  })
  window.open(`/portfolio-snapshot/print?${q}`, '_blank')
}

/** ISO -> M/D/YYYY by regex; `new Date('2026-03-31')` is midnight UTC and
 *  renders as the previous day in US timezones. Same fix as format.fmtDate. */
function fmtDate(v?: string | null): string {
  if (!v) return ''
  const m = String(v).match(/^(\d{4})-(\d{2})-(\d{2})/)
  return m ? `${parseInt(m[2])}/${parseInt(m[3])}/${m[1]}` : String(v)
}
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

/** Every Financial deal row, grouped rows and ownership-flagged alike.
 *
 *  Both sets carry manual figures and both are counted into the portfolio
 *  total by ``_subtotal`` on the backend (``all_rows + flagged_rows``), so a
 *  recount that saw only the grouped rows would undercount.
 */
function financialRows(): any[] {
  const fin = bundle.value?.subtabs?.financial
  if (!fin) return []
  const rows: any[] = []
  for (const blk of Object.values<any>(fin.groups || {})) rows.push(...(blk.deals || []))
  rows.push(...(fin.ownership_flagged || []))
  return rows
}

/** The Loan subtab's rows. Its groups are plain arrays of deals, not the
 *  `{deals, subtotal}` blocks the Financial subtab uses — see assemble_loan. */
function loanRows(): any[] {
  const ln = bundle.value?.subtabs?.loan
  if (!ln) return []
  const rows: any[] = []
  for (const rs of Object.values<any>(ln.groups || {})) rows.push(...(rs || []))
  rows.push(...(ln.ownership_flagged || []))
  return rows
}

/** Manual number fields that belong to the Loan subtab's ratio columns.
 *  Mirrors MANUAL_RATIO_FIELDS on the backend. */
const RATIO_FIELDS = ['ltv', 'ytd_dscr', 'debt_yield']

/** Recount the "N entered" cells after one manual figure changed.
 *
 *  A count, not a formatting decision: the backend rule is "not in (None,
 *  PENDING)", and a row's raw ``net_roe`` / ``itd`` is a number or null — the
 *  PENDING sentinel only ever lives in the ``*_display`` twin, which the
 *  server sends us. So `!= null` reproduces it exactly. The display string
 *  itself is never derived here; see manual_display in
 *  portfolio_snapshot_financial.py.
 *
 *  total_excluding_dev.manual_entered is deliberately left alone — it is not
 *  rendered, and its own manual cells are hardcoded PENDING on the backend.
 */
function recountManual() {
  const fin = bundle.value?.subtabs?.financial
  if (!fin) return
  const tally = (rows: any[]) => ({
    itd: rows.filter((r) => r.itd != null).length,
    net_roe: rows.filter((r) => r.net_roe != null).length,
  })
  for (const blk of Object.values<any>(fin.groups || {})) {
    if (blk.subtotal) blk.subtotal.manual_entered = tally(blk.deals || [])
  }
  if (fin.total) fin.total.manual_entered = tally(financialRows())
  resumITD()
}

/** Re-add the ITD sums after one deal's figure changed.
 *
 *  ITD is summed onto every aggregate row by `_subtotal` in
 *  portfolio_snapshot_financial.py; this repeats that addition on the client so
 *  the subtotals move with the cell the analyst just left, instead of waiting
 *  for the next full `/bundle`. Deliberately kept to the ARITHMETIC — the
 *  formatting rule stays server-side, and this reuses the same `fmtItd` the
 *  cells render through. The authoritative figures arrive on the next load.
 *
 *  The excluding-development row sums the non-development deals only, so it
 *  filters on the vcode list the backend published on that row rather than
 *  re-deriving the population here.
 */
function resumITD() {
  const fin = bundle.value?.subtabs?.financial
  if (!fin) return
  const sum = (rows: any[]) => {
    const vals = rows.map((r) => r.itd).filter((v) => v != null)
    return vals.length ? vals.reduce((a, b) => a + b, 0) : null
  }
  const put = (row: any, rows: any[]) => {
    if (!row) return
    row.itd = sum(rows)
    row.itd_display = row.itd == null ? null : fmtItd(row.itd)
    row.itd_deal_count = rows.filter((r) => r.itd != null).length
  }
  for (const blk of Object.values<any>(fin.groups || {})) {
    put(blk.subtotal, blk.deals || [])
  }
  const all = financialRows()
  put(fin.total, all)
  const ex = fin.total_excluding_dev
  if (ex) {
    const removed = new Set<string>(ex.excluded_vcodes || [])
    put(ex, all.filter((r) => !removed.has(String(r.vcode || '').toUpperCase())))
  }
}

/** The manual-figure objects that are NOT deals: each fund subtotal, Portfolio
 *  Totals, and the excluding-development row. Keyed by the reserved vcode the
 *  backend stores their Net ROE against (AGG_TOTAL_VCODE and friends), so a
 *  save targeting one of them lands on the right object. */
function aggregateTargets(): Record<string, any> {
  const fin = bundle.value?.subtabs?.financial
  if (!fin) return {}
  const out: Record<string, any> = {}
  for (const [g, blk] of Object.entries<any>(fin.groups || {})) {
    if (blk.subtotal) out[`__GROUP__:${g}`] = blk.subtotal
  }
  if (fin.total) out['__TOTAL__'] = fin.total
  if (fin.total_excluding_dev) out['__EXCLUDING_DEV__'] = fin.total_excluding_dev
  return out
}

function onSaveValue(p: { vcode: string; field: string; value: string | number | null }) {
  withSave(async () => {
    const res = await api.put(`${BASE}/value`, { ...ctx(), ...p })
    // Patch the single row we changed. This used to `await load()`, which
    // refetched /bundle — rebuilding all four subtabs server-side and swapping
    // the body for the "Building snapshot…" placeholder — on every entry. That
    // read as a page refresh: focus and scroll lost, and the child's draft
    // watcher reseeded, discarding anything half-typed in another cell.
    // The Loan tab's comments never did this (see onSaveComment), which is why
    // they always felt right.
    const d = res.data || {}
    // The Loan subtab's typed ratio cells go through the same endpoint, so the
    // patch is routed by FIELD. They cannot use the Financial branch: a ratio
    // is patched onto `<field>_manual`, never onto the raw `ltv`/`ytd_dscr`/
    // `debt_yield`, which stay the computed figures the subtotals weight (see
    // MANUAL_RATIO_SEEDS in portfolio_snapshot_loan.py). Writing the typed
    // number into the raw field would silently move the fund subtotals — and
    // mix percentage points into a column of decimals.
    if (RATIO_FIELDS.includes(p.field)) {
      const row = loanRows().find((r) => r.vcode === p.vcode)
      if (row) {
        row[`${p.field}_manual`] = d.value ?? null
        row[`${p.field}_display`] = d.display
        row[`${p.field}_entered`] = true
        if (d.source) row[`${p.field}_source`] = d.source
      }
      return
    }
    // A deal row, or one of the aggregate rows — Net ROE is typeable at every
    // level, and an aggregate stores against a reserved key rather than a
    // vcode. Same endpoint, same response, same patch.
    const row = financialRows().find((r) => r.vcode === p.vcode)
      || aggregateTargets()[p.vcode]
    if (row) {
      // The stored number, not the raw string: the backend parses "1,234"/"5%"
      // and the row must hold what was persisted.
      row[p.field] = d.value ?? null
      row[`${p.field}_display`] = d.display
      if (d.source) row[`${p.field}_source`] = d.source
      recountManual()
    }
  })
}

/** Splice a footnote mutation's response back into the Financial subtab.
 *
 *  Both `footnotes` and `footnote_marks` must move together: the markers on
 *  the column headers and the property names are numbers into that exact list,
 *  so replacing one without the other would leave a marker pointing at a
 *  footnote that has been renumbered or removed. The server composes both from
 *  one call (`_footnote_payload`), which is why they cannot disagree.
 */
function applyFootnotes(data: any) {
  const fin = bundle.value?.subtabs?.financial
  if (!fin) return
  fin.footnotes = data?.footnotes || []
  fin.footnote_marks = data?.footnote_marks || { column: {}, property: {} }
  // Standing notes this quarter has removed, so the UI can offer them back.
  fin.standing_removed = data?.standing_removed || []
}

function onAddFootnote(p: { anchor: string; text: string }) {
  withSave(async () => {
    const res = await api.post(`${BASE}/footnote`, { ...ctx(), ...p })
    applyFootnotes(res.data)
  }, 'Footnote added')
}

function onRemoveFootnote(id: number) {
  withSave(async () => {
    const res = await api.delete(`${BASE}/footnote/${id}`, { params: ctx() })
    applyFootnotes(res.data)
  }, 'Footnote removed')
}

/** Reword a footnote.
 *
 *  An analyst-entered note is addressed by its database id. One of the page's
 *  STANDING notes has no row, so it is addressed by its stable key and the
 *  wording is stored against a reserved anchor for this investor and quarter
 *  only — every other quarter keeps the transcribed default, and "Restore"
 *  drops the override rather than retyping it.
 */
function onEditFootnote(p: { id: number | null; standingKey: string | null; text: string }) {
  withSave(async () => {
    const res = p.standingKey
      ? await api.put(`${BASE}/footnote/standing/${encodeURIComponent(p.standingKey)}`,
                      { ...ctx(), text: p.text })
      : await api.put(`${BASE}/footnote/${p.id}`, { ...ctx(), text: p.text })
    applyFootnotes(res.data)
  }, 'Footnote updated')
}

function onRemoveStandingFootnote(key: string) {
  withSave(async () => {
    const res = await api.delete(
      `${BASE}/footnote/standing/${encodeURIComponent(key)}`, { params: ctx() })
    applyFootnotes(res.data)
  }, 'Footnote removed from this quarter')
}

function onRestoreStandingFootnote(key: string) {
  withSave(async () => {
    const res = await api.post(
      `${BASE}/footnote/standing/${encodeURIComponent(key)}/restore`, { ...ctx() })
    applyFootnotes(res.data)
  }, 'Standard wording restored')
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

// --- freeze as sent ---
// Deliberately NOT part of the review strip: freezing records that a quarter
// was sent, approving records a decision somebody made, and the two are
// separate acts. Putting the button among the approval controls would invite
// the reader to treat it as one more step in that chain.
const showFreezeConfirm = ref(false)
const freezing = ref(false)
const freezeError = ref<string | null>(null)

const frozenAsSent = computed(() => bundle.value?.frozen_reason === 'as-sent')
const frozenBy = computed(() => bundle.value?.frozen_by || bundle.value?.approved_by || '')
const frozenOn = computed(() => {
  const raw = bundle.value?.frozen_at || bundle.value?.approved_at
  if (!raw) return ''
  const m = String(raw).match(/^(\d{4})-(\d{2})-(\d{2})/)
  return m ? `${parseInt(m[2])}/${parseInt(m[3])}/${m[1]}` : String(raw).slice(0, 10)
})
const frozenSourceLabel = computed(() => {
  const man = bundle.value?.source_manifest
  if (man?.file) {
    const n = man.overlay_cells_applied
    return `seeded from ${man.file}` + (n ? ` (${n} published cell${n === 1 ? '' : 's'})` : '')
  }
  return 'captured from the app at the moment of freezing'
})

// --- admin: freeze as sent from a published overlay -----------------------
//
// THE PREVIEW AND THE FREEZE ARE ONE ENDPOINT. `confirm:false` resolves every
// printed deal title and reports what WOULD be written; `confirm:true` repeats
// the identical resolution and writes. Two calls to one endpoint rather than
// two endpoints, so the thing previewed cannot differ from the thing frozen.
const auth = useAuthStore()
const showOverlayPanel = ref(false)
const overlayFile = ref<File | null>(null)
const overlayPreview = ref<any>(null)
const overlayBusy = ref(false)
const overlayError = ref<string | null>(null)
const overlayDone = ref<any>(null)

function pickOverlay(e: Event) {
  const f = (e.target as HTMLInputElement).files?.[0] || null
  overlayFile.value = f
  overlayPreview.value = null
  overlayDone.value = null
  overlayError.value = null
}

async function sendOverlay(confirm: boolean) {
  if (!overlayFile.value) return
  overlayBusy.value = true
  overlayError.value = null
  try {
    const doc = JSON.parse(await overlayFile.value.text())
    const res = await api.post('/api/portfolio-snapshot/freeze-overlay', {
      investor: selectedInvestor.value, quarter: selectedQuarter.value,
      overlay: doc, confirm,
    })
    if (confirm) {
      overlayDone.value = res.data
      // A per-report error means that report is still live; surface it rather
      // than reporting a blanket success.
      const failed = (res.data?.reports || []).filter((r: any) => r.error)
      if (failed.length) {
        overlayError.value = failed.map((r: any) => `${r.investor}: ${r.error}`).join(' | ')
      } else {
        await load()
      }
    } else {
      overlayPreview.value = res.data
    }
  } catch (e: any) {
    overlayError.value = e?.response?.data?.error || e?.message || 'Overlay freeze failed'
  } finally {
    overlayBusy.value = false
  }
}

async function doFreeze() {
  freezing.value = true
  freezeError.value = null
  try {
    const res = await api.post('/api/portfolio-snapshot/freeze', {
      investor: selectedInvestor.value, quarter: selectedQuarter.value,
    })
    if (!res.data?.frozen) throw new Error(res.data?.error || 'Freeze did not complete')
    showFreezeConfirm.value = false
    await load()
  } catch (e: any) {
    // The quarter is STILL LIVE when this fires. Say so on screen rather than
    // leaving a half-finished state that looks frozen.
    freezeError.value = e?.response?.data?.error || e?.message || 'Freeze failed'
  } finally {
    freezing.value = false
  }
}

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
        <!--
          Opens the consolidated 4-page document in its own tab. A new tab, not
          this one: the tabbed view mounts a single subtab and printing it yields
          one page, and navigating away would also throw away any unsaved edit
          in a textarea here.
        -->
        <button
          class="btn-print"
          :disabled="!bundle || loading"
          title="Open the 4-page report (page 1 charts, Financial, Operating, Loan)"
          @click="openPrint"
        >
          Print report
        </button>
        <span v-if="saving" class="save-note">Saving…</span>
        <span v-else-if="savedFlash" class="save-note ok">{{ savedFlash }}</span>
      </div>
    </div>

    <!--
      Reference-PDF page header. Shared by all three table subtabs, which is why
      it lives in the shell: pages 2, 3 and 4 all carry the "TIAA" client line
      and the "Current Portfolio Update" subtitle. The big centred title belongs
      to pages 1-2 only, so it is hidden on Operating and Loan.
    -->
    <div v-if="bundle" class="pdf-header">
      <h1 v-if="activeTab === 'summary' || activeTab === 'financial'" class="pdf-title">
        PORTFOLIO SNAPSHOT
      </h1>
      <div class="pdf-client">{{ investorName }}</div>
      <div class="pdf-sub">
        Current Portfolio Update (Balances as of {{ fmtDate(resolution?.quarter_end) }},
        $ millions)
      </div>
    </div>

    <!-- Print-only header -->
    <div class="print-header">
      <div class="print-meta">{{ selectedQuarter }} · {{ TABS.find(t => t.key === activeTab)?.label }}</div>
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

    <!-- Frozen vs live. The reader must never be left guessing which of the two
         they are looking at, so this states it on every load, not just when
         something is unusual. -->
    <div v-if="bundle && isFrozen" class="banner frozen">
      <strong>
        {{ frozenAsSent ? 'Frozen as sent' : 'Approved version' }}{{ frozenOn ? ` — ${frozenOn}` : '' }}
      </strong>
      <span>
        This is the stored copy of what was sent. It is not recomputed, so
        later data changes cannot move it.
        <template v-if="frozenBy">Frozen by {{ frozenBy }}.</template>
        <template v-if="bundle.frozen_version"> Version {{ bundle.frozen_version }}.</template>
      </span>
      <span class="banner-meta">{{ frozenSourceLabel }}</span>
      <span v-if="bundle.data_version" class="banner-meta">{{ bundle.data_version }}</span>
    </div>
    <div v-else-if="bundle" class="banner live">
      <strong>Live data</strong>
      <span>{{ sourceNote || 'In progress — computed from current data and will change as data changes.' }}</span>
      <button class="btn-sm primary freeze-btn" :disabled="!canLoad || loading"
              @click="showFreezeConfirm = true">Freeze as sent</button>
      <!-- ADMIN ONLY, and a second button rather than a mode of the first:
           this one freezes from a document, and conflating "freeze what the
           app computes" with "freeze what we posted" is the confusion the
           whole overlay exists to remove. -->
      <button v-if="auth.isAdmin" class="btn-sm freeze-btn"
              :disabled="!canLoad || loading"
              @click="showOverlayPanel = !showOverlayPanel">
        Freeze as sent (with published overlay)…
      </button>
    </div>

    <!-- The published-overlay freeze. Admin only, preview before write. -->
    <div v-if="showOverlayPanel && auth.isAdmin" class="freeze-confirm">
      <strong>Freeze {{ selectedQuarter }} for {{ investorName }} from the sent PDF</strong>
      <p class="muted">
        Build the overlay first with
        <code>python scripts/build_26q2_overlay.py</code>, then upload
        <code>overlay_26q2.json</code>. Every figure it carries was read from
        the document that was sent, with its page and the file's SHA-256.
      </p>
      <input type="file" accept="application/json" @change="pickOverlay" />

      <div v-if="overlayFile" class="freeze-actions">
        <button class="btn-sm" :disabled="overlayBusy" @click="sendOverlay(false)">
          {{ overlayBusy ? 'Reading…' : 'Preview' }}
        </button>
        <button class="btn-sm primary"
                :disabled="overlayBusy || !overlayPreview"
                @click="sendOverlay(true)">
          {{ overlayBusy ? 'Freezing…' : 'Confirm and freeze' }}
        </button>
        <button class="btn-sm" :disabled="overlayBusy"
                @click="showOverlayPanel = false; overlayPreview = null; overlayError = null">
          Cancel
        </button>
      </div>

      <!-- The preview IS the review: what would be written, per report. -->
      <div v-if="overlayPreview" class="overlay-preview">
        <div v-for="rep in overlayPreview.reports" :key="rep.investor" class="overlay-rep">
          <strong>{{ rep.investor }}</strong>
          <span v-if="rep.error" class="banner err">{{ rep.error }}</span>
          <template v-else>
            <div class="muted">
              {{ rep.source?.file }} · sha256 {{ (rep.source?.sha256 || '').slice(0, 12) }}…
            </div>
            <div>
              {{ rep.resolved }} of {{ rep.roster_size }} One Pagers matched ·
              <strong>{{ rep.cells_total }}</strong> cells would be overwritten
              <template v-if="rep.printed_units_total">
                ({{ rep.printed_units_total }} kept as printed text)
              </template>
            </div>
            <div v-if="rep.unresolved?.length" class="banner err">
              {{ rep.unresolved.length }} printed page(s) matched no deal:
              {{ rep.unresolved.join(', ') }} — freezing is blocked until these
              are resolved, or the record would be incomplete.
            </div>
            <div v-if="rep.already_frozen" class="banner err">
              Already frozen — Re-freeze it instead if it must change.
            </div>
            <div v-if="rep.unmapped_labels?.length" class="muted">
              {{ rep.unmapped_labels.length }} printed label(s) have no vetted
              field and are recorded, not applied.
            </div>

            <!-- What would actually CHANGE, against live. The expectation is
                 printed beside the count so a big deviation is obvious to a
                 reader; nothing is enforced on it. -->
            <div v-if="rep.live_diff" class="overlay-diff">
              <strong>
                {{ rep.live_diff.differs_total }} of
                {{ rep.live_diff.cells_total }} cells differ from live
              </strong>
              <span class="muted">
                — expected ≈{{ rep.live_diff.expected_differences }}
              </span>
              <span v-if="Math.abs(rep.live_diff.differs_total - rep.live_diff.expected_differences) > 40"
                    class="banner err">
                that is a long way from the expectation — check before freezing
              </span>
              <div class="muted">
                per page:
                <span v-for="(v, pg) in rep.live_diff.by_page" :key="pg">
                  p{{ pg }} {{ v.differs }}/{{ v.cells }}&nbsp;
                </span>
              </div>

              <div v-if="rep.live_diff.warnings?.length" class="banner err">
                <div v-for="(w, i) in rep.live_diff.warnings" :key="i">
                  <strong>{{ w.column }}</strong> — {{ w.detail }}
                </div>
                Acknowledge each to proceed: a whole column differing, or a
                ratio far from 1, is what a units or column-shift error looks
                like.
              </div>

              <div v-if="rep.live_diff.sentinels_live_non_blank?.length" class="muted">
                {{ rep.live_diff.sentinels_live_non_blank.length }} printed
                “—”/“n/a”/“Dev” cell(s) sit over a live value — the printed
                text is stored so the page shows what was sent:
                <span v-for="(s, i) in rep.live_diff.sentinels_live_non_blank.slice(0, 4)" :key="i">
                  {{ s.path }} (live {{ s.live }});
                </span>
              </div>

              <div v-if="rep.unapplied_count" class="banner err">
                {{ rep.unapplied_unacknowledged }} of {{ rep.unapplied_count }}
                printed cell(s) would NOT land in the report — the frozen copy
                would not reproduce the page.
              </div>
            </div>
            <table class="overlay-table">
              <thead><tr><th>p.</th><th>One Pager</th><th>deal</th><th class="right">cells</th></tr></thead>
              <tbody>
                <tr v-for="r in rep.per_report" :key="r.title">
                  <td>{{ r.page }}</td>
                  <td>{{ r.title }}</td>
                  <td :class="{ err: !r.vcode }">{{ r.vcode || 'no match' }}</td>
                  <td class="right">{{ r.cells }}</td>
                </tr>
              </tbody>
            </table>
          </template>
        </div>
      </div>

      <p v-if="overlayError" class="banner err">{{ overlayError }}</p>
      <p v-else-if="overlayDone" class="banner">
        Frozen. {{ (overlayDone.reports || []).filter((r: any) => r.frozen).length }}
        report(s) stored as sent.
      </p>
    </div>

    <!-- Confirmation. Freezing is not destructive but it IS a commitment: from
         here on this quarter stops following the data. Worth one click. -->
    <div v-if="showFreezeConfirm" class="freeze-confirm">
      <strong>Freeze {{ selectedQuarter }} for {{ investorName }}?</strong>
      <p>
        This locks {{ selectedQuarter }} for {{ investorName }}. Later data
        changes won't affect it. The stored copy keeps every Snapshot subtab,
        every One Pager, and the roster in the order it was sent.
      </p>
      <p class="muted">
        Typed fields — Net ROE, ITD, comments and footnotes — become read-only.
        An admin can Re-freeze or Unfreeze it afterwards, with a reason.
      </p>
      <p v-if="freezeError" class="banner err">
        {{ freezeError }} — {{ selectedQuarter }} is still live.
      </p>
      <div class="freeze-actions">
        <button class="btn-sm primary" :disabled="freezing" @click="doFreeze">
          {{ freezing ? 'Freezing…' : 'Freeze as sent' }}
        </button>
        <button class="btn-sm" :disabled="freezing"
                @click="showFreezeConfirm = false; freezeError = null">Cancel</button>
      </div>
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
            @edit-footnote="onEditFootnote"
            @remove-standing-footnote="onRemoveStandingFootnote"
            @restore-standing-footnote="onRestoreStandingFootnote"
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
            :screen-note="true"
            @save-comment="onSaveComment"
            @save-value="onSaveValue"
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

.btn-print {
  padding: 8px 16px;
  border: 1px solid var(--color-accent);
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
  font-weight: 600;
  background: var(--color-surface);
  color: var(--color-accent);
}
.btn-print:hover:not(:disabled) { background: #eef2fa; }
.btn-print:disabled { opacity: 0.6; cursor: not-allowed; }

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
/* The button sits in the live banner, so it is next to the words that say the
   quarter is still live — the state it changes. */
.freeze-btn { margin-left: auto; }

.freeze-confirm {
  margin: 8px 0 12px;
  padding: 12px 14px;
  border: 1px solid #f0c36d;
  border-left: 4px solid #e0a800;
  border-radius: 4px;
  background: #fffbf0;
  font-size: 12px;
}
.freeze-confirm strong { display: block; margin-bottom: 6px; font-size: 13px; }
.freeze-confirm p { margin: 0 0 8px; line-height: 1.45; }
.freeze-confirm p.muted { color: #6b7684; }
.overlay-preview { margin-top: 10px; }
.overlay-diff { margin-top: 8px; padding: 6px 8px; background: #f7f9fc; border-radius: 4px; }
.overlay-rep { margin-bottom: 12px; }
.overlay-table { width: 100%; border-collapse: collapse; margin-top: 6px; font-size: 11px; }
.overlay-table th, .overlay-table td { border-bottom: 1px solid #eee; padding: 2px 6px; text-align: left; }
.overlay-table .right { text-align: right; }
.overlay-table td.err { color: #b00020; font-weight: 600; }
.freeze-actions { display: flex; gap: 8px; }

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

.print-header { display: none; }

/* Reference-PDF page header. Serif caps title over a heavy rule, then the
   client line and the balances-as-of subtitle, matching page 2. */
.pdf-header { margin: 4px 0 10px 0; }
.pdf-title {
  font-family: Georgia, "Times New Roman", serif;
  font-size: 26px;
  font-weight: 700;
  letter-spacing: 0.5px;
  text-align: center;
  margin: 0 0 10px 0;
  padding-bottom: 10px;
  border-bottom: 3px solid var(--color-text);
}
.pdf-client { font-size: 13px; font-weight: 700; }
.pdf-sub { font-size: 12px; color: var(--color-text-secondary); }

@media print {
  /* This view's real margin. It used to be `@page { margin: 0.5in }`, which
     could not be scoped and so reset the page box for every other print view
     in the app — the One Pager included. Same 0.5in on paper, but it stops at
     this view's own container. See the note in App.vue. */
  .snapshot { padding: 0.5in; }
  .snap-header { display: none; }
  .print-header { display: block; margin-bottom: 8px; }
  .print-header h2 { font-size: 16px; margin: 0 0 2px 0; }
  .print-meta { font-size: 12px; color: #666; }
  .pdf-title { font-size: 20px; padding-bottom: 7px; border-bottom-width: 2px; }
  .review-strip { display: none; }
  .return-form { display: none; }
  .banner { display: none; }
  .diag { display: none; }
  .tabbar { display: none; }
  .tabbody { padding: 0; }
}
</style>
