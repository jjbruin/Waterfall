<script setup lang="ts">
/**
 * Accounting workpaper packages — the quarterly close.
 *
 * THE STATEMENTS ARE THE PRODUCT; the checklist is how you get there. So the
 * drafted statements sit at the top of a package with their tie-outs visible,
 * and each carries a chip saying whether the system's own check passed. What
 * the app is producing should be the first thing on screen, not something you
 * find by downloading a workbook.
 *
 * AND THE WORK HAPPENS WHERE THE EVIDENCE IS. Clicking a step loads exactly
 * what that step asserts — its guidance, the checks the system can run, the
 * accounts or figures behind it, and the exhibit slots it expects. A preparer
 * should never have to leave the step to find out whether it is true. The
 * step -> evidence mapping is accounting knowledge and lives on the server
 * (workpaper_workbench.py), not here.
 *
 * TWO TABS, BECAUSE THEY ARE TWO JOBS. Production tracking is the CFO's view
 * ACROSS every reporting entity -- his order, his target dates, who prepared
 * and who signed. The workbench is ONE entity's package. They were on one
 * screen, so the cross-entity grid and the single-package drawer competed for
 * it and neither got the room. Jim, Sep 16 2026: "separating the production
 * tracking from the individual package workbench".
 *
 * The tracker replicates `2Q26 - PSC Reporting Checklist & Calendar.xlsx`,
 * which is what the CFO runs the close from today.
 */
import { ref, computed, onMounted } from 'vue'
import api from '../api/client'
import { useAuthStore } from '../stores/auth'

const auth = useAuthStore()

// Production tracking, or one entity's package. The tracker opens first: it is
// the screen that answers "where is the close", which is the question asked
// most often and by the most people.
const tab = ref<'tracker' | 'workbench'>('tracker')

// The CFO's grid. Separate from `tracker` below, which is the older
// per-package STEP checklist and still backs the workbench's progress figures.
const sched = ref<any>(null)
const schedLoading = ref(false)
// Group the rows by the property/portfolio column, the way the spreadsheet
// does. Off by default: the CFO's own order is the point, and grouping
// overrides it.
const groupByProperty = ref(false)
const carryFrom = ref<number | null>(null)
// Who can be assigned. Sourced from accounts carrying an accounting role, so
// the list fills itself as those accounts are created.
const preparers = ref<any[]>([])

const cycles = ref<any[]>([])
const cycleId = ref<number | null>(null)
const tracker = ref<any>(null)
const loading = ref(false)
const error = ref('')
const msg = ref('')

const detail = ref<any>(null)
const statements = ref<any>(null)
const openStatement = ref<string>('')
const activeStep = ref<string>('')
const evidence = ref<any>(null)
const evidenceLoading = ref(false)

const showNewCycle = ref(false)
const newLabel = ref('')
const newEnd = ref('')
const uploadSlot = ref('other')
const uploadCaption = ref('')
const returnNote = ref('')

// THE ACCOUNTING SECTION IS THE CFO'S, so the gate is not "is this an admin".
// Gating the CFO's own columns -- his order, his target dates -- on `admin`
// hid them from the one person they belong to, while the API would have
// accepted his writes: the buttons simply were not rendered. Jim, Sep 16 2026:
// "give the cfo control of syncing entities and starting a new close cycle and
// everything else in the accounting section going forward."
// ONE DEFINITION, in the auth store, mirroring ACCOUNTING_ROLES on the server.
// It was ['admin', 'cfo'] here, which locked out the accountants and the
// accounting manager who actually prepare the close -- the same class of
// mistake as v480's, where the SCREEN was the thing keeping the CFO out while
// the API would have accepted his writes.
const canManageClose = computed(() => auth.canEditAccounting)
// Narrower than working in the close: opening a cycle and setting the dates
// the close is measured against are the CFO's.
const canSetDates = computed(() => auth.canSetCloseDates)
const STATEMENT_KEYS = ['balance_sheet', 'income_statement', 'soi',
                        'members_capital', 'cash_flow']

function flash(m: string) {
  msg.value = m
  setTimeout(() => (msg.value = ''), 4000)
}

async function loadCycles() {
  const res = await api.get('/api/workpapers/cycles')
  cycles.value = res.data.cycles || []
  if (!cycleId.value && cycles.value.length) cycleId.value = cycles.value[0].id
  if (cycleId.value) { await loadTracker(); await loadSchedule() }
  await loadPreparers()
}

async function loadPreparers() {
  try {
    preparers.value = (await api.get('/api/workpapers/preparers')).data.preparers || []
  } catch { preparers.value = [] }
}

/** Fill the Property column from the deal each entity holds. Never overwrites
 *  a typed value — a value the CFO typed is a decision, a derived one a guess. */
/** Fill the Property column by walking commitments. Never overwrites a typed
 *  value, and every value it does write says how it was reached. */
async function fillProperties() {
  if (!cycleId.value) return
  try {
    const r = await api.post(
      `/api/workpapers/cycles/${cycleId.value}/schedule/properties`, {})
    await loadSchedule()
    flash(`Filled ${r.data.filled} propert${r.data.filled === 1 ? 'y' : 'ies'} from the deals.` +
      (r.data.kept_existing ? ` ${r.data.kept_existing} already had a value and were left alone.` : '') +
      (r.data.annotated ? ` ${r.data.annotated} already-filled row(s) now show how they were derived.` : '') +
      (r.data.unresolved_count ? ` ${r.data.unresolved_count} could not be resolved.` : '') +
      ' Inferred names are marked — a superscript 2 means it was reached two levels down and is worth a glance.')
  } catch (e: any) { error.value = e.response?.data?.error || e.message }
}

async function loadSchedule() {
  if (!cycleId.value) return
  schedLoading.value = true
  error.value = ''
  try {
    sched.value = (await api.get(
      `/api/workpapers/cycles/${cycleId.value}/schedule`)).data
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    schedLoading.value = false
  }
}

/** Every write goes through here so one failure cannot leave the grid showing
 *  a value the server rejected. On success the row is reloaded from the server
 *  rather than patched locally — `overdue` and `out_of_sequence` are computed
 *  there, and a locally-patched cell would show a stale flag beside a fresh
 *  value, which is the worst of both. */
async function schedWrite(url: string, body: any, ok: string) {
  try {
    const res = await api.put(url, body)
    if (res.data?.error) { error.value = res.data.error; return false }
    await loadSchedule()
    if (ok) flash(ok)
    return true
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
    return false
  }
}

const setOrder = (pid: number, v: string) =>
  schedWrite(`/api/workpapers/packages/${pid}/schedule/order`, { order: v }, '')
function preparerName(ini: string | null) {
  if (!ini) return ''
  const p = preparers.value.find(
    (x: any) => String(x.initials).toUpperCase() === String(ini).toUpperCase())
  return p?.name ? `${p.initials} — ${p.name}` : String(ini)
}

const setPreparer = (pid: number, v: string) =>
  schedWrite(`/api/workpapers/packages/${pid}/schedule/preparer`, { preparer: v }, '')
const setProperty = (pid: number, v: string) =>
  schedWrite(`/api/workpapers/packages/${pid}/schedule/property`, { property_name: v }, '')
const setTarget = (pid: number, deliverable: string, v: string) =>
  schedWrite(`/api/workpapers/packages/${pid}/schedule/target`,
             { deliverable, target_date: v }, '')

/** A sign-off is two fields — WHO and WHEN — and the spreadsheet keeps them in
 *  one cell ("KH 7/7/26"). Kept apart here: a name and a date sorted, filtered
 *  and validated as free text is how the spreadsheet ended up with three date
 *  formats in one column. */
function signoff(pid: number, deliverable: string, stage: string,
                 by: string | null, on: string | null) {
  return schedWrite(`/api/workpapers/packages/${pid}/schedule/signoff`,
                    { deliverable, stage, signed_by: by, signed_on: on }, '')
}

/** Sign as the current user, today — the common case, one click. */
function signNow(pid: number, deliverable: string, stage: string) {
  const who = (auth.user?.username || '').slice(0, 3).toUpperCase()
  return signoff(pid, deliverable, stage, who,
                 new Date().toISOString().slice(0, 10))
}

/** One entity, or the whole cycle in the CFO's order. Opens the print view in
 *  a new tab so the close screen is not replaced by a document. */
function printStatements(entityId?: string) {
  if (!cycleId.value) return
  const q = new URLSearchParams()
  if (entityId) q.set('entities', entityId)
  else q.set('cycle_id', String(cycleId.value))
  const pe = sched.value?.cycle?.period_end
  if (pe) q.set('period_end', String(pe))
  window.open(`/workpapers/print?${q.toString()}`, '_blank')
}

async function renumber() {
  if (!cycleId.value) return
  try {
    const r = await api.post(
      `/api/workpapers/cycles/${cycleId.value}/schedule/renumber`, {})
    await loadSchedule()
    flash(`Renumbered ${r.data.renumbered} rows 1-${r.data.renumbered}, in the order shown.`)
  } catch (e: any) { error.value = e.response?.data?.error || e.message }
}

async function carryForward() {
  if (!cycleId.value || !carryFrom.value) return
  try {
    const r = await api.post(
      `/api/workpapers/cycles/${cycleId.value}/schedule/carry-forward`,
      { from_cycle_id: carryFrom.value })
    await loadSchedule()
    const miss = (r.data.not_in_source || []).length
    flash(`Order, preparer and property copied onto ${r.data.applied} rows.` +
          (miss ? ` ${miss} entity(ies) are new this quarter and still need placing.` : ''))
  } catch (e: any) { error.value = e.response?.data?.error || e.message }
}

/** Rows as rendered: the server's order, optionally broken into property
 *  groups. Grouping preserves the CFO's order WITHIN each group and orders the
 *  groups by their first row, so turning it on never reshuffles his sequence. */
const schedGroups = computed(() => {
  const rows = sched.value?.rows || []
  if (!groupByProperty.value) return [{ name: '', rows }]
  const out: any[] = []
  const idx: Record<string, number> = {}
  for (const r of rows) {
    const k = r.property_name || '(no property)'
    if (idx[k] === undefined) { idx[k] = out.length; out.push({ name: k, rows: [] }) }
    out[idx[k]].rows.push(r)
  }
  return out
})

const schedTotals = computed(() => {
  const rows = sched.value?.rows || []
  const cells = rows.reduce((a: number, r: any) => a + r.cell_count, 0)
  const done = rows.reduce((a: number, r: any) => a + r.signed_count, 0)
  const late = rows.reduce(
    (a: number, r: any) => a + r.deliverables.filter((d: any) => d.overdue).length, 0)
  return { rows: rows.length, cells, done, late,
           pct: cells ? Math.round((100 * done) / cells) : 0 }
})

async function loadTracker() {
  if (!cycleId.value) return
  loading.value = true
  error.value = ''
  try {
    tracker.value = (await api.get(`/api/workpapers/cycles/${cycleId.value}/tracker`)).data
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    loading.value = false
  }
}

async function createCycle() {
  if (!newLabel.value.trim() || !newEnd.value) return
  try {
    const res = await api.post('/api/workpapers/cycles', {
      period_label: newLabel.value.trim(), period_end: newEnd.value,
    })
    showNewCycle.value = false
    newLabel.value = ''; newEnd.value = ''
    await loadCycles()
    cycleId.value = res.data.id
    await loadTracker()
    flash('Close cycle created — packages generated for every REP entity')
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  }
}

async function sync() {
  const res = await api.post(`/api/workpapers/cycles/${cycleId.value}/sync`)
  await loadTracker()
  flash(`${res.data.created} package(s) added — ${res.data.rep_entities} entities tagged REP`)
}

async function setDue(stepKey: string, due: string) {
  let refusal = ''
  let warnings: string[] = []
  try {
    const res = await api.put(`/api/workpapers/cycles/${cycleId.value}/steps`, {
      step_key: stepKey, due_date: due || null,
    })
    warnings = res.data.warnings || []
  } catch (e: any) {
    refusal = e.response?.data?.error || e.message
  }
  // RELOAD FIRST, THEN SPEAK. The reload is what puts a refused field back to
  // its stored value -- but it also clears `error` on entry, so a message set
  // before it is wiped and the field just snaps back with no explanation.
  await loadTracker()
  if (refusal) error.value = refusal
  else if (warnings.length) flash(`Saved. ${warnings.join(' ')}`)
}

/** The picker's value arrives from a <select>, so it is a string. */
function onPickEntity(v: string) {
  if (!v) { detail.value = null; return }
  openPackage(Number(v))
}

/** Where the open package sits in the TRACKER's order, or -1. Drives prev/next
 *  so an accountant can work the CFO's sequence straight down without going
 *  back to the grid between each one. */
const benchIndex = computed(() => {
  const rows = sched.value?.rows || []
  const id = detail.value?.package?.id
  return id == null ? -1 : rows.findIndex((r: any) => r.package_id === id)
})
const hasPrev = computed(() => benchIndex.value > 0)
const hasNext = computed(() => {
  const n = (sched.value?.rows || []).length
  return benchIndex.value >= 0 && benchIndex.value < n - 1
})
function prevEntity() {
  if (hasPrev.value) openPackage(sched.value.rows[benchIndex.value - 1].package_id)
}
function nextEntity() {
  if (hasNext.value) openPackage(sched.value.rows[benchIndex.value + 1].package_id)
}

async function openPackage(id: number) {
  // Selecting an entity IS switching to the workbench. Leaving the tab alone
  // would load a package the user cannot see and look like nothing happened.
  tab.value = 'workbench'
  detail.value = (await api.get(`/api/workpapers/packages/${id}`)).data
  statements.value = null
  evidence.value = null
  activeStep.value = ''
  loadStatements()
  // Open on the first unfinished step: that is where the work is.
  const next = detail.value.steps.find((s: any) => !s.done) || detail.value.steps[0]
  if (next) selectStep(next.key)
}

async function loadStatements() {
  if (!detail.value) return
  try {
    statements.value = (await api.get(
      `/api/workpapers/packages/${detail.value.package.id}/statements`)).data
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  }
}

async function selectStep(key: string) {
  activeStep.value = key
  evidenceLoading.value = true
  evidence.value = null
  try {
    evidence.value = (await api.get(
      `/api/workpapers/packages/${detail.value.package.id}/steps/${key}/evidence`)).data
    const slots = evidence.value.exhibit_slots || []
    if (slots.length) uploadSlot.value = slots[0]
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    evidenceLoading.value = false
  }
}

async function toggleStep(step: any) {
  await api.put(`/api/workpapers/packages/${detail.value.package.id}/steps`, {
    step_key: step.key, done: !step.done,
  })
  const id = detail.value.package.id
  detail.value = (await api.get(`/api/workpapers/packages/${id}`)).data
  await loadTracker()
}

async function act(action: string) {
  try {
    await api.post(`/api/workpapers/packages/${detail.value.package.id}/transition`, {
      action, note: returnNote.value,
    })
    returnNote.value = ''
    const id = detail.value.package.id
    detail.value = (await api.get(`/api/workpapers/packages/${id}`)).data
    await loadTracker()
    flash('Updated')
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  }
}

async function uploadExhibit(ev: Event) {
  const input = ev.target as HTMLInputElement
  if (!input.files?.length) return
  const fd = new FormData()
  fd.append('file', input.files[0])
  fd.append('slot_key', uploadSlot.value)
  fd.append('caption', uploadCaption.value)
  try {
    await api.post(`/api/workpapers/packages/${detail.value.package.id}/exhibits`, fd,
      { headers: { 'Content-Type': 'multipart/form-data' } })
    uploadCaption.value = ''
    input.value = ''
    const id = detail.value.package.id
    detail.value = (await api.get(`/api/workpapers/packages/${id}`)).data
    if (activeStep.value) await selectStep(activeStep.value)
    await loadTracker()
    flash('Exhibit attached')
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  }
}

async function removeExhibit(id: number) {
  if (!confirm('Remove this exhibit from the package?')) return
  await api.delete(`/api/workpapers/exhibits/${id}`)
  const pid = detail.value.package.id
  detail.value = (await api.get(`/api/workpapers/packages/${pid}`)).data
  await loadTracker()
}

async function downloadPackage(id: number, name: string) {
  const res = await api.get(`/api/workpapers/packages/${id}/download`, { responseType: 'blob' })
  const url = URL.createObjectURL(res.data)
  const a = document.createElement('a')
  a.href = url; a.download = name; a.click()
  URL.revokeObjectURL(url)
}

const exhibitUrl = (id: number) => `/api/workpapers/exhibits/${id}`
const slotLabel = (key: string) =>
  (detail.value?.slots || []).find((s: any) => s.key === key)?.label || key

const stateClass = (s: string) => ({
  not_started: 'st-grey', in_progress: 'st-blue', submitted: 'st-amber',
  manager_approved: 'st-teal', cfo_approved: 'st-green', returned: 'st-red',
} as Record<string, string>)[s] || 'st-grey'

function fmt(v: any) {
  if (v === null || v === undefined || v === '') return '—'
  if (typeof v === 'number') return v.toLocaleString(undefined,
    { minimumFractionDigits: 2, maximumFractionDigits: 2 })
  return String(v)
}

// Exhibits relevant to the step in view, so the preparer attaches evidence
// against the assertion rather than into a general pile.
const stepExhibits = computed(() => {
  const slots = evidence.value?.exhibit_slots || []
  if (!slots.length || !detail.value) return []
  return detail.value.exhibits.filter((e: any) => slots.includes(e.slot_key))
})

onMounted(loadCycles)
</script>

<template>
  <div class="page">
    <div class="page-head">
      <div>
        <h1>Workpaper Packages</h1>
        <p class="subtitle">
          Quarterly close — one package per entity tagged <code>REP</code> in MRI
        </p>
      </div>
      <div class="head-actions">
        <select v-model="cycleId" @change="loadTracker" class="sel">
          <option v-for="c in cycles" :key="c.id" :value="c.id">
            {{ c.period_label }} — {{ c.period_end }}
          </option>
        </select>
        <button v-if="canManageClose" class="btn" @click="sync" :disabled="!cycleId">Sync entities</button>
        <button v-if="canSetDates" class="btn primary" @click="showNewCycle = !showNewCycle">
          New close cycle
        </button>
      </div>
    </div>

    <div v-if="showNewCycle && canSetDates" class="new-cycle">
      <input v-model="newLabel" placeholder="Label, e.g. Q2 2026" />
      <input v-model="newEnd" type="date" />
      <button class="btn primary" @click="createCycle">Create</button>
      <span class="hint">Creates a package for every entity tagged REP, and an
        empty deadline for each step of the close.</span>
    </div>

    <nav class="tabs" role="tablist">
      <button class="tab" :class="{ active: tab === 'tracker' }" role="tab"
              :aria-selected="tab === 'tracker'" @click="tab = 'tracker'">
        Production tracker
      </button>
      <button class="tab" :class="{ active: tab === 'workbench' }" role="tab"
              :aria-selected="tab === 'workbench'" @click="tab = 'workbench'">
        Package workbench
        <span v-if="detail" class="tab-sub">{{ detail.package.entityid }}</span>
      </button>
    </nav>

    <div v-if="msg" class="banner ok">{{ msg }}</div>
    <div v-if="error" class="banner err">{{ error }}</div>

    <!-- ── Tab 1: production tracker ───────────────────── -->
    <div v-show="tab === 'tracker'" class="tracker-tab">
      <div v-if="schedLoading" class="placeholder">Loading the tracker…</div>

      <template v-else-if="sched && !sched.error">
        <div class="sched-bar">
          <div class="sched-stat">
            <b>{{ schedTotals.rows }}</b> reporting entities
          </div>
          <div class="sched-stat">
            <b>{{ schedTotals.done }}</b> of {{ schedTotals.cells }} sign-offs
            <span class="muted">({{ schedTotals.pct }}%)</span>
          </div>
          <div class="sched-stat" :class="{ bad: schedTotals.late }">
            <b>{{ schedTotals.late }}</b> past target
          </div>
          <label class="chk">
            <input type="checkbox" v-model="groupByProperty" />
            Group by property
          </label>
          <div class="spacer" />
          <button class="btn" @click="printStatements()"
                  title="Opens every entity's statements in this cycle, in your
                         order, as one printable document.">
            Print statements
          </button>
          <template v-if="canManageClose">
            <select v-model.number="carryFrom" class="sel sm">
              <option :value="null">Carry forward from…</option>
              <option v-for="c in cycles.filter(c => c.id !== cycleId)"
                      :key="c.id" :value="c.id">{{ c.period_label }}</option>
            </select>
            <button class="btn" :disabled="!carryFrom" @click="carryForward"
                    title="Copies the order, preparer and property only — never
                           a sign-off or a target date, which are facts about
                           their own quarter.">Carry forward</button>
            <button class="btn" @click="fillProperties"
                    title="Reads the deal each entity holds and fills the Property
                           column. A value already typed is left alone.">
              Fill properties
            </button>
            <button class="btn" @click="renumber"
                    title="Rewrites the order as 1..n in the order shown, so a
                           row can be inserted between two others again.">
              Renumber 1-n
            </button>
          </template>
        </div>

        <!-- What the grid cannot show by being correct: rows nobody has placed,
             an ambiguous order, and a sign-off recorded ahead of its turn. -->
        <div v-if="sched.diagnostics.unordered_count" class="note">
          <b>{{ sched.diagnostics.unordered_count }}</b> entit{{ sched.diagnostics.unordered_count === 1 ? 'y has' : 'ies have' }}
          no order yet and sort to the bottom:
          <span class="muted">{{ sched.diagnostics.unordered.join(', ') }}</span>
        </div>
        <div v-for="d in sched.diagnostics.duplicate_orders" :key="d.order" class="note">
          Order <b>{{ d.order }}</b> is used by {{ d.entities.join(' and ') }} —
          their sequence falls back to name.
        </div>
        <div v-if="sched.diagnostics.out_of_sequence_rows.length" class="note warn">
          Signed ahead of turn:
          <span v-for="r in sched.diagnostics.out_of_sequence_rows" :key="r.entityid" class="oos">
            {{ r.entityid }} ({{ r.cells.join('; ') }})
          </span>
          <div class="muted">Recorded, not refused — a correction gets re-signed
            out of order routinely.</div>
        </div>

        <div v-if="!sched.rows.length" class="placeholder">
          No packages in this cycle. If entities are tagged REP in MRI, press
          <strong>Sync entities</strong>; if none are, nothing is due.
        </div>

        <div v-else class="grid-wrap sched-wrap">
          <table class="grid sched">
            <thead>
              <tr>
                <th class="sticky-l ord" rowspan="2" title="The CFO's order. Type a number; the grid sorts by it.">#</th>
                <th class="sticky-e" rowspan="2">Entity</th>
                <th rowspan="2">Property</th>
                <th rowspan="2" title="Initials of the assigned accountant">Prep</th>
                <th v-for="d in sched.deliverables" :key="d.key"
                    :colspan="d.stages.length + 1" class="grp" :title="d.note">
                  {{ d.label }}
                </th>
              </tr>
              <tr>
                <template v-for="d in sched.deliverables" :key="d.key">
                  <th class="tgt-h" title="Target date the CFO set for this deliverable">Target</th>
                  <th v-for="st in d.stages" :key="st.key" class="stg-h">
                    <div>{{ st.label }}</div>
                    <div class="owner">{{ st.owner }}</div>
                  </th>
                </template>
              </tr>
            </thead>
            <tbody v-for="g in schedGroups" :key="g.name || '_'">
              <tr v-if="g.name" class="grp-row">
                <td :colspan="4 + sched.deliverables.reduce((a: number, d: any) => a + d.stages.length + 1, 0)">
                  {{ g.name }}
                </td>
              </tr>
              <tr v-for="r in g.rows" :key="r.package_id"
                  :class="{ open: detail?.package?.id === r.package_id }">
                <td class="sticky-l ord">
                  <input v-if="canManageClose" class="ord-in" type="number" min="0"
                         :value="r.sort_order ?? ''"
                         @change="setOrder(r.package_id, ($event.target as HTMLInputElement).value)" />
                  <span v-else>{{ r.sort_order ?? '—' }}</span>
                </td>
                <td class="sticky-e">
                  <button class="link" @click="openPackage(r.package_id)">{{ r.entityid }}</button>
                  <div class="ent-name">{{ r.entity_name }}</div>
                </td>
                <!-- A DERIVED PROPERTY SAYS SO. The name is a starting point
                     reached by walking commitments, and at two levels it can be
                     confidently wrong — TGA6 is a fund that happens to reach one
                     deal, so it comes back as that deal where the CFO's sheet
                     says "Various". The marker is how a reader tells an
                     inference from a decision; typing over it clears the
                     marker, because then it is his. -->
                <td class="prop" :class="{ inferred: r.property_basis }">
                  <input v-if="canManageClose" class="txt-in" :value="r.property_name || ''"
                         placeholder="—"
                         :title="r.property_basis
                                 ? 'Inferred: ' + r.property_basis + '. Type over it to make it yours.'
                                 : ''"
                         @change="setProperty(r.package_id, ($event.target as HTMLInputElement).value)" />
                  <span v-else>{{ r.property_name || '—' }}</span>
                  <span v-if="r.property_basis" class="inf-mark"
                        :title="'Inferred: ' + r.property_basis">
                    {{ r.property_basis.includes('levels down') ? '²' : '¹' }}
                  </span>
                </td>
                <!-- INITIALS ONLY in this column, which is why the name lives
                     in the option text and the title rather than the cell. -->
                <td class="prep">
                  <select v-if="canManageClose" class="ini-sel"
                          :value="r.preparer || ''"
                          :title="preparerName(r.preparer)"
                          @change="setPreparer(r.package_id, ($event.target as HTMLSelectElement).value)">
                    <option value="">—</option>
                    <option v-for="pp in preparers" :key="pp.initials" :value="pp.initials">
                      {{ pp.initials }}{{ pp.name ? ' — ' + pp.name : '' }}
                    </option>
                  </select>
                  <span v-else>{{ r.preparer || '—' }}</span>
                </td>
                <template v-for="d in r.deliverables" :key="d.key">
                  <td class="tgt" :class="{ late: d.overdue, ok: d.complete }">
                    <!-- The DATE is the CFO's; the sign-off cells beside it
                         are the team's. They record work against the deadline,
                         they do not move it. -->
                    <input v-if="canSetDates" class="date-in" type="date"
                           :value="d.target_date || ''"
                           @change="setTarget(r.package_id, d.key, ($event.target as HTMLInputElement).value)" />
                    <span v-else>{{ d.target_date || '—' }}</span>
                  </td>
                  <td v-for="st in d.stages" :key="st.key" class="sign"
                      :class="{ signed: st.signed }">
                    <!-- THE DATE ONLY. The column header already says whose
                         signature this is — Preparer, Acctg Mgr, CFO — so
                         repeating the initials in every cell says it twice
                         (Jim, Sep 17 2026). Who signed is still RECORDED and is
                         on the tooltip; it is the display that was redundant. -->
                    <template v-if="st.signed">
                      <div class="when"
                           :title="(st.signed_by ? st.signed_by + ' — ' : '') + (st.signed_on || '')">
                        {{ st.signed_on || '—' }}
                      </div>
                      <button class="x" title="Clear this sign-off"
                              @click.stop="signoff(r.package_id, d.key, st.key, '', '')">×</button>
                    </template>
                    <button v-else class="sign-btn"
                            :title="`Sign ${d.label} — ${st.label} as ${st.owner}, dated today`"
                            @click="signNow(r.package_id, d.key, st.key)">+</button>
                  </td>
                </template>
              </tr>
            </tbody>
          </table>
        </div>
        <p class="legend">
          Each sign-off records the initials typed and the date, and the account
          that saved it. <b>+</b> signs as you, today; click the <b>×</b> on a
          signed cell to clear it.
        </p>
      </template>
    </div>

    <!-- == Tab 2: one entity's package =============================== -->
    <div v-show="tab === 'workbench'" class="bench-tab">
      <!-- THE ENTITY PICKER IS THE TAB'S OWN CONTROL, not a row click on
           another screen. An accountant works one entity at a time and should
           not have to find it in a 58-row grid first. Jim, Sep 16 2026:
           "Accountants should be able to select the reporting entity that they
           want to work on with a dropdown at the top of the workbench tab."
           Options come from the tracker's own rows, so the picker carries the
           CFO's order and each entity's progress with it. -->
      <div class="bench-pick">
        <label>Reporting entity</label>
        <select class="sel wide"
                :value="detail?.package?.id ?? ''"
                @change="onPickEntity(($event.target as HTMLSelectElement).value)">
          <option value="">- select an entity -</option>
          <option v-for="r in (sched?.rows || [])" :key="r.package_id"
                  :value="r.package_id">
            {{ r.sort_order != null ? r.sort_order + '. ' : '' }}{{ r.entityid }}
            - {{ r.entity_name }} ({{ r.signed_count }}/{{ r.cell_count }} signed)
          </option>
        </select>
        <button v-if="detail" class="btn nav" @click="prevEntity" :disabled="!hasPrev"
                title="Previous entity in the tracker's order">&lsaquo;</button>
        <button v-if="detail" class="btn nav" @click="nextEntity" :disabled="!hasNext"
                title="Next entity in the tracker's order">&rsaquo;</button>
        <span v-if="!sched || !sched.rows || !sched.rows.length" class="muted">
          No entities in this cycle yet - press <b>Sync entities</b>.
        </span>
      </div>

      <div v-if="!detail" class="placeholder">
        Pick a reporting entity above to open its package.
      </div>

    <div v-if="detail" class="drawer">
      <div class="drawer-head">
        <div>
          <h2>{{ detail.package.entity_name || detail.package.entityid }}</h2>
          <div class="sub">
            {{ detail.package.entityid }} · {{ detail.package.period_label }} ·
            period ended {{ detail.package.period_end }}
            <span class="chip" :class="stateClass(detail.package.state)">
              {{ detail.package.state_label }}</span>
          </div>
        </div>
        <div class="head-actions">
          <button class="btn primary"
                  @click="downloadPackage(detail.package.id,
                          `${detail.package.entityid} - WP - ${detail.package.period_end}.xlsx`)">
            Download package
          </button>
          <button class="btn"
                  @click="printStatements(detail.package.entityid)">
            Print statements
          </button>
          <button class="btn" @click="tab = 'tracker'">Back to tracker</button>
        </div>
      </div>

      <!-- What the system is producing, first. -->
      <section class="statements">
        <div class="sec-head">
          <h3>Drafted financial statements</h3>
          <span class="muted small" v-if="statements">
            <template v-if="statements.unmapped_count">
              {{ statements.unmapped_count }} account(s) unmapped,
              {{ fmt(statements.unmapped_total) }} not on any line
            </template>
            <template v-else>every account mapped to a statement line</template>
          </span>
        </div>
        <div v-if="!statements" class="muted small">Building statements…</div>
        <div v-else class="stmt-cards">
          <button v-for="k in STATEMENT_KEYS" :key="k" class="stmt-card"
                  :class="{ active: openStatement === k,
                            ok: statements[k].ties === true,
                            bad: statements[k].ties === false }"
                  @click="openStatement = openStatement === k ? '' : k">
            <div class="stmt-title">{{ statements[k].title }}</div>
            <div class="stmt-tie">{{ statements[k].tie_label }}</div>
          </button>
        </div>

        <div v-if="openStatement && statements" class="stmt-body">
          <!-- sectioned statements -->
          <template v-if="statements[openStatement].sections">
            <div v-for="sec in statements[openStatement].sections" :key="sec.section">
              <h4>{{ sec.section }}</h4>
              <table class="mini">
                <tbody>
                  <!-- Dormant = no balance and no movement. Hidden here for
                       the same reason the printed statement hides them; the
                       count below keeps them from being silently absent. -->
                  <tr v-for="l in sec.lines.filter((x: any) => !x.dormant)" :key="l.fs_line">
                    <td>{{ l.fs_line }}</td>
                    <td class="num">{{ fmt(l.amount) }}</td>
                  </tr>
                  <tr class="tot"><td>Total {{ sec.section }}</td>
                    <td class="num">{{ fmt(sec.total) }}</td></tr>
                  <tr v-if="sec.dormant_count"><td colspan="2" class="muted small">
                    {{ sec.dormant_count }} line(s) with no balance and no movement not shown
                  </td></tr>
                </tbody>
              </table>
            </div>
            <!-- THE CLOSING LINE, whichever statement this is: liabilities
                 and members' capital against total assets, net income under
                 expenses, or the net change in cash against the balance
                 sheet's own movement. One block because the engine gives all
                 three the same shape. Where there is something to compare
                 against, it says whether it ties rather than leaving a reader
                 to subtract two numbers thirty lines apart. -->
            <table v-if="statements[openStatement].footing" class="mini lc-foot">
              <tbody>
                <tr class="tot grand">
                  <td>{{ statements[openStatement].footing.label }}</td>
                  <td class="num">{{ fmt(statements[openStatement].footing.amount) }}</td>
                </tr>
                <tr v-if="statements[openStatement].footing.ties === false">
                  <td colspan="2" class="muted small warn">
                    Does not tie to {{ statements[openStatement].footing.compare_label.toLowerCase() }}
                    of {{ fmt(statements[openStatement].footing.compare_amount) }}
                    — a difference of {{ fmt(statements[openStatement].footing.difference) }}.
                  </td>
                </tr>
                <tr v-else-if="statements[openStatement].footing.ties">
                  <td colspan="2" class="muted small">
                    Ties to {{ statements[openStatement].footing.compare_label.toLowerCase() }}
                    of {{ fmt(statements[openStatement].footing.compare_amount) }}.
                  </td>
                </tr>
              </tbody>
            </table>
          </template>
          <!-- schedule of investments -->
          <table v-else-if="openStatement === 'soi'" class="mini">
            <thead><tr><th>Investment</th><th class="num">Interest</th>
              <th class="num">Per relationships</th><th class="num">Cost</th>
              <th class="num">Fair value</th></tr></thead>
            <tbody>
              <tr v-for="l in statements.soi.lines" :key="l.related_entity"
                  :class="{ warn: l.ownership_disagrees }">
                <td>{{ l.name || l.related_entity }}</td>
                <td class="num">{{ l.ownership_pct?.toFixed(4) }}%</td>
                <td class="num">{{ l.ownership_pct_relationships?.toFixed(2) ?? '—' }}%</td>
                <td class="num">{{ fmt(l.cost) }}</td>
                <td class="num">{{ fmt(l.fair_value) }}</td>
              </tr>
            </tbody>
          </table>
          <!-- members' capital -->
          <table v-else-if="openStatement === 'members_capital'" class="mini">
            <thead><tr><th>Movement</th>
              <th v-for="m in statements.members_capital.members" :key="m.InvestorID" class="num">
                {{ m.InvestorName || m.InvestorID }}</th>
              <th class="num">Total</th></tr></thead>
            <tbody>
              <tr v-for="(r, i) in statements.members_capital.rows" :key="i"
                  :class="{ tot: r.kind !== 'movement' }">
                <td>{{ r.label }}</td>
                <td v-for="m in statements.members_capital.members" :key="m.InvestorID" class="num">
                  {{ fmt(r.by_member[m.InvestorID]) }}</td>
                <td class="num">{{ fmt(r.total) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <!-- Checklist on the left, that step's evidence on the right. -->
      <section class="work">
        <div class="steps">
          <h3>Close checklist</h3>
          <button v-for="s in detail.steps" :key="s.key" class="step-row"
                  :class="{ active: activeStep === s.key, late: s.overdue, done: s.done }"
                  @click="selectStep(s.key)">
            <span class="tick" :class="{ on: s.done }"
                  @click.stop="toggleStep(s)"
                  :title="s.done ? 'Mark not done' : 'Mark done'">{{ s.done ? '✓' : '' }}</span>
            <span class="step-text">
              <span class="step-label">{{ s.label }}</span>
              <span class="step-meta">
                <span class="owner-tag">{{ s.owner }}</span>
                <span v-if="s.due_date" :class="{ overdue: s.overdue }">due {{ s.due_date }}</span>
                <span v-else class="muted">no deadline</span>
              </span>
            </span>
          </button>

          <h3 style="margin-top:14px">Approval</h3>
          <div class="approve-row">
            <button class="btn" @click="act('submit')">Submit</button>
            <button class="btn" @click="act('approve_manager')">Manager</button>
            <button class="btn" @click="act('approve_cfo')">CFO</button>
            <button class="btn warn" @click="act('return_to_preparer')">Return</button>
          </div>
          <input v-model="returnNote" class="note-input"
                 placeholder="Note — required when returning" />
        </div>

        <div class="evidence">
          <div v-if="evidenceLoading" class="muted">Loading evidence…</div>
          <template v-else-if="evidence">
            <h3>{{ evidence.step.label }}</h3>
            <p class="guidance">{{ evidence.guidance }}</p>

            <div v-if="evidence.checks.length" class="checks">
              <div v-for="(c, i) in evidence.checks" :key="i" class="check" :class="c.status">
                <span class="pill">{{ c.status === 'pass' ? '✓'
                                    : c.status === 'fail' ? '!' : '·' }}</span>
                <span class="c-label">{{ c.label }}</span>
                <span class="c-value">{{ c.value }}</span>
                <span v-if="c.detail" class="c-detail">{{ c.detail }}</span>
              </div>
            </div>

            <div v-for="(t, i) in evidence.tables" :key="i" class="ev-table">
              <h4>{{ t.title }}
                <span class="muted small">{{ t.row_count }} row(s)</span>
                <span v-if="t.note" class="muted small">· {{ t.note }}</span>
              </h4>
              <div class="table-scroll">
                <table class="mini">
                  <thead><tr><th v-for="c in t.columns" :key="c">{{ c }}</th></tr></thead>
                  <tbody>
                    <tr v-for="(r, j) in t.rows" :key="j">
                      <td v-for="c in t.columns" :key="c"
                          :class="{ num: typeof r[c] === 'number' }">{{ fmt(r[c]) }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <p v-if="t.truncated" class="muted small">
                Showing the first {{ t.rows.length }} of {{ t.row_count }} — the full
                set is in the downloaded package.
              </p>
            </div>

            <!-- Attach evidence against the assertion being made. -->
            <div v-if="evidence.exhibit_slots.length" class="ev-exhibits">
              <h4>Supporting exhibits for this step</h4>
              <div class="upload">
                <select v-model="uploadSlot" class="sel">
                  <option v-for="k in evidence.exhibit_slots" :key="k" :value="k">
                    {{ slotLabel(k) }}</option>
                </select>
                <input v-model="uploadCaption" placeholder="Caption (optional)" />
                <input type="file" @change="uploadExhibit" />
              </div>
              <table v-if="stepExhibits.length" class="mini">
                <tbody>
                  <tr v-for="e in stepExhibits" :key="e.id">
                    <td>{{ slotLabel(e.slot_key) }}</td>
                    <td><a :href="exhibitUrl(e.id)">{{ e.filename }}</a>
                      <span v-if="e.caption" class="muted small"> · {{ e.caption }}</span></td>
                    <td class="num">{{ ((e.size_bytes || 0) / 1024).toFixed(1) }} KB</td>
                    <td><button class="link danger" @click="removeExhibit(e.id)">remove</button></td>
                  </tr>
                </tbody>
              </table>
              <p v-else class="muted small">Nothing attached for this step yet.</p>
            </div>
          </template>
          <div v-else class="muted">Select a step to see what it needs.</div>
        </div>
      </section>

      <section class="activity">
        <h3>Activity</h3>
        <div v-for="(e, i) in detail.events" :key="i" class="event">
          <span class="muted">{{ e.created_at }}</span>
          <strong>{{ e.action }}</strong>
          <span v-if="e.from_state">{{ e.from_state }} → {{ e.to_state }}</span>
          <span>· {{ e.actor }}</span>
          <span v-if="e.note" class="note">“{{ e.note }}”</span>
        </div>
        <p v-if="!detail.events.length" class="muted small">Nothing recorded yet.</p>
      </section>
    </div>
    </div>
  </div>
</template>

<style scoped>
/* min-width:0 because .page is a flex child: without it the page grows to the
   widest table and the whole BODY scrolls sideways. */
.page { padding: 18px 22px; min-width: 0; max-width: 100%; }
.page-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; }
h1 { margin: 0; font-size: 22px; }
h3 { margin: 0 0 8px; font-size: 14px; }
h4 { margin: 12px 0 4px; font-size: 12.5px; font-weight: 600; }
.subtitle { margin: 2px 0 0; color: var(--color-text-secondary); font-size: 13px; }
.head-actions { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.sel, input[type=date], .note-input, input:not([type]) {
  padding: 5px 8px; border: 1px solid var(--color-border); border-radius: 4px;
  background: var(--color-surface); color: var(--color-text); font-size: 13px; }
.btn { padding: 5px 12px; border: 1px solid var(--color-border); border-radius: 4px;
  background: var(--color-surface); color: var(--color-text); cursor: pointer; font-size: 13px; }
.btn.primary { background: #1f3864; color: #fff; border-color: #1f3864; }
.btn.warn { border-color: #c47b00; color: #8a5a00; }
.new-cycle { display: flex; gap: 8px; align-items: center; margin: 12px 0; flex-wrap: wrap; }
.hint { color: var(--color-text-secondary); font-size: 12px; }
.banner { margin: 10px 0; padding: 8px 12px; border-radius: 5px; font-size: 13px; }
.banner.ok { background: #eaf6ec; border: 1px solid #9ccfa6; color: #205c2c; }
.banner.err { background: #fdecea; border: 1px solid #e0a09a; color: #7a231b; }
.placeholder { margin: 24px 0; color: var(--color-text-secondary); }

.grid-wrap { overflow-x: auto; margin-top: 14px; border: 1px solid var(--color-border);
  border-radius: 6px; }
.grid { border-collapse: collapse; font-size: 12px; width: 100%; }
.grid th, .grid td { border: 1px solid var(--color-border); padding: 5px 7px; vertical-align: top; }
.grid thead th { background: #f4f6fa; position: sticky; top: 0; }
.grid tr.open td { background: #eef3fb; }
.sticky-l { position: sticky; left: 0; background: var(--color-surface); z-index: 2; min-width: 150px; }
.step-col { min-width: 108px; }
.step-name { font-weight: 600; }
.step-owner { color: var(--color-text-secondary); font-size: 10px; text-transform: uppercase; }
.due { width: 100%; margin-top: 3px; font-size: 11px; padding: 2px 4px; }
.due-ro { font-size: 11px; color: var(--color-text-secondary); margin-top: 3px; }
.ent-name { color: var(--color-text-secondary); font-size: 11px; }
.cell { text-align: center; font-weight: 700; }
.cell.done { background: #eaf6ec; color: #2c7a3d; }
.cell.late { background: #fdecea; color: #b3261e; }
.num { text-align: right; }
.bar { height: 4px; background: #e6e9ef; border-radius: 2px; margin-top: 3px; }
.bar i { display: block; height: 100%; background: #1f3864; border-radius: 2px; }
.chip { padding: 1px 7px; border-radius: 9px; font-size: 11px; white-space: nowrap; }
.st-grey { background: #eceff3; color: #555; }
.st-blue { background: #e3edfa; color: #1f3864; }
.st-amber { background: #fdf3e0; color: #8a5a00; }
.st-teal { background: #e0f2f1; color: #00695c; }
.st-green { background: #eaf6ec; color: #2c7a3d; }
.st-red { background: #fdecea; color: #b3261e; }

.drawer { margin-top: 20px; border: 1px solid var(--color-border); border-radius: 6px;
  background: var(--color-surface); }
.drawer-head { display: flex; justify-content: space-between; align-items: flex-start;
  gap: 12px; padding: 14px 16px; border-bottom: 1px solid var(--color-border); flex-wrap: wrap; }
.drawer-head h2 { margin: 0; font-size: 17px; }
.sub { color: var(--color-text-secondary); font-size: 12px; margin-top: 3px;
  display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }

.statements { padding: 14px 16px; border-bottom: 1px solid var(--color-border);
  background: #fafbfd; }
.sec-head { display: flex; justify-content: space-between; align-items: baseline;
  gap: 12px; flex-wrap: wrap; }
.stmt-cards { display: flex; gap: 8px; flex-wrap: wrap; }
.stmt-card { flex: 1 1 180px; text-align: left; padding: 8px 10px; cursor: pointer;
  border: 1px solid var(--color-border); border-left: 4px solid #c9ced8;
  border-radius: 5px; background: var(--color-surface); }
.stmt-card.ok { border-left-color: #2c7a3d; }
.stmt-card.bad { border-left-color: #b3261e; }
.stmt-card.active { background: #eef3fb; }
.stmt-title { font-weight: 600; font-size: 13px; }
.stmt-tie { font-size: 11.5px; color: var(--color-text-secondary); margin-top: 2px; }
.stmt-body { margin-top: 12px; background: var(--color-surface); padding: 10px 12px;
  border: 1px solid var(--color-border); border-radius: 5px; }

.work { display: grid; grid-template-columns: minmax(240px, 320px) minmax(0, 1fr);
  gap: 18px; padding: 14px 16px; }
@media (max-width: 900px) { .work { grid-template-columns: 1fr; } }
.steps { min-width: 0; }
.step-row { display: flex; gap: 8px; align-items: flex-start; width: 100%; text-align: left;
  padding: 6px 8px; border: 1px solid transparent; border-radius: 5px;
  background: none; cursor: pointer; font-size: 13px; color: var(--color-text); }
.step-row:hover { background: #f4f6fa; }
.step-row.active { background: #eef3fb; border-color: #c3d3ea; }
.step-row.late { box-shadow: inset 3px 0 0 #b3261e; }
.step-row.done .step-label { color: var(--color-text-secondary); }
.tick { flex: 0 0 auto; width: 16px; height: 16px; border: 1px solid var(--color-border);
  border-radius: 3px; display: inline-flex; align-items: center; justify-content: center;
  font-size: 11px; margin-top: 1px; background: var(--color-surface); }
.tick.on { background: #eaf6ec; border-color: #9ccfa6; color: #2c7a3d; }
.step-text { display: flex; flex-direction: column; min-width: 0; }
.step-meta { display: flex; gap: 8px; font-size: 11px; color: var(--color-text-secondary); }
.owner-tag { text-transform: uppercase; font-size: 10px; background: #eceff3;
  padding: 0 4px; border-radius: 3px; }
.overdue { color: #b3261e; font-weight: 600; }

.evidence { min-width: 0; }
.guidance { margin: 0 0 10px; font-size: 13px; color: var(--color-text-secondary); }
.checks { display: grid; gap: 4px; margin-bottom: 10px; }
.check { display: flex; gap: 8px; align-items: baseline; font-size: 12.5px;
  padding: 4px 8px; border-radius: 4px; background: #f4f6fa; }
.check.pass { background: #eaf6ec; }
.check.fail { background: #fdecea; }
.check .pill { flex: 0 0 16px; text-align: center; font-weight: 700; }
.check.pass .pill { color: #2c7a3d; }
.check.fail .pill { color: #b3261e; }
.c-label { flex: 1 1 auto; }
.c-value { font-weight: 600; }
.c-detail { color: var(--color-text-secondary); font-size: 11.5px; }

.ev-table { margin-top: 10px; }
.table-scroll { overflow-x: auto; max-height: 340px; overflow-y: auto; }
.mini { width: 100%; border-collapse: collapse; font-size: 12px; }
.mini th, .mini td { border-bottom: 1px solid var(--color-border); padding: 3px 6px;
  text-align: left; white-space: nowrap; }
.mini thead th { position: sticky; top: 0; background: #f4f6fa; }
.mini td.num, .mini th.num { text-align: right; }
.mini tr.tot td { font-weight: 700; border-top: 1px solid var(--color-border); }
.mini tr.warn { background: #fdf3e0; }
.ev-exhibits { margin-top: 14px; padding-top: 10px; border-top: 1px solid var(--color-border); }
.upload { display: flex; gap: 8px; align-items: center; margin: 6px 0; flex-wrap: wrap; }

.activity { padding: 12px 16px; border-top: 1px solid var(--color-border); }
.event { font-size: 12px; padding: 2px 0; display: flex; gap: 8px; flex-wrap: wrap; }
.event .note { font-style: italic; }
.link { background: none; border: none; color: #1f3864; cursor: pointer; padding: 0;
  font-weight: 600; font-size: 12px; }
.link.danger { color: #b3261e; font-weight: 400; }
.approve-row { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 6px; }
.note-input { width: 100%; }
.muted { color: var(--color-text-secondary); }
.small { font-size: 12px; }

/* == Tabs, tracker, workbench picker ============================== */
.tabs { display: flex; gap: 2px; margin-top: 16px; border-bottom: 1px solid #dde3ec; }
.tab {
  background: none; border: none; cursor: pointer;
  padding: 9px 16px; font-size: 13px; font-weight: 600; color: #7a8394;
  border-bottom: 2px solid transparent; margin-bottom: -1px;
}
.tab.active { color: #1d4e7e; border-bottom-color: #1d4e7e; }
.tab-sub { font-weight: 500; color: #9aa3b2; margin-left: 6px; font-size: 11px; }

.tracker-tab, .bench-tab { margin-top: 14px; }

.sched-bar {
  display: flex; align-items: center; gap: 18px; flex-wrap: wrap;
  border: 1px solid #e2e6ee; border-radius: 8px; background: #fff;
  padding: 10px 14px; margin-bottom: 12px; font-size: 12px;
}
.sched-bar .spacer { flex: 1 1 auto; }
.sched-stat b { font-size: 15px; color: #1d4e7e; }
.sched-stat.bad b { color: #b4232a; }
.sched-bar .chk { display: flex; align-items: center; gap: 6px; cursor: pointer; }
.sel.sm { font-size: 12px; padding: 4px 6px; }
.sel.wide { min-width: 460px; }

.note {
  font-size: 12px; color: #5a6475; background: #f7f9fc;
  border-left: 3px solid #c8d2e0; padding: 7px 12px; margin-bottom: 8px;
  border-radius: 0 4px 4px 0;
}
.note.warn { border-left-color: #d9a441; background: #fdf8ef; }
.note .oos { display: inline-block; margin-right: 12px; }
.note .muted { color: #8a93a4; margin-top: 3px; }

/* The grid is wide by construction - 4 fixed columns plus 4 deliverables of
   target + stages. It scrolls in both directions with the order and the entity
   pinned, because those two are what tells you which row you are reading. */
.sched-wrap { max-height: 70vh; overflow: auto; }
table.grid.sched { font-size: 11.5px; border-collapse: separate; border-spacing: 0; }
table.grid.sched th, table.grid.sched td {
  border-bottom: 1px solid #edf0f5; padding: 3px 6px; white-space: nowrap;
}
/* THE WHOLE THEAD STICKS AS ONE BLOCK.
   It used to be per-row: row 1 at `top: 0` and row 2 at a second offset that
   had to equal row 1's height. That offset was first hardcoded at 34px and then
   measured at runtime, and BOTH were wrong — measured in a standalone repro of
   this exact markup, row 1 is 22px and row 2 is 34px, so the 34px fallback put
   row 2 twelve pixels too low, over the first data row. Jim reported the
   overlap twice, the second time after the "fix".
   Sticking the thead removes the arithmetic instead of correcting it: two rows
   that move together cannot be mispositioned relative to each other, at any
   font size or zoom, with no JavaScript. */
table.grid.sched thead { position: sticky; top: 0; z-index: 3; }
table.grid.sched thead th {
  background: #f4f6fa;
  border-bottom: 1px solid #dde3ec;
}
th.grp {
  text-align: center; font-size: 11px; letter-spacing: .02em;
  border-left: 2px solid #dde3ec !important;
}
th.stg-h, th.tgt-h { font-weight: 600; font-size: 10.5px; text-align: center; }
th.tgt-h { border-left: 2px solid #dde3ec !important; }
th .owner { font-weight: 500; color: #9aa3b2; font-size: 9.5px; }
.sticky-l { position: sticky; left: 0; background: #fff; z-index: 2; }
.sticky-e { position: sticky; left: 40px; background: #fff; z-index: 2; }
thead .sticky-l, thead .sticky-e {
  position: sticky; z-index: 4; background: #f4f6fa;
}
thead .sticky-l { left: 0; }
thead .sticky-e { left: 40px; }
/* Wide enough for four digits and no wider: 11.5px digits are ~7px each, so
   28px of glyph plus the input's own padding. Was 46px and looked like a column
   with nothing in it (Jim, Sep 17 2026). */
td.ord, th.ord {
  width: 40px; min-width: 40px; max-width: 40px; text-align: center;
}
.grp-row td {
  background: #eef2f8; font-weight: 700; color: #33415a;
  font-size: 11px; padding: 4px 8px !important;
}
.ord-in { width: 32px; text-align: center; -moz-appearance: textfield; }
.ord-in::-webkit-outer-spin-button,
.ord-in::-webkit-inner-spin-button { -webkit-appearance: none; margin: 0; }
.txt-in { width: 118px; }
.ini-sel {
  width: 52px; font: inherit; padding: 1px 2px; text-align: center;
  border: 1px solid transparent; background: transparent; border-radius: 3px;
  color: inherit;
}
.ini-sel:hover { border-color: #dde3ec; }
.ini-sel:focus { border-color: #1d4e7e; background: #fff; outline: none; }
.date-in { width: 112px; }
.ord-in, .txt-in, .ini-in, .date-in {
  border: 1px solid transparent; background: transparent; border-radius: 3px;
  padding: 2px 4px; font: inherit; color: inherit;
}
.ord-in:hover, .txt-in:hover, .ini-in:hover, .date-in:hover { border-color: #dde3ec; }
.ord-in:focus, .txt-in:focus, .ini-in:focus, .date-in:focus {
  border-color: #1d4e7e; background: #fff; outline: none;
}
td.tgt { border-left: 2px solid #edf0f5; text-align: center; }
td.tgt.late { background: #fdecec; }
td.tgt.ok { background: #f2f9f3; }
td.sign { text-align: center; position: relative; min-width: 62px; }
td.sign.signed { background: #f2f9f3; }
td.sign .when { color: #2c6e3f; font-weight: 600; font-size: 10.5px; }
.sign-btn {
  border: 1px dashed #ccd4e0; background: none; color: #aab3c2;
  border-radius: 3px; width: 20px; height: 18px; line-height: 1; cursor: pointer;
}
.sign-btn:hover { border-color: #1d4e7e; color: #1d4e7e; background: #eef4fb; }
td.sign .x {
  position: absolute; top: 0; right: 1px; border: none; background: none;
  color: #7aa98a; cursor: pointer; font-size: 13px; line-height: 1;
  padding: 0 2px; font-weight: 700;
}
td.sign .x:hover { color: #b4232a; }
.legend { font-size: 11px; color: #8a93a4; margin-top: 8px; }

.bench-pick {
  display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
  border: 1px solid #e2e6ee; border-radius: 8px; background: #fff;
  padding: 12px 14px; margin-bottom: 14px;
}
.bench-pick label { font-size: 12px; font-weight: 600; color: #5a6475; }
.btn.nav { padding: 4px 10px; font-size: 14px; line-height: 1; }
.bench-tab .drawer { margin-top: 0; }


.lc-foot { margin-top: 10px; }
.lc-foot .tot.grand td {
  border-top: 1px solid #33415a; border-bottom: 3px double #33415a;
  font-weight: 700; padding-top: 4px;
}
.small.warn { color: #b4232a; }


td.prop.inferred .txt-in { color: #5a6475; font-style: italic; }
.inf-mark {
  color: #8a6d35; font-weight: 700; font-size: 11px; cursor: help;
  margin-left: 1px; vertical-align: super;
}

</style>
