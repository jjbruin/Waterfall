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
 */
import { ref, computed, onMounted } from 'vue'
import api from '../api/client'
import { useAuthStore } from '../stores/auth'

const auth = useAuthStore()

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

const isAdmin = computed(() => auth.user?.role === 'admin')
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
  if (cycleId.value) await loadTracker()
}

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
  await api.put(`/api/workpapers/cycles/${cycleId.value}/steps`, {
    step_key: stepKey, due_date: due || null,
  })
  await loadTracker()
}

async function openPackage(id: number) {
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
        <button v-if="isAdmin" class="btn" @click="sync" :disabled="!cycleId">Sync entities</button>
        <button v-if="isAdmin" class="btn primary" @click="showNewCycle = !showNewCycle">
          New close cycle
        </button>
      </div>
    </div>

    <div v-if="showNewCycle" class="new-cycle">
      <input v-model="newLabel" placeholder="Label, e.g. Q2 2026" />
      <input v-model="newEnd" type="date" />
      <button class="btn primary" @click="createCycle">Create</button>
      <span class="hint">Creates a package for every entity tagged REP, and an
        empty deadline for each step of the close.</span>
    </div>

    <div v-if="msg" class="banner ok">{{ msg }}</div>
    <div v-if="error" class="banner err">{{ error }}</div>
    <div v-if="loading" class="placeholder">Loading…</div>

    <template v-if="tracker && !loading">
      <div v-if="!tracker.packages.length" class="placeholder">
        No packages in this cycle. If entities are tagged REP in MRI, press
        <strong>Sync entities</strong>; if none are, nothing is due.
      </div>

      <div v-else class="grid-wrap">
        <table class="grid">
          <thead>
            <tr>
              <th class="sticky-l">Entity</th>
              <th>Status</th>
              <th class="num">Progress</th>
              <th class="num">Exhibits</th>
              <th v-for="s in tracker.steps" :key="s.key" class="step-col" :title="s.label">
                <div class="step-name">{{ s.label }}</div>
                <div class="step-owner">{{ s.owner }}</div>
                <input v-if="isAdmin" class="due" type="date" :value="s.due_date || ''"
                       @change="setDue(s.key, ($event.target as HTMLInputElement).value)"
                       title="CFO deadline for this step" />
                <div v-else class="due-ro">{{ s.due_date || '—' }}</div>
              </th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="p in tracker.packages" :key="p.id"
                :class="{ open: detail?.package?.id === p.id }">
              <td class="sticky-l">
                <button class="link" @click="openPackage(p.id)">{{ p.entityid }}</button>
                <div class="ent-name">{{ p.entity_name }}</div>
              </td>
              <td><span class="chip" :class="stateClass(p.state)">{{ p.state_label }}</span></td>
              <td class="num">
                {{ p.steps_complete }}/{{ p.steps_total }}
                <div class="bar"><i :style="{ width: (100 * p.steps_complete / p.steps_total) + '%' }" /></div>
              </td>
              <td class="num">{{ p.exhibit_count }}</td>
              <td v-for="s in p.steps" :key="s.key" class="cell"
                  :class="{ done: s.done, late: s.overdue }"
                  :title="s.done ? `${s.completed_by} — ${s.completed_at}`
                                 : (s.due_date ? `due ${s.due_date}` : 'no deadline set')">
                {{ s.done ? '✓' : (s.overdue ? '!' : '') }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </template>

    <!-- ── Package ─────────────────────────────────────────────────── -->
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
          <button class="btn" @click="detail = null">Close</button>
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
</style>
