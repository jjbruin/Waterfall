<script setup lang="ts">
/**
 * Accounting workpaper packages — the quarterly close.
 *
 * Two questions this screen has to answer at a glance, because they are the
 * ones asked in a close meeting:
 *   "where is every package right now?"   -> the tracker grid
 *   "what is late, and whose is it?"      -> overdue counts and the red cells
 *
 * The population is not editable here. An entity gets a package because
 * accounting tagged it ENTGRPID='REP' in MRI; Sync re-reads that tag rather
 * than offering an add button, so the app and MRI cannot disagree about who
 * is in scope.
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
const detailLoading = ref(false)
const preview = ref<any>(null)
const showNewCycle = ref(false)
const newLabel = ref('')
const newEnd = ref('')
const uploadSlot = ref('other')
const uploadCaption = ref('')
const returnNote = ref('')

const isAdmin = computed(() => auth.user?.role === 'admin')

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
  detailLoading.value = true
  preview.value = null
  try {
    detail.value = (await api.get(`/api/workpapers/packages/${id}`)).data
  } finally {
    detailLoading.value = false
  }
}

async function loadPreview() {
  if (!detail.value) return
  preview.value = (await api.get(
    `/api/workpapers/packages/${detail.value.package.id}/preview`)).data
}

async function toggleStep(step: any) {
  await api.put(`/api/workpapers/packages/${detail.value.package.id}/steps`, {
    step_key: step.key, done: !step.done,
  })
  await openPackage(detail.value.package.id)
  await loadTracker()
}

async function act(action: string) {
  try {
    await api.post(`/api/workpapers/packages/${detail.value.package.id}/transition`, {
      action, note: returnNote.value,
    })
    returnNote.value = ''
    await openPackage(detail.value.package.id)
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
    await openPackage(detail.value.package.id)
    await loadTracker()
    flash('Exhibit attached')
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  }
}

async function removeExhibit(id: number) {
  if (!confirm('Remove this exhibit from the package?')) return
  await api.delete(`/api/workpapers/exhibits/${id}`)
  await openPackage(detail.value.package.id)
  await loadTracker()
}

function exhibitUrl(id: number) { return `/api/workpapers/exhibits/${id}` }

async function downloadPackage(id: number, name: string) {
  const res = await api.get(`/api/workpapers/packages/${id}/download`, { responseType: 'blob' })
  const url = URL.createObjectURL(res.data)
  const a = document.createElement('a')
  a.href = url
  a.download = name
  a.click()
  URL.revokeObjectURL(url)
}

const stateClass = (s: string) => ({
  not_started: 'st-grey', in_progress: 'st-blue', submitted: 'st-amber',
  manager_approved: 'st-teal', cfo_approved: 'st-green', returned: 'st-red',
} as Record<string, string>)[s] || 'st-grey'

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

      <!-- Tracker grid: entity x step -->
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
            <tr v-for="p in tracker.packages" :key="p.id">
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

    <!-- Package detail -->
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
          <button class="btn" @click="loadPreview">Preview figures</button>
          <button class="btn primary"
                  @click="downloadPackage(detail.package.id,
                          `${detail.package.entityid} - WP - ${detail.package.period_end}.xlsx`)">
            Download package
          </button>
          <button class="btn" @click="detail = null">Close</button>
        </div>
      </div>

      <div class="drawer-body">
        <section>
          <h3>Close checklist</h3>
          <div v-for="s in detail.steps" :key="s.key" class="step-row" :class="{ late: s.overdue }">
            <label>
              <input type="checkbox" :checked="s.done" @change="toggleStep(s)" />
              <span class="step-label">{{ s.label }}</span>
            </label>
            <span class="step-meta">
              <span class="owner-tag">{{ s.owner }}</span>
              <span v-if="s.due_date" :class="{ overdue: s.overdue }">due {{ s.due_date }}</span>
              <span v-else class="muted">no deadline</span>
              <span v-if="s.done" class="muted">· {{ s.completed_by }} {{ s.completed_at }}</span>
            </span>
          </div>
        </section>

        <section>
          <h3>Supporting exhibits</h3>
          <p class="muted small">
            Attached here, placed in the workbook on download. Spreadsheets and CSVs
            become tabs, images are embedded; anything else is listed on the Exhibits
            tab and travels with the file.
          </p>
          <div class="upload">
            <select v-model="uploadSlot" class="sel">
              <option v-for="s in detail.slots" :key="s.key" :value="s.key">{{ s.label }}</option>
            </select>
            <input v-model="uploadCaption" placeholder="Caption (optional)" />
            <input type="file" @change="uploadExhibit" />
          </div>
          <table v-if="detail.exhibits.length" class="mini">
            <thead><tr><th>Slot</th><th>File</th><th class="num">KB</th><th>By</th><th></th></tr></thead>
            <tbody>
              <tr v-for="e in detail.exhibits" :key="e.id">
                <td>{{ (detail.slots.find((s: any) => s.key === e.slot_key) || {}).label || e.slot_key }}</td>
                <td><a :href="exhibitUrl(e.id)">{{ e.filename }}</a>
                  <div v-if="e.caption" class="muted small">{{ e.caption }}</div></td>
                <td class="num">{{ ((e.size_bytes || 0) / 1024).toFixed(1) }}</td>
                <td>{{ e.uploaded_by }}</td>
                <td><button class="link danger" @click="removeExhibit(e.id)">remove</button></td>
              </tr>
            </tbody>
          </table>
          <p v-else class="muted small">No exhibits attached yet.</p>
        </section>

        <section>
          <h3>Approval</h3>
          <div class="approve-row">
            <button class="btn" @click="act('submit')">Submit for review</button>
            <button class="btn" @click="act('approve_manager')">Manager approve</button>
            <button class="btn" @click="act('approve_cfo')">CFO approve</button>
            <button class="btn warn" @click="act('return_to_preparer')">Return to preparer</button>
            <button v-if="isAdmin" class="btn" @click="act('reopen')">Reopen</button>
          </div>
          <input v-model="returnNote" class="note-input"
                 placeholder="Note — required when returning a package" />
        </section>

        <section v-if="preview">
          <h3>Figures</h3>
          <p class="muted small">
            Trial balance through {{ preview.periods.ytd_last }} —
            {{ preview.trial_balance_rows }} accounts,
            <strong>{{ preview.unmapped_accounts.length }}</strong> not mapped to a
            statement line.
          </p>
          <div class="table-scroll">
          <table class="mini">
            <thead><tr><th>Account</th><th>Name</th><th>FS line</th>
              <th class="num">YTD beginning</th><th class="num">Change</th>
              <th class="num">Ending</th></tr></thead>
            <tbody>
              <tr v-for="r in preview.trial_balance.slice(0, 40)" :key="r.acctnum"
                  :class="{ unmapped: !r.fs_line }">
                <td>{{ r.acctnum }}</td><td>{{ r.acctname }}</td>
                <td>{{ r.fs_line || '— unmapped —' }}</td>
                <td class="num">{{ r.ytd_beginning?.toLocaleString() }}</td>
                <td class="num">{{ r.ytd_change?.toLocaleString() }}</td>
                <td class="num">{{ r.ytd_ending?.toLocaleString() }}</td>
              </tr>
            </tbody>
          </table>
          </div>
        </section>

        <section>
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
   widest table and the whole BODY scrolls sideways, dragging the drawer off
   screen with it. With it, the grid scrolls inside .grid-wrap as intended. */
.page { padding: 18px 22px; min-width: 0; max-width: 100%; }
.page-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; }
h1 { margin: 0; font-size: 22px; }
.subtitle { margin: 2px 0 0; color: var(--color-text-secondary); font-size: 13px; }
.head-actions { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.sel, input[type=date], input[type=text], .note-input, input:not([type]) {
  padding: 5px 8px; border: 1px solid var(--color-border); border-radius: 4px;
  background: var(--color-surface); color: var(--color-text); font-size: 13px;
}
.btn { padding: 5px 12px; border: 1px solid var(--color-border); border-radius: 4px;
  background: var(--color-surface); color: var(--color-text); cursor: pointer; font-size: 13px; }
.btn.primary { background: #1f3864; color: #fff; border-color: #1f3864; }
.btn.warn { border-color: #c47b00; color: #8a5a00; }
.btn:disabled { opacity: .5; cursor: default; }
.new-cycle { display: flex; gap: 8px; align-items: center; margin: 12px 0; flex-wrap: wrap; }
.hint { color: var(--color-text-secondary); font-size: 12px; }
.banner { margin: 10px 0; padding: 8px 12px; border-radius: 5px; font-size: 13px; }
.banner.ok { background: #eaf6ec; border: 1px solid #9ccfa6; color: #205c2c; }
.banner.err { background: #fdecea; border: 1px solid #e0a09a; color: #7a231b; }
.placeholder { margin: 24px 0; color: var(--color-text-secondary); }

.grid-wrap { overflow-x: auto; margin-top: 14px; border: 1px solid var(--color-border); border-radius: 6px; }
.grid { border-collapse: collapse; font-size: 12px; width: 100%; }
.grid th, .grid td { border: 1px solid var(--color-border); padding: 5px 7px; vertical-align: top; }
.grid thead th { background: #f4f6fa; position: sticky; top: 0; }
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
.drawer-body { padding: 14px 16px; display: grid; gap: 20px; }
section h3 { margin: 0 0 6px; font-size: 14px; }
.step-row { display: flex; justify-content: space-between; gap: 12px; padding: 4px 6px;
  border-bottom: 1px solid #f0f1f4; font-size: 13px; }
.step-row.late { background: #fff7f6; }
.step-label { margin-left: 6px; }
.step-meta { display: flex; gap: 10px; align-items: center; font-size: 11.5px;
  color: var(--color-text-secondary); }
.owner-tag { text-transform: uppercase; font-size: 10px; background: #eceff3;
  padding: 1px 5px; border-radius: 3px; }
.overdue { color: #b3261e; font-weight: 600; }
.muted { color: var(--color-text-secondary); }
.small { font-size: 12px; }
.upload { display: flex; gap: 8px; align-items: center; margin: 8px 0; flex-wrap: wrap; }
.table-scroll { overflow-x: auto; }
.mini { width: 100%; border-collapse: collapse; font-size: 12px; }
.mini th, .mini td { border-bottom: 1px solid var(--color-border); padding: 4px 6px; text-align: left; }
.mini th.num, .mini td.num { text-align: right; }
.mini tr.unmapped { background: #fff7f6; }
.link { background: none; border: none; color: #1f3864; cursor: pointer; padding: 0;
  font-weight: 600; font-size: 12px; }
.link.danger { color: #b3261e; font-weight: 400; }
.approve-row { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 8px; }
.note-input { width: 100%; }
.event { font-size: 12px; padding: 3px 0; display: flex; gap: 8px; flex-wrap: wrap; }
.event .note { font-style: italic; }
</style>
