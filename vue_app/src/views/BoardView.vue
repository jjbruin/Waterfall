<script setup lang="ts">
// Board section, Phase 0 (Oct 5 2026): meetings, the as-of date each schedule is
// drawn at, the narrative blocks, and -- for the admin account only -- who holds
// what. Phase 1 (Oct 7 2026) adds the first schedule views -- pages 26, 27 and
// 29-31 -- each read from the server at the as-of THIS MEETING carries for it;
// the screen computes nothing. Every gate here is ALSO enforced by the server;
// the screen only hides what would be refused.
import { ref, computed, onMounted } from 'vue'
import api from '../api/client'
import InvestmentMetricsTable from '@/components/reports/InvestmentMetricsTable.vue'

const me = ref<{ permissions: string[]; manages_access: boolean } | null>(null)
const tab = ref<'meetings' | 'access' | 'log'>('meetings')
const error = ref('')
const notice = ref('')
const can = (p: string) => !!me.value?.permissions.includes(p)

function fail(e: any) { error.value = e?.response?.data?.message || e?.response?.data?.error || String(e) }

// ---- meetings ----
const meetings = ref<any[]>([])
const meeting = ref<any>(null)
const draft = ref({ title: '', meeting_date: '', default_as_of: '' })
const editable = computed(() => !!meeting.value?.editable)

async function loadMeetings() {
  meetings.value = (await api.get('/api/board/meetings')).data.meetings
}
async function openMeeting(id: number) {
  error.value = ''; notice.value = ''
  meeting.value = (await api.get(`/api/board/meetings/${id}`)).data
}
async function createMeeting() {
  error.value = ''
  try {
    const m = (await api.post('/api/board/meetings', draft.value)).data
    draft.value = { title: '', meeting_date: '', default_as_of: '' }
    await loadMeetings(); meeting.value = m
  } catch (e) { fail(e) }
}
async function saveSchedule(s: any, patch: Record<string, any>) {
  error.value = ''; notice.value = ''
  try {
    const r = (await api.put(`/api/board/meetings/${meeting.value.id}/schedules`,
      { schedules: [{ key: s.key, ...patch }] })).data
    meeting.value = r
    if (r.warnings?.length) notice.value = r.warnings.join('; ')
  } catch (e) { fail(e); await openMeeting(meeting.value.id) }
}
// ---- schedule views (Phase 1) ----
const view = ref<any>(null)
const viewKey = ref('')
const viewLoading = ref(false)
const showNotes = ref(false)
async function openView(s: any) {
  error.value = ''
  if (viewKey.value === s.key) { viewKey.value = ''; view.value = null; return }
  viewKey.value = s.key; view.value = null; viewLoading.value = true; showNotes.value = false
  try {
    view.value = (await api.get(`/api/board/meetings/${meeting.value.id}/schedules/${s.key}/view`)).data
  } catch (e) { fail(e); viewKey.value = '' } finally { viewLoading.value = false }
}
// Dollars in, $ millions out. null is "the engine has no figure": a dash, never 0.
function m(v: number | null | undefined, dp = 1) {
  if (v === null || v === undefined) return '—'
  return '$' + (v / 1e6).toLocaleString('en-US', { minimumFractionDigits: dp, maximumFractionDigits: dp })
}
function pct(v: number | null | undefined, dp = 1) {
  return v === null || v === undefined ? '—' : (v * 100).toFixed(dp) + '%'
}

const narrativeDirty = ref<Record<string, boolean>>({})
async function saveNarrative(n: any) {
  error.value = ''
  try {
    meeting.value = (await api.put(`/api/board/meetings/${meeting.value.id}/narratives/${n.key}`,
      { body: n.body })).data
    narrativeDirty.value[n.key] = false
  } catch (e) { fail(e) }
}

// ---- access (admin account) ----
const access = ref<{ users: any[]; permissions: any[] } | null>(null)
async function loadAccess() { access.value = (await api.get('/api/board/access')).data }
async function putAccess(u: any, body: Record<string, any>) {
  error.value = ''
  try {
    const r = (await api.put(`/api/board/access/${u.id}`, body)).data
    Object.assign(u, r)
  } catch (e) { fail(e); await loadAccess() }
}
function togglePerm(u: any, key: string, on: boolean) {
  putAccess(u, { permissions: { [key]: on } })
}

const log = ref<any[]>([])
async function loadLog() { log.value = (await api.get('/api/board/audit')).data.entries }

async function show(t: 'meetings' | 'access' | 'log') {
  tab.value = t; error.value = ''
  try {
    if (t === 'access') await loadAccess()
    if (t === 'log') await loadLog()
  } catch (e) { fail(e) }
}

onMounted(async () => {
  try {
    me.value = (await api.get('/api/board/me')).data
    await loadMeetings()
  } catch (e) { fail(e) }
})
</script>

<template>
  <div class="board">
    <h1>Board</h1>
    <p class="muted">Meetings and the schedules each will carry. Every schedule is drawn from an
      engine the app already owns, at its own as-of date; the figures arrive phase by phase.</p>

    <div class="tabs">
      <button :class="{ on: tab === 'meetings' }" @click="show('meetings')">Meetings</button>
      <template v-if="me?.manages_access">
        <button :class="{ on: tab === 'access' }" @click="show('access')">Access</button>
        <button :class="{ on: tab === 'log' }" @click="show('log')">Access log</button>
      </template>
    </div>
    <div v-if="error" class="err">{{ error }}</div>
    <div v-if="notice" class="warn">{{ notice }}</div>

    <!-- MEETINGS -->
    <section v-if="tab === 'meetings'">
      <div class="row">
        <div class="list">
          <div v-for="m in meetings" :key="m.id" class="item" :class="{ on: meeting?.id === m.id }"
               @click="openMeeting(m.id)">
            <strong>{{ m.title }}</strong>
            <span class="muted">{{ m.meeting_date }} · {{ m.status }}</span>
          </div>
          <div v-if="!meetings.length" class="muted">No meetings yet.</div>
          <form v-if="can('board_build')" class="new" @submit.prevent="createMeeting">
            <strong>New meeting</strong>
            <label>Title <input v-model="draft.title" placeholder="e.g. Q1 2026 Board Meeting" /></label>
            <label>Meeting date <input v-model="draft.meeting_date" type="date" /></label>
            <label>Default as-of <input v-model="draft.default_as_of" type="date" /></label>
            <button class="btn-primary" type="submit">Create</button>
          </form>
        </div>

        <div v-if="meeting" class="detail">
          <h2>{{ meeting.title }}</h2>
          <p class="muted">Meeting {{ meeting.meeting_date }} · default as-of {{ meeting.default_as_of }}
            · {{ meeting.status }}<span v-if="!editable"> — no longer editable</span></p>

          <h3>Schedules</h3>
          <table class="grid">
            <thead><tr><th>In</th><th>Pages</th><th>Schedule</th><th>As of</th><th>Phase</th><th>Status</th><th></th></tr></thead>
            <tbody>
              <tr v-for="s in meeting.schedules" :key="s.key" :class="{ off: !s.included }">
                <td><input type="checkbox" :checked="s.included" :disabled="!editable || !can('board_edit')"
                           @change="saveSchedule(s, { included: ($event.target as HTMLInputElement).checked })" /></td>
                <td>{{ s.pages }}</td>
                <td>{{ s.title }}<div class="muted small">{{ s.source }}</div></td>
                <td>
                  <input type="date" :value="s.as_of" :disabled="!editable || !can('board_edit')"
                         @change="saveSchedule(s, { as_of: ($event.target as HTMLInputElement).value })" />
                  <div v-if="s.as_of_after_meeting" class="warn small">after the meeting date</div>
                </td>
                <td>{{ s.phase }}</td>
                <td>{{ s.status }}</td>
                <td><button v-if="s.view" class="btn-secondary" @click="openView(s)">
                  {{ viewKey === s.key ? 'Hide' : 'View' }}</button></td>
              </tr>
            </tbody>
          </table>

          <!-- SCHEDULE VIEW: figures from the server, at this meeting's as-of for the schedule -->
          <div v-if="viewKey" class="view">
            <div v-if="viewLoading" class="muted">Building the view from the engines…</div>
            <template v-else-if="view">
              <h3>p. {{ view.schedule.pages }} · {{ view.schedule.title }}
                <span class="muted small">as of {{ view.schedule.as_of }}</span></h3>

              <!-- p.26 -->
              <template v-if="view.key === 'capitalization'">
                <p class="muted small">$ in millions</p>
                <table class="grid fig">
                  <thead><tr><th></th><th>Deals</th><th>Properties</th><th>Total Gross Capitalization</th>
                    <th>PSC Capital</th><th>3rd Party Capital</th><th>Total Net Pref. Equity*</th></tr></thead>
                  <tbody>
                    <tr v-for="l in view.lines" :key="l.key">
                      <td>{{ l.label }}</td><td>{{ l.deals }}</td><td>{{ l.properties }}</td>
                      <td>{{ m(l.gross_cap, 2) }}</td><td>{{ m(l.psc) }}</td><td>{{ m(l.third_party) }}</td>
                      <td>{{ m(l.total) }}</td>
                    </tr>
                    <tr class="tot"><td>{{ view.total.label }}</td><td>{{ view.total.deals }}</td>
                      <td>{{ view.total.properties }}</td><td>{{ m(view.total.gross_cap, 2) }}</td>
                      <td>{{ m(view.total.psc) }}</td><td>{{ m(view.total.third_party) }}</td>
                      <td>{{ m(view.total.total) }}</td></tr>
                  </tbody>
                </table>
                <h4>3rd Party Capital Sources</h4>
                <table class="grid fig narrow">
                  <thead><tr><th>Investor</th><th>Current AUM**</th><th></th></tr></thead>
                  <tbody>
                    <tr v-for="x in view.third_party_sources" :key="x.group">
                      <td>{{ x.label }}</td><td>{{ m(x.amount) }}</td><td>{{ pct(x.share) }}</td></tr>
                    <tr class="tot"><td>TOTAL</td><td>{{ m(view.third_party_total) }}</td><td>100%</td></tr>
                  </tbody>
                </table>
                <p class="muted small">*Portfolio data through {{ view.schedule.as_of }}. **Current AUM is third
                  party net preferred equity; excludes the {{ m(view.unfunded) }}M unfunded.</p>
              </template>

              <!-- p.27 -->
              <template v-else-if="view.key === 'performance'">
                <table class="grid fig">
                  <thead><tr><th></th><th>Pref. Equity</th><th>Proj. IRR</th><th>Final Realized Gross IRR</th>
                    <th>Proceeds-to-Date</th><th>CoC Act. Rtns. Since Close</th></tr></thead>
                  <tbody>
                    <tr v-for="r in view.rows" :key="r.key">
                      <td>{{ r.label }}</td>
                      <td :title="r.pref_basis">{{ m(r.pref) }}{{ r.key === 'current' ? '*' : '' }}</td>
                      <td>{{ r.key === 'exited' ? 'N/A' : pct(r.proj_irr) }}</td>
                      <td>{{ r.key === 'current' ? 'N/A' : pct(r.realized_irr) }}</td>
                      <td>{{ m(r.proceeds) }}</td><td>{{ pct(r.coc) }}</td>
                    </tr>
                    <tr class="tot"><td>Total</td><td></td><td></td><td></td>
                      <td>{{ m(view.total.proceeds) }}</td><td>{{ pct(view.total.coc) }}</td></tr>
                  </tbody>
                </table>
                <p class="muted small">*Preferred equity balance includes unfunded commitments at
                  {{ view.schedule.as_of }}. Proceeds and CoC are through {{ view.schedule.as_of }}.</p>
              </template>

              <!-- pp.29-31: the Investment Metrics payload itself -->
              <template v-else-if="view.key === 'investment_summaries'">
                <template v-for="t in ['current', 'sold']" :key="t">
                  <h4>{{ view.investment_metrics[t].title }}
                    <span class="muted small">{{ view.investment_metrics.as_of_display }} ·
                      {{ view.investment_metrics.units_note }}</span></h4>
                  <div class="scroller">
                    <InvestmentMetricsTable :table="view.investment_metrics[t]"
                      :total-markers="t === 'sold' ? view.investment_metrics.sold.total_markers : undefined"
                      :grand-total="t === 'sold' ? view.investment_metrics.grand_total : undefined" />
                  </div>
                  <div class="muted small">
                    <div v-for="f in view.investment_metrics[t].footnotes" :key="f.n">({{ f.n }}) {{ f.text }}</div>
                  </div>
                </template>
                <p class="muted small">{{ view.investment_metrics.disclaimer }}</p>
              </template>

              <div v-if="view.notes?.length" class="notes">
                <button class="btn-secondary" @click="showNotes = !showNotes">
                  {{ showNotes ? 'Hide' : 'Show' }} notes ({{ view.notes.length }})</button>
                <ul v-if="showNotes" class="small"><li v-for="(n, i) in view.notes" :key="i">{{ n }}</li></ul>
              </div>
            </template>
          </div>

          <h3>Narrative</h3>
          <div v-for="n in meeting.narratives" :key="n.key" class="narr">
            <label><strong>{{ n.title }}</strong> <span class="muted small">p. {{ n.pages }}
              <template v-if="n.updated_by"> · {{ n.updated_by }}, {{ n.updated_at }}</template></span></label>
            <textarea v-model="n.body" rows="4" :disabled="!editable || !can('board_edit')"
                      @input="narrativeDirty[n.key] = true" />
            <button v-if="can('board_edit') && editable" class="btn-secondary"
                    :disabled="!narrativeDirty[n.key]" @click="saveNarrative(n)">Save</button>
          </div>
        </div>
      </div>
    </section>

    <!-- ACCESS: the admin account only -->
    <section v-if="tab === 'access' && access">
      <p class="muted">Board is ticked for nobody until granted here. A permission needs the Board
        section too: removing Board, or passing its end date, removes them all. Every change is logged.</p>
      <table class="grid">
        <thead>
          <tr><th>User</th><th>Role</th><th>Board</th><th>Until</th>
            <th v-for="p in access.permissions" :key="p.key" :title="p.describe">{{ p.label }}</th></tr>
        </thead>
        <tbody>
          <tr v-for="u in access.users" :key="u.id">
            <td>{{ u.username }}<div class="muted small">{{ u.email }}</div></td>
            <td>{{ u.role }}</td>
            <td><input type="checkbox" :checked="u.board"
                       @change="putAccess(u, { board: ($event.target as HTMLInputElement).checked })" /></td>
            <td><input type="date" :value="u.board_until || ''" :disabled="!u.board"
                       @change="putAccess(u, { board: true, board_until: ($event.target as HTMLInputElement).value || null })" /></td>
            <td v-for="p in access.permissions" :key="p.key">
              <input type="checkbox" :checked="u.permissions.includes(p.key)" :disabled="!u.board"
                     @change="togglePerm(u, p.key, ($event.target as HTMLInputElement).checked)" />
            </td>
          </tr>
        </tbody>
      </table>
    </section>

    <section v-if="tab === 'log'">
      <table class="grid">
        <thead><tr><th>When</th><th>Who</th><th>Action</th><th>User</th><th>Detail</th></tr></thead>
        <tbody>
          <tr v-for="e in log" :key="e.id">
            <td>{{ e.at }}</td><td>{{ e.actor }}</td><td>{{ e.action }}</td>
            <td>{{ e.target || '' }}</td><td class="small">{{ e.detail ? JSON.stringify(e.detail) : '' }}</td>
          </tr>
          <tr v-if="!log.length"><td colspan="5" class="muted">Nothing logged yet.</td></tr>
        </tbody>
      </table>
    </section>
  </div>
</template>

<style scoped>
.board { padding: 16px 24px; }
h1 { margin: 0 0 4px; }
.muted { color: var(--color-text-muted, #6b7280); }
.small { font-size: 12px; }
.tabs { display: flex; gap: 4px; margin: 12px 0; border-bottom: 1px solid var(--color-border, #ddd); }
.tabs button { background: none; border: none; padding: 6px 12px; cursor: pointer; color: inherit;
  border-bottom: 2px solid transparent; }
.tabs button.on { border-bottom-color: var(--color-primary, #1f4e79); font-weight: 600; }
.err { color: #b42318; margin: 6px 0; }
.warn { color: #b45309; }
.row { display: flex; gap: 20px; align-items: flex-start; }
.list { width: 260px; display: flex; flex-direction: column; gap: 6px; }
.item { padding: 8px; border: 1px solid var(--color-border, #ddd); border-radius: 6px; cursor: pointer;
  display: flex; flex-direction: column; }
.item.on { border-color: var(--color-primary, #1f4e79); }
.new { display: flex; flex-direction: column; gap: 6px; margin-top: 12px; padding: 8px;
  border: 1px dashed var(--color-border, #ccc); border-radius: 6px; }
.new label { display: flex; flex-direction: column; font-size: 12.5px; }
.detail { flex: 1; min-width: 0; }
.grid { border-collapse: collapse; width: 100%; font-size: 13px; }
.grid th, .grid td { border-bottom: 1px solid var(--color-border, #e5e7eb); padding: 5px 8px;
  text-align: left; vertical-align: top; }
.grid tr.off td { opacity: .55; }
.view { margin: 14px 0 20px; padding: 12px; border: 1px solid var(--color-border, #e5e7eb); border-radius: 6px; }
.view h3 { margin: 0 0 6px; }
.view h4 { margin: 14px 0 6px; }
.grid.fig td:not(:first-child), .grid.fig th:not(:first-child) { text-align: right; }
.grid.fig tr.tot td { font-weight: 600; border-top: 2px solid var(--color-border, #d1d5db); }
.grid.narrow { width: auto; min-width: 360px; }
.scroller { overflow-x: auto; }
.notes { margin-top: 10px; }
.narr { display: flex; flex-direction: column; gap: 4px; margin-bottom: 12px; }
.narr textarea { width: 100%; font: inherit; padding: 6px; }
.narr button { align-self: flex-start; }
.btn-primary, .btn-secondary { padding: 5px 12px; border-radius: 6px; font-size: 13px; cursor: pointer;
  border: 1px solid var(--color-primary, #1f4e79); }
.btn-primary { background: var(--color-primary, #1f4e79); color: #fff; font-weight: 600; }
.btn-secondary { background: var(--color-surface, #fff); color: var(--color-primary, #1f4e79); }
.btn-primary:disabled, .btn-secondary:disabled { opacity: .55; cursor: default; }
</style>
