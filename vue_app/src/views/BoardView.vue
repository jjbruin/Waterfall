<script setup lang="ts">
// Board section, Phase 0 (Oct 5 2026): meetings, the as-of date each schedule is
// drawn at, the narrative blocks, and -- for the admin account only -- who holds
// what. Phase 1 (Oct 7 2026) adds the first schedule views -- pages 26, 27 and
// 29-31 -- each read from the server at the as-of THIS MEETING carries for it;
// the screen computes nothing. An open meeting is shown as the deck itself
// (BoardDeck), with the schedule set-up and the narrative on their own tabs.
// Every gate here is ALSO enforced by the server; the screen only hides what
// would be refused.
import { ref, computed, onMounted } from 'vue'
import api from '../api/client'
import BoardDeck from '@/components/board/BoardDeck.vue'

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
// An open meeting fills the page: the deck first, set-up and narrative on their own tabs.
const mtab = ref<'deck' | 'schedules' | 'narrative'>('deck')
async function openMeeting(id: number) {
  error.value = ''; notice.value = ''
  meeting.value = (await api.get(`/api/board/meetings/${id}`)).data
  mtab.value = 'deck'
}
function closeMeeting() { meeting.value = null; error.value = ''; notice.value = '' }
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
const narrativeDirty = ref<Record<string, boolean>>({})
async function saveNarrative(n: any) {
  error.value = ''
  try {
    meeting.value = (await api.put(`/api/board/meetings/${meeting.value.id}/narratives/${n.key}`,
      { body: n.body })).data
    narrativeDirty.value[n.key] = false
  } catch (e) { fail(e) }
}

// ---- the package: footnotes saved in the deck, attachments in a narrative section ----
function onNotesSaved(key: string, value: any) {
  const pn = { ...(meeting.value.page_notes || {}) }
  if (value) pn[key] = value
  else delete pn[key]
  meeting.value = { ...meeting.value, page_notes: pn }
}
const uploading = ref<Record<string, boolean>>({})
async function attach(n: any, ev: Event) {
  const input = ev.target as HTMLInputElement
  const files = Array.from(input.files || [])
  input.value = ''
  if (!files.length) return
  error.value = ''; uploading.value[n.key] = true
  try {
    for (const f of files) {
      const fd = new FormData()
      fd.append('file', f)
      await api.post(`/api/board/meetings/${meeting.value.id}/narratives/${n.key}/attachments`, fd)
    }
  } catch (e) { fail(e) } finally {
    uploading.value[n.key] = false
    await refreshAttachments()
  }
}
async function refreshAttachments() {
  const m = (await api.get(`/api/board/meetings/${meeting.value.id}`)).data
  // Keep unsaved narrative text: only the attachments come from the server.
  const byKey: Record<string, any> = Object.fromEntries(m.narratives.map((x: any) => [x.key, x.attachments]))
  meeting.value = { ...meeting.value, narratives: meeting.value.narratives.map((x: any) => ({ ...x, attachments: byKey[x.key] || [] })) }
}
async function updateAttachment(a: any, body: Record<string, any>) {
  error.value = ''
  try { await api.put(`/api/board/meetings/${meeting.value.id}/attachments/${a.id}`, body) } catch (e) { fail(e) }
  await refreshAttachments()
}
async function removeAttachment(a: any) {
  if (!confirm(`Remove "${a.filename}" from this section?`)) return
  error.value = ''
  try { await api.delete(`/api/board/meetings/${meeting.value.id}/attachments/${a.id}`) } catch (e) { fail(e) }
  await refreshAttachments()
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
    <p v-if="!meeting" class="muted">Meetings and the schedules each will carry. Every schedule is drawn from an
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

    <!-- MEETINGS: the list, until one is opened -->
    <section v-if="tab === 'meetings' && !meeting">
      <div class="list">
        <div v-for="m in meetings" :key="m.id" class="item" @click="openMeeting(m.id)">
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
    </section>

    <!-- ONE MEETING: the deck, its schedules, its narrative -->
    <section v-if="tab === 'meetings' && meeting" class="detail">
      <div class="mhead">
        <button class="btn-secondary" @click="closeMeeting">‹ All meetings</button>
        <h2>{{ meeting.title }}</h2>
        <span class="muted">Meeting {{ meeting.meeting_date }} · default as-of {{ meeting.default_as_of }}
          · {{ meeting.status }}<span v-if="!editable"> — no longer editable</span></span>
      </div>
      <div class="tabs sub">
        <button :class="{ on: mtab === 'deck' }" @click="mtab = 'deck'">Deck</button>
        <button :class="{ on: mtab === 'schedules' }" @click="mtab = 'schedules'">Schedules &amp; dates</button>
        <button :class="{ on: mtab === 'narrative' }" @click="mtab = 'narrative'">Narrative</button>
      </div>

      <BoardDeck v-if="mtab === 'deck'" :meeting="meeting" :can-edit="editable && can('board_edit')"
                 @notes-saved="onNotesSaved" />

      <template v-if="mtab === 'schedules'">
        <p class="muted small">Which schedules the deck carries, and the as-of date each is drawn at.
          A changed date re-fetches only that schedule.</p>
        <table class="grid">
          <thead><tr><th>In</th><th>Pages</th><th>Schedule</th><th>As of</th><th>Phase</th><th>Status</th></tr></thead>
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
            </tr>
          </tbody>
        </table>
      </template>

      <template v-if="mtab === 'narrative'">
        <p class="muted small">The narrative sections appear in the <strong>Full</strong> package. A blank line starts a
          new paragraph; lines starting with "-" are bullets; a line starting "## " is a sub-heading. Attachments
          (images, or PDFs -- each PDF page becomes a page image) follow the text. A section longer than a page
          continues onto further pages, breaking between paragraphs, bullets and attachments.</p>
        <div v-for="n in meeting.narratives" :key="n.key" class="narr">
          <label><strong>{{ n.title }}</strong> <span class="muted small">p. {{ n.pages }}
            <template v-if="n.updated_by"> · {{ n.updated_by }}, {{ n.updated_at }}</template></span></label>
          <textarea v-model="n.body" rows="6" :disabled="!editable || !can('board_edit')"
                    @input="narrativeDirty[n.key] = true" />
          <button v-if="can('board_edit') && editable" class="btn-secondary"
                  :disabled="!narrativeDirty[n.key]" @click="saveNarrative(n)">Save</button>
          <div class="att">
            <div v-for="(a, i) in n.attachments || []" :key="a.id" class="att-row">
              <span class="att-name" :title="a.filename">{{ a.filename }}</span>
              <span class="muted small">{{ a.page_count }} page{{ a.page_count === 1 ? '' : 's' }}</span>
              <input class="att-cap" :value="a.caption" placeholder="Caption (optional)"
                     :disabled="!editable || !can('board_edit')"
                     @change="updateAttachment(a, { caption: ($event.target as HTMLInputElement).value })" />
              <template v-if="editable && can('board_edit')">
                <button class="ico" title="Move up" :disabled="i === 0" @click="updateAttachment(a, { move: -1 })">↑</button>
                <button class="ico" title="Move down" :disabled="i === (n.attachments || []).length - 1"
                        @click="updateAttachment(a, { move: 1 })">↓</button>
                <button class="ico" title="Remove" @click="removeAttachment(a)">✕</button>
              </template>
            </div>
            <label v-if="editable && can('board_edit')" class="att-add">
              <input type="file" accept="image/png,image/jpeg,image/gif,image/webp,application/pdf" multiple
                     @change="attach(n, $event)" />
              <span class="btn-secondary">{{ uploading[n.key] ? 'Attaching…' : '+ Attach image or PDF' }}</span>
            </label>
          </div>
        </div>
      </template>
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
.list { max-width: 420px; display: flex; flex-direction: column; gap: 6px; }
.item { padding: 8px; border: 1px solid var(--color-border, #ddd); border-radius: 6px; cursor: pointer;
  display: flex; flex-direction: column; }
.new { display: flex; flex-direction: column; gap: 6px; margin-top: 12px; padding: 8px;
  border: 1px dashed var(--color-border, #ccc); border-radius: 6px; }
.new label { display: flex; flex-direction: column; font-size: 12.5px; }
.detail { min-width: 0; }
.mhead { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }
.mhead h2 { margin: 0; }
.tabs.sub { margin: 10px 0 12px; }
.grid { border-collapse: collapse; width: 100%; font-size: 13px; }
.grid th, .grid td { border-bottom: 1px solid var(--color-border, #e5e7eb); padding: 5px 8px;
  text-align: left; vertical-align: top; }
.grid tr.off td { opacity: .55; }
.narr { display: flex; flex-direction: column; gap: 4px; margin-bottom: 12px; }
.narr textarea { width: 100%; font: inherit; padding: 6px; }
.narr button { align-self: flex-start; }
.att { display: flex; flex-direction: column; gap: 4px; margin-top: 2px; }
.att-row { display: flex; align-items: center; gap: 8px; font-size: 13px; }
.att-name { max-width: 260px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.att-cap { flex: 1; max-width: 360px; font: inherit; font-size: 12.5px; padding: 3px 6px; }
.att-add input { display: none; }
.att-add { align-self: flex-start; cursor: pointer; }
.ico { width: 26px; height: 26px; border: 1px solid var(--color-border, #d1d5db); border-radius: 4px;
  background: var(--color-surface, #fff); color: inherit; cursor: pointer; }
.ico:disabled { opacity: .35; }
.btn-primary, .btn-secondary { padding: 5px 12px; border-radius: 6px; font-size: 13px; cursor: pointer;
  border: 1px solid var(--color-primary, #1f4e79); }
.btn-primary { background: var(--color-primary, #1f4e79); color: #fff; font-weight: 600; }
.btn-secondary { background: var(--color-surface, #fff); color: var(--color-primary, #1f4e79); }
.btn-primary:disabled, .btn-secondary:disabled { opacity: .55; cursor: default; }
</style>
