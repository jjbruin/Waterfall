<script setup lang="ts">
// "From SharePoint" -- a button beside an importer's own upload control. It
// opens a browser over the user's OneDrive and the SharePoint libraries they
// can read, downloads what they tick, and emits the files as browser File
// objects: the host hands them to the SAME code a local upload goes through.
//
// Hidden when the server has no Entra app configured (/auth/sso/sharepoint).
// Never truncates: a pick over the host's limit is refused, with the count.
import { computed, onMounted, ref } from 'vue'
import * as sp from '../../services/sharepoint'
import type { SpItem, SpSite, SpDrive } from '../../services/sharepoint'

const props = withDefaults(defineProps<{
  accept?: string          // same syntax as <input accept>, e.g. ".pdf,.csv"
  multiple?: boolean
  folders?: boolean        // a whole folder may be picked (its files, recursively)
  maxFiles?: number        // the importer's own limit
  label?: string
  disabled?: boolean
  rememberAs?: string      // remembers the last folder per importer
  buttonClass?: string
}>(), {
  accept: '', multiple: false, folders: false, maxFiles: 200,
  label: 'From SharePoint', disabled: false, rememberAs: '', buttonClass: 'btn-secondary',
})
const emit = defineEmits<{ (e: 'picked', files: File[]): void }>()

const enabled = ref(false)
const open = ref(false)
const busy = ref('')
const error = ref('')

// where we are
type View = 'home' | 'site' | 'folder'
const view = ref<View>('home')
const sites = ref<SpSite[]>([])
const siteQuery = ref('')
const site = ref<SpSite | null>(null)
const libraries = ref<SpDrive[]>([])
const drive = ref<SpDrive | null>(null)
const crumbs = ref<SpItem[]>([])           // folders below the drive root
const items = ref<SpItem[]>([])
const link = ref('')

// what is ticked, by item id
const picked = ref<Map<string, SpItem>>(new Map())

const pickedList = computed(() => [...picked.value.values()])
const acceptLabel = computed(() => props.accept ? props.accept.split(',').join(', ') : 'any file')

onMounted(async () => {
  enabled.value = (await sp.sharepointConfig()).enabled
})

// Load MSAL as the pointer nears the button, so the click that opens the
// sign-in popup is not spent waiting for a download (browsers block a popup
// opened too long after the click).
function warm() { sp.warmUp() }

const STORE_KEY = computed(() => props.rememberAs ? `sp-picker:${props.rememberAs}` : '')

function remember() {
  if (!STORE_KEY.value || !drive.value) return
  try {
    localStorage.setItem(STORE_KEY.value, JSON.stringify({
      drive: drive.value, site: site.value, crumbs: crumbs.value,
    }))
  } catch { /* storage unavailable: just not remembered */ }
}

// MSAL's codes, said in words. The rest carry a readable message already.
const MSAL_ERRORS: Record<string, string> = {
  popup_window_error: 'The Microsoft sign-in window was blocked. Allow pop-ups for this site, then try again.',
  empty_window_error: 'The Microsoft sign-in window was blocked. Allow pop-ups for this site, then try again.',
  user_cancelled: 'Microsoft sign-in was closed before it finished.',
  interaction_in_progress: 'A Microsoft sign-in is already open in another window. Finish or close it, then try again.',
  consent_required: 'Your Microsoft account has not been allowed to use Waterfall XIRR. Ask IT to assign you to the app.',
  access_denied: 'Your Microsoft account has not been allowed to use Waterfall XIRR. Ask IT to assign you to the app.',
}

async function run<T>(what: string, fn: () => Promise<T>): Promise<T | undefined> {
  busy.value = what
  error.value = ''
  try { return await fn() } catch (e: any) {
    error.value = MSAL_ERRORS[e?.errorCode] || e?.message || String(e)
  } finally { busy.value = '' }
}

async function start() {
  open.value = true
  picked.value = new Map()
  error.value = ''
  let saved: any = null
  try { saved = STORE_KEY.value ? JSON.parse(localStorage.getItem(STORE_KEY.value) || 'null') : null } catch { saved = null }
  if (saved?.drive) {
    site.value = saved.site || null
    drive.value = saved.drive
    crumbs.value = saved.crumbs || []
    view.value = 'folder'
    const ok = await run('Opening the last folder…', loadFolder)
    if (ok !== undefined) return
    // the saved folder is gone or no longer readable: start from home
  }
  await goHome()
}

async function goHome() {
  view.value = 'home'
  site.value = null
  drive.value = null
  crumbs.value = []
  items.value = []
  await run('Signing in to Microsoft…', async () => {
    sites.value = await sp.followedSites()
  })
}

async function search() {
  await run('Searching…', async () => { sites.value = await sp.searchSites(siteQuery.value) })
}

async function openSite(s: SpSite) {
  site.value = s
  view.value = 'site'
  await run('Loading libraries…', async () => { libraries.value = await sp.siteLibraries(s.id) })
}

async function openMyDrive() {
  site.value = null
  await run('Opening your OneDrive…', async () => {
    drive.value = await sp.myDrive()
    crumbs.value = []
    view.value = 'folder'
    await loadFolder()
  })
}

async function openLibrary(d: SpDrive) {
  drive.value = d
  crumbs.value = []
  view.value = 'folder'
  await run('Loading…', loadFolder)
}

async function loadFolder() {
  if (!drive.value) return
  const here = crumbs.value[crumbs.value.length - 1]
  items.value = await sp.children(drive.value.id, here ? here.id : 'root')
  remember()
  return true
}

async function enter(f: SpItem) {
  crumbs.value = [...crumbs.value, f]
  await run('Loading…', loadFolder)
}

async function toCrumb(i: number) {
  crumbs.value = crumbs.value.slice(0, i)
  await run('Loading…', loadFolder)
}

async function openLink() {
  if (!link.value.trim()) return
  await run('Opening link…', async () => {
    const it = await sp.resolveLink(link.value)
    const path = await sp.pathTo(it)
    drive.value = { id: it.driveId, name: 'Linked location', webUrl: '' }
    site.value = null
    view.value = 'folder'
    if (it.isFolder) {
      crumbs.value = path
    } else {
      crumbs.value = path.slice(0, -1)
      picked.value = new Map([[it.id, it]])
    }
    await loadFolder()
    link.value = ''
  })
}

function selectable(it: SpItem) {
  return it.isFolder ? props.folders : sp.accepts(it.name, props.accept)
}

function toggle(it: SpItem) {
  if (!selectable(it)) return
  const m = new Map(props.multiple ? picked.value : [])
  if (picked.value.has(it.id)) m.delete(it.id); else m.set(it.id, it)
  picked.value = m
}

function size(n: number) {
  if (n >= 1024 * 1024) return (n / 1024 / 1024).toFixed(1) + ' MB'
  if (n >= 1024) return Math.round(n / 1024) + ' KB'
  return n + ' B'
}

async function importPicked() {
  const sel = pickedList.value
  if (!sel.length) return
  const files: File[] = []
  await run('Preparing…', async () => {
    // expand folders into their acceptable files, keeping the folder path
    const todo: { item: SpItem; rel?: string }[] = []
    for (const it of sel) {
      if (it.isFolder) {
        busy.value = `Listing ${it.name}…`
        const under = await sp.filesUnder(it, props.accept, props.maxFiles)
        if (!under.length) throw new Error(`${it.name} holds no ${acceptLabel.value} files.`)
        todo.push(...under)
      } else {
        todo.push({ item: it })
      }
    }
    if (todo.length > props.maxFiles) {
      throw new Error(`That is more than ${props.maxFiles} files, the most this import takes at once. `
        + 'Pick a smaller folder or fewer files.')
    }
    for (let i = 0; i < todo.length; i++) {
      busy.value = `Downloading ${i + 1} of ${todo.length}: ${todo[i].item.name}`
      files.push(await sp.download(todo[i].item, todo[i].rel))
    }
    open.value = false
    emit('picked', files)
  })
}
</script>

<template>
  <button v-if="enabled" type="button" :class="buttonClass" :disabled="disabled"
          :title="`Choose ${acceptLabel} from SharePoint or OneDrive`"
          @mouseenter="warm" @focus="warm" @click="start">
    {{ label }}
  </button>

  <Teleport to="body">
    <div v-if="open" class="sp-backdrop" @click.self="!busy && (open = false)">
      <div class="sp-modal" role="dialog" aria-label="Choose from SharePoint">
        <div class="sp-head">
          <strong>Choose from SharePoint</strong>
          <span class="sp-muted">{{ acceptLabel }}<template v-if="multiple"> · up to {{ maxFiles }}</template></span>
          <button type="button" class="sp-x" :disabled="!!busy" @click="open = false" aria-label="Close">×</button>
        </div>

        <div class="sp-nav">
          <button type="button" class="sp-link" :disabled="!!busy" @click="goHome">Sites</button>
          <button type="button" class="sp-link" :disabled="!!busy" @click="openMyDrive">My OneDrive</button>
          <form class="sp-paste" @submit.prevent="openLink">
            <input v-model="link" placeholder="…or paste a SharePoint / OneDrive link" :disabled="!!busy" />
            <button type="submit" class="btn-secondary" :disabled="!!busy || !link.trim()">Open</button>
          </form>
        </div>

        <div v-if="view === 'folder' && drive" class="sp-crumbs">
          <template v-if="site"><button type="button" class="sp-link" @click="openSite(site!)">{{ site.name }}</button> ›</template>
          <button type="button" class="sp-link" @click="toCrumb(0)">{{ drive.name }}</button>
          <template v-for="(c, i) in crumbs" :key="c.id">
            › <button type="button" class="sp-link" @click="toCrumb(i + 1)">{{ c.name }}</button>
          </template>
        </div>
        <div v-else-if="view === 'site' && site" class="sp-crumbs">
          <button type="button" class="sp-link" @click="goHome">Sites</button> › {{ site.name }}
        </div>

        <div v-if="error" class="sp-error">{{ error }}</div>

        <div class="sp-body">
          <div v-if="busy" class="sp-muted sp-busy">{{ busy }}</div>

          <!-- sites -->
          <template v-else-if="view === 'home'">
            <form class="sp-search" @submit.prevent="search">
              <input v-model="siteQuery" placeholder="Search SharePoint sites" />
              <button type="submit" class="btn-secondary">Search</button>
            </form>
            <p v-if="!sites.length" class="sp-muted">
              No sites to show. Search for one, open your OneDrive, or paste a link.
            </p>
            <ul class="sp-list">
              <li v-for="s in sites" :key="s.id">
                <button type="button" class="sp-row" @click="openSite(s)">
                  <span class="sp-ico">🏢</span><span class="sp-name">{{ s.name }}</span>
                  <span class="sp-muted sp-url">{{ s.webUrl }}</span>
                </button>
              </li>
            </ul>
          </template>

          <!-- a site's libraries -->
          <ul v-else-if="view === 'site'" class="sp-list">
            <li v-for="d in libraries" :key="d.id">
              <button type="button" class="sp-row" @click="openLibrary(d)">
                <span class="sp-ico">📚</span><span class="sp-name">{{ d.name }}</span>
              </button>
            </li>
            <li v-if="!libraries.length" class="sp-muted">This site has no document libraries you can read.</li>
          </ul>

          <!-- a folder -->
          <table v-else class="sp-table">
            <tbody>
              <tr v-for="it in items" :key="it.id"
                  :class="{ dim: !it.isFolder && !selectable(it), on: picked.has(it.id) }">
                <td class="sp-chk">
                  <input v-if="selectable(it)" :type="multiple ? 'checkbox' : 'radio'"
                         :checked="picked.has(it.id)" @change="toggle(it)"
                         :aria-label="`Select ${it.name}`" />
                </td>
                <td>
                  <button v-if="it.isFolder" type="button" class="sp-row" @click="enter(it)">
                    <span class="sp-ico">📁</span><span class="sp-name">{{ it.name }}</span>
                    <span class="sp-muted" v-if="it.childCount !== undefined">{{ it.childCount }} items</span>
                  </button>
                  <span v-else class="sp-file" @click="toggle(it)">
                    <span class="sp-ico">📄</span><span class="sp-name">{{ it.name }}</span>
                  </span>
                </td>
                <td class="sp-muted sp-r">{{ it.isFolder ? '' : size(it.size) }}</td>
                <td class="sp-muted sp-r">{{ it.modified ? it.modified.slice(0, 10) : '' }}</td>
              </tr>
              <tr v-if="!items.length"><td colspan="4" class="sp-muted">This folder is empty.</td></tr>
            </tbody>
          </table>
        </div>

        <div class="sp-foot">
          <span class="sp-muted">
            <template v-if="pickedList.length">
              {{ pickedList.length }} selected<template v-if="pickedList.some(p => p.isFolder)"> (folders include every {{ acceptLabel }} file inside)</template>
            </template>
            <template v-else-if="sp.signedInAs()">Signed in to Microsoft as {{ sp.signedInAs() }}</template>
          </span>
          <button type="button" class="btn-secondary" :disabled="!!busy" @click="open = false">Cancel</button>
          <button type="button" class="btn-primary" :disabled="!!busy || !pickedList.length" @click="importPicked">
            Use {{ pickedList.length > 1 ? pickedList.length + ' items' : 'selection' }}
          </button>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<style scoped>
.sp-backdrop { position: fixed; inset: 0; z-index: 1100; background: rgba(15, 20, 30, .45);
  display: flex; align-items: flex-start; justify-content: center; padding: 6vh 16px; }
.sp-modal { background: var(--color-bg, #fff); color: var(--color-text); border-radius: 8px;
  width: min(820px, 100%); max-height: 86vh; display: flex; flex-direction: column;
  box-shadow: 0 12px 40px rgba(0, 0, 0, .25); }
.sp-head, .sp-nav, .sp-crumbs, .sp-foot { padding: 8px 14px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.sp-head { border-bottom: 1px solid var(--color-border); }
.sp-head strong { font-size: 15px; }
.sp-x { margin-left: auto; background: none; border: none; font-size: 22px; line-height: 1; cursor: pointer; color: var(--color-text-secondary); }
.sp-nav { border-bottom: 1px solid var(--color-border); }
.sp-paste, .sp-search { display: flex; gap: 6px; flex: 1; min-width: 240px; }
.sp-paste input, .sp-search input { flex: 1; padding: 5px 8px; border: 1px solid var(--color-border);
  border-radius: 6px; background: var(--color-surface); color: var(--color-text); font-size: 13px; min-width: 0; }
.sp-search { margin-bottom: 8px; }
.sp-crumbs { font-size: 13px; gap: 4px; }
.sp-body { overflow: auto; padding: 4px 14px 8px; min-height: 240px; flex: 1; }
.sp-busy { padding: 24px 0; text-align: center; }
.sp-error { margin: 6px 14px 0; padding: 6px 10px; border-radius: 6px; background: rgba(170, 40, 40, .1); color: #a33; font-size: 13px; }
.sp-foot { border-top: 1px solid var(--color-border); justify-content: flex-end; }
.sp-foot > .sp-muted { margin-right: auto; }
.sp-muted { color: var(--color-text-secondary); font-size: 12.5px; }
.sp-link { background: none; border: none; padding: 0; color: var(--color-primary, #1f4e79); cursor: pointer; font-size: 13px; }
.sp-link:disabled { opacity: .5; cursor: default; }
.sp-list { list-style: none; margin: 0; padding: 0; }
.sp-row { display: flex; align-items: center; gap: 8px; width: 100%; padding: 6px 4px; background: none;
  border: none; border-bottom: 1px solid var(--color-border); cursor: pointer; text-align: left; color: var(--color-text); font-size: 13px; }
.sp-row:hover, .sp-table tr:hover { background: rgba(31, 78, 121, .05); }
.sp-url { margin-left: auto; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 45%; }
.sp-name { overflow-wrap: anywhere; }
.sp-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.sp-table td { padding: 2px 4px; border-bottom: 1px solid var(--color-border); }
.sp-table .sp-row { border-bottom: none; padding: 4px 0; }
.sp-file { display: flex; align-items: center; gap: 8px; padding: 4px 0; cursor: pointer; }
.sp-chk { width: 24px; }
.sp-r { text-align: right; white-space: nowrap; }
tr.dim { opacity: .45; }
tr.dim .sp-file { cursor: default; }
tr.on { background: rgba(31, 78, 121, .08); }
.btn-primary, .btn-secondary { padding: 5px 12px; border-radius: 6px; font-size: 13px; cursor: pointer;
  border: 1px solid var(--color-primary, #1f4e79); line-height: 1.3; }
.btn-primary { background: var(--color-primary, #1f4e79); color: #fff; font-weight: 600; }
.btn-secondary { background: var(--color-surface, #fff); color: var(--color-primary, #1f4e79); }
.btn-primary:disabled, .btn-secondary:disabled { opacity: .55; cursor: default; }
@media (max-width: 600px) {
  .sp-table td:nth-child(4) { display: none; }
  .sp-url { display: none; }
}
</style>
