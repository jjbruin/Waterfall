// SharePoint / OneDrive file access for the "From SharePoint" picker.
//
// Runs entirely in the browser: MSAL signs the user in to the Waterfall XIRR
// Entra app with DELEGATED, READ-ONLY Graph scopes, so a user only ever sees
// files they can already open in SharePoint, and no Microsoft token reaches
// our server. The picked files are handed to the SAME importer a local upload
// uses -- nothing here parses or interprets a file.
// MSAL is imported on first use, not here: the picker sits in the sidebar, and
// every page would otherwise download the sign-in library to show a button.
import type { AccountInfo, PublicClientApplication } from '@azure/msal-browser'
import api from '../api/client'

const GRAPH = 'https://graph.microsoft.com/v1.0'
// Read-only. Sites.Read.All lists the SharePoint sites and libraries;
// Files.Read.All opens files in them and in the user's OneDrive.
const SCOPES = ['Files.Read.All', 'Sites.Read.All']

export interface SpConfig { enabled: boolean; client_id?: string; tenant_id?: string }

export interface SpItem {
  id: string
  name: string
  driveId: string
  isFolder: boolean
  size: number
  childCount?: number
  modified?: string
  webUrl?: string
}

export interface SpSite { id: string; name: string; webUrl: string }
export interface SpDrive { id: string; name: string; webUrl: string }

let configPromise: Promise<SpConfig> | null = null
let pca: PublicClientApplication | null = null
let msal: typeof import('@azure/msal-browser') | null = null

/** The server's answer to "is the picker switched on?" -- asked once per page load. */
export function sharepointConfig(): Promise<SpConfig> {
  if (!configPromise) {
    configPromise = api.get('/auth/sso/sharepoint')
      .then(r => r.data as SpConfig)
      .catch(() => {
        // not signed in yet, or the server unreachable: ask again next time
        // rather than hiding the button for the rest of the page's life
        configPromise = null
        return { enabled: false }
      })
  }
  return configPromise
}

async function client(): Promise<PublicClientApplication> {
  if (pca) return pca
  const cfg = await sharepointConfig()
  if (!cfg.enabled || !cfg.client_id || !cfg.tenant_id) {
    throw new Error('SharePoint access is not configured.')
  }
  msal = await import('@azure/msal-browser')
  const app = new msal.PublicClientApplication({
    auth: {
      clientId: cfg.client_id,
      authority: `https://login.microsoftonline.com/${cfg.tenant_id}`,
      // Registered in Entra as an SPA redirect URI; see msal-redirect.html.
      redirectUri: `${window.location.origin}/msal-redirect.html`,
    },
    // Per tab, gone when the tab closes: the Microsoft token never outlives
    // the session it was granted in.
    cache: { cacheLocation: 'sessionStorage' },
  })
  await app.initialize()
  pca = app
  return app
}

/** Get the picker ready ahead of the click, so the sign-in popup opens inside
 *  the click's user gesture rather than after a slow network round trip. */
export async function warmUp(): Promise<void> {
  try { await client() } catch { /* not configured: the button stays hidden */ }
}

async function token(): Promise<string> {
  const app = await client()
  const account: AccountInfo | undefined = app.getActiveAccount() || app.getAllAccounts()[0]
  if (account) {
    try {
      return (await app.acquireTokenSilent({ scopes: SCOPES, account })).accessToken
    } catch (e) {
      if (!(msal && e instanceof msal.InteractionRequiredAuthError)) throw e
    }
  }
  const r = await app.acquireTokenPopup({ scopes: SCOPES })
  app.setActiveAccount(r.account)
  return r.accessToken
}

/** The signed-in Microsoft account, if this tab has one. */
export function signedInAs(): string | null {
  const a = pca?.getActiveAccount() || pca?.getAllAccounts()[0]
  return a?.username || null
}

async function graph<T = any>(pathOrUrl: string): Promise<T> {
  const url = pathOrUrl.startsWith('https://') ? pathOrUrl : GRAPH + pathOrUrl
  const r = await fetch(url, { headers: { Authorization: `Bearer ${await token()}` } })
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`
    try { msg = (await r.json())?.error?.message || msg } catch { /* not JSON */ }
    if (r.status === 403) msg = `You do not have access to this location. (${msg})`
    if (r.status === 404) msg = `Not found -- it may have been moved or deleted. (${msg})`
    throw new Error(msg)
  }
  return r.json()
}

/** Every page of a Graph collection. */
async function all<T = any>(path: string): Promise<T[]> {
  const out: T[] = []
  let next: string | undefined = path
  while (next) {
    const page: any = await graph(next)
    out.push(...(page.value || []))
    next = page['@odata.nextLink']
  }
  return out
}

const ITEM_FIELDS = 'id,name,size,file,folder,parentReference,lastModifiedDateTime,webUrl'

function toItem(x: any, driveId?: string): SpItem {
  return {
    id: x.id,
    name: x.name,
    driveId: driveId || x.parentReference?.driveId,
    isFolder: !!x.folder,
    size: x.size || 0,
    childCount: x.folder?.childCount,
    modified: x.lastModifiedDateTime,
    webUrl: x.webUrl,
  }
}

export async function myDrive(): Promise<SpDrive> {
  const d = await graph('/me/drive?$select=id,name,webUrl')
  return { id: d.id, name: 'My OneDrive', webUrl: d.webUrl }
}

export async function followedSites(): Promise<SpSite[]> {
  const v = await all('/me/followedSites?$select=id,displayName,webUrl')
  return v.map(s => ({ id: s.id, name: s.displayName, webUrl: s.webUrl }))
}

export async function searchSites(q: string): Promise<SpSite[]> {
  const v = await all(`/sites?search=${encodeURIComponent(q || '*')}&$select=id,displayName,webUrl`)
  return v.map(s => ({ id: s.id, name: s.displayName, webUrl: s.webUrl }))
}

export async function siteLibraries(siteId: string): Promise<SpDrive[]> {
  const v = await all(`/sites/${siteId}/drives?$select=id,name,webUrl`)
  return v.map(d => ({ id: d.id, name: d.name, webUrl: d.webUrl }))
}

export async function children(driveId: string, itemId = 'root'): Promise<SpItem[]> {
  const v = await all(`/drives/${driveId}/items/${itemId}/children?$top=500&$select=${ITEM_FIELDS}`)
  const items = v.map(x => toItem(x, driveId))
  // folders first, then by name -- the order Explorer and SharePoint show
  return items.sort((a, b) => (a.isFolder === b.isFolder ? a.name.localeCompare(b.name) : a.isFolder ? -1 : 1))
}

/** The item a pasted SharePoint / OneDrive link points at (file or folder). */
export async function resolveLink(link: string): Promise<SpItem> {
  const b64 = btoa(unescape(encodeURIComponent(link.trim())))
    .replace(/=+$/, '').replace(/\//g, '_').replace(/\+/g, '-')
  const x = await graph(`/shares/u!${b64}/driveItem?$select=${ITEM_FIELDS}`)
  return toItem(x)
}

/** The path from the drive root to an item, for the breadcrumb. */
export async function pathTo(item: SpItem): Promise<SpItem[]> {
  const x = await graph(`/drives/${item.driveId}/items/${item.id}?$select=${ITEM_FIELDS}`)
  // parentReference.path looks like "/drives/{id}/root:/A/B"
  const p: string = x.parentReference?.path || ''
  const rel = p.includes('root:') ? p.split('root:')[1] : ''
  const crumbs: SpItem[] = []
  let acc = ''
  for (const seg of rel.split('/').filter(Boolean)) {
    acc += '/' + seg
    const f = await graph(`/drives/${item.driveId}/root:${encodeURI(acc)}?$select=${ITEM_FIELDS}`)
    crumbs.push(toItem(f, item.driveId))
  }
  crumbs.push(toItem(x, item.driveId))
  return crumbs
}

/** True when ``name`` satisfies an <input accept> list like ".pdf,.csv". */
export function accepts(name: string, accept?: string): boolean {
  if (!accept) return true
  const n = name.toLowerCase()
  return accept.split(',').map(s => s.trim().toLowerCase()).filter(Boolean)
    .some(ext => ext.startsWith('.') ? n.endsWith(ext) : true)
}

/** Every acceptable file under a folder, with its path relative to the folder's
 *  parent -- the same shape a browser folder upload gives (webkitRelativePath),
 *  so importers that read the folder name keep working. */
export async function filesUnder(folder: SpItem, accept?: string, limit = 1000,
                                 prefix = folder.name): Promise<{ item: SpItem; rel: string }[]> {
  const out: { item: SpItem; rel: string }[] = []
  for (const c of await children(folder.driveId, folder.id)) {
    if (out.length > limit) break
    if (c.isFolder) out.push(...await filesUnder(c, accept, limit - out.length, `${prefix}/${c.name}`))
    else if (accepts(c.name, accept)) out.push({ item: c, rel: `${prefix}/${c.name}` })
  }
  return out
}

const MIME: Record<string, string> = {
  pdf: 'application/pdf', csv: 'text/csv',
  xlsx: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  xls: 'application/vnd.ms-excel', jpg: 'image/jpeg', jpeg: 'image/jpeg', png: 'image/png',
  gif: 'image/gif', webp: 'image/webp', heic: 'image/heic', heif: 'image/heif',
  tif: 'image/tiff', tiff: 'image/tiff', bmp: 'image/bmp',
}

/** Download one item as a browser File, exactly as if it had been chosen from disk. */
export async function download(item: SpItem, rel?: string): Promise<File> {
  // A fresh, short-lived pre-authenticated URL: fetched without our token,
  // and SharePoint allows it cross-origin.
  const meta = await graph(`/drives/${item.driveId}/items/${item.id}?$select=id,name,size,@microsoft.graph.downloadUrl`)
  const url = meta['@microsoft.graph.downloadUrl']
  if (!url) throw new Error(`${item.name}: SharePoint did not provide a download link.`)
  const r = await fetch(url)
  if (!r.ok) throw new Error(`${item.name}: download failed (${r.status})`)
  const blob = await r.blob()
  if (blob.size !== (meta.size ?? blob.size)) {
    throw new Error(`${item.name}: downloaded ${blob.size} bytes, SharePoint lists ${meta.size}`)
  }
  const ext = item.name.split('.').pop()?.toLowerCase() || ''
  const file = new File([blob], item.name, { type: blob.type || MIME[ext] || 'application/octet-stream' })
  if (rel) Object.defineProperty(file, 'webkitRelativePath', { value: rel })
  return file
}
