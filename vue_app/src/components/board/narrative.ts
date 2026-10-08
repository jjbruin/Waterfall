/**
 * Narrative sections: from the text the editors wrote (and the attachments they
 * placed) to deck pages.
 *
 * THE TEXT. A blank line starts a new paragraph; lines that all begin "-" or "•"
 * are a bullet list; a line beginning "## " is a sub-heading. Attachments follow
 * the text, in their order, one block per attachment page.
 *
 * PAGINATION IS MEASURED, NOT ESTIMATED. Each block is rendered into a hidden
 * copy of the page body -- same width, same type (the ``bd-narr`` styles in
 * NarrativePageSlide) -- and packed onto pages. A page breaks BETWEEN blocks;
 * a paragraph too long to finish on the page breaks at a sentence end, a list
 * between items, so a section continues onto further pages where the text
 * naturally pauses. An attachment page taller than a whole page is scaled down
 * to fit one. The section's last page keeps room for its footnotes.
 */

export interface Block {
  type: 'h' | 'p' | 'ul' | 'img'
  text?: string
  items?: string[]
  aid?: number
  n?: number
  width?: number
  height?: number
  caption?: string
  /** Display size of an attachment page, set by layout. */
  dw?: number
  dh?: number
}

/** Width of the text column: the page body (990) less the narrative's side padding. */
export const COLUMN = 950
/** Height of a page body under the title: 825 - 112 (title) - 56 (foot). */
export const BODY_HEIGHT = 657
const CAPTION = 26

export function parseBody(body: string): Block[] {
  const out: Block[] = []
  for (const para of (body || '').split(/\n\s*\n/)) {
    const lines = para.split('\n').map((l) => l.trim()).filter(Boolean)
    if (!lines.length) continue
    if (lines.length === 1 && lines[0].startsWith('## ')) { out.push({ type: 'h', text: lines[0].slice(3).trim() }); continue }
    if (lines.every((l) => /^[-•]\s*/.test(l))) {
      out.push({ type: 'ul', items: lines.map((l) => l.replace(/^[-•]\s*/, '')) })
      continue
    }
    out.push({ type: 'p', text: lines.join(' ') })
  }
  return out
}

export function attachmentBlocks(attachments: any[]): Block[] {
  const out: Block[] = []
  for (const a of attachments || []) {
    for (const pg of a.pages || []) {
      out.push({ type: 'img', aid: a.id, n: pg.n, width: pg.width, height: pg.height,
        caption: pg.n === 1 ? (a.caption || '') : '' })
    }
  }
  return out
}

// ---------------------------------------------------------------- measuring

let host: HTMLElement | null = null
function measurer(): HTMLElement {
  if (host && document.body.contains(host)) return host
  host = document.createElement('div')
  host.className = 'bd-narr'
  host.setAttribute('aria-hidden', 'true')
  Object.assign(host.style, { position: 'absolute', left: '-40000px', top: '0', width: COLUMN + 40 + 'px',
    visibility: 'hidden', pointerEvents: 'none' })
  document.body.appendChild(host)
  return host
}

function element(b: Block): HTMLElement {
  if (b.type === 'ul') {
    const ul = document.createElement('ul')
    ul.className = 'bd-ul'
    for (const it of b.items || []) { const li = document.createElement('li'); li.textContent = it; ul.appendChild(li) }
    return ul
  }
  const el = document.createElement(b.type === 'h' ? 'h3' : 'p')
  el.className = b.type === 'h' ? 'bd-h' : 'bd-p'
  el.textContent = b.text || ''
  return el
}

/** Height a block takes on the page, margins included. */
export function measure(b: Block): number {
  if (b.type === 'img') return (b.dh || 0) + (b.caption ? CAPTION : 0) + 14
  const h = measurer()
  const box = document.createElement('div')
  box.style.display = 'flow-root'
  box.appendChild(element(b))
  h.appendChild(box)
  const px = box.getBoundingClientRect().height
  h.removeChild(box)
  return px
}

/** Height of a footnote block in the frame's style (BoardSlide .notes). */
export function measureNotes(footnotes: string[], disclosure?: string | null): number {
  if (!footnotes.length && !disclosure) return 0
  const h = measurer()
  const box = document.createElement('div')
  Object.assign(box.style, { display: 'flow-root', width: '990px', paddingTop: '8px', fontSize: '11.5px', lineHeight: '1.35',
    fontFamily: "Calibri, Carlito, 'Segoe UI', Arial, sans-serif" })
  for (const f of footnotes) { const d = document.createElement('div'); d.textContent = f; box.appendChild(d) }
  if (disclosure) {
    const d = document.createElement('div')
    Object.assign(d.style, { marginTop: '4px', fontSize: '10.5px' })
    d.textContent = disclosure
    box.appendChild(d)
  }
  h.appendChild(box)
  const px = box.getBoundingClientRect().height
  h.removeChild(box)
  return px
}

function sizeImage(b: Block, avail: number): Block {
  const w = b.width || 1, ht = b.height || 1
  const maxH = avail - (b.caption ? CAPTION : 0) - 14
  let dw = Math.min(COLUMN, w), dh = dw * ht / w
  if (dh > maxH) { dh = maxH; dw = dh * w / ht }
  return { ...b, dw: Math.round(dw), dh: Math.round(dh) }
}

function sentences(t: string): string[] {
  return t.split(/(?<=[.!?;:])\s+(?=[A-Z0-9"'(“])/)
}

/** Pages of blocks for one section. ``reserve`` is the footnote height kept on its last page. */
export function paginate(blocks: Block[], reserve = 0, avail = BODY_HEIGHT): Block[][] {
  const pages: Block[][] = [[]]
  let used = 0
  const room = () => avail - used
  const newPage = () => { pages.push([]); used = 0 }
  const place = (b: Block, h: number) => { pages[pages.length - 1].push(b); used += h }

  const queue = blocks.map((b) => (b.type === 'img' ? sizeImage(b, avail) : b))
  while (queue.length) {
    const b = queue.shift() as Block
    const h = measure(b)
    if (h <= room()) { place(b, h); continue }
    // Does not fit what is left of this page: break inside it at a natural point.
    if (b.type === 'p' || b.type === 'ul') {
      const parts = b.type === 'p' ? sentences(b.text || '') : (b.items || [])
      let k = 0
      for (let i = 1; i <= parts.length - 1; i++) {
        const head: Block = b.type === 'p' ? { type: 'p', text: parts.slice(0, i).join(' ') } : { type: 'ul', items: parts.slice(0, i) }
        if (measure(head) <= room()) k = i
        else break
      }
      if (k > 0) {
        const head: Block = b.type === 'p' ? { type: 'p', text: parts.slice(0, k).join(' ') } : { type: 'ul', items: parts.slice(0, k) }
        const rest: Block = b.type === 'p' ? { type: 'p', text: parts.slice(k).join(' ') } : { type: 'ul', items: parts.slice(k) }
        place(head, measure(head))
        newPage()
        queue.unshift(rest)
        continue
      }
    }
    if (pages[pages.length - 1].length) { newPage(); queue.unshift(b); continue }
    // Alone on a fresh page and still too tall (a sentence longer than a page): place it.
    place(b, h)
  }
  // The last page keeps room for the section's footnotes: if they do not fit, the
  // last block moves to a page of its own.
  if (reserve && used + reserve > avail && pages[pages.length - 1].length > 1) {
    const last = pages[pages.length - 1].pop() as Block
    pages.push([last])
  }
  return pages.filter((p, i) => p.length || i === 0)
}
