/**
 * Investment Metrics as an Excel workbook, styled like the reference workbook.
 *
 * No second request and no second calculation: every figure is the engine's,
 * taken from the payload already on screen. Cells hold REAL numbers (a dollar
 * amount in millions, a percentage as a fraction) with the reference's number
 * formats, so the sheet sorts, sums and filters; the text the screen shows is
 * what those formats render.
 *
 * What is copied from `Reference format.xlsx`: Garamond 11, bold centred
 * headings in three stacked rows with the grouped headings merged, the boxed
 * title block (medium rule above the as-of date and below the title), the
 * thin vertical rules (taken from the payload's `vertical_rules`, the same
 * list the print sheet uses), the grey band on every second deal row, the
 * orange banner and the column-label strip under it, the column widths, the
 * number formats, the italic 14pt footnotes, the Total / Average and Grand
 * Total rows, and the merged disclaimer. Not copied: the workbook's hidden
 * helper columns, its input cells, and its formulas -- totals are the engine's
 * values, not a second calculation.
 *
 * A figure the app does not have is an EMPTY cell. A label (Dev., Lease up,
 * N/A) is text. The Total row's footnote markers ride on the number format,
 * so the cell stays a number.
 */

const FONT = 'Garamond'
const WIDTHS = [35.8, 28.5, 22.7, 13, 24.5, 12.5, 21, 12.5, 19.7, 12.5, 21,
  12.5, 13, 13, 12.5, 16.5, 23, 16.7, 15.8, 12.5, 12.8, 13]
/** The title block and the units note span these many columns (I:AD). */
const BLOCK_COLS = 22
const UNITS_COL = 19                                   // AB
/** The reference's row band: white darkened 15% (theme 0, tint -0.15). */
const BAND = 'FFD9D9D9'
/** The reference's orange banner and the column-label strip beneath it. */
const ORANGE = 'FFFFC000'
const BLUE = 'FF104862'          // theme accent 1 (156082) darkened 25%
const BANNER = 'Orangewood Portfolio One-Pager'
/** The strip's labels by column, as the reference has them; blue where the
 *  reference's font is blue (the capitalization and UW IRR columns). */
const STRIP: Array<[string, boolean]> = [
  ['Property_Name', false], ['Asset_Type', false], ['MSA', false],
  ['Purchase_Date', false], ['Partner', false], ['Formula', true],
  ['First Lien Mortgage', true], ['Formula', true], ['PSC Pref Equity', true],
  ['Formula', true], ['First-Loss Equity', true], ['Formula', true],
  ['UW IRR', true], ['', true], ['', true], ['Proj_YR_1_CoC', false],
  ['', false], ['', false], ['', false], ['Pref Coupon', false],
  ['CF Split', false], ['IRR LB', false],
]

const MONEY_KEYS = new Set(['total_size', 'first_lien', 'first_loss'])
const MONEY_FMT = '"$"#,##0.0_);[Red]\\("$"#,##0.0\\)'
// The reference formats these two plainly on a deal row and with the
// red-parenthesis / thousands format only on the Total row.
const PREF_FMT = '"$"#,##0.0'
const PROCEEDS_FMT = '"$"0.0'
const PCT_KEYS = new Set([
  'first_lien_pct', 'pref_pct', 'first_loss_pct', 'uw_irr', 'realized_irr',
  'proj_yr1_coc', 'act_yr1_coc', 'proj_coc_since_close',
  'act_coc_since_close', 'pref_coupon', 'residual_cf_split', 'irr_lookback',
])
const TEXT_KEYS = new Set(['asset_class', 'dma', 'partner'])

type Cell = any
type Sheet = any

function isoToDate(iso?: string | null): Date | null {
  if (!iso) return null
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(iso)
  return m ? new Date(Date.UTC(+m[1], +m[2] - 1, +m[3])) : null
}

function font(cell: Cell, o: { b?: boolean; i?: boolean; sz?: number } = {}) {
  cell.font = { name: FONT, size: o.sz ?? 11, bold: !!o.b, italic: !!o.i }
}

function border(cell: Cell, sides: Record<string, string>) {
  const next: Record<string, any> = { ...(cell.border || {}) }
  for (const [side, style] of Object.entries(sides)) next[side] = { style }
  cell.border = next
}

function numFmtFor(key: string, markers?: number[], isTotal?: boolean): string | null {
  let f: string | null = null
  if (MONEY_KEYS.has(key)) f = MONEY_FMT
  else if (key === 'pref') f = isTotal ? MONEY_FMT : PREF_FMT
  else if (key === 'proceeds') f = isTotal ? PREF_FMT : PROCEEDS_FMT
  else if (PCT_KEYS.has(key)) f = '0.0%'
  if (f && markers && markers.length) {
    f += `" ${markers.map((m) => `(${m})`).join('')}"`
  }
  return f
}

/** Write one deal row, or a Total / Grand Total row. */
function writeValues(ws: Sheet, r: number, cols: any[], row: any, o: {
  bold?: boolean
  nameText?: string
  markers?: Record<string, number[]>
  isTotal?: boolean
}) {
  cols.forEach((c, i) => {
    if (c.key === '_spacer') return
    const cell = ws.getCell(r, i + 1)
    font(cell, { b: o.bold })
    if (c.key === 'name') {
      cell.value = o.nameText ?? ''
      return
    }
    if (c.key === 'invest_date') {
      const d = isoToDate(row?.invest_date)
      if (d && !o.isTotal) {
        cell.value = d
        cell.numFmt = 'mmm\\-yyyy'
      }
      cell.alignment = { horizontal: 'left' }
      return
    }
    if (TEXT_KEYS.has(c.key)) {
      if (!o.isTotal && row?.[c.key]) cell.value = row[c.key]
      if (c.key === 'partner') cell.alignment = { horizontal: 'left' }
      return
    }
    cell.alignment = { horizontal: 'center' }
    const label = row?.labels?.[c.key]
    if (label) {
      cell.value = label
      return
    }
    const v = row?.[c.key]
    if (v === null || v === undefined) return
    cell.value = v
    const f = numFmtFor(c.key, o.markers?.[c.key], o.isTotal)
    if (f) cell.numFmt = f
  })
}

function writeHeadings(ws: Sheet, r0: number, table: any, cols: any[]) {
  const group = table.cap_group || { start: -1, span: 0 }
  const pairs: any[] = table.cap_pairs || []
  const inGroup = (i: number) => i >= group.start && i < group.start + group.span
  const put = (r: number, i: number, v: string) => {
    if (!v || !String(v).trim()) return
    const cell = ws.getCell(r, i + 1)
    cell.value = v
    font(cell, { b: true })
    cell.alignment = { horizontal: 'center' }
  }
  // THE REFERENCE PUTS "PSC" OVER "Residual CF Split", ONE COLUMN RIGHT OF THE
  // "Pref Coupon" it belongs to (AB11, and AC88 on the Sold sheet). The screen
  // and the print sheet keep it over Pref Coupon; the Excel export follows the
  // reference, so the heading is moved one column here and nowhere else.
  const couponAt = cols.findIndex((c) => c.key === 'pref_coupon')
  cols.forEach((c, i) => {
    if (c.key === '_spacer') return
    if (i === group.start) {
      put(r0, i, group.heading)
      ws.mergeCells(r0, i + 1, r0, i + group.span)
    } else if (i === couponAt && couponAt >= 0) {
      /* moved one column right, below */
    } else if (i === couponAt + 1 && couponAt >= 0) put(r0, i, cols[couponAt].row1)
    else if (!inGroup(i)) put(r0, i, c.row1)
    const pair = pairs.find((p) => p.start === i)
    if (pair) {
      put(r0 + 1, i, pair.label)
      ws.mergeCells(r0 + 1, i + 1, r0 + 1, i + pair.span)
    } else if (!inGroup(i)) put(r0 + 1, i, c.row2)
    put(r0 + 2, i, c.row3)
  })
}

/** Returns the next free row. */
function addTable(ws: Sheet, data: any, table: any, r: number, o: {
  asOf?: boolean
  totalMarkers?: Record<string, number[]>
  grandTotal?: any
}): number {
  const cols: any[] = table.columns || []
  const n = cols.length
  const block = Math.min(BLOCK_COLS, n)
  const unitsAt = Math.min(UNITS_COL, n - 1)
  const unitsTo = Math.min(BLOCK_COLS - 1, n - 1)

  if (o.asOf) {
    const asOf = isoToDate(data.as_of)
    const cell = ws.getCell(r, unitsAt + 1)
    if (asOf) { cell.value = asOf; cell.numFmt = 'd\\-mmm\\-yy' }
    // Style every cell BEFORE merging: a merged cell takes its master's style.
    for (let c = 1; c <= block; c++) {
      font(ws.getCell(r, c), { b: true })
      border(ws.getCell(r, c), { top: 'medium' })
    }
    ws.mergeCells(r, unitsAt + 1, r, unitsTo + 1)
    r++
  }
  // Title (left) and units note (right), ruled underneath.
  const t = ws.getCell(r, 1)
  t.value = table.title
  const u = ws.getCell(r, unitsAt + 1)
  u.value = data.units_note
  u.alignment = { horizontal: 'right' }
  for (let c = 1; c <= block; c++) {
    font(ws.getCell(r, c), { b: true, i: c === unitsAt + 1 })
    border(ws.getCell(r, c), { bottom: 'medium' })
  }
  ws.mergeCells(r, unitsAt + 1, r, unitsTo + 1)
  r += 2                                              // blank row under the title

  const headRow = r
  writeHeadings(ws, r, table, cols)
  r += 4                                               // 3 heading rows + 1 blank

  let ri = 0
  for (const row of table.rows || []) {
    const m = (row.markers || []).map((k: number) => `(${k})`).join('')
    writeValues(ws, r, cols, row, {
      nameText: m ? `${row.name} ${m}` : row.name,
    })
    // The reference greys every SECOND deal row, starting with the second
    // (conditional formatting off its helper column). Written as a plain fill.
    if (ri % 2 === 1) {
      cols.forEach((c, i) => {
        if (c.key === '_spacer') return
        ws.getCell(r, i + 1).fill = {
          type: 'pattern', pattern: 'solid', fgColor: { argb: BAND },
        }
      })
    }
    ri++
    r++
  }
  r++                                                  // blank row before the total
  writeValues(ws, r, cols, table.total, {
    bold: true, nameText: table.total?.label || '', markers: o.totalMarkers,
    isTotal: true,
  })
  let last = r
  r++
  if (o.grandTotal) {
    r++
    writeValues(ws, r, cols, o.grandTotal, {
      bold: true, nameText: o.grandTotal.label || '', isTotal: true,
    })
    last = r
    r++
  }

  // The reference's thin vertical rules: left edge of the named columns, from
  // the first heading row to the foot of the last total row.
  // A rule inside the capitalization group starts under the group heading (it
  // would otherwise fall on a merged cell), as the reference's does.
  const grp = table.cap_group || { start: -1, span: 0 }
  for (const idx of table.vertical_rules || []) {
    const inGrp = idx >= grp.start && idx < grp.start + grp.span
    for (let rr = inGrp ? headRow + 1 : headRow; rr <= last; rr++) {
      border(ws.getCell(rr, idx + 1), { left: 'thin' })
    }
  }
  // The rules come in pairs, and each pair is a BOX in the reference: the
  // capitalization pair for PSC Pref. Equity, and the Proceeds-to-Act.-Since-
  // Close block. Boxes are ruled across the top (the first under the group
  // heading, the second on the first heading row). The first is closed at the
  // foot of the last total row; the second only on the Sold table, where the
  // Grand Total closes it -- on Current the reference leaves it open.
  const rules: number[] = table.vertical_rules || []
  const boxes: Array<[number, number]> = []
  for (let k = 0; k + 1 < rules.length; k += 2) boxes.push([rules[k], rules[k + 1]])
  boxes.forEach(([from, to], k) => {
    for (let c = from; c < to; c++) {
      border(ws.getCell(k === 0 ? headRow + 1 : headRow, c + 1), { top: 'thin' })
      if (k === 0 || o.grandTotal) border(ws.getCell(last, c + 1), { bottom: 'thin' })
    }
  })

  r += 2                                               // two blank rows
  for (const f of table.footnotes || []) {
    const cell = ws.getCell(r, 1)
    cell.value = `(${f.n}) ${f.text}`
    font(cell, { i: true, sz: 14 })
    r++
  }
  r += 2
  const d = ws.getCell(r, 1)
  d.value = data.disclaimer
  font(d, { i: true, sz: 14 })
  d.alignment = { horizontal: 'left', vertical: 'top', wrapText: true }
  ws.mergeCells(r, 1, r + 2, Math.min(21, n))
  for (let k = 0; k < 3; k++) ws.getRow(r + k).height = 24
  return r + 3
}

export function buildInvestmentMetricsWorkbook(ExcelJS: any, data: any): any {
  const wb = new ExcelJS.Workbook()
  wb.creator = 'Waterfall XIRR'
  const ws = wb.addWorksheet('Investment Metrics', {
    views: [{ showGridLines: false }],
    pageSetup: {
      orientation: 'landscape', fitToPage: true, fitToWidth: 1, fitToHeight: 0,
    },
  })
  WIDTHS.forEach((w, i) => { ws.getColumn(i + 1).width = w })
  ws.getColumn(WIDTHS.length + 1).width = 13        // Sold carries one more column

  let r = 1
  // A DRAFT banner must not disappear on the way out of the app.
  if (data.draft) {
    const b = ws.getCell(r, 1)
    b.value = data.draft_banner
    font(b, { b: true })
    r += 2
  }
  // Orange banner, and under it the strip of column labels, across I:AD.
  for (let c = 1; c <= BLOCK_COLS; c++) {
    const b = ws.getCell(r, c)
    b.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: ORANGE } }
    font(b, { b: true })
    b.alignment = { horizontal: 'center' }
    const [label, blue] = STRIP[c - 1]
    const sc = ws.getCell(r + 1, c)
    sc.value = label || null
    sc.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: ORANGE } }
    sc.font = { name: FONT, size: 11, bold: true, color: { argb: blue ? BLUE : 'FF000000' } }
    sc.alignment = { horizontal: 'center' }
  }
  ws.getCell(r, 1).value = BANNER
  ws.mergeCells(r, 1, r, BLOCK_COLS)
  r += 2
  r = addTable(ws, data, data.current, r, { asOf: true })
  r += 5
  addTable(ws, data, data.sold, r, {
    totalMarkers: data.sold.total_markers,
    grandTotal: data.grand_total,
  })
  return wb
}

export async function downloadInvestmentMetricsXlsx(data: any): Promise<void> {
  // Loaded on demand: the library is large and only this button needs it.
  const ExcelJS: any = (await import('exceljs')).default
  const wb = buildInvestmentMetricsWorkbook(ExcelJS, data)
  const buf = await wb.xlsx.writeBuffer()
  const blob = new Blob([buf], {
    type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `investment_metrics_${data.as_of || 'report'}.xlsx`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}
