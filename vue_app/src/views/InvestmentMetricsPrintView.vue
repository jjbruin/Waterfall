<script setup lang="ts">
/**
 * The printed PSC Investment Summary — Current on sheet 1, Sold on sheet 2.
 *
 * Opened in its own tab by the Print button on InvestmentMetricsView, on the
 * same pattern as the One Pager and the Portfolio Snapshot. It prints itself
 * once the data is in.
 *
 * WHY THE SHEET LOOKS LIKE THIS. Every dimension below was measured off the
 * reference PDF rather than chosen: the page is letter LANDSCAPE (792 x 612pt)
 * with the table drawn from x=18pt to x=734pt and the border box from y=53.64
 * to y=404.16, so the document occupies the top 57% of the sheet and the rest
 * is white. That is not an oversight in the reference — it is a spreadsheet
 * printed to fit one page wide — and stretching the table down the page to
 * "use the space" would not look like the document it replaces.
 *
 * THE BODY TEXT IS 3.6pt AND THE FOOTNOTES ARE 4.68pt. That is genuinely what
 * the reference uses, and the footnotes really are LARGER than the table. It
 * reads as a mistake and is not one.
 */
import { computed, nextTick, onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import api from '@/api/client'
import InvestmentMetricsTable from '@/components/reports/InvestmentMetricsTable.vue'

const route = useRoute()
const data = ref<any>(null)
const loadError = ref('')
const loading = ref(true)

const page = computed(() => data.value?.page || {})

const ROW = 4.68          // the grid everything sits on, in points
const TITLEBAR = 2 * ROW  // as-of line + title line
const HEADER = 3 * ROW    // the three stacked heading rows

/**
 * The vertical rules, as positioned overlays rather than cell borders.
 *
 * THEY CANNOT BE `border-left` ON A CELL. Measured on the rendered sheet, a
 * cell border came out as a run of disconnected fragments — present on some
 * rows and missing on others, because a shaded row's background and the
 * neighbouring cell's fill take turns covering it. The reference draws each
 * rule as ONE tall mark from the heading down to the foot of the total row,
 * and a rule with holes in it reads as a printing fault rather than a column
 * group.
 *
 * Each rule's left edge is the running sum of the column widths in front of
 * it — the same widths the table's <col> elements use, so the rule and the
 * column boundary cannot drift apart. A rule inside the capitalization block
 * starts one row lower, where its pair heading begins.
 */
function verticalRules(tbl: any) {
  if (!tbl?.columns) return []
  const capStart = tbl.cap_group?.start ?? 6
  const capEnd = capStart + (tbl.cap_group?.span ?? 6)
  const offsets: number[] = []
  let x = 0
  for (const c of tbl.columns) { offsets.push(x); x += c.width }
  // Down to the foot of the total row: title block, the blank row under it,
  // three heading rows, the blank lead row, the deals, a blank, the total.
  const bottom = TITLEBAR + ROW + HEADER + ROW + (tbl.rows.length + 2) * ROW
  return (tbl.vertical_rules || []).map((i: number) => {
    // A rule inside the capitalization block starts a row lower, level with
    // its pair heading rather than with the group heading above it.
    const inCap = i >= capStart && i < capEnd
    const top = TITLEBAR + ROW + (inCap ? ROW : 0)
    return { left: offsets[i], top, height: bottom - top }
  })
}

/**
 * The document title is blanked while printing so the browser cannot put
 * "Waterfall XIRR" across the top of every sheet. `@page { margin: 0 }` in
 * App.vue is the other half of that — a zero page margin leaves Chrome no room
 * to draw a header — and this sheet's own padding stands in for the margin.
 *
 * NO RENDERED TIMESTAMP, for the same reason the Portfolio Snapshot dropped
 * its one: the reference document carries none, and a printed "30/09/2026,
 * 11:42" in the corner is ours, not the browser's.
 */
function doPrint() {
  const orig = document.title
  document.title = ' '
  nextTick(() => {
    window.print()
    document.title = orig
  })
}

onMounted(async () => {
  try {
    const asOf = (route.query.as_of as string) || ''
    const res = await api.get('/api/investment-metrics',
      { params: asOf ? { as_of: asOf } : {} })
    data.value = res.data
  } catch (e: any) {
    loadError.value = e?.response?.data?.error || e?.message || 'failed to load'
  } finally {
    loading.value = false
  }
  if (!loadError.value && route.query.autoprint !== '0') {
    await nextTick()
    setTimeout(doPrint, 350)
  }
})
</script>

<template>
  <div class="print-doc">
    <div v-if="loading" class="msg">Loading…</div>
    <div v-else-if="loadError" class="msg err">{{ loadError }}</div>

    <template v-else>
      <section
        v-for="(part, idx) in ['current', 'sold']"
        :key="part"
        class="sheet"
        :class="{ last: idx === 1 }"
      >
        <!--
          THE DRAFT MARK ADDS NO LAYOUT. Both pieces are absolutely positioned
          on the sheet, outside the frame's flow, so the table's geometry is
          bit-for-bit what it is with the flag off — which is what lets the
          print-fidelity guardrail keep measuring the real document. The
          watermark sits in the blank lower half the reference leaves empty,
          and the line above the frame uses the sheet's own top margin.
        -->
        <template v-if="data.draft">
          <div class="draft-line">{{ data.draft_banner }}</div>
          <div class="draft-wm">{{ data.draft_mark }}</div>
        </template>

        <div class="frame" :class="part">
          <!--
            The title block is always TWO rows of the same 4.68pt grid as the
            table. Row 1 carries the as-of date on the right and row 2 the
            title on the left with the units note on the right — and on the
            Sold sheet row 1 is EMPTY, because the reference prints the date
            once, on the Current page only.
          -->
          <div class="titlebar">
            <div class="tline">
              <!-- The empty span is load-bearing: `space-between` with a
                   single child aligns it LEFT, and the date belongs in the
                   top-right corner. -->
              <span></span>
              <span class="asof">{{ part === 'current' ? data.as_of_display : '' }}</span>
            </div>
            <div class="tline">
              <span class="ttl">{{ data[part].title }}</span>
              <span class="units">{{ data.units_note }}</span>
            </div>
          </div>

          <div
            v-for="(r, ri) in verticalRules(data[part])"
            :key="'vr' + ri"
            class="vrule-overlay"
            :style="{ left: r.left + 'pt', top: r.top + 'pt', height: r.height + 'pt' }"
          ></div>

          <InvestmentMetricsTable
            :table="data[part]"
            :total-markers="part === 'sold' ? data.sold.total_markers : undefined"
            :grand-total="part === 'sold' ? data.grand_total : undefined"
          />

          <div class="notes">
            <div v-for="f in data[part].footnotes" :key="f.n" class="note">
              ({{ f.n }}) {{ f.text }}
            </div>
          </div>
          <div class="disclaimer">{{ data.disclaimer }}</div>
        </div>
      </section>
    </template>
  </div>
</template>

<style scoped>
/* ── screen preview ───────────────────────────────────────────────────── */
.print-doc { background: #f3f4f6; padding: 16px; }
.msg { padding: 24px; font: 14px system-ui, sans-serif; }
.msg.err { color: #b91c1c; }
.sheet {
  background: #fff;
  width: 792pt;
  height: 612pt;
  margin: 0 auto 16px auto;
  box-shadow: 0 1px 6px rgba(0, 0, 0, .25);
  position: relative;
}

/* The bordered box and everything in it, in the reference's own points.
   EVERYTHING SITS ON A 4.68pt GRID — the title block is two of those rows, the
   header three, each table row one, and the gaps between blocks whole numbers
   of them. That is not a tidy coincidence: the reference is a spreadsheet
   printed to fit, so its "spacing" is blank rows. */
.frame {
  position: absolute;
  overflow: hidden;
  left: 18pt;
  top: 53.64pt;
  width: 716.16pt;
  /* OUTLINE, NOT BORDER. A border takes layout space, so every column inside
     would sit 0.48pt right of where the reference puts it and every vertical
     rule with it. An outline is drawn outside the box and shifts nothing. */
  outline: 0.48pt solid #000;
  box-sizing: border-box;
  font-family: Garamond, "EB Garamond", "Adobe Garamond Pro",
               "Cormorant Garamond", Georgia, "Times New Roman", serif;
  font-size: 3.6pt;
  line-height: 4.68pt;
  color: #000;
}
/* The Sold sheet's frame starts three rows lower than the Current sheet's —
   measured, not assumed: the reference's top border is at y=53.64 on page 1
   and y=67.68 on page 2. */
.frame.sold { top: 67.68pt; }

.titlebar {
  font-weight: bold;
  border-bottom: 0.48pt solid #000;
  height: 9.36pt;
  box-sizing: border-box;
}
.tline {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  height: 4.68pt;
  line-height: 4.68pt;
  padding: 0 3.12pt 0 3.6pt;
}
.units, .asof { white-space: nowrap; }

:deep(.im-grid) {
  border-collapse: separate;
  border-spacing: 0;
}
:deep(.im-grid th),
:deep(.im-grid td) {
  height: 4.68pt;
  line-height: 4.68pt;
  font-size: 3.6pt;
}
/* The name column's text starts 3.6pt inside the border, as the reference's
   does; every other column sits flush on its own boundary. */
:deep(.im-grid td:first-child),
:deep(.im-grid th:first-child) { padding-left: 3.6pt; }

:deep(.im-grid thead th) { font-weight: bold; }
/* 1.08pt is the reference's own gap between one column's rule and the next. */
:deep(.im-grid thead .rule) {
  border-bottom: 0.12pt solid #000;
  margin-right: 1.08pt;
}

/* ALTERNATE ROWS ARE SHADED. The reference bands them #D9D9D9 — sampled off
   the rasterized page, not guessed — and the shading stops before the trailing
   spacer column. Set here rather than in the shared component because the
   screen view deliberately bands more lightly.

   `background-clip: padding-box` keeps the band off the cell's own left
   border, which is how the vertical rules survive crossing a shaded row. */
:deep(.im-grid tbody tr.band td:not(.spacer)) {
  background: #D9D9D9;
  background-clip: padding-box;
}
/* The cell-border form is switched OFF on the printed sheet; the overlays
   above draw these. Leaving both on would double the rule where they agree and
   show the fragments where they do not. */
:deep(.im-grid .vrule) { border-left: none; }
/* z-index is load-bearing. Without it the shaded rows paint OVER the rule and
   it comes out as a row of dashes — present on the white rows, gone on the
   grey ones — which reads as a printing fault rather than a column group.
   Measured on the rendered sheet before the fix. */
.vrule-overlay {
  position: absolute;
  z-index: 3;
  width: 0.24pt;
  background: #000;
}

/* ── the draft mark ───────────────────────────────────────────────────── */
/* Both pieces are positioned against the SHEET, so neither is in the frame's
   flow and neither can move a column. With the flag off they are not rendered
   at all — not hidden, absent — so the printed document is the reference's
   geometry exactly. */
.draft-line {
  position: absolute;
  top: 6pt;
  left: 18pt;
  font-family: Garamond, Georgia, "Times New Roman", serif;
  font-size: 7pt;
  font-weight: bold;
  letter-spacing: .06em;
  color: #b91c1c;
}
.draft-wm {
  position: absolute;
  left: 0;
  right: 0;
  top: 430pt;                 /* the blank lower half the reference leaves */
  text-align: center;
  font-family: Garamond, Georgia, "Times New Roman", serif;
  font-size: 66pt;
  font-weight: bold;
  letter-spacing: .3em;
  color: rgba(185, 28, 28, .18);
  pointer-events: none;
  user-select: none;
}
:deep(.im-grid tbody tr.gap td) { height: 4.68pt; background: none; }
:deep(.im-grid tbody tr.total td) { font-weight: bold; }
:deep(.im-grid .mark) { font-weight: inherit; }

/* A long MRI deal name must CLIP at its column, not run across the next one.
   `overflow: hidden` on a table cell is not reliable — the cell is not a block
   container for this purpose — so the clipping happens on an inner block. */
:deep(.im-grid tbody td) { position: relative; }
:deep(.im-grid tbody td > *),
:deep(.im-grid tbody td) {
  overflow: hidden;
  text-overflow: clip;
}

.notes { padding: 9.36pt 3.6pt 0 3.6pt; }
.note,
.disclaimer {
  font-style: italic;
  font-size: 4.68pt;
  line-height: 4.68pt;
}
.disclaimer {
  padding: 9.36pt 36pt 6pt 3.6pt;
  text-align: left;
}

/* ── the printed document ─────────────────────────────────────────────── */
@media print {
  /* The page box is declared ONCE, globally, in App.vue — `@page` cannot be
     scoped, so a second one here would change every other printing view in the
     app depending on which route was visited last (v421). This asks for the
     named landscape box; nothing else. */
  .print-doc { background: #fff; padding: 0; }
  /* The app's own page tint reaches the printer otherwise: the rendered sheet
     came out filled #F8F9FA edge to edge against a reference that is pure
     white. Nothing on screen shows it, because on screen the sheet is meant to
     sit on a tint. */
  :global(html), :global(body), :global(#app) { background: #fff !important; }
  /* The draft mark must survive the trip to the printer — Chrome drops
     colour unless asked, and a watermark printed white is no watermark. */
  .draft-line, .draft-wm {
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }
  .sheet {
    page: landscape-sheet;
    box-shadow: none;
    margin: 0;
    width: 792pt;
    height: 612pt;
    page-break-after: always;
    break-after: page;
  }
  /* The last sheet must not emit a trailing blank page. */
  .sheet.last { page-break-after: auto; break-after: auto; }
  .no-print { display: none !important; }
}
</style>
