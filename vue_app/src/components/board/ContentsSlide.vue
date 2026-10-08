<script setup lang="ts">
/**
 * The table of contents: each part, and each page set in it, with the page
 * number it PRINTS on. The numbers are the deck's own positions, worked out
 * after narrative sections have been paginated, so they cannot drift from the
 * pages. In the viewer an entry turns to its page.
 */
import BoardSlide from './BoardSlide.vue'

defineProps<{
  parts: { roman: string; title: string; page: number; items: { title: string; page: number }[] }[]
  page: string | number
  notes?: { footnotes: string[]; disclosure?: string | null }
}>()
const emit = defineEmits<{ (e: 'go', page: number): void }>()
const ROMAN = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x', 'xi', 'xii', 'xiii', 'xiv', 'xv']
</script>

<template>
  <BoardSlide title="Table of Contents" :page="page" :footnotes="notes?.footnotes" :disclosure="notes?.disclosure">
    <div class="toc">
      <template v-for="p in parts" :key="p.roman">
        <div class="row part" @click="emit('go', p.page)">
          <span class="n">{{ p.roman }}.</span><span class="t">{{ p.title }}</span><span class="dots"></span>
          <span class="pg">{{ p.page }}</span>
        </div>
        <div v-for="(it, i) in p.items" :key="p.roman + i" class="row item" @click="emit('go', it.page)">
          <span class="n">{{ ROMAN[i] || i + 1 }}.</span><span class="t">{{ it.title }}</span><span class="dots"></span>
          <span class="pg">{{ it.page }}</span>
        </div>
      </template>
    </div>
  </BoardSlide>
</template>

<style scoped>
.toc { padding: 4px 60px 0; font-size: 21px; line-height: 1.38; color: #000; }
.row { display: flex; align-items: baseline; cursor: pointer; }
.row:hover .t { text-decoration: underline; }
.row .n { width: 46px; flex: none; }
.row.item { padding-left: 54px; font-size: 19px; }
.row.item .n { width: 44px; }
.row.part { margin-top: 6px; font-weight: 600; }
.row .t { flex: none; color: #0563c1; }
.row .dots { flex: 1; border-bottom: 1px dotted #9ca3af; margin: 0 8px 6px; }
.row .pg { flex: none; width: 34px; text-align: right; font-variant-numeric: tabular-nums; }
</style>
