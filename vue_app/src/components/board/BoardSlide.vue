<script setup lang="ts">
/**
 * One board-deck page, drawn on the deck's own canvas: 1100 x 825 (4:3), the
 * green rule across the top, the title, "INTERNAL USE ONLY" and the page number
 * at the foot -- the January 2026 deck's frame. The viewer scales the whole
 * canvas to fit the screen, so what is on screen is laid out exactly as the
 * page will be, at any window size. A page is always white, as printed.
 *
 * THE FOOTNOTES AND THE DISCLOSURE ARE THE FRAME'S, on every page: one place,
 * one style, edited from one panel. The page's content gets whatever height is
 * left above them.
 */
defineProps<{
  title?: string
  page: string | number
  footnotes?: string[]
  disclosure?: string | null
  hideNumber?: boolean
}>()
</script>

<template>
  <div class="slide">
    <div class="bar"></div>
    <h1 v-if="title" class="title">{{ title }}</h1>
    <div class="main" :class="{ untitled: !title }">
      <div class="body"><slot /></div>
      <div v-if="(footnotes && footnotes.length) || disclosure" class="notes">
        <div v-for="(f, i) in footnotes || []" :key="i" class="fn">{{ f }}</div>
        <div v-if="disclosure" class="disc">{{ disclosure }}</div>
      </div>
    </div>
    <div class="foot">INTERNAL USE ONLY</div>
    <div v-if="!hideNumber" class="pg">{{ page }}</div>
  </div>
</template>

<style scoped>
.slide {
  position: relative; width: 1100px; height: 825px; overflow: hidden;
  background: #fff; color: #000;
  font-family: Calibri, Carlito, 'Segoe UI', Arial, sans-serif;
}
.bar { position: absolute; left: 55px; right: 55px; top: 0; height: 8px; background: #00b274; }
.title { position: absolute; left: 60px; top: 34px; margin: 0; font-size: 40px; font-weight: 400;
  line-height: 1.1; color: #000; }
.main { position: absolute; left: 55px; right: 55px; top: 112px; bottom: 56px; display: flex; flex-direction: column; }
.main.untitled { top: 40px; }
.body { position: relative; flex: 1; min-height: 0; }
.notes { flex: none; padding-top: 8px; font-size: 11.5px; line-height: 1.35; color: #000; }
.fn { font-style: italic; }
.disc { margin-top: 4px; font-size: 10.5px; color: #404040; }
.foot { position: absolute; left: 78px; bottom: 18px; font-size: 13px; color: #595959; letter-spacing: .2px; }
.pg { position: absolute; right: 40px; bottom: 36px; font-size: 13px; color: #595959; }
</style>
