<script setup lang="ts">
/**
 * One board-deck page, drawn on the deck's own canvas: 1100 x 825 (4:3), the
 * green rule across the top, the title, "INTERNAL USE ONLY" and the page number
 * at the foot -- the January 2026 deck's frame. The viewer scales the whole
 * canvas to fit the screen, so what is on screen is laid out exactly as the
 * page will be, at any window size. A page is always white, as printed.
 */
defineProps<{ title: string; page: string | number }>()
</script>

<template>
  <div class="slide">
    <div class="bar"></div>
    <h1 class="title">{{ title }}</h1>
    <div class="body"><slot /></div>
    <div class="foot">INTERNAL USE ONLY</div>
    <div class="pg">{{ page }}</div>
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
.body { position: absolute; left: 55px; right: 55px; top: 112px; bottom: 58px; }
.foot { position: absolute; left: 78px; bottom: 18px; font-size: 13px; color: #595959; letter-spacing: .2px; }
.pg { position: absolute; right: 40px; bottom: 36px; font-size: 13px; color: #595959; }
</style>
