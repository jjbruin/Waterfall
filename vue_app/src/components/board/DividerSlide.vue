<script setup lang="ts">
/** A part's divider page, as the deck's ("I. Company Highlights & Goals"). */
import BoardSlide from './BoardSlide.vue'

defineProps<{
  roman: string
  title: string
  page: string | number
  notes?: { footnotes: string[]; disclosure?: string | null }
}>()
</script>

<template>
  <BoardSlide :page="page" :footnotes="notes?.footnotes" :disclosure="notes?.disclosure">
    <div class="div"><span>{{ roman }}. {{ title }}</span></div>
  </BoardSlide>
</template>

<style scoped>
.div { height: 100%; display: flex; align-items: center; justify-content: center; }
.div span { font-size: 46px; color: #000; padding-bottom: 10px; border-bottom: 4px solid #00b274; }
</style>
