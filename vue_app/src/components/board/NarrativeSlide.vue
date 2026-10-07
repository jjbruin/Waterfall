<script setup lang="ts">
/** A narrative page (strategy, outlook, ...): the text the editors wrote, set as the deck sets text. */
import { computed } from 'vue'
import BoardSlide from './BoardSlide.vue'

const props = defineProps<{ title: string; body: string; page: string | number }>()

// Blank lines separate paragraphs; a line starting "-" or "•" is a bullet.
const blocks = computed(() => props.body.split(/\n\s*\n/).map((p) => {
  const lines = p.split('\n').map((l) => l.trim()).filter(Boolean)
  const bullets = lines.length && lines.every((l) => /^[-•]\s*/.test(l))
  return bullets ? { list: lines.map((l) => l.replace(/^[-•]\s*/, '')) } : { text: lines.join(' ') }
}))
</script>

<template>
  <BoardSlide :title="title" :page="page">
    <div class="narr">
      <template v-for="(b, i) in blocks" :key="i">
        <ul v-if="b.list"><li v-for="(l, j) in b.list" :key="j">{{ l }}</li></ul>
        <p v-else>{{ b.text }}</p>
      </template>
    </div>
  </BoardSlide>
</template>

<style scoped>
.narr { font-size: 20px; line-height: 1.45; color: #000; padding: 10px 20px; }
.narr p { margin: 0 0 14px; }
.narr ul { margin: 0 0 14px; padding-left: 26px; }
.narr li { margin-bottom: 6px; }
</style>
