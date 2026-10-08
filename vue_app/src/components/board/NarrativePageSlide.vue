<script setup lang="ts">
/**
 * One page of a narrative section: the blocks ``narrative.paginate`` put on it.
 * The type lives in the GLOBAL ``bd-narr`` styles below because the paginator
 * measures blocks in a hidden element with the same classes -- a scoped style
 * would not reach it, and a page measured in one type and drawn in another
 * would overflow.
 */
import BoardSlide from './BoardSlide.vue'
import type { Block } from './narrative'

defineProps<{
  title: string
  page: string | number
  blocks: Block[]
  images: Record<string, string>
  notes?: { footnotes: string[]; disclosure?: string | null }
}>()
</script>

<template>
  <BoardSlide :title="title" :page="page" :footnotes="notes?.footnotes" :disclosure="notes?.disclosure">
    <div class="bd-narr">
      <template v-for="(b, i) in blocks" :key="i">
        <h3 v-if="b.type === 'h'" class="bd-h">{{ b.text }}</h3>
        <p v-else-if="b.type === 'p'" class="bd-p">{{ b.text }}</p>
        <ul v-else-if="b.type === 'ul'" class="bd-ul"><li v-for="(it, j) in b.items" :key="j">{{ it }}</li></ul>
        <figure v-else-if="b.type === 'img'" class="bd-fig">
          <img v-if="images[`${b.aid}:${b.n}`]" :src="images[`${b.aid}:${b.n}`]"
               :style="{ width: b.dw + 'px', height: b.dh + 'px' }" alt="" />
          <div v-else class="bd-img-wait" :style="{ width: b.dw + 'px', height: b.dh + 'px' }">Loading attachment…</div>
          <figcaption v-if="b.caption">{{ b.caption }}</figcaption>
        </figure>
      </template>
    </div>
  </BoardSlide>
</template>

<style>
/* GLOBAL on purpose -- see the script comment. Everything is under .bd-narr. */
.bd-narr { padding: 0 20px; color: #000; font-family: Calibri, Carlito, 'Segoe UI', Arial, sans-serif; }
.bd-narr .bd-p { font-size: 19px; line-height: 1.42; margin: 0 0 12px; }
.bd-narr .bd-h { font-size: 21px; font-weight: 700; margin: 4px 0 8px; color: #1f4e79; }
.bd-narr .bd-ul { font-size: 19px; line-height: 1.42; margin: 0 0 12px; padding-left: 28px; }
.bd-narr .bd-ul li { margin-bottom: 4px; }
.bd-narr .bd-fig { margin: 0 0 14px; text-align: center; }
.bd-narr .bd-fig img { display: inline-block; border: 1px solid #d9d9d9; }
.bd-narr .bd-fig figcaption { font-size: 14px; font-style: italic; color: #404040; height: 26px; line-height: 26px; }
.bd-narr .bd-img-wait { display: inline-flex; align-items: center; justify-content: center; background: #f2f2f2;
  color: #6b7280; font-size: 14px; }
</style>
