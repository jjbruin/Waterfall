<script setup lang="ts">
/**
 * The input side of a two-column page, collapsible to give the analysis the room.
 *
 * Jim, Sep 19 2026: "many of our pages have input sections on the left and
 * analysis sections on the right. Can we provide the same little arrow as exists
 * in the side bar to expand or collapse the input sections throughout the app?
 * This would make more room on the screen for the analysis."
 *
 * Same idiom as `AppSidebar`'s own toggle -- a plain `<` / `>` -- so it reads as
 * the one control it already is rather than a new vocabulary per page.
 *
 * TWO THINGS THIS DELIBERATELY DOES NOT DO:
 *
 * It does not hide its own toggle. The collapsed rail keeps the arrow AND the
 * panel's name, because a control that vanishes when used cannot be undone by
 * anyone who did not already know it was there -- v482 shipped a clear button
 * that was invisible until hover and had to be fixed after it could not be
 * found.
 *
 * It does not size the parent. A grid track is set on the container, so a child
 * narrowing itself reclaims nothing; each page binds `input-collapsed` on its own
 * layout element and states the collapsed track in its own CSS. Explicit beats a
 * component reaching upward into a layout it cannot see.
 */
import { ref, watch, onMounted } from 'vue'

const props = withDefaults(defineProps<{
  /** Shown on the collapsed rail so a hidden panel still says what it is. */
  label?: string
  /** Remembers the choice for this panel, per browser. Omit to always open. */
  storageKey?: string
}>(), { label: 'Inputs', storageKey: '' })

const collapsed = defineModel<boolean>({ default: false })

const KEY = props.storageKey ? `panel.collapsed.${props.storageKey}` : ''

// Browser storage is a per-viewer convenience here and nothing depends on it:
// a private window, cleared site data or a throwing accessor just means the
// panel opens, which is the right default anyway.
onMounted(() => {
  if (!KEY) return
  try {
    const v = localStorage.getItem(KEY)
    if (v !== null) collapsed.value = v === '1'
  } catch { /* opens, as if never set */ }
})

watch(collapsed, (v) => {
  if (!KEY) return
  try { localStorage.setItem(KEY, v ? '1' : '0') } catch { /* not remembered */ }
})

function toggle() { collapsed.value = !collapsed.value }
</script>

<template>
  <div class="cpanel" :class="{ collapsed }">
    <button class="cpanel-toggle" type="button" @click="toggle"
            :aria-expanded="!collapsed"
            :title="collapsed ? `Show ${label}` : `Hide ${label}`">
      {{ collapsed ? '>' : '<' }}
    </button>
    <div v-if="collapsed" class="cpanel-rail" @click="toggle"
         :title="`Show ${label}`">{{ label }}</div>
    <div v-show="!collapsed" class="cpanel-body"><slot /></div>
  </div>
</template>

<style scoped>
.cpanel { position: relative; min-width: 0; }

/* Collapsed, the panel is a rail wide enough for the arrow and the label. The
   page's own CSS narrows the grid track or flex basis to match. */
.cpanel.collapsed { width: 30px; min-width: 30px; max-width: 30px; flex: 0 0 30px; }

.cpanel-toggle {
  position: absolute; top: 0; right: 0; z-index: 2;
  background: none; border: none; cursor: pointer;
  color: #6b7686; font-size: 14px; line-height: 1; padding: 3px 5px;
}
.cpanel-toggle:hover { color: #1f2937; }
.cpanel.collapsed .cpanel-toggle { right: auto; left: 0; }

/* The name, turned on its side, so a collapsed panel is not an anonymous strip. */
.cpanel-rail {
  writing-mode: vertical-rl; text-orientation: mixed;
  margin-top: 24px; padding: 6px 0;
  font-size: 11px; letter-spacing: 0.06em; text-transform: uppercase;
  color: #6b7686; cursor: pointer; user-select: none; white-space: nowrap;
}
.cpanel-rail:hover { color: #1f2937; }

/* Room for the arrow so it never lands on the first control. */
.cpanel-body { padding-top: 18px; }
</style>
