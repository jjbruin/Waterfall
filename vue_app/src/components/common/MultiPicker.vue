<script setup lang="ts">
/**
 * A multi-select that LOOKS like one.
 *
 * This replaced a native `<select multiple>`. Everything below was already
 * possible with one — you ctrl-click, or shift-click a range — but nothing on
 * screen says so, so it reads as a list where picking a second item drops the
 * first. The CFO asked whether he could select several entities and several
 * accounts in the same query; he already could, and that is the clearest
 * evidence the control was answering the wrong question.
 *
 * Checkboxes, a search box for the long lists, a count, and Select all / Clear.
 */
import { computed, ref } from 'vue'

const props = withDefaults(defineProps<{
  modelValue: string[]
  /** { id, label } — label is what is searched and shown. */
  options: Array<{ id: string; label: string }>
  label: string
  /** Show the search box. Off for short lists, where it is just clutter. */
  searchable?: boolean
  maxHeight?: string
  hint?: string
}>(), { searchable: true, maxHeight: '190px' })

const emit = defineEmits<{ (e: 'update:modelValue', v: string[]): void }>()

const term = ref('')

const shown = computed(() => {
  const t = term.value.trim().toLowerCase()
  if (!t) return props.options
  return props.options.filter(o => o.label.toLowerCase().includes(t))
})

const selected = computed(() => new Set(props.modelValue))

function toggle(id: string) {
  const s = new Set(props.modelValue)
  s.has(id) ? s.delete(id) : s.add(id)
  // Emitted in the options' own order, so the criteria line on an export reads
  // the same way twice for the same set.
  emit('update:modelValue', props.options.filter(o => s.has(o.id)).map(o => o.id))
}

/** Select all applies to what the SEARCH is showing, not the whole list — with a
 *  filter typed, "all" meaning every hidden option too would be a trap. */
function selectAllShown() {
  const s = new Set(props.modelValue)
  shown.value.forEach(o => s.add(o.id))
  emit('update:modelValue', props.options.filter(o => s.has(o.id)).map(o => o.id))
}
function clearAll() { emit('update:modelValue', []) }
</script>

<template>
  <div class="mp">
    <div class="mp-head">
      <label>{{ label }}</label>
      <span class="mp-count">
        {{ modelValue.length ? `${modelValue.length} selected` : 'all' }}
      </span>
    </div>

    <div class="mp-actions">
      <button type="button" class="mp-link" @click="selectAllShown">
        {{ term.trim() ? 'Select shown' : 'Select all' }}
      </button>
      <button type="button" class="mp-link" :disabled="!modelValue.length"
              @click="clearAll">Clear</button>
    </div>

    <input v-if="searchable" v-model="term" class="mp-search" type="search"
           :placeholder="`Search ${label.toLowerCase()}…`" />

    <div class="mp-list" :style="{ maxHeight }">
      <label v-for="o in shown" :key="o.id" class="mp-item"
             :class="{ on: selected.has(o.id) }">
        <input type="checkbox" :checked="selected.has(o.id)" @change="toggle(o.id)" />
        <span class="mp-label">{{ o.label }}</span>
      </label>
      <!-- A search matching nothing says so, rather than showing an empty box that
           reads as "there are none of these". -->
      <div v-if="!shown.length" class="mp-empty">
        {{ options.length ? 'Nothing matches that search.' : 'Nothing to choose from.' }}
      </div>
    </div>

    <div v-if="hint" class="mp-hint">{{ hint }}</div>
  </div>
</template>

<style scoped>
.mp { display: flex; flex-direction: column; min-width: 230px; }
.mp-head { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; }
.mp-head label {
  font-size: 11px; text-transform: uppercase; letter-spacing: .03em;
  color: var(--color-text-secondary);
}
.mp-count { font-size: 11px; font-style: italic; color: var(--color-text-secondary); }
.mp-actions { display: flex; gap: 10px; margin: 3px 0 4px; }
.mp-link {
  border: none; background: none; padding: 0; cursor: pointer;
  font-size: 11px; color: var(--color-primary, #2f6f4f); text-decoration: underline;
}
.mp-link:disabled { color: var(--color-text-secondary); cursor: default; text-decoration: none; }
.mp-search {
  border: 1px solid var(--color-border); border-radius: 4px; padding: 4px 6px;
  font-size: 12.5px; margin-bottom: 4px;
  background: var(--color-surface); color: var(--color-text);
}
.mp-list {
  overflow-y: auto; border: 1px solid var(--color-border); border-radius: 4px;
  background: var(--color-surface);
}
.mp-item {
  display: flex; align-items: center; gap: 7px; padding: 3px 7px;
  font-size: 12.5px; cursor: pointer; line-height: 1.3;
}
.mp-item:hover { background: rgba(127, 127, 127, 0.10); }
.mp-item.on { background: rgba(47, 111, 79, 0.12); }
.mp-item input { flex: none; cursor: pointer; }
.mp-label { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.mp-empty { padding: 8px; font-size: 12px; color: var(--color-text-secondary); }
.mp-hint { font-size: 11px; color: var(--color-text-secondary); margin-top: 5px; line-height: 1.35; }
</style>
