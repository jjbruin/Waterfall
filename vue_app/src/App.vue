<script setup lang="ts">
import { computed, watch } from 'vue'
import { useRoute } from 'vue-router'
import { useAuthStore } from './stores/auth'
import { useVersionCheck } from './composables/useVersionCheck'
import AppSidebar from './components/layout/AppSidebar.vue'
import AppHeader from './components/layout/AppHeader.vue'
import ToastNotifications from './components/common/ToastNotifications.vue'
import AiAssistant from './components/common/AiAssistant.vue'

const route = useRoute()
const auth = useAuthStore()
const { updateAvailable, onNavigate, reloadApp } = useVersionCheck()

const isLoginPage = computed(() => route.path === '/login')

// Check for new version on every route change
watch(() => route.path, () => onNavigate())
</script>

<template>
  <ToastNotifications />
  <div v-if="updateAvailable" class="update-banner" @click="reloadApp">
    A new version is available. Click here to reload.
  </div>
  <div v-if="isLoginPage" class="login-layout">
    <router-view />
  </div>
  <div v-else class="app-layout">
    <AppSidebar />
    <div class="main-content">
      <AppHeader />
      <main class="page-content">
        <router-view />
      </main>
    </div>
    <AiAssistant />
  </div>
</template>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

:root {
  --sidebar-width: 240px;
  --header-height: 48px;
  --color-primary: #1F4E79;
  --color-pref: #548235;
  --color-op: #A6A6A6;
  --color-floating: #ED7D31;
  --color-accent: #4472C4;
  --color-bg: #f8f9fa;
  --color-surface: #ffffff;
  --color-text: #212529;
  --color-text-secondary: #6c757d;
  --color-border: #dee2e6;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  color: var(--color-text);
  background: var(--color-bg);
}

.app-layout {
  display: flex;
  min-height: 100vh;
}

.main-content {
  flex: 1;
  margin-left: var(--sidebar-width);
  display: flex;
  flex-direction: column;
  transition: margin-left 0.2s;
}

.page-content {
  padding: 24px;
  flex: 1;
}

.login-layout {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
  background: var(--color-bg);
}

/* Responsive: on narrow screens, sidebar overlaps content */
@media (max-width: 900px) {
  :root {
    --sidebar-width: 200px;
  }
}

.update-banner {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 10000;
  background: #1565c0;
  color: white;
  text-align: center;
  padding: 10px 16px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}
.update-banner:hover {
  background: #0d47a1;
}

/* Print: hide app shell, let page content fill the page */
@media print {
  /* THE ONLY @page RULE IN THE APPLICATION. Keep it that way.
     ------------------------------------------------------------------
     `@page` cannot be scoped. Vue's scoped CSS works by appending a
     [data-v-hash] attribute to selectors, and an at-rule that styles the page
     box has no selector to append to — so a `@page` written inside any
     component's <style scoped> escapes into the global sheet and applies to
     every route in the app. Routes are lazy-loaded and Vite leaves a route's
     stylesheet in the document after you navigate away, so with more than one
     `@page` in the build the winner is decided by which route you happened to
     visit last. That is not a hypothetical: a `@page { margin: 0.5in }` added
     to the Portfolio Snapshot on Aug 24 2026 (830934d) silently changed the
     One Pager's printed output, growing its side margins and — because a
     non-zero page margin is what gives Chrome room to draw its own header and
     footer — putting the "Waterfall XIRR" title back on the page. Nothing in
     OnePagerView.vue had been touched since July.

     Zero margin is deliberate and load-bearing: it is what suppresses the
     browser's own header/footer (title, URL, date, page number). Each print
     view supplies its real margins as padding on its own container, which
     also lets them differ per view without fighting over one page box.

     scripts/print_page_rule_check.py fails the build if a second one appears. */
  @page {
    size: letter portrait;
    margin: 0;
  }

  .sidebar {
    display: none !important;
  }
  .app-header {
    display: none !important;
  }
  .main-content {
    margin-left: 0 !important;
  }
  .page-content {
    padding: 0 !important;
  }
}
</style>
