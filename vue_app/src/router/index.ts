import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from '../stores/auth'

const routes = [
  {
    path: '/login',
    name: 'Login',
    component: () => import('../views/LoginView.vue'),
    meta: { requiresAuth: false },
  },
  {
    path: '/',
    redirect: '/dashboard',
  },
  {
    path: '/dashboard',
    name: 'Dashboard',
    component: () => import('../views/DashboardView.vue'),
  },
  {
    path: '/deal-analysis',
    name: 'Deal Analysis',
    component: () => import('../views/DealAnalysisView.vue'),
  },
  {
    path: '/property-financials',
    name: 'Property Financials',
    component: () => import('../views/PropertyFinancialsView.vue'),
  },
  {
    path: '/ownership',
    name: 'Ownership & Partnerships',
    component: () => import('../views/OwnershipView.vue'),
  },
  {
    path: '/waterfall-setup',
    name: 'Waterfall Setup',
    component: () => import('../views/WaterfallSetupView.vue'),
  },
  {
    path: '/reports',
    name: 'Reports',
    component: () => import('../views/ReportsView.vue'),
  },
  {
    path: '/sold-portfolio',
    name: 'Sold Portfolio',
    component: () => import('../views/SoldPortfolioView.vue'),
  },
  {
    path: '/psckoc',
    name: 'PSCKOC',
    component: () => import('../views/PsckocView.vue'),
  },
  {
    path: '/settings',
    name: 'Settings',
    component: () => import('../views/SettingsView.vue'),
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

// Navigation guard
router.beforeEach((to) => {
  const auth = useAuthStore()
  if (to.meta.requiresAuth !== false && !auth.isAuthenticated) {
    return { name: 'Login', query: { redirect: to.fullPath } }
  }
})

export default router
