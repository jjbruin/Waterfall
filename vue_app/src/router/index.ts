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
    path: '/forgot-password',
    name: 'ForgotPassword',
    component: () => import('../views/ForgotPasswordView.vue'),
    meta: { requiresAuth: false },
  },
  {
    path: '/reset-password',
    name: 'ResetPassword',
    component: () => import('../views/ResetPasswordView.vue'),
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
    path: '/one-pager',
    name: 'One Pager',
    component: () => import('../views/OnePagerView.vue'),
  },
  {
    path: '/portfolio-snapshot',
    name: 'Portfolio Snapshot',
    component: () => import('../views/PortfolioSnapshotView.vue'),
  },
  {
    path: '/review-tracking',
    name: 'Review Tracking',
    component: () => import('../views/ReviewTrackingView.vue'),
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
    path: '/portfolio-analysis',
    name: 'Portfolio Analysis',
    component: () => import('../views/PortfolioAnalysisView.vue'),
  },
  {
    path: '/psckoc',
    name: 'PSCKOC',
    component: () => import('../views/PsckocView.vue'),
  },
  {
    path: '/surveillance',
    name: 'Surveillance',
    component: () => import('../views/SurveillanceView.vue'),
  },
  {
    path: '/data-explorer',
    name: 'Data Explorer',
    component: () => import('../views/DataExplorerView.vue'),
  },
  {
    path: '/pipeline',
    name: 'Pipeline',
    component: () => import('../views/PipelineView.vue'),
  },
  {
    path: '/lease-review',
    name: 'Lease Review',
    component: () => import('../views/LeaseReviewView.vue'),
  },
  {
    path: '/lease-risk-analysis',
    name: 'Lease Risk Analysis',
    component: () => import('../views/LeaseRiskAnalysisView.vue'),
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
router.beforeEach(async (to) => {
  const auth = useAuthStore()
  if (to.meta.requiresAuth !== false && !auth.isAuthenticated) {
    return { name: 'Login', query: { redirect: to.fullPath } }
  }
  // Restore user object from token after page refresh
  if (auth.isAuthenticated && !auth.user) {
    await auth.fetchMe()
  }
})

export default router
