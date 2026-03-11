<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useAuthStore } from '../stores/auth'
import api from '../api/client'

const router = useRouter()
const route = useRoute()
const auth = useAuthStore()

const username = ref('')
const password = ref('')
const error = ref('')
const loading = ref(false)
const ssoEnabled = ref(false)
const ssoProvider = ref('')

onMounted(async () => {
  // Check if SSO is configured
  try {
    const res = await api.get('/auth/sso/config')
    ssoEnabled.value = res.data.enabled
    ssoProvider.value = res.data.provider || ''
  } catch {
    // SSO not available
  }

  // Handle SSO callback token from URL fragment
  const hash = window.location.hash
  if (hash.includes('token=')) {
    const token = hash.split('token=')[1]?.split('&')[0]
    if (token) {
      localStorage.setItem('token', token)
      auth.token = token
      await auth.fetchMe()
      const redirect = (route.query.redirect as string) || '/dashboard'
      router.push(redirect)
      return
    }
  }
  if (hash.includes('sso_error=')) {
    error.value = 'SSO authentication failed. Please try again or use username/password.'
  }
})

async function handleLogin() {
  error.value = ''
  loading.value = true
  try {
    await auth.login(username.value, password.value)
    const redirect = (route.query.redirect as string) || '/dashboard'
    router.push(redirect)
  } catch (e: any) {
    error.value = e.response?.data?.error || 'Login failed'
  } finally {
    loading.value = false
  }
}

function handleSsoLogin() {
  window.location.href = '/auth/sso/login'
}

const ssoLabel = {
  azure: 'Sign in with Microsoft',
  okta: 'Sign in with Okta',
}
</script>

<template>
  <div class="login-card">
    <h2>Waterfall XIRR</h2>
    <p class="subtitle">Sign in to continue</p>

    <!-- SSO Button -->
    <button v-if="ssoEnabled" class="btn-sso" @click="handleSsoLogin">
      {{ ssoLabel[ssoProvider as keyof typeof ssoLabel] || 'Sign in with SSO' }}
    </button>
    <div v-if="ssoEnabled" class="divider"><span>or</span></div>

    <form @submit.prevent="handleLogin">
      <div class="field">
        <label>Username</label>
        <input v-model="username" type="text" autocomplete="username" required />
      </div>
      <div class="field">
        <label>Password</label>
        <input v-model="password" type="password" autocomplete="current-password" required />
      </div>
      <p v-if="error" class="error">{{ error }}</p>
      <button type="submit" :disabled="loading" class="btn-login">
        {{ loading ? 'Signing in...' : 'Sign In' }}
      </button>
    </form>
  </div>
</template>

<style scoped>
.login-card {
  background: var(--color-surface);
  padding: 40px;
  border-radius: 12px;
  box-shadow: 0 4px 24px rgba(0, 0, 0, 0.1);
  width: 360px;
}

.login-card h2 {
  margin-bottom: 4px;
  color: var(--color-primary);
}

.subtitle {
  color: var(--color-text-secondary);
  margin-bottom: 24px;
  font-size: 14px;
}

.field {
  margin-bottom: 16px;
}

.field label {
  display: block;
  font-size: 13px;
  font-weight: 500;
  margin-bottom: 4px;
}

.field input {
  width: 100%;
  padding: 8px 12px;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  font-size: 14px;
}

.error {
  color: #dc3545;
  font-size: 13px;
  margin-bottom: 12px;
}

.btn-login {
  width: 100%;
  padding: 10px;
  background: var(--color-primary);
  color: white;
  border: none;
  border-radius: 6px;
  font-size: 14px;
  cursor: pointer;
}

.btn-login:hover { background: #163d5f; }
.btn-login:disabled { opacity: 0.6; cursor: not-allowed; }

.btn-sso {
  width: 100%;
  padding: 10px;
  background: #fff;
  color: #333;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  font-size: 14px;
  cursor: pointer;
  margin-bottom: 16px;
}
.btn-sso:hover { background: #f5f5f5; }

.divider {
  text-align: center;
  margin-bottom: 16px;
  position: relative;
}
.divider::before {
  content: '';
  position: absolute;
  left: 0;
  right: 0;
  top: 50%;
  border-top: 1px solid var(--color-border);
}
.divider span {
  position: relative;
  background: var(--color-surface);
  padding: 0 12px;
  font-size: 12px;
  color: var(--color-text-secondary);
}
</style>
