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

// Force change password state
const newPassword = ref('')
const confirmPassword = ref('')
const changeLoading = ref(false)
const changeError = ref('')
const tempPassword = ref('')

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
    const result = await auth.login(username.value, password.value)
    if (result?.mustChangePassword) {
      // Store the current password so user doesn't have to re-type it
      tempPassword.value = password.value
      password.value = ''
      return
    }
    const redirect = (route.query.redirect as string) || '/dashboard'
    router.push(redirect)
  } catch (e: any) {
    error.value = e.response?.data?.error || 'Login failed'
  } finally {
    loading.value = false
  }
}

async function handleForceChange() {
  changeError.value = ''
  if (newPassword.value !== confirmPassword.value) {
    changeError.value = 'Passwords do not match'
    return
  }
  if (newPassword.value.length < 8) {
    changeError.value = 'Password must be at least 8 characters'
    return
  }
  if (newPassword.value === tempPassword.value) {
    changeError.value = 'New password must be different from current password'
    return
  }
  changeLoading.value = true
  try {
    await auth.forceChangePassword(tempPassword.value, newPassword.value)
    const redirect = (route.query.redirect as string) || '/dashboard'
    router.push(redirect)
  } catch (e: any) {
    changeError.value = e.response?.data?.error || 'Failed to change password'
  } finally {
    changeLoading.value = false
  }
}

function cancelForceChange() {
  auth.mustChangePassword = false
  auth.pendingUsername = ''
  tempPassword.value = ''
  newPassword.value = ''
  confirmPassword.value = ''
  changeError.value = ''
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

    <!-- Force Change Password Form -->
    <template v-if="auth.mustChangePassword">
      <p class="subtitle">You must change your password before continuing</p>
      <p class="info-text">Welcome, <strong>{{ auth.pendingUsername }}</strong>. Please set a new password.</p>

      <form @submit.prevent="handleForceChange">
        <div class="field">
          <label>New Password</label>
          <input v-model="newPassword" type="password" autocomplete="new-password"
                 placeholder="At least 8 characters" required />
        </div>
        <div class="field">
          <label>Confirm New Password</label>
          <input v-model="confirmPassword" type="password" autocomplete="new-password" required />
        </div>
        <p v-if="changeError" class="error">{{ changeError }}</p>
        <button type="submit" :disabled="changeLoading" class="btn-login">
          {{ changeLoading ? 'Changing...' : 'Set New Password' }}
        </button>
        <button type="button" class="btn-back" @click="cancelForceChange">Back to Login</button>
      </form>
    </template>

    <!-- Normal Login Form -->
    <template v-else>
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
      <div class="forgot-link">
        <router-link to="/forgot-password">Forgot password?</router-link>
      </div>
    </template>
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

.info-text {
  font-size: 13px;
  margin-bottom: 16px;
  padding: 10px 12px;
  background: #f0f7ff;
  border-radius: 6px;
  border-left: 3px solid var(--color-primary);
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

.btn-back {
  width: 100%;
  padding: 10px;
  background: transparent;
  color: var(--color-text-secondary);
  border: 1px solid var(--color-border);
  border-radius: 6px;
  font-size: 13px;
  cursor: pointer;
  margin-top: 8px;
}
.btn-back:hover { background: #f5f5f5; }

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

.forgot-link {
  text-align: center;
  margin-top: 16px;
  font-size: 13px;
}
.forgot-link a {
  color: var(--color-primary);
  text-decoration: none;
}
.forgot-link a:hover {
  text-decoration: underline;
}
</style>
