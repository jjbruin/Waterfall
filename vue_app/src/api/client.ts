import axios from 'axios'

const api = axios.create({
  baseURL: '/',
  timeout: 300_000, // 5 min — PSCKOC computation can take a while
  headers: { 'Content-Type': 'application/json' },
})

// Request interceptor: attach JWT
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Response interceptor: handle 401 + server errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token')
      // Only redirect if not already on the login page (prevents infinite loop)
      if (window.location.pathname !== '/login') {
        window.location.href = '/login'
      }
    }
    // Log server errors to console for debugging
    if (error.response?.status >= 500) {
      console.error('[API Error]', error.response.status, error.config?.url, error.response.data)
    }
    return Promise.reject(error)
  },
)

export default api
