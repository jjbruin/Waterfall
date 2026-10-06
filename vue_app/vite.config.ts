import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { fileURLToPath, URL } from 'node:url'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
        // Disable buffering for SSE endpoints
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes, req) => {
            if (req.url?.includes('init-stream')) {
              proxyRes.headers['cache-control'] = 'no-cache'
              proxyRes.headers['x-accel-buffering'] = 'no'
            }
          })
        },
      },
      '/auth': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
    },
  },
  build: {
    chunkSizeWarningLimit: 650,
    rollupOptions: {
      // A second page: the Microsoft sign-in popup returns to it (see
      // src/msalRedirect.ts). Without it in the build, production 404s there
      // and the SharePoint picker can never get a token.
      input: {
        main: fileURLToPath(new URL('./index.html', import.meta.url)),
        msalRedirect: fileURLToPath(new URL('./msal-redirect.html', import.meta.url)),
      },
      output: {
        manualChunks: {
          echarts: ['echarts', 'vue-echarts'],
        },
      },
    },
  },
  resolve: {
    alias: {
      '@': '/src',
    },
  },
})
