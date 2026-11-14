import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev: /api/* is proxied to the Flask server (so no CORS setup is needed locally).
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: process.env.VITE_PROXY_TARGET || 'http://localhost:5000',
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ''),
      },
    },
  },
})
