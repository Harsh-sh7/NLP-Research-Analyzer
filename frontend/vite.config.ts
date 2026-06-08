import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  // Load env variables so we can read VITE_API_URL in config itself
  const env = loadEnv(mode, process.cwd(), '')

  // Derive the backend origin from VITE_API_URL (strip the /api path)
  const apiUrl = env.VITE_API_URL ?? 'http://localhost:8000/api'
  const backendOrigin = apiUrl.replace(/\/api\/?$/, '')

  return {
    plugins: [react()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    // Dev server proxy — forwards /api requests to the backend during
    // local development, avoiding CORS issues without any extra config.
    server: {
      port: 5173,
      proxy: {
        '/api': {
          target: backendOrigin,
          changeOrigin: true,
          secure: false,
        },
      },
    },
    build: {
      // Generate source maps only for non-production builds
      sourcemap: mode !== 'production',
      // Raise the chunk warning threshold a little for ML-heavy deps
      chunkSizeWarningLimit: 1000,
    },
  }
})
