import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  
  // Define environment variables that should be exposed to the client
  define: {
    // These will be available as process.env.VITE_* in your React app
    'process.env.VITE_API_BASE_URL': JSON.stringify(process.env.VITE_API_BASE_URL || 'http://localhost:3000'),
    'process.env.VITE_WEBSOCKET_URL': JSON.stringify(process.env.VITE_WEBSOCKET_URL || 'ws://localhost:3000/ws'),
  },
  
  // Server configuration for development
  server: {
    port: 5173,
    host: true, // Needed for Docker/container environments
    proxy: {
      // Optional: Proxy API calls during development to avoid CORS issues
      '/api': {
        target: process.env.VITE_API_BASE_URL || 'http://localhost:3000',
        changeOrigin: true,
        secure: false,
      }
    }
  },
  
  // Build configuration
  build: {
    outDir: 'dist',
    sourcemap: false, // Set to true if you want source maps in production
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          // Add other large dependencies here to optimize loading
        }
      }
    }
  },
  
  // Preview configuration (for fly.io deployment)
  preview: {
    port: 4173,
    host: true,
  },
  
  // Environment variables configuration
  envPrefix: 'VITE_', // Only env vars starting with VITE_ will be exposed
})