// vite.config.js
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',           // accept connections on all interfaces
    port: 3000,
    proxy: {
      // everything under /api goes to your FastAPI
      '^/(market|technical|predict|tokens|subscribers|predictions|ws)': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        ws: true,               // proxy WebSockets too
      }
    }
  }
});
// This configuration sets up a Vite development server for a React application
// that proxies API requests to a FastAPI backend running on port 8000.
// The server listens on all interfaces