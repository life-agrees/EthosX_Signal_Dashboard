// In your frontend JavaScript (e.g., config.js or api.js)

// For local development, use localhost:8000
// For production on Render, use your backend service URL
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'https://ethosx-signal-dashboard.onrender.com'  // ← Your actual backend URL
  : 'http://localhost:8000';

const WS_BASE_URL = process.env.NODE_ENV === 'production'
  ? 'wss://ethosx-signal-dashboard.onrender.com'   // ← Your actual backend URL (WebSocket)
  : 'ws://localhost:8000';

// Use these in your API calls
export const apiCall = async (endpoint, options = {}) => {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });
  return response;
};

// WebSocket connection
export const createWebSocketConnection = () => {
  return new WebSocket(`${WS_BASE_URL}/ws`);
};

// Example usage:
// apiCall('/subscribe', { method: 'POST', body: JSON.stringify(data) })
// apiCall('/predict', { method: 'POST', body: JSON.stringify(data) })