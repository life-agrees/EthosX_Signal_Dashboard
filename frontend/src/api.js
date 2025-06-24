// src/api.js - FIXED VERSION
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'https://ethosx-signal-dashboard.onrender.com'
  : 'http://localhost:8000';

const WS_BASE_URL = process.env.NODE_ENV === 'production'
  ? 'wss://ethosx-signal-dashboard.onrender.com'
  : 'ws://localhost:8000';

// Enhanced API call with better error handling
export const apiCall = async (endpoint, options = {}) => {
  const url = `${API_BASE_URL}${endpoint}`;
  
  console.log(`🔄 API Call: ${url}`);
  console.log(`🌍 Environment: ${process.env.NODE_ENV || 'development'}`);
  
  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        ...options.headers,
      },
    });
    
    console.log(`✅ Response [${endpoint}]: ${response.status} ${response.statusText}`);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error(`❌ API Error [${endpoint}]:`, errorText);
      throw new Error(`API Error: ${response.status} - ${errorText}`);
    }
    
    return response;
  } catch (error) {
    console.error(`💥 API call failed for ${endpoint}:`, error);
    throw error;
  }
};

// Specific API functions that match your FastAPI endpoints
export const getMarketData = async (token) => {
  const response = await apiCall(`/market/${token}`);
  return response.json();
};

export const getTechnicalData = async (token) => {
  const response = await apiCall(`/technical/${token}`);
  return response.json();
};

export const getSupportedTokens = async () => {
  const response = await apiCall('/tokens');
  return response.json();
};

export const getPrediction = async (token) => {
  const response = await apiCall(`/predictions/${token}`);
  return response.json();
};

export const getSubscriberCount = async () => {
  const response = await apiCall('/subscribers');
  return response.json();
};

export const getHealthStatus = async () => {
  const response = await apiCall('/health');
  return response.json();
};

// WebSocket connection
export const createWebSocketConnection = () => {
  const wsUrl = `${WS_BASE_URL}/ws`;
  console.log(`🔌 WebSocket connecting to: ${wsUrl}`);
  return new WebSocket(wsUrl);
};

// Export the base URLs
export { API_BASE_URL, WS_BASE_URL };