import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, BarChart, Bar } from 'recharts';
import { TrendingUp, TrendingDown, Minus, AlertCircle, Users, Zap, Activity, DollarSign, BarChart3, Bell, BellOff, Wifi, WifiOff, RefreshCw } from 'lucide-react';
import { debounce } from 'lodash';
import Header from "./components/Header"
import MarketData from './components/MarketData';
import Prediction from './components/Prediction';
import Charts from './components/Charts';
import Loading from './components/Loading';
import Controls from './components/Controls';
import PredictionHistory from './components/PredictionHistory';
import Alerts from './components/Alerts';
import Cards from './components/Cards';

// Constants
const WEBSOCKET_RECONNECT_DELAY = 5000;
const HEALTH_CHECK_INTERVAL = 30000;
const MAX_ALERTS = 5;
const MAX_PREDICTION_HISTORY = 10;
const PING_INTERVAL = 30000;
const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

// Error Boundary Component
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('Dashboard Error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-gray-900 text-white">
          <div className="text-center">
            <AlertCircle className="mx-auto mb-4 h-12 w-12 text-red-400" />
            <h2 className="text-xl font-semibold mb-2">Something went wrong</h2>
            <p className="text-gray-400 mb-4">The dashboard encountered an error</p>
            <button 
              onClick={() => window.location.reload()}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
            >
              Reload Dashboard
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

const EthosXDashboard = () => {
  // Core state
  const [selectedToken, setSelectedToken] = useState('BTC');
  const [email, setEmail] = useState('');
  const [isSubscribed, setIsSubscribed] = useState(false);
  const [subscriberCount, setSubscriberCount] = useState(0);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [theme, setTheme] = useState('dark');
  const [supportedTokens, setSupportedTokens] = useState(['BTC', 'SOL', 'DOGE', 'FART']);
  const [isGeneratingPrediction, setIsGeneratingPrediction] = useState(false);

  // Environment detection
  const isProduction = import.meta.env.PROD || window.location.hostname.includes('fly.dev');

  // Debug logging
  console.log('=== ENVIRONMENT DEBUG ===');
  console.log('import.meta.env.MODE:', import.meta.env.MODE);
  console.log('import.meta.env.PROD:', import.meta.env.PROD);
  console.log('import.meta.env.DEV:', import.meta.env.DEV);
  console.log('Raw VITE_API_BASE_URL:', import.meta.env.VITE_API_BASE_URL);
  console.log('Raw VITE_WEBSOCKET_URL:', import.meta.env.VITE_WEBSOCKET_URL);
  
  // API Configuration
  const API_BASE_URL = isProduction 
    ? 'https://ethosx-signals.fly.dev'
    : (import.meta?.env?.VITE_API_BASE_URL || "http://localhost:8000");
  const WS_URL = isProduction
    ? 'wss://ethosx-signals.fly.dev/ws'
    : (import.meta?.env?.VITE_WEBSOCKET_URL || "ws://localhost:8000/ws");

  console.log('Final API_BASE_URL:', API_BASE_URL);
  console.log('Final WS_URL:', WS_URL);
  console.log('=== END DEBUG ===');
  
  // WebSocket ref to avoid memory leaks
  const wsRef = useRef(null);
  const pingIntervalRef = useRef(null);
  const abortControllerRef = useRef(null);
  
  // Real-time data state
  const [marketData, setMarketData] = useState({});
  const [technicalData, setTechnicalData] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [predictionHistory, setPredictionHistory] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [isAutoRefresh, setIsAutoRefresh] = useState(true);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [apiHealth, setApiHealth] = useState(null);
  const [priceHistory, setPriceHistory] = useState([]);
  
  // Rate limiting state
  const [apiCallCount, setApiCallCount] = useState(0);
  const [lastApiCallTime, setLastApiCallTime] = useState(0);
  
  // Memoized theme classes
  const themeClasses = useMemo(() => 
    theme === 'dark' 
      ? 'bg-gray-900 text-white' 
      : 'bg-gray-50 text-gray-900',
    [theme]
  );
  
  const cardClasses = useMemo(() =>
    theme === 'dark'
      ? 'bg-gray-800 border-gray-700'
      : 'bg-white border-gray-200',
    [theme]
  );

  // Enhanced rate limiting with different strategies
  const checkRateLimit = useCallback(() => {
    const now = Date.now();
    const timeSinceLastCall = now - lastApiCallTime;
  
    // Reset counter every minute
    if (timeSinceLastCall > 60000) {
      setApiCallCount(0);
    }
  
    // Check if we're exceeding the rate limit (30 action calls per minute)
    if (apiCallCount >= 30) {
      console.warn('⚠️ Rate limit exceeded for action endpoints');
      return false;
    }
  
    // Minimum 2 seconds between action calls
    if (timeSinceLastCall < 2000) {
      console.warn('⚠️ Too many requests - please wait');
      return false;
    }
    
    setApiCallCount(prev => prev + 1);
    setLastApiCallTime(now);
    return true;
  }, [apiCallCount, lastApiCallTime]);

  // Enhanced error handling with user-friendly messages
  const getErrorMessage = (error, endpoint) => {
    if (error.message.includes('Rate limit exceeded')) {
      return 'Too many requests. Please wait a moment before trying again.';
    }
    
    if (error.message.includes('API Error: 429')) {
      return 'Server is busy. Please try again in a few seconds.';
    }
    
    if (error.message.includes('API Error: 500')) {
      return 'Server error. Our team has been notified.';
    }
    
    if (error.message.includes('Failed to fetch')) {
      return 'Network error. Please check your connection.';
    }
    
    if (endpoint.includes('/predict')) {
      return 'Prediction service is temporarily unavailable.';
    }
    
    if (endpoint.includes('/subscribe')) {
      return 'Subscription service is temporarily unavailable.';
    }
    
    return 'Something went wrong. Please try again.';
  };


  const apiCall = useCallback(async (endpoint, options = {}) => {
    // Define endpoint categories for different rate limiting strategies
    const endpointCategories = {
      // No rate limiting for essential data endpoints
      essential: ['/health', '/tokens', '/market/', '/technical/', '/subscribers'],
      // Light rate limiting for user queries
      user: ['/predictions/'],
      // Strict rate limiting for user actions
      actions: ['/predict', '/subscribe']
  };

  // Determine endpoint category
  const getEndpointCategory = (endpoint) => {
    if (endpointCategories.essential.some(prefix => endpoint === prefix || endpoint.startsWith(prefix))) {
      return 'essential';
    }
    if (endpointCategories.user.some(prefix => endpoint === prefix || endpoint.startsWith(prefix))) {
      return 'user';
    }
    if (endpointCategories.actions.some(prefix => endpoint === prefix || endpoint.startsWith(prefix))) {
      return 'actions';
    }
    return 'unknown';
  };

  const category = getEndpointCategory(endpoint);
  
  // Apply rate limiting based on category
  if (category === 'actions' && !checkRateLimit()) {
    throw new Error('Rate limit exceeded. Please wait before making more requests.');
  }
  
  // Optional: Add lighter rate limiting for user endpoints
  if (category === 'user') {
    const now = Date.now();
    const timeSinceLastCall = now - lastApiCallTime;
    
    // Minimum 500ms between user query calls
    if (timeSinceLastCall < 500) {
      await new Promise(resolve => setTimeout(resolve, 500 - timeSinceLastCall));
    }
  }

  // Create abort controller for this request
  const controller = new AbortController();
  abortControllerRef.current = controller;

  // Add timeout for requests
  const timeoutId = setTimeout(() => {
    controller.abort();
  }, 30000); // 30 second timeout

  try {
    console.log(`📡 API Call: ${endpoint} (Category: ${category})`);
    
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      signal: controller.signal,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });
    
    clearTimeout(timeoutId);
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API Error: ${response.status} ${response.statusText} - ${errorText}`);
    }
    
    const data = await response.json();
    console.log(`✅ API Success: ${endpoint}`);
    return data;
    
  } catch (error) {
    clearTimeout(timeoutId);
    
    if (error.name === 'AbortError') {
      console.log(`⏹️ Request aborted: ${endpoint}`);
      return null;
    }
    
    console.error(`❌ API call failed for ${endpoint}:`, error);
    
    // Add retry logic for essential endpoints
    if (category === 'essential' && !options._isRetry) {
      console.log(`🔄 Retrying essential endpoint: ${endpoint}`);
      await new Promise(resolve => setTimeout(resolve, 1000));
      return apiCall(endpoint, { ...options, _isRetry: true });
    }
    
    throw error;
  }
}, [API_BASE_URL, checkRateLimit, lastApiCallTime]);

  // Health check with improved error handling
  const checkApiHealth = useCallback(async () => {
    try {
      const health = await apiCall('/health');
      if (!health) return false; // Request was aborted
      
      setApiHealth(health);
      console.log('API Health:', health);
      
      if (health.data_source && health.data_source.includes('data_fetch.py')) {
        console.log('✅ API server is aligned with data_fetch.py');
      }
      
      setError(null);
      return true;
    } catch (error) {
      console.error('API health check failed:', error);
      setError('API server unavailable');
      setApiHealth(null);
      return false;
    }
  }, [apiCall]);

  // Enhanced email validation
  const validateEmail = useCallback((email) => {
    if (!email || typeof email !== 'string') {
      return false;
    }
    
    return EMAIL_REGEX.test(email.trim());
  }, []);

  // Debounced fetch functions
  const fetchSupportedTokens = useCallback(async () => {
    try {
      const data = await apiCall('/tokens');
      if (!data) return; // Request was aborted
      
      setSupportedTokens(data.tokens);
      console.log('Supported tokens:', data.tokens);
      
      if (data.tokens.length > 0 && !data.tokens.includes(selectedToken)) {
        setSelectedToken(data.tokens[0]);
      }
    } catch (error) {
      console.error('Failed to fetch supported tokens:', error);
      setError('Failed to load supported tokens');
    }
  }, [apiCall, selectedToken]);
  
  // Usage example with better error handling
  const fetchMarketData = useCallback(async () => {
    try {
      // Batch requests with proper error handling
      const marketPromises = supportedTokens.map(async (token) => {
        try {
          const data = await apiCall(`/market/${token}`);
          return { token, data, success: true };
        } catch (error) {
          console.warn(`Failed to fetch market data for ${token}:`, error);
          return { token, data: null, success: false, error: getErrorMessage(error, `/market/${token}`) };
        }
      });

      const results = await Promise.allSettled(marketPromises);
      const newMarketData = {};
      const errors = [];
      
      results.forEach((result) => {
        if (result.status === 'fulfilled') {
          const { token, data, success, error } = result.value;
          if (success && data) {
            newMarketData[token] = {
              price: data.price,
              change: data.change_24h,
              volume: data.volume_24h,
              oi: data.open_interest,
              funding_rate: data.funding_rate,
              timestamp: data.timestamp
            };
          } else if (error) {
            errors.push(`${token}: ${error}`);
          }
        }
      });
      // Update state with successful data
      if (Object.keys(newMarketData).length > 0) {
        setMarketData(prev => ({ ...prev, ...newMarketData }));
        setConnectionStatus('connected');
        setError(null);
      }
      // Show errors if any
      if (errors.length > 0) {
        console.warn('Market data fetch errors:', errors);
        // Only show error if no data was fetched at all
        if (Object.keys(newMarketData).length === 0) {
          setError('Failed to fetch market data');
        }
      }

    } catch (error) {
      console.error('Failed to fetch market data:', error);
      setConnectionStatus('disconnected');
      setError(getErrorMessage(error, '/market/'));
    }
  }, [supportedTokens, apiCall]);

  // Fetch technical indicators using your API
  const fetchTechnicalData = useCallback(async () => {
    try {
      const technicalPromises = supportedTokens.map(async (token) => {
        try {
          const data = await apiCall(`/technical/${token}`);
          return { token, data };
        } catch (error) {
          console.warn(`Failed to fetch technical data for ${token}:`, error);
          return { token, data: null };
        }
      });
      
      const results = await Promise.all(technicalPromises);
      const newTechnicalData = {};
      
      results.forEach(({ token, data }) => {
        if (data) {
          newTechnicalData[token] = {
            rsi: data.rsi_14,
            macd: data.macd_diff,
            sentiment: data.sentiment,
            funding: data.funding_rate,
            timestamp: data.timestamp
          };
        }
      });
      
      setTechnicalData(prev => ({ ...prev, ...newTechnicalData }));
    } catch (error) {
      console.error('Failed to fetch technical data:', error);
    }
  }, [supportedTokens]);



  // Debounced versions
  const debouncedFetchMarketData = useMemo(
    () => debounce(fetchMarketData, 1000),
    [fetchMarketData]
  );

  const debouncedFetchTechnicalData = useMemo(
    () => debounce(fetchTechnicalData, 1000),
    [fetchTechnicalData]
  );

  const fetchSubscriberCount = useCallback(async () => {
    try {
      const data = await apiCall('/subscribers');
      if (!data) return; // Request was aborted
      setSubscriberCount(data.count);
    } catch (error) {
      console.error('Failed to fetch subscriber count:', error);
    }
  }, [apiCall]);

  const fetchLatestPrediction = useCallback(async () => {
    try {
      const data = await apiCall(`/predictions/${selectedToken}`);
      if (!data) return; // Request was aborted
      
      const signalColorMap = {
        'Long CALL': 'text-green-400',
        'Long PUT': 'text-red-400'
      };
      
      const predictionData = {
        ...data,
        signalColor: signalColorMap[data.signal] || 'text-gray-400'
      };
      
      setPrediction(predictionData);
    } catch (error) {
      console.warn(`No prediction available for ${selectedToken}:`, error);
    }
  }, [selectedToken, apiCall]);

  // Enhanced prediction function with better error handling
  const makePrediction = useCallback(async () => {
    if (isGeneratingPrediction) return;
    
    try {
      setIsGeneratingPrediction(true);
      setLoading(true);
      
      const data = await apiCall('/predict', {
        method: 'POST',
        body: JSON.stringify({ token: selectedToken })
      });
      
      if (!data) return; // Request was aborted
      
      const signalColorMap = {
        'Long CALL': 'text-green-400',
        'Long PUT': 'text-red-400'
      };
      
      const predictionData = {
        ...data,
        signalColor: signalColorMap[data.signal] || 'text-gray-400'
      };
      
      setPrediction(predictionData);
      setPredictionHistory(prev => [...prev.slice(-(MAX_PREDICTION_HISTORY - 1)), predictionData]);
      
      // Add alert if high confidence
      if (data.confidence > 0.8) {
        const newAlert = {
          id: Date.now(),
          message: `High confidence ${data.signal} signal for ${selectedToken}`,
          confidence: data.confidence,
          timestamp: new Date().toLocaleTimeString(),
          type: data.signal === 'Long CALL' ? 'success' : data.signal === 'Long PUT' ? 'danger' : 'warning'
        };
        setAlerts(prev => [newAlert, ...prev.slice(0, MAX_ALERTS - 1)]);
      }
      
      setAlerts(prev => [{
        id: Date.now(),
        message: `New ${data.signal} signal generated for ${selectedToken}`,
        type: 'info',
        timestamp: new Date().toLocaleTimeString()
      }, ...prev.slice(0, MAX_ALERTS - 1)]);
      
    } catch (error) {
      console.error('Prediction failed:', error);
      setAlerts(prev => [{
        id: Date.now(),
        message: 'Prediction failed - check model status',
        type: 'error',
        timestamp: new Date().toLocaleTimeString()
      }, ...prev.slice(0, MAX_ALERTS - 1)]);
    } finally {
      setLoading(false);
      setIsGeneratingPrediction(false);
    }
  }, [selectedToken, isGeneratingPrediction, apiCall]);

  // Enhanced subscription with better validation
  const handleSubscribe = useCallback(async () => {
    if (!validateEmail(email)) {
      setAlerts(prev => [{
        id: Date.now(),
        message: 'Please enter a valid email address',
        type: 'error',
        timestamp: new Date().toLocaleTimeString()
      }, ...prev.slice(0, MAX_ALERTS - 1)]);
      return;
    }

    try {
      const data = await apiCall('/subscribe', {
        method: 'POST',
        body: JSON.stringify({ email: email.trim() })
      });
      
      if (!data) return; // Request was aborted
      
      setIsSubscribed(true);
      setAlerts(prev => [{
        id: Date.now(),
        message: 'Successfully subscribed to alerts!',
        type: 'success',
        timestamp: new Date().toLocaleTimeString()
      }, ...prev.slice(0, MAX_ALERTS - 1)]);
      
      fetchSubscriberCount();
    } catch (error) {
      setAlerts(prev => [{
        id: Date.now(),
        message: 'Subscription failed. Please try again.',
        type: 'error',
        timestamp: new Date().toLocaleTimeString()
      }, ...prev.slice(0, MAX_ALERTS - 1)]);
    }
  }, [email, validateEmail, apiCall, fetchSubscriberCount]);

  // Enhanced WebSocket connection with proper cleanup
  const connectWebSocket = useCallback(() => {
    // Clean up existing connection
    if (wsRef.current) {
      if (wsRef.current.readyState !== WebSocket.CLOSED) {
        wsRef.current.close();
      }
      wsRef.current = null;
    }

    // Clear existing ping interval
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = null;
    }

    console.log(`🔌 WebSocket connecting to API server: ${WS_URL}`);

    try {
      const newWs = new WebSocket(WS_URL);

      newWs.onopen = () => {
        console.log("WebSocket connected to API server");
        setConnectionStatus("connected");
        
        // Send subscription message
        newWs.send(JSON.stringify({
          type: "subscribe",
          tokens: supportedTokens
        }));

        // Keep-alive ping
        pingIntervalRef.current = setInterval(() => {
          if (newWs.readyState === WebSocket.OPEN) {
            newWs.send(JSON.stringify({ type: "ping" }));
          }
        }, PING_INTERVAL);
      };

      newWs.onmessage = (evt) => {
        try {
          const msg = JSON.parse(evt.data);
          console.log('WebSocket message:', msg);

          switch (msg.type) {
            case 'market_data':
              setMarketData(prev => ({
                ...prev,
                [msg.token]: {
                  price: msg.data.price,
                  change: msg.data.change_24h,
                  volume: msg.data.volume_24h,
                  oi: msg.data.open_interest,
                  funding_rate: msg.data.funding_rate,
                  timestamp: msg.data.timestamp
                }
              }));
              
              setPriceHistory(prev => [
                ...prev.slice(-29),
                {
                  time: new Date().toLocaleTimeString(),
                  price: msg.data.price,
                  volume: msg.data.volume_24h
                }
              ]);
              break;

            case 'prediction':
              const signalColorMap = {
                'Long CALL': 'text-green-400',
                'Long PUT': 'text-red-400',
              };
              
              if (msg.token === selectedToken) {
                setPrediction({
                  ...msg.data,
                  signalColor: signalColorMap[msg.data.signal] || 'text-gray-400'
                });
              }
              break;

            case 'alert':
              setAlerts(prev => [{
                id: Date.now(),
                message: msg.message,
                type: msg.alert_type || 'info',
                timestamp: new Date().toLocaleTimeString()
              }, ...prev.slice(0, MAX_ALERTS - 1)]);
              break;

            case 'pong':
              // Handle pong response
              break;

            default:
              console.log('Unknown message type:', msg.type);
          }
        } catch (err) {
          console.error("WebSocket message parsing error:", err);
        }
      };

      newWs.onclose = (event) => {
        console.warn("WebSocket disconnected:", event.code, event.reason);
        setConnectionStatus("disconnected");
        
        // Clear ping interval
        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current);
          pingIntervalRef.current = null;
        }
        
        // Reconnect if auto-refresh is enabled and connection wasn't manually closed
        if (isAutoRefresh && event.code !== 1000) {
          setTimeout(() => {
            if (isAutoRefresh) {
              connectWebSocket();
            }
          }, WEBSOCKET_RECONNECT_DELAY);
        }
      };

      newWs.onerror = (error) => {
        console.error("WebSocket error:", error);
        setConnectionStatus("error");
      };

      wsRef.current = newWs;
    } catch (error) {
      console.error("Failed to create WebSocket:", error);
      setConnectionStatus("error");
    }
  }, [WS_URL, isAutoRefresh, supportedTokens, selectedToken]);

  // Initialize data on component mount
  useEffect(() => {
    const initializeData = async () => {
      setLoading(true);
      try {
        const isHealthy = await checkApiHealth();
        if (!isHealthy) {
          setError('API server is not responding');
          return;
        }

        await fetchSupportedTokens();
        await fetchSubscriberCount();
        
        if (isAutoRefresh) {
          connectWebSocket();
        }
      } catch (error) {
        console.error('Failed to initialize dashboard:', error);
        setError('Failed to initialize dashboard');
      } finally {
        setLoading(false);
      }
    };

    initializeData();
  }, []); // Empty dependency array - only run on mount

  // Fetch data when tokens are loaded
  useEffect(() => {
    if (supportedTokens.length > 0) {
      debouncedFetchMarketData();
      debouncedFetchTechnicalData();
    }
  }, [supportedTokens, debouncedFetchMarketData, debouncedFetchTechnicalData]);

  // Fetch prediction when selected token changes
  useEffect(() => {
    if (selectedToken) {
      fetchLatestPrediction();
    }
  }, [selectedToken, fetchLatestPrediction]);

  // Periodic health checks and data refresh
  useEffect(() => {
    if (!isAutoRefresh) return;
    
    const interval = setInterval(async () => {
      await checkApiHealth();
      
      if (connectionStatus === 'disconnected') {
        debouncedFetchMarketData();
        debouncedFetchTechnicalData();
      }
    }, HEALTH_CHECK_INTERVAL);
    
    return () => clearInterval(interval);
  }, [isAutoRefresh, connectionStatus, debouncedFetchMarketData, debouncedFetchTechnicalData, checkApiHealth]);

  // Initialize price history when market data is available
  useEffect(() => {
    if (marketData[selectedToken] && priceHistory.length === 0) {
      const initialHistory = Array.from({ length: 30 }, (_, i) => ({
        time: new Date(Date.now() - (29 - i) * 60000).toLocaleTimeString(),
        price: marketData[selectedToken].price * (1 + (Math.random() - 0.5) * 0.01),
        volume: marketData[selectedToken].volume * (1 + (Math.random() - 0.5) * 0.05)
      }));
      setPriceHistory(initialHistory);
    }
  }, [selectedToken, marketData, priceHistory.length]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      // Cleanup WebSocket
      if (wsRef.current) {
        wsRef.current.close();
      }
      
      // Cleanup ping interval
      if (pingIntervalRef.current) {
        clearInterval(pingIntervalRef.current);
      }
      
      // Cleanup pending API calls
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []); // Empty dependency array - only cleanup on unmount

  const currentData = marketData[selectedToken];
  const currentTech = technicalData[selectedToken];

  // Early return for loading state
  if (loading && Object.keys(marketData).length === 0) {
    return (
      <Loading 
        loading={loading}
        error={error}
        apiHealth={apiHealth}
        theme={theme}
      />
    );
  }
  
  return (
    <ErrorBoundary>
      <div className={`min-h-screen ${themeClasses} transition-colors duration-300`}>
        <Header
          connectionStatus={connectionStatus}
          apiHealth={apiHealth}
          isAutoRefresh={isAutoRefresh}
          setIsAutoRefresh={setIsAutoRefresh}
          connectWebSocket={connectWebSocket}
          ws={wsRef.current}
          theme={theme}
          setTheme={setTheme}
        />
        
        <div className="p-6 space-y-6">
          <Controls 
            selectedToken={selectedToken}
            setSelectedToken={setSelectedToken}
            supportedTokens={supportedTokens}
            marketData={marketData}
            makePrediction={makePrediction}
            loading={loading}
            isGeneratingPrediction={isGeneratingPrediction}
            apiHealth={apiHealth}
            isSubscribed={isSubscribed}
            email={email}
            setEmail={setEmail}
            handleSubscribe={handleSubscribe}
            subscriberCount={subscriberCount}
            theme={theme}
          />
          
          <MarketData 
            currentData={currentData}
            currentTech={currentTech}
            selectedToken={selectedToken}
            theme={theme}
          />
          
          <Charts 
            priceHistory={priceHistory}
            currentTech={currentTech}
            theme={theme}
          />
          
          <Prediction 
            prediction={prediction}
            theme={theme}
          />
          
          <Alerts 
            alerts={alerts}
            theme={theme}
          />

          <PredictionHistory 
            predictionHistory={predictionHistory}
            theme={theme}
          />
        </div>
      </div>
    </ErrorBoundary>
  );
};

export default EthosXDashboard;