import React, { useState, useEffect, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, BarChart, Bar } from 'recharts';
import { TrendingUp, TrendingDown, Minus, AlertCircle, Users, Zap, Activity, DollarSign, BarChart3, Bell, BellOff, Wifi, WifiOff, RefreshCw } from 'lucide-react';

const EthosXDashboard = () => {
  // Core state
  const [selectedToken, setSelectedToken] = useState('BTC');
  const [email, setEmail] = useState('');
  const [isSubscribed, setIsSubscribed] = useState(false);
  const [subscriberCount, setSubscriberCount] = useState(0);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [theme, setTheme] = useState('dark');
  const [supportedTokens, setSupportedTokens] = useState(['BTC', 'ETH', 'SOL', 'DOGE']);
  
  // API Configuration
  const API_BASE_URL = ''; // Adjust this to your server URL
  const [ws, setWs] = useState(null);
  
  // Real-time data state (now from API)
  const [marketData, setMarketData] = useState({});
  const [technicalData, setTechnicalData] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [predictionHistory, setPredictionHistory] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [isAutoRefresh, setIsAutoRefresh] = useState(true);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Price history for charts
  const [priceHistory, setPriceHistory] = useState([]);
  
  // API Helper functions
  const apiCall = async (endpoint, options = {}) => {
    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        ...options,
      });
      
      if (!response.ok) {
        throw new Error(`API Error: ${response.status} ${response.statusText}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error(`API call failed for ${endpoint}:`, error);
      throw error;
    }
  };

  // Initialize supported tokens
  const fetchSupportedTokens = useCallback(async () => {
    try {
      const data = await apiCall('/tokens');
      setSupportedTokens(data.tokens);
      if (data.tokens.length > 0 && !data.tokens.includes(selectedToken)) {
        setSelectedToken(data.tokens[0]);
      }
    } catch (error) {
      console.error('Failed to fetch supported tokens:', error);
      setError('Failed to load supported tokens');
    }
  }, [selectedToken]);

  // Fetch market data for all tokens
  const fetchMarketData = useCallback(async () => {
    try {
      const marketPromises = supportedTokens.map(async (token) => {
        try {
          const data = await apiCall(`/market/${token}`);
          return { token, data };
        } catch (error) {
          console.warn(`Failed to fetch market data for ${token}:`, error);
          return { token, data: null };
        }
      });
      
      const results = await Promise.all(marketPromises);
      const newMarketData = {};
      
      results.forEach(({ token, data }) => {
        if (data) {
          newMarketData[token] = {
            price: data.price,
            change: data.change_24h,
            volume: data.volume_24h,
            oi: data.open_interest,
            timestamp: data.timestamp
          };
        }
      });
      
      setMarketData(prev => ({ ...prev, ...newMarketData }));
      setConnectionStatus('connected');
      setError(null);
    } catch (error) {
      console.error('Failed to fetch market data:', error);
      setConnectionStatus('disconnected');
    }
  }, [supportedTokens]);

  // Fetch technical indicators for all tokens
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

  // Fetch subscriber count
  const fetchSubscriberCount = useCallback(async () => {
    try {
      const data = await apiCall('/subscribers');
      setSubscriberCount(data.count);
    } catch (error) {
      console.error('Failed to fetch subscriber count:', error);
    }
  }, []);

  // Get latest prediction for selected token
  const fetchLatestPrediction = useCallback(async () => {
    try {
      const data = await apiCall(`/predictions/${selectedToken}`);
      
      const signalColorMap = {
        'BUY': 'text-green-400',
        'SELL': 'text-red-400',
        'HOLD': 'text-yellow-400'
      };
      
      const predictionData = {
        ...data,
        signalColor: signalColorMap[data.signal] || 'text-gray-400'
      };
      
      setPrediction(predictionData);
    } catch (error) {
      console.warn(`No prediction available for ${selectedToken}:`, error);
    }
  }, [selectedToken]);

  // Make new prediction
  const makePrediction = useCallback(async () => {
    try {
      setLoading(true);
      const data = await apiCall('/predict', {
        method: 'POST',
        body: JSON.stringify({
          token: selectedToken
        })
      });
      
      const signalColorMap = {
        'BUY': 'text-green-400',
        'SELL': 'text-red-400',
        'HOLD': 'text-yellow-400'
      };
      
      const predictionData = {
        ...data,
        signalColor: signalColorMap[data.signal] || 'text-gray-400'
      };
      
      setPrediction(predictionData);
      setPredictionHistory(prev => [...prev.slice(-9), predictionData]);
      
      // Add alert if high confidence
      if (data.confidence > 0.8) {
        const newAlert = {
          id: Date.now(),
          message: `High confidence ${data.signal} signal for ${selectedToken}`,
          confidence: data.confidence,
          timestamp: new Date().toLocaleTimeString(),
          type: data.signal === 'BUY' ? 'success' : data.signal === 'SELL' ? 'danger' : 'warning'
        };
        setAlerts(prev => [newAlert, ...prev.slice(0, 4)]);
      }
      
      setAlerts(prev => [{
        id: Date.now(),
        message: `New ${data.signal} signal generated for ${selectedToken}`,
        type: 'info',
        timestamp: new Date().toLocaleTimeString()
      }, ...prev.slice(0, 4)]);
      
    } catch (error) {
      console.error('Prediction failed:', error);
      setAlerts(prev => [{
        id: Date.now(),
        message: 'Prediction failed - check model status',
        type: 'error',
        timestamp: new Date().toLocaleTimeString()
      }, ...prev.slice(0, 4)]);
    } finally {
      setLoading(false);
    }
  }, [selectedToken]);

  // Subscribe to email alerts
  const handleSubscribe = async () => {
    if (!email.includes('@')) {
      setAlerts(prev => [{
        id: Date.now(),
        message: 'Please enter a valid email address',
        type: 'error',
        timestamp: new Date().toLocaleTimeString()
      }, ...prev.slice(0, 4)]);
      return;
    }

    try {
      await apiCall('/subscribe', {
        method: 'POST',
        body: JSON.stringify({ email })
      });
      
      setIsSubscribed(true);
      setAlerts(prev => [{
        id: Date.now(),
        message: 'Successfully subscribed to alerts!',
        type: 'success',
        timestamp: new Date().toLocaleTimeString()
      }, ...prev.slice(0, 4)]);
      
      // Refresh subscriber count
      fetchSubscriberCount();
    } catch (error) {
      setAlerts(prev => [{
        id: Date.now(),
        message: 'Subscription failed. Please try again.',
        type: 'error',
        timestamp: new Date().toLocaleTimeString()
      }, ...prev.slice(0, 4)]);
    }
  };

  // WebSocket connection for real-time updates
  const connectWebSocket = useCallback(() => {
    if (ws) {
      ws.close();
    }

    try {
      const wsUrl = API_BASE_URL.replace('http://', 'ws://').replace('https://', 'wss://') + '/ws';
      const newWs = new WebSocket(wsUrl);
      let pingInterval;
      
      newWs.onopen = () => {
        console.log('WebSocket connected');
        setConnectionStatus('connected');
        // send initial ping
        newWs.send(JSON.stringify({ type: 'ping' }));
        // then ping every 30s to keep the socket alive
        pingInterval = setInterval(() => {
          newWs.send(JSON.stringify({ type: 'ping' }));
        }, 30000);
      };
      
      newWs.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          
          if (message.type === 'market_data') {
            setMarketData(prev => ({
              ...prev,
              [message.token]: {
                price: message.data.price,
                change: message.data.change_24h,
                volume: message.data.volume_24h,
                oi: message.data.open_interest,
                timestamp: message.data.timestamp
              }
            }));
            
            setTechnicalData(prev => ({
              ...prev,
              [message.token]: {
                rsi: message.data.rsi_14,
                macd: message.data.macd_diff,
                sentiment: message.data.sentiment,
                funding: message.data.funding_rate,
                timestamp: message.data.timestamp
              }
            }));
            
            // Update price history for selected token
            if (message.token === selectedToken) {
              setPriceHistory(prev => {
                const newPoint = {
                  time: new Date().toLocaleTimeString(),
                  price: message.data.price,
                  volume: message.data.volume_24h
                };
                return [...prev.slice(-29), newPoint];
              });
            }
          }
          
          if (message.type === 'prediction') {
            const signalColorMap = {
              'BUY': 'text-green-400',
              'SELL': 'text-red-400',
              'HOLD': 'text-yellow-400'
            };
            
            const predictionData = {
              ...message.data,
              signalColor: signalColorMap[message.data.signal] || 'text-gray-400'
            };
            
            if (message.token === selectedToken) {
              setPrediction(predictionData);
              setPredictionHistory(prev => [...prev.slice(-9), predictionData]);
            }
          }
        } catch (error) {
          console.error('WebSocket message parsing error:', error);
        }
      };
      
      newWs.onclose = () => {
        console.log('WebSocket disconnected');
        clearInterval(pingInterval);
        setConnectionStatus('disconnected');
        // Attempt to reconnect after 5 seconds
        setTimeout(() => {
          if (isAutoRefresh) {
            connectWebSocket();
          }
        }, 5000);
      };
      
      newWs.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionStatus('disconnected');
        clearInterval(pingInterval);
      };
      
      setWs(newWs);
    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
      setConnectionStatus('disconnected');
    }
  }, [selectedToken, isAutoRefresh, ws]);

  // Initialize data on component mount
  useEffect(() => {
    const initializeData = async () => {
      setLoading(true);
      try {
        await fetchSupportedTokens();
        await fetchSubscriberCount();
        // Connect WebSocket for real-time updates
        if (isAutoRefresh) {
          connectWebSocket();
        }
      } catch (error) {
        setError('Failed to initialize dashboard');
      } finally {
        setLoading(false);
      }
    };

    initializeData();
  }, []);

  // Fetch data when tokens are loaded
  useEffect(() => {
    if (supportedTokens.length > 0) {
      fetchMarketData();
      fetchTechnicalData();
    }
  }, [supportedTokens, fetchMarketData, fetchTechnicalData]);

  // Fetch prediction when selected token changes
  useEffect(() => {
    if (selectedToken) {
      fetchLatestPrediction();
    }
  }, [selectedToken, fetchLatestPrediction]);

  // Periodic data refresh (fallback when WebSocket is not available)
  useEffect(() => {
    if (!isAutoRefresh) return;
    
    const interval = setInterval(() => {
      if (connectionStatus === 'disconnected') {
        fetchMarketData();
        fetchTechnicalData();
      }
    }, 15000); // 15 seconds
    
    return () => clearInterval(interval);
  }, [isAutoRefresh, connectionStatus, fetchMarketData, fetchTechnicalData]);

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

  // Cleanup WebSocket on unmount
  useEffect(() => {
    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, [ws]);

  const currentData = marketData[selectedToken];
  const currentTech = technicalData[selectedToken];
  
  const themeClasses = theme === 'dark' 
    ? 'bg-gray-900 text-white' 
    : 'bg-gray-50 text-gray-900';
  
  const cardClasses = theme === 'dark'
    ? 'bg-gray-800 border-gray-700'
    : 'bg-white border-gray-200';

  // Loading state
  if (loading && Object.keys(marketData).length === 0) {
    return (
      <div className={`min-h-screen ${themeClasses} flex items-center justify-center`}>
        <div className="text-center">
          <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-4" />
          <p>Loading dashboard...</p>
          {error && <p className="text-red-400 mt-2">{error}</p>}
        </div>
      </div>
    );
  }
  
  return (
    <div className={`min-h-screen ${themeClasses} transition-colors duration-300`}>
      {/* Header */}
      <div className={`${cardClasses} border-b px-6 py-4`}>
        <div className="flex justify-between items-center">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <BarChart3 className="w-8 h-8 text-blue-500" />
              <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent">
                EthosX OPerps Dashboard
              </h1>
            </div>
            <div className="flex items-center space-x-2">
              {connectionStatus === 'connected' ? (
                <><Wifi className="w-4 h-4 text-green-400" /><span className="text-sm text-green-400">Live</span></>
              ) : (
                <><WifiOff className="w-4 h-4 text-red-400" /><span className="text-sm text-red-400">Disconnected</span></>
              )}
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <button
              onClick={() => {
                setIsAutoRefresh(!isAutoRefresh);
                if (!isAutoRefresh) {
                  connectWebSocket();
                } else if (ws) {
                  ws.close();
                }
              }}
              className={`p-2 rounded-lg ${isAutoRefresh ? 'bg-green-600' : 'bg-gray-600'} hover:opacity-80 transition-opacity`}
            >
              {isAutoRefresh ? <Bell className="w-4 h-4" /> : <BellOff className="w-4 h-4" />}
            </button>
            <button
              onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
              className="p-2 rounded-lg bg-gray-600 hover:bg-gray-500 transition-colors"
            >
              {theme === 'dark' ? '☀️' : '🌙'}
            </button>
          </div>
        </div>
      </div>
      
      <div className="p-6 space-y-6">
        {/* Controls */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Token Selection */}
          <div className={`${cardClasses} border rounded-xl p-6`}>
            <h3 className="text-lg font-semibold mb-4">Token Selection</h3>
            <div className="grid grid-cols-2 gap-2 mb-4">
              {supportedTokens.map(token => (
                <button
                  key={token}
                  onClick={() => setSelectedToken(token)}
                  className={`p-3 rounded-lg border-2 transition-all ${
                    selectedToken === token
                      ? 'border-blue-500 bg-blue-500/10'
                      : 'border-gray-600 hover:border-gray-500'
                  }`}
                >
                  <div className="text-sm font-medium">{token}</div>
                  <div className={`text-xs ${marketData[token]?.change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {marketData[token]?.change >= 0 ? '+' : ''}{marketData[token]?.change?.toFixed(2) || '0.00'}%
                  </div>
                </button>
              ))}
            </div>
            <button
              onClick={makePrediction}
              disabled={loading}
              className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 text-white font-semibold py-3 px-4 rounded-lg transition-all transform hover:scale-105 disabled:hover:scale-100"
            >
              <Zap className="w-4 h-4 inline mr-2" />
              {loading ? 'Generating...' : 'Generate Signal'}
            </button>
          </div>
          
          {/* Subscription */}
          <div className={`${cardClasses} border rounded-xl p-6`}>
            <h3 className="text-lg font-semibold mb-4 flex items-center">
              <Users className="w-5 h-5 mr-2" />
              Alert Subscription
            </h3>
            {!isSubscribed ? (
              <div className="space-y-3">
                <input
                  type="email"
                  placeholder="Enter your email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg focus:border-blue-500 focus:outline-none"
                />
                <button
                  onClick={handleSubscribe}
                  className="w-full bg-green-600 hover:bg-green-700 text-white font-semibold py-3 px-4 rounded-lg transition-colors"
                >
                  Subscribe
                </button>
              </div>
            ) : (
              <div className="text-center">
                <div className="text-green-400 mb-2">✓ Subscribed</div>
                <div className="text-sm text-gray-400">You'll receive high-confidence alerts</div>
              </div>
            )}
            <div className="text-sm text-gray-400 mt-4">
              {subscriberCount} active subscribers
            </div>
          </div>
        </div>
        
        {/* Market Data Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className={`${cardClasses} border rounded-xl p-6`}>
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-sm font-medium text-gray-400">Price</h4>
              <DollarSign className="w-4 h-4 text-gray-400" />
            </div>
            <div className="text-2xl font-bold">
              ${currentData?.price ? currentData.price.toLocaleString(undefined, {
                maximumFractionDigits: currentData.price < 1 ? 6 : 2
              }) : 'Loading...'}
            </div>
            <div className={`text-sm flex items-center ${currentData?.change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {currentData?.change >= 0 ? <TrendingUp className="w-4 h-4 mr-1" /> : <TrendingDown className="w-4 h-4 mr-1" />}
              {currentData?.change >= 0 ? '+' : ''}{currentData?.change?.toFixed(2) || '0.00'}%
            </div>
          </div>
          
          <div className={`${cardClasses} border rounded-xl p-6`}>
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-sm font-medium text-gray-400">24h Volume</h4>
              <Activity className="w-4 h-4 text-gray-400" />
            </div>
            <div className="text-2xl font-bold">
              ${currentData?.volume ? (currentData.volume / 1e9).toFixed(2) : '0.00'}B
            </div>
            <div className="text-sm text-gray-400">Trading volume</div>
          </div>
          
          <div className={`${cardClasses} border rounded-xl p-6`}>
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-sm font-medium text-gray-400">Open Interest</h4>
              <BarChart3 className="w-4 h-4 text-gray-400" />
            </div>
            <div className="text-2xl font-bold">
              ${currentData?.oi ? (currentData.oi / 1e6).toFixed(1) : '0.0'}M
            </div>
            <div className="text-sm text-gray-400">Contracts</div>
          </div>
          
          <div className={`${cardClasses} border rounded-xl p-6`}>
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-sm font-medium text-gray-400">RSI</h4>
              <div className={`w-2 h-2 rounded-full ${
                currentTech?.rsi > 70 ? 'bg-red-400' : 
                currentTech?.rsi < 30 ? 'bg-green-400' : 'bg-yellow-400'
              }`} />
            </div>
            <div className="text-2xl font-bold">{currentTech?.rsi?.toFixed(1) || 'N/A'}</div>
            <div className="text-sm text-gray-400">
              {currentTech?.rsi > 70 ? 'Overbought' : 
               currentTech?.rsi < 30 ? 'Oversold' : 'Neutral'}
            </div>
          </div>
        </div>
        
        {/* Main Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className={`${cardClasses} border rounded-xl p-6`}>
            <h3 className="text-lg font-semibold mb-4">Price Action</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={priceHistory}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="time" stroke="#9CA3AF" fontSize={12} />
                <YAxis stroke="#9CA3AF" fontSize={12} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: theme === 'dark' ? '#1F2937' : '#FFFFFF',
                    border: '1px solid #374151',
                    borderRadius: '8px'
                  }}
                />
                <Line type="monotone" dataKey="price" stroke="#3B82F6" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          
          <div className={`${cardClasses} border rounded-xl p-6`}>
            <h3 className="text-lg font-semibold mb-4">Technical Indicators</h3>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between mb-2">
                  <span className="text-sm">Sentiment</span>
                  <span className={`text-sm ${currentTech?.sentiment > 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {currentTech?.sentiment?.toFixed(3) || 'N/A'}
                  </span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full ${currentTech?.sentiment > 0 ? 'bg-green-400' : 'bg-red-400'}`}
                    style={{ width: `${Math.abs(currentTech?.sentiment || 0) * 50 + 50}%` }}
                  />
                </div>
              </div>
              
              <div>
                <div className="flex justify-between mb-2">
                  <span className="text-sm">MACD</span>
                  <span className={`text-sm ${currentTech?.macd > 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {currentTech?.macd?.toFixed(4) || 'N/A'}
                  </span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full ${currentTech?.macd > 0 ? 'bg-green-400' : 'bg-red-400'}`}
                    style={{ width: `${Math.min(100, Math.abs(currentTech?.macd || 0) * 1000 + 50)}%` }}
                  />
                </div>
              </div>
              
              <div>
                <div className="flex justify-between mb-2">
                  <span className="text-sm">Funding Rate</span>
                  <span className={`text-sm ${currentTech?.funding > 0 ? 'text-red-400' : 'text-green-400'}`}>
                    {((currentTech?.funding || 0) * 100).toFixed(4)}%
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Prediction Display */}
        {prediction && (
          <div className={`${cardClasses} border rounded-xl p-6`}>
            <h3 className="text-lg font-semibold mb-4">Latest Prediction</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="text-center">
                <div className={`text-4xl font-bold ${prediction.signalColor} mb-2`}>
                  {prediction.signal}
                </div>
                <div className="text-sm text-gray-400">Signal</div>
              </div>
              <div className="text-center">
                <div className="text-4xl font-bold text-blue-400 mb-2">
                  {(prediction.confidence * 100).toFixed(1)}%
                </div>
                <div className="text-sm text-gray-400">Confidence</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-semibold mb-1">AI Model</div>
                <div className="text-sm text-gray-400">{prediction.timestamp}</div>
              </div>
            </div>
            
            {/* Confidence meter */}
            <div className="mt-6">
              <div className="flex justify-between mb-2">
                <span className="text-sm">Confidence Level</span>
                <span className="text-sm">{(prediction.confidence * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-3">
                <div
                  className="h-3 rounded-full bg-gradient-to-r from-red-500 via-yellow-500 to-green-500"
                  style={{ width: `${prediction.confidence * 100}%` }}
                />
              </div>
            </div>
          </div>
        )}
        
        {/* Alerts */}
        {alerts.length > 0 && (
          <div className={`${cardClasses} border rounded-xl p-6`}>
            <h3 className="text-lg font-semibold mb-4 flex items-center">
              <AlertCircle className="w-5 h-5 mr-2" />
              Recent Alerts
            </h3>
            <div className="space-y-3">
              {alerts.map(alert => (
                <div
                  key={alert.id}
                  className={`p-3 rounded-lg border-l-4 ${
                    alert.type === 'success' ? 'bg-green-900/20 border-green-400' :
                    alert.type === 'danger' ? 'bg-red-900/20 border-red-400' :
                    alert.type === 'warning' ? 'bg-yellow-900/20 border-yellow-400' :
                    'bg-gray-900/20 border-gray-400'
                  }`}
                >
                  <div className="flex justify-between items-start">
                    <div className="text-sm">{alert.message}</div>
                    <div className="text-xs text-gray-400">{alert.timestamp}</div>
                  </div>
                  {alert.confidence && (
                    <div className="text-xs text-gray-400 mt-1">
                      Confidence: {(alert.confidence * 100).toFixed(1)}%
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
        
        {/* Prediction History */}
        {predictionHistory.length > 0 && (
          <div className={`${cardClasses} border rounded-xl p-6`}>
            <h3 className="text-lg font-semibold mb-4">Prediction History</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-700">
                    <th className="text-left p-2">Time</th>
                    <th className="text-left p-2">Token</th>
                    <th className="text-left p-2">Signal</th>
                    <th className="text-left p-2">Confidence</th>
                    <th className="text-left p-2">Model</th>
                  </tr>
                </thead>
                <tbody>
                  {predictionHistory.slice(-10).reverse().map((pred, idx) => (
                    <tr key={idx} className="border-b border-gray-800">
                      <td className="p-2">{pred.timestamp}</td>
                      <td className="p-2 font-medium">{pred.token}</td>
                      <td className={`p-2 font-bold ${pred.signalColor}`}>{pred.signal}</td>
                      <td className="p-2">{(pred.confidence * 100).toFixed(1)}%</td>
                      <td className="p-2">{pred.model}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default EthosXDashboard;