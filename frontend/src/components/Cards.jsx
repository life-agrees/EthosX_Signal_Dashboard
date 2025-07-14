import React from 'react';
import './Cards.css';

// Generic Card Container Component
export const Card = ({ 
  children, 
  className = '', 
  type = 'dashboard', // dashboard, market-data, prediction, alert
  theme = 'dark',
  hover = true 
}) => {
  const baseClasses = 'dashboard-card';
  const typeClasses = {
    dashboard: '',
    'market-data': 'market-data-card',
    prediction: 'prediction-card',
    alert: 'alert-card'
  };
  
  const themeClasses = theme === 'dark'
    ? 'bg-gray-800 border-gray-700'
    : 'bg-white border-gray-200';
  
  return (
    <div className={`
      ${baseClasses} 
      ${typeClasses[type]} 
      ${themeClasses} 
      border rounded-xl p-6 
      ${className}
    `}>
      {children}
    </div>
  );
};

// Prediction Card Component using your CSS
export const PredictionCard = ({ prediction, theme = 'dark' }) => {
  if (!prediction) return null;
  
  return (
    <Card type="prediction" theme={theme}>
      <h3 className="text-lg font-semibold mb-4">Latest Prediction</h3>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="text-center">
          <div className={`signal-badge ${prediction.signal.toLowerCase()}`}>
            {prediction.signal}
          </div>
          <div className="text-sm text-gray-400 mt-2">Signal</div>
        </div>
        <div className="text-center">
          <div className={`confidence-badge ${getConfidenceLevel(prediction.confidence)}`}>
            {(prediction.confidence * 100).toFixed(1)}%
          </div>
          <div className="text-sm text-gray-400 mt-2">Confidence</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-semibold mb-1">AI Model</div>
          <div className="text-sm text-gray-400">{prediction.model}</div>
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
    </Card>
  );
};

// Helper function for confidence levels
const getConfidenceLevel = (confidence) => {
  if (confidence >= 0.8) return 'confidence-high';
  if (confidence >= 0.6) return 'confidence-medium';
  return 'confidence-low';
};

// Alert Card Component using your CSS
export const AlertCard = ({ alert }) => {
  return (
    <Card type="alert" className="p-3">
      <div className="flex justify-between items-start">
        <div className="text-sm">{alert.message}</div>
        <div className="text-xs text-gray-400">{alert.timestamp}</div>
      </div>
      {alert.confidence && (
        <div className={`confidence-badge ${getConfidenceLevel(alert.confidence)} mt-2`}>
          {(alert.confidence * 100).toFixed(1)}%
        </div>
      )}
    </Card>
  );
};


// Connection Status Card Component
export const ConnectionStatusCard = ({ status, theme = 'dark' }) => {
  return (
    <Card theme={theme} className="p-4">
      <div className={`connection-status ${status.toLowerCase()}`}>
        <div className="w-3 h-3 rounded-full bg-current opacity-75"></div>
        <span className="capitalize">{status}</span>
      </div>
    </Card>
  );
};

// Token Selection Card Component
export const TokenCard = ({ token, isSelected, marketData, onClick }) => {
  return (
    <Card 
      className={`p-3 cursor-pointer transition-all ${
        isSelected
          ? 'border-blue-500 bg-blue-500/10'
          : 'border-gray-600 hover:border-gray-500'
      }`}
      onClick={() => onClick(token)}
    >
      <div className="text-sm font-medium">{token}</div>
      <div className={`text-xs ${marketData?.change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
        {marketData?.change >= 0 ? '+' : ''}{marketData?.change?.toFixed(2) || '0.00'}%
      </div>
    </Card>
  );
};

// Stats Card Component
export const StatsCard = ({ title, value, change, icon, theme = 'dark' }) => {
  const changeColor = change >= 0 ? 'text-green-400' : 'text-red-400';
  
  return (
    <Card theme={theme}>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-gray-400">{title}</p>
          <p className="text-2xl font-bold">{value}</p>
          {change !== undefined && (
            <p className={`text-sm ${changeColor}`}>
              {change >= 0 ? '+' : ''}{change.toFixed(2)}%
            </p>
          )}
        </div>
        {icon && (
          <div className="text-2xl opacity-75">
            {icon}
          </div>
        )}
      </div>
    </Card>
  );
};

export default Card;