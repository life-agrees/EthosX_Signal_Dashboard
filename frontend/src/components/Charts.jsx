import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import './Charts.css';

const Charts = ({ 
  priceHistory, 
  currentTech, 
  theme = 'dark' 
}) => {
  const getTooltipStyles = () => ({
    backgroundColor: theme === 'dark' ? '#1F2937' : '#FFFFFF',
    border: theme === 'dark' ? '1px solid #374151' : '1px solid #E5E7EB',
    borderRadius: '8px',
    boxShadow: theme === 'dark' ? '0 4px 6px -1px rgba(0, 0, 0, 0.1)' : '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
  });

  const getGridColor = () => theme === 'dark' ? '#374151' : '#E5E7EB';
  const getAxisColor = () => theme === 'dark' ? '#9CA3AF' : '#6B7280';

  return (
    <div className="charts-container">
      <div className={`chart-card ${theme === 'light' ? 'light' : ''}`}>
        <h3 className="chart-title">Price Action</h3>
        <div className="chart-wrapper">
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={priceHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke={getGridColor()} />
              <XAxis 
                dataKey="time" 
                stroke={getAxisColor()} 
                fontSize={12} 
                tick={{ fill: getAxisColor() }}
              />
              <YAxis 
                stroke={getAxisColor()} 
                fontSize={12} 
                tick={{ fill: getAxisColor() }}
              />
              <Tooltip contentStyle={getTooltipStyles()} />
              <Line 
                type="monotone" 
                dataKey="price" 
                stroke="#3B82F6" 
                strokeWidth={2} 
                dot={false}
                activeDot={{ r: 4, fill: '#3B82F6' }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
      
      <div className={`chart-card ${theme === 'light' ? 'light' : ''}`}>
        <h3 className="chart-title">Technical Indicators</h3>
        <div className="technical-indicators">
          
          <div className="indicator-row">
            <div className="indicator-header">
              <span className="indicator-label">Sentiment</span>
              <span className={`indicator-value ${currentTech?.sentiment > 0 ? 'positive' : 'negative'}`}>
                {currentTech?.sentiment?.toFixed(3) || 'N/A'}
              </span>
            </div>
            <div className="indicator-bar">
              <div
                className={`indicator-fill ${currentTech?.sentiment > 0 ? 'positive' : 'negative'}`}
                style={{ width: `${Math.abs(currentTech?.sentiment || 0) * 50 + 50}%` }}
              />
            </div>
          </div>
          
          <div className="indicator-row">
            <div className="indicator-header">
              <span className="indicator-label">MACD</span>
              <span className={`indicator-value ${currentTech?.macd > 0 ? 'positive' : 'negative'}`}>
                {currentTech?.macd?.toFixed(4) || 'N/A'}
              </span>
            </div>
            <div className="indicator-bar">
              <div
                className={`indicator-fill ${currentTech?.macd > 0 ? 'positive' : 'negative'}`}
                style={{ width: `${Math.min(100, Math.abs(currentTech?.macd || 0) * 1000 + 50)}%` }}
              />
            </div>
          </div>
          
          <div className="indicator-row">
            <div className="indicator-header">
              <span className="indicator-label">Funding Rate</span>
              <span className={`indicator-value ${currentTech?.funding > 0 ? 'negative' : 'positive'}`}>
                {((currentTech?.funding || 0) * 100).toFixed(4)}%
              </span>
            </div>
            <div className="indicator-bar">
              <div
                className={`indicator-fill ${currentTech?.funding > 0 ? 'negative' : 'positive'}`}
                style={{ width: `${Math.min(100, Math.abs(currentTech?.funding || 0) * 10000 + 30)}%` }}
              />
            </div>
          </div>
          
        </div>
      </div>
    </div>
  );
};

export default Charts;