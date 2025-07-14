import React from 'react';
import { TrendingUp, TrendingDown, DollarSign, Activity, BarChart3 } from 'lucide-react';
import Animation from './Animation'; // Update path as needed
import Badges from './Badges'; // Update path as needed
import './MarketData.css';

const MarketData = ({ currentData, currentTech, selectedToken, theme = 'dark' }) => {
  const lightClass = theme === 'light' ? ' light' : '';

  return (
    <div className="market-data-grid">
      <Animation type="fadeIn" duration="0.6s" delay="0.2s">
        {/* Price Card */}
        <div className={`market-card${lightClass}`}>
          <div className="market-card-header">
            <h4 className="market-card-title">Price</h4>
            <DollarSign className="market-card-icon" />
          </div>
          <div className="market-card-value">
            ${currentData?.price
              ? currentData.price.toLocaleString(undefined, {
                  maximumFractionDigits: currentData.price < 1 ? 6 : 2,
                })
              : 'Loading...'}
          </div>
          <div className={`market-card-change ${
            currentData?.change > 0 ? 'positive' : currentData?.change < 0 ? 'negative' : 'neutral'
          }`}> 
            {currentData?.change > 0 ? <TrendingUp className="market-card-icon" /> : <TrendingDown className="market-card-icon" />}
            {currentData?.change >= 0 ? '+' : ''}{currentData?.change?.toFixed(2) || '0.00'}%
          </div>
          <div className="market-card-subtitle">24h change</div>
        </div>

        {/* Volume Card */}
        <div className={`market-card${lightClass}`}>
          <div className="market-card-header">
            <h4 className="market-card-title">24h Volume</h4>
            <Activity className="market-card-icon" />
          </div>
          <div className="market-card-value">
            ${currentData?.volume ? (currentData.volume / 1e9).toFixed(2) : '0.00'}B
          </div>
          <div className="market-card-subtitle">Trading volume</div>
        </div>

        {/* Open Interest Card */}
        <div className={`market-card${lightClass}`}>
          <div className="market-card-header">
            <h4 className="market-card-title">Open Interest</h4>
            <BarChart3 className="market-card-icon" />
          </div>
          <div className="market-card-value">
            ${currentData?.oi ? (currentData.oi / 1e6).toFixed(1) : '0.0'}M
          </div>
          <div className="market-card-subtitle">Contracts</div>
        </div>

        {/* RSI Card */}
        <div className={`market-card${lightClass}`}>
          <div className="market-card-header">
            <h4 className="market-card-title">RSI</h4>
            <span className={`rsi-indicator ${
              currentTech?.rsi > 70 ? 'rsi-overbought' : currentTech?.rsi < 30 ? 'rsi-oversold' : 'rsi-neutral'
            }`} />
          </div>
          <div className="market-card-value">
            {currentTech?.rsi?.toFixed(1) || 'N/A'}
          </div>
          <Badges
            type={currentTech?.rsi > 70 ? 'danger' : currentTech?.rsi < 30 ? 'success' : 'default'}
            size="small"
          >
            {currentTech?.rsi > 70 ? 'Overbought' : currentTech?.rsi < 30 ? 'Oversold' : 'Neutral'}
          </Badges>
        </div>
      </Animation>
    </div>
  );
};

export default MarketData;