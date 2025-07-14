import React from 'react';
import { BarChart3, Wifi, WifiOff, Bell, BellOff } from 'lucide-react';
import Badges from './Badges';
import './Header.css';

const Header = ({ 
  connectionStatus, 
  apiHealth, 
  isAutoRefresh, 
  setIsAutoRefresh, 
  connectWebSocket, 
  ws, 
  theme, 
  setTheme 
}) => {
  return (
    <div className={`header ${theme}`}>
      <div className="header-content">
        <div className="header-left">
          <div className="header-logo">
            <BarChart3 className="w-8 h-8 text-blue-500" />
            <h1 className="header-title">EthosX Dashboard</h1>
          </div>
          
          <div className="header-status">
            {connectionStatus === 'connected' ? (
              <>
                <Wifi className="w-4 h-4 text-green-400" />
                <Badges type="success" size="small">Live</Badges>
              </>
            ) : (
              <>
                <WifiOff className="w-4 h-4 text-red-400" />
                <Badges type="danger" size="small">Disconnected</Badges>
              </>
            )}
          </div>
          
          {apiHealth && (
            <div className="header-status">
              <Badges type={apiHealth.status === 'healthy' ? 'success' : 'danger'} size="small">
                API: {apiHealth.status}
              </Badges>
              <Badges type={apiHealth.prediction_service?.available ? 'success' : 'warning'} size="small">
                Models: {apiHealth.prediction_service?.available ? 'Ready' : 'Unavailable'}
              </Badges>
            </div>
          )}
        </div>
        
        <div className="header-right">
          <button
            onClick={() => {
              setIsAutoRefresh(!isAutoRefresh);
              if (!isAutoRefresh) {
                connectWebSocket();
              } else if (ws) {
                ws.close();
              }
            }}
            className={`header-button auto-refresh-btn ${!isAutoRefresh ? 'disabled' : ''}`}
          >
            {isAutoRefresh ? <Bell className="w-4 h-4" /> : <BellOff className="w-4 h-4" />}
          </button>
          
          <button
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            className="header-button theme-toggle"
          >
            {theme === 'dark' ? '☀️' : '🌙'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default Header;