import React from 'react';
import { RefreshCw } from 'lucide-react';
import Animation from './Animation';
import './Loading.css';

const Loading = ({ loading, error, apiHealth, theme = 'dark' }) => {
  if (!loading) return null;

  return (
    <div className={`loading-screen ${theme}`}>
      <Animation type="fadeIn" duration="0.5s">
        <div className="loading-container">
          {/* Spinner Icon */}
          <RefreshCw className="spinner" />
          <p>Loading dashboard...</p>

          {/* Error Message */}
          {error && (
            <Animation type="slideIn" duration="0.4s" delay="0.2s">
              <p className="text-red-400 mt-2">{error}</p>
            </Animation>
          )}

          {/* API Health Info */}
          {apiHealth && (
            <Animation type="fadeIn" duration="0.4s" delay="0.3s">
              <div className="mt-4 text-sm text-gray-400">
                <p>API Status: {apiHealth.status}</p>
                <p>Data Source: {apiHealth.data_source}</p>
              </div>
            </Animation>
          )}

          {/* Example: Dots Spinner (optional) */}
          {/*
          <div className="spinner-dots">
            <span className="spinner-dot" />
            <span className="spinner-dot" />
            <span className="spinner-dot" />
          </div>
          */}

          {/* Example: Progress Bar (optional) */}
          {/*
          <div className="loading-progress">
            <div className="loading-progress-bar" />
          </div>
          */}
        </div>
      </Animation>
    </div>
  );
};

export default Loading;