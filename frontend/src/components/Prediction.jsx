import React from 'react';
import Animation from './Animation';
import Badges from './Badges';
import './Prediction.css';

const Prediction = ({ prediction, theme = 'dark' }) => {
  if (!prediction) return null;
  const lightClass = theme === 'light' ? ' light' : '';

  return (
    <Animation type="fadeIn" duration="0.5s">
      <div className={`prediction-card${lightClass}`}>  
        <h3 className="prediction-title">Latest Prediction</h3>
        <div className="prediction-grid">
          {/* Signal */}
          <div className="prediction-metric">
            <Badges
              type={prediction.signal === 'BUY' ? 'success' : prediction.signal === 'SELL' ? 'danger' : 'warning'}
              size="large"
              className="signal-pulse prediction-value"
            >
              {prediction.signal}
            </Badges>
            <div className="prediction-label">Signal</div>
          </div>

          {/* Confidence */}
          <div className="prediction-metric">
            <div className="prediction-value confidence">
              {(prediction.confidence * 100).toFixed(1)}%
            </div>
            <div className="prediction-label">Confidence</div>
          </div>

          {/* Model */}
          <div className="prediction-metric prediction-model">
            <div className="prediction-model-title">AI Model</div>
            <div className="prediction-model-name">{prediction.model}</div>
          </div>
        </div>

        {/* Confidence Meter */}
        <div className="confidence-meter">
          <div className="confidence-header">
            <span className="confidence-label">Confidence Level</span>
            <span className="confidence-percentage">{(prediction.confidence * 100).toFixed(1)}%</span>
          </div>
          <div className="confidence-bar">
            <div
              className="confidence-fill"
              style={{ width: `${prediction.confidence * 100}%` }}
            />
          </div>
        </div>
      </div>
    </Animation>
  );
};

export default Prediction;