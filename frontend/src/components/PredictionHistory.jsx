import React from 'react';
import { Clock } from 'lucide-react';
import Animation from './Animation';
import './PredictionHistory.css';

const PredictionHistory = ({ predictionHistory, theme = 'dark' }) => {
  if (!predictionHistory || predictionHistory.length === 0) return null;

  return (
    <Animation type="fadeIn" duration="0.6s">
      <div className="prediction-history" data-theme={theme}>
        <div className="prediction-history__header">
          <Clock className="prediction-history__header-icon" />
          Prediction History
        </div>
        <div className="prediction-history__table-container">
          <table className="prediction-history__table">
            <thead>
              <tr>
                <th>Time</th>
                <th>Token</th>
                <th>Signal</th>
                <th>Confidence</th>
                <th>Model</th>
              </tr>
            </thead>
            <tbody>
              {predictionHistory.slice(-10).reverse().map((pred, idx) => (
                <tr key={idx}>
                  <td className="prediction-history__time" data-label="Time">
                    {pred.timestamp}
                  </td>
                  <td className="prediction-history__token" data-label="Token">
                    {pred.token}
                  </td>
                  <td
                    className={`prediction-history__signal prediction-history__signal--${pred.signal.toLowerCase()}`}
                    data-label="Signal"
                  >
                    {pred.signal}
                  </td>
                  <td className="prediction-history__confidence" data-label="Confidence">
                    <div className="prediction-history__confidence-text">
                      {(pred.confidence * 100).toFixed(1)}%
                    </div>
                    <div className="prediction-history__confidence-bar">
                      <div
                        className="prediction-history__confidence-fill"
                        style={{ width: `${pred.confidence * 100}%` }}
                      />
                    </div>
                  </td>
                  <td className="prediction-history__model" data-label="Model">
                    {pred.model}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </Animation>
  );
};

export default PredictionHistory;