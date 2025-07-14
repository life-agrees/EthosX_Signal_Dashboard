import React from 'react';
import { Zap } from 'lucide-react';

const TokenSelection = ({ 
  selectedToken, 
  setSelectedToken, 
  supportedTokens, 
  marketData, 
  makePrediction, 
  loading, 
  isGeneratingPrediction, 
  apiHealth, 
  theme = 'dark' 
}) => {
  return (
    <div className="controls__panel controls__token-selection">
      <div className="controls__panel-header">
        <Zap className="controls__panel-icon" />
        Token Selection
      </div>
      
      <div className="controls__token-grid">
        {supportedTokens.map(token => (
          <button
            key={token}
            onClick={() => setSelectedToken(token)}
            className={`controls__token-button ${
              selectedToken === token ? 'controls__token-button--selected' : ''
            }`}
          >
            <div className="controls__token-name">{token}</div>
            <div className={`controls__token-change ${
              marketData[token]?.change >= 0 
                ? 'controls__token-change--positive' 
                : 'controls__token-change--negative'
            }`}>
              {marketData[token]?.change >= 0 ? '+' : ''}{marketData[token]?.change?.toFixed(2) || '0.00'}%
            </div>
          </button>
        ))}
      </div>
      
      <button
        onClick={makePrediction}
        disabled={loading || isGeneratingPrediction || !apiHealth?.prediction_service?.available}
        className="controls__generate-button"
      >
        {(loading || isGeneratingPrediction) && (
          <div className="controls__loading-spinner" />
        )}
        <Zap className="controls__generate-button-icon" />
        {(loading || isGeneratingPrediction) ? 'Generating...' : 'Generate Signal'}
      </button>
    </div>
  );
};

export default TokenSelection;