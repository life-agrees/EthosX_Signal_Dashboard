import React from 'react';
import TokenSelection from './TokenSelection';
import Subscription from './Subscription';
import './Controls.css';

const Controls = ({ 
  selectedToken, 
  setSelectedToken, 
  supportedTokens, 
  marketData, 
  makePrediction, 
  loading, 
  isGeneratingPrediction, 
  apiHealth, 
  isSubscribed, 
  email, 
  setEmail, 
  handleSubscribe, 
  subscriberCount, 
  theme = 'dark' 
}) => {
  // Apply theme to document for CSS variables
  React.useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  return (
    <div className="controls">
      <TokenSelection 
        selectedToken={selectedToken}
        setSelectedToken={setSelectedToken}
        supportedTokens={supportedTokens}
        marketData={marketData}
        makePrediction={makePrediction}
        loading={loading}
        isGeneratingPrediction={isGeneratingPrediction}
        apiHealth={apiHealth}
        theme={theme}
      />
      
      <Subscription 
        isSubscribed={isSubscribed}
        email={email}
        setEmail={setEmail}
        handleSubscribe={handleSubscribe}
        subscriberCount={subscriberCount}
        theme={theme}
      />
    </div>
  );
};

export default Controls;