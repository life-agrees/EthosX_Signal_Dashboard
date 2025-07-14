import React from 'react';
import { Users } from 'lucide-react';

const Subscription = ({ 
  isSubscribed, 
  email, 
  setEmail, 
  handleSubscribe, 
  subscriberCount, 
  theme = 'dark' 
}) => {
  return (
    <div className="controls__panel controls__subscription">
      <div className="controls__panel-header">
        <Users className="controls__panel-icon" />
        Alert Subscription
      </div>
      
      {!isSubscribed ? (
        <div className="controls__subscription-form">
          <input
            type="email"
            placeholder="Enter your email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="controls__email-input"
          />
          <button
            onClick={handleSubscribe}
            className="controls__subscribe-button"
          >
            Subscribe
          </button>
        </div>
      ) : (
        <div className="controls__subscription-success">
          <div className="controls__subscription-success-icon">✓</div>
          <div className="controls__subscription-success-text">Subscribed</div>
          <div className="controls__subscription-success-subtext">
            You'll receive high-confidence alerts
          </div>
        </div>
      )}
      
      <div className="controls__subscriber-count">
        {subscriberCount} active subscribers
      </div>
    </div>
  );
};

export default Subscription;