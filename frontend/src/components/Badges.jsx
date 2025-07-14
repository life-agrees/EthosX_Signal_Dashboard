import React from 'react';
import './Badges.css';

const Badges = ({ 
  type = 'default', 
  size = 'medium', 
  variant = 'filled', // filled, outline, gradient
  children, 
  className = '',
  icon = null,
  dot = false,
  dotColor = null,
  animated = false,
  onClick = null
}) => {
  // Base badge class from your CSS
  const baseClass = 'badge';
  
  // Size classes from your CSS
  const sizeClasses = {
    small: 'badge-sm',
    medium: '', // default badge size
    large: 'badge-lg'
  };

  // Type classes from your CSS
  const getTypeClass = (type, variant) => {
    const typeMap = {
      // Alert types
      success: 'badge-success',
      danger: 'badge-danger',
      warning: 'badge-warning',
      info: 'badge-info',
      error: 'badge-error',
      
      // Signal types
      buy: 'badge-buy',
      sell: 'badge-sell',
      hold: 'badge-hold',
      
      // Connection status
      connected: 'badge-connected',
      disconnected: 'badge-disconnected',
      connecting: 'badge-connecting',
      
      // RSI status
      overbought: 'badge-overbought',
      oversold: 'badge-oversold',
      neutral: 'badge-neutral',
      
      // API status
      'api-healthy': 'badge-api-healthy',
      'api-unhealthy': 'badge-api-unhealthy',
      'api-loading': 'badge-api-loading',
      
      // Confidence levels
      'confidence-low': 'badge-confidence-low',
      'confidence-medium': 'badge-confidence-medium',
      'confidence-high': 'badge-confidence-high',
      
      // Token
      token: 'badge-token',
      'token-selected': 'badge-token-selected'
    };

    // Handle variants
    if (variant === 'outline') {
      const outlineMap = {
        success: 'badge-outline badge-outline-green',
        danger: 'badge-outline badge-outline-red',
        warning: 'badge-outline badge-outline-yellow',
        info: 'badge-outline badge-outline-blue',
        error: 'badge-outline badge-outline-red'
      };
      return outlineMap[type] || 'badge-outline';
    }

    if (variant === 'gradient') {
      const gradientMap = {
        success: 'badge-gradient-green',
        danger: 'badge-gradient-red',
        info: 'badge-gradient-blue',
        warning: 'badge-gradient-purple'
      };
      return gradientMap[type] || 'badge-gradient-blue';
    }

    return typeMap[type] || 'badge-neutral';
  };

  // Status dot component
  const StatusDot = ({ color, animated }) => {
    const dotClasses = {
      green: 'status-dot-green',
      red: 'status-dot-red',
      yellow: 'status-dot-yellow',
      blue: 'status-dot-blue',
      gray: 'status-dot-gray'
    };

    return (
      <span 
        className={`
          status-dot 
          ${dotClasses[color] || 'status-dot-gray'}
          ${animated ? 'status-dot-pulse' : ''}
        `}
      />
    );
  };

  // Combine all classes
  const badgeClasses = `
    ${baseClass}
    ${sizeClasses[size]}
    ${getTypeClass(type, variant)}
    ${icon ? 'badge-with-icon' : ''}
    ${onClick ? 'cursor-pointer' : ''}
    ${className}
  `.trim();

  return (
    <span 
      className={badgeClasses}
      onClick={onClick}
    >
      {dot && <StatusDot color={dotColor || type} animated={animated} />}
      {icon && <span className="badge-icon">{icon}</span>}
      {children}
    </span>
  );
};

// Specialized badge components using your CSS classes
export const SignalBadge = ({ signal, ...props }) => (
  <Badges type={signal} {...props} />
);

export const ConnectionBadge = ({ status, animated = true, ...props }) => (
  <Badges 
    type={status} 
    dot={true} 
    dotColor={status === 'connected' ? 'green' : status === 'disconnected' ? 'red' : 'yellow'}
    animated={animated}
    {...props}
  />
);

export const ConfidenceBadge = ({ confidence, ...props }) => {
  const getConfidenceType = (conf) => {
    if (conf >= 0.8) return 'confidence-high';
    if (conf >= 0.6) return 'confidence-medium';
    return 'confidence-low';
  };

  return (
    <Badges 
      type={getConfidenceType(confidence)} 
      {...props}
    >
      {(confidence * 100).toFixed(1)}%
    </Badges>
  );
};

export const TokenBadge = ({ token, selected = false, ...props }) => (
  <Badges 
    type={selected ? 'token-selected' : 'token'} 
    {...props}
  >
    {token}
  </Badges>
);

export const NumberBadge = ({ number, ...props }) => (
  <Badges 
    className="badge-number" 
    {...props}
  >
    {number}
  </Badges>
);

export const RSIBadge = ({ rsiValue, ...props }) => {
  const getRSIType = (rsi) => {
    if (rsi >= 70) return 'overbought';
    if (rsi <= 30) return 'oversold';
    return 'neutral';
  };

  return (
    <Badges 
      type={getRSIType(rsiValue)} 
      {...props}
    >
      RSI: {rsiValue}
    </Badges>
  );
};

export const APIStatusBadge = ({ status, ...props }) => (
  <Badges 
    type={`api-${status}`} 
    dot={true}
    dotColor={status === 'healthy' ? 'green' : status === 'unhealthy' ? 'red' : 'gray'}
    {...props}
  >
    {status.toUpperCase()}
  </Badges>
);

export default Badges;