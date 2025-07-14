import React from 'react';
import './Animation.css';

const Animation = ({ 
  children, 
  type = 'fadeIn', 
  duration = '0.3s',   
  delay = '0s',
  className = '' 
}) => {
  // Map animation types to CSS classes
  const getAnimationClass = (animationType) => {
    const animationMap = {
      fadeIn: 'fadeIn',
      slideIn: 'slideIn',
      scaleIn: 'scaleIn',
      bounce: 'bounce',
      pulse: 'pulse',
      spin: 'spinner',
      chartEnter: 'chart-enter',
      alertEnter: 'alert-enter',
      staggerItem: 'stagger-item'
    };
    return animationMap[animationType] || animationType;
  };

  const animationStyle = {
    animationDuration: duration,
    animationDelay: delay,
    animationFillMode: 'both'
  };

  const animationClass = getAnimationClass(type);

  return (
    <div 
      className={`${animationClass} ${className}`} 
      style={animationStyle}
    >
      {children}
    </div>
  );
};

export default Animation;