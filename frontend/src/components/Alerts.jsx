import React from 'react';
import { AlertCircle, X } from 'lucide-react';
import Animation from './Animation'; // Update path as needed
import './Alerts.css'; // Fixed import name

const Alert = ({ 
  alerts, 
  theme = 'dark',
  onDismiss
}) => {
  const cardClasses = theme === 'dark'
    ? 'bg-gray-800 border-gray-700'
    : 'bg-white border-gray-200';

  const textClasses = theme === 'dark'
    ? 'text-white'
    : 'text-gray-900';

  const mutedTextClasses = theme === 'dark'
    ? 'text-gray-400'
    : 'text-gray-500';

  if (alerts.length === 0) return null;

  const getAlertTypeClass = (type) => {
    switch (type) {
      case 'success': return 'alert-success';
      case 'danger': return 'alert-danger';
      case 'warning': return 'alert-warning';
      case 'info': return 'alert-info';
      case 'error': return 'alert-error';
      default: return 'alert-info';
    }
  };

  return (
    <Animation type="fadeIn" duration="0.6s" className="mb-8">
      <div className={`${cardClasses} border rounded-xl p-6 card-hover`}>
        <h3 className={`text-lg font-semibold mb-4 flex items-center ${textClasses}`}>
          <AlertCircle className="w-5 h-5 mr-2" />
          Recent Alerts
        </h3>
        
        <div className="space-y-3">
          {alerts.map((alert, index) => (
            <Animation 
              key={alert.id} 
              type="slideIn" 
              duration="0.4s"
              delay={`${index * 0.1}s`}
            >
              <div
                className={`
                  alert-item
                  ${getAlertTypeClass(alert.type)}
                  p-4 rounded-lg border-l-4 relative
                `}
              >
                <div className="flex justify-between items-start">
                  <div className={`text-sm font-medium ${textClasses}`}>
                    {alert.message}
                  </div>
                  <div className="flex items-center gap-2">
                    <div className={`text-xs ${mutedTextClasses} whitespace-nowrap`}>
                      {alert.timestamp}
                    </div>
                    {onDismiss && (
                      <button
                        onClick={() => onDismiss(alert.id)}
                        className={`alert-close p-1 rounded hover:bg-black/10 ${mutedTextClasses}`}
                        aria-label="Dismiss alert"
                      >
                        <X className="w-3 h-3" />
                      </button>
                    )}
                  </div>
                </div>
                
                {alert.confidence && (
                  <div className={`text-xs ${mutedTextClasses} mt-2`}>
                    Confidence: {(alert.confidence * 100).toFixed(1)}%
                  </div>
                )}
              </div>
            </Animation>
          ))}
        </div>
      </div>
    </Animation>
  );
};

export default Alert;