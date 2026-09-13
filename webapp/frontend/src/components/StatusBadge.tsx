import React from 'react';
import { AlertTriangle, AlertCircle, CheckCircle2, TrendingUp, HelpCircle } from 'lucide-react';

interface StatusBadgeProps {
  status: string;
  className?: string;
  size?: 'sm' | 'md' | 'lg';
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  className = '',
  size = 'md',
}) => {
  const norm = status.trim().toUpperCase();

  let label = status;
  let bg = 'bg-slate-100 text-slate-700 border-slate-200';
  let Icon = HelpCircle;

  if (norm.includes('HIGH_STRESS') || norm.includes('HIGH STRESS')) {
    label = 'High Stress Risk';
    bg = 'bg-amber-50 text-amber-900 border-amber-300 font-semibold';
    Icon = AlertTriangle;
  } else if (norm.includes('WATCH')) {
    label = 'Watch';
    bg = 'bg-yellow-50 text-yellow-900 border-yellow-300 font-semibold';
    Icon = AlertCircle;
  } else if (norm.includes('STABLE')) {
    label = 'Stable';
    bg = 'bg-emerald-50 text-emerald-900 border-emerald-300 font-semibold';
    Icon = CheckCircle2;
  } else if (norm.includes('IMPROVING')) {
    label = 'Improving';
    bg = 'bg-teal-50 text-teal-900 border-teal-300 font-semibold';
    Icon = TrendingUp;
  } else {
    label = 'Prediction Unavailable';
    bg = 'bg-slate-100 text-slate-600 border-slate-200';
    Icon = HelpCircle;
  }

  const sizeClasses = {
    sm: 'text-xs px-2 py-0.5 gap-1',
    md: 'text-sm px-3 py-1 gap-1.5',
    lg: 'text-base px-3.5 py-1.5 gap-2 font-bold',
  };

  const iconSizes = {
    sm: 'w-3.5 h-3.5',
    md: 'w-4 h-4',
    lg: 'w-5 h-5',
  };

  return (
    <span
      className={`inline-flex items-center rounded-full border shadow-sm ${bg} ${sizeClasses[size]} ${className}`}
    >
      <Icon className={`${iconSizes[size]} flex-shrink-0`} />
      <span>{label}</span>
    </span>
  );
};
