import React from 'react';
import { ShieldCheck, ShieldAlert, Shield } from 'lucide-react';

interface ConfidenceBadgeProps {
  confidence: 'HIGH' | 'MEDIUM' | 'LOW' | string;
  className?: string;
}

export const ConfidenceBadge: React.FC<ConfidenceBadgeProps> = ({
  confidence,
  className = '',
}) => {
  const norm = (confidence || '').toUpperCase();

  let label = 'Low Confidence';
  let badgeClass = 'bg-slate-100 text-slate-600 border-slate-200';
  let Icon = Shield;

  if (norm === 'HIGH') {
    label = 'High Confidence';
    badgeClass = 'bg-emerald-50 text-emerald-800 border-emerald-300';
    Icon = ShieldCheck;
  } else if (norm === 'MEDIUM') {
    label = 'Moderate Confidence';
    badgeClass = 'bg-amber-50 text-amber-800 border-amber-300';
    Icon = ShieldAlert;
  }

  return (
    <div className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-semibold border ${badgeClass} ${className}`}>
      <Icon className="w-3.5 h-3.5 flex-shrink-0" />
      <span>Data Confidence: {label}</span>
    </div>
  );
};
