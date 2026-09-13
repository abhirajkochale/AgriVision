import React from 'react';
import { Lightbulb, Info, AlertTriangle, CheckCircle2, AlertCircle } from 'lucide-react';
import { DashboardRecommendation } from '../types/api';
import { ConfidenceBadge } from './ConfidenceBadge';

interface RecommendationCardProps {
  recommendation: DashboardRecommendation;
}

export const RecommendationCard: React.FC<RecommendationCardProps> = ({ recommendation }) => {
  const { status, severity, headline, explanation, recommended_action, data_confidence } = recommendation;

  // Determine accent theme based on severity
  let borderTheme = 'border-slate-200 bg-white';
  let badgeTheme = 'bg-slate-100 text-slate-800';
  let Icon = Info;

  if (severity === 'high') {
    borderTheme = 'border-amber-200 bg-gradient-to-br from-white to-amber-50/40';
    badgeTheme = 'bg-amber-100 text-amber-900 border border-amber-300';
    Icon = AlertTriangle;
  } else if (severity === 'medium') {
    borderTheme = 'border-yellow-200 bg-gradient-to-br from-white to-yellow-50/40';
    badgeTheme = 'bg-yellow-100 text-yellow-900 border border-yellow-300';
    Icon = AlertCircle;
  } else if (severity === 'low' || severity === 'positive') {
    borderTheme = 'border-emerald-200 bg-gradient-to-br from-white to-emerald-50/40';
    badgeTheme = 'bg-emerald-100 text-emerald-900 border border-emerald-300';
    Icon = CheckCircle2;
  }

  return (
    <div className={`rounded-2xl p-5 border shadow-sm ${borderTheme} flex flex-col justify-between transition-all`}>
      <div>
        {/* Card Header with Confidence Badge */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2.5 mb-4">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-xl bg-agri-100 text-agri-800 flex items-center justify-center flex-shrink-0">
              <Lightbulb className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-base font-bold text-slate-900">Farmer Advisory</h3>
              <p className="text-xs text-slate-500">Plain-language field management guidance</p>
            </div>
          </div>
          <ConfidenceBadge confidence={data_confidence} />
        </div>

        {/* Status & Headline */}
        <div className="bg-white/80 backdrop-blur-sm rounded-xl p-4 border border-slate-200/80 shadow-xs mb-3.5">
          <div className="flex items-center gap-2 mb-2">
            <span className={`text-xs font-bold px-2.5 py-0.5 rounded-full inline-flex items-center gap-1 ${badgeTheme}`}>
              <Icon className="w-3.5 h-3.5" />
              {status}
            </span>
          </div>
          <h4 className="text-lg font-black text-slate-900 tracking-tight leading-snug">
            {headline}
          </h4>
          <p className="text-sm text-slate-700 font-normal mt-1 leading-relaxed">
            {explanation}
          </p>
        </div>

        {/* Recommended Action */}
        <div className="bg-white/90 rounded-xl p-4 border border-slate-200/90 shadow-xs">
          <span className="text-xs font-bold text-slate-500 uppercase tracking-wider block mb-1">
            Recommended Action
          </span>
          <p className="text-sm font-semibold text-slate-900 leading-normal">
            {recommended_action}
          </p>
        </div>
      </div>

      {/* Trust & Safety Disclaimer */}
      <div className="mt-4 pt-3 border-t border-slate-200/60 text-[11px] text-slate-500 flex items-center gap-1.5">
        <Info className="w-3.5 h-3.5 text-slate-400 flex-shrink-0" />
        <span>Advisory signal based on satellite vegetation index trends. Not a crop disease diagnosis.</span>
      </div>
    </div>
  );
};
