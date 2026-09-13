import React from 'react';
import { Sprout, Wifi, WifiOff, Calendar } from 'lucide-react';

interface HeaderProps {
  monitoringPeriod?: string;
  isOnline: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  monitoringPeriod = 'Week of Sep 03, 2026',
  isOnline,
}) => {
  return (
    <header className="bg-white border-b border-slate-200 sticky top-0 z-20">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-3.5 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        {/* Brand identity */}
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-agri-600 flex items-center justify-center text-white shadow-sm shadow-agri-600/20 flex-shrink-0">
            <Sprout className="w-6 h-6" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-extrabold tracking-tight text-slate-900">AgriVision</h1>
              <span className="text-[10px] uppercase font-bold tracking-wider px-1.5 py-0.5 rounded bg-agri-100 text-agri-800">
                Farmer AI
              </span>
            </div>
            <p className="text-xs text-slate-500 font-medium">Vegetation Health & Operational Forecasts</p>
          </div>
        </div>

        {/* Status & Period Metadata */}
        <div className="flex items-center flex-wrap gap-2 text-xs">
          <div className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-slate-100 text-slate-700 font-medium">
            <Calendar className="w-3.5 h-3.5 text-slate-500" />
            <span>{monitoringPeriod}</span>
          </div>

          <div
            className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg font-medium ${
              isOnline
                ? 'bg-emerald-50 text-emerald-700 border border-emerald-200/60'
                : 'bg-rose-50 text-rose-700 border border-rose-200/60'
            }`}
          >
            {isOnline ? (
              <>
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                <Wifi className="w-3.5 h-3.5" />
                <span>Live Connected</span>
              </>
            ) : (
              <>
                <WifiOff className="w-3.5 h-3.5" />
                <span>Backend Offline</span>
              </>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};
