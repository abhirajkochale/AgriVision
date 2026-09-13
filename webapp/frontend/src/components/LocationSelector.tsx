import React from 'react';
import { MapPin, Check } from 'lucide-react';
import { LocationDetailResponse } from '../types/api';

interface LocationSelectorProps {
  locations: LocationDetailResponse[];
  selectedPointId: string;
  onSelect: (pointId: string) => void;
  isLoading?: boolean;
}

export const LocationSelector: React.FC<LocationSelectorProps> = ({
  locations,
  selectedPointId,
  onSelect,
  isLoading,
}) => {
  // Quick pick featured locations for farmer demonstrations
  const featured = ['P069', 'P077', 'P065', 'P147'];

  return (
    <div className="bg-white rounded-2xl p-4 border border-slate-200 shadow-sm">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-3">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-slate-100 flex items-center justify-center text-slate-600">
            <MapPin className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-slate-900">Monitored Field Location</h2>
            <p className="text-xs text-slate-500">Select an observation plot across Maharashtra</p>
          </div>
        </div>

        {/* Dropdown selector */}
        <div className="relative min-w-[240px]">
          <select
            value={selectedPointId}
            onChange={(e) => onSelect(e.target.value)}
            disabled={isLoading}
            className="w-full bg-slate-50 hover:bg-slate-100 text-slate-900 font-semibold text-sm rounded-xl px-3 py-2 border border-slate-300 focus:outline-none focus:ring-2 focus:ring-agri-500 transition-colors cursor-pointer"
          >
            {locations.map((loc) => {
              const extra = loc.case_study_group ? ` (${loc.case_study_group})` : '';
              return (
                <option key={loc.point_id} value={loc.point_id}>
                  {loc.point_id} — {loc.district}
                  {extra}
                </option>
              );
            })}
          </select>
        </div>
      </div>

      {/* Quick Access Badges for Key Demo Case Studies */}
      <div className="pt-2 border-t border-slate-100 flex items-center flex-wrap gap-2 text-xs">
        <span className="text-slate-400 font-medium">Quick Select:</span>
        {featured.map((pid) => {
          const isSelected = selectedPointId === pid;
          const label = pid === 'P069'
            ? 'P069 (High Stress)'
            : pid === 'P077'
            ? 'P077 (Watch)'
            : pid === 'P065'
            ? 'P065 (Stable)'
            : 'P147 (Kavathe/Satara)';

          return (
            <button
              key={pid}
              onClick={() => onSelect(pid)}
              className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-lg font-semibold transition-all ${
                isSelected
                  ? 'bg-agri-700 text-white shadow-sm'
                  : 'bg-slate-100 hover:bg-slate-200 text-slate-700'
              }`}
            >
              {isSelected && <Check className="w-3 h-3" />}
              <span>{label}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
};
