import React from 'react';
import { Map, Layers } from 'lucide-react';
import { DashboardLocationInfo } from '../types/api';

interface MapPlaceholderProps {
  location: DashboardLocationInfo;
}

export const MapPlaceholder: React.FC<MapPlaceholderProps> = ({ location }) => {
  return (
    <div className="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-blue-100 text-blue-800 flex items-center justify-center">
            <Map className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-base font-bold text-slate-900">Geographic Field Map</h3>
            <p className="text-xs text-slate-500">Spatial plot localization</p>
          </div>
        </div>
        <span className="text-[11px] font-semibold text-blue-700 bg-blue-50 border border-blue-200 px-2.5 py-1 rounded-full flex items-center gap-1">
          <Layers className="w-3 h-3" /> Step 5 Feature
        </span>
      </div>

      <div className="bg-slate-50 rounded-xl p-5 border border-dashed border-slate-200 text-center flex flex-col items-center justify-center py-8">
        <div className="w-12 h-12 rounded-full bg-slate-200/80 flex items-center justify-center text-slate-600 mb-3">
          <Map className="w-6 h-6" />
        </div>
        <h4 className="text-sm font-bold text-slate-800">
          Plot {location.point_id} · {location.district}
          {location.case_study_group ? ` (${location.case_study_group})` : ''}
        </h4>
        <p className="text-xs text-slate-500 font-mono mt-1">
          GPS: {location.latitude.toFixed(4)}° N, {location.longitude.toFixed(4)}° E
        </p>
        <p className="text-xs text-slate-400 mt-2 max-w-sm">
          Interactive Maharashtra satellite map with visual marker clustering will be activated in Step 5.
        </p>
      </div>
    </div>
  );
};
