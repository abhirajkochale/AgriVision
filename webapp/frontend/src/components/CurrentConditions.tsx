import React from 'react';
import { CloudRain, Thermometer, Leaf, Clock, CheckCircle2, XCircle } from 'lucide-react';
import { DashboardCurrentConditions } from '../types/api';

interface CurrentConditionsProps {
  conditions: DashboardCurrentConditions;
}

export const CurrentConditions: React.FC<CurrentConditionsProps> = ({ conditions }) => {
  const formatNdvi = (val?: number | null) => {
    if (val === null || val === undefined) {
      return { text: 'No Reading', sub: 'Cloud obscured' };
    }
    const num = val.toFixed(2);
    let desc = 'Healthy canopy';
    if (val < 0.3) desc = 'Sparse / Bare ground';
    else if (val < 0.6) desc = 'Moderate vegetation';
    return { text: num, sub: desc };
  };

  const ndviInfo = formatNdvi(conditions.current_ndvi);

  return (
    <div className="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-base font-bold text-slate-900">Current Field Conditions</h3>
          <p className="text-xs text-slate-500">Real-time satellite & weather observations</p>
        </div>
        <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-slate-100 text-slate-600">
          <Clock className="w-3.5 h-3.5" />
          <span>Freshness: {conditions.freshness_status}</span>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3.5">
        {/* NDVI Card */}
        <div className="bg-slate-50/80 rounded-xl p-3.5 border border-slate-200/80 flex flex-col justify-between">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Vegetation Index (NDVI)</span>
            <div className="w-7 h-7 rounded-lg bg-emerald-100 text-emerald-700 flex items-center justify-center">
              <Leaf className="w-4 h-4" />
            </div>
          </div>
          <div>
            <div className="text-2xl font-black text-slate-900 tracking-tight">
              {ndviInfo.text}
            </div>
            <p className="text-xs font-medium text-slate-600 mt-0.5">{ndviInfo.sub}</p>
          </div>
          <div className="mt-3 pt-2.5 border-t border-slate-200/60 flex items-center justify-between text-[11px] text-slate-500">
            <span>Sentinel-2</span>
            {conditions.current_ndvi_available ? (
              <span className="inline-flex items-center gap-1 text-emerald-700 font-medium">
                <CheckCircle2 className="w-3 h-3" /> Available
              </span>
            ) : (
              <span className="inline-flex items-center gap-1 text-slate-400">
                <XCircle className="w-3 h-3" /> Missing
              </span>
            )}
          </div>
        </div>

        {/* GSMaP Rainfall Card */}
        <div className="bg-slate-50/80 rounded-xl p-3.5 border border-slate-200/80 flex flex-col justify-between">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Weekly Rainfall</span>
            <div className="w-7 h-7 rounded-lg bg-sky-100 text-sky-700 flex items-center justify-center">
              <CloudRain className="w-4 h-4" />
            </div>
          </div>
          <div>
            <div className="text-2xl font-black text-slate-900 tracking-tight">
              {conditions.current_gsmap_rainfall_mm !== null && conditions.current_gsmap_rainfall_mm !== undefined
                ? `${conditions.current_gsmap_rainfall_mm.toFixed(1)} mm`
                : 'N/A'}
            </div>
            <p className="text-xs font-medium text-slate-600 mt-0.5">7-day accumulated</p>
          </div>
          <div className="mt-3 pt-2.5 border-t border-slate-200/60 flex items-center justify-between text-[11px] text-slate-500">
            <span>GSMaP Satellite</span>
            {conditions.current_gsmap_available ? (
              <span className="inline-flex items-center gap-1 text-emerald-700 font-medium">
                <CheckCircle2 className="w-3 h-3" /> Available
              </span>
            ) : (
              <span className="inline-flex items-center gap-1 text-slate-400">
                <XCircle className="w-3 h-3" /> Missing
              </span>
            )}
          </div>
        </div>

        {/* Temperature Card */}
        <div className="bg-slate-50/80 rounded-xl p-3.5 border border-slate-200/80 flex flex-col justify-between">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Mean Temperature</span>
            <div className="w-7 h-7 rounded-lg bg-orange-100 text-orange-700 flex items-center justify-center">
              <Thermometer className="w-4 h-4" />
            </div>
          </div>
          <div>
            <div className="text-2xl font-black text-slate-900 tracking-tight">
              {conditions.current_temperature_c !== null && conditions.current_temperature_c !== undefined
                ? `${conditions.current_temperature_c.toFixed(1)} °C`
                : 'N/A'}
            </div>
            <p className="text-xs font-medium text-slate-600 mt-0.5">Weekly field average</p>
          </div>
          <div className="mt-3 pt-2.5 border-t border-slate-200/60 flex items-center justify-between text-[11px] text-slate-500">
            <span>ERA5-Land</span>
            {conditions.current_temperature_available ? (
              <span className="inline-flex items-center gap-1 text-emerald-700 font-medium">
                <CheckCircle2 className="w-3 h-3" /> Available
              </span>
            ) : (
              <span className="inline-flex items-center gap-1 text-slate-400">
                <XCircle className="w-3 h-3" /> Missing
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
