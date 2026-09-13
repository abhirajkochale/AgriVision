import React from 'react';
import { TrendingDown, TrendingUp, Minus, AlertCircle } from 'lucide-react';
import { DashboardPrediction } from '../types/api';
import { StatusBadge } from './StatusBadge';

interface ForecastCardProps {
  prediction: DashboardPrediction;
}

export const ForecastCard: React.FC<ForecastCardProps> = ({ prediction }) => {
  const isAvail = prediction.prediction_available && prediction.predicted_ndvi_next_week !== null;
  const change = prediction.predicted_ndvi_change;
  const changePct = prediction.predicted_ndvi_change_percent;

  const isDecline = change !== null && change !== undefined && change < -0.01;
  const isIncrease = change !== null && change !== undefined && change > 0.01;

  return (
    <div className="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm flex flex-col justify-between">
      <div>
        <div className="flex items-center justify-between mb-3">
          <div>
            <h3 className="text-base font-bold text-slate-900">Next-Week Forecast</h3>
            <p className="text-xs text-slate-500">Predicted crop canopy trajectory</p>
          </div>
          <StatusBadge status={prediction.operational_trend_status} size="sm" />
        </div>

        {isAvail ? (
          <div className="bg-slate-50 rounded-xl p-4 border border-slate-100 mt-2">
            <div className="flex items-baseline justify-between">
              <div>
                <span className="text-xs font-semibold text-slate-500 block uppercase tracking-wider">
                  Forecasted NDVI
                </span>
                <span className="text-3xl font-black text-slate-900 tracking-tight">
                  {prediction.predicted_ndvi_next_week?.toFixed(2)}
                </span>
              </div>

              {/* Change Pill */}
              <div className="text-right">
                <span className="text-xs font-semibold text-slate-500 block uppercase tracking-wider">
                  Predicted Change
                </span>
                <div
                  className={`inline-flex items-center gap-1 font-bold text-sm px-2.5 py-1 rounded-lg ${
                    isDecline
                      ? 'bg-amber-100 text-amber-900'
                      : isIncrease
                      ? 'bg-emerald-100 text-emerald-900'
                      : 'bg-slate-200 text-slate-800'
                  }`}
                >
                  {isDecline && <TrendingDown className="w-4 h-4 text-amber-700" />}
                  {isIncrease && <TrendingUp className="w-4 h-4 text-emerald-700" />}
                  {!isDecline && !isIncrease && <Minus className="w-4 h-4 text-slate-600" />}
                  <span>
                    {change !== null && change !== undefined ? (change > 0 ? `+${change.toFixed(2)}` : change.toFixed(2)) : '0.00'}
                    {changePct !== null && changePct !== undefined ? ` (${changePct > 0 ? `+${changePct.toFixed(1)}` : `${changePct.toFixed(1)}`}%)` : ''}
                  </span>
                </div>
              </div>
            </div>

            <p className="text-xs text-slate-600 mt-3 pt-2.5 border-t border-slate-200/60">
              Operational status: <strong className="font-semibold text-slate-900">{prediction.prediction_status}</strong>
            </p>
          </div>
        ) : (
          <div className="bg-slate-50 rounded-xl p-4 border border-slate-200/80 text-center mt-2 flex flex-col items-center justify-center py-6">
            <div className="w-10 h-10 rounded-full bg-slate-200/80 flex items-center justify-center text-slate-500 mb-2">
              <AlertCircle className="w-5 h-5" />
            </div>
            <h4 className="text-sm font-bold text-slate-800">Forecast Unavailable</h4>
            <p className="text-xs text-slate-500 max-w-xs mt-1">
              {prediction.reason || 'Input observations are too old or incomplete for next-week forecast.'}
            </p>
          </div>
        )}
      </div>

      <div className="mt-4 pt-3 border-t border-slate-100 text-[11px] text-slate-500 flex items-center justify-between">
        <span>7-day prediction window</span>
        <span>Decision support signal</span>
      </div>
    </div>
  );
};
