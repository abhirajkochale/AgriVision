import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { History, CloudOff } from 'lucide-react';
import { WeeklyObservation } from '../types/api';

interface HistoryChartProps {
  history: WeeklyObservation[];
  predictedNdvi?: number | null;
}

export const HistoryChart: React.FC<HistoryChartProps> = ({ history, predictedNdvi }) => {
  // Format chart data
  const data = history.map((item) => {
    // Format "2026-06-18" to "W25 (Jun 18)"
    const dateParts = item.Week_Start.split('-');
    const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const month = monthNames[parseInt(dateParts[1], 10) - 1] || '';
    const day = parseInt(dateParts[2], 10);
    const shortLabel = `${month} ${day}`;

    return {
      date: item.Week_Start,
      label: shortLabel,
      weekNumber: item.Week_Number,
      ndvi: item.NDVI !== null ? parseFloat(item.NDVI.toFixed(3)) : null,
      rainfall: item.Rainfall_mm !== null ? parseFloat(item.Rainfall_mm.toFixed(1)) : null,
      temp: item.Temperature_C !== null ? parseFloat(item.Temperature_C.toFixed(1)) : null,
    };
  });

  const validObservations = data.filter((d) => d.ndvi !== null).length;
  const missingObservations = data.length - validObservations;

  return (
    <div className="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-emerald-100 text-emerald-800 flex items-center justify-center flex-shrink-0">
            <History className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-base font-bold text-slate-900">12-Week Vegetation Health History</h3>
            <p className="text-xs text-slate-500">Sentinel-2 satellite NDVI trajectory</p>
          </div>
        </div>

        <div className="flex items-center gap-3 text-xs text-slate-500">
          <span className="inline-flex items-center gap-1.5 font-medium">
            <span className="w-2.5 h-2.5 rounded-full bg-agri-600"></span>
            Valid Observation ({validObservations})
          </span>
          {missingObservations > 0 && (
            <span className="inline-flex items-center gap-1.5 font-medium text-slate-400">
              <CloudOff className="w-3.5 h-3.5" />
              Cloud Gaps ({missingObservations})
            </span>
          )}
        </div>
      </div>

      <div className="h-64 sm:h-72 w-full pt-2">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 10, right: 15, left: -20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
            <XAxis
              dataKey="label"
              tick={{ fontSize: 11, fill: '#64748b' }}
              axisLine={{ stroke: '#cbd5e1' }}
              tickLine={false}
            />
            <YAxis
              domain={[0, 1]}
              ticks={[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]}
              tick={{ fontSize: 11, fill: '#64748b' }}
              axisLine={{ stroke: '#cbd5e1' }}
              tickLine={false}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const p = payload[0].payload;
                  const isNull = p.ndvi === null;
                  return (
                    <div className="bg-slate-900 text-white text-xs rounded-xl p-3 shadow-xl border border-slate-700">
                      <p className="font-bold text-slate-300 border-b border-slate-700/80 pb-1 mb-1.5">
                        Week {p.weekNumber} · {p.date}
                      </p>
                      {isNull ? (
                        <p className="text-amber-300 flex items-center gap-1 font-medium">
                          <CloudOff className="w-3.5 h-3.5" /> Cloud cover / Missing NDVI
                        </p>
                      ) : (
                        <p className="text-emerald-400 font-bold text-sm">
                          NDVI: {p.ndvi}
                        </p>
                      )}
                      {p.rainfall !== null && (
                        <p className="text-slate-300 mt-1">Rainfall: {p.rainfall} mm</p>
                      )}
                      {p.temp !== null && (
                        <p className="text-slate-300">Temp: {p.temp} °C</p>
                      )}
                    </div>
                  );
                }
                return null;
              }}
            />
            {predictedNdvi !== null && predictedNdvi !== undefined && (
              <ReferenceLine
                y={predictedNdvi}
                stroke="#d97706"
                strokeDasharray="4 4"
                label={{
                  value: `Forecast: ${predictedNdvi.toFixed(2)}`,
                  fill: '#b45309',
                  fontSize: 10,
                  position: 'right',
                }}
              />
            )}
            <Line
              type="monotone"
              dataKey="ndvi"
              stroke="#2a6148"
              strokeWidth={2.5}
              connectNulls={false}
              dot={{ r: 4, fill: '#2a6148', strokeWidth: 1.5, stroke: '#fff' }}
              activeDot={{ r: 6, fill: '#1d4132', stroke: '#fff', strokeWidth: 2 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-3 pt-2 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-500">
        <span>Gaps in line indicate cloudy weeks without satellite visibility. Missing values are preserved.</span>
        <span className="hidden sm:inline">Range: 0.0 (Bare Ground) - 1.0 (Dense Green Canopy)</span>
      </div>
    </div>
  );
};
