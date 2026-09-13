import React, { useEffect, useState, useMemo } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import { MapPin, RefreshCw, AlertCircle } from 'lucide-react';
import { api } from '../services/api';
import { MapMarkerLocation } from '../types/api';
import { StatusBadge } from './StatusBadge';

interface MonitoringMapProps {
  selectedPointId: string;
  onSelectPoint: (pointId: string) => void;
}

// Color and style mapper matching StatusBadge rules
function getMarkerTheme(status: string) {
  const norm = status.trim().toUpperCase();
  if (norm.includes('HIGH_STRESS') || norm.includes('HIGH STRESS')) {
    return {
      fill: '#dc2626',
      border: '#991b1b',
      text: '#ffffff',
      pulse: true,
      label: 'High Stress Risk',
    };
  }
  if (norm.includes('WATCH')) {
    return {
      fill: '#d97706',
      border: '#92400e',
      text: '#ffffff',
      pulse: false,
      label: 'Watch',
    };
  }
  if (norm.includes('STABLE')) {
    return {
      fill: '#16a34a',
      border: '#15803d',
      text: '#ffffff',
      pulse: false,
      label: 'Stable',
    };
  }
  if (norm.includes('IMPROVING')) {
    return {
      fill: '#0d9488',
      border: '#115e59',
      text: '#ffffff',
      pulse: false,
      label: 'Improving',
    };
  }
  return {
    fill: '#64748b',
    border: '#475569',
    text: '#ffffff',
    pulse: false,
    label: 'Prediction Unavailable',
  };
}

// Create custom SVG Leaflet divIcon
function createDivIcon(pointId: string, status: string, isSelected: boolean): L.DivIcon {
  const theme = getMarkerTheme(status);
  const size = isSelected ? 34 : 26;
  const pulseHtml = (theme.pulse || isSelected)
    ? `<span style="position: absolute; inset: -4px; border-radius: 9999px; background-color: ${theme.fill}; opacity: 0.35; animation: ping 2s cubic-bezier(0, 0, 0.2, 1) infinite;"></span>`
    : '';

  const html = `
    <div class="agri-marker-pin ${isSelected ? 'is-selected' : ''}" style="position: relative; width: ${size}px; height: ${size}px; background-color: ${theme.fill}; border: 2px solid ${isSelected ? '#ffffff' : theme.border}; color: ${theme.text}; font-size: ${isSelected ? '11px' : '9px'}; font-weight: 800;">
      ${pulseHtml}
      <span style="position: relative; z-index: 2; text-shadow: 0 1px 2px rgba(0,0,0,0.4);">${pointId.replace('P', '')}</span>
    </div>
  `;

  return L.divIcon({
    html,
    className: 'agri-marker-wrapper',
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
    popupAnchor: [0, -size / 2 - 4],
  });
}

// MapController manages initial fitBounds and subsequent flyTo upon selection change
interface MapControllerProps {
  markers: MapMarkerLocation[];
  selectedPointId: string;
  initialFitDone: boolean;
  setInitialFitDone: (done: boolean) => void;
}

const MapController: React.FC<MapControllerProps> = ({
  markers,
  selectedPointId,
  initialFitDone,
  setInitialFitDone,
}) => {
  const map = useMap();

  // Fit bounds to all markers only once on initial data load
  useEffect(() => {
    if (markers.length > 0 && !initialFitDone) {
      const bounds = L.latLngBounds(markers.map((m) => [m.latitude, m.longitude]));
      map.fitBounds(bounds, { padding: [40, 40], maxZoom: 11 });
      setInitialFitDone(true);
    }
  }, [markers, initialFitDone, map, setInitialFitDone]);

  // When selectedPointId changes and initial fit is done, fly/center to that marker
  useEffect(() => {
    if (initialFitDone && selectedPointId) {
      const target = markers.find((m) => m.point_id === selectedPointId);
      if (target) {
        map.flyTo([target.latitude, target.longitude], Math.max(map.getZoom(), 10), {
          duration: 0.8,
        });
      }
    }
  }, [selectedPointId, initialFitDone, markers, map]);

  return null;
};

export const MonitoringMap: React.FC<MonitoringMapProps> = ({
  selectedPointId,
  onSelectPoint,
}) => {
  const [markers, setMarkers] = useState<MapMarkerLocation[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [initialFitDone, setInitialFitDone] = useState<boolean>(false);

  // Load map markers from /api/map
  useEffect(() => {
    async function loadMarkers() {
      try {
        setIsLoading(true);
        setError(null);
        const data = await api.getMapLocations();
        setMarkers(data);
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : 'Failed loading map points.');
      } finally {
        setIsLoading(false);
      }
    }
    loadMarkers();
  }, []);

  // Default center roughly covering Maharashtra (Latur & Satara)
  const defaultCenter: [number, number] = useMemo(() => [18.25, 76.5], []);

  return (
    <div className="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm flex flex-col justify-between">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-agri-100 text-agri-800 flex items-center justify-center flex-shrink-0">
            <MapPin className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-base font-bold text-slate-900">Regional Monitoring Map</h3>
            <p className="text-xs text-slate-500">26 monitored field locations across Maharashtra</p>
          </div>
        </div>

        <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-slate-100 text-slate-700">
          {markers.length} Field Plots
        </span>
      </div>

      {/* Map Body Container */}
      <div className="relative w-full h-[360px] sm:h-[420px] rounded-xl overflow-hidden border border-slate-200">
        {isLoading && (
          <div className="absolute inset-0 bg-slate-50/90 backdrop-blur-xs flex flex-col items-center justify-center z-30 text-slate-600 gap-2">
            <RefreshCw className="w-6 h-6 animate-spin text-agri-600" />
            <span className="text-xs font-semibold">Loading Maharashtra plots...</span>
          </div>
        )}

        {error && (
          <div className="absolute inset-0 bg-rose-50/90 flex flex-col items-center justify-center z-30 p-4 text-center">
            <AlertCircle className="w-6 h-6 text-rose-600 mb-1" />
            <p className="text-xs font-bold text-rose-900">Failed loading map data</p>
            <p className="text-[11px] text-rose-700 max-w-xs mt-0.5">{error}</p>
          </div>
        )}

        <MapContainer
          center={defaultCenter}
          zoom={8}
          scrollWheelZoom={false}
          className="w-full h-full"
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />

          <MapController
            markers={markers}
            selectedPointId={selectedPointId}
            initialFitDone={initialFitDone}
            setInitialFitDone={setInitialFitDone}
          />

          {markers.map((pt) => {
            const isSelected = pt.point_id === selectedPointId;
            const icon = createDivIcon(pt.point_id, pt.operational_trend_status, isSelected);

            return (
              <Marker
                key={pt.point_id}
                position={[pt.latitude, pt.longitude]}
                icon={icon}
                eventHandlers={{
                  click: () => onSelectPoint(pt.point_id),
                }}
              >
                <Popup>
                  <div className="p-3.5 min-w-[200px] text-slate-800">
                    <div className="flex items-center justify-between border-b border-slate-100 pb-2 mb-2">
                      <h4 className="font-extrabold text-sm text-slate-900">
                        {pt.point_id} · {pt.district}
                      </h4>
                      <StatusBadge status={pt.operational_trend_status} size="sm" />
                    </div>

                    <div className="space-y-1.5 text-xs">
                      <div className="flex justify-between">
                        <span className="text-slate-500">Current NDVI:</span>
                        <span className="font-bold text-slate-900">
                          {pt.current_ndvi !== null && pt.current_ndvi !== undefined
                            ? pt.current_ndvi.toFixed(2)
                            : 'Missing (Cloud)'}
                        </span>
                      </div>

                      <div className="flex justify-between">
                        <span className="text-slate-500">Next-Week Forecast:</span>
                        <span className="font-bold text-slate-900">
                          {pt.prediction_available && pt.predicted_ndvi_next_week !== null && pt.predicted_ndvi_next_week !== undefined
                            ? pt.predicted_ndvi_next_week.toFixed(2)
                            : 'Unavailable'}
                        </span>
                      </div>

                      {pt.predicted_ndvi_change !== null && pt.predicted_ndvi_change !== undefined && (
                        <div className="flex justify-between">
                          <span className="text-slate-500">Predicted Change:</span>
                          <span
                            className={`font-bold ${
                              pt.predicted_ndvi_change < 0 ? 'text-amber-700' : 'text-emerald-700'
                            }`}
                          >
                            {pt.predicted_ndvi_change > 0
                              ? `+${pt.predicted_ndvi_change.toFixed(2)}`
                              : pt.predicted_ndvi_change.toFixed(2)}
                          </span>
                        </div>
                      )}
                    </div>

                    <button
                      onClick={() => onSelectPoint(pt.point_id)}
                      className="mt-3 w-full py-1.5 bg-agri-700 hover:bg-agri-800 text-white rounded-lg text-xs font-bold transition-colors shadow-xs"
                    >
                      {isSelected ? 'Currently Viewing' : 'Select This Field'}
                    </button>
                  </div>
                </Popup>
              </Marker>
            );
          })}
        </MapContainer>
      </div>

      {/* Compact Status Color Legend */}
      <div className="mt-3 pt-3 border-t border-slate-100 flex flex-wrap items-center justify-between gap-2 text-[11px] text-slate-600">
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-emerald-600"></span>
          <span>Stable</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-amber-600"></span>
          <span>Watch</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-rose-600"></span>
          <span>High Stress Risk</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-slate-500"></span>
          <span>Unavailable / Stale</span>
        </div>
      </div>
    </div>
  );
};
