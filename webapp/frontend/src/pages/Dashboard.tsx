import React, { useEffect, useState, useCallback } from 'react';
import { RefreshCw, AlertCircle } from 'lucide-react';
import { api } from '../services/api';
import {
  LocationDashboardResponse,
  LocationDetailResponse,
  WeeklyObservation,
} from '../types/api';
import { Header } from '../components/Header';
import { LocationSelector } from '../components/LocationSelector';
import { CurrentConditions } from '../components/CurrentConditions';
import { ForecastCard } from '../components/ForecastCard';
import { RecommendationCard } from '../components/RecommendationCard';
import { HistoryChart } from '../components/HistoryChart';
import { MonitoringMap } from '../components/MonitoringMap';

export const Dashboard: React.FC = () => {
  const [locations, setLocations] = useState<LocationDetailResponse[]>([]);
  const [selectedPointId, setSelectedPointId] = useState<string>('P069');
  const [dashboardData, setDashboardData] = useState<LocationDashboardResponse | null>(null);
  const [historyData, setHistoryData] = useState<WeeklyObservation[]>([]);

  const [isLoadingLocations, setIsLoadingLocations] = useState<boolean>(true);
  const [isLoadingData, setIsLoadingData] = useState<boolean>(true);
  const [isOnline, setIsOnline] = useState<boolean>(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // 1. Initial health check and locations load
  useEffect(() => {
    async function loadInitial() {
      try {
        setIsLoadingLocations(true);
        const [health, locs] = await Promise.all([
          api.checkHealth().catch(() => null),
          api.getLocations(),
        ]);

        setIsOnline(Boolean(health && health.status === 'ok'));
        setLocations(locs);

        // Ensure default is valid in loaded locations
        if (locs.length > 0 && !locs.some((l) => l.point_id === 'P069')) {
          setSelectedPointId(locs[0].point_id);
        }
      } catch (err: unknown) {
        setIsOnline(false);
        setErrorMessage(err instanceof Error ? err.message : 'Failed to connect to backend service.');
      } finally {
        setIsLoadingLocations(false);
      }
    }

    loadInitial();
  }, []);

  // 2. Fetch point dashboard and history on selection change
  const loadPointData = useCallback(async (pointId: string) => {
    try {
      setIsLoadingData(true);
      setErrorMessage(null);

      const [dash, hist] = await Promise.all([
        api.getLocationDashboard(pointId),
        api.getLocationHistory(pointId),
      ]);

      setDashboardData(dash);
      setHistoryData(hist);
      setIsOnline(true);
    } catch (err: unknown) {
      setErrorMessage(err instanceof Error ? err.message : `Failed loading data for ${pointId}`);
    } finally {
      setIsLoadingData(false);
    }
  }, []);

  useEffect(() => {
    if (selectedPointId) {
      loadPointData(selectedPointId);
    }
  }, [selectedPointId, loadPointData]);

  const handleSelectPoint = (pointId: string) => {
    setSelectedPointId(pointId);
  };

  const handleRetry = () => {
    loadPointData(selectedPointId);
  };

  return (
    <div className="min-h-screen bg-[#f8fafc] text-slate-800 flex flex-col">
      {/* 1. Header */}
      <Header
        monitoringPeriod={dashboardData?.current_conditions.latest_ndvi_week ? `Week of ${dashboardData.current_conditions.latest_ndvi_week}` : 'Operational Week 2026-09-03'}
        isOnline={isOnline}
      />

      {/* Main Container */}
      <main className="flex-1 max-w-6xl w-full mx-auto px-4 sm:px-6 py-5 sm:py-7 space-y-5">
        {/* 2. Location Selector */}
        <LocationSelector
          locations={locations}
          selectedPointId={selectedPointId}
          onSelect={handleSelectPoint}
          isLoading={isLoadingData || isLoadingLocations}
        />

        {/* Error Notification */}
        {errorMessage && (
          <div className="bg-rose-50 border border-rose-200 rounded-2xl p-4 text-rose-800 flex items-start gap-3">
            <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5 text-rose-600" />
            <div className="flex-1 text-sm">
              <h4 className="font-bold">Backend Connection Notice</h4>
              <p className="mt-0.5 text-rose-700">{errorMessage}</p>
            </div>
            <button
              onClick={handleRetry}
              className="px-3 py-1 bg-white hover:bg-rose-100 text-rose-800 border border-rose-300 rounded-lg text-xs font-bold transition-colors"
            >
              Retry
            </button>
          </div>
        )}

        {/* Loading Skeleton */}
        {isLoadingData && !dashboardData && (
          <div className="bg-white rounded-2xl p-12 border border-slate-200 text-center flex flex-col items-center justify-center space-y-3">
            <RefreshCw className="w-8 h-8 text-agri-600 animate-spin" />
            <p className="text-sm font-semibold text-slate-600">Retrieving operational field data for {selectedPointId}...</p>
          </div>
        )}

        {/* 3. Dashboard Content */}
        {dashboardData && (
          <div className="space-y-5">
            {/* Top Grid: Conditions & Forecast */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
              <CurrentConditions conditions={dashboardData.current_conditions} />
              <ForecastCard prediction={dashboardData.prediction} />
            </div>

            {/* Middle: Farmer Recommendation Card */}
            <RecommendationCard recommendation={dashboardData.recommendation} />

            {/* Bottom Grid: Historical NDVI Chart & Interactive Map */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
              <div className="lg:col-span-2">
                <HistoryChart
                  history={historyData}
                  predictedNdvi={dashboardData.prediction.predicted_ndvi_next_week}
                />
              </div>
              <div className="lg:col-span-1">
                <MonitoringMap
                  selectedPointId={selectedPointId}
                  onSelectPoint={handleSelectPoint}
                />
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-slate-200 mt-auto py-4 text-center text-xs text-slate-400">
        <div className="max-w-6xl mx-auto px-4">
          AgriVision AI · Satellite-Driven Vegetation Intelligence · Built for Maharashtra Farmers
        </div>
      </footer>
    </div>
  );
};
