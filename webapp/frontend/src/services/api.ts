import {
  HealthResponse,
  LocationDashboardResponse,
  LocationDetailResponse,
  MapMarkerLocation,
  RecommendationResponse,
  WeeklyObservation,
} from '../types/api';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorDetail = `Request failed with status ${response.status}`;
    try {
      const errJson = await response.json();
      errorDetail = errJson.detail || errJson.message || errorDetail;
    } catch {
      // ignore JSON parse error
    }
    throw new Error(errorDetail);
  }
  return response.json() as Promise<T>;
}

export const api = {
  async checkHealth(): Promise<HealthResponse> {
    const res = await fetch(`${API_BASE_URL}/api/health`);
    return handleResponse<HealthResponse>(res);
  },

  async getLocations(): Promise<LocationDetailResponse[]> {
    const res = await fetch(`${API_BASE_URL}/api/locations`);
    return handleResponse<LocationDetailResponse[]>(res);
  },

  async getLocation(pointId: string): Promise<LocationDetailResponse> {
    const res = await fetch(`${API_BASE_URL}/api/locations/${encodeURIComponent(pointId)}`);
    return handleResponse<LocationDetailResponse>(res);
  },

  async getLocationHistory(pointId: string): Promise<WeeklyObservation[]> {
    const res = await fetch(`${API_BASE_URL}/api/locations/${encodeURIComponent(pointId)}/history`);
    return handleResponse<WeeklyObservation[]>(res);
  },

  async getLocationRecommendation(pointId: string): Promise<RecommendationResponse> {
    const res = await fetch(`${API_BASE_URL}/api/locations/${encodeURIComponent(pointId)}/recommendation`);
    return handleResponse<RecommendationResponse>(res);
  },

  async getLocationDashboard(pointId: string): Promise<LocationDashboardResponse> {
    const res = await fetch(`${API_BASE_URL}/api/locations/${encodeURIComponent(pointId)}/dashboard`);
    return handleResponse<LocationDashboardResponse>(res);
  },

  async getMapLocations(): Promise<MapMarkerLocation[]> {
    const res = await fetch(`${API_BASE_URL}/api/map`);
    return handleResponse<MapMarkerLocation[]>(res);
  },
};
