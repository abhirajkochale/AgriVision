export interface HealthResponse {
  status: string;
  service: string;
  timestamp: string;
  version: string;
}

export interface MonitoringCoverage {
  total_monitored_points: number;
  latur_points: number;
  kavathe_khanapur_points: number;
}

export interface LocationDetailResponse {
  point_id: string;
  district: string;
  latitude: number;
  longitude: number;
  case_study_group?: string | null;
  current_ndvi?: number | null;
  current_gsmap_rainfall_mm?: number | null;
  current_temperature_c?: number | null;
  latest_ndvi_week?: string | null;
  latest_rainfall_week?: string | null;
  latest_temperature_week?: string | null;
  common_latest_week?: string | null;
  data_age_days?: number | null;
  freshness_status: string;
  prediction_eligibility: string;
  current_ndvi_available: boolean;
  current_gsmap_available: boolean;
  current_temperature_available: boolean;
  current_inputs_complete: boolean;
  prediction_available: boolean;
  predicted_ndvi_next_week?: number | null;
  predicted_ndvi_change?: number | null;
  predicted_ndvi_change_percent?: number | null;
  prediction_status: string;
  operational_trend_status: string;
  reason?: string | null;
}

export interface WeeklyObservation {
  Week_Start: string;
  Week_End: string;
  Week_Number: number;
  Year: number;
  NDVI: number | null;
  Rainfall_mm: number | null;
  Temperature_C: number | null;
}

export interface RecommendationResponse {
  point_id: string;
  status: string;
  severity: 'high' | 'medium' | 'low' | 'positive' | 'unknown';
  current_ndvi?: number | null;
  predicted_ndvi_next_week?: number | null;
  predicted_ndvi_change?: number | null;
  predicted_ndvi_change_percent?: number | null;
  headline: string;
  explanation: string;
  recommended_action: string;
  data_confidence: 'HIGH' | 'MEDIUM' | 'LOW';
  prediction_available: boolean;
}

export interface DashboardLocationInfo {
  point_id: string;
  district: string;
  latitude: number;
  longitude: number;
  case_study_group?: string | null;
}

export interface DashboardCurrentConditions {
  current_ndvi?: number | null;
  current_gsmap_rainfall_mm?: number | null;
  current_temperature_c?: number | null;
  latest_ndvi_week?: string | null;
  latest_rainfall_week?: string | null;
  latest_temperature_week?: string | null;
  common_latest_week?: string | null;
  data_age_days?: number | null;
  freshness_status: string;
  current_ndvi_available: boolean;
  current_gsmap_available: boolean;
  current_temperature_available: boolean;
  current_inputs_complete: boolean;
}

export interface DashboardPrediction {
  prediction_available: boolean;
  predicted_ndvi_next_week?: number | null;
  predicted_ndvi_change?: number | null;
  predicted_ndvi_change_percent?: number | null;
  prediction_status: string;
  operational_trend_status: string;
  reason?: string | null;
}

export interface DashboardRecommendation {
  status: string;
  severity: 'high' | 'medium' | 'low' | 'positive' | 'unknown';
  headline: string;
  explanation: string;
  recommended_action: string;
  data_confidence: 'HIGH' | 'MEDIUM' | 'LOW';
}

export interface LocationDashboardResponse {
  location: DashboardLocationInfo;
  current_conditions: DashboardCurrentConditions;
  prediction: DashboardPrediction;
  recommendation: DashboardRecommendation;
}

export interface MapMarkerLocation {
  point_id: string;
  district: string;
  latitude: number;
  longitude: number;
  operational_trend_status: string;
  prediction_status: string;
  prediction_available: boolean;
  current_ndvi?: number | null;
  predicted_ndvi_next_week?: number | null;
  predicted_ndvi_change?: number | null;
}
