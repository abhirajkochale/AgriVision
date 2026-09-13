from datetime import datetime
from typing import Dict, Optional, Union
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="Backend operational status")
    service: str = Field(default="AgriVision AI Backend", description="Service identifier")
    timestamp: datetime = Field(..., description="Current UTC timestamp")
    version: str = Field(default="1.0.0", description="Backend service version")


class MonitoringCoverage(BaseModel):
    total_monitored_points: int = Field(..., description="Total points actively monitored across regions")
    latur_points: int = Field(..., description="Monitored points located in Latur district")
    kavathe_khanapur_points: int = Field(..., description="Monitored points located in Kavathe-Khanapur")


class DataAvailability(BaseModel):
    current_ndvi_available_points: int = Field(..., description="Points with current Sentinel-2 NDVI observation")
    current_gsmap_available_points: int = Field(..., description="Points with current GSMaP operational rainfall observation")
    current_temperature_available_points: int = Field(..., description="Points with current ERA5-Land temperature observation")
    complete_current_input_points: int = Field(..., description="Points where all current inputs (NDVI, rainfall, temp) are present")


class PredictionStatus(BaseModel):
    prediction_ready_points: int = Field(..., description="Points fulfilling all input and freshness criteria for inference")
    predictions_generated: int = Field(..., description="Predictions successfully produced by ExtraTrees model")
    no_prediction_points: int = Field(..., description="Points without predictions due to data gaps or freshness limits")


class FreshnessDistribution(BaseModel):
    ready: int = Field(..., description="Observations within 0-14 days age")
    warning: int = Field(..., description="Observations within 15-28 days age")
    stale: int = Field(..., description="Observations older than 28 days")
    very_stale: int = Field(..., description="Observations categorized as very stale")


class OperationalTrendDistribution(BaseModel):
    high_stress_risk: int = Field(..., description="Points predicted with high vegetation stress risk")
    watch: int = Field(..., description="Points categorized under advisory watch status")
    stable: int = Field(..., description="Points predicted with stable vegetation health")
    improving: int = Field(..., description="Points predicted with improving vegetation health")
    no_prediction: int = Field(..., description="Points without operational trend prediction")


class PredictionStatistics(BaseModel):
    predicted_ndvi_minimum: Optional[float] = Field(None, description="Minimum predicted NDVI across eligible points")
    predicted_ndvi_maximum: Optional[float] = Field(None, description="Maximum predicted NDVI across eligible points")
    predicted_ndvi_mean: Optional[float] = Field(None, description="Mean predicted NDVI across eligible points")
    predicted_ndvi_change_minimum: Optional[float] = Field(None, description="Minimum predicted NDVI change across eligible points")
    predicted_ndvi_change_maximum: Optional[float] = Field(None, description="Maximum predicted NDVI change across eligible points")
    predicted_ndvi_change_mean: Optional[float] = Field(None, description="Mean predicted NDVI change across eligible points")


class CaseStudyIndicators(BaseModel):
    p069_high_stress_risk: bool = Field(..., description="Whether P069 is under HIGH_STRESS_RISK")
    p077_watch_status: bool = Field(..., description="Whether P077 is under WATCH status")
    p147_prediction_available: bool = Field(..., description="Whether a prediction is available for P147")


class OperationalSummaryResponse(BaseModel):
    latest_realtime_week: str = Field(..., description="Start date of the latest realtime monitoring window (YYYY-MM-DD)")
    monitoring_coverage: MonitoringCoverage
    data_availability: DataAvailability
    prediction_status: PredictionStatus
    freshness_distribution: FreshnessDistribution
    operational_trend_distribution: OperationalTrendDistribution
    prediction_statistics: PredictionStatistics
    case_studies: CaseStudyIndicators
    raw_metrics: Dict[str, Union[str, int, float, bool, None]] = Field(
        ..., description="Direct 1:1 key-value map of the source operational metrics CSV"
    )


class LocationDetailResponse(BaseModel):
    point_id: str = Field(..., description="Unique identifier for the monitored point")
    district: str = Field(..., description="District name")
    latitude: float = Field(..., description="Geographic latitude coordinate")
    longitude: float = Field(..., description="Geographic longitude coordinate")
    case_study_group: Optional[str] = Field(None, description="Case study group designation if applicable")
    current_ndvi: Optional[float] = Field(None, description="Current observed Sentinel-2 NDVI")
    current_gsmap_rainfall_mm: Optional[float] = Field(None, description="Current observed GSMaP weekly rainfall in mm")
    current_temperature_c: Optional[float] = Field(None, description="Current observed ERA5-Land temperature in Celsius")
    latest_ndvi_week: Optional[str] = Field(None, description="Week start of the most recent NDVI observation")
    latest_rainfall_week: Optional[str] = Field(None, description="Week start of the most recent GSMaP rainfall observation")
    latest_temperature_week: Optional[str] = Field(None, description="Week start of the most recent temperature observation")
    common_latest_week: Optional[str] = Field(None, description="Most recent week where all modalities share valid observations")
    data_age_days: Optional[int] = Field(None, description="Age of observations in days relative to analysis date")
    freshness_status: str = Field(..., description="Freshness category: READY, WARNING, STALE, or VERY_STALE")
    prediction_eligibility: str = Field(..., description="Inference eligibility status: ELIGIBLE or NOT_READY")
    current_ndvi_available: bool = Field(..., description="Whether current NDVI observation is present")
    current_gsmap_available: bool = Field(..., description="Whether current GSMaP rainfall is present")
    current_temperature_available: bool = Field(..., description="Whether current temperature is present")
    current_inputs_complete: bool = Field(..., description="Whether all 3 current inputs are available")
    prediction_available: bool = Field(..., description="Whether next-week model prediction was generated")
    predicted_ndvi_next_week: Optional[float] = Field(None, description="Predicted NDVI for next week if available")
    predicted_ndvi_change: Optional[float] = Field(None, description="Predicted NDVI absolute change")
    predicted_ndvi_change_percent: Optional[float] = Field(None, description="Predicted NDVI percentage change")
    prediction_status: str = Field(..., description="Prediction execution status: READY or INCOMPLETE_INPUTS")
    operational_trend_status: str = Field(..., description="Trend category: HIGH_STRESS_RISK, WATCH, STABLE, IMPROVING, or NO_PREDICTION")
    reason: Optional[str] = Field(None, description="Operational explanation or decision support reason")


class WeeklyObservation(BaseModel):
    Week_Start: str = Field(..., description="Start date of the 7-day monitoring window")
    Week_End: str = Field(..., description="End date of the 7-day monitoring window")
    Week_Number: int = Field(..., description="ISO week number of the year")
    Year: int = Field(..., description="Observation year")
    NDVI: Optional[float] = Field(None, description="Harmonized Sentinel-2 NDVI observation or null if cloudy/missing")
    Rainfall_mm: Optional[float] = Field(None, description="GSMaP cumulative rainfall for the week in mm")
    Temperature_C: Optional[float] = Field(None, description="ERA5-Land mean temperature for the week in Celsius")


class LocationPredictionResponse(BaseModel):
    point_id: str = Field(..., description="Monitored location ID")
    Week_Start: str = Field(..., description="Monitoring week start date")
    current_ndvi: Optional[float] = Field(None, description="Current NDVI observation")
    gsmap_rainfall_mm: Optional[float] = Field(None, description="Current GSMaP rainfall in mm")
    temperature_c: Optional[float] = Field(None, description="Current temperature in Celsius")
    predicted_ndvi_next_week: Optional[float] = Field(None, description="Predicted next-week NDVI or null if unavailable")
    predicted_ndvi_change: Optional[float] = Field(None, description="Predicted NDVI change or null")
    predicted_ndvi_change_percent: Optional[float] = Field(None, description="Predicted NDVI change percent or null")
    prediction_status: str = Field(..., description="Inference status, e.g. READY or INCOMPLETE_INPUTS")
    operational_trend_status: str = Field(..., description="Operational trend label, e.g. HIGH_STRESS_RISK, WATCH, STABLE, NO_PREDICTION")
    freshness_status: str = Field(..., description="Freshness status of inputs")
    data_age_days: Optional[int] = Field(None, description="Age in days of the observation")
    prediction_available: bool = Field(..., description="True if next-week prediction exists, false otherwise")
    reason: Optional[str] = Field(None, description="Detailed explanation, particularly why prediction is unavailable")


class MapMarkerLocation(BaseModel):
    point_id: str = Field(..., description="Monitored location ID")
    district: str = Field(..., description="District name")
    latitude: float = Field(..., description="Latitude coordinate")
    longitude: float = Field(..., description="Longitude coordinate")
    operational_trend_status: str = Field(..., description="Trend alert level for map marker color coding")
    prediction_status: str = Field(..., description="Prediction status")
    prediction_available: bool = Field(..., description="Whether prediction exists for this location")
    current_ndvi: Optional[float] = Field(None, description="Current NDVI observation for tooltip preview")
    predicted_ndvi_next_week: Optional[float] = Field(None, description="Predicted next-week NDVI")
    predicted_ndvi_change: Optional[float] = Field(None, description="Predicted NDVI change")


class RecommendationResponse(BaseModel):
    point_id: str = Field(..., description="Monitored point ID")
    status: str = Field(..., description="Farmer-friendly operational status (e.g. High Stress Risk, Watch, Stable, Improving, Prediction Unavailable)")
    severity: str = Field(..., description="Severity level: high, medium, low, positive, or unknown")
    current_ndvi: Optional[float] = Field(None, description="Current observed vegetation index")
    predicted_ndvi_next_week: Optional[float] = Field(None, description="Predicted vegetation index for the upcoming week")
    predicted_ndvi_change: Optional[float] = Field(None, description="Predicted vegetation index absolute change")
    predicted_ndvi_change_percent: Optional[float] = Field(None, description="Predicted vegetation index percentage change")
    headline: str = Field(..., description="Farmer-facing headline summary")
    explanation: str = Field(..., description="Plain-language explanation grounded only in operational trend observations")
    recommended_action: str = Field(..., description="Safe, practical field monitoring action without medical/disease diagnosis")
    data_confidence: str = Field(..., description="Data confidence category: HIGH, MEDIUM, or LOW based on data completeness and freshness")
    prediction_available: bool = Field(..., description="Whether a next-week prediction is actively available")


class DashboardLocationInfo(BaseModel):
    point_id: str = Field(..., description="Monitored point identifier")
    district: str = Field(..., description="District name")
    latitude: float = Field(..., description="Latitude coordinate")
    longitude: float = Field(..., description="Longitude coordinate")
    case_study_group: Optional[str] = Field(None, description="Case study group if applicable")


class DashboardCurrentConditions(BaseModel):
    current_ndvi: Optional[float] = Field(None, description="Current observed Sentinel-2 NDVI")
    current_gsmap_rainfall_mm: Optional[float] = Field(None, description="Current weekly GSMaP rainfall in mm")
    current_temperature_c: Optional[float] = Field(None, description="Current weekly ERA5-Land temperature in Celsius")
    latest_ndvi_week: Optional[str] = Field(None, description="Latest observation date for NDVI")
    latest_rainfall_week: Optional[str] = Field(None, description="Latest observation date for GSMaP rainfall")
    latest_temperature_week: Optional[str] = Field(None, description="Latest observation date for temperature")
    common_latest_week: Optional[str] = Field(None, description="Most recent week where all modalities share valid observations")
    data_age_days: Optional[int] = Field(None, description="Age of observations in days")
    freshness_status: str = Field(..., description="Freshness status: READY, WARNING, STALE, or VERY_STALE")
    current_ndvi_available: bool = Field(..., description="Whether current NDVI observation is present")
    current_gsmap_available: bool = Field(..., description="Whether current GSMaP rainfall is present")
    current_temperature_available: bool = Field(..., description="Whether current temperature is present")
    current_inputs_complete: bool = Field(..., description="Whether all 3 current inputs are available")


class DashboardPrediction(BaseModel):
    prediction_available: bool = Field(..., description="Whether next-week prediction is available")
    predicted_ndvi_next_week: Optional[float] = Field(None, description="Predicted next-week NDVI or null")
    predicted_ndvi_change: Optional[float] = Field(None, description="Predicted NDVI absolute change")
    predicted_ndvi_change_percent: Optional[float] = Field(None, description="Predicted NDVI percentage change")
    prediction_status: str = Field(..., description="Prediction status: READY or INCOMPLETE_INPUTS")
    operational_trend_status: str = Field(..., description="Operational trend label")
    reason: Optional[str] = Field(None, description="Operational explanation or decision support reason")


class DashboardRecommendation(BaseModel):
    status: str = Field(..., description="Farmer-friendly operational status")
    severity: str = Field(..., description="Severity category: high, medium, low, positive, or unknown")
    headline: str = Field(..., description="Farmer-friendly headline")
    explanation: str = Field(..., description="Plain-language explanation grounded in data")
    recommended_action: str = Field(..., description="Recommended field monitoring action")
    data_confidence: str = Field(..., description="Data confidence: HIGH, MEDIUM, or LOW")


class LocationDashboardResponse(BaseModel):
    location: DashboardLocationInfo = Field(..., description="Point geographical and grouping metadata")
    current_conditions: DashboardCurrentConditions = Field(..., description="Current environmental and satellite observations")
    prediction: DashboardPrediction = Field(..., description="Model forecast metrics and inference status")
    recommendation: DashboardRecommendation = Field(..., description="Farmer-friendly interpretation and action recommendations")


