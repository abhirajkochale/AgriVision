from typing import Optional

from app.schemas import (
    DashboardCurrentConditions,
    DashboardLocationInfo,
    DashboardPrediction,
    DashboardRecommendation,
    LocationDashboardResponse,
    LocationDetailResponse,
    RecommendationResponse,
)
from app.services.location_data import get_location_by_id


def compute_data_confidence(op_status: LocationDetailResponse) -> str:
    """
    Classify data confidence based solely on existing data availability and freshness:
      - HIGH: prediction is available, all 3 current inputs are complete, and freshness is READY.
      - MEDIUM: prediction is available, but data has a weaker freshness status.
      - LOW: prediction unavailable, or current inputs incomplete/stale.
    """
    if op_status.prediction_available:
        if op_status.current_inputs_complete and op_status.freshness_status.strip().upper() == "READY":
            return "HIGH"
        return "MEDIUM"
    return "LOW"


def get_location_recommendation(point_id: str) -> RecommendationResponse:
    """
    Translate technical operational trend output into clear, farmer-friendly guidance.
    Strictly observes agricultural safety rules:
      - No crop disease diagnoses.
      - No unverified specific root cause claims.
      - No fabricated missing values or predictions.
    """
    op_status = get_location_by_id(point_id)
    trend = op_status.operational_trend_status.strip().upper()
    confidence = compute_data_confidence(op_status)

    if trend == "HIGH_STRESS_RISK" and op_status.prediction_available:
        status = "High Stress Risk"
        severity = "high"
        headline = "Vegetation decline is forecast."
        explanation = "The model predicts a notable decline in vegetation index over the next week."
        action = "Inspect the field for visible crop stress and continue close monitoring."
        pred_avail = True
        pred_next = op_status.predicted_ndvi_next_week
        pred_change = op_status.predicted_ndvi_change
        pred_pct = op_status.predicted_ndvi_change_percent

    elif trend == "WATCH" and op_status.prediction_available:
        status = "Watch"
        severity = "medium"
        headline = "A moderate vegetation decline is forecast."
        explanation = "The model predicts a decrease in vegetation index that warrants closer observation."
        action = "Monitor the field closely and check for visible signs of crop stress."
        pred_avail = True
        pred_next = op_status.predicted_ndvi_next_week
        pred_change = op_status.predicted_ndvi_change
        pred_pct = op_status.predicted_ndvi_change_percent

    elif trend == "STABLE" and op_status.prediction_available:
        status = "Stable"
        severity = "low"
        headline = "Vegetation is expected to remain relatively stable."
        explanation = "The predicted next-week vegetation index is close to the current value."
        action = "Continue regular crop monitoring."
        pred_avail = True
        pred_next = op_status.predicted_ndvi_next_week
        pred_change = op_status.predicted_ndvi_change
        pred_pct = op_status.predicted_ndvi_change_percent

    elif trend == "IMPROVING" and op_status.prediction_available:
        status = "Improving"
        severity = "positive"
        headline = "Vegetation improvement is forecast."
        explanation = "The predicted next-week vegetation index is higher than the current value."
        action = "Continue regular monitoring and maintain current field management practices."
        pred_avail = True
        pred_next = op_status.predicted_ndvi_next_week
        pred_change = op_status.predicted_ndvi_change
        pred_pct = op_status.predicted_ndvi_change_percent

    else:
        # NO_PREDICTION or ineligible state
        status = "Prediction Unavailable"
        severity = "unknown"
        headline = "Vegetation forecast is currently unavailable."
        if op_status.reason:
            clean_reason = op_status.reason.strip()
            # Lowercase first character for smooth grammatical flow
            formatted_reason = clean_reason[0].lower() + clean_reason[1:] if len(clean_reason) > 1 else clean_reason
            explanation = f"Prediction is unavailable because {formatted_reason}"
            if not explanation.endswith("."):
                explanation += "."
        else:
            explanation = "Prediction is unavailable because required current input data is incomplete or unavailable."
        action = "Check again when a newer satellite observation becomes available."
        pred_avail = False
        pred_next = None
        pred_change = None
        pred_pct = None

    return RecommendationResponse(
        point_id=op_status.point_id,
        status=status,
        severity=severity,
        current_ndvi=op_status.current_ndvi,
        predicted_ndvi_next_week=pred_next,
        predicted_ndvi_change=pred_change,
        predicted_ndvi_change_percent=pred_pct,
        headline=headline,
        explanation=explanation,
        recommended_action=action,
        data_confidence=confidence,
        prediction_available=pred_avail,
    )


def get_location_dashboard(point_id: str) -> LocationDashboardResponse:
    """
    Return a comprehensive, composite operational dashboard payload for a monitored point,
    integrating location metadata, current environmental conditions, forecast metrics,
    and farmer-friendly guidance.
    """
    op_status = get_location_by_id(point_id)
    rec = get_location_recommendation(point_id)

    location_info = DashboardLocationInfo(
        point_id=op_status.point_id,
        district=op_status.district,
        latitude=op_status.latitude,
        longitude=op_status.longitude,
        case_study_group=op_status.case_study_group,
    )

    current_conditions = DashboardCurrentConditions(
        current_ndvi=op_status.current_ndvi,
        current_gsmap_rainfall_mm=op_status.current_gsmap_rainfall_mm,
        current_temperature_c=op_status.current_temperature_c,
        latest_ndvi_week=op_status.latest_ndvi_week,
        latest_rainfall_week=op_status.latest_rainfall_week,
        latest_temperature_week=op_status.latest_temperature_week,
        common_latest_week=op_status.common_latest_week,
        data_age_days=op_status.data_age_days,
        freshness_status=op_status.freshness_status,
        current_ndvi_available=op_status.current_ndvi_available,
        current_gsmap_available=op_status.current_gsmap_available,
        current_temperature_available=op_status.current_temperature_available,
        current_inputs_complete=op_status.current_inputs_complete,
    )

    prediction = DashboardPrediction(
        prediction_available=op_status.prediction_available,
        predicted_ndvi_next_week=op_status.predicted_ndvi_next_week,
        predicted_ndvi_change=op_status.predicted_ndvi_change,
        predicted_ndvi_change_percent=op_status.predicted_ndvi_change_percent,
        prediction_status=op_status.prediction_status,
        operational_trend_status=op_status.operational_trend_status,
        reason=op_status.reason,
    )

    recommendation = DashboardRecommendation(
        status=rec.status,
        severity=rec.severity,
        headline=rec.headline,
        explanation=rec.explanation,
        recommended_action=rec.recommended_action,
        data_confidence=rec.data_confidence,
    )

    return LocationDashboardResponse(
        location=location_info,
        current_conditions=current_conditions,
        prediction=prediction,
        recommendation=recommendation,
    )
