import math
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from app.config import settings
from app.schemas import (
    CaseStudyIndicators,
    DataAvailability,
    FreshnessDistribution,
    MonitoringCoverage,
    OperationalSummaryResponse,
    OperationalTrendDistribution,
    PredictionStatistics,
    PredictionStatus,
)


class OperationalDataNotFoundError(FileNotFoundError):
    """Raised when the operational metrics CSV output cannot be found."""
    pass


def _parse_value(val: Any) -> Union[str, int, float, bool, None]:
    """Parse string/numeric representations from CSV rows into typed Python values."""
    if pd.isna(val):
        return None

    val_str = str(val).strip()

    # Boolean values
    if val_str.lower() in ("true", "yes"):
        return True
    if val_str.lower() in ("false", "no"):
        return False

    # Integer values
    try:
        if val_str.isdigit() or (val_str.startswith("-") and val_str[1:].isdigit()):
            return int(val_str)
    except ValueError:
        pass

    # Floating point values
    try:
        f_val = float(val_str)
        if math.isnan(f_val):
            return None
        return f_val
    except ValueError:
        pass

    return val_str


def _extract_int(d: Dict[str, Any], key: str, default: int = 0) -> int:
    val = d.get(key)
    if val is None:
        return default
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


def _extract_float(d: Dict[str, Any], key: str) -> Optional[float]:
    val = d.get(key)
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (ValueError, TypeError):
        return None


def _extract_bool(d: Dict[str, Any], key: str, default: bool = False) -> bool:
    val = d.get(key)
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    val_str = str(val).strip().lower()
    if val_str in ("true", "1", "yes"):
        return True
    if val_str in ("false", "0", "no"):
        return False
    return default


def get_operational_metrics(metrics_file: Optional[Path] = None) -> OperationalSummaryResponse:
    """
    Read the operational metrics CSV output from the realtime pipeline
    and construct an OperationalSummaryResponse Pydantic model.
    """
    file_path = metrics_file or settings.operational_metrics_file

    if not file_path.exists():
        raise OperationalDataNotFoundError(
            f"Operational metrics report not found at: {file_path}. "
            "Please ensure the AgriVision realtime monitoring pipeline has been run."
        )

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        raise RuntimeError(f"Failed reading operational metrics CSV: {exc}") from exc

    if "Metric" not in df.columns or "Value" not in df.columns:
        raise ValueError(
            f"Operational metrics CSV at {file_path} is missing expected columns 'Metric' and 'Value'."
        )

    raw_metrics: Dict[str, Union[str, int, float, bool, None]] = {}
    for _, row in df.iterrows():
        metric_name = str(row["Metric"]).strip()
        raw_metrics[metric_name] = _parse_value(row["Value"])

    coverage = MonitoringCoverage(
        total_monitored_points=_extract_int(raw_metrics, "Total_Monitored_Points"),
        latur_points=_extract_int(raw_metrics, "Latur_Points"),
        kavathe_khanapur_points=_extract_int(raw_metrics, "Kavathe_Khanapur_Points"),
    )

    data_availability = DataAvailability(
        current_ndvi_available_points=_extract_int(raw_metrics, "Current_NDVI_Available_Points"),
        current_gsmap_available_points=_extract_int(raw_metrics, "Current_GSMaP_Available_Points"),
        current_temperature_available_points=_extract_int(raw_metrics, "Current_Temperature_Available_Points"),
        complete_current_input_points=_extract_int(raw_metrics, "Complete_Current_Input_Points"),
    )

    prediction_status = PredictionStatus(
        prediction_ready_points=_extract_int(raw_metrics, "Prediction_Ready_Points"),
        predictions_generated=_extract_int(raw_metrics, "Predictions_Generated"),
        no_prediction_points=_extract_int(raw_metrics, "No_Prediction_Points"),
    )

    freshness_distribution = FreshnessDistribution(
        ready=_extract_int(raw_metrics, "Freshness_READY"),
        warning=_extract_int(raw_metrics, "Freshness_WARNING"),
        stale=_extract_int(raw_metrics, "Freshness_STALE"),
        very_stale=_extract_int(raw_metrics, "Freshness_VERY_STALE"),
    )

    trend_distribution = OperationalTrendDistribution(
        high_stress_risk=_extract_int(raw_metrics, "Trend_HIGH_STRESS_RISK"),
        watch=_extract_int(raw_metrics, "Trend_WATCH"),
        stable=_extract_int(raw_metrics, "Trend_STABLE"),
        improving=_extract_int(raw_metrics, "Trend_IMPROVING"),
        no_prediction=_extract_int(raw_metrics, "Trend_NO_PREDICTION"),
    )

    prediction_statistics = PredictionStatistics(
        predicted_ndvi_minimum=_extract_float(raw_metrics, "Predicted_NDVI_Minimum"),
        predicted_ndvi_maximum=_extract_float(raw_metrics, "Predicted_NDVI_Maximum"),
        predicted_ndvi_mean=_extract_float(raw_metrics, "Predicted_NDVI_Mean"),
        predicted_ndvi_change_minimum=_extract_float(raw_metrics, "Predicted_NDVI_Change_Minimum"),
        predicted_ndvi_change_maximum=_extract_float(raw_metrics, "Predicted_NDVI_Change_Maximum"),
        predicted_ndvi_change_mean=_extract_float(raw_metrics, "Predicted_NDVI_Change_Mean"),
    )

    case_studies = CaseStudyIndicators(
        p069_high_stress_risk=_extract_bool(raw_metrics, "P069_High_Stress_Risk"),
        p077_watch_status=_extract_bool(raw_metrics, "P077_Watch_Status"),
        p147_prediction_available=_extract_bool(raw_metrics, "P147_Prediction_Available"),
    )

    latest_week = str(raw_metrics.get("Latest_Realtime_Week", ""))

    return OperationalSummaryResponse(
        latest_realtime_week=latest_week,
        monitoring_coverage=coverage,
        data_availability=data_availability,
        prediction_status=prediction_status,
        freshness_distribution=freshness_distribution,
        operational_trend_distribution=trend_distribution,
        prediction_statistics=prediction_statistics,
        case_studies=case_studies,
        raw_metrics=raw_metrics,
    )
