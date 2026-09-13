import math
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd

from app.config import settings
from app.schemas import (
    LocationDetailResponse,
    LocationPredictionResponse,
    MapMarkerLocation,
    WeeklyObservation,
)
from app.services.operational_data import OperationalDataNotFoundError


class PointNotFoundError(Exception):
    """Raised when a requested point ID is not found in the monitored dataset."""
    def __init__(self, point_id: str):
        super().__init__(f"Monitored point '{point_id}' was not found.")
        self.point_id = point_id


def _clean_str(val: Any) -> Optional[str]:
    if pd.isna(val):
        return None
    s = str(val).strip()
    return s if s else None


def _clean_float(val: Any) -> Optional[float]:
    if pd.isna(val):
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (ValueError, TypeError):
        return None


def _clean_int(val: Any) -> Optional[int]:
    if pd.isna(val):
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def _clean_bool(val: Any, default: bool = False) -> bool:
    if pd.isna(val):
        return default
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return default


def _load_operational_status_df(file_path: Optional[Path] = None) -> pd.DataFrame:
    path = file_path or settings.operational_status_file
    if not path.exists():
        raise OperationalDataNotFoundError(
            f"Operational status file not found at: {path}. "
            "Please ensure the AgriVision realtime monitoring pipeline has been run."
        )
    try:
        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception as exc:
        raise RuntimeError(f"Failed reading operational status CSV: {exc}") from exc


def _row_to_location_detail(row: pd.Series) -> LocationDetailResponse:
    return LocationDetailResponse(
        point_id=str(row["point_id"]).strip(),
        district=str(row["district"]).strip(),
        latitude=float(row["latitude"]),
        longitude=float(row["longitude"]),
        case_study_group=_clean_str(row.get("Case_Study_Group")),
        current_ndvi=_clean_float(row.get("Current_NDVI")),
        current_gsmap_rainfall_mm=_clean_float(row.get("Current_GSMaP_Rainfall_mm")),
        current_temperature_c=_clean_float(row.get("Current_Temperature_C")),
        latest_ndvi_week=_clean_str(row.get("latest_ndvi_week")),
        latest_rainfall_week=_clean_str(row.get("latest_rainfall_week")),
        latest_temperature_week=_clean_str(row.get("latest_temperature_week")),
        common_latest_week=_clean_str(row.get("common_latest_week")),
        data_age_days=_clean_int(row.get("data_age_days")),
        freshness_status=str(row.get("freshness_status", "")).strip(),
        prediction_eligibility=str(row.get("prediction_eligibility", "")).strip(),
        current_ndvi_available=_clean_bool(row.get("Current_NDVI_Available")),
        current_gsmap_available=_clean_bool(row.get("Current_GSMaP_Available")),
        current_temperature_available=_clean_bool(row.get("Current_Temperature_Available")),
        current_inputs_complete=_clean_bool(row.get("Current_Inputs_Complete")),
        prediction_available=_clean_bool(row.get("Prediction_Available")),
        predicted_ndvi_next_week=_clean_float(row.get("Predicted_NDVI_Next_Week")),
        predicted_ndvi_change=_clean_float(row.get("Predicted_NDVI_Change")),
        predicted_ndvi_change_percent=_clean_float(row.get("Predicted_NDVI_Change_Percent")),
        prediction_status=str(row.get("Prediction_Status", "")).strip(),
        operational_trend_status=str(row.get("Operational_Trend_Status", "")).strip(),
        reason=_clean_str(row.get("reason")),
    )


def get_all_locations() -> List[LocationDetailResponse]:
    """Return operational detail for all 26 monitored locations."""
    df = _load_operational_status_df()
    return [_row_to_location_detail(row) for _, row in df.iterrows()]


def get_location_by_id(point_id: str) -> LocationDetailResponse:
    """Return operational detail for a specific monitored location by point_id."""
    df = _load_operational_status_df()
    clean_id = point_id.strip()
    match = df[df["point_id"].astype(str).str.strip().str.upper() == clean_id.upper()]
    if match.empty:
        raise PointNotFoundError(clean_id)
    return _row_to_location_detail(match.iloc[0])


def get_location_history(point_id: str) -> List[WeeklyObservation]:
    """
    Return chronological weekly history observations for a specific monitored location.
    Missing observations (such as cloudy NDVI) remain null.
    """
    # Verify point exists first
    _ = get_location_by_id(point_id)

    history_path = settings.realtime_weekly_file
    if not history_path.exists():
        raise OperationalDataNotFoundError(
            f"Realtime weekly history file not found at: {history_path}. "
            "Please ensure the AgriVision realtime monitoring pipeline has been run."
        )

    try:
        df = pd.read_csv(history_path)
        df.columns = [str(c).strip() for c in df.columns]
    except Exception as exc:
        raise RuntimeError(f"Failed reading realtime weekly observations CSV: {exc}") from exc

    clean_id = point_id.strip()
    match_df = df[df["point_id"].astype(str).str.strip().str.upper() == clean_id.upper()].copy()

    if match_df.empty:
        return []

    # Sort chronologically by Week_Start
    match_df.sort_values(by=["Week_Start"], ascending=True, inplace=True)

    records: List[WeeklyObservation] = []
    for _, row in match_df.iterrows():
        records.append(
            WeeklyObservation(
                Week_Start=str(row["Week_Start"]).strip(),
                Week_End=str(row["Week_End"]).strip(),
                Week_Number=int(row["Week_Number"]),
                Year=int(row["Year"]),
                NDVI=_clean_float(row.get("NDVI")),
                Rainfall_mm=_clean_float(row.get("Rainfall_mm")),
                Temperature_C=_clean_float(row.get("Temperature_C")),
            )
        )
    return records


def get_location_prediction(point_id: str) -> LocationPredictionResponse:
    """
    Return current prediction information for a specific point.
    If no prediction exists (e.g. P147 due to missing current data), returns
    HTTP 200 representation with prediction_available=False and preserved reason.
    """
    # Authoritative operational status row
    op_status = get_location_by_id(point_id)

    pred_path = settings.realtime_predictions_file
    if not pred_path.exists():
        raise OperationalDataNotFoundError(
            f"Realtime predictions file not found at: {pred_path}. "
            "Please ensure the AgriVision realtime monitoring pipeline has been run."
        )

    try:
        pred_df = pd.read_csv(pred_path)
        pred_df.columns = [str(c).strip() for c in pred_df.columns]
    except Exception as exc:
        raise RuntimeError(f"Failed reading realtime predictions CSV: {exc}") from exc

    clean_id = point_id.strip()
    match = pred_df[pred_df["point_id"].astype(str).str.strip().str.upper() == clean_id.upper()]

    if not match.empty:
        pred_row = match.iloc[0]
        week_start = str(pred_row.get("Week_Start", op_status.latest_ndvi_week or "")).strip()
        current_ndvi = _clean_float(pred_row.get("NDVI")) if _clean_float(pred_row.get("NDVI")) is not None else op_status.current_ndvi
        rainfall_val = _clean_float(pred_row.get("Rainfall_GSMaP")) if _clean_float(pred_row.get("Rainfall_GSMaP")) is not None else op_status.current_gsmap_rainfall_mm
        temp_val = _clean_float(pred_row.get("Temperature_C")) if _clean_float(pred_row.get("Temperature_C")) is not None else op_status.current_temperature_c
        pred_ndvi = _clean_float(pred_row.get("Predicted_NDVI_Next_Week")) if _clean_float(pred_row.get("Predicted_NDVI_Next_Week")) is not None else op_status.predicted_ndvi_next_week
        pred_change = _clean_float(pred_row.get("Predicted_NDVI_Change")) if _clean_float(pred_row.get("Predicted_NDVI_Change")) is not None else op_status.predicted_ndvi_change
        pred_change_pct = _clean_float(pred_row.get("Predicted_NDVI_Change_Percent")) if _clean_float(pred_row.get("Predicted_NDVI_Change_Percent")) is not None else op_status.predicted_ndvi_change_percent
        pred_status = str(pred_row.get("Prediction_Status", op_status.prediction_status)).strip()
        trend_status = str(pred_row.get("Operational_Trend_Status", op_status.operational_trend_status)).strip()
        freshness = str(pred_row.get("Freshness_Status", op_status.freshness_status)).strip()
        data_age = _clean_int(pred_row.get("Data_Age_Days")) if _clean_int(pred_row.get("Data_Age_Days")) is not None else op_status.data_age_days
        pred_available = op_status.prediction_available
    else:
        week_start = op_status.latest_ndvi_week or ""
        current_ndvi = op_status.current_ndvi
        rainfall_val = op_status.current_gsmap_rainfall_mm
        temp_val = op_status.current_temperature_c
        pred_ndvi = op_status.predicted_ndvi_next_week
        pred_change = op_status.predicted_ndvi_change
        pred_change_pct = op_status.predicted_ndvi_change_percent
        pred_status = op_status.prediction_status
        trend_status = op_status.operational_trend_status
        freshness = op_status.freshness_status
        data_age = op_status.data_age_days
        pred_available = op_status.prediction_available

    return LocationPredictionResponse(
        point_id=op_status.point_id,
        Week_Start=week_start,
        current_ndvi=current_ndvi,
        gsmap_rainfall_mm=rainfall_val,
        temperature_c=temp_val,
        predicted_ndvi_next_week=pred_ndvi,
        predicted_ndvi_change=pred_change,
        predicted_ndvi_change_percent=pred_change_pct,
        prediction_status=pred_status,
        operational_trend_status=trend_status,
        freshness_status=freshness,
        data_age_days=data_age,
        prediction_available=pred_available,
        reason=op_status.reason,
    )


def get_map_locations() -> List[MapMarkerLocation]:
    """Return lightweight map marker metadata for all 26 monitored locations."""
    df = _load_operational_status_df()
    markers: List[MapMarkerLocation] = []
    for _, row in df.iterrows():
        markers.append(
            MapMarkerLocation(
                point_id=str(row["point_id"]).strip(),
                district=str(row["district"]).strip(),
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                operational_trend_status=str(row.get("Operational_Trend_Status", "")).strip(),
                prediction_status=str(row.get("Prediction_Status", "")).strip(),
                prediction_available=_clean_bool(row.get("Prediction_Available")),
                current_ndvi=_clean_float(row.get("Current_NDVI")),
                predicted_ndvi_next_week=_clean_float(row.get("Predicted_NDVI_Next_Week")),
                predicted_ndvi_change=_clean_float(row.get("Predicted_NDVI_Change")),
            )
        )
    return markers
