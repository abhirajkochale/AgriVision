from datetime import datetime, timezone
from typing import List
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.schemas import (
    HealthResponse,
    LocationDashboardResponse,
    LocationDetailResponse,
    LocationPredictionResponse,
    MapMarkerLocation,
    OperationalSummaryResponse,
    RecommendationResponse,
    WeeklyObservation,
)
from app.services.location_data import (
    PointNotFoundError,
    get_all_locations,
    get_location_by_id,
    get_location_history,
    get_location_prediction,
    get_map_locations,
)
from app.services.operational_data import (
    OperationalDataNotFoundError,
    get_operational_metrics,
)
from app.services.recommendation import (
    get_location_dashboard,
    get_location_recommendation,
)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "FastAPI backend service for AgriVision AI. "
        "Provides operational decision support metrics from the 2026 realtime monitoring pipeline."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS middleware for local frontend development (e.g., Vite/React on localhost:3000 or localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(OperationalDataNotFoundError)
async def operational_data_not_found_handler(
    request: Request,
    exc: OperationalDataNotFoundError,
) -> JSONResponse:
    """Handle missing operational CSV files gracefully with HTTP 404 and structured JSON."""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": "OperationalDataNotFound",
            "message": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


@app.exception_handler(PointNotFoundError)
async def point_not_found_handler(
    request: Request,
    exc: PointNotFoundError,
) -> JSONResponse:
    """Handle unknown point IDs with HTTP 404 and structured JSON."""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": "PointNotFound",
            "detail": f"Monitored point '{exc.point_id}' was not found among the 26 AgriVision locations.",
            "point_id": exc.point_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


@app.get(
    "/api/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check endpoint",
)
def get_health() -> HealthResponse:
    """Return backend operational status confirming that the API service is running."""
    return HealthResponse(
        status="ok",
        service=settings.PROJECT_NAME,
        timestamp=datetime.now(timezone.utc),
        version=settings.VERSION,
    )


@app.get(
    "/api/summary",
    response_model=OperationalSummaryResponse,
    tags=["Operational"],
    summary="Get operational metrics summary",
)
def get_summary() -> OperationalSummaryResponse:
    """
    Expose current operational metrics from the AgriVision realtime monitoring pipeline.
    Reads ml/realtime/final_report/AgriVision_Operational_Metrics.csv.
    """
    try:
        return get_operational_metrics()
    except OperationalDataNotFoundError:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred while processing operational metrics: {exc}",
        )


@app.get(
    "/api/locations",
    response_model=List[LocationDetailResponse],
    tags=["Locations"],
    summary="List all monitored locations",
)
def get_locations() -> List[LocationDetailResponse]:
    """Return operational details and current predictions for all 26 monitored locations."""
    try:
        return get_all_locations()
    except OperationalDataNotFoundError:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed loading monitored locations: {exc}",
        )


@app.get(
    "/api/locations/{point_id}",
    response_model=LocationDetailResponse,
    tags=["Locations"],
    summary="Get single location details",
)
def get_location(point_id: str) -> LocationDetailResponse:
    """Return operational details and status for a single monitored location."""
    try:
        return get_location_by_id(point_id)
    except (PointNotFoundError, OperationalDataNotFoundError):
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed retrieving location '{point_id}': {exc}",
        )


@app.get(
    "/api/locations/{point_id}/history",
    response_model=List[WeeklyObservation],
    tags=["Locations"],
    summary="Get location weekly historical observations",
)
def get_history(point_id: str) -> List[WeeklyObservation]:
    """
    Return historical weekly observations for the requested location ordered chronologically.
    Missing observations (e.g. cloud-obscured NDVI) remain null.
    """
    try:
        return get_location_history(point_id)
    except (PointNotFoundError, OperationalDataNotFoundError):
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed retrieving history for '{point_id}': {exc}",
        )


@app.get(
    "/api/locations/{point_id}/prediction",
    response_model=LocationPredictionResponse,
    tags=["Locations"],
    summary="Get current prediction for a location",
)
def get_prediction(point_id: str) -> LocationPredictionResponse:
    """
    Return next-week prediction information for the requested location.
    If no prediction exists (e.g. P147), returns prediction_available=false with preserved reason.
    """
    try:
        return get_location_prediction(point_id)
    except (PointNotFoundError, OperationalDataNotFoundError):
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed retrieving prediction for '{point_id}': {exc}",
        )


@app.get(
    "/api/map",
    response_model=List[MapMarkerLocation],
    tags=["Map"],
    summary="Lightweight map marker data",
)
def get_map() -> List[MapMarkerLocation]:
    """Return lightweight marker data for all 26 locations optimized for map rendering."""
    try:
        return get_map_locations()
    except OperationalDataNotFoundError:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed retrieving map locations: {exc}",
        )


@app.get(
    "/api/locations/{point_id}/recommendation",
    response_model=RecommendationResponse,
    tags=["Recommendations"],
    summary="Farmer-friendly operational recommendation",
)
def get_recommendation(point_id: str) -> RecommendationResponse:
    """
    Return safe, farmer-friendly interpretation of the operational trend for the point.
    Grounds guidance only in operational trend observations without plant disease diagnoses.
    """
    try:
        return get_location_recommendation(point_id)
    except (PointNotFoundError, OperationalDataNotFoundError):
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed generating recommendation for '{point_id}': {exc}",
        )


@app.get(
    "/api/locations/{point_id}/dashboard",
    response_model=LocationDashboardResponse,
    tags=["Dashboard"],
    summary="Unified location operational dashboard",
)
def get_dashboard(point_id: str) -> LocationDashboardResponse:
    """
    Return a single composite operational dashboard payload for the requested location,
    integrating location info, current conditions, prediction metrics, and recommendation.
    """
    try:
        return get_location_dashboard(point_id)
    except (PointNotFoundError, OperationalDataNotFoundError):
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed generating dashboard for '{point_id}': {exc}",
        )


