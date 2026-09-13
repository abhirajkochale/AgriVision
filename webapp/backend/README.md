# AgriVision AI - Webapp Backend Foundation

FastAPI backend providing operational decision support data for the AgriVision farmer web application.

---

## 1. Overview

This backend acts as the API interface on top of AgriVision's existing 2026 realtime ML pipeline. It exposes:
- **System Health Status** (`/api/health`)
- **Operational Metrics Summary** (`/api/summary`), reading directly from `ml/realtime/final_report/AgriVision_Operational_Metrics.csv`

The architecture is fully decoupled from model training and GEE data ingestion, allowing fast, deterministic responses and future scalability (e.g. database persistence or live background triggers).

---

## 2. Requirements & Installation

Navigate to this directory and install dependencies:

```bash
cd webapp/backend
pip install -r requirements.txt
```

---

## 3. Starting the Server

From the `webapp/backend` directory:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Or from the project root (`c:\Projects\AgriVision`):

```bash
python -m uvicorn app.main:app --app-dir webapp/backend --host 127.0.0.1 --port 8000 --reload
```

Interactive OpenAPI documentation is available at:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

---

## 4. API Endpoints

### Health Check
- **Route:** `GET /api/health`
- **Response:**
  ```json
  {
    "status": "ok",
    "service": "AgriVision AI Backend",
    "timestamp": "2026-09-14T00:45:00.000000Z",
    "version": "1.0.0"
  }
  ```

### Operational Metrics Summary
- **Route:** `GET /api/summary`
- **Response:** Structured JSON containing monitoring coverage, data availability, prediction counts, freshness and trend distributions, prediction statistics, case studies, and complete 1:1 raw metrics.

### Monitored Locations List
- **Route:** `GET /api/locations`
- **Response:** List of all 26 monitored locations with comprehensive operational status, sensor data availability, and current predictions.

### Single Location Details
- **Route:** `GET /api/locations/{point_id}`
- **Examples:** `/api/locations/P069`, `/api/locations/P147`
- **Response:** Detailed operational status and prediction data for the requested location. Returns HTTP 404 (`PointNotFound`) if the ID does not exist.

### Location Historical Observations
- **Route:** `GET /api/locations/{point_id}/history`
- **Response:** Chronologically sorted list of weekly observations (`Week_Start`, `Week_End`, `Week_Number`, `Year`, `NDVI`, `Rainfall_mm`, `Temperature_C`). Missing values remain null.

### Location Prediction
- **Route:** `GET /api/locations/{point_id}/prediction`
- **Response:** Current next-week NDVI prediction, changes, confidence indicators, and trend status. If inference is unavailable (e.g. P147 due to stale data), returns HTTP 200 with `prediction_available: false` and explanatory `reason`.

### Map Marker Metadata
- **Route:** `GET /api/map`
- **Response:** Lightweight list of all 26 locations optimized for frontend map rendering (`point_id`, `district`, `latitude`, `longitude`, `operational_trend_status`, `prediction_status`, `prediction_available`, `current_ndvi`, `predicted_ndvi_next_week`, `predicted_ndvi_change`).

### Farmer-Friendly Recommendation
- **Route:** `GET /api/locations/{point_id}/recommendation`
- **Response:** Plain-language, actionable guidance for farmers translating the operational vegetation trend without medical/disease diagnosis. Includes:
  - `status`: `"High Stress Risk"`, `"Watch"`, `"Stable"`, `"Improving"`, or `"Prediction Unavailable"`
  - `severity`: `"high"`, `"medium"`, `"low"`, `"positive"`, or `"unknown"`
  - `headline`: Direct operational forecast headline
  - `explanation`: Safe, grounded explanation
  - `recommended_action`: Monitoring and management guidance
  - `data_confidence`: `"HIGH"`, `"MEDIUM"`, or `"LOW"` based purely on data completeness and freshness

### Unified Location Dashboard
- **Route:** `GET /api/locations/{point_id}/dashboard`
- **Response:** Single composite payload for frontend dashboards merging:
  - `location`: Point ID, coordinates, district, case study designation
  - `current_conditions`: Latest NDVI, rainfall, temperature, freshness dates
  - `prediction`: Next-week NDVI prediction, changes, inference status
  - `recommendation`: Farmer-friendly status, severity, explanation, action, and data confidence

---

## 5. Configuration & Environment Variables

All settings can be customized using environment variables prefixed with `AGRIVISION_`:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `AGRIVISION_PROJECT_ROOT` | Absolute path to AgriVision root | Resolved automatically (`parents[3]`) |
| `AGRIVISION_METRICS_FILE_PATH` | Path to operational metrics CSV | Resolved relative to `PROJECT_ROOT` |
| `AGRIVISION_OPERATIONAL_STATUS_FILE_PATH` | Path to operational status CSV | Resolved relative to `PROJECT_ROOT` |
| `AGRIVISION_REALTIME_WEEKLY_FILE_PATH` | Path to weekly observations CSV | Resolved relative to `PROJECT_ROOT` |
| `AGRIVISION_REALTIME_PREDICTIONS_FILE_PATH` | Path to realtime predictions CSV | Resolved relative to `PROJECT_ROOT` |
| `AGRIVISION_CORS_ORIGINS` | List of allowed CORS origins | Localhost React/Vite development ports |

---

## 6. Error Handling

- **Missing Operational Files:** Returns HTTP 404 with structured JSON:
  ```json
  {
    "error": "OperationalDataNotFound",
    "message": "Operational metrics report not found at ...",
    "timestamp": "2026-09-14T00:45:00.000000Z"
  }
  ```
- **Unknown Point ID:** Returns HTTP 404 with structured JSON:
  ```json
  {
    "error": "PointNotFound",
    "detail": "Monitored point 'INVALID' was not found among the 26 AgriVision locations.",
    "point_id": "INVALID",
    "timestamp": "2026-09-14T00:45:00.000000Z"
  }
  ```
