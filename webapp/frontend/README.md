# AgriVision AI - Farmer Web Application Frontend

Farmer-friendly field monitoring and operational forecast web application built with React, Vite, TypeScript, and Tailwind CSS.

---

## 1. Overview

This frontend connects to the AgriVision FastAPI backend (`http://127.0.0.1:8000`) and presents real-time operational vegetation intelligence:
- **Location Selector:** Browse all 26 monitored locations across Maharashtra (Latur & Kavathe-Khanapur/Satara).
- **Current Field Conditions:** Recent Sentinel-2 NDVI, GSMaP operational rainfall, ERA5-Land temperature, and freshness indicators.
- **Next-Week Forecast:** Model-predicted vegetation index and change trajectory.
- **Farmer Advisory Card:** Actionable field guidance with safety constraints (no disease diagnoses or unverified root causes).
- **Data Confidence Badge:** Clear `HIGH`, `MEDIUM`, or `LOW` classification based on input freshness.
- **12-Week Vegetation History:** Interactive Recharts timeline handling cloudy/missing weeks gracefully.
- **Interactive Monitoring Map:** Regional Leaflet map displaying all 26 field plots across Maharashtra, color-coded by operational trend status with two-way dashboard synchronization.

---

## 2. Setup & Installation

From the `webapp/frontend` directory:

```bash
npm install
```

---

## 3. Running the Application

### Start Frontend Dev Server
```bash
npm run dev
```

The application will be accessible at:
```
http://127.0.0.1:5173
```

### Production Build
```bash
npm run build
```

---

## 4. Environment Variables

Create a `.env` or `.env.local` file to customize the API base URL:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```
Default is `http://127.0.0.1:8000`.
