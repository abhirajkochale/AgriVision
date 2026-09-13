# pyrefly: ignore [missing-import]

from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd


# ============================================================================
# AGRIVISION AI
# REAL-TIME GSMaP NEXT-WEEK NDVI PREDICTION
# 2026 OPERATIONAL PIPELINE
#
# INPUT:
#   ml/realtime/data/AgriVision_Realtime_Weekly_Kavathe_Latur.csv
#
# MODEL:
#   ml/model_outputs/AgriVision_GSMaP_Tuned_ExtraTrees_Model.joblib
#
# OUTPUT:
#   ml/realtime/predictions/realtime_predictions_2026.csv
#   ml/realtime/predictions/realtime_prediction_summary.txt
#
# IMPORTANT:
# - Uses GSMaP rainfall.
# - Reconstructs the same 33 features used during model training.
# - Does NOT fit a new imputer.
# - The saved model already contains the training-fitted imputer.
# - Missing lag/rolling features are allowed.
# - Current NDVI, rainfall and temperature must exist for a READY
#   operational prediction.
# - No NDVI/rainfall/temperature values are fabricated.
# - Calendar-week gaps are preserved when constructing lag features.
# ============================================================================


print("=" * 90)
print("AGRIVISION AI - REAL-TIME GSMaP NDVI PREDICTION")
print("2026 OPERATIONAL PIPELINE")
print("=" * 90)


# ============================================================================
# 1. PROJECT PATHS
# ============================================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

REALTIME_DATA_DIR = (
    PROJECT_DIR
    / "ml"
    / "realtime"
    / "data"
)

PREDICTION_OUTPUT_DIR = (
    PROJECT_DIR
    / "ml"
    / "realtime"
    / "predictions"
)

MODEL_FILE = (
    PROJECT_DIR
    / "ml"
    / "model_outputs"
    / "AgriVision_GSMaP_Tuned_ExtraTrees_Model.joblib"
)

REALTIME_DATA_FILE = (
    REALTIME_DATA_DIR
    / "AgriVision_Realtime_Weekly_Kavathe_Latur.csv"
)

OUTPUT_FILE = (
    PREDICTION_OUTPUT_DIR
    / "realtime_predictions_2026.csv"
)

SUMMARY_FILE = (
    PREDICTION_OUTPUT_DIR
    / "realtime_prediction_summary.txt"
)

PREDICTION_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================================
# 2. CHECK INPUT FILES
# ============================================================================

print()
print("=" * 90)
print("CHECKING INPUT FILES")
print("=" * 90)

print()
print("Realtime merged dataset:")
print(REALTIME_DATA_FILE)

if not REALTIME_DATA_FILE.exists():
    raise FileNotFoundError(
        "\nRealtime merged dataset not found:\n"
        f"{REALTIME_DATA_FILE}\n\n"
        "Run realtime_data.py first."
    )

print("PASS - Realtime merged dataset found")

print()
print("GSMaP model:")
print(MODEL_FILE)

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        "\nGSMaP model not found:\n"
        f"{MODEL_FILE}"
    )

print("PASS - GSMaP model found")


# ============================================================================
# 3. LOAD REALTIME DATA
# ============================================================================

print()
print("=" * 90)
print("LOADING REALTIME DATA")
print("=" * 90)

df = pd.read_csv(
    REALTIME_DATA_FILE
)

print()
print(
    f"Rows loaded:    {len(df):,}"
)

print(
    f"Columns loaded: {len(df.columns)}"
)

print()
print("Columns:")
print(
    list(df.columns)
)


# ============================================================================
# 4. LOAD MODEL
# ============================================================================

print()
print("=" * 90)
print("LOADING GSMaP MODEL")
print("=" * 90)

model = joblib.load(
    MODEL_FILE
)

print()
print(
    "PASS - GSMaP model loaded"
)

print(
    f"Model type: {type(model).__name__}"
)


# ============================================================================
# 5. REQUIRED RAW COLUMNS
# ============================================================================

print()
print("=" * 90)
print("CHECKING REALTIME DATA SCHEMA")
print("=" * 90)


required_columns = [
    "point_id",
    "Week_Start",
    "NDVI",
]


missing_required = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_required:
    raise ValueError(
        "Realtime dataset is missing required columns:\n"
        + "\n".join(
            missing_required
        )
    )


# ============================================================================
# 6. NORMALIZE RAINFALL COLUMN
# ============================================================================

if "Rainfall_GSMaP" in df.columns:

    rainfall_column = "Rainfall_GSMaP"

elif "Rainfall_Current" in df.columns:

    rainfall_column = "Rainfall_Current"

elif "Rainfall_mm" in df.columns:

    rainfall_column = "Rainfall_mm"

else:

    raise ValueError(
        "\nNo rainfall column found.\n"
        "Expected one of:\n"
        "Rainfall_GSMaP\n"
        "Rainfall_Current\n"
        "Rainfall_mm"
    )


df["Rainfall_GSMaP"] = pd.to_numeric(
    df[rainfall_column],
    errors="coerce"
)


# ============================================================================
# 7. NORMALIZE TEMPERATURE COLUMN
# ============================================================================

if "Temperature_C" in df.columns:

    temperature_column = "Temperature_C"

elif "Temperature_Current" in df.columns:

    temperature_column = "Temperature_Current"

else:

    raise ValueError(
        "\nNo temperature column found.\n"
        "Expected one of:\n"
        "Temperature_C\n"
        "Temperature_Current"
    )


df["Temperature_C"] = pd.to_numeric(
    df[temperature_column],
    errors="coerce"
)


# ============================================================================
# 8. NORMALIZE NDVI + DATES
# ============================================================================

df["NDVI"] = pd.to_numeric(
    df["NDVI"],
    errors="coerce"
)

df["Week_Start"] = pd.to_datetime(
    df["Week_Start"],
    errors="coerce"
)

if df["Week_Start"].isna().any():

    invalid_dates = int(
        df["Week_Start"].isna().sum()
    )

    raise ValueError(
        f"Found {invalid_dates} invalid Week_Start values."
    )


# ============================================================================
# 9. OPTIONAL METADATA NORMALIZATION
# ============================================================================

for column in [
    "latitude",
    "longitude",
    "Latitude",
    "Longitude",
]:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


if "district" not in df.columns:

    df["district"] = np.nan


# ============================================================================
# 10. CREATE CANONICAL COORDINATE COLUMNS
# ============================================================================

if "Latitude" not in df.columns:

    if "latitude" in df.columns:

        df["Latitude"] = df["latitude"]


if "Longitude" not in df.columns:

    if "longitude" in df.columns:

        df["Longitude"] = df["longitude"]


if "Latitude" not in df.columns:

    raise ValueError(
        "Latitude/latitude column not found."
    )


if "Longitude" not in df.columns:

    raise ValueError(
        "Longitude/longitude column not found."
    )


# ============================================================================
# 11. CHECK DUPLICATES
# ============================================================================

duplicate_count = (
    df
    .duplicated(
        subset=[
            "point_id",
            "Week_Start",
        ]
    )
    .sum()
)

print()
print(
    f"Duplicate point-week rows: "
    f"{duplicate_count}"
)

if duplicate_count != 0:

    raise ValueError(
        "Duplicate point_id + Week_Start rows detected."
    )

print(
    "PASS - No duplicate point-week rows"
)


# ============================================================================
# 12. SAVE STATIC POINT METADATA BEFORE GRID EXPANSION
# ============================================================================

print()
print("=" * 90)
print("CAPTURING POINT METADATA")
print("=" * 90)


metadata_columns = [
    "district",
    "Latitude",
    "Longitude",
]


metadata_lookup = (
    df[
        [
            "point_id"
        ]
        + metadata_columns
    ]
    .drop_duplicates(
        subset=[
            "point_id"
        ]
    )
    .copy()
)


# If a metadata value is missing on the first occurrence,
# recover it from any valid row belonging to the point.
for column in metadata_columns:

    valid_lookup = (
        df[
            [
                "point_id",
                column,
            ]
        ]
        .dropna(
            subset=[
                column
            ]
        )
        .drop_duplicates(
            subset=[
                "point_id"
            ]
        )
    )

    lookup_dict = dict(
        zip(
            valid_lookup[
                "point_id"
            ],
            valid_lookup[
                column
            ],
        )
    )

    metadata_lookup[
        column
    ] = metadata_lookup[
        "point_id"
    ].map(
        lookup_dict
    )


print(
    f"Unique points: "
    f"{df['point_id'].nunique():,}"
)


# ============================================================================
# 13. SORT ORIGINAL DATA
# ============================================================================

df = (
    df
    .sort_values(
        [
            "point_id",
            "Week_Start",
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================================
# 14. BUILD COMPLETE CALENDAR-WEEK GRID
# ============================================================================
#
# This is critical for correct lags.
#
# Example:
#
#   2026-08-27 -> available
#   2026-09-03 -> missing
#   2026-09-10 -> available
#
# The 2026-09-10 NDVI_lag_1 must be NaN.
#
# It must NOT use 2026-08-27 as lag_1.
# ============================================================================

print()
print("=" * 90)
print("BUILDING CALENDAR-WEEK GRID")
print("=" * 90)


point_ids = (
    df[
        "point_id"
    ]
    .drop_duplicates()
    .sort_values()
    .tolist()
)

week_starts = (
    df[
        "Week_Start"
    ]
    .drop_duplicates()
    .sort_values()
    .tolist()
)


print()
print(
    f"Points: "
    f"{len(point_ids):,}"
)

print(
    f"Observed weekly dates: "
    f"{len(week_starts):,}"
)


if not point_ids:

    raise ValueError(
        "No points found."
    )


if not week_starts:

    raise ValueError(
        "No weekly dates found."
    )


# Create complete point × week grid.
full_index = pd.MultiIndex.from_product(
    [
        point_ids,
        week_starts,
    ],
    names=[
        "point_id",
        "Week_Start",
    ]
)


# Keep observation values only in the original dataframe.
# Metadata will be restored afterward.
grid_df = (
    df
    .set_index(
        [
            "point_id",
            "Week_Start",
        ]
    )
    .reindex(
        full_index
    )
    .reset_index()
)


# ============================================================================
# 15. RESTORE STATIC METADATA
# ============================================================================

metadata_map = (
    metadata_lookup
    .set_index(
        "point_id"
    )
)


for column in metadata_columns:

    grid_df[column] = (
        grid_df[
            "point_id"
        ]
        .map(
            metadata_map[
                column
            ]
        )
    )


# ============================================================================
# 16. CREATE TEMPORAL FEATURES
# ============================================================================

print()
print(
    "Creating temporal features..."
)


grid_df["Year"] = (
    grid_df[
        "Week_Start"
    ]
    .dt.year
)


# The historical model's Week_Of_Year feature
# uses ISO calendar week.
grid_df["Week_Of_Year"] = (
    grid_df[
        "Week_Start"
    ]
    .dt.isocalendar()
    .week
    .astype(int)
)


grid_df["Month"] = (
    grid_df[
        "Week_Start"
    ]
    .dt.month
)


grid_df["Day_Of_Year"] = (
    grid_df[
        "Week_Start"
    ]
    .dt.dayofyear
)


grid_df["Year_Index"] = (
    grid_df[
        "Year"
    ]
    - 2017
)


grid_df["Week_Sin"] = (
    np.sin(
        2
        * np.pi
        * grid_df[
            "Week_Of_Year"
        ]
        / 52
    )
)


grid_df["Week_Cos"] = (
    np.cos(
        2
        * np.pi
        * grid_df[
            "Week_Of_Year"
        ]
        / 52
    )
)


grid_df["Month_Sin"] = (
    np.sin(
        2
        * np.pi
        * grid_df[
            "Month"
        ]
        / 12
    )
)


grid_df["Month_Cos"] = (
    np.cos(
        2
        * np.pi
        * grid_df[
            "Month"
        ]
        / 12
    )
)


# ============================================================================
# 17. CURRENT-WEEK FEATURES
# ============================================================================

grid_df["Rainfall_Current"] = (
    grid_df[
        "Rainfall_GSMaP"
    ]
)

grid_df["Temperature_Current"] = (
    grid_df[
        "Temperature_C"
    ]
)


# ============================================================================
# 18. GROUP DATA
# ============================================================================

grouped = grid_df.groupby(
    "point_id",
    sort=False
)


# ============================================================================
# 19. NDVI LAGS
# ============================================================================

print(
    "Creating NDVI lag features..."
)


for lag in [
    1,
    2,
    3,
    4,
    8,
]:

    grid_df[
        f"NDVI_lag_{lag}"
    ] = (
        grouped[
            "NDVI"
        ]
        .shift(
            lag
        )
    )


# ============================================================================
# 20. NDVI CHANGE FEATURES
# ============================================================================

grid_df[
    "NDVI_change_1w"
] = (
    grid_df[
        "NDVI"
    ]
    - grid_df[
        "NDVI_lag_1"
    ]
)


grid_df[
    "NDVI_change_2w"
] = (
    grid_df[
        "NDVI"
    ]
    - grid_df[
        "NDVI_lag_2"
    ]
)


grid_df[
    "NDVI_change_4w"
] = (
    grid_df[
        "NDVI"
    ]
    - grid_df[
        "NDVI_lag_4"
    ]
)


# ============================================================================
# 21. GSMaP RAINFALL LAGS
# ============================================================================

print(
    "Creating GSMaP rainfall features..."
)


for lag in [
    1,
    2,
    4,
]:

    grid_df[
        f"Rainfall_lag_{lag}"
    ] = (
        grouped[
            "Rainfall_GSMaP"
        ]
        .shift(
            lag
        )
    )


# Previous-week rainfall series.
rainfall_previous = (
    grouped[
        "Rainfall_GSMaP"
    ]
    .shift(
        1
    )
)


grid_df[
    "Rainfall_rolling_2w"
] = (
    rainfall_previous
    .groupby(
        grid_df[
            "point_id"
        ]
    )
    .transform(
        lambda s:
            s.rolling(
                window=2,
                min_periods=2
            ).sum()
    )
)


grid_df[
    "Rainfall_rolling_4w"
] = (
    rainfall_previous
    .groupby(
        grid_df[
            "point_id"
        ]
    )
    .transform(
        lambda s:
            s.rolling(
                window=4,
                min_periods=4
            ).sum()
    )
)


grid_df[
    "Rainfall_rolling_8w"
] = (
    rainfall_previous
    .groupby(
        grid_df[
            "point_id"
        ]
    )
    .transform(
        lambda s:
            s.rolling(
                window=8,
                min_periods=8
            ).sum()
    )
)


# ============================================================================
# 22. TEMPERATURE LAGS
# ============================================================================

print(
    "Creating temperature features..."
)


for lag in [
    1,
    2,
    4,
]:

    grid_df[
        f"Temperature_lag_{lag}"
    ] = (
        grouped[
            "Temperature_C"
        ]
        .shift(
            lag
        )
    )


temperature_previous = (
    grouped[
        "Temperature_C"
    ]
    .shift(
        1
    )
)


grid_df[
    "Temperature_rolling_2w"
] = (
    temperature_previous
    .groupby(
        grid_df[
            "point_id"
        ]
    )
    .transform(
        lambda s:
            s.rolling(
                window=2,
                min_periods=2
            ).mean()
    )
)


grid_df[
    "Temperature_rolling_4w"
] = (
    temperature_previous
    .groupby(
        grid_df[
            "point_id"
        ]
    )
    .transform(
        lambda s:
            s.rolling(
                window=4,
                min_periods=4
            ).mean()
    )
)


grid_df[
    "Temperature_rolling_8w"
] = (
    temperature_previous
    .groupby(
        grid_df[
            "point_id"
        ]
    )
    .transform(
        lambda s:
            s.rolling(
                window=8,
                min_periods=8
            ).mean()
    )
)


# ============================================================================
# 23. EXACT 33-MODEL-FEATURE LIST
# ============================================================================

feature_columns = [

    # Current state
    "NDVI",
    "Rainfall_Current",
    "Temperature_Current",

    # NDVI history
    "NDVI_lag_1",
    "NDVI_lag_2",
    "NDVI_lag_3",
    "NDVI_lag_4",
    "NDVI_lag_8",

    # NDVI changes
    "NDVI_change_1w",
    "NDVI_change_2w",
    "NDVI_change_4w",

    # GSMaP rainfall
    "Rainfall_lag_1",
    "Rainfall_lag_2",
    "Rainfall_lag_4",
    "Rainfall_rolling_2w",
    "Rainfall_rolling_4w",
    "Rainfall_rolling_8w",

    # Temperature
    "Temperature_lag_1",
    "Temperature_lag_2",
    "Temperature_lag_4",
    "Temperature_rolling_2w",
    "Temperature_rolling_4w",
    "Temperature_rolling_8w",

    # Temporal
    "Week_Of_Year",
    "Month",
    "Day_Of_Year",
    "Year_Index",
    "Week_Sin",
    "Week_Cos",
    "Month_Sin",
    "Month_Cos",

    # Spatial
    "Latitude",
    "Longitude",
]


print()
print(
    f"Model features: "
    f"{len(feature_columns)}"
)

if len(feature_columns) != 33:

    raise ValueError(
        "Feature count is not 33."
    )

print(
    "PASS - Exact 33-feature model structure"
)


# ============================================================================
# 24. FIND LATEST AVAILABLE WEEK
# ============================================================================
#
# The realtime dataset currently contains a fixed set of recent weekly
# windows. We use the latest calendar week represented in that dataset
# for every point.
# ============================================================================

latest_week = (
    grid_df[
        "Week_Start"
    ]
    .max()
)


print()
print("=" * 90)
print("LATEST REALTIME WEEK")
print("=" * 90)

print()
print(
    f"Latest Week_Start in realtime dataset: "
    f"{latest_week.date()}"
)


latest_df = (
    grid_df[
        grid_df[
            "Week_Start"
        ]
        == latest_week
    ]
    .copy()
)


latest_df = (
    latest_df
    .sort_values(
        "point_id"
    )
    .reset_index(
        drop=True
    )
)


print(
    f"Latest-week point rows: "
    f"{len(latest_df):,}"
)

print(
    f"Unique points: "
    f"{latest_df['point_id'].nunique():,}"
)


# ============================================================================
# 25. VERIFY POINT COVERAGE
# ============================================================================

expected_points = (
    len(
        point_ids
    )
)

actual_points = (
    latest_df[
        "point_id"
    ]
    .nunique()
)

if actual_points != expected_points:

    raise ValueError(
        f"Latest-week point coverage mismatch. "
        f"Expected {expected_points}, "
        f"got {actual_points}."
    )


print(
    "PASS - All realtime points represented in latest week"
)


# ============================================================================
# 26. ACTUAL DATA FRESHNESS
# ============================================================================

print()
print("=" * 90)
print("DATA FRESHNESS")
print("=" * 90)


# Use actual current machine date.
current_date = (
    pd.Timestamp(
        datetime.now().date()
    )
)


latest_df[
    "Data_Age_Days"
] = (
    current_date
    - latest_df[
        "Week_Start"
    ]
).dt.days


def freshness_status(
    age_days
):

    if age_days <= 14:

        return "READY"

    if age_days <= 28:

        return "STALE"

    return "VERY_STALE"


latest_df[
    "Freshness_Status"
] = (
    latest_df[
        "Data_Age_Days"
    ]
    .apply(
        freshness_status
    )
)


# ============================================================================
# 27. CURRENT INPUT AVAILABILITY
# ============================================================================

latest_df[
    "NDVI_Available"
] = (
    latest_df[
        "NDVI"
    ]
    .notna()
)


latest_df[
    "Rainfall_Available"
] = (
    latest_df[
        "Rainfall_GSMaP"
    ]
    .notna()
)


latest_df[
    "Temperature_Available"
] = (
    latest_df[
        "Temperature_C"
    ]
    .notna()
)


latest_df[
    "Current_Inputs_Complete"
] = (
    latest_df[
        [
            "NDVI_Available",
            "Rainfall_Available",
            "Temperature_Available",
        ]
    ]
    .all(
        axis=1
    )
)


# ============================================================================
# 28. PREDICTION ELIGIBILITY
# ============================================================================
#
# Current NDVI is mandatory because it is an explicit model feature and is
# also required for the operational interpretation of predicted change.
#
# Current GSMaP rainfall and current temperature are mandatory as well.
#
# Lag/rolling features are NOT required to be complete because the saved
# model's training-time SimpleImputer handles missing model inputs.
# ============================================================================

latest_df[
    "Prediction_Eligible"
] = (
    latest_df[
        "Current_Inputs_Complete"
    ]
)


def prediction_status(
    row
):

    if not row[
        "Current_Inputs_Complete"
    ]:

        return "INCOMPLETE_INPUTS"

    if row[
        "Freshness_Status"
    ] == "VERY_STALE":

        return "VERY_STALE"

    if row[
        "Freshness_Status"
    ] == "STALE":

        return "STALE"

    return "READY"


latest_df[
    "Prediction_Status"
] = latest_df.apply(
    prediction_status,
    axis=1
)


# ============================================================================
# 29. GENERATE PREDICTIONS
# ============================================================================

print()
print("=" * 90)
print("GENERATING NEXT-WEEK NDVI PREDICTIONS")
print("=" * 90)


eligible_mask = (
    latest_df[
        "Prediction_Eligible"
    ]
)


eligible_df = (
    latest_df[
        eligible_mask
    ]
    .copy()
)


print()
print(
    f"Prediction-eligible points: "
    f"{len(eligible_df):,}"
)


# We keep a prediction column for every latest point.
# Ineligible points receive NaN.

latest_df[
    "Predicted_NDVI_Next_Week"
] = np.nan


if len(
    eligible_df
) > 0:

    X_live = (
        eligible_df[
            feature_columns
        ]
        .copy()
    )

    print()
    print(
        "Running saved GSMaP prediction pipeline..."
    )

    live_predictions = (
        model.predict(
            X_live
        )
    )

    # Map predictions back by point_id.
    prediction_map = dict(
        zip(
            eligible_df[
                "point_id"
            ],
            live_predictions,
        )
    )

    latest_df[
        "Predicted_NDVI_Next_Week"
    ] = (
        latest_df[
            "point_id"
        ]
        .map(
            prediction_map
        )
    )

else:

    print()
    print(
        "WARNING - No point currently has "
        "complete current inputs."
    )


# ============================================================================
# 30. PREDICTED CHANGE
# ============================================================================

latest_df[
    "Predicted_NDVI_Change"
] = (
    latest_df[
        "Predicted_NDVI_Next_Week"
    ]
    - latest_df[
        "NDVI"
    ]
)


latest_df[
    "Predicted_NDVI_Change_Percent"
] = np.where(

    latest_df[
        "NDVI"
    ].abs() > 1e-8,

    (
        latest_df[
            "Predicted_NDVI_Change"
        ]
        / latest_df[
            "NDVI"
        ]
        * 100
    ),

    np.nan
)


# ============================================================================
# 31. OPERATIONAL TREND STATUS
# ============================================================================
#
# These labels are NOT disease diagnoses.
# They only describe predicted NDVI direction/magnitude.
#
# <= -0.10 -> HIGH_STRESS_RISK
# <= -0.05 -> WATCH
# >= +0.05 -> IMPROVING
# otherwise -> STABLE
# ============================================================================

def interpret_trend(
    row
):

    prediction = (
        row[
            "Predicted_NDVI_Next_Week"
        ]
    )

    current = (
        row[
            "NDVI"
        ]
    )

    if pd.isna(
        prediction
    ):

        return "NO_PREDICTION"

    if pd.isna(
        current
    ):

        return "NO_CURRENT_NDVI"

    change = (
        prediction
        - current
    )

    if change <= -0.10:

        return "HIGH_STRESS_RISK"

    if change <= -0.05:

        return "WATCH"

    if change >= 0.05:

        return "IMPROVING"

    return "STABLE"


latest_df[
    "Operational_Trend_Status"
] = latest_df.apply(
    interpret_trend,
    axis=1
)


# ============================================================================
# 32. PREDICTION RANGE CHECK
# ============================================================================

valid_predictions = (
    latest_df[
        "Predicted_NDVI_Next_Week"
    ]
    .dropna()
)


print()
print("=" * 90)
print("PREDICTION VALIDATION")
print("=" * 90)


if len(
    valid_predictions
) > 0:

    prediction_min = (
        valid_predictions
        .min()
    )

    prediction_max = (
        valid_predictions
        .max()
    )

    print()
    print(
        f"Prediction minimum: "
        f"{prediction_min:.6f}"
    )

    print(
        f"Prediction maximum: "
        f"{prediction_max:.6f}"
    )

    if (
        prediction_min < -1
        or prediction_max > 1
    ):

        print()
        print(
            "WARNING - Prediction outside expected NDVI range [-1, 1]."
        )

    else:

        print(
            "PASS - Predictions within NDVI range [-1, 1]"
        )

else:

    print()
    print(
        "No predictions available for range validation."
    )


# ============================================================================
# 33. FINAL OUTPUT COLUMNS
# ============================================================================

output_columns = [

    "point_id",

    "Week_Start",

    "Year",

    "district",

    "Latitude",

    "Longitude",

    "NDVI",

    "Rainfall_GSMaP",

    "Temperature_C",

    "NDVI_lag_1",

    "NDVI_lag_2",

    "NDVI_lag_4",

    "NDVI_lag_8",

    "Rainfall_lag_1",

    "Rainfall_lag_2",

    "Rainfall_lag_4",

    "Rainfall_rolling_2w",

    "Rainfall_rolling_4w",

    "Rainfall_rolling_8w",

    "Temperature_lag_1",

    "Temperature_lag_2",

    "Temperature_lag_4",

    "Temperature_rolling_2w",

    "Temperature_rolling_4w",

    "Temperature_rolling_8w",

    "Predicted_NDVI_Next_Week",

    "Predicted_NDVI_Change",

    "Predicted_NDVI_Change_Percent",

    "Data_Age_Days",

    "Freshness_Status",

    "NDVI_Available",

    "Rainfall_Available",

    "Temperature_Available",

    "Current_Inputs_Complete",

    "Prediction_Eligible",

    "Prediction_Status",

    "Operational_Trend_Status",
]


prediction_output = (
    latest_df[
        output_columns
    ]
    .sort_values(
        "point_id"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================================
# 34. SAVE PREDICTIONS
# ============================================================================

prediction_output.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================================
# 35. SUMMARY COUNTS
# ============================================================================

total_points = (
    len(
        prediction_output
    )
)

ready_points = int(
    (
        prediction_output[
            "Prediction_Status"
        ]
        == "READY"
    )
    .sum()
)

stale_points = int(
    (
        prediction_output[
            "Prediction_Status"
        ]
        == "STALE"
    )
    .sum()
)

very_stale_points = int(
    (
        prediction_output[
            "Prediction_Status"
        ]
        == "VERY_STALE"
    )
    .sum()
)

incomplete_points = int(
    (
        prediction_output[
            "Prediction_Status"
        ]
        == "INCOMPLETE_INPUTS"
    )
    .sum()
)

prediction_count = int(
    prediction_output[
        "Predicted_NDVI_Next_Week"
    ]
    .notna()
    .sum()
)


# ============================================================================
# 36. TREND COUNTS
# ============================================================================

trend_counts = (
    prediction_output[
        "Operational_Trend_Status"
    ]
    .value_counts(
        dropna=False
    )
    .to_dict()
)


# ============================================================================
# 37. CONSOLE SUMMARY
# ============================================================================

print()
print("=" * 90)
print("REALTIME PREDICTION SUMMARY")
print("=" * 90)

print()
print(
    f"Current date: "
    f"{current_date.date()}"
)

print(
    f"Latest realtime Week_Start: "
    f"{latest_week.date()}"
)

print(
    f"Total points: "
    f"{total_points}"
)

print(
    f"READY: "
    f"{ready_points}"
)

print(
    f"STALE: "
    f"{stale_points}"
)

print(
    f"VERY_STALE: "
    f"{very_stale_points}"
)

print(
    f"INCOMPLETE_INPUTS: "
    f"{incomplete_points}"
)

print(
    f"Predictions generated: "
    f"{prediction_count}"
)

print()
print(
    "Operational trend counts:"
)

for status, count in sorted(
    trend_counts.items()
):

    print(
        f"  {status}: {count}"
    )


# ============================================================================
# 38. PREVIEW
# ============================================================================

print()
print("=" * 90)
print("PREDICTION PREVIEW")
print("=" * 90)

preview_columns = [

    "point_id",

    "district",

    "Week_Start",

    "NDVI",

    "Rainfall_GSMaP",

    "Temperature_C",

    "Predicted_NDVI_Next_Week",

    "Predicted_NDVI_Change",

    "Data_Age_Days",

    "Freshness_Status",

    "Prediction_Status",

    "Operational_Trend_Status",
]


print()

print(
    prediction_output[
        preview_columns
    ]
    .head(30)
    .to_string(
        index=False
    )
)


# ============================================================================
# 39. SPECIAL CHECKS FOR KAVATHE + LATUR
# ============================================================================

print()
print("=" * 90)
print("CASE-STUDY POINT CHECKS")
print("=" * 90)


# Kavathe-Khanapur case-study point.
kavathe_rows = (
    prediction_output[
        prediction_output[
            "point_id"
        ] == "P147"
    ]
)


print()
print(
    "Kavathe-Khanapur / P147:"
)

if len(
    kavathe_rows
) > 0:

    print(
        kavathe_rows[
            preview_columns
        ].to_string(
            index=False
        )
    )

else:

    print(
        "P147 not found."
    )


# Latur points.
latur_rows = (
    prediction_output[
        prediction_output[
            "district"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
        == "latur"
    ]
)


print()
print(
    f"Latur points: "
    f"{len(latur_rows)}"
)


if len(
    latur_rows
) > 0:

    print()

    print(
        latur_rows[
            preview_columns
        ]
        .head(25)
        .to_string(
            index=False
        )
    )


# ============================================================================
# 40. WRITE SUMMARY REPORT
# ============================================================================

summary_lines = [

    "AGRIVISION AI - REALTIME GSMaP PREDICTION REPORT",

    "2026",

    "",

    "REALTIME INPUT",

    str(
        REALTIME_DATA_FILE
    ),

    "",

    "MODEL",

    str(
        MODEL_FILE
    ),

    "",

    "CURRENT DATE",

    str(
        current_date.date()
    ),

    "",

    "LATEST REALTIME WEEK",

    str(
        latest_week.date()
    ),

    "",

    f"Total points: {total_points}",

    f"READY: {ready_points}",

    f"STALE: {stale_points}",

    f"VERY_STALE: {very_stale_points}",

    f"INCOMPLETE_INPUTS: {incomplete_points}",

    f"Predictions generated: {prediction_count}",

    "",

    "MODEL FEATURES",

    "33 features",

    "",

    "RAINFALL SOURCE",

    "GSMaP",

    "",

    "PREDICTION ELIGIBILITY",

    "Current NDVI + current GSMaP rainfall + current temperature",

    "must all be available.",

    "",

    "MISSING FEATURE POLICY",

    "Missing lag/rolling features are preserved.",

    "The saved training-time SimpleImputer handles model-feature missingness.",

    "No new live-data imputer is fitted.",

    "",

    "NO DATA FABRICATION",

    "No NDVI values are fabricated.",

    "No rainfall values are fabricated.",

    "No temperature values are fabricated.",

    "",

    "OPERATIONAL TREND STATUS",

    "HIGH_STRESS_RISK: predicted NDVI change <= -0.10",

    "WATCH: predicted NDVI change <= -0.05",

    "IMPROVING: predicted NDVI change >= +0.05",

    "STABLE: otherwise",

    "",

    "IMPORTANT",

    "Operational trend labels are decision-support signals only.",

    "They are not validated disease diagnoses or ground-truth crop-health classes.",

    "",

    "OUTPUT",

    str(
        OUTPUT_FILE
    ),
]


with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(
            summary_lines
        )
    )


# ============================================================================
# 41. FINAL
# ============================================================================

print()
print("=" * 90)
print("REALTIME GSMaP PREDICTION COMPLETED")
print("=" * 90)

print()
print(
    "Prediction output:"
)

print(
    OUTPUT_FILE
)

print()
print(
    "Summary report:"
)

print(
    SUMMARY_FILE
)

print()
print("=" * 90)
print("DONE")
print("=" * 90)