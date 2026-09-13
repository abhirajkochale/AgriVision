from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


# =============================================================================
# AGRIVISION AI
# FINAL OPERATIONAL REPORT GENERATOR
# =============================================================================
#
# PURPOSE
# -------
# Combine the existing 2026 realtime monitoring outputs into:
#
#   1. A complete point-level operational status CSV
#   2. A compact operational summary
#   3. A case-study CSV for Latur + Kavathe/Khanapur
#   4. A metrics CSV containing operational counts
#
# IMPORTANT
# ---------
# This script DOES NOT:
#   - retrain the ML model
#   - modify predictions
#   - impute missing NDVI
#   - fabricate realtime observations
#   - change prediction eligibility
#
# It only aggregates the outputs already produced by:
#
#   realtime_data.py
#   realtime_status.py
#   realtime_prediction.py
#
# =============================================================================


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REALTIME_DIR = (
    PROJECT_ROOT
    / "ml"
    / "realtime"
)

DATA_DIR = (
    REALTIME_DIR
    / "data"
)

STATUS_DIR = (
    REALTIME_DIR
    / "status"
)

PREDICTIONS_DIR = (
    REALTIME_DIR
    / "predictions"
)

OUTPUT_DIR = (
    REALTIME_DIR
    / "final_report"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 2. INPUT FILES
# =============================================================================

REALTIME_DATA_FILE = (
    DATA_DIR
    / "AgriVision_Realtime_Weekly_Kavathe_Latur.csv"
)

STATUS_FILE = (
    STATUS_DIR
    / "realtime_point_status.csv"
)

PREDICTION_FILE = (
    PREDICTIONS_DIR
    / "realtime_predictions_2026.csv"
)


# =============================================================================
# 3. OUTPUT FILES
# =============================================================================

OPERATIONAL_STATUS_FILE = (
    OUTPUT_DIR
    / "AgriVision_Operational_Status_2026.csv"
)

OPERATIONAL_SUMMARY_FILE = (
    OUTPUT_DIR
    / "AgriVision_Operational_Summary.txt"
)

CASE_STUDY_FILE = (
    OUTPUT_DIR
    / "AgriVision_Case_Study_2026.csv"
)

OPERATIONAL_METRICS_FILE = (
    OUTPUT_DIR
    / "AgriVision_Operational_Metrics.csv"
)


# =============================================================================
# 4. HELPER FUNCTIONS
# =============================================================================

def require_file(
    path: Path,
    description: str,
):
    """
    Verify that an expected input file exists.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"\n{description} not found:\n"
            f"{path}"
        )


def normalize_point_id(
    series: pd.Series,
) -> pd.Series:
    """
    Normalize point IDs consistently.
    """

    return (
        series
        .astype(str)
        .str.strip()
    )


def clean_dataframe_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Strip whitespace from column names.
    """

    df = df.copy()

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    return df


def first_existing_column(
    df: pd.DataFrame,
    candidates,
    required=True,
):
    """
    Return the first matching column from candidates.
    """

    for column in candidates:

        if column in df.columns:

            return column

    if required:

        raise ValueError(
            "\nNone of these columns were found:\n"
            + "\n".join(
                f"  - {column}"
                for column in candidates
            )
        )

    return None


def numeric_or_nan(
    series: pd.Series,
) -> pd.Series:

    return pd.to_numeric(
        series,
        errors="coerce",
    )


# =============================================================================
# 5. LOAD INPUTS
# =============================================================================

print(
    "=" * 90
)

print(
    "AGRIVISION AI - FINAL OPERATIONAL REPORT"
)

print(
    "2026 OPERATIONAL SUMMARY"
)

print(
    "=" * 90
)


print()
print(
    "Checking input files..."
)


require_file(
    REALTIME_DATA_FILE,
    "Realtime merged dataset",
)

require_file(
    STATUS_FILE,
    "Realtime status file",
)

require_file(
    PREDICTION_FILE,
    "Realtime prediction file",
)


print(
    "PASS - All operational input files found."
)


# =============================================================================
# 6. READ DATA
# =============================================================================

print()
print(
    "=" * 90
)

print(
    "LOADING OPERATIONAL DATA"
)

print(
    "=" * 90
)


realtime_df = pd.read_csv(
    REALTIME_DATA_FILE
)

status_df = pd.read_csv(
    STATUS_FILE
)

prediction_df = pd.read_csv(
    PREDICTION_FILE
)


realtime_df = clean_dataframe_columns(
    realtime_df
)

status_df = clean_dataframe_columns(
    status_df
)

prediction_df = clean_dataframe_columns(
    prediction_df
)


# Normalize point IDs.
for df in [
    realtime_df,
    status_df,
    prediction_df,
]:

    if "point_id" in df.columns:

        df["point_id"] = normalize_point_id(
            df["point_id"]
        )


print()
print(
    f"Realtime rows:     {len(realtime_df):,}"
)

print(
    f"Status rows:       {len(status_df):,}"
)

print(
    f"Prediction rows:   {len(prediction_df):,}"
)


# =============================================================================
# 7. NORMALIZE DATE COLUMNS
# =============================================================================

if "Week_Start" in realtime_df.columns:

    realtime_df[
        "Week_Start"
    ] = pd.to_datetime(
        realtime_df[
            "Week_Start"
        ],
        errors="coerce",
    )


if "latest_ndvi_week" in status_df.columns:

    status_df[
        "latest_ndvi_week"
    ] = pd.to_datetime(
        status_df[
            "latest_ndvi_week"
        ],
        errors="coerce",
    )


if "latest_rainfall_week" in status_df.columns:

    status_df[
        "latest_rainfall_week"
    ] = pd.to_datetime(
        status_df[
            "latest_rainfall_week"
        ],
        errors="coerce",
    )


if "latest_temperature_week" in status_df.columns:

    status_df[
        "latest_temperature_week"
    ] = pd.to_datetime(
        status_df[
            "latest_temperature_week"
        ],
        errors="coerce",
    )


if "common_latest_week" in status_df.columns:

    status_df[
        "common_latest_week"
    ] = pd.to_datetime(
        status_df[
            "common_latest_week"
        ],
        errors="coerce",
    )


if "Week_Start" in prediction_df.columns:

    prediction_df[
        "Week_Start"
    ] = pd.to_datetime(
        prediction_df[
            "Week_Start"
        ],
        errors="coerce",
    )


# =============================================================================
# 8. VALIDATE CORE STRUCTURE
# =============================================================================

required_realtime = [
    "point_id",
    "Week_Start",
    "NDVI",
    "Rainfall_mm",
    "Temperature_C",
]


missing_realtime = [
    column
    for column in required_realtime
    if column not in realtime_df.columns
]


if missing_realtime:

    raise ValueError(
        "\nRealtime dataset is missing required columns:\n"
        + "\n".join(
            missing_realtime
        )
    )


if "point_id" not in status_df.columns:

    raise ValueError(
        "Status dataset does not contain point_id."
    )


if "point_id" not in prediction_df.columns:

    raise ValueError(
        "Prediction dataset does not contain point_id."
    )


# =============================================================================
# 9. DETERMINE CURRENT OPERATIONAL WEEK
# =============================================================================

latest_realtime_week = (
    realtime_df[
        "Week_Start"
    ]
    .max()
)


if pd.isna(
    latest_realtime_week
):

    raise ValueError(
        "Could not determine latest realtime week."
    )


latest_realtime_data = (
    realtime_df[
        realtime_df[
            "Week_Start"
        ]
        == latest_realtime_week
    ]
    .copy()
)


latest_realtime_data = (
    latest_realtime_data
    .sort_values(
        "point_id"
    )
    .reset_index(
        drop=True
    )
)


# =============================================================================
# 10. KEEP ONE STATUS RECORD PER POINT
# =============================================================================

status_keep_columns = [
    "point_id",
]


optional_status_columns = [
    "district",
    "latest_ndvi_week",
    "latest_rainfall_week",
    "latest_temperature_week",
    "common_latest_week",
    "data_age_days",
    "current_ndvi",
    "current_rainfall",
    "current_temperature",
    "freshness_status",
    "prediction_eligibility",
    "reason",
]


for column in optional_status_columns:

    if column in status_df.columns:

        status_keep_columns.append(
            column
        )


status_use = (
    status_df[
        status_keep_columns
    ]
    .drop_duplicates(
        subset=[
            "point_id"
        ]
    )
    .copy()
)


# =============================================================================
# 11. KEEP ONE PREDICTION RECORD PER POINT
# =============================================================================

prediction_columns = [
    "point_id",
]


optional_prediction_columns = [
    "Week_Start",
    "NDVI",
    "Rainfall_GSMaP",
    "Temperature_C",
    "Predicted_NDVI_Next_Week",
    "Predicted_NDVI_Change",
    "Predicted_NDVI_Change_Percent",
    "Data_Age_Days",
    "Freshness_Status",
    "Prediction_Status",
    "Operational_Trend_Status",
]


for column in optional_prediction_columns:

    if column in prediction_df.columns:

        prediction_columns.append(
            column
        )


prediction_use = (
    prediction_df[
        prediction_columns
    ]
    .drop_duplicates(
        subset=[
            "point_id"
        ]
    )
    .copy()
)


# =============================================================================
# 12. PREPARE CURRENT WEEK OBSERVATIONS
# =============================================================================

current_columns = [
    "point_id",
    "Week_Start",
    "NDVI",
    "Rainfall_mm",
    "Temperature_C",
]


metadata_columns = [
    "latitude",
    "longitude",
    "district",
    "sampling_priority",
    "region_type",
]


for column in metadata_columns:

    if column in latest_realtime_data.columns:

        current_columns.append(
            column
        )


current_use = (
    latest_realtime_data[
        current_columns
    ]
    .copy()
)


# Rename the current-week rainfall to match the operational terminology.
current_use = (
    current_use
    .rename(
        columns={
            "Rainfall_mm":
            "Current_GSMaP_Rainfall_mm"
        }
    )
)


current_use = (
    current_use
    .rename(
        columns={
            "NDVI":
            "Current_NDVI",
            "Temperature_C":
            "Current_Temperature_C",
        }
    )
)


# =============================================================================
# 13. MERGE CURRENT DATA + STATUS + PREDICTIONS
# =============================================================================

print()
print(
    "=" * 90
)

print(
    "BUILDING OPERATIONAL POINT TABLE"
)

print(
    "=" * 90
)


operational_df = (
    current_use
    .merge(
        status_use,
        on="point_id",
        how="left",
        suffixes=(
            "",
            "_status",
        ),
    )
)


operational_df = (
    operational_df
    .merge(
        prediction_use,
        on="point_id",
        how="left",
        suffixes=(
            "",
            "_prediction",
        ),
    )
)


# =============================================================================
# 14. RESOLVE DUPLICATE / OVERLAPPING FIELDS
# =============================================================================

# District
if (
    "district" not in operational_df.columns
    and "district_status" in operational_df.columns
):

    operational_df[
        "district"
    ] = operational_df[
        "district_status"
    ]


# Current NDVI
if (
    "Current_NDVI" not in operational_df.columns
    and "current_ndvi" in operational_df.columns
):

    operational_df[
        "Current_NDVI"
    ] = operational_df[
        "current_ndvi"
    ]


# Current rainfall
if (
    "Current_GSMaP_Rainfall_mm" not in operational_df.columns
    and "current_rainfall" in operational_df.columns
):

    operational_df[
        "Current_GSMaP_Rainfall_mm"
    ] = operational_df[
        "current_rainfall"
    ]


# Current temperature
if (
    "Current_Temperature_C" not in operational_df.columns
    and "current_temperature" in operational_df.columns
):

    operational_df[
        "Current_Temperature_C"
    ] = operational_df[
        "current_temperature"
    ]


# =============================================================================
# 15. ENSURE STANDARD OPERATIONAL COLUMNS
# =============================================================================

standard_defaults = {

    "Current_NDVI": np.nan,

    "Current_GSMaP_Rainfall_mm": np.nan,

    "Current_Temperature_C": np.nan,

    "latest_ndvi_week": pd.NaT,

    "latest_rainfall_week": pd.NaT,

    "latest_temperature_week": pd.NaT,

    "common_latest_week": pd.NaT,

    "data_age_days": np.nan,

    "freshness_status": "UNKNOWN",

    "prediction_eligibility": "NOT_READY",

    "reason": "",

    "Predicted_NDVI_Next_Week": np.nan,

    "Predicted_NDVI_Change": np.nan,

    "Predicted_NDVI_Change_Percent": np.nan,

    "Prediction_Status": "NO_PREDICTION",

    "Operational_Trend_Status": "NO_PREDICTION",

}


for column, default_value in standard_defaults.items():

    if column not in operational_df.columns:

        operational_df[
            column
        ] = default_value


# =============================================================================
# 16. NORMALIZE NUMERIC FIELDS
# =============================================================================

numeric_columns = [
    "Current_NDVI",
    "Current_GSMaP_Rainfall_mm",
    "Current_Temperature_C",
    "data_age_days",
    "Predicted_NDVI_Next_Week",
    "Predicted_NDVI_Change",
    "Predicted_NDVI_Change_Percent",
]


for column in numeric_columns:

    operational_df[
        column
    ] = numeric_or_nan(
        operational_df[
            column
        ]
    )


# =============================================================================
# 17. NORMALIZE STATUS LABELS
# =============================================================================

for column in [
    "freshness_status",
    "prediction_eligibility",
    "Prediction_Status",
    "Operational_Trend_Status",
]:

    if column in operational_df.columns:

        operational_df[
            column
        ] = (
            operational_df[
                column
            ]
            .fillna("")
            .astype(str)
            .str.strip()
        )


# =============================================================================
# 18. FIX PREDICTION FIELDS FROM PREDICTION OUTPUT
# =============================================================================

# Some fields may arrive with _prediction suffix if both current and
# prediction data contained the same column.

prediction_suffix_mapping = {

    "Prediction_Status_prediction":
        "Prediction_Status",

    "Operational_Trend_Status_prediction":
        "Operational_Trend_Status",

    "Predicted_NDVI_Next_Week_prediction":
        "Predicted_NDVI_Next_Week",

    "Predicted_NDVI_Change_prediction":
        "Predicted_NDVI_Change",

    "Predicted_NDVI_Change_Percent_prediction":
        "Predicted_NDVI_Change_Percent",

}


for source_column, target_column in prediction_suffix_mapping.items():

    if source_column in operational_df.columns:

        operational_df[
            target_column
        ] = operational_df[
            source_column
        ]


# =============================================================================
# 19. DETERMINE OPERATIONAL PREDICTION AVAILABLE FLAG
# =============================================================================

operational_df[
    "Prediction_Available"
] = (
    operational_df[
        "Predicted_NDVI_Next_Week"
    ]
    .notna()
)


operational_df[
    "Prediction_Available"
] = (
    operational_df[
        "Prediction_Available"
    ]
    .astype(bool)
)


# =============================================================================
# 20. CREATE CURRENT DATA QUALITY FLAGS
# =============================================================================

operational_df[
    "Current_NDVI_Available"
] = (
    operational_df[
        "Current_NDVI"
    ]
    .notna()
)


operational_df[
    "Current_GSMaP_Available"
] = (
    operational_df[
        "Current_GSMaP_Rainfall_mm"
    ]
    .notna()
)


operational_df[
    "Current_Temperature_Available"
] = (
    operational_df[
        "Current_Temperature_C"
    ]
    .notna()
)


operational_df[
    "Current_Inputs_Complete"
] = (
    operational_df[
        [
            "Current_NDVI_Available",
            "Current_GSMaP_Available",
            "Current_Temperature_Available",
        ]
    ]
    .all(
        axis=1
    )
)


# =============================================================================
# 21. OPERATIONAL CASE LABEL
# =============================================================================

def case_label(
    row,
):

    point_id = str(
        row[
            "point_id"
        ]
    )

    district = str(
        row.get(
            "district",
            ""
        )
    ).strip().lower()

    if point_id == "P147":

        return "Kavathe-Khanapur"

    if district == "latur":

        return "Latur"

    return "Other"


operational_df[
    "Case_Study_Group"
] = operational_df.apply(
    case_label,
    axis=1,
)


# =============================================================================
# 22. ORDER THE MAIN OUTPUT
# =============================================================================

main_columns = [
    "point_id",
    "district",
    "latitude",
    "longitude",
    "Case_Study_Group",
    "Week_Start",
    "Current_NDVI",
    "Current_GSMaP_Rainfall_mm",
    "Current_Temperature_C",
    "latest_ndvi_week",
    "latest_rainfall_week",
    "latest_temperature_week",
    "common_latest_week",
    "data_age_days",
    "freshness_status",
    "prediction_eligibility",
    "Current_NDVI_Available",
    "Current_GSMaP_Available",
    "Current_Temperature_Available",
    "Current_Inputs_Complete",
    "Prediction_Available",
    "Predicted_NDVI_Next_Week",
    "Predicted_NDVI_Change",
    "Predicted_NDVI_Change_Percent",
    "Prediction_Status",
    "Operational_Trend_Status",
    "reason",
]


existing_main_columns = [
    column
    for column in main_columns
    if column in operational_df.columns
]


operational_df = (
    operational_df[
        existing_main_columns
    ]
    .sort_values(
        [
            "Case_Study_Group",
            "point_id",
        ]
    )
    .reset_index(
        drop=True
    )
)


# =============================================================================
# 23. SAVE OPERATIONAL STATUS
# =============================================================================

operational_df.to_csv(
    OPERATIONAL_STATUS_FILE,
    index=False,
)


# =============================================================================
# 24. BUILD CASE-STUDY TABLE
# =============================================================================

case_study_df = (
    operational_df[
        operational_df[
            "Case_Study_Group"
        ]
        .isin(
            [
                "Latur",
                "Kavathe-Khanapur",
            ]
        )
    ]
    .copy()
)


case_study_columns = [
    "point_id",
    "district",
    "Case_Study_Group",
    "Current_NDVI",
    "Current_GSMaP_Rainfall_mm",
    "Current_Temperature_C",
    "Predicted_NDVI_Next_Week",
    "Predicted_NDVI_Change",
    "Predicted_NDVI_Change_Percent",
    "Prediction_Status",
    "Operational_Trend_Status",
    "freshness_status",
    "prediction_eligibility",
    "reason",
]


case_study_columns = [
    column
    for column in case_study_columns
    if column in case_study_df.columns
]


case_study_df = (
    case_study_df[
        case_study_columns
    ]
    .sort_values(
        [
            "Case_Study_Group",
            "point_id",
        ]
    )
    .reset_index(
        drop=True
    )
)


case_study_df.to_csv(
    CASE_STUDY_FILE,
    index=False,
)


# =============================================================================
# 25. OPERATIONAL COUNTS
# =============================================================================

total_points = int(
    operational_df[
        "point_id"
    ]
    .nunique()
)


prediction_count = int(
    operational_df[
        "Prediction_Available"
    ]
    .sum()
)


no_prediction_count = (
    total_points
    - prediction_count
)


ready_count = int(
    (
        operational_df[
            "Prediction_Status"
        ]
        == "READY"
    )
    .sum()
)


incomplete_count = int(
    (
        operational_df[
            "Prediction_Status"
        ]
        == "INCOMPLETE_INPUTS"
    )
    .sum()
)


stale_count = int(
    (
        operational_df[
            "Prediction_Status"
        ]
        == "STALE"
    )
    .sum()
)


very_stale_count = int(
    (
        operational_df[
            "Prediction_Status"
        ]
        == "VERY_STALE"
    )
    .sum()
)


# =============================================================================
# 26. FRESHNESS DISTRIBUTION
# =============================================================================

freshness_counts = (
    operational_df[
        "freshness_status"
    ]
    .replace(
        {
            "ready": "READY",
            "warning": "WARNING",
            "stale": "STALE",
            "very_stale": "VERY_STALE",
        }
    )
    .str.upper()
    .value_counts()
    .to_dict()
)


fresh_ready = int(
    freshness_counts.get(
        "READY",
        0
    )
)

fresh_warning = int(
    freshness_counts.get(
        "WARNING",
        0
    )
)

fresh_stale = int(
    freshness_counts.get(
        "STALE",
        0
    )
)

fresh_very_stale = int(
    freshness_counts.get(
        "VERY_STALE",
        0
    )
)


# =============================================================================
# 27. TREND DISTRIBUTION
# =============================================================================

trend_counts = (
    operational_df[
        "Operational_Trend_Status"
    ]
    .replace(
        {
            "": "NO_PREDICTION"
        }
    )
    .value_counts()
    .to_dict()
)


high_stress_count = int(
    trend_counts.get(
        "HIGH_STRESS_RISK",
        0
    )
)


watch_count = int(
    trend_counts.get(
        "WATCH",
        0
    )
)


stable_count = int(
    trend_counts.get(
        "STABLE",
        0
    )
)


improving_count = int(
    trend_counts.get(
        "IMPROVING",
        0
    )
)


no_prediction_trend_count = int(
    trend_counts.get(
        "NO_PREDICTION",
        0
    )
)


# =============================================================================
# 28. CURRENT OBSERVATION COVERAGE
# =============================================================================

current_ndvi_count = int(
    operational_df[
        "Current_NDVI_Available"
    ]
    .sum()
)


current_rainfall_count = int(
    operational_df[
        "Current_GSMaP_Available"
    ]
    .sum()
)


current_temperature_count = int(
    operational_df[
        "Current_Temperature_Available"
    ]
    .sum()
)


complete_current_count = int(
    operational_df[
        "Current_Inputs_Complete"
    ]
    .sum()
)


# =============================================================================
# 29. PREDICTION STATISTICS
# =============================================================================

predicted_values = (
    operational_df[
        "Predicted_NDVI_Next_Week"
    ]
    .dropna()
)


predicted_change_values = (
    operational_df[
        "Predicted_NDVI_Change"
    ]
    .dropna()
)


if len(
    predicted_values
) > 0:

    predicted_ndvi_min = float(
        predicted_values.min()
    )

    predicted_ndvi_max = float(
        predicted_values.max()
    )

    predicted_ndvi_mean = float(
        predicted_values.mean()
    )

else:

    predicted_ndvi_min = np.nan
    predicted_ndvi_max = np.nan
    predicted_ndvi_mean = np.nan


if len(
    predicted_change_values
) > 0:

    predicted_change_min = float(
        predicted_change_values.min()
    )

    predicted_change_max = float(
        predicted_change_values.max()
    )

    predicted_change_mean = float(
        predicted_change_values.mean()
    )

else:

    predicted_change_min = np.nan
    predicted_change_max = np.nan
    predicted_change_mean = np.nan


# =============================================================================
# 30. CASE-STUDY COUNTS
# =============================================================================

latur_count = int(
    (
        operational_df[
            "Case_Study_Group"
        ]
        == "Latur"
    )
    .sum()
)


kavathe_count = int(
    (
        operational_df[
            "Case_Study_Group"
        ]
        == "Kavathe-Khanapur"
    )
    .sum()
)


latur_prediction_count = int(
    (
        case_study_df[
            "Case_Study_Group"
        ]
        .eq("Latur")
        &
        case_study_df[
            "Prediction_Status"
        ]
        .eq("READY")
    )
    .sum()
)


p147_row = (
    operational_df[
        operational_df[
            "point_id"
        ]
        == "P147"
    ]
)


# =============================================================================
# 31. BUILD METRICS TABLE
# =============================================================================

metrics = [

    {
        "Metric":
        "Latest_Realtime_Week",

        "Value":
        latest_realtime_week.strftime(
            "%Y-%m-%d"
        ),
    },

    {
        "Metric":
        "Total_Monitored_Points",

        "Value":
        total_points,
    },

    {
        "Metric":
        "Latur_Points",

        "Value":
        latur_count,
    },

    {
        "Metric":
        "Kavathe_Khanapur_Points",

        "Value":
        kavathe_count,
    },

    {
        "Metric":
        "Current_NDVI_Available_Points",

        "Value":
        current_ndvi_count,
    },

    {
        "Metric":
        "Current_GSMaP_Available_Points",

        "Value":
        current_rainfall_count,
    },

    {
        "Metric":
        "Current_Temperature_Available_Points",

        "Value":
        current_temperature_count,
    },

    {
        "Metric":
        "Complete_Current_Input_Points",

        "Value":
        complete_current_count,
    },

    {
        "Metric":
        "Prediction_Ready_Points",

        "Value":
        ready_count,
    },

    {
        "Metric":
        "Predictions_Generated",

        "Value":
        prediction_count,
    },

    {
        "Metric":
        "No_Prediction_Points",

        "Value":
        no_prediction_count,
    },

    {
        "Metric":
        "Freshness_READY",

        "Value":
        fresh_ready,
    },

    {
        "Metric":
        "Freshness_WARNING",

        "Value":
        fresh_warning,
    },

    {
        "Metric":
        "Freshness_STALE",

        "Value":
        fresh_stale,
    },

    {
        "Metric":
        "Freshness_VERY_STALE",

        "Value":
        fresh_very_stale,
    },

    {
        "Metric":
        "Trend_HIGH_STRESS_RISK",

        "Value":
        high_stress_count,
    },

    {
        "Metric":
        "Trend_WATCH",

        "Value":
        watch_count,
    },

    {
        "Metric":
        "Trend_STABLE",

        "Value":
        stable_count,
    },

    {
        "Metric":
        "Trend_IMPROVING",

        "Value":
        improving_count,
    },

    {
        "Metric":
        "Trend_NO_PREDICTION",

        "Value":
        no_prediction_trend_count,
    },

    {
        "Metric":
        "Predicted_NDVI_Minimum",

        "Value":
        predicted_ndvi_min,
    },

    {
        "Metric":
        "Predicted_NDVI_Maximum",

        "Value":
        predicted_ndvi_max,
    },

    {
        "Metric":
        "Predicted_NDVI_Mean",

        "Value":
        predicted_ndvi_mean,
    },

    {
        "Metric":
        "Predicted_NDVI_Change_Minimum",

        "Value":
        predicted_change_min,
    },

    {
        "Metric":
        "Predicted_NDVI_Change_Maximum",

        "Value":
        predicted_change_max,
    },

    {
        "Metric":
        "Predicted_NDVI_Change_Mean",

        "Value":
        predicted_change_mean,
    },

        {
        "Metric":
        "P069_High_Stress_Risk",

        "Value":
        bool(
            (
                (
                    operational_df[
                        "point_id"
                    ] == "P069"
                )
                &
                (
                    operational_df[
                        "Operational_Trend_Status"
                    ]
                    == "HIGH_STRESS_RISK"
                )
            ).any()
        ),
    },

        {
        "Metric":
        "P077_Watch_Status",

        "Value":
        bool(
            (
                (
                    operational_df[
                        "point_id"
                    ] == "P077"
                )
                &
                (
                    operational_df[
                        "Operational_Trend_Status"
                    ]
                    == "WATCH"
                )
            ).any()
        ),
    },

    {
        "Metric":
        "P147_Prediction_Available",

        "Value":
        int(
            p147_row[
                "Prediction_Available"
            ].any()
        )
        if not p147_row.empty
        else False,
    },
]


metrics_df = pd.DataFrame(
    metrics
)


metrics_df.to_csv(
    OPERATIONAL_METRICS_FILE,
    index=False,
)


# =============================================================================
# 32. BUILD HUMAN-READABLE SUMMARY
# =============================================================================

run_timestamp = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S"
)


summary_lines = [

    "AGRIVISION AI",
    "FINAL 2026 OPERATIONAL REPORT",
    "=" * 80,
    "",
    f"Report generated: {run_timestamp}",
    (
        "Latest realtime week: "
        f"{latest_realtime_week.strftime('%Y-%m-%d')}"
    ),
    "",
    "MONITORING COVERAGE",
    "-" * 80,
    f"Total monitored points: {total_points}",
    f"Latur points: {latur_count}",
    f"Kavathe-Khanapur points: {kavathe_count}",
    "",
    "CURRENT DATA AVAILABILITY",
    "-" * 80,
    f"Current NDVI available: {current_ndvi_count}/{total_points}",
    (
        "Current GSMaP rainfall available: "
        f"{current_rainfall_count}/{total_points}"
    ),
    (
        "Current temperature available: "
        f"{current_temperature_count}/{total_points}"
    ),
    (
        "Complete current inputs: "
        f"{complete_current_count}/{total_points}"
    ),
    "",
    "PREDICTION STATUS",
    "-" * 80,
    f"Prediction-ready points: {ready_count}",
    f"Predictions generated: {prediction_count}",
    f"No-prediction points: {no_prediction_count}",
    "",
    "FRESHNESS DISTRIBUTION",
    "-" * 80,
    f"READY: {fresh_ready}",
    f"WARNING: {fresh_warning}",
    f"STALE: {fresh_stale}",
    f"VERY_STALE: {fresh_very_stale}",
    "",
    "OPERATIONAL TREND DISTRIBUTION",
    "-" * 80,
    f"HIGH_STRESS_RISK: {high_stress_count}",
    f"WATCH: {watch_count}",
    f"STABLE: {stable_count}",
    f"IMPROVING: {improving_count}",
    f"NO_PREDICTION: {no_prediction_trend_count}",
    "",
    "PREDICTION STATISTICS",
    "-" * 80,
    (
        "Predicted NDVI minimum: "
        f"{predicted_ndvi_min:.6f}"
        if not np.isnan(predicted_ndvi_min)
        else "Predicted NDVI minimum: N/A"
    ),
    (
        "Predicted NDVI maximum: "
        f"{predicted_ndvi_max:.6f}"
        if not np.isnan(predicted_ndvi_max)
        else "Predicted NDVI maximum: N/A"
    ),
    (
        "Predicted NDVI mean: "
        f"{predicted_ndvi_mean:.6f}"
        if not np.isnan(predicted_ndvi_mean)
        else "Predicted NDVI mean: N/A"
    ),
    (
        "Predicted NDVI change minimum: "
        f"{predicted_change_min:.6f}"
        if not np.isnan(predicted_change_min)
        else "Predicted NDVI change minimum: N/A"
    ),
    (
        "Predicted NDVI change maximum: "
        f"{predicted_change_max:.6f}"
        if not np.isnan(predicted_change_max)
        else "Predicted NDVI change maximum: N/A"
    ),
    (
        "Predicted NDVI change mean: "
        f"{predicted_change_mean:.6f}"
        if not np.isnan(predicted_change_mean)
        else "Predicted NDVI change mean: N/A"
    ),
    "",
    "CASE STUDIES",
    "-" * 80,
    (
        "P069: HIGH_STRESS_RISK"
        if (
            (
                operational_df[
                    "point_id"
                ]
                == "P069"
            )
            &
            (
                operational_df[
                    "Operational_Trend_Status"
                ]
                == "HIGH_STRESS_RISK"
            )
        ).any()
        else
        "P069: no HIGH_STRESS_RISK prediction in current report"
    ),
    (
        "P077: WATCH"
        if (
            (
                operational_df[
                    "point_id"
                ]
                == "P077"
            )
            &
            (
                operational_df[
                    "Operational_Trend_Status"
                ]
                == "WATCH"
            )
        ).any()
        else
        "P077: no WATCH prediction in current report"
    ),
    (
        "P147: prediction available"
        if (
            not p147_row.empty
            and
            p147_row[
                "Prediction_Available"
            ].any()
        )
        else
        "P147: NO_PREDICTION due to insufficient current NDVI availability"
    ),
    "",
    "INTERPRETATION",
    "-" * 80,
    (
        "Operational trend labels describe predicted NDVI direction "
        "and are decision-support signals only."
    ),
    (
        "They are not validated crop-disease diagnoses."
    ),
    (
        "Missing realtime observations are retained as missing; "
        "this report does not fabricate NDVI, rainfall or temperature."
    ),
    "",
    "OUTPUT FILES",
    "-" * 80,
    str(
        OPERATIONAL_STATUS_FILE
    ),
    str(
        CASE_STUDY_FILE
    ),
    str(
        OPERATIONAL_METRICS_FILE
    ),
    str(
        OPERATIONAL_SUMMARY_FILE
    ),
]


with open(
    OPERATIONAL_SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        "\n".join(
            summary_lines
        )
    )


# =============================================================================
# 33. CONSOLE OUTPUT
# =============================================================================

print()
print(
    "=" * 90
)

print(
    "FINAL OPERATIONAL SUMMARY"
)

print(
    "=" * 90
)

print()
print(
    f"Latest realtime week: "
    f"{latest_realtime_week.strftime('%Y-%m-%d')}"
)

print(
    f"Total monitored points: "
    f"{total_points}"
)

print(
    f"Prediction-ready points: "
    f"{ready_count}"
)

print(
    f"Predictions generated: "
    f"{prediction_count}"
)

print(
    f"High-stress-risk points: "
    f"{high_stress_count}"
)

print(
    f"Watch points: "
    f"{watch_count}"
)

print(
    f"Stable points: "
    f"{stable_count}"
)

print(
    f"No-prediction points: "
    f"{no_prediction_count}"
)


# =============================================================================
# 34. PREDICTION TABLE PREVIEW
# =============================================================================

print()
print(
    "=" * 90
)

print(
    "CURRENT PREDICTIONS"
)

print(
    "=" * 90
)

prediction_preview_columns = [
    "point_id",
    "district",
    "Current_NDVI",
    "Current_GSMaP_Rainfall_mm",
    "Current_Temperature_C",
    "Predicted_NDVI_Next_Week",
    "Predicted_NDVI_Change",
    "Operational_Trend_Status",
]


prediction_preview_columns = [
    column
    for column in prediction_preview_columns
    if column in operational_df.columns
]


current_predictions = (
    operational_df[
        operational_df[
            "Prediction_Available"
        ]
    ]
    [
        prediction_preview_columns
    ]
    .copy()
)


if current_predictions.empty:

    print(
        "No predictions currently available."
    )

else:

    print(
        current_predictions
        .to_string(
            index=False
        )
    )


# =============================================================================
# 35. FINAL OUTPUT PATHS
# =============================================================================

print()
print(
    "=" * 90
)

print(
    "FINAL OPERATIONAL REPORT COMPLETED"
)

print(
    "=" * 90
)

print()
print(
    "Operational status:"
)

print(
    OPERATIONAL_STATUS_FILE
)

print()
print(
    "Case study:"
)

print(
    CASE_STUDY_FILE
)

print()
print(
    "Metrics:"
)

print(
    OPERATIONAL_METRICS_FILE
)

print()
print(
    "Summary:"
)

print(
    OPERATIONAL_SUMMARY_FILE
)

print()
print(
    "=" * 90
)

print(
    "DONE"
)

print(
    "=" * 90
)