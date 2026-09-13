from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "ml"
    / "realtime"
    / "data"
    / "AgriVision_Realtime_Weekly_Kavathe_Latur.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "realtime"
    / "status"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

# Current date/time is taken from the machine in UTC.
# We evaluate freshness relative to the start of the
# latest completed weekly window.

# Maximum allowed age for a live prediction.
#
# 0-14 days  -> READY
# 15-28 days -> WARNING
# >28 days   -> STALE
#
# The 14-day rule is deliberately conservative because the
# model predicts the NEXT week and should use reasonably recent
# vegetation/weather information.

READY_MAX_AGE_DAYS = 14
WARNING_MAX_AGE_DAYS = 28


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "Week_Start",
    "Week_End",
    "Week_Number",
    "Year",
    "point_id",
    "latitude",
    "longitude",
    "district",
    "NDVI",
    "Rainfall_mm",
    "Temperature_C",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nReal-time data file not found:\n"
            f"{INPUT_FILE}\n\n"
            "Run realtime_data.py first."
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "\nMissing required columns:\n"
            + "\n".join(missing_columns)
        )

    df["Week_Start"] = pd.to_datetime(
        df["Week_Start"],
        errors="coerce",
    )

    df["Week_End"] = pd.to_datetime(
        df["Week_End"],
        errors="coerce",
    )

    numeric_columns = [
        "Week_Number",
        "Year",
        "latitude",
        "longitude",
        "NDVI",
        "Rainfall_mm",
        "Temperature_C",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df["point_id"] = (
        df["point_id"]
        .astype(str)
        .str.strip()
    )

    df["district"] = (
        df["district"]
        .astype(str)
        .str.strip()
    )

    return df


# ============================================================
# SOURCE STATUS FOR ONE POINT
# ============================================================

def get_source_latest_week(point_df, column):
    """
    Return the latest Week_Start where a source has
    a non-null value.
    """

    available = point_df[
        point_df[column].notna()
    ]

    if available.empty:
        return pd.NaT

    return available[
        "Week_Start"
    ].max()


def get_source_value_at_week(
    point_df,
    column,
    week_start,
):
    """
    Return the value for a source at a specific week.
    """

    rows = point_df[
        (
            point_df["Week_Start"]
            == week_start
        )
        & (
            point_df[column].notna()
        )
    ]

    if rows.empty:
        return np.nan

    return rows.iloc[0][column]


# ============================================================
# DETERMINE STATUS
# ============================================================

def classify_status(
    common_latest_week,
    today,
):
    if pd.isna(
        common_latest_week
    ):
        return (
            "NO_COMMON_DATA",
            np.nan,
        )

    age_days = (
        today
        - common_latest_week.date()
    ).days

    if age_days <= READY_MAX_AGE_DAYS:

        return (
            "READY",
            age_days,
        )

    if age_days <= WARNING_MAX_AGE_DAYS:

        return (
            "WARNING",
            age_days,
        )

    return (
        "STALE",
        age_days,
    )


# ============================================================
# BUILD POINT STATUS
# ============================================================

def build_status_table(
    df,
):
    today = datetime.now(
        timezone.utc
    ).date()

    rows = []

    for point_id, point_df in (
        df.groupby(
            "point_id"
        )
    ):

        point_df = (
            point_df
            .sort_values(
                "Week_Start"
            )
            .copy()
        )

        district = (
            point_df["district"]
            .iloc[0]
            if not point_df.empty
            else ""
        )

        latitude = (
            point_df["latitude"]
            .iloc[0]
            if not point_df.empty
            else np.nan
        )

        longitude = (
            point_df["longitude"]
            .iloc[0]
            if not point_df.empty
            else np.nan
        )

        # ----------------------------------------------------
        # Latest available week for each source
        # ----------------------------------------------------

        latest_ndvi = (
            get_source_latest_week(
                point_df,
                "NDVI",
            )
        )

        latest_rainfall = (
            get_source_latest_week(
                point_df,
                "Rainfall_mm",
            )
        )

        latest_temperature = (
            get_source_latest_week(
                point_df,
                "Temperature_C",
            )
        )

        # ----------------------------------------------------
        # Latest common week
        #
        # A common week requires all three sources.
        # ----------------------------------------------------

        source_weeks = []

        for source_column in [
            "NDVI",
            "Rainfall_mm",
            "Temperature_C",
        ]:

            valid_weeks = set(
                point_df.loc[
                    point_df[
                        source_column
                    ].notna(),
                    "Week_Start",
                ]
            )

            if not valid_weeks:

                source_weeks = []

                break

            if not source_weeks:

                source_weeks = valid_weeks

            else:

                source_weeks = (
                    source_weeks
                    & valid_weeks
                )

        if source_weeks:

            common_latest_week = max(
                source_weeks
            )

        else:

            common_latest_week = pd.NaT

        # ----------------------------------------------------
        # Current values at common week
        # ----------------------------------------------------

        current_ndvi = (
            get_source_value_at_week(
                point_df,
                "NDVI",
                common_latest_week,
            )
            if not pd.isna(
                common_latest_week
            )
            else np.nan
        )

        current_rainfall = (
            get_source_value_at_week(
                point_df,
                "Rainfall_mm",
                common_latest_week,
            )
            if not pd.isna(
                common_latest_week
            )
            else np.nan
        )

        current_temperature = (
            get_source_value_at_week(
                point_df,
                "Temperature_C",
                common_latest_week,
            )
            if not pd.isna(
                common_latest_week
            )
            else np.nan
        )

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        status, age_days = (
            classify_status(
                common_latest_week,
                today,
            )
        )

        # ----------------------------------------------------
        # Required historical range
        #
        # The model needs up to 8 weeks of lag/rolling data.
        # We calculate how many recent weekly windows are
        # available in the raw live table.
        # ----------------------------------------------------

        if not pd.isna(
            common_latest_week
        ):

            history_start = (
                common_latest_week
                - pd.Timedelta(
                    weeks=8
                )
            )

            recent_history = point_df[
                (
                    point_df[
                        "Week_Start"
                    ]
                    >= history_start
                )
                & (
                    point_df[
                        "Week_Start"
                    ]
                    <= common_latest_week
                )
            ]

        else:

            recent_history = (
                pd.DataFrame()
            )

        # Count usable observations in the recent history.
        ndvi_history_count = (
            recent_history["NDVI"]
            .notna()
            .sum()
            if not recent_history.empty
            else 0
        )

        rainfall_history_count = (
            recent_history["Rainfall_mm"]
            .notna()
            .sum()
            if not recent_history.empty
            else 0
        )

        temperature_history_count = (
            recent_history["Temperature_C"]
            .notna()
            .sum()
            if not recent_history.empty
            else 0
        )

        # ----------------------------------------------------
        # Point-level prediction eligibility
        # ----------------------------------------------------

        eligibility = (
            "NOT_READY"
        )

        reason = (
            "No valid common observation week."
        )

        if status == "READY":

            if (
                not pd.isna(
                    current_ndvi
                )
                and not pd.isna(
                    current_rainfall
                )
                and not pd.isna(
                    current_temperature
                )
            ):

                eligibility = "ELIGIBLE"

                reason = (
                    "Recent common NDVI, "
                    "rainfall and temperature "
                    "data available."
                )

            else:

                eligibility = "NOT_READY"

                reason = (
                    "Current common-week "
                    "feature values incomplete."
                )

        elif status == "WARNING":

            eligibility = (
                "NOT_READY"
            )

            reason = (
                "Common data exists but "
                "is older than the preferred "
                "freshness window."
            )

        elif status == "STALE":

            eligibility = (
                "NOT_READY"
            )

            reason = (
                "Common data is too old "
                "for real-time prediction."
            )

        # ----------------------------------------------------
        # Save row
        # ----------------------------------------------------

        rows.append(
            {
                "point_id": point_id,
                "district": district,
                "latitude": latitude,
                "longitude": longitude,

                "latest_ndvi_week": latest_ndvi,
                "latest_rainfall_week": latest_rainfall,
                "latest_temperature_week": (
                    latest_temperature
                ),

                "common_latest_week": (
                    common_latest_week
                ),

                "data_age_days": age_days,

                "current_ndvi": current_ndvi,
                "current_rainfall": current_rainfall,
                "current_temperature": (
                    current_temperature
                ),

                "ndvi_recent_history_count": (
                    ndvi_history_count
                ),
                "rainfall_recent_history_count": (
                    rainfall_history_count
                ),
                "temperature_recent_history_count": (
                    temperature_history_count
                ),

                "freshness_status": status,
                "prediction_eligibility": eligibility,
                "reason": reason,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    status_df,
):
    print(
        "\n" + "=" * 80
    )

    print(
        "REAL-TIME DATA FRESHNESS SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        f"Total points: "
        f"{len(status_df)}"
    )

    print(
        "\nFreshness status:"
    )

    print(
        status_df[
            "freshness_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\nPrediction eligibility:"
    )

    print(
        status_df[
            "prediction_eligibility"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\nDetailed point status:"
    )

    display_columns = [
        "point_id",
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

    print(
        status_df[
            display_columns
        ]
        .to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print(
        "=" * 80
    )

    print(
        "AGRIVISION AI - REAL-TIME DATA STATUS"
    )

    print(
        "=" * 80
    )

    print(
        f"\nInput:"
    )

    print(
        INPUT_FILE
    )

    df = load_data()

    print(
        f"\nLoaded rows: "
        f"{len(df):,}"
    )

    print(
        f"Points: "
        f"{df['point_id'].nunique()}"
    )

    print(
        f"Weeks: "
        f"{df['Week_Start'].nunique()}"
    )

    status_df = build_status_table(
        df
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    output_file = (
        OUTPUT_DIR
        / "realtime_point_status.csv"
    )

    status_df.to_csv(
        output_file,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(
        status_df
    )

    # --------------------------------------------------------
    # Save ready points separately
    # --------------------------------------------------------

    ready_df = status_df[
        status_df[
            "prediction_eligibility"
        ]
        == "ELIGIBLE"
    ].copy()

    ready_file = (
        OUTPUT_DIR
        / "prediction_ready_points.csv"
    )

    ready_df.to_csv(
        ready_file,
        index=False,
    )

    # --------------------------------------------------------
    # Save status summary
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        [
            {
                "Metric": "Total_Points",
                "Value": len(status_df),
            },
            {
                "Metric": "Ready_Points",
                "Value": len(ready_df),
            },
            {
                "Metric": "Warning_Points",
                "Value": int(
                    (
                        status_df[
                            "freshness_status"
                        ]
                        == "WARNING"
                    ).sum()
                ),
            },
            {
                "Metric": "Stale_Points",
                "Value": int(
                    (
                        status_df[
                            "freshness_status"
                        ]
                        == "STALE"
                    ).sum()
                ),
            },
            {
                "Metric": "No_Common_Data",
                "Value": int(
                    (
                        status_df[
                            "freshness_status"
                        ]
                        == "NO_COMMON_DATA"
                    ).sum()
                ),
            },
        ]
    )

    summary_file = (
        OUTPUT_DIR
        / "realtime_status_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False,
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "REAL-TIME STATUS ANALYSIS COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        "\nPoint status file:"
    )

    print(
        output_file
    )

    print(
        "\nPrediction-ready points:"
    )

    print(
        ready_file
    )

    print(
        "\nSummary:"
    )

    print(
        summary_file
    )

    print(
        "\nNo predictions were generated."
    )

    print(
        "This script only determines whether "
        "a point has sufficiently fresh data."
    )


if __name__ == "__main__":
    main()