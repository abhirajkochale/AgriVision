from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "ml"
    / "feature_outputs"
    / "AgriVision_Maharashtra_ML_Features_2017_2025.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "model_outputs"
    / "ndvi_quality_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "NDVI_next_week"

YEAR_COLUMN = "Year"

SUSPICIOUS_LOW = -0.8
SUSPICIOUS_HIGH = 0.95


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("AGRIVISION AI - NDVI QUALITY AUDIT")
print("=" * 80)

print(
    f"\nInput file:\n{INPUT_FILE}"
)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Feature dataset not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE
)

print(
    f"\nLoaded dataset shape: {df.shape}"
)


# ============================================================
# VALIDATE
# ============================================================

required_columns = [
    TARGET,
    YEAR_COLUMN,
]

for col in required_columns:

    if col not in df.columns:
        raise ValueError(
            f"Missing required column: {col}"
        )


df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce",
)

df[YEAR_COLUMN] = pd.to_numeric(
    df[YEAR_COLUMN],
    errors="coerce",
)


# ============================================================
# BASIC NDVI STATISTICS
# ============================================================

target = df[TARGET].dropna()

print("\n" + "=" * 80)
print("BASIC NDVI TARGET STATISTICS")
print("=" * 80)

print(
    f"Valid target observations : {len(target):,}"
)

print(
    f"Minimum                   : {target.min():.8f}"
)

print(
    f"Maximum                   : {target.max():.8f}"
)

print(
    f"Mean                      : {target.mean():.8f}"
)

print(
    f"Median                    : {target.median():.8f}"
)

print(
    f"Std                       : {target.std():.8f}"
)


# ============================================================
# EXACT BOUNDARY VALUES
# ============================================================

exact_minus_one = df[
    df[TARGET] == -1.0
].copy()

exact_plus_one = df[
    df[TARGET] == 1.0
].copy()

print("\n" + "=" * 80)
print("EXACT NDVI BOUNDARY VALUES")
print("=" * 80)

print(
    f"Exact -1.0 observations: "
    f"{len(exact_minus_one):,}"
)

print(
    f"Exact +1.0 observations: "
    f"{len(exact_plus_one):,}"
)


# ============================================================
# SUSPICIOUS EXTREMES
# ============================================================

very_low = df[
    df[TARGET] <= SUSPICIOUS_LOW
].copy()

very_high = df[
    df[TARGET] >= SUSPICIOUS_HIGH
].copy()

print("\n" + "=" * 80)
print("SUSPICIOUS EXTREMES")
print("=" * 80)

print(
    f"Target <= {SUSPICIOUS_LOW}: "
    f"{len(very_low):,}"
)

print(
    f"Target >= {SUSPICIOUS_HIGH}: "
    f"{len(very_high):,}"
)


# ============================================================
# SAVE EXTREME RECORDS
# ============================================================

extreme_columns = [
    col
    for col in [
        "point_id",
        "district",
        "region_type",
        "study_area",
        "Year",
        "Week_Number",
        "Week_Start",
        "Week_End",
        "Latitude",
        "Longitude",
        "NDVI",
        TARGET,
        "Rainfall_Current",
        "Temperature_Current",
        "NDVI_change_1w",
        "NDVI_change_2w",
        "NDVI_change_4w",
    ]
    if col in df.columns
]

very_low[
    extreme_columns
].sort_values(
    TARGET
).to_csv(
    OUTPUT_DIR
    / "very_low_ndvi_records.csv",
    index=False,
)

very_high[
    extreme_columns
].sort_values(
    TARGET,
    ascending=False,
).to_csv(
    OUTPUT_DIR
    / "very_high_ndvi_records.csv",
    index=False,
)


# ============================================================
# EXACT -1 ANALYSIS
# ============================================================

if not exact_minus_one.empty:

    print("\n" + "=" * 80)
    print("EXACT -1 NDVI RECORDS")
    print("=" * 80)

    display_columns = [
        col
        for col in [
            "point_id",
            "district",
            "Year",
            "Week_Number",
            "Week_Start",
            "Latitude",
            "Longitude",
            "NDVI",
            TARGET,
            "Rainfall_Current",
            "Temperature_Current",
        ]
        if col in exact_minus_one.columns
    ]

    print(
        exact_minus_one[
            display_columns
        ]
        .head(50)
        .to_string(index=False)
    )

    exact_minus_one[
        display_columns
    ].to_csv(
        OUTPUT_DIR
        / "exact_minus_one_records.csv",
        index=False,
    )


# ============================================================
# EXACT -1 BY YEAR
# ============================================================

minus_one_by_year = (
    exact_minus_one
    .groupby(
        YEAR_COLUMN
    )
    .size()
    .reset_index(
        name="Exact_Minus_One_Count"
    )
    .sort_values(
        YEAR_COLUMN
    )
)

minus_one_by_year[
    "Percentage_of_Year_Targets"
] = (
    minus_one_by_year[
        "Exact_Minus_One_Count"
    ]
    /
    df.groupby(
        YEAR_COLUMN
    )[TARGET]
    .count()
    .reindex(
        minus_one_by_year[
            YEAR_COLUMN
        ]
    )
    .values
    * 100
)

minus_one_by_year.to_csv(
    OUTPUT_DIR
    / "exact_minus_one_by_year.csv",
    index=False,
)


# ============================================================
# EXACT -1 BY DISTRICT
# ============================================================

if "district" in exact_minus_one.columns:

    minus_one_by_district = (
        exact_minus_one
        .groupby(
            "district"
        )
        .size()
        .reset_index(
            name="Exact_Minus_One_Count"
        )
        .sort_values(
            "Exact_Minus_One_Count",
            ascending=False,
        )
    )

    minus_one_by_district.to_csv(
        OUTPUT_DIR
        / "exact_minus_one_by_district.csv",
        index=False,
    )

    print("\n" + "=" * 80)
    print("DISTRICTS WITH MOST EXACT -1 VALUES")
    print("=" * 80)

    print(
        minus_one_by_district
        .head(20)
        .to_string(index=False)
    )


# ============================================================
# EXACT -1 BY POINT
# ============================================================

if "point_id" in exact_minus_one.columns:

    minus_one_by_point = (
        exact_minus_one
        .groupby(
            [
                "point_id",
                "district"
                if "district"
                in exact_minus_one.columns
                else "point_id",
            ]
        )
        .size()
        .reset_index(
            name="Exact_Minus_One_Count"
        )
    )

    # Remove duplicated grouping column if necessary.
    minus_one_by_point = (
        minus_one_by_point
        .loc[
            :,
            ~minus_one_by_point.columns.duplicated()
        ]
    )

    minus_one_by_point = (
        minus_one_by_point
        .sort_values(
            "Exact_Minus_One_Count",
            ascending=False,
        )
    )

    minus_one_by_point.to_csv(
        OUTPUT_DIR
        / "exact_minus_one_by_point.csv",
        index=False,
    )

    print("\n" + "=" * 80)
    print("POINTS WITH MOST EXACT -1 VALUES")
    print("=" * 80)

    print(
        minus_one_by_point
        .head(20)
        .to_string(index=False)
    )


# ============================================================
# TEMPORAL PATTERN OF EXTREMES
# ============================================================

if "Month" in df.columns:

    df["Month"] = pd.to_numeric(
        df["Month"],
        errors="coerce",
    )

    extreme_month = (
        df[
            df[TARGET] <= SUSPICIOUS_LOW
        ]
        .groupby(
            "Month"
        )
        .size()
        .reset_index(
            name="Very_Low_NDVI_Count"
        )
        .sort_values(
            "Month"
        )
    )

    extreme_month.to_csv(
        OUTPUT_DIR
        / "very_low_ndvi_by_month.csv",
        index=False,
    )


# ============================================================
# WEEK PATTERN
# ============================================================

if "Week_Number" in df.columns:

    low_by_week = (
        df[
            df[TARGET] <= SUSPICIOUS_LOW
        ]
        .groupby(
            "Week_Number"
        )
        .size()
        .reset_index(
            name="Very_Low_NDVI_Count"
        )
        .sort_values(
            "Very_Low_NDVI_Count",
            ascending=False,
        )
    )

    low_by_week.to_csv(
        OUTPUT_DIR
        / "very_low_ndvi_by_week.csv",
        index=False,
    )


# ============================================================
# REPEATED SUSPICIOUS POINTS
# ============================================================

if "point_id" in df.columns:

    point_summary = (
        df.groupby(
            [
                "point_id",
                "district"
                if "district"
                in df.columns
                else "point_id",
            ]
        )[TARGET]
        .agg(
            Total="count",
            Exact_Minus_One=lambda x: (
                x == -1.0
            ).sum(),
            Very_Low=lambda x: (
                x <= SUSPICIOUS_LOW
            ).sum(),
            Very_High=lambda x: (
                x >= SUSPICIOUS_HIGH
            ).sum(),
            Minimum="min",
            Maximum="max",
            Mean="mean",
        )
        .reset_index()
    )

    point_summary = (
        point_summary
        .loc[
            :,
            ~point_summary.columns.duplicated()
        ]
    )

    point_summary[
        "Exact_Minus_One_Percent"
    ] = (
        point_summary[
            "Exact_Minus_One"
        ]
        /
        point_summary[
            "Total"
        ]
        * 100
    )

    point_summary = point_summary.sort_values(
        [
            "Exact_Minus_One",
            "Very_Low",
        ],
        ascending=False,
    )

    point_summary.to_csv(
        OUTPUT_DIR
        / "point_ndvi_quality_summary.csv",
        index=False,
    )


# ============================================================
# EXTREME TARGET + CURRENT NDVI RELATIONSHIP
# ============================================================

if "NDVI" in df.columns:

    extreme_transition_df = df[
        df[TARGET] <= SUSPICIOUS_LOW
    ].copy()

    extreme_transition_df[
        "Current_NDVI"
    ] = pd.to_numeric(
        extreme_transition_df["NDVI"],
        errors="coerce",
    )

    transition_summary = (
        extreme_transition_df[
            [
                "Current_NDVI",
                TARGET,
            ]
        ]
        .describe()
    )

    transition_summary.to_csv(
        OUTPUT_DIR
        / "extreme_transition_statistics.csv"
    )


# ============================================================
# CLEANED DATASET SCENARIO
# ============================================================

# We DO NOT modify the original dataset.
# We create a separate audit-only scenario where suspicious
# target values are excluded for sensitivity analysis.

audit_mask = (
    df[TARGET].notna()
    & (df[TARGET] > SUSPICIOUS_LOW)
    & (df[TARGET] < SUSPICIOUS_HIGH)
)

cleaned_sensitivity_df = df[
    audit_mask
].copy()

excluded_count = (
    len(df) - len(cleaned_sensitivity_df)
)

excluded_percentage = (
    excluded_count
    / len(df)
    * 100
)

print("\n" + "=" * 80)
print("SENSITIVITY DATASET")
print("=" * 80)

print(
    f"Original rows with target : {len(df):,}"
)

print(
    f"Rows retained             : "
    f"{len(cleaned_sensitivity_df):,}"
)

print(
    f"Rows excluded             : "
    f"{excluded_count:,}"
)

print(
    f"Excluded percentage       : "
    f"{excluded_percentage:.4f}%"
)

cleaned_sensitivity_df.to_csv(
    OUTPUT_DIR
    / "audit_sensitivity_dataset.csv",
    index=False,
)


# ============================================================
# SUMMARY TABLE
# ============================================================

summary = pd.DataFrame(
    [
        {
            "Metric": "Valid_Target_Observations",
            "Value": len(target),
        },
        {
            "Metric": "Exact_Minus_One",
            "Value": len(exact_minus_one),
        },
        {
            "Metric": "Exact_Plus_One",
            "Value": len(exact_plus_one),
        },
        {
            "Metric": "Target_Less_Equal_Minus_0.8",
            "Value": len(very_low),
        },
        {
            "Metric": "Target_Greater_Equal_0.95",
            "Value": len(very_high),
        },
        {
            "Metric": "Sensitivity_Excluded",
            "Value": excluded_count,
        },
        {
            "Metric": "Sensitivity_Excluded_Percent",
            "Value": excluded_percentage,
        },
    ]
)

summary.to_csv(
    OUTPUT_DIR
    / "ndvi_quality_summary.csv",
    index=False,
)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# REPORT
# ============================================================

report_path = (
    OUTPUT_DIR
    / "ndvi_quality_audit_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as report:

    report.write(
        "AGRIVISION AI - NDVI QUALITY AUDIT REPORT\n"
    )

    report.write(
        "=" * 80 + "\n\n"
    )

    report.write(
        f"Valid target observations: "
        f"{len(target):,}\n"
    )

    report.write(
        f"Exact -1.0 observations: "
        f"{len(exact_minus_one):,}\n"
    )

    report.write(
        f"Exact +1.0 observations: "
        f"{len(exact_plus_one):,}\n"
    )

    report.write(
        f"Target <= {SUSPICIOUS_LOW}: "
        f"{len(very_low):,}\n"
    )

    report.write(
        f"Target >= {SUSPICIOUS_HIGH}: "
        f"{len(very_high):,}\n"
    )

    report.write(
        f"\nSensitivity rows excluded: "
        f"{excluded_count:,}\n"
    )

    report.write(
        f"Sensitivity excluded percent: "
        f"{excluded_percentage:.4f}%\n\n"
    )

    report.write(
        "IMPORTANT:\n"
    )

    report.write(
        "The original dataset was NOT modified.\n"
    )

    report.write(
        "The sensitivity dataset is for investigation only.\n"
    )

    report.write(
        "Extreme values must be checked against the original "
        "satellite processing chain before any production data "
        "is removed.\n"
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("NDVI QUALITY AUDIT COMPLETE")
print("=" * 80)

print(
    f"\nOutputs saved to:\n{OUTPUT_DIR}"
)

print(
    "\nImportant:"
)

print(
    "NO original dataset was modified."
)

print(
    "NO production model was modified."
)

print(
    "Review the audit results before deciding "
    "whether any observations should be excluded."
)