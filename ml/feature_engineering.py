from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# AGRIVISION AI
# FEATURE ENGINEERING + TARGET CONSTRUCTION
# 2017–2025
#
# IMPORTANT DESIGN:
# - Keep every row with a real current NDVI
#   and a real next-week NDVI target.
# - Do NOT drop rows because historical lag features
#   are missing.
# - Do NOT fabricate or interpolate NDVI.
# - Missing feature values are handled later in model_training.py.
# ============================================================


print("=" * 75)
print("AGRIVISION - FEATURE ENGINEERING")
print("2017–2025")
print("=" * 75)


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_FILE = (
    PROJECT_DIR
    / "dataset"
    / "AgriVision_Maharashtra_Combined_Master_2017_2025.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "ml"
    / "feature_outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FEATURE_DATASET = (
    OUTPUT_DIR
    / "AgriVision_Maharashtra_ML_Features_2017_2025.csv"
)

REPORT_FILE = (
    OUTPUT_DIR
    / "feature_engineering_report.txt"
)

MISSING_FILE = (
    OUTPUT_DIR
    / "feature_missing_values.csv"
)

STATISTICS_FILE = (
    OUTPUT_DIR
    / "feature_statistics.csv"
)

TARGET_STATISTICS_FILE = (
    OUTPUT_DIR
    / "target_statistics.csv"
)

TARGET_CORRELATIONS_FILE = (
    OUTPUT_DIR
    / "target_correlations.csv"
)

YEAR_SUMMARY_FILE = (
    OUTPUT_DIR
    / "training_rows_by_year.csv"
)


# ============================================================
# 2. CHECK INPUT DATASET
# ============================================================

print()
print("Dataset:")
print(DATASET_FILE)

if not DATASET_FILE.exists():

    raise FileNotFoundError(
        f"\nCombined dataset not found:\n{DATASET_FILE}"
    )

print("PASS - Combined master dataset found")


# ============================================================
# 3. LOAD DATASET
# ============================================================

print()
print("=" * 75)
print("LOADING COMBINED DATASET")
print("=" * 75)

df = pd.read_csv(
    DATASET_FILE
)

print()
print(f"Rows:    {len(df):,}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# 4. DATE / NUMERIC PREPARATION
# ============================================================

df["Week_Start"] = pd.to_datetime(
    df["Week_Start"],
    errors="coerce"
)

df["Week_End"] = pd.to_datetime(
    df["Week_End"],
    errors="coerce"
)

numeric_columns = [
    "Year",
    "Week_Number",
    "latitude",
    "longitude",
    "NDVI",
    "Rainfall_mm",
    "Temperature_C",
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# 5. BASIC DATA VALIDATION
# ============================================================

print()
print("=" * 75)
print("BASIC DATA VALIDATION")
print("=" * 75)

duplicates = (
    df
    .duplicated(
        subset=[
            "point_id",
            "Week_Start"
        ]
    )
    .sum()
)

print(
    f"Duplicate point-week rows: {duplicates}"
)

if duplicates != 0:

    raise ValueError(
        "Duplicate point-week rows detected."
    )

print(
    "PASS - No duplicate point-week rows"
)


# ============================================================
# 6. SORT TEMPORALLY
# ============================================================

df = (
    df
    .sort_values(
        by=[
            "point_id",
            "Week_Start"
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# 7. CONSTRUCT ACTUAL NEXT-WEEK NDVI TARGET
# ============================================================

print()
print("=" * 75)
print("CONSTRUCTING NEXT-WEEK NDVI TARGET")
print("=" * 75)


target_lookup = df[
    [
        "point_id",
        "Week_Start",
        "NDVI"
    ]
].copy()


target_lookup = target_lookup.rename(
    columns={
        "Week_Start": "Target_Week_Start",
        "NDVI": "NDVI_next_week"
    }
)


df["Target_Week_Start"] = (
    df["Week_Start"]
    + pd.Timedelta(days=7)
)


df = df.merge(
    target_lookup,
    on=[
        "point_id",
        "Target_Week_Start"
    ],
    how="left",
    validate="one_to_one"
)


# ============================================================
# 8. TARGET AVAILABILITY
# ============================================================

df["Current_NDVI_Available"] = (
    df["NDVI"]
    .notna()
    .astype(int)
)

df["Next_NDVI_Available"] = (
    df["NDVI_next_week"]
    .notna()
    .astype(int)
)

df["Usable_Target"] = (
    df["NDVI"].notna()
    &
    df["NDVI_next_week"].notna()
).astype(int)


print()
print(
    f"Current NDVI available: "
    f"{df['Current_NDVI_Available'].sum():,}"
)

print(
    f"Next-week NDVI available: "
    f"{df['Next_NDVI_Available'].sum():,}"
)

print(
    f"Usable target rows: "
    f"{df['Usable_Target'].sum():,}"
)


# ============================================================
# 9. TEMPORAL FEATURES
# ============================================================

print()
print("=" * 75)
print("CREATING TEMPORAL FEATURES")
print("=" * 75)


df["Week_Of_Year"] = (
    df["Week_Start"]
    .dt.isocalendar()
    .week
    .astype(int)
)

df["Month"] = (
    df["Week_Start"]
    .dt.month
)

df["Day_Of_Year"] = (
    df["Week_Start"]
    .dt.dayofyear
)

df["Year_Index"] = (
    df["Year"] - 2017
)


# Cyclic week features
df["Week_Sin"] = np.sin(
    2
    * np.pi
    * df["Week_Of_Year"]
    / 52
)

df["Week_Cos"] = np.cos(
    2
    * np.pi
    * df["Week_Of_Year"]
    / 52
)


# Cyclic month features
df["Month_Sin"] = np.sin(
    2
    * np.pi
    * df["Month"]
    / 12
)

df["Month_Cos"] = np.cos(
    2
    * np.pi
    * df["Month"]
    / 12
)


# ============================================================
# 10. GROUPED DATA
# ============================================================

grouped = df.groupby(
    "point_id",
    sort=False
)


# ============================================================
# 11. NDVI LAG FEATURES
# ============================================================

print()
print("=" * 75)
print("CREATING NDVI LAG FEATURES")
print("=" * 75)


for lag in [1, 2, 3, 4, 8]:

    df[f"NDVI_lag_{lag}"] = (
        grouped["NDVI"]
        .shift(lag)
    )


# ============================================================
# 12. NDVI CHANGE FEATURES
# ============================================================

df["NDVI_change_1w"] = (
    df["NDVI"]
    - df["NDVI_lag_1"]
)

df["NDVI_change_2w"] = (
    df["NDVI"]
    - df["NDVI_lag_2"]
)

df["NDVI_change_4w"] = (
    df["NDVI"]
    - df["NDVI_lag_4"]
)


# ============================================================
# 13. RAINFALL LAGS
# ============================================================

print()
print("=" * 75)
print("CREATING RAINFALL FEATURES")
print("=" * 75)


for lag in [1, 2, 4]:

    df[f"Rainfall_lag_{lag}"] = (
        grouped["Rainfall_mm"]
        .shift(lag)
    )


# ============================================================
# 14. RAINFALL ROLLING FEATURES
# ============================================================

# Shift by one week first.
# Therefore the rolling windows contain only historical data.

rainfall_previous = (
    grouped["Rainfall_mm"]
    .shift(1)
)


df["Rainfall_rolling_2w"] = (
    rainfall_previous
    .groupby(df["point_id"])
    .transform(
        lambda s: s.rolling(
            window=2,
            min_periods=2
        ).sum()
    )
)


df["Rainfall_rolling_4w"] = (
    rainfall_previous
    .groupby(df["point_id"])
    .transform(
        lambda s: s.rolling(
            window=4,
            min_periods=4
        ).sum()
    )
)


df["Rainfall_rolling_8w"] = (
    rainfall_previous
    .groupby(df["point_id"])
    .transform(
        lambda s: s.rolling(
            window=8,
            min_periods=8
        ).sum()
    )
)


# ============================================================
# 15. TEMPERATURE LAGS
# ============================================================

print()
print("=" * 75)
print("CREATING TEMPERATURE FEATURES")
print("=" * 75)


for lag in [1, 2, 4]:

    df[f"Temperature_lag_{lag}"] = (
        grouped["Temperature_C"]
        .shift(lag)
    )


# ============================================================
# 16. TEMPERATURE ROLLING FEATURES
# ============================================================

temperature_previous = (
    grouped["Temperature_C"]
    .shift(1)
)


df["Temperature_rolling_2w"] = (
    temperature_previous
    .groupby(df["point_id"])
    .transform(
        lambda s: s.rolling(
            window=2,
            min_periods=2
        ).mean()
    )
)


df["Temperature_rolling_4w"] = (
    temperature_previous
    .groupby(df["point_id"])
    .transform(
        lambda s: s.rolling(
            window=4,
            min_periods=4
        ).mean()
    )
)


df["Temperature_rolling_8w"] = (
    temperature_previous
    .groupby(df["point_id"])
    .transform(
        lambda s: s.rolling(
            window=8,
            min_periods=8
        ).mean()
    )
)


# ============================================================
# 17. CURRENT CLIMATE FEATURES
# ============================================================

df["Rainfall_Current"] = (
    df["Rainfall_mm"]
)

df["Temperature_Current"] = (
    df["Temperature_C"]
)


# ============================================================
# 18. SPATIAL FEATURES
# ============================================================

df["Latitude"] = (
    df["latitude"]
)

df["Longitude"] = (
    df["longitude"]
)


# ============================================================
# 19. SELECT ONLY VALID SUPERVISED TARGET ROWS
# ============================================================

print()
print("=" * 75)
print("PREPARING SUPERVISED DATASET")
print("=" * 75)


# IMPORTANT:
# Only target availability is used for filtering.
#
# We DO NOT drop rows because NDVI lags are missing.
#
# This preserves the approximately 33,655 legitimate
# supervised observations.

model_df = df[
    df["Usable_Target"] == 1
].copy()


print()
print(
    f"Rows before target filtering: "
    f"{len(df):,}"
)

print(
    f"Rows after target filtering:  "
    f"{len(model_df):,}"
)


expected_target_rows = 33655


if len(model_df) != expected_target_rows:

    print()
    print(
        "WARNING: Expected approximately "
        f"{expected_target_rows:,} usable target rows, "
        f"but found {len(model_df):,}."
    )


# ============================================================
# 20. TARGET VALIDATION
# ============================================================

target_min = (
    model_df["NDVI_next_week"]
    .min()
)

target_max = (
    model_df["NDVI_next_week"]
    .max()
)


print()
print(
    f"NDVI_next_week minimum: "
    f"{target_min}"
)

print(
    f"NDVI_next_week maximum: "
    f"{target_max}"
)


if (
    target_min < -1
    or target_max > 1
):

    raise ValueError(
        "NDVI_next_week contains values outside [-1, 1]."
    )

print(
    "PASS - Target values within [-1, 1]"
)


# ============================================================
# 21. FEATURE LIST
# ============================================================

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

    # Rainfall
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


# ============================================================
# 22. FEATURE MISSINGNESS
# ============================================================

print()
print("=" * 75)
print("FEATURE AVAILABILITY")
print("=" * 75)


feature_missing = (
    model_df[
        feature_columns
    ]
    .isna()
    .sum()
    .reset_index()
)


feature_missing.columns = [
    "feature",
    "missing_count"
]


feature_missing["missing_percent"] = (
    feature_missing["missing_count"]
    / len(model_df)
    * 100
)


print(
    feature_missing.to_string(
        index=False
    )
)


feature_missing.to_csv(
    MISSING_FILE,
    index=False
)


# ============================================================
# 23. FEATURE COMPLETENESS SUMMARY
# ============================================================

complete_feature_rows = (
    model_df[
        feature_columns
    ]
    .notna()
    .all(axis=1)
    .sum()
)


print()
print(
    f"Rows with every feature available: "
    f"{complete_feature_rows:,}"
)

print(
    f"Rows requiring feature imputation: "
    f"{len(model_df) - complete_feature_rows:,}"
)


# ============================================================
# 24. DO NOT DROP ROWS FOR MISSING FEATURES
# ============================================================

# Missing historical features remain NaN here.
#
# This is intentional.
#
# model_training.py will perform preprocessing using
# the training set only, so that the imputation process
# does not leak information from validation/test periods.


# ============================================================
# 25. FINAL TARGET CHECK
# ============================================================

if model_df[
    "NDVI_next_week"
].isna().any():

    raise ValueError(
        "Missing target values remain."
    )

print()
print(
    "PASS - No missing target values"
)


# ============================================================
# 26. FEATURE STATISTICS
# ============================================================

feature_statistics = (
    model_df[
        feature_columns
        + ["NDVI_next_week"]
    ]
    .describe()
    .T
    .reset_index()
    .rename(
        columns={
            "index": "feature"
        }
    )
)


feature_statistics.to_csv(
    STATISTICS_FILE,
    index=False
)


# ============================================================
# 27. TARGET STATISTICS
# ============================================================

target_statistics = pd.DataFrame({

    "metric": [
        "count",
        "mean",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
    ],

    "value": [

        model_df[
            "NDVI_next_week"
        ].count(),

        model_df[
            "NDVI_next_week"
        ].mean(),

        model_df[
            "NDVI_next_week"
        ].std(),

        model_df[
            "NDVI_next_week"
        ].min(),

        model_df[
            "NDVI_next_week"
        ].quantile(0.25),

        model_df[
            "NDVI_next_week"
        ].median(),

        model_df[
            "NDVI_next_week"
        ].quantile(0.75),

        model_df[
            "NDVI_next_week"
        ].max(),
    ],
})


target_statistics.to_csv(
    TARGET_STATISTICS_FILE,
    index=False
)


# ============================================================
# 28. TARGET CORRELATIONS
# ============================================================

correlation_matrix = (
    model_df[
        feature_columns
        + ["NDVI_next_week"]
    ]
    .corr()
)


target_correlations = (
    correlation_matrix[
        ["NDVI_next_week"]
    ]
    .drop(
        index="NDVI_next_week"
    )
    .rename(
        columns={
            "NDVI_next_week":
            "correlation"
        }
    )
    .sort_values(
        "correlation",
        ascending=False
    )
)


target_correlations.to_csv(
    TARGET_CORRELATIONS_FILE
)


# ============================================================
# 29. FINAL ML DATASET
# ============================================================

output_columns = [

    "Week_Start",
    "Year",
    "Week_Number",

    "point_id",
    "district",

    "Latitude",
    "Longitude",
]


output_columns += feature_columns

output_columns.append(
    "NDVI_next_week"
)


model_df = (
    model_df[
        output_columns
    ]
    .sort_values(
        by=[
            "Week_Start",
            "point_id"
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 30. SAVE FEATURE DATASET
# ============================================================

model_df.to_csv(
    FEATURE_DATASET,
    index=False
)


# ============================================================
# 31. YEAR-WISE SUMMARY
# ============================================================

year_summary = (
    model_df
    .groupby("Year")
    .size()
    .reset_index(
        name="usable_training_rows"
    )
)


year_summary["percentage"] = (
    year_summary[
        "usable_training_rows"
    ]
    / len(model_df)
    * 100
)


year_summary.to_csv(
    YEAR_SUMMARY_FILE,
    index=False
)


# ============================================================
# 32. REPORT
# ============================================================

report_lines = [

    "AGRIVISION AI - FEATURE ENGINEERING REPORT",

    "2017–2025",

    "",

    f"Original combined rows: "
    f"{len(df):,}",

    f"Usable target rows: "
    f"{len(model_df):,}",

    f"Rows with all features available: "
    f"{complete_feature_rows:,}",

    f"Rows requiring feature imputation: "
    f"{len(model_df) - complete_feature_rows:,}",

    "",

    "Target: NDVI_next_week",

    "Target definition: actual NDVI for the same point_id "
    "exactly 7 days after the current Week_Start.",

    "",

    "No interpolation or fabricated NDVI target values were used.",

    "Rows were NOT removed because of missing historical "
    "feature values.",

    "Missing feature values will be handled during model "
    "training using training-only preprocessing.",

    "",

    f"Number of model features: "
    f"{len(feature_columns)}",

    "",

    "Year-wise supervised rows:",
]


for _, row in year_summary.iterrows():

    report_lines.append(
        f"{int(row['Year'])}: "
        f"{int(row['usable_training_rows']):,}"
    )


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(report_lines)
    )


# ============================================================
# 33. FINAL SUMMARY
# ============================================================

print()
print("=" * 75)
print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
print("=" * 75)

print()

print(
    f"Original combined rows: "
    f"{len(df):,}"
)

print(
    f"Usable target rows: "
    f"{len(model_df):,}"
)

print(
    f"Rows with every feature available: "
    f"{complete_feature_rows:,}"
)

print(
    f"Rows requiring feature imputation: "
    f"{len(model_df) - complete_feature_rows:,}"
)

print(
    f"Number of features: "
    f"{len(feature_columns)}"
)

print()
print("Year-wise supervised rows:")

print(
    year_summary.to_string(
        index=False
    )
)

print()
print("Output:")

print(
    FEATURE_DATASET
)

print()
print("=" * 75)
print("DONE")
print("=" * 75)