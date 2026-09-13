# pyrefly: ignore [missing-import]

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# AGRIVISION AI
# GSMaP FEATURE ENGINEERING + TARGET CONSTRUCTION
# 2017–2025
#
# IMPORTANT:
# - Uses GSMaP rainfall instead of CHIRPS rainfall
# - Keeps ALL 33,655 supervised target rows
# - DOES NOT drop rows with missing lag/rolling features
# - Missing feature values are handled later during model
#   training using training-only imputation
# ============================================================

print("=" * 80)
print("AGRIVISION AI - GSMaP FEATURE ENGINEERING")
print("2017–2025")
print("=" * 80)


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

BASE_DATASET_FILE = (
    PROJECT_DIR
    / "dataset"
    / "AgriVision_Maharashtra_Combined_Master_2017_2025.csv"
)

GSMAP_DATASET_FILE = (
    PROJECT_DIR
    / "dataset"
    / "Rainfall_GSMaP"
    / "AgriVision_Maharashtra_GSMaP_Rainfall_Master_2017_2025.csv"
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
    / "AgriVision_Maharashtra_ML_Features_GSMaP_2017_2025.csv"
)

REPORT_FILE = (
    OUTPUT_DIR
    / "feature_engineering_report_gsmap.txt"
)

MISSING_FILE = (
    OUTPUT_DIR
    / "feature_missing_values_gsmap.csv"
)

STATISTICS_FILE = (
    OUTPUT_DIR
    / "feature_statistics_gsmap.csv"
)

TARGET_STATISTICS_FILE = (
    OUTPUT_DIR
    / "target_statistics_gsmap.csv"
)

TARGET_CORRELATIONS_FILE = (
    OUTPUT_DIR
    / "target_correlations_gsmap.csv"
)


# ============================================================
# 2. CHECK INPUT FILES
# ============================================================

print()
print("=" * 80)
print("CHECKING INPUT FILES")
print("=" * 80)

print()
print("Base combined dataset:")
print(BASE_DATASET_FILE)

if not BASE_DATASET_FILE.exists():
    raise FileNotFoundError(
        f"\nCombined master dataset not found:\n{BASE_DATASET_FILE}"
    )

print("PASS - Combined master dataset found")

print()
print("GSMaP rainfall dataset:")
print(GSMAP_DATASET_FILE)

if not GSMAP_DATASET_FILE.exists():
    raise FileNotFoundError(
        f"\nGSMaP rainfall dataset not found:\n{GSMAP_DATASET_FILE}"
    )

print("PASS - GSMaP rainfall dataset found")


# ============================================================
# 3. LOAD DATASETS
# ============================================================

print()
print("=" * 80)
print("LOADING DATASETS")
print("=" * 80)

df = pd.read_csv(BASE_DATASET_FILE)
gsmap = pd.read_csv(GSMAP_DATASET_FILE)

print()
print(f"Base dataset rows:   {len(df):,}")
print(f"GSMaP dataset rows:  {len(gsmap):,}")


# ============================================================
# 4. DATE / NUMERIC PREPARATION - BASE DATA
# ============================================================

print()
print("=" * 80)
print("PREPARING BASE DATA")
print("=" * 80)

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
    if column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# ============================================================
# 5. PREPARE GSMaP DATA
# ============================================================

print()
print("=" * 80)
print("PREPARING GSMaP DATA")
print("=" * 80)

gsmap["Week_Start"] = pd.to_datetime(
    gsmap["Week_Start"],
    errors="coerce"
)

gsmap["Week_End"] = pd.to_datetime(
    gsmap["Week_End"],
    errors="coerce"
)

gsmap["Year"] = pd.to_numeric(
    gsmap["Year"],
    errors="coerce"
)

gsmap["Week_Number"] = pd.to_numeric(
    gsmap["Week_Number"],
    errors="coerce"
)

gsmap["Rainfall_GSMaP"] = pd.to_numeric(
    gsmap["Rainfall_GSMaP"],
    errors="coerce"
)


# ============================================================
# 6. BASIC GSMaP VALIDATION
# ============================================================

print()
print("=" * 80)
print("VALIDATING GSMaP MASTER")
print("=" * 80)

gsmap_duplicates = gsmap.duplicated(
    subset=["point_id", "Week_Start"]
).sum()

print(
    f"Duplicate GSMaP point-week rows: "
    f"{gsmap_duplicates}"
)

if gsmap_duplicates != 0:
    raise ValueError(
        "Duplicate point-week rows detected in GSMaP dataset."
    )

gsmap_missing_rainfall = gsmap["Rainfall_GSMaP"].isna().sum()

print(
    f"Missing GSMaP rainfall values: "
    f"{gsmap_missing_rainfall:,}"
)

if gsmap_missing_rainfall != 0:
    raise ValueError(
        "GSMaP rainfall contains missing values."
    )

print(
    f"Unique GSMaP points: "
    f"{gsmap['point_id'].nunique()}"
)

print(
    f"Unique GSMaP weeks: "
    f"{gsmap['Week_Start'].nunique()}"
)

print("PASS - GSMaP master validation completed")


# ============================================================
# 7. SORT BASE DATA
# ============================================================

df = (
    df
    .sort_values(
        by=["point_id", "Week_Start"]
    )
    .reset_index(drop=True)
)


# ============================================================
# 8. CREATE GSMaP RAINFALL-ONLY DATAFRAME
# ============================================================

gsmap_rainfall = gsmap[
    [
        "point_id",
        "Week_Start",
        "Rainfall_GSMaP",
        "GSMaP_Image_Count",
    ]
].copy()


# ============================================================
# 9. MERGE GSMaP RAINFALL
# ============================================================

print()
print("=" * 80)
print("MERGING GSMaP RAINFALL INTO MASTER DATASET")
print("=" * 80)

df = df.merge(
    gsmap_rainfall,
    on=[
        "point_id",
        "Week_Start",
    ],
    how="left",
    validate="one_to_one"
)

print(
    f"Rows after GSMaP merge: "
    f"{len(df):,}"
)

if len(df) != 78156:
    raise ValueError(
        "Unexpected row count after GSMaP merge. "
        f"Expected 78,156, got {len(df):,}."
    )

missing_gsmap_after_merge = (
    df["Rainfall_GSMaP"]
    .isna()
    .sum()
)

print(
    f"Missing GSMaP rainfall after merge: "
    f"{missing_gsmap_after_merge:,}"
)

if missing_gsmap_after_merge != 0:
    raise ValueError(
        "Missing GSMaP rainfall values remain after merge."
    )

print("PASS - GSMaP merge completed")


# ============================================================
# 10. REMOVE CHIRPS RAINFALL FROM ACTIVE FEATURE PIPELINE
# ============================================================

# The original combined master contains Rainfall_mm
# from CHIRPS. It must NOT be used by the GSMaP model.

df["Rainfall_mm_CHIRPS_Backup"] = df["Rainfall_mm"]

df["Rainfall_mm"] = df["Rainfall_GSMaP"]


# ============================================================
# 11. DUPLICATE CHECK AFTER MERGE
# ============================================================

duplicates = df.duplicated(
    subset=[
        "point_id",
        "Week_Start"
    ]
).sum()

print(
    f"Duplicate point-week rows after merge: "
    f"{duplicates}"
)

if duplicates != 0:
    raise ValueError(
        "Duplicate point-week rows detected after GSMaP merge."
    )


# ============================================================
# 12. TARGET: ACTUAL NEXT-WEEK NDVI
# ============================================================

print()
print("=" * 80)
print("CONSTRUCTING NEXT-WEEK NDVI TARGET")
print("=" * 80)

target_lookup = df[
    [
        "point_id",
        "Week_Start",
        "NDVI",
    ]
].copy()

target_lookup = target_lookup.rename(
    columns={
        "Week_Start": "Target_Week_Start",
        "NDVI": "NDVI_next_week",
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
        "Target_Week_Start",
    ],
    how="left",
    validate="one_to_one"
)

df["Current_NDVI_Available"] = (
    df["NDVI"].notna().astype(int)
)

df["Next_NDVI_Available"] = (
    df["NDVI_next_week"].notna().astype(int)
)

df["Usable_Target"] = (
    df["NDVI"].notna()
    & df["NDVI_next_week"].notna()
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

expected_target_rows = 33655

if int(df["Usable_Target"].sum()) != expected_target_rows:
    raise ValueError(
        "Unexpected usable target count. "
        f"Expected {expected_target_rows:,}, "
        f"got {int(df['Usable_Target'].sum()):,}."
    )

print(
    "PASS - Usable target count matches "
    "original CHIRPS experiment: 33,655"
)


# ============================================================
# 13. GROUPED DATA FOR LAGS
# ============================================================

grouped = df.groupby(
    "point_id",
    sort=False
)


# ============================================================
# 14. TEMPORAL FEATURES
# ============================================================

print()
print("=" * 80)
print("CREATING TEMPORAL FEATURES")
print("=" * 80)

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
    df["Year"]
    - 2017
)

df["Week_Sin"] = (
    np.sin(
        2
        * np.pi
        * df["Week_Of_Year"]
        / 52
    )
)

df["Week_Cos"] = (
    np.cos(
        2
        * np.pi
        * df["Week_Of_Year"]
        / 52
    )
)

df["Month_Sin"] = (
    np.sin(
        2
        * np.pi
        * df["Month"]
        / 12
    )
)

df["Month_Cos"] = (
    np.cos(
        2
        * np.pi
        * df["Month"]
        / 12
    )
)


# ============================================================
# 15. NDVI LAGS
# ============================================================

print()
print("=" * 80)
print("CREATING NDVI LAG FEATURES")
print("=" * 80)

for lag in [1, 2, 3, 4, 8]:

    df[f"NDVI_lag_{lag}"] = (
        grouped["NDVI"]
        .shift(lag)
    )


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
# 16. GSMaP RAINFALL LAGS + ROLLING
# ============================================================

print()
print("=" * 80)
print("CREATING GSMaP RAINFALL FEATURES")
print("=" * 80)

for lag in [1, 2, 4]:

    df[f"Rainfall_lag_{lag}"] = (
        grouped["Rainfall_GSMaP"]
        .shift(lag)
    )


# IMPORTANT:
# Shift first so rolling features contain only
# previous rainfall observations.
rainfall_previous = (
    grouped["Rainfall_GSMaP"]
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
# 17. TEMPERATURE LAGS + ROLLING
# ============================================================

print()
print("=" * 80)
print("CREATING TEMPERATURE FEATURES")
print("=" * 80)

for lag in [1, 2, 4]:

    df[f"Temperature_lag_{lag}"] = (
        grouped["Temperature_C"]
        .shift(lag)
    )


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
# 18. CURRENT-WEEK FEATURES
# ============================================================

df["Rainfall_Current"] = (
    df["Rainfall_GSMaP"]
)

df["Temperature_Current"] = (
    df["Temperature_C"]
)


# ============================================================
# 19. SPATIAL FEATURES
# ============================================================

df["Latitude"] = (
    df["latitude"]
)

df["Longitude"] = (
    df["longitude"]
)


# ============================================================
# 20. SELECT USABLE TARGET ROWS ONLY
# ============================================================

print()
print("=" * 80)
print("PREPARING SUPERVISED GSMaP DATASET")
print("=" * 80)

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

if len(model_df) != expected_target_rows:
    raise ValueError(
        "Final supervised population is not 33,655 rows."
    )


# ============================================================
# 21. TARGET VALIDATION
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

if target_min < -1 or target_max > 1:
    raise ValueError(
        "NDVI_next_week contains values outside [-1, 1]."
    )

print("PASS - Target values within [-1, 1]")


# ============================================================
# 22. FEATURE LIST
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


# ============================================================
# 23. FEATURE COUNT CHECK
# ============================================================

print()
print(
    f"Number of model features: "
    f"{len(feature_columns)}"
)

if len(feature_columns) != 33:
    raise ValueError(
        "Expected exactly 33 model features."
    )

print("PASS - 33 model features")


# ============================================================
# 24. FEATURE MISSINGNESS
# ============================================================

print()
print("=" * 80)
print("FEATURE AVAILABILITY")
print("=" * 80)

feature_missing = (
    model_df[feature_columns]
    .isna()
    .sum()
    .reset_index()
)

feature_missing.columns = [
    "feature",
    "missing_count",
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
# 25. IMPORTANT:
#     DO NOT REMOVE ROWS WITH MISSING FEATURES
# ============================================================

print()
print("=" * 80)
print("FEATURE FILTERING")
print("=" * 80)

print(
    "NO feature-based row deletion will be performed."
)

print(
    "All 33,655 supervised target rows are retained."
)

print(
    "Missing lag/rolling features will be handled later "
    "by the model-training pipeline."
)

print(
    "Imputation must be fitted on TRAINING data only "
    "to prevent data leakage."
)


# ============================================================
# 26. FINAL TARGET CHECK
# ============================================================

if model_df["NDVI_next_week"].isna().any():
    raise ValueError(
        "Missing target values remain."
    )

print(
    "PASS - No missing target values"
)


# ============================================================
# 27. FINAL OUTPUT COLUMNS
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
    .reset_index(drop=True)
)


# ============================================================
# 28. FINAL DATASET VALIDATION
# ============================================================

print()
print("=" * 80)
print("FINAL DATASET VALIDATION")
print("=" * 80)

print(
    f"Final rows: "
    f"{len(model_df):,}"
)

print(
    f"Unique points: "
    f"{model_df['point_id'].nunique()}"
)

print(
    f"Unique years: "
    f"{model_df['Year'].nunique()}"
)

print(
    f"Unique weeks: "
    f"{model_df['Week_Start'].nunique()}"
)

if len(model_df) != 33655:
    raise ValueError(
        f"Expected 33,655 rows, got {len(model_df):,}"
    )

if model_df["point_id"].nunique() != 167:
    raise ValueError(
        "Expected 167 unique sampling points."
    )

if model_df["Year"].nunique() != 9:
    raise ValueError(
        "Expected 9 years from 2017 to 2025."
    )

print(
    "PASS - Final dataset has exactly 33,655 rows"
)


# ============================================================
# 29. FEATURE STATISTICS
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
# 30. TARGET STATISTICS
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
        model_df["NDVI_next_week"].count(),
        model_df["NDVI_next_week"].mean(),
        model_df["NDVI_next_week"].std(),
        model_df["NDVI_next_week"].min(),
        model_df["NDVI_next_week"].quantile(0.25),
        model_df["NDVI_next_week"].median(),
        model_df["NDVI_next_week"].quantile(0.75),
        model_df["NDVI_next_week"].max(),
    ],
})

target_statistics.to_csv(
    TARGET_STATISTICS_FILE,
    index=False
)


# ============================================================
# 31. TARGET CORRELATIONS
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
            "NDVI_next_week": "correlation"
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
# 32. YEAR-WISE SUMMARY
# ============================================================

year_summary = (
    model_df
    .groupby("Year")
    .size()
    .reset_index(
        name="supervised_rows"
    )
)

year_summary["percentage"] = (
    year_summary["supervised_rows"]
    / len(model_df)
    * 100
)


# ============================================================
# 33. GSMaP RAINFALL SUMMARY
# ============================================================

gsmap_rainfall_summary = {
    "mean": model_df["Rainfall_Current"].mean(),
    "median": model_df["Rainfall_Current"].median(),
    "std": model_df["Rainfall_Current"].std(),
    "min": model_df["Rainfall_Current"].min(),
    "max": model_df["Rainfall_Current"].max(),
}


# ============================================================
# 34. REPORT
# ============================================================

report_lines = [

    "AGRIVISION AI - GSMaP FEATURE ENGINEERING REPORT",

    "2017–2025",

    "",

    f"Original combined rows: {len(df):,}",

    f"GSMaP rainfall rows: {len(gsmap):,}",

    f"Usable target rows: "
    f"{int(df['Usable_Target'].sum()):,}",

    f"Final ML rows: {len(model_df):,}",

    "",

    "Rainfall source: GSMaP",

    "CHIRPS rainfall was NOT used as an active model feature.",

    "",

    "Target: NDVI_next_week",

    "Target definition: actual NDVI for the same point_id "
    "exactly 7 days after the current Week_Start.",

    "",

    "No interpolation or fabricated NDVI target values were used.",

    "No future target information was used as a predictor.",

    "",

    f"Number of model features: "
    f"{len(feature_columns)}",

    "",

    "Missing feature values were intentionally retained.",

    "Rows were NOT removed because of missing lag/rolling features.",

    "Missing-feature handling belongs to the model-training pipeline.",

    "The imputer must be fitted using training data only.",

    "",

    "GSMaP rainfall statistics:",

    f"Mean:   {gsmap_rainfall_summary['mean']:.6f}",

    f"Median: {gsmap_rainfall_summary['median']:.6f}",

    f"Std:    {gsmap_rainfall_summary['std']:.6f}",

    f"Min:    {gsmap_rainfall_summary['min']:.6f}",

    f"Max:    {gsmap_rainfall_summary['max']:.6f}",

    "",

    "Year-wise supervised rows:",
]

for _, row in year_summary.iterrows():

    report_lines.append(
        f"{int(row['Year'])}: "
        f"{int(row['supervised_rows']):,}"
    )


report_lines.extend([
    "",
    "Expected supervised population: 33,655",
    f"Actual supervised population: {len(model_df):,}",
    "",
    "STATUS: READY FOR GSMaP MODEL TRAINING",
])


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(report_lines)
    )


# ============================================================
# 35. SAVE FINAL FEATURE DATASET
# ============================================================

model_df.to_csv(
    FEATURE_DATASET,
    index=False
)


# ============================================================
# 36. FINAL CONSOLE SUMMARY
# ============================================================

print()
print("=" * 80)
print("GSMaP FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
print("=" * 80)

print()
print(
    f"Base rows:              {len(df):,}"
)

print(
    f"GSMaP rainfall rows:    {len(gsmap):,}"
)

print(
    f"Usable target rows:     "
    f"{int(df['Usable_Target'].sum()):,}"
)

print(
    f"Final ML rows:          {len(model_df):,}"
)

print(
    f"Number of features:     {len(feature_columns)}"
)

print(
    f"Unique points:          "
    f"{model_df['point_id'].nunique()}"
)

print(
    f"Unique years:           "
    f"{model_df['Year'].nunique()}"
)

print()
print("Year-wise supervised rows:")
print(
    year_summary.to_string(
        index=False
    )
)

print()
print("GSMaP rainfall statistics:")
print(
    f"Mean:   {gsmap_rainfall_summary['mean']:.6f}"
)
print(
    f"Median: {gsmap_rainfall_summary['median']:.6f}"
)
print(
    f"Std:    {gsmap_rainfall_summary['std']:.6f}"
)
print(
    f"Min:    {gsmap_rainfall_summary['min']:.6f}"
)
print(
    f"Max:    {gsmap_rainfall_summary['max']:.6f}"
)

print()
print("Missing-feature rows were NOT removed.")
print(
    "The dataset retains all 33,655 supervised target rows."
)

print()
print("Output:")
print(FEATURE_DATASET)

print()
print("Report:")
print(REPORT_FILE)

print()
print("=" * 80)
print("READY FOR GSMaP MODEL TRAINING")
print("=" * 80)