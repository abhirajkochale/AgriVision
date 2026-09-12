from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# AGRIVISION AI
# MODERATE ML FEATURE ENGINEERING
# Maharashtra 2021-2025
# ============================================================

print("=" * 70)
print("AGRIVISION - MODERATE ML FEATURE ENGINEERING")
print("=" * 70)


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_DIR
    / "dataset"
    / "AgriVision_Maharashtra_Combined_Master_2021_2025.csv"
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

OUTPUT_FILE = (
    PROJECT_DIR
    / "dataset"
    / "AgriVision_Maharashtra_ML_Ready_Moderate_2021_2025.csv"
)

REPORT_FILE = (
    OUTPUT_DIR
    / "moderate_feature_engineering_report.txt"
)


# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nCombined master dataset not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print()
print("Input dataset:")
print(INPUT_FILE)

print()
print(f"Input rows: {len(df):,}")
print(f"Input columns: {len(df.columns)}")


# ------------------------------------------------------------
# 3. DATE CONVERSION
# ------------------------------------------------------------

df["Week_Start"] = pd.to_datetime(
    df["Week_Start"],
    errors="coerce"
)

df["Week_End"] = pd.to_datetime(
    df["Week_End"],
    errors="coerce"
)


# ------------------------------------------------------------
# 4. SORT
# ------------------------------------------------------------

df = df.sort_values(
    ["point_id", "Week_Start"]
).reset_index(drop=True)


# ------------------------------------------------------------
# 5. CREATE NEXT-WEEK NDVI TARGET
# ------------------------------------------------------------

print()
print("=" * 70)
print("CREATING NEXT-WEEK NDVI TARGET")
print("=" * 70)

df["NDVI_next_week"] = (
    df.groupby("point_id")["NDVI"]
    .shift(-1)
)

df["Next_Week_Start"] = (
    df.groupby("point_id")["Week_Start"]
    .shift(-1)
)

df["Next_Week_Gap_Days"] = (
    df["Next_Week_Start"]
    - df["Week_Start"]
).dt.days


# ------------------------------------------------------------
# 6. VALID TARGET
# ------------------------------------------------------------

df["Valid_NDVI_Target"] = (
    df["NDVI"].notna()
    & df["NDVI_next_week"].notna()
    & (df["Next_Week_Gap_Days"] == 7)
)

valid_target_rows = int(
    df["Valid_NDVI_Target"].sum()
)

print(
    f"Valid consecutive NDVI target rows: "
    f"{valid_target_rows:,}"
)


# ------------------------------------------------------------
# 7. KEEP ONLY VALID TARGET ROWS
# ------------------------------------------------------------

ml = df[
    df["Valid_NDVI_Target"]
].copy()

print(
    f"Rows after target validation: "
    f"{len(ml):,}"
)


# ------------------------------------------------------------
# 8. CURRENT FEATURES
# ------------------------------------------------------------

required_current_features = [
    "NDVI",
    "Rainfall_mm",
    "Temperature_C",
]

for column in required_current_features:

    if column not in ml.columns:
        raise ValueError(
            f"Missing required feature: {column}"
        )


# ------------------------------------------------------------
# 9. LAG FEATURES
# ------------------------------------------------------------

print()
print("=" * 70)
print("CREATING LAG FEATURES")
print("=" * 70)


# Only one NDVI lag is required in the moderate pipeline.
ml["NDVI_lag_1"] = (
    df.groupby("point_id")["NDVI"]
    .shift(1)
    .loc[ml.index]
)


# Climate lags remain available because rainfall
# and temperature are complete.

for variable in [
    "Rainfall_mm",
    "Temperature_C",
]:

    for lag in [1, 2, 3]:

        column_name = (
            f"{variable}_lag_{lag}"
        )

        ml[column_name] = (
            df.groupby("point_id")[variable]
            .shift(lag)
            .loc[ml.index]
        )


# ------------------------------------------------------------
# 10. ROLLING CLIMATE FEATURES
# ------------------------------------------------------------

print()
print("=" * 70)
print("CREATING ROLLING CLIMATE FEATURES")
print("=" * 70)


for variable in [
    "Rainfall_mm",
    "Temperature_C",
]:

    grouped = (
        df.groupby("point_id")[variable]
    )

    for window in [3, 4]:

        column_name = (
            f"{variable}_rolling_{window}w"
        )

        rolling_values = (
            grouped
            .rolling(
                window=window,
                min_periods=window
            )
            .mean()
            .reset_index(
                level=0,
                drop=True
            )
        )

        ml[column_name] = (
            rolling_values.loc[ml.index]
        )


# ------------------------------------------------------------
# 11. TEMPORAL FEATURES
# ------------------------------------------------------------

print()
print("=" * 70)
print("CREATING TEMPORAL FEATURES")
print("=" * 70)

ml["Month"] = (
    ml["Week_Start"].dt.month
)

ml["Quarter"] = (
    ml["Week_Start"].dt.quarter
)

ml["Week_of_Year"] = (
    ml["Week_Number"]
)

ml["Monsoon_Season"] = (
    ml["Month"]
    .isin([6, 7, 8, 9])
    .astype(int)
)

ml["Week_Sin"] = np.sin(
    2 * np.pi * ml["Week_of_Year"] / 52
)

ml["Week_Cos"] = np.cos(
    2 * np.pi * ml["Week_of_Year"] / 52
)


# ------------------------------------------------------------
# 12. SPATIAL FEATURES
# ------------------------------------------------------------

ml["latitude"] = pd.to_numeric(
    ml["latitude"],
    errors="coerce"
)

ml["longitude"] = pd.to_numeric(
    ml["longitude"],
    errors="coerce"
)


# ------------------------------------------------------------
# 13. FINAL FEATURE LIST
# ------------------------------------------------------------

feature_columns = [

    # Current conditions
    "NDVI",
    "Rainfall_mm",
    "Temperature_C",

    # Recent vegetation
    "NDVI_lag_1",

    # Rainfall history
    "Rainfall_mm_lag_1",
    "Rainfall_mm_lag_2",
    "Rainfall_mm_lag_3",

    # Temperature history
    "Temperature_C_lag_1",
    "Temperature_C_lag_2",
    "Temperature_C_lag_3",

    # Rolling rainfall
    "Rainfall_mm_rolling_3w",
    "Rainfall_mm_rolling_4w",

    # Rolling temperature
    "Temperature_C_rolling_3w",
    "Temperature_C_rolling_4w",

    # Seasonal features
    "Month",
    "Quarter",
    "Week_of_Year",
    "Monsoon_Season",
    "Week_Sin",
    "Week_Cos",

    # Spatial features
    "latitude",
    "longitude",
]


# ------------------------------------------------------------
# 14. CHECK FEATURES EXIST
# ------------------------------------------------------------

missing_features = [
    column
    for column in feature_columns
    if column not in ml.columns
]

if missing_features:
    raise ValueError(
        "Missing feature columns:\n"
        + "\n".join(missing_features)
    )


# ------------------------------------------------------------
# 15. FEATURE MISSINGNESS
# ------------------------------------------------------------

feature_missing = (
    ml[feature_columns]
    .isna()
    .sum()
    .sort_values(
        ascending=False
    )
)

feature_missing_pct = (
    feature_missing
    / len(ml)
    * 100
)

feature_missing_report = pd.DataFrame({
    "missing_count": feature_missing,
    "missing_percentage": feature_missing_pct,
})

feature_missing_report.to_csv(
    OUTPUT_DIR
    / "moderate_feature_missing_values.csv"
)


print()
print("=" * 70)
print("FEATURE MISSING VALUES")
print("=" * 70)

print(
    feature_missing_report.to_string()
)


# ------------------------------------------------------------
# 16. STRICT COMPLETENESS ONLY FOR REQUIRED FEATURES
# ------------------------------------------------------------

rows_before = len(ml)

ml = ml.dropna(
    subset=feature_columns
).copy()

rows_after = len(ml)

rows_removed = (
    rows_before - rows_after
)

print()
print(
    f"Rows before feature filter: "
    f"{rows_before:,}"
)

print(
    f"Rows after feature filter:  "
    f"{rows_after:,}"
)

print(
    f"Rows removed:               "
    f"{rows_removed:,}"
)


# ------------------------------------------------------------
# 17. TARGET VALIDATION
# ------------------------------------------------------------

missing_targets = (
    ml["NDVI_next_week"]
    .isna()
    .sum()
)

invalid_targets = (
    (ml["NDVI_next_week"] < -1)
    | (ml["NDVI_next_week"] > 1)
).sum()

bad_target_gaps = (
    ml["Next_Week_Gap_Days"] != 7
).sum()


print()
print("=" * 70)
print("TARGET VALIDATION")
print("=" * 70)

print(
    f"Missing targets: {missing_targets}"
)

print(
    f"Invalid target values: {invalid_targets}"
)

print(
    f"Target gaps != 7 days: {bad_target_gaps}"
)

if missing_targets != 0:
    raise ValueError(
        "Missing NDVI_next_week values detected."
    )

if invalid_targets != 0:
    raise ValueError(
        "Invalid NDVI_next_week values detected."
    )

if bad_target_gaps != 0:
    raise ValueError(
        "Non-consecutive target gaps detected."
    )


# ------------------------------------------------------------
# 18. REMOVE TEMPORARY COLUMNS
# ------------------------------------------------------------

temporary_columns = [
    "NDVI_next_week",
    "Next_Week_Start",
    "Next_Week_Gap_Days",
    "Valid_NDVI_Target",
]

# Keep NDVI_next_week but remove the other temporary columns.

ml = ml.drop(
    columns=[
        "Next_Week_Start",
        "Next_Week_Gap_Days",
        "Valid_NDVI_Target",
    ]
)


# ------------------------------------------------------------
# 19. FINAL COLUMN ORDER
# ------------------------------------------------------------

identifier_columns = [
    "Week_Start",
    "Week_End",
    "Week_Number",
    "Year",
    "point_id",
    "latitude",
    "longitude",
    "district",
    "sampling_priority",
    "region_type",
]

target_column = [
    "NDVI_next_week"
]

final_columns = (
    identifier_columns
    + feature_columns
    + target_column
)

final_columns = list(
    dict.fromkeys(final_columns)
)

ml = ml[
    final_columns
].copy()


# ------------------------------------------------------------
# 20. SORT
# ------------------------------------------------------------

ml = ml.sort_values(
    ["Week_Start", "point_id"]
).reset_index(
    drop=True
)


# ------------------------------------------------------------
# 21. DUPLICATE CHECK
# ------------------------------------------------------------

duplicates = (
    ml.duplicated(
        subset=[
            "point_id",
            "Week_Start"
        ]
    ).sum()
)

print()
print(
    f"Duplicate point-week rows: "
    f"{duplicates}"
)

if duplicates != 0:
    raise ValueError(
        "Duplicate point-week rows detected."
    )


# ------------------------------------------------------------
# 22. YEAR COUNTS
# ------------------------------------------------------------

year_counts = (
    ml.groupby("Year")
    .size()
    .sort_index()
)

print()
print("=" * 70)
print("MODERATE ML ROWS BY YEAR")
print("=" * 70)

print(
    year_counts.to_string()
)


# ------------------------------------------------------------
# 23. TARGET STATISTICS
# ------------------------------------------------------------

target_statistics = (
    ml["NDVI_next_week"]
    .describe()
    .to_frame()
    .T
)

target_statistics.to_csv(
    OUTPUT_DIR
    / "moderate_target_statistics.csv",
    index=False
)


# ------------------------------------------------------------
# 24. FEATURE STATISTICS
# ------------------------------------------------------------

feature_statistics = (
    ml[
        feature_columns
        + ["NDVI_next_week"]
    ]
    .describe()
    .T
)

feature_statistics.to_csv(
    OUTPUT_DIR
    / "moderate_feature_statistics.csv"
)


# ------------------------------------------------------------
# 25. FEATURE-TARGET CORRELATIONS
# ------------------------------------------------------------

target_correlations = (
    ml[
        feature_columns
        + ["NDVI_next_week"]
    ]
    .corr()["NDVI_next_week"]
    .sort_values(
        ascending=False
    )
)

target_correlations.to_csv(
    OUTPUT_DIR
    / "moderate_target_correlations.csv"
)


# ------------------------------------------------------------
# 26. REPORT
# ------------------------------------------------------------

report = []

report.append(
    "AGRIVISION AI - MODERATE FEATURE ENGINEERING REPORT"
)

report.append(
    "=" * 70
)

report.append(
    f"Input rows: {len(df):,}"
)

report.append(
    f"Valid consecutive target rows: "
    f"{valid_target_rows:,}"
)

report.append(
    f"Final ML-ready rows: "
    f"{len(ml):,}"
)

report.append(
    f"Rows removed by feature completeness: "
    f"{rows_removed:,}"
)

report.append(
    ""
)

report.append(
    "Target:"
)

report.append(
    "NDVI_next_week = actual observed NDVI "
    "exactly 7 days later"
)

report.append(
    ""
)

report.append(
    "Moderate feature policy:"
)

report.append(
    "Current NDVI + rainfall + temperature"
)

report.append(
    "One NDVI lag"
)

report.append(
    "Three rainfall lags"
)

report.append(
    "Three temperature lags"
)

report.append(
    "Three- and four-week climate rolling means"
)

report.append(
    "Seasonal and spatial features"
)

report.append(
    ""
)

report.append(
    "No interpolation"
)

report.append(
    "No fabricated NDVI values"
)

report.append(
    "No future information used in predictors"
)

report.append(
    ""
)

report.append(
    "Feature columns:"
)

for column in feature_columns:

    report.append(
        f"- {column}"
    )


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(report)
    )


# ------------------------------------------------------------
# 27. SAVE
# ------------------------------------------------------------

ml.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# 28. FINAL SUMMARY
# ------------------------------------------------------------

print()
print("=" * 70)
print("MODERATE ML-READY DATASET CREATED")
print("=" * 70)

print()
print("Output:")
print(OUTPUT_FILE)

print()
print(
    f"Final rows:    {len(ml):,}"
)

print(
    f"Final columns: {len(ml.columns):,}"
)

print()
print(
    "Rows by year:"
)

print(
    year_counts.to_string()
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)