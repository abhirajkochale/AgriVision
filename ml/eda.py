from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# AGRIVISION AI
# EXPLORATORY DATA ANALYSIS
# Maharashtra Combined Master Dataset
# ============================================================

print("=" * 70)
print("AGRIVISION - EXPLORATORY DATA ANALYSIS")
print("=" * 70)


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATASET_FILE = (
    PROJECT_DIR
    / "dataset"
    / "AgriVision_Maharashtra_Combined_Master_2021_2025.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "ml"
    / "eda_outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print()
print("Dataset:")
print(DATASET_FILE)

print()
print("Output directory:")
print(OUTPUT_DIR)


# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

if not DATASET_FILE.exists():
    raise FileNotFoundError(
        f"\nCombined master dataset not found:\n{DATASET_FILE}"
    )

df = pd.read_csv(DATASET_FILE)

print()
print("=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)

print(f"Rows:    {len(df):,}")
print(f"Columns: {len(df.columns)}")


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
# 4. BASIC SCHEMA
# ------------------------------------------------------------

schema = pd.DataFrame({
    "column": df.columns,
    "dtype": [str(df[col].dtype) for col in df.columns],
    "missing": [df[col].isna().sum() for col in df.columns],
    "missing_pct": [
        df[col].isna().mean() * 100
        for col in df.columns
    ],
    "unique_values": [
        df[col].nunique(dropna=True)
        for col in df.columns
    ],
})

schema.to_csv(
    OUTPUT_DIR / "schema.csv",
    index=False
)


# ------------------------------------------------------------
# 5. MISSING VALUES
# ------------------------------------------------------------

missing = (
    df.isna()
    .sum()
    .sort_values(ascending=False)
)

missing_pct = (
    df.isna()
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

missing_report = pd.DataFrame({
    "missing_count": missing,
    "missing_percentage": missing_pct
})

missing_report.to_csv(
    OUTPUT_DIR / "missing_values.csv"
)


print()
print("Missing values:")
print(missing_report.to_string())


# ------------------------------------------------------------
# 6. NUMERIC DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

numeric_columns = [
    "NDVI",
    "Rainfall_mm",
    "Temperature_C",
    "Sentinel2_Images",
    "CHIRPS_Days",
    "ERA5_Days",
]

numeric_columns = [
    col for col in numeric_columns
    if col in df.columns
]

descriptive_statistics = (
    df[numeric_columns]
    .describe()
    .T
)

descriptive_statistics.to_csv(
    OUTPUT_DIR / "descriptive_statistics.csv"
)


print()
print("=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)

print(descriptive_statistics.to_string())


# ------------------------------------------------------------
# 7. DATASET VALIDATION
# ------------------------------------------------------------

print()
print("=" * 70)
print("RANGE VALIDATION")
print("=" * 70)


ndvi_invalid = (
    (df["NDVI"].notna())
    & (
        (df["NDVI"] < -1)
        | (df["NDVI"] > 1)
    )
).sum()

rainfall_invalid = (
    (df["Rainfall_mm"].notna())
    & (df["Rainfall_mm"] < 0)
).sum()

temperature_invalid = (
    df["Temperature_C"].notna()
    & ~df["Temperature_C"].between(
        -50,
        60
    )
).sum()


print(f"Invalid NDVI values: {ndvi_invalid}")
print(f"Negative rainfall values: {rainfall_invalid}")
print(
    "Temperature values outside "
    "-50°C to 60°C:",
    temperature_invalid
)


# ------------------------------------------------------------
# 8. NDVI COVERAGE
# ------------------------------------------------------------

total_rows = len(df)

ndvi_available = df["NDVI"].notna().sum()

ndvi_missing = df["NDVI"].isna().sum()

ndvi_coverage = (
    ndvi_available / total_rows
) * 100


print()
print("=" * 70)
print("NDVI COVERAGE")
print("=" * 70)

print(f"Total point-weeks: {total_rows:,}")
print(f"NDVI available:   {ndvi_available:,}")
print(f"NDVI missing:     {ndvi_missing:,}")
print(f"NDVI coverage:    {ndvi_coverage:.2f}%")


# ------------------------------------------------------------
# 9. YEAR-WISE SUMMARY
# ------------------------------------------------------------

year_summary = (
    df.groupby("Year")
    .agg(
        records=("point_id", "size"),
        points=("point_id", "nunique"),
        weeks=("Week_Start", "nunique"),
        ndvi_available=("NDVI", "count"),
        ndvi_mean=("NDVI", "mean"),
        rainfall_mean=("Rainfall_mm", "mean"),
        temperature_mean=("Temperature_C", "mean"),
    )
)

year_summary["ndvi_coverage_pct"] = (
    year_summary["ndvi_available"]
    / year_summary["records"]
    * 100
)

year_summary.to_csv(
    OUTPUT_DIR / "yearly_summary.csv"
)


print()
print("=" * 70)
print("YEAR-WISE SUMMARY")
print("=" * 70)

print(year_summary.to_string())


# ------------------------------------------------------------
# 10. DISTRICT SUMMARY
# ------------------------------------------------------------

district_summary = (
    df.groupby("district")
    .agg(
        records=("point_id", "size"),
        points=("point_id", "nunique"),
        ndvi_available=("NDVI", "count"),
        ndvi_mean=("NDVI", "mean"),
        rainfall_mean=("Rainfall_mm", "mean"),
        temperature_mean=("Temperature_C", "mean"),
    )
)

district_summary["ndvi_coverage_pct"] = (
    district_summary["ndvi_available"]
    / district_summary["records"]
    * 100
)

district_summary = district_summary.sort_values(
    "ndvi_coverage_pct"
)

district_summary.to_csv(
    OUTPUT_DIR / "district_summary.csv"
)


# ------------------------------------------------------------
# 11. POINT-LEVEL NDVI COVERAGE
# ------------------------------------------------------------

point_summary = (
    df.groupby("point_id")
    .agg(
        records=("Week_Start", "size"),
        ndvi_available=("NDVI", "count"),
        ndvi_mean=("NDVI", "mean"),
        rainfall_mean=("Rainfall_mm", "mean"),
        temperature_mean=("Temperature_C", "mean"),
        district=("district", "first"),
    )
)

point_summary["ndvi_coverage_pct"] = (
    point_summary["ndvi_available"]
    / point_summary["records"]
    * 100
)

point_summary.to_csv(
    OUTPUT_DIR / "point_summary.csv"
)


# ------------------------------------------------------------
# 12. WEEK-LEVEL NDVI COVERAGE
# ------------------------------------------------------------

weekly_coverage = (
    df.groupby(
        ["Year", "Week_Number"]
    )
    .agg(
        total_points=("point_id", "size"),
        ndvi_available=("NDVI", "count"),
    )
    .reset_index()
)

weekly_coverage["ndvi_coverage_pct"] = (
    weekly_coverage["ndvi_available"]
    / weekly_coverage["total_points"]
    * 100
)

weekly_coverage.to_csv(
    OUTPUT_DIR / "weekly_ndvi_coverage.csv",
    index=False
)


# ------------------------------------------------------------
# 13. CONSECUTIVE NDVI OBSERVATIONS
# ------------------------------------------------------------

print()
print("=" * 70)
print("CONSECUTIVE NDVI ANALYSIS")
print("=" * 70)

df = df.sort_values(
    ["point_id", "Week_Start"]
).reset_index(drop=True)


df["next_week_ndvi"] = (
    df.groupby("point_id")["NDVI"]
    .shift(-1)
)


df["next_week_date"] = (
    df.groupby("point_id")["Week_Start"]
    .shift(-1)
)


# A valid consecutive pair requires:
# 1. Current NDVI exists
# 2. Next NDVI exists
# 3. Next timestamp is exactly 7 days later

df["is_consecutive_ndvi_pair"] = (
    df["NDVI"].notna()
    & df["next_week_ndvi"].notna()
    & (
        df["next_week_date"]
        == df["Week_Start"]
        + pd.Timedelta(days=7)
    )
)

consecutive_pairs = (
    df["is_consecutive_ndvi_pair"].sum()
)

print(
    "Usable current-week → next-week NDVI pairs:",
    f"{consecutive_pairs:,}"
)

print(
    "Potential NDVI observations:",
    f"{ndvi_available:,}"
)


# ------------------------------------------------------------
# 14. TARGET AVAILABILITY BY YEAR
# ------------------------------------------------------------

target_by_year = (
    df.groupby("Year")
    .agg(
        current_ndvi=("NDVI", "count"),
        usable_next_week_pairs=(
            "is_consecutive_ndvi_pair",
            "sum"
        ),
    )
)

target_by_year.to_csv(
    OUTPUT_DIR / "target_availability_by_year.csv"
)


print()
print(
    target_by_year.to_string()
)


# ------------------------------------------------------------
# 15. CORRELATION MATRIX
# ------------------------------------------------------------

correlation_columns = [
    "NDVI",
    "Rainfall_mm",
    "Temperature_C",
]

correlation_matrix = (
    df[correlation_columns]
    .corr()
)

correlation_matrix.to_csv(
    OUTPUT_DIR / "correlation_matrix.csv"
)


print()
print("=" * 70)
print("CORRELATION MATRIX")
print("=" * 70)

print(correlation_matrix.to_string())


# ------------------------------------------------------------
# 16. WEEKLY MEAN TRENDS
# ------------------------------------------------------------

weekly_trends = (
    df.groupby("Week_Start")
    .agg(
        NDVI_mean=("NDVI", "mean"),
        Rainfall_mean=("Rainfall_mm", "mean"),
        Temperature_mean=("Temperature_C", "mean"),
    )
    .reset_index()
)

weekly_trends.to_csv(
    OUTPUT_DIR / "weekly_trends.csv",
    index=False
)


# ------------------------------------------------------------
# 17. PLOT 1 - NDVI DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

df["NDVI"].dropna().hist(bins=50)

plt.xlabel("NDVI")
plt.ylabel("Frequency")
plt.title("AgriVision NDVI Distribution")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "ndvi_distribution.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 18. PLOT 2 - RAINFALL DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

df["Rainfall_mm"].hist(bins=50)

plt.xlabel("Weekly Rainfall (mm)")
plt.ylabel("Frequency")
plt.title("AgriVision Weekly Rainfall Distribution")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "rainfall_distribution.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 19. PLOT 3 - TEMPERATURE DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

df["Temperature_C"].hist(bins=50)

plt.xlabel("Weekly Mean Temperature (°C)")
plt.ylabel("Frequency")
plt.title("AgriVision Weekly Temperature Distribution")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "temperature_distribution.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 20. PLOT 4 - NDVI TREND
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    weekly_trends["Week_Start"],
    weekly_trends["NDVI_mean"]
)

plt.xlabel("Week")
plt.ylabel("Mean NDVI")
plt.title("Mean Weekly NDVI - Maharashtra")

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "ndvi_temporal_trend.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 21. PLOT 5 - RAINFALL TREND
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    weekly_trends["Week_Start"],
    weekly_trends["Rainfall_mean"]
)

plt.xlabel("Week")
plt.ylabel("Mean Weekly Rainfall (mm)")
plt.title("Mean Weekly Rainfall - Maharashtra")

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "rainfall_temporal_trend.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 22. PLOT 6 - TEMPERATURE TREND
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    weekly_trends["Week_Start"],
    weekly_trends["Temperature_mean"]
)

plt.xlabel("Week")
plt.ylabel("Mean Temperature (°C)")
plt.title("Mean Weekly Temperature - Maharashtra")

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "temperature_temporal_trend.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 23. OUTLIER SUMMARY
# ------------------------------------------------------------

outlier_summary = {}

for column in [
    "NDVI",
    "Rainfall_mm",
    "Temperature_C",
]:

    series = df[column].dropna()

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outlier_count = (
        (series < lower)
        | (series > upper)
    ).sum()

    outlier_summary[column] = {
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "lower_bound": lower,
        "upper_bound": upper,
        "outlier_count": outlier_count,
        "outlier_percentage": (
            outlier_count
            / len(series)
            * 100
        ),
    }

outlier_df = pd.DataFrame(
    outlier_summary
).T

outlier_df.to_csv(
    OUTPUT_DIR / "outlier_summary.csv"
)


# ------------------------------------------------------------
# 24. EDA REPORT
# ------------------------------------------------------------

report_lines = []

report_lines.append(
    "AGRIVISION AI - EDA REPORT"
)

report_lines.append(
    "=" * 60
)

report_lines.append(
    f"Rows: {len(df):,}"
)

report_lines.append(
    f"Columns: {len(df.columns) - 4:,}"
)

report_lines.append(
    f"Points: {df['point_id'].nunique()}"
)

report_lines.append(
    f"Weeks: {df['Week_Start'].nunique()}"
)

report_lines.append(
    "Years: "
    + ", ".join(
        map(
            str,
            sorted(df["Year"].dropna().unique())
        )
    )
)

report_lines.append(
    f"NDVI available: {ndvi_available:,}"
)

report_lines.append(
    f"NDVI missing: {ndvi_missing:,}"
)

report_lines.append(
    f"NDVI coverage: {ndvi_coverage:.2f}%"
)

report_lines.append(
    f"Usable consecutive NDVI pairs: "
    f"{consecutive_pairs:,}"
)

report_lines.append(
    ""
)

report_lines.append(
    "Important validation:"
)

report_lines.append(
    f"Invalid NDVI values: {ndvi_invalid}"
)

report_lines.append(
    f"Negative rainfall values: "
    f"{rainfall_invalid}"
)

report_lines.append(
    f"Extreme temperature values: "
    f"{temperature_invalid}"
)


with open(
    OUTPUT_DIR / "eda_report.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(report_lines)
    )


# ------------------------------------------------------------
# 25. CLEAN END
# ------------------------------------------------------------

print()
print("=" * 70)
print("EDA COMPLETE")
print("=" * 70)

print()
print("Outputs created in:")
print(OUTPUT_DIR)

print()
print(
    "Most important result:"
)

print(
    "Usable current-week → next-week NDVI pairs:",
    f"{consecutive_pairs:,}"
)

print()
print("DONE")
print("=" * 70)