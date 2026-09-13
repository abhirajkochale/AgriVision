from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# AGRIVISION AI
# EDA FOR COMBINED MASTER DATASET
# 2017–2025
# ============================================================


print("=" * 75)
print("AGRIVISION - EXPLORATORY DATA ANALYSIS")
print("Combined Master Dataset 2017–2025")
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
    / "eda_outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print()
print("Project folder:")
print(PROJECT_DIR)

print()
print("Dataset:")
print(DATASET_FILE)

print()
print("Output folder:")
print(OUTPUT_DIR)


# ============================================================
# 2. CHECK INPUT
# ============================================================

if not DATASET_FILE.exists():

    raise FileNotFoundError(
        f"\nCombined dataset not found:\n{DATASET_FILE}"
    )

print()
print("PASS - Combined master dataset found")


# ============================================================
# 3. LOAD DATASET
# ============================================================

print()
print("=" * 75)
print("LOADING DATASET")
print("=" * 75)

df = pd.read_csv(DATASET_FILE)

print()
print(f"Rows:    {len(df):,}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# 4. BASIC DATA PREPARATION
# ============================================================

df["Week_Start"] = pd.to_datetime(
    df["Week_Start"],
    errors="coerce"
)

df["Week_End"] = pd.to_datetime(
    df["Week_End"],
    errors="coerce"
)

df["Year"] = pd.to_numeric(
    df["Year"],
    errors="coerce"
)

df["Week_Number"] = pd.to_numeric(
    df["Week_Number"],
    errors="coerce"
)

df["NDVI"] = pd.to_numeric(
    df["NDVI"],
    errors="coerce"
)

df["Rainfall_mm"] = pd.to_numeric(
    df["Rainfall_mm"],
    errors="coerce"
)

df["Temperature_C"] = pd.to_numeric(
    df["Temperature_C"],
    errors="coerce"
)


# ============================================================
# 5. DATASET OVERVIEW
# ============================================================

print()
print("=" * 75)
print("DATASET OVERVIEW")
print("=" * 75)

overview = pd.DataFrame({
    "Metric": [
        "Rows",
        "Columns",
        "Unique points",
        "Unique weeks",
        "Minimum date",
        "Maximum date",
        "Years",
        "Districts",
    ],
    "Value": [
        len(df),
        len(df.columns),
        df["point_id"].nunique(),
        df["Week_Start"].nunique(),
        df["Week_Start"].min().date(),
        df["Week_Start"].max().date(),
        ", ".join(
            map(
                str,
                sorted(
                    df["Year"].dropna().astype(int).unique()
                )
            )
        ),
        df["district"].nunique(),
    ]
})

print()
print(overview.to_string(index=False))

overview.to_csv(
    OUTPUT_DIR / "dataset_overview.csv",
    index=False
)


# ============================================================
# 6. SCHEMA
# ============================================================

print()
print("=" * 75)
print("SCHEMA")
print("=" * 75)

schema = pd.DataFrame({
    "column": df.columns,
    "dtype": [
        str(df[column].dtype)
        for column in df.columns
    ],
    "non_null": [
        df[column].notna().sum()
        for column in df.columns
    ],
    "missing": [
        df[column].isna().sum()
        for column in df.columns
    ],
    "unique_values": [
        df[column].nunique(dropna=True)
        for column in df.columns
    ],
})

print(schema.to_string(index=False))

schema.to_csv(
    OUTPUT_DIR / "schema.csv",
    index=False
)


# ============================================================
# 7. MISSING VALUE ANALYSIS
# ============================================================

print()
print("=" * 75)
print("MISSING VALUE ANALYSIS")
print("=" * 75)

missing = (
    df.isna()
    .sum()
    .reset_index()
)

missing.columns = [
    "column",
    "missing_count"
]

missing["missing_percent"] = (
    missing["missing_count"]
    / len(df)
    * 100
)

print(
    missing[
        missing["missing_count"] > 0
    ].to_string(index=False)
)

missing.to_csv(
    OUTPUT_DIR / "missing_values.csv",
    index=False
)


# ============================================================
# 8. VARIABLE STATISTICS
# ============================================================

print()
print("=" * 75)
print("DESCRIPTIVE STATISTICS")
print("=" * 75)

numeric_columns = [
    "NDVI",
    "Rainfall_mm",
    "Temperature_C",
]

descriptive_statistics = (
    df[numeric_columns]
    .describe()
    .T
    .reset_index()
)

descriptive_statistics = (
    descriptive_statistics
    .rename(columns={"index": "variable"})
)

print(
    descriptive_statistics.to_string(
        index=False
    )
)

descriptive_statistics.to_csv(
    OUTPUT_DIR / "descriptive_statistics.csv",
    index=False
)


# ============================================================
# 9. OUTLIER ANALYSIS
# ============================================================

print()
print("=" * 75)
print("OUTLIER ANALYSIS")
print("=" * 75)

outlier_rows = []

for column in numeric_columns:

    series = df[column].dropna()

    if len(series) == 0:
        continue

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    count = (
        (series < lower_bound)
        | (series > upper_bound)
    ).sum()

    outlier_rows.append({
        "variable": column,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": int(count),
        "outlier_percent": (
            count / len(series) * 100
        ),
    })

outlier_summary = pd.DataFrame(
    outlier_rows
)

print(
    outlier_summary.to_string(
        index=False
    )
)

outlier_summary.to_csv(
    OUTPUT_DIR / "outlier_summary.csv",
    index=False
)


# ============================================================
# 10. CORRELATION MATRIX
# ============================================================

print()
print("=" * 75)
print("CORRELATION MATRIX")
print("=" * 75)

correlation_columns = [
    "NDVI",
    "Rainfall_mm",
    "Temperature_C",
]

correlation_matrix = (
    df[correlation_columns]
    .corr()
)

print(
    correlation_matrix.to_string()
)

correlation_matrix.to_csv(
    OUTPUT_DIR / "correlation_matrix.csv"
)


# ============================================================
# 11. YEARLY SUMMARY
# ============================================================

print()
print("=" * 75)
print("YEARLY SUMMARY")
print("=" * 75)

yearly_summary = (
    df.groupby("Year")
    .agg(
        records=("point_id", "size"),
        points=("point_id", "nunique"),
        ndvi_available=("NDVI", "count"),
        rainfall_mean=("Rainfall_mm", "mean"),
        rainfall_median=("Rainfall_mm", "median"),
        temperature_mean=("Temperature_C", "mean"),
        temperature_median=("Temperature_C", "median"),
        ndvi_mean=("NDVI", "mean"),
        ndvi_median=("NDVI", "median"),
    )
    .reset_index()
)

yearly_summary["ndvi_coverage_percent"] = (
    yearly_summary["ndvi_available"]
    / yearly_summary["records"]
    * 100
)

print(
    yearly_summary.to_string(
        index=False
    )
)

yearly_summary.to_csv(
    OUTPUT_DIR / "yearly_summary.csv",
    index=False
)


# ============================================================
# 12. DISTRICT SUMMARY
# ============================================================

print()
print("=" * 75)
print("DISTRICT SUMMARY")
print("=" * 75)

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
    .reset_index()
)

district_summary["ndvi_coverage_percent"] = (
    district_summary["ndvi_available"]
    / district_summary["records"]
    * 100
)

district_summary = (
    district_summary
    .sort_values("ndvi_mean", ascending=False)
)

print(
    district_summary.to_string(
        index=False
    )
)

district_summary.to_csv(
    OUTPUT_DIR / "district_summary.csv",
    index=False
)


# ============================================================
# 13. POINT SUMMARY
# ============================================================

point_summary = (
    df.groupby("point_id")
    .agg(
        district=("district", "first"),
        records=("Week_Start", "size"),
        ndvi_available=("NDVI", "count"),
        ndvi_mean=("NDVI", "mean"),
        rainfall_mean=("Rainfall_mm", "mean"),
        temperature_mean=("Temperature_C", "mean"),
    )
    .reset_index()
)

point_summary["ndvi_coverage_percent"] = (
    point_summary["ndvi_available"]
    / point_summary["records"]
    * 100
)

point_summary.to_csv(
    OUTPUT_DIR / "point_summary.csv",
    index=False
)


# ============================================================
# 14. WEEKLY NDVI COVERAGE
# ============================================================

print()
print("=" * 75)
print("WEEKLY NDVI COVERAGE")
print("=" * 75)

weekly_ndvi_coverage = (
    df.groupby(
        [
            "Year",
            "Week_Number",
            "Week_Start"
        ]
    )
    .agg(
        total_points=("point_id", "size"),
        ndvi_available=("NDVI", "count"),
    )
    .reset_index()
)

weekly_ndvi_coverage["ndvi_coverage_percent"] = (
    weekly_ndvi_coverage["ndvi_available"]
    / weekly_ndvi_coverage["total_points"]
    * 100
)

weekly_ndvi_coverage["ndvi_complete_week"] = (
    weekly_ndvi_coverage["ndvi_available"]
    == 167
)

print()
print(
    "Year-week combinations:",
    len(weekly_ndvi_coverage)
)

print(
    "Complete NDVI weeks:",
    weekly_ndvi_coverage[
        "ndvi_complete_week"
    ].sum()
)

print(
    "Empty NDVI weeks:",
    (
        weekly_ndvi_coverage[
            "ndvi_available"
        ]
        == 0
    ).sum()
)

print()
print(
    weekly_ndvi_coverage.head(20)
    .to_string(index=False)
)

weekly_ndvi_coverage.to_csv(
    OUTPUT_DIR / "weekly_ndvi_coverage.csv",
    index=False
)


# ============================================================
# 15. CRITICAL:
# NEXT-WEEK NDVI TARGET AVAILABILITY
# ============================================================

print()
print("=" * 75)
print("NEXT-WEEK NDVI TARGET AVAILABILITY")
print("=" * 75)


# We need the actual following Week_Start for the same
# point_id. We never fabricate or interpolate NDVI.

target_df = df[
    [
        "point_id",
        "Week_Start",
        "NDVI",
    ]
].copy()

target_df = target_df.rename(
    columns={
        "NDVI": "NDVI_current"
    }
)

target_df["Target_Week_Start"] = (
    target_df["Week_Start"]
    + pd.Timedelta(days=7)
)


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


target_df = target_df.merge(
    target_lookup,
    on=[
        "point_id",
        "Target_Week_Start",
    ],
    how="left",
)


target_df["usable_next_week_target"] = (
    target_df["NDVI_current"].notna()
    & target_df["NDVI_next_week"].notna()
)


target_total = len(target_df)

current_ndvi_available = (
    target_df["NDVI_current"].notna().sum()
)

next_ndvi_available = (
    target_df["NDVI_next_week"].notna().sum()
)

usable_target_count = (
    target_df[
        "usable_next_week_target"
    ].sum()
)

print()
print(
    f"Total point-weeks: "
    f"{target_total:,}"
)

print(
    f"Current NDVI available: "
    f"{current_ndvi_available:,}"
)

print(
    f"Next-week NDVI available: "
    f"{next_ndvi_available:,}"
)

print(
    f"Usable next-week target rows: "
    f"{usable_target_count:,}"
)

print(
    f"Usable target percentage: "
    f"{usable_target_count / target_total * 100:.2f}%"
)


# ============================================================
# 16. TARGET AVAILABILITY BY YEAR
# ============================================================

target_by_year = (
    target_df.assign(
        Year=target_df["Week_Start"].dt.year
    )
    .groupby("Year")
    .agg(
        total_rows=("point_id", "size"),
        current_ndvi_available=(
            "NDVI_current",
            "count"
        ),
        next_week_ndvi_available=(
            "NDVI_next_week",
            "count"
        ),
        usable_target_rows=(
            "usable_next_week_target",
            "sum"
        ),
    )
    .reset_index()
)

target_by_year["usable_target_percent"] = (
    target_by_year["usable_target_rows"]
    / target_by_year["total_rows"]
    * 100
)

print()
print(
    target_by_year.to_string(
        index=False
    )
)

target_by_year.to_csv(
    OUTPUT_DIR / "target_availability_by_year.csv",
    index=False
)


# ============================================================
# 17. WEEKLY TREND DATA
# ============================================================

weekly_trends = (
    df.groupby("Week_Start")
    .agg(
        NDVI_mean=("NDVI", "mean"),
        NDVI_median=("NDVI", "median"),
        Rainfall_mean=("Rainfall_mm", "mean"),
        Rainfall_median=("Rainfall_mm", "median"),
        Temperature_mean=("Temperature_C", "mean"),
        Temperature_median=("Temperature_C", "median"),
        NDVI_available=("NDVI", "count"),
    )
    .reset_index()
)

weekly_trends.to_csv(
    OUTPUT_DIR / "weekly_trends.csv",
    index=False
)


# ============================================================
# 18. PLOTS
# ============================================================

print()
print("=" * 75)
print("GENERATING PLOTS")
print("=" * 75)


# ------------------------------------------------------------
# NDVI DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

df["NDVI"].dropna().plot(
    kind="hist",
    bins=50
)

plt.xlabel("NDVI")
plt.ylabel("Frequency")
plt.title("NDVI Distribution - 2017–2025")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "ndvi_distribution.png",
    dpi=200
)

plt.close()


# ------------------------------------------------------------
# RAINFALL DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

df["Rainfall_mm"].plot(
    kind="hist",
    bins=50
)

plt.xlabel("Weekly Rainfall (mm)")
plt.ylabel("Frequency")
plt.title("Weekly Rainfall Distribution - 2017–2025")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "rainfall_distribution.png",
    dpi=200
)

plt.close()


# ------------------------------------------------------------
# TEMPERATURE DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

df["Temperature_C"].plot(
    kind="hist",
    bins=50
)

plt.xlabel("Weekly Temperature (°C)")
plt.ylabel("Frequency")
plt.title("Weekly Temperature Distribution - 2017–2025")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "temperature_distribution.png",
    dpi=200
)

plt.close()


# ------------------------------------------------------------
# NDVI TEMPORAL TREND
# ------------------------------------------------------------

plt.figure(figsize=(14, 6))

plt.plot(
    weekly_trends["Week_Start"],
    weekly_trends["NDVI_mean"]
)

plt.xlabel("Week")
plt.ylabel("Mean NDVI")
plt.title("Weekly Mean NDVI Trend - 2017–2025")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "ndvi_temporal_trend.png",
    dpi=200
)

plt.close()


# ------------------------------------------------------------
# RAINFALL TEMPORAL TREND
# ------------------------------------------------------------

plt.figure(figsize=(14, 6))

plt.plot(
    weekly_trends["Week_Start"],
    weekly_trends["Rainfall_mean"]
)

plt.xlabel("Week")
plt.ylabel("Mean Weekly Rainfall (mm)")
plt.title("Weekly Mean Rainfall Trend - 2017–2025")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "rainfall_temporal_trend.png",
    dpi=200
)

plt.close()


# ------------------------------------------------------------
# TEMPERATURE TEMPORAL TREND
# ------------------------------------------------------------

plt.figure(figsize=(14, 6))

plt.plot(
    weekly_trends["Week_Start"],
    weekly_trends["Temperature_mean"]
)

plt.xlabel("Week")
plt.ylabel("Mean Temperature (°C)")
plt.title("Weekly Mean Temperature Trend - 2017–2025")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "temperature_temporal_trend.png",
    dpi=200
)

plt.close()


# ============================================================
# 19. REPORT
# ============================================================

report_lines = []

report_lines.append(
    "AGRIVISION AI - EDA REPORT"
)

report_lines.append(
    "Combined Master Dataset 2017–2025"
)

report_lines.append("")

report_lines.append(
    f"Total rows: {len(df):,}"
)

report_lines.append(
    f"Total columns: {len(df.columns)}"
)

report_lines.append(
    f"Unique points: {df['point_id'].nunique()}"
)

report_lines.append(
    f"Unique weeks: {df['Week_Start'].nunique()}"
)

report_lines.append(
    f"Years: {sorted(df['Year'].dropna().astype(int).unique())}"
)

report_lines.append("")

report_lines.append(
    f"NDVI observations: {df['NDVI'].notna().sum():,}"
)

report_lines.append(
    f"NDVI coverage: "
    f"{df['NDVI'].notna().mean() * 100:.2f}%"
)

report_lines.append(
    f"Rainfall observations: "
    f"{df['Rainfall_mm'].notna().sum():,}"
)

report_lines.append(
    f"Temperature observations: "
    f"{df['Temperature_C'].notna().sum():,}"
)

report_lines.append("")

report_lines.append(
    f"Usable next-week NDVI target rows: "
    f"{usable_target_count:,}"
)

report_lines.append(
    f"Usable next-week target coverage: "
    f"{usable_target_count / target_total * 100:.2f}%"
)

report_lines.append("")

report_lines.append(
    f"NDVI range: "
    f"{df['NDVI'].min()} to {df['NDVI'].max()}"
)

report_lines.append(
    f"Rainfall range: "
    f"{df['Rainfall_mm'].min():.4f} to "
    f"{df['Rainfall_mm'].max():.4f} mm"
)

report_lines.append(
    f"Temperature range: "
    f"{df['Temperature_C'].min():.4f} to "
    f"{df['Temperature_C'].max():.4f} °C"
)

report_lines.append("")

report_lines.append(
    "Next step: feature engineering and target construction "
    "based on actual next-week NDVI observations."
)


with open(
    OUTPUT_DIR / "eda_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(report_lines)
    )


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print()
print("=" * 75)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 75)

print()
print(
    f"Rows analyzed: "
    f"{len(df):,}"
)

print(
    f"NDVI observations: "
    f"{df['NDVI'].notna().sum():,}"
)

print(
    f"NDVI coverage: "
    f"{df['NDVI'].notna().mean() * 100:.2f}%"
)

print(
    f"Usable next-week target rows: "
    f"{usable_target_count:,}"
)

print(
    f"Target coverage: "
    f"{usable_target_count / target_total * 100:.2f}%"
)

print()
print(
    "EDA outputs saved to:"
)

print(
    OUTPUT_DIR
)

print()
print("=" * 75)
print("DONE")
print("=" * 75)