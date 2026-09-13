from pathlib import Path
import pandas as pd


# ============================================================
# AGRIVISION AI
# Combine GSMaP Rainfall Datasets
# 2017-2025
# ============================================================

BASE_DIR = Path(r"C:\Projects\AgriVision")

INPUT_DIR = (
    BASE_DIR
    / "dataset"
    / "Rainfall_GSMaP"
)

OUTPUT_FILE = (
    INPUT_DIR
    / "AgriVision_Maharashtra_GSMaP_Rainfall_Master_2017_2025.csv"
)


# ============================================================
# EXPECTED FILES
# ============================================================

years = range(2017, 2026)

files = [
    INPUT_DIR / f"AgriVision_GSMaP_Rainfall_{year}.csv"
    for year in years
]


# ============================================================
# VALIDATE FILES
# ============================================================

print("=" * 70)
print("AGRIVISION - GSMaP DATASET COMBINER")
print("=" * 70)

print(f"\nInput directory:")
print(INPUT_DIR)

missing_files = [
    file for file in files
    if not file.exists()
]

if missing_files:

    print("\nERROR: Missing files:")

    for file in missing_files:
        print(f"  - {file.name}")

    raise FileNotFoundError(
        "One or more yearly GSMaP files are missing."
    )


print("\nAll 9 yearly files found.")


# ============================================================
# EXPECTED COLUMNS
# ============================================================

required_columns = [
    "point_id",
    "Year",
    "Week_Number",
    "Week_Start",
    "Week_End",
    "Rainfall_GSMaP",
    "GSMaP_Image_Count",
]


# ============================================================
# READ EACH YEAR
# ============================================================

dataframes = []

for year, file in zip(years, files):

    print()
    print("-" * 70)
    print(f"Reading {year}...")
    print(f"File: {file.name}")

    df = pd.read_csv(file)

    print(f"Rows: {len(df):,}")

    # --------------------------------------------------------
    # Check columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"{file.name} is missing columns: "
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Check year consistency
    # --------------------------------------------------------

    actual_years = (
        pd.to_numeric(
            df["Year"],
            errors="coerce"
        )
        .dropna()
        .unique()
    )

    if len(actual_years) != 1 or actual_years[0] != year:

        raise ValueError(
            f"{file.name} contains unexpected years: "
            f"{actual_years}"
        )

    # --------------------------------------------------------
    # Basic row-count check
    # --------------------------------------------------------

    expected_rows = 167 * 52

    if len(df) != expected_rows:

        raise ValueError(
            f"{file.name} has {len(df):,} rows. "
            f"Expected {expected_rows:,}."
        )

    # --------------------------------------------------------
    # Add to collection
    # --------------------------------------------------------

    dataframes.append(
        df[required_columns].copy()
    )


# ============================================================
# COMBINE
# ============================================================

print()
print("=" * 70)
print("COMBINING DATASETS")
print("=" * 70)

combined = pd.concat(
    dataframes,
    ignore_index=True
)

print(
    f"\nCombined rows: {len(combined):,}"
)


# ============================================================
# STANDARDIZE TYPES
# ============================================================

combined["point_id"] = (
    combined["point_id"]
    .astype(str)
    .str.strip()
)

combined["Year"] = pd.to_numeric(
    combined["Year"],
    errors="coerce"
).astype("Int64")

combined["Week_Number"] = pd.to_numeric(
    combined["Week_Number"],
    errors="coerce"
).astype("Int64")

combined["Rainfall_GSMaP"] = pd.to_numeric(
    combined["Rainfall_GSMaP"],
    errors="coerce"
)

combined["GSMaP_Image_Count"] = pd.to_numeric(
    combined["GSMaP_Image_Count"],
    errors="coerce"
)

combined["Week_Start"] = pd.to_datetime(
    combined["Week_Start"],
    errors="coerce"
)

combined["Week_End"] = pd.to_datetime(
    combined["Week_End"],
    errors="coerce"
)


# ============================================================
# SORT
# ============================================================

combined = combined.sort_values(
    [
        "point_id",
        "Week_Start"
    ]
).reset_index(drop=True)


# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 70)
print("VALIDATION")
print("=" * 70)

expected_total = 167 * 52 * 9

print(
    f"\nExpected total rows : {expected_total:,}"
)

print(
    f"Actual total rows   : {len(combined):,}"
)


# ------------------------------------------------------------
# Points
# ------------------------------------------------------------

unique_points = combined["point_id"].nunique()

print(
    f"\nUnique points       : {unique_points}"
)


# ------------------------------------------------------------
# Weeks
# ------------------------------------------------------------

unique_weeks = combined["Week_Start"].nunique()

print(
    f"Unique weeks        : {unique_weeks}"
)


# ------------------------------------------------------------
# Years
# ------------------------------------------------------------

unique_years = combined["Year"].nunique()

print(
    f"Unique years        : {unique_years}"
)


# ------------------------------------------------------------
# Duplicate point-week records
# ------------------------------------------------------------

duplicates = combined.duplicated(
    subset=[
        "point_id",
        "Week_Start"
    ],
    keep=False
)

duplicate_count = duplicates.sum()

print(
    f"Duplicate point-weeks: {duplicate_count}"
)


# ------------------------------------------------------------
# Missing values
# ------------------------------------------------------------

print("\nMissing values:")

for column in required_columns:

    missing = combined[column].isna().sum()

    print(
        f"  {column:<22} "
        f"{missing:,}"
    )


# ------------------------------------------------------------
# GSMaP image counts
# ------------------------------------------------------------

print("\nGSMaP image-count distribution:")

print(
    combined[
        "GSMaP_Image_Count"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# YEAR-WISE VALIDATION
# ============================================================

print()
print("=" * 70)
print("YEAR-WISE VALIDATION")
print("=" * 70)

year_summary = (
    combined
    .groupby("Year")
    .agg(
        Rows=("point_id", "size"),
        Points=("point_id", "nunique"),
        Weeks=("Week_Start", "nunique"),
        Rainfall_Mean=("Rainfall_GSMaP", "mean"),
        Rainfall_Median=("Rainfall_GSMaP", "median"),
        Rainfall_Min=("Rainfall_GSMaP", "min"),
        Rainfall_Max=("Rainfall_GSMaP", "max"),
    )
    .reset_index()
)

print(
    year_summary.to_string(
        index=False
    )
)


# ============================================================
# FINAL SAFETY CHECKS
# ============================================================

if len(combined) != expected_total:

    raise RuntimeError(
        "TOTAL ROW COUNT CHECK FAILED."
    )


if unique_points != 167:

    raise RuntimeError(
        f"Expected 167 points, found "
        f"{unique_points}."
    )


if unique_weeks != 468:

    raise RuntimeError(
        f"Expected 468 weeks, found "
        f"{unique_weeks}."
    )


if duplicate_count != 0:

    raise RuntimeError(
        f"Found {duplicate_count} duplicate "
        "point-week records."
    )


# Every yearly file should contribute exactly 8,684 rows.
rows_per_year = (
    combined
    .groupby("Year")
    .size()
)

if not (rows_per_year == 8684).all():

    raise RuntimeError(
        "One or more years do not contain "
        "exactly 8,684 rows."
    )


# ============================================================
# SAVE MASTER
# ============================================================

combined.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("GSMaP MASTER DATASET CREATED SUCCESSFULLY")
print("=" * 70)

print(
    f"\nOutput file:\n{OUTPUT_FILE}"
)

print(
    f"\nRows       : {len(combined):,}"
)

print(
    f"Points     : {unique_points}"
)

print(
    f"Weeks      : {unique_weeks}"
)

print(
    f"Years      : {unique_years}"
)

print(
    f"Duplicates : {duplicate_count}"
)

print()
print("Dataset is ready for the GSMaP ML experiment.")
print("=" * 70)