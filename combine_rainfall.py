from pathlib import Path
import pandas as pd


# ============================================================
# AGRIVISION - RAINFALL MASTER DATASET BUILDER
# ============================================================

print("=" * 60)
print("AGRIVISION - RAINFALL MASTER DATASET")
print("=" * 60)


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------
# Automatically uses the folder where this Python script exists.
# Therefore it works on both Windows and macOS.

PROJECT_DIR = Path(__file__).resolve().parent

RAINFALL_DIR = PROJECT_DIR / "dataset" / "Rainfall"

OUTPUT_FILE = (
    PROJECT_DIR
    / "dataset"
    / "AgriVision_Maharashtra_Rainfall_Master_2021_2025.csv"
)


# ------------------------------------------------------------
# 2. EXPECTED FILES
# ------------------------------------------------------------
EXPECTED_FILES = [
    "AgriVision_Maharashtra_Rainfall_2021_W1_W26.csv",
    "AgriVision_Maharashtra_Rainfall_2021_W27_W52.csv",

    "AgriVision_Maharashtra_Rainfall_2022_W1_W26.csv",
    "AgriVision_Maharashtra_Rainfall_2022_W27_W52.csv",

    "AgriVision_Maharashtra_Rainfall_2023_W1_W26.csv",
    "AgriVision_Maharashtra_Rainfall_2023_W27_W52.csv",

    "AgriVision_Maharashtra_Rainfall_2024_W1_W26.csv",
    "AgriVision_Maharashtra_Rainfall_2024_W27_W52.csv",

    "AgriVision_Maharashtra_Rainfall_2025_W1_W26.csv",
    "AgriVision_Maharashtra_Rainfall_2025_W27_W52.csv",
]


# ------------------------------------------------------------
# 3. DISPLAY PATH
# ------------------------------------------------------------
print()
print("Project folder:")
print(PROJECT_DIR)

print()
print("Rainfall folder:")
print(RAINFALL_DIR)


# ------------------------------------------------------------
# 4. CHECK RAINFALL FOLDER
# ------------------------------------------------------------
if not RAINFALL_DIR.exists():
    raise FileNotFoundError(
        f"\nRainfall folder does not exist:\n{RAINFALL_DIR}\n\n"
        "Make sure your folder structure is:\n"
        "AgriVision/\n"
        "  combine_rainfall.py\n"
        "  dataset/\n"
        "    Rainfall/\n"
        "      rainfall CSV files"
    )


# ------------------------------------------------------------
# 5. FIND FILES
# ------------------------------------------------------------
files = sorted(RAINFALL_DIR.glob("AgriVision_Maharashtra_Rainfall_*.csv"))

print()
print(f"Files found: {len(files)}")

for file in files:
    print("  -", file.name)


# ------------------------------------------------------------
# 6. VERIFY EXACTLY 10 FILES
# ------------------------------------------------------------
if len(files) != 10:
    print()
    print("Expected these 10 files:")

    for name in EXPECTED_FILES:
        print("  -", name)

    raise ValueError(
        f"\nExpected 10 rainfall batch files, but found {len(files)}.\n"
        f"Check this folder:\n{RAINFALL_DIR}"
    )


# ------------------------------------------------------------
# 7. CHECK EXPECTED FILE NAMES
# ------------------------------------------------------------
actual_names = {file.name for file in files}
expected_names = set(EXPECTED_FILES)

missing_files = expected_names - actual_names
unexpected_files = actual_names - expected_names

if missing_files:
    print()
    print("MISSING FILES:")
    for name in sorted(missing_files):
        print("  -", name)

if unexpected_files:
    print()
    print("UNEXPECTED FILES:")
    for name in sorted(unexpected_files):
        print("  -", name)

if missing_files:
    raise ValueError("One or more expected rainfall batch files are missing.")


# ------------------------------------------------------------
# 8. READ ALL FILES
# ------------------------------------------------------------
dataframes = []

reference_columns = None

for file in files:

    print()
    print(f"Reading: {file.name}")

    df = pd.read_csv(file)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # Make sure every file has the same columns
    if reference_columns is None:
        reference_columns = list(df.columns)

    else:
        if list(df.columns) != reference_columns:
            raise ValueError(
                f"\nColumn mismatch detected in:\n{file.name}\n\n"
                f"Expected:\n{reference_columns}\n\n"
                f"Found:\n{list(df.columns)}"
            )

    dataframes.append(df)


# ------------------------------------------------------------
# 9. COMBINE
# ------------------------------------------------------------
print()
print("=" * 60)
print("COMBINING FILES")
print("=" * 60)

master = pd.concat(dataframes, ignore_index=True)

print()
print(f"Combined rows: {len(master):,}")
print(f"Combined columns: {len(master.columns)}")


# ------------------------------------------------------------
# 10. REQUIRED COLUMNS
# ------------------------------------------------------------
required_columns = [
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
    "Rainfall_mm",
    "Rainfall_Valid",
    "CHIRPS_Days",
    "rainfall_source",
    "study_area",
    "dataset",
]

missing_columns = [
    col for col in required_columns
    if col not in master.columns
]

if missing_columns:
    raise ValueError(
        f"\nMissing required columns:\n{missing_columns}"
    )


# ------------------------------------------------------------
# 11. DATE CONVERSION
# ------------------------------------------------------------
master["Week_Start"] = pd.to_datetime(
    master["Week_Start"],
    errors="coerce"
)

master["Week_End"] = pd.to_datetime(
    master["Week_End"],
    errors="coerce"
)


# ------------------------------------------------------------
# 12. NUMERIC CONVERSION
# ------------------------------------------------------------
numeric_columns = [
    "Week_Number",
    "Year",
    "latitude",
    "longitude",
    "Rainfall_mm",
    "Rainfall_Valid",
    "CHIRPS_Days",
]

for col in numeric_columns:
    master[col] = pd.to_numeric(
        master[col],
        errors="coerce"
    )


# ------------------------------------------------------------
# 13. SORT
# ------------------------------------------------------------
master = master.sort_values(
    by=["Week_Start", "point_id"]
).reset_index(drop=True)


# ------------------------------------------------------------
# 14. VALIDATION
# ------------------------------------------------------------
print()
print("=" * 60)
print("VALIDATION")
print("=" * 60)


# Expected total
expected_rows = 167 * 260

print()
print(f"Expected theoretical records: {expected_rows:,}")
print(f"Actual records:              {len(master):,}")

if len(master) == expected_rows:
    print("PASS - Total record count")
else:
    print(
        "INFO - Total is below theoretical maximum "
        "(this is not an error if source data has gaps)."
    )


# Unique points
unique_points = master["point_id"].nunique()

print()
print(f"Unique point IDs: {unique_points}")

if unique_points == 167:
    print("PASS - 167 sampling points")
else:
    print("WARNING - Unexpected number of sampling points")


# Unique weeks
unique_weeks = master["Week_Start"].nunique()

print()
print(f"Unique weeks: {unique_weeks}")

if unique_weeks == 260:
    print("PASS - 260 weeks")
else:
    print("WARNING - Unexpected number of weeks")


# Years
years = sorted(master["Year"].dropna().astype(int).unique())

print()
print(f"Years present: {years}")

if years == [2021, 2022, 2023, 2024, 2025]:
    print("PASS - Correct year range")
else:
    print("WARNING - Unexpected years")


# Duplicate point + week
duplicate_point_week = master.duplicated(
    subset=["point_id", "Week_Start"]
).sum()

print()
print(f"Duplicate point + week records: {duplicate_point_week}")

if duplicate_point_week == 0:
    print("PASS - No duplicate point-week records")
else:
    print("FAIL - Duplicate point-week records found")


# Full duplicate rows
full_duplicates = master.duplicated().sum()

print()
print(f"Full duplicate rows: {full_duplicates}")

if full_duplicates == 0:
    print("PASS - No full duplicate rows")
else:
    print("FAIL - Full duplicates found")


# Missing values
missing_total = master[required_columns].isna().sum().sum()

print()
print(f"Missing values in required columns: {missing_total}")

if missing_total == 0:
    print("PASS - No missing required values")
else:
    print("FAIL - Missing values found")

    print()
    print("Missing values by column:")

    missing_by_column = master[required_columns].isna().sum()

    for col, count in missing_by_column.items():
        if count > 0:
            print(f"  {col}: {count}")


# Rainfall negative check
negative_rainfall = (
    master["Rainfall_mm"] < 0
).sum()

print()
print(f"Negative rainfall values: {negative_rainfall}")

if negative_rainfall == 0:
    print("PASS - No negative rainfall")
else:
    print("FAIL - Negative rainfall detected")


# Rainfall valid check
invalid_rainfall = (
    master["Rainfall_Valid"] != 1
).sum()

print()
print(f"Rainfall_Valid != 1: {invalid_rainfall}")

if invalid_rainfall == 0:
    print("PASS - All rainfall records valid")
else:
    print("WARNING - Invalid rainfall records found")


# CHIRPS days
wrong_chirps_days = (
    master["CHIRPS_Days"] != 7
).sum()

print()
print(f"CHIRPS_Days != 7: {wrong_chirps_days}")

if wrong_chirps_days == 0:
    print("PASS - Every record has 7 CHIRPS days")
else:
    print("FAIL - Incorrect CHIRPS day count found")


# ------------------------------------------------------------
# 15. YEAR-WISE RECORD COUNT
# ------------------------------------------------------------
print()
print("=" * 60)
print("YEAR-WISE RECORD COUNT")
print("=" * 60)

year_counts = (
    master.groupby("Year")
    .size()
    .sort_index()
)

for year, count in year_counts.items():

    expected_year = 167 * 52

    status = "PASS" if count == expected_year else "WARNING"

    print(
        f"{int(year)}: {count:,} records "
        f"(expected {expected_year:,}) - {status}"
    )


# ------------------------------------------------------------
# 16. YEAR + WEEK VALIDATION
# ------------------------------------------------------------
print()
print("=" * 60)
print("WEEKLY COVERAGE")
print("=" * 60)

weekly_counts = (
    master.groupby(
        ["Year", "Week_Number"]
    )
    .size()
)

bad_weeks = weekly_counts[
    weekly_counts != 167
]

if len(bad_weeks) == 0:

    print("PASS - Every year/week contains all 167 points")

else:

    print(
        f"WARNING - {len(bad_weeks)} year/week combinations "
        "do not contain all 167 points:"
    )

    for index, count in bad_weeks.items():

        year, week = index

        print(
            f"  {int(year)} W{int(week):02d}: "
            f"{count} records"
        )


# ------------------------------------------------------------
# 17. COORDINATE CONSISTENCY
# ------------------------------------------------------------
print()
print("=" * 60)
print("POINT CONSISTENCY")
print("=" * 60)

point_consistency = (
    master.groupby("point_id")
    .agg(
        latitude_unique=("latitude", "nunique"),
        longitude_unique=("longitude", "nunique"),
        district_unique=("district", "nunique"),
    )
)

bad_points = point_consistency[
    (point_consistency["latitude_unique"] > 1)
    | (point_consistency["longitude_unique"] > 1)
    | (point_consistency["district_unique"] > 1)
]

if len(bad_points) == 0:

    print(
        "PASS - Coordinates and district are consistent "
        "for every point"
    )

else:

    print(
        f"FAIL - {len(bad_points)} points have inconsistent metadata"
    )


# ------------------------------------------------------------
# 18. SOURCE CONSISTENCY
# ------------------------------------------------------------
print()
print("Rainfall source values:")
print(master["rainfall_source"].value_counts().to_string())

print()
print("Study area values:")
print(master["study_area"].value_counts().to_string())

print()
print("Dataset values:")
print(master["dataset"].value_counts().to_string())


# ------------------------------------------------------------
# 19. RAINFALL STATISTICS
# ------------------------------------------------------------
print()
print("=" * 60)
print("RAINFALL STATISTICS")
print("=" * 60)

print()
print(
    f"Minimum weekly rainfall: "
    f"{master['Rainfall_mm'].min():.4f} mm"
)

print(
    f"Maximum weekly rainfall: "
    f"{master['Rainfall_mm'].max():.4f} mm"
)

print(
    f"Mean weekly rainfall: "
    f"{master['Rainfall_mm'].mean():.4f} mm"
)

print(
    f"Median weekly rainfall: "
    f"{master['Rainfall_mm'].median():.4f} mm"
)


# ------------------------------------------------------------
# 20. DATE RANGE
# ------------------------------------------------------------
print()
print("=" * 60)
print("DATE RANGE")
print("=" * 60)

print(
    "Week_Start:",
    master["Week_Start"].min().date(),
    "to",
    master["Week_Start"].max().date()
)

print(
    "Week_End:",
    master["Week_End"].min().date(),
    "to",
    master["Week_End"].max().date()
)


# ------------------------------------------------------------
# 21. FINAL COLUMN ORDER
# ------------------------------------------------------------
master = master[required_columns]


# ------------------------------------------------------------
# 22. CREATE OUTPUT DIRECTORY
# ------------------------------------------------------------
OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# 23. SAVE MASTER DATASET
# ------------------------------------------------------------
master.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# 24. FINAL SUMMARY
# ------------------------------------------------------------
print()
print("=" * 60)
print("MASTER DATASET CREATED SUCCESSFULLY")
print("=" * 60)

print()
print("Output file:")
print(OUTPUT_FILE)

print()
print(f"Total records: {len(master):,}")
print(f"Total columns: {len(master.columns)}")
print(f"Unique points: {master['point_id'].nunique()}")
print(f"Unique weeks: {master['Week_Start'].nunique()}")

print()
print("Final columns:")
for i, column in enumerate(master.columns, start=1):
    print(f"{i:2}. {column}")

print()
print("=" * 60)
print("DONE")
print("=" * 60)