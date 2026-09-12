from pathlib import Path
import pandas as pd


# ============================================================
# AGRIVISION - TEMPERATURE MASTER DATASET BUILDER
# ============================================================

print("=" * 60)
print("AGRIVISION - TEMPERATURE MASTER DATASET")
print("=" * 60)


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

# Automatically uses the folder containing this script.
# Works on both Windows and macOS.

PROJECT_DIR = Path(__file__).resolve().parent

TEMPERATURE_DIR = PROJECT_DIR / "dataset" / "Temperature"

OUTPUT_FILE = (
    PROJECT_DIR
    / "dataset"
    / "AgriVision_Maharashtra_Temperature_Master_2021_2025.csv"
)


# ------------------------------------------------------------
# 2. EXPECTED FILES
# ------------------------------------------------------------

EXPECTED_FILES = [
    "AgriVision_Maharashtra_Temperature_2021_W01_W26.csv",
    "AgriVision_Maharashtra_Temperature_2021_W27_W52.csv",

    "AgriVision_Maharashtra_Temperature_2022_W01_W26.csv",
    "AgriVision_Maharashtra_Temperature_2022_W27_W52.csv",

    "AgriVision_Maharashtra_Temperature_2023_W01_W26.csv",
    "AgriVision_Maharashtra_Temperature_2023_W27_W52.csv",

    "AgriVision_Maharashtra_Temperature_2024_W01_W26.csv",
    "AgriVision_Maharashtra_Temperature_2024_W27_W52.csv",

    "AgriVision_Maharashtra_Temperature_2025_W01_W26.csv",
    "AgriVision_Maharashtra_Temperature_2025_W27_W52.csv",
]


# ------------------------------------------------------------
# 3. DISPLAY PATHS
# ------------------------------------------------------------

print()
print("Project folder:")
print(PROJECT_DIR)

print()
print("Temperature folder:")
print(TEMPERATURE_DIR)


# ------------------------------------------------------------
# 4. CHECK FOLDER
# ------------------------------------------------------------

if not TEMPERATURE_DIR.exists():
    raise FileNotFoundError(
        f"\nTemperature folder does not exist:\n{TEMPERATURE_DIR}\n\n"
        "Expected structure:\n"
        "AgriVision/\n"
        "  combine_temperature.py\n"
        "  dataset/\n"
        "    Temperature/\n"
        "      temperature CSV files"
    )


# ------------------------------------------------------------
# 5. FIND CSV FILES
# ------------------------------------------------------------

files = sorted(
    TEMPERATURE_DIR.glob(
        "AgriVision_Maharashtra_Temperature_*.csv"
    )
)

print()
print(f"Files found: {len(files)}")

for file in files:
    print("  -", file.name)


# ------------------------------------------------------------
# 6. VERIFY FILE COUNT
# ------------------------------------------------------------

if len(files) != 10:
    print()
    print("Expected these 10 files:")

    for name in EXPECTED_FILES:
        print("  -", name)

    raise ValueError(
        f"\nExpected 10 temperature batch files, "
        f"but found {len(files)}.\n"
        f"Check this folder:\n{TEMPERATURE_DIR}"
    )


# ------------------------------------------------------------
# 7. VERIFY FILENAMES
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

    raise ValueError(
        "\nOne or more expected temperature batch files are missing."
    )

if unexpected_files:
    print()
    print("WARNING - Unexpected files found:")

    for name in sorted(unexpected_files):
        print("  -", name)


# ------------------------------------------------------------
# 8. READ ALL FILES
# ------------------------------------------------------------

dataframes = []

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
    "Temperature_C",
    "Temperature_Valid",
    "ERA5_Days",
    "temperature_source",
    "study_area",
    "dataset",
]

reference_columns = None

for file in files:

    print()
    print(f"Reading: {file.name}")

    df = pd.read_csv(file)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # --------------------------------------------------------
    # Remove automatic GEE export columns if present
    # --------------------------------------------------------

    gee_columns = [
        "system:index",
        ".geo",
    ]

    for col in gee_columns:
        if col in df.columns:
            df = df.drop(columns=[col])

    print(f"Columns after GEE cleanup: {len(df.columns)}")

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"\nMissing required columns in:\n{file.name}\n"
            f"{missing_columns}"
        )

    # Keep only the intended schema
    df = df[required_columns]

    # Check identical schema
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

master = pd.concat(
    dataframes,
    ignore_index=True
)

print()
print(f"Combined rows: {len(master):,}")
print(f"Combined columns: {len(master.columns)}")


# ------------------------------------------------------------
# 10. DATE CONVERSION
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
# 11. NUMERIC CONVERSION
# ------------------------------------------------------------

numeric_columns = [
    "Week_Number",
    "Year",
    "latitude",
    "longitude",
    "Temperature_C",
    "Temperature_Valid",
    "ERA5_Days",
]

for col in numeric_columns:

    master[col] = pd.to_numeric(
        master[col],
        errors="coerce"
    )


# ------------------------------------------------------------
# 12. SORT
# ------------------------------------------------------------

master = master.sort_values(
    by=["Week_Start", "point_id"]
).reset_index(drop=True)


# ------------------------------------------------------------
# 13. VALIDATION
# ------------------------------------------------------------

print()
print("=" * 60)
print("VALIDATION")
print("=" * 60)


# Expected theoretical record count
expected_rows = 167 * 260

print()
print(f"Expected theoretical records: {expected_rows:,}")
print(f"Actual records:              {len(master):,}")

if len(master) == expected_rows:
    print("PASS - Total record count")
else:
    print("FAIL - Unexpected total record count")


# Unique points
unique_points = master["point_id"].nunique()

print()
print(f"Unique point IDs: {unique_points}")

if unique_points == 167:
    print("PASS - 167 sampling points")
else:
    print("FAIL - Unexpected number of sampling points")


# Unique weeks
unique_weeks = master["Week_Start"].nunique()

print()
print(f"Unique weeks: {unique_weeks}")

if unique_weeks == 260:
    print("PASS - 260 weeks")
else:
    print("FAIL - Unexpected number of weeks")


# Years
years = sorted(
    master["Year"]
    .dropna()
    .astype(int)
    .unique()
)

print()
print(f"Years present: {years}")

if years == [2021, 2022, 2023, 2024, 2025]:
    print("PASS - Correct year range")
else:
    print("FAIL - Unexpected years")


# Duplicate point-week
duplicate_point_week = master.duplicated(
    subset=["point_id", "Week_Start"]
).sum()

print()
print(
    f"Duplicate point + week records: "
    f"{duplicate_point_week}"
)

if duplicate_point_week == 0:
    print("PASS - No duplicate point-week records")
else:
    print("FAIL - Duplicate point-week records found")


# Full duplicates
full_duplicates = master.duplicated().sum()

print()
print(f"Full duplicate rows: {full_duplicates}")

if full_duplicates == 0:
    print("PASS - No full duplicate rows")
else:
    print("FAIL - Full duplicate rows found")


# Missing values
missing_total = (
    master[required_columns]
    .isna()
    .sum()
    .sum()
)

print()
print(
    f"Missing values in required columns: "
    f"{missing_total}"
)

if missing_total == 0:
    print("PASS - No missing required values")
else:
    print("FAIL - Missing required values found")

    print()
    print("Missing values by column:")

    missing_by_column = (
        master[required_columns]
        .isna()
        .sum()
    )

    for col, count in missing_by_column.items():

        if count > 0:
            print(f"  {col}: {count}")


# Temperature valid flag
invalid_temperature = (
    master["Temperature_Valid"] != 1
).sum()

print()
print(
    f"Temperature_Valid != 1: "
    f"{invalid_temperature}"
)

if invalid_temperature == 0:
    print("PASS - All temperature records valid")
else:
    print("FAIL - Invalid temperature records found")


# ERA5 days
wrong_days = (
    master["ERA5_Days"] != 7
).sum()

print()
print(
    f"ERA5_Days != 7: "
    f"{wrong_days}"
)

if wrong_days == 0:
    print("PASS - Every record contains 7 ERA5 days")
else:
    print("FAIL - Incorrect ERA5 day count found")


# Negative temperatures
negative_temperature = (
    master["Temperature_C"] < 0
).sum()

print()
print(
    f"Negative temperature records: "
    f"{negative_temperature}"
)

if negative_temperature == 0:
    print("PASS - No negative temperatures")
else:
    print(
        "WARNING - Negative temperatures found "
        "(inspect before final use)"
    )


# ------------------------------------------------------------
# 14. YEAR-WISE RECORD COUNTS
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

    status = (
        "PASS"
        if count == expected_year
        else "FAIL"
    )

    print(
        f"{int(year)}: "
        f"{count:,} records "
        f"(expected {expected_year:,}) - {status}"
    )


# ------------------------------------------------------------
# 15. WEEKLY COVERAGE
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

    print(
        "PASS - Every year/week contains "
        "all 167 points"
    )

else:

    print(
        f"FAIL - {len(bad_weeks)} "
        "year/week combinations are incomplete:"
    )

    for index, count in bad_weeks.items():

        year, week = index

        print(
            f"  {int(year)} "
            f"W{int(week):02d}: "
            f"{count} records"
        )


# ------------------------------------------------------------
# 16. WEEK DATE CONSISTENCY
# ------------------------------------------------------------

print()
print("=" * 60)
print("WEEK DATE CONSISTENCY")
print("=" * 60)

expected_week_end = (
    master["Week_Start"]
    + pd.Timedelta(days=7)
)

bad_week_dates = (
    master["Week_End"] != expected_week_end
).sum()

print(
    f"Incorrect Week_End values: "
    f"{bad_week_dates}"
)

if bad_week_dates == 0:
    print(
        "PASS - Every Week_End is exactly "
        "7 days after Week_Start"
    )
else:
    print("FAIL - Week date inconsistency found")


# ------------------------------------------------------------
# 17. WEEK NUMBER CONSISTENCY
# ------------------------------------------------------------

def expected_week_number(row):
    jan1 = pd.Timestamp(year=int(row["Year"]), month=1, day=1)
    delta_days = (row["Week_Start"] - jan1).days
    return (delta_days // 7) + 1


master["Expected_Week_Number"] = master.apply(
    expected_week_number,
    axis=1
)

bad_week_numbers = (
    master["Week_Number"]
    != master["Expected_Week_Number"]
).sum()

print()
print(
    f"Incorrect week numbers: "
    f"{bad_week_numbers}"
)

if bad_week_numbers == 0:
    print("PASS - Week numbering is consistent")
else:
    print("FAIL - Week numbering inconsistency found")

master = master.drop(
    columns=["Expected_Week_Number"]
)


# ------------------------------------------------------------
# 18. POINT CONSISTENCY
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
        "PASS - Coordinates and districts are "
        "consistent for every point"
    )
else:
    print(
        f"FAIL - {len(bad_points)} points "
        "have inconsistent metadata"
    )


# ------------------------------------------------------------
# 19. SOURCE CONSISTENCY
# ------------------------------------------------------------

print()
print("Temperature source values:")
print(
    master["temperature_source"]
    .value_counts()
    .to_string()
)

print()
print("Study area values:")
print(
    master["study_area"]
    .value_counts()
    .to_string()
)

print()
print("Dataset values:")
print(
    master["dataset"]
    .value_counts()
    .to_string()
)


# ------------------------------------------------------------
# 20. TEMPERATURE STATISTICS
# ------------------------------------------------------------

print()
print("=" * 60)
print("TEMPERATURE STATISTICS")
print("=" * 60)

print()
print(
    f"Minimum temperature: "
    f"{master['Temperature_C'].min():.4f} °C"
)

print(
    f"Maximum temperature: "
    f"{master['Temperature_C'].max():.4f} °C"
)

print(
    f"Mean temperature: "
    f"{master['Temperature_C'].mean():.4f} °C"
)

print(
    f"Median temperature: "
    f"{master['Temperature_C'].median():.4f} °C"
)


# ------------------------------------------------------------
# 21. DATE RANGE
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
# 22. YEAR-WISE TEMPERATURE STATISTICS
# ------------------------------------------------------------

print()
print("=" * 60)
print("YEAR-WISE TEMPERATURE STATISTICS")
print("=" * 60)

year_stats = (
    master.groupby("Year")["Temperature_C"]
    .agg(["min", "max", "mean", "median"])
)

print(year_stats.to_string())


# ------------------------------------------------------------
# 23. FINAL COLUMN ORDER
# ------------------------------------------------------------

master = master[required_columns]


# ------------------------------------------------------------
# 24. CREATE OUTPUT DIRECTORY
# ------------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# 25. SAVE MASTER DATASET
# ------------------------------------------------------------

master.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# 26. FINAL SUMMARY
# ------------------------------------------------------------

print()
print("=" * 60)
print("MASTER TEMPERATURE DATASET CREATED")
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