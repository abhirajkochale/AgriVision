from pathlib import Path
import pandas as pd


# ============================================================
# AGRIVISION - TEMPERATURE MASTER DATASET BUILDER
# 2017–2025
# ============================================================

print("=" * 70)
print("AGRIVISION - TEMPERATURE MASTER DATASET")
print("2017–2025")
print("=" * 70)


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_DIR / "dataset"
TEMPERATURE_DIR = DATASET_DIR / "Temperature"

OUTPUT_FILE = (
    DATASET_DIR
    / "AgriVision_Maharashtra_Temperature_Master_2017_2025.csv"
)


print()
print("Project folder:")
print(PROJECT_DIR)

print()
print("Temperature folder:")
print(TEMPERATURE_DIR)


# ------------------------------------------------------------
# 2. EXPECTED INPUT FILES
# ------------------------------------------------------------

EXPECTED_FILES = [
    "AgriVision_Maharashtra_Temperature_2017_W1_W26.csv",
    "AgriVision_Maharashtra_Temperature_2017_W27_W52.csv",

    "AgriVision_Maharashtra_Temperature_2018_W1_W26.csv",
    "AgriVision_Maharashtra_Temperature_2018_W27_W52.csv",

    "AgriVision_Maharashtra_Temperature_2019_W1_W26.csv",
    "AgriVision_Maharashtra_Temperature_2019_W27_W52.csv",

    "AgriVision_Maharashtra_Temperature_2020_W1_W26.csv",
    "AgriVision_Maharashtra_Temperature_2020_W27_W52.csv",

    "AgriVision_Maharashtra_Temperature_Master_2021_2025.csv",
]


# ------------------------------------------------------------
# 3. CHECK FOLDER
# ------------------------------------------------------------

if not TEMPERATURE_DIR.exists():
    raise FileNotFoundError(
        f"\nTemperature folder does not exist:\n{TEMPERATURE_DIR}"
    )


# ------------------------------------------------------------
# 4. CHECK INPUT FILES
# ------------------------------------------------------------

print()
print("=" * 70)
print("CHECKING INPUT FILES")
print("=" * 70)

missing_files = []

for filename in EXPECTED_FILES:

    path = TEMPERATURE_DIR / filename

    if path.exists():
        print(f"PASS - {filename}")
    else:
        print(f"FAIL - {filename}")
        missing_files.append(filename)

if missing_files:

    raise FileNotFoundError(
        "\nMissing temperature files:\n"
        + "\n".join(f"  - {x}" for x in missing_files)
    )


# ------------------------------------------------------------
# 5. READ FILES
# ------------------------------------------------------------

print()
print("=" * 70)
print("READING TEMPERATURE FILES")
print("=" * 70)

dataframes = []

reference_columns = None

for filename in EXPECTED_FILES:

    path = TEMPERATURE_DIR / filename

    df = pd.read_csv(path)

    print()
    print(filename)
    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # Check schema consistency
    if reference_columns is None:

        reference_columns = list(df.columns)

    else:

        if list(df.columns) != reference_columns:

            raise ValueError(
                f"\nColumn mismatch found in {filename}\n\n"
                f"Expected:\n{reference_columns}\n\n"
                f"Found:\n{list(df.columns)}"
            )

    dataframes.append(df)


# ------------------------------------------------------------
# 6. COMBINE
# ------------------------------------------------------------

print()
print("=" * 70)
print("COMBINING TEMPERATURE DATA")
print("=" * 70)

master = pd.concat(
    dataframes,
    ignore_index=True
)

print()
print(f"Combined records: {len(master):,}")
print(f"Combined columns: {len(master.columns)}")


# ------------------------------------------------------------
# 7. REQUIRED COLUMNS
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
    "Temperature_C",
    "Temperature_Valid",
    "ERA5_Days",
    "temperature_source",
    "study_area",
    "dataset",
]

missing_columns = [
    column
    for column in required_columns
    if column not in master.columns
]

if missing_columns:

    raise ValueError(
        f"\nMissing required columns:\n{missing_columns}"
    )


# ------------------------------------------------------------
# 8. DATE CONVERSION
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
# 9. NUMERIC CONVERSION
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

for column in numeric_columns:

    master[column] = pd.to_numeric(
        master[column],
        errors="coerce"
    )


# ------------------------------------------------------------
# 10. NORMALIZE COORDINATES
# ------------------------------------------------------------
# Coordinates represent fixed sampling points.
# Six decimal places is more than sufficient for these
# fixed geographic sampling locations and prevents meaningless
# floating-point precision differences across CSV batches.

master["latitude"] = master["latitude"].round(6)
master["longitude"] = master["longitude"].round(6)


# ------------------------------------------------------------
# 11. NORMALIZE TEXT METADATA
# ------------------------------------------------------------

master["point_id"] = (
    master["point_id"]
    .astype(str)
    .str.strip()
)

master["district"] = (
    master["district"]
    .astype(str)
    .str.strip()
)

master["sampling_priority"] = (
    master["sampling_priority"]
    .astype(str)
    .str.strip()
)

master["region_type"] = (
    master["region_type"]
    .astype(str)
    .str.strip()
)

master["temperature_source"] = (
    master["temperature_source"]
    .astype(str)
    .str.strip()
)

master["study_area"] = (
    master["study_area"]
    .astype(str)
    .str.strip()
)

master["dataset"] = (
    master["dataset"]
    .astype(str)
    .str.strip()
)


# ------------------------------------------------------------
# 12. SORT
# ------------------------------------------------------------

master = master.sort_values(
    by=["Week_Start", "point_id"]
).reset_index(drop=True)


# ============================================================
# 13. BASIC VALIDATION
# ============================================================

print()
print("=" * 70)
print("BASIC VALIDATION")
print("=" * 70)


# ------------------------------------------------------------
# TOTAL RECORDS
# ------------------------------------------------------------

expected_records = 167 * 52 * 9

print()
print(f"Expected records: {expected_records:,}")
print(f"Actual records:   {len(master):,}")

if len(master) == expected_records:

    print("PASS - 78,156 records")

else:

    raise ValueError(
        f"Expected {expected_records:,} records "
        f"but found {len(master):,}"
    )


# ------------------------------------------------------------
# UNIQUE POINTS
# ------------------------------------------------------------

unique_points = master["point_id"].nunique()

print()
print(f"Unique point IDs: {unique_points}")

if unique_points == 167:

    print("PASS - 167 sampling points")

else:

    raise ValueError(
        f"Expected 167 point IDs but found {unique_points}"
    )


# ------------------------------------------------------------
# UNIQUE WEEKS
# ------------------------------------------------------------

unique_weeks = master["Week_Start"].nunique()

print()
print(f"Unique weeks: {unique_weeks}")

if unique_weeks == 468:

    print("PASS - 468 weekly dates")

else:

    raise ValueError(
        f"Expected 468 unique weeks but found {unique_weeks}"
    )


# ------------------------------------------------------------
# YEARS
# ------------------------------------------------------------

years = sorted(
    master["Year"]
    .dropna()
    .astype(int)
    .unique()
)

expected_years = list(range(2017, 2026))

print()
print(f"Years present: {years}")

if years == expected_years:

    print("PASS - Correct years 2017–2025")

else:

    raise ValueError(
        f"Unexpected year range: {years}"
    )


# ------------------------------------------------------------
# DUPLICATE POINT-WEEK
# ------------------------------------------------------------

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

    raise ValueError(
        "Duplicate point_id + Week_Start records found."
    )


# ------------------------------------------------------------
# FULL DUPLICATES
# ------------------------------------------------------------

full_duplicates = master.duplicated().sum()

print()
print(f"Full duplicate rows: {full_duplicates}")

if full_duplicates == 0:

    print("PASS - No full duplicate rows")

else:

    raise ValueError(
        "Full duplicate rows detected."
    )


# ------------------------------------------------------------
# MISSING REQUIRED VALUES
# ------------------------------------------------------------

missing_values = (
    master[required_columns]
    .isna()
    .sum()
    .sum()
)

print()
print(
    f"Missing values in required columns: "
    f"{missing_values}"
)

if missing_values == 0:

    print("PASS - No missing required values")

else:

    print()
    print("Missing values by column:")

    missing_by_column = (
        master[required_columns]
        .isna()
        .sum()
    )

    for column, count in missing_by_column.items():

        if count > 0:
            print(f"  {column}: {count}")

    raise ValueError(
        "Missing required values detected."
    )


# ============================================================
# 14. TEMPERATURE VALIDATION
# ============================================================

print()
print("=" * 70)
print("TEMPERATURE VALIDATION")
print("=" * 70)

invalid_temperature = (
    master["Temperature_Valid"] != 1
).sum()

wrong_era5_days = (
    master["ERA5_Days"] != 7
).sum()

print()
print(
    f"Temperature_Valid != 1: "
    f"{invalid_temperature}"
)

print(
    f"ERA5_Days != 7: "
    f"{wrong_era5_days}"
)

if (
    invalid_temperature == 0
    and wrong_era5_days == 0
):

    print("PASS - All temperature records valid")

else:

    raise ValueError(
        "Temperature validation failed."
    )


# ============================================================
# 15. WEEKLY COVERAGE
# ============================================================

print()
print("=" * 70)
print("WEEKLY COVERAGE")
print("=" * 70)

weekly_counts = (
    master
    .groupby(["Year", "Week_Number"])
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

    print()
    print("Incomplete weeks:")

    for index, count in bad_weeks.items():

        year, week = index

        print(
            f"  {int(year)} "
            f"W{int(week):02d}: "
            f"{count} records"
        )

    raise ValueError(
        "Incomplete weekly coverage detected."
    )


# ============================================================
# 16. YEAR-WISE COUNTS
# ============================================================

print()
print("=" * 70)
print("YEAR-WISE RECORD COUNTS")
print("=" * 70)

year_counts = (
    master
    .groupby("Year")
    .size()
    .sort_index()
)

expected_per_year = 167 * 52

for year, count in year_counts.items():

    print(
        f"{int(year)}: "
        f"{count:,} records "
        f"(expected {expected_per_year:,})"
    )

    if count != expected_per_year:

        raise ValueError(
            f"Incorrect record count for {year}."
        )


# ============================================================
# 17. POINT METADATA CONSISTENCY
# ============================================================

print()
print("=" * 70)
print("POINT METADATA CONSISTENCY")
print("=" * 70)

metadata_check = (
    master
    .groupby("point_id")
    .agg(
        latitude_unique=("latitude", "nunique"),
        longitude_unique=("longitude", "nunique"),
        district_unique=("district", "nunique"),
    )
)

bad_points = metadata_check[
    (metadata_check["latitude_unique"] > 1)
    | (metadata_check["longitude_unique"] > 1)
    | (metadata_check["district_unique"] > 1)
]

if len(bad_points) == 0:

    print(
        "PASS - Coordinates and district are "
        "consistent for every point"
    )

else:

    print(
        f"FAIL - {len(bad_points)} points still have "
        "metadata inconsistencies after normalization."
    )

    print()
    print("DETAILS:")
    print("-" * 70)

    for point_id in bad_points.index:

        print()
        print(f"Point ID: {point_id}")

        details = (
            master[
                master["point_id"] == point_id
            ][
                [
                    "Year",
                    "point_id",
                    "latitude",
                    "longitude",
                    "district",
                ]
            ]
            .drop_duplicates()
            .sort_values("Year")
        )

        print(
            details.to_string(index=False)
        )

    raise ValueError(
        "Real metadata inconsistencies remain. "
        "Master dataset was not created."
    )


# ============================================================
# 18. WEEK DATE CONSISTENCY
# ============================================================

print()
print("=" * 70)
print("WEEK DATE VALIDATION")
print("=" * 70)

expected_week_end = (
    master["Week_Start"]
    + pd.Timedelta(days=7)
)

incorrect_week_end = (
    master["Week_End"]
    != expected_week_end
).sum()

print(
    f"Incorrect Week_End values: "
    f"{incorrect_week_end}"
)

if incorrect_week_end == 0:

    print(
        "PASS - Week_End is exactly "
        "7 days after Week_Start"
    )

else:

    raise ValueError(
        "Week_End inconsistency detected."
    )


# ============================================================
# 19. TEMPERATURE STATISTICS
# ============================================================

print()
print("=" * 70)
print("TEMPERATURE STATISTICS")
print("=" * 70)

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


# ============================================================
# 20. SOURCE CONSISTENCY
# ============================================================

print()
print("=" * 70)
print("SOURCE / METADATA VALUES")
print("=" * 70)

print()
print("Temperature source:")
print(
    master["temperature_source"]
    .value_counts()
    .to_string()
)

print()
print("Study area:")
print(
    master["study_area"]
    .value_counts()
    .to_string()
)

print()
print("Dataset:")
print(
    master["dataset"]
    .value_counts()
    .to_string()
)


# ============================================================
# 21. DATE RANGE
# ============================================================

print()
print("=" * 70)
print("DATE RANGE")
print("=" * 70)

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


# ============================================================
# 22. FINAL COLUMN ORDER
# ============================================================

master = master[required_columns]


# ============================================================
# 23. SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

master.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 24. FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("TEMPERATURE MASTER DATASET CREATED SUCCESSFULLY")
print("=" * 70)

print()
print("Output file:")
print(OUTPUT_FILE)

print()
print(f"Rows: {len(master):,}")
print(f"Columns: {len(master.columns)}")
print(
    f"Unique points: "
    f"{master['point_id'].nunique()}"
)
print(
    f"Unique weeks: "
    f"{master['Week_Start'].nunique()}"
)

print()
print("Years:")
print(
    sorted(
        master["Year"]
        .astype(int)
        .unique()
    )
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)