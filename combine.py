from pathlib import Path
import pandas as pd


# ============================================================
# AGRIVISION AI
# THREE-WAY MASTER DATASET BUILDER
# 2017–2025
#
# NDVI + RAINFALL + TEMPERATURE
# ============================================================

print("=" * 75)
print("AGRIVISION - THREE-WAY MASTER DATASET")
print("2017–2025")
print("=" * 75)


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"

# Masters are stored in their respective folders
NDVI_FILE = (
    DATASET_DIR
    / "NDVI"
    / "AgriVision_Maharashtra_NDVI_Master_2017_2025.csv"
)

RAINFALL_FILE = (
    DATASET_DIR
    / "Rainfall"
    / "AgriVision_Maharashtra_Rainfall_Master_2017_2025.csv"
)

TEMPERATURE_FILE = (
    DATASET_DIR
    / "Temperature"
    / "AgriVision_Maharashtra_Temperature_Master_2017_2025.csv"
)

# Final combined master is stored directly inside dataset/
OUTPUT_FILE = (
    DATASET_DIR
    / "AgriVision_Maharashtra_Combined_Master_2017_2025.csv"
)


print()
print("Project folder:")
print(PROJECT_DIR)

print()
print("Dataset folder:")
print(DATASET_DIR)


# ============================================================
# 2. CHECK INPUT FILES
# ============================================================

print()
print("=" * 75)
print("CHECKING INPUT FILES")
print("=" * 75)

input_files = {
    "NDVI": NDVI_FILE,
    "Rainfall": RAINFALL_FILE,
    "Temperature": TEMPERATURE_FILE,
}

for name, path in input_files.items():

    if not path.exists():

        raise FileNotFoundError(
            f"\n{name} file not found:\n{path}"
        )

    print(f"PASS - {name}: {path}")


# ============================================================
# 3. READ DATASETS
# ============================================================

print()
print("=" * 75)
print("READING DATASETS")
print("=" * 75)

ndvi = pd.read_csv(NDVI_FILE)
rainfall = pd.read_csv(RAINFALL_FILE)
temperature = pd.read_csv(TEMPERATURE_FILE)

print()
print(f"NDVI records:         {len(ndvi):,}")
print(f"Rainfall records:     {len(rainfall):,}")
print(f"Temperature records:  {len(temperature):,}")


# ============================================================
# 4. REQUIRED COLUMNS
# ============================================================

print()
print("=" * 75)
print("CHECKING DATASET SCHEMAS")
print("=" * 75)

required_ndvi_columns = [
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
    "NDVI",
    "NDVI_Valid",
    "Sentinel2_Images",
    "Cloud_Probability_Threshold",
    "Scene_Cloud_Threshold",
]

required_rainfall_columns = [
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
]

required_temperature_columns = [
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
]

dataset_checks = {
    "NDVI": (ndvi, required_ndvi_columns),
    "Rainfall": (rainfall, required_rainfall_columns),
    "Temperature": (temperature, required_temperature_columns),
}

for name, (df, required_columns) in dataset_checks.items():

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"\n{name} is missing columns:\n"
            f"{missing_columns}"
        )

    print(f"PASS - {name} schema")


# ============================================================
# 5. DATE CONVERSION
# ============================================================

for df in [ndvi, rainfall, temperature]:

    df["Week_Start"] = pd.to_datetime(
        df["Week_Start"],
        errors="coerce"
    )

    df["Week_End"] = pd.to_datetime(
        df["Week_End"],
        errors="coerce"
    )


# ============================================================
# 6. NORMALIZE JOIN / METADATA FIELDS
# ============================================================

for df in [ndvi, rainfall, temperature]:

    df["point_id"] = (
        df["point_id"]
        .astype(str)
        .str.strip()
    )

    df["latitude"] = (
        pd.to_numeric(
            df["latitude"],
            errors="coerce"
        )
        .round(6)
    )

    df["longitude"] = (
        pd.to_numeric(
            df["longitude"],
            errors="coerce"
        )
        .round(6)
    )

    df["district"] = (
        df["district"]
        .astype(str)
        .str.strip()
    )


# ============================================================
# 7. JOIN KEY
# ============================================================

JOIN_KEY = [
    "point_id",
    "Week_Start",
]


# ============================================================
# 8. CHECK DUPLICATES IN EACH DATASET
# ============================================================

print()
print("=" * 75)
print("CHECKING JOIN-KEY DUPLICATES")
print("=" * 75)

datasets = {
    "NDVI": ndvi,
    "Rainfall": rainfall,
    "Temperature": temperature,
}

for name, df in datasets.items():

    duplicate_count = (
        df
        .duplicated(subset=JOIN_KEY)
        .sum()
    )

    print(
        f"{name} duplicate point-week records: "
        f"{duplicate_count}"
    )

    if duplicate_count != 0:

        raise ValueError(
            f"{name} contains duplicate "
            f"point_id + Week_Start records."
        )


# ============================================================
# 9. CHECK RAINFALL / TEMPERATURE FRAMEWORK
# ============================================================

print()
print("=" * 75)
print("CHECKING BASE FRAMEWORK")
print("=" * 75)

rainfall_keys = set(
    zip(
        rainfall["point_id"],
        rainfall["Week_Start"]
    )
)

temperature_keys = set(
    zip(
        temperature["point_id"],
        temperature["Week_Start"]
    )
)

print(
    f"Rainfall join keys:     {len(rainfall_keys):,}"
)

print(
    f"Temperature join keys:  {len(temperature_keys):,}"
)

if rainfall_keys != temperature_keys:

    only_rainfall = rainfall_keys - temperature_keys
    only_temperature = temperature_keys - rainfall_keys

    print()
    print(
        f"Keys only in rainfall:    "
        f"{len(only_rainfall):,}"
    )

    print(
        f"Keys only in temperature: "
        f"{len(only_temperature):,}"
    )

    raise ValueError(
        "Rainfall and temperature point-week "
        "frameworks do not match."
    )

print(
    "PASS - Rainfall and temperature have "
    "identical point-week keys"
)


# ============================================================
# 10. CREATE BASE TABLE
# ============================================================

base_columns = [
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

rainfall_values = [
    "Rainfall_mm",
    "Rainfall_Valid",
    "CHIRPS_Days",
    "rainfall_source",
]

temperature_values = [
    "Temperature_C",
    "Temperature_Valid",
    "ERA5_Days",
    "temperature_source",
]

ndvi_values = [
    "NDVI",
    "NDVI_Valid",
    "Sentinel2_Images",
    "Cloud_Probability_Threshold",
    "Scene_Cloud_Threshold",
]


base = rainfall[
    base_columns + rainfall_values
].copy()

temperature_small = temperature[
    JOIN_KEY + temperature_values
].copy()

ndvi_small = ndvi[
    JOIN_KEY + ndvi_values
].copy()


# ============================================================
# 11. MERGE TEMPERATURE
# ============================================================

print()
print("=" * 75)
print("MERGING TEMPERATURE")
print("=" * 75)

combined = base.merge(
    temperature_small,
    on=JOIN_KEY,
    how="left",
    validate="one_to_one"
)

print(
    f"Records after temperature merge: "
    f"{len(combined):,}"
)


# ============================================================
# 12. MERGE NDVI
# ============================================================

print()
print("=" * 75)
print("MERGING NDVI")
print("=" * 75)

combined = combined.merge(
    ndvi_small,
    on=JOIN_KEY,
    how="left",
    validate="one_to_one"
)

print(
    f"Records after NDVI merge: "
    f"{len(combined):,}"
)


# ============================================================
# 13. GLOBAL METADATA
# ============================================================

combined["study_area"] = "Maharashtra"

combined["dataset"] = (
    "AgriVision Combined NDVI Rainfall Temperature"
)


# ============================================================
# 14. SORT
# ============================================================

combined = (
    combined
    .sort_values(
        by=["Week_Start", "point_id"]
    )
    .reset_index(drop=True)
)


# ============================================================
# 15. MASTER VALIDATION
# ============================================================

print()
print("=" * 75)
print("COMBINED DATASET VALIDATION")
print("=" * 75)


# ------------------------------------------------------------
# TOTAL RECORDS
# ------------------------------------------------------------

expected_records = 167 * 52 * 9

print()
print(
    f"Expected records: {expected_records:,}"
)

print(
    f"Actual records:   {len(combined):,}"
)

if len(combined) == expected_records:

    print(
        "PASS - 78,156 combined records"
    )

else:

    raise ValueError(
        "Unexpected combined record count."
    )


# ------------------------------------------------------------
# UNIQUE POINTS
# ------------------------------------------------------------

unique_points = (
    combined["point_id"]
    .nunique()
)

print()
print(
    f"Unique points: {unique_points}"
)

if unique_points == 167:

    print("PASS - 167 points")

else:

    raise ValueError(
        "Unexpected point count."
    )


# ------------------------------------------------------------
# UNIQUE WEEKS
# ------------------------------------------------------------

unique_weeks = (
    combined["Week_Start"]
    .nunique()
)

print()
print(
    f"Unique weeks: {unique_weeks}"
)

if unique_weeks == 468:

    print("PASS - 468 weeks")

else:

    raise ValueError(
        "Unexpected week count."
    )


# ------------------------------------------------------------
# YEARS
# ------------------------------------------------------------

years = sorted(
    combined["Year"]
    .dropna()
    .astype(int)
    .unique()
)

expected_years = list(range(2017, 2026))

print()
print(
    f"Years: {years}"
)

if years == expected_years:

    print(
        "PASS - Correct years 2017–2025"
    )

else:

    raise ValueError(
        f"Incorrect years: {years}"
    )


# ------------------------------------------------------------
# DUPLICATE JOIN KEYS
# ------------------------------------------------------------

duplicate_join_keys = (
    combined
    .duplicated(
        subset=JOIN_KEY
    )
    .sum()
)

print()
print(
    "Duplicate point-week records: "
    f"{duplicate_join_keys}"
)

if duplicate_join_keys == 0:

    print(
        "PASS - No duplicate join keys"
    )

else:

    raise ValueError(
        "Duplicate point-week records found."
    )


# ------------------------------------------------------------
# FULL DUPLICATES
# ------------------------------------------------------------

full_duplicates = (
    combined
    .duplicated()
    .sum()
)

print()
print(
    f"Full duplicate rows: "
    f"{full_duplicates}"
)

if full_duplicates == 0:

    print(
        "PASS - No full duplicates"
    )

else:

    raise ValueError(
        "Full duplicate rows found."
    )


# ============================================================
# 16. RAINFALL VALIDATION
# ============================================================

print()
print("=" * 75)
print("RAINFALL VALIDATION")
print("=" * 75)

rainfall_missing = (
    combined["Rainfall_mm"]
    .isna()
    .sum()
)

rainfall_invalid = (
    combined["Rainfall_Valid"] != 1
).sum()

rainfall_wrong_days = (
    combined["CHIRPS_Days"] != 7
).sum()

rainfall_negative = (
    combined["Rainfall_mm"] < 0
).sum()

print(
    f"Missing rainfall:       "
    f"{rainfall_missing}"
)

print(
    f"Invalid rainfall flags: "
    f"{rainfall_invalid}"
)

print(
    f"CHIRPS_Days != 7:       "
    f"{rainfall_wrong_days}"
)

print(
    f"Negative rainfall:      "
    f"{rainfall_negative}"
)

if (
    rainfall_missing == 0
    and rainfall_invalid == 0
    and rainfall_wrong_days == 0
    and rainfall_negative == 0
):

    print(
        "PASS - Rainfall fully valid"
    )

else:

    raise ValueError(
        "Rainfall validation failed."
    )


# ============================================================
# 17. TEMPERATURE VALIDATION
# ============================================================

print()
print("=" * 75)
print("TEMPERATURE VALIDATION")
print("=" * 75)

temperature_missing = (
    combined["Temperature_C"]
    .isna()
    .sum()
)

temperature_invalid = (
    combined["Temperature_Valid"] != 1
).sum()

temperature_wrong_days = (
    combined["ERA5_Days"] != 7
).sum()

print(
    f"Missing temperature:       "
    f"{temperature_missing}"
)

print(
    f"Invalid temperature flags: "
    f"{temperature_invalid}"
)

print(
    f"ERA5_Days != 7:             "
    f"{temperature_wrong_days}"
)

if (
    temperature_missing == 0
    and temperature_invalid == 0
    and temperature_wrong_days == 0
):

    print(
        "PASS - Temperature fully valid"
    )

else:

    raise ValueError(
        "Temperature validation failed."
    )


# ============================================================
# 18. NDVI COVERAGE
# ============================================================

print()
print("=" * 75)
print("NDVI COVERAGE")
print("=" * 75)

ndvi_available = (
    combined["NDVI"]
    .notna()
    .sum()
)

ndvi_missing = (
    combined["NDVI"]
    .isna()
    .sum()
)

ndvi_coverage = (
    ndvi_available
    / len(combined)
    * 100
)

print(
    f"Valid NDVI records: "
    f"{ndvi_available:,}"
)

print(
    f"Missing NDVI rows:  "
    f"{ndvi_missing:,}"
)

print(
    f"NDVI coverage:      "
    f"{ndvi_coverage:.2f}%"
)


# ------------------------------------------------------------
# NDVI RANGE
# ------------------------------------------------------------

ndvi_min = combined["NDVI"].min()
ndvi_max = combined["NDVI"].max()

print()
print(
    f"NDVI minimum: {ndvi_min}"
)

print(
    f"NDVI maximum: {ndvi_max}"
)

if (
    ndvi_min >= -1
    and ndvi_max <= 1
):

    print(
        "PASS - NDVI values are within [-1, 1]"
    )

else:

    raise ValueError(
        "NDVI values outside [-1, 1] detected."
    )


# ------------------------------------------------------------
# NDVI VALID FLAG
# ------------------------------------------------------------

ndvi_valid_count = (
    combined["NDVI_Valid"] == 1
).sum()

print()
print(
    f"NDVI_Valid = 1: "
    f"{ndvi_valid_count:,}"
)

if ndvi_valid_count == ndvi_available:

    print(
        "PASS - NDVI validity matches "
        "available observations"
    )

else:

    raise ValueError(
        "NDVI validity flag mismatch."
    )


# ============================================================
# 19. COMPLETE CASES
# ============================================================

print()
print("=" * 75)
print("COMPLETE CASES")
print("=" * 75)

complete_case_mask = (
    combined["NDVI"].notna()
    & combined["Rainfall_mm"].notna()
    & combined["Temperature_C"].notna()
)

complete_case_count = (
    complete_case_mask.sum()
)

print(
    "Rows with NDVI + rainfall + temperature: "
    f"{complete_case_count:,}"
)

if complete_case_count == ndvi_available:

    print(
        "PASS - Every NDVI observation has "
        "matching rainfall and temperature"
    )

else:

    raise ValueError(
        "Complete-case count does not match NDVI coverage."
    )


# ============================================================
# 20. POINT METADATA CONSISTENCY
# ============================================================

print()
print("=" * 75)
print("POINT METADATA CONSISTENCY")
print("=" * 75)

for column in [
    "latitude",
    "longitude",
    "district",
]:

    unique_counts = (
        combined
        .groupby("point_id")[column]
        .nunique()
    )

    inconsistent = (
        unique_counts > 1
    ).sum()

    print(
        f"{column} inconsistent points: "
        f"{inconsistent}"
    )

    if inconsistent == 0:

        print(
            f"PASS - {column} consistent"
        )

    else:

        raise ValueError(
            f"{column} inconsistency detected."
        )


# ============================================================
# 21. WEEK DATE CONSISTENCY
# ============================================================

print()
print("=" * 75)
print("WEEK DATE VALIDATION")
print("=" * 75)

expected_week_end = (
    combined["Week_Start"]
    + pd.Timedelta(days=7)
)

bad_week_end = (
    combined["Week_End"]
    != expected_week_end
).sum()

print(
    f"Incorrect Week_End values: "
    f"{bad_week_end}"
)

if bad_week_end == 0:

    print(
        "PASS - Week_End is exactly "
        "7 days after Week_Start"
    )

else:

    raise ValueError(
        "Week_End inconsistency detected."
    )


# ============================================================
# 22. FINAL COLUMN ORDER
# ============================================================

final_columns = [
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

    "NDVI",
    "NDVI_Valid",
    "Sentinel2_Images",
    "Cloud_Probability_Threshold",
    "Scene_Cloud_Threshold",

    "Rainfall_mm",
    "Rainfall_Valid",
    "CHIRPS_Days",
    "rainfall_source",

    "Temperature_C",
    "Temperature_Valid",
    "ERA5_Days",
    "temperature_source",

    "study_area",
    "dataset",
]

combined = combined[
    final_columns
]


# ============================================================
# 23. SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

combined.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 24. FINAL SUMMARY
# ============================================================

print()
print("=" * 75)
print("COMBINED MASTER DATASET CREATED SUCCESSFULLY")
print("=" * 75)

print()
print("Output:")
print(OUTPUT_FILE)

print()
print(
    f"Rows:    {len(combined):,}"
)

print(
    f"Columns: {len(combined.columns)}"
)

print()
print(
    f"Rainfall available:    "
    f"{combined['Rainfall_mm'].notna().sum():,}"
)

print(
    f"Temperature available: "
    f"{combined['Temperature_C'].notna().sum():,}"
)

print(
    f"NDVI available:        "
    f"{combined['NDVI'].notna().sum():,}"
)

print()
print(
    "Complete NDVI + rainfall + temperature rows: "
    f"{combined[['NDVI', 'Rainfall_mm', 'Temperature_C']].notna().all(axis=1).sum():,}"
)

print()
print("Years:")
print(
    sorted(
        combined["Year"]
        .astype(int)
        .unique()
    )
)

print()
print("=" * 75)
print("DONE")
print("=" * 75)