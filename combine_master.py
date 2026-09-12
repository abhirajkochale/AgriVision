from pathlib import Path
import pandas as pd


# ============================================================
# AGRIVISION AI
# THREE-WAY MASTER DATASET BUILDER
# NDVI + RAINFALL + TEMPERATURE
# ============================================================

print("=" * 70)
print("AGRIVISION - THREE-WAY MASTER DATASET")
print("=" * 70)


# ------------------------------------------------------------
# 1. PROJECT PATH
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_DIR / "dataset"

NDVI_FILE = (
    DATASET_DIR
    / "AgriVision_Maharashtra_NDVI_Master_2021_2025.csv"
)

RAINFALL_FILE = (
    DATASET_DIR
    / "AgriVision_Maharashtra_Rainfall_Master_2021_2025.csv"
)

TEMPERATURE_FILE = (
    DATASET_DIR
    / "AgriVision_Maharashtra_Temperature_Master_2021_2025.csv"
)

OUTPUT_FILE = (
    DATASET_DIR
    / "AgriVision_Maharashtra_Combined_Master_2021_2025.csv"
)


print()
print("Project folder:")
print(PROJECT_DIR)

print()
print("Dataset folder:")
print(DATASET_DIR)


# ------------------------------------------------------------
# 2. CHECK INPUT FILES
# ------------------------------------------------------------

input_files = {
    "NDVI": NDVI_FILE,
    "Rainfall": RAINFALL_FILE,
    "Temperature": TEMPERATURE_FILE,
}

print()
print("=" * 70)
print("CHECKING INPUT FILES")
print("=" * 70)

for name, path in input_files.items():

    if not path.exists():
        raise FileNotFoundError(
            f"\n{name} file not found:\n{path}"
        )

    print(f"PASS - {name}: {path.name}")


# ------------------------------------------------------------
# 3. READ DATASETS
# ------------------------------------------------------------

print()
print("=" * 70)
print("READING DATASETS")
print("=" * 70)

ndvi = pd.read_csv(NDVI_FILE)
rainfall = pd.read_csv(RAINFALL_FILE)
temperature = pd.read_csv(TEMPERATURE_FILE)

print()
print(f"NDVI records:         {len(ndvi):,}")
print(f"Rainfall records:     {len(rainfall):,}")
print(f"Temperature records:  {len(temperature):,}")


# ------------------------------------------------------------
# 4. CONVERT DATES
# ------------------------------------------------------------

for df in [ndvi, rainfall, temperature]:

    df["Week_Start"] = pd.to_datetime(
        df["Week_Start"],
        errors="coerce"
    )

    df["Week_End"] = pd.to_datetime(
        df["Week_End"],
        errors="coerce"
    )


# ------------------------------------------------------------
# 5. DEFINE JOIN KEY
# ------------------------------------------------------------

JOIN_KEY = [
    "point_id",
    "Week_Start",
]


# ------------------------------------------------------------
# 6. CHECK DUPLICATES IN EACH DATASET
# ------------------------------------------------------------

print()
print("=" * 70)
print("CHECKING JOIN-KEY DUPLICATES")
print("=" * 70)

datasets = {
    "NDVI": ndvi,
    "Rainfall": rainfall,
    "Temperature": temperature,
}

for name, df in datasets.items():

    duplicates = df.duplicated(
        subset=JOIN_KEY
    ).sum()

    print(
        f"{name} duplicate point-week records: "
        f"{duplicates}"
    )

    if duplicates != 0:
        raise ValueError(
            f"{name} contains duplicate point_id + Week_Start records."
        )


# ------------------------------------------------------------
# 7. REMOVE REDUNDANT COLUMNS BEFORE MERGE
# ------------------------------------------------------------

# Rainfall is our complete 43,420-row weekly framework.
# It provides the canonical metadata columns.

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

temp = temperature[
    JOIN_KEY + temperature_values
].copy()

ndvi_small = ndvi[
    JOIN_KEY + ndvi_values
].copy()


# ------------------------------------------------------------
# 8. MERGE TEMPERATURE
# ------------------------------------------------------------

print()
print("=" * 70)
print("MERGING TEMPERATURE")
print("=" * 70)

combined = base.merge(
    temp,
    on=JOIN_KEY,
    how="left",
    validate="one_to_one"
)

print(
    f"Records after temperature merge: "
    f"{len(combined):,}"
)


# ------------------------------------------------------------
# 9. MERGE NDVI
# ------------------------------------------------------------

print()
print("=" * 70)
print("MERGING NDVI")
print("=" * 70)

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


# ------------------------------------------------------------
# 10. ADD GLOBAL METADATA
# ------------------------------------------------------------

combined["study_area"] = "Maharashtra"

combined["dataset"] = (
    "AgriVision Combined NDVI Rainfall Temperature"
)


# ------------------------------------------------------------
# 11. SORT
# ------------------------------------------------------------

combined = combined.sort_values(
    by=["Week_Start", "point_id"]
).reset_index(drop=True)


# ------------------------------------------------------------
# 12. VALIDATION
# ------------------------------------------------------------

print()
print("=" * 70)
print("COMBINED DATASET VALIDATION")
print("=" * 70)


# ------------------------------------------------------------
# TOTAL RECORDS
# ------------------------------------------------------------

print()
print(f"Expected records: {167 * 260:,}")
print(f"Actual records:   {len(combined):,}")

if len(combined) == 43420:
    print("PASS - 43,420 combined records")
else:
    print("FAIL - Unexpected combined record count")


# ------------------------------------------------------------
# POINTS
# ------------------------------------------------------------

unique_points = combined["point_id"].nunique()

print()
print(f"Unique points: {unique_points}")

if unique_points == 167:
    print("PASS - 167 points")
else:
    print("FAIL - Unexpected point count")


# ------------------------------------------------------------
# WEEKS
# ------------------------------------------------------------

unique_weeks = combined["Week_Start"].nunique()

print()
print(f"Unique weeks: {unique_weeks}")

if unique_weeks == 260:
    print("PASS - 260 weeks")
else:
    print("FAIL - Unexpected week count")


# ------------------------------------------------------------
# YEARS
# ------------------------------------------------------------

years = sorted(
    combined["Year"].dropna().astype(int).unique()
)

print()
print(f"Years: {years}")

if years == [2021, 2022, 2023, 2024, 2025]:
    print("PASS - Correct years")
else:
    print("FAIL - Incorrect years")


# ------------------------------------------------------------
# DUPLICATE JOIN KEYS
# ------------------------------------------------------------

duplicates = combined.duplicated(
    subset=JOIN_KEY
).sum()

print()
print(
    f"Duplicate point-week records: "
    f"{duplicates}"
)

if duplicates == 0:
    print("PASS - No duplicate join keys")
else:
    print("FAIL - Duplicate join keys found")


# ------------------------------------------------------------
# FULL DUPLICATES
# ------------------------------------------------------------

full_duplicates = combined.duplicated().sum()

print()
print(
    f"Full duplicate rows: "
    f"{full_duplicates}"
)

if full_duplicates == 0:
    print("PASS - No full duplicates")
else:
    print("FAIL - Full duplicates found")


# ------------------------------------------------------------
# 13. RAINFALL COMPLETENESS
# ------------------------------------------------------------

print()
print("=" * 70)
print("RAINFALL VALIDATION")
print("=" * 70)

rainfall_missing = combined["Rainfall_mm"].isna().sum()

rainfall_invalid = (
    combined["Rainfall_Valid"] != 1
).sum()

rainfall_wrong_days = (
    combined["CHIRPS_Days"] != 7
).sum()

print(
    f"Missing rainfall values: {rainfall_missing}"
)

print(
    f"Invalid rainfall flags: {rainfall_invalid}"
)

print(
    f"CHIRPS_Days != 7: {rainfall_wrong_days}"
)

if (
    rainfall_missing == 0
    and rainfall_invalid == 0
    and rainfall_wrong_days == 0
):
    print("PASS - Rainfall fully complete")
else:
    print("FAIL - Rainfall issue detected")


# ------------------------------------------------------------
# 14. TEMPERATURE COMPLETENESS
# ------------------------------------------------------------

print()
print("=" * 70)
print("TEMPERATURE VALIDATION")
print("=" * 70)

temperature_missing = (
    combined["Temperature_C"].isna().sum()
)

temperature_invalid = (
    combined["Temperature_Valid"] != 1
).sum()

temperature_wrong_days = (
    combined["ERA5_Days"] != 7
).sum()

print(
    f"Missing temperature values: "
    f"{temperature_missing}"
)

print(
    f"Invalid temperature flags: "
    f"{temperature_invalid}"
)

print(
    f"ERA5_Days != 7: "
    f"{temperature_wrong_days}"
)

if (
    temperature_missing == 0
    and temperature_invalid == 0
    and temperature_wrong_days == 0
):
    print("PASS - Temperature fully complete")
else:
    print("FAIL - Temperature issue detected")


# ------------------------------------------------------------
# 15. NDVI COVERAGE
# ------------------------------------------------------------

print()
print("=" * 70)
print("NDVI COVERAGE")
print("=" * 70)

ndvi_available = (
    combined["NDVI"].notna().sum()
)

ndvi_missing = (
    combined["NDVI"].isna().sum()
)

print(
    f"Valid NDVI records: "
    f"{ndvi_available:,}"
)

print(
    f"Missing NDVI records: "
    f"{ndvi_missing:,}"
)

print(
    f"NDVI coverage: "
    f"{(ndvi_available / len(combined)) * 100:.2f}%"
)


# ------------------------------------------------------------
# EXPECTED NDVI COVERAGE
# ------------------------------------------------------------

if ndvi_available == 27797:
    print(
        "PASS - NDVI coverage matches "
        "validated master"
    )
else:
    print(
        "WARNING - NDVI count differs from "
        "expected 27,797"
    )


# ------------------------------------------------------------
# 16. NDVI VALID FLAG
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
        "PASS - NDVI validity flag matches "
        "available NDVI records"
    )
else:
    print(
        "WARNING - NDVI validity flag mismatch"
    )


# ------------------------------------------------------------
# 17. COMPLETE-CASE COUNT
# ------------------------------------------------------------

complete_cases = combined[
    combined["NDVI"].notna()
    & combined["Rainfall_mm"].notna()
    & combined["Temperature_C"].notna()
].copy()

print()
print("=" * 70)
print("COMPLETE CASES")
print("=" * 70)

print(
    f"Rows with NDVI + rainfall + temperature: "
    f"{len(complete_cases):,}"
)

# Because rainfall and temperature are complete,
# this should equal available NDVI records.

if len(complete_cases) == ndvi_available:
    print(
        "PASS - Complete-case count matches "
        "available NDVI records"
    )
else:
    print(
        "WARNING - Complete-case mismatch"
    )


# ------------------------------------------------------------
# 18. METADATA CONSISTENCY
# ------------------------------------------------------------

print()
print("=" * 70)
print("METADATA CONSISTENCY")
print("=" * 70)


for column in [
    "latitude",
    "longitude",
    "district",
]:
    unique_counts = (
        combined.groupby("point_id")[column]
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
        print(
            f"FAIL - {column} inconsistent"
        )


# ------------------------------------------------------------
# 19. WEEK DATE CONSISTENCY
# ------------------------------------------------------------

print()
print("=" * 70)
print("WEEK DATE VALIDATION")
print("=" * 70)

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
    print(
        "FAIL - Week_End inconsistency"
    )


# ------------------------------------------------------------
# 20. RANGES
# ------------------------------------------------------------

print()
print("=" * 70)
print("VALUE RANGES")
print("=" * 70)

print(
    f"NDVI min: "
    f"{combined['NDVI'].min()}"
)

print(
    f"NDVI max: "
    f"{combined['NDVI'].max()}"
)

print(
    f"Rainfall min: "
    f"{combined['Rainfall_mm'].min():.4f} mm"
)

print(
    f"Rainfall max: "
    f"{combined['Rainfall_mm'].max():.4f} mm"
)

print(
    f"Temperature min: "
    f"{combined['Temperature_C'].min():.4f} °C"
)

print(
    f"Temperature max: "
    f"{combined['Temperature_C'].max():.4f} °C"
)


# ------------------------------------------------------------
# 21. FINAL COLUMN ORDER
# ------------------------------------------------------------

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

combined = combined[final_columns]


# ------------------------------------------------------------
# 22. SAVE
# ------------------------------------------------------------

combined.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# 23. FINAL SUMMARY
# ------------------------------------------------------------

print()
print("=" * 70)
print("COMBINED MASTER DATASET CREATED SUCCESSFULLY")
print("=" * 70)

print()
print("Output:")
print(OUTPUT_FILE)

print()
print(f"Rows: {len(combined):,}")
print(f"Columns: {len(combined.columns)}")

print()
print("NDVI available:")
print(f"  {combined['NDVI'].notna().sum():,}")

print("NDVI missing:")
print(f"  {combined['NDVI'].isna().sum():,}")

print()
print("Rainfall available:")
print(f"  {combined['Rainfall_mm'].notna().sum():,}")

print("Temperature available:")
print(f"  {combined['Temperature_C'].notna().sum():,}")

print()
print("Complete NDVI + Rainfall + Temperature rows:")
print(
    f"  {combined[['NDVI', 'Rainfall_mm', 'Temperature_C']].notna().all(axis=1).sum():,}"
)

print()
print("Final columns:")

for i, column in enumerate(
    combined.columns,
    start=1
):
    print(f"{i:2}. {column}")

print()
print("=" * 70)
print("DONE")
print("=" * 70)