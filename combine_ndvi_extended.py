from pathlib import Path
import pandas as pd


# ============================================================
# AGRIVISION - NDVI MASTER 2017-2025
# ============================================================

print("=" * 70)
print("AGRIVISION - NDVI MASTER DATASET 2017-2025")
print("=" * 70)


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

NDVI_DIR = PROJECT_DIR / "dataset" / "NDVI"

OUTPUT_FILE = (
    NDVI_DIR
    / "AgriVision_Maharashtra_NDVI_Master_2017_2025.csv"
)


# ------------------------------------------------------------
# 2. SOURCE FILES
# ------------------------------------------------------------

EXTENDED_FILES = [
    "AgriVision_Maharashtra_NDVI_2017_W1_W26.csv",
    "AgriVision_Maharashtra_NDVI_2017_W27_W52.csv",
    "AgriVision_Maharashtra_NDVI_2018_W1_W26.csv",
    "AgriVision_Maharashtra_NDVI_2018_W27_W52.csv",
    "AgriVision_Maharashtra_NDVI_2019_W1_W26.csv",
    "AgriVision_Maharashtra_NDVI_2019_W27_W52.csv",
    "AgriVision_Maharashtra_NDVI_2020_W1_W26.csv",
    "AgriVision_Maharashtra_NDVI_2020_W27_W52.csv",
]

EXISTING_MASTER = (
    "AgriVision_Maharashtra_NDVI_Master_2021_2025.csv"
)


ALL_FILES = EXTENDED_FILES + [EXISTING_MASTER]


# ------------------------------------------------------------
# 3. CHECK FILES
# ------------------------------------------------------------

print()
print("NDVI folder:")
print(NDVI_DIR)

for filename in ALL_FILES:

    path = NDVI_DIR / filename

    if not path.exists():

        raise FileNotFoundError(
            f"\nMissing NDVI file:\n{path}"
        )

    print("PASS -", filename)


# ------------------------------------------------------------
# 4. REQUIRED COLUMNS
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
    "NDVI",
    "NDVI_Valid",
    "Sentinel2_Images",
    "Cloud_Probability_Threshold",
    "Scene_Cloud_Threshold",
    "study_area",
    "dataset",
]


# ------------------------------------------------------------
# 5. READ FILES
# ------------------------------------------------------------

dataframes = []

print()
print("=" * 70)
print("READING NDVI FILES")
print("=" * 70)

for filename in ALL_FILES:

    path = NDVI_DIR / filename

    df = pd.read_csv(path)

    # Remove GEE-generated columns if present
    drop_columns = [
        "system:index",
        ".geo",
    ]

    for col in drop_columns:
        if col in df.columns:
            df = df.drop(columns=[col])

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"\nMissing columns in {filename}:\n{missing}"
        )

    df = df[required_columns].copy()

    df["Week_Start"] = pd.to_datetime(
        df["Week_Start"],
        errors="coerce"
    )

    df["Week_End"] = pd.to_datetime(
        df["Week_End"],
        errors="coerce"
    )

    dataframes.append(df)

    print(
        f"{filename}: {len(df):,} rows"
    )


# ------------------------------------------------------------
# 6. COMBINE
# ------------------------------------------------------------

master = pd.concat(
    dataframes,
    ignore_index=True
)

master = master.sort_values(
    ["Week_Start", "point_id"]
).reset_index(drop=True)


print()
print("=" * 70)
print("COMBINED DATASET")
print("=" * 70)

print(
    f"Combined records: {len(master):,}"
)


# ------------------------------------------------------------
# 7. DUPLICATE CHECK
# ------------------------------------------------------------

duplicate_point_week = (
    master.duplicated(
        subset=["point_id", "Week_Start"]
    ).sum()
)

full_duplicates = master.duplicated().sum()

print()
print(
    f"Duplicate point-week records: "
    f"{duplicate_point_week}"
)

print(
    f"Full duplicate rows: "
    f"{full_duplicates}"
)

if duplicate_point_week != 0:

    raise ValueError(
        "Duplicate point_id + Week_Start records found."
    )

if full_duplicates != 0:

    raise ValueError(
        "Full duplicate rows found."
    )


# ------------------------------------------------------------
# 8. BASIC VALIDATION
# ------------------------------------------------------------

print()
print("=" * 70)
print("VALIDATION")
print("=" * 70)

unique_points = master["point_id"].nunique()

unique_weeks = master["Week_Start"].nunique()

years = sorted(
    master["Year"]
    .dropna()
    .astype(int)
    .unique()
)

print(
    f"Unique points: {unique_points}"
)

print(
    f"Unique observed weeks: {unique_weeks}"
)

print(
    f"Years: {years}"
)

if unique_points != 167:
    raise ValueError(
        f"Expected 167 points, found {unique_points}"
    )

if years != list(range(2017, 2026)):
    raise ValueError(
        f"Unexpected year range: {years}"
    )


# ------------------------------------------------------------
# 9. MISSING VALUES
# ------------------------------------------------------------

missing_values = (
    master[required_columns]
    .isna()
    .sum()
)

missing_total = missing_values.sum()

print()
print(
    f"Missing values in required columns: "
    f"{missing_total}"
)

if missing_total != 0:

    print(
        missing_values[
            missing_values > 0
        ].to_string()
    )

    raise ValueError(
        "Missing required values found."
    )


# ------------------------------------------------------------
# 10. NDVI RANGE
# ------------------------------------------------------------

invalid_ndvi = (
    (master["NDVI"] < -1)
    | (master["NDVI"] > 1)
).sum()

print()
print(
    f"NDVI values outside [-1,1]: "
    f"{invalid_ndvi}"
)

if invalid_ndvi != 0:

    raise ValueError(
        "Invalid NDVI values found."
    )


# ------------------------------------------------------------
# 11. NDVI VALID FLAG
# ------------------------------------------------------------

invalid_flag = (
    master["NDVI_Valid"] != 1
).sum()

print(
    f"NDVI_Valid != 1: "
    f"{invalid_flag}"
)

if invalid_flag != 0:

    raise ValueError(
        "Invalid NDVI validity flags found."
    )


# ------------------------------------------------------------
# 12. WEEK DATE CONSISTENCY
# ------------------------------------------------------------

expected_week_end = (
    master["Week_Start"]
    + pd.Timedelta(days=7)
)

bad_week_end = (
    master["Week_End"]
    != expected_week_end
).sum()

print(
    f"Incorrect Week_End values: "
    f"{bad_week_end}"
)

if bad_week_end != 0:

    raise ValueError(
        "Week_End is not exactly 7 days after Week_Start."
    )


# ------------------------------------------------------------
# 13. YEAR-WISE COUNTS
# ------------------------------------------------------------

year_counts = (
    master.groupby("Year")
    .size()
    .sort_index()
)

print()
print("=" * 70)
print("YEAR-WISE NDVI RECORD COUNT")
print("=" * 70)

print(
    year_counts.to_string()
)


# ------------------------------------------------------------
# 14. WEEKLY COVERAGE
# ------------------------------------------------------------

weekly_counts = (
    master.groupby(
        ["Year", "Week_Number"]
    )
    .size()
)

print()
print("=" * 70)
print("WEEKLY COVERAGE")
print("=" * 70)

missing_week_combinations = []

for year in range(2017, 2026):

    for week in range(1, 53):

        count = weekly_counts.get(
            (year, week),
            0
        )

        if count == 0:

            missing_week_combinations.append(
                (year, week)
            )

print(
    f"Year-week combinations with zero records: "
    f"{len(missing_week_combinations)}"
)

if missing_week_combinations:

    print()

    for year, week in missing_week_combinations:

        print(
            f"  {year} W{week:02d}"
        )


# ------------------------------------------------------------
# 15. COVERAGE STATISTICS
# ------------------------------------------------------------

possible_point_weeks = (
    167 * 468
)

coverage_pct = (
    len(master)
    / possible_point_weeks
    * 100
)

print()
print(
    f"Theoretical point-weeks: "
    f"{possible_point_weeks:,}"
)

print(
    f"Observed NDVI point-weeks: "
    f"{len(master):,}"
)

print(
    f"Overall NDVI coverage: "
    f"{coverage_pct:.2f}%"
)


# ------------------------------------------------------------
# 16. POINT-LEVEL COVERAGE
# ------------------------------------------------------------

point_counts = (
    master.groupby("point_id")
    .size()
)

print()
print(
    f"Minimum observations for a point: "
    f"{point_counts.min()}"
)

print(
    f"Maximum observations for a point: "
    f"{point_counts.max()}"
)

print(
    f"Mean observations per point: "
    f"{point_counts.mean():.2f}"
)


# ------------------------------------------------------------
# 17. NDVI STATISTICS
# ------------------------------------------------------------

print()
print("=" * 70)
print("NDVI STATISTICS")
print("=" * 70)

print(
    f"Minimum NDVI: "
    f"{master['NDVI'].min():.6f}"
)

print(
    f"Maximum NDVI: "
    f"{master['NDVI'].max():.6f}"
)

print(
    f"Mean NDVI: "
    f"{master['NDVI'].mean():.6f}"
)

print(
    f"Median NDVI: "
    f"{master['NDVI'].median():.6f}"
)


# ------------------------------------------------------------
# 18. YEAR-WISE NDVI STATISTICS
# ------------------------------------------------------------

year_stats = (
    master.groupby("Year")["NDVI"]
    .agg(
        ["count", "min", "max", "mean", "median"]
    )
)

year_stats["coverage_pct"] = (
    year_stats["count"]
    / (167 * 52)
    * 100
)

print()
print(
    year_stats.to_string()
)


# ------------------------------------------------------------
# 19. CONSECUTIVE NDVI PAIRS
# ------------------------------------------------------------

print()
print("=" * 70)
print("CONSECUTIVE NDVI PAIRS")
print("=" * 70)

sorted_master = master.sort_values(
    ["point_id", "Week_Start"]
).copy()

sorted_master["Next_Week_Start"] = (
    sorted_master
    .groupby("point_id")["Week_Start"]
    .shift(-1)
)

sorted_master["Next_NDVI"] = (
    sorted_master
    .groupby("point_id")["NDVI"]
    .shift(-1)
)

sorted_master["Next_Week_Gap_Days"] = (
    sorted_master["Next_Week_Start"]
    - sorted_master["Week_Start"]
).dt.days

valid_pairs = (
    sorted_master["NDVI"].notna()
    & sorted_master["Next_NDVI"].notna()
    & (
        sorted_master["Next_Week_Gap_Days"]
        == 7
    )
)

consecutive_pairs = int(
    valid_pairs.sum()
)

print(
    f"Valid current → next-week NDVI pairs: "
    f"{consecutive_pairs:,}"
)


# ------------------------------------------------------------
# 20. CONSECUTIVE PAIRS BY YEAR
# ------------------------------------------------------------

pair_year_counts = (
    sorted_master.loc[
        valid_pairs
    ]
    .groupby("Year")
    .size()
    .sort_index()
)

print()
print(
    pair_year_counts.to_string()
)


# ------------------------------------------------------------
# 21. SAVE
# ------------------------------------------------------------

master.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# 22. FINAL SUMMARY
# ------------------------------------------------------------

print()
print("=" * 70)
print("NDVI MASTER 2017-2025 CREATED SUCCESSFULLY")
print("=" * 70)

print()
print(
    "Output:"
)

print(
    OUTPUT_FILE
)

print()
print(
    f"Total records: "
    f"{len(master):,}"
)

print(
    f"Total columns: "
    f"{len(master.columns)}"
)

print(
    f"Consecutive NDVI pairs: "
    f"{consecutive_pairs:,}"
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)