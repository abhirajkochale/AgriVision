from pathlib import Path
import pandas as pd


# ============================================================
# AGRIVISION AI
# COMBINE 10 NDVI BATCHES
# ============================================================

# Main project folder
PROJECT_DIR = Path(r"C:\Projects\AgriVision")

# Your actual NDVI dataset folder
NDVI_DIR = PROJECT_DIR / "dataset" / "NDVI"

# Final master dataset
OUTPUT_FILE = PROJECT_DIR / "dataset" / "AgriVision_Maharashtra_NDVI_Master_2021_2025.csv"


# ============================================================
# 1. FIND FILES
# ============================================================

files = sorted(
    NDVI_DIR.glob("AgriVision_Maharashtra_NDVI_*.csv")
)

print("==============================================")
print("AGRIVISION - NDVI MASTER DATASET")
print("==============================================")

print("\nNDVI folder:")
print(NDVI_DIR)

print("\nFiles found:", len(files))

for file in files:
    print(" -", file.name)


# ============================================================
# 2. REQUIRE EXACTLY 10 BATCH FILES
# ============================================================

if len(files) != 10:
    raise ValueError(
        f"\nExpected 10 NDVI batch files, but found {len(files)}.\n"
        f"Check this folder:\n{NDVI_DIR}"
    )


# ============================================================
# 3. READ ALL FILES
# ============================================================

dataframes = []

for file in files:

    print(f"\nReading: {file.name}")

    df = pd.read_csv(file)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    dataframes.append(df)


# ============================================================
# 4. CHECK COLUMN CONSISTENCY
# ============================================================

expected_columns = list(dataframes[0].columns)

for file, df in zip(files, dataframes):

    if list(df.columns) != expected_columns:
        raise ValueError(
            f"\nColumn mismatch detected in:\n{file.name}"
        )

print("\nAll files have identical column structure ✅")


# ============================================================
# 5. COMBINE
# ============================================================

master = pd.concat(
    dataframes,
    ignore_index=True
)


# ============================================================
# 6. CONVERT DATE COLUMNS
# ============================================================

master["Week_Start"] = pd.to_datetime(
    master["Week_Start"],
    errors="raise"
)

master["Week_End"] = pd.to_datetime(
    master["Week_End"],
    errors="raise"
)


# ============================================================
# 7. SORT
# ============================================================

master = master.sort_values(
    by=[
        "Week_Start",
        "point_id"
    ]
).reset_index(drop=True)


# ============================================================
# 8. BASIC INFORMATION
# ============================================================

print("\n==============================================")
print("MASTER DATASET INFORMATION")
print("==============================================")

print("\nTotal rows:")
print(len(master))

print("\nMaximum theoretical rows:")
print(167 * 260)

print("\nTotal columns:")
print(len(master.columns))

print("\nUnique point IDs:")
print(master["point_id"].nunique())

print("\nUnique weeks:")
print(master["Week_Start"].nunique())

print("\nYears:")
print(sorted(master["Year"].unique()))


# ============================================================
# 9. DUPLICATE POINT-WEEK CHECK
# ============================================================

duplicate_point_week = master.duplicated(
    subset=[
        "point_id",
        "Week_Start"
    ]
).sum()

print("\nDuplicate point_id + Week_Start:")
print(duplicate_point_week)

if duplicate_point_week != 0:
    raise ValueError(
        "\nERROR: Duplicate point-week records found!"
    )


# ============================================================
# 10. FULL DUPLICATE CHECK
# ============================================================

duplicate_full_rows = master.duplicated().sum()

print("\nDuplicate complete rows:")
print(duplicate_full_rows)


# ============================================================
# 11. MISSING VALUE CHECK
# ============================================================

missing = master.isna().sum()

print("\nMissing values:")

if missing.sum() == 0:
    print("NONE ✅")
else:
    print(missing[missing > 0])


# ============================================================
# 12. NDVI RANGE CHECK
# ============================================================

ndvi_min = master["NDVI"].min()
ndvi_max = master["NDVI"].max()

invalid_ndvi = (
    (master["NDVI"] < -1) |
    (master["NDVI"] > 1)
).sum()

print("\nNDVI minimum:")
print(ndvi_min)

print("\nNDVI maximum:")
print(ndvi_max)

print("\nNDVI outside [-1, 1]:")
print(invalid_ndvi)


# ============================================================
# 13. NDVI VALID FLAG
# ============================================================

invalid_ndvi_flag = (
    master["NDVI_Valid"] != 1
).sum()

print("\nNDVI_Valid values other than 1:")
print(invalid_ndvi_flag)


# ============================================================
# 14. YEAR-WISE RECORD COUNTS
# ============================================================

print("\n==============================================")
print("RECORDS BY YEAR")
print("==============================================")

year_counts = (
    master
    .groupby("Year")
    .size()
    .sort_index()
)

print(year_counts)


# ============================================================
# 15. WEEK-WISE RECORD COUNTS
# ============================================================

print("\n==============================================")
print("RECORDS BY YEAR + WEEK")
print("==============================================")

weekly_counts = (
    master
    .groupby(
        [
            "Year",
            "Week_Number"
        ]
    )
    .size()
    .sort_index()
)

print(weekly_counts)


# ============================================================
# 16. POINT CONSISTENCY CHECK
# ============================================================

point_consistency = (
    master
    .groupby("point_id")
    .agg(
        unique_latitude=("latitude", "nunique"),
        unique_longitude=("longitude", "nunique"),
        unique_district=("district", "nunique")
    )
)

inconsistent_points = point_consistency[
    (point_consistency["unique_latitude"] > 1) |
    (point_consistency["unique_longitude"] > 1) |
    (point_consistency["unique_district"] > 1)
]

print("\n==============================================")
print("POINT CONSISTENCY")
print("==============================================")

print(
    "Points with inconsistent coordinates/district:",
    len(inconsistent_points)
)


# ============================================================
# 17. DATASET VALUE CHECK
# ============================================================

print("\nDataset values:")
print(master["dataset"].value_counts())

print("\nStudy area values:")
print(master["study_area"].value_counts())


# ============================================================
# 18. SAVE MASTER DATASET
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
# 19. FINAL RESULT
# ============================================================

print("\n==============================================")
print("MASTER NDVI DATASET CREATED ✅")
print("==============================================")

print("\nFinal rows:")
print(len(master))

print("\nFinal columns:")
print(len(master.columns))

print("\nSaved to:")
print(OUTPUT_FILE)

print("\n==============================================")
print("DONE")
print("==============================================")