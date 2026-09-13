from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List

import ee
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ID = "agrivision-505417"

BASE_DIR = Path(r"C:\Projects\AgriVision")

POINTS_CSV = (
    BASE_DIR
    / "dataset"
    / "Maharashtra_AgriVision_Sampling_Points.csv"
)

OUTPUT_DIR = BASE_DIR / "dataset" / "Rainfall_GSMaP"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Historical period
START_YEAR = 2017
END_YEAR = 2025

# Process this many weeks in one batch.
# Smaller values reduce server-side computation pressure.
WEEKS_PER_BATCH = 26

# Dataset
GSMAP_COLLECTION = "JAXA/GPM_L3/GSMaP/v8/operational"
GSMAP_BAND = "hourlyPrecipRateGC"

# Sampling scale.
# GSMaP is a satellite precipitation product with much finer
# native spatial resolution than CHIRPS.
SCALE_METERS = 10000


# ============================================================
# EARTH ENGINE INITIALIZATION
# ============================================================

def initialize_earth_engine() -> None:
    try:
        ee.Initialize(project=PROJECT_ID)
        print(f"Earth Engine initialized with project: {PROJECT_ID}")

    except Exception as exc:
        print("Earth Engine initialization failed.")
        print("Run:")
        print("    earthengine authenticate")
        print(f"    earthengine set_project {PROJECT_ID}")
        raise exc


# ============================================================
# POINT LOADING
# ============================================================

def load_points() -> pd.DataFrame:
    if not POINTS_CSV.exists():
        raise FileNotFoundError(
            f"Sampling point file not found:\n{POINTS_CSV}"
        )

    df = pd.read_csv(POINTS_CSV)

    print("\nDetected CSV columns:")
    print(df.columns.tolist())

    # --------------------------------------------------------
    # Normalize column names for matching
    # --------------------------------------------------------

    normalized = {}

    for col in df.columns:
        key = (
            str(col)
            .strip()
            .lower()
            .replace(" ", "")
            .replace("_", "")
            .replace("-", "")
        )
        normalized[key] = col

    # --------------------------------------------------------
    # Find point ID
    # --------------------------------------------------------

    point_id_candidates = [
        "pointid",
        "id",
        "sampleid",
    ]

    point_id_col = None

    for candidate in point_id_candidates:
        if candidate in normalized:
            point_id_col = normalized[candidate]
            break

    if point_id_col is None:
        raise ValueError(
            "Could not find the point ID column.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # --------------------------------------------------------
    # Find latitude
    # --------------------------------------------------------

    latitude_candidates = [
        "latitude",
        "lat",
        "y",
        "pointlatitude",
    ]

    latitude_col = None

    for candidate in latitude_candidates:
        if candidate in normalized:
            latitude_col = normalized[candidate]
            break

    # --------------------------------------------------------
    # Find longitude
    # --------------------------------------------------------

    longitude_candidates = [
        "longitude",
        "lon",
        "lng",
        "long",
        "x",
        "pointlongitude",
    ]

    longitude_col = None

    for candidate in longitude_candidates:
        if candidate in normalized:
            longitude_col = normalized[candidate]
            break

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if latitude_col is None or longitude_col is None:
        raise ValueError(
            "\nCould not identify coordinate columns.\n"
            f"Available columns: {df.columns.tolist()}\n"
            f"Detected point ID: {point_id_col}\n"
            f"Detected latitude: {latitude_col}\n"
            f"Detected longitude: {longitude_col}"
        )

    print("\nUsing columns:")
    print(f"  Point ID : {point_id_col}")
    print(f"  Latitude : {latitude_col}")
    print(f"  Longitude: {longitude_col}")

    # --------------------------------------------------------
    # Create canonical columns used by the rest of the script
    # --------------------------------------------------------

    df = df.rename(
        columns={
            point_id_col: "point_id",
            latitude_col: "Latitude",
            longitude_col: "Longitude",
        }
    )

    df = df[
        [
            "point_id",
            "Latitude",
            "Longitude",
        ]
    ].copy()

    df["point_id"] = df["point_id"].astype(str)

    df["Latitude"] = pd.to_numeric(
        df["Latitude"],
        errors="coerce",
    )

    df["Longitude"] = pd.to_numeric(
        df["Longitude"],
        errors="coerce",
    )

    # Remove invalid coordinates
    df = df.dropna(
        subset=[
            "point_id",
            "Latitude",
            "Longitude",
        ]
    ).reset_index(drop=True)

    # Basic geographic validation
    invalid_lat = (
        (df["Latitude"] < -90)
        | (df["Latitude"] > 90)
    )

    invalid_lon = (
        (df["Longitude"] < -180)
        | (df["Longitude"] > 180)
    )

    invalid = invalid_lat | invalid_lon

    if invalid.any():
        print(
            f"\nRemoving {invalid.sum()} rows "
            "with invalid coordinates."
        )

        df = df.loc[
            ~invalid
        ].reset_index(drop=True)

    # Duplicate point IDs
    if df["point_id"].duplicated().any():

        duplicates = (
            df.loc[
                df["point_id"].duplicated(),
                "point_id",
            ]
            .tolist()
        )

        raise ValueError(
            f"Duplicate point IDs detected: {duplicates}"
        )

    print(
        f"\nLoaded {len(df)} sampling points successfully."
    )

    print("\nFirst 5 points:")
    print(df.head().to_string(index=False))

    return df


# ============================================================
# EE FEATURE COLLECTION
# ============================================================

def dataframe_to_feature_collection(
    points_df: pd.DataFrame,
) -> ee.FeatureCollection:

    features = []

    for row in points_df.itertuples(index=False):
        geometry = ee.Geometry.Point(
            [
                float(row.Longitude),
                float(row.Latitude),
            ]
        )

        feature = ee.Feature(
            geometry,
            {
                "point_id": row.point_id,
            },
        )

        features.append(feature)

    return ee.FeatureCollection(features)


# ============================================================
# WEEK CREATION
# ============================================================

def generate_week_windows(
    start_year: int,
    end_year: int,
) -> List[dict]:
    """
    AgriVision weekly convention:

        Week 1 = Jan 1
        Week N = Jan 1 + (N-1)*7 days

    Each week's end is exclusive.

    Example:
        2017-01-01 -> 2017-01-08
    """

    windows = []

    for year in range(start_year, end_year + 1):

        year_start = pd.Timestamp(
            year=year,
            month=1,
            day=1,
        )

        # 53 possible week starts can appear with this convention
        # around year boundaries, but the project uses 52 weeks/year
        # consistently in the historical master.
        for week in range(1, 53):

            week_start = (
                year_start
                + pd.Timedelta(days=(week - 1) * 7)
            )

            week_end = week_start + pd.Timedelta(days=7)

            # Do not allow the nominal week to cross into the
            # following year.
            if week_start.year != year:
                continue

            windows.append(
                {
                    "Year": year,
                    "Week_Number": week,
                    "Week_Start": week_start.strftime(
                        "%Y-%m-%d"
                    ),
                    "Week_End": week_end.strftime(
                        "%Y-%m-%d"
                    ),
                }
            )

    return windows


# ============================================================
# GSMaP WEEKLY EXTRACTION
# ============================================================

def extract_week_batch(
    points_fc: ee.FeatureCollection,
    batch: List[dict],
) -> pd.DataFrame:

    rows = []

    collection = ee.ImageCollection(
        GSMAP_COLLECTION
    ).select(GSMAP_BAND)

    for week_info in batch:

        week_start = week_info["Week_Start"]
        week_end = week_info["Week_End"]

        start_date = ee.Date(week_start)
        end_date = ee.Date(week_end)

        weekly_collection = (
            collection
            .filterDate(
                start_date,
                end_date,
            )
        )

        # GSMaP band:
        # hourly precipitation rate.
        #
        # Weekly rainfall therefore needs temporal integration.
        #
        # Each image represents an hourly precipitation rate.
        # Convert to mm/hour * number of hours represented.
        #
        # Summing the hourly rates is the intended weekly
        # accumulated precipitation approximation for this
        # operational product.

        weekly_rainfall = (
            weekly_collection
            .sum()
        )

        # Reduce at all points in one server-side operation.
        sampled = weekly_rainfall.reduceRegions(
            collection=points_fc,
            reducer=ee.Reducer.mean(),
            scale=SCALE_METERS,
            tileScale=4,
        )

        sampled = sampled.map(
            lambda feature: feature.set(
                {
                    "Year": week_info["Year"],
                    "Week_Number": week_info["Week_Number"],
                    "Week_Start": week_info["Week_Start"],
                    "Week_End": week_info["Week_End"],
                    "GSMaP_Image_Count":
                        weekly_collection.size(),
                }
            )
        )

        result = sampled.getInfo()

        features = result.get("features", [])

        for feature in features:

            properties = feature.get(
                "properties",
                {},
            )

            rows.append(
                {
                    "point_id": properties.get(
                        "point_id"
                    ),
                    "Year": properties.get(
                        "Year"
                    ),
                    "Week_Number": properties.get(
                        "Week_Number"
                    ),
                    "Week_Start": properties.get(
                        "Week_Start"
                    ),
                    "Week_End": properties.get(
                        "Week_End"
                    ),
                    "Rainfall_GSMaP":
                        properties.get(
                            "mean"
                        ),
                    "GSMaP_Image_Count":
                        properties.get(
                            "GSMaP_Image_Count"
                        ),
                }
            )

        print(
            f"Processed "
            f"{week_start} -> {week_end} | "
            f"images={weekly_collection.size().getInfo()} | "
            f"points={len(features)}"
        )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    initialize_earth_engine()

    points_df = load_points()

    points_fc = dataframe_to_feature_collection(
        points_df
    )

    weeks = generate_week_windows(
        START_YEAR,
        END_YEAR,
    )

    print(
        f"Total week windows: {len(weeks)}"
    )

    batches = [
        weeks[i:i + WEEKS_PER_BATCH]
        for i in range(
            0,
            len(weeks),
            WEEKS_PER_BATCH,
        )
    ]

    print(
        f"Total batches: {len(batches)}"
    )

    all_parts = []

    for batch_index, batch in enumerate(
        batches,
        start=1,
    ):

        print()
        print("=" * 70)
        print(
            f"BATCH {batch_index}/{len(batches)}"
        )
        print(
            f"{batch[0]['Week_Start']} "
            f"-> "
            f"{batch[-1]['Week_End']}"
        )
        print("=" * 70)

        try:
            part = extract_week_batch(
                points_fc,
                batch,
            )

        except Exception as exc:

            print(
                f"Batch {batch_index} failed:"
            )
            print(exc)

            # Save successful batches before stopping.
            if all_parts:
                partial = pd.concat(
                    all_parts,
                    ignore_index=True,
                )

                partial_path = (
                    OUTPUT_DIR
                    / "AgriVision_GSMaP_Rainfall_Partial.csv"
                )

                partial.to_csv(
                    partial_path,
                    index=False,
                )

                print(
                    f"Partial results saved to:\n"
                    f"{partial_path}"
                )

            raise

        all_parts.append(part)

        # Save each batch immediately so a failed later
        # batch does not destroy previous progress.
        batch_path = (
            OUTPUT_DIR
            / f"GSMaP_batch_{batch_index:02d}.csv"
        )

        part.to_csv(
            batch_path,
            index=False,
        )

        print(
            f"Saved batch:\n{batch_path}"
        )

        # Small pause between server requests
        time.sleep(1)

    if not all_parts:
        raise RuntimeError(
            "No GSMaP results were produced."
        )

    result = pd.concat(
        all_parts,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # CLEAN TYPES
    # --------------------------------------------------------

    result["point_id"] = (
        result["point_id"]
        .astype(str)
    )

    result["Year"] = pd.to_numeric(
        result["Year"],
        errors="coerce",
    ).astype("Int64")

    result["Week_Number"] = pd.to_numeric(
        result["Week_Number"],
        errors="coerce",
    ).astype("Int64")

    result["Rainfall_GSMaP"] = pd.to_numeric(
        result["Rainfall_GSMaP"],
        errors="coerce",
    )

    result["GSMaP_Image_Count"] = pd.to_numeric(
        result["GSMaP_Image_Count"],
        errors="coerce",
    )

    result["Week_Start"] = pd.to_datetime(
        result["Week_Start"]
    )

    result["Week_End"] = pd.to_datetime(
        result["Week_End"]
    )

    # Remove accidental duplicates.
    result = (
        result
        .sort_values(
            [
                "point_id",
                "Week_Start",
            ]
        )
        .drop_duplicates(
            subset=[
                "point_id",
                "Week_Start",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    expected_rows = (
        len(points_df) * len(weeks)
    )

    print()
    print("=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)

    print(
        f"Expected rows: {expected_rows:,}"
    )

    print(
        f"Actual rows:   {len(result):,}"
    )

    print(
        f"Unique points: {result['point_id'].nunique()}"
    )

    print(
        f"Unique weeks:  {result['Week_Start'].nunique()}"
    )

    duplicates = result.duplicated(
        subset=[
            "point_id",
            "Week_Start",
        ]
    ).sum()

    print(
        f"Duplicate point-weeks: {duplicates}"
    )

    missing_rainfall = result[
        "Rainfall_GSMaP"
    ].isna().sum()

    print(
        f"Missing rainfall values: "
        f"{missing_rainfall:,}"
    )

    if missing_rainfall:
        print(
            "Missing rainfall values are retained "
            "as NaN rather than fabricated."
        )

    # --------------------------------------------------------
    # SAVE MASTER
    # --------------------------------------------------------

    master_path = (
        OUTPUT_DIR
        / "AgriVision_Maharashtra_GSMaP_Rainfall_Master_2017_2025.csv"
    )

    result.to_csv(
        master_path,
        index=False,
    )

    print()
    print(
        f"Master GSMaP rainfall dataset saved to:\n"
        f"{master_path}"
    )


if __name__ == "__main__":
    main()