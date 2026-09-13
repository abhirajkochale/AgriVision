from pathlib import Path
from datetime import datetime, timedelta, timezone

import ee
import pandas as pd


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ID = "agrivision-505417"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SAMPLING_POINTS_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "Maharashtra_AgriVision_Sampling_Points.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "realtime"
    / "data"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# REAL-TIME CONFIGURATION
# ============================================================

# Number of completed weekly windows to retrieve.
#
# 12 weeks are retained because the ML model requires:
#   NDVI_lag_8
#   Rainfall_lag_4
#   Temperature_lag_4
#   8-week rolling features
#
HISTORY_WEEKS = 12


# ------------------------------------------------------------
# Sentinel-2 settings
# ------------------------------------------------------------

SENTINEL_CLOUD_PROBABILITY = 65
SENTINEL_SCENE_CLOUD_PERCENT = 80

SENTINEL_DATASET = (
    "COPERNICUS/S2_SR_HARMONIZED"
)

SENTINEL_CLOUD_DATASET = (
    "COPERNICUS/S2_CLOUD_PROBABILITY"
)


# ------------------------------------------------------------
# GSMaP operational rainfall
# ------------------------------------------------------------
#
# Google Earth Engine catalog:
#
# JAXA/GPM_L3/GSMaP/v8/operational
#
# hourlyPrecipRateGC:
#   Gauge-adjusted hourly precipitation rate
#   Units = mm/hour
#
# Weekly rainfall:
#   Sum of hourly precipitation rates
#   over the completed 7-day window.
#
GSMAP_DATASET = (
    "JAXA/GPM_L3/GSMaP/v8/operational"
)

GSMAP_RAINFALL_BAND = (
    "hourlyPrecipRateGC"
)


# ------------------------------------------------------------
# ERA5-Land temperature
# ------------------------------------------------------------

ERA5_DATASET = (
    "ECMWF/ERA5_LAND/DAILY_AGGR"
)


# ============================================================
# EARTH ENGINE INITIALIZATION
# ============================================================

def initialize_earth_engine():
    """
    Initialize Google Earth Engine using the AgriVision project.
    """

    try:

        ee.Initialize(
            project=PROJECT_ID
        )

        print(
            "Earth Engine initialized successfully."
        )

    except Exception as exc:

        raise RuntimeError(
            "\nEarth Engine initialization failed.\n"
            "Make sure you have run:\n"
            "earthengine authenticate\n"
            "and:\n"
            "earthengine set_project agrivision-505417"
        ) from exc


# ============================================================
# LOAD CANONICAL TARGET POINTS
# ============================================================

def load_target_points():
    """
    Load the official AgriVision sampling-point CSV.

    Target locations:
      - P147 = Kavathe/Khanapur case-study point
      - all canonical Latur sampling points
    """

    if not SAMPLING_POINTS_FILE.exists():

        raise FileNotFoundError(
            f"\nSampling-point file not found:\n"
            f"{SAMPLING_POINTS_FILE}"
        )

    points = pd.read_csv(
        SAMPLING_POINTS_FILE
    )

    required_columns = [
        "point_id",
        "latitude",
        "longitude",
        "district",
        "sampling_priority",
        "region_type",
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in points.columns
    ]

    if missing_columns:

        raise ValueError(
            "\nSampling-point CSV is missing columns:\n"
            + "\n".join(
                missing_columns
            )
        )

    points["point_id"] = (
        points["point_id"]
        .astype(str)
        .str.strip()
    )

    points["district"] = (
        points["district"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Kavathe/Khanapur
    # --------------------------------------------------------

    kavathe = points[
        points["point_id"] == "P147"
    ].copy()

    if kavathe.empty:

        raise ValueError(
            "Canonical Kavathe/Khanapur point P147 "
            "was not found in the sampling-point CSV."
        )

    # --------------------------------------------------------
    # Latur
    # --------------------------------------------------------

    latur = points[
        points["district"]
        .str.lower()
        == "latur"
    ].copy()

    if latur.empty:

        raise ValueError(
            "No Latur points were found in the "
            "sampling-point CSV."
        )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    target_points = pd.concat(
        [
            kavathe,
            latur,
        ],
        ignore_index=True,
    )

    target_points = (
        target_points
        .drop_duplicates(
            subset=[
                "point_id"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"Loaded {len(target_points)} target points."
    )

    print(
        f"Kavathe/Khanapur: "
        f"{(
            target_points["point_id"]
            == "P147"
        ).sum()}"
    )

    print(
        f"Latur: "
        f"{(
            target_points["district"]
            .str.lower()
            == "latur"
        ).sum()}"
    )

    return target_points


# ============================================================
# WEEK FRAMEWORK
# ============================================================

def get_week_information(date_value):
    """
    AgriVision weekly convention:

    Week 1 starts on January 1.

    Week N starts:
        Jan 1 + (N - 1) * 7 days

    Week_End is exclusive.
    """

    jan1 = date_value.replace(
        month=1,
        day=1,
    )

    days_since_jan1 = (
        date_value - jan1
    ).days

    week_number = (
        days_since_jan1 // 7
    ) + 1

    week_start = (
        jan1
        + timedelta(
            days=(
                (week_number - 1)
                * 7
            )
        )
    )

    week_end = (
        week_start
        + timedelta(
            days=7
        )
    )

    return (
        week_number,
        week_start,
        week_end,
    )


def build_completed_week_windows():
    """
    Build the most recent completed weekly windows.

    The current incomplete week is never included.

    Example:
      If the current AgriVision week is
      2026-09-10 -> 2026-09-17,

      then the latest completed week is:
      2026-09-03 -> 2026-09-10.
    """

    today = (
        datetime.now(
            timezone.utc
        ).date()
    )

    (
        current_week_number,
        current_week_start,
        current_week_end,
    ) = get_week_information(
        today
    )

    # Current week is incomplete.
    # Therefore use the immediately preceding week.
    latest_completed_week_start = (
        current_week_start
        - timedelta(
            days=7
        )
    )

    windows = []

    for offset in range(
        HISTORY_WEEKS - 1,
        -1,
        -1,
    ):

        week_start = (
            latest_completed_week_start
            - timedelta(
                days=(
                    offset
                    * 7
                )
            )
        )

        week_end = (
            week_start
            + timedelta(
                days=7
            )
        )

        jan1 = week_start.replace(
            month=1,
            day=1,
        )

        week_number = (
            (
                week_start
                - jan1
            ).days
            // 7
        ) + 1

        windows.append(
            {
                "Year": week_start.year,
                "Week_Number": week_number,
                "Week_Start": week_start,
                "Week_End": week_end,
            }
        )

    return pd.DataFrame(
        windows
    )


# ============================================================
# EARTH ENGINE POINT COLLECTION
# ============================================================

def build_point_feature_collection(
    points_df,
):
    """
    Convert pandas sampling points into
    an Earth Engine FeatureCollection.
    """

    features = []

    for _, row in points_df.iterrows():

        point = ee.Geometry.Point(
            [
                float(
                    row["longitude"]
                ),
                float(
                    row["latitude"]
                ),
            ]
        )

        feature = ee.Feature(
            point,
            {
                "point_id": str(
                    row["point_id"]
                ),
            },
        )

        features.append(
            feature
        )

    return ee.FeatureCollection(
        features
    )


# ============================================================
# SENTINEL-2 CLOUD PROBABILITY JOIN
# ============================================================

def join_sentinel_cloud_probability(
    sentinel_collection,
    cloud_collection,
):
    """
    Match Sentinel-2 SR images with their
    cloud-probability images using system:index.
    """

    joined = ee.Join.saveFirst(
        "cloud_probability"
    ).apply(
        primary=sentinel_collection,
        secondary=cloud_collection,
        condition=ee.Filter.equals(
            leftField="system:index",
            rightField="system:index",
        ),
    )

    return ee.ImageCollection(
        joined
    )


# ============================================================
# SENTINEL-2 MASK
# ============================================================

def mask_sentinel2(
    image,
):
    """
    Apply the AgriVision historical Sentinel-2 masking logic:

      - cloud probability <= 65
      - remove SCL:
          3  = cloud shadow
          8  = medium-probability cloud
          9  = high-probability cloud
          10 = cirrus
          11 = snow/ice
    """

    cloud_probability = ee.Image(
        image.get(
            "cloud_probability"
        )
    ).select(
        "probability"
    )

    cloud_mask = (
        cloud_probability
        .lte(
            SENTINEL_CLOUD_PROBABILITY
        )
    )

    scl = image.select(
        "SCL"
    )

    scl_mask = (
        scl.neq(3)
        .And(
            scl.neq(8)
        )
        .And(
            scl.neq(9)
        )
        .And(
            scl.neq(10)
        )
        .And(
            scl.neq(11)
        )
    )

    return (
        image
        .updateMask(
            cloud_mask
            .And(
                scl_mask
            )
        )
    )


# ============================================================
# WEEKLY SENTINEL-2 NDVI
# ============================================================

def fetch_weekly_ndvi(
    points_df,
    week_windows,
):
    """
    Fetch weekly median NDVI.

    Important:
      - No interpolation.
      - No fabrication.
      - If a week has no valid Sentinel-2 imagery,
        that point-week is absent.
      - Empty weeks cannot crash the pipeline.
    """

    print(
        "\nFetching weekly Sentinel-2 NDVI..."
    )

    point_collection = (
        build_point_feature_collection(
            points_df
        )
    )

    rows = []

    for _, week in week_windows.iterrows():

        week_start = week[
            "Week_Start"
        ]

        week_end = week[
            "Week_End"
        ]

        start_string = (
            week_start.isoformat()
        )

        end_string = (
            week_end.isoformat()
        )

        # ----------------------------------------------------
        # Sentinel-2 SR
        # ----------------------------------------------------

        sentinel = (
            ee.ImageCollection(
                SENTINEL_DATASET
            )
            .filterDate(
                start_string,
                end_string,
            )
            .filterBounds(
                point_collection
            )
            .filter(
                ee.Filter.lte(
                    "CLOUDY_PIXEL_PERCENTAGE",
                    SENTINEL_SCENE_CLOUD_PERCENT,
                )
            )
        )

        sentinel_count = int(
            sentinel
            .size()
            .getInfo()
        )

        if sentinel_count == 0:

            print(
                f"  {week_start} "
                f"→ NO Sentinel-2 scenes, skipped"
            )

            continue

        # ----------------------------------------------------
        # Cloud probability
        # ----------------------------------------------------

        cloud_collection = (
            ee.ImageCollection(
                SENTINEL_CLOUD_DATASET
            )
            .filterDate(
                start_string,
                end_string,
            )
            .filterBounds(
                point_collection
            )
        )

        cloud_count = int(
            cloud_collection
            .size()
            .getInfo()
        )

        if cloud_count == 0:

            print(
                f"  {week_start} "
                f"→ NO cloud-probability scenes, skipped"
            )

            continue

        # ----------------------------------------------------
        # Join
        # ----------------------------------------------------

        joined = (
            join_sentinel_cloud_probability(
                sentinel,
                cloud_collection,
            )
        )

        joined_count = int(
            joined
            .size()
            .getInfo()
        )

        if joined_count == 0:

            print(
                f"  {week_start} "
                f"→ NO matched scenes, skipped"
            )

            continue

        # ----------------------------------------------------
        # Mask + NDVI
        # ----------------------------------------------------

        processed = (
            joined
            .map(
                mask_sentinel2
            )
            .map(
                lambda image:
                image.addBands(
                    image
                    .normalizedDifference(
                        [
                            "B8",
                            "B4",
                        ]
                    )
                    .rename(
                        "NDVI"
                    )
                )
            )
        )

        processed_count = int(
            processed
            .size()
            .getInfo()
        )

        if processed_count == 0:

            print(
                f"  {week_start} "
                f"→ NO usable images after masking, skipped"
            )

            continue

        # ----------------------------------------------------
        # Weekly median
        # ----------------------------------------------------

        weekly_ndvi = (
            processed
            .select(
                "NDVI"
            )
            .median()
        )

        band_names = (
            weekly_ndvi
            .bandNames()
            .getInfo()
        )

        if not band_names:

            print(
                f"  {week_start} "
                f"→ weekly image has no bands, skipped"
            )

            continue

        # ----------------------------------------------------
        # Sample points
        # ----------------------------------------------------

        sampled = (
            weekly_ndvi
            .reduceRegions(
                collection=point_collection,
                reducer=ee.Reducer.median(),
                scale=10,
            )
            .filter(
                ee.Filter.notNull(
                    [
                        "median"
                    ]
                )
            )
        )

        result = (
            sampled
            .getInfo()
        )

        features = result.get(
            "features",
            [],
        )

        feature_count = len(
            features
        )

        if feature_count == 0:

            print(
                f"  {week_start} "
                f"→ 0 valid NDVI points"
            )

            continue

        # ----------------------------------------------------
        # Save rows
        # ----------------------------------------------------

        for feature in features:

            props = feature.get(
                "properties",
                {},
            )

            ndvi_value = props.get(
                "median"
            )

            point_id = props.get(
                "point_id"
            )

            if (
                ndvi_value is None
                or point_id is None
            ):
                continue

            rows.append(
                {
                    "Week_Start": week_start,
                    "Week_End": week_end,
                    "Week_Number": int(
                        week[
                            "Week_Number"
                        ]
                    ),
                    "Year": int(
                        week[
                            "Year"
                        ]
                    ),
                    "point_id": str(
                        point_id
                    ),
                    "NDVI": float(
                        ndvi_value
                    ),
                }
            )

        print(
            f"  {week_start} "
            f"→ {feature_count} points"
        )

    columns = [
        "Week_Start",
        "Week_End",
        "Week_Number",
        "Year",
        "point_id",
        "NDVI",
    ]

    if not rows:

        return pd.DataFrame(
            columns=columns
        )

    return pd.DataFrame(
        rows
    )[
        columns
    ]


# ============================================================
# WEEKLY GSMaP OPERATIONAL RAINFALL
# ============================================================

def fetch_weekly_rainfall(
    points_df,
    week_windows,
):
    """
    Fetch weekly GSMaP v8 operational rainfall.

    Dataset:
        JAXA/GPM_L3/GSMaP/v8/operational

    Band:
        hourlyPrecipRateGC

    Resolution:
        approximately 0.1 degree

    Input unit:
        mm/hour

    Aggregation:
        Sum across the 168 hourly observations
        belonging to the completed 7-day week.

    No interpolation or fabrication is performed.
    """

    print(
        "\nFetching weekly GSMaP operational rainfall..."
    )

    print(
        f"Dataset: {GSMAP_DATASET}"
    )

    print(
        f"Band: {GSMAP_RAINFALL_BAND}"
    )

    point_collection = (
        build_point_feature_collection(
            points_df
        )
    )

    rows = []

    expected_hours = 7 * 24

    for _, week in week_windows.iterrows():

        week_start = week[
            "Week_Start"
        ]

        week_end = week[
            "Week_End"
        ]

        start_string = (
            week_start.isoformat()
        )

        end_string = (
            week_end.isoformat()
        )

        # ----------------------------------------------------
        # GSMaP operational collection
        # ----------------------------------------------------

        collection = (
            ee.ImageCollection(
                GSMAP_DATASET
            )
            .filterDate(
                start_string,
                end_string,
            )
            .filterBounds(
                point_collection
            )
        )

        collection_count = int(
            collection
            .size()
            .getInfo()
        )

        print(
            f"  {week_start} "
            f"→ {collection_count} GSMaP hourly images"
        )

        # ----------------------------------------------------
        # No data
        # ----------------------------------------------------

        if collection_count == 0:

            print(
                f"  {week_start} "
                f"→ NO GSMaP observations, skipped"
            )

            continue

        # ----------------------------------------------------
        # Completeness warning
        #
        # A complete 7-day hourly window should normally
        # contain 168 hourly observations.
        #
        # We DO NOT fabricate missing hours.
        # ----------------------------------------------------

        if collection_count != expected_hours:

            print(
                f"  WARNING: expected "
                f"{expected_hours} hourly observations, "
                f"found {collection_count}."
            )

        # ----------------------------------------------------
        # Sum hourly rainfall
        # ----------------------------------------------------

        weekly_rainfall = (
            collection
            .select(
                GSMAP_RAINFALL_BAND
            )
            .sum()
            .rename(
                "Rainfall_mm"
            )
        )

        band_names = (
            weekly_rainfall
            .bandNames()
            .getInfo()
        )

        if not band_names:

            print(
                f"  {week_start} "
                f"→ rainfall image has no bands, skipped"
            )

            continue

        # ----------------------------------------------------
        # Sample target points
        # ----------------------------------------------------

        sampled = (
            weekly_rainfall
            .reduceRegions(
                collection=point_collection,
                reducer=ee.Reducer.mean(),
                scale=11132,
            )
            .filter(
                ee.Filter.notNull(
                    [
                        "mean"
                    ]
                )
            )
        )

        result = (
            sampled
            .getInfo()
        )

        features = result.get(
            "features",
            [],
        )

        feature_count = len(
            features
        )

        if feature_count == 0:

            print(
                f"  {week_start} "
                f"→ 0 valid GSMaP rainfall points"
            )

            continue

        # ----------------------------------------------------
        # Save rows
        # ----------------------------------------------------

        for feature in features:

            props = feature.get(
                "properties",
                {},
            )

            rainfall = props.get(
                "mean"
            )

            point_id = props.get(
                "point_id"
            )

            if (
                rainfall is None
                or point_id is None
            ):
                continue

            # Safety check:
            # rainfall cannot be physically negative.
            rainfall_value = max(
                0.0,
                float(
                    rainfall
                )
            )

            rows.append(
                {
                    "Week_Start": week_start,
                    "Week_End": week_end,
                    "Week_Number": int(
                        week[
                            "Week_Number"
                        ]
                    ),
                    "Year": int(
                        week[
                            "Year"
                        ]
                    ),
                    "point_id": str(
                        point_id
                    ),
                    "Rainfall_mm": rainfall_value,
                }
            )

        print(
            f"  {week_start} "
            f"→ {feature_count} points"
        )

    columns = [
        "Week_Start",
        "Week_End",
        "Week_Number",
        "Year",
        "point_id",
        "Rainfall_mm",
    ]

    if not rows:

        return pd.DataFrame(
            columns=columns
        )

    rainfall_df = pd.DataFrame(
        rows
    )[
        columns
    ]

    # --------------------------------------------------------
    # GSMaP validation
    # --------------------------------------------------------

    negative_count = int(
        (
            rainfall_df[
                "Rainfall_mm"
            ]
            < 0
        ).sum()
    )

    if negative_count > 0:

        raise ValueError(
            f"GSMaP rainfall contains "
            f"{negative_count} negative values."
        )

    return rainfall_df


# ============================================================
# WEEKLY ERA5-LAND TEMPERATURE
# ============================================================

def fetch_weekly_temperature(
    points_df,
    week_windows,
):
    """
    Fetch ERA5-Land daily 2m temperature.

    Conversion:
      Kelvin -> Celsius

    Aggregation:
      Mean across the completed 7-day week.
    """

    print(
        "\nFetching weekly ERA5-Land temperature..."
    )

    point_collection = (
        build_point_feature_collection(
            points_df
        )
    )

    rows = []

    for _, week in week_windows.iterrows():

        week_start = week[
            "Week_Start"
        ]

        week_end = week[
            "Week_End"
        ]

        collection = (
            ee.ImageCollection(
                ERA5_DATASET
            )
            .filterDate(
                week_start.isoformat(),
                week_end.isoformat(),
            )
            .filterBounds(
                point_collection
            )
        )

        collection_count = int(
            collection
            .size()
            .getInfo()
        )

        if collection_count == 0:

            print(
                f"  {week_start} "
                f"→ NO ERA5 observations, skipped"
            )

            continue

        temperature_celsius = (
            collection
            .select(
                "temperature_2m"
            )
            .mean()
            .subtract(
                273.15
            )
            .rename(
                "Temperature_C"
            )
        )

        band_names = (
            temperature_celsius
            .bandNames()
            .getInfo()
        )

        if not band_names:

            print(
                f"  {week_start} "
                f"→ temperature image has no bands, skipped"
            )

            continue

        sampled = (
            temperature_celsius
            .reduceRegions(
                collection=point_collection,
                reducer=ee.Reducer.mean(),
                scale=11132,
            )
            .filter(
                ee.Filter.notNull(
                    [
                        "mean"
                    ]
                )
            )
        )

        result = (
            sampled
            .getInfo()
        )

        features = result.get(
            "features",
            [],
        )

        feature_count = len(
            features
        )

        if feature_count == 0:

            print(
                f"  {week_start} "
                f"→ 0 temperature points"
            )

            continue

        for feature in features:

            props = feature.get(
                "properties",
                {},
            )

            temperature = props.get(
                "mean"
            )

            point_id = props.get(
                "point_id"
            )

            if (
                temperature is None
                or point_id is None
            ):
                continue

            rows.append(
                {
                    "Week_Start": week_start,
                    "Week_End": week_end,
                    "Week_Number": int(
                        week[
                            "Week_Number"
                        ]
                    ),
                    "Year": int(
                        week[
                            "Year"
                        ]
                    ),
                    "point_id": str(
                        point_id
                    ),
                    "Temperature_C": float(
                        temperature
                    ),
                }
            )

        print(
            f"  {week_start} "
            f"→ {feature_count} points"
        )

    columns = [
        "Week_Start",
        "Week_End",
        "Week_Number",
        "Year",
        "point_id",
        "Temperature_C",
    ]

    if not rows:

        return pd.DataFrame(
            columns=columns
        )

    return pd.DataFrame(
        rows
    )[
        columns
    ]


# ============================================================
# MERGE DATA SOURCES
# ============================================================

def merge_realtime_data(
    points_df,
    ndvi_df,
    rainfall_df,
    temperature_df,
):
    """
    Merge NDVI, GSMaP rainfall and temperature using:

        point_id + Week_Start

    Missing source observations remain NaN.
    No imputation/fabrication is performed here.
    """

    metadata_columns = [
        "point_id",
        "latitude",
        "longitude",
        "district",
        "sampling_priority",
        "region_type",
    ]

    metadata = points_df[
        metadata_columns
    ].copy()

    merged = (
        ndvi_df
        .merge(
            rainfall_df[
                [
                    "Week_Start",
                    "Week_End",
                    "Week_Number",
                    "Year",
                    "point_id",
                    "Rainfall_mm",
                ]
            ],
            on=[
                "Week_Start",
                "Week_End",
                "Week_Number",
                "Year",
                "point_id",
            ],
            how="outer",
        )
        .merge(
            temperature_df[
                [
                    "Week_Start",
                    "Week_End",
                    "Week_Number",
                    "Year",
                    "point_id",
                    "Temperature_C",
                ]
            ],
            on=[
                "Week_Start",
                "Week_End",
                "Week_Number",
                "Year",
                "point_id",
            ],
            how="outer",
        )
        .merge(
            metadata,
            on="point_id",
            how="left",
        )
    )

    ordered_columns = [
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
        "Rainfall_mm",
        "Temperature_C",
    ]

    existing_columns = [
        column
        for column in ordered_columns
        if column in merged.columns
    ]

    merged = merged[
        existing_columns
    ].copy()

    merged = (
        merged
        .sort_values(
            [
                "point_id",
                "Week_Start",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return merged


# ============================================================
# VALIDATION
# ============================================================

def validate_realtime_data(
    data,
    expected_points,
    expected_weeks,
    rainfall_df,
):
    """
    Structural and source validation.

    Missing NDVI is permitted because cloudy weeks can
    legitimately contain no valid Sentinel-2 observation.

    Rainfall is expected to be available for all target
    point-weeks whenever GSMaP covers the requested week.
    """

    print(
        "\n"
        + "=" * 80
    )

    print(
        "REAL-TIME DATA VALIDATION"
    )

    print(
        "=" * 80
    )

    print(
        f"Rows: {len(data):,}"
    )

    print(
        f"Unique points: "
        f"{data['point_id'].nunique()}"
    )

    print(
        f"Unique weeks: "
        f"{data['Week_Start'].nunique()}"
    )

    duplicate_count = (
        data
        .duplicated(
            subset=[
                "point_id",
                "Week_Start",
            ]
        )
        .sum()
    )

    print(
        f"Duplicate point-weeks: "
        f"{duplicate_count}"
    )

    if "NDVI" in data.columns:

        print(
            f"Missing NDVI: "
            f"{data['NDVI'].isna().sum()}"
        )

    if "Rainfall_mm" in data.columns:

        rainfall_missing = int(
            data[
                "Rainfall_mm"
            ]
            .isna()
            .sum()
        )

        print(
            f"Missing GSMaP rainfall: "
            f"{rainfall_missing}"
        )

        rainfall_valid = (
            data[
                "Rainfall_mm"
            ]
            .notna()
            .sum()
        )

        print(
            f"Valid GSMaP rainfall rows: "
            f"{rainfall_valid:,}"
        )

    if "Temperature_C" in data.columns:

        print(
            f"Missing temperature: "
            f"{data['Temperature_C'].isna().sum()}"
        )

    print(
        f"Expected target points: "
        f"{expected_points}"
    )

    print(
        f"Expected completed weeks: "
        f"{expected_weeks}"
    )

    # --------------------------------------------------------
    # Structural validation
    # --------------------------------------------------------

    if duplicate_count > 0:

        raise ValueError(
            "Duplicate point-week records found."
        )

    if (
        data[
            "point_id"
        ]
        .nunique()
        != expected_points
    ):

        print(
            "\nWARNING:"
            "\nNot all target points appear "
            "in the merged data."
        )

    if (
        data[
            "Week_Start"
        ]
        .nunique()
        != expected_weeks
    ):

        print(
            "\nWARNING:"
            "\nNot all requested weeks appear "
            "in the merged data."
        )

    # --------------------------------------------------------
    # Rainfall coverage validation
    # --------------------------------------------------------

    rainfall_expected_rows = (
        expected_points
        * expected_weeks
    )

    rainfall_actual_rows = len(
        rainfall_df
    )

    print(
        "\nExpected GSMaP point-weeks: "
        f"{rainfall_expected_rows:,}"
    )

    print(
        "Actual GSMaP point-weeks: "
        f"{rainfall_actual_rows:,}"
    )

    rainfall_duplicates = (
        rainfall_df
        .duplicated(
            subset=[
                "point_id",
                "Week_Start",
            ]
        )
        .sum()
    )

    print(
        "GSMaP duplicate point-weeks: "
        f"{rainfall_duplicates}"
    )

    if rainfall_duplicates > 0:

        raise ValueError(
            "Duplicate GSMaP point-week records found."
        )


# ============================================================
# SAVE SOURCE-SPECIFIC OUTPUTS
# ============================================================

def save_source_outputs(
    ndvi_df,
    rainfall_df,
    temperature_df,
):
    """
    Save the source-specific realtime datasets.
    """

    ndvi_path = (
        OUTPUT_DIR
        / "realtime_ndvi.csv"
    )

    rainfall_path = (
        OUTPUT_DIR
        / "realtime_rainfall.csv"
    )

    temperature_path = (
        OUTPUT_DIR
        / "realtime_temperature.csv"
    )

    ndvi_df.to_csv(
        ndvi_path,
        index=False,
    )

    rainfall_df.to_csv(
        rainfall_path,
        index=False,
    )

    temperature_df.to_csv(
        temperature_path,
        index=False,
    )

    print(
        "\nSource-specific files saved:"
    )

    print(
        ndvi_path
    )

    print(
        rainfall_path
    )

    print(
        temperature_path
    )


# ============================================================
# SAVE FINAL MERGED OUTPUT
# ============================================================

def save_final_output(
    realtime_df,
):
    """
    Save final operational realtime dataset.
    """

    output_file = (
        OUTPUT_DIR
        / "AgriVision_Realtime_Weekly_Kavathe_Latur.csv"
    )

    realtime_df.to_csv(
        output_file,
        index=False,
    )

    return output_file


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "AGRIVISION AI - REAL-TIME WEEKLY DATA FETCH"
    )

    print(
        "2026 OPERATIONAL PIPELINE"
    )

    print(
        "=" * 80
    )

    print(
        "\nRainfall source:"
    )

    print(
        f"  GSMaP v8 Operational"
    )

    print(
        f"  {GSMAP_DATASET}"
    )

    print(
        f"  Band: {GSMAP_RAINFALL_BAND}"
    )

    # --------------------------------------------------------
    # Initialize Earth Engine
    # --------------------------------------------------------

    initialize_earth_engine()

    # --------------------------------------------------------
    # Load official points
    # --------------------------------------------------------

    points_df = (
        load_target_points()
    )

    # --------------------------------------------------------
    # Build completed weekly windows
    # --------------------------------------------------------

    week_windows = (
        build_completed_week_windows()
    )

    print(
        "\nWeekly windows:"
    )

    print(
        week_windows[
            [
                "Year",
                "Week_Number",
                "Week_Start",
                "Week_End",
            ]
        ]
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # NDVI
    # --------------------------------------------------------

    ndvi_df = (
        fetch_weekly_ndvi(
            points_df,
            week_windows,
        )
    )

    # --------------------------------------------------------
    # GSMaP rainfall
    # --------------------------------------------------------

    rainfall_df = (
        fetch_weekly_rainfall(
            points_df,
            week_windows,
        )
    )

    # --------------------------------------------------------
    # Temperature
    # --------------------------------------------------------

    temperature_df = (
        fetch_weekly_temperature(
            points_df,
            week_windows,
        )
    )

    # --------------------------------------------------------
    # Save individual sources
    # --------------------------------------------------------

    save_source_outputs(
        ndvi_df,
        rainfall_df,
        temperature_df,
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    realtime_df = (
        merge_realtime_data(
            points_df,
            ndvi_df,
            rainfall_df,
            temperature_df,
        )
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validate_realtime_data(
        realtime_df,
        expected_points=len(
            points_df
        ),
        expected_weeks=len(
            week_windows
        ),
        rainfall_df=rainfall_df,
    )

    # --------------------------------------------------------
    # Save final merged output
    # --------------------------------------------------------

    output_file = (
        save_final_output(
            realtime_df
        )
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "REAL-TIME DATA FETCH COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        "\nFinal merged file:"
    )

    print(
        output_file
    )

    print(
        "\nFinal shape:"
    )

    print(
        realtime_df.shape
    )

    # --------------------------------------------------------
    # Coverage summaries
    # --------------------------------------------------------

    print(
        "\nNDVI coverage:"
    )

    print(
        f"{ndvi_df['point_id'].nunique() if not ndvi_df.empty else 0} "
        f"points with NDVI observations"
    )

    print(
        f"{ndvi_df['Week_Start'].nunique() if not ndvi_df.empty else 0} "
        f"weeks with NDVI observations"
    )

    print(
        "\nGSMaP rainfall coverage:"
    )

    print(
        f"{rainfall_df['point_id'].nunique() if not rainfall_df.empty else 0} "
        f"points with rainfall observations"
    )

    print(
        f"{rainfall_df['Week_Start'].nunique() if not rainfall_df.empty else 0} "
        f"weeks with rainfall observations"
    )

    print(
        f"{len(rainfall_df):,} "
        f"point-week rainfall observations"
    )

    if not rainfall_df.empty:

        print(
            "\nGSMaP rainfall statistics:"
        )

        print(
            f"Minimum: "
            f"{rainfall_df['Rainfall_mm'].min():.4f} mm"
        )

        print(
            f"Maximum: "
            f"{rainfall_df['Rainfall_mm'].max():.4f} mm"
        )

        print(
            f"Mean: "
            f"{rainfall_df['Rainfall_mm'].mean():.4f} mm"
        )

        print(
            f"Median: "
            f"{rainfall_df['Rainfall_mm'].median():.4f} mm"
        )

    print(
        "\nTemperature coverage:"
    )

    print(
        f"{temperature_df['point_id'].nunique() if not temperature_df.empty else 0} "
        f"points with temperature observations"
    )

    print(
        f"{temperature_df['Week_Start'].nunique() if not temperature_df.empty else 0} "
        f"weeks with temperature observations"
    )

    print(
        "\nNo NDVI values were fabricated or interpolated."
    )

    print(
        "No rainfall values were fabricated or interpolated."
    )

    print(
        "No temperature values were fabricated or interpolated."
    )

    print(
        "\nOperational rainfall source: GSMaP v8."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()