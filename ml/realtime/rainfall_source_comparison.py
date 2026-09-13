from pathlib import Path
from datetime import date, timedelta

import ee
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
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
    / "rainfall_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# Data sources
# ------------------------------------------------------------

CHIRPS_DATASET = (
    "UCSB-CHG/CHIRPS/DAILY"
)

GSMAP_DATASET = (
    "JAXA/GPM_L3/GSMaP/v8/operational"
)

GSMAP_BAND = (
    "hourlyPrecipRateGC"
)


# ------------------------------------------------------------
# Comparison period
# ------------------------------------------------------------
#
# We deliberately use the recent period where CHIRPS has
# overlap with GSMaP.
#
# This is NOT yet the model backtest.
# It is the first source-comparability test.
#

START_DATE = date(
    2026,
    6,
    18,
)

END_DATE = date(
    2026,
    8,
    1,
)


# ============================================================
# EARTH ENGINE
# ============================================================

def initialize_earth_engine():

    try:

        ee.Initialize(
            project=PROJECT_ID
        )

        print(
            "Earth Engine initialized successfully."
        )

    except Exception as exc:

        raise RuntimeError(
            "Earth Engine initialization failed."
        ) from exc


# ============================================================
# LOAD TARGET POINTS
# ============================================================

def load_target_points():

    if not SAMPLING_POINTS_FILE.exists():

        raise FileNotFoundError(
            f"Sampling-point file not found:\n"
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
    ]

    missing = [
        column
        for column in required_columns
        if column not in points.columns
    ]

    if missing:

        raise ValueError(
            "Missing columns:\n"
            + "\n".join(missing)
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

    kavathe = points[
        points["point_id"]
        == "P147"
    ].copy()

    latur = points[
        points["district"]
        .str.lower()
        == "latur"
    ].copy()

    target = pd.concat(
        [
            kavathe,
            latur,
        ],
        ignore_index=True,
    )

    target = (
        target
        .drop_duplicates(
            subset=["point_id"]
        )
        .reset_index(drop=True)
    )

    print(
        f"Target points: {len(target)}"
    )

    return target


# ============================================================
# POINT COLLECTION
# ============================================================

def build_point_collection(
    points_df,
):

    features = []

    for _, row in points_df.iterrows():

        geometry = ee.Geometry.Point(
            [
                float(
                    row["longitude"]
                ),
                float(
                    row["latitude"]
                ),
            ]
        )

        features.append(
            ee.Feature(
                geometry,
                {
                    "point_id": str(
                        row["point_id"]
                    ),
                },
            )
        )

    return ee.FeatureCollection(
        features
    )


# ============================================================
# WEEK WINDOWS
# ============================================================

def build_week_windows():

    windows = []

    current = START_DATE

    while current < END_DATE:

        week_end = (
            current
            + timedelta(
                days=7
            )
        )

        if week_end > END_DATE:
            week_end = END_DATE

        jan1 = current.replace(
            month=1,
            day=1,
        )

        week_number = (
            (
                current
                - jan1
            ).days
            // 7
        ) + 1

        windows.append(
            {
                "Year": current.year,
                "Week_Number": week_number,
                "Week_Start": current,
                "Week_End": week_end,
            }
        )

        current = week_end

    return pd.DataFrame(
        windows
    )


# ============================================================
# CHIRPS
# ============================================================

def fetch_chirps(
    points_df,
    weeks_df,
):

    print(
        "\nFetching CHIRPS weekly rainfall..."
    )

    points = build_point_collection(
        points_df
    )

    rows = []

    for _, week in weeks_df.iterrows():

        start = week[
            "Week_Start"
        ]

        end = week[
            "Week_End"
        ]

        collection = (
            ee.ImageCollection(
                CHIRPS_DATASET
            )
            .filterDate(
                start.isoformat(),
                end.isoformat(),
            )
            .filterBounds(
                points
            )
        )

        count = int(
            collection
            .size()
            .getInfo()
        )

        if count == 0:

            print(
                f"  {start} → no CHIRPS"
            )

            continue

        weekly = (
            collection
            .select(
                "precipitation"
            )
            .sum()
        )

        sampled = (
            weekly
            .reduceRegions(
                collection=points,
                reducer=ee.Reducer.mean(),
                scale=5566,
            )
            .filter(
                ee.Filter.notNull(
                    ["mean"]
                )
            )
        )

        info = sampled.getInfo()

        features = info.get(
            "features",
            [],
        )

        print(
            f"  {start} → "
            f"{len(features)} points"
        )

        for feature in features:

            props = feature[
                "properties"
            ]

            rows.append(
                {
                    "Week_Start": start,
                    "Week_End": end,
                    "Week_Number": int(
                        week[
                            "Week_Number"
                        ]
                    ),
                    "Year": int(
                        week["Year"]
                    ),
                    "point_id": props.get(
                        "point_id"
                    ),
                    "CHIRPS_mm": props.get(
                        "mean"
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# GSMAP
# ============================================================

def fetch_gsmap(
    points_df,
    weeks_df,
):

    print(
        "\nFetching GSMaP weekly rainfall..."
    )

    points = build_point_collection(
        points_df
    )

    rows = []

    for _, week in weeks_df.iterrows():

        start = week[
            "Week_Start"
        ]

        end = week[
            "Week_End"
        ]

        collection = (
            ee.ImageCollection(
                GSMAP_DATASET
            )
            .filterDate(
                start.isoformat(),
                end.isoformat(),
            )
            .filterBounds(
                points
            )
        )

        count = int(
            collection
            .size()
            .getInfo()
        )

        if count == 0:

            print(
                f"  {start} → no GSMaP"
            )

            continue

        weekly = (
            collection
            .select(
                GSMAP_BAND
            )
            .sum()
            .divide(
                1.0
            )
        )

        sampled = (
            weekly
            .reduceRegions(
                collection=points,
                reducer=ee.Reducer.mean(),
                scale=11132,
            )
            .filter(
                ee.Filter.notNull(
                    ["mean"]
                )
            )
        )

        info = sampled.getInfo()

        features = info.get(
            "features",
            [],
        )

        print(
            f"  {start} → "
            f"{len(features)} points"
        )

        for feature in features:

            props = feature[
                "properties"
            ]

            rows.append(
                {
                    "Week_Start": start,
                    "Week_End": end,
                    "Week_Number": int(
                        week[
                            "Week_Number"
                        ]
                    ),
                    "Year": int(
                        week["Year"]
                    ),
                    "point_id": props.get(
                        "point_id"
                    ),
                    "GSMaP_mm": props.get(
                        "mean"
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MERGE
# ============================================================

def merge_sources(
    chirps_df,
    gsmap_df,
):

    merged = pd.merge(
        chirps_df,
        gsmap_df,
        on=[
            "Week_Start",
            "Week_End",
            "Week_Number",
            "Year",
            "point_id",
        ],
        how="outer",
    )

    return merged.sort_values(
        [
            "point_id",
            "Week_Start",
        ]
    ).reset_index(
        drop=True
    )


# ============================================================
# STATISTICS
# ============================================================

def calculate_statistics(
    merged,
):

    comparable = merged[
        merged["CHIRPS_mm"].notna()
        & merged["GSMaP_mm"].notna()
    ].copy()

    if comparable.empty:

        return (
            comparable,
            pd.DataFrame(),
            pd.DataFrame(),
        )

    comparable[
        "Difference_mm"
    ] = (
        comparable["GSMaP_mm"]
        - comparable["CHIRPS_mm"]
    )

    comparable[
        "Absolute_Difference_mm"
    ] = np.abs(
        comparable[
            "Difference_mm"
        ]
    )

    comparable[
        "Relative_Difference_Percent"
    ] = np.where(
        comparable["CHIRPS_mm"] > 0,
        (
            comparable["Difference_mm"]
            / comparable["CHIRPS_mm"]
        ) * 100,
        np.nan,
    )

    # --------------------------------------------------------
    # Overall statistics
    # --------------------------------------------------------

    x = comparable[
        "CHIRPS_mm"
    ].values

    y = comparable[
        "GSMaP_mm"
    ].values

    correlation = np.corrcoef(
        x,
        y,
    )[0, 1]

    mae = np.mean(
        np.abs(
            y - x
        )
    )

    rmse = np.sqrt(
        np.mean(
            (
                y - x
            ) ** 2
        )
    )

    bias = np.mean(
        y - x
    )

    overall = pd.DataFrame(
        [
            {
                "Comparable_Records": len(
                    comparable
                ),
                "Pearson_Correlation": correlation,
                "MAE_mm": mae,
                "RMSE_mm": rmse,
                "Mean_Bias_GSMaP_minus_CHIRPS_mm": bias,
                "CHIRPS_Mean_mm": np.mean(x),
                "GSMaP_Mean_mm": np.mean(y),
                "CHIRPS_Median_mm": np.median(x),
                "GSMaP_Median_mm": np.median(y),
            }
        ]
    )

    # --------------------------------------------------------
    # Point-level statistics
    # --------------------------------------------------------

    point_rows = []

    for point_id, group in comparable.groupby(
        "point_id"
    ):

        x_point = group[
            "CHIRPS_mm"
        ].values

        y_point = group[
            "GSMaP_mm"
        ].values

        if len(x_point) < 2:
            continue

        correlation_point = np.corrcoef(
            x_point,
            y_point,
        )[0, 1]

        mae_point = np.mean(
            np.abs(
                y_point
                - x_point
            )
        )

        rmse_point = np.sqrt(
            np.mean(
                (
                    y_point
                    - x_point
                ) ** 2
            )
        )

        bias_point = np.mean(
            y_point
            - x_point
        )

        point_rows.append(
            {
                "point_id": point_id,
                "Records": len(
                    group
                ),
                "Correlation": correlation_point,
                "MAE_mm": mae_point,
                "RMSE_mm": rmse_point,
                "Bias_mm": bias_point,
            }
        )

    point_stats = pd.DataFrame(
        point_rows
    )

    return (
        comparable,
        overall,
        point_stats,
    )


# ============================================================
# INTERPRETATION
# ============================================================

def interpret(
    overall,
):

    if overall.empty:

        return (
            "No overlapping CHIRPS/GSMaP observations "
            "were available."
        )

    correlation = float(
        overall.iloc[0][
            "Pearson_Correlation"
        ]
    )

    rmse = float(
        overall.iloc[0][
            "RMSE_mm"
        ]
    )

    bias = float(
        overall.iloc[0][
            "Mean_Bias_GSMaP_minus_CHIRPS_mm"
        ]
    )

    if correlation >= 0.80:

        correlation_text = (
            "strong"
        )

    elif correlation >= 0.60:

        correlation_text = (
            "moderate"
        )

    else:

        correlation_text = (
            "weak"
        )

    return (
        f"Overall GSMaP/CHIRPS weekly agreement is "
        f"{correlation_text} "
        f"(Pearson r={correlation:.3f}). "
        f"GSMaP mean error relative to CHIRPS is "
        f"{rmse:.3f} mm RMSE with mean bias "
        f"{bias:.3f} mm."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "AGRIVISION AI - RAINFALL SOURCE COMPARISON"
    )

    print(
        "=" * 80
    )

    print(
        "\nThis is a source-comparability test."
    )

    print(
        "It does NOT change the production model."
    )

    print(
        f"\nComparison period:"
    )

    print(
        f"{START_DATE} → {END_DATE}"
    )

    initialize_earth_engine()

    points = (
        load_target_points()
    )

    weeks = (
        build_week_windows()
    )

    print(
        "\nWeeks:"
    )

    print(
        weeks[
            [
                "Week_Start",
                "Week_End",
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Fetch
    # --------------------------------------------------------

    chirps = fetch_chirps(
        points,
        weeks,
    )

    gsmap = fetch_gsmap(
        points,
        weeks,
    )

    # --------------------------------------------------------
    # Save raw results
    # --------------------------------------------------------

    chirps_file = (
        OUTPUT_DIR
        / "chirps_comparison.csv"
    )

    gsmap_file = (
        OUTPUT_DIR
        / "gsmap_comparison.csv"
    )

    chirps.to_csv(
        chirps_file,
        index=False,
    )

    gsmap.to_csv(
        gsmap_file,
        index=False,
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    merged = merge_sources(
        chirps,
        gsmap,
    )

    merged_file = (
        OUTPUT_DIR
        / "chirps_vs_gsmap_weekly.csv"
    )

    merged.to_csv(
        merged_file,
        index=False,
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    (
        comparable,
        overall,
        point_stats,
    ) = calculate_statistics(
        merged
    )

    if not overall.empty:

        overall_file = (
            OUTPUT_DIR
            / "overall_comparison.csv"
        )

        point_file = (
            OUTPUT_DIR
            / "point_comparison.csv"
        )

        comparable_file = (
            OUTPUT_DIR
            / "comparable_records.csv"
        )

        overall.to_csv(
            overall_file,
            index=False,
        )

        point_stats.to_csv(
            point_file,
            index=False,
        )

        comparable.to_csv(
            comparable_file,
            index=False,
        )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "COMPARISON RESULTS"
    )

    print(
        "=" * 80
    )

    print(
        f"\nCHIRPS records: "
        f"{len(chirps):,}"
    )

    print(
        f"GSMaP records: "
        f"{len(gsmap):,}"
    )

    print(
        f"Comparable records: "
        f"{len(comparable):,}"
    )

    if not overall.empty:

        print(
            "\nOverall statistics:"
        )

        print(
            overall
            .round(6)
            .to_string(
                index=False
            )
        )

        print(
            "\nInterpretation:"
        )

        print(
            interpret(
                overall
            )
        )

        print(
            "\nWorst point-level RMSE:"
        )

        print(
            point_stats
            .sort_values(
                "RMSE_mm",
                ascending=False,
            )
            .head(10)
            .round(6)
            .to_string(
                index=False
            )
        )

    else:

        print(
            "\nNo comparable observations."
        )

    print(
        "\n" + "=" * 80
    )

    print(
        "RAINfall SOURCE COMPARISON COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        "\nOutputs:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nNo model was modified."
    )


if __name__ == "__main__":
    main()