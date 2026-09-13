from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "ml"
    / "feature_outputs"
    / "AgriVision_Maharashtra_ML_Features_2017_2025.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "ml"
    / "model_outputs"
    / "AgriVision_Tuned_ExtraTrees_Model.joblib"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "model_outputs"
    / "error_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "NDVI_next_week"

TEST_START_YEAR = 2024
TEST_END_YEAR = 2025


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "NDVI",
    "Rainfall_Current",
    "Temperature_Current",

    "NDVI_lag_1",
    "NDVI_lag_2",
    "NDVI_lag_3",
    "NDVI_lag_4",
    "NDVI_lag_8",

    "NDVI_change_1w",
    "NDVI_change_2w",
    "NDVI_change_4w",

    "Rainfall_lag_1",
    "Rainfall_lag_2",
    "Rainfall_lag_4",
    "Rainfall_rolling_2w",
    "Rainfall_rolling_4w",
    "Rainfall_rolling_8w",

    "Temperature_lag_1",
    "Temperature_lag_2",
    "Temperature_lag_4",
    "Temperature_rolling_2w",
    "Temperature_rolling_4w",
    "Temperature_rolling_8w",

    "Week_Of_Year",
    "Month",
    "Day_Of_Year",
    "Year_Index",
    "Week_Sin",
    "Week_Cos",
    "Month_Sin",
    "Month_Cos",

    "Latitude",
    "Longitude",
]


# ============================================================
# FUNCTIONS
# ============================================================

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def print_table(df, title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    if df.empty:
        print("No results.")
        return

    print(
        df.round(6).to_string(
            index=False
        )
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("AGRIVISION AI - ERROR ANALYSIS")
print("=" * 80)

print(
    f"\nInput:\n{INPUT_FILE}"
)

print(
    f"\nModel:\n{MODEL_FILE}"
)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Feature dataset not found:\n{INPUT_FILE}"
    )

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Tuned model not found:\n{MODEL_FILE}\n\n"
        "Run model_tuning.py first."
    )


df = pd.read_csv(
    INPUT_FILE
)

model = joblib.load(
    MODEL_FILE
)

print(
    f"\nLoaded dataset: {df.shape}"
)


# ============================================================
# VALIDATE COLUMNS
# ============================================================

required_columns = (
    FEATURE_COLUMNS
    + [
        TARGET,
        "Year",
    ]
)

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        "\nMissing required columns:\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# CLEAN NUMERIC DATA
# ============================================================

for col in FEATURE_COLUMNS:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce",
    )


df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce",
)


df["Year"] = pd.to_numeric(
    df["Year"],
    errors="coerce",
)


df = df[
    df[TARGET].notna()
].copy()


# ============================================================
# TEST DATA
# ============================================================

test_df = df[
    (df["Year"] >= TEST_START_YEAR)
    & (df["Year"] <= TEST_END_YEAR)
].copy()

test_df = test_df.reset_index(
    drop=True
)

print(
    f"\nTest rows: {len(test_df):,}"
)


# ============================================================
# PREDICTIONS
# ============================================================

X_test = test_df[
    FEATURE_COLUMNS
]

y_test = test_df[
    TARGET
]

print(
    "\nGenerating predictions..."
)

predictions = model.predict(
    X_test
)

test_df[
    "Predicted_NDVI"
] = predictions

test_df[
    "Error"
] = (
    test_df[TARGET]
    - test_df["Predicted_NDVI"]
)

test_df[
    "Absolute_Error"
] = np.abs(
    test_df["Error"]
)

test_df[
    "Squared_Error"
] = (
    test_df["Error"]
    ** 2
)


# ============================================================
# OVERALL TEST METRICS
# ============================================================

overall_metrics = calculate_metrics(
    y_test,
    predictions,
)

overall_df = pd.DataFrame(
    [
        {
            "Dataset": "2024-2025",
            "Samples": len(test_df),
            **overall_metrics,
        }
    ]
)

print_table(
    overall_df,
    "OVERALL TEST PERFORMANCE",
)

overall_df.to_csv(
    OUTPUT_DIR
    / "overall_test_metrics.csv",
    index=False,
)


# ============================================================
# WORST PREDICTIONS
# ============================================================

worst_50 = (
    test_df
    .sort_values(
        "Absolute_Error",
        ascending=False,
    )
    .head(50)
    .copy()
)

worst_columns = [
    col
    for col in [
        "point_id",
        "district",
        "region_type",
        "study_area",
        "Year",
        "Week_Number",
        "Week_Start",
        "Latitude",
        "Longitude",
        "NDVI",
        "NDVI_next_week",
        "Predicted_NDVI",
        "Error",
        "Absolute_Error",
        "Rainfall_Current",
        "Temperature_Current",
        "NDVI_change_1w",
        "NDVI_change_2w",
        "NDVI_change_4w",
    ]
    if col in worst_50.columns
]

worst_50 = worst_50[
    worst_columns
]

worst_50.to_csv(
    OUTPUT_DIR
    / "worst_50_predictions.csv",
    index=False,
)

print_table(
    worst_50.head(20),
    "TOP 20 WORST PREDICTIONS",
)


# ============================================================
# YEAR-WISE PERFORMANCE
# ============================================================

year_results = []

for year in sorted(
    test_df["Year"]
    .dropna()
    .unique()
):

    subset = test_df[
        test_df["Year"] == year
    ]

    metrics = calculate_metrics(
        subset[TARGET],
        subset["Predicted_NDVI"],
    )

    year_results.append(
        {
            "Year": int(year),
            "Samples": len(subset),
            **metrics,
        }
    )


year_results_df = pd.DataFrame(
    year_results
)

year_results_df.to_csv(
    OUTPUT_DIR
    / "year_wise_performance.csv",
    index=False,
)

print_table(
    year_results_df,
    "YEAR-WISE PERFORMANCE",
)


# ============================================================
# MONTH-WISE PERFORMANCE
# ============================================================

month_results = []

if "Month" in test_df.columns:

    for month in sorted(
        test_df["Month"]
        .dropna()
        .unique()
    ):

        subset = test_df[
            test_df["Month"] == month
        ]

        metrics = calculate_metrics(
            subset[TARGET],
            subset["Predicted_NDVI"],
        )

        month_results.append(
            {
                "Month": int(month),
                "Samples": len(subset),
                **metrics,
            }
        )


month_results_df = pd.DataFrame(
    month_results
)

month_results_df.to_csv(
    OUTPUT_DIR
    / "month_wise_performance.csv",
    index=False,
)

print_table(
    month_results_df,
    "MONTH-WISE PERFORMANCE",
)


# ============================================================
# RAINFALL CONDITION ANALYSIS
# ============================================================

rainfall_condition_results = []


if "Rainfall_Current" in test_df.columns:

    rainfall_values = (
        test_df["Rainfall_Current"]
        .dropna()
    )

    if not rainfall_values.empty:

        q1 = rainfall_values.quantile(
            0.25
        )

        q3 = rainfall_values.quantile(
            0.75
        )

        conditions = {
            "Low_Rainfall": (
                test_df["Rainfall_Current"]
                <= q1
            ),
            "Normal_Rainfall": (
                (
                    test_df["Rainfall_Current"]
                    > q1
                )
                & (
                    test_df["Rainfall_Current"]
                    <= q3
                )
            ),
            "High_Rainfall": (
                test_df["Rainfall_Current"]
                > q3
            ),
        }

        for condition_name, mask in (
            conditions.items()
        ):

            subset = test_df[
                mask
            ]

            if subset.empty:
                continue

            metrics = calculate_metrics(
                subset[TARGET],
                subset["Predicted_NDVI"],
            )

            rainfall_condition_results.append(
                {
                    "Condition": condition_name,
                    "Samples": len(subset),
                    "Rainfall_Min": subset[
                        "Rainfall_Current"
                    ].min(),
                    "Rainfall_Max": subset[
                        "Rainfall_Current"
                    ].max(),
                    **metrics,
                }
            )


rainfall_results_df = pd.DataFrame(
    rainfall_condition_results
)

if not rainfall_results_df.empty:

    rainfall_results_df.to_csv(
        OUTPUT_DIR
        / "rainfall_condition_performance.csv",
        index=False,
    )

    print_table(
        rainfall_results_df,
        "RAINFALL-CONDITION PERFORMANCE",
    )


# ============================================================
# TEMPERATURE CONDITION ANALYSIS
# ============================================================

temperature_results = []


if "Temperature_Current" in test_df.columns:

    temperature_values = (
        test_df["Temperature_Current"]
        .dropna()
    )

    if not temperature_values.empty:

        q1 = temperature_values.quantile(
            0.25
        )

        q3 = temperature_values.quantile(
            0.75
        )

        conditions = {
            "Low_Temperature": (
                test_df["Temperature_Current"]
                <= q1
            ),
            "Normal_Temperature": (
                (
                    test_df["Temperature_Current"]
                    > q1
                )
                & (
                    test_df["Temperature_Current"]
                    <= q3
                )
            ),
            "High_Temperature": (
                test_df["Temperature_Current"]
                > q3
            ),
        }

        for condition_name, mask in (
            conditions.items()
        ):

            subset = test_df[
                mask
            ]

            if subset.empty:
                continue

            metrics = calculate_metrics(
                subset[TARGET],
                subset["Predicted_NDVI"],
            )

            temperature_results.append(
                {
                    "Condition": condition_name,
                    "Samples": len(subset),
                    "Temperature_Min": subset[
                        "Temperature_Current"
                    ].min(),
                    "Temperature_Max": subset[
                        "Temperature_Current"
                    ].max(),
                    **metrics,
                }
            )


temperature_results_df = pd.DataFrame(
    temperature_results
)

if not temperature_results_df.empty:

    temperature_results_df.to_csv(
        OUTPUT_DIR
        / "temperature_condition_performance.csv",
        index=False,
    )

    print_table(
        temperature_results_df,
        "TEMPERATURE-CONDITION PERFORMANCE",
    )


# ============================================================
# ERROR MAGNITUDE BINS
# ============================================================

error_bins = [
    0.00,
    0.025,
    0.05,
    0.075,
    0.10,
    np.inf,
]

error_labels = [
    "0-0.025",
    "0.025-0.05",
    "0.05-0.075",
    "0.075-0.10",
    ">0.10",
]

test_df[
    "Error_Band"
] = pd.cut(
    test_df["Absolute_Error"],
    bins=error_bins,
    labels=error_labels,
    include_lowest=True,
)


error_band_results = (
    test_df[
        "Error_Band"
    ]
    .value_counts(
        sort=False
    )
    .rename_axis(
        "Error_Band"
    )
    .reset_index(
        name="Samples"
    )
)

error_band_results[
    "Percentage"
] = (
    error_band_results[
        "Samples"
    ]
    / len(test_df)
    * 100
)

error_band_results.to_csv(
    OUTPUT_DIR
    / "error_band_distribution.csv",
    index=False,
)

print_table(
    error_band_results,
    "ERROR MAGNITUDE DISTRIBUTION",
)


# ============================================================
# LOCATION-SPECIFIC PERFORMANCE
# ============================================================

location_results = []


def evaluate_subset(
    subset,
    location_name,
):

    if subset.empty:
        return None

    metrics = calculate_metrics(
        subset[TARGET],
        subset["Predicted_NDVI"],
    )

    return {
        "Location": location_name,
        "Samples": len(subset),
        **metrics,
    }


# -------------------------
# Latur
# -------------------------

if "district" in test_df.columns:

    latur_mask = (
        test_df["district"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "latur"
    )

    latur_df = test_df[
        latur_mask
    ]

    result = evaluate_subset(
        latur_df,
        "Latur",
    )

    if result is not None:
        location_results.append(
            result
        )


# -------------------------
# Kavathe-Khanapur
# -------------------------

kavathe_mask = pd.Series(
    False,
    index=test_df.index,
)


if "point_id" in test_df.columns:

    kavathe_mask = (
        kavathe_mask
        | (
            test_df["point_id"]
            .astype(str)
            .str.upper()
            == "P147"
        )
    )


if "region_type" in test_df.columns:

    kavathe_mask = (
        kavathe_mask
        | (
            test_df["region_type"]
            .astype(str)
            .str.lower()
            .str.contains(
                "kavathe",
                na=False,
            )
        )
    )


if "study_area" in test_df.columns:

    kavathe_mask = (
        kavathe_mask
        | (
            test_df["study_area"]
            .astype(str)
            .str.lower()
            .str.contains(
                "kavathe",
                na=False,
            )
        )
    )


kavathe_df = test_df[
    kavathe_mask
]

result = evaluate_subset(
    kavathe_df,
    "Kavathe-Khanapur",
)

if result is not None:
    location_results.append(
        result
    )


# -------------------------
# Satara
# -------------------------

if "district" in test_df.columns:

    satara_mask = (
        test_df["district"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "satara"
    )

    satara_df = test_df[
        satara_mask
    ]

    result = evaluate_subset(
        satara_df,
        "Satara",
    )

    if result is not None:
        location_results.append(
            result
        )


location_results_df = pd.DataFrame(
    location_results
)

if not location_results_df.empty:

    location_results_df.to_csv(
        OUTPUT_DIR
        / "location_performance.csv",
        index=False,
    )

    print_table(
        location_results_df,
        "LOCATION-SPECIFIC PERFORMANCE",
    )


# ============================================================
# KAVATHE WORST PREDICTIONS
# ============================================================

if not kavathe_df.empty:

    kavathe_worst = (
        kavathe_df
        .sort_values(
            "Absolute_Error",
            ascending=False,
        )
        .head(20)
        .copy()
    )

    kavathe_worst.to_csv(
        OUTPUT_DIR
        / "kavathe_khanapur_worst_predictions.csv",
        index=False,
    )

    print_table(
        kavathe_worst[
            [
                col
                for col in [
                    "point_id",
                    "Year",
                    "Week_Number",
                    "Week_Start",
                    "NDVI",
                    "NDVI_next_week",
                    "Predicted_NDVI",
                    "Error",
                    "Absolute_Error",
                    "Rainfall_Current",
                    "Temperature_Current",
                ]
                if col in kavathe_worst.columns
            ]
        ],
        "KAVATHE-KHANAPUR WORST PREDICTIONS",
    )


# ============================================================
# LATUR WORST PREDICTIONS
# ============================================================

if not latur_df.empty:

    latur_worst = (
        latur_df
        .sort_values(
            "Absolute_Error",
            ascending=False,
        )
        .head(20)
        .copy()
    )

    latur_worst.to_csv(
        OUTPUT_DIR
        / "latur_worst_predictions.csv",
        index=False,
    )

    print_table(
        latur_worst[
            [
                col
                for col in [
                    "point_id",
                    "Year",
                    "Week_Number",
                    "Week_Start",
                    "district",
                    "NDVI",
                    "NDVI_next_week",
                    "Predicted_NDVI",
                    "Error",
                    "Absolute_Error",
                    "Rainfall_Current",
                    "Temperature_Current",
                ]
                if col in latur_worst.columns
            ]
        ],
        "LATUR WORST PREDICTIONS",
    )


# ============================================================
# SAVE ALL TEST PREDICTIONS
# ============================================================

test_columns_to_save = [
    col
    for col in [
        "point_id",
        "district",
        "region_type",
        "study_area",
        "Year",
        "Week_Number",
        "Week_Start",
        "Week_End",
        "Latitude",
        "Longitude",
        "NDVI",
        "Rainfall_Current",
        "Temperature_Current",
        TARGET,
        "Predicted_NDVI",
        "Error",
        "Absolute_Error",
        "Squared_Error",
        "NDVI_change_1w",
        "NDVI_change_2w",
        "NDVI_change_4w",
        "Error_Band",
    ]
    if col in test_df.columns
]

test_df[
    test_columns_to_save
].to_csv(
    OUTPUT_DIR
    / "all_test_predictions_with_errors.csv",
    index=False,
)


# ============================================================
# SUMMARY REPORT
# ============================================================

report_path = (
    OUTPUT_DIR
    / "error_analysis_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as report:

    report.write(
        "AGRIVISION AI - ERROR ANALYSIS REPORT\n"
    )

    report.write(
        "=" * 80 + "\n\n"
    )

    report.write(
        "Model: Tuned ExtraTrees\n"
    )

    report.write(
        "Evaluation period: 2024-2025\n\n"
    )

    report.write(
        "Overall Performance\n"
    )

    report.write(
        f"Samples: {len(test_df):,}\n"
    )

    report.write(
        f"MAE: {overall_metrics['MAE']:.8f}\n"
    )

    report.write(
        f"RMSE: {overall_metrics['RMSE']:.8f}\n"
    )

    report.write(
        f"R2: {overall_metrics['R2']:.8f}\n\n"
    )

    if not location_results_df.empty:

        report.write(
            "Location Performance\n"
        )

        for _, row in (
            location_results_df.iterrows()
        ):

            report.write(
                f"{row['Location']}: "
                f"samples={int(row['Samples'])}, "
                f"MAE={row['MAE']:.8f}, "
                f"RMSE={row['RMSE']:.8f}, "
                f"R2={row['R2']:.8f}\n"
            )

        report.write("\n")

    report.write(
        "Error interpretation thresholds:\n"
    )

    report.write(
        "0.00-0.025 = very small error\n"
    )

    report.write(
        "0.025-0.05 = small error\n"
    )

    report.write(
        "0.05-0.075 = moderate error\n"
    )

    report.write(
        "0.075-0.10 = high error\n"
    )

    report.write(
        ">0.10 = very high error\n"
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("ERROR ANALYSIS COMPLETE")
print("=" * 80)

print(
    "\nOutputs saved to:"
)

print(
    OUTPUT_DIR
)

print(
    "\nMost important files:"
)

print(
    "  all_test_predictions_with_errors.csv"
)

print(
    "  worst_50_predictions.csv"
)

print(
    "  year_wise_performance.csv"
)

print(
    "  month_wise_performance.csv"
)

print(
    "  rainfall_condition_performance.csv"
)

print(
    "  temperature_condition_performance.csv"
)

print(
    "  location_performance.csv"
)

print(
    "  kavathe_khanapur_worst_predictions.csv"
)

print(
    "  latur_worst_predictions.csv"
)

print(
    "  error_analysis_report.txt"
)

print(
    "\nError analysis finished successfully."
)