from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "model_outputs"
    / "ndvi_sensitivity"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "NDVI_next_week"

TRAIN_START_YEAR = 2017
TRAIN_END_YEAR = 2022

VALIDATION_YEAR = 2023

TEST_START_YEAR = 2024
TEST_END_YEAR = 2025

RANDOM_STATE = 42

# Same tuned configuration that we selected earlier.
MODEL_PARAMETERS = {
    "n_estimators": 500,
    "max_depth": 30,
    "max_features": 0.8,
    "min_samples_split": 8,
    "min_samples_leaf": 2,
}

# Same conservative audit thresholds.
LOW_THRESHOLD = -0.8
HIGH_THRESHOLD = 0.95


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
# METRICS
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


# ============================================================
# MODEL
# ============================================================

def build_model():
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                ExtraTreesRegressor(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    **MODEL_PARAMETERS,
                ),
            ),
        ]
    )


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_data(df):
    data = df.copy()

    for column in FEATURE_COLUMNS:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data[TARGET] = pd.to_numeric(
        data[TARGET],
        errors="coerce",
    )

    data["Year"] = pd.to_numeric(
        data["Year"],
        errors="coerce",
    )

    data = data[
        data[TARGET].notna()
    ].copy()

    return data


def create_splits(data, apply_filter=False):
    """
    Create chronological train/validation/test splits.

    If apply_filter=True, remove suspicious target values
    using the audit thresholds.
    """

    working = data.copy()

    if apply_filter:

        suspicious_mask = (
            (working[TARGET] <= LOW_THRESHOLD)
            | (working[TARGET] >= HIGH_THRESHOLD)
        )

        working = working[
            ~suspicious_mask
        ].copy()

    train_df = working[
        (working["Year"] >= TRAIN_START_YEAR)
        & (working["Year"] <= TRAIN_END_YEAR)
    ].copy()

    validation_df = working[
        working["Year"] == VALIDATION_YEAR
    ].copy()

    test_df = working[
        (working["Year"] >= TEST_START_YEAR)
        & (working["Year"] <= TEST_END_YEAR)
    ].copy()

    return (
        train_df,
        validation_df,
        test_df,
    )


def train_and_evaluate(
    train_df,
    validation_df,
    test_df,
):
    """
    Train using 2017-2022,
    evaluate validation on 2023,
    retrain on 2017-2023,
    evaluate final model on 2024-2025.
    """

    X_train = train_df[
        FEATURE_COLUMNS
    ]

    y_train = train_df[
        TARGET
    ]

    X_validation = validation_df[
        FEATURE_COLUMNS
    ]

    y_validation = validation_df[
        TARGET
    ]

    X_test = test_df[
        FEATURE_COLUMNS
    ]

    y_test = test_df[
        TARGET
    ]

    # --------------------------------------------------------
    # First model: 2017-2022
    # --------------------------------------------------------

    validation_model = build_model()

    validation_model.fit(
        X_train,
        y_train,
    )

    validation_pred = (
        validation_model.predict(
            X_validation
        )
    )

    validation_metrics = (
        calculate_metrics(
            y_validation,
            validation_pred,
        )
    )

    # --------------------------------------------------------
    # Final model: 2017-2023
    # --------------------------------------------------------

    train_validation_df = pd.concat(
        [
            train_df,
            validation_df,
        ],
        ignore_index=True,
    )

    X_train_validation = (
        train_validation_df[
            FEATURE_COLUMNS
        ]
    )

    y_train_validation = (
        train_validation_df[
            TARGET
        ]
    )

    final_model = build_model()

    final_model.fit(
        X_train_validation,
        y_train_validation,
    )

    test_pred = (
        final_model.predict(
            X_test
        )
    )

    test_metrics = (
        calculate_metrics(
            y_test,
            test_pred,
        )
    )

    return {
        "validation_model": validation_model,
        "final_model": final_model,
        "validation_prediction": validation_pred,
        "test_prediction": test_pred,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }


# ============================================================
# LOCATION EVALUATION
# ============================================================

def evaluate_location(
    test_df,
    predictions,
    location_name,
):
    """
    Evaluate predictions for a specific location.
    """

    if test_df.empty:
        return None

    mask = pd.Series(
        False,
        index=test_df.index,
    )

    if location_name == "Latur":

        if "district" not in test_df.columns:
            return None

        mask = (
            test_df["district"]
            .astype(str)
            .str.strip()
            .str.lower()
            == "latur"
        )

    elif location_name == "Kavathe-Khanapur":

        if "point_id" in test_df.columns:

            mask = (
                mask
                | (
                    test_df["point_id"]
                    .astype(str)
                    .str.upper()
                    == "P147"
                )
            )

        if "region_type" in test_df.columns:

            mask = (
                mask
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

            mask = (
                mask
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

    else:

        return None

    if not mask.any():
        return None

    location_df = test_df[
        mask
    ].copy()

    location_positions = np.flatnonzero(
        mask.to_numpy()
    )

    location_predictions = predictions[
        location_positions
    ]

    y_true = location_df[
        TARGET
    ].values

    metrics = calculate_metrics(
        y_true,
        location_predictions,
    )

    return {
        "Location": location_name,
        "Samples": len(location_df),
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "R2": metrics["R2"],
    }


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("AGRIVISION AI - NDVI SENSITIVITY ANALYSIS")
print("=" * 80)

print(
    f"\nInput file:\n{INPUT_FILE}"
)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Feature dataset not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE
)

print(
    f"\nLoaded dataset shape: {df.shape}"
)

required_columns = (
    FEATURE_COLUMNS
    + [
        TARGET,
        "Year",
    ]
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "\nMissing required columns:\n"
        + "\n".join(missing_columns)
    )

df = prepare_data(
    df
)

print(
    f"Usable supervised rows: "
    f"{len(df):,}"
)


# ============================================================
# CREATE ORIGINAL DATA SPLIT
# ============================================================

(
    original_train,
    original_validation,
    original_test,
) = create_splits(
    df,
    apply_filter=False,
)


# ============================================================
# CREATE FILTERED DATA SPLIT
# ============================================================

(
    filtered_train,
    filtered_validation,
    filtered_test,
) = create_splits(
    df,
    apply_filter=True,
)


# ============================================================
# SPLIT COUNTS
# ============================================================

print("\n" + "=" * 80)
print("ORIGINAL VS FILTERED SPLIT")
print("=" * 80)

split_summary = pd.DataFrame(
    [
        {
            "Dataset": "Original",
            "Train": len(original_train),
            "Validation": len(original_validation),
            "Test": len(original_test),
            "Total": (
                len(original_train)
                + len(original_validation)
                + len(original_test)
            ),
        },
        {
            "Dataset": "Filtered",
            "Train": len(filtered_train),
            "Validation": len(filtered_validation),
            "Test": len(filtered_test),
            "Total": (
                len(filtered_train)
                + len(filtered_validation)
                + len(filtered_test)
            ),
        },
    ]
)

print(
    split_summary.to_string(
        index=False
    )
)

split_summary.to_csv(
    OUTPUT_DIR
    / "split_summary.csv",
    index=False,
)


# ============================================================
# COUNT REMOVED OBSERVATIONS BY SPLIT
# ============================================================

removed_train = (
    len(original_train)
    - len(filtered_train)
)

removed_validation = (
    len(original_validation)
    - len(filtered_validation)
)

removed_test = (
    len(original_test)
    - len(filtered_test)
)

removed_summary = pd.DataFrame(
    [
        {
            "Split": "Train_2017_2022",
            "Removed": removed_train,
        },
        {
            "Split": "Validation_2023",
            "Removed": removed_validation,
        },
        {
            "Split": "Test_2024_2025",
            "Removed": removed_test,
        },
    ]
)

removed_summary[
    "Percentage"
] = (
    removed_summary["Removed"]
    / np.array(
        [
            len(original_train),
            len(original_validation),
            len(original_test),
        ]
    )
    * 100
)

print("\n" + "=" * 80)
print("REMOVED OBSERVATIONS BY SPLIT")
print("=" * 80)

print(
    removed_summary
    .round(6)
    .to_string(index=False)
)

removed_summary.to_csv(
    OUTPUT_DIR
    / "removed_by_split.csv",
    index=False,
)


# ============================================================
# ORIGINAL MODEL
# ============================================================

print("\n" + "=" * 80)
print("TRAINING ORIGINAL DATA MODEL")
print("=" * 80)

original_result = train_and_evaluate(
    original_train,
    original_validation,
    original_test,
)

print(
    "\nOriginal validation:"
)

print(
    original_result[
        "validation_metrics"
    ]
)

print(
    "\nOriginal final test:"
)

print(
    original_result[
        "test_metrics"
    ]
)


# ============================================================
# FILTERED MODEL
# ============================================================

print("\n" + "=" * 80)
print("TRAINING FILTERED DATA MODEL")
print("=" * 80)

filtered_result = train_and_evaluate(
    filtered_train,
    filtered_validation,
    filtered_test,
)

print(
    "\nFiltered validation:"
)

print(
    filtered_result[
        "validation_metrics"
    ]
)

print(
    "\nFiltered final test:"
)

print(
    filtered_result[
        "test_metrics"
    ]
)


# ============================================================
# OVERALL COMPARISON
# ============================================================

comparison_rows = [
    {
        "Dataset_Version": "Original",
        "Validation_Samples": len(
            original_validation
        ),
        "Test_Samples": len(
            original_test
        ),
        "Validation_MAE": (
            original_result[
                "validation_metrics"
            ]["MAE"]
        ),
        "Validation_RMSE": (
            original_result[
                "validation_metrics"
            ]["RMSE"]
        ),
        "Validation_R2": (
            original_result[
                "validation_metrics"
            ]["R2"]
        ),
        "Test_MAE": (
            original_result[
                "test_metrics"
            ]["MAE"]
        ),
        "Test_RMSE": (
            original_result[
                "test_metrics"
            ]["RMSE"]
        ),
        "Test_R2": (
            original_result[
                "test_metrics"
            ]["R2"]
        ),
    },
    {
        "Dataset_Version": "Audit_Filtered",
        "Validation_Samples": len(
            filtered_validation
        ),
        "Test_Samples": len(
            filtered_test
        ),
        "Validation_MAE": (
            filtered_result[
                "validation_metrics"
            ]["MAE"]
        ),
        "Validation_RMSE": (
            filtered_result[
                "validation_metrics"
            ]["RMSE"]
        ),
        "Validation_R2": (
            filtered_result[
                "validation_metrics"
            ]["R2"]
        ),
        "Test_MAE": (
            filtered_result[
                "test_metrics"
            ]["MAE"]
        ),
        "Test_RMSE": (
            filtered_result[
                "test_metrics"
            ]["RMSE"]
        ),
        "Test_R2": (
            filtered_result[
                "test_metrics"
            ]["R2"]
        ),
    },
]

comparison_df = pd.DataFrame(
    comparison_rows
)

print("\n" + "=" * 80)
print("ORIGINAL VS FILTERED PERFORMANCE")
print("=" * 80)

print(
    comparison_df
    .round(6)
    .to_string(index=False)
)

comparison_df.to_csv(
    OUTPUT_DIR
    / "original_vs_filtered_comparison.csv",
    index=False,
)


# ============================================================
# DIFFERENCE ANALYSIS
# ============================================================

original_test_metrics = (
    original_result[
        "test_metrics"
    ]
)

filtered_test_metrics = (
    filtered_result[
        "test_metrics"
    ]
)

original_validation_metrics = (
    original_result[
        "validation_metrics"
    ]
)

filtered_validation_metrics = (
    filtered_result[
        "validation_metrics"
    ]
)


difference_df = pd.DataFrame(
    [
        {
            "Metric": "Validation_MAE",
            "Original": original_validation_metrics["MAE"],
            "Filtered": filtered_validation_metrics["MAE"],
            "Filtered_Minus_Original": (
                filtered_validation_metrics["MAE"]
                - original_validation_metrics["MAE"]
            ),
        },
        {
            "Metric": "Validation_RMSE",
            "Original": original_validation_metrics["RMSE"],
            "Filtered": filtered_validation_metrics["RMSE"],
            "Filtered_Minus_Original": (
                filtered_validation_metrics["RMSE"]
                - original_validation_metrics["RMSE"]
            ),
        },
        {
            "Metric": "Validation_R2",
            "Original": original_validation_metrics["R2"],
            "Filtered": filtered_validation_metrics["R2"],
            "Filtered_Minus_Original": (
                filtered_validation_metrics["R2"]
                - original_validation_metrics["R2"]
            ),
        },
        {
            "Metric": "Test_MAE",
            "Original": original_test_metrics["MAE"],
            "Filtered": filtered_test_metrics["MAE"],
            "Filtered_Minus_Original": (
                filtered_test_metrics["MAE"]
                - original_test_metrics["MAE"]
            ),
        },
        {
            "Metric": "Test_RMSE",
            "Original": original_test_metrics["RMSE"],
            "Filtered": filtered_test_metrics["RMSE"],
            "Filtered_Minus_Original": (
                filtered_test_metrics["RMSE"]
                - original_test_metrics["RMSE"]
            ),
        },
        {
            "Metric": "Test_R2",
            "Original": original_test_metrics["R2"],
            "Filtered": filtered_test_metrics["R2"],
            "Filtered_Minus_Original": (
                filtered_test_metrics["R2"]
                - original_test_metrics["R2"]
            ),
        },
    ]
)

difference_df.to_csv(
    OUTPUT_DIR
    / "performance_differences.csv",
    index=False,
)


# ============================================================
# LOCATION PERFORMANCE
# ============================================================

location_rows = []

# Original
for location_name in [
    "Latur",
    "Kavathe-Khanapur",
]:

    result = evaluate_location(
        original_test,
        original_result[
            "test_prediction"
        ],
        location_name,
    )

    if result is not None:

        result[
            "Dataset_Version"
        ] = "Original"

        location_rows.append(
            result
        )


# Filtered
for location_name in [
    "Latur",
    "Kavathe-Khanapur",
]:

    result = evaluate_location(
        filtered_test,
        filtered_result[
            "test_prediction"
        ],
        location_name,
    )

    if result is not None:

        result[
            "Dataset_Version"
        ] = "Audit_Filtered"

        location_rows.append(
            result
        )


location_comparison_df = pd.DataFrame(
    location_rows
)

if not location_comparison_df.empty:

    location_comparison_df = (
        location_comparison_df[
            [
                "Dataset_Version",
                "Location",
                "Samples",
                "MAE",
                "RMSE",
                "R2",
            ]
        ]
    )

    print("\n" + "=" * 80)
    print("LOCATION COMPARISON")
    print("=" * 80)

    print(
        location_comparison_df
        .round(6)
        .to_string(index=False)
    )

    location_comparison_df.to_csv(
        OUTPUT_DIR
        / "location_comparison.csv",
        index=False,
    )


# ============================================================
# TEST PREDICTION CHANGES
# ============================================================

# Only compare rows that remain in BOTH test datasets.
# Match using stable identifiers if available.

common_key_columns = [
    col
    for col in [
        "point_id",
        "Year",
        "Week_Number",
        "Week_Start",
    ]
    if col in original_test.columns
    and col in filtered_test.columns
]

if common_key_columns:

    original_prediction_df = (
        original_test[
            common_key_columns
            + [TARGET]
        ]
        .copy()
    )

    original_prediction_df[
        "Original_Prediction"
    ] = original_result[
        "test_prediction"
    ]

    filtered_prediction_df = (
        filtered_test[
            common_key_columns
            + [TARGET]
        ]
        .copy()
    )

    filtered_prediction_df[
        "Filtered_Prediction"
    ] = filtered_result[
        "test_prediction"
    ]

    merged_predictions = pd.merge(
        original_prediction_df,
        filtered_prediction_df,
        on=common_key_columns,
        suffixes=(
            "_OriginalRow",
            "_FilteredRow",
        ),
    )

    merged_predictions[
        "Prediction_Difference"
    ] = (
        merged_predictions[
            "Filtered_Prediction"
        ]
        - merged_predictions[
            "Original_Prediction"
        ]
    )

    merged_predictions.to_csv(
        OUTPUT_DIR
        / "common_test_prediction_changes.csv",
        index=False,
    )


# ============================================================
# SHOULD FILTERING BE CONSIDERED?
# ============================================================

test_rmse_change_percent = (
    (
        filtered_test_metrics["RMSE"]
        - original_test_metrics["RMSE"]
    )
    / original_test_metrics["RMSE"]
) * 100

test_mae_change_percent = (
    (
        filtered_test_metrics["MAE"]
        - original_test_metrics["MAE"]
    )
    / original_test_metrics["MAE"]
) * 100

test_r2_change = (
    filtered_test_metrics["R2"]
    - original_test_metrics["R2"]
)


# A negative RMSE/MAE change is better.
# A positive R2 change is better.
#
# We use a small practical threshold rather than declaring
# tiny numerical changes as meaningful improvements.

if (
    test_rmse_change_percent < -0.5
    and test_r2_change > 0.001
):

    recommendation = (
        "FILTERED VERSION SHOWS A MEANINGFUL IMPROVEMENT. "
        "Investigate the extreme observations further before "
        "considering removal from production data."
    )

elif (
    abs(test_rmse_change_percent) <= 0.5
    and abs(test_r2_change) <= 0.001
):

    recommendation = (
        "FILTERING HAS NEGLIGIBLE PRACTICAL IMPACT. "
        "KEEP THE ORIGINAL DATASET AND DOCUMENT THE QUALITY AUDIT."
    )

else:

    recommendation = (
        "FILTERING DOES NOT PROVIDE A CLEAR BENEFIT. "
        "KEEP THE ORIGINAL DATASET UNLESS SATELLITE-LEVEL "
        "INVESTIGATION PROVES OBSERVATIONS INVALID."
    )


print("\n" + "=" * 80)
print("SENSITIVITY INTERPRETATION")
print("=" * 80)

print(
    f"Test RMSE change: "
    f"{test_rmse_change_percent:.4f}%"
)

print(
    f"Test MAE change: "
    f"{test_mae_change_percent:.4f}%"
)

print(
    f"Test R2 change: "
    f"{test_r2_change:.6f}"
)

print(
    f"\nRecommendation:\n"
    f"{recommendation}"
)


# ============================================================
# REPORT
# ============================================================

report_path = (
    OUTPUT_DIR
    / "ndvi_sensitivity_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as report:

    report.write(
        "AGRIVISION AI - NDVI SENSITIVITY ANALYSIS\n"
    )

    report.write(
        "=" * 80 + "\n\n"
    )

    report.write(
        "Original dataset vs audit-filtered dataset\n\n"
    )

    report.write(
        "Model:\n"
    )

    report.write(
        "Tuned ExtraTrees\n"
    )

    report.write(
        "\nChronological split:\n"
    )

    report.write(
        "Train: 2017-2022\n"
    )

    report.write(
        "Validation: 2023\n"
    )

    report.write(
        "Test: 2024-2025\n\n"
    )

    report.write(
        "Audit filtering rule:\n"
    )

    report.write(
        f"Remove target <= {LOW_THRESHOLD}\n"
    )

    report.write(
        f"Remove target >= {HIGH_THRESHOLD}\n\n"
    )

    report.write(
        "Original test metrics:\n"
    )

    report.write(
        f"MAE: {original_test_metrics['MAE']:.8f}\n"
    )

    report.write(
        f"RMSE: {original_test_metrics['RMSE']:.8f}\n"
    )

    report.write(
        f"R2: {original_test_metrics['R2']:.8f}\n\n"
    )

    report.write(
        "Filtered test metrics:\n"
    )

    report.write(
        f"MAE: {filtered_test_metrics['MAE']:.8f}\n"
    )

    report.write(
        f"RMSE: {filtered_test_metrics['RMSE']:.8f}\n"
    )

    report.write(
        f"R2: {filtered_test_metrics['R2']:.8f}\n\n"
    )

    report.write(
        "Changes:\n"
    )

    report.write(
        f"RMSE change: "
        f"{test_rmse_change_percent:.4f}%\n"
    )

    report.write(
        f"MAE change: "
        f"{test_mae_change_percent:.4f}%\n"
    )

    report.write(
        f"R2 change: "
        f"{test_r2_change:.6f}\n\n"
    )

    report.write(
        "Recommendation:\n"
    )

    report.write(
        recommendation
        + "\n"
    )

    report.write(
        "\nIMPORTANT:\n"
    )

    report.write(
        "This experiment does not prove that extreme "
        "satellite observations are invalid. It only "
        "measures model sensitivity to their inclusion.\n"
    )

    report.write(
        "The original master dataset was not modified.\n"
    )


# ============================================================
# OPTIONAL SAVE OF FILTERED MODEL
# ============================================================

filtered_model_path = (
    OUTPUT_DIR
    / "AgriVision_Audit_Filtered_ExtraTrees_Model.joblib"
)

joblib.dump(
    filtered_result[
        "final_model"
    ],
    filtered_model_path,
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("NDVI SENSITIVITY ANALYSIS COMPLETE")
print("=" * 80)

print(
    "\nOutputs saved to:"
)

print(
    OUTPUT_DIR
)

print(
    "\nOriginal model was NOT replaced."
)

print(
    "Original master dataset was NOT modified."
)

print(
    "Review the sensitivity results before "
    "making any production-data decision."
)