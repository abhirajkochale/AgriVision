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
    / "ndvi_cleaning_sensitivity"
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


# Same tuned ExtraTrees configuration
MODEL_PARAMETERS = {
    "n_estimators": 500,
    "max_depth": 30,
    "max_features": 0.8,
    "min_samples_split": 8,
    "min_samples_leaf": 2,
}


# Same conservative quality-audit thresholds
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
                    strategy="median",
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
# CLEAN / PREPARE
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


# ============================================================
# TRAINING-SIDE FILTER ONLY
# ============================================================

def filter_training_data(data):
    """
    Remove suspicious target values ONLY from training/
    validation data.

    The test set is NEVER filtered.
    """

    suspicious_mask = (
        (data[TARGET] <= LOW_THRESHOLD)
        | (data[TARGET] >= HIGH_THRESHOLD)
    )

    filtered = data[
        ~suspicious_mask
    ].copy()

    return filtered


# ============================================================
# TRAIN / VALIDATE / FINAL RETRAIN
# ============================================================

def train_model_pipeline(
    train_df,
    validation_df,
    test_df,
):
    # --------------------------------------------------------
    # Initial training
    # --------------------------------------------------------

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

    model_validation = build_model()

    model_validation.fit(
        X_train,
        y_train,
    )

    validation_pred = (
        model_validation.predict(
            X_validation
        )
    )

    validation_metrics = calculate_metrics(
        y_validation,
        validation_pred,
    )

    # --------------------------------------------------------
    # Retrain on train + validation
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

    # --------------------------------------------------------
    # FINAL TEST
    # IMPORTANT: test_df is untouched
    # --------------------------------------------------------

    X_test = test_df[
        FEATURE_COLUMNS
    ]

    y_test = test_df[
        TARGET
    ]

    test_pred = (
        final_model.predict(
            X_test
        )
    )

    test_metrics = calculate_metrics(
        y_test,
        test_pred,
    )

    return {
        "validation_model": model_validation,
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
    prediction,
    location_name,
):
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

    positions = np.flatnonzero(
        mask.to_numpy()
    )

    y_true = test_df.loc[
        mask,
        TARGET,
    ].values

    y_pred = prediction[
        positions
    ]

    metrics = calculate_metrics(
        y_true,
        y_pred,
    )

    return {
        "Location": location_name,
        "Samples": len(y_true),
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "R2": metrics["R2"],
    }


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("AGRIVISION AI - TRAINING CLEANING SENSITIVITY")
print("=" * 80)

print(
    f"\nInput file:\n{INPUT_FILE}"
)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{INPUT_FILE}"
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
    col
    for col in required_columns
    if col not in df.columns
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
# FIXED TEST SET
# ============================================================

full_train = df[
    (df["Year"] >= TRAIN_START_YEAR)
    & (df["Year"] <= TRAIN_END_YEAR)
].copy()

full_validation = df[
    df["Year"] == VALIDATION_YEAR
].copy()

full_test = df[
    (df["Year"] >= TEST_START_YEAR)
    & (df["Year"] <= TEST_END_YEAR)
].copy()


# ============================================================
# CLEANED TRAIN / VALIDATION
# ============================================================

clean_train = filter_training_data(
    full_train
)

clean_validation = filter_training_data(
    full_validation
)


# ============================================================
# SPLIT REPORT
# ============================================================

removed_train = (
    len(full_train)
    - len(clean_train)
)

removed_validation = (
    len(full_validation)
    - len(clean_validation)
)

split_df = pd.DataFrame(
    [
        {
            "Dataset": "Original",
            "Train": len(full_train),
            "Validation": len(full_validation),
            "Test": len(full_test),
        },
        {
            "Dataset": "Training_Cleaned",
            "Train": len(clean_train),
            "Validation": len(clean_validation),
            "Test": len(full_test),
        },
    ]
)

print("\n" + "=" * 80)
print("FIXED TEST DESIGN")
print("=" * 80)

print(
    split_df.to_string(
        index=False
    )
)

print(
    f"\nRemoved from training : {removed_train}"
)

print(
    f"Removed from validation: {removed_validation}"
)

print(
    f"Test observations kept EXACTLY the same: "
    f"{len(full_test)}"
)

split_df.to_csv(
    OUTPUT_DIR
    / "split_design.csv",
    index=False,
)


# ============================================================
# ORIGINAL MODEL
# ============================================================

print("\n" + "=" * 80)
print("MODEL A - ORIGINAL TRAINING DATA")
print("=" * 80)

original_result = train_model_pipeline(
    full_train,
    full_validation,
    full_test,
)

print(
    "\nValidation:"
)

print(
    original_result[
        "validation_metrics"
    ]
)

print(
    "\nFinal test:"
)

print(
    original_result[
        "test_metrics"
    ]
)


# ============================================================
# CLEANED TRAINING MODEL
# ============================================================

print("\n" + "=" * 80)
print("MODEL B - CLEANED TRAINING DATA")
print("=" * 80)

clean_result = train_model_pipeline(
    clean_train,
    clean_validation,
    full_test,
)

print(
    "\nValidation:"
)

print(
    clean_result[
        "validation_metrics"
    ]
)

print(
    "\nFinal test:"
)

print(
    clean_result[
        "test_metrics"
    ]
)


# ============================================================
# OVERALL COMPARISON
# ============================================================

comparison_df = pd.DataFrame(
    [
        {
            "Model": "Original_Training",
            "Validation_Samples": len(
                full_validation
            ),
            "Test_Samples": len(
                full_test
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
            "Model": "Cleaned_Training",
            "Validation_Samples": len(
                clean_validation
            ),
            "Test_Samples": len(
                full_test
            ),
            "Validation_MAE": (
                clean_result[
                    "validation_metrics"
                ]["MAE"]
            ),
            "Validation_RMSE": (
                clean_result[
                    "validation_metrics"
                ]["RMSE"]
            ),
            "Validation_R2": (
                clean_result[
                    "validation_metrics"
                ]["R2"]
            ),
            "Test_MAE": (
                clean_result[
                    "test_metrics"
                ]["MAE"]
            ),
            "Test_RMSE": (
                clean_result[
                    "test_metrics"
                ]["RMSE"
                ],
            ),
            "Test_R2": (
                clean_result[
                    "test_metrics"
                ]["R2"]
            ),
        },
    ]
)

print("\n" + "=" * 80)
print("FAIR ORIGINAL VS CLEANED COMPARISON")
print("=" * 80)

print(
    comparison_df
    .round(6)
    .to_string(index=False)
)

comparison_df.to_csv(
    OUTPUT_DIR
    / "fair_original_vs_cleaned.csv",
    index=False,
)


# ============================================================
# TEST DIFFERENCES
# ============================================================

original_test = (
    original_result[
        "test_metrics"
    ]
)

clean_test = (
    clean_result[
        "test_metrics"
    ]
)

rmse_change_percent = (
    (
        clean_test["RMSE"]
        - original_test["RMSE"]
    )
    / original_test["RMSE"]
) * 100

mae_change_percent = (
    (
        clean_test["MAE"]
        - original_test["MAE"]
    )
    / original_test["MAE"]
) * 100

r2_change = (
    clean_test["R2"]
    - original_test["R2"]
)


difference_df = pd.DataFrame(
    [
        {
            "Metric": "Test_MAE",
            "Original": original_test["MAE"],
            "Cleaned": clean_test["MAE"],
            "Difference": (
                clean_test["MAE"]
                - original_test["MAE"]
            ),
        },
        {
            "Metric": "Test_RMSE",
            "Original": original_test["RMSE"],
            "Cleaned": clean_test["RMSE"],
            "Difference": (
                clean_test["RMSE"]
                - original_test["RMSE"]
            ),
        },
        {
            "Metric": "Test_R2",
            "Original": original_test["R2"],
            "Cleaned": clean_test["R2"],
            "Difference": r2_change,
        },
    ]
)

difference_df.to_csv(
    OUTPUT_DIR
    / "fair_test_differences.csv",
    index=False,
)


# ============================================================
# LOCATION COMPARISON
# ============================================================

location_rows = []

for location in [
    "Latur",
    "Kavathe-Khanapur",
]:

    original_location = evaluate_location(
        full_test,
        original_result[
            "test_prediction"
        ],
        location,
    )

    cleaned_location = evaluate_location(
        full_test,
        clean_result[
            "test_prediction"
        ],
        location,
    )

    if original_location is not None:

        original_location[
            "Model"
        ] = "Original_Training"

        location_rows.append(
            original_location
        )

    if cleaned_location is not None:

        cleaned_location[
            "Model"
        ] = "Cleaned_Training"

        location_rows.append(
            cleaned_location
        )


location_df = pd.DataFrame(
    location_rows
)

if not location_df.empty:

    location_df = location_df[
        [
            "Model",
            "Location",
            "Samples",
            "MAE",
            "RMSE",
            "R2",
        ]
    ]

    print("\n" + "=" * 80)
    print("FAIR LOCATION COMPARISON")
    print("=" * 80)

    print(
        location_df
        .round(6)
        .to_string(index=False)
    )

    location_df.to_csv(
        OUTPUT_DIR
        / "fair_location_comparison.csv",
        index=False,
    )


# ============================================================
# COMMON TEST PREDICTIONS
# ============================================================

key_columns = [
    col
    for col in [
        "point_id",
        "Year",
        "Week_Number",
        "Week_Start",
    ]
    if col in full_test.columns
]

if key_columns:

    predictions_df = (
        full_test[
            key_columns
            + [
                TARGET,
            ]
        ]
        .copy()
    )

    predictions_df[
        "Original_Prediction"
    ] = original_result[
        "test_prediction"
    ]

    predictions_df[
        "Cleaned_Training_Prediction"
    ] = clean_result[
        "test_prediction"
    ]

    predictions_df[
        "Prediction_Change"
    ] = (
        predictions_df[
            "Cleaned_Training_Prediction"
        ]
        - predictions_df[
            "Original_Prediction"
        ]
    )

    predictions_df[
        "Original_Absolute_Error"
    ] = np.abs(
        predictions_df[
            TARGET
        ]
        - predictions_df[
            "Original_Prediction"
        ]
    )

    predictions_df[
        "Cleaned_Training_Absolute_Error"
    ] = np.abs(
        predictions_df[
            TARGET
        ]
        - predictions_df[
            "Cleaned_Training_Prediction"
        ]
    )

    predictions_df.to_csv(
        OUTPUT_DIR
        / "same_test_predictions.csv",
        index=False,
    )


# ============================================================
# SAVE CLEANED TRAINING MODEL
# ============================================================

cleaned_model_path = (
    OUTPUT_DIR
    / "AgriVision_Cleaned_Training_ExtraTrees_Model.joblib"
)

joblib.dump(
    clean_result[
        "final_model"
    ],
    cleaned_model_path,
)


# ============================================================
# DECISION
# ============================================================

if (
    rmse_change_percent < -0.5
    and r2_change > 0.001
):

    decision = (
        "CLEANED TRAINING DATA PROVIDES A MEANINGFUL "
        "IMPROVEMENT ON THE SAME UNTOUCHED TEST SET. "
        "Further satellite/source investigation is justified "
        "before promoting the cleaned model."
    )

elif (
    abs(rmse_change_percent) <= 0.5
    and abs(r2_change) <= 0.001
):

    decision = (
        "CLEANING HAS NEGLIGIBLE PRACTICAL IMPACT. "
        "KEEP THE ORIGINAL DATASET AND DOCUMENT THE AUDIT."
    )

else:

    decision = (
        "CLEANING DOES NOT CLEARLY IMPROVE PERFORMANCE "
        "ON THE SAME TEST SET. KEEP THE ORIGINAL DATASET "
        "UNLESS SOURCE-LEVEL INVESTIGATION PROVES THE "
        "EXTREME VALUES INVALID."
    )


# ============================================================
# REPORT
# ============================================================

report_path = (
    OUTPUT_DIR
    / "training_cleaning_sensitivity_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as report:

    report.write(
        "AGRIVISION AI - TRAINING CLEANING SENSITIVITY\n"
    )

    report.write(
        "=" * 80 + "\n\n"
    )

    report.write(
        "Test set was kept identical for both models.\n\n"
    )

    report.write(
        "Original model test performance:\n"
    )

    report.write(
        f"MAE: {original_test['MAE']:.8f}\n"
    )

    report.write(
        f"RMSE: {original_test['RMSE']:.8f}\n"
    )

    report.write(
        f"R2: {original_test['R2']:.8f}\n\n"
    )

    report.write(
        "Cleaned-training model test performance:\n"
    )

    report.write(
        f"MAE: {clean_test['MAE']:.8f}\n"
    )

    report.write(
        f"RMSE: {clean_test['RMSE']:.8f}\n"
    )

    report.write(
        f"R2: {clean_test['R2']:.8f}\n\n"
    )

    report.write(
        "Changes:\n"
    )

    report.write(
        f"RMSE change: {rmse_change_percent:.4f}%\n"
    )

    report.write(
        f"MAE change: {mae_change_percent:.4f}%\n"
    )

    report.write(
        f"R2 change: {r2_change:.6f}\n\n"
    )

    report.write(
        "Decision:\n"
    )

    report.write(
        decision
        + "\n\n"
    )

    report.write(
        "IMPORTANT:\n"
    )

    report.write(
        "The suspicious observations were removed only "
        "from training and validation for Model B.\n"
    )

    report.write(
        "All 2024-2025 test observations were retained "
        "for BOTH models.\n"
    )

    report.write(
        "The original feature dataset was not modified.\n"
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("TRAINING CLEANING SENSITIVITY COMPLETE")
print("=" * 80)

print(
    f"\nTest RMSE change: "
    f"{rmse_change_percent:.4f}%"
)

print(
    f"Test MAE change: "
    f"{mae_change_percent:.4f}%"
)

print(
    f"Test R2 change: "
    f"{r2_change:.6f}"
)

print(
    f"\nDecision:\n{decision}"
)

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
    "Original feature dataset was NOT modified."
)