# pyrefly: ignore [missing-import]

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline


# ============================================================
# AGRIVISION AI
# GSMaP MODEL TRAINING + EVALUATION
# 2017–2025
#
# PURPOSE:
# Compare GSMaP rainfall against the validated CHIRPS model
# using EXACTLY the same:
#   - target
#   - feature set
#   - chronological split
#   - ExtraTrees hyperparameters
#   - training-only imputation strategy
#
# Split:
#   Train      = 2017–2022
#   Validation = 2023
#   Test       = 2024–2025
#
# Final model:
#   Retrain on 2017–2023
#   Evaluate on 2024–2025
# ============================================================


print("=" * 90)
print("AGRIVISION AI - GSMaP MODEL TRAINING")
print("2017–2025")
print("=" * 90)


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

FEATURE_DATASET = (
    PROJECT_DIR
    / "ml"
    / "feature_outputs"
    / "AgriVision_Maharashtra_ML_Features_GSMaP_2017_2025.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "ml"
    / "model_outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_FILE = (
    OUTPUT_DIR
    / "AgriVision_GSMaP_Tuned_ExtraTrees_Model.joblib"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "AgriVision_GSMaP_Test_Predictions.csv"
)

VALIDATION_PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "AgriVision_GSMaP_Validation_Predictions.csv"
)

FEATURE_IMPORTANCE_FILE = (
    OUTPUT_DIR
    / "AgriVision_GSMaP_Feature_Importance.csv"
)

REPORT_FILE = (
    OUTPUT_DIR
    / "AgriVision_GSMaP_Model_Report.txt"
)


# ============================================================
# 2. CHECK INPUT
# ============================================================

print()
print("=" * 90)
print("CHECKING INPUT DATA")
print("=" * 90)

print()
print("Feature dataset:")
print(FEATURE_DATASET)

if not FEATURE_DATASET.exists():
    raise FileNotFoundError(
        f"\nGSMaP feature dataset not found:\n{FEATURE_DATASET}"
    )

print("PASS - GSMaP feature dataset found")


# ============================================================
# 3. LOAD DATA
# ============================================================

print()
print("=" * 90)
print("LOADING GSMaP FEATURE DATASET")
print("=" * 90)

df = pd.read_csv(
    FEATURE_DATASET,
    parse_dates=["Week_Start"]
)

print()
print(f"Rows:    {len(df):,}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# 4. BASIC VALIDATION
# ============================================================

print()
print("=" * 90)
print("VALIDATING DATASET")
print("=" * 90)

expected_rows = 33655
expected_points = 167
expected_features = 33

if len(df) != expected_rows:
    raise ValueError(
        f"Expected {expected_rows:,} rows, "
        f"found {len(df):,}."
    )

if df["point_id"].nunique() != expected_points:
    raise ValueError(
        f"Expected {expected_points} unique points, "
        f"found {df['point_id'].nunique()}."
    )

if df["NDVI_next_week"].isna().any():
    raise ValueError(
        "Missing target values detected."
    )

required_years = set(range(2017, 2026))

actual_years = set(
    df["Year"].dropna().astype(int).unique()
)

if actual_years != required_years:
    raise ValueError(
        f"Expected years 2017–2025, found {sorted(actual_years)}"
    )

duplicate_count = df.duplicated(
    subset=["point_id", "Week_Start"]
).sum()

print(
    f"Duplicate point-week rows: "
    f"{duplicate_count}"
)

if duplicate_count != 0:
    raise ValueError(
        "Duplicate point-week rows detected."
    )

print("PASS - Dataset validation completed")


# ============================================================
# 5. FEATURE LIST
# ============================================================

feature_columns = [

    # Current state
    "NDVI",
    "Rainfall_Current",
    "Temperature_Current",

    # NDVI history
    "NDVI_lag_1",
    "NDVI_lag_2",
    "NDVI_lag_3",
    "NDVI_lag_4",
    "NDVI_lag_8",

    # NDVI changes
    "NDVI_change_1w",
    "NDVI_change_2w",
    "NDVI_change_4w",

    # GSMaP rainfall
    "Rainfall_lag_1",
    "Rainfall_lag_2",
    "Rainfall_lag_4",
    "Rainfall_rolling_2w",
    "Rainfall_rolling_4w",
    "Rainfall_rolling_8w",

    # Temperature
    "Temperature_lag_1",
    "Temperature_lag_2",
    "Temperature_lag_4",
    "Temperature_rolling_2w",
    "Temperature_rolling_4w",
    "Temperature_rolling_8w",

    # Temporal
    "Week_Of_Year",
    "Month",
    "Day_Of_Year",
    "Year_Index",
    "Week_Sin",
    "Week_Cos",
    "Month_Sin",
    "Month_Cos",

    # Spatial
    "Latitude",
    "Longitude",
]

if len(feature_columns) != expected_features:
    raise ValueError(
        f"Expected {expected_features} features, "
        f"found {len(feature_columns)}."
    )

missing_feature_columns = [
    column
    for column in feature_columns
    if column not in df.columns
]

if missing_feature_columns:
    raise ValueError(
        "Missing feature columns:\n"
        + "\n".join(missing_feature_columns)
    )

print()
print(
    f"Number of model features: "
    f"{len(feature_columns)}"
)

print("PASS - 33 model features found")


# ============================================================
# 6. TARGET
# ============================================================

TARGET = "NDVI_next_week"

X = df[
    feature_columns
].copy()

y = df[
    TARGET
].copy()


# ============================================================
# 7. TEMPORAL SPLIT
# ============================================================

print()
print("=" * 90)
print("CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT")
print("=" * 90)

train_mask = (
    df["Year"]
    .between(2017, 2022)
)

validation_mask = (
    df["Year"] == 2023
)

test_mask = (
    df["Year"]
    .between(2024, 2025)
)

X_train = X.loc[train_mask].copy()
y_train = y.loc[train_mask].copy()

X_validation = X.loc[validation_mask].copy()
y_validation = y.loc[validation_mask].copy()

X_test = X.loc[test_mask].copy()
y_test = y.loc[test_mask].copy()

print()
print(
    f"Training samples:       {len(X_train):,}"
)

print(
    f"Validation samples:     {len(X_validation):,}"
)

print(
    f"Test samples:           {len(X_test):,}"
)

expected_train = 20506
expected_validation = 4665
expected_test = 8484

if len(X_train) != expected_train:
    raise ValueError(
        f"Unexpected training size. "
        f"Expected {expected_train:,}, "
        f"got {len(X_train):,}."
    )

if len(X_validation) != expected_validation:
    raise ValueError(
        f"Unexpected validation size. "
        f"Expected {expected_validation:,}, "
        f"got {len(X_validation):,}."
    )

if len(X_test) != expected_test:
    raise ValueError(
        f"Unexpected test size. "
        f"Expected {expected_test:,}, "
        f"got {len(X_test):,}."
    )

print()
print("PASS - Exact CHIRPS experiment split reproduced")


# ============================================================
# 8. MISSING FEATURE CHECK
# ============================================================

print()
print("=" * 90)
print("MISSING FEATURE COUNTS")
print("=" * 90)

train_missing = (
    X_train
    .isna()
    .sum()
    .sum()
)

validation_missing = (
    X_validation
    .isna()
    .sum()
    .sum()
)

test_missing = (
    X_test
    .isna()
    .sum()
    .sum()
)

print(
    f"Missing feature values - train:      "
    f"{train_missing:,}"
)

print(
    f"Missing feature values - validation:  "
    f"{validation_missing:,}"
)

print(
    f"Missing feature values - test:        "
    f"{test_missing:,}"
)

print()
print(
    "These missing values will be handled by "
    "training-only median imputation."
)


# ============================================================
# 9. MODEL CONFIGURATION
# ============================================================

print()
print("=" * 90)
print("MODEL CONFIGURATION")
print("=" * 90)

MODEL_PARAMS = {
    "n_estimators": 500,
    "max_depth": 30,
    "max_features": 0.8,
    "min_samples_split": 8,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1,
}

print()
for key, value in MODEL_PARAMS.items():
    print(
        f"{key}: {value}"
    )


# ============================================================
# 10. PIPELINE
# ============================================================

def create_model_pipeline():

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
                    **MODEL_PARAMS
                ),
            ),
        ]
    )


# ============================================================
# 11. VALIDATION MODEL
# ============================================================

print()
print("=" * 90)
print("TRAINING VALIDATION MODEL")
print("=" * 90)

validation_model = create_model_pipeline()

validation_model.fit(
    X_train,
    y_train
)

validation_prediction = (
    validation_model.predict(
        X_validation
    )
)


# ============================================================
# 12. METRIC FUNCTION
# ============================================================

def calculate_metrics(
    actual,
    predicted
):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


# ============================================================
# 13. VALIDATION METRICS
# ============================================================

validation_metrics = calculate_metrics(
    y_validation,
    validation_prediction
)

print()
print("Validation results:")
print(
    f"MAE:  {validation_metrics['MAE']:.8f}"
)
print(
    f"RMSE: {validation_metrics['RMSE']:.8f}"
)
print(
    f"R2:   {validation_metrics['R2']:.8f}"
)


# ============================================================
# 14. VALIDATION PERSISTENCE BASELINE
# ============================================================

validation_persistence = (
    X_validation["NDVI"]
    .to_numpy()
)

validation_baseline_metrics = calculate_metrics(
    y_validation,
    validation_persistence
)

print()
print("Validation persistence baseline:")
print(
    f"MAE:  "
    f"{validation_baseline_metrics['MAE']:.8f}"
)

print(
    f"RMSE: "
    f"{validation_baseline_metrics['RMSE']:.8f}"
)

print(
    f"R2:   "
    f"{validation_baseline_metrics['R2']:.8f}"
)


# ============================================================
# 15. SAVE VALIDATION PREDICTIONS
# ============================================================

validation_output = df.loc[
    validation_mask,
    [
        "Week_Start",
        "Year",
        "Week_Number",
        "point_id",
        "district",
        "Latitude",
        "Longitude",
    ]
].copy()

validation_output[
    "Actual_NDVI_next_week"
] = y_validation.to_numpy()

validation_output[
    "Predicted_NDVI_next_week"
] = validation_prediction

validation_output[
    "Prediction_Error"
] = (
    validation_output[
        "Predicted_NDVI_next_week"
    ]
    - validation_output[
        "Actual_NDVI_next_week"
    ]
)

validation_output[
    "Absolute_Error"
] = (
    validation_output[
        "Prediction_Error"
    ]
    .abs()
)

validation_output.to_csv(
    VALIDATION_PREDICTIONS_FILE,
    index=False
)


# ============================================================
# 16. FINAL MODEL
# ============================================================

print()
print("=" * 90)
print("TRAINING FINAL GSMaP MODEL")
print("=" * 90)

# Final production-style training:
# 2017–2023 = all historical data available before
# the held-out 2024–2025 test period.

final_train_mask = (
    df["Year"]
    .between(2017, 2023)
)

final_test_mask = (
    df["Year"]
    .between(2024, 2025)
)

X_final_train = X.loc[
    final_train_mask
].copy()

y_final_train = y.loc[
    final_train_mask
].copy()

X_final_test = X.loc[
    final_test_mask
].copy()

y_final_test = y.loc[
    final_test_mask
].copy()

print()
print(
    f"Final training samples: "
    f"{len(X_final_train):,}"
)

print(
    f"Final test samples:     "
    f"{len(X_final_test):,}"
)


if len(X_final_train) != 25171:
    raise ValueError(
        "Unexpected final training size. "
        f"Expected 25,171, got {len(X_final_train):,}."
    )

if len(X_final_test) != 8484:
    raise ValueError(
        "Unexpected final test size. "
        f"Expected 8,484, got {len(X_final_test):,}."
    )


final_model = create_model_pipeline()

final_model.fit(
    X_final_train,
    y_final_train
)

print()
print("PASS - Final GSMaP ExtraTrees model trained")


# ============================================================
# 17. FINAL TEST PREDICTIONS
# ============================================================

test_prediction = (
    final_model.predict(
        X_final_test
    )
)


# ============================================================
# 18. FINAL TEST METRICS
# ============================================================

test_metrics = calculate_metrics(
    y_final_test,
    test_prediction
)

print()
print("=" * 90)
print("FINAL GSMaP TEST PERFORMANCE")
print("=" * 90)

print()
print(
    f"MAE:  {test_metrics['MAE']:.8f}"
)

print(
    f"RMSE: {test_metrics['RMSE']:.8f}"
)

print(
    f"R2:   {test_metrics['R2']:.8f}"
)


# ============================================================
# 19. TEST PERSISTENCE BASELINE
# ============================================================

test_persistence = (
    X_final_test["NDVI"]
    .to_numpy()
)

test_baseline_metrics = calculate_metrics(
    y_final_test,
    test_persistence
)

print()
print("Test persistence baseline:")

print(
    f"MAE:  "
    f"{test_baseline_metrics['MAE']:.8f}"
)

print(
    f"RMSE: "
    f"{test_baseline_metrics['RMSE']:.8f}"
)

print(
    f"R2:   "
    f"{test_baseline_metrics['R2']:.8f}"
)


# ============================================================
# 20. TEST PREDICTIONS DATAFRAME
# ============================================================

test_output = df.loc[
    final_test_mask,
    [
        "Week_Start",
        "Year",
        "Week_Number",
        "point_id",
        "district",
        "Latitude",
        "Longitude",
        "Rainfall_Current",
        "Temperature_Current",
    ]
].copy()

test_output[
    "Actual_NDVI_next_week"
] = y_final_test.to_numpy()

test_output[
    "Predicted_NDVI_next_week"
] = test_prediction

test_output[
    "Prediction_Error"
] = (
    test_output[
        "Predicted_NDVI_next_week"
    ]
    - test_output[
        "Actual_NDVI_next_week"
    ]
)

test_output[
    "Absolute_Error"
] = (
    test_output[
        "Prediction_Error"
    ]
    .abs()
)

test_output.to_csv(
    PREDICTIONS_FILE,
    index=False
)


# ============================================================
# 21. FEATURE IMPORTANCE
# ============================================================

print()
print("=" * 90)
print("FEATURE IMPORTANCE")
print("=" * 90)

trained_extra_trees = (
    final_model.named_steps[
        "model"
    ]
)

feature_importance = pd.DataFrame({
    "feature": feature_columns,
    "importance": (
        trained_extra_trees
        .feature_importances_
    ),
})

feature_importance = (
    feature_importance
    .sort_values(
        "importance",
        ascending=False
    )
    .reset_index(drop=True)
)

feature_importance.to_csv(
    FEATURE_IMPORTANCE_FILE,
    index=False
)

print()
print(
    feature_importance.head(
        15
    ).to_string(
        index=False
    )
)


# ============================================================
# 22. LOCATION PERFORMANCE
# ============================================================

print()
print("=" * 90)
print("LOCATION PERFORMANCE")
print("=" * 90)

location_results = []

for district in [
    "Latur",
    "Satara",
]:

    location_mask = (
        test_output["district"]
        == district
    )

    if not location_mask.any():
        continue

    actual = test_output.loc[
        location_mask,
        "Actual_NDVI_next_week"
    ]

    predicted = test_output.loc[
        location_mask,
        "Predicted_NDVI_next_week"
    ]

    metrics = calculate_metrics(
        actual,
        predicted
    )

    location_results.append({
        "location": district,
        "samples": int(
            location_mask.sum()
        ),
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "R2": metrics["R2"],
    })


# Kavathe-Khanapur is represented by the
# case-study point in the dataset.
kavathe_mask = (
    test_output["point_id"]
    == "P147"
)

if kavathe_mask.any():

    actual = test_output.loc[
        kavathe_mask,
        "Actual_NDVI_next_week"
    ]

    predicted = test_output.loc[
        kavathe_mask,
        "Predicted_NDVI_next_week"
    ]

    metrics = calculate_metrics(
        actual,
        predicted
    )

    location_results.append({
        "location": "Kavathe-Khanapur",
        "samples": int(
            kavathe_mask.sum()
        ),
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "R2": metrics["R2"],
    })


location_results_df = pd.DataFrame(
    location_results
)

if not location_results_df.empty:

    print()

    print(
        location_results_df.to_string(
            index=False
        )
    )


# ============================================================
# 23. YEAR-WISE PERFORMANCE
# ============================================================

print()
print("=" * 90)
print("YEAR-WISE PERFORMANCE")
print("=" * 90)

year_results = []

for year in sorted(
    test_output["Year"]
    .unique()
):

    year_mask = (
        test_output["Year"]
        == year
    )

    actual = test_output.loc[
        year_mask,
        "Actual_NDVI_next_week"
    ]

    predicted = test_output.loc[
        year_mask,
        "Predicted_NDVI_next_week"
    ]

    metrics = calculate_metrics(
        actual,
        predicted
    )

    year_results.append({
        "Year": int(year),
        "samples": int(
            year_mask.sum()
        ),
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "R2": metrics["R2"],
    })

year_results_df = pd.DataFrame(
    year_results
)

print()
print(
    year_results_df.to_string(
        index=False
    )
)


# ============================================================
# 24. COMPARE WITH CHIRPS
# ============================================================

# Validated CHIRPS tuned model results.
CHIRPS_TEST_MAE = 0.043138945761492443
CHIRPS_TEST_RMSE = 0.06870671369197726
CHIRPS_TEST_R2 = 0.8960145325793286

mae_difference = (
    test_metrics["MAE"]
    - CHIRPS_TEST_MAE
)

rmse_difference = (
    test_metrics["RMSE"]
    - CHIRPS_TEST_RMSE
)

r2_difference = (
    test_metrics["R2"]
    - CHIRPS_TEST_R2
)

mae_percentage = (
    mae_difference
    / CHIRPS_TEST_MAE
    * 100
)

rmse_percentage = (
    rmse_difference
    / CHIRPS_TEST_RMSE
    * 100
)

r2_percentage_points = (
    r2_difference
    * 100
)


print()
print("=" * 90)
print("CHIRPS vs GSMaP")
print("=" * 90)

comparison_df = pd.DataFrame({
    "Metric": [
        "MAE",
        "RMSE",
        "R2",
    ],

    "CHIRPS": [
        CHIRPS_TEST_MAE,
        CHIRPS_TEST_RMSE,
        CHIRPS_TEST_R2,
    ],

    "GSMaP": [
        test_metrics["MAE"],
        test_metrics["RMSE"],
        test_metrics["R2"],
    ],
})

print()
print(
    comparison_df.to_string(
        index=False
    )
)

print()
print(
    f"GSMaP - CHIRPS MAE difference: "
    f"{mae_difference:+.8f} "
    f"({mae_percentage:+.3f}%)"
)

print(
    f"GSMaP - CHIRPS RMSE difference: "
    f"{rmse_difference:+.8f} "
    f"({rmse_percentage:+.3f}%)"
)

print(
    f"GSMaP - CHIRPS R2 difference: "
    f"{r2_difference:+.8f} "
    f"({r2_percentage_points:+.4f} percentage points)"
)


# ============================================================
# 25. SAVE MODEL
# ============================================================

joblib.dump(
    final_model,
    MODEL_FILE
)

print()
print(
    f"Model saved to:\n{MODEL_FILE}"
)


# ============================================================
# 26. REPORT
# ============================================================

report_lines = [

    "AGRIVISION AI - GSMaP MODEL TRAINING REPORT",

    "2017–2025",

    "",

    "RAIN FALL SOURCE",

    "GSMaP V8 operational",

    "",

    "FEATURE DATASET",

    str(FEATURE_DATASET),

    "",

    "MODEL",

    "ExtraTreesRegressor",

    "",

    "HYPERPARAMETERS",

    "n_estimators=500",

    "max_depth=30",

    "max_features=0.8",

    "min_samples_split=8",

    "min_samples_leaf=2",

    "random_state=42",

    "",

    "IMPUTATION",

    "SimpleImputer(strategy='median')",

    "Imputer is fitted only on the training portion.",

    "No validation/test information is used to fit imputation.",

    "",

    "CHRONOLOGICAL SPLIT",

    "Training: 2017–2022",

    "Validation: 2023",

    "Final training: 2017–2023",

    "Test: 2024–2025",

    "",

    "SAMPLE COUNTS",

    f"Training: {len(X_train):,}",

    f"Validation: {len(X_validation):,}",

    f"Final training: {len(X_final_train):,}",

    f"Test: {len(X_final_test):,}",

    "",

    "VALIDATION PERFORMANCE",

    f"MAE: {validation_metrics['MAE']:.8f}",

    f"RMSE: {validation_metrics['RMSE']:.8f}",

    f"R2: {validation_metrics['R2']:.8f}",

    "",

    "VALIDATION PERSISTENCE BASELINE",

    f"MAE: {validation_baseline_metrics['MAE']:.8f}",

    f"RMSE: {validation_baseline_metrics['RMSE']:.8f}",

    f"R2: {validation_baseline_metrics['R2']:.8f}",

    "",

    "FINAL TEST PERFORMANCE - GSMaP",

    f"MAE: {test_metrics['MAE']:.8f}",

    f"RMSE: {test_metrics['RMSE']:.8f}",

    f"R2: {test_metrics['R2']:.8f}",

    "",

    "TEST PERSISTENCE BASELINE",

    f"MAE: {test_baseline_metrics['MAE']:.8f}",

    f"RMSE: {test_baseline_metrics['RMSE']:.8f}",

    f"R2: {test_baseline_metrics['R2']:.8f}",

    "",

    "VALIDATED CHIRPS TEST PERFORMANCE",

    f"MAE: {CHIRPS_TEST_MAE:.8f}",

    f"RMSE: {CHIRPS_TEST_RMSE:.8f}",

    f"R2: {CHIRPS_TEST_R2:.8f}",

    "",

    "GSMaP MINUS CHIRPS",

    f"MAE difference: {mae_difference:+.8f}",

    f"RMSE difference: {rmse_difference:+.8f}",

    f"R2 difference: {r2_difference:+.8f}",

    "",

    "STATUS",

    "GSMaP model training and evaluation completed.",

    "Final decision between CHIRPS and GSMaP should be based on",

    "the side-by-side test metrics plus operational-data requirements.",
]


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(report_lines)
    )


# ============================================================
# 27. FINAL SUMMARY
# ============================================================

print()
print("=" * 90)
print("GSMaP MODEL TRAINING COMPLETED")
print("=" * 90)

print()
print("FINAL TEST:")
print(
    f"MAE  = {test_metrics['MAE']:.8f}"
)
print(
    f"RMSE = {test_metrics['RMSE']:.8f}"
)
print(
    f"R2   = {test_metrics['R2']:.8f}"
)

print()
print("CHIRPS REFERENCE:")
print(
    f"MAE  = {CHIRPS_TEST_MAE:.8f}"
)
print(
    f"RMSE = {CHIRPS_TEST_RMSE:.8f}"
)
print(
    f"R2   = {CHIRPS_TEST_R2:.8f}"
)

print()
print("OUTPUT FILES:")
print(MODEL_FILE)
print(PREDICTIONS_FILE)
print(VALIDATION_PREDICTIONS_FILE)
print(FEATURE_IMPORTANCE_FILE)
print(REPORT_FILE)

print()
print("=" * 90)
print("DONE")
print("=" * 90)