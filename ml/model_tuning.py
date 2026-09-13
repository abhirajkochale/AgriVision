from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid
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

OUTPUT_DIR = PROJECT_ROOT / "ml" / "model_outputs"
TUNING_DIR = OUTPUT_DIR / "tuning"

TUNING_DIR.mkdir(
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
# HYPERPARAMETER SEARCH SPACE
# ============================================================
#
# We deliberately keep this bounded.
# The objective is robust tuning, not brute-force computation.
#

PARAM_GRID = {
    "n_estimators": [
        300,
        500,
        700,
    ],

    "max_depth": [
        None,
        20,
        30,
        40,
    ],

    "min_samples_split": [
        2,
        4,
        8,
    ],

    "min_samples_leaf": [
        1,
        2,
        4,
    ],

    "max_features": [
        0.6,
        0.8,
        1.0,
    ],
}


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
# MODEL BUILDER
# ============================================================

def build_model(params):
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
                    bootstrap=False,
                    **params,
                ),
            ),
        ]
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("AGRIVISION AI - EXTRATREES HYPERPARAMETER TUNING")
print("=" * 80)

print(f"\nInput file:")
print(INPUT_FILE)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Feature dataset not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE
)

print(
    f"\nLoaded dataset shape: "
    f"{df.shape}"
)


# ============================================================
# VALIDATION
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
        "Missing required columns:\n"
        + "\n".join(missing_columns)
    )


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
# CHRONOLOGICAL SPLIT
# ============================================================

train_df = df[
    (df["Year"] >= TRAIN_START_YEAR)
    & (df["Year"] <= TRAIN_END_YEAR)
].copy()

validation_df = df[
    df["Year"] == VALIDATION_YEAR
].copy()

test_df = df[
    (df["Year"] >= TEST_START_YEAR)
    & (df["Year"] <= TEST_END_YEAR)
].copy()


print("\nChronological split:")
print(
    f"Training   : {len(train_df):,}"
)
print(
    f"Validation : {len(validation_df):,}"
)
print(
    f"Test       : {len(test_df):,}"
)


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


# ============================================================
# CURRENT MODEL REFERENCE
# ============================================================

CURRENT_MODEL_FILE = (
    OUTPUT_DIR
    / "AgriVision_Best_NDVI_Model.joblib"
)

if not CURRENT_MODEL_FILE.exists():
    raise FileNotFoundError(
        "\nCurrent trained model was not found:\n"
        f"{CURRENT_MODEL_FILE}\n\n"
        "Run model_training.py first."
    )

current_model = joblib.load(
    CURRENT_MODEL_FILE
)

print(
    "\nLoaded current production candidate:"
)
print(
    CURRENT_MODEL_FILE
)


# ============================================================
# CURRENT MODEL VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("CURRENT MODEL REFERENCE")
print("=" * 80)

current_validation_pred = (
    current_model.predict(
        X_validation
    )
)

current_validation_metrics = (
    calculate_metrics(
        y_validation,
        current_validation_pred,
    )
)

print(
    f"Validation MAE : "
    f"{current_validation_metrics['MAE']:.6f}"
)

print(
    f"Validation RMSE: "
    f"{current_validation_metrics['RMSE']:.6f}"
)

print(
    f"Validation R2  : "
    f"{current_validation_metrics['R2']:.6f}"
)


# ============================================================
# GRID SEARCH
# ============================================================

parameter_combinations = list(
    ParameterGrid(
        PARAM_GRID
    )
)

total_combinations = len(
    parameter_combinations
)

print("\n" + "=" * 80)
print("HYPERPARAMETER SEARCH")
print("=" * 80)

print(
    f"Total parameter combinations: "
    f"{total_combinations}"
)

print(
    "Selection metric: validation RMSE"
)

results = []

best_model = None
best_params = None
best_rmse = np.inf


for index, params in enumerate(
    parameter_combinations,
    start=1,
):

    print(
        f"\n[{index}/{total_combinations}] "
        f"Testing: {params}"
    )

    model = build_model(
        params
    )

    model.fit(
        X_train,
        y_train,
    )

    validation_pred = model.predict(
        X_validation
    )

    metrics = calculate_metrics(
        y_validation,
        validation_pred,
    )

    results.append(
        {
            "Run": index,
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "R2": metrics["R2"],
            **params,
        }
    )

    print(
        f"MAE={metrics['MAE']:.6f} | "
        f"RMSE={metrics['RMSE']:.6f} | "
        f"R2={metrics['R2']:.6f}"
    )

    if metrics["RMSE"] < best_rmse:

        best_rmse = metrics[
            "RMSE"
        ]

        best_model = model

        best_params = params.copy()

        print(
            "  >>> NEW BEST MODEL"
        )


# ============================================================
# SAVE SEARCH RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "RMSE"
).reset_index(
    drop=True
)

results_df.to_csv(
    TUNING_DIR
    / "extratrees_tuning_results.csv",
    index=False,
)


# ============================================================
# TOP MODELS
# ============================================================

print("\n" + "=" * 80)
print("TOP 10 TUNED MODELS")
print("=" * 80)

print(
    results_df
    .head(10)
    .round(6)
    .to_string(index=False)
)


# ============================================================
# BEST VALIDATION MODEL
# ============================================================

print("\n" + "=" * 80)
print("BEST TUNED MODEL")
print("=" * 80)

print(
    f"Best validation RMSE: "
    f"{best_rmse:.6f}"
)

print(
    "\nBest parameters:"
)

for key, value in best_params.items():
    print(
        f"  {key}: {value}"
    )


best_validation_pred = (
    best_model.predict(
        X_validation
    )
)

best_validation_metrics = (
    calculate_metrics(
        y_validation,
        best_validation_pred,
    )
)

print("\nValidation performance:")

print(
    f"MAE : "
    f"{best_validation_metrics['MAE']:.6f}"
)

print(
    f"RMSE: "
    f"{best_validation_metrics['RMSE']:.6f}"
)

print(
    f"R2  : "
    f"{best_validation_metrics['R2']:.6f}"
)


# ============================================================
# RETRAIN ON 2017-2023
# ============================================================

print("\n" + "=" * 80)
print("RETRAINING BEST TUNED MODEL")
print("=" * 80)

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

final_tuned_model = build_model(
    best_params
)

print(
    "Training final tuned model "
    "on 2017-2023..."
)

final_tuned_model.fit(
    X_train_validation,
    y_train_validation,
)


# ============================================================
# FINAL TEST
# ============================================================

print("\n" + "=" * 80)
print("FINAL 2024-2025 TEST")
print("=" * 80)

tuned_test_pred = (
    final_tuned_model.predict(
        X_test
    )
)

tuned_test_metrics = (
    calculate_metrics(
        y_test,
        tuned_test_pred,
    )
)

current_test_pred = (
    current_model.predict(
        X_test
    )
)

current_test_metrics = (
    calculate_metrics(
        y_test,
        current_test_pred,
    )
)


comparison = pd.DataFrame(
    [
        {
            "Model": "Current ExtraTrees",
            "MAE": current_test_metrics["MAE"],
            "RMSE": current_test_metrics["RMSE"],
            "R2": current_test_metrics["R2"],
        },
        {
            "Model": "Tuned ExtraTrees",
            "MAE": tuned_test_metrics["MAE"],
            "RMSE": tuned_test_metrics["RMSE"],
            "R2": tuned_test_metrics["R2"],
        },
    ]
)

print(
    comparison.round(6)
    .to_string(index=False)
)


# ============================================================
# IMPROVEMENT
# ============================================================

current_rmse = (
    current_test_metrics["RMSE"]
)

tuned_rmse = (
    tuned_test_metrics["RMSE"]
)

rmse_improvement = (
    (
        current_rmse
        - tuned_rmse
    )
    / current_rmse
) * 100

current_mae = (
    current_test_metrics["MAE"]
)

tuned_mae = (
    tuned_test_metrics["MAE"]
)

mae_improvement = (
    (
        current_mae
        - tuned_mae
    )
    / current_mae
) * 100

current_r2 = (
    current_test_metrics["R2"]
)

tuned_r2 = (
    tuned_test_metrics["R2"]
)

r2_improvement = (
    tuned_r2
    - current_r2
)


print("\nImprovement over current model:")

print(
    f"RMSE improvement: "
    f"{rmse_improvement:.3f}%"
)

print(
    f"MAE improvement : "
    f"{mae_improvement:.3f}%"
)

print(
    f"R2 improvement  : "
    f"{r2_improvement:.6f}"
)


# ============================================================
# LOCATION-SPECIFIC EVALUATION
# ============================================================

def evaluate_subset(
    subset_df,
    model,
):

    if subset_df.empty:
        return None

    X_subset = subset_df[
        FEATURE_COLUMNS
    ]

    y_subset = subset_df[
        TARGET
    ]

    prediction = model.predict(
        X_subset
    )

    metrics = calculate_metrics(
        y_subset,
        prediction,
    )

    return metrics


location_results = []


# ------------------------------------------------------------
# Latur
# ------------------------------------------------------------

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
    ].copy()

    if not latur_df.empty:

        metrics = evaluate_subset(
            latur_df,
            final_tuned_model,
        )

        location_results.append(
            {
                "Location": "Latur",
                "Samples": len(latur_df),
                **metrics,
            }
        )


# ------------------------------------------------------------
# Kavathe-Khanapur
# ------------------------------------------------------------

case_study_mask = pd.Series(
    False,
    index=test_df.index,
)


if "point_id" in test_df.columns:

    case_study_mask = (
        case_study_mask
        | (
            test_df["point_id"]
            .astype(str)
            .str.upper()
            == "P147"
        )
    )


if "region_type" in test_df.columns:

    case_study_mask = (
        case_study_mask
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

    case_study_mask = (
        case_study_mask
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
    case_study_mask
].copy()


if not kavathe_df.empty:

    metrics = evaluate_subset(
        kavathe_df,
        final_tuned_model,
    )

    location_results.append(
        {
            "Location": "Kavathe-Khanapur",
            "Samples": len(kavathe_df),
            **metrics,
        }
    )


location_results_df = pd.DataFrame(
    location_results
)

if not location_results_df.empty:

    print(
        "\n" + "=" * 80
    )
    print(
        "LOCATION-SPECIFIC PERFORMANCE"
    )
    print(
        "=" * 80
    )

    print(
        location_results_df
        .round(6)
        .to_string(index=False)
    )

    location_results_df.to_csv(
        TUNING_DIR
        / "tuned_location_results.csv",
        index=False,
    )


# ============================================================
# FINAL PREDICTIONS
# ============================================================

final_predictions_df = test_df.copy()

final_predictions_df[
    "Predicted_NDVI_Tuned"
] = tuned_test_pred

final_predictions_df[
    "Absolute_Error_Tuned"
] = np.abs(
    y_test.values
    - tuned_test_pred
)

final_predictions_df.to_csv(
    TUNING_DIR
    / "tuned_test_predictions_2024_2025.csv",
    index=False,
)


# ============================================================
# SAVE BEST MODEL
# ============================================================

best_model_path = (
    OUTPUT_DIR
    / "AgriVision_Tuned_ExtraTrees_Model.joblib"
)

joblib.dump(
    final_tuned_model,
    best_model_path,
)


# ============================================================
# SAVE PARAMETERS
# ============================================================

parameter_rows = []

for key, value in best_params.items():

    parameter_rows.append(
        {
            "Parameter": key,
            "Value": value,
        }
    )

parameter_df = pd.DataFrame(
    parameter_rows
)

parameter_df.to_csv(
    TUNING_DIR
    / "best_extratrees_parameters.csv",
    index=False,
)


# ============================================================
# FINAL REPORT
# ============================================================

report_path = (
    TUNING_DIR
    / "tuning_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as report:

    report.write(
        "AGRIVISION AI - EXTRATREES TUNING REPORT\n"
    )

    report.write(
        "=" * 80 + "\n\n"
    )

    report.write(
        "Training period: 2017-2022\n"
    )

    report.write(
        "Validation period: 2023\n"
    )

    report.write(
        "Final test period: 2024-2025\n\n"
    )

    report.write(
        "Best parameters:\n"
    )

    for key, value in best_params.items():

        report.write(
            f"{key}: {value}\n"
        )

    report.write(
        "\nValidation performance:\n"
    )

    report.write(
        f"MAE: "
        f"{best_validation_metrics['MAE']:.8f}\n"
    )

    report.write(
        f"RMSE: "
        f"{best_validation_metrics['RMSE']:.8f}\n"
    )

    report.write(
        f"R2: "
        f"{best_validation_metrics['R2']:.8f}\n"
    )

    report.write(
        "\nFinal test performance:\n"
    )

    report.write(
        f"MAE: "
        f"{tuned_test_metrics['MAE']:.8f}\n"
    )

    report.write(
        f"RMSE: "
        f"{tuned_test_metrics['RMSE']:.8f}\n"
    )

    report.write(
        f"R2: "
        f"{tuned_test_metrics['R2']:.8f}\n"
    )

    report.write(
        "\nImprovement over current model:\n"
    )

    report.write(
        f"RMSE improvement: "
        f"{rmse_improvement:.4f}%\n"
    )

    report.write(
        f"MAE improvement: "
        f"{mae_improvement:.4f}%\n"
    )

    report.write(
        f"R2 improvement: "
        f"{r2_improvement:.6f}\n"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("TUNING COMPLETE")
print("=" * 80)

print(
    f"\nBest parameters:"
)

for key, value in best_params.items():
    print(
        f"  {key}: {value}"
    )

print(
    f"\nTuned test MAE : "
    f"{tuned_test_metrics['MAE']:.6f}"
)

print(
    f"Tuned test RMSE: "
    f"{tuned_test_metrics['RMSE']:.6f}"
)

print(
    f"Tuned test R2  : "
    f"{tuned_test_metrics['R2']:.6f}"
)

print(
    f"\nSaved tuned model:"
)

print(
    best_model_path
)

print(
    f"\nTuning outputs:"
)

print(
    TUNING_DIR
)

print(
    "\nDo NOT replace the current model yet."
)

print(
    "We will only promote the tuned model "
    "after reviewing the results."
)