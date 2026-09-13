from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "ml"
    / "feature_outputs"
    / "AgriVision_Maharashtra_ML_Features_2017_2025.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "ml" / "model_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "NDVI_next_week"

TRAIN_START_YEAR = 2017
TRAIN_END_YEAR = 2022

VALIDATION_YEAR = 2023

TEST_START_YEAR = 2024
TEST_END_YEAR = 2025


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [
    # Current conditions
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

    # Rainfall history
    "Rainfall_lag_1",
    "Rainfall_lag_2",
    "Rainfall_lag_4",
    "Rainfall_rolling_2w",
    "Rainfall_rolling_4w",
    "Rainfall_rolling_8w",

    # Temperature history
    "Temperature_lag_1",
    "Temperature_lag_2",
    "Temperature_lag_4",
    "Temperature_rolling_2w",
    "Temperature_rolling_4w",
    "Temperature_rolling_8w",

    # Temporal features
    "Week_Of_Year",
    "Month",
    "Day_Of_Year",
    "Year_Index",
    "Week_Sin",
    "Week_Cos",
    "Month_Sin",
    "Month_Cos",

    # Spatial features
    "Latitude",
    "Longitude",
]


# ============================================================
# MODEL CONFIGURATION
# ============================================================

RANDOM_STATE = 42

MODELS = {
    "Ridge": Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                Ridge(
                    alpha=1.0
                ),
            ),
        ]
    ),

    "RandomForest": Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=400,
                    max_depth=None,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    max_features=0.8,
                    bootstrap=True,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),

    "ExtraTrees": Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=400,
                    max_depth=None,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    max_features=0.8,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),

    "GradientBoosting": Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=300,
                    learning_rate=0.03,
                    max_depth=3,
                    min_samples_split=5,
                    min_samples_leaf=3,
                    subsample=0.9,
                    loss="huber",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),

    "HistGradientBoosting": Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=400,
                    learning_rate=0.04,
                    max_leaf_nodes=31,
                    max_depth=None,
                    min_samples_leaf=20,
                    l2_regularization=0.1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics."""

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def evaluate_predictions(model_name, y_true, y_pred, dataset_name):
    """Return metrics in a consistent format."""

    metrics = calculate_metrics(y_true, y_pred)

    return {
        "Model": model_name,
        "Dataset": dataset_name,
        "Samples": len(y_true),
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "R2": metrics["R2"],
    }


def persistence_prediction(df):
    """
    Persistence baseline:
    next week's NDVI = current week's NDVI.
    """

    return df["NDVI"].values.copy()


def save_dataframe(df, filename):
    """Save dataframe inside model output directory."""

    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    return path


def print_metrics_table(df, title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    if df.empty:
        print("No results available.")
        return

    display_df = df.copy()

    numeric_cols = [
        "MAE",
        "RMSE",
        "R2",
    ]

    for col in numeric_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(6)

    print(display_df.to_string(index=False))


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("AGRIVISION AI - MODEL TRAINING")
print("=" * 80)

print(f"\nProject root:")
print(PROJECT_ROOT)

print(f"\nInput file:")
print(INPUT_FILE)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nInput feature file was not found:\n{INPUT_FILE}\n\n"
        "Run feature_engineering.py first."
    )

df = pd.read_csv(INPUT_FILE)

print(f"\nLoaded dataset shape: {df.shape}")


# ============================================================
# BASIC DATA VALIDATION
# ============================================================

required_columns = FEATURE_COLUMNS + [TARGET, "Year"]

missing_required = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_required:
    raise ValueError(
        "\nMissing required columns:\n"
        + "\n".join(missing_required)
    )

# Keep only rows having a real target.
df = df[df[TARGET].notna()].copy()

# Ensure numeric data types where needed.
for col in FEATURE_COLUMNS:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

df["Year"] = pd.to_numeric(
    df["Year"],
    errors="coerce"
)

df = df[df[TARGET].notna()].copy()

# Sort chronologically.
sort_columns = [
    col
    for col in [
        "Year",
        "Week_Number",
        "Week_Start",
        "point_id",
    ]
    if col in df.columns
]

if sort_columns:
    df = df.sort_values(
        sort_columns
    ).reset_index(drop=True)


print(f"Usable supervised rows: {len(df):,}")
print(
    f"Year range: "
    f"{int(df['Year'].min())} - {int(df['Year'].max())}"
)


# ============================================================
# DATASET SPLIT
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


if train_df.empty:
    raise ValueError("Training dataset is empty.")

if validation_df.empty:
    raise ValueError("Validation dataset is empty.")

if test_df.empty:
    raise ValueError("Test dataset is empty.")


print("\nDataset split:")
print(
    f"Training   : {len(train_df):,} rows "
    f"({TRAIN_START_YEAR}-{TRAIN_END_YEAR})"
)
print(
    f"Validation : {len(validation_df):,} rows "
    f"({VALIDATION_YEAR})"
)
print(
    f"Test       : {len(test_df):,} rows "
    f"({TEST_START_YEAR}-{TEST_END_YEAR})"
)


# ============================================================
# FEATURE / TARGET MATRICES
# ============================================================

X_train = train_df[FEATURE_COLUMNS]
y_train = train_df[TARGET]

X_validation = validation_df[FEATURE_COLUMNS]
y_validation = validation_df[TARGET]

X_test = test_df[FEATURE_COLUMNS]
y_test = test_df[TARGET]


# ============================================================
# MISSING VALUE REPORT
# ============================================================

missing_report = []

for col in FEATURE_COLUMNS:
    missing_count = int(
        X_train[col].isna().sum()
    )

    missing_percent = (
        missing_count / len(X_train) * 100
    )

    missing_report.append(
        {
            "Feature": col,
            "Training_Missing_Count": missing_count,
            "Training_Missing_Percent": missing_percent,
        }
    )

missing_report_df = pd.DataFrame(
    missing_report
)

missing_report_df = missing_report_df.sort_values(
    "Training_Missing_Percent",
    ascending=False,
)

save_dataframe(
    missing_report_df,
    "training_feature_missingness.csv",
)


# ============================================================
# PERSISTENCE BASELINE
# ============================================================

print("\n" + "=" * 80)
print("PERSISTENCE BASELINE")
print("=" * 80)

baseline_validation_pred = persistence_prediction(
    validation_df
)

baseline_test_pred = persistence_prediction(
    test_df
)

baseline_validation = evaluate_predictions(
    "Persistence",
    y_validation,
    baseline_validation_pred,
    "Validation_2023",
)

baseline_test = evaluate_predictions(
    "Persistence",
    y_test,
    baseline_test_pred,
    "Test_2024_2025",
)

baseline_results = pd.DataFrame(
    [
        baseline_validation,
        baseline_test,
    ]
)

print_metrics_table(
    baseline_results,
    "Persistence Baseline Results",
)


# ============================================================
# TRAIN ALL CANDIDATE MODELS
# ============================================================

validation_results = []
test_results = []

trained_models = {}

prediction_frames = []


for model_name, model in MODELS.items():

    print("\n" + "=" * 80)
    print(f"TRAINING: {model_name}")
    print("=" * 80)

    print("Fitting on 2017-2022...")

    model.fit(
        X_train,
        y_train,
    )

    trained_models[model_name] = model

    print("Predicting validation set (2023)...")

    validation_pred = model.predict(
        X_validation
    )

    validation_metrics = evaluate_predictions(
        model_name,
        y_validation,
        validation_pred,
        "Validation_2023",
    )

    validation_results.append(
        validation_metrics
    )

    print(
        f"Validation MAE : "
        f"{validation_metrics['MAE']:.6f}"
    )

    print(
        f"Validation RMSE: "
        f"{validation_metrics['RMSE']:.6f}"
    )

    print(
        f"Validation R2  : "
        f"{validation_metrics['R2']:.6f}"
    )

    print("Predicting final test set (2024-2025)...")

    test_pred = model.predict(
        X_test
    )

    test_metrics = evaluate_predictions(
        model_name,
        y_test,
        test_pred,
        "Test_2024_2025",
    )

    test_results.append(
        test_metrics
    )

    print(
        f"Test MAE : "
        f"{test_metrics['MAE']:.6f}"
    )

    print(
        f"Test RMSE: "
        f"{test_metrics['RMSE']:.6f}"
    )

    print(
        f"Test R2  : "
        f"{test_metrics['R2']:.6f}"
    )

    # Save validation and test predictions.
    validation_prediction_df = validation_df.copy()

    validation_prediction_df[
        "Prediction"
    ] = validation_pred

    validation_prediction_df[
        "Absolute_Error"
    ] = np.abs(
        y_validation.values - validation_pred
    )

    validation_prediction_df[
        "Squared_Error"
    ] = (
        y_validation.values - validation_pred
    ) ** 2

    validation_prediction_df[
        "Model"
    ] = model_name

    prediction_frames.append(
        validation_prediction_df[
            [
                col
                for col in [
                    "point_id",
                    "district",
                    "region_type",
                    "Year",
                    "Week_Number",
                    "Week_Start",
                    "Latitude",
                    "Longitude",
                    TARGET,
                    "Prediction",
                    "Absolute_Error",
                    "Squared_Error",
                    "Model",
                ]
                if col in validation_prediction_df.columns
            ]
        ].assign(
            Dataset="Validation_2023"
        )
    )

    test_prediction_df = test_df.copy()

    test_prediction_df[
        "Prediction"
    ] = test_pred

    test_prediction_df[
        "Absolute_Error"
    ] = np.abs(
        y_test.values - test_pred
    )

    test_prediction_df[
        "Squared_Error"
    ] = (
        y_test.values - test_pred
    ) ** 2

    test_prediction_df[
        "Model"
    ] = model_name

    prediction_frames.append(
        test_prediction_df[
            [
                col
                for col in [
                    "point_id",
                    "district",
                    "region_type",
                    "Year",
                    "Week_Number",
                    "Week_Start",
                    "Latitude",
                    "Longitude",
                    TARGET,
                    "Prediction",
                    "Absolute_Error",
                    "Squared_Error",
                    "Model",
                ]
                if col in test_prediction_df.columns
            ]
        ].assign(
            Dataset="Test_2024_2025"
        )
    )


# ============================================================
# COMPARISON TABLES
# ============================================================

validation_results_df = pd.DataFrame(
    validation_results
)

test_results_df = pd.DataFrame(
    test_results
)

validation_results_df = validation_results_df.sort_values(
    "RMSE"
).reset_index(drop=True)

test_results_df = test_results_df.sort_values(
    "RMSE"
).reset_index(drop=True)

save_dataframe(
    validation_results_df,
    "validation_model_comparison.csv",
)

save_dataframe(
    test_results_df,
    "test_model_comparison.csv",
)

print_metrics_table(
    validation_results_df,
    "MODEL COMPARISON - VALIDATION 2023",
)

print_metrics_table(
    test_results_df,
    "MODEL COMPARISON - TEST 2024-2025",
)


# ============================================================
# SELECT BEST MODEL USING VALIDATION ONLY
# ============================================================

best_model_name = validation_results_df.iloc[0]["Model"]

print("\n" + "=" * 80)
print("MODEL SELECTION")
print("=" * 80)

print(
    f"Best validation model: {best_model_name}"
)

print(
    "Selection criterion: lowest validation RMSE."
)


# ============================================================
# RETRAIN BEST MODEL ON TRAIN + VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("FINAL MODEL RETRAINING")
print("=" * 80)

train_validation_df = pd.concat(
    [
        train_df,
        validation_df,
    ],
    ignore_index=True,
)

X_train_validation = train_validation_df[
    FEATURE_COLUMNS
]

y_train_validation = train_validation_df[
    TARGET
]

best_model = clone(
    MODELS[best_model_name]
)

print(
    f"Retraining {best_model_name} "
    f"on 2017-2023..."
)

best_model.fit(
    X_train_validation,
    y_train_validation,
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

final_test_pred = best_model.predict(
    X_test
)

final_test_metrics = calculate_metrics(
    y_test,
    final_test_pred,
)

print("\n" + "=" * 80)
print("FINAL TEST PERFORMANCE")
print("=" * 80)

print(
    f"Model : {best_model_name}"
)

print(
    f"MAE   : {final_test_metrics['MAE']:.6f}"
)

print(
    f"RMSE  : {final_test_metrics['RMSE']:.6f}"
)

print(
    f"R2    : {final_test_metrics['R2']:.6f}"
)


# ============================================================
# FINAL TEST PREDICTIONS
# ============================================================

final_predictions_df = test_df.copy()

final_predictions_df[
    "Predicted_NDVI"
] = final_test_pred

final_predictions_df[
    "NDVI_Error"
] = (
    final_predictions_df[TARGET]
    - final_predictions_df["Predicted_NDVI"]
)

final_predictions_df[
    "Absolute_Error"
] = np.abs(
    final_predictions_df["NDVI_Error"]
)

final_predictions_df[
    "Model"
] = best_model_name

save_dataframe(
    final_predictions_df,
    "final_test_predictions_2024_2025.csv",
)


# ============================================================
# LOCATION-SPECIFIC EVALUATION
# ============================================================

def evaluate_location(
    location_df,
    location_name,
):
    """
    Evaluate final model for a specific region/location.
    """

    if location_df.empty:
        return None

    X_location = location_df[
        FEATURE_COLUMNS
    ]

    y_location = location_df[
        TARGET
    ]

    predictions = best_model.predict(
        X_location
    )

    metrics = calculate_metrics(
        y_location,
        predictions,
    )

    result = {
        "Location": location_name,
        "Samples": len(location_df),
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "R2": metrics["R2"],
    }

    return result


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

    latur_result = evaluate_location(
        latur_df,
        "Latur",
    )

    if latur_result is not None:
        location_results.append(
            latur_result
        )


# ------------------------------------------------------------
# Kavathe / Khanapur case study
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

kavathe_result = evaluate_location(
    kavathe_df,
    "Kavathe-Khanapur",
)

if kavathe_result is not None:
    location_results.append(
        kavathe_result
    )


location_results_df = pd.DataFrame(
    location_results
)

if not location_results_df.empty:

    save_dataframe(
        location_results_df,
        "location_specific_test_results.csv",
    )

    print_metrics_table(
        location_results_df.rename(
            columns={
                "Location": "Model"
            }
        ),
        "LOCATION-SPECIFIC FINAL TEST RESULTS",
    )
else:
    print(
        "\nNo location-specific rows "
        "were found for Kavathe-Khanapur/Latur."
    )


# ============================================================
# DISTRICT-LEVEL PERFORMANCE
# ============================================================

if "district" in test_df.columns:

    district_results = []

    for district in sorted(
        test_df["district"]
        .dropna()
        .astype(str)
        .unique()
    ):

        district_df = test_df[
            test_df["district"]
            .astype(str)
            == district
        ].copy()

        if len(district_df) < 5:
            continue

        X_district = district_df[
            FEATURE_COLUMNS
        ]

        y_district = district_df[
            TARGET
        ]

        district_pred = best_model.predict(
            X_district
        )

        metrics = calculate_metrics(
            y_district,
            district_pred,
        )

        district_results.append(
            {
                "District": district,
                "Samples": len(district_df),
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
            }
        )

    district_results_df = pd.DataFrame(
        district_results
    )

    if not district_results_df.empty:

        district_results_df = (
            district_results_df
            .sort_values("RMSE")
            .reset_index(drop=True)
        )

        save_dataframe(
            district_results_df,
            "district_level_test_results.csv",
        )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def extract_feature_importance(
    fitted_pipeline,
    feature_names,
):
    """
    Extract feature importance from supported tree models.
    """

    model = fitted_pipeline.named_steps[
        "model"
    ]

    if hasattr(
        model,
        "feature_importances_",
    ):
        importances = (
            model.feature_importances_
        )

        return pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": importances,
            }
        ).sort_values(
            "Importance",
            ascending=False,
        )

    if hasattr(
        model,
        "coef_",
    ):
        coefficients = np.abs(
            model.coef_
        )

        return pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": coefficients,
            }
        ).sort_values(
            "Importance",
            ascending=False,
        )

    return pd.DataFrame(
        columns=[
            "Feature",
            "Importance",
        ]
    )


feature_importance_df = (
    extract_feature_importance(
        best_model,
        FEATURE_COLUMNS,
    )
)

if not feature_importance_df.empty:

    save_dataframe(
        feature_importance_df,
        "feature_importance.csv",
    )

    print("\n" + "=" * 80)
    print("TOP 15 FEATURES")
    print("=" * 80)

    print(
        feature_importance_df
        .head(15)
        .round(6)
        .to_string(index=False)
    )


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = (
    OUTPUT_DIR
    / "AgriVision_Best_NDVI_Model.joblib"
)

joblib.dump(
    best_model,
    final_model_path,
)

print(
    f"\nSaved trained model:\n"
    f"{final_model_path}"
)


# ============================================================
# SAVE MODEL METADATA
# ============================================================

metadata = {
    "project": "AgriVision AI",
    "target": TARGET,
    "feature_count": len(FEATURE_COLUMNS),
    "features": FEATURE_COLUMNS,
    "training_period": "2017-2022",
    "validation_period": "2023",
    "test_period": "2024-2025",
    "best_model": best_model_name,
    "validation_selection_metric": "RMSE",
    "final_test_MAE": final_test_metrics["MAE"],
    "final_test_RMSE": final_test_metrics["RMSE"],
    "final_test_R2": final_test_metrics["R2"],
    "training_rows": len(train_df),
    "validation_rows": len(validation_df),
    "test_rows": len(test_df),
}

metadata_df = pd.DataFrame(
    [
        {
            "Parameter": key,
            "Value": value,
        }
        for key, value in metadata.items()
    ]
)

save_dataframe(
    metadata_df,
    "model_metadata.csv",
)


# ============================================================
# SAVE COMPLETE REPORT
# ============================================================

report_path = (
    OUTPUT_DIR
    / "model_training_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as report:

    report.write(
        "AGRIVISION AI - MODEL TRAINING REPORT\n"
    )

    report.write(
        "=" * 80 + "\n\n"
    )

    report.write(
        "Dataset\n"
    )
    report.write(
        f"Input file: {INPUT_FILE}\n"
    )
    report.write(
        f"Supervised rows: {len(df):,}\n"
    )
    report.write(
        f"Feature count: {len(FEATURE_COLUMNS)}\n"
    )
    report.write(
        f"Target: {TARGET}\n\n"
    )

    report.write(
        "Chronological Split\n"
    )
    report.write(
        f"Training: {TRAIN_START_YEAR}-{TRAIN_END_YEAR}\n"
    )
    report.write(
        f"Validation: {VALIDATION_YEAR}\n"
    )
    report.write(
        f"Test: {TEST_START_YEAR}-{TEST_END_YEAR}\n\n"
    )

    report.write(
        "Model Selection\n"
    )
    report.write(
        "Models compared:\n"
    )

    for model_name in MODELS:
        report.write(
            f"- {model_name}\n"
        )

    report.write(
        "\nSelection criterion: "
        "lowest validation RMSE.\n"
    )

    report.write(
        f"Selected model: "
        f"{best_model_name}\n\n"
    )

    report.write(
        "Final Test Performance\n"
    )

    report.write(
        f"MAE: "
        f"{final_test_metrics['MAE']:.8f}\n"
    )

    report.write(
        f"RMSE: "
        f"{final_test_metrics['RMSE']:.8f}\n"
    )

    report.write(
        f"R2: "
        f"{final_test_metrics['R2']:.8f}\n\n"
    )

    report.write(
        "Persistence Baseline - Test\n"
    )

    report.write(
        f"MAE: "
        f"{baseline_test['MAE']:.8f}\n"
    )

    report.write(
        f"RMSE: "
        f"{baseline_test['RMSE']:.8f}\n"
    )

    report.write(
        f"R2: "
        f"{baseline_test['R2']:.8f}\n\n"
    )

    if not location_results_df.empty:

        report.write(
            "Location-Specific Test Performance\n"
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
        "Important Methodological Notes\n"
    )

    report.write(
        "- No random train/test split was used.\n"
    )

    report.write(
        "- Validation data was not used when "
        "selecting preprocessing parameters.\n"
    )

    report.write(
        "- Test data from 2024-2025 remained "
        "untouched during model selection.\n"
    )

    report.write(
        "- Missing feature values are handled "
        "inside model pipelines.\n"
    )

    report.write(
        "- The final selected model was retrained "
        "on 2017-2023 before final 2024-2025 evaluation.\n"
    )

print(
    f"\nSaved report:\n"
    f"{report_path}"
)


# ============================================================
# SAVE ALL CANDIDATE MODELS
# ============================================================

models_dir = OUTPUT_DIR / "candidate_models"
models_dir.mkdir(
    parents=True,
    exist_ok=True,
)

for model_name, model in trained_models.items():

    model_path = (
        models_dir
        / f"{model_name}_validation_trained.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)

print(
    f"\nBest model: {best_model_name}"
)

print(
    f"Final test MAE : "
    f"{final_test_metrics['MAE']:.6f}"
)

print(
    f"Final test RMSE: "
    f"{final_test_metrics['RMSE']:.6f}"
)

print(
    f"Final test R2  : "
    f"{final_test_metrics['R2']:.6f}"
)

print(
    "\nOutputs saved to:"
)

print(
    OUTPUT_DIR
)

print("\nFiles generated:")

for path in sorted(
    OUTPUT_DIR.rglob("*")
):

    if path.is_file():
        print(
            f"  {path.relative_to(OUTPUT_DIR)}"
        )

print(
    "\nAgriVision model training finished successfully."
)