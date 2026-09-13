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
    / "ablation"
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


# ============================================================
# TUNED EXTRATREES PARAMETERS
# ============================================================

MODEL_PARAMETERS = {
    "n_estimators": 500,
    "max_depth": 30,
    "max_features": 0.8,
    "min_samples_split": 8,
    "min_samples_leaf": 2,
}


# ============================================================
# FEATURE GROUPS
# ============================================================

FEATURE_GROUPS = {

    # --------------------------------------------------------
    # A. Current NDVI only
    # --------------------------------------------------------
    "NDVI_Only": [
        "NDVI",
    ],

    # --------------------------------------------------------
    # B. Current NDVI + NDVI history
    # --------------------------------------------------------
    "NDVI_History": [
        "NDVI",
        "NDVI_lag_1",
        "NDVI_lag_2",
        "NDVI_lag_3",
        "NDVI_lag_4",
        "NDVI_lag_8",
    ],

    # --------------------------------------------------------
    # C. NDVI + NDVI changes
    # --------------------------------------------------------
    "NDVI_Dynamics": [
        "NDVI",
        "NDVI_lag_1",
        "NDVI_lag_2",
        "NDVI_lag_3",
        "NDVI_lag_4",
        "NDVI_lag_8",
        "NDVI_change_1w",
        "NDVI_change_2w",
        "NDVI_change_4w",
    ],

    # --------------------------------------------------------
    # D. NDVI + rainfall
    # --------------------------------------------------------
    "NDVI_Rainfall": [
        "NDVI",
        "Rainfall_Current",
        "NDVI_lag_1",
        "NDVI_lag_2",
        "NDVI_lag_3",
        "NDVI_lag_4",
        "NDVI_lag_8",
        "Rainfall_lag_1",
        "Rainfall_lag_2",
        "Rainfall_lag_4",
        "Rainfall_rolling_2w",
        "Rainfall_rolling_4w",
        "Rainfall_rolling_8w",
    ],

    # --------------------------------------------------------
    # E. NDVI + temperature
    # --------------------------------------------------------
    "NDVI_Temperature": [
        "NDVI",
        "Temperature_Current",
        "NDVI_lag_1",
        "NDVI_lag_2",
        "NDVI_lag_3",
        "NDVI_lag_4",
        "NDVI_lag_8",
        "Temperature_lag_1",
        "Temperature_lag_2",
        "Temperature_lag_4",
        "Temperature_rolling_2w",
        "Temperature_rolling_4w",
        "Temperature_rolling_8w",
    ],

    # --------------------------------------------------------
    # F. NDVI + rainfall + temperature
    # --------------------------------------------------------
    "NDVI_Weather": [
        "NDVI",
        "Rainfall_Current",
        "Temperature_Current",

        "NDVI_lag_1",
        "NDVI_lag_2",
        "NDVI_lag_3",
        "NDVI_lag_4",
        "NDVI_lag_8",

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
    ],

    # --------------------------------------------------------
    # G. NDVI + temporal
    # --------------------------------------------------------
    "NDVI_Temporal": [
        "NDVI",

        "NDVI_lag_1",
        "NDVI_lag_2",
        "NDVI_lag_3",
        "NDVI_lag_4",
        "NDVI_lag_8",

        "NDVI_change_1w",
        "NDVI_change_2w",
        "NDVI_change_4w",

        "Week_Of_Year",
        "Month",
        "Day_Of_Year",
        "Year_Index",
        "Week_Sin",
        "Week_Cos",
        "Month_Sin",
        "Month_Cos",
    ],

    # --------------------------------------------------------
    # H. NDVI + weather + temporal
    # --------------------------------------------------------
    "NDVI_Weather_Temporal": [
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
    ],

    # --------------------------------------------------------
    # I. Full feature set
    # --------------------------------------------------------
    "Full_Features": [
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

    return mae, rmse, r2


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
# LOAD DATA
# ============================================================

print("=" * 80)
print("AGRIVISION AI - FEATURE ABLATION ANALYSIS")
print("=" * 80)

print(
    f"\nInput file:\n{INPUT_FILE}"
)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nFeature dataset not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE
)

print(
    f"\nLoaded dataset shape: {df.shape}"
)


# ============================================================
# VALIDATE FEATURE COLUMNS
# ============================================================

all_required_columns = set(
    [TARGET, "Year"]
)

for group_features in FEATURE_GROUPS.values():
    all_required_columns.update(
        group_features
    )

missing_columns = [
    col
    for col in sorted(
        all_required_columns
    )
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "\nMissing required columns:\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# CLEAN DATA
# ============================================================

for col in all_required_columns:

    if col == "Year":
        continue

    df[col] = pd.to_numeric(
        df[col],
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


# ============================================================
# RESULTS
# ============================================================

results = []

trained_models = {}


# ============================================================
# RUN ABLATION EXPERIMENTS
# ============================================================

total_groups = len(
    FEATURE_GROUPS
)

for index, (
    group_name,
    features,
) in enumerate(
    FEATURE_GROUPS.items(),
    start=1,
):

    print("\n" + "=" * 80)
    print(
        f"[{index}/{total_groups}] "
        f"{group_name}"
    )
    print("=" * 80)

    print(
        f"Features: {len(features)}"
    )

    X_train = train_df[
        features
    ]

    y_train = train_df[
        TARGET
    ]

    X_validation = validation_df[
        features
    ]

    y_validation = validation_df[
        TARGET
    ]

    X_test = test_df[
        features
    ]

    y_test = test_df[
        TARGET
    ]

    model = build_model()

    print(
        "Training on 2017-2022..."
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Evaluating validation 2023..."
    )

    validation_pred = model.predict(
        X_validation
    )

    validation_mae, validation_rmse, validation_r2 = (
        calculate_metrics(
            y_validation,
            validation_pred,
        )
    )

    print(
        f"Validation -> "
        f"MAE={validation_mae:.6f}, "
        f"RMSE={validation_rmse:.6f}, "
        f"R2={validation_r2:.6f}"
    )

    print(
        "Evaluating test 2024-2025..."
    )

    test_pred = model.predict(
        X_test
    )

    test_mae, test_rmse, test_r2 = (
        calculate_metrics(
            y_test,
            test_pred,
        )
    )

    print(
        f"Test -> "
        f"MAE={test_mae:.6f}, "
        f"RMSE={test_rmse:.6f}, "
        f"R2={test_r2:.6f}"
    )

    results.append(
        {
            "Feature_Group": group_name,
            "Feature_Count": len(features),

            "Validation_MAE": validation_mae,
            "Validation_RMSE": validation_rmse,
            "Validation_R2": validation_r2,

            "Test_MAE": test_mae,
            "Test_RMSE": test_rmse,
            "Test_R2": test_r2,

            "Features": " | ".join(features),
        }
    )

    trained_models[group_name] = model


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)


# Sort by validation RMSE.
results_validation = (
    results_df
    .sort_values(
        "Validation_RMSE"
    )
    .reset_index(drop=True)
)


# Sort separately by final test RMSE.
results_test = (
    results_df
    .sort_values(
        "Test_RMSE"
    )
    .reset_index(drop=True)
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_DIR
    / "ablation_results_all.csv",
    index=False,
)

results_validation.to_csv(
    OUTPUT_DIR
    / "ablation_results_by_validation.csv",
    index=False,
)

results_test.to_csv(
    OUTPUT_DIR
    / "ablation_results_by_test.csv",
    index=False,
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 80)
print("ABLATION RESULTS - VALIDATION 2023")
print("=" * 80)

print(
    results_validation[
        [
            "Feature_Group",
            "Feature_Count",
            "Validation_MAE",
            "Validation_RMSE",
            "Validation_R2",
        ]
    ]
    .round(6)
    .to_string(index=False)
)


print("\n" + "=" * 80)
print("ABLATION RESULTS - TEST 2024-2025")
print("=" * 80)

print(
    results_test[
        [
            "Feature_Group",
            "Feature_Count",
            "Test_MAE",
            "Test_RMSE",
            "Test_R2",
        ]
    ]
    .round(6)
    .to_string(index=False)
)


# ============================================================
# BEST VALIDATION GROUP
# ============================================================

best_validation_group = (
    results_validation.iloc[0]
)


print("\n" + "=" * 80)
print("BEST FEATURE GROUP BY VALIDATION")
print("=" * 80)

print(
    f"Group: "
    f"{best_validation_group['Feature_Group']}"
)

print(
    f"Validation RMSE: "
    f"{best_validation_group['Validation_RMSE']:.6f}"
)

print(
    f"Validation R2: "
    f"{best_validation_group['Validation_R2']:.6f}"
)


# ============================================================
# BEST TEST GROUP
# ============================================================

best_test_group = (
    results_test.iloc[0]
)


print("\n" + "=" * 80)
print("BEST FEATURE GROUP BY FINAL TEST")
print("=" * 80)

print(
    f"Group: "
    f"{best_test_group['Feature_Group']}"
)

print(
    f"Test RMSE: "
    f"{best_test_group['Test_RMSE']:.6f}"
)

print(
    f"Test R2: "
    f"{best_test_group['Test_R2']:.6f}"
)


# ============================================================
# FULL MODEL REFERENCE
# ============================================================

full_row = results_df[
    results_df["Feature_Group"]
    == "Full_Features"
]

if not full_row.empty:

    full_row = full_row.iloc[0]

    print("\n" + "=" * 80)
    print("FULL MODEL REFERENCE")
    print("=" * 80)

    print(
        f"Test MAE : "
        f"{full_row['Test_MAE']:.6f}"
    )

    print(
        f"Test RMSE: "
        f"{full_row['Test_RMSE']:.6f}"
    )

    print(
        f"Test R2  : "
        f"{full_row['Test_R2']:.6f}"
    )


# ============================================================
# LOCATION EVALUATION
# ============================================================

print("\n" + "=" * 80)
print("LOCATION ROBUSTNESS - BEST VALIDATION FEATURE GROUP")
print("=" * 80)


selected_group_name = (
    best_validation_group[
        "Feature_Group"
    ]
)

selected_model = trained_models[
    selected_group_name
]

selected_features = FEATURE_GROUPS[
    selected_group_name
]


def evaluate_location(
    subset_df,
    model,
    features,
):

    if subset_df.empty:
        return None

    X = subset_df[
        features
    ]

    y = subset_df[
        TARGET
    ]

    prediction = model.predict(
        X
    )

    mae, rmse, r2 = calculate_metrics(
        y,
        prediction,
    )

    return (
        mae,
        rmse,
        r2,
    )


location_results = []


# ------------------------------------------------------------
# Latur
# ------------------------------------------------------------

if "district" in test_df.columns:

    mask = (
        test_df["district"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "latur"
    )

    latur_df = test_df[
        mask
    ].copy()

    metrics = evaluate_location(
        latur_df,
        selected_model,
        selected_features,
    )

    if metrics is not None:

        location_results.append(
            {
                "Feature_Group": selected_group_name,
                "Location": "Latur",
                "Samples": len(latur_df),
                "MAE": metrics[0],
                "RMSE": metrics[1],
                "R2": metrics[2],
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


metrics = evaluate_location(
    kavathe_df,
    selected_model,
    selected_features,
)

if metrics is not None:

    location_results.append(
        {
            "Feature_Group": selected_group_name,
            "Location": "Kavathe-Khanapur",
            "Samples": len(kavathe_df),
            "MAE": metrics[0],
            "RMSE": metrics[1],
            "R2": metrics[2],
        }
    )


location_results_df = pd.DataFrame(
    location_results
)


if not location_results_df.empty:

    print(
        location_results_df
        .round(6)
        .to_string(index=False)
    )

    location_results_df.to_csv(
        OUTPUT_DIR
        / "ablation_best_group_location_results.csv",
        index=False,
    )


# ============================================================
# FEATURE GROUP PERFORMANCE DIFFERENCES
# ============================================================

if not full_row.empty:

    full_test_rmse = (
        float(full_row["Test_RMSE"])
    )

    full_test_mae = (
        float(full_row["Test_MAE"])
    )

    comparison_rows = []

    for _, row in results_test.iterrows():

        comparison_rows.append(
            {
                "Feature_Group": row[
                    "Feature_Group"
                ],

                "Test_RMSE": row[
                    "Test_RMSE"
                ],

                "RMSE_Difference_vs_Full": (
                    row["Test_RMSE"]
                    - full_test_rmse
                ),

                "Test_MAE": row[
                    "Test_MAE"
                ],

                "MAE_Difference_vs_Full": (
                    row["Test_MAE"]
                    - full_test_mae
                ),

                "Test_R2": row[
                    "Test_R2"
                ],
            }
        )

    difference_df = pd.DataFrame(
        comparison_rows
    )

    difference_df.to_csv(
        OUTPUT_DIR
        / "ablation_differences_vs_full.csv",
        index=False,
    )


# ============================================================
# SAVE REPORT
# ============================================================

report_path = (
    OUTPUT_DIR
    / "ablation_report.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8",
) as report:

    report.write(
        "AGRIVISION AI - FEATURE ABLATION REPORT\n"
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
        "Test period: 2024-2025\n\n"
    )

    report.write(
        "Model: Tuned ExtraTrees\n\n"
    )

    report.write(
        "Parameters:\n"
    )

    for key, value in MODEL_PARAMETERS.items():

        report.write(
            f"{key}: {value}\n"
        )

    report.write(
        "\nBest feature group by validation RMSE:\n"
    )

    report.write(
        f"{selected_group_name}\n"
    )

    report.write(
        f"Validation RMSE: "
        f"{best_validation_group['Validation_RMSE']:.8f}\n"
    )

    report.write(
        f"Validation R2: "
        f"{best_validation_group['Validation_R2']:.8f}\n\n"
    )

    report.write(
        "Best feature group by test RMSE:\n"
    )

    report.write(
        f"{best_test_group['Feature_Group']}\n"
    )

    report.write(
        f"Test RMSE: "
        f"{best_test_group['Test_RMSE']:.8f}\n"
    )

    report.write(
        f"Test R2: "
        f"{best_test_group['Test_R2']:.8f}\n\n"
    )

    report.write(
        "Interpretation should use validation results "
        "for feature-group selection. The test set is "
        "reported as final held-out evidence.\n"
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("ABLATION ANALYSIS COMPLETE")
print("=" * 80)

print(
    f"\nBest group by validation:"
)

print(
    selected_group_name
)

print(
    "\nOutputs saved to:"
)

print(
    OUTPUT_DIR
)

print(
    "\nImportant:"
)

print(
    "Do not replace the final production model yet."
)

print(
    "Review the ablation results before choosing "
    "the final feature configuration."
)
