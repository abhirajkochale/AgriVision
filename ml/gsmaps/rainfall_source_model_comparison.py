# pyrefly: ignore [missing-import]

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.stats import (
    ttest_rel,
    wilcoxon,
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# AGRIVISION AI
# CHIRPS vs GSMaP FORMAL MODEL COMPARISON
# 2017–2025
#
# PURPOSE:
# Compare CHIRPS and GSMaP rainfall sources using the
# EXACT SAME 8,484 held-out 2024–2025 observations.
#
# Models:
#   CHIRPS -> validated tuned ExtraTrees
#   GSMaP  -> tuned ExtraTrees trained with the same protocol
#
# Primary comparison:
#   MAE
#   RMSE
#   R2
#
# Paired comparison:
#   absolute-error difference
#   Wilcoxon signed-rank test
#   paired t-test
#   bootstrap 95% CI
#
# ============================================================


print("=" * 90)
print("AGRIVISION AI - CHIRPS vs GSMaP MODEL COMPARISON")
print("2017–2025")
print("=" * 90)


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]


# -------------------------
# CHIRPS
# -------------------------

CHIRPS_FEATURE_FILE = (
    PROJECT_DIR
    / "ml"
    / "feature_outputs"
    / "AgriVision_Maharashtra_ML_Features_2017_2025.csv"
)

CHIRPS_MODEL_FILE = (
    PROJECT_DIR
    / "ml"
    / "model_outputs"
    / "AgriVision_Tuned_ExtraTrees_Model.joblib"
)


# -------------------------
# GSMaP
# -------------------------

GSMAP_FEATURE_FILE = (
    PROJECT_DIR
    / "ml"
    / "feature_outputs"
    / "AgriVision_Maharashtra_ML_Features_GSMaP_2017_2025.csv"
)

GSMAP_MODEL_FILE = (
    PROJECT_DIR
    / "ml"
    / "model_outputs"
    / "AgriVision_GSMaP_Tuned_ExtraTrees_Model.joblib"
)


# -------------------------
# Output directory
# -------------------------

OUTPUT_DIR = (
    PROJECT_DIR
    / "ml"
    / "model_outputs"
    / "rainfall_source_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Output files

PAIRED_FILE = (
    OUTPUT_DIR
    / "paired_test_comparison.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "summary_metrics.csv"
)

YEAR_FILE = (
    OUTPUT_DIR
    / "year_wise_comparison.csv"
)

LOCATION_FILE = (
    OUTPUT_DIR
    / "location_wise_comparison.csv"
)

RAINFALL_CONDITION_FILE = (
    OUTPUT_DIR
    / "rainfall_condition_comparison.csv"
)

WIN_COUNT_FILE = (
    OUTPUT_DIR
    / "paired_win_counts.csv"
)

BOOTSTRAP_FILE = (
    OUTPUT_DIR
    / "bootstrap_results.csv"
)

REPORT_FILE = (
    OUTPUT_DIR
    / "rainfall_source_comparison_report.txt"
)


# ============================================================
# 2. CHECK INPUTS
# ============================================================

print()
print("=" * 90)
print("CHECKING INPUT FILES")
print("=" * 90)


input_files = {
    "CHIRPS feature dataset": CHIRPS_FEATURE_FILE,
    "CHIRPS model": CHIRPS_MODEL_FILE,
    "GSMaP feature dataset": GSMAP_FEATURE_FILE,
    "GSMaP model": GSMAP_MODEL_FILE,
}


for name, path in input_files.items():

    print()
    print(f"{name}:")
    print(path)

    if not path.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )

    print("PASS")


# ============================================================
# 3. LOAD DATASETS AND MODELS
# ============================================================

print()
print("=" * 90)
print("LOADING DATASETS AND MODELS")
print("=" * 90)


chirps_df = pd.read_csv(
    CHIRPS_FEATURE_FILE,
    parse_dates=["Week_Start"]
)

gsmap_df = pd.read_csv(
    GSMAP_FEATURE_FILE,
    parse_dates=["Week_Start"]
)


print()
print(
    f"CHIRPS feature rows: "
    f"{len(chirps_df):,}"
)

print(
    f"GSMaP feature rows:  "
    f"{len(gsmap_df):,}"
)


chirps_model = joblib.load(
    CHIRPS_MODEL_FILE
)

gsmap_model = joblib.load(
    GSMAP_MODEL_FILE
)


print()
print("PASS - Both models loaded")


# ============================================================
# 4. TEST SET FILTER
# ============================================================

print()
print("=" * 90)
print("SELECTING IDENTICAL 2024–2025 TEST SET")
print("=" * 90)


chirps_test = (
    chirps_df[
        chirps_df["Year"].between(
            2024,
            2025
        )
    ]
    .copy()
)

gsmap_test = (
    gsmap_df[
        gsmap_df["Year"].between(
            2024,
            2025
        )
    ]
    .copy()
)


print()
print(
    f"CHIRPS test rows: "
    f"{len(chirps_test):,}"
)

print(
    f"GSMaP test rows:  "
    f"{len(gsmap_test):,}"
)


EXPECTED_TEST_ROWS = 8484


if len(chirps_test) != EXPECTED_TEST_ROWS:

    raise ValueError(
        f"CHIRPS test rows should be "
        f"{EXPECTED_TEST_ROWS:,}, "
        f"got {len(chirps_test):,}."
    )


if len(gsmap_test) != EXPECTED_TEST_ROWS:

    raise ValueError(
        f"GSMaP test rows should be "
        f"{EXPECTED_TEST_ROWS:,}, "
        f"got {len(gsmap_test):,}."
    )


# ============================================================
# 5. ALIGN TEST SETS
# ============================================================

print()
print("=" * 90)
print("ALIGNING TEST OBSERVATIONS")
print("=" * 90)


KEY_COLUMNS = [
    "point_id",
    "Week_Start",
]


chirps_test = (
    chirps_test
    .sort_values(KEY_COLUMNS)
    .reset_index(drop=True)
)

gsmap_test = (
    gsmap_test
    .sort_values(KEY_COLUMNS)
    .reset_index(drop=True)
)


# Ensure exact point-week keys

chirps_keys = (
    chirps_test[
        KEY_COLUMNS
    ]
    .astype(str)
    .agg(
        "|".join,
        axis=1
    )
)

gsmap_keys = (
    gsmap_test[
        KEY_COLUMNS
    ]
    .astype(str)
    .agg(
        "|".join,
        axis=1
    )
)


if not chirps_keys.equals(
    gsmap_keys
):

    raise ValueError(
        "CHIRPS and GSMaP test observations "
        "are not perfectly aligned."
    )


print(
    "PASS - Exact point-week alignment confirmed"
)


# ============================================================
# 6. TARGET ALIGNMENT
# ============================================================

CHIRPS_TARGET = (
    chirps_test["NDVI_next_week"]
    .to_numpy()
)

GSMAP_TARGET = (
    gsmap_test["NDVI_next_week"]
    .to_numpy()
)


target_difference = (
    CHIRPS_TARGET
    - GSMAP_TARGET
)


max_target_difference = (
    np.abs(target_difference)
    .max()
)


print()
print(
    f"Maximum target difference between "
    f"datasets: {max_target_difference:.12f}"
)


if max_target_difference > 1e-12:

    raise ValueError(
        "CHIRPS and GSMaP test targets are not identical."
    )


print(
    "PASS - Actual NDVI targets are identical"
)


y_test = CHIRPS_TARGET


# ============================================================
# 7. FEATURE LISTS
# ============================================================

feature_columns = [

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


if len(feature_columns) != 33:

    raise ValueError(
        "Expected exactly 33 features."
    )


# ============================================================
# 8. GENERATE PREDICTIONS
# ============================================================

print()
print("=" * 90)
print("GENERATING PAIRED PREDICTIONS")
print("=" * 90)


X_chirps_test = (
    chirps_test[
        feature_columns
    ]
    .copy()
)

X_gsmap_test = (
    gsmap_test[
        feature_columns
    ]
    .copy()
)


print()
print("Generating CHIRPS predictions...")

chirps_predictions = (
    chirps_model
    .predict(
        X_chirps_test
    )
)


print("Generating GSMaP predictions...")

gsmap_predictions = (
    gsmap_model
    .predict(
        X_gsmap_test
    )
)


print()
print("PASS - Predictions generated")


# ============================================================
# 9. METRIC FUNCTION
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
# 10. OVERALL METRICS
# ============================================================

print()
print("=" * 90)
print("OVERALL TEST PERFORMANCE")
print("=" * 90)


chirps_metrics = calculate_metrics(
    y_test,
    chirps_predictions
)

gsmap_metrics = calculate_metrics(
    y_test,
    gsmap_predictions
)


summary_df = pd.DataFrame({

    "Metric": [
        "MAE",
        "RMSE",
        "R2",
    ],

    "CHIRPS": [
        chirps_metrics["MAE"],
        chirps_metrics["RMSE"],
        chirps_metrics["R2"],
    ],

    "GSMaP": [
        gsmap_metrics["MAE"],
        gsmap_metrics["RMSE"],
        gsmap_metrics["R2"],
    ],
})


summary_df["GSMaP_minus_CHIRPS"] = (
    summary_df["GSMaP"]
    - summary_df["CHIRPS"]
)


print()
print(
    summary_df.to_string(
        index=False
    )
)


# ============================================================
# 11. PAIRED ERROR ANALYSIS
# ============================================================

print()
print("=" * 90)
print("PAIRED ERROR ANALYSIS")
print("=" * 90)


chirps_error = (
    chirps_predictions
    - y_test
)

gsmap_error = (
    gsmap_predictions
    - y_test
)


chirps_abs_error = (
    np.abs(
        chirps_error
    )
)

gsmap_abs_error = (
    np.abs(
        gsmap_error
    )
)


absolute_error_difference = (
    gsmap_abs_error
    - chirps_abs_error
)


# Interpretation:
#
# negative -> GSMaP has lower error
# positive -> CHIRPS has lower error


gsmap_better_count = int(
    np.sum(
        absolute_error_difference
        < 0
    )
)

chirps_better_count = int(
    np.sum(
        absolute_error_difference
        > 0
    )
)

tie_count = int(
    np.sum(
        absolute_error_difference
        == 0
    )
)


win_count_df = pd.DataFrame({

    "Outcome": [
        "GSMaP lower absolute error",
        "CHIRPS lower absolute error",
        "Equal absolute error",
    ],

    "Samples": [
        gsmap_better_count,
        chirps_better_count,
        tie_count,
    ],

    "Percentage": [
        gsmap_better_count
        / len(y_test)
        * 100,

        chirps_better_count
        / len(y_test)
        * 100,

        tie_count
        / len(y_test)
        * 100,
    ],
})


print()
print(
    win_count_df.to_string(
        index=False
    )
)


# ============================================================
# 12. DESCRIPTIVE ERROR-DIFFERENCE STATISTICS
# ============================================================

difference_statistics = {

    "Mean absolute error difference":
        np.mean(
            absolute_error_difference
        ),

    "Median absolute error difference":
        np.median(
            absolute_error_difference
        ),

    "Std absolute error difference":
        np.std(
            absolute_error_difference,
            ddof=1
        ),

    "Minimum difference":
        np.min(
            absolute_error_difference
        ),

    "Maximum difference":
        np.max(
            absolute_error_difference
        ),
}


print()
print(
    "Interpretation:"
)

print(
    "Negative mean difference = GSMaP better"
)

print(
    "Positive mean difference = CHIRPS better"
)

print()

for name, value in difference_statistics.items():

    print(
        f"{name}: {value:.10f}"
    )


# ============================================================
# 13. PAIRED T-TEST
# ============================================================

print()
print("=" * 90)
print("PAIRED STATISTICAL TESTS")
print("=" * 90)


paired_t = ttest_rel(
    gsmap_abs_error,
    chirps_abs_error
)


print()
print("Paired t-test on absolute errors:")

print(
    f"Statistic: "
    f"{paired_t.statistic:.10f}"
)

print(
    f"p-value:   "
    f"{paired_t.pvalue:.10e}"
)


# ============================================================
# 14. WILCOXON SIGNED-RANK TEST
# ============================================================

# If all differences are exactly zero, the test
# is not applicable.

nonzero_difference_count = int(
    np.sum(
        absolute_error_difference
        != 0
    )
)


if nonzero_difference_count > 0:

    wilcoxon_result = wilcoxon(
        gsmap_abs_error,
        chirps_abs_error,
        alternative="two-sided",
        zero_method="wilcox",
        method="auto",
    )

    print()
    print(
        "Wilcoxon signed-rank test:"
    )

    print(
        f"Statistic: "
        f"{wilcoxon_result.statistic:.10f}"
    )

    print(
        f"p-value:   "
        f"{wilcoxon_result.pvalue:.10e}"
    )

else:

    wilcoxon_result = None

    print()
    print(
        "Wilcoxon test not applicable: "
        "all paired absolute errors are identical."
    )


# ============================================================
# 15. BOOTSTRAP CONFIDENCE INTERVAL
# ============================================================

print()
print("=" * 90)
print("BOOTSTRAP 95% CI")
print("=" * 90)


BOOTSTRAP_ITERATIONS = 5000
BOOTSTRAP_RANDOM_STATE = 42

rng = np.random.default_rng(
    BOOTSTRAP_RANDOM_STATE
)


n = len(
    absolute_error_difference
)


bootstrap_means = np.empty(
    BOOTSTRAP_ITERATIONS
)


for i in range(
    BOOTSTRAP_ITERATIONS
):

    sample = rng.choice(
        absolute_error_difference,
        size=n,
        replace=True,
    )

    bootstrap_means[i] = (
        np.mean(sample)
    )


bootstrap_lower = np.percentile(
    bootstrap_means,
    2.5
)

bootstrap_upper = np.percentile(
    bootstrap_means,
    97.5
)

bootstrap_mean = np.mean(
    bootstrap_means
)


print()
print(
    f"Mean difference: "
    f"{bootstrap_mean:.10f}"
)

print(
    f"95% CI lower:   "
    f"{bootstrap_lower:.10f}"
)

print(
    f"95% CI upper:   "
    f"{bootstrap_upper:.10f}"
)


# ============================================================
# 16. CREATE PAIRED TEST OUTPUT
# ============================================================

paired_df = chirps_test[
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


paired_df[
    "Actual_NDVI_next_week"
] = y_test


paired_df[
    "CHIRPS_Predicted_NDVI"
] = chirps_predictions


paired_df[
    "GSMaP_Predicted_NDVI"
] = gsmap_predictions


paired_df[
    "CHIRPS_Error"
] = chirps_error


paired_df[
    "GSMaP_Error"
] = gsmap_error


paired_df[
    "CHIRPS_Absolute_Error"
] = chirps_abs_error


paired_df[
    "GSMaP_Absolute_Error"
] = gsmap_abs_error


paired_df[
    "GSMaP_minus_CHIRPS_Absolute_Error"
] = absolute_error_difference


paired_df[
    "Better_Source"
] = np.where(
    gsmap_abs_error
    < chirps_abs_error,
    "GSMaP",
    np.where(
        chirps_abs_error
        < gsmap_abs_error,
        "CHIRPS",
        "Tie"
    )
)


# ============================================================
# 17. ADD COMMON RAINFALL INFORMATION
# ============================================================

paired_df[
    "CHIRPS_Rainfall"
] = chirps_test[
    "Rainfall_Current"
].to_numpy()


paired_df[
    "GSMaP_Rainfall"
] = gsmap_test[
    "Rainfall_Current"
].to_numpy()


paired_df[
    "Rainfall_Difference_GSMaP_minus_CHIRPS"
] = (
    paired_df[
        "GSMaP_Rainfall"
    ]
    - paired_df[
        "CHIRPS_Rainfall"
    ]
)


# ============================================================
# 18. RAINFALL CONDITION DEFINITION
# ============================================================

# To keep the comparison externally consistent,
# rainfall-condition bins are defined using the CHIRPS
# rainfall distribution from the same test observations.
#
# Quantiles:
#   Low    = <= 33rd percentile
#   Normal = >33rd and <=66th percentile
#   High   = >66th percentile
#
# These bins are used ONLY for stratified analysis.
# They do not affect model predictions.

rainfall_q33 = (
    paired_df[
        "CHIRPS_Rainfall"
    ]
    .quantile(
        1 / 3
    )
)

rainfall_q66 = (
    paired_df[
        "CHIRPS_Rainfall"
    ]
    .quantile(
        2 / 3
    )
)


def classify_rainfall(
    value
):

    if value <= rainfall_q33:
        return "Low_Rainfall"

    elif value <= rainfall_q66:
        return "Normal_Rainfall"

    return "High_Rainfall"


paired_df[
    "Rainfall_Condition"
] = (
    paired_df[
        "CHIRPS_Rainfall"
    ]
    .apply(
        classify_rainfall
    )
)


# ============================================================
# 19. YEAR-WISE COMPARISON
# ============================================================

print()
print("=" * 90)
print("YEAR-WISE COMPARISON")
print("=" * 90)


year_results = []


for year in [
    2024,
    2025,
]:

    subset = paired_df[
        paired_df["Year"]
        == year
    ]

    actual = subset[
        "Actual_NDVI_next_week"
    ]

    chirps_pred = subset[
        "CHIRPS_Predicted_NDVI"
    ]

    gsmap_pred = subset[
        "GSMaP_Predicted_NDVI"
    ]

    chirps_m = calculate_metrics(
        actual,
        chirps_pred
    )

    gsmap_m = calculate_metrics(
        actual,
        gsmap_pred
    )

    year_results.append({

        "Year": year,

        "Samples": len(
            subset
        ),

        "CHIRPS_MAE":
            chirps_m["MAE"],

        "GSMaP_MAE":
            gsmap_m["MAE"],

        "GSMaP_minus_CHIRPS_MAE":
            gsmap_m["MAE"]
            - chirps_m["MAE"],

        "CHIRPS_RMSE":
            chirps_m["RMSE"],

        "GSMaP_RMSE":
            gsmap_m["RMSE"],

        "GSMaP_minus_CHIRPS_RMSE":
            gsmap_m["RMSE"]
            - chirps_m["RMSE"],

        "CHIRPS_R2":
            chirps_m["R2"],

        "GSMaP_R2":
            gsmap_m["R2"],

        "GSMaP_minus_CHIRPS_R2":
            gsmap_m["R2"]
            - chirps_m["R2"],
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
# 20. LOCATION-WISE COMPARISON
# ============================================================

print()
print("=" * 90)
print("LOCATION-WISE COMPARISON")
print("=" * 90)


location_definitions = {

    "Latur":
        paired_df[
            paired_df["district"]
            == "Latur"
        ],

    "Satara":
        paired_df[
            paired_df["district"]
            == "Satara"
        ],

    "Kavathe-Khanapur":
        paired_df[
            paired_df["point_id"]
            == "P147"
        ],
}


location_results = []


for location, subset in (
    location_definitions.items()
):

    if len(subset) == 0:
        continue

    actual = subset[
        "Actual_NDVI_next_week"
    ]

    chirps_pred = subset[
        "CHIRPS_Predicted_NDVI"
    ]

    gsmap_pred = subset[
        "GSMaP_Predicted_NDVI"
    ]

    chirps_m = calculate_metrics(
        actual,
        chirps_pred
    )

    gsmap_m = calculate_metrics(
        actual,
        gsmap_pred
    )

    location_results.append({

        "Location":
            location,

        "Samples":
            len(subset),

        "CHIRPS_MAE":
            chirps_m["MAE"],

        "GSMaP_MAE":
            gsmap_m["MAE"],

        "GSMaP_minus_CHIRPS_MAE":
            gsmap_m["MAE"]
            - chirps_m["MAE"],

        "CHIRPS_RMSE":
            chirps_m["RMSE"],

        "GSMaP_RMSE":
            gsmap_m["RMSE"],

        "GSMaP_minus_CHIRPS_RMSE":
            gsmap_m["RMSE"]
            - chirps_m["RMSE"],

        "CHIRPS_R2":
            chirps_m["R2"],

        "GSMaP_R2":
            gsmap_m["R2"],

        "GSMaP_minus_CHIRPS_R2":
            gsmap_m["R2"]
            - chirps_m["R2"],
    })


location_results_df = pd.DataFrame(
    location_results
)


print()

print(
    location_results_df.to_string(
        index=False
    )
)


# ============================================================
# 21. RAINFALL-CONDITION COMPARISON
# ============================================================

print()
print("=" * 90)
print("RAINFALL-CONDITION COMPARISON")
print("=" * 90)

print()
print(
    f"CHIRPS rainfall 33rd percentile: "
    f"{rainfall_q33:.6f}"
)

print(
    f"CHIRPS rainfall 66th percentile: "
    f"{rainfall_q66:.6f}"
)


condition_results = []


for condition in [
    "Low_Rainfall",
    "Normal_Rainfall",
    "High_Rainfall",
]:

    subset = paired_df[
        paired_df["Rainfall_Condition"]
        == condition
    ]

    if len(subset) == 0:
        continue

    actual = subset[
        "Actual_NDVI_next_week"
    ]

    chirps_pred = subset[
        "CHIRPS_Predicted_NDVI"
    ]

    gsmap_pred = subset[
        "GSMaP_Predicted_NDVI"
    ]

    chirps_m = calculate_metrics(
        actual,
        chirps_pred
    )

    gsmap_m = calculate_metrics(
        actual,
        gsmap_pred
    )

    condition_results.append({

        "Condition":
            condition,

        "Samples":
            len(subset),

        "Rainfall_Min_CHIRPS":
            subset[
                "CHIRPS_Rainfall"
            ].min(),

        "Rainfall_Max_CHIRPS":
            subset[
                "CHIRPS_Rainfall"
            ].max(),

        "CHIRPS_MAE":
            chirps_m["MAE"],

        "GSMaP_MAE":
            gsmap_m["MAE"],

        "GSMaP_minus_CHIRPS_MAE":
            gsmap_m["MAE"]
            - chirps_m["MAE"],

        "CHIRPS_RMSE":
            chirps_m["RMSE"],

        "GSMaP_RMSE":
            gsmap_m["RMSE"],

        "GSMaP_minus_CHIRPS_RMSE":
            gsmap_m["RMSE"]
            - chirps_m["RMSE"],

        "CHIRPS_R2":
            chirps_m["R2"],

        "GSMaP_R2":
            gsmap_m["R2"],

        "GSMaP_minus_CHIRPS_R2":
            gsmap_m["R2"]
            - chirps_m["R2"],
    })


rainfall_condition_df = pd.DataFrame(
    condition_results
)


print()

print(
    rainfall_condition_df.to_string(
        index=False
    )
)


# ============================================================
# 22. SOURCE-SPECIFIC RAINFALL DISTRIBUTION
# ============================================================

print()
print("=" * 90)
print("RAINFALL SOURCE DISTRIBUTION")
print("=" * 90)


rainfall_distribution = pd.DataFrame({

    "Metric": [
        "Mean",
        "Median",
        "Std",
        "Min",
        "25%",
        "75%",
        "Max",
    ],

    "CHIRPS": [
        paired_df[
            "CHIRPS_Rainfall"
        ].mean(),

        paired_df[
            "CHIRPS_Rainfall"
        ].median(),

        paired_df[
            "CHIRPS_Rainfall"
        ].std(),

        paired_df[
            "CHIRPS_Rainfall"
        ].min(),

        paired_df[
            "CHIRPS_Rainfall"
        ].quantile(
            0.25
        ),

        paired_df[
            "CHIRPS_Rainfall"
        ].quantile(
            0.75
        ),

        paired_df[
            "CHIRPS_Rainfall"
        ].max(),
    ],

    "GSMaP": [
        paired_df[
            "GSMaP_Rainfall"
        ].mean(),

        paired_df[
            "GSMaP_Rainfall"
        ].median(),

        paired_df[
            "GSMaP_Rainfall"
        ].std(),

        paired_df[
            "GSMaP_Rainfall"
        ].min(),

        paired_df[
            "GSMaP_Rainfall"
        ].quantile(
            0.25
        ),

        paired_df[
            "GSMaP_Rainfall"
        ].quantile(
            0.75
        ),

        paired_df[
            "GSMaP_Rainfall"
        ].max(),
    ],
})


print()

print(
    rainfall_distribution.to_string(
        index=False
    )
)


# ============================================================
# 23. MODEL IMPROVEMENT PERCENTAGES
# ============================================================

mae_difference = (
    gsmap_metrics["MAE"]
    - chirps_metrics["MAE"]
)

rmse_difference = (
    gsmap_metrics["RMSE"]
    - chirps_metrics["RMSE"]
)

r2_difference = (
    gsmap_metrics["R2"]
    - chirps_metrics["R2"]
)


mae_percent_change = (
    mae_difference
    / chirps_metrics["MAE"]
    * 100
)

rmse_percent_change = (
    rmse_difference
    / chirps_metrics["RMSE"]
    * 100
)


print()
print("=" * 90)
print("OVERALL DIFFERENCE")
print("=" * 90)


print()
print(
    f"GSMaP - CHIRPS MAE: "
    f"{mae_difference:+.10f}"
)

print(
    f"MAE percentage change: "
    f"{mae_percent_change:+.4f}%"
)

print()

print(
    f"GSMaP - CHIRPS RMSE: "
    f"{rmse_difference:+.10f}"
)

print(
    f"RMSE percentage change: "
    f"{rmse_percent_change:+.4f}%"
)

print()

print(
    f"GSMaP - CHIRPS R2: "
    f"{r2_difference:+.10f}"
)


# ============================================================
# 24. DECISION LOGIC
# ============================================================

print()
print("=" * 90)
print("PRELIMINARY DECISION")
print("=" * 90)


alpha = 0.05


if (
    mae_difference < 0
    and rmse_difference < 0
    and r2_difference > 0
):

    preliminary_result = (
        "GSMaP has better aggregate test metrics."
    )

else:

    preliminary_result = (
        "CHIRPS has better aggregate test metrics."
    )


if wilcoxon_result is not None:

    if (
        wilcoxon_result.pvalue
        < alpha
    ):

        significance_text = (
            "The paired Wilcoxon test is significant "
            "at alpha=0.05."
        )

    else:

        significance_text = (
            "The paired Wilcoxon test is NOT significant "
            "at alpha=0.05."
        )

else:

    significance_text = (
        "Wilcoxon significance could not be evaluated."
    )


print()
print(
    preliminary_result
)

print(
    significance_text
)

print()
print(
    "IMPORTANT: A statistically significant result, "
    "if present, should not automatically be interpreted "
    "as practically important. Effect size and operational "
    "data availability must also be considered."
)


# ============================================================
# 25. SAVE OUTPUTS
# ============================================================

paired_df.to_csv(
    PAIRED_FILE,
    index=False
)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)

year_results_df.to_csv(
    YEAR_FILE,
    index=False
)

location_results_df.to_csv(
    LOCATION_FILE,
    index=False
)

rainfall_condition_df.to_csv(
    RAINFALL_CONDITION_FILE,
    index=False
)

win_count_df.to_csv(
    WIN_COUNT_FILE,
    index=False
)


bootstrap_df = pd.DataFrame({

    "Metric": [
        "Mean absolute-error difference",
        "Bootstrap 95% CI lower",
        "Bootstrap 95% CI upper",
    ],

    "Value": [
        bootstrap_mean,
        bootstrap_lower,
        bootstrap_upper,
    ],

})

bootstrap_df.to_csv(
    BOOTSTRAP_FILE,
    index=False
)


# ============================================================
# 26. REPORT
# ============================================================

report_lines = [

    "AGRIVISION AI - CHIRPS vs GSMaP FORMAL MODEL COMPARISON",

    "Evaluation period: 2024–2025",

    "",

    "TEST SET",

    f"Samples: {len(y_test):,}",

    "Same point_id + Week_Start observations used by both models.",

    "Actual NDVI targets verified identical.",

    "",

    "OVERALL PERFORMANCE",

    f"CHIRPS MAE: {chirps_metrics['MAE']:.10f}",

    f"GSMaP MAE: {gsmap_metrics['MAE']:.10f}",

    f"GSMaP minus CHIRPS MAE: {mae_difference:+.10f}",

    f"MAE percentage change: {mae_percent_change:+.6f}%",

    "",

    f"CHIRPS RMSE: {chirps_metrics['RMSE']:.10f}",

    f"GSMaP RMSE: {gsmap_metrics['RMSE']:.10f}",

    f"GSMaP minus CHIRPS RMSE: {rmse_difference:+.10f}",

    f"RMSE percentage change: {rmse_percent_change:+.6f}%",

    "",

    f"CHIRPS R2: {chirps_metrics['R2']:.10f}",

    f"GSMaP R2: {gsmap_metrics['R2']:.10f}",

    f"GSMaP minus CHIRPS R2: {r2_difference:+.10f}",

    "",

    "PAIRED WIN COUNTS",

    f"GSMaP lower absolute error: "
    f"{gsmap_better_count:,} "
    f"({gsmap_better_count / len(y_test) * 100:.4f}%)",

    f"CHIRPS lower absolute error: "
    f"{chirps_better_count:,} "
    f"({chirps_better_count / len(y_test) * 100:.4f}%)",

    f"Equal absolute error: "
    f"{tie_count:,} "
    f"({tie_count / len(y_test) * 100:.4f}%)",

    "",

    "PAIRED ABSOLUTE-ERROR DIFFERENCE",

    f"Mean: "
    f"{difference_statistics['Mean absolute error difference']:.10f}",

    f"Median: "
    f"{difference_statistics['Median absolute error difference']:.10f}",

    "",

    "PAIRED T-TEST",

    f"Statistic: {paired_t.statistic:.10f}",

    f"p-value: {paired_t.pvalue:.10e}",

    "",

    "WILCOXON SIGNED-RANK TEST",

]

if wilcoxon_result is not None:

    report_lines.extend([

        f"Statistic: "
        f"{wilcoxon_result.statistic:.10f}",

        f"p-value: "
        f"{wilcoxon_result.pvalue:.10e}",

    ])

else:

    report_lines.append(
        "Not applicable."
    )


report_lines.extend([

    "",

    "BOOTSTRAP",

    f"Mean difference: "
    f"{bootstrap_mean:.10f}",

    f"95% CI lower: "
    f"{bootstrap_lower:.10f}",

    f"95% CI upper: "
    f"{bootstrap_upper:.10f}",

    "",

    "PRELIMINARY DECISION",

    preliminary_result,

    significance_text,

    "",

    "FINAL INTERPRETATION SHOULD CONSIDER",

    "1. Aggregate predictive performance",

    "2. Paired error distribution",

    "3. Year-wise consistency",

    "4. Location-wise consistency",

    "5. Rainfall-condition behavior",

    "6. Statistical significance",

    "7. Operational rainfall-data availability",

    "",

    "OUTPUT DIRECTORY",

    str(OUTPUT_DIR),

])


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(
            report_lines
        )
    )


# ============================================================
# 27. FINAL CONSOLE SUMMARY
# ============================================================

print()
print("=" * 90)
print("FORMAL CHIRPS vs GSMaP COMPARISON COMPLETED")
print("=" * 90)


print()
print("OVERALL:")

print(
    f"CHIRPS -> "
    f"MAE {chirps_metrics['MAE']:.8f}, "
    f"RMSE {chirps_metrics['RMSE']:.8f}, "
    f"R2 {chirps_metrics['R2']:.8f}"
)

print(
    f"GSMaP  -> "
    f"MAE {gsmap_metrics['MAE']:.8f}, "
    f"RMSE {gsmap_metrics['RMSE']:.8f}, "
    f"R2 {gsmap_metrics['R2']:.8f}"
)

print()
print(
    f"GSMaP lower-error observations: "
    f"{gsmap_better_count:,}"
)

print(
    f"CHIRPS lower-error observations: "
    f"{chirps_better_count:,}"
)

print(
    f"Ties: {tie_count:,}"
)

print()
print(
    f"Wilcoxon p-value: "
    f"{wilcoxon_result.pvalue:.10e}"
    if wilcoxon_result is not None
    else "Wilcoxon p-value: N/A"
)

print(
    f"Bootstrap 95% CI for "
    f"(GSMaP absolute error - CHIRPS absolute error): "
    f"[{bootstrap_lower:.10f}, {bootstrap_upper:.10f}]"
)

print()
print("OUTPUT FILES:")

print(PAIRED_FILE)
print(SUMMARY_FILE)
print(YEAR_FILE)
print(LOCATION_FILE)
print(RAINFALL_CONDITION_FILE)
print(WIN_COUNT_FILE)
print(BOOTSTRAP_FILE)
print(REPORT_FILE)

print()
print("=" * 90)
print("DONE")
print("=" * 90)