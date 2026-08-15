import os
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def perform_feature_engineering():
    base_dir = Path(__file__).resolve().parent.parent
    dataset_path = base_dir / 'dataset' / 'AgriVision_Weekly_Dataset_2021_2025_IMPROVED.xlsx'
    output_dir = base_dir / 'ml' / 'feature_outputs'
    output_dataset_path = base_dir / 'dataset' / 'AgriVision_Feature_Engineered.xlsx'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Loading dataset from {dataset_path}")
    df = pd.read_excel(dataset_path)
    
    original_rows = len(df)
    
    df['Week_Start'] = pd.to_datetime(df['Week_Start'])
    df = df.sort_values('Week_Start').reset_index(drop=True)
    
    # Target definition: NDVI_next_week
    # Shift NDVI by -1, but only keep it if the next row is exactly 7 days later
    df['next_week_date'] = df['Week_Start'].shift(-1)
    valid_next = (df['next_week_date'] - df['Week_Start']).dt.days == 7
    df['NDVI_next_week'] = np.where(valid_next, df['NDVI'].shift(-1), np.nan)
    
    rows_with_target = df['NDVI_next_week'].notna().sum()
    rows_removed_due_to_gaps = len(df) - rows_with_target  # these are essentially the rows right before a gap
    
    # Lag features
    lags = [1, 2, 3, 4]
    features_to_lag = ['NDVI', 'Rainfall_mm', 'Temperature_C', 'LST_C']
    
    for feature in features_to_lag:
        for lag in lags:
            df[f'prev_date_lag_{lag}'] = df['Week_Start'].shift(lag)
            valid_lag = (df['Week_Start'] - df[f'prev_date_lag_{lag}']).dt.days == 7 * lag
            df[f'{feature}_lag_{lag}'] = np.where(valid_lag, df[feature].shift(lag), np.nan)
            
    # Clean up intermediate lag columns
    df.drop(columns=[f'prev_date_lag_{lag}' for lag in lags], inplace=True)
            
    # Rolling features (window=4)
    # A rolling 4 includes the current week + 3 previous weeks.
    # It is valid only if lag_3 is valid (i.e. exactly 21 days ago).
    # We will compute rolling normally, then mask out invalid ones.
    valid_rolling_4 = (df['Week_Start'] - df['Week_Start'].shift(3)).dt.days == 21
    
    df['NDVI_rolling_mean_4'] = np.where(valid_rolling_4, df['NDVI'].rolling(window=4).mean(), np.nan)
    df['NDVI_rolling_std_4'] = np.where(valid_rolling_4, df['NDVI'].rolling(window=4).std(), np.nan)
    
    df['Rainfall_rolling_sum_4'] = np.where(valid_rolling_4, df['Rainfall_mm'].rolling(window=4).sum(), np.nan)
    df['Rainfall_rolling_mean_4'] = np.where(valid_rolling_4, df['Rainfall_mm'].rolling(window=4).mean(), np.nan)
    
    df['Temperature_rolling_mean_4'] = np.where(valid_rolling_4, df['Temperature_C'].rolling(window=4).mean(), np.nan)
    df['LST_rolling_mean_4'] = np.where(valid_rolling_4, df['LST_C'].rolling(window=4).mean(), np.nan)
    
    # Seasonal features
    df['month'] = df['Week_Start'].dt.month
    df['week_of_year'] = df['Week_Start'].dt.isocalendar().week.astype(float) # Using ISO week
    
    df['sin_week'] = np.sin(2 * np.pi * df['week_of_year'] / 52.0)
    df['cos_week'] = np.cos(2 * np.pi * df['week_of_year'] / 52.0)
    
    # Drop temp columns
    df.drop(columns=['next_week_date'], inplace=True)
    
    # Dataset quality after FE
    final_rows = len(df)
    missing_values = df.isnull().sum().to_dict()
    duplicate_rows = int(df.duplicated().sum())
    duplicate_weeks = int(df.duplicated(subset=['Week_Start']).sum())
    
    date_start = str(df['Week_Start'].min().date())
    date_end = str(df['Week_Start'].max().date())
    
    # Target stats
    target_stats = df['NDVI_next_week'].describe().to_dict()
    
    # Correlations
    candidate_predictors = [
        'NDVI', 'NDVI_lag_1', 'NDVI_lag_2', 'NDVI_lag_4', 
        'Rainfall_mm', 'Temperature_C', 'LST_C', 
        'Rainfall_rolling_sum_4', 'Temperature_rolling_mean_4', 'LST_rolling_mean_4'
    ]
    correlations = {}
    for predictor in candidate_predictors:
        if predictor in df.columns:
            # Dropna for correlation computation
            corr_val = df['NDVI_next_week'].corr(df[predictor])
            correlations[predictor] = corr_val
            
    # Propose split boundaries
    # Easiest way: drop rows with missing target or essential features? 
    # The prompt doesn't say to drop them in the final exported dataset, but let's identify split boundaries
    # 2021-2023 -> Train (roughly 60%), 2024 -> Val (20%), 2025 -> Test (20%)
    train_end = "2023-12-31"
    val_end = "2024-12-31"
    
    split_info = {
        "Train": f"<= {train_end}",
        "Validation": f"> {train_end} and <= {val_end}",
        "Test": f"> {val_end}"
    }
    
    # Save files
    df.to_excel(output_dataset_path, index=False)
    logging.info(f"Feature-engineered dataset saved to {output_dataset_path}")
    
    # Output to ml/feature_outputs/
    summary = {
        "original_rows": original_rows,
        "feature_engineered_rows": final_rows,
        "rows_with_valid_target": int(rows_with_target),
        "rows_removed_due_to_temporal_gaps": int(rows_removed_due_to_gaps),
        "duplicate_rows": duplicate_rows,
        "duplicate_Week_Start": duplicate_weeks,
        "date_range": {"start": date_start, "end": date_end},
        "target_statistics": target_stats,
        "split_boundaries": split_info
    }
    
    with open(output_dir / 'feature_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
        
    df.describe().to_csv(output_dir / 'feature_statistics.csv')
    df['NDVI_next_week'].describe().to_csv(output_dir / 'target_statistics.csv')
    pd.Series(missing_values).to_csv(output_dir / 'feature_missing_values.csv')
    pd.Series(correlations).to_csv(output_dir / 'target_correlations.csv')
    
    # Final feature list
    feature_list = df.columns.tolist()
    
    with open(output_dir / 'feature_engineering_report.txt', 'w') as f:
        f.write("=== TARGET DEFINITION & TEMPORAL FEATURE ENGINEERING ===\n")
        f.write(f"Original Rows: {original_rows}\n")
        f.write(f"Rows with valid target: {rows_with_target}\n")
        f.write(f"Rows missing target due to gaps/end: {rows_removed_due_to_gaps}\n")
        f.write("\nFinal Features List:\n")
        for ft in feature_list:
            f.write(f"- {ft}\n")
        f.write("\nTarget Correlations:\n")
        for k, v in correlations.items():
            f.write(f"{k}: {v}\n")
            
    logging.info("Feature engineering completed.")

if __name__ == "__main__":
    perform_feature_engineering()
