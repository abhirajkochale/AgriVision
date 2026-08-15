import os
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as plt_sns
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def perform_eda():
    # Setup paths
    base_dir = Path(__file__).resolve().parent.parent
    dataset_path = base_dir / 'dataset' / 'AgriVision_Weekly_Dataset_2021_2025_IMPROVED.xlsx'
    output_dir = base_dir / 'ml' / 'eda_outputs'
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1 & 2. Validate file existence
    if not dataset_path.exists():
        logging.error(f"Dataset not found at {dataset_path}")
        report_missing = {
            "error": "Dataset missing",
            "path": str(dataset_path)
        }
        with open(output_dir / 'eda_summary.json', 'w') as f:
            json.dump(report_missing, f, indent=4)
        return False

    logging.info(f"Loading dataset from {dataset_path}")
    df = pd.read_excel(dataset_path)

    # 3. Validate expected columns
    expected_cols = [
        'Week_Start', 'Week_End', 'Week_Number', 'Year',
        'NDVI', 'Rainfall_mm', 'Temperature_C', 'LST_C',
        'Sentinel2_Images', 'CHIRPS_Images', 'ERA5_Images', 'MODIS_LST_Images',
        'study_area'
    ]
    
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        logging.warning(f"Missing expected columns: {missing_cols}")
    
    # 4 & 5. Parse dates and validate types
    df['Week_Start'] = pd.to_datetime(df['Week_Start'])
    df['Week_End'] = pd.to_datetime(df['Week_End'])
    
    # Sort chronologically just to be safe for gap checking (but we check ordering first)
    is_sorted = df['Week_Start'].is_monotonic_increasing
    
    # 6. Check missing values
    missing_values = df.isnull().sum().to_dict()
    
    # 7. Check duplicate rows
    duplicate_rows = int(df.duplicated().sum())
    
    # 8. Check duplicate Week_Start
    duplicate_weeks = int(df.duplicated(subset=['Week_Start']).sum())
    
    # 10. Check date gaps
    df_sorted = df.sort_values('Week_Start').reset_index(drop=True)
    date_diffs = df_sorted['Week_Start'].diff()
    # A normal weekly gap is 7 days. Greater than 7 means a gap.
    gaps = df_sorted[date_diffs > pd.Timedelta(days=7)]
    gap_records = []
    for idx, row in gaps.iterrows():
        prev_date = df_sorted.loc[idx-1, 'Week_Start']
        curr_date = row['Week_Start']
        gap_records.append(f"{prev_date.date()} -> {curr_date.date()}")
    
    # 11 & 12. Generate descriptive statistics and numerical range checks
    numeric_cols = ['NDVI', 'Rainfall_mm', 'Temperature_C', 'LST_C']
    stats = df[numeric_cols].describe().to_dict()
    
    # 13. Correlation matrix
    corr_matrix = df[numeric_cols].corr()
    corr_dict = corr_matrix.to_dict()
    
    # 24. Produce machine-readable EDA summary
    eda_summary = {
        "dataset_shape": df.shape,
        "is_chronologically_sorted": bool(is_sorted),
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "duplicate_Week_Start": duplicate_weeks,
        "date_range": {
            "start": str(df['Week_Start'].min().date()),
            "end": str(df['Week_Start'].max().date())
        },
        "identified_gaps": gap_records,
        "statistics": stats,
        "correlations": corr_dict
    }
    
    with open(output_dir / 'eda_summary.json', 'w') as f:
        json.dump(eda_summary, f, indent=4)
        
    # Save descriptive statistics and correlations to CSV
    df[numeric_cols].describe().to_csv(output_dir / 'descriptive_statistics.csv')
    corr_matrix.to_csv(output_dir / 'correlation_matrix.csv')
    
    # Yearly counts
    yearly_counts = df['Year'].value_counts().sort_index().reset_index()
    yearly_counts.columns = ['Year', 'Count']
    yearly_counts.to_csv(output_dir / 'yearly_counts.csv', index=False)
    
    # Weekly gaps
    gaps_df = pd.DataFrame({'Missing_Period': gap_records})
    gaps_df.to_csv(output_dir / 'weekly_gaps.csv', index=False)
        
    # 25. Produce a human-readable EDA report
    with open(output_dir / 'eda_report.txt', 'w') as f:
        f.write("=== AGRI-VISION AI EDA REPORT ===\n\n")
        f.write(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
        f.write(f"Chronologically Sorted: {is_sorted}\n")
        f.write(f"Duplicate Rows: {duplicate_rows}\n")
        f.write(f"Duplicate Week_Start: {duplicate_weeks}\n")
        f.write(f"Date Range: {df['Week_Start'].min().date()} to {df['Week_Start'].max().date()}\n\n")
        f.write("Identified Date Gaps (>7 days):\n")
        for gap in gap_records:
            f.write(f" - {gap}\n")
        f.write("\nMissing Values:\n")
        for k, v in missing_values.items():
            f.write(f" - {k}: {v}\n")
        f.write("\nStatistics:\n")
        f.write(df[numeric_cols].describe().to_string())
        f.write("\n\nCorrelations:\n")
        f.write(corr_matrix.to_string())
        
    # --- PLOTTING ---
    plt_sns.set_theme(style="whitegrid")
    
    # 14-17. Analyze distributions and 22. Feature distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Feature Distributions')
    plt_sns.histplot(df['NDVI'], kde=True, ax=axes[0, 0], color='green').set_title('NDVI Distribution')
    plt_sns.histplot(df['Rainfall_mm'], kde=True, ax=axes[0, 1], color='blue').set_title('Rainfall (mm) Distribution')
    plt_sns.histplot(df['Temperature_C'], kde=True, ax=axes[1, 0], color='orange').set_title('Temperature (°C) Distribution')
    plt_sns.histplot(df['LST_C'], kde=True, ax=axes[1, 1], color='red').set_title('LST (°C) Distribution')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_distributions.png')
    plt.close()
    
    # 18-21. Generate temporal trends
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    fig.suptitle('Temporal Trends (2021-2025)')
    
    axes[0].plot(df_sorted['Week_Start'], df_sorted['NDVI'], color='green', marker='o', markersize=2, linestyle='-')
    axes[0].set_title('NDVI Trend')
    axes[0].set_ylabel('NDVI')
    
    axes[1].plot(df_sorted['Week_Start'], df_sorted['Rainfall_mm'], color='blue', marker='o', markersize=2, linestyle='-')
    axes[1].set_title('Rainfall Trend')
    axes[1].set_ylabel('Rainfall (mm)')
    
    axes[2].plot(df_sorted['Week_Start'], df_sorted['Temperature_C'], color='orange', marker='o', markersize=2, linestyle='-')
    axes[2].set_title('Temperature Trend')
    axes[2].set_ylabel('Temperature (°C)')
    
    axes[3].plot(df_sorted['Week_Start'], df_sorted['LST_C'], color='red', marker='o', markersize=2, linestyle='-')
    axes[3].set_title('LST Trend')
    axes[3].set_ylabel('LST (°C)')
    axes[3].set_xlabel('Date')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'temporal_trends.png')
    plt.close()
    
    # 23. Generate correlation heatmap
    plt.figure(figsize=(8, 6))
    plt_sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".3f", vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png')
    plt.close()
    
    logging.info("EDA completed successfully. Outputs saved to ml/eda_outputs/")
    return True

if __name__ == "__main__":
    perform_eda()
