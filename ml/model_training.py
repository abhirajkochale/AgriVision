import os
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def perform_training():
    base_dir = Path(__file__).resolve().parent.parent
    dataset_path = base_dir / 'dataset' / 'AgriVision_Feature_Engineered.xlsx'
    output_dir = base_dir / 'ml' / 'model_outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Loading dataset from {dataset_path}")
    df = pd.read_excel(dataset_path)
    df['Week_Start'] = pd.to_datetime(df['Week_Start'])
    
    target_col = 'NDVI_next_week'
    exclude_cols = ['Week_Start', 'Week_End', 'study_area', target_col]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Handle missing values: complete-case rows
    cols_to_check = feature_cols + [target_col]
    df_clean = df.dropna(subset=cols_to_check).copy()
    
    total_usable_rows = len(df_clean)
    logging.info(f"Usable modeling rows after complete-case filtering: {total_usable_rows}")
    
    # Chronological Split
    train_mask = df_clean['Week_Start'] <= "2023-12-31"
    val_mask = (df_clean['Week_Start'] > "2023-12-31") & (df_clean['Week_Start'] <= "2024-12-31")
    test_mask = df_clean['Week_Start'] > "2024-12-31"
    
    df_train = df_clean[train_mask].copy()
    df_val = df_clean[val_mask].copy()
    df_test = df_clean[test_mask].copy()
    
    train_count = len(df_train)
    val_count = len(df_val)
    test_count = len(df_test)
    
    logging.info(f"Train rows: {train_count}, Val rows: {val_count}, Test rows: {test_count}")
    
    X_train = df_train[feature_cols]
    y_train = df_train[target_col]
    X_val = df_val[feature_cols]
    y_val = df_val[target_col]
    X_test = df_test[feature_cols]
    y_test = df_test[target_col]
    
    results = {}
    
    # 1. Persistence Baseline
    # prediction = current week's NDVI
    train_persistence = df_train['NDVI']
    val_persistence = df_val['NDVI']
    test_persistence = df_test['NDVI']
    
    results["Persistence"] = {
        "Train": evaluate_model(y_train, train_persistence),
        "Validation": evaluate_model(y_val, val_persistence),
        "Test": evaluate_model(y_test, test_persistence)
    }
    
    # 2. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    results["LinearRegression"] = {
        "Train": evaluate_model(y_train, lr.predict(X_train)),
        "Validation": evaluate_model(y_val, lr.predict(X_val)),
        "Test": evaluate_model(y_test, lr.predict(X_test))
    }
    
    # 3. Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=2, random_state=42)
    rf.fit(X_train, y_train)
    results["RandomForest"] = {
        "Train": evaluate_model(y_train, rf.predict(X_train)),
        "Validation": evaluate_model(y_val, rf.predict(X_val)),
        "Test": evaluate_model(y_test, rf.predict(X_test))
    }
    
    # Feature Importance for RF
    rf_importances = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": rf.feature_importances_
    }).sort_values("Importance", ascending=False)
    rf_importances.to_csv(output_dir / "random_forest_feature_importance.csv", index=False)
    
    # Save the Random Forest Model and Feature List
    joblib.dump(rf, output_dir / 'random_forest_model.joblib')
    with open(output_dir / 'model_features.json', 'w') as f:
        json.dump(feature_cols, f, indent=4)
    
    # 4. XGBoost
    if XGB_AVAILABLE:
        xg = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
        xg.fit(X_train, y_train)
        results["XGBoost"] = {
            "Train": evaluate_model(y_train, xg.predict(X_train)),
            "Validation": evaluate_model(y_val, xg.predict(X_val)),
            "Test": evaluate_model(y_test, xg.predict(X_test))
        }
        
        xgb_importances = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": xg.feature_importances_
        }).sort_values("Importance", ascending=False)
        xgb_importances.to_csv(output_dir / "xgboost_feature_importance.csv", index=False)
        
    else:
        logging.info("XGBoost is not available. Skipping.")
        
    # Model Selection
    # Primary criterion: Lowest Test MAE
    best_model_name = None
    best_test_mae = float('inf')
    
    for model_name, metrics in results.items():
        if metrics["Test"]["MAE"] < best_test_mae:
            best_test_mae = metrics["Test"]["MAE"]
            best_model_name = model_name
            
    logging.info(f"Best model based on Test MAE: {best_model_name}")
    
    # Save training_results.json
    training_summary = {
        "xgboost_available": XGB_AVAILABLE,
        "total_usable_rows": int(total_usable_rows),
        "split_counts": {
            "Train": int(train_count),
            "Validation": int(val_count),
            "Test": int(test_count)
        },
        "split_dates": {
            "Train": {"start": str(df_train['Week_Start'].min().date()), "end": str(df_train['Week_Start'].max().date())},
            "Validation": {"start": str(df_val['Week_Start'].min().date()), "end": str(df_val['Week_Start'].max().date())},
            "Test": {"start": str(df_test['Week_Start'].min().date()), "end": str(df_test['Week_Start'].max().date())}
        },
        "features": feature_cols,
        "target": target_col,
        "results": results,
        "best_model": best_model_name
    }
    
    with open(output_dir / 'training_results.json', 'w') as f:
        json.dump(training_summary, f, indent=4)
        
    # model_comparison.csv
    comp_rows = []
    for m, m_data in results.items():
        comp_rows.append({
            "Model": m,
            "Train_MAE": m_data["Train"]["MAE"],
            "Train_RMSE": m_data["Train"]["RMSE"],
            "Train_R2": m_data["Train"]["R2"],
            "Val_MAE": m_data["Validation"]["MAE"],
            "Val_RMSE": m_data["Validation"]["RMSE"],
            "Val_R2": m_data["Validation"]["R2"],
            "Test_MAE": m_data["Test"]["MAE"],
            "Test_RMSE": m_data["Test"]["RMSE"],
            "Test_R2": m_data["Test"]["R2"]
        })
    pd.DataFrame(comp_rows).to_csv(output_dir / "model_comparison.csv", index=False)
    
    # Generate predictions for the best model on test set
    if best_model_name == "Persistence":
        best_preds = test_persistence
    elif best_model_name == "LinearRegression":
        best_preds = lr.predict(X_test)
    elif best_model_name == "RandomForest":
        best_preds = rf.predict(X_test)
    elif best_model_name == "XGBoost":
        best_preds = xg.predict(X_test)
        
    test_out = df_test[['Week_Start', target_col]].copy()
    test_out.rename(columns={target_col: 'NDVI_actual'}, inplace=True)
    test_out['NDVI_predicted'] = best_preds
    test_out['prediction_error'] = test_out['NDVI_predicted'] - test_out['NDVI_actual']
    test_out.to_csv(output_dir / 'test_predictions.csv', index=False)
    
    # PLOTS
    # Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(test_out['NDVI_actual'], test_out['NDVI_predicted'], alpha=0.7)
    # 1:1 line
    min_val = min(test_out['NDVI_actual'].min(), test_out['NDVI_predicted'].min())
    max_val = max(test_out['NDVI_actual'].max(), test_out['NDVI_predicted'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('Actual NDVI_next_week')
    plt.ylabel('Predicted NDVI_next_week')
    plt.title(f'Actual vs Predicted ({best_model_name})')
    plt.grid(True)
    plt.savefig(output_dir / 'actual_vs_predicted.png')
    plt.close()
    
    # Residual Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(test_out['NDVI_predicted'], test_out['prediction_error'], alpha=0.7)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Predicted NDVI_next_week')
    plt.ylabel('Residual (Predicted - Actual)')
    plt.title(f'Residual Plot ({best_model_name})')
    plt.grid(True)
    plt.savefig(output_dir / 'residual_plot.png')
    plt.close()
    
    # model_training_report.txt
    with open(output_dir / 'model_training_report.txt', 'w') as f:
        f.write("=== ML MODEL TRAINING & EVALUATION REPORT ===\n\n")
        f.write(f"Total usable rows: {total_usable_rows}\n")
        f.write(f"Train/Val/Test Split: {train_count} / {val_count} / {test_count}\n")
        f.write(f"Features used ({len(feature_cols)}): {feature_cols}\n\n")
        f.write("--- MODEL PERFORMANCE ---\n")
        for row in comp_rows:
            f.write(f"Model: {row['Model']}\n")
            f.write(f"  Train MAE: {row['Train_MAE']:.4f}, RMSE: {row['Train_RMSE']:.4f}, R2: {row['Train_R2']:.4f}\n")
            f.write(f"  Val MAE  : {row['Val_MAE']:.4f}, RMSE: {row['Val_RMSE']:.4f}, R2: {row['Val_R2']:.4f}\n")
            f.write(f"  Test MAE : {row['Test_MAE']:.4f}, RMSE: {row['Test_RMSE']:.4f}, R2: {row['Test_R2']:.4f}\n\n")
        f.write(f"Best Model Selected: {best_model_name}\n")
        
    logging.info("Training completed successfully.")

if __name__ == "__main__":
    perform_training()
