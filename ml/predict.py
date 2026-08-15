import os
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NDVIModelPredictor:
    def __init__(self, model_dir=None):
        if model_dir is None:
            self.model_dir = Path(__file__).resolve().parent / 'model_outputs'
        else:
            self.model_dir = Path(model_dir)
            
        self.rf_model_path = self.model_dir / 'random_forest_model.joblib'
        self.features_path = self.model_dir / 'model_features.json'
        
        self.rf_model = None
        self.feature_list = None
        
        self._load_artifacts()
        
    def _load_artifacts(self):
        if not self.rf_model_path.exists() or not self.features_path.exists():
            raise FileNotFoundError("Model artifacts not found. Please ensure the model has been trained and saved.")
            
        self.rf_model = joblib.load(self.rf_model_path)
        with open(self.features_path, 'r') as f:
            self.feature_list = json.load(f)
            
        logging.info("Random Forest model and feature list loaded successfully.")
        
    def predict_rf(self, input_data: dict) -> float:
        """
        Predict NDVI_next_week using the Random Forest model.
        input_data must contain all keys defined in self.feature_list
        """
        missing_features = [f for f in self.feature_list if f not in input_data]
        if missing_features:
            raise ValueError(f"Missing required features for RF inference: {missing_features}")
            
        # Create a DataFrame with exactly one row and preserve exact feature order
        df_input = pd.DataFrame([input_data])[self.feature_list]
        
        prediction = self.rf_model.predict(df_input)
        return float(prediction[0])
        
    def predict_persistence(self, current_ndvi: float) -> float:
        """
        Predict NDVI_next_week using the Persistence baseline.
        """
        if current_ndvi is None:
            raise ValueError("Current NDVI is required for persistence prediction.")
        return float(current_ndvi)

def verify_model():
    """
    Verify the saved model by predicting on a known test-set row.
    """
    base_dir = Path(__file__).resolve().parent.parent
    dataset_path = base_dir / 'dataset' / 'AgriVision_Feature_Engineered.xlsx'
    
    logging.info("Loading feature-engineered dataset for verification...")
    df = pd.read_excel(dataset_path)
    df['Week_Start'] = pd.to_datetime(df['Week_Start'])
    
    # Get a test set row (e.g. from 2025)
    test_rows = df[(df['Week_Start'] > "2024-12-31") & df['NDVI_next_week'].notna()]
    if test_rows.empty:
        logging.warning("No test rows found for verification.")
        return False
        
    # Pick the first test row
    test_row = test_rows.iloc[0]
    
    predictor = NDVIModelPredictor()
    
    # Prepare input dictionary
    input_dict = test_row[predictor.feature_list].to_dict()
    
    rf_pred = predictor.predict_rf(input_dict)
    pers_pred = predictor.predict_persistence(input_dict.get('NDVI'))
    actual = test_row['NDVI_next_week']
    
    logging.info("=== VERIFICATION RESULT ===")
    logging.info(f"Target Date: {test_row['Week_Start'].date()}")
    logging.info(f"Actual NDVI_next_week: {actual:.4f}")
    logging.info(f"Random Forest Prediction: {rf_pred:.4f}")
    logging.info(f"Persistence Prediction: {pers_pred:.4f}")
    
    # Save verification report
    report_path = predictor.model_dir / 'model_artifact_report.txt'
    with open(report_path, 'w') as f:
        f.write("=== MODEL ARTIFACT REPORT ===\n\n")
        f.write(f"Model Type: RandomForestRegressor\n")
        f.write(f"Hyperparameters: {predictor.rf_model.get_params()}\n")
        f.write(f"Feature Count: {len(predictor.feature_list)}\n")
        f.write(f"Exact Feature List: {predictor.feature_list}\n")
        f.write(f"Model File Path: {predictor.rf_model_path}\n")
        f.write(f"Inference Script Path: {Path(__file__).resolve()}\n")
        f.write(f"Inference Schema Path: {predictor.model_dir / 'inference_schema.json'}\n\n")
        f.write("=== VERIFICATION RESULT ===\n")
        f.write(f"Actual NDVI_next_week: {actual:.4f}\n")
        f.write(f"Random Forest Prediction: {rf_pred:.4f}\n")
        f.write(f"Persistence Prediction: {pers_pred:.4f}\n")
        
    logging.info(f"Model artifact report saved to {report_path}")
    return True

if __name__ == "__main__":
    verify_model()
