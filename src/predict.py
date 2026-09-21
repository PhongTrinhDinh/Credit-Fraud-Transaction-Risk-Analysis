import pandas as pd
import xgboost as xgb
import json
import joblib

try:
    print("Starting Fraud Engine... Load Artifacts into RAM...")
    
    # 1. Load Preprocessor
    preprocessor = joblib.load("models/preprocessor.pkl")
    
    # 2. Load Model
    model = xgb.XGBClassifier()
    model.load_model("models/xgb_fraud_model.json")
    
    # 3. Load Threshold
    with open("models/threshold_config.json", "r") as f:
        config = json.load(f)
        OPTIMAL_THRESHOLD = config["optimal_threshold"]
        
    print("Done!")
    
except Exception as e:
    print(f"ERROR: Not found file model. Run train.py first. Detail: {e}")
    
def predict_fraud_transaction(transaction_dict: dict) -> dict:

    df_new = pd.DataFrame([transaction_dict])
    
    X_processed = preprocessor.transform(df_new)
    
    fraud_probability = float(model.predict_proba(X_processed)[0, 1])
    
    is_fraud = fraud_probability >= OPTIMAL_THRESHOLD

    return {
        "status": "BLOCK" if is_fraud else "APPROVE",
        "fraud_probability": round(fraud_probability, 4),
        "threshold_used": OPTIMAL_THRESHOLD
    }