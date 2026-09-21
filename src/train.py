import pandas as pd
import numpy as np
import xgboost as xgb
import json
from preprocessed import build_preprocessor, save_preprocessor
from sklearn.model_selection import train_test_split

C_FP = 100.0  
FEE_CHARGEBACK = 25.0       
FEE_PENALTY = 15.0

def calculate_weights(y, amounts):
    weights = np.ones(len(y))
    weights[y == 0] = C_FP
    weights[y == 1] = amounts[y == 1] + FEE_CHARGEBACK + FEE_PENALTY
    return weights / np.mean(weights)

def get_optimal_threshold(y_true, y_prob, amounts):
    thresholds = np.arange(0.01, 1.00, 0.01)
    costs = []
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cost_fp = ((y_true == 0) & (y_pred == 1)).sum() * C_FP
        fn_mask = (y_true == 1) & (y_pred == 0)
        cost_fn = amounts[fn_mask].sum() + (fn_mask.sum() * (FEE_CHARGEBACK + FEE_PENALTY))
        costs.append(cost_fp + cost_fn)
        
    best_idx = np.argmin(costs)
    return thresholds[best_idx]

def run_training_pipeline():
    print("1. Reading data...")
    df = pd.read_csv("data/credit_card_fraud_2026.csv")
    if 'transaction_id' in df.columns:
        df = df.drop(columns=['transaction_id'])
    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    print("2. Preprocessing...")
    preprocessor = build_preprocessor()
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    save_preprocessor(preprocessor, "models/preprocessor.pkl")
    
    print("3. Calculate weights and training XGBoost...")
    train_weights = calculate_weights(y_train, X_train['amount_usd'].values)
    
    # Khởi tạo mô hình với các tham số tốt nhất (đã tune từ Optuna)
    model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train_processed, y_train, sample_weight=train_weights)
    
    # LƯU MÔ HÌNH XGBOOST (Định dạng JSON là an toàn và nhanh nhất)
    model.save_model("models/xgb_fraud_model.json")
    print("Saved XGBoost in models/xgb_fraud_model.json")
    
    print("4. Finding Optimal Threshold...")
    y_prob_test = model.predict_proba(X_test_processed)[:, 1]
    optimal_thresh = float(get_optimal_threshold(y_test.values, y_prob_test, X_test['amount_usd'].values))
    
    with open("models/threshold_config.json", "w") as f:
        json.dump({"optimal_threshold": optimal_thresh}, f)
    print(f"Đã lưu Threshold ({optimal_thresh}) tại models/threshold_config.json")

if __name__ == "__main__":
    run_training_pipeline()