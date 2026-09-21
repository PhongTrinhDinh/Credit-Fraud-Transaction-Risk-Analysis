import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

CATEGORICAL_FEATURES = [
    'merchant_category', 'card_type', 'auth_method', 'channel', 
    'device_type', 'is_foreign_transaction', 'is_new_merchant', 
    'used_vpn', 'ip_country_mismatch', 'billing_shipping_mismatch', 
    'is_ai_generated_scam_attempt'
]

FEATURES_TO_CHECK = [
    'amount_usd',
    'account_balance_usd',
    'distance_from_home_km',
    'hours_since_last_txn',
    'velocity_score',
    'txn_count_last_24h'
]

def build_preprocessor():
    cat_pipeline = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[('cat', cat_pipeline, CATEGORICAL_FEATURES)],
        remainder='passthrough'
    )
    
    return preprocessor

def save_preprocessor(preprocessor, filepath='models/preprocessor.pkl'):
    joblib.dump(preprocessor, filepath)
    print(f"Saved Preprocessor in {filepath}")
    
def load_preprocessor(filepath='models/preprocessor.pkl'):
    return joblib.load(filepath)