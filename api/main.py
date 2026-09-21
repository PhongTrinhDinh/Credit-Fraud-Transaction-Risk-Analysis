from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# import predict function from src
from src.predict import predict_fraud_transaction

# initialize FastAPI app
app = FastAPI(
    title='Fraud Detection API',
    description='Real-time credit card fraud detection API using XGBoost',
    version="1.0.0"
)

class Transaction(BaseModel):
    amount_usd: float = Field(..., description='Transaction amount (USD)')
    merchant_category: str = Field(..., description='Category of the merchant')
    card_type: str = Field(..., description='Type of the credit card')
    auth_method: str = Field(..., description='Authentication method')
    channel: str = Field(..., description='Transaction channel')
    device_type: str = Field(..., description='Type of device used')
    is_foreign_transaction: bool = Field(..., description='Is it a foreign transaction')
    hours_since_last_txn: float = Field(..., description='Hours since last transaction')
    txn_count_last_24h: int = Field(..., description='Transaction count in last 24 hours')
    distance_from_home_km: float = Field(..., description='Distance from home in km')
    card_age_months: int = Field(..., description='Card age in months')
    customer_age: int = Field(..., description='Customer age in years')
    account_balance_usd: float = Field(..., description='Account balance in USD')
    is_new_merchant: bool = Field(..., description='Is it a new merchant')
    used_vpn: bool = Field(..., description='Did the user use a VPN')
    ip_country_mismatch: bool = Field(..., description='IP and country mismatch')
    billing_shipping_mismatch: bool = Field(..., description='Billing and shipping mismatch')
    cvv_retry_count: int = Field(..., description='Number of CVV retries')
    velocity_score: float = Field(..., description='Velocity score')
    time_of_day_hour: int = Field(..., description='Time of day (hour)')
    day_of_week: int = Field(..., description='Day of week')
    is_ai_generated_scam_attempt: bool = Field(..., description='Is it an AI generated scam attempt')
    merchant_risk_score: float = Field(..., description='Merchant risk score')
    prior_disputes: int = Field(..., description='Number of prior disputes')

# create endpoint
@app.post("/predict")
def predict_fraud(transaction: Transaction):
    try:
        transaction_dict = transaction.model_dump()
        # Ensure we don't pass transaction_id or is_fraud if they somehow get included
        result = predict_fraud_transaction(transaction_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# endpoint health check
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Fraud Detection System is running smoothly."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)