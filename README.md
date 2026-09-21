# 💳 Real-time Credit Card Fraud & Transaction Risk Analysis

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.138-009688.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-F37626.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-1.7.2-F7931E.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)

An end-to-end, production-ready Machine Learning system designed to detect fraudulent credit card transactions in real-time. This project goes beyond basic classification accuracy by implementing a **Cost-Sensitive Learning** approach, mathematically optimizing the decision threshold to minimize total business financial risk (balancing the cost of manual review friction vs. the heavy damages of fraud chargebacks).

---

## ✨ Key Features

- **Cost-Sensitive Optimization**: Custom sample weights dynamically penalize false negatives (fraud escaping detection) heavily while controlling false positives to minimize customer churn.
- **Robust Preprocessing Pipeline**: Built-in handling of multivariate outliers (Isolation Forest) and rigid categorical one-hot encoding workflows utilizing `scikit-learn`'s ColumnTransformer.
- **Microservice Architecture**: Exposes a real-time inference API via **FastAPI** with highly strict **Pydantic** schema validation to protect the engine from malformed inputs.
- **Production-Ready Docker**: Containerized into a highly optimized, minimalist image using `python:3.12-slim` focusing purely on inference speed and low memory footprint.

---

## 📁 Project Structure

```text
├── api/
│   └── main.py              # FastAPI application & Pydantic Schemas
├── src/
│   ├── preprocessed.py      # Scikit-learn Pipeline (Imputation, Encoding)
│   ├── train.py             # XGBoost training & threshold tuning logic
│   └── predict.py           # Real-time inference engine
├── models/                  # Pickled transformers, JSON models, and config
├── Dockerfile               # Container configuration
├── requirements.txt         # Pinned exact versions for reproducibility
└── README.md
```

---

## 🚀 Quick Start (Running via Docker)

You don't need to install Python locally. Just ensure you have Docker installed.

**1. Clone the repository**
```bash
git clone https://github.com/PhongTrinhDinh/Credit-Fraud-Transaction-Risk-Analysis.git
cd Credit-Fraud-Transaction-Risk-Analysis
```

**2. Build the Docker Image**
```bash
docker build -t fraud-detection-api:latest .
```

**3. Run the Container**
```bash
docker run -d -p 8000:8000 --name fraud_api fraud-detection-api:latest
```

**4. Check API Health**
Visit: [http://localhost:8000/health](http://localhost:8000/health)

---

## 🧪 Testing the API

FastAPI provides an automatic, interactive UI for testing. 
Once the container is running, go to: **[http://localhost:8000/docs](http://localhost:8000/docs)**

Alternatively, you can test the prediction endpoint directly via `cURL` using this **dummy payload**:

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "amount_usd": 125.50,
  "merchant_category": "electronics",
  "card_type": "credit",
  "auth_method": "chip",
  "channel": "in_store",
  "device_type": "pos",
  "is_foreign_transaction": false,
  "hours_since_last_txn": 2.5,
  "txn_count_last_24h": 3,
  "distance_from_home_km": 15.2,
  "card_age_months": 34,
  "customer_age": 29,
  "account_balance_usd": 4500.00,
  "is_new_merchant": false,
  "used_vpn": false,
  "ip_country_mismatch": false,
  "billing_shipping_mismatch": false,
  "cvv_retry_count": 0,
  "velocity_score": 0.45,
  "time_of_day_hour": 14,
  "day_of_week": 3,
  "is_ai_generated_scam_attempt": false,
  "merchant_risk_score": 0.12,
  "prior_disputes": 0
}'
```

**Expected Response:**
```json
{
  "status": "APPROVE",
  "fraud_probability": 0.0002,
  "threshold_used": 0.17
}
```

---

## 🧠 Model Training (Optional)

If you wish to retrain the model with new data:
1. Setup a local environment and install dependencies: `pip install -r requirements.txt`
2. Ensure your dataset is placed at `data/credit_card_fraud_2026.csv`
3. Run the training pipeline:
   ```bash
   python src/train.py
   ```
   *This will update the preprocessor, XGBoost model, and the optimal threshold JSON config inside the `models/` directory.*
