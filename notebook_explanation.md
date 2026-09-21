# 📘 Credit Card Fraud & Transaction Risk Analysis (Notebook Explanation)

This document provides a detailed explanation of the algorithmic thinking, data approach, and Machine Learning model optimization process presented in the `notebook.ipynb` file. Unlike standard classification problems (which only focus on Accuracy/F1-score), this notebook solves the problem from a **Cost-Sensitive Learning** perspective, aiming to optimize business profitability.

---

## 1. Exploratory Data Analysis (EDA)

The first step is to thoroughly understand the characteristics of the `credit_card_fraud_2026.csv` dataset.
- **Target Distribution:** Financial fraud data is always in a state of **extreme class imbalance**. Fraudulent transactions (Fraud = 1) typically account for less than 1% of total transactions. Relying on Accuracy as an evaluation metric would lead to significant errors.

- **Numerical Features Distribution:** By visualizing with `seaborn.histplot`, we observe the dispersion of risk factors such as `amount_usd`, `velocity_score`, and `distance_from_home_km`.
![Numerical Features Distribution](<figures/Distribution of Numerical Features.png>)
- **Categorical Features Distribution:** Analyzing user behavior based on `device_type`, `card_type`, and `merchant_category`.
![Categorical Features Distribution](<figures/Distribution of Categorical Features.png>)

---

## 2. Data Preprocessing & Multicollinearity

- Convert `boolean` columns (True/False) to numerical format (1/0).
- Apply **One-Hot Encoding** (`pd.get_dummies` with `drop_first=True`) for categorical variables to avoid the Dummy Variable Trap.
- Visualize the **Correlation Matrix** using a Heatmap to eliminate multicollinear variables (highly correlated features), allowing the model to focus on the most important characteristics.
![Correlation Matrix](<figures/Correlation Matrix.png>)

---

## 3. Multivariate Outlier Detection

Instead of simply filtering outliers using z-scores, the notebook applies the **Isolation Forest** algorithm on a set of critical features: `amount_usd`, `account_balance_usd`, `distance_from_home_km`, and `velocity_score`.
- **Significance:** A $10,000 transaction might be normal for a wealthy individual. However, if it occurs 500km away from home, and the card was swiped in another country just 1 hour ago, it becomes a Multivariate Outlier. Isolation Forest helps isolate this group before feeding the data into the prediction model.
![Outlier Detection](<figures\Transaction analysis and anomaly detection.png>)

---

## 4. Baseline XGBoost Model Training

- The chosen algorithm is **XGBoost** - the "King" of tabular data.
- Handles data imbalance using the `scale_pos_weight` parameter (ratio of Majority / Minority class).
- Evaluates model strength through:
  - **Confusion Matrix:** Visually reviewing the number of False Positives (false alarms) and False Negatives (missed frauds).
  - **PR-AUC (Precision-Recall Area Under Curve):** The gold standard metric for imbalanced data.
  - **Feature Importances:** Analyzing which factors (e.g., `velocity_score`, `is_foreign_transaction`) contribute the most to the decision to block a transaction.
![Confusion Matrix](<figures/Confusion Matrix.png>)
![PR Curve and Features Importances](<figures\P-R Curve Comparision and Features Importances.png>)

---

## 5. Cost-Benefit Analysis - The Core of the Project

Instead of selecting a default probability threshold (Threshold = 0.5) to block a card, we frame the problem as minimizing financial damages.

*Defining the financial problem:*
- **False Positive (False Alarm):** The system incorrectly blocks a valid transaction. 
  - *Cost (C_FP):* $100 (includes $15 manual review cost + $85 risk of customer frustration and service abandonment - Churn Cost).
- **False Negative (Missed Fraud):** The system misses a fraudulent transaction.
  - *Cost (Cost_FN):* Exactly the amount lost (`amount_usd`) plus a chargeback fee (`FEE_CHARGEBACK` = $25) and a bank penalty fee (`FEE_PENALTY` = $15).

The notebook ran a test loop from threshold `0.01` to `0.99` to plot a U-shaped Cost Curve. The bottom of the U-shape represents the **Optimal Threshold** - where the total financial damage in USD is minimized.
![Cost Curve](<figures/Cost-Benefit Analysis.png>)

---

## 6. Hyperparameter Tuning with Optuna and Custom Weights

To maximize the defense level, XGBoost is automatically fine-tuned using the **Optuna** library:
- **Custom Sample Weights:** This is an extremely advanced technique. Instead of penalizing the model with a generic penalty when it predicts incorrectly, the notebook assigns XGBoost an array of `train_weights` based on the *actual dollar amount of that specific transaction*.
  - *If the model misses a $5 transaction -> Light penalty.*
  - *If the model misses a $50,000 transaction -> Extremely heavy penalty.*
- Optuna searches through 30 Trials to find the set of hyperparameters (learning_rate, max_depth, n_estimators, gamma) that yields the highest PR-AUC score.
![Confusion Matrix with Tuned XGBoost model](<figures\Confusion matrix - Tuned XGBoost.png>)
---

## 7. Production Automation Report

At the end of the notebook, there is a detailed report summarizing:
- The optimal threshold to lock in (e.g., `0.17`).
- The minimum total risk cost achieved.
- How many customers were inconvenienced (False Positives) and how many fraud cases slipped through (False Negatives), along with the corresponding financial damages.

The algorithmic logic and constants (thresholds) generated from this Notebook have been packaged into a production-ready API in the `/api` and `/src` directories of the project.
