# Telco Customer Churn Prediction

Predicting which customers are likely to cancel their service using machine learning, with explainable AI to surface actionable business insights.

## Overview

Customer churn is costly — retaining an existing customer is significantly cheaper than acquiring a new one. This project builds a churn prediction system that not only classifies at-risk customers but also explains *why* each customer is flagged, enabling targeted retention campaigns.

## Dataset

**IBM Telco Customer Churn** — [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

- 7,043 customers, 21 features
- Target: `Churn` (Yes / No) — 26.5% churn rate

## Project Steps

### Step 1 — EDA
- Identified hidden nulls in `TotalCharges` stored as whitespace strings (not caught by `isnull()`)
- Class distribution: 73.5% no churn / 26.5% churn — imbalanced but manageable
- Key correlations: short `tenure` and high `MonthlyCharges` strongly associated with churn

### Step 2 — Preprocessing
- **Label Encoding** for binary columns (gender, Partner, etc.)
- **One-Hot Encoding** for multi-class columns (Contract, InternetService, etc.)
- **StandardScaler** to normalize feature scale — `TotalCharges` (0–8,000) vs `tenure` (0–72)
- **Stratified train/test split** (80/20) to preserve churn ratio in both sets

### Step 3 — Logistic Regression (Baseline)

| Metric | Score |
|--------|-------|
| F1 (churn) | 0.61 |
| ROC-AUC | 0.84 |
| Recall (churn) | 0.57 |

### Step 4 — Neural Network (Keras)

Architecture: `Dense(64) → Dropout(0.3) → Dense(32) → Dropout(0.3) → Dense(1)`

Regularization: L2 + Dropout + EarlyStopping (patience=5)

| Metric | Score |
|--------|-------|
| F1 (churn) | 0.57 |
| ROC-AUC | 0.84 |
| Recall (churn) | 0.51 |

> **Logistic Regression outperformed the Neural Network** — a key finding. With only 7,000 rows of structured tabular data, a simple linear model generalizes better. Neural networks excel at large-scale unstructured data (images, text), not small tabular datasets.

### Step 5 — Threshold Tuning

Default threshold (0.5) optimizes for accuracy. Lowering to **0.3** increases Recall at the cost of Precision — a deliberate business tradeoff.

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|----|
| 0.3 | 0.52 | **0.75** | 0.61 |
| 0.5 | 0.66 | 0.57 | 0.61 |
| 0.6 | 0.71 | 0.40 | 0.51 |

**Decision:** Use threshold = 0.3 because the cost of missing a churner (False Negative) is higher than the cost of an unnecessary retention call (False Positive).

### Step 6 — SHAP (Explainable AI)

Top churn drivers identified by SHAP:
- **tenure** — short tenure strongly predicts churn
- **Contract type** — month-to-month contracts have much higher churn risk than 2-year contracts
- **MonthlyCharges** — customers on expensive plans with short tenure are at highest risk
- **InternetService (Fiber optic)** — associated with higher churn, likely due to price sensitivity

Example business insight from force plot:
> *"Customer flagged as high churn risk: Fiber optic + streaming services + month-to-month contract + tenure < 6 months → recommend locking in with a discounted annual contract."*

## Results Summary

| Model | F1 (churn) | ROC-AUC | Recall |
|-------|-----------|---------|--------|
| Logistic Regression (threshold=0.5) | 0.61 | 0.84 | 0.57 |
| Logistic Regression (threshold=0.3) | 0.61 | 0.84 | **0.75** |
| Neural Network | 0.57 | 0.84 | 0.51 |

**Best model:** Logistic Regression with threshold = 0.3

## Tech Stack

- **Python** — pandas, numpy, matplotlib, seaborn
- **scikit-learn** — LogisticRegression, StandardScaler, train_test_split, metrics
- **TensorFlow / Keras** — Neural Network with Dropout and EarlyStopping
- **SHAP** — Explainable AI for feature attribution

## Setup

```bash
git clone https://github.com/klasanpetch/customer-churn-prediction.git
cd customer-churn-prediction
uv sync
jupyter notebook customer_churn_prediction.ipynb
```
