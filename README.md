<!-- # FILE: README.md -->
# Maternal Mortality Prediction System (XGBoost)

A production-ready full-stack machine learning system that predicts Maternal Mortality Ratio (MMR) using XGBoost and serves interactive global analytics through FastAPI + a mission-control style dashboard.

## 1. Project Overview

This system ingests a global maternal mortality dataset (1990-2021), engineers time-series features per country, trains an XGBoost regressor, and exposes prediction plus analytics endpoints for a rich single-page frontend.

### Architecture Diagram (ASCII)

```text
+-----------------------------+            +-----------------------------+
|  data/Maternal_Mortality.csv|            | frontend/index.html         |
|  (195 countries, 1990-2021) |            | (Chart.js dashboard)        |
+--------------+--------------+            +-------------+---------------+
               |                                         |
               v                                         |
+------------------------------------------+             |
| backend/data_processor.py                |             |
| - load_and_reshape (wide -> long)        |             |
| - engineer_features (lags, rolling, enc) |             |
+----------------------+-------------------+             |
                       |                                 |
                       v                                 |
+------------------------------------------+             |
| backend/model.py                         |             |
| - train_model (XGBRegressor)             |             |
| - load_model (cached .pkl)               |             |
| - predict_single / predict_country_future|             |
+----------------------+-------------------+             |
                       |                                 |
                       v                                 v
+---------------------------------------------------------------+
| backend/main.py (FastAPI)                                     |
| - startup data/model initialization                            |
| - /api/model, /api/countries, /api/predict, /api/analytics    |
| - serves frontend / and mounts /static                         |
+---------------------------------------------------------------+
```

## 2. Prerequisites

- Python 3.10+
- pip
- Dataset file: data/Maternal_Mortality.csv

## 3. Installation And Run

1. Clone your repository and open terminal in project root.
2. Change directory:

```bash
cd mmr-prediction
```

3. Create and activate virtual environment.

Windows (cmd):

```bash
python -m venv .venv
.venv\Scripts\activate
```

4. Install dependencies:

```bash
pip install -r backend/requirements.txt
```

5. Place dataset at:

```text
data/Maternal_Mortality.csv
```

6. Start API server:

```bash
uvicorn backend.main:app --reload
```

7. Open dashboard:

```text
http://localhost:8000/
```

## 4. API Documentation

| Method | Endpoint | Description |
|---|---|---|
| GET | / | Serve dashboard frontend |
| GET | /api/health | Service health + model load status |
| GET | /api/model/metrics | MAE, RMSE, R², feature importance |
| GET | /api/model/retrain | Force retraining and return new metrics |
| GET | /api/countries | Country summary list |
| GET | /api/countries/{iso3} | Country metadata + full historical timeseries |
| POST | /api/predict/country | Country forecast response with confidence bounds |
| POST | /api/predict/single | Single custom MMR prediction |
| GET | /api/analytics/continents | Average MMR per continent per year |
| GET | /api/analytics/hdi-groups | Average MMR per HDI group per year |
| GET | /api/analytics/top-worst | Top 10 highest MMR + top 10 most improved |
| GET | /api/analytics/correlation | HDI rank vs MMR scatter points + Pearson correlation |

## 5. Dataset Description

Input CSV schema includes:

- ISO3
- Country
- Continent
- Hemisphere
- Human Development Groups
- UNDP Developing Regions
- HDI Rank (2021)
- Maternal Mortality Ratio (deaths per 100,000 live births) (1990..2021)

The backend reshapes year columns to a long time-series format for model training and analytics.

## 6. Model Architecture

### Why XGBoost?

XGBoost is used because it handles:

- non-linear relationships between socio-development features and MMR
- mixed feature types (continuous + encoded categorical)
- robust performance with moderate-sized structured datasets
- stable feature importance extraction for interpretability

### Hyperparameters

The model uses:

- n_estimators=500
- max_depth=6
- learning_rate=0.05
- subsample=0.8
- colsample_bytree=0.8
- min_child_weight=3
- reg_alpha=0.1
- reg_lambda=1.0
- random_state=42
- n_jobs=-1

### Train/Test Strategy

Temporal split:

- Train: Year <= 2017
- Test: Year > 2017

Metrics reported:

- MAE
- RMSE
- R²

## 7. Feature Engineering

Features used by the model:

- Year
- HDI_Rank
- lag_1, lag_2, lag_3 (country-level historical lags)
- rolling_mean_5 (lagged rolling mean)
- mmr_change
- year_norm = (Year - 1990) / 31
- continent_enc (label encoded)
- hdi_group_enc (ordinal: Low=0, Medium=1, High=2, Very High=3)
- undp_enc (label encoded)

Additional derived feature for analytics:

- mmr_pct_change

Rows with missing engineered values are dropped before training.

## 8. Screenshots Description (ASCII Mock)

```text
+--------------------------------------------------------------------------------+
| Maternal Mortality Intelligence Platform       [MAE | RMSE | R2] [Retrain]     |
| XGBoost-Powered Global Health Analytics                                       |
+----------------------+----------------------+----------------------+------------+
| Countries            | Global Avg MMR       | Highest MMR Country  | Best Drop |
+--------------------------------------------------------------------------------+
| Filters              | Country Cards Grid (paginated, sparklines, latest MMR) |
| Search/Continent/HDI |                                                            |
+--------------------------------------------------------------------------------+
| Continent Multi-Line Chart               | HDI Group Stacked Area Chart          |
+--------------------------------------------------------------------------------+
| Feature Importance (Horizontal Bars)     | HDI Rank vs MMR Correlation Scatter   |
+--------------------------------------------------------------------------------+
| Top 10 Highest MMR Table                 | Top 10 Most Improved Table            |
+--------------------------------------------------------------------------------+
| Custom Prediction Form + Risk Badge + Animated Result                           |
+--------------------------------------------------------------------------------+
```

Country modal includes:

- historical + forecast line chart
- confidence interval band
- key-year markers
- reduction stats
- CSV export button

## 9. Future Improvements

1. Add model versioning and experiment tracking (MLflow).
2. Add advanced confidence intervals via quantile regression or bootstrapping.
3. Introduce SHAP-based local explainability per prediction.
4. Add authentication and role-based access for sensitive workflows.
5. Add database persistence for prediction history and audit logs.
6. Add background retraining jobs and CI/CD deployment pipeline.
