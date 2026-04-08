<!-- # FILE: IMPLEMENTATION_GUIDE.md -->
# Implementation Guide: Maternal Mortality Prediction System

This guide explains how to set up, run, use, and extend the MMR prediction platform end to end.

## 1. Environment Setup

1. Open terminal in the project parent directory.
2. Enter project root:

```bash
cd mmr-prediction
```

3. Create virtual environment:

```bash
python -m venv .venv
```

4. Activate virtual environment:

Windows cmd:

```bash
.venv\Scripts\activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

5. Install dependencies:

```bash
pip install -r backend/requirements.txt
```

## 2. Data Placement

Place your dataset here exactly:

```text
data/Maternal_Mortality.csv
```

Expected data includes country metadata + yearly MMR columns from 1990 to 2021.

## 3. Run The Backend

Start FastAPI server:

```bash
uvicorn backend.main:app --reload
```

Behavior on startup:

- loads CSV
- reshapes and engineers features
- loads cached model from backend/mmr_model.pkl or trains new model if missing
- prepares encoder maps and analytics cache in memory

## 4. Accessing The Frontend

Open:

```text
http://localhost:8000/
```

The dashboard is served by FastAPI root route and includes:

- global KPIs
- searchable country explorer
- country forecasting modal
- continent/HDI analytics charts
- feature importance
- ranking tables
- custom prediction form

## 5. Retraining The Model

Use either option:

1. Dashboard button: Retrain Model
2. API endpoint:

```text
GET /api/model/retrain
```

When retraining is triggered:

- model is fit with configured XGBoost hyperparameters
- metrics are recalculated
- model is saved to backend/mmr_model.pkl
- metrics are saved to backend/mmr_model_metrics.json

## 6. How To Interpret Predictions

### Country Forecast Endpoint

`POST /api/predict/country`

- uses iterative forecasting from latest known country history
- each future year uses predicted value to build next lag features
- returns confidence approximation: lower/upper = predicted ± 15%

### Single Prediction Endpoint

`POST /api/predict/single`

Use when manually testing what-if scenarios with custom features.

### Risk Levels In UI

- Low: MMR < 100
- Medium: 100 <= MMR < 300
- High: 300 <= MMR < 600
- Critical: MMR >= 600

## 7. Extending The System

### Add New Features

1. Edit feature engineering in backend/data_processor.py
2. Add feature name to FEATURES list in backend/model.py
3. Update single prediction schema and frontend form (if user-input feature)
4. Retrain model and validate metrics

### Add New Models

1. Implement alternative trainer in backend/model.py (e.g., LightGBM, CatBoost)
2. Add model selection config in backend/main.py
3. Expose model comparison endpoint and dashboard toggle

### Add New Analytics

1. Create aggregation function in backend/data_processor.py
2. Add endpoint in backend/main.py
3. Render chart/table in frontend/index.html

## 8. Troubleshooting

### Error: Dataset not found

Cause:

- CSV missing from data directory

Fix:

- verify path is data/Maternal_Mortality.csv

### Error: Required column missing

Cause:

- CSV headers differ from expected schema

Fix:

- align headers to expected names
- update alias handling in backend/data_processor.py if needed

### Error: Model not initialized (503)

Cause:

- startup initialization failed

Fix:

- check terminal logs for data loading/training errors
- confirm dependencies installed and CSV readable

### Error: Invalid category in /api/predict/single

Cause:

- continent/hdi_group/undp_region not in known encoder classes

Fix:

- use values from dropdowns provided by frontend

### Frontend appears blank or charts missing

Cause:

- backend not running or CDN script blocked

Fix:

- verify http://localhost:8000/api/health
- refresh once online so Chart.js/cache can initialize

## 9. Deployment Guide (Docker + Cloud)

### A. Dockerize

Create Dockerfile (example):

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r backend/requirements.txt
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t mmr-prediction .
docker run -p 8000:8000 mmr-prediction
```

### B. Cloud Deployment Targets

Recommended options:

1. Azure App Service (containerized or code-based deployment)
2. AWS ECS/Fargate
3. Google Cloud Run
4. Render / Railway for fast deployment

### C. Production Hardening Checklist

1. restrict CORS to trusted origins
2. enable HTTPS only
3. add API rate limiting
4. add auth for retrain and prediction endpoints if needed
5. externalize model/data storage (S3/Blob/GCS)
6. configure structured logging + monitoring
7. add CI tests and automated deployment workflow
