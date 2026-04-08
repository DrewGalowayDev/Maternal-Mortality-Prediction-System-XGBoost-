# FILE: backend/model.py
"""XGBoost model training, evaluation, loading, and forecasting utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from backend.data_processor import engineer_features, load_and_reshape


FEATURES = [
    "Year",
    "HDI_Rank",
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_mean_5",
    "mmr_change",
    "year_norm",
    "continent_enc",
    "hdi_group_enc",
    "undp_enc",
]
TARGET = "MMR"

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(__file__).resolve().parent / "mmr_model.pkl"
METRICS_PATH = Path(__file__).resolve().parent / "mmr_model_metrics.json"
DEFAULT_CSV_PATH = BASE_DIR / "data" / "Maternal_Mortality.csv"

HDI_ORDINAL_MAP: dict[str, int] = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "very high": 3,
}


def _build_model() -> XGBRegressor:
    """Build an XGBoost regressor with configured hyperparameters."""
    return XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
    )


def _split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by temporal cutoff at year 2017."""
    train_df = df[df["Year"] <= 2017].copy()
    test_df = df[df["Year"] > 2017].copy()

    if train_df.empty:
        raise ValueError("Training split is empty. Check dataset year coverage.")
    if test_df.empty:
        raise ValueError("Test split is empty. Ensure data contains years after 2017.")

    return train_df, test_df


def _feature_importance_dict(model: XGBRegressor) -> dict[str, float]:
    """Return feature importances as a dictionary."""
    return {
        feature: float(importance)
        for feature, importance in zip(FEATURES, model.feature_importances_)
    }


def _sanitize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Round metrics and ensure JSON-safe primitive types."""
    return {
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "r2": float(metrics["r2"]),
        "feature_importance": {
            key: float(value)
            for key, value in dict(metrics["feature_importance"]).items()
        },
    }


def train_model(df: pd.DataFrame) -> dict[str, Any]:
    """
    Train and evaluate an XGBoost regressor, then persist model artifact.

    Steps:
    1) Split where Year <= 2017 for train, Year > 2017 for test
    2) Fit configured XGBRegressor
    3) Evaluate with MAE, RMSE, R2
    4) Save model to backend/mmr_model.pkl
    5) Return metrics and feature importance dictionary
    """
    missing_cols = set(FEATURES + [TARGET]).difference(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns for training: {sorted(missing_cols)}")

    train_df, test_df = _split_train_test(df)

    x_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    x_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    model = _build_model()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    metrics: dict[str, Any] = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
        "feature_importance": _feature_importance_dict(model),
    }
    metrics = _sanitize_metrics(metrics)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    with METRICS_PATH.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    return metrics


def load_saved_metrics() -> dict[str, Any] | None:
    """Load saved training metrics from disk if present."""
    if not METRICS_PATH.exists():
        return None

    with METRICS_PATH.open("r", encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


def _prepare_default_training_data() -> pd.DataFrame:
    """Prepare training features from default CSV path."""
    if not DEFAULT_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DEFAULT_CSV_PATH}. "
            "Place Maternal_Mortality.csv under the data directory."
        )

    base_df = load_and_reshape(str(DEFAULT_CSV_PATH))
    return engineer_features(base_df)


def load_model() -> XGBRegressor:
    """Load model from disk; if missing, train and persist one first."""
    if not MODEL_PATH.exists():
        training_df = _prepare_default_training_data()
        train_model(training_df)

    loaded_model = joblib.load(MODEL_PATH)
    if not isinstance(loaded_model, XGBRegressor):
        raise TypeError("Loaded model artifact is not an XGBRegressor instance.")

    return loaded_model


def evaluate_model(df: pd.DataFrame, model: XGBRegressor) -> dict[str, Any]:
    """Evaluate a loaded model on the fixed temporal split."""
    missing_cols = set(FEATURES + [TARGET]).difference(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns for evaluation: {sorted(missing_cols)}")

    _, test_df = _split_train_test(df)
    x_test = test_df[FEATURES]
    y_test = test_df[TARGET]
    predictions = model.predict(x_test)

    return {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
        "feature_importance": _feature_importance_dict(model),
    }


def _build_label_map(values: pd.Series) -> dict[str, int]:
    """Create deterministic label encoding map using sorted unique values."""
    classes = sorted(values.dropna().astype(str).str.strip().unique().tolist())
    return {name: idx for idx, name in enumerate(classes)}


def predict_country_future(
    df: pd.DataFrame,
    model: XGBRegressor,
    iso3: str,
    forecast_years: int = 10,
) -> list[dict[str, float | int]]:
    """
    Generate iterative multi-year forecast for a country.

    Confidence interval is approximated using +/- 15% around prediction.
    """
    country_df = (
        df[df["ISO3"].astype(str).str.upper() == iso3.upper()]
        .sort_values("Year")
        .reset_index(drop=True)
    )

    if country_df.empty:
        raise ValueError(f"Country with ISO3 '{iso3}' was not found.")

    if forecast_years < 1:
        raise ValueError("forecast_years must be at least 1.")

    if forecast_years > 30:
        raise ValueError("forecast_years cannot exceed 30.")

    latest_row = country_df.iloc[-1]
    history_values = country_df["MMR"].astype(float).tolist()
    last_year = int(latest_row["Year"])

    hdi_rank = float(latest_row["HDI_Rank"])

    continent_map = _build_label_map(df["Continent"])
    undp_map = _build_label_map(df["UNDP_Region"])

    continent_name = str(latest_row["Continent"]).strip()
    hdi_group_name = str(latest_row["HDI_Group"]).strip().lower()
    undp_name = str(latest_row["UNDP_Region"]).strip()

    if continent_name not in continent_map:
        raise ValueError(f"Unable to encode continent '{continent_name}'.")
    if undp_name not in undp_map:
        raise ValueError(f"Unable to encode UNDP region '{undp_name}'.")
    if hdi_group_name not in HDI_ORDINAL_MAP:
        raise ValueError(f"Unable to encode HDI group '{hdi_group_name}'.")

    continent_enc = continent_map[continent_name]
    hdi_group_enc = HDI_ORDINAL_MAP[hdi_group_name]
    undp_enc = undp_map[undp_name]

    forecast: list[dict[str, float | int]] = []

    for step in range(1, forecast_years + 1):
        next_year = last_year + step

        lag_1 = float(history_values[-1])
        lag_2 = float(history_values[-2]) if len(history_values) >= 2 else lag_1
        lag_3 = float(history_values[-3]) if len(history_values) >= 3 else lag_2

        rolling_window = history_values[-5:] if len(history_values) >= 5 else history_values
        rolling_mean_5 = float(np.mean(rolling_window))

        mmr_change = lag_1 - lag_2
        year_norm = (next_year - 1990) / 31.0

        row = np.array(
            [
                [
                    next_year,
                    hdi_rank,
                    lag_1,
                    lag_2,
                    lag_3,
                    rolling_mean_5,
                    mmr_change,
                    year_norm,
                    continent_enc,
                    hdi_group_enc,
                    undp_enc,
                ]
            ],
            dtype=float,
        )

        predicted = float(model.predict(row)[0])
        predicted = max(predicted, 0.0)

        history_values.append(predicted)

        lower = max(predicted * 0.85, 0.0)
        upper = predicted * 1.15

        forecast.append(
            {
                "year": int(next_year),
                "predicted_mmr": float(predicted),
                "lower": float(lower),
                "upper": float(upper),
            }
        )

    return forecast


def predict_single(
    model: XGBRegressor,
    year: int,
    hdi_rank: float,
    lag_1: float,
    lag_2: float,
    lag_3: float,
    rolling_mean_5: float,
    mmr_change: float,
    year_norm: float,
    continent_enc: int,
    hdi_group_enc: int,
    undp_enc: int,
) -> float:
    """Predict maternal mortality ratio for a single feature row."""
    row = np.array(
        [
            [
                float(year),
                float(hdi_rank),
                float(lag_1),
                float(lag_2),
                float(lag_3),
                float(rolling_mean_5),
                float(mmr_change),
                float(year_norm),
                float(continent_enc),
                float(hdi_group_enc),
                float(undp_enc),
            ]
        ],
        dtype=float,
    )

    predicted = float(model.predict(row)[0])
    return max(predicted, 0.0)


def get_feature_importance(model: XGBRegressor) -> list[dict[str, float | str]]:
    """Return sorted feature importances in descending order."""
    records = [
        {"feature": feature, "importance": float(importance)}
        for feature, importance in zip(FEATURES, model.feature_importances_)
    ]
    records.sort(key=lambda item: float(item["importance"]), reverse=True)
    return records
