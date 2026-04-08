# FILE: backend/main.py
"""FastAPI application for Maternal Mortality Prediction System."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.data_processor import (
    build_encoding_maps,
    engineer_features,
    get_continent_avg,
    get_country_timeseries,
    get_hdi_group_avg,
    load_and_reshape,
)
from backend.model import (
    evaluate_model,
    get_feature_importance,
    load_model,
    load_saved_metrics,
    predict_country_future,
    predict_single,
    train_model,
)
from backend.schemas import (
    CountryInfo,
    ModelMetrics,
    PredictionRequest,
    PredictionResponse,
    SinglePredictionRequest,
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Maternal_Mortality.csv"
FRONTEND_DIR = BASE_DIR / "frontend"
INDEX_PATH = FRONTEND_DIR / "index.html"

HDI_FALLBACK_MAP: dict[str, int] = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "very high": 3,
}


@dataclass
class RuntimeState:
    """In-memory runtime cache for data, encoders, model, and metrics."""

    raw_df: pd.DataFrame | None = None
    feature_df: pd.DataFrame | None = None
    model: Any | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    encoders: dict[str, dict[str, int]] = field(default_factory=dict)


runtime_state = RuntimeState()

app = FastAPI(
    title="Maternal Mortality Prediction API",
    version="1.0.0",
    description="XGBoost-powered maternal mortality analytics and forecasting service.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


def _get_runtime() -> RuntimeState:
    """Return runtime state if initialized; otherwise raise HTTP 503."""
    if (
        runtime_state.raw_df is None
        or runtime_state.feature_df is None
        or runtime_state.model is None
    ):
        raise HTTPException(status_code=503, detail="Service not initialized.")
    return runtime_state


def _normalize_key(value: str) -> str:
    """Normalize strings for case-insensitive category lookups."""
    return value.strip().lower()


def _encode_with_map(mapping: dict[str, int], raw_value: str, field_name: str) -> int:
    """Encode a category value using case-insensitive lookup over a map."""
    normalized_map = {_normalize_key(key): value for key, value in mapping.items()}
    normalized_value = _normalize_key(raw_value)

    if normalized_value not in normalized_map:
        valid_values = sorted(mapping.keys())
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid value for {field_name}: '{raw_value}'. "
                f"Allowed values: {valid_values}"
            ),
        )

    return int(normalized_map[normalized_value])


def _to_feature_importance_list(
    feature_importance: dict[str, float] | list[dict[str, Any]] | None,
) -> list[dict[str, float | str]]:
    """Normalize feature importance payload to sorted list form."""
    if feature_importance is None:
        return []

    if isinstance(feature_importance, dict):
        records = [
            {"feature": feature, "importance": float(value)}
            for feature, value in feature_importance.items()
        ]
    else:
        records = [
            {
                "feature": str(item.get("feature", "unknown")),
                "importance": float(item.get("importance", 0.0)),
            }
            for item in feature_importance
        ]

    records.sort(key=lambda item: float(item["importance"]), reverse=True)
    return records


def _to_model_metrics(metrics: dict[str, Any], model_obj: Any) -> ModelMetrics:
    """Convert runtime metrics dictionary into response model payload."""
    feature_importance = metrics.get("feature_importance")
    if not feature_importance:
        feature_importance = get_feature_importance(model_obj)

    return ModelMetrics(
        mae=float(metrics.get("mae", 0.0)),
        rmse=float(metrics.get("rmse", 0.0)),
        r2=float(metrics.get("r2", 0.0)),
        feature_importance=_to_feature_importance_list(feature_importance),
    )


def _build_country_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build one-row-per-country summary table for list and ranking endpoints."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "iso3",
                "country",
                "continent",
                "hdi_group",
                "latest_mmr",
                "mmr_1990",
                "pct_change",
                "undp_region",
            ]
        )

    latest_year = int(df["Year"].max())

    latest_df = (
        df[df["Year"] == latest_year][
            ["ISO3", "Country", "Continent", "HDI_Group", "UNDP_Region", "MMR"]
        ]
        .rename(columns={"MMR": "latest_mmr"})
        .copy()
    )

    # Prefer 1990 values; fallback to the first available value per country.
    mmr_1990_df = df[df["Year"] == 1990][["ISO3", "MMR"]].rename(columns={"MMR": "mmr_1990"})

    first_available_df = (
        df.sort_values(["ISO3", "Year"]).groupby("ISO3", as_index=False)["MMR"].first()
    ).rename(columns={"MMR": "mmr_first_available"})

    merged_df = latest_df.merge(mmr_1990_df, how="left", on="ISO3")
    merged_df = merged_df.merge(first_available_df, how="left", on="ISO3")

    merged_df["mmr_1990"] = merged_df["mmr_1990"].fillna(merged_df["mmr_first_available"])

    merged_df["pct_change"] = np.where(
        merged_df["mmr_1990"] > 0,
        ((merged_df["mmr_1990"] - merged_df["latest_mmr"]) / merged_df["mmr_1990"]) * 100.0,
        0.0,
    )

    summary_df = merged_df.rename(
        columns={
            "ISO3": "iso3",
            "Country": "country",
            "Continent": "continent",
            "HDI_Group": "hdi_group",
            "UNDP_Region": "undp_region",
        }
    )

    return summary_df[
        [
            "iso3",
            "country",
            "continent",
            "hdi_group",
            "undp_region",
            "latest_mmr",
            "mmr_1990",
            "pct_change",
        ]
    ].sort_values("country")


def _initialize_runtime(force_retrain: bool = False) -> None:
    """Load data, engineer features, and load/retrain model in memory."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Place Maternal_Mortality.csv into data directory."
        )

    raw_df = load_and_reshape(str(DATA_PATH))
    feature_df = engineer_features(raw_df)

    if feature_df.empty:
        raise ValueError("Feature-engineered dataset is empty after dropping NaN rows.")

    if force_retrain:
        metrics = train_model(feature_df)
        model_obj = load_model()
    else:
        model_obj = load_model()
        metrics = load_saved_metrics()
        if metrics is None:
            metrics = evaluate_model(feature_df, model_obj)

    runtime_state.raw_df = raw_df
    runtime_state.feature_df = feature_df
    runtime_state.model = model_obj
    runtime_state.metrics = metrics
    runtime_state.encoders = build_encoding_maps(raw_df)


@app.on_event("startup")
def startup_event() -> None:
    """Initialize application state during server startup."""
    _initialize_runtime(force_retrain=False)


@app.get("/", include_in_schema=False)
def serve_dashboard() -> FileResponse:
    """Serve the frontend dashboard HTML file."""
    if not INDEX_PATH.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(INDEX_PATH)


@app.get("/api/health")
def health_check() -> dict[str, Any]:
    """Return lightweight service health status."""
    model_loaded = runtime_state.model is not None
    return {"status": "ok", "model_loaded": model_loaded}


@app.get("/api/model/metrics", response_model=ModelMetrics)
def get_model_metrics() -> ModelMetrics:
    """Return model quality metrics and feature importance."""
    current = _get_runtime()
    return _to_model_metrics(current.metrics, current.model)


@app.get("/api/model/retrain", response_model=ModelMetrics)
def retrain_model_endpoint() -> ModelMetrics:
    """Force model retraining and return updated evaluation metrics."""
    _initialize_runtime(force_retrain=True)
    current = _get_runtime()
    return _to_model_metrics(current.metrics, current.model)


@app.get("/api/countries", response_model=list[CountryInfo])
def list_countries() -> list[CountryInfo]:
    """Return all countries with latest MMR and long-term change summary."""
    current = _get_runtime()
    summary_df = _build_country_summary(current.raw_df)

    response_items = [
        CountryInfo(
            iso3=str(row["iso3"]),
            country=str(row["country"]),
            continent=str(row["continent"]),
            hdi_group=str(row["hdi_group"]),
            undp_region=str(row["undp_region"]),
            latest_mmr=float(row["latest_mmr"]),
            mmr_1990=float(row["mmr_1990"]),
            pct_change=float(row["pct_change"]),
        )
        for row in summary_df.to_dict(orient="records")
    ]

    return response_items


@app.get("/api/countries/{iso3}")
def get_country_detail(iso3: str) -> dict[str, Any]:
    """Return country summary and full historical MMR time series."""
    current = _get_runtime()

    iso3_norm = iso3.strip().upper()
    country_df = get_country_timeseries(current.raw_df, iso3_norm)
    if country_df.empty:
        raise HTTPException(status_code=404, detail=f"Country '{iso3_norm}' not found.")

    summary_df = _build_country_summary(current.raw_df)
    summary_rows = summary_df[summary_df["iso3"].astype(str).str.upper() == iso3_norm]
    if summary_rows.empty:
        raise HTTPException(status_code=404, detail=f"Summary for '{iso3_norm}' not found.")

    summary_row = summary_rows.iloc[0].to_dict()

    timeseries = [
        {"year": int(row["Year"]), "mmr": float(row["MMR"])}
        for row in country_df[["Year", "MMR"]].to_dict(orient="records")
    ]

    metadata = {
        "iso3": str(summary_row["iso3"]),
        "country": str(summary_row["country"]),
        "continent": str(summary_row["continent"]),
        "hdi_group": str(summary_row["hdi_group"]),
        "undp_region": str(summary_row["undp_region"]),
        "latest_mmr": float(summary_row["latest_mmr"]),
        "mmr_1990": float(summary_row["mmr_1990"]),
        "pct_change": float(summary_row["pct_change"]),
    }

    return {"country_info": metadata, "timeseries": timeseries}


@app.post("/api/predict/country", response_model=PredictionResponse)
def predict_country(request: PredictionRequest) -> PredictionResponse:
    """Return historical data and multi-year forecast for a selected country."""
    current = _get_runtime()

    iso3_norm = request.iso3.strip().upper()
    country_df = get_country_timeseries(current.raw_df, iso3_norm)
    if country_df.empty:
        raise HTTPException(status_code=404, detail=f"Country '{iso3_norm}' not found.")

    try:
        forecast = predict_country_future(
            df=current.raw_df,
            model=current.model,
            iso3=iso3_norm,
            forecast_years=request.forecast_years,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    historical = [
        {"year": int(row["Year"]), "mmr": float(row["MMR"])}
        for row in country_df[["Year", "MMR"]].to_dict(orient="records")
    ]

    metrics_payload = _to_model_metrics(current.metrics, current.model).model_dump()

    return PredictionResponse(
        iso3=iso3_norm,
        country=str(country_df.iloc[-1]["Country"]),
        historical=historical,
        forecast=forecast,
        model_metrics=metrics_payload,
    )


@app.post("/api/predict/single")
def predict_single_endpoint(request: SinglePredictionRequest) -> dict[str, float]:
    """Predict MMR for a manually provided single feature vector."""
    current = _get_runtime()

    encoders = current.encoders
    continent_map = encoders.get("continent", {})
    undp_map = encoders.get("undp_region", {})
    hdi_map = encoders.get("hdi_group", {})

    if not continent_map or not undp_map or not hdi_map:
        raise HTTPException(status_code=500, detail="Encoding maps are not initialized.")

    continent_enc = _encode_with_map(continent_map, request.continent, "continent")
    undp_enc = _encode_with_map(undp_map, request.undp_region, "undp_region")

    hdi_normalized = _normalize_key(request.hdi_group)
    hdi_from_encoder = {
        _normalize_key(key): int(value) for key, value in hdi_map.items()
    }

    if hdi_normalized in hdi_from_encoder:
        hdi_group_enc = hdi_from_encoder[hdi_normalized]
    elif hdi_normalized in HDI_FALLBACK_MAP:
        hdi_group_enc = HDI_FALLBACK_MAP[hdi_normalized]
    else:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid HDI group '{request.hdi_group}'. "
                "Allowed values: Low, Medium, High, Very High"
            ),
        )

    year_norm = (request.year - 1990) / 31.0

    prediction = predict_single(
        model=current.model,
        year=request.year,
        hdi_rank=request.hdi_rank,
        lag_1=request.lag_1,
        lag_2=request.lag_2,
        lag_3=request.lag_3,
        rolling_mean_5=request.rolling_mean_5,
        mmr_change=request.mmr_change,
        year_norm=year_norm,
        continent_enc=continent_enc,
        hdi_group_enc=hdi_group_enc,
        undp_enc=undp_enc,
    )

    return {"predicted_mmr": float(prediction)}


@app.get("/api/analytics/continents")
def analytics_continents() -> list[dict[str, Any]]:
    """Return average yearly MMR per continent."""
    current = _get_runtime()
    continent_df = get_continent_avg(current.raw_df).rename(
        columns={"Continent": "continent", "Year": "year"}
    )

    return [
        {
            "continent": str(row["continent"]),
            "year": int(row["year"]),
            "avg_mmr": float(row["avg_mmr"]),
        }
        for row in continent_df.to_dict(orient="records")
    ]


@app.get("/api/analytics/hdi-groups")
def analytics_hdi_groups() -> list[dict[str, Any]]:
    """Return average yearly MMR per HDI group."""
    current = _get_runtime()
    hdi_df = get_hdi_group_avg(current.raw_df).rename(
        columns={"HDI_Group": "hdi_group", "Year": "year"}
    )

    return [
        {
            "hdi_group": str(row["hdi_group"]),
            "year": int(row["year"]),
            "avg_mmr": float(row["avg_mmr"]),
        }
        for row in hdi_df.to_dict(orient="records")
    ]


@app.get("/api/analytics/top-worst")
def analytics_top_worst() -> dict[str, list[dict[str, Any]]]:
    """Return top worst countries and most improved countries in latest period."""
    current = _get_runtime()
    summary_df = _build_country_summary(current.raw_df)

    top_worst_df = summary_df.sort_values("latest_mmr", ascending=False).head(10)
    improved_df = summary_df.sort_values("pct_change", ascending=False).head(10)

    top_worst = [
        {
            "iso3": str(row["iso3"]),
            "country": str(row["country"]),
            "latest_mmr": float(row["latest_mmr"]),
            "pct_change": float(row["pct_change"]),
        }
        for row in top_worst_df.to_dict(orient="records")
    ]

    most_improved = [
        {
            "iso3": str(row["iso3"]),
            "country": str(row["country"]),
            "latest_mmr": float(row["latest_mmr"]),
            "pct_change": float(row["pct_change"]),
        }
        for row in improved_df.to_dict(orient="records")
    ]

    return {"top_worst": top_worst, "most_improved": most_improved}


@app.get("/api/analytics/correlation")
def analytics_correlation() -> dict[str, Any]:
    """Return HDI rank vs MMR scatter points and Pearson correlation."""
    current = _get_runtime()

    latest_year = int(current.raw_df["Year"].max())
    latest_df = current.raw_df[current.raw_df["Year"] == latest_year].copy()
    latest_df = latest_df.dropna(subset=["HDI_Rank", "MMR"])

    if latest_df.empty:
        raise HTTPException(status_code=404, detail="No data available for correlation analysis.")

    if len(latest_df) > 1:
        correlation = float(latest_df["HDI_Rank"].corr(latest_df["MMR"]))
    else:
        correlation = 0.0

    points = [
        {
            "iso3": str(row["ISO3"]),
            "country": str(row["Country"]),
            "hdi_rank": float(row["HDI_Rank"]),
            "mmr": float(row["MMR"]),
        }
        for row in latest_df[["ISO3", "Country", "HDI_Rank", "MMR"]].to_dict(orient="records")
    ]

    return {
        "year": latest_year,
        "pearson_correlation": correlation,
        "points": points,
    }
