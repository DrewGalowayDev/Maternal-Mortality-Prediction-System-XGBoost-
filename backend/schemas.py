# FILE: backend/schemas.py
"""Pydantic schemas for API request/response payloads."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request body for country-level forecasting."""

    iso3: str
    forecast_years: int = Field(default=10, ge=1, le=30)


class SinglePredictionRequest(BaseModel):
    """Request body for one-off custom MMR prediction."""

    year: int = Field(ge=1990, le=2050)
    hdi_rank: float = Field(ge=1, le=200)
    lag_1: float = Field(ge=0)
    lag_2: float = Field(ge=0)
    lag_3: float = Field(ge=0)
    rolling_mean_5: float = Field(ge=0)
    mmr_change: float
    continent: str
    hdi_group: str
    undp_region: str


class ForecastPoint(BaseModel):
    """Single forecast point with approximate confidence interval."""

    year: int
    predicted_mmr: float
    lower: float
    upper: float


class PredictionResponse(BaseModel):
    """Response body for country-level forecast endpoint."""

    iso3: str
    country: str
    historical: list[dict]
    forecast: list[ForecastPoint]
    model_metrics: dict


class CountryInfo(BaseModel):
    """Lightweight country summary record."""

    iso3: str
    country: str
    continent: str
    hdi_group: str
    undp_region: str | None = None
    latest_mmr: float
    mmr_1990: float
    pct_change: float


class ModelMetrics(BaseModel):
    """Model quality and feature importance payload."""

    mae: float
    rmse: float
    r2: float
    feature_importance: list[dict]
