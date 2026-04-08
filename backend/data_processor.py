# FILE: backend/data_processor.py
"""Data loading, reshaping, and feature engineering utilities."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from sklearn.preprocessing import LabelEncoder


ID_COLUMN_ALIASES: dict[str, str] = {
    "ISO3": "ISO3",
    "Country": "Country",
    "Continent": "Continent",
    "Hemisphere": "Hemisphere",
    "Human Development Groups": "HDI_Group",
    "UNDP Developing Regions": "UNDP_Region",
    "HDI Rank (2021)": "HDI_Rank",
}

HDI_ORDINAL_MAP: dict[str, int] = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "very high": 3,
}

YEAR_COLUMN_PATTERN = re.compile(r"\((\d{4})\)\s*$")


def _resolve_required_columns(columns: list[str]) -> dict[str, str]:
    """Resolve expected source columns and map them to normalized names."""
    normalized_lookup = {col.strip().lower(): col for col in columns}
    resolved: dict[str, str] = {}

    for source_name, normalized_name in ID_COLUMN_ALIASES.items():
        key = source_name.strip().lower()
        matched_col = normalized_lookup.get(key)
        if matched_col is None:
            raise ValueError(
                f"Required column '{source_name}' not found in dataset. "
                f"Available columns: {columns}"
            )
        resolved[matched_col] = normalized_name

    return resolved


def _extract_year_columns(columns: list[str]) -> dict[str, int]:
    """Extract MMR year columns and return a map of column name -> year."""
    year_columns: dict[str, int] = {}

    for column in columns:
        col_str = str(column).strip()

        # Common expected pattern:
        # Maternal Mortality Ratio (deaths per 100,000 live births) (1990)
        match = YEAR_COLUMN_PATTERN.search(col_str)
        if match:
            year = int(match.group(1))
            if 1900 <= year <= 2100:
                year_columns[column] = year
                continue

        # Fallback for plain year-like columns.
        if col_str.isdigit() and len(col_str) == 4:
            year = int(col_str)
            if 1900 <= year <= 2100:
                year_columns[column] = year

    if not year_columns:
        raise ValueError("No year-based MMR columns found in dataset.")

    return year_columns


def load_and_reshape(csv_path: str) -> pd.DataFrame:
    """
    Melt wide to long format for maternal mortality time series.

    Returns a DataFrame with columns:
    ISO3, Country, Continent, Hemisphere, HDI_Group, UNDP_Region,
    HDI_Rank, Year (int), MMR (float)
    """
    raw_df = pd.read_csv(csv_path)

    rename_map = _resolve_required_columns(list(raw_df.columns))
    work_df = raw_df.rename(columns=rename_map).copy()

    year_map = _extract_year_columns(list(work_df.columns))
    id_vars = list(ID_COLUMN_ALIASES.values())

    long_df = work_df.melt(
        id_vars=id_vars,
        value_vars=list(year_map.keys()),
        var_name="MMR_Source_Column",
        value_name="MMR",
    )

    long_df["Year"] = long_df["MMR_Source_Column"].map(year_map).astype(int)
    long_df["MMR"] = pd.to_numeric(long_df["MMR"], errors="coerce")
    long_df["HDI_Rank"] = pd.to_numeric(long_df["HDI_Rank"], errors="coerce")

    long_df = long_df.drop(columns=["MMR_Source_Column"]).dropna(subset=["MMR"])
    long_df = long_df.sort_values(["ISO3", "Year"]).reset_index(drop=True)

    return long_df[
        [
            "ISO3",
            "Country",
            "Continent",
            "Hemisphere",
            "HDI_Group",
            "UNDP_Region",
            "HDI_Rank",
            "Year",
            "MMR",
        ]
    ]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer model features for maternal mortality forecasting.

    Added features:
    - lag_1, lag_2, lag_3
    - rolling_mean_5
    - mmr_change
    - mmr_pct_change
    - year_norm
    - continent_enc
    - hdi_group_enc
    - undp_enc

    Rows containing NaN values introduced by feature generation are dropped.
    """
    required_cols = {
        "ISO3",
        "Country",
        "Continent",
        "Hemisphere",
        "HDI_Group",
        "UNDP_Region",
        "HDI_Rank",
        "Year",
        "MMR",
    }
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for feature engineering: {sorted(missing)}")

    work_df = df.copy().sort_values(["ISO3", "Year"]).reset_index(drop=True)
    country_groups = work_df.groupby("ISO3", group_keys=False)

    work_df["lag_1"] = country_groups["MMR"].shift(1)
    work_df["lag_2"] = country_groups["MMR"].shift(2)
    work_df["lag_3"] = country_groups["MMR"].shift(3)

    work_df["rolling_mean_5"] = country_groups["MMR"].transform(
        lambda series: series.shift(1).rolling(window=5, min_periods=1).mean()
    )

    work_df["mmr_change"] = country_groups["MMR"].diff(1)
    work_df["mmr_pct_change"] = country_groups["MMR"].pct_change() * 100.0
    work_df["year_norm"] = (work_df["Year"] - 1990) / 31.0

    continent_encoder = LabelEncoder()
    undp_encoder = LabelEncoder()

    work_df["continent_enc"] = continent_encoder.fit_transform(
        work_df["Continent"].astype(str).str.strip()
    )

    work_df["hdi_group_enc"] = (
        work_df["HDI_Group"].astype(str).str.strip().str.lower().map(HDI_ORDINAL_MAP)
    )

    work_df["undp_enc"] = undp_encoder.fit_transform(
        work_df["UNDP_Region"].astype(str).str.strip()
    )

    feature_na_columns = [
        "lag_1",
        "lag_2",
        "lag_3",
        "rolling_mean_5",
        "mmr_change",
        "mmr_pct_change",
        "hdi_group_enc",
        "HDI_Rank",
    ]

    work_df = work_df.dropna(subset=feature_na_columns).reset_index(drop=True)

    work_df["continent_enc"] = work_df["continent_enc"].astype(int)
    work_df["hdi_group_enc"] = work_df["hdi_group_enc"].astype(int)
    work_df["undp_enc"] = work_df["undp_enc"].astype(int)

    return work_df


def get_country_timeseries(df: pd.DataFrame, iso3: str) -> pd.DataFrame:
    """Return all rows for a country (ISO3), sorted by year ascending."""
    if "ISO3" not in df.columns or "Year" not in df.columns:
        raise ValueError("DataFrame must contain ISO3 and Year columns.")

    return (
        df[df["ISO3"].astype(str).str.upper() == iso3.upper()]
        .sort_values("Year")
        .reset_index(drop=True)
    )


def get_continent_avg(df: pd.DataFrame) -> pd.DataFrame:
    """Return average MMR per Continent per Year."""
    required_cols = {"Continent", "Year", "MMR"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns for continent analytics: {sorted(missing)}")

    return (
        df.groupby(["Continent", "Year"], as_index=False)["MMR"]
        .mean()
        .rename(columns={"MMR": "avg_mmr"})
        .sort_values(["Continent", "Year"])
        .reset_index(drop=True)
    )


def get_hdi_group_avg(df: pd.DataFrame) -> pd.DataFrame:
    """Return average MMR per HDI Group per Year."""
    required_cols = {"HDI_Group", "Year", "MMR"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns for HDI analytics: {sorted(missing)}")

    return (
        df.groupby(["HDI_Group", "Year"], as_index=False)["MMR"]
        .mean()
        .rename(columns={"MMR": "avg_mmr"})
        .sort_values(["HDI_Group", "Year"])
        .reset_index(drop=True)
    )


def build_encoding_maps(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Build deterministic category-to-integer maps used by the API layer."""
    continent_classes = sorted(df["Continent"].astype(str).str.strip().unique().tolist())
    undp_classes = sorted(df["UNDP_Region"].astype(str).str.strip().unique().tolist())

    return {
        "continent": {name: idx for idx, name in enumerate(continent_classes)},
        "hdi_group": {
            "Low": 0,
            "Medium": 1,
            "High": 2,
            "Very High": 3,
        },
        "undp_region": {name: idx for idx, name in enumerate(undp_classes)},
    }


def to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert DataFrame rows to JSON-serializable records."""
    return df.to_dict(orient="records")
