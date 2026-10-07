# features.py
"""Feature engineering: calendar, zone, trend, weather and event features.

Ablation Step 1 uses ``CALENDAR_TREND_FEATURES`` only. When the data
engineering team delivers ``data/processed/master_train.csv``, append the new
weather/event column names to the relevant set and re-run ``train.py`` to get
Ablation Steps 2-4 without touching any other code.
"""

from __future__ import annotations

import pandas as pd

from cleaning import CANONICAL_ZONES, TARGET, TIME_COL, ZONE_COL
from integration import EVENT_FEATURES, WEATHER_FEATURES

TREND_FEATURE = "days_elapsed"
ORIGIN_TIME = pd.Timestamp("2025-01-01 00:00:00")

BASE_FEATURES = [
    ZONE_COL,
    "hour",
    "dayofweek",
    "month",
    "day",
    "weekofyear",
    "is_weekend",
]

CALENDAR_TREND_FEATURES = BASE_FEATURES + [TREND_FEATURE]

EXTRA_FEATURES: list[str] = []


def full_features(extra: list[str] | None = None) -> list[str]:
    """Default feature set: calendar + trend + weather + events (+ extras)."""
    extra = EXTRA_FEATURES if extra is None else extra
    return CALENDAR_TREND_FEATURES + WEATHER_FEATURES + EVENT_FEATURES + list(extra)


FEATURE_GROUPS = {
    "calendar_zone_trend": list(CALENDAR_TREND_FEATURES),
    "weather": list(WEATHER_FEATURES),
    "events": list(EVENT_FEATURES),
}


def feature_combo(*groups: str) -> list[str]:
    """Build a feature list from FEATURE_GROUPS names, e.g. ('calendar_zone_trend', 'weather')."""
    out: list[str] = []
    for group in groups:
        missing = [f for f in FEATURE_GROUPS[group] if f not in out]
        out.extend(missing)
    return out


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach calendar features (hour, weekday, month, ...) and a linear trend."""
    out = df.copy()
    ts = out[TIME_COL]
    out["hour"] = ts.dt.hour.astype("int16")
    out["dayofweek"] = ts.dt.dayofweek.astype("int16")
    out["month"] = ts.dt.month.astype("int16")
    out["day"] = ts.dt.day.astype("int16")
    out["weekofyear"] = ts.dt.isocalendar().week.astype("int32").to_numpy()
    out["is_weekend"] = (out["dayofweek"] >= 5).astype("int8")
    out[TREND_FEATURE] = ((ts - ORIGIN_TIME).dt.total_seconds() / 86400.0).astype("float32")
    if ZONE_COL in out.columns:
        out[ZONE_COL] = pd.Categorical(out[ZONE_COL], categories=CANONICAL_ZONES)
    return out


def build_xy(
    df: pd.DataFrame,
    features: list[str] | None = None,
    target: str = TARGET,
) -> tuple[pd.DataFrame, pd.Series]:
    """Slice a frame into the model matrix ``X`` and target ``y``."""
    features = full_features() if features is None else features
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise KeyError(f"features not present in frame: {missing}")
    X = df.loc[:, features].copy()
    if ZONE_COL in X.columns and not isinstance(X[ZONE_COL].dtype, pd.CategoricalDtype):
        X[ZONE_COL] = pd.Categorical(X[ZONE_COL], categories=CANONICAL_ZONES)
    y = df[target].astype("float64") if target in df.columns else None
    return X, y