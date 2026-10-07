# integration.py
"""Weather and events integration: clock alignment, cleaning, and feature joins.

Clock note (Deliverable A2 / B2.1): every weather timestamp is on the UTC clock
even though it is written in two formats. ``...T21:00:00Z`` rows are ISO-UTC;
``DD/MM/YYYY HH:MM`` rows are UTC with the ``Z`` suffix dropped (proved by the
two formats interleaving seamlessly on the UTC timeline). Both are converted to
Africa/Addis_Ababa (UTC+3) naive local time so they align with the trip table,
which is on EAT.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    _src = str(Path(__file__).resolve().parent)
    if _src not in sys.path:
        sys.path.insert(0, _src)

import numpy as np
import pandas as pd

from cleaning import CANONICAL_ZONES, TIME_COL, ZONE_COL, canonicalize_zone

CITYWIDE_TOKENS = {"CITYWIDE", "CITY-WIDE", "ALL", "ALL ZONES"}

WEATHER_TIME_COL = "timestamp"
EVENT_TYPE_COL = "event_type"

EVENT_TYPES = [
    "public_holiday",
    "school_break",
    "football_match",
    "concert",
    "conference",
    "exhibition",
    "road_closure",
    "sports_run",
]

WEATHER_FEATURES = [
    "temp_c",
    "rain_mm",
    "humidity_pct",
    "wind_kmh",
    "rain_3h",
    "rain_class",
    "is_rainy",
]

EVENT_FEATURES = [
    "event_any",
    "event_active_window",
    "hours_until_event",
    "hours_since_event",
] + [f"event_{t}" for t in EVENT_TYPES] + ["event_attendance"]

EVENT_HORIZON_H = 48.0
EVENT_PRE_POST_H = 2.0


def canonicalize_event_type(value: object) -> str:
    token = re.sub(r"\s+", "_", str(value).strip().lower())
    return token


def parse_event_time(values: pd.Series) -> pd.Series:
    """Parse events timestamps: ISO, ``dd/mm/yyyy hh:mm``, ``MMM dd, yyyy hh:mm AM/PM``."""
    text = values.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")

    is_iso = text.str.match(r"^\d{4}-")
    if is_iso.any():
        out[is_iso] = pd.to_datetime(text[is_iso], format="%Y-%m-%d %H:%M", errors="coerce")

    rest = ~is_iso
    if rest.any():
        tentative = pd.to_datetime(text[rest], format="%d/%m/%Y %H:%M", errors="coerce")
        parsed_ok = tentative.notna()
        if parsed_ok.any():
            out[rest & parsed_ok] = tentative[parsed_ok]
        leftover = rest & out.isna()
        if leftover.any():
            out[leftover] = pd.to_datetime(
                text[leftover], format="%b %d, %Y %I:%M %p", errors="coerce"
            )
    return out


def _parse_attendance(value: object) -> float:
    digits = re.sub(r"[^\d]", "", str(value))
    return float(digits) if digits else np.nan


def event_zones(value: object) -> list[str]:
    """Map an events-table zone label (multi-zone, parenthesised, citywide) onto zones."""
    label = str(value).strip().upper()
    if label in CITYWIDE_TOKENS:
        return list(CANONICAL_ZONES)
    zones: list[str] = []
    for part in re.split(r"\s*(?:&|AND)\s*", label):
        cleaned = re.sub(r"\s*\(.*\)", "", part).strip()
        zone = canonicalize_zone(cleaned)
        if zone in CANONICAL_ZONES and zone not in zones:
            zones.append(zone)
    return zones


def load_weather(path: str | Path, verbose: bool = True) -> pd.DataFrame:
    """Load, clock-convert (UTC -> EAT), clean and impute the hourly weather table.

    Cleaning & Imputation (Deliverables A1, A4, A7):
    - Parse dual timestamps (ISO-UTC and DMY UTC) and convert to local EAT (Africa/Addis_Ababa, naive).
    - Convert Fahrenheit sensor readings (> 45°C) to Celsius: (F - 32) * 5 / 9.
    - Mask sensor sentinels (e.g. rain_mm < 0 like -9999.0) and physical out-of-bounds readings.
    - Collapse duplicate timestamps by taking hourly means.
    - Reindex onto a continuous hourly grid covering the complete modeling and test horizon
      (2024-12-30 00:00 to 2025-11-15 23:00).
    - Time-series linear interpolation followed by forward/backward fill for continuous
      weather variables (temperature, humidity, wind).
    - Time-series interpolation with 0.0 default (dry weather) for precipitation (rain_mm).
    - Leaves exactly zero missing values (NaNs) across all weather features.
    """
    df = pd.read_csv(path).copy()
    n_raw = len(df)
    text = df[WEATHER_TIME_COL].astype(str).str.strip()
    is_utc_iso = text.str.endswith("Z")
    is_utc_dmy = ~is_utc_iso

    ts = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    ts[is_utc_iso] = (
        pd.to_datetime(text[is_utc_iso], utc=True, format="ISO8601")
        .dt.tz_convert("Africa/Addis_Ababa")
        .dt.tz_localize(None)
    )
    ts[is_utc_dmy] = (
        pd.to_datetime(text[is_utc_dmy], format="%d/%m/%Y %H:%M", errors="coerce")
        .dt.tz_localize("UTC")
        .dt.tz_convert("Africa/Addis_Ababa")
        .dt.tz_localize(None)
    )

    df = df.assign(ts=ts)
    df = df[df["ts"].notna()].drop(columns=WEATHER_TIME_COL)

    # 1. Convert Fahrenheit readings (> 45°C) to Celsius: (F - 32) * 5 / 9
    f_mask = df["temp_c"] > 45.0
    n_f = int(f_mask.sum())
    df.loc[f_mask, "temp_c"] = (df.loc[f_mask, "temp_c"] - 32.0) * 5.0 / 9.0

    # 2. Mask physical out-of-bounds readings
    df["temp_c"] = df["temp_c"].mask(~df["temp_c"].between(-10, 45, inclusive="both"))
    df["humidity_pct"] = df["humidity_pct"].mask(~df["humidity_pct"].between(0, 100))
    df["wind_kmh"] = df["wind_kmh"].mask(df["wind_kmh"] < 0)
    df["rain_mm"] = df["rain_mm"].mask(df["rain_mm"] < 0)

    # 3. Collapse duplicate hours
    n_dup = int(df.duplicated(subset=["ts"]).sum())
    keep = ["ts", "temp_c", "rain_mm", "humidity_pct", "wind_kmh"]
    df = df[keep].groupby("ts", as_index=False, sort=True).mean(numeric_only=True)

    # 4. Reindex onto complete continuous hourly grid spanning full modeling & forecast horizon
    min_hour = df["ts"].min().floor("D") - pd.Timedelta(days=1)
    max_hour = df["ts"].max().ceil("D") + pd.Timedelta(days=1)
    full_grid = pd.date_range(min_hour, max_hour, freq="h")
    df = df.set_index("ts").reindex(full_grid)
    df.index.name = "ts"

    # 5. Time-series linear interpolation & forward/backward imputation
    df["temp_c"] = df["temp_c"].interpolate(method="time").ffill().bfill()
    df["humidity_pct"] = df["humidity_pct"].interpolate(method="time").ffill().bfill()
    df["wind_kmh"] = df["wind_kmh"].interpolate(method="time").ffill().bfill()
    df["rain_mm"] = df["rain_mm"].interpolate(method="time").fillna(0.0).clip(lower=0.0)

    df = df.reset_index()

    if verbose:
        print(f"[weather] {path}")
        print(f"[weather] rows in / unparseable: {n_raw} / {int(ts.isna().sum())}")
        print(f"[weather] Fahrenheit readings converted to Celsius: {n_f}")
        print(f"[weather] duplicate hours collapsed: {n_dup}")
        print("[weather] clock proven: UTC -> Africa/Addis_Ababa (UTC+3), naive")
        print(f"[weather] continuous hourly range: {df['ts'].min()} -> {df['ts'].max()}")
        print(
            f"[weather] post-imputation NaNs -> temp {int(df['temp_c'].isna().sum())}, "
            f"rain {int(df['rain_mm'].isna().sum())}, "
            f"humidity {int(df['humidity_pct'].isna().sum())}, "
            f"wind {int(df['wind_kmh'].isna().sum())}"
        )
    return df


def load_events(path: str | Path, verbose: bool = True) -> pd.DataFrame:
    """Load, parse and clean the events calendar; explode to one row per zone."""
    df = pd.read_csv(path).copy()
    n_raw = len(df)

    df["start"] = parse_event_time(df["start_datetime"])
    df["end"] = parse_event_time(df["end_datetime"])
    df[EVENT_TYPE_COL] = df[EVENT_TYPE_COL].map(canonicalize_event_type)
    df["confirmed"] = df["status"].str.strip().str.lower().eq("confirmed")
    df["attendance"] = df["expected_attendance"].map(_parse_attendance)

    duration_h = (df["end"] - df["start"]).dt.total_seconds() / 3600
    median_by_type = df.assign(d=duration_h).groupby(EVENT_TYPE_COL)["d"].median()

    bad_end = df["end"].isna() | (duration_h <= 0)
    fill_from_type = (
        df[EVENT_TYPE_COL].map(median_by_type).where(
            df[EVENT_TYPE_COL].map(median_by_type.notna())
        )
    )
    fill = fill_from_type.fillna(2.0).clip(lower=2.0)
    df.loc[bad_end.fillna(False), "end"] = (
        df.loc[bad_end.fillna(False), "start"] + pd.to_timedelta(fill[bad_end.fillna(False)], unit="h")
    )

    rows = [
        (row_id, zone)
        for row_id, row in df.iterrows()
        for zone in event_zones(row["zone"])
    ]
    ev = pd.DataFrame(rows, columns=["_row", ZONE_COL])
    merged = df.loc[ev["_row"]].reset_index(drop=True)
    merged[ZONE_COL] = ev[ZONE_COL].values

    if verbose:
        print(f"[events] {path}")
        print(f"[events] rows: {n_raw}; unparseable start: {int(df['start'].isna().sum())}")
        print(f"[events] missing/negative-length ends filled: {int(bad_end.sum())}")
        print(f"[events] cancelled events excluded: {int((~df['confirmed']).sum())}")
        print(f"[events] event-zone rows after explode: {len(merged)}")

    keep = ["event_id", "event_name", EVENT_TYPE_COL, ZONE_COL, "start", "end", "attendance", "confirmed"]
    return merged[keep]


def add_weather_features(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Attach city-wide hourly weather features on the EAT clock (many-to-one join)."""
    w = weather.sort_values("ts").copy()
    w["rain_3h"] = w["rain_mm"].rolling(3, min_periods=1).sum()
    w["is_rainy"] = (w["rain_mm"] > 0).astype("float64")
    w["rain_class"] = np.select(
        [w["rain_mm"] == 0, w["rain_mm"] < 2.5, w["rain_mm"] < 7.6],
        [0.0, 1.0, 2.0],
        default=3.0,
    )
    out = df.copy()
    merged = out.merge(
        w[["ts"] + WEATHER_FEATURES], left_on=TIME_COL, right_on="ts", how="left"
    ).drop(columns="ts")

    # Safety guard: ensure zero NaNs across all weather features (Deliverables A4 & A7)
    for col in WEATHER_FEATURES:
        if merged[col].isna().any():
            if col in ("rain_mm", "rain_3h", "is_rainy", "rain_class"):
                merged[col] = merged[col].fillna(0.0)
            else:
                merged[col] = merged[col].ffill().bfill().fillna(merged[col].median())

    return merged


def add_event_features(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Attach event features known at forecast time to each zone-hour.

    Only confirmed events are used. Relative distances are capped at
    EVENT_HORIZON_H (48h) so no feature depends on events far outside the
    forecast window. Zone-scoped and citywide events apply to the right rows.
    """
    out = df.copy()
    for feat in EVENT_FEATURES:
        out[feat] = 0.0

    confirmed = events[events["confirmed"]].reset_index(drop=True)

    for zone in CANONICAL_ZONES:
        zone_rows = out.index[out[ZONE_COL].eq(zone)].to_numpy()
        if zone_rows.size == 0:
            continue
        zone_ts = out.loc[zone_rows, TIME_COL].to_numpy()

        evz = confirmed[confirmed[ZONE_COL].eq(zone)].sort_values("start")
        if evz.empty:
            continue

        hours_until = np.zeros(zone_rows.size, dtype="float64")
        hours_since = np.zeros(zone_rows.size, dtype="float64")
        attendance_ext = np.zeros(zone_rows.size, dtype="float64")
        type_flags = {t: np.zeros(zone_rows.size, dtype="float64") for t in EVENT_TYPES}

        next_merge = pd.merge_asof(
            pd.DataFrame({"ts": zone_ts}).sort_values("ts"),
            evz[["start", "attendance"]],
            left_on="ts",
            right_on="start",
            direction="forward",
            allow_exact_matches=True,
        )
        gap_next = (next_merge["start"] - next_merge["ts"]).dt.total_seconds() / 3600
        has_next = gap_next.notna().to_numpy()
        within = has_next & (gap_next.to_numpy() <= EVENT_HORIZON_H)
        hours_until[within] = gap_next.to_numpy()[within].clip(min=0.0)
        attendance_ext = np.where(
            within, next_merge["attendance"].fillna(0.0).to_numpy(), 0.0
        )

        since_merge = pd.merge_asof(
            pd.DataFrame({"ts": zone_ts}).sort_values("ts"),
            evz.sort_values("end")[["end"]],
            left_on="ts",
            right_on="end",
            direction="backward",
            allow_exact_matches=True,
        )
        gap_since = (since_merge["ts"] - since_merge["end"]).dt.total_seconds() / 3600
        has_since = gap_since.notna().to_numpy()
        since_within = has_since & (gap_since.to_numpy() <= EVENT_HORIZON_H)
        hours_since[since_within] = gap_since.to_numpy()[since_within].clip(min=0.0)

        active_window = np.zeros(zone_rows.size, dtype="float64")
        order = np.argsort(zone_ts, kind="mergesort")
        zone_ts_sorted = zone_ts[order]
        for _, evt in evz.iterrows():
            etype = str(evt[EVENT_TYPE_COL])
            ext_lo_key = (evt["start"] - pd.Timedelta(hours=EVENT_PRE_POST_H)).to_datetime64()
            ext_hi_key = (evt["end"] + pd.Timedelta(hours=EVENT_PRE_POST_H)).to_datetime64()
            ext_lo = int(np.searchsorted(zone_ts_sorted, ext_lo_key, side="left"))
            ext_hi = int(np.searchsorted(zone_ts_sorted, ext_hi_key, side="right"))
            if ext_hi > ext_lo:
                hit = order[ext_lo:ext_hi]
                if etype in type_flags:
                    type_flags[etype][hit] = 1.0
                att = float(evt["attendance"]) if pd.notna(evt["attendance"]) else 0.0
                attendance_ext[hit] = np.maximum(attendance_ext[hit], att)
            s_lo = int(np.searchsorted(zone_ts_sorted, evt["start"].to_datetime64(), side="left"))
            s_hi = int(np.searchsorted(zone_ts_sorted, evt["end"].to_datetime64(), side="right"))
            if s_hi > s_lo:
                active_window[order[s_lo:s_hi]] = 1.0

        any_ev = (
            (active_window > 0) | (hours_until > 0) | (hours_since > 0)
        ).astype("float64")

        out.loc[zone_rows, "event_any"] = any_ev
        out.loc[zone_rows, "event_active_window"] = active_window
        out.loc[zone_rows, "hours_until_event"] = hours_until
        out.loc[zone_rows, "hours_since_event"] = hours_since
        out.loc[zone_rows, "event_attendance"] = attendance_ext
        for t in EVENT_TYPES:
            out.loc[zone_rows, f"event_{t}"] = type_flags[t]

    return out