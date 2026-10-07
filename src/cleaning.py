# cleaning.py
"""Data loading and cleaning for the Addis Ababa ride-demand pipeline."""

from __future__ import annotations

import re

import pandas as pd

TARGET = "trips"
TIME_COL = "pickup_hour"
ZONE_COL = "zone"

LEAKAGE_COLUMNS = ["avg_fare_birr", "avg_wait_min", "active_drivers"]

CANONICAL_ZONES = [
    "ARAT KILO",
    "AYAT",
    "BOLE",
    "CMC",
    "GERJI",
    "KAZANCHIS",
    "KOLFE",
    "LIDETA",
    "MEGENAGNA",
    "MERKATO",
    "PIASSA",
    "SARBET",
]

ZONE_ALIASES = {
    "BOLE RD": "BOLE",
    "KOLFE KERANIO": "KOLFE",
    "C.M.C": "CMC",
    "PIAZZA": "PIASSA",
    "MERCATO": "MERKATO",
    "KAZANCHES": "KAZANCHIS",
    "MEGENAGA": "MEGENAGNA",
}


def canonicalize_zone(value: object) -> str:
    """Map raw zone spellings (case, spacing, aliases) onto the 12 canonical zones."""
    zone = re.sub(r"\s+", " ", str(value).strip()).upper()
    return ZONE_ALIASES.get(zone, zone)


def parse_pickup_hour(values: pd.Series) -> pd.Series:
    """Parse the mixed timestamp formats in the raw data into naive local datetimes.

    Handles ``2025-06-26 11:00``, ``16/08/2025 04:00`` (day-first) and
    ``2025-05-27T23:00:00+03:00`` (UTC-offset ISO).
    """
    if pd.api.types.is_datetime64_any_dtype(values):
        values = values.copy()
        if values.dt.tz is not None:
            values = values.dt.tz_convert("Africa/Addis_Ababa").dt.tz_localize(None)
        return values

    text = values.astype(str).str.strip()
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")

    is_iso = text.str.match(r"^\d{4}-")
    has_offset = is_iso & text.str.contains(r"[+-]\d{2}:\d{2}$")

    naive = is_iso & ~has_offset
    if naive.any():
        parsed.loc[naive] = pd.to_datetime(text[naive], errors="coerce")

    if has_offset.any():
        parsed.loc[has_offset] = (
            pd.to_datetime(text[has_offset], errors="coerce", utc=True)
            .dt.tz_convert("Africa/Addis_Ababa")
            .dt.tz_localize(None)
        )

    day_first = ~is_iso
    if day_first.any():
        parsed.loc[day_first] = pd.to_datetime(
            text[day_first], format="%d/%m/%Y %H:%M", errors="coerce"
        )

    return parsed


def load_ride_demand(
    path: str | "Path",
    leakage_columns: list[str] | None = None,
    drop_invalid_targets: bool = True,
    deduplicate: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load a ride-demand CSV and return a leakage-free, modelling-ready frame.

    Steps: parse timestamps, canonicalise zones, drop operational/leakage
    columns, drop rows with missing or negative targets, collapse duplicate
    (zone, hour) records, sort chronologically.
    """
    leakage_columns = LEAKAGE_COLUMNS if leakage_columns is None else leakage_columns
    df = pd.read_csv(path)
    n_raw = len(df)

    df[TIME_COL] = parse_pickup_hour(df[TIME_COL])
    unparseable = int(df[TIME_COL].isna().sum())
    df = df[df[TIME_COL].notna()].copy()

    if ZONE_COL in df.columns:
        df[ZONE_COL] = df[ZONE_COL].map(canonicalize_zone)

    dropped_leakage = [c for c in leakage_columns if c in df.columns]
    df = df.drop(columns=dropped_leakage)

    n_invalid = 0
    if drop_invalid_targets and TARGET in df.columns:
        valid = df[TARGET].notna() & (df[TARGET] >= 0)
        n_invalid = int((~valid).sum())
        df = df[valid].copy()

    n_dupes = 0
    if deduplicate and df.duplicated(subset=[ZONE_COL, TIME_COL]).any():
        key = [ZONE_COL, TIME_COL]
        numeric = [c for c in df.columns if c not in key and pd.api.types.is_numeric_dtype(df[c])]
        other = [c for c in df.columns if c not in key and c not in numeric]
        agg = {c: "mean" for c in numeric}
        agg.update({c: "first" for c in other})
        n_before = len(df)
        df = df.groupby(key, as_index=False, sort=False).agg(agg)
        n_dupes = n_before - len(df)

    df = df.sort_values([TIME_COL, ZONE_COL], kind="mergesort").reset_index(drop=True)

    if verbose:
        print(f"[clean] {path}")
        print(f"[clean] raw rows                : {n_raw}")
        print(f"[clean] unparseable timestamps  : {unparseable}")
        print(f"[clean] dropped leakage columns : {dropped_leakage}")
        print(f"[clean] dropped invalid targets : {n_invalid}")
        print(f"[clean] duplicate rows collapsed: {n_dupes}")
        print(f"[clean] usable rows             : {len(df)}")
        if TARGET in df.columns:
            print(
                f"[clean] date range              : {df[TIME_COL].min()} -> {df[TIME_COL].max()}"
            )
        print(f"[clean] zones                   : {sorted(df[ZONE_COL].unique())}")

    return df
