# predict.py
"""Test set inference and submission generation.

Generates Deliverable G4: submission/team_quatro_submission.csv
- Exactly 4,032 rows matching ride_demand_test.csv row_id order
- Columns: row_id, predicted_trips
- Non-negative, no NaNs, strictly numeric
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from cleaning import TIME_COL, ZONE_COL, canonicalize_zone, parse_pickup_hour
from features import add_calendar_features, build_xy
from integration import (
    add_event_features,
    add_weather_features,
    load_events,
    load_weather,
)

CONFIG = {
    "test_path": "data/raw/ride_demand_test.csv",
    "template_path": "data/raw/submission_template.csv",
    "weather_path": "data/raw/weather_hourly.csv",
    "events_path": "data/raw/events_calendar.csv",
    "model_path": "models/final_model.joblib",
    "submission_out": "submission/team_quatro_submission.csv",
    "processed_test_out": "data/processed/master_test.csv",
}


def generate_predictions(config: dict | None = None, verbose: bool = True) -> pd.DataFrame:
    cfg = CONFIG if config is None else config
    test_path = Path(cfg["test_path"])
    template_path = Path(cfg["template_path"])
    model_path = Path(cfg["model_path"])
    sub_out = Path(cfg["submission_out"])

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Run train.py first.")

    # 1. Load test data preserving original row_id order
    test_raw = pd.read_csv(test_path)
    n_expected = len(test_raw)
    if verbose:
        print(f"[predict] Loaded {n_expected} test records from {test_path}")

    # 2. Clean and standardize timestamps and zone labels
    test_df = test_raw.copy()
    test_df[TIME_COL] = parse_pickup_hour(test_df[TIME_COL])
    test_df[ZONE_COL] = test_df[ZONE_COL].map(canonicalize_zone)

    # 3. Load weather (forecast horizon Nov 1-14) and events
    weather = load_weather(cfg["weather_path"], verbose=verbose)
    events = load_events(cfg["events_path"], verbose=verbose)

    # 4. Attach weather, event, and calendar features
    test_joined = add_weather_features(test_df, weather)
    test_joined = add_event_features(test_joined, events)
    test_joined = add_calendar_features(test_joined)

    # Export master_test.csv (Deliverable A8)
    test_proc_out = Path(cfg.get("processed_test_out", "data/processed/master_test.csv"))
    test_proc_out.parent.mkdir(parents=True, exist_ok=True)
    test_joined.to_csv(test_proc_out, index=False)
    if verbose:
        print(f"[predict] Exported processed test table to {test_proc_out}")

    # 5. Load model bundle
    bundle = joblib.load(model_path)
    model = bundle["model"]
    features = bundle["features"]
    if verbose:
        print(f"[predict] Loaded model bundle ({len(features)} features)")

    # 6. Extract feature matrix X
    X_test, _ = build_xy(test_joined, features=features, target="trips")

    # 7. Predict
    raw_preds = model.predict(X_test)

    # Post-process: clip negative values to 0, round to 2 decimal places
    clipped_preds = np.clip(raw_preds, 0, None)
    predicted_trips = np.round(clipped_preds, 2)

    submission = pd.DataFrame({
        "row_id": test_raw["row_id"],
        "predicted_trips": predicted_trips,
    })

    # 8. Strict Rule G4 Validation Checks
    assert len(submission) == 4032, f"Expected 4032 rows, got {len(submission)}"
    assert list(submission.columns) == ["row_id", "predicted_trips"], f"Invalid columns: {submission.columns}"
    assert submission["predicted_trips"].isna().sum() == 0, "Submission contains NaN predictions!"
    assert (submission["predicted_trips"] < 0).sum() == 0, "Submission contains negative predictions!"

    if template_path.exists():
        template = pd.read_csv(template_path)
        assert (submission["row_id"].values == template["row_id"].values).all(), (
            "row_id ordering does not match submission_template.csv!"
        )

    # 9. Save submission
    sub_out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(sub_out, index=False)
    if verbose:
        print(f"[predict] Successfully generated submission file: {sub_out}")
        print(f"[predict] Summary stats: min={predicted_trips.min():.2f}, "
              f"max={predicted_trips.max():.2f}, mean={predicted_trips.mean():.2f}")

    return submission


if __name__ == "__main__":
    generate_predictions()
