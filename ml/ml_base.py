# ml/ml_base.py
"""
Reusable ML base utilities for time-series forecasting with saved Keras models + scalers.
- Auto-detects model directory if not explicitly provided (searches ./models/)
- Loads model, input & target scalers, feature_cols.json
- Preprocesses raw data (resample/interpolate, smoothing, delta, rm5, sin/cos)
- Assembles feature matrix preserving duplicate feature names (important!)
- Iterative multi-step forecasting (returns last predicted timestamp & history)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from tensorflow.keras.models import load_model

# ----------------------------
# Helper: find model dir automatically (searches ./models)
# ----------------------------
def find_model_dir_for_target(target_name: str, root_dir: str = "./models") -> Optional[str]:
    """Search for a folder under root_dir that contains a .keras file and feature_cols.json.
       Prioritize folders whose name contains the target_name if possible."""
    if not os.path.isdir(root_dir):
        return None

    candidates = []
    for entry in os.listdir(root_dir):
        path = os.path.join(root_dir, entry)
        if not os.path.isdir(path):
            continue
        files = set(os.listdir(path))
        if any(f.endswith(".keras") for f in files) and "feature_cols.json" in files and "input_scaler.joblib" in files and "target_scaler.joblib" in files:
            candidates.append(path)

    # prefer candidate whose dirname contains the target_name
    for c in candidates:
        if target_name and target_name.lower() in os.path.basename(c).lower():
            return c
    # otherwise return first candidate
    return candidates[0] if candidates else None

# ----------------------------
# Artifact loader
# ----------------------------
def load_artifacts(model_dir: str) -> Dict[str, Any]:
    """Load model, input_scaler, target_scaler and feature list from model_dir."""
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"model_dir not found: {model_dir}")

    # pick .keras file (prefer *_final.keras)
    keras_files = [f for f in os.listdir(model_dir) if f.endswith(".keras")]
    if not keras_files:
        raise FileNotFoundError(f"No .keras model file found in {model_dir}")
    final_cands = [f for f in keras_files if f.endswith("_final.keras")]
    model_file = final_cands[0] if final_cands else keras_files[0]
    model_path = os.path.join(model_dir, model_file)

    input_scaler_path = os.path.join(model_dir, "input_scaler.joblib")
    target_scaler_path = os.path.join(model_dir, "target_scaler.joblib")
    feat_path = os.path.join(model_dir, "feature_cols.json")

    for p in (model_path, input_scaler_path, target_scaler_path, feat_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required artifact not found: {p}")

    model = load_model(model_path, compile=False)
    input_scaler = joblib.load(input_scaler_path)
    target_scaler = joblib.load(target_scaler_path)
    with open(feat_path, "r") as fh:
        FEATURE_COLS = json.load(fh)

    if not isinstance(FEATURE_COLS, list) or len(FEATURE_COLS) == 0:
        raise ValueError("feature_cols.json must be a non-empty list")

    return {
        "model": model,
        "input_scaler": input_scaler,
        "target_scaler": target_scaler,
        "feature_cols": FEATURE_COLS,
        "model_file": model_file
    }

# ----------------------------
# Preprocessing (match training)
# ----------------------------
def preprocess_df_for_prediction(df_raw: pd.DataFrame, target: str, sample_interval: int = 2) -> pd.DataFrame:
    """
    df_raw: DataFrame with at least 'timestamp' and target column.
    Returns: DataFrame resampled to sample_interval minutes with engineered features.
    """
    if 'timestamp' not in df_raw.columns:
        raise KeyError("Input dataframe must contain 'timestamp' column")

    df = df_raw.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

    # resample to fixed interval and interpolate (use mean then time interpolation)
    df = df.set_index('timestamp').resample(f"{sample_interval}T").mean().interpolate(method='time')

    if target not in df.columns:
        raise KeyError(f"Target '{target}' not present after resample. Available cols: {list(df.columns)}")

    # small median smoothing to remove single-sample glitches
    df[target] = df[target].rolling(3, min_periods=1, center=True).median()

    # feature engineering (deltas and rolling means)
    df[f"{target}_delta1"] = df[target].diff().fillna(0)
    df[f"{target}_rm5"] = df[target].rolling(5, min_periods=1).mean()

    # cyclical hour features
    hour = df.index.hour + df.index.minute / 60.0
    df["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)

    df = df.reset_index()
    return df

# ----------------------------
# Assemble feature window (preserve duplicates)
# ----------------------------
def assemble_feature_window(df_preprocessed: pd.DataFrame, feature_cols: list, lookback: int) -> pd.DataFrame:
    """
    Return a DataFrame of shape (lookback, n_features) preserving duplicate feature names in order.
    The underlying preprocessed dataframe must contain the base columns required.
    """
    if len(df_preprocessed) < lookback:
        raise ValueError(f"Not enough rows in preprocessed df: have {len(df_preprocessed)}, need {lookback}")

    window_base = df_preprocessed.tail(lookback).reset_index(drop=True)

    # Verify all unique required base columns exist
    unique_required = set(feature_cols)
    missing = [c for c in unique_required if c not in window_base.columns]
    if missing:
        raise KeyError(f"Missing required base features after preprocessing: {missing}. Available: {list(window_base.columns)}")

    # Build an array preserving duplicate order
    cols_arrays = []
    for c in feature_cols:
        cols_arrays.append(window_base[c].values)

    # stacked array shape (lookback, n_features)
    X = np.column_stack(cols_arrays)
    return window_base, X

# ----------------------------
# Iterative forecast core
# ----------------------------
def iterative_forecast_from_window(window_base: pd.DataFrame,
                                   X_init: np.ndarray,
                                   artifacts: Dict[str, Any],
                                   target: str,
                                   steps: int,
                                   lookback: int,
                                   sample_interval: int = 2,
                                   log_target: bool = True) -> Dict[str, Any]:
    """
    Run iterative forecasting. X_init is the assembled array (lookback, n_feats).
    Returns dict with predicted_value, predicted_timestamp (last), steps and history list.
    """
    model = artifacts["model"]
    input_scaler = artifacts["input_scaler"]
    target_scaler = artifacts["target_scaler"]
    FEATURE_COLS = artifacts["feature_cols"]

    preds = []
    total_steps = 0
    last_new_ts = None

    window = window_base.copy()  # DataFrame with base columns (unique names)
    while total_steps < steps:
        # assemble X from current window preserving duplicates
        cols_arrays = [window[c].values for c in FEATURE_COLS]
        X = np.column_stack(cols_arrays)   # shape (lookback, n_feats)

        # sanity check
        if X.shape[0] != lookback or X.shape[1] != len(FEATURE_COLS):
            raise ValueError(f"Unexpected X shape: {X.shape}, expected ({lookback},{len(FEATURE_COLS)})")

        # scale and predict
        try:
            X_scaled = input_scaler.transform(X).reshape(1, lookback, X.shape[1])
        except Exception as e:
            raise RuntimeError(f"input_scaler.transform failed: {e}. X.shape: {X.shape}")

        out_s = model.predict(X_scaled, verbose=0)
        out_inv = target_scaler.inverse_transform(out_s.reshape(-1,1)).flatten()[0]
        pred_real = float(np.expm1(out_inv)) if log_target else float(out_inv)

        preds.append(pred_real)

        # step window forward
        last_ts = pd.to_datetime(window["timestamp"].iloc[-1])
        new_ts = last_ts + pd.Timedelta(minutes=sample_interval)
        last_new_ts = new_ts

        prev_target = float(window[target].iloc[-1])
        new_row = {
            "timestamp": new_ts,
            target: pred_real,
            f"{target}_delta1": pred_real - prev_target,
            f"{target}_rm5": float(np.mean(list(window[target].tail(4)) + [pred_real])),
            "sin_hour": float(np.sin(2 * np.pi * (new_ts.hour + new_ts.minute/60.0) / 24.0)),
            "cos_hour": float(np.cos(2 * np.pi * (new_ts.hour + new_ts.minute/60.0) / 24.0))
        }

        window = pd.concat([window, pd.DataFrame([new_row])], ignore_index=True).tail(lookback).reset_index(drop=True)
        total_steps += 1

    return {
        "predicted_value": preds[-1],
        "predicted_timestamp": str(last_new_ts),
        "steps": steps,
        "history": preds
    }

# ----------------------------
# Public entry points
# ----------------------------
def predict_from_dataframe(df_raw: pd.DataFrame,
                           target: str,
                           minutes: int = 30,
                           model_dir: Optional[str] = None,
                           lookback: int = 60,
                           sample_interval: int = 2,
                           log_target: bool = True) -> Dict[str, Any]:
    """
    Predict target for next `minutes` minutes given a raw dataframe with 'timestamp' and target column.
    If model_dir is None, auto-detect under ./models.
    """
    if model_dir is None:
        model_dir = find_model_dir_for_target(target)
        if model_dir is None:
            raise FileNotFoundError("Model directory not provided and auto-detection failed under ./models")

    artifacts = load_artifacts(model_dir)
    feature_cols = artifacts["feature_cols"]

    dfp = preprocess_df_for_prediction(df_raw, target, sample_interval=sample_interval)

    # assemble window and X
    window_base, X_init = assemble_feature_window(dfp, feature_cols, lookback)

    steps = max(1, int(np.ceil(minutes / sample_interval)))

    return iterative_forecast_from_window(window_base, X_init, artifacts, target, steps, lookback, sample_interval, log_target)

def predict_from_mongo(history_collection,
                       target: str,
                       minutes: int = 30,
                       model_dir: Optional[str] = None,
                       lookback: int = 60,
                       sample_interval: int = 2,
                       log_target: bool = True) -> Dict[str, Any]:
    """
    Fetch data from Mongo (history_collection), preprocess and predict.
    If model_dir is None, tries to auto-detect a model folder under ./models.
    """
    # fetch rows (newest first) and build dataframe in ascending time order
    cursor = history_collection.find({}, {"_id": 0, "timestamp": 1, target: 1}).sort("timestamp", -1).limit(max(1000, lookback*5))
    rows = list(cursor)
    if not rows:
        raise ValueError("No data returned from history_collection")

    df_raw = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return predict_from_dataframe(df_raw, target, minutes=minutes, model_dir=model_dir, lookback=lookback, sample_interval=sample_interval, log_target=log_target)

# ----------------------------
# Minimal fake collection helper for unit tests
# ----------------------------
class FakeCollection:
    """Wrap dataframe to mimic PyMongo cursor (yields newest first)."""
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy().reset_index(drop=True)
    def find(self, *args, **kwargs):
        # emulate .find(...).sort("timestamp", -1).limit(n) by yielding newest-first dicts
        rows = self.df.to_dict("records")
        for r in rows[::-1]:
            yield r
