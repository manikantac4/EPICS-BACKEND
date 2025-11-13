import os
import math
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


def generic_predict_from_mongo(
    history_collection,
    target,
    model_dir,
    steps,                 # <<<<<<<<<< NOW STEPS ARE PASSED
    lookback=60,
    sample_interval=2
):
    """
    Generic ML predictor for MongoDB-based sensor forecasting.
    """

    # -------------------------
    # LOAD MODEL + SCALERS
    # -------------------------
    model_path = os.path.join(model_dir, [f for f in os.listdir(model_dir) if f.endswith(".keras")][0])
    input_scaler_path = os.path.join(model_dir, "input_scaler.joblib")
    target_scaler_path = os.path.join(model_dir, "target_scaler.joblib")

    model = load_model(model_path)
    input_scaler = joblib.load(input_scaler_path)
    target_scaler = joblib.load(target_scaler_path)

    # -------------------------
    # FETCH RAW DATA FROM MONGO
    # -------------------------
    cursor = history_collection.find(
        {},
        {"_id": 0, "timestamp": 1, target: 1}
    ).sort("timestamp", -1).limit(1000)

    rows = list(cursor)
    if not rows:
        raise Exception("No MongoDB data found for prediction.")

    df_raw = pd.DataFrame(rows).sort_values("timestamp")

    # -------------------------
    # PREPROCESS
    # -------------------------
    df = df_raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    # resample
    df = df.set_index("timestamp").resample(f"{sample_interval}T").mean().interpolate()

    # smoothing
    df[target] = df[target].rolling(3, min_periods=1, center=True).median()

    # feature engineering
    df[f"{target}_delta1"] = df[target].diff().fillna(0)
    df[f"{target}_rm5"] = df[target].rolling(5, min_periods=1).mean()

    hour = df.index.hour + df.index.minute/60
    df["sin_hour"] = np.sin(2*np.pi*hour/24)
    df["cos_hour"] = np.cos(2*np.pi*hour/24)

    df = df.reset_index()

    # CHECK LOOKBACK
    if len(df) < lookback:
        raise Exception(f"Not enough data: have {len(df)}, need {lookback}")

    feature_cols = [
        target,
        f"{target}_delta1",
        f"{target}_rm5",
        "sin_hour",
        "cos_hour"
    ]

    window = df[["timestamp"] + feature_cols].tail(lookback).reset_index(drop=True)

    # -------------------------
    # ITERATIVE FORECASTING
    # -------------------------
    preds = []
    total_steps = 0

    while total_steps < steps:

        X = window[feature_cols].values
        X_scaled = input_scaler.transform(X.reshape(-1, len(feature_cols))).reshape(1, lookback, len(feature_cols))

        scaled_out = model.predict(X_scaled, verbose=0)
        pred_real = float(target_scaler.inverse_transform(scaled_out.reshape(-1, 1)).flatten()[0])

        preds.append(pred_real)

        # STEP WINDOW
        last_ts = pd.to_datetime(window["timestamp"].iloc[-1])
        new_ts = last_ts + pd.Timedelta(minutes=sample_interval)

        new_row = {
            "timestamp": new_ts,
            target: pred_real,
            f"{target}_delta1": pred_real - window[target].iloc[-1],
            f"{target}_rm5": np.mean(list(window[target].tail(4)) + [pred_real]),
            "sin_hour": np.sin(2 * np.pi * (new_ts.hour + new_ts.minute/60) / 24),
            "cos_hour": np.cos(2 * np.pi * (new_ts.hour + new_ts.minute/60) / 24)
        }

        window = pd.concat([window, pd.DataFrame([new_row])], ignore_index=True).tail(lookback)

        total_steps += 1

    return {
        "predicted_value": preds[-1],
        "predicted_timestamp": str(window["timestamp"].iloc[-1]),
        "steps": steps,
        "history": preds
    }
