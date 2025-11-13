# ml/temperature.py
import math
from flask import jsonify, request
from ml.base_predictor import generic_predict_from_mongo

def predict_temperature_api(history_collection):
    """
    Predict Temperature (°C) (temperature). Query param: ?minutes=<int>
    """
    try:
        minutes = int(request.args.get("minutes", 60))
        sample_interval = 2
        steps = max(1, math.ceil(minutes / sample_interval))

        result = generic_predict_from_mongo(
            history_collection=history_collection,
            target="temperature",
            model_dir="./models/temp model",
            steps=steps,
            lookback=60,
            sample_interval=sample_interval
        )

        result["requested_minutes"] = minutes
        return jsonify({"status": "success", **result})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
