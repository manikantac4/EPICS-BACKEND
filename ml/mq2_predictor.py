# ml/mq2_predictor.py
from flask import jsonify, request
from ml.ml_base import predict_from_mongo

def predict_mq2_api(history_collection):
    """
    Predict MQ2 ppm for next N minutes.
    """
    try:
        minutes = int(request.args.get("minutes", 30))   # e.g. /predict_mq2?minutes=30

        result = predict_from_mongo(
            history_collection=history_collection,
            target="mq2_ppm",
            minutes=minutes,          # new arg
            model_dir="./models/mq2_ppm",  # OR None for auto-detect
            lookback=60,
            sample_interval=2,
            log_target=True
        )

        result["requested_minutes"] = minutes
        return jsonify({"status": "success", **result})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
