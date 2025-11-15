# ml/voc_predictor.py
from flask import jsonify, request
from ml.ml_base import predict_from_mongo

def predict_voc_api(history_collection):
    """
    HTTP endpoint handler for VOC predictions.
    Query param: ?minutes=<int>
    """
    try:
        minutes = int(request.args.get("minutes", 30))
        res = predict_from_mongo(
            history_collection=history_collection,
            target="voc_ppm",
            minutes=minutes,
            model_dir="./models/voc_ppm",   # or None to auto-detect under ./models
            lookback=60,
            sample_interval=2,
            log_target=True
        )
        res["requested_minutes"] = minutes
        return jsonify({"status": "success", **res})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
