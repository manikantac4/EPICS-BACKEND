from flask import jsonify, request
from ml.base_predictor import generic_predict_from_mongo


def predict_co_api(history_collection):
    try:
        minutes = int(request.args.get("minutes", 30))
        steps = max(1, minutes // 2)  # sample interval = 2 minutes

        result = generic_predict_from_mongo(
            history_collection=history_collection,
            target="co_ppm",
            model_dir="./models/co model",
            steps=steps,
            lookback=60,
            sample_interval=2
        )

        result["requested_minutes"] = minutes

        return jsonify({
            "status": "success",
            **result
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
