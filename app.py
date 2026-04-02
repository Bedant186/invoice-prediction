import os
import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_PATH = "model/invoice_model.pkl"
FEATURE_PATH = "model/feature_columns.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURE_PATH):
    raise FileNotFoundError(
        "Model files not found. Please run train_model.py first."
    )

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Invoice Payment Delay Prediction API is running"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400

        missing_fields = [field for field in feature_columns if field not in data]
        if missing_fields:
            return jsonify({
                "error": "Missing required fields",
                "missing_fields": missing_fields
            }), 400

        input_data = [[data[field] for field in feature_columns]]

        prediction = int(model.predict(input_data)[0])
        probability = float(model.predict_proba(input_data)[0][1])

        risk_label = "High Risk" if probability >= 0.7 else (
            "Medium Risk" if probability >= 0.4 else "Low Risk"
        )

        return jsonify({
            "prediction": prediction,
            "prediction_label": "Late Payment" if prediction == 1 else "On-Time Payment",
            "late_payment_probability": round(probability, 4),
            "risk_category": risk_label
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)