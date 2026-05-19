from flask import Flask, request, jsonify
import numpy as np
import joblib
import tensorflow as tf

app = Flask(__name__)

model = tf.keras.models.load_model("final_pso_cnn_balanced_model.keras")
scaler = joblib.load("pso_scaler.pkl")
selected_features = joblib.load("pso_selected_features.pkl")

@app.route("/")
def home():
    return "IDS API is running (PSO + CNN model)"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features' in request body"}), 400

    features = np.array(data["features"], dtype=np.float32)

    if features.ndim == 1:
        features = features.reshape(1, -1)

    if features.shape[1] < int(np.max(selected_features)) + 1:
        return jsonify({"error": f"Expected at least {int(np.max(selected_features)) + 1} features"}), 400

    features_selected = features[:, selected_features]
    features_scaled = scaler.transform(features_selected)
    features_cnn = features_scaled.reshape(features_scaled.shape[0], features_scaled.shape[1], 1)

    pred_prob = model.predict(features_cnn, verbose=0)
    pred = (pred_prob > 0.5).astype(int).flatten()

    return jsonify({
        "prediction": int(pred[0]),
        "result": "ATTACK" if pred[0] == 1 else "NORMAL",
        "confidence": round(float(pred_prob[0][0]), 4)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
