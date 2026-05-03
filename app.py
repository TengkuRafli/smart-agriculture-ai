from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
CORS(app, resources={r"/*": {"origins": "*"}})
import numpy as np
import pickle
import os

app = Flask(__name__)
CORS(app)

latest_result = {}

# ================= LOAD MODEL (AMAN UNTUK RAILWAY) =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_file(filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{filename} tidak ditemukan di {BASE_DIR}")
    return pickle.load(open(path, "rb"))

model_full = load_file("model_full.pkl")
model_no_n = load_file("model_no_n.pkl")

scaler_full = load_file("scaler_full.pkl")
scaler_no_n = load_file("scaler_no_n.pkl")

encoder = load_file("label_encoder.pkl")

# ================= RELAY =================
relay_status = "OFF"

# ================= WEB =================
@app.route('/')
def index():
    return render_template('index.html')


# ================= FORM =================
@app.route('/predict', methods=['POST'])
def predict():
    global latest_result

    try:
        # ambil input
        temperature = float(request.form.get('temperature'))
        humidity = float(request.form.get('humidity'))
        soil_moisture = float(request.form.get('soil_moisture'))

        N = request.form.get('N')
        P = request.form.get('P')
        K = request.form.get('K')
        ph = request.form.get('ph')

        # convert aman
        N = float(N) if N not in (None, "") else None
        P = float(P) if P not in (None, "") else None
        K = float(K) if K not in (None, "") else None
        ph = float(ph) if ph not in (None, "") else 6.5

        penalty = 1.0

        if P is None:
            P = 50
            penalty *= 0.85

        if K is None:
            K = 50
            penalty *= 0.85

        use_full_model = True if (N is not None and N > 0) else False

        # ===== PREDIKSI =====
        if use_full_model:
            input_data = np.array([[N, P, K, temperature, humidity, ph, soil_moisture]])
            input_scaled = scaler_full.transform(input_data)

            pred = model_full.predict(input_scaled)
            prob = np.max(model_full.predict_proba(input_scaled))
            mode = "FULL"
        else:
            input_data = np.array([[P, K, temperature, humidity, ph, soil_moisture]])
            input_scaled = scaler_no_n.transform(input_data)

            pred = model_no_n.predict(input_scaled)
            prob = np.max(model_no_n.predict_proba(input_scaled))
            mode = "NO_N"

        hasil = encoder.inverse_transform(pred)[0]
        confidence = round(float(prob * penalty) * 100, 2)

        latest_result = {
            "plant": hasil,
            "confidence": confidence,
            "mode": mode,
            "sensor": {
                "temperature": temperature,
                "humidity": humidity,
                "soil_moisture": soil_moisture,
                "N": N, 
                "P": P, 
                "K": K, 
                "ph": ph
            }
        }

        return render_template(
            'index.html',
            prediction=hasil,
            temperature=temperature,
            humidity=humidity,
            soil_moisture=soil_moisture,
            N=N, P=P, K=K, ph=ph,
            confidence=confidence,
            mode=mode
        )

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")


# ================= API =================
@app.route('/api/predict', methods=['POST'])
def api_predict():
    global latest_result

    try:
        data = request.get_json()

        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        soil_moisture = float(data['soil_moisture'])

        N = data.get('N')
        P = data.get('P')
        K = data.get('K')
        ph = data.get('ph', 6.5)

        penalty = 1.0

        if P is None:
            P = 50
            penalty *= 0.85

        if K is None:
            K = 50
            penalty *= 0.85

        use_full_model = True if (N is not None and N > 0) else False

        if use_full_model:
            input_data = np.array([[N, P, K, temperature, humidity, ph, soil_moisture]])
            input_scaled = scaler_full.transform(input_data)

            pred = model_full.predict(input_scaled)
            prob = np.max(model_full.predict_proba(input_scaled))
            mode = "FULL"
        else:
            input_data = np.array([[P, K, temperature, humidity, ph, soil_moisture]])
            input_scaled = scaler_no_n.transform(input_data)

            pred = model_no_n.predict(input_scaled)
            prob = np.max(model_no_n.predict_proba(input_scaled))
            mode = "NO_N"

        hasil = encoder.inverse_transform(pred)[0]
        confidence = round(float(prob * penalty) * 100, 2)

        latest_result = {
    "plant": hasil,
    "confidence": confidence,
    "mode": mode,
    "sensor": {
        "temperature": temperature,
        "humidity": humidity,
        "soil_moisture": soil_moisture,
        "N": N,
        "P": P,
        "K": K,
        "ph": ph
    }
}

        return jsonify(latest_result)

    except Exception as e:
        return jsonify({"error": str(e)})


# ================= RELAY =================
@app.route('/api/relay', methods=['GET'])
def get_relay():
    return jsonify({"relay": relay_status})


@app.route('/api/relay', methods=['POST'])
def set_relay():
    global relay_status
    data = request.get_json()

    status = data.get("status")

    if status == "ON":
        relay_status = "ON"
    elif status == "OFF":
        relay_status = "OFF"

    return jsonify({"relay": relay_status})


# ================= DASHBOARD =================
@app.route('/api/latest')
def api_latest():
    return jsonify(latest_result)


# ================= RUN (WAJIB UNTUK RAILWAY) =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
