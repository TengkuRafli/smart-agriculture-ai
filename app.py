from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os

app = Flask(__name__)

# load model
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        temperature = float(request.form.get('temperature'))
        humidity = float(request.form.get('humidity'))
        soil_moisture = float(request.form.get('soil_moisture'))

        # dummy sementara
        N, P, K, ph = 50, 50, 50, 6.5

        data = np.array([[N, P, K, temperature, humidity, ph, soil_moisture]])
        data_scaled = scaler.transform(data)

        pred = model.predict(data_scaled)
        hasil = encoder.inverse_transform(pred)

        return render_template(
            'index.html',
            prediction=hasil[0],
            temperature=temperature,
            humidity=humidity,
            soil_moisture=soil_moisture
        )

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()

        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        soil_moisture = float(data['soil_moisture'])

        N, P, K, ph = 50, 50, 50, 6.5

        input_data = np.array([[N, P, K, temperature, humidity, ph, soil_moisture]])
        input_scaled = scaler.transform(input_data)

        pred = model.predict(input_scaled)
        hasil = encoder.inverse_transform(pred)

        return jsonify({
            "prediction": hasil[0]
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# 🔥 WAJIB untuk Railway
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
