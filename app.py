from flask import Flask, render_template, request
from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # input dari form (yang kamu punya)
        temperature = float(request.form.get('temperature'))
        humidity = float(request.form.get('humidity'))
        soil_moisture = float(request.form.get('soil_moisture'))

        # 🔥 nilai dummy (bisa kamu ubah biar realistis)
        N = 50
        P = 50
        K = 50
        ph = 6.5

        # urutan HARUS sama seperti training
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

        # dummy (sementara)
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

if __name__ == "__main__":
    app.run()
