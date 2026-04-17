import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# load data
data = pd.read_csv("data_tanaman.csv")

X = data[['temperature', 'humidity']]
y = data['label']
 
# encoding label
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# model
model = RandomForestClassifier()
model.fit(X_scaled, y_encoded)

# simpan
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))

print("Model berhasil disimpan!")