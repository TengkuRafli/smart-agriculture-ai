import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =========================
# 1. LOAD DATA
# =========================
data = pd.read_csv("data_tanaman_updated.csv", sep=';')

print("Sebelum cleaning:")
print(data.head())

# =========================
# 2. BERSIHKAN DATA 🔥
# =========================
fitur = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'soil_moisture']

for col in fitur:
    # ubah ke string dulu
    data[col] = data[col].astype(str)
    
    # hapus spasi
    data[col] = data[col].str.strip()
    
    # ubah koma jadi titik
    data[col] = data[col].str.replace(',', '.', regex=False)
    
    # ubah ke angka (kalau error jadi NaN)
    data[col] = pd.to_numeric(data[col], errors='coerce')

# hapus data yang rusak (NaN)
data = data.dropna()

print("\nSetelah cleaning:")
print(data.head())

# =========================
# 3. FITUR & LABEL
# =========================
X = data[fitur]
y = data['label']

print("Jumlah data setelah cleaning:", X.shape)

# =========================
# 4. ENCODING
# =========================
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# =========================
# 5. SCALING
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 6. SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# =========================
# 7. TRAIN MODEL
# =========================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# =========================
# 8. EVALUASI
# =========================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Akurasi model:", acc)

# =========================
# 9. SIMPAN MODEL
# =========================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))

print("🔥 Model berhasil disimpan!")