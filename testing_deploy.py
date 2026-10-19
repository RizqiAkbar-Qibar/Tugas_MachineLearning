from flask import Flask, request, jsonify
import joblib

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Memuat model klasterisasi dan scaler yang sesuai
kmeans_model = joblib.load('kmeans_energy_model.joblib')
energy_scaler = joblib.load('energy_scaler.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    # Mengambil data JSON dari request
    json_data = request.json['data']  # Format 2D array, contoh: [[2500, 2600, 2700, 2800, 2900]]
    
    # 1. Lakukan scaling pada data baru menggunakan scaler yang sama saat training
    scaled_data = energy_scaler.transform(json_data)
    
    # 2. Prediksi klaster (menghasilkan angka 0, 1, atau 2)
    cluster_preds = kmeans_model.predict(scaled_data)
    
    # 3. Mapping hasil prediksi angka ke label deskriptif (Rendah, Sedang, Tinggi)
    # (Opsional: bisa disesuaikan dengan mapping klaster Anda sebelumnya)
    label_mapping = {0: 'Rendah', 1: 'Sedang', 2: 'Tinggi'}
    labeled_preds = [label_mapping.get(c, str(c)) for c in cluster_preds]
    
    return jsonify({
        'cluster_index': cluster_preds.tolist(),
        'cluster_label': labeled_preds
    })

if __name__ == '__main__':
    app.run(debug=True)