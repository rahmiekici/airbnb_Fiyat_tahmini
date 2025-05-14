from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

# Modeli yükle
model = joblib.load('lgb_model.pkl')  # modelini burada pickle'lamış olmalısın

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # JSON'u doğrudan DataFrame'e çevir
        df = pd.DataFrame({k: v for k, v in data.items()})
        
        # Tahmin yap
        prediction = model.predict(df)
        return jsonify({'prediction': float(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
