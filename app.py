from flask import Flask, request, jsonify, render_template
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)
model = load_model('model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_data():
    try:
        file = request.files['file']
        data = pd.read_csv(file)
        return jsonify(data.head().to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract the JSON data from the request
        data = request.json
        
        # Ensure the JSON data is in the correct format
        if not isinstance(data, list):
            return jsonify({"error": "Input data must be a list of dictionaries"}), 400

        # Convert JSON data to a DataFrame
        df = pd.DataFrame(data)

        # Ensure columns exist
        required_columns = ['Value1', 'Value2']
        if not all(col in df.columns for col in required_columns):
            return jsonify({"error": "Missing required columns"}), 400

        # Prepare input data for prediction
        input_data = df[required_columns].values

        # Make predictions
        predictions = model.predict(input_data)

        return jsonify(predictions.tolist())
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
