# app.py
import os
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import joblib
import numpy as np

# --- Setup ---
load_dotenv()
app = Flask(__name__)
CORS(app)

# --- Configure the Generative AI Model ---
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    gen_model = genai.GenerativeModel('gemini-1.5-flash')
    print("✅ Generative AI model configured successfully.")
except Exception as e:
    print(f"❌ Error configuring Generative AI: {e}")
    gen_model = None

# --- Load ML Models ---
regressor = joblib.load('power_forecasting_model.joblib')
scaler = joblib.load('scaler.joblib')
kmeans = joblib.load('kmeans_model.joblib')
print("✅ ML models and scaler loaded successfully.")

EXPECTED_FEATURES = [
    'global_active_power', 'global_reactive_power', 'voltage',
    'global_intensity', 'sub_metering_1', 'sub_metering_2',
    'sub_metering_3', 'total_sub_metering', 'hour', 'weekday'
]
CLUSTER_MAP = {0: "Evening Peak", 1: "Low Night Usage", 2: "Weekday Daytime"}

# --- UPDATED: Simplified AI prompt ---
def get_generative_recommendation(context):
    if not gen_model:
        return "Smart recommendations are currently unavailable."

    prompt = f"""
    You are an AI Energy Assistant for a smart home.
    Based on the following data, provide a single, short, friendly, and actionable recommendation (under 30 words) to help the user save energy.

    Context:
    - Predicted Power Consumption for the next hour: {context['prediction']:.2f} kW
    - Detected Usage Pattern: {context['pattern']}
    - Current Hour: {context['hour']}
    - Sub Metering 3 (likely Heater/AC) is using: {context['sub_metering_3']} kW

    Provide a new, helpful energy-saving tip based on the context.
    """
    try:
        response = gen_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"❌ Error generating content from Gemini: {e}")
        return "Could not generate a recommendation at this time."


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_values = [data[feature] for feature in EXPECTED_FEATURES]
        input_array = np.array(input_values).reshape(1, -1)
        
        power_prediction = regressor.predict(input_array)
        
        input_scaled = scaler.transform(input_array)
        cluster_index = kmeans.predict(input_scaled)
        pattern_prediction = CLUSTER_MAP.get(cluster_index[0], "Unknown")

        context = {**data, 'prediction': power_prediction[0], 'pattern': pattern_prediction}
        recommendation = get_generative_recommendation(context)

        return jsonify({
            'prediction': power_prediction[0],
            'pattern': pattern_prediction,
            'recommendation': recommendation
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)