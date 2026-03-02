from flask import Flask, render_template, request
import pickle
import numpy as np
from datetime import datetime

# 1. INITIALIZE THE APP FIRST (This fixes your NameError)
app = Flask(__name__)

# 2. LOAD YOUR ASSETS
# Ensure these files are in your my_flask_app folder
try:
    model = pickle.load(open('Random_Forest.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except FileNotFoundError:
    print("Error: .pkl files not found. Please check your directory.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_page')
def predict_page():
    return render_template('predict.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/theory')
def theory():
    return render_template('theory.html')

@app.route('/predict', methods=['POST'])
def predict():
    summary = {} # Initialize to prevent Jinja2 UndefinedError
    try:
        # Capture Data using .get() for safety
        user_name = request.form.get('name', 'Guest Patient')
        raw_age = float(request.form.get('age', 0))
        gender_val = int(request.form.get('gender', 0))
        height = float(request.form.get('height', 0))
        weight = float(request.form.get('weight', 0))
        ap_hi = float(request.form.get('ap_hi', 0))
        ap_lo = float(request.form.get('ap_lo', 0))
        
        smoke = 1 if 'smoke' in request.form else 0
        alco = 1 if 'alco' in request.form else 0
        active = 1 if 'active' in request.form else 0
        chol_val = int(request.form.get('chol', 1))
        gluc_val = int(request.form.get('gluc', 1))
        
        # BMI Calculation
        bmi_calc = weight / ((height / 100) ** 2) if height > 0 else 0
        
        # Build Summary for the Report
        summary = {
            "Patient Name": user_name,
            "Age": f"{int(raw_age)} Years",
            "Gender": "Male" if gender_val == 1 else "Female",
            "Blood Pressure": f"{int(ap_hi)}/{int(ap_lo)} mmHg",
            "BMI": round(bmi_calc, 2),
            "Cholesterol": ["Normal", "Above Normal", "High"][chol_val-1],
            "Glucose": ["Normal", "Above Normal", "High"][gluc_val-1],
            "Lifestyle": f"{'Smoker' if smoke else 'Non-Smoker'}, {'Active' if active else 'Sedentary'}"
        }

        # Scaling numericals (assuming 6 inputs for scaler: age, h, w, hi, lo, bmi)
        to_be_scaled = np.array([[raw_age, height, weight, ap_hi, ap_lo, bmi_calc]])
        scaled_values = scaler.transform(to_be_scaled)[0]

        # Construct final 14-feature vector
        features = [
            scaled_values[0], gender_val, scaled_values[1], scaled_values[2],
            scaled_values[3], scaled_values[4], smoke, alco, active,
            scaled_values[5], 
            1 if chol_val == 2 else 0, 1 if chol_val == 3 else 0,
            1 if gluc_val == 2 else 0, 1 if gluc_val == 3 else 0
        ]

        # Prediction
        prediction = model.predict([features])[0]
        result = "⚠️ High Risk" if prediction == 1 else "✅ Low Risk"
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

        return render_template('predict.html', 
                               prediction_text=result, 
                               summary=summary, 
                               now=current_time)

    except Exception as e:
        # Fallback to prevent template crash
        return render_template('predict.html', 
                               prediction_text=f"Error: {str(e)}", 
                               summary=summary,
                               now=datetime.now().strftime("%H:%M"))

if __name__ == "__main__":
    app.run(debug=True)