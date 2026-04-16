from flask import Flask, render_template, request
import joblib
import os
import numpy as np

app = Flask(__name__)

# Load model and preprocessors
model_path = os.path.join('models', 'best_model.pkl')
scaler_path = os.path.join('models', 'scaler.pkl')

# Attempt to load, handled gracefully if absent
if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
else:
    model = None
    scaler = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return render_template('result.html', prediction_text="Model not found. Please train the model first by running model_trainer.py.", final=False)
        
    if request.method == 'POST':
        try:
            attendance = float(request.form['attendance'])
            study_hours = float(request.form['study_hours'])
            previous_grade = float(request.form['previous_grade'])
            participation = float(request.form['participation'])
            extracurricular = request.form['extracurricular']
            
            # Feature engineering (mapping to binary as done in training)
            extracurricular_val = 1 if extracurricular == 'Yes' else 0
            
            # Predict the final score
            features = np.array([[attendance, study_hours, previous_grade, participation, extracurricular_val]])
            features_scaled = scaler.transform(features)
            
            prediction = model.predict(features_scaled)
            final_pred = round(prediction[0], 2)
            
            # Ensure the prediction is within plausible bounds (0 to 100)
            final_pred = min(max(final_pred, 0), 100)
            
            # Generate actionable insights based on predicted score
            if final_pred >= 85:
                message = "Excellent! The student is on track for a top grade."
                msg_class = "success"
                icon = "fa-check-circle"
            elif final_pred >= 70:
                message = "Good performance, but there is room for improvement."
                msg_class = "info"
                icon = "fa-thumbs-up"
            elif final_pred >= 50:
                message = "Average. The student might need additional focus to secure a better grade."
                msg_class = "warning"
                icon = "fa-exclamation-triangle"
            else:
                message = "At Risk! The student requires immediate tutoring or intervention."
                msg_class = "danger"
                icon = "fa-radiation"
                
            prediction_text = f"{final_pred}%"
                
            return render_template('result.html', 
                                   prediction_text=prediction_text, 
                                   message=message,
                                   msg_class=msg_class,
                                   icon=icon,
                                   final=True)
                                   
        except Exception as e:
            return render_template('result.html', prediction_text=f"Error occurred: {str(e)}", final=False)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
