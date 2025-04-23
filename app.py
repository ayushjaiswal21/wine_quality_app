# app.py
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load models and scaler
knn_model = joblib.load('models/knn_model.pkl')
svm_model = joblib.load('models/svm_model.pkl')
scaler = joblib.load('models/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from form
    features = [
        float(request.form['fixed_acidity']),
        float(request.form['volatile_acidity']),
        float(request.form['citric_acid']),
        float(request.form['residual_sugar']),
        float(request.form['chlorides']),
        float(request.form['free_sulfur_dioxide']),
        float(request.form['total_sulfur_dioxide']),
        float(request.form['density']),
        float(request.form['ph']),
        float(request.form['sulphates']),
        float(request.form['alcohol'])
    ]
    
    # Scale features
    input_data = np.array(features).reshape(1, -1)
    scaled_data = scaler.transform(input_data)
    
    # Get model type
    model_type = request.form['model_type']
    
    # Predict
    if model_type == 'knn':
        prediction = knn_model.predict(scaled_data)
        result = f'Predicted Quality Score: {prediction[0]}'
    else:
        prediction = svm_model.predict(scaled_data)
        result = 'Good Quality' if prediction[0] else 'Bad Quality'
    
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)