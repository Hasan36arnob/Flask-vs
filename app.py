from flask import Flask, request, jsonify, render_template, redirect, url_for
import joblib
import numpy as np
from pymongo import MongoClient
import datetime

app = Flask(__name__)

# MongoDB connection using the provided connection string
mongo_uri = "mongodb+srv://arnobhasanice:NVSZUMkLUTWnfFXR@fl1.qnsxy.mongodb.net/?retryWrites=true&w=majority&appName=fl1"
client = MongoClient(mongo_uri)
db = client['diabetes_db']
collection = db['predictions']

# Define model version
model_version = "1.0.0"

# Load the trained model and label encoders
try:
    model = joblib.load('diabetes_model.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
except Exception as e:
    print(f"Error loading model or encoders: {e}")
    model = None
    label_encoders = None

# Define default feature importances (equal importance if model doesn't provide them)
default_feature_importances = {
    'Age': 0.15,
    'Gender': 0.05,
    'Polyuria': 0.12,
    'Polydipsia': 0.12,
    'sudden weight loss': 0.08,
    'weakness': 0.06,
    'Polyphagia': 0.07,
    'visual blurring': 0.07,
    'Itching': 0.05,
    'Irritability': 0.05,
    'delayed healing': 0.06,
    'partial paresis': 0.06,
    'muscle stiffness': 0.05,
    'Alopecia': 0.03,
    'Obesity': 0.08
}

# Routes
@app.route('/')
def home():
    """Home page route"""
    return render_template('index.html')

@app.route('/about')
def about():
    """About page route"""
    return render_template('about.html')

@app.route('/contact')
def contact():
    """Contact page route"""
    return render_template('contact.html')

@app.route('/diet')
def diet():
    return render_template('diet.html')

@app.route('/exercise')
def exercise():
    return render_template('exercise.html')

@app.route('/tracking')
def tracking():
    return render_template('tracking.html')

@app.route('/medication')
def medication():
    return render_template('medication.html')

@app.route('/doctor')
def doctor():
    return render_template('doctor.html')

@app.route('/predict_page')
def predict_page():
    """Route to the prediction page"""
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction route for diabetes with risk categorization"""
    if model is None or label_encoders is None:
        return render_template('predict.html', prediction_text="Error: Model or encoders not loaded.")
    
    try:
        # Parse input data from the form
        input_data = {}
        
        # Handle Age separately since it's numeric
        age = request.form.get('Age', '').strip()
        input_data['Age'] = int(age) if age else 0  # Use 0 or another default value for empty age
        
        # List of all categorical fields
        categorical_fields = [
            'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss',
            'weakness', 'Polyphagia', 'visual blurring', 'Itching',
            'Irritability', 'delayed healing', 'partial paresis',
            'muscle stiffness', 'Alopecia', 'Obesity'
        ]
        
        # Handle categorical inputs
        for field in categorical_fields:
            value = request.form.get(field, '').strip()
            input_data[field] = value if value else 'No Answer'  # Use 'No Answer' for empty fields
        
        # Encode categorical inputs
        features = []
        for key in input_data.keys():
            if key == 'Age':
                features.append(input_data[key])
            else:
                # For empty or 'No Answer' values, use a default encoding
                if input_data[key] == 'No Answer':
                    features.append(0)  # Use 0 as default encoding for missing values
                else:
                    features.append(int(label_encoders[key].transform([input_data[key]])[0]))

        # Convert to numpy array and reshape
        features = np.array(features).reshape(1, -1)

        # Calculate confidence score with improved weighting
        answered_questions = sum(1 for k, v in input_data.items() if v not in ['', 'No Answer', 0])
        total_questions = len(input_data)

        # Apply a logarithmic scale to confidence score
        import math
        confidence_factor = math.log(1 + 9 * (answered_questions / total_questions)) / math.log(10)
        confidence_score = confidence_factor * 100

        # Make prediction and calculate probability
        probabilities = model.predict_proba(features)
        probability_positive = float(probabilities[0][1] * 100)

        # Get feature importances - use default if model doesn't provide them
        try:
            # Try to get feature importances from model
            feature_importance_values = model.feature_importances_
            feature_importance = dict(zip(input_data.keys(), feature_importance_values))
        except AttributeError:
            # If model doesn't have feature_importances_, use our default values
            feature_importance = {}
            for key in input_data.keys():
                feature_importance[key] = default_feature_importances.get(key, 0.05)

        # Weight confidence adjustment 
        if feature_importance:
            importance_sum = sum(feature_importance.values())
            weighted_importance = sum(importance / importance_sum for importance in feature_importance.values())
            importance_factor = (weighted_importance / len(feature_importance)) if len(feature_importance) > 0 else 0
            adjusted_confidence = confidence_score * (0.7 + 0.3 * importance_factor)
        else:
            adjusted_confidence = confidence_score * 0.7

        # Adjust probability based on confidence score
        adjusted_probability = probability_positive * (adjusted_confidence / 100)

        # Categorize risk
        if adjusted_probability > 70:
            risk_category = "Very High Risk"
            color = "#8B0000"
        elif adjusted_probability > 55:
            risk_category = "High Risk"
            color = "#FF0000"
        elif adjusted_probability > 40:
            risk_category = "Moderate Risk"
            color = "#FFA500"
        elif adjusted_probability > 25:
            risk_category = "Low Risk"
            color = "#FFFF00"
        else:
            risk_category = "Very Low Risk"
            color = "#008000"

        # Format prediction text
        prediction_text = (
            f'<div class="result-card">'
            f'  <h2>Risk Assessment Results</h2>'
            f'  <div class="risk-indicator" style="background-color:{color};">'
            f'    <h3>Risk of Developing Diabetes: {risk_category}</h3>'
            f'    <p>Estimated Risk: {adjusted_probability:.1f}%</p>'
            f'  </div>'
            f'  <div class="confidence-info">'
            f'    <p>Confidence Level: {adjusted_confidence:.1f}%</p>'
            f'    <p>Based on {answered_questions} of {total_questions} answered questions</p>'
            f'    <p>Data completeness: {(answered_questions / total_questions * 100):.1f}%</p>'
            f'  </div>'
            f'</div>'
        )

        # Save prediction to MongoDB
        prediction_record = {
            'input_data': {k: int(v) if isinstance(v, np.integer) else v for k, v in input_data.items()},
            'probability_raw': float(probability_positive),
            'probability_adjusted': float(adjusted_probability),
            'confidence_score': float(adjusted_confidence),
            'data_completeness': float(answered_questions / total_questions * 100),
            'risk_category': risk_category,
            'answered_questions': answered_questions,
            'total_questions': total_questions,
            'timestamp': datetime.datetime.now(),
            'model_version': model_version
        }
        collection.insert_one(prediction_record)

        # Feature importance ranking
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_factors = sorted_features[:min(5, len(sorted_features))]

        return render_template(
            'result.html',
            prediction_text=prediction_text,
            risk_category=risk_category,
            probability=adjusted_probability,
            confidence_score=adjusted_confidence,
            data_completeness=answered_questions / total_questions * 100,
            sorted_features=sorted_features,
            top_factors=top_factors,
            back_url=url_for('home')
        )

    except Exception as e:
        return render_template('predict.html', prediction_text=f"Error during prediction: {str(e)}")

@app.route('/suggest', methods=['GET'])
def suggest():
    """Suggestion route based on feature ranking"""
    try:
        # Fetch the latest prediction from MongoDB
        latest_prediction = collection.find_one(sort=[('_id', -1)])
        if not latest_prediction:
            return render_template('suggest.html', suggestions="No predictions found.")

        # Get features where user actually provided positive responses
        input_data = latest_prediction['input_data']
        relevant_features = []
        
        # Get feature importances - use default if not available
        try:
            feature_importance = dict(zip(input_data.keys(), model.feature_importances_))
        except AttributeError:
            feature_importance = default_feature_importances

        for feature, value in input_data.items():
            # Check if the value indicates a positive response
            if value in [1, 'Yes', True]:
                relevant_features.append((feature, feature_importance.get(feature, 0.05)))

        # Sort by feature importance and get top 3
        sorted_features = sorted(relevant_features, key=lambda x: x[1], reverse=True)[:3]

        # Generate suggestions
        suggestions = []
        for feature, _ in sorted_features:
            if feature == 'Age':
                suggestions.append("Monitor your age-related health risks regularly.")
            elif feature == 'Gender':
                suggestions.append("Be aware of gender-specific risk factors for diabetes.")
            elif feature == 'Polyuria':
                suggestions.append("Consult a doctor if you experience frequent urination.")
            elif feature == 'Polydipsia':
                suggestions.append("Stay hydrated but consult a doctor if excessive thirst persists.")
            elif feature == 'sudden weight loss':
                suggestions.append("Seek medical advice for unexplained weight loss.")
            elif feature == 'weakness':
                suggestions.append("Ensure proper nutrition and rest to manage weakness.")
            elif feature == 'Polyphagia':
                suggestions.append("Monitor your eating habits and consult a doctor if needed.")
            elif feature == 'visual blurring':
                suggestions.append("Get your eyes checked if you experience blurred vision.")
            elif feature == 'Itching':
                suggestions.append("Use moisturizers and consult a dermatologist if itching persists.")
            elif feature == 'Irritability':
                suggestions.append("Practice stress management techniques.")
            elif feature == 'delayed healing':
                suggestions.append("Seek medical advice for slow-healing wounds.")
            elif feature == 'partial paresis':
                suggestions.append("Consult a neurologist for muscle weakness.")
            elif feature == 'muscle stiffness':
                suggestions.append("Stretch regularly and consult a doctor if stiffness persists.")
            elif feature == 'Alopecia':
                suggestions.append("Consult a dermatologist for hair loss issues.")
            elif feature == 'Obesity':
                suggestions.append("Adopt a healthy diet and exercise regularly.")

        return render_template('suggest.html', suggestions=suggestions, back_url=url_for('home'))
    except Exception as e:
        return render_template('suggest.html', suggestions=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)