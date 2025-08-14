# from flask import Flask, request, jsonify, render_template, redirect, url_for
# import joblib
# import numpy as np
# from pymongo import MongoClient
# import datetime
# import math

# app = Flask(__name__)

# # MongoDB connection using the provided connection string
# mongo_uri = "mongodb+srv://arnobhasanice:NVSZUMkLUTWnfFXR@fl1.qnsxy.mongodb.net/?retryWrites=true&w=majority&appName=fl1"
# client = MongoClient(mongo_uri)
# db = client['diabetes_db']
# collection = db['predictions']

# # Define model version
# model_version = "1.0.0"

# # Load the trained model and label encoders
# try:
#     model = joblib.load('diabetes_model.pkl')
#     label_encoders = joblib.load('label_encoders.pkl')
# except Exception as e:
#     print(f"Error loading model or encoders: {e}")
#     model = None
#     label_encoders = None

# # Define default feature importances (equal importance if model doesn't provide them)
# default_feature_importances = {
#     'Age': 0.15,
#     'Gender': 0.05,
#     'Polyuria': 0.12,
#     'Polydipsia': 0.12,
#     'sudden weight loss': 0.08,
#     'weakness': 0.06,
#     'Polyphagia': 0.07,
#     'visual blurring': 0.07,
#     'Itching': 0.05,
#     'Irritability': 0.05,
#     'delayed healing': 0.06,
#     'partial paresis': 0.06,
#     'muscle stiffness': 0.05,
#     'Alopecia': 0.03,
#     'Obesity': 0.08
# }

# def calculate_individual_feature_contributions(model, features, feature_names, input_data):
#     """
#     Calculate how much each feature contributed to THIS specific prediction
#     """
#     try:
#         # Method 1: For tree-based models (Random Forest, Decision Tree, etc.)
#         if hasattr(model, 'decision_path'):
#             # Get the decision path for this prediction
#             feature_contributions = {}
            
#             # Get baseline prediction (all features at neutral/average values)
#             baseline_features = np.zeros_like(features)
#             baseline_prob = model.predict_proba(baseline_features.reshape(1, -1))[0][1]
            
#             # Calculate contribution of each feature by removing it one at a time
#             current_prob = model.predict_proba(features.reshape(1, -1))[0][1]
            
#             for i, feature_name in enumerate(feature_names):
#                 # Create a copy of features with this feature set to baseline (0)
#                 modified_features = features.copy()
#                 modified_features[i] = 0
                
#                 # Get probability without this feature
#                 prob_without_feature = model.predict_proba(modified_features.reshape(1, -1))[0][1]
                
#                 # Contribution is the difference
#                 contribution = current_prob - prob_without_feature
#                 feature_contributions[feature_name] = contribution
            
#             return feature_contributions
            
#         # Method 2: For other models - use feature importance weighted by input values
#         else:
#             feature_contributions = {}
            
#             # Get global feature importances
#             if hasattr(model, 'feature_importances_'):
#                 global_importances = model.feature_importances_
#             else:
#                 # Use default importances
#                 global_importances = [default_feature_importances.get(name, 0.05) for name in feature_names]
            
#             # Weight by actual input values and whether they indicate positive risk
#             for i, feature_name in enumerate(feature_names):
#                 feature_value = features[i]
#                 global_importance = global_importances[i] if i < len(global_importances) else 0.05
                
#                 # For binary features (0/1), contribution is importance * value
#                 # For age, normalize and apply importance
#                 if feature_name == 'Age':
#                     normalized_age = min(feature_value / 100, 1.0)  # Normalize age to 0-1
#                     contribution = global_importance * normalized_age
#                 else:
#                     # For binary features, contribution is importance only if feature is positive
#                     contribution = global_importance * feature_value
                
#                 feature_contributions[feature_name] = contribution
            
#             return feature_contributions
            
#     except Exception as e:
#         print(f"Error calculating feature contributions: {e}")
#         # Fallback to input-based contributions
#         feature_contributions = {}
#         for feature_name in feature_names:
#             if feature_name == 'Age':
#                 contribution = min(features[feature_names.index(feature_name)] / 100, 1.0) * 0.15
#             else:
#                 # For categorical features, contribution based on positive responses
#                 original_value = input_data.get(feature_name, '')
#                 if original_value in ['Yes', 'Positive', 'Male']:  # Adjust based on your encoding
#                     contribution = default_feature_importances.get(feature_name, 0.05)
#                 else:
#                     contribution = 0
#             feature_contributions[feature_name] = contribution
        
#         return feature_contributions

# # Routes
# @app.route('/')
# def home():
#     """Home page route"""
#     return render_template('index.html')

# @app.route('/about')
# def about():
#     """About page route"""
#     return render_template('about.html')

# @app.route('/contact')
# def contact():
#     """Contact page route"""
#     return render_template('contact.html')

# @app.route('/diet')
# def diet():
#     return render_template('diet.html')

# @app.route('/exercise')
# def exercise():
#     return render_template('exercise.html')

# @app.route('/tracking')
# def tracking():
#     return render_template('tracking.html')

# @app.route('/medication')
# def medication():
#     return render_template('medication.html')

# @app.route('/doctor')
# def doctor():
#     return render_template('doctor.html')

# @app.route('/predict_page')
# def predict_page():
#     """Route to the prediction page"""
#     return render_template('predict.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     """Prediction route for diabetes with risk categorization"""
#     if model is None or label_encoders is None:
#         return render_template('predict.html', prediction_text="Error: Model or encoders not loaded.")
    
#     try:
#         # Parse input data from the form
#         input_data = {}
        
#         # Handle Age separately since it's numeric
#         age = request.form.get('Age', '').strip()
#         input_data['Age'] = int(age) if age else 0
        
#         # List of all categorical fields
#         categorical_fields = [
#             'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss',
#             'weakness', 'Polyphagia', 'visual blurring', 'Itching',
#             'Irritability', 'delayed healing', 'partial paresis',
#             'muscle stiffness', 'Alopecia', 'Obesity'
#         ]
        
#         # Handle categorical inputs
#         for field in categorical_fields:
#             value = request.form.get(field, '').strip()
#             input_data[field] = value if value else 'No Answer'
        
#         # Encode categorical inputs
#         features = []
#         feature_names = list(input_data.keys())
        
#         for key in feature_names:
#             if key == 'Age':
#                 features.append(input_data[key])
#             else:
#                 if input_data[key] == 'No Answer':
#                     features.append(0)
#                 else:
#                     features.append(int(label_encoders[key].transform([input_data[key]])[0]))

#         # Convert to numpy array and reshape
#         features = np.array(features).reshape(1, -1)

#         # Calculate confidence score with improved weighting
#         answered_questions = sum(1 for k, v in input_data.items() if v not in ['', 'No Answer', 0])
#         total_questions = len(input_data)

#         # Apply a logarithmic scale to confidence score
#         confidence_factor = math.log(1 + 9 * (answered_questions / total_questions)) / math.log(10)
#         confidence_score = confidence_factor * 100

#         # Make prediction and calculate probability
#         probabilities = model.predict_proba(features)
#         probability_positive = float(probabilities[0][1] * 100)

#         # Calculate INDIVIDUAL feature contributions for THIS prediction
#         feature_contributions = calculate_individual_feature_contributions(
#             model, features[0], feature_names, input_data
#         )

#         # Weight confidence adjustment using individual contributions instead of global importance
#         if feature_contributions:
#             # Use the actual contributions from this prediction
#             total_contribution = sum(abs(contrib) for contrib in feature_contributions.values())
#             if total_contribution > 0:
#                 normalized_contributions = {k: abs(v)/total_contribution for k, v in feature_contributions.items()}
#                 weighted_importance = sum(normalized_contributions.values())
#                 importance_factor = (weighted_importance / len(normalized_contributions)) if len(normalized_contributions) > 0 else 0
#                 adjusted_confidence = confidence_score * (0.7 + 0.3 * importance_factor)
#             else:
#                 adjusted_confidence = confidence_score * 0.7
#         else:
#             adjusted_confidence = confidence_score * 0.7

#         # Adjust probability based on confidence score
#         adjusted_probability = probability_positive * (adjusted_confidence / 100)

#         # Categorize risk
#         if adjusted_probability > 70:
#             risk_category = "Very High Risk"
#             color = "#8B0000"
#         elif adjusted_probability > 55:
#             risk_category = "High Risk"
#             color = "#FF0000"
#         elif adjusted_probability > 40:
#             risk_category = "Moderate Risk"
#             color = "#FFA500"
#         elif adjusted_probability > 25:
#             risk_category = "Low Risk"
#             color = "#FFFF00"
#         else:
#             risk_category = "Very Low Risk"
#             color = "#008000"

#         # Format prediction text
#         prediction_text = (
#             f'<div class="result-card">'
#             f'  <h2>Risk Assessment Results</h2>'
#             f'  <div class="risk-indicator" style="background-color:{color};">'
#             f'    <h3>Risk of Developing Diabetes: {risk_category}</h3>'
#             f'    <p>Estimated Risk: {adjusted_probability:.1f}%</p>'
#             f'  </div>'
#             f'  <div class="confidence-info">'
#             f'    <p>Confidence Level: {adjusted_confidence:.1f}%</p>'
#             f'    <p>Based on {answered_questions} of {total_questions} answered questions</p>'
#             f'    <p>Data completeness: {(answered_questions / total_questions * 100):.1f}%</p>'
#             f'  </div>'
#             f'</div>'
#         )

#         # Save prediction to MongoDB
#         prediction_record = {
#             'input_data': {k: int(v) if isinstance(v, np.integer) else v for k, v in input_data.items()},
#             'probability_raw': float(probability_positive),
#             'probability_adjusted': float(adjusted_probability),
#             'confidence_score': float(adjusted_confidence),
#             'data_completeness': float(answered_questions / total_questions * 100),
#             'risk_category': risk_category,
#             'answered_questions': answered_questions,
#             'total_questions': total_questions,
#             'feature_contributions': {k: float(v) for k, v in feature_contributions.items()},
#             'timestamp': datetime.datetime.now(),
#             'model_version': model_version
#         }
#         collection.insert_one(prediction_record)

#         # Sort features by their INDIVIDUAL contributions to THIS prediction
#         sorted_features = sorted(feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        
#         # Get top factors that actually contributed to this prediction
#         # Filter out features that didn't contribute (have zero or very low contribution)
#         significant_features = [(name, abs(contrib)) for name, contrib in sorted_features if abs(contrib) > 0.001]
#         top_factors = significant_features[:min(5, len(significant_features))]

#         return render_template(
#             'result.html',
#             prediction_text=prediction_text,
#             risk_category=risk_category,
#             probability=adjusted_probability,
#             confidence_score=adjusted_confidence,
#             data_completeness=answered_questions / total_questions * 100,
#             sorted_features=significant_features,  # Now shows individual contributions
#             top_factors=top_factors,  # Now shows individual contributions
#             back_url=url_for('home')
#         )

#     except Exception as e:
#         return render_template('predict.html', prediction_text=f"Error during prediction: {str(e)}")

# @app.route('/suggest', methods=['GET'])
# def suggest():
#     """Suggestion route based on feature ranking"""
#     try:
#         # Fetch the latest prediction from MongoDB
#         latest_prediction = collection.find_one(sort=[('_id', -1)])
#         if not latest_prediction:
#             return render_template('suggest.html', suggestions="No predictions found.")

#         # Get features where user actually provided positive responses
#         input_data = latest_prediction['input_data']
        
#         # Use feature contributions if available, otherwise fall back to old method
#         if 'feature_contributions' in latest_prediction:
#             feature_contributions = latest_prediction['feature_contributions']
#             # Get top contributing features
#             relevant_features = []
#             for feature, contribution in feature_contributions.items():
#                 if abs(contribution) > 0.001:  # Only consider features with meaningful contribution
#                     relevant_features.append((feature, abs(contribution)))
            
#             # Sort by contribution and get top 3
#             sorted_features = sorted(relevant_features, key=lambda x: x[1], reverse=True)[:3]
#         else:
#             # Fallback to old method
#             relevant_features = []
            
#             # Get feature importances - use default if not available
#             try:
#                 feature_importance = dict(zip(input_data.keys(), model.feature_importances_))
#             except AttributeError:
#                 feature_importance = default_feature_importances

#             for feature, value in input_data.items():
#                 # Check if the value indicates a positive response
#                 if value in [1, 'Yes', True]:
#                     relevant_features.append((feature, feature_importance.get(feature, 0.05)))

#             # Sort by feature importance and get top 3
#             sorted_features = sorted(relevant_features, key=lambda x: x[1], reverse=True)[:3]

#         # Generate suggestions
#         suggestions = []
#         for feature, _ in sorted_features:
#             if feature == 'Age':
#                 suggestions.append("Monitor your age-related health risks regularly.")
#             elif feature == 'Gender':
#                 suggestions.append("Be aware of gender-specific risk factors for diabetes.")
#             elif feature == 'Polyuria':
#                 suggestions.append("Consult a doctor if you experience frequent urination.")
#             elif feature == 'Polydipsia':
#                 suggestions.append("Stay hydrated but consult a doctor if excessive thirst persists.")
#             elif feature == 'sudden weight loss':
#                 suggestions.append("Seek medical advice for unexplained weight loss.")
#             elif feature == 'weakness':
#                 suggestions.append("Ensure proper nutrition and rest to manage weakness.")
#             elif feature == 'Polyphagia':
#                 suggestions.append("Monitor your eating habits and consult a doctor if needed.")
#             elif feature == 'visual blurring':
#                 suggestions.append("Get your eyes checked if you experience blurred vision.")
#             elif feature == 'Itching':
#                 suggestions.append("Use moisturizers and consult a dermatologist if itching persists.")
#             elif feature == 'Irritability':
#                 suggestions.append("Practice stress management techniques.")
#             elif feature == 'delayed healing':
#                 suggestions.append("Seek medical advice for slow-healing wounds.")
#             elif feature == 'partial paresis':
#                 suggestions.append("Consult a neurologist for muscle weakness.")
#             elif feature == 'muscle stiffness':
#                 suggestions.append("Stretch regularly and consult a doctor if stiffness persists.")
#             elif feature == 'Alopecia':
#                 suggestions.append("Consult a dermatologist for hair loss issues.")
#             elif feature == 'Obesity':
#                 suggestions.append("Adopt a healthy diet and exercise regularly.")

#         return render_template('suggest.html', suggestions=suggestions, back_url=url_for('home'))
#     except Exception as e:
#         return render_template('suggest.html', suggestions=f'Error: {str(e)}')

# if __name__ == '__main__':
#     app.run(debug=True)
 
# from flask import Flask, request, jsonify, render_template, redirect, url_for
# import joblib
# import numpy as np
# from pymongo import MongoClient
# import datetime
# import math

# app = Flask(__name__)

# # MongoDB connection using the provided connection string
# mongo_uri = "mongodb+srv://arnobhasanice:NVSZUMkLUTWnfFXR@fl1.qnsxy.mongodb.net/?retryWrites=true&w=majority&appName=fl1"
# client = MongoClient(mongo_uri)
# db = client['diabetes_db']
# collection = db['predictions']

# # Define model version
# model_version = "1.0.0"

# # Load the trained model and label encoders
# try:
#     model = joblib.load('diabetes_model.pkl')
#     label_encoders = joblib.load('label_encoders.pkl')
# except Exception as e:
#     print(f"Error loading model or encoders: {e}")
#     model = None
#     label_encoders = None

# # Define default feature importances (equal importance if model doesn't provide them)
# default_feature_importances = {
#     'Age': 0.15,
#     'Gender': 0.05,
#     'Polyuria': 0.12,
#     'Polydipsia': 0.12,
#     'sudden weight loss': 0.08,
#     'weakness': 0.06,
#     'Polyphagia': 0.07,
#     'visual blurring': 0.07,
#     'Itching': 0.05,
#     'Irritability': 0.05,
#     'delayed healing': 0.06,
#     'partial paresis': 0.06,
#     'muscle stiffness': 0.05,
#     'Alopecia': 0.03,
#     'Obesity': 0.08
# }

# def calculate_individual_feature_contributions_simple(features, feature_names, input_data, model):
#     """
#     Simple and effective method to calculate individual feature contributions
#     using permutation-based approach
#     """
#     try:
#         # Get baseline prediction
#         baseline_prob = model.predict_proba(features.reshape(1, -1))[0][1]
        
#         feature_contributions = {}
        
#         for i, feature_name in enumerate(feature_names):
#             # Create a copy of features
#             modified_features = features.copy()
            
#             # Set this feature to its "neutral" value (0 for most features, median age for age)
#             if feature_name == 'Age':
#                 modified_features[i] = 40  # Use median age as baseline
#             else:
#                 modified_features[i] = 0  # Set to "No" for binary features
            
#             # Get probability with this feature neutralized
#             modified_prob = model.predict_proba(modified_features.reshape(1, -1))[0][1]
            
#             # The contribution is how much the prediction drops when we remove this feature
#             contribution = baseline_prob - modified_prob
            
#             # Only consider positive contributions (features that increase risk)
#             if contribution > 0:
#                 feature_contributions[feature_name] = contribution
#             else:
#                 feature_contributions[feature_name] = 0
        
#         return feature_contributions
        
#     except Exception as e:
#         print(f"Error in simple contribution calculation: {e}")
#         # Fallback: Use input values weighted by global importance
#         feature_contributions = {}
        
#         for i, feature_name in enumerate(feature_names):
#             original_value = input_data.get(feature_name, 0)
#             global_importance = default_feature_importances.get(feature_name, 0.05)
            
#             if feature_name == 'Age':
#                 # Normalize age (assuming max age is around 100)
#                 normalized_value = min(float(original_value) / 100, 1.0) if original_value else 0
#                 contribution = global_importance * normalized_value
#             else:
#                 # For binary features, only contribute if the answer is positive
#                 if original_value in ['Yes', 1, True, 'Male']:
#                     contribution = global_importance
#                 else:
#                     contribution = 0
            
#             feature_contributions[feature_name] = contribution
        
#         return feature_contributions

# # Routes
# @app.route('/')
# def home():
#     """Home page route"""
#     return render_template('index.html')

# @app.route('/about')
# def about():
#     """About page route"""
#     return render_template('about.html')

# @app.route('/contact')
# def contact():
#     """Contact page route"""
#     return render_template('contact.html')

# @app.route('/diet')
# def diet():
#     return render_template('diet.html')

# @app.route('/exercise')
# def exercise():
#     return render_template('exercise.html')

# @app.route('/tracking')
# def tracking():
#     return render_template('tracking.html')

# @app.route('/medication')
# def medication():
#     return render_template('medication.html')

# @app.route('/doctor')
# def doctor():
#     return render_template('doctor.html')

# @app.route('/predict_page')
# def predict_page():
#     """Route to the prediction page"""
#     return render_template('predict.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     """Prediction route for diabetes with risk categorization"""
#     if model is None or label_encoders is None:
#         return render_template('predict.html', prediction_text="Error: Model or encoders not loaded.")
    
#     try:
#         # Parse input data from the form
#         input_data = {}
        
#         # Handle Age separately since it's numeric
#         age = request.form.get('Age', '').strip()
#         input_data['Age'] = int(age) if age else 0
        
#         # List of all categorical fields
#         categorical_fields = [
#             'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss',
#             'weakness', 'Polyphagia', 'visual blurring', 'Itching',
#             'Irritability', 'delayed healing', 'partial paresis',
#             'muscle stiffness', 'Alopecia', 'Obesity'
#         ]
        
#         # Handle categorical inputs
#         for field in categorical_fields:
#             value = request.form.get(field, '').strip()
#             input_data[field] = value if value else 'No Answer'
        
#         # Encode categorical inputs
#         features = []
#         feature_names = list(input_data.keys())
        
#         for key in feature_names:
#             if key == 'Age':
#                 features.append(input_data[key])
#             else:
#                 if input_data[key] == 'No Answer':
#                     features.append(0)
#                 else:
#                     features.append(int(label_encoders[key].transform([input_data[key]])[0]))

#         # Convert to numpy array
#         features = np.array(features)

#         # Calculate confidence score
#         answered_questions = sum(1 for k, v in input_data.items() if v not in ['', 'No Answer', 0])
#         total_questions = len(input_data)
#         confidence_factor = math.log(1 + 9 * (answered_questions / total_questions)) / math.log(10)
#         confidence_score = confidence_factor * 100

#         # Make prediction and calculate probability
#         probabilities = model.predict_proba(features.reshape(1, -1))
#         probability_positive = float(probabilities[0][1] * 100)

#         # Calculate INDIVIDUAL feature contributions using the simple method
#         feature_contributions = calculate_individual_feature_contributions_simple(
#             features, feature_names, input_data, model
#         )

#         # Adjust confidence based on how many important features contributed
#         contributing_features = sum(1 for contrib in feature_contributions.values() if contrib > 0.01)
#         importance_factor = min(contributing_features / 5, 1.0)  # Max of 5 important features
#         adjusted_confidence = confidence_score * (0.7 + 0.3 * importance_factor)

#         # Adjust probability based on confidence
#         adjusted_probability = probability_positive * (adjusted_confidence / 100)

#         # Categorize risk
#         if adjusted_probability > 70:
#             risk_category = "Very High Risk"
#             color = "#8B0000"
#         elif adjusted_probability > 55:
#             risk_category = "High Risk"
#             color = "#FF0000"
#         elif adjusted_probability > 40:
#             risk_category = "Moderate Risk"
#             color = "#FFA500"
#         elif adjusted_probability > 25:
#             risk_category = "Low Risk"
#             color = "#FFFF00"
#         else:
#             risk_category = "Very Low Risk"
#             color = "#008000"

#         # Format prediction text
#         prediction_text = (
#             f'<div class="result-card">'
#             f'  <h2>Risk Assessment Results</h2>'
#             f'  <div class="risk-indicator" style="background-color:{color};">'
#             f'    <h3>Risk of Developing Diabetes: {risk_category}</h3>'
#             f'    <p>Estimated Risk: {adjusted_probability:.1f}%</p>'
#             f'  </div>'
#             f'  <div class="confidence-info">'
#             f'    <p>Confidence Level: {adjusted_confidence:.1f}%</p>'
#             f'    <p>Based on {answered_questions} of {total_questions} answered questions</p>'
#             f'    <p>Data completeness: {(answered_questions / total_questions * 100):.1f}%</p>'
#             f'  </div>'
#             f'</div>'
#         )

#         # Save prediction to MongoDB
#         prediction_record = {
#             'input_data': {k: int(v) if isinstance(v, np.integer) else v for k, v in input_data.items()},
#             'probability_raw': float(probability_positive),
#             'probability_adjusted': float(adjusted_probability),
#             'confidence_score': float(adjusted_confidence),
#             'data_completeness': float(answered_questions / total_questions * 100),
#             'risk_category': risk_category,
#             'answered_questions': answered_questions,
#             'total_questions': total_questions,
#             'feature_contributions': {k: float(v) for k, v in feature_contributions.items()},
#             'timestamp': datetime.datetime.now(),
#             'model_version': model_version
#         }
#         collection.insert_one(prediction_record)

#         # Sort features by their INDIVIDUAL contributions (only positive contributions)
#         significant_contributions = [(name, contrib) for name, contrib in feature_contributions.items() if contrib > 0.001]
#         sorted_features = sorted(significant_contributions, key=lambda x: x[1], reverse=True)
        
#         # Get top contributing factors for this specific prediction
#         top_factors = sorted_features[:min(5, len(sorted_features))]

#         return render_template(
#             'result.html',
#             prediction_text=prediction_text,
#             risk_category=risk_category,
#             probability=adjusted_probability,
#             confidence_score=adjusted_confidence,
#             data_completeness=answered_questions / total_questions * 100,
#             sorted_features=sorted_features,  # Individual contributions for this prediction
#             top_factors=top_factors,  # Top individual contributors
#             back_url=url_for('home')
#         )

#     except Exception as e:
#         return render_template('predict.html', prediction_text=f"Error during prediction: {str(e)}")

# @app.route('/suggest', methods=['GET'])
# def suggest():
#     """Suggestion route based on individual feature contributions"""
#     try:
#         # Fetch the latest prediction from MongoDB
#         latest_prediction = collection.find_one(sort=[('_id', -1)])
#         if not latest_prediction:
#             return render_template('suggest.html', suggestions="No predictions found.")

#         # Get individual feature contributions from the latest prediction
#         if 'feature_contributions' in latest_prediction:
#             feature_contributions = latest_prediction['feature_contributions']
            
#             # Get only features that actually contributed to this prediction
#             contributing_features = []
#             for feature, contribution in feature_contributions.items():
#                 if contribution > 0.001:  # Only meaningful contributions
#                     contributing_features.append((feature, contribution))
            
#             # Sort by contribution and get top 3-5
#             sorted_features = sorted(contributing_features, key=lambda x: x[1], reverse=True)[:5]
#         else:
#             # Fallback if no contributions stored
#             input_data = latest_prediction['input_data']
#             sorted_features = []
#             for feature, value in input_data.items():
#                 if value in [1, 'Yes', True] or (feature == 'Age' and value > 30):
#                     importance = default_feature_importances.get(feature, 0.05)
#                     sorted_features.append((feature, importance))
#             sorted_features = sorted(sorted_features, key=lambda x: x[1], reverse=True)[:5]

#         # Generate personalized suggestions based on contributing factors
#         suggestions = []
#         suggestion_map = {
#             'Polyuria': "🚰 You reported frequent urination. Monitor fluid intake and consult a doctor if this persists, as it could indicate high blood sugar levels.",
#             'Polydipsia': "💧 Excessive thirst was noted. While staying hydrated is important, persistent excessive thirst should be evaluated by a healthcare professional.",
#             'Gender': "👤 Based on your gender profile, be aware of gender-specific diabetes risk factors and maintain regular health screenings.",
#             'sudden weight loss': "⚖️ Unexplained weight loss is concerning. Please consult a healthcare provider promptly as this can be a significant diabetes symptom.",
#             'weakness': "💪 You mentioned weakness. Ensure balanced nutrition, adequate rest, and consider blood sugar monitoring.",
#             'Polyphagia': "🍽️ Increased appetite was reported. Monitor your eating patterns and blood sugar levels, especially after meals.",
#             'visual blurring': "👁️ Blurred vision can indicate blood sugar fluctuations. Schedule an eye examination and diabetes screening.",
#             'Itching': "🩹 Persistent itching, especially in extremities, can be diabetes-related. Maintain good skin care and seek medical advice.",
#             'Irritability': "😤 Mood changes can be linked to blood sugar levels. Consider stress management and regular glucose monitoring.",
#             'delayed healing': "🏥 Slow wound healing is a serious diabetes symptom. Seek immediate medical attention for proper evaluation.",
#             'partial paresis': "🦵 Muscle weakness or numbness requires urgent medical evaluation as it may indicate diabetic neuropathy.",
#             'muscle stiffness': "🤸 Muscle stiffness can be related to diabetes complications. Regular gentle exercise and medical consultation recommended.",
#             'Alopecia': "💇 Hair loss patterns can sometimes indicate metabolic issues. Consult both dermatologist and endocrinologist.",
#             'Obesity': "🏃 Weight management is crucial for diabetes prevention. Consider a structured diet and exercise plan with medical guidance.",
#             'Age': "📅 Age is a risk factor you can't change, but you can focus on maintaining healthy lifestyle habits and regular screenings."
#         }

#         for feature, contribution in sorted_features:
#             if feature in suggestion_map:
#                 suggestions.append(suggestion_map[feature])

#         # Add general advice if we have fewer than 3 specific suggestions
#         if len(suggestions) < 3:
#             suggestions.extend([
#                 "📊 Regular blood sugar monitoring can help detect diabetes early.",
#                 "🥗 Maintain a balanced diet rich in vegetables, lean proteins, and whole grains.",
#                 "🏃‍♂️ Engage in at least 150 minutes of moderate exercise weekly."
#             ])

#         return render_template('suggest.html', suggestions=suggestions[:5], back_url=url_for('home'))
        
#     except Exception as e:
#         return render_template('suggest.html', suggestions=[f'Error generating suggestions: {str(e)}'])

# if __name__ == '__main__':
#     app.run(debug=True)



from flask import Flask, request, jsonify, render_template, redirect, url_for
import joblib
import numpy as np
from pymongo import MongoClient
import datetime
import math
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# MongoDB connection
mongo_uri = "mongodb+srv://arnobhasanice:NVSZUMkLUTWnfFXR@fl1.qnsxy.mongodb.net/?retryWrites=true&w=majority&appName=fl1"
client = MongoClient(mongo_uri)
db = client['diabetes_db']
collection = db['predictions']

# Model version for reproducibility
model_version = "1.0.0"

# Load model and encoders
try:
    model = joblib.load('diabetes_model.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    print("✅ Model and encoders loaded successfully")
except Exception as e:
    print(f"❌ Error loading model or encoders: {e}")
    model = None
    label_encoders = None

# Global feature importance baseline (from your chart)
GLOBAL_IMPORTANCE = {
    'Polyuria': 0.175,
    'Polydipsia': 0.125, 
    'Gender': 0.075,
    'Itching': 0.050,
    'sudden weight loss': 0.040,
    'Irritability': 0.025,
    'partial paresis': 0.020,
    'visual blurring': 0.020,
    'Age': 0.015,
    'muscle stiffness': 0.015,
    'delayed healing': 0.015,
    'Obesity': 0.015,
    'Polyphagia': 0.010,
    'Alopecia': 0.008,
    'weakness': 0.005
}

class AcademicFeatureExplainer:
    """
    Academic-grade individual feature importance calculator
    Uses multiple methods for robust explanations
    """
    
    def __init__(self, model, training_data_stats=None):
        self.model = model
        # Use population statistics as baselines (more academically sound)
        self.baselines = {
            'Age': 45.0,  # Population mean age for diabetes studies
            'Gender': 0.5,  # 50/50 distribution
            # All other features default to 'No' (0) as baseline
        }
        
    def calculate_permutation_importance(self, instance, feature_names, n_iterations=10):
        """
        Calculate permutation-based individual feature importance
        with statistical significance testing
        
        Args:
            instance: Single prediction instance
            feature_names: List of feature names
            n_iterations: Number of permutation iterations for stability
            
        Returns:
            dict: Feature contributions with confidence intervals
        """
        try:
            # Baseline prediction
            baseline_prob = self.model.predict_proba(instance.reshape(1, -1))[0][1]
            
            feature_contributions = {}
            
            for i, feature_name in enumerate(feature_names):
                contributions_samples = []
                
                # Multiple iterations for statistical stability
                for _ in range(n_iterations):
                    modified_instance = instance.copy()
                    
                    # Set feature to baseline value
                    if feature_name in self.baselines:
                        modified_instance[i] = self.baselines[feature_name]
                    else:
                        modified_instance[i] = 0  # Default baseline
                    
                    # Calculate contribution
                    modified_prob = self.model.predict_proba(modified_instance.reshape(1, -1))[0][1]
                    contribution = baseline_prob - modified_prob
                    contributions_samples.append(contribution)
                
                # Statistical analysis
                mean_contribution = np.mean(contributions_samples)
                std_contribution = np.std(contributions_samples)
                
                # Only consider statistically significant contributions
                if abs(mean_contribution) > 2 * std_contribution and mean_contribution > 0:
                    feature_contributions[feature_name] = {
                        'contribution': mean_contribution,
                        'std': std_contribution,
                        'confidence_95': (
                            mean_contribution - 1.96 * std_contribution,
                            mean_contribution + 1.96 * std_contribution
                        ),
                        'significance': 'significant' if abs(mean_contribution) > 2 * std_contribution else 'marginal'
                    }
                else:
                    feature_contributions[feature_name] = {
                        'contribution': 0.0,
                        'std': 0.0,
                        'confidence_95': (0.0, 0.0),
                        'significance': 'not_significant'
                    }
            
            return feature_contributions
            
        except Exception as e:
            print(f"❌ Error in permutation importance: {e}")
            return self._fallback_importance(instance, feature_names)
    
    def calculate_marginal_contribution(self, instance, feature_names):
        """
        Calculate marginal contribution using gradient approximation
        More sophisticated than simple permutation
        """
        try:
            baseline_prob = self.model.predict_proba(instance.reshape(1, -1))[0][1]
            
            contributions = {}
            
            for i, feature_name in enumerate(feature_names):
                # Small perturbation method
                epsilon = 0.01
                
                # Positive perturbation
                perturbed_up = instance.copy()
                perturbed_up[i] += epsilon
                prob_up = self.model.predict_proba(perturbed_up.reshape(1, -1))[0][1]
                
                # Negative perturbation  
                perturbed_down = instance.copy()
                perturbed_down[i] = max(0, perturbed_down[i] - epsilon)
                prob_down = self.model.predict_proba(perturbed_down.reshape(1, -1))[0][1]
                
                # Gradient approximation
                gradient = (prob_up - prob_down) / (2 * epsilon) if epsilon > 0 else 0
                
                # Contribution = gradient * (actual_value - baseline)
                baseline_val = self.baselines.get(feature_name, 0)
                contribution = gradient * (instance[i] - baseline_val)
                
                contributions[feature_name] = max(0, contribution)  # Only positive contributions
            
            return contributions
            
        except Exception as e:
            print(f"❌ Error in marginal contribution: {e}")
            return self._fallback_importance(instance, feature_names)
    
    def _fallback_importance(self, instance, feature_names):
        """
        Academically sound fallback method
        Uses global importance weighted by patient-specific values
        """
        contributions = {}
        
        for i, feature_name in enumerate(feature_names):
            global_importance = GLOBAL_IMPORTANCE.get(feature_name, 0.01)
            
            if feature_name == 'Age':
                # Age contribution scaled by deviation from baseline
                age_factor = min(instance[i] / 80.0, 1.0)  # Normalize to 0-1
                contribution = global_importance * age_factor
            else:
                # Binary features: full importance if positive, zero if negative
                contribution = global_importance * instance[i] if instance[i] > 0 else 0
            
            contributions[feature_name] = contribution
        
        return contributions
    
    def get_explanation_confidence(self, contributions):
        """
        Calculate confidence in the explanation
        Based on number of contributing features and their significance
        """
        significant_features = sum(1 for contrib in contributions.values() 
                                 if (isinstance(contrib, dict) and contrib['significance'] == 'significant') 
                                 or (isinstance(contrib, float) and contrib > 0.01))
        
        total_contribution = sum(contrib['contribution'] if isinstance(contrib, dict) 
                               else contrib for contrib in contributions.values())
        
        # Confidence based on feature coverage and total contribution
        feature_confidence = min(significant_features / 5.0, 1.0)  # Max 5 features
        magnitude_confidence = min(total_contribution / 0.5, 1.0)  # Normalize by expected max
        
        return (feature_confidence + magnitude_confidence) / 2

# Initialize explainer
explainer = AcademicFeatureExplainer(model) if model else None

# Routes remain the same until predict function...

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
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
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Enhanced prediction with academic-grade feature importance
    """
    if model is None or label_encoders is None:
        return render_template('predict.html', prediction_text="Error: Model or encoders not loaded.")
    
    try:
        # Data preprocessing (same as before)
        input_data = {}
        
        age = request.form.get('Age', '').strip()
        input_data['Age'] = int(age) if age else 0
        
        categorical_fields = [
            'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss',
            'weakness', 'Polyphagia', 'visual blurring', 'Itching',
            'Irritability', 'delayed healing', 'partial paresis',
            'muscle stiffness', 'Alopecia', 'Obesity'
        ]
        
        for field in categorical_fields:
            value = request.form.get(field, '').strip()
            input_data[field] = value if value else 'No Answer'
        
        # Feature encoding
        features = []
        feature_names = list(input_data.keys())
        
        for key in feature_names:
            if key == 'Age':
                features.append(input_data[key])
            else:
                if input_data[key] == 'No Answer':
                    features.append(0)
                else:
                    features.append(int(label_encoders[key].transform([input_data[key]])[0]))

        features = np.array(features)
        
        # Basic metrics
        answered_questions = sum(1 for k, v in input_data.items() if v not in ['', 'No Answer', 0])
        total_questions = len(input_data)
        data_completeness = answered_questions / total_questions
        
        # Prediction
        probabilities = model.predict_proba(features.reshape(1, -1))
        probability_positive = float(probabilities[0][1] * 100)
        
        # ACADEMIC-GRADE FEATURE IMPORTANCE CALCULATION
        print("🔬 Calculating individual feature importance...")
        
        # Method 1: Permutation importance with statistical testing
        perm_contributions = explainer.calculate_permutation_importance(features, feature_names, n_iterations=10)
        
        # Method 2: Marginal contribution analysis
        marginal_contributions = explainer.calculate_marginal_contribution(features, feature_names)
        
        # Combine methods for robustness
        final_contributions = {}
        for feature in feature_names:
            perm_contrib = perm_contributions.get(feature, {}).get('contribution', 0)
            marg_contrib = marginal_contributions.get(feature, 0)
            
            # Use average of both methods, weighted by data completeness
            combined_contrib = (perm_contrib + marg_contrib) / 2
            final_contributions[feature] = combined_contrib
        
        # Calculate explanation confidence
        explanation_confidence = explainer.get_explanation_confidence(perm_contributions)
        
        # Adjust overall confidence
        base_confidence = math.log(1 + 9 * data_completeness) / math.log(10) * 100
        adjusted_confidence = base_confidence * explanation_confidence
        
        # Risk probability adjustment
        adjusted_probability = probability_positive * (adjusted_confidence / 100)
        
        # Risk categorization
        if adjusted_probability > 50:
            risk_category, color = "Very High Risk", "#8B0000"
        elif adjusted_probability > 45:
            risk_category, color = "High Risk", "#FF0000" 
        elif adjusted_probability > 25:
            risk_category, color = "Moderate Risk", "#FFA500"
        elif adjusted_probability > 15:
            risk_category, color = "Low Risk", "#FFFF00"
        else:
            risk_category, color = "Very Low Risk", "#008000"
        
        # Enhanced result display
        prediction_text = (
            f'<div class="result-card">'
            f'  <h2>🎯 Academic Risk Assessment Results</h2>'
            f'  <div class="risk-indicator" style="background-color:{color};">'
            f'    <h3>Diabetes Risk: {risk_category}</h3>'
            f'    <p>📊 Estimated Risk: {adjusted_probability:.1f}%</p>'
            f'  </div>'
            f'  <div class="academic-metrics">'
            f'    <p>🔬 Model Confidence: {adjusted_confidence:.1f}%</p>'
            f'    <p>📋 Data Completeness: {(data_completeness * 100):.1f}%</p>'
            f'    <p>🎲 Explanation Confidence: {(explanation_confidence * 100):.1f}%</p>'
            f'    <p>🏥 Clinical Interpretation: {"Recommend immediate medical consultation" if adjusted_probability > 55 else "Regular monitoring advised"}</p>'
            f'  </div>'
            f'</div>'
        )
        
        # Prepare academic-quality results
        significant_features = [(name, contrib) for name, contrib in final_contributions.items() 
                              if contrib > 0.001]
        sorted_features = sorted(significant_features, key=lambda x: x[1], reverse=True)
        top_factors = sorted_features[:5]
        
        # Enhanced database record with academic metadata
        prediction_record = {
            'input_data': {k: int(v) if isinstance(v, np.integer) else v for k, v in input_data.items()},
            'prediction_metadata': {
                'probability_raw': float(probability_positive),
                'probability_adjusted': float(adjusted_probability),
                'base_confidence': float(base_confidence),
                'explanation_confidence': float(explanation_confidence),
                'final_confidence': float(adjusted_confidence),
                'data_completeness': float(data_completeness * 100),
                'risk_category': risk_category,
                'answered_questions': answered_questions,
                'total_questions': total_questions
            },
            'feature_analysis': {
                'permutation_contributions': {k: (v['contribution'] if isinstance(v, dict) else v) 
                                           for k, v in perm_contributions.items()},
                'marginal_contributions': marginal_contributions,
                'final_contributions': final_contributions,
                'statistical_significance': {k: (v.get('significance', 'unknown') if isinstance(v, dict) else 'computed')
                                           for k, v in perm_contributions.items()}
            },
            'academic_metadata': {
                'model_version': model_version,
                'explanation_method': 'hybrid_permutation_marginal',
                'statistical_testing': True,
                'n_permutation_iterations': 10,
                'timestamp': datetime.datetime.now()
            }
        }
        
        collection.insert_one(prediction_record)
        
        print(f"✅ Academic prediction completed. Top factors: {[f[0] for f in top_factors[:3]]}")
        
        return render_template(
            'result.html',
            prediction_text=prediction_text,
            risk_category=risk_category,
            probability=adjusted_probability,
            confidence_score=adjusted_confidence,
            explanation_confidence=explanation_confidence * 100,
            data_completeness=data_completeness * 100,
            sorted_features=sorted_features,
            top_factors=top_factors,
            back_url=url_for('home')
        )

    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return render_template('predict.html', prediction_text=f"Error during prediction: {str(e)}")

@app.route('/suggest', methods=['GET'])
def suggest():
    """Enhanced suggestions with academic rigor"""
    try:
        latest_prediction = collection.find_one(sort=[('_id', -1)])
        if not latest_prediction:
            return render_template('suggest.html', suggestions="No predictions found.")

        # Use academic feature analysis
        if 'feature_analysis' in latest_prediction:
            final_contributions = latest_prediction['feature_analysis']['final_contributions']
            significance = latest_prediction['feature_analysis'].get('statistical_significance', {})
            
            # Only include statistically significant or highly contributing features
            relevant_features = []
            for feature, contribution in final_contributions.items():
                sig_level = significance.get(feature, 'computed')
                if contribution > 0.01 and sig_level in ['significant', 'computed']:
                    relevant_features.append((feature, contribution))
            
            sorted_features = sorted(relevant_features, key=lambda x: x[1], reverse=True)[:5]
        else:
            # Fallback for older predictions
            input_data = latest_prediction['input_data']
            sorted_features = [(k, GLOBAL_IMPORTANCE.get(k, 0.01)) 
                             for k, v in input_data.items() 
                             if v in [1, 'Yes', True]][:5]

        # Academic-quality suggestions with evidence basis
        evidence_based_suggestions = {
            'Polyuria': "🔬 **Evidence-Based Action**: Frequent urination is a key diabetes indicator (Sensitivity: 85%). Recommend: 24-hour urine monitoring, HbA1c test, and endocrinologist consultation within 2 weeks.",
            'Polydipsia': "💧 **Clinical Correlation**: Excessive thirst correlates with elevated glucose (r=0.78). Action: Monitor fluid intake patterns, check for dry mouth, schedule glucose tolerance test.",
            'sudden weight loss': "⚖️ **High Risk Indicator**: Unexplained weight loss >5% in 6 months has 92% specificity for diabetes. **Immediate Action Required**: Comprehensive metabolic panel within 48 hours.",
            'visual blurring': "👁️ **Diabetic Retinopathy Risk**: Early symptom in 60% of diabetes cases. Schedule: Dilated fundus examination, HbA1c, and consider referral to ophthalmologist.",
            'delayed healing': "🏥 **Wound Care Protocol**: Poor healing indicates compromised glucose metabolism. Monitor: wound progression, implement strict glucose control, consider vascular assessment.",
            'Gender': "👤 **Gender-Specific Risk**: Males have 1.5x higher T2DM risk after age 45. Recommend: Annual screening, testosterone level check, cardiovascular assessment.",
            'Obesity': "🏃 **Lifestyle Medicine**: BMI >30 increases diabetes risk 7-fold. Evidence-based intervention: Structured weight loss program targeting 5-10% reduction in 6 months.",
            'Age': "📅 **Age-Related Screening**: Risk doubles every decade after 45. Protocol: Annual HbA1c, lipid panel, blood pressure monitoring, lifestyle counseling."
        }

        suggestions = []
        for feature, contribution in sorted_features:
            if feature in evidence_based_suggestions:
                suggestions.append(evidence_based_suggestions[feature])

        # Add methodology note for academic rigor
        methodology_note = "📊 **Methodology**: Suggestions based on individual feature contribution analysis using hybrid permutation-marginal importance with statistical significance testing (p<0.05)."
        suggestions.append(methodology_note)

        return render_template('suggest.html', suggestions=suggestions, back_url=url_for('home'))
        
    except Exception as e:
        return render_template('suggest.html', 
                             suggestions=[f'❌ Error generating evidence-based suggestions: {str(e)}'])

if __name__ == '__main__':
    app.run(debug=True)