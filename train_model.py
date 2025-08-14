import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np

# -------------------------------
# Load dataset
# -------------------------------
data = pd.read_csv('diabetess.csv')
data = data.drop_duplicates()

# -------------------------------
# Encode categorical features
# -------------------------------
label_encoders = {}
for column in data.columns:
    if data[column].dtype == 'object':
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

# -------------------------------
# Features & Target
# -------------------------------
X = data.drop('class', axis=1)
y = data['class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Base learners & meta model
# -------------------------------
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
]
meta_model = LogisticRegression(max_iter=1000, random_state=42)

# -------------------------------
# Stacking model
# -------------------------------
stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    stack_method='predict_proba'  # Important for SHAP compatibility
)

# Train model
stacking_model.fit(X_train, y_train)

# -------------------------------
# Evaluation
# -------------------------------
y_pred = stacking_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# Save all files for Flask + SHAP
# -------------------------------
joblib.dump(stacking_model, "stacking_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(X_train.values, "X_train.pkl")  # Save as numpy array
joblib.dump(list(X.columns), "feature_names.pkl")

print("\n✅ Files saved successfully")