import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.utils import resample

# 1. Load and prepare data
print("Loading and preparing data...")
data = pd.read_csv('diabetess.csv')
feature_names = joblib.load('feature_names.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# 2. Clean data - remove asymptomatic positives
print("\nCleaning data...")
symptom_cols = [col for col in data.columns if col not in ['Age', 'Gender', 'class']]
asymptomatic_pos = data[(data[symptom_cols] == 'No').all(axis=1) & (data['class'] == 'Positive')]
data_clean = data.drop(asymptomatic_pos.index)
print(f"Removed {len(asymptomatic_pos)} problematic cases")

# 3. Balance classes
print("\nBalancing classes...")
negatives = data_clean[data_clean['class'] == 'Negative']
positives = data_clean[data_clean['class'] == 'Positive']

negatives_upsampled = resample(negatives,
                              replace=True,
                              n_samples=len(positives),
                              random_state=42)
balanced_data = pd.concat([positives, negatives_upsampled]).sample(frac=1)

# 4. Rebuild model with proper XGBoost config
print("\nRebuilding model...")

def preprocess(df):
    df_processed = df.copy()
    for col in df.columns:
        if col in label_encoders:
            df_processed[col] = label_encoders[col].transform(df[col])
    return df_processed

def constrained_predict(model, X):
    probas = model.predict_proba(X)
    symptom_cols = [col for col in X.columns if col not in ['Age', 'Gender']]
    no_symptoms = (X[symptom_cols] == 0).all(axis=1)
    probas[no_symptoms, 1] = 0  # Force 0% risk if no symptoms
    return (probas[:, 1] > 0.5).astype(int)

X = balanced_data.drop('class', axis=1)
y = balanced_data['class']

X_processed = preprocess(X)
y_encoded = label_encoders['class'].transform(y)

# Updated XGBoost config without deprecated parameters
base_models = [
    ('rf', RandomForestClassifier(class_weight='balanced', random_state=42)),
    ('dt', DecisionTreeClassifier(class_weight='balanced', random_state=42)),
    ('xgb', XGBClassifier(scale_pos_weight=len(y_encoded[y_encoded==0])/len(y_encoded[y_encoded==1]),
                         eval_metric='logloss'))
]

meta_model = LogisticRegression(class_weight='balanced', max_iter=1000)
model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)

model.fit(X_processed, y_encoded)

# 5. Test the fixed model
print("\nTesting model...")
test_cases = [
    [65, 'Female'] + ['No']*13,  # Asymptomatic
    [40, 'Male', 'Yes', 'Yes'] + ['No']*11  # Symptomatic
]

for i, case in enumerate(test_cases):
    case_df = pd.DataFrame([case], columns=feature_names)
    processed_case = preprocess(case_df)
    
    orig_proba = model.predict_proba(processed_case)[0][1]
    constrained_pred = constrained_predict(model, processed_case)[0]
    
    print(f"\nTest Case {i+1}:")
    print("Features:", case[:5], "...")  # Show first 5 features for brevity
    print("Original probability:", f"{orig_proba*100:.1f}%")
    print("Constrained prediction:", "Positive" if constrained_pred else "Negative")

# 6. Save everything
print("\nSaving final model...")
joblib.dump(model, 'diabetes_model_final.pkl')
joblib.dump(constrained_predict, 'predict_function.pkl')
joblib.dump(label_encoders, 'label_encoders_final.pkl')
joblib.dump(feature_names, 'feature_names_final.pkl')

print("\n✅ Final model saved with:")
print("- diabetes_model_final.pkl")
print("- predict_function.pkl")
print("- label_encoders_final.pkl")
print("- feature_names_final.pkl")