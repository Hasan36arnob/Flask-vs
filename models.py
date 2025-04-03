import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load the dataset from a CSV file
data = pd.read_csv('diabetess.csv')

# Initialize the LabelEncoder for encoding categorical variables
label_encoder = LabelEncoder()
data_encoded = data.copy()

# Loop through each column and encode if it is categorical (dtype == 'object')
for col in data_encoded.columns:
    if data_encoded[col].dtype == 'object':
        data_encoded[col] = label_encoder.fit_transform(data_encoded[col])

# Define the target column
target_column = 'class'

# Split the dataset into features and target
X = data_encoded.drop(columns=[target_column])
y = data_encoded[target_column]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Random Forest ---
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"Random Forest Accuracy: {rf_accuracy}")

# --- Gradient Boosting ---
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)
gb_predictions = gb_model.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_predictions)
print(f"Gradient Boosting Accuracy: {gb_accuracy}")

# --- AdaBoost ---
ab_model = AdaBoostClassifier(random_state=42)
ab_model.fit(X_train, y_train)
ab_predictions = ab_model.predict(X_test)
ab_accuracy = accuracy_score(y_test, ab_predictions)
print(f"AdaBoost Accuracy: {ab_accuracy}")

# --- Logistic Regression ---
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_predictions)
print(f"Logistic Regression Accuracy: {lr_accuracy}")

# --- Ridge Classifier ---
ridge_model = RidgeClassifier(random_state=42)
ridge_model.fit(X_train, y_train)
ridge_predictions = ridge_model.predict(X_test)
ridge_accuracy = accuracy_score(y_test, ridge_predictions)
print(f"Ridge Classifier Accuracy: {ridge_accuracy}")

# --- Naive Bayes ---
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_predictions)
print(f"Naive Bayes Accuracy: {nb_accuracy}")

# --- MLP Classifier (Neural Network) ---
mlp_model = MLPClassifier(max_iter=1000, random_state=42)
mlp_model.fit(X_train, y_train)
mlp_predictions = mlp_model.predict(X_test)
mlp_accuracy = accuracy_score(y_test, mlp_predictions)
print(f"MLP Classifier Accuracy: {mlp_accuracy}")

# --- LDA ---
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train, y_train)
lda_predictions = lda_model.predict(X_test)
lda_accuracy = accuracy_score(y_test, lda_predictions)
print(f"LDA Accuracy: {lda_accuracy}")