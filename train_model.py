import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
 
# Load the dataset
data = pd.read_csv('diabetess.csv')

# Encode categorical features
label_encoders = {}
for column in data.columns:
    if data[column].dtype == 'object':
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

# Split features and target
X = data.drop('class', axis=1)
y = data['class']

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the Random Forest model
model = RandomForestClassifier(random_state=42)  # Initialize the Random Forest Classifier
model.fit(X_train, y_train)  # Train the model

# Evaluate the model
y_pred = model.predict(X_test)  # Predict the target values for the test set
accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
print("Accuracy:", accuracy)  # Print the accuracy
print("Classification Report:\n", classification_report(y_test, y_pred)) # Print the classification report

