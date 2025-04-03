
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

data = pd.read_csv('diabetess.csv') # Load the dataset from a CSV file

# Step 2: Preprocess the dataset

        
label_encoder = LabelEncoder()
data_encoded = data.copy()

for col in data_encoded.columns:
    if data_encoded[col].dtype == 'object':
        data_encoded[col] = label_encoder.fit_transform(data_encoded[col])       
         

target_column = 'class'

X = data_encoded.drop(columns=[target_column])
y = data_encoded[target_column] 


X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.2, random_state = 42 
    
)
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
     
    "AdaBoost" : AdaBoostClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Ridge Classifier": RidgeClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "MLP Classifier": MLPClassifier(max_iter=1000, random_state=42),
    "LDA": LinearDiscriminantAnalysis() 
    
}
    
results = {}
for name , model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, output_dict = True)
    accuracy = accuracy_score(y_test, predictions)
    
    results[name] = {
        "accuracy": accuracy,
        "precision": report['weighted avg']['precision'],
        "recall": report['weighted avg']['recall'],
        "f1-score": report['weighted avg']['f1-score']
    }  

results_df = pd.DataFrame(results).T 
results_df = results_df.sort_values(by="accuracy", ascending=False)

print("Model Performance Comparison:")
print(results_df) 


 # Plot results with better visualization
plt.figure(figsize=(12, 6))  # Adjust the figure size for better readability
plt.bar(results_df.index, results_df['accuracy'], color='red')  # Bar plot

# Set title and labels
plt.title('Model Comparison', fontsize=16)  # Larger font size for the title
plt.ylabel('Accuracy', fontsize=14)  # Larger font size for the y-axis label
plt.xlabel('Models', fontsize=14)  # Larger font size for the x-axis label

# Adjust x-axis labels
plt.xticks(rotation=30, ha='right', fontsize=12)  # Rotate labels and align to the right

# Display the plot
plt.tight_layout()  # Adjust layout to avoid clipping
plt.show()
