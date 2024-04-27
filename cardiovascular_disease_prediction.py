# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
data = pd.read_csv("heart_disease_dataset.csv")

# Data pre-processing
# Checking for missing values
print("Missing values:")
print(data.isnull().sum())

# Exploratory Data Analysis (EDA) and Visualization
# Pairplot to visualize relationships between variables
sns.pairplot(data)
plt.show()

# Correlation matrix
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Splitting the data into features and target variable
X = data.drop(columns=["target"])
y = data["target"]

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training and evaluation
models = {
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier()
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of {name}: {accuracy}")
    print(f"Classification Report of {name}:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix of {name}:\n{confusion_matrix(y_test, y_pred)}")
    print("="*50)

# Building Machine Learning model for heart disease detection
# Random Forest Classifier seems to perform the best
final_model = RandomForestClassifier()
final_model.fit(X_train_scaled, y_train)

# Saving the model
import joblib
joblib.dump(final_model, "heart_disease_detection_model.pkl")
