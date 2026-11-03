
# CodeAlpha Data Science Internship
# Task 1: Iris Flower Classification

import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Step 1: Load the CSV dataset
df = pd.read_csv("iris.csv")

print("First five rows of the dataset:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

# Save dataset information
df.head().to_csv("outputs/first_five_rows.csv", index=False)
df.isnull().sum().to_csv("outputs/missing_values.csv")

# Step 2: Select input features and target
X = df[
    [
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm"
    ]
]

y = df["Species"]

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Step 4: Create the machine learning model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Step 5: Train the model
model.fit(X_train, y_train)

# Step 6: Predict flower species
y_pred = model.predict(X_test)

# Step 7: Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(report)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Save evaluation results
with open("outputs/model_evaluation.txt", "w") as file:
    file.write("Model Accuracy: " + str(round(accuracy * 100, 2)) + "%\n\n")
    file.write("Classification Report:\n")
    file.write(report)
    file.write("\nConfusion Matrix:\n")
    file.write(str(cm))

# Save actual and predicted results
results = pd.DataFrame({
    "Actual Species": y_test.values,
    "Predicted Species": y_pred
})
results.to_csv("outputs/predictions.csv", index=False)

# Step 8: Predict a new flower
new_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=[
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm"
    ]
)

prediction = model.predict(new_flower)

print("\nPredicted species of the new flower:")
print(prediction[0])

# Save new flower prediction
new_result = new_flower.copy()
new_result["Predicted Species"] = prediction[0]
new_result.to_csv("outputs/new_flower_prediction.csv", index=False)

print("\nAll output files saved successfully in the outputs folder!")
