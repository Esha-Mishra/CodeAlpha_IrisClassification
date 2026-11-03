# CodeAlpha Task 1: Iris Flower Classification

## Project Overview

This project is completed as part of the CodeAlpha Data Science Internship.

The objective of this project is to build a Machine Learning classification model that predicts the species of Iris flowers based on their measurements.

The model classifies flowers into three species:

- Iris Setosa
- Iris Versicolor
- Iris Virginica

## Dataset

The project uses the Iris flower dataset (`iris.csv`).

The input features used for classification are:

- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

The target variable is `Species`.

## Technologies and Libraries Used

- Python
- Pandas
- Scikit-learn

## Machine Learning Algorithm

Random Forest Classifier is used to train the classification model.

## Project Workflow

1. Load the Iris dataset using Pandas.
2. Explore the dataset and check for missing values.
3. Select input features and target variable.
4. Split the dataset into training and testing sets (80:20).
5. Train the Random Forest Classifier.
6. Predict Iris flower species using test data.
7. Evaluate model performance using:
   - Accuracy Score
   - Classification Report
   - Confusion Matrix
8. Predict the species of a new Iris flower sample.

## Model Evaluation

The model is evaluated using test data. Accuracy, precision, recall, F1-score and confusion matrix are used to assess its performance.

## Output Files

The project generates an `outputs` folder containing:

- `first_five_rows.csv`
- `missing_values.csv`
- `model_evaluation.txt`
- `predictions.csv`
- `new_flower_prediction.csv`

## How to Run the Project

1. Install Python on your computer.
2. Install the required libraries:

```bash
pip install pandas scikit-learn
```

3. Keep `iris.csv` and `iris_classification.py` in the same project folder.
4. Run the following command:

```bash
python iris_classification.py
```

5. The results will be displayed in the terminal and saved in the `outputs` folder.

## Conclusion

This project demonstrates the application of supervised Machine Learning for multi-class classification using the Random Forest algorithm. It helps understand data preparation, model training, prediction and model evaluation.

## Internship

**CodeAlpha Data Science Internship**

**Project:** Iris Flower Classification
