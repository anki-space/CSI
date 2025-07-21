import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer 
import joblib 
def train_and_save_model():
    print("Loading Breast Cancer dataset...")
    breast_cancer = load_breast_cancer(as_frame=True) 
    df = breast_cancer.frame
    X = df[breast_cancer.feature_names] 
    y = df['target'] # Target is 0 for malignant, 1 for benign

    # Split data into training and testing sets and ensure stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y  
    )

    print("Training Logistic Regression model...")
    # Using a solver that handles multi-class classification well (even for binary)
    model = LogisticRegression(max_iter=1000, solver='liblinear') 
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Training Complete! Accuracy on test set: {accuracy:.2f}")

    # Save the trained model to a file
    model_filename = 'logistic_regression_breast_cancer_model.pkl'
    joblib.dump(model, model_filename)
    print(f"Model saved successfully as '{model_filename}'")

    # Save the full dataset for visualization purposes in the app
    df_filename = 'breast_cancer_data.pkl'
    joblib.dump(df, df_filename)
    print(f"Dataset saved successfully as '{df_filename}'")

    # Save feature names and target names for use in the app
    feature_names_filename = 'breast_cancer_feature_names.pkl'
    joblib.dump(breast_cancer.feature_names, feature_names_filename)
    print(f"Feature names saved successfully as '{feature_names_filename}'")

    target_names_filename = 'breast_cancer_target_names.pkl'
    joblib.dump(breast_cancer.target_names, target_names_filename)
    print(f"Target names saved successfully as '{target_names_filename}'")


if __name__ == "__main__":
    train_and_save_model()
