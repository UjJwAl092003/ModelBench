"""
ModelBench ML Core - Medical Insurance Cost Prediction
Author: Ujjwal Singh (262IS036) & Aryan (262IS007)
Course: CS801 Computing Lab, M.Tech CSE, NITK Surathkal

This module implements clean, modular, and leak-free regression pipelines
(Linear Regression, Ridge Regularization, and Polynomial Features)
for empirical model evaluation and benchmarking.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def load_dataset(filepath=None):
    """
    Loads the Medical Insurance dataset from the datasets directory.
    Fallback logic checks both 'datasets/Insurance/insurance.csv' and 'datasets/insurance.csv'.
    """
    if filepath and os.path.exists(filepath):
        df = pd.read_csv(filepath)
    elif os.path.exists("datasets/Insurance/insurance.csv"):
        df = pd.read_csv("datasets/Insurance/insurance.csv")
    elif os.path.exists("datasets/insurance.csv"):
        df = pd.read_csv("datasets/insurance.csv")
    elif os.path.exists("../datasets/Insurance/insurance.csv"):
        df = pd.read_csv("../datasets/Insurance/insurance.csv")
    else:
        raise FileNotFoundError("Insurance dataset (insurance.csv) not found in expected dataset paths.")
    
    return df


def split_data(df, target_col="charges", test_size=0.2, random_state=42):
    """
    Separates features and target, and performs an 80/20 train/test split.
    Splitting occurs BEFORE any transformation to prevent data leakage.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def build_preprocessor(numeric_features=None, categorical_features=None):
    """
    Constructs a ColumnTransformer that applies:
    - StandardScaler to numeric columns
    - OneHotEncoder(drop='first') to categorical columns (avoids dummy variable trap)
    """
    if numeric_features is None:
        numeric_features = ["age", "bmi", "children"]
    if categorical_features is None:
        categorical_features = ["sex", "smoker", "region"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
        ]
    )
    return preprocessor


def build_linear_pipeline(preprocessor):
    """Constructs a Scikit-Learn Pipeline for Baseline OLS Linear Regression."""
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])


def build_ridge_pipeline(preprocessor, alpha=1.0):
    """Constructs a Scikit-Learn Pipeline for Ridge Regression with L2 penalty alpha."""
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=alpha))
    ])


def build_poly_ridge_pipeline(preprocessor, degree=2, alpha=10.0):
    """Constructs a Scikit-Learn Pipeline for Polynomial Features + Ridge Regularization."""
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("regressor", Ridge(alpha=alpha))
    ])


def evaluate_model(pipeline, X_train, y_train, X_test, y_test, model_name="Model"):
    """
    Fits the pipeline on X_train, y_train and computes standard evaluation metrics:
    - R² Score (variance explained)
    - Root Mean Squared Error (RMSE)
    - Mean Absolute Error (MAE)
    - Mean Squared Error (MSE)
    """
    pipeline.fit(X_train, y_train)
    
    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)
    
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    test_mae = mean_absolute_error(y_test, test_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    
    return {
        "Model": model_name,
        "Train R²": round(float(train_r2), 4),
        "Test R²": round(float(test_r2), 4),
        "Test MAE ($)": round(float(test_mae), 2),
        "Test RMSE ($)": round(float(test_rmse), 2),
        "Test MSE": round(float(test_mse), 2),
        "pipeline": pipeline
    }


def get_diagnostic_data(pipeline, X_test, y_test):
    """
    Returns actual target values, predicted values, and residuals for diagnostic plotting.
    """
    test_pred = pipeline.predict(X_test)
    residuals = y_test.values - test_pred
    
    return {
        "actual": y_test.values.tolist(),
        "predicted": test_pred.tolist(),
        "residuals": residuals.tolist()
    }


def compare_all_models(X_train, y_train, X_test, y_test, preprocessor):
    """
    Trains and benchmarks:
    1. Linear Regression (Baseline OLS)
    2. Ridge Regression across multiple alpha values (0.01, 0.1, 1.0, 10.0, 100.0)
    3. Polynomial Features (Degree 2) + Ridge (alpha=10.0)
    Returns a consolidated pandas DataFrame.
    """
    results = []
    
    # 1. Baseline Linear Regression
    lr_pipe = build_linear_pipeline(preprocessor)
    results.append(evaluate_model(lr_pipe, X_train, y_train, X_test, y_test, "Linear Regression (OLS)"))
    
    # 2. Ridge Regression with varying alpha
    for a in [0.01, 0.1, 1.0, 10.0, 100.0]:
        ridge_pipe = build_ridge_pipeline(preprocessor, alpha=a)
        results.append(evaluate_model(ridge_pipe, X_train, y_train, X_test, y_test, f"Ridge Regression (alpha={a})"))
    
    # 3. Polynomial Features (Degree 2) + Ridge
    poly_pipe = build_poly_ridge_pipeline(preprocessor, degree=2, alpha=10.0)
    results.append(evaluate_model(poly_pipe, X_train, y_train, X_test, y_test, "Polynomial Reg (d=2, alpha=10.0)"))
    
    summary_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "pipeline"} for r in results
    ])
    return summary_df, results


def predict_sample(pipeline, patient_dict):
    """
    Takes a single patient dictionary and returns the predicted insurance cost in dollars.
    Example patient_dict:
    {'age': 35, 'sex': 'male', 'bmi': 28.5, 'children': 2, 'smoker': 'no', 'region': 'southeast'}
    """
    sample_df = pd.DataFrame([patient_dict])
    prediction = pipeline.predict(sample_df)[0]
    return float(prediction)


if __name__ == "__main__":
    print("=" * 70)
    print(" MODELBENCH ML CORE: MEDICAL INSURANCE REGRESSION EXPERIMENTS")
    print("=" * 70)
    
    # Load dataset
    df = load_dataset()
    print(f"\n[1] Dataset Loaded Successfully: {df.shape[0]} rows, {df.shape[1]} columns.")
    
    # Split
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"[2] Data Split: Train set = {X_train.shape[0]} samples, Test set = {X_test.shape[0]} samples.")
    
    # Preprocessor
    preprocessor = build_preprocessor()
    
    # Run Benchmark Comparison
    print("\n[3] Running Model Training & Cross-Model Benchmarking...")
    comparison_df, model_results = compare_all_models(X_train, y_train, X_test, y_test, preprocessor)
    print("\n" + comparison_df.to_string(index=False))
    
    # Test Prediction Simulation
    best_model_pipeline = model_results[-1]["pipeline"] # Polynomial Ridge
    sample_patient = {
        "age": 35,
        "sex": "male",
        "bmi": 28.5,
        "children": 2,
        "smoker": "yes",
        "region": "southeast"
    }
    predicted_cost = predict_sample(best_model_pipeline, sample_patient)
    print("\n[4] Sample Interactive 'What-If' Simulation:")
    print(f"    Patient profile: {sample_patient}")
    print(f"    Predicted Medical Insurance Charge: ${predicted_cost:,.2f}")
    print("=" * 70)
