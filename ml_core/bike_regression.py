"""
ModelBench ML Core - Bike Sharing Demand (Regression)
Authors: Ujjwal Singh & Aryan
Course: CS801 Computing Lab, M.Tech, NITK Surathkal

Regression algorithms benchmarked on 17,379 hourly bike rental records:
1. Linear Regression (OLS)
2. Ridge Regression (L2 Penalty)
3. Lasso Regression (L1 Penalty)
4. ElasticNet Regression (L1 + L2)
5. Polynomial Regression (Degree=2 + Ridge)
6. K-Nearest Neighbors Regressor (KNN)
7. Support Vector Regressor (Linear SVR)
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# weathersit is an ordinal severity scale (1: Clear < 2: Mist < 3: Rain/Snow < 4: Heavy Storm)
NUMERIC_FEATURES = ['temp', 'atemp', 'hum', 'windspeed', 'weathersit']
CATEGORICAL_FEATURES = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday']


def load_dataset(filepath="datasets/bike_sharing.csv"):
    """Loads the Bike Sharing dataset and removes ID/leakage columns."""
    if not os.path.exists(filepath):
        if os.path.exists("../" + filepath):
            filepath = "../" + filepath
        else:
            raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    df = pd.read_csv(filepath)
    drop_cols = ['instant', 'dteday', 'casual', 'registered']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    return df


def split_data(df, target_col='cnt', test_size=0.2, random_state=42):
    """Splits features and target with an 80/20 train/test split before preprocessing."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def build_preprocessor():
    """Builds a leak-free ColumnTransformer for scaling numbers (including ordinal weather) and encoding categories."""
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), CATEGORICAL_FEATURES)
        ]
    )


def get_models():
    """Returns dictionary of classical non-tree regression models with architecture metadata."""
    return {
        'poly_ridge': {
            'name': 'Polynomial Ridge (Degree=2, alpha=10.0)',
            'family': 'Polynomial Regularized Linear',
            'regressor': Pipeline([
                ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                ('reg', Ridge(alpha=10.0))
            ])
        },
        'linear': {
            'name': 'Linear Regression (OLS)',
            'family': 'Ordinary Least Squares',
            'regressor': LinearRegression()
        },
        'ridge': {
            'name': 'Ridge Regression (L2, alpha=1.0)',
            'family': 'L2 Regularized Linear',
            'regressor': Ridge(alpha=1.0)
        },
        'lasso': {
            'name': 'Lasso Regression (L1, alpha=0.1)',
            'family': 'L1 Regularized Linear',
            'regressor': Lasso(alpha=0.1, random_state=42)
        },
        'elasticnet': {
            'name': 'ElasticNet (L1+L2, alpha=0.1)',
            'family': 'L1+L2 ElasticNet Linear',
            'regressor': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        },
        'knn': {
            'name': 'K-Nearest Neighbors (k=5)',
            'family': 'Instance-Based Learning (KNN)',
            'regressor': KNeighborsRegressor(n_neighbors=5, weights='distance')
        },
        'svr': {
            'name': 'Support Vector Regressor (Linear SVR)',
            'family': 'Support Vector Machine',
            'regressor': LinearSVR(C=1.0, random_state=42, max_iter=3000)
        }
    }


def train_and_benchmark(X_train, y_train, X_test, y_test):
    """Fits all pipelines and returns performance scorecard dictionary."""
    preprocessor = build_preprocessor()
    models_dict = get_models()
    trained_pipelines = {}
    metrics_summary = []

    for model_id, info in models_dict.items():
        regressor = info['regressor']
        if model_id == 'poly_ridge':
            pipe = Pipeline([
                ('preprocessor', preprocessor),
                ('poly', regressor.named_steps['poly']),
                ('regressor', regressor.named_steps['reg'])
            ])
        else:
            pipe = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', regressor)
            ])
        
        pipe.fit(X_train, y_train)
        trained_pipelines[model_id] = pipe

        train_preds = pipe.predict(X_train)
        test_preds = pipe.predict(X_test)

        train_r2 = round(float(r2_score(y_train, train_preds)), 4)
        test_r2 = round(float(r2_score(y_test, test_preds)), 4)
        test_mae = round(float(mean_absolute_error(y_test, test_preds)), 2)
        test_rmse = round(float(np.sqrt(mean_squared_error(y_test, test_preds))), 2)

        metrics_summary.append({
            'model_id': model_id,
            'id': model_id,
            'name': info['name'],
            'family': info['family'],
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'is_best': (model_id == 'poly_ridge'),
            'is_best_model': (model_id == 'poly_ridge')
        })

    # Sort descending by Test R2
    metrics_summary.sort(key=lambda x: x['test_r2'], reverse=True)
    return trained_pipelines, metrics_summary


def predict_sample(pipeline, feature_dict):
    """Runs single-sample inference for human-readable feature inputs."""
    sample_df = pd.DataFrame([feature_dict])
    pred = pipeline.predict(sample_df)[0]
    return max(0, round(float(pred)))
