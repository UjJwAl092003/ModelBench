"""
ModelBench ML Core - Online Shoppers Purchasing Intention (Classification)
Author: Ujjwal Singh & Aryan
Course: CS801 Computing Lab, M.Tech, NITK Surathkal

Classification algorithms benchmarked on 12,330 e-commerce sessions:
1. Logistic Regression (Baseline)
2. Ridge Logistic Regression (L2 Penalty, C=0.1)
3. Lasso Logistic Regression (L1 Penalty, C=0.5)
4. K-Nearest Neighbors (k=7)
5. Gaussian Naive Bayes (Probabilistic)
6. Support Vector Classifier (Linear SVC)
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


NUMERIC_FEATURES = [
    'Administrative', 'Administrative_Duration',
    'Informational', 'Informational_Duration',
    'ProductRelated', 'ProductRelated_Duration',
    'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay'
]

CATEGORICAL_FEATURES = [
    'Month', 'OperatingSystems', 'Browser',
    'Region', 'TrafficType', 'VisitorType', 'Weekend'
]


def load_dataset(filepath="datasets/online_shoppers.csv"):
    """Loads Online Shoppers dataset and formats target as binary integer."""
    if not os.path.exists(filepath):
        if os.path.exists("../" + filepath):
            filepath = "../" + filepath
        else:
            raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    df = pd.read_csv(filepath)
    df['Revenue'] = df['Revenue'].astype(int)
    df['Weekend'] = df['Weekend'].astype(int)
    return df


def split_data(df, target_col='Revenue', test_size=0.2, random_state=42):
    """Stratified 80/20 train/test split to preserve class balance."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def build_preprocessor():
    """Builds a leak-free ColumnTransformer producing dense arrays for all classifiers."""
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), CATEGORICAL_FEATURES)
        ]
    )


def get_models():
    """Returns classical non-tree classification models."""
    return {
        'logistic': ('Logistic Regression (Baseline)', LogisticRegression(max_iter=1000, random_state=42)),
        'ridge_log': ('Ridge Logistic (L2, C=0.1)', LogisticRegression(C=0.1, max_iter=1000, random_state=42)),
        'lasso_log': ('Lasso Logistic (L1, C=0.5)', LogisticRegression(penalty='l1', solver='liblinear', C=0.5, max_iter=1000, random_state=42)),
        'knn': ('K-Nearest Neighbors (k=7)', KNeighborsClassifier(n_neighbors=7, weights='distance')),
        'naive_bayes': ('Gaussian Naive Bayes', GaussianNB()),
        'svc': ('Support Vector Machine (Linear SVC)', CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42, max_iter=3000)))
    }


def train_and_benchmark(X_train, y_train, X_test, y_test):
    """Fits all classification pipelines and returns performance scorecard."""
    preprocessor = build_preprocessor()
    models = get_models()
    trained_pipelines = {}
    metrics_summary = []

    for model_id, (model_name, classifier) in models.items():
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])
        
        pipe.fit(X_train, y_train)
        trained_pipelines[model_id] = pipe

        test_preds = pipe.predict(X_test)
        
        # Probabilities for ROC-AUC
        if hasattr(pipe, "predict_proba"):
            probs = pipe.predict_proba(X_test)[:, 1]
            roc_auc = round(float(roc_auc_score(y_test, probs)), 4)
        else:
            roc_auc = 0.0

        acc = round(float(accuracy_score(y_test, test_preds) * 100), 2)
        prec = round(float(precision_score(y_test, test_preds, zero_division=0) * 100), 2)
        rec = round(float(recall_score(y_test, test_preds, zero_division=0) * 100), 2)
        f1 = round(float(f1_score(y_test, test_preds, zero_division=0) * 100), 2)

        metrics_summary.append({
            'model_id': model_id,
            'Model': model_name,
            'Accuracy (%)': acc,
            'Precision (%)': prec,
            'Recall (%)': rec,
            'F1-Score (%)': f1,
            'ROC-AUC': roc_auc
        })

    # Sort descending by F1-Score
    metrics_summary.sort(key=lambda x: x['F1-Score (%)'], reverse=True)
    return trained_pipelines, metrics_summary


def predict_sample(pipeline, feature_dict):
    """Runs single-session inference for online shoppers."""
    sample_df = pd.DataFrame([feature_dict])
    pred = pipeline.predict(sample_df)[0]
    prob = 0.5
    if hasattr(pipeline, "predict_proba"):
        prob = round(float(pipeline.predict_proba(sample_df)[0][1]), 4)
    
    return {
        'prediction': int(pred),
        'label': 'Purchase Completed (Revenue = True)' if pred == 1 else 'Browsing Only (Revenue = False)',
        'purchase_probability': prob,
        'probability_percent': f"{prob * 100:.1f}%"
    }


def get_confusion_matrix(pipeline, X_test, y_test):
    """Computes confusion matrix values for client-side heatmap rendering."""
    test_preds = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, test_preds)
    return {
        'tn': int(cm[0, 0]),
        'fp': int(cm[0, 1]),
        'fn': int(cm[1, 0]),
        'tp': int(cm[1, 1])
    }
