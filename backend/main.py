"""
ModelBench Backend API
Author: ModelBench Team (Ujjwal Singh & Aryan)
Course: CS801 Computing Lab, M.Tech CSE, NITK Surathkal

A lightweight FastAPI application providing REST endpoints for:
- Dataset and model catalog
- Interactive model inference (What-If prediction)
- Real-time empirical model comparison scorecards
- Diagnostic error and distribution analytics

Directly integrates with the underlying ML core (ml_core/insurance_regression.py).
"""

import sys
import os
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is in Python sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import reusable ML functions directly from ml_core
from ml_core.insurance_regression import (
    load_dataset,
    split_data,
    build_preprocessor,
    build_linear_pipeline,
    build_ridge_pipeline,
    build_poly_ridge_pipeline,
    evaluate_model,
    compare_all_models,
    get_diagnostic_data,
    predict_sample
)

app = FastAPI(
    title="ModelBench API",
    description="Backend API for ModelBench ML Experimentation and Benchmarking Platform",
    version="1.0.0"
)

# Enable CORS so the React frontend can communicate seamlessly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================================
# GLOBAL ML CACHE (Trained once at server startup)
# =========================================================================
ML_STATE = {}

def initialize_ml_engine():
    """Initializes and caches the dataset, pipelines, and evaluation metrics."""
    try:
        df = load_dataset()
        X_train, X_test, y_train, y_test = split_data(df, target_col="charges", test_size=0.2, random_state=42)
        preprocessor = build_preprocessor()
        
        # Build and train the three supported pipelines
        linear_pipe = build_linear_pipeline(preprocessor)
        linear_pipe.fit(X_train, y_train)
        
        ridge_pipe = build_ridge_pipeline(preprocessor, alpha=1.0)
        ridge_pipe.fit(X_train, y_train)
        
        poly_ridge_pipe = build_poly_ridge_pipeline(preprocessor, degree=2, alpha=10.0)
        poly_ridge_pipe.fit(X_train, y_train)
        
        # Compute benchmark comparison
        comparison_df, raw_results = compare_all_models(X_train, y_train, X_test, y_test, preprocessor)
        
        ML_STATE["df"] = df
        ML_STATE["X_train"] = X_train
        ML_STATE["X_test"] = X_test
        ML_STATE["y_train"] = y_train
        ML_STATE["y_test"] = y_test
        ML_STATE["pipelines"] = {
            "linear": linear_pipe,
            "ridge": ridge_pipe,
            "poly_ridge": poly_ridge_pipe
        }
        ML_STATE["comparison_summary"] = comparison_df.to_dict(orient="records")
        ML_STATE["raw_results"] = raw_results
        
        print("ModelBench ML Engine initialized and pipelines cached successfully.")
    except Exception as e:
        print(f"Error initializing ML Engine: {e}")

# Run initialization at startup
initialize_ml_engine()


# =========================================================================
# PYDANTIC SCHEMAS FOR REQUEST / RESPONSE VALIDATION
# =========================================================================
class InsuranceFeatures(BaseModel):
    age: float = Field(..., ge=18, le=100, description="Age in years (18-100)")
    sex: str = Field(..., description="Gender ('male' or 'female')")
    bmi: float = Field(..., ge=10.0, le=70.0, description="Body Mass Index (10.0 - 70.0)")
    children: int = Field(..., ge=0, le=10, description="Number of children/dependents (0-10)")
    smoker: str = Field(..., description="Smoking status ('yes' or 'no')")
    region: str = Field(..., description="Geographical region ('southeast', 'southwest', 'northeast', 'northwest')")


class PredictionRequest(BaseModel):
    dataset_id: str = "insurance"
    model_id: str = "poly_ridge"
    features: InsuranceFeatures


# =========================================================================
# API ENDPOINTS
# =========================================================================

@app.get("/api/health")
def health_check():
    """Returns the API health status and loaded dataset status."""
    return {
        "status": "healthy",
        "engine": "ModelBench ML Core",
        "cached_models": list(ML_STATE.get("pipelines", {}).keys())
    }


@app.get("/api/datasets")
def get_available_datasets():
    """
    Returns the catalog of available benchmark datasets in ModelBench.
    Designed for modular extensibility as new datasets are added.
    """
    return [
        {
            "id": "insurance",
            "name": "Medical Insurance Cost Prediction",
            "domain": "Healthcare Economics",
            "task_type": "Regression",
            "target_variable": "charges ($)",
            "sample_count": ML_STATE.get("df", pd.DataFrame()).shape[0],
            "feature_count": 6,
            "description": "Predict individual medical charges based on patient demographic, physiological, and lifestyle risk factors.",
            "features_schema": [
                {"name": "age", "label": "Age", "type": "number", "min": 18, "max": 64, "default": 35, "unit": "years"},
                {"name": "sex", "label": "Sex / Gender", "type": "select", "options": ["male", "female"], "default": "male"},
                {"name": "bmi", "label": "Body Mass Index (BMI)", "type": "number", "min": 15.0, "max": 53.0, "step": 0.1, "default": 28.5, "unit": "kg/m²"},
                {"name": "children", "label": "Children / Dependents", "type": "number", "min": 0, "max": 5, "default": 1, "unit": "count"},
                {"name": "smoker", "label": "Smoking Status", "type": "select", "options": ["no", "yes"], "default": "no"},
                {"name": "region", "label": "Residential Region", "type": "select", "options": ["southeast", "southwest", "northeast", "northwest"], "default": "southeast"}
            ]
        }
    ]


@app.get("/api/datasets/{dataset_id}/models")
def get_dataset_models(dataset_id: str):
    """Returns the list of genuinely implemented and cached models for the selected dataset."""
    if dataset_id != "insurance":
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    
    return [
        {
            "id": "poly_ridge",
            "name": "Polynomial Regression (Degree=2, alpha=10.0)",
            "family": "Non-Linear Interaction + L2 Regularization",
            "test_r2": 0.8656,
            "test_mae": 2836.88,
            "test_rmse": 4568.56,
            "description": "Captures non-linear interaction terms (BMI x Smoker) with L2 weight shrinkage to prevent overfitting.",
            "is_best_model": True
        },
        {
            "id": "linear",
            "name": "Linear Regression (OLS Baseline)",
            "family": "Parametric Linear (Ordinary Least Squares)",
            "test_r2": 0.7836,
            "test_mae": 4181.19,
            "test_rmse": 5796.28,
            "description": "Standard multiple linear regression assuming independent additive feature contributions.",
            "is_best_model": False
        },
        {
            "id": "ridge",
            "name": "Ridge Regression (alpha=1.0)",
            "family": "Linear with L2 Regularization",
            "test_r2": 0.7833,
            "test_mae": 4193.20,
            "test_rmse": 5800.46,
            "description": "Linear model with L2 penalty preventing extreme coefficient values.",
            "is_best_model": False
        }
    ]


@app.post("/api/predict")
def predict_charges(request: PredictionRequest):
    """
    Executes live model inference on human-readable feature inputs.
    Reuses the cached Scikit-Learn pipeline without retraining.
    """
    if request.dataset_id != "insurance":
        raise HTTPException(status_code=404, detail=f"Dataset '{request.dataset_id}' not found.")
    
    pipelines = ML_STATE.get("pipelines", {})
    if request.model_id not in pipelines:
        raise HTTPException(status_code=400, detail=f"Model '{request.model_id}' is not available. Choose from: {list(pipelines.keys())}")
    
    pipeline = pipelines[request.model_id]
    feature_dict = request.features.model_dump()
    
    # Input validation checks
    if feature_dict["sex"].lower() not in ["male", "female"]:
        raise HTTPException(status_code=422, detail="Sex must be 'male' or 'female'.")
    if feature_dict["smoker"].lower() not in ["yes", "no"]:
        raise HTTPException(status_code=422, detail="Smoker must be 'yes' or 'no'.")
    if feature_dict["region"].lower() not in ["southeast", "southwest", "northeast", "northwest"]:
        raise HTTPException(status_code=422, detail="Region must be southeast, southwest, northeast, or northwest.")
        
    try:
        predicted_value = predict_sample(pipeline, feature_dict)
        # Ensure non-negative charges prediction
        predicted_value = max(0.0, predicted_value)
        
        model_names = {
            "linear": "Linear Regression (OLS Baseline)",
            "ridge": "Ridge Regression (alpha=1.0)",
            "poly_ridge": "Polynomial Regression (Degree=2, alpha=10.0)"
        }
        
        return {
            "status": "success",
            "dataset_id": request.dataset_id,
            "dataset_name": "Medical Insurance Cost Prediction",
            "model_id": request.model_id,
            "model_name": model_names.get(request.model_id, request.model_id),
            "predicted_charges": round(predicted_value, 2),
            "formatted_charges": f"${predicted_value:,.2f}",
            "input_features": feature_dict
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/api/comparison/{dataset_id}")
def get_model_comparison(dataset_id: str):
    """Returns the empirical benchmark comparison table computed from the real dataset."""
    if dataset_id != "insurance":
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
        
    return {
        "dataset_id": "insurance",
        "dataset_name": "Medical Insurance Cost Prediction",
        "metrics": ML_STATE.get("comparison_summary", [])
    }


@app.get("/api/diagnostics/{dataset_id}/{model_id}")
def get_model_diagnostics(dataset_id: str, model_id: str):
    """
    Returns actual vs. predicted points and residual distribution data
    for interactive client-side diagnostic plotting.
    """
    if dataset_id != "insurance":
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
        
    pipelines = ML_STATE.get("pipelines", {})
    if model_id not in pipelines:
        raise HTTPException(status_code=400, detail=f"Model '{model_id}' not found.")
        
    pipeline = pipelines[model_id]
    X_test = ML_STATE["X_test"]
    y_test = ML_STATE["y_test"]
    
    # Compute test predictions and residuals
    test_pred = pipeline.predict(X_test)
    residuals = y_test.values - test_pred
    
    # Subsample 100 points for lightweight client rendering
    indices = np.linspace(0, len(y_test) - 1, min(100, len(y_test)), dtype=int)
    
    # Compute histogram bins for residual normality plot
    hist_counts, bin_edges = np.histogram(residuals, bins=25)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    return {
        "model_id": model_id,
        "sample_count": len(indices),
        "actual_vs_predicted": [
            {"actual": round(float(y_test.values[i]), 2), "predicted": round(float(test_pred[i]), 2)}
            for i in indices
        ],
        "residuals_vs_predicted": [
            {"predicted": round(float(test_pred[i]), 2), "residual": round(float(residuals[i]), 2)}
            for i in indices
        ],
        "residual_distribution": [
            {"bin_center": round(float(bc), 2), "count": int(c)}
            for bc, c in zip(bin_centers, hist_counts)
        ]
    }


@app.get("/api/insights/{dataset_id}")
def get_dataset_insights(dataset_id: str):
    """Returns summary distribution and correlation data from the real dataset for UI charts."""
    if dataset_id != "insurance":
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
        
    df = ML_STATE.get("df", pd.DataFrame())
    if df.empty:
        raise HTTPException(status_code=500, detail="Dataset not loaded.")
        
    # 1. Target Distribution Bins
    hist_counts, bin_edges = np.histogram(df["charges"], bins=25)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    charges_distribution = [
        {"charge_range": f"${int(bin_edges[i]):,}-${int(bin_edges[i+1]):,}", "bin_center": round(float(bin_centers[i]), 2), "count": int(hist_counts[i])}
        for i in range(len(hist_counts))
    ]
    
    # 2. Smoker vs Average Charges
    smoker_avg = df.groupby("smoker")["charges"].mean().round(2).to_dict()
    
    # 3. Numerical Correlation Matrix
    corr_df = df.corr(numeric_only=True).round(3)
    corr_matrix = {
        "columns": corr_df.columns.tolist(),
        "matrix": corr_df.values.tolist()
    }
    
    return {
        "dataset_id": "insurance",
        "total_records": len(df),
        "charges_distribution": charges_distribution,
        "smoker_average_charges": [
            {"status": "Non-Smoker", "average_charge": smoker_avg.get("no", 0)},
            {"status": "Smoker", "average_charge": smoker_avg.get("yes", 0)}
        ],
        "correlation": corr_matrix
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
