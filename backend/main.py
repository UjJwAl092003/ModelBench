"""
ModelBench Backend API - FastAPI Application
Authors: Ujjwal Singh & Aryan
Course: CS801 Computing Lab, M.Tech, NITK Surathkal

A robust ML integration backend serving two canonical datasets:
1. Bike Sharing Demand (Regression - 17,379 rows)
2. Online Shoppers Purchasing Intention (Classification - 12,330 rows)
"""

from contextlib import asynccontextmanager
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ml_core import bike_regression, shoppers_classification

# In-memory ML engine state
ML_STATE: Dict[str, Any] = {
    "bike": {},
    "shoppers": {}
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes and caches both ML pipelines in memory at startup."""
    print("Initializing ModelBench ML Engine...")
    
    # 1. Bike Sharing Demand (Regression)
    print("Loading Bike Sharing Demand dataset (17,379 rows)...")
    df_bike = bike_regression.load_dataset()
    X_tr_b, X_te_b, y_tr_b, y_te_b = bike_regression.split_data(df_bike)
    b_pipelines, b_metrics = bike_regression.train_and_benchmark(X_tr_b, y_tr_b, X_te_b, y_te_b)
    ML_STATE["bike"] = {
        "df": df_bike,
        "X_test": X_te_b,
        "y_test": y_te_b,
        "pipelines": b_pipelines,
        "metrics": b_metrics
    }
    print(f"Bike Sharing models ready: {list(b_pipelines.keys())}")

    # 2. Online Shoppers (Classification)
    print("Loading Online Shoppers dataset (12,330 rows)...")
    df_shop = shoppers_classification.load_dataset()
    X_tr_s, X_te_s, y_tr_s, y_te_s = shoppers_classification.split_data(df_shop)
    s_pipelines, s_metrics = shoppers_classification.train_and_benchmark(X_tr_s, y_tr_s, X_te_s, y_te_s)
    ML_STATE["shoppers"] = {
        "df": df_shop,
        "X_test": X_te_s,
        "y_test": y_te_s,
        "pipelines": s_pipelines,
        "metrics": s_metrics
    }
    print(f"Online Shoppers models ready: {list(s_pipelines.keys())}")
    print("ModelBench ML Engine initialized successfully!")

    yield
    print("Shutting down ModelBench ML Engine...")


app = FastAPI(
    title="ModelBench API",
    description="Empirical ML benchmarking backend for regression and classification",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionPayload(BaseModel):
    dataset_id: str = "bike_sharing"
    model_id: str = ""
    features: Dict[str, Any] = {}


# =========================================================================
# API ENDPOINTS
# =========================================================================

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "engine": "ModelBench Dual-Engine",
        "datasets": ["bike_sharing", "online_shoppers"],
        "bike_models": list(ML_STATE["bike"].get("pipelines", {}).keys()),
        "shoppers_models": list(ML_STATE["shoppers"].get("pipelines", {}).keys())
    }


@app.get("/api/datasets")
def get_datasets():
    """Returns catalog of both benchmark datasets with human-readable schema."""
    return [
        {
            "id": "bike_sharing",
            "name": "Bike Sharing Demand",
            "domain": "Smart Cities & Urban Transport",
            "task_type": "Regression",
            "target_variable": "Total Hourly Rentals (cnt)",
            "sample_count": 17379,
            "feature_count": 12,
            "description": "Predict hourly count of bicycle rentals in Washington D.C. based on weather, hour, and season.",
            "default_model": "poly_ridge"
        },
        {
            "id": "online_shoppers",
            "name": "Online Shoppers Purchasing Intention",
            "domain": "E-Commerce & User Intent",
            "task_type": "Classification",
            "target_variable": "Purchase Completed (Revenue: True/False)",
            "sample_count": 12330,
            "feature_count": 17,
            "description": "Classify whether an e-commerce visitor will make a purchase based on browsing session metrics.",
            "default_model": "ridge_log"
        }
    ]


@app.get("/api/datasets/{dataset_id}/models")
def get_dataset_models(dataset_id: str):
    """Returns the benchmarked models and performance metrics for the chosen dataset."""
    if dataset_id == "bike_sharing":
        return ML_STATE["bike"].get("metrics", [])
    elif dataset_id == "online_shoppers":
        return ML_STATE["shoppers"].get("metrics", [])
    raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")


@app.post("/api/predict")
def predict(payload: PredictionPayload):
    """Executes live model inference on submitted human-readable features."""
    ds_id = payload.dataset_id if payload.dataset_id in ["bike_sharing", "online_shoppers"] else "bike_sharing"
    m_id = payload.model_id
    features = payload.features or {}

    try:
        if ds_id == "bike_sharing":
            state = ML_STATE["bike"]
            pipelines = state.get("pipelines", {})
            if not m_id or m_id not in pipelines:
                m_id = "poly_ridge" if "poly_ridge" in pipelines else list(pipelines.keys())[0]
            pipe = pipelines[m_id]

            full_feat = {
                "season": int(features.get("season", 3)),
                "yr": 1,
                "mnth": int(features.get("mnth", 7)),
                "hr": int(features.get("hr", 17)),
                "holiday": 0,
                "weekday": 4,
                "workingday": int(features.get("workingday", 1)),
                "weathersit": int(features.get("weathersit", 1)),
                "temp": float(features.get("temp", 0.65)),
                "atemp": float(features.get("temp", 0.65)),
                "hum": float(features.get("hum", 0.50)),
                "windspeed": float(features.get("windspeed", 0.20))
            }
            pred_bikes = bike_regression.predict_sample(pipe, full_feat)
            
            # Weather description for subtitle
            weather_labels = {1: "Clear Skies", 2: "Mist / Cloudy", 3: "Light Rain / Snow", 4: "Heavy Rain / Storm"}
            w_desc = weather_labels.get(full_feat["weathersit"], "Standard Weather")
            
            return {
                "status": "success",
                "task_type": "Regression",
                "dataset_id": ds_id,
                "model_id": m_id,
                "prediction": pred_bikes,
                "formatted_result": f"{pred_bikes:,} bikes / hour",
                "subtitle": f"Estimated hourly rental volume ({m_id}) under {w_desc}"
            }

        elif ds_id == "online_shoppers":
            state = ML_STATE["shoppers"]
            pipelines = state.get("pipelines", {})
            if not m_id or m_id not in pipelines:
                m_id = "ridge_log" if "ridge_log" in pipelines else list(pipelines.keys())[0]
            pipe = pipelines[m_id]

            full_feat = {
                "Administrative": int(features.get("Administrative", 1)),
                "Administrative_Duration": float(features.get("Administrative_Duration", 20.0)),
                "Informational": 0,
                "Informational_Duration": 0.0,
                "ProductRelated": int(features.get("ProductRelated", 25)),
                "ProductRelated_Duration": float(features.get("ProductRelated_Duration", 800.0)),
                "BounceRates": float(features.get("BounceRates", 0.01)),
                "ExitRates": float(features.get("ExitRates", 0.03)),
                "PageValues": float(features.get("PageValues", 15.0)),
                "SpecialDay": 0.0,
                "Month": str(features.get("Month", "Nov")),
                "OperatingSystems": 2,
                "Browser": 2,
                "Region": 1,
                "TrafficType": 2,
                "VisitorType": str(features.get("VisitorType", "Returning_Visitor")),
                "Weekend": int(features.get("Weekend", 0))
            }
            res = shoppers_classification.predict_sample(pipe, full_feat)
            return {
                "status": "success",
                "task_type": "Classification",
                "dataset_id": ds_id,
                "model_id": m_id,
                "prediction": res["prediction"],
                "formatted_result": res["label"],
                "confidence": res["confidence"],
                "confidence_percent": res["confidence_percent"],
                "purchase_probability": res["purchase_probability"],
                "probability_percent": res["probability_percent"],
                "subtitle": f"Prediction Confidence: {res['confidence_percent']} | Purchase Probability: {res['probability_percent']}"
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    raise HTTPException(status_code=404, detail=f"Dataset '{ds_id}' not found.")


@app.get("/api/comparison/{dataset_id}")
def get_comparison(dataset_id: str):
    """Returns the empirical benchmark comparison table for the chosen dataset."""
    if dataset_id == "bike_sharing":
        return {
            "dataset_id": "bike_sharing",
            "task_type": "Regression",
            "metrics": ML_STATE["bike"].get("metrics", [])
        }
    elif dataset_id == "online_shoppers":
        return {
            "dataset_id": "online_shoppers",
            "task_type": "Classification",
            "metrics": ML_STATE["shoppers"].get("metrics", [])
        }
    raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")


@app.get("/api/diagnostics/{dataset_id}/{model_id}")
def get_diagnostics(dataset_id: str, model_id: str):
    """Returns diagnostic data: actual vs predicted scatter for regression, or confusion matrix for classification."""
    if dataset_id == "bike_sharing":
        state = ML_STATE["bike"]
        pipelines = state.get("pipelines", {})
        if not model_id or model_id not in pipelines:
            model_id = "poly_ridge" if "poly_ridge" in pipelines else list(pipelines.keys())[0]
        pipe = pipelines[model_id]
        X_test = state["X_test"]
        y_test = state["y_test"]

        preds = pipe.predict(X_test)
        indices = np.linspace(0, len(y_test) - 1, min(100, len(y_test)), dtype=int)
        
        return {
            "type": "scatter",
            "model_id": model_id,
            "points": [
                {"actual": round(float(y_test.iloc[i]), 1), "predicted": max(0.0, round(float(preds[i]), 1))}
                for i in indices
            ]
        }
    
    elif dataset_id == "online_shoppers":
        state = ML_STATE["shoppers"]
        pipelines = state.get("pipelines", {})
        if not model_id or model_id not in pipelines:
            model_id = "ridge_log" if "ridge_log" in pipelines else list(pipelines.keys())[0]
        pipe = pipelines[model_id]
        X_test = state["X_test"]
        y_test = state["y_test"]

        cm = shoppers_classification.get_confusion_matrix(pipe, X_test, y_test)
        return {
            "type": "confusion_matrix",
            "model_id": model_id,
            "confusion_matrix": cm
        }

    raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")


@app.get("/api/insights/{dataset_id}")
def get_insights(dataset_id: str):
    """Returns distribution summary for UI chart rendering."""
    if dataset_id == "bike_sharing":
        df = ML_STATE["bike"]["df"]
        hist_counts, bin_edges = np.histogram(df["cnt"], bins=20)
        return {
            "task_type": "Regression",
            "bins": [
                {"range": f"{int(bin_edges[i])}-{int(bin_edges[i+1])}", "count": int(hist_counts[i])}
                for i in range(len(hist_counts))
            ],
            "stats": {"min": int(df["cnt"].min()), "mean": round(float(df["cnt"].mean()), 1), "max": int(df["cnt"].max())}
        }
    elif dataset_id == "online_shoppers":
        df = ML_STATE["shoppers"]["df"]
        revenue_counts = df["Revenue"].value_counts().to_dict()
        return {
            "task_type": "Classification",
            "class_distribution": [
                {"label": "Browsing Only (0)", "count": int(revenue_counts.get(0, 0))},
                {"label": "Purchase Made (1)", "count": int(revenue_counts.get(1, 0))}
            ]
        }
    raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
