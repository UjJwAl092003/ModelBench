# ModelBench

**ModelBench** is an interactive machine learning experimentation and benchmarking platform developed as an M.Tech project for **CS801: Computing Lab** at the Department of Computer Science and Engineering, National Institute of Technology Karnataka (NITK), Surathkal.

---

## 📌 Project Overview

ModelBench focuses on the **empirical evaluation, diagnostic analysis, and comparative benchmarking of classical Machine Learning models** across curated, real-world datasets of substantial scale (~10,000–17,000 observations):

1. **Regression Benchmark:** *Bike Sharing Demand Dataset* (17,379 hourly observations) — Predicting urban rental demand (`cnt`) using weather, seasonal, and temporal interaction features.
2. **Classification Benchmark:** *Online Shoppers Purchasing Intention Dataset* (12,330 user sessions) — Classifying transaction conversion (`Revenue`) from high-dimensional browsing session telemetry with real-world class imbalance (~84.5% non-purchasing vs ~15.5% purchasing).

The platform enforces strict machine learning engineering practices:
- Leak-free preprocessing via Scikit-Learn `ColumnTransformer` and `Pipeline` objects fit strictly on training splits.
- Classical, non-tree algorithmic focus: Ordinary Least Squares (OLS), Ridge ($L_2$), Lasso ($L_1$), ElasticNet, Polynomial Feature Expansion ($d=2$), $K$-Nearest Neighbors (KNN), Gaussian Naive Bayes, and Support Vector Machines (Linear SVR / Linear SVC).
- Comprehensive statistical evaluation: $R^2$, MAE, RMSE for regression; Accuracy, Precision, Recall, F1-Score, and ROC-AUC for classification.
- Diagnostic visual analytics: Target distributions, Actual vs. Predicted scatter diagnostics ($y=x$ reference), and full 2×2 Confusion Matrix Heatmaps with Sensitivity/Specificity metrics.

---

## 👥 Team & Course Information

* **Course:** CS801 — Computing Lab (M.Tech I Year, Semester I)
* **Instructor:** Prof. P. Santhi Thilagam
* **Department:** Computer Science and Engineering, NITK Surathkal
* **Team Members:**
  * **Ujjwal Singh** (Roll No: `262IS036`)
  * **Aryan** (Roll No: `262IS007`)

---

## 📁 Repository Structure

```text
ModelBench/
│
├── datasets/                     # Curated real-world benchmark datasets
│   ├── bike_sharing.csv          # Hourly bike sharing demand (17,379 rows, Regression)
│   └── online_shoppers.csv       # E-commerce browsing sessions (12,330 rows, Classification)
│
├── notebooks/                    # Jupyter notebooks with complete EDA, mathematics, & experiments
│   ├── regression/
│   │   └── 01_bike_sharing_regression.ipynb
│   └── classification/
│       └── 02_online_shoppers_classification.ipynb
│
├── ml_core/                      # Production-ready, modular ML pipelines
│   ├── bike_regression.py        # Bike sharing preprocessing & 7 regression models
│   └── shoppers_classification.py# Shoppers preprocessing & 6 classification models
│
├── backend/                      # High-performance FastAPI application layer
│   └── main.py                   # In-memory pipeline cache, prediction, and diagnostic endpoints
│
├── frontend/                     # Interactive React + Vite web dashboard
│   ├── src/
│   │   ├── App.jsx               # Dynamic task-switching UI (Predict, Compare, Diagnostics)
│   │   └── index.css             # Clean design system & styling
│   └── package.json
│
├── docs/                         # Presentations, LaTeX reports, and viva preparation docs
├── README.md                     # Project overview and instructions
└── requirements.txt              # Python dependencies
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.10+ and Node.js 18+ installed.

### 2. Environment Setup
```bash
# Create and activate virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Running the Backend API
The backend exposes the pre-trained Scikit-Learn pipelines via FastAPI:
```bash
# From repository root:
python -m uvicorn backend.main:app --port 8000 --reload
```
API Documentation & Swagger UI will be live at: `http://127.0.0.1:8000/docs`

### 4. Running the Frontend Dashboard
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` (or `http://localhost:3000`) in your browser to interact with live model predictions, comparative benchmark scorecards, and diagnostic plots.

### 5. Running Jupyter Notebooks
```bash
jupyter notebook
```
Explore `notebooks/regression/01_bike_sharing_regression.ipynb` and `notebooks/classification/02_online_shoppers_classification.ipynb` for detailed exploratory data analysis, mathematical formulations, and step-by-step benchmark evaluations.
