# ModelBench

**ModelBench** is an interactive machine learning experimentation and benchmarking platform developed as an M.Tech project for **CS801: Computing Lab** at the Department of Computer Science and Engineering, National Institute of Technology Karnataka (NITK), Surathkal.

---

## 📌 Project Overview

ModelBench focuses primarily on the **empirical evaluation, diagnostic analysis, and comparative benchmarking of classical Machine Learning models** across curated, real-world datasets.

The project addresses the practical challenge of fragmented, ad-hoc experimentation in isolated notebooks by providing a consistent framework for:
- Standardized data cleaning, feature encoding, and scaling.
- Reproducible train/test data splitting.
- Training and hyperparameter exploration across multiple algorithms.
- Evaluation using task-appropriate statistical metrics ($R^2$, RMSE, MAE, Accuracy, F1-Score, ROC-AUC, etc.).
- Diagnostic error visualizations (Residual plots, Actual vs. Predicted curves, Confusion Matrices).
- Side-by-side comparative performance analysis on identical datasets.

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
├── datasets/                     # Curated real-world CSV datasets (Insurance, Auto MPG, Housing, etc.)
│
├── notebooks/                    # Jupyter notebooks for exploratory data analysis & model experiments
│   ├── regression/               # Regression experiments (Linear, Multiple, Polynomial, Ridge)
│   ├── classification/           # Classification experiments (Logistic, KNN, Trees, SVM)
│   └── other/                    # Clustering and unsupervised learning experiments
│
├── ml_core/                      # Extracted, reusable Python ML functions and pipelines
│
├── backend/                      # Lightweight application API layer (FastAPI)
│
├── frontend/                     # Minimal interactive web dashboard (React / Plotly)
│
├── docs/                         # Project reports, extended abstract (LaTeX), and presentation slides
│
├── .gitignore                    # Git ignore rules for Python, Jupyter, and IDE files
├── README.md                     # Project overview and documentation
└── requirements.txt              # Core Python dependencies
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Environment Setup
```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install required dependencies
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
The user interface connects directly to the FastAPI ML endpoints:
```bash
# Navigate to frontend and start the Vite dev server:
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser to interact with live model predictions, comparative benchmark scorecards, and diagnostic plots.

### 5. Running Jupyter Notebooks
```bash
jupyter notebook
```
Navigate to `notebooks/regression/` to explore the complete step-by-step EDA, pipeline training, diagnostic plots, and viva study notes.

---

## 🔬 Development Roadmap

1. **Phase 1 (Abstract & Baseline Setup):** Dataset curation, EDA, and baseline regression experiments.
2. **Phase 2 (Core Models & MidSem):** Multiple Linear, Polynomial, Ridge, and baseline classification models.
3. **Phase 3 (Benchmarking & Tree/Ensemble Models):** Decision Trees, Random Forests, SVM, and comparative performance dashboard.
4. **Phase 4 (Final Evaluation & Verification):** Interactive sensitivity analysis, comprehensive diagnostics, and final project reporting.
