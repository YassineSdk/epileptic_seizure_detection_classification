# Epileptic Seizure Detection & Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.140+-009688)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.3+-orange)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.7+-green)](https://lightgbm.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2+-red)](https://catboost.ai/)
[![Weights & Biases](https://img.shields.io/badge/W%26B-Experiments-yellow)](https://wandb.ai/)

> **Automated multi-class epileptic seizure classification** using multichannel EEG recordings from the Bangalore EEG Epilepsy Dataset (BEED). This project delivers a production-ready ML pipeline — from exploratory data analysis and hyperparameter optimization to a stacking ensemble model served via a FastAPI REST endpoint.

---

## Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Exploratory Findings](#-exploratory-findings)
- [Modeling Pipeline](#-modeling-pipeline)
- [Results Summary](#-results-summary)
- [Deployment: FastAPI API](#-deployment-fastapi-api)
- [Getting Started](#-getting-started)
- [Dependencies](#-dependencies)
- [Experiment Tracking](#-experiment-tracking)
- [Logging](#-logging)

---

## 🧬 Project Overview

Epilepsy is one of the most common neurological disorders, affecting millions worldwide. The cornerstone of epilepsy diagnosis is the electroencephalogram (EEG), but manual interpretation is time-consuming and requires specialized expertise. This project automates the classification of EEG recordings into **four clinical categories** using ensemble gradient boosting models, deployed as a REST API for real-time inference.

### Seizure Classification Targets

| Class | Label | Description |
|-------|-------|-------------|
| **0** | Healthy | Control group — no epileptic activity |
| **1** | Generalized Seizure | Seizures involving both hemispheres simultaneously |
| **2** | Focal Seizure | Seizures originating in a localized brain region |
| **3** | Seizure Event | Recording of ongoing seizure activity (e.g., eye blinking, staring) |

---

## Dataset

The **Bangalore EEG Epilepsy Dataset (BEED)** consists of multichannel EEG recordings from four subject groups. Each sample contains **16 numerical features** (`X1`–`X16`), representing voltage measurements from 16 EEG channels, along with the target class (`y`).

### Key Statistics

| Attribute | Value |
|-----------|-------|
| **Total Samples** | ~11,500 |
| **Features** | 16 EEG channel voltage readings |
| **Target** | Multiclass (0 = Healthy, 1 = Generalized, 2 = Focal, 3 = Event) |
| **Class Distribution** | Perfectly balanced — each class represents **25%** of the data |

---

## Project Structure

```
epileptic_seizure_detection_classification/
├── Data/
│   └── BEED_Data.csv                     # EEG dataset (16-channel + target)
├── models/
│   ├── stacking.joblib                   # Final stacking ensemble model
│   ├── XGB_v1.joblib                     # Optimized XGBoost model
│   ├── LGBM_v1.joblib                    # Optimized LightGBM model
│   └── CAT_v1.joblib                     # Optimized CatBoost model
├── notebooks/
│   ├── EDA.ipynb                         # Exploratory Data Analysis
│   ├── Modeling.ipynb                    # Baseline model training & evaluation
│   ├── Hyperparam_Tunning.ipynb          # Optuna hyperparameter optimization
│   └── Stacked_Ensemble_Model.ipynb      # Final stacking ensemble construction
├── src/
│   ├── main.py                           # FastAPI application (REST API)
│   ├── Predictor.py                      # Inference logic & class mapping
│   ├── ModelLoader.py                    # Model loading from disk / W&B
│   ├── helpers.py                        # File validation & data structure checks
│   ├── config_logger.py                  # Centralized logging setup
│   └── requirements.txt                  # Python dependencies
├── logs/
│   └── api.log                           # API request logging
├── .gitignore
├── .env.example                          # W&B API key template
└── README.md
```

---

## Exploratory Findings

The EDA revealed several important insights about the EEG data:

### Data Distribution
- **High Variability:** Healthy subjects (class 0) exhibit substantially greater feature variability than seizure-related classes, suggesting more diverse neural activity patterns.
- **Class Separation:** Median values across the four classes are often close, indicating that individual features alone cannot provide strong class separation — the classifier benefits from learning multivariate relationships.

### Correlation Structure
- **Strong Multicollinearity:** Adjacent EEG channels show strong positive correlations, consistent with spatially proximal electrode recordings. The structured correlation pattern suggests tree-based models (which handle multicollinearity well) are preferred over linear models.

### Feature Relevance
- **All 16 Features Matter:** Mutual information analysis confirms that discriminative information is distributed across all predictors rather than concentrated in a subset.

### PCA Visualization
- **Moderate Cluster Separation:** 2D PCA projection shows classes with distinct but overlapping distributions, supporting the need for non-linear classifiers.

---

## Modeling Pipeline

### 1️. Baseline Comparison

Five classifiers were benchmarked using default hyperparameters:

| Model | Accuracy | F1-Weighted | Precision | Recall |
|-------|----------|-------------|-----------|--------|
| **CatBoost** | **96.88%** | **96.88%** | **96.89%** | **96.88%** |
| **LightGBM** | 96.67% | 96.67% | 96.67% | 96.67% |
| **XGBoost** | 96.42% | 96.42% | 96.42% | 96.42% |
| Random Forest | 95.83% | 95.84% | — | — |
| Logistic Regression | 46.83% | 47.52% | — | — |

> **Key Insight:** Ensemble gradient boosting models significantly outperform linear models, confirming the non-linear nature of EEG-seizure relationships. The top three models were selected for further optimization.

### 2️. Cross-Validation Stability

A 5-fold stratified cross-validation assessed performance stability:

| Model | Mean F1 | Std F1 |
|-------|---------|--------|
| XGBoost | 96.30% | 0.002 |
| LightGBM | 96.37% | 0.004 |
| CatBoost | 96.79% | 0.002 |

> Minimal variance across folds confirms strong generalization and stable performance.

### 3️. Hyperparameter Optimization

**Optuna** was used for Bayesian hyperparameter search (20 trials per model) with separate validation splits (70/30 train/validation from the training set). Key tuned parameters include:

| Model | Tuned Parameters |
|-------|-----------------|
| **XGBoost** | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda` |
| **LightGBM** | `n_estimators`, `learning_rate`, `num_leaves`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda` |
| **CatBoost** | `iterations`, `depth`, `learning_rate`, `l2_leaf_reg`, `random_strength`, `bagging_temperature`, `border_count` |

### 4️. Final Model: Stacking Ensemble

The final model is a **stacking ensemble** that combines the three optimized gradient boosters:

```
┌─────────────┐   ┌────────────────────┐   ┌──────────────┐
│   XGBoost   │──▶│                    │   │              │
├─────────────┤   │                    │   │   Predicted  │
│   LightGBM  │──▶│   Stacking         │──▶│   Class      │
├─────────────┤   │   Classifier       │   │  (0 / 1 / 2 / 3)
│   CatBoost  │──▶│                    │   │              │
└─────────────┘   │   Meta-Learner:    │   └──────────────┘
                  │   LogisticRegression│
                  └────────────────────┘
```

| Configuration | Details |
|--------------|---------|
| **Base Estimators** | XGBoost, LightGBM, CatBoost (optimized) |
| **Meta-Learner** | Logistic Regression (max_iter=1000) |
| **Stacking Method** | `predict_proba` |
| **Cross-Validation** | 5-fold |
| **Final F1-Weighted** | **~97.2%** |

---

## 📈 Results Summary

| Stage | Model | F1-Weighted | Notes |
|-------|-------|-------------|-------|
| Baseline | CatBoost | **96.88%** | Best default model |
| Cross-Validation | CatBoost | 96.79% ± 0.002 | Stable across folds |
| Tuned | XGBoost / LGBM / CAT | ~97% | Optuna optimization |
| **Final** | **Stacking Ensemble** | **~97.2%** | **Best overall performance** |

---

## 🌐 Deployment: FastAPI API

The trained stacking model is deployed as a REST API using **FastAPI** with uvicorn.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check — returns API status & info |
| `GET` | `/docs` | Interactive Swagger documentation |
| `POST` | `/inference` | Upload EEG data for classification |

### Inference Request

```
POST /inference
Content-Type: multipart/form-data

datafile: <CSV / Excel file with 16 columns (X1–X16)>
```

The API supports:
- **CSV** (`.csv`) and **Excel** (`.xls` / `.xlsx`) formats
- **Single sample:** Returns one label (e.g., `"healthy"`)
- **Batch prediction:** Returns a list of labels

### Input Validation

| Rule | Description |
|------|-------------|
| ✅ File Extension | Allowed: `.csv`, `.xls`, `.xlsx` |
| ✅ Column Count | Exactly 16 feature columns (`X1`–`X16`) |
| ✅ Missing Values | No null values allowed |
| ✅ Data Types | All values must be numeric |
| ✅ Unnamed Columns | Automatically removed if present |

### Output Classes

```python
CLASS_NAMES = ["healthy", "generalize", "focal", "event"]
```

### Example Usage

```python
import requests

url = "http://localhost:8000/inference"
files = {"datafile": open("eeg_sample.csv", "rb")}
response = requests.post(url, files=files)
print(response.json())
# Output: {"preds": "healthy"} or {"preds": ["healthy", "focal", ...]}
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- [Weights & Biases](https://wandb.ai/) account (for model download)
- W&B API Key (set in `.env`)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/epileptic_seizure_detection_classification.git
cd epileptic_seizure_detection_classification

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r src/requirements.txt

# Set up environment variables
cp .env.example .env
# Add your W&B API key to .env:
# WANDB_API_KEY=your_wandb_api_key
```

### Running Notebooks

```bash
jupyter notebook notebooks/
```

Execute notebooks in order:
1. **`EDA.ipynb`** — Exploratory analysis & insights
2. **`Modeling.ipynb`** — Baseline model comparison
3. **`Hyperparam_Tunning.ipynb`** — Hyperparameter optimization with Optuna
4. **`Stacked_Ensemble_Model.ipynb`** — Final stacking ensemble

### Running the API

```bash
cd src/
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Visit [http://localhost:8000/docs](http://localhost:8000/docs) for interactive API documentation.

---

## Dependencies

```
fastapi==0.140.13
uvicorn[standard]==0.37.0

numpy==2.4.4
pandas==2.3.3
scipy==1.17.1
scikit-learn==1.8.0
joblib==1.5.3

wandb==0.28.1
python-dotenv==1.2.2
python-multipart==0.0.20

catboost==1.2.10
lightgbm==4.7.0
xgboost==3.3.0
```

---

## Experiment Tracking

All experiments are logged to **[Weights & Biases](https://wandb.ai/)**:

| Detail | Value |
|--------|-------|
| **Project** | `Epeliptic_seizure_classification` |
| **Baseline Group** | `Baseline-comparison-v1` |
| **Models Tracked** | XGBoost, LightGBM, CatBoost, Logistic Regression, Random Forest |
| **Artifacts** | Trained `.joblib` models with metadata (hyperparameters, performance metrics) |

The model loader (`ModelLoader.py`) automatically downloads the latest stacking ensemble artifact from W&B if no cached model exists locally, making deployment seamless.

---

## Logging

The API logs all inference requests to `logs/api.log` with the following format:

```
2025-03-22 14:30:15 | INFO     | EEG_API | a prediction request received:
2025-03-22 14:30:15 | INFO     | EEG_API | the data set is valid, starting the inference job unit
```

---

## License

This project is for research and educational purposes.

---

## Acknowledgments

- **Bangalore EEG Epilepsy Dataset (BEED)** — for providing the EEG recordings
- **[Weights & Biases](https://wandb.ai/)** — for experiment tracking and model registry
- **[Optuna](https://optuna.org/)** — for hyperparameter optimization
- **[Scikit-learn](https://scikit-learn.org/)** — for the stacking classifier framework
