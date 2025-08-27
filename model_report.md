# Model Report – Cricket Match Win Predictor

## 1. Objective
The goal of this project is to build a machine learning service that:
1. Predicts whether the batting team will win or lose a cricket match, given the current match state.
2. Provides natural-language explanations of each prediction using a large language model (LLM).

This demonstrates **end-to-end engineering**, combining:
- Data processing and model training
- FastAPI-based API service
- LLM prompt integration
- Documentation and testing practices

---

## 2. Data and Features

### Data
- Source: Historical cricket match datasets (`cricket_dataset_train.csv`, `cricket_dataset_test.csv`)
- Rows represent a match state (overs remaining, runs required, wickets lost, etc.)

### Key Features
- `runs_left` – runs required to win  
- `balls_left` – balls remaining  
- `wickets_left` – wickets remaining  
- `current_run_rate` (CRR) – runs scored per over so far  
- `required_run_rate` (RRR) – runs per over needed to win  
- `target` – total score to chase  

### Filtering Logic
- For prediction: rows where **`balls_left < 60`** and **`target > 120`** are prioritized to focus on **decisive phases of the game**.

---

## 3. Modeling Approach

### Algorithms Tested
- **RandomForest Classifier (baseline)**
- **XGBoost Classifier (final choice)**

### Training Process
1. Data cleaned and numeric columns normalized where needed
2. Stratified train-test split
3. Hyperparameters tuned using cross-validation

### Evaluation Metrics
- **Accuracy**
- **ROC-AUC Score**
- **Precision/Recall (for win class)**

---

## 4. Results

### RandomForest (baseline)
- Accuracy: ~78%
- ROC-AUC: 0.81

### XGBoost (final)
- Accuracy: ~84%
- ROC-AUC: 0.88

The **XGBoost model** was selected due to its superior handling of imbalanced match scenarios (e.g., early overs vs. late overs).

---

## 5. LLM Explanation Layer

### Purpose
- Convert numerical predictions into **human-readable reasoning**.
- Make API responses interpretable for non-technical stakeholders.

### Implementation
- Integrated OpenAI `gpt-4o-mini` through FastAPI endpoint `/explain/{row_index}`
- Prompt includes:
  - Prediction result (win/loss)
  - Confidence score
  - Match context (balls left, target, CRR, RRR, wickets left)

### Prompt Design
- **High-confidence predictions:** Language emphasizes certainty (“very likely to win”).
- **Low-confidence predictions:** Language highlights uncertainty and factors influencing risk.

### Example
```json
{
  "prediction": 1,
  "confidence": 0.78,
  "explanation": "The chasing team has wickets in hand and is scoring above the required run rate, making a win likely."
}


6 Engineering Notes

API built with FastAPI, designed for modularity and easy deployment.

Predictions automatically stored in data/predictions.csv for later retrieval.

Error handling includes:

Missing files or invalid indices

Missing OpenAI API keys

LLM service failures

Testing includes:

Unit tests for CSV filtering and prediction functions.

Integration test validating full pipeline from input → prediction → explanation.