# 🏠 House Price Prediction — End-to-End ML

Predict housing prices from structural and amenity features. Full ML lifecycle:
EDA → feature engineering → model selection → hyperparameter tuning → nested-CV
evaluation → containerized deployment.

## Quick start

    docker-compose up --build

- API docs:   http://localhost:8000/docs
- Frontend:   http://localhost:8501

## Metrics (5-fold CV, seed=42)

| Metric | Value |
|--------|-------|
| MAE    | ₹7,72,935 |
| RMSE   | ₹10,97,321 |
| R²     | 0.655 |

Nested CV bounded the search-selection bias at ~0.4%.

## Approach

| Phase | What |
|-------|------|
| 1  | EDA — skew, correlations, baselines (₹9.70L MAE) |
| 2  | Pipeline + KFold CV — linear vs tree models |
| 3  | Feature engineering — ablated 5 candidates, all rejected (<1% gain) |
| 4  | RandomizedSearchCV + nested CV + learning-curve diagnosis |
| 5  | FastAPI service + joblib pipeline + Docker |
| 6  | Streamlit frontend + docker-compose |

## Key findings

- Dataset has a **signal ceiling at R² ≈ 0.66**; more rows would help <2% (feature-limited).
- `area`, `bathrooms`, `prefarea` dominate; `bedrooms` is *negatively* correlated with price
  after controlling for area — smaller rooms per bedroom cost more per sq ft.
- Log-target (`log1p`) improves linear models by 2.2%; tree models less.
- **Jensen bias**: `expm1` inverse of log predictions under-predicts mean by ~2.15%.
  API exposes `price_inr` (MAE-optimal) and `price_inr_bias_corrected` (mean-calibrated).
- Residual analysis revealed a `semi-furnished` encoding trap (ordinal → biased);
  fixed via one-hot for linear models, retained ordinal for trees (model-dependent).

## Repo layout

    src/preprocess.py   ColumnTransformer + schema
    src/features.py     Feature engineering registry
    src/train.py        Ablation harness
    src/tune.py         RandomizedSearchCV
    src/nested_cv.py    Honest evaluation
    src/diagnose.py     Learning curve + residuals
    src/finalize.py     Final fit + serialize
    app/main.py         FastAPI
    app/streamlit_app.py UI
    etc......

## Limitations & next steps

To break the R² 0.66 ceiling, we need *new* features, not new models:
- Year built / property age
- Distance to transit / school / hospital
- Locality-level crime, income, walkability score

More rows would help <2% — **the bottleneck is features, not data.**