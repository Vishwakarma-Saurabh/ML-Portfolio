# Phase 4 — Tuning + Evaluation + Diagnosis

## A. Tuning (RandomizedSearchCV, 40 iter, log1p target)
| Model | Plain CV MAE | Params |
|-------|--------------|--------|
| RandomForest     | ₹7,65,367 | depth=19, max_features=0.5, n=329 |
| GradientBoosting | ₹7,69,366 | depth=5, lr=0.02, n=148, subsample=0.8 |

## B. Nested CV 
| Model | Nested MAE | Gap vs plain CV |
|-------|------------|-----------------|
| Ridge | ₹7,80,678 | 0 (RidgeCV is inherently honest) |
| RF    | ₹7,68,323 | +₹2,956 (0.4%) — small but real |

→ DEPLOYMENT NUMBER: **₹7,68,323 MAE** (RF, all-features, log1p)

## C. Learning curve
- Train MAE: ₹7,52,779, CV MAE: ₹7,80,678 → gap ₹27,899 (3.6%)
- Curve flattens at ~80% training size
- **Diagnosis: Feature-limited (Shape B).** More rows would help <2%.

## D. Residual diagnostics — TWO FINDINGS
1. **Systematic mean bias +₹1L across all groups** → Jensen's inequality
   from log1p → expm1. Fix with Duan's smearing correction in Phase 5.
2. **semi-furnished group under-predicted by +₹2.64L vs −₹64k for furnished.**
   Root cause: OrdinalEncoder forces a linear relationship that violates
   data (semi is market sweet spot). Fix: TEST one-hot encoding for
   furnishingstatus in CV.

## Actions before Phase 5
- [ ] A/B test: OrdinalEncoder vs OneHotEncoder for furnishingstatus
- [ ] Add Duan's smearing correction to the final predict path
- [ ] Choose final model: RF (tuned) vs GB (tuned) based on A/B + stability
- [ ] Persist final model as a single joblib pipeline