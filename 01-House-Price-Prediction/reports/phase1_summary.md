# Phase 1 — EDA Summary

## Dataset
- 545 rows × 13 cols
- 0 missing, 0 duplicate
- Target: price (₹), range 17.5L – 1.33Cr, mean 47.6L, median 43.4L, skew 1.21, kurt 1.96

## Key findings
1. price is right-skewed (skew ≈ 1.21) → candidate for log1p transform
2. area is the strongest numeric feature (corr 0.54)
3. bathrooms (0.52) > bedrooms (0.37) — bathrooms are a better luxury proxy
4. prefarea and airconditioning show large price gaps → high-signal
5. hotwaterheating is ~4% yes → rare feature, risk of overfit on linear models

## Baselines
| Model         |   MAE (₹)        |   RMSE (₹)            |   R²         |
|---------------|----------------------------------------------------------
| T0 Median     | | MAE: 1,763,904 | RMSE: 5,567,995,084,862 | R2: -0.102 |
| T1 Area-only  | | MAE: 1,474,748 | RMSE: 3,675,286,604,768 | R2: 0.273  |
| T2 Linear     | | MAE: 970,043   | RMSE: 1,754,318,687,331 | R2: 0.653  |

## Decisions for Phase 2
- Use pipeline + ColumnTransformer for reproducibility
- Use KFold CV (n=5) for honest scores
- Keep all features (no feature dropped for now)
- Try log1p(price) as a target-transform experiment

<!-- i have also find all the feature correlation by mapping them. -->