# Phase 1 — EDA Summary

## Dataset
- 545 rows × 13 cols, 0 missing, 0 duplicate
- Target: price (₹), range 17.5L – 1.33Cr
- mean 47.6L, median 43.4L, skew 1.21, kurtosis 1.96

## Key findings
1. price is right-skewed (skew 1.21) → candidate for log1p transform
2. area is the strongest numeric feature (corr 0.54)
3. bathrooms (0.52) > bedrooms (0.37) → bathrooms are a better luxury proxy
4. prefarea and airconditioning show large median price gaps → high-signal binaries
5. hotwaterheating is ~4.6% yes (25/545) → rare; watch for coefficient inflation in linear models
6. furnishingstatus is ORDINAL (unfurnished < semi-furnished < furnished).

## Baselines (single train/test split, seed=42)
| Model       | MAE (₹)   | RMSE (₹)  | R²     |
|-------------|-----------|-----------|--------|
| T0 Mean     | 17,48,625 | 22,68,298 | −0.018 |
| T0 Median   | 17,63,904 | 23,59,660 | −0.102 |
| T1 Area-only| 14,74,748 | 19,17,104 |  0.273 |
| T2 Linear   |  9,70,043 | 13,24,507 |  0.653 |

## Notes
- Negative R² on T0-Median is EXPECTED: mean > median under right skew.
- Baselines are single-split estimates. Phase 2 replaces them with 5-fold KFold CV.

## Decisions for Phase 2
- Pipeline + ColumnTransformer (reproducible; no leakage)
- KFold(n_splits=5, shuffle=True, random_state=42)
- Encoding: numeric → StandardScaler; binary → OneHotEncoder(drop="if_binary");
  furnishingstatus → OrdinalEncoder (ordered)
- Keep all features
- A/B test target = price (raw) vs log1p(price)
- Models: RidgeCV, LassoCV, ElasticNet, RandomForest, GradientBoosting