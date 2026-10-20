# Phase 2 — CV Baseline (5-fold, seed=42)

## Best model
Ridge on log1p(price): MAE ₹7,83,333, RMSE ₹10,84,278, R² 0.663

## Notable observations
1. All models within ₹7.82L–₹8.06L MAE → dataset signal ceiling
   under raw features ≈ R² 0.66.
2. No tree model outperforms linear → raw features are approximately
   additive (Phase 3 feature engineering will target interactions).
3. ElasticNet ≡ Lasso on raw target (l1_ratio selected 1.0) → L2
   provides no benefit → drop ElasticNet from zoo.
4. log1p(price) helps Ridge by 2.2% MAE, tree models by <1%. → Use
   log1p for linear models in Phase 3+.
5. Improvement over Phase 1 (₹9.70L) = 19.4% MAE. Phase 3 must
   break ₹7.8L to justify feature engineering.

## Decisions for Phase 3
- Winner backbone: Ridge on log1p(price)
- Feature engineering targets:
  * area_per_room  (area / (bedrooms+bathrooms))
  * total_rooms    (bedrooms + bathrooms)
  * amenities      (count of "yes" flags)
  * area_x_prefarea (interaction — the one real "multiplicative" signal)
- Re-run 5-fold CV; accept feature only if CV MAE drops ≥ 1%
- Trees get a second chance after feature engineering