import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline

from .features import FEATURE_BUILDERS
from .preprocess import TARGET, build_preprocessor
from .train import apply_features, load_data


def run_smearing_experiment():
  print("═══ Experiment E8 — Smearing Correction ═══")
  df = load_data()
  df_all = apply_features(df, list(FEATURE_BUILDERS.keys()))
  X, y = df_all.drop(columns=TARGET), df_all[TARGET]

  if isinstance(y, (pd.DataFrame, pd.Series)):
    y_vals = y.values.ravel()
  else:
    y_vals = np.ravel(y)

  y_train_log = np.log1p(y_vals)

  ridge_log_model = RidgeCV(alphas=np.logspace(-2, 4, 50))
  pipe_log = Pipeline([
      ("prep", build_preprocessor(list(X.columns))),
      ("model", ridge_log_model),
  ])

  pipe_log.fit(X, y_train_log)
  pred_train_log = np.ravel(pipe_log.predict(X))

  smearing = np.mean(np.exp(y_train_log - pred_train_log))
  print(f"Smearing factor: {smearing:.4f}   (expect ~1.02–1.05)")


if __name__ == "__main__":
  run_smearing_experiment()