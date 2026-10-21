
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import learning_curve, KFold, cross_val_predict
from sklearn.compose import TransformedTargetRegressor

from .preprocess import build_preprocessor, TARGET
from .features import FEATURE_BUILDERS
from .train import apply_features, load_data


def build_final_pipeline(cols):
    return Pipeline([
        ("prep",  build_preprocessor(list(cols))),
        ("model", TransformedTargetRegressor(
            regressor=RidgeCV(alphas=np.logspace(-2, 4, 50)),
            func=np.log1p, inverse_func=np.expm1,
        )),
    ])


def plot_learning_curve(X, y):
    pipe = build_final_pipeline(X.columns)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    train_sizes, train_scores, cv_scores = learning_curve(
        pipe, X, y, cv=cv,
        train_sizes=np.linspace(0.2, 1.0, 6),
        scoring="neg_mean_absolute_error", n_jobs=-1,
    )
    train_mae = -train_scores.mean(axis=1)
    cv_mae    = -cv_scores.mean(axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mae, "o-", label="Train MAE")
    plt.plot(train_sizes, cv_mae,    "s-", label="CV MAE")
    plt.xlabel("Training set size")
    plt.ylabel("MAE (₹)")
    plt.title("Learning curve — Ridge, all-features")
    plt.legend()
    plt.grid(True)
    plt.savefig("reports/figures/phase4c_learning_curve.png", dpi=120)
    plt.show()

    print(f"Train MAE at 100%: ₹{train_mae[-1]:,.0f}")
    print(f"CV    MAE at 100%: ₹{cv_mae[-1]:,.0f}")
    print(f"Gap             : ₹{train_mae[-1] - cv_mae[-1]:,.0f}")


def plot_residuals(X, y):
    pipe = build_final_pipeline(X.columns)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)
    residuals = y - y_pred

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[0].axhline(0, color="red", lw=1)
    axes[0].set_xlabel("Predicted price (₹)")
    axes[0].set_ylabel("Residual (₹)")
    axes[0].set_title("Residual vs Predicted")

    axes[1].hist(residuals, bins=40, edgecolor="black")
    mean_val = float(np.ravel(residuals).mean())
    axes[1].set_title(f"Residual distribution (mean={mean_val:,.0f})")

    axes[2].scatter(y, y_pred, alpha=0.5, s=10)
    axes[2].plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=1)
    axes[2].set_xlabel("Actual price (₹)")
    axes[2].set_ylabel("Predicted price (₹)")
    axes[2].set_title("Predicted vs Actual")

    plt.tight_layout()
    plt.savefig("reports/figures/phase4c_residuals.png", dpi=120)
    plt.show()

    df_diag = X.copy()
    df_diag["residual"] = residuals.values
    print("\nMean residual by furnishingstatus:")
    print(df_diag.groupby("furnishingstatus")["residual"].mean().round(0))
    print("\nMean residual by prefarea:")
    print(df_diag.groupby("prefarea")["residual"].mean().round(0))


def main():
    df = apply_features(load_data(), list(FEATURE_BUILDERS.keys()))
    X, y = df.drop(columns=TARGET), df[TARGET]
    print("═══ Learning curve ═══")
    plot_learning_curve(X, y)
    print("\n═══ Residual diagnostics ═══")
    plot_residuals(X, y)


if __name__ == "__main__":
    main()