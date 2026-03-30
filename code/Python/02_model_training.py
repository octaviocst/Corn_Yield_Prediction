"""
Corn Yield Prediction with Machine Learning and Remote Sensing
Script 02 — Model Training & Evaluation (Python — scikit-learn)

Description:
    Trains and evaluates three ML algorithms (RF, SVM, kNN) for each
    atmospheric correction method and season using repeated k-fold
    cross-validation. Saves per-model metrics and observed vs. predicted
    values for downstream visualization.

Input:
    - data/processed/{season}/dataset_{correction}.csv  (from 01_data_preparation.py)

Output (per season/correction/model):
    - results/{season}/{correction}_{model}_metricas.csv     — R², RMSE, MAE, MSE
    - results/{season}/{correction}_{model}_obs_pred.csv     — obs vs. pred (test set)
    - results/figures/{season}/scatter_{correction}_{model}.png

Authors: Octávio Pereira da Costa · Thiago Orlando Costa Barboza
Institution: UFLA — Universidade Federal de Lavras
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures

from sklearn.ensemble        import RandomForestRegressor
from sklearn.svm             import SVR
from sklearn.neighbors       import KNeighborsRegressor
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import (train_test_split,
                                     RepeatedKFold,
                                     cross_val_score)
from sklearn.metrics         import (mean_absolute_error,
                                     mean_squared_error,
                                     r2_score)

warnings.filterwarnings("ignore")

# =============================================================================
# 0. CONFIGURATION
# =============================================================================

PATH_PROCESSED = "data/processed"
PATH_RESULTS   = "results"

TARGET_COL  = "Prod_13_KRIG"
TRAIN_SPLIT = 0.80   # 80 % train / 20 % test
RANDOM_SEED = 123

# Repeated k-fold CV settings
CV_FOLDS   = 10
CV_REPEATS = 10

# Algorithms — name : estimator (scaling handled inside Pipeline)
MODELS = {
    "RF":  RandomForestRegressor(n_estimators=500, random_state=RANDOM_SEED),
    "SVM": SVR(kernel="rbf", C=1.0, epsilon=0.1),
    "KNN": KNeighborsRegressor(n_neighbors=5),
}

# Rename L2A → Sen2Cor in output files (consistent with the paper)
CORRECTION_LABELS = {"L2A": "Sen2Cor"}

# Y-axis limits for scatter plots differ between seasons
PLOT_LIMITS = {2020: (6, 9), 2022: (9, 14)}

# =============================================================================
# 1. HELPER FUNCTIONS
# =============================================================================

def evaluate_model(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   model_name: str,
                   correction: str) -> pd.DataFrame:
    """Return a one-row DataFrame with MAE, MSE, RMSE and R² metrics."""
    return pd.DataFrame([{
        "Modelo"   : model_name,
        "Correcao" : correction,
        "MAE"      : mean_absolute_error(y_true, y_pred),
        "MSE"      : mean_squared_error(y_true, y_pred),
        "RMSE"     : np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2"       : r2_score(y_true, y_pred),
    }])


def make_scatter_plot(y_obs: np.ndarray, y_pred: np.ndarray,
                      title: str, lims: tuple,
                      save_path: str) -> None:
    """Save an observed vs. predicted scatter plot with R² annotation."""
    r2 = r2_score(y_obs, y_pred)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_obs, y_pred, alpha=0.6, s=25, color="steelblue", edgecolors="none")
    ax.plot(lims, lims, color="red", linestyle="--", linewidth=1.2, label="1:1 line")
    ax.annotate(f"R² = {r2:.2f}",
                xy=(lims[0] + (lims[1] - lims[0]) * 0.05,
                    lims[1] - (lims[1] - lims[0]) * 0.08),
                fontsize=12, color="navy")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Observed (sc/ha)", fontsize=11)
    ax.set_ylabel("Predicted (sc/ha)", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# =============================================================================
# 2. MAIN TRAINING LOOP
# =============================================================================

def train_season(year: int) -> None:
    processed_dir = os.path.join(PATH_PROCESSED, str(year))
    csv_files     = [f for f in os.listdir(processed_dir)
                     if f.startswith("dataset_") and f.endswith(".csv")]

    if not csv_files:
        warnings.warn(f"No processed files found for {year}. "
                       "Run 01_data_preparation.py first.")
        return

    plot_lims = PLOT_LIMITS.get(year, (0, 20))

    for csv_file in sorted(csv_files):
        # Extract correction label from filename (e.g. "dataset_DOS.csv" → "DOS")
        correction_raw = csv_file.replace("dataset_", "").replace(".csv", "")
        correction     = CORRECTION_LABELS.get(correction_raw, correction_raw)

        print(f"\n  Correction: {correction}")

        df = pd.read_csv(os.path.join(processed_dir, csv_file))

        # Drop non-feature columns
        drop_cols = [c for c in ["ID", "id_2"] if c in df.columns]
        df = df.drop(columns=drop_cols)

        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]

        # Train / test split (stratified by yield quantile for balance)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1 - TRAIN_SPLIT,
            random_state=RANDOM_SEED
        )

        # Output directories
        results_dir = os.path.join(PATH_RESULTS, str(year))
        figures_dir = os.path.join(PATH_RESULTS, "figures", str(year))
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)

        cv = RepeatedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS,
                           random_state=RANDOM_SEED)

        for model_name, estimator in MODELS.items():
            print(f"    Training: {model_name}", end="  ")

            # Pipeline: StandardScaler → estimator
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model",  estimator)
            ])

            # Cross-validation R² scores on training set
            cv_r2 = cross_val_score(pipeline, X_train, y_train,
                                    cv=cv, scoring="r2", n_jobs=-1)
            cv_rmse = np.sqrt(-cross_val_score(pipeline, X_train, y_train,
                                               cv=cv,
                                               scoring="neg_mean_squared_error",
                                               n_jobs=-1))

            # Save CV summary
            cv_summary = pd.DataFrame({
                "fold_repeat": range(len(cv_r2)),
                "Rsquared"   : cv_r2,
                "RMSE"       : cv_rmse,
            })
            prefix = os.path.join(results_dir,
                                   f"{correction}_{model_name}")
            cv_summary.to_csv(f"{prefix}_resultados.csv", index=False)

            # Fit on full training set
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            # Test-set metrics
            metrics = evaluate_model(y_test.values, y_pred,
                                     model_name, correction)
            metrics.to_csv(f"{prefix}_metricas.csv", index=False)
            print(f"R² = {metrics['R2'].values[0]:.3f}  "
                  f"RMSE = {metrics['RMSE'].values[0]:.3f}")

            # Observed vs. predicted CSV
            obs_pred = pd.DataFrame({
                "Observado": y_test.values,
                "Predito"  : y_pred,
            })
            obs_pred.to_csv(f"{prefix}_obs_pred.csv", index=False)

            # Scatter plot
            scatter_path = os.path.join(figures_dir,
                                         f"scatter_{correction}_{model_name}.png")
            make_scatter_plot(
                y_test.values, y_pred,
                title     = f"{correction} — {model_name} ({year})",
                lims      = plot_lims,
                save_path = scatter_path,
            )


def main():
    for year in [2020, 2022]:
        print(f"\n=== Season {year} ===")
        train_season(year)

    print(f"\nTraining complete. Results saved to: {PATH_RESULTS}")


if __name__ == "__main__":
    main()
