"""
Corn Yield Prediction with Machine Learning and Remote Sensing
Script 04 — Results Visualization (R² and RMSE by Model and Correction)

Description:
    Reads the metrics CSVs produced by 02_model_training.py and generates
    publication-quality grouped bar + line charts showing R² (bars) and
    RMSE (lines) for each ML algorithm × atmospheric correction combination,
    for both training (CV) and validation (test set) results.

Input:
    - results/{season}/{correction}_{model}_metricas.csv
    - results/{season}/{correction}_{model}_resultados.csv

Output:
    - results/figures/metrics_validation_{season}.png
    - results/figures/metrics_training_{season}.png

Authors: Octávio Pereira da Costa · Thiago Orlando Costa Barboza
Institution: UFLA — Universidade Federal de Lavras
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# =============================================================================
# 0. CONFIGURATION
# =============================================================================

PATH_RESULTS = "results"
PATH_FIGURES = os.path.join(PATH_RESULTS, "figures")

# Display order
CORRECTION_ORDER = ["Sen2Cor", "iCOR", "DOS", "L1C"]
MODEL_ORDER      = ["RF", "SVM", "KNN"]

# Grayscale fills for R² bars
BAR_COLORS  = {"Sen2Cor": "#CCCCCC", "iCOR": "#555555", "DOS": "#888888", "L1C": "#AAAAAA"}

# Saturated colors for RMSE lines
LINE_COLORS = {"Sen2Cor": "#2E7D32", "iCOR": "#6A1B9A", "DOS": "#1565C0", "L1C": "#C62828"}

BAR_WIDTH  = 0.18   # width of each bar within a group
BAR_GAP    = 0.22   # spacing between bar groups (models)

# =============================================================================
# 1. DATA LOADERS
# =============================================================================

def load_metrics(year: int) -> pd.DataFrame | None:
    """Load all *_metricas.csv files for a given season into one DataFrame."""
    results_dir = os.path.join(PATH_RESULTS, str(year))
    files = [f for f in os.listdir(results_dir)
             if f.endswith("_metricas.csv")]

    if not files:
        warnings.warn(f"No metrics files found for {year}. "
                       "Run 02_model_training.py first.")
        return None

    frames = []
    for f in files:
        df = pd.read_csv(os.path.join(results_dir, f))
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    # Normalize labels
    df["Modelo"]   = df["Modelo"].str.upper()
    df["Correcao"] = df["Correcao"].str.replace("L2A", "Sen2Cor", regex=False)

    # Keep only expected levels
    df = df[df["Correcao"].isin(CORRECTION_ORDER) &
            df["Modelo"].isin(MODEL_ORDER)]

    return df


def load_training_summary(year: int) -> pd.DataFrame | None:
    """
    Load all *_resultados.csv files (CV folds) and compute per-combination
    mean ± range for R² and RMSE.
    """
    results_dir = os.path.join(PATH_RESULTS, str(year))
    files = [f for f in os.listdir(results_dir)
             if f.endswith("_resultados.csv")]

    if not files:
        return None

    rows = []
    for f in files:
        # Parse correction and model from filename: {correction}_{model}_resultados.csv
        base  = f.replace("_resultados.csv", "")
        parts = base.split("_")
        correction = parts[0]
        model      = parts[1].upper() if len(parts) > 1 else "UNKNOWN"

        df = pd.read_csv(os.path.join(results_dir, f))
        correction = correction.replace("L2A", "Sen2Cor")

        rows.append({
            "Correcao"  : correction,
            "Modelo"    : model,
            "R2_mean"   : df["Rsquared"].mean(),
            "R2_min"    : df["Rsquared"].min(),
            "R2_max"    : df["Rsquared"].max(),
            "RMSE_mean" : df["RMSE"].mean(),
            "RMSE_min"  : df["RMSE"].min(),
            "RMSE_max"  : df["RMSE"].max(),
        })

    df = pd.DataFrame(rows)
    df["R2_range"]   = df["R2_max"]   - df["R2_min"]
    df["RMSE_range"] = df["RMSE_max"] - df["RMSE_min"]

    df = df[df["Correcao"].isin(CORRECTION_ORDER) &
            df["Modelo"].isin(MODEL_ORDER)]
    return df


# =============================================================================
# 2. PLOTTING FUNCTIONS
# =============================================================================

def _model_x_positions(n_corrections: int = 4) -> dict:
    """
    Return x-axis positions for each (model, correction) combination.
    Models are spaced by BAR_GAP; corrections are offset within each group.
    """
    positions = {}
    offsets = np.linspace(-(n_corrections - 1) / 2,
                           (n_corrections - 1) / 2,
                           n_corrections) * BAR_WIDTH * 1.15

    for i, model in enumerate(MODEL_ORDER):
        center = i * BAR_GAP
        for j, corr in enumerate(CORRECTION_ORDER):
            positions[(model, corr)] = center + offsets[j]

    return positions


def plot_metrics_validation(df: pd.DataFrame, title: str, save_path: str) -> None:
    """Bar (R²) + line (RMSE) chart for test-set validation metrics."""
    pos = _model_x_positions()

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    for corr in CORRECTION_ORDER:
        sub = df[df["Correcao"] == corr].set_index("Modelo")

        x_bars = [pos[(m, corr)] for m in MODEL_ORDER if m in sub.index]
        r2_vals = [sub.loc[m, "R2"] for m in MODEL_ORDER if m in sub.index]
        rmse_vals = [sub.loc[m, "RMSE"] for m in MODEL_ORDER if m in sub.index]
        x_line = [pos[(m, corr)] for m in MODEL_ORDER if m in sub.index]

        ax1.bar(x_bars, r2_vals, width=BAR_WIDTH * 0.95,
                color=BAR_COLORS[corr], edgecolor="black",
                linewidth=0.6, label=corr, zorder=2)

        for xb, rv in zip(x_bars, r2_vals):
            ax1.text(xb, rv + 0.01, f"{rv:.2f}",
                     ha="center", va="bottom", fontsize=7)

        ax2.plot(x_line, rmse_vals, color=LINE_COLORS[corr],
                 marker="o", markersize=5, linewidth=1.5,
                 label=corr, zorder=3)

    model_centers = [i * BAR_GAP for i in range(len(MODEL_ORDER))]
    ax1.set_xticks(model_centers)
    ax1.set_xticklabels(MODEL_ORDER, fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("R²", fontsize=11)
    ax2.set_ylabel("RMSE", fontsize=11)
    ax1.set_title(title, fontsize=12, fontweight="bold")
    ax1.set_xlabel("Algorithm", fontsize=11)

    # Combined legend
    bars_leg   = ax1.get_legend_handles_labels()
    lines_leg  = ax2.get_legend_handles_labels()
    unique_bars  = dict(zip(bars_leg[1],  bars_leg[0]))
    unique_lines = dict(zip(lines_leg[1], lines_leg[0]))

    ax1.legend(list(unique_bars.values()),
               [f"{k} (R²)" for k in unique_bars.keys()],
               loc="upper left", fontsize=8, framealpha=0.8)
    ax2.legend(list(unique_lines.values()),
               [f"{k} (RMSE)" for k in unique_lines.keys()],
               loc="upper right", fontsize=8, framealpha=0.8)

    ax1.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_metrics_training(df: pd.DataFrame, title: str, save_path: str) -> None:
    """Bar (R² mean ± range) + line (RMSE mean ± range) chart for CV training."""
    pos = _model_x_positions()

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    for corr in CORRECTION_ORDER:
        sub = df[df["Correcao"] == corr].set_index("Modelo")

        x_bars  = [pos[(m, corr)] for m in MODEL_ORDER if m in sub.index]
        r2_mean  = [sub.loc[m, "R2_mean"]   for m in MODEL_ORDER if m in sub.index]
        r2_range = [sub.loc[m, "R2_range"]  for m in MODEL_ORDER if m in sub.index]
        rm_mean  = [sub.loc[m, "RMSE_mean"] for m in MODEL_ORDER if m in sub.index]
        rm_range = [sub.loc[m, "RMSE_range"] for m in MODEL_ORDER if m in sub.index]
        x_line   = [pos[(m, corr)] for m in MODEL_ORDER if m in sub.index]

        ax1.bar(x_bars, r2_mean, width=BAR_WIDTH * 0.95,
                color=BAR_COLORS[corr], edgecolor="black",
                linewidth=0.6, label=corr, zorder=2)

        ax1.errorbar(x_bars,
                     [r + rng for r, rng in zip(r2_mean, r2_range)],
                     yerr=r2_range,
                     fmt="none", ecolor="black", capsize=3,
                     elinewidth=0.8, zorder=4)

        for xb, rv in zip(x_bars, r2_mean):
            ax1.text(xb, rv * 0.5, f"{rv:.2f}",
                     ha="center", va="center", fontsize=6.5,
                     color="white", fontweight="bold")

        ax2.plot(x_line, rm_mean, color=LINE_COLORS[corr],
                 marker="o", markersize=5, linewidth=1.5,
                 label=corr, zorder=3)
        ax2.errorbar(x_line, rm_mean, yerr=rm_range,
                     fmt="none", ecolor=LINE_COLORS[corr],
                     capsize=3, elinewidth=0.8, zorder=3)

    model_centers = [i * BAR_GAP for i in range(len(MODEL_ORDER))]
    ax1.set_xticks(model_centers)
    ax1.set_xticklabels(MODEL_ORDER, fontsize=11)
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("R² (mean ± range)", fontsize=11)
    ax2.set_ylabel("RMSE (mean ± range)", fontsize=11)
    ax1.set_title(title, fontsize=12, fontweight="bold")
    ax1.set_xlabel("Algorithm", fontsize=11)

    bars_leg  = ax1.get_legend_handles_labels()
    lines_leg = ax2.get_legend_handles_labels()
    unique_bars  = dict(zip(bars_leg[1],  bars_leg[0]))
    unique_lines = dict(zip(lines_leg[1], lines_leg[0]))

    ax1.legend(list(unique_bars.values()),
               [f"{k} (R²)" for k in unique_bars.keys()],
               loc="upper left", fontsize=8, framealpha=0.8)
    ax2.legend(list(unique_lines.values()),
               [f"{k} (RMSE)" for k in unique_lines.keys()],
               loc="upper right", fontsize=8, framealpha=0.8)

    ax1.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# 3. MAIN
# =============================================================================

def main():
    os.makedirs(PATH_FIGURES, exist_ok=True)

    for year in [2020, 2022]:
        season_label = "Safrinha (2020)" if year == 2020 else "Safra (2022)"
        print(f"\n=== Generating figures for {year} ===")

        # Validation chart
        val_df = load_metrics(year)
        if val_df is not None:
            plot_metrics_validation(
                val_df,
                title     = f"R² and RMSE — Validation | {season_label}",
                save_path = os.path.join(PATH_FIGURES,
                                          f"metrics_validation_{year}.png"),
            )

        # Training (CV) chart
        train_df = load_training_summary(year)
        if train_df is not None:
            plot_metrics_training(
                train_df,
                title     = f"R² and RMSE — Training CV | {season_label}",
                save_path = os.path.join(PATH_FIGURES,
                                          f"metrics_training_{year}.png"),
            )

    print(f"\nDone! Figures saved to: {PATH_FIGURES}")


if __name__ == "__main__":
    main()
