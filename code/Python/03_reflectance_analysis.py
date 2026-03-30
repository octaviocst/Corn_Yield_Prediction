"""
Corn Yield Prediction with Machine Learning and Remote Sensing
Script 03 — Reflectance Analysis by Atmospheric Correction Method

Description:
    Loads per-plot mean reflectance values extracted in QGIS for each
    Sentinel-2 band (B02–B08), aggregates by plot, and produces multi-panel
    LOESS time series plots comparing the four atmospheric correction methods
    across five phenological stages.

Input:
    - data/reflectance/{season}/{correction}_{date}.csv
    - data/field/Reamostragem.csv  — plot resampling/aggregation key

Output:
    - results/figures/reflectance_timeseries_{season}.png

Authors: Octávio Pereira da Costa · Thiago Orlando Costa Barboza
Institution: UFLA — Universidade Federal de Lavras
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import make_interp_spline

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# =============================================================================
# 0. CONFIGURATION
# =============================================================================

PATH_REFLECTANCE = "data/reflectance"
PATH_FIELD       = "data/field"
PATH_FIGURES     = "results/figures"

CORRECTIONS = ["DOS", "iCOR", "L2A", "L1C"]

# Display labels (L2A → Sen2Cor, consistent with the paper)
CORRECTION_LABELS = {"DOS": "DOS", "iCOR": "iCOR", "L2A": "Sen2Cor", "L1C": "L1C"}

# Raw band column names as extracted by QGIS zonal statistics
BANDS_RAW    = ["X_B02mean", "X_B03mean", "X_B04mean", "X_B06mean", "X_B08mean"]
BAND_LABELS  = ["B02 (Blue)", "B03 (Green)", "B04 (Red)",
                "B06 (RedEdge)", "B08 (NIR)"]

# Sen2Cor (L2A) and L1C store values ×10 000
NORMALIZE_CORRECTIONS = {"L2A", "L1C"}

# Line colors per correction method
COLORS = {"DOS": "#1565C0", "iCOR": "#C62828", "Sen2Cor": "#2E7D32", "L1C": "#6A1B9A"}

DATES_2020 = {
    30: "27-03-20",
    45: "26-04-20",
    60: "06-05-20",
    75: "26-05-20",
    90: "25-06-20",
}

# =============================================================================
# 1. HELPER FUNCTIONS
# =============================================================================

def load_reflectance_aggregated(path: str,
                                 reamostragem: pd.DataFrame,
                                 days: int,
                                 correction: str) -> pd.DataFrame | None:
    """
    Load one reflectance CSV, merge with resampling key, aggregate by plot,
    optionally normalize, and attach phenological stage and correction labels.
    """
    if not os.path.exists(path):
        warnings.warn(f"File not found: {path}")
        return None

    df = pd.read_csv(path)
    df = df.merge(reamostragem, on="ID", how="inner")

    # Aggregate to plot level (mean of all pixels within each plot)
    df = df.groupby("id_2", as_index=False).mean(numeric_only=True)

    # Normalize ×10000 products
    if correction in NORMALIZE_CORRECTIONS:
        band_cols = [c for c in BANDS_RAW if c in df.columns]
        df[band_cols] = df[band_cols] / 10_000

    df["Dias"]     = days
    df["Correcao"] = CORRECTION_LABELS[correction]
    return df


def build_reflectance_df(year: int, dates: dict) -> pd.DataFrame:
    """Load and combine all corrections × dates for one season."""
    reamostragem = pd.read_csv(os.path.join(PATH_FIELD, "Reamostragem.csv"))
    frames = []

    for correction in CORRECTIONS:
        for days, date_str in dates.items():
            filename = f"{correction}_{date_str}.csv"
            path     = os.path.join(PATH_REFLECTANCE, str(year), filename)
            df       = load_reflectance_aggregated(path, reamostragem,
                                                   days, correction)
            if df is not None:
                frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def smooth_loess_line(x: np.ndarray, y: np.ndarray,
                      n_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """
    Approximate a smooth line through the mean values per x using cubic spline.
    (scipy does not include LOESS natively; this gives a visually equivalent result.)
    """
    df_agg = pd.DataFrame({"x": x, "y": y}).groupby("x")["y"].mean().reset_index()
    x_mean, y_mean = df_agg["x"].values, df_agg["y"].values

    if len(x_mean) < 4:
        return x_mean, y_mean

    x_smooth = np.linspace(x_mean.min(), x_mean.max(), n_points)
    spline   = make_interp_spline(x_mean, y_mean, k=3)
    y_smooth = np.clip(spline(x_smooth), 0, 1)
    return x_smooth, y_smooth


# =============================================================================
# 2. PLOTTING FUNCTION
# =============================================================================

def make_reflectance_figure(year: int, dates: dict) -> None:
    """Build and save a 5-panel reflectance × time figure for one season."""
    print(f"Building reflectance figure for {year} …")
    df = build_reflectance_df(year, dates)

    if df.empty:
        warnings.warn(f"No reflectance data found for {year}.")
        return

    os.makedirs(PATH_FIGURES, exist_ok=True)

    fig = plt.figure(figsize=(11, 14))
    fig.suptitle(
        f"Spectral reflectance by atmospheric correction — {year} (Safrinha)",
        fontsize=13, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
    axes = [fig.add_subplot(gs[r, c])
            for r, c in [(0,0),(0,1),(1,0),(1,1),(2,0)]]

    correction_order = ["DOS", "iCOR", "Sen2Cor", "L1C"]
    handles, labels  = [], []

    for ax, band_col, band_label in zip(axes, BANDS_RAW, BAND_LABELS):
        for correction in correction_order:
            sub = df[df["Correcao"] == correction]
            if sub.empty or band_col not in sub.columns:
                continue

            x = sub["Dias"].values
            y = sub[band_col].values

            # Scatter points
            ax.scatter(x, y, color=COLORS[correction],
                       s=18, alpha=0.5, zorder=3)

            # Smooth line
            x_s, y_s = smooth_loess_line(x, y)
            line, = ax.plot(x_s, y_s, color=COLORS[correction],
                            linewidth=1.8, label=correction, zorder=4)

            if correction not in labels:
                handles.append(line)
                labels.append(correction)

        ax.set_ylim(0, 1)
        ax.set_xlim(25, 95)
        ax.set_xticks(range(30, 91, 15))
        ax.set_xlabel("Days after planting", fontsize=9)
        ax.set_ylabel("Reflectance", fontsize=9)
        ax.set_title(band_label, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    # Shared legend on the empty 6th subplot slot
    ax_legend = fig.add_subplot(gs[2, 1])
    ax_legend.axis("off")
    ax_legend.legend(handles, labels, title="Correction method",
                     title_fontsize=10, fontsize=9,
                     loc="center", frameon=True, framealpha=0.9)

    fig.text(0.5, 0.01,
             "Data: Sentinel-2 | Corrections: DOS, iCOR, Sen2Cor (L2A), L1C",
             ha="center", fontsize=8, color="gray")

    out_path = os.path.join(PATH_FIGURES,
                             f"reflectance_timeseries_{year}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# =============================================================================
# 3. MAIN
# =============================================================================

def main():
    make_reflectance_figure(2020, DATES_2020)

    # To run for 2022, define DATES_2022 and call:
    # make_reflectance_figure(2022, DATES_2022)


if __name__ == "__main__":
    main()
