"""
Corn Yield Prediction with Machine Learning and Remote Sensing
Script 01 — Data Preparation & Vegetation Index Calculation

Description:
    Loads spectral reflectance data extracted in QGIS/SNAP from Sentinel-2
    imagery (four atmospheric correction methods: DOS, iCOR, Sen2Cor, L1C),
    joins field productivity data, computes vegetation indices, and prepares
    the final datasets for model training.

Seasons:
    - 2020: Safrinha (off-season corn)
    - 2022: Safra (main season corn)

Input data:
    - data/reflectance/{season}/{correction}_{date}.csv  — spectral bands per plot
    - data/field/Prod_{year}.csv                         — kriging yield (sc/ha)
    - data/field/Topo_{year}.csv                         — topographic variables

Output:
    - data/processed/{season}/dataset_{correction}.csv   — model-ready datasets

Atmospheric corrections applied externally:
    - DOS    : Dark Object Subtraction (QGIS Semi-Automatic Classification Plugin)
    - iCOR   : iCOR plugin (SNAP)
    - Sen2Cor: ESA Sen2Cor processor (SNAP)
    - L1C    : No correction (Top-of-Atmosphere reflectance)

Authors: Octávio Pereira da Costa · Thiago Orlando Costa Barboza
Institution: UFLA — Universidade Federal de Lavras
"""

import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# 0. CONFIGURATION
# =============================================================================

PATH_REFLECTANCE = "data/reflectance"
PATH_FIELD       = "data/field"
PATH_OUTPUT      = "data/processed"

CORRECTIONS = ["DOS", "iCOR", "L2A", "L1C"]

# Bands to keep (Sentinel-2: Blue, Green, Red, RedEdge, NIR)
BANDS = ["B02", "B03", "B04", "B06", "B08"]

# Columns to drop from raw reflectance files
DROP_COLS = ["Lat", "Long", "latitude", "longitude", "X_B11mean"]

# Sen2Cor (L2A) and L1C products are stored ×10000 → normalize to [0, 1]
NORMALIZE_CORRECTIONS = {"L2A", "L1C"}

# Acquisition dates per phenological stage (days after planting)
DATES_2020 = {
    30: "27-03-20",
    45: "26-04-20",
    60: "06-05-20",
    75: "26-05-20",
    90: "25-06-20",
}

DATES_2022 = {
    30: "17-11-21",
    45: "27-12-21",
    60: "21-01-22",
    75: "25-02-22",
    90: "07-03-22",
}

# =============================================================================
# 1. HELPER FUNCTIONS
# =============================================================================

def standardize_band_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw band columns (e.g. X_B02mean) to standard names (B02…B08)."""
    rename_map = {}
    for band in BANDS:
        matches = [c for c in df.columns if band.lower() in c.lower()]
        if matches:
            rename_map[matches[0]] = band
    return df.rename(columns=rename_map)


def calculate_vegetation_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Add NDVI, NDRE, EVI and GNDVI columns."""
    df = df.copy()
    df["NDVI"]  = (df["B08"] - df["B04"]) / (df["B08"] + df["B04"])
    df["NDRE"]  = (df["B08"] - df["B06"]) / (df["B08"] + df["B06"])
    df["EVI"]   = 2.5 * (df["B08"] - df["B04"]) / (
                      df["B08"] + 6 * df["B04"] - 7.5 * df["B02"] + 1)
    df["GNDVI"] = (df["B08"] - df["B03"]) / (df["B08"] + df["B03"])
    return df


def load_reflectance(path: str, days: int, prod: pd.DataFrame,
                     normalize: bool = False,
                     max_id: int | None = None) -> pd.DataFrame | None:
    """
    Load a single reflectance CSV and merge with productivity data.

    Parameters
    ----------
    path      : Path to the reflectance CSV file.
    days      : Days after planting (phenological stage label).
    prod      : Productivity DataFrame (already preprocessed).
    normalize : If True, divide band values by 10 000 (for L2A / L1C products).
    max_id    : Optional upper bound for plot IDs (used in 2022 season).
    """
    if not os.path.exists(path):
        warnings.warn(f"File not found: {path}")
        return None

    df = pd.read_csv(path)

    # Optional ID filter (2022 season limits experimental area to ID ≤ 4400)
    if max_id is not None:
        df = df[df["ID"] <= max_id]

    # Drop unnecessary spatial columns
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # Standardize band column names
    df = standardize_band_names(df)

    # Normalize DN → reflectance (Sen2Cor and L1C store values ×10 000)
    if normalize:
        for band in BANDS:
            if band in df.columns:
                df[band] = df[band] / 10_000

    # Add phenological stage label
    df["Dias"] = days

    # Merge with field productivity data
    df = df.merge(prod, on="ID", how="inner")

    # Compute vegetation indices
    df = calculate_vegetation_indices(df)

    return df


# =============================================================================
# 2. FIELD DATA LOADERS
# =============================================================================

def load_productivity(year: int, max_id: int | None = None) -> pd.DataFrame:
    """Load and preprocess the productivity CSV for a given season."""
    path = os.path.join(PATH_FIELD, f"Prod_{year}.csv")
    prod = pd.read_csv(path)

    drop = [c for c in ["Lat", "Long", "X_Prod_IDWmean"] if c in prod.columns]
    prod = prod.drop(columns=drop)
    prod = prod.rename(columns={"X_Prod_Krigmean": "Prod_13_KRIG"})

    if max_id is not None:
        prod = prod[prod["ID"] <= max_id]

    return prod


def load_topography(year: int) -> pd.DataFrame:
    """Load and preprocess the topography CSV for a given season."""
    path = os.path.join(PATH_FIELD, f"Topo_{year}.csv")
    topo = pd.read_csv(path)

    drop = [c for c in ["Lat", "Long", "X_IPTmean", "X_IRTmean",
                         "X_Rugosmean", "X_Sombrmean"] if c in topo.columns]
    topo = topo.drop(columns=drop)
    topo = topo.rename(columns={"X_Declivmean": "Declividade"})

    return topo


# =============================================================================
# 3. BUILD DATASET PER SEASON × CORRECTION
# =============================================================================

def build_dataset(year: int, correction: str,
                  dates: dict, prod: pd.DataFrame) -> pd.DataFrame | None:
    """Concatenate all phenological stages for one season/correction pair."""
    normalize = correction in NORMALIZE_CORRECTIONS
    max_id    = 4400 if year == 2022 else None

    frames = []
    for days, date_str in dates.items():
        filename = f"{correction}_{date_str}.csv"
        path     = os.path.join(PATH_REFLECTANCE, str(year), filename)
        df       = load_reflectance(path, days, prod, normalize, max_id)
        if df is not None:
            frames.append(df)

    if not frames:
        return None

    return pd.concat(frames, ignore_index=True)


# =============================================================================
# 4. MAIN
# =============================================================================

def main():
    seasons = [
        {"year": 2020, "dates": DATES_2020},
        {"year": 2022, "dates": DATES_2022},
    ]

    for season in seasons:
        year  = season["year"]
        dates = season["dates"]
        max_id = 4400 if year == 2022 else None

        prod = load_productivity(year, max_id)
        out_dir = os.path.join(PATH_OUTPUT, str(year))
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n=== Processing season {year} ===")

        for correction in CORRECTIONS:
            print(f"  Correction: {correction}")
            dataset = build_dataset(year, correction, dates, prod)

            if dataset is not None and len(dataset) > 0:
                out_path = os.path.join(out_dir, f"dataset_{correction}.csv")
                dataset.to_csv(out_path, index=False)
                print(f"  Saved: {out_path}  ({len(dataset)} rows)")
            else:
                print(f"  Skipped (no data found)")

    print(f"\nDone! Processed datasets saved to: {PATH_OUTPUT}")


if __name__ == "__main__":
    main()
