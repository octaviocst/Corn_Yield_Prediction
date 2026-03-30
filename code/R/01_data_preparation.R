# =============================================================================
# Corn Yield Prediction with Machine Learning and Remote Sensing
# Script 01 — Data Preparation & Vegetation Index Calculation
#
# Description:
#   Loads spectral reflectance data extracted in QGIS/SNAP from Sentinel-2
#   imagery (four atmospheric correction methods: DOS, iCOR, Sen2Cor, L1C),
#   joins field productivity data, computes vegetation indices, and prepares
#   the final datasets for model training.
#
# Seasons:
#   - 2020: Safrinha (off-season corn)
#   - 2022: Safra (main season corn)
#
# Input data:
#   - data/reflectance/{season}/{correction}_{date}.csv  — spectral bands per plot
#   - data/field/Prod_{year}.csv                         — kriging yield (sc/ha)
#   - data/field/Topo_{year}.csv                         — topographic variables
#
# Output:
#   - data/processed/{season}/dataset_{correction}.csv   — model-ready datasets
#
# Atmospheric corrections applied externally:
#   - DOS    : Dark Object Subtraction (QGIS Semi-Automatic Classification Plugin)
#   - iCOR   : iCOR plugin (SNAP)
#   - Sen2Cor: ESA Sen2Cor processor (SNAP)
#   - L1C    : No correction (Top-of-Atmosphere reflectance)
#
# Authors: Octávio Pereira da Costa · Thiago Orlando Costa Barboza
# Institution: UFLA — Universidade Federal de Lavras
# =============================================================================

library(dplyr)

# -----------------------------------------------------------------------------
# 0. CONFIGURATION
# Set your paths here before running
# -----------------------------------------------------------------------------

# Path to reflectance CSVs (subfolders: 2020/ and 2022/)
PATH_REFLECTANCE <- "data/reflectance"

# Path to field data (productivity and topography)
PATH_FIELD <- "data/field"

# Output path for processed datasets
PATH_OUTPUT <- "data/processed"
dir.create(file.path(PATH_OUTPUT, "2020"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(PATH_OUTPUT, "2022"), recursive = TRUE, showWarnings = FALSE)

# Atmospheric correction methods to process
CORRECTIONS <- c("DOS", "iCOR", "L2A", "L1C")

# Sentinel-2 bands used (B02=Blue, B03=Green, B04=Red, B06=RedEdge, B08=NIR)
BANDS <- c("B02", "B03", "B04", "B06", "B08")

# Acquisition dates per phenological stage (days after planting)
DATES_2020 <- list(
  "30" = "27-03-20",
  "45" = "26-04-20",
  "60" = "06-05-20",
  "75" = "26-05-20",
  "90" = "25-06-20"
)

DATES_2022 <- list(
  "30" = "17-11-21",
  "45" = "27-12-21",
  "60" = "21-01-22",
  "75" = "25-02-22",
  "90" = "07-03-22"
)

# -----------------------------------------------------------------------------
# 1. HELPER FUNCTIONS
# -----------------------------------------------------------------------------

#' Standardize band column names to B02, B03, B04, B06, B08
standardize_band_names <- function(dados) {
  for (banda in c("B02", "B03", "B04", "B06", "B08")) {
    col <- grep(banda, names(dados), ignore.case = TRUE, value = TRUE)
    if (length(col) > 0) names(dados)[names(dados) == col] <- banda
  }
  return(dados)
}

#' Calculate NDVI, NDRE, EVI and GNDVI from spectral bands
calculate_vegetation_indices <- function(dados) {
  dados %>% mutate(
    NDVI  = (B08 - B04) / (B08 + B04),
    NDRE  = (B08 - B06) / (B08 + B06),
    EVI   = 2.5 * (B08 - B04) / (B08 + 6 * B04 - 7.5 * B02 + 1),
    GNDVI = (B08 - B03) / (B08 + B03)
  )
}

#' Load and preprocess a single reflectance CSV
#' @param path     Path to the reflectance CSV file
#' @param days     Days after planting (numeric label)
#' @param prod     Productivity data frame (already loaded)
#' @param normalize Divide bands by 10000 (TRUE for L2A/L1C Sen2Cor products)
#' @param max_id   Optional: filter to IDs <= max_id (e.g. 4400 for 2022 season)
load_reflectance <- function(path, days, prod, normalize = FALSE, max_id = NULL) {
  dados <- read.csv(path)

  # Optional ID filter (applied in 2022 season to remove extra plots)
  if (!is.null(max_id)) dados <- dados %>% filter(ID <= max_id)

  # Remove unnecessary spatial columns
  remove_cols <- c("Lat", "Long", "latitude", "longitude", "X_B11mean")
  dados <- dados[, !(names(dados) %in% remove_cols)]

  # Standardize band names
  dados <- standardize_band_names(dados)

  # Normalize DN to reflectance (Sen2Cor and L1C products store values ×10000)
  if (normalize) {
    dados <- dados %>% mutate(across(all_of(BANDS), ~ . / 10000))
  }

  # Add phenological stage label
  dados$Dias <- days

  # Join with field productivity data
  dados <- inner_join(dados, prod, by = "ID")

  # Calculate vegetation indices
  dados <- calculate_vegetation_indices(dados)

  return(dados)
}

# -----------------------------------------------------------------------------
# 2. LOAD FIELD DATA
# -----------------------------------------------------------------------------

load_productivity <- function(year) {
  path <- file.path(PATH_FIELD, paste0("Prod_", year, ".csv"))
  prod <- read.csv(path) %>%
    select(-Lat, -Long, -X_Prod_IDWmean) %>%
    rename(Prod_13_KRIG = X_Prod_Krigmean)

  # 2022 season: limit to experimental area (IDs <= 4400)
  if (year == 2022) prod <- prod %>% filter(ID <= 4400)

  return(prod)
}

load_topography <- function(year) {
  path <- file.path(PATH_FIELD, paste0("Topo_", year, ".csv"))
  topo <- read.csv(path) %>%
    select(-Lat, -Long, -X_IPTmean, -X_IRTmean, -X_Rugosmean, -X_Sombrmean) %>%
    rename(Declividade = X_Declivmean)
  return(topo)
}

# -----------------------------------------------------------------------------
# 3. BUILD DATASETS BY SEASON AND CORRECTION METHOD
# -----------------------------------------------------------------------------

build_dataset <- function(year, correction, dates, prod) {
  # Sen2Cor (L2A) and L1C products require normalization (stored ×10000)
  normalize <- correction %in% c("L2A", "L1C")
  max_id    <- if (year == 2022) 4400 else NULL

  season_list <- lapply(names(dates), function(day) {
    filename <- paste0(correction, "_", dates[[day]], ".csv")
    path     <- file.path(PATH_REFLECTANCE, as.character(year), filename)

    if (!file.exists(path)) {
      warning(paste("File not found:", path))
      return(NULL)
    }

    load_reflectance(path, as.numeric(day), prod, normalize, max_id)
  })

  do.call(rbind, Filter(Negate(is.null), season_list))
}

# -----------------------------------------------------------------------------
# 4. PROCESS ALL SEASON × CORRECTION COMBINATIONS
# -----------------------------------------------------------------------------

seasons <- list(
  list(year = 2020, dates = DATES_2020),
  list(year = 2022, dates = DATES_2022)
)

for (season in seasons) {
  year  <- season$year
  dates <- season$dates
  prod  <- load_productivity(year)

  message("\n=== Processing season ", year, " ===")

  for (correction in CORRECTIONS) {
    message("  Correction: ", correction)

    dataset <- build_dataset(year, correction, dates, prod)

    if (!is.null(dataset) && nrow(dataset) > 0) {
      out_path <- file.path(PATH_OUTPUT, as.character(year),
                            paste0("dataset_", correction, ".csv"))
      write.csv(dataset, out_path, row.names = FALSE)
      message("  Saved: ", out_path, " (", nrow(dataset), " rows)")
    } else {
      message("  Skipped (no data)")
    }
  }
}

message("\nDone! Processed datasets saved to: ", PATH_OUTPUT)
