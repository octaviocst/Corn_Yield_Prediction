# =============================================================================
# Corn Yield Prediction with Machine Learning and Remote Sensing
# Script 03 — Reflectance Analysis by Atmospheric Correction Method
#
# Description:
#   Loads per-plot mean reflectance values extracted in QGIS for each
#   Sentinel-2 band (B02–B08), aggregates by plot, and produces multi-panel
#   LOESS time series plots comparing the four atmospheric correction methods
#   across five phenological stages.
#
# Input:
#   - data/reflectance/{season}/{correction}_{date}.csv
#   - data/field/Reamostragem.csv  — plot resampling/aggregation key
#
# Output:
#   - results/figures/reflectance_timeseries_{season}.png
#
# Authors: Octávio Pereira da Costa · Thiago Orlando Costa Barboza
# Institution: UFLA — Universidade Federal de Lavras
# =============================================================================

library(dplyr)
library(tidyr)
library(ggplot2)
library(patchwork)

# -----------------------------------------------------------------------------
# 0. CONFIGURATION
# -----------------------------------------------------------------------------

PATH_REFLECTANCE <- "data/reflectance"
PATH_FIELD       <- "data/field"
PATH_FIGURES     <- "results/figures"
dir.create(PATH_FIGURES, recursive = TRUE, showWarnings = FALSE)

CORRECTIONS <- c("DOS", "iCOR", "L2A", "L1C")
BANDS_PLOT  <- c("X_B02mean", "X_B03mean", "X_B04mean", "X_B06mean", "X_B08mean")
BAND_LABELS <- c("B02 (Blue)", "B03 (Green)", "B04 (Red)", "B06 (RedEdge)", "B08 (NIR)")

# Colors and correction display names
CORRECTION_COLORS <- c(DOS = "blue2", iCOR = "red2", Sen2Cor = "green3", L1C = "purple2")
CORRECTION_LABELS <- c(DOS = "DOS", iCOR = "iCOR", L2A = "Sen2Cor", L1C = "L1C")

DATES_2020 <- c("30" = "27-03-20", "45" = "26-04-20", "60" = "06-05-20",
                "75" = "26-05-20", "90" = "25-06-20")

# -----------------------------------------------------------------------------
# 1. HELPER FUNCTIONS
# -----------------------------------------------------------------------------

#' Load and aggregate reflectance CSV by plot (id_2 grouping)
load_reflectance_aggregated <- function(path, reamostragem, days, correction) {
  if (!file.exists(path)) {
    warning("File not found: ", path)
    return(NULL)
  }

  dados <- read.csv(path) %>%
    inner_join(reamostragem, by = "ID") %>%
    group_by(id_2) %>%
    summarise(across(everything(), mean, na.rm = TRUE), .groups = "drop")

  # Sen2Cor (L2A) and L1C: convert from ×10000 scale to reflectance (0–1)
  if (correction %in% c("L2A", "L1C")) {
    dados <- dados %>%
      mutate(across(starts_with("X_B0"), ~ . / 10000))
  }

  dados$Dias      <- days
  dados$Correcao  <- CORRECTION_LABELS[correction]
  return(dados)
}

#' Build combined reflectance data frame for all corrections and dates
build_reflectance_df <- function(year, dates, corrections) {
  reamostragem <- read.csv(file.path(PATH_FIELD, "Reamostragem.csv"))
  all_data <- list()

  for (correction in corrections) {
    for (day in names(dates)) {
      filename <- paste0(correction, "_", dates[[day]], ".csv")
      path     <- file.path(PATH_REFLECTANCE, as.character(year), filename)
      df       <- load_reflectance_aggregated(path, reamostragem, as.numeric(day), correction)
      if (!is.null(df)) all_data <- c(all_data, list(df))
    }
  }

  do.call(rbind, all_data)
}

# -----------------------------------------------------------------------------
# 2. PLOTTING FUNCTION
# -----------------------------------------------------------------------------

#' Create a single-band LOESS time series panel
plot_band <- function(data, band_col, band_label, show_legend = FALSE) {
  p <- ggplot(data, aes(x = Dias, y = .data[[band_col]],
                        color = Correcao, linetype = Correcao)) +
    geom_point(size = 2) +
    geom_smooth(method = "loess", se = FALSE, linewidth = 0.9) +
    scale_y_continuous(limits = c(0, 1)) +
    scale_x_continuous(limits = c(25, 95), breaks = seq(30, 90, by = 15)) +
    scale_color_manual(values = CORRECTION_COLORS, name = "Correction") +
    scale_linetype_manual(values = rep("solid", 4),
                          guide  = guide_legend(title = "Correction")) +
    labs(x = "Days after planting", y = "Reflectance", title = band_label) +
    theme_classic(base_size = 11) +
    theme(plot.margin = unit(c(0.2, 0.3, 0.2, 0.2), "cm"))

  if (!show_legend) p <- p + theme(legend.position = "none")

  return(p)
}

#' Build and save a multi-panel reflectance plot for one season
make_reflectance_figure <- function(year, dates) {
  message("Building reflectance figure for ", year)
  df <- build_reflectance_df(year, dates, CORRECTIONS)

  if (is.null(df) || nrow(df) == 0) {
    warning("No data for year ", year)
    return(invisible(NULL))
  }

  panels <- lapply(seq_along(BANDS_PLOT), function(i) {
    show_leg <- (i == length(BANDS_PLOT))  # legend only on last panel
    plot_band(df, BANDS_PLOT[i], BAND_LABELS[i], show_leg)
  })

  fig <- wrap_plots(panels, ncol = 2) +
    plot_annotation(
      title   = paste0("Spectral reflectance by atmospheric correction — ", year,
                       " (Safrinha season)"),
      caption = "Data: Sentinel-2 | Corrections: DOS, iCOR, Sen2Cor (L2A), L1C"
    )

  out_path <- file.path(PATH_FIGURES,
                        paste0("reflectance_timeseries_", year, ".png"))
  ggsave(out_path, plot = fig, width = 10, height = 12, dpi = 300)
  message("Saved: ", out_path)
}

# -----------------------------------------------------------------------------
# 3. RUN
# -----------------------------------------------------------------------------

make_reflectance_figure(2020, DATES_2020)

# To run for 2022, define DATES_2022 and call:
# make_reflectance_figure(2022, DATES_2022)
