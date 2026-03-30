# =============================================================================
# Corn Yield Prediction with Machine Learning and Remote Sensing
# Script 04 — Results Visualization (R² and RMSE by Model and Correction)
#
# Description:
#   Reads the metrics CSVs produced by 02_model_training.R and generates
#   publication-quality bar + line charts showing R² (bars) and RMSE (lines)
#   for each ML algorithm × atmospheric correction combination, for both
#   training (CV) and validation (test set) results.
#
# Input:
#   - results/{season}/{correction}_{model}_metricas.csv
#   - results/{season}/{correction}_{model}_resultados.csv
#
# Output:
#   - results/figures/metrics_validation_{season}.png
#   - results/figures/metrics_training_{season}.png
#
# Authors: Octávio Pereira da Costa · Thiago Orlando Costa Barboza
# Institution: UFLA — Universidade Federal de Lavras
# =============================================================================

library(dplyr)
library(ggplot2)
library(scales)

# -----------------------------------------------------------------------------
# 0. CONFIGURATION
# -----------------------------------------------------------------------------

PATH_RESULTS <- "results"
PATH_FIGURES <- file.path(PATH_RESULTS, "figures")
dir.create(PATH_FIGURES, recursive = TRUE, showWarnings = FALSE)

# Factor level order for consistent plotting
CORRECTION_LEVELS <- c("Sen2Cor", "iCOR", "DOS", "L1C")
MODEL_LEVELS      <- c("RF", "SVM", "KNN")

# Grayscale fills for bars (R²) and colors for lines (RMSE)
FILL_COLORS  <- c(Sen2Cor = "gray80", iCOR = "gray30", DOS = "gray45", L1C = "gray60")
LINE_COLORS  <- c(Sen2Cor = "green3", iCOR = "purple2", DOS = "blue2",  L1C = "red2")

# -----------------------------------------------------------------------------
# 1. HELPER FUNCTIONS
# -----------------------------------------------------------------------------

#' Load all *_metricas.csv files from a season results folder
load_metrics <- function(year) {
  results_dir <- file.path(PATH_RESULTS, as.character(year))
  files <- list.files(results_dir, pattern = "_metricas\\.csv$", full.names = TRUE)

  if (length(files) == 0) {
    warning("No metrics files found for year ", year,
            ". Run 02_model_training.R first.")
    return(NULL)
  }

  df <- do.call(rbind, lapply(files, read.csv)) %>%
    mutate(
      Modelo   = toupper(Modelo),
      Correcao = gsub("L2A", "Sen2Cor", Correcao),
      Correcao = factor(Correcao, levels = CORRECTION_LEVELS),
      Modelo   = factor(Modelo,   levels = MODEL_LEVELS)
    )
  return(df)
}

#' Load and summarise *_resultados.csv files (CV training performance)
load_training_summary <- function(year) {
  results_dir <- file.path(PATH_RESULTS, as.character(year))
  files <- list.files(results_dir, pattern = "_resultados\\.csv$", full.names = TRUE)

  if (length(files) == 0) return(NULL)

  all_stats <- lapply(files, function(f) {
    # Extract correction and model from filename
    base       <- tools::file_path_sans_ext(basename(f))
    parts      <- strsplit(base, "_")[[1]]
    correction <- parts[1]
    model      <- parts[2]

    df <- read.csv(f)
    df %>%
      summarise(
        R2_mean   = mean(Rsquared, na.rm = TRUE),
        R2_min    = min(Rsquared,  na.rm = TRUE),
        R2_max    = max(Rsquared,  na.rm = TRUE),
        RMSE_mean = mean(RMSE,     na.rm = TRUE),
        RMSE_min  = min(RMSE,      na.rm = TRUE),
        RMSE_max  = max(RMSE,      na.rm = TRUE)
      ) %>%
      mutate(
        Correcao  = gsub("L2A", "Sen2Cor", correction),
        Modelo    = toupper(model),
        R2_range  = R2_max  - R2_min,
        RMSE_range = RMSE_max - RMSE_min
      )
  })

  df <- do.call(rbind, all_stats) %>%
    mutate(
      Correcao = factor(Correcao, levels = CORRECTION_LEVELS),
      Modelo   = factor(Modelo,   levels = MODEL_LEVELS)
    )
  return(df)
}

# -----------------------------------------------------------------------------
# 2. PLOT FUNCTIONS
# -----------------------------------------------------------------------------

#' Bar (R²) + line (RMSE) chart — validation (test set)
plot_metrics_validation <- function(df, title) {
  ggplot(df, aes(x = Modelo, y = R2, fill = Correcao)) +
    geom_bar(stat = "identity",
             position = position_dodge(width = 0.8),
             color = "black", width = 0.7) +
    geom_text(aes(label = round(R2, 2)),
              position = position_dodge(width = 0.8),
              vjust = -0.5, size = 2.8) +
    geom_line(aes(x     = as.numeric(Modelo) + (as.numeric(Correcao) - 2.5) * 0.2,
                  y     = RMSE,
                  group = Correcao,
                  color = Correcao),
              linewidth = 1) +
    geom_point(aes(x     = as.numeric(Modelo) + (as.numeric(Correcao) - 2.5) * 0.2,
                   y     = RMSE,
                   color = Correcao),
               size = 2.5) +
    scale_fill_manual(values = FILL_COLORS,  name = "Correction (R²)") +
    scale_color_manual(values = LINE_COLORS, name = "Correction (RMSE)") +
    scale_y_continuous(
      name     = "R²",
      limits   = c(0, 1),
      sec.axis = sec_axis(~ ., name = "RMSE",
                          labels = label_number(accuracy = 0.01))
    ) +
    labs(title = title, x = "Algorithm") +
    theme_classic(base_size = 12)
}

#' Bar (R²) + line (RMSE) chart — training (CV mean ± range)
plot_metrics_training <- function(df, title) {
  ggplot(df, aes(x = Modelo, y = R2_mean, fill = Correcao)) +
    geom_bar(stat = "identity",
             position = position_dodge(width = 0.8),
             color = "black", width = 0.7) +
    geom_errorbar(aes(ymin = R2_mean - R2_range, ymax = R2_mean + R2_range),
                  position = position_dodge(width = 0.8),
                  width = 0.2, color = "black") +
    geom_text(aes(label = round(R2_mean, 2)),
              position = position_dodge(width = 0.8),
              vjust = 10, size = 2.8) +
    geom_line(aes(x     = as.numeric(Modelo) + (as.numeric(Correcao) - 2.5) * 0.2,
                  y     = RMSE_mean,
                  group = Correcao,
                  color = Correcao),
              linewidth = 1) +
    geom_point(aes(x     = as.numeric(Modelo) + (as.numeric(Correcao) - 2.5) * 0.2,
                   y     = RMSE_mean,
                   color = Correcao),
               size = 2.5) +
    geom_errorbar(
      aes(x    = as.numeric(Modelo) + (as.numeric(Correcao) - 2.5) * 0.2,
          ymin = RMSE_mean - RMSE_range,
          ymax = RMSE_mean + RMSE_range,
          color = Correcao),
      width = 0.15
    ) +
    scale_fill_manual(values = FILL_COLORS,  name = "Correction (R²)") +
    scale_color_manual(values = LINE_COLORS, name = "Correction (RMSE)") +
    scale_y_continuous(
      name     = "R²",
      limits   = c(0, 1),
      sec.axis = sec_axis(~ ., name = "RMSE",
                          labels = label_number(accuracy = 0.01))
    ) +
    labs(title = title, x = "Algorithm") +
    theme_classic(base_size = 12)
}

# -----------------------------------------------------------------------------
# 3. RUN
# -----------------------------------------------------------------------------

for (year in c(2020, 2022)) {
  season_label <- if (year == 2020) "Safrinha (2020)" else "Safra (2022)"
  message("\n=== Generating figures for ", year, " ===")

  # Validation chart
  val_df <- load_metrics(year)
  if (!is.null(val_df)) {
    p_val <- plot_metrics_validation(val_df,
                                     paste("R² and RMSE — Validation |", season_label))
    ggsave(file.path(PATH_FIGURES, paste0("metrics_validation_", year, ".png")),
           plot = p_val, width = 9, height = 6, dpi = 300)
    message("  Saved: metrics_validation_", year, ".png")
  }

  # Training (CV) chart
  train_df <- load_training_summary(year)
  if (!is.null(train_df)) {
    p_train <- plot_metrics_training(train_df,
                                     paste("R² and RMSE — Training CV |", season_label))
    ggsave(file.path(PATH_FIGURES, paste0("metrics_training_", year, ".png")),
           plot = p_train, width = 9, height = 6, dpi = 300)
    message("  Saved: metrics_training_", year, ".png")
  }
}

message("\nDone! Figures saved to: ", PATH_FIGURES)
