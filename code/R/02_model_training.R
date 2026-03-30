# =============================================================================
# Corn Yield Prediction with Machine Learning and Remote Sensing
# Script 02 — Model Training & Evaluation (R — caret)
#
# Description:
#   Trains and evaluates three ML algorithms (RF, SVM, kNN) for each
#   atmospheric correction method and season using repeated k-fold
#   cross-validation. Saves per-model metrics and observed vs. predicted
#   values for downstream visualization.
#
# Input:
#   - data/processed/{season}/dataset_{correction}.csv  (from 01_data_preparation.R)
#
# Output (per season/correction/model):
#   - results/{season}/{correction}_{model}_metricas.csv     — R², RMSE, MAE, MSE
#   - results/{season}/{correction}_{model}_resultados.csv   — obs vs. pred (CV)
#   - results/{season}/{correction}_{model}_obs_pred.csv     — obs vs. pred (test set)
#   - results/figures/{season}/scatter_{correction}_{model}.png
#
# Note on model translation:
#   The models in this study correspond to the paper's notation as:
#     RF  → method = "ranger"    (Random Forest via ranger)
#     SVM → method = "svmRadial" (Support Vector Machine, RBF kernel)
#     kNN → method = "knn"       (k-Nearest Neighbors)
#
# Authors: Octávio Pereira da Costa · Thiago Orlando Costa Barboza
# Institution: UFLA — Universidade Federal de Lavras
# =============================================================================

library(dplyr)
library(caret)
library(ggplot2)

# -----------------------------------------------------------------------------
# 0. CONFIGURATION
# -----------------------------------------------------------------------------

PATH_PROCESSED <- "data/processed"
PATH_RESULTS   <- "results"

# Algorithms to train (caret method strings)
MODELS <- list(
  RF  = "ranger",
  SVM = "svmRadial",
  KNN = "knn"
)

# Cross-validation settings
CV_FOLDS   <- 10
CV_REPEATS <- 10
TRAIN_SPLIT <- 0.80  # 80% train / 20% test

set.seed(123)

# -----------------------------------------------------------------------------
# 1. HELPER FUNCTIONS
# -----------------------------------------------------------------------------

#' Evaluate a trained model on a test set
#' @return data.frame with MAE, MSE, RMSE, R²
evaluate_model <- function(model, test_data, model_name, correction) {
  preds <- predict(model, test_data)
  data.frame(
    Modelo   = model_name,
    Correcao = correction,
    MAE      = mean(abs(preds - test_data$Prod_13_KRIG)),
    MSE      = mean((preds - test_data$Prod_13_KRIG)^2),
    RMSE     = sqrt(mean((preds - test_data$Prod_13_KRIG)^2)),
    R2       = cor(preds, test_data$Prod_13_KRIG)^2
  )
}

#' Build observed vs. predicted scatter plot
make_scatter_plot <- function(obs, pred, title, xlims = NULL, ylims = NULL) {
  r2 <- round(cor(pred, obs)^2, 2)
  df <- data.frame(Observado = obs, Predito = pred)

  lims <- if (!is.null(xlims)) xlims else range(c(obs, pred))

  ggplot(df, aes(x = Observado, y = Predito)) +
    geom_point(alpha = 0.6, size = 1.5) +
    geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
    annotate("text", x = lims[1] + diff(lims) * 0.05,
             y = lims[2] - diff(lims) * 0.05,
             label = paste0("R² = ", r2), size = 4.5, color = "blue", hjust = 0) +
    labs(title = title, x = "Observed (sc/ha)", y = "Predicted (sc/ha)") +
    coord_cartesian(xlim = lims, ylim = lims) +
    theme_bw(base_size = 12)
}

# -----------------------------------------------------------------------------
# 2. MAIN TRAINING LOOP
# -----------------------------------------------------------------------------

for (year in c(2020, 2022)) {
  processed_dir <- file.path(PATH_PROCESSED, as.character(year))
  csv_files     <- list.files(processed_dir, pattern = "dataset_.*\\.csv$", full.names = TRUE)

  if (length(csv_files) == 0) {
    warning("No processed files found for year ", year, ". Run 01_data_preparation.R first.")
    next
  }

  # Axis limits differ between seasons (yield ranges)
  plot_lims <- if (year == 2020) c(6, 9) else c(9, 14)

  for (csv_path in csv_files) {
    # Extract correction method from filename (e.g. "dataset_DOS.csv" → "DOS")
    correction <- sub("dataset_(.*)\\.csv", "\\1", basename(csv_path))
    # Rename L2A to Sen2Cor for consistency with the paper
    correction_label <- ifelse(correction == "L2A", "Sen2Cor", correction)

    message("\n=== ", year, " | ", correction_label, " ===")

    # Load dataset
    dados <- read.csv(csv_path)

    # Remove ID if present (not a feature)
    if ("ID" %in% names(dados)) dados <- dados %>% select(-ID)

    # Train / test split
    set.seed(123)
    in_train <- createDataPartition(dados$Prod_13_KRIG, p = TRAIN_SPLIT, list = FALSE)
    train_data <- dados[in_train, ]
    test_data  <- dados[-in_train, ]

    # Cross-validation control
    ctrl <- trainControl(method = "repeatedcv",
                         number = CV_FOLDS,
                         repeats = CV_REPEATS)

    # Output directories
    results_dir <- file.path(PATH_RESULTS, as.character(year))
    figures_dir <- file.path(PATH_RESULTS, "figures", as.character(year))
    dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
    dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)

    for (model_name in names(MODELS)) {
      method <- MODELS[[model_name]]
      message("  Training: ", model_name, " (", method, ")")

      set.seed(123)
      model <- tryCatch(
        caret::train(
          Prod_13_KRIG ~ .,
          data       = train_data,
          method     = method,
          preProcess = c("center", "scale"),
          trControl  = ctrl
        ),
        error = function(e) {
          message("  ERROR in ", model_name, ": ", e$message)
          return(NULL)
        }
      )

      if (is.null(model)) next

      prefix <- file.path(results_dir,
                          paste0(correction_label, "_", model_name))

      # --- CV results (training performance) ---
      write.csv(model$results,
                paste0(prefix, "_resultados.csv"),
                row.names = FALSE)

      # --- Test set metrics ---
      metrics <- evaluate_model(model, test_data, model_name, correction_label)
      write.csv(metrics, paste0(prefix, "_metricas.csv"), row.names = FALSE)
      message("    R² = ", round(metrics$R2, 3),
              " | RMSE = ", round(metrics$RMSE, 3))

      # --- Observed vs. Predicted (test set) ---
      preds <- predict(model, test_data)
      obs_pred <- data.frame(Observado = test_data$Prod_13_KRIG, Predito = preds)
      write.csv(obs_pred, paste0(prefix, "_obs_pred.csv"), row.names = FALSE)

      # --- Scatter plot ---
      plot_title <- paste(correction_label, model_name, year)
      p <- make_scatter_plot(test_data$Prod_13_KRIG, preds, plot_title, plot_lims)
      ggsave(file.path(figures_dir,
                       paste0("scatter_", correction_label, "_", model_name, ".png")),
             plot = p, width = 6, height = 6, dpi = 300)
    }
  }
}

message("\nTraining complete. Results saved to: ", PATH_RESULTS)
