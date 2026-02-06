#!/usr/bin/env Rscript

root <- file.path("python", "tests", "data", "P-Tree-Public-Data-main")
out_file <- file.path("python", "tests", "data", "public_data_stats_r.csv")

read_stats <- function(path) {
  df <- read.csv(path, check.names = FALSE)
  # drop common index column if present
  drop_cols <- names(df) %in% c("Unnamed: 0", "")
  if (any(drop_cols)) {
    df <- df[, !drop_cols, drop = FALSE]
  }
  is_num <- vapply(df, is.numeric, logical(1))
  df <- df[, is_num, drop = FALSE]
  num_cols <- names(df)
  stats <- data.frame(
    file = basename(path),
    column = num_cols,
    mean = NA_real_,
    sd = NA_real_,
    sharpe = NA_real_,
    n = nrow(df),
    stringsAsFactors = FALSE
  )
  for (i in seq_along(num_cols)) {
    x <- df[[num_cols[i]]]
    stats$mean[i] <- mean(x)
    stats$sd[i] <- sd(x)
    stats$sharpe[i] <- ifelse(stats$sd[i] == 0, 0, stats$mean[i] / stats$sd[i])
  }
  stats
}

folders <- c(
  "Train_1981_2020",
  "Train_1981_2000_Test_2001_2020",
  "Train_2001_2020_Test_1981_2000"
)

all_stats <- NULL
for (f in folders) {
  dir_path <- file.path(root, f)
  csvs <- list.files(dir_path, pattern = "\\.csv$", full.names = TRUE)
  if (length(csvs) == 0) {
    next
  }
  for (csv in csvs) {
    stats <- read_stats(csv)
    stats$folder <- f
    all_stats <- rbind(all_stats, stats)
  }
}

if (is.null(all_stats)) {
  stop("No CSV files found.")
}

write.csv(all_stats, out_file, row.names = FALSE)
cat("Wrote", out_file, "\n")
