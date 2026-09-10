#' Read MainWood summaries (Parquet or CSV) into R.
#'
#' The pipeline writes `data/summaries_for_plots/<Region>_<Scenario>.parquet`.
#' Parquet is about 5x smaller than the CSV it replaced and keeps full float
#' precision. CSV is still available -- see "If you would rather have CSV" below.
#'
#' Quick start
#' -----------
#'   install.packages("arrow")          # once; or "nanoparquet", see below
#'   source("read_summaries.R")
#'
#'   s <- read_summary("../data/summaries_for_plots/Vaud_BAU.parquet")
#'
#'   # only the columns you need -- with Parquet the rest is never read from disk,
#'   # which matters for the multi-GB regions
#'   s <- read_summary("../data/summaries_for_plots/Surselva_WOOD.parquet",
#'                     columns = c("year", "Baumart", "diameter_class",
#'                                 "Volumen OR [m3]"))
#'
#'   # a summary too large to hold in memory: query it lazily with dplyr
#'   library(dplyr)
#'   totals <- open_summary("../data/summaries_for_plots/Surselva_WOOD.parquet") |>
#'     filter(simtype == "1") |>
#'     group_by(year, diameter_class) |>
#'     summarise(volume = sum(`Volumen OR [m3]`), .groups = "drop") |>
#'     collect()
#'
#' Which package
#' -------------
#'   arrow        full featured, lazy dplyr queries, larger install
#'   nanoparquet  tiny, zero dependencies, read_parquet() only
#' Either works with read_summary(); open_summary() needs arrow.
#'
#' If you would rather have CSV
#' ----------------------------
#' Ask for the CSV directly -- the pipeline still writes it on request
#' (`--format csv`) -- or convert it yourself:
#'
#'   python code/summary_to_csv.py data/summaries_for_plots/          # all files
#'   python code/summary_to_csv.py data/summaries_for_plots/ --gzip   # 5x smaller
#'
#' The result is identical to the CSVs this pipeline produced before 2026-09,
#' so an existing R pipeline needs no changes at all. read_summary() below also
#' reads those CSVs, so you can point it at either format.

.summary_extensions <- c(".parquet", ".csv.gz", ".csv")

#' Resolve a summary path given with or without an extension.
#' @param path Path to a summary, with or without extension.
#' @return The path that exists, preferring Parquet.
resolve_summary <- function(path) {
  if (file.exists(path) && grepl("\\.(parquet|csv|csv\\.gz)$", path)) {
    return(path)
  }
  stem <- sub("\\.(parquet|csv\\.gz|csv)$", "", path)
  for (ext in .summary_extensions) {
    candidate <- paste0(stem, ext)
    if (file.exists(candidate)) return(candidate)
  }
  stop("No summary found for '", path, "' (tried ",
       paste(.summary_extensions, collapse = ", "), ")")
}

#' Read a summary into a data.frame.
#'
#' @param path Path to a .parquet, .csv or .csv.gz summary, with or without
#'   the extension.
#' @param columns Optional character vector of columns to read. With Parquet the
#'   remaining columns are never read from disk.
#' @return A data.frame (tibble if arrow or readr is used).
read_summary <- function(path, columns = NULL) {
  path <- resolve_summary(path)

  if (grepl("\\.parquet$", path)) {
    if (requireNamespace("arrow", quietly = TRUE)) {
      # col_select must not be passed when columns is NULL: dplyr::all_of(NULL)
      # selects nothing and you would get a zero-column table back. Passing the
      # names straight to arrow also avoids depending on dplyr here.
      if (is.null(columns)) {
        return(arrow::read_parquet(path))
      }
      return(arrow::read_parquet(path, col_select = tidyselect::all_of(columns)))
    }
    if (requireNamespace("nanoparquet", quietly = TRUE)) {
      out <- nanoparquet::read_parquet(path)
      if (!is.null(columns)) out <- out[, columns, drop = FALSE]
      return(out)
    }
    stop("Reading Parquet needs the 'arrow' or 'nanoparquet' package.\n",
         "  install.packages(\"nanoparquet\")   # small, read-only\n",
         "  install.packages(\"arrow\")         # full featured\n",
         "Alternatively convert to CSV:\n",
         "  python code/summary_to_csv.py <folder with the .parquet files>")
  }

  # CSV: the first column is an unnamed row number from pandas, drop it
  if (requireNamespace("readr", quietly = TRUE)) {
    out <- readr::read_csv(path, show_col_types = FALSE)
    if (names(out)[1] %in% c("...1", "")) out <- out[, -1, drop = FALSE]
  } else {
    out <- utils::read.csv(path, check.names = FALSE)
    if (names(out)[1] %in% c("X", "")) out <- out[, -1, drop = FALSE]
  }
  if (!is.null(columns)) out <- out[, columns, drop = FALSE]
  out
}

#' Open a Parquet summary lazily, for tables too large to hold in memory.
#'
#' Returns an arrow Dataset you can pipe through dplyr verbs; nothing is read
#' until you call collect().
#'
#' @param path Path to a .parquet summary.
#' @return An arrow Dataset.
open_summary <- function(path) {
  if (!requireNamespace("arrow", quietly = TRUE)) {
    stop("open_summary() needs the 'arrow' package: install.packages(\"arrow\")")
  }
  arrow::open_dataset(resolve_summary(path))
}

#' List the summaries available in a folder.
#'
#' @param folder Path to data/summaries_for_plots.
#' @return A data.frame of file name, format and size in MB.
list_summaries <- function(folder = "../data/summaries_for_plots") {
  files <- list.files(folder, pattern = "\\.(parquet|csv|csv\\.gz)$", full.names = TRUE)
  if (length(files) == 0) {
    message("No summaries in ", folder)
    return(invisible(data.frame()))
  }
  data.frame(
    file   = basename(files),
    format = sub(".*\\.(parquet|csv\\.gz|csv)$", "\\1", files),
    size_MB = round(file.size(files) / 1e6, 1),
    row.names = NULL,
    stringsAsFactors = FALSE
  )
}

#' What the columns mean.
#'
#' Volumes are already weighted by the planting share and already rescaled to the
#' real stand area, in cubic metres (not per hectare).
summary_columns <- function() {
  data.frame(
    column = c("year", "Baumart", "Laengenklasse", "Staerkenklasse",
               "diameter_class", "Volumen OR [m3]", "Volumen IR [m3]",
               "Wert [CHF]", "Anzahl", "simtype", "stand", "planted_species",
               "plantation", "cohort", "area", "sim_area (m2)",
               "is_soft", "is_hard",
               "Volumen OR [m3]_for_sawmills", "Volumen OR [m3]_not_for_sawmills"),
    meaning = c("simulation year (>= 2020)",
                "tree species, German (Foehre/Loerche carry an 'oe' for the umlaut)",
                "SorSim length class (L1/L2/L3/Restholz)",
                "SorSim diameter class (1a..8, Restholz)",
                "grouped diameter: <20cm / 20-40cm / >40cm",
                "volume over bark, m3, weighted and rescaled to the stand",
                "volume under bark, m3, weighted and rescaled",
                "value in CHF, weighted",
                "number of assortment pieces",
                "climate scenario: 1 = RCP 8.5, 7 = RCP 4.5",
                "stand id, joins to stand.details.csv fsID",
                "planted species code, 999 = the no-planting variant",
                "TRUE for single-species plantation stands",
                "'dead' (harvested trees) or 'alive' (standing stock)",
                "stand area in hectares",
                "simulated area, always 62500 m2 (100 patches of 625 m2)",
                "softwood",
                "hardwood",
                "share suitable for sawmills (diameter >40cm x species quality)",
                "the remainder"),
    row.names = NULL,
    stringsAsFactors = FALSE
  )
}
