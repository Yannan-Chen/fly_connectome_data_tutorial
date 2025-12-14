# Package Installation and Loading for Fly Connectome Tutorial
# =============================================================
# This script handles all package dependencies for the tutorial

# Core R packages
# ---------------
packages_core <- c(
  "arrow",      # Reading feather/parquet files
  "tidyverse",  # Data manipulation and plotting
  "ggplot2",    # Plotting
  "patchwork",  # Combining plots
  "nat",        # Neuron morphology analysis
  "nat.nblast", # compare neuron morphologies
  "plotly",     # interactive plots
  "duckdb"      # SQL database for efficient Parquet queries
)

# Install if needed
for (pkg in packages_core) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg)
  }
}

# Load core packages
library(arrow)
library(tidyverse)
library(ggplot2)
library(patchwork)
library(nat)
library(duckdb)
library(plotly)
library(nat.nblast)
library(doMC)

cat("✓ Core packages loaded\n")

# Set up parallelisation
numCores <- parallel::detectCores()
cores <- max(1,numCores-1)
registerDoMC(cores)
cat("Using ", cores, " cores")

# Python/GCS packages (conditional)
# ----------------------------------
setup_gcs_access <- function() {
  cat("Setting up GCS access...\n")

  if (!require("reticulate", quietly = TRUE)) {
    install.packages("reticulate")
  }
  library(reticulate)

  # Python packages needed for GCS
  python_packages <- c("gcsfs", "fsspec", "pyarrow")

  # Check if conda is available
  has_conda <- system("which conda",
                     intern = FALSE,
                     ignore.stdout = TRUE,
                     ignore.stderr = TRUE) == 0

  if (has_conda) {
    cat("Using conda Python environment...\n")

    env_name <- "r-gcs"
    tryCatch({
      use_condaenv(env_name, required = FALSE)
    }, error = function(e) {
      cat("Creating conda environment:", env_name, "\n")
      conda_create(env_name, packages = c("python=3.10"))
      use_condaenv(env_name, required = TRUE)
    })

    # Install Python packages if needed
    for (pkg in python_packages) {
      if (!py_module_available(pkg)) {
        cat("Installing Python package:", pkg, "\n")
        conda_install(env_name, pkg, pip = TRUE)
      }
    }
  } else {
    cat("Conda not found. Using system Python...\n")
    cat("If you encounter SSL errors, install Anaconda: https://www.anaconda.com/download\n")

    # Try to install with pip
    for (pkg in python_packages) {
      if (!py_module_available(pkg)) {
        py_install(pkg, pip = TRUE)
      }
    }
  }

  cat("✓ GCS access configured\n")
}
