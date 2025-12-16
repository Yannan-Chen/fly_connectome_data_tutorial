# Package Installation and Loading for Fly Connectome Tutorial
# =============================================================
# This script handles all package dependencies for the tutorial

# Core R packages
# ---------------
packages_core <- c(
  "arrow",           # Reading feather/parquet files
  "tidyverse",       # Data manipulation and plotting
  "ggplot2",         # Plotting
  "patchwork",       # Combining plots
  "nat",             # Neuron morphology analysis
  "nat.nblast",      # compare neuron morphologies
  "nat.flybrains",   # Transform neurons between fly template spaces
  "plotly",          # Interactive plots
  "duckdb",          # SQL database for efficient Parquet queries
  "ggdendro",        # Dendrogram visualization
  "heatmaply",       # Interactive heatmaps with dendrograms
  "pheatmap",        # Static heatmaps
  "umap",            # Dimensionality reduction (Tutorial 04)
  "uwot",            # UMAP implementation (Tutorial 03)
  "htmlwidgets",     # Save interactive plots as HTML
  "igraph",          # Network analysis
  "ggraph",          # Network visualization with ggplot2
  "tidygraph",       # Tidy network analysis
  "lsa",             # Cosine similarity
  "Matrix",          # Sparse matrices
  "dynamicTreeCut",  # Dynamic tree cutting for clustering
  "readobj",         # Reading 3D mesh files (.obj format)
  "foreach",         # Parallel for loops
  "doSNOW"           # Parallel backend for foreach (cross-platform, supports progress bars)
)

# Packages from GitHub/Natverse
# ------------------------------
packages_github <- c(
  "natverse/influencer"  # Indirect connectivity analysis
)

# Install core packages if needed
for (pkg in packages_core) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg)
  }
}

# Install GitHub packages if needed
if (!require("remotes", quietly = TRUE)) {
  install.packages("remotes")
}
library(remotes)

for (pkg in packages_github) {
  pkg_name <- basename(pkg)
  if (!require(pkg_name, character.only = TRUE, quietly = TRUE)) {
    cat("Installing", pkg, "from GitHub...\n")
    remotes::install_github(pkg)
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
library(ggdendro)
library(heatmaply)
library(pheatmap)
library(umap)
library(uwot)
library(htmlwidgets)
library(igraph)
library(ggraph)
library(tidygraph)
library(lsa)
library(Matrix)
library(dynamicTreeCut)
library(readobj)
library(foreach)
library(doSNOW)
library(influencer)
library(nat.flybrains)
cat("✓ Core packages loaded\n")

# Set up parallelisation
numCores <- parallel::detectCores()
cores <- max(1, numCores - 1)
cl <- makeCluster(cores)
registerDoSNOW(cl)
cat("Using", cores, "cores for parallel processing\n")

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

  # Suppress Python FutureWarnings from pandas
  if (py_module_available("warnings")) {
    py_run_string("import warnings; warnings.filterwarnings('ignore', category=FutureWarning)")
    py_run_string("import warnings; warnings.filterwarnings('ignore', category=DeprecationWarning)")
    cat("✓ Python warnings suppressed\n")
  }
}

# Warning Suppression
# -------------------
# Suppress common R warnings that clutter tutorial output

# Suppress summarise() grouping messages
options(dplyr.summarise.inform = FALSE)

# Suppress namespace conflict messages
suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
})

# Suppress specific ggplot2 warnings
options(ggplot2.continuous.colour = "viridis")
options(ggplot2.continuous.fill = "viridis")

cat("✓ Warning suppression configured\n")
