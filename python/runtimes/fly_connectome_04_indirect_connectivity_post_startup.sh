#!/bin/bash
# Tutorial 04 Runtime Script - Google Colab Compatible
# CRITICAL: This script MUST succeed or fail loudly

set -e  # Exit on any error
set -o pipefail  # Exit on pipe failures

echo "========================================="
echo "SJCABS Tutorial 04: Indirect Connectivity"
echo "Installing Python packages..."
echo "========================================="

# Update pip
python3 -m pip install --quiet --upgrade pip

# CRITICAL: Uninstall and reinstall protobuf to fix version conflicts
echo "Fixing protobuf version..."
python3 -m pip uninstall -y protobuf 2>/dev/null || true
python3 -m pip install --no-cache-dir "protobuf>=3.20,<5.0"
echo "✓ protobuf installed"

# Install core packages
echo "Installing core data science packages..."
python3 -m pip install --quiet \
    pandas==2.3.3 \
    "numpy>=2.0,<2.1" \
    pyarrow \
    gcsfs

# Install visualization
echo "Installing visualization packages..."
python3 -m pip install --quiet \
    plotly==5.24.1 \
    kaleido \
    matplotlib \
    seaborn

# Install analysis tools
echo "Installing analysis packages..."
python3 -m pip install --quiet \
    networkx \
    scipy \
    scikit-learn \
    umap-learn \
    ipywidgets \
    jupyter \
    tqdm \
    joblib

echo "✓ Core packages installed"

# Install InfluenceCalculator - REQUIRED, NO GRACEFUL FAILURES
echo ""
echo "Installing ConnectomeInfluenceCalculator..."
echo "This is REQUIRED for Tutorial 04"

# Check if conda is available (local) or use Colab approach
if command -v conda >/dev/null 2>&1; then
    # Local conda environment
    echo "  Detected conda - using conda-forge for PETSc/SLEPc"
    conda install -c conda-forge petsc petsc4py slepc slepc4py -y --quiet
else
    # Google Colab environment
    echo "  Detected Colab - installing PETSc/SLEPc via system packages"
    
    # Install system dependencies (Colab has sudo)
    apt-get update -qq
    apt-get install -y -qq libpetsc-real-dev libslepc-real-dev build-essential gfortran
    echo "  ✓ System libraries installed"
    
    # Install Python wrappers
    echo "  Installing PETSc/SLEPc Python bindings..."
    python3 -m pip install --no-cache-dir petsc4py slepc4py
    echo "  ✓ PETSc/SLEPc Python bindings installed"
fi

# Clone and install InfluenceCalculator
echo "  Downloading ConnectomeInfluenceCalculator from GitHub..."
TEMP_DIR="/tmp/ic_install_$$"
git clone --quiet https://github.com/DrugowitschLab/ConnectomeInfluenceCalculator.git "$TEMP_DIR"

# Fix known pyproject.toml issue
if [ -f "$TEMP_DIR/pyproject.toml" ]; then
    sed -i 's/^license = "BSD-3-Clause"/license = {text = "BSD-3-Clause"}/' "$TEMP_DIR/pyproject.toml"
fi

# Install
echo "  Installing ConnectomeInfluenceCalculator..."
python3 -m pip install "$TEMP_DIR"

# Cleanup
rm -rf "$TEMP_DIR"

# CRITICAL: Verify installation - FAIL if this doesn't work
echo ""
echo "Verifying InfluenceCalculator installation..."
python3 -c "from InfluenceCalculator import InfluenceCalculator; print('✓ InfluenceCalculator imported successfully')"

echo ""
echo "========================================="
echo "✓ ALL PACKAGES INSTALLED SUCCESSFULLY"
echo "Ready for Tutorial 04: Indirect Connectivity"
echo "========================================="
