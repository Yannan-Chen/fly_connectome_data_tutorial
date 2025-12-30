#!/bin/bash
# Tutorial 02 Runtime Script - Google Colab Compatible

set -e

echo "========================================="
echo "SJCABS Tutorial 02: Neuron Morphology"
echo "Installing Python packages..."
echo "========================================="

# Install system dependencies if in Colab
if [ "$EUID" -eq 0 ] || sudo -n true 2>/dev/null; then
    echo "Installing system libraries..."
    apt-get update -qq
    apt-get install -y -qq libgomp1
    echo "✓ System libraries installed"
fi

# Update pip
python3 -m pip install --quiet --upgrade pip

# Fix protobuf
echo "Installing protobuf (compatible version)..."
python3 -m pip uninstall -y protobuf 2>/dev/null || true
python3 -m pip install --no-cache-dir "protobuf>=3.20,<5.0"

# Core packages
python3 -m pip install --quiet --upgrade \
    pandas==2.3.3 \
    "numpy>=2.0,<2.1" \
    pyarrow \
    gcsfs

# Visualization
python3 -m pip install --quiet --upgrade \
    plotly==5.24.1 \
    kaleido \
    matplotlib \
    seaborn

# Scientific computing
python3 -m pip install --quiet --upgrade \
    scipy \
    scikit-learn

# Neuroscience tools
python3 -m pip install --quiet --upgrade \
    navis==1.10.0 \
    trimesh \
    pykdtree \
    ncollpyde

# Optional: flybrains for transforms
python3 -m pip install --quiet --upgrade \
    navis-flybrains || echo "ℹ flybrains installation optional"

# Optional: CAVE client for mesh downloading (Extension 03)
python3 -m pip install --quiet --upgrade \
    caveclient \
    cloud-volume || echo "ℹ caveclient installation optional"

# Jupyter support
python3 -m pip install --quiet --upgrade \
    ipywidgets \
    jupyter \
    tqdm

echo ""
echo "Verifying installations..."
python3 -c "import navis; print(f'✓ navis {navis.__version__}')"
python3 -c "import pandas as pd; print(f'✓ pandas {pd.__version__}')"
python3 -c "import gcsfs; print('✓ gcsfs installed')"
python3 -c "import plotly; print('✓ plotly installed')"
python3 -c "import trimesh; print('✓ trimesh installed')"

echo ""
echo "Optional packages:"
python3 -c "import caveclient; print(f'✓ caveclient {caveclient.__version__}')" 2>/dev/null || echo "ℹ caveclient not installed (Extension 03 will be skipped)"

echo ""
echo "========================================="
echo "Environment setup complete!"
echo "Ready for Tutorial 02: Neuron Morphology"
echo "========================================="
