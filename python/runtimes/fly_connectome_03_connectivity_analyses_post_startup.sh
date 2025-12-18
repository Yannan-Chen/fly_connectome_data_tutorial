#!/bin/bash
# connectivity_analyses_post_startup.sh - SJCABS Tutorial 03: Connectivity Analyses
# Runtime script for Vertex AI Colab - Tutorial 03 Environment Setup

set -e

echo "========================================="
echo "SJCABS Tutorial 03: Connectivity Analyses"
echo "Installing Python packages..."
echo "========================================="

# Update pip
python3 -m pip install --quiet --upgrade pip

# Install core data science packages
python3 -m pip install --quiet --upgrade \
    pandas==2.3.3 \
    numpy \
    pyarrow \
    gcsfs

# Install visualization packages
python3 -m pip install --quiet --upgrade \
    plotly \
    kaleido \
    matplotlib \
    seaborn

# Install network analysis and clustering tools
python3 -m pip install --quiet --upgrade \
    networkx \
    scipy \
    scikit-learn \
    umap-learn

# Install navis for neuroscience tools
python3 -m pip install --quiet --upgrade \
    navis==1.10.0 \
    trimesh \
    pykdtree \
    ncollpyde

# Install Jupyter widgets for interactive plots
python3 -m pip install --quiet --upgrade \
    ipywidgets \
    tqdm

# Verify key installations
echo ""
echo "Verifying installations..."
python3 -c "import pandas as pd; print(f'✓ pandas {pd.__version__}')" || echo "✗ pandas failed"
python3 -c "import gcsfs; print('✓ gcsfs installed')" || echo "✗ gcsfs failed"
python3 -c "import plotly; print('✓ plotly installed')" || echo "✗ plotly failed"
python3 -c "import networkx; print('✓ networkx installed')" || echo "✗ networkx failed"
python3 -c "import umap; print('✓ umap installed')" || echo "✗ umap failed"

echo ""
echo "========================================="
echo "Environment setup complete!"
echo "Ready for Tutorial 03: Connectivity Analyses"
echo "========================================="
