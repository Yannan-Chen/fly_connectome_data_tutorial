#!/bin/bash
# neuron_morphology_post_startup.sh - SJCABS Tutorial 02: Neuron Morphology
# Runtime script for Vertex AI Colab - Tutorial 02 Environment Setup

set -e

echo "========================================="
echo "SJCABS Tutorial 02: Neuron Morphology"
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

# Install navis and neuroscience tools (essential for morphology analysis)
python3 -m pip install --quiet --upgrade \
    navis==1.10.0 \
    trimesh \
    pykdtree \
    ncollpyde \
    scipy \
    scikit-learn \
    networkx \
    umap-learn

# Install Jupyter widgets for interactive plots
python3 -m pip install --quiet --upgrade \
    ipywidgets \
    tqdm

# Verify key installations
echo ""
echo "Verifying installations..."
python3 -c "import navis; print(f'✓ navis {navis.__version__}')" || echo "✗ navis failed"
python3 -c "import pandas as pd; print(f'✓ pandas {pd.__version__}')" || echo "✗ pandas failed"
python3 -c "import gcsfs; print('✓ gcsfs installed')" || echo "✗ gcsfs failed"
python3 -c "import plotly; print('✓ plotly installed')" || echo "✗ plotly failed"
python3 -c "import trimesh; print('✓ trimesh installed')" || echo "✗ trimesh failed"

echo ""
echo "========================================="
echo "Environment setup complete!"
echo "Ready for Tutorial 02: Neuron Morphology"
echo "========================================="
