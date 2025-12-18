# Installation Scripts

This directory contains setup scripts for the fly connectome tutorials.

## Python Environment Setup

### Quick Start

```bash
# Make the script executable (if not already)
chmod +x inst/setup_python_env.sh

# Run the setup script
./inst/setup_python_env.sh
```

### What It Does

The `setup_python_env.sh` script will:

1. **Check for conda** - Verifies that Anaconda or Miniconda is installed
2. **Create environment** - Creates a new conda environment called `sjcabs` with Python 3.10
3. **Install core packages:**
   - Scientific computing: numpy, pandas, scipy
   - Data handling: pyarrow, gcsfs
   - Visualization: plotly, kaleido, matplotlib, seaborn
   - Jupyter: jupyter, jupyterlab
4. **Install neuroscience packages:**
   - **navis** - Python library for neuron analysis and visualization
   - **flybrains** - Template brain transformations for Drosophila
   - **ConnectomeInfluenceCalculator** - Calculate indirect connectivity influence scores
5. **Install analysis packages:**
   - umap-learn - Dimensionality reduction
   - trimesh - 3D mesh processing
   - joblib, tqdm - Parallel processing and progress bars

### Optional: Flybrains Templates

During installation, you'll be asked whether to download flybrains template registrations (~500MB). These are needed for transforming neurons between different brain templates (e.g., MANC → BANC).

If you skip this step, you can download them later:

```python
import flybrains
flybrains.download_jrc_transforms()
flybrains.download_jrc_vnc_transforms()
flybrains.register_transforms()
```

### Using the Environment

After installation, activate the environment:

```bash
conda activate sjcabs
```

Run the Python tutorials:

```bash
jupyter lab python/fly_connectome_01_data_access.ipynb
```

Deactivate when done:

```bash
conda deactivate
```

### Troubleshooting

**conda not found:**
- Install Miniconda: https://docs.conda.io/en/latest/miniconda.html
- Or Anaconda: https://www.anaconda.com/download

**Environment already exists:**
- The script will ask if you want to recreate it
- Or manually remove: `conda env remove -n sjcabs`

**Import errors:**
- Ensure environment is activated: `conda activate sjcabs`
- Test imports: `python -c "import navis; import pandas; print('✓ OK')"`

**SSL/Certificate errors:**
- This usually happens with system Python
- Using conda environment should avoid this issue

## Package Versions

The script installs the latest compatible versions. For reproducibility, you can export the exact environment:

```bash
conda activate sjcabs
conda env export > environment.yml
```

And recreate it later:

```bash
conda env create -f environment.yml
```

## Dependencies Summary

| Package | Purpose |
|---------|---------|
| **navis** | Neuron analysis, NBLAST, visualization |
| **flybrains** | Template brain transformations |
| **InfluenceCalculator** | Indirect connectivity analysis |
| **pandas** | Data manipulation |
| **numpy** | Numerical computing |
| **scipy** | Scientific computing, clustering |
| **pyarrow** | Feather file format |
| **gcsfs** | Google Cloud Storage access |
| **plotly** | Interactive 3D visualization |
| **umap-learn** | Dimensionality reduction |
| **trimesh** | 3D mesh processing |
| **joblib** | Parallel processing |
| **tqdm** | Progress bars |

## Support

For issues with:
- **Tutorial content:** See main README.md
- **Python packages:** Check package documentation
- **navis:** https://navis.readthedocs.io/
- **flybrains:** https://github.com/navis-org/navis-flybrains
