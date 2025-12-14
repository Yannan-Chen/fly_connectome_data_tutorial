#!/bin/bash
# install_reticulate_dependencies.sh
# Installation script for GCS Parquet reading dependencies
# Installs gcsfs, pyarrow, and pandas into r-reticulate environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}📊 Fly Connectome Tutorial - Dependency Installer${NC}"
echo -e "${BLUE}===================================================${NC}"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for conda/mamba
CONDA_CMD=""
if command_exists conda; then
    CONDA_CMD="conda"
    echo -e "${GREEN}✓ Found conda${NC}"
elif command_exists mamba; then
    CONDA_CMD="mamba"
    echo -e "${GREEN}✓ Found mamba${NC}"
else
    echo -e "${RED}✗ Error: Neither conda nor mamba found${NC}"
    echo "  Please install Miniconda or Anaconda:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Locate r-reticulate environment
echo ""
echo -e "${YELLOW}🔍 Looking for r-reticulate environment...${NC}"

R_RETICULATE_PATH=""

# Check common locations
POSSIBLE_PATHS=(
    "$HOME/Library/r-miniconda/envs/r-reticulate"
    "$HOME/.local/share/r-miniconda/envs/r-reticulate"
    "$HOME/r-miniconda/envs/r-reticulate"
)

for path in "${POSSIBLE_PATHS[@]}"; do
    if [ -d "$path" ]; then
        R_RETICULATE_PATH="$path"
        echo -e "${GREEN}✓ Found r-reticulate at: $path${NC}"
        break
    fi
done

# If not found in common locations, search conda env list
if [ -z "$R_RETICULATE_PATH" ]; then
    echo "  Searching conda environments..."
    if $CONDA_CMD env list | grep -q "r-reticulate"; then
        R_RETICULATE_PATH=$($CONDA_CMD env list | grep "r-reticulate" | awk '{print $NF}')
        echo -e "${GREEN}✓ Found r-reticulate at: $R_RETICULATE_PATH${NC}"
    fi
fi

# If still not found, offer to create it
if [ -z "$R_RETICULATE_PATH" ]; then
    echo -e "${YELLOW}⚠  r-reticulate environment not found${NC}"
    echo "  Would you like to create it? (y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo -e "${YELLOW}📦 Creating r-reticulate environment with Python 3.10...${NC}"
        $CONDA_CMD create -n r-reticulate python=3.10 -y
        R_RETICULATE_PATH=$($CONDA_CMD env list | grep "r-reticulate" | awk '{print $NF}')
        echo -e "${GREEN}✓ Environment created${NC}"
    else
        echo -e "${RED}✗ Installation cancelled${NC}"
        exit 1
    fi
fi

# Get Python path
PYTHON_PATH="$R_RETICULATE_PATH/bin/python"

if [ ! -f "$PYTHON_PATH" ]; then
    echo -e "${RED}✗ Python not found at: $PYTHON_PATH${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}📦 Checking installed packages...${NC}"

# Function to check if a Python package is installed
check_package() {
    local package=$1
    if $PYTHON_PATH -c "import $package" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to get package version
get_version() {
    local package=$1
    $PYTHON_PATH -c "import $package; print($package.__version__)" 2>/dev/null || echo "unknown"
}

# Check each required package
NEEDS_INSTALL=false

echo ""
echo "Checking gcsfs..."
if check_package gcsfs; then
    version=$(get_version gcsfs)
    echo -e "${GREEN}  ✓ gcsfs installed (version $version)${NC}"
else
    echo -e "${YELLOW}  ✗ gcsfs not installed${NC}"
    NEEDS_INSTALL=true
fi

echo "Checking pyarrow..."
if check_package pyarrow; then
    version=$(get_version pyarrow)
    echo -e "${GREEN}  ✓ pyarrow installed (version $version)${NC}"

    # Check if version is >= 22.0.0
    major_version=$(echo $version | cut -d. -f1)
    if [ "$major_version" -lt 22 ]; then
        echo -e "${YELLOW}  ⚠  pyarrow version $version < 22.0.0 (upgrade recommended)${NC}"
        NEEDS_INSTALL=true
    fi
else
    echo -e "${YELLOW}  ✗ pyarrow not installed${NC}"
    NEEDS_INSTALL=true
fi

echo "Checking pandas..."
if check_package pandas; then
    version=$(get_version pandas)
    echo -e "${GREEN}  ✓ pandas installed (version $version)${NC}"
else
    echo -e "${YELLOW}  ✗ pandas not installed${NC}"
    NEEDS_INSTALL=true
fi

# Install/upgrade if needed
if [ "$NEEDS_INSTALL" = true ]; then
    echo ""
    echo -e "${YELLOW}📥 Installing/upgrading packages...${NC}"
    echo "  This may take a few minutes..."
    echo ""

    # Install with pip
    $PYTHON_PATH -m pip install --upgrade --quiet gcsfs pyarrow pandas

    echo -e "${GREEN}✓ Installation complete${NC}"

    # Verify installation
    echo ""
    echo -e "${YELLOW}✅ Verifying installation...${NC}"

    echo "  gcsfs: $(get_version gcsfs)"
    echo "  pyarrow: $(get_version pyarrow)"
    echo "  pandas: $(get_version pandas)"
else
    echo ""
    echo -e "${GREEN}✓ All required packages are already installed${NC}"
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}🎉 Setup Complete!${NC}"
echo ""
echo -e "${YELLOW}📝 Environment Configuration:${NC}"
echo "  Python: $PYTHON_PATH"
echo "  Environment: r-reticulate"
echo ""
echo -e "${YELLOW}🔑 Authentication Setup:${NC}"
echo "  For GCS access, authenticate with:"
echo "  ${BLUE}gcloud auth application-default login${NC}"
echo ""
echo -e "${YELLOW}💡 Usage in R:${NC}"
echo "  The tutorial will automatically use the r-reticulate environment"
echo "  Just run: ${BLUE}source('packages.R'); source('functions.R')${NC}"
echo ""
