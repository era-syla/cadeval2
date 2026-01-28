#!/bin/bash
# Setup script for CadQuery + Blender rendering environment on H100 cluster
# This creates two environments: one for CadQuery conversion, one for Blender rendering

set -e

echo "=============================================="
echo "Setting up CadQuery + Rendering Environment"
echo "=============================================="

# Check if mamba or conda is available
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
    echo "Using mamba for faster installation"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
    echo "Using conda"
else
    echo "ERROR: Neither mamba nor conda found. Please install miniconda/mamba first."
    exit 1
fi

# ============================================
# Environment 1: CadQuery (for .py -> .step conversion)
# ============================================
ENV_NAME_CQ="cadquery_env"

echo ""
echo "Step 1: Creating CadQuery environment ($ENV_NAME_CQ)"
echo "----------------------------------------------"

# Check if environment already exists
if $CONDA_CMD env list | grep -q "^$ENV_NAME_CQ "; then
    echo "Environment $ENV_NAME_CQ already exists. Skipping creation."
else
    $CONDA_CMD create -n $ENV_NAME_CQ python=3.10 -y
fi

# Activate and install packages
echo "Installing CadQuery and dependencies..."
eval "$($CONDA_CMD shell.bash hook)"
$CONDA_CMD activate $ENV_NAME_CQ

# Install cadquery from conda-forge (most reliable method)
$CONDA_CMD install -c conda-forge cadquery -y

# Install additional utilities
pip install tqdm

echo "CadQuery environment setup complete!"

# ============================================
# Environment 2: Blender/Rendering (for .step -> images)
# ============================================
ENV_NAME_RENDER="cad_render_env"

echo ""
echo "Step 2: Creating Rendering environment ($ENV_NAME_RENDER)"
echo "----------------------------------------------"

# Check if environment already exists
if $CONDA_CMD env list | grep -q "^$ENV_NAME_RENDER "; then
    echo "Environment $ENV_NAME_RENDER already exists. Skipping creation."
else
    $CONDA_CMD create -n $ENV_NAME_RENDER python=3.11 -y
fi

$CONDA_CMD activate $ENV_NAME_RENDER

# Install pythonocc-core for STEP file reading
$CONDA_CMD install -c conda-forge pythonocc-core -y

# Install bpy (Blender as Python module)
pip install bpy==4.2.0

# Install blendify and other rendering deps
pip install blendify numpy Pillow trimesh tqdm seaborn

# Fix potential numpy version conflicts
pip install -U numpy

# Install freeimage for image processing (needed by some rendering)
$CONDA_CMD install -c conda-forge freeimage=3.17 --no-deps -y || true

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Usage:"
echo "  1. For CadQuery conversion (.py -> .step):"
echo "     conda activate $ENV_NAME_CQ"
echo "     python convert_cq_to_step.py --input-dir <source> --output-dir <dest>"
echo ""
echo "  2. For rendering (.step -> images):"
echo "     conda activate $ENV_NAME_RENDER"
echo "     python generate_images.py --data_path <path> --gray_images"
echo ""
echo "  3. For the full pipeline:"
echo "     python prepare_training_data.py --input-dir <cq_scripts> --output-dir <training_data>"
echo ""
