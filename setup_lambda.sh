#!/usr/bin/env bash
# ==============================================================================
# Simplex-Splat: Lambda Labs GPU Instance Setup
# Run: bash setup_lambda.sh
# Tested on: Ubuntu 22.04 + NVIDIA A10/A100 (Lambda Cloud)
# ==============================================================================
set -euo pipefail

echo "=== Simplex-Splat Lambda Setup ==="

# ---------- System packages ---------------------------------------------------
echo "[1/7] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    libomp5 libpng-dev libjpeg-dev libtiff-dev \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    wget unzip git tmux htop vulkan-utils

# ---------- Conda environment -------------------------------------------------
echo "[2/7] Creating conda environment..."
conda create -n simplex-splat python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate simplex-splat

# ---------- PyTorch (CUDA 11.8) -----------------------------------------------
echo "[3/7] Installing PyTorch with CUDA..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118

# ---------- CARLA 0.9.15 ------------------------------------------------------
echo "[4/7] Installing CARLA..."
CARLA_VERSION="0.9.15"
CARLA_DIR="$HOME/carla"

if [ ! -d "$CARLA_DIR" ]; then
    mkdir -p "$CARLA_DIR"
    cd "$CARLA_DIR"
    wget -q "https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_${CARLA_VERSION}.tar.gz"
    tar xzf "CARLA_${CARLA_VERSION}.tar.gz"
    rm "CARLA_${CARLA_VERSION}.tar.gz"
    echo "CARLA installed at $CARLA_DIR"
else
    echo "CARLA already exists at $CARLA_DIR, skipping."
fi

pip install carla==0.9.15
pip install pygame

# ---------- SplaTAM -----------------------------------------------------------
echo "[5/7] Installing SplaTAM..."
SPLATAM_DIR="$HOME/SplaTAM"

if [ ! -d "$SPLATAM_DIR" ]; then
    cd "$HOME"
    git clone --recursive https://github.com/spla-tam/SplaTAM.git
    cd "$SPLATAM_DIR"
    pip install -r requirements.txt
    # Install diff-gaussian-rasterization (3DGS renderer)
    pip install submodules/diff-gaussian-rasterization
    # Install simple-knn
    pip install submodules/simple-knn
    echo "SplaTAM installed at $SPLATAM_DIR"
else
    echo "SplaTAM already exists, skipping."
fi

# ---------- MAVBE / Simplex-Splat project deps --------------------------------
echo "[6/7] Installing project dependencies..."
PROJECT_DIR="$HOME/MAVBE"
if [ -d "$PROJECT_DIR" ]; then
    cd "$PROJECT_DIR"
    pip install -r requirements.txt
    pip install torchreid
fi

# Common deps for the project
pip install open3d trimesh scikit-image

# ---------- Verify installation -----------------------------------------------
echo "[7/7] Verifying setup..."
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')

import carla
print(f'CARLA Python API: OK')

import pygame
print(f'Pygame: OK')
"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To start CARLA server (in a tmux session):"
echo "  tmux new -s carla"
echo "  cd ~/carla && ./CarlaUE4.sh -RenderOffScreen -quality-level=Epic"
echo ""
echo "To activate the environment:"
echo "  conda activate simplex-splat"
echo ""
echo "To connect from your Mac (SSH tunnel):"
echo "  ssh -L 2000:localhost:2000 -L 2001:localhost:2001 -L 2002:localhost:2002 ubuntu@<LAMBDA_IP>"
