#!/usr/bin/env bash
# ==============================================================================
# Script: download_weights.sh
# Purpose: Automatically download pretrained foundation model weights for
#          Grounding DINO Swin-T and Segment Anything 2 (SAM 2 Hiera Large).
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT_DIR="$SCRIPT_DIR/checkpoints"
mkdir -p "$CKPT_DIR"

echo "=========================================================="
echo "    Downloading Pretrained Model Checkpoints"
echo "=========================================================="

# 1. Grounding DINO Swin-T
GDINO_FILE="$CKPT_DIR/groundingdino_swint_ogc.pth"
if [ ! -f "$GDINO_FILE" ]; then
    echo "[1/2] Downloading groundingdino_swint_ogc.pth..."
    wget -c -O "$GDINO_FILE" "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
else
    echo "[1/2] groundingdino_swint_ogc.pth is already present."
fi

# 2. Segment Anything 2 (SAM 2.0 Hiera Large)
SAM2_FILE="$CKPT_DIR/sam2_hiera_large.pt"
if [ ! -f "$SAM2_FILE" ]; then
    echo "[2/2] Downloading sam2_hiera_large.pt..."
    wget -c -O "$SAM2_FILE" "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
else
    echo "[2/2] sam2_hiera_large.pt is already present."
fi

echo ""
echo "All model checkpoints are ready in: $CKPT_DIR"
