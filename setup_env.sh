#!/usr/bin/env bash
# ==============================================================================
# Script: setup_env.sh
# Purpose: Install submodules in editable mode and configure symlinks for
#          sample dataset and checkpoints.
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo "    Environment Setup for Automatic Car Segmentation"
echo "=========================================================="

# 1. Install submodules in editable mode
echo "[1/3] Installing GroundingDINO in editable mode..."
pip install -e "$SCRIPT_DIR/GroundingDINO" --no-build-isolation

echo "[2/3] Installing Segment Anything 2 (SAM 2) in editable mode..."
pip install -e "$SCRIPT_DIR/segment-anything-2" --no-build-isolation

# 3. Verify symlinks and checkpoints
echo "[3/3] Verifying file paths and sample data links..."
if [ ! -e "$SCRIPT_DIR/groundingdino_swint_ogc.pth" ] && [ -f "$SCRIPT_DIR/checkpoints/groundingdino_swint_ogc.pth" ]; then
    ln -sf "checkpoints/groundingdino_swint_ogc.pth" "$SCRIPT_DIR/groundingdino_swint_ogc.pth"
fi

if [ ! -e "$SCRIPT_DIR/sam2_hiera_large.pt" ] && [ -f "$SCRIPT_DIR/checkpoints/sam2_hiera_large.pt" ]; then
    ln -sf "checkpoints/sam2_hiera_large.pt" "$SCRIPT_DIR/sam2_hiera_large.pt"
fi

if [ ! -e "$SCRIPT_DIR/IMAGE DATASET" ] && [ -d "$SCRIPT_DIR/sample_data/IMAGE DATASET" ]; then
    ln -sf "sample_data/IMAGE DATASET" "$SCRIPT_DIR/IMAGE DATASET"
fi

echo ""
echo "=== Setup completed successfully! ==="
echo "You can now run any of the segmentation pipelines:"
echo "  python frontSegmentator.py"
echo "  python lateralSegmentator.py"
echo "  python rearSegmentator.py"
echo "  python segment_3drealcar_custom_parts.py"
