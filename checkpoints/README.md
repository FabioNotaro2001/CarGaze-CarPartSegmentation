# Pretrained Model Checkpoints

This directory is intended to store the pre-trained neural network weights required by the segmentation pipeline:

1. **Grounding DINO Swin-T**:
   - Filename: `groundingdino_swint_ogc.pth`
   - Source: [IDEA-Research GroundingDINO Release v0.1.0-alpha](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth)

2. **Segment Anything 2 (SAM 2.0 Hiera Large)**:
   - Filename: `sam2_hiera_large.pt`
   - Source: [Meta Segment Anything 2](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt)

## Automatic Download

You can automatically download these weights by running either:

```bash
# On Linux / macOS / Git Bash:
bash download_weights.sh

# Cross-platform (Windows / Linux / macOS):
python download_weights.py
```
