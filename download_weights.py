#!/usr/bin/env python3
"""
download_weights.py
===================
Cross-platform utility script to download pretrained model checkpoints for
Grounding DINO and Segment Anything 2 (SAM 2).

Usage:
    python download_weights.py
"""

import os
import sys
import urllib.request

WEIGHTS = {
    "groundingdino_swint_ogc.pth": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
    "sam2_hiera_large.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
}


def download_with_progress(url: str, output_path: str) -> None:
    """Download a file from a URL to an output path with progress reporting."""
    def _reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100.0 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  Progress: {percent:6.2f}% ({mb_downloaded:6.1f} / {mb_total:6.1f} MB)")
            sys.stdout.flush()
        else:
            mb_downloaded = downloaded / (1024 * 1024)
            sys.stdout.write(f"\r  Downloaded: {mb_downloaded:6.1f} MB")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, output_path, reporthook=_reporthook)
    sys.stdout.write("\n")


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(script_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    print("=" * 60)
    print(" Downloading Pretrained Model Checkpoints")
    print("=" * 60)

    for idx, (filename, url) in enumerate(WEIGHTS.items(), start=1):
        target_path = os.path.join(ckpt_dir, filename)
        print(f"\n[{idx}/{len(WEIGHTS)}] Checking {filename}...")
        if os.path.exists(target_path):
            file_size_mb = os.path.getsize(target_path) / (1024 * 1024)
            print(f"  -> Already exists ({file_size_mb:.1f} MB). Skipping.")
        else:
            print(f"  -> Downloading from {url}")
            try:
                download_with_progress(url, target_path)
                print(f"  -> Successfully downloaded to {target_path}")
            except Exception as e:
                print(f"  -> Error downloading {filename}: {e}", file=sys.stderr)
                if os.path.exists(target_path):
                    os.remove(target_path)
                sys.exit(1)

    print("\n" + "=" * 60)
    print(f" All checkpoints ready in: {ckpt_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
