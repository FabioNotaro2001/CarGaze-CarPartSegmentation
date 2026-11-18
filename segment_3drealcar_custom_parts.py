"""
segment_3drealcar_custom_parts.py
=================================
Unified multi-view batch segmentation pipeline for vehicle datasets (e.g., 3DRealCar).

This orchestrator coordinates the viewpoint-specific segmentation pipelines
(frontSegmentator, lateralSegmentator, rearSegmentator), dynamically routes images
using zero-shot vision-language classification (CLIP) or metadata rules, and generates
standardized multi-channel outputs:
- Discrete 8-bit binary masks for each detected vehicle component.
- 2D Integer semantic label maps (semantic_custom_ids.png) compatible with benchmark evaluations.
- High-resolution color overlays with legends (semantic_custom_overlay.png).
- Rich structured JSON metadata (metadata.json) detailing detection confidence and pixel counts.
"""
import gc
import importlib.util
import json
import os
import re
import zipfile
import urllib.request
from collections import OrderedDict, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
from transformers import CLIPModel, CLIPProcessor

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import groundingdino.datasets.transforms as T


# =============================================================================
# --- 1. CONFIGURATION ---
# =============================================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_BASE_DIR = os.environ.get(
    "3DREALCAR_DIR",
    os.path.join(PROJECT_DIR, "datasets", "3DRealCar"),
)

DOWNLOAD_3DREALCAR_IF_MISSING = os.environ.get("DOWNLOAD_3DREALCAR_IF_MISSING", "1") == "1"
DOWNLOAD_MODE = os.environ.get("3DREALCAR_DOWNLOAD_MODE", "segment").strip().lower()

GDRIVE_FOLDER_URL = os.environ.get(
    "3DREALCAR_GDRIVE_URL",
    "https://drive.google.com/drive/folders/1csYZdO4sJmxV-tMQRPPVSiSp53_3y6fG?usp=drive_link",
)
GDRIVE_SEGMENT_FILE_ID = os.environ.get(
    "3DREALCAR_SEGMENT_FILE_ID",
    "1IRaTx34ilNN_WUfz3jriPYSUyJsBLW0A",
).strip()
DIRECT_ZIP_URL = os.environ.get("3DREALCAR_ZIP_URL", "").strip()

DATA_SPLITS = [s.strip() for s in os.environ.get("DATA_SPLITS", "val").split(",") if s.strip()]
MAX_IMAGES_PER_SPLIT = int(os.environ.get("MAX_IMAGES_PER_SPLIT", "0"))
MAX_IMAGES_PER_SPLIT = None if MAX_IMAGES_PER_SPLIT <= 0 else MAX_IMAGES_PER_SPLIT

OUTPUT_MASKS_FOLDER = os.environ.get(
    "OUTPUT_MASKS_FOLDER",
    os.path.join(PROJECT_DIR, "SEMANTIC_MASKS_OUTPUT_3DREALCAR_CUSTOM"),
)
OUTPUT_OVERLAYS_FOLDER = os.environ.get(
    "OUTPUT_OVERLAYS_FOLDER",
    os.path.join(PROJECT_DIR, "SEMANTIC_OVERLAYS_OUTPUT_3DREALCAR_CUSTOM"),
)
SAVE_OVERLAYS = os.environ.get("SAVE_OVERLAYS", "1") == "1"
SAVE_LABEL_MAP = os.environ.get("SAVE_LABEL_MAP", "1") == "1"
SAVE_METADATA = os.environ.get("SAVE_METADATA", "1") == "1"

# Force specific viewpoint to bypass CLIP classification (e.g., FORCE_VIEW=front, lateral, rear)
FORCE_VIEW = os.environ.get("FORCE_VIEW", "").strip().lower()
USE_VIEW_ROUTING = os.environ.get("USE_VIEW_ROUTING", "1") == "1"

# Directory containing view-specific segmentator scripts
SEGMENTATOR_DIR = os.environ.get("SEGMENTATOR_DIR", PROJECT_DIR)
FRONT_SEGMENTATOR_PATH = os.environ.get(
    "FRONT_SEGMENTATOR_PATH",
    os.path.join(SEGMENTATOR_DIR, "frontSegmentator.py"),
)
LATERAL_SEGMENTATOR_PATH = os.environ.get(
    "LATERAL_SEGMENTATOR_PATH",
    os.path.join(SEGMENTATOR_DIR, "lateralSegmentator.py"),
)
REAR_SEGMENTATOR_PATH = os.environ.get(
    "REAR_SEGMENTATOR_PATH",
    os.path.join(SEGMENTATOR_DIR, "rearSegmentator.py"),
)

# Pretrained model checkpoints (configurable via environment variables)
SAM2_CHECKPOINT = os.environ.get(
    "SAM2_CHECKPOINT",
    os.path.join(PROJECT_DIR, "checkpoints", "sam2_hiera_large.pt"),
)
SAM2_CONFIG = os.environ.get(
    "SAM2_CONFIG",
    "sam2/sam2_hiera_l.yaml",
)

# Grounding DINO checkpoint
GDINO_CHECKPOINT = os.environ.get(
    "GDINO_CHECKPOINT",
    os.path.join(PROJECT_DIR, "checkpoints", "groundingdino_swint_ogc.pth"),
)
GDINO_CONFIG = os.environ.get("GDINO_CONFIG", "GroundingDINO_SwinT_OGC.py")

BOX_THRESHOLD = float(os.environ.get("BOX_THRESHOLD", "0.25"))
TEXT_THRESHOLD = float(os.environ.get("TEXT_THRESHOLD", "0.20"))

PRED_BACKGROUND_ID = 0


# =============================================================================
# --- 2. CUSTOM CLASS REGISTRY ACROSS ALL VIEWPOINTS ---
# =============================================================================

# Registry of all semantic classes identified across front, lateral, and rear views.
# Integer IDs define pixel values in composite semantic_custom_ids.png maps.
CUSTOM_CLASS_REGISTRY = OrderedDict([
    # Front view classes
    ("car brand badge emblem", {"id": 1, "name": "Emblem", "color": (255, 0, 0), "views": ["front"]}),
    ("car front windshield glass", {"id": 2, "name": "Windshield", "color": (0, 255, 0), "views": ["front"]}),
    ("car hood", {"id": 3, "name": "Hood", "color": (0, 0, 255), "views": ["front"]}),
    ("car left side mirror", {"id": 4, "name": "Left Mirror", "color": (0, 255, 255), "views": ["front"]}),
    ("car right side mirror", {"id": 5, "name": "Right Mirror", "color": (255, 0, 255), "views": ["front"]}),
    ("car left headlight", {"id": 6, "name": "Left Headlight", "color": (255, 255, 0), "views": ["front"]}),
    ("car right headlight", {"id": 7, "name": "Right Headlight", "color": (0, 165, 255), "views": ["front"]}),
    ("car left front tire", {"id": 8, "name": "Left Tire", "color": (128, 0, 128), "views": ["front"]}),
    ("car right front tire", {"id": 9, "name": "Right Tire", "color": (0, 128, 128), "views": ["front"]}),
    ("car left lower bumper light", {"id": 10, "name": "Left Fog Light", "color": (100, 255, 100), "views": ["front"]}),
    ("car right lower bumper light", {"id": 11, "name": "Right Fog Light", "color": (200, 100, 255), "views": ["front"]}),
    ("car front grille", {"id": 12, "name": "Front Grille", "color": (20, 69, 139), "views": ["front"]}),

    # Lateral view classes
    ("front wheel", {"id": 21, "name": "Front Wheel", "color": (0, 255, 0), "views": ["lateral"]}),
    ("rear wheel", {"id": 22, "name": "Rear Wheel", "color": (0, 0, 255), "views": ["lateral"]}),
    ("headlight", {"id": 23, "name": "Headlight", "color": (255, 255, 0), "views": ["lateral"]}),
    ("taillight", {"id": 24, "name": "Taillight", "color": (255, 0, 255), "views": ["lateral"]}),
    ("rearview mirror", {"id": 25, "name": "Rearview Mirror", "color": (0, 165, 255), "views": ["lateral"]}),
    ("front door", {"id": 26, "name": "Front Door", "color": (255, 140, 0), "views": ["lateral"]}),
    ("rear door", {"id": 27, "name": "Rear Door", "color": (128, 255, 0), "views": ["lateral"]}),
    ("front door window", {"id": 28, "name": "Front Door Window", "color": (255, 215, 0), "views": ["lateral"]}),
    ("rear door window", {"id": 29, "name": "Rear Door Window", "color": (0, 191, 255), "views": ["lateral"]}),
    ("front door handle", {"id": 30, "name": "Front Door Handle", "color": (147, 20, 255), "views": ["lateral"]}),
    ("rear door handle", {"id": 31, "name": "Rear Door Handle", "color": (20, 147, 255), "views": ["lateral"]}),
    ("fuel door", {"id": 32, "name": "Fuel Door", "color": (40, 40, 40), "views": ["lateral"]}),
    ("front bumper", {"id": 33, "name": "Front Bumper", "color": (128, 0, 128), "views": ["lateral"]}),
    ("rear bumper", {"id": 34, "name": "Rear Bumper", "color": (0, 128, 128), "views": ["lateral", "rear"]}),

    # Rear view classes
    ("rear windshield", {"id": 41, "name": "Rear Windshield", "color": (0, 255, 0), "views": ["rear"]}),
    ("logo", {"id": 42, "name": "Logo", "color": (0, 0, 255), "views": ["rear"]}),
    ("left mirror", {"id": 43, "name": "Left Mirror", "color": (0, 255, 255), "views": ["rear"]}),
    ("right mirror", {"id": 44, "name": "Right Mirror", "color": (255, 255, 0), "views": ["rear"]}),
    ("left taillight", {"id": 45, "name": "Left Taillight", "color": (255, 0, 255), "views": ["rear"]}),
    ("right taillight", {"id": 46, "name": "Right Taillight", "color": (255, 128, 255), "views": ["rear"]}),
    ("left wheel", {"id": 47, "name": "Left Wheel", "color": (128, 0, 128), "views": ["rear"]}),
    ("right wheel", {"id": 48, "name": "Right Wheel", "color": (0, 128, 128), "views": ["rear"]}),
    ("exhaust tips", {"id": 49, "name": "Exhaust Tips", "color": (255, 0, 0), "views": ["rear"]}),
])

VIEW_CLASS_ORDER = {
    "front": [
        "car brand badge emblem",
        "car front windshield glass",
        "car hood",
        "car left side mirror",
        "car right side mirror",
        "car left headlight",
        "car right headlight",
        "car left front tire",
        "car right front tire",
        "car left lower bumper light",
        "car right lower bumper light",
        "car front grille",
    ],
    "lateral": [
        "front wheel",
        "rear wheel",
        "headlight",
        "taillight",
        "rearview mirror",
        "front door",
        "rear door",
        "front door window",
        "rear door window",
        "front door handle",
        "rear door handle",
        "fuel door",
        "front bumper",
        "rear bumper",
    ],
    "rear": [
        "rear windshield",
        "logo",
        "left mirror",
        "right mirror",
        "left taillight",
        "right taillight",
        "left wheel",
        "right wheel",
        "exhaust tips",
        "rear bumper",
    ],
}

SEGMENTATOR_PATHS = {
    "front": FRONT_SEGMENTATOR_PATH,
    "lateral": LATERAL_SEGMENTATOR_PATH,
    "rear": REAR_SEGMENTATOR_PATH,
}

SEGMENTATOR_METHODS = {
    "front": "predict_hybrid",
    "lateral": "predict_parts",
    "rear": "predict_parts",
}


# =============================================================================
# --- 3. DATASET HANDLING & IMAGE SCANNING ---
# =============================================================================

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_DIR_NAMES = ["images", "image", "imgs", "img", "img_dir", "JPEGImages", "rgb"]
MASK_LIKE_DIR_NAMES = {
    "mask", "masks", "label", "labels", "annotation", "annotations", "ann", "ann_dir",
    "segmentation", "segmentations", "parsing", "parsing_maps", "semantic", "semantics", "depth",
}


def assert_file_exists(path, description):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{description} not found:\n"
            f"  {path}\n\n"
            f"Current working directory:\n"
            f"  {os.getcwd()}\n"
        )


def has_3drealcar_images(dataset_base_dir):
    base = Path(dataset_base_dir)
    if not base.exists():
        return False
    if any(base.rglob("frame_*.jpg")):
        return True
    for img_dir_name in IMAGE_DIR_NAMES:
        if any((base / img_dir_name).rglob("*.jpg")) or any((base / img_dir_name).rglob("*.png")):
            return True
    return any(p.suffix.lower() in IMAGE_EXTS for p in base.rglob("*"))


def download_and_extract_zip(url, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    zip_path = os.path.join(dst_dir, "3drealcar_download.zip")
    print(f"[!] Downloading ZIP archive: {url}")
    urllib.request.urlretrieve(url, zip_path)
    print("[!] Extracting archive in progress...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dst_dir)
    os.remove(zip_path)


def _import_gdown_or_explain():
    try:
        import gdown
        return gdown
    except ImportError as exc:
        raise RuntimeError(
            "gdown is not installed. Please install it via:\n"
            "  pip install gdown\n\n"
            "Oppure scarica manualmente 3DRealCar_Segment_Dataset.zip dalla cartella ufficiale "
            "e imposta:\n"
            "  export 3DREALCAR_DIR=/path/to/3DRealCar_Segment_Dataset"
        ) from exc


def _raise_gdrive_download_help(original_error, file_id=None):
    browser_url = GDRIVE_FOLDER_URL
    if file_id:
        browser_url = f"https://drive.google.com/uc?id={file_id}"

    raise RuntimeError(
        "Download automatico da Google Drive fallito. Google Drive puo' aver imposto quota, "
        "conferma browser, permessi o throttling.\n\n"
        "Recommended manual steps:\n"
        "  1) apri dal browser questo link:\n"
        f"     {browser_url}\n"
        "  2) scarica 3DRealCar_Segment_Dataset.zip manualmente;\n"
        "  3) estrailo;\n"
        "  4) rilancia con:\n"
        "     export 3DREALCAR_DIR=/path/to/3DRealCar\n"
        "     export DOWNLOAD_3DREALCAR_IF_MISSING=0\n\n"
        f"Original error:\n{original_error}"
    )


def download_from_google_drive_segment_file(file_id, dst_dir):
    gdown = _import_gdown_or_explain()
    os.makedirs(dst_dir, exist_ok=True)
    zip_path = os.path.join(dst_dir, "3DRealCar_Segment_Dataset.zip")
    print("[!] Downloading 2D segmentation dataset:")
    print(f"    File ID: {file_id}")
    print(f"    Output : {zip_path}")

    try:
        result = gdown.download(id=file_id, output=zip_path, quiet=False, use_cookies=True)
    except Exception as exc:
        _raise_gdrive_download_help(exc, file_id=file_id)

    if result is None or not os.path.exists(zip_path):
        _raise_gdrive_download_help("gdown.download returned None or failed to create destination archive.", file_id=file_id)

    print("[!] Extracting archive in progress...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dst_dir)
    except zipfile.BadZipFile as exc:
        _raise_gdrive_download_help(
            f"Downloaded file is not a valid zip archive: {zip_path}. Google Drive may have returned an HTML error page.",
            file_id=file_id,
        )

    if os.environ.get("KEEP_3DREALCAR_ZIP", "0") != "1":
        os.remove(zip_path)


def download_from_google_drive_folder(folder_url, dst_dir):
    gdown = _import_gdown_or_explain()
    os.makedirs(dst_dir, exist_ok=True)
    print(f"[!] Downloading 3DRealCar from Google Drive folder:\n    {folder_url}")
    print("[!] Note: Downloading the entire folder downloads multiple large archives and may exceed Google Drive rate quotas.")
    try:
        gdown.download_folder(url=folder_url, output=dst_dir, quiet=False, use_cookies=True)
    except Exception as exc:
        _raise_gdrive_download_help(exc)


def ensure_3drealcar_available():
    if has_3drealcar_images(DATASET_BASE_DIR):
        print(f"[!] Dataset located: {DATASET_BASE_DIR}")
        return

    if not DOWNLOAD_3DREALCAR_IF_MISSING:
        raise FileNotFoundError(
            "3DRealCar dataset not found locally and automatic download is disabled.\n"
            f"Cartella attesa: {DATASET_BASE_DIR}\n\n"
            "Soluzioni:\n"
            "  1) export 3DREALCAR_DIR=/path/to/3DRealCar\n"
            "  2) export DOWNLOAD_3DREALCAR_IF_MISSING=1\n"
            "  3) export 3DREALCAR_ZIP_URL=https://...  se hai un link diretto a uno zip"
        )

    print("[!] 3DRealCar dataset not found locally.")

    if DOWNLOAD_MODE == "none":
        raise FileNotFoundError(
            "3DRealCar dataset not found locally and DOWNLOAD_MODE=none.\n"
            f"Cartella attesa: {DATASET_BASE_DIR}\n"
        )

    if DIRECT_ZIP_URL:
        download_and_extract_zip(DIRECT_ZIP_URL, DATASET_BASE_DIR)
    elif DOWNLOAD_MODE == "segment":
        download_from_google_drive_segment_file(GDRIVE_SEGMENT_FILE_ID, DATASET_BASE_DIR)
    elif DOWNLOAD_MODE == "folder":
        download_from_google_drive_folder(GDRIVE_FOLDER_URL, DATASET_BASE_DIR)
    else:
        raise ValueError(
            f"Invalid 3DREALCAR_DOWNLOAD_MODE: {DOWNLOAD_MODE}. Valid options: segment, folder, none."
        )

    if not has_3drealcar_images(DATASET_BASE_DIR):
        raise FileNotFoundError(
            "Download completed, but no valid vehicle images were found.\n"
            f"Controlla la struttura sotto: {DATASET_BASE_DIR}"
        )


def is_image_file(path):
    p = Path(path)
    if p.suffix.lower() not in IMAGE_EXTS:
        return False
    lowered_parts = {part.lower() for part in p.parts}
    if lowered_parts.intersection(MASK_LIKE_DIR_NAMES):
        return False
    # Prevent reprocessing previous output directories
    output_names = {
        Path(OUTPUT_MASKS_FOLDER).name.lower(),
        Path(OUTPUT_OVERLAYS_FOLDER).name.lower(),
        "semantic_masks_output_3drealcar_custom",
        "semantic_overlays_output_3drealcar_custom",
    }
    if lowered_parts.intersection(output_names):
        return False
    return True


def resolve_split_image_dir(dataset_base_dir, split):
    base = Path(dataset_base_dir)
    split_aliases = [split]
    if split == "valid":
        split_aliases.append("val")
    if split == "val":
        split_aliases.append("valid")

    candidates = []
    for split_name in split_aliases:
        for img_name in IMAGE_DIR_NAMES:
            candidates.append(base / img_name / split_name)
            candidates.append(base / split_name / img_name)
        candidates.extend([
            base / "JPEGImages" / split_name,
            base / "img_dir" / split_name,
        ])

    for img_dir in candidates:
        if img_dir.exists() and any(is_image_file(p) for p in img_dir.rglob("*")):
            return str(img_dir)

    return None


def collect_samples_for_split(dataset_base_dir, split):
    img_dir = resolve_split_image_dir(dataset_base_dir, split)
    samples = []

    if img_dir is not None:
        img_dir_path = Path(img_dir)
        img_files = sorted([p for p in img_dir_path.rglob("*") if is_image_file(p)])
        for img_path in img_files:
            samples.append({"image_path": str(img_path), "split": split})
        return samples, img_dir

    base = Path(dataset_base_dir)
    frame_files = sorted([p for p in base.rglob("frame_*.jpg") if is_image_file(p)])
    if not frame_files:
        frame_files = sorted([p for p in base.rglob("*") if is_image_file(p)])

    for img_path in frame_files:
        samples.append({"image_path": str(img_path), "split": split})

    return samples, None


# =============================================================================
# --- 4. ROBUST MODEL & SEGMENTATOR LOADER ---
# =============================================================================


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def resolve_gdino_config_path():
    if os.path.isabs(GDINO_CONFIG) and os.path.exists(GDINO_CONFIG):
        return GDINO_CONFIG

    import groundingdino

    candidates = [
        os.path.join(os.path.dirname(groundingdino.__file__), "config", GDINO_CONFIG),
        os.path.join(PROJECT_DIR, "GroundingDINO", "groundingdino", "config", GDINO_CONFIG),
        os.path.join(PROJECT_DIR, GDINO_CONFIG),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "GroundingDINO configuration file not found. Searched paths:\n"
        + "\n".join(f"  {p}" for p in candidates)
    )


def resolve_sam2_config_dir_and_name(sam2_cfg):
    if os.path.isabs(sam2_cfg):
        if not os.path.exists(sam2_cfg):
            raise FileNotFoundError(f"SAM2 configuration file not found: {sam2_cfg}")
        p = Path(sam2_cfg)
        parts = list(p.parts)
        if "configs" in parts:
            idx = parts.index("configs")
            cfg_name = str(Path(*parts[idx + 1:]))
            return str(Path(*parts[:idx + 1])), cfg_name
        return str(p.parent), p.name

    import sam2

    cfg_name = sam2_cfg
    if cfg_name.startswith("configs/"):
        cfg_name = cfg_name[len("configs/"):]

    sam2_pkg_dir = os.path.dirname(sam2.__file__)
    candidates = [
        os.path.join(sam2_pkg_dir, "configs"),
        os.path.join(os.path.dirname(sam2_pkg_dir), "configs"),
        os.path.join(os.path.dirname(sam2_pkg_dir), "sam2", "configs"),
        os.path.join(PROJECT_DIR, "segment-anything-2", "sam2", "configs"),
        os.path.join(PROJECT_DIR, "sam2", "configs"),
    ]

    for conf_dir in candidates:
        if os.path.exists(os.path.join(conf_dir, cfg_name)):
            return conf_dir, cfg_name

    # Fallback when SAM2_CONFIG is specified relatively and Hydra resolves it via installed package.
    for conf_dir in candidates:
        if os.path.isdir(conf_dir):
            return conf_dir, cfg_name

    raise FileNotFoundError(
        "SAM2 configuration file not found. Searched paths:\n"
        + "\n".join(f"  {os.path.join(d, cfg_name)}" for d in candidates)
    )


def import_python_file(module_name, file_path):
    assert_file_exists(file_path, f"Script {module_name}")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_gdino_model(gdino_cfg_path, gdino_ckpt_path, device):
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict

    args = SLConfig.fromfile(gdino_cfg_path)
    args.device = device

    model = build_model(args)
    checkpoint = safe_torch_load(gdino_ckpt_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    model = model.to(device)
    return model


def load_sam2_predictor(sam2_cfg, sam2_ckpt_path, device):
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    sam2_conf_dir, sam2_cfg_name = resolve_sam2_config_dir_and_name(sam2_cfg)
    print(f" -> SAM2 config dir : {sam2_conf_dir}")
    print(f" -> SAM2 config     : {sam2_cfg_name}")
    print(f" -> SAM2 checkpoint : {sam2_ckpt_path}")

    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=sam2_conf_dir, version_base="1.2")

    sam2_model = build_sam2(sam2_cfg_name, sam2_ckpt_path, device=device)
    sam2_model.eval()
    return sam2_model, SAM2ImagePredictor(sam2_model)


def instantiate_segmentator_from_module(module, gdino_cfg_path, view_type):
    """
    Instantiate LangSAM2Basic from view module reusing its specific parsing
    methods and heuristics while sharing resolved model instances and paths.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Execution device for {view_type} viewpoint: {device}")

    cls = getattr(module, "LangSAM2Basic")
    segmentator = cls.__new__(cls)
    segmentator.device = device

    print(f"Loading Grounding DINO for {view_type} viewpoint...")
    segmentator.grounding_dino = load_gdino_model(gdino_cfg_path, GDINO_CHECKPOINT, device)

    print(f"Loading SAM 2 for {view_type} viewpoint...")
    segmentator.sam2_model, segmentator.sam2_predictor = load_sam2_predictor(
        SAM2_CONFIG,
        SAM2_CHECKPOINT,
        device,
    )

    segmentator.image_transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Image cache attributes utilized primarily by rearSegmentator
    segmentator._cached_image_id = None
    segmentator._cached_image_np = None
    segmentator._cached_image_gray = None
    segmentator._sam_image_is_set = False

    return segmentator


class ViewSpecificDispatcher:
    def __init__(self, gdino_cfg_path):
        self.gdino_cfg_path = gdino_cfg_path
        self.modules = {}
        self.loaded_segmentator = None
        self.loaded_view = None

    def get_module(self, view_type):
        if view_type not in self.modules:
            module_name = f"custom_{view_type}_segmentator"
            self.modules[view_type] = import_python_file(module_name, SEGMENTATOR_PATHS[view_type])
        return self.modules[view_type]

    def unload_current_segmentator(self):
        if self.loaded_segmentator is not None:
            del self.loaded_segmentator
        self.loaded_segmentator = None
        self.loaded_view = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_view(self, view_type):
        if self.loaded_view == view_type and self.loaded_segmentator is not None:
            return self.loaded_segmentator

        self.unload_current_segmentator()
        module = self.get_module(view_type)
        self.loaded_segmentator = instantiate_segmentator_from_module(module, self.gdino_cfg_path, view_type)
        self.loaded_view = view_type
        return self.loaded_segmentator

    def predict(self, view_type, image_normal, image_white, source_name=None):
        segmentator = self.load_view(view_type)
        method_name = SEGMENTATOR_METHODS[view_type]
        predict_method = getattr(segmentator, method_name)

        if view_type == "front":
            return predict_method(
                image_normal,
                image_white,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
            )

        if view_type == "lateral":
            return predict_method(
                image_normal,
                image_white,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
            )

        if view_type == "rear":
            return predict_method(
                image_normal,
                image_white,
                source_name=source_name,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
            )

        raise ValueError(f"Unsupported viewpoint: {view_type}")

    def save_aggregate_overlay(self, view_type, image_pil, masks, output_path):
        module = self.get_module(view_type)
        if hasattr(module, "generate_summary_image_with_colored_legend"):
            summary_img = module.generate_summary_image_with_colored_legend(image_pil, masks)
        else:
            summary_img = generate_generic_summary_image(image_pil, masks, view_type)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, summary_img)


# =============================================================================
# --- 5. ZERO-SHOT VIEWPOINT CLASSIFIER (CLIP) ---
# =============================================================================

class ViewpointClassifier:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print("Loading zero-shot CLIP viewpoint classifier for dynamic routing...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.labels = [
            "a photo of the front of a car",
            "a photo of the side of a car",
            "a photo of the rear of a car",
        ]
        self.classes = ["front", "lateral", "rear"]
        self.model.eval()

    def classify(self, image_pil):
        inputs = self.processor(
            text=self.labels,
            images=image_pil,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).detach().cpu().numpy()[0]

        best_idx = int(probs.argmax())
        return self.classes[best_idx], float(probs[best_idx])


# =============================================================================
# --- 6. OUTPUT GENERATION (MASKS, OVERLAYS, LABEL MAPS) ---
# =============================================================================


def normalize_for_filename(text):
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "mask"


def torch_mask_to_numpy(mask):
    if mask is None:
        return None
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = np.asarray(mask)

    if mask_np.ndim > 2:
        mask_np = np.squeeze(mask_np)
    return mask_np.astype(bool)


def prepare_dual_images(img_path):
    image_raw = Image.open(img_path)

    if image_raw.mode == "RGBA":
        black_bg = Image.new("RGB", image_raw.size, (0, 0, 0))
        black_bg.paste(image_raw, mask=image_raw.split()[3])
        image_normal = black_bg

        white_bg = Image.new("RGB", image_raw.size, (255, 255, 255))
        white_bg.paste(image_raw, mask=image_raw.split()[3])
        image_white = white_bg
    else:
        image_normal = image_raw.convert("RGB")
        image_white = image_normal.copy()

    return image_normal, image_white


def generate_generic_summary_image(image_pil, masks, view_type):
    img_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    h, w, _ = img_bgr.shape
    background_faded = cv2.addWeighted(
        img_bgr,
        0.4,
        np.full((h, w, 3), 220, dtype=np.uint8),
        0.6,
        0,
    )
    mask_composite = np.zeros((h, w, 3), dtype=np.uint8)

    for class_key in VIEW_CLASS_ORDER.get(view_type, list(masks.keys())):
        if class_key not in masks:
            continue
        mask_np = torch_mask_to_numpy(masks[class_key])
        if mask_np is None or not np.any(mask_np):
            continue
        color = CUSTOM_CLASS_REGISTRY.get(class_key, {}).get("color", (255, 255, 255))
        mask_composite[mask_np] = color

    final_blended = background_faded.copy()
    mask_indices = np.any(mask_composite > 0, axis=-1)
    final_blended[mask_indices] = mask_composite[mask_indices]
    return final_blended


def compose_custom_label_map(masks, view_type, h, w):
    label_map = np.full((h, w), PRED_BACKGROUND_ID, dtype=np.uint8)
    class_order = VIEW_CLASS_ORDER.get(view_type, list(masks.keys()))

    for class_key in class_order:
        if class_key not in masks:
            continue
        mask_np = torch_mask_to_numpy(masks[class_key])
        if mask_np is None or not np.any(mask_np):
            continue
        class_id = int(CUSTOM_CLASS_REGISTRY.get(class_key, {}).get("id", 255))
        if class_id < 0 or class_id > 255:
            raise ValueError(f"Class ID out of uint8 range for {class_key}: {class_id}")
        label_map[mask_np] = class_id

    return label_map


def make_output_subdir(sample, view_type):
    img_path = Path(sample["image_path"])
    split = sample.get("split", "all")

    try:
        rel_parent = img_path.parent.relative_to(Path(DATASET_BASE_DIR))
        rel_safe = "__".join(rel_parent.parts)
    except Exception:
        rel_safe = img_path.parent.name

    img_name = img_path.stem
    return os.path.join(OUTPUT_MASKS_FOLDER, split, view_type, rel_safe, img_name)


def save_prediction_outputs(sample, view_type, image_normal, masks, dispatcher):
    img_path = sample["image_path"]
    img_name = Path(img_path).stem
    w, h = image_normal.size

    out_img_dir = make_output_subdir(sample, view_type)
    os.makedirs(out_img_dir, exist_ok=True)

    saved_classes = []
    for class_key in VIEW_CLASS_ORDER.get(view_type, list(masks.keys())):
        if class_key not in masks:
            continue
        mask_np = torch_mask_to_numpy(masks[class_key])
        if mask_np is None or not np.any(mask_np):
            continue

        safe_name = normalize_for_filename(class_key)
        mask_path = os.path.join(out_img_dir, f"{safe_name}.png")
        cv2.imwrite(mask_path, mask_np.astype(np.uint8) * 255)

        class_cfg = CUSTOM_CLASS_REGISTRY.get(class_key, {})
        saved_classes.append({
            "key": class_key,
            "display_name": class_cfg.get("name", class_key),
            "custom_id": int(class_cfg.get("id", 255)),
            "mask_file": os.path.basename(mask_path),
            "pixels": int(mask_np.sum()),
        })

    if SAVE_LABEL_MAP and saved_classes:
        label_map = compose_custom_label_map(masks, view_type, h, w)
        cv2.imwrite(os.path.join(out_img_dir, "semantic_custom_ids.png"), label_map)

    if SAVE_OVERLAYS and saved_classes:
        overlay_path = os.path.join(
            OUTPUT_OVERLAYS_FOLDER,
            sample.get("split", "all"),
            view_type,
            f"{img_name}_aggregated.png",
        )
        dispatcher.save_aggregate_overlay(view_type, image_normal, masks, overlay_path)

    if SAVE_METADATA:
        metadata = {
            "image_path": img_path,
            "view": view_type,
            "view_confidence": sample.get("view_confidence"),
            "custom_background_id": PRED_BACKGROUND_ID,
            "classes": saved_classes,
        }
        with open(os.path.join(out_img_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    return out_img_dir, saved_classes


def print_class_registry():
    print("\nConfigured semantic classes by viewpoint:")
    for view_type in ["front", "lateral", "rear"]:
        print(f"\n[{view_type.upper()}]")
        for class_key in VIEW_CLASS_ORDER[view_type]:
            cfg = CUSTOM_CLASS_REGISTRY[class_key]
            print(f"  {cfg['id']:3d}: {class_key:34s} -> {cfg['name']}")


def validate_custom_class_registry():
    ids = defaultdict(list)
    for key, cfg in CUSTOM_CLASS_REGISTRY.items():
        ids[int(cfg["id"])].append(key)
    duplicated = {class_id: keys for class_id, keys in ids.items() if len(keys) > 1}
    if duplicated:
        raise ValueError(f"Duplicate custom class IDs detected: {duplicated}")

    missing = []
    for view_type, class_order in VIEW_CLASS_ORDER.items():
        for class_key in class_order:
            if class_key not in CUSTOM_CLASS_REGISTRY:
                missing.append((view_type, class_key))
    if missing:
        raise ValueError(f"Classes in VIEW_CLASS_ORDER not found in registry: {missing}")


# =============================================================================
# --- 7. MAIN PIPELINE EXECUTION ---
# =============================================================================


def classify_samples(samples):
    if FORCE_VIEW:
        if FORCE_VIEW not in {"front", "lateral", "rear"}:
            raise ValueError("FORCE_VIEW must be one of: front, lateral, rear")
        for sample in samples:
            sample["view"] = FORCE_VIEW
            sample["view_confidence"] = 1.0
        print(f"\nViewpoint forced to '{FORCE_VIEW}' via FORCE_VIEW; skipping CLIP classification.")
        return samples

    if not USE_VIEW_ROUTING:
        raise ValueError(
            "USE_VIEW_ROUTING=0 non e' supportato in questa versione: le classi custom sono view-specific. "
            "Usa CLIP oppure imposta FORCE_VIEW=front/lateral/rear."
        )

    classifier = ViewpointClassifier()
    classified = []

    print("\nClassifying viewpoints for dataset images...")
    for idx, sample in enumerate(samples, start=1):
        img_path = sample["image_path"]
        img_file = os.path.basename(img_path)
        try:
            image = Image.open(img_path).convert("RGB")
            view_type, confidence = classifier.classify(image)
            sample["view"] = view_type
            sample["view_confidence"] = confidence
            classified.append(sample)
            print(f"  [{idx}/{len(samples)}] {img_file}: {view_type.upper()} ({confidence * 100:.1f}%)")
        except Exception as exc:
            print(f"  [{idx}/{len(samples)}] {img_file}: viewpoint classification failed -> {exc}")

    del classifier
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return classified


def main():
    validate_custom_class_registry()
    ensure_3drealcar_available()

    print_class_registry()

    print("\nVerifying prerequisite checkpoints and scripts...")
    assert_file_exists(GDINO_CHECKPOINT, "Checkpoint GroundingDINO")
    assert_file_exists(SAM2_CHECKPOINT, "Checkpoint SAM2")
    for view_type, segmentator_path in SEGMENTATOR_PATHS.items():
        assert_file_exists(segmentator_path, f"Script segmentator {view_type}")

    gdino_config_path = resolve_gdino_config_path()
    assert_file_exists(gdino_config_path, "Config GroundingDINO")
    print(f"GroundingDINO config     : {gdino_config_path}")
    print(f"GroundingDINO checkpoint : {GDINO_CHECKPOINT}")

    all_samples = []
    split_payloads = {}
    for split in DATA_SPLITS:
        samples, img_dir = collect_samples_for_split(DATASET_BASE_DIR, split)
        if MAX_IMAGES_PER_SPLIT is not None:
            samples = samples[:MAX_IMAGES_PER_SPLIT]
        split_payloads[split] = {"samples": samples, "img_dir": img_dir}
        all_samples.extend(samples)

    if not all_samples:
        print("\nNo processable images found.")
        return

    print("\nDataset ready:")
    for split, payload in split_payloads.items():
        img_dir = payload["img_dir"]
        samples = payload["samples"]
        print("\n" + "=" * 80)
        print(f"SPLIT / SCAN: {split}")
        print("=" * 80)
        if img_dir:
            print(f"Image directory: {img_dir}")
        else:
            print(f"No split subfolder found: recursively scanning {DATASET_BASE_DIR}")
        print(f"Images identified: {len(samples)}")

    all_samples = classify_samples(all_samples)
    if not all_samples:
        print("\nNo images were successfully classified.")
        return

    os.makedirs(OUTPUT_MASKS_FOLDER, exist_ok=True)
    if SAVE_OVERLAYS:
        os.makedirs(OUTPUT_OVERLAYS_FOLDER, exist_ok=True)

    grouped_by_view = defaultdict(list)
    for sample in all_samples:
        grouped_by_view[sample["view"]].append(sample)

    print("\nViewpoint Routing Summary:")
    for view_type in ["front", "lateral", "rear"]:
        print(f"  {view_type:7s}: {len(grouped_by_view.get(view_type, []))} images")

    dispatcher = ViewSpecificDispatcher(gdino_cfg_path=gdino_config_path)
    total_processed = 0
    total_masks = 0

    # Group by viewpoint to load Grounding DINO and SAM 2 once per viewpoint mode and reuse specialized heuristic pipeline
    for view_type in ["front", "lateral", "rear"]:
        samples = grouped_by_view.get(view_type, [])
        if not samples:
            continue

        print("\n" + "=" * 80)
        print(f"PROCESSING VIEWPOINT: {view_type.upper()} ({len(samples)} images)")
        print("=" * 80)

        # Preload viewpoint segmentator to catch any checkpoint or config issues early
        dispatcher.load_view(view_type)

        for idx, sample in enumerate(samples, start=1):
            img_path = sample["image_path"]
            img_file = os.path.basename(img_path)
            confidence = sample.get("view_confidence")
            confidence_text = "N/A" if confidence is None else f"{confidence * 100:.1f}%"

            print("\n" + "-" * 80)
            print(f"[{idx}/{len(samples)}] Processing: {img_file}")
            print(f" -> Image path : {img_path}")
            print(f" -> Viewpoint  : {view_type.upper()} ({confidence_text})")

            try:
                image_normal, image_white = prepare_dual_images(img_path)
                masks = dispatcher.predict(
                    view_type=view_type,
                    image_normal=image_normal,
                    image_white=image_white,
                    source_name=img_path,
                )

                if not masks:
                    print(" -> No components detected.")
                    total_processed += 1
                    del image_normal, image_white, masks
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

                out_dir, saved_classes = save_prediction_outputs(
                    sample=sample,
                    view_type=view_type,
                    image_normal=image_normal,
                    masks=masks,
                    dispatcher=dispatcher,
                )

                total_processed += 1
                total_masks += len(saved_classes)
                if saved_classes:
                    found_names = ", ".join(item["display_name"] for item in saved_classes)
                    print(f" -> Saved classes : {len(saved_classes)} | {found_names}")
                    print(f" -> Output dir    : {out_dir}")
                else:
                    print(" -> Masks produced but none were non-empty/valid.")

                del image_normal, image_white, masks
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as exc:
                print(f" -> ERROR on {img_file}: {exc}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        dispatcher.unload_current_segmentator()

    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE - MASKS AND OVERLAYS GENERATED")
    print("=" * 80)
    print(f"Processed images : {total_processed}")
    print(f"Saved masks      : {total_masks}")
    print(f"Masks output dir : {OUTPUT_MASKS_FOLDER}")
    if SAVE_OVERLAYS:
        print(f"Overlays output  : {OUTPUT_OVERLAYS_FOLDER}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("TF32 acceleration enabled.")

    main()
