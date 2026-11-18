"""
rearSegmentator.py
==================
Automated semantic part segmentation pipeline for rear-view vehicle images.

This module combines Grounding DINO (open-vocabulary object detection) and
Segment Anything 2 (SAM 2, high-precision promptable segmentation) with
silhouette boundary analysis, bilateral reflection scoring, darkness/contrast
metrics, and domain-specific geometric priors for rear automotive anatomy.

Key Methodological Highlights:
- Vehicle Silhouette & Bounding Box: Determines vehicle outer bounds from foreground
  separation to anchor all relative horizontal/vertical coordinates.
- Rear Windshield & Dynamic Logo Query: Segments rear windshield, validates geometry,
  and builds vehicle identity queries (brand, emblem, text wordmark) from path metadata.
- Bilateral Mirror Reflection Analysis: Matches rear-view mirror candidates using
  mirrored shape similarity and bilateral symmetry axis projection.
- Taillight & Lightbar Detection: Disambiguates discrete bilateral taillight pairs vs.
  continuous horizontal lightbars using redness metrics, height-to-width ratios, and
  bilateral symmetry matching.
- Wheel Darkness & Geometry Fallback: Isolates visible rear wheel contact patches
  using darkness and vertical positioning, with symmetry recovery for partially obscured wheels.
- Exhaust Tips Clustering: Detects and clusters lateral or central exhaust tips
  using darkness, circularity, border contrast, and bilateral pairing.
- Rear Bumper Delimitation: Bounds rear bumper beneath taillights and windshield,
  subtracting wheels and exhaust tips for a precise semantic partition.
"""
import gc
import os

import cv2
import numpy as np
from PIL import Image
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from groundingdino.util.inference import predict as predict_gdino
import groundingdino.datasets.transforms as T

# =============================================================================
# --- 1. BASE CONFIGURATION ---
# =============================================================================
DATASET_FOLDER = "IMAGE DATASET"
OUTPUT_MASKS_FOLDER = "SEMANTIC_MASKS"

SAM2_CHECKPOINT = "sam2_hiera_large.pt"
SAM2_CONFIG = "sam2/sam2_hiera_l.yaml"
GDINO_CHECKPOINT = "groundingdino_swint_ogc.pth"
GDINO_CONFIG = "GroundingDINO_SwinT_OGC.py"

BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.20
WINDSHIELD_MAX_VEHICLE_HEIGHT_RATIO = 0.66
WINDSHIELD_MAX_VEHICLE_BOTTOM_RATIO = 0.62
WINDSHIELD_MAX_LEFT_OFFSET_RATIO = 0.18
LOGO_EMBLEM_MAX_WIDTH_RATIO = 0.16
LOGO_EMBLEM_MAX_HEIGHT_RATIO = 0.16
LOGO_TEXT_MAX_WIDTH_RATIO = 0.42
LOGO_TEXT_MAX_HEIGHT_RATIO = 0.12
LOGO_MAX_CENTER_OFFSET_RATIO = 0.32
LOGO_MAX_BAND_FROM_WINDSHIELD_RATIO = 0.30
VEHICLE_MASK_THRESHOLD = 8
MIRROR_MAX_VEHICLE_WIDTH_RATIO = 0.20
MIRROR_MAX_VEHICLE_HEIGHT_RATIO = 0.20
MIRROR_MAX_VEHICLE_AREA_RATIO = 0.030
MIRROR_MIN_SIDE_OFFSET_RATIO = 0.18
MIRROR_MAX_BOTTOM_RATIO = 0.62
MIRROR_MAX_BELOW_LOGO_RATIO = 0.10
MIRROR_MAX_CANDIDATES_PER_SIDE = 4
MIRROR_SUPPORT_MAX_X_RATIO = 0.14
MIRROR_SUPPORT_MAX_Y_RATIO = 0.08
MIRROR_SUPPORT_MAX_SIZE_DELTA_RATIO = 0.55
TAILLIGHT_MIN_SIDE_OFFSET_RATIO = 0.12
TAILLIGHT_MAX_SIDE_OFFSET_RATIO = 0.50
TAILLIGHT_MIN_VEHICLE_WIDTH_RATIO = 0.03
TAILLIGHT_MIN_VEHICLE_HEIGHT_RATIO = 0.02
TAILLIGHT_MAX_VEHICLE_WIDTH_RATIO = 0.30
TAILLIGHT_MAX_VEHICLE_HEIGHT_RATIO = 0.22
TAILLIGHT_MAX_VEHICLE_AREA_RATIO = 0.050
TAILLIGHT_MIN_TOP_RATIO = 0.14
TAILLIGHT_MAX_BOTTOM_RATIO = 0.58
TAILLIGHT_MAX_VERTICAL_FROM_LOGO_RATIO = 0.26
TAILLIGHT_MAX_CANDIDATES_PER_SIDE = 5
TAILLIGHT_BAR_MIN_VEHICLE_WIDTH_RATIO = 0.40
TAILLIGHT_BAR_MAX_VEHICLE_WIDTH_RATIO = 0.96
TAILLIGHT_BAR_MAX_VEHICLE_HEIGHT_RATIO = 0.14
TAILLIGHT_BAR_MAX_VERTICAL_FROM_LOGO_RATIO = 0.18
TAILLIGHT_BAR_CENTER_GAP_RATIO = 0.05
TAILLIGHT_MIN_REDNESS = 0.05
TAILLIGHT_PAIR_MIN_SCORE = 0.55
TAILLIGHT_MIRROR_MAX_OVERLAP_RATIO = 0.08
TAILLIGHT_FALLBACK_MAX_VEHICLE_WIDTH_RATIO = 0.55
TAILLIGHT_MIRROR_FALLBACK_MAX_OVERLAP_RATIO = 0.25
WHEEL_MIN_SIDE_OFFSET_RATIO = 0.28
WHEEL_MAX_SIDE_OFFSET_RATIO = 0.52
WHEEL_MIN_VEHICLE_WIDTH_RATIO = 0.025
WHEEL_MAX_VEHICLE_WIDTH_RATIO = 0.22
WHEEL_MIN_VEHICLE_HEIGHT_RATIO = 0.10
WHEEL_MAX_VEHICLE_HEIGHT_RATIO = 0.38
WHEEL_MAX_VEHICLE_AREA_RATIO = 0.060
WHEEL_MIN_TOP_RATIO = 0.54
WHEEL_MIN_BOTTOM_RATIO = 0.82
WHEEL_MAX_VERTICAL_FROM_TARGET_RATIO = 0.18
WHEEL_MAX_OUTER_EDGE_GAP_RATIO = 0.14
WHEEL_MAX_ASPECT_RATIO = 1.10
WHEEL_MIN_DARKNESS = 0.24
WHEEL_MAX_CANDIDATES_PER_SIDE = 4
WHEEL_FALLBACK_CENTER_OFFSET_RATIO = 0.39
WHEEL_FALLBACK_CENTER_Y_RATIO = 0.84
WHEEL_FALLBACK_BOX_WIDTH_RATIO = 0.18
WHEEL_FALLBACK_BOX_HEIGHT_RATIO = 0.30
WHEEL_DARK_FALLBACK_THRESHOLD = 96
EXHAUST_BOX_MIN_VEHICLE_WIDTH_RATIO = 0.02
EXHAUST_BOX_MIN_VEHICLE_HEIGHT_RATIO = 0.02
EXHAUST_BOX_MAX_VEHICLE_WIDTH_RATIO = 0.55
EXHAUST_BOX_MAX_VEHICLE_HEIGHT_RATIO = 0.24
EXHAUST_BOX_MAX_VEHICLE_AREA_RATIO = 0.060
EXHAUST_BOX_MIN_TOP_RATIO = 0.16
EXHAUST_BOX_MAX_BOTTOM_RATIO = 0.96
EXHAUST_COMPONENT_MIN_VEHICLE_WIDTH_RATIO = 0.010
EXHAUST_COMPONENT_MIN_VEHICLE_HEIGHT_RATIO = 0.010
EXHAUST_COMPONENT_MAX_VEHICLE_WIDTH_RATIO = 0.14
EXHAUST_COMPONENT_MAX_VEHICLE_HEIGHT_RATIO = 0.12
EXHAUST_COMPONENT_MAX_VEHICLE_AREA_RATIO = 0.010
EXHAUST_CENTRAL_MIN_VEHICLE_AREA_RATIO = 0.0010
EXHAUST_MAX_ASPECT_RATIO = 2.60
EXHAUST_CORE_DARK_THRESHOLD = 120
EXHAUST_MIN_DARKNESS = 0.34
EXHAUST_MIN_BORDER_CONTRAST = 2.0
EXHAUST_MIN_CIRCULARITY = 0.18
EXHAUST_MAX_BOX_BORDER_TOUCH_RATIO = 0.18
EXHAUST_MAX_WHEEL_OVERLAP_RATIO = 0.12
EXHAUST_MAX_LIGHT_OVERLAP_RATIO = 0.10
EXHAUST_MAX_LOGO_OVERLAP_RATIO = 0.12
EXHAUST_MAX_VERTICAL_GROUP_GAP_RATIO = 0.18
EXHAUST_MAX_CANDIDATES = 6
EXHAUST_LATERAL_MIN_SIDE_OFFSET_RATIO = 0.18
EXHAUST_LATERAL_MIN_BOTTOM_RATIO = 0.66
EXHAUST_CENTRAL_MAX_CENTER_OFFSET_RATIO = 0.18
EXHAUST_CENTRAL_MAX_ABOVE_LOGO_RATIO = 0.26
EXHAUST_CENTRAL_MAX_BELOW_LOGO_RATIO = 0.22
EXHAUST_FALLBACK_LATERAL_SEARCH_WIDTH_RATIO = 0.34
EXHAUST_FALLBACK_LATERAL_TOP_RATIO = 0.54
EXHAUST_FALLBACK_SIDE_CLUSTER_CENTER_OFFSET_RATIO = 0.28
EXHAUST_FALLBACK_SIDE_CLUSTER_CENTER_Y_RATIO = 0.82
EXHAUST_FALLBACK_SIDE_CLUSTER_WIDTH_RATIO = 0.22
EXHAUST_FALLBACK_SIDE_CLUSTER_HEIGHT_RATIO = 0.18
EXHAUST_FALLBACK_CENTRAL_WIDTH_RATIO = 0.34
EXHAUST_FALLBACK_CENTRAL_HEIGHT_RATIO = 0.22
EXHAUST_SYMMETRY_MAX_X_RATIO = 0.10
EXHAUST_SYMMETRY_MAX_Y_RATIO = 0.08
BUMPER_FALLBACK_MIN_TOP_RATIO = 0.58
BUMPER_MIN_TOP_BELOW_LOGO_RATIO = 0.08
MANUAL_TAILLIGHT_BOX_WIDTH_RATIO = 0.22
MANUAL_TAILLIGHT_BOX_HEIGHT_RATIO = 0.14
MANUAL_TAILLIGHT_BOX_MIN_WIDTH = 84.0
MANUAL_TAILLIGHT_BOX_MIN_HEIGHT = 42.0
MANUAL_EXHAUST_BOX_WIDTH_RATIO = 0.10
MANUAL_EXHAUST_BOX_HEIGHT_RATIO = 0.12
MANUAL_EXHAUST_BOX_MIN_WIDTH = 54.0
MANUAL_EXHAUST_BOX_MIN_HEIGHT = 54.0

MANUAL_COMPONENT_OVERRIDES = [
    {
        "brand": "polestar",
        "model": "4",
        "taillight": {
            "mode": "replace",
            "points": {
                "left taillight": (551, 769),
                "right taillight": (1985, 769),
            },
        },
    },
    {
        "brand": "ferrari",
        "model": "849",
        "taillight": {
            "mode": "replace",
            "points": {
                "left taillight": (428, 899),
                "right taillight": (2150, 899),
            },
        },
        "exhaust tips": {
            "mode": "merge",
            "points": [
                (957, 1137),
            ],
        },
    },
    {
        "brand": "peugeot",
        "model": "208",
        "exhaust tips": {
            "mode": "replace",
            "points": [
                (1209, 1104),
                (1266, 1104),
            ],
        },
    },
    {
        "brand": "lamborghini",
        "model": "revuelto",
        "exhaust tips": {
            "mode": "replace",
            "points": [
                (1721, 1254),
                (2396, 1246),
            ],
        },
    },
]

CLASS_MAP = {
    "rear windshield": ["Rear Windshield", (0, 255, 0)],      # Verde
    "logo": ["Logo", (0, 0, 255)],                            # Rosso
    "left mirror": ["Left Mirror", (0, 255, 255)],           # Giallo
    "right mirror": ["Right Mirror", (255, 255, 0)],         # Ciano
    "left taillight": ["Left Taillight", (255, 0, 255)],     # Magenta
    "right taillight": ["Right Taillight", (255, 128, 255)], # Rosa
    "left wheel": ["Left Wheel", (128, 0, 128)],             # Viola
    "right wheel": ["Right Wheel", (0, 128, 128)],           # Ottanio
    "exhaust tips": ["Exhaust Tips", (255, 0, 0)],           # Blu
    "rear bumper": ["Rear Bumper", (0, 165, 255)],           # Arancione
}

PART_SPECS = {
    "rear windshield": {
        "source": "normal",
        "captions": [
            "car rear windshield glass",
            "rear windshield glass",
            "rear window glass",
            "rear glass",
            "back window",
        ],
    },
    "logo": {
        "source": "normal",
        "emblem_captions": [
            "car brand badge emblem",
            "car logo emblem",
            "brand badge emblem",
            "cross emblem",
            "cross logo",
            "star emblem",
            "star logo",
        ],
        "text_captions": [
            "car brand lettering",
            "car model lettering",
            "car badge text",
            "rear badge text",
            "brand wordmark",
            "model wordmark",
        ],
    },
    "mirror": {
        "source": "normal",
        "captions": [
            "car side mirror",
            "car wing mirror",
            "car rearview mirror",
            "car door mirror",
            "vehicle side mirror housing",
            "outside rearview mirror",
        ],
    },
    "taillight": {
        "source": "normal",
        "captions": [
            "car taillight",
            "tail light",
            "rear light",
            "rear lamp",
            "tail lamp",
            "brake light",
            "rear position light",
            "rear parking light",
            "rear combination lamp",
            "rear led strip",
            "tail light strip",
            "rear light strip",
            "taillight bar",
            "tail light bar",
            "rear light bar",
            "rear light signature",
            "led taillight",
        ],
    },
    "wheel": {
        "source": "white",
        "captions": [
            "car rear tire",
            "car rear wheel",
            "rear tire",
            "rear wheel",
            "car tire",
            "car wheel",
            "vehicle tire",
            "vehicle wheel",
            "rear tyre",
            "car rear tyre",
            "tire tread",
            "wheel tread",
        ],
    },
    "exhaust tips": {
        "source": "normal",
        "captions": [
            "car exhaust tip",
            "exhaust tip",
            "car exhaust outlet",
            "exhaust outlet",
            "rear exhaust outlet",
            "tailpipe",
            "dual exhaust tip",
            "dual tailpipe",
            "dual center exhaust",
            "center exhaust outlet",
            "central exhaust tip",
            "dual round exhaust tips",
            "twin round exhaust tips",
            "double exhaust outlet",
            "hexagonal center exhaust",
            "dual hexagonal center exhaust",
            "upper center dual exhaust",
            "top center exhaust outlet",
            "upper rear exhaust outlet",
            "muffler tip",
            "round exhaust tip",
            "hexagonal exhaust outlet",
            "hexagonal tailpipe",
        ],
    },
    "rear bumper": {
        "source": "normal",
        "captions": [
            "car rear bumper",
            "rear bumper",
            "vehicle rear bumper",
            "rear bumper cover",
            "car back bumper",
            "bumper",
        ],
    },
}

def normalize_source_name(source_name):
    if not source_name:
        return ""
    return str(source_name).replace("\\", "/").lower()

def normalize_text_term(value):
    if not value:
        return ""
    cleaned = str(value).lower().replace("_", " ").replace("-", " ")
    return " ".join(cleaned.split())

def unique_nonempty(values):
    seen = set()
    unique_values = []
    for value in values:
        normalized = normalize_text_term(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_values.append(normalized)
    return unique_values

def strip_trailing_year(model_name):
    tokens = normalize_text_term(model_name).split()
    if tokens and tokens[-1].isdigit() and len(tokens[-1]) == 4:
        tokens = tokens[:-1]
    return " ".join(tokens)

def parse_vehicle_identity(source_name):
    normalized_path = normalize_source_name(source_name)
    path_parts = [part for part in normalized_path.split("/") if part]
    if len(path_parts) < 4:
        return "", ""

    brand_name = normalize_text_term(path_parts[-3])
    model_name = strip_trailing_year(path_parts[-2])
    return brand_name, model_name

def source_matches_vehicle(source_name, brand_name, model_fragment):
    source_brand, source_model = parse_vehicle_identity(source_name)
    target_brand = normalize_text_term(brand_name)
    target_model = normalize_text_term(model_fragment)

    if target_brand and source_brand != target_brand:
        return False
    if target_model and target_model not in source_model:
        return False
    return True

def expand_search_terms(term):
    term = normalize_text_term(term)
    if not term:
        return []

    expanded_terms = [term]

    if "&" in term:
        expanded_terms.extend([
            term.replace("&", " & "),
            term.replace("&", " and "),
        ])

    return unique_nonempty(expanded_terms)

def build_logo_query_terms(source_name):
    brand_name, model_name = parse_vehicle_identity(source_name)
    all_terms = []

    if brand_name:
        all_terms.extend(expand_search_terms(brand_name))

    if model_name:
        all_terms.extend(expand_search_terms(model_name))
        for token in model_name.split():
            if token == "model":
                continue
            if token.isdigit() and len(token) > 4:
                continue
            if len(token) < 2 and not token.isdigit():
                continue
            all_terms.extend(expand_search_terms(token))

    return unique_nonempty(all_terms)

def build_logo_emblem_captions(source_name):
    captions = list(PART_SPECS["logo"]["emblem_captions"])
    for term in build_logo_query_terms(source_name):
        captions.extend([
            f"{term} emblem",
            f"{term} badge emblem",
            f"{term} logo",
        ])
    return unique_nonempty(captions)

def build_logo_text_captions(source_name):
    captions = list(PART_SPECS["logo"]["text_captions"])
    for term in build_logo_query_terms(source_name):
        captions.extend([
            term,
            f"{term} lettering",
            f"{term} badge text",
            f"{term} text",
            f"{term} wordmark",
        ])
    return unique_nonempty(captions)

# =============================================================================
# --- 2. VISUALIZATION HELPERS ---
# =============================================================================
def generate_summary_image_with_colored_legend(image_pil, unique_masks):
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

    for prompt, mask in reversed(list(unique_masks.items())):
        if prompt not in CLASS_MAP:
            continue
        mask_np = mask.cpu().numpy().astype(np.uint8)
        color = CLASS_MAP[prompt][1]
        mask_composite[mask_np > 0] = color

    final_blended = background_faded.copy()
    mask_indices = np.any(mask_composite > 0, axis=-1)
    final_blended[mask_indices] = mask_composite[mask_indices]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    line_spacing = 40
    entry_height = 30

    max_text_width = 0
    items_to_draw = []

    for prompt in unique_masks:
        if prompt in CLASS_MAP:
            legend_name = CLASS_MAP[prompt][0]
            color = CLASS_MAP[prompt][1]
            (tw, _), _ = cv2.getTextSize(legend_name, font, font_scale, thickness)
            max_text_width = max(max_text_width, tw)
            items_to_draw.append((legend_name, color))

    if not items_to_draw:
        return background_faded

    legend_width = max_text_width + 60
    legend_canvas = np.full((h, legend_width, 3), 220, dtype=np.uint8)

    current_y = 20 + entry_height
    for name, color in items_to_draw:
        cv2.rectangle(legend_canvas, (10, current_y - 20), (30, current_y), color, -1)
        cv2.putText(
            legend_canvas,
            name,
            (40, current_y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        current_y += line_spacing

    return np.hstack((final_blended, legend_canvas))

# =============================================================================
# --- 3. LangSAM2 WRAPPER ---
# =============================================================================
class LangSAM2Basic:
    def __init__(self, sam2_cfg, sam2_ckpt, gdino_cfg, gdino_ckpt):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Loading Grounding DINO model...")
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.models import build_model
        from groundingdino.util.utils import clean_state_dict

        args = SLConfig.fromfile(gdino_cfg)
        args.device = self.device
        self.grounding_dino = build_model(args)
        checkpoint = torch.load(gdino_ckpt, map_location="cpu")
        self.grounding_dino.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        self.grounding_dino.eval()
        self.grounding_dino = self.grounding_dino.to(self.device)

        print("Loading Segment Anything 2 (SAM 2) model...")
        from hydra.core.global_hydra import GlobalHydra
        from hydra import initialize_config_dir
        import sam2

        GlobalHydra.instance().clear()
        sam2_pkg_dir = os.path.dirname(sam2.__file__)
        conf_dir = os.path.join(sam2_pkg_dir, "configs")
        if not os.path.exists(conf_dir):
            conf_dir = os.path.join(os.path.dirname(sam2_pkg_dir), "sam2_configs")

        initialize_config_dir(config_dir=conf_dir, version_base="1.2")
        self.sam2_model = build_sam2(sam2_cfg, sam2_ckpt, device=self.device)
        self.sam2_model.eval()
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
        self.image_transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self._cached_image_id = None
        self._cached_image_np = None
        self._cached_image_gray = None
        self._sam_image_is_set = False

    def clear_cached_image(self):
        self._cached_image_id = None
        self._cached_image_np = None
        self._cached_image_gray = None
        self._sam_image_is_set = False

    def prepare_cached_image(self, image_pil):
        if image_pil is None:
            return None

        image_id = id(image_pil)
        if self._cached_image_id != image_id:
            self._cached_image_id = image_id
            self._cached_image_np = np.array(image_pil)
            self._cached_image_gray = None
            self._sam_image_is_set = False

        return self._cached_image_np

    def get_image_np(self, image_pil):
        return self.prepare_cached_image(image_pil)

    def get_image_gray(self, image_pil):
        image_np = self.prepare_cached_image(image_pil)
        if image_np is None:
            return None

        if self._cached_image_gray is None:
            self._cached_image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        return self._cached_image_gray

    def ensure_sam_image(self, image_pil):
        image_np = self.prepare_cached_image(image_pil)
        if image_np is None:
            return None

        if not self._sam_image_is_set:
            self.sam2_predictor.set_image(image_np)
            self._sam_image_is_set = True

        return image_np

    def scale_boxes_to_image(self, boxes, w, h):
        if len(boxes) == 0:
            return np.empty((0, 4), dtype=np.float32)

        boxes = boxes * torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes.cpu().numpy()

    def get_box_center(self, box):
        return (
            float((box[0] + box[2]) / 2.0),
            float((box[1] + box[3]) / 2.0),
        )

    def get_box_area(self, box):
        return float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))

    def get_box_size(self, box):
        return (
            float(max(0.0, box[2] - box[0])),
            float(max(0.0, box[3] - box[1])),
        )

    def clip_box_to_image(self, box, image_size):
        if box is None or image_size is None:
            return None

        image_w, image_h = image_size
        x1 = float(np.clip(box[0], 0.0, max(0.0, image_w - 1.0)))
        y1 = float(np.clip(box[1], 0.0, max(0.0, image_h - 1.0)))
        x2 = float(np.clip(box[2], 0.0, max(0.0, image_w - 1.0)))
        y2 = float(np.clip(box[3], 0.0, max(0.0, image_h - 1.0)))
        if x2 <= x1 or y2 <= y1:
            return None
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def make_box_from_center(self, center_x, center_y, width, height, image_size):
        box = np.array(
            [
                float(center_x) - float(width) / 2.0,
                float(center_y) - float(height) / 2.0,
                float(center_x) + float(width) / 2.0,
                float(center_y) + float(height) / 2.0,
            ],
            dtype=np.float32,
        )
        return self.clip_box_to_image(box, image_size)

    def calculate_iou(self, box_a, box_b):
        x_left = max(box_a[0], box_b[0])
        y_top = max(box_a[1], box_b[1])
        x_right = min(box_a[2], box_b[2])
        y_bottom = min(box_a[3], box_b[3])

        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = self.get_box_area(box_a) + self.get_box_area(box_b) - intersection
        if union <= 0:
            return 0.0
        return float(intersection / union)

    def deduplicate_candidates(self, candidates, iou_threshold=0.35):
        unique_candidates = []
        for box, logit, caption in sorted(candidates, key=lambda item: item[1], reverse=True):
            if any(self.calculate_iou(box, kept_box) > iou_threshold for kept_box, _, _ in unique_candidates):
                continue
            unique_candidates.append((box, float(logit), caption))
        return unique_candidates

    def get_mask_bbox(self, mask):
        if mask is None:
            return None

        mask_np = mask.cpu().numpy().astype(bool)
        ys, xs = np.where(mask_np)
        if len(xs) == 0 or len(ys) == 0:
            return None

        return np.array(
            [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
            dtype=np.float32,
        )

    def get_central_mask_component_bbox(self, mask, reference_x=None):
        if mask is None:
            return None

        mask_np = mask.cpu().numpy().astype(np.uint8)
        if not np.any(mask_np):
            return None

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
        if num_labels <= 1:
            return self.get_mask_bbox(mask)

        if reference_x is None:
            full_bbox = self.get_mask_bbox(mask)
            if full_bbox is None:
                return None
            reference_x = self.get_box_center(full_bbox)[0]

        best_bbox = None
        best_key = None

        for label_idx in range(1, num_labels):
            x = stats[label_idx, cv2.CC_STAT_LEFT]
            y = stats[label_idx, cv2.CC_STAT_TOP]
            width = stats[label_idx, cv2.CC_STAT_WIDTH]
            height = stats[label_idx, cv2.CC_STAT_HEIGHT]
            area = stats[label_idx, cv2.CC_STAT_AREA]
            if area <= 0:
                continue

            component_bbox = np.array(
                [float(x), float(y), float(x + width - 1), float(y + height - 1)],
                dtype=np.float32,
            )
            component_cx, _ = self.get_box_center(component_bbox)
            component_key = (abs(component_cx - reference_x), -float(area))

            if best_key is None or component_key < best_key:
                best_key = component_key
                best_bbox = component_bbox

        return best_bbox

    def get_manual_component_override(self, source_name, part_name):
        if not source_name or not part_name:
            return None

        for override in MANUAL_COMPONENT_OVERRIDES:
            if not source_matches_vehicle(source_name, override.get("brand"), override.get("model")):
                continue

            part_override = override.get(part_name)
            if part_override:
                return part_override

        return None

    def mask_contains_point(self, mask, point):
        if mask is None or point is None:
            return False

        mask_np = mask.cpu().numpy().astype(bool)
        if not np.any(mask_np):
            return False

        point_x = int(np.clip(np.rint(float(point[0])), 0, mask_np.shape[1] - 1))
        point_y = int(np.clip(np.rint(float(point[1])), 0, mask_np.shape[0] - 1))
        return bool(mask_np[point_y, point_x])

    def restrict_mask_to_box(self, mask, box):
        if mask is None or box is None:
            return None

        result = mask.bool().clone()
        if not torch.any(result):
            return None

        height, width = result.shape
        x1 = int(np.clip(np.floor(box[0]), 0, max(0, width - 1)))
        y1 = int(np.clip(np.floor(box[1]), 0, max(0, height - 1)))
        x2 = int(np.clip(np.ceil(box[2]), 0, max(0, width - 1)))
        y2 = int(np.clip(np.ceil(box[3]), 0, max(0, height - 1)))
        if x2 <= x1 or y2 <= y1:
            return None

        box_mask = torch.zeros_like(result, dtype=torch.bool)
        box_mask[y1:y2 + 1, x1:x2 + 1] = True
        result = torch.logical_and(result, box_mask)
        if not torch.any(result):
            return None

        return result

    def build_search_box_from_point(
        self,
        point,
        image_size,
        vehicle_bbox,
        width_ratio,
        height_ratio,
        min_width,
        min_height,
    ):
        if point is None or image_size is None:
            return None

        reference_width = float(image_size[0])
        reference_height = float(image_size[1])
        if vehicle_bbox is not None:
            reference_width = max(1.0, float(vehicle_bbox[2]) - float(vehicle_bbox[0]) + 1.0)
            reference_height = max(1.0, float(vehicle_bbox[3]) - float(vehicle_bbox[1]) + 1.0)

        return self.make_box_from_center(
            float(point[0]),
            float(point[1]),
            max(float(min_width), width_ratio * reference_width),
            max(float(min_height), height_ratio * reference_height),
            image_size,
        )

    def extract_mask_component_near_point(self, mask, point):
        if mask is None or point is None:
            return None

        mask_np = mask.cpu().numpy().astype(np.uint8)
        if not np.any(mask_np):
            return None

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
        if num_labels <= 1:
            return mask.bool()

        point_x = float(point[0])
        point_y = float(point[1])
        pixel_x = int(np.clip(np.rint(point_x), 0, labels.shape[1] - 1))
        pixel_y = int(np.clip(np.rint(point_y), 0, labels.shape[0] - 1))
        target_label = int(labels[pixel_y, pixel_x])
        if target_label > 0:
            return torch.from_numpy(labels == target_label)

        best_mask = None
        best_key = None
        for label_idx in range(1, num_labels):
            area = stats[label_idx, cv2.CC_STAT_AREA]
            if area <= 0:
                continue

            x = float(stats[label_idx, cv2.CC_STAT_LEFT])
            y = float(stats[label_idx, cv2.CC_STAT_TOP])
            width = float(stats[label_idx, cv2.CC_STAT_WIDTH])
            height = float(stats[label_idx, cv2.CC_STAT_HEIGHT])
            x2 = x + width - 1.0
            y2 = y + height - 1.0
            component_bbox = np.array([x, y, x2, y2], dtype=np.float32)
            component_cx, component_cy = self.get_box_center(component_bbox)

            dx = 0.0 if x <= point_x <= x2 else min(abs(point_x - x), abs(point_x - x2))
            dy = 0.0 if y <= point_y <= y2 else min(abs(point_y - y), abs(point_y - y2))
            component_key = (
                float(np.hypot(dx, dy)),
                float(np.hypot(component_cx - point_x, component_cy - point_y)),
                -float(area),
            )

            if best_key is None or component_key < best_key:
                best_key = component_key
                best_mask = torch.from_numpy(labels == label_idx)

        return best_mask

    def select_mask_near_point(self, masks, point):
        if not masks or point is None:
            return None

        best_mask = None
        best_key = None
        for mask in masks:
            component_mask = self.extract_mask_component_near_point(mask, point)
            if component_mask is None or not torch.any(component_mask):
                continue

            component_bbox = self.get_mask_bbox(component_mask)
            if component_bbox is None:
                continue

            component_cx, component_cy = self.get_box_center(component_bbox)
            component_key = (
                0 if self.mask_contains_point(component_mask, point) else 1,
                float(np.hypot(component_cx - float(point[0]), component_cy - float(point[1]))),
                -float(self.get_mask_area(component_mask)),
            )
            if best_key is None or component_key < best_key:
                best_key = component_key
                best_mask = component_mask.bool()

        return best_mask

    def build_manual_taillight_masks_from_points(self, image_normal, image_size, vehicle_bbox, points_by_prompt):
        manual_masks = {}
        for prompt, point in (points_by_prompt or {}).items():
            search_box = self.build_search_box_from_point(
                point,
                image_size,
                vehicle_bbox,
                MANUAL_TAILLIGHT_BOX_WIDTH_RATIO,
                MANUAL_TAILLIGHT_BOX_HEIGHT_RATIO,
                MANUAL_TAILLIGHT_BOX_MIN_WIDTH,
                MANUAL_TAILLIGHT_BOX_MIN_HEIGHT,
            )
            if search_box is None:
                continue

            sam_mask = self.predict_mask_from_box(image_normal, search_box)
            localized_mask = self.restrict_mask_to_box(sam_mask, search_box)
            component_mask = self.extract_mask_component_near_point(localized_mask, point)
            if component_mask is None:
                component_mask = self.extract_mask_component_near_point(sam_mask, point)
            if component_mask is None or not torch.any(component_mask):
                continue

            manual_masks[prompt] = component_mask.bool()

        return manual_masks

    def build_manual_exhaust_mask_from_points(
        self,
        image_normal,
        image_size,
        vehicle_mask,
        vehicle_bbox,
        points,
        logo_mask=None,
        taillight_masks=None,
        mirror_masks=None,
        wheel_masks=None,
    ):
        manual_mask = None

        for point in points or []:
            search_box = self.build_search_box_from_point(
                point,
                image_size,
                vehicle_bbox,
                MANUAL_EXHAUST_BOX_WIDTH_RATIO,
                MANUAL_EXHAUST_BOX_HEIGHT_RATIO,
                MANUAL_EXHAUST_BOX_MIN_WIDTH,
                MANUAL_EXHAUST_BOX_MIN_HEIGHT,
            )
            if search_box is None:
                continue

            component_mask = self.select_mask_near_point(
                self.extract_exhaust_candidate_masks(
                    image_normal,
                    search_box,
                    None,
                    vehicle_mask,
                    caption="fallback manual exhaust point",
                ),
                point,
            )

            if component_mask is None:
                sam_mask = self.predict_mask_from_box(image_normal, search_box)
                component_mask = self.select_mask_near_point(
                    self.extract_exhaust_candidate_masks(
                        image_normal,
                        search_box,
                        sam_mask,
                        vehicle_mask,
                        caption="manual exhaust point",
                    ),
                    point,
                )
                if component_mask is None:
                    localized_mask = self.restrict_mask_to_box(sam_mask, search_box)
                    component_mask = self.extract_mask_component_near_point(localized_mask, point)

            if component_mask is None or not torch.any(component_mask):
                continue

            if manual_mask is None:
                manual_mask = component_mask.bool()
            else:
                manual_mask = torch.logical_or(manual_mask, component_mask.bool())

        if manual_mask is None or not torch.any(manual_mask):
            return None

        masks_to_remove = []
        if logo_mask is not None:
            masks_to_remove.append(logo_mask)
        masks_to_remove.extend((taillight_masks or {}).values())
        masks_to_remove.extend((mirror_masks or {}).values())
        masks_to_remove.extend((wheel_masks or {}).values())
        manual_mask = self.subtract_masks(manual_mask, masks_to_remove)
        if manual_mask is None or not torch.any(manual_mask):
            return None

        return manual_mask.bool()

    def compute_mask_redness(self, image_pil, mask):
        if image_pil is None or mask is None:
            return 0.0

        mask_np = mask.cpu().numpy().astype(bool)
        if not np.any(mask_np):
            return 0.0

        image_np = self.get_image_np(image_pil)
        pixels = image_np[mask_np]
        if pixels.size == 0:
            return 0.0

        red = pixels[:, 0].astype(np.float32)
        green = pixels[:, 1].astype(np.float32)
        blue = pixels[:, 2].astype(np.float32)

        red_pixels = (red >= 70.0) & (red >= green + 16.0) & (red >= blue + 16.0)
        red_ratio = float(red_pixels.mean())
        red_margin = np.maximum(red - np.maximum(green, blue), 0.0) / 255.0
        margin_score = float(red_margin.mean())

        return 0.65 * red_ratio + 0.35 * margin_score

    def compute_mask_darkness(self, image_pil, mask):
        if image_pil is None or mask is None:
            return 0.0

        mask_np = mask.cpu().numpy().astype(bool)
        if not np.any(mask_np):
            return 0.0

        gray = self.get_image_gray(image_pil)
        pixels = gray[mask_np].astype(np.float32)
        if pixels.size == 0:
            return 0.0

        dark_ratio = float((pixels <= 90.0).mean())
        shadow_ratio = float((pixels <= 128.0).mean())
        inverse_brightness = float(np.maximum(255.0 - pixels, 0.0).mean() / 255.0)

        return 0.50 * dark_ratio + 0.15 * shadow_ratio + 0.35 * inverse_brightness

    def compute_mask_border_contrast(self, image_pil, mask):
        if image_pil is None or mask is None:
            return 0.0

        mask_np = mask.cpu().numpy().astype(np.uint8)
        if not np.any(mask_np):
            return 0.0

        gray = self.get_image_gray(image_pil)
        border_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated_mask = cv2.dilate(mask_np, border_kernel, iterations=1)
        border_mask = np.logical_and(dilated_mask > 0, mask_np == 0)
        inner_mask = mask_np > 0

        if not np.any(border_mask) or not np.any(inner_mask):
            return 0.0

        border_mean = float(gray[border_mask].astype(np.float32).mean())
        inner_mean = float(gray[inner_mask].astype(np.float32).mean())
        return border_mean - inner_mean

    def compute_mask_circularity(self, mask):
        if mask is None:
            return 0.0

        mask_np = mask.cpu().numpy().astype(np.uint8)
        if not np.any(mask_np):
            return 0.0

        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0

        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        if area <= 0.0 or perimeter <= 0.0:
            return 0.0

        return float((4.0 * np.pi * area) / (perimeter * perimeter))

    def get_mask_box_border_touch_ratio(self, mask, box):
        if mask is None or box is None:
            return 1.0

        mask_np = mask.cpu().numpy().astype(bool)
        if not np.any(mask_np):
            return 1.0

        h, w = mask_np.shape
        x1 = int(np.clip(np.floor(box[0]), 0, max(0, w - 1)))
        y1 = int(np.clip(np.floor(box[1]), 0, max(0, h - 1)))
        x2 = int(np.clip(np.ceil(box[2]), 0, max(0, w - 1)))
        y2 = int(np.clip(np.ceil(box[3]), 0, max(0, h - 1)))
        if x2 <= x1 or y2 <= y1:
            return 1.0

        border_mask = np.zeros_like(mask_np, dtype=bool)
        border_mask[y1:y2 + 1, x1] = True
        border_mask[y1:y2 + 1, x2] = True
        border_mask[y1, x1:x2 + 1] = True
        border_mask[y2, x1:x2 + 1] = True

        mask_pixels = int(mask_np.sum())
        if mask_pixels <= 0:
            return 1.0

        border_pixels = int(np.logical_and(mask_np, border_mask).sum())
        return float(border_pixels) / float(mask_pixels)

    def get_mask_area(self, mask):
        if mask is None:
            return 0
        return int(mask.bool().sum().item())

    def get_mask_overlap_ratio(self, mask_a, mask_b):
        if mask_a is None or mask_b is None:
            return 0.0

        mask_a = mask_a.bool()
        mask_b = mask_b.bool()
        if not torch.any(mask_a) or not torch.any(mask_b):
            return 0.0

        area_a = self.get_mask_area(mask_a)
        if area_a <= 0:
            return 0.0

        overlap_area = self.get_mask_area(torch.logical_and(mask_a, mask_b))
        return float(overlap_area) / float(area_a)

    def build_vehicle_mask(self, image_normal):
        image_np = self.get_image_np(image_normal)
        raw_mask = np.any(image_np > VEHICLE_MASK_THRESHOLD, axis=-1).astype(np.uint8)

        if not np.any(raw_mask):
            return None

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
        if num_labels <= 1:
            return torch.from_numpy(raw_mask.astype(bool))

        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return torch.from_numpy(labels == largest_label)

    def predict_mask_from_box(self, image_pil, box):
        if box is None:
            return None

        self.ensure_sam_image(image_pil)
        with torch.inference_mode():
            masks, _, _ = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=np.array([box]),
                multimask_output=False,
            )
        return torch.from_numpy(masks).squeeze(1).any(dim=0)

    def collect_candidates(
        self,
        transformed_image,
        captions,
        image_size,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    ):
        w, h = image_size
        candidates = []
        normalized_captions = unique_nonempty(captions)

        with torch.inference_mode():
            for caption in normalized_captions:
                boxes, logits, _ = predict_gdino(
                    model=self.grounding_dino,
                    image=transformed_image,
                    caption=caption,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    device=self.device,
                )

                if len(boxes) == 0:
                    continue

                boxes_np = self.scale_boxes_to_image(boxes, w, h)
                candidates.extend(
                    (boxes_np[i], float(logits[i]), caption)
                    for i in range(len(boxes_np))
                )

        return self.deduplicate_candidates(candidates)

    def get_best_box(self, transformed_image, captions, image_size, box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD):
        candidates = self.collect_candidates(
            transformed_image,
            captions,
            image_size,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        if not candidates:
            return None

        return max(candidates, key=lambda item: item[1])[0]

    def get_bumper_min_top_from_taillights(self, taillight_masks):
        if not taillight_masks:
            return None

        taillight_bottoms = []
        for mask in taillight_masks.values():
            taillight_bbox = self.get_mask_bbox(mask)
            if taillight_bbox is None:
                continue
            taillight_bottoms.append(float(taillight_bbox[3]))

        if not taillight_bottoms:
            return None

        return max(taillight_bottoms)

    def get_bumper_min_top(self, vehicle_bbox, taillight_masks=None, logo_bbox=None):
        taillight_min_top = self.get_bumper_min_top_from_taillights(taillight_masks)
        if taillight_min_top is not None:
            return taillight_min_top

        if vehicle_bbox is None:
            return None

        vehicle_top = float(vehicle_bbox[1])
        vehicle_height = max(1.0, vehicle_bbox[3] - vehicle_bbox[1] + 1.0)
        fallback_min_top = vehicle_top + BUMPER_FALLBACK_MIN_TOP_RATIO * vehicle_height

        if logo_bbox is None:
            return fallback_min_top

        logo_bottom = float(logo_bbox[3])
        logo_safe_min_top = logo_bottom + BUMPER_MIN_TOP_BELOW_LOGO_RATIO * vehicle_height
        return max(fallback_min_top, logo_safe_min_top)

    def trim_mask_above_y(self, mask, min_y):
        if mask is None or min_y is None:
            return mask

        mask = mask.bool().clone()
        if not torch.any(mask):
            return None

        cutoff = int(np.clip(np.floor(min_y), 0, mask.shape[0]))
        if cutoff > 0:
            mask[:cutoff, :] = False

        if not torch.any(mask):
            return None

        return mask

    def subtract_masks(self, base_mask, masks_to_remove):
        if base_mask is None:
            return None

        result = base_mask.bool().clone()
        for mask in masks_to_remove or []:
            if mask is None:
                continue

            mask = mask.bool()
            if not torch.any(mask):
                continue

            result = torch.logical_and(result, torch.logical_not(mask))

        if not torch.any(result):
            return None

        return result

    def build_bumper_mask(
        self,
        image_normal,
        transformed_image,
        image_size,
        vehicle_bbox=None,
        logo_bbox=None,
        taillight_masks=None,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    ):
        bumper_candidates = self.collect_candidates(
            transformed_image,
            PART_SPECS["rear bumper"]["captions"],
            image_size,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        if not bumper_candidates:
            return None

        bumper_box = max(bumper_candidates, key=lambda item: item[1])[0]

        min_top = self.get_bumper_min_top(
            vehicle_bbox,
            taillight_masks=taillight_masks,
            logo_bbox=logo_bbox,
        )
        if min_top is not None:
            bumper_box = self.clip_box_to_image(
                np.array(
                    [
                        bumper_box[0],
                        max(float(bumper_box[1]), float(min_top)),
                        bumper_box[2],
                        bumper_box[3],
                    ],
                    dtype=np.float32,
                ),
                image_size,
            )
            if bumper_box is None:
                return None

        bumper_mask = self.predict_mask_from_box(image_normal, bumper_box)
        bumper_mask = self.trim_mask_above_y(bumper_mask, min_top)
        if bumper_mask is None or not torch.any(bumper_mask):
            return None

        return bumper_mask

    def is_valid_windshield_mask(self, mask, vehicle_bbox):
        mask_bbox = self.get_mask_bbox(mask)
        if mask_bbox is None or vehicle_bbox is None:
            return False

        vehicle_height = max(1.0, vehicle_bbox[3] - vehicle_bbox[1] + 1.0)
        mask_height = max(1.0, mask_bbox[3] - mask_bbox[1] + 1.0)
        _, mask_cy = self.get_box_center(mask_bbox)

        return (
            mask_height <= WINDSHIELD_MAX_VEHICLE_HEIGHT_RATIO * vehicle_height and
            mask_bbox[3] <= vehicle_bbox[1] + WINDSHIELD_MAX_VEHICLE_BOTTOM_RATIO * vehicle_height and
            mask_cy <= vehicle_bbox[1] + 0.50 * vehicle_height
        )

    def is_windshield_mask_too_far_left(self, mask, vehicle_bbox):
        mask_bbox = self.get_mask_bbox(mask)
        if mask_bbox is None or vehicle_bbox is None:
            return False

        vehicle_width = max(1.0, vehicle_bbox[2] - vehicle_bbox[0] + 1.0)
        vehicle_cx, _ = self.get_box_center(vehicle_bbox)
        mask_cx, _ = self.get_box_center(mask_bbox)

        return mask_cx < (vehicle_cx - WINDSHIELD_MAX_LEFT_OFFSET_RATIO * vehicle_width)

    def build_dark_windshield_fallback_mask(self, image_normal, vehicle_mask):
        vehicle_bbox = self.get_mask_bbox(vehicle_mask)
        if vehicle_bbox is None:
            return None

        gray = self.get_image_gray(image_normal)
        vehicle_mask_np = vehicle_mask.cpu().numpy().astype(bool)

        x1, y1, x2, y2 = [int(v) for v in vehicle_bbox]
        vehicle_width = max(1, x2 - x1 + 1)
        vehicle_height = max(1, y2 - y1 + 1)
        vehicle_cx = (x1 + x2) / 2.0

        row_indices = np.arange(gray.shape[0])[:, None]
        upper_limit = y1 + int(0.58 * vehicle_height)
        search_mask = vehicle_mask_np & (row_indices >= y1) & (row_indices <= upper_limit)
        dark_mask = (gray < 110) & search_mask

        dark_mask_u8 = (dark_mask.astype(np.uint8) * 255)
        kernel = np.ones((5, 5), dtype=np.uint8)
        dark_mask_u8 = cv2.morphologyEx(dark_mask_u8, cv2.MORPH_OPEN, kernel)
        dark_mask_u8 = cv2.morphologyEx(dark_mask_u8, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask_u8, connectivity=8)
        best_component_mask = None
        best_score = None

        for label_idx in range(1, num_labels):
            x = stats[label_idx, cv2.CC_STAT_LEFT]
            y = stats[label_idx, cv2.CC_STAT_TOP]
            width = stats[label_idx, cv2.CC_STAT_WIDTH]
            height = stats[label_idx, cv2.CC_STAT_HEIGHT]
            area = stats[label_idx, cv2.CC_STAT_AREA]

            component_box = np.array(
                [float(x), float(y), float(x + width - 1), float(y + height - 1)],
                dtype=np.float32,
            )
            component_cx, component_cy = self.get_box_center(component_box)

            if width < 0.18 * vehicle_width or height < 0.08 * vehicle_height:
                continue
            if width > 0.90 * vehicle_width or height > 0.55 * vehicle_height:
                continue
            if component_cy > y1 + 0.46 * vehicle_height:
                continue
            if abs(component_cx - vehicle_cx) > 0.20 * vehicle_width:
                continue

            score = (
                float(area)
                - 1.5 * abs(component_cx - vehicle_cx)
                - 0.6 * abs(component_cy - (y1 + 0.28 * vehicle_height))
            )

            if best_score is None or score > best_score:
                best_score = score
                best_component_mask = labels == label_idx

        if best_component_mask is None:
            return None

        return torch.from_numpy(best_component_mask)

    def get_windshield_candidate_rejection_reasons(self, mask_bbox, vehicle_bbox):
        if mask_bbox is None:
            return ["missing mask bounding box"]
        if vehicle_bbox is None:
            return ["missing vehicle bounding box"]

        vehicle_height = max(1.0, vehicle_bbox[3] - vehicle_bbox[1] + 1.0)
        mask_height = max(1.0, mask_bbox[3] - mask_bbox[1] + 1.0)
        mask_cx, mask_cy = self.get_box_center(mask_bbox)
        vehicle_cx, _ = self.get_box_center(vehicle_bbox)
        reasons = []

        if mask_height > WINDSHIELD_MAX_VEHICLE_HEIGHT_RATIO * vehicle_height:
            reasons.append(
                f"mask height exceeds threshold ({mask_height / vehicle_height:.3f} > {WINDSHIELD_MAX_VEHICLE_HEIGHT_RATIO:.3f})"
            )
        if mask_bbox[3] > vehicle_bbox[1] + WINDSHIELD_MAX_VEHICLE_BOTTOM_RATIO * vehicle_height:
            reasons.append(
                f"mask bottom exceeds threshold ({(mask_bbox[3] - vehicle_bbox[1]) / vehicle_height:.3f} > {WINDSHIELD_MAX_VEHICLE_BOTTOM_RATIO:.3f})"
            )
        if mask_cy > vehicle_bbox[1] + 0.50 * vehicle_height:
            reasons.append(
                f"vertical center exceeds threshold ({(mask_cy - vehicle_bbox[1]) / vehicle_height:.3f} > 0.500)"
            )
        if mask_cx < (vehicle_cx - WINDSHIELD_MAX_LEFT_OFFSET_RATIO * max(1.0, vehicle_bbox[2] - vehicle_bbox[0] + 1.0)):
            reasons.append("mask positioned too far to the left")

        if not reasons:
            reasons.append("outside valid and relaxed criteria")

        return reasons

    def get_best_windshield_result(
        self,
        image_normal,
        transformed_image,
        image_size,
        vehicle_mask,
        captions,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    ):
        candidates = self.collect_candidates(
            transformed_image,
            captions,
            image_size,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        vehicle_bbox = self.get_mask_bbox(vehicle_mask)
        valid_results = []
        relaxed_results = []

        for candidate_idx, (box, logit, caption) in enumerate(
            sorted(candidates, key=lambda item: item[1], reverse=True),
            start=1,
        ):
            mask = self.predict_mask_from_box(image_normal, box)
            mask_bbox = self.get_mask_bbox(mask)
            if mask_bbox is None:
                continue

            if self.is_valid_windshield_mask(mask, vehicle_bbox):
                valid_results.append((float(logit), box, mask))
                continue

            if vehicle_bbox is not None:
                vehicle_height = max(1.0, vehicle_bbox[3] - vehicle_bbox[1] + 1.0)
                _, mask_cy = self.get_box_center(mask_bbox)
                if (
                    mask_bbox[3] <= vehicle_bbox[1] + 0.68 * vehicle_height and
                    mask_cy <= vehicle_bbox[1] + 0.54 * vehicle_height
                ):
                    mask_height = max(1.0, mask_bbox[3] - mask_bbox[1] + 1.0)
                    relaxed_results.append((mask_height, -float(logit), box, mask))
                    continue

        if valid_results:
            valid_results.sort(key=lambda item: item[0], reverse=True)
            _, best_box, best_mask = valid_results[0]
            if self.is_windshield_mask_too_far_left(best_mask, vehicle_bbox):
                return None, None
            return best_box, best_mask

        if relaxed_results:
            relaxed_results.sort(key=lambda item: (item[0], item[1]))
            _, _, best_box, best_mask = relaxed_results[0]
            if self.is_windshield_mask_too_far_left(best_mask, vehicle_bbox):
                return None, None
            return best_box, best_mask

        fallback_mask = self.build_dark_windshield_fallback_mask(image_normal, vehicle_mask)
        if fallback_mask is not None:
            if self.is_windshield_mask_too_far_left(fallback_mask, vehicle_bbox):
                return None, None
            return self.get_mask_bbox(fallback_mask), fallback_mask

        return None, None

    def get_logo_band(self, vehicle_bbox, windshield_box, image_height):
        if vehicle_bbox is None:
            return 0.20 * image_height, 0.60 * image_height

        vehicle_top = float(vehicle_bbox[1])
        vehicle_height = max(1.0, vehicle_bbox[3] - vehicle_bbox[1] + 1.0)
        band_top = vehicle_top + 0.12 * vehicle_height
        band_bottom = vehicle_top + 0.58 * vehicle_height

        if windshield_box is not None:
            windshield_bottom = float(windshield_box[3])
            band_top = max(vehicle_top, windshield_bottom - 0.03 * vehicle_height)
            band_bottom = min(
                float(vehicle_bbox[3]),
                windshield_bottom + LOGO_MAX_BAND_FROM_WINDSHIELD_RATIO * vehicle_height,
            )

        return band_top, band_bottom

    def select_logo_emblem_box(self, candidates, vehicle_bbox, windshield_box, image_size):
        w, h = image_size
        img_cx = w / 2.0
        band_top, band_bottom = self.get_logo_band(vehicle_bbox, windshield_box, h)
        valid_candidates = []
        target_y = (band_top + band_bottom) / 2.0

        if vehicle_bbox is not None and windshield_box is not None:
            vehicle_height = max(1.0, vehicle_bbox[3] - vehicle_bbox[1] + 1.0)
            target_y = float(windshield_box[3]) + 0.14 * vehicle_height
            band_bottom = min(band_bottom, float(windshield_box[3]) + 0.24 * vehicle_height)

        for box, logit, _ in candidates:
            box_w = float(max(0.0, box[2] - box[0]))
            box_h = float(max(0.0, box[3] - box[1]))
            box_cx, box_cy = self.get_box_center(box)

            if box_w > LOGO_EMBLEM_MAX_WIDTH_RATIO * w or box_h > LOGO_EMBLEM_MAX_HEIGHT_RATIO * h:
                continue
            if abs(box_cx - img_cx) > 0.16 * w:
                continue
            if box_cy < band_top or box_cy > band_bottom:
                continue

            score = (
                float(logit)
                - 1.8 * abs(box_cx - img_cx) / max(1.0, w)
                - 2.2 * abs(box_cy - target_y) / max(1.0, h)
            )
            valid_candidates.append((score, box))

        if not valid_candidates:
            fallback_candidates = []

            if vehicle_bbox is None:
                return None

            vehicle_top = float(vehicle_bbox[1])
            vehicle_height = max(1.0, vehicle_bbox[3] - vehicle_bbox[1] + 1.0)
            fallback_top = vehicle_top + 0.26 * vehicle_height
            fallback_bottom = vehicle_top + 0.64 * vehicle_height
            fallback_target_y = vehicle_top + 0.46 * vehicle_height

            for box, logit, _ in candidates:
                box_w = float(max(0.0, box[2] - box[0]))
                box_h = float(max(0.0, box[3] - box[1]))
                box_cx, box_cy = self.get_box_center(box)

                if box_w > LOGO_EMBLEM_MAX_WIDTH_RATIO * w or box_h > LOGO_EMBLEM_MAX_HEIGHT_RATIO * h:
                    continue
                if abs(box_cx - img_cx) > 0.12 * w:
                    continue
                if box_cy < fallback_top or box_cy > fallback_bottom:
                    continue

                score = (
                    float(logit)
                    - 2.4 * abs(box_cx - img_cx) / max(1.0, w)
                    - 1.4 * abs(box_cy - fallback_target_y) / max(1.0, h)
                )
                fallback_candidates.append((score, box))

            if not fallback_candidates:
                return None

            fallback_candidates.sort(key=lambda item: item[0], reverse=True)
            return fallback_candidates[0][1]

        valid_candidates.sort(key=lambda item: item[0], reverse=True)
        return valid_candidates[0][1]

    def select_logo_text_boxes(self, candidates, vehicle_bbox, windshield_box, image_size, anchor_x):
        w, h = image_size
        band_top, band_bottom = self.get_logo_band(vehicle_bbox, windshield_box, h)
        valid_candidates = []
        target_y = (band_top + band_bottom) / 2.0

        if vehicle_bbox is not None and windshield_box is not None:
            vehicle_height = max(1.0, vehicle_bbox[3] - vehicle_bbox[1] + 1.0)
            target_y = float(windshield_box[3]) + 0.16 * vehicle_height

        for box, logit, _ in candidates:
            box_w = float(max(0.0, box[2] - box[0]))
            box_h = float(max(0.0, box[3] - box[1]))
            box_cx, box_cy = self.get_box_center(box)

            if box_w <= 0.01 * w or box_h <= 0.008 * h:
                continue
            if box_w > LOGO_TEXT_MAX_WIDTH_RATIO * w or box_h > LOGO_TEXT_MAX_HEIGHT_RATIO * h:
                continue
            if abs(box_cx - anchor_x) > LOGO_MAX_CENTER_OFFSET_RATIO * w:
                continue
            if box_cy < band_top or box_cy > band_bottom:
                continue
            if box[3] > band_bottom:
                continue

            score = (
                float(logit)
                - 1.0 * abs(box_cx - anchor_x) / max(1.0, w)
                - 1.4 * abs(box_cy - target_y) / max(1.0, h)
            )
            valid_candidates.append((score, box))

        valid_candidates.sort(key=lambda item: item[0], reverse=True)
        selected_boxes = []

        for logit, box in valid_candidates:
            if any(self.calculate_iou(box, kept_box) > 0.20 for kept_box in selected_boxes):
                continue
            selected_boxes.append(box)

        return selected_boxes

    def build_logo_mask(
        self,
        image_normal,
        transformed_image,
        image_size,
        source_name,
        vehicle_bbox=None,
        windshield_box=None,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    ):
        emblem_candidates = self.collect_candidates(
            transformed_image,
            build_logo_emblem_captions(source_name),
            image_size,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        text_candidates = self.collect_candidates(
            transformed_image,
            build_logo_text_captions(source_name),
            image_size,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        if not emblem_candidates and not text_candidates:
            return None

        emblem_box = self.select_logo_emblem_box(
            emblem_candidates,
            vehicle_bbox,
            windshield_box,
            image_size,
        )
        anchor_x = self.get_box_center(emblem_box)[0] if emblem_box is not None else image_size[0] / 2.0
        text_boxes = self.select_logo_text_boxes(
            text_candidates,
            vehicle_bbox,
            windshield_box,
            image_size,
            anchor_x,
        )

        selected_boxes = []
        if emblem_box is not None:
            selected_boxes.append(emblem_box)
        selected_boxes.extend(text_boxes)

        if not selected_boxes:
            return None

        combined_mask = None
        for box in selected_boxes:
            mask = self.predict_mask_from_box(image_normal, box)
            if combined_mask is None:
                combined_mask = mask.bool()
            else:
                combined_mask = torch.logical_or(combined_mask, mask.bool())

        return combined_mask

    def get_mirror_anchor(self, vehicle_bbox, logo_bbox):
        if logo_bbox is not None:
            return self.get_box_center(logo_bbox)

        if vehicle_bbox is None:
            return None, None

        vehicle_cx, _ = self.get_box_center(vehicle_bbox)
        vehicle_height = max(1.0, vehicle_bbox[3] - vehicle_bbox[1] + 1.0)
        anchor_y = float(vehicle_bbox[1]) + 0.36 * vehicle_height
        return vehicle_cx, anchor_y

    def get_taillight_anchor(self, vehicle_bbox, logo_bbox):
        if logo_bbox is not None:
            return self.get_box_center(logo_bbox)

        if vehicle_bbox is None:
            return None, None

        vehicle_cx, _ = self.get_box_center(vehicle_bbox)
        vehicle_height = max(1.0, vehicle_bbox[3] - vehicle_bbox[1] + 1.0)
        anchor_y = float(vehicle_bbox[1]) + 0.44 * vehicle_height
        return vehicle_cx, anchor_y

    def compute_reflected_shape_similarity(self, left_mask, right_mask, target_size=40):
        left_bbox = self.get_mask_bbox(left_mask)
        right_bbox = self.get_mask_bbox(right_mask)
        if left_bbox is None or right_bbox is None:
            return 0.0

        left_np = left_mask.cpu().numpy().astype(np.uint8)
        right_np = right_mask.cpu().numpy().astype(np.uint8)

        lx1, ly1, lx2, ly2 = [int(v) for v in left_bbox]
        rx1, ry1, rx2, ry2 = [int(v) for v in right_bbox]

        left_crop = left_np[ly1:ly2 + 1, lx1:lx2 + 1]
        right_crop = right_np[ry1:ry2 + 1, rx1:rx2 + 1]
        if left_crop.size == 0 or right_crop.size == 0:
            return 0.0

        left_small = cv2.resize(left_crop, (target_size, target_size), interpolation=cv2.INTER_NEAREST) > 0
        right_small = cv2.resize(right_crop, (target_size, target_size), interpolation=cv2.INTER_NEAREST) > 0
        left_small = np.fliplr(left_small)

        union = np.logical_or(left_small, right_small).sum()
        if union == 0:
            return 0.0

        intersection = np.logical_and(left_small, right_small).sum()
        return float(intersection / union)

    def compute_mirror_shape_similarity(self, left_mask, right_mask, target_size=40):
        return self.compute_reflected_shape_similarity(
            left_mask,
            right_mask,
            target_size=target_size,
        )

    def get_mirror_mask_rejection_reasons(self, mask_bbox, mask_pixels, vehicle_bbox, logo_bbox):
        if mask_bbox is None or vehicle_bbox is None:
            return ["missing mask bounding box"]

        axis_x, anchor_y = self.get_mirror_anchor(vehicle_bbox, logo_bbox)
        if axis_x is None or anchor_y is None:
            return ["missing mirror reference axis"]

        vehicle_width = max(1.0, vehicle_bbox[2] - vehicle_bbox[0] + 1.0)
        vehicle_height = max(1.0, vehicle_bbox[3] - vehicle_bbox[1] + 1.0)
        vehicle_area = vehicle_width * vehicle_height
        max_bottom = float(vehicle_bbox[1]) + MIRROR_MAX_BOTTOM_RATIO * vehicle_height
        min_top = float(vehicle_bbox[1]) + 0.04 * vehicle_height

        mask_cx, mask_cy = self.get_box_center(mask_bbox)
        mask_w, mask_h = self.get_box_size(mask_bbox)
        reasons = []

        if abs(mask_cx - axis_x) < MIRROR_MIN_SIDE_OFFSET_RATIO * vehicle_width:
            reasons.append(
                f"mask too close to vehicle center axis ({abs(mask_cx - axis_x) / vehicle_width:.3f} < {MIRROR_MIN_SIDE_OFFSET_RATIO:.3f})"
            )
        if mask_w > MIRROR_MAX_VEHICLE_WIDTH_RATIO * vehicle_width:
            reasons.append(
                f"mask width exceeds threshold ({mask_w / vehicle_width:.3f} > {MIRROR_MAX_VEHICLE_WIDTH_RATIO:.3f})"
            )
        if mask_h > MIRROR_MAX_VEHICLE_HEIGHT_RATIO * vehicle_height:
            reasons.append(
                f"mask height exceeds threshold ({mask_h / vehicle_height:.3f} > {MIRROR_MAX_VEHICLE_HEIGHT_RATIO:.3f})"
            )
        if mask_pixels > MIRROR_MAX_VEHICLE_AREA_RATIO * vehicle_area:
            reasons.append(
                f"mask area exceeds threshold ({mask_pixels / vehicle_area:.3f} > {MIRROR_MAX_VEHICLE_AREA_RATIO:.3f})"
            )
        if mask_cy < min_top or mask_cy > max_bottom:
            reasons.append(
                f"mask vertical center outside valid band (cy={mask_cy:.1f}, band=[{min_top:.1f}, {max_bottom:.1f}])"
            )
        if logo_bbox is not None and mask_cy > anchor_y + MIRROR_MAX_BELOW_LOGO_RATIO * vehicle_height:
            reasons.append(
                f"mask positioned too far below logo anchor (cy={mask_cy:.1f} > {anchor_y + MIRROR_MAX_BELOW_LOGO_RATIO * vehicle_height:.1f})"
            )

        return reasons

    def build_mirror_entry_from_mask(self, mask, box, logit, caption, vehicle_bbox, logo_bbox):
        if mask is None or not torch.any(mask):
            return None, ["empty SAM mask"]

        mask = mask.bool()
        mask_bbox = self.get_mask_bbox(mask)
        if mask_bbox is None:
            return None, ["bbox maschera assente"]

        mask_pixels = int(torch.count_nonzero(mask).item())
        mask_reasons = self.get_mirror_mask_rejection_reasons(
            mask_bbox,
            mask_pixels,
            vehicle_bbox,
            logo_bbox,
        )
        if mask_reasons:
            return None, mask_reasons

        axis_x, anchor_y = self.get_mirror_anchor(vehicle_bbox, logo_bbox)
        vehicle_width = max(1.0, vehicle_bbox[2] - vehicle_bbox[0] + 1.0)
        vehicle_height = max(1.0, vehicle_bbox[3] - vehicle_bbox[1] + 1.0)
        mask_cx, mask_cy = self.get_box_center(mask_bbox)
        mask_w, mask_h = self.get_box_size(mask_bbox)

        side = "left" if mask_cx < axis_x else "right"
        side_offset_ratio = abs(mask_cx - axis_x) / vehicle_width
        vertical_penalty = abs(mask_cy - anchor_y) / vehicle_height if anchor_y is not None else 0.0
        lateral_reward = min(side_offset_ratio / 0.30, 1.0)
        score = float(logit) + 0.20 * lateral_reward - 0.70 * vertical_penalty

        return {
            "side": side,
            "box": box,
            "mask": mask,
            "mask_bbox": mask_bbox,
            "caption": caption,
            "logit": float(logit),
            "score": score,
            "cx": mask_cx,
            "cy": mask_cy,
            "width": mask_w,
            "height": mask_h,
            "pixel_count": mask_pixels,
        }, None

    def recover_missing_mirror_from_symmetry(
        self,
        image_normal,
        image_size,
        source_entry,
        missing_side,
        vehicle_bbox,
        logo_bbox,
    ):
        if source_entry is None or vehicle_bbox is None or missing_side not in {"left", "right"}:
            return None

        axis_x, anchor_y = self.get_mirror_anchor(vehicle_bbox, logo_bbox)
        if axis_x is None or anchor_y is None:
            return None

        source_dx = source_entry["cx"] - axis_x
        if source_dx == 0:
            return None

        target_cx = 2.0 * axis_x - source_entry["cx"]
        target_cy = source_entry["cy"]
        search_width = max(42.0, source_entry["width"] * 2.4)
        search_height = max(60.0, source_entry["height"] * 2.2)
        synthetic_box = self.make_box_from_center(
            target_cx,
            target_cy,
            search_width,
            search_height,
            image_size,
        )
        if synthetic_box is None:
            return None

        mask = self.predict_mask_from_box(image_normal, synthetic_box)
        entry, reasons = self.build_mirror_entry_from_mask(
            mask,
            synthetic_box,
            source_entry["logit"],
            f"mirrored {source_entry['caption']}",
            vehicle_bbox,
            logo_bbox,
        )
        if entry is None:
            return None

        if entry["side"] != missing_side:
            return None

        vehicle_width = max(1.0, vehicle_bbox[2] - vehicle_bbox[0] + 1.0)
        vehicle_height = max(1.0, vehicle_bbox[3] - vehicle_bbox[1] + 1.0)
        expected_cx = 2.0 * axis_x - source_entry["cx"]
        x_ratio = abs(entry["cx"] - expected_cx) / vehicle_width
        y_ratio = abs(entry["cy"] - source_entry["cy"]) / vehicle_height
        shape_similarity = self.compute_reflected_shape_similarity(
            source_entry["mask"],
            entry["mask"],
        )
        if x_ratio > 0.10 or y_ratio > 0.10 or shape_similarity < 0.12:
            return None

        return entry

    def build_mirror_entries(
        self,
        image_normal,
        transformed_image,
        image_size,
        vehicle_bbox,
        logo_bbox,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    ):
        if vehicle_bbox is None:
            return []

        candidates = self.collect_candidates(
            transformed_image,
            PART_SPECS["mirror"]["captions"],
            image_size,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        if not candidates:
            return []

        axis_x, anchor_y = self.get_mirror_anchor(vehicle_bbox, logo_bbox)
        if axis_x is None:
            return []

        vehicle_width = max(1.0, vehicle_bbox[2] - vehicle_bbox[0] + 1.0)
        vehicle_height = max(1.0, vehicle_bbox[3] - vehicle_bbox[1] + 1.0)
        vehicle_area = vehicle_width * vehicle_height
        max_bottom = float(vehicle_bbox[1]) + MIRROR_MAX_BOTTOM_RATIO * vehicle_height
        min_top = float(vehicle_bbox[1]) + 0.04 * vehicle_height

        raw_entries = []
        for candidate_idx, (box, logit, caption) in enumerate(
            sorted(candidates, key=lambda item: item[1], reverse=True),
            start=1,
        ):
            raw_cx, raw_cy = self.get_box_center(box)
            box_w, box_h = self.get_box_size(box)
            box_area = self.get_box_area(box)
            raw_reasons = []

            if abs(raw_cx - axis_x) < MIRROR_MIN_SIDE_OFFSET_RATIO * vehicle_width:
                raw_reasons.append(
                    f"box too close to vehicle center axis ({abs(raw_cx - axis_x) / vehicle_width:.3f} < {MIRROR_MIN_SIDE_OFFSET_RATIO:.3f})"
                )
            if box_w > MIRROR_MAX_VEHICLE_WIDTH_RATIO * vehicle_width:
                raw_reasons.append(
                    f"box width exceeds threshold ({box_w / vehicle_width:.3f} > {MIRROR_MAX_VEHICLE_WIDTH_RATIO:.3f})"
                )
            if box_h > MIRROR_MAX_VEHICLE_HEIGHT_RATIO * vehicle_height:
                raw_reasons.append(
                    f"box height exceeds threshold ({box_h / vehicle_height:.3f} > {MIRROR_MAX_VEHICLE_HEIGHT_RATIO:.3f})"
                )
            if box_area > MIRROR_MAX_VEHICLE_AREA_RATIO * vehicle_area:
                raw_reasons.append(
                    f"box area exceeds threshold ({box_area / vehicle_area:.3f} > {MIRROR_MAX_VEHICLE_AREA_RATIO:.3f})"
                )
            if raw_cy < min_top or raw_cy > max_bottom:
                raw_reasons.append(
                    f"box vertical center outside valid band (cy={raw_cy:.1f}, band=[{min_top:.1f}, {max_bottom:.1f}])"
                )
            if logo_bbox is not None and raw_cy > anchor_y + MIRROR_MAX_BELOW_LOGO_RATIO * vehicle_height:
                raw_reasons.append(
                    f"box positioned too far below logo anchor (cy={raw_cy:.1f} > {anchor_y + MIRROR_MAX_BELOW_LOGO_RATIO * vehicle_height:.1f})"
                )

            if raw_reasons:
                continue

            mask = self.predict_mask_from_box(image_normal, box)
            entry, mask_reasons = self.build_mirror_entry_from_mask(
                mask,
                box,
                logit,
                caption,
                vehicle_bbox,
                logo_bbox,
            )

            if entry is None:
                mask_bbox = None
                if mask is not None and torch.any(mask):
                    mask_bbox = self.get_mask_bbox(mask.bool())
                continue

            raw_entries.append(entry)

        unique_entries = []
        for entry in sorted(raw_entries, key=lambda item: item["score"], reverse=True):
            overlapping_entry = next(
                (
                    kept
                    for kept in unique_entries
                    if entry["side"] == kept["side"] and
                    self.calculate_iou(entry["mask_bbox"], kept["mask_bbox"]) > 0.30
                ),
                None,
            )
            if overlapping_entry is not None:
                continue
            unique_entries.append(entry)

        limited_entries = []
        side_counts = {"left": 0, "right": 0}
        for entry in unique_entries:
            if side_counts[entry["side"]] >= MIRROR_MAX_CANDIDATES_PER_SIDE:
                continue
            limited_entries.append(entry)
            side_counts[entry["side"]] += 1

        return limited_entries

    def score_mirror_pair(self, left_entry, right_entry, vehicle_bbox, logo_bbox):
        axis_x, anchor_y = self.get_mirror_anchor(vehicle_bbox, logo_bbox)
        if axis_x is None or anchor_y is None:
            return None

        vehicle_width = max(1.0, vehicle_bbox[2] - vehicle_bbox[0] + 1.0)
        vehicle_height = max(1.0, vehicle_bbox[3] - vehicle_bbox[1] + 1.0)
        vehicle_diag = max(1.0, float(np.hypot(vehicle_width, vehicle_height)))

        left_dx = axis_x - left_entry["cx"]
        right_dx = right_entry["cx"] - axis_x
        if left_dx <= 0 or right_dx <= 0:
            return None

        left_dy = left_entry["cy"] - anchor_y
        right_dy = right_entry["cy"] - anchor_y

        mirror_penalty = abs(left_dx - right_dx) / vehicle_width
        vertical_penalty = abs(left_dy - right_dy) / vehicle_height
        radial_penalty = abs(
            np.hypot(left_dx, left_dy) - np.hypot(right_dx, right_dy)
        ) / vehicle_diag
        width_penalty = abs(left_entry["width"] - right_entry["width"]) / max(
            1.0,
            left_entry["width"],
            right_entry["width"],
        )
        height_penalty = abs(left_entry["height"] - right_entry["height"]) / max(
            1.0,
            left_entry["height"],
            right_entry["height"],
        )
        area_penalty = abs(left_entry["pixel_count"] - right_entry["pixel_count"]) / max(
            1.0,
            left_entry["pixel_count"],
            right_entry["pixel_count"],
        )
        shape_similarity = self.compute_mirror_shape_similarity(
            left_entry["mask"],
            right_entry["mask"],
        )

        return (
            left_entry["score"]
            + right_entry["score"]
            + 0.9 * shape_similarity
            - 1.5 * mirror_penalty
            - 1.1 * vertical_penalty
            - 0.9 * radial_penalty
            - 0.7 * (width_penalty + height_penalty)
            - 0.5 * area_penalty
        )

    def find_mirrored_support(self, entry, opposite_entries, vehicle_bbox, logo_bbox):
        if entry is None or not opposite_entries:
            return None

        axis_x, _ = self.get_mirror_anchor(vehicle_bbox, logo_bbox)
        if axis_x is None:
            return None

        vehicle_width = max(1.0, vehicle_bbox[2] - vehicle_bbox[0] + 1.0)
        vehicle_height = max(1.0, vehicle_bbox[3] - vehicle_bbox[1] + 1.0)
        target_cx = 2.0 * axis_x - entry["cx"]
        target_cy = entry["cy"]

        best_candidate = None
        best_support_score = None

        for candidate in opposite_entries:
            x_ratio = abs(candidate["cx"] - target_cx) / vehicle_width
            y_ratio = abs(candidate["cy"] - target_cy) / vehicle_height
            width_ratio = abs(candidate["width"] - entry["width"]) / max(
                1.0,
                candidate["width"],
                entry["width"],
            )
            height_ratio = abs(candidate["height"] - entry["height"]) / max(
                1.0,
                candidate["height"],
                entry["height"],
            )

            if x_ratio > MIRROR_SUPPORT_MAX_X_RATIO:
                continue
            if y_ratio > MIRROR_SUPPORT_MAX_Y_RATIO:
                continue
            if width_ratio > MIRROR_SUPPORT_MAX_SIZE_DELTA_RATIO:
                continue
            if height_ratio > MIRROR_SUPPORT_MAX_SIZE_DELTA_RATIO:
                continue

            support_score = (
                candidate["score"]
                - 1.4 * x_ratio
                - 0.8 * y_ratio
                - 0.6 * (width_ratio + height_ratio)
            )
            if best_support_score is None or support_score > best_support_score:
                best_support_score = support_score
                best_candidate = candidate

        return best_candidate

    def select_mirror_pair(self, entries, vehicle_bbox, logo_bbox):
        left_entries = [entry for entry in entries if entry["side"] == "left"]
        right_entries = [entry for entry in entries if entry["side"] == "right"]

        best_pair = None
        best_pair_score = None

        for left_entry in left_entries:
            for right_entry in right_entries:
                pair_score = self.score_mirror_pair(left_entry, right_entry, vehicle_bbox, logo_bbox)
                if pair_score is None:
                    continue
                if best_pair_score is None or pair_score > best_pair_score:
                    best_pair_score = pair_score
                    best_pair = (left_entry, right_entry)

        if best_pair is not None:
            left_best = max(left_entries, key=lambda item: item["score"], default=None)
            right_best = max(right_entries, key=lambda item: item["score"], default=None)
            strongest_entry = left_best
            if strongest_entry is None or (
                right_best is not None and right_best["score"] > strongest_entry["score"]
            ):
                strongest_entry = right_best

            if strongest_entry is not None:
                opposite_entries = right_entries if strongest_entry["side"] == "left" else left_entries
                mirrored_support = self.find_mirrored_support(
                    strongest_entry,
                    opposite_entries,
                    vehicle_bbox,
                    logo_bbox,
                )
                if mirrored_support is None:
                    if strongest_entry["side"] == "left":
                        return strongest_entry, None
                    return None, strongest_entry

            return best_pair

        left_best = max(left_entries, key=lambda item: item["score"], default=None)
        right_best = max(right_entries, key=lambda item: item["score"], default=None)

        if left_best is not None and right_best is not None:
            if left_best["score"] >= right_best["score"]:
                if self.find_mirrored_support(left_best, right_entries, vehicle_bbox, logo_bbox) is None:
                    return left_best, None
            else:
                if self.find_mirrored_support(right_best, left_entries, vehicle_bbox, logo_bbox) is None:
                    return None, right_best

        return left_best, right_best

    def build_mirror_masks(
        self,
        image_normal,
        transformed_image,
        image_size,
        vehicle_bbox,
        logo_bbox,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    ):
        entries = self.build_mirror_entries(
            image_normal,
            transformed_image,
            image_size,
            vehicle_bbox,
            logo_bbox,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        if not entries:
            return {}

        left_entry, right_entry = self.select_mirror_pair(
            entries,
            vehicle_bbox,
            logo_bbox,
        )

        if left_entry is not None and right_entry is None:
            right_entry = self.recover_missing_mirror_from_symmetry(
                image_normal,
                image_size,
                left_entry,
                "right",
                vehicle_bbox,
                logo_bbox,
            )
        elif right_entry is not None and left_entry is None:
            left_entry = self.recover_missing_mirror_from_symmetry(
                image_normal,
                image_size,
                right_entry,
                "left",
                vehicle_bbox,
                logo_bbox,
            )

        mirror_masks = {}

        if left_entry is not None:
            mirror_masks["left mirror"] = left_entry["mask"]
        if right_entry is not None:
            mirror_masks["right mirror"] = right_entry["mask"]

        return mirror_masks

    def get_wheel_anchor(self, vehicle_bbox, logo_bbox):
        if vehicle_bbox is None:
            return None, None

        if logo_bbox is not None:
            axis_x, _ = self.get_box_center(logo_bbox)
        else:
            axis_x, _ = self.get_box_center(vehicle_bbox)

        vehicle_height = max(1.0, vehicle_bbox[3] - vehicle_bbox[1] + 1.0)
        target_y = float(vehicle_bbox[1]) + WHEEL_FALLBACK_CENTER_Y_RATIO * vehicle_height
        if logo_bbox is not None:
            target_y = max(target_y, float(logo_bbox[3]) + 0.22 * vehicle_height)

        return axis_x, target_y

    def extract_best_wheel_component_mask(self, mask, vehicle_bbox, logo_bbox):
        if mask is None or not torch.any(mask):
            return None

        if vehicle_bbox is None:
            return mask.bool()

        mask_np = mask.cpu().numpy().astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
        if num_labels <= 1:
            return mask.bool()

        axis_x, target_y = self.get_wheel_anchor(vehicle_bbox, logo_bbox)
        if axis_x is None or target_y is None:
            return mask.bool()

        vehicle_width = max(1.0, vehicle_bbox[2] - vehicle_bbox[0] + 1.0)
        vehicle_height = max(1.0, vehicle_bbox[3] - vehicle_bbox[1] + 1.0)
        vehicle_area = vehicle_width * vehicle_height

        best_label = None
        best_score = None

        for label_idx in range(1, num_labels):
            x = stats[label_idx, cv2.CC_STAT_LEFT]
            y = stats[label_idx, cv2.CC_STAT_TOP]
            width = stats[label_idx, cv2.CC_STAT_WIDTH]
            height = stats[label_idx, cv2.CC_STAT_HEIGHT]
            area = stats[label_idx, cv2.CC_STAT_AREA]
            if area <= 0:
                continue

            component_bbox = np.array(
                [float(x), float(y), float(x + width - 1), float(y + height - 1)],
                dtype=np.float32,
            )
            component_cx, component_cy = self.get_box_center(component_bbox)
            side_offset_ratio = abs(component_cx - axis_x) / vehicle_width
            outer_gap_ratio = min(
                abs(float(component_bbox[0]) - float(vehicle_bbox[0])),
                abs(float(vehicle_bbox[2]) - float(component_bbox[2])),
            ) / vehicle_width
            bottom_ratio = (float(component_bbox[3]) - float(vehicle_bbox[1])) / vehicle_height
            vertical_gap_ratio = abs(component_cy - target_y) / vehicle_height
            area_ratio = float(area) / max(1.0, vehicle_area)

            score = (
                1.60 * side_offset_ratio
                + 1.80 * bottom_ratio
                + 0.80 * area_ratio
                - 1.40 * outer_gap_ratio
                - 1.00 * vertical_gap_ratio
            )
            if best_score is None or score > best_score:
                best_score = score
                best_label = label_idx

        if best_label is None:
            return mask.bool()

        return torch.from_numpy(labels == best_label)

    def get_wheel_mask_rejection_reasons(self, mask_bbox, mask_pixels, vehicle_bbox, logo_bbox):
        if mask_bbox is None or vehicle_bbox is None:
            return ["missing mask bounding box"]

        axis_x, target_y = self.get_wheel_anchor(vehicle_bbox, logo_bbox)
        if axis_x is None or target_y is None:
            return ["missing wheel reference axis"]

        vehicle_width = max(1.0, vehicle_bbox[2] - vehicle_bbox[0] + 1.0)
        vehicle_height = max(1.0, vehicle_bbox[3] - vehicle_bbox[1] + 1.0)
        vehicle_area = vehicle_width * vehicle_height

        mask_cx, mask_cy = self.get_box_center(mask_bbox)
        mask_w, mask_h = self.get_box_size(mask_bbox)
        side = "left" if mask_cx < axis_x else "right"
        side_offset_ratio = abs(mask_cx - axis_x) / vehicle_width
        outer_gap_ratio = (
            max(0.0, float(mask_bbox[0]) - float(vehicle_bbox[0])) / vehicle_width
            if side == "left"
            else max(0.0, float(vehicle_bbox[2]) - float(mask_bbox[2])) / vehicle_width
        )
        top_ratio = (float(mask_bbox[1]) - float(vehicle_bbox[1])) / vehicle_height
        bottom_ratio = (float(mask_bbox[3]) - float(vehicle_bbox[1])) / vehicle_height
        vertical_gap_ratio = abs(mask_cy - target_y) / vehicle_height
        aspect_ratio = mask_w / max(1.0, mask_h)
        reasons = []

        if side_offset_ratio < WHEEL_MIN_SIDE_OFFSET_RATIO:
            reasons.append(
                f"mask too close to vehicle center ({side_offset_ratio:.3f} < {WHEEL_MIN_SIDE_OFFSET_RATIO:.3f})"
            )
        if side_offset_ratio > WHEEL_MAX_SIDE_OFFSET_RATIO:
            reasons.append(
                f"mask too far lateral ({side_offset_ratio:.3f} > {WHEEL_MAX_SIDE_OFFSET_RATIO:.3f})"
            )
        if outer_gap_ratio > WHEEL_MAX_OUTER_EDGE_GAP_RATIO:
            reasons.append(
                f"mask gap to outer vehicle edge exceeds threshold ({outer_gap_ratio:.3f} > {WHEEL_MAX_OUTER_EDGE_GAP_RATIO:.3f})"
            )
        if mask_w < WHEEL_MIN_VEHICLE_WIDTH_RATIO * vehicle_width:
            reasons.append(
                f"mask width below minimum threshold ({mask_w / vehicle_width:.3f} < {WHEEL_MIN_VEHICLE_WIDTH_RATIO:.3f})"
            )
        if mask_h < WHEEL_MIN_VEHICLE_HEIGHT_RATIO * vehicle_height:
            reasons.append(
                f"mask height below minimum threshold ({mask_h / vehicle_height:.3f} < {WHEEL_MIN_VEHICLE_HEIGHT_RATIO:.3f})"
            )
        if mask_w > WHEEL_MAX_VEHICLE_WIDTH_RATIO * vehicle_width:
            reasons.append(
                f"maschera troppo larga ({mask_w / vehicle_width:.3f} > {WHEEL_MAX_VEHICLE_WIDTH_RATIO:.3f})"
            )
        if mask_h > WHEEL_MAX_VEHICLE_HEIGHT_RATIO * vehicle_height:
            reasons.append(
                f"maschera troppo alta ({mask_h / vehicle_height:.3f} > {WHEEL_MAX_VEHICLE_HEIGHT_RATIO:.3f})"
            )
        if mask_pixels > WHEEL_MAX_VEHICLE_AREA_RATIO * vehicle_area:
            reasons.append(
                f"maschera troppo estesa ({mask_pixels / vehicle_area:.3f} > {WHEEL_MAX_VEHICLE_AREA_RATIO:.3f})"
            )
        if top_ratio < WHEEL_MIN_TOP_RATIO:
            reasons.append(
                f"mask positioned too high in vehicle ({top_ratio:.3f} < {WHEEL_MIN_TOP_RATIO:.3f})"
            )
        if bottom_ratio < WHEEL_MIN_BOTTOM_RATIO:
            reasons.append(
                f"mask bottom does not extend low enough ({bottom_ratio:.3f} < {WHEEL_MIN_BOTTOM_RATIO:.3f})"
            )
        if vertical_gap_ratio > WHEEL_MAX_VERTICAL_FROM_TARGET_RATIO:
            reasons.append(
                f"mask vertical distance from target anchor exceeds threshold ({vertical_gap_ratio:.3f} > {WHEEL_MAX_VERTICAL_FROM_TARGET_RATIO:.3f})"
            )
        if aspect_ratio > WHEEL_MAX_ASPECT_RATIO:
            reasons.append(
                f"mask aspect ratio (w/h) exceeds maximum ({aspect_ratio:.3f} > {WHEEL_MAX_ASPECT_RATIO:.3f})"
            )
        if logo_bbox is not None and float(mask_bbox[1]) <= float(logo_bbox[3]) - 0.02 * vehicle_height:
            reasons.append("mask insufficiently below logo anchor")

        return reasons

    def build_wheel_entry_from_mask(
        self,
        image_for_mask,
        mask,
        box,
        logit,
        caption,
        vehicle_bbox,
        logo_bbox,
    ):
        if mask is None or not torch.any(mask):
            return None, ["empty SAM mask"]

        mask = self.extract_best_wheel_component_mask(mask.bool(), vehicle_bbox, logo_bbox)
        if mask is None or not torch.any(mask):
            return None, ["componente ruota assente"]

        mask = mask.bool()
        mask_bbox = self.get_mask_bbox(mask)
        if mask_bbox is None:
            return None, ["bbox maschera assente"]

        mask_pixels = int(torch.count_nonzero(mask).item())
        mask_reasons = self.get_wheel_mask_rejection_reasons(
            mask_bbox,
            mask_pixels,
            vehicle_bbox,
            logo_bbox,
        )
        if mask_reasons:
            return None, mask_reasons

        darkness = self.compute_mask_darkness(image_for_mask, mask)
        if darkness < WHEEL_MIN_DARKNESS:
            return None, [f"darkness insufficiente ({darkness:.3f} < {WHEEL_MIN_DARKNESS:.3f})"]

        axis_x, target_y = self.get_wheel_anchor(vehicle_bbox, logo_bbox)
        vehicle_width = max(1.0, vehicle_bbox[2] - vehicle_bbox[0] + 1.0)
        vehicle_height = max(1.0, vehicle_bbox[3] - vehicle_bbox[1] + 1.0)
        mask_cx, mask_cy = self.get_box_center(mask_bbox)
        mask_w, mask_h = self.get_box_size(mask_bbox)
        side = "left" if mask_cx < axis_x else "right"
        side_offset_ratio = abs(mask_cx - axis_x) / vehicle_width
        outer_gap_ratio = (
            max(0.0, float(mask_bbox[0]) - float(vehicle_bbox[0])) / vehicle_width
            if side == "left"
            else max(0.0, float(vehicle_bbox[2]) - float(mask_bbox[2])) / vehicle_width
        )
        bottom_ratio = (float(mask_bbox[3]) - float(vehicle_bbox[1])) / vehicle_height
        vertical_penalty = abs(mask_cy - target_y) / vehicle_height if target_y is not None else 0.0
        score = (
            float(logit)
            + 0.80 * darkness
            + 0.45 * bottom_ratio
            + 0.30 * side_offset_ratio
            - 0.75 * vertical_penalty
            - 0.55 * outer_gap_ratio
        )

        return {
            "side": side,
            "box": box,
            "mask": mask,
            "mask_bbox": mask_bbox,
            "caption": caption,
            "logit": float(logit),
            "score": score,
            "cx": mask_cx,
            "cy": mask_cy,
            "width": mask_w,
            "height": mask_h,
            "pixel_count": mask_pixels,
            "darkness": darkness,
        }, None

    def score_wheel_pair(self, left_entry, right_entry, vehicle_bbox, logo_bbox):
        axis_x, target_y = self.get_wheel_anchor(vehicle_bbox, logo_bbox)
        if axis_x is None or target_y is None:
            return None

        vehicle_width = max(1.0, vehicle_bbox[2] - vehicle_bbox[0] + 1.0)
        vehicle_height = max(1.0, vehicle_bbox[3] - vehicle_bbox[1] + 1.0)
        vehicle_diag = max(1.0, float(np.hypot(vehicle_width, vehicle_height)))

        left_dx = axis_x - left_entry["cx"]
        right_dx = right_entry["cx"] - axis_x
        if left_dx <= 0 or right_dx <= 0:
            return None

        left_dy = left_entry["cy"] - target_y
        right_dy = right_entry["cy"] - target_y

        mirror_penalty = abs(left_dx - right_dx) / vehicle_width
        vertical_penalty = abs(left_entry["cy"] - right_entry["cy"]) / vehicle_height
        radial_penalty = abs(
            np.hypot(left_dx, left_dy) - np.hypot(right_dx, right_dy)
        ) / vehicle_diag
        width_penalty = abs(left_entry["width"] - right_entry["width"]) / max(
            1.0,
            left_entry["width"],
            right_entry["width"],
        )
        height_penalty = abs(left_entry["height"] - right_entry["height"]) / max(
            1.0,
            left_entry["height"],
            right_entry["height"],
        )
        area_penalty = abs(left_entry["pixel_count"] - right_entry["pixel_count"]) / max(
            1.0,
            left_entry["pixel_count"],
            right_entry["pixel_count"],
        )
        darkness_gap = abs(left_entry["darkness"] - right_entry["darkness"])
        shape_similarity = self.compute_reflected_shape_similarity(
            left_entry["mask"],
            right_entry["mask"],
        )

        return (
            left_entry["score"]
            + right_entry["score"]
            + 0.80 * shape_similarity
            + 0.40 * (left_entry["darkness"] + right_entry["darkness"])
            - 1.80 * mirror_penalty
            - 1.15 * vertical_penalty
            - 0.65 * radial_penalty
            - 0.60 * (width_penalty + height_penalty)
            - 0.45 * area_penalty
            - 0.30 * darkness_gap
        )

    def recover_missing_wheel_from_symmetry(
        self,
        image_for_mask,
        image_size,
        source_entry,
        missing_side,
        vehicle_bbox,
        logo_bbox,
    ):
        if source_entry is None or vehicle_bbox is None or missing_side not in {"left", "right"}:
            return None

        axis_x, target_y = self.get_wheel_anchor(vehicle_bbox, logo_bbox)
        if axis_x is None or target_y is None:
            return None

        vehicle_width = max(1.0, vehicle_bbox[2] - vehicle_bbox[0] + 1.0)
        vehicle_height = max(1.0, vehicle_bbox[3] - vehicle_bbox[1] + 1.0)
        target_cx = 2.0 * axis_x - source_entry["cx"]
        target_cy = max(source_entry["cy"], target_y - 0.02 * vehicle_height)
        search_width = max(56.0, source_entry["width"] * 1.55)
        search_height = max(100.0, source_entry["height"] * 1.35)
        synthetic_box = self.make_box_from_center(
            target_cx,
            target_cy,
            search_width,
            search_height,
            image_size,
        )
        if synthetic_box is None:
            return None

        mask = self.predict_mask_from_box(image_for_mask, synthetic_box)
        entry, reasons = self.build_wheel_entry_from_mask(
            image_for_mask,
            mask,
            synthetic_box,
            source_entry["logit"],
            f"mirrored {source_entry['caption']}",
            vehicle_bbox,
            logo_bbox,
        )
        if entry is None:
            return None

        if entry["side"] != missing_side:
            return None

        x_ratio = abs(entry["cx"] - target_cx) / vehicle_width
        y_ratio = abs(entry["cy"] - source_entry["cy"]) / vehicle_height
        shape_similarity = self.compute_reflected_shape_similarity(
            source_entry["mask"],
            entry["mask"],
        )
        if x_ratio > 0.14 or y_ratio > 0.10 or shape_similarity < 0.02:
            return None

        return entry

    def recover_wheel_from_anchor_box(
        self,
        image_for_mask,
        image_size,
        side,
        vehicle_bbox,
        logo_bbox,
    ):
        if vehicle_bbox is None or side not in {"left", "right"}:
            return None

        axis_x, target_y = self.get_wheel_anchor(vehicle_bbox, logo_bbox)
        if axis_x is None or target_y is None:
            return None

        vehicle_width = max(1.0, vehicle_bbox[2] - vehicle_bbox[0] + 1.0)
        vehicle_height = max(1.0, vehicle_bbox[3] - vehicle_bbox[1] + 1.0)
        center_x = (
            axis_x - WHEEL_FALLBACK_CENTER_OFFSET_RATIO * vehicle_width
            if side == "left"
            else axis_x + WHEEL_FALLBACK_CENTER_OFFSET_RATIO * vehicle_width
        )
        center_y = target_y
        search_width = max(60.0, WHEEL_FALLBACK_BOX_WIDTH_RATIO * vehicle_width)
        search_height = max(96.0, WHEEL_FALLBACK_BOX_HEIGHT_RATIO * vehicle_height)
        synthetic_box = self.make_box_from_center(
            center_x,
            center_y,
            search_width,
            search_height,
            image_size,
        )
        if synthetic_box is None:
            return None

        mask = self.predict_mask_from_box(image_for_mask, synthetic_box)
        entry, reasons = self.build_wheel_entry_from_mask(
            image_for_mask,
            mask,
            synthetic_box,
            0.0,
            f"anchor fallback {side} wheel",
            vehicle_bbox,
            logo_bbox,
        )
        if entry is None:
            return None

        if entry["side"] != side:
            return None

        return entry

    def build_dark_wheel_fallback_mask(
        self,
        image_normal,
        vehicle_mask,
        vehicle_bbox,
        logo_bbox,
        side,
    ):
        if (
            image_normal is None or
            vehicle_mask is None or
            vehicle_bbox is None or
            side not in {"left", "right"}
        ):
            return None

        axis_x, target_y = self.get_wheel_anchor(vehicle_bbox, logo_bbox)
        if axis_x is None or target_y is None:
            return None

        gray = self.get_image_gray(image_normal)
        vehicle_mask_np = vehicle_mask.cpu().numpy().astype(bool)

        vehicle_left = float(vehicle_bbox[0])
        vehicle_top = float(vehicle_bbox[1])
        vehicle_right = float(vehicle_bbox[2])
        vehicle_bottom = float(vehicle_bbox[3])
        vehicle_width = max(1.0, vehicle_right - vehicle_left + 1.0)
        vehicle_height = max(1.0, vehicle_bottom - vehicle_top + 1.0)

        col_indices = np.arange(gray.shape[1])[None, :]
        row_indices = np.arange(gray.shape[0])[:, None]
        min_y = max(
            vehicle_top + WHEEL_MIN_TOP_RATIO * vehicle_height,
            target_y - 0.10 * vehicle_height,
        )
        outer_band_width = 0.24 * vehicle_width

        if side == "left":
            side_mask = (
                (col_indices >= vehicle_left) &
                (col_indices <= vehicle_left + outer_band_width) &
                (col_indices <= axis_x - 0.10 * vehicle_width)
            )
        else:
            side_mask = (
                (col_indices <= vehicle_right) &
                (col_indices >= vehicle_right - outer_band_width) &
                (col_indices >= axis_x + 0.10 * vehicle_width)
            )

        search_mask = (
            vehicle_mask_np &
            side_mask &
            (row_indices >= min_y) &
            (row_indices <= vehicle_bottom)
        )
        dark_mask = (gray <= WHEEL_DARK_FALLBACK_THRESHOLD) & search_mask
        if not np.any(dark_mask):
            return None

        dark_mask_u8 = (dark_mask.astype(np.uint8) * 255)
        kernel = np.ones((5, 5), dtype=np.uint8)
        dark_mask_u8 = cv2.morphologyEx(dark_mask_u8, cv2.MORPH_OPEN, kernel)
        dark_mask_u8 = cv2.morphologyEx(dark_mask_u8, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask_u8, connectivity=8)
        best_mask = None
        best_score = None

        for label_idx in range(1, num_labels):
            component_mask = labels == label_idx
            if not np.any(component_mask):
                continue

            component_mask_t = torch.from_numpy(component_mask)
            component_bbox = self.get_mask_bbox(component_mask_t)
            if component_bbox is None:
                continue

            component_pixels = int(component_mask.sum())
            rejection_reasons = self.get_wheel_mask_rejection_reasons(
                component_bbox,
                component_pixels,
                vehicle_bbox,
                logo_bbox,
            )
            if rejection_reasons:
                continue

            darkness = self.compute_mask_darkness(image_normal, component_mask_t)
            if darkness < WHEEL_MIN_DARKNESS:
                continue

            component_cx, component_cy = self.get_box_center(component_bbox)
            side_offset_ratio = abs(component_cx - axis_x) / vehicle_width
            bottom_ratio = (float(component_bbox[3]) - vehicle_top) / vehicle_height
            vertical_gap_ratio = abs(component_cy - target_y) / vehicle_height
            score = (
                1.20 * darkness
                + 0.80 * side_offset_ratio
                + 0.60 * bottom_ratio
                - 0.80 * vertical_gap_ratio
            )
            if best_score is None or score > best_score:
                best_score = score
                best_mask = component_mask_t

        if best_mask is None:
            return None

        return best_mask.bool()

    def build_wheel_entries(
        self,
        image_for_mask,
        transformed_image,
        image_size,
        vehicle_bbox,
        logo_bbox,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    ):
        if vehicle_bbox is None:
            return []

        candidates = self.collect_candidates(
            transformed_image,
            PART_SPECS["wheel"]["captions"],
            image_size,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        if not candidates:
            return []

        axis_x, target_y = self.get_wheel_anchor(vehicle_bbox, logo_bbox)
        if axis_x is None or target_y is None:
            return []

        vehicle_width = max(1.0, vehicle_bbox[2] - vehicle_bbox[0] + 1.0)
        vehicle_height = max(1.0, vehicle_bbox[3] - vehicle_bbox[1] + 1.0)

        raw_entries = []
        for candidate_idx, (box, logit, caption) in enumerate(
            sorted(candidates, key=lambda item: item[1], reverse=True),
            start=1,
        ):
            raw_cx, raw_cy = self.get_box_center(box)
            box_w, box_h = self.get_box_size(box)
            side = "left" if raw_cx < axis_x else "right"
            side_offset_ratio = abs(raw_cx - axis_x) / vehicle_width
            outer_gap_ratio = (
                max(0.0, float(box[0]) - float(vehicle_bbox[0])) / vehicle_width
                if side == "left"
                else max(0.0, float(vehicle_bbox[2]) - float(box[2])) / vehicle_width
            )
            top_ratio = (float(box[1]) - float(vehicle_bbox[1])) / vehicle_height
            bottom_ratio = (float(box[3]) - float(vehicle_bbox[1])) / vehicle_height
            vertical_gap_ratio = abs(raw_cy - target_y) / vehicle_height

            raw_reasons = []
            if side_offset_ratio < 0.22 or side_offset_ratio > 0.58:
                raw_reasons.append("box outside lateral wheel band")
            if outer_gap_ratio > 0.22:
                raw_reasons.append("box too distant from outer vehicle edge")
            if box_w > 0.28 * vehicle_width:
                raw_reasons.append("box width exceeds threshold")
            if box_h < 0.06 * vehicle_height or box_h > 0.46 * vehicle_height:
                raw_reasons.append("box height outside plausible bounds")
            if top_ratio < 0.46:
                raw_reasons.append("box height exceeds threshold")
            if bottom_ratio < 0.72:
                raw_reasons.append("box does not reach low enough")
            if vertical_gap_ratio > 0.28:
                raw_reasons.append("box too far from vertical target anchor")

            if raw_reasons:
                continue

            mask = self.predict_mask_from_box(image_for_mask, box)
            entry, mask_reasons = self.build_wheel_entry_from_mask(
                image_for_mask,
                mask,
                box,
                logit,
                caption,
                vehicle_bbox,
                logo_bbox,
            )
            if entry is None:
                mask_bbox = None
                if mask is not None and torch.any(mask):
                    mask_bbox = self.get_mask_bbox(mask.bool())
                continue

            raw_entries.append(entry)
        unique_entries = []
        for entry in sorted(raw_entries, key=lambda item: item["score"], reverse=True):
            overlapping_entry = next(
                (
                    kept
                    for kept in unique_entries
                    if entry["side"] == kept["side"] and
                    self.calculate_iou(entry["mask_bbox"], kept["mask_bbox"]) > 0.30
                ),
                None,
            )
            if overlapping_entry is not None:
                continue
            unique_entries.append(entry)

        limited_entries = []
        side_counts = {"left": 0, "right": 0}
        for entry in unique_entries:
            if side_counts[entry["side"]] >= WHEEL_MAX_CANDIDATES_PER_SIDE:
                continue
            limited_entries.append(entry)
            side_counts[entry["side"]] += 1

        return limited_entries

    def select_wheel_pair(self, entries, vehicle_bbox, logo_bbox):
        left_entries = [entry for entry in entries if entry["side"] == "left"]
        right_entries = [entry for entry in entries if entry["side"] == "right"]

        best_pair = None
        best_pair_score = None

        for left_entry in left_entries:
            for right_entry in right_entries:
                pair_score = self.score_wheel_pair(left_entry, right_entry, vehicle_bbox, logo_bbox)
                if pair_score is None:
                    continue
                if best_pair_score is None or pair_score > best_pair_score:
                    best_pair_score = pair_score
                    best_pair = (left_entry, right_entry)

        if best_pair is not None:
            return best_pair

        left_best = max(left_entries, key=lambda item: item["score"], default=None)
        right_best = max(right_entries, key=lambda item: item["score"], default=None)

        return left_best, right_best

    def build_wheel_masks(
        self,
        image_normal,
        image_for_mask,
        transformed_image,
        image_size,
        vehicle_mask,
        vehicle_bbox,
        logo_bbox,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    ):
        entries = self.build_wheel_entries(
            image_for_mask,
            transformed_image,
            image_size,
            vehicle_bbox,
            logo_bbox,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        left_entry, right_entry = self.select_wheel_pair(
            entries,
            vehicle_bbox,
            logo_bbox,
        )

        if left_entry is not None and right_entry is None:
            right_entry = self.recover_missing_wheel_from_symmetry(
                image_for_mask,
                image_size,
                left_entry,
                "right",
                vehicle_bbox,
                logo_bbox,
            )
        elif right_entry is not None and left_entry is None:
            left_entry = self.recover_missing_wheel_from_symmetry(
                image_for_mask,
                image_size,
                right_entry,
                "left",
                vehicle_bbox,
                logo_bbox,
            )

        if left_entry is None:
            left_entry = self.recover_wheel_from_anchor_box(
                image_for_mask,
                image_size,
                "left",
                vehicle_bbox,
                logo_bbox,
            )
        if right_entry is None:
            right_entry = self.recover_wheel_from_anchor_box(
                image_for_mask,
                image_size,
                "right",
                vehicle_bbox,
                logo_bbox,
            )

        if left_entry is None:
            left_dark_mask = self.build_dark_wheel_fallback_mask(
                image_normal,
                vehicle_mask,
                vehicle_bbox,
                logo_bbox,
                "left",
            )
            if left_dark_mask is not None:
                left_entry, _ = self.build_wheel_entry_from_mask(
                    image_normal,
                    left_dark_mask,
                    self.get_mask_bbox(left_dark_mask),
                    0.0,
                    "dark fallback left wheel",
                    vehicle_bbox,
                    logo_bbox,
                )
        if right_entry is None:
            right_dark_mask = self.build_dark_wheel_fallback_mask(
                image_normal,
                vehicle_mask,
                vehicle_bbox,
                logo_bbox,
                "right",
            )
            if right_dark_mask is not None:
                right_entry, _ = self.build_wheel_entry_from_mask(
                    image_normal,
                    right_dark_mask,
                    self.get_mask_bbox(right_dark_mask),
                    0.0,
                    "dark fallback right wheel",
                    vehicle_bbox,
                    logo_bbox,
                )

        wheel_masks = {}
        if left_entry is not None:
            wheel_masks["left wheel"] = left_entry["mask"]
        if right_entry is not None:
            wheel_masks["right wheel"] = right_entry["mask"]

        return wheel_masks

    def get_exhaust_anchor(self, vehicle_bbox, logo_bbox):
        if vehicle_bbox is None:
            return None, None

        if logo_bbox is not None:
            anchor_x, anchor_y = self.get_box_center(logo_bbox)
            return anchor_x, anchor_y

        vehicle_cx, vehicle_cy = self.get_box_center(vehicle_bbox)
        return vehicle_cx, vehicle_cy

    def get_exhaust_side(self, mask_bbox, vehicle_bbox, logo_bbox):
        if mask_bbox is None or vehicle_bbox is None:
            return "unknown"

        axis_x, _ = self.get_exhaust_anchor(vehicle_bbox, logo_bbox)
        if axis_x is None:
            return "unknown"

        vehicle_width = max(1.0, vehicle_bbox[2] - vehicle_bbox[0] + 1.0)
        mask_cx, _ = self.get_box_center(mask_bbox)
        delta_ratio = (mask_cx - axis_x) / vehicle_width
        if delta_ratio < -0.04:
            return "left"
        if delta_ratio > 0.04:
            return "right"
        return "center"

    def classify_exhaust_position(self, mask_bbox, vehicle_bbox, logo_bbox):
        if mask_bbox is None or vehicle_bbox is None:
            return None, 0.0, ["geometria assente"]

        anchor_x, anchor_y = self.get_exhaust_anchor(vehicle_bbox, logo_bbox)
        if anchor_x is None or anchor_y is None:
            return None, 0.0, ["ancora scarico assente"]

        vehicle_width = max(1.0, vehicle_bbox[2] - vehicle_bbox[0] + 1.0)
        vehicle_height = max(1.0, vehicle_bbox[3] - vehicle_bbox[1] + 1.0)
        mask_cx, mask_cy = self.get_box_center(mask_bbox)
        side_offset_ratio = abs(mask_cx - anchor_x) / vehicle_width
        bottom_ratio = (float(mask_bbox[3]) - float(vehicle_bbox[1])) / vehicle_height
        reasons = []

        if (
            side_offset_ratio >= EXHAUST_LATERAL_MIN_SIDE_OFFSET_RATIO and
            bottom_ratio >= EXHAUST_LATERAL_MIN_BOTTOM_RATIO
        ):
            region_score = (
                min(side_offset_ratio / 0.35, 1.0)
                + min(bottom_ratio / 0.95, 1.0)
            ) / 2.0
            return "lateral", region_score, []

        if side_offset_ratio <= EXHAUST_CENTRAL_MAX_CENTER_OFFSET_RATIO:
            if logo_bbox is not None:
                logo_top = float(logo_bbox[1])
                logo_bottom = float(logo_bbox[3])
                min_y = logo_top - EXHAUST_CENTRAL_MAX_ABOVE_LOGO_RATIO * vehicle_height
                max_y = logo_bottom + EXHAUST_CENTRAL_MAX_BELOW_LOGO_RATIO * vehicle_height
                if min_y <= mask_cy <= max_y:
                    center_score = 1.0 - min(
                        side_offset_ratio / max(EXHAUST_CENTRAL_MAX_CENTER_OFFSET_RATIO, 1e-6),
                        1.0,
                    )
                    logo_band_score = 1.0 - min(
                        abs(mask_cy - anchor_y) / max(0.18 * vehicle_height, 1.0),
                        1.0,
                    )
                    return "central", 0.55 * center_score + 0.45 * logo_band_score, []
                reasons.append(
                    f"central candidate outside logo band (cy={mask_cy:.1f}, band=[{min_y:.1f}, {max_y:.1f}])"
                )
            else:
                top_ratio = (float(mask_bbox[1]) - float(vehicle_bbox[1])) / vehicle_height
                if 0.20 <= top_ratio <= 0.58:
                    region_score = 1.0 - min(side_offset_ratio / max(EXHAUST_CENTRAL_MAX_CENTER_OFFSET_RATIO, 1e-6), 1.0)
                    return "central", region_score, []
                reasons.append("central candidate without logo outside plausible central band")

        if side_offset_ratio < EXHAUST_LATERAL_MIN_SIDE_OFFSET_RATIO:
            reasons.append(
                f"too central for a lateral exhaust ({side_offset_ratio:.3f} < {EXHAUST_LATERAL_MIN_SIDE_OFFSET_RATIO:.3f})"
            )
        if side_offset_ratio > EXHAUST_CENTRAL_MAX_CENTER_OFFSET_RATIO:
            reasons.append(
                f"too lateral for a central exhaust ({side_offset_ratio:.3f} > {EXHAUST_CENTRAL_MAX_CENTER_OFFSET_RATIO:.3f})"
            )
        if bottom_ratio < EXHAUST_LATERAL_MIN_BOTTOM_RATIO:
            reasons.append(
                f"too high for a lateral exhaust ({bottom_ratio:.3f} < {EXHAUST_LATERAL_MIN_BOTTOM_RATIO:.3f})"
            )

        return None, 0.0, reasons

    def build_exhaust_fallback_boxes(self, image_size, vehicle_bbox, logo_bbox):
        if image_size is None or vehicle_bbox is None:
            return []

        vehicle_left = float(vehicle_bbox[0])
        vehicle_top = float(vehicle_bbox[1])
        vehicle_right = float(vehicle_bbox[2])
        vehicle_bottom = float(vehicle_bbox[3])
        vehicle_width = max(1.0, vehicle_right - vehicle_left + 1.0)
        vehicle_height = max(1.0, vehicle_bottom - vehicle_top + 1.0)
        anchor_x, anchor_y = self.get_exhaust_anchor(vehicle_bbox, logo_bbox)

        boxes = []
        lower_top = vehicle_top + EXHAUST_FALLBACK_LATERAL_TOP_RATIO * vehicle_height
        lower_width = EXHAUST_FALLBACK_LATERAL_SEARCH_WIDTH_RATIO * vehicle_width
        lower_height = min(vehicle_bottom - lower_top, 0.26 * vehicle_height)

        left_box = self.clip_box_to_image(
            np.array(
                [
                    vehicle_left,
                    lower_top,
                    vehicle_left + lower_width,
                    lower_top + lower_height,
                ],
                dtype=np.float32,
            ),
            image_size,
        )
        right_box = self.clip_box_to_image(
            np.array(
                [
                    vehicle_right - lower_width,
                    lower_top,
                    vehicle_right,
                    lower_top + lower_height,
                ],
                dtype=np.float32,
            ),
            image_size,
        )
        if left_box is not None:
            boxes.append((left_box, 0.0, "fallback lower left exhaust zone"))
        if right_box is not None:
            boxes.append((right_box, 0.0, "fallback lower right exhaust zone"))

        if anchor_x is not None:
            cluster_width = EXHAUST_FALLBACK_SIDE_CLUSTER_WIDTH_RATIO * vehicle_width
            cluster_height = EXHAUST_FALLBACK_SIDE_CLUSTER_HEIGHT_RATIO * vehicle_height
            cluster_center_y = vehicle_top + EXHAUST_FALLBACK_SIDE_CLUSTER_CENTER_Y_RATIO * vehicle_height
            left_cluster_box = self.make_box_from_center(
                anchor_x - EXHAUST_FALLBACK_SIDE_CLUSTER_CENTER_OFFSET_RATIO * vehicle_width,
                cluster_center_y,
                cluster_width,
                cluster_height,
                image_size,
            )
            right_cluster_box = self.make_box_from_center(
                anchor_x + EXHAUST_FALLBACK_SIDE_CLUSTER_CENTER_OFFSET_RATIO * vehicle_width,
                cluster_center_y,
                cluster_width,
                cluster_height,
                image_size,
            )
            if left_cluster_box is not None:
                boxes.append((left_cluster_box, 0.0, "fallback side-cluster left exhaust zone"))
            if right_cluster_box is not None:
                boxes.append((right_cluster_box, 0.0, "fallback side-cluster right exhaust zone"))

        if anchor_x is not None:
            central_width = EXHAUST_FALLBACK_CENTRAL_WIDTH_RATIO * vehicle_width
            if logo_bbox is not None:
                logo_top = float(logo_bbox[1])
                logo_bottom = float(logo_bbox[3])
                central_ranges = [
                    (
                        logo_top - EXHAUST_CENTRAL_MAX_ABOVE_LOGO_RATIO * vehicle_height,
                        logo_top + 0.02 * vehicle_height,
                        "fallback upper central exhaust zone",
                    ),
                    (
                        logo_top - 0.06 * vehicle_height,
                        logo_bottom + 0.10 * vehicle_height,
                        "fallback middle central exhaust zone",
                    ),
                    (
                        logo_bottom - 0.02 * vehicle_height,
                        logo_bottom + EXHAUST_CENTRAL_MAX_BELOW_LOGO_RATIO * vehicle_height,
                        "fallback lower central exhaust zone",
                    ),
                ]
                for central_top, central_bottom, caption in central_ranges:
                    central_box = self.clip_box_to_image(
                        np.array(
                            [
                                anchor_x - central_width / 2.0,
                                central_top,
                                anchor_x + central_width / 2.0,
                                central_bottom,
                            ],
                            dtype=np.float32,
                        ),
                        image_size,
                    )
                    if central_box is not None:
                        boxes.append((central_box, 0.0, caption))
            else:
                central_cy = vehicle_top + 0.40 * vehicle_height
                central_height = EXHAUST_FALLBACK_CENTRAL_HEIGHT_RATIO * vehicle_height
                central_box = self.make_box_from_center(
                    anchor_x,
                    central_cy,
                    central_width,
                    central_height,
                    image_size,
                )
                if central_box is not None:
                    boxes.append((central_box, 0.0, "fallback central exhaust zone"))

        return boxes

    def get_exhaust_raw_box_rejection_reasons(self, box, vehicle_bbox):
        if box is None or vehicle_bbox is None:
            return ["missing candidate box or vehicle bounding box"]

        vehicle_width = max(1.0, vehicle_bbox[2] - vehicle_bbox[0] + 1.0)
        vehicle_height = max(1.0, vehicle_bbox[3] - vehicle_bbox[1] + 1.0)
        vehicle_area = vehicle_width * vehicle_height
        box_w, box_h = self.get_box_size(box)
        box_area = self.get_box_area(box)
        top_ratio = (float(box[1]) - float(vehicle_bbox[1])) / vehicle_height
        bottom_ratio = (float(box[3]) - float(vehicle_bbox[1])) / vehicle_height
        reasons = []

        if box_w < EXHAUST_BOX_MIN_VEHICLE_WIDTH_RATIO * vehicle_width:
            reasons.append(
                f"box width below minimum threshold ({box_w / vehicle_width:.3f} < {EXHAUST_BOX_MIN_VEHICLE_WIDTH_RATIO:.3f})"
            )
        if box_h < EXHAUST_BOX_MIN_VEHICLE_HEIGHT_RATIO * vehicle_height:
            reasons.append(
                f"box height below minimum threshold ({box_h / vehicle_height:.3f} < {EXHAUST_BOX_MIN_VEHICLE_HEIGHT_RATIO:.3f})"
            )
        if box_w > EXHAUST_BOX_MAX_VEHICLE_WIDTH_RATIO * vehicle_width:
            reasons.append(
                f"box troppo largo ({box_w / vehicle_width:.3f} > {EXHAUST_BOX_MAX_VEHICLE_WIDTH_RATIO:.3f})"
            )
        if box_h > EXHAUST_BOX_MAX_VEHICLE_HEIGHT_RATIO * vehicle_height:
            reasons.append(
                f"box troppo alto ({box_h / vehicle_height:.3f} > {EXHAUST_BOX_MAX_VEHICLE_HEIGHT_RATIO:.3f})"
            )
        if box_area > EXHAUST_BOX_MAX_VEHICLE_AREA_RATIO * vehicle_area:
            reasons.append(
                f"box troppo esteso ({box_area / vehicle_area:.3f} > {EXHAUST_BOX_MAX_VEHICLE_AREA_RATIO:.3f})"
            )
        if top_ratio < EXHAUST_BOX_MIN_TOP_RATIO:
            reasons.append(
                f"box positioned too high in vehicle ({top_ratio:.3f} < {EXHAUST_BOX_MIN_TOP_RATIO:.3f})"
            )
        if bottom_ratio > EXHAUST_BOX_MAX_BOTTOM_RATIO:
            reasons.append(
                f"box positioned too low in vehicle ({bottom_ratio:.3f} > {EXHAUST_BOX_MAX_BOTTOM_RATIO:.3f})"
            )

        return reasons

    def extract_exhaust_candidate_masks(self, image_normal, box, sam_mask, vehicle_mask, caption=None):
        if image_normal is None or box is None:
            return []

        gray = self.get_image_gray(image_normal)
        if gray is None:
            return []

        box_region = np.zeros(gray.shape, dtype=bool)
        x1 = int(np.clip(np.floor(box[0]), 0, gray.shape[1] - 1))
        y1 = int(np.clip(np.floor(box[1]), 0, gray.shape[0] - 1))
        x2 = int(np.clip(np.ceil(box[2]), 0, gray.shape[1] - 1))
        y2 = int(np.clip(np.ceil(box[3]), 0, gray.shape[0] - 1))
        if x2 <= x1 or y2 <= y1:
            return []

        box_region[y1:y2 + 1, x1:x2 + 1] = True
        use_box_region_only = str(caption).startswith("fallback ")
        search_region = box_region
        if not use_box_region_only and sam_mask is not None and torch.any(sam_mask):
            sam_region = sam_mask.cpu().numpy().astype(bool)
            sam_intersection = np.logical_and(sam_region, box_region)
            if np.any(sam_intersection):
                search_region = sam_intersection
            else:
                search_region = np.logical_or(sam_region, box_region)

        if not np.any(search_region):
            return []

        dark_region = np.logical_and(gray <= EXHAUST_CORE_DARK_THRESHOLD, search_region)
        dark_region_u8 = (dark_region.astype(np.uint8) * 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dark_region_u8 = cv2.morphologyEx(dark_region_u8, cv2.MORPH_OPEN, kernel)
        dark_region_u8 = cv2.morphologyEx(dark_region_u8, cv2.MORPH_CLOSE, kernel)

        candidate_masks = []
        if np.any(dark_region_u8):
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_region_u8, connectivity=8)
            for label_idx in range(1, num_labels):
                area = stats[label_idx, cv2.CC_STAT_AREA]
                if area <= 0:
                    continue

                component_mask = labels == label_idx
                component_mask = np.logical_and(component_mask, search_region)
                if not np.any(component_mask):
                    continue

                candidate_masks.append(torch.from_numpy(component_mask))

        if candidate_masks:
            return candidate_masks

        if sam_mask is not None and torch.any(sam_mask):
            fallback_mask = torch.logical_and(
                sam_mask.bool(),
                torch.from_numpy(search_region),
            )
            if torch.any(fallback_mask):
                return [fallback_mask.bool()]

        return []

    def get_exhaust_mask_rejection_reasons(
        self,
        mask,
        mask_bbox,
        mask_pixels,
        vehicle_bbox,
        logo_mask=None,
        logo_bbox=None,
        taillight_masks=None,
        mirror_masks=None,
        wheel_masks=None,
    ):
        if mask is None or mask_bbox is None or vehicle_bbox is None:
            return ["missing mask bounding box"]

        vehicle_width = max(1.0, vehicle_bbox[2] - vehicle_bbox[0] + 1.0)
        vehicle_height = max(1.0, vehicle_bbox[3] - vehicle_bbox[1] + 1.0)
        vehicle_area = vehicle_width * vehicle_height
        mask_w, mask_h = self.get_box_size(mask_bbox)
        top_ratio = (float(mask_bbox[1]) - float(vehicle_bbox[1])) / vehicle_height
        bottom_ratio = (float(mask_bbox[3]) - float(vehicle_bbox[1])) / vehicle_height
        aspect_ratio = mask_w / max(1.0, mask_h)
        reasons = []

        if mask_w < EXHAUST_COMPONENT_MIN_VEHICLE_WIDTH_RATIO * vehicle_width:
            reasons.append(
                f"mask width below minimum threshold ({mask_w / vehicle_width:.3f} < {EXHAUST_COMPONENT_MIN_VEHICLE_WIDTH_RATIO:.3f})"
            )
        if mask_h < EXHAUST_COMPONENT_MIN_VEHICLE_HEIGHT_RATIO * vehicle_height:
            reasons.append(
                f"maschera troppo bassa ({mask_h / vehicle_height:.3f} < {EXHAUST_COMPONENT_MIN_VEHICLE_HEIGHT_RATIO:.3f})"
            )
        if mask_w > EXHAUST_COMPONENT_MAX_VEHICLE_WIDTH_RATIO * vehicle_width:
            reasons.append(
                f"maschera troppo larga ({mask_w / vehicle_width:.3f} > {EXHAUST_COMPONENT_MAX_VEHICLE_WIDTH_RATIO:.3f})"
            )
        if mask_h > EXHAUST_COMPONENT_MAX_VEHICLE_HEIGHT_RATIO * vehicle_height:
            reasons.append(
                f"maschera troppo alta ({mask_h / vehicle_height:.3f} > {EXHAUST_COMPONENT_MAX_VEHICLE_HEIGHT_RATIO:.3f})"
            )
        if mask_pixels > EXHAUST_COMPONENT_MAX_VEHICLE_AREA_RATIO * vehicle_area:
            reasons.append(
                f"maschera troppo estesa ({mask_pixels / vehicle_area:.3f} > {EXHAUST_COMPONENT_MAX_VEHICLE_AREA_RATIO:.3f})"
            )
        if top_ratio < EXHAUST_BOX_MIN_TOP_RATIO:
            reasons.append(
                f"maschera troppo in alto ({top_ratio:.3f} < {EXHAUST_BOX_MIN_TOP_RATIO:.3f})"
            )
        if bottom_ratio > EXHAUST_BOX_MAX_BOTTOM_RATIO:
            reasons.append(
                f"maschera troppo in basso ({bottom_ratio:.3f} > {EXHAUST_BOX_MAX_BOTTOM_RATIO:.3f})"
            )
        if aspect_ratio > EXHAUST_MAX_ASPECT_RATIO:
            reasons.append(
                f"mask aspect ratio exceeds maximum ({aspect_ratio:.3f} > {EXHAUST_MAX_ASPECT_RATIO:.3f})"
            )

        position_type, _, position_reasons = self.classify_exhaust_position(
            mask_bbox,
            vehicle_bbox,
            logo_bbox,
        )
        if position_type is None:
            reasons.extend(position_reasons)
        elif (
            position_type == "central" and
            (float(mask_pixels) / max(1.0, vehicle_area)) < EXHAUST_CENTRAL_MIN_VEHICLE_AREA_RATIO
        ):
            reasons.append(
                f"central mask area below minimum ({float(mask_pixels) / max(1.0, vehicle_area):.4f} < {EXHAUST_CENTRAL_MIN_VEHICLE_AREA_RATIO:.4f})"
            )

        max_wheel_overlap = max(
            (self.get_mask_overlap_ratio(mask, other_mask) for other_mask in (wheel_masks or {}).values()),
            default=0.0,
        )
        if max_wheel_overlap > EXHAUST_MAX_WHEEL_OVERLAP_RATIO:
            reasons.append(
                f"excessive overlap with wheel ({max_wheel_overlap:.3f} > {EXHAUST_MAX_WHEEL_OVERLAP_RATIO:.3f})"
            )

        max_taillight_overlap = max(
            (self.get_mask_overlap_ratio(mask, other_mask) for other_mask in (taillight_masks or {}).values()),
            default=0.0,
        )
        if max_taillight_overlap > EXHAUST_MAX_LIGHT_OVERLAP_RATIO:
            reasons.append(
                f"excessive overlap with taillight ({max_taillight_overlap:.3f} > {EXHAUST_MAX_LIGHT_OVERLAP_RATIO:.3f})"
            )

        max_mirror_overlap = max(
            (self.get_mask_overlap_ratio(mask, other_mask) for other_mask in (mirror_masks or {}).values()),
            default=0.0,
        )
        if max_mirror_overlap > EXHAUST_MAX_LIGHT_OVERLAP_RATIO:
            reasons.append(
                f"excessive overlap with mirror ({max_mirror_overlap:.3f} > {EXHAUST_MAX_LIGHT_OVERLAP_RATIO:.3f})"
            )

        logo_overlap = self.get_mask_overlap_ratio(mask, logo_mask)
        if logo_overlap > EXHAUST_MAX_LOGO_OVERLAP_RATIO:
            reasons.append(
                f"excessive overlap with logo ({logo_overlap:.3f} > {EXHAUST_MAX_LOGO_OVERLAP_RATIO:.3f})"
            )

        return reasons

    def build_exhaust_entry_from_mask(
        self,
        image_normal,
        mask,
        source_box,
        logit,
        caption,
        vehicle_bbox,
        logo_mask=None,
        logo_bbox=None,
        taillight_masks=None,
        mirror_masks=None,
        wheel_masks=None,
    ):
        if mask is None or not torch.any(mask):
            return None, ["maschera exhaust vuota"]

        mask = mask.bool()
        mask_bbox = self.get_mask_bbox(mask)
        if mask_bbox is None:
            return None, ["bbox maschera assente"]

        mask_pixels = int(torch.count_nonzero(mask).item())
        rejection_reasons = self.get_exhaust_mask_rejection_reasons(
            mask,
            mask_bbox,
            mask_pixels,
            vehicle_bbox,
            logo_mask=logo_mask,
            logo_bbox=logo_bbox,
            taillight_masks=taillight_masks,
            mirror_masks=mirror_masks,
            wheel_masks=wheel_masks,
        )
        if rejection_reasons:
            return None, rejection_reasons

        darkness = self.compute_mask_darkness(image_normal, mask)
        if darkness < EXHAUST_MIN_DARKNESS:
            return None, [f"darkness insufficiente ({darkness:.3f} < {EXHAUST_MIN_DARKNESS:.3f})"]

        border_contrast = self.compute_mask_border_contrast(image_normal, mask)
        if border_contrast < EXHAUST_MIN_BORDER_CONTRAST:
            return None, [f"border contrast insufficiente ({border_contrast:.1f} < {EXHAUST_MIN_BORDER_CONTRAST:.1f})"]

        border_touch_ratio = self.get_mask_box_border_touch_ratio(mask, source_box)
        if border_touch_ratio > EXHAUST_MAX_BOX_BORDER_TOUCH_RATIO:
            return None, [f"touch sul bordo box troppo alto ({border_touch_ratio:.3f} > {EXHAUST_MAX_BOX_BORDER_TOUCH_RATIO:.3f})"]

        circularity = self.compute_mask_circularity(mask)
        if circularity < EXHAUST_MIN_CIRCULARITY:
            return None, [f"circularity insufficiente ({circularity:.3f} < {EXHAUST_MIN_CIRCULARITY:.3f})"]

        position_type, region_score, _ = self.classify_exhaust_position(
            mask_bbox,
            vehicle_bbox,
            logo_bbox,
        )

        vehicle_height = max(1.0, vehicle_bbox[3] - vehicle_bbox[1] + 1.0)
        vehicle_top = float(vehicle_bbox[1])
        bbox_area = max(1.0, self.get_box_area(mask_bbox))
        compactness = float(mask_pixels) / float(bbox_area)
        mask_cx, mask_cy = self.get_box_center(mask_bbox)
        mask_w, mask_h = self.get_box_size(mask_bbox)
        side = self.get_exhaust_side(mask_bbox, vehicle_bbox, logo_bbox)
        bottom_ratio = (float(mask_bbox[3]) - vehicle_top) / vehicle_height
        source_cx, source_cy = self.get_box_center(source_box)
        source_w, source_h = self.get_box_size(source_box)
        alignment_penalty = (
            abs(mask_cx - source_cx) / max(1.0, source_w)
            + abs(mask_cy - source_cy) / max(1.0, source_h)
        )
        score = (
            float(logit)
            + 0.95 * darkness
            + 0.012 * border_contrast
            + 0.45 * circularity
            + 0.35 * compactness
            + 0.45 * region_score
            + 0.10 * bottom_ratio
            - 0.20 * alignment_penalty
        )

        return {
            "side": side,
            "box": source_box,
            "mask": mask,
            "mask_bbox": mask_bbox,
            "caption": caption,
            "logit": float(logit),
            "score": score,
            "cx": mask_cx,
            "cy": mask_cy,
            "width": mask_w,
            "height": mask_h,
            "pixel_count": mask_pixels,
            "darkness": darkness,
            "contrast": border_contrast,
            "circularity": circularity,
            "position": position_type,
            "border_touch": border_touch_ratio,
        }, None

    def build_exhaust_entries(
        self,
        image_normal,
        transformed_image,
        image_size,
        vehicle_mask,
        vehicle_bbox,
        logo_mask=None,
        logo_bbox=None,
        taillight_masks=None,
        mirror_masks=None,
        wheel_masks=None,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    ):
        if vehicle_bbox is None:
            return []

        candidates = self.collect_candidates(
            transformed_image,
            PART_SPECS["exhaust tips"]["captions"],
            image_size,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        candidates.extend(self.build_exhaust_fallback_boxes(image_size, vehicle_bbox, logo_bbox))
        candidates = self.deduplicate_candidates(candidates, iou_threshold=0.25)
        if not candidates:
            return []

        raw_entries = []
        for candidate_idx, (box, logit, caption) in enumerate(
            sorted(candidates, key=lambda item: item[1], reverse=True),
            start=1,
        ):
            raw_reasons = []
            if not str(caption).startswith("fallback "):
                raw_reasons = self.get_exhaust_raw_box_rejection_reasons(box, vehicle_bbox)
            if raw_reasons:
                continue

            sam_mask = self.predict_mask_from_box(image_normal, box)
            component_masks = self.extract_exhaust_candidate_masks(
                image_normal,
                box,
                sam_mask,
                vehicle_mask,
                caption=caption,
            )
            if not component_masks:
                continue

            for component_idx, component_mask in enumerate(component_masks, start=1):
                entry, mask_reasons = self.build_exhaust_entry_from_mask(
                    image_normal,
                    component_mask,
                    box,
                    logit,
                    caption,
                    vehicle_bbox,
                    logo_mask=logo_mask,
                    logo_bbox=logo_bbox,
                    taillight_masks=taillight_masks,
                    mirror_masks=mirror_masks,
                    wheel_masks=wheel_masks,
                )

                if entry is None:
                    mask_bbox = self.get_mask_bbox(component_mask)
                    continue

                raw_entries.append(entry)

        unique_entries = []
        for entry in sorted(raw_entries, key=lambda item: item["score"], reverse=True):
            overlapping_entry = next(
                (
                    kept
                    for kept in unique_entries
                    if self.calculate_iou(entry["mask_bbox"], kept["mask_bbox"]) > 0.30
                ),
                None,
            )
            if overlapping_entry is not None:
                continue
            unique_entries.append(entry)

        return unique_entries

    def recover_exhaust_entries_from_symmetry(
        self,
        image_normal,
        image_size,
        selected_entries,
        vehicle_bbox,
        logo_mask=None,
        logo_bbox=None,
        taillight_masks=None,
        mirror_masks=None,
        wheel_masks=None,
    ):
        if not selected_entries or vehicle_bbox is None:
            return []

        axis_x, _ = self.get_exhaust_anchor(vehicle_bbox, logo_bbox)
        if axis_x is None:
            return []

        left_entries = [entry for entry in selected_entries if entry.get("side") == "left"]
        right_entries = [entry for entry in selected_entries if entry.get("side") == "right"]
        if not left_entries and not right_entries:
            return []

        source_entries = []
        existing_target_entries = []
        missing_side = None
        target_missing_count = 0

        if not left_entries or not right_entries:
            if left_entries:
                source_entries = sorted(left_entries, key=lambda item: item["score"], reverse=True)
                existing_target_entries = []
                missing_side = "right"
            else:
                source_entries = sorted(right_entries, key=lambda item: item["score"], reverse=True)
                existing_target_entries = []
                missing_side = "left"

            if not all(entry.get("position") == "central" for entry in source_entries):
                return []
            target_missing_count = len(source_entries)
        elif len(left_entries) != len(right_entries):
            if len(left_entries) < len(right_entries):
                source_entries = sorted(right_entries, key=lambda item: item["score"], reverse=True)
                existing_target_entries = list(left_entries)
                missing_side = "left"
            else:
                source_entries = sorted(left_entries, key=lambda item: item["score"], reverse=True)
                existing_target_entries = list(right_entries)
                missing_side = "right"

            if not all(entry.get("position") == "lateral" for entry in (left_entries + right_entries)):
                return []
            target_missing_count = abs(len(left_entries) - len(right_entries))
        else:
            return []

        vehicle_width = max(1.0, vehicle_bbox[2] - vehicle_bbox[0] + 1.0)
        vehicle_height = max(1.0, vehicle_bbox[3] - vehicle_bbox[1] + 1.0)
        recovered_entries = []

        for source_idx, source_entry in enumerate(source_entries, start=1):
            if len(recovered_entries) >= target_missing_count:
                break

            target_cx = 2.0 * axis_x - source_entry["cx"]
            target_cy = source_entry["cy"]
            mirrored_counterpart = next(
                (
                    kept
                    for kept in (existing_target_entries + recovered_entries)
                    if (
                        abs(kept["cx"] - target_cx) / vehicle_width <= EXHAUST_SYMMETRY_MAX_X_RATIO and
                        abs(kept["cy"] - target_cy) / vehicle_height <= EXHAUST_SYMMETRY_MAX_Y_RATIO
                    )
                ),
                None,
            )
            if mirrored_counterpart is not None:
                continue

            search_width = max(56.0, source_entry["width"] * 1.45)
            search_height = max(56.0, source_entry["height"] * 1.45)
            synthetic_box = self.make_box_from_center(
                target_cx,
                target_cy,
                search_width,
                search_height,
                image_size,
            )
            if synthetic_box is None:
                continue

            sam_mask = self.predict_mask_from_box(image_normal, synthetic_box)
            component_masks = self.extract_exhaust_candidate_masks(
                image_normal,
                synthetic_box,
                sam_mask,
                None,
                caption=f"symmetry {source_entry['caption']}",
            )
            if not component_masks:
                continue

            best_recovered_entry = None
            best_recovered_score = None
            for component_idx, component_mask in enumerate(component_masks, start=1):
                entry, reasons = self.build_exhaust_entry_from_mask(
                    image_normal,
                    component_mask,
                    synthetic_box,
                    source_entry["logit"],
                    f"symmetry {source_entry['caption']}",
                    vehicle_bbox,
                    logo_mask=logo_mask,
                    logo_bbox=logo_bbox,
                    taillight_masks=taillight_masks,
                    mirror_masks=mirror_masks,
                    wheel_masks=wheel_masks,
                )
                if entry is None:
                    continue

                if entry["side"] != missing_side:
                    continue

                x_ratio = abs(entry["cx"] - target_cx) / vehicle_width
                y_ratio = abs(entry["cy"] - source_entry["cy"]) / vehicle_height
                shape_similarity = self.compute_reflected_shape_similarity(
                    source_entry["mask"],
                    entry["mask"],
                )
                if x_ratio > EXHAUST_SYMMETRY_MAX_X_RATIO or y_ratio > EXHAUST_SYMMETRY_MAX_Y_RATIO:
                    continue
                if shape_similarity < 0.10:
                    continue

                recovered_score = entry["score"] + 0.50 * shape_similarity
                if best_recovered_score is None or recovered_score > best_recovered_score:
                    best_recovered_score = recovered_score
                    best_recovered_entry = entry

            if best_recovered_entry is None:
                continue

            overlapping_entry = next(
                (
                    kept
                    for kept in (selected_entries + recovered_entries)
                    if self.calculate_iou(best_recovered_entry["mask_bbox"], kept["mask_bbox"]) > 0.25
                ),
                None,
            )
            if overlapping_entry is not None:
                continue

            recovered_entries.append(best_recovered_entry)

        return recovered_entries

    def select_exhaust_entries(self, entries, vehicle_bbox):
        if not entries or vehicle_bbox is None:
            return []

        vehicle_height = max(1.0, vehicle_bbox[3] - vehicle_bbox[1] + 1.0)
        sorted_entries = sorted(entries, key=lambda item: item["score"], reverse=True)
        best_entry = sorted_entries[0]
        band_center_y = best_entry["cy"]

        selected_entries = []
        for entry in sorted_entries:
            vertical_gap_ratio = abs(entry["cy"] - band_center_y) / vehicle_height
            if vertical_gap_ratio > EXHAUST_MAX_VERTICAL_GROUP_GAP_RATIO:
                continue

            overlapping_entry = next(
                (
                    kept
                    for kept in selected_entries
                    if self.calculate_iou(entry["mask_bbox"], kept["mask_bbox"]) > 0.25
                ),
                None,
            )
            if overlapping_entry is not None:
                continue

            selected_entries.append(entry)
            if len(selected_entries) >= EXHAUST_MAX_CANDIDATES:
                break

        return selected_entries

    def build_exhaust_mask(
        self,
        image_normal,
        transformed_image,
        image_size,
        vehicle_mask,
        vehicle_bbox,
        logo_mask=None,
        logo_bbox=None,
        taillight_masks=None,
        mirror_masks=None,
        wheel_masks=None,
        source_name=None,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    ):
        manual_override = self.get_manual_component_override(source_name, "exhaust tips")
        manual_mask = None
        if manual_override is not None:
            manual_mask = self.build_manual_exhaust_mask_from_points(
                image_normal,
                image_size,
                vehicle_mask,
                vehicle_bbox,
                manual_override.get("points"),
                logo_mask=logo_mask,
                taillight_masks=taillight_masks,
                mirror_masks=mirror_masks,
                wheel_masks=wheel_masks,
            )
            if manual_mask is not None and manual_override.get("mode") == "replace":
                return manual_mask

        entries = self.build_exhaust_entries(
            image_normal,
            transformed_image,
            image_size,
            vehicle_mask,
            vehicle_bbox,
            logo_mask=logo_mask,
            logo_bbox=logo_bbox,
            taillight_masks=taillight_masks,
            mirror_masks=mirror_masks,
            wheel_masks=wheel_masks,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        selected_entries = self.select_exhaust_entries(
            entries,
            vehicle_bbox,
        )
        recovered_entries = self.recover_exhaust_entries_from_symmetry(
            image_normal,
            image_size,
            selected_entries,
            vehicle_bbox,
            logo_mask=logo_mask,
            logo_bbox=logo_bbox,
            taillight_masks=taillight_masks,
            mirror_masks=mirror_masks,
            wheel_masks=wheel_masks,
        )
        if recovered_entries:
            selected_entries = selected_entries + recovered_entries
        if not selected_entries and manual_mask is None:
            return None

        exhaust_mask = None
        for entry in selected_entries:
            if exhaust_mask is None:
                exhaust_mask = entry["mask"].bool()
            else:
                exhaust_mask = torch.logical_or(exhaust_mask, entry["mask"].bool())

        if manual_mask is not None:
            if exhaust_mask is None:
                exhaust_mask = manual_mask.bool()
            else:
                exhaust_mask = torch.logical_or(exhaust_mask, manual_mask.bool())

        masks_to_remove = []
        if logo_mask is not None:
            masks_to_remove.append(logo_mask)
        masks_to_remove.extend((taillight_masks or {}).values())
        masks_to_remove.extend((mirror_masks or {}).values())
        masks_to_remove.extend((wheel_masks or {}).values())
        exhaust_mask = self.subtract_masks(exhaust_mask, masks_to_remove)
        if exhaust_mask is None or not torch.any(exhaust_mask):
            return None

        return exhaust_mask.bool()

    def get_taillight_mask_rejection_reasons(
        self,
        mask_bbox,
        mask_pixels,
        vehicle_bbox,
        logo_bbox,
        max_width_ratio=TAILLIGHT_MAX_VEHICLE_WIDTH_RATIO,
    ):
        if mask_bbox is None or vehicle_bbox is None:
            return ["missing mask bounding box"]

        axis_x, anchor_y = self.get_taillight_anchor(vehicle_bbox, logo_bbox)
        if axis_x is None or anchor_y is None:
            return ["missing taillight reference axis"]

        vehicle_width = max(1.0, vehicle_bbox[2] - vehicle_bbox[0] + 1.0)
        vehicle_height = max(1.0, vehicle_bbox[3] - vehicle_bbox[1] + 1.0)
        vehicle_area = vehicle_width * vehicle_height
        min_top = float(vehicle_bbox[1]) + TAILLIGHT_MIN_TOP_RATIO * vehicle_height
        max_bottom = float(vehicle_bbox[1]) + TAILLIGHT_MAX_BOTTOM_RATIO * vehicle_height

        mask_cx, mask_cy = self.get_box_center(mask_bbox)
        mask_w, mask_h = self.get_box_size(mask_bbox)
        side_offset_ratio = abs(mask_cx - axis_x) / vehicle_width
        reasons = []

        if side_offset_ratio < TAILLIGHT_MIN_SIDE_OFFSET_RATIO:
            reasons.append(
                f"maschera troppo centrale ({side_offset_ratio:.3f} < {TAILLIGHT_MIN_SIDE_OFFSET_RATIO:.3f})"
            )
        if side_offset_ratio > TAILLIGHT_MAX_SIDE_OFFSET_RATIO:
            reasons.append(
                f"maschera troppo laterale ({side_offset_ratio:.3f} > {TAILLIGHT_MAX_SIDE_OFFSET_RATIO:.3f})"
            )
        if mask_w < TAILLIGHT_MIN_VEHICLE_WIDTH_RATIO * vehicle_width:
            reasons.append(
                f"maschera troppo stretta ({mask_w / vehicle_width:.3f} < {TAILLIGHT_MIN_VEHICLE_WIDTH_RATIO:.3f})"
            )
        if mask_h < TAILLIGHT_MIN_VEHICLE_HEIGHT_RATIO * vehicle_height:
            reasons.append(
                f"maschera troppo bassa ({mask_h / vehicle_height:.3f} < {TAILLIGHT_MIN_VEHICLE_HEIGHT_RATIO:.3f})"
            )
        if mask_w > max_width_ratio * vehicle_width:
            reasons.append(
                f"maschera troppo larga ({mask_w / vehicle_width:.3f} > {max_width_ratio:.3f})"
            )
        if mask_h > TAILLIGHT_MAX_VEHICLE_HEIGHT_RATIO * vehicle_height:
            reasons.append(
                f"maschera troppo alta ({mask_h / vehicle_height:.3f} > {TAILLIGHT_MAX_VEHICLE_HEIGHT_RATIO:.3f})"
            )
        if mask_pixels > TAILLIGHT_MAX_VEHICLE_AREA_RATIO * vehicle_area:
            reasons.append(
                f"maschera troppo estesa ({mask_pixels / vehicle_area:.3f} > {TAILLIGHT_MAX_VEHICLE_AREA_RATIO:.3f})"
            )
        if mask_cy < min_top or mask_cy > max_bottom:
            reasons.append(
                f"mask vertical center outside valid band (cy={mask_cy:.1f}, band=[{min_top:.1f}, {max_bottom:.1f}])"
            )
        if logo_bbox is not None and abs(mask_cy - anchor_y) > TAILLIGHT_MAX_VERTICAL_FROM_LOGO_RATIO * vehicle_height:
            reasons.append(
                f"mask vertical distance from logo exceeds threshold ({abs(mask_cy - anchor_y) / vehicle_height:.3f} > {TAILLIGHT_MAX_VERTICAL_FROM_LOGO_RATIO:.3f})"
            )

        return reasons

    def build_taillight_entry_from_mask(
        self,
        image_normal,
        mask,
        box,
        logit,
        caption,
        vehicle_bbox,
        logo_bbox,
        max_width_ratio=TAILLIGHT_MAX_VEHICLE_WIDTH_RATIO,
    ):
        if mask is None or not torch.any(mask):
            return None, ["empty SAM mask"]

        mask = mask.bool()
        mask_bbox = self.get_mask_bbox(mask)
        if mask_bbox is None:
            return None, ["bbox maschera assente"]

        mask_pixels = int(torch.count_nonzero(mask).item())
        mask_reasons = self.get_taillight_mask_rejection_reasons(
            mask_bbox,
            mask_pixels,
            vehicle_bbox,
            logo_bbox,
            max_width_ratio=max_width_ratio,
        )
        if mask_reasons:
            return None, mask_reasons

        redness = self.compute_mask_redness(image_normal, mask)
        if redness < TAILLIGHT_MIN_REDNESS:
            return None, [f"redness insufficiente ({redness:.3f} < {TAILLIGHT_MIN_REDNESS:.3f})"]

        axis_x, anchor_y = self.get_taillight_anchor(vehicle_bbox, logo_bbox)
        vehicle_width = max(1.0, vehicle_bbox[2] - vehicle_bbox[0] + 1.0)
        vehicle_height = max(1.0, vehicle_bbox[3] - vehicle_bbox[1] + 1.0)
        mask_cx, mask_cy = self.get_box_center(mask_bbox)
        mask_w, mask_h = self.get_box_size(mask_bbox)
        side_offset_ratio = abs(mask_cx - axis_x) / vehicle_width
        preferred_bottom = float(vehicle_bbox[1]) + 0.50 * vehicle_height
        low_penalty = max(0.0, (mask_cy - preferred_bottom) / vehicle_height)
        aspect_ratio = mask_w / max(1.0, mask_h)
        thin_reward = min(aspect_ratio / 5.0, 1.0)

        side = "left" if mask_cx < axis_x else "right"
        vertical_penalty = abs(mask_cy - anchor_y) / vehicle_height if anchor_y is not None else 0.0
        offset_penalty = abs(side_offset_ratio - 0.34)
        score = (
            float(logit)
            + 0.70 * redness
            + 0.12 * thin_reward
            - 1.05 * vertical_penalty
            - 0.35 * offset_penalty
            - 1.20 * low_penalty
        )

        return {
            "side": side,
            "box": box,
            "mask": mask,
            "mask_bbox": mask_bbox,
            "caption": caption,
            "logit": float(logit),
            "score": score,
            "cx": mask_cx,
            "cy": mask_cy,
            "width": mask_w,
            "height": mask_h,
            "pixel_count": mask_pixels,
            "redness": redness,
        }, None

    def is_taillight_bar_candidate(self, box, vehicle_bbox, logo_bbox):
        if box is None or vehicle_bbox is None:
            return False

        axis_x, anchor_y = self.get_taillight_anchor(vehicle_bbox, logo_bbox)
        if axis_x is None or anchor_y is None:
            return False

        vehicle_width = max(1.0, vehicle_bbox[2] - vehicle_bbox[0] + 1.0)
        vehicle_height = max(1.0, vehicle_bbox[3] - vehicle_bbox[1] + 1.0)
        box_w, box_h = self.get_box_size(box)
        _, box_cy = self.get_box_center(box)
        width_ratio = box_w / vehicle_width
        height_ratio = box_h / vehicle_height
        vertical_ratio = abs(box_cy - anchor_y) / vehicle_height

        return (
            TAILLIGHT_BAR_MIN_VEHICLE_WIDTH_RATIO <= width_ratio <= TAILLIGHT_BAR_MAX_VEHICLE_WIDTH_RATIO and
            height_ratio <= TAILLIGHT_BAR_MAX_VEHICLE_HEIGHT_RATIO and
            float(vehicle_bbox[1]) + 0.10 * vehicle_height <= box_cy <= float(vehicle_bbox[1]) + 0.52 * vehicle_height and
            vertical_ratio <= TAILLIGHT_BAR_MAX_VERTICAL_FROM_LOGO_RATIO
        )

    def build_taillight_entries_from_bar_candidate(
        self,
        image_normal,
        image_size,
        box,
        logit,
        caption,
        vehicle_bbox,
        logo_bbox,
    ):
        axis_x, _ = self.get_taillight_anchor(vehicle_bbox, logo_bbox)
        vehicle_width = max(1.0, vehicle_bbox[2] - vehicle_bbox[0] + 1.0)
        box_w, box_h = self.get_box_size(box)
        center_gap = max(18.0, TAILLIGHT_BAR_CENTER_GAP_RATIO * vehicle_width)
        vertical_pad = max(6.0, 0.25 * box_h)
        y1 = box[1] - vertical_pad
        y2 = box[3] + vertical_pad

        left_box = self.clip_box_to_image(
            np.array(
                [
                    box[0],
                    y1,
                    min(axis_x - center_gap, box[0] + 0.58 * box_w),
                    y2,
                ],
                dtype=np.float32,
            ),
            image_size,
        )
        right_box = self.clip_box_to_image(
            np.array(
                [
                    max(axis_x + center_gap, box[2] - 0.58 * box_w),
                    y1,
                    box[2],
                    y2,
                ],
                dtype=np.float32,
            ),
            image_size,
        )

        entries = []
        for side_name, subbox in (("left", left_box), ("right", right_box)):
            if subbox is None:
                continue

            submask = self.predict_mask_from_box(image_normal, subbox)
            entry, reasons = self.build_taillight_entry_from_mask(
                image_normal,
                submask,
                subbox,
                logit,
                f"split {caption}",
                vehicle_bbox,
                logo_bbox,
            )
            if entry is None:
                mask_bbox = None
                if submask is not None and torch.any(submask):
                    mask_bbox = self.get_mask_bbox(submask.bool())
                continue

            if entry["side"] != side_name:
                continue

            entries.append(entry)
        return entries

    def recover_missing_taillight_from_symmetry(
        self,
        image_normal,
        image_size,
        source_entry,
        missing_side,
        vehicle_bbox,
        logo_bbox,
    ):
        if source_entry is None or vehicle_bbox is None or missing_side not in {"left", "right"}:
            return None

        axis_x, _ = self.get_taillight_anchor(vehicle_bbox, logo_bbox)
        if axis_x is None:
            return None

        target_cx = 2.0 * axis_x - source_entry["cx"]
        target_cy = source_entry["cy"]
        search_width = max(56.0, source_entry["width"] * 1.35)
        search_height = max(34.0, source_entry["height"] * 1.45)
        synthetic_box = self.make_box_from_center(
            target_cx,
            target_cy,
            search_width,
            search_height,
            image_size,
        )
        if synthetic_box is None:
            return None

        mask = self.predict_mask_from_box(image_normal, synthetic_box)
        entry, reasons = self.build_taillight_entry_from_mask(
            image_normal,
            mask,
            synthetic_box,
            source_entry["logit"],
            f"mirrored {source_entry['caption']}",
            vehicle_bbox,
            logo_bbox,
        )
        if entry is None:
            return None

        if entry["side"] != missing_side:
            return None

        vehicle_width = max(1.0, vehicle_bbox[2] - vehicle_bbox[0] + 1.0)
        vehicle_height = max(1.0, vehicle_bbox[3] - vehicle_bbox[1] + 1.0)
        expected_cx = 2.0 * axis_x - source_entry["cx"]
        x_ratio = abs(entry["cx"] - expected_cx) / vehicle_width
        y_ratio = abs(entry["cy"] - source_entry["cy"]) / vehicle_height
        if x_ratio > 0.12 or y_ratio > 0.10:
            return None

        return entry

    def recover_taillight_from_anchor_box(
        self,
        image_normal,
        image_size,
        side,
        vehicle_bbox,
        logo_bbox,
    ):
        if vehicle_bbox is None or side not in {"left", "right"}:
            return None

        axis_x, anchor_y = self.get_taillight_anchor(vehicle_bbox, logo_bbox)
        if axis_x is None or anchor_y is None:
            return None

        vehicle_width = max(1.0, vehicle_bbox[2] - vehicle_bbox[0] + 1.0)
        vehicle_height = max(1.0, vehicle_bbox[3] - vehicle_bbox[1] + 1.0)
        center_x = axis_x - 0.32 * vehicle_width if side == "left" else axis_x + 0.32 * vehicle_width
        center_y = anchor_y - 0.02 * vehicle_height
        search_width = max(64.0, 0.24 * vehicle_width)
        search_height = max(38.0, 0.11 * vehicle_height)
        synthetic_box = self.make_box_from_center(
            center_x,
            center_y,
            search_width,
            search_height,
            image_size,
        )
        if synthetic_box is None:
            return None

        mask = self.predict_mask_from_box(image_normal, synthetic_box)
        entry, reasons = self.build_taillight_entry_from_mask(
            image_normal,
            mask,
            synthetic_box,
            0.0,
            f"anchor fallback {side} taillight",
            vehicle_bbox,
            logo_bbox,
        )
        if entry is None:
            return None

        if entry["side"] != side:
            return None

        return entry

    def recover_taillight_from_mirror_box(
        self,
        image_normal,
        image_size,
        side,
        mirror_mask,
        vehicle_bbox,
        logo_bbox,
    ):
        if vehicle_bbox is None or mirror_mask is None or side not in {"left", "right"}:
            return None

        mirror_bbox = self.get_mask_bbox(mirror_mask)
        if mirror_bbox is None:
            return None

        axis_x, _ = self.get_taillight_anchor(vehicle_bbox, logo_bbox)
        if axis_x is None:
            return None

        vehicle_width = max(1.0, vehicle_bbox[2] - vehicle_bbox[0] + 1.0)
        vehicle_height = max(1.0, vehicle_bbox[3] - vehicle_bbox[1] + 1.0)
        search_top = max(
            float(vehicle_bbox[1]) + TAILLIGHT_MIN_TOP_RATIO * vehicle_height,
            float(mirror_bbox[3]) - 0.02 * vehicle_height,
        )
        search_bottom = min(
            float(vehicle_bbox[1]) + TAILLIGHT_MAX_BOTTOM_RATIO * vehicle_height,
            float(mirror_bbox[3]) + max(0.12 * vehicle_height, 2.0 * max(1.0, mirror_bbox[3] - mirror_bbox[1])),
        )

        if side == "left":
            search_left = max(float(vehicle_bbox[0]), float(mirror_bbox[0]) - 0.04 * vehicle_width)
            search_right = min(axis_x - 0.02 * vehicle_width, float(mirror_bbox[2]) + 0.22 * vehicle_width)
        else:
            search_left = max(axis_x + 0.02 * vehicle_width, float(mirror_bbox[0]) - 0.22 * vehicle_width)
            search_right = min(float(vehicle_bbox[2]), float(mirror_bbox[2]) + 0.04 * vehicle_width)

        synthetic_box = self.clip_box_to_image(
            np.array([search_left, search_top, search_right, search_bottom], dtype=np.float32),
            image_size,
        )
        if synthetic_box is None:
            return None

        mask = self.predict_mask_from_box(image_normal, synthetic_box)
        entry, reasons = self.build_taillight_entry_from_mask(
            image_normal,
            mask,
            synthetic_box,
            0.0,
            f"mirror fallback {side} taillight",
            vehicle_bbox,
            logo_bbox,
            max_width_ratio=TAILLIGHT_FALLBACK_MAX_VEHICLE_WIDTH_RATIO,
        )
        if entry is None:
            return None

        if entry["side"] != side:
            return None

        mirror_overlap = self.get_mask_overlap_ratio(entry["mask"], mirror_mask)
        vertical_gap = entry["cy"] - float(mirror_bbox[3])
        if mirror_overlap > TAILLIGHT_MIRROR_FALLBACK_MAX_OVERLAP_RATIO:
            return None
        if vertical_gap < -0.03 * vehicle_height or vertical_gap > 0.22 * vehicle_height:
            return None

        return entry

    def build_taillight_entries(
        self,
        image_normal,
        transformed_image,
        image_size,
        vehicle_bbox,
        logo_bbox,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    ):
        if vehicle_bbox is None:
            return []

        candidates = self.collect_candidates(
            transformed_image,
            PART_SPECS["taillight"]["captions"],
            image_size,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        if not candidates:
            return []

        axis_x, anchor_y = self.get_taillight_anchor(vehicle_bbox, logo_bbox)
        if axis_x is None:
            return []

        vehicle_width = max(1.0, vehicle_bbox[2] - vehicle_bbox[0] + 1.0)
        vehicle_height = max(1.0, vehicle_bbox[3] - vehicle_bbox[1] + 1.0)
        vehicle_area = vehicle_width * vehicle_height
        min_top = float(vehicle_bbox[1]) + TAILLIGHT_MIN_TOP_RATIO * vehicle_height
        max_bottom = float(vehicle_bbox[1]) + TAILLIGHT_MAX_BOTTOM_RATIO * vehicle_height

        raw_entries = []
        bar_entries = []
        for candidate_idx, (box, logit, caption) in enumerate(
            sorted(candidates, key=lambda item: item[1], reverse=True),
            start=1,
        ):
            raw_cx, raw_cy = self.get_box_center(box)
            box_w, box_h = self.get_box_size(box)
            box_area = self.get_box_area(box)
            side_offset_ratio = abs(raw_cx - axis_x) / vehicle_width
            raw_reasons = []

            if side_offset_ratio < TAILLIGHT_MIN_SIDE_OFFSET_RATIO:
                raw_reasons.append(
                    f"box troppo centrale ({side_offset_ratio:.3f} < {TAILLIGHT_MIN_SIDE_OFFSET_RATIO:.3f})"
                )
            if side_offset_ratio > TAILLIGHT_MAX_SIDE_OFFSET_RATIO:
                raw_reasons.append(
                    f"box troppo laterale ({side_offset_ratio:.3f} > {TAILLIGHT_MAX_SIDE_OFFSET_RATIO:.3f})"
                )
            if box_w < TAILLIGHT_MIN_VEHICLE_WIDTH_RATIO * vehicle_width:
                raw_reasons.append(
                    f"box troppo stretto ({box_w / vehicle_width:.3f} < {TAILLIGHT_MIN_VEHICLE_WIDTH_RATIO:.3f})"
                )
            if box_h < TAILLIGHT_MIN_VEHICLE_HEIGHT_RATIO * vehicle_height:
                raw_reasons.append(
                    f"box troppo basso ({box_h / vehicle_height:.3f} < {TAILLIGHT_MIN_VEHICLE_HEIGHT_RATIO:.3f})"
                )
            if box_w > TAILLIGHT_MAX_VEHICLE_WIDTH_RATIO * vehicle_width:
                raw_reasons.append(
                    f"box troppo largo ({box_w / vehicle_width:.3f} > {TAILLIGHT_MAX_VEHICLE_WIDTH_RATIO:.3f})"
                )
            if box_h > TAILLIGHT_MAX_VEHICLE_HEIGHT_RATIO * vehicle_height:
                raw_reasons.append(
                    f"box troppo alto ({box_h / vehicle_height:.3f} > {TAILLIGHT_MAX_VEHICLE_HEIGHT_RATIO:.3f})"
                )
            if box_area > TAILLIGHT_MAX_VEHICLE_AREA_RATIO * vehicle_area:
                raw_reasons.append(
                    f"box troppo esteso ({box_area / vehicle_area:.3f} > {TAILLIGHT_MAX_VEHICLE_AREA_RATIO:.3f})"
                )
            if raw_cy < min_top or raw_cy > max_bottom:
                raw_reasons.append(
                    f"box vertical center outside valid band (cy={raw_cy:.1f}, band=[{min_top:.1f}, {max_bottom:.1f}])"
                )
            if logo_bbox is not None and abs(raw_cy - anchor_y) > TAILLIGHT_MAX_VERTICAL_FROM_LOGO_RATIO * vehicle_height:
                raw_reasons.append(
                    f"box vertical distance from logo exceeds threshold ({abs(raw_cy - anchor_y) / vehicle_height:.3f} > {TAILLIGHT_MAX_VERTICAL_FROM_LOGO_RATIO:.3f})"
                )

            if self.is_taillight_bar_candidate(box, vehicle_bbox, logo_bbox):
                split_entries = self.build_taillight_entries_from_bar_candidate(
                    image_normal,
                    image_size,
                    box,
                    logit,
                    caption,
                    vehicle_bbox,
                    logo_bbox,
                )
                bar_entries.extend(split_entries)
                continue

            if raw_reasons:
                continue

            mask = self.predict_mask_from_box(image_normal, box)
            entry, mask_reasons = self.build_taillight_entry_from_mask(
                image_normal,
                mask,
                box,
                logit,
                caption,
                vehicle_bbox,
                logo_bbox,
            )

            if entry is None:
                mask_bbox = None
                if mask is not None and torch.any(mask):
                    mask_bbox = self.get_mask_bbox(mask.bool())
                continue

            raw_entries.append(entry)

        raw_entries.extend(bar_entries)

        unique_entries = []
        for entry in sorted(raw_entries, key=lambda item: item["score"], reverse=True):
            overlapping_entry = next(
                (
                    kept
                    for kept in unique_entries
                    if entry["side"] == kept["side"] and
                    self.calculate_iou(entry["mask_bbox"], kept["mask_bbox"]) > 0.30
                ),
                None,
            )
            if overlapping_entry is not None:
                continue
            unique_entries.append(entry)

        limited_entries = []
        side_counts = {"left": 0, "right": 0}
        for entry in unique_entries:
            if side_counts[entry["side"]] >= TAILLIGHT_MAX_CANDIDATES_PER_SIDE:
                continue
            limited_entries.append(entry)
            side_counts[entry["side"]] += 1

        return limited_entries

    def score_taillight_pair(self, left_entry, right_entry, vehicle_bbox, logo_bbox):
        axis_x, anchor_y = self.get_taillight_anchor(vehicle_bbox, logo_bbox)
        if axis_x is None or anchor_y is None:
            return None

        vehicle_width = max(1.0, vehicle_bbox[2] - vehicle_bbox[0] + 1.0)
        vehicle_height = max(1.0, vehicle_bbox[3] - vehicle_bbox[1] + 1.0)
        vehicle_diag = max(1.0, float(np.hypot(vehicle_width, vehicle_height)))

        left_dx = axis_x - left_entry["cx"]
        right_dx = right_entry["cx"] - axis_x
        if left_dx <= 0 or right_dx <= 0:
            return None

        left_dy = left_entry["cy"] - anchor_y
        right_dy = right_entry["cy"] - anchor_y

        mirror_penalty = abs(left_dx - right_dx) / vehicle_width
        vertical_penalty = abs(left_entry["cy"] - right_entry["cy"]) / vehicle_height
        radial_penalty = abs(
            np.hypot(left_dx, left_dy) - np.hypot(right_dx, right_dy)
        ) / vehicle_diag
        width_penalty = abs(left_entry["width"] - right_entry["width"]) / max(
            1.0,
            left_entry["width"],
            right_entry["width"],
        )
        height_penalty = abs(left_entry["height"] - right_entry["height"]) / max(
            1.0,
            left_entry["height"],
            right_entry["height"],
        )
        area_penalty = abs(left_entry["pixel_count"] - right_entry["pixel_count"]) / max(
            1.0,
            left_entry["pixel_count"],
            right_entry["pixel_count"],
        )
        redness_gap = abs(left_entry["redness"] - right_entry["redness"])
        shape_similarity = self.compute_reflected_shape_similarity(
            left_entry["mask"],
            right_entry["mask"],
        )

        return (
            left_entry["score"]
            + right_entry["score"]
            + 1.10 * shape_similarity
            + 0.55 * (left_entry["redness"] + right_entry["redness"])
            - 1.80 * mirror_penalty
            - 1.20 * vertical_penalty
            - 0.90 * radial_penalty
            - 0.75 * (width_penalty + height_penalty)
            - 0.50 * area_penalty
            - 0.40 * redness_gap
        )

    def get_taillight_pair_rejection_reasons(self, left_entry, right_entry, vehicle_bbox, logo_bbox, pair_score):
        if left_entry is None or right_entry is None or vehicle_bbox is None:
            return ["incomplete taillight pair"]

        axis_x, anchor_y = self.get_taillight_anchor(vehicle_bbox, logo_bbox)
        if axis_x is None or anchor_y is None:
            return ["missing taillight reference axis"]

        vehicle_width = max(1.0, vehicle_bbox[2] - vehicle_bbox[0] + 1.0)
        vehicle_height = max(1.0, vehicle_bbox[3] - vehicle_bbox[1] + 1.0)
        left_dx = axis_x - left_entry["cx"]
        right_dx = right_entry["cx"] - axis_x
        shape_similarity = self.compute_reflected_shape_similarity(
            left_entry["mask"],
            right_entry["mask"],
        )

        reasons = []
        if pair_score < TAILLIGHT_PAIR_MIN_SCORE:
            reasons.append(f"pair score below minimum ({pair_score:.3f} < {TAILLIGHT_PAIR_MIN_SCORE:.3f})")
        if left_dx <= 0 or right_dx <= 0:
            reasons.append("one candidate is not positioned on the expected lateral side")
        if abs(left_dx - right_dx) / vehicle_width > 0.22:
            reasons.append("asymmetric horizontal offsets from center axis exceed threshold")
        if abs(left_entry["cy"] - right_entry["cy"]) / vehicle_height > 0.10:
            reasons.append("vertical misalignment between bilateral pair exceeds threshold")
        if abs(left_entry["width"] - right_entry["width"]) / max(1.0, left_entry["width"], right_entry["width"]) > 0.70:
            reasons.append("disproportionate candidate widths exceed tolerance")
        if abs(left_entry["height"] - right_entry["height"]) / max(1.0, left_entry["height"], right_entry["height"]) > 0.70:
            reasons.append("disproportionate candidate heights exceed tolerance")
        if min(left_entry["redness"], right_entry["redness"]) < TAILLIGHT_MIN_REDNESS:
            reasons.append("candidate does not meet minimum redness threshold")
        if shape_similarity < 0.04:
            reasons.append(f"bilateral shape similarity below threshold ({shape_similarity:.3f} < 0.040)")

        return reasons

    def select_taillight_pair(self, entries, vehicle_bbox, logo_bbox):
        left_entries = [entry for entry in entries if entry["side"] == "left"]
        right_entries = [entry for entry in entries if entry["side"] == "right"]

        pair_candidates = []
        for left_entry in left_entries:
            for right_entry in right_entries:
                pair_score = self.score_taillight_pair(left_entry, right_entry, vehicle_bbox, logo_bbox)
                if pair_score is None:
                    continue
                pair_candidates.append((pair_score, left_entry, right_entry))

        pair_candidates.sort(key=lambda item: item[0], reverse=True)
        for pair_score, left_entry, right_entry in pair_candidates:
            rejection_reasons = self.get_taillight_pair_rejection_reasons(
                left_entry,
                right_entry,
                vehicle_bbox,
                logo_bbox,
                pair_score,
            )
            if rejection_reasons:
                continue

            return left_entry, right_entry

        return None, None

    def build_taillight_masks(
        self,
        image_normal,
        transformed_image,
        image_size,
        vehicle_bbox,
        logo_bbox,
        source_name=None,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    ):
        manual_override = self.get_manual_component_override(source_name, "taillight")
        manual_masks = {}
        expected_manual_prompts = set()
        if manual_override is not None:
            expected_manual_prompts = set((manual_override.get("points") or {}).keys())
            manual_masks = self.build_manual_taillight_masks_from_points(
                image_normal,
                image_size,
                vehicle_bbox,
                manual_override.get("points"),
            )
            if (
                manual_override.get("mode") == "replace" and
                expected_manual_prompts and
                expected_manual_prompts.issubset(manual_masks.keys())
            ):
                return manual_masks

        entries = self.build_taillight_entries(
            image_normal,
            transformed_image,
            image_size,
            vehicle_bbox,
            logo_bbox,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        left_entry, right_entry = self.select_taillight_pair(
            entries,
            vehicle_bbox,
            logo_bbox,
        )

        if left_entry is None and right_entry is None:
            left_candidates = [entry for entry in entries if entry["side"] == "left"]
            right_candidates = [entry for entry in entries if entry["side"] == "right"]
            left_best = max(left_candidates, key=lambda item: item["score"], default=None)
            right_best = max(right_candidates, key=lambda item: item["score"], default=None)

            if left_best is not None and right_best is None:
                right_entry = self.recover_missing_taillight_from_symmetry(
                    image_normal,
                    image_size,
                    left_best,
                    "right",
                    vehicle_bbox,
                    logo_bbox,
                )
                left_entry = left_best
            elif right_best is not None and left_best is None:
                left_entry = self.recover_missing_taillight_from_symmetry(
                    image_normal,
                    image_size,
                    right_best,
                    "left",
                    vehicle_bbox,
                    logo_bbox,
                )
                right_entry = right_best

            if left_entry is None:
                left_entry = self.recover_taillight_from_anchor_box(
                    image_normal,
                    image_size,
                    "left",
                    vehicle_bbox,
                    logo_bbox,
                )
            if right_entry is None:
                right_entry = self.recover_taillight_from_anchor_box(
                    image_normal,
                    image_size,
                    "right",
                    vehicle_bbox,
                    logo_bbox,
                )

            if left_entry is None or right_entry is None:
                return manual_masks if manual_masks else {}

            recovered_pair_score = self.score_taillight_pair(left_entry, right_entry, vehicle_bbox, logo_bbox)
            if recovered_pair_score is None:
                return manual_masks if manual_masks else {}
            recovered_pair_reasons = self.get_taillight_pair_rejection_reasons(
                left_entry,
                right_entry,
                vehicle_bbox,
                logo_bbox,
                recovered_pair_score,
            )
            if recovered_pair_reasons:
                return manual_masks if manual_masks else {}

        taillight_masks = {
            "left taillight": left_entry["mask"],
            "right taillight": right_entry["mask"],
        }
        taillight_masks.update(manual_masks)
        return taillight_masks

    def refine_taillights_with_mirror_context(
        self,
        taillight_masks,
        mirror_masks,
        image_normal,
        image_size,
        vehicle_bbox,
        logo_bbox,
    ):
        taillight_masks = taillight_masks or {}

        left_tl = taillight_masks.get("left taillight")
        right_tl = taillight_masks.get("right taillight")
        left_mirror = mirror_masks.get("left mirror")
        right_mirror = mirror_masks.get("right mirror")

        left_overlap = max(
            self.get_mask_overlap_ratio(left_tl, left_mirror),
            self.get_mask_overlap_ratio(left_tl, right_mirror),
        )
        right_overlap = max(
            self.get_mask_overlap_ratio(right_tl, left_mirror),
            self.get_mask_overlap_ratio(right_tl, right_mirror),
        )

        if left_overlap > TAILLIGHT_MIRROR_MAX_OVERLAP_RATIO:
            left_tl = None
        if right_overlap > TAILLIGHT_MIRROR_MAX_OVERLAP_RATIO:
            right_tl = None

        left_entry = None
        right_entry = None

        if left_tl is not None and right_tl is None:
            source_entry, _ = self.build_taillight_entry_from_mask(
                image_normal,
                left_tl,
                self.get_mask_bbox(left_tl),
                0.0,
                "existing left taillight",
                vehicle_bbox,
                logo_bbox,
            )
            right_entry = self.recover_missing_taillight_from_symmetry(
                image_normal,
                image_size,
                source_entry,
                "right",
                vehicle_bbox,
                logo_bbox,
            ) if source_entry is not None else None
            right_tl = right_entry["mask"] if right_entry is not None else None
        elif right_tl is not None and left_tl is None:
            source_entry, _ = self.build_taillight_entry_from_mask(
                image_normal,
                right_tl,
                self.get_mask_bbox(right_tl),
                0.0,
                "existing right taillight",
                vehicle_bbox,
                logo_bbox,
            )
            left_entry = self.recover_missing_taillight_from_symmetry(
                image_normal,
                image_size,
                source_entry,
                "left",
                vehicle_bbox,
                logo_bbox,
            ) if source_entry is not None else None
            left_tl = left_entry["mask"] if left_entry is not None else None

        if left_tl is not None:
            left_entry, _ = self.build_taillight_entry_from_mask(
                image_normal,
                left_tl,
                self.get_mask_bbox(left_tl),
                0.0,
                "refined left taillight",
                vehicle_bbox,
                logo_bbox,
                max_width_ratio=TAILLIGHT_FALLBACK_MAX_VEHICLE_WIDTH_RATIO,
            )
        if right_tl is not None:
            right_entry, _ = self.build_taillight_entry_from_mask(
                image_normal,
                right_tl,
                self.get_mask_bbox(right_tl),
                0.0,
                "refined right taillight",
                vehicle_bbox,
                logo_bbox,
                max_width_ratio=TAILLIGHT_FALLBACK_MAX_VEHICLE_WIDTH_RATIO,
            )

        if left_entry is None and left_mirror is not None:
            left_entry = self.recover_taillight_from_mirror_box(
                image_normal,
                image_size,
                "left",
                left_mirror,
                vehicle_bbox,
                logo_bbox,
            )
        if right_entry is None and right_mirror is not None:
            right_entry = self.recover_taillight_from_mirror_box(
                image_normal,
                image_size,
                "right",
                right_mirror,
                vehicle_bbox,
                logo_bbox,
            )

        if left_entry is not None and right_entry is not None:
            pair_score = self.score_taillight_pair(left_entry, right_entry, vehicle_bbox, logo_bbox)
            if pair_score is not None:
                rejection_reasons = self.get_taillight_pair_rejection_reasons(
                    left_entry,
                    right_entry,
                    vehicle_bbox,
                    logo_bbox,
                    pair_score,
                )
                if rejection_reasons:
                    left_entry = self.recover_taillight_from_mirror_box(
                        image_normal,
                        image_size,
                        "left",
                        left_mirror,
                        vehicle_bbox,
                        logo_bbox,
                    ) if left_mirror is not None else None
                    right_entry = self.recover_taillight_from_mirror_box(
                        image_normal,
                        image_size,
                        "right",
                        right_mirror,
                        vehicle_bbox,
                        logo_bbox,
                    ) if right_mirror is not None else None

                    if left_entry is None or right_entry is None:
                        left_entry = None
                        right_entry = None
                    else:
                        pair_score = self.score_taillight_pair(left_entry, right_entry, vehicle_bbox, logo_bbox)
                        if pair_score is None:
                            left_entry = None
                            right_entry = None
                        else:
                            rejection_reasons = self.get_taillight_pair_rejection_reasons(
                                left_entry,
                                right_entry,
                                vehicle_bbox,
                                logo_bbox,
                                pair_score,
                            )
                            if rejection_reasons:
                                left_entry = None
                                right_entry = None

        refined_masks = {}
        if left_entry is not None:
            refined_masks["left taillight"] = left_entry["mask"]
        if right_entry is not None:
            refined_masks["right taillight"] = right_entry["mask"]
        return refined_masks

    def predict_parts(self, image_normal, image_white, source_name=None, box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD):
        self.prepare_cached_image(image_normal)
        try:
            w, h = image_normal.size
            img_normal_trans, _ = self.image_transform(image_normal, None)
            img_white_trans, _ = self.image_transform(image_white, None)
            transformed_images = {
                "normal": img_normal_trans,
                "white": img_white_trans,
            }

            unique_masks = {}
            vehicle_mask = self.build_vehicle_mask(image_normal)
            vehicle_bbox = self.get_mask_bbox(vehicle_mask)
            logo_mask = None
            logo_bbox = None

            windshield_spec = PART_SPECS["rear windshield"]
            windshield_box, windshield_mask = self.get_best_windshield_result(
                image_normal,
                transformed_images[windshield_spec["source"]],
                (w, h),
                vehicle_mask,
                windshield_spec["captions"],
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
            if windshield_box is not None and windshield_mask is not None:
                unique_masks["rear windshield"] = windshield_mask.bool()
            logo_spec = PART_SPECS["logo"]
            logo_mask = self.build_logo_mask(
                image_normal,
                transformed_images[logo_spec["source"]],
                (w, h),
                source_name=source_name,
                vehicle_bbox=vehicle_bbox,
                windshield_box=windshield_box,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
            if logo_mask is not None and torch.any(logo_mask):
                unique_masks["logo"] = logo_mask.bool()
                vehicle_cx = self.get_box_center(vehicle_bbox)[0] if vehicle_bbox is not None else None
                logo_bbox = self.get_central_mask_component_bbox(
                    logo_mask,
                    reference_x=vehicle_cx,
                )
            taillight_spec = PART_SPECS["taillight"]
            taillight_masks = self.build_taillight_masks(
                image_normal,
                transformed_images[taillight_spec["source"]],
                (w, h),
                vehicle_bbox,
                logo_bbox,
                source_name=source_name,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
            unique_masks.update(taillight_masks)

            mirror_spec = PART_SPECS["mirror"]
            mirror_masks = self.build_mirror_masks(
                image_normal,
                transformed_images[mirror_spec["source"]],
                (w, h),
                vehicle_bbox,
                logo_bbox,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
            unique_masks.update(mirror_masks)

            taillight_manual_override = self.get_manual_component_override(source_name, "taillight")
            manual_taillight_prompts = set((taillight_manual_override or {}).get("points", {}).keys())
            if (
                taillight_manual_override is not None and
                taillight_manual_override.get("mode") == "replace" and
                manual_taillight_prompts and
                manual_taillight_prompts.issubset(taillight_masks.keys())
            ):
                refined_taillight_masks = taillight_masks
            else:
                refined_taillight_masks = self.refine_taillights_with_mirror_context(
                    taillight_masks,
                    mirror_masks,
                    image_normal,
                    (w, h),
                    vehicle_bbox,
                    logo_bbox,
                )
                unique_masks.pop("left taillight", None)
                unique_masks.pop("right taillight", None)
                unique_masks.update(refined_taillight_masks)

            final_taillight_masks = refined_taillight_masks or taillight_masks

            wheel_spec = PART_SPECS["wheel"]
            wheel_masks = self.build_wheel_masks(
                image_normal,
                image_white,
                transformed_images[wheel_spec["source"]],
                (w, h),
                vehicle_mask,
                vehicle_bbox,
                logo_bbox,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
            unique_masks.update(wheel_masks)

            exhaust_spec = PART_SPECS["exhaust tips"]
            exhaust_mask = self.build_exhaust_mask(
                image_normal,
                transformed_images[exhaust_spec["source"]],
                (w, h),
                vehicle_mask,
                vehicle_bbox,
                logo_mask=logo_mask,
                logo_bbox=logo_bbox,
                taillight_masks=final_taillight_masks,
                mirror_masks=mirror_masks,
                wheel_masks=wheel_masks,
                source_name=source_name,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
            if exhaust_mask is not None and torch.any(exhaust_mask):
                unique_masks["exhaust tips"] = exhaust_mask.bool()
            bumper_spec = PART_SPECS["rear bumper"]
            bumper_mask = self.build_bumper_mask(
                image_normal,
                transformed_images[bumper_spec["source"]],
                (w, h),
                vehicle_bbox=vehicle_bbox,
                logo_bbox=logo_bbox,
                taillight_masks=final_taillight_masks,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
            bumper_masks_to_remove = list(wheel_masks.values())
            if exhaust_mask is not None:
                bumper_masks_to_remove.append(exhaust_mask)
            bumper_mask = self.subtract_masks(bumper_mask, bumper_masks_to_remove)
            if bumper_mask is not None and torch.any(bumper_mask):
                unique_masks["rear bumper"] = bumper_mask.bool()

            if exhaust_mask is not None and "rear bumper" in unique_masks:
                unique_masks["rear bumper"] = self.subtract_masks(
                    unique_masks["rear bumper"],
                    [exhaust_mask],
                )
                if unique_masks["rear bumper"] is None:
                    unique_masks.pop("rear bumper", None)

            return unique_masks
        finally:
            self.clear_cached_image()

def resolve_path(p, script_dir):
    if os.path.exists(p):
        return p
    for cand in [
        os.path.join(script_dir, p),
        os.path.join(script_dir, "checkpoints", p),
        os.path.join(script_dir, "checkpoints", os.path.basename(p)),
    ]:
        if os.path.exists(cand):
            return cand
    return p

def main():
    print("Initializing Rear View Segmentation Pipeline (Windshield, Logo, Taillights, Mirrors, Wheels, Exhaust, Bumper)...")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    sam2_ckpt = resolve_path(SAM2_CHECKPOINT, script_dir)
    gdino_ckpt = resolve_path(GDINO_CHECKPOINT, script_dir)

    import groundingdino
    gdino_config_path = os.path.join(os.path.dirname(groundingdino.__file__), "config", GDINO_CONFIG)

    model = LangSAM2Basic(
        sam2_cfg=SAM2_CONFIG,
        sam2_ckpt=sam2_ckpt,
        gdino_cfg=gdino_config_path,
        gdino_ckpt=gdino_ckpt,
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, DATASET_FOLDER)
    if not os.path.exists(dataset_dir) and os.path.exists(os.path.join(script_dir, "sample_data", DATASET_FOLDER)):
        dataset_dir = os.path.join(script_dir, "sample_data", DATASET_FOLDER)
    out_base_dir = os.path.join(script_dir, OUTPUT_MASKS_FOLDER)

    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img_name = os.path.splitext(file)[0]
            if img_name.lower() != "rear":
                continue

            img_path = os.path.abspath(os.path.join(root, file))
            rel_path = os.path.relpath(img_path, dataset_dir)

            print(f"\nProcessing: {file}")

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
                image_np = np.array(image_normal)
                foreground_mask = (np.any(image_np > VEHICLE_MASK_THRESHOLD, axis=-1).astype(np.uint8) * 255)
                white_bg = Image.new("RGB", image_normal.size, (255, 255, 255))
                white_bg.paste(image_normal, mask=Image.fromarray(foreground_mask, mode="L"))
                image_white = white_bg

            out_img_dir = os.path.join(out_base_dir, os.path.dirname(rel_path), img_name)
            os.makedirs(out_img_dir, exist_ok=True)

            masks = model.predict_parts(
                image_normal,
                image_white,
                source_name=rel_path,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
            )

            if masks:
                print("Generating aggregated visualization overlay...")
                summary_img = generate_summary_image_with_colored_legend(image_normal, masks)
                summary_path = os.path.join(out_img_dir, f"{img_name}_aggregated.png")
                cv2.imwrite(summary_path, summary_img)
                print(f"  -> Saved aggregated preview: {os.path.basename(summary_path)}")

                for prompt, mask in masks.items():
                    mask_img = (mask.cpu().numpy() * 255).astype(np.uint8)
                    safe_prompt = prompt.replace(" ", "_")
                    cv2.imwrite(os.path.join(out_img_dir, f"{safe_prompt}.png"), mask_img)
                    print(f"  -> Detected and saved mask: {prompt}")
            else:
                print("  -> WARNING: No components detected for this image.")

            del image_raw
            del image_normal
            del image_white
            gc.collect()

if __name__ == "__main__":
    if torch.cuda.is_available():
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    main()
