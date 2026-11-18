"""
frontSegmentator.py
===================
Automated semantic part segmentation pipeline for frontal-view vehicle images.

This module combines Grounding DINO (open-vocabulary object detection) and
Segment Anything 2 (SAM 2, high-precision promptable segmentation) with
domain-specific geometric and bilateral symmetry constraints tailored to
front-view automotive anatomy.

Key Methodological Highlights:
- Dual-Pass Inference: Distinguishes standard vehicle components (windshield, hood,
  grille, headlights, fog lights, emblem) processed on the original image from
  tires processed on a synthetic white canvas to overcome low-contrast wheel well shadows.
- Geometric Prior Filtering: Enforces relative spatial relations (windshield above hood,
  mirrors bounded vertically between windshield top and hood bottom, headlights below hood
  midpoint, grille bounded laterally between headlights/fogs).
- Bilateral Symmetry Recovery: If only one symmetric element (e.g., one headlight or one
  mirror) is detected, the missing counterpart is mirrored across the vehicle's central axis.
- Multi-Stage Fallback Strategies: Includes generic candidate fallbacks for difficult headlights,
  tires, and fog lights, as well as strict IoU/IoA overlap rejection against headlights.
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

# Class registry mapping prompt names to [Display Name, Color (BGR)]
CLASS_MAP = {
    "car brand badge emblem": ["Emblem", (255, 0, 0)],                   # Blue
    "car front windshield glass": ["Windshield", (0, 255, 0)],           # Green
    "car hood": ["Hood", (0, 0, 255)],                                   # Red
    "car left side mirror": ["Left Mirror", (0, 255, 255)],              # Yellow
    "car right side mirror": ["Right Mirror", (255, 0, 255)],            # Magenta
    "car left headlight": ["Left Headlight", (255, 255, 0)],             # Cyan
    "car right headlight": ["Right Headlight", (0, 165, 255)],           # Orange
    "car left front tire": ["Left Tire", (128, 0, 128)],                 # Purple
    "car right front tire": ["Right Tire", (0, 128, 128)],               # Teal
    "car left lower bumper light": ["Left Fog Light", (100, 255, 100)],  # Light Green
    "car right lower bumper light": ["Right Fog Light", (200, 100, 255)],# Lavender
    "car front grille": ["Front Grille", (20, 69, 139)],                 # Brown
}

PROMPTS = list(CLASS_MAP.keys())

# Prompt separation for dual-pass inference (standard vs. tire isolation)
PROMPTS_TIRES = ["car left front tire", "car right front tire"]
PROMPTS_NORMAL = [p for p in PROMPTS if p not in PROMPTS_TIRES]
PROMPTS_TIRES_FALLBACK = ["car front tire", "car front wheel", "car tire", "car wheel"]


# =============================================================================
# --- 2. VISUALIZATION HELPERS ---
# =============================================================================
def generate_summary_image_with_colored_legend(image_pil: Image.Image, unique_masks: dict) -> np.ndarray:
    """
    Generate an aggregated color overlay image with all segmented components
    superimposed on a faded background, accompanied by a lateral color legend.

    Args:
        image_pil: Original input PIL Image.
        unique_masks: Dictionary mapping prompt labels to binary boolean/tensor masks.

    Returns:
        np.ndarray: BGR composite image containing the blended overlay and side legend.
    """
    img_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    h, w, _ = img_bgr.shape

    background_faded = cv2.addWeighted(img_bgr, 0.4, np.full((h, w, 3), 220, dtype=np.uint8), 0.6, 0)

    mask_composite = np.zeros((h, w, 3), dtype=np.uint8)
    masks_found_count = 0

    for prompt, mask in reversed(list(unique_masks.items())):
        if prompt not in CLASS_MAP:
            continue
        mask_np = mask.cpu().numpy().astype(np.uint8)
        color = CLASS_MAP[prompt][1]
        mask_composite[mask_np > 0] = color
        masks_found_count += 1

    final_blended = background_faded.copy()
    if masks_found_count > 0:
        mask_indices = np.any(mask_composite > 0, axis=-1)
        final_blended[mask_indices] = mask_composite[mask_indices]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    line_spacing = 40
    entry_height = 30

    max_text_width = 0
    items_to_draw = []

    for prompt, mask in unique_masks.items():
        if prompt in CLASS_MAP:
            legend_name = CLASS_MAP[prompt][0]
            color = CLASS_MAP[prompt][1]
            (tw, th), _ = cv2.getTextSize(legend_name, font, font_scale, thickness)
            if tw > max_text_width:
                max_text_width = tw
            items_to_draw.append((legend_name, color))

    if not items_to_draw:
        return background_faded

    legend_width = max_text_width + 60
    legend_height = h

    legend_canvas = np.full((legend_height, legend_width, 3), 220, dtype=np.uint8)

    current_y = 20 + entry_height
    for name, color in items_to_draw:
        cv2.rectangle(legend_canvas, (10, current_y - 20), (10 + 20, current_y), color, -1)
        cv2.putText(legend_canvas, name, (40, current_y), font, font_scale, color, thickness, cv2.LINE_AA)
        current_y += line_spacing

    combined_img = np.hstack((final_blended, legend_canvas))
    return combined_img


# =============================================================================
# --- 3. LangSAM2 WRAPPER ---
# =============================================================================
class LangSAM2Basic:
    """
    Unified Grounding DINO + SAM 2 pipeline wrapper with automotive
    front-view geometric parsing and bilateral symmetry filtering.
    """

    def __init__(self, sam2_cfg: str, sam2_ckpt: str, gdino_cfg: str, gdino_ckpt: str):
        """
        Initialize Grounding DINO detector and Segment Anything 2 predictor.

        Args:
            sam2_cfg: Path or model configuration name for SAM 2.
            sam2_ckpt: Path to pretrained SAM 2 checkpoint (.pt).
            gdino_cfg: Path to Grounding DINO model configuration Python file.
            gdino_ckpt: Path to Grounding DINO checkpoint (.pth).
        """
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
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

    def get_center(self, box):
        """Calculate (center_x, center_y) for a bounding box [x1, y1, x2, y2]."""
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

    def calculate_iou(self, boxA, boxB):
        """Calculate Intersection-over-Union (IoU) between two bounding boxes."""
        x_left = max(boxA[0], boxB[0])
        y_top = max(boxA[1], boxB[1])
        x_right = min(boxA[2], boxB[2])
        y_bottom = min(boxA[3], boxB[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = intersection_area / float(boxA_area + boxB_area - intersection_area)
        return iou

    def scale_boxes_to_image(self, boxes, w, h):
        """Convert normalized cx, cy, w, h boxes to pixel coordinates [x1, y1, x2, y2]."""
        if len(boxes) == 0:
            return np.empty((0, 4), dtype=np.float32)

        boxes = boxes * torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes.cpu().numpy()

    def format_generic_candidates(self, boxes, logits, w, h):
        """Format raw Grounding DINO outputs into scaled pixel boxes with confidence logits."""
        boxes_np = self.scale_boxes_to_image(boxes, w, h)
        if len(boxes_np) == 0:
            return []

        logits_np = logits.cpu().numpy()
        return list(zip(boxes_np, logits_np))

    def select_tires_by_geometry(self, tire_candidates, left_range, right_range, min_cy=0, min_box_size=(0, 0)):
        """
        Filter and select the highest-scoring left and right front tire candidates
        within specified spatial lateral windows and vertical bounds.
        """
        valid_left_tires = []
        valid_right_tires = []
        min_box_w, min_box_h = min_box_size

        for box, logit in tire_candidates:
            cx, cy = self.get_center(box)
            box_w = box[2] - box[0]
            box_h = box[3] - box[1]

            if cy < min_cy:
                continue
            if box_w < min_box_w or box_h < min_box_h:
                continue

            if left_range[0] < cx < left_range[1]:
                valid_left_tires.append((box, logit))
            elif right_range[0] < cx < right_range[1]:
                valid_right_tires.append((box, logit))

        best_left_tire = max(valid_left_tires, key=lambda c: c[1])[0] if valid_left_tires else None
        best_right_tire = max(valid_right_tires, key=lambda c: c[1])[0] if valid_right_tires else None
        return best_left_tire, best_right_tire

    def predict_hybrid(self, image_normal: Image.Image, image_white: Image.Image,
                       box_threshold: float = 0.25, text_threshold: float = 0.20):
        """
        Execute dual-pass detection, geometric parsing, symmetry validation,
        and high-precision SAM 2 segmentation for front-view vehicle images.

        Args:
            image_normal: Input image composited over dark canvas.
            image_white: Input image composited over white canvas for tire isolation.
            box_threshold: Detection confidence threshold for bounding boxes.
            text_threshold: Text alignment threshold for textual prompts.

        Returns:
            dict: Mapping of detected class prompt to 2D binary PyTorch boolean mask tensor.
        """
        w, h = image_normal.size

        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        img_norm_trans, _ = transform(image_normal, None)
        img_white_trans, _ = transform(image_white, None)

        # --- PASS 1: Detection of standard vehicle components on dark background ---
        boxes_n, logits_n, phrases_n = predict_gdino(
            model=self.grounding_dino,
            image=img_norm_trans,
            caption=" . ".join(PROMPTS_NORMAL),
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device
        )

        # --- PASS 2: Detection of tires on white background (high-contrast boundary isolation) ---
        boxes_w, logits_w, phrases_w = predict_gdino(
            model=self.grounding_dino,
            image=img_white_trans,
            caption=" . ".join(PROMPTS_TIRES),
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device
        )

        # Helper to map detected phrases to target prompt categories
        def process_dino_results(boxes, logits, phrases, target_prompts):
            candidates_dict = {p: [] for p in target_prompts}
            if len(boxes) > 0:
                boxes_np = self.scale_boxes_to_image(boxes, w, h)
                logits_np = logits.cpu().numpy()

                for box, logit, phrase in zip(boxes_np, logits_np, phrases):
                    for p in target_prompts:
                        if phrase.lower() in p.lower() or p.lower() in phrase.lower():
                            candidates_dict[p].append((box, logit))
                            break
            return candidates_dict

        # Combine results from both detection passes
        candidates_normal = process_dino_results(boxes_n, logits_n, phrases_n, PROMPTS_NORMAL)
        candidates_tires = process_dino_results(boxes_w, logits_w, phrases_w, PROMPTS_TIRES)
        candidates = {**candidates_normal, **candidates_tires}

        final_boxes = {}

        # =====================================================================
        # 1. EMBLEM / BRAND LOGO DETECTION & FILTERING
        # =====================================================================
        if candidates["car brand badge emblem"]:
            valid_logos = []
            img_cx = w / 2
            for box, logit in candidates["car brand badge emblem"]:
                box_w = box[2] - box[0]
                box_h = box[3] - box[1]
                box_cx, _ = self.get_center(box)
                is_small_enough = box_w < w * 0.20 and box_h < h * 0.20
                is_centered = abs(box_cx - img_cx) < (w * 0.15)
                if is_small_enough and is_centered:
                    valid_logos.append((box, logit))

            valid_logos.sort(key=lambda c: c[1], reverse=True)
            selected_logos = []
            for box, logit in valid_logos:
                is_overlapping = False
                for sel_box in selected_logos:
                    if self.calculate_iou(box, sel_box) > 0.3:
                        is_overlapping = True
                        break
                if not is_overlapping:
                    selected_logos.append(box)
                    if len(selected_logos) == 2:
                        break

            if selected_logos:
                final_boxes["car brand badge emblem"] = selected_logos

        # =====================================================================
        # 2. FRONT WINDSHIELD DETECTION
        # =====================================================================
        if candidates["car front windshield glass"]:
            best_windshield = max(candidates["car front windshield glass"], key=lambda c: c[1])[0]
            final_boxes["car front windshield glass"] = best_windshield

        # =====================================================================
        # 3. HOOD DETECTION
        # =====================================================================
        if candidates["car hood"]:
            valid_hoods = []
            if "car front windshield glass" in final_boxes:
                w_box = final_boxes["car front windshield glass"]
                _, w_cy = self.get_center(w_box)
                for box, logit in candidates["car hood"]:
                    _, h_cy = self.get_center(box)
                    if h_cy > w_cy:
                        valid_hoods.append((box, logit))
            else:
                valid_hoods = candidates["car hood"]

            if valid_hoods:
                best_hood = max(valid_hoods, key=lambda c: c[1])[0]
                final_boxes["car hood"] = best_hood

        # =====================================================================
        # GEOMETRIC REFERENCE ANCHORS CALCULATION
        # =====================================================================
        logo_cx = w / 2
        if "car brand badge emblem" in final_boxes:
            primary_logo_box = final_boxes["car brand badge emblem"][0]
            logo_cx, _ = self.get_center(primary_logo_box)

        w_top = 0
        if "car front windshield glass" in final_boxes:
            w_top = final_boxes["car front windshield glass"][1]

        h_bottom = h
        if "car hood" in final_boxes:
            h_bottom = final_boxes["car hood"][3]

        # =====================================================================
        # 4. SIDE MIRRORS DETECTION & BILATERAL SYMMETRY MIRRORING
        # =====================================================================
        valid_left_mirrors = []
        valid_right_mirrors = []
        for prompt in ["car left side mirror", "car right side mirror"]:
            for box, logit in candidates[prompt]:
                cx, cy = self.get_center(box)
                if w_top < cy < h_bottom:
                    if cx < logo_cx:
                        valid_left_mirrors.append((box, logit))
                    elif cx > logo_cx:
                        valid_right_mirrors.append((box, logit))

        best_left_mirror = max(valid_left_mirrors, key=lambda c: c[1])[0] if valid_left_mirrors else None
        best_right_mirror = max(valid_right_mirrors, key=lambda c: c[1])[0] if valid_right_mirrors else None

        # Bilateral emergency symmetry mirroring if only one side is detected
        if best_right_mirror is not None and best_left_mirror is None:
            x1, y1, x2, y2 = best_right_mirror
            best_left_mirror = np.array([2 * logo_cx - x2, y1, 2 * logo_cx - x1, y2])
        elif best_left_mirror is not None and best_right_mirror is None:
            x1, y1, x2, y2 = best_left_mirror
            best_right_mirror = np.array([2 * logo_cx - x2, y1, 2 * logo_cx - x1, y2])

        if best_left_mirror is not None:
            final_boxes["car left side mirror"] = best_left_mirror
        if best_right_mirror is not None:
            final_boxes["car right side mirror"] = best_right_mirror

        # =====================================================================
        # 5. HEADLIGHTS DETECTION WITH FALLBACK STRATEGY
        # =====================================================================
        valid_left_headlights = []
        valid_right_headlights = []

        # --- Primary Detection Strategy ---
        for prompt in ["car left headlight", "car right headlight"]:
            for box, logit in candidates[prompt]:
                cx, cy = self.get_center(box)
                if cx < logo_cx:
                    valid_left_headlights.append((box, logit))
                elif cx > logo_cx:
                    valid_right_headlights.append((box, logit))

        best_left_hl = max(valid_left_headlights, key=lambda c: c[1])[0] if valid_left_headlights else None
        best_right_hl = max(valid_right_headlights, key=lambda c: c[1])[0] if valid_right_headlights else None

        # --- Geometric Fallback Strategy ---
        # Activated only when no headlights are detected by the primary prompt
        if best_left_hl is None and best_right_hl is None:
            fallback_prompts = ["car headlight", "front light", "headlamp", "led light"]
            boxes_hl_fb, logits_hl_fb, _ = predict_gdino(
                model=self.grounding_dino,
                image=img_norm_trans,
                caption=" . ".join(fallback_prompts),
                box_threshold=0.10,
                text_threshold=0.10,
                device=self.device
            )

            generic_hl_cands = self.format_generic_candidates(boxes_hl_fb, logits_hl_fb, w, h)

            left_cands = []
            right_cands = []

            hood_bottom = final_boxes["car hood"][3] if "car hood" in final_boxes else h * 0.5
            hood_cy = self.get_center(final_boxes["car hood"])[1] if "car hood" in final_boxes else h * 0.3

            for box, logit in generic_hl_cands:
                cx, cy = self.get_center(box)
                # Geometric constraint: cy > hood_cy ensures the headlight is below the hood center,
                # immediately eliminating side mirror false positives.
                if cy > hood_cy and box[1] < hood_bottom + (h * 0.1):
                    if cx < logo_cx - (w * 0.05):
                        left_cands.append(box)
                    elif cx > logo_cx + (w * 0.05):
                        right_cands.append(box)

            best_pair = None
            min_score = float('inf')

            for l_box in left_cands:
                l_cx, l_cy = self.get_center(l_box)
                for r_box in right_cands:
                    r_cx, r_cy = self.get_center(r_box)

                    y_diff = abs(l_cy - r_cy)
                    sym_diff = abs((logo_cx - l_cx) - (r_cx - logo_cx))

                    if y_diff < (h * 0.05) and sym_diff < (w * 0.1):
                        score = y_diff + sym_diff
                        if score < min_score:
                            min_score = score
                            best_pair = (l_box, r_box)

            if best_pair:
                best_left_hl, best_right_hl = best_pair
            else:
                best_left_hl = left_cands[0] if left_cands else None
                best_right_hl = right_cands[0] if right_cands else None

        # Bilateral emergency symmetry mirroring if only one headlight is found
        if best_right_hl is not None and best_left_hl is None:
            x1, y1, x2, y2 = best_right_hl
            best_left_hl = np.array([2 * logo_cx - x2, y1, 2 * logo_cx - x1, y2])
        elif best_left_hl is not None and best_right_hl is None:
            x1, y1, x2, y2 = best_left_hl
            best_right_hl = np.array([2 * logo_cx - x2, y1, 2 * logo_cx - x1, y2])

        if best_left_hl is not None:
            final_boxes["car left headlight"] = best_left_hl
        if best_right_hl is not None:
            final_boxes["car right headlight"] = best_right_hl

        # =====================================================================
        # 6. FRONT TIRES DETECTION (WITH GEOMETRIC FALLBACKS)
        # =====================================================================
        valid_left_tires = []
        valid_right_tires = []

        left_hl_cy = self.get_center(best_left_hl)[1] if best_left_hl is not None else 0
        right_hl_cy = self.get_center(best_right_hl)[1] if best_right_hl is not None else 0

        l_col_min, l_col_max = (0, w * 0.35)
        r_col_min, r_col_max = (w * 0.65, w)
        margin_x = w * 0.05

        if best_left_hl is not None and "car left side mirror" in final_boxes:
            l_mirror_box = final_boxes["car left side mirror"]
            l_col_min = l_mirror_box[0] - margin_x
            l_col_max = best_left_hl[2] + margin_x

        if best_right_hl is not None and "car right side mirror" in final_boxes:
            r_mirror_box = final_boxes["car right side mirror"]
            r_col_min = best_right_hl[0] - margin_x
            r_col_max = r_mirror_box[2] + margin_x

        for prompt in ["car left front tire", "car right front tire"]:
            for box, logit in candidates[prompt]:
                cx, cy = self.get_center(box)
                if cy < left_hl_cy and cy < right_hl_cy:
                    continue
                if l_col_min < cx < l_col_max:
                    valid_left_tires.append((box, logit))
                elif r_col_min < cx < r_col_max:
                    valid_right_tires.append((box, logit))

        best_left_tire = max(valid_left_tires, key=lambda c: c[1])[0] if valid_left_tires else None
        best_right_tire = max(valid_right_tires, key=lambda c: c[1])[0] if valid_right_tires else None

        if best_left_tire is None or best_right_tire is None:
            strict_tire_candidates = candidates["car left front tire"] + candidates["car right front tire"]
            fallback_min_cy = max(left_hl_cy, right_hl_cy, h * 0.45)
            if "car hood" in final_boxes:
                fallback_min_cy = max(fallback_min_cy, final_boxes["car hood"][3] - h * 0.03)

            relaxed_gap = w * 0.06
            relaxed_left_range = (0, max(0, logo_cx - relaxed_gap))
            relaxed_right_range = (min(w, logo_cx + relaxed_gap), w)
            min_tire_size = (w * 0.05, h * 0.08)

            fallback_left_tire, fallback_right_tire = self.select_tires_by_geometry(
                strict_tire_candidates,
                relaxed_left_range,
                relaxed_right_range,
                min_cy=fallback_min_cy,
                min_box_size=min_tire_size
            )

            if best_left_tire is None and fallback_left_tire is not None:
                best_left_tire = fallback_left_tire
            if best_right_tire is None and fallback_right_tire is not None:
                best_right_tire = fallback_right_tire

        if best_left_tire is None or best_right_tire is None:
            boxes_w_fb, logits_w_fb, _ = predict_gdino(
                model=self.grounding_dino,
                image=img_white_trans,
                caption=" . ".join(PROMPTS_TIRES_FALLBACK),
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=self.device
            )

            generic_tire_candidates = self.format_generic_candidates(boxes_w_fb, logits_w_fb, w, h)
            if generic_tire_candidates:
                fallback_min_cy = max(left_hl_cy, right_hl_cy, h * 0.45)
                if "car hood" in final_boxes:
                    fallback_min_cy = max(fallback_min_cy, final_boxes["car hood"][3] - h * 0.03)

                relaxed_gap = w * 0.06
                relaxed_left_range = (0, max(0, logo_cx - relaxed_gap))
                relaxed_right_range = (min(w, logo_cx + relaxed_gap), w)
                min_tire_size = (w * 0.05, h * 0.08)

                fallback_left_tire, fallback_right_tire = self.select_tires_by_geometry(
                    generic_tire_candidates,
                    relaxed_left_range,
                    relaxed_right_range,
                    min_cy=fallback_min_cy,
                    min_box_size=min_tire_size
                )

                if best_left_tire is None and fallback_left_tire is not None:
                    best_left_tire = fallback_left_tire
                if best_right_tire is None and fallback_right_tire is not None:
                    best_right_tire = fallback_right_tire

        # Bilateral emergency symmetry mirroring for tires
        if best_right_tire is not None and best_left_tire is None:
            x1, y1, x2, y2 = best_right_tire
            best_left_tire = np.array([2 * logo_cx - x2, y1, 2 * logo_cx - x1, y2])
        elif best_left_tire is not None and best_right_tire is None:
            x1, y1, x2, y2 = best_left_tire
            best_right_tire = np.array([2 * logo_cx - x2, y1, 2 * logo_cx - x1, y2])

        if best_left_tire is not None:
            final_boxes["car left front tire"] = best_left_tire
        if best_right_tire is not None:
            final_boxes["car right front tire"] = best_right_tire

        # =====================================================================
        # 7. LOWER BUMPER LIGHTS / FOG LIGHTS DETECTION & VALIDATION
        # =====================================================================
        valid_left_fogs = []
        valid_right_fogs = []

        # Reference anchors (Tires)
        left_tire_cy = self.get_center(best_left_tire)[1] if best_left_tire is not None else h
        right_tire_cy = self.get_center(best_right_tire)[1] if best_right_tire is not None else h

        # Reference anchors (Main Headlights)
        left_hl_cy = self.get_center(best_left_hl)[1] if best_left_hl is not None else 0
        right_hl_cy = self.get_center(best_right_hl)[1] if best_right_hl is not None else 0

        # --- Primary Detection Strategy ---
        for prompt in ["car left lower bumper light", "car right lower bumper light"]:
            for box, logit in candidates[prompt]:
                cx, cy = self.get_center(box)
                if cx < logo_cx:
                    if (best_left_hl is None or cy > left_hl_cy) and (cy < left_tire_cy):
                        valid_left_fogs.append((box, logit))
                elif cx > logo_cx:
                    if (best_right_hl is None or cy > right_hl_cy) and (cy < right_tire_cy):
                        valid_right_fogs.append((box, logit))

        best_left_fog = max(valid_left_fogs, key=lambda c: c[1])[0] if valid_left_fogs else None
        best_right_fog = max(valid_right_fogs, key=lambda c: c[1])[0] if valid_right_fogs else None

        # --- Geometric & Dimensional Fallback Strategy ---
        if best_left_fog is None and best_right_fog is None:
            fallback_fog_prompts = ["fog light", "lower light", "lower bumper light", "fog lamp", "led light"]
            boxes_fog_fb, logits_fog_fb, _ = predict_gdino(
                model=self.grounding_dino,
                image=img_norm_trans,
                caption=" . ".join(fallback_fog_prompts),
                box_threshold=0.10,
                text_threshold=0.10,
                device=self.device
            )

            generic_fog_cands = self.format_generic_candidates(boxes_fog_fb, logits_fog_fb, w, h)

            left_f_cands = []
            right_f_cands = []

            for box, logit in generic_fog_cands:
                cx, cy = self.get_center(box)

                is_below_hl_left = cy > left_hl_cy + (h * 0.02)
                is_below_hl_right = cy > right_hl_cy + (h * 0.02)

                if cx < logo_cx - (w * 0.05):
                    if is_below_hl_left and cy < left_tire_cy:
                        left_f_cands.append(box)
                elif cx > logo_cx + (w * 0.05):
                    if is_below_hl_right and cy < right_tire_cy:
                        right_f_cands.append(box)

            best_pair = None
            min_score = float('inf')

            # Search for symmetric bilateral pair
            for l_box in left_f_cands:
                l_cx, l_cy = self.get_center(l_box)
                for r_box in right_f_cands:
                    r_cx, r_cy = self.get_center(r_box)

                    y_diff = abs(l_cy - r_cy)
                    sym_diff = abs((logo_cx - l_cx) - (r_cx - logo_cx))

                    if y_diff < (h * 0.05) and sym_diff < (w * 0.1):
                        score = y_diff + sym_diff
                        if score < min_score:
                            min_score = score
                            best_pair = (l_box, r_box)

            # Dimensional & Proportional Validation
            if best_pair:
                l_box, r_box = best_pair

                l_area = (l_box[2] - l_box[0]) * (l_box[3] - l_box[1])
                r_area = (r_box[2] - r_box[0]) * (r_box[3] - r_box[1])

                # 1. Pair ratio check: larger area should not exceed 3x smaller area
                area_ratio_fogs = max(l_area, r_area) / min(l_area, r_area) if min(l_area, r_area) > 0 else float('inf')
                is_similar_pair = area_ratio_fogs < 3.0

                # 2. Main headlight proportionality check
                is_proportional_to_hl = True
                if best_left_hl is not None:
                    hl_area = (best_left_hl[2] - best_left_hl[0]) * (best_left_hl[3] - best_left_hl[1])
                    if hl_area > 0:
                        ratio_l = l_area / hl_area
                        ratio_r = r_area / hl_area
                        # Fog light area should be between 15% and 250% of headlight area
                        if not (0.15 < ratio_l < 2.5) or not (0.15 < ratio_r < 2.5):
                            is_proportional_to_hl = False

                if is_similar_pair and is_proportional_to_hl:
                    best_left_fog, best_right_fog = best_pair
                else:
                    best_left_fog = None
                    best_right_fog = None

        # Bilateral emergency symmetry mirroring for fog lights
        if best_right_fog is not None and best_left_fog is None:
            x1, y1, x2, y2 = best_right_fog
            best_left_fog = np.array([2 * logo_cx - x2, y1, 2 * logo_cx - x1, y2])
        elif best_left_fog is not None and best_right_fog is None:
            x1, y1, x2, y2 = best_left_fog
            best_right_fog = np.array([2 * logo_cx - x2, y1, 2 * logo_cx - x1, y2])

        # Overlap Rejection: Exclude fog lights intersecting with main headlights
        def is_highly_overlapping(box_fog, box_hl, max_iou=0.15, max_ioa=0.40):
            if box_hl is None or box_fog is None:
                return False

            # 1. Standard IoU check
            if self.calculate_iou(box_fog, box_hl) > max_iou:
                return True

            # 2. IoA check (Intersection over Area of the fog light)
            x_left = max(box_fog[0], box_hl[0])
            y_top = max(box_fog[1], box_hl[1])
            x_right = min(box_fog[2], box_hl[2])
            y_bottom = min(box_fog[3], box_hl[3])

            if x_right > x_left and y_bottom > y_top:
                intersection_area = (x_right - x_left) * (y_bottom - y_top)
                fog_area = (box_fog[2] - box_fog[0]) * (box_fog[3] - box_fog[1])
                if fog_area > 0 and (intersection_area / float(fog_area)) > max_ioa:
                    return True

            return False

        if is_highly_overlapping(best_left_fog, best_left_hl):
            best_left_fog = None

        if is_highly_overlapping(best_right_fog, best_right_hl):
            best_right_fog = None

        if best_left_fog is not None:
            final_boxes["car left lower bumper light"] = best_left_fog
        if best_right_fog is not None:
            final_boxes["car right lower bumper light"] = best_right_fog

        # =====================================================================
        # 8. FRONT GRILLE DETECTION (UP TO 2 GRILLES)
        # =====================================================================
        valid_grilles = []
        hood_cy = self.get_center(final_boxes["car hood"])[1] if "car hood" in final_boxes else 0

        left_limit_x = 0
        if best_left_hl is not None:
            left_limit_x = self.get_center(best_left_hl)[0]
        elif best_left_fog is not None:
            left_limit_x = self.get_center(best_left_fog)[0]

        right_limit_x = w
        if best_right_hl is not None:
            right_limit_x = self.get_center(best_right_hl)[0]
        elif best_right_fog is not None:
            right_limit_x = self.get_center(best_right_fog)[0]

        if candidates["car front grille"]:
            for box, logit in candidates["car front grille"]:
                cx, cy = self.get_center(box)
                if left_limit_x < cx < right_limit_x and cy > hood_cy:
                    valid_grilles.append((box, logit))

            if valid_grilles:
                valid_grilles.sort(key=lambda c: c[1], reverse=True)
                selected_grilles = []
                for box, logit in valid_grilles:
                    is_overlapping = False
                    for sel_box in selected_grilles:
                        if self.calculate_iou(box, sel_box) > 0.3:
                            is_overlapping = True
                            break
                    if not is_overlapping:
                        selected_grilles.append(box)
                        if len(selected_grilles) == 2:
                            break

                if selected_grilles:
                    final_boxes["car front grille"] = selected_grilles

        # =====================================================================
        # SAM 2 MASK INFERENCE (Dual-Pass: Standard Image and White Image)
        # =====================================================================
        unique_masks = {}

        boxes_normal = {k: v for k, v in final_boxes.items() if k not in PROMPTS_TIRES}
        boxes_tires = {k: v for k, v in final_boxes.items() if k in PROMPTS_TIRES}

        # 1. Segment non-tire components using the original image
        if boxes_normal:
            self.sam2_predictor.set_image(np.array(image_normal))
            for prompt, box_data in boxes_normal.items():
                boxes_np_array = np.array(box_data) if isinstance(box_data, list) else np.array([box_data])
                masks, _, _ = self.sam2_predictor.predict(
                    point_coords=None, point_labels=None, box=boxes_np_array, multimask_output=False
                )
                unique_masks[prompt] = torch.any(torch.tensor(masks).squeeze(1), dim=0)

        # 2. Segment tire components using the white-background image
        if boxes_tires:
            self.sam2_predictor.set_image(np.array(image_white))
            for prompt, box_data in boxes_tires.items():
                boxes_np_array = np.array(box_data) if isinstance(box_data, list) else np.array([box_data])
                masks, _, _ = self.sam2_predictor.predict(
                    point_coords=None, point_labels=None, box=boxes_np_array, multimask_output=False
                )
                unique_masks[prompt] = torch.any(torch.tensor(masks).squeeze(1), dim=0)

        return unique_masks


# =============================================================================
# --- 4. BATCH INFERENCE ENTRYPOINT ---
# =============================================================================
def resolve_path(p: str, script_dir: str) -> str:
    """Resolve file path across local directory and checkpoints directory."""
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
    """Main execution function for front-view car part segmentation."""
    print("Initializing Front View Segmentation Pipeline (Dual-Pass Background)...")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    sam2_ckpt = resolve_path(SAM2_CHECKPOINT, script_dir)
    gdino_ckpt = resolve_path(GDINO_CHECKPOINT, script_dir)

    import groundingdino
    gdino_config_path = os.path.join(os.path.dirname(groundingdino.__file__), "config", GDINO_CONFIG)

    model = LangSAM2Basic(
        sam2_cfg=SAM2_CONFIG,
        sam2_ckpt=sam2_ckpt,
        gdino_cfg=gdino_config_path,
        gdino_ckpt=gdino_ckpt
    )

    dataset_dir = os.path.join(script_dir, DATASET_FOLDER)
    if not os.path.exists(dataset_dir) and os.path.exists(os.path.join(script_dir, "sample_data", DATASET_FOLDER)):
        dataset_dir = os.path.join(script_dir, "sample_data", DATASET_FOLDER)
    out_base_dir = os.path.join(script_dir, OUTPUT_MASKS_FOLDER)

    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.abspath(os.path.join(root, file))
                rel_path = os.path.relpath(img_path, dataset_dir)

                if "front" not in rel_path.lower():
                    continue

                print(f"\nProcessing: {file}")

                # Load image and construct dual representations in memory
                image_raw = Image.open(img_path)

                if image_raw.mode == 'RGBA':
                    # Composite on explicit black background
                    black_bg = Image.new('RGB', image_raw.size, (0, 0, 0))
                    black_bg.paste(image_raw, mask=image_raw.split()[3])
                    image_normal = black_bg

                    # Composite on explicit white background
                    white_bg = Image.new('RGB', image_raw.size, (255, 255, 255))
                    white_bg.paste(image_raw, mask=image_raw.split()[3])
                    image_white = white_bg
                else:
                    image_normal = image_raw.convert("RGB")
                    image_white = image_normal.copy()

                img_name = os.path.splitext(file)[0]
                out_img_dir = os.path.join(out_base_dir, os.path.dirname(rel_path), img_name)
                os.makedirs(out_img_dir, exist_ok=True)

                masks = model.predict_hybrid(
                    image_normal,
                    image_white,
                    box_threshold=BOX_THRESHOLD,
                    text_threshold=TEXT_THRESHOLD
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
