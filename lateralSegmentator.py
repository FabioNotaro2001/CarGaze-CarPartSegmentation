"""
lateralSegmentator.py
=====================
Automated semantic part segmentation pipeline for lateral-view (side-profile)
vehicle images.

This module combines Grounding DINO (open-vocabulary object detection) and
Segment Anything 2 (SAM 2, high-precision promptable segmentation) with
rigorous spatial hierarchy constraints, wheel-anchored coordinate systems,
and multi-stage organic mask refinement tailored to side-profile automotive anatomy.

Key Methodological Highlights:
- Wheel-Anchored Spatial Frame: Detects front and rear wheels to establish the
  vehicle wheelbase, longitudinal orientation, and ground plane baseline.
- Light Orientation & Fallbacks: Resolves headlights vs. taillights using spatial
  wheel anchors and multi-tiered fallback prompts.
- Door & Window Partitioning: Enforces bilateral bounds between wheels for front and
  rear doors, door windows, door handles, and fuel/charging flaps.
- Organic Door/Window Splitting: If SAM 2 merges a door window into the door panel mask,
  a horizontal profile analysis detects the window-sill inflection line and prompts SAM 2
  separately for the window and door panel.
- Semantic Clean Partitioning: Subtracts door handles and windows from door panel masks
  to ensure disjoint, non-overlapping semantic masks.
- Bumpers with Chromatic Fallback: Detects front and rear bumpers using light and wheel
  anchors, with an HSV/grayscale chromatic fallback for dark rear diffusers.
"""
import os
import cv2
import numpy as np
from PIL import Image
import gc
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from groundingdino.util.inference import load_model as load_gdino, predict as predict_gdino
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
REAR_DOOR_MIN_WIDTH_RATIO = 0.75
REAR_DOOR_AS_FRONT_WIDTH_RATIO = 1.35
FRONT_DOOR_MAX_FRONT_WHEEL_IOU = 0.15
FRONT_DOOR_MAX_FRONT_WHEEL_IOA = 0.30
DOOR_MASK_WINDOW_HEIGHT_RATIO = 1.50
DOOR_MASK_WINDOW_UPPER_EXTENSION_RATIO = 0.95
DOOR_WINDOW_MIN_WIDTH_RATIO = 0.45
DOOR_WINDOW_MAX_WIDTH_RATIO = 1.15
DOOR_WINDOW_MIN_OVERLAP_RATIO = 0.45
DOOR_WINDOW_MAX_HEIGHT_RATIO = 0.75
DOOR_WINDOW_MAX_BOTTOM_RATIO = 0.58
DOOR_WINDOW_MASK_MIN_OVERLAP_RATIO = 0.10
DOOR_MASK_MIN_VEHICLE_HEIGHT_RATIO_FOR_SPLIT = 0.75
REAR_DOOR_WINDOW_MIN_FRONT_HEIGHT_RATIO = 0.60
REAR_DOOR_WINDOW_SPLIT_MIN_FRONT_HEIGHT_RATIO = 0.66
REAR_DOOR_WINDOW_MAX_FRONT_HEIGHT_RATIO = 1.35
FRONT_DOOR_WINDOW_MIN_CENTER_OFFSET_RATIO = -0.05
FALLBACK_DOOR_WINDOW_WIDTH_RATIO = 0.72
FALLBACK_DOOR_WINDOW_HEIGHT_RATIO = 0.42
FALLBACK_DOOR_WINDOW_BOTTOM_OFFSET_RATIO = 0.05
FALLBACK_FRONT_DOOR_WINDOW_CENTER_SHIFT_RATIO = 0.04
FALLBACK_REAR_DOOR_WINDOW_CENTER_SHIFT_RATIO = 0.00
HANDLE_MIRROR_MAX_OVERLAP_RATIO = 0.30
FUEL_DOOR_MIN_WHEEL_AREA_RATIO = 0.003
FUEL_DOOR_MAX_WHEEL_AREA_RATIO = 0.28
FUEL_DOOR_MIN_WHEEL_SIDE_RATIO = 0.08
FUEL_DOOR_MAX_WHEEL_SIDE_RATIO = 0.58
FUEL_DOOR_MAX_FORBIDDEN_IOA = 0.35
FUEL_DOOR_MAX_WHEEL_IOA = 0.18
FUEL_DOOR_BOX_MASK_INSET_RATIO = 0.10

# Class mapping: [Display name, Color (BGR)]
CLASS_MAP = {
    "front wheel": ["Front Wheel", (0, 255, 0)],          # Green
    "rear wheel": ["Rear Wheel", (0, 0, 255)],            # Red
    "headlight": ["Headlight", (255, 255, 0)],            # Cyan
    "taillight": ["Taillight", (255, 0, 255)],            # Magenta
    "rearview mirror": ["Rearview Mirror", (0, 165, 255)],# Orange
    "front door": ["Front Door", (255, 140, 0)],          # Blue Orange
    "rear door": ["Rear Door", (128, 255, 0)],            # Lime Green
    "front door window": ["Front Door Window", (255, 215, 0)],
    "rear door window": ["Rear Door Window", (0, 191, 255)],
    "front door handle": ["Front Door Handle", (147, 20, 255)], # Dark Pink
    "rear door handle": ["Rear Door Handle", (20, 147, 255)],   # Light Brown
    "fuel door": ["Fuel Door", (40, 40, 40)],
    # Bumper classes
    "front bumper": ["Front Bumper", (128, 0, 128)],  # Dark Purple
    "rear bumper": ["Rear Bumper", (0, 128, 128)],    # Dark Teal
}

# =============================================================================
# --- 2. VISUALIZATION HELPERS ---
# =============================================================================
def generate_summary_image_with_colored_legend(image_pil, unique_masks):
    img_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    h, w, _ = img_bgr.shape
    
    background_faded = cv2.addWeighted(img_bgr, 0.4, np.full((h, w, 3), 220, dtype=np.uint8), 0.6, 0)
    
    mask_composite = np.zeros((h, w, 3), dtype=np.uint8)
    masks_found_count = 0
    
    for prompt, mask in reversed(list(unique_masks.items())):
        if prompt not in CLASS_MAP: continue
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
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

    def scale_boxes_to_image(self, boxes, w, h):
        if len(boxes) == 0:
            return np.empty((0, 4), dtype=np.float32)

        boxes = boxes * torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes.cpu().numpy()

    def get_box_center(self, box):
        return (
            float((box[0] + box[2]) / 2),
            float((box[1] + box[3]) / 2),
        )

    def get_box_area(self, box):
        return float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))

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

    def calculate_ioa(self, box_a, box_b):
        x_left = max(box_a[0], box_b[0])
        y_top = max(box_a[1], box_b[1])
        x_right = min(box_a[2], box_b[2])
        y_bottom = min(box_a[3], box_b[3])

        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        area_a = self.get_box_area(box_a)
        if area_a <= 0:
            return 0.0
        return float(intersection / area_a)

    def get_mask_bbox(self, mask):
        mask_np = mask.cpu().numpy().astype(bool)
        ys, xs = np.where(mask_np)
        if len(xs) == 0 or len(ys) == 0:
            return None
        return np.array(
            [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
            dtype=np.float32
        )

    def get_mask_area(self, mask):
        return int(mask.bool().sum().item())

    def get_mask_overlap_ratio(self, mask_a, mask_b):
        if mask_a is None or mask_b is None:
            return 0.0

        mask_a = mask_a.bool()
        mask_b = mask_b.bool()

        if not torch.any(mask_a) or not torch.any(mask_b):
            return 0.0

        overlap_area = self.get_mask_area(torch.logical_and(mask_a, mask_b))
        area_a = self.get_mask_area(mask_a)
        if area_a <= 0:
            return 0.0
        return float(overlap_area) / float(area_a)

    def get_box_height(self, box):
        if box is None:
            return None
        return float(max(1.0, box[3] - box[1] + 1.0))

    def get_mask_height(self, mask):
        if mask is None or not torch.any(mask):
            return None
        return self.get_box_height(self.get_mask_bbox(mask))

    def get_masks_union_bbox(self, masks):
        valid_masks = [mask.bool() for mask in masks if mask is not None and torch.any(mask)]
        if not valid_masks:
            return None

        union_mask = valid_masks[0].clone()
        for mask in valid_masks[1:]:
            union_mask = torch.logical_or(union_mask, mask.bool())
        return self.get_mask_bbox(union_mask)

    def get_interval_overlap(self, start_a, end_a, start_b, end_b):
        return max(0.0, min(end_a, end_b) - max(start_a, start_b))

    def deduplicate_candidates(self, candidates, iou_threshold=0.35):
        if not candidates:
            return []

        unique_candidates = []
        for logit, box in sorted(candidates, key=lambda item: item[0], reverse=True):
            if any(self.calculate_iou(box, kept_box) > iou_threshold for _, kept_box in unique_candidates):
                continue
            unique_candidates.append((float(logit), box))
        return unique_candidates

    def rank_light_candidate(self, candidate, target_label, image_width, anchor_box=None):
        logit, box = candidate
        cx, cy = self.get_box_center(box)
        edge_distance = cx if target_label == "headlight" else (image_width - cx)

        if anchor_box is not None:
            anchor_cx, _ = self.get_box_center(anchor_box)
            anchor_top = float(anchor_box[1])
            vertical_distance = abs(cy - anchor_top)
            horizontal_distance = abs(cx - anchor_cx)
        else:
            vertical_distance = 0.0
            horizontal_distance = edge_distance

        return (edge_distance, vertical_distance, horizontal_distance, -float(logit))

    def select_light_candidate(self, candidates, target_label, image_size, ref_wheel_area=float("inf"), front_wheel_box=None, rear_wheel_box=None):
        if not candidates:
            return None

        w, h = image_size
        anchor_box = front_wheel_box if target_label == "headlight" else rear_wheel_box
        rear_wheel_cx = None
        if rear_wheel_box is not None:
            rear_wheel_cx, _ = self.get_box_center(rear_wheel_box)
        profiles = []

        if anchor_box is not None:
            anchor_cx, anchor_cy = self.get_box_center(anchor_box)
            anchor_left, anchor_top, anchor_right, _ = anchor_box

            # Apply progressively relaxed constraints until a best-effort candidate is selected.
            profiles = [
                {
                    "max_area_ratio": 0.80,
                    "max_x": anchor_cx if target_label == "headlight" else None,
                    "min_x": anchor_cx if target_label == "taillight" else None,
                    "edge_fraction": 0.50,
                    "min_y": max(0.0, anchor_top - 0.10 * h),
                    "max_y": min(float(h), anchor_cy + 0.10 * h),
                },
                {
                    "max_area_ratio": 1.20,
                    "max_x": min(float(w), anchor_cx + 0.08 * w) if target_label == "headlight" else None,
                    "min_x": max(0.0, anchor_cx - 0.08 * w) if target_label == "taillight" else None,
                    "edge_fraction": 0.58,
                    "min_y": max(0.0, anchor_top - 0.16 * h),
                    "max_y": min(float(h), anchor_cy + 0.16 * h),
                },
                {
                    "max_area_ratio": 1.80,
                    "max_x": min(float(w), anchor_right + 0.10 * w) if target_label == "headlight" else None,
                    "min_x": max(0.0, anchor_left - 0.10 * w) if target_label == "taillight" else None,
                    "edge_fraction": 0.68,
                    "min_y": max(0.0, anchor_top - 0.24 * h),
                    "max_y": min(float(h), anchor_cy + 0.24 * h),
                },
            ]

        profiles.extend([
            {
                "max_area_ratio": 2.50,
                "edge_fraction": 0.72,
                "min_y": 0.20 * h,
                "max_y": 0.82 * h,
            },
            {
                "max_area_ratio": None,
                "edge_fraction": 0.85,
                "min_y": 0.15 * h,
                "max_y": 0.90 * h,
            },
        ])

        for profile in profiles:
            filtered_candidates = []

            for candidate in candidates:
                _, box = candidate
                cx, cy = self.get_box_center(box)
                area = self.get_box_area(box)
                max_area_ratio = profile.get("max_area_ratio")

                # Absolute constraint: taillight must reside to the right of the rear wheel center.
                if target_label == "taillight" and rear_wheel_cx is not None and cx <= rear_wheel_cx:
                    continue

                if max_area_ratio is not None and np.isfinite(ref_wheel_area):
                    if area > max_area_ratio * ref_wheel_area:
                        continue

                if target_label == "headlight":
                    max_x = profile.get("max_x")
                    if max_x is not None and cx > max_x:
                        continue
                    if cx > profile["edge_fraction"] * w:
                        continue
                else:
                    min_x = profile.get("min_x")
                    if min_x is not None and cx < min_x:
                        continue
                    if cx < (1.0 - profile["edge_fraction"]) * w:
                        continue

                if cy < profile["min_y"] or cy > profile["max_y"]:
                    continue

                filtered_candidates.append(candidate)

            if filtered_candidates:
                filtered_candidates.sort(
                    key=lambda item: self.rank_light_candidate(item, target_label, w, anchor_box)
                )
                return filtered_candidates[0][1]

        return None

    def select_door_boxes(self, candidates, image_size, ref_wheel_area=float("inf"), front_wheel_box=None, rear_wheel_box=None):
        if not candidates:
            return {}

        w, h = image_size

        if front_wheel_box is not None and rear_wheel_box is not None:
            front_cx, _ = self.get_box_center(front_wheel_box)
            rear_cx, _ = self.get_box_center(rear_wheel_box)

            if front_cx > rear_cx:
                front_wheel_box, rear_wheel_box = rear_wheel_box, front_wheel_box
                front_cx, rear_cx = rear_cx, front_cx

            gap_left = front_cx
            gap_right = rear_cx
            wheel_top = min(front_wheel_box[1], rear_wheel_box[1])
            wheel_bottom = max(front_wheel_box[3], rear_wheel_box[3])
        else:
            gap_left = 0.22 * w
            gap_right = 0.82 * w
            wheel_top = 0.58 * h
            wheel_bottom = 0.92 * h

        gap_width = max(1.0, gap_right - gap_left)
        min_area = 0.35 * ref_wheel_area if np.isfinite(ref_wheel_area) else 0.012 * w * h
        max_area = 10.0 * ref_wheel_area if np.isfinite(ref_wheel_area) else 0.35 * w * h

        filtered_candidates = []
        for logit, box in self.deduplicate_candidates(candidates):
            box_w = float(max(0.0, box[2] - box[0]))
            box_h = float(max(0.0, box[3] - box[1]))
            area = self.get_box_area(box)
            cx, cy = self.get_box_center(box)
            horizontal_overlap = self.get_interval_overlap(box[0], box[2], gap_left, gap_right)

            if box_w < 0.07 * w or box_h < 0.12 * h:
                continue
            if area < min_area or area > max_area:
                continue
            if horizontal_overlap / max(1.0, box_w) < 0.35:
                continue
            if cx < gap_left - 0.12 * gap_width or cx > gap_right + 0.12 * gap_width:
                continue
            if cy < 0.24 * h or cy > 0.78 * h:
                continue
            if box[1] < 0.18 * h:
                continue
            if box[3] < wheel_top + 0.03 * h:
                continue
            if box[3] > wheel_bottom + 0.02 * h:
                continue

            filtered_candidates.append((float(logit), box))

        if not filtered_candidates:
            return {}

        filtered_candidates.sort(key=lambda item: self.get_box_center(item[1])[0])

        if front_wheel_box is not None and rear_wheel_box is not None and len(filtered_candidates) >= 2:
            target_front_x = gap_left + 0.36 * gap_width
            target_rear_x = gap_left + 0.68 * gap_width
            pair_candidates = []

            for left_idx in range(len(filtered_candidates)):
                left_logit, left_box = filtered_candidates[left_idx]
                left_cx, left_cy = self.get_box_center(left_box)
                left_w = float(max(0.0, left_box[2] - left_box[0]))

                if left_w >= 0.92 * gap_width:
                    continue
                if abs(left_cx - front_cx) >= abs(left_cx - rear_cx):
                    continue
                if (
                    self.calculate_iou(left_box, front_wheel_box) > FRONT_DOOR_MAX_FRONT_WHEEL_IOU or
                    self.calculate_ioa(left_box, front_wheel_box) > FRONT_DOOR_MAX_FRONT_WHEEL_IOA
                ):
                    continue

                for right_idx in range(left_idx + 1, len(filtered_candidates)):
                    right_logit, right_box = filtered_candidates[right_idx]
                    right_cx, right_cy = self.get_box_center(right_box)
                    right_w = float(max(0.0, right_box[2] - right_box[0]))

                    if right_w >= 0.92 * gap_width:
                        continue
                    if abs(right_cx - rear_cx) >= abs(right_cx - front_cx):
                        continue
                    if self.calculate_iou(left_box, right_box) > 0.20:
                        continue
                    if right_cx - left_cx < 0.16 * gap_width:
                        continue
                    if left_cx > gap_left + 0.62 * gap_width:
                        continue
                    if right_cx < gap_left + 0.38 * gap_width:
                        continue
                    if abs(left_cy - right_cy) > 0.12 * h:
                        continue

                    coverage_ratio = (left_w + right_w) / gap_width
                    if coverage_ratio < 0.28 or coverage_ratio > 1.35:
                        continue

                    score = (
                        abs(left_cx - target_front_x) +
                        abs(right_cx - target_rear_x) +
                        0.30 * abs(left_cy - right_cy) -
                        45.0 * (left_logit + right_logit)
                    )
                    pair_candidates.append((score, left_box, right_box))

            if pair_candidates:
                pair_candidates.sort(key=lambda item: item[0])
                _, front_box, rear_box = pair_candidates[0]
                front_width = float(max(0.0, front_box[2] - front_box[0]))
                rear_width = float(max(0.0, rear_box[2] - rear_box[0]))

                if rear_width < REAR_DOOR_MIN_WIDTH_RATIO * front_width:
                    return {"front door": front_box}

                if rear_width > REAR_DOOR_AS_FRONT_WIDTH_RATIO * front_width:
                    return {"front door": rear_box}

                return {
                    "front door": front_box,
                    "rear door": rear_box,
                }

        target_single_x = gap_left + 0.45 * gap_width

        def single_door_rank(candidate):
            logit, box = candidate
            cx, cy = self.get_box_center(box)
            box_w = float(max(0.0, box[2] - box[0]))
            width_bonus = min(1.0, box_w / gap_width)
            wheel_overlap_penalty = 0.0
            front_distance_penalty = 0.0

            if front_wheel_box is not None:
                wheel_overlap_penalty += 800.0 * self.calculate_iou(box, front_wheel_box)
                wheel_overlap_penalty += 800.0 * self.calculate_ioa(box, front_wheel_box)

            if front_wheel_box is not None and rear_wheel_box is not None:
                front_distance = abs(cx - front_cx)
                rear_distance = abs(cx - rear_cx)
                if front_distance >= rear_distance:
                    front_distance_penalty = 250.0 + (front_distance - rear_distance)

            return (
                abs(cx - target_single_x) - 30.0 * float(logit) - 25.0 * width_bonus +
                wheel_overlap_penalty + front_distance_penalty,
                abs(cy - 0.55 * h),
            )

        best_single_box = min(filtered_candidates, key=single_door_rank)[1]
        return {"front door": best_single_box}

    def build_box_mask(self, box, image_size, inset_ratio=0.0):
        if box is None:
            return None

        w, h = image_size
        x1 = int(np.floor(max(0.0, float(box[0]))))
        y1 = int(np.floor(max(0.0, float(box[1]))))
        x2 = int(np.ceil(min(float(w - 1), float(box[2]))))
        y2 = int(np.ceil(min(float(h - 1), float(box[3]))))

        if x2 <= x1 or y2 <= y1:
            return None

        box_w = x2 - x1 + 1
        box_h = y2 - y1 + 1
        inset_x = int(round(inset_ratio * box_w))
        inset_y = int(round(inset_ratio * box_h))

        if box_w - 2 * inset_x < 3:
            inset_x = 0
        if box_h - 2 * inset_y < 3:
            inset_y = 0

        x1 += inset_x
        x2 -= inset_x
        y1 += inset_y
        y2 -= inset_y

        if x2 <= x1 or y2 <= y1:
            return None

        mask = torch.zeros((h, w), dtype=torch.bool)
        mask[y1:y2 + 1, x1:x2 + 1] = True
        return mask

    def fuel_door_mask_is_unstable(self, mask, box, ref_wheel_area=float("inf")):
        if mask is None or box is None or not torch.any(mask):
            return True

        mask_bbox = self.get_mask_bbox(mask)
        if mask_bbox is None:
            return True

        box_w = float(max(1.0, box[2] - box[0]))
        box_h = float(max(1.0, box[3] - box[1]))
        box_area = self.get_box_area(box)
        mask_bbox_w = float(max(1.0, mask_bbox[2] - mask_bbox[0]))
        mask_bbox_h = float(max(1.0, mask_bbox[3] - mask_bbox[1]))
        mask_area = float(self.get_mask_area(mask))

        if box_area <= 0:
            return True
        if mask_area > 3.5 * box_area:
            return True
        if mask_bbox_w > 1.8 * box_w or mask_bbox_h > 1.8 * box_h:
            return True
        if np.isfinite(ref_wheel_area) and mask_area > 0.30 * ref_wheel_area:
            return True

        return False

    def select_fuel_door_candidate(
        self,
        candidates,
        image_size,
        ref_wheel_area=float("inf"),
        front_wheel_box=None,
        rear_wheel_box=None,
        front_door_box=None,
        rear_door_box=None,
        forbidden_boxes=None,
    ):
        if not candidates or rear_wheel_box is None:
            return None

        w, h = image_size
        forbidden_boxes = [box for box in (forbidden_boxes or []) if box is not None]
        deduplicated_candidates = self.deduplicate_candidates(candidates, iou_threshold=0.12)

        rear_cx, _ = self.get_box_center(rear_wheel_box)
        rear_wheel_w = float(max(1.0, rear_wheel_box[2] - rear_wheel_box[0]))
        rear_wheel_h = float(max(1.0, rear_wheel_box[3] - rear_wheel_box[1]))

        if front_wheel_box is not None:
            front_cx, _ = self.get_box_center(front_wheel_box)
            if front_cx > rear_cx:
                front_wheel_box, rear_wheel_box = rear_wheel_box, front_wheel_box
                front_cx, rear_cx = rear_cx, front_cx
                rear_wheel_w = float(max(1.0, rear_wheel_box[2] - rear_wheel_box[0]))
                rear_wheel_h = float(max(1.0, rear_wheel_box[3] - rear_wheel_box[1]))
            gap_left = front_cx
            gap_width = max(1.0, rear_cx - front_cx)
        else:
            gap_width = max(0.30 * w, 3.0 * rear_wheel_w)
            gap_left = max(0.0, rear_cx - gap_width)

        if np.isfinite(ref_wheel_area):
            reference_wheel_area = float(ref_wheel_area)
        else:
            reference_wheel_area = rear_wheel_w * rear_wheel_h

        profiles = [
            {
                "target_x": rear_cx - 0.04 * gap_width,
                "target_y": rear_wheel_box[1] - 0.18 * rear_wheel_h,
                "min_x": rear_cx - 0.34 * gap_width,
                "max_x": rear_cx + 0.12 * gap_width,
                "min_y": rear_wheel_box[1] - 0.72 * rear_wheel_h,
                "max_y": rear_wheel_box[1] + 0.08 * rear_wheel_h,
                "front_door_penalty": 0.45,
                "rear_door_bonus": 0.18,
                "max_score": 2.05,
            },
            {
                "target_x": rear_cx - 0.02 * gap_width,
                "target_y": rear_wheel_box[1] - 0.42 * rear_wheel_h,
                "min_x": rear_cx - 0.28 * gap_width,
                "max_x": rear_cx + 0.10 * gap_width,
                "min_y": rear_wheel_box[1] - 1.05 * rear_wheel_h,
                "max_y": rear_wheel_box[1] - 0.05 * rear_wheel_h,
                "front_door_penalty": 0.35,
                "rear_door_bonus": 0.12,
                "max_score": 2.20,
            },
            {
                "target_x": gap_left + 0.62 * gap_width,
                "target_y": rear_wheel_box[1] - 0.18 * rear_wheel_h,
                "min_x": gap_left + 0.40 * gap_width,
                "max_x": gap_left + 0.82 * gap_width,
                "min_y": rear_wheel_box[1] - 0.78 * rear_wheel_h,
                "max_y": rear_wheel_box[1] + 0.10 * rear_wheel_h,
                "front_door_penalty": 0.22,
                "rear_door_bonus": 0.10,
                "max_score": 2.25,
            },
            {
                "target_x": gap_left + 0.36 * gap_width,
                "target_y": rear_wheel_box[1] - 0.16 * rear_wheel_h,
                "min_x": gap_left + 0.14 * gap_width,
                "max_x": gap_left + 0.54 * gap_width,
                "min_y": rear_wheel_box[1] - 0.76 * rear_wheel_h,
                "max_y": rear_wheel_box[1] + 0.10 * rear_wheel_h,
                "front_door_penalty": 0.08,
                "rear_door_bonus": 0.00,
                "max_score": 2.30,
            },
        ]

        for profile in profiles:
            ranked_candidates = []

            for logit, box in deduplicated_candidates:
                box_w = float(max(0.0, box[2] - box[0]))
                box_h = float(max(0.0, box[3] - box[1]))
                area = self.get_box_area(box)
                cx, cy = self.get_box_center(box)
                aspect_ratio = box_w / max(1.0, box_h)
                size_ratio = np.sqrt(max(1.0, area) / max(1.0, reference_wheel_area))

                if box_w < FUEL_DOOR_MIN_WHEEL_SIDE_RATIO * rear_wheel_w:
                    continue
                if box_h < FUEL_DOOR_MIN_WHEEL_SIDE_RATIO * rear_wheel_h:
                    continue
                if box_w > FUEL_DOOR_MAX_WHEEL_SIDE_RATIO * rear_wheel_w:
                    continue
                if box_h > FUEL_DOOR_MAX_WHEEL_SIDE_RATIO * rear_wheel_h:
                    continue
                if area < FUEL_DOOR_MIN_WHEEL_AREA_RATIO * reference_wheel_area:
                    continue
                if area > FUEL_DOOR_MAX_WHEEL_AREA_RATIO * reference_wheel_area:
                    continue
                if aspect_ratio < 0.55 or aspect_ratio > 1.75:
                    continue
                if cx < profile["min_x"] or cx > profile["max_x"]:
                    continue
                if cy < profile["min_y"] or cy > profile["max_y"]:
                    continue

                if front_wheel_box is not None and self.calculate_ioa(box, front_wheel_box) > FUEL_DOOR_MAX_WHEEL_IOA:
                    continue
                if self.calculate_ioa(box, rear_wheel_box) > FUEL_DOOR_MAX_WHEEL_IOA:
                    continue

                if any(self.calculate_ioa(box, forbidden_box) > FUEL_DOOR_MAX_FORBIDDEN_IOA for forbidden_box in forbidden_boxes):
                    continue

                front_door_ioa = self.calculate_ioa(box, front_door_box) if front_door_box is not None else 0.0
                rear_door_ioa = self.calculate_ioa(box, rear_door_box) if rear_door_box is not None else 0.0

                score = (
                    1.55 * abs(cx - profile["target_x"]) / gap_width +
                    1.20 * abs(cy - profile["target_y"]) / rear_wheel_h +
                    0.95 * abs(np.log(max(1e-6, aspect_ratio))) +
                    1.20 * abs(size_ratio - 0.30) +
                    profile["front_door_penalty"] * front_door_ioa -
                    profile["rear_door_bonus"] * rear_door_ioa -
                    0.60 * float(logit)
                )

                ranked_candidates.append((score, box))

            if ranked_candidates:
                ranked_candidates.sort(key=lambda item: item[0])
                best_score, best_box = ranked_candidates[0]
                if best_score <= profile["max_score"]:
                    return best_box

        return None

    def select_door_window_candidate(self, candidates, door_box, image_size, other_door_box=None, reference_window_box=None, min_center_x=None):
        if not candidates or door_box is None:
            return None

        door_w = float(max(1.0, door_box[2] - door_box[0]))
        door_h = float(max(1.0, door_box[3] - door_box[1]))
        door_cx, door_cy = self.get_box_center(door_box)
        other_cx = self.get_box_center(other_door_box)[0] if other_door_box is not None else None
        reference_window_h = None
        if reference_window_box is not None:
            reference_window_h = float(max(1.0, reference_window_box[3] - reference_window_box[1]))

        deduplicated_candidates = self.deduplicate_candidates(candidates, iou_threshold=0.25)

        filtered_candidates = []
        for idx, (logit, box) in enumerate(deduplicated_candidates):
            box_w = float(max(0.0, box[2] - box[0]))
            box_h = float(max(0.0, box[3] - box[1]))
            cx, cy = self.get_box_center(box)
            overlap = self.get_interval_overlap(box[0], box[2], door_box[0], door_box[2])
            overlap_ratio = overlap / max(1.0, min(box_w, door_w))
            reject_reason = None

            if box_w < DOOR_WINDOW_MIN_WIDTH_RATIO * door_w:
                reject_reason = f"box troppo stretto ({box_w:.1f} < {DOOR_WINDOW_MIN_WIDTH_RATIO * door_w:.1f})"
            elif box_w > DOOR_WINDOW_MAX_WIDTH_RATIO * door_w:
                reject_reason = f"box troppo largo ({box_w:.1f} > {DOOR_WINDOW_MAX_WIDTH_RATIO * door_w:.1f})"
            elif box_h < 0.12 * door_h:
                reject_reason = f"box troppo basso ({box_h:.1f} < {0.12 * door_h:.1f})"
            elif box_h > DOOR_WINDOW_MAX_HEIGHT_RATIO * door_h:
                reject_reason = f"box troppo alto ({box_h:.1f} > {DOOR_WINDOW_MAX_HEIGHT_RATIO * door_h:.1f})"
            elif overlap_ratio < DOOR_WINDOW_MIN_OVERLAP_RATIO:
                reject_reason = f"overlap insufficiente ({overlap_ratio:.2f} < {DOOR_WINDOW_MIN_OVERLAP_RATIO:.2f})"
            elif cy >= door_cy:
                reject_reason = f"centro troppo in basso (cy={cy:.1f} >= door_cy={door_cy:.1f})"
            elif box[3] > door_box[1] + DOOR_WINDOW_MAX_BOTTOM_RATIO * door_h:
                reject_reason = f"bordo inferiore troppo basso ({box[3]:.1f} > {door_box[1] + DOOR_WINDOW_MAX_BOTTOM_RATIO * door_h:.1f})"
            elif box[0] < door_box[0] - 0.28 * door_w:
                reject_reason = f"box troppo a sinistra ({box[0]:.1f} < {door_box[0] - 0.28 * door_w:.1f})"
            elif box[2] > door_box[2] + 0.20 * door_w:
                reject_reason = f"box troppo a destra ({box[2]:.1f} > {door_box[2] + 0.20 * door_w:.1f})"
            elif min_center_x is not None and cx < min_center_x:
                reject_reason = f"centro troppo a sinistra ({cx:.1f} < {min_center_x:.1f})"
            elif other_cx is not None and abs(cx - door_cx) >= abs(cx - other_cx):
                reject_reason = "piu' vicino all'altra porta che a quella corrente"

            if reject_reason is not None:
                continue

            if reference_window_h is not None:
                if box_h < REAR_DOOR_WINDOW_MIN_FRONT_HEIGHT_RATIO * reference_window_h:
                    continue
                if box_h > REAR_DOOR_WINDOW_MAX_FRONT_HEIGHT_RATIO * reference_window_h:
                    continue

            score = (
                abs(cx - door_cx) +
                0.60 * abs(box[3] - door_box[1]) -
                40.0 * float(logit) -
                120.0 * overlap_ratio
            )
            if reference_window_h is not None:
                score += 1.20 * abs(box_h - reference_window_h)
            filtered_candidates.append((score, float(logit), box))

        if not filtered_candidates:
            return None

        filtered_candidates.sort(key=lambda item: item[0])
        _, _, best_box = filtered_candidates[0]
        return best_box

    def build_fallback_door_window_box(self, door_label, door_box, image_size, reference_window_box=None):
        w, h = image_size
        door_w = float(max(1.0, door_box[2] - door_box[0]))
        door_h = float(max(1.0, door_box[3] - door_box[1]))
        door_cx, _ = self.get_box_center(door_box)

        if reference_window_box is not None:
            ref_w = float(max(1.0, reference_window_box[2] - reference_window_box[0]))
            ref_h = float(max(1.0, reference_window_box[3] - reference_window_box[1]))
            window_w = min(DOOR_WINDOW_MAX_WIDTH_RATIO * door_w, max(DOOR_WINDOW_MIN_WIDTH_RATIO * door_w, ref_w))
            window_h = min(DOOR_WINDOW_MAX_HEIGHT_RATIO * door_h, max(0.18 * door_h, ref_h))
        else:
            window_w = min(DOOR_WINDOW_MAX_WIDTH_RATIO * door_w, max(DOOR_WINDOW_MIN_WIDTH_RATIO * door_w, FALLBACK_DOOR_WINDOW_WIDTH_RATIO * door_w))
            window_h = min(DOOR_WINDOW_MAX_HEIGHT_RATIO * door_h, max(0.18 * door_h, FALLBACK_DOOR_WINDOW_HEIGHT_RATIO * door_h))

        center_shift_ratio = (
            FALLBACK_FRONT_DOOR_WINDOW_CENTER_SHIFT_RATIO
            if door_label == "front door"
            else FALLBACK_REAR_DOOR_WINDOW_CENTER_SHIFT_RATIO
        )
        window_cx = door_cx + center_shift_ratio * door_w
        window_bottom = float(door_box[1]) + FALLBACK_DOOR_WINDOW_BOTTOM_OFFSET_RATIO * door_h
        window_top = window_bottom - window_h

        x1 = max(0.0, window_cx - window_w / 2.0)
        x2 = min(float(w - 1), window_cx + window_w / 2.0)
        y1 = max(0.0, window_top)
        y2 = min(float(h - 1), window_bottom)

        if x2 <= x1 or y2 <= y1:
            return None

        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def door_mask_includes_window(self, door_mask, wheel_box):
        if wheel_box is None:
            return False

        mask_bbox = self.get_mask_bbox(door_mask)
        if mask_bbox is None:
            return False

        mask_w = float(max(1.0, mask_bbox[2] - mask_bbox[0] + 1.0))
        mask_h = float(max(1.0, mask_bbox[3] - mask_bbox[1] + 1.0))
        wheel_h = float(max(1.0, wheel_box[3] - wheel_box[1] + 1.0))
        upper_extension = max(0.0, float(wheel_box[1]) - float(mask_bbox[1]))

        return (
            mask_h >= DOOR_MASK_WINDOW_HEIGHT_RATIO * wheel_h and
            upper_extension >= 0.55 * wheel_h
        ) or (
            upper_extension >= DOOR_MASK_WINDOW_UPPER_EXTENSION_RATIO * wheel_h and
            mask_h / mask_w >= 0.90
        )

    def split_combined_door_mask(self, door_mask, min_upper_height=None):
        mask_np = door_mask.cpu().numpy().astype(bool)
        bbox = self.get_mask_bbox(door_mask)
        if bbox is None:
            return None, None

        x1, y1, x2, y2 = [int(v) for v in bbox]
        crop = mask_np[y1:y2 + 1, x1:x2 + 1]
        if crop.size == 0:
            return None, None

        row_counts = crop.sum(axis=1).astype(np.float32)
        if row_counts.max() <= 0:
            return None, None

        window = max(7, int(0.035 * len(row_counts)))
        if window % 2 == 0:
            window += 1
        kernel = np.ones(window, dtype=np.float32) / float(window)
        smooth_counts = np.convolve(row_counts, kernel, mode="same")

        search_start = int(0.16 * len(smooth_counts))
        search_end = max(search_start + 1, int(0.68 * len(smooth_counts)))
        threshold = 0.82 * float(smooth_counts.max())
        sustain = max(4, int(0.04 * len(smooth_counts)))

        split_idx = None
        for idx in range(search_start, search_end):
            end_idx = min(len(smooth_counts), idx + sustain)
            if end_idx <= idx:
                continue
            if float(np.mean(smooth_counts[idx:end_idx])) >= threshold:
                split_idx = idx
                break

        if split_idx is None:
            split_idx = int(0.38 * len(smooth_counts))

        max_split_idx = len(smooth_counts) - 2
        if max_split_idx < 0:
            return None, None
        split_idx = min(max(0, split_idx), max_split_idx)

        # Maintain initial profile extraction up to split index
        
        if min_upper_height is not None:
            min_upper_rows = int(np.ceil(float(min_upper_height)))
            if min_upper_rows > 0:
                min_split_idx = min(max_split_idx, max(0, min_upper_rows - 1))
                split_idx = max(split_idx, min_split_idx)

        split_y = y1 + split_idx
        
        # Construct two discrete bounding boxes for window and door panel
        upper_box = np.array([x1, y1, x2, split_y], dtype=np.float32)
        lower_box = np.array([x1, split_y, x2, y2], dtype=np.float32)

        return upper_box, lower_box

    def refine_door_component_masks(self, unique_masks, final_boxes, image_normal, image_white):
        refined_masks = {
            prompt: mask.bool()
            for prompt, mask in unique_masks.items()
        }

        vehicle_bbox = self.get_masks_union_bbox([
            refined_masks.get("front wheel"),
            refined_masks.get("rear wheel"),
            refined_masks.get("headlight"),
            refined_masks.get("taillight"),
            refined_masks.get("front door"),
            refined_masks.get("rear door"),
        ])
        vehicle_height = None
        if vehicle_bbox is not None:
            vehicle_height = float(max(1.0, vehicle_bbox[3] - vehicle_bbox[1] + 1.0))

        for door_label, window_label, wheel_label in [
            ("front door", "front door window", "front wheel"),
            ("rear door", "rear door window", "rear wheel"),
        ]:
            door_mask = refined_masks.get(door_label)
            if door_mask is None:
                continue

            includes_window = self.door_mask_includes_window(door_mask, final_boxes.get(wheel_label))
            window_mask = refined_masks.get(window_label)
            door_bbox = self.get_mask_bbox(door_mask)
            force_split_from_height = False
            if vehicle_height is not None and door_bbox is not None:
                door_height = float(max(1.0, door_bbox[3] - door_bbox[1] + 1.0))
                force_split_from_height = (
                    door_height >= DOOR_MASK_MIN_VEHICLE_HEIGHT_RATIO_FOR_SPLIT * vehicle_height
                )

            should_try_split = force_split_from_height or (includes_window and window_mask is None)

            if window_mask is not None:
                overlapped_window = torch.logical_and(door_mask, window_mask)
                overlap_area = self.get_mask_area(overlapped_window)
                window_area = self.get_mask_area(window_mask)
                overlap_ratio = (float(overlap_area) / float(window_area)) if window_area > 0 else 0.0

                if includes_window and window_area > 0:
                    if overlap_ratio >= DOOR_WINDOW_MASK_MIN_OVERLAP_RATIO:
                        lower_door = torch.logical_and(door_mask, torch.logical_not(window_mask))
                        if torch.any(lower_door):
                            refined_masks[door_label] = lower_door
                            refined_masks[window_label] = window_mask.bool()
                            continue

                    should_try_split = True
                elif force_split_from_height:
                    should_try_split = True
                else:
                    refined_masks[window_label] = window_mask
                    continue

            if should_try_split:
                split_min_upper_height = None
                if door_label == "rear door":
                    front_window_height = self.get_mask_height(refined_masks.get("front door window"))
                    if front_window_height is None:
                        front_window_height = self.get_box_height(final_boxes.get("front door window"))
                    if front_window_height is not None:
                        split_min_upper_height = (
                            REAR_DOOR_WINDOW_SPLIT_MIN_FRONT_HEIGHT_RATIO * front_window_height
                        )

                # Derive two regional boxes for organic shape extraction
                upper_box, lower_box = self.split_combined_door_mask(
                    door_mask,
                    min_upper_height=split_min_upper_height
                )
                
                if upper_box is not None and lower_box is not None:
                    # Prompt SAM 2 to segment real organic shapes from derived regional boxes
                    
                    # 1. Segment organic window shape in the upper section
                    self.sam2_predictor.set_image(np.array(image_normal))
                    w_masks, _, _ = self.sam2_predictor.predict(box=np.array([upper_box]), multimask_output=False)
                    split_window = torch.any(torch.tensor(w_masks).squeeze(1), dim=0)
                    
                    # 2. Segment organic door panel in the lower section
                    self.sam2_predictor.set_image(np.array(image_white))
                    d_masks, _, _ = self.sam2_predictor.predict(box=np.array([lower_box]), multimask_output=False)
                    split_door = torch.any(torch.tensor(d_masks).squeeze(1), dim=0)
                    
                    # Subtract window mask from door panel to prevent overlapping pixels
                    split_door = torch.logical_and(split_door, torch.logical_not(split_window))

                    refined_masks[window_label] = split_window.bool()
                    refined_masks[door_label] = split_door.bool()
                    continue

            if window_mask is not None:
                refined_masks[window_label] = window_mask

        empty_prompts = [
            prompt for prompt, mask in refined_masks.items()
            if not torch.any(mask)
        ]
        for prompt in empty_prompts:
            del refined_masks[prompt]

        return refined_masks
    
    def select_bumper_candidate(self, candidates, bumper_label, final_boxes, image_size):
        if not candidates:
            return None

        w, h = image_size
        deduplicated_candidates = self.deduplicate_candidates(candidates, iou_threshold=0.20)
        
        # Identify reference anchors depending on whether searching front or rear
        is_front = (bumper_label == "front bumper")
        anchor_light_label = "headlight" if is_front else "taillight"
        anchor_wheel_label = "front wheel" if is_front else "rear wheel"
        
        anchor_light_box = final_boxes.get(anchor_light_label)
        anchor_wheel_box = final_boxes.get(anchor_wheel_label)
        
        filtered_candidates = []
        
        for logit, box in deduplicated_candidates:
            cx, cy = self.get_box_center(box)
            box_w = float(max(0.0, box[2] - box[0]))
            box_h = float(max(0.0, box[3] - box[1]))
            
            # Discard undersized bounding boxes
            if box_w < 0.05 * w or box_h < 0.05 * h:
                continue

            reject = False
            
            # Constraints based on light anchor (if available)
            if anchor_light_box is not None:
                light_cx, light_cy = self.get_box_center(anchor_light_box)
                light_bottom = float(anchor_light_box[3])
                light_top = float(anchor_light_box[1])
                
                # Primary constraint: Bumper center should be below light center and not extend too high.
                if box[1] > light_cy - (0.05 * h): 
                    # Acceptable: Starts below or slightly above light vertical midpoint
                    pass
                else:
                    reject = True  # Too high (e.g., confused hood/roof for bumper)
                    
                # Horizontal spatial constraint
                if is_front and cx > light_cx + 0.15 * w:
                    reject = True  # Front bumper candidate positioned too far backward
                if not is_front and cx < light_cx - 0.15 * w:
                    reject = True  # Rear bumper candidate positioned too far forward
            
            # Fallback constraints based on wheel anchor if light is missing
            elif anchor_wheel_box is not None:
                wheel_cx, wheel_cy = self.get_box_center(anchor_wheel_box)
                if is_front and cx > wheel_cx + 0.10 * w:
                    reject = True
                if not is_front and cx < wheel_cx - 0.10 * w:
                    reject = True
                    
            if reject:
                continue
                
            # Calculate score prioritizing confidence and proximity to vehicle extremity
            edge_distance = cx if is_front else (w - cx)
            score = (edge_distance / w) - 2.0 * float(logit)
            filtered_candidates.append((score, box))
            
        if filtered_candidates:
            filtered_candidates.sort(key=lambda item: item[0])
            return filtered_candidates[0][1]  # Return highest-scoring box candidate
            
        return None

    def build_rear_bumper_fallback_mask(self, image_white, final_boxes):
        rear_wheel_box = final_boxes.get("rear wheel")
        taillight_box = final_boxes.get("taillight")
        if rear_wheel_box is None or taillight_box is None:
            return None

        image_rgb = np.array(image_white)
        h, w = image_rgb.shape[:2]

        rear_wheel_cx, _ = self.get_box_center(rear_wheel_box)
        _, taillight_cy = self.get_box_center(taillight_box)
        rear_wheel_w = float(max(1.0, rear_wheel_box[2] - rear_wheel_box[0]))
        rear_wheel_area = self.get_box_area(rear_wheel_box)
        taillight_h = float(max(1.0, taillight_box[3] - taillight_box[1]))

        # Chromatic fallback for dark diffusers/bumpers: isolate dark/gray regions behind rear wheel and below taillight.
        search_left = int(np.floor(max(0.0, max(rear_wheel_cx, rear_wheel_box[2] - 0.08 * rear_wheel_w))))
        search_top = int(np.floor(max(0.0, max(taillight_box[1] + 0.10 * taillight_h, taillight_cy - 0.02 * h))))
        search_right = w - 1
        search_bottom = h - 1

        if search_right <= search_left or search_bottom <= search_top:
            return None

        roi = image_rgb[search_top:search_bottom + 1, search_left:search_right + 1]
        if roi.size == 0:
            return None

        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        roi_sat = roi_hsv[:, :, 1]
        roi_val = roi_hsv[:, :, 2]

        dark_gray_mask = np.logical_or(
            np.logical_and(roi_gray <= 150, roi_sat <= 110),
            np.logical_and(roi_gray <= 120, roi_val <= 140),
        )
        if not np.any(dark_gray_mask):
            return None

        candidate_mask = (dark_gray_mask.astype(np.uint8) * 255)
        candidate_mask = cv2.morphologyEx(
            candidate_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=2,
        )
        candidate_mask = cv2.morphologyEx(
            candidate_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask, connectivity=8)
        min_component_area = max(0.002 * w * h, 0.05 * rear_wheel_area)

        best_label = None
        best_score = None

        for label_idx in range(1, num_labels):
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            comp_x = float(search_left + stats[label_idx, cv2.CC_STAT_LEFT])
            comp_y = float(search_top + stats[label_idx, cv2.CC_STAT_TOP])
            comp_w = float(stats[label_idx, cv2.CC_STAT_WIDTH])
            comp_h = float(stats[label_idx, cv2.CC_STAT_HEIGHT])
            comp_right = comp_x + comp_w
            comp_bottom = comp_y + comp_h

            if area < min_component_area:
                continue
            if comp_w < 0.05 * w or comp_h < 0.04 * h:
                continue
            if comp_w > 0.36 * w or comp_h > 0.40 * h:
                continue
            if comp_right <= rear_wheel_box[2] - 0.01 * w:
                continue
            if comp_x <= rear_wheel_cx - 0.02 * w:
                continue
            if comp_y <= taillight_box[1] - 0.02 * h:
                continue
            if comp_bottom <= taillight_cy + 0.02 * h:
                continue

            score = (area, comp_right, -comp_y)
            if best_score is None or score > best_score:
                best_score = score
                best_label = label_idx

        if best_label is None:
            return None

        full_mask = np.zeros((h, w), dtype=bool)
        full_mask[search_top:search_bottom + 1, search_left:search_right + 1] = (labels == best_label)
        return torch.from_numpy(full_mask).bool()

    def predict_parts(self, image_normal, image_white, box_threshold=0.25, text_threshold=0.20):
        w, h = image_white.size
        
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        img_normal_trans, _ = transform(image_normal, None)
        img_white_trans, _ = transform(image_white, None)
        
        # --- 1. Dynamic Threshold Detection Loop ---
        current_b_thresh = box_threshold
        current_t_thresh = text_threshold
        max_attempts = 4
        
        boxes, logits, phrases = None, None, None
        
        for attempt in range(max_attempts):
            boxes, logits, phrases = predict_gdino(
                model=self.grounding_dino,
                image=img_white_trans,
                # Query primary lateral elements including rearview mirror
                caption="wheel . headlight . taillight . rearview mirror", 
                box_threshold=current_b_thresh,
                text_threshold=current_t_thresh,
                device=self.device
            )
            
            light_count = sum(1 for p in phrases if "headlight" in p or "taillight" in p)
            
            if light_count >= 2:
                break
                
            current_b_thresh = max(0.05, current_b_thresh - 0.08)
            current_t_thresh = max(0.05, current_t_thresh - 0.08)
            
        unique_masks = {}
        if len(boxes) > 0:
            boxes_np = self.scale_boxes_to_image(boxes, w, h)
            
            wheel_boxes = []
            light_boxes = []
            mirror_boxes = [] # candidate collector
            
            for i, phrase in enumerate(phrases):
                if "wheel" in phrase:
                    wheel_boxes.append(boxes_np[i])
                elif "headlight" in phrase or "taillight" in phrase:
                    light_boxes.append((float(logits[i]), boxes_np[i]))
                elif "mirror" in phrase: # candidate collector
                    mirror_boxes.append((float(logits[i]), boxes_np[i]))
            
            final_boxes = {}
            front_wheel_cx = None
            rear_wheel_cx = None
            
            # --- Single Wheel Reference Area Calculation ---
            ref_wheel_area = float('inf')
            if len(wheel_boxes) > 0:
                areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in wheel_boxes]
                ref_wheel_area = sum(areas) / len(areas)
            
            # --- 2. Wheels Localization & Orientation ---
            if len(wheel_boxes) > 0:
                boxes_with_centers = []
                for box in wheel_boxes:
                    cx = (box[0] + box[2]) / 2
                    boxes_with_centers.append((cx, box))
                boxes_with_centers.sort(key=lambda x: x[0])
                
                if len(boxes_with_centers) >= 1:
                    front_wheel_cx = boxes_with_centers[0][0]
                    final_boxes["front wheel"] = boxes_with_centers[0][1]
                if len(boxes_with_centers) >= 2:
                    rear_wheel_cx = boxes_with_centers[-1][0]
                    final_boxes["rear wheel"] = boxes_with_centers[-1][1]

            # --- 3. Primary Lights Localization ---
            if len(light_boxes) > 0:
                valid_size_lights = [item for item in light_boxes if (item[1][2]-item[1][0])*(item[1][3]-item[1][1]) < 0.8 * ref_wheel_area]
                if not valid_size_lights:
                    valid_size_lights = light_boxes 
                
                # Assign Headlight candidate
                valid_hl = [item for item in valid_size_lights if front_wheel_cx is None or (item[1][0]+item[1][2])/2 < front_wheel_cx]
                if valid_hl:
                    valid_hl.sort(key=lambda x: (x[1][1] + x[1][3]) / 2)
                    final_boxes["headlight"] = valid_hl[0][1]
                    
                # Assign Taillight candidate
                valid_tl = [item for item in valid_size_lights if rear_wheel_cx is None or (item[1][0]+item[1][2])/2 > rear_wheel_cx]
                if valid_tl:
                    valid_tl.sort(key=lambda x: x[0], reverse=True) 
                    final_boxes["taillight"] = valid_tl[0][1]

            # =========================================================================
            # --- 3.5 Fallback Search for Missing Lights ---
            # =========================================================================
            missing_headlight = "headlight" not in final_boxes
            missing_taillight = "taillight" not in final_boxes

            if missing_headlight or missing_taillight:
                front_wheel_box = final_boxes.get("front wheel")
                rear_wheel_box = final_boxes.get("rear wheel")
                fallback_queries = []

                if missing_headlight:
                    fallback_queries.append((
                        "headlight",
                        [
                            ("headlight . front light . headlamp . front lamp", 0.15, 0.15),
                            ("daytime running light . drl . led strip . front light", 0.10, 0.10),
                            ("car light . lamp . led . side marker", 0.06, 0.06),
                        ],
                    ))

                if missing_taillight:
                    fallback_queries.append((
                        "taillight",
                        [
                            ("taillight . tail light . rear light . rear lamp", 0.15, 0.15),
                            ("brake light . tail lamp . rear led . marker light", 0.10, 0.10),
                            ("car light . led strip . lamp . reflector", 0.06, 0.06),
                        ],
                    ))

                for missing_label, query_steps in fallback_queries:
                    fallback_candidates = list(light_boxes)

                    for caption, fb_box_threshold, fb_text_threshold in query_steps:
                        fb_boxes, fb_logits, _ = predict_gdino(
                            model=self.grounding_dino,
                            image=img_white_trans,
                            caption=caption,
                            box_threshold=fb_box_threshold,
                            text_threshold=fb_text_threshold,
                            device=self.device
                        )

                        if len(fb_boxes) > 0:
                            fb_boxes_np = self.scale_boxes_to_image(fb_boxes, w, h)
                            fallback_candidates.extend(
                                (float(fb_logits[i]), fb_boxes_np[i]) for i in range(len(fb_boxes_np))
                            )

                        selected_box = self.select_light_candidate(
                            fallback_candidates,
                            missing_label,
                            (w, h),
                            ref_wheel_area=ref_wheel_area,
                            front_wheel_box=front_wheel_box,
                            rear_wheel_box=rear_wheel_box,
                        )

                        if selected_box is not None:
                            final_boxes[missing_label] = selected_box
                            break

                missing_headlight = "headlight" not in final_boxes
                missing_taillight = "taillight" not in final_boxes

                # Generic synonyms fallback for vehicle lights not matched by specific prompts
                fallback_caption = "car light . led . lamp"
                fb_boxes, fb_logits, fb_phrases = predict_gdino(
                    model=self.grounding_dino,
                    image=img_white_trans,
                    caption=fallback_caption,
                    box_threshold=0.15, # Fixed reasonable threshold for fallback detection
                    text_threshold=0.15,
                    device=self.device
                )
                
                if len(fb_boxes) > 0:
                    fb_boxes_np = self.scale_boxes_to_image(fb_boxes, w, h)
                    fb_light_boxes = [(float(fb_logits[i]), fb_boxes_np[i]) for i in range(len(fb_boxes))]
                    
                    # Apply wheel-based proportional size bounds
                    fb_valid_size = [item for item in fb_light_boxes if (item[1][2]-item[1][0])*(item[1][3]-item[1][1]) < 0.8 * ref_wheel_area]
                    if not fb_valid_size:
                        fb_valid_size = fb_light_boxes

                    # Attempt assignment of missing headlight candidate
                    if missing_headlight:
                        fb_valid_hl = [item for item in fb_valid_size if front_wheel_cx is None or (item[1][0]+item[1][2])/2 < front_wheel_cx]
                        if fb_valid_hl:
                            fb_valid_hl.sort(key=lambda x: (x[1][1] + x[1][3]) / 2) # Prioritize higher candidate
                            final_boxes["headlight"] = fb_valid_hl[0][1]

                    # Attempt assignment of missing taillight candidate
                    if missing_taillight:
                        fb_valid_tl = [item for item in fb_valid_size if rear_wheel_cx is None or (item[1][0]+item[1][2])/2 > rear_wheel_cx]
                        if fb_valid_tl:
                            fb_valid_tl.sort(key=lambda x: x[0], reverse=True) # Prioritize rightmost candidate
                            final_boxes["taillight"] = fb_valid_tl[0][1]
            # =========================================================================

            # Reject oversized taillight false positives that approach wheel area.
            if "taillight" in final_boxes and np.isfinite(ref_wheel_area):
                taillight_area = self.get_box_area(final_boxes["taillight"])
                if taillight_area >= 0.75 * ref_wheel_area:
                    del final_boxes["taillight"]

            # =========================================================================
            # --- Rearview Mirror Localization ---
            # =========================================================================
            if len(mirror_boxes) > 0:
                # Sort candidates by confidence score in descending order
                mirror_boxes.sort(key=lambda x: x[0], reverse=True)
                # Select highest-confidence candidate
                final_boxes["rearview mirror"] = mirror_boxes[0][1]

            # =========================================================================
            # --- Lateral Doors Localization ---
            # =========================================================================
            door_candidates = []
            for caption, door_box_threshold, door_text_threshold in [
                ("front door . rear door . car front door . car rear door", 0.16, 0.14),
                ("car door . side door . vehicle door", 0.12, 0.10),
                ("door panel . side panel", 0.10, 0.08),
            ]:
                door_boxes, door_logits, _ = predict_gdino(
                    model=self.grounding_dino,
                    image=img_white_trans,
                    caption=caption,
                    box_threshold=door_box_threshold,
                    text_threshold=door_text_threshold,
                    device=self.device
                )

                if len(door_boxes) > 0:
                    door_boxes_np = self.scale_boxes_to_image(door_boxes, w, h)
                    door_candidates.extend(
                        (float(door_logits[i]), door_boxes_np[i]) for i in range(len(door_boxes_np))
                    )

            final_boxes.update(
                self.select_door_boxes(
                    door_candidates,
                    (w, h),
                    ref_wheel_area=ref_wheel_area,
                    front_wheel_box=final_boxes.get("front wheel"),
                    rear_wheel_box=final_boxes.get("rear wheel"),
                )
            )

            # =========================================================================
            # --- Door Windows Localization ---
            # =========================================================================
            window_candidates = []
            if "front door" in final_boxes or "rear door" in final_boxes:
                for caption, window_box_threshold, window_text_threshold in [
                    ("front side window . rear side window . side window", 0.14, 0.12),
                    ("door window . door glass . side glass . window glass", 0.10, 0.08),
                ]:
                    window_boxes, window_logits, _ = predict_gdino(
                        model=self.grounding_dino,
                        image=img_normal_trans,
                        caption=caption,
                        box_threshold=window_box_threshold,
                        text_threshold=window_text_threshold,
                        device=self.device
                    )

                    if len(window_boxes) > 0:
                        window_boxes_np = self.scale_boxes_to_image(window_boxes, w, h)
                        window_candidates.extend(
                            (float(window_logits[i]), window_boxes_np[i]) for i in range(len(window_boxes_np))
                        )

                front_door_box = final_boxes.get("front door")
                rear_door_box = final_boxes.get("rear door")

                if front_door_box is not None:
                    front_window_box = self.select_door_window_candidate(
                        window_candidates,
                        front_door_box,
                        (w, h),
                        other_door_box=rear_door_box,
                        min_center_x=self.get_box_center(front_door_box)[0] + FRONT_DOOR_WINDOW_MIN_CENTER_OFFSET_RATIO * (front_door_box[2] - front_door_box[0]),
                    )
                    if front_window_box is not None:
                        final_boxes["front door window"] = front_window_box
                    else:
                        fallback_front_window_box = self.build_fallback_door_window_box(
                            "front door",
                            front_door_box,
                            (w, h),
                            reference_window_box=final_boxes.get("rear door window"),
                        )
                        if fallback_front_window_box is not None:
                            final_boxes["front door window"] = fallback_front_window_box

                if rear_door_box is not None:
                    rear_window_box = self.select_door_window_candidate(
                        window_candidates,
                        rear_door_box,
                        (w, h),
                        other_door_box=front_door_box,
                        reference_window_box=final_boxes.get("front door window"),
                    )
                    if rear_window_box is not None:
                        final_boxes["rear door window"] = rear_window_box
                    else:
                        fallback_rear_window_box = self.build_fallback_door_window_box(
                            "rear door",
                            rear_door_box,
                            (w, h),
                            reference_window_box=final_boxes.get("front door window"),
                        )
                        if fallback_rear_window_box is not None:
                            final_boxes["rear door window"] = fallback_rear_window_box

            # =========================================================================
            # --- Door Handles Localization ---
            # =========================================================================
            if "front door" in final_boxes or "rear door" in final_boxes:
                # 1. Detect candidate door handles across image
                handle_boxes, handle_logits, _ = predict_gdino(
                    model=self.grounding_dino,
                    image=img_normal_trans, # Use standard image representation for fine reflections and details
                    caption="car door handle . door handle",
                    box_threshold=0.15,
                    text_threshold=0.15,
                    device=self.device
                )

                if len(handle_boxes) > 0:
                    handle_boxes_np = self.scale_boxes_to_image(handle_boxes, w, h)
                    
                    front_door_box = final_boxes.get("front door")
                    rear_door_box = final_boxes.get("rear door")
                    
                    front_handles = []
                    rear_handles = []

                    # 2. Filter handles constrained within door bounding boxes
                    for i, hbox in enumerate(handle_boxes_np):
                        logit = float(handle_logits[i])
                        h_area = self.get_box_area(hbox)
                        
                        if front_door_box is not None:
                            # Constrain handle containment within door (IoA > 0.60, area < 10% of door area).
                            if self.calculate_ioa(hbox, front_door_box) > 0.60 and h_area < 0.10 * self.get_box_area(front_door_box):
                                front_handles.append((logit, hbox))
                                
                        if rear_door_box is not None:
                            if self.calculate_ioa(hbox, rear_door_box) > 0.60 and h_area < 0.10 * self.get_box_area(rear_door_box):
                                rear_handles.append((logit, hbox))

                    # 3. Assign highest-confidence handle for each door
                    if front_handles:
                        front_handles.sort(key=lambda x: x[0], reverse=True)
                        final_boxes["front door handle"] = front_handles[0][1]
                        
                    if rear_handles:
                        rear_handles.sort(key=lambda x: x[0], reverse=True)
                        final_boxes["rear door handle"] = rear_handles[0][1]

            # =========================================================================
            # --- Fuel / Charging Port Door Localization ---
            # =========================================================================
            if "rear wheel" in final_boxes:
                fuel_door_candidates = []
                for caption, fuel_box_threshold, fuel_text_threshold in [
                    ("fuel door . fuel flap . gas door . fuel filler door . gas cap", 0.14, 0.12),
                    ("fuel lid . filler flap . petrol cap . circular fuel cap . square fuel door", 0.10, 0.08),
                ]:
                    fuel_boxes, fuel_logits, _ = predict_gdino(
                        model=self.grounding_dino,
                        image=img_normal_trans,
                        caption=caption,
                        box_threshold=fuel_box_threshold,
                        text_threshold=fuel_text_threshold,
                        device=self.device
                    )

                    if len(fuel_boxes) > 0:
                        fuel_boxes_np = self.scale_boxes_to_image(fuel_boxes, w, h)
                        fuel_door_candidates.extend(
                            (float(fuel_logits[i]), fuel_boxes_np[i]) for i in range(len(fuel_boxes_np))
                        )

                selected_fuel_door = self.select_fuel_door_candidate(
                    fuel_door_candidates,
                    (w, h),
                    ref_wheel_area=ref_wheel_area,
                    front_wheel_box=final_boxes.get("front wheel"),
                    rear_wheel_box=final_boxes.get("rear wheel"),
                    front_door_box=final_boxes.get("front door"),
                    rear_door_box=final_boxes.get("rear door"),
                    forbidden_boxes=[
                        final_boxes.get("front door window"),
                        final_boxes.get("rear door window"),
                        final_boxes.get("front door handle"),
                        final_boxes.get("rear door handle"),
                        final_boxes.get("rearview mirror"),
                        final_boxes.get("headlight"),
                        final_boxes.get("taillight"),
                    ],
                )

                if selected_fuel_door is not None:
                    final_boxes["fuel door"] = selected_fuel_door

            # =========================================================================
            # --- Bumpers Localization (Front & Rear) ---
            # =========================================================================
            for bumper_label, caption in [
                ("front bumper", "car front bumper . front bumper"),
                ("rear bumper", "car rear bumper . rear bumper"),
            ]:
                bumper_boxes, bumper_logits, _ = predict_gdino(
                    model=self.grounding_dino,
                    image=img_white_trans, # White background representation highlights body panels
                    caption=caption,
                    box_threshold=0.15,
                    text_threshold=0.15,
                    device=self.device
                )

                if len(bumper_boxes) > 0:
                    bumper_boxes_np = self.scale_boxes_to_image(bumper_boxes, w, h)
                    bumper_candidates = [
                        (float(bumper_logits[i]), bumper_boxes_np[i]) 
                        for i in range(len(bumper_boxes_np))
                    ]
                    
                    selected_bumper = self.select_bumper_candidate(
                        bumper_candidates,
                        bumper_label,
                        final_boxes,
                        (w, h)
                    )
                    
                    if selected_bumper is not None:
                        final_boxes[bumper_label] = selected_bumper

            # --- 4. SAM 2 Mask Generation & Refinement ---
            # (Component mask refinement)

            # --- 4. SAM 2 Mask Generation & Refinement ---
            for prompt, box in final_boxes.items():
                current_image = image_normal if "window" in prompt or prompt == "fuel door" else image_white
                self.sam2_predictor.set_image(np.array(current_image))
                masks, _, _ = self.sam2_predictor.predict(
                    point_coords=None, point_labels=None, box=np.array([box]), multimask_output=False
                )
                unique_masks[prompt] = torch.any(torch.tensor(masks).squeeze(1), dim=0)

            # (End of predict_parts refinement pipeline)
            unique_masks = self.refine_door_component_masks(
                unique_masks, 
                final_boxes, 
                image_normal, 
                image_white,
            )
            rear_bumper_mask = unique_masks.get("rear bumper")
            if rear_bumper_mask is None or not torch.any(rear_bumper_mask):
                fallback_rear_bumper_mask = self.build_rear_bumper_fallback_mask(image_white, final_boxes)
                if fallback_rear_bumper_mask is not None and torch.any(fallback_rear_bumper_mask):
                    print("  -> Rear bumper recovered via chromatic fallback.")
                    unique_masks["rear bumper"] = fallback_rear_bumper_mask
                    fallback_rear_bumper_box = self.get_mask_bbox(fallback_rear_bumper_mask)
                    if fallback_rear_bumper_box is not None:
                        final_boxes["rear bumper"] = fallback_rear_bumper_box
            fuel_door_box = final_boxes.get("fuel door")
            fuel_door_mask = unique_masks.get("fuel door")
            if fuel_door_box is not None and fuel_door_mask is not None:
                if self.fuel_door_mask_is_unstable(fuel_door_mask, fuel_door_box, ref_wheel_area=ref_wheel_area):
                    fallback_fuel_mask = self.build_box_mask(
                        fuel_door_box,
                        (w, h),
                        inset_ratio=FUEL_DOOR_BOX_MASK_INSET_RATIO,
                    )
                    if fallback_fuel_mask is not None:
                        unique_masks["fuel door"] = fallback_fuel_mask
                    else:
                        del unique_masks["fuel door"]
                        final_boxes.pop("fuel door", None)
            # =========================================================================
            # Discard handle candidates that overlap excessively with rearview mirror
            # =========================================================================
            mirror_mask = unique_masks.get("rearview mirror")
            if mirror_mask is not None:
                for handle_label in ["front door handle", "rear door handle"]:
                    handle_mask = unique_masks.get(handle_label)
                    overlap_ratio = self.get_mask_overlap_ratio(handle_mask, mirror_mask)
                    if overlap_ratio >= HANDLE_MIRROR_MAX_OVERLAP_RATIO:
                        print(
                            f"  -> Discarding {handle_label}: excessive overlap with rearview mirror "
                            f"{overlap_ratio:.2f} >= {HANDLE_MIRROR_MAX_OVERLAP_RATIO:.2f}"
                        )
                        del unique_masks[handle_label]
                        final_boxes.pop(handle_label, None)
            # =========================================================================
            # Subtract door handles from door panel masks for disjoint segmentation
            # =========================================================================
            for door_label, handle_label in [
                ("front door", "front door handle"),
                ("rear door", "rear door handle")
            ]:
                if door_label in unique_masks and handle_label in unique_masks:
                    door_mask = unique_masks[door_label]
                    handle_mask = unique_masks[handle_label]
                    # Retain only door panel pixels outside the handle mask
                    unique_masks[door_label] = torch.logical_and(door_mask, torch.logical_not(handle_mask))

            fuel_door_mask = unique_masks.get("fuel door")
            if fuel_door_mask is not None:
                for door_label in ["front door", "rear door"]:
                    if door_label in unique_masks:
                        unique_masks[door_label] = torch.logical_and(
                            unique_masks[door_label],
                            torch.logical_not(fuel_door_mask),
                        )

            front_bumper_mask = unique_masks.get("front bumper")
            headlight_mask = unique_masks.get("headlight")
            if front_bumper_mask is not None and headlight_mask is not None:
                unique_masks["headlight"] = torch.logical_and(
                    headlight_mask,
                    torch.logical_not(front_bumper_mask),
                )
                if not torch.any(unique_masks["headlight"]):
                    del unique_masks["headlight"]
                    final_boxes.pop("headlight", None)

        return unique_masks

# =============================================================================
# --- 4. BATCH INFERENCE ENTRYPOINT ---
# =============================================================================
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
    print("Initializing Lateral View Segmentation Pipeline (Wheels & Lights with Spatial Constraints)...")
    
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
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, DATASET_FOLDER)
    if not os.path.exists(dataset_dir) and os.path.exists(os.path.join(script_dir, "sample_data", DATASET_FOLDER)):
        dataset_dir = os.path.join(script_dir, "sample_data", DATASET_FOLDER)
    out_base_dir = os.path.join(script_dir, OUTPUT_MASKS_FOLDER)

    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.abspath(os.path.join(root, file))
                rel_path = os.path.relpath(img_path, dataset_dir)
                
                if "lateral" not in rel_path.lower(): continue

                print(f"\nProcessing: {file}")
                
                # Construct dual RGB / white canvas representations in memory
                image_raw = Image.open(img_path)
                
                if image_raw.mode == 'RGBA':
                    black_bg = Image.new('RGB', image_raw.size, (0, 0, 0))
                    black_bg.paste(image_raw, mask=image_raw.split()[3])
                    image_normal = black_bg
                    
                    white_bg = Image.new('RGB', image_raw.size, (255, 255, 255))
                    white_bg.paste(image_raw, mask=image_raw.split()[3]) 
                    image_white = white_bg
                else:
                    image_normal = image_raw.convert("RGB")
                    image_white = image_normal.copy()
                # ------------------------------

                img_name = os.path.splitext(file)[0]
                out_img_dir = os.path.join(out_base_dir, os.path.dirname(rel_path), img_name)
                os.makedirs(out_img_dir, exist_ok=True)

                masks = model.predict_parts(
                    image_normal,
                    image_white, 
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
