"""Model inference helpers for card detections on image frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from cv_pipeline.detection.utils import iou

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional runtime dependency
    YOLO = None  # type: ignore


@dataclass
class Detection:
    """One predicted card detection."""

    bbox: Tuple[int, int, int, int]
    class_id: int
    label: str
    confidence: float


def _containment(small: Tuple[int, int, int, int], big: Tuple[int, int, int, int]) -> float:
    """Fraction of small box's area that is inside big box."""
    x1 = max(small[0], big[0])
    y1 = max(small[1], big[1])
    x2 = min(small[2], big[2])
    y2 = min(small[3], big[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    small_area = max(1, (small[2] - small[0]) * (small[3] - small[1]))
    return inter / small_area


def _nms(detections: List[Detection], iou_threshold: float = 0.5) -> List[Detection]:
    """Apply non-maximum suppression — only suppress near-duplicates."""
    if not detections:
        return []
    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    keep: List[Detection] = []
    for det in sorted_dets:
        suppressed = False
        for kept in keep:
            # Only suppress if boxes are nearly identical (same card detected twice)
            if iou(det.bbox, kept.bbox) > iou_threshold:
                suppressed = True
                break
            # Only suppress if one box is almost entirely inside another
            if _containment(det.bbox, kept.bbox) > 0.8:
                suppressed = True
                break
        if not suppressed:
            keep.append(det)
    return keep


class TwoStageDetector:
    """Two-stage: table detector finds cards, classifier identifies rank+suit."""

    def __init__(self, table_detector_path: str, card_classifier_path: str) -> None:
        self.table_detector = YOLO(table_detector_path) if YOLO and Path(table_detector_path).exists() else None
        self.card_classifier = YOLO(card_classifier_path) if YOLO and Path(card_classifier_path).exists() else None

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.25) -> List[Detection]:
        if self.table_detector is None or self.card_classifier is None:
            return []

        # Stage 1: find card regions on the table
        table_results = self.table_detector.predict(frame, conf=conf_threshold, verbose=False)
        detections: List[Detection] = []

        for result in table_results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                x1, y1, x2, y2 = xyxy

                # Pad the crop slightly
                h, w = frame.shape[:2]
                pad_x = int((x2 - x1) * 0.1)
                pad_y = int((y2 - y1) * 0.1)
                cx1 = max(0, x1 - pad_x)
                cy1 = max(0, y1 - pad_y)
                cx2 = min(w, x2 + pad_x)
                cy2 = min(h, y2 + pad_y)
                crop = frame[cy1:cy2, cx1:cx2]

                if crop.size == 0:
                    continue

                # Filter by aspect ratio — real cards are roughly 1.2-1.8 tall:wide
                crop_h, crop_w = crop.shape[:2]
                if crop_w < 20 or crop_h < 20:
                    continue
                aspect = max(crop_h, crop_w) / max(1, min(crop_h, crop_w))
                if aspect > 3.0:
                    continue

                # Stage 2: classify the cropped card
                cls_results = self.card_classifier.predict(crop, verbose=False)
                probs = cls_results[0].probs
                top1_idx = probs.top1
                top1_conf = float(probs.top1conf)

                # Reject low-confidence classifications
                if top1_conf < 0.4:
                    continue

                label = self.card_classifier.names[top1_idx]

                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        class_id=top1_idx,
                        label=label,
                        confidence=top1_conf,
                    )
                )

        return _nms(detections, iou_threshold=0.4)
