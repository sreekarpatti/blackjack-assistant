"""Utility helpers for detection labels, bounding boxes, and rendering."""

from typing import List, Sequence, Tuple

import cv2
import numpy as np

SUITS = ["c", "d", "h", "s"]
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
CARD_CLASSES = [f"{rank}{suit}" for suit in SUITS for rank in RANKS]
CLASS_TO_ID = {label: idx for idx, label in enumerate(CARD_CLASSES)}
ID_TO_CLASS = {idx: label for label, idx in CLASS_TO_ID.items()}


def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute intersection over union for two XYXY boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def draw_detections(
    frame: np.ndarray,
    detections: List[Tuple[Tuple[int, int, int, int], str, float]],
) -> np.ndarray:
    """Draw card detections on an image frame.

    Args:
        frame: Image frame in BGR.
        detections: List of bbox, label, confidence tuples.

    Returns:
        Annotated frame.
    """
    output = frame.copy()
    for (x1, y1, x2, y2), label, conf in detections:
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(
            output,
            f"{label} {conf:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return output
