"""Model inference helpers for card detections on image frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from cv_pipeline.detection.utils import ID_TO_CLASS

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


class CardDetector:
    """Thin YOLO wrapper returning normalized detection objects."""

    def __init__(self, model_path: str) -> None:
        """Create detector with optional YOLO backend.

        Args:
            model_path: Path to a .pt weight file.
        """
        self.model_path = model_path
        self.model = YOLO(model_path) if YOLO is not None and Path(model_path).exists() else None

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.25) -> List[Detection]:
        """Run card detection on one frame.

        Args:
            frame: Input BGR frame.
            conf_threshold: Minimum confidence.

        Returns:
            List of detections.
        """
        if self.model is None:
            return []

        results = self.model.predict(frame, conf=conf_threshold, verbose=False)
        detections: List[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                label = ID_TO_CLASS.get(cls_id, str(cls_id))
                detections.append(
                    Detection(
                        bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                        class_id=cls_id,
                        label=label,
                        confidence=conf,
                    )
                )
        return detections

    def detect_dicts(self, frame: np.ndarray) -> List[Dict[str, object]]:
        """Return detections as JSON-friendly dictionaries.

        Args:
            frame: Input frame.

        Returns:
            Detection dictionaries.
        """
        return [d.__dict__ for d in self.detect(frame)]
