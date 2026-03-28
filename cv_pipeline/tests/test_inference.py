"""Smoke test for detection output schema without requiring model weights."""

import numpy as np

from cv_pipeline.detection.inference import CardDetector


def test_detector_returns_list_of_dicts() -> None:
    """Detector should return a list even if model is unavailable."""
    detector = CardDetector("cv_pipeline/detection/weights/missing.pt")
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    out = detector.detect_dicts(frame)
    assert isinstance(out, list)
