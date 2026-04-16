"""Smoke test for detection output schema without requiring model weights."""

import numpy as np

from cv_pipeline.detection.inference import TwoStageDetector


def test_detector_returns_list_of_dicts() -> None:
    """Detector should return a list even if model is unavailable."""
    detector = TwoStageDetector("missing.pt", "missing.pt")
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    out = detector.detect(frame)
    assert isinstance(out, list)
