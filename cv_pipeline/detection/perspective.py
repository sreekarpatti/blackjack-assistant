"""Perspective warping utilities for table normalization."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import cv2
import numpy as np


def warp(frame: np.ndarray, points: Iterable[Iterable[float]] | None = None) -> np.ndarray:
    """Apply perspective warp to frame.

    Args:
        frame: Input image frame.
        points: Optional list of 4 source points [x, y].

    Returns:
        Warped frame. Identity transform if points are missing.
    """
    if points is None:
        return frame

    pts: List[Tuple[float, float]] = [(float(p[0]), float(p[1])) for p in points]
    if len(pts) != 4:
        return frame

    h, w = frame.shape[:2]
    src = np.array(pts, dtype=np.float32)
    dst = np.array([(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, matrix, (w, h))
