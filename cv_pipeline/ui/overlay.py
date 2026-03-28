"""Overlay drawing for detections, counts, and recommendation banners."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import cv2
import numpy as np

from cv_pipeline.detection.tracker import TrackedCard

ACTION_COLORS = {
    "HIT": (0, 180, 0),
    "STAND": (0, 0, 220),
    "DOUBLE": (220, 0, 0),
    "SPLIT": (0, 220, 220),
}


def draw(
    frame: np.ndarray,
    tracks: Iterable[TrackedCard],
    running_count: int,
    true_count: float,
    advisory: Dict[str, object],
) -> np.ndarray:
    """Render HUD, card boxes, recommendation, and bet units.

    Args:
        frame: Input BGR frame.
        tracks: Active tracked cards.
        running_count: Hi-Lo running count.
        true_count: Hi-Lo true count.
        advisory: Action recommendation payload.

    Returns:
        Annotated output frame.
    """
    out = frame.copy()
    for track in tracks:
        x1, y1, x2, y2 = track.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(
            out,
            f"#{track.track_id} {track.card.label}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        out,
        f"RC: {running_count:+d} | TC: {true_count:+.1f}",
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    action = str(advisory.get("action", "WAIT"))
    color = ACTION_COLORS.get(action, (180, 180, 180))
    h, w = out.shape[:2]
    banner_w, banner_h = 300, 56
    x = (w - banner_w) // 2
    y = h - banner_h - 20
    cv2.rectangle(out, (x, y), (x + banner_w, y + banner_h), color, -1)
    cv2.putText(
        out,
        action,
        (x + 20, y + 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )

    bet_units = advisory.get("bet_units", 1)
    cv2.putText(
        out,
        f"Bet: {bet_units} units",
        (w - 220, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out
