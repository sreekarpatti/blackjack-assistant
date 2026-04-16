"""Frame processing pipeline from raw frame to annotated advisory output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import yaml

from cv_pipeline.detection.inference import TwoStageDetector
from cv_pipeline.detection.perspective import warp
from cv_pipeline.detection.tracker import ByteTrackWrapper
from cv_pipeline.strategy import advisor
from cv_pipeline.strategy.counter import HiLoCounter
from cv_pipeline.strategy.fsm import GameState, RoundState, update_from_tracks
from cv_pipeline.strategy.shoe import ShoeState
from cv_pipeline.ui.overlay import draw


@dataclass
class RuntimeContext:
    """Runtime singletons for process_frame usage."""

    detector: TwoStageDetector
    tracker: ByteTrackWrapper
    counter: HiLoCounter
    config: Dict[str, object]


_RUNTIME: RuntimeContext | None = None


def initialize_runtime(config_path: str) -> None:
    """Initialize global runtime context.

    Args:
        config_path: Path to cv_pipeline config.
    """
    global _RUNTIME
    try:
        cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed config file {config_path}: {exc}") from exc
    counter = HiLoCounter(shoe=ShoeState(decks_total=int(cfg["counting"].get("decks_total", 6))))
    _RUNTIME = RuntimeContext(
        detector=TwoStageDetector(
            table_detector_path=str(cfg.get("table_detector_path", "")),
            card_classifier_path=str(cfg.get("card_classifier_path", "")),
        ),
        tracker=ByteTrackWrapper(
            max_missed_frames=int(cfg["tracking"].get("max_missed_frames", 30)),
            iou_match_threshold=float(cfg["tracking"].get("iou_match_threshold", 0.2)),
        ),
        counter=counter,
        config=cfg,
    )


def process_frame(frame: np.ndarray, state: GameState) -> Tuple[np.ndarray, GameState]:
    """Process one frame and produce annotated output.

    Internal order: warp -> detect -> track -> FSM -> advise -> overlay.

    Args:
        frame: Input frame.
        state: Mutable game state.

    Returns:
        Tuple of annotated frame and updated state.

    Raises:
        RuntimeError: If runtime context was not initialized.
    """
    if _RUNTIME is None:
        raise RuntimeError("Call initialize_runtime(config_path) before process_frame().")

    cfg = _RUNTIME.config
    warped = warp(frame, cfg.get("table", {}).get("warp_points"))
    detections = _RUNTIME.detector.detect(warped, conf_threshold=0.35)
    tracks = _RUNTIME.tracker.update(detections)

    for track in tracks:
        if track.seen_frames >= 10:
            _RUNTIME.counter.update_track(track.track_id, track.card)

    state = update_from_tracks(
        state,
        tracks,
        warped.shape,
        dealer_zone_ratio=float(cfg.get("table", {}).get("dealer_zone_ratio", 0.4)),
    )

    if state.state == RoundState.PLAYER_TURN and state.dealer_hand.cards and state.player_hands:
        state.advisory = advisor.suggest(
            player_hand=state.player_hands[state.active_hand_index],
            dealer_upcard=state.dealer_hand.cards[0],
            true_count=_RUNTIME.counter.true_count,
            can_double=True,
            can_split=True,
            bet_spread=cfg.get("counting", {}).get("bet_spread", {}),
        )

    annotated = draw(
        warped,
        tracks=tracks,
        running_count=_RUNTIME.counter.running_count,
        true_count=_RUNTIME.counter.true_count,
        advisory=state.advisory,
    )
    return annotated, state
