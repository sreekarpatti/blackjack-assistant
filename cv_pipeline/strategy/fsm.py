"""Finite state machine for Blackjack round progression and card zoning."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from common.hand import Hand
from cv_pipeline.detection.tracker import TrackedCard


class RoundState(str, Enum):
    """Round lifecycle states."""

    WAITING = "WAITING"
    DEALING = "DEALING"
    PLAYER_TURN = "PLAYER_TURN"
    DEALER_TURN = "DEALER_TURN"
    PAYOUT = "PAYOUT"


@dataclass
class GameState:
    """Mutable state shared across frame processing calls."""

    state: RoundState = RoundState.WAITING
    dealer_hand: Hand = field(default_factory=Hand)
    player_hands: List[Hand] = field(default_factory=lambda: [Hand()])
    active_hand_index: int = 0
    split_active: bool = False
    split_boundary_x: Optional[float] = None
    advisory: Dict[str, object] = field(default_factory=dict)


def _center_x(bbox: Tuple[int, int, int, int]) -> float:
    """Get x-center coordinate for a bounding box.

    Args:
        bbox: Bounding box in xyxy format.

    Returns:
        Horizontal center position in pixels.
    """
    x1, _, x2, _ = bbox
    return (x1 + x2) / 2.0


def _partition_player_tracks(
    tracks: List[TrackedCard],
    frame_width: int,
    boundary_x: Optional[float],
) -> Tuple[List[TrackedCard], List[TrackedCard], float]:
    """Partition player-zone tracks into left and right split hands.

    Args:
        tracks: Player-zone tracks.
        frame_width: Frame width in pixels.
        boundary_x: Optional existing split boundary.

    Returns:
        Tuple of left tracks, right tracks, and chosen boundary.
    """
    if not tracks:
        return [], [], boundary_x if boundary_x is not None else frame_width / 2.0

    centers = sorted(_center_x(track.bbox) for track in tracks)
    chosen_boundary = boundary_x if boundary_x is not None else centers[len(centers) // 2]
    left = [track for track in tracks if _center_x(track.bbox) <= chosen_boundary]
    right = [track for track in tracks if _center_x(track.bbox) > chosen_boundary]

    # Fallback when all cards temporarily cluster on one side.
    if not left or not right:
        sorted_tracks = sorted(tracks, key=lambda t: _center_x(t.bbox))
        pivot = max(1, len(sorted_tracks) // 2)
        left = sorted_tracks[:pivot]
        right = sorted_tracks[pivot:]
        if left and right:
            chosen_boundary = (_center_x(left[-1].bbox) + _center_x(right[0].bbox)) / 2.0
        else:
            chosen_boundary = frame_width / 2.0

    return left, right, chosen_boundary


def classify_zone(
    bbox: Tuple[int, int, int, int],
    frame_height: int,
    dealer_zone_ratio: float = 0.4,
) -> str:
    """Classify card bbox as dealer or player zone.

    Args:
        bbox: Bounding box tuple.
        frame_height: Height of warped frame.
        dealer_zone_ratio: Fraction of top frame used as dealer zone.

    Returns:
        `DEALER_ZONE` or `PLAYER_ZONE`.
    """
    _, y1, _, y2 = bbox
    center_y = (y1 + y2) / 2.0
    return "DEALER_ZONE" if center_y < frame_height * dealer_zone_ratio else "PLAYER_ZONE"


def update_from_tracks(
    game_state: GameState,
    tracks: List[TrackedCard],
    frame_shape: Tuple[int, int, int],
    dealer_zone_ratio: float = 0.4,
) -> GameState:
    """Advance FSM and hand assignment using tracked cards.

    Args:
        game_state: Current state object.
        tracks: Active track list.
        frame_shape: Frame shape tuple.
        dealer_zone_ratio: Dealer zone threshold ratio.

    Returns:
        Updated game state.
    """
    height, width = frame_shape[0], frame_shape[1]
    dealer_tracks: List[TrackedCard] = []
    player_tracks: List[TrackedCard] = []

    for track in tracks:
        zone = classify_zone(track.bbox, height, dealer_zone_ratio)
        if zone == "DEALER_ZONE":
            dealer_tracks.append(track)
        else:
            player_tracks.append(track)

    game_state.dealer_hand = Hand(cards=[track.card for track in dealer_tracks])

    if not player_tracks:
        game_state.player_hands = [Hand()]
        game_state.active_hand_index = 0
        game_state.split_active = False
        game_state.split_boundary_x = None
    elif game_state.split_active:
        left_tracks, right_tracks, boundary = _partition_player_tracks(
            tracks=player_tracks,
            frame_width=width,
            boundary_x=game_state.split_boundary_x,
        )
        game_state.split_boundary_x = boundary
        game_state.player_hands = [
            Hand(cards=[track.card for track in left_tracks]),
            Hand(cards=[track.card for track in right_tracks]),
        ]

        if game_state.player_hands[0].total() >= 21 and game_state.player_hands[1].cards:
            game_state.active_hand_index = 1
        else:
            game_state.active_hand_index = 0
    else:
        ordered_tracks = sorted(player_tracks, key=lambda t: _center_x(t.bbox))
        primary_hand = Hand(cards=[track.card for track in ordered_tracks])
        game_state.player_hands = [primary_hand]
        game_state.active_hand_index = 0

        spread = _center_x(ordered_tracks[-1].bbox) - _center_x(ordered_tracks[0].bbox)
        split_trigger = primary_hand.is_pair() and len(ordered_tracks) >= 3 and spread > (0.12 * width)
        if split_trigger:
            left_tracks, right_tracks, boundary = _partition_player_tracks(
                tracks=ordered_tracks,
                frame_width=width,
                boundary_x=None,
            )
            game_state.split_active = True
            game_state.split_boundary_x = boundary
            game_state.player_hands = [
                Hand(cards=[track.card for track in left_tracks]),
                Hand(cards=[track.card for track in right_tracks]),
            ]

    # State transition policy for observed round progression.
    if game_state.state == RoundState.WAITING and (dealer_tracks or player_tracks):
        game_state.state = RoundState.DEALING
    elif game_state.state == RoundState.DEALING and len(player_tracks) >= 2:
        game_state.state = RoundState.PLAYER_TURN
    elif game_state.state == RoundState.PLAYER_TURN and len(dealer_tracks) >= 2:
        game_state.state = RoundState.DEALER_TURN
    elif game_state.state == RoundState.DEALER_TURN and game_state.dealer_hand.total() >= 17:
        game_state.state = RoundState.PAYOUT
    elif game_state.state == RoundState.PAYOUT and not tracks:
        game_state.state = RoundState.WAITING
        game_state.dealer_hand = Hand()
        game_state.player_hands = [Hand()]
        game_state.active_hand_index = 0
        game_state.split_active = False
        game_state.split_boundary_x = None

    return game_state
