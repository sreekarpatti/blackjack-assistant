"""Unit tests for FSM transitions and zoning."""

from common.card import Card
from cv_pipeline.detection.tracker import TrackedCard
from cv_pipeline.strategy.fsm import GameState, RoundState, classify_zone, update_from_tracks


def test_classify_zone() -> None:
    """Top of frame should map to dealer zone."""
    assert classify_zone((10, 10, 40, 40), frame_height=400, dealer_zone_ratio=0.4) == "DEALER_ZONE"
    assert classify_zone((10, 260, 40, 320), frame_height=400, dealer_zone_ratio=0.4) == "PLAYER_ZONE"


def test_fsm_waiting_to_dealing() -> None:
    """Presence of any tracked card should move WAITING to DEALING."""
    state = GameState()
    track = TrackedCard(track_id=1, card=Card.from_label("7c"), bbox=(10, 300, 50, 360), confidence=0.9)
    out = update_from_tracks(state, [track], frame_shape=(400, 600, 3), dealer_zone_ratio=0.4)
    assert out.state == RoundState.DEALING


def test_fsm_split_tracks_two_player_hands() -> None:
    """Player split layout should be represented as two independent sub-hands."""
    state = GameState(state=RoundState.PLAYER_TURN)
    tracks = [
        TrackedCard(track_id=1, card=Card.from_label("8c"), bbox=(110, 280, 150, 340), confidence=0.9),
        TrackedCard(track_id=2, card=Card.from_label("8d"), bbox=(190, 280, 230, 340), confidence=0.9),
        TrackedCard(track_id=3, card=Card.from_label("3c"), bbox=(80, 300, 120, 360), confidence=0.9),
        TrackedCard(track_id=4, card=Card.from_label("Ks"), bbox=(420, 300, 470, 360), confidence=0.9),
    ]

    out = update_from_tracks(state, tracks, frame_shape=(400, 600, 3), dealer_zone_ratio=0.4)

    assert out.split_active is True
    assert len(out.player_hands) == 2
    assert len(out.player_hands[0].cards) >= 1
    assert len(out.player_hands[1].cards) >= 1
