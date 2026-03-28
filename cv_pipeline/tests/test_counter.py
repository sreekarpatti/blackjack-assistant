"""Unit tests for Hi-Lo counting behavior."""

from common.card import Card
from cv_pipeline.strategy.counter import HiLoCounter
from cv_pipeline.strategy.shoe import ShoeState


def test_running_and_true_count_updates() -> None:
    """Running and true count should update from uncounted cards only."""
    counter = HiLoCounter(shoe=ShoeState(decks_total=6))
    low = Card(rank="2", suit="c", hi_lo_value=1)
    high = Card(rank="K", suit="h", hi_lo_value=-1)

    counter.update(low)
    counter.update(high)
    assert counter.running_count == 0
    assert counter.true_count == 0.0


def test_duplicate_card_not_recounted() -> None:
    """Card should only affect running count once."""
    counter = HiLoCounter(shoe=ShoeState(decks_total=1))
    card = Card(rank="5", suit="d", hi_lo_value=1)
    counter.update(card)
    counter.update(card)
    assert counter.running_count == 1
