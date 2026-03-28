"""Unit tests for recommendation lookups."""

from common.card import Card
from common.hand import Hand
from cv_pipeline.strategy.advisor import suggest


def test_basic_strategy_hit_on_hard_16_vs_ace() -> None:
    """Hard 16 versus Ace defaults to hit in baseline strategy."""
    hand = Hand(cards=[Card.from_label("10h"), Card.from_label("6c")])
    dealer = Card.from_label("As")
    out = suggest(hand, dealer, true_count=-1.0, can_double=False, can_split=False, bet_spread={"<=1": 1, "2": 2, "3": 4, ">=4": 6})
    assert out["action"] == "HIT"


def test_illustrious_18_deviation_applies() -> None:
    """16 vs 10 should stand when TC threshold is met."""
    hand = Hand(cards=[Card.from_label("10h"), Card.from_label("6c")])
    dealer = Card.from_label("10s")
    out = suggest(hand, dealer, true_count=1.0, can_double=False, can_split=False, bet_spread={"<=1": 1, "2": 2, "3": 4, ">=4": 6})
    assert out["action"] == "STAND"
