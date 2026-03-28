"""Tests for deterministic baseline strategy behavior."""

from common.card import Card
from common.hand import Hand
from rl_agent.agents.baseline_agent import BasicStrategyBaseline


def test_baseline_stands_on_hard_17() -> None:
    """Baseline should stand on hard 17 regardless of dealer upcard."""
    agent = BasicStrategyBaseline()
    hand = Hand(cards=[Card.from_label("10h"), Card.from_label("7d")])
    action = agent.act(hand, dealer_upcard_rank="A", can_double=False, can_split=False)
    assert action == 0
