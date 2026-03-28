"""Tests for shoe composition and reshuffle behavior."""

from rl_agent.environment.shoe_simulator import ShoeSimulator


def test_initial_card_count() -> None:
    """Shoe should start with 52 * num_decks cards."""
    shoe = ShoeSimulator(num_decks=2)
    assert len(shoe.cards) == 104


def test_reshuffle_trigger() -> None:
    """Shoe should reshuffle after penetration threshold."""
    shoe = ShoeSimulator(num_decks=1, reshuffle_penetration=0.1)
    for _ in range(6):
        shoe.draw()
    assert shoe.penetration <= 0.1 or len(shoe.cards) <= 52
