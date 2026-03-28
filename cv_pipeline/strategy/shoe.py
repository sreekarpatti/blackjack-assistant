"""Shoe state tracking for running and true-count calculations."""

from dataclasses import dataclass


@dataclass
class ShoeState:
    """Represents a multi-deck shoe progress."""

    decks_total: int = 6
    cards_seen: int = 0

    @property
    def decks_remaining(self) -> float:
        """Estimate decks remaining in the shoe.

        Returns:
            Remaining deck count clamped to minimum 0.25.
        """
        remaining_cards = max(self.decks_total * 52 - self.cards_seen, 0)
        return max(remaining_cards / 52.0, 0.25)

    def observe_card(self, n_cards: int = 1) -> None:
        """Mark card observations as consumed from shoe.

        Args:
            n_cards: Number of cards to count as seen.
        """
        self.cards_seen += n_cards

    def reset(self) -> None:
        """Reset shoe usage counters."""
        self.cards_seen = 0
