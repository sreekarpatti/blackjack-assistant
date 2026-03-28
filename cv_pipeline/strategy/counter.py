"""Hi-Lo counter engine with running and true count outputs."""

from __future__ import annotations

from dataclasses import dataclass

from common.card import Card
from cv_pipeline.strategy.shoe import ShoeState


@dataclass
class HiLoCounter:
    """Maintain running and true counts for the current shoe."""

    shoe: ShoeState
    running_count: int = 0

    def update(self, card: Card) -> None:
        """Apply a card to running count if not counted yet.

        Args:
            card: Card observation.
        """
        if card.is_counted:
            return
        self.running_count += card.hi_lo_value
        card.is_counted = True
        self.shoe.observe_card(1)

    @property
    def true_count(self) -> float:
        """Compute rounded true count.

        Returns:
            Running count divided by decks remaining, rounded to nearest 0.5.
        """
        raw = self.running_count / self.shoe.decks_remaining
        return round(raw * 2) / 2

    def reset(self) -> None:
        """Reset running count and shoe state for new shoe."""
        self.running_count = 0
        self.shoe.reset()
