"""Hi-Lo counter engine with running and true count outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Set

from common.card import Card
from cv_pipeline.strategy.shoe import ShoeState


@dataclass
class HiLoCounter:
    """Maintain running and true counts for the current shoe."""

    shoe: ShoeState
    running_count: int = 0
    _counted_track_ids: Set[int] = field(default_factory=set)

    def update_track(self, track_id: int, card: Card) -> None:
        """Apply a tracked card to running count if this track hasn't been counted.

        Args:
            track_id: Unique track identifier.
            card: Card observation.
        """
        if track_id in self._counted_track_ids:
            return
        self._counted_track_ids.add(track_id)
        self.running_count += card.hi_lo_value
        self.shoe.observe_card(1)

    def update(self, card: Card) -> None:
        """Legacy update method for backwards compatibility.

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
        self._counted_track_ids.clear()
        self.shoe.reset()
