"""Multi-deck shoe simulation with penetration-based reshuffling."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List

from common.card import Card, card_to_hi_lo

RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
SUITS = ["c", "d", "h", "s"]


@dataclass
class ShoeSimulator:
    """Represents a finite multi-deck shoe with running count tracking."""

    num_decks: int = 6
    reshuffle_penetration: float = 0.75
    cards: List[Card] = field(default_factory=list)
    running_count: int = 0
    cards_dealt: int = 0

    def __post_init__(self) -> None:
        """Initialize and shuffle shoe on creation."""
        self.reshuffle()

    def reshuffle(self) -> None:
        """Create a fresh shuffled shoe and reset counters."""
        self.cards = [
            Card(rank=r, suit=s, hi_lo_value=card_to_hi_lo(r))
            for _ in range(self.num_decks)
            for s in SUITS
            for r in RANKS
        ]
        random.shuffle(self.cards)
        self.running_count = 0
        self.cards_dealt = 0

    @property
    def penetration(self) -> float:
        """Compute current shoe penetration ratio.

        Returns:
            Fraction of dealt cards in [0, 1].
        """
        total = self.num_decks * 52
        return self.cards_dealt / total if total > 0 else 1.0

    @property
    def true_count(self) -> float:
        """Compute Hi-Lo true count from current shoe state.

        Returns:
            Rounded true count to nearest 0.5.
        """
        remaining_decks = max((self.num_decks * 52 - self.cards_dealt) / 52.0, 0.25)
        return round((self.running_count / remaining_decks) * 2) / 2

    def draw(self) -> Card:
        """Draw one card, reshuffling automatically near cutoff.

        Returns:
            Dealt card.
        """
        if not self.cards or self.penetration >= self.reshuffle_penetration:
            self.reshuffle()
        card = self.cards.pop()
        self.running_count += card.hi_lo_value
        self.cards_dealt += 1
        return card
