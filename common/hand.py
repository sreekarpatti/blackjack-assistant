"""Blackjack hand model with utility methods."""

from dataclasses import dataclass, field
from typing import List

from common.card import Card


@dataclass
class Hand:
    """A Blackjack hand composed of card objects."""

    cards: List[Card] = field(default_factory=list)

    def add(self, card: Card) -> None:
        """Append a card to the hand.

        Args:
            card: Card to append.
        """
        self.cards.append(card)

    def total(self) -> int:
        """Compute Blackjack hand total with Ace soft/hard handling.

        Returns:
            Best non-busting total if available, otherwise minimum bust total.
        """
        total = 0
        aces = 0
        for card in self.cards:
            if card.rank == "A":
                aces += 1
                total += 11
            elif card.rank in {"K", "Q", "J", "10"}:
                total += 10
            else:
                total += int(card.rank)

        while total > 21 and aces > 0:
            total -= 10
            aces -= 1
        return total

    def is_soft(self) -> bool:
        """Check whether the hand is currently soft.

        Returns:
            True when one Ace is counted as 11 in the best total.
        """
        total = 0
        aces = 0
        for card in self.cards:
            if card.rank == "A":
                aces += 1
                total += 11
            elif card.rank in {"K", "Q", "J", "10"}:
                total += 10
            else:
                total += int(card.rank)
        return aces > 0 and total <= 21

    def is_pair(self) -> bool:
        """Check whether first two cards form a pair by rank value.

        Returns:
            True if exactly two cards with same Blackjack rank value.
        """
        if len(self.cards) != 2:
            return False

        def rank_value(rank: str) -> str:
            return "10" if rank in {"10", "J", "Q", "K"} else rank

        return rank_value(self.cards[0].rank) == rank_value(self.cards[1].rank)
