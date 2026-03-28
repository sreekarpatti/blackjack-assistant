"""Dealer play policy implementation for Blackjack simulation."""

from __future__ import annotations

from common.hand import Hand
from rl_agent.environment.shoe_simulator import ShoeSimulator


def play_dealer_hand(hand: Hand, shoe: ShoeSimulator, hit_soft_17: bool = True) -> Hand:
    """Play out dealer hand using fixed casino rules.

    Args:
        hand: Dealer starting hand.
        shoe: Shoe simulator to draw cards from.
        hit_soft_17: Whether dealer hits soft 17.

    Returns:
        Final dealer hand.
    """
    while True:
        total = hand.total()
        soft = hand.is_soft()
        if total < 17:
            hand.add(shoe.draw())
            continue
        if total == 17 and soft and hit_soft_17:
            hand.add(shoe.draw())
            continue
        break
    return hand
