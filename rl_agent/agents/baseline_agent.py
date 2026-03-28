"""Deterministic baseline agent using shared basic strategy tables."""

from __future__ import annotations

from common.hand import Hand
from common.strategy_tables import BASIC_STRATEGY


def _dealer_upcard_value(rank: str) -> int:
    """Convert upcard rank to strategy lookup value.

    Args:
        rank: Dealer upcard rank token.

    Returns:
        Integer upcard value where Ace=11 and face cards=10.
    """
    if rank == "A":
        return 11
    if rank in {"10", "J", "Q", "K"}:
        return 10
    return int(rank)


class BasicStrategyBaseline:
    """Baseline policy for benchmarking PPO agent EV."""

    def act(self, player_hand: Hand, dealer_upcard_rank: str, can_double: bool, can_split: bool) -> int:
        """Return Discrete action index from basic strategy.

        Args:
            player_hand: Current player hand.
            dealer_upcard_rank: Dealer upcard rank.
            can_double: Whether double is legal.
            can_split: Whether split is legal.

        Returns:
            Integer action where 0=STAND, 1=HIT, 2=DOUBLE, 3=SPLIT.
        """
        dealer_value = _dealer_upcard_value(dealer_upcard_rank)
        if player_hand.is_pair() and can_split:
            rank = player_hand.cards[0].rank
            pair_value = 11 if rank == "A" else (10 if rank in {"10", "J", "Q", "K"} else int(rank))
            token = BASIC_STRATEGY.get(("pair", pair_value, dealer_value), "HIT")
        elif player_hand.is_soft():
            token = BASIC_STRATEGY.get(("soft", player_hand.total(), dealer_value), "HIT")
        else:
            token = BASIC_STRATEGY.get(("hard", player_hand.total(), dealer_value), "HIT")

        if token == "DOUBLE" and not can_double:
            token = "HIT"
        if token == "SPLIT" and not can_split:
            token = "HIT"

        return {"STAND": 0, "HIT": 1, "DOUBLE": 2, "SPLIT": 3}[token]
