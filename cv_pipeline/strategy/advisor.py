"""Action recommendation engine using deviations then basic strategy."""

from __future__ import annotations

from typing import Dict, Tuple

from common.card import Card
from common.hand import Hand
from common.strategy_tables import BASIC_STRATEGY, ILLUSTRIOUS_18
from cv_pipeline.strategy.ev_calculator import units_from_true_count


def upcard_value(card: Card) -> int:
    """Convert dealer upcard rank to strategy lookup value.

    Args:
        card: Dealer upcard.

    Returns:
        2-11 value where Ace is 11.
    """
    if card.rank == "A":
        return 11
    if card.rank in {"10", "J", "Q", "K"}:
        return 10
    return int(card.rank)


def pair_value(hand: Hand) -> int:
    """Resolve pair rank value from a two-card hand.

    Args:
        hand: Player hand.

    Returns:
        Pair rank in [2..11].
    """
    rank = hand.cards[0].rank
    if rank == "A":
        return 11
    if rank in {"10", "J", "Q", "K"}:
        return 10
    return int(rank)


def _lookup_basic_action(hand: Hand, dealer: int, can_double: bool, can_split: bool) -> str:
    """Lookup baseline strategy action for current context.

    Args:
        hand: Player hand.
        dealer: Dealer upcard numeric value.
        can_double: Whether double is legal.
        can_split: Whether split is legal.

    Returns:
        Strategy action token.
    """
    if hand.is_pair() and can_split:
        action = BASIC_STRATEGY.get(("pair", pair_value(hand), dealer), "HIT")
    elif hand.is_soft() and hand.total() <= 21:
        action = BASIC_STRATEGY.get(("soft", hand.total(), dealer), "HIT")
    else:
        action = BASIC_STRATEGY.get(("hard", hand.total(), dealer), "HIT")

    if action == "DOUBLE" and not can_double:
        return "HIT"
    if action == "SPLIT" and not can_split:
        return "HIT"
    return action


def suggest(
    player_hand: Hand,
    dealer_upcard: Card,
    true_count: float,
    can_double: bool,
    can_split: bool,
    bet_spread: Dict[str, int],
) -> Dict[str, object]:
    """Return strategic action and bet sizing.

    Args:
        player_hand: Current player hand.
        dealer_upcard: Dealer visible upcard.
        true_count: Current true count.
        can_double: Rule-based availability.
        can_split: Rule-based availability.
        bet_spread: Configured TC -> units mapping.

    Returns:
        Dict with action, bet_units, and reasoning text.
    """
    dealer = upcard_value(dealer_upcard)
    base_action = _lookup_basic_action(player_hand, dealer, can_double, can_split)

    key: Tuple[int, int, str] = (player_hand.total(), dealer, base_action)
    final_action = base_action
    reasoning = "Basic Strategy"
    if key in ILLUSTRIOUS_18:
        deviation_action, tc_threshold = ILLUSTRIOUS_18[key]
        if true_count >= tc_threshold:
            final_action = deviation_action
            reasoning = f"Illustrious 18 deviation at TC >= {tc_threshold}"

    return {
        "action": final_action,
        "bet_units": units_from_true_count(true_count, bet_spread),
        "reasoning": reasoning,
    }
