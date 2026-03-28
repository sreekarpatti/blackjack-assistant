"""Expected-value helpers including Kelly-style bet sizing."""

from __future__ import annotations

from typing import Dict


def kelly_bet_fraction(edge: float, bankroll_risk_unit: float = 1.0) -> float:
    """Estimate Kelly fraction for even-money outcomes.

    Args:
        edge: Estimated edge in decimal units, e.g. 0.01 for 1%.
        bankroll_risk_unit: Risk normalization denominator.

    Returns:
        Kelly fraction clamped to [0, 1].
    """
    if bankroll_risk_unit <= 0:
        return 0.0
    fraction = edge / bankroll_risk_unit
    return max(0.0, min(1.0, fraction))


def units_from_true_count(true_count: float, spread: Dict[str, int]) -> int:
    """Convert true count to bet units based on configured spread.

    Args:
        true_count: Current true count.
        spread: Mapping with keys `<=1`, `2`, `3`, `>=4`.

    Returns:
        Integer units to wager.
    """
    if true_count <= 1:
        return int(spread.get("<=1", 1))
    if true_count < 3:
        return int(spread.get("2", 2))
    if true_count < 4:
        return int(spread.get("3", 4))
    return int(spread.get(">=4", 6))
