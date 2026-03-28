"""Card model and utilities shared by CV and RL components."""

from dataclasses import dataclass

RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
SUITS = ["c", "d", "h", "s"]


def card_to_hi_lo(rank: str) -> int:
    """Map card rank to Hi-Lo counting value.

    Args:
        rank: Card rank string, e.g. "2", "10", "K", "A".

    Returns:
        Hi-Lo value: +1 for 2-6, 0 for 7-9, -1 for 10/A face cards.
    """
    if rank in {"2", "3", "4", "5", "6"}:
        return 1
    if rank in {"7", "8", "9"}:
        return 0
    return -1


@dataclass
class Card:
    """Representation of a single playing card.

    Attributes:
        rank: Rank string.
        suit: Suit string in {c,d,h,s}.
        hi_lo_value: Hi-Lo count contribution.
        is_counted: Whether this physical card observation already affected the running count.
    """

    rank: str
    suit: str
    hi_lo_value: int
    is_counted: bool = False

    @property
    def label(self) -> str:
        """Compact card label used by detector classes and overlays.

        Returns:
            Two/three-char label, e.g. "2c", "10h", "As".
        """
        return f"{self.rank}{self.suit}"

    @classmethod
    def from_label(cls, label: str) -> "Card":
        """Construct a card from detector label text.

        Args:
            label: Card label like "Kh" or "10d".

        Returns:
            Card instance with computed Hi-Lo value.
        """
        rank, suit = label[:-1], label[-1]
        return cls(rank=rank, suit=suit, hi_lo_value=card_to_hi_lo(rank))
