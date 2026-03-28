"""Basic strategy and Illustrious 18 deviation tables shared by all components."""

from typing import Dict, Tuple

# Keys are tuples describing decision context.
# hard: ("hard", player_total, dealer_upcard)
# soft: ("soft", soft_total, dealer_upcard)
# pair: ("pair", pair_rank_value, dealer_upcard)
BASIC_STRATEGY: Dict[Tuple[str, int, int], str] = {}

for dealer in range(2, 12):
    for total in range(4, 22):
        if total >= 17:
            BASIC_STRATEGY[("hard", total, dealer)] = "STAND"
        elif 13 <= total <= 16 and 2 <= dealer <= 6:
            BASIC_STRATEGY[("hard", total, dealer)] = "STAND"
        elif total == 12 and 4 <= dealer <= 6:
            BASIC_STRATEGY[("hard", total, dealer)] = "STAND"
        elif total == 11:
            BASIC_STRATEGY[("hard", total, dealer)] = "DOUBLE"
        elif total == 10 and 2 <= dealer <= 9:
            BASIC_STRATEGY[("hard", total, dealer)] = "DOUBLE"
        elif total == 9 and 3 <= dealer <= 6:
            BASIC_STRATEGY[("hard", total, dealer)] = "DOUBLE"
        else:
            BASIC_STRATEGY[("hard", total, dealer)] = "HIT"

for dealer in range(2, 12):
    for soft_total in range(13, 22):
        action = "HIT"
        if soft_total >= 19:
            action = "STAND"
        elif soft_total == 18:
            if 3 <= dealer <= 6:
                action = "DOUBLE"
            elif dealer in {2, 7, 8}:
                action = "STAND"
            else:
                action = "HIT"
        elif soft_total == 17 and 3 <= dealer <= 6:
            action = "DOUBLE"
        elif soft_total in {15, 16} and 4 <= dealer <= 6:
            action = "DOUBLE"
        elif soft_total in {13, 14} and 5 <= dealer <= 6:
            action = "DOUBLE"
        BASIC_STRATEGY[("soft", soft_total, dealer)] = action

for dealer in range(2, 12):
    for pair_value in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
        action = "HIT"
        if pair_value in {11, 8}:
            action = "SPLIT"
        elif pair_value == 9:
            action = "SPLIT" if dealer in {2, 3, 4, 5, 6, 8, 9} else "STAND"
        elif pair_value == 7:
            action = "SPLIT" if 2 <= dealer <= 7 else "HIT"
        elif pair_value == 6:
            action = "SPLIT" if 2 <= dealer <= 6 else "HIT"
        elif pair_value == 4:
            action = "SPLIT" if dealer in {5, 6} else "HIT"
        elif pair_value in {2, 3}:
            action = "SPLIT" if 2 <= dealer <= 7 else "HIT"
        elif pair_value == 5:
            action = "DOUBLE" if 2 <= dealer <= 9 else "HIT"
        elif pair_value == 10:
            action = "STAND"
        BASIC_STRATEGY[("pair", pair_value, dealer)] = action

# Format: (player_total, dealer_upcard, base_action) -> (deviation_action, tc_threshold)
ILLUSTRIOUS_18 = {
    (16, 10, "HIT"): ("STAND", 0.0),
    (15, 10, "HIT"): ("STAND", 4.0),
    (10, 10, "DOUBLE"): ("DOUBLE", 4.0),
    (12, 3, "HIT"): ("STAND", 2.0),
    (12, 2, "HIT"): ("STAND", 3.0),
    (11, 11, "DOUBLE"): ("DOUBLE", 1.0),
    (9, 2, "HIT"): ("DOUBLE", 1.0),
    (10, 11, "HIT"): ("DOUBLE", 4.0),
    (9, 7, "HIT"): ("DOUBLE", 3.0),
    (16, 9, "HIT"): ("STAND", 5.0),
    (13, 2, "HIT"): ("STAND", -1.0),
    (12, 4, "STAND"): ("HIT", 0.0),
    (12, 5, "STAND"): ("HIT", -2.0),
    (12, 6, "STAND"): ("HIT", -1.0),
    (13, 3, "HIT"): ("STAND", -2.0),
    (15, 9, "HIT"): ("STAND", 2.0),
    (15, 11, "HIT"): ("STAND", 1.0),
    (13, 11, "HIT"): ("STAND", -1.0),
}
