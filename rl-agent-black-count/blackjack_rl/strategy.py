
# pair card value -> set of dealer upcards where we split
PAIR_SPLITS = {
    11: {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},  # aces always split
    10: set(),                               # tens never split
    9:  {2, 3, 4, 5, 6, 8, 9},
    8:  {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},  # eights always split
    7:  {2, 3, 4, 5, 6, 7},
    6:  {2, 3, 4, 5, 6},
    5:  set(),                               # treat as hard 10
    4:  {5, 6},
    3:  {2, 3, 4, 5, 6, 7},
    2:  {2, 3, 4, 5, 6, 7},
}

# soft totals: player sum -> dealer upcard -> action (0=H 1=S 2=D)
SOFT_TOTALS = {
    21: {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1},
    20: {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1},
    19: {1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1},
    18: {1: 0, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 1, 8: 1, 9: 0, 10: 0},
    17: {1: 0, 2: 0, 3: 2, 4: 2, 5: 2, 6: 2, 7: 0, 8: 0, 9: 0, 10: 0},
    16: {1: 0, 2: 0, 3: 0, 4: 2, 5: 2, 6: 2, 7: 0, 8: 0, 9: 0, 10: 0},
    15: {1: 0, 2: 0, 3: 0, 4: 2, 5: 2, 6: 2, 7: 0, 8: 0, 9: 0, 10: 0},
    14: {1: 0, 2: 0, 3: 0, 4: 0, 5: 2, 6: 2, 7: 0, 8: 0, 9: 0, 10: 0},
    13: {1: 0, 2: 0, 3: 0, 4: 0, 5: 2, 6: 2, 7: 0, 8: 0, 9: 0, 10: 0},
}

# hard totals: player sum -> dealer upcard -> action
HARD_TOTALS = {
    17: {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1},
    16: {1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: 0},
    15: {1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: 0},
    14: {1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: 0},
    13: {1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: 0},
    12: {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: 0},
    11: {1: 0, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2},
    10: {1: 0, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 0},
    9:  {1: 0, 2: 0, 3: 2, 4: 2, 5: 2, 6: 2, 7: 0, 8: 0, 9: 0, 10: 0},
}


def basic_strategy(player_sum, dealer_upcard, usable_ace, can_double, can_split):
    du = int(dealer_upcard)

    # check split first
    if can_split:
        pair_card = player_sum // 2
        if usable_ace and player_sum == 12:
            pair_card = 11
        if pair_card in PAIR_SPLITS and du in PAIR_SPLITS[pair_card]:
            return 3

    # soft totals
    if usable_ace and player_sum <= 21:
        row = SOFT_TOTALS.get(player_sum)
        if row is None:
            return 1
        action = row.get(du, 0)
        if action == 2:
            return 2 if can_double else 0
        return action

    # hard totals
    if player_sum >= 17:
        return 1
    if player_sum <= 8:
        return 0

    row = HARD_TOTALS.get(player_sum)
    if row is None:
        return 0

    action = row.get(du, 0)
    if action == 2:
        return 2 if can_double else 0
    return action
