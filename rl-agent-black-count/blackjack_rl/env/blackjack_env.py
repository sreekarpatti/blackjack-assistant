import gymnasium as gym
import numpy as np
import pickle
from pathlib import Path
from gymnasium import spaces


SNAPSHOT_PATH = Path(__file__).resolve().parents[2] / "snapshot_library.pkl"
ALL_BUCKETS = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
RESHUFFLE_AT = 78

BET_RAMP = {-4: 1, -3: 1, -2: 1, -1: 1, 0: 1, 1: 2, 2: 4, 3: 6, 4: 8}


def hi_lo_value(card):
    if 2 <= card <= 6:
        return 1
    elif card in (10, 11):
        return -1
    return 0


def hand_total(cards):
    total = sum(cards)
    aces = cards.count(11)
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1
    return total, aces > 0


class BlackjackEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, snapshot_library=None, use_bet_sizing=False, bucket_subset=None):
        super().__init__()

        if snapshot_library is None:
            with open(SNAPSHOT_PATH, "rb") as f:
                self._library = pickle.load(f)
        else:
            self._library = snapshot_library

        self._use_bet_sizing = use_bet_sizing
        self._buckets = list(bucket_subset) if bucket_subset is not None else list(ALL_BUCKETS)
        self._base_bet = 1.0

        low = np.array([4, 1, 0, -20, 0, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([31, 10, 1, 20, 1, 1, 3, 6, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        self._rng = np.random.default_rng()
        self._hand_queue = []
        self._current_hand_idx = 0
        self._num_splits = 0
        self._dealer_upcard = 2
        self._dealer_hole = 2
        self._shoe = []
        self._running_count = 0
        self._decks_remaining = 6.0
        self._bucket = 0
        self._pre_terminated = False
        self._pre_terminal_reward = 0.0

    def _deal_card(self):
        if len(self._shoe) == 0:
            self._reshuffle()
        card = self._shoe.pop()
        self._running_count += hi_lo_value(card)
        self._decks_remaining = len(self._shoe) / 52.0
        return card

    def _reshuffle(self):
        snap = self._library[self._bucket][
            int(self._rng.integers(0, len(self._library[self._bucket])))
        ]
        self._shoe = list(snap["remaining_cards"])
        self._running_count = int(snap["running_count"])
        self._decks_remaining = float(snap["decks_remaining"])

    @property
    def _true_count(self):
        return self._running_count / max(self._decks_remaining, 0.5)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._bucket = int(self._rng.choice(self._buckets))
        idx = int(self._rng.integers(0, len(self._library[self._bucket])))
        snap = self._library[self._bucket][idx]
        self._shoe = list(snap["remaining_cards"])
        self._running_count = int(snap["running_count"])
        self._decks_remaining = float(snap["decks_remaining"])
        self._base_bet = float(BET_RAMP[self._bucket]) if self._use_bet_sizing else 1.0

        p1 = self._deal_card()
        d1 = self._deal_card()
        p2 = self._deal_card()
        self._dealer_hole = self._deal_card()
        self._dealer_upcard = d1

        starting_hand = {
            "cards": [p1, p2],
            "bet": self._base_bet,
            "done": False,
            "is_split_aces": False,
            "reward": None,
        }
        self._hand_queue = [starting_hand]
        self._current_hand_idx = 0
        self._num_splits = 0
        self._pre_terminated = False
        self._pre_terminal_reward = 0.0

        info = {"true_count_bucket": self._bucket}

        dealer_bj = self._is_blackjack([d1, self._dealer_hole])
        if d1 in (10, 11) and dealer_bj:
            self._resolve_dealer_blackjack()
            total = sum(h["reward"] for h in self._hand_queue)
            self._pre_terminated = True
            self._pre_terminal_reward = total
            return self._get_obs(), info

        player_total, _ = hand_total([p1, p2])
        if player_total == 21:
            starting_hand["done"] = True
            bj_payout = 1.5 * self._base_bet
            starting_hand["reward"] = bj_payout
            self._pre_terminated = True
            self._pre_terminal_reward = bj_payout
            return self._get_obs(), info

        return self._get_obs(), info

    def _is_blackjack(self, cards):
        total, _ = hand_total(cards)
        return total == 21 and len(cards) == 2

    def _resolve_dealer_blackjack(self):
        for hand in self._hand_queue:
            hand["done"] = True
            total, _ = hand_total(hand["cards"])
            player_bj = total == 21 and len(hand["cards"]) == 2
            hand["reward"] = 0.0 if player_bj else -1.0 * hand["bet"]

    def _get_obs(self):
        idx = min(self._current_hand_idx, len(self._hand_queue) - 1)
        hand = self._hand_queue[idx]
        total, usable_ace = hand_total(hand["cards"])
        can_double = self._can_double(hand)
        can_split = self._can_split(hand)
        upcard_val = 1 if self._dealer_upcard == 11 else self._dealer_upcard
        return np.array([
            float(total),
            float(upcard_val),
            float(usable_ace),
            float(np.clip(self._true_count, -20, 20)),
            float(can_double),
            float(can_split),
            float(self._num_splits),
            float(np.clip(self._decks_remaining, 0, 6)),
            float(hand["is_split_aces"]),
        ], dtype=np.float32)

    def _can_double(self, hand):
        return len(hand["cards"]) == 2 and not hand["done"]

    def _can_split(self, hand):
        if hand["done"] or self._num_splits >= 3 or hand["is_split_aces"]:
            return False
        cards = hand["cards"]
        return len(cards) == 2 and cards[0] == cards[1]

    def action_masks(self):
        idx = min(self._current_hand_idx, len(self._hand_queue) - 1)
        hand = self._hand_queue[idx]
        mask = np.array([True, True, False, False], dtype=bool)
        mask[2] = self._can_double(hand)
        mask[3] = self._can_split(hand)
        return mask

    def step(self, action):
        if self._pre_terminated:
            self._pre_terminated = False
            return self._get_obs(), self._pre_terminal_reward, True, False, {
                "true_count_bucket": self._bucket
            }

        if len(self._shoe) < RESHUFFLE_AT:
            self._reshuffle()

        mask = self.action_masks()
        hand = self._hand_queue[self._current_hand_idx]
        penalty = 0.0

        if not mask[action]:
            action = 1
            penalty = -1.0

        if action == 0:
            return self._do_hit(hand, penalty)
        elif action == 1:
            return self._do_stand(hand, penalty)
        elif action == 2:
            return self._do_double(hand, penalty)
        elif action == 3:
            return self._do_split(hand, penalty)

        return self._get_obs(), penalty, False, False, {"true_count_bucket": self._bucket}

    def _do_hit(self, hand, penalty):
        card = self._deal_card()
        hand["cards"].append(card)
        total, _ = hand_total(hand["cards"])
        if total > 21:
            hand["done"] = True
            hand["reward"] = -1.0 * hand["bet"]
            return self._advance_or_finish(extra_reward=penalty)
        return self._get_obs(), penalty, False, False, {"true_count_bucket": self._bucket}

    def _do_stand(self, hand, penalty):
        hand["done"] = True
        return self._advance_or_finish(extra_reward=penalty)

    def _do_double(self, hand, penalty):
        hand["bet"] = 2.0 * self._base_bet
        card = self._deal_card()
        hand["cards"].append(card)
        total, _ = hand_total(hand["cards"])
        hand["done"] = True
        if total > 21:
            hand["reward"] = -2.0
            return self._advance_or_finish(extra_reward=penalty)
        return self._advance_or_finish(extra_reward=penalty)

    def _do_split(self, hand, penalty):
        c1, c2 = hand["cards"]
        self._num_splits += 1
        splitting_aces = (c1 == 11)

        new1 = self._deal_card()
        new2 = self._deal_card()

        hand1 = {"cards": [c1, new1], "bet": self._base_bet,
                 "done": splitting_aces, "is_split_aces": splitting_aces, "reward": None}
        hand2 = {"cards": [c2, new2], "bet": self._base_bet,
                 "done": splitting_aces, "is_split_aces": splitting_aces, "reward": None}

        self._hand_queue[self._current_hand_idx:self._current_hand_idx + 1] = [hand1, hand2]

        if splitting_aces:
            return self._advance_or_finish(extra_reward=penalty)
        return self._get_obs(), penalty, False, False, {"true_count_bucket": self._bucket}

    def _advance_or_finish(self, extra_reward=0.0):
        while (self._current_hand_idx < len(self._hand_queue) and
               self._hand_queue[self._current_hand_idx]["done"]):
            self._current_hand_idx += 1

        if self._current_hand_idx < len(self._hand_queue):
            return self._get_obs(), extra_reward, False, False, {"true_count_bucket": self._bucket}

        reward = self._resolve_dealer() + extra_reward
        return self._get_obs(), reward, True, False, {"true_count_bucket": self._bucket}

    def _resolve_dealer(self):
        dealer_cards = [self._dealer_upcard, self._dealer_hole]
        while True:
            total, usable_ace = hand_total(dealer_cards)
            if total > 17:
                break
            if total == 17 and not usable_ace:
                break
            dealer_cards.append(self._deal_card())

        dealer_total, _ = hand_total(dealer_cards)
        net_reward = 0.0

        for hand in self._hand_queue:
            if hand["reward"] is not None:
                net_reward += hand["reward"]
                continue
            player_total, _ = hand_total(hand["cards"])
            bet = hand["bet"]
            if dealer_total > 21 or player_total > dealer_total:
                net_reward += bet
            elif player_total < dealer_total:
                net_reward -= bet

        return net_reward

    def render(self):
        pass

    def close(self):
        pass
