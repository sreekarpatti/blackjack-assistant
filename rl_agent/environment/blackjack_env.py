"""Custom Gymnasium Blackjack environment with counting-aware observations."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from common.card import Card
from common.hand import Hand
from rl_agent.environment.dealer_logic import play_dealer_hand
from rl_agent.environment.shoe_simulator import ShoeSimulator

ACTION_STAND = 0
ACTION_HIT = 1
ACTION_DOUBLE = 2
ACTION_SPLIT = 3


class BlackjackEnv(gym.Env):
    """Blackjack environment with Discrete(4) action space."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        num_decks: int = 6,
        reshuffle_penetration: float = 0.75,
        hit_soft_17: bool = True,
    ) -> None:
        """Initialize environment.

        Args:
            num_decks: Number of decks in shoe.
            reshuffle_penetration: Fraction threshold to reshuffle.
            hit_soft_17: Dealer rule toggle.
        """
        super().__init__()
        self.shoe = ShoeSimulator(num_decks=num_decks, reshuffle_penetration=reshuffle_penetration)
        self.hit_soft_17 = hit_soft_17
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.array([4, 1, -10, 0, 0], dtype=np.float32),
            high=np.array([21, 10, 10, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )
        self.player_hand = Hand()
        self.dealer_hand = Hand()
        self.split_hands: List[Hand] = []
        self.split_bets: List[float] = []
        self.split_done: List[bool] = []
        self.active_split_index = 0
        self.done = False
        self.curriculum_phase = 2

    def set_curriculum_phase(self, phase: int) -> None:
        """Set curriculum phase mode.

        Args:
            phase: 1 for reduced state, 2 for full state.
        """
        self.curriculum_phase = phase

    def _deal_initial(self) -> None:
        """Deal initial two-card hands for player and dealer."""
        self.player_hand = Hand(cards=[self.shoe.draw(), self.shoe.draw()])
        self.dealer_hand = Hand(cards=[self.shoe.draw(), self.shoe.draw()])
        self.split_hands = []
        self.split_bets = []
        self.split_done = []
        self.active_split_index = 0

    def _current_player_hand(self) -> Hand:
        """Get currently active player hand.

        Returns:
            Active hand in single-hand or split-hand mode.
        """
        if self.split_hands:
            return self.split_hands[self.active_split_index]
        return self.player_hand

    def _advance_split_turn(self) -> bool:
        """Move active split index to next unfinished hand.

        Returns:
            True if a next hand exists, else False.
        """
        for index, finished in enumerate(self.split_done):
            if not finished:
                self.active_split_index = index
                return True
        return False

    def _dealer_upcard_value(self) -> int:
        """Convert dealer upcard to [1..10] numeric value.

        Returns:
            Dealer upcard numeric value.
        """
        up = self.dealer_hand.cards[0]
        if up.rank == "A":
            return 1
        if up.rank in {"10", "J", "Q", "K"}:
            return 10
        return int(up.rank)

    def _can_split(self) -> int:
        """Check split availability indicator.

        Returns:
            1 if hand is pair, else 0.
        """
        if self.split_hands:
            return 0
        return 1 if self.player_hand.is_pair() else 0

    def _obs(self) -> np.ndarray:
        """Build observation vector.

        Returns:
            Observation float vector.
        """
        obs = np.array(
            [
                self._current_player_hand().total(),
                self._dealer_upcard_value(),
                float(np.clip(self.shoe.true_count, -10, 10)),
                1 if self._current_player_hand().is_soft() else 0,
                self._can_split(),
            ],
            dtype=np.float32,
        )
        if self.curriculum_phase == 1:
            obs[2] = 0
            obs[3] = 0
            obs[4] = 0
        return obs

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset env for new episode.

        Args:
            seed: RNG seed.
            options: Optional gym reset options.

        Returns:
            Tuple of initial observation and info.
        """
        super().reset(seed=seed)
        self.done = False
        self._deal_initial()
        return self._obs(), {}

    def _resolve(self, bet: float) -> float:
        """Resolve terminal outcome.

        Args:
            bet: Bet size for this hand.

        Returns:
            Reward normalized by base unit.
        """
        player_total = self._current_player_hand().total()
        if player_total > 21:
            return -bet

        self.dealer_hand = play_dealer_hand(self.dealer_hand, self.shoe, self.hit_soft_17)
        dealer_total = self.dealer_hand.total()

        if dealer_total > 21 or player_total > dealer_total:
            return bet
        if player_total < dealer_total:
            return -bet
        return 0.0

    def _resolve_against_dealer(self, hand: Hand, bet: float, dealer_total: int) -> float:
        """Resolve one hand against an already finalized dealer total.

        Args:
            hand: Player hand.
            bet: Bet amount for hand.
            dealer_total: Final dealer total (or bust value >21).

        Returns:
            Reward contribution for this hand.
        """
        player_total = hand.total()
        if player_total > 21:
            return -bet
        if dealer_total > 21 or player_total > dealer_total:
            return bet
        if player_total < dealer_total:
            return -bet
        return 0.0

    def _finalize_split_round(self) -> float:
        """Settle all split hands versus dealer and finish episode.

        Returns:
            Total reward across both split hands.
        """
        self.dealer_hand = play_dealer_hand(self.dealer_hand, self.shoe, self.hit_soft_17)
        dealer_total = self.dealer_hand.total()
        total_reward = 0.0
        for hand, bet in zip(self.split_hands, self.split_bets):
            total_reward += self._resolve_against_dealer(hand, bet, dealer_total)
        self.done = True
        return float(total_reward)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Advance one environment transition.

        Args:
            action: One of 0..3 (stand, hit, double, split).

        Returns:
            Observation, reward, terminated, truncated, info tuple.
        """
        if self.done:
            return self._obs(), 0.0, True, False, {}

        bet = 1.0
        reward = 0.0

        # Split-hand progression mode.
        if self.split_hands:
            hand = self._current_player_hand()
            index = self.active_split_index

            if action == ACTION_DOUBLE and len(hand.cards) == 2:
                hand.add(self.shoe.draw())
                self.split_bets[index] = 2.0
                self.split_done[index] = True
            elif action == ACTION_HIT or action == ACTION_SPLIT:
                # In split mode a second split is not modeled; treat as HIT.
                hand.add(self.shoe.draw())
                if hand.total() >= 21:
                    self.split_done[index] = True
            else:  # ACTION_STAND
                self.split_done[index] = True

            if all(self.split_done):
                reward = self._finalize_split_round()
            else:
                self._advance_split_turn()

            return self._obs(), float(reward), self.done, False, {}

        if action == ACTION_HIT:
            self.player_hand.add(self.shoe.draw())
            if self.player_hand.total() >= 21:
                reward = self._resolve(bet)
                self.done = True
        elif action == ACTION_DOUBLE:
            self.player_hand.add(self.shoe.draw())
            reward = self._resolve(2.0)
            self.done = True
        elif action == ACTION_SPLIT:
            if self.player_hand.is_pair():
                left = Hand(cards=[self.player_hand.cards[0]])
                right = Hand(cards=[self.player_hand.cards[1]])
                left.add(self.shoe.draw())
                right.add(self.shoe.draw())
                self.split_hands = [left, right]
                self.split_bets = [1.0, 1.0]
                self.split_done = [False, False]
                self.active_split_index = 0
            else:
                self.player_hand.add(self.shoe.draw())
            if self.player_hand.total() >= 21:
                reward = self._resolve(bet)
                self.done = True
        else:
            reward = self._resolve(bet)
            self.done = True

        return self._obs(), float(reward), self.done, False, {}
