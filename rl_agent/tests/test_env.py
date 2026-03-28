"""Tests for Gym API compatibility and basic reward bounds."""

from common.card import Card, card_to_hi_lo
from rl_agent.environment.blackjack_env import BlackjackEnv


def test_env_reset_and_step_shapes() -> None:
    """Environment should return valid observation shape and step tuple."""
    env = BlackjackEnv()
    obs, info = env.reset()
    assert obs.shape == (5,)
    assert isinstance(info, dict)

    obs, reward, done, truncated, info = env.step(0)
    assert obs.shape == (5,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert truncated is False
    assert isinstance(info, dict)


def test_reward_reasonable_range() -> None:
    """Reward should be bounded by double-down range in starter env."""
    env = BlackjackEnv()
    env.reset()
    _, reward, _, _, _ = env.step(2)
    assert -2.0 <= reward <= 2.0


def test_split_action_enters_split_mode() -> None:
    """Split action on a pair should create two playable split hands."""
    env = BlackjackEnv()
    env.reset()
    env.player_hand.cards = [
        Card(rank="8", suit="c", hi_lo_value=card_to_hi_lo("8")),
        Card(rank="8", suit="d", hi_lo_value=card_to_hi_lo("8")),
    ]

    obs, reward, done, truncated, info = env.step(3)

    assert len(env.split_hands) == 2
    assert env._can_split() == 0
    assert obs.shape == (5,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert truncated is False
    assert isinstance(info, dict)
