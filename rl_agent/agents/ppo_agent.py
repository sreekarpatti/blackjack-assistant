"""PPO policy construction helpers for Blackjack environment."""

from __future__ import annotations

from typing import Optional

from stable_baselines3 import PPO


def build_ppo_agent(env, learning_rate: float = 3e-4, n_steps: int = 2048, batch_size: int = 256) -> PPO:
    """Construct PPO model.

    Args:
        env: Gym-compatible environment.
        learning_rate: PPO learning rate.
        n_steps: Rollout horizon.
        batch_size: SGD minibatch size.

    Returns:
        PPO model instance.
    """
    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        verbose=1,
    )


def load_or_build(env, model_path: Optional[str] = None, **kwargs) -> PPO:
    """Load a model from disk or create a new one.

    Args:
        env: Environment to attach.
        model_path: Optional zip checkpoint path.
        **kwargs: PPO constructor settings when building.

    Returns:
        PPO model.
    """
    if model_path:
        return PPO.load(model_path, env=env)
    return build_ppo_agent(env, **kwargs)
