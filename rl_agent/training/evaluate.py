"""Evaluation utilities for PPO agent versus baseline strategy agent."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from stable_baselines3 import PPO

from common.card import Card
from common.hand import Hand
from rl_agent.agents.baseline_agent import BasicStrategyBaseline
from rl_agent.environment.blackjack_env import BlackjackEnv


def evaluate_model(model_path: str, config_path: str, episodes: int = 1000) -> dict:
    """Evaluate trained model and baseline over fixed episodes.

    Args:
        model_path: Path to saved PPO model zip.
        config_path: Path to RL config YAML.
        episodes: Number of episodes for each policy.

    Returns:
        Metrics dictionary.
    """
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    env_cfg = cfg["environment"]

    env = BlackjackEnv(
        num_decks=int(env_cfg.get("num_decks", 6)),
        reshuffle_penetration=float(env_cfg.get("reshuffle_penetration", 0.75)),
        hit_soft_17=bool(env_cfg.get("hit_soft_17", True)),
    )

    model = PPO.load(model_path, env=env)
    baseline = BasicStrategyBaseline()

    ppo_rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(int(action))
            total += reward
        ppo_rewards.append(total)

    baseline_rewards = []
    for _ in range(episodes):
        _, _ = env.reset()
        player = env.player_hand
        dealer_rank = env.dealer_hand.cards[0].rank
        action = baseline.act(player, dealer_rank, can_double=True, can_split=player.is_pair())
        _, reward, _, _, _ = env.step(action)
        baseline_rewards.append(reward)

    return {
        "ppo_avg_ev": sum(ppo_rewards) / max(1, len(ppo_rewards)),
        "baseline_avg_ev": sum(baseline_rewards) / max(1, len(baseline_rewards)),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description="Evaluate PPO model vs baseline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    metrics = evaluate_model(args.model, args.config, args.episodes)
    print(metrics)


if __name__ == "__main__":
    main()
