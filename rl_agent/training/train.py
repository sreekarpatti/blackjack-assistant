"""Main PPO training script with two-phase curriculum learning."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from rl_agent.agents.ppo_agent import build_ppo_agent
from rl_agent.environment.blackjack_env import BlackjackEnv
from rl_agent.training.callbacks import CheckpointEveryNSteps


def train(config_path: str) -> str:
    """Run curriculum PPO training and save final model.

    Args:
        config_path: Path to rl_agent config file.

    Returns:
        Path to final saved model zip.
    """
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    env_cfg = cfg["environment"]
    tr_cfg = cfg["training"]

    env = BlackjackEnv(
        num_decks=int(env_cfg.get("num_decks", 6)),
        reshuffle_penetration=float(env_cfg.get("reshuffle_penetration", 0.75)),
        hit_soft_17=bool(env_cfg.get("hit_soft_17", True)),
    )

    model = build_ppo_agent(
        env,
        learning_rate=float(tr_cfg.get("learning_rate", 3e-4)),
        n_steps=int(tr_cfg.get("n_steps", 2048)),
        batch_size=int(tr_cfg.get("batch_size", 256)),
    )

    checkpoint_cb = CheckpointEveryNSteps(
        save_every=int(tr_cfg.get("checkpoint_every", 500_000)),
        save_dir="rl_agent/models",
    )

    phase1 = int(tr_cfg.get("curriculum_phase1_steps", 1_000_000))
    total = int(tr_cfg.get("total_timesteps", 5_000_000))
    phase2 = max(0, total - phase1)

    env.set_curriculum_phase(1)
    model.learn(total_timesteps=phase1, callback=checkpoint_cb)

    env.set_curriculum_phase(2)
    if phase2 > 0:
        model.learn(total_timesteps=phase2, callback=checkpoint_cb, reset_num_timesteps=False)

    out = "rl_agent/models/best_model.zip"
    model.save(out)
    return out


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train PPO blackjack agent")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    out = train(args.config)
    print(f"Saved model: {out}")


if __name__ == "__main__":
    main()
