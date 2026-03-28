"""Stable-Baselines3 callbacks for checkpointing and lightweight EV logging."""

from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback


class CheckpointEveryNSteps(BaseCallback):
    """Save checkpoints at fixed interval steps."""

    def __init__(self, save_every: int, save_dir: str, verbose: int = 0) -> None:
        """Initialize callback.

        Args:
            save_every: Step interval.
            save_dir: Destination directory.
            verbose: Verbosity level.
        """
        super().__init__(verbose=verbose)
        self.save_every = save_every
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        """Save model periodically.

        Returns:
            True to continue training.
        """
        if self.n_calls % self.save_every == 0:
            self.model.save(str(self.save_dir / f"ppo_step_{self.n_calls}"))
        return True
