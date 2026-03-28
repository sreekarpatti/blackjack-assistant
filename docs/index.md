# Blackjack Assistant Documentation

This repository contains two decoupled components that share core card and strategy types.

## Guides

- CV pipeline guide: ../cv_pipeline/README.md
- RL agent guide: ../rl_agent/README.md
- Top-level setup and usage: ../README.md

## Components

- common/: shared dataclasses and strategy tables
- cv_pipeline/: detection, tracking, counting, advice, and video annotation
- rl_agent/: environment, PPO training pipeline, and baseline benchmarking

## Notes

- The CV pipeline accepts video file input and writes annotated output videos.
- The RL component is intentionally decoupled from cv_pipeline and imports shared logic from common only.
