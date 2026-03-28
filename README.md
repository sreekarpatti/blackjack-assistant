# Vision-Based Blackjack Strategic Assistant

A monorepo with two decoupled components:

- `cv_pipeline/`: real-time (video-file) card detection/tracking + Blackjack advisory overlay.
- `rl_agent/`: Gymnasium environment + PPO training pipeline for Blackjack policy learning.
- `common/`: shared data models and strategy tables.

## Quick Start

### 1. Install dependencies

```bash
make install
```

### 2. Train CV model

```bash
make train-cv
```

### 3. Train RL agent

```bash
make train-rl
```

### 4. Run tests

```bash
make test
```

### 5. Run assistant on a video file

```bash
make run SOURCE=path/to/video.mp4
```

## Architecture

The projects are intentionally decoupled. Shared objects are imported from `common`.
No component imports the other directly.

## Notes

- CV pipeline supports video file sources (`.mp4`, `.avi`, `.mov`), not webcam input.
- Annotated output is written to `cv_pipeline/output/annotated_output.mp4`.
- Strategy order is Illustrious 18 deviations first, then baseline basic strategy.
