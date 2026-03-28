# GitHub Copilot Agent Prompt
## Project: Vision-Based Blackjack Strategic Assistant

---

## 🎯 Project Summary

You are building a **real-time computer vision + strategy assistant for Blackjack**. The system passively observes a Blackjack table via webcam or video feed, detects and tracks cards using a fine-tuned YOLOv8 model, maintains a running Hi-Lo card count, and displays optimal strategic advice (Hit / Stand / Double / Split) along with True Count-based bet sizing recommendations.

The project has **two primary technical components, structured as separate self-contained sub-projects** within one monorepo:

1. **`cv_pipeline/`** — Computer Vision: card detection, classification, and tracking using YOLOv8 + ByteTrack + OpenCV. Runs the real-time assistant.
2. **`rl_agent/`** — Reinforcement Learning: a custom Gym environment + PPO agent that learns optimal Blackjack strategy including card counting.

These two components are **intentionally decoupled**. They do not import from each other. Shared data types (e.g., `Card`, `HandState`) live in a `common/` package that both import from.

---

## 📁 Required File Structure

Scaffold the following complete project structure from an empty folder. Create every file listed, with appropriate starter code, docstrings, and comments:

```
blackjack-assistant/
│
├── README.md                              # Project overview, setup, and usage for BOTH components
├── .gitignore                             # Python, YOLO weights, datasets, venv, RL checkpoints
├── Makefile                               # Targets: install, train-cv, train-rl, test-cv, test-rl, test, run
│
│
├── common/                                # ── SHARED PACKAGE ──────────────────────────────────────
│   ├── __init__.py                        # Shared types used by both cv_pipeline and rl_agent
│   ├── card.py                            # Card dataclass: rank, suit, hi_lo_value, is_counted flag
│   ├── hand.py                            # Hand dataclass: List[Card], total(), is_soft(), is_pair()
│   └── strategy_tables.py                 # Hard-coded Basic Strategy tables (Hard/Soft/Pair splits)
│                                          # and Illustrious 18 True Count deviation table
│
│
├── cv_pipeline/                           # ── COMPONENT 1: COMPUTER VISION ────────────────────────
│   ├── README.md                          # How to train the CV model and run the assistant
│   ├── requirements.txt                   # CV-specific deps: ultralytics, opencv, bytetracker, albumentations
│   ├── config.yaml                        # CV config: model path, video source, zone boundaries, overlay settings
│   │
│   ├── data/
│   │   ├── README.md                      # Instructions for downloading the 3 Kaggle datasets
│   │   ├── raw/                           # Downloaded datasets (gitignored)
│   │   ├── processed/                     # Merged + cleaned images with YOLO labels
│   │   └── splits/
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   │
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── dataset_prep.py                # Merge 3 Kaggle datasets, augment, output YOLO-format labels
│   │   ├── train.py                       # Fine-tune YOLOv8-Nano; saves best.pt to detection/weights/
│   │   ├── inference.py                   # Load model, detect cards in a frame → bboxes + class labels
│   │   ├── tracker.py                     # ByteTrack wrapper: stable track_id per card, occlusion handling
│   │   ├── perspective.py                 # OpenCV homography warp to top-down table view
│   │   └── utils.py                       # Label maps (52 classes), NMS helpers, bbox drawing
│   │
│   ├── strategy/
│   │   ├── __init__.py
│   │   ├── shoe.py                        # Shoe state: decks_total, cards_seen, decks_remaining
│   │   ├── counter.py                     # Hi-Lo engine: running_count, true_count, reset()
│   │   ├── fsm.py                         # FSM: WAITING→DEALING→PLAYER_TURN→DEALER_TURN→PAYOUT→WAITING
│   │   │                                  # Zone detection from bbox positions; split sub-hand tracking
│   │   ├── advisor.py                     # Illustrious 18 deviations → Basic Strategy → action + bet_units
│   │   └── ev_calculator.py               # Kelly Criterion bet sizing based on true count
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── frame_processor.py             # frame → warp → detect → track → FSM → advise → annotate
│   │   └── session.py                     # Opens video file, loops frames, writes annotated_output.mp4
│   │
│   ├── ui/
│   │   ├── __init__.py
│   │   └── overlay.py                     # HUD: bboxes, card labels, RC/TC counter, action banner, bet size
│   │
│   ├── output/                            # Annotated output videos written here (gitignored)
│   │
│   ├── tests/
│   │   ├── test_counter.py                # Hi-Lo running count and true count correctness
│   │   ├── test_advisor.py                # Basic Strategy + Illustrious 18 lookup correctness
│   │   ├── test_fsm.py                    # FSM state transitions and zone assignment
│   │   └── test_inference.py              # Smoke test: model load + single-frame detection output format
│   │
│   └── scripts/
│       ├── download_datasets.sh           # kaggle CLI script to pull all 3 datasets
│       ├── run_training.sh                # Runs detection/train.py with correct args
│       └── run_assistant.sh               # Launches pipeline/session.py on a video file
│
│
└── rl_agent/                              # ── COMPONENT 2: REINFORCEMENT LEARNING ─────────────────
    ├── README.md                          # How to train and evaluate the RL agent
    ├── requirements.txt                   # RL-specific deps: gymnasium, stable-baselines3, wandb
    ├── config.yaml                        # RL config: num_decks, reshuffle_penetration, bet_spread, PPO hyperparams
    │
    ├── environment/
    │   ├── __init__.py
    │   ├── blackjack_env.py               # Custom gymnasium.Env: BlackjackEnv
    │   │                                  # State: [player_total, dealer_upcard, true_count, is_soft, can_split]
    │   │                                  # Actions: Discrete(4) — Stand, Hit, Double, Split
    │   │                                  # Reward: chip delta normalized by bet size
    │   ├── shoe_simulator.py              # Simulates a 6-deck shoe; reshuffles at configurable penetration
    │   └── dealer_logic.py                # Deterministic dealer rules: hit soft 17, stand on hard 17+
    │
    ├── agents/
    │   ├── __init__.py
    │   ├── ppo_agent.py                   # PPO agent setup via stable-baselines3; curriculum training logic
    │   └── baseline_agent.py              # BasicStrategyBaseline: deterministic agent for benchmarking
    │
    ├── training/
    │   ├── __init__.py
    │   ├── train.py                       # Main training script: curriculum PPO for 5M timesteps
    │   │                                  # Phase 1 (1M steps): state = (player_total, dealer_upcard) only
    │   │                                  # Phase 2 (4M steps): full state including true_count
    │   ├── callbacks.py                   # SB3 EvalCallback, checkpoint saving every 500k steps, EV logging
    │   └── evaluate.py                    # Evaluate agent: win rate, avg EV vs BasicStrategyBaseline
    │
    ├── models/                            # Saved PPO checkpoints (gitignored except best_model.zip)
    │
    ├── tests/
    │   ├── test_env.py                    # Gym API compliance, state/action space shapes, reward range
    │   ├── test_shoe_simulator.py         # Deck composition correctness, reshuffle trigger
    │   └── test_baseline_agent.py         # BasicStrategyBaseline matches known Basic Strategy decisions
    │
    └── notebooks/
        ├── 01_env_exploration.ipynb       # Step through BlackjackEnv manually, inspect state/reward
        ├── 02_training_curves.ipynb       # Plot reward, EV, win rate over training timesteps
        └── 03_agent_vs_baseline.ipynb     # Head-to-head: PPO agent vs Basic Strategy over 100k hands
```

---

## 🔧 Technical Specifications

Each component has its own `requirements.txt`. Do not merge them.

### `cv_pipeline/requirements.txt`
```
ultralytics>=8.0          # YOLOv8
opencv-python>=4.8
numpy
torch>=2.0
torchvision
bytetracker                # or lap + custom ByteTrack impl
PyYAML
kaggle                     # dataset download
albumentations             # augmentation
wandb                      # experiment tracking
pytest
```

### `rl_agent/requirements.txt`
```
gymnasium>=0.29
stable-baselines3>=2.0
numpy
torch>=2.0
PyYAML
wandb
pytest
```

---

## 📐 Detailed Implementation Instructions per Module

### `common/card.py` and `common/hand.py`
- `Card`: dataclass with `rank: str`, `suit: str`, `hi_lo_value: int`, `is_counted: bool = False`
- `Hand`: dataclass wrapping `List[Card]` with methods `total() -> int`, `is_soft() -> bool`, `is_pair() -> bool`
- Both `cv_pipeline` and `rl_agent` import from `common` — **no duplication**

### `cv_pipeline/detection/dataset_prep.py`
- Download and merge three Kaggle datasets: `"Complete Playing Card Dataset"`, `"Playing Cards Object Detection"`, `"Standard 52-Card Deck Dataset"`
- Output: 52 classes (one per card: `2c, 3c, ..., Ac, 2d, ..., Kh, As`)
- Apply augmentations: random brightness/contrast, motion blur, perspective jitter, synthetic shadow overlays (to simulate casino lighting)
- Output YOLO-format `.txt` label files alongside images
- 80/10/10 train/val/test split

### `cv_pipeline/detection/train.py`
- Load `yolov8n.pt` as base
- Fine-tune on the merged dataset for 100 epochs with early stopping (patience=15)
- Save best checkpoint to `cv_pipeline/detection/weights/best.pt`
- Log mAP@50, mAP@50-95, per-class precision/recall

### `cv_pipeline/detection/tracker.py`
- Wrap ByteTrack to assign stable `track_id` to each card
- A card with a given `track_id` is only counted once in the Hi-Lo engine (use `is_counted` flag on `Card` from `common/`)
- Handle card occlusion: if a track disappears for <10 frames, hold its state; if >10 frames, mark as gone

### `cv_pipeline/strategy/counter.py`
- Implement standard **Hi-Lo** system using `hi_lo_value` from `common/card.py`:
  - Cards 2–6: +1, Cards 7–9: 0, Cards 10/J/Q/K/A: -1
- `running_count`: integer, updated live
- `true_count`: `running_count / decks_remaining` (float, rounded to nearest 0.5)
- `reset()`: called at start of new shoe

### `cv_pipeline/strategy/advisor.py`
- Input: `player_hand: Hand`, `dealer_upcard: Card`, `true_count: float`, `can_double: bool`, `can_split: bool`
- Logic order: Check Illustrious 18 deviations (from `common/strategy_tables.py`) first → then Basic Strategy table lookup
- Return: `{"action": "HIT"|"STAND"|"DOUBLE"|"SPLIT", "bet_units": int, "reasoning": str}`
- **Bet sizing**: at `true_count <= 1` → 1 unit; TC 2 → 2 units; TC 3 → 4 units; TC 4+ → 6 units (configurable in `cv_pipeline/config.yaml`)

### `cv_pipeline/strategy/fsm.py`
- States: `WAITING → DEALING → PLAYER_TURN → DEALER_TURN → PAYOUT → WAITING`
- Zone detection: cards in the top 40% of the warped frame = DEALER_ZONE; bottom 60% = PLAYER_ZONE(s)
- Split handling: when a pair is split, create two sub-hand objects and track each independently

### `cv_pipeline/pipeline/frame_processor.py`
- Single function: `process_frame(frame: np.ndarray, state: GameState) -> (annotated_frame: np.ndarray, state: GameState)`
- Internally calls: `perspective.warp()` → `inference.detect()` → `tracker.update()` → `fsm.update()` → `advisor.suggest()` → `overlay.draw()`

### `cv_pipeline/ui/overlay.py`
- Draw on the warped frame:
  - Green bounding boxes + card label (e.g., `"Kh"`) on each detected card
  - Top-left HUD: `RC: +4 | TC: +2.0`
  - Bottom-center: Large colored action banner — GREEN=HIT, RED=STAND, BLUE=DOUBLE, YELLOW=SPLIT
  - Bottom-right: `Bet: 4 units`

### `rl_agent/environment/blackjack_env.py`
- Extend `gymnasium.Env`
- Import `Card`, `Hand` from `common/`
- **State space**: `[player_total (4-21), dealer_upcard (1-10), true_count (-10 to +10), is_soft (0/1), can_split (0/1)]`
- **Action space**: `Discrete(4)` — Stand, Hit, Double, Split
- **Reward**: chip delta normalized by bet size (so +1.0 = won one unit)
- Simulate a 6-deck shoe; reshuffle at 75% penetration

### `rl_agent/agents/baseline_agent.py`
- `BasicStrategyBaseline`: deterministic agent using tables from `common/strategy_tables.py`
- Used as a performance benchmark — the PPO agent must eventually exceed its EV

### `rl_agent/training/train.py`
- Use **PPO** from `stable-baselines3`
- Train for 5M timesteps
- Curriculum: first 1M steps use only `(player_total, dealer_upcard)` state (basic strategy learning), then expand to full state including `true_count`
- Save checkpoint every 500k steps to `rl_agent/models/`
- Compare final agent EV vs. `BasicStrategyBaseline` and log to wandb

---

## 🚀 Entry Points

**CV Pipeline** — `cv_pipeline/scripts/run_assistant.sh`:
```bash
python -m cv_pipeline.pipeline.session --source path/to/video.mp4 --config cv_pipeline/config.yaml
```

**RL Training** — `rl_agent/training/train.py`:
```bash
python -m rl_agent.training.train --config rl_agent/config.yaml
```

- `--source` accepts a **video file path** (`.mp4`, `.avi`, `.mov`) for the CV pipeline
- No webcam/live feed support is needed — video file input only

---

## ✅ Additional Instructions for Copilot

1. **Every Python file must have a module-level docstring** explaining its role.
2. **Every class and function must have docstrings** with Args/Returns documented.
3. Use **type hints** throughout.
4. Each component has its **own `config.yaml`** — `cv_pipeline/config.yaml` and `rl_agent/config.yaml`. These are the sole sources of truth for their respective tunable parameters. Do not create a single shared config.
5. `common/` is a proper Python package. Both `cv_pipeline` and `rl_agent` import from it. **Never duplicate** `Card`, `Hand`, or strategy tables between the two components.
6. Each component has its own `tests/` subdirectory. Use mock data — not live model inference — for all unit tests.
7. The root `Makefile` has targets: `install-cv`, `install-rl`, `install` (both), `train-cv`, `train-rl`, `test-cv`, `test-rl`, `test` (both), `run`.
8. **Output annotated video**: `cv_pipeline/pipeline/session.py` writes processed output to `cv_pipeline/output/annotated_output.mp4` via `cv2.VideoWriter`. This is the primary deliverable for each CV run.
9. **If anything is ambiguous or requires a design decision** (e.g., which ByteTrack library to pin, how to handle multi-player zone layout, how `common/` should be installed for imports to work), output a note in the Copilot chat window explaining what was decided and why — do not silently make assumptions.
10. **Do not scaffold a web app or Flask server** — the UI is OpenCV windows and the annotated output video only.
