# RL Agent

This component trains a PPO Blackjack agent in a custom Gymnasium environment.

## Install

```bash
pip install -r rl_agent/requirements.txt
```

## Train

```bash
python -m rl_agent.training.train --config rl_agent/config.yaml
```

## Evaluate

```bash
python -m rl_agent.training.evaluate --config rl_agent/config.yaml --model rl_agent/models/best_model.zip
```

## Notes

- Environment state includes true count and split/double affordances.
- Curriculum training begins with a reduced state and then transitions to full state.
