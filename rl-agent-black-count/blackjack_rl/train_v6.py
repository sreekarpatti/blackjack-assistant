import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from blackjack_rl.env.blackjack_env import BlackjackEnv
from blackjack_rl.agent.q_agent import QLearningAgent

total_eps = 20_000_000
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    env = BlackjackEnv(use_bet_sizing=True, bucket_subset=[1, 2, 3, 4])
    agent = QLearningAgent(
        alpha=0.05,
        epsilon_start=0.50,
        epsilon_end=0.01,
        epsilon_decay=0.9999996,
    )

    print("V6 TD Q-Learning — clean scratch from Q=0")
    print(f"Episodes: {total_eps:,}  |  Buckets: TC+1 to TC+4  |  Scaled rewards")
    print(f"Epsilon: {agent.epsilon:.2f} -> {agent.epsilon_end:.2f}  |  alpha={agent.alpha}")

    running_rewards = []
    log_every = 500_000
    window = 10_000
    applied = 0
    skipped = 0

    for ep in range(1, total_eps + 1):
        state, info = env.reset()
        done = False
        hand_result = 0.0

        while not done:
            mask = env.action_masks()
            action = agent.select_action(state, mask, phase=2)
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            next_mask = env.action_masks() if not done else np.ones(4, dtype=bool)

            bucket = int(np.clip(round(float(state[3])), -4, 4))
            if bucket >= 1:
                agent.update(state, action, reward, next_state, done, next_mask)
                applied += 1
            else:
                skipped += 1

            state = next_state
            hand_result += reward

        agent.decay_epsilon()
        running_rewards.append(hand_result)
        if len(running_rewards) > window:
            running_rewards.pop(0)

        if ep % log_every == 0:
            avg = np.mean(running_rewards)
            print(f"Episode {ep:>11,} | epsilon: {agent.epsilon:.5f} | "
                  f"avg reward (last {window:,}): {avg:+.4f}")

    save_path = os.path.join(root_dir, "qtable_v6_final.npy")
    agent.save(save_path)
    print(f"\nSaved Q-table to: {save_path}")
    print(f"Updates applied: {applied:,}  |  Skipped (TC<=0): {skipped:,}")


if __name__ == "__main__":
    main()
