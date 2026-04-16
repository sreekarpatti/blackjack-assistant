import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from blackjack_rl.env.blackjack_env import BlackjackEnv, BET_RAMP
from blackjack_rl.agent.q_agent import QLearningAgent
from blackjack_rl.strategy import basic_strategy

NUM_EPISODES = 100_000
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fresh_bucket_stats():
    return {b: {"wins": 0, "losses": 0, "pushes": 0, "total": 0,
                "reward": 0.0, "units": 0.0} for b in range(-4, 5)}


def run_agent(env, agent, n_episodes):
    wins = losses = pushes = 0
    total_reward = 0.0
    units_wagered = 0.0
    by_count = fresh_bucket_stats()

    for _ in range(n_episodes):
        state, info = env.reset()
        bucket = info.get("true_count_bucket", 0)
        bet = env._base_bet
        hand_result = 0.0
        done = False

        while not done:
            mask = env.action_masks()
            action = agent.select_greedy(state, mask)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            hand_result += reward

        total_reward += hand_result
        units_wagered += bet
        by_count[bucket]["reward"] += hand_result
        by_count[bucket]["units"] += bet

        if hand_result > 0:
            wins += 1
            by_count[bucket]["wins"] += 1
        elif hand_result < 0:
            losses += 1
            by_count[bucket]["losses"] += 1
        else:
            pushes += 1
            by_count[bucket]["pushes"] += 1
        by_count[bucket]["total"] += 1

    return wins, losses, pushes, total_reward, units_wagered, by_count


def run_basic_strategy(env, n_episodes):
    wins = losses = pushes = 0
    total_reward = 0.0
    units_wagered = 0.0
    by_count = fresh_bucket_stats()

    for _ in range(n_episodes):
        state, info = env.reset()
        bucket = info.get("true_count_bucket", 0)
        bet = env._base_bet
        hand_result = 0.0
        done = False

        while not done:
            mask = env.action_masks()
            player_sum = float(state[0])
            dealer_up = float(state[1])
            usable_ace = bool(state[2])
            can_double = bool(state[4]) and mask[2]
            can_split = bool(state[5]) and mask[3]

            action = basic_strategy(player_sum, dealer_up, usable_ace, can_double, can_split)
            if not mask[action]:
                action = 1

            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            hand_result += reward

        total_reward += hand_result
        units_wagered += bet
        by_count[bucket]["reward"] += hand_result
        by_count[bucket]["units"] += bet

        if hand_result > 0:
            wins += 1
            by_count[bucket]["wins"] += 1
        elif hand_result < 0:
            losses += 1
            by_count[bucket]["losses"] += 1
        else:
            pushes += 1
            by_count[bucket]["pushes"] += 1
        by_count[bucket]["total"] += 1

    return wins, losses, pushes, total_reward, units_wagered, by_count


def print_results(agent_flat, bs_flat, agent_bet, bs_bet, n):
    af_w, af_l, af_p, af_r, af_u, af_b = agent_flat
    bf_w, bf_l, bf_p, bf_r, bf_u, bf_b = bs_flat
    ab_w, ab_l, ab_p, ab_r, ab_u, ab_b = agent_bet
    bb_w, bb_l, bb_p, bb_r, bb_u, bb_b = bs_bet

    W = 72
    print("\n" + "=" * W)
    print(f"{'FLAT BETTING  —  {:,} episodes'.format(n):^{W}}")
    print("=" * W)
    print(f"{'':22s} {'Agent':>12s} {'Basic Strategy':>16s}")
    print(f"{'Win Rate:':22s} {af_w/n*100:>11.1f}% {bf_w/n*100:>15.1f}%")
    print(f"{'Loss Rate:':22s} {af_l/n*100:>11.1f}% {bf_l/n*100:>15.1f}%")
    print(f"{'Push Rate:':22s} {af_p/n*100:>11.1f}% {bf_p/n*100:>15.1f}%")
    print(f"{'Avg Reward/hand:':22s} {af_r/n:>+12.4f} {bf_r/n:>+15.4f}")
    print(f"{'EV/unit wagered:':22s} {af_r/af_u*100:>+11.2f}% {bf_r/bf_u*100:>+14.2f}%")

    print("\n" + "=" * W)
    print(f"{'BET SIZING  (1-8 spread)  —  {:,} episodes'.format(n):^{W}}")
    print("=" * W)
    print(f"{'':22s} {'Agent':>12s} {'Basic Strategy':>16s}")
    print(f"{'Win Rate:':22s} {ab_w/n*100:>11.1f}% {bb_w/n*100:>15.1f}%")
    print(f"{'Loss Rate:':22s} {ab_l/n*100:>11.1f}% {bb_l/n*100:>15.1f}%")
    print(f"{'Push Rate:':22s} {ab_p/n*100:>11.1f}% {bb_p/n*100:>15.1f}%")
    print(f"{'Avg Reward/hand:':22s} {ab_r/n:>+12.4f} {bb_r/n:>+15.4f}")
    print(f"{'Total units wagered:':22s} {ab_u:>12,.0f} {bb_u:>15,.0f}")
    print(f"{'EV/unit wagered:':22s} {ab_r/ab_u*100:>+11.2f}% {bb_r/bb_u*100:>+14.2f}%")

    print("\n" + "=" * W)
    print(f"{'BY TRUE COUNT BUCKET  —  BET SIZING':^{W}}")
    print("=" * W)
    print("  Bet ramp: TC<=0->1u  TC+1->2u  TC+2->4u  TC+3->6u  TC+4->8u")
    print("-" * W)
    print(f"{'TC':>5s} | {'Bet':>4s} | {'Agent WR':>9s} | {'BS WR':>7s} | "
          f"{'Agent EV/u':>11s} | {'BS EV/u':>9s} | {'N':>7s}")
    print("-" * W)

    for bucket in sorted(ab_b.keys()):
        a = ab_b[bucket]
        b = bb_b[bucket]
        if a["total"] == 0:
            continue
        agent_wr = a["wins"] / a["total"] * 100
        bs_wr = b["wins"] / b["total"] * 100
        agent_ev = a["reward"] / a["units"] * 100 if a["units"] > 0 else 0
        bs_ev = b["reward"] / b["units"] * 100 if b["units"] > 0 else 0
        bet = BET_RAMP[bucket]
        sign = "+" if bucket > 0 else " "
        print(f"  {sign}{bucket:>2d}   | {bet:>4d} | {agent_wr:>8.1f}% | {bs_wr:>6.1f}% | "
              f"{agent_ev:>+10.2f}% | {bs_ev:>+8.2f}% | {a['total']:>7,}")

    print("=" * W)
    print("  NOTE: Buckets sampled uniformly — in a real casino, TC>=+2 occurs ~15% of hands.")


def main():
    for candidate in ["qtable_v7_mc.npy", "qtable_v6_final.npy"]:
        qtable_path = os.path.join(root_dir, candidate)
        if os.path.exists(qtable_path):
            break

    if not os.path.exists(qtable_path):
        print("No qtable found. Run a training script first.")
        sys.exit(1)

    print(f"Using Q-table: {os.path.basename(qtable_path)}")

    agent = QLearningAgent()
    agent.load(qtable_path)
    agent.epsilon = 0.0

    env_af = BlackjackEnv(use_bet_sizing=False)
    env_bf = BlackjackEnv(use_bet_sizing=False)
    env_ab = BlackjackEnv(use_bet_sizing=True)
    env_bb = BlackjackEnv(use_bet_sizing=True)

    print(f"Evaluating agent (flat) over {NUM_EPISODES:,} episodes...")
    agent_flat = run_agent(env_af, agent, NUM_EPISODES)

    print(f"Evaluating basic strategy (flat) over {NUM_EPISODES:,} episodes...")
    bs_flat = run_basic_strategy(env_bf, NUM_EPISODES)

    print(f"Evaluating agent (bet sizing) over {NUM_EPISODES:,} episodes...")
    agent_bet = run_agent(env_ab, agent, NUM_EPISODES)

    print(f"Evaluating basic strategy (bet sizing) over {NUM_EPISODES:,} episodes...")
    bs_bet = run_basic_strategy(env_bb, NUM_EPISODES)

    print_results(agent_flat, bs_flat, agent_bet, bs_bet, NUM_EPISODES)


if __name__ == "__main__":
    main()
