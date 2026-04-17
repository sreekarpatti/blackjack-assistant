import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blackjack_rl.env.blackjack_env import hand_total, hi_lo_value, BET_RAMP
from blackjack_rl.agent.q_agent import QLearningAgent
from blackjack_rl.strategy import basic_strategy

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUM_DECKS = 6
UNIT = 10
STARTING_BANKROLL = 1_000
HANDS_PER_SESSION = 500
NUM_SESSIONS = 1_000
RESHUFFLE_AT = 78


class LiveShoe:
    def __init__(self, num_decks=NUM_DECKS, rng=None):
        self.num_decks = num_decks
        self.rng = rng if rng is not None else np.random.default_rng()
        self._cards = []
        self._running_count = 0
        self._build()

    def _build(self):
        deck = []
        for v in range(2, 10):
            deck.extend([v] * 4 * self.num_decks)
        deck.extend([10] * 16 * self.num_decks)
        deck.extend([11] * 4 * self.num_decks)
        arr = np.array(deck, dtype=np.int32)
        self.rng.shuffle(arr)
        self._cards = arr.tolist()
        self._running_count = 0

    def deal(self):
        if len(self._cards) < RESHUFFLE_AT:
            self._build()
        card = self._cards.pop()
        self._running_count += hi_lo_value(card)
        return card

    @property
    def true_count(self):
        decks_left = len(self._cards) / 52.0
        return self._running_count / max(decks_left, 0.5)

    @property
    def count_bucket(self):
        return int(np.clip(round(self.true_count), -4, 4))

    def current_bet(self):
        return float(UNIT * BET_RAMP[self.count_bucket])


def play_hand(shoe, strategy_fn, bankroll):
    bet = min(shoe.current_bet(), bankroll)
    if bet <= 0:
        return 0.0

    p1 = shoe.deal()
    d1 = shoe.deal()
    p2 = shoe.deal()
    d_hole = shoe.deal()

    upcard = 1 if d1 == 11 else d1

    dealer_bj = hand_total([d1, d_hole])[0] == 21 and len([d1, d_hole]) == 2
    if d1 in (10, 11) and dealer_bj:
        player_total, _ = hand_total([p1, p2])
        return 0.0 if player_total == 21 else -bet

    player_total, _ = hand_total([p1, p2])
    if player_total == 21:
        return 1.5 * bet

    hands = [{"cards": [p1, p2], "bet": bet, "done": False,
               "is_split_ace": False, "result": None}]
    splits_done = 0
    i = 0

    while i < len(hands):
        h = hands[i]
        if h["done"]:
            i += 1
            continue

        while not h["done"]:
            total, usable_ace = hand_total(h["cards"])
            can_double = len(h["cards"]) == 2 and not h["is_split_ace"]
            can_split = (len(h["cards"]) == 2
                         and h["cards"][0] == h["cards"][1]
                         and splits_done < 3
                         and not h["is_split_ace"])

            state = np.array([
                float(total),
                float(upcard),
                float(usable_ace),
                float(np.clip(shoe.true_count, -20, 20)),
                float(can_double),
                float(can_split),
                float(splits_done),
                float(np.clip(len(shoe._cards) / 52.0, 0, 6)),
                float(h["is_split_ace"]),
            ], dtype=np.float32)
            mask = np.array([True, True, can_double, can_split], dtype=bool)

            action = strategy_fn(state, mask)
            if not mask[action]:
                action = 1

            if action == 0:
                h["cards"].append(shoe.deal())
                t, _ = hand_total(h["cards"])
                if t > 21:
                    h["done"] = True
                    h["result"] = -h["bet"]

            elif action == 1:
                h["done"] = True

            elif action == 2:
                h["bet"] = 2.0 * bet
                h["cards"].append(shoe.deal())
                h["done"] = True
                t, _ = hand_total(h["cards"])
                if t > 21:
                    h["result"] = -h["bet"]

            elif action == 3:
                c1, c2 = h["cards"]
                splits_done += 1
                is_aces = c1 == 11
                h1 = {"cards": [c1, shoe.deal()], "bet": bet,
                       "done": is_aces, "is_split_ace": is_aces, "result": None}
                h2 = {"cards": [c2, shoe.deal()], "bet": bet,
                       "done": is_aces, "is_split_ace": is_aces, "result": None}
                hands[i:i+1] = [h1, h2]
                h = hands[i]

        i += 1

    dealer_cards = [d1, d_hole]
    while True:
        dt, dua = hand_total(dealer_cards)
        if dt > 17 or (dt == 17 and not dua):
            break
        dealer_cards.append(shoe.deal())
    dealer_total, _ = hand_total(dealer_cards)

    profit = 0.0
    for h in hands:
        if h["result"] is not None:
            profit += h["result"]
            continue
        pt, _ = hand_total(h["cards"])
        b = h["bet"]
        if pt > 21:
            profit -= b
        elif dealer_total > 21 or pt > dealer_total:
            profit += b
        elif pt < dealer_total:
            profit -= b
    return profit


def run_session(strategy_fn, seed, n_hands=HANDS_PER_SESSION,
                starting_bankroll=STARTING_BANKROLL):
    rng = np.random.default_rng(seed)
    shoe = LiveShoe(rng=rng)
    bankroll = float(starting_bankroll)
    peak = float(starting_bankroll)
    ruined = False
    hands_played = 0

    for _ in range(n_hands):
        if bankroll <= 0:
            ruined = True
            break
        bankroll += play_hand(shoe, strategy_fn, bankroll)
        if bankroll > peak:
            peak = bankroll
        hands_played += 1

    return {"final": bankroll, "peak": peak, "ruined": ruined, "hands_played": hands_played}


def pct(x, total):
    return x / total * 100 if total > 0 else 0.0


def print_report(all_results, labels, n_sims, n_hands, start_brl, unit):
    finals = [np.array([r["final"] for r in res]) for res in all_results]
    ruin_counts = [sum(r["ruined"] for r in res) for res in all_results]

    W = 76
    sep = "=" * W
    col = 13

    print(f"\n{sep}")
    print(f"  BANKROLL SIMULATION  —  {n_sims:,} sessions × {n_hands:,} hands")
    print(f"  Start: ${start_brl:,}  |  Unit: ${unit}  |  Spread: $10 -> $80  (1-8)")
    print(f"  Bet ramp: TC<=0->$10  TC+1->$20  TC+2->$40  TC+3->$60  TC+4->$80")
    print(sep)

    header = f"  {'':30s}"
    for lbl in labels:
        header += f"  {lbl:>{col}s}"
    print(header)

    def row(label, values):
        line = f"  {label:30s}"
        for v in values:
            line += f"  {v:>{col}}"
        print(line)

    row("Avg final bankroll:", [f"${f.mean():>10,.0f}" for f in finals])
    row("Median final bankroll:", [f"${np.median(f):>10,.0f}" for f in finals])
    row("Std dev:", [f"${f.std():>10,.0f}" for f in finals])
    row("Best session:", [f"${f.max():>10,.0f}" for f in finals])
    row("Worst session:", [f"${f.min():>10,.0f}" for f in finals])
    row("Ruin rate:", [f"{pct(r, n_sims):>10.1f}%" for r in ruin_counts])

    profit_pcts = [(f.mean() - start_brl) / start_brl * 100 for f in finals]
    row("Avg profit/loss:", [f"{p:>+10.1f}%" for p in profit_pcts])

    avg_units = n_hands * unit * (5*1 + 1*2 + 1*4 + 1*6 + 1*8) / 9
    evs = [(f.mean() - start_brl) / avg_units * 100 for f in finals]
    row("Est. EV/unit wagered:", [f"{e:>+10.2f}%" for e in evs])

    print(f"\n  HEAD-TO-HEAD  vs {labels[0]}  (same starting shoe per seed)")
    base = finals[0]
    for i in range(1, len(labels)):
        wins = int(np.sum(base > finals[i]))
        losses = int(np.sum(finals[i] > base))
        ties = n_sims - wins - losses
        print(f"  {labels[0]} vs {labels[i]}:")
        print(f"    {labels[0]} wins: {wins:>5,} ({pct(wins, n_sims):.1f}%)  "
              f"{labels[i]} wins: {losses:>5,} ({pct(losses, n_sims):.1f}%)  "
              f"Ties: {ties:>4,} ({pct(ties, n_sims):.1f}%)")

    print(f"\n  BANKROLL DISTRIBUTION")
    dist_header = f"  {'Range':>18s}"
    for lbl in labels:
        dist_header += f"  {lbl:>{col}s}"
    print(dist_header)

    ranges = [
        ("Ruined ($0)", lambda x: x <= 0),
        ("$1-$499", lambda x: 0 < x < 500),
        ("$500-$999", lambda x: 500 <= x < 1000),
        ("$1,000-$1,499", lambda x: 1000 <= x < 1500),
        ("$1,500-$1,999", lambda x: 1500 <= x < 2000),
        ("$2,000+", lambda x: x >= 2000),
    ]
    for label, fn in ranges:
        vals = [f"{pct(int(np.sum([fn(x) for x in f])), n_sims):>10.1f}%" for f in finals]
        line = f"  {label:>18s}"
        for v in vals:
            line += f"  {v:>{col}s}"
        print(line)
    print(sep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qtable", type=str, default=None)
    parser.add_argument("--sims", type=int, default=NUM_SESSIONS)
    parser.add_argument("--hands", type=int, default=HANDS_PER_SESSION)
    args = parser.parse_args()

    if args.qtable:
        qtable_path = args.qtable
    else:
        for candidate in ["qtable_v7_mc.npy", "qtable_v6_final.npy"]:
            qtable_path = os.path.join(root_dir, candidate)
            if os.path.exists(qtable_path):
                break

    if not os.path.exists(qtable_path):
        print("No qtable found. Run training first.")
        sys.exit(1)

    agent = QLearningAgent()
    agent.load(qtable_path)
    agent.epsilon = 0.0
    print(f"Loaded Q-table: {os.path.basename(qtable_path)}")

    def bs_fn(state, mask):
        player_sum = float(state[0])
        dealer_up = float(state[1])
        usable_ace = bool(state[2])
        can_double = bool(state[4]) and mask[2]
        can_split = bool(state[5]) and mask[3]
        action = basic_strategy(player_sum, dealer_up, usable_ace, can_double, can_split)
        if not mask[action]:
            action = 1
        return action

    def agent_fn(state, mask):
        return agent.select_greedy(state, mask)

    # BS for TC<=+1, full agent for TC>=+2
    def hybrid_fn(state, mask):
        bucket = int(np.clip(round(float(state[3])), -4, 4))
        if bucket <= 1:
            return bs_fn(state, mask)
        return agent.select_greedy(state, mask)

    # BS handles doubles/splits; agent decides hit/stand at TC>=+2
    def bs_anchored_fn(state, mask):
        player_sum = float(state[0])
        dealer_up = float(state[1])
        usable_ace = bool(state[2])
        can_double = bool(state[4]) and mask[2]
        can_split = bool(state[5]) and mask[3]
        bucket = int(np.clip(round(float(state[3])), -4, 4))

        if bucket <= 1:
            action = basic_strategy(player_sum, dealer_up, usable_ace, can_double, can_split)
            if not mask[action]:
                action = 1
            return action

        bs_action = basic_strategy(player_sum, dealer_up, usable_ace, can_double, can_split)
        if not mask[bs_action]:
            bs_action = 1
        if bs_action in (2, 3):
            return bs_action
        return agent.select_greedy(state, mask)

    # agent only overrides hit/stand on hard 12-16 at TC>=+2
    def index_anchored_fn(state, mask):
        player_sum = float(state[0])
        dealer_up = float(state[1])
        usable_ace = bool(state[2])
        can_double = bool(state[4]) and mask[2]
        can_split = bool(state[5]) and mask[3]
        bucket = int(np.clip(round(float(state[3])), -4, 4))

        bs_action = basic_strategy(player_sum, dealer_up, usable_ace, can_double, can_split)
        if not mask[bs_action]:
            bs_action = 1

        if (bucket >= 2 and not usable_ace
                and 12 <= int(player_sum) <= 16
                and bs_action in (0, 1)):
            return agent.select_greedy(state, mask)
        return bs_action

    print(f"Running {args.sims:,} paired sessions x {args.hands:,} hands each ...")
    print(f"  Starting bankroll: ${STARTING_BANKROLL:,}  |  Unit bet: ${UNIT}")
    print(f"  Strategies: BS-Anchored MC (RL H/S @ TC>=+2, BS for D/P), Basic Strategy")

    anchored_results = []
    bs_results = []

    for i in range(args.sims):
        if (i + 1) % 100 == 0:
            print(f"  Simulation {i+1:>5,} / {args.sims:,} ...", end="\r")
        anchored_results.append(run_session(bs_anchored_fn, seed=i,
                                             n_hands=args.hands,
                                             starting_bankroll=STARTING_BANKROLL))
        bs_results.append(run_session(bs_fn, seed=i,
                                       n_hands=args.hands,
                                       starting_bankroll=STARTING_BANKROLL))

    print(f"  Done.{' ' * 40}")
    print_report(
        [anchored_results, bs_results],
        ["BS-Anchored MC", "Basic Strat"],
        args.sims, args.hands, STARTING_BANKROLL, UNIT,
    )


if __name__ == "__main__":
    main()
