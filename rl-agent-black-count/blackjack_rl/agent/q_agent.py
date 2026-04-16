import numpy as np


class QLearningAgent:

    MIN_SUM = 4
    MAX_SUM = 21
    MIN_TC = -4
    MAX_TC = 4

    def __init__(self, alpha=0.1, gamma=0.99, epsilon_start=1.0,
                 epsilon_end=0.05, epsilon_decay=0.9999995, seed=42):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self._rng = np.random.default_rng(seed)
        self.Q = np.zeros((18, 10, 2, 9, 2, 2, 4), dtype=np.float64)

    def _encode(self, state):
        player_sum = int(np.clip(state[0], 4, 21))
        dealer_up = int(np.clip(state[1], 1, 10))
        usable_ace = int(state[2])
        count = int(np.clip(round(state[3]), -4, 4))
        can_double = int(state[4])
        can_split = int(state[5])
        return (
            player_sum - self.MIN_SUM,
            dealer_up - 1,
            usable_ace,
            count - self.MIN_TC,
            can_double,
            can_split,
        )

    def select_action(self, state, action_mask, phase=2):
        mask = action_mask.copy()
        if phase == 1:
            mask[2] = False
            mask[3] = False

        legal = np.where(mask)[0]
        if len(legal) == 0:
            legal = np.array([0, 1])

        if self._rng.random() < self.epsilon:
            return int(self._rng.choice(legal))

        s = self._encode(state)
        q = self.Q[s][:]
        q_masked = np.full(4, -np.inf)
        q_masked[legal] = q[legal]
        return int(np.argmax(q_masked))

    def select_greedy(self, state, action_mask):
        legal = np.where(action_mask)[0]
        if len(legal) == 0:
            legal = np.array([0, 1])
        s = self._encode(state)
        q = self.Q[s][:]
        q_masked = np.full(4, -np.inf)
        q_masked[legal] = q[legal]
        return int(np.argmax(q_masked))

    def update(self, state, action, reward, next_state, done, next_mask=None):
        s = self._encode(state)
        if done:
            target = reward
        else:
            if next_mask is None:
                next_mask = np.ones(4, dtype=bool)
            valid_next = np.where(next_mask)[0]
            if len(valid_next) == 0:
                valid_next = np.array([0, 1])
            s_next = self._encode(next_state)
            target = reward + self.gamma * np.max(self.Q[s_next][valid_next])
        self.Q[s][action] += self.alpha * (target - self.Q[s][action])

    # MC update — uses actual episode return instead of bootstrapped target
    def mc_update(self, state, action, G):
        s = self._encode(state)
        self.Q[s][action] += self.alpha * (G - self.Q[s][action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path):
        np.save(path, self.Q)

    def load(self, path):
        self.Q = np.load(path)
