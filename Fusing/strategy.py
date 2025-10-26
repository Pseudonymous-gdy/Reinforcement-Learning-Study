import os
import sys
# Allow running this file directly: ensure repo root is on sys.path
if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
try:
    from Fusing.env import Environment
except ModuleNotFoundError:  # fallback if executed from within Fusing directory
    from env import Environment


class ProbabilisticStrategy:
    """A simple fusion of reward-UCB and dueling-UCB decisions.

    With probability alpha each step, select an arm using UCB1 on reward feedback;
    with probability 1-alpha, select a pair using a dueling-UCB heuristic, then
    pull the winner arm to obtain reward. Both sources update their own statistics.

    Notes
    - Reward setting uses classic UCB1: value[i] + sqrt(2 ln t / n_i).
    - Dueling setting maintains pairwise win rates v[i,j] for i vs j and a symmetric
      comparison count n[i,j] == n[j,i]. Pair selection favors uncertain pairs: if a
      pair hasn't been compared, it is prioritized; otherwise uncertainty is measured
      roughly by closeness to 0.5 plus an exploration bonus.
    - Step always results in pulling a single arm and receiving a reward.
    """

    def __init__(self, env: Environment, alpha: float = 0.5, seed: Optional[int] = None):
        assert 0.0 <= alpha <= 1.0, "alpha must be in [0, 1]"
        self.env = env
        self.alpha = alpha  # Probability to choose reward-UCB branch

        # Internal RNG for strategy decisions (separate from env RNG)
        self.rng = np.random.default_rng(seed)

        # strategy 1: reward setting (UCB Adopted)
        self.counts: List[int] = [0] * env.number_of_bandits
        self.values: List[float] = [0.0] * env.number_of_bandits

        # strategy 2: dueling setting (UCB-based heuristic)
        n = env.number_of_bandits
        self.dueling_counts: np.ndarray = np.zeros((n, n), dtype=int)
        self.dueling_values: np.ndarray = np.zeros((n, n), dtype=float)  # v[i,j] ~ P(i beats j)

        # Book-keeping
        self.t_reward: int = 0  # total reward pulls
        self.t_duel: int = 0    # total dueling comparisons
        self.cumulative_reward: float = 0.0
        self.history: List[Dict[str, Any]] = []

    # ------------------------------ Reward-UCB ------------------------------ #
    def _ucb_scores(self, t: int) -> List[float]:
        scores: List[float] = []
        for i in range(self.env.number_of_bandits):
            if self.counts[i] == 0:
                scores.append(float('inf'))  # force exploration
            else:
                bonus = np.sqrt(2.0 * np.log(max(1, t)) / self.counts[i])
                scores.append(self.values[i] + bonus)
        return scores

    def select_arm_ucb(self) -> int:
        t = max(1, self.t_reward)
        scores = self._ucb_scores(t)
        return int(np.argmax(scores))

    def update_reward(self, arm: int, reward: float) -> None:
        self.t_reward += 1
        self.counts[arm] += 1
        n = self.counts[arm]
        # incremental mean update
        self.values[arm] += (reward - self.values[arm]) / n
        self.cumulative_reward += float(reward)

    # ------------------------------ Dueling-UCB ----------------------------- #
    def _pair_uncertainty(self, i: int, j: int) -> float:
        """Higher means more uncertain -> more exploration.

        If not compared yet, return inf to prioritize.
        Else return closeness to 0.5 plus exploration bonus.
        """
        n_ij = int(self.dueling_counts[i, j])
        if n_ij == 0:
            return float('inf')
        v_ij = float(self.dueling_values[i, j])
        t = max(1, self.t_duel)
        bonus = np.sqrt(2.0 * np.log(t) / n_ij)
        return (0.5 - abs(v_ij - 0.5)) + bonus

    def select_pair_dueling(self) -> Tuple[int, int]:
        n = self.env.number_of_bandits
        # First, pick any unseen pair
        for i in range(n):
            for j in range(i + 1, n):
                if self.dueling_counts[i, j] == 0:
                    return i, j
        # Otherwise pick the most uncertain pair
        best_pair = (0, 1)
        best_uncert = -1.0
        for i in range(n):
            for j in range(i + 1, n):
                u = self._pair_uncertainty(i, j)
                if u > best_uncert:
                    best_uncert = u
                    best_pair = (i, j)
        return best_pair

    def update_dueling(self, a: int, b: int, result: int) -> int:
        """Update pairwise stats.

        result: 1 means a beats b; 0 means b beats a.
        Returns the winner index.
        """
        self.t_duel += 1
        # symmetric total count
        n = int(self.dueling_counts[a, b]) + 1
        self.dueling_counts[a, b] = n
        self.dueling_counts[b, a] = n

        # update wins for a over b
        wins_ab_prev = float(self.dueling_values[a, b]) * (n - 1)
        wins_ab = wins_ab_prev + (1 if result == 1 else 0)
        v_ab = wins_ab / n
        self.dueling_values[a, b] = v_ab
        self.dueling_values[b, a] = 1.0 - v_ab

        return a if result == 1 else b

    # ------------------------------ Running logic --------------------------- #
    def step(self) -> Dict[str, Any]:
        """Run one decision step and return a record with details.

        Mixture policy:
        - With prob alpha: choose arm via reward-UCB.
        - Else: choose pair via dueling-UCB, compare to get winner arm.
        In both cases, we finally pull a single arm to obtain reward and update stats.
        """
        use_reward_ucb = bool(self.rng.random() < self.alpha)
        chosen_arm: int
        duel_pair: Optional[Tuple[int, int]] = None
        duel_result: Optional[int] = None

        if use_reward_ucb:
            chosen_arm = self.select_arm_ucb()
        else:
            a, b = self.select_pair_dueling()
            duel_pair = (a, b)
            duel_result = self.env.get_dueling(a, b)  # 1 if a > b else 0
            winner = self.update_dueling(a, b, duel_result)
            chosen_arm = winner

        reward = float(self.env.get_reward(chosen_arm))
        self.update_reward(chosen_arm, reward)

        record = {
            "use_reward_ucb": use_reward_ucb,
            "chosen_arm": chosen_arm,
            "reward": reward,
            "cumulative_reward": self.cumulative_reward,
            "t_reward": self.t_reward,
            "t_duel": self.t_duel,
        }
        if duel_pair is not None:
            record.update({
                "duel_pair": duel_pair,
                "duel_result": duel_result,  # 1 means first beats second
                "duel_v": float(self.dueling_values[duel_pair[0], duel_pair[1]]),
                "duel_n": int(self.dueling_counts[duel_pair[0], duel_pair[1]]),
            })
        self.history.append(record)
        return record

    def run(self, steps: int) -> List[Dict[str, Any]]:
        for _ in range(steps):
            self.step()
        return self.history

    # ------------------------------ Utilities ------------------------------- #
    def get_best_arm_by_reward(self) -> int:
        # If some arms never pulled, pick the one with highest UCB score now
        if any(c == 0 for c in self.counts):
            return int(np.argmax(self._ucb_scores(max(1, self.t_reward))))
        return int(np.argmax(self.values))

    def get_best_arm_by_dueling(self) -> int:
        # Copeland-like: count how many others an arm is estimated to beat (>0.5)
        n = self.env.number_of_bandits
        scores = np.zeros(n, dtype=float)
        for i in range(n):
            scores[i] = float(np.sum(self.dueling_values[i, :] > 0.5))
        return int(np.argmax(scores))

    def get_history(self) -> List[Dict[str, Any]]:
        return self.history

    def reset(self, seed: Optional[int] = None) -> None:
        # Reset strategy state; environment reset should be done externally as needed
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        n = self.env.number_of_bandits
        self.counts = [0] * n
        self.values = [0.0] * n
        self.dueling_counts = np.zeros((n, n), dtype=int)
        self.dueling_values = np.zeros((n, n), dtype=float)
        self.t_reward = 0
        self.t_duel = 0
        self.cumulative_reward = 0.0
        self.history = []


if __name__ == "__main__":
    # Tiny smoke run
    env = Environment(6, distribution='bernoulli', seed=123)
    strat = ProbabilisticStrategy(env, alpha=0.6, seed=999)
    logs = strat.run(steps=20)
    print("best arm by reward:", strat.get_best_arm_by_reward())
    print("best arm by dueling:", strat.get_best_arm_by_dueling())
    print("last record:", logs[-1])